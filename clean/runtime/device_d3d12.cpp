// SPDX-License-Identifier: MIT
// Original standalone D3D12 device for Model M, 2026-10-15.
// Basis: this project's Device interface, HLSL arena ABI and hybrid root signature; public Microsoft D3D12 APIs.
// No HIP object, imported memory, external semaphore or order_mark is used. Module paths select arithmetic variants
// by identity only: the native device neither reads nor loads their HIP code objects.
#include "device_d3d12.h"

#if defined(_WIN32)
#include "interop/d3d12_include.h"
#include "interop/descriptor_publish.h"
#include "interop/texture_shape.h"
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include "../converter/src/json.h"
#include "../converter/src/sha256.h"
#include "gelu_lut_table.h"
#include "gelu_t5_table.h"
#include "native_direct_pi.h"
#include "embedded.h"
#include "rdna4.h"
#include "wmma_probe.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace wgrt {
namespace {

template<class T> struct NativeCom {
  T* p = nullptr;
  NativeCom() = default;
  NativeCom(const NativeCom&) = delete;
  NativeCom& operator=(const NativeCom&) = delete;
  NativeCom(NativeCom&& x) noexcept : p(x.p) { x.p = nullptr; }
  NativeCom& operator=(NativeCom&& x) noexcept {
    if (this != &x) { reset(); p = x.p; x.p = nullptr; }
    return *this;
  }
  ~NativeCom() { reset(); }
  void reset() { if (p) p->Release(); p = nullptr; }
  T** put() { reset(); return &p; }
  void** pv() { return reinterpret_cast<void**>(put()); }
  T* get() const { return p; }
  T* operator->() const { return p; }
  explicit operator bool() const { return p != nullptr; }
};

constexpr unsigned kNativeArgs = 57, kNativeConstants = 2, kNativeGroups = 65535;
constexpr uint64_t kNativeAlign = 256;
constexpr unsigned kNativeSlots = 8, kMaxEvents = 65536;
constexpr uint64_t kTransientSubmitBytes = 64ull << 20;
constexpr unsigned kProfileQueries = 4096;

struct NativeArg { char kind; unsigned bytes; };
struct NativeKernel {
  std::string name, file, co, rs = "buffer";
  unsigned block = 0, wave = 0, dwords = 0;
  // group: lanes per D3D12 workgroup when the port splits each launched workgroup of `block` lanes into block / group
  // workgroups (manifest "group", build_dx12.py group=; 0 = block). The launch keeps the HIP geometry (grid x block).
  unsigned group = 0, gscale = 1;  // gscale: D3D12 workgroups per launched workgroup (manifest "gscale", default block / group)
  std::vector<NativeArg> args;
  wgc::Digest digest{};
  NativeCom<ID3D12PipelineState> pso;
};
struct NativeAllocation { uint64_t reserved, requested; };
struct NativeEvent { uint64_t fence = 0; bool recorded = false, pending = false, external = false; };

std::string native_basename(const std::string& s) { return s.substr(s.find_last_of("/\\") + 1); }
bool native_identifier(const std::string& s, bool slash) {
  if (s.empty() || s.front() == '/' || s.back() == '/') return false;
  unsigned slashes = 0;
  for (unsigned char c : s) {
    if (c == '/' && slash && ++slashes == 1) continue;
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) return false;
  }
  return true;
}
bool native_filename(const std::string& s) {
  if (s.empty() || s == "." || s == "..") return false;
  for (unsigned char c : s)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
          c == '_' || c == '-' || c == '.')) return false;
  return s.find("..") == std::string::npos;
}

// clean/fe-minimize: native D3D12 contexts being torn down, per game device (process-wide). A host that recreates its
// upscaler after an error creates the new context before it destroys the old one; the new context's arena then waits
// for the old one's teardown on the same device (its GPU memory released) instead of running out of memory next to it.
struct TeardownBook {
  std::mutex mu;
  std::condition_variable cv;
  std::map<const void*, int> busy;
};
TeardownBook& teardown_book() {
  static TeardownBook* book = new TeardownBook;  // never destroyed: teardowns may run during process exit
  return *book;
}
void teardown_begin(const void* device) {
  TeardownBook& b = teardown_book();
  std::lock_guard<std::mutex> lock(b.mu);
  ++b.busy[device];
}
void teardown_end(const void* device) {
  TeardownBook& b = teardown_book();
  {
    std::lock_guard<std::mutex> lock(b.mu);
    auto it = b.busy.find(device);
    if (it != b.busy.end() && --it->second <= 0) b.busy.erase(it);
  }
  b.cv.notify_all();
}
// Waits until no teardown runs on `device` (at most `ms`). The time waited in ms, or -1 when none was running.
int64_t teardown_wait(const void* device, int ms) {
  TeardownBook& b = teardown_book();
  std::unique_lock<std::mutex> lock(b.mu);
  if (!b.busy.count(device)) return -1;
  const auto t0 = std::chrono::steady_clock::now();
  b.cv.wait_for(lock, std::chrono::milliseconds(ms), [&] { return !b.busy.count(device); });
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}

class NativeD3D12Device final : public Device, public NativeD3D12Ext {
 public:
  ~NativeD3D12Device() override {
    // clean/fe-minimize: a teardown started by begin_teardown ends here, after the GPU memory is released (every path)
    struct TeardownEnd {
      const void* device;
      bool on;
      ~TeardownEnd() { if (on) teardown_end(device); }
    } teardown_end_guard{device_.get(), teardown_registered_};
    // Keep every allocator, staging buffer and pipeline alive until the queue finishes. A timeout in a diagnostic
    // caller must not turn teardown into release of in-flight GPU resources.
    const bool completed = finish_queue();
    const std::string completion_error = completed ? std::string() : err_;
    const bool debug_valid = drain_debug_messages();
    if (completed) profile_drain_all();
    profile_finish(completed && debug_valid);
    if (!debug_valid) debug_log(true, "native D3D12 teardown validation: " + err_);
    if (!completed) {
      err_ = completion_error;
      log(true, "native D3D12 teardown: " + err_);
      // A removed device cannot execute more work. Otherwise keep ownership until process exit if the OS cannot
      // establish completion; releasing a live allocator/resource here would turn an error into GPU use-after-free.
      if (device_ && SUCCEEDED(device_->GetDeviceRemovedReason())) {
        log(true, "native D3D12 teardown: completion unavailable; retaining GPU objects until process exit");
        debug_summary();
        retain_gpu_objects(); return;
      }
    }
    for (auto& h : hosts_) if (h.second.resource) h.second.resource->Unmap(0, nullptr);
    hosts_.clear();
    kernels_.clear();
    query_readback_.reset(); query_heap_.reset();
    list_.reset();
    for (auto& s : slots_) { s.keep.clear(); s.keep_fences.clear(); s.profile.clear(); s.allocator.reset(); }
    for (auto& set : direct_textures_) for (auto& texture : set) texture.reset();
    if (direct_constants_map_ && direct_constants_) direct_constants_->Unmap(0, nullptr);
    direct_constants_map_ = nullptr;
    if (marker_map_ && marker_) marker_->Unmap(0, nullptr);
    marker_map_ = nullptr; marker_.reset();
    direct_constants_.reset(); direct_heap_.reset(); direct_root_.reset();
    arena_.reset(); channel_.reset(); heap_.reset(); root_.reset(); gelu_t5_.reset();
    queue_.reset(); fence_.reset();
    // Also collect diagnostics produced by resource/allocator release itself before releasing the info queue.
    drain_debug_messages(); debug_summary();
    info_queue_.reset(); device_.reset();
    if (event_) CloseHandle(event_);
    if (d3d12_dll_) FreeLibrary(d3d12_dll_);
    if (dxgi_dll_) FreeLibrary(dxgi_dll_);
  }

  bool init(const std::string& build, ID3D12Device* borrowed_device = nullptr, ID3D12CommandQueue* borrowed_queue = nullptr,
            uint64_t arena_override = 0) {
    uint64_t mb = 2560, batch = 0;
    if (!environment_number("WG_D3D12_ARENA_MB", 16, 4095, &mb) ||
        !environment_number("WG_D3D12_BATCH", 0, 65535, &batch)) return false;
    arena_bytes_ = mb << 20; batch_ = static_cast<unsigned>(batch);
    if (arena_override) arena_bytes_ = (arena_override + (1u << 20) - 1) & ~uint64_t((1u << 20) - 1);
    // The hybrid selection knob cannot silently omit native stages or create a fallback route.
    const char* selection = std::getenv("WG_DX12_KERNELS");
    if (selection && std::string(selection) != "all")
      return fail("native D3D12 requires all ports; unset WG_DX12_KERNELS or set it to all");
    if (!read_manifest(build + "/dx12/manifest.json")) return false;
    if (borrowed_device || borrowed_queue) {
      if (!borrow_device(borrowed_device, borrowed_queue)) return false;
    } else if (!create_device()) return false;
    if (!create_objects()) return false;
    if (!profile_init()) return false;
    return reserve_events(64) && (!borrowed_ || driver_probe()) && gelu_lut_upload() && gelu_t5_upload();
  }

  // The MLP kernels' exact GELU + FP8 table (hlsl/wg_gelu_lut.hlsli) at its fixed arena offset, reserved in
  // create_objects; recorded before any launch on this queue.
  bool gelu_lut_upload() {
    std::vector<uint32_t> table(wg::kGeluLutBytes / 4);
    wg::gelu_lut_expand(table.data());
    if (!transient_room(wg::kGeluLutBytes)) return false;
    NativeCom<ID3D12Resource> staging;
    if (!buffer(wg::kGeluLutBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, staging)) return false;
    void* mapped = nullptr;
    D3D12_RANGE no_read{0, 0};
    if (!check(staging->Map(0, &no_read, &mapped), "Map(GELU table)")) return false;
    std::memcpy(mapped, table.data(), wg::kGeluLutBytes);
    D3D12_RANGE written{0, wg::kGeluLutBytes}; staging->Unmap(0, &written);
    if (!open_list()) return false;
    transition_arena(D3D12_RESOURCE_STATE_COPY_DEST);
    list_->CopyBufferRegion(arena_.get(), wg::kGeluLutBase, staging.get(), 0, wg::kGeluLutBytes);
    keep(std::move(staging), wg::kGeluLutBytes); return true;
  }
  // The same table indexed by the FP32 bits of w (hlsl/wg_gelu_t5.hlsli, gelu_t5_table.h): its own read-only buffer
  // (created in create_objects), filled here, before any launch on this queue; a structured SRV (t0, space 1) at
  // descriptor kGeluT5Slot of heap_ and of every direct set.
  void gelu_t5_view(D3D12_CPU_DESCRIPTOR_HANDLE h) {
    D3D12_SHADER_RESOURCE_VIEW_DESC v{}; v.Format = DXGI_FORMAT_UNKNOWN; v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Buffer.NumElements = wg::kGeluT5Entries; v.Buffer.StructureByteStride = 8;
    device_->CreateShaderResourceView(gelu_t5_.get(), &v, h);
  }
  bool gelu_t5_upload() {
    std::vector<uint32_t> table(wg::kGeluT5Bytes / 4);
    wg::gelu_t5_expand(table.data());
    if (!transient_room(wg::kGeluT5Bytes)) return false;
    NativeCom<ID3D12Resource> staging;
    if (!buffer(wg::kGeluT5Bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, staging)) return false;
    void* mapped = nullptr;
    D3D12_RANGE no_read{0, 0};
    if (!check(staging->Map(0, &no_read, &mapped), "Map(GELU T5 table)")) return false;
    std::memcpy(mapped, table.data(), wg::kGeluT5Bytes);
    D3D12_RANGE written{0, wg::kGeluT5Bytes}; staging->Unmap(0, &written);
    if (!open_list()) return false;
    list_->CopyBufferRegion(gelu_t5_.get(), 0, staging.get(), 0, wg::kGeluT5Bytes);
    transition(gelu_t5_.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    keep(std::move(staging), wg::kGeluT5Bytes); return true;
  }

  ID3D12Resource* arena_resource() const override { return arena_.get(); }
  bool arena_offset(DevPtr pointer, size_t bytes, uint64_t& offset, std::string* error) override {
    offset = 0;
    if (!bytes || !range(pointer, bytes, "arena_offset")) {
      if (!bytes) fail("native D3D12 arena_offset: zero byte count");
      return extension_result(false, error);
    }
    offset = static_cast<uint64_t>(pointer); return extension_result(true, error);
  }
  bool bind_direct_frame(const NativeDirectFrame& frame, std::string* error) override {
    if (inside_pass_) return extension_result(fail("native D3D12 bind_direct_frame: called from record_pass"), error);
    // An unsuccessful bind must never leave a previous frame's descriptors authorized for a new direct dispatch.
    direct_bound_ = false;
    if (!frame.params.rw || !frame.params.rh || frame.params.exposure_mode > 2)
      return extension_result(fail("native D3D12 direct frame: invalid extent or exposure mode"), error);
    ID3D12Resource* textures[5] = {frame.color, frame.depth, frame.motion, frame.exposure, frame.output};
    const DXGI_FORMAT formats[5] = {frame.color_format, frame.depth_format, frame.motion_format,
                                   frame.exposure_format, frame.output_format};
    std::array<NativeCom<ID3D12Resource>, 5> retained;
    std::array<NativeCom<IUnknown>, 5> identities;
    for (unsigned i = 0; i < 5; ++i) {
      if (!validate_direct_texture(textures[i], formats[i], i, frame.params)) return extension_result(false, error);
      if (textures[i]) {
        textures[i]->AddRef(); retained[i].p = textures[i];
        if (!identity(textures[i], identities[i])) return extension_result(false, error);
      }
    }
    for (unsigned i = 0; i < 4; ++i)
      if (identities[i] && identities[i].get() == identities[4].get())
        return extension_result(fail("native D3D12 direct frame: output aliases an input texture"), error);
    // Own queue: set 0 and its upload CBV are reusable only after all prior native work, including final
    // output/state restoration passes, has completed. Caller's list: the next ring set, once the frame that used it
    // (kDirectRing frames ago) has completed. Old texture references remain retained on every error path.
    unsigned set = 0;
    if (external_) {
      if (!create_direct_objects()) return extension_result(false, error);
      set = (direct_slot_ + 1) % kDirectRing;
      if (!wait_external(direct_serial_[set])) return extension_result(false, error);
    } else if (!sync() || !create_direct_objects()) {
      return extension_result(false, error);
    }
    char* constants = direct_constants_map_ + 256u * set;
    std::memset(constants, 0, 256);
    std::memcpy(constants, &frame.params, sizeof frame.params);
    // Words 11..13: the pre-exposure division window of native_direct_color_div3 (native_direct_pi.h), from color_pi.
    const NativeDirectPiWindow window = native_direct_pi_window(frame.params.color_pi);
    std::memcpy(constants + 44, &window.rcp, 4);
    std::memcpy(constants + 48, &window.lo, 4);
    std::memcpy(constants + 52, &window.hi, 4);
    const UINT stride = descriptor_stride_;
    D3D12_CPU_DESCRIPTOR_HANDLE descriptor = direct_heap_->GetCPUDescriptorHandleForHeapStart();
    descriptor.ptr += (static_cast<uint64_t>(set) * kDirectSet + 1) * stride;
    for (unsigned i = 0; i < 4; ++i) {
      const D3D12_SHADER_RESOURCE_VIEW_DESC view = wgi::mip0_srv(formats[i]);  // mip 0 of any mip count
      device_->CreateShaderResourceView(textures[i], &view, descriptor);
      descriptor.ptr += stride;
    }
    const D3D12_UNORDERED_ACCESS_VIEW_DESC output = wgi::mip0_uav(frame.output_format);  // mip 0 of any mip count
    device_->CreateUnorderedAccessView(frame.output, nullptr, &output, descriptor);
    D3D12_CPU_DESCRIPTOR_HANDLE first = direct_heap_->GetCPUDescriptorHandleForHeapStart();   // descriptor_publish.h
    first.ptr += (static_cast<uint64_t>(set) * kDirectSet + 1) * stride;
    wg::publish_descriptors(reinterpret_cast<const void*>(first.ptr), 5u * static_cast<size_t>(stride));
    // References are now associated with the descriptors even if the debug layer reports a validation error.
    direct_textures_[set] = std::move(retained);
    direct_slot_ = set; direct_serial_[set] = external_ ? ext_recorded_ + 1 : 0;
    native_binding_dirty_ = true;
    if (!drain_debug_messages()) return extension_result(false, error);
    direct_bound_ = true;
    return extension_result(true, error);
  }
  bool end_direct_frame(std::string* error) override {
    if (inside_pass_) return extension_result(fail("native D3D12 end_direct_frame: called from record_pass"), error);
    if (!direct_bound_) return extension_result(fail("native D3D12 end_direct_frame: no active binding"), error);
    direct_bound_ = false;
    // Descriptor memory, CBV and texture references remain unchanged until a successful synchronizing bind or
    // safe teardown. End is permission revocation, not a fence and not completion of the caller's output use.
    return extension_result(true, error);
  }
  bool record_pass(D3D12_RESOURCE_STATES state, const RecordPass& record_pass, std::string* error,
                   const char* profile_label = "record_pass") override {
    // Restrict to states supported by both accepted queue types and our whole-buffer ownership model.
    if (state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS && state != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE &&
        state != D3D12_RESOURCE_STATE_COPY_SOURCE && state != D3D12_RESOURCE_STATE_COPY_DEST)
      return extension_result(fail("native D3D12 record_pass: unsupported arena state"), error);
    if (!record_pass) return extension_result(fail("native D3D12 record_pass: empty callback"), error);
    if (inside_pass_) return extension_result(fail("native D3D12 record_pass: recursive recording"), error);
    if (!open_list()) return extension_result(false, error);
    transition_arena(state); uav_barrier();
    native_binding_dirty_ = true; inside_pass_ = true;
    const auto stamp = profile_begin("external_pass", profile_label ? profile_label : "record_pass");
    std::string detail;
    bool ok = false;
    try { ok = record_pass(cl(), detail); }
    catch (const std::exception& e) { detail = std::string("callback threw: ") + e.what(); }
    catch (...) { detail = "callback threw an unknown exception"; }
    inside_pass_ = false; profile_end(stamp);
    // A failed callback may have recorded valid partial work. Keep it ordered and retained like a successful pass;
    // its owner still has to sync before releasing resources. No failed callback is a completed frame.
    need_uav_barrier_ = true;
    return extension_result(ok || fail("native D3D12 record_pass: " + (detail.empty() ? "callback failed" : detail)), error);
  }
  bool queue_wait(ID3D12Fence* ready, uint64_t value, std::string* error) override {
    if (inside_pass_) return extension_result(fail("native D3D12 queue_wait: called from record_pass"), error);
    if (external_) return extension_result(fail("native D3D12 queue_wait: recording into the caller's list"), error);
    if (!ready || value == UINT64_MAX)
      return extension_result(fail("native D3D12 queue_wait: invalid fence or value"), error);
    NativeCom<ID3D12Device> owner;
    if (!check(ready->GetDevice(__uuidof(ID3D12Device), owner.pv()), "Fence::GetDevice") ||
        !same_identity(device_.get(), owner.get(), "input fence device")) return extension_result(false, error);
    NativeCom<IUnknown> ready_identity, own_identity;
    if (!identity(ready, ready_identity) || !identity(fence_.get(), own_identity)) return extension_result(false, error);
    if (ready_identity.get() == own_identity.get() && value > last_submitted_)
      return extension_result(fail("native D3D12 queue_wait: future own-fence wait would deadlock"), error);
    if (ready->GetCompletedValue() == UINT64_MAX)
      return extension_result(fail("native D3D12 queue_wait: input fence device removed"), error);
    if (!submit()) return extension_result(false, error);
    NativeCom<ID3D12Fence> retained; ready->AddRef(); retained.p = ready;
    slots_[current_slot_].keep_fences.emplace_back(std::move(retained));
    if (!check(queue_->Wait(ready, value), "CommandQueue::Wait")) return extension_result(false, error);
    // A queue wait without a command list still needs a completion signal before its fence reference can retire.
    queue_unfenced_ = true;
    return extension_result(true, error);
  }
  bool submit_completion(ID3D12Fence** result, uint64_t* value, std::string* error) override {
    if (result) *result = nullptr;
    if (value) *value = 0;
    if (inside_pass_) return extension_result(fail("native D3D12 submit_completion: called from record_pass"), error);
    if (external_) return extension_result(fail("native D3D12 submit_completion: recording into the caller's list"), error);
    if (!result || !value) return extension_result(fail("native D3D12 submit_completion: null result"), error);
    // Even an empty submission returns a fresh ordered queue signal, never an inferred completed marker.
    queue_unfenced_ = true;
    if (!submit()) return extension_result(false, error);
    fence_->AddRef(); *result = fence_.get(); *value = last_submitted_;
    return extension_result(true, error);
  }
  bool sync(std::string* error) override { return extension_result(sync(), error); }

  bool begin_external(ID3D12GraphicsCommandList* list, std::string* error) override {
    if (inside_pass_ || external_) return extension_result(fail("native D3D12 begin_external: already recording"), error);
    if (!list) return extension_result(fail("native D3D12 begin_external: null command list"), error);
    // Create-time work on our own queue must have finished before the caller's list can run (normally a no-op).
    if (!finish_queue() || !create_direct_objects() || !create_marker()) return extension_result(false, error);
    ext_list_ = list; external_ = true;
    // The arena is a buffer: it starts each ExecuteCommandLists in COMMON and is promoted implicitly to UNORDERED_ACCESS
    // by the first dispatch; only dispatches touch it while recording into the caller's list.
    arena_state_ = D3D12_RESOURCE_STATE_COMMON;
    native_binding_dirty_ = true; need_uav_barrier_ = true;
    return extension_result(true, error);
  }
  bool end_external(std::string* error) override {
    if (!external_ || inside_pass_) return extension_result(fail("native D3D12 end_external: not recording"), error);
    need_uav_barrier_ = true; uav_barrier();  // the caller's passes after ours read our results
    ++ext_recorded_;
    bool ok = true;
    NativeCom<ID3D12GraphicsCommandList2> l2;
    if (marker_map_ && SUCCEEDED(ext_list_->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), l2.pv())) && l2) {
      D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{marker_->GetGPUVirtualAddress(), static_cast<UINT>(ext_recorded_)};
      D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
      l2->WriteBufferImmediate(1, &p, &mode);
    } else if (marker_map_) {
      ok = fail("native D3D12 end_external: the caller's list has no ID3D12GraphicsCommandList2 (completion marker)");
    }
    ext_list_ = nullptr; external_ = false;
    arena_state_ = D3D12_RESOURCE_STATE_COMMON; native_binding_dirty_ = true;
    direct_bound_ = false;
    return extension_result(ok, error);
  }
  uint64_t arena_high_water() const override { return high_; }
  uint64_t arena_capacity() const override { return arena_bytes_; }
  uint64_t external_recorded() const override { return ext_recorded_; }
  uint64_t external_done() const override {
    if (!marker_map_) return ext_recorded_;
    const uint32_t low = *marker_map_;
    uint64_t v = (ext_recorded_ & ~0xffffffffull) | low;
    if (v > ext_recorded_) v -= 0x100000000ull;
    return v;
  }
  bool external_markers() const override { return marker_map_ != nullptr; }
  bool warm_up(std::string* error) override {
    if (external_) return extension_result(fail("native D3D12 warm_up: recording into the caller's list"), error);
    for (auto& entry : kernels_) {
      NativeKernel& k = entry.second;
      if (k.pso || !variant_matches(k) || (k.rs != "buffer" && k.rs != "direct")) continue;
      if (!create_pso(k)) return extension_result(false, error);
    }
    return extension_result(true, error);
  }

  std::string name() const override { return adapter_name_ + " (native D3D12)"; }
  std::string note() const override {
    return "native D3D12: arena " + std::to_string(arena_bytes_ >> 20) +
           " MiB; no HIP runtime or fallback; timestamp frequency " + std::to_string(timestamp_frequency_) + " Hz" +
           (borrowed_ ? "; borrowed device/queue; exact FP16/FP8 driver-channel probe passed" : "");
  }
  bool adapter_luid(uint8_t out[8]) const override {
    if (!out) return false;
    std::memcpy(out, &adapter_luid_, sizeof adapter_luid_); return true;
  }

  bool alloc(DevPtr* out, size_t bytes) override {
    if (!out) return fail("native D3D12 alloc: null output pointer");
    *out = 0;
    if (!bytes || bytes > arena_bytes_ - kNativeAlign) return fail("native D3D12 alloc: invalid byte count");
    const uint64_t need = (static_cast<uint64_t>(bytes) + kNativeAlign - 1) & ~(kNativeAlign - 1);
    for (auto i = free_.begin(); i != free_.end(); ++i) {
      if (i->second < need) continue;
      const uint64_t offset = i->first, rest = i->second - need;
      free_.erase(i);
      if (rest) free_[offset + need] = rest;
      used_[offset] = {need, bytes};
      high_ = std::max(high_, offset + need);
      in_use_ += need; *out = static_cast<DevPtr>(offset); return true;
    }
    return fail("native D3D12 arena exhausted for " + std::to_string(bytes) +
                " bytes; WG_D3D12_ARENA_MB is " + std::to_string(arena_bytes_ >> 20));
  }
  void free(DevPtr p) override {
    if (!p) return;
    auto i = used_.find(p);
    if (i == used_.end()) { fail("native D3D12 free: unknown allocation"); log(true, err_); return; }
    // Allocation reuse must never race an already recorded copy or dispatch.
    if (!finish_queue()) { log(true, "native D3D12 free retained allocation: " + err_); return; }
    // Debug validation can fail after the fence completed. Report it, but still reclaim this safe allocation.
    drain_debug_messages();
    uint64_t offset = i->first, size = i->second.reserved;
    in_use_ -= size; used_.erase(i);
    auto next = free_.lower_bound(offset);
    if (next != free_.end() && offset + size == next->first) { size += next->second; next = free_.erase(next); }
    if (next != free_.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == offset) { prev->second += size; return; }
    }
    free_[offset] = size;
  }

  bool upload(DevPtr dst, const void* src, size_t bytes) override {
    if (external_) return fail("native D3D12 upload: not available while recording into the caller's list");
    ProfileCpu cpu(this, "upload_api", bytes);
    if (!bytes) return true;
    if (!src || !range(dst, bytes, "upload")) return src ? false : fail("native D3D12 upload: null source");
    if (!transient_room(bytes)) return false;
    NativeCom<ID3D12Resource> staging;
    if (!buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, staging)) return false;
    void* mapped = nullptr;
    D3D12_RANGE no_read{0, 0};
    if (!check(staging->Map(0, &no_read, &mapped), "Map(upload)")) return false;
    { ProfileCpu transfer(this, "upload_memcpy", bytes); std::memcpy(mapped, src, bytes); }
    D3D12_RANGE written{0, bytes}; staging->Unmap(0, &written);
    if (!open_list()) return false;
    transition_arena(D3D12_RESOURCE_STATE_COPY_DEST);
    const auto stamp = profile_begin("upload", "host_to_arena", bytes);
    list_->CopyBufferRegion(arena_.get(), dst, staging.get(), 0, bytes);
    profile_end(stamp);
    keep(std::move(staging), bytes); return true;
  }
  bool download(void* dst, DevPtr src, size_t bytes) override {
    if (external_) return fail("native D3D12 download: not available while recording into the caller's list");
    ProfileCpu cpu(this, "download_api_including_wait", bytes);
    if (!bytes) return true;
    if (!dst || !range(src, bytes, "download")) return dst ? false : fail("native D3D12 download: null destination");
    NativeCom<ID3D12Resource> staging;
    if (!buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false, staging) || !open_list())
      return false;
    transition_arena(D3D12_RESOURCE_STATE_COPY_SOURCE);
    ID3D12Resource* resource = staging.get();
    const auto stamp = profile_begin("download", "arena_to_host", bytes);
    list_->CopyBufferRegion(resource, 0, arena_.get(), src, bytes);
    profile_end(stamp);
    keep(std::move(staging), bytes);
    if (!sync()) return false;
    void* mapped = nullptr; D3D12_RANGE read{0, bytes};
    if (!check(resource->Map(0, &read, &mapped), "Map(readback)")) return false;
    { ProfileCpu transfer(this, "readback_memcpy", bytes); std::memcpy(dst, mapped, bytes); }
    D3D12_RANGE no_write{0, 0}; resource->Unmap(0, &no_write); return true;
  }
  bool copy(DevPtr dst, DevPtr src, size_t bytes) override {
    if (external_) return fail("native D3D12 copy: not available while recording into the caller's list");
    if (!bytes) return true;
    if (!range(src, bytes, "copy source") || !range(dst, bytes, "copy destination")) return false;
    if (dst == src) return true;
    if (!transient_room(bytes)) return false;
    NativeCom<ID3D12Resource> scratch;
    if (!buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, false, scratch) || !open_list())
      return false;
    // One arena cannot simultaneously have COPY_SOURCE and COPY_DEST states. Scratch also defines overlap as memmove.
    transition_arena(D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto first = profile_begin("copy", "arena_to_scratch", bytes);
    list_->CopyBufferRegion(scratch.get(), 0, arena_.get(), src, bytes);
    profile_end(first);
    transition(scratch.get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition_arena(D3D12_RESOURCE_STATE_COPY_DEST);
    const auto second = profile_begin("copy", "scratch_to_arena", bytes);
    list_->CopyBufferRegion(arena_.get(), dst, scratch.get(), 0, bytes);
    profile_end(second);
    keep(std::move(scratch), bytes); return true;
  }
  bool memset(DevPtr dst, int value, size_t bytes) override {
    if (!bytes) return true;
    if (!range(dst, bytes, "memset")) return false;
    uint32_t count = static_cast<uint32_t>(bytes), byte = static_cast<unsigned>(value) & 255u;
    void* args[] = {&dst, &count, &byte};
    const uint64_t words = ((dst & 3u) + bytes + 3u) / 4u;
    return launch("native_fill", static_cast<unsigned>((words + 255u) / 256u), 256, args);
  }

  void* alloc_host(size_t bytes) override {
    if (!bytes) { fail("native D3D12 alloc_host: zero bytes"); return nullptr; }
    Host h; h.bytes = bytes;
    if (!buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false, h.resource)) return nullptr;
    void* mapped = nullptr; D3D12_RANGE read{0, bytes};
    if (!check(h.resource->Map(0, &read, &mapped), "Map(host readback)")) return nullptr;
    hosts_.emplace(reinterpret_cast<uintptr_t>(mapped), std::move(h)); return mapped;
  }
  void free_host(void* p) override {
    if (!p) return;
    auto i = hosts_.find(reinterpret_cast<uintptr_t>(p));
    if (i == hosts_.end()) { fail("native D3D12 free_host: unknown pointer"); log(true, err_); return; }
    if (!finish_queue()) { log(true, "native D3D12 free_host retained buffer: " + err_); return; }
    drain_debug_messages();
    D3D12_RANGE no_write{0, 0}; i->second.resource->Unmap(0, &no_write); hosts_.erase(i);
  }
  bool download_async(void* dst, DevPtr src, size_t bytes) override {
    if (external_) return fail("native D3D12 download_async: not available while recording into the caller's list");
    if (!bytes) return true;
    if (!range(src, bytes, "download_async")) return false;
    const uintptr_t address = reinterpret_cast<uintptr_t>(dst);
    auto i = hosts_.upper_bound(address);
    if (i == hosts_.begin()) return fail("native D3D12 download_async: destination is not alloc_host memory");
    --i; const uint64_t offset = address - i->first;
    if (offset >= i->second.bytes || bytes > i->second.bytes - offset)
      return fail("native D3D12 download_async: destination range exceeds alloc_host memory");
    if (!open_list()) return false;
    transition_arena(D3D12_RESOURCE_STATE_COPY_SOURCE);
    const auto stamp = profile_begin("download", "arena_to_host_async", bytes);
    list_->CopyBufferRegion(i->second.resource.get(), offset, arena_.get(), src, bytes);
    profile_end(stamp); return true;
  }

  bool load_module(const std::string& path) override {
    const std::string co = native_basename(path);
    if (!valid_co(co)) return fail("native D3D12: invalid code-object identity " + co);
    const std::string family = co.substr(0, co.find("__") == std::string::npos ? co.size() - 3 : co.find("__"));
    for (const auto& loaded : untagged_cos_) {
      const std::string previous = loaded.substr(0, loaded.find("__") == std::string::npos ? loaded.size() - 3 : loaded.find("__"));
      if (previous == family && loaded != co)
        return fail("native D3D12: ambiguous untagged code-object variants " + loaded + " and " + co);
    }
    untagged_cos_.insert(co); return true;
  }
  bool load_module_tagged(const std::string& path, const std::string& tag) override {
    const std::string co = native_basename(path);
    if (!valid_co(co) || !native_identifier(tag, false)) return fail("native D3D12: invalid tagged code-object identity");
    // A previously queued dispatch has already captured its PSO and arguments. This changes future launches only.
    tagged_cos_[tag] = co; return true;
  }
  void unordered_next() override { unordered_next_ = true; }
  // The port of `name` that the loaded code objects select. One name can have ports of several builds of its HIP kernel
  // (manifest entries with the same name and different "co", for example post_w64.co and post_w64__orig.co); without
  // a selected one, the first port (variant_loaded then names what it requires).
  NativeKernel* find_port(const std::string& name) {
    auto range = kernels_.equal_range(name);
    for (auto i = range.first; i != range.second; ++i)
      if (variant_matches(i->second)) return &i->second;
    return range.first == range.second ? nullptr : &range.first->second;
  }
  bool launch(const std::string& name, unsigned grid, unsigned block, void** args) override {
    NativeKernel* port = find_port(name);
    if (!port) return fail("native D3D12 unsupported kernel: " + name + "; no HIP fallback");
    NativeKernel& k = *port;
    // Legacy tex ports have a different ABI and remain unsupported. A direct port needs this frame's typed
    // descriptors before variant/argument inspection, including for a caller with no argument array.
    const bool direct = k.rs == "direct";
    if (k.rs != "buffer" && !direct) return fail("native D3D12 unsupported root signature " + k.rs + " for " + name);
    if (direct && !direct_bound_) return fail("native D3D12 direct kernel requires a bound direct frame: " + name);
    // Check the variant BEFORE touching args: a 37-DWORD post variant cannot satisfy a 42-DWORD port.
    if (!variant_loaded(k)) return false;
    if (block != k.block) return fail("native D3D12 kernel " + name + ": block must be " + std::to_string(k.block));
    if (!grid || grid > 0x7fffffffu) return fail("native D3D12 kernel " + name + ": grid must be 1..2147483647");
    const uint64_t groups = (uint64_t)grid * k.gscale;   // D3D12 workgroups
    if (groups > 0x7fffffffu) return fail("native D3D12 kernel " + name + ": too many workgroups");
    uint32_t constants[kNativeConstants + kNativeArgs] = {};
    if (!pack(k, args, constants + kNativeConstants)) return false;
    if ((!k.pso && !create_pso(k)) || !open_list()) return false;
    if (native_binding_dirty_ || direct_root_bound_ != direct) bind_native(direct);
    transition_arena(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (unordered_next_) need_uav_barrier_ = false;  // unordered_next(): no barrier after the previous launch
    unordered_next_ = false;
    uav_barrier();
    cl()->SetPipelineState(k.pso.get()); constants[1] = static_cast<uint32_t>(groups);
    const auto stamp = profile_begin(name == "native_fill" ? "fill" : "dispatch", name.c_str());
    for (uint64_t base = 0; base < groups; base += kNativeGroups) {
      constants[0] = static_cast<uint32_t>(base);
      cl()->SetComputeRoot32BitConstants(2, kNativeConstants + k.dwords, constants, 0);
      cl()->Dispatch(static_cast<unsigned>(std::min<uint64_t>(groups - base, kNativeGroups)), 1, 1);
    }
    profile_end(stamp);
    need_uav_barrier_ = true;
    trace_launch(name, grid, block, "d3d12");
    ++dispatches_;
    return !batch_ || external_ || dispatches_ < batch_ || submit();
  }

  bool sync() override {
    const bool completed = finish_queue();
    const std::string completion_error = completed ? std::string() : err_;
    const bool valid = drain_debug_messages();
    if (completed) profile_drain_all();
    if (!completed) err_ = completion_error + (valid ? "" : "; " + err_);
    if (completed && valid && !profile_failure_.empty()) return fail(profile_failure_);
    return completed && valid;
  }
  bool reserve_events(int n) override {
    if (n < 0 || static_cast<unsigned>(n) > kMaxEvents) return fail("native D3D12: invalid event pool size");
    if (static_cast<size_t>(n) <= events_.size()) return true;
    if (!sync()) return false;
    NativeCom<ID3D12QueryHeap> queries;
    D3D12_QUERY_HEAP_DESC q{}; q.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; q.Count = static_cast<unsigned>(n);
    if (!check(device_->CreateQueryHeap(&q, __uuidof(ID3D12QueryHeap), queries.pv()), "CreateQueryHeap")) return false;
    NativeCom<ID3D12Resource> readback;
    if (!buffer(static_cast<uint64_t>(n) * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST, false, readback)) return false;
    // Completed timestamps survive a grow in CPU storage. New recordings resolve into the new pool.
    for (size_t i = 0; i < events_.size(); ++i)
      if (events_[i].recorded && !cache_timestamp(static_cast<int>(i))) return false;
    events_.resize(n); timestamps_.resize(n); cached_.resize(n, false);
    query_heap_ = std::move(queries); query_readback_ = std::move(readback); return true;
  }
  bool record(int event) override {
    if (!event_index(event)) return false;
    if (!open_list()) return false;
    // UAV dependencies must complete before the timestamp, including when no following dispatch inserts a barrier.
    uav_barrier();
    cl()->EndQuery(query_heap_.get(), D3D12_QUERY_TYPE_TIMESTAMP, static_cast<UINT>(event));
    cl()->ResolveQueryData(query_heap_.get(), D3D12_QUERY_TYPE_TIMESTAMP, static_cast<UINT>(event), 1,
                            query_readback_.get(), static_cast<uint64_t>(event) * sizeof(uint64_t));
    if (external_) { events_[event] = {ext_recorded_ + 1, true, false, true}; cached_[event] = false; return true; }
    events_[event] = {0, true, true, false}; cached_[event] = false;
    slots_[current_slot_].events.push_back(event); return true;
  }
  int event_done(int event) override {
    if (!recorded_event(event)) return -1;
    if (events_[event].external) return external_done() >= events_[event].fence ? 1 : 0;
    if (events_[event].pending && !submit()) return -1;
    const uint64_t completed = fence_->GetCompletedValue();
    if (completed == UINT64_MAX) { fail("native D3D12: device removed while checking event"); return -1; }
    return completed >= events_[event].fence ? 1 : 0;
  }
  bool elapsed_ms(int start, int stop, float* ms) override {
    return recorded_event(start) && recorded_event(stop) && sync() && elapsed_done_ms(start, stop, ms);
  }
  bool elapsed_done_ms(int start, int stop, float* ms) override {
    if (!ms) return fail("native D3D12 elapsed: null result");
    if (event_done(start) != 1 || event_done(stop) != 1) return fail("native D3D12 elapsed: events are not complete");
    if (!cache_timestamp(start) || !cache_timestamp(stop)) return false;
    if (timestamps_[stop] < timestamps_[start]) return fail("native D3D12 elapsed: events are reversed");
    *ms = static_cast<float>(static_cast<double>(timestamps_[stop] - timestamps_[start]) *
                             1000.0 / static_cast<double>(timestamp_frequency_));
    return true;
  }
  void begin_teardown() override {
    log(false, "native D3D12 teardown: " + std::to_string(in_use_) + " arena bytes in use");
    if (!teardown_registered_ && device_) {  // clean/fe-minimize: a new context on this device waits for it
      teardown_registered_ = true;
      teardown_begin(device_.get());
    }
  }

 private:
  struct ProfileOp { std::string kind, name; uint64_t bytes, sequence; };
  struct ProfilePage {
    NativeCom<ID3D12QueryHeap> queries;
    NativeCom<ID3D12Resource> readback;
    std::vector<ProfileOp> operations;
  };
  struct ProfileStamp { unsigned page = UINT32_MAX, operation = 0; };
  struct ProfileCpu {
    NativeD3D12Device* owner;
    const char* name;
    uint64_t start, bytes;
    ProfileCpu(NativeD3D12Device* device, const char* operation, uint64_t count = 0)
        : owner(device->profile_enabled_ ? device : nullptr), name(operation),
          start(owner ? profile_clock() : 0), bytes(count) {}
    ~ProfileCpu() { if (owner) owner->profile_cpu(name, start, profile_clock(), bytes); }
  };
  struct Slot {
    NativeCom<ID3D12CommandAllocator> allocator;
    uint64_t fence = 0, bytes = 0;
    std::vector<NativeCom<ID3D12Resource>> keep;
    std::vector<NativeCom<ID3D12Fence>> keep_fences;
    std::vector<int> events;
    std::vector<ProfilePage> profile;
    bool profile_drained = true;
  };
  struct Host { NativeCom<ID3D12Resource> resource; size_t bytes = 0; };

  static uint64_t profile_clock() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
  }
  static std::string profile_quote(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
      if (c == '\\' || c == '"') { result += '\\'; result += static_cast<char>(c); }
      else if (c < 32) {
        char escaped[7]; std::snprintf(escaped, sizeof escaped, "\\u%04x", static_cast<unsigned>(c)); result += escaped;
      } else result += static_cast<char>(c);
    }
    return result + '"';
  }
  void profile_error(const std::string& error) {
    if (profile_failure_.empty()) {
      profile_failure_ = "native D3D12 profiling: " + error;
      debug_log(true, profile_failure_);
    }
  }
  bool profile_init() {
    const char* path = std::getenv("WG_D3D12_PROFILE");
    if (!path || !*path) return true;
    // Appending permits several fresh processes or repeated create/destroy to share a diagnostic capture. Every
    // record includes a unique device session; simultaneous processes must still use separate output files.
    profile_output_.open(path, std::ios::out | std::ios::app);
    if (!profile_output_) return fail("native D3D12 profiling: cannot open WG_D3D12_PROFILE output");
    profile_session_ = std::to_string(GetCurrentProcessId()) + "-" + std::to_string(profile_clock());
    profile_enabled_ = true;
    profile_output_ << std::setprecision(12) << "{\"type\":\"meta\",\"version\":1,\"session\":"
                    << profile_quote(profile_session_) << ",\"pid\":" << GetCurrentProcessId()
                    << ",\"frequency_hz\":" << timestamp_frequency_ << ",\"adapter\":" << profile_quote(adapter_name_)
                    << ",\"diagnostic_only\":true,\"gpu_definition\":\"bottom-of-pipe timestamp intervals;"
                       " instrumentation perturbs scheduling; barrier intervals are not isolated hardware cost\","
                       "\"cpu_definition\":\"steady-clock wall intervals; upload_api includes staging allocation and memcpy;"
                       " download_api_including_wait includes submission, wait and memcpy; nested scopes overlap\"}\n";
    profile_output_.flush();
    return profile_output_ ? true : fail("native D3D12 profiling: cannot write capture header");
  }
  ProfileStamp profile_begin(const char* kind, const char* name, uint64_t bytes = 0) {
    if (!profile_enabled_ || !profile_failure_.empty() || external_) return {};
    Slot& slot = slots_[current_slot_];
    if (slot.profile.empty() || slot.profile.back().operations.size() == kProfileQueries / 2) {
      ProfilePage page;
      D3D12_QUERY_HEAP_DESC q{}; q.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; q.Count = kProfileQueries;
      if (!check(device_->CreateQueryHeap(&q, __uuidof(ID3D12QueryHeap), page.queries.pv()), "CreateQueryHeap(profile)") ||
          !buffer(kProfileQueries * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK,
                  D3D12_RESOURCE_STATE_COPY_DEST, false, page.readback)) {
        profile_error(err_); return {};
      }
      slot.profile.emplace_back(std::move(page));
    }
    ProfilePage& page = slot.profile.back();
    ProfileStamp stamp{static_cast<unsigned>(slot.profile.size() - 1), static_cast<unsigned>(page.operations.size())};
    page.operations.push_back({kind, name, bytes, ++profile_sequence_});
    slot.profile_drained = false;
    list_->EndQuery(page.queries.get(), D3D12_QUERY_TYPE_TIMESTAMP, stamp.operation * 2);
    return stamp;
  }
  void profile_end(ProfileStamp stamp) {
    if (stamp.page == UINT32_MAX) return;
    // A start that succeeded must always receive its end, even if another profiler operation reported an error.
    ProfilePage& page = slots_[current_slot_].profile[stamp.page];
    list_->EndQuery(page.queries.get(), D3D12_QUERY_TYPE_TIMESTAMP, stamp.operation * 2 + 1);
  }
  void profile_resolve(Slot& slot) {
    for (auto& page : slot.profile) if (!page.operations.empty())
      list_->ResolveQueryData(page.queries.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                              static_cast<UINT>(page.operations.size() * 2), page.readback.get(), 0);
  }
  void profile_drain(Slot& slot) {
    if (!profile_enabled_ || slot.profile_drained) return;
    // Caller must establish this slot's completion; never map query data belonging to pending commands.
    const uint64_t completed = fence_->GetCompletedValue();
    if (!slot.fence || completed == UINT64_MAX || completed < slot.fence) {
      profile_error("attempted query readback before its submission fence completed"); return;
    }
    for (auto& page : slot.profile) {
      void* mapped = nullptr; D3D12_RANGE read{0, page.operations.size() * 2 * sizeof(uint64_t)};
      if (!check(page.readback->Map(0, &read, &mapped), "Map(profile readback)")) { profile_error(err_); continue; }
      const auto* ticks = static_cast<const uint64_t*>(mapped);
      for (size_t n = 0; n < page.operations.size(); ++n) {
        const auto& op = page.operations[n]; const uint64_t start = ticks[n * 2], stop = ticks[n * 2 + 1];
        if (stop < start) profile_error("reversed GPU timestamp interval");
        profile_output_ << "{\"type\":\"gpu\",\"session\":" << profile_quote(profile_session_)
                        << ",\"sequence\":" << op.sequence << ",\"fence\":" << slot.fence
                        << ",\"kind\":" << profile_quote(op.kind) << ",\"name\":" << profile_quote(op.name)
                        << ",\"bytes\":" << op.bytes << ",\"start_tick\":" << start << ",\"stop_tick\":" << stop
                        << ",\"valid\":" << (stop >= start ? "true" : "false") << ",\"ms\":"
                        << (stop >= start ? static_cast<double>(stop - start) * 1000.0 / timestamp_frequency_ : 0.0)
                        << "}\n";
      }
      D3D12_RANGE no_write{0, 0}; page.readback->Unmap(0, &no_write);
    }
    slot.profile_drained = true;
    if (!profile_output_) profile_error("capture write failed");
  }
  void profile_drain_all() {
    if (!profile_enabled_) return;
    for (auto& slot : slots_) profile_drain(slot);
    profile_output_.flush();
    if (!profile_output_) profile_error("capture flush failed");
  }
  void profile_cpu(const char* name, uint64_t start, uint64_t stop, uint64_t bytes) {
    profile_output_ << "{\"type\":\"cpu\",\"session\":" << profile_quote(profile_session_)
                    << ",\"name\":" << profile_quote(name) << ",\"bytes\":" << bytes
                    << ",\"start_ns\":" << start << ",\"stop_ns\":" << stop
                    << ",\"ms\":" << static_cast<double>(stop - start) / 1e6 << "}\n";
    if (!profile_output_) profile_error("CPU capture write failed");
  }
  void profile_finish(bool completed) {
    if (!profile_enabled_) return;
    profile_output_ << "{\"type\":\"end\",\"session\":" << profile_quote(profile_session_)
                    << ",\"gpu_operations\":" << profile_sequence_ << ",\"valid\":"
                    << (completed && profile_failure_.empty() ? "true" : "false") << ",\"error\":"
                    << profile_quote(profile_failure_) << "}\n";
    profile_output_.flush();
  }

  bool fail(const std::string& s) { err_ = s; return false; }
  bool extension_result(bool ok, std::string* error) {
    if (error) *error = ok ? std::string() : err_;
    return ok;
  }
  bool identity(IUnknown* object, NativeCom<IUnknown>& canonical) {
    return object && check(object->QueryInterface(__uuidof(IUnknown), canonical.pv()), "QueryInterface(IUnknown)");
  }
  bool same_identity(IUnknown* a, IUnknown* b, const char* label) {
    NativeCom<IUnknown> x, y;
    if (!identity(a, x) || !identity(b, y)) return fail(std::string("native D3D12: cannot identify ") + label);
    return x.get() == y.get() || fail(std::string("native D3D12: ") + label + " has different canonical device identity");
  }
  bool finish_queue() {
    if (inside_pass_) return fail("native D3D12: synchronization from record_pass callback is not allowed");
    if (external_) return fail("native D3D12: synchronization while recording into the caller's command list");
    if (!queue_ || !fence_) return true;
    return submit() && wait(last_submitted_);
  }
  void debug_log(bool warning, const std::string& message) {
    // Device-only executables have no log sink. Explicit debug runs must still retain their validation evidence.
    if (log_) log(warning, message);
    else { std::fprintf(stderr, "%s\n", message.c_str()); std::fflush(stderr); }
  }
  void debug_failure(const std::string& message) {
    if (debug_failure_.empty()) debug_failure_ = message;
    debug_log(true, message);
  }
  void debug_summary() {
    if (!info_queue_) return;
    debug_log(false, "native D3D12 debug summary: " + std::to_string(debug_messages_) + " messages, " +
                        std::to_string(debug_errors_) + " errors/corruption, " + std::to_string(debug_warnings_) +
                        " warnings, " + std::to_string(debug_discarded_) + " discarded");
  }
  bool drain_debug_messages() {
    if (!info_queue_) return true;
    if (borrowed_) {
      const uint64_t denied = info_queue_->GetNumMessagesDeniedByStorageFilter();
      if (denied != debug_denied_)
        debug_failure("native D3D12 debug: borrowed storage filter denied messages; validation is incomplete");
      debug_denied_ = denied;
      if (info_queue_->GetNumStoredMessagesAllowedByRetrievalFilter() != info_queue_->GetNumStoredMessages())
        debug_failure("native D3D12 debug: borrowed retrieval filter hides messages; validation is incomplete");
    }
    const uint64_t discarded = info_queue_->GetNumMessagesDiscardedByMessageCountLimit();
    if (discarded > debug_discarded_)
      debug_failure("native D3D12 debug: message limit discarded " + std::to_string(discarded - debug_discarded_) +
                    " messages; validation is incomplete");
    debug_discarded_ = discarded;
    const uint64_t count = info_queue_->GetNumStoredMessagesAllowedByRetrievalFilter();
    if (borrowed_ && count < debug_cursor_)
      debug_failure("native D3D12 debug: borrowed InfoQueue cleared externally; validation is incomplete");
    for (uint64_t index = borrowed_ ? std::min(debug_cursor_, count) : 0; index < count; ++index) {
      SIZE_T size = 0;
      HRESULT hr = info_queue_->GetMessage(index, nullptr, &size);
      if (FAILED(hr) || size < sizeof(D3D12_MESSAGE) || size > 16u * 1024u * 1024u) {
        debug_failure("native D3D12 debug: could not size message " + std::to_string(index)); continue;
      }
      std::vector<unsigned char> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (FAILED(info_queue_->GetMessage(index, message, &size))) {
        debug_failure("native D3D12 debug: could not retrieve message " + std::to_string(index)); continue;
      }
      ++debug_messages_;
      const bool error = message->Severity == D3D12_MESSAGE_SEVERITY_ERROR ||
                         message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION;
      const bool warning = message->Severity == D3D12_MESSAGE_SEVERITY_WARNING;
      const char* severity = message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ? "CORRUPTION" :
                             error ? "ERROR" : warning ? "WARNING" : "INFO";
      std::string description;
      const uintptr_t begin = reinterpret_cast<uintptr_t>(storage.data());
      const uintptr_t text = reinterpret_cast<uintptr_t>(message->pDescription);
      if (text < begin || text - begin > storage.size() ||
          message->DescriptionByteLength > storage.size() - (text - begin)) {
        debug_failure("native D3D12 debug: invalid description in message " + std::to_string(index)); continue;
      }
      description.assign(message->pDescription, message->DescriptionByteLength);
      while (!description.empty() && description.back() == '\0') description.pop_back();
      std::replace(description.begin(), description.end(), '\r', ' ');
      std::replace(description.begin(), description.end(), '\n', ' ');
      const std::string line = "native D3D12 debug " + std::string(severity) + " id=" +
                               std::to_string(static_cast<unsigned>(message->ID)) + ": " + description;
      if (error) { ++debug_errors_; debug_failure(line); }
      else { if (warning) ++debug_warnings_; debug_log(warning, line); }
    }
    if (borrowed_) debug_cursor_ = count;
    else info_queue_->ClearStoredMessages();
    // Validation failure is sticky; a later empty queue must never turn a failed debug run into a pass.
    return debug_failure_.empty() ? true : fail(debug_failure_);
  }
  bool check(HRESULT hr, const char* operation) {
    if (SUCCEEDED(hr)) return true;
    char text[96]; std::snprintf(text, sizeof text, "%s failed (0x%08lx)", operation, static_cast<unsigned long>(hr));
    std::string error = text;
    if (device_) {
      HRESULT removed = device_->GetDeviceRemovedReason();
      if (FAILED(removed)) { std::snprintf(text, sizeof text, "; device removed (0x%08lx)", static_cast<unsigned long>(removed)); error += text; }
    }
    return fail("native D3D12: " + error);
  }
  bool environment_number(const char* name, uint64_t low, uint64_t high, uint64_t* value) {
    const char* s = std::getenv(name);
    if (!s) return true;
    if (!*s) return fail(std::string(name) + " must be an integer");
    for (const char* p = s; *p; ++p) if (*p < '0' || *p > '9') return fail(std::string(name) + " must be an unsigned integer");
    errno = 0; char* end = nullptr; unsigned long long parsed = std::strtoull(s, &end, 10);
    if (errno || !end || *end || parsed < low || parsed > high)
      return fail(std::string(name) + " must be " + std::to_string(low) + ".." + std::to_string(high));
    *value = parsed; return true;
  }
  bool range(DevPtr pointer, size_t bytes, const char* operation) {
    auto i = used_.upper_bound(pointer);
    if (i != used_.begin()) {
      --i; const uint64_t offset = pointer - i->first;
      if (offset < i->second.requested && bytes <= i->second.requested - offset) return true;
    }
    return fail(std::string("native D3D12 ") + operation + ": range is outside a live allocation");
  }
  static bool valid_co(const std::string& co) {
    return native_filename(co) && co.size() > 3 && co.substr(co.size() - 3) == ".co";
  }
  // variant_loaded without an error: whether the runtime's loaded code objects select this port.
  bool variant_matches(const NativeKernel& k) const {
    if (k.co == "-") return true;
    const size_t slash = k.name.find('/');
    if (slash == std::string::npos) return untagged_cos_.count(k.co) != 0;
    auto i = tagged_cos_.find(k.name.substr(0, slash));
    return i != tagged_cos_.end() && i->second == k.co;
  }
  // The completion marker of caller-list frames: a CPU-visible buffer that WriteBufferImmediate writes (COPY_DEST).
  bool create_marker() {
    if (marker_) return true;
    if (!buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false, marker_)) return false;
    void* mapped = nullptr; D3D12_RANGE read{0, 256};
    if (!check(marker_->Map(0, &read, &mapped), "Map(completion marker)")) return false;
    marker_map_ = static_cast<volatile uint32_t*>(mapped);
    *marker_map_ = 0;
    return true;
  }
  // Waits (bounded, 5 s) until the caller-list frame `serial` has completed; 0 = nothing to wait for.
  bool wait_external(uint64_t serial) {
    if (!serial || external_done() >= serial) return true;
    for (int i = 0; i < 5000 && external_done() < serial; ++i) {
      if (device_ && FAILED(device_->GetDeviceRemovedReason())) return fail("native D3D12: device removed");
      Sleep(1);
    }
    return external_done() >= serial ? true
        : fail("native D3D12: the caller's command list with frame " + std::to_string(serial) +
               " did not complete within 5 s (descriptor set reuse)");
  }
  ID3D12GraphicsCommandList* cl() const { return external_ ? ext_list_ : list_.get(); }
  bool variant_loaded(const NativeKernel& k) {
    if (k.co == "-") return true;
    const size_t slash = k.name.find('/');
    if (slash == std::string::npos) {
      if (untagged_cos_.count(k.co)) return true;
    } else {
      auto i = tagged_cos_.find(k.name.substr(0, slash));
      if (i != tagged_cos_.end() && i->second == k.co) return true;
    }
    return fail("native D3D12 unsupported code-object variant for " + k.name + ": port requires " + k.co + "; no HIP fallback");
  }
  bool pack(const NativeKernel& k, void** args, uint32_t* constants) {
    if (!args && !k.args.empty()) return fail("native D3D12 kernel " + k.name + ": null argument array");
    unsigned d = 0;
    for (size_t n = 0; n < k.args.size(); ++n) {
      if (!args[n]) return fail("native D3D12 kernel " + k.name + ": null argument value");
      const NativeArg& a = k.args[n];
      if (a.kind == 'p') {
        DevPtr pointer = 0; std::memcpy(&pointer, args[n], sizeof pointer);
        if (pointer && !range(pointer, 1, "kernel pointer")) return false;
        constants[d++] = static_cast<uint32_t>(pointer);
      } else {
        std::memcpy(constants + d, args[n], a.bytes); d += a.bytes / 4;
      }
    }
    return true;
  }
  bool event_index(int event) {
    return event >= 0 && static_cast<size_t>(event) < events_.size() ? true : fail("native D3D12: invalid event index");
  }
  bool recorded_event(int event) {
    return event_index(event) && (events_[event].recorded || fail("native D3D12: event was not recorded"));
  }
  bool cache_timestamp(int event) {
    if (cached_[event]) return true;
    void* mapped = nullptr;
    const SIZE_T offset = static_cast<SIZE_T>(event) * sizeof(uint64_t);
    D3D12_RANGE read{offset, offset + sizeof(uint64_t)};
    if (!check(query_readback_->Map(0, &read, &mapped), "Map(timestamp readback)")) return false;
    std::memcpy(&timestamps_[event], static_cast<const char*>(mapped) + offset, sizeof(uint64_t));
    D3D12_RANGE no_write{0, 0}; query_readback_->Unmap(0, &no_write); cached_[event] = true; return true;
  }
  bool wait(uint64_t value) {
    if (!value) return true;
    uint64_t completed = fence_->GetCompletedValue();
    if (completed == UINT64_MAX) return fail("native D3D12: device removed while waiting for queue");
    if (completed >= value) return true;
    if (!check(fence_->SetEventOnCompletion(value, event_), "SetEventOnCompletion")) return false;
    ProfileCpu cpu(this, "fence_wait_block");
    const DWORD result = WaitForSingleObject(event_, INFINITE);
    if (result != WAIT_OBJECT_0) return fail("native D3D12: queue wait failed (Win32 " + std::to_string(GetLastError()) + ")");
    completed = fence_->GetCompletedValue();
    return completed != UINT64_MAX && completed >= value ? true : fail("native D3D12: queue wait ended without completion");
  }
  bool open_list() {
    if (inside_pass_) return fail("native D3D12: device operations from record_pass callback are not allowed");
    if (external_) return true;
    if (list_open_) return true;
    if (queue_unfenced_ && !submit()) return false;
    const unsigned next = (current_slot_ + 1) % kNativeSlots;
    Slot& slot = slots_[next];
    if (!wait(slot.fence) || !check(slot.allocator->Reset(), "CommandAllocator::Reset")) return false;
    profile_drain(slot);
    slot.profile.clear(); slot.profile_drained = true;
    slot.keep.clear(); slot.keep_fences.clear(); slot.events.clear(); slot.bytes = 0;
    if (!check(list_->Reset(slot.allocator.get(), nullptr), "CommandList::Reset")) return false;
    current_slot_ = next; list_open_ = true; dispatches_ = 0;
    bind_native();
    return true;
  }
  void bind_native(bool direct = false) {
    // Caller-list recording uses the direct heap for both root signatures (each set starts with the channel UAV, the
    // buffer signature's whole table), so the caller's list never switches descriptor heaps inside one of our frames.
    const bool ring = direct || external_;
    ID3D12DescriptorHeap* descriptor_heap = ring ? direct_heap_.get() : heap_.get();
    ID3D12GraphicsCommandList* l = cl();
    l->SetDescriptorHeaps(1, &descriptor_heap);
    l->SetComputeRootSignature(direct ? direct_root_.get() : root_.get());
    l->SetComputeRootUnorderedAccessView(0, arena_->GetGPUVirtualAddress());
    l->SetComputeRootUnorderedAccessView(1, arena_->GetGPUVirtualAddress());
    D3D12_GPU_DESCRIPTOR_HANDLE table = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    if (ring) table.ptr += static_cast<uint64_t>(direct_slot_) * kDirectSet * descriptor_stride_;
    l->SetComputeRootDescriptorTable(3, table);
    direct_root_bound_ = direct;
    native_binding_dirty_ = false;
  }
  bool submit() {
    if (inside_pass_) return fail("native D3D12: submission from record_pass callback is not allowed");
    if (external_) return fail("native D3D12: submission while recording into the caller's command list");
    if (!list_open_ && !queue_unfenced_) return true;
    ProfileCpu cpu(this, "submit_api");
    if (list_open_) {
      if (profile_enabled_) profile_resolve(slots_[current_slot_]);
      if (!check(list_->Close(), "CommandList::Close")) return false;
      list_open_ = false;
      ID3D12CommandList* commands = list_.get();
      queue_->ExecuteCommandLists(1, &commands); queue_unfenced_ = true;
      // Buffer resources decay to COMMON at the end of each ExecuteCommandLists call. The next list is separately
      // submitted on this ordered queue and can implicitly promote its first arena access without a transition.
      arena_state_ = D3D12_RESOURCE_STATE_COMMON;
    }
    const uint64_t value = ++next_fence_;
    if (!check(queue_->Signal(fence_.get(), value), "CommandQueue::Signal")) return false;
    queue_unfenced_ = false;
    last_submitted_ = value; Slot& slot = slots_[current_slot_]; slot.fence = value;
    for (int event : slot.events) if (events_[event].pending) { events_[event].pending = false; events_[event].fence = value; }
    slot.events.clear(); return true;
  }
  void retain_gpu_objects() {
    // Exceptional failure path only: detach COM owners rather than Release objects still referenced by GPU commands.
    for (auto& h : hosts_) h.second.resource.p = nullptr;
    for (auto& k : kernels_) k.second.pso.p = nullptr;
    for (auto& s : slots_) {
      for (auto& r : s.keep) r.p = nullptr;
      for (auto& f : s.keep_fences) f.p = nullptr;
      for (auto& p : s.profile) { p.queries.p = nullptr; p.readback.p = nullptr; }
      s.allocator.p = nullptr;
    }
    query_readback_.p = nullptr; query_heap_.p = nullptr; list_.p = nullptr; arena_.p = nullptr;
    channel_.p = nullptr; heap_.p = nullptr; root_.p = nullptr; gelu_t5_.p = nullptr; queue_.p = nullptr;
    fence_.p = nullptr;
    for (auto& set : direct_textures_) for (auto& texture : set) texture.p = nullptr;
    marker_.p = nullptr;
    direct_constants_.p = nullptr; direct_heap_.p = nullptr; direct_root_.p = nullptr;
    info_queue_.p = nullptr; device_.p = nullptr;
    // Keep system DLL references and the completion-event handle too: an outstanding registration can still use it.
    d3d12_dll_ = dxgi_dll_ = nullptr; event_ = nullptr;
  }
  bool transient_room(uint64_t bytes) {
    if (list_open_ && slots_[current_slot_].bytes &&
        (bytes > kTransientSubmitBytes || slots_[current_slot_].bytes > kTransientSubmitBytes - bytes))
      return submit();
    return true;
  }
  void keep(NativeCom<ID3D12Resource>&& resource, uint64_t bytes) {
    Slot& slot = slots_[current_slot_]; slot.keep.emplace_back(std::move(resource)); slot.bytes += bytes;
  }
  void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource; barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before; barrier.Transition.StateAfter = after;
    const auto stamp = profile_begin("barrier", "transition");
    cl()->ResourceBarrier(1, &barrier);
    profile_end(stamp);
  }
  void transition_arena(D3D12_RESOURCE_STATES target) {
    if (target == arena_state_) return;
    if (arena_state_ != D3D12_RESOURCE_STATE_COMMON) transition(arena_.get(), arena_state_, target);
    arena_state_ = target;
    // Transition barriers complete earlier arena UAV writes; the channel itself remains covered by the UAV barrier.
  }
  void uav_barrier() {
    if (!need_uav_barrier_) return;
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = nullptr;
    const auto stamp = profile_begin("barrier", "uav");
    cl()->ResourceBarrier(1, &b); need_uav_barrier_ = false;
    profile_end(stamp);
  }
  // clean/fe-minimize: the arena, after another context's teardown on this device has released its memory (waited for
  // at most 5 s), and tried again for up to 5 s more when the device is out of memory while such a teardown finishes
  // (a host that recreates its upscaler creates the new context before it destroys the old one). Logged when it waited.
  bool arena_buffer() {
    const int64_t waited = teardown_wait(device_.get(), 5000);
    if (waited >= 0)
      log(false, "native D3D12: waited " + std::to_string(waited) + " ms for another context's teardown on this device "
                 "(its GPU memory released) before allocating the arena");
    D3D12_HEAP_PROPERTIES h{}; h.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = arena_bytes_;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto t0 = std::chrono::steady_clock::now();
    HRESULT hr = E_FAIL;
    for (int attempt = 0;; ++attempt) {
      hr = device_->CreateCommittedResource(&h, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                            __uuidof(ID3D12Resource), arena_.pv());
      const auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
      if (hr != E_OUTOFMEMORY || spent >= 5000) {
        if (attempt > 0)
          log(hr != S_OK, "native D3D12: the arena (" + std::to_string(arena_bytes_) + " bytes) " +
                              (SUCCEEDED(hr) ? "allocated" : "still out of memory") + " after " + std::to_string(attempt) +
                              " retr" + (attempt == 1 ? "y" : "ies") + " in " + std::to_string(spent) + " ms");
        break;
      }
      if (teardown_wait(device_.get(), 250) < 0) Sleep(250);  // a teardown that starts meanwhile releases memory
    }
    return check(hr, "CreateCommittedResource");
  }
  bool buffer(uint64_t bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, bool uav,
              NativeCom<ID3D12Resource>& result) {
    D3D12_HEAP_PROPERTIES h{}; h.Type = type;
    D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = bytes;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (uav) d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return check(device_->CreateCommittedResource(&h, D3D12_HEAP_FLAG_NONE, &d, state, nullptr,
                                                  __uuidof(ID3D12Resource), result.pv()), "CreateCommittedResource");
  }

  bool read_manifest(const std::string& path) {
    // The file on disk, else the copy built into the library under the same name (clean/runner/build.py).
    std::vector<char> bytes;
    if (!wg_read_file(path, &bytes)) return fail("native D3D12 manifest missing: " + path);
    if (bytes.empty() || bytes.size() > 4 * 1024 * 1024) return fail("native D3D12 manifest: invalid file size");
    const std::string text(bytes.begin(), bytes.end());
    wgc::Json manifest; std::string error;
    if (!wgc::parse_json(text, manifest, error)) return fail("native D3D12 manifest: " + error);
    const auto* format = manifest.find("format"); const auto* kernels = manifest.find("kernels");
    int64_t version = 0;
    if (!format || !format->as_int64(version) || version != 1 || !kernels || !kernels->is_array() || kernels->items.empty())
      return fail("native D3D12 manifest: expected format 1 with nonempty kernels array");
    const std::string directory = path.substr(0, path.find_last_of("/\\") + 1);
    for (const auto& entry : kernels->items) {
      NativeKernel k; std::string filename, arguments, hash;
      auto string_field = [&](const char* key, std::string& value) {
        const auto* p = entry.find(key); if (!p || !p->is_string()) return false; value = p->text; return true;
      };
      auto number_field = [&](const char* key, unsigned& value) {
        const auto* p = entry.find(key); int64_t n = 0;
        if (!p || !p->as_int64(n) || n < 0 || n > UINT32_MAX) return false;
        value = static_cast<unsigned>(n); return true;
      };
      if (!string_field("name", k.name) || !string_field("file", filename) || !string_field("co", k.co) ||
          !string_field("args", arguments) || !string_field("sha256", hash) || !number_field("block", k.block) ||
          !number_field("wave", k.wave) || !number_field("arg_dwords", k.dwords))
        return fail("native D3D12 manifest: incomplete or wrongly typed kernel entry");
      if (const auto* gp = entry.find("group")) {
        int64_t gn = 0;
        if (!gp->as_int64(gn) || gn <= 0 || gn > (int64_t)k.block || k.block % (unsigned)gn || (unsigned)gn % k.wave)
          return fail("native D3D12 manifest: invalid group for " + k.name);
        k.group = (unsigned)gn == k.block ? 0u : (unsigned)gn;
        k.gscale = k.block / (unsigned)gn;
        if (const auto* sp = entry.find("gscale")) {
          int64_t sn = 0;
          if (!sp->as_int64(sn) || sn < 1 || sn > (int64_t)(k.block / (unsigned)gn))
            return fail("native D3D12 manifest: invalid gscale for " + k.name);
          k.gscale = (unsigned)sn;
        }
      }
      const auto* signature = entry.find("rs");
      if (signature) {
        if (!signature->is_string() || (signature->text != "buffer" && signature->text != "tex" && signature->text != "direct"))
          return fail("native D3D12 manifest: unsupported root signature for " + k.name);
        k.rs = signature->text;
      }
      if (!native_identifier(k.name, true) || !native_filename(filename) || filename.size() < 6 ||
          filename.substr(filename.size() - 5) != ".dxil" || (k.co != "-" && !valid_co(k.co)) ||
          !wgc::parse_digest_hex(hash, k.digest) || k.block == 0 || k.block > 1024 ||
          (k.wave != 32 && k.wave != 64) || k.block % k.wave || k.dwords > kNativeArgs)
        return fail("native D3D12 manifest: invalid kernel metadata for " + k.name);
      unsigned words = 0;
      for (size_t start = 0; start < arguments.size();) {
        const size_t end = arguments.find(',', start);
        const std::string token = arguments.substr(start, end == std::string::npos ? end : end - start);
        unsigned bytes = 4; char kind = token.empty() ? 0 : token[0];
        if (token != "p" && token != "u") {
          if (kind != 'b' || token.size() < 2) return fail("native D3D12 manifest: bad argument for " + k.name);
          uint64_t parsed = 0;
          for (size_t n = 1; n < token.size(); ++n) {
            if (token[n] < '0' || token[n] > '9' || parsed > 228)
              return fail("native D3D12 manifest: bad struct argument for " + k.name);
            parsed = parsed * 10 + token[n] - '0';
          }
          if (!parsed || parsed > 228 || parsed % 4) return fail("native D3D12 manifest: struct size must be 4..228 and DWORD aligned");
          bytes = static_cast<unsigned>(parsed);
        }
        words += bytes / 4; k.args.push_back({kind, bytes});
        if (words > kNativeArgs) return fail("native D3D12 manifest: too many argument DWORDs");
        if (end == std::string::npos) break;
        start = end + 1;
        if (start == arguments.size()) return fail("native D3D12 manifest: trailing argument comma");
      }
      if (words != k.dwords) return fail("native D3D12 manifest: argument DWORD count mismatch for " + k.name);
      k.file = directory + filename; const std::string key = k.name;
      auto same = kernels_.equal_range(key);   // one port per (name, code object)
      for (auto i = same.first; i != same.second; ++i)
        if (i->second.co == k.co) return fail("native D3D12 manifest: duplicate kernel " + key + " (" + k.co + ")");
      kernels_.emplace(key, std::move(k));
    }
    return true;
  }
  bool borrow_device(ID3D12Device* supplied, ID3D12CommandQueue* supplied_queue) {
    if (!supplied || !supplied_queue) return fail("native D3D12 borrowed factory requires a device and queue");
    NativeCom<ID3D12Device> queue_device;
    if (!check(supplied_queue->GetDevice(__uuidof(ID3D12Device), queue_device.pv()), "CommandQueue::GetDevice") ||
        !same_identity(supplied, queue_device.get(), "queue device")) return false;
    const D3D12_COMMAND_QUEUE_DESC description = supplied_queue->GetDesc();
    if (description.Type != D3D12_COMMAND_LIST_TYPE_DIRECT && description.Type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
      return fail("native D3D12 borrowed queue must be DIRECT or COMPUTE");
    if (supplied->GetNodeCount() != 1 || description.NodeMask > 1)
      return fail("native D3D12 borrowed device requires a single GPU node");
    borrowed_ = true; queue_type_ = description.Type;
    supplied->AddRef(); device_.p = supplied;
    supplied_queue->AddRef(); queue_.p = supplied_queue;
    adapter_luid_ = device_->GetAdapterLuid();

    dxgi_dll_ = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    d3d12_dll_ = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dxgi_dll_ || !d3d12_dll_) return fail("native D3D12: system DXGI/D3D12 unavailable");
    using MakeFactory = HRESULT(WINAPI*)(REFIID, void**);
    auto factory_function = reinterpret_cast<MakeFactory>(GetProcAddress(dxgi_dll_, "CreateDXGIFactory1"));
    serialize_ = reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(GetProcAddress(d3d12_dll_, "D3D12SerializeRootSignature"));
    if (!factory_function || !serialize_) return fail("native D3D12: system entry points unavailable");
    NativeCom<IDXGIFactory4> factory;
    NativeCom<IDXGIAdapter1> adapter;
    if (!check(factory_function(__uuidof(IDXGIFactory4), factory.pv()), "CreateDXGIFactory1") ||
        !check(factory->EnumAdapterByLuid(adapter_luid_, __uuidof(IDXGIAdapter1), adapter.pv()), "EnumAdapterByLuid"))
      return false;
    DXGI_ADAPTER_DESC1 details{};
    if (!check(adapter->GetDesc1(&details), "GetDesc1(borrowed adapter)")) return false;
    char name[256] = {};
    WideCharToMultiByte(CP_UTF8, 0, details.Description, -1, name, sizeof name - 1, nullptr, nullptr);
    adapter_name_ = name;
    if (!rdna4_adapter(details.VendorId, (details.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0, adapter_name_))
      return fail("native D3D12 borrowed device requires the supported Radeon RX 9070/9060 adapter");
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    if (!check(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm), "CheckFeatureSupport(shader model)") ||
        sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) return fail("native D3D12 requires shader model 6.6");
    const char* debug = std::getenv("WG_DX12_DEBUG");
    if (debug && std::string(debug) == "1") {
      // The caller had to enable debugging before creating its device. Never enable it late or change/clear its
      // shared filters, message limit, or stored messages. A caller clearing this queue mid-run invalidates evidence.
      if (!check(device_->QueryInterface(__uuidof(ID3D12InfoQueue), info_queue_.pv()), "QueryInterface(borrowed InfoQueue)"))
        return false;
      debug_cursor_ = info_queue_->GetNumStoredMessagesAllowedByRetrievalFilter();
      debug_discarded_ = info_queue_->GetNumMessagesDiscardedByMessageCountLimit();
      debug_denied_ = info_queue_->GetNumMessagesDeniedByStorageFilter();
      debug_log(false, "native D3D12 debug: borrowed InfoQueue observed without changing caller settings");
    }
    return true;
  }
  bool driver_probe() {
    auto probe = kernels_.find("native_driver_probe");
    if (probe == kernels_.end() || probe->second.co != "-" || probe->second.rs != "buffer" ||
        probe->second.block != 32 || probe->second.wave != 32 || probe->second.dwords != 1 ||
        probe->second.args.size() != 1 || probe->second.args[0].kind != 'p')
      return fail("native D3D12 borrowed device requires native_driver_probe (co=-, buffer, block/wave=32, args=p)");
    // Operands with distinct values per lane and element, expected results computed on the CPU (wmma_probe.h).
    DevPtr output = 0;
    if (!alloc(&output, wmma_probe::kBytes)) return false;
    uint8_t buffer[wmma_probe::kBytes];
    wmma_probe::fill(buffer);
    float values[wmma_probe::kResults] = {};
    void* args[] = {&output};
    bool ok = upload(output, buffer, sizeof buffer) && launch("native_driver_probe", 1, 32, args) &&
              download(values, output, sizeof values);
    // tests of the fallback (interop test, WG_TEST_DRIVER_PROBE_FAIL=1): the probe reports a mismatch
    const char* fail_test = std::getenv("WG_TEST_DRIVER_PROBE_FAIL");
    if (ok && fail_test && fail_test[0] == '1') values[7] = -1.0f;
    const int bad = ok ? wmma_probe::first_mismatch(values) : -1;
    if (bad >= 0) ok = fail("native D3D12 borrowed driver-channel probe mismatch at float " + std::to_string(bad));
    const std::string error = err_;
    free(output);
    if (!ok) err_ = error;
    return ok;
  }
  bool create_device() {
    dxgi_dll_ = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    d3d12_dll_ = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dxgi_dll_ || !d3d12_dll_) return fail("native D3D12: system DXGI/D3D12 unavailable");
    using MakeFactory = HRESULT(WINAPI*)(REFIID, void**);
    auto factory_function = reinterpret_cast<MakeFactory>(GetProcAddress(dxgi_dll_, "CreateDXGIFactory1"));
    auto create = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(d3d12_dll_, "D3D12CreateDevice"));
    serialize_ = reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(GetProcAddress(d3d12_dll_, "D3D12SerializeRootSignature"));
    if (!factory_function || !create || !serialize_) return fail("native D3D12: system entry points unavailable");
    const char* debug = std::getenv("WG_DX12_DEBUG");
    if (debug && std::string(debug) == "1") {
      auto get_debug = reinterpret_cast<PFN_D3D12_GET_DEBUG_INTERFACE>(GetProcAddress(d3d12_dll_, "D3D12GetDebugInterface"));
      NativeCom<ID3D12Debug> layer;
      if (!get_debug || !check(get_debug(__uuidof(ID3D12Debug), layer.pv()), "GetDebugInterface"))
        return fail("native D3D12: requested debug layer is unavailable");
      layer->EnableDebugLayer();
    }
    NativeCom<IDXGIFactory4> factory;
    if (!check(factory_function(__uuidof(IDXGIFactory4), factory.pv()), "CreateDXGIFactory1")) return false;
    NativeCom<IDXGIAdapter1> selected;
    for (UINT index = 0; ; ++index) {
      NativeCom<IDXGIAdapter1> adapter;
      HRESULT hr = factory->EnumAdapters1(index, adapter.put());
      if (hr == DXGI_ERROR_NOT_FOUND) break;
      if (!check(hr, "EnumAdapters1")) return false;
      DXGI_ADAPTER_DESC1 description{};
      if (!check(adapter->GetDesc1(&description), "GetDesc1")) return false;
      char name[256] = {};
      WideCharToMultiByte(CP_UTF8, 0, description.Description, -1, name, sizeof name - 1, nullptr, nullptr);
      // The driver-channel WMMA ports are qualified only for the project's RDNA4 target. Do not select an AMD iGPU.
      if (rdna4_adapter(description.VendorId, (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0, name)) {
        selected = std::move(adapter); adapter_name_ = name; adapter_luid_ = description.AdapterLuid; break;
      }
    }
    if (!selected) return fail("native D3D12: no supported Radeon RX 9070/9060 adapter");
    if (!check(create(selected.get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), device_.pv()), "D3D12CreateDevice")) return false;
    if (debug && std::string(debug) == "1") {
      if (!check(device_->QueryInterface(__uuidof(ID3D12InfoQueue), info_queue_.pv()), "QueryInterface(InfoQueue)") ||
          !check(info_queue_->SetMessageCountLimit(65536), "InfoQueue::SetMessageCountLimit")) return false;
      info_queue_->ClearStorageFilter(); info_queue_->ClearRetrievalFilter();
      debug_log(false, "native D3D12 debug: InfoQueue enabled; ERROR/CORRUPTION fail synchronization validation");
    }
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    if (!check(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm), "CheckFeatureSupport(shader model)") ||
        sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) return fail("native D3D12 requires shader model 6.6");
    return true;
  }
  // The typeless member of a format's family (the formats one resource can be viewed as), or the format itself.
  static DXGI_FORMAT format_family(DXGI_FORMAT f) {
    const unsigned v = static_cast<unsigned>(f);
    struct Range { unsigned lo, hi; DXGI_FORMAT family; };
    static const Range ranges[] = {
        {1, 4, DXGI_FORMAT_R32G32B32A32_TYPELESS}, {5, 8, DXGI_FORMAT_R32G32B32_TYPELESS},
        {9, 14, DXGI_FORMAT_R16G16B16A16_TYPELESS}, {15, 18, DXGI_FORMAT_R32G32_TYPELESS},
        {19, 22, DXGI_FORMAT_R32G8X24_TYPELESS}, {23, 25, DXGI_FORMAT_R10G10B10A2_TYPELESS},
        {27, 32, DXGI_FORMAT_R8G8B8A8_TYPELESS}, {33, 38, DXGI_FORMAT_R16G16_TYPELESS},
        {39, 43, DXGI_FORMAT_R32_TYPELESS}, {44, 47, DXGI_FORMAT_R24G8_TYPELESS},
        {48, 52, DXGI_FORMAT_R8G8_TYPELESS}, {53, 59, DXGI_FORMAT_R16_TYPELESS},
        {60, 65, DXGI_FORMAT_R8_TYPELESS}};
    for (const Range& r : ranges) if (v >= r.lo && v <= r.hi) return r.family;
    if (f == DXGI_FORMAT_B8G8R8A8_UNORM || f == DXGI_FORMAT_B8G8R8A8_TYPELESS || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB)
      return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    if (f == DXGI_FORMAT_B8G8R8X8_UNORM || f == DXGI_FORMAT_B8G8R8X8_TYPELESS || f == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB)
      return DXGI_FORMAT_B8G8R8X8_TYPELESS;
    return f;
  }
  static bool srgb_format(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ||
           f == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
  }
  // A typed view whose loads/stores are float: not typeless, not integer, not a depth-stencil-only format.
  static bool float_view(DXGI_FORMAT f) {
    switch (f) {
      case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32_FLOAT: case DXGI_FORMAT_R16G16B16A16_FLOAT:
      case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R32G32_FLOAT:
      case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R11G11B10_FLOAT:
      case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R8G8B8A8_SNORM:
      case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16G16_SNORM:
      case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: case DXGI_FORMAT_R8G8_UNORM:
      case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16_SNORM:
      case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_B8G8R8A8_UNORM:
      case DXGI_FORMAT_B8G8R8X8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
      case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
        return true;
      default:
        return false;
    }
  }
  bool validate_direct_texture(ID3D12Resource* resource, DXGI_FORMAT format, unsigned index,
                               const NativeDirectParams& params) {
    static const char* names[] = {"color", "depth", "motion", "exposure", "output"};
    const std::string label = std::string("native D3D12 direct ") + names[index];
    // Views read (or written) as float through typed loads and stores: the same operations as the Converter's passes
    // on the same view formats (any typed float/UNORM/SNORM view of the resource's format family; the caller picks the
    // view format as the Converter does: srv_format / uav_format). sRGB views are never stores here.
    // "input refused: " (0.6.0 follow-up): what no later frame changes (the texture's shape, format family, flags, the
    // device's support for its format); the backend answers WG_ERROR_INPUT for it (wronggpu_upscaler.h)
    auto refuse = [this](const std::string& why) { return fail("input refused: " + why); };
    if (!float_view(format) || (index == 4 && srgb_format(format)))
      return refuse(label + ": unsupported typed view format " + std::to_string(static_cast<unsigned>(format)));
    if (!resource) {
      if (index == 3 && params.exposure_mode != 1) return true;
      return fail(label + ": required resource is null");
    }
    NativeCom<ID3D12Device> owner;
    if (!check(resource->GetDevice(__uuidof(ID3D12Device), owner.pv()), "DirectTexture::GetDevice") ||
        !same_identity(device_.get(), owner.get(), names[index])) return false;
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    // any number of mips: mip 0 is read (or written) through the views below (interop/texture_shape.h)
    if (!wgi::texture_shape_ok(desc)) return refuse(label + ": " + wgi::texture_shape_rule());
    // The view must belong to the resource's format family (typeless resources, depth formats read through their
    // shader view, as the Converter's srv_format).
    if (desc.Format != format && format_family(desc.Format) != format_family(format))
      return refuse(label + ": view format " + std::to_string(static_cast<unsigned>(format)) +
                  " is not of the resource's format family " + std::to_string(static_cast<unsigned>(desc.Format)));
    if (index < 3 && (static_cast<uint64_t>(params.rx) + params.rw > desc.Width ||
                      static_cast<uint64_t>(params.ry) + params.rh > desc.Height))
      return fail(label + ": active input region exceeds the texture");
    if (index < 4 && (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
      return refuse(label + ": resource denies shader access");
    if (index == 4 && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
      return refuse(label + ": output requires unordered access");
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{}; support.Format = format;
    if (!check(device_->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof support),
               "CheckFeatureSupport(direct texture format)")) return false;
    if (!(support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) ||
        (index < 4 && !(support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD)) ||
        (index == 4 && !(support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)))
      return refuse(label + ": required typed texture operation is unsupported on this device");
    return true;
  }
  bool create_direct_objects() {
    if (direct_root_ && direct_heap_ && direct_constants_) return true;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kDirectSet * kDirectRing; heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (!check(device_->CreateDescriptorHeap(&heap, __uuidof(ID3D12DescriptorHeap), direct_heap_.pv()),
               "CreateDescriptorHeap(direct frame)")) return false;
    if (!buffer(256 * kDirectRing, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, direct_constants_))
      return false;
    void* mapped = nullptr; D3D12_RANGE no_read{0, 0};
    if (!check(direct_constants_->Map(0, &no_read, &mapped), "Map(direct frame constants)")) return false;
    direct_constants_map_ = static_cast<char*>(mapped);
    std::memset(direct_constants_map_, 0, 256 * kDirectRing);
    descriptor_stride_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    for (unsigned set = 0; set < kDirectRing; ++set) {
      D3D12_CPU_DESCRIPTOR_HANDLE descriptor = direct_heap_->GetCPUDescriptorHandleForHeapStart();
      descriptor.ptr += static_cast<uint64_t>(set) * kDirectSet * descriptor_stride_;
      D3D12_UNORDERED_ACCESS_VIEW_DESC channel{}; channel.Format = DXGI_FORMAT_R32_TYPELESS;
      channel.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; channel.Buffer.NumElements = 64;
      channel.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
      device_->CreateUnorderedAccessView(channel_.get(), nullptr, &channel, descriptor);
      descriptor.ptr += 6 * descriptor_stride_;
      D3D12_CONSTANT_BUFFER_VIEW_DESC constants{};
      constants.BufferLocation = direct_constants_->GetGPUVirtualAddress() + 256ull * set; constants.SizeInBytes = 256;
      device_->CreateConstantBufferView(&constants, descriptor);
      descriptor.ptr += (kGeluT5Slot - 6) * descriptor_stride_;
      gelu_t5_view(descriptor);  // the buffer signature's GELU T5 range (caller-list recording binds these sets)
    }

    // Root cost remains 64 DWORDs: two root UAVs (4), 59 argument/grid constants, one descriptor table (1).
    // b1 lives in the descriptor table; no global argument-ABI or buffer-root-signature change is needed.
    D3D12_DESCRIPTOR_RANGE ranges[4] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[0].NumDescriptors = 1;
    ranges[0].RegisterSpace = 0x7FFF0ADE; ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].NumDescriptors = 4;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 3; ranges[2].OffsetInDescriptorsFromTableStart = 5;
    ranges[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV; ranges[3].NumDescriptors = 1;
    ranges[3].BaseShaderRegister = 1; ranges[3].OffsetInDescriptorsFromTableStart = 6;
    D3D12_ROOT_PARAMETER parameters[4] = {};
    parameters[0].ParameterType = parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[1].Descriptor.ShaderRegister = 1;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[2].Constants.Num32BitValues = kNativeConstants + kNativeArgs;
    parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[3].DescriptorTable.NumDescriptorRanges = 4; parameters[3].DescriptorTable.pDescriptorRanges = ranges;
    D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters = 4; signature.pParameters = parameters;
    NativeCom<ID3DBlob> blob, errors;
    if (!check(serialize_(&signature, D3D_ROOT_SIGNATURE_VERSION_1, blob.put(), errors.put()),
               "D3D12SerializeRootSignature(direct frame)")) return false;
    return check(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                              __uuidof(ID3D12RootSignature), direct_root_.pv()),
                 "CreateRootSignature(direct frame)");
  }
  bool create_objects() {
    D3D12_COMMAND_QUEUE_DESC queue{}; queue.Type = queue_type_;
    if (!queue_ && !check(device_->CreateCommandQueue(&queue, __uuidof(ID3D12CommandQueue), queue_.pv()), "CreateCommandQueue"))
      return false;
    if (!check(queue_->GetTimestampFrequency(&timestamp_frequency_), "GetTimestampFrequency") || !timestamp_frequency_)
      return fail("native D3D12: queue timestamp frequency unavailable");
    if (!check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence_.pv()), "CreateFence")) return false;
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_) return fail("native D3D12: CreateEvent failed");
    for (auto& slot : slots_)
      if (!check(device_->CreateCommandAllocator(queue.Type, __uuidof(ID3D12CommandAllocator), slot.allocator.pv()), "CreateCommandAllocator"))
        return false;
    if (!check(device_->CreateCommandList(0, queue.Type, slots_[0].allocator.get(), nullptr,
                                          __uuidof(ID3D12GraphicsCommandList), list_.pv()), "CreateCommandList") ||
        !check(list_->Close(), "CommandList::Close(initial)")) return false;
    if (!arena_buffer() ||
        !buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true, channel_)) return false;
    static_assert(wg::kGeluLutBase == kNativeAlign, "the GELU table is the first arena range after the null header");
    free_[kNativeAlign + wg::kGeluLutBytes] = arena_bytes_ - kNativeAlign - wg::kGeluLutBytes;
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kGeluT5Slot + 1; heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (!check(device_->CreateDescriptorHeap(&heap, __uuidof(ID3D12DescriptorHeap), heap_.pv()), "CreateDescriptorHeap")) return false;
    if (!buffer(wg::kGeluT5Bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, false, gelu_t5_)) return false;
    {
      D3D12_CPU_DESCRIPTOR_HANDLE h = heap_->GetCPUDescriptorHandleForHeapStart();
      h.ptr += kGeluT5Slot * device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
      gelu_t5_view(h);
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{}; uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; uav.Buffer.NumElements = 64; uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device_->CreateUnorderedAccessView(channel_.get(), nullptr, &uav, heap_->GetCPUDescriptorHandleForHeapStart());
    D3D12_DESCRIPTOR_RANGE range{}; range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range.NumDescriptors = 1; range.RegisterSpace = 0x7FFF0ADE;
    D3D12_ROOT_PARAMETER parameters[4] = {};
    parameters[0].ParameterType = parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[1].Descriptor.ShaderRegister = 1;
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[2].Constants.Num32BitValues = kNativeConstants + kNativeArgs;
    parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    D3D12_DESCRIPTOR_RANGE ranges[2] = {range, {}};
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].NumDescriptors = 1; ranges[1].RegisterSpace = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = kGeluT5Slot;  // GELU T5 table (t0, space 1)
    parameters[3].DescriptorTable.NumDescriptorRanges = 2; parameters[3].DescriptorTable.pDescriptorRanges = ranges;
    D3D12_ROOT_SIGNATURE_DESC signature{}; signature.NumParameters = 4; signature.pParameters = parameters;
    NativeCom<ID3DBlob> blob, errors;
    if (!check(serialize_(&signature, D3D_ROOT_SIGNATURE_VERSION_1, blob.put(), errors.put()), "D3D12SerializeRootSignature"))
      return false;
    return check(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                              __uuidof(ID3D12RootSignature), root_.pv()), "CreateRootSignature");
  }
  bool create_pso(NativeKernel& k) {
    if (k.rs == "direct" && !create_direct_objects()) return false;
    // The file on disk, else the copy built into the library under its own name (embedded.h).
    std::vector<char> code;
    if (!wg_read_file(k.file, &code)) return fail("native D3D12 shader missing: " + k.file);
    if (code.size() < 32 || code.size() > 64 * 1024 * 1024)
      return fail("native D3D12 shader: invalid file size for " + k.name);
    if (
        std::memcmp(code.data(), "DXBC", 4) != 0 || wgc::Sha256::hash(code.data(), code.size()) != k.digest)
      return fail("native D3D12 shader integrity mismatch: " + k.name);
    D3D12_COMPUTE_PIPELINE_STATE_DESC p{}; p.pRootSignature = k.rs == "direct" ? direct_root_.get() : root_.get();
    p.CS.pShaderBytecode = code.data(); p.CS.BytecodeLength = code.size();
    return check(device_->CreateComputePipelineState(&p, __uuidof(ID3D12PipelineState), k.pso.pv()),
                 ("CreateComputePipelineState(" + k.name + ")").c_str());
  }

  NativeCom<ID3D12Device> device_;
  NativeCom<ID3D12InfoQueue> info_queue_;
  NativeCom<ID3D12CommandQueue> queue_;
  NativeCom<ID3D12GraphicsCommandList> list_;
  NativeCom<ID3D12Fence> fence_;
  NativeCom<ID3D12Resource> arena_, channel_, query_readback_;
  bool teardown_registered_ = false;  // clean/fe-minimize: begin_teardown registered this device's teardown
  NativeCom<ID3D12DescriptorHeap> heap_;
  NativeCom<ID3D12Resource> gelu_t5_;  // GELU T5 table (gelu_t5_table.h), read-only
  NativeCom<ID3D12RootSignature> root_;
  NativeCom<ID3D12Resource> direct_constants_;
  NativeCom<ID3D12DescriptorHeap> direct_heap_;
  NativeCom<ID3D12RootSignature> direct_root_;
  static constexpr unsigned kDirectSet = 8;  // channel UAV, 4 SRVs, output UAV, CBV, GELU T5 SRV
  static constexpr unsigned kGeluT5Slot = 7;  // the GELU T5 SRV in heap_ and in every direct set
  std::array<NativeCom<ID3D12Resource>, 5> direct_textures_[kDirectRing];
  uint64_t direct_serial_[kDirectRing] = {};  // external frame serial that last used each set (0: none)
  unsigned direct_slot_ = 0;
  UINT descriptor_stride_ = 0;
  char* direct_constants_map_ = nullptr;
  // recording into the caller's command list
  ID3D12GraphicsCommandList* ext_list_ = nullptr;
  bool external_ = false;
  uint64_t ext_recorded_ = 0;
  NativeCom<ID3D12Resource> marker_;
  volatile uint32_t* marker_map_ = nullptr;
  NativeCom<ID3D12QueryHeap> query_heap_;
  PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize_ = nullptr;
  HMODULE dxgi_dll_ = nullptr, d3d12_dll_ = nullptr;
  HANDLE event_ = nullptr;
  LUID adapter_luid_{};
  std::string adapter_name_;
  std::string debug_failure_;
  std::ofstream profile_output_;
  std::string profile_session_, profile_failure_;
  bool profile_enabled_ = false;
  bool borrowed_ = false, native_binding_dirty_ = false, inside_pass_ = false;
  bool direct_bound_ = false, direct_root_bound_ = false;
  D3D12_COMMAND_LIST_TYPE queue_type_ = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  uint64_t debug_cursor_ = 0, debug_denied_ = 0;
  uint64_t profile_sequence_ = 0;
  uint64_t debug_messages_ = 0, debug_errors_ = 0, debug_warnings_ = 0, debug_discarded_ = 0;
  Slot slots_[kNativeSlots];
  unsigned current_slot_ = 0, dispatches_ = 0, batch_ = 0;
  uint64_t high_ = 0;
  uint64_t arena_bytes_ = 0, in_use_ = 0, next_fence_ = 0, last_submitted_ = 0, timestamp_frequency_ = 0;
  bool list_open_ = false, need_uav_barrier_ = false, queue_unfenced_ = false, unordered_next_ = false;
  D3D12_RESOURCE_STATES arena_state_ = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  std::map<uint64_t, uint64_t> free_;
  std::map<uint64_t, NativeAllocation> used_;
  std::map<uintptr_t, Host> hosts_;
  std::multimap<std::string, NativeKernel> kernels_;   // by name; several ports of one name differ in "co"
  std::set<std::string> untagged_cos_;
  std::map<std::string, std::string> tagged_cos_;
  std::vector<NativeEvent> events_;
  std::vector<uint64_t> timestamps_;
  std::vector<bool> cached_;
};

}  // namespace

std::unique_ptr<Device> make_d3d12_device(const std::string& build_dir, std::string* error) {
  std::unique_ptr<NativeD3D12Device> device(new NativeD3D12Device());
  if (!device->init(build_dir)) { if (error) *error = device->err(); return nullptr; }
  return std::unique_ptr<Device>(device.release());
}

NativeD3D12Ext* d3d12_ext(Device* device) { return dynamic_cast<NativeD3D12Ext*>(device); }

std::unique_ptr<Device> make_d3d12_device_on(ID3D12Device* supplied, ID3D12CommandQueue* queue,
                                          const std::string& build_dir, std::string* error, uint64_t arena_bytes) {
  if (!supplied || !queue) {
    if (error) *error = "native D3D12 borrowed factory requires a device and queue";
    return nullptr;
  }
  std::unique_ptr<NativeD3D12Device> device(new NativeD3D12Device());
  if (!device->init(build_dir, supplied, queue, arena_bytes)) { if (error) *error = device->err(); return nullptr; }
  if (error) error->clear();
  return std::unique_ptr<Device>(device.release());
}

}  // namespace wgrt
#else
namespace wgrt {
std::unique_ptr<Device> make_d3d12_device(const std::string&, std::string* error) {
  if (error) *error = "native D3D12 is supported only on Windows";
  return nullptr;
}
}  // namespace wgrt
#endif
