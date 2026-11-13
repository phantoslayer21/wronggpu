// SPDX-License-Identifier: MIT
// Hybrid D3D12 device (runtime backend 2; see device.h and clean/dx12/README.md).
//
// Memory: one D3D12 committed buffer (default heap, shared, unordered access) imported into HIP, so the same memory has
// a HIP address. alloc() sub-allocates from it (offset 0 is never handed out: a kernel argument 0 means null), and the
// DevPtr returned is the HIP address: every kernel without a D3D12 port, and every copy, upload and memset, keeps running
// on the HIP stream unchanged.
// Kernels: <build_dir>/dx12/manifest.json (clean/dx12/build_dx12.py) lists the D3D12 ports by the name the runtime
// launches ("exposure_auto_diet", "bneck/swin_bneck6_nc") and the HIP code object each port reproduces ("co"). A port is
// used only while the runtime has that code object loaded (untagged, or under the name's tag): another build variant of
// the same kernel name (post_w64.co instead of post_w64__orig.co under "w64", a swin.co variant, ...) runs on HIP. A launch of a listed and enabled kernel (WG_DX12_KERNELS) is
// recorded as a Dispatch on a D3D12 compute queue with its pointer arguments turned into arena byte offsets; any pointer
// outside the arena sends that launch to HIP instead. One root signature for all kernels: u0 and u1 = the arena (root
// UAVs; u1 is the shader's globallycoherent view), b0 = 2 + up to 57 constants (group base, grid, then the arguments),
// a descriptor table with the driver channel's UAV (u0, space 0x7FFF0ADE) and the MLP kernels' GELU T5 table (t0, space 1;
// hlsl/wg_gelu_t5.hlsli: a D3D12-only read-only buffer, filled at creation).
// Ordering: one shared fence, imported into HIP as an external semaphore. Consecutive D3D12 dispatches go into one command
// list with a UAV barrier between them. HIP -> D3D12: the HIP stream signals the next fence value and the queue waits for
// it. D3D12 -> HIP (any HIP operation, sync, event): the list is executed, the queue signals and the HIP stream waits.
#if defined(_WIN32)

#include "device.h"
#include "gelu_lut_table.h"
#include "gelu_t5_table.h"

#include "interop/d3d12_include.h"
#include <dxgi1_4.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>

#include "../converter/src/json.h"

namespace wgrt {

namespace {

template <class T>
struct Com {  // minimal owning COM pointer
  T* p = nullptr;
  Com() = default;
  Com(const Com&) = delete;
  Com& operator=(const Com&) = delete;
  Com(Com&& o) noexcept : p(o.p) { o.p = nullptr; }
  Com& operator=(Com&& o) noexcept {
    if (this != &o) reset(), p = o.p, o.p = nullptr;
    return *this;
  }
  ~Com() { reset(); }
  void reset() {
    if (p) p->Release();
    p = nullptr;
  }
  T** put() {
    reset();
    return &p;
  }
  void** pv() { return (void**)put(); }
  T* operator->() const { return p; }
  T* get() const { return p; }
  explicit operator bool() const { return p != nullptr; }
};

std::string hr_text(HRESULT hr) {
  char b[32];
  std::snprintf(b, sizeof b, "0x%08lx", (unsigned long)hr);
  return b;
}

std::string env(const char* name, const char* def = "") {
  const char* v = std::getenv(name);
  return v ? v : def;
}

constexpr UINT kRuntimeConstants = 2;  // wg_group_base_, wg_grid_
constexpr UINT kMaxArgDwords = 57;     // 64 DWORDs - 2 x 2 (root UAVs) - 1 (table) - 2 (runtime constants)
constexpr UINT kMaxGroups = 65535;
constexpr uint64_t kAlign = 256;
constexpr uint64_t kOrderMark = 128;  // arena byte offset of the HIP ordering memset (order_mark, in the null header)
constexpr int kAllocators = 8;

struct Arg {
  char kind;      // 'p' pointer, 'u' 32-bit scalar, 'b' bytes
  unsigned size;  // bytes for 'b'
};

struct Kernel {
  std::string name, file, co;  // co: file name of the HIP code object this port reproduces
  unsigned block = 0, wave = 32, dwords = 0;
  unsigned group = 0;  // lanes per D3D12 workgroup of a split port (manifest "group"; 0 = block)
  unsigned gscale = 1;  // D3D12 workgroups per launched workgroup (manifest "gscale", default block / group)
  std::vector<Arg> args;
  Com<ID3D12PipelineState> pso;
  bool fallback_noted = false, variant_noted = false;
  unsigned co_gen = ~0u;  // module_gen_ when co_ok was decided
  bool co_ok = false;
};

class Dx12Device : public Device {
 public:
  ~Dx12Device() override {
    if (hip_) {
      to_hip();
      hip_->sync();
    }
    if (queue_ && fence_) wait_cpu(signal_queue());
    if (hip_) {
      if (arena_mem_) hip_->release_memory(arena_mem_, arena_hip_);
      if (sem_) hip_->release_fence(sem_);
    }
    if (fence_handle_) CloseHandle(fence_handle_);
    if (arena_handle_) CloseHandle(arena_handle_);
    if (event_) CloseHandle(event_);
    kernels_.clear();
    for (auto& a : allocs_) a.alloc.reset();
    list_.reset();
    heap_.reset();
    ags_buf_.reset();
    gelu_t5_.reset();
    arena_.reset();
    root_.reset();
    fence_.reset();
    queue_.reset();
    device_.reset();
    hip_.reset();  // after the imports are released
  }

  bool init(const std::string& rocm_dir, const std::string& build_dir) {
    std::string e;
    batch_ = (unsigned)std::max(0L, std::atol(env("WG_DX12_BATCH", "0").c_str()));
    hip_ = make_hip_device(rocm_dir, &e);
    if (!hip_) return fail(e);
    hip_->set_log_sink([this](bool w, const std::string& t) { log(w, t); });
    if (!create_device()) return false;
    if (!read_manifest(build_dir + "/dx12/manifest.json")) return false;
    if (!create_objects() || !gelu_t5_upload()) return false;
    return true;
  }

  std::string name() const override { return hip_->name() + " (hybrid D3D12)"; }
  std::string note() const override {
    std::string n = hip_->note();
    std::string k;
    for (auto& kv : kernels_) k += (k.empty() ? "" : ",") + kv.first;
    n += (n.empty() ? "" : "; ") + std::string("D3D12: ") + adapter_name_ + ", arena " +
         std::to_string(arena_bytes_ >> 20) + " MiB, kernels on D3D12: " + (k.empty() ? "none" : k);
    return n;
  }
  bool adapter_luid(uint8_t luid[8]) const override { return hip_->adapter_luid(luid); }

  // ---- memory: sub-allocation of the arena -------------------------------------------------------------------------
  bool alloc(DevPtr* p, size_t bytes) override {
    if (!arena_bytes_) return fwd(hip_->alloc(p, bytes));  // WG_DX12_ARENA_MB=0 (diagnostics): plain HIP memory
    const uint64_t need = std::max<uint64_t>(kAlign, (bytes + kAlign - 1) / kAlign * kAlign);
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second < need) continue;
      const uint64_t off = it->first, rest = it->second - need;
      free_.erase(it);
      if (rest) free_[off + need] = rest;
      used_[off] = need;
      in_use_ += need;
      *p = arena_hip_ + off;
      return true;
    }
    uint64_t largest = 0, total = 0;
    for (auto& kv : free_) largest = std::max(largest, kv.second), total += kv.second;
    return fail("the D3D12 arena is exhausted: " + std::to_string(bytes) + " bytes asked, " + std::to_string(total >> 20) +
                " MiB free (largest block " + std::to_string(largest >> 20) + " MiB) of " +
                std::to_string(arena_bytes_ >> 20) + " MiB; set WG_DX12_ARENA_MB larger (at most 4095)");
  }
  void free(DevPtr p) override {
    if (!p) return;
    if (!arena_bytes_) return hip_->free(p);
    auto u = used_.find(p - arena_hip_);
    if (p < arena_hip_ || u == used_.end()) return;
    uint64_t off = u->first, size = u->second;
    in_use_ -= size;
    used_.erase(u);
    auto next = free_.lower_bound(off);
    if (next != free_.end() && off + size == next->first) size += next->second, next = free_.erase(next);
    if (next != free_.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == off) {
        prev->second += size;
        return;
      }
    }
    free_[off] = size;
  }

  // ---- HIP-stream operations: pending D3D12 work first -------------------------------------------------------------
  bool upload(DevPtr dst, const void* src, size_t bytes) override { return to_hip() && fwd(hip_->upload(dst, src, bytes)); }
  bool download(void* dst, DevPtr src, size_t bytes) override { return to_hip() && fwd(hip_->download(dst, src, bytes)); }
  bool copy(DevPtr dst, DevPtr src, size_t bytes) override { return to_hip() && fwd(hip_->copy(dst, src, bytes)); }
  bool memset(DevPtr dst, int value, size_t bytes) override { return to_hip() && fwd(hip_->memset(dst, value, bytes)); }
  void* alloc_host(size_t bytes) override { return hip_->alloc_host(bytes); }
  void free_host(void* p) override { hip_->free_host(p); }
  bool download_async(void* dst, DevPtr src, size_t bytes) override {
    return to_hip() && fwd(hip_->download_async(dst, src, bytes));
  }
  bool load_module(const std::string& path) override {
    if (!fwd(hip_->load_module(path))) return false;
    untagged_cos_.insert(base_name(path));
    ++module_gen_;
    return true;
  }
  bool load_module_tagged(const std::string& path, const std::string& tag) override {
    if (!fwd(hip_->load_module_tagged(path, tag))) return false;
    tagged_cos_[tag] = base_name(path);
    ++module_gen_;
    return true;
  }
  bool sync() override { return to_hip() && fwd(hip_->sync()); }
  bool record(int event) override { return to_hip() && fwd(hip_->record(event)); }
  bool elapsed_ms(int a, int b, float* ms) override { return fwd(hip_->elapsed_ms(a, b, ms)); }
  bool reserve_events(int n) override { return fwd(hip_->reserve_events(n)); }
  int event_done(int event) override { return hip_->event_done(event); }
  bool elapsed_done_ms(int a, int b, float* ms) override { return fwd(hip_->elapsed_done_ms(a, b, ms)); }
  bool import_memory(void* h, uint64_t alloc_bytes, uint64_t bytes, DevPtr* ptr, void** mem) override {
    return fwd(hip_->import_memory(h, alloc_bytes, bytes, ptr, mem));
  }
  void release_memory(void* mem, DevPtr ptr) override { hip_->release_memory(mem, ptr); }
  bool import_fence(void* h, void** sem) override { return fwd(hip_->import_fence(h, sem)); }
  void release_fence(void* sem) override { hip_->release_fence(sem); }
  bool wait_fence(void* sem, uint64_t v) override { return to_hip() && fwd(hip_->wait_fence(sem, v)); }
  bool signal_fence(void* sem, uint64_t v) override { return to_hip() && fwd(hip_->signal_fence(sem, v)); }
  bool wait_value32(DevPtr p, uint32_t v) override { return to_hip() && fwd(hip_->wait_value32(p, v)); }
  bool has_wait_value32() const override { return hip_->has_wait_value32(); }
  bool write_value32(DevPtr p, uint32_t v) override { return to_hip() && fwd(hip_->write_value32(p, v)); }
  bool set_stream_priority(bool high) override { return fwd(hip_->set_stream_priority(high)); }
  void begin_teardown() override { hip_->begin_teardown(); }

  // ---- launches ----------------------------------------------------------------------------------------------------
  bool launch(const std::string& kernel, unsigned grid, unsigned block, void** args) override {
    auto it = kernels_.find(kernel);
    uint32_t c[kRuntimeConstants + kMaxArgDwords];
    if (it == kernels_.end() || !arena_ || !variant_loaded(it->second) || !pack_args(it->second, args, c + kRuntimeConstants)) {
      return to_hip() && fwd(hip_->launch(kernel, grid, block, args));
    }
    Kernel& k = it->second;
    if (block != k.block)
      return fail("D3D12 kernel " + kernel + " is built for " + std::to_string(k.block) + " lanes, launched with " +
                  std::to_string(block));
    if (!k.pso && !create_pso(k)) return false;
    if (!to_dx()) return false;
    trace_launch(kernel, grid, block, "dx12");
    if (need_barrier_) {  // a D3D12 dispatch since the last switch from HIP (in this list or an executed one)
      D3D12_RESOURCE_BARRIER b = {};
      b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      b.UAV.pResource = nullptr;  // every UAV access before / after (the arena)
      list_->ResourceBarrier(1, &b);
    }
    list_->SetPipelineState(k.pso.get());
    const unsigned groups = grid * k.gscale;  // D3D12 workgroups (split port)
    c[1] = groups;
    for (unsigned base = 0; base < groups; base += kMaxGroups) {
      c[0] = base;
      list_->SetComputeRoot32BitConstants(2, kRuntimeConstants + k.dwords, c, 0);
      list_->Dispatch(std::min(groups - base, kMaxGroups), 1, 1);
    }
    ++dispatches_;
    need_barrier_ = true;
    if (batch_ && dispatches_ >= batch_) return submit();  // start the GPU on it now (the queue stays D3D12's)
    return true;
  }
  bool launch_timed(const std::string& kernel, unsigned grid, unsigned block, void** args, int start_event,
                    int stop_event) override {
    auto it = kernels_.find(kernel);
    if (it == kernels_.end() || !arena_ || !variant_loaded(it->second))
      return to_hip() && fwd(hip_->launch_timed(kernel, grid, block, args, start_event, stop_event));
    return Device::launch_timed(kernel, grid, block, args, start_event, stop_event);
  }

 private:
  bool fail(const std::string& m) {
    err_ = m;
    return false;
  }
  bool fwd(bool ok) {
    if (!ok) err_ = hip_->err();
    return ok;
  }
  bool check(HRESULT hr, const char* what) {
    if (SUCCEEDED(hr)) return true;
    std::string m = std::string(what) + " failed: " + hr_text(hr);
    if (device_ && FAILED(device_->GetDeviceRemovedReason()))
      m += " (device removed: " + hr_text(device_->GetDeviceRemovedReason()) + ")";
    return fail(m);
  }

  static std::string base_name(const std::string& path) { return path.substr(path.find_last_of("/\\") + 1); }
  // Is the HIP code object this port reproduces the one the runtime loaded for the kernel's name?
  bool variant_loaded(Kernel& k) {
    if (k.co_gen != module_gen_) {
      k.co_gen = module_gen_;
      const size_t slash = k.name.find('/');
      std::string loaded;
      if (k.co == "-") {  // a D3D12-only kernel (the device self-test's): no HIP code object to match
        k.co_ok = true;
      } else if (slash == std::string::npos) {
        k.co_ok = !k.co.empty() && untagged_cos_.count(k.co);
        for (const std::string& m : untagged_cos_) loaded += (loaded.empty() ? "" : ", ") + m;
      } else {
        auto t = tagged_cos_.find(k.name.substr(0, slash));
        k.co_ok = !k.co.empty() && t != tagged_cos_.end() && t->second == k.co;
        if (t != tagged_cos_.end()) loaded = t->second;
      }
      if (!k.co_ok && !k.variant_noted) {
        log(false, "D3D12: " + k.name + " ports " + (k.co.empty() ? std::string("(no code object in the manifest)") : k.co) +
                       ", but the runtime loaded " + (loaded.empty() ? std::string("nothing for it") : loaded) +
                       "; it runs on HIP");
        k.variant_noted = true;
      }
    }
    return k.co_ok;
  }

  // Pointer and scalar arguments -> root constants. False: a pointer outside the arena (that launch runs on HIP).
  bool pack_args(Kernel& k, void** args, uint32_t* out) {
    unsigned d = 0;
    for (size_t i = 0; i < k.args.size(); ++i) {
      const Arg& a = k.args[i];
      if (a.kind == 'p') {
        DevPtr p = 0;
        std::memcpy(&p, args[i], sizeof p);
        if (p && (p < arena_hip_ + kAlign || p >= arena_hip_ + arena_bytes_)) {
          if (!k.fallback_noted)
            log(false, "D3D12: " + k.name + " has a pointer argument outside the arena; it runs on HIP"),
                k.fallback_noted = true;
          return false;
        }
        out[d++] = p ? (uint32_t)(p - arena_hip_) : 0u;
      } else if (a.kind == 'u') {
        std::memcpy(&out[d++], args[i], 4);
      } else {
        std::memcpy(&out[d], args[i], a.size);
        d += a.size / 4;
      }
    }
    return true;
  }

  // The HIP runtime orders an external-semaphore signal / wait only with the stream's compute work, not with its DMA
  // copies (measured 2026-10-15 with the exposure stress test: a D3D12 kernel read a pinned upload before it had
  // landed, and a download right after D3D12 work, even after hipStreamSynchronize, returned the old bytes). So a tiny
  // HIP memset (a compute blit, which HIP orders after its own copies, and its own copies after it) goes before every
  // signal and after every wait. It writes 4 bytes of the arena's null header (offsets 0..255 are never allocated).
  bool order_mark() { return fwd(hip_->memset(arena_hip_ + kOrderMark, 0, 4)); }
  // HIP -> D3D12: the queue waits for everything the HIP stream has been given so far.
  bool to_dx() {
    if (on_dx_) return true;
    const uint64_t v = ++fence_value_;
    if (!order_mark() || !hip_->signal_fence(sem_, v)) return fwd(false);
    if (!check(queue_->Wait(fence_.get(), v), "ID3D12CommandQueue::Wait")) return false;
    if (!open_list()) return false;
    on_dx_ = true;
    return true;
  }
  // D3D12 -> HIP: the recorded dispatches are executed and the HIP stream waits for them.
  bool to_hip() {
    if (!on_dx_) return true;
    on_dx_ = false;
    need_barrier_ = false;
    const uint64_t v = execute();
    return v && fwd(hip_->wait_fence(sem_, v)) && order_mark();
  }
  // Executes the open list; returns the fence value the queue signals after it (0: failed).
  uint64_t execute() {
    if (!check(list_->Close(), "ID3D12GraphicsCommandList::Close")) return 0;
    ID3D12CommandList* l = list_.get();
    queue_->ExecuteCommandLists(1, &l);
    const uint64_t v = signal_queue();
    if (v) allocs_[cur_alloc_].value = v;
    dispatches_ = 0;
    return v;
  }
  // Eager submission (WG_DX12_BATCH): the dispatches recorded so far go to the GPU now and a new list is opened; the
  // next dispatch still gets its UAV barrier (need_barrier_), and HIP still waits only at the next to_hip().
  bool submit() { return execute() && open_list(); }
  uint64_t signal_queue() {
    const uint64_t v = ++fence_value_;
    return check(queue_->Signal(fence_.get(), v), "ID3D12CommandQueue::Signal") ? v : 0;
  }
  bool wait_cpu(uint64_t v) {
    if (!v || fence_->GetCompletedValue() >= v) return true;
    if (!check(fence_->SetEventOnCompletion(v, event_), "ID3D12Fence::SetEventOnCompletion")) return false;
    WaitForSingleObject(event_, INFINITE);
    return true;
  }
  bool open_list() {
    cur_alloc_ = (cur_alloc_ + 1) % kAllocators;
    Alloc& a = allocs_[cur_alloc_];
    if (!wait_cpu(a.value)) return false;  // the GPU is done with that allocator's previous list
    if (!check(a.alloc->Reset(), "ID3D12CommandAllocator::Reset")) return false;
    if (!check(list_->Reset(a.alloc.get(), nullptr), "ID3D12GraphicsCommandList::Reset")) return false;
    ID3D12DescriptorHeap* h = heap_.get();
    list_->SetDescriptorHeaps(1, &h);
    list_->SetComputeRootSignature(root_.get());
    list_->SetComputeRootUnorderedAccessView(0, arena_->GetGPUVirtualAddress());
    list_->SetComputeRootUnorderedAccessView(1, arena_->GetGPUVirtualAddress());
    list_->SetComputeRootDescriptorTable(3, heap_->GetGPUDescriptorHandleForHeapStart());
    dispatches_ = 0;
    return true;
  }

  bool create_device() {
    HMODULE dxgi = LoadLibraryW(L"dxgi.dll"), d3d12 = LoadLibraryW(L"d3d12.dll");
    if (!dxgi || !d3d12) return fail("D3D12 backend: dxgi.dll / d3d12.dll not found");
    typedef HRESULT(WINAPI * PfnFactory)(REFIID, void**);
    auto make_factory = (PfnFactory)(void*)GetProcAddress(dxgi, "CreateDXGIFactory1");
    auto create = (PFN_D3D12_CREATE_DEVICE)(void*)GetProcAddress(d3d12, "D3D12CreateDevice");
    serialize_ = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)(void*)GetProcAddress(d3d12, "D3D12SerializeRootSignature");
    if (!make_factory || !create || !serialize_) return fail("D3D12 backend: D3D12 / DXGI entry points missing");
    if (env("WG_DX12_DEBUG") == "1") {
      auto get_debug = (PFN_D3D12_GET_DEBUG_INTERFACE)(void*)GetProcAddress(d3d12, "D3D12GetDebugInterface");
      Com<ID3D12Debug> dbg;
      if (get_debug && SUCCEEDED(get_debug(__uuidof(ID3D12Debug), dbg.pv()))) dbg->EnableDebugLayer();
    }
    Com<IDXGIFactory4> f;
    if (!check(make_factory(__uuidof(IDXGIFactory4), f.pv()), "CreateDXGIFactory1")) return false;
    Com<IDXGIAdapter1> a;
    uint8_t l[8];
    if (hip_->adapter_luid(l)) {  // the adapter of the HIP device
      LUID luid;
      std::memcpy(&luid, l, 8);
      if (FAILED(f->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), a.pv()))) a.reset();
    }
    for (UINT i = 0; !a; ++i) {  // else the first AMD hardware adapter with the HIP device's name
      Com<IDXGIAdapter1> x;
      if (f->EnumAdapters1(i, x.put()) == DXGI_ERROR_NOT_FOUND) break;
      DXGI_ADAPTER_DESC1 d;
      x->GetDesc1(&d);
      char nm[256] = {0};
      WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nm, sizeof nm - 1, nullptr, nullptr);
      if (d.VendorId == 0x1002 && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && hip_->name() == nm) a.p = x.p, x.p = nullptr;
    }
    if (!a) return fail("D3D12 backend: no D3D12 adapter matches the HIP device " + hip_->name());
    DXGI_ADAPTER_DESC1 d;
    a->GetDesc1(&d);
    char nm[256] = {0};
    WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nm, sizeof nm - 1, nullptr, nullptr);
    adapter_name_ = nm;
    if (!check(create(a.get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), device_.pv()), "D3D12CreateDevice"))
      return false;
    D3D12_FEATURE_DATA_SHADER_MODEL sm = {D3D_SHADER_MODEL_6_6};
    if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm)) ||
        sm.HighestShaderModel < D3D_SHADER_MODEL_6_6)
      return fail("D3D12 backend: the driver does not support shader model 6.6");
    return true;
  }

  bool read_manifest(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {  // no ports built: every kernel runs on HIP
      log(false, "D3D12: no " + path + " (clean/dx12/build_dx12.py); every kernel runs on HIP");
      return true;
    }
    const std::string text((std::istreambuf_iterator<char>(f)), {});
    wgc::Json j;
    std::string e;
    if (!wgc::parse_json(text, j, e)) return fail("D3D12 manifest " + path + ": " + e);
    const wgc::Json* ks = j.find("kernels");
    int64_t format = 0;
    if (!j.find("format") || !j.find("format")->as_int64(format) || format != 1 || !ks || !ks->is_array())
      return fail("D3D12 manifest " + path + ": not format 1 with a kernel list");
    const std::string dir = path.substr(0, path.find_last_of("/\\") + 1);
    // enabled kernels: WG_DX12_KERNELS = "all" (default), "none" / "", or a comma list of names as launched
    const char* sel_env = std::getenv("WG_DX12_KERNELS");
    const std::string sel = sel_env ? sel_env : "all";
    std::set<std::string> want;
    bool all = sel == "all";
    for (size_t a = 0; !all && a < sel.size();) {
      size_t b = sel.find(',', a);
      if (b == std::string::npos) b = sel.size();
      std::string s = sel.substr(a, b - a);
      if (!s.empty() && s != "none") want.insert(s);
      a = b + 1;
    }
    std::set<std::string> seen;
    for (const wgc::Json& k : ks->items) {
      Kernel x;
      int64_t block = 0, wave = 0, dwords = 0;
      const wgc::Json *n = k.find("name"), *fl = k.find("file"), *ar = k.find("args"), *co = k.find("co");
      if (!n || !fl || !ar || !k.find("block") || !k.find("block")->as_int64(block) || !k.find("wave") ||
          !k.find("wave")->as_int64(wave) || !k.find("arg_dwords") || !k.find("arg_dwords")->as_int64(dwords))
        return fail("D3D12 manifest " + path + ": incomplete kernel entry");
      x.name = n->text;
      x.file = dir + fl->text;
      if (co) x.co = co->text;  // missing (an old manifest): the port is never used
      x.block = (unsigned)block;
      if (const wgc::Json* gp = k.find("group")) {
        int64_t gn = 0;
        if (!gp->as_int64(gn) || gn <= 0 || gn > block || block % gn || gn % (wave ? wave : 32))
          return fail("D3D12 manifest " + path + ": invalid group for " + x.name);
        x.group = gn == block ? 0u : (unsigned)gn;
        x.gscale = (unsigned)(block / gn);
        if (const wgc::Json* sp = k.find("gscale")) {
          int64_t sn = 0;
          if (!sp->as_int64(sn) || sn < 1 || sn > block / gn)
            return fail("D3D12 manifest " + path + ": invalid gscale for " + x.name);
          x.gscale = (unsigned)sn;
        }
      }
      x.wave = (unsigned)wave;
      x.dwords = (unsigned)dwords;
      unsigned count = 0;
      for (size_t a = 0; a <= ar->text.size();) {
        size_t b = ar->text.find(',', a);
        if (b == std::string::npos) b = ar->text.size();
        const std::string s = ar->text.substr(a, b - a);
        a = b + 1;
        if (s == "p" || s == "u") {
          x.args.push_back({s[0], 4});
          ++count;
        } else if (s.size() > 1 && s[0] == 'b' && std::atoi(s.c_str() + 1) > 0 && std::atoi(s.c_str() + 1) % 4 == 0) {
          x.args.push_back({'b', (unsigned)std::atoi(s.c_str() + 1)});
          count += x.args.back().size / 4;
        } else {
          return fail("D3D12 manifest: kernel " + x.name + ": argument kind '" + s + "'");
        }
      }
      if (count != x.dwords || count > kMaxArgDwords)
        return fail("D3D12 manifest: kernel " + x.name + ": argument DWORDs do not add up");
      seen.insert(x.name);
      if (all || want.count(x.name)) {
        std::string nm = x.name;
        kernels_.emplace(nm, std::move(x));
      }
    }
    for (const std::string& w : want)
      if (!seen.count(w)) return fail("WG_DX12_KERNELS names " + w + ", which has no D3D12 port in " + path);
    return true;
  }

  bool create_objects() {
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (!check(device_->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), queue_.pv()), "CreateCommandQueue"))
      return false;
    for (auto& a : allocs_)
      if (!check(device_->CreateCommandAllocator(qd.Type, __uuidof(ID3D12CommandAllocator), a.alloc.pv()),
                 "CreateCommandAllocator"))
        return false;
    if (!check(device_->CreateCommandList(0, qd.Type, allocs_[0].alloc.get(), nullptr,
                                          __uuidof(ID3D12GraphicsCommandList), list_.pv()),
               "CreateCommandList") ||
        !check(list_->Close(), "ID3D12GraphicsCommandList::Close"))
      return false;
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    // the shared fence
    if (!check(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), fence_.pv()), "CreateFence") ||
        !check(device_->CreateSharedHandle(fence_.get(), nullptr, GENERIC_ALL, nullptr, &fence_handle_),
               "CreateSharedHandle (fence)"))
      return false;
    if (!hip_->import_fence(fence_handle_, &sem_)) return fwd(false);
    // the arena
    const long mb = std::atol(env("WG_DX12_ARENA_MB", "2560").c_str());
    if (mb == 0) {  // diagnostics: no arena, so every allocation is HIP memory and every kernel runs on HIP
      log(true, "D3D12: WG_DX12_ARENA_MB=0: no arena, every kernel runs on HIP (diagnostics)");
    } else if (mb < 16 || mb > 4095) {
      return fail("WG_DX12_ARENA_MB must be 0 (diagnostics) or 16 .. 4095");
    }
    arena_bytes_ = (uint64_t)mb << 20;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (arena_bytes_ && !create_arena(hp, rd)) return false;
    // the driver channel's UAV (any small raw buffer) in a shader-visible heap
    rd.Width = 256;
    if (!check(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                __uuidof(ID3D12Resource), ags_buf_.pv()),
               "CreateCommittedResource (channel)"))
      return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 2;  // channel UAV, GELU T5 SRV
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (!check(device_->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), heap_.pv()), "CreateDescriptorHeap"))
      return false;
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_R32_TYPELESS;
    ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = 64;
    ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device_->CreateUnorderedAccessView(ags_buf_.get(), nullptr, &ud, heap_->GetCPUDescriptorHandleForHeapStart());
    // the GELU T5 table (gelu_t5_table.h; filled by gelu_t5_upload) behind a structured SRV in descriptor 1
    rd.Width = wg::kGeluT5Bytes;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (!check(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                __uuidof(ID3D12Resource), gelu_t5_.pv()),
               "CreateCommittedResource (GELU T5 table)"))
      return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format = DXGI_FORMAT_UNKNOWN;
    sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Buffer.NumElements = wg::kGeluT5Entries;
    sd.Buffer.StructureByteStride = 8;
    D3D12_CPU_DESCRIPTOR_HANDLE t5h = heap_->GetCPUDescriptorHandleForHeapStart();
    t5h.ptr += device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device_->CreateShaderResourceView(gelu_t5_.get(), &sd, t5h);
    // the root signature of every kernel
    D3D12_DESCRIPTOR_RANGE range[2] = {};
    range[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range[0].NumDescriptors = 1;
    range[0].RegisterSpace = 0x7FFF0ADE;
    range[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;  // GELU T5 table: t0, space 1
    range[1].NumDescriptors = 1;
    range[1].RegisterSpace = 1;
    range[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER rp[4] = {};
    rp[0].ParameterType = rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[2].Constants.Num32BitValues = kRuntimeConstants + kMaxArgDwords;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[3].DescriptorTable.NumDescriptorRanges = 2;
    rp[3].DescriptorTable.pDescriptorRanges = range;
    D3D12_ROOT_SIGNATURE_DESC rs = {};
    rs.NumParameters = 4;
    rs.pParameters = rp;
    Com<ID3DBlob> blob, err;
    if (FAILED(serialize_(&rs, D3D_ROOT_SIGNATURE_VERSION_1, blob.put(), err.put())))
      return fail(std::string("D3D12SerializeRootSignature: ") +
                  (err ? std::string((const char*)err->GetBufferPointer(), err->GetBufferSize()) : "?"));
    return check(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                              __uuidof(ID3D12RootSignature), root_.pv()),
                 "CreateRootSignature");
  }

  // The arena: one shared committed buffer, imported into HIP.
  bool create_arena(const D3D12_HEAP_PROPERTIES& hp, D3D12_RESOURCE_DESC rd) {
    rd.Width = arena_bytes_;
    if (!check(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                __uuidof(ID3D12Resource), arena_.pv()),
               "CreateCommittedResource (arena)") ||
        !check(device_->CreateSharedHandle(arena_.get(), nullptr, GENERIC_ALL, nullptr, &arena_handle_),
               "CreateSharedHandle (arena)"))
      return false;
    const D3D12_RESOURCE_ALLOCATION_INFO info = device_->GetResourceAllocationInfo(0, 1, &rd);
    if (!hip_->import_memory(arena_handle_, info.SizeInBytes, arena_bytes_, &arena_hip_, &arena_mem_)) return fwd(false);
    // offset 0 is null; [256, 256 + 16 KB) is the MLP kernels' GELU + FP8 table (hlsl/wg_gelu_lut.hlsli)
    static_assert(wg::kGeluLutBase == kAlign, "the GELU table is the first arena range after the null header");
    free_[kAlign + wg::kGeluLutBytes] = arena_bytes_ - kAlign - wg::kGeluLutBytes;
    std::vector<uint32_t> table(wg::kGeluLutBytes / 4);
    wg::gelu_lut_expand(table.data());
    return fwd(hip_->upload(arena_hip_ + wg::kGeluLutBase, table.data(), wg::kGeluLutBytes));
  }

  // Fills the GELU T5 table: one copy on the D3D12 queue, waited for on the CPU (before any launch).
  bool gelu_t5_upload() {
    std::vector<uint32_t> table(wg::kGeluT5Bytes / 4);
    wg::gelu_t5_expand(table.data());
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = wg::kGeluT5Bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Com<ID3D12Resource> staging;
    if (!check(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                __uuidof(ID3D12Resource), staging.pv()),
               "CreateCommittedResource (GELU T5 staging)"))
      return false;
    void* mapped = nullptr;
    D3D12_RANGE no_read = {0, 0};
    if (!check(staging->Map(0, &no_read, &mapped), "Map (GELU T5 staging)")) return false;
    std::memcpy(mapped, table.data(), wg::kGeluT5Bytes);
    D3D12_RANGE written = {0, wg::kGeluT5Bytes};
    staging->Unmap(0, &written);
    if (!open_list()) return false;
    list_->CopyBufferRegion(gelu_t5_.get(), 0, staging.get(), 0, wg::kGeluT5Bytes);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = gelu_t5_.get();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    list_->ResourceBarrier(1, &b);
    const uint64_t v = execute();
    return v && wait_cpu(v);
  }

  bool create_pso(Kernel& k) {
    std::ifstream f(k.file, std::ios::binary);
    const std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.empty()) return fail("D3D12 kernel file missing: " + k.file);
    D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
    d.pRootSignature = root_.get();
    d.CS.pShaderBytecode = code.data();
    d.CS.BytecodeLength = code.size();
    return check(device_->CreateComputePipelineState(&d, __uuidof(ID3D12PipelineState), k.pso.pv()),
                 ("CreateComputePipelineState(" + k.name + ")").c_str());
  }

  std::unique_ptr<Device> hip_;
  Com<ID3D12Device> device_;
  Com<ID3D12CommandQueue> queue_;
  Com<ID3D12GraphicsCommandList> list_;
  Com<ID3D12Fence> fence_;
  Com<ID3D12Resource> arena_, ags_buf_, gelu_t5_;
  Com<ID3D12DescriptorHeap> heap_;
  Com<ID3D12RootSignature> root_;
  PFN_D3D12_SERIALIZE_ROOT_SIGNATURE serialize_ = nullptr;
  struct Alloc {
    Com<ID3D12CommandAllocator> alloc;
    uint64_t value = 0;  // fence value after its last list
  };
  Alloc allocs_[kAllocators];
  int cur_alloc_ = 0;
  HANDLE event_ = nullptr, fence_handle_ = nullptr, arena_handle_ = nullptr;
  void* sem_ = nullptr;
  void* arena_mem_ = nullptr;
  DevPtr arena_hip_ = 0;
  uint64_t arena_bytes_ = 0, in_use_ = 0, fence_value_ = 0;
  std::map<uint64_t, uint64_t> free_, used_;  // offset -> bytes
  std::map<std::string, Kernel> kernels_;
  std::set<std::string> untagged_cos_;               // file names of the code objects loaded without a tag
  std::map<std::string, std::string> tagged_cos_;    // tag -> file name of the code object loaded under it
  unsigned module_gen_ = 0;
  std::string adapter_name_;
  bool on_dx_ = false;         // the D3D12 queue owns the work since the last switch from HIP (a list is open)
  bool need_barrier_ = false;  // a D3D12 dispatch since that switch: the next one needs a UAV barrier first
  unsigned dispatches_ = 0;    // dispatches in the open list
  unsigned batch_ = 0;         // WG_DX12_BATCH: execute the list every this many dispatches (0: only at a switch)
};

}  // namespace

std::unique_ptr<Device> make_dx12_device(const std::string& rocm_dir, const std::string& build_dir, std::string* err) {
  std::unique_ptr<Dx12Device> d(new Dx12Device());
  if (!d->init(rocm_dir, build_dir)) {
    *err = d->err();
    return nullptr;
  }
  return std::unique_ptr<Device>(d.release());
}

}  // namespace wgrt

#endif  // _WIN32
