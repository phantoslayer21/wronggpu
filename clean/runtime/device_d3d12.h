// SPDX-License-Identifier: MIT
// Original explicit-queue extension of our native D3D12 Device, 2026-10-15.
// Basis: Device's ordered-stream contract and public Microsoft COM/D3D12 queue/fence APIs.
#pragma once

#include "device.h"

#if defined(_WIN32)
#include "interop/d3d12_include.h"

namespace wgrt {

// Direct-frame b1 payload, four 16-byte rows. Texture origin is shared by color/depth/render-resolution motion.
// exposure_mode: 0 = caller constant, 1 = t3 texture, 2 = computed. Texture states remain the caller's contract.
struct NativeDirectParams {
  uint32_t rx = 0, ry = 0, rw = 0, rh = 0;
  float msx = 1, msy = 1, mox = 0, moy = 0;
  float color_pi = 1;
  uint32_t exposure_mode = 0;
  // 1: an RGBA16F output gets the HIP output-copy bytes (framebuffer_half: +-65504 clamp, round to nearest even; the
  // typed store itself rounds toward zero on this driver); 0: the plain typed store (the Converter's output pass).
  // native_direct.hlsli nd_output_rne16 (b1 word 10).
  uint32_t output_rne16 = 0;
  // padding[0..2] (words 11..13) are written by bind_direct_frame: native_direct.hlsli nd_pi_rcp / nd_pi_lo / nd_pi_hi,
  // the pre-exposure division window derived from color_pi (native_direct_pi.h). Callers leave them zero.
  uint32_t padding[5] = {};
};
static_assert(sizeof(NativeDirectParams) == 64, "direct-frame shader constants must be 16 DWORDs");
static_assert(offsetof(NativeDirectParams, msx) == 16 && offsetof(NativeDirectParams, color_pi) == 32 &&
              offsetof(NativeDirectParams, exposure_mode) == 36 && offsetof(NativeDirectParams, output_rne16) == 40 &&
              offsetof(NativeDirectParams, padding) == 44,
              "direct-frame shader constant rows must match native_direct.hlsli b1");

struct NativeDirectFrame {
  ID3D12Resource *color = nullptr, *depth = nullptr, *motion = nullptr, *exposure = nullptr, *output = nullptr;
  DXGI_FORMAT color_format = DXGI_FORMAT_UNKNOWN, depth_format = DXGI_FORMAT_UNKNOWN;
  DXGI_FORMAT motion_format = DXGI_FORMAT_UNKNOWN, exposure_format = DXGI_FORMAT_R32_FLOAT;
  DXGI_FORMAT output_format = DXGI_FORMAT_UNKNOWN;
  NativeDirectParams params;
};

// One serialized caller, one frame in flight initially. Every pass records into a list owned by the Device;
// caller command lists are never adopted. The arena, list and returned extension are borrowed from Device.
class NativeD3D12Ext {
 public:
  virtual ~NativeD3D12Ext() = default;
  virtual ID3D12Resource* arena_resource() const = 0;
  virtual bool arena_offset(DevPtr pointer, size_t bytes, uint64_t& offset, std::string* err) = 0;

  // Starts a serialized direct frame: validates resources, waits for all previous recorded/submitted work before
  // rewriting descriptors/constants, and retains texture references through completion. The caller transitions
  // input textures to NON_PIXEL_SHADER_RESOURCE and output to UNORDERED_ACCESS before direct dispatch and restores
  // their states afterward. This method records no texture barriers. Exposure can be null outside texture mode.
  // Binding indices: t0=color, t1=depth, t2=motion, t3=exposure, u3=output, b1=NativeDirectParams.
  virtual bool bind_direct_frame(const NativeDirectFrame& frame, std::string* err) = 0;
  // Revokes permission for subsequent rs=direct launches; does not release pending resources, rewrite descriptors,
  // submit commands or claim completion. Caller must still submit/sync final texture restoration passes.
  virtual bool end_direct_frame(std::string* err) = 0;

  // The callback may bind its own compute state and record conversion/copy commands. It must not close/reset/
  // execute the list, call Device/extension operations, or change the arena's requested state. It restores caller texture states itself. The next
  // native kernel rebinds its root signature, heap and arena views. All objects referenced by the callback,
  // including descriptors, remain caller-owned and must survive sync(), including after callback failure.
  using RecordPass = std::function<bool(ID3D12GraphicsCommandList*, std::string&)>;
  virtual bool record_pass(D3D12_RESOURCE_STATES arena_state, const RecordPass& record, std::string* err,
                           const char* profile_label = "record_pass") = 0;

  // The producer must already have arranged a signal of input_ready, independently of work recorded after this
  // call. The fence must belong to this Device; future waits on our own completion fence are rejected. A reference
  // to the waited fence is retained through native queue completion. No CPU wait is performed here.
  virtual bool queue_wait(ID3D12Fence* input_ready, uint64_t value, std::string* err) = 0;

  // Submits all recorded work, including final conversion and texture-state restoration, then signals the native
  // fence. Success gives the caller an AddRef'd fence (Release it) and its completion value. Never a marker buffer.
  virtual bool submit_completion(ID3D12Fence** fence, uint64_t* value, std::string* err) = 0;
  // Waits for final recorded passes as well as kernels. Descriptor reuse and caller resource destruction require
  // successful completion; on unknown completion the caller must retain its whole dependent resource set.
  virtual bool sync(std::string* err) = 0;

  // ---- Recording into the GAME's command list (interop executor=dx12) ----------------------------------------------
  // From begin_external until end_external every launch, timestamp, direct-frame binding and record_pass is recorded
  // into `list` (open, of this device, DIRECT or COMPUTE): no submission, CPU wait, upload, download or copy happens
  // meanwhile, and the work runs in the caller's own submission order. The list's descriptor heap, compute root
  // signature and pipeline state are changed (FidelityFX contract: the caller binds its state again). A global UAV
  // barrier comes first and last. Direct frames bound meanwhile take the next of kDirectRing descriptor/constant sets
  // (no CPU wait unless that set's frame, kDirectRing frames ago, has not completed). end_external writes the frame's
  // completion marker (ID3D12GraphicsCommandList2::WriteBufferImmediate) behind the recorded work.
  virtual bool begin_external(ID3D12GraphicsCommandList* list, std::string* err) = 0;
  virtual bool end_external(std::string* err) = 0;
  // Frames recorded so far / known complete (the marker, read without waiting). Without ID3D12GraphicsCommandList2
  // there is no marker and every recorded frame counts as complete (external_markers() false).
  virtual uint64_t external_recorded() const = 0;
  virtual uint64_t external_done() const = 0;
  virtual bool external_markers() const = 0;
  // Pipeline states of every kernel the loaded code objects allow, now (the first frame then compiles nothing).
  virtual bool warm_up(std::string* err) = 0;
  // Highest arena byte any allocation reached, and the arena's size (a second device made with arena_bytes =
  // the high-water mark of a first one that made the same allocations fits them exactly).
  virtual uint64_t arena_high_water() const = 0;
  virtual uint64_t arena_capacity() const = 0;
  static constexpr unsigned kDirectRing = 16;
};

NativeD3D12Ext* d3d12_ext(Device* device);

// Adds references to the supplied device and DIRECT/COMPUTE queue, verifies canonical COM identity and the
// supported adapter, and runs the exact native_driver_probe on that device. No HIP runtime is loaded. A caller
// requesting WG_DX12_DEBUG must enable the debug layer before it creates its device; this factory never enables
// the layer on an existing device or modifies that device's InfoQueue filters.
// arena_bytes: the arena's size (rounded up to MiB); 0 = WG_D3D12_ARENA_MB or 2560 MiB.
std::unique_ptr<Device> make_d3d12_device_on(ID3D12Device* device, ID3D12CommandQueue* queue,
                                          const std::string& build_dir, std::string* err, uint64_t arena_bytes = 0);

}  // namespace wgrt
#endif
