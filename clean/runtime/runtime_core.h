// SPDX-License-Identifier: MIT
// C++ entry points of the core runtime for code in the same library (the D3D12 interop, spec 10 sections 1, 6, 8).
// The C API in wg_runtime.h wraps the same functions.
#pragma once

#include <memory>
#include <string>

#include "device.h"
#include "wg_runtime.h"

namespace wgrt {

// Per-frame inputs and output in external device memory (D3D12 resources shared with HIP), instead of the runtime's
// own input buffers (wg_rt_upload_inputs) and output buffer. Formats as clean/FORMATS.md "Per-frame runtime":
// color [rh][rw] 4 x f32, depth [rh][rw] f32, motion [rh][rw] 2 x f32, out [oh][ow] 3 x f32.
struct FrameIo {
  DevPtr color = 0, depth = 0, motion = 0, out = 0;
  // Exposure source: -1 = the configuration (auto-exposure or WgRtFrame.exposure); 0 = WgRtFrame.exposure;
  // 1 = the float at expo_src (e.g. the game's exposure texel, copied there on the GPU); 2 = compute (spec 07).
  int exposure_mode = -1;
  DevPtr expo_src = 0;
  // Timing log: >= 0 records device events timing_base + 0..4 (start, after enc0 incl. exposure, after the Swin
  // stages, after dec0, end after post / filter); the caller reserves them (Device::reserve_events).
  int timing_base = -1;
  // The caller's output pass reads `out` as RGBA16F (8 bytes per pixel) instead of 3 x f32: needed for the post kernels
  // with FP16 output (post_fused*_o16); the runtime refuses those otherwise.
  bool out_rgba16f = false;
  // Interop copy paths (options input_copy / output_copy). raw_inputs: color, depth and motion above are the game's
  // textures copied as they are into rows of the given pitches (bytes): color RGBA16F, depth R32F, motion with
  // motion_channels FP16 channels, raw values (m = ms * raw + mo, here); the runtime converts them into its own dense
  // buffers (kernels/frame.hip input_convert), and exposure reads color / pi with 1 / pi = 1 (not bit-identical to the
  // default path). out_half_pitch > 0: out is written as RGBA16F rows of that pitch (bytes) with pi folded in.
  bool raw_inputs = false;
  int color_pitch = 0, depth_pitch = 0, motion_pitch = 0, motion_channels = 4;
  float ms[2] = {1, 1}, mo[2] = {0, 0};
  int out_half_pitch = 0;
  // Motion vectors at output resolution (spec 03 draft 4 "Motion at output resolution", spec 06 section 3; runtime
  // option motion_outres=original): motion_hi != 0 is the raw motion of the M = motion_hi_w x motion_hi_h region (the
  // output size; the subrect origin already applied) as rows of motion_hi_pitch bytes, a texel being two FP16
  // (motion_hi_wide = 0, 4 bytes) or two f32 (1, 8 bytes), with 4 spare bytes after the last texel
  // (kernels/wg_motion_hi.h). mv_scale_hi converts the raw motion to pixels of M, b_mv is the per-axis offset added
  // to the raw motion first (spec 03). The input encoder and the reconstruction then read this buffer; `motion` (render
  // size) is not read by them (other users, e.g. the interop's sparkle guard, still may). Ignored (the frame runs on
  // `motion` as before) when the runtime was not created with motion_outres=original or the frame's sizes are not
  // taken by the kernels of that path.
  DevPtr motion_hi = 0;
  int motion_hi_pitch = 0, motion_hi_wide = 0, motion_hi_w = 0, motion_hi_h = 0;
  float mv_scale_hi[2] = {1, 1}, b_mv[2] = {0, 0};
  // Native direct resources (native_direct below) with motion_outres=original: the kernels read the output-size motion
  // from the bound motion texture (texel (0, 0) = the origin of M; the interop's motionhi pass reads the same texels),
  // so motion_hi stays 0 and this flag says the frame brings it; motion_hi_w / _h / _pitch (= 4 motion_hi_w, FP16
  // texels: wide 0), mv_scale_hi and b_mv as above. Every frame of such a runtime sets it (runtime_frame refuses
  // otherwise).
  bool motion_hi_direct = false;
  // Interop option handover=staged (model K only): an imported fence the runtime signals to out_signal_value in
  // stream order right before the frame's output stage (the game's queue then polls the done flag).
  void* out_signal = nullptr;
  uint64_t out_signal_value = 0;
  // Private native D3D12 texture route: inputs/output are bound through the device's typed resource descriptors,
  // so the four DevPtrs may be zero. Must match runtime_create's native_direct_io mode. Intermediate/history
  // buffers keep the normal arena ABI. Descriptor formats, lifetimes and states belong to the caller.
  bool native_direct = false;
};

struct Runtime;

// dev: a device created by the caller (the interop's HIP device with external memory), or null to create one from
// cfg.backend / cfg.rocm_dir. external_io: every frame passes its inputs and output in FrameIo, so the runtime does
// not allocate its own (wg_rt_upload_inputs and the "input.*" / "out" buffers are then unavailable). Returns null and
// sets *err on failure. native_direct_io additionally skips dense color/depth/motion conversion buffers and requires
// external_io + backend 3 + the ported Model M route. Typed textures are bound by the private native caller.
Runtime* runtime_create(const WgRtConfig& cfg, std::unique_ptr<Device> dev, std::string* err,
                        bool external_io = false, bool native_direct_io = false);
void runtime_destroy(Runtime* rt);
// Enqueues one whole frame (does not wait). io may be null.
bool runtime_frame(Runtime* rt, const WgRtFrame& frame, const FrameIo* io);
// Same mode and direct-frame checks used by runtime_frame, before any caller texture transitions or queue work.
// Only runtime_error may change: no GPU calls and no frame/history/reset mutation. Buffer-mode frames are unchanged.
bool runtime_direct_preflight(Runtime* rt, const WgRtFrame& frame, const FrameIo* io);
const std::string& runtime_error(Runtime* rt);
Device* runtime_device(Runtime* rt);
// Color/output pre-exposure used by the runtime: finite positive input, else 1; also 1 with pre_exposure=original.
// The direct-resource caller puts this value in its b1 constants before recording any frame dispatches.
// Read-only: does not change frame/history state or the raw-color exposure measurement's separate pi rule.
float runtime_effective_pre_exposure(Runtime* rt, float pre_exposure);
// The kernels in use, one line (as wg_rt_kernels): for the front end's log at create, so that every run documents its
// kernel choice. Valid until the next call.
const char* runtime_kernels(Runtime* rt);
// Notes of create for the front end's log (as wg_rt_notes): one per line, or empty. Valid while the runtime lives.
const char* runtime_notes(Runtime* rt);
// Device memory the runtime allocated itself (model data, state, intermediates), in bytes.
uint64_t runtime_memory_bytes(Runtime* rt);
// The reconstruction size D in use (model K: the output size).
void runtime_recon_size(Runtime* rt, int* dw, int* dh);
// The model in use: "M" or "K" (runtime option model=K, clean/k/runtime/runtime_k.h).
const char* runtime_model(Runtime* rt);
// What the timing marks of FrameIo.timing_base bracket: four labels (model M: enc0, Swin, dec0, post; model K:
// exposure, input, network, output) and, for model K, a description for the log (null for model M).
const char* const* runtime_timing_labels(Runtime* rt);
const char* runtime_timing_info(Runtime* rt);
// Whether cfg (recon_output) or the environment (WG_RECON_SIZE=output) asks for the experimental D = O.
bool recon_output_requested(const WgRtConfig& cfg);

}  // namespace wgrt
