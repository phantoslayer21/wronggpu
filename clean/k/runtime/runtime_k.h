// SPDX-License-Identifier: MIT
// Model K in the per-frame runtime (wronggpu_runtime.dll): one whole frame of spec 40 on the runtime's HIP stream,
// the C++ port of clean/k/runner/k_frame.py KFrame with the default network variant "fused"
// (clean/k/runner/k_net.py, k_launch.py) and the default fast input stage (k_io.co).
//
// Selected by the runtime option model=K (model_k.h model_from_options); wgrt::Runtime (clean/runtime/runtime.cpp)
// creates a KRuntime instead of model M's state and forwards every frame to it. The interop (D3D12, FrameIo) and the
// test entry points of wg_runtime.h (wg_rt_create_external, wg_rt_frame_io, wg_rt_upload_inputs, wg_rt_frame,
// wg_rt_buffer, wg_rt_kernels) work as for model M.
//
// Model data: the converter's model.wgcm for preset K (spec 48), loaded and checked by model_k_load.h against the
// preset-K map shipped with the runtime, packed by k_pack.h. Default location <model_dir>/K/model.wgcm (runtime option
// model_k_dir=<folder> overrides); map <build_dir>/preset-K.v1.map.json (on disk, else built into the library), else
// next to model.wgcm (runtime option model_k_map=<file> overrides).
//
// Per frame (clean/k/FORMATS.md "Runtime frame"): [raw inputs: input_convert] -> exposure (spec 07 auto-exposure when
// asked, exposure_auto_diet) -> k_expo_scalars (kappa, rho, lambda on the GPU from the exposure pair) -> input stage
// (k_in_color, k_in_base) -> eleven stage kernels -> output stage (k_out_color, k_out_pixel, k_out_hidden) writing
// the game output in the caller's format. Histories Hc, L, Hh double-buffered and zero at create; reset on the game's
// flag, the first frame and a render-size change; frame index f = 0 on a reset frame, then + 1.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../../runtime/device.h"
#include "../../runtime/wg_runtime.h"

namespace wgrt {

struct FrameIo;

// network=fused8 (clean/k/kernels/k_block8.h): KF8Params, the second argument of the FP8 stage kernels (the FP8
// copies of the weights, FP32 biases and power-of-two scales of one stage), and a stage's FP8 data (runtime_k.cpp
// fp8_stage).
struct KF8Params {
  DevPtr wqkv, mix, wo, w1, w2, bo, b1, b2;
  float pv, sz, sy, L, Hs, A2, B2, c0, sx, pad;
};
struct F8Stage {
  std::vector<uint8_t> bytes;  // the device buffer: qkv8, mix8, wo8, w1_8, w2_8, bo, b1, b2 (256-byte aligned parts)
  KF8Params q = {};            // pointers relative to the buffer start
  std::string why;             // empty: usable; else why this stage keeps its FP16 kernel
};

class KRuntime {
 public:
  using AllocFn = std::function<bool(DevPtr*, size_t)>;  // device memory owned (and freed) by the caller

  // After the caller made the device. external_io: inputs and output always come from FrameIo (no own output).
  bool create(const WgRtConfig& cfg, Device* dev, AllocFn alloc, bool external_io);
  // Enqueues one frame (does not wait).
  bool frame(const WgRtFrame& fr, const FrameIo* io);
  // Test path (wg_rt_upload_inputs): dense RGB float32, depth, motion (render pixels) into the own input buffers.
  bool upload_inputs(int rw, int rh, const float* rgb, const float* depth, const float* motion);
  const char* kernels();
  bool buffer(const std::string& name, DevPtr* p, size_t* bytes);
  const WgRtFrameInfo& info() const { return info_; }
  const std::map<std::string, std::vector<uint8_t>>& blobs() const { return blobs_; }
  const std::string& err() const { return err_; }
  const std::string& model_path() const { return model_path_; }
  // What the timing marks 0..4 bracket (FrameIo.timing_base): "exposure", "input", "network", "output".
  static const char* const kTimingLabels[4];
  static const char* timing_info();

 private:
  bool fail(const std::string& m) {
    err_ = m;
    return false;
  }
  bool dfail() { return fail(dev_->err()); }
  bool alloc(DevPtr* p, size_t bytes, bool zero = false);
  bool launch(const char* kernel, unsigned grid, unsigned block, void* arg0, void* arg1 = nullptr,
              void* arg2 = nullptr, void* arg3 = nullptr, void* arg4 = nullptr, void* arg5 = nullptr,
              void* arg6 = nullptr, void* arg7 = nullptr);
  bool network(int f);

  WgRtConfig cfg_ = {};
  Device* dev_ = nullptr;
  AllocFn alloc_;
  bool external_io_ = false;
  std::string err_, build_dir_, model_path_, map_path_, kernels_text_;
  int W_ = 0, H_ = 0, W16_ = 0, H16_ = 0, Bw_ = 0, Bh_ = 0, max_rw_ = 0, max_rh_ = 0;
  bool hdr_ = false, depth_inverted_ = false, f16_ = false;  // f16_: runtime option network=fused16 (or fused8)
  // network=fused8: FP8 forms of e0, e1, d1, d0 (k_block8.h); per stage its FP8 data (runtime_k.cpp fp8_stage)
  bool f8_ = false;
  std::map<std::string, F8Stage> f8data_;
  std::map<std::string, DevPtr> f8dev_;
  std::string f8why_;  // stages that keep their FP16 kernel, and why
  std::vector<std::string> f8stages_;  // runtime option fp8_stages (default e0, e1, d1, d0)
  bool tiled_input_ = false, tiled_last_ = false, tiled_color_last_ = false, input_uniforms_last_ = false;
  uint64_t tiled_launches_ = 0, tiled_color_launches_ = 0, input_uniform_launches_ = 0;
  uint64_t legacy_color_launches_ = 0, legacy_base_launches_ = 0;
  bool diag_input2_ = false, diag_copyin_ = false;  // runtime option k_diag (timing diagnostics)
  // d0 hands the head to k_out_pixel_c as the compact pixel head (k_stage2*_d0_hp; default). Runtime option
  // k_head=full: the 40-channel head map and k_out_pixel as before (same output bits; buffer "k.N" needs it)
  bool compact_head_ = true;
  // runtime option k_fast=1 (opt-in, not bit-exact): the "_f" variants of the input and output stage kernels
  // (k_io.hip FAST forms: hardware reciprocals instead of correctly rounded divisions, the luma history cubics as
  // packed FP16, the filter weight sum instead of the largest exponent; clean/k/tests/test_k_fast.py)
  bool fast_ = false;
  int offsets_table_[8][2] = {};
  std::map<std::string, std::vector<uint8_t>> blobs_;
  std::map<std::string, DevPtr> wdev_, net_;  // stage blobs; network maps (X0, S0..S4, A, B, OUT)
  std::map<std::string, size_t> net_bytes_;
  DevPtr Hc_[2] = {0, 0}, L_[2] = {0, 0}, Hh_[2] = {0, 0};
  DevPtr color_ = 0, depth_ = 0, motion_ = 0, out_ = 0, cpre_ = 0, cpre2_ = 0, ctap_ = 0, ctapT_ = 0;
  DevPtr exp2tab_ = 0, tables_ = 0, expo_ = 0, expo_partials_ = 0, expo_computed_ = 0, expo_sync_ = 0, xs_ = 0;
  // frame state
  int frames_ = 0, parity_ = 0, fidx_ = 0, last_rw_ = 0, last_rh_ = 0, level_last_ = -1, omode_last_ = -1;
  float jprev_[2] = {0, 0};
  int pending_mark_ = -1;
  WgRtFrameInfo info_ = {};
};

}  // namespace wgrt
