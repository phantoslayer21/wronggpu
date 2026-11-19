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
// asked, exposure_auto_diet) -> k_expo_scalars (E_cur, E_prev, E_aux of spec 42 draft 3 section 4, then kappa, rho,
// lambda on the GPU) -> input stage (exact by default: k_io_exact.co) -> eleven stage kernels -> output stage
// (k_out_color, k_out_pixel, k_out_hidden) writing the game output in the caller's format. Histories Hc, L, Hh
// double-buffered and zero at create; reset on the game's flag and the first frame only (not on a render-size
// change); frame index f = 0 on a reset frame, then + 1.
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
  // native_direct (backend 3, external_io; the D3D12 executor, interop/executor_d3d12.cpp): the game's colour, depth,
  // motion and exposure textures and its output texture are bound by the caller (NativeD3D12Ext::bind_direct_frame):
  // the kernels that read inputs or write the output run as their "_direct" D3D12 forms (see direct_names), no dense
  // input buffers are allocated, and only the exact default path is taken (network=fused, exact input, tiled input).
  // On backend 3 create first checks that <build>/dx12/manifest.json has a port of every kernel the context can
  // launch (d3d12_missing): a context with a missing port is refused before the model is loaded, so the D3D12
  // executor refuses it and the HIP executor runs it (executor_select.h).
  bool create(const WgRtConfig& cfg, Device* dev, AllocFn alloc, bool external_io, bool native_direct = false);
  // Enqueues one frame (does not wait).
  bool frame(const WgRtFrame& fr, const FrameIo* io);
  // native_direct: whether frame() would take this frame (nothing enqueued); false with err() saying why. The D3D12
  // executor calls it before it records anything ("frame not taken": the HIP executor then runs the context). A frame
  // whose render size is above 0.68 x the output (DLAA-like; no tiled input stage) or whose jitter is outside
  // [-0.5, 0.5] is not taken: the legacy input kernels have no direct forms.
  bool direct_preflight(const WgRtFrame& fr, const FrameIo* io);
  // Kernel names (as launched) a context of this configuration can launch on backend 3, and those of them the
  // manifest has no port of (comma list; empty: every kernel is ported). Valid after create's option parsing.
  std::vector<std::string> d3d12_kernels() const;
  std::string d3d12_missing() const;
  // Test path (wg_rt_upload_inputs / wg_rt_upload_inputs_rgba): dense colour float32 of `channels` values per pixel (3:
  // RGB, the alpha is 1 as for a colour without an alpha channel; 4: RGBA, the alpha as given), depth, motion (render
  // pixels) into the own input buffers. The output alpha follows the colour's alpha (issue #26).
  bool upload_inputs(int rw, int rh, const float* color, const float* depth, const float* motion, int channels = 3);
  const char* kernels();
  bool buffer(const std::string& name, DevPtr* p, size_t* bytes);
  const WgRtFrameInfo& info() const { return info_; }
  const std::map<std::string, std::vector<uint8_t>>& blobs() const { return blobs_; }
  const std::string& err() const { return err_; }
  // notes of create for the caller's log (wg_rt_notes), one per line, or empty: the round-to-odd softmax ports
  const std::string& notes() const { return notes_; }
  // true when create prepared every kernel this context can select (the native D3D12 device): the caller then
  // prepares nothing more (the executor's warm_up would also compile kernels the context never runs)
  bool prepared() const { return prepared_; }
  // A line for the caller's log that a frame produced (k_approx=a1 on the legacy input path: once per context), or an
  // empty string; taking it clears it.
  std::string take_frame_note() {
    std::string n;
    n.swap(frame_note_);
    return n;
  }
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
  // after(stage name): called after each stage's launch (k_color_at / k_warm launches)
  bool network(int f, const std::function<bool(const char*)>& after = nullptr);
  bool warm(const char* at);
  // the tiled input stage's guard (frame and direct_preflight): sizes, the 0.68 ratio, finite jitter in [-0.5, 0.5]
  bool tiled_guard(int rw, int rh, float jx, float jy) const;
  bool options_ok_direct();
  // the softmax weight rounding (runtime_k.cpp "create-time self-check"): smx_select picks the mix form or the __rodd
  // ports after checking both on this device (native D3D12), or refuses; smx_check hashes one form's stage outputs at
  // this model's window offsets (all: every possible offset), smx_probe counts one form's probe mismatches,
  // smx_case_name names a self-check case
  bool smx_select(int backend, const std::string& hook);
  bool smx_check(bool rodd, bool all, std::vector<std::pair<std::string, uint64_t>>* out);
  bool smx_probe(bool rodd, int* bad);
  std::string smx_case_name(int i, int ox, int oy, bool rodd) const;
  // the stage kernel network() launches: the __rodd form of a softmax stage after a failed probe
  std::string stage_kernel(const std::string& k, const char* stage) const;
  // deep-split (runtime option k_split, native D3D12 device): stage m as its four phase launches (k_split_m_*,
  // clean/k/dx12/kernels/k_block_split.hlsl) in place of one k_stagew4_m launch; launch() redirects k_stagew4_m[__rodd]
  // here, so network() and the create-time self-check both run it. split_alloc (create): the phases' scratch, sized for
  // the most windows stage m can have in a frame of this context or in the self-check's m case.
  bool split_m(bool rodd, void* params);
  bool split_alloc();
  int fill_ = -1;  // option k_fill (test hook): the byte every frame-written buffer and the histories start with

  WgRtConfig cfg_ = {};
  Device* dev_ = nullptr;
  AllocFn alloc_;
  bool external_io_ = false;
  bool native_direct_ = false;  // create's native_direct: the "_direct" kernels on the caller's bound textures
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
  bool s_breakpoints_fallback_ = false;
  uint64_t tiled_launches_ = 0, tiled_color_launches_ = 0, input_uniform_launches_ = 0;
  uint64_t legacy_color_launches_ = 0, legacy_base_launches_ = 0;
  bool input_uniforms_ = true;  // runtime option k_in_uniforms (0: always k_in_tiled, a port-check diagnostic)
  bool diag_input2_ = false, diag_copyin_ = false;  // runtime option k_diag (timing diagnostics)
  // d0 hands the head to k_out_pixel_c as the compact pixel head (k_stage2*_d0_hp; default). Runtime option
  // k_head=full: the 40-channel head map and k_out_pixel as before (same output bits; buffer "k.N" needs it)
  bool compact_head_ = true;
  // runtime option k_fast=1 (opt-in, not bit-exact): the "_f" variants of the input and output stage kernels
  // (k_io.hip FAST forms: hardware reciprocals instead of correctly rounded divisions, the luma history cubics as
  // packed FP16, the filter weight sum instead of the largest exponent; clean/k/tests/test_k_fast.py)
  bool fast_ = false;
  // input-stage kernels from k_io_exact.co (tag "kx"): the default; off with k_fast_in=1 (or k_in_exact=0, k_fast=1)
  bool in_exact_ = true;
  bool expo_v051_ = false;  // runtime option k_expo_rule=v051 (A/B only): the 0.5.1 exposure rule
  int offsets_table_[8][2] = {};
  std::map<std::string, std::vector<uint8_t>> blobs_;
  std::map<std::string, DevPtr> narrow_bias_;  // FP32 bias planes for the D3D12 two-window narrow stages
  DevPtr d0_compact_head_ = 0;                  // the compact head's two permuted tiles
  std::map<std::string, DevPtr> wdev_, net_;  // stage blobs; network maps (X0, S0..S4, A, B, OUT)
  std::map<std::string, size_t> net_bytes_;
  DevPtr Hc_[2] = {0, 0}, L_[2] = {0, 0}, Hh_[2] = {0, 0};
  DevPtr color_ = 0, depth_ = 0, motion_ = 0, out_ = 0, cpre_ = 0, cpre2_ = 0, ctap_ = 0, ctapT_ = 0;
  DevPtr s_breakpoints_ = 0;
  DevPtr exp2tab_ = 0, tables_ = 0, expo_ = 0, expo_partials_ = 0, expo_computed_ = 0, expo_sync_ = 0, xs_ = 0;
  DevPtr kexpo_ = 0;  // k_expo_scalars state: E_cur, E_prev, E_aux of the last frame, E_cur kept for the next
  // frame state
  int frames_ = 0, parity_ = 0, fidx_ = 0, last_rw_ = 0, last_rh_ = 0, level_last_ = -1, omode_last_ = -1;
  float jprev_[2] = {0, 0};
  int pending_mark_ = -1;
  // D3D12 latency options (create: k_color_at, k_warm, k_warm_line, k_warm_groups): the stage after which k_out_color
  // runs unordered (empty: before the input stage, HIP's order), the stage after which k_warm runs (empty: off)
  std::string color_at_, warm_at_, warm_to_;
  // the softmax stages run their round-to-odd (__rodd) ports (smx_select: the mix form failed the create-time
  // self-check, native D3D12 device only)
  bool smx_rodd_ = false;
  // runtime option k_approx (opt-in, the native D3D12 device; never the default): close-approximation variants by tag
  // ("a1": the tiled input stage on the transcendental unit, kernels kx/k_in_tiled*__a1)
  bool approx_a1_ = false;
  bool approx_a2_ = false;  // "a2": the output stage (k_out_color, k_out_pixel_c) with KO_APPROX, kernels *__a2
  std::string approx_note_;  // the note of create when k_approx is set
  // " k_approx=a1[,a2]" for the kernels line, or empty
  std::string approx_tag() const {
    if (!approx_a1_ && !approx_a2_) return std::string();
    return std::string(" k_approx=") + (approx_a1_ ? "a1" : "") + (approx_a1_ && approx_a2_ ? "," : "") +
           (approx_a2_ ? "a2" : "");
  }
  // runtime option k_split (backend 3 only; default "m"): stage m runs as its phase launches (split_m); "off": one
  // k_stagew4_m launch. split_cap_: windows the scratch holds; split_z_ / x1_ / a_ / h_: the phases' FP16 scratch
  bool split_m_ = false;
  int split_cap_ = 0;
  DevPtr split_z_ = 0, split_x1_ = 0, split_a_ = 0, split_h_ = 0;
  size_t split_bytes_[4] = {0, 0, 0, 0};  // the bytes of split_z_, split_x1_, split_a_, split_h_ (split_alloc)
  std::string smx_note_;
  std::string smx_test_;  // option k_smx_test (developer builds only)
  std::string notes_;  // notes of create for the caller's log (notes()), one per line, or empty
  bool prepared_ = false;  // prepared()
  std::string frame_note_;         // take_frame_note()
  bool a1_legacy_noted_ = false;   // the k_approx=a1 legacy-input line was produced (once per context)
  unsigned warm_line_ = 128, warm_groups_ = 64;
  DevPtr warm_dummy_ = 0;
  WgRtFrameInfo info_ = {};
};

}  // namespace wgrt
