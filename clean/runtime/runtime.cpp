// SPDX-License-Identifier: MIT
// Per-frame runtime, core part (spec 10 sections 0, 2-5, 7, 8). See wg_runtime.h and clean/FORMATS.md.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../converter/src/json.h"
#include "../k/runtime/model_k.h"
#ifndef WG_NO_MODEL_K  // build.py --no-model-k: a build without model K (model=K fails at create, no_model_k.cpp)
#include "../k/runtime/runtime_k.h"
#endif
#ifdef WG_MODEL_L  // Model L (#89; model_select.h model_l_built): its model data and weight packer (clean/l/runtime)
#include "../l/runtime/l_model.h"
#include "../l/runtime/l_pack.h"
#endif
#include "device.h"
#include "embedded.h"
#include "fpconv.h"
#include "model.h"
#include "model_select.h"
#include "runtime_core.h"
#include "wg_runtime.h"

namespace wgrt {

namespace {

// Kernel parameter structs, passed by value; must match the kernels (static_asserts there).
struct EncParams {  // kernels/encoder.hip WgEncParams
  int rw, rh, fw, fh, hw, hh, frame, reset, depth_inverted, pad0;
  float dw, dh, kx, ky, ikx, iky, sx, sy, bx, by, jx, jy, jpx, jpy, dis_scale, idw, idh, irw, irh;
};
static_assert(sizeof(EncParams) == 116, "EncParams");
struct DecParams {  // kernels/decoder.hip WgDecParams
  int rw, rh, fw, fh, reset, pad0;
  float dw, dh, kx, ky, jx, jy, pjx, pjy;
};
static_assert(sizeof(DecParams) == 56, "DecParams");
struct ExpParams {  // kernels/exposure.hip WgExpParams
  int rw, rh, nx, ny, nparts, pad0;
  float inv_pi, K, inv_n;
};
static_assert(sizeof(ExpParams) == 36, "ExpParams");
struct PostParams {  // kernels/post.hip WgPostParams
  int rw, rh, dw, dh, reset, pad0, pad1, pad2;
  float rx, ry, jx, jy, e, e_prev, ratio, pad3;
};
static_assert(sizeof(PostParams) == 64, "PostParams");
// Motion vectors at output resolution (kernels/post_motion_outres.h WgMoParams, encoder_motion_outres.h WgEncMoParams)
struct MoParams {
  int mw, mh, pitch, wide;
  float sx, sy, bx, by, idw, idh;
};
static_assert(sizeof(MoParams) == 40, "MoParams");
struct EncMoParams {
  int mw, mh, pitch, wide;
  float kmx, kmy;
};
static_assert(sizeof(EncMoParams) == 24, "EncMoParams");

// Display image of kernels/post.hip: three FP16 planes with row pitch display_pitch(dw) (runner/post.py).
size_t display_pitch(int dw) { return ((size_t)dw + 63) & ~(size_t)63; }
size_t display_bytes(int dw, int dh) { return 3 * display_pitch(dw) * (size_t)dh * 2; }

}  // namespace (reopened below)

bool recon_output_requested(const WgRtConfig& c) {
  if (c.recon_output) return true;
  const char* e = std::getenv("WG_RECON_SIZE");
  return e && std::strcmp(e, "output") == 0;
}

namespace {

const char* const kSwinOrder[10] = {"enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"};

// Spec 02 "Window partition": (ox, oy) per stage and parity f.
void window_offset(const std::string& s, int f, int* ox, int* oy) {
  const bool second = s == "b1" || s == "b3" || s == "b5" || s == "dec2" || s == "dec1";
  if (second) {
    *ox = f ? 2 : 0;
    *oy = 0;
  } else {
    *ox = f ? 0 : 2;
    *oy = 2;
  }
}

int cdiv(long long a, long long b) { return (int)((a + b - 1) / b); }
int ceil32(int v) { return (v + 31) / 32 * 32; }

// Default kernel choice: the fastest variant of each stage that is exact (bit-identical to the spec kernels or
// accepted by the spec team). See WgRtConfig in wg_runtime.h.
const char* const kDefaultExposure = "base";       // exposure_partial + exposure_final
const char* const kDefaultPost = "post_recon2_r";   // bit-identical drop-in of post_recon (clean/post 413aa4c)
// Default post path: post_recon2_r_band + post_filter_c1_s4_band over post_bands bands (bit-identical, measured in
// game: -0.03 ms), unless the post kernel is chosen, the filter is chosen unbanded, or recon=output; a frame whose
// size the banded kernels do not take runs post_recon + kDefaultFilter (unbanded).
const char* const kDefaultBandPost = "post_recon2_r_band";
const char* const kDefaultFilter = "post_filter_c1_s4";  // bit-identical drop-in of post_filter

// Folder of the emulator libraries (backend 1): that of emu_library, else the build folder.
std::string emu_lib_dir(const WgRtConfig& c) {
  if (!c.emu_library || !*c.emu_library) return c.build_dir ? c.build_dir : ".";
  const std::string e = c.emu_library;
  const size_t k = e.find_last_of("/\\");
  return k == std::string::npos ? std::string(".") : e.substr(0, k);
}

// Default of WG_D3D12_FUSION (native D3D12 backend only; see fz_pm / fz_xe / fz_mb / fz_rv / fz_xo), mirrored by compare.py's
// DEFAULT_D3D12_FUSION. With the fused kernels regenerated from the live swin_block.hlsl / bneck.hlsl
// (kernels/fused_*_gen.py, run by build_dx12.py) the four measured on the in-game direct path (interop hipbench TOTAL,
// executor=dx12, 4 interleaved rounds of fresh processes): none 2.2920 ms, pm 2.2855, xe 2.2875, mb 2.2735,
// pm,xe 2.2812, pm,xe,mb 2.2618, pm,xe,mb,rv 2.2560 (-36 us; every round -34 .. -37 us). Bit-identical (compare.py,
// dx12same). "none" = the unfused launches. xo (dec2's block overlapping the bottleneck's tail, waiting on its b5 row
// counters) on top: 2.2485 -> 2.2370 ms (-11.5 us, 5 rounds, every round -10 .. -14.5 us). xd (d1p12 overlapping dec2's
// fused block, waiting on its window-row counters) on top of xo: 2.2330 -> 2.2280 ms (-5 us, median of 13 paired rounds,
// 10 faster; one 8-round run -11.5 us with every round faster). Re-measured in one process (runtime A/B, the same frames
// alternating every 300 frames; inter-process noise was +-25 us): xd is +5.0 us at 1707x960 and +7.5 us at 1920x1080,
// -2.25 us at 1280x720, so it now runs on frames of at most kXdMaxPixels render pixels only.
const char* const kDefaultD3d12Fusion = "pm,xe,mb,rv,xo,xd";
// xd only up to this many render pixels per frame (see fz_xd): it is faster at 1280x720 and slower at 1707x960 and above
constexpr long long kXdMaxPixels = 1200000;

// Config field, else environment variable, else default.
std::string choose(const char* field, const char* env, const char* def) {
  if (field && *field) return field;
  const char* v = std::getenv(env);
  if (v && *v) return v;
  return def;
}

// post_recon2* / 3* / 4*: the render footprint of a workgroup (32 D columns x 8 row pairs) must fit the kernel's LDS
// staging area (kR2Cap = 336 render pixels), D needs 4 columns or more (post_recon2* read the history taps as
// spans, history_cr_span), and the sizes must fit post_recon2_body's 32-bit byte offsets (mul24 products: full
// history 8 dw dh, display planes 6 pitch dh, fallback colour 16 rw rh bytes <= 2^32; mul24 factors 8 dw, 2 pitch,
// 16 rw < 2^24); runner/post.py recon2_supported / offsets32_ok.
bool recon2_supported(int rw, int rh, int dw, int dh) {
  const long long pitch = ((long long)dw + 63) & ~63LL, two32 = 1LL << 32, two24 = 1LL << 24;
  const bool off32 = 8LL * dw * dh <= two32 && 6LL * pitch * dh <= two32 && 16LL * rw * rh <= two32 &&
                     16LL * pitch < two24 && 16LL * rw < two24;
  return dw >= 4 && off32 && ((long long)(31.0 * rw / dw) + 4) * ((long long)(14.0 * rh / dh) + 5) <= 336;
}

// post_tf* (true fusion: reconstruction + output filter in one kernel, the display image never goes to memory): the
// render footprint of every step (64 D columns x 8 row pairs) must fit the staging area, cap 352 texels for
// post_tf_p12 (D >= 2.25 R: Quality ... Ultra Performance) or 644 for post_tf_dl_p12 (native AA, D = 1.5 R), plus the
// conditions of recon2_supported (dw >= 4, 32-bit offsets); runner/post.py tf_supported.
bool tf_supported(int rw, int rh, int dw, int dh, int cap) {
  return recon2_supported(rw, rh, dw, dh) &&
         ((long long)(63.0 * rw / dw) + 4) * ((long long)(14.0 * rh / dh) + 5) <= cap;
}
constexpr int kTfCap = 352, kTfDlCap = 644;
// post_tf4_dl_* (D3D12 only, dx12/kernels/post.hlsl; Brief 15): the same kernel with NW = 4 row pairs per step (256 lanes)
// and a staging cap of 448, for native AA and every ratio between 1.0 and 1.5 on the native D3D12 backend (HIP's
// post_tf_dl_* needs 41472 bytes of LDS, more than D3D12's 32 KB). The step footprint of NW = 4 (64 D columns x 4 row
// pairs) must fit 448, plus recon2_supported; runner/post.py tf_supported(nw=4, cap=TF4_DL_CAP),
// tests/test_post_tf_geometry.py (every render size from O / 1.5 to O fits).
constexpr int kTf4Cap = 448;
bool tf4_supported(int rw, int rh, int dw, int dh) {
  return recon2_supported(rw, rh, dw, dh) &&
         ((long long)(63.0 * rw / dw) + 4) * ((long long)(6.0 * rh / dh) + 5) <= kTf4Cap;
}
constexpr int kTfCpairs = 68;  // output row pairs per post_tf workgroup (3 C + 4 = 208 D rows: 13 full steps)
// The workgroups dispatched last (about one round of resident workgroups: kTfSlots = 4 per WGP of an RX 9070 XT) take
// chunks of kTfTailPairs instead, which shortens the tail of the launch (runner/post_sizes.py tf_chunks).
constexpr int kTfTailPairs = 36, kTfSlots = 128;
// Chunk plan of post_tf: k1 chunks of cp output row pairs, then chunks of c2; returns the number of chunks.
int tf_chunks(int ow, int oh, int cp, int* k1, int* c2) {
  const int npo = cdiv(oh, 2), strips = cdiv(ow, 40);
  *c2 = std::min(kTfTailPairs, cp);
  const int m = cdiv(kTfSlots, strips);
  *k1 = std::max(0, (npo - m * *c2) / cp);
  return *k1 + cdiv(npo - *k1 * cp, *c2);
}

// post_fused* / post_fused3*: every band footprint must fit the LDS staging area (kFuStageCap = 336 render pixels);
// runner/post.py fused_supported.
bool fused_supported(int rw, int rh, int dw, int dh) {
  return ((long long)(63.0 * rw / dw) + 4) * ((long long)(4.0 * rh / dh) + 5) <= 336;
}

// Output filter variants of kernels/post.hip: output columns per workgroup and output row pairs per lane (S).
bool filter_shape(const std::string& k, int* tile_x, int* s) {
  if (k == "post_filter_quality9" || k == "post_filter_quality9_o16") {
    *tile_x = 40;
    *s = 4;
    return true;
  }
  static const char* const names[] = {"post_filter",       "post_filter_s4",        "post_filter_s16",
                                      "post_filter_c1",    "post_filter_c1_s4",     "post_filter_c1_s16",
                                      "post_filter_c1_s4_o16", "post_filter_c1_s4_band", "post_filter_c1_s2_band"};
  for (const char* n : names)
    if (k == n) {
      const auto ends = [&](const std::string& t) {
        return k.size() >= t.size() && k.compare(k.size() - t.size(), t.size(), t) == 0;
      };
      *tile_x = k.find("_c1") != std::string::npos ? 40 : 128;
      *s = ends("_s4") || ends("_s4_o16") || ends("_s4_band") ? 4 : ends("_s2_band") ? 2 : ends("_s16") ? 16 : 8;
      return true;
    }
  return false;
}

// Reconstruction kernels that read the post_prep records (kernels/post.hip; runner/post.py PREP_VARIANTS).
bool uses_prep(const std::string& k) {
  return k.compare(0, 11, "post_recon5") == 0 || k.compare(0, 11, "post_recon6") == 0;
}

bool read_text(const std::string& path, std::string* out) {  // on disk, else built in (embedded.h)
  std::vector<char> bytes;
  if (!wg_read_file(path, &bytes)) return false;
  out->assign(bytes.begin(), bytes.end());
  return true;
}

int json_int(const wgc::Json* j, int def) {
  std::int64_t v = 0;
  return j && j->as_int64(v) ? (int)v : def;
}

}  // namespace

struct Runtime {
  WgRtConfig cfg = {};
  std::string build_dir, err;
  std::unique_ptr<Device> dev;
#ifndef WG_NO_MODEL_K
  std::unique_ptr<KRuntime> k;  // model K (runtime option model=K): the whole frame runs there (clean/k/runtime)
#endif
  // Model L (runtime option model=L, builds with WG_MODEL_L): Model M's frame with Model L's encoder in step 2 and its
  // decoder in step 4 (clean/l/FORMATS.md); the ten Swin stages, exposure and post are Model M's.
  bool model_l = false;
#ifdef WG_MODEL_L
  // The maps of clean/l/FORMATS.md in three buffers (no kernel works in place):
  //   l_wide [fh + 4][fw + 4] x 64 bytes: q (FP8 halo, 32 bytes a pixel), then x2 (FP16 halo; q is dead after the
  //          first linear), then the decoder's y2 (FP16, [fh][fw])
  //   l_narrow [fh + 4][fw + 4] x 32 bytes: x1 (FP8 halo), then the decoder's y (FP8, [fh][fw])
  //   l_x3 [fh][fw] x 32 bytes: the encoder block's output
  DevPtr l_wide = 0, l_narrow = 0, l_x3 = 0;
  bool l_nc[2] = {false, false};  // enc0_block, dec0_block: clamp-free block kernel (bit 0 of the blob's first word)
#endif
  DRow drow;
  int block_threads[5] = {256, 256, 256, 256, 256}, gemm_threads = 256, nt = 2, wpw[3] = {1, 1, 1};  // = variants.DEFAULT_GEOMETRY
  int block_wpg[5] = {0, 0, 0, 0, 0};  // windows per workgroup per block kernel; 0 = threads / 32 * wpw
  // kernel choice (choose_kernels)
  std::string swin_co = "swin.co", enc0_co = "encoder.co", dec0_co = "decoder.co", exposure_co = "exposure.co";
  std::string exposure_variant, post_kernel, filter_kernel, post_last, filter_last, kernels_text;
  bool force_clamp = false;            // Swin: always the clamping block kernels (variant geometry; v2 has no _nc)
  bool fuse = false;                   // Swin: fused merge / expand kernels (variant geometry "fuse")
  // Encoder / decoder stages that run their fused kernel from the base build (swin.co holds the fused kernels):
  // swin_fuse_stages / WG_SWIN_FUSE_STAGES, default dec1 (spec team: dec1 0.367 vs 0.379 ms, bit-identical; the
  // encoder side is slower).
  std::vector<std::string> fuse_stages;
  std::string fuse_stages_text;
  std::map<std::string, bool> swin_nc;  // Swin stage -> launch the clamp-free block kernel (weights' FLAGS bit 0)
  // Per-stage Swin build variants (swin_stage_variants / WG_SWIN_STAGE_VARIANTS): code object loaded with the
  // variant's name as tag, launch geometry from variants.json. Stages not listed use the global choice above.
  struct SwinGeo {
    std::string tag;
    int threads[5] = {256, 256, 256, 256, 256}, wpg[5] = {0, 0, 0, 0, 0}, wpw[3] = {1, 1, 1};
    int gemm_threads = 256, nt = 2;
    bool force_clamp = false;
    bool fuse = false;  // encoder / decoder blocks with the patch merge / expand fused in (swin.hip, variant fuse)
    int mega_threads = 0, mega_groups = 0;  // bottleneck megakernel swin_bneck6 (bmega* builds), 0 = none
    int mid_threads = 0, mid_groups = 0;    // persistent kernel swin_mid (bmega* builds), 0 = none
  };
  // Bottleneck megakernel: b0 ... b5 in one launch (swin_bneck6 of a bmega* build, bit-identical to the base build),
  // swin_bneck / WG_SWIN_BNECK; default bmegahg with swin.co; "none" = six block launches.
  std::string bneck_variant;
  // Persistent kernel swin_mid (enc2 + merge, b0 ... b5, expand + dec2 in one launch; bmega* builds, bit-identical):
  // swin_mid / WG_SWIN_MID ("1" = bmegahg, or a bmega* build); off by default
  std::string mid_variant;
  int mid_threads = 256, mid_groups = 96;
  DevPtr mid_sync = 0;
  // Fused dec1 + dec0 (swin_block_c64_add_dec0 of dec1_dec0.co, bit-identical; dec1_dec0 / WG_DEC1_DEC0=fused)
  bool dec1_dec0 = false;
  // Native D3D12 fusions (backend 3 only; environment WG_D3D12_FUSION, a comma list, default
  // kDefaultD3d12Fusion; "none" = off), each bit-identical to the launches it replaces (kernels/fused_swin.hlsl):
  //   pm  enc1's patch_merge_c64 as the prologue of enc2's block (swin_block_c96_pm_nc reads enc1's map; enc1.merged
  //       is still written): one launch less
  //   xe  dec1's expand_c96 as the epilogue of dec2's fused block (swin_block_c96_add_xe_nc writes dec1's expanded
  //       map): one launch less
  //   mb  enc2's patch_merge_c96 as the first phase of the persistent bottleneck kernel (bneck/swin_bneck6_mb_nc,
  //       kernels/fused_bneck.hlsl: merge items ahead of b0's in its queue; enc2.merged is still written): one launch
  //       less; on top of pm,xe measured -6 us per frame (5 paired rounds, all faster; -10 us in a 3-round run)
  //       (mb0: diagnostics, the same kernel without its merge phase after a separate merge launch)
  // Measured together (1706x960 -> 2560x1440, warm, interleaved fresh processes): -14 us per frame (D3D12 timestamps,
  // median of 4 paired rounds; earlier 3-round runs -15 / -15 / -19 us), -18 us stream.
  bool fz_pm = false, fz_xe = false, fz_mb = false, fz_mb0 = false;
  //   rv  the encoder, enc2's fused block (pm) and dec2's (xe) with their workgroups in reverse order
  //       (the encoder itself runs in reverse order in every form since 6611e02, so rv no longer swaps it;
  //       swin_block_c96_pm_rev_nc / swin_block_c96_add_xe_rev_nc,
  //       fused_swin.hlsl FZ_REV): along the chain each kernel's first workgroups read the rows its producer wrote last
  //       (still in the L2); every workgroup does the same work, the same bytes. On top of pm,xe,mb measured -7 / -10 us
  //       per frame (5 / 6 paired rounds, the 6 all faster; -8 us in a 5-round run of the experiment build)
  bool fz_rv = false;
  //   xo  dec2's fused block (xe, forward) launched without the barrier after the bottleneck (mb), so that it overlaps
  //       the bottleneck's last items: swin_block_c96_add_xe_xo_nc waits per workgroup on the b5 window-row counters of
  //       bneck/swin_bneck6_mbo_nc (which leaves them for it to reset); needs xe and mb this frame
  bool fz_xo = false, fz_xo_now = false;
  //   xd  the fused dec1 + dec0 kernel (d1p12, pack12 frames) launched without the barrier after dec2's fused block (xo),
  //       so that it overlaps dec2's last workgroups: swin_block_c96_add_xe_xd_nc counts its windows in dec2's window
  //       rows after storing e1 (coherent stores), d1p12/swin_block_c64_add_dec0_xd_nc (kernels/fused_dec.hlsl) waits
  //       per workgroup on the rows of the e1 rows it gathers and resets them; needs xo this frame
  bool fz_xd = false, fz_xd_now = false;
  DevPtr xd_sync = 0;  // word 0: d1p12's done counter, 1: spin timeouts, 16 ..: dec2's window-row counters
  //       xd runs only on frames of at most xd_max_pixels render pixels (WG_D3D12_XD_MAX_PIXELS, default kXdMaxPixels;
  //       0 = any size): measured in one process against the same frames without it (runtime A/B, WG_AB_FUSION,
  //       3 x 4 windows of 300 frames each): 1280x720 -2.25 us, 1707x960 +5.0 us, 1920x1080 +7.5 us per frame
  long long xd_max_pixels = [] {
    const char* e = std::getenv("WG_D3D12_XD_MAX_PIXELS");
    return e && *e ? std::atoll(e) : kXdMaxPixels;
  }();
  // A/B diagnostics (WG_AB_*): the two fusion sets and the block length (0: off)
  struct FzSet {
    bool pm = false, xe = false, mb = false, mb0 = false, rv = false, xo = false, xd = false;
    template <class R> void apply(R* r) const {
      r->fz_pm = pm, r->fz_xe = xe, r->fz_mb = mb, r->fz_mb0 = mb0, r->fz_rv = rv, r->fz_xo = xo, r->fz_xd = xd;
    }
  };
  FzSet fz_set[2];
  int ab_block = 0;
  bool fz_any_xd = false;
  std::vector<std::string> nobar_ab, nobar_now;
  static std::vector<std::string> split_list(const std::string& s0) {
    std::vector<std::string> v;
    std::string t;
    for (char ch : s0 + ",") {
      if (ch == ',') { if (!t.empty()) v.push_back(t); t.clear(); } else t += ch;
    }
    return v;
  }
  bool parse_fusion(const std::string& text, FzSet* z) {
    *z = FzSet();
    for (size_t a = 0; text != "none" && a <= text.size();) {
      size_t e = text.find(',', a);
      if (e == std::string::npos) e = text.size();
      const std::string f = text.substr(a, e - a);
      a = e + 1;
      if (f == "pm") z->pm = true;
      else if (f == "xe") z->xe = true;
      else if (f == "mb") z->mb = true;
      else if (f == "mb0") z->mb = z->mb0 = true;
      else if (f == "rv") z->rv = true;
      else if (f == "xo") z->xo = true;
      else if (f == "xd") z->xd = true;
      else if (!f.empty()) return fail("WG_D3D12_FUSION: unknown fusion \"" + f + "\" (pm, xe, mb, rv, xo, xd or none)");
    }
    return true;
  }
  int xo_off[2][2] = {};  // this frame's bottleneck window offsets: even grid (b0), odd grid (b1, b3, b5)
  bool fz_pm_now = false, fz_xe_now = false, fz_mb_now = false;  // this frame
  std::string fz_text = "none";
  std::string d1d0_pre;
  int bneck_threads = 256, bneck_groups = 128;
  DevPtr bneck_sync = 0, bneck_tmp = 0;
  std::map<std::string, SwinGeo> stage_geo;
  std::string stage_variants_text;
  int filter_tile_x = 40, filter_s = 4;
  // post_fused* / post_fused3* (one launch for reconstruction + filter): full-history bytes per pixel (8 for the
  // post_recon path) and RGBA16F output (_o16) instead of 3 x f32
  bool post_fused = false, out_f16 = false, post_band = false, hist_planar = false;
  bool p12 = false;  // dec0 pack12 + post *_p12 (A: 8 bytes per render pixel, C: 4, B unused)
  bool p12_frame = false;  // this frame runs the pack12 pair (else: decoder.co + post_recon)
  // dec1 + dec0 fused writing pack12 (dec1_dec0__pack12.co, tag d1p12): on the pack12 frames, dec1's block runs with
  // dec0 as its epilogue; option d1d0_p12 (default 1), bit-identical to dec1 + d0p12/coef_decoder
  bool d1d0_p12 = false;
  bool d1d0_p12_opt = true;
  // dec0's divisions (kernels/decoder.hip, both bit-identical, default 1): option dec_rdiv = reciprocal + Markstein
  // correction (WG_DEC_RDIV), option dec_bdiv = the B softmax by one shared rcp(s) + an FMA correction (WG_DEC_BDIV).
  // Either 0: the pack12 modules decoder__pack12__div<r><b>.co and dec1_dec0__pack12__div<r><b>.co (r: dec_rdiv=0,
  // b: dec_bdiv=0); dec_div = those letters as loaded ("" = the default modules)
  bool dec_rdiv = true, dec_bdiv = true;
  std::string dec_div;
  std::string p12_path, dec0_name;  // pack12 decoder module (tagged "d0p12"); dec0 as reported
  int post_bands = 4;
  // True fusion (post_tf*, runtime option post_tf / WG_POST_TF, default 1): frames of the default banded pack12 path
  // whose size post_tf takes run one post_tf launch instead of the banded reconstruction + filter sequence.
  bool post_tf = true;
  bool output_quality9 = false;  // private Model M final-resampling quality profile; histories/weights unchanged
  // output_filter=quality5: the 5-tap output filter with WrongGPU's quality5 weights. The post kernels come from
  // post_w64__q5.co (post.hip built with WG_LZ_PROFILE=1: the same kernels with other filter constants), so every
  // path (true fusion included) runs as in the default, at the same cost.
  bool output_quality5 = false;
  // The output filter of spec 06 draft 4 (runtime option output_filter=original|analytic|quality5|quality9; default
  // or no option: "original" when the model file has the weights, else "analytic" with a note for the log).
  // original: the five weights of the model file (group filter / output_taps), rounded to FP16 once here, and FP16
  // accumulation. The post kernels come from post_w64__orig.co (post.hip built with WG_LZ_PROFILE=2: the same kernels,
  // each filtering one with the five weights as trailing arguments), so every path runs as with the constants.
  // analytic: spec 06's own Lanczos set, the constants of post_w64.co / post.co (what "default" was before draft 4).
  bool output_original = false;
  std::string output_filter_opt;  // the option as given ("" = none)
  uint32_t lz_bits[5] = {0, 0, 0, 0, 0};  // FP16 bit patterns of the weights w(-2) ... w(+2), even output index
  std::string notes_text;         // lines for the front end's log at create (runtime_notes)
  // Motion vectors at output resolution (runtime option motion_outres=original|point; no option: frames never use
  // FrameIo.motion_hi and nothing is loaded for it). 1 = "original": the kernels of encoder_mo.co / post_mo_w64*.co
  // (spec 03 draft 4, spec 06 section 3) run on frames that bring the raw motion; 2 = "point": such frames run on the
  // render-size motion as before (the interop's point resampling, our earlier stand-in).
  int motion_outres = 0;
  bool mo_last = false;           // the last frame ran the output-resolution-motion kernels
  // test path (wg_rt_upload_motion_hi): own buffer and parameters, used by every frame while hi_w > 0
  DevPtr hi_buf = 0;
  size_t hi_cap = 0;
  int hi_w = 0, hi_h = 0, hi_wide = 0;
  float hi_scale[2] = {1, 1}, hi_b[2] = {0, 0};
  bool pi_original = false;  // pre_exposure=original: pre-exposure only in the exposure measurement (see frame_body)
  int post_tf_cpairs = kTfCpairs;
  // native D3D12 backend (backend 3): frames that post_tf does not take at cap 352 run post_tf4_dl_* (tf4_supported);
  // runtime option post_tf4=force (diagnostics, equality checks): every post_tf frame that tf4_supported takes runs it
  bool d3d12_native = false, post_tf4_force = false;
  bool recon_out16 = false, frame_out_f16 = false;  // this frame: D = O output by post_recon2_r_out (RGBA16F)
  int hist_bytes = 8;
  int post_wave = 64;        // 64: post kernels from post_w64.co (tag "w64"), 32: from post.co
  // swin_resident=N (runtime option / WG_SWIN_RESIDENT): the plain C = 64 block launches (enc1; dec1 when not fused)
  // run the weight-resident persistent kernels swin_block_c64[_add]_r[_nc] with N workgroups of 512 lanes; 0 = off
  int swin_resident = 0;
  bool post_diet = true;     // option post_diet (default 1; 0 = the previous kernels): exposure_pair_diet + the _diet post kernels (kernels/post.hip); bit-identical
  // On top of post_diet, the true-fusion kernel post_tf_*_diet_<suffix> (kernels/post.hip WG_TF_KERNEL_DIET_X; all three
  // off = the plain post_tf_*_diet), each bit-identical and on by default:
  bool tf_vdiet = true;      // option tf_vdiet (suffix v): the vertical pass diet
  bool post_sel = true;      // option post_sel (suffix s): candidate motions by selects (post_select_m)
  bool post_idx = true;      // option post_idx (suffix i): unclamped, pre-scaled LDS texel offsets
  bool post_trim = true;     // option post_trim (suffix t, on top of vsi only): in_range as bit compares, no canonicalise before the Dec clamp, uniform address products on the SALU
  // runtime options (runtime_options, then WG_RUNTIME_OPTIONS: "key=value;..."; other keys belong to the interop)
  bool use_graph = false, high_priority = false;
  std::string options_text;
  std::string post_pre;      // "w64/" or ""
  int max_rw = 0, max_rh = 0, dw = 0, dh = 0, ow = 0, oh = 0, hw = 0, hh = 0;
  std::map<std::string, std::vector<uint8_t>> blobs;
  std::map<std::string, DevPtr> wdev;
  float post_s = 0.0f;
  // The model's output-filter weights w(-2) ... w(+2) for an even output index (spec 06 draft 4, spec 08 group
  // filter / output_taps), when the model data has them.
  bool has_taps = false;
  float output_taps[5] = {0, 0, 0, 0, 0};
  // model.wgcm was converted with another version of the map than the one shipped here (accepted: every stage equals
  // this map's digest; load_wgcm_model). Such a file has no output filter weights.
  bool model_other_map = false;
  std::string model_source;
  std::vector<DevPtr> owned;
  uint64_t owned_bytes = 0;
  bool external_io = false;    // inputs and output always come from FrameIo (interop): no own buffers for them
  bool native_direct_io = false;  // typed resources bound by the private native caller; no dense I/O conversion buffers
  bool recon_output = false;   // experimental: D = O, output filter = identity (post_copy)

  // buffers
  DevPtr color = 0, depth = 0, motion = 0, expo = 0, expo_partials = 0, expo_computed = 0, feat0 = 0;
  DevPtr expo_sync = 0;  // exposure_auto_diet's workgroup counter (zero between launches)
  // Direct resources: the per-frame 64K tables of the direct kernels (clean/dx12/hlsl/wg_f16lut.hlsli, wg_f16lut_in.hlsli:
  // 256-byte header, zero at create, then the post output table and the encoder colour table), written
  // by f16lut_build_direct after the exposure step and read by the direct encoder and post kernels. The direct kernels do
  // not read a colour buffer (they read the game's texture), so it is passed as their colour pointer.
  DevPtr f16lut = 0;
  static constexpr size_t kF16LutBytes = 256 + 65536 * 12;
  DevPtr mnear[2] = {0, 0}, mfar = 0, mcenter = 0, latent[2] = {0, 0}, reduced = 0;
  DevPtr skip1 = 0, merged1 = 0, skip2 = 0, merged2 = 0, bout[6] = {0}, e2 = 0, out2 = 0, e1 = 0, out1 = 0;
  DevPtr A = 0, B = 0, C = 0, hist[2] = {0, 0}, display = 0, out = 0, color_net = 0;
  DevPtr prep = 0;  // post_prep records (post_recon5*, 6*): 32 bytes per render pixel; allocated only for those
  std::vector<float> staging;

  // frame state (spec 10 section 3)
  int frames = 0, n = 0, f = 0, last_rw = 0, last_rh = 0;
  float jprev[2] = {0, 0};
  WgRtFrameInfo info = {};

  bool fail(const std::string& m) {
    err = m;
    return false;
  }
  bool dfail() { return fail(dev->err()); }

  bool alloc(DevPtr* p, size_t bytes) {
    if (!dev->alloc(p, bytes)) return dfail();
    owned.push_back(*p);
    owned_bytes += bytes;
    return true;
  }

  ~Runtime() {
    if (dev) {
      dev->sync();
      for (DevPtr p : owned) dev->free(p);
    }
  }

  bool create(const WgRtConfig& c, std::unique_ptr<Device> given = nullptr, bool external = false,
              bool native_direct = false) {
    cfg = c;
    external_io = external;
    native_direct_io = native_direct;
    build_dir = c.build_dir ? c.build_dir : ".";
    ModelId model_id = ModelId::M;  // runtime option model=K (clean/k/runtime/model_k.h); default model M
    if (std::string merr; !model_from_options(c, &model_id, &merr)) return fail(merr);
    if (c.backend < 0 || c.backend > 3) return fail("unknown runtime backend");
    // model_select.h (Model M); Model L on native D3D12 in builds with it (its missing D3D12 ports are refused below)
    const bool native_ok = model_info(model_id).native_d3d12 || (model_id == ModelId::L && model_l_built());
    if (native_direct_io && (!external_io || c.backend != 3 || !native_ok))
      return fail("native direct resources require external_io, backend 3 and " + native_d3d12_models());
    if (c.backend == 3 && !native_ok) return fail("native D3D12 supports " + native_d3d12_models() + " only");
    d3d12_native = c.backend == 3;
    if (c.backend == 3 && external && !native_direct_io)
      return fail("native D3D12 external I/O requires the private direct-resource route");
#ifndef WG_NO_MODEL_K
    if (model_id == ModelId::K) return create_k(c, std::move(given));
#else
    if (model_id == ModelId::K) return fail("Model K is not part of this build");  // (model_from_options says so first)
#endif
    model_l = model_id == ModelId::L;  // (model_from_options accepts it only in builds with WG_MODEL_L)
    if (model_l && (c.backend == 1 || c.backend == 2))
      return fail("Model L runs on the HIP backend (backend 0) and native D3D12 (backend 3) only");
#ifdef WG_MODEL_L
    if (model_l && c.backend == 3) {
      const std::string missing = l_native_missing();
      if (!missing.empty()) return fail("Model L on native D3D12: not yet ported to D3D12: " + missing);
    }
#endif
    if (c.max_rw <= 0 || c.max_rh <= 0 || c.ow <= 0 || c.oh <= 0) return fail("invalid render or output size");
    // post_core's fallback colour read uses a 32-bit byte offset (mul24, kernels/post.hip): the render colour image
    // must stay within 4 GB (16 rw rh <= 2^32, 16 rw < 2^24; any D3D12 texture, at most 16384 x 16384, does)
    if (16LL * c.max_rw * c.max_rh > (1LL << 32) || 16LL * c.max_rw >= (1LL << 24))
      return fail("render size too large (colour image over 4 GB)");
    max_rw = c.max_rw;
    max_rh = c.max_rh;
    ow = c.ow;
    oh = c.oh;
    recon_output = recon_output_requested(c);
    dw = recon_output ? ow : (3 * ow + 1) / 2;  // experimental recon=output: D = O
    dh = recon_output ? oh : (3 * oh + 1) / 2;
    hw = dw / 2;
    hh = dh / 2;
    for (int k = 0; k < 5; ++k) {
      if (c.swin_threads) block_threads[k] = c.swin_threads;
      if (c.block_threads[k]) block_threads[k] = c.block_threads[k];
      block_wpg[k] = c.block_wpg[k];
    }
    if (c.gemm_threads) gemm_threads = c.gemm_threads;
    if (c.nt) nt = c.nt;
    if (c.wpw64) wpw[0] = c.wpw64;
    if (c.wpw96) wpw[1] = c.wpw96;
    if (c.wpw128) wpw[2] = c.wpw128;
    if (!choose_kernels(c)) return false;
    if (model_l && recon_output) return fail("Model L: recon=output (D = O) is Model M's experimental mode");
    // Model L's reconstruction kernel l_post_recon is HDR; its LDR form (11/11/10-bit display codes) is not written yet.
    // (Native D3D12 runs Model M's pack12 post for Model L, which takes both.)
    if (model_l && !c.hdr && c.backend != 3)
      return fail("Model L: LDR colour (WgRtConfig.hdr = 0) needs Model L's LDR reconstruction kernel, which is not "
                  "written yet (l_post_recon is HDR)");
    if (model_l && enc0_co != "encoder.co")
      return fail("Model L: enc0 variants are Model M's encoder; Model L's runs from l_enc_values.co");
    // Model L's coefficient kernel (l_coef) writes Model M's coef_decoder formats, not pack12: the _p12 post kernels
    // are not for it (choose_kernels does not choose pack12 by default for Model L; asked for, it is refused here)
    // On native D3D12 Model L stores its coefficients in pack12 (l_coef_pack12, lossless) for Model M's pack12 post
    // path, the only post path ported there; on HIP it writes the base format (l_coef) for the post_recon path.
    if (model_l && p12 && c.backend != 3)
      return fail("Model L: post_pack=p12 (the pack12 decoder) is Model M's; Model L's l_coef writes the base "
                  "coefficient format (pack12 is Model L's native D3D12 form only)");
    if (model_l && !p12 && c.backend == 3)
      return fail("Model L on native D3D12 needs the pack12 post path (post_pack=p12; Model M's decoder__pack12.co "
                  "selects it)");
    bool any = false;
    for (int i = 0; i < 16; ++i) any |= c.d_row[i] != 0;
    drow = drow_gfx12();
    if (any)
      for (int i = 0; i < 16; ++i) drow[i / 8][i % 8] = c.d_row[i];
    if (native_direct_io && drow != drow_gfx12())
      return fail("native direct resources require the ported gfx12 WMMA accumulator layout");

    // model data (section 7): load, check, pack once
    std::map<std::string, StageTensors> model;
    std::string e;
    if (!c.model_dir) return fail("no model directory");
    const std::string wgcm = std::string(c.model_dir) + "/model.wgcm";
    bool use_wgcm = c.model_format == 2;
    if (c.model_format == 0 && !model_l) {
      std::FILE* fp = std::fopen(wgcm.c_str(), "rb");
      use_wgcm = fp != nullptr;
      if (fp) std::fclose(fp);
    }
    if (model_l) {
#ifdef WG_MODEL_L
      // Model L: the converter's model.wgcm in its folder under the model directory (model_select.h), checked against
      // the preset-L map (whole-file hash, every stage's keyed digest, names, types, shapes, values). Raw tensors are
      // refused: nothing in them shows that they are Model L's (a folder with Model M's shared stages would load).
      if (c.model_format == 1)
        return fail("Model L reads only the converter's model.wgcm (raw tensors cannot be verified as Model L's)");
      const std::string dir = std::string(c.model_dir) + "/" + model_info(ModelId::L).model_subdir;
      if (!l::load_wgcm_model(dir + "/model.wgcm", l_map_path(c), &model, &e, &model_other_map)) return fail(e);
      use_wgcm = true;
#endif
    } else if (use_wgcm) {
      if (!c.map_path) return fail("model.wgcm needs the converter map (map_path)");
      if (!load_wgcm_model(wgcm, c.map_path, &model, &e, &model_other_map)) return fail(e);
    } else if (!load_raw_model(c.model_dir, &model, &e)) {
      return fail(e);
    }
    model_source = use_wgcm ? "model.wgcm" : "raw";
    if (c.swin_force_clamp) force_clamp = true;
    if (c.swin_fuse) fuse = true;
    for (const std::string& s : model_stages()) {
      if (s == "post" || (model_l && (s == "enc0" || s == "dec0"))) continue;  // Model L's: below
      if (!pack_stage_blob(s, model[s], drow, &blobs[s], &e)) return fail(e);
    }
#ifdef WG_MODEL_L
    if (model_l) {  // enc0.values, enc0.Win1, enc0.Win2, enc0_block, enc0.project, dec0.project, dec0_block, dec0.coef
      std::map<std::string, std::vector<uint8_t>> lb;
      if (!l::pack_model(model, drow, &lb, &e)) return fail(e);
      for (auto& kv : lb) blobs[kv.first] = std::move(kv.second);
      for (int i = 0; i < 2; ++i) {
        uint32_t flags = 0;
        std::memcpy(&flags, blobs[i ? "dec0_block" : "enc0_block"].data(), 4);
        l_nc[i] = (flags & 1u) != 0;
        if (!l_nc[i] && c.backend == 3)
          return fail(std::string("Model L on native D3D12: ") + (i ? "dec0_block" : "enc0_block") +
                      "'s weights need the clamped block, which is not ported to D3D12 (only the _nc forms are)");
      }
    }
#endif
    // kernels/swin.hip: the first word of a Swin stage's packed weights says whether the clamp-free block kernel
    // (`<name>_nc`, identical results) may be used.
    for (const char* s : kSwinOrder) {
      uint32_t flags = 0;
      std::memcpy(&flags, blobs[s].data(), 4);
      swin_nc[s] = (flags & 1u) != 0;
    }
    post_s = model["post"]["s"].f[0];
    blobs["post"].assign(16, 0);                       // runner/post.py pack: s as f32, zero padding to 16 bytes
    std::memcpy(blobs["post"].data(), &post_s, 4);
    // optional group filter / output_taps (spec 08 draft 2): the five weights as loaded (blob "filter": 5 x f32, host
    // side only; wg_rt_stage_blob reports "no blob" when the model has none)
    has_taps = model_output_taps(model, output_taps);
    if (has_taps) {
      blobs[kFilterStage].resize(sizeof(output_taps));
      std::memcpy(blobs[kFilterStage].data(), output_taps, sizeof(output_taps));
    }

    // device and code objects
    if (given) {
      dev = std::move(given);
    } else if (c.backend == 1) {
      const std::string lib = c.emu_library ? c.emu_library : build_dir + "/wg_emu.so";
      dev = make_emu_device(lib, &e);
    } else if (c.backend == 2) {  // hybrid D3D12: the D3D12 ports of <build_dir>/dx12 + HIP (device_dx12.cpp)
      dev = make_dx12_device(c.rocm_dir ? c.rocm_dir : "", build_dir, &e);
    } else if (c.backend == 3) {
#if defined(_WIN32)
      dev = make_d3d12_device(build_dir, &e);
#else
      return fail("native D3D12 requires Windows");
#endif
    } else {
      dev = make_hip_device(c.rocm_dir ? c.rocm_dir : "", &e);
    }
    if (!dev) return fail(e);
    for (const std::string& co : {swin_co, enc0_co, dec0_co, exposure_co, std::string("post.co"),
                                  std::string("frame.co")}) {
      if (model_l && (co == enc0_co || co == dec0_co)) continue;  // Model L's encoder and decoder: below
      if (!dev->load_module(build_dir + "/" + co))
        return fail(dev->err() + (co.find("__") != std::string::npos
                                      ? " (variant code objects: clean/runner/build.py --variants / --perf-variants)"
                                      : ""));
    }
    // Model L's code objects (clean/l/FORMATS.md): the encoder's values under the tag lev (lev/input_encoder, next to
    // Model M's input_encoder), the linears / projection / channel copy and the width-32 block untagged
    if (model_l && (!dev->load_module_tagged(build_dir + "/l_enc_values.co", "lev") ||
                    !dev->load_module(build_dir + "/l_ops.co") || !dev->load_module(build_dir + "/l_swin.co") ||
                    !dev->load_module(build_dir + "/l_post.co")))
      return fail(dev->err() + " (Model L's code objects l_enc_values.co, l_ops.co, l_swin.co, l_post.co)");

    if (!stage_variants(c)) return false;
    if (!runtime_options(c)) return false;
    if (model_l) {
      // Model L reads dec1's output map (dec1.out) in its decoder front: not the paths that keep it on chip
      if (motion_outres == 1) return fail("Model L: motion_outres=original needs Model M's encoder (encoder_mo.co)");
      d1d0_p12_opt = false;
    }
    if (!mid_setup(c)) return false;
    if (!bneck_setup(c)) return false;
    // D = O (fast mode, recon=output) runs without pack12, so dec1 + dec0 fused is the default there (bit-identical;
    // measured -0.027 ms per 1440p-Q frame); the D = 1.5 x path keeps pack12 and the separate kernels.
    if (choose(c.dec1_dec0, "WG_DEC1_DEC0", recon_output && !p12 ? "fused" : "separate") == "fused") {
      if (model_l) return fail("Model L: WG_DEC1_DEC0=fused keeps dec1's map on chip; its decoder front reads it");
      if (stage_geo.count("dec1")) return fail("WG_DEC1_DEC0=fused together with a per-stage variant of dec1");
      if (block_threads[1] != 256 || wpw[0] != 1 || (block_wpg[1] > 0 && block_wpg[1] != 8))
        return fail("WG_DEC1_DEC0=fused needs dec1 at 256 lanes with one window per wave (the base geometry)");
      if (c.backend != 1) {  // HIP: its own code object (it also holds swin.hip's kernels under the same names)
        if (!dev->load_module_tagged(build_dir + "/dec1_dec0.co", "d1d0")) return dfail();
        d1d0_pre = "d1d0/";
      }
      dec1_dec0 = true;
    }
    if (p12) {  // the pack12 decoder next to decoder.co (the fallback frames' decoder)
      if (dec1_dec0) return fail("WG_DEC1_DEC0=fused writes the base dec0 format, not pack12 (WG_POST_PACK=p12)");
      if (c.backend != 1) {  // dec_rdiv=0 / dec_bdiv=0: the pack12 decoder modules with those divisions as before
        dec_div = std::string(dec_rdiv ? "" : "r") + (dec_bdiv ? "" : "b");
        if (!dec_div.empty()) p12_path = build_dir + "/decoder__pack12__div" + dec_div + ".co";
      }
      if (!dev->load_module_tagged(p12_path, "d0p12"))
        return fail(dev->err() + " (dec0 pack12: clean/runner/build.py builds decoder__pack12.co)");
      // the fused dec1 + dec0 in pack12 form, when dec1 runs in the base geometry
      const std::string dsfx = dec_div.empty() ? "" : "__div" + dec_div;
      const std::string d1p = build_dir + "/dec1_dec0__pack12" + dsfx + ".co";
      if (d1d0_p12_opt && c.backend != 1 && !stage_geo.count("dec1") && block_threads[1] == 256 && wpw[0] == 1 &&
          !(block_wpg[1] > 0 && block_wpg[1] != 8) && module_available(d1p)) {
        if (!dev->load_module_tagged(d1p, "d1p12")) return dfail();
        d1d0_p12 = true;
      }
    }
    if (output_original) {
      // "original" needs its own build of the post kernels (wave64, HIP). Asked for by name: an error without it;
      // chosen by default: the analytic filter instead, with a note.
      const bool have = post_wave == 64 && c.backend != 1 && module_available(build_dir + "/post_w64__orig.co");
      if (c.backend == 3 && !have) return fail("native D3D12 unsupported post_w64__orig.co route");
      if (!have && output_filter_opt == "original")
        return fail("output_filter=original needs the wave64 post code object post_w64__orig.co (HIP backend)");
      if (!have) {
        output_original = false;
        notes_text = "output filter: analytic (the model file has the original's weights, but this build has no "
                     "post_w64__orig.co for them)";
      }
    }
    if (post_wave == 64) {
      const std::string w64 = build_dir + (output_quality5   ? "/post_w64__q5.co"
                                           : output_original ? "/post_w64__orig.co"
                                                             : "/post_w64.co");
      if (c.backend == 3 && !module_available(w64))
        return fail("native D3D12 unsupported code-object variant: " + w64 + "; no HIP fallback");
      if (c.backend != 1 && module_available(w64)) {
        if (!dev->load_module_tagged(w64, "w64")) return dfail();
        post_pre = "w64/";
      } else {
        post_wave = 32;  // emulator, or a build without post_w64.co
      }
    }
    if (output_quality5 && post_wave != 64)
      return fail("output_filter=quality5 needs the wave64 post code object post_w64__q5.co (HIP backend)");
    if (motion_outres == 1) {
      // motion_outres=original: the input encoder and the reconstruction kernels for motion at output resolution, in
      // code objects of their own (tags "emo", "pmo"; the post one of the output-filter build in use). They stand in
      // for the default kernels only (encoder.co, the pack12 post path in wave64); otherwise, or without the files,
      // such frames keep the render-size motion (a note says so).
      const std::string emo = build_dir + "/encoder_mo.co";
      const std::string pmo = build_dir + (output_quality5   ? "/post_mo_w64__q5.co"
                                           : output_original ? "/post_mo_w64__orig.co"
                                                             : "/post_mo_w64.co");
      const char* why = c.backend == 1                                  ? "the emulator backend"
                        : enc0_co != "encoder.co"                       ? "an enc0 variant is selected"
                        : !p12 || post_wave != 64 || post_fused         ? "the post path is not the default pack12 one"
                        : !module_available(emo) || !module_available(pmo) ? "this build has no encoder_mo.co / post_mo_w64*.co"
                                                                        : nullptr;
      if (why) {
        if (c.backend == 3) return fail(std::string("native D3D12 unsupported motion_outres=original: ") + why);
        motion_outres = 2;
        notes_text += std::string(notes_text.empty() ? "" : "\n") +
                      "motion at output resolution: resampled to render pixels as before (motion_outres=original "
                      "not available: " + why + ")";
      } else if (!dev->load_module_tagged(emo, "emo") || !dev->load_module_tagged(pmo, "pmo")) {
        return dfail();
      }
    }
    // dec1 and dec2 (dec2 measured 2026-10-07: -0.012 ms per 1440p-Q frame, bit-identical)
    fuse_stages_text = choose(c.swin_fuse_stages, "WG_SWIN_FUSE_STAGES", swin_co == "swin.co" ? "dec1,dec2" : "none");
    if (fuse_stages_text != "none") {
      for (size_t a = 0; a <= fuse_stages_text.size();) {
        size_t b = fuse_stages_text.find_first_of(",/", a);
        if (b == std::string::npos) b = fuse_stages_text.size();
        const std::string st = fuse_stages_text.substr(a, b - a);
        a = b + 1;
        if (st.empty()) continue;
        const SwinStageInfo* si = swin_stage_info(st);
        if (!si || si->kind == 'b')
          return fail("swin fuse stages: \"" + st + "\" is not an encoder or decoder stage");
        fuse_stages.push_back(st);
      }
    }
    if (c.backend == 3) {
      fz_text = choose(nullptr, "WG_D3D12_FUSION", kDefaultD3d12Fusion);
      if (!parse_fusion(fz_text, &fz_set[0])) return false;
      fz_set[1] = fz_set[0];
      // Diagnostics (A/B in one process): WG_AB_FUSION / WG_AB_NOBAR apply on the frames of every odd block of
      // WG_AB_BLOCK frames (default 300, the interop timing log's window), WG_D3D12_FUSION / WG_D3D12_NOBAR on the others.
      const char* abf = std::getenv("WG_AB_FUSION");
      const char* abn = std::getenv("WG_AB_NOBAR");
      const char* abb = std::getenv("WG_AB_BLOCK");
      const char* abk = std::getenv("WG_AB_KSUB");
      if ((abf && *abf) || (abn && *abn) || (abk && *abk)) {
        ab_block = abb && *abb ? std::max(1, std::atoi(abb)) : 300;
        if (abf && *abf && !parse_fusion(abf, &fz_set[1])) return false;
        nobar_ab = split_list(abn ? abn : "");
        if (!(abn && *abn)) nobar_ab = nobar_list;
      }
      fz_set[0].apply(this);
      fz_any_xd = fz_set[0].xd || fz_set[1].xd;
    }
    // the fused dec1 + dec0 kernel takes dec1's block, so dec1's expand stays a separate launch
    if (dec1_dec0)
      fuse_stages.erase(std::remove(fuse_stages.begin(), fuse_stages.end(), std::string("dec1")), fuse_stages.end());
    if (native_direct_io && !validate_native_direct_config()) return false;
    for (auto& kv : blobs) {
      if (kv.first == kFilterStage) continue;  // the output-filter weights are not read from device memory
      DevPtr p;
      if (!alloc(&p, kv.second.size()) || !dev->upload(p, kv.second.data(), kv.second.size())) return dfail();
      wdev[kv.first] = p;
    }
    return alloc_buffers() && dev->sync() ? true : (err.empty() ? dfail() : false);
  }

#ifndef WG_NO_MODEL_K
  // Model K (clean/k/runtime/runtime_k.h): the device as for model M, then the K state; model M's state is not made.
  bool create_k(const WgRtConfig& c, std::unique_ptr<Device> given) {
    if (c.max_rw <= 0 || c.max_rh <= 0 || c.ow <= 0 || c.oh <= 0) return fail("invalid render or output size");
    max_rw = c.max_rw;
    max_rh = c.max_rh;
    ow = dw = c.ow;
    oh = dh = c.oh;
    std::string e;
    if (given) {
      dev = std::move(given);
    } else if (c.backend == 1) {
      dev = make_emu_device(c.emu_library ? c.emu_library : build_dir + "/wg_emu.so", &e);
    } else if (c.backend == 2) {
      dev = make_dx12_device(c.rocm_dir ? c.rocm_dir : "", build_dir, &e);
    } else {
      dev = make_hip_device(c.rocm_dir ? c.rocm_dir : "", &e);
    }
    if (!dev) return fail(e);
    k.reset(new KRuntime());
    if (!k->create(c, dev.get(), [this](DevPtr* p, size_t n) { return alloc(p, n); }, external_io))
      return fail(k->err());
    blobs = k->blobs();
    model_source = "model.wgcm";
    return true;
  }
#endif

  // Model L's converter map: map_path itself when it names that map's file, else that file next to map_path (or in
  // the build folder without one): the maps are shipped side by side next to the runtime library
  // (clean/runner/build.py).
  std::string l_map_path(const WgRtConfig& c) const {
    const std::string file = model_info(ModelId::L).map_file;
    const std::string given = c.map_path ? c.map_path : "";
    const size_t slash = given.find_last_of("\\/");
    if (!given.empty() && given.substr(slash == std::string::npos ? 0 : slash + 1) == file) return given;
    return (slash == std::string::npos ? build_dir : given.substr(0, slash)) + "/" + file;
  }

  // Native route selection uses manifest identities, never HIP .co file presence. The fused pack12 decoder supplies
  // the selected path's pack12 capability; its standalone decoder fallback is still rejected at launch if reached.
  bool module_available(const std::string& path) const {
    if (cfg.backend != 3) return wg_file_exists(path);
    std::string co = path.substr(path.find_last_of("/\\") + 1);
    if (co == "decoder__pack12.co") co = "dec1_dec0__pack12.co";
    std::string text, error;
    wgc::Json manifest;
    if (!read_text(build_dir + "/dx12/manifest.json", &text) || !wgc::parse_json(text, manifest, error)) return false;
    const auto* kernels = manifest.find("kernels");
    if (!kernels || !kernels->is_array()) return false;
    for (const auto& k : kernels->items) {
      const auto* identity = k.find("co");
      if (identity && identity->is_string() && identity->text == co) return true;
    }
    return false;
  }

  // Kernel choice per stage (WgRtConfig fields, environment variables, defaults; wg_runtime.h).
  bool choose_kernels(const WgRtConfig& c) {
    if (c.swin_code_object && *c.swin_code_object) {
      swin_co = c.swin_code_object;
    } else {
      const std::string v = choose(nullptr, "WG_SWIN_VARIANT", "base");
      if (v != "base" && !swin_variant(v)) return false;
    }
    const std::string ev = choose(c.enc0_variant, "WG_ENC0_VARIANT", "base");
    std::string dv = choose(c.dec0_variant, "WG_DEC0_VARIANT", "base");
    post_kernel = choose(c.post_kernel, "WG_POST_KERNEL", kDefaultPost);
    {
      const bool post_given = (c.post_kernel && *c.post_kernel) || choose(nullptr, "WG_POST_KERNEL", "") != "";
      const std::string fk = choose(c.filter_kernel, "WG_FILTER_KERNEL", kDefaultFilter);
      const bool filter_unbanded = fk != kDefaultFilter && fk.find("_band") == std::string::npos;
      if (!post_given && !filter_unbanded && !recon_output) post_kernel = kDefaultBandPost;
    }
#ifdef WG_MODEL_L
    // Model L reconstructs with its own kernel, l_post_recon (l_post.co; clean/l/FORMATS.md step 5), which takes Model
    // M's post_recon arguments and buffers: the unbanded post_recon route (8-byte history, base coefficient format,
    // no pack12), then Model M's output filter (unbanded). Model M's post kernels are not for Model L. On native D3D12
    // Model L's frames run Model M's pack12 post path for now (l_coef_pack12 feeds it); frames that path does not take
    // run l_post_recon there too (its D3D12 port, then the analytic filter).
    if (model_l && c.backend != 3) {
      if ((c.post_kernel && *c.post_kernel) || choose(nullptr, "WG_POST_KERNEL", "") != "")
        return fail("Model L: its reconstruction is l_post_recon; a post kernel (post_kernel / WG_POST_KERNEL) "
                    "chooses among Model M's");
      if (choose(c.post_pack, "WG_POST_PACK", "none") == "p12")
        return fail("Model L: post_pack=p12 (the pack12 decoder) is Model M's; Model L's l_coef writes the base "
                    "coefficient format");
      post_kernel = "post_recon";
    }
#endif
    // post_pack / WG_POST_PACK=p12: the _p12 form of the post kernel (post_recon2_r[_hp][_band] -> ..._p12). Default
    // p12 (measured in game with the banded path: -0.04 ms, exact) when nothing else is chosen for post / dec0, not
    // with recon=output or the fused dec1 + dec0, and only when the pack12 decoder is there (build.py builds it).
    p12_path = c.backend == 1 ? emu_lib_dir(c) + "/wg_emu__pack12.so" : build_dir + "/decoder__pack12.co";
    bool p12_default = false;
    {
      const bool post_given = (c.post_kernel && *c.post_kernel) || choose(nullptr, "WG_POST_KERNEL", "") != "";
      const bool dec0_given = (c.dec0_variant && *c.dec0_variant) || choose(nullptr, "WG_DEC0_VARIANT", "") != "";
      p12_default = (!model_l || c.backend == 3) && module_available(p12_path) && !post_given && !dec0_given &&
                    !recon_output &&
                    choose(c.dec1_dec0, "WG_DEC1_DEC0", "separate") != "fused";
    }
    const std::string pack = choose(c.post_pack, "WG_POST_PACK", p12_default ? "p12" : "none");
    if (pack == "p12" && post_kernel.find("_p12") == std::string::npos) {
      bool ok = false;
      for (const char* k : {"post_recon2_r", "post_recon2_r_hp", "post_recon2_r_band", "post_recon2_r_hp_band"})
        ok = ok || post_kernel == k;
      if (!ok) return fail("WG_POST_PACK=p12 needs post_recon2_r[_hp][_band] (post kernel " + post_kernel + ")");
      post_kernel += "_p12";
    } else if (pack != "p12" && pack != "none") {
      return fail("unknown post pack " + pack + " (p12 or none)");
    }
    // dec0 "pack12" (A, B, C in 12 bytes: image A8 + image C4) goes with the _p12 post kernels; the runtime switches
    // them together (a _p12 post kernel selects dec0 pack12). The pack12 decoder is a second module (tagged "d0p12");
    // frames the _p12 kernels do not take (render footprint, recon2_supported) run decoder.co's coef_decoder and
    // post_recon.
    p12 = post_kernel.find("_p12") != std::string::npos;
    if (p12 && dv == "base") dv = "pack12";
    if (p12 != (dv == "pack12"))
      return fail("dec0 pack12 and the _p12 post kernels (post_recon2_r_p12, ...) are used together");
    if (p12) dv = "base";  // decoder.co is the module of the fallback frames
    if (dv == "v1") return fail("dec0 variant v1 needs the version-1 tables; the runtime packs the current ones");
    enc0_co = ev == "base" ? "encoder.co" : "encoder__" + ev + ".co";
    dec0_co = dv == "base" ? "decoder.co" : "decoder__" + dv + ".co";
    dec0_name = p12 ? "decoder__pack12.co" : dec0_co;
    exposure_variant = choose(c.exposure_variant, "WG_EXPOSURE_VARIANT", kDefaultExposure);
    exposure_co = exposure_variant == "base" ? "exposure.co" : "exposure__" + exposure_variant + ".co";
    // enc0 / dec0 builds that change a buffer format need a matching runtime / output stage (not wired here)
    for (const char* v : {"colorh", "motionh", "in16"})
      if (ev == v)
        return fail("enc0 variant " + ev + " reads a different input format; the runtime does not provide it");
    post_fused = post_kernel.compare(0, 10, "post_fused") == 0;
    if (!post_fused && post_kernel.compare(0, 10, "post_recon") != 0)
      return fail("post kernel " + post_kernel + " is neither a post_recon nor a post_fused variant");
    if (post_kernel.size() > 4 && post_kernel.compare(post_kernel.size() - 4, 4, "_out") == 0)
      return fail("post kernel " + post_kernel + " belongs to the output mode D = O (recon = output), not wired here");
    // full-history format by name (FORMATS.md): _h4 compact, _h6 RGB16F, _h8 RGBA16F; post_fused* default 6 bytes
    hist_planar = post_kernel.find("_hp") != std::string::npos;  // planar 6-byte history (R, G | B planes)
    hist_bytes = post_kernel.find("_h4") != std::string::npos   ? 4
                 : post_kernel.find("_h6") != std::string::npos || hist_planar ? 6
                 : post_kernel.find("_h8") != std::string::npos ? 8
                 : post_fused                                    ? 6
                                                                 : 8;
    filter_kernel = choose(c.filter_kernel, "WG_FILTER_KERNEL", kDefaultFilter);
    // banded two-kernel path (post_recon2_r_band / _hp_band + post_filter_c1_s*_band): launched alternately over
    // horizontal bands (FORMATS.md); bands from runtime option post_bands / WG_POST_BANDS (default 4)
    post_band = post_kernel.find("_band") != std::string::npos;
    const bool filter_band =
        filter_kernel.size() > 5 && filter_kernel.compare(filter_kernel.size() - 5, 5, "_band") == 0;
    if (post_band && !filter_band) {
      if (filter_kernel != kDefaultFilter)
        return fail("post kernel " + post_kernel + " needs a banded filter (post_filter_c1_s4_band / _s2_band)");
      filter_kernel = "post_filter_c1_s4_band";
    }
    if (filter_band && !post_band) return fail("filter " + filter_kernel + " needs a banded post kernel (*_band)");
    if (post_band && recon_output) return fail("the banded post path does not run with recon=output (D = O)");
    if (!post_fused && !filter_shape(filter_kernel, &filter_tile_x, &filter_s))
      return fail("unknown output filter kernel " + filter_kernel);
    if (recon_output && post_fused)
      return fail("post kernel " + post_kernel + " filters; recon=output (D = O) needs a post_recon kernel");
    // RGBA16F output (8 bytes per pixel) from the fused kernel or the filter (D = O: post_copy writes 3 x f32)
    out_f16 = !recon_output && (post_fused ? post_kernel : filter_kernel).find("_o16") != std::string::npos;
    // post kernels in wave64 (post_w64.co: post.hip built with -mwavefrontsize64; bit-identical, faster on gfx1201);
    // WG_POST_WAVE=32 uses post.co
    post_wave = choose(nullptr, "WG_POST_WAVE", c.post_wave == 32 ? "32" : "64") == "32" ? 32 : 64;
    return true;
  }

  // Entry `name` of <build_dir>/variants.json (clean/runner/build.py --variants): code object file and geometry.
  bool read_variant_geo(const std::string& name, std::string* file, SwinGeo* g) {
    const std::string path = build_dir + "/variants.json";
    std::string text, jerr;
    wgc::Json j;
    if (!read_text(path, &text)) return fail("Swin variant " + name + ": cannot read " + path);
    if (!wgc::parse_json(text, j, jerr)) return fail(path + ": " + jerr);
    const wgc::Json* e = j.find(name);
    const wgc::Json* f = e ? e->find("file") : nullptr;
    const wgc::Json* geo = e ? e->find("geometry") : nullptr;
    if (!f || !f->is_string() || !geo) return fail("Swin variant " + name + " was not built (see " + path + ")");
    if (json_int(geo->find("profile"), 0) != 0)
      return fail("Swin variant " + name + " is an instrumented profiling build (Runner.profile only)");
    *file = f->text;
    static const char* const kernels[5] = {"swin_block_c64", "swin_block_c64_add", "swin_block_c96",
                                           "swin_block_c96_add", "swin_block_c128"};
    const wgc::Json* th = geo->find("threads");
    const wgc::Json* wp = geo->find("wpw");
    const wgc::Json* wg = geo->find("wpg");
    for (int k = 0; k < 5; ++k) {
      g->threads[k] = json_int(th ? th->find(kernels[k]) : nullptr, 256);
      g->wpg[k] = json_int(wg ? wg->find(kernels[k]) : nullptr, 0);
    }
    g->wpw[0] = json_int(wp ? wp->find("swin_block_c64") : nullptr, 1);
    g->wpw[1] = json_int(wp ? wp->find("swin_block_c96") : nullptr, 1);
    g->wpw[2] = json_int(wp ? wp->find("swin_block_c128") : nullptr, 1);
    g->gemm_threads = json_int(geo->find("gemm_threads"), 256);
    g->nt = json_int(geo->find("nt"), 2);
    const wgc::Json* fc = geo->find("force_clamp");
    g->force_clamp = fc && fc->is_bool() && fc->boolean;
    const wgc::Json* fu = geo->find("fuse");
    g->fuse = fu && fu->is_bool() && fu->boolean;
    const wgc::Json* mg = geo->find("mega");
    g->mega_threads = mg ? json_int(mg->find("threads"), 0) : 0;
    g->mega_groups = mg ? json_int(mg->find("groups"), 0) : 0;
    const wgc::Json* md = geo->find("mid");
    g->mid_threads = md ? json_int(md->find("threads"), 0) : 0;
    g->mid_groups = md ? json_int(md->find("groups"), 0) : 0;
    return true;
  }

  // Global Swin build variant `name` (WG_SWIN_VARIANT). Geometry set in the configuration takes precedence.
  bool swin_variant(const std::string& name) {
    SwinGeo g;
    if (!read_variant_geo(name, &swin_co, &g)) return false;
    for (int k = 0; k < 5; ++k) {
      if (!cfg.block_threads[k] && !cfg.swin_threads) block_threads[k] = g.threads[k];
      if (!cfg.block_wpg[k]) block_wpg[k] = g.wpg[k];
    }
    if (!cfg.wpw64) wpw[0] = g.wpw[0];
    if (!cfg.wpw96) wpw[1] = g.wpw[1];
    if (!cfg.wpw128) wpw[2] = g.wpw[2];
    if (!cfg.gemm_threads) gemm_threads = g.gemm_threads;
    if (!cfg.nt) nt = g.nt;
    if (g.force_clamp) force_clamp = true;
    fuse = g.fuse;
    return true;
  }

  // Runtime options: graph=1 (HIP graph per frame), stream_priority=high (highest stream priority). Keys of the
  // configuration string first, then WG_RUNTIME_OPTIONS (its keys win, as in the interop). Unknown keys are ignored.
  bool runtime_options(const WgRtConfig& c) {
    const char* env = std::getenv("WG_RUNTIME_OPTIONS");
    options_text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
    for (size_t a = 0; a <= options_text.size();) {
      size_t b = options_text.find_first_of(";,", a);
      if (b == std::string::npos) b = options_text.size();
      const std::string kv = options_text.substr(a, b - a);
      a = b + 1;
      const size_t eq = kv.find('=');
      if (eq == std::string::npos) continue;
      const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
      if (k == "graph") use_graph = v == "1" || v == "on" || v == "true";
      if (k == "stream_priority") high_priority = v == "high";
      if (k == "post_bands") post_bands = std::max(1, std::atoi(v.c_str()));
      if (k == "post_tf") post_tf = !(v == "0" || v == "off" || v == "false");
      if (k == "output_filter") {
        if (v != "default" && v != "original" && v != "analytic" && v != "quality5" && v != "quality9")
          return fail("output_filter must be original, analytic, quality5 or quality9 (or default; Model M)");
        output_filter_opt = v;
      }
      if (k == "pre_exposure") {
        if (v != "original" && v != "scale") return fail("pre_exposure must be scale or original");
        pi_original = v == "original";
      }
      if (k == "d1d0_p12") d1d0_p12_opt = !(v == "0" || v == "off" || v == "false");
      if (k == "dec_rdiv") dec_rdiv = !(v == "0" || v == "off" || v == "false");
      if (k == "dec_bdiv") dec_bdiv = !(v == "0" || v == "off" || v == "false");
      if (k == "swin_resident") swin_resident = std::max(0, std::atoi(v.c_str()));
      if (k == "post_tf_cpairs") post_tf_cpairs = std::max(2, std::atoi(v.c_str()) & ~1);  // even
      if (k == "post_tf4") {
        if (v != "auto" && v != "force") return fail("post_tf4 must be auto or force");
        post_tf4_force = v == "force";
      }
      if (k == "post_diet") post_diet = !(v == "0" || v == "off" || v == "false");
      if (k == "tf_vdiet") tf_vdiet = !(v == "0" || v == "off" || v == "false");
      if (k == "post_idx") post_idx = !(v == "0" || v == "off" || v == "false");
      if (k == "post_trim") post_trim = !(v == "0" || v == "off" || v == "false");
      if (k == "post_sel") post_sel = !(v == "0" || v == "off" || v == "false");
      if (k == "motion_outres") {
        if (v != "original" && v != "point") return fail("motion_outres must be original or point");
        motion_outres = v == "original" ? 1 : 2;
      }
    }
    if (const char* nb = std::getenv("WG_POST_BANDS")) post_bands = std::max(1, std::atoi(nb));
    if (const char* sr = std::getenv("WG_SWIN_RESIDENT")) swin_resident = std::max(0, std::atoi(sr));
    if (const char* t = std::getenv("WG_POST_TF")) post_tf = !(std::string(t) == "0" || std::string(t) == "off");
    // The output filter (spec 06 draft 4, spec 10 section 5 step 6): the original's weights from the model file by
    // default, the analytic set when the model file has none (noted for the log) or when it is selected.
    output_quality9 = output_filter_opt == "quality9";
    output_quality5 = output_filter_opt == "quality5";
    const bool filter_default = output_filter_opt.empty() || output_filter_opt == "default";
    if (output_filter_opt == "original" && !has_taps)
      return fail("output_filter=original: the model file has no output filter weights (group filter / output_taps; "
                  "written by the converter of spec 08 draft 2 when its source has them). Convert the model again, "
                  "or use output_filter=analytic");
    output_original = output_filter_opt == "original" || (filter_default && has_taps);
    if (filter_default && !has_taps)
      notes_text = std::string("output filter: analytic (the model file has no output filter weights, group filter / "
                               "output_taps") +
                   (model_other_map ? "; it was converted with an earlier converter map, convert again to get them)"
                                    : ")");
    if (output_original) {
      for (int i = 0; i < 5; ++i) {
        lz_bits[i] = f32_to_f16(output_taps[i]);  // h(w_k): rounded to FP16 once
        if ((lz_bits[i] & 0x7c00u) == 0x7c00u)
          return fail("model data damaged: output filter weight " + std::to_string(i) + " is outside the FP16 range");
      }
    }
    if (output_quality9) {
      if (recon_output || post_fused || post_kernel.compare(0, 13, "post_recon2_r") != 0)
        return fail("output_filter=quality9 needs the exact post_recon2_r path and D=1.5 O");
      const size_t band = post_kernel.find("_band");
      if (band != std::string::npos) post_kernel.erase(band, 5);
      post_band = post_tf = false;
      filter_kernel = "post_filter_quality9";
      filter_tile_x = 40;
      filter_s = 4;
      out_f16 = false;
    }
    if (native_direct_io && (use_graph || high_priority))
      return fail("native direct resources do not support graph capture or HIP stream priority");
    if (use_graph && !dev->supports_graphs()) use_graph = false;  // emulator, or a HIP runtime without graphs
    // not supported (emulator, older HIP runtime): reported as stream_priority=default by kernels()
    if (high_priority && !dev->set_stream_priority(true)) high_priority = false;
    return true;
  }

  // The typed-resource shaders reproduce only the selected original-filter route. Refuse a different arithmetic
  // variant at create, before the first frame can modify recurrent history; the staged buffer route is unchanged.
  bool validate_native_direct_config() {
    // Motion at output resolution: motion_outres=original runs the direct forms of the emo / pmo kernels
    // (emo/input_encoder_mo_direct, pmo/post_tf_f32_p12_mo_direct: they read the game's motion texture; every frame must
    // say so, FrameIo.motion_hi_direct); motion_outres=point runs the render-size route on the motion texture the caller
    // binds (the D3D12 executor's point resampling of the game's motion, interop/convert.cpp motionpt).
    if (enc0_co != "encoder.co" || exposure_co != "exposure.co" || dec0_co != "decoder.co" ||
        swin_co != "swin.co" || !stage_geo.empty())
      return fail("native direct resources require the original encoder, exposure, decoder and Swin code-object variants");
    // the wave64 fused post: the "original" filter build with the model file's weights (post_w64__orig.co, 42 DWORDs),
    // or the analytic filter build (post_w64.co, 22 DWORDs: no weights; a model file without them, as Model L's)
    if (output_quality5 || output_quality9 || post_pre != "w64/" || (output_original && !has_taps))
      return fail("native direct resources require the wave64 fused post: post_w64__orig.co with the model's "
                  "output-filter weights, or the analytic post_w64.co");
    if (recon_output || !p12 || !post_band || post_fused || out_f16 || post_wave != 64 ||
        post_kernel != "post_recon2_r_band_p12" || filter_kernel != "post_filter_c1_s4_band" ||
        !post_tf || !post_diet || !tf_vdiet || !post_sel || !post_idx || !post_trim)
      return fail("native direct resources support only post_tf_f32_p12_diet_vsit; alternate post/TF routes are unsupported");
    if ((!d1d0_p12 && !model_l) || !dec_div.empty() || dec1_dec0 || bneck_variant != "bmegahg" ||
        bneck_threads != 256 || bneck_groups != 128 || !mid_variant.empty() || swin_resident || force_clamp || fuse)
      return fail("native direct resources require default pack12 decoder, bmegahg bottleneck and nonresident unclamped Swin routes");
    if (gemm_threads != 256 || nt != 2 || wpw[0] != 1 || wpw[1] != 1 || wpw[2] != 1)
      return fail("native direct resources require the ported Swin/GEMM geometry");
    for (int i = 0; i < 5; ++i)
      if (block_threads[i] != 256 || (block_wpg[i] != 0 && block_wpg[i] != 8))
        return fail("native direct resources require 256-lane, eight-window Swin blocks");
    for (const char* stage : kSwinOrder)
      if (!swin_nc[stage]) return fail("native direct resources do not support clamped Swin stage " + std::string(stage));
    if (std::find(fuse_stages.begin(), fuse_stages.end(), "dec2") == fuse_stages.end())
      return fail("native direct resources require the fused dec2 expand/block port");
    for (const std::string& stage : fuse_stages)
      if (stage != "dec1" && stage != "dec2")
        return fail("native direct resources do not support fused Swin stage " + stage);
    return true;
  }

  float effective_pre_exposure(float value) const {
    return !pi_original && std::isfinite(value) && value > 0.0f ? value : 1.0f;
  }

  bool validate_native_direct_frame(const WgRtFrame& fr, const FrameIo* io) {
    if (!io || !io->native_direct) return fail("native direct runtime requires FrameIo.native_direct");
    if (io->raw_inputs || io->out_half_pitch || io->out_rgba16f || io->motion_hi || io->out_signal || hi_w)
      return fail("native direct frame does not support raw buffer conversion, pitched/packed output or a motion_hi buffer");
    if (io->motion_hi_direct != (motion_outres == 1))
      return fail(motion_outres == 1 ? "native direct frame: motion_outres=original needs FrameIo.motion_hi_direct"
                                     : "native direct frame: motion_hi_direct needs the runtime option motion_outres=original");
    if (io->motion_hi_direct && (io->motion_hi_w <= 0 || io->motion_hi_h <= 0 || io->motion_hi_wide ||
                                 io->motion_hi_pitch != io->motion_hi_w * 4))
      return fail("native direct frame: invalid output-resolution motion size");
    if (io->exposure_mode < -1 || io->exposure_mode > 2)
      return fail("native direct frame has an unsupported exposure mode");
    if (fr.reset < 0 || fr.reset > 2) return fail("native direct frame reset must be 0, 1 or 2");
    if (fr.rw <= 0 || fr.rh <= 0 || fr.rw > max_rw || fr.rh > max_rh)
      return fail("native direct render size is outside the configured maximum");
    if (ceil32(fr.rw) > 2 * fr.rw - 1 || ceil32(fr.rh) > 2 * fr.rh - 1)
      return fail("native direct render size is too small for its padding (spec 03)");
    if (!tf_supported(fr.rw, fr.rh, dw, dh, kTfCap) && !tf4_supported(fr.rw, fr.rh, dw, dh))
      return fail("native direct frame geometry is unsupported by post_tf_f32_p12_diet_vsit and post_tf4_dl_f32_p12_diet_vsit; "
                  "no fallback");
    return true;
  }

  bool direct_preflight(const WgRtFrame& fr, const FrameIo* io) {
    if ((io && io->native_direct) != native_direct_io)
      return fail("FrameIo.native_direct does not match the runtime's I/O mode");
    return !native_direct_io || validate_native_direct_frame(fr, io);
  }

  // Persistent kernel swin_mid (after the device and the per-stage choice exist): code object tagged "mid", its
  // geometry, the second F/8 scratch map and the zeroed sync words (the kernel leaves them zero after every launch).
  bool mid_setup(const WgRtConfig& c) {
    std::string v = choose(c.swin_mid, "WG_SWIN_MID", "0");
    if (v == "0" || v == "none" || v == "off") return true;
    if (v == "1" || v == "on") v = "bmegahg";
    for (int i = 1; i < 9; ++i)
      if (stage_geo.count(kSwinOrder[i]))
        return fail("swin mid together with a per-stage variant of " + std::string(kSwinOrder[i]));
    SwinGeo g;
    std::string file;
    if (!read_variant_geo(v, &file, &g)) return false;
    if (!g.mid_threads || !g.mid_groups) return fail("Swin variant " + v + " has no persistent kernel swin_mid");
    const std::string lib_dir = c.emu_library ? std::string(c.emu_library).substr(
                                                    0, std::string(c.emu_library).find_last_of("/\\"))
                                              : build_dir;
    const std::string path = c.backend == 1 ? lib_dir + "/wg_emu__" + v + ".so" : build_dir + "/" + file;
    if (!dev->load_module_tagged(path, "mid")) return dfail();
    mid_variant = v;
    mid_threads = g.mid_threads;
    mid_groups = g.mid_groups;
    const size_t FH = ceil32(max_rh), FW = ceil32(max_rw), H4 = FH / 4, H8 = FH / 8, W8 = FW / 8;
    const size_t nsync = 16 + 8 * ((H4 + 6) / 4);  // kMegaSyncHead + 8 phases x window rows (swin_v2.hip)
    return alloc(&bneck_tmp, H8 * W8 * 128) && alloc(&mid_sync, 4 * nsync) &&
           (dev->memset(mid_sync, 0, 4 * nsync) || dfail());
  }

  // Bottleneck megakernel (after the device and the per-stage choice exist): code object tagged "bneck", its geometry,
  // the scratch map and the zeroed sync words (the kernel leaves them zero after every launch).
  bool bneck_setup(const WgRtConfig& c) {
    bool b_override = false;
    for (int i = 0; i < 6; ++i) b_override |= stage_geo.count("b" + std::to_string(i)) != 0;
    const bool by_default = !(c.swin_bneck && *c.swin_bneck) && !std::getenv("WG_SWIN_BNECK");
    if (!mid_variant.empty()) {  // swin_mid runs b0 ... b5 itself
      if (!by_default && choose(c.swin_bneck, "WG_SWIN_BNECK", "none") != "none")
        return fail("WG_SWIN_BNECK together with WG_SWIN_MID");
      return true;
    }
    const std::string v =
        choose(c.swin_bneck, "WG_SWIN_BNECK", swin_co == "swin.co" && !b_override ? "bmegahg" : "none");
    if (v == "none") return true;
    if (b_override) return fail("swin bneck " + v + " together with per-stage variants of b0 ... b5");
    SwinGeo g;
    std::string file;
    const std::string lib_dir = c.emu_library ? std::string(c.emu_library).substr(
                                                    0, std::string(c.emu_library).find_last_of("/\\"))
                                              : build_dir;
    const std::string path_emu = lib_dir + "/wg_emu__" + v + ".so";
    if (by_default) {  // a build without the megakernel (or the emulator without its library): six launches
      std::string text, jerr;
      wgc::Json j;
      if (!read_text(build_dir + "/variants.json", &text) || !wgc::parse_json(text, j, jerr) || !j.find(v)) return true;
      std::FILE* fp = c.backend == 1 ? std::fopen(path_emu.c_str(), "rb") : nullptr;
      if (c.backend == 1 && !fp) return true;
      if (fp) std::fclose(fp);
    }
    if (!read_variant_geo(v, &file, &g)) return false;
    if (!g.mega_threads || !g.mega_groups) return fail("Swin variant " + v + " has no bottleneck megakernel");
    if (!dev->load_module_tagged(c.backend == 1 ? path_emu : build_dir + "/" + file, "bneck")) return dfail();
    bneck_variant = v;
    bneck_threads = g.mega_threads;
    bneck_groups = g.mega_groups;
    const size_t FH = ceil32(max_rh), FW = ceil32(max_rw), H8 = FH / 8, W8 = FW / 8;
    // kMegaSyncHead + 6 blocks x window rows (swin_v2.hip); one more row set for the fused merge
    // (bneck/swin_bneck6_mb_nc, WG_D3D12_FUSION mb)
    const size_t nsync = 16 + 7 * ((H8 + 6) / 4);
    return (bneck_tmp || alloc(&bneck_tmp, H8 * W8 * 128)) && alloc(&bneck_sync, 4 * nsync) &&
           (dev->memset(bneck_sync, 0, 4 * nsync) || dfail());
  }

  // Per-stage Swin variants: "stage=variant[,stage=variant...]" (after the device exists).
  bool stage_variants(const WgRtConfig& c) {
    stage_variants_text = choose(c.swin_stage_variants, "WG_SWIN_STAGE_VARIANTS", "");
    std::map<std::string, SwinGeo> loaded;
    size_t pos = 0;
    const std::string& t = stage_variants_text;
    while (pos < t.size()) {
      size_t comma = t.find(',', pos);
      if (comma == std::string::npos) comma = t.size();
      const std::string kv = t.substr(pos, comma - pos);
      pos = comma + 1;
      if (kv.empty()) continue;
      const size_t eq = kv.find('=');
      const std::string stages = kv.substr(0, eq), v = eq == std::string::npos ? "" : kv.substr(eq + 1);
      if (v.empty()) return fail("swin stage variants: bad entry \"" + kv + "\"");
      std::vector<std::string> list;  // "b0/b2/b4=gelu16": several stages, one variant
      for (size_t a = 0; a <= stages.size();) {
        size_t b = stages.find('/', a);
        if (b == std::string::npos) b = stages.size();
        list.push_back(stages.substr(a, b - a));
        a = b + 1;
      }
      for (const std::string& st : list)
        if (!swin_stage_info(st))
          return fail("swin stage variants: unknown stage \"" + st + "\" in \"" + kv + "\"");
      if (v == "base") {
        for (const std::string& st : list) stage_geo.erase(st);
        continue;
      }
      if (!loaded.count(v)) {
        SwinGeo g;
        std::string file;
        if (!read_variant_geo(v, &file, &g)) return false;
        g.tag = v;
        const std::string lib_dir = c.emu_library ? std::string(c.emu_library).substr(
                                                        0, std::string(c.emu_library).find_last_of("/\\"))
                                                  : build_dir;
        const std::string path = c.backend == 1 ? lib_dir + "/wg_emu__" + v + ".so" : build_dir + "/" + file;
        if (!dev->load_module_tagged(path, v)) return dfail();
        loaded[v] = g;
      }
      for (const std::string& st : list) stage_geo[st] = loaded[v];
    }
    return true;
  }

  const char* kernels() {
#ifndef WG_NO_MODEL_K
    if (k) return k->kernels();
#endif
    kernels_text = "swin=" + swin_co + " enc0=" + enc0_co + " dec0=" + dec0_name + " exposure=" + exposure_variant +
                   " post=" + post_kernel + " filter=" + (post_fused ? std::string("fused") : filter_kernel) +
                   " post_last=" + post_last +
                   " filter_last=" + (filter_last.empty() ? std::string("none") : filter_last) +
                   " history_bytes=" + std::to_string(hist_bytes) +
                   " history_format=" + (hist_bytes == 4 ? "h4" : hist_bytes == 8 ? "h8" : hist_planar ? "hp" : "h6") +
                   " post_bands=" + (post_band ? std::to_string(post_bands) : std::string("0")) +
                   " post_tf=" + (post_tf ? std::to_string(post_tf_cpairs) : std::string("0")) +
                   " out=" + ((frames ? frame_out_f16 : out_f16) ? "rgba16f" : "f32") +
                   " post_wave=" + std::to_string(post_wave) +
                   " motion_outres=" + (motion_outres == 1 ? "original" : motion_outres == 2 ? "point" : "off") +
                   " motion_last=" + (mo_last ? "output" : "render") +
                   " recon=" + (recon_output ? "output" : "spec") +
                   " graph=" + (use_graph ? "1" : "0") + " stream_priority=" + (high_priority ? "high" : "default") +
                   " swin_nc=";
    std::string nc;
    for (const char* st : kSwinOrder) {
      const auto sg = stage_geo.find(st);
      if (swin_nc[st] && !(sg == stage_geo.end() ? force_clamp : sg->second.force_clamp))
        nc += (nc.empty() ? "" : ",") + std::string(st);
    }
    kernels_text += nc.empty() ? "none" : nc;
    std::string sv;
    for (const char* st : kSwinOrder) {
      const auto sg = stage_geo.find(st);
      if (sg != stage_geo.end()) sv += (sv.empty() ? "" : ",") + std::string(st) + ":" + sg->second.tag;
    }
    kernels_text += " swin_stages=" + (sv.empty() ? std::string("none") : sv);
    std::string fz;
    for (const char* st : kSwinOrder) {
      const auto sg = stage_geo.find(st);
      const SwinStageInfo* si = swin_stage_info(st);
      const bool on = sg != stage_geo.end()
                          ? sg->second.fuse
                          : (fuse || std::find(fuse_stages.begin(), fuse_stages.end(), st) != fuse_stages.end());
      if (on && si->kind != 'b') fz += (fz.empty() ? "" : ",") + std::string(st);
    }
    kernels_text += " swin_fused=" + (fz.empty() ? std::string("none") : fz);
    kernels_text += " swin_bneck=" + (bneck_variant.empty() ? std::string("none") : bneck_variant);
    kernels_text += " swin_resident=" + std::to_string(swin_resident);
    kernels_text += std::string(" d1d0_p12=") + (d1d0_p12 ? "1" : "0");
    kernels_text += std::string(" dec_rdiv=") + (dec_div.find('r') == std::string::npos ? "1" : "0");
    kernels_text += std::string(" dec_bdiv=") + (dec_div.find('b') == std::string::npos ? "1" : "0");
    kernels_text += std::string(" dec1_dec0=") + (dec1_dec0 ? "fused" : "separate");
    if (fz_pm || fz_xe || fz_mb || fz_rv || fz_xo || fz_xd) kernels_text += " d3d12_fusion=" + fz_text;
    kernels_text += " swin_mid=" + (mid_variant.empty() ? std::string("none") : mid_variant);
    kernels_text += std::string(" post_pack=") + (p12 ? "p12" : "none");
    kernels_text += std::string(" pre_exposure=") + (pi_original ? "original" : "scale");
    // the output filter in use (not the option: "default" resolves to original or analytic) and whether the model
    // file carries the original's weights
    kernels_text += std::string(" output_filter=") +
                    (output_quality9 ? "quality9" : output_quality5 ? "quality5" : output_original ? "original" : "analytic");
    kernels_text += std::string(" output_taps=") + (has_taps ? "model" : "none");
    kernels_text += std::string(" model_map=") + (model_source == "raw" ? "none" : model_other_map ? "other" : "same");
    if (native_direct_io) kernels_text += " native_io=direct_textures";
    // Model L: its encoder and decoder (enc0= / dec0= above name Model M's, not loaded for Model L)
    if (model_l) kernels_text += " model=L l_kernels=l_enc_values.co,l_ops.co,l_swin.co,l_post.co l_post=l_post_recon";
    return kernels_text.c_str();
  }

  bool alloc_buffers() {
    const size_t R = (size_t)max_rw * max_rh, FW = ceil32(max_rw), FH = ceil32(max_rh);
    const size_t t2 = (FW / 2) * (FH / 2), t4 = (FW / 4) * (FH / 4), t8 = (FW / 8) * (FH / 8);
    const size_t Dn = (size_t)dw * dh, On = (size_t)ow * oh;
    const int ny = std::max(256, max_rh / 2);
    // External buffer I/O keeps depth/motion/color_net for raw-input conversion. Direct textures need none of the
    // five dense I/O buffers; their shader loads/stores use the caller's typed descriptors instead.
    bool ok = (external_io || (alloc(&color, R * 16) && alloc(&out, On * 12))) &&
              (native_direct_io || (alloc(&depth, R * 4) && alloc(&motion, R * 8) && alloc(&color_net, R * 16))) &&
              alloc(&expo, 32) &&
              (!native_direct_io || (alloc(&f16lut, kF16LutBytes) && (dev->memset(f16lut, 0, 256) || dfail()))) &&
              alloc(&expo_partials, 4 * (size_t)cdiv(ny, 4)) && alloc(&expo_computed, 4) && alloc(&expo_sync, 4) &&
              (dev->memset(expo_sync, 0, 4) || dfail()) && alloc(&feat0, t2 * 64) &&
              alloc(&mnear[0], R * 4) && alloc(&mnear[1], R * 4) && alloc(&mfar, R * 4) && alloc(&mcenter, R * 4) &&
              alloc(&latent[0], R * 4) && alloc(&latent[1], R * 4) && alloc(&reduced, (size_t)hw * hh * 4) &&
              alloc(&skip1, t2 * 64) && alloc(&merged1, t4 * 96) && alloc(&skip2, t4 * 96) &&
              alloc(&merged2, t8 * 128) && alloc(&e2, t4 * 96) && alloc(&out2, t4 * 96) && alloc(&e1, t2 * 64) &&
              alloc(&out1, t2 * 64) && alloc(&A, R * (p12 ? 8 : 4)) && alloc(&B, R * 4) && alloc(&C, R * 8) &&
              alloc(&hist[0], Dn * 8) && alloc(&hist[1], Dn * 8) && alloc(&display, display_bytes(dw, dh));
    for (int i = 0; ok && i < 6; ++i) ok = alloc(&bout[i], t8 * 128);
    if (ok && (fz_xd || fz_any_xd)) {
      const size_t nxd = 16 + (FH / 4 + 7) / 4 + 1;
      ok = alloc(&xd_sync, 4 * nxd) && (dev->memset(xd_sync, 0, 4 * nxd) || dfail());
    }
    if (ok && uses_prep(post_kernel)) ok = alloc(&prep, R * 32);
#ifdef WG_MODEL_L
    if (ok && model_l) {
      const size_t halo = (FH + 4) * (FW + 4);
      ok = alloc(&l_wide, halo * 64) && alloc(&l_narrow, halo * 32) && alloc(&l_x3, FH * FW * 32);
    }
#endif
    return ok;
  }

  bool upload_inputs(int rw, int rh, const float* rgb, const float* dep, const float* mot) {
#ifndef WG_NO_MODEL_K
    if (k) return k->upload_inputs(rw, rh, rgb, dep, mot) || fail(k->err());
#endif
    if (rw <= 0 || rh <= 0 || rw > max_rw || rh > max_rh) return fail("render size larger than the maximum");
    if (external_io) return fail("inputs come from the interop (external_io)");
    if (!dev->sync()) return dfail();  // the staging copy of the previous upload may still be in use
    const size_t npix = (size_t)rw * rh;
    staging.assign(npix * 4, 0.0f);
    for (size_t i = 0; i < npix; ++i)
      for (int c = 0; c < 3; ++c) staging[4 * i + c] = rgb[3 * i + c];
    if (!dev->upload(color, staging.data(), npix * 16) || !dev->upload(depth, dep, npix * 4) ||
        !dev->upload(motion, mot, npix * 8))
      return dfail();
    return true;
  }

  // Test path of motion vectors at output resolution (wg_rt_upload_motion_hi): the motion of M = mw x mh as dense
  // rows of two FP16 (wide = 0) or two f32 per texel, in an own buffer; mw = 0: back to the render-size motion.
  bool upload_motion_hi(int mw, int mh, int wide, const void* texels, const float* scale, const float* b) {
#ifndef WG_NO_MODEL_K
    if (k) return fail("motion at output resolution: not for model K here");
#endif
    if (mw <= 0 || mh <= 0 || !texels) {
      hi_w = hi_h = 0;
      return true;
    }
    if (motion_outres != 1) return fail("wg_rt_upload_motion_hi needs the runtime option motion_outres=original");
    if (!dev->sync()) return dfail();
    const size_t bytes = (size_t)mw * mh * (wide ? 8 : 4);
    if (bytes + 8 > hi_cap) {  // (an outgrown buffer stays allocated until destroy: tests only)
      if (!alloc(&hi_buf, bytes + 8)) return false;
      hi_cap = bytes + 8;
    }
    if (!dev->upload(hi_buf, texels, bytes) || !dev->sync()) return dfail();
    hi_w = mw;
    hi_h = mh;
    hi_wide = wide ? 1 : 0;
    for (int i = 0; i < 2; ++i) hi_scale[i] = scale ? scale[i] : 1.0f, hi_b[i] = b ? b[i] : 0.0f;
    return true;
  }

  // Timing log: event index to bind as the start event of the next kernel launched (-1 = none).
  int pending_mark = -1;
  long long frames_ab = 0;

  // Diagnostics (WG_D3D12_NOBAR, a comma list of kernel names): launch these without the barrier after the previous
  // launch. Results are racy (wrong); only for measuring the cost of a kernel boundary.
  std::vector<std::string> nobar_list = [] {
    const char* e = std::getenv("WG_D3D12_NOBAR");
    return split_list(e ? e : "");
  }();
  // Diagnostics (WG_D3D12_KSUB / WG_AB_KSUB for the A/B blocks, comma list of from=to): launch kernel "to" (same
  // arguments) where the runtime launches "from".
  static std::vector<std::pair<std::string, std::string>> ksub_parse(const char* e) {
    std::vector<std::pair<std::string, std::string>> m;
    for (const std::string& t : split_list(e ? e : "")) {
      const size_t q = t.find('=');
      if (q != std::string::npos) m.emplace_back(t.substr(0, q), t.substr(q + 1));
    }
    return m;
  }
  std::vector<std::pair<std::string, std::string>> ksub_list = ksub_parse(std::getenv("WG_D3D12_KSUB")),
                                                   ksub_ab = ksub_parse(std::getenv("WG_AB_KSUB"));
  bool ksub_b = false;
  bool launch(const char* k, int grid, int block, std::vector<void*> args) {
    if (grid <= 0) return true;
    for (const std::string& n : (ab_block ? nobar_now : nobar_list))
      if (n == k) dev->unordered_next();
    for (const auto& m : ((ab_block && ksub_b) ? ksub_ab : ksub_list))
      if (m.first == k) k = m.second.c_str();
    if (pending_mark >= 0) {
      const int e = pending_mark;
      pending_mark = -1;
      if (!dev->launch_timed(k, (unsigned)grid, (unsigned)block, args.data(), e, -1)) return dfail();
      return true;
    }
    if (!dev->launch(k, (unsigned)grid, (unsigned)block, args.data())) return dfail();
    return true;
  }

#ifdef WG_MODEL_L
  // ---- Model L (clean/l/FORMATS.md: maps, blobs, kernels, launch order) ----
  // Window offsets (ox, oy) of Model L's two width-32 blocks for frame parity f: the default table's window_offsets
  // (clean/l/reference/table.py), "first" for the encoder's block, "second" for the decoder's.
  static void l_offsets(bool decoder, int f, int* ox, int* oy) {
    static const int kOff[2][2][2] = {{{2, 2}, {0, 2}}, {{0, 0}, {2, 0}}};
    *ox = kOff[decoder][f & 1][0];
    *oy = kOff[decoder][f & 1][1];
  }

  // One width-32 Swin block on the fh x fw grid (l_swin.co): "_enc" reads the FP16 halo map x2 and writes FP8, "_dec"
  // reads FP8 and writes FP16.
  bool l_block(const char* kind, DevPtr x, DevPtr out, DevPtr w, bool nc, int H, int W, bool decoder, int f) {
    DevPtr none = 0;
    int ox = 0, oy = 0;
    l_offsets(decoder, f, &ox, &oy);
    const int nwin = ((W + ox + 3) / 4) * ((H + oy + 3) / 4);
    const std::string k = std::string("l_swin_block_c32_") + kind + (nc ? "_nc" : "");
    return launch(k.c_str(), cdiv(nwin, 8), 256, {&x, &none, &out, &w, &H, &W, &ox, &oy});
  }

  // Step 2 for Model L: the 20 values and their dithered FP8 for every pixel of the halo map (Model M's input encoder
  // arguments, q in place of the features; the motion surfaces as Model M), the two per-pixel linears, the encoder
  // block, and the gather of each token's 2 x 2 pixels into the 64 features enc1 reads (feat0).
  bool l_encoder(EncParams& p, DevPtr cnet, DevPtr din, DevPtr min_, int cur, int nxt, int fh, int fw, int f) {
    DevPtr wv = wdev["enc0.values"], w1 = wdev["enc0.Win1"], w2 = wdev["enc0.Win2"], wp = wdev["enc0.project"];
    int npix = (fh + 4) * (fw + 4), tw = fw / 2, th = fh / 2;
    return launch(native_direct_io ? "lev/input_encoder_direct" : "lev/input_encoder",
                  cdiv(((fh + 4) / 2) * ((fw + 4) / 2), 128), 256,
                  {&cnet, &din, &min_, &reduced, &latent[cur], &mnear[cur], &l_wide, &mnear[nxt], &mfar, &mcenter,
                   &expo, &wv, &p}) &&
           launch("l_linear_32_32_relu", cdiv(npix, 128), 256, {&l_wide, &l_narrow, &w1, &npix}) &&
           launch("l_linear_32_32_f16", cdiv(npix, 128), 256, {&l_narrow, &l_wide, &w2, &npix}) &&
           l_block("enc", l_wide, l_x3, wdev["enc0_block"], l_nc[0], fh, fw, false, f) &&
           launch("l_project_gather", cdiv(tw * th, 128), 256, {&l_x3, &feat0, &wp, &tw, &th});
  }

  // Step 4 for Model L: the projection of dec1's output to 32 values per pixel (y), the decoder block (y2, FP16), and
  // Model L's per-pixel functions into A, B, C and the new latent (Model M's buffers and formats; no motion, previous
  // latent, jitter or reset is read).
  bool l_decoder(DecParams& p, int nxt, int fh, int fw, int f) {
    DevPtr wd = wdev["dec0.project"], wc = wdev["dec0.coef"];
    int tw = fw / 2, th = fh / 2;
    return launch("l_dec_project", cdiv(tw * th, 128), 256, {&out1, &l_narrow, &wd, &tw, &th}) &&
           l_block("dec", l_narrow, l_wide, wdev["dec0_block"], l_nc[1], fh, fw, true, f) &&
           launch(p12_frame ? "l_coef_pack12" : "l_coef", cdiv(tw * th, 128), 256,
                  {&l_wide, &A, &B, &C, &latent[nxt], &wc, &p});
  }

  // Model L's kernels on native D3D12 that <build_dir>/dx12/manifest.json lacks, comma-separated ("" = all there):
  // its encoder (the direct form on the direct-resource route), linears, clamp-free blocks (the only D3D12 forms),
  // projections and pack12 coefficient store, and dec1's own block with its expand (Model M runs dec1 inside its fused
  // decoder there; Model L's decoder reads dec1's map). Model M's other kernels are its native frame's.
  std::string l_native_missing() const {
    std::string text, error, missing;
    wgc::Json manifest;
    std::vector<std::string> have;
    if (read_text(build_dir + "/dx12/manifest.json", &text) && wgc::parse_json(text, manifest, error)) {
      const auto* kernels = manifest.find("kernels");
      if (kernels && kernels->is_array())
        for (const auto& k : kernels->items) {
          const auto* name = k.find("name");
          if (name && name->is_string()) have.push_back(name->text);
        }
    }
    const std::vector<std::string> need = {native_direct_io ? "lev/input_encoder_direct" : "lev/input_encoder",
                                           "l_linear_32_32_relu",     "l_linear_32_32_f16",
                                           "l_swin_block_c32_enc_nc", "l_project_gather",
                                           "swin_block_c64_add_ex_nc", "l_dec_project",
                                           "l_swin_block_c32_dec_nc", "l_coef_pack12"};
    for (const std::string& k : need)
      if (std::find(have.begin(), have.end(), k) == have.end()) missing += (missing.empty() ? "" : ", ") + k;
    return missing;
  }
#endif

  // The arguments of a kernel that runs the 5-tap output filter (post_filter*, post_tf*, post_fused*): with
  // output_filter=original (post_w64__orig.co, kernels/post.hip WG_LZ_KARGS) the five weights follow as u32 FP16 bit
  // patterns; the other builds take no such arguments.
  std::vector<void*> with_taps(std::vector<void*> args) {
    if (output_original)
      for (uint32_t& b : lz_bits) args.push_back(&b);
    return args;
  }

  // Swin stage s (spec 02), launch plan as clean/runner/stage_runner.py.
  bool swin_stage(const std::string& s, DevPtr x, DevPtr skip_in, DevPtr out_main, DevPtr out_merged, DevPtr ebuf,
                  int H, int W, int ox, int oy, bool expand_only = false) {
    const SwinStageInfo* si = swin_stage_info(s);
    const int ci = si->C == 64 ? 0 : (si->C == 96 ? 1 : 2);
    // block kernel: swin_block_c<C>[_add][_nc]
    // per-stage variant (tagged code object and its geometry) or the global choice
    const auto sg = stage_geo.find(s);
    const SwinGeo* g = sg == stage_geo.end() ? nullptr : &sg->second;
    const std::string pre = g ? g->tag + "/" : "";
    const bool fc = g ? g->force_clamp : force_clamp;
    // fused merge / expand (variant fuse): encoder / decoder blocks, even window offsets (all real ones are)
    const bool listed = std::find(fuse_stages.begin(), fuse_stages.end(), s) != fuse_stages.end();
    const bool fused = (g ? g->fuse : (fuse || listed)) && si->kind != 'b' && ox % 2 == 0 && oy % 2 == 0 && !expand_only;
    // block kernel: [tag/]swin_block_c<C>[_add][_mg | _ex][_nc]
    const std::string blk = pre +
                            std::string(si->C == 64 ? "swin_block_c64"
                                                    : (si->C == 96 ? "swin_block_c96" : "swin_block_c128")) +
                            (si->kind == 'd' ? "_add" : "") +
                            (fused ? (si->kind == 'd' ? "_ex" : "_mg") : "") + (swin_nc[s] && !fc ? "_nc" : "");
    DevPtr w = wdev[s], tail = w + swin_tail_offset(s), nul = 0;
    const int nwin = ((W + ox + 3) / 4) * ((H + oy + 3) / 4);
    if (fused) {  // one launch: 256 lanes, one window per wave; the extra argument is the merged map (encoder)
      const int fgrid = cdiv(nwin, 8);
      if (si->kind == 'd' && s == "dec2" && fz_xe_now) {  // + dec1's expand as the epilogue (into dec1's map)
        DevPtr e_out = e1, e_w = wdev["dec1"] + swin_tail_offset("dec1");
        if (fz_xo_now) {  // overlapping the bottleneck: b5's grid as in its launch (odd grid, offsets off[3])
          const int h8 = H / 2, w8 = W / 2, oxo = xo_off[1][0];
          int oyo = xo_off[1][1];
          const int nr_e = (h8 + xo_off[0][1] + 3) >> 2;
          int nwy = (h8 + oyo + 3) >> 2, nwx = (w8 + oxo + 3) >> 2;
          const int NR = nr_e > nwy ? nr_e : nwy;
          DevPtr rows = bneck_sync + 4 * (16 + 5 * NR), done = bneck_sync + 4 * 14;
          dev->unordered_next();
          DevPtr xd_rows = xd_sync + 4 * 16;
          return fz_xd_now ? launch("swin_block_c96_add_xe_xd_nc", fgrid, 256,
                                    {&x, &skip_in, &out_main, &w, &H, &W, &ox, &oy, &nul, &e_out, &e_w, &rows, &done,
                                     &oyo, &nwx, &nwy, &xd_rows})
                           : launch("swin_block_c96_add_xe_xo_nc", fgrid, 256,
                                    {&x, &skip_in, &out_main, &w, &H, &W, &ox, &oy, &nul, &e_out, &e_w, &rows, &done,
                                     &oyo, &nwx, &nwy});
        }
        return launch(fz_rv ? "swin_block_c96_add_xe_rev_nc" : "swin_block_c96_add_xe_nc", fgrid, 256,
                      {&x, &skip_in, &out_main, &w, &H, &W, &ox, &oy, &nul, &e_out, &e_w});
      }
      if (si->kind == 'd')
        return launch(blk.c_str(), fgrid, 256, {&x, &skip_in, &out_main, &w, &H, &W, &ox, &oy, &nul});
      return launch(blk.c_str(), fgrid, 256, {&x, &nul, &out_main, &w, &H, &W, &ox, &oy, &out_merged});
    }
    // block kernel index of block_threads: c64, c64_add, c96, c96_add, c128
    const int bk = ci == 2 ? 4 : 2 * ci + (si->kind == 'd' ? 1 : 0);
    const int threads = g ? g->threads[bk] : block_threads[bk];
    const int wpw_c = g ? g->wpw[ci] : wpw[ci];
    // windows per workgroup (a build may split one window over several waves: geometry "wpg")
    const int wpg_k = g ? g->wpg[bk] : block_wpg[bk];
    const int wpg = wpg_k > 0 ? wpg_k : (threads / 32) * std::max(wpw_c, 1);
    const int bgrid = cdiv(nwin, wpg);
    // weight-resident persistent kernel (C = 64, global choice only, base geometry)
    int nwg_res = std::min(swin_resident, cdiv(nwin, 16));
    const bool resident = nwg_res > 0 && si->C == 64 && !g && wpg_k <= 0 && wpw_c == 1;
    const std::string rblk = std::string(si->kind == 'd' ? "swin_block_c64_add_r" : "swin_block_c64_r") +
                             (swin_nc[s] && !fc ? "_nc" : "");
    const int gthreads = g ? g->gemm_threads : gemm_threads;
    const int gpw = gthreads / 32 * 16 * (g ? g->nt : nt);
    if (si->kind == 'd') {
      int Hl = H / 2, Wl = W / 2;
      const std::string exk = pre + (si->C_low == 128 ? "expand_c128" : "expand_c96");
      return launch(exk.c_str(), cdiv((long long)Hl * Wl, gpw), gthreads, {&x, &ebuf, &tail, &Hl, &Wl}) &&
             (expand_only ||  // fused dec1 + dec0: the block runs inside swin_block_c64_add_dec0 (step 4)
              (resident ? launch(rblk.c_str(), nwg_res, 512, {&ebuf, &skip_in, &out_main, &w, &H, &W, &ox, &oy, &nwg_res})
                        : launch(blk.c_str(), bgrid, threads, {&ebuf, &skip_in, &out_main, &w, &H, &W, &ox, &oy})));
    }
    if (s == "enc2" && fz_pm_now) {  // enc1's merge as the prologue: the block reads enc1's map (skip1)
      DevPtr src = skip1, mw = wdev["enc1"] + swin_tail_offset("enc1"), mout = merged1;
      if (!launch(fz_rv ? "swin_block_c96_pm_rev_nc" : "swin_block_c96_pm_nc", bgrid, 256,
                  {&src, &nul, &out_main, &w, &H, &W, &ox, &oy, &mw, &mout}))
        return false;
    } else if (resident ? !launch(rblk.c_str(), nwg_res, 512, {&x, &nul, &out_main, &w, &H, &W, &ox, &oy, &nwg_res})
                        : !launch(blk.c_str(), bgrid, threads, {&x, &nul, &out_main, &w, &H, &W, &ox, &oy})) {
      return false;
    }
    if (si->kind == 'e' && s == "enc1" && fz_pm_now) return true;  // the merge runs in enc2's block
    if (si->kind == 'e' && s == "enc2" && fz_mb_now && !fz_mb0) return true;  // the merge runs in the bottleneck kernel
    if (si->kind == 'e') {
      const std::string mk = pre + (si->C == 64 ? "patch_merge_c64" : "patch_merge_c96");
      return launch(mk.c_str(), cdiv((long long)(H / 2) * (W / 2), gpw), gthreads,
                    {&out_main, &out_merged, &tail, &H, &W});
    }
    return true;
  }

  // Banded two-kernel path: post_recon2_r*_band over the recon tile rows of band b, then every filter tile row
  // whose D input rows are complete (FORMATS.md, runner/post.py band_plan); then the pre-exposure scale.
  // hp > 0 (interop output copy): the filter bands write RGBA16F rows of hp bytes with pi folded in
  // (post_filter_c1_s4_h_band); otherwise filter_kernel into the 3 x f32 output.
  bool post_banded(int cur, int nxt, PostParams p, DevPtr w, DevPtr cnet, DevPtr outp, const WgRtFrame& fr, int hp,
                   float pi) {
    const int tx = cdiv(dw, 32), rrows = cdiv(cdiv(dh, 2), 8), fx = cdiv(ow, 40);
    const int frows = cdiv(cdiv(oh, 2), 2 * filter_s);
    const int nb = std::max(1, std::min(post_bands, rrows));
    int a = dw, b_ = dh, c = ow, d = oh, hpitch = hp;
    float dec_scale = (float)(0.01 / (fr.exposure == 0.0f ? 1.0 : (double)fr.exposure)), p_h = pi;
    const std::string rk = post_pre + post_kernel,
                      fk = post_pre + (hp > 0 ? std::string("post_filter_c1_s4_h_band") : filter_kernel);
    if (hp > 0) filter_last = "post_filter_c1_s4_h_band";
    int fdone = 0;
    for (int b = 0; b < nb; ++b) {
      const int r0 = b * rrows / nb, r1 = (b + 1) * rrows / nb;
      if (r1 > r0) {
        int tile0 = r0 * tx;
        if (!launch(rk.c_str(), (r1 - r0) * tx, 256,
                    {&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced, &display,
                     &p, &expo, &tile0}))
          return false;
      }
      const int d_done = std::min(16 * r1, dh);  // D rows 0 ... d_done - 1 complete
      int f1 = fdone;
      while (f1 < frows) {
        const int last_pair = std::min((f1 + 1) * 2 * filter_s, cdiv(oh, 2)) - 1;
        if (std::min(3 * last_pair + 4, dh - 1) >= d_done) break;
        ++f1;
      }
      if (b == nb - 1) f1 = frows;
      if (f1 > fdone) {
        int tile0 = fdone * fx;
        if (hp > 0 ? !launch(fk.c_str(), (f1 - fdone) * fx, 64,
                             with_taps({&display, &outp, &a, &b_, &c, &d, &dec_scale, &expo, &p_h, &hpitch, &tile0}))
                   : !launch(fk.c_str(), (f1 - fdone) * fx, 64,
                             with_taps({&display, &outp, &a, &b_, &c, &d, &dec_scale, &expo, &tile0})))
          return false;
        fdone = f1;
      }
    }
    return true;
  }

  // Step 5 on the post_recon path: post_recon* (or post_prep + post_recon5* / 6*) into the display image.
  bool post_recon_stage(int rw, int rh, int cur, int nxt, PostParams p, DevPtr w, DevPtr cnet) {
    const int grid = cdiv(dw, 32) * cdiv(cdiv(dh, 2), 8);

    // post_recon2* and later need the footprint to fit their LDS staging and D >= 4 columns (recon2_supported);
    // 6* also R < D per axis (runner/post.py recon6_supported).
    // a banded kernel's frames off the banded path (footprint, output copy): its unbanded form
    std::string pk = post_kernel;
    if (post_band) pk.erase(pk.find("_band"), 5);
    const bool v6 = pk.compare(0, 11, "post_recon6") == 0;
    post_last = pk == "post_recon" ||
                        (recon2_supported(rw, rh, dw, dh) && (!v6 || (rw < dw && rh < dh)))
                    ? pk
                    : "post_recon";
    if (post_last != pk && hist_bytes != 8)
      return fail(pk + " (" + std::to_string(hist_bytes) +
                  "-byte history) cannot fall back to post_recon here");
#ifdef WG_MODEL_L
    if (model_l) {  // Model L: l_post_recon (l_post.co, untagged) with post_recon's arguments
      post_last = "l_post_recon";
      return launch("l_post_recon", grid, 256,
                    {&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced, &display,
                     &p, &expo});
    }
#endif
    if (uses_prep(post_last)) {  // pre-pass (one record per render pixel), then post_recon5* / 6*
      if (!launch((post_pre + "post_prep").c_str(), cdiv((long long)rw * rh, 256), 256,
                  {&cnet, &A, &B, &C, &prep, &p, &expo}) ||
          !launch((post_pre + post_last).c_str(), grid, 256,
                  {&prep, &cnet, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced, &display, &p,
                   &expo}))
        return false;
    } else if (p12_frame) {  // the _p12 kernels take tile0 after expo (0 when not banded)
      int tile0 = 0;
      if (!launch((post_pre + post_last).c_str(), grid, 256,
                  {&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced, &display,
                   &p, &expo, &tile0}))
        return false;
    } else if (!launch((post_pre + post_last).c_str(), grid, 256,
                       {&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced,
                        &display, &p, &expo})) {
      return false;
    }
    return true;
  }

  // One frame; with the runtime option graph=1 its launches are captured into a HIP graph and launched as one (not
  // while the timing log records events).
  bool frame(const WgRtFrame& fr, const FrameIo* io = nullptr) {
    if (ab_block) {  // A/B diagnostics: odd blocks run set B
      const bool b = ((frames_ab++ / ab_block) & 1) != 0;
      fz_set[b ? 1 : 0].apply(this);
      nobar_now = b ? nobar_ab : nobar_list;
      ksub_b = b && !ksub_ab.empty();
    }
    if (!direct_preflight(fr, io)) return false;
#ifndef WG_NO_MODEL_K
    if (k) {  // model K: its own frame (no graph capture); reset = 2 starts K over as newly created (KRuntime::frame)
      if (fr.reset >= 2) frames = 0;
      if (!k->frame(fr, io)) return fail(k->err());
      info = k->info();
      ++frames;
      return true;
    }
#endif
    const bool cap = use_graph && !(io && io->timing_base >= 0);
    if (cap && !dev->begin_capture()) return dfail();
    const bool ok = frame_body(fr, io);
    if (!cap) return ok;
    if (!ok) {
      dev->abort_capture();
      return false;
    }
    return dev->end_capture_launch() || dfail();
  }

  bool frame_body(const WgRtFrame& fr, const FrameIo* io) {
    // reset = 2 (interop: a new context of the game took this upscaler over, wronggpu_upscaler.h): this frame is the
    // first frame of a newly created runtime. Everything a frame leaves for the next one on the host starts over; the
    // device buffers keep their contents, of which a first frame reads nothing.
    if (fr.reset >= 2) {
      frames = n = f = last_rw = last_rh = 0;
      jprev[0] = jprev[1] = 0.0f;
    }
    const int rw = fr.rw, rh = fr.rh;
    if (rw <= 0 || rh <= 0 || rw > max_rw || rh > max_rh) return fail("render size larger than the maximum");
    const int fw = ceil32(rw), fh = ceil32(rh);
    if (fw > 2 * rw - 1 || fh > 2 * rh - 1) return fail("render size too small for its padding (spec 03)");
    const int reset = (fr.reset || frames == 0 || rw != last_rw || rh != last_rh) ? 1 : 0;
    // timing marks (FrameIo.timing_base >= 0): events base + 0 start of enc0 (first kernel after the input wait), 1 start
    // of the Swin stages, 2 start of dec0, 3 start of post, 4 end of the last kernel
    const int tb = io ? io->timing_base : -1;
    // Marks 0-3 are the start of the next kernel launched (the stage's first kernel), mark 4 the end of an empty
    // kernel after the last one (kernels on the stream run in order): with hipExtModuleLaunchKernel these events
    // bracket the kernels themselves, which plain events between launches do not on every HIP runtime.
    int zero = 0;
    auto mark = [&](int k) {
      if (tb < 0) return true;
      if (k < 4) return pending_mark = tb + k, true;
      pending_mark = -1;
      return dev->launch_timed("timing_barrier", 1, 32, std::vector<void*>{&zero}.data(), -1, tb + 4) || dfail();
    };
    if (!mark(0)) return false;
    const float jpx = reset ? fr.jitter[0] : jprev[0], jpy = reset ? fr.jitter[1] : jprev[1];
    const int cur = f, nxt = 1 - f;

    info = {};
    info.index = frames;
    info.n = n;
    info.parity = f;
    info.reset = reset;
    info.rw = rw;
    info.rh = rh;
    info.jitter_prev[0] = jpx;
    info.jitter_prev[1] = jpy;

    // Inputs and output: the runtime's own buffers, or external ones (interop).
    DevPtr cin = io && io->color ? io->color : color, din = io && io->depth ? io->depth : depth;
    DevPtr min_ = io && io->motion ? io->motion : motion, outp = io && io->out ? io->out : out;
    if (!native_direct_io && (!cin || !din || !min_ || !outp)) return fail("frame without input or output buffers (interop)");

    // Pre-exposure pi (spec 10 draft 2, section 5 step 1): 1 if missing, <= 0 or not finite. The stages that read
    // the color (enc0, post) get color / pi; exposure (spec 07) reads the original color with pi; the filter's
    // result is multiplied by pi below.
    // pre_exposure=original (runtime option; spec 10 draft 4): the color is not divided by pi and the result is not
    // multiplied by it; pi acts only through the exposure: inside the measurement of spec 07 (there it counts as 1
    // without the HDR flag), and a supplied exposure x becomes x / pi.
    const float pi_in = (std::isfinite(fr.pre_exposure) && fr.pre_exposure > 0.0f) ? fr.pre_exposure : 1.0f;
    const float pi = effective_pre_exposure(fr.pre_exposure);
    DevPtr cnet = cin;
    const bool raw = io && io->raw_inputs;
    // Motion vectors at output resolution (motion_outres=original): the frame brings the raw motion of M (FrameIo, or
    // the test path's own buffer) and its sizes are taken by the kernels of that path (the pack12 post path:
    // recon2_supported). The input encoder and the reconstruction then read that buffer.
    MoParams mop = {};
    EncMoParams emop = {};
    DevPtr mhi = 0;
    {
      const bool own = !(io && io->motion_hi) && hi_w > 0;
      mhi = io && io->motion_hi ? io->motion_hi : own ? hi_buf : 0;
      // native direct resources: the kernels read the game's motion texture; the pointer argument (not read) is f16lut
      if (native_direct_io && io && io->motion_hi_direct) mhi = f16lut;
      if (motion_outres != 1 || raw || !p12 || !recon2_supported(rw, rh, dw, dh)) mhi = 0;
      if (mhi) {
        mop.mw = emop.mw = own ? hi_w : io->motion_hi_w;
        mop.mh = emop.mh = own ? hi_h : io->motion_hi_h;
        mop.wide = emop.wide = own ? hi_wide : (io->motion_hi_wide ? 1 : 0);
        mop.pitch = emop.pitch = own ? hi_w * (hi_wide ? 8 : 4) : io->motion_hi_pitch;
        if (mop.mw <= 0 || mop.mh <= 0 || mop.pitch < mop.mw * (mop.wide ? 8 : 4))
          return fail("motion at output resolution: invalid size or pitch");
        const float* sc = own ? hi_scale : io->mv_scale_hi;
        const float* bb = own ? hi_b : io->b_mv;
        // s = RN32(mv_scale / RN32(M / D)), k_M = RN32(M / R) (spec 03 draft 4): FP32, one rounding per operation
        const volatile float kdx = (float)mop.mw / (float)dw, kdy = (float)mop.mh / (float)dh;
        mop.sx = sc[0] / kdx;
        mop.sy = sc[1] / kdy;
        mop.bx = bb[0];
        mop.by = bb[1];
        mop.idw = 1.0f / (float)dw;  // correctly rounded (host division); the kernels divide by D with them
        mop.idh = 1.0f / (float)dh;
        emop.kmx = (float)mop.mw / (float)rw;
        emop.kmy = (float)mop.mh / (float)rh;
      }
    }
    const bool mo = mhi != 0;
    mo_last = mo;
    if (raw) {  // interop input copy: the game's formats -> dense color / pi, depth, motion in render pixels
      if (!depth || !motion) return fail("raw inputs need the runtime's own depth / motion buffers");
      DevPtr c16 = cin, d32 = din, m16 = min_;
      int cp = io->color_pitch, dp = io->depth_pitch, mp = io->motion_pitch, mch = io->motion_channels;
      float msx = io->ms[0], msy = io->ms[1], mox = io->mo[0], moy = io->mo[1], p_ = pi;
      int a = rw, b = rh;
      if (!launch("input_convert", cdiv(rw * rh, 256), 256,
                  {&c16, &cp, &d32, &dp, &m16, &mp, &mch, &msx, &msy, &mox, &moy, &p_, &color_net, &depth, &motion,
                   &a, &b}))
        return false;
      cnet = cin = color_net;  // exposure then reads color / pi with 1 / pi = 1 (see FrameIo)
      din = depth;
      min_ = motion;
    } else if (!native_direct_io && pi != 1.0f) {
      int npix = rw * rh;
      float p_ = pi;
      if (!launch("color_divide", cdiv(npix, 256), 256, {&cin, &color_net, &p_, &npix})) return false;
      cnet = color_net;
    }
    if (native_direct_io) cnet = f16lut;  // the direct kernels' colour pointer: the output table (wg_f16lut.hlsli)

    // 1. exposure (spec 10 section 5.1, spec 07); pair state [e, e_prev, last e] on the GPU
    const int mode = io && io->exposure_mode >= 0 ? io->exposure_mode : (cfg.auto_exposure ? 2 : 0);
    int use_computed = mode != 0 ? 1 : 0, first = frames == 0 ? 1 : 0;
    float value = fr.exposure;
    DevPtr esrc = mode == 1 ? io->expo_src : expo_computed;
    if (mode == 2) {
      ExpParams p = {};
      p.rw = rw;
      p.rh = rh;
      p.nx = std::max(256, rw / 2);
      p.ny = std::max(256, rh / 2);
      p.nparts = cdiv(p.ny, 4);
      p.inv_pi = pi_original ? (cfg.hdr ? 1.0f / pi_in : 1.0f) : raw ? 1.0f : 1.0f / pi_in;
      const double a = cfg.alpha > 0 ? cfg.alpha : 0.18;
      p.K = (float)(a / (1.0 - a));
      p.inv_n = (float)(1.0 / ((double)p.nx * p.ny));
      if (post_diet) {  // one launch: partial sums, the last workgroup finishes (bit-identical to the three kernels)
        if (!launch(native_direct_io ? "exposure_auto_diet_direct" : "exposure_auto_diet", p.nparts, 256,
                    {&cin, &expo_partials, &expo_sync, &expo_computed, &expo, &p,
                                                          &first}))
          return false;
      } else if (!launch("exposure_partial", p.nparts, 256, {&cin, &expo_partials, &p}) ||
                 !launch("exposure_final", 1, 256, {&expo_partials, &expo_computed, &p})) {
        return false;
      }
    }
    float escale = 1.0f;
    if (pi_original && mode != 2) {  // a supplied exposure (or none: 1) divided by pi
      escale = 1.0f / pi_in;
      if (mode == 0 && value == 0.0f) value = 1.0f;
    }
    if (!(mode == 2 && post_diet) &&
        !launch(native_direct_io ? "exposure_pair_diet_direct" : post_diet ? "exposure_pair_diet" : "exposure_pair", 1, 32,
                {&expo, &esrc, &value, &use_computed, &first, &escale}))
      return false;

    // the direct path's 64K tables of this frame (dx12/hlsl/wg_f16lut_in.hlsli, wg_f16lut.hlsli): one lane per FP16 pattern
    if (native_direct_io && !launch("f16lut_build_direct", 256, 256, {&f16lut, &expo})) return false;

    // 2. input encoder (spec 03)
    {
      EncParams p = {};
      p.rw = rw;
      p.rh = rh;
      p.fw = fw;
      p.fh = fh;
      p.hw = hw;
      p.hh = hh;
      p.frame = n;
      p.reset = reset;
      p.depth_inverted = cfg.depth_inverted ? 1 : 0;
      p.dw = (float)dw;
      p.dh = (float)dh;
      p.kx = (float)((double)dw / rw);
      p.ky = (float)((double)dh / rh);
      p.ikx = (float)((double)rw / dw);
      p.iky = (float)((double)rh / dh);
      p.sx = (float)((double)fr.mv_scale[0] * dw / rw);
      p.sy = (float)((double)fr.mv_scale[1] * dh / rh);
      p.bx = fr.mv_bias[0];
      p.by = fr.mv_bias[1];
      p.jx = fr.jitter[0];
      p.jy = fr.jitter[1];
      p.jpx = jpx;
      p.jpy = jpy;
      p.dis_scale = (float)(540.0 / rh);
      p.idw = (float)(1.0 / dw);
      p.idh = (float)(1.0 / dh);
      p.irw = (float)(1.0 / rw);
      p.irh = (float)(1.0 / rh);
      DevPtr w = wdev["enc0"];
      const int ntok = (fh / 2) * (fw / 2);
      if (model_l) {
#ifdef WG_MODEL_L
        if (!l_encoder(p, cnet, din, min_, cur, nxt, fh, fw, f)) return false;
#endif
      } else if (mo) {  // kernels/encoder_mo.hip: one motion texel per render pixel, no depth neighbourhood
        p.sx = mop.sx;
        p.sy = mop.sy;
        p.bx = mop.bx;
        p.by = mop.by;
        if (!launch(native_direct_io ? "emo/input_encoder_mo_direct" : "emo/input_encoder_mo", cdiv(ntok, 128), 256,
                    {&cnet, &mhi, &reduced, &latent[cur], &mnear[cur], &feat0, &mnear[nxt], &mfar, &mcenter, &expo, &w,
                     &p, &emop}))
          return false;
      } else if (!launch(native_direct_io ? "input_encoder_direct" : "input_encoder",
                         cdiv(ntok, 128), 256,
                  {&cnet, &din, &min_, &reduced, &latent[cur], &mnear[cur], &feat0, &mnear[nxt], &mfar, &mcenter,
                   &expo, &w, &p}))
        return false;
    }

    // 3. Swin stages (spec 02) with the window offsets of parity f
    if (!mark(1)) return false;
    int off[10][2];
    for (int i = 0; i < 10; ++i) {
      window_offset(kSwinOrder[i], f, &off[i][0], &off[i][1]);
      info.offsets[i][0] = off[i][0];
      info.offsets[i][1] = off[i][1];
    }
    const int H2 = fh / 2, W2 = fw / 2, H4 = fh / 4, W4 = fw / 4, H8 = fh / 8, W8 = fw / 8;
    {  // native fusions this frame: base geometry, clamp-free blocks, enc1 / enc2 unfused, dec2 fused, dec1's block
       // in the fused dec1 + dec0 kernel (step 4)
      auto base_geo = [&](const char* st) { return stage_geo.find(st) == stage_geo.end(); };
      auto listed = [&](const char* st) {
        return fuse || std::find(fuse_stages.begin(), fuse_stages.end(), std::string(st)) != fuse_stages.end();
      };
      const bool d1_in_dec0 = dec1_dec0 || (d1d0_p12 && p12 && recon2_supported(rw, rh, dw, dh));
      fz_pm_now = fz_pm && mid_variant.empty() && !force_clamp && base_geo("enc1") && base_geo("enc2") &&
                  swin_nc["enc1"] && swin_nc["enc2"] && !listed("enc1") && !listed("enc2") && swin_resident <= 0 &&
                  block_threads[2] == 256 && wpw[1] == 1 && (block_wpg[2] == 0 || block_wpg[2] == 8) &&
                  H2 % 2 == 0 && W2 % 2 == 0;
      fz_xe_now = fz_xe && mid_variant.empty() && !force_clamp && base_geo("dec2") && base_geo("dec1") &&
                  swin_nc["dec2"] && listed("dec2") && off[8][0] % 2 == 0 && off[8][1] % 2 == 0 && d1_in_dec0 &&
                  swin_stage_info("dec1")->C_low == 96;
      bool bsafe = true;
      for (int i = 0; i < 6; ++i) bsafe = bsafe && swin_nc[kSwinOrder[2 + i]] && base_geo(kSwinOrder[2 + i]);
      fz_mb_now = fz_mb && mid_variant.empty() && !bneck_variant.empty() && bsafe && !force_clamp &&
                  base_geo("enc2") && swin_nc["enc2"] && !listed("enc2") && bneck_threads == 256 &&
                  swin_stage_info("enc2")->C == 96 && H4 / 2 == H8 && W4 / 2 == W8;
      fz_xo_now = fz_xo && fz_xe_now && fz_mb_now && !fz_mb0;
      fz_xd_now = fz_xd && fz_xo_now && d1d0_p12 && p12 && recon2_supported(rw, rh, dw, dh) && swin_nc["dec1"] &&
                  xd_sync != 0 && (xd_max_pixels <= 0 || (long long)rw * rh <= xd_max_pixels);
      for (int i = 0; i < 2; ++i) xo_off[0][i] = off[2][i], xo_off[1][i] = off[3][i];
    }
    if (!swin_stage("enc1", feat0, 0, skip1, merged1, 0, H2, W2, off[0][0], off[0][1])) return false;
    if (!mid_variant.empty()) {
      // enc2 + merge, b0 ... b5, expand + dec2 in one launch (swin_mid; scratch maps b5.out, which holds b5's output
      // afterwards, and bneck_tmp); enc2.out (skip2) and dec2.out kept, merged2, b0 ... b4 outputs and e2 not written;
      // dec1 runs as usual (own launch, fused expand / dec1_dec0 as chosen)
      bool safe = true;
      for (int i = 1; i < 9; ++i) safe = safe && swin_nc[kSwinOrder[i]];
      DevPtr wm[8];
      for (int i = 0; i < 8; ++i) wm[i] = wdev[kSwinOrder[1 + i]];
      int groups = mid_groups, h4 = H4, w4 = W4;
      int oxe = off[1][0], oye = off[1][1], ox0 = off[2][0], oy0 = off[2][1], ox1 = off[3][0], oy1 = off[3][1];
      int oxd = off[8][0], oyd = off[8][1];
      const std::string k = std::string("mid/swin_mid") + (safe ? "_nc" : "");
      if (!launch(k.c_str(), groups, mid_threads,
                  {&merged1, &out2, &skip2, &bout[5], &bneck_tmp, &mid_sync, &wm[0], &wm[1], &wm[2], &wm[3], &wm[4],
                   &wm[5], &wm[6], &wm[7], &h4, &w4, &oxe, &oye, &ox0, &oy0, &ox1, &oy1, &oxd, &oyd, &groups}))
        return false;
    } else if (!swin_stage("enc2", merged1, 0, skip2, merged2, 0, H4, W4, off[1][0], off[1][1])) {
      return false;
    }
    DevPtr x = merged2;
    if (!mid_variant.empty()) {
      // done above
    } else if (!bneck_variant.empty()) {  // b0 ... b5 in one launch; b0 ... b4 outputs are not kept (only b5.out)
      bool safe = true;
      for (int i = 0; i < 6; ++i) safe = safe && swin_nc[kSwinOrder[2 + i]];
      DevPtr wb[6];
      for (int i = 0; i < 6; ++i) wb[i] = wdev["b" + std::to_string(i)];
      int groups = bneck_groups, oxe = off[2][0], oye = off[2][1], oxo = off[3][0], oyo = off[3][1];
      int h8 = H8, w8 = W8;
      const std::string k = std::string("bneck/swin_bneck6") + (safe ? "_nc" : "");
      if (fz_mb_now) {  // + enc2's merge as the first phase (reads enc2's map skip2, writes merged2)
        DevPtr mx = fz_mb0 ? 0 : skip2, mw = wdev["enc2"] + swin_tail_offset("enc2");
        int w4 = W4;
        if (!launch(fz_xo_now ? "bneck/swin_bneck6_mbo_nc" : "bneck/swin_bneck6_mb_nc", groups, bneck_threads,
                    {&merged2, &bout[5], &bneck_tmp, &bneck_sync, &wb[0], &wb[1], &wb[2], &wb[3], &wb[4], &wb[5], &h8,
                     &w8, &oxe, &oye, &oxo, &oyo, &groups, &mx, &mw, &w4}))
          return false;
      } else if (!launch(k.c_str(), groups, bneck_threads,
                         {&merged2, &bout[5], &bneck_tmp, &bneck_sync, &wb[0], &wb[1], &wb[2], &wb[3], &wb[4], &wb[5],
                          &h8, &w8, &oxe, &oye, &oxo, &oyo, &groups})) {
        return false;
      }
      x = bout[5];
    }
    for (int i = 0; bneck_variant.empty() && mid_variant.empty() && i < 6; ++i) {
      const std::string s = "b" + std::to_string(i);
      if (!swin_stage(s, x, 0, bout[i], 0, 0, H8, W8, off[2 + i][0], off[2 + i][1])) return false;
      x = bout[i];
    }
    // pack12 frame with the fused dec1 + dec0: dec1's expand only here, the block runs in step 4
    const bool fused_p12 = d1d0_p12 && p12 && recon2_supported(rw, rh, dw, dh);
    if ((mid_variant.empty() && !swin_stage("dec2", bout[5], skip2, out2, 0, e2, H4, W4, off[8][0], off[8][1])) ||
        (!fz_xe_now &&  // xe: dec1's expand ran in dec2's block
         !swin_stage("dec1", out2, skip1, out1, 0, e1, H2, W2, off[9][0], off[9][1], dec1_dec0 || fused_p12)))
      return false;

    // 4. coefficient decoder (spec 05)
    if (!mark(2)) return false;
    {
      DecParams p = {};
      p.rw = rw;
      p.rh = rh;
      p.fw = fw;
      p.fh = fh;
      p.reset = reset;
      p.dw = (float)dw;
      p.dh = (float)dh;
      p.kx = (float)((double)dw / rw);
      p.ky = (float)((double)dh / rh);
      p.jx = fr.jitter[0];
      p.jy = fr.jitter[1];
      p.pjx = jpx;
      p.pjy = jpy;
      DevPtr w = wdev["dec0"];
      p12_frame = p12 && recon2_supported(rw, rh, dw, dh);
      if (model_l) {  // (no pack12 for Model L: refused at create)
#ifdef WG_MODEL_L
        if (!l_decoder(p, nxt, fh, fw, f)) return false;
#endif
      } else if (p12_frame && fused_p12) {  // dec1's block with dec0 (pack12) as its epilogue
        DevPtr w1 = wdev["dec1"];
        int ox = off[9][0], oy = off[9][1], h2 = H2, w2 = W2;
        const int nwin = ((W2 + ox + 3) / 4) * ((H2 + oy + 3) / 4);
        const std::string k = std::string("d1p12/swin_block_c64_add_dec0") + (swin_nc["dec1"] ? "_nc" : "");
        if (fz_xd_now) {  // overlapping dec2's fused block: waits on its window rows (dec2's grid, offsets off[8])
          int xoy = off[8][1], xnwx = (W4 + off[8][0] + 3) >> 2;
          dev->unordered_next();
          if (!launch("d1p12/swin_block_c64_add_dec0_xd_nc", cdiv(nwin, 8), 256,
                      {&e1, &skip1, &w1, &h2, &w2, &ox, &oy, &mnear[nxt], &latent[cur], &A, &B, &C, &latent[nxt], &w,
                       &xd_sync, &xoy, &xnwx, &p}))
            return false;
        } else if (!launch(k.c_str(), cdiv(nwin, 8), 256,
                           {&e1, &skip1, &w1, &h2, &w2, &ox, &oy, &mnear[nxt], &latent[cur], &A, &B, &C, &latent[nxt],
                            &w, &p})) {
          return false;
        }
      } else if (p12_frame) {  // pack12 coefficients for the _p12 post kernels
        if (!launch("d0p12/coef_decoder", cdiv((fh / 2) * (fw / 2), 128), 256,
                    {&out1, &mnear[nxt], &latent[cur], &A, &B, &C, &latent[nxt], &w, &p}))
          return false;
      } else if (dec1_dec0) {  // dec1's block with dec0 as its epilogue (dec1's map stays on chip; dec1.out not kept)
        DevPtr w1 = wdev["dec1"];
        int ox = off[9][0], oy = off[9][1], h2 = H2, w2 = W2;
        const int nwin = ((W2 + ox + 3) / 4) * ((H2 + oy + 3) / 4);
        const std::string k = d1d0_pre + "swin_block_c64_add_dec0" + (swin_nc["dec1"] ? "_nc" : "");
        if (!launch(k.c_str(), cdiv(nwin, 8), 256,
                    {&e1, &skip1, &w1, &h2, &w2, &ox, &oy, &mnear[nxt], &latent[cur], &A, &B, &C, &latent[nxt], &w,
                     &p}))
          return false;
      } else if (!launch("coef_decoder", cdiv((fh / 2) * (fw / 2), 128), 256,
                         {&out1, &mnear[nxt], &latent[cur], &A, &B, &C, &latent[nxt], &w, &p})) {
        return false;
      }
    }

    // interop output copy (FrameIo.out_half_pitch > 0): the output is RGBA16F rows of that pitch with pi folded in
    const int hp = io ? io->out_half_pitch : 0;
    if (post_fused && hp > 0 && !(out_f16 && pi == 1.0f && hp == 8 * ow))
      return fail("output copy (RGBA16F rows) with post kernel " + post_kernel +
                  ": needs a fused _o16 kernel, pre-exposure 1 and rows of 8 ow bytes");
    if (out_f16 && hp <= 0 && io && io->out && !io->out_rgba16f)
      return fail("the post kernels write RGBA16F; the caller's output buffer is 3 x f32 (FrameIo.out_rgba16f)");
    if (out_f16 && pi != 1.0f && (post_fused || hp <= 0))
      return fail("RGBA16F output (_o16 kernels) with pre-exposure != 1");
    // D = O with the default reconstruction: post_recon2_r_out writes the final output as RGBA16F when the output
    // buffer takes that format (own buffer, or FrameIo.out_rgba16f) and pi = 1; otherwise post_recon2_r + post_copy
    // (3 x f32, the interop's format).
    recon_out16 = recon_output && post_kernel == "post_recon2_r" &&
                  (hp > 0 ? hp == 8 * ow : (!io || !io->out || io->out_rgba16f)) && pi == 1.0f &&
                  recon2_supported(rw, rh, dw, dh);
    frame_out_f16 = out_f16 || recon_out16 || hp > 0;
    // the banded kernels need the footprint of post_recon2_r; other frames: the unbanded kernel (post_recon where it
    // does not fit, 8-byte history) + the unbanded default filter
    // (and write 3 x f32: not with the output copy)
    // With the interop output copy (hp > 0) the default banded filter's RGBA16F-rows form post_filter_c1_s4_h_band
    // (pi folded in, any pitch) keeps the path banded; bit-identical to the unbanded copy filters.
    // (motion at output resolution: the true-fusion kernel post_tf_*_mo where the default one would run, otherwise
    // post_recon2_r_p12_mo and the unbanded filter; kernels/post_mo.hip)
    const bool band_now = post_band && recon2_supported(rw, rh, dw, dh) &&
                          (hp <= 0 || filter_kernel == "post_filter_c1_s4_band") && !mo;
    const bool tf_mo_route = mo && post_band && (hp <= 0 || filter_kernel == "post_filter_c1_s4_band") && post_tf &&
                             post_diet && post_kernel == "post_recon2_r_band_p12" && filter_kernel == "post_filter_c1_s4_band";
    // (native D3D12, f32 / direct forms: pmo/post_tf4_dl_* where the cap-352 kernel does not fit, or post_tf4=force)
    const bool tf_mo4 = tf_mo_route && d3d12_native && hp <= 0 && tf4_supported(rw, rh, dw, dh) &&
                        (post_tf4_force || !tf_supported(rw, rh, dw, dh, kTfCap));
    const bool tf_mo = tf_mo_route && (tf_mo4 || tf_supported(rw, rh, dw, dh, kTfCap));
    // True fusion: a frame of the default banded pack12 path (post kernels in wave64) whose size post_tf takes runs
    // post_tf_p12 (Quality ... Ultra Performance) or post_tf_dl_p12 (native AA): bit-identical to the banded
    // sequence (runner/post_exact.py), one launch, the display image is not written. Output as the banded path: RGBA16F
    // rows with pi folded in for the interop output copy (hp > 0), otherwise 3 x f32 (then scale_rgb when pi != 1).
    std::string tf_name;  // post_tf_*_diet[_<suffix>] (tf_kernel points into it)
    const char* tf_kernel = nullptr;
    int tf_block = 512;   // post_tf4_dl_*: 256
    if (band_now && post_tf && p12_frame && post_wave == 64 && post_kernel == "post_recon2_r_band_p12" &&
        filter_kernel == "post_filter_c1_s4_band") {
      // Native D3D12 (Brief 15): post_tf4_dl_f32_p12_diet_vsit[_direct] where the cap-352 kernel does not fit (native AA
      // and every ratio between 1.0 and 1.5), or every frame it takes with post_tf4=force; f32 / direct forms only
      const bool tf4 = d3d12_native && hp <= 0 && post_diet && tf_vdiet && post_sel && post_idx && post_trim &&
                       tf4_supported(rw, rh, dw, dh) && (post_tf4_force || !tf_supported(rw, rh, dw, dh, kTfCap));
      if (tf4) {
        tf_name = std::string("post_tf4_dl_f32_p12_diet_vsit") + (native_direct_io ? "_direct" : "");
        tf_kernel = tf_name.c_str();
        tf_block = 256;
      } else if (tf_supported(rw, rh, dw, dh, kTfCap)) {
        if (post_diet) {  // the _diet kernel with the suffix of its options (kernels/post.hip WG_TF_KERNEL_DIET_X)
          std::string sfx = std::string(tf_vdiet ? "v" : "") + (post_sel ? "s" : "") + (post_idx ? "i" : "");
          if (post_trim && sfx == "vsi") sfx += "t";  // post_trim: only the default kernel has the t form
          tf_name = std::string(hp > 0 ? "post_tf_p12_diet" : "post_tf_f32_p12_diet") + (sfx.empty() ? "" : "_" + sfx);
          if (native_direct_io) tf_name += "_direct";
          tf_kernel = tf_name.c_str();
        } else {
          tf_kernel = hp > 0 ? "post_tf_p12" : "post_tf_f32_p12";
        }
      } else if (tf_supported(rw, rh, dw, dh, kTfDlCap)) {
        tf_kernel = hp > 0 ? "post_tf_dl_p12" : "post_tf_dl_f32_p12";
      }
    }
    if (!mark(3)) return false;
    // 5. output reconstruction (spec 06, kernels/post.hip): history copy f -> copy 1 - f, reduced history, display.
    //    The exposures come from the GPU pair (expo); the values in the parameters are only the host-side default.
    {
      const float e = fr.exposure == 0.0f ? 1.0f : fr.exposure;
      PostParams p = {};
      p.rw = rw;
      p.rh = rh;
      p.dw = dw;
      p.dh = dh;
      p.reset = reset;
      p.rx = (float)((double)rw / dw);
      p.ry = (float)((double)rh / dh);
      p.jx = fr.jitter[0];
      p.jy = fr.jitter[1];
      p.e = e;
      p.e_prev = e;
      p.ratio = 1.0f;
      DevPtr w = wdev["post"];
      if (post_fused) {  // reconstruction + output filter in one launch, straight into the output buffer
        if (!fused_supported(rw, rh, dw, dh))
          return fail(post_kernel + ": render size " + std::to_string(rw) + "x" + std::to_string(rh) +
                      " too large for D (fused_supported)");
        int ow_i = ow, oh_i = oh;
        float dec_scale = (float)(0.01 / (fr.exposure == 0.0f ? 1.0 : (double)fr.exposure));
        post_last = post_kernel;
        filter_last = "fused";
        if (!launch((post_pre + post_kernel).c_str(), cdiv(ow, 40) * cdiv(oh, 96), 192,
                    with_taps({&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced,
                               &outp, &p, &ow_i, &oh_i, &dec_scale, &expo})))
          return false;
        if (pi != 1.0f) {  // spec 10 draft 2: the filter's result times pi
          int n3 = 3 * ow * oh;
          float p_ = pi;
          if (!launch("scale_rgb", cdiv(n3, 256), 256, {&outp, &p_, &n3})) return false;
        }
      } else if (tf_mo) {  // reconstruction + output filter in one kernel, motion at output resolution
        int ow_i = ow, oh_i = oh, hpitch = hp > 0 ? hp : 12 * ow, cp = post_tf_cpairs, k1 = 0, c2 = 0;
        const int nch = tf_chunks(ow, oh, cp, &k1, &c2);
        float dec_scale = (float)(0.01 / (fr.exposure == 0.0f ? 1.0 : (double)fr.exposure)), p_h = hp > 0 ? pi : 1.0f;
        post_last = tf_mo4 ? "post_tf4_dl_f32_p12_mo" : hp > 0 ? "post_tf_p12_mo" : "post_tf_f32_p12_mo";
        if (native_direct_io) post_last += "_direct";  // typed colour input and output texture (dx12/kernels/post.hlsl)
        filter_last = "fused";
        if (!launch(("pmo/" + post_last).c_str(), cdiv(ow, 40) * nch, tf_mo4 ? 256 : 512,
                    with_taps({&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mhi, &w, &hist[nxt], &reduced, &outp, &p,
                               &expo, &ow_i, &oh_i, &dec_scale, &p_h, &hpitch, &cp, &k1, &c2, &mop})))
          return false;
        // spec 10 draft 2: the filter's result times pi (folded in with the copy; direct shader stores include it)
        if (!native_direct_io && hp <= 0 && pi != 1.0f) {
          int n3 = 3 * ow * oh;
          float p_ = pi;
          if (!launch("scale_rgb", cdiv(n3, 256), 256, {&outp, &p_, &n3})) return false;
        }
      } else if (mo) {  // reconstruction with motion at output resolution; the unbanded filter follows (step 6)
        post_last = "post_recon2_r_p12_mo";
        if (!launch("pmo/post_recon2_r_p12_mo", cdiv(dw, 32) * cdiv(cdiv(dh, 2), 8), 256,
                    {&cnet, &A, &B, &C, &hist[cur], &mhi, &w, &hist[nxt], &reduced, &display, &p, &expo, &mop}))
          return false;
      } else if (tf_kernel) {  // reconstruction + output filter in one kernel (kernels/post.hip, post_tf*)
        int ow_i = ow, oh_i = oh, hpitch = hp > 0 ? hp : 12 * ow, cp = post_tf_cpairs, k1 = 0, c2 = 0;
        const int nch = tf_chunks(ow, oh, cp, &k1, &c2);
        float dec_scale = (float)(0.01 / (fr.exposure == 0.0f ? 1.0 : (double)fr.exposure)), p_h = hp > 0 ? pi : 1.0f;
        post_last = tf_kernel;
        filter_last = "fused";
        if (!launch((post_pre + tf_kernel).c_str(), cdiv(ow, 40) * nch, tf_block,
                    with_taps({&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced,
                               &outp, &p, &expo, &ow_i, &oh_i, &dec_scale, &p_h, &hpitch, &cp, &k1, &c2})))
          return false;
        if (!native_direct_io && hp <= 0 && pi != 1.0f) {  // direct shader stores already include this final multiply
          int n3 = 3 * ow * oh;
          float p_ = pi;
          if (!launch("scale_rgb", cdiv(n3, 256), 256, {&outp, &p_, &n3})) return false;
        }
      } else if (band_now) {  // reconstruction and filter launched alternately over bands (runner/post.py band_plan)
        post_last = post_kernel;
        filter_last = filter_kernel;
        if (!post_banded(cur, nxt, p, w, cnet, outp, fr, hp, pi)) return false;
        if (hp <= 0 && pi != 1.0f) {  // spec 10 draft 2: the filter's result times pi (folded in with the copy)
          int n3 = 3 * ow * oh;
          float p_ = pi;
          if (!launch("scale_rgb", cdiv(n3, 256), 256, {&outp, &p_, &n3})) return false;
        }
      } else if (recon_out16) {  // D = O: post_recon2_r_out writes the output (RGBA16F), no display, no filter
        float dec_scale = (float)(0.01 / (fr.exposure == 0.0f ? 1.0 : (double)fr.exposure));
        post_last = "post_recon2_r_out";
        filter_last = "none";
        const int tiles_x = cdiv(dw, 32), ntiles = tiles_x * cdiv(cdiv(dh, 2), 8);
        // post_diet: tile / tiles_x on the GPU as mulhi(tile, magic), magic = ceil(2^32 / tiles_x); exact for every
        // tile of this launch (checked here), else the default kernel
        unsigned div_magic = (unsigned)((0x100000000ull + (unsigned)tiles_x - 1) / (unsigned)tiles_x);
        bool diet_ok = post_diet;
        for (int t = 0; diet_ok && t < ntiles; ++t)
          diet_ok = (unsigned)(((unsigned long long)(unsigned)t * div_magic) >> 32) == (unsigned)(t / tiles_x);
        if (diet_ok) {
          post_last = "post_recon2_r_out_diet";
          if (!launch((post_pre + post_last).c_str(), ntiles, 256,
                      {&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced, &outp,
                       &p, &dec_scale, &expo, &div_magic}))
            return false;
        } else if (!launch((post_pre + post_last).c_str(), ntiles, 256,
                           {&cnet, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt], &reduced,
                            &outp, &p, &dec_scale, &expo}))
          return false;
      } else if (!post_recon_stage(rw, rh, cur, nxt, p, w, cnet)) {
        return false;
      }
    }
    // 6. output filter (spec 06) into the output buffer (O, RGB f32); in the fused kernels above
    if (!post_fused && !recon_out16 && !band_now && !tf_kernel && !tf_mo) {
      int a = dw, b = dh, c = ow, d = oh;
      float dec_scale = (float)(0.01 / (fr.exposure == 0.0f ? 1.0 : (double)fr.exposure));
      float p_h = pi;
      int hpitch = hp;
      if (hp > 0 && recon_output) {  // interop output copy, D = O: the decode as RGBA16F rows, pi folded in
        if (!launch((post_pre + "post_copy_h").c_str(), cdiv(ow * oh, 256), 256,
                    {&display, &outp, &c, &d, &dec_scale, &expo, &p_h, &hpitch}))
          return false;
        filter_last = "post_copy_h";
      } else if (hp > 0) {
        // interop output copy: RGBA16F rows. With the default filter, pi = 1 and dense rows (pitch 8 ow):
        // post_filter_c1_s4_o16 (the FP16 rounding of post_filter_c1_s4); otherwise post_filter_h (pi folded in,
        // any pitch). Both equal the FP16 rounding (to nearest even) of the f32 output of the default path.
        const std::string fk = post_band ? std::string(kDefaultFilter) : filter_kernel;
        if (fk == "post_filter_quality9") {
          const int grid = cdiv(ow, 40) * cdiv(cdiv(oh, 2), 8);
          if (!launch((post_pre + "post_filter_quality9_h").c_str(), grid, 64,
                      {&display, &outp, &a, &b, &c, &d, &dec_scale, &expo, &p_h, &hpitch}))
            return false;
          filter_last = "post_filter_quality9_h";
        } else if (fk == "post_filter_c1_s4" && pi == 1.0f && hp == 8 * ow) {
          int tx = 0, fs = 0;
          filter_shape("post_filter_c1_s4_o16", &tx, &fs);
          if (!launch((post_pre + "post_filter_c1_s4_o16").c_str(), cdiv(ow, tx) * cdiv(cdiv(oh, 2), 2 * fs), 64,
                      with_taps({&display, &outp, &a, &b, &c, &d, &dec_scale, &expo})))
            return false;
          filter_last = "post_filter_c1_s4_o16";
        } else {
          const int grid = cdiv(ow, 128) * cdiv(cdiv(oh, 2), 2 * 8);
          if (!launch((post_pre + "post_filter_h").c_str(), grid, 64,
                      with_taps({&display, &outp, &a, &b, &c, &d, &dec_scale, &expo, &p_h, &hpitch})))
            return false;
          filter_last = "post_filter_h";
        }
      } else if (recon_output) {  // experimental D = O: the decode only (identity output filter)
        if (!launch((post_pre + "post_copy").c_str(), cdiv(ow * oh, 256), 256,
                    {&display, &outp, &c, &d, &dec_scale, &expo}))
          return false;
        filter_last = "post_copy";
      } else {
        int tx = filter_tile_x, fs = filter_s;
        const std::string fk = post_band ? std::string(kDefaultFilter) : filter_kernel;  // banded: frame fallback
        if (post_band) filter_shape(fk, &tx, &fs);
        const int grid = cdiv(ow, tx) * cdiv(cdiv(oh, 2), 2 * fs);
        // (post_filter_quality9 is its own filter and never runs with output_filter=original)
        if (!launch((post_pre + fk).c_str(), grid, 64,
                    with_taps({&display, &outp, &a, &b, &c, &d, &dec_scale, &expo})))
          return false;
        filter_last = fk;
      }
      if (hp <= 0 && pi != 1.0f) {  // spec 10 draft 2: the filter's result times pi
        int n3 = 3 * ow * oh;
        float p_ = pi;
        if (!launch("scale_rgb", cdiv(n3, 256), 256, {&outp, &p_, &n3})) return false;
      }
    }

    if (!mark(4)) return false;
    // 7. frame state update
    jprev[0] = fr.jitter[0];
    jprev[1] = fr.jitter[1];
    last_rw = rw;
    last_rh = rh;
    n = (n + 1) % 8;
    f = 1 - f;
    ++frames;
    return true;
  }

  // Named buffers of the last frame (FORMATS.md, "Per-frame runtime").
  bool buffer(const std::string& nm, DevPtr* p, size_t* bytes) {
#ifndef WG_NO_MODEL_K
    if (k) return k->buffer(nm, p, bytes) || fail(k->err());
#endif
    if (frames == 0) return fail("no frame has run yet");
    const int lf = info.parity, rw = info.rw, rh = info.rh, fw = ceil32(rw), fh = ceil32(rh);
    const size_t R = (size_t)rw * rh, t2 = (fw / 2) * (fh / 2), t4 = (fw / 4) * (fh / 4), t8 = (fw / 8) * (fh / 8);
    const size_t Dn = (size_t)dw * dh;
    struct E {
      DevPtr p;
      size_t b;
    };
    std::map<std::string, E> m = {
        {"exposure.pair", {expo, 12}},
        {"enc0.features", {feat0, t2 * 64}},  {"enc0.m_near", {mnear[1 - lf], R * 4}},
        {"enc0.m_far", {mfar, R * 4}},        {"enc0.m_center", {mcenter, R * 4}},
        {"enc1.skip", {skip1, t2 * 64}},      {"enc1.merged", {merged1, t4 * 96}},
        {"enc2.skip", {skip2, t4 * 96}},      {"enc2.merged", {merged2, t8 * 128}},
        {"dec2.out", {out2, t4 * 96}},        {"dec1.out", {out1, t2 * 64}},
        {"dec0.A", {A, R * (p12 ? 8 : 4)}},               {"dec0.B", {B, R * 4}},
        {"dec0.C", {C, R * 8}},               {"dec0.latent", {latent[1 - lf], R * 4}},
        {"post.history", {hist[1 - lf], Dn * hist_bytes}}, {"post.reduced", {reduced, (size_t)hw * hh * 4}},
        {"post.display", {display, display_bytes(dw, dh)}},
        {"filter.out", {out, (size_t)ow * oh * (frame_out_f16 ? 8 : 12)}},
        {"state.latent_prev", {latent[lf], R * 4}}, {"state.motion_history", {mnear[lf], R * 4}},
        {"state.history_prev", {hist[lf], Dn * hist_bytes}}, {"input.color", {color, R * 16}},
        {"input.depth", {depth, R * 4}},      {"input.motion", {motion, R * 8}},
    };
    for (int i = 0; i < 6; ++i) m["b" + std::to_string(i) + ".out"] = {bout[i], t8 * 128};
#ifdef WG_MODEL_L
    if (model_l) {  // Model L's maps (FORMATS.md "Maps"; the frame reuses them: q / x2 / y2, x1 / y, x3)
      const size_t halo = (size_t)(fw + 4) * (fh + 4);
      m["l.wide"] = {l_wide, halo * 64};
      m["l.narrow"] = {l_narrow, halo * 32};
      m["l.x3"] = {l_x3, (size_t)fw * fh * 32};
    }
#endif
    auto it = m.find(nm);
    if (it == m.end()) return fail("unknown buffer name " + nm);
    *p = it->second.p;
    *bytes = it->second.b;
    return true;
  }
};

Runtime* runtime_create(const WgRtConfig& cfg, std::unique_ptr<Device> dev, std::string* err, bool external_io,
                        bool native_direct_io) {
  std::unique_ptr<Runtime> r(new Runtime());
  if (!r->create(cfg, std::move(dev), external_io, native_direct_io)) {
    *err = r->err;
    return nullptr;
  }
  return r.release();
}
void runtime_destroy(Runtime* rt) { delete rt; }
bool runtime_frame(Runtime* rt, const WgRtFrame& frame, const FrameIo* io) { return rt->frame(frame, io); }
bool runtime_direct_preflight(Runtime* rt, const WgRtFrame& frame, const FrameIo* io) {
  return rt->direct_preflight(frame, io);
}
const std::string& runtime_error(Runtime* rt) { return rt->err; }
Device* runtime_device(Runtime* rt) { return rt->dev.get(); }
float runtime_effective_pre_exposure(Runtime* rt, float value) { return rt->effective_pre_exposure(value); }
const char* runtime_kernels(Runtime* rt) { return rt->kernels(); }
const char* runtime_notes(Runtime* rt) { return rt->notes_text.c_str(); }
uint64_t runtime_memory_bytes(Runtime* rt) { return rt->owned_bytes; }
#ifdef WG_NO_MODEL_K
const char* runtime_timing_info(Runtime*) { return nullptr; }
#else
const char* runtime_timing_info(Runtime* rt) { return rt->k ? KRuntime::timing_info() : nullptr; }
#endif
const char* const* runtime_timing_labels(Runtime* rt) {
  static const char* const kM[4] = {"enc0", "Swin", "dec0", "post"};
#ifdef WG_NO_MODEL_K
  return (void)rt, kM;
#else
  return rt->k ? KRuntime::kTimingLabels : kM;
#endif
}
#ifdef WG_NO_MODEL_K
const char* runtime_model(Runtime* rt) { return model_info(rt->model_l ? ModelId::L : ModelId::M).name; }
#else
const char* runtime_model(Runtime* rt) {
  return model_info(rt->k ? ModelId::K : rt->model_l ? ModelId::L : ModelId::M).name;
}
#endif
void runtime_recon_size(Runtime* rt, int* dw, int* dh) {
  *dw = rt->dw;
  *dh = rt->dh;
}

}  // namespace wgrt

using wgrt::Runtime;

extern "C" {

static int create_runtime(const WgRtConfig* cfg, void** rt, char* err, int errlen, bool external) {
  *rt = nullptr;
  std::unique_ptr<Runtime> r(new Runtime());
  if (!cfg || !r->create(*cfg, nullptr, external)) {
    const std::string m = cfg ? r->err : "no configuration";
    if (err && errlen > 0) std::snprintf(err, (size_t)errlen, "%s", m.c_str());
    return 1;
  }
  *rt = r.release();
  return 0;
}
WG_RT_API int wg_rt_create(const WgRtConfig* cfg, void** rt, char* err, int errlen) {
  return create_runtime(cfg, rt, err, errlen, false);
}
WG_RT_API int wg_rt_recon_size(void* rt, int* dw, int* dh) {
  wgrt::runtime_recon_size((Runtime*)rt, dw, dh);
  return 0;
}
WG_RT_API int wg_rt_create_external(const WgRtConfig* cfg, void** rt, char* err, int errlen) {
  return create_runtime(cfg, rt, err, errlen, true);
}
WG_RT_API int wg_rt_alloc(void* rt, size_t bytes, uintptr_t* ptr) {
  wgrt::DevPtr p = 0;
  if (!((Runtime*)rt)->alloc(&p, bytes)) return 1;
  *ptr = p;
  return 0;
}
WG_RT_API int wg_rt_upload(void* rt, uintptr_t dst, const void* src, size_t bytes) {
  Runtime* r = (Runtime*)rt;
  return r->dev->upload(dst, src, bytes) ? 0 : (r->dfail(), 1);
}
WG_RT_API int wg_rt_frame_io(void* rt, const WgRtFrame* fr, const WgRtExtIo* x) {
  wgrt::FrameIo io;
  io.color = x->color;
  io.depth = x->depth;
  io.motion = x->motion;
  io.out = x->out;
  io.exposure_mode = x->exposure_mode;
  io.expo_src = x->expo_src;
  io.raw_inputs = x->raw_inputs != 0;
  io.color_pitch = x->color_pitch;
  io.depth_pitch = x->depth_pitch;
  io.motion_pitch = x->motion_pitch;
  io.motion_channels = x->motion_channels;
  for (int k = 0; k < 2; ++k) io.ms[k] = x->ms[k], io.mo[k] = x->mo[k];
  io.out_half_pitch = x->out_half_pitch;
  return ((Runtime*)rt)->frame(*fr, &io) ? 0 : 1;
}
WG_RT_API int wg_rt_frame_io_timed(void* rt, const WgRtFrame* fr, const WgRtExtIo* x, float ms[4]) {
  // the interop's timing marks (FrameIo.timing_base) on events 8 ... 12, read after the frame (waits)
  Runtime* r = (Runtime*)rt;
  constexpr int kBase = 8;
  if (!r->dev->reserve_events(kBase + 5)) return r->dfail(), 1;
  wgrt::FrameIo io;
  io.color = x->color;
  io.depth = x->depth;
  io.motion = x->motion;
  io.out = x->out;
  io.exposure_mode = x->exposure_mode;
  io.expo_src = x->expo_src;
  io.raw_inputs = x->raw_inputs != 0;
  io.color_pitch = x->color_pitch;
  io.depth_pitch = x->depth_pitch;
  io.motion_pitch = x->motion_pitch;
  io.motion_channels = x->motion_channels;
  for (int k = 0; k < 2; ++k) io.ms[k] = x->ms[k], io.mo[k] = x->mo[k];
  io.out_half_pitch = x->out_half_pitch;
  io.timing_base = kBase;
  if (!r->frame(*fr, &io)) return 1;
  if (!r->dev->sync()) return r->dfail(), 1;
  for (int k = 0; k < 4; ++k)
    if (!r->dev->elapsed_ms(kBase + k, kBase + k + 1, &ms[k])) return r->dfail(), 1;
  return 0;
}
WG_RT_API const char* wg_rt_timing_labels(void* rt) {
  static thread_local std::string s;
  const char* const* l = wgrt::runtime_timing_labels((Runtime*)rt);
  s = std::string(l[0]) + "," + l[1] + "," + l[2] + "," + l[3];
  return s.c_str();
}
WG_RT_API int wg_rt_frame_external(void* rt, const WgRtFrame* fr, uintptr_t color, uintptr_t depth, uintptr_t motion,
                                   uintptr_t out, int exposure_mode, uintptr_t expo_src) {
  wgrt::FrameIo io;
  io.color = color;
  io.depth = depth;
  io.motion = motion;
  io.out = out;
  io.exposure_mode = exposure_mode;
  io.expo_src = expo_src;
  return ((Runtime*)rt)->frame(*fr, &io) ? 0 : 1;
}

WG_RT_API void wg_rt_destroy(void* rt) { delete (Runtime*)rt; }
WG_RT_API const char* wg_rt_error(void* rt) { return rt ? ((Runtime*)rt)->err.c_str() : "no runtime"; }

WG_RT_API int wg_rt_upload_inputs(void* rt, int rw, int rh, const float* c, const float* d, const float* m) {
  return ((Runtime*)rt)->upload_inputs(rw, rh, c, d, m) ? 0 : 1;
}
WG_RT_API int wg_rt_upload_motion_hi(void* rt, int mw, int mh, int wide, const void* texels, const float* mv_scale,
                                     const float* b_mv) {
  return ((Runtime*)rt)->upload_motion_hi(mw, mh, wide, texels, mv_scale, b_mv) ? 0 : 1;
}
WG_RT_API int wg_rt_frame(void* rt, const WgRtFrame* fr) { return ((Runtime*)rt)->frame(*fr) ? 0 : 1; }
WG_RT_API int wg_rt_frame_timed(void* rt, const WgRtFrame* fr, float* ms) {
  Runtime* r = (Runtime*)rt;
  if (!r->dev->record(0)) return r->dfail(), 1;
  if (!r->frame(*fr)) return 1;
  if (!r->dev->record(1) || !r->dev->elapsed_ms(0, 1, ms)) return r->dfail(), 1;
  return 0;
}
WG_RT_API int wg_rt_sync(void* rt) {
  Runtime* r = (Runtime*)rt;
  return r->dev->sync() ? 0 : (r->dfail(), 1);
}
WG_RT_API int wg_rt_info(void* rt, WgRtFrameInfo* info) {
  *info = ((Runtime*)rt)->info;
  return 0;
}
WG_RT_API int wg_rt_buffer(void* rt, const char* name, uintptr_t* ptr, size_t* bytes) {
  wgrt::DevPtr p = 0;
  if (!((Runtime*)rt)->buffer(name, &p, bytes)) return 1;
  *ptr = (uintptr_t)p;
  return 0;
}
WG_RT_API int wg_rt_download(void* rt, uintptr_t ptr, void* dst, size_t bytes) {
  Runtime* r = (Runtime*)rt;
  return r->dev->download(dst, (wgrt::DevPtr)ptr, bytes) ? 0 : (r->dfail(), 1);
}
WG_RT_API const char* wg_rt_kernels(void* rt) { return ((Runtime*)rt)->kernels(); }
WG_RT_API const char* wg_rt_notes(void* rt) { return ((Runtime*)rt)->notes_text.c_str(); }
WG_RT_API const char* wg_rt_model_source(void* rt) { return ((Runtime*)rt)->model_source.c_str(); }
WG_RT_API int wg_rt_stage_blob(void* rt, const char* stage, const void** data, size_t* bytes) {
  Runtime* r = (Runtime*)rt;
  auto it = r->blobs.find(stage);
  if (it == r->blobs.end()) return r->fail(std::string("no blob for stage ") + stage), 1;
  *data = it->second.data();
  *bytes = it->second.size();
  return 0;
}

}  // extern "C"
