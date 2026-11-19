// SPDX-License-Identifier: MIT
// Model K in the per-frame runtime; see runtime_k.h. Port of clean/k/runner/k_frame.py (KFrame), k_net.py (KNet,
// variant "fused"), k_launch.py (stage_fused_launch) and k_io.py (frame_in, out_uniforms, input / output launches).
#include "runtime_k.h"
#include "k_input_uniforms.h"
#include "k_mix_probe.h"
#include "k_s_breakpoints.h"
#include "k_smx_selfcheck.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "../../converter/src/json.h"
#include "../../converter/src/sha256.h"
#include "../../runtime/embedded.h"
#include "../../runtime/fpconv.h"
#include "../../runtime/runtime_core.h"
#include "k_filter_table.h"
#include "k_pack.h"
#include "model_k.h"
#include "model_k_load.h"

// The host-side FP32 values of k_io.py out_uniforms: each operation rounded on its own (no contraction).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

namespace wgrt {

namespace {

// Kernel parameter structs, passed by value; must match clean/k/kernels/k_io.hip, k_block.hip and k_block_wide.hip
// (static_asserts there and in clean/k/runner/k_io.py, k_launch.py).
struct KFrameIn {
  DevPtr color, motion, depth, xs;
  int cpitch, mpitch, zpitch, cbx, cby, mbx, mby, zbx, zby, W, H, rw, rh, Bw, Bh, flags;
  float jx, jy, sx, sy, mox, moy, kappa, rho, lam, s;
  int mch;
  float rmsx, rmsy, rmox, rmoy;
  int pad_;
};
static_assert(sizeof(KFrameIn) == 160, "KFrameIn");
struct KOutUni {
  float invW, invH, rhox, rhoy, rho2x, rho2y, jmx, jmy, spx, spy, invW16, invH16, kinv;
  uint16_t kux, kvy;
};
static_assert(sizeof(KOutUni) == 56, "KOutUni");
struct KOutParams {
  KFrameIn f;
  DevPtr N, L, Hc, table, exp2tab, out, Lout, Hcout, Hhout;
  int opitch, omode;
  DevPtr ctap, ctapT;
  KOutUni u;
};
static_assert(sizeof(KOutParams) == 312, "KOutParams");
struct KInParams {
  KFrameIn f;
  DevPtr Hc, Hh, x0, cpre, cpre2;
};
static_assert(sizeof(KInParams) == 200, "KInParams");
struct KInColorParams { KInParams input; DevPtr ctap, ctapT; };
static_assert(sizeof(KInColorParams) == 216, "KInColorParams");
struct KInUniformParams { KInParams input; KiHost host; };
struct KInColorUniformParams { KInColorParams input; KiHost host; };
static_assert(sizeof(KInUniformParams) == 248 && sizeof(KInColorUniformParams) == 264, "input host arguments");
static_assert(offsetof(KInUniformParams, host) == 200 && offsetof(KInColorUniformParams, host) == 216, "input host offsets");
struct KExpoArgs {  // k_io.hip KExpoArgs
  float value;
  int src_mode, aux, first, hdr, reset;
  float e1, e2, inv_pi;
};
static_assert(sizeof(KExpoArgs) == 36, "KExpoArgs");
struct KBlkParams {
  DevPtr x, skip, y, merged, head, g1, g2, wqkv, mix, wo, bo, w1, b1, w2, b2, wa, ba, wz, bz;
  int w, h, ox, oy, nwx, nwy, cw, ch;
};
static_assert(sizeof(KBlkParams) == 184, "KBlkParams");
// k_warm (clean/k/dx12/kernels/k_warm.hlsl, D3D12 only): up to twelve byte ranges r[i] of n[i] bytes, one 4-byte load
// per `line` bytes of each, the XOR of the loaded words stored to dummy only when it equals key (a word nobody reads)
struct KWarmParams {
  DevPtr r[12], dummy;
  uint32_t n[12], line, key;
};
static_assert(sizeof(KWarmParams) == 160, "KWarmParams");
struct ExpParams {  // clean/kernels/exposure.hip WgExpParams
  int rw, rh, nx, ny, nparts, pad0;
  float inv_pi, K, inv_n;
};
static_assert(sizeof(ExpParams) == 36, "ExpParams");

enum { KIO_HDR = 1, KIO_MVLO = 2, KIO_INV = 4, KIO_RESET = 8, KIO_C16 = 16, KIO_M16 = 32 };
enum { OUT_RGBA32F = 0, OUT_RGBA16F = 1, OUT_RGB32F = 2 };
constexpr int kThreads = 128;  // k_io.py THREADS, k_launch.py THREADS
constexpr float kS = 0.5f;     // spec 44 draft 3 base-size factor (clean/k/reference/layout.py DEFAULT_S)
constexpr size_t kTableBytes = (size_t)kFilterEntries * kFilterValues * 2;

int cdiv(long long a, long long b) { return (int)((a + b - 1) / b); }
int ceil16(int n) { return (n + 15) / 16 * 16; }
size_t al256(size_t n) { return (n + 255) / 256 * 256; }
size_t pad16(size_t n) { return (n + 15) / 16 * 16; }

// k_pack.py linear_size / linear_sections / block_sections / stage_sections (offsets inside a stage blob)
size_t linear_size(size_t n, size_t k) { return al256(pad16(n) * k * 2) + al256(pad16(n) * 2); }
size_t linear_bias(size_t n, size_t k) { return al256(pad16(n) * k * 2); }
struct BlockSec {
  size_t g1, g2, qkv, mix, wo, bo, w1, b1, w2, b2, size;
};
BlockSec block_sections(const KStageInfo& s) {
  const size_t C = s.C, D = s.D(), H = s.heads, HID = s.hidden();
  const size_t sizes[10] = {C * 2, C * 2, (s.fixed ? 1 : 3) * H * 32 * C * 2, H * 64 * 64 * 2, C * D * 2, C * 2,
                            HID * C * 2, HID * 2, C * HID * 2, C * 2};
  size_t off[10], o = 0;
  for (int i = 0; i < 10; ++i) {
    off[i] = o;
    o += al256(sizes[i]);
  }
  return {off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7], off[8], off[9], o};
}
struct StageSec {
  size_t embed = 0, expand = 0, block = 0, merge = 0, head = 0;
};
StageSec stage_sections(const KStageInfo& s) {  // order: embed, expand, block, merge, head
  StageSec r;
  size_t o = 0;
  if (s.embed) r.embed = o, o += linear_size(32, 16);
  if (s.kind == 'd') r.expand = o, o += linear_size(4 * s.C, s.C_prev);
  r.block = o;
  o += block_sections(s).size;
  if (s.kind == 'e') r.merge = o, o += linear_size(s.C_next, 4 * s.C);
  if (s.head) r.head = o;
  return r;
}

// Bias widening for the FP32 planes. Preserve the binary16 NaN payload and quiet signaling NaNs.
uint32_t k_bias32_bits(uint16_t h) {
  const uint32_t payload = h & 0x3ffu;
  if ((h & 0x7c00u) == 0x7c00u && payload)
    return ((uint32_t)(h & 0x8000u) << 16) | 0x7fc00000u | (payload << 13);
  return f32_bits(f16_to_f32(h));
}

std::vector<uint32_t> k_narrow_bias32(const KStageInfo& s, const uint8_t* packed,
                                      const uint8_t* compact_head = nullptr) {
  const auto stage = stage_sections(s);
  const auto block = block_sections(s);
  std::vector<uint32_t> plane;
  const auto append = [&](const uint8_t* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      uint16_t half;
      std::memcpy(&half, data + i * sizeof(half), sizeof(half));
      plane.push_back(k_bias32_bits(half));
    }
  };
  if (s.embed) append(packed + stage.embed + linear_bias(s.C, 16), s.C);
  if (s.kind == 'd') append(packed + stage.expand + linear_bias(4 * s.C, s.C_prev), 4 * s.C);
  append(packed + stage.block + block.bo, s.C);
  append(packed + stage.block + block.b1, 4 * s.C);
  append(packed + stage.block + block.b2, s.C);
  if (s.kind == 'e') append(packed + stage.merge + linear_bias(s.C_next, 4 * s.C), s.C_next);
  if (s.head) {
    append(compact_head + linear_bias(32, s.C), 32);
    plane.resize(plane.size() + 16, 0);
  }
  if (!s.fixed) append(packed + stage.block + block.mix, (size_t)s.heads * 64 * 64);
  return plane;
}

// ---- network=fused8: FP8 forms of the level-0/1 stages (clean/k/kernels/k_block8.h) --------------------------------
static_assert(sizeof(KF8Params) == 104, "KF8Params of k_block8.h");
float blob_h(const std::vector<uint8_t>& b, size_t off, size_t i) {
  uint16_t v;
  std::memcpy(&v, &b[off + 2 * i], 2);
  return f16_to_f32(v);
}
// largest k with m 2^k <= 448 (m > 0, finite)
int fit448(double m) { return (int)std::floor(std::log2(448.0 / m)); }
// The FP8 data of stage s (e0, e1, d1 or d0) from its FP16 blob. Scales (k_block8.h "Scales"): weights by the largest
// power of two that keeps them <= 448; n unscaled (|n_c| <= |g1_c|); v by the bound |v_j| <= sum_c |Wv_jc| |g1_c|;
// z by its bound (fixed mixing: max_t sum_s |M_ts| times v's; softmax: v's); a unscaled and clamped at 448; h 2^1.
F8Stage fp8_stage(const KStageInfo& s, const std::vector<uint8_t>& blob) {
  F8Stage r;
  const StageSec sec = stage_sections(s);
  const BlockSec bs = block_sections(s);
  const size_t blk = sec.block, C = s.C, H = s.heads, KT = C / 16, QKV = s.fixed ? 1 : 3, HID = s.hidden(),
               D = s.D();
  auto maxabs = [&](size_t off, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(blob_h(blob, off, i)));
    return m;
  };
  std::vector<double> g1(C);
  double g1max = 0;
  for (size_t c = 0; c < C; ++c) g1max = std::max(g1max, g1[c] = std::fabs(blob_h(blob, blk + bs.g1, c)));
  // v: tile (head hd, ct, kt) of the qkv section, B layout: lane l = column j % 16 + 16 hh, element e = channel
  // 16 kt + 8 hh + e
  double wvmax = 0, bv = 0;
  for (size_t hd = 0; hd < H; ++hd)
    for (size_t j = 0; j < 32; ++j) {
      double row = 0;
      const size_t ct = j / 16;
      for (size_t c = 0; c < C; ++c) {
        const size_t kt = c / 16, hh = (c % 16) / 8, e = c % 8;
        const size_t tile = hd * QKV * 2 * KT + (QKV - 1) * 2 * KT + ct * KT + kt;
        const double w = std::fabs(blob_h(blob, blk + bs.qkv, tile * 256 + (j % 16 + 16 * hh) * 8 + e));
        wvmax = std::max(wvmax, w);
        row += w * g1[c];
      }
      bv = std::max(bv, row);
    }
  bv *= 1.01;  // |x r| <= 1 up to the FP16 roundings of s and r
  double rm = 0, mmax = 0;
  if (s.fixed)
    for (size_t t = 0; t < H * 64; ++t) {
      double row = 0;
      for (size_t k = 0; k < 64; ++k) {
        const double m = std::fabs(blob_h(blob, blk + bs.mix, t * 64 + k));
        row += m;
        mmax = std::max(mmax, m);
      }
      rm = std::max(rm, row);
    }
  const double wo = maxabs(blk + bs.wo, C * D), w1 = maxabs(blk + bs.w1, HID * C), w2 = maxabs(blk + bs.w2, C * HID);
  for (double v : {g1max, wvmax, bv, wo, w1, w2})
    if (!std::isfinite(v) || v <= 0) return r.why = "weights not finite or all zero", r;
  if (s.fixed && (!std::isfinite(rm) || rm <= 0)) return r.why = "mixing matrix not finite or zero", r;
  if (s.fixed && 1.01 * g1max >= 448.0) return r.why = "|g1| >= 448", r;
  const int kv = s.fixed ? fit448(std::max(bv, wvmax)) : 0, km = s.fixed ? fit448(mmax) : 0;
  const int kpv = fit448(bv), kz = fit448(s.fixed ? rm * bv : bv), ko = fit448(wo), k1 = fit448(w1),
            k2 = fit448(w2), kh = 1;
  const int m = kh - k1;
  r.q.pv = std::ldexp(1.0f, kpv);
  r.q.sz = s.fixed ? std::ldexp(1.0f, kz - kv - km) : std::ldexp(1.0f, kz - kpv - 8);
  r.q.sy = std::ldexp(1.0f, -(ko + kz));
  r.q.L = std::ldexp(1.0f, k1 + 1);
  r.q.Hs = std::ldexp(440.0f, -m);
  r.q.A2 = std::ldexp(0.412109375f, m - k1);
  r.q.B2 = std::ldexp(0.0810546875f, m - 2 * k1);
  r.q.c0 = std::ldexp(1.0f, m - 1);
  r.q.sx = std::ldexp(1.0f, -(k2 + kh));
  // the buffer
  std::vector<uint8_t>& b = r.bytes;
  auto part = [&](size_t n) {
    const size_t o = b.size();
    b.resize(al256(o + n), 0);
    return (DevPtr)o;
  };
  auto f8 = [&](DevPtr dst, size_t src, size_t n, int k) {
    for (size_t i = 0; i < n; ++i) b[dst + i] = fp8_encode(std::ldexp((double)blob_h(blob, src, i), k));
  };
  auto f32 = [&](DevPtr dst, size_t src, size_t n, int k) {
    for (size_t i = 0; i < n; ++i) {
      const float v = std::ldexp(blob_h(blob, src, i), k);
      std::memcpy(&b[dst + 4 * i], &v, 4);
    }
  };
  const size_t nq = QKV * H * 32 * C;
  r.q.wqkv = part(nq);
  r.q.mix = part(s.fixed ? H * 64 * 64 : 0);
  r.q.wo = part(C * D);
  r.q.w1 = part(HID * C);
  r.q.w2 = part(C * HID);
  r.q.bo = part(4 * C);
  r.q.b1 = part(4 * HID);
  r.q.b2 = part(4 * C);
  f8(r.q.wqkv, blk + bs.qkv, nq, kv);
  if (s.fixed) f8(r.q.mix, blk + bs.mix, H * 64 * 64, km);
  f8(r.q.wo, blk + bs.wo, C * D, ko);
  f8(r.q.w1, blk + bs.w1, HID * C, k1);
  f8(r.q.w2, blk + bs.w2, C * HID, k2);
  f32(r.q.bo, blk + bs.bo, C, ko + kz);
  f32(r.q.b1, blk + bs.b1, HID, k1);
  f32(r.q.b2, blk + bs.b2, C, k2 + kh);
  return r;
}
bool fp8_stage_name(const char* n) {
  return !std::strcmp(n, "e0") || !std::strcmp(n, "e1") || !std::strcmp(n, "d1") || !std::strcmp(n, "d0");
}

// layout.py FUSED_IO (the fused network: merged maps and decoder outputs alternate between A and B)
struct FusedIo {
  const char *in, *skip, *block, *merged, *head;
};
const FusedIo kFusedIo[11] = {
    {"X0", nullptr, "S0", "A", nullptr}, {"A", nullptr, "S1", "B", nullptr}, {"B", nullptr, "S2", "A", nullptr},
    {"A", nullptr, "S3", "B", nullptr},  {"B", nullptr, "S4", "A", nullptr}, {"A", nullptr, "B", nullptr, nullptr},
    {"B", "S4", "A", nullptr, nullptr},  {"A", "S3", "B", nullptr, nullptr}, {"B", "S2", "A", nullptr, nullptr},
    {"A", "S1", "B", nullptr, nullptr},  {"B", "S0", nullptr, nullptr, "OUT"},
};
// k_launch.py WIDE_DEFAULT (k_net.py DEFAULT_PATHS of variant "fused"): k_stagew<NG>_<name>; 0 = k_stage_<name>.
// f16 (variant "fused16", WIDE_DEFAULT16): k_stagew4h_<name> for e4, m, d4; 0 = k_stageh_<name>.
// FP32: e4 and d4 with NG = 4 when they have at most 24 windows (k_launch.py WIDE4_SMALL_MAX_WINDOWS; 1440p output: 24,
// faster with 4; 4K output: 40, faster with 2).
// Defaults of the D3D12 latency options k_color_at / k_warm (KRuntime::create).
const char* const kColorAtDefault = "input";
const char* const kWarmAtDefault = "e0";
const char* const kWarmToDefault = "d0";

// SMX-SELFCHECK-SOURCE BEGIN (wide_ng, block_params, stage_kernel_base: the self-check reference depends on them)
int wide_ng(const std::string& s, bool f16, int nwin) {
  if (f16) return s == "e4" || s == "m" || s == "d4" ? 4 : 0;
  if (s == "m") return 4;
  if (s == "e4" || s == "d4") return nwin <= 24 ? 4 : 2;
  if (s == "e3" || s == "d3") return 2;
  return 0;
}

// The block parameters of stage st (KBlkParams): its weight blob, its maps and the window grid of a level map of lw x lh
// with window offset (ox, oy); iw x ih is a decoder's input map. network() and the softmax self-check (smx_check).
KBlkParams block_params(const KStageInfo& st, DevPtr blob, DevPtr x, DevPtr y, DevPtr skip, DevPtr merged, DevPtr head,
                        int lw, int lh, int iw, int ih, int ox, int oy) {
  const StageSec sec = stage_sections(st);
  const BlockSec bs = block_sections(st);
  const DevPtr blk = blob + sec.block;
  KBlkParams p = {};
  p.x = x;
  p.y = y;
  p.g1 = blk + bs.g1;
  p.g2 = blk + bs.g2;
  p.wqkv = blk + bs.qkv;
  p.mix = blk + bs.mix;
  p.wo = blk + bs.wo;
  p.bo = blk + bs.bo;
  p.w1 = blk + bs.w1;
  p.b1 = blk + bs.b1;
  p.w2 = blk + bs.w2;
  p.b2 = blk + bs.b2;
  p.w = lw;
  p.h = lh;
  p.ox = ox;
  p.oy = oy;
  p.nwx = cdiv(lw + ox, 8);  // layout.py window_grid
  p.nwy = cdiv(lh + oy, 8);
  if (st.embed) {
    p.wa = blob + sec.embed;
    p.ba = blob + sec.embed + linear_bias(32, 16);
  }
  if (st.kind == 'd') {
    p.wa = blob + sec.expand;
    p.ba = blob + sec.expand + linear_bias(4 * st.C, st.C_prev);
    p.skip = skip;
    p.cw = iw;
    p.ch = ih;
  }
  if (st.kind == 'e') {
    p.wz = blob + sec.merge;
    p.bz = blob + sec.merge + linear_bias(st.C_next, 4 * st.C);
    p.merged = merged;
  }
  if (st.head) {
    p.wz = blob + sec.head;
    p.bz = blob + sec.head + linear_bias(40, st.C);
    p.head = head;
  }
  return p;
}

// The stage kernel of a non-head stage as network() launches it (before a __rodd suffix): k_stagew<NG>_<name> for the
// wide stages, k_stage2_<name> (two windows per workgroup) for e0, e1, d1 with its block output map and d0, else
// k_stage_<name>; *ng = NG (0: not wide), *w2 = two windows per workgroup.
std::string stage_kernel_base(const KStageInfo& st, bool f16, bool has_y, int nwin, int* ng, bool* w2) {
  *ng = wide_ng(st.name, f16, nwin);
  // e0, e1, d1, d0: two windows per workgroup (k_stage2_*, k_launch.py WINDOWS2_STAGES; k_net.py WINDOWS2_VARIANTS;
  // d1's needs its block output map p.y, which the network always gives it)
  *w2 = !*ng && (std::strcmp(st.name, "e0") == 0 || std::strcmp(st.name, "e1") == 0 ||
                 (std::strcmp(st.name, "d1") == 0 && has_y) || std::strcmp(st.name, "d0") == 0);
  const std::string h = f16 ? "h" : "";
  return (*ng ? "k_stagew" + std::to_string(*ng) : std::string(*w2 ? "k_stage2" : "k_stage")) + h + "_" + st.name;
}
// SMX-SELFCHECK-SOURCE END

// Runtime options of model K (runtime_options, then WG_RUNTIME_OPTIONS whose keys win): model_k_dir, model_k_map,
// network (fused, the default; fused16: FP16 accumulation, the spec 49 speed approximation, opt-in), k_diag (timing
// diagnostics, same output: "input2" runs the input stage once more before its timing mark, so the "input" part is a
// second run and the first is counted in "exposure"; "copyin" copies the interop's colour, depth and motion into the
// own buffers first (hipMemcpyAsync, before the first mark); "input2+copyin" both), k_head (compact, the default: d0 writes
// the compact pixel head for k_out_pixel_c; full: the 40-channel head map and k_out_pixel, same output bits, and the
// buffer "k.N" for diagnostics), k_fast (0, the default: the exact stage kernels; 1: opt-in, not bit-exact, the "_f"
// variants of the input and output stage kernels, see k_io.hip KIO_FAST; it brings the fast input stage with it),
// k_fast_in (0, the default: the input-stage kernels from k_io_exact.co (WGK_IN_EXACT, every operation correctly
// rounded: bit for bit spec 42, as KFrame(exact_input=True)), with the output colour taps prepared by k_out_color as
// KFrame's exact plan does; 1: opt-in speed mode, the input stage from k_io.co, whose log2 / 2^x / reciprocals are the
// GPU's estimates (spec 42 section 9: within one FP16 step; not bit-exact)), k_in_exact (the older name: k_in_exact=1 is the default, k_in_exact=0 is k_fast_in=1; k_in_exact=1
// is refused with k_fast=1 or k_fast_in=1), k_expo_rule (spec, the default: spec 42 draft 3 section 4, E_cur = the
// game's exposure or 1, E_aux = the auto-exposure value; v051: opt-in, for A/B only, the 0.5.1 rule E_cur = E_aux =
// the auto-exposure value when it runs).
// whether key= appears in the runtime options or WG_RUNTIME_OPTIONS (also with an empty value)
bool option_given(const WgRtConfig& c, const std::string& key) {
  const char* env = std::getenv("WG_RUNTIME_OPTIONS");
  const std::string text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find(';', a);
    if (b == std::string::npos) b = text.size();
    if (text.compare(a, key.size() + 1, key + "=") == 0) return true;
    a = b + 1;
  }
  return false;
}
// a value for a one-line message: control characters as '?'
std::string one_line(std::string v) {
  for (char& ch : v)
    if ((unsigned char)ch < 32 || ch == 127) ch = '?';
  return v;
}

std::string option(const WgRtConfig& c, const std::string& key) {
  const char* env = std::getenv("WG_RUNTIME_OPTIONS");
  const std::string text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
  std::string v;
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find(';', a);
    if (b == std::string::npos) b = text.size();
    const std::string kv = text.substr(a, b - a);
    a = b + 1;
    if (kv.compare(0, key.size() + 1, key + "=") == 0) v = kv.substr(key.size() + 1);
  }
  return v;
}

std::string join(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  const char last = dir.back();
  return last == '/' || last == '\\' ? dir + name : dir + "/" + name;
}

}  // namespace

const char* const KRuntime::kTimingLabels[4] = {"exposure", "input", "network", "output"};

const char* KRuntime::timing_info() {
  return "model K: HIP events per frame, in stream order: [0] start of the first kernel after the input wait "
         "(exposure_auto_diet, or exposure_auto_diet_c16 with input_copy, if auto-exposure, else k_expo_scalars) -> [1] start "
         "of input/color preparation -> [2] start of the first neural kernel -> [3] start of k_out_pixel (k_out_pixel_c with the compact pixel head) -> [4] end of "
         "an empty kernel after the final output-stage kernel; exposure = [0]-[1] (input conversion, exposure, k_expo_scalars), input "
         "= [1]-[2] (k_in_tiled_color[_uniform], or separate k_out_color followed by k_in_tiled[_uniform] or k_in_color plus k_in_base/k_in_reset), network = [2]-[3] (eleven stage kernels, including hidden-history output in d0; on the D3D12 devices also "
         "k_out_color, launched unordered after the stage k_color_at names, and k_warm when k_warm is on), output = [3]-[4] ("
         "k_out_pixel writing the game output)";
}

bool KRuntime::alloc(DevPtr* p, size_t bytes, bool zero) {
  bytes = std::max<size_t>(bytes, 256);
  if (!alloc_(p, bytes)) return fail("model K: device memory (" + std::to_string(bytes >> 20) + " MiB): " + dev_->err());
  if (zero && !dev_->memset(*p, 0, bytes)) return dfail();
  return true;
}

bool KRuntime::create(const WgRtConfig& c, Device* dev, AllocFn alloc_fn, bool external_io, bool native_direct) {
  cfg_ = c;
  dev_ = dev;
  alloc_ = std::move(alloc_fn);
  external_io_ = external_io;
  native_direct_ = native_direct;
  if (native_direct_ && (!external_io_ || c.backend != 3))
    return fail("model K: native direct resources need external I/O on the native D3D12 device (backend 3)");
  build_dir_ = c.build_dir ? c.build_dir : ".";
  W_ = c.ow;
  H_ = c.oh;
  max_rw_ = c.max_rw;
  max_rh_ = c.max_rh;
  hdr_ = c.hdr != 0;
  depth_inverted_ = c.depth_inverted != 0;
  if (W_ <= 0 || H_ <= 0 || max_rw_ <= 0 || max_rh_ <= 0) return fail("model K: invalid render or output size");
  // 32-bit byte offsets in k_io.hip (kio_ptr): every image below 4 GB
  if (16LL * W_ * H_ >= (1LL << 32) || 16LL * max_rw_ * max_rh_ >= (1LL << 32))
    return fail("model K: render or output size too large");
  W16_ = ceil16(W_);
  H16_ = ceil16(H_);
  k_base_size(W_, H_, kS, &Bw_, &Bh_);

  // model data (spec 48 / spec 10 section 7): load, check against the map, pack once
  if (!c.model_dir && option(c, "model_k_dir").empty()) return fail("model K: no model directory");
  std::string kdir = option(c, "model_k_dir");
  if (kdir.empty()) kdir = join(c.model_dir, "K");
  model_path_ = join(kdir, "model.wgcm");
  const std::string net = option(c, "network");
  if (!net.empty() && net != "fused" && net != "fused16" && net != "fused8")
    return fail("model K: unknown network variant " + net + " (fused, fused16 or fused8)");
  // fused8: fused16 with the FP8 forms of e0, e1, d1, d0 (k_block8.h; opt-in speed approximation, not bit-exact)
  f16_ = net == "fused16" || net == "fused8";
  f8_ = net == "fused8";
  // fp8_stages: which of e0, e1, d1, d0 run their FP8 form with network=fused8 (comma list; default all four)
  std::string f8list = option(c, "fp8_stages");
  if (f8list.empty()) f8list = "e0,e1,d1,d0";
  for (size_t a = 0; a <= f8list.size();) {
    size_t b = f8list.find(',', a);
    if (b == std::string::npos) b = f8list.size();
    const std::string n = f8list.substr(a, b - a);
    a = b + 1;
    if (n.empty() || n == "none") continue;
    if (!fp8_stage_name(n.c_str())) return fail("model K: fp8_stages: " + n + " is not one of e0, e1, d1, d0");
    f8stages_.push_back(n);
  }
  const std::string tiled = option(c, "input_tiled");
  if (!tiled.empty() && tiled != "0" && tiled != "1") return fail("input_tiled must be 0 or 1");
  tiled_input_ = tiled != "0";
  // k_in_uniforms (1, the default: the tiled input kernel takes the host's frame-uniform values, k_in_tiled_uniform,
  // where input_uniforms_safe allows; 0: diagnostic, always k_in_tiled, which computes the same values on the GPU: the
  // same output, used by the D3D12 port check of k_in_tiled)
  const std::string kuni = option(c, "k_in_uniforms");
  if (!kuni.empty() && kuni != "0" && kuni != "1") return fail("model K: k_in_uniforms must be 0 or 1");
  input_uniforms_ = kuni != "0";
  const std::string diag = option(c, "k_diag");
  if (!diag.empty() && diag != "input2" && diag != "copyin" && diag != "input2+copyin" && diag != "copyin+input2")
    return fail("model K: unknown k_diag " + diag + " (input2, copyin, input2+copyin)");
  diag_input2_ = diag.find("input2") != std::string::npos;
  diag_copyin_ = diag.find("copyin") != std::string::npos;
  const std::string khead = option(c, "k_head");
  if (!khead.empty() && khead != "compact" && khead != "full")
    return fail("model K: unknown k_head " + khead + " (compact or full)");
  compact_head_ = khead != "full";
  const std::string kfast = option(c, "k_fast");
  if (!kfast.empty() && kfast != "0" && kfast != "1") return fail("model K: k_fast must be 0 or 1");
  fast_ = kfast == "1";
  const std::string kexact = option(c, "k_in_exact");
  if (!kexact.empty() && kexact != "0" && kexact != "1") return fail("model K: k_in_exact must be 0 or 1");
  const std::string kfastin = option(c, "k_fast_in");
  if (!kfastin.empty() && kfastin != "0" && kfastin != "1") return fail("model K: k_fast_in must be 0 or 1");
  if (kexact == "1" && kfastin == "1") return fail("model K: k_in_exact=1 and k_fast_in=1 exclude each other");
  if (kexact == "1" && fast_) return fail("model K: k_in_exact=1 and k_fast=1 exclude each other");
  // the exact input stage is the default (spec 09 / 49 accuracy); k_fast_in=1 (or k_in_exact=0, or k_fast=1, whose
  // "_f" kernels include the fast input stage) selects the fast one
  in_exact_ = !(fast_ || kfastin == "1" || kexact == "0");
  // k_approx (opt-in, never the default): close approximations of the D3D12 ports, judged by image quality against
  // the exact path, which stays the default and the reference. "a1": the tiled input stage's log2 / 2^x / divisions
  // on the transcendental unit (kernels kx/k_in_tiled*__a1; clean/k/dx12/hlsl/k_in_common.hlsli KI_HWMATH): fast-close
  // where the stage supports it; a frame on the legacy input path (kx/k_in_color + k_in_base / k_in_reset: a render
  // size above 0.68 x the output, a jitter outside [-0.5, 0.5], input_tiled=0) runs the exact input stage, which is
  // never refused (exact is always within quality) and is logged once per context (take_frame_note).
  // "a2": the output stage's divisions on v_rcp_f32 with one quotient correction and two reciprocals without their
  // Newton step (kernels k_out_color*__a2, k_out_pixel_c*__a2; clean/k/dx12/hlsl/k_out.hlsli KO_APPROX). A comma list
  // selects several. The native D3D12 device only (HIP and the hybrid device refuse); taking one adds a note of create
  // (not bit-exact).
  if (option_given(c, "k_approx")) {
    const std::string kapprox = option(c, "k_approx");
    if (kapprox.empty()) return fail("model K: k_approx is empty (a1, a2 or a1,a2)");
    size_t at = 0;
    while (at <= kapprox.size()) {
      const size_t end = std::min(kapprox.find(',', at), kapprox.size());
      const std::string tag = kapprox.substr(at, end - at);
      if (tag == "a1") approx_a1_ = true;
      else if (tag == "a2") approx_a2_ = true;
      else if (tag.empty()) return fail("model K: k_approx has an empty item: " + one_line(kapprox) + " (a1, a2)");
      else return fail("model K: unknown k_approx " + one_line(tag) + " (a1, a2)");
      at = end + 1;
    }
    // the native D3D12 device only: it prepares the selected forms at create and refuses the fast input stage; the
    // hybrid device (backend 2) and HIP are refused
    if (c.backend != 3) return fail("model K: k_approx runs on the native D3D12 device only");
    if (approx_a1_ && !in_exact_) return fail("model K: k_approx=a1 needs the exact input kernels (k_io_exact.co)");
    if (approx_a2_ && fast_) return fail("model K: k_approx=a2 and k_fast=1 exclude each other");
    // a2 has forms of k_out_color and k_out_pixel_c only: the full head's k_out_pixel would stay exact
    if (approx_a2_ && !compact_head_) return fail("model K: k_approx=a2 needs the compact head (k_head=compact, the default)");
    approx_note_ = std::string("model K: close approximation k_approx=") + (approx_a1_ ? "a1" : "") +
                   (approx_a1_ && approx_a2_ ? "," : "") + (approx_a2_ ? "a2" : "") +
                   " (opt-in faster approximation; the default is the standard path)" +
                   (approx_a1_ ? std::string("; a1 applies to the tiled input stage, frames on the legacy input path "
                                             "run the standard input stage")
                               : std::string());
  }
  const std::string krule = option(c, "k_expo_rule");
  if (!krule.empty() && krule != "spec" && krule != "v051")
    return fail("model K: unknown k_expo_rule " + krule + " (spec or v051)");
  expo_v051_ = krule == "v051";
  // Latency of the small-grid stages on the D3D12 devices (no kernel arithmetic and no data flow changes):
  // k_color_at (D3D12 devices, backends 2 and 3; default kColorAtDefault): launch k_out_color unordered right after
  // the named stage (dev_->unordered_next(): no UAV barrier between that stage and k_out_color; the stage after it
  // waits for both), so it runs beside a latency-bound stage instead of alone before the input stage. Nothing the
  // network reads is written by k_out_color (ctap, ctapT: read by k_out_pixel_c only) and nothing it reads is written
  // by the network. "input": before the input stage (HIP's order; HIP and the host emulator always keep it).
  // k_warm (backend 3 only; default kWarmAtDefault; "off"): a k_warm dispatch launched unordered right after the
  // named stage ("input": after the input stage, before e0) reads every 128-byte line (k_warm_line) of the weight
  // blobs of the stages after it up to k_warm_to (default kWarmToDefault) into the caches, so the stages do not wait
  // for cold weights. k_warm_groups: its grid (default 64).
  const std::string kcol = option(c, "k_color_at");
  if (!kcol.empty() && kcol != "input" && !k_stage(kcol)) return fail("model K: k_color_at: unknown stage " + kcol);
  const std::string cat = kcol.empty() ? std::string(kColorAtDefault) : kcol;
  color_at_ = c.backend >= 2 && cat != "input" ? cat : std::string();
  const std::string kwarm = option(c, "k_warm");
  if (!kwarm.empty() && kwarm != "off" && kwarm != "input" &&
      (!k_stage(kwarm) || kwarm == "d2" || kwarm == "d1" || kwarm == "d0"))
    return fail("model K: k_warm: " + kwarm + " is not off, input or a stage before d2");
  warm_to_ = option(c, "k_warm_to");
  if (warm_to_.empty()) warm_to_ = kWarmToDefault;
  if (!k_stage(warm_to_)) return fail("model K: k_warm_to: unknown stage " + warm_to_);
  const std::string wat = kwarm.empty() ? std::string(kWarmAtDefault) : kwarm;
  warm_at_ = c.backend == 3 && wat != "off" ? wat : std::string();
  const std::string kwl = option(c, "k_warm_line"), kwg = option(c, "k_warm_groups");
  if (!kwl.empty()) warm_line_ = (unsigned)std::atoi(kwl.c_str());
  if (!kwg.empty()) warm_groups_ = (unsigned)std::atoi(kwg.c_str());
  if (warm_line_ < 4 || warm_line_ > 4096 || (warm_line_ & (warm_line_ - 1)))
    return fail("model K: k_warm_line must be a power of two in 4..4096");
  if (warm_groups_ < 1 || warm_groups_ > 1024) return fail("model K: k_warm_groups must be 1..1024");
  // SMX-SELFCHECK-SOURCE BEGIN (k_split: routes the self-check's m case; the digest covers it)
  // k_split (deep-split; the native D3D12 device, backend 3): "m" (the default) runs stage m as four phase launches
  // spread over many more workgroups (k_split_m_attn[__rodd], k_split_m_wo, k_split_m_w1, k_split_m_w2:
  // clean/k/dx12/kernels/k_block_split.hlsl), the same arithmetic and output bytes as the one k_stagew4_m launch; the
  // create-time self-check runs that form for its m case (launch() redirects k_stagew4_m). "off": the one launch. Other
  // devices always take the one launch (the phase kernels are D3D12 ports only).
  const std::string ksplit = option(c, "k_split");
  if (!ksplit.empty() && ksplit != "m" && ksplit != "off") return fail("model K: k_split must be m or off");
  split_m_ = c.backend == 3 && ksplit != "off";
  // SMX-SELFCHECK-SOURCE END
  // k_fill=<0..255> (test use, changes no value a frame writes): every buffer a frame writes and both history copies are
  // filled with this byte after allocation, before the first frame, so that a test with a different byte per backend
  // sees a write a frame did not make (first-frame and reset-only writes included) as a difference
  const std::string kfill = option(c, "k_fill");
  if (!kfill.empty()) {
    char* end = nullptr;
    const long v = std::strtol(kfill.c_str(), &end, 10);
    if (end == kfill.c_str() || *end || v < 0 || v > 255) return fail("model K: k_fill must be 0..255");
    fill_ = (int)v;
  }
  // k_smx_test: the softmax self-check's test hook (smx_select), developer builds only
  smx_test_ = option(c, "k_smx_test");
#ifdef WG_DEV_DIAGNOSTICS
  if (!smx_test_.empty() && smx_test_ != "mix_fail" && smx_test_ != "both_fail" && smx_test_ != "rodd_unchecked" &&
      smx_test_ != "split_fail" && smx_test_ != "report")
    return fail("model K: k_smx_test must be mix_fail, both_fail, rodd_unchecked, split_fail or report");
#else
  if (!smx_test_.empty()) return fail("model K: k_smx_test is a test hook of developer builds only");
#endif
  const std::string s_test = option(c, "k_s_breakpoints_test");
  bool force_s_mismatch = false;
#ifdef WG_DEV_DIAGNOSTICS
  if (!s_test.empty() && s_test != "mismatch") return fail("model K: k_s_breakpoints_test must be mismatch");
  force_s_mismatch = s_test == "mismatch";
#else
  if (!s_test.empty()) return fail("model K: k_s_breakpoints_test is a test hook of developer builds only");
#endif
  const ki_s_breakpoints::Table* s_table = nullptr;
  if ((c.backend == 2 || c.backend == 3) && tiled_input_ && in_exact_ && !approx_a1_) {
    static const auto table = ki_s_breakpoints::make_table();
    static const auto digest = ki_s_breakpoints::table_digest(table);
    s_breakpoints_fallback_ = force_s_mismatch || digest != ki_s_breakpoints::kProvenDigest;
    if (s_breakpoints_fallback_)
      notes_ += (notes_.empty() ? "" : "\n") +
                std::string("model K: S breakpoint digest mismatch; using the pre-breakpoint exact tiled S path");
    else
      s_table = &table;
  }
  // The native D3D12 device (backend 3: the D3D12 executor, or k_bench.py's own-buffer timing) runs the exact default
  // path only, and only when every kernel this context can launch has a port in <build>/dx12/manifest.json; both are
  // checked here, before the model is loaded, so a refused context costs nothing more (for model K the context ends and
  // the game keeps its own upscaler: there is no HIP route in a game).
  if (c.backend == 3) {
    if (!options_ok_direct()) return false;
    const std::string missing = d3d12_missing();
    if (!missing.empty()) return fail("model K: the D3D12 build has no port of " + missing);
  }
  map_path_ = option(c, "model_k_map");
  if (map_path_.empty()) {
    map_path_ = join(build_dir_, "preset-K.v1.map.json");  // next to the library, else built into it
    if (!wg_file_exists(map_path_) && wg_file_exists(join(kdir, "preset-K.v1.map.json")))
      map_path_ = join(kdir, "preset-K.v1.map.json");
  }
  std::vector<char> mchars;
  if (!wg_read_file(map_path_, &mchars)) return fail("model K: the converter map " + map_path_ + " is missing");
  const std::vector<uint8_t> mbytes(mchars.begin(), mchars.end());
  KModel model;
  std::string e;
  if (!wg_file_exists(model_path_))
    return fail("model K: no converted model at " + model_path_ + " (convert the preset-K model there)");
  if (!load_k_model_map(model_path_, mbytes, map_path_, &model, &e)) return fail("model K: " + e);
  std::memcpy(offsets_table_, model.window_offsets, sizeof offsets_table_);
  for (const KStageInfo& s : k_stage_table()) {
    std::vector<uint8_t> b = k_pack_stage(s.name, model.stages[s.name], &e);
    if (b.empty()) return fail("model K: " + e);
    if (b.size() != k_stage_blob_bytes(s)) return fail(std::string("model K: blob size of stage ") + s.name);
    if (f8_ && std::find(f8stages_.begin(), f8stages_.end(), s.name) != f8stages_.end()) {
      F8Stage f8 = fp8_stage(s, b);
      if (f8.why.empty()) f8data_[s.name] = std::move(f8);
      else f8why_ += std::string(" ") + s.name + ":" + f8.why;
    }
    blobs_[s.name] = std::move(b);
  }

  // code objects: the stage kernels (k_block.co: k_stage_*, k_block_wide.co: k_stagew*), the I/O stages and the
  // exposure scalars (k_io.co), model M's exposure (spec 07) and frame helpers (timing_barrier)
  for (const char* co : {"k_block.co", "k_block_wide.co", "k_io.co", "exposure.co", "frame.co"})
    if (!dev_->load_module(join(build_dir_, co))) return fail("model K: " + dev_->err());
  // k_in_exact=1: the same input kernels with exact arithmetic, addressed as "kx/<name>"
  if (in_exact_ && !dev_->load_module_tagged(join(build_dir_, "k_io_exact.co"), "kx"))
    return fail("model K: " + dev_->err());
  // The second tag selects the pre-breakpoint shader ports and resolves to the same exact kernels on HIP.
  if (s_breakpoints_fallback_ && !dev_->load_module_tagged(join(build_dir_, "k_io_exact.co"), "kxs"))
    return fail("model K: the pre-breakpoint exact tiled S path is unavailable: " + dev_->err());
  // buffers
  for (auto& kv : blobs_) {
    DevPtr p;
    if (!alloc(&p, kv.second.size()) || !dev_->upload(p, kv.second.data(), kv.second.size())) return false;
    wdev_[kv.first] = p;
  }
  if (c.backend == 2 || c.backend == 3) {
    for (const char* name : {"e0", "e1", "d1", "d0"}) {
      const KStageInfo& st = *k_stage(name);
      const auto& weight = blobs_.at(name);
      std::vector<uint8_t> compact;
      if (st.head) {
        if (!compact_head_) continue;
        compact = k_pack_compact_head(weight.data() + stage_sections(st).head, st.C);
        if (!alloc(&d0_compact_head_, compact.size()) ||
            !dev_->upload(d0_compact_head_, compact.data(), compact.size())) return false;
      }
      const auto bias = k_narrow_bias32(st, weight.data(), compact.data());
      const size_t bytes = bias.size() * sizeof(bias[0]);
      DevPtr& plane = narrow_bias_[name];
      if (!alloc(&plane, bytes) || !dev_->upload(plane, bias.data(), bytes)) return false;
    }
  }
  for (auto& kv : f8data_) {  // network=fused8: the FP8 weights and FP32 biases; device pointers into KF8Params
    DevPtr p;
    if (!alloc(&p, kv.second.bytes.size()) || !dev_->upload(p, kv.second.bytes.data(), kv.second.bytes.size()))
      return false;
    KF8Params& q = kv.second.q;
    for (DevPtr* a : {&q.wqkv, &q.mix, &q.wo, &q.w1, &q.w2, &q.bo, &q.b1, &q.b2}) *a += p;
    f8dev_[kv.first] = p;
  }
  for (const KBuffer& b : k_buffer_plan(Bw_, Bh_)) {
    DevPtr p;
    if (!alloc(&p, b.bytes)) return false;
    net_[b.name] = p;
    net_bytes_[b.name] = b.bytes;
  }
  const size_t R = (size_t)max_rw_ * max_rh_, P16 = (size_t)W16_ * H16_;
  for (int p = 0; p < 2; ++p)  // histories, zero at create (KFrame)
    if (!alloc(&Hc_[p], P16 * 4 * 2, true) || !alloc(&L_[p], 4 * P16 * 2, true) ||
        !alloc(&Hh_[p], (size_t)Bw_ * Bh_ * 4 * 2, true))
      return false;
  const int ny = std::max(256, max_rh_ / 2);
  // native_direct: no dense inputs (the kernels read the game's textures). The legacy input stage's maps (cpre, cpre2)
  // are needed by every context: its direct forms (kx/k_in_color_direct, then kx/k_in_reset or kx/k_in_base_direct) run
  // on a frame outside the tiled guard (a render size above 0.68 x the output, a jitter outside [-0.5, 0.5],
  // input_tiled=0) and write and read them
  if (!(native_direct_ || (alloc(&color_, R * 16) && alloc(&depth_, R * 4) && alloc(&motion_, R * 8))) ||
      !alloc(&cpre_, R * 16) || !alloc(&cpre2_, R * 8) || !alloc(&ctap_, R * 8) || !alloc(&ctapT_, R * 2) ||
      // exp2tab: read only by the host emulator's k_out_pixel (the GPU uses v_exp_f16); a valid, zeroed buffer
      !alloc(&exp2tab_, 65536 * 2, true) || !alloc(&tables_, 8 * kTableBytes) ||
      !alloc(&expo_, 32, true) || !alloc(&expo_partials_, 4 * (size_t)cdiv(ny, 4)) || !alloc(&expo_computed_, 4, true) ||
      !alloc(&expo_sync_, 4, true) || !alloc(&xs_, 16, true) || !alloc(&kexpo_, 16, true) ||
      (!warm_at_.empty() && !alloc(&warm_dummy_, 16, true)))
    return false;
  // every buffer an input-stage kernel of this context can be handed is allocated: a null pointer would be packed as
  // arena offset 0 (clean/runtime/struct_args.h) and alias the arena's first allocation
  if (!cpre_ || !cpre2_ || !ctap_ || !ctapT_ || (!native_direct_ && (!color_ || !depth_ || !motion_)))
    return fail("model K: an input-stage buffer is not allocated (it would be packed as arena offset 0)");
  // the own output buffer (tests): dense RGB f32, dense RGBA f32 (FrameIo.out_rgba32f) or RGBA16F rows that fit
  if (!external_io_ && !alloc(&out_, (size_t)W_ * H_ * 16, true)) return false;
  if (split_m_ && !split_alloc()) return false;  // k_split's scratch (filled with the rest by k_fill, below)
  if (fill_ >= 0) {  // k_fill (test hook): the sentinel in everything a frame writes, before the first frame
    auto fillb = [&](DevPtr p, size_t bytes) { return !p || dev_->memset(p, fill_, std::max<size_t>(bytes, 256)); };
    bool ok = true;
    for (const auto& kv : net_) ok = ok && fillb(kv.second, net_bytes_[kv.first]);
    for (int p = 0; p < 2; ++p)
      ok = ok && fillb(Hc_[p], P16 * 4 * 2) && fillb(L_[p], 4 * P16 * 2) && fillb(Hh_[p], (size_t)Bw_ * Bh_ * 4 * 2);
    ok = ok && fillb(ctap_, R * 8) && fillb(ctapT_, R * 2) && fillb(cpre_, R * 16) && fillb(cpre2_, R * 8) &&
         fillb(xs_, 16) && fillb(kexpo_, 16) &&
         fillb(color_, R * 16) && fillb(depth_, R * 4) && fillb(motion_, R * 8);  // the own input copies (unused with caller buffers)
    if (split_m_)  // k_split's scratch: written by the phases of stage m, so filled like the rest
      ok = ok && fillb(split_z_, split_bytes_[0]) && fillb(split_x1_, split_bytes_[1]) &&
           fillb(split_a_, split_bytes_[2]) && fillb(split_h_, split_bytes_[3]);
    if (!ok) return dfail();
  }
  // the filter-shape table (spec 43 section 7) of every level, computed once; a frame reads its level's set
  std::vector<uint16_t> tab((size_t)8 * kFilterEntries * kFilterValues);
  for (int l = 0; l < 8; ++l) k_filter_table(l, 1, tab.data() + (size_t)l * kFilterEntries * kFilterValues);
  if (!dev_->upload(tables_, tab.data(), tab.size() * 2)) return dfail();
  if (s_table) {
    if (!alloc(&s_breakpoints_, sizeof *s_table) ||
        !dev_->upload(s_breakpoints_, s_table->data(), sizeof *s_table))
      return dfail();
  }
  if (!dev_->sync()) return dfail();
  // the native device prepares (loads, verifies and compiles) every kernel this context can select now: a missing or
  // corrupt shader fails create here, not at the first frame
  if (c.backend == 3) {
    for (const std::string& k : d3d12_kernels())
      if (!dev_->prepare(k)) return dfail();
    prepared_ = true;
  }
  // the create-time self-check. A split stage m (k_split=m) that fails it is not a refusal: stage m takes the one launch
  // (k_split=off's path: the same bytes, the same reference hashes), the self-check runs again on that path, and create
  // refuses only when that fails too (both reasons in the error)
  const std::string notes_before = notes_;
  if (!smx_select(c.backend, smx_test_)) {
    if (!split_m_) return false;
    std::string split_why = err_;
    for (char& ch : split_why)
      if (ch == '\n' || ch == '\r') ch = ' ';
    notes_ = notes_before;
    split_m_ = false;
    if (!smx_select(c.backend, smx_test_)) {
      std::string one_launch = err_;
      for (char& ch : one_launch)
        if (ch == '\n' || ch == '\r') ch = ' ';
      return fail(one_launch + " (the split form of stage m failed first: " + split_why + ")");
    }
    notes_ += (notes_.empty() ? "" : "\n") +
              std::string("model K: the create-time self-check failed on the split form of stage m (") + split_why +
              "); stage m takes the one launch (k_stagew4_m), output unchanged";
    err_.clear();
  }
  if (!approx_note_.empty()) notes_ += (notes_.empty() ? "" : "\n") + approx_note_;
  return true;
}

// ---- the softmax weight rounding: create-time self-check (native D3D12 device) ---------------------------------------
// The softmax stages' weight rounding A = f16(w inv) (clean/k/dx12/hlsl/k_smx.hlsli) has two forms: the mix form (the
// exact sum p + e into the driver's round-to-nearest conversion, exact when the driver folds it into one
// v_fma_mix*_f16) and the round-to-odd ports (suffix __rodd, exact when the driver keeps the round-to-odd integer
// operations and converts round-to-nearest). Both depend on how the player's driver compiles each stage shader, so
// neither is taken on trust. At create the native D3D12 device checks the shaders that will run:
// - k_mix_probe and k_mix_probe__rodd on k_mix_probe.h's records (FP16-midpoint products with residuals of both signs,
//   exact ties, FP16 subnormals) against the exact results;
// - every softmax stage kernel this runtime can select, in both forms, at the window offsets this model's frames use
//   (k_stage_offsets over the eight entries of its window table: e2, e3, d3, d2, d1 at (4, 4); e4 and d4 at
//   (2 t - 4) mod 8; m at the table's entries t), on the runtime's own synthetic inputs and weights (kSmxCases); each
//   case's outputs are hashed and compared with k_smx_selfcheck.h, the HIP reference's hashes of every offset any window
//   table can give (clean/k/dx12/gen_k_smx_selfcheck.py, bound to the shader sources by smx_selfcheck_digest.py).
// The mix form is used when its probe and all its cases match; else the round-to-odd ports when theirs all match (one
// note of create); else create fails, naming the first case that differs. Every kernel the context can select is
// prepared at create (KRuntime::create, Device::prepare): a missing or corrupt shader fails create.
// Test hook (developer builds only, WG_DEV_DIAGNOSTICS): runtime option k_smx_test = mix_fail (the mix form counts as
// failed), both_fail (both forms count as failed), rodd_unchecked (the round-to-odd ports without the check; for the
// frame test's negative control), split_fail (the split form of stage m counts as failed, so create takes the one
// launch; with k_split=off it has no effect), report (adds the computed hashes as a note; on any device, every possible
// offset).
namespace {
// SMX-SELFCHECK-SOURCE BEGIN (the cases, the synthetic data, the offsets and the check: k_smx_selfcheck.h's digest covers this text)
struct SmxCase {
  const char* stage;
  int wb, hb;  // base map size
  int sets;    // independent synthetic data sets, the outputs of all hashed together
};
// e2, e3, d3, d2, d1 (k_stage_e2, k_stagew2_e3, k_stagew2_d3, k_stage_d2, k_stage2_d1) and e4 / d4 with NG = 4 (at most 24
// windows: 20 at 512 x 384; on 8 data sets) at a 512 x 384 base map; e4 / d4 with NG = 2 (above 24 windows) and m
// (k_stagew4_m, 42 or more windows) at 1536 x 1280. Sized so that a one-stage wrong-rounding mutant of either form
// changes its case's outputs (the mutant runs in the #254 evidence).
const SmxCase kSmxCases[] = {{"e2", 512, 384, 1},   {"e3", 512, 384, 1},   {"d3", 512, 384, 1}, {"d2", 512, 384, 1},
                             {"d1", 512, 384, 1},   {"e4", 512, 384, 8},   {"d4", 512, 384, 8}, {"e4", 1536, 1280, 1},
                             {"d4", 1536, 1280, 1}, {"m", 1536, 1280, 1}};
constexpr uint32_t kSmxSeed = 0x6b5eu;  // synthetic data (xorshift32)
// FP16 words of uniform values in [-scale, scale): also finite FP32 values where two words are read as one
void smx_fill(std::vector<uint16_t>* v, uint32_t* state, float scale) {
  for (uint16_t& h : *v) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    h = f32_to_f16(((float)(x >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale);
  }
}
constexpr int kSmxCount = (int)(sizeof kSmxCases / sizeof kSmxCases[0]);

const KStageInfo& stage_info(const char* name) {
  for (const KStageInfo& s : k_stage_table())
    if (std::strcmp(s.name, name) == 0) return s;
  return k_stage_table()[0];
}
int stage_index(const char* name) {
  const auto& t = k_stage_table();
  for (size_t i = 0; i < t.size(); ++i)
    if (std::strcmp(t[i].name, name) == 0) return (int)i;
  return 0;
}
size_t level_tokens(int wb, int hb, int level) {
  int w, h;
  k_level_size(wb, hb, level, &w, &h);
  return (size_t)w * h;
}
// The window offsets of stage `stage` in frames (k_stage_offsets, the network's rule): over the eight entries of
// `table`, or (table == nullptr) over every table entry 0 .. 7 x 0 .. 7 any model can hold. Distinct, sorted.
std::vector<std::pair<int, int>> smx_offsets(const char* stage, const int (*table)[2]) {
  std::vector<std::pair<int, int>> v;
  const int si = stage_index(stage);
  for (int e = 0; e < (table ? 8 : 64); ++e) {
    int t[8][2], off[11][2];
    for (int k = 0; k < 8; ++k) {
      t[k][0] = table ? table[k][0] : e % 8;
      t[k][1] = table ? table[k][1] : e / 8;
    }
    k_stage_offsets(table ? e : 0, t, off);
    v.emplace_back(off[si][0], off[si][1]);
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}
}  // namespace

std::string KRuntime::smx_case_name(int i, int ox, int oy, bool rodd) const {
  const SmxCase& c = kSmxCases[i];
  const KStageInfo& st = stage_info(c.stage);
  int lw, lh, ng = 0;
  bool w2 = false;
  k_level_size(c.wb, c.hb, st.level, &lw, &lh);
  const std::string k = stage_kernel_base(st, false, true, cdiv(lw + ox, 8) * cdiv(lh + oy, 8), &ng, &w2);
  return k + (rodd ? "__rodd" : "") + "@" + std::to_string(c.wb) + "x" + std::to_string(c.hb) +
         (c.sets > 1 ? "*" + std::to_string(c.sets) : std::string()) + "/o" + std::to_string(ox) + "," + std::to_string(oy);
}

// One form of every case at its offsets (this model's: all = false; every possible one: all = true): (case name of the
// mix form, the outputs' SHA-256, first 8 bytes big-endian) into *out.
bool KRuntime::smx_check(bool rodd, bool all, std::vector<std::pair<std::string, uint64_t>>* out) {
  size_t in_bytes = 0, y_bytes = 0, m_bytes = 0, w_bytes = 0;
  for (const SmxCase& c : kSmxCases) {
    const KStageInfo& st = stage_info(c.stage);
    const size_t own = level_tokens(c.wb, c.hb, st.level) * st.C * 2;
    in_bytes = std::max({in_bytes, own, st.kind == 'd' ? level_tokens(c.wb, c.hb, st.level + 1) * st.C_prev * 2 : 0});
    y_bytes = std::max(y_bytes, own);
    if (st.kind == 'e') m_bytes = std::max(m_bytes, level_tokens(c.wb, c.hb, st.level + 1) * st.C_next * 2);
    w_bytes = std::max(w_bytes, blobs_.at(c.stage).size());
  }
  // inputs (the input map and a decoder's skip map read the same synthetic data) and one weight blob for every stage
  std::vector<uint16_t> in(in_bytes / 2), w(w_bytes / 2);
  DevPtr din = 0, dw = 0, dy = 0, dm = 0, db = 0;
  int loaded = -1;  // the data set in din / dw
  auto load = [&](int set) {
    if (set == loaded) return true;
    uint32_t state = kSmxSeed + 0x9e3779b9u * (uint32_t)set;
    if (!state) state = 1;
    smx_fill(&in, &state, 1.0f);
    smx_fill(&w, &state, 0.25f);
    loaded = set;
    return dev_->upload(din, in.data(), in_bytes) && dev_->upload(dw, w.data(), w_bytes);
  };
  bool ok = dev_->alloc(&din, in_bytes) && dev_->alloc(&dw, w_bytes) && dev_->alloc(&dy, y_bytes) &&
            dev_->alloc(&dm, std::max<size_t>(m_bytes, 256));
  if (ok && narrow_bias_.count("d1")) {
    const auto& d1 = *k_stage("d1");
    const size_t words = 10u * d1.C + (size_t)d1.heads * 64 * 64;
    ok = dev_->alloc(&db, words * sizeof(uint32_t));
  }
  std::vector<uint8_t> buf;
  for (int i = 0; ok && i < kSmxCount; ++i) {
    const SmxCase& c = kSmxCases[i];
    const KStageInfo& st = stage_info(c.stage);
    int lw, lh, iw, ih;
    k_level_size(c.wb, c.hb, st.level, &lw, &lh);
    k_level_size(c.wb, c.hb, st.kind == 'd' ? st.level + 1 : st.level, &iw, &ih);
    const size_t yb = level_tokens(c.wb, c.hb, st.level) * st.C * 2;
    const size_t mb = st.kind == 'e' ? level_tokens(c.wb, c.hb, st.level + 1) * st.C_next * 2 : 0;
    buf.resize(yb + mb);
    for (const auto& o : smx_offsets(c.stage, all ? nullptr : offsets_table_)) {
      KBlkParams p = block_params(st, dw, din, dy, din, dm, 0, lw, lh, iw, ih, o.first, o.second);
      int ng = 0;
      bool w2 = false;
      const std::string k = stage_kernel_base(st, false, true, p.nwx * p.nwy, &ng, &w2) + (rodd ? "__rodd" : "");
      const bool use_bias = db && (k == "k_stage2_d1" || k == "k_stage2_d1__rodd");
      wgc::Sha256 sha;
      for (int set = 0; ok && set < c.sets; ++set) {
        ok = load(set);
        if (ok && use_bias) {
          const auto bias = k_narrow_bias32(st, reinterpret_cast<const uint8_t*>(w.data()));
          ok = dev_->upload(db, bias.data(), bias.size() * sizeof(bias[0]));
        }
        ok = ok && dev_->memset(dy, 0, y_bytes) && dev_->memset(dm, 0, std::max<size_t>(m_bytes, 256)) &&
             launch(k.c_str(), (unsigned)(w2 ? cdiv(p.nwx * p.nwy, 2) : p.nwx * p.nwy),
                    (unsigned)(kThreads * (ng ? ng : 1)), &p, use_bias ? &db : nullptr) &&
             dev_->download(buf.data(), dy, yb) && (!mb || dev_->download(buf.data() + yb, dm, mb));
        if (ok) sha.update(buf.data(), buf.size());
      }
      if (!ok) break;
      const wgc::Digest d = sha.finish();
      uint64_t v = 0;
      for (int b = 0; b < 8; ++b) v = (v << 8) | d[b];
      out->emplace_back(smx_case_name(i, o.first, o.second, false), v);
    }
  }
  for (DevPtr p : {din, dw, dy, dm, db})
    if (p) dev_->free(p);
  return ok || dfail();
}

// SMX-SELFCHECK-SOURCE END

// The mix / round-to-odd probe: the number of results that differ from the exact ones.
bool KRuntime::smx_probe(bool rodd, int* bad) {
  DevPtr in = 0, out = 0;
  const size_t out_bytes = (size_t)k_mix_probe::kLanes * 16;
  if (!dev_->alloc(&in, sizeof k_mix_probe::kRecords)) return dfail();
  if (!dev_->alloc(&out, out_bytes)) {
    dev_->free(in);
    return dfail();
  }
  std::vector<uint16_t> got((size_t)k_mix_probe::kLanes * 8);
  const bool ok = dev_->upload(in, k_mix_probe::kRecords, sizeof k_mix_probe::kRecords) &&
                  dev_->memset(out, 0xA5, out_bytes) &&
                  launch(rodd ? "k_mix_probe__rodd" : "k_mix_probe", (unsigned)(k_mix_probe::kLanes / 32), 32, &in, &out) &&
                  dev_->download(got.data(), out, out_bytes);
  dev_->free(in);
  dev_->free(out);
  if (!ok) return dfail();
  *bad = 0;
  for (size_t i = 0; i < got.size(); ++i) *bad += got[i] != k_mix_probe::kExpected[i];
  return true;
}

// smx_select (create, last step): the softmax form of this runtime, from the checks above (native D3D12 device), or only
// the report hook's hashes (other devices: the HIP reference for clean/k/dx12/gen_k_smx_selfcheck.py).
bool KRuntime::smx_select(int backend, const std::string& hook) {
  auto hex = [](uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
    return std::string(b);
  };
  auto report = [&](const char* form, const std::vector<std::pair<std::string, uint64_t>>& h, int probe_bad) {
    std::string r = std::string("model K self-check ") + form + ":" +
                    (probe_bad >= 0 ? " probe=" + std::to_string(probe_bad) : std::string());
    for (const auto& e : h) r += " " + e.first + "=" + hex(e.second);
    return r;
  };
  auto add_note = [&](const std::string& line) { notes_ += (notes_.empty() ? "" : "\n") + line; };
  if (backend != 3) {  // the report hook only: the reference's hashes at every possible offset
    if (hook != "report") return true;
    std::vector<std::pair<std::string, uint64_t>> h;
    if (!smx_check(false, true, &h)) return false;
    add_note(report("reference", h, -1));
    return true;
  }
  if (hook == "split_fail" && split_m_)  // test hook: the split form of stage m counts as failed (create's fallback)
    return fail("model K: test hook: the split self-check of stage m failed (k_smx_test=split_fail)");
  if (hook == "rodd_unchecked") {
    smx_rodd_ = true;
    smx_note_ = " smx=rodd(unchecked)";
    add_note("model K: test hook: the round-to-odd softmax ports without the create-time check");
    return true;
  }
  int bad[2] = {0, 0};
  std::vector<std::pair<std::string, uint64_t>> h[2];
  const bool all = hook == "report";
  if (!smx_probe(false, &bad[0]) || !smx_probe(true, &bad[1]) || !smx_check(false, all, &h[0]) ||
      !smx_check(true, all, &h[1]))
    return false;
  // the first check that differs, per form ("" = none); a case without a reference refuses create
  std::string first[2];
  for (int f = 0; f < 2; ++f) {
    if (bad[f]) first[f] = std::string(f ? "k_mix_probe__rodd" : "k_mix_probe") + " " + std::to_string(bad[f]) + " of " +
                           std::to_string(k_mix_probe::kLanes * 8);
    for (const auto& e : h[f]) {
      int at = -1;
      for (int j = 0; j < k_smx_selfcheck::kCount && at < 0; ++j)
        if (e.first == k_smx_selfcheck::kNames[j]) at = j;
      if (at < 0) return fail("model K: k_smx_selfcheck.h has no reference for " + e.first + " (regenerate it)");
      if (first[f].empty() && e.second != k_smx_selfcheck::kExpected[at]) {
        const size_t a = e.first.find('@');
        first[f] = f ? e.first.substr(0, a) + "__rodd" + e.first.substr(a) : e.first;
      }
    }
  }
  if (hook == "mix_fail" || hook == "both_fail") first[0] = "test hook";
  if (hook == "both_fail") first[1] = "test hook";
  if (hook == "report") {
    add_note(report("mix", h[0], bad[0]));
    add_note(report("rodd", h[1], bad[1]));
  }
  if (first[0].empty()) {
    smx_rodd_ = false;
    smx_note_ = " smx=mix";
    if (!first[1].empty())
      add_note("model K: the round-to-odd softmax ports failed the create-time check on this driver (" + first[1] +
               "); the mix form passed it and is used");
    return true;
  }
  if (first[1].empty()) {
    smx_rodd_ = true;
    std::string token = first[0] == "test hook" ? std::string("hook") : first[0].substr(0, first[0].find(' '));
    smx_note_ = " smx=rodd(" + token + ")";
    add_note("model K: softmax weight rounding takes the round-to-odd ports (the mix form failed the create-time check: " +
             first[0] + "); they passed it, output unchanged");
    return true;
  }
  return fail("model K: the D3D12 softmax stages are not exact on this driver (create-time check: mix form " + first[0] +
              ", round-to-odd form " + first[1] + "); not started");
}

std::string KRuntime::stage_kernel(const std::string& k, const char* stage) const {
  // e0, e1 and d0 mix with fixed weights (no softmax): no __rodd form
  const bool softmax = std::strcmp(stage, "e0") != 0 && std::strcmp(stage, "e1") != 0 && std::strcmp(stage, "d0") != 0;
  return smx_rodd_ && softmax ? k + "__rodd" : k;
}

bool KRuntime::launch(const char* kernel, unsigned grid, unsigned block, void* a0, void* a1, void* a2, void* a3,
                      void* a4, void* a5, void* a6, void* a7) {
  if (grid == 0) return true;
  // SMX-SELFCHECK-SOURCE BEGIN (k_split: stage m's kernel-name redirect; the digest covers it)
  // k_split: stage m (the frame's and the self-check's) as its phase launches
  if (split_m_ && std::strncmp(kernel, "k_stagew4_m", 11) == 0 && (!kernel[11] || std::strcmp(kernel + 11, "__rodd") == 0))
    return split_m(kernel[11] != 0, a0);
  // SMX-SELFCHECK-SOURCE END
  void* args[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
  if (pending_mark_ >= 0) {
    const int e = pending_mark_;
    pending_mark_ = -1;
    return dev_->launch_timed(kernel, grid, block, args, e, -1) || dfail();
  }
  return dev_->launch(kernel, grid, block, args) || dfail();
}

// SMX-SELFCHECK-SOURCE BEGIN (k_split: the phase launches and scratch; the digest covers them)
// k_split: stage m as four launches (clean/k/dx12/kernels/k_block_split.hlsl), joined by the dispatch boundaries:
// attention per (window, head), Wo per (window, output tile), W1 per (window, two hidden tiles), W2 per (window, output
// tile). Each phase reads the block parameters of k_stagew4_m's launch and the scratch the phase before wrote.
bool KRuntime::split_m(bool rodd, void* params) {
  const KBlkParams& p = *static_cast<const KBlkParams*>(params);
  const KStageInfo& st = *k_stage("m");
  const int nwin = p.nwx * p.nwy, kt = st.C / 16, ht = 4 * st.C / 16;
  if (nwin > split_cap_)
    return fail("model K: k_split: stage m has " + std::to_string(nwin) + " windows, the scratch holds " +
                std::to_string(split_cap_));
  // k_split_m_w1 runs HT / 2 groups per window: one group per two hidden tiles (KS_HB = 2 in
  // clean/k/dx12/kernels/k_block_split.hlsl). A change of KS_HB must change this grid too.
  return launch(rodd ? "k_split_m_attn__rodd" : "k_split_m_attn", (unsigned)(nwin * st.heads), 128, params, &split_z_,
                &split_x1_, &split_a_, &split_h_) &&
         launch("k_split_m_wo", (unsigned)(nwin * kt), 128, params, &split_z_, &split_x1_, &split_a_, &split_h_) &&
         launch("k_split_m_w1", (unsigned)(nwin * (ht / 2)), 128, params, &split_z_, &split_x1_, &split_a_, &split_h_) &&
         launch("k_split_m_w2", (unsigned)(nwin * kt), 128, params, &split_z_, &split_x1_, &split_a_, &split_h_);
}

// The phases' scratch (FP16): Z [windows][64][8 heads x 32], X1 and A [windows][64][C], H [windows][64][4 C], for the most
// windows stage m has at any window offset (0 .. 7) in this context's frames or in the self-check's m case.
bool KRuntime::split_alloc() {
  const KStageInfo& st = *k_stage("m");
  if (st.C != 160 || st.heads != 8)  // the phase kernels are compiled for stage m's C = 160, 8 heads
    return fail("model K: k_split: stage m is not C = 160 with 8 heads");
  int cap = 0;
  auto grow = [&](int wb, int hb) {
    int lw, lh;
    k_level_size(wb, hb, st.level, &lw, &lh);
    for (int oy = 0; oy < 8; ++oy)
      for (int ox = 0; ox < 8; ++ox) cap = std::max(cap, k_window_count(lw, lh, ox, oy));
  };
  grow(Bw_, Bh_);
  for (const SmxCase& sc : kSmxCases)
    if (std::strcmp(sc.stage, "m") == 0) grow(sc.wb, sc.hb);
  split_cap_ = cap;
  const size_t tok = (size_t)cap * 64;
  split_bytes_[0] = tok * st.heads * 32 * 2;  // Z
  split_bytes_[1] = tok * st.C * 2;           // X1
  split_bytes_[2] = tok * st.C * 2;           // A
  split_bytes_[3] = tok * 4 * st.C * 2;       // H
  return alloc(&split_z_, split_bytes_[0]) && alloc(&split_x1_, split_bytes_[1]) &&
         alloc(&split_a_, split_bytes_[2]) && alloc(&split_h_, split_bytes_[3]);
}
// SMX-SELFCHECK-SOURCE END

bool KRuntime::upload_inputs(int rw, int rh, const float* color, const float* dep, const float* mot, int channels) {
  if (rw <= 0 || rh <= 0 || rw > max_rw_ || rh > max_rh_) return fail("render size larger than the maximum");
  if (external_io_) return fail("inputs come from the interop (external_io)");
  if (channels != 3 && channels != 4) return fail("model K: the colour has 3 or 4 values per pixel");
  if (!dev_->sync()) return dfail();
  const size_t n = (size_t)rw * rh;
  // the dense colour's fourth value is the colour input's alpha, unmodified (the output alpha follows it through the
  // output stage's current-colour bilinear; issue #26); an RGB colour has none: 1, the value a texture fetch
  // gives a missing alpha
  std::vector<float> c(n * 4, 1.0f);
  for (size_t i = 0; i < n; ++i)
    for (int k = 0; k < channels; ++k) c[4 * i + k] = color[(size_t)channels * i + k];
  return (dev_->upload(color_, c.data(), n * 16) && dev_->upload(depth_, dep, n * 4) &&
          dev_->upload(motion_, mot, n * 8) && dev_->sync()) ||
         dfail();
}

// The eleven stages for frame index f (k_net.py KNet.stage_plan, variant "fused" or "fused16"; k_launch.py
// stage_fused_launch).
bool KRuntime::network(int f, const std::function<bool(const char*)>& after) {
  int off[11][2];
  k_stage_offsets(f, offsets_table_, off);
  const auto& table = k_stage_table();
  for (size_t i = 0; i < table.size(); ++i) {
    const KStageInfo& st = table[i];
    const FusedIo& io = kFusedIo[i];
    int lw, lh, iw, ih;
    k_level_size(Bw_, Bh_, st.level, &lw, &lh);
    k_level_size(Bw_, Bh_, st.kind == 'd' ? st.level + 1 : st.level, &iw, &ih);
    const int ox = off[i][0], oy = off[i][1];
    if (st.kind != 'm' && (ox % 2 || oy % 2)) return fail(std::string("model K: odd window offset at ") + st.name);
    const DevPtr blob = wdev_[st.name];
    KBlkParams p = block_params(st, blob, net_[io.in], io.block ? net_[io.block] : 0, io.skip ? net_[io.skip] : 0,
                                io.merged ? net_[io.merged] : 0, io.head ? net_[io.head] : 0, lw, lh, iw, ih, ox, oy);
    int ng = 0;
    bool w2 = false;
    const std::string k = stage_kernel_base(st, f16_, p.y != 0, p.nwx * p.nwy, &ng, &w2);
    if (st.kind == 'm' && !ng) return fail("model K: stage m needs its wide kernel");
    const unsigned grid = (unsigned)(w2 ? cdiv(p.nwx * p.nwy, 2) : p.nwx * p.nwy);
    auto f8 = f8data_.find(st.name);
    if (f8_ && w2 && f8 != f8data_.end()) {  // network=fused8: k_stage2q_* (k_block8.h)
      KF8Params q = f8->second.q;
      if (st.head) {
        DevPtr hidden = Hh_[1 - parity_];
        if (!launch(compact_head_ ? "k_stage2q_d0_hp" : "k_stage2q_d0_hidden", grid, kThreads, &p, &q, &hidden))
          return false;
      } else if (!launch(("k_stage2q_" + std::string(st.name)).c_str(), grid, kThreads, &p, &q))
        return false;
    } else if (st.head) {  // d0 always uses the G2 stage above; its epilogue writes the hidden history (FP32 and FP16 forms)
      DevPtr hidden = Hh_[1 - parity_];
      // k_head=compact: p.head (the OUT buffer) receives the compact pixel head [2 Bh][2 Bw][4] instead of the head map
      const char* k0 = compact_head_ ? (f16_ ? "k_stage2h_d0_hp" : "k_stage2_d0_hp")
                                     : (f16_ ? "k_stage2h_d0_hidden" : "k_stage2_d0_hidden");
      if (compact_head_ && !f16_ && d0_compact_head_) {
        p.wz = d0_compact_head_;
        if (!launch(k0, grid, kThreads, &p, &hidden, &narrow_bias_.at(st.name))) return false;
      } else if (!launch(k0, grid, kThreads, &p, &hidden)) {
        return false;
      }
    } else {
      const std::string kernel = stage_kernel(k, st.name);
      const auto bias_plane = narrow_bias_.find(st.name);
      const bool use_bias = k == "k_stage2_e0" || k == "k_stage2_e1" || k == "k_stage2_d1";
      void* bias = use_bias && bias_plane != narrow_bias_.end() ? &bias_plane->second : nullptr;
      if (!launch(kernel.c_str(), grid, (unsigned)(kThreads * (ng ? ng : 1)), &p, bias)) return false;
    }
    if (after && !after(st.name)) return false;
    if (i < 10) {
      info_.offsets[i][0] = ox;  // WgRtFrameInfo has ten rows: e0 ... d1 (d0's offset is (4, 4))
      info_.offsets[i][1] = oy;
    }
  }
  return true;
}

// k_warm (backend 3): unordered after stage `at` (or the input stage), the weight blobs of the stages after it up to
// warm_to_ into the caches.
bool KRuntime::warm(const char* at) {
  KWarmParams w = {};
  w.dummy = warm_dummy_;
  w.line = warm_line_;
  w.key = 0x9e3779b9u;
  int n = 0;
  bool past = std::strcmp(at, "input") == 0;  // "input": from e0 on
  uint64_t lines = 0;
  for (const KStageInfo& st : k_stage_table()) {
    if (past && n < 12) {
      const size_t bytes = blobs_[st.name].size() & ~(size_t)3;
      w.r[n] = wdev_[st.name];
      w.n[n++] = (uint32_t)bytes;
      lines += (bytes + warm_line_ - 1) / warm_line_;
    }
    if (std::strcmp(st.name, at) == 0) past = true;
    if (warm_to_ == st.name) break;
  }
  if (!n) return true;
  const unsigned grid = (unsigned)std::min<uint64_t>(warm_groups_, (lines + 255) / 256);
  dev_->unordered_next();
  return launch("k_warm", grid, 256, &w);
}

bool KRuntime::frame(const WgRtFrame& fr, const FrameIo* io) {
  // reset = 2 (interop: a new context of the game took this upscaler over, wronggpu_upscaler.h): this frame is the
  // first frame of a newly created K runtime, as for model M. The host state a frame leaves for the next one starts
  // over (frame counter, so the exposure's first frame too; frame index f; history parity; last size; jitter). The
  // device histories keep their contents: a reset frame reads none of them (validity false, k_in_reset, E_prev =
  // E_cur), so the outputs equal a new runtime's (test_k_runtime_gpu.py test_reset2_as_new).
  if (fr.reset >= 2) {
    frames_ = fidx_ = parity_ = last_rw_ = last_rh_ = 0;
    jprev_[0] = jprev_[1] = 0.0f;
  }
  const int rw = fr.rw, rh = fr.rh;
  if (rw <= 0 || rh <= 0 || rw > max_rw_ || rh > max_rh_) return fail("render size larger than the maximum");
  if (native_direct_ && !direct_preflight(fr, io)) return false;
  // Model K resets on the game's flag and on the first frame only (spec 44: the reset restarts the frame index). A
  // render-size change keeps the history: the histories are at output and base resolution, and the input and output
  // stages reproject them with this frame's render size (model M's spec 10 rule does not apply to K).
  const bool reset = fr.reset || frames_ == 0;
  const int f = reset ? 0 : fidx_;
  const int p = parity_, q = 1 - parity_;
  const int tb = io ? io->timing_base : -1;
  int zero = 0;
  auto mark = [&](int k) {
    if (tb < 0) return true;
    if (k < 4) return pending_mark_ = tb + k, true;
    pending_mark_ = -1;
    void* args[1] = {&zero};
    return dev_->launch_timed("timing_barrier", 1, 32, args, -1, tb + 4) || dfail();
  };
  info_ = {};
  info_.index = frames_;
  info_.n = f;  // model K: the frame index f of spec 44 (0 on a reset frame)
  info_.parity = p;
  info_.reset = reset ? 1 : 0;
  info_.rw = rw;
  info_.rh = rh;
  info_.jitter_prev[0] = reset ? fr.jitter[0] : jprev_[0];
  info_.jitter_prev[1] = reset ? fr.jitter[1] : jprev_[1];
  if (!mark(0)) return false;

  // inputs: the caller's dense buffers (FrameIo), the interop's raw copies converted here, or the own test buffers;
  // native_direct: none (the "_direct" kernels read the bound textures; their pointer arguments stay 0)
  DevPtr cin = io && io->color ? io->color : color_, din = io && io->depth ? io->depth : depth_;
  DevPtr min_ = io && io->motion ? io->motion : motion_;
  if (native_direct_) cin = din = min_ = 0;
  // input_copy (io->raw_inputs): the interop's copies of the game's textures in pitched rows, read directly (KIO_C16:
  // RGBA16F colour, not divided by pi: K applies kappa; R32F depth; KIO_M16: FP16 motion of motion_channels channels
  // with m = ms r + mo applied on load, as input_convert would have written it)
  const bool raw = io && io->raw_inputs;
  int cpitch = rw, mpitch = rw, zpitch = rw, mch = 2;
  if (raw && cfg_.backend >= 2)  // the interop's HIP input copies: the D3D12 devices read dense buffers or the game's textures
    return fail("model K: input-copy frames (raw_inputs) run on the HIP backend only");
  if (raw) {
    mch = io->motion_channels;
    if ((mch != 2 && mch != 4) || io->color_pitch % 8 || io->depth_pitch % 4 || io->motion_pitch % (2 * mch))
      return fail("model K: unexpected input-copy layout");
    cpitch = io->color_pitch / 8;
    zpitch = io->depth_pitch / 4;
    mpitch = io->motion_pitch / (2 * mch);
  }
  if (!native_direct_ && (!cin || !din || !min_)) return fail("frame without input buffers (interop)");
  if (diag_copyin_ && !raw && cin != color_) {  // k_diag=copyin: the inputs from K's own allocations (dense, as FrameIo)
    const size_t n = (size_t)rw * rh;
    if (!dev_->copy(color_, cin, n * 16) || !dev_->copy(depth_, din, n * 4) || !dev_->copy(motion_, min_, n * 8))
      return dfail();
    cin = color_;
    din = depth_;
    min_ = motion_;
  }

  // exposure: spec 42 draft 3 section 4 (k_io.hip k_expo_scalars) on the GPU. Pre-exposure pi: 1 if missing, <= 0
  // or not finite (spec 10); model K does not divide the colour by pi, kappa = (1 / pi) e1 E_cur does.
  // The front end's exposure source (FrameIo.exposure_mode): 0 a value (1 when the game gives nothing), 1 the game's
  // exposure texel, 2 "compute it" (no texture, auto-exposure requested). E_cur is the game's value or texel, 1 with
  // mode 2. Auto-exposure (spec 07, HDR only) runs when the game asked for it at creation (WgRtConfig.auto_exposure)
  // or the front end asks this frame (mode 2), also next to a game texture; its value is E_aux only.
  const float pi = (std::isfinite(fr.pre_exposure) && fr.pre_exposure > 0.0f) ? fr.pre_exposure : 1.0f;
  const int mode = io && io->exposure_mode >= 0 ? io->exposure_mode : (cfg_.auto_exposure ? 2 : 0);
  const bool auto_run = hdr_ && (mode == 2 || cfg_.auto_exposure);
  int first = frames_ == 0 ? 1 : 0;
  if (auto_run) {
    ExpParams ep = {};
    ep.rw = rw;
    ep.rh = rh;
    ep.nx = std::max(256, rw / 2);
    ep.ny = std::max(256, rh / 2);
    ep.nparts = cdiv(ep.ny, 4);
    ep.inv_pi = 1.0f / pi;
    const double a = cfg_.alpha > 0 ? cfg_.alpha : 0.18;
    ep.K = (float)(a / (1.0 - a));
    ep.inv_n = (float)(1.0 / ((double)ep.nx * ep.ny));
    if (raw ? !launch("exposure_auto_diet_c16", (unsigned)ep.nparts, 256, &cin, &cpitch, &expo_partials_, &expo_sync_,
                      &expo_computed_, &expo_, &ep, &first)
            // native_direct: model M's direct form (dx12/kernels/exposure.hlsl: the same sums on the colour texels,
            // nd_raw_color_xy, which are the input pass's dense f32 colour; inv_pi as here)
            : !launch(native_direct_ ? "exposure_auto_diet_direct" : "exposure_auto_diet", (unsigned)ep.nparts, 256,
                      &cin, &expo_partials_, &expo_sync_, &expo_computed_, &expo_, &ep, &first))
      return false;
  }
  KExpoArgs xa = {};
  xa.value = fr.exposure;
  xa.src_mode = mode == 1 ? 1 : mode == 2 ? 2 : 0;
  xa.aux = auto_run ? 1 : 0;
  xa.first = first;
  xa.hdr = hdr_;
  xa.reset = reset;
  // e1 = e2 = the game's exposure scale (spec 42 section 4); the front-end interface carries none, so 1
  xa.e1 = 1.0f;
  xa.e2 = 1.0f;
  xa.inv_pi = 1.0f / pi;
  DevPtr src = mode == 1 && io ? io->expo_src : expo_computed_;
  // native_direct, mode 1: k_expo_scalars_direct reads E_cur from the bound exposure texture (t3, the texel the input
  // pass copies into expo_src) instead of *src; src is then a valid but unread pointer
  const bool expo_tex = native_direct_ && mode == 1;
  if (expo_tex) src = expo_computed_;
  if (mode == 1 && !src) return fail("exposure from the game's texture without a source buffer");
  // k_expo_rule=v051 (A/B only): the 0.5.1 rule, the auto-exposure value as E_cur (so also as E_aux and E_prev)
  if (expo_v051_ && mode == 2 && auto_run) xa.src_mode = 1, xa.aux = 0, src = expo_computed_;
  if (!launch(expo_tex ? "k_expo_scalars_direct" : "k_expo_scalars", 1, 32, &kexpo_, &src, &expo_computed_, &xs_, &xa))
    return false;

  // the frame's inputs (k_io.py frame_in): dense render-size images, motion in render pixels ([MVlo]; the interop
  // converts output-resolution motion to render pixels in its input pass)
  KFrameIn fin = {};
  fin.color = cin;
  fin.motion = min_;
  fin.depth = din;
  fin.xs = xs_;
  fin.cpitch = cpitch;
  fin.mpitch = mpitch;
  fin.zpitch = zpitch;
  fin.W = W_;
  fin.H = H_;
  fin.rw = rw;
  fin.rh = rh;
  fin.Bw = Bw_;
  fin.Bh = Bh_;
  fin.flags = (hdr_ ? KIO_HDR : 0) | KIO_MVLO | (depth_inverted_ ? KIO_INV : 0) | (reset ? KIO_RESET : 0) |
              (raw ? KIO_C16 | KIO_M16 : 0);
  fin.mch = mch;
  if (raw) {
    fin.rmsx = io->ms[0];
    fin.rmsy = io->ms[1];
    fin.rmox = io->mo[0];
    fin.rmoy = io->mo[1];
  }
  fin.jx = fr.jitter[0];
  fin.jy = fr.jitter[1];
  fin.sx = fr.mv_scale[0];
  fin.sy = fr.mv_scale[1];
  fin.mox = fr.mv_bias[0];
  fin.moy = fr.mv_bias[1];
  fin.kappa = fin.rho = fin.lam = 1.0f;  // replaced by xs on the GPU
  fin.s = kS;

  // input stage (spec 42)
  KInParams ip = {};
  ip.f = fin;
  ip.Hc = Hc_[p];
  ip.Hh = Hh_[p];
  ip.x0 = net_["X0"];
  ip.cpre = cpre_;
  ip.cpre2 = cpre2_;
  const char* input_kernel = in_exact_ ? (reset ? "kx/k_in_reset" : "kx/k_in_base") : (reset ? "k_in_reset" : "k_in_base");
  // k_fast=1: the "_f" variant of an input / output stage kernel (k_io.hip KIO_FAST), else the exact kernel itself
  std::string kname_buf;
  // native_direct: the kernels that read the inputs or write the output (k_out_color, the tiled input kernels,
  // k_out_pixel[_c]) run as their "_direct" forms (same arguments; t0..t3 / u3 / b1 instead of the dense pointers)
  auto kname = [&](const char* n) -> const char* {
    const bool io_kernel = std::strncmp(n, "k_in_", 5) == 0 || std::strncmp(n, "k_out_", 6) == 0;
    if (in_exact_ && std::strncmp(n, "k_in_", 5) == 0) {  // input-stage kernels from k_io_exact.co
      const bool fallback = s_breakpoints_fallback_ && std::strncmp(n, "k_in_tiled", 10) == 0;
      kname_buf = std::string(fallback ? "kxs/" : "kx/") + n;
    } else if (fast_) {
      kname_buf = std::string(n) + "_f";
    } else {
      kname_buf = n;
    }
    if (native_direct_ && io_kernel) kname_buf += "_direct";
    if (approx_a1_ && kname_buf.rfind("kx/k_in_tiled", 0) == 0) kname_buf += "__a1";
    if (approx_a2_ && (kname_buf.rfind("k_out_color", 0) == 0 || kname_buf.rfind("k_out_pixel_c", 0) == 0))
      kname_buf += "__a2";
    return kname_buf.c_str();
  };
  // backend 1 (the host emulator) has no tiled input kernels; HIP (0) and the D3D12 devices (2 hybrid, 3 native) do
  const bool use_tiled = tiled_input_ && cfg_.backend != 1 && fin.s == 0.5f && tiled_guard(rw, rh, fin.jx, fin.jy);
  // k_approx=a1 on a legacy-input frame: the exact input stage runs (a1 has no legacy form); one line for the caller's
  // log, the first time in this context (take_frame_note)
  if (approx_a1_ && !use_tiled && !a1_legacy_noted_) {
    a1_legacy_noted_ = true;
    frame_note_ = "model K: k_approx=a1 applies to the tiled input stage only; this context's frames on the legacy input path (a render size above 0.68 x the output, a jitter outside [-0.5, 0.5], or input_tiled=0) run the standard input stage (this line once per context)";
  }
  // k_in_exact: the output colour taps by k_out_color (KFrame's exact plan), not by the tiled input kernel
  const bool use_tiled_color = use_tiled && !in_exact_ && !reset && 4LL * Bw_ >= W_ && 4LL * Bh_ >= H_;
  KiHost input_host = {};
  const bool use_uniforms = use_tiled && !native_direct_ && input_uniforms_ && input_uniforms_safe(fin, &input_host);
  auto input_stage = [&]() {
    if (use_tiled) {
      if (in_exact_ && !approx_a1_ && cfg_.backend >= 2 && !s_breakpoints_fallback_ && !s_breakpoints_)
        return fail("model K: exact tiled input would get a null S breakpoint table (arena offset 0); the frame is refused");
      KInParams local = ip;
      local.cpre = s_breakpoints_;
      local.cpre2 = 0;
      if (use_tiled_color) {
        KInColorParams fused = {local, ctap_, ctapT_};
        if (use_uniforms) {
          KInColorUniformParams supplied = {fused, input_host};
          if (!launch(kname("k_in_tiled_color_uniform"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &supplied)) return false;
        } else if (!launch(kname("k_in_tiled_color"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &fused)) return false;
        ++tiled_color_launches_;
      } else if (use_uniforms) {
        KInUniformParams supplied = {local, input_host};
        if (!launch(kname("k_in_tiled_uniform"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &supplied)) return false;
      } else if (!launch(kname("k_in_tiled"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &local)) return false;
      if (use_uniforms) ++input_uniform_launches_;
      ++tiled_launches_;
    } else {
      const std::string color_name = std::string(in_exact_ ? "kx/k_in_color" : "k_in_color") + (native_direct_ ? "_direct" : "");
      // the legacy input kernels write and read cpre / cpre2: never launched with a null one (packed as arena offset 0)
      if (!ip.cpre || !ip.cpre2)
        return fail("model K: " + color_name + " would get a null cpre / cpre2 (arena offset 0); the frame is refused");
      if (!launch(color_name.c_str(), cdiv((long long)rw * rh, kThreads), kThreads, &ip)) return false;
      ++legacy_color_launches_;
      // native_direct: k_in_color / k_in_base read the game's textures (their "_direct" forms); k_in_reset reads buffers only
      const std::string base_name = native_direct_ && !reset ? std::string(input_kernel) + "_direct" : input_kernel;
      if (!launch(base_name.c_str(), cdiv(4LL * Bw_ * Bh_, kThreads), kThreads, &ip)) return false;
      ++legacy_base_launches_;
    }
    return true;
  };
  if (diag_input2_ && !input_stage()) return false;
  // The output colour taps use only this frame's colour and exposure (spec 43 4.2/5.3).
  // Produce them before the input colour prepass: no dependency on network output or histories.
  // Their cost belongs to the input timing window below; the full-frame scope is unchanged.
  KOutParams color = {};
  color.f = fin;
  color.ctap = ctap_;
  color.ctapT = ctapT_;
  if (!mark(1)) return false;
  // k_color_at (D3D12 devices): k_out_color runs inside network() instead, unordered after that stage
  const bool color_late = !use_tiled_color && !color_at_.empty();
  if (!use_tiled_color && !color_late &&
      !launch(kname("k_out_color"), cdiv((long long)rw * rh, kThreads), kThreads, &color))
    return false;
  if (!input_stage()) return false;
  tiled_last_ = use_tiled;
  tiled_color_last_ = use_tiled_color;
  input_uniforms_last_ = use_uniforms;
  if (warm_at_ == "input" && !warm("input")) return false;  // k_warm=input: before the first stage

  // network (spec 41 / 44)
  auto after = [&](const char* s) -> bool {
    if (warm_at_ == s && !warm(s)) return false;
    if (color_late && color_at_ == s) {
      dev_->unordered_next();
      if (!launch(kname("k_out_color"), cdiv((long long)rw * rh, kThreads), kThreads, &color)) return false;
    }
    return true;
  };
  if (!mark(2) || !network(f, after)) return false;

  // output stage (spec 43) into the caller's output: RGBA16F rows (interop output copy), dense RGBA f32 (the interop's
  // output pass for model K, FrameIo.out_rgba32f: the fourth value is the output alpha), dense RGB f32 (FORMATS.md
  // "Per-frame runtime"; no alpha), or the own buffer (tests) in one of these layouts
  if (!mark(3)) return false;
  // interop option handover=staged: the output stage is next (FrameIo.out_signal)
  if (io && io->out_signal && !dev_->signal_fence(io->out_signal, io->out_signal_value)) return dfail();
  const int level = k_filter_level(rw, rh, W_, H_);
  KOutParams op = {};
  op.f = fin;
  op.N = net_["OUT"];
  op.L = L_[p];
  op.Hc = Hc_[p];
  op.table = tables_ + (size_t)level * kTableBytes;
  op.exp2tab = exp2tab_;
  op.Lout = L_[q];
  op.Hcout = Hc_[q];
  op.Hhout = Hh_[q];
  op.ctap = ctap_;
  op.ctapT = ctapT_;
  const int hp = io ? io->out_half_pitch : 0;
  op.out = io && io->out ? io->out : out_;
  // native_direct: k_out_pixel[_c]_direct writes the bound output texture (u3; omode below stays RGB f32: the
  // RGBA16F output-copy bytes are chosen by the b1 flag output_rne16, as for model M's direct post kernel)
  if (native_direct_) op.out = 0;
  if (!op.out && !native_direct_) return fail("frame without an output buffer (interop)");
  // the own output buffer (no FrameIo output) holds RGBA16F rows only if they fit (k_bench.py's backend 3 timing)
  if (hp > 0 && op.out == out_ && (size_t)hp * H_ > (size_t)W_ * H_ * 16)
    return fail("model K: RGBA16F rows do not fit the own output buffer");
  if (hp > 0) {
    if (hp % 8) return fail("model K: output rows must be a multiple of 8 bytes");
    op.omode = OUT_RGBA16F;
    op.opitch = hp / 8;
  } else if (io && io->out_rgba32f) {
    op.omode = OUT_RGBA32F;
    op.opitch = W_;
  } else {
    op.omode = OUT_RGB32F;
    op.opitch = W_;
  }
  {  // k_io.py out_uniforms: FP32 operations each rounded once (kinv comes from xs on the GPU)
    KOutUni& u = op.u;
    u.invW = 1.0f / (float)W_;
    u.invH = 1.0f / (float)H_;
    u.rhox = (float)rw * u.invW;
    u.rhoy = (float)rh * u.invH;
    u.rho2x = 0.5f * u.rhox;
    u.rho2y = 0.5f * u.rhoy;
    u.jmx = fin.jx - 0.5f;
    u.jmy = fin.jy - 0.5f;
    u.spx = fin.sx / u.rhox;  // [MVlo]
    u.spy = fin.sy / u.rhoy;
    u.invW16 = 1.0f / (float)W16_;
    u.invH16 = 1.0f / (float)H16_;
    u.kinv = 1.0f / fin.kappa;
    u.kux = f32_to_f16(-8.0f / u.rho2x);
    u.kvy = f32_to_f16(-8.0f / u.rho2y);
  }
  if (!launch(kname(compact_head_ ? "k_out_pixel_c" : "k_out_pixel"), cdiv(W16_, 64) * (H16_ / 2), kThreads, &op)) return false;
  if (!mark(4)) return false;

  // frame state (KFrame._advance)
  level_last_ = level;
  omode_last_ = op.omode;
  parity_ = q;
  fidx_ = f + 1;
  last_rw_ = rw;
  last_rh_ = rh;
  jprev_[0] = fr.jitter[0];
  jprev_[1] = fr.jitter[1];
  ++frames_;
  return true;
}

bool KRuntime::tiled_guard(int rw, int rh, float jx, float jy) const {
  return W_ > 0 && W_ <= 16384 && H_ > 0 && H_ <= 16384 && Bw_ > 0 && Bw_ <= 4096 && Bh_ > 0 && Bh_ <= 4096 &&
         rw > 0 && rw <= .68 * W_ && rh > 0 && rh <= .68 * H_ && std::isfinite(jx) && std::isfinite(jy) &&
         std::fabs(jx) <= .5f && std::fabs(jy) <= .5f;
}

// Backend 3 (native D3D12): only the exact default path has D3D12 ports (owner 2026-10-09: K exact to the original;
// the relaxed variants stay on HIP). k_head=full and k_expo_rule=v051 keep exact kernels and are allowed.
bool KRuntime::options_ok_direct() {
  const char* why = f16_                           ? "network=fused16 / fused8"
                    : fast_                        ? "k_fast=1"
                    : !in_exact_                   ? "the fast input stage (k_fast_in=1 / k_in_exact=0)"
                    : (diag_input2_ || diag_copyin_) ? "k_diag"
                                                     : nullptr;
  if (why)
    return fail(std::string("model K on the native D3D12 device runs the exact default path only (") + why +
                " runs on HIP)");
  return true;
}

std::vector<std::string> KRuntime::d3d12_kernels() const {
  const std::string d = native_direct_ ? "_direct" : "";
  std::vector<std::string> v;
  if (hdr_) v.push_back("exposure_auto_diet" + d);  // auto-exposure (HDR only): at create or per frame (mode 2)
  v.push_back("k_expo_scalars");
  if (native_direct_) v.push_back("k_expo_scalars_direct");  // exposure from the game's texture (mode 1)
  v.push_back("k_out_color" + d);
  if (tiled_input_) {
    const std::string tag = s_breakpoints_fallback_ ? "kxs/" : "kx/";
    const std::string suffix = approx_a1_ ? "__a1" : "";
    if (!native_direct_ && input_uniforms_) v.push_back(tag + "k_in_tiled_uniform" + suffix);
    v.push_back(tag + "k_in_tiled" + d + suffix);  // input_uniforms_safe refusing the host uniforms
  }
  if (approx_a2_) {  // k_approx=a2 (opt-in)
    v.push_back("k_out_color" + d + "__a2");
    if (compact_head_) v.push_back("k_out_pixel_c" + d + "__a2");
  }
  // the legacy input kernels: a render size above 0.68 x the output (up to 1.0, native AA / DLAA), a jitter outside
  // [-0.5, 0.5], or input_tiled=0 (k_in_reset reads buffers only: no direct form)
  v.push_back("kx/k_in_color" + d);
  v.push_back("kx/k_in_reset");
  v.push_back("kx/k_in_base" + d);
  // the eleven stages as network() names them; e4 / d4 take NG = 2 above 24 windows, which depends on the frame's
  // window offset (even, 0..6): every form this output size can launch
  for (const KStageInfo& st : k_stage_table()) {
    int lw, lh;
    k_level_size(Bw_, Bh_, st.level, &lw, &lh);
    for (int o = 0; o <= 6; o += 2) {
      const std::string n = st.name;
      const int ng = wide_ng(n, false, k_window_count(lw, lh, o, o));
      std::string k;
      if (st.head) k = compact_head_ ? "k_stage2_d0_hp" : "k_stage2_d0_hidden";
      else if (ng) k = "k_stagew" + std::to_string(ng) + "_" + n;
      else k = (n == "e0" || n == "e1" || n == "d1" ? "k_stage2_" : "k_stage_") + n;
      if (std::find(v.begin(), v.end(), k) == v.end()) v.push_back(k);
      // the round-to-odd form the create-time self-check can select (KRuntime::smx_select)
      if (!st.head && n != "e0" && n != "e1" && std::find(v.begin(), v.end(), k + "__rodd") == v.end())
        v.push_back(k + "__rodd");
    }
  }
  if (split_m_)  // k_split: stage m's phase kernels (k_stagew4_m[__rodd] stay listed above, unused while split)
    for (const char* k : {"k_split_m_attn", "k_split_m_attn__rodd", "k_split_m_wo", "k_split_m_w1", "k_split_m_w2"})
      v.push_back(k);
  v.push_back("k_mix_probe");
  v.push_back("k_mix_probe__rodd");
  v.push_back((compact_head_ ? "k_out_pixel_c" : "k_out_pixel") + d);
  if (!warm_at_.empty()) v.push_back("k_warm");  // k_warm (option k_warm, backend 3)
  v.push_back("timing_barrier");  // the timing marks (timing_log=1)
  return v;
}

std::string KRuntime::d3d12_missing() const {
  const std::string path = join(build_dir_, "dx12/manifest.json");
  std::vector<char> bytes;
  wgc::Json m;
  std::string jerr;
  if (!wg_read_file(path, &bytes) || !wgc::parse_json(std::string(bytes.begin(), bytes.end()), m, jerr))
    return "any kernel (" + path + " missing or unreadable)";
  std::map<std::string, std::string> co;  // port name -> the code object it stands for
  if (const auto* ks = m.find("kernels"); ks && ks->is_array())
    for (const auto& k : ks->items) {
      const auto* n = k.find("name");
      const auto* c = k.find("co");
      if (n && n->is_string() && c && c->is_string()) co[n->text] = c->text;
    }
  std::string missing;
  for (const std::string& k : d3d12_kernels()) {
    auto it = co.find(k);
    // the code object the port stands for must be one this runtime loads (else the device refuses it at launch)
    const bool tagged = k.rfind("kx/", 0) == 0 || k.rfind("kxs/", 0) == 0;
    const bool co_ok = it != co.end() &&
                       (tagged ? it->second == "k_io_exact.co"
                               : (it->second == "k_block.co" || it->second == "k_block_wide.co" ||
                                  it->second == "k_io.co" || it->second == "exposure.co" || it->second == "frame.co"));
    if (co_ok) continue;
    missing += (missing.empty() ? "" : ", ") + k + (it == co.end() ? "" : " (port of " + it->second + ")");
  }
  return missing;
}

bool KRuntime::direct_preflight(const WgRtFrame& fr, const FrameIo* io) {
  if (!native_direct_) return fail("model K: not a native direct runtime");
  if (!io || !io->native_direct) return fail("model K: the native direct runtime needs FrameIo.native_direct");
  if (io->raw_inputs || io->out_half_pitch || io->out_rgba16f || io->motion_hi || io->motion_hi_direct || io->out_signal)
    return fail("model K direct frame: no input copies, output rows, output-resolution motion buffer or staged hand-over");
  if (io->exposure_mode < -1 || io->exposure_mode > 2) return fail("model K direct frame: unsupported exposure mode");
  if (fr.reset < 0 || fr.reset > 2) return fail("model K direct frame: reset must be 0, 1 or 2");
  if (fr.rw <= 0 || fr.rh <= 0 || fr.rw > max_rw_ || fr.rh > max_rh_)
    return fail("model K direct frame: render size outside the configured maximum");
  if (!std::isfinite(fr.jitter[0]) || !std::isfinite(fr.jitter[1])) return fail("model K direct frame: non-finite jitter");
  // every render size up to the maximum and every finite jitter run on D3D12: the tiled input stage where its guard
  // holds, else the legacy input kernels (both have direct forms)
  return true;
}

const char* KRuntime::kernels() {
  kernels_text_ = std::string(native_direct_ ? "io=direct_textures " : "") + "model=K network=" + (f8_ ? "fused8" : f16_ ? "fused16" : "fused") +
                  (f8_ ? " fp8_stages=" + std::to_string(f8data_.size()) + (f8why_.empty() ? "" : " fp8_kept:" + f8why_)
                       : std::string()) +
                  (in_exact_ ? " input=k_io_exact.co" : " input=k_io.co") + " output=k_io.co exposure=k_expo_scalars" +
                  std::string(" input_tiled=") + (tiled_input_ ? "1" : "0") +
                  " input_actual=" + (tiled_last_ ? "tiled" : "legacy") +
                  " tiled_launches=" + std::to_string(tiled_launches_) +
                  " input_uniforms=" + (input_uniforms_last_ ? "host" : "gpu") +
                  " input_uniform_launches=" + std::to_string(input_uniform_launches_) +
                  " color_actual=" + (tiled_color_last_ ? "tiled" : "separate") +
                  " tiled_color_launches=" + std::to_string(tiled_color_launches_) +
                  " legacy_color_launches=" + std::to_string(legacy_color_launches_) +
                  " legacy_base_launches=" + std::to_string(legacy_base_launches_) +
                  (fast_ ? " k_fast=1" : "") + (expo_v051_ ? " k_expo_rule=v051" : "") +
                  smx_note_ + (split_m_ ? " split=m" : "") + approx_tag() + " hidden=fused_d0" + std::string(" head=") +
                  (compact_head_ ? "compact" : "full") +
                  " base=" + std::to_string(Bw_) + "x" + std::to_string(Bh_) +
                  " hdr=" + (hdr_ ? "1" : "0") + " mv=render" +
                  " filter_level=" + std::to_string(level_last_) +
                  " out=" + (omode_last_ == OUT_RGBA16F ? "rgba16f" : omode_last_ == OUT_RGBA32F ? "rgba32f" : "f32") +
                  std::string(compact_head_ ? " post=k_out_pixel_c filter=fused post_last=k_out_pixel_c filter_last=fused"
                                            : " post=k_out_pixel filter=fused post_last=k_out_pixel filter_last=fused") +
                  (diag_input2_ || diag_copyin_
                       ? std::string(" k_diag=") + (diag_input2_ ? "input2" : "") +
                             (diag_input2_ && diag_copyin_ ? "+" : "") + (diag_copyin_ ? "copyin" : "")
                       : std::string());
  return kernels_text_.c_str();
}

bool KRuntime::buffer(const std::string& name, DevPtr* p, size_t* bytes) {
  if (frames_ == 0) return fail("no frame has run yet");
  const int lp = parity_;  // the copies the last frame wrote (the next frame reads them)
  const size_t P16 = (size_t)W16_ * H16_, R = (size_t)info_.rw * info_.rh;
  struct E {
    DevPtr p;
    size_t b;
  };
  std::map<std::string, E> m = {
      // exposure.pair: [E_cur, E_prev, E_aux] of the last frame (raw values, k_expo_scalars); exposure.measured: the
      // last auto-exposure value (spec 07)
      {"exposure.pair", {kexpo_, 12}},      {"k.scalars", {xs_, 16}}, {"exposure.measured", {expo_computed_, 4}},
      {"k.x0", {net_["X0"], net_bytes_["X0"]}}, {"k.N", {net_["OUT"], net_bytes_["OUT"]}},
      {"k.Hc", {Hc_[lp], P16 * 8}},         {"k.L", {L_[lp], P16 * 8}},
      {"k.Hh", {Hh_[lp], (size_t)Bw_ * Bh_ * 8}}, {"filter.out", {out_, (size_t)W_ * H_ * (omode_last_ == OUT_RGBA32F ? 16 : 12)}},
      {"input.color", {color_, R * 16}},    {"input.depth", {depth_, R * 4}},
      {"input.motion", {motion_, R * 8}},
      // the network's intermediate maps after the frame (D3D12 port checks, clean/k/dx12/k_compare.py): the skip maps
      // S0..S4 (every encoder stage's block output), the ping-pong maps A / B (their last writers' contents) and the
      // output colour taps of k_out_color / the tiled colour kernel
      {"k.S0", {net_["S0"], net_bytes_["S0"]}}, {"k.S1", {net_["S1"], net_bytes_["S1"]}},
      {"k.S2", {net_["S2"], net_bytes_["S2"]}}, {"k.S3", {net_["S3"], net_bytes_["S3"]}},
      {"k.S4", {net_["S4"], net_bytes_["S4"]}}, {"k.A", {net_["A"], net_bytes_["A"]}},
      {"k.B", {net_["B"], net_bytes_["B"]}},   {"k.ctap", {ctap_, R * 8}}, {"k.ctapT", {ctapT_, R * 2}},
  };
  // the head map exists only with k_head=full; the default frame holds the compact pixel head in the same buffer
  if (name == "k.N" && compact_head_)
    return fail("buffer k.N (the 40-channel head map) needs the runtime option k_head=full; the default writes the "
                "compact pixel head (buffer k.Np)");
  if (name == "k.Np") {  // ... and the compact pixel head only without it
    if (!compact_head_)
      return fail("buffer k.Np (the compact pixel head) exists only with k_head=compact, the default; with "
                  "k_head=full the frame writes the 40-channel head map (buffer k.N)");
    *p = net_["OUT"];
    *bytes = (size_t)Bw_ * Bh_ * 32;
    return true;
  }
  auto it = m.find(name);
  if (it == m.end() || !it->second.p) return fail("unknown buffer name " + name);
  *p = it->second.p;
  *bytes = it->second.b;
  return true;
}

}  // namespace wgrt
