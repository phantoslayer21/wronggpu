// SPDX-License-Identifier: MIT
// Swin block kernels, design 2 ("v2"), for gfx1201 (RDNA4): an independent, speed-oriented design of the five fused
// Swin block kernels of spec/02-swin-block.md. Same kernel names, arguments, packed-weight and feature-map formats
// (clean/FORMATS.md, clean/runner/packing.py) and launch interface as kernels/swin.hip, so the runner and the runtime
// load it as a build variant (clean/runner/variants.py: "v2", code object swin__v2.co). The patch-merge and expand
// kernels are shared: this file includes kernels/swin.hip for them (its own block kernels are not exported here).
//
// Design (results/2026-10-05-perf-targets.md: on this GPU WMMA and vector instructions never overlap, so a block costs
// its matrix time plus its vector time; the matrix part already runs near peak): as few vector instructions per token
// as possible, nothing redundant, few transcendentals.
//   - MLP hidden path (the largest vector item): GELU(x) = clamp(x, 0, 448) - c(|x|) with the correction
//     c(a) = a Phi(-a) read from an FP16 table in LDS (index = FP16 bits of |x|), one FP32 subtraction
//     (v_fma_mix_f32) and one FP8 rounding: about 5 vector instructions per value and no transcendental, against
//     about 6 + 2 transcendentals (4 FMAs each) for gelusig. WG_V2_GELU selects it, gelusig, the exact GELU or a
//     direct q8(GELU) table.
//   - RMSNorm (C = 64): the q8 clamp is skipped when every |gamma| <= 440 / sqrt(C) (one check per wave; exact, since
//     |x_c| r <= sqrt(C)); FP8 inputs are decoded two at a time (v_cvt_pk_f32_fp8); the sums are FMA chains.
//   - Attention: the bias table is the accumulator of the first S = U^T U WMMA; the softmax skips the max subtraction
//     (the soft cap bounds tau*g(S) to +-7.63, so exp cannot overflow; mathematically identical); clamps are single
//     v_med3 (no canonicalizing v_max); V is rounded to FP16 by v_fma_mixlo/hi_f16 (two values per dword).
//   - The stored output clamps without the per-value NaN test (a NaN in the residual stream is still made visible: it
//     poisons the token's first stored value).
//   - C = 64 (enc1, dec1): the whole 48 KB block section is copied into LDS once per workgroup (with the 16 KB GELU
//     table: 64 KB; all global loads of the copy are issued before any LDS store, so the start costs one memory
//     latency); after that single barrier each wave runs its windows with no further barrier. 512 lanes, 2 windows
//     per wave.
//   - C = 96 / 128: weights streamed through double-buffered LDS units (they do not fit), the next unit fetched in
//     slices between the phases of the current one, one window per wave, 512-lane workgroups: every staged unit
//     serves 16 windows, 12 barriers per workgroup, about 170 VGPRs (8 waves per SIMD with 1536 registers).
//     Variants: 256-lane groups (v2t256), 2 windows per wave (v2w2), at most 32 KB LDS per group (v2lds32). (A split
//     design with 2-4 waves per window was measured slower, barrier-bound, and removed.)
//   - Bottleneck megakernel swin_bneck6 (WG_V2_BASEMATH builds, variant bmega): the six C = 128 blocks in one
//     persistent launch, ordered work queue with per-window-row completion counters instead of five launch
//     boundaries; arithmetic exactly as swin.hip (bit-identical to six base launches).
//   - WG_V2_PROFILE (variant v2prof): per-phase cycle counters (runner/profile_v2.py).
//   - MLP: the next hidden tile's W1 WMMAs are issued before the current tile's GELU (WG_V2_PIPE; no overlap gain on
//     this GPU, kept for latency hiding).
//   - Compiler control: scheduling fences and opaque per-tile offsets keep the compiler from hoisting all fragment
//     loads of a unit into registers (it spilled), and the accumulators are pinned before the per-lane store branch so
//     no WMMA can be sunk into it (WMMA needs every lane active).
//
// Arithmetic is the spec's (q8 / f16 roundings at the same points); FP32 sums are formed in a different order than
// in swin.hip, so rare FP8 rounding-midpoint cases can differ from it (counted as "inexact" by the sweep), except in
// WG_V2_BASEMATH builds, which repeat swin.hip's operations exactly.
//
// Mapping onto WMMA is dictated by the shared packed format (see swin.hip): one window = 16 tokens = the N dimension
// (token t in lanes t and t + 16), weights are the A operand, activations in "layout L" (lane half h holds channels
// [h C/2, (h+1) C/2), register group g holds channels h C/2 + 8g .. + 7).
#define WG_SWIN_NO_BLOCK_KERNELS 1
#include "swin.hip"

// ---- build-time knobs (variants.py "v2*" builds; defaults are the v2 configuration) --------------------------------
#ifndef WG_V2_THREADS_C64
#define WG_V2_THREADS_C64 512
#endif
#ifndef WG_V2_THREADS_C64ADD
#define WG_V2_THREADS_C64ADD WG_V2_THREADS_C64
#endif
#ifndef WG_V2_THREADS_C96
#define WG_V2_THREADS_C96 512
#endif
#ifndef WG_V2_THREADS_C96ADD
#define WG_V2_THREADS_C96ADD WG_V2_THREADS_C96
#endif
#ifndef WG_V2_THREADS_C128
#define WG_V2_THREADS_C128 512
#endif
// Windows per wave. C=64 (weights resident): the wave runs them one after another. C=96/128 (streamed): the windows
// of a wave share every staged weight unit and are interleaved.
#ifndef WG_V2_WPW_C64
#define WG_V2_WPW_C64 2
#endif
#ifndef WG_V2_WPW_C96
#define WG_V2_WPW_C96 1
#endif
#ifndef WG_V2_WPW_C128
#define WG_V2_WPW_C128 1
#endif
// Streamed kernels: header vectors read from global memory instead of LDS (less LDS per workgroup).
#ifndef WG_V2_HDR_GLOBAL
#define WG_V2_HDR_GLOBAL 0
#endif
// Timing instrumentation (variant v2prof): cycle counts per phase (SHADER_CYCLES) summed into the device global
// wg_v2_prof, read back by runner/profile_v2.py. Off in normal builds.
#ifndef WG_V2_PROFILE
#define WG_V2_PROFILE 0
#endif
// Occupancy floor (waves per SIMD the register allocator must leave room for; 1 = no constraint).
#ifndef WG_V2_MINW_C64
#define WG_V2_MINW_C64 1
#endif
#ifndef WG_V2_MINW_C96
#define WG_V2_MINW_C96 1
#endif
#ifndef WG_V2_MINW_C128
#define WG_V2_MINW_C128 1
#endif
// GELU of the MLP hidden layer:
//   3 (default) GELU(x) = clamp(x, 0, 448) - c(|x|), c(a) = a Phi(-a) from a 16 KB FP16 table in LDS indexed by the FP16
//     bits of |x|, one FP32 subtraction, one FP8 rounding (0.6 % of the FP8 results one step off the exact GELU, none
//     more; on random networks half of gelusig's downstream error); about 5 vector instructions per value
//   4 q8(GELU(x)) read directly from a 16 KB FP8 table indexed by the FP16 bits of x (0.9 % one step off, but large
//     |x| only to 2^-8 relative: on random networks 4x gelusig's downstream error); about 2 instructions per value
//   2 gelusig (wg_platform.h mode 2; 0.8 % one step off; 2 transcendentals per value)
//   0 wg_gelu (wg_platform.h WG_GELU_MODE, exact by default)
// The tables are a fixed function of GELU (swin_v2_gelu_table.h, generated by runner/gelu_table.py), not model data.
//
// WG_V2_BASEMATH 1 (variant bmega, which also holds the bottleneck megakernel): every arithmetic step is written
// exactly as in swin.hip (same operations in the same order, same FP8 packs, wg_gelu = gelusig), so the kernels give
// the same bytes as six launches of the base build (plain or _nc). Forces WG_V2_GELU 0. swin.hip's own arithmetic
// switches are followed too (WG_GELU_MODE, WG_GELU_MIX, WG_SMAX16, WG_NORM_DOT4): with the macros of a base variant
// (e.g. vec16) the kernels match that variant.
#ifndef WG_V2_BASEMATH
#define WG_V2_BASEMATH 0
#endif
#if WG_V2_BASEMATH
#undef WG_V2_GELU
#define WG_V2_GELU 0
#endif
#ifndef WG_V2_GELU
#define WG_V2_GELU 3
#endif
// Mode 3: mantissa bits of the index (7: 8 KB table; 8: 16 KB).
#ifndef WG_V2_GTAB_BITS
#define WG_V2_GTAB_BITS 8
#endif
#if WG_V2_GELU == 3 || WG_V2_GELU == 4
#include "swin_v2_gelu_table.h"
#if WG_V2_GELU == 4
#define WG_V2_TAB_TYPE u8
#define WG_V2_TAB_BYTES WG_V2_GTAB8_BYTES
#define WG_V2_TAB_INIT WG_V2_GTAB8_INIT
#define WG_V2_TAB_BITS 8
#elif WG_V2_GTAB_BITS == 7
#define WG_V2_TAB_TYPE unsigned short
#define WG_V2_TAB_BYTES WG_V2_CTAB7_BYTES
#define WG_V2_TAB_INIT WG_V2_CTAB7_INIT
#define WG_V2_TAB_BITS 7
#elif WG_V2_GTAB_BITS == 8
#define WG_V2_TAB_TYPE unsigned short
#define WG_V2_TAB_BYTES WG_V2_CTAB8_BYTES
#define WG_V2_TAB_INIT WG_V2_CTAB8_INIT
#define WG_V2_TAB_BITS 8
#else
#error "WG_V2_GTAB_BITS must be 7 or 8"
#endif
#if defined(__AMDGCN__)
__attribute__((device))
#endif
__attribute__((aligned(16))) const WG_V2_TAB_TYPE wg_v2_gtab_data[WG_V2_TAB_BYTES / sizeof(WG_V2_TAB_TYPE)] = {WG_V2_TAB_INIT};
#define WG_V2_GTAB_BYTES WG_V2_TAB_BYTES
#define wg_v2_gtab ((const u8*)wg_v2_gtab_data)
#else
#define WG_V2_GTAB_BYTES 16
#define WG_V2_TAB_BITS 8
#define wg_v2_gtab ((const u8*)nullptr)
#endif

#ifndef WG_V2_PIPE
#define WG_V2_PIPE 1
#endif
// Largest MLP unit staged at once by the streamed kernels (bytes); head units are one head each.
#ifndef WG_V2_UNIT_BYTES
#define WG_V2_UNIT_BYTES 16384
#endif
// C=64 kernels: 1 = whole block section resident in LDS (default), 0 = streamed like C=96/128 (comparison build).
#ifndef WG_V2_RESIDENT_C64
#define WG_V2_RESIDENT_C64 1
#endif

// WG_V2_SCHED_FENCE 1: scheduling fences (no instruction moves across) at tile and phase boundaries, so the compiler's
// scheduler cannot hoist the fragment loads of later tiles and run out of registers.
#ifndef WG_V2_SCHED_FENCE
#define WG_V2_SCHED_FENCE 1
#endif
#if defined(__AMDGCN__) && WG_V2_SCHED_FENCE
#define WG_V2_FENCE() __builtin_amdgcn_sched_barrier(0)
#else
#define WG_V2_FENCE() ((void)0)
#endif

// Opaque byte offset: the compiler cannot see through it, so it cannot merge (and hoist) the fragment loads of different
// tiles / phases into one early batch that would hold dozens of registers. (Merging inside one tile is kept.)
// WG_V2_KEEP(v): the value is materialized here, in uniform control flow. Used before the per-lane store branch: the
// compiler must not sink the last WMMAs into that branch (WMMA needs every lane of the wave active).
#if defined(__AMDGCN__)
#define WG_V2_OPAQUE(v) __asm__ volatile("" : "+s"(v))
#define WG_V2_KEEP(v) __asm__ volatile("" : "+v"(v))
#else
#define WG_V2_OPAQUE(v) ((void)0)
#define WG_V2_KEEP(v) ((void)0)
#endif

// ---- timing instrumentation (WG_V2_PROFILE) ---------------------------------------------------------------------
// wg_v2_prof[2 k] += cycles spent in phase k by a wave, wg_v2_prof[2 k + 1] += 1 per visit (lane 0 of each wave).
// Phase ids and names: runner/profile_v2.py (PHASES).
#define WG_V2_NPROF 32
#if defined(__AMDGCN__)
extern "C" __attribute__((device)) unsigned long long wg_v2_prof[2 * WG_V2_NPROF];
__attribute__((device)) unsigned long long wg_v2_prof[2 * WG_V2_NPROF] = {};
#else
extern "C" { unsigned long long wg_v2_prof[2 * WG_V2_NPROF] = {}; }
#endif
#if WG_V2_PROFILE
WG_FN unsigned long long wg_v2_clock() {
#if defined(__AMDGCN__)
  return __builtin_readcyclecounter();  // s_getreg SHADER_CYCLES (gfx12 has no s_memtime)
#else
  static unsigned long long c = 0;
  return ++c;
#endif
}
#define WG_V2_PROF_START() unsigned long long wg_prof_t = wg_v2_clock()
#define WG_V2_PROF_RESET() (wg_prof_t = wg_v2_clock())
#define WG_V2_PROF(k)                                                                    \
  do {                                                                                   \
    const unsigned long long wg_prof_n = wg_v2_clock();                                  \
    if ((wg_lane_id() & 31u) == 0) {                                                     \
      __atomic_fetch_add(&wg_v2_prof[2 * (k)], wg_prof_n - wg_prof_t, __ATOMIC_RELAXED); \
      __atomic_fetch_add(&wg_v2_prof[2 * (k) + 1], 1ull, __ATOMIC_RELAXED);             \
    }                                                                                    \
    wg_prof_t = wg_v2_clock();                                                           \
  } while (0)
#else
#define WG_V2_PROF_START() ((void)0)
#define WG_V2_PROF_RESET() ((void)0)
#define WG_V2_PROF(k) ((void)0)
#endif
// Barrier with its wait accounted to phase k (the work before it is accounted to k - 1 by the caller).
#define WG_V2_BARRIER(k) \
  do {                   \
    wg_barrier();        \
    WG_V2_PROF(k);       \
  } while (0)

namespace wgv2 {

constexpr int kGtabBytes = (WG_V2_GELU == 3 || WG_V2_GELU == 4) ? WG_V2_GTAB_BYTES : 0;

constexpr float kTau = 0.35337f;
constexpr float kCap = 32.369f;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kAlpha = kTau * kLog2e;                          // z = c (kAlpha + kBeta c^2) = tau g(S) log2(e)
constexpr float kBeta = -kTau * kLog2e / (3.0f * kCap * kCap);
constexpr float kNormEps = 1.0f / 8192.0f;                       // 2^-13; RMSNorm eps = 2^-13 / C
// Header vectors (FORMATS.md "Swin block section": a 16-byte flags word, then gamma1, gamma2, bp, b2, b1, bias).
// The kernels address them from `hdr` = the section + 16 (gamma1); copies to LDS leave the flags word out.
template <int C>
struct Hv {
  static constexpr int G1 = 0, G2 = G1 + 4 * C, BP = G2 + 4 * C, B2 = BP + 4 * C, B1 = B2 + 4 * C;
};

constexpr int largest_divisor_upto(int n, int limit) {
  int d = limit < n ? limit : n;
  while (d > 1 && n % d != 0) --d;
  return d < 1 ? 1 : d;
}

// Byte offsets of the packed block section (FORMATS.md "Swin block section"; identical to swin.hip / packing.py).
template <int C, int HEADS, int MLP>
struct Lay {
  static constexpr int KC = C / 16, MT = MLP / 16, FRAG = 256;
  static constexpr int FLAGS = 0, G1 = 16, G2 = G1 + 4 * C, BP = G2 + 4 * C, B2 = BP + 4 * C, B1 = B2 + 4 * C;
  static constexpr int BIAS = B1 + 4 * MLP;
  static constexpr int STREAM = BIAS + 1024 * HEADS;
  static constexpr int VEC = G1;                                 // `hdr` pointers point here (section + VEC)
  static constexpr int HDR = STREAM - VEC;                       // header bytes after the flags word
  static constexpr int QKV_FRAGS = 4 * KC;                       // per head: [Wqk|Wv] x (2 x KC), then Wp (KC x 2)
  static constexpr int HEAD_BYTES = 6 * KC * FRAG;
  static constexpr int TILE_FRAGS = 2 * KC;                      // per hidden tile: W1 (KC), then W2 (KC)
  static constexpr int TILE_BYTES = TILE_FRAGS * FRAG;
  static constexpr int head(int hd) { return STREAM + hd * HEAD_BYTES; }
  static constexpr int tile(int ht) { return STREAM + HEADS * HEAD_BYTES + ht * TILE_BYTES; }
  static constexpr int SIZE = STREAM + HEADS * HEAD_BYTES + MT * TILE_BYTES;
  // Streamed kernels: hidden tiles per staged MLP unit.
  static constexpr int HPU = largest_divisor_upto(MT, WG_V2_UNIT_BYTES / TILE_BYTES);
  static constexpr int MLP_BYTES = HPU * TILE_BYTES;
  static constexpr int NMLP = MT / HPU;
  static constexpr int MAXU = HEAD_BYTES > MLP_BYTES ? HEAD_BYTES : MLP_BYTES;
  static_assert(SIZE % 16 == 0 && STREAM % 16 == 0, "16-byte chunks");
  static_assert(G1 == 16 && B1 == 16 + 16 * C, "header offsets of FORMATS.md (flags word, then the vectors)");
};

// ---- small helpers -------------------------------------------------------------------------------------------------

WG_FN int mirror(int p, int n) {
  p = p < 0 ? -p : p;
  return p >= n ? 2 * (n - 1) - p : p;
}

// Fragment `index` of a fragment array (LDS or global): lane reads its 8 bytes.
WG_FN v2i frag(const u8* base, int index, u32 lane) { return ((const v2i*)(base + index * 256))[lane]; }

WG_FN float hsum8(v8f a) { return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])); }

WG_FN v8f fma8(v8f a, v8f b, v8f c) {
  v8f r;
#pragma unroll
  for (int v = 0; v < 8; ++v) r[v] = __builtin_fmaf(a[v], b[v], c[v]);
  return r;
}

// FP8 -> FP32 for 8 values, two per instruction on the GPU (v_cvt_pk_f32_fp8; E4M3FN like v_cvt_f32_fp8).
WG_FN v8f unpack8(v2i p) {
#if defined(__AMDGCN__) && !defined(WG_SOFT_FP8)
  typedef float v2f __attribute__((ext_vector_type(2)));
  const v2f a = __builtin_amdgcn_cvt_pk_f32_fp8(p[0], false), b = __builtin_amdgcn_cvt_pk_f32_fp8(p[0], true);
  const v2f c = __builtin_amdgcn_cvt_pk_f32_fp8(p[1], false), d = __builtin_amdgcn_cvt_pk_f32_fp8(p[1], true);
  v8f r;
  r[0] = a[0]; r[1] = a[1]; r[2] = b[0]; r[3] = b[1];
  r[4] = c[0]; r[5] = c[1]; r[6] = d[0]; r[7] = d[1];
  return r;
#else
  return wg_unpack_fp8(p);
#endif
}

// FP32 -> FP8 for 8 values that are already inside [-448, 448] (or NaN, which encodes as NaN): no clamp.
WG_FN v2i cvt8(v8f d) {
#if defined(__AMDGCN__) && !defined(WG_SOFT_FP8)
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(d[0], d[1], 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(d[2], d[3], lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(d[4], d[5], 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(d[6], d[7], hi, true);
  v2i r;
  r[0] = lo;
  r[1] = hi;
  return r;
#else
  return wg_pack_q8(d);
#endif
}

// Clamp to [lo, hi] as one v_med3_num_f32 (fminf/fmaxf would add a canonicalizing v_max per value). NaN -> finite.
WG_FN float clampf(float x, float lo, float hi) {
#if defined(__AMDGCN__)
  return __builtin_amdgcn_fmed3f(x, lo, hi);
#else
  return __builtin_fminf(__builtin_fmaxf(x, lo), hi);
#endif
}
WG_FN float sat448(float x) { return clampf(x, -448.0f, 448.0f); }

// FP32 -> FP16 (round to nearest even) for 8 values, two per dword with v_fma_mixlo_f16 / v_fma_mixhi_f16 (the
// compiler's own sequence is v_cvt_f16_f32 per value plus a v_pack_b32_f16 per pair). Checked by probe_v2_conv.
WG_FN v8h to_f16(v8f d) {
#if defined(__AMDGCN__)
  u32 r[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __asm__("v_fma_mixlo_f16 %0, %1, 1.0, 0" : "=v"(r[q]) : "v"(d[2 * q]));
    __asm__("v_fma_mixhi_f16 %0, %1, 1.0, 0" : "+v"(r[q]) : "v"(d[2 * q + 1]));
  }
  v4u w;
  w[0] = r[0]; w[1] = r[1]; w[2] = r[2]; w[3] = r[3];
  return __builtin_bit_cast(v8h, w);
#else
  return wg_to_f16(d);
#endif
}

// q8 of values that stay in the kernel (NaN becomes +-448, see wg_pack_q8_inner).
WG_FN v2i q8_inner(v8f d) {
  v8f s;
#pragma unroll
  for (int v = 0; v < 8; ++v) s[v] = sat448(d[v]);
  return cvt8(s);
}

// q8 of an inner value (input of the next matrix multiply). SAFE: the weights' flags word proves every inner pack in
// range (packing.block_bounds), no clamp. WG_V2_BASEMATH: swin.hip's own pack (same bytes for finite values).
template <bool SAFE>
WG_FN v2i pk_in(v8f d) {
#if WG_V2_BASEMATH
  return pack_in<SAFE>(d);
#else
  return SAFE ? cvt8(d) : q8_inner(d);
#endif
}

// q8 of values known to be >= -448 (GELU outputs): one v_min per value.
WG_FN v2i q8_upper(v8f d) {
  v8f s;
#pragma unroll
  for (int v = 0; v < 8; ++v) s[v] = __builtin_fminf(d[v], 448.0f);
  return cvt8(s);
}

// q8 of the stored block output. `poison` is +-0 normally and NaN when the token's residual stream holds a NaN or an
// infinity (see store()); it is added to the first value after the clamp, so such a token shows up as NaN in the
// output, as with swin.hip's NaN-preserving pack, without a NaN test per value.
WG_FN v2i q8_out(v8f d, float poison) {
  v8f s;
#pragma unroll
  for (int v = 0; v < 8; ++v) s[v] = sat448(d[v]);
  s[0] += poison;
  return cvt8(s);
}

// Two floats -> FP16 rounded toward zero, as the bits of one dword (low half = a): v_cvt_pk_rtz_f16_f32.
WG_FN u32 pk_rtz(float a, float b) {
#if defined(__AMDGCN__)
  return __builtin_bit_cast(u32, __builtin_amdgcn_cvt_pkrtz(a, b));
#else
  auto rtz = [](float v) -> u32 {
    _Float16 h = (_Float16)v;  // round to nearest even, then step toward zero if that rounded away from zero
    if (v == v && __builtin_fabsf((float)h) > __builtin_fabsf(v)) {
      unsigned short bits = __builtin_bit_cast(unsigned short, h);
      bits = (unsigned short)(bits - 1);  // same sign, next smaller magnitude (never crosses zero: |h| > |v| >= 0)
      h = __builtin_bit_cast(_Float16, bits);
    }
    return (u32)__builtin_bit_cast(unsigned short, h);
  };
  return rtz(a) | (rtz(b) << 16);
#endif
}

// GELU + q8 of one hidden tile (8 values per lane). gt: the GELU table in LDS (mode 3).
template <bool SAFE>
WG_FN v2i gelu_q8(v8f h, const u8* gt) {
#if defined(WG_V2_ABLATE_HIDDEN) && WG_V2_ABLATE_HIDDEN
  (void)gt;
  v2i r;  // timing-only: 6 float adds instead of GELU and the FP8 conversion; every value stays alive
  r[0] = __builtin_bit_cast(int, (h[0] + h[1]) + (h[2] + h[3]));
  r[1] = __builtin_bit_cast(int, (h[4] + h[5]) + (h[6] + h[7]));
  return r;
#elif WG_V2_GELU == 3
  // y = clamp(x, 0, 448) - c(|x|): the two FP16 corrections of a pair land in one dword (ds_load_u16, _d16_hi) and are
  // subtracted in FP32 by v_fma_mix_f32 (FP16 operand, exact); y >= -0.17, so one FP8 rounding without clamp.
  constexpr int M = WG_V2_TAB_BITS;
  constexpr u32 MASK = (1u << (5 + M)) - 1u;
  typedef unsigned short v2u16 __attribute__((ext_vector_type(2)));
  const unsigned short* ct = (const unsigned short*)gt;
  v8f y;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const u32 d = pk_rtz(h[2 * q], h[2 * q + 1]);
    const v2u16 c = {ct[(d >> (10 - M)) & MASK], ct[(d >> (26 - M)) & MASK]};
    const float m0 = clampf(h[2 * q], 0.0f, 448.0f), m1 = clampf(h[2 * q + 1], 0.0f, 448.0f);
#if defined(__AMDGCN__)
    const u32 cc = __builtin_bit_cast(u32, c);
    float y0, y1;
    __asm__("v_fma_mix_f32 %0, %1, -1.0, %2 op_sel_hi:[1,0,0]" : "=v"(y0) : "v"(cc), "v"(m0));
    __asm__("v_fma_mix_f32 %0, %1, -1.0, %2 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(y1) : "v"(cc), "v"(m1));
    y[2 * q] = y0;
    y[2 * q + 1] = y1;
#else
    // (scalar copies first: clang 18 bit-casts a vector element subscript as element 0 on the host)
    const unsigned short c0 = c[0], c1 = c[1];
    y[2 * q] = m0 - (float)__builtin_bit_cast(_Float16, c0);
    y[2 * q + 1] = m1 - (float)__builtin_bit_cast(_Float16, c1);
#endif
  }
  return cvt8(y);
#elif WG_V2_GELU == 4
  // q8(GELU(x)) = table[top 6 + M bits of FP16(x)]: no arithmetic beyond the conversion and the index.
  constexpr int M = WG_V2_TAB_BITS;
  typedef unsigned short v2u16 __attribute__((ext_vector_type(2)));
  // Bytes of values 0, 2 (lo) and 1, 3 (hi) go to the two 16-bit halves of two dwords (ds_load_u8 / _d16_hi), so
  // one shift-or per 4 values assembles the packed FP8 dword.
  u32 r2[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const u32 d0 = pk_rtz(h[4 * q], h[4 * q + 1]), d1 = pk_rtz(h[4 * q + 2], h[4 * q + 3]);
    const v2u16 a = {gt[(d0 & 0xffffu) >> (10 - M)], gt[(d1 & 0xffffu) >> (10 - M)]};
    const v2u16 b = {gt[d0 >> (26 - M)], gt[d1 >> (26 - M)]};
    r2[q] = __builtin_bit_cast(u32, a) | (__builtin_bit_cast(u32, b) << 8);
  }
  v2i r;
  r[0] = (int)r2[0];
  r[1] = (int)r2[1];
  return r;
#elif WG_V2_BASEMATH && (WG_GELU_MODE == 3 || WG_GELU_MODE == 4)
  (void)gt;
  return wg_hidden_mix<SAFE, (WG_GELU_MODE == 3 ? 4 : WG_GELU_MIX)>(h);  // swin.hip: packed FP16 / mixed hidden path
#else
  (void)gt;
  v8f g;
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const float x = h[v];
#if WG_V2_GELU == 2
    // gelusig: x / (1 + 2^(-z log2 e)), z = x (1.59989589 + 0.06951594 x^2) (constants of wg_platform.h mode 2)
    const float tt = __builtin_fmaf(x * x, -0.10029031f, -2.30816186f);
    g[v] = x * wg_rcp(1.0f + wg_exp2(x * tt));
#else
    g[v] = wg_gelu(x);
#endif
  }
#if WG_V2_BASEMATH
  return pack_in<SAFE>(g);  // swin.hip: SAFE ? wg_pack_q8_nc : wg_pack_q8_inner
#else
  return SAFE ? cvt8(g) : q8_upper(g);
#endif
#endif
}

// Global -> LDS copy in two steps: fetch() issues every global load of the unit (all of them before any store, so the
// latencies overlap), commit() writes them to LDS.
template <int BYTES, int THREADS>
struct Pre {
  static constexpr int CH = BYTES / 16, PF = (CH + THREADS - 1) / THREADS;
  v4u r[PF > 0 ? PF : 1];
  WG_FN void fetch(const u8* src, u32 tid) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const u32 idx = tid + (u32)(i * THREADS);
      if (CH % THREADS == 0 || (int)idx < CH) r[i] = ((const v4u*)src)[idx];
    }
  }
  WG_FN void commit(u8* dst, u32 tid) const {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const u32 idx = tid + (u32)(i * THREADS);
      if (CH % THREADS == 0 || (int)idx < CH) ((v4u*)dst)[idx] = r[i];
    }
  }
};

template <int N>
WG_FN void load_raw(const u8* p, v4u* r) {
#pragma unroll
  for (int i = 0; i < N; ++i) r[i] = ((const v4u*)p)[i];
}

// 16-byte chunk q of a lane's raw bytes as two fragments.
template <int KC>
WG_FN void raw_to_frags(const v4u* r, v2i* f) {
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    f[k][0] = (int)r[k / 2][2 * (k % 2)];
    f[k][1] = (int)r[k / 2][2 * (k % 2) + 1];
  }
}

// ---- one window's state and the block steps ---------------------------------------------------------------------

// Raw FP8 input bytes of a lane (C/2 bytes of token t, half hh) and where its output goes.
template <int C, bool ADD>
struct Raw {
  v4u x[C / 32];
  v4u a[ADD ? C / 32 : 1];
  int dst;
};

template <int C, bool ADD>
WG_FN void gather(Raw<C, ADD>& r, const u8* x, const u8* xadd, int win, bool valid, int nwx, int H, int W, int ox,
                  int oy, u32 t, u32 hh) {
  const int wy = win / nwx, wx = win - wy * nwx;
  const int pr = 4 * wy - oy + (int)(t >> 2), pc = 4 * wx - ox + (int)(t & 3u);
  const bool real = valid && pr >= 0 && pr < H && pc >= 0 && pc < W;
  r.dst = real ? pr * W + pc : -1;
  const size_t src = (size_t)(mirror(pr, H) * W + mirror(pc, W)) * C + hh * (C / 2);
  load_raw<C / 32>(x + src, r.x);
  if (ADD) load_raw<C / 32>(xadd + src, r.a);
}

// Steps 1-2 and the start of step 7: acc = x + bp, a = q8(RMSNorm(x; gamma1)). hdr: block section (LDS or global).
// SAFE: no clamp in the q8 (the flags word proves the packs in range, or every |gamma| <= 440 / sqrt(C), gamma_safe():
// then |x_c r gamma_c| <= sqrt(C) |gamma_c| < 448).
template <int C, bool ADD, bool SAFE>
WG_FN void norm1(const Raw<C, ADD>& r, const u8* hdr, u32 hh, v8f* acc, v2i* af) {
  constexpr int KC = C / 16;
  v2i xf[KC], xb[ADD ? KC : 1];
  raw_to_frags<KC>(r.x, xf);
  if (ADD) raw_to_frags<KC>(r.a, xb);
  const float* g1 = (const float*)(hdr + Hv<C>::G1) + hh * (C / 2);
  const float* bp = (const float*)(hdr + Hv<C>::BP) + hh * 8;
#if WG_V2_BASEMATH
  // swin.hip's sums and products, in its order
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    acc[k] = unpack8(xf[k]);
    if (ADD) acc[k] += unpack8(xb[k]);
    if (!(WG_NORM_DOT4 && !ADD)) ss += hsum8(acc[k] * acc[k]);
  }
  if (WG_NORM_DOT4 && !ADD) {  // swin.hip WG_NORM_DOT4: sum of squares from FP8 dot products
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      ss = wg_dot4_fp8(xf[k][0], xf[k][0], ss);
      ss = wg_dot4_fp8(xf[k][1], xf[k][1], ss);
    }
  }
  ss += wg_xor16(ss);
  const float rs = wg_rsq((ss + kNormEps) * (1.0f / C));
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    if (k % 2 == 0) WG_V2_FENCE();
    af[k] = pack_in<SAFE>(acc[k] * rs * wg_load8f(g1 + 8 * k));
    acc[k] += wg_load8f(bp + k * 16);
  }
#else
  v8f sq = wg_splat(0.0f);
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    acc[k] = unpack8(xf[k]);
    if (ADD) acc[k] += unpack8(xb[k]);  // decoder input e + S, exact in FP32
    sq = k == 0 ? acc[k] * acc[k] : fma8(acc[k], acc[k], sq);
  }
  float ss = hsum8(sq);
  ss += wg_xor16(ss);
  const float rs = wg_rsq((ss + kNormEps) * (1.0f / C));
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    if (k % 2 == 0) WG_V2_FENCE();  // load the vectors a few at a time
    const v8f n = acc[k] * (wg_load8f(g1 + 8 * k) * rs);
    af[k] = SAFE ? cvt8(n) : q8_inner(n);
    acc[k] += wg_load8f(bp + 16 * k);
  }
#endif
}

// Step 8 and b2: m = q8(RMSNorm(x1; gamma2)), acc = x1 + b2. Returns the sum of squares of x1 (for store()).
template <int C, bool SAFE>
WG_FN float norm2(v8f* acc, const u8* hdr, u32 hh, v2i* mf) {
  constexpr int KC = C / 16;
  const float* g2 = (const float*)(hdr + Hv<C>::G2) + hh * (C / 2);
  const float* b2 = (const float*)(hdr + Hv<C>::B2) + hh * 8;
#if WG_V2_BASEMATH
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < KC; ++k) ss += hsum8(acc[k] * acc[k]);
  ss += wg_xor16(ss);
  const float rs = wg_rsq((ss + kNormEps) * (1.0f / C));
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    if (k % 2 == 0) WG_V2_FENCE();
    mf[k] = pack_in<SAFE>(acc[k] * rs * wg_load8f(g2 + 8 * k));
    acc[k] += wg_load8f(b2 + k * 16);
  }
#else
  v8f sq = acc[0] * acc[0];
#pragma unroll
  for (int k = 1; k < KC; ++k) sq = fma8(acc[k], acc[k], sq);
  float ss = hsum8(sq);
  ss += wg_xor16(ss);
  const float rs = wg_rsq((ss + kNormEps) * (1.0f / C));
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    if (k % 2 == 0) WG_V2_FENCE();
    const v8f n = acc[k] * (wg_load8f(g2 + 8 * k) * rs);
    mf[k] = SAFE ? cvt8(n) : q8_inner(n);
    acc[k] += wg_load8f(b2 + 16 * k);
  }
#endif
  return ss;
}

// RMSNorm outputs are bounded: |x_c| r <= sqrt(C) (r = 1 / rms, up to the rsq rounding), so with every |gamma| <=
// 440 / sqrt(C) no q8 input of steps 2 and 8 can leave [-448, 448] and the clamp can be skipped (exact). Checked once
// per wave (resident kernel, where a wave runs several windows): each lane takes the maximum over its half's channels
// of gamma1 and gamma2, the halves are combined, the result is the same in every lane.
template <int C>
WG_FN bool gamma_safe(const u8* hdr, u32 hh) {
  const float* g = (const float*)(hdr + Hv<C>::G1) + hh * (C / 2);
  v8f m = wg_splat(0.0f);
#pragma unroll
  for (int k = 0; k < C / 16; ++k) {
    const v8f a = wg_load8f(g + 8 * k), b = wg_load8f(g + C + 8 * k);  // gamma1, gamma2 (offset 4C bytes)
#pragma unroll
    for (int v = 0; v < 8; ++v) m[v] = __builtin_fmaxf(m[v], __builtin_fmaxf(__builtin_fabsf(a[v]), __builtin_fabsf(b[v])));
  }
  float mx = __builtin_fmaxf(__builtin_fmaxf(__builtin_fmaxf(m[0], m[1]), __builtin_fmaxf(m[2], m[3])),
                             __builtin_fmaxf(__builtin_fmaxf(m[4], m[5]), __builtin_fmaxf(m[6], m[7])));
  mx = __builtin_fmaxf(mx, wg_xor16(mx));
  const int ok = mx <= 440.0f / __builtin_sqrtf((float)C) ? 1 : 0;  // NaN: not ok
#if defined(__AMDGCN__)
  return __builtin_amdgcn_readfirstlane(ok) != 0;
#else
  return ok != 0;
#endif
}

// Steps 3-7 for head hd of WPW windows: U = the head's fragments (QKV part, then its slice of Wp).
// hook.step(p), p = 0..3, is called at the start, between the three phases and at the end (weight staging slices).
template <int C, int HEADS, int MLP, int WPW, bool SAFE, class Hook>
WG_FN void attention(const u8* U, const u8* hdr, int hd, const v2i (*af)[C / 16], v8f (*acc)[C / 16], u32 lane,
                     Hook& hook) {
  typedef Lay<C, HEADS, MLP> L;
  constexpr int KC = L::KC;
  const u32 t = lane & 15u, hh = lane >> 4;
  // u = q8(Wqk a) (rows: the head's 32 channels in two tiles, columns: tokens); V^T = a^T Wv^T in FP16.
  v2i uf[WPW][2];
  v8h vt[WPW][2];
  hook.step(0);
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    if (mt == 1) hook.step(1);
    v8f aq[WPW], av[WPW];
    int off = mt * KC * 2 * 256;
    WG_V2_OPAQUE(off);
    const u8* Q = U + off;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      const v2i fq = frag(Q, kc * 2, lane), fv = frag(Q, kc * 2 + 1, lane);
#pragma unroll
      for (int i = 0; i < WPW; ++i) {
        aq[i] = wg_wmma_fp8(fq, af[i][kc], kc == 0 ? wg_splat(0.0f) : aq[i]);
        av[i] = wg_wmma_fp8(af[i][kc], fv, kc == 0 ? wg_splat(0.0f) : av[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < WPW; ++i) {
      uf[i][mt] = pk_in<SAFE>(aq[i]);
#if WG_V2_BASEMATH
      vt[i][mt] = wg_to_f16(av[i]);
#else
      vt[i][mt] = to_f16(av[i]);
#endif
    }
  }
  v2i of[WPW][2];
#if defined(WG_V2_ABLATE_ATTN) && WG_V2_ABLATE_ATTN
#pragma unroll
  for (int i = 0; i < WPW; ++i) {
    of[i][0] = uf[i][0] ^ __builtin_bit_cast(v2i, vt[i][0].lo);
    of[i][1] = uf[i][1] ^ __builtin_bit_cast(v2i, vt[i][1].hi);
  }
  (void)t;
  (void)hdr;
#else
  // Logits: lane (t, h) holds S(t, s) for 8 keys s; the packed bias table has the same order, so it is the
  // accumulator of the first WMMA. S = U^T U + B (U^T U is symmetric).
  const v8f bias = wg_load8f((const float*)(hdr + L::BIAS - L::VEC) + ((hd * 16 + (int)t) * 2 + (int)hh) * 8);
#pragma unroll
  for (int i = 0; i < WPW; ++i) {
#if WG_V2_BASEMATH
    // swin.hip: S = U^T U, then c = clamp(S + B), z = tau log2e (c - c^3 / (3 cap^2)), max subtracted
    v8f s = wg_wmma_fp8(uf[i][0], uf[i][0], wg_splat(0.0f));
    s = wg_wmma_fp8(uf[i][1], uf[i][1], s);
#if WG_SMAX16
    // swin.hip WG_SMAX16 (variant vec16): soft cap and softmax in packed FP16, same operations
    constexpr float k2 = kTau * kLog2e, k1 = kTau * kLog2e / (3.0f * kCap * kCap);
    const v2h capp = wg_splat2(kCap), capn = wg_splat2(-kCap), kk2 = wg_splat2(k2), kk1 = wg_splat2(-k1);
    v2h zq[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      v2h c = wg_h2(s[2 * k], s[2 * k + 1]) + wg_h2(bias[2 * k], bias[2 * k + 1]);
      c = __builtin_elementwise_min(__builtin_elementwise_max(c, capn), capp);
      zq[k] = c * __builtin_elementwise_fma(c * c, kk1, kk2);
    }
    const v2h m2 = __builtin_elementwise_max(__builtin_elementwise_max(zq[0], zq[1]),
                                              __builtin_elementwise_max(zq[2], zq[3]));
    float mx = wg_fmax((float)m2[0], (float)m2[1]);
    mx = wg_fmax(mx, wg_xor16(mx));
    const v2h mxv = wg_splat2(mx);
    v2h e2[4];
    float sum = 0.0f;
    const v2h ones = wg_splat2(1.0f);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      e2[k] = wg_exp2_h2(zq[k] - mxv);
      sum = wg_dot2_acc(e2[k], ones, sum);
    }
    sum += wg_xor16(sum);
    const v2h rv = wg_splat2(wg_rcp(sum));
    v8h p;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const v2h pk = e2[k] * rv;
      p[2 * k] = pk[0];
      p[2 * k + 1] = pk[1];
    }
#else
    v8f z;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float c = wg_fmin(wg_fmax(s[v] + bias[v], -kCap), kCap);
      z[v] = (c - c * c * c * (1.0f / (3.0f * kCap * kCap))) * (kTau * kLog2e);
    }
    float mx = hmax8(z);
    mx = wg_fmax(mx, wg_xor16(mx));
    v8f e;
#pragma unroll
    for (int v = 0; v < 8; ++v) e[v] = wg_exp2(z[v] - mx);
    float sum = hsum8(e);
    sum += wg_xor16(sum);
    const v8h p = wg_to_f16(e * wg_rcp(sum));
#endif
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) of[i][dt] = pack_in<SAFE>(wg_wmma_f16(vt[i][dt], p, wg_splat(0.0f)));
#else
    v8f s = wg_wmma_fp8(uf[i][0], uf[i][0], bias);
    s = wg_wmma_fp8(uf[i][1], uf[i][1], s);
    // A = softmax(tau g(S)): |tau g| <= tau 2c/3 = 7.63, so exp2 needs no max subtraction (same value).
    v8f e;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const float c = clampf(s[v], -kCap, kCap);
      e[v] = wg_exp2(c * __builtin_fmaf(c * c, kBeta, kAlpha));
    }
    float sum = hsum8(e);
    sum += wg_xor16(sum);
    const v8h p = wg_to_f16(e * wg_rcp(sum));
    // o = A V (rows: value channels, columns: tokens), q8.
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) of[i][dt] = pk_in<SAFE>(wg_wmma_f16(vt[i][dt], p, wg_splat(0.0f)));
#endif
  }
#endif
  // x1 += Wp[:, head hd] q8(o_hd)
  hook.step(2);
  int poff = L::QKV_FRAGS * 256;
  WG_V2_OPAQUE(poff);
  const u8* P = U + poff;
#pragma unroll
  for (int mt = 0; mt < KC; ++mt)
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      const v2i f = frag(P, mt * 2 + dt, lane);
#pragma unroll
      for (int i = 0; i < WPW; ++i) acc[i][mt] = wg_wmma_fp8(f, of[i][dt], acc[i][mt]);
    }
  hook.step(3);
}

// Steps 9-10 for n hidden tiles of WPW windows: acc += W2[:, tile] q8(GELU(W1[tile] m + b1[tile])).
// T = the tiles' fragments ([W1 (KC)][W2 (KC)] per tile), b1t = b1 of the first tile (tile-ordered, lane half applied).
template <int KC, int WPW>
WG_FN void w1(const u8* T, int j, const float* b1t, const v2i (*mf)[KC], v8f* h, u32 lane) {
  const v8f b = wg_load8f(b1t + 16 * j);
  int off = j * 2 * KC * 256;
  WG_V2_OPAQUE(off);
  T += off;
#pragma unroll
  for (int kc = 0; kc < KC; ++kc) {
    const v2i f = frag(T, kc, lane);
#pragma unroll
    for (int i = 0; i < WPW; ++i) h[i] = wg_wmma_fp8(f, mf[i][kc], kc == 0 ? b : h[i]);
  }
}

template <int KC, int WPW, bool SAFE>
WG_FN void w2(const u8* T, int j, const v8f* h, v8f (*acc)[KC], u32 lane, const u8* gt) {
  v2i hf[WPW];
#pragma unroll
  for (int i = 0; i < WPW; ++i) hf[i] = gelu_q8<SAFE>(h[i], gt);
  int off = (j * 2 * KC + KC) * 256;
  WG_V2_OPAQUE(off);
  T += off;
#pragma unroll
  for (int mt = 0; mt < KC; ++mt) {
    const v2i f = frag(T, mt, lane);
#pragma unroll
    for (int i = 0; i < WPW; ++i) acc[i][mt] = wg_wmma_fp8(f, hf[i], acc[i][mt]);
  }
}

// N > 0: N tiles, fully unrolled, hook.step(j) before tile j's work and hook.step(N) at the end (staging slices).
// N = 0: n tiles in a loop (no hook).
template <int KC, int WPW, int N, bool SAFE, class Hook>
WG_FN void mlp(const u8* T, int n, const float* b1t, const v2i (*mf)[KC], v8f (*acc)[KC], u32 lane, Hook& hook,
                const u8* gt) {
#if defined(WG_V2_ABLATE_MLP) && WG_V2_ABLATE_MLP
  (void)T; (void)n; (void)b1t; (void)lane; (void)gt;
#pragma unroll
  for (int j = 0; j <= N; ++j) hook.step(j);
#pragma unroll
  for (int i = 0; i < WPW; ++i)
#pragma unroll
    for (int k = 0; k < KC; ++k) acc[i][k] += unpack8(mf[i][k]);
#else
  if (N > 0) n = N;
#if WG_V2_PIPE
  // Software pipeline: tile j + 1's W1 chain is issued before tile j's GELU, in the same basic block.
  v8f hn[WPW];
  hook.step(0);
  w1<KC, WPW>(T, 0, b1t, mf, hn, lane);
  if (N > 0) {
#pragma unroll
    for (int j = 0; j + 1 < (N > 0 ? N : 1); ++j) {
      v8f hc[WPW];
#pragma unroll
      for (int i = 0; i < WPW; ++i) hc[i] = hn[i];
      hook.step(j + 1);
      w1<KC, WPW>(T, j + 1, b1t, mf, hn, lane);
      w2<KC, WPW, SAFE>(T, j, hc, acc, lane, gt);
    }
    w2<KC, WPW, SAFE>(T, n - 1, hn, acc, lane, gt);
  } else {
    // Unrolled by two (ping-pong between hn and hb, no register copies); n is even (MLP / 16 is even).
    v8f hb[WPW];
#pragma unroll 1
    for (int j = 0; j + 2 < n; j += 2) {
      w1<KC, WPW>(T, j + 1, b1t, mf, hb, lane);
      w2<KC, WPW, SAFE>(T, j, hn, acc, lane, gt);
      w1<KC, WPW>(T, j + 2, b1t, mf, hn, lane);
      w2<KC, WPW, SAFE>(T, j + 1, hb, acc, lane, gt);
    }
    w1<KC, WPW>(T, n - 1, b1t, mf, hb, lane);
    w2<KC, WPW, SAFE>(T, n - 2, hn, acc, lane, gt);
    w2<KC, WPW, SAFE>(T, n - 1, hb, acc, lane, gt);
  }
  hook.step(N);
#else
  if (N > 0) {
#pragma unroll
    for (int j = 0; j < (N > 0 ? N : 1); ++j) {
      v8f h[WPW];
      hook.step(j);
      w1<KC, WPW>(T, j, b1t, mf, h, lane);
      w2<KC, WPW, SAFE>(T, j, h, acc, lane, gt);
    }
    hook.step(N);
  } else {
#pragma unroll 1
    for (int j = 0; j < n; ++j) {
      v8f h[WPW];
      w1<KC, WPW>(T, j, b1t, mf, h, lane);
      w2<KC, WPW, SAFE>(T, j, h, acc, lane, gt);
    }
  }
#endif
#endif
}

struct NoHook {
  WG_FN void step(int) { WG_V2_FENCE(); }
};

// Output q8(x2) for the real window members. ss = the sum of squares of x1 from norm2 (NaN or inf if x1 holds a NaN
// or an infinity): ss * 0 is +-0 or NaN.
template <int C>
WG_FN void store(u8* out, int dst, const v8f* acc, float ss, u32 hh) {
  constexpr int KC = C / 16;
  v8f a8[KC];
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    a8[k] = acc[k];
    WG_V2_KEEP(a8[k]);
  }
  if (dst < 0) return;
  v4u o[KC / 2];
#pragma unroll
  for (int k = 0; k < KC; k += 2) {
#if WG_V2_BASEMATH
    (void)ss;
    const v2i a = wg_pack_q8(a8[k]), b = wg_pack_q8(a8[k + 1]);  // swin.hip: NaN-preserving clamp per value
#else
    const v2i a = q8_out(a8[k], k == 0 ? ss * 0.0f : 0.0f), b = q8_out(a8[k + 1], 0.0f);
#endif
    o[k / 2][0] = (u32)a[0];
    o[k / 2][1] = (u32)a[1];
    o[k / 2][2] = (u32)b[0];
    o[k / 2][3] = (u32)b[1];
  }
  v4u* p = (v4u*)(out + (size_t)dst * C + hh * (C / 2));
#pragma unroll
  for (int k = 0; k < KC / 2; ++k) p[k] = o[k];
}

// store() for a window whose q8 output is also kept in registers (keep[k], every lane, real member or not), for the
// fused patch merge of the persistent kernels. swin.hip's packs (WG_V2_BASEMATH).
template <int C>
WG_FN void store_keep(u8* out, int dst, const v8f* acc, u32 hh, v2i* keep) {
  constexpr int KC = C / 16;
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    keep[k] = wg_pack_q8(acc[k]);
    WG_V2_KEEP(keep[k]);
  }
  if (dst < 0) return;
  v4u o[KC / 2];
#pragma unroll
  for (int k = 0; k < KC; k += 2) {
    o[k / 2][0] = (u32)keep[k][0];
    o[k / 2][1] = (u32)keep[k][1];
    o[k / 2][2] = (u32)keep[k + 1][0];
    o[k / 2][3] = (u32)keep[k + 1][1];
  }
  v4u* p = (v4u*)(out + (size_t)dst * C + hh * (C / 2));
#pragma unroll
  for (int k = 0; k < KC / 2; ++k) p[k] = o[k];
}

// ---- kernel bodies --------------------------------------------------------------------------------------------------

// Resident weights: the whole block section (without its flags word) is copied into LDS once; each wave then runs its
// WPW windows one after another with no further barrier (needs SIZE <= 64 KB: C = 64). lds[i] = section byte VEC + i.
template <int C, int HEADS, int MLP, int WPW, bool ADD, int THREADS, bool SAFE>
WG_FN void block_resident(const u8* __restrict__ x, const u8* __restrict__ xadd, u8* __restrict__ out,
                          const u8* __restrict__ w, int H, int W, int ox, int oy) {
  typedef Lay<C, HEADS, MLP> L;
  constexpr int KC = L::KC;
  constexpr int BYTES = L::SIZE - L::VEC;
  WG_LDS __attribute__((aligned(16))) u8 lds[BYTES];
  WG_LDS __attribute__((aligned(16))) u8 gt[kGtabBytes > 0 ? kGtabBytes : 16];
  static_assert(BYTES + kGtabBytes <= 65536, "resident weights and the GELU table must fit in LDS");
  WG_V2_PROF_START();
  const u32 tid = wg_lane_id(), lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int nwx = (W + ox + 3) >> 2, nwin = nwx * ((H + oy + 3) >> 2);
  const int win0 = (int)(wg_block_id() * (THREADS / WG_WAVE) + (tid >> 5)) * WPW;

  Raw<C, ADD> raw;
  gather<C, ADD>(raw, x, xadd, win0 < nwin ? win0 : nwin - 1, win0 < nwin, nwx, H, W, ox, oy, t, hh);
  {
    Pre<BYTES, THREADS> wpre;
    Pre<kGtabBytes, THREADS> gpre;
    wpre.fetch(w + L::VEC, tid);
    gpre.fetch(wg_v2_gtab, tid);
    wpre.commit(lds, tid);
    gpre.commit(gt, tid);
  }
  WG_V2_PROF(20);
  WG_V2_BARRIER(21);
  const bool safe = SAFE || gamma_safe<C>(lds, hh);  // norm packs in range (the other packs: SAFE only)

  const float* b1t = (const float*)(lds + L::B1 - L::VEC) + hh * 8;
#pragma unroll 1
  for (int i = 0; i < WPW; ++i) {
    if (win0 + i >= nwin) break;  // wave-uniform; no barrier follows
    const Raw<C, ADD> cur = raw;
    if (i + 1 < WPW && win0 + i + 1 < nwin)  // prefetch the next window's input
      gather<C, ADD>(raw, x, xadd, win0 + i + 1, true, nwx, H, W, ox, oy, t, hh);
    v8f acc[1][KC];
    v2i f[1][KC];
    if (safe) norm1<C, ADD, true>(cur, lds, hh, acc[0], f[0]);
    else norm1<C, ADD, false>(cur, lds, hh, acc[0], f[0]);
    WG_V2_PROF(22);
    NoHook none;
#pragma unroll 1
    for (int hd = 0; hd < HEADS; ++hd)
      attention<C, HEADS, MLP, 1, SAFE>(lds + L::head(hd) - L::VEC, lds, hd, f, acc, lane, none);
    WG_V2_PROF(23);
    const float ss = safe ? norm2<C, true>(acc[0], lds, hh, f[0]) : norm2<C, false>(acc[0], lds, hh, f[0]);
    WG_V2_PROF(24);
    mlp<KC, 1, 0, SAFE>(lds + L::tile(0) - L::VEC, L::MT, b1t, f, acc, lane, none, gt);
    WG_V2_PROF(25);
    store<C>(out, cur.dst, acc[0], ss, hh);
    WG_V2_PROF(26);
  }
}

// Copies the next weight unit global -> LDS in POINTS slices while the current unit is used: step(p) writes slice
// p - 1 (fetched one step earlier) to LDS and issues the global loads of slice p, so at most two slices are held in
// registers and every load has one phase of work to arrive. Slice p = rows [p PF / POINTS, (p+1) PF / POINTS) of the
// unit's 16-byte chunks (row i = chunk tid + i THREADS).
template <int BYTES, int THREADS, int POINTS>
struct Slicer {
  static constexpr int CH = BYTES / 16, PF = (CH + THREADS - 1) / THREADS, RMAX = (PF + POINTS - 1) / POINTS;
  static constexpr int lo(int p) { return p * PF / POINTS; }
  const v4u* src;
  v4u* dst;
  u32 tid;
  v4u pre[2][RMAX > 0 ? RMAX : 1];
  WG_FN Slicer(const u8* s, u8* d, u32 t) : src((const v4u*)s), dst((v4u*)d), tid(t) {}
  WG_FN void step(int p) {
    WG_V2_FENCE();
    if (p > 0) {
#pragma unroll
      for (int i = lo(p - 1); i < lo(p); ++i) {
        const u32 idx = tid + (u32)(i * THREADS);
        if (CH % THREADS == 0 || (int)idx < CH) dst[idx] = pre[(p - 1) & 1][i - lo(p - 1)];
      }
    }
    if (p < POINTS) {
#pragma unroll
      for (int i = lo(p); i < lo(p + 1); ++i) {
        const u32 idx = tid + (u32)(i * THREADS);
        if (CH % THREADS == 0 || (int)idx < CH) pre[p & 1][i - lo(p)] = src[idx];
      }
    }
  }
};

// Streamed weights (C = 96, 128): one head unit per head, then MLP units of HPU hidden tiles, double-buffered in LDS
// (one barrier per unit); the WPW windows of a wave are interleaved on every unit.
//
// streamed_body runs one group of THREADS / 32 * WPW windows from their gathered input `raw` to the stored output.
// On entry the block's header vectors are at `hdr` (section + VEC) and head 0's unit is in LDS buffer `buf` (visible to every wave);
// unit u of the block then uses buffer (buf + u) & 1. `last` is the staging hook of the last MLP unit (the megakernel
// prefetches the next group's header and head-0 unit there, into buffer (buf + HEADS + NMLP) & 1).
template <int C, int HEADS, int MLP, int WPW, bool ADD, int THREADS, bool SAFE, class Last>
WG_FN void streamed_body(const Raw<C, ADD>* raw, u8* out, const u8* __restrict__ w, u8* lds, int buf, const u8* hdr,
                         const u8* gt, u32 tid, Last& last, v2i* keep = nullptr) {
  typedef Lay<C, HEADS, MLP> L;
  constexpr int KC = L::KC, MAXU = L::MAXU;
  typedef Slicer<L::HEAD_BYTES, THREADS, 3> HeadNext;      // during a head: the next head's unit
  typedef Slicer<L::MLP_BYTES, THREADS, 3> MlpFirst;       // during the last head: the first MLP unit
  typedef Slicer<L::MLP_BYTES, THREADS, L::HPU> MlpNext;   // during an MLP unit: the next one
  WG_V2_PROF_START();
  const u32 lane = tid & 31u, hh = lane >> 4;

  v8f acc[WPW][KC];
  v2i f[WPW][KC];
  int dst[WPW];
#pragma unroll
  for (int i = 0; i < WPW; ++i) {
    dst[i] = raw[i].dst;
    norm1<C, ADD, SAFE>(raw[i], hdr, hh, acc[i], f[i]);
  }
  WG_V2_PROF(22);
#pragma unroll 1
  for (int hd = 0; hd + 1 < HEADS; ++hd) {
    HeadNext next(w + L::head(hd + 1), lds + (buf ^ 1) * MAXU, tid);
    attention<C, HEADS, MLP, WPW, SAFE>(lds + buf * MAXU, hdr, hd, f, acc, lane, next);
    WG_V2_PROF(27);
    WG_V2_BARRIER(28);
    buf ^= 1;
  }
  {
    MlpFirst next(w + L::tile(0), lds + (buf ^ 1) * MAXU, tid);
    attention<C, HEADS, MLP, WPW, SAFE>(lds + buf * MAXU, hdr, HEADS - 1, f, acc, lane, next);
    WG_V2_PROF(27);
    WG_V2_BARRIER(28);
    buf ^= 1;
  }
  float ss[WPW];
#pragma unroll
  for (int i = 0; i < WPW; ++i) ss[i] = norm2<C, SAFE>(acc[i], hdr, hh, f[i]);
  WG_V2_PROF(24);
  const float* b1 = (const float*)(hdr + L::B1 - L::VEC) + hh * 8;
#pragma unroll 1
  for (int u = 0; u + 1 < L::NMLP; ++u) {
    MlpNext next(w + L::tile((u + 1) * L::HPU), lds + (buf ^ 1) * MAXU, tid);
    mlp<KC, WPW, L::HPU, SAFE>(lds + buf * MAXU, L::HPU, b1 + u * L::HPU * 16, f, acc, lane, next, gt);
    WG_V2_PROF(29);
    WG_V2_BARRIER(28);
    buf ^= 1;
  }
  mlp<KC, WPW, L::HPU, SAFE>(lds + buf * MAXU, L::HPU, b1 + (L::NMLP - 1) * L::HPU * 16, f, acc, lane, last, gt);
  WG_V2_PROF(29);
  if (keep != nullptr) {  // WPW = 1: the window's q8 output also stays in registers (fused patch merge)
    (void)ss;
    store_keep<C>(out, dst[0], acc[0], hh, keep);
  } else {
#pragma unroll
    for (int i = 0; i < WPW; ++i) store<C>(out, dst[i], acc[i], ss[i], hh);
  }
  WG_V2_PROF(26);
}

template <int C, int HEADS, int MLP, int WPW, bool ADD, int THREADS, bool SAFE>
WG_FN void block_streamed(const u8* __restrict__ x, const u8* __restrict__ xadd, u8* __restrict__ out,
                          const u8* __restrict__ w, int H, int W, int ox, int oy) {
  typedef Lay<C, HEADS, MLP> L;
  constexpr int MAXU = L::MAXU;
  WG_LDS __attribute__((aligned(16))) u8 lds[2 * MAXU];
  // Header vectors in LDS (LDS reads keep the register budget low), or read from global memory (WG_V2_HDR_GLOBAL:
  // 7-8 KB less LDS per workgroup).
  constexpr int kHdrBytes = WG_V2_HDR_GLOBAL ? 0 : L::HDR;
  WG_LDS __attribute__((aligned(16))) u8 hdr_lds[kHdrBytes > 0 ? kHdrBytes : 16];
  const u8* const hdr = WG_V2_HDR_GLOBAL ? w + L::VEC : hdr_lds;
  WG_LDS __attribute__((aligned(16))) u8 gt[kGtabBytes > 0 ? kGtabBytes : 16];
  WG_V2_PROF_START();
  const u32 tid = wg_lane_id(), lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int nwx = (W + ox + 3) >> 2, nwin = nwx * ((H + oy + 3) >> 2);
  const int win0 = (int)(wg_block_id() * (THREADS / WG_WAVE) + (tid >> 5)) * WPW;

  Raw<C, ADD> raw[WPW];
#pragma unroll
  for (int i = 0; i < WPW; ++i)
    gather<C, ADD>(raw[i], x, xadd, win0 + i < nwin ? win0 + i : nwin - 1, win0 + i < nwin, nwx, H, W, ox, oy, t, hh);
  {
    Pre<kHdrBytes, THREADS> hpre;
    Pre<L::HEAD_BYTES, THREADS> upre;
    Pre<kGtabBytes, THREADS> gpre;
    hpre.fetch(w + L::VEC, tid);
    upre.fetch(w + L::head(0), tid);
    gpre.fetch(wg_v2_gtab, tid);
    hpre.commit(hdr_lds, tid);
    upre.commit(lds, tid);
    gpre.commit(gt, tid);
  }
  WG_V2_PROF(0);
  WG_V2_BARRIER(1);
  NoHook none;
  streamed_body<C, HEADS, MLP, WPW, ADD, THREADS, SAFE>(raw, out, w, lds, 0, hdr, gt, tid, none);
}

// ---- bottleneck megakernel (WG_V2_BASEMATH builds: variant bmega) -------------------------------------------------
// The six C = 128 blocks b0..b5 in one persistent launch. The work is an ordered queue of items: item = one group of
// IPW = THREADS / 32 * WPW consecutive windows (raster order) of one block, block-major (all items of b0, then of b1,
// ...). A workgroup takes the next item with an atomic counter, waits until the window rows of the previous block that
// the item depends on are complete (per-row counters in global memory), runs it (streamed_body: the block's weights
// staged through LDS once per item) and adds its windows to its rows' counters. The dependency covers both the
// item's input (the previous block's output, read with the mirror at the borders) and the buffer it overwrites (two
// buffers alternate: x -> tmp -> out -> tmp -> out -> tmp -> out, so block k overwrites what block k - 1 read).
// No co-residency is assumed: an item only waits on smaller items, and the smallest unfinished item always runs, so
// any grid size works (the best is the resident capacity). While an item runs, its workgroup already holds the next
// one (claimed one item ahead, so the claim's latency is hidden).
// The last workgroup to finish resets the counters (the sync words are zero before and after every launch).
// Lanes per workgroup: 256 (8 windows per item, about 170 VGPRs: 8 waves per SIMD by registers). 512 lanes (16 windows
// per item, about 200 VGPRs, one workgroup per WGP) measured 50 % slower.
#ifndef WG_V2_MEGA_THREADS
#define WG_V2_MEGA_THREADS 256
#endif
#ifndef WG_V2_MEGA_WPW
#define WG_V2_MEGA_WPW 1
#endif
#ifndef WG_V2_MEGA_MINW
#define WG_V2_MEGA_MINW 1
#endif
// 1 (default): header vectors read from global memory (32 KB LDS per workgroup: 4 per WGP; measured fastest);
// 0: staged in LDS with the head-0 unit (40 KB: 3 per WGP).
#ifndef WG_V2_MEGA_HDR_GLOBAL
#define WG_V2_MEGA_HDR_GLOBAL 1
#endif

// Sync words (u32): [0] item queue, [1] finished workgroups, [16 + k NR + r] windows done in window row r of block k.
constexpr int kMegaSyncHead = 16;

#if defined(__AMDGCN__)
WG_FN u32 at_add(u32* p, u32 v) { return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
WG_FN u32 at_add_release(u32* p, u32 v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
WG_FN u32 at_add_acq_rel(u32* p, u32 v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}
WG_FN u32 at_load(const u32* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
WG_FN void at_store(u32* p, u32 v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
WG_FN void fence_release() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); }
WG_FN void fence_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
WG_FN void spin_pause() { __builtin_amdgcn_s_sleep(2); }
// A value read from LDS that is the same in every lane: tell the compiler (scalar registers, uniform branches).
WG_FN int uniform(int v) { return __builtin_amdgcn_readfirstlane(v); }
#else  // emulator: workgroups run one after another, so every wait is already satisfied when it is reached
WG_FN u32 at_add(u32* p, u32 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
WG_FN u32 at_add_release(u32* p, u32 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
WG_FN u32 at_add_acq_rel(u32* p, u32 v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
WG_FN u32 at_load(const u32* p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
WG_FN void at_store(u32* p, u32 v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
WG_FN void fence_release() {}
WG_FN void fence_acquire() {}
WG_FN void spin_pause() {}
WG_FN int uniform(int v) { return v; }
#endif

// Window grid of one block.
struct Grid {
  int ox, oy, nwx, nwy, nwin, items;
};
WG_FN Grid grid_of(int H, int W, int ox, int oy, int ipw) {
  Grid g;
  g.ox = ox;
  g.oy = oy;
  g.nwx = (W + ox + 3) >> 2;
  g.nwy = (H + oy + 3) >> 2;
  g.nwin = g.nwx * g.nwy;
  g.items = (g.nwin + ipw - 1) / ipw;
  return g;
}

template <int WPW, int THREADS, bool SAFE>
WG_FN void bneck6(const u8* x, u8* out, u8* tmp, u32* sync, const u8* w0, const u8* w1, const u8* w2, const u8* w3,
                  const u8* w4, const u8* w5, int H, int W, int oxe, int oye, int oxo, int oyo, int groups) {
  constexpr int C = 128, HEADS = 4, MLP = 512;
  typedef Lay<C, HEADS, MLP> L;
  constexpr int MAXU = L::MAXU, IPW = THREADS / WG_WAVE * WPW, NUNITS = HEADS + L::NMLP;
  WG_LDS __attribute__((aligned(16))) u8 lds[2 * MAXU];
  constexpr bool HG = WG_V2_MEGA_HDR_GLOBAL != 0;
  constexpr int HDR = HG ? 0 : L::HDR;  // LDS bytes of the staged header
  WG_LDS __attribute__((aligned(16))) u8 hdrs[HG ? 16 : L::HDR];
  WG_LDS u32 slot[4];
  WG_V2_PROF_START();
  const u32 tid = wg_lane_id();
  const Grid ge = grid_of(H, W, oxe, oye, IPW), go = grid_of(H, W, oxo, oyo, IPW);  // even / odd blocks
  const int total = 3 * (ge.items + go.items);
  const int NR = ge.nwy > go.nwy ? ge.nwy : go.nwy;
  u32* const rows = sync + kMegaSyncHead;
  auto weights = [&](int k) { return k == 0 ? w0 : k == 1 ? w1 : k == 2 ? w2 : k == 3 ? w3 : k == 4 ? w4 : w5; };
  auto locate = [&](int item, int& k, int& local) {  // item -> (block, item within the block)
    k = 0;
    local = item;
    while (k < 5 && local >= (k & 1 ? go.items : ge.items)) {
      local -= k & 1 ? go.items : ge.items;
      ++k;
    }
  };

  if (tid == 0) {
    slot[0] = at_add(sync, 1u);
    slot[1] = at_add(sync, 1u);
  }
  wg_barrier();
  int cur = uniform((int)slot[0]), nxt = uniform((int)slot[1]);
  int buf = 0, it = 0;
#pragma unroll 1
  while (cur < total) {
    // Lane values recomputed in every item: otherwise the compiler keeps their loop-invariant address arithmetic
    // live across the whole loop (about 30 registers).
    u32 ltid = wg_lane_id();
    WG_V2_KEEP(ltid);
    const u32 lt = ltid & 15u, lhh = (ltid & 31u) >> 4;
    int k, local;
    locate(cur, k, local);
    const Grid g = k & 1 ? go : ge;
    const u8* src = k == 0 ? x : (k & 1 ? tmp : out);
    u8* dstbuf = k & 1 ? out : tmp;
    const int wf = local * IPW, wl = (wf + IPW < g.nwin ? wf + IPW : g.nwin) - 1;
    const int r0 = wf / g.nwx, r1 = wl / g.nwx;
    {  // this item's header and head-0 unit (published by the wait's barrier)
      Pre<HDR, THREADS> hpre;
      Pre<L::HEAD_BYTES, THREADS> upre;
      hpre.fetch(weights(k) + L::VEC, ltid);
      upre.fetch(weights(k) + L::head(0), ltid);
      hpre.commit(hdrs, ltid);
      upre.commit(lds + buf * MAXU, ltid);
    }
    u32 after = (u32)total;  // the item after the next one (claimed now, used two items later)
    if (ltid == 0 && nxt < total) after = at_add(sync, 1u);
    if (k > 0 && ltid == 0) {
      // rows of block k - 1 this item depends on: those that wrote its input rows [lo, hi] (mirrored) and those that
      // read the rows it overwrites (a subset of [lo, hi], plus the mirror readers: window row 0 reads rows <= 3, the
      // last window row reads rows >= H - 4)
      const Grid p = k & 1 ? ge : go;
      const int a = 4 * r0 - g.oy, b = 4 * r1 - g.oy + 3;
      const int lo = a < 0 ? 0 : (b >= H && 2 * (H - 1) - b < a ? 2 * (H - 1) - b : a);
      const int hi = (b > -a ? b : -a) < H - 1 ? (b > -a ? b : -a) : H - 1;
      int q0 = (lo + p.oy) >> 2, q1 = (hi + p.oy) >> 2;
      if (lo <= 3) q0 = 0;
      if (hi >= H - 4 || q1 > p.nwy - 1) q1 = p.nwy - 1;
      const u32* rc = rows + (k - 1) * NR;
#pragma unroll 1
      for (int q = q0; q <= q1; ++q)
        while (at_load(rc + q) < (u32)p.nwx) spin_pause();
      fence_acquire();
    }
    wg_barrier();
    fence_acquire();  // every wave: no stale input from its caches
    WG_V2_PROF(2);
    Raw<C, false> raw[WPW];
#pragma unroll
    for (int i = 0; i < WPW; ++i) {
      const int win = wf + (int)(ltid >> 5) * WPW + i;
      gather<C, false>(raw[i], src, nullptr, win <= wl ? win : wl, win <= wl, g.nwx, H, W, g.ox, g.oy, lt, lhh);
    }
    const int nbuf = (buf + NUNITS) & 1;  // the buffer after the item's last unit: the next head 0 goes there
    const u8* hdr = HG ? weights(k) + L::VEC : hdrs;
    NoHook none;
    streamed_body<C, HEADS, MLP, WPW, false, THREADS, SAFE>(raw, dstbuf, weights(k), lds, buf, hdr, wg_v2_gtab, ltid,
                                                            none);
    fence_release();  // every wave: its stores are visible device-wide before the counters move
    if (ltid == 0) slot[2 + (it & 1)] = after;
    wg_barrier();
    if (ltid == 0) {
      u32* rc = rows + k * NR;
#pragma unroll 1
      for (int r = r0; r <= r1; ++r) {
        const int s0 = r * g.nwx > wf ? r * g.nwx : wf, s1 = (r + 1) * g.nwx - 1 < wl ? (r + 1) * g.nwx - 1 : wl;
        at_add_release(rc + r, (u32)(s1 - s0 + 1));
      }
    }
    WG_V2_PROF(3);
    cur = nxt;
    nxt = uniform((int)slot[2 + (it & 1)]);
    buf = nbuf;
    ++it;
  }
  if (tid == 0) {
    fence_release();
    if (at_add_acq_rel(sync + 1, 1u) == (u32)groups - 1u) {  // last workgroup: every other one has finished
      at_store(sync, 0u);
      for (int j = 0; j < 6 * NR; ++j) at_store(rows + j, 0u);
      at_store(sync + 1, 0u);
    }
  }
}

// ---- persistent "mid" kernel (WG_V2_BASEMATH builds): enc2 + its patch merge, b0..b5, dec2 + its expand -----------
// Same scheme as bneck6, with eight phases in one ordered queue (items of 8 windows, 256 lanes):
//   phase 0  enc2 block on x (F/4, 96 ch) -> skip (F/4, 96), fused patch merge -> A (F/8, 128)  [swin.hip fused_merge]
//   phase 1..6  b0..b5 on A -> B -> A -> B -> A -> B -> A (F/8, 128)
//   phase 7  dec2 block: fused expand of A (swin.hip fused_expand) + skip -> out (F/4, 96)
// An item waits for the window rows of the phases it reads (and, for b1..b5, of the previous block's readers of the
// buffer it overwrites), as in bneck6. Arithmetic, merge and expand exactly as swin.hip, so the result is the bytes of
// launching enc2 (block, merge), b0..b5 and dec2 (expand, block) of the base build.
#ifndef WG_V2_MID_THREADS
#define WG_V2_MID_THREADS 256
#endif

// Map rows [lo, hi] that window rows r0..r1 of a grid with row offset oy read from a map of H rows (border mirror).
WG_FN void read_rows(int r0, int r1, int oy, int H, int& lo, int& hi) {
  const int a = 4 * r0 - oy, b = 4 * r1 - oy + 3;
  lo = a < 0 ? 0 : (b >= H && 2 * (H - 1) - b < a ? 2 * (H - 1) - b : a);
  const int m = b > -a ? b : -a;
  hi = m < H - 1 ? m : H - 1;
}

// Window rows [q0, q1] of grid g that write map rows [lo, hi]; with `readers`, also the window rows that read them
// through the border mirror (row 0 reads rows <= 3, the last row reads rows >= H - 4).
WG_FN void writer_rows(const Grid& g, int H, int lo, int hi, bool readers, int& q0, int& q1) {
  q0 = (lo + g.oy) >> 2;
  q1 = (hi + g.oy) >> 2;
  if (readers && lo <= 3) q0 = 0;
  if (readers && hi >= H - 4) q1 = g.nwy - 1;
  q0 = q0 < 0 ? 0 : q0;
  q1 = q1 > g.nwy - 1 ? g.nwy - 1 : q1;
}

// One lane: wait until window rows q0..q1 (counters rc) have all nwx windows done.
WG_FN void wait_rows(const u32* rc, int q0, int q1, int nwx) {
#pragma unroll 1
  for (int q = q0; q <= q1; ++q)
    while (at_load(rc + q) < (u32)nwx) spin_pause();
}

// Decoder input of a lane with the fused expand: expanded tokens from LDS (records of fused_expand), skip from global.
template <int C>
WG_FN void gather_expanded(Raw<C, true>& r, const u8* e_lds, const u8* skip, int win, bool valid, int nwx, int H,
                           int W, int ox, int oy, u32 t, u32 hh, u32 wave) {
  const int wy = win / nwx, wx = win - wy * nwx;
  const int pr = 4 * wy - oy + (int)(t >> 2), pc = 4 * wx - ox + (int)(t & 3u);
  const bool real = valid && pr >= 0 && pr < H && pc >= 0 && pc < W;
  r.dst = real ? pr * W + pc : -1;
  const int mr = mirror(pr, H), mc = mirror(pc, W);
  load_raw<C / 32>(skip + (size_t)(mr * W + mc) * C + hh * (C / 2), r.a);
  const int low = ((mr >> 1) - wg_low0(4 * wy - oy, H)) * 2 + ((mc >> 1) - wg_low0(4 * wx - ox, W));
  const int slot = (int)wave * 16 + low * 4 + (((mr & 1) << 1) | (mc & 1));  // swin.hip's eslot
  load_raw<C / 32>(e_lds + (size_t)slot * C + hh * (C / 2), r.x);
}

// Head 0's unit of an item's block (bytes = HEAD_BYTES of its width) -> LDS buffer 0: every global load before any
// store. One copy for every phase (per-phase copies, one per width, cost about 24 registers).
WG_FN void stage_head(const u8* src, u8* dst, int bytes, u32 tid) {
  constexpr int PF = (Lay<128, 4, 512>::HEAD_BYTES / 16 + 255) / 256;
  const int ch = bytes >> 4;
  v4u r[PF];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if ((int)tid + i * 256 < ch) r[i] = ((const v4u*)src)[tid + i * 256];
#pragma unroll
  for (int i = 0; i < PF; ++i)
    if ((int)tid + i * 256 < ch) ((v4u*)dst)[tid + i * 256] = r[i];
}

// One 8-window item of a block phase: MCO > 0 adds the fused patch merge (output `aux`), ECL > 0 the fused expand
// (x = the low-resolution map, xadd = skip). Head 0's unit is in LDS buffer 0 on entry.
template <int C, int HEADS, int MLP, int MCO, int ECL, bool SAFE>
struct MidPhase {
  typedef Lay<C, HEADS, MLP> L;
  static constexpr int TAIL = ((L::SIZE + 255) / 256) * 256, KC = C / 16, THREADS = 256;
  static_assert((HEADS + L::NMLP) % 2 == 0, "every item starts in LDS buffer 0");
  WG_FN static void run(const u8* x, const u8* xadd, u8* out, u8* aux, const u8* __restrict__ w, int H, int W,
                        const Grid& g, int local, u8* lds, u32 tid) {
    WG_V2_PROF_START();
    const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4, wave = tid >> 5;
    const int wf = local * 8, wl = (wf + 8 < g.nwin ? wf + 8 : g.nwin) - 1, win = wf + (int)wave;
    Raw<C, (ECL > 0)> raw;
    if constexpr (ECL > 0) {
      fused_expand<C, ECL>(x, w + TAIL, lds + L::MAXU, H, W, g.nwx, g.nwin, g.ox, g.oy, tid, local);
      wg_barrier();  // the expanded tokens are in LDS
      gather_expanded<C>(raw, lds + L::MAXU, xadd, win <= wl ? win : wl, win <= wl, g.nwx, H, W, g.ox, g.oy, t, hh,
                         wave);
      wg_barrier();  // read before head 1's unit overwrites them
      WG_V2_PROF(4);
    } else {
      gather<C, false>(raw, x, nullptr, win <= wl ? win : wl, win <= wl, g.nwx, H, W, g.ox, g.oy, t, hh);
    }
    NoHook none;
    if constexpr (MCO > 0) {
      v2i keep[KC];
      streamed_body<C, HEADS, MLP, 1, (ECL > 0), THREADS, SAFE>(&raw, out, w, lds, 0, w + L::VEC, wg_v2_gtab, tid,
                                                                none, keep);
      WG_V2_PROF_RESET();
      wg_barrier();  // every wave is done with the weight buffers
      store_frags<KC>(lds + (size_t)wave * 16 * C + t * C + hh * (C / 2), keep);
      wg_barrier();
      fused_merge<C, MCO>(lds, w + TAIL, aux, H, W, g.nwx, g.nwin, g.ox, g.oy, tid, local);
      WG_V2_PROF(5);
    } else {
      streamed_body<C, HEADS, MLP, 1, (ECL > 0), THREADS, SAFE>(&raw, out, w, lds, 0, w + L::VEC, wg_v2_gtab, tid,
                                                                none);
    }
  }
};

template <bool SAFE>
WG_FN void mid(const u8* x, u8* out, u8* skip, u8* A, u8* B, u32* sync, const u8* we, const u8* w0, const u8* w1,
               const u8* w2, const u8* w3, const u8* w4, const u8* w5, const u8* wd, int H, int W, int oxe, int oye,
               int oxb0, int oyb0, int oxb1, int oyb1, int oxd, int oyd, int groups) {
  typedef MidPhase<96, 4, 384, 128, 0, SAFE> Enc;
  typedef MidPhase<128, 4, 512, 0, 0, SAFE> Blk;
  typedef MidPhase<96, 4, 384, 0, 128, SAFE> Dec;
  constexpr int MAXU = Lay<128, 4, 512>::MAXU;
  WG_LDS __attribute__((aligned(16))) u8 lds[2 * MAXU];
  WG_LDS u32 slot[4];
  WG_V2_PROF_START();
  const u32 tid = wg_lane_id();
  const int Hb = H >> 1, Wb = W >> 1;
  const Grid ge = grid_of(H, W, oxe, oye, 8), gb0 = grid_of(Hb, Wb, oxb0, oyb0, 8),
             gb1 = grid_of(Hb, Wb, oxb1, oyb1, 8), gd = grid_of(H, W, oxd, oyd, 8);
  const int total = ge.items + 3 * (gb0.items + gb1.items) + gd.items;
  const int NR = ge.nwy > gd.nwy ? ge.nwy : gd.nwy;  // window rows per phase (the F/8 grids have fewer)
  u32* const rows = sync + kMegaSyncHead;
  auto grid = [&](int p) { return p == 0 ? ge : p == 7 ? gd : ((p - 1) & 1) ? gb1 : gb0; };
  auto bw = [&](int k) { return k == 0 ? w0 : k == 1 ? w1 : k == 2 ? w2 : k == 3 ? w3 : k == 4 ? w4 : w5; };
  auto locate = [&](int item, int& p, int& local) {
    p = 0;
    local = item;
    while (p < 7 && local >= grid(p).items) {
      local -= grid(p).items;
      ++p;
    }
  };

  if (tid == 0) {
    slot[0] = at_add(sync, 1u);
    slot[1] = at_add(sync, 1u);
  }
  wg_barrier();
  int cur = uniform((int)slot[0]), nxt = uniform((int)slot[1]), it = 0;
#pragma unroll 1
  while (cur < total) {
    u32 ltid = wg_lane_id();  // recomputed per item (see bneck6)
    WG_V2_KEEP(ltid);
    int p, local;
    locate(cur, p, local);
    const Grid g = grid(p);
    const int wf = local * 8, wl = (wf + 8 < g.nwin ? wf + 8 : g.nwin) - 1;
    const int r0 = wf / g.nwx, r1 = wl / g.nwx;
    const u8* w = p == 0 ? we : p == 7 ? wd : bw(p - 1);
    const bool c96 = p == 0 || p == 7;  // one call (two inlined copies cost registers)
    stage_head(w + (c96 ? Enc::L::head(0) : Blk::L::head(0)), lds, c96 ? Enc::L::HEAD_BYTES : Blk::L::HEAD_BYTES, ltid);
    u32 after = (u32)total;
    if (ltid == 0 && nxt < total) after = at_add(sync, 1u);
    if (p > 0 && ltid == 0) {
      int lo, hi, q0, q1;
      if (p == 1) {  // b0 reads merged rows [lo, hi]: enc2 rows 2 lo .. 2 hi + 1
        read_rows(r0, r1, g.oy, Hb, lo, hi);
        writer_rows(ge, H, 2 * lo, 2 * hi + 1, false, q0, q1);
        wait_rows(rows, q0, q1, ge.nwx);
      } else if (p < 7) {  // b1..b5: the previous block's writers of its input and readers of its output buffer
        read_rows(r0, r1, g.oy, Hb, lo, hi);
        const Grid q = grid(p - 1);
        writer_rows(q, Hb, lo, hi, true, q0, q1);
        wait_rows(rows + (p - 1) * NR, q0, q1, q.nwx);
      } else {  // dec2: b5's rows of the low-resolution tokens it expands, enc2's rows of the skip it adds
        lo = Hb;
        hi = 0;
#pragma unroll 1
        for (int r = r0; r <= r1; ++r) {
          const int l = wg_low0(4 * r - g.oy, H), h = l + 1 < Hb ? l + 1 : Hb - 1;
          lo = l < lo ? l : lo;
          hi = h > hi ? h : hi;
        }
        const Grid q = grid(6);
        writer_rows(q, Hb, lo, hi, false, q0, q1);
        wait_rows(rows + 6 * NR, q0, q1, q.nwx);
        read_rows(r0, r1, g.oy, H, lo, hi);
        writer_rows(ge, H, lo, hi, false, q0, q1);
        wait_rows(rows, q0, q1, ge.nwx);
      }
      fence_acquire();
    }
    wg_barrier();
    fence_acquire();  // every wave: no stale input from its caches
    WG_V2_PROF(2);
    if (p == 0) {
      Enc::run(x, nullptr, skip, A, w, H, W, g, local, lds, ltid);
    } else if (p == 7) {
      Dec::run(A, skip, out, nullptr, w, H, W, g, local, lds, ltid);
    } else {
      const int k = p - 1;
      Blk::run(k & 1 ? B : A, nullptr, k & 1 ? A : B, nullptr, w, Hb, Wb, g, local, lds, ltid);
    }
    WG_V2_PROF_RESET();  // the item's work is in the phases of run()
    fence_release();  // every wave: its stores are visible device-wide before the counters move
    if (ltid == 0) slot[2 + (it & 1)] = after;
    wg_barrier();
    if (ltid == 0) {
      u32* rc = rows + p * NR;
#pragma unroll 1
      for (int r = r0; r <= r1; ++r) {
        const int s0 = r * g.nwx > wf ? r * g.nwx : wf, s1 = (r + 1) * g.nwx - 1 < wl ? (r + 1) * g.nwx - 1 : wl;
        at_add_release(rc + r, (u32)(s1 - s0 + 1));
      }
    }
    WG_V2_PROF(3);
    cur = nxt;
    nxt = uniform((int)slot[2 + (it & 1)]);
    ++it;
  }
  if (tid == 0) {
    fence_release();
    if (at_add_acq_rel(sync + 1, 1u) == (u32)groups - 1u) {  // last workgroup: every other one has finished
      at_store(sync, 0u);
      for (int j = 0; j < 8 * NR; ++j) at_store(rows + j, 0u);
      at_store(sync + 1, 0u);
    }
  }
}

}  // namespace wgv2

// ---- exported kernels (names and arguments as in swin.hip) --------------------------------------------------------
#if WG_V2_RESIDENT_C64
#define WG_V2_BODY_C64 block_resident
#else
#define WG_V2_BODY_C64 block_streamed
#endif

// Every block kernel exists twice, as in swin.hip: `name` clamps every inner FP8 pack to +-448, `name_nc` does not
// (launched by the runner when bit 0 of the weights' flags word says the weights prove every inner pack in range,
// packing.block_bounds); both give identical results where `name_nc` is used.
#define WG_V2_KERNEL(name, body, C, HEADS, MLP, WPW, ADD, MINW, THREADS)                                         \
  WG_KERNEL_NW(name, THREADS, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,      \
                                    int oy) {                                                                     \
    wgv2::body<C, HEADS, MLP, WPW, ADD, THREADS, false>(x, xadd, out, w, H, W, ox, oy);                           \
  }                                                                                                               \
  WG_KERNEL_NW(name##_nc, THREADS, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox, \
                                         int oy) {                                                                \
    wgv2::body<C, HEADS, MLP, WPW, ADD, THREADS, true>(x, xadd, out, w, H, W, ox, oy);                            \
  }

WG_V2_KERNEL(swin_block_c64, WG_V2_BODY_C64, 64, 2, 256, WG_V2_WPW_C64, false, WG_V2_MINW_C64, WG_V2_THREADS_C64)
WG_V2_KERNEL(swin_block_c64_add, WG_V2_BODY_C64, 64, 2, 256, WG_V2_WPW_C64, true, WG_V2_MINW_C64, WG_V2_THREADS_C64ADD)
WG_V2_KERNEL(swin_block_c96, block_streamed, 96, 4, 384, WG_V2_WPW_C96, false, WG_V2_MINW_C96, WG_V2_THREADS_C96)
WG_V2_KERNEL(swin_block_c96_add, block_streamed, 96, 4, 384, WG_V2_WPW_C96, true, WG_V2_MINW_C96, WG_V2_THREADS_C96ADD)
WG_V2_KERNEL(swin_block_c128, block_streamed, 128, 4, 512, WG_V2_WPW_C128, false, WG_V2_MINW_C128, WG_V2_THREADS_C128)

// Bottleneck megakernel (WG_V2_BASEMATH builds): b0..b5 in one launch, bit-identical to six launches of the base
// build. x: input of b0; out: output of b5; tmp: an H x W x 128 FP8 scratch map; sync: 16 + 6 ((H + 6) / 4) zeroed
// u32 words (left zero); w0..w5: the packed weights of b0..b5; (oxe, oye) / (oxo, oyo): window offsets of the even
// (b0, b2, b4) and odd (b1, b3, b5) blocks; groups: the number of workgroups launched (any; best: the resident
// capacity). swin_bneck6_nc: every block's weights have flags bit 0 set (no clamp in the inner packs).
#if WG_V2_BASEMATH
#define WG_V2_MEGA_KERNEL(name, SAFE)                                                                               \
  WG_KERNEL_NW(name, WG_V2_MEGA_THREADS, WG_V2_MEGA_MINW)(const u8* x, u8* out, u8* tmp, u32* sync, const u8* w0,   \
                                                          const u8* w1, const u8* w2, const u8* w3, const u8* w4,   \
                                                          const u8* w5, int H, int W, int oxe, int oye, int oxo,    \
                                                          int oyo, int groups) {                                     \
    wgv2::bneck6<WG_V2_MEGA_WPW, WG_V2_MEGA_THREADS, SAFE>(x, out, tmp, sync, w0, w1, w2, w3, w4, w5, H, W, oxe, oye, \
                                                           oxo, oyo, groups);                                        \
  }
WG_V2_MEGA_KERNEL(swin_bneck6, false)
WG_V2_MEGA_KERNEL(swin_bneck6_nc, true)

// Persistent "mid" kernel (stage "mid" = enc2 with its merge, b0..b5, dec2 with its expand; FORMATS.md). x: enc1's
// merged output (F/4, 96 ch); out: dec2's output (F/4, 96); skip: enc2's output (F/4, 96, scratch; holds the skip
// connection afterwards); A, B: F/8 x 128 scratch maps (A holds b5's output afterwards); sync: 16 + 8 ((H + 6) / 4) zeroed
// u32 words (left zero); weights: the enc2, b0..b5 and dec2 stage blobs (with their merge / expand tails); H, W: the
// F/4 size; offsets of enc2, the even and odd b-blocks, dec2; groups: the number of workgroups launched.
#define WG_V2_MID_KERNEL(name, SAFE)                                                                                  \
  WG_KERNEL_NW(name, 256, WG_V2_MEGA_MINW)(const u8* x, u8* out, u8* skip, u8* A, u8* B, u32* sync, const u8* we,       \
                                           const u8* w0, const u8* w1, const u8* w2, const u8* w3, const u8* w4,       \
                                           const u8* w5, const u8* wd, int H, int W, int oxe, int oye, int oxb0,       \
                                           int oyb0, int oxb1, int oyb1, int oxd, int oyd, int groups) {               \
    wgv2::mid<SAFE>(x, out, skip, A, B, sync, we, w0, w1, w2, w3, w4, w5, wd, H, W, oxe, oye, oxb0, oyb0, oxb1, oyb1, \
                    oxd, oyd, groups);                                                                                 \
  }
WG_V2_MID_KERNEL(swin_mid, false)
WG_V2_MID_KERNEL(swin_mid_nc, true)
#endif

// Probe for runner/selftest.py (synthetic data): the v2 conversion helpers against the reference, every byte position.
// unpacked[i] = unpack8(codes[i]); packed[i] = q8_inner(x[i]); upper[i] = q8_upper(x[i]); gelu[i] = GELU + q8 of x[i];
// half[i] = to_f16(x[i]).
WG_KERNEL(probe_v2_conv)(const v8f* x, const v2i* codes, v8f* unpacked, v2i* packed, v2i* upper, v2i* gelu, v8h* half,
                         int n) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
  if ((int)i < n) {
    half[i] = wgv2::to_f16(x[i]);
    unpacked[i] = wgv2::unpack8(codes[i]);
    packed[i] = wgv2::q8_inner(x[i]);
    upper[i] = wgv2::q8_upper(x[i]);
    gelu[i] = wgv2::gelu_q8<false>(x[i], wg_v2_gtab);  // the table read from global memory here
  }
}

#if defined(WG_EMU)
WG_EMU_REGISTER(swin_block_c64)
WG_EMU_REGISTER(swin_block_c64_add)
WG_EMU_REGISTER(swin_block_c96)
WG_EMU_REGISTER(swin_block_c96_add)
WG_EMU_REGISTER(swin_block_c128)
WG_EMU_REGISTER(swin_block_c64_nc)
WG_EMU_REGISTER(swin_block_c64_add_nc)
WG_EMU_REGISTER(swin_block_c96_nc)
WG_EMU_REGISTER(swin_block_c96_add_nc)
WG_EMU_REGISTER(swin_block_c128_nc)
WG_EMU_REGISTER(probe_v2_conv)
#if WG_V2_BASEMATH
WG_EMU_REGISTER(swin_bneck6)
WG_EMU_REGISTER(swin_bneck6_nc)
WG_EMU_REGISTER(swin_mid)
WG_EMU_REGISTER(swin_mid_nc)
#endif
#endif
