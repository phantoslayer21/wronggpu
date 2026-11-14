// SPDX-License-Identifier: MIT
// Swin stages of spec/02-swin-block.md for gfx1201 (RDNA4): fused Swin block, patch merge, expand.
//
// Data layouts and weight packing are described in clean/FORMATS.md and produced by clean/runner/packing.py.
//
// Mapping onto WMMA (16x16x16, wave32): one window = 16 tokens = the N dimension (token t of the window, raster
// index 4*row + col, lives in lanes t and t + 16). Weights are the M x K operand. With this mapping the FP32 result
// of one WMMA (lane holds column n = token, 8 rows) already has the shape of the next WMMA's FP8 B operand (lane
// holds column n = token, 8 K values), so no data moves between lanes except for the two RMSNorm sums and the
// softmax max/sum (one lane-exchange each).
//
// "Layout L" of a token's C channels: lane half h (lanes 16h..16h+15) holds channels [h*C/2, (h+1)*C/2); register
// group g of that half holds channels h*C/2 + 8g + 0..7. Feature maps in memory are [H][W][C] FP8 bytes, so each
// lane reads or writes C/2 contiguous bytes.
//
// Weights: each workgroup (8 waves) copies the packed weight fragments into LDS once and all its waves read them
// from there; the Swin block streams them in double-buffered units of at most 8 KB, so every weight byte is read
// from global memory once per 8 or 16 windows instead of once per window.
#include "wg_platform.h"

// Build-time knobs (build.py variants; defaults are the shipped configuration):
//   WG_SWIN_THREADS  default lanes per Swin block workgroup (128, 256 or 512); WG_THREADS_<kernel> overrides one
//   WG_WPW_C64 / _C96 / _C128  windows per wave in the block kernel of that width
//   WG_UNIT_BYTES    upper limit of a staged MLP weight unit
//   WG_GEMM_THREADS  lanes per patch-merge / expand workgroup (128 or 256);  WG_NT  token groups of 16 per wave
#ifndef WG_SWIN_THREADS
#define WG_SWIN_THREADS 256
#endif
#ifndef WG_WPW_C64
#define WG_WPW_C64 1
#endif
#ifndef WG_WPW_C96
#define WG_WPW_C96 1
#endif
#ifndef WG_WPW_C128
#define WG_WPW_C128 1
#endif
// Per-kernel workgroup sizes (lanes); all default to WG_SWIN_THREADS. (dec1 at 512 lanes was 6 % SLOWER than at 256 in
// 6 of 6 interleaved rounds, results/2026-10-05-swin-82d4bd8.md.)
#ifndef WG_THREADS_C64
#define WG_THREADS_C64 WG_SWIN_THREADS
#endif
#ifndef WG_THREADS_C64ADD
#define WG_THREADS_C64ADD WG_SWIN_THREADS
#endif
#ifndef WG_THREADS_C96
#define WG_THREADS_C96 WG_SWIN_THREADS
#endif
#ifndef WG_THREADS_C96ADD
#define WG_THREADS_C96ADD WG_SWIN_THREADS
#endif
#ifndef WG_THREADS_C128
#define WG_THREADS_C128 WG_SWIN_THREADS
#endif
// MLP: hidden tiles processed together (1 or 2). With 2, the two W1 accumulation chains of a pair are independent and
// are interleaved; the sums are added in the same order, so the result is bit-identical. (Measured: no gain, the
// dependent chains are not the limit; kept as a sweep option.) 2026-10-07: default 2, -0.007 ms per 1440p-Q frame with
// every Swin kernel (block kernels, fused dec1 + dec0, bottleneck megakernel), bit-identical on the real model.
#ifndef WG_MLP_GROUP
#define WG_MLP_GROUP 2
#endif
// WG_MINW_C64 / _C96 / _C128: waves per SIMD the register allocator must leave room for (the occupancy floor).
// C=96 is held at 5 (it would otherwise drift to 4 with a couple of extra registers).
#ifndef WG_MINW_C64
#define WG_MINW_C64 1
#endif
#ifndef WG_MINW_C96
#define WG_MINW_C96 5
#endif
#ifndef WG_MINW_C96MG
#define WG_MINW_C96MG WG_MINW_C96
#endif
#ifndef WG_MINW_C96EX
#define WG_MINW_C96EX WG_MINW_C96
#endif
#ifndef WG_MINW_C128
#define WG_MINW_C128 1
#endif
#ifndef WG_UNIT_BYTES
#define WG_UNIT_BYTES 16384
#endif
#ifndef WG_GEMM_THREADS
#define WG_GEMM_THREADS 256
#endif
#ifndef WG_NT
#define WG_NT 2
#endif
// WG_WAVE_SCALAR 1: the wave index of a block kernel goes through readfirstlane, so the window / row / column
// arithmetic (the integer division by nwx, the border tests) runs on the scalar unit instead of in every lane. Integer
// index math only (bit-identical). Not for the fused-expand kernels (ECL > 0): there it raised the VGPR count (200 ->
// 224, one wave per SIMD less) and was slower. 0: the vector form.
#ifndef WG_WAVE_SCALAR
#define WG_WAVE_SCALAR 1
#endif

namespace {

constexpr float kTau = 0.35337f;
constexpr float kCap = 32.369f;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kNormEps = 1.0f / 8192.0f;  // 2^-13; RMSNorm eps = 2^-13 / C

WG_FN int wg_mirror(int p, int n) {
  p = p < 0 ? -p : p;
  return p >= n ? 2 * (n - 1) - p : p;
}

// Load N/2 x 8-byte fragments (N*8 bytes, 16-byte aligned).
template <int N>
WG_FN void load_frags(const u8* p, v2i* f) {
  const v4u* q = (const v4u*)p;
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    v4u w = q[i];
    f[2 * i][0] = (int)w[0];
    f[2 * i][1] = (int)w[1];
    f[2 * i + 1][0] = (int)w[2];
    f[2 * i + 1][1] = (int)w[3];
  }
}

template <int N>
WG_FN void store_frags(u8* p, const v2i* f) {
  v4u* q = (v4u*)p;
#pragma unroll
  for (int i = 0; i < N / 2; ++i) {
    v4u w;
    w[0] = (u32)f[2 * i][0];
    w[1] = (u32)f[2 * i][1];
    w[2] = (u32)f[2 * i + 1][0];
    w[3] = (u32)f[2 * i + 1][1];
    q[i] = w;
  }
}

WG_FN float hsum8(v8f a) { return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])); }
WG_FN float hmax8(v8f a) {
  return wg_fmax(wg_fmax(wg_fmax(a[0], a[1]), wg_fmax(a[2], a[3])), wg_fmax(wg_fmax(a[4], a[5]), wg_fmax(a[6], a[7])));
}

// Byte offsets of the packed block weights (see FORMATS.md, "Swin block section").
// A header of FP32 vectors is followed by the matrix fragments as one stream, in the order the kernel consumes
// them, cut into units that the workgroup stages through LDS one after another (one barrier per unit):
//   per head hd: unit HEAD(hd) = QKV frags: for mt < 2, kc < KC: [Wqk frag (hd, mt, kc)][Wv frag (hd, mt, kc)]
//                                 then WP frags: for mt < KC, dt < 2: Wp frag (mt, 2 hd + dt)
//   per u < NMLP: unit MLP(u) = for j < HPU (ht = u HPU + j): [W1 frags (ht, kc < KC)][W2 frags (mt < KC, ht)]
constexpr int largest_divisor_upto(int n, int limit) {
  int d = limit < n ? limit : n;
  while (n % d != 0) --d;
  return d;
}

template <int C, int HEADS, int MLP>
struct BlockLayout {
  static constexpr int KC = C / 16;          // 16-channel groups of the model width
  static constexpr int MT = MLP / 16;        // 16-unit groups of the MLP hidden layer
  static constexpr int FRAG = 256;
  static constexpr int FLAGS = 0;            // u32 flags (bit 0: every inner FP8 pack is proven in range), 16 bytes
  static constexpr int G1 = 16;              // gamma1 [C] f32
  static constexpr int G2 = G1 + 4 * C;      // gamma2 [C] f32
  static constexpr int BP = G2 + 4 * C;      // bp [C] f32, tile order
  static constexpr int B2 = BP + 4 * C;      // b2 [C] f32, tile order
  static constexpr int B1 = B2 + 4 * C;      // b1 [MLP] f32, tile order
  static constexpr int BIAS = B1 + 4 * MLP;  // position bias [HEADS][16][2][8] f32
  static constexpr int STREAM = BIAS + 4 * HEADS * 256;
  static constexpr int QKV_FRAGS = 4 * KC;
  static constexpr int HEAD_BYTES = (QKV_FRAGS + 2 * KC) * FRAG;
  static constexpr int HPU = largest_divisor_upto(MT, WG_UNIT_BYTES / (2 * KC * FRAG));  // MLP hidden tiles per unit
  static constexpr int MLP_BYTES = 2 * KC * HPU * FRAG;
  static constexpr int NMLP = MT / HPU;
  static constexpr int MAXU = HEAD_BYTES > MLP_BYTES ? HEAD_BYTES : MLP_BYTES;
  static constexpr int head(int hd) { return STREAM + hd * HEAD_BYTES; }
  static constexpr int mlp(int u) { return STREAM + HEADS * HEAD_BYTES + u * MLP_BYTES; }
  static constexpr int SIZE = STREAM + HEADS * HEAD_BYTES + NMLP * MLP_BYTES;
  static_assert(HPU >= 1 && MT % HPU == 0, "MLP units must tile the hidden layer");
  static_assert(STREAM % 16 == 0, "stream must be 16-byte aligned");
};

#if defined(WG_ABLATE_STAGING) && WG_ABLATE_STAGING
#define WG_STAGING(...)
#else
#define WG_STAGING(...) __VA_ARGS__
#endif


// Copies weight units global -> LDS: fetch() issues the global loads early (into registers), commit() writes them
// to LDS later, so the load latency overlaps the computation on the previous unit. Each transfer moves exactly the
// unit's size (BYTES). When BYTES is not a whole number of (threads x 16 B) rows, only the last row is guarded by a
// lane test; with whole rows there is no per-lane guard at all. MAXU is the largest unit (register/LDS sizing).
template <int MAXU, int THREADS>
struct Stager {
  static constexpr int PFMAX = (MAXU / 16 + THREADS - 1) / THREADS;  // chunks per lane, largest unit
  v4u pre[PFMAX];
  template <int BYTES>
  WG_FN void fetch(const u8* src, u32 tid) {
    static_assert(BYTES % 16 == 0 && BYTES <= MAXU, "a unit is made of 16-byte chunks and fits the buffers");
    constexpr int CHUNKS = BYTES / 16, PF = (CHUNKS + THREADS - 1) / THREADS;
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const u32 idx = tid + (u32)(i * THREADS);
      if (CHUNKS % THREADS == 0 || i + 1 < PF || (int)idx < CHUNKS) pre[i] = ((const v4u*)src)[idx];
    }
  }
  template <int BYTES>
  WG_FN void commit(u8* dst, u32 tid) {
    constexpr int CHUNKS = BYTES / 16, PF = (CHUNKS + THREADS - 1) / THREADS;
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const u32 idx = tid + (u32)(i * THREADS);
      if (CHUNKS % THREADS == 0 || i + 1 < PF || (int)idx < CHUNKS) ((v4u*)dst)[idx] = pre[i];
    }
  }
};

// One staged WMMA fragment: lane l reads bytes [8l, 8l + 8) of fragment `index` of an LDS unit.
WG_FN v2i lfrag(const u8* unit, int index, u32 lane) { return ((const v2i*)(unit + index * 256))[lane]; }
#if defined(WG_ABLATE_BIAS) && WG_ABLATE_BIAS
#define WG_BIAS8(p) wg_splat(0.0f)
#else
#define WG_BIAS8(p) wg_load8f(p)
#endif
#ifndef WG_AV8
#define WG_AV8 0
#endif
// WG_NORM_DOT4: the first RMSNorm's sum of squares (blocks whose input is a plain FP8 map, i.e. not the decoder blocks)
// from v_dot4_f32_fp8_fp8 on the loaded FP8 bytes: 2 instructions per 16-channel group instead of 8 multiply-adds.
// Changes only the FP32 summation order.
// WG_MLP_PF: LDS prefetch in the MLP tile loop. The compiler issues the W2 fragment loads right before the W2 WMMAs (after
// the GELU), so every tile pays two LDS round trips (W2 here, W1 of the next tile at the loop top) in the wave's own
// dependency chain. 1: load the W2 fragments of the tile right after its W1 WMMAs are issued (hidden behind the GELU);
// 2: also load the W1 fragments of the next tile of the unit before this tile's W2 WMMAs. Costs 2 x KC (4 x KC)
// registers per tile of the group. Bit-identical.
#ifndef WG_MLP_PF
#define WG_MLP_PF 0
#endif
#ifndef WG_NORM_DOT4
#define WG_NORM_DOT4 0
#endif
// WG_PROFILE 1: timing-only build. The block kernels get two more arguments (u32* prof, u32 prof_stride): only waves
// whose global index is a multiple of prof_stride are instrumented (0 = none), because reading a clock costs the
// instrumented wave and, for the realtime message, a shared resource (the first version instrumented every wave and
// slowed the kernel about 7x). An instrumented wave reads the clock at every phase boundary and writes WG_PROF_WORDS
// words: the time spent in each phase, with all loads / LDS operations drained and the phase's results forced complete
// at every boundary (so each phase pays its own latency; this serializes the phases, it is a diagnostic, not the speed
// of the normal kernel). WG_PROFILE 2: no draining (the stalls land in the phase that first uses the data).
// Clock for the phase boundaries (WG_PROF_CLOCK): 1 (default) the shader cycle counter, one scalar s_getreg per read, no
// message traffic; 0 the 100 MHz realtime clock through s_sendmsg_rtn. Every instrumented wave also reads the realtime
// clock twice (start, end) so cycles can be converted to time and waves can be placed on a common timeline.
// Words: 0 load (global gather + first weight unit + barrier), 1 norm1, 2 QKV projections, 3 attention (S, softmax,
// A.V), 4 Wp projection, 5 head-loop staging (commit + barrier), 6 norm2, 7 W1, 8 GELU + pack, 9 W2, 10 MLP staging,
// 11 store (all in clock units), 12 total (clock units, includes the reads), 13 start tick (realtime, low 32 bits),
// 14 HW_ID1, 15 cost of one clock read (clock units), 16 total realtime ticks (10 ns), 17..19 unused.
#ifndef WG_FUSE
#define WG_FUSE 0  // runner geometry only: launch the fused merge / expand kernels (swin_block_*_mg / *_ex)
#endif
#ifndef WG_PROFILE
#define WG_PROFILE 0
#endif
#ifndef WG_PROF_CLOCK
#define WG_PROF_CLOCK 1
#endif
#define WG_PROF_WORDS 20
#if WG_PROFILE
#if WG_PROF_CLOCK
#define WG_PROF_NOW() wg_cycles()
#else
#define WG_PROF_NOW() ((u32)wg_realtime())
#endif
#define WG_PROF_ARG_DECL , u32* __restrict__ prof, u32 prof_stride
#define WG_PROF_ARG , prof, prof_stride
#define WG_PROF_BEGIN(wave)                                                              \
  const u32 prof_wave = (u32)wg_uniform((int)(wave));                                    \
  const bool prof_on = prof_stride != 0u && prof_wave % prof_stride == 0u;               \
  u32 prof_acc[WG_PROF_WORDS];                                                           \
  _Pragma("unroll") for (int k = 0; k < WG_PROF_WORDS; ++k) prof_acc[k] = 0;             \
  u64 prof_rt0 = 0;                                                                      \
  u32 prof_t = 0, prof_c0 = 0;                                                           \
  if (prof_on) {                                                                         \
    prof_rt0 = wg_realtime();                                                            \
    prof_c0 = WG_PROF_NOW();                                                             \
    prof_t = WG_PROF_NOW();                                                              \
    prof_acc[15] = prof_t - prof_c0;                                                     \
  }
#define WG_PHASE(p, syncval)                                                              \
  do {                                                                                    \
    if (prof_on) {                                                                        \
      if (WG_PROFILE == 1) { wg_drain(); wg_sync(syncval); }                              \
      const u32 prof_n = WG_PROF_NOW();                                                   \
      prof_acc[p] += prof_n - prof_t;                                                     \
      prof_t = prof_n;                                                                    \
    }                                                                                     \
  } while (0)
#define WG_PROF_END()                                                                     \
  do {                                                                                    \
    if (prof_on) {                                                                        \
      prof_acc[12] = WG_PROF_NOW() - prof_c0;                                             \
      prof_acc[13] = (u32)prof_rt0;                                                       \
      prof_acc[14] = wg_hwid();                                                           \
      prof_acc[16] = (u32)(wg_realtime() - prof_rt0);                                     \
      if (lane == 0) {                                                                    \
        _Pragma("unroll") for (int k = 0; k < WG_PROF_WORDS; ++k)                         \
          prof[(size_t)prof_wave * WG_PROF_WORDS + k] = prof_acc[k];                      \
      }                                                                                   \
    }                                                                                     \
  } while (0)
#else
#define WG_PROF_ARG_DECL
#define WG_PROF_ARG
#define WG_PROF_BEGIN(wave)
#define WG_PHASE(p, syncval) ((void)0)
#define WG_PROF_END() ((void)0)
#endif
#ifndef WG_SMAX16
#define WG_SMAX16 0
#endif
template <bool SAFE>
WG_FN v2i pack_in(v8f d);
#if defined(WG_ABLATE_QPACK) && WG_ABLATE_QPACK
template <bool SAFE>
WG_FN v2i pack_in(v8f d) {  // timing-only stand-in for wg_pack_q8_inner: 6 adds, every value stays alive
  v2i r;
  r[0] = __builtin_bit_cast(int, (d[0] + d[1]) + (d[2] + d[3]));
  r[1] = __builtin_bit_cast(int, (d[4] + d[5]) + (d[6] + d[7]));
  return r;
}
#else
template <bool SAFE>
WG_FN v2i pack_in(v8f d) { return SAFE ? wg_pack_q8_nc(d) : wg_pack_q8_inner(d); }
#endif
#if defined(WG_ABLATE_LDS) && WG_ABLATE_LDS
WG_FN v2i fake_frag(int index, u32 lane) {  // timing-only: a fragment made from registers, no LDS read
  v2i r;
  r[0] = (int)lane * 7 + index;
  r[1] = index * 13 - (int)lane;
  return r;
}
#define WG_MLP_FRAG(U, idx, lane) fake_frag(idx, lane)
#else
#define WG_MLP_FRAG(U, idx, lane) lfrag(U, idx, lane)
#endif

// Fused expand (decoder blocks, WG_FUSE): the block input e = q8(We y + be) of a workgroup's 8 windows is computed here
// from the low-resolution map y instead of being read from a separate expand kernel's output. A window's 16 tokens (at
// mirrored positions near the map border) come from at most 2 x 2 low-resolution tokens with top-left (a, b), each
// expanded to its 4 sub-pixels, so a tile of 16 columns = 4 windows x 4 low-resolution tokens. The 8 waves share the
// M tiles of the expansion (4 sub-pixels x C/16 channel groups); every wave runs 2 N tiles. Results (FP8) go to LDS
// records [window][low token][sub-pixel][C] in exactly the format of the standalone expand's output.
// Requires even window offsets (all real ones are) and 8 windows per workgroup.
WG_FN int wg_low0(int p0, int n) {  // smallest low-resolution coordinate any of the 4 window rows / columns maps to
  int m = wg_mirror(p0, n);
  const int m1 = wg_mirror(p0 + 1, n), m2 = wg_mirror(p0 + 2, n), m3 = wg_mirror(p0 + 3, n);
  m = m1 < m ? m1 : m;
  m = m2 < m ? m2 : m;
  m = m3 < m ? m3 : m;
  return m >> 1;
}

template <int C, int CL>
WG_FN void fused_expand(const u8* __restrict__ y, const u8* __restrict__ tail, u8* e_lds, int H, int W, int nwx,
                        int nwin, int ox, int oy, u32 tid, int blk) {
  constexpr int KC = CL / 16, G = C / 16, MT = 4 * G, WE = 16 * C;
  static_assert(MT % 8 == 0, "the expansion tiles must divide over the 8 waves");
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4, wave = tid >> 5;
  const int Hl = H >> 1, Wl = W >> 1;
  v2i yf[2][KC];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    int win = blk * 8 + 4 * n + (int)(t >> 2);
    win = win < nwin ? win : nwin - 1;
    const int wy = win / nwx, wx = win - wy * nwx;
    int lr = wg_low0(4 * wy - oy, H) + (int)((t >> 1) & 1u), lc = wg_low0(4 * wx - ox, W) + (int)(t & 1u);
    lr = lr < Hl ? lr : Hl - 1;
    lc = lc < Wl ? lc : Wl - 1;
    load_frags<KC>(y + (size_t)(lr * Wl + lc) * CL + hh * (CL / 2), yf[n]);
  }
  const float* be = (const float*)tail + hh * 8;
#pragma unroll 1
  for (int mt = (int)wave; mt < MT; mt += 8) {
    const int k = mt / G, g = mt - k * G;
    v8f acc[2];
    const v8f b = wg_load8f(be + mt * 16);
    acc[0] = b;
    acc[1] = b;
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      const v2i f = ((const v2i*)(tail + WE + (mt * KC + kc) * 256))[lane];
      acc[0] = wg_wmma_fp8(f, yf[0][kc], acc[0]);
      acc[1] = wg_wmma_fp8(f, yf[1][kc], acc[1]);
    }
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int slot = (4 * n + (int)(t >> 2)) * 16 + (int)(t & 3u) * 4 + k;
      *(v2i*)(e_lds + (size_t)slot * C + hh * (C / 2) + 8 * g) = wg_pack_q8(acc[n]);
    }
  }
}

// Fused patch merge (spec 02), second half: the 8 windows' q8 outputs are in LDS as [window][token][C]; the 32 merged
// tokens (2 tiles of 16 = 4 windows x 4 2x2 groups) are computed by the 8 waves, each taking output tiles mt = wave,
// wave + 8, ... Same arithmetic and order as patch_merge. tail = the merge weights (bias first, then fragments).
// A separate (not inlined) function: inlined into the C=96 block it raised the kernel's register use from 188 to 211
// (or 17 spilled registers at the occupancy floor), although the merge itself needs few registers.
template <int C, int MCO>
WG_FN_NOINLINE void fused_merge(const u8* lds, const u8* __restrict__ tail, u8* __restrict__ aux, int H, int W, int nwx,
                               int nwin, int ox, int oy, u32 tid, int blk) {
  constexpr int KZ = C / 4, MO = MCO / 16;
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int Wm = W >> 1;
  const u8* zp[2];  // this lane's merged token in LDS, 2C bytes = KZ fragments (read per step, not held in registers)
  int tok[2];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int wi = 4 * n + (int)(t >> 2), gy = (int)((t >> 1) & 1u), gx = (int)(t & 1u);
    const int gw = blk * 8 + wi;
    const int win = gw < nwin ? gw : nwin - 1;
    const int wy = win / nwx, wx = win - wy * nwx;
    const int pr = 4 * wy - oy + 2 * gy, pc = 4 * wx - ox + 2 * gx;
    tok[n] = gw < nwin && pr >= 0 && pr < H && pc >= 0 && pc < W ? (pr >> 1) * Wm + (pc >> 1) : -1;
    zp[n] = lds + (size_t)wi * 16 * C + (size_t)(4 * (2 * gy + (int)hh) + 2 * gx) * C;
  }
  const float* bm = (const float*)tail + hh * 8;
  const u8* wm = tail + 4 * MCO;
#pragma unroll 1
  for (int mt = (int)(tid >> 5); mt < MO; mt += 8) {
    v8f macc[2];
    const v8f b = wg_load8f(bm + mt * 16);
    macc[0] = b;
    macc[1] = b;
#pragma unroll 4
    for (int g = 0; g < KZ; ++g) {
      const v2i f = ((const v2i*)(wm + (mt * KZ + g) * 256))[lane];
      macc[0] = wg_wmma_fp8(f, ((const v2i*)zp[0])[g], macc[0]);
      macc[1] = wg_wmma_fp8(f, ((const v2i*)zp[1])[g], macc[1]);
    }
#pragma unroll
    for (int n = 0; n < 2; ++n)
      if (tok[n] >= 0) *(v2i*)(aux + (size_t)tok[n] * MCO + hh * (MCO / 2) + 8 * mt) = wg_pack_q8(macc[n]);
  }
}

// One Swin block (spec 02 steps 1-10) on WPW windows per wave, 8 waves per workgroup; output q8(x2) for the real
// window members. ADD: the block input is x + xadd (decoder stages: expanded source + skip), both FP8 maps.
// MCO > 0 (encoder, WG_FUSE): the patch merge to MCO channels is done at the end of the block (output in `aux`);
// ECL > 0 (decoder): x is the low-resolution source with ECL channels and the expand is done at the start. Both need
// THREADS = 256, WPW = 1 and even window offsets.
// Output epilogue hook (EPI): the default stores q8(x2) to `out`. A fused kernel in another file (dec1_dec0.hip, branch
// clean/perf-enc) passes its own EPI type instead: EPI::run<C, KC, WPW, THREADS>(acc, dst, W, lds, lds_bytes, tid, epi)
// is called by EVERY lane of the workgroup after the block (acc = x2 in FP32, layout L; dst = output index or -1 for
// mirrored members; the 2 * MAXU bytes of `lds` may be reused after a barrier). Contract kept stable for that kernel:
// template parameters, layout L of acc, and this call (see clean/kernels/dec1_dec0.hip).
struct SwinStoreEpi {
  static constexpr bool kFused = false;
};

// Gather source hook (SRC): the default reads the FP8 map x [H][W][C] at the mirrored member position. A custom SRC
// (kCustom; Model L, clean/l/kernels/l_swin.hip) provides SRC::load<C, KC>(x, pr, pc, H, W, hh, f): the FP32 values
// of the lane's channels hh * C/2 + 8 k + j (k < KC) of the member at grid position (pr, pc), which may lie outside
// the grid. Plain blocks only (no ADD, merge or expand).
struct SwinFp8Src {
  static constexpr bool kCustom = false;
};

template <int C, int HEADS, int MLP, int WPW, bool ADD, int GROUP, bool SAFE, int THREADS, int MCO = 0, int ECL = 0,
          class EPI = SwinStoreEpi, bool RESIDENT = false, class SRC = SwinFp8Src>
WG_FN void swin_block(const u8* __restrict__ x, const u8* __restrict__ xadd, u8* __restrict__ out,
                      const u8* __restrict__ w, int H, int W, int ox, int oy, u8* __restrict__ aux WG_PROF_ARG_DECL,
                      const void* epi = nullptr, int nwg = 0) {
  typedef BlockLayout<C, HEADS, MLP> L;
  constexpr int KC = L::KC, HALF = C / 2, MAXU = L::MAXU;
  // RESIDENT (kernels *_r): the block's whole weight stream (every head and MLP unit) stays in LDS for the whole
  // launch; each wave runs windows wave_id, wave_id + total waves, ... with no staging and no barrier after the first
  // copy. Same arithmetic per window as the staged kernel (bit-identical).
  constexpr int RES = L::SIZE - L::STREAM;
  static_assert(!RESIDENT || (MCO == 0 && ECL == 0 && !EPI::kFused && WPW == 1), "resident: plain block kernels only");
  static_assert(!SRC::kCustom || (MCO == 0 && ECL == 0 && !ADD), "custom gather source: plain block kernels only");
  WG_LDS __attribute__((aligned(16))) u8 lds[RESIDENT ? (RES + 15) / 16 * 16 : 2 * MAXU];
  Stager<MAXU, THREADS> stage;
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int nwx = (W + ox + 3) >> 2, nwy = (H + oy + 3) >> 2, nwin = nwx * nwy;
  const u32 wave_id = (WG_WAVE_SCALAR && ECL == 0) ? (u32)__builtin_amdgcn_readfirstlane((int)(tid >> 5)) : (tid >> 5);
  int win0 = (int)(wg_block_id() * (THREADS / WG_WAVE) + wave_id) * WPW;
  static_assert(MCO == 0 && ECL == 0 || (THREADS == 256 && WPW == 1), "fused merge / expand: 8 windows per workgroup");
  static_assert(ECL == 0 || 128 * C <= MAXU, "fused expand: the 8 windows' expanded tokens must fit one weight buffer");
  static_assert(MCO == 0 || 128 * C <= 2 * MAXU, "fused merge: the 8 windows' output tokens must fit the weight buffers");
  constexpr int TAIL = ((L::SIZE + 255) / 256) * 256;  // merge / expand weights follow the block (FORMATS.md)

  WG_PROF_BEGIN(wg_block_id() * (THREADS / WG_WAVE) + (tid >> 5))
  if constexpr (RESIDENT) {  // the whole stream into LDS once
    for (int i = (int)tid; i < RES / 16; i += THREADS) ((v4u*)lds)[i] = ((const v4u*)(w + L::STREAM))[i];
    wg_barrier_lds();
  } else {
    stage.template fetch<L::HEAD_BYTES>(w + L::head(0), tid);  // unit 0 is always loaded
  }
  for (;;) {  // resident: one window per pass; otherwise a single pass
  if constexpr (RESIDENT) { if (win0 >= nwin) break; }
  // re-read the header vectors (norm gains, biases) per window: otherwise they are hoisted out of the window loop and
  // held in registers (235 VGPRs instead of about 165, half the occupancy)
  if constexpr (RESIDENT) __asm__ volatile("" : "+s"(w));

  // Gather: token t of the window, mirrored at the borders.
  int dst[WPW];
  int eslot[WPW];  // fused expand: LDS record of this token
  v2i xa[WPW][KC], xb[WPW][ADD ? KC : 1];
  v8f xf[WPW][SRC::kCustom ? KC : 1];
  if constexpr (ECL > 0) fused_expand<C, ECL>(x, w + TAIL, lds + MAXU, H, W, nwx, nwin, ox, oy, tid, (int)wg_block_id());
#pragma unroll
  for (int i = 0; i < WPW; ++i) {
    const int win = win0 + i < nwin ? win0 + i : nwin - 1;
    const int wy = win / nwx, wx = win - wy * nwx;
    const int pr = 4 * wy - oy + (int)(t >> 2), pc = 4 * wx - ox + (int)(t & 3u);
    const bool real = win0 + i < nwin && pr >= 0 && pr < H && pc >= 0 && pc < W;
    dst[i] = real ? pr * W + pc : -1;
    const int mr = wg_mirror(pr, H), mc = wg_mirror(pc, W);
    const size_t src = ((size_t)(mr * W + mc) * C) + hh * HALF;
    if constexpr (ECL > 0) {
      const int low = ((mr >> 1) - wg_low0(4 * wy - oy, H)) * 2 + ((mc >> 1) - wg_low0(4 * wx - ox, W));
      eslot[i] = (int)(tid >> 5) * 16 + low * 4 + (((mr & 1) << 1) | (mc & 1));
    } else {
      eslot[i] = 0;
    }
#if defined(WG_ABLATE_LOAD) && WG_ABLATE_LOAD
    (void)src;  // timing-only: register data instead of global loads
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      xa[i][k][0] = (int)(tid * 0x01010101u + (u32)k * 0x08080808u + 0x30303030u);
      xa[i][k][1] = (int)(tid * 0x03030303u + (u32)k * 0x04040404u + 0x32323232u);
      if (ADD) xb[i][k] = xa[i][k];
    }
#else
    if constexpr (SRC::kCustom) {
      SRC::template load<C, KC>(x, pr, pc, H, W, hh, xf[i]);
    } else {
      if constexpr (ECL == 0) load_frags<KC>(x + src, xa[i]);
      if (ADD) load_frags<KC>(xadd + src, xb[i]);
    }
#endif
  }

  if constexpr (!RESIDENT) {
    stage.template commit<L::HEAD_BYTES>(lds, tid);
    wg_barrier_lds();
  }
  if constexpr (ECL > 0) {  // the expanded tokens are in LDS now
#pragma unroll
    for (int i = 0; i < WPW; ++i) load_frags<KC>(lds + MAXU + (size_t)eslot[i] * C + hh * HALF, xa[i]);
    wg_barrier_lds();  // they are read before the weight buffer they live in is refilled
  }
  int buf = 0;
  WG_PHASE(0, xa[0][0][0]);

#if defined(WG_ABLATE_FLOOR) && WG_ABLATE_FLOOR
  // timing-only: nothing between the gather and the store (the memory / launch floor of the kernel)
  v8f acc[WPW][KC];
#pragma unroll
  for (int i = 0; i < WPW; ++i)
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      acc[i][k] = wg_unpack_fp8(xa[i][k]);
      if (ADD) acc[i][k] += wg_unpack_fp8(xb[i][k]);
    }
  (void)buf;
#else
  // Steps 1-2: a = q8(RMSNorm(x; gamma1)). The accumulator for step 7, x1 = x + Wp q8(o) + bp, starts as x + bp;
  // each head adds its slice of Wp q8(o) as soon as its o is known (only one head's o is ever live).
  v2i af[WPW][KC];
  v8f acc[WPW][KC];
  {
    const float* g1 = (const float*)(w + L::G1) + hh * HALF;
    const float* bp = (const float*)(w + L::BP) + hh * 8;
#pragma unroll
    for (int i = 0; i < WPW; ++i) {
      float ss = 0.0f;
#pragma unroll
      for (int k = 0; k < KC; ++k) {
        if constexpr (SRC::kCustom) acc[i][k] = xf[i][k];
        else acc[i][k] = wg_unpack_fp8(xa[i][k]);
        if (ADD) acc[i][k] += wg_unpack_fp8(xb[i][k]);
        if (!(WG_NORM_DOT4 && !ADD && !SRC::kCustom)) ss += hsum8(acc[i][k] * acc[i][k]);
      }
      if (WG_NORM_DOT4 && !ADD && !SRC::kCustom) {
#pragma unroll
        for (int k = 0; k < KC; ++k) {
          ss = wg_dot4_fp8(xa[i][k][0], xa[i][k][0], ss);
          ss = wg_dot4_fp8(xa[i][k][1], xa[i][k][1], ss);
        }
      }
#if defined(WG_ABLATE_NORM) && WG_ABLATE_NORM
      const float r = 1.0f + 0.0f * ss;
#else
      ss += wg_xor16(ss);
      const float r = wg_rsq((ss + kNormEps) * (1.0f / C));
#endif
#pragma unroll
      for (int k = 0; k < KC; ++k) {
        af[i][k] = pack_in<SAFE>(acc[i][k] * r * wg_load8f(g1 + 8 * k));
        acc[i][k] += WG_BIAS8(bp + k * 16);
      }
    }
  }

  WG_PHASE(1, af[0][0][0]);

  // Steps 3-7, one head at a time; one staged unit per head (QKV fragments, then the head's slice of Wp).
#pragma unroll 1
  for (int hd = 0; hd < HEADS; ++hd) {
    const bool last = hd + 1 == HEADS;
    if constexpr (!RESIDENT) {
      WG_STAGING(if (last) stage.template fetch<L::MLP_BYTES>(w + L::mlp(0), tid);
                 else stage.template fetch<L::HEAD_BYTES>(w + L::head(hd + 1), tid);)
    }
    const u8* U = RESIDENT ? lds + hd * L::HEAD_BYTES : lds + buf * MAXU;
    // u = q8(Wqk a): result rows are this head's 32 channels in two tiles, columns are tokens.
    v2i uf[WPW][2];
    // V^T = a^T Wv^T: rows are tokens (key index s), columns are the 32 value channels in two tiles.
#if WG_AV8
    v2i vt[WPW][2];
#else
    v8h vt[WPW][2];
#endif
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      v8f accq[WPW], accv[WPW];
#pragma unroll
      for (int i = 0; i < WPW; ++i) accq[i] = accv[i] = wg_splat(0.0f);
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
        const v2i fq = lfrag(U, (mt * KC + kc) * 2, lane);
        const v2i fv = lfrag(U, (mt * KC + kc) * 2 + 1, lane);
#pragma unroll
        for (int i = 0; i < WPW; ++i) {
#if defined(WG_ABLATE_QKV) && WG_ABLATE_QKV
          accq[i][kc & 7] += (float)(fq[0] ^ af[i][kc][0]);  // timing-only stand-in: keeps fq, fv and af alive
          accv[i][kc & 7] += (float)(fv[1] ^ af[i][kc][1]);
#else
          accq[i] = wg_wmma_fp8(fq, af[i][kc], accq[i]);
          accv[i] = wg_wmma_fp8(af[i][kc], fv, accv[i]);
#endif
        }
      }
#pragma unroll
      for (int i = 0; i < WPW; ++i) {
        uf[i][mt] = pack_in<SAFE>(accq[i]);
#if WG_AV8
        vt[i][mt] = pack_in<SAFE>(accv[i]);
#else
        vt[i][mt] = wg_to_f16(accv[i]);
#endif
      }
    }
    WG_PHASE(2, uf[0][1][0]);
#if defined(WG_ABLATE_ATTN) && WG_ABLATE_ATTN
    v2i of[WPW][2];
#pragma unroll
    for (int i = 0; i < WPW; ++i) { of[i][0] = uf[i][0]; of[i][1] = uf[i][1]; (void)vt[i][0]; (void)vt[i][1]; }
    (void)t;
#else
    // Lane (t, h) holds logits S(t, s) for the 8 keys s = drow(h, v); bias is packed in the same order.
    const v8f bias = wg_load8f((const float*)(w + L::BIAS) + ((hd * 16 + (int)t) * 2 + (int)hh) * 8);
    v2i of[WPW][2];
#pragma unroll
    for (int i = 0; i < WPW; ++i) {
      v8f s = wg_wmma_fp8(uf[i][0], uf[i][0], wg_splat(0.0f));  // S = U^T U is symmetric
      s = wg_wmma_fp8(uf[i][1], uf[i][1], s);
#if WG_SMAX16 && !WG_AV8 && !(defined(WG_ABLATE_CAP) && WG_ABLATE_CAP) && !(defined(WG_ABLATE_SMAX) && WG_ABLATE_SMAX)
      // soft cap and softmax in packed FP16: z = tau log2e (c - c^3 / (3 cap^2)), c = clamp(S + B, -cap, cap)
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
#if defined(WG_ABLATE_CAP) && WG_ABLATE_CAP
        z[v] = s[v] * (kTau * kLog2e) + 0.0f * bias[v];
#else
        const float c = wg_fmin(wg_fmax(s[v] + bias[v], -kCap), kCap);
        z[v] = (c - c * c * c * (1.0f / (3.0f * kCap * kCap))) * (kTau * kLog2e);
#endif
      }
#if defined(WG_ABLATE_SMAX) && WG_ABLATE_SMAX
      v8f e;
#pragma unroll
      for (int v = 0; v < 8; ++v) e[v] = wg_exp2(z[v]);
      const v8h p = wg_to_f16(e);
#else
      float mx = hmax8(z);
      mx = wg_fmax(mx, wg_xor16(mx));
      v8f e;
#pragma unroll
      for (int v = 0; v < 8; ++v) e[v] = wg_exp2(z[v] - mx);
      float sum = hsum8(e);
      sum += wg_xor16(sum);
#if WG_AV8
      const v2i p = wg_pack_q8_nc(e * wg_rcp(sum));  // probabilities in [0, 1]: no clamp needed
#else
      const v8h p = wg_to_f16(e * wg_rcp(sum));
#endif
#endif
#endif
      // o = A V: rows are value channels, columns are tokens; K = keys, paired by drow on both sides.
#pragma unroll
#if WG_AV8
      for (int dt = 0; dt < 2; ++dt) of[i][dt] = pack_in<SAFE>(wg_wmma_fp8(vt[i][dt], p, wg_splat(0.0f)));
#else
      for (int dt = 0; dt < 2; ++dt) of[i][dt] = pack_in<SAFE>(wg_wmma_f16(vt[i][dt], p, wg_splat(0.0f)));
#endif
    }
#endif
    WG_PHASE(3, of[0][1][0]);
    // x1 += Wp[:, head hd] q8(o_hd)
#pragma unroll
    for (int mt = 0; mt < KC; ++mt) {
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        const v2i f = lfrag(U, L::QKV_FRAGS + mt * 2 + dt, lane);
#pragma unroll
        for (int i = 0; i < WPW; ++i) {
#if defined(WG_ABLATE_WP) && WG_ABLATE_WP
          acc[i][mt][dt] += (float)(f[0] ^ of[i][dt][0] ^ of[i][dt][1]);  // timing-only stand-in
#else
          acc[i][mt] = wg_wmma_fp8(f, of[i][dt], acc[i][mt]);
#endif
        }
      }
    }
    WG_PHASE(4, acc[0][KC - 1]);
    if constexpr (!RESIDENT) {
      WG_STAGING(if (last) stage.template commit<L::MLP_BYTES>(lds + (buf ^ 1) * MAXU, tid);
                 else stage.template commit<L::HEAD_BYTES>(lds + (buf ^ 1) * MAXU, tid);
                 wg_barrier_lds(); buf ^= 1;)
    }
    WG_PHASE(5, 0);
  }

  // Step 8: m = q8(RMSNorm(x1; gamma2)). Then add b2 to the residual accumulator.
  v2i mf[WPW][KC];
  {
    const float* g2 = (const float*)(w + L::G2) + hh * HALF;
    const float* b2 = (const float*)(w + L::B2) + hh * 8;
#pragma unroll
    for (int i = 0; i < WPW; ++i) {
      float ss = 0.0f;
#pragma unroll
      for (int k = 0; k < KC; ++k) ss += hsum8(acc[i][k] * acc[i][k]);
#if defined(WG_ABLATE_NORM) && WG_ABLATE_NORM
      const float r = 1.0f + 0.0f * ss;
#else
      ss += wg_xor16(ss);
      const float r = wg_rsq((ss + kNormEps) * (1.0f / C));
#endif
#pragma unroll
      for (int k = 0; k < KC; ++k) {
        mf[i][k] = pack_in<SAFE>(acc[i][k] * r * wg_load8f(g2 + 8 * k));
        acc[i][k] += WG_BIAS8(b2 + k * 16);
      }
    }
  }

  WG_PHASE(6, mf[0][KC - 1][0]);

  // Steps 9-10: x2 = x1 + W2 q8(GELU(W1 m + b1)) + b2, one 16-unit slice of the hidden layer at a time.
  const float* b1 = (const float*)(w + L::B1) + hh * 8;
#if defined(WG_ABLATE_MLP) && WG_ABLATE_MLP
  (void)b1;
  for (int i = 0; i < WPW; ++i)
    for (int k = 0; k < KC; ++k) acc[i][k] += wg_unpack_fp8(mf[i][k]);
#else
  static_assert(L::HPU % GROUP == 0, "MLP groups must tile the hidden tiles of a unit");
#pragma unroll 1
  for (int u = 0; u < L::NMLP; ++u) {
    const bool more = u + 1 < L::NMLP;
    if constexpr (!RESIDENT) {
      WG_STAGING(if (more) stage.template fetch<L::MLP_BYTES>(w + L::mlp(u + 1), tid);)
    }
    const u8* U = RESIDENT ? lds + HEADS * L::HEAD_BYTES + u * L::MLP_BYTES : lds + buf * MAXU;
#if WG_MLP_PF >= 2
    v2i f1[GROUP][KC];  // W1 fragments of the next tile group (filled by the previous iteration, except for the first)
#endif
#pragma unroll 1
    for (int j0 = 0; j0 < L::HPU; j0 += GROUP) {
#if WG_MLP_PF >= 2
      if (j0 == 0) {
#pragma unroll
        for (int g = 0; g < GROUP; ++g)
#pragma unroll
          for (int kc = 0; kc < KC; ++kc) f1[g][kc] = WG_MLP_FRAG(U, g * 2 * KC + kc, lane);
      }
#endif
      // W1 for GROUP hidden tiles (16 hidden units x 16 tokens per window each): hacc = b1 + W1[tile] m.
      v8f hacc[GROUP][WPW];
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        const v8f b = WG_BIAS8(b1 + (u * L::HPU + j0 + g) * 16);
#pragma unroll
        for (int i = 0; i < WPW; ++i) hacc[g][i] = b;
      }
#if !(defined(WG_ABLATE_W1) && WG_ABLATE_W1)
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
#if WG_MLP_PF >= 2
          const v2i f = f1[g][kc];
#else
          const v2i f = WG_MLP_FRAG(U, (j0 + g) * 2 * KC + kc, lane);
#endif
#pragma unroll
          for (int i = 0; i < WPW; ++i) hacc[g][i] = wg_wmma_fp8(f, mf[i][kc], hacc[g][i]);
        }
      }
#endif
      WG_PHASE(7, hacc[GROUP - 1][0]);
#if WG_MLP_PF >= 1
      v2i f2[GROUP][KC];  // this tile's W2 fragments, loaded now and used after the GELU
#pragma unroll
      for (int g = 0; g < GROUP; ++g)
#pragma unroll
        for (int mt = 0; mt < KC; ++mt) f2[g][mt] = WG_MLP_FRAG(U, (j0 + g) * 2 * KC + KC + mt, lane);
      __asm__ volatile("" ::: "memory");  // no load moves across this (IR level) ...
      WG_SCHED_FENCE();  // ... and no instruction at all (machine scheduler): the loads stay ahead of the GELU
#endif
      // GELU + q8.
      v2i hf[GROUP][WPW];
#pragma unroll
      for (int g = 0; g < GROUP; ++g)
#pragma unroll
        for (int i = 0; i < WPW; ++i) {
#if (WG_GELU_MODE == 3 || WG_GELU_MODE == 4) && !(defined(WG_ABLATE_PACK) && WG_ABLATE_PACK) && !(defined(WG_ABLATE_GELU) && WG_ABLATE_GELU)
          hf[g][i] = wg_hidden_mix<SAFE, (WG_GELU_MODE == 3 ? 4 : WG_GELU_MIX)>(hacc[g][i]);
          continue;
#endif
          v8f ge;
#pragma unroll
          for (int v = 0; v < 8; ++v) ge[v] = wg_gelu(hacc[g][i][v]);
#if defined(WG_ABLATE_PACK) && WG_ABLATE_PACK
          v2i r;  // timing-only: 6 float adds instead of the clamps and FP8 conversions; every value stays alive
          r[0] = __builtin_bit_cast(int, (ge[0] + ge[1]) + (ge[2] + ge[3]));
          r[1] = __builtin_bit_cast(int, (ge[4] + ge[5]) + (ge[6] + ge[7]));
          hf[g][i] = r;
#else
          hf[g][i] = SAFE ? wg_pack_q8_nc(ge) : wg_pack_q8_inner(ge);
#endif
        }
      WG_PHASE(8, hf[GROUP - 1][0][0]);
      // W2: acc += W2[:, tile] q8(GELU(hacc)), tiles of the group in order.
#if defined(WG_ABLATE_W2) && WG_ABLATE_W2
#pragma unroll
      for (int g = 0; g < GROUP; ++g)
#pragma unroll
        for (int i = 0; i < WPW; ++i) acc[i][0][0] += (float)(hf[g][i][0] ^ hf[g][i][1]);  // keeps ALL of hf alive
#else
#if WG_MLP_PF >= 2
      if (j0 + GROUP < L::HPU) {  // the next tile group's W1 fragments, behind the W2 WMMAs of this one
#pragma unroll
        for (int g = 0; g < GROUP; ++g)
#pragma unroll
          for (int kc = 0; kc < KC; ++kc) f1[g][kc] = WG_MLP_FRAG(U, (j0 + GROUP + g) * 2 * KC + kc, lane);
      }
      __asm__ volatile("" ::: "memory");
      WG_SCHED_FENCE();
#endif
#pragma unroll
      for (int mt = 0; mt < KC; ++mt) {
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
#if WG_MLP_PF >= 1
          const v2i f = f2[g][mt];
#else
          const v2i f = WG_MLP_FRAG(U, (j0 + g) * 2 * KC + KC + mt, lane);
#endif
#pragma unroll
          for (int i = 0; i < WPW; ++i) acc[i][mt] = wg_wmma_fp8(f, hf[g][i], acc[i][mt]);
        }
      }
#endif
      WG_PHASE(9, acc[0][KC - 1]);
    }
    if constexpr (!RESIDENT) {
      WG_STAGING(if (more) { stage.template commit<L::MLP_BYTES>(lds + (buf ^ 1) * MAXU, tid); wg_barrier_lds(); buf ^= 1; })
    }
    WG_PHASE(10, 0);
  }
#endif

#endif  // WG_ABLATE_FLOOR

  if constexpr (EPI::kFused) {
    EPI::template run<C, KC, WPW, THREADS>(acc, dst, W, lds, 2 * MAXU, tid, epi);
  } else {
  // Output q8(x2), real window members only.
#pragma unroll
  for (int i = 0; i < WPW; ++i) {
    if (dst[i] < 0) continue;
    v2i o[KC];
#pragma unroll
    for (int k = 0; k < KC; ++k) o[k] = wg_pack_q8(acc[i][k]);
#if defined(WG_ABLATE_STORE) && WG_ABLATE_STORE
    int chk = 0;  // timing-only: store only if an impossible pattern shows up, so the packed values stay alive
#pragma unroll
    for (int k = 0; k < KC; ++k) chk ^= o[k][0] ^ o[k][1];
    if (chk == (int)0xdeadbeefu) store_frags<KC>(out + (size_t)dst[i] * C + hh * HALF, o);
#else
    store_frags<KC>(out + (size_t)dst[i] * C + hh * HALF, o);
#endif
  }
  if constexpr (MCO > 0) {
    wg_barrier_lds();  // every wave is done with the weight buffers
    {
      v2i o[KC];
#pragma unroll
      for (int k = 0; k < KC; ++k) o[k] = wg_pack_q8(acc[0][k]);
      store_frags<KC>(lds + (size_t)(tid >> 5) * 16 * C + t * C + hh * HALF, o);
    }
    wg_barrier_lds();
    fused_merge<C, MCO>(lds, w + TAIL, aux, H, W, nwx, nwin, ox, oy, tid, (int)wg_block_id());
  }
  }
  WG_PHASE(11, 0);
  if constexpr (!RESIDENT) break;
  win0 += nwg * (THREADS / WG_WAVE);
  }
  WG_PROF_END();
}

// Copy `bytes` (multiple of 16) from global memory to LDS with the whole workgroup, then synchronize.
template <int THREADS>
WG_FN void copy_to_lds(u8* dst, const u8* src, int bytes, u32 tid) {
  for (int i = (int)tid; i < bytes / 16; i += THREADS) ((v4u*)dst)[i] = ((const v4u*)src)[i];
  wg_barrier_lds();
}

constexpr int kGemmThreads = WG_GEMM_THREADS;  // patch merge / expand workgroup: waves share the weights staged in LDS

// Patch merge (spec 02): merged(i, j) = q8(Wm concat(TL, TR, BL, BR) + bm) from the q8 block output s [H][W][C].
// Lane half h loads the 2C contiguous bytes of row 2i + h (left and right token), i.e. z[h*2C, (h+1)*2C).
// NT groups of 16 merged tokens per wave; Wm is staged in LDS once per workgroup.
template <int C, int CO, int NT>
WG_FN void patch_merge(const u8* __restrict__ s, u8* __restrict__ out, const u8* __restrict__ w, int H, int W) {
  constexpr int KZ = C / 4, MO = CO / 16;  // 4C / 16 input groups, CO / 16 output tiles
  constexpr int BM = 0, WM = 4 * CO, WBYTES = MO * KZ * 256;
  WG_LDS __attribute__((aligned(16))) u8 lds[WBYTES];
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int Wm = W >> 1, ntok = (H >> 1) * Wm;
  const int tok0 = (int)(wg_block_id() * (kGemmThreads / WG_WAVE) + (tid >> 5)) * NT * 16 + (int)t;
  v2i zf[NT][KZ];
  int tok[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    tok[i] = tok0 + 16 * i;
    const int tk = tok[i] < ntok ? tok[i] : ntok - 1;
    const int r = tk / Wm, c = tk - r * Wm;
    load_frags<KZ>(s + (size_t)((2 * r + (int)hh) * W + 2 * c) * C, zf[i]);
  }
  copy_to_lds<kGemmThreads>(lds, w + WM, WBYTES, tid);
  const float* bm = (const float*)(w + BM) + hh * 8;
  // One output tile (8 channels per lane) at a time, stored right away: keeps register use low.
#pragma unroll 1
  for (int mt = 0; mt < MO; ++mt) {
    v8f acc[NT];
    const v8f b = wg_load8f(bm + mt * 16);
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i] = b;
#pragma unroll
    for (int g = 0; g < KZ; ++g) {
      const v2i f = lfrag(lds, mt * KZ + g, lane);
#pragma unroll
      for (int i = 0; i < NT; ++i) acc[i] = wg_wmma_fp8(f, zf[i][g], acc[i]);
    }
#pragma unroll
    for (int i = 0; i < NT; ++i)
      if (tok[i] < ntok) *(v2i*)(out + (size_t)tok[i] * CO + hh * (CO / 2) + 8 * mt) = wg_pack_q8(acc[i]);
  }
}

// Expand (spec 02): e = q8(We y + be) for each low-resolution token, written as an FP8 map e [2Hl][2Wl][C]
// (sub-pixel k goes to (2r + k/2, 2c + k%2)). The decoder block then adds the skip map (ADD = true).
template <int CL, int C, int NT>
WG_FN void expand(const u8* __restrict__ y, u8* __restrict__ e, const u8* __restrict__ w, int Hl, int Wl) {
  constexpr int KC = CL / 16, G = C / 16;
  constexpr int BE = 0, WE = 16 * C, WBYTES = 4 * G * KC * 256;
  WG_LDS __attribute__((aligned(16))) u8 lds[WBYTES];
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
  const int ntok = Hl * Wl;
  const int tok0 = (int)(wg_block_id() * (kGemmThreads / WG_WAVE) + (tid >> 5)) * NT * 16 + (int)t;
  v2i yf[NT][KC];
  int tok[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    tok[i] = tok0 + 16 * i;
    const int tk = tok[i] < ntok ? tok[i] : ntok - 1;
    load_frags<KC>(y + (size_t)tk * CL + hh * (CL / 2), yf[i]);
  }
  copy_to_lds<kGemmThreads>(lds, w + WE, WBYTES, tid);
  const float* be = (const float*)(w + BE) + hh * 8;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    v2i res[NT][G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int mt = k * G + g;
      v8f acc[NT];
      const v8f b = wg_load8f(be + mt * 16);
#pragma unroll
      for (int i = 0; i < NT; ++i) acc[i] = b;
#pragma unroll
      for (int kc = 0; kc < KC; ++kc) {
        const v2i f = lfrag(lds, mt * KC + kc, lane);
#pragma unroll
        for (int i = 0; i < NT; ++i) acc[i] = wg_wmma_fp8(f, yf[i][kc], acc[i]);
      }
#pragma unroll
      for (int i = 0; i < NT; ++i) res[i][g] = wg_pack_q8(acc[i]);
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      if (tok[i] >= ntok) continue;
      const int r = tok[i] / Wl, c = tok[i] - r * Wl;
      const size_t dst = (size_t)((2 * r + (k >> 1)) * (2 * Wl) + 2 * c + (k & 1)) * C + hh * (C / 2);
      store_frags<G>(e + dst, res[i]);
    }
  }
}

}  // namespace

// WG_SWIN_NO_EXPORTS: another kernel file includes this one for the templates only (dec1_dec0.hip).
#if !defined(WG_SWIN_NO_EXPORTS)
// Exported kernels. Grid sizes: see clean/runner/stage_runner.py (Launch plan) and FORMATS.md.
// Every block kernel exists twice: `name` clamps every inner FP8 pack to +-448, `name_nc` ("no clamp") skips the clamp.
// The first word of the packed weights (bit 0) says whether the weights PROVE that no inner pack can exceed the
// range (packing.py: block_bounds); the runner launches `name_nc` then. Both give identical results where it is used.
#ifndef WG_NUMV_C96
#define WG_NUMV_C96 0  // hard VGPR cap of the C = 96 block kernels (0: none); 192 = 8 waves per SIMD
#endif
#if WG_NUMV_C96 > 0
#define WG_KERNEL_C96(name, threads, waves) WG_KERNEL_NWV(name, threads, waves, WG_NUMV_C96)
#else
#define WG_KERNEL_C96(name, threads, waves) WG_KERNEL_NW(name, threads, waves)
#endif
#define WG_BLOCK_KERNEL(name, C, HEADS, MLP, WPW, ADD, MINW, THREADS)                                        \
  WG_KERNEL_NW(name, THREADS, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,    \
                                    int oy WG_PROF_ARG_DECL) {                                                \
    swin_block<C, HEADS, MLP, WPW, ADD, WG_MLP_GROUP, false, THREADS>(x, xadd, out, w, H, W, ox, oy, nullptr WG_PROF_ARG); \
  }                                                                                                          \
  WG_KERNEL_NW(name##_nc, THREADS, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox, \
                                         int oy WG_PROF_ARG_DECL) {                                          \
    swin_block<C, HEADS, MLP, WPW, ADD, WG_MLP_GROUP, true, THREADS>(x, xadd, out, w, H, W, ox, oy, nullptr WG_PROF_ARG);  \
  }
#define WG_BLOCK_KERNEL96(name, C, HEADS, MLP, WPW, ADD, MINW, THREADS)                                        \
  WG_KERNEL_C96(name, THREADS, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,    \
                                    int oy WG_PROF_ARG_DECL) {                                                \
    swin_block<C, HEADS, MLP, WPW, ADD, WG_MLP_GROUP, false, THREADS>(x, xadd, out, w, H, W, ox, oy, nullptr WG_PROF_ARG); \
  }                                                                                                          \
  WG_KERNEL_C96(name##_nc, THREADS, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox, \
                                         int oy WG_PROF_ARG_DECL) {                                          \
    swin_block<C, HEADS, MLP, WPW, ADD, WG_MLP_GROUP, true, THREADS>(x, xadd, out, w, H, W, ox, oy, nullptr WG_PROF_ARG);  \
  }

// WG_SWIN_NO_BLOCK_KERNELS: compile only the shared patch-merge / expand kernels, none of the block kernels (plain, _nc,
// fused) (kernels/swin_v2.hip includes this file for them and exports its own block kernels under the same names).
#if !defined(WG_SWIN_NO_BLOCK_KERNELS)
WG_BLOCK_KERNEL(swin_block_c64, 64, 2, 256, WG_WPW_C64, false, WG_MINW_C64, WG_THREADS_C64)
WG_BLOCK_KERNEL(swin_block_c64_add, 64, 2, 256, WG_WPW_C64, true, WG_MINW_C64, WG_THREADS_C64ADD)
WG_BLOCK_KERNEL96(swin_block_c96, 96, 4, 384, WG_WPW_C96, false, WG_MINW_C96, WG_THREADS_C96)
WG_BLOCK_KERNEL96(swin_block_c96_add, 96, 4, 384, WG_WPW_C96, true, WG_MINW_C96, WG_THREADS_C96ADD)
WG_BLOCK_KERNEL(swin_block_c128, 128, 4, 512, WG_WPW_C128, false, WG_MINW_C128, WG_THREADS_C128)
// Weight-resident persistent C = 64 block kernels (`*_r`, RESIDENT): 512 lanes, the block's weight stream (about 45 KB)
// in LDS once per workgroup, two workgroups per WGP; launch nwg workgroups (about 2 per WGP), each wave runs windows
// wave_id, wave_id + 16 nwg, ... Bit-identical to swin_block_c64[_add][_nc].
#define WG_RESIDENT_KERNEL(name, C, HEADS, MLP, ADD, MINW)                                                           WG_KERNEL_NW(name##_r, 512, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,                                           int oy, int nwg WG_PROF_ARG_DECL) {                                                swin_block<C, HEADS, MLP, 1, ADD, WG_MLP_GROUP, false, 512, 0, 0, SwinStoreEpi, true>(                                 x, xadd, out, w, H, W, ox, oy, nullptr WG_PROF_ARG, nullptr, nwg);                                           }                                                                                                                  WG_KERNEL_NW(name##_r_nc, 512, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,                                           int oy, int nwg WG_PROF_ARG_DECL) {                                             swin_block<C, HEADS, MLP, 1, ADD, WG_MLP_GROUP, true, 512, 0, 0, SwinStoreEpi, true>(                                  x, xadd, out, w, H, W, ox, oy, nullptr WG_PROF_ARG, nullptr, nwg);                                           }
WG_RESIDENT_KERNEL(swin_block_c64, 64, 2, 256, false, WG_MINW_C64)
WG_RESIDENT_KERNEL(swin_block_c64_add, 64, 2, 256, true, WG_MINW_C64)

// Fused kernels (runner geometry "fuse", variant `fuse`): the encoder blocks also do their patch merge (`_mg`, merged
// map in the extra argument), the decoder blocks also do their expand (`_ex`, x is the low-resolution map, the extra
// argument is unused). 256 lanes, one window per wave.
#define WG_FUSED_KERNEL(name, C, HEADS, MLP, ADD, MCO, ECL, MINW)                                                   \
  WG_KERNEL_NW(name, 256, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox, int oy,   \
                                u8* aux WG_PROF_ARG_DECL) {                                                         \
    swin_block<C, HEADS, MLP, 1, ADD, WG_MLP_GROUP, false, 256, MCO, ECL>(x, xadd, out, w, H, W, ox, oy, aux WG_PROF_ARG); \
  }                                                                                                                 \
  WG_KERNEL_NW(name##_nc, 256, MINW)(const u8* x, const u8* xadd, u8* out, const u8* w, int H, int W, int ox,      \
                                     int oy, u8* aux WG_PROF_ARG_DECL) {                                            \
    swin_block<C, HEADS, MLP, 1, ADD, WG_MLP_GROUP, true, 256, MCO, ECL>(x, xadd, out, w, H, W, ox, oy, aux WG_PROF_ARG); \
  }

WG_FUSED_KERNEL(swin_block_c64_mg, 64, 2, 256, false, 96, 0, WG_MINW_C64)
WG_FUSED_KERNEL(swin_block_c96_mg, 96, 4, 384, false, 128, 0, WG_MINW_C96MG)
WG_FUSED_KERNEL(swin_block_c64_add_ex, 64, 2, 256, true, 0, 96, WG_MINW_C64)
WG_FUSED_KERNEL(swin_block_c96_add_ex, 96, 4, 384, true, 0, 128, WG_MINW_C96EX)
#endif  // WG_SWIN_NO_BLOCK_KERNELS

WG_KERNEL_N(patch_merge_c64, WG_GEMM_THREADS)(const u8* s, u8* out, const u8* w, int H, int W) { patch_merge<64, 96, WG_NT>(s, out, w, H, W); }
WG_KERNEL_N(patch_merge_c96, WG_GEMM_THREADS)(const u8* s, u8* out, const u8* w, int H, int W) { patch_merge<96, 128, WG_NT>(s, out, w, H, W); }

WG_KERNEL_N(expand_c128, WG_GEMM_THREADS)(const u8* y, u8* e, const u8* w, int Hl, int Wl) { expand<128, 96, WG_NT>(y, e, w, Hl, Wl); }
WG_KERNEL_N(expand_c96, WG_GEMM_THREADS)(const u8* y, u8* e, const u8* w, int Hl, int Wl) { expand<96, 64, WG_NT>(y, e, w, Hl, Wl); }

#if defined(WG_EMU)
#if !defined(WG_SWIN_NO_BLOCK_KERNELS)
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
WG_EMU_REGISTER(swin_block_c64_mg)
WG_EMU_REGISTER(swin_block_c64_mg_nc)
WG_EMU_REGISTER(swin_block_c96_mg)
WG_EMU_REGISTER(swin_block_c96_mg_nc)
WG_EMU_REGISTER(swin_block_c64_add_ex)
WG_EMU_REGISTER(swin_block_c64_add_ex_nc)
WG_EMU_REGISTER(swin_block_c96_add_ex)
WG_EMU_REGISTER(swin_block_c96_add_ex_nc)
#endif
WG_EMU_REGISTER(patch_merge_c64)
WG_EMU_REGISTER(patch_merge_c96)
WG_EMU_REGISTER(expand_c128)
WG_EMU_REGISTER(expand_c96)
#endif
#endif  // !WG_SWIN_NO_EXPORTS
