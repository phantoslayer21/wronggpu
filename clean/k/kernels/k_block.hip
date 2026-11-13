// SPDX-License-Identifier: MIT
// Model K block of spec 41 for gfx1201 (task K5): the block, and whole stages with the block's neighbours fused in.
//
// Kernels: k_block_c<C>_h<H>_<mix> (the block on its input map, mirror border rule), k_blockp_... (the block on a
// decoder stage's padded input map, written by k_expandp_*), and k_stage_<name>: one launch per stage with the
// operations around the block fused in (spec 44 gives every merging and every decoder stage an even window offset, so
// a window holds whole 2x2 merge groups and exactly 4 x 4 coarse tokens):
//   e0      embedding (16 -> 32, ReLU) of each window member on load, merge in the epilogue
//   e1..e4  merge in the epilogue: the block output goes to LDS, and the window's 16 groups are one WMMA column set
//   d4..d0  expand + skip on load: the window's 4 x 4 coarse tokens (mirrored at the coarse level) are one WMMA column
//           set, expanded to 4C channels into LDS, where every member takes its sub-position's C channels and adds the
//           skip map (mirrored at this level): exactly spec 41 draft 3's border rule
//   d0      also the output head (32 -> 40) in the epilogue
// The stage kernels write the block output only when P.y is set (d0's is not needed by the network).
//
// One 128-lane workgroup (4 waves) processes one 8x8 window. Wave w owns the window-local tokens 16 w .. 16 w + 15
// (window rows 2w, 2w + 1) as the 16 WMMA columns; lane l holds token 16 w + l % 16 and, in every fragment, the 8
// channels 16 kt + 8 (l / 16) + 0..7 of it (k_common.h). Steps (spec 41 "One block"):
//
//  1. Load the window's tokens and normalize: s = f16(sum_c f16(x_c^2)) (FP32 sum of the rounded squares: the two
//     lane halves' partial sums meet in one lane exchange), r = f16(1/sqrt(s)), n_c = f16(x_c f16(r g1_c)); s = inf
//     gives r = 0, and s = 0 gives r = 0 as well (spec 41 draft 2: n = 0 there).
//     Encoder and middle stages read their input map through the mirror border rule (spec 41 "Windows"). Decoder
//     stages read a padded map that already holds every window member (spec 41 draft 3 border rule, written by the
//     padded expand of k_linear.hip): member (row, col) of window (a, b) at padded position (8b + row, 8a + col).
//  2. Per head h (heads in sequence, all four waves together):
//     - q (softmax only) = f16(Wq n) stays in registers: lane = token, 8 channels = the B operand of the scores.
//     - k = f16(Wk n) is written to LDS token-major, K[s][c] (the A operand of the scores: row = key token).
//     - v is computed the other way round, v^T = n^T Wv^T: A = the token fragments themselves (an A fragment and a B
//       fragment have the same register layout), B = the Wv tiles; the result has lane = channel, 8 consecutive
//       tokens, and is written to LDS channel-major, VT[c][s] (the A operand of z: row = channel).
//     - barrier
//     - softmax: scores^T = K q (+ B[h][t][s] as the accumulator start, so f16(q.k + B) rounds once): lane = query t,
//       elements = 8 keys, which is directly the B operand of z. u = clamp(S, +-c), g = f16(u f16(1 - f16(kappa
//       f16(u u)))), w = 2^g (FP32), A = f16(w (1 / sum w)) (FP32 reciprocal; the sum over 64 keys
//       is 32 per lane plus one lane exchange). fixed: A = M[h][t][s] loaded directly in the same layout.
//     - z_h^T = VT A^T, rounded: lane = query token, 8 channels of head h = the B operand of the output projection.
//     - y += Wo[:, 32 h .. 32 h + 32] z_h (FP32 accumulators started from bo, rounded once after the last head).
//     K and VT are double-buffered by head parity (one barrier per head) where that keeps the workgroups per WGP
//     (d1 .. d4 and the G = 2 stages: one buffer, a second barrier per head). Both are stored tile by tile in fragment
//     order (no padding, conflict-free); an encoder's merge staging shares their space (see "LDS" in k_block).
//  3. x1 = f16(x + f16(Wo z + bo)) (x re-read from memory, or from LDS when the stage built it: fewer live registers).
//  4. MLP: a = f16(x1 g2); per hidden tile of 16: p = f16(W1 a + b1) (accumulator started from b1), h = act(p), and
//     h is at once the B operand of W2 (FP32 accumulators started from b2); x2 = f16(x1 + f16(W2 h + b2)).
//  5. Store x2 for the window's real positions; then the merge or the head.
//
// The steps above describe the FP32-accumulating kernels (k_block_*, k_blockp_*, k_stage_*, k_stage2_*: spec 41 as
// written; the network's default). The FP16-accumulating kernels k_stageh_* and k_stage2h_* (a spec 49 speed
// approximation, opt-in: network variant "fused16", stage suffix "@f16") run the product classes of the mask WGK_F16ACC (k_common.h KF_*) on v_wmma_f16_16x16x16_f16 instead:
// template parameter F16. m has its own stage kernel only in that form (k_stageh_m; FP32: the plain block).
//
// Weight blob (clean/k/FORMATS.md "Block blob", clean/k/runner/k_pack.py pack_block): the section pointers come in
// KBlkParams. The v-tiles are used as B operands (swapped orientation), which needs the gfx12 accumulator layout
// (d_row rows v + 8 h, measured on the RX 9070 XT); the packer refuses other layouts for blocks.
// Cross-lane handoffs in this file use LDS only. Global inputs/weights are read-only
// during a launch; global outputs are consumed by later stream-ordered kernels.
// LDS-scoped barriers preserve those handoffs without invalidating global caches.
// FP32 <-> FP16 conversions of the accumulators two per dword (k_common.h wgk_round8 / wgk_widen8; same bits). Not in
// k_block_wide.hip: there the wide e3 / e4 kernels spill with it and were slower.
#ifndef WGK_PAIR_CVT
#define WGK_PAIR_CVT 1
#endif
#include "k_common.h"
#include "k_norm.h"
#include "k_fp.h"

// Spec 43 section 6.8: same ordered sum and final half bits as the original.
WG_FN float wgk_hidden_norm2(const float h[4]) {
#pragma clang fp contract(off)
  float ss = h[1] * h[1];
  ss = kf_fma(h[0], h[0], ss);
  ss = kf_fma(h[2], h[2], ss);
  ss = kf_fma(h[3], h[3], ss);
  return ss + 1.0f;
}

WG_FN void wgk_store_hidden_exact(h16* dst, const float h[4], float x) {
#pragma clang fp contract(off)
  const float nu = kf_rsqrt(x);
#pragma unroll
  for (int k = 0; k < 4; ++k) dst[k] = kh_round(h[k] * nu);
}

WG_FN bool wgk_hidden_guard_arch_enabled() {
#if defined(__AMDGCN__) && defined(__gfx1201__)
  return true;
#else
  return false;
#endif
}

WG_FN bool wgk_hidden_half_safe(float p) {
  const u32 a = __builtin_bit_cast(u32, p) & 0x7fffffffu;
  const u32 low = a & 0x1fffu;
  return a >= 0x38800000u && a <= 0x477fe000u &&
         (low < 0x0ff8u || low > 0x1008u);
}

// Return the branch actually used, allowing the production-source audit to
// verify both paths. The normal caller discards this compile-time-inlined value.
WG_FN bool wgk_store_hidden_guarded(h16* dst, const float h[4]) {
#pragma clang fp contract(off)
  const float x = wgk_hidden_norm2(h);
#if defined(__AMDGCN__) && defined(__gfx1201__)
  // Exhaustive gfx1201 bound: hardware rsq differs from kf_rsqrt by at most
  // one adjacent positive float for every x in [1,2^34]. Rounded products
  // then differ by at most seven float steps. Accept normal-half products
  // only when more than eight steps from the half-rounding midpoint.
  if (x >= 1.0f && x <= 17179869184.0f) {
    const float estimate = __builtin_amdgcn_rsqf(x);
    float product[4];
    bool safe = true;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      product[k] = h[k] * estimate;
      safe = safe && wgk_hidden_half_safe(product[k]);
    }
    if (safe) {
#pragma unroll
      for (int k = 0; k < 4; ++k) dst[k] = kh_round(product[k]);
      return true;
    }
  }
#endif
  // Preserve tiny/zero values, midpoint neighborhoods, exceptional values,
  // and other architectures through the original normalization of all four.
  wgk_store_hidden_exact(dst, h, x);
  return false;
}

WG_FN void wgk_store_hidden(h16* dst, const float h[4]) {
  (void)wgk_store_hidden_guarded(dst, h);
}

enum { KMIX_FIXED = 0, KMIX_SOFTMAX = 1 };
enum { KIN_MAP = 0, KIN_PADDED = 1, KIN_EMBED = 2, KIN_EXPAND = 3 };

#define WGK_SOFTMAX_C ((h16)21.640625f)
#define WGK_SOFTMAX_KAPPA ((h16)0.0007119178771972656f)

// Launch arguments (must match KBlkParams in clean/k/runner/k_launch.py).
struct KBlkParams {
  const h16* x;       // MAP: block input map [H][W][C]; PADDED: the padded map; EMBED: network input [H][W][16];
                      // EXPAND: the coarser map [ch][cw][CIN]
  const h16* skip;    // EXPAND: skip map [H][W][C]
  h16* y;             // block output map [H][W][C] (null: not written); must not alias x
  h16* merged;        // merge output [ceil(H/2)][ceil(W/2)][CNEXT]
  h16* head;          // head output [H][W][40]
  const h16* g1;      // [C]
  const h16* g2;      // [C]
  const h16* wqkv;    // per head: tiles of Wq, Wk, Wv (softmax) or Wv (fixed), each [2][C/16]
  const h16* mix;     // [heads][64][64]: B (softmax) or M (fixed), [t][s]
  const h16* wo;      // Wo tiles [C/16][D/16]
  const h16* bo;      // [C]
  const h16* w1;      // W1 tiles [4C/16][C/16]
  const h16* b1;      // [4C]
  const h16* w2;      // W2 tiles, input-tile major [4C/16][C/16]
  const h16* b2;      // [C]
  const h16* wa;      // input linear: embedding We tiles [2][1] / expand Wx tiles [4C/16][CIN/16]
  const h16* ba;      // its bias
  const h16* wz;      // output linear: merge Wm tiles [CNEXT/16][4C/16] / head Wh tiles [3][2] (48 rows)
  const h16* bz;      // its bias (head: 48)
  int w, h;           // map size in tokens (the block level)
  int ox, oy;         // window offset, 0 <= ox, oy < 8 (even for merging and decoder stages)
  int nwx, nwy;       // window grid
  int cw, ch;         // EXPAND: coarser map size
};
static_assert(sizeof(KBlkParams) == 184, "KBlkParams must match clean/k/runner/k_launch.py");

// K and VT in LDS are fragment-major: each 16 x 16 tile is stored as the 32 lanes x 8 halfs of its WMMA A fragment, in
// lane order (K: tile (key tile st, channel tile ct) at 2 st + ct; VT: tile (channel tile ct, key tile st) at 4 ct + st).
// The lane that computes a K or v^T fragment (token / channel 16 i + l % 16, 8 values 8 (l / 16) ..) holds exactly
// slot l of the tile, and the reader of an A fragment wants slot l as well: every access is tile base + 16 l bytes,
// conflict-free without padding or swizzle (unpadded row-major tiles: 8-way conflicts, e0 + d0 ~ +0.2 ms at 1440p).
constexpr int KB_FRAG = 256;  // halfs per tile
// Workgroups per WGP that an LDS footprint allows (the kernels run in WGP mode: 128 KB per WGP shared by its four
// SIMDs, so with 4-wave workgroups this is also the waves per SIMD).
constexpr int kb_wg_per_wgp(int bytes) { return bytes > 0 ? 131072 / bytes : 64; }
constexpr int kb_max(int a, int b) { return a > b ? a : b; }
constexpr int kb_min(int a, int b) { return a < b ? a : b; }
#ifndef WGK_MLP_UNROLL
#define WGK_MLP_UNROLL 2  // hidden tiles per MLP loop iteration with FP16 accumulators ...
#endif
#ifndef WGK_MLP_UNROLL_C
#define WGK_MLP_UNROLL_C 64  // ... up to this width (wider: one per iteration)
#endif
#ifndef WGK_MLP_UNROLL16_C64
#define WGK_MLP_UNROLL16_C64 WGK_MLP_UNROLL  // the same for 32 < C <= WGK_MLP_UNROLL_C (4: 1440p -2 us, 4K +5 us)
#endif
// Hidden tiles per MLP loop iteration with FP32 accumulators: C = 32 (e0, d0) and C = 64 (e1, e2, d2, d1); wider: one.
// Two at C = 64 measured faster in whole frames (fused, 1440p: about -14 us); the compiler's choice was one.
#ifndef WGK_MLP_UNROLL32_C32
#define WGK_MLP_UNROLL32_C32 1
#endif
#ifndef WGK_MLP_UNROLL32_C64
#define WGK_MLP_UNROLL32_C64 2
#endif

// IN: input mode (KIN_*); CIN: channels of the input map for EMBED (16) / EXPAND (C_prev); CNEXT: merge output
// channels (0: no merge); HEAD: output head after the block; G: windows per workgroup. With G > 1 every wave holds the
// same two window rows of G consecutive windows (G WMMA column sets), so each weight tile it loads serves G windows:
// the weight reads per token drop by G at the cost of G times the accumulator registers.
// F16: the FP16-accumulating product classes (KF_* of k_common.h; 0 = FP32 accumulation everywhere, spec 41 as written).
// PIX (with HIDDEN): the head goes out as the compact pixel head instead of the 40-channel map (see "compact pixel
// head" at the HEAD epilogue).
// XG (decoder stages with a block output map): x = f16(e + S) goes through P.y instead of LDS (see "LDS" below).
template <int C, int HEADS, int MIX, int IN = KIN_MAP, int CIN = C, int CNEXT = 0, bool HEAD = false, int G = 1,
          int F16 = 0, bool HIDDEN = false, bool PIX = false, bool XG = false>
WG_FN void k_block(const KBlkParams& P, h16* hidden = nullptr) {
  static_assert(!HIDDEN || (HEAD && G == 2), "hidden fusion requires the G2 head stage");
  static_assert(!PIX || HIDDEN, "the compact pixel head needs the hidden epilogue (nothing else can read Hh's inputs)");
  typedef KAcc<(F16 & KF_IN) != 0> AI;
  typedef KAcc<(F16 & KF_QKV) != 0> AQ;
  typedef KAcc<(F16 & KF_SC) != 0> AS;
  typedef KAcc<(F16 & KF_Z) != 0> AZ;
  typedef KAcc<(F16 & KF_WO) != 0> AO;
  typedef KAcc<(F16 & KF_W1) != 0> A1;
  typedef KAcc<(F16 & KF_W2) != 0> A2;
  typedef KAcc<(F16 & KF_MERGE) != 0> AM;
  typedef KAcc<(F16 & KF_HEAD) != 0> AH;
  constexpr int KT = C / 16;        // channel tiles of a token
  constexpr int DT = HEADS * 2;     // channel tiles of z (D = 32 heads)
  constexpr int HT = 4 * C / 16;    // hidden tiles of the MLP
  constexpr int QKV = MIX == KMIX_SOFTMAX ? 3 : 1;
  constexpr bool XS = IN == KIN_EMBED || IN == KIN_EXPAND;  // the block input is built here and kept in LDS
  constexpr bool MS = CNEXT > 0;
  static_assert(IN != KIN_EMBED || (C == 32 && CIN == 16), "embedding: 16 -> 32");
  static_assert(!HEAD || C == 32, "head: 32 -> 40");
  // LDS: per window K (softmax) and VT of a head, in NB buffers; and x: the block input (XS, live until step 3), then
  // the block output for the merge. A merge-only x (XA) is written after the last head, so it shares the K / VT space.
  // NB = 2 (double-buffered by head parity: one barrier per head) unless that allows fewer workgroups per WGP than
  // min(8, single-buffered) (8 waves per SIMD: about where the VGPRs of these kernels limit anyway); NB = 1 needs a
  // second barrier per head before K / VT are overwritten.
  constexpr int KH = QKV == 3 ? 8 * KB_FRAG : 0, VH = 8 * KB_FRAG, XH = (XS || MS) ? 64 * C : 0;
  // XG (decoders with a block output map): the expand staging shares the K / VT space, and x = f16(e + S) goes to the
  // output map P.y at the real positions (step 3 reads it back there; x of the other members is not needed: their x1 and
  // x2 are discarded and nothing after step 2 mixes tokens). The same values, less LDS (more workgroups per WGP at G = 2).
  constexpr bool XA = (MS && !XS) || XG;
  constexpr int LH1 = XA ? kb_max(G * (KH + VH), G * XH) : G * (KH + VH) + G * XH;
  constexpr int LH2 = XA ? kb_max(G * 2 * (KH + VH), G * XH) : G * 2 * (KH + VH) + G * XH;
  constexpr int NB = kb_wg_per_wgp(2 * LH2) >= kb_min(8, kb_wg_per_wgp(2 * LH1)) ? 2 : 1;
  WG_LDS __attribute__((aligned(16))) h16 lds[NB == 2 ? LH2 : LH1];
  h16* const lds_x = XA ? lds : lds + G * NB * (KH + VH);

  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, wave = tid >> 5;
  const int n16 = (int)(lane & 15u), hh = (int)(lane >> 4);
  const int nwin = P.nwx * P.nwy;
  const int t = 16 * (int)wave + n16;  // window-local token of this lane (in each of the G windows)
  int wa[G], wb[G], gx[G], gy[G];
  bool real[G];
  size_t mtok[G];
  const h16* xt[G];
  h16* xl[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    // a missing last window (nwin not a multiple of G) repeats the previous one and stores nothing
    const int wraw = (int)wg_block_id() * G + g;
    const int win = wraw < nwin ? wraw : nwin - 1;
    wa[g] = win % P.nwx;
    wb[g] = win / P.nwx;
    gx[g] = 8 * wa[g] - P.ox + (t & 7);
    gy[g] = 8 * wb[g] - P.oy + (t >> 3);
    real[g] = wraw < nwin && gx[g] >= 0 && gx[g] < P.w && gy[g] >= 0 && gy[g] < P.h;
    mtok[g] = (size_t)wgk_mirror(gy[g], P.h) * P.w + wgk_mirror(gx[g], P.w);  // mirrored member (spec 41)
    const size_t tok =
        IN == KIN_PADDED ? (size_t)(8 * wb[g] + (t >> 3)) * (8 * P.nwx) + 8 * wa[g] + (t & 7) : mtok[g];
    xt[g] = P.x + tok * C + 8 * hh;  // MAP / PADDED
    xl[g] = (lds_x + g * XH) + t * C + 8 * hh;  // XS / MS: this token in LDS
  }

  // g1 first: its loads overlap the block input's (they were issued after the first window's sum)
  v8h g1v[KT];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) g1v[kt] = wgk_ld8(P.g1 + 16 * kt + 8 * hh);

  // 1. the block input
  v8h n[G][KT];
  if (IN == KIN_EMBED) {  // e(t) = max(f16(We in(t) + be), 0) of the mirrored member
    v8h in[G];
#pragma unroll
    for (int g = 0; g < G; ++g) in[g] = wgk_ld8(P.x + mtok[g] * 16 + 8 * hh);
#pragma unroll
    for (int ot = 0; ot < KT; ++ot) {
      const v8h a = wgk_ld8(P.wa + ot * 256 + lane * 8);
      const typename AI::T b0 = AI::start(wgk_ld8(P.ba + 16 * ot + 8 * hh));
#pragma unroll
      for (int g = 0; g < G; ++g) {
        n[g][ot] = wgk_maxc8(AI::out(AI::mma(a, in[g], b0)), (h16)0.0f);
        wgk_st8(xl[g] + 16 * ot, n[g][ot]);
      }
    }
  } else if (IN == KIN_EXPAND) {
    // the window's 4 x 4 coarse tokens: column j = lane % 16 is coarse (R0 + j / 4, C0 + j % 4), mirrored
    constexpr int KTI = CIN / 16, TOT = 4 * C / 16;
    const int jr = n16 >> 2, jc = n16 & 3;
    v8h yb[G][KTI];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int cr = (8 * wb[g] - P.oy) / 2 + jr, cc = (8 * wa[g] - P.ox) / 2 + jc;  // even offsets: exact halves
      const h16* yc = P.x + ((size_t)wgk_mirror(cr, P.ch) * P.cw + wgk_mirror(cc, P.cw)) * CIN + 8 * hh;
#pragma unroll
      for (int kt = 0; kt < KTI; ++kt) yb[g][kt] = wgk_ld8(yc + 16 * kt);
    }
    for (int tt = (int)wave; tt < TOT; tt += 4) {  // output tile tt of 4C: sub-position s, channels c0 ..
      typename AI::T e[G];
      const typename AI::T b0 = AI::start(wgk_ld8(P.ba + 16 * tt + 8 * hh));
#pragma unroll
      for (int g = 0; g < G; ++g) e[g] = b0;
#pragma unroll
      for (int kt = 0; kt < KTI; ++kt) {
        const v8h a = wgk_ld8(P.wa + (size_t)(tt * KTI + kt) * 256 + lane * 8);
#pragma unroll
        for (int g = 0; g < G; ++g) e[g] = AI::mma(a, yb[g][kt], e[g]);
      }
      const int s = (16 * tt) / C, c0 = 16 * tt - s * C;
      const int ft = (2 * jr + (s >> 1)) * 8 + 2 * jc + (s & 1);  // the window member this sub-position feeds
#pragma unroll
      for (int g = 0; g < G; ++g) wgk_st8((lds_x + g * XH) + ft * C + c0 + 8 * hh, AI::out(e[g]));
    }
    wg_barrier_lds();
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const h16* sk = P.skip + mtok[g] * C + 8 * hh;
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        n[g][kt] = wgk_add8(wgk_ld8(xl[g] + 16 * kt), wgk_ld8(sk + 16 * kt));  // x = f16(e + S); own slots only
        if (!XG)
          wgk_st8(xl[g] + 16 * kt, n[g][kt]);
        else if (real[g])
          wgk_st8(P.y + ((size_t)gy[g] * P.w + gx[g]) * C + 8 * hh + 16 * kt, n[g][kt]);
      }
    }
    if (XG) wg_barrier_lds();  // the staging is read; LDS goes to K / VT
  } else {
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) n[g][kt] = wgk_ld8(xt[g] + 16 * kt);
  }

  // normalization: s = f16(sum f16(x^2)) (FP32 sum), r = f16(1 / sqrt(s)); s = inf or 0 gives r = 0. The G windows'
  // sums are independent chains, written interleaved (each in channel order, as before).
  float ss[G];
#pragma unroll
  for (int g = 0; g < G; ++g) ss[g] = 0.0f;
#pragma unroll
  for (int kt = 0; kt < KT; ++kt)
#pragma unroll
    for (int g = 0; g < G; ++g) ss[g] = wgk_sumsq8(n[g][kt], ss[g]);  // + f16(x_c^2), channel order
#pragma unroll
  for (int g = 0; g < G; ++g) ss[g] += wgk_xor16(ss[g]);
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const h16 s16 = (h16)ss[g];
    const h16 r16 = wgk_norm_reciprocal<true>(s16);
    const v8h rv = wgk_splat8h(r16);
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) n[g][kt] = wgk_mul8(n[g][kt], wgk_mul8(rv, g1v[kt]));
  }

  // 2. token mixing, head by head; y accumulates the output projection
  typename AO::T yacc[G][KT];
#pragma unroll
  for (int ot = 0; ot < KT; ++ot) {
    const typename AO::T b0 = AO::start(wgk_ld8(P.bo + 16 * ot + 8 * hh));
#pragma unroll
    for (int g = 0; g < G; ++g) yacc[g][ot] = b0;
  }

  const h16* wl = P.wqkv + lane * 8;
  for (int hd = 0; hd < HEADS; ++hd) {
    const h16* wh = wl + (size_t)hd * QKV * 2 * KT * 256;
    // this lane's slot of the head's K / VT tiles; window g: + g * NB * (KH + VH), VT at + KH
    h16* const kv = lds + (NB == 2 ? (hd & 1) : 0) * (KH + VH) + lane * 8;
    v8h q[G][2];
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
      if (MIX == KMIX_SOFTMAX) {
        typename AQ::T aq[G], ak[G];
#pragma unroll
        for (int g = 0; g < G; ++g) aq[g] = ak[g] = AQ::zero();
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
          const v8h wq = wgk_ld8(wh + (size_t)(0 * 2 * KT + ct * KT + kt) * 256);
          const v8h wk = wgk_ld8(wh + (size_t)(1 * 2 * KT + ct * KT + kt) * 256);
#pragma unroll
          for (int g = 0; g < G; ++g) {
            aq[g] = AQ::mma(wq, n[g][kt], aq[g]);
            ak[g] = AQ::mma(wk, n[g][kt], ak[g]);
          }
        }
        if (NB == 1 && hd > 0 && ct == 0) wg_barrier_lds();  // every wave has read the previous head's K / VT
#pragma unroll
        for (int g = 0; g < G; ++g) {
          q[g][ct] = AQ::out(aq[g]);
          wgk_st8(kv + g * NB * (KH + VH) + (2 * (int)wave + ct) * KB_FRAG, AQ::out(ak[g]));  // K tile (wave, ct)
        }
      }
      // v^T: rows = channels 16 ct + lane % 16 (B columns), elements = tokens 16 wave + 8 hh + 0..7
      typename AQ::T av[G];
#pragma unroll
      for (int g = 0; g < G; ++g) av[g] = AQ::zero();
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        const v8h wv = wgk_ld8(wh + (size_t)((QKV - 1) * 2 * KT + ct * KT + kt) * 256);
#pragma unroll
        for (int g = 0; g < G; ++g) av[g] = AQ::mma(n[g][kt], wv, av[g]);
      }
      if (NB == 1 && MIX == KMIX_FIXED && hd > 0 && ct == 0) wg_barrier_lds();
#pragma unroll
      for (int g = 0; g < G; ++g)
        wgk_st8(kv + g * NB * (KH + VH) + KH + (4 * ct + (int)wave) * KB_FRAG, AQ::out(av[g]));  // VT (ct, wave)
    }
    wg_barrier_lds();

    const h16* mrow = P.mix + ((size_t)hd * 64 + t) * 64 + 8 * hh;
    v8h fixed_pa[4];
    if (MIX == KMIX_FIXED) {  // A^T = M[h][t][s]: the same for every window
#pragma unroll
      for (int st = 0; st < 4; ++st) fixed_pa[st] = wgk_ld8(mrow + 16 * st);
    }
    v8h z[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const h16* lk = kv + g * NB * (KH + VH);
      const h16* lv = lk + KH;
      // A^T fragments: lane = query t, elements = keys 16 st + 8 hh + 0..7
      v8h pa[4];
      if (MIX == KMIX_FIXED) {
#pragma unroll
        for (int st = 0; st < 4; ++st) pa[st] = fixed_pa[st];
      } else {
        v8h gg[4];
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          typename AS::T sc = AS::start(wgk_ld8(mrow + 16 * st));  // + B[h][t][s]
#pragma unroll
          for (int ct = 0; ct < 2; ++ct)
            sc = AS::mma(wgk_ld8(lk + (2 * st + ct) * KB_FRAG), q[g][ct], sc);
          const v8h u = wgk_clamp8c(AS::out(sc), -WGK_SOFTMAX_C, WGK_SOFTMAX_C);
          const v8h u2 = wgk_mul8(u, u);
          const v8h k2 = wgk_mul8(wgk_splat8h(WGK_SOFTMAX_KAPPA), u2);
          const v8h o = wgk_sub8(wgk_splat8h((h16)1.0f), k2);
          gg[st] = wgk_mul8(u, o);
        }
        wgk_softmax_weights<(F16 & KF_SM) != 0>(gg, pa);
      }
      // z_h^T = VT A^T
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) {
        typename AZ::T az = AZ::zero();
#pragma unroll
        for (int st = 0; st < 4; ++st)
          az = AZ::mma(wgk_ld8(lv + (4 * ct + st) * KB_FRAG), pa[st], az);
        z[g][ct] = AZ::out(az);
      }
    }
    // the output projection: each Wo tile once for the G windows
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
      const int dt = 2 * hd + ct;  // channel tile of z (head-major)
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) {
        const v8h wo = wgk_ld8(P.wo + (size_t)(ot * DT + dt) * 256 + lane * 8);
#pragma unroll
        for (int g = 0; g < G; ++g) yacc[g][ot] = AO::mma(wo, z[g][ct], yacc[g][ot]);
      }
    }
  }

  // 3. x1 = f16(x + y);  4. a = f16(x1 g2)
  v8h x1[G][KT], a[G][KT];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    const v8h g2 = wgk_ld8(P.g2 + 16 * kt + 8 * hh);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const v8h xin = !XG ? wgk_ld8(XS ? xl[g] + 16 * kt : xt[g] + 16 * kt)
                      : real[g] ? wgk_ld8(P.y + ((size_t)gy[g] * P.w + gx[g]) * C + 8 * hh + 16 * kt) : wgk_zero8();
      x1[g][kt] = wgk_add8(xin, AO::out(yacc[g][kt]));
      a[g][kt] = wgk_mul8(x1[g][kt], g2);
    }
  }

  // 4. MLP
  typename A2::T acc[G][KT];
#pragma unroll
  for (int ot = 0; ot < KT; ++ot) {
    const typename A2::T b0 = A2::start(wgk_ld8(P.b2 + 16 * ot + 8 * hh));
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g][ot] = b0;
  }
  const h16* w1 = P.w1 + lane * 8;
  const h16* w2 = P.w2 + lane * 8;
  // FP32 accumulators: WGK_MLP_UNROLL32_C32 / _C64 (one per iteration was the compiler's choice). With FP16 ones the
  // compiler unrolled it completely and spilled (C = 32); two hidden tiles per iteration up to C = 64
  // (WGK_MLP_UNROLL16_C64), one above (m: 256 VGPRs + spills otherwise)
  constexpr bool MLP_H = (F16 & (KF_W1 | KF_W2)) != 0;
  constexpr int MLP_UNROLL = MLP_H ? (C <= 32 ? WGK_MLP_UNROLL : C <= WGK_MLP_UNROLL_C ? WGK_MLP_UNROLL16_C64 : 1)
                                   : (C <= 32 ? WGK_MLP_UNROLL32_C32 : C <= 64 ? WGK_MLP_UNROLL32_C64 : 1);
#pragma unroll MLP_UNROLL
  for (int ht = 0; ht < HT; ++ht) {
    typename A1::T p[G];
    const typename A1::T b0 = A1::start(wgk_ld8(P.b1 + 16 * ht + 8 * hh));
#pragma unroll
    for (int g = 0; g < G; ++g) p[g] = b0;
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      const v8h wt = wgk_ld8(w1 + (size_t)(ht * KT + kt) * 256);
#pragma unroll
      for (int g = 0; g < G; ++g) p[g] = A1::mma(wt, a[g][kt], p[g]);
    }
    v8h hv[G];
#pragma unroll
    for (int g = 0; g < G; ++g) hv[g] = wgk_act8(A1::out(p[g]));
#pragma unroll
    for (int ot = 0; ot < KT; ++ot) {
      const v8h wt = wgk_ld8(w2 + (size_t)(ht * KT + ot) * 256);
#pragma unroll
      for (int g = 0; g < G; ++g) acc[g][ot] = A2::mma(wt, hv[g], acc[g][ot]);
    }
  }

  // 5. x2, stored for the real positions
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int ot = 0; ot < KT; ++ot) x1[g][ot] = wgk_add8(x1[g][ot], A2::out(acc[g][ot]));  // x1 now holds x2
    if (real[g] && P.y) {
      h16* yt = P.y + ((size_t)gy[g] * P.w + gx[g]) * C + 8 * hh;
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) wgk_st8(yt + 16 * ot, x1[g][ot]);
    }
  }

  if (MS) {  // 2x2 merge of each window's 16 groups: f16(Wm concat(TL, TR, BL, BR) + bm)
    // own slots only (each lane rewrites the slots it alone read); members outside the map are zero, as the merge of
    // a map with an odd size reads them (spec 44 sizes are even, so there are none)
    if (XA) wg_barrier_lds();  // x shares the K / VT space: every wave has read the last head's
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) wgk_st8(xl[g] + 16 * ot, real[g] ? x1[g][ot] : wgk_zero8());
    wg_barrier_lds();
    constexpr int KTM = 4 * C / 16, NTM = CNEXT / 16;
    const int gr = n16 >> 2, gc = n16 & 3;  // group (column n16): window rows 2 gr, 2 gr + 1, columns 2 gc, 2 gc + 1
    const int mw = (P.w + 1) / 2, mh = (P.h + 1) / 2;
    for (int tt = (int)wave; tt < NTM; tt += 4) {
      typename AM::T m[G];
      const typename AM::T b0 = AM::start(wgk_ld8(P.bz + 16 * tt + 8 * hh));
#pragma unroll
      for (int g = 0; g < G; ++g) m[g] = b0;
#pragma unroll 4
      for (int kt = 0; kt < KTM; ++kt) {
        const int s = (16 * kt) / C, c0 = 16 * kt - s * C;
        const int ft = (2 * gr + (s >> 1)) * 8 + 2 * gc + (s & 1);
        const v8h wm = wgk_ld8(P.wz + (size_t)(tt * KTM + kt) * 256 + lane * 8);
#pragma unroll
        for (int g = 0; g < G; ++g) m[g] = AM::mma(wm, wgk_ld8((lds_x + g * XH) + ft * C + c0 + 8 * hh), m[g]);
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const int my = (8 * wb[g] - P.oy) / 2 + gr, mx = (8 * wa[g] - P.ox) / 2 + gc;
        const bool here = (int)wg_block_id() * G + g < nwin;
        if (here && my >= 0 && my < mh && mx >= 0 && mx < mw)
          wgk_st8(P.merged + ((size_t)my * mw + mx) * CNEXT + 16 * tt + 8 * hh, AM::out(m[g]));
      }
    }
  }

  if (HEAD) {  // out = f16(Wh x2 + bh), 40 of 48 padded rows
    v8h heads[G][3];  // optimized away for ordinary standalone head kernels
#pragma unroll
    for (int ot = 0; ot < 3; ++ot) {
      typename AH::T o[G];
      const typename AH::T b0 = AH::start(wgk_ld8(P.bz + 16 * ot + 8 * hh));
#pragma unroll
      for (int g = 0; g < G; ++g) o[g] = b0;
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        const v8h wt = wgk_ld8(P.wz + (ot * KT + kt) * 256 + lane * 8);
#pragma unroll
        for (int g = 0; g < G; ++g) o[g] = AH::mma(wt, x1[g][kt], o[g]);
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const v8h rounded = AH::out(o[g]);
        if (HIDDEN) heads[g][ot] = rounded;
        if (!PIX && real[g] && 16 * ot + 8 * hh < 40)
          wgk_st8(P.head + ((size_t)gy[g] * P.w + gx[g]) * 40 + 16 * ot + 8 * hh, rounded);
      }
    }
    if (PIX) {
      // Compact pixel head: k_out_pixel reads only channels 0, 1 and 9 of a display position's slot (spec 43 3.2,
      // 5.1), and the hidden history (channels 3..7) is written below. So instead of the 40-channel map, P.head is
      // [2 h][2 w][4] FP16 in display-position raster order: slot k of token (gx, gy) at row 2 gy + (k >> 1), column
      // 2 gx + (k & 1), holding (channel 10 k, 10 k + 1, 10 k + 9, 0), the same rounded values. A lane of half hh
      // holds channels 16 ot + 8 hh + 0..7, so it has channels 10 k, 10 k + 1 of slots k = hh and k = 2 + hh and gets
      // their third value from its partner lane (one exchange of two halfs per window).
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const v8h h0 = heads[g][0], h1 = heads[g][1], h2 = heads[g][2];
        const h16 a0 = hh ? h0[2] : h0[0], a1 = hh ? h0[3] : h0[1];  // slot hh: channels 10 hh, 10 hh + 1
        const h16 b0 = hh ? h1[6] : h1[4], b1 = hh ? h1[7] : h1[5];  // slot 2 + hh: channels 20 + 10 hh, + 1
        // to the partner: channel 9 + 10 k of its slots (hh = 0 holds 19, 39; hh = 1 holds 9, 29)
        const v2h send = {hh ? h0[1] : h1[3], hh ? h1[5] : h2[7]};
        const v2h recv = __builtin_bit_cast(v2h, wgk_xor16(__builtin_bit_cast(float, send)));
        if (real[g]) {
          h16* const d = P.head + ((size_t)(2 * gy[g]) * (2 * P.w) + 2 * gx[g] + hh) * 4;
          const v4h sa = {a0, a1, recv[0], (h16)0.0f}, sb = {b0, b1, recv[1], (h16)0.0f};
          *(v4h*)d = sa;
          *(v4h*)(d + (size_t)(2 * P.w) * 4) = sb;
        }
      }
    }
    if (HIDDEN) {
#pragma clang fp contract(off)
      // Each lane-half owns one G2 window for this epilogue. The partner lane
      // sends the opposite window's fragment, so xor16 supplies our missing half.
      // All lanes exchange before the real-position guard; a missing last window
      // still supplies its partner's data but writes no hidden output.
      float hs[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float v[5];
#pragma unroll
        for (int j = 0; j < 5; ++j) {
          const int c = 10 * k + 3 + j;
          const float own = (float)(hh == 0 ? heads[0][c / 16][c % 8] : heads[1][c / 16][c % 8]);
          const float send = (float)(hh == 0 ? heads[1][c / 16][c % 8] : heads[0][c / 16][c % 8]);
          const float other = wgk_xor16(send);
          v[j] = ((c % 16) / 8) == hh ? own : other;
        }
        hs[k] = 0.2f * ((((v[0] + v[1]) + v[2]) + v[3]) + v[4]);
      }
      if (real[hh]) wgk_store_hidden(hidden + ((size_t)gy[hh] * P.w + gx[hh]) * 4, hs);
    }
  }
}

// The eleven stages use five (C, heads, mixing) combinations (spec 40 table).
#define K_BLOCK_SHAPES(X)                                                                       \
  X(32, 2, fixed, KMIX_FIXED) X(64, 2, fixed, KMIX_FIXED) X(64, 2, softmax, KMIX_SOFTMAX)       \
  X(96, 4, softmax, KMIX_SOFTMAX) X(128, 4, softmax, KMIX_SOFTMAX) X(160, 8, softmax, KMIX_SOFTMAX)

#define K_DEF_BLOCK(c, hd, name, mix)                                                                    \
  WG_KERNEL_N(k_block_c##c##_h##hd##_##name, 128)(KBlkParams P) { k_block<c, hd, mix>(P); }              \
  WG_KERNEL_N(k_blockp_c##c##_h##hd##_##name, 128)(KBlkParams P) { k_block<c, hd, mix, KIN_PADDED>(P); }
K_BLOCK_SHAPES(K_DEF_BLOCK)

// Whole stages (spec 40 table): name, C, heads, mixing, input mode, CIN, CNEXT, head. Stage m is the plain block.
#define K_STAGES(X)                                                          \
  X(e0, 32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false)                         \
  X(e1, 64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false)                           \
  X(e2, 64, 2, KMIX_SOFTMAX, KIN_MAP, 64, 96, false)                         \
  X(e3, 96, 4, KMIX_SOFTMAX, KIN_MAP, 96, 128, false)                        \
  X(e4, 128, 4, KMIX_SOFTMAX, KIN_MAP, 128, 160, false)                      \
  X(d4, 128, 4, KMIX_SOFTMAX, KIN_EXPAND, 160, 0, false)                     \
  X(d3, 96, 4, KMIX_SOFTMAX, KIN_EXPAND, 128, 0, false)                      \
  X(d2, 64, 2, KMIX_SOFTMAX, KIN_EXPAND, 96, 0, false)                       \
  X(d1, 64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false)                       \
  X(d0, 32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true)
#define K_DEF_STAGE(name, c, hd, mix, in, cin, cnext, head) \
  WG_KERNEL_N(k_stage_##name, 128)(KBlkParams P) { k_block<c, hd, mix, in, cin, cnext, head>(P); }
K_STAGES(K_DEF_STAGE)
// Two windows per workgroup (G = 2) for the width-32 stages at full base resolution, the most token-heavy ones:
// half the weight reads per token (build variant for timing; the stage plan "fused2" uses them).
#define K_STAGES_G2(X)                                          \
  X(e0, 32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false)            \
  X(d0, 32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true)
#define K_DEF_STAGE2(name, c, hd, mix, in, cin, cnext, head) \
  WG_KERNEL_N(k_stage2_##name, 128)(KBlkParams P) { k_block<c, hd, mix, in, cin, cnext, head, 2>(P); }
K_STAGES_G2(K_DEF_STAGE2)
// Two windows per workgroup at level 1 as well (e1, d1; k_launch.py WINDOWS2_STAGES): half the weight reads per token
// there too (whole frames, fused, 1440p Quality -8 us, 4K Performance -27 us). d1 runs with XG (x through its block
// output map, see k_block), which keeps its LDS at 16 KB; it needs P.y (always set in the network). e2 / d2 with two
// windows measured no faster (the FP32 forms spill).
K_DEF_STAGE2(e1, 64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false)
WG_KERNEL_N(k_stage2_d1, 128)(KBlkParams P) {
  k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, 0, false, false, true>(P);
}
WG_KERNEL_N(k_stage2h_d1, 128)(KBlkParams P) {
  k_block<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, WGK_F16ACC, false, false, true>(P);
}
// Complete FP32 frames can emit Hh directly from the rounded d0 head; standalone
// stage calls and other network variants retain their original entry points.
WG_KERNEL_N(k_stage2_d0_hidden, 128)(KBlkParams P, h16* hidden) {
  k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, 0, true>(P, hidden);
}
// FP16-accumulating forms (spec 49 "Speed approximations", opt-in variants "fused16" / "<stage>@f16"): the same
// kernels with the product classes WGK_F16ACC (k_common.h, default KF_DEFAULT) accumulating in FP16. Name: the FP32
// kernel's prefix + "h" (k_stageh_<name>, m included; k_stage2h_<e0, d0>).
#define K_DEF_STAGEH(name, c, hd, mix, in, cin, cnext, head) \
  WG_KERNEL_N(k_stageh_##name, 128)(KBlkParams P) { k_block<c, hd, mix, in, cin, cnext, head, 1, WGK_F16ACC>(P); }
K_STAGES(K_DEF_STAGEH)
K_DEF_STAGEH(m, 160, 8, KMIX_SOFTMAX, KIN_MAP, 160, 0, false)
#define K_DEF_STAGE2H(name, c, hd, mix, in, cin, cnext, head) \
  WG_KERNEL_N(k_stage2h_##name, 128)(KBlkParams P) { k_block<c, hd, mix, in, cin, cnext, head, 2, WGK_F16ACC>(P); }
K_STAGES_G2(K_DEF_STAGE2H)
K_DEF_STAGE2H(e1, 64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false)
// The same epilogue for complete "fused16" frames: Hh is the same function of the rounded head (spec 43 section 6.8),
// whichever way the head's matrix product accumulated, so k_out_hidden's separate pass over the head is not needed.
WG_KERNEL_N(k_stage2h_d0_hidden, 128)(KBlkParams P, h16* hidden) {
  k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, WGK_F16ACC, true>(P, hidden);
}
// Complete frames whose output stage is k_out_pixel_c: the hidden epilogue and the compact pixel head (P.head is the
// [2 h][2 w][4] FP16 map) instead of the 40-channel head map. The same rounded head values.
WG_KERNEL_N(k_stage2_d0_hp, 128)(KBlkParams P, h16* hidden) {
  k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, 0, true, true>(P, hidden);
}
WG_KERNEL_N(k_stage2h_d0_hp, 128)(KBlkParams P, h16* hidden) {
  k_block<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, WGK_F16ACC, true, true>(P, hidden);
}

// FP8 forms of the level-0/1 stages (opt-in network variant "fused8"; GPU only): k_block8.h
#include "k_block8.h"

#if defined(WG_EMU)
WG_EMU_REGISTER(k_stage2_d0_hidden)
WG_EMU_REGISTER(k_stage2h_d0_hidden)
WG_EMU_REGISTER(k_stage2_d0_hp)
WG_EMU_REGISTER(k_stage2h_d0_hp)
#define K_REG_BLOCK(c, hd, name, mix) \
  WG_EMU_REGISTER(k_block_c##c##_h##hd##_##name) WG_EMU_REGISTER(k_blockp_c##c##_h##hd##_##name)
K_BLOCK_SHAPES(K_REG_BLOCK)
#define K_REG_STAGE(name, c, hd, mix, in, cin, cnext, head) WG_EMU_REGISTER(k_stage_##name)
K_STAGES(K_REG_STAGE)
#define K_REG_STAGE2(name, c, hd, mix, in, cin, cnext, head) WG_EMU_REGISTER(k_stage2_##name)
K_STAGES_G2(K_REG_STAGE2)
#define K_REG_STAGEH(name, c, hd, mix, in, cin, cnext, head) WG_EMU_REGISTER(k_stageh_##name)
K_STAGES(K_REG_STAGEH)
WG_EMU_REGISTER(k_stageh_m)
#define K_REG_STAGE2H(name, c, hd, mix, in, cin, cnext, head) WG_EMU_REGISTER(k_stage2h_##name)
K_STAGES_G2(K_REG_STAGE2H)
WG_EMU_REGISTER(k_stage2_e1)
WG_EMU_REGISTER(k_stage2h_e1)
WG_EMU_REGISTER(k_stage2_d1)
WG_EMU_REGISTER(k_stage2h_d1)
#endif
