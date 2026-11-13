// SPDX-License-Identifier: MIT
// Model K block of spec 41, wide form for the small stages (levels 3..5: e3, e4, m, d4, d3).
//
// The stage kernels of k_block.hip run one 8x8 window per 128-lane workgroup. At levels 3..5 there are only 20..252
// windows at 1440p, so those kernels run about one wave per SIMD and every weight load is waited on: they are latency
// bound. Here one window is processed by NG wave groups of 4 token waves (128 NG lanes); every group holds the
// window's 64 tokens (16 per wave, the layout of k_block.hip / k_common.h) and the groups split the work:
//
//   input     MAP: each group loads and normalizes all channels itself (cheap, cached). EXPAND: the expand's 4C/16
//             output tiles are spread over all 4 NG waves, then each group adds the skip map for all channels.
//   heads     in rounds of NG: group g computes head r NG + g of round r (q, k, v, scores, softmax, z) with its own
//             K / VT LDS buffers, and writes its z_h to LDS. After a barrier every group reads the round's 2 NG z
//             tiles.
//   Wo, MLP   the output channel tiles are owned: group g owns ot = g, g + NG, ... of the residual stream (Wo
//             accumulators, x1, W2 accumulators, x2). The hidden tiles go in chunks of NG M: group g computes W1, the
//             activation and h for M of them and writes h to LDS (double-buffered); after a barrier every group adds
//             W2 h over the chunk's tiles in order for its own output tiles.
//   merge / expand: their output tiles are spread over all 4 NG waves.
// The first weight tiles needed after a barrier (the scores' B, Wo, W2) are loaded before it, and the ownership has no
// branches (a group without a last tile computes tile KT - 1 again and does not store it): both measured faster.
//
// Every FP32 accumulator sums in the same order as k_block.hip (bias first, then Wo over the z tiles of heads 0, 1, ..;
// W2 over hidden tiles 0, 1, ..) and is rounded once, and every elementwise step is the same, so the results are bit
// for bit those of k_stage_* (spec 41 would allow any FP32 order; this one needs no FP32 reduction through LDS).
// The FP16-accumulating forms k_stagew<NG>h_* (template parameter F16, a spec 49 speed approximation, opt-in) keep the
// same orders, so their FP16 accumulator chains round at the same points: bit for bit those of k_stageh_*.
// Softmax mixing only (all stages at levels 3..5 use it).
//
// Kernels: k_stagew<NG>_<name> for e3, e4, m, d4, d3 and NG = 2, 4 (m: 4 only; with NG = 2 its LDS does not fit), and
// k_stagew<NG>h_<name> (FP16 accumulation, below).
// Launch: one workgroup of 128 NG lanes per window (clean/k/runner/k_launch.py stage_fused_launch, wide=NG). The
// network uses NG = 2, and 4 for m (k_launch.WIDE_DEFAULT): with NG = 4 the stages with 252 windows (e3, d3) are slower
// (56 KB LDS: one workgroup per CU), the ones with 66 about the same.
// Cross-lane handoffs in this file use LDS only. Global inputs/weights are read-only
// during a launch; global outputs are consumed by later stream-ordered kernels.
// LDS-scoped barriers preserve those handoffs without invalidating global caches.
// The compiler's clamps (k_common.h WGK_ASM_CLAMP): the asm form measured slower here (+3.8 us per frame, fused 1440p).
#ifndef WGK_ASM_CLAMP
#define WGK_ASM_CLAMP 0
#endif
#include "k_common.h"
#include "k_norm.h"

enum { KMIX_FIXED = 0, KMIX_SOFTMAX = 1 };
enum { KIN_MAP = 0, KIN_PADDED = 1, KIN_EMBED = 2, KIN_EXPAND = 3 };

#define WGK_SOFTMAX_C ((h16)21.640625f)
#define WGK_SOFTMAX_KAPPA ((h16)0.0007119178771972656f)

// Launch arguments: the same structure as KBlkParams of k_block.hip (and clean/k/runner/k_launch.py).
struct KBlkParams {
  const h16* x;
  const h16* skip;
  h16* y;
  h16* merged;
  h16* head;
  const h16* g1;
  const h16* g2;
  const h16* wqkv;
  const h16* mix;
  const h16* wo;
  const h16* bo;
  const h16* w1;
  const h16* b1;
  const h16* w2;
  const h16* b2;
  const h16* wa;
  const h16* ba;
  const h16* wz;
  const h16* bz;
  int w, h;
  int ox, oy;
  int nwx, nwy;
  int cw, ch;
};
static_assert(sizeof(KBlkParams) == 184, "KBlkParams must match clean/k/runner/k_launch.py");

constexpr int KW_KSTRIDE = 40;  // LDS row of K: 32 channels + 8 halfs padding
constexpr int KW_VSTRIDE = 72;  // LDS row of VT: 64 tokens + 8 halfs padding
constexpr int kw_max(int a, int b) { return a > b ? a : b; }

// C, HEADS, MIX as k_block; IN: KIN_MAP or KIN_EXPAND; CIN: EXPAND input channels (C_prev); CNEXT: merge output
// channels (0: no merge); NG: wave groups.
// F16: the FP16-accumulating product classes (KF_* of k_common.h; 0 = FP32 accumulation, spec 41 as written).
template <int C, int HEADS, int MIX, int IN, int CIN, int CNEXT, int NG, int F16 = 0>
WG_FN void k_block_wide(const KBlkParams& P) {
  typedef KAcc<(F16 & KF_IN) != 0> AI;
  typedef KAcc<(F16 & KF_QKV) != 0> AQ;
  typedef KAcc<(F16 & KF_SC) != 0> AS;
  typedef KAcc<(F16 & KF_Z) != 0> AZ;
  typedef KAcc<(F16 & KF_WO) != 0> AO;
  typedef KAcc<(F16 & KF_W1) != 0> A1;
  typedef KAcc<(F16 & KF_W2) != 0> A2;
  typedef KAcc<(F16 & KF_MERGE) != 0> AM;
  static_assert(MIX == KMIX_SOFTMAX, "wide block: softmax mixing only");
  static_assert(IN == KIN_MAP || IN == KIN_EXPAND, "wide block: map or expand input");
  static_assert(HEADS % NG == 0, "heads go to the groups in rounds of NG");
  constexpr int KT = C / 16;                  // channel tiles of a token
  constexpr int DT = HEADS * 2;               // channel tiles of z
  constexpr int HT = 4 * C / 16;              // hidden tiles of the MLP
  constexpr int NW = 4 * NG;                  // waves of the workgroup
  constexpr int NOWN = (KT + NG - 1) / NG;    // output tiles a group owns (at most)
  constexpr int M = 2;                        // hidden tiles per group and chunk (4: much slower, LDS)
  constexpr int CH = NG * M;                  // hidden tiles per chunk
  constexpr int PF = 2;  // weight tiles per owned tile loaded before a barrier for the first steps after it
  // keep the second q / k / v tile set's loads after the first one's products: without it the decoder stages and m
  // spill (d4: 47 VGPRs); the encoder stages are faster without
  constexpr bool QFENCE = IN == KIN_EXPAND || C > 128;
  static_assert(HT % CH == 0, "whole chunks");
  constexpr bool MS = CNEXT > 0;
  // LDS, in halfs. Attention: per group K [64][KW_KSTRIDE] and VT [32][KW_VSTRIDE], then Z [64][ZS] (the round's z).
  // MLP: A [64][C] (a = f16(x1 g2), later x2 for the merge) at OFF_A, H [2][CH][64][16] after it. EXPAND: the block
  // input [64][C] at 0. The phases are separated by barriers (see below).
  constexpr int KVH = 64 * KW_KSTRIDE + 32 * KW_VSTRIDE;
  constexpr int ZS = 32 * NG + 8;
  constexpr int ATT = NG * KVH + 64 * ZS;
  constexpr int OFF_A = 64 * C <= NG * KVH ? 0 : ATT;  // A must not overlap Z (other groups may still read it)
  constexpr int OFF_H = OFF_A + 64 * C;
  constexpr int LDS_H = kw_max(ATT, OFF_H + 2 * CH * 64 * 16);
  static_assert(LDS_H * 2 <= 65536, "LDS");
  WG_LDS __attribute__((aligned(16))) h16 lds[LDS_H];

  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, wave = tid >> 5;
  const int grp = (int)(wave >> 2), wv = (int)(wave & 3u);
  const int n16 = (int)(lane & 15u), hh = (int)(lane >> 4);
  const int t = 16 * wv + n16;  // window-local token of this lane
  const int win = (int)wg_block_id();
  const int wa = win % P.nwx, wb = win / P.nwx;
  const int gx = 8 * wa - P.ox + (t & 7), gy = 8 * wb - P.oy + (t >> 3);
  const bool real = gx >= 0 && gx < P.w && gy >= 0 && gy < P.h;
  const size_t mtok = (size_t)wgk_mirror(gy, P.h) * P.w + wgk_mirror(gx, P.w);  // mirrored member (spec 41)
  h16* xl = lds + t * C + 8 * hh;          // EXPAND: this token of the block input
  h16* al = lds + OFF_A + t * C + 8 * hh;  // this token of A

  // owned output tiles: ot = grp + NG i (i < NOWN), present when ot < KT. When KT is not a multiple of NG, the groups
  // without a last tile compute tile KT - 1 as well (no branches; the critical path is the groups that own one) and do
  // not store it.
  bool own[NOWN];
  int otv[NOWN];
#pragma unroll
  for (int i = 0; i < NOWN; ++i) {
    own[i] = KT % NG == 0 || grp + NG * i < KT;
    otv[i] = own[i] ? grp + NG * i : KT - 1;
  }

  // 1. the block input (n: all channels; xr: the owned tiles, for the residual)
  v8h n[KT], xr[NOWN];
  if (IN == KIN_EXPAND) {
    // the window's 4 x 4 coarse tokens: column j = lane % 16 is coarse (R0 + j / 4, C0 + j % 4), mirrored
    constexpr int KTI = CIN / 16, TOT = 4 * C / 16;
    const int jr = n16 >> 2, jc = n16 & 3;
    const int cr = (8 * wb - P.oy) / 2 + jr, cc = (8 * wa - P.ox) / 2 + jc;  // even offsets: exact halves
    const h16* yc = P.x + ((size_t)wgk_mirror(cr, P.ch) * P.cw + wgk_mirror(cc, P.cw)) * CIN + 8 * hh;
    v8h yb[KTI];
#pragma unroll
    for (int kt = 0; kt < KTI; ++kt) yb[kt] = wgk_ld8(yc + 16 * kt);
    for (int tt = (int)wave; tt < TOT; tt += NW) {  // output tile tt of 4C: sub-position s, channels c0 ..
      typename AI::T e = AI::start(wgk_ld8(P.ba + 16 * tt + 8 * hh));
#pragma unroll
      for (int kt = 0; kt < KTI; ++kt)
        e = AI::mma(wgk_ld8(P.wa + (size_t)(tt * KTI + kt) * 256 + lane * 8), yb[kt], e);
      const int s = (16 * tt) / C, c0 = 16 * tt - s * C;
      const int ft = (2 * jr + (s >> 1)) * 8 + 2 * jc + (s & 1);  // the window member this sub-position feeds
      wgk_st8(lds + ft * C + c0 + 8 * hh, AI::out(e));
    }
    wg_barrier_lds();
    const h16* sk = P.skip + mtok * C + 8 * hh;
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) n[kt] = wgk_add8(wgk_ld8(xl + 16 * kt), wgk_ld8(sk + 16 * kt));  // x = f16(e + S)
#pragma unroll
    for (int i = 0; i < NOWN; ++i) xr[i] = wgk_add8(wgk_ld8(xl + 16 * otv[i]), wgk_ld8(sk + 16 * otv[i]));
    wg_barrier_lds();  // the block input is read; LDS goes to K / VT
  } else {
    const h16* xt = P.x + mtok * C + 8 * hh;
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) n[kt] = wgk_ld8(xt + 16 * kt);
#pragma unroll
    for (int i = 0; i < NOWN; ++i) xr[i] = wgk_ld8(xt + 16 * otv[i]);
  }

  // normalization: s = f16(sum f16(x^2)) (FP32 sum), r = f16(1 / sqrt(s)); s = inf or 0 gives r = 0
  {
    float ss = 0.0f;
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = (float)n[kt][j];
        ss += (float)(h16)(v * v);  // f16(x_c^2)
      }
    }
    ss += wgk_xor16(ss);
    const h16 s16 = (h16)ss;
    const h16 r16 = wgk_norm_reciprocal(s16);
    const v8h rv = wgk_splat8h(r16);
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) n[kt] = wgk_mul8(n[kt], wgk_mul8(rv, wgk_ld8(P.g1 + 16 * kt + 8 * hh)));
  }

  // 2. token mixing in rounds of NG heads; y (owned tiles) accumulates the output projection
  typename AO::T yacc[NOWN];
#pragma unroll
  for (int i = 0; i < NOWN; ++i) yacc[i] = AO::start(wgk_ld8(P.bo + 16 * otv[i] + 8 * hh));

  h16* lk = lds + grp * KVH;
  h16* lv = lk + 64 * KW_KSTRIDE;
  h16* lz = lds + NG * KVH;
  const h16* wl = P.wqkv + lane * 8;
  for (int r = 0; r < HEADS / NG; ++r) {
    const int hd = r * NG + grp;
    const h16* wh = wl + (size_t)hd * 3 * 2 * KT * 256;
    v8h q[2];
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
      typename AQ::T aq = AQ::zero(), ak = AQ::zero(), av = AQ::zero();
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        aq = AQ::mma(wgk_ld8(wh + (size_t)(0 * 2 * KT + ct * KT + kt) * 256), n[kt], aq);
        ak = AQ::mma(wgk_ld8(wh + (size_t)(1 * 2 * KT + ct * KT + kt) * 256), n[kt], ak);
        // v^T: rows = channels 16 ct + lane % 16 (B columns), elements = tokens 16 wv + 8 hh + 0..7
        av = AQ::mma(n[kt], wgk_ld8(wh + (size_t)(2 * 2 * KT + ct * KT + kt) * 256), av);
      }
      q[ct] = AQ::out(aq);
      wgk_st8(lk + t * KW_KSTRIDE + 16 * ct + 8 * hh, AQ::out(ak));  // K[token t][channels]
      wgk_st8(lv + (16 * ct + n16) * KW_VSTRIDE + 16 * wv + 8 * hh, AQ::out(av));
      if (QFENCE) WG_SCHED_FENCE();
    }
    // softmax: scores^T = K q + B[h][t][s] (lane = query t, elements = keys 16 st + 8 hh + 0..7); B is loaded before
    // the barrier
    const h16* mrow = P.mix + ((size_t)hd * 64 + t) * 64 + 8 * hh;
    v8h bm[4];
#pragma unroll
    for (int st = 0; st < 4; ++st) bm[st] = wgk_ld8(mrow + 16 * st);
    wg_barrier_lds();  // K / VT of the group's head complete; the previous round's Z reads are done

    v8h pa[4];
    {
      v8h gg[4];
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        typename AS::T sc = AS::start(bm[st]);
#pragma unroll
        for (int ct = 0; ct < 2; ++ct)
          sc = AS::mma(wgk_ld8(lk + (16 * st + n16) * KW_KSTRIDE + 16 * ct + 8 * hh), q[ct], sc);
        v8h u = AS::out(sc);
        u = wgk_min8(wgk_max8(u, wgk_splat8h(-WGK_SOFTMAX_C)), wgk_splat8h(WGK_SOFTMAX_C));
        const v8h u2 = wgk_mul8(u, u);
        const v8h k2 = wgk_mul8(wgk_splat8h(WGK_SOFTMAX_KAPPA), u2);
        const v8h o = wgk_sub8(wgk_splat8h((h16)1.0f), k2);
        gg[st] = wgk_mul8(u, o);
      }
      wgk_softmax_weights<(F16 & KF_SM) != 0>(gg, pa);
    }
    // z_h^T = VT A^T -> Z[t][32 grp + 16 ct ..]
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
      typename AZ::T az = AZ::zero();
#pragma unroll
      for (int st = 0; st < 4; ++st)
        az = AZ::mma(wgk_ld8(lv + (16 * ct + n16) * KW_VSTRIDE + 16 * st + 8 * hh), pa[st], az);
      wgk_st8(lz + t * ZS + 32 * grp + 16 * ct + 8 * hh, AZ::out(az));
    }
    // the output projection for the owned tiles over the round's z tiles dt = 2 r NG + j, in order (the first
    // PF tiles loaded before the barrier)
    const h16* wor = P.wo + (size_t)2 * r * NG * 256 + lane * 8;
    v8h wpf[PF][NOWN];
#pragma unroll
    for (int j = 0; j < PF; ++j)
#pragma unroll
      for (int i = 0; i < NOWN; ++i) wpf[j][i] = wgk_ld8(wor + (size_t)(otv[i] * DT + j) * 256);
    wg_barrier_lds();  // the round's z complete; K / VT reads are done
#pragma unroll
    for (int j = 0; j < 2 * NG; ++j) {
      const v8h zj = wgk_ld8(lz + t * ZS + 16 * j + 8 * hh);
#pragma unroll
      for (int i = 0; i < NOWN; ++i)
        yacc[i] = AO::mma(j < PF ? wpf[j < PF ? j : 0][i] : wgk_ld8(wor + (size_t)(otv[i] * DT + j) * 256), zj,
                              yacc[i]);
    }
  }

  // 3. x1 = f16(x + y) and a = f16(x1 g2) for the owned tiles; a goes to A (no Z, K or VT is read any more there)
  v8h x1[NOWN];
#pragma unroll
  for (int i = 0; i < NOWN; ++i) {
    x1[i] = wgk_add8(xr[i], AO::out(yacc[i]));
    if (own[i]) wgk_st8(al + 16 * otv[i], wgk_mul8(x1[i], wgk_ld8(P.g2 + 16 * otv[i] + 8 * hh)));
  }
  wg_barrier_lds();
  v8h a[KT];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) a[kt] = wgk_ld8(al + 16 * kt);

  // 4. MLP in chunks of CH hidden tiles: W1 + act by the groups (M tiles each), W2 for the owned output tiles
  typename A2::T acc[NOWN];
#pragma unroll
  for (int i = 0; i < NOWN; ++i) acc[i] = A2::start(wgk_ld8(P.b2 + 16 * otv[i] + 8 * hh));
  const h16* w1 = P.w1 + lane * 8;
  const h16* w2 = P.w2 + lane * 8;
  for (int c = 0; c < HT / CH; ++c) {
    h16* hb = lds + OFF_H + (c & 1) * CH * 1024;  // tile j of the chunk: [64 tokens][16]
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const int ht = c * CH + grp * M + m;
      typename A1::T p = A1::start(wgk_ld8(P.b1 + 16 * ht + 8 * hh));
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) p = A1::mma(wgk_ld8(w1 + (size_t)(ht * KT + kt) * 256), a[kt], p);
      wgk_st8(hb + (grp * M + m) * 1024 + t * 16 + 8 * hh, wgk_act8(A1::out(p)));
    }
    const h16* w2c = w2 + (size_t)c * CH * KT * 256;
    v8h wpf[PF][NOWN];
#pragma unroll
    for (int j = 0; j < PF; ++j)
#pragma unroll
      for (int i = 0; i < NOWN; ++i) wpf[j][i] = wgk_ld8(w2c + (size_t)(j * KT + otv[i]) * 256);
    wg_barrier_lds();  // h of the chunk complete (the other buffer's reads finished before the previous barrier)
#pragma unroll
    for (int j = 0; j < CH; ++j) {
      const v8h hv = wgk_ld8(hb + j * 1024 + t * 16 + 8 * hh);
#pragma unroll
      for (int i = 0; i < NOWN; ++i)
        acc[i] = A2::mma(j < PF ? wpf[j < PF ? j : 0][i] : wgk_ld8(w2c + (size_t)(j * KT + otv[i]) * 256), hv,
                             acc[i]);
    }
  }

  // 5. x2 = f16(x1 + f16(W2 h + b2)), stored for the real positions
#pragma unroll
  for (int i = 0; i < NOWN; ++i) x1[i] = wgk_add8(x1[i], A2::out(acc[i]));  // x1 now holds x2
  if (real && P.y) {
    h16* yt = P.y + ((size_t)gy * P.w + gx) * C + 8 * hh;
#pragma unroll
    for (int i = 0; i < NOWN; ++i)
      if (own[i]) wgk_st8(yt + 16 * otv[i], x1[i]);
  }

  if (MS) {  // 2x2 merge of the window's 16 groups: f16(Wm concat(TL, TR, BL, BR) + bm); x2 goes to A (A's reads are
             // done since the first chunk; members outside the map are zero, as in k_block.hip)
#pragma unroll
    for (int i = 0; i < NOWN; ++i)
      if (own[i]) wgk_st8(al + 16 * otv[i], real ? x1[i] : wgk_zero8());
    wg_barrier_lds();
    constexpr int KTM = 4 * C / 16, NTM = CNEXT / 16;
    const int gr = n16 >> 2, gc = n16 & 3;  // group (column n16): window rows 2 gr, 2 gr + 1, columns 2 gc, 2 gc + 1
    const int mw = (P.w + 1) / 2, mh = (P.h + 1) / 2;
    const int my = (8 * wb - P.oy) / 2 + gr, mx = (8 * wa - P.ox) / 2 + gc;
    const h16* xa = lds + OFF_A;
    for (int tt = (int)wave; tt < NTM; tt += NW) {
      typename AM::T m = AM::start(wgk_ld8(P.bz + 16 * tt + 8 * hh));
#pragma unroll 4
      for (int kt = 0; kt < KTM; ++kt) {
        const int s = (16 * kt) / C, c0 = 16 * kt - s * C;
        const int ft = (2 * gr + (s >> 1)) * 8 + 2 * gc + (s & 1);
        const v8h wm = wgk_ld8(P.wz + (size_t)(tt * KTM + kt) * 256 + lane * 8);
        m = AM::mma(wm, wgk_ld8(xa + ft * C + c0 + 8 * hh), m);
      }
      if (my >= 0 && my < mh && mx >= 0 && mx < mw)
        wgk_st8(P.merged + ((size_t)my * mw + mx) * CNEXT + 16 * tt + 8 * hh, AM::out(m));
    }
  }
}

// name, C, heads, input mode, CIN, CNEXT (spec 40 table)
#define K_WIDE_STAGES(X)                              \
  X(e3, 96, 4, KIN_MAP, 96, 128, 2)                   \
  X(e3, 96, 4, KIN_MAP, 96, 128, 4)                   \
  X(e4, 128, 4, KIN_MAP, 128, 160, 2)                 \
  X(e4, 128, 4, KIN_MAP, 128, 160, 4)                 \
  X(m, 160, 8, KIN_MAP, 160, 0, 4)                    \
  X(d4, 128, 4, KIN_EXPAND, 160, 0, 2)                \
  X(d4, 128, 4, KIN_EXPAND, 160, 0, 4)                \
  X(d3, 96, 4, KIN_EXPAND, 128, 0, 2)                 \
  X(d3, 96, 4, KIN_EXPAND, 128, 0, 4)
#define K_DEF_WIDE(name, c, hd, in, cin, cnext, ng)          \
  WG_KERNEL_N(k_stagew##ng##_##name, 128 * ng)(KBlkParams P) { \
    k_block_wide<c, hd, KMIX_SOFTMAX, in, cin, cnext, ng>(P);  \
  }
K_WIDE_STAGES(K_DEF_WIDE)
// FP16-accumulating forms (spec 49 "Speed approximations", opt-in): k_stagew<NG>h_<name>, the product classes
// WGK_F16ACC (k_common.h) accumulating in FP16; bit for bit k_stageh_<name> of k_block.hip (same order of the FP16
// accumulator chains).
#define K_DEF_WIDEH(name, c, hd, in, cin, cnext, ng)                     \
  WG_KERNEL_N(k_stagew##ng##h_##name, 128 * ng)(KBlkParams P) {          \
    k_block_wide<c, hd, KMIX_SOFTMAX, in, cin, cnext, ng, WGK_F16ACC>(P); \
  }
K_WIDE_STAGES(K_DEF_WIDEH)

#if defined(WG_EMU)
#define K_REG_WIDE(name, c, hd, in, cin, cnext, ng) WG_EMU_REGISTER(k_stagew##ng##_##name)
K_WIDE_STAGES(K_REG_WIDE)
#define K_REG_WIDEH(name, c, hd, in, cin, cnext, ng) WG_EMU_REGISTER(k_stagew##ng##h_##name)
K_WIDE_STAGES(K_REG_WIDEH)
#endif
