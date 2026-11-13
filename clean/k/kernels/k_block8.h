// SPDX-License-Identifier: MIT
// Model K: FP8 forms of the level-0/1 stage kernels (opt-in network variant "fused8"; not bit-exact, a speed
// approximation beyond "fused16"). Included by k_block.hip after k_block; GPU only (gfx12 FP8 WMMA).
//
// What changes against the FP16-accumulating kernels of "fused16" (k_stage2h_*, everything else is the same code and
// the same FP16 classes WGK_F16ACC): the matrix products with the most instructions at levels 0 and 1 run on
// v_wmma_f32_16x16x16_fp8_fp8 (OCP E4M3 inputs, FP32 accumulation; twice the rate of the FP16 instruction on the
// RX 9070 XT, measured 7.7 against 15.5 cycles per instruction and SIMD, and both forms issue in turn with the VALU):
//   KF8_ATT  token mixing: v (fixed mixing: n' = fp8(n) as the A operand; softmax: v from the FP16 product), the mixing
//            z = A v (VT in LDS as FP8, half the bytes; A = fp8(M 2^km) or fp8(softmax weights 2^8)), and the output
//            projection Wo z; x1 = f16(x + acc 2^-s) in one rounding (v_fma_mix*).
//   KF8_MLP  the MLP: a' = fp8(clamp(x1 g2, +-448)), p = W1' a' + b1' in FP32, the activation evaluated in FP32 on the
//            scaled accumulator with the scales folded into its constants (one more min for saturation), h' = fp8 of
//            it, and x2 = f16(x1 + (W2' h' + b2') 2^-s) in one rounding.
// Kept in FP16 / FP32 as in "fused16": the normalization (r, f16(r g1)), q, k and the scores, the softmax (exp2, sum,
// reciprocal), the residual stream x, x1, x2 (FP16 maps), the embedding / expand, the merge and the head.
//
// Scales (KF8Params, chosen per stage on the host from the weights, runtime_k.cpp fp8_stage): every weight tensor
// W' = fp8(W 2^k) with 2^k the largest power of two that keeps max |W'| <= 448 (no saturation); activations that feed
// an FP8 operand are scaled by powers of two folded into constants, with bounds that keep them below 448 where a bound
// exists (n: |n_c| <= |g1_c|; v and z: from the weights) and an explicit clamp where none does (a, h). The FP8
// conversion (v_cvt_pk_fp8_f32) rounds to nearest even and gives NaN above 464, never a saturated value: every value
// converted here is bounded or clamped first.
#pragma once
#if defined(__AMDGCN__)

enum { KF8_ATT = 1, KF8_MLP = 2, KF8_ALL = 3 };
// hidden tiles per iteration of the FP8 MLP loop at C = 32 and C = 64 (network time at 1440p Quality against 1 / 2:
// -2.4 / -4.0 us; with 2 at C = 64 e1 and d1 spill)
#ifndef WGK_MLP8_UNROLL_C32
#define WGK_MLP8_UNROLL_C32 2
#endif
#ifndef WGK_MLP8_UNROLL_C64
#define WGK_MLP8_UNROLL_C64 1
#endif
#ifndef WGK_F8
#define WGK_F8 KF8_ALL
#endif

// Second argument of the FP8 stage kernels (must match KF8Params in clean/k/runtime/runtime_k.cpp).
struct KF8Params {
  const u8* wqkv;  // FP8 copy of the block's qkv section, element for element: fp8(W 2^kv) (only the v tiles are read)
  const u8* mix;   // fixed mixing: fp8(M 2^km), [heads][64][64]
  const u8* wo;    // fp8(Wo 2^ko), Wo's tile order
  const u8* w1;    // fp8(W1 2^k1)
  const u8* w2;    // fp8(W2 2^k2)
  const float* bo;  // bo 2^(ko + kz), FP32
  const float* b1;  // b1 2^k1 (a is not scaled)
  const float* b2;  // b2 2^(k2 + kh)
  float pv;  // softmax: VT' = fp8(acc_v pv) (fixed mixing: VT' = fp8(acc_v), the scales fit by the bound)
  float sz;  // z' = fp8(acc_z sz)
  float sy;  // x1 = f16(x + acc_y sy)
  float L, Hs, A2, B2, c0;  // activation on acc1 = 2^k1 p: T = med3(acc1, -L, L), S = min(acc1, Hs),
                            // h' = S (c0 + T (A2 - B2 |T|)) = fp8 value of h 2^kh
  float sx;  // x2 = f16(x1 + acc2 sx)
  float pad;
};
static_assert(sizeof(KF8Params) == 104, "KF8Params must match clean/k/runtime/runtime_k.cpp");

WG_FN v2i wgk8_ld(const u8* p) { return *(const v2i*)p; }
WG_FN void wgk8_st(u8* p, v2i v) { *(v2i*)p = v; }
WG_FN v8f wgk8_ldf(const float* p) { return *(const v8f*)p; }  // 32-byte aligned

// 8 FP32 values -> 8 OCP E4M3 bytes (element j in byte j), round to nearest even; |d| < 464 required.
WG_FN v2i wgk8_cvt(v8f d) {
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(d[0], d[1], 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(d[2], d[3], lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(d[4], d[5], 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(d[6], d[7], hi, true);
  return (v2i){lo, hi};
}
WG_FN v8f wgk8_scale(v8f d, float s) { return d * s; }
WG_FN v8f wgk8_clamp(v8f d, float lim) {
  v8f r;
#pragma unroll
  for (int j = 0; j < 8; ++j) r[j] = __builtin_amdgcn_fmed3f(d[j], -lim, lim);
  return r;
}

// (float)a * (float)b per element of two FP16 fragments (exact products), v_fma_mix_f32 a * b + 0.
WG_FN v8f wgk8_mul_hh(v8h a, v8h b) {
  const v4u ua = __builtin_bit_cast(v4u, a), ub = __builtin_bit_cast(v4u, b);
  v8f r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float lo, hi;
    __asm__("v_fma_mix_f32 %0, %1, %2, 0 op_sel_hi:[1,1,0]" : "=v"(lo) : "v"(ua[i]), "v"(ub[i]));
    __asm__("v_fma_mix_f32 %0, %1, %2, 0 op_sel:[1,1,0] op_sel_hi:[1,1,0]" : "=v"(hi) : "v"(ua[i]), "v"(ub[i]));
    r[2 * i] = lo;
    r[2 * i + 1] = hi;
  }
  return r;
}
// (float)a * s per element of an FP16 fragment (v_fma_mix_f32 a * s + 0; s FP32, uniform).
WG_FN v8f wgk8_mul_hs(v8h a, float s) {
  const v4u ua = __builtin_bit_cast(v4u, a);
  v8f r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float lo, hi;
    __asm__("v_fma_mix_f32 %0, %1, %2, 0 op_sel_hi:[1,0,0]" : "=v"(lo) : "v"(ua[i]), "s"(s));
    __asm__("v_fma_mix_f32 %0, %1, %2, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(hi) : "v"(ua[i]), "s"(s));
    r[2 * i] = lo;
    r[2 * i + 1] = hi;
  }
  return r;
}
// f16(x + acc s) per element, one rounding (v_fma_mixlo_f16 / v_fma_mixhi_f16 acc * s + x).
WG_FN v8h wgk8_resid(v8f acc, float s, v8h x) {
  const v4u ux = __builtin_bit_cast(v4u, x);
  v4u r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    u32 d;
    __asm__("v_fma_mixlo_f16 %0, %1, %2, %3 op_sel_hi:[0,0,1]" : "=v"(d) : "v"(acc[2 * i]), "s"(s), "v"(ux[i]));
    __asm__("v_fma_mixhi_f16 %0, %1, %2, %3 op_sel:[0,0,1] op_sel_hi:[0,0,1]"
            : "+v"(d)
            : "v"(acc[2 * i + 1]), "s"(s), "v"(ux[i]));
    r[i] = d;
  }
  return __builtin_bit_cast(v8h, r);
}
// The activation of spec 40 on the scaled FP32 accumulator (see KF8Params), saturated, as FP8.
WG_FN v2i wgk8_act(v8f p, const KF8Params& Q) {
  v8f h;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float T = __builtin_amdgcn_fmed3f(p[j], -Q.L, Q.L);
    const float S = __builtin_fminf(p[j], Q.Hs);
    const float u = __builtin_fmaf(-Q.B2, __builtin_fabsf(T), Q.A2);
    const float q = __builtin_fmaf(T, u, Q.c0);
    h[j] = S * q;
  }
  return wgk8_cvt(h);
}
// Softmax weights as FP8 of A 2^8 (the FP16 form of wgk_softmax_weights<true>: w = f16(2^g), FP32 sum, then
// A 2^8 = w (2^8 / sum) in FP32, rounded once to FP8).
WG_FN void wgk8_softmax_weights(const v8h (&gg)[4], v2i (&pa)[4]) {
  float sum = 0.0f;
  v8h wh[4];
#pragma unroll
  for (int st = 0; st < 4; ++st) {
    wh[st] = wgk_exp2h8(gg[st]);
    sum = wgk_sum8h(wh[st], sum);
  }
  sum += wg_xor16(sum);
  const float s = 256.0f * wg_rcp(sum);
#pragma unroll
  for (int st = 0; st < 4; ++st) pa[st] = wgk8_cvt(wgk8_mul_hs(wh[st], s));
}

// The stage kernel of k_block with the FP8 classes F8 (KF8_*); template parameters as k_block (F16 = WGK_F16ACC for
// everything that is not FP8).
template <int C, int HEADS, int MIX, int IN = KIN_MAP, int CIN = C, int CNEXT = 0, bool HEAD = false, int G = 1,
          bool HIDDEN = false, bool PIX = false, bool XG = false, int F8 = WGK_F8>
WG_FN void k_block8(const KBlkParams& P, const KF8Params& Q, h16* hidden = nullptr) {
  constexpr int F16 = WGK_F16ACC;
  constexpr bool ATT8 = (F8 & KF8_ATT) != 0, MLP8 = (F8 & KF8_MLP) != 0;
  static_assert(!HIDDEN || (HEAD && G == 2), "hidden fusion requires the G2 head stage");
  static_assert(!PIX || HIDDEN, "the compact pixel head needs the hidden epilogue");
  typedef KAcc<(F16 & KF_IN) != 0> AI;
  typedef KAcc<(F16 & KF_QKV) != 0> AQ;
  typedef KAcc<(F16 & KF_SC) != 0> AS;
  typedef KAcc<(F16 & KF_Z) != 0> AZ;
  typedef KAcc<(F16 & KF_WO) != 0> AO;
  typedef KAcc<(F16 & KF_W1) != 0> A1;
  typedef KAcc<(F16 & KF_W2) != 0> A2;
  typedef KAcc<(F16 & KF_MERGE) != 0> AM;
  typedef KAcc<(F16 & KF_HEAD) != 0> AH;
  constexpr int KT = C / 16;
  constexpr int DT = HEADS * 2;
  constexpr int HT = 4 * C / 16;
  constexpr int QKV = MIX == KMIX_SOFTMAX ? 3 : 1;
  constexpr bool XS = IN == KIN_EMBED || IN == KIN_EXPAND;
  constexpr bool MS = CNEXT > 0;
  static_assert(IN != KIN_EMBED || (C == 32 && CIN == 16), "embedding: 16 -> 32");
  static_assert(!HEAD || C == 32, "head: 32 -> 40");
  // LDS as in k_block; VT holds FP8 with ATT8 (8 tiles of 256 bytes: half of VH)
  constexpr int KH = QKV == 3 ? 8 * KB_FRAG : 0, VH = ATT8 ? 4 * KB_FRAG : 8 * KB_FRAG,
                XH = (XS || MS) ? 64 * C : 0;
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
  const int t = 16 * (int)wave + n16;
  int wa[G], wb[G], gx[G], gy[G];
  bool real[G];
  size_t mtok[G];
  const h16* xt[G];
  h16* xl[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const int wraw = (int)wg_block_id() * G + g;
    const int win = wraw < nwin ? wraw : nwin - 1;
    wa[g] = win % P.nwx;
    wb[g] = win / P.nwx;
    gx[g] = 8 * wa[g] - P.ox + (t & 7);
    gy[g] = 8 * wb[g] - P.oy + (t >> 3);
    real[g] = wraw < nwin && gx[g] >= 0 && gx[g] < P.w && gy[g] >= 0 && gy[g] < P.h;
    mtok[g] = (size_t)wgk_mirror(gy[g], P.h) * P.w + wgk_mirror(gx[g], P.w);
    const size_t tok =
        IN == KIN_PADDED ? (size_t)(8 * wb[g] + (t >> 3)) * (8 * P.nwx) + 8 * wa[g] + (t & 7) : mtok[g];
    xt[g] = P.x + tok * C + 8 * hh;
    xl[g] = (lds_x + g * XH) + t * C + 8 * hh;
  }

  // 1. the block input (as k_block)
  v8h n[G][KT];
  if (IN == KIN_EMBED) {
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
    constexpr int KTI = CIN / 16, TOT = 4 * C / 16;
    const int jr = n16 >> 2, jc = n16 & 3;
    v8h yb[G][KTI];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int cr = (8 * wb[g] - P.oy) / 2 + jr, cc = (8 * wa[g] - P.ox) / 2 + jc;
      const h16* yc = P.x + ((size_t)wgk_mirror(cr, P.ch) * P.cw + wgk_mirror(cc, P.cw)) * CIN + 8 * hh;
#pragma unroll
      for (int kt = 0; kt < KTI; ++kt) yb[g][kt] = wgk_ld8(yc + 16 * kt);
    }
    for (int tt = (int)wave; tt < TOT; tt += 4) {
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
      const int ft = (2 * jr + (s >> 1)) * 8 + 2 * jc + (s & 1);
#pragma unroll
      for (int g = 0; g < G; ++g) wgk_st8((lds_x + g * XH) + ft * C + c0 + 8 * hh, AI::out(e[g]));
    }
    wg_barrier_lds();
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const h16* sk = P.skip + mtok[g] * C + 8 * hh;
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        n[g][kt] = wgk_add8(wgk_ld8(xl[g] + 16 * kt), wgk_ld8(sk + 16 * kt));
        if (!XG)
          wgk_st8(xl[g] + 16 * kt, n[g][kt]);
        else if (real[g])
          wgk_st8(P.y + ((size_t)gy[g] * P.w + gx[g]) * C + 8 * hh + 16 * kt, n[g][kt]);
      }
    }
    if (XG) wg_barrier_lds();
  } else {
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) n[g][kt] = wgk_ld8(xt[g] + 16 * kt);
  }

  // normalization (as k_block): rg = f16(r g1); n = f16(x rg), or (fixed mixing, ATT8) n' = fp8(x rg) in one rounding
  constexpr bool N8 = ATT8 && MIX == KMIX_FIXED;
  v2i n8[G][KT];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float ss = 0.0f;
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) ss = wgk_sumsq8(n[g][kt], ss);
    ss += wgk_xor16(ss);
    const h16 s16 = (h16)ss;
    const h16 r16 = wgk_norm_reciprocal(s16);
    const v8h rv = wgk_splat8h(r16);
#pragma unroll
    for (int kt = 0; kt < KT; ++kt) {
      const v8h rg = wgk_mul8(rv, wgk_ld8(P.g1 + 16 * kt + 8 * hh));
      if (N8)
        n8[g][kt] = wgk8_cvt(wgk8_mul_hh(n[g][kt], rg));  // |x r| <= 1: |n| <= |g1| <= 448 (host check)
      else
        n[g][kt] = wgk_mul8(n[g][kt], rg);
    }
  }

  // 2. token mixing
  typedef KAcc<false> AF;  // FP32 accumulators of the FP8 products
  typedef typename AO::T YT;
  v8f yacc8[G][KT];
  YT yacc[G][KT];
#pragma unroll
  for (int ot = 0; ot < KT; ++ot) {
    if (ATT8) {
      const v8f b0 = wgk8_ldf(Q.bo + 16 * ot + 8 * hh);
#pragma unroll
      for (int g = 0; g < G; ++g) yacc8[g][ot] = b0;
    } else {
      const YT b0 = AO::start(wgk_ld8(P.bo + 16 * ot + 8 * hh));
#pragma unroll
      for (int g = 0; g < G; ++g) yacc[g][ot] = b0;
    }
  }

  const h16* wl = P.wqkv + lane * 8;
  const u8* wl8 = Q.wqkv + lane * 8;
  for (int hd = 0; hd < HEADS; ++hd) {
    const h16* wh = wl + (size_t)hd * QKV * 2 * KT * 256;
    const u8* wh8 = wl8 + (size_t)hd * QKV * 2 * KT * 256;
    h16* const kvb = lds + (NB == 2 ? (hd & 1) : 0) * (KH + VH);
    h16* const kv = kvb + lane * 8;
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
        if (NB == 1 && hd > 0 && ct == 0) wg_barrier_lds();
#pragma unroll
        for (int g = 0; g < G; ++g) {
          q[g][ct] = AQ::out(aq[g]);
          wgk_st8(kv + g * NB * (KH + VH) + (2 * (int)wave + ct) * KB_FRAG, AQ::out(ak[g]));
        }
      }
      if (ATT8) {
        // v^T as FP32 accumulators -> FP8 VT tile (ct, wave): 256 bytes per tile, 8 per lane
        v8f av[G];
#pragma unroll
        for (int g = 0; g < G; ++g) av[g] = wgk_zero8f();
        if (N8) {
#pragma unroll
          for (int kt = 0; kt < KT; ++kt) {
            const v2i wv = wgk8_ld(wh8 + (size_t)((QKV - 1) * 2 * KT + ct * KT + kt) * 256);
#pragma unroll
            for (int g = 0; g < G; ++g) av[g] = wg_wmma_fp8(n8[g][kt], wv, av[g]);
          }
        } else {
#pragma unroll
          for (int kt = 0; kt < KT; ++kt) {
            const v8h wv = wgk_ld8(wh + (size_t)((QKV - 1) * 2 * KT + ct * KT + kt) * 256);
#pragma unroll
            for (int g = 0; g < G; ++g) av[g] = AF::mma(n[g][kt], wv, av[g]);
          }
        }
        if (NB == 1 && MIX == KMIX_FIXED && hd > 0 && ct == 0) wg_barrier_lds();
#pragma unroll
        for (int g = 0; g < G; ++g) {
          u8* vt = (u8*)(kvb + g * NB * (KH + VH) + KH) + (4 * ct + (int)wave) * 256 + lane * 8;
          wgk8_st(vt, wgk8_cvt(N8 ? av[g] : wgk8_scale(av[g], Q.pv)));
        }
      } else {
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
          wgk_st8(kv + g * NB * (KH + VH) + KH + (4 * ct + (int)wave) * KB_FRAG, AQ::out(av[g]));
      }
    }
    wg_barrier_lds();

    const h16* mrow = P.mix + ((size_t)hd * 64 + t) * 64 + 8 * hh;
    const u8* mrow8 = Q.mix + ((size_t)hd * 64 + t) * 64 + 8 * hh;
    v8h fixed_pa[4];
    v2i fixed_pa8[4];
    if (MIX == KMIX_FIXED) {
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        if (ATT8)
          fixed_pa8[st] = wgk8_ld(mrow8 + 16 * st);
        else
          fixed_pa[st] = wgk_ld8(mrow + 16 * st);
      }
    }
    v8h z[G][2];
    v2i z8[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const h16* lk = kv + g * NB * (KH + VH);
      const h16* lv = lk + KH;
      const u8* lv8 = (const u8*)(kvb + g * NB * (KH + VH) + KH) + lane * 8;
      v8h pa[4];
      v2i pa8[4];
      if (MIX == KMIX_FIXED) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          pa[st] = fixed_pa[st];
          pa8[st] = fixed_pa8[st];
        }
      } else {
        v8h gg[4];
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          typename AS::T sc = AS::start(wgk_ld8(mrow + 16 * st));
#pragma unroll
          for (int ct = 0; ct < 2; ++ct) sc = AS::mma(wgk_ld8(lk + (2 * st + ct) * KB_FRAG), q[g][ct], sc);
          const v8h u = wgk_clamp8c(AS::out(sc), -WGK_SOFTMAX_C, WGK_SOFTMAX_C);
          const v8h u2 = wgk_mul8(u, u);
          const v8h k2 = wgk_mul8(wgk_splat8h(WGK_SOFTMAX_KAPPA), u2);
          const v8h o = wgk_sub8(wgk_splat8h((h16)1.0f), k2);
          gg[st] = wgk_mul8(u, o);
        }
        if (ATT8)
          wgk8_softmax_weights(gg, pa8);
        else
          wgk_softmax_weights<(F16 & KF_SM) != 0>(gg, pa);
      }
#pragma unroll
      for (int ct = 0; ct < 2; ++ct) {
        if (ATT8) {
          v8f az = wgk_zero8f();
#pragma unroll
          for (int st = 0; st < 4; ++st) az = wg_wmma_fp8(wgk8_ld(lv8 + (4 * ct + st) * 256), pa8[st], az);
          z8[g][ct] = wgk8_cvt(wgk8_scale(az, Q.sz));
        } else {
          typename AZ::T az = AZ::zero();
#pragma unroll
          for (int st = 0; st < 4; ++st) az = AZ::mma(wgk_ld8(lv + (4 * ct + st) * KB_FRAG), pa[st], az);
          z[g][ct] = AZ::out(az);
        }
      }
    }
#pragma unroll
    for (int ct = 0; ct < 2; ++ct) {
      const int dt = 2 * hd + ct;
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) {
        if (ATT8) {
          const v2i wo = wgk8_ld(Q.wo + (size_t)(ot * DT + dt) * 256 + lane * 8);
#pragma unroll
          for (int g = 0; g < G; ++g) yacc8[g][ot] = wg_wmma_fp8(wo, z8[g][ct], yacc8[g][ot]);
        } else {
          const v8h wo = wgk_ld8(P.wo + (size_t)(ot * DT + dt) * 256 + lane * 8);
#pragma unroll
          for (int g = 0; g < G; ++g) yacc[g][ot] = AO::mma(wo, z[g][ct], yacc[g][ot]);
        }
      }
    }
  }

  // 3. x1 = f16(x + y);  4. a = x1 g2 (FP16, or FP8 of the FP32 product, clamped)
  v8h x1[G][KT], a[G][KT];
  v2i a8[G][KT];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    const v8h g2 = wgk_ld8(P.g2 + 16 * kt + 8 * hh);
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const v8h xin = !XG ? wgk_ld8(XS ? xl[g] + 16 * kt : xt[g] + 16 * kt)
                      : real[g] ? wgk_ld8(P.y + ((size_t)gy[g] * P.w + gx[g]) * C + 8 * hh + 16 * kt) : wgk_zero8();
      if (ATT8)
        x1[g][kt] = wgk8_resid(yacc8[g][kt], Q.sy, xin);
      else
        x1[g][kt] = wgk_add8(xin, AO::out(yacc[g][kt]));
      if (MLP8)
        a8[g][kt] = wgk8_cvt(wgk8_clamp(wgk8_mul_hh(x1[g][kt], g2), 448.0f));
      else
        a[g][kt] = wgk_mul8(x1[g][kt], g2);
    }
  }

  // 4. MLP
  if (MLP8) {
    v8f acc[G][KT];
#pragma unroll
    for (int ot = 0; ot < KT; ++ot) {
      const v8f b0 = wgk8_ldf(Q.b2 + 16 * ot + 8 * hh);
#pragma unroll
      for (int g = 0; g < G; ++g) acc[g][ot] = b0;
    }
    const u8* w1 = Q.w1 + lane * 8;
    const u8* w2 = Q.w2 + lane * 8;
    constexpr int MLP_UNROLL = C <= 32 ? WGK_MLP8_UNROLL_C32 : WGK_MLP8_UNROLL_C64;
#pragma unroll MLP_UNROLL
    for (int ht = 0; ht < HT; ++ht) {
      v8f p[G];
      const v8f b0 = wgk8_ldf(Q.b1 + 16 * ht + 8 * hh);
#pragma unroll
      for (int g = 0; g < G; ++g) p[g] = b0;
#pragma unroll
      for (int kt = 0; kt < KT; ++kt) {
        const v2i wt = wgk8_ld(w1 + (size_t)(ht * KT + kt) * 256);
#pragma unroll
        for (int g = 0; g < G; ++g) p[g] = wg_wmma_fp8(wt, a8[g][kt], p[g]);
      }
      v2i hv[G];
#pragma unroll
      for (int g = 0; g < G; ++g) hv[g] = wgk8_act(p[g], Q);
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) {
        const v2i wt = wgk8_ld(w2 + (size_t)(ht * KT + ot) * 256);
#pragma unroll
        for (int g = 0; g < G; ++g) acc[g][ot] = wg_wmma_fp8(wt, hv[g], acc[g][ot]);
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) x1[g][ot] = wgk8_resid(acc[g][ot], Q.sx, x1[g][ot]);  // x1 now holds x2
  } else {
    typename A2::T acc[G][KT];
#pragma unroll
    for (int ot = 0; ot < KT; ++ot) {
      const typename A2::T b0 = A2::start(wgk_ld8(P.b2 + 16 * ot + 8 * hh));
#pragma unroll
      for (int g = 0; g < G; ++g) acc[g][ot] = b0;
    }
    const h16* w1 = P.w1 + lane * 8;
    const h16* w2 = P.w2 + lane * 8;
    constexpr int MLP_UNROLL = C <= 32 ? WGK_MLP_UNROLL : C <= WGK_MLP_UNROLL_C ? WGK_MLP_UNROLL16_C64 : 1;
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
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) x1[g][ot] = wgk_add8(x1[g][ot], A2::out(acc[g][ot]));
  }

  // 5. x2, stored for the real positions (as k_block)
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (real[g] && P.y) {
      h16* yt = P.y + ((size_t)gy[g] * P.w + gx[g]) * C + 8 * hh;
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) wgk_st8(yt + 16 * ot, x1[g][ot]);
    }
  }

  if (MS) {  // merge (as k_block)
    if (XA) wg_barrier_lds();
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int ot = 0; ot < KT; ++ot) wgk_st8(xl[g] + 16 * ot, real[g] ? x1[g][ot] : wgk_zero8());
    wg_barrier_lds();
    constexpr int KTM = 4 * C / 16, NTM = CNEXT / 16;
    const int gr = n16 >> 2, gc = n16 & 3;
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

  if (HEAD) {  // as k_block
    v8h heads[G][3];
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
#pragma unroll
      for (int g = 0; g < G; ++g) {
        const v8h h0 = heads[g][0], h1 = heads[g][1], h2 = heads[g][2];
        const h16 a0 = hh ? h0[2] : h0[0], a1 = hh ? h0[3] : h0[1];
        const h16 b0 = hh ? h1[6] : h1[4], b1 = hh ? h1[7] : h1[5];
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

// The four two-window stages of levels 0 and 1 (k_launch.py WINDOWS2_STAGES), as their "fused16" kernels
// (k_stage2h_e0, k_stage2h_e1, k_stage2h_d1 with XG, k_stage2h_d0_hp / _hidden) with the FP8 classes WGK_F8.
WG_KERNEL_N(k_stage2q_e0, 128)(KBlkParams P, KF8Params Q) {
  k_block8<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 2>(P, Q);
}
WG_KERNEL_N(k_stage2q_e1, 128)(KBlkParams P, KF8Params Q) {
  k_block8<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 2>(P, Q);
}
WG_KERNEL_N(k_stage2q_d1, 128)(KBlkParams P, KF8Params Q) {
  k_block8<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, false, false, true>(P, Q);
}
WG_KERNEL_N(k_stage2q_d0_hp, 128)(KBlkParams P, KF8Params Q, h16* hidden) {
  k_block8<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, true>(P, Q, hidden);
}
WG_KERNEL_N(k_stage2q_d0_hidden, 128)(KBlkParams P, KF8Params Q, h16* hidden) {
  k_block8<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true>(P, Q, hidden);
}

#endif  // __AMDGCN__
