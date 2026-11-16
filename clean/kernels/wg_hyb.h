// SPDX-License-Identifier: MIT
// HIP twin of clean/dx12/hlsl/wg_hyb.hlsli: the element-wise steps of the HYBRID block form (SW_HYB / WG_L_FORM=hybrid),
// the same values computed directly instead of from the D3D12 arena tables.
// - RN16: the (_Float16) conversion (v_cvt_f16_f32, round to nearest even).
// - FP16 x FP16 and FP16 +- FP16 in float32 rounded once to FP16 are exact (24 >= 2 x 11 + 2), and every product here
//   is of two FP16 values (exact in float32), so clang's default contraction into fma gives the same bits.
// - rsqrt / rcp: 1 / sqrtf(x) and 1 / x with HIP's default correctly rounded division and square root, as the oracle's
//   float32 (clean/l/reference/oracle.py _rsqrt32, _rcp32).
// - the softmax weight's two fma steps: the exhaustive fact softmax_e_f32_equals_fma16 (clean/l/reference/hyb_tables.py).
// Every helper is checked for all FP16 inputs against the oracle by clean/l/runner/l_hyb_probe_check.py --hip.
#pragma once
#include "wg_platform.h"

#define WG_HYB_SCORE_C 0.0171813964843750f      // FP16 0x2466
#define WG_HYB_SCORE_CLAMP 0.55615234375f       // FP16 0x3873
#define WG_HYB_POLY_T 0.92724609375f            // FP16 0x3B6B
#define WG_HYB_POLY_P 1.375f                    // FP16 0x3D80
#define WG_HYB_GELU_A 0.412109375f              // FP16 0x3698
#define WG_HYB_GELU_B 0.0810546875f             // FP16 0x2D30
#define WG_HYB_MERGE_CLAMP 6.28125f

WG_FN float hyb_rn16(float x) { return (float)(_Float16)x; }
WG_FN v8f hyb_round16(v8f d) {
  v8f r;
  for (int i = 0; i < 8; ++i) r[i] = hyb_rn16(d[i]);
  return r;
}
WG_FN unsigned hyb_bits16(float x) { return (unsigned)__builtin_bit_cast(unsigned short, (_Float16)x); }
WG_FN float hyb_from16(unsigned b) { return (float)__builtin_bit_cast(_Float16, (unsigned short)b); }

// the chain end before an FP8 point: RN16, then the saturating FP8 pack (two roundings, by design)
WG_FN v2i hyb_q8_of_chain(v8f acc) { return wg_pack_q8(hyb_round16(acc)); }

// GELU16 (oracle gelu_q7): every step its own FP16 rounding; NaN takes y = -2 (fmax returns the number)
WG_FN float hyb_gelu16(float h) {
  const float y = __builtin_fminf(__builtin_fmaxf(h, -2.0f), 2.0f);
  const float inner = hyb_rn16(WG_HYB_GELU_A - hyb_rn16(WG_HYB_GELU_B * __builtin_fabsf(y)));
  return hyb_rn16(h * hyb_rn16(0.5f + hyb_rn16(y * inner)));
}
WG_FN v2i hyb_gelu_q8(v8f h) {
  v8f g;
  for (int i = 0; i < 8; ++i) g[i] = hyb_gelu16(h[i]);
  return wg_pack_q8(g);
}

// the softmax weight e of 8 FP16 scores (float values of FP16 numbers)
WG_FN v8f hyb_softmax_e8(v8f s) {
  v8f e;
  for (int i = 0; i < 8; ++i) {
    float u = hyb_rn16(s[i] * WG_HYB_SCORE_C);
    u = __builtin_fminf(__builtin_fmaxf(u, -WG_HYB_SCORE_CLAMP), WG_HYB_SCORE_CLAMP);
    const float t = hyb_rn16(__builtin_fmaf(u, -u, WG_HYB_POLY_T));
    const float p = hyb_rn16(__builtin_fmaf(u, t, WG_HYB_POLY_P));
    e[i] = hyb_from16((hyb_bits16(p) & 0x03ffu) << 5);
  }
  return e;
}

WG_FN float hyb_rsq16(float s) { return hyb_rn16(1.0f / __builtin_sqrtf(s)); }   // S' >= 2^-13
WG_FN float hyb_rcp16(float x) { return hyb_rn16(1.0f / x); }                     // normal sum, normal result

WG_FN float hyb_merge_clamp(float x) {
  return x != x ? x : __builtin_fminf(__builtin_fmaxf(x, -WG_HYB_MERGE_CLAMP), WG_HYB_MERGE_CLAMP);
}

// The merges' FP8 point: FP8(clamp(RN16(chain), +-6.28125)), NaN kept (oracle run_stage_q7: q8(clamp(mm8(...)))).
WG_FN v2i hyb_merge_q8(v8f acc) {
  v8f r = hyb_round16(acc);
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = hyb_merge_clamp(r[i]);
  return wg_pack_q8(r);
}
WG_FN int hyb_mirror_fine(int p, int n) {
  const int a = p < 0 ? -p : p;
  const int m = 2 * n - 2 - a;
  return (n - 1) < (a < m ? a : m) ? (n - 1) : (a < m ? a : m);
}
WG_FN int hyb_mirror_coarse(int p, int n, int nc) {
  const int c = p >> 1;
  const int a = c < 0 ? -c : c;
  const int m = 2 * (n >> 1) - 2 - a;
  const int v = a < m ? a : m;
  return (nc - 1) < v ? (nc - 1) : v;
}

// ---- the block's norm and softmax row in swin.hip's register layout (swin_block<..., HYB = true>) -------------------
// Lane (t, hh) holds token t's channels hh C/2 + 8 k + v in tiles k, and the scores of keys 8 hh + v (gfx12 D rows).
// The sums follow the original's fixed FP16 orders (oracle norm_q7, softmax_q7), the partner half by lane exchange; both
// lanes of a token compute the same sums. Same arithmetic as clean/dx12/hlsl/hyb_block.hlsli (FP16 steps exact).
WG_FN unsigned hyb_xor16_u(unsigned v) { return __builtin_bit_cast(unsigned, wg_xor16(__builtin_bit_cast(float, v))); }
WG_FN float hyb_add16(float a, float b) { return hyb_rn16(a + b); }   // exact: FP16 operands, one float rounding first

template <int KC>
WG_FN void hyb_norm(const v8f* x, const float* gp, unsigned hh, v8f* y) {
#pragma clang fp contract(off)
  v8f T[2 * KC];
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    v8f q, qp;
#pragma unroll
    for (int v = 0; v < 8; ++v) q[v] = hyb_rn16(x[k][v] * x[k][v]);     // q = RN16(x x), its own rounding
#pragma unroll
    for (int v = 0; v < 8; ++v) qp[v] = wg_xor16(q[v]);
    T[k] = hh == 0 ? q : qp;
    T[KC + k] = hh == 0 ? qp : q;
  }
  constexpr int tiles = 2 * KC;
  v8f pl[4], pr[4];
  int n = 0;
#pragma unroll
  for (int j = 0; j < tiles; j += 4, ++n)
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      pl[n][v] = hyb_add16(T[j][v], T[j + 2][v]);
      pr[n][v] = hyb_add16(T[j + 1][v], T[j + 3][v]);
    }
  auto group = [&](const v8f* p, int m, int v) {
    if (m == 1) return p[0][v];
    if (m == 2) return hyb_add16(p[0][v], p[1][v]);
    if (m == 3) return hyb_add16(hyb_add16(p[0][v], p[1][v]), p[2][v]);
    return hyb_add16(hyb_add16(p[0][v], p[1][v]), hyb_add16(p[2][v], p[3][v]));
  };
  float L[8];
#pragma unroll
  for (int v = 0; v < 8; ++v) L[v] = hyb_add16(group(pl, n, v), group(pr, n, v));
  const float h0 = hyb_add16(hyb_add16(L[0], L[2]), hyb_add16(L[4], L[6]));
  const float h1 = hyb_add16(hyb_add16(L[1], L[3]), hyb_add16(L[5], L[7]));
  const float S2 = hyb_add16(hyb_add16(h0, h1), 0x1p-13f);
  const float r = hyb_rsq16(S2);
#pragma unroll
  for (int k = 0; k < KC; ++k)
#pragma unroll
    for (int v = 0; v < 8; ++v) y[k][v] = hyb_rn16(x[k][v] * hyb_rn16(r * gp[8 * k + v]));
}

// the attention weights P of the lane's 8 keys from its 8 score chain values (started at B), as FP16
WG_FN v8h hyb_softmax_p(v8f s, unsigned hh) {
#pragma clang fp contract(off)
  const v8f e = hyb_softmax_e8(hyb_round16(s));
  v8f ep;
#pragma unroll
  for (int v = 0; v < 8; ++v) ep[v] = wg_xor16(e[v]);
  const v8f lo = hh == 0 ? e : ep, hi = hh == 0 ? ep : e;
  float A[8];
#pragma unroll
  for (int v = 0; v < 8; ++v) A[v] = hyb_add16(lo[v], hi[v]);   // v = 2 t: a_t, v = 2 t + 1: b_t
  const float a = hyb_add16(hyb_add16(A[0], A[2]), hyb_add16(A[4], A[6]));
  const float b = hyb_add16(hyb_add16(A[1], A[3]), hyb_add16(A[5], A[7]));
  const float w = hyb_rcp16(hyb_add16(a, b));
  v8h p;
#pragma unroll
  for (int v = 0; v < 8; ++v) p[v] = (_Float16)(e[v] * w);
  return p;
}
