// SPDX-License-Identifier: MIT
// Model K (spec/40-model-k.md): FP16 helpers shared by the model K kernels.
//
// Everything in model K is FP16 (spec 40, "Numerics"). Elementwise operations round to FP16 after every operation, in
// the order the spec writes them; matrix products use v_wmma_f32_16x16x16_f16 (FP16 inputs, FP32 accumulation) and
// round the result to FP16 once, or, in the FP16-accumulating kernels (k_stage*h_*, a spec 49 speed approximation),
// v_wmma_f16_16x16x16_f16 (wgk_wmma16 below: one FP16 rounding per instruction).
//
// WMMA mapping used by every model K kernel ("transposed" form, as in the model M kernels): the weights are the A
// operand (rows = output channels), the tokens are the B operand (columns = tokens). With the gfx12 wave32 layout
//   A: lane l holds row l % 16, K elements 8 (l / 16) + 0..7
//   B: lane l holds column l % 16, K elements 8 (l / 16) + 0..7
//   D: lane l holds column l % 16, rows drow(l / 16, 0..7) (gfx12: 8 (l / 16) + 0..7; the weight packing absorbs any
//      other row order, see clean/k/runner/k_pack.py)
// a lane owns one token and 8 contiguous channels of it in both B and D. So a token's channels are always read and
// written as 16-byte vectors, and the FP16 result of one product is directly the B operand of the next one.
//
// Built for gfx1201 with clang's AMDGPU builtins (-nogpuinc, spec 90), and for the host lane emulator with -DWG_EMU.
#pragma once

#include "wg_platform.h"

typedef _Float16 h16;
typedef _Float16 v4h __attribute__((ext_vector_type(4)));

// Note (clang 18): __builtin_bit_cast(T, v[i]) of a vector element reads element 0 whatever i is (seen on x86; the
// frontend is shared with the GPU build). Copy the element into a scalar first, then bit-cast the scalar.

// ---- 16-byte fragments -------------------------------------------------------------------------------------------

WG_FN v8h wgk_ld8(const h16* p) { return *(const v8h*)p; }  // p 16-byte aligned
WG_FN void wgk_st8(h16* p, v8h v) { *(v8h*)p = v; }
WG_FN v8h wgk_zero8() {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = (h16)0.0f;
  return r;
}
WG_FN v8f wgk_zero8f() { return wg_splat(0.0f); }

// Value of v in lane (lane ^ 16): the other half of the wave, which holds the other 8 channels of the same token.
WG_FN float wgk_xor16(float v) { return wg_xor16(v); }

// FP32 accumulators -> FP16, round to nearest even (v_cvt_f16_f32 / v_cvt_pk_rtz is NOT used: it truncates).
#if defined(__AMDGCN__) && defined(WGK_PAIR_CVT) && WGK_PAIR_CVT
// GPU: two values per dword without the v_pack_b32_f16 the compiler adds to every pair of v_cvt_f16_f32 (a third of
// the conversion instructions of the FP32-accumulating stage kernels): v_cvt_f16_f32 writes the low half, and
// v_fma_mixhi_f16 x * 1 + (-0) writes the high half and keeps the low one. x * 1 + (-0) is x exactly (signed zeros
// included), rounded once to FP16 to nearest even with the FP16 denormal mode of v_cvt_f16_f32: the same bits.
WG_FN u32 wgk_cvt2h(float lo, float hi) {
  u32 r;
  __asm__("v_cvt_f16_f32 %0, %1" : "=v"(r) : "v"(lo));
  __asm__("v_fma_mixhi_f16 %0, %1, 1.0, neg(0)" : "+v"(r) : "v"(hi));
  return r;
}
WG_FN v8h wgk_round8(v8f d) {
  v4u r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r[i] = wgk_cvt2h(d[2 * i], d[2 * i + 1]);
  return __builtin_bit_cast(v8h, r);
}
#else
WG_FN v8h wgk_round8(v8f d) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = (h16)d[i];
  return r;
}
#endif

// ---- exact FP16 elementwise operations -----------------------------------------------------------------------------
// Each helper is one IEEE FP16 operation, rounded to nearest even. The explicit casts keep the host build exact as
// well: on x86 _Float16 arithmetic may be evaluated in float ("excess precision") until the next cast or assignment;
// a single +, - or * of two FP16 values is exact in float or rounds once there, and rounding that float to FP16
// gives the correctly rounded FP16 result (float has 24 >= 2 * 11 + 2 bits, so the double rounding is harmless).
// Contraction into FMA is switched off: an FMA would round once where the spec rounds twice.

WG_FN h16 wgk_add(h16 a, h16 b) {
#pragma clang fp contract(off)
  return (h16)(a + b);
}
WG_FN h16 wgk_sub(h16 a, h16 b) {
#pragma clang fp contract(off)
  return (h16)(a - b);
}
WG_FN h16 wgk_mul(h16 a, h16 b) {
#pragma clang fp contract(off)
  return (h16)(a * b);
}

// ---- the activation of spec 40 (exact) ----------------------------------------------------------------------------
//
//   t = min(max(x, -2), 2)
//   p = f16(a - f16(b |t|))      a = 0.412109375 (0x3698), b = 0.0810546875 (0x2d30)
//   q = f16(0.5 + f16(t p))
//   act(x) = f16(x q)
//
// The clamp uses IEEE maxNum / minNum (v_max_f16 / v_min_f16): a NaN input gives t = -2, q = 0 and act = NaN * 0 =
// NaN, the same result as a NaN-propagating clamp. act(+inf) = +inf, act(-inf) = -inf * 0 = NaN (as the formula says),
// act(x) = -0 for finite x <= -2. Verified bit-exactly over all 65536 inputs (clean/k/tests/test_k_act.py, and on the
// GPU by stage "k.act" of the runner).

#define WGK_ACT_A ((h16)0.412109375f)
#define WGK_ACT_B ((h16)0.0810546875f)

// Clamps without the compiler's canonicalization. For __builtin_elementwise_max / min (maxNum / minNum) of a value the
// compiler cannot prove canonical (the result of inline asm such as wgk_cvt2h below, or of a WMMA) it first emits
// v_pk_max_num_f16 x, x, x, which only quiets a signalling NaN (FP16 denormals are kept). No value these kernels clamp
// can be a signalling NaN (they are results of conversions, WMMAs or FP16 arithmetic, which return quiet NaNs), so the
// canonicalization is the identity here; the clamp itself is the same v_pk_max_num_f16 / v_pk_min_num_f16 with the
// same constant: the same bits, one VALU instruction fewer per pair and clamp (the MLP activation, the softmax clamp,
// the embedding ReLU). WGK_ASM_CLAMP=0: the builtins.
#ifndef WGK_ASM_CLAMP
#define WGK_ASM_CLAMP 1
#endif
#if defined(__AMDGCN__) && WGK_ASM_CLAMP
#define WGK_HAVE_ASM_CLAMP 1
// min(max(x, lo), hi) per half, lo / hi splatted constants (any FP16 value: a literal operand)
WG_FN v2h wgk_clamp2(v2h x, v2h lo, v2h hi) {
  u32 m, r;
  __asm__("v_pk_max_num_f16 %0, %1, %2" : "=v"(m) : "v"(__builtin_bit_cast(u32, x)), "s"(__builtin_bit_cast(u32, lo)));
  __asm__("v_pk_min_num_f16 %0, %1, %2" : "=v"(r) : "v"(m), "s"(__builtin_bit_cast(u32, hi)));
  return __builtin_bit_cast(v2h, r);
}
WG_FN v2h wgk_maxc2(v2h x, v2h lo) {
  u32 r;
  __asm__("v_pk_max_num_f16 %0, %1, %2" : "=v"(r) : "v"(__builtin_bit_cast(u32, x)), "s"(__builtin_bit_cast(u32, lo)));
  return __builtin_bit_cast(v2h, r);
}
#else
#define WGK_HAVE_ASM_CLAMP 0
#endif

WG_FN h16 wgk_act(h16 x) {
#pragma clang fp contract(off)
  h16 t = __builtin_elementwise_max(x, (h16)-2.0f);
  t = __builtin_elementwise_min(t, (h16)2.0f);
  const h16 at = __builtin_elementwise_abs(t);
  const h16 bt = (h16)(WGK_ACT_B * at);
  const h16 p = (h16)(WGK_ACT_A - bt);
  const h16 tp = (h16)(t * p);
  const h16 q = (h16)((h16)0.5f + tp);
  return (h16)(x * q);
}

// Two values at once: packed FP16 (v_pk_max_f16, v_pk_mul_f16, v_pk_add_f16 ...), same operations and rounding.
WG_FN v2h wgk_act2(v2h x) {
#if defined(__AMDGCN__)
#pragma clang fp contract(off)
  const v2h lo = {(h16)-2.0f, (h16)-2.0f}, hi = {(h16)2.0f, (h16)2.0f};
  const v2h a = {WGK_ACT_A, WGK_ACT_A}, b = {WGK_ACT_B, WGK_ACT_B}, half = {(h16)0.5f, (h16)0.5f};
#if WGK_HAVE_ASM_CLAMP
  const v2h t = wgk_clamp2(x, lo, hi);
#else
  v2h t = __builtin_elementwise_max(x, lo);
  t = __builtin_elementwise_min(t, hi);
#endif
  const v2h at = __builtin_elementwise_abs(t);
  const v2h bt = b * at;
  const v2h p = a - bt;
  const v2h tp = t * p;
  const v2h q = half + tp;
  return x * q;
#else  // host: vector _Float16 arithmetic may keep excess precision; use the scalar form
  v2h r;
  r[0] = wgk_act(x[0]);
  r[1] = wgk_act(x[1]);
  return r;
#endif
}

WG_FN v8h wgk_act8(v8h x) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    v2h p = {x[i], x[i + 1]};
    p = wgk_act2(p);
    r[i] = p[0];
    r[i + 1] = p[1];
  }
  return r;
}

// Elementwise FP16 add of two fragments (packed adds on the GPU).
WG_FN v8h wgk_add8(v8h a, v8h b) {
#if defined(__AMDGCN__)
#pragma clang fp contract(off)
  return a + b;
#else
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = wgk_add(a[i], b[i]);
  return r;
#endif
}

// Fragment versions of the exact elementwise operations (packed FP16 on the GPU, the exact scalar form on the host).
#if defined(__AMDGCN__)
WG_FN v8h wgk_mul8(v8h a, v8h b) {
#pragma clang fp contract(off)
  return a * b;
}
WG_FN v8h wgk_sub8(v8h a, v8h b) {
#pragma clang fp contract(off)
  return a - b;
}
#else
WG_FN v8h wgk_mul8(v8h a, v8h b) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = wgk_mul(a[i], b[i]);
  return r;
}
WG_FN v8h wgk_sub8(v8h a, v8h b) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = wgk_sub(a[i], b[i]);
  return r;
}
#endif
WG_FN v8h wgk_splat8h(h16 v) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = v;
  return r;
}
WG_FN v8h wgk_max8(v8h a, v8h b) { return __builtin_elementwise_max(a, b); }
WG_FN v8h wgk_min8(v8h a, v8h b) { return __builtin_elementwise_min(a, b); }
// min(max(x, lo), hi) and max(x, lo) with splatted constants (the asm clamps above on the GPU; same bits)
WG_FN v8h wgk_clamp8c(v8h x, h16 lo, h16 hi) {
#if WGK_HAVE_ASM_CLAMP
  const v2h l = {lo, lo}, h = {hi, hi};
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const v2h p = wgk_clamp2((v2h){x[i], x[i + 1]}, l, h);
    r[i] = p[0];
    r[i + 1] = p[1];
  }
  return r;
#else
  return wgk_min8(wgk_max8(x, wgk_splat8h(lo)), wgk_splat8h(hi));
#endif
}
WG_FN v8h wgk_maxc8(v8h x, h16 lo) {
#if WGK_HAVE_ASM_CLAMP
  const v2h l = {lo, lo};
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const v2h p = wgk_maxc2((v2h){x[i], x[i + 1]}, l);
    r[i] = p[0];
    r[i + 1] = p[1];
  }
  return r;
#else
  return wgk_max8(x, wgk_splat8h(lo));
#endif
}
// FP16 fragment -> FP32 accumulator (exact), e.g. a bias that a matrix product starts from.
#if defined(__AMDGCN__) && defined(WGK_PAIR_CVT) && WGK_PAIR_CVT
// GPU: the high half by v_fma_mix_f32 x_hi * 1 + (-0) (exact) instead of a shift and v_cvt_f32_f16.
WG_FN v8f wgk_widen8(v8h a) {
  const v4u u = __builtin_bit_cast(v4u, a);
  v8f r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float lo, hi;
    __asm__("v_cvt_f32_f16 %0, %1" : "=v"(lo) : "v"(u[i]));
    __asm__("v_fma_mix_f32 %0, %1, 1.0, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(hi) : "v"(u[i]));
    r[2 * i] = lo;
    r[2 * i + 1] = hi;
  }
  return r;
}
#else
WG_FN v8f wgk_widen8(v8h a) {
  v8f r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = (float)a[i];
  return r;
}
#endif

// ---- FP16-accumulating matrix product (spec 49 "Speed approximations") ----------------------------------------------
// D = f16(A x B + C), 16x16x16, FP16 inputs and accumulator (v_wmma_f16_16x16x16_f16). Measured on the RX 9070 XT:
// the instruction rounds to nearest even once per instruction (1 + 0.5009 ulp -> +1 ulp, ties -> even), the same
// throughput as the FP32-accumulating form. A chain of n instructions therefore rounds n times instead of once (the
// reference itself keeps its running sums in FP16, spec 41). The host emulator sums the product in FP32 (the
// emulator's WMMA) and rounds once per call.
WG_FN v8h wgk_wmma16(v8h a, v8h b, v8h c) {
#if defined(__AMDGCN__)
  return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c);
#else
  return wgk_round8(wg_wmma_f16(a, b, wgk_widen8(c)));
#endif
}

// Accumulator of one matrix-product class: KAcc<false> = FP32 (spec 41), KAcc<true> = FP16 (the approximation above).
// start(b): the accumulator starting at the FP16 fragment b (a bias); out(c): the FP16 result.
template <bool H>
struct KAcc {
  typedef v8f T;
  static WG_FN T start(v8h b) { return wgk_widen8(b); }
  static WG_FN T zero() { return wgk_zero8f(); }
  static WG_FN T mma(v8h a, v8h b, T c) { return wg_wmma_f16(a, b, c); }
  static WG_FN v8h out(T c) { return wgk_round8(c); }
};
template <>
struct KAcc<true> {
  typedef v8h T;
  static WG_FN T start(v8h b) { return b; }
  static WG_FN T zero() { return wgk_zero8(); }
  static WG_FN T mma(v8h a, v8h b, T c) { return wgk_wmma16(a, b, c); }
  static WG_FN v8h out(T c) { return c; }
};

// FP16-accumulating product classes (template parameter F16 of k_block / k_block_wide): one bit per class. A set bit
// runs that class with KAcc<true> (the accumulator starts at the FP16 bias and its result feeds the next product
// directly). KF_SM: the softmax weights in FP16 (w = f16(2^g) by v_exp_f16, sum in FP32, A = f16(f16(w s) 2^-7) with
// s = f16(2^7 / sum)).
enum {
  KF_IN = 1,       // embedding / expand
  KF_QKV = 2,      // q, k, v
  KF_SC = 4,       // scores q.k + B
  KF_Z = 8,        // z = A v
  KF_WO = 16,      // output projection
  KF_W1 = 32,      // MLP W1
  KF_W2 = 64,      // MLP W2
  KF_MERGE = 128,  // 2x2 merge (K = 4C, up to 512: the longest sums)
  KF_SM = 256,     // softmax weights in FP16
  KF_HEAD = 512,   // output head
  KF_ALL = 1023,   // all of the above
  // the default of the FP16-accumulating kernels: all but the merge, which keeps FP32 accumulation (it lowers the
  // whole-network error at about the same speed; clean/k/README.md "Speed approximations")
  KF_DEFAULT = KF_ALL & ~KF_MERGE
};
// The mask the FP16-accumulating kernels are built with (override: -DWGK_F16ACC=<mask>, for measurements).
#ifndef WGK_F16ACC
#define WGK_F16ACC KF_DEFAULT
#endif

// 2^x of 8 FP16 values, FP16 result (v_exp_f16; it equals the correctly rounded table on all 65536 inputs, see the
// output stage in clean/k/README.md), and c + the sum of 8 FP16 values in FP32 (v_dot2_f32_f16 with 1, 1).
#if defined(__AMDGCN__)
WG_FN v8h wgk_exp2h8(v8h x) { return __builtin_elementwise_exp2(x); }
WG_FN float wgk_sum8h(v8h x, float c) {
  const v2h one = {(h16)1.0f, (h16)1.0f};
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const v2h p = {x[i], x[i + 1]};
    c = __builtin_amdgcn_fdot2(p, one, c, false);
  }
  return c;
}
#else
WG_FN v8h wgk_exp2h8(v8h x) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = (h16)__builtin_exp2((double)x[i]);
  return r;
}
WG_FN float wgk_sum8h(v8h x, float c) {
#pragma unroll
  for (int i = 0; i < 8; i += 2) c += (float)x[i] + (float)x[i + 1];
  return c;
}
#endif

// Softmax weights of one query (lane = query, 4 fragments of 8 keys: the clamped polynomial g of spec 41 already
// applied) -> the A^T fragments pa. SMH: FP16 weights (KF_SM), else FP32 weights as spec 41 writes them
// (w = 2^g in FP32, A = f16(w (1 / sum w))). The sum over 64 keys is 32 per lane plus one lane exchange.
template <bool SMH>
WG_FN void wgk_softmax_weights(const v8h (&gg)[4], v8h (&pa)[4]) {
  float sum = 0.0f;
  if (SMH) {  // w = f16(2^g) (|g| <= 14.43: 2^-14.43 <= w <= 2^14.43), FP32 sum
    v8h wh[4];
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      wh[st] = wgk_exp2h8(gg[st]);
      sum = wgk_sum8h(wh[st], sum);
    }
    sum += wg_xor16(sum);
    // A = f16(f16(w s) 2^-7) with s = f16(2^7 / sum): 64 2^-14.43 <= sum <= 64 2^14.43 keeps s in the normal FP16
    // range (2^-13.43 .. 2^15.43), and w s <= 2^7; the scaling by 2^-7 is exact
    const v8h s = wgk_splat8h((h16)(128.0f * wg_rcp(sum))), lo = wgk_splat8h((h16)0.0078125f);
#pragma unroll
    for (int st = 0; st < 4; ++st) pa[st] = wgk_mul8(wgk_mul8(wh[st], s), lo);
  } else {
    float w[4][8];
#if !(defined(__AMDGCN__) && defined(WGK_PAIR_CVT) && WGK_PAIR_CVT)
#pragma unroll
    for (int st = 0; st < 4; ++st)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        w[st][j] = wg_exp2((float)gg[st][j]);
        sum += w[st][j];
      }
    sum += wg_xor16(sum);
    const float inv = wg_rcp(sum);
#pragma unroll
    for (int st = 0; st < 4; ++st)
#pragma unroll
      for (int j = 0; j < 8; ++j) pa[st][j] = (h16)(w[st][j] * inv);
#else  // the same operations with the pair conversions
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const v8f gf = wgk_widen8(gg[st]);  // exact
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        w[st][j] = wg_exp2(gf[j]);
        sum += w[st][j];
      }
    }
    sum += wg_xor16(sum);
    const float inv = wg_rcp(sum);
    // A = f16(w inv) rounded once from the exact product, as the compiled form of the scalar code above has always
    // done it (v_fma_mixlo_f16 / v_fma_mixhi_f16 w * inv + 0; w > 0): the bits of the default network.
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      v4u r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        u32 d;
        __asm__("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(d) : "v"(w[st][2 * i]), "v"(inv));
        __asm__("v_fma_mixhi_f16 %0, %1, %2, 0" : "+v"(d) : "v"(w[st][2 * i + 1]), "v"(inv));
        r[i] = d;
      }
      pa[st] = __builtin_bit_cast(v8h, r);
    }
#endif
  }
}

// ss + the sum of f16(x_c^2) over the 8 channels of x, in channel order, each addition in FP32 (the block's
// normalization). f16(x^2) of an FP16 x is one FP16 multiply (the FP32 square of an FP16 value is exact, so rounding it
// to FP16 is the correctly rounded FP16 product). GPU with WGK_PAIR_CVT: packed FP16 squares, and each addition as
// v_fma_mix_f32 x2_half * 1 + ss (one rounding of the exact sum: the FP32 addition), no conversions or shifts.
WG_FN float wgk_sumsq8(v8h x, float ss) {
#if defined(__AMDGCN__) && defined(WGK_PAIR_CVT) && WGK_PAIR_CVT
  const v4u u = __builtin_bit_cast(v4u, wgk_mul8(x, x));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __asm__("v_fma_mix_f32 %0, %1, 1.0, %0 op_sel_hi:[1,0,0]" : "+v"(ss) : "v"(u[i]));
    __asm__("v_fma_mix_f32 %0, %1, 1.0, %0 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "+v"(ss) : "v"(u[i]));
  }
#else
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float v = (float)x[j];
    ss += (float)(h16)(v * v);  // f16(x_c^2)
  }
#endif
  return ss;
}

// Mirror about the edge token, edge not repeated (spec 41): -1 -> 1, n -> n - 2; repeated for far positions.
WG_FN int wgk_mirror(int p, int n) {
  if (p >= 0 && p < n) return p;
  if (n == 1) return 0;
  const int period = 2 * (n - 1);
  int r = p % period;
  if (r < 0) r += period;
  return r < n ? r : period - r;
}
// floor(a / 2) for any sign (spec 41 draft 3: floor semantics)
WG_FN int wgk_floor2(int a) { return a >= 0 ? a / 2 : -((1 - a) / 2); }

// ---- lane exchange ---------------------------------------------------------------------------------------------------

// Value of v in lane (lane ^ 16): the other half of the wave, which holds the other 8 channels of the same token.

#if !defined(WG_EMU)
#define WG_EMU_REGISTER(name)
#endif
