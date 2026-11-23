// SPDX-License-Identifier: MIT
// Platform layer: the few GPU primitives the kernels use.
//
// Device build (clang -x hip --offload-arch=gfx1201 -nogpuinc): maps to clang's AMDGPU builtins. No HIP or C/C++
// library headers are included (spec 90: the kernels must build with -nogpuinc).
// Host build (-DWG_EMU, plain C++): maps to the lane emulator in emu/wg_emu.h, so the same kernel source can be run
// and tested on a CPU against the NumPy reference.
#pragma once

typedef unsigned char u8;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef decltype(sizeof(0)) size_t;

typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef u32 v4u __attribute__((ext_vector_type(4)));
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(WG_EMU)
typedef _Float16 v4h __attribute__((ext_vector_type(4)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
#endif

#define WG_WAVE 32    // gfx12 and gfx11 kernels run in wave32
#define WG_BLOCK 128  // default workgroup size (4 waves); the Swin block kernels use 256

// gfx11 (RDNA 3): exact target names only. LLVM files gfx1170-1172 under gfx11, but they have the gfx12 matrix design.
// There is no FP8 conversion or matrix instruction (software conversions), the F16 matrix instruction takes 16 K values
// per lane with both halves of the wave alike, and the other half of the wave is read with v_permlanex16.
#if defined(__gfx1100__) || defined(__gfx1101__)
#define WG_GFX11 1
#ifndef WG_SOFT_FP8
#define WG_SOFT_FP8 1
#endif
// The faster exact software conversion (below). -DWG_SLOW_Q8 (or -DWG_SLOW_Q8=1) selects the plain form again (timing
// comparisons); -DWG_SLOW_Q8=0 the faster one.
#if !defined(WG_SLOW_Q8) || !WG_SLOW_Q8
#ifndef WG_FAST_Q8
#define WG_FAST_Q8 1
#endif
#endif
#endif
#if defined(WG_FAST_Q8) && (defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__))
#error "the gfx11 FP32 -> E4M3 conversion needs IEEE arithmetic (it adds and subtracts a constant, and tests for NaN): no -ffast-math or -ffinite-math-only"
#endif

#if defined(__AMDGCN__)

#define WG_FN __attribute__((device)) __attribute__((always_inline)) inline
#define WG_FN_NOINLINE __attribute__((device)) __attribute__((noinline))
#define WG_KERNEL_N(name, threads) \
  extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(threads, threads))) void name
// Same, and ask the register allocator to leave room for at least `waves` resident waves per SIMD.
#define WG_KERNEL_NW(name, threads, waves) \
  extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(threads, threads))) \
  __attribute__((amdgpu_waves_per_eu(waves))) void name
// Same with a hard register cap (amdgpu_num_vgpr): the allocator must fit `vgprs` VGPRs (it spills if it has to);
// amdgpu_waves_per_eu alone is only a hint the allocator may give up on.
#define WG_KERNEL_NWV(name, threads, waves, vgprs) \
  extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(threads, threads))) \
  __attribute__((amdgpu_waves_per_eu(waves))) __attribute__((amdgpu_num_vgpr(vgprs))) void name
#define WG_KERNEL(name) WG_KERNEL_N(name, WG_BLOCK)
#define WG_LDS __attribute__((shared))

WG_FN u32 wg_lane_id() { return __builtin_amdgcn_workitem_id_x(); }
WG_FN u32 wg_block_id() { return __builtin_amdgcn_workgroup_id_x(); }

#if defined(WG_GFX11)
// gfx11 has no FP8 matrix instruction, and its F16 one takes 16 K values per lane (both halves of the wave alike) where
// the gfx12 one takes 8: wg_wmma_fp8 and wg_wmma_f16 are made from this instruction below (the code is shared with the
// emulator). D = A x B + C, 16x16x16, FP16 inputs, FP32 accumulate (v_wmma_f32_16x16x16_f16): lane l holds row (A) or
// column (B) l % 16 with K indices 0..15 in order; element v of D in lane l is row 2 v + l / 16, column l % 16.
WG_FN v8f wg_wmma16_f16(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }

// The 32-bit value of the same variable in lane (lane ^ 16), i.e. the other half of the wave: one v_permlanex16_b32,
// which reads the other 16-lane row of the same 32 lanes; selectors 0x76543210 / 0xfedcba98 take its lane i for lane i.
// fi: a lane of the other half that is not executing is read as well, as ds_bpermute does.
WG_FN u32 wg_xchg16_u(u32 v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, true, false); }
WG_FN float wg_xor16(float v) { return __builtin_bit_cast(float, wg_xchg16_u(__builtin_bit_cast(u32, v))); }
#else
// D = A x B + C, 16x16x16, FP8 E4M3 inputs, FP32 accumulate (v_wmma_f32_16x16x16_fp8_fp8).
WG_FN v8f wg_wmma_fp8(v2i a, v2i b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c); }
// D = A x B + C, 16x16x16, FP16 inputs, FP32 accumulate (v_wmma_f32_16x16x16_f16).
WG_FN v8f wg_wmma_f16(v8h a, v8h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c); }

// Value of the same variable in lane (lane ^ 16), i.e. the other half of the wave.
WG_FN float wg_xor16(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((wg_lane_id() ^ 16u) << 2),
                                                                 __builtin_bit_cast(int, v)));
}
#endif

// Workgroup barrier with LDS visibility (release before, acquire after).
WG_FN void wg_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

// Workgroup barrier for kernels whose waves hand each other values only through LDS (no global store of one wave is
// read by another wave of the workgroup before the kernel ends). The fences cover the LDS only ("local"), so the
// acquire side does not invalidate the vector cache (no global_inv after the barrier). WG_LDS_BARRIER=0: wg_barrier().
#ifndef WG_LDS_BARRIER
#define WG_LDS_BARRIER 1
#endif
WG_FN void wg_barrier_lds() {
#if WG_LDS_BARRIER
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
  wg_barrier();
#endif
}

// Scheduling hint: the next `size` instructions of the class `mask` go here, in order (see LLVM IGroupLP:
// 0x002 vector ALU, 0x008 matrix, 0x100 DS read, 0x400 transcendental). Used to interleave matrix and vector work.
#define WG_SCHED_VALU 0x0002
#define WG_SCHED_WMMA 0x0008
#define WG_SCHED_TRANS 0x0400
#define WG_SCHED(mask, size) __builtin_amdgcn_sched_group_barrier(mask, size, 0)  // mask and size must be constants
#define WG_SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)  // nothing is moved across this point

WG_FN float wg_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
WG_FN float wg_rsq(float x) { return __builtin_amdgcn_rsqf(x); }
WG_FN float wg_rcp(float x) { return __builtin_amdgcn_rcpf(x); }
WG_FN float wg_rint(float x) { return __builtin_rintf(x); }
WG_FN float wg_fmax(float a, float b) { return __builtin_fmaxf(a, b); }
WG_FN float wg_fmin(float a, float b) { return __builtin_fminf(a, b); }

#else  // WG_EMU: host emulation

#include "emu/wg_emu.h"

#endif

// ---------------------------------------------------------------------------------------------------------------
// The matrix wrappers of gfx11 (and of its emulator build, -DWG_EMU_GFX11). Operands keep the gfx12 shape (a lane holds
// 8 K values of its half of the wave), so kernels and weight packing stay as they are: the lane fetches the 8 values of
// the other half (one half-wave exchange per dword), widens the 16 to F16 and calls the F16 matrix instruction. K slot
// (h, j) is index 8 h + j: a lane of half 1 puts the fetched values first, so both halves hold the 16 values in the same
// order. (The instruction reads the operands of the odd D rows from lanes 16-31 and those of the even rows from lanes
// 0-15, so the sum would be the same without the select; its rounding is not: K order is part of the arithmetic.)
#if defined(WG_GFX11) || defined(WG_EMU_GFX11)
typedef u32 v8u __attribute__((ext_vector_type(8)));

// Two E4M3 codes (bytes 0 and 1 of c) as F16 bits scaled by 2^-8: the 7 magnitude bits under the F16 exponent (bias 15,
// E4M3 has bias 7) read as value / 2^8, and E4M3 subnormals become F16 subnormals of the same scale. The NaN code 0x7f
// (magnitude bits all ones, 0x3f80) gets bit 14 as well, 0x7f80: an F16 NaN, which the multiply by 256 keeps. (A compare
// and select per code costs enough registers to make the Swin block kernels spill; this form costs about 3.)
WG_FN u32 wg_e4m3x2_scaled(u32 c) {
  const u32 b0 = c & 0xffu, b1 = (c >> 8) & 0xffu;
  const u32 m0 = b0 & 0x7fu, m1 = b1 & 0x7fu;
  const u32 h0 = (m0 << 7) | ((b0 & 0x80u) << 8) | (((m0 + 1u) & 0x80u) << 7);
  const u32 h1 = (m1 << 7) | ((b1 & 0x80u) << 8) | (((m1 + 1u) & 0x80u) << 7);
  return h0 | (h1 << 16);
}
// 4 E4M3 codes (a dword) as 4 F16 values, exactly (all 256 codes are exact in F16): the multiply by 256 is exact.
WG_FN v4h wg_widen_e4m3x4(u32 w) {
  const v2h k = {(_Float16)256.0f, (_Float16)256.0f};
  const v2h lo = __builtin_bit_cast(v2h, wg_e4m3x2_scaled(w)) * k;
  const v2h hi = __builtin_bit_cast(v2h, wg_e4m3x2_scaled(w >> 16)) * k;
  v4h r = {lo[0], lo[1], hi[0], hi[1]};
  return r;
}
WG_FN bool wg_in_upper_half() { return (wg_lane_id() & 16u) != 0; }

WG_FN v16h wg_k16_codes(v2i p) {
  const u32 x0 = (u32)p[0], x1 = (u32)p[1];
  const u32 y0 = wg_xchg16_u(x0), y1 = wg_xchg16_u(x1);
  const bool hi = wg_in_upper_half();
  const u32 f0 = hi ? y0 : x0, f1 = hi ? y1 : x1;  // indices 0..7
  const u32 s0 = hi ? x0 : y0, s1 = hi ? x1 : y1;  // indices 8..15
  const v4h a = wg_widen_e4m3x4(f0), b = wg_widen_e4m3x4(f1), c = wg_widen_e4m3x4(s0), d = wg_widen_e4m3x4(s1);
  v16h r = {a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3]};
  return r;
}
WG_FN v16h wg_k16_f16(v8h p) {
  const v4u x = __builtin_bit_cast(v4u, p);
  v4u y;
#pragma unroll
  for (int i = 0; i < 4; ++i) y[i] = wg_xchg16_u(x[i]);
  const bool hi = wg_in_upper_half();
  v8u r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[i] = hi ? y[i] : x[i];
    r[4 + i] = hi ? x[i] : y[i];
  }
  return __builtin_bit_cast(v16h, r);
}
// D = A x B + C, 16x16x16, FP8 E4M3 inputs (8 codes per lane), FP32 accumulate. The accumulator rows come out as
// 2 v + h (packing.py: d_row "gfx11").
WG_FN v8f wg_wmma_fp8(v2i a, v2i b, v8f c) { return wg_wmma16_f16(wg_k16_codes(a), wg_k16_codes(b), c); }
// The same for FP16 inputs (8 halves per lane).
WG_FN v8f wg_wmma_f16(v8h a, v8h b, v8f c) { return wg_wmma16_f16(wg_k16_f16(a), wg_k16_f16(b), c); }
// One operand already holds its 16 F16 values in K order (swin.hip WG_F16_W: the weights, widened once per workgroup);
// the other is made as above.
WG_FN v8f wg_wmma_fp8(v16h a, v2i b, v8f c) { return wg_wmma16_f16(a, wg_k16_codes(b), c); }
WG_FN v8f wg_wmma_fp8(v2i a, v16h b, v8f c) { return wg_wmma16_f16(wg_k16_codes(a), b, c); }
// Both operands already hold their 16 F16 values in K order (swin.hip WG_F16_A: an activation made once and used many times).
WG_FN v8f wg_wmma_fp8(v16h a, v16h b, v8f c) { return wg_wmma16_f16(a, b, c); }
WG_FN v8f wg_wmma_f16(v8h a, v16h b, v8f c) { return wg_wmma16_f16(wg_k16_f16(a), b, c); }
// 8 E4M3 codes (one lane's fragment) as 8 F16 values: the half of a K group that one lane of a half-wave contributes.
WG_FN v8h wg_widen_e4m3x8(v2i p) {
  const v4h a = wg_widen_e4m3x4((u32)p[0]), b = wg_widen_e4m3x4((u32)p[1]);
  v8h r = {a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
  return r;
}

// ---- E4M3 rounding of FP32 values as FP32 arithmetic (the method is explained with wg_f32_to_e4m3_t, below), delivered as
// F16 operands without code bytes (swin.hip WG_F16_Q); also what the faster code conversion (WG_FAST_Q8) is made from ----
// v_med3_f32 clamps, because fminf / fmaxf get a canonicalising v_max in front of them on gfx11.
WG_FN float wg_q8m_clamp(float x, float lo, float hi) {
#if defined(__AMDGCN__)
  return __builtin_amdgcn_fmed3f(x, lo, hi);
#else
  return x < lo ? lo : (x > hi ? hi : x);
#endif
}
WG_FN u32 wg_q8m_bits(float f) { return __builtin_bit_cast(u32, f); }
WG_FN float wg_q8m_flt(u32 u) { return __builtin_bit_cast(float, u); }

// C = 1.5 * 2^(E + 20) for |x| (E clamped to [-6, 8]), as FP32 bits.
WG_FN u32 wg_q8m_magic(float ax) {
  return (wg_q8m_bits(wg_q8m_clamp(ax, 0.015625f, 448.0f)) & 0x7f800000u) + 0x0a400000u;
}


// Two floats as two F16 (v_cvt_pk_rtz_f16_f32); exact for the values here (at most 4 significant bits).
WG_FN u32 wg_cvt_pk_f16(float a, float b) {
#if defined(__AMDGCN__)
  return __builtin_bit_cast(u32, __builtin_amdgcn_cvt_pkrtz(a, b));
#else
  const v2h r = {(_Float16)a, (_Float16)b};
  return __builtin_bit_cast(u32, r);
#endif
}
// The E4M3-rounded value as F16 bits (every E4M3 value is exact in F16), the next operand of the F16 matrix instruction.
// CL: saturate at +-448 (also infinity); NANFIX: a NaN gives an F16 NaN. v_cvt_pk_rtz_f16_f32 is exact here: the value has
// at most 4 significant bits. The sign of a zero result is the input's.
template <bool NANFIX, bool CL>
WG_FN float wg_q8m_round(float f) {
  const float ax = __builtin_fabsf(f);
  const float c = wg_q8m_flt(wg_q8m_magic(ax));
  float r = (ax + c) - c;
  if (CL) r = wg_q8m_clamp(r, 0.0f, 448.0f);
  u32 rb = wg_q8m_bits(r);
  if (NANFIX) rb = (f != f) ? 0x7fc00000u : rb;
  return wg_q8m_flt(rb | (wg_q8m_bits(f) & 0x80000000u));
}
template <bool NANFIX, bool CL>
WG_FN v8h wg_q8_to_f16_t(v8f d) {
  u32 w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    w[i] = wg_cvt_pk_f16(wg_q8m_round<NANFIX, CL>(d[2 * i]), wg_q8m_round<NANFIX, CL>(d[2 * i + 1]));
  return __builtin_bit_cast(v8h, w);
}
WG_FN v8h wg_q8_to_f16(v8f d) { return wg_q8_to_f16_t<true, true>(d); }         // a NaN gives an F16 NaN
WG_FN v8h wg_q8_to_f16_inner(v8f d) { return wg_q8_to_f16_t<false, true>(d); }  // a NaN gives +-0
WG_FN v8h wg_q8_to_f16_nc(v8f d) { return wg_q8_to_f16_t<false, false>(d); }    // |x| <= 464, not NaN
#endif

// ---------------------------------------------------------------------------------------------------------------
// FP8 E4M3FN conversion in software (round to nearest even, saturating at +-448, NaN -> NaN).
// Written from the format definition: bias 7, 3 mantissa bits, max finite 0x7E = 448, NaN 0x7F, no infinities.

#if defined(WG_FAST_Q8)
// gfx11: the same conversion (round to nearest even, saturating at +-448, NaN -> sign|0x7f, FP32 denormals -> a signed
// zero) with fewer instructions than the plain form below (-DWG_SLOW_Q8), and the same rounding delivered as F16 bits for
// the next F16 matrix operand. Checked against the definition over all 2^32 FP32 patterns (runner/q8_exhaustive.py).
// With z = |x| clamped to [2^-6, 448] and C = 1.5 * 2^(E + 20), E the exponent of z, the FP32 sum |x| + C rounds |x| to a
// multiple of 2^(E - 3) (ties to even: the mantissa of C is even) and the difference sum - C is exact; the floor 2^-6 gives
// the E4M3 subnormal step 2^-9. The compiler must keep (|x| + C) - C (no reassociation: no fast-math, see above).
// E4M3 code bytes. The code comes straight from the rounded sum: bits(sum) - bits(C) is the index on the grid and
// code = 8 (E + 6) + index = (bits(C) >> 20) - 1132 + index; min(., 0x7e) saturates (also +-inf). NANFIX: a NaN gives
// sign|0x7f, else it gives sign|0x7e.
template <bool NANFIX>
WG_FN u32 wg_f32_to_e4m3_t(float f) {
  const u32 b = wg_q8m_bits(f);
  const float ax = __builtin_fabsf(f);
  const u32 cb = wg_q8m_magic(ax);
  u32 c = (wg_q8m_bits(ax + wg_q8m_flt(cb)) - cb) + (cb >> 20) - 1132u;
  c = c < 0x7eu ? c : 0x7eu;
  if (NANFIX) c = (f != f) ? 0x7fu : c;
  return c | ((b >> 24) & 0x80u);
}
WG_FN u32 wg_f32_to_e4m3(float f) { return wg_f32_to_e4m3_t<true>(f); }

template <bool NANFIX>
WG_FN v2i wg_pack_q8_t(v8f d) {
  u32 w[2];
#pragma unroll
  for (int h = 0; h < 2; ++h)
    w[h] = wg_f32_to_e4m3_t<NANFIX>(d[4 * h]) | (wg_f32_to_e4m3_t<NANFIX>(d[4 * h + 1]) << 8) |
           (wg_f32_to_e4m3_t<NANFIX>(d[4 * h + 2]) << 16) | (wg_f32_to_e4m3_t<NANFIX>(d[4 * h + 3]) << 24);
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}
// Quantize 8 floats to FP8 and pack them into 8 bytes (element i -> byte i, little-endian).
WG_FN v2i wg_pack_q8(v8f d) { return wg_pack_q8_t<true>(d); }
// For values that stay inside the kernel: a NaN becomes +-448, as in the gfx12 form.
WG_FN v2i wg_pack_q8_inner(v8f d) { return wg_pack_q8_t<false>(d); }
// For inputs proven within +-464 and not NaN: the bytes of wg_pack_q8.
WG_FN v2i wg_pack_q8_nc(v8f d) { return wg_pack_q8_t<false>(d); }

#else
WG_FN u32 wg_f32_to_e4m3(float f) {
  // Branch-free: every path is computed and the result picked with selects (no divergent control flow).
  const u32 b = __builtin_bit_cast(u32, f);
  const u32 s = (b >> 24) & 0x80u;
  const u32 a = b & 0x7fffffffu;
  // Subnormal range |f| < 2^-6: step 2^-9; result 0..8 (8 is the smallest normal, 2^-6).
  const u32 sub = (u32)wg_rint(wg_fmin(__builtin_bit_cast(float, a), 0.015625f) * 512.0f);
  // Normal range: round the 23-bit mantissa to 3 bits (ties to even), rebias the exponent 127 -> 7.
  const u32 nrm = ((a + 0x7ffffu + ((a >> 20) & 1u)) >> 20) - (120u << 3);
  u32 code = a < 0x3c800000u ? sub : nrm;  // 0x3c800000 = 2^-6
  code = a >= 0x43e00000u ? 0x7eu : code;  // 0x43e00000 = 448: saturate (also +-inf)
  code = a > 0x7f800000u ? 0x7fu : code;   // NaN
  return s | code;
}
#endif

WG_FN float wg_e4m3_to_f32(u32 c) {
  const u32 em = c & 0x7fu;
  const float nrm = __builtin_bit_cast(float, (em + (120u << 3)) << 20);
  const float sub = (float)em * (1.0f / 512.0f);
  float v = em >= 8u ? nrm : sub;
  v = em == 0x7fu ? __builtin_bit_cast(float, 0x7fc00000u) : v;
  return __builtin_bit_cast(float, ((c & 0x80u) << 24) | __builtin_bit_cast(u32, v));
}

#if defined(__AMDGCN__) && !defined(WG_SOFT_FP8)
// Hardware FP8 conversions (v_cvt_pk_fp8_f32 rounds to nearest even; v_cvt_f32_fp8 decodes E4M3FN). The encoder
// does not saturate by itself, so values are clamped to +-448 first (NaN kept). Checked on the GPU against the
// reference q8 by runner/selftest.py (results/2026-10-05-swin-selftest.md; probe_fp8_pack for these exact
// functions). Build with -DWG_SOFT_FP8 (build.py --soft-fp8) to use the software conversions instead.
#ifndef WG_SAT_MINMAX
#define WG_SAT_MINMAX 1  // measured: -0.035 ms per 1440p-Q frame, bit-identical for finite values
#endif
#if WG_SAT_MINMAX
// IEEE-754-2019 maximum/minimum propagate a NaN, so the clamp needs no separate NaN test (v_maximumminimum_f32).
WG_FN float wg_sat448(float x) { return __builtin_elementwise_minimum(__builtin_elementwise_maximum(x, -448.0f), 448.0f); }
#else
WG_FN float wg_sat448(float x) { return x != x ? x : __builtin_fminf(__builtin_fmaxf(x, -448.0f), 448.0f); }
#endif
// Same clamp without the NaN pass-through (2 instructions less per value; a NaN becomes +-448). Used for values that
// stay inside the kernel; see wg_pack_q8_inner.
WG_FN float wg_sat448_inner(float x) { return __builtin_fminf(__builtin_fmaxf(x, -448.0f), 448.0f); }

WG_FN v2i wg_pack_q8(v8f d) {
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448(d[0]), wg_sat448(d[1]), 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448(d[2]), wg_sat448(d[3]), lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448(d[4]), wg_sat448(d[5]), 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448(d[6]), wg_sat448(d[7]), hi, true);
  v2i r;
  r[0] = lo;
  r[1] = hi;
  return r;
}

// Conversion without any clamp, for packs whose input is proven to stay within +-460 (see packing.py: block_bounds, and
// the flags word of the packed weights). Finite values in that range give exactly the bytes of wg_pack_q8.
WG_FN v2i wg_pack_q8_nc(v8f d) {
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(d[0], d[1], 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(d[2], d[3], lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(d[4], d[5], 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(d[6], d[7], hi, true);
  v2i r;
  r[0] = lo;
  r[1] = hi;
  return r;
}

// wg_pack_q8 for values that stay inside the kernel (the inputs of the next matrix multiply). A NaN is not passed
// through here (it becomes +-448), but it cannot hide: a NaN that enters through the block input or the residual
// stream stays NaN in the FP32 accumulators and reaches the stored output, which is packed with wg_pack_q8. Spec 09
// counts non-finite outputs there. Finite inputs give exactly the same bytes as wg_pack_q8 (checked by selftest.py).
WG_FN v2i wg_pack_q8_inner(v8f d) {
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448_inner(d[0]), wg_sat448_inner(d[1]), 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448_inner(d[2]), wg_sat448_inner(d[3]), lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448_inner(d[4]), wg_sat448_inner(d[5]), 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(wg_sat448_inner(d[6]), wg_sat448_inner(d[7]), hi, true);
  v2i r;
  r[0] = lo;
  r[1] = hi;
  return r;
}

WG_FN v8f wg_unpack_fp8(v2i p) {
  v8f r;
  r[0] = __builtin_amdgcn_cvt_f32_fp8(p[0], 0);
  r[1] = __builtin_amdgcn_cvt_f32_fp8(p[0], 1);
  r[2] = __builtin_amdgcn_cvt_f32_fp8(p[0], 2);
  r[3] = __builtin_amdgcn_cvt_f32_fp8(p[0], 3);
  r[4] = __builtin_amdgcn_cvt_f32_fp8(p[1], 0);
  r[5] = __builtin_amdgcn_cvt_f32_fp8(p[1], 1);
  r[6] = __builtin_amdgcn_cvt_f32_fp8(p[1], 2);
  r[7] = __builtin_amdgcn_cvt_f32_fp8(p[1], 3);
  return r;
}
#else
// Software conversions (host emulator, or -DWG_SOFT_FP8).

#if !defined(WG_FAST_Q8)
// Quantize 8 floats to FP8 and pack them into 8 bytes (element i -> byte i, little-endian).
WG_FN v2i wg_pack_q8(v8f d) {
  u32 lo = wg_f32_to_e4m3(d[0]) | (wg_f32_to_e4m3(d[1]) << 8) | (wg_f32_to_e4m3(d[2]) << 16) |
           (wg_f32_to_e4m3(d[3]) << 24);
  u32 hi = wg_f32_to_e4m3(d[4]) | (wg_f32_to_e4m3(d[5]) << 8) | (wg_f32_to_e4m3(d[6]) << 16) |
           (wg_f32_to_e4m3(d[7]) << 24);
  v2i r;
  r[0] = (int)lo;
  r[1] = (int)hi;
  return r;
}

WG_FN v2i wg_pack_q8_inner(v8f d) { return wg_pack_q8(d); }
WG_FN v2i wg_pack_q8_nc(v8f d) { return wg_pack_q8(d); }
#endif

WG_FN v8f wg_unpack_fp8(v2i p) {
  v8f r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r[i] = wg_e4m3_to_f32(((u32)p[0] >> (8 * i)) & 0xffu);
    r[4 + i] = wg_e4m3_to_f32(((u32)p[1] >> (8 * i)) & 0xffu);
  }
  return r;
}
#endif

WG_FN v8h wg_to_f16(v8f d) {
  v8h r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = (_Float16)d[i];
  return r;
}

WG_FN v8f wg_splat(float x) {
  v8f r;
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = x;
  return r;
}

WG_FN v8f wg_load8f(const float* p) {
  const v4f* q = (const v4f*)p;
  v4f a = q[0], b = q[1];
  v8f r;
  r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
  r[4] = b[0]; r[5] = b[1]; r[6] = b[2]; r[7] = b[3];
  return r;
}

// ---- timing instrumentation (WG_PROFILE builds only; see swin.hip and runner/profile_stages.py) ----
// The global 100 MHz realtime clock (one tick = 10 ns), the hardware id of the SIMD a wave runs on, a drain of all
// outstanding loads / LDS operations / stores, and a use of a vector value in a scalar register (forces the producer,
// e.g. a WMMA, to be complete). s_memtime and the shader-cycles register do not exist on gfx12 in this toolchain.
#if defined(__AMDGCN__) && defined(WG_GFX11)
// The counters of an instrumented (WG_PROFILE) build use gfx12 instructions (s_wait_kmcnt, hwreg 29, ...).
#define WG_NO_GFX11_PROFILE(name) \
  __attribute__((device)) name __attribute__((error("WG_PROFILE timing counters are gfx12 only (not gfx11 / RDNA 3)")))
WG_NO_GFX11_PROFILE(u64 wg_realtime());
WG_NO_GFX11_PROFILE(u32 wg_cycles());
WG_NO_GFX11_PROFILE(u32 wg_hwid());
WG_NO_GFX11_PROFILE(void wg_drain());
WG_FN int wg_uniform(int v) { return __builtin_amdgcn_readfirstlane(v); }
WG_FN void wg_sync_i(int v) {
  const int s = __builtin_amdgcn_readfirstlane(v);
  __asm__ volatile("" : : "s"(s) : "memory");
}
#elif defined(__AMDGCN__)
WG_FN u64 wg_realtime() {
  u64 t;
  __asm__ volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0x0" : "=s"(t) : : "memory");
  return t;
}
WG_FN u32 wg_cycles() {  // shader cycle counter, low 32 bits (a scalar read, no message traffic)
  u32 r;
  __asm__ volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(r));
  return r;
}
WG_FN int wg_uniform(int v) { return __builtin_amdgcn_readfirstlane(v); }
WG_FN u32 wg_hwid() {
  u32 r;
  __asm__ volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(r));
  return r;
}
WG_FN void wg_drain() { __asm__ volatile("s_wait_loadcnt 0x0\n\ts_wait_dscnt 0x0\n\ts_wait_storecnt 0x0" : : : "memory"); }
WG_FN void wg_sync_i(int v) {
  const int s = __builtin_amdgcn_readfirstlane(v);
  __asm__ volatile("" : : "s"(s) : "memory");
}
#else
// host emulation: counters that advance on every read, so the instrumentation can be tested
WG_FN u64 wg_realtime() { static u64 t = 0; return t += 3; }
WG_FN u32 wg_hwid() { return 0; }
WG_FN u32 wg_cycles() { static u32 t = 0; return t += 5; }
WG_FN int wg_uniform(int v) { return v; }
WG_FN void wg_drain() {}
WG_FN void wg_sync_i(int) {}
#endif
WG_FN void wg_sync(float v) { wg_sync_i(__builtin_bit_cast(int, v)); }
WG_FN void wg_sync(int v) { wg_sync_i(v); }
WG_FN void wg_sync(v8f v) { wg_sync(v[0]); }

// ---- FP8 dot product on the vector ALU (WG_NORM_DOT4): c + sum of 4 products of FP8 bytes, accumulated in FP32 ----
#if defined(__AMDGCN__) && defined(WG_GFX11)
__attribute__((device)) float wg_dot4_fp8(int a, int b, float c)
    __attribute__((error("WG_NORM_DOT4 needs v_dot4_f32_fp8_fp8, which gfx11 (RDNA 3) does not have")));
#elif defined(__AMDGCN__)
WG_FN float wg_dot4_fp8(int a, int b, float c) {
  float r;
  __asm__("v_dot4_f32_fp8_fp8 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}
#else
WG_FN float wg_dot4_fp8(int a, int b, float c) {
  v2i pa = {a, 0}, pb = {b, 0};
  const v8f x = wg_unpack_fp8(pa), y = wg_unpack_fp8(pb);
  return c + ((x[0] * y[0] + x[1] * y[1]) + (x[2] * y[2] + x[3] * y[3]));
}
#endif

// ---- helpers for the packed FP16 softmax (WG_SMAX16) ----
#if defined(__AMDGCN__)
WG_FN v2h wg_exp2_h2(v2h x) { return __builtin_elementwise_exp2(x); }  // two v_exp_f16
WG_FN float wg_dot2_acc(v2h a, v2h b, float c) {                       // c + a0 b0 + a1 b1 in FP32
  float r;
  __asm__("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}
#else
WG_FN v2h wg_exp2_h2(v2h x) {
  v2h r;
  r[0] = (_Float16)__builtin_exp2f((float)x[0]);
  r[1] = (_Float16)__builtin_exp2f((float)x[1]);
  return r;
}
WG_FN float wg_dot2_acc(v2h a, v2h b, float c) { return c + (float)a[0] * (float)b[0] + (float)a[1] * (float)b[1]; }
#endif
WG_FN v2h wg_splat2(float x) { v2h r = {(_Float16)x, (_Float16)x}; return r; }

// ---- packed FP16 hidden path (WG_GELU_MODE 3): GELU of 8 FP32 values and their FP8 conversion, returns the packed bytes.
// Coefficients p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 (t = u^2) fitted by the clean team to minimize FP8 flips
// against the exact GELU with the FP16 arithmetic below simulated (clean/reference/swin.py: gelu_poly16).
// The polynomial is halved (Phi/2 = 0.25 + u p'(u^2), p' = p / 2) so that y = x Phi comes out of one v_dot2_f32_f16 with
// the same half of x and of Phi/2 in both slots: dot2((x, x), (Phi/2, Phi/2)) = 2 x Phi/2, exact in FP32, and it also
// replaces the FP16 -> FP32 conversion.
#define WG_P16_INV_T 0.33333334f
#define WG_P16_C0 0.5985f
#define WG_P16_C1 -0.8731f
#define WG_P16_C2 1.0131f
#define WG_P16_C3 -0.69135f
#define WG_P16_C4 0.20285f
#if defined(__AMDGCN__)
WG_FN v2h wg_h2(float a, float b) { return __builtin_bit_cast(v2h, __builtin_amdgcn_cvt_pkrtz(a, b)); }  // round toward zero
#else
WG_FN v2h wg_h2(float a, float b) {  // host emulation of v_cvt_pkrtz_f16_f32: round toward zero
  auto rtz = [](float f) {
    _Float16 h = (_Float16)f;
    float back = (float)h;
    if ((f > 0.0f && back > f) || (f < 0.0f && back < f)) {
      unsigned short bits;
      __builtin_memcpy(&bits, &h, 2);
      --bits;  // one step toward zero in magnitude
      __builtin_memcpy(&h, &bits, 2);
    }
    return h;
  };
  v2h r;
  r[0] = rtz(a);
  r[1] = rtz(b);
  return r;
}
#endif
#if defined(__AMDGCN__)
// y = 2 * a[HI] * b[HI] in FP32: one v_dot2_f32_f16 whose two multiplies both take the same half (operand select
// modifiers; the compiler's own shuffle lowering costs 3 extra instructions per value, so this is inline assembly).
template <int HI>
WG_FN float wg_dot_sel(v2h a, v2h b) {
  float r;
  if (HI) __asm__("v_dot2_f32_f16 %0, %1, %2, 0 op_sel:[1,1,0] op_sel_hi:[1,1,1]" : "=v"(r) : "v"(a), "v"(b));
  else __asm__("v_dot2_f32_f16 %0, %1, %2, 0 op_sel_hi:[0,0,1]" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
#else
template <int HI>
WG_FN float wg_dot_sel(v2h a, v2h b) { return 2.0f * (float)a[HI] * (float)b[HI]; }
#endif

// Hidden path of a tile: the first KP pairs of the 8 values by the packed FP16 polynomial (no transcendental), the rest by
// the sigmoid form of mode 2 (2 transcendentals per value, which run in the transcendental unit beside the vector ALU).
// KP = 4 is mode 3, KP < 4 is mode 4 (WG_GELU_MIX = KP): a mix that balances the two units (see rates.py).
WG_FN float wg_gelu_sigmoid(float x);
template <bool SAFE, int KP>
WG_FN v2i wg_hidden_mix(v8f h) {
  const v2h inv = {(_Float16)WG_P16_INV_T, (_Float16)WG_P16_INV_T};
  const v2h one = {(_Float16)1.0f, (_Float16)1.0f}, mone = {(_Float16)-1.0f, (_Float16)-1.0f};
  const v2h quarter = {(_Float16)0.25f, (_Float16)0.25f}, top = {(_Float16)448.0f, (_Float16)448.0f};
  const v2h c0 = {(_Float16)WG_P16_C0, (_Float16)WG_P16_C0}, c1 = {(_Float16)WG_P16_C1, (_Float16)WG_P16_C1};
  const v2h c2 = {(_Float16)WG_P16_C2, (_Float16)WG_P16_C2}, c3 = {(_Float16)WG_P16_C3, (_Float16)WG_P16_C3};
  const v2h c4 = {(_Float16)WG_P16_C4, (_Float16)WG_P16_C4};
  v8f y;
#pragma unroll
  for (int i = 0; i < KP; ++i) {
    const v2h x = wg_h2(h[2 * i], h[2 * i + 1]);
    v2h u = x * inv;
    u = __builtin_elementwise_min(__builtin_elementwise_max(u, mone), one);
    const v2h t = u * u;
    v2h p = __builtin_elementwise_fma(t, c4, c3);
    p = __builtin_elementwise_fma(t, p, c2);
    p = __builtin_elementwise_fma(t, p, c1);
    p = __builtin_elementwise_fma(t, p, c0);
    const v2h r = __builtin_elementwise_fma(u, p, quarter);       // Phi / 2
    const v2h xc = SAFE ? x : __builtin_elementwise_min(x, top);  // y <= 448 (Phi <= 1.0007)
    y[2 * i] = wg_dot_sel<0>(xc, r);
    y[2 * i + 1] = wg_dot_sel<1>(xc, r);
  }
#pragma unroll
  for (int v = 2 * KP; v < 8; ++v) y[v] = wg_gelu_sigmoid(h[v]);
  return (SAFE || KP == 4) ? wg_pack_q8_nc(y) : wg_pack_q8_inner(y);
}
template <bool SAFE>
WG_FN v2i wg_hidden16(v8f h) { return wg_hidden_mix<SAFE, 4>(h); }

// GELU x * Phi(x). WG_GELU_MODE selects the implementation:
//   2  (default) sigmoid form x / (1 + 2^(-z log2 e)), z = x (a + b x^2), (a, b) fitted to minimize FP8 flips: 1.3-1.7 %
//      of the FP8 results differ from the exact GELU, none by more than one step; 5 VALU + 2 transcendentals instead of
//      9 + 2. Accepted by the spec team (whole-frame rule of spec 09 draft 2, results/2026-10-05-swin-02d4620.md).
//   3  packed FP16 polynomial, no transcendental: Phi(x) ~ 0.5 + u p(u^2), u = clamp(x / 3, -1, 1), p of degree 4, all in
//      v_pk_*_f16 (2 values per instruction); y = x Phi. 2-6 % of the FP8 results differ from the exact GELU, < 0.9 % by
//      more than one step (N(0,1)..N(0,4)); about half the vector cycles of mode 2. See wg_hidden16.
//   4  mix: WG_GELU_MIX pairs of every 4 by the mode-3 polynomial, the others by mode 2 (see wg_hidden_mix)
//   1  A&S 7.1.25 erfc (3 terms): max abs error 2.6e-5, 0.05-0.14 % of the FP8 results differ, none by more than one step
//   0  exact: A&S 7.1.26 erfc, max abs error 4.7e-7, 0.001 % of the FP8-rounded results differ from the exact GELU
#ifndef WG_GELU_MODE
#define WG_GELU_MODE 2
#endif
#ifndef WG_GELU_MIX
#define WG_GELU_MIX 1
#endif
#if defined(WG_ABLATE_GELU) && WG_ABLATE_GELU
WG_FN float wg_gelu(float x) { return x; }  // timing-only ablation
#elif WG_GELU_MODE == 2 || WG_GELU_MODE == 4
WG_FN float wg_gelu(float x) {
  // z log2(e) with z = x (1.59989589 + 0.06951594 x^2): 2.30816186 and 0.10029031 are those coefficients times log2(e)
  const float tt = __builtin_fmaf(x * x, -0.10029031f, -2.30816186f);
  return x * wg_rcp(1.0f + wg_exp2(x * tt));
}
#elif WG_GELU_MODE == 1
// With Q = Phi(-|x|) = 0.5 erfc(|x| / sqrt2): GELU(x) = 0.5 x + |x| (0.5 - Q). erfc(z) ~ t (a1 + t (a2 + a3 t)) exp(-z^2),
// t = 1 / (1 + p z) (Abramowitz & Stegun, Handbook of Mathematical Functions, 7.1.25); the 0.5 of Q is folded into
// the coefficients and z^2 = x^2 / 2 into the exp2 argument.
WG_FN float wg_gelu(float x) {
  const float ax = __builtin_fabsf(x);
  const float t = wg_rcp(__builtin_fmaf(ax, 0.47047f * 0.70710678118654752f, 1.0f));
  float poly = 0.5f * 0.7478556f;
  poly = __builtin_fmaf(poly, t, 0.5f * -0.0958798f);
  poly = __builtin_fmaf(poly, t, 0.5f * 0.3480242f);
  const float e = wg_exp2(x * x * -0.72134752044448170f);
  const float r = __builtin_fmaf(-(poly * t), e, 0.5f);
  return __builtin_fmaf(ax, r, 0.5f * x);
}
#else
// Standard GELU x * Phi(x). With Q = Phi(-|x|) = 0.5 erfc(|x| / sqrt2): GELU(x) = 0.5 x + |x| (0.5 - Q) (for x < 0 this
// is x Q, for x > 0 it is x - x Q), which needs neither a sign test nor a max. erfc from the approximation 7.1.26 of
// Abramowitz & Stegun (Handbook of Mathematical Functions, 1964): erfc(z) ~ t (a1 + t (a2 + ... a5 t)) exp(-z^2),
// t = 1 / (1 + p z), |error| < 1.5e-7. The factor 0.5 of Q is folded into the polynomial coefficients; z^2 = x^2 / 2
// is folded into the exp2 argument. Max abs error 4.7e-7 in FP32 (0.0006 % of the FP8-rounded results differ from the
// exact GELU, none by more than one step; measured over 1.3e7 values).
WG_FN float wg_gelu(float x) {
  const float ax = __builtin_fabsf(x);
  const float t = wg_rcp(__builtin_fmaf(ax, 0.3275911f * 0.70710678118654752f, 1.0f));
  float poly = 0.5f * 1.061405429f;
  poly = __builtin_fmaf(poly, t, 0.5f * -1.453152027f);
  poly = __builtin_fmaf(poly, t, 0.5f * 1.421413741f);
  poly = __builtin_fmaf(poly, t, 0.5f * -0.284496736f);
  poly = __builtin_fmaf(poly, t, 0.5f * 0.254829592f);
  const float e = wg_exp2(x * x * -0.72134752044448170f);  // -0.5 log2(e): exp(-z^2) = 2^(-x^2 log2(e) / 2)
  const float r = __builtin_fmaf(-(poly * t), e, 0.5f);    // 0.5 - Phi(-|x|)
  return __builtin_fmaf(ax, r, 0.5f * x);
}
#endif

// The sigmoid GELU of mode 2, also in builds with another WG_GELU_MODE (used by the mixed hidden path).
WG_FN float wg_gelu_sigmoid(float x) {
  const float tt = __builtin_fmaf(x * x, -0.10029031f, -2.30816186f);
  return x * wg_rcp(1.0f + wg_exp2(x * tt));
}
