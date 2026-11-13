// SPDX-License-Identifier: MIT
// Shared helpers of the spec 51 kernels (DESIGN-51.md): number formats of 51.1 on the gfx1201 D3D12 path.
// Arithmetic mode and driver rules: clean/dx12/README.md, hlsl/wg_math.hlsli (compile only with clean/rr_f/gpu/build.py,
// which uses clean/dx12/build_dx12.py's exact mode).
#pragma once

// A zero the driver's compiler cannot prove (wg_grid_ < 2^31): opaque fences and rolled loops (clean/dx12 README).
uint rr_zero() { return wg_grid_ >> 31; }
// The same 0 with no known bits for the compiler (0 or 0xffffffff): a fence must hide every bit of the value (with
// rr_zero() the compiler knew that only bit 0 could change and folded the high f16 half of a packed fma into an f32
// v_fma_mix, skipping its f16 rounding).
uint rr_zero_bits() { return 0u - (wg_grid_ >> 31); }
#ifndef RR_FAND
#define RR_FAND 0
#endif
#if RR_FAND
// the same opaque identity as an AND with all ones (0 - 0 complemented): v_and_b32 can pair (VOPD), v_xor_b32 cannot
float rr_fence(float x) { return asfloat(asuint(x) & ~rr_zero_bits()); }
uint rr_fence_u(uint x) { return x & ~rr_zero_bits(); }
#else
float rr_fence(float x) { return asfloat(asuint(x) ^ rr_zero_bits()); }
uint rr_fence_u(uint x) { return x ^ rr_zero_bits(); }
#endif

// ---- Q8 (51.1) -------------------------------------------------------------------------------------------------------
// Q8 of 8 f32 values (each an f16 value or an exact f32): finite values beyond +-448 and +-inf give +-448, a NaN the NaN
// code of its sign. The driver conversion (v_cvt_pk_fp8_f32) rounds to nearest even on the e4m3 grid; it encodes
// |x| > 464 as 0x7f | sign, so a NaN is first replaced by +-1000 of its sign (the conversion itself gives 0xff for
// every NaN). Zeros keep their sign (-0 -> 0x80).
float rr_q8_prep(float x)
{
    const float m = min(max(x, -448.0f), 448.0f);   // v_max_num / v_min_num: a NaN gives -448 here
    const float nanv = asfloat((asuint(x) & 0x80000000u) | 0x447a0000u);   // +-1000
    return select(isnan(x), nanv, m);
}
uint2 rr_q8(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = rr_q8_prep(d.v[i]);
    return wg_pack_q8_nc_ags(d);
}
// Q8 of values known to be finite and within +-448 (no clamp needed).
uint2 rr_q8_small(v8f d) { return wg_pack_q8_nc_ags(d); }

// e4m3 bytes -> f32 (exact; NaN codes give a NaN; v_cvt_pk_f32_fp8 through the driver conversion).
v8f rr_d8(uint2 q) { return wg_v8f_frag(wg_ags_convert(wg_frag2(q), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC)); }

// ---- f16 ------------------------------------------------------------------------------------------------------------
// R(x) of 8 exact f32 values as f32 (v_cvt_f16_f32 RNE through the driver conversion, then the exact widening).
v8f rr_r16(v8f d)
{
    const uint4 h = wg_to_f16_ags(d);
    v8f r;
    r.v[0] = f16tof32(h.x); r.v[1] = f16tof32(h.x >> 16); r.v[2] = f16tof32(h.y); r.v[3] = f16tof32(h.y >> 16);
    r.v[4] = f16tof32(h.z); r.v[5] = f16tof32(h.z >> 16); r.v[6] = f16tof32(h.w); r.v[7] = f16tof32(h.w >> 16);
    return r;
}
#if defined(RR_B16R)
uint4 rr_h16(v8f d)   // clean/rr-f-a14-l01 A40: round toward zero (v_cvt_pkrtz_f16_f32), NOT exact
{
    return uint4(wg_h2(d.v[0], d.v[1]), wg_h2(d.v[2], d.v[3]), wg_h2(d.v[4], d.v[5]), wg_h2(d.v[6], d.v[7]));
}
#else
uint4 rr_h16(v8f d) { return wg_to_f16_ags(d); }
#endif
v8f rr_unh(uint4 h)
{
    v8f r;
    r.v[0] = f16tof32(h.x); r.v[1] = f16tof32(h.x >> 16); r.v[2] = f16tof32(h.y); r.v[3] = f16tof32(h.y >> 16);
    r.v[4] = f16tof32(h.z); r.v[5] = f16tof32(h.z >> 16); r.v[6] = f16tof32(h.w); r.v[7] = f16tof32(h.w >> 16);
    return r;
}
half2 rr_h2(uint v) { return wg_as_h2(v); }
uint rr_b2(half2 h) { return wg_h2_bits(h); }

// ---- borders (51.6) --------------------------------------------------------------------------------------------------
// v' = 0 for N = 1; otherwise P = 2 (N - 1), u = v mod P (non-negative), v' = min(u, P - u).
int rr_reflect(int v, int N)
{
    const int P = 2 * (N - 1);
    const int Ps = max(P, 1);
    int u = v % Ps;                    // truncating remainder
    u = u < 0 ? u + Ps : u;
    const int r = min(u, P - u);
    return N <= 1 ? 0 : r;
}

// ---- window-block constants (51.2), passed as five dwords of two f16 each -----------------------------------------
// wc0 = A_SCALE | A_CLAMP << 16, wc1 = A_CLAMPn | A_P1 << 16, wc2 = A_P0 | ACT_CL << 16, wc3 = ACT_CLn | ACT_C1 << 16,
// wc4 = ACT_C2 | ACT_C3 << 16.
struct RrWc { half2 ascale, aclamp, aclampn, ap1, ap0, actcl, actcln, c1, c2, c3; };
half2 rr_splat_lo(uint d) { return rr_h2((d & 0xffffu) * 0x10001u); }
half2 rr_splat_hi(uint d) { return rr_h2((d >> 16) * 0x10001u); }
RrWc rr_wc(uint w0, uint w1, uint w2, uint w3, uint w4)
{
    RrWc c;
    c.ascale = rr_splat_lo(w0); c.aclamp = rr_splat_hi(w0); c.aclampn = rr_splat_lo(w1); c.ap1 = rr_splat_hi(w1);
    c.ap0 = rr_splat_lo(w2); c.actcl = rr_splat_hi(w2); c.actcln = rr_splat_lo(w3); c.c1 = rr_splat_hi(w3);
    c.c2 = rr_splat_lo(w4); c.c3 = rr_splat_hi(w4);
    return c;
}

// ---- 51.3 step 3: attention weight of two f16 scores (packed), exact ------------------------------------------------
// u = R(s A_SCALE) (v_pk_mul_f16), c = clamp (minNum / maxNum: a NaN u gives A_CLAMPn), inner = R(A_P1 - c c) and
// p = R(c inner + A_P0) (v_pk_fma_f16, one rounding each), then w: for p in [1 + 1/32, 1 + 31/32) (the converter's
// range check) bits(w) = (bits(p) << 5) & 0x7FE0 per half (DESIGN-51.md 8.2), no rounding.
uint rr_attn_w2(uint s2, RrWc k)
{
    const half2 u = rr_h2(s2) * k.ascale;
    const half2 c = min(max(u, k.aclampn), k.aclamp);
    const half2 inner = mad(-c, c, k.ap1);
    const half2 p = mad(c, inner, k.ap0);
    return (rr_b2(p) << 5) & 0x7FE07FE0u;
}

// ---- 51.3 step 8: activation of two f16 values (packed) --------------------------------------------------------------
// k = clamp(x), u = R(ACT_C2 - |k| ACT_C1), v = R(k u + ACT_C3), act = R(x v); the result is fenced so that the driver
// cannot fold the last multiply into a following widening or conversion.
uint rr_act2(uint x2, RrWc c)
{
    const half2 x = rr_h2(x2);
    const half2 k = min(max(x, c.actcln), c.actcl);
    const half2 u = mad(-abs(k), c.c1, c.c2);
    const half2 v = mad(k, u, c.c3);
    return rr_fence_u(rr_b2(x * v));
}
