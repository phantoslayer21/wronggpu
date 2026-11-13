// SPDX-License-Identifier: MIT
// c64 Swin block (swin_block_c64_nc) instruction-diet helpers (additive; included by swin_block.hlsl after
// speed_swin.hlsli).
#pragma once
#include "wg_gelu_t5.hlsli"

// The 8 values of a fragment renamed by the permutation i -> (a i + b) mod 8 (a odd): sw_perm8 / sw_unperm8 are inverse
// renamings (no arithmetic). Used to choose the order in which the driver's register allocator sees a loop-carried
// accumulator's values (see SW_REV_PHI in swin_block.hlsl).
v8f sw_perm8(v8f x, uint a, uint b)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = x.v[(a * i + b) & 7u];
    return r;
}
v8f sw_unperm8(v8f x, uint a, uint b)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[(a * i + b) & 7u] = x.v[i];
    return r;
}

// q8(GELU) table loads of 8 values from the arena table (hlsl/wg_gelu_lut.hlsli: same table, slots, entries, fma and
// conversion as wg_gelu_lut_fetch8, so HIP's bytes), with the cheaper offsets of kernels/bneck_gelu.hlsli
// (bn_gelu_fetch8, copied here so the c64 path does not depend on the bneck track's file): the pre-mask's bits 2..4 are
// opaque (0 at run time), so (d >> 2) & 0x7fff is one v_bfe_u32; the stages over the 8 values in order.
wg_gelu_lut_e sw_gelu_fetch8_bn(v8f h)
{
    const uint z = wg_bid_ >> 31;
    const uint msk = 0xffe0ffe0u | (z << 16) | (z * 0x1cu);
    float wv[8];
    [unroll] for (uint i = 0; i < 8; ++i) wv[i] = mad(h.v[i], WG_GELU_LUT_S, WG_GELU_LUT_B);
#if WG_GELU_T5
    // T5 (hlsl/wg_gelu_t5.hlsli): entries indexed by the FP32 bits of w (one shift per value), the same entries
    wg_gelu_lut_e r;
    [unroll] for (uint i5 = 0; i5 < 8; ++i5) r.e[i5] = wg_gelu_t5[wg_gelu_t5_index(wv[i5])];
#else
    uint d[4];
    [unroll] for (uint i1 = 0; i1 < 4; ++i1) d[i1] = f32tof16(wv[2 * i1]) | (f32tof16(wv[2 * i1 + 1]) << 16);
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) d[i2] = d[i2] & msk;
    uint o[8];
    [unroll] for (uint i3 = 0; i3 < 4; ++i3) o[2 * i3] = WG_GELU_LUT_TAB + ((d[i3] >> 2) & 0x7fffu);
    [unroll] for (uint i4 = 0; i4 < 4; ++i4) o[2 * i4 + 1] = WG_GELU_LUT_TAB + (d[i4] >> 18);
    wg_gelu_lut_e r;
    [unroll] for (uint i5 = 0; i5 < 8; ++i5) r.e[i5] = wg_mem.Load2(o[i5]);
#endif
    return r;
}

// The same float through an opaque integer or (a second register holding r; wg_grid_ >> 31 is 0 at run time).
float sw_opaque_copy(float r) { return asfloat(asuint(r) | (wg_grid_ >> 31)); }
// d = (a * r) * g per value (HIP's order), the odd values' product by the copy r2 (= r).
v8f sw_norm_mul8(v8f a, float r, float r2, v8f g)
{
    v8f d;
    [unroll] for (uint v = 0; v < 8; ++v) d.v[v] = a.v[v] * ((v & 1u) ? r2 : r);
    [unroll] for (uint v1 = 0; v1 < 8; ++v1) d.v[v1] = d.v[v1] * g.v[v1];
    return d;
}
