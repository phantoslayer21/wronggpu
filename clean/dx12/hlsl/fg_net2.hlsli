// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): float16 operations of net2's D3D12 kernels (clean/dx12/kernels/fg_net2_*.hlsl), with the
// bits of the CPU reference clean/fg/reference/net2.py. Values are float16 bits in the low half of a uint.
//
// - Convolution chunks: the shared exact integer code clean/fg/kernels/fg_exact16.h (exact dot product, one nearest-even
//   rounding per chunk; an exactly zero accumulator is +0), as the reference's fp16acc.
// - fg2_mul: the product of two float16 values is exact in single precision; wg_f2h rounds it once (IEEE signs).
// - fg2_fma: a b + c with one float16 rounding (wg_fma_f16: fused single-precision result made round-to-odd from its exact
//   residual, then rounded once). The single-precision steps never meet a denormal here: every nonzero product or sum of
//   float16 values is at least 2^-48.
// - fg2_leaky, fg2_clamp6, fg2_plus0: compares on the bits (NaN passes through unchanged).
// The including file includes wg_platform.hlsli first.
#pragma once

#define FG_FN
#define FG_I64 int64_t
#define FG_FBH32(v) firstbithigh((uint)(v))
#include "../../fg/kernels/fg_exact16.h"

#define FG2_SLOPE16 0x211Fu      // h(0.01)
#define FG2_ONE16 0x3C00u
#define FG2_TWO16 0x4000u
#define FG2_QUARTER16 0x3400u
#define FG2_SIX16 0x4600u
#define FG2_MSIX16 0xC600u

uint fg2_mul(uint a, uint b) { return wg_f2h(wg_h2f(a) * wg_h2f(b)); }
uint fg2_fma(uint a, uint b, uint c) { return wg_fma_f16(wg_h2f(a), wg_h2f(b), wg_h2f(c)); }
uint fg2_add(uint a, uint b) { return wg_fma_f16(wg_h2f(a), 1.0f, wg_h2f(b)); }

// max(h(s v), v): h(s v) has v's sign, so zeros of different signs never meet
uint fg2_leaky(uint v)
{
    if (fg_is_nan16(v) != 0u) return v;
    const uint t = fg2_mul(FG2_SLOPE16, v);
    return fg_key16(t) > fg_key16(v) ? t : v;
}

uint fg2_clamp6(uint v)
{
    if (fg_is_nan16(v) != 0u) return v;
    if (fg_key16(v) > fg_key16(FG2_SIX16)) return FG2_SIX16;
    if (fg_key16(v) < fg_key16(FG2_MSIX16)) return FG2_MSIX16;
    return v;
}

// h(v + 0): -0 becomes +0
uint fg2_plus0(uint v) { return (v & 0x7FFFu) == 0u ? 0u : v; }

// the half-pixel 2x rule along one axis: output index o of an axis with n input samples
struct Fg2Tap { int i0; int i1; uint w0; uint w1; };

Fg2Tap fg2_up_tap(int o, int n)
{
    Fg2Tap t;
    const float s = ((float)o - 0.5f) * 0.5f;
    t.i0 = (int)s;
    t.i1 = (int)(s + 1.0f);
    if (t.i1 >= n) { t.i0 = n - 1; t.i1 = n - 1; }
    if (s < 0.0f) { t.i0 = 0; t.i1 = 0; }
    t.w1 = wg_f2h(s - (float)t.i0);
    t.w0 = wg_f2h(((float)t.i0 - s) + 1.0f);
    return t;
}
