// SPDX-License-Identifier: MIT
#pragma once
// Original spec51.1 exact arithmetic; proof and standalone tests in rr_exact_dot.
// IEEE binary64 -> binary16 RNE directly from significand bits. No f32 narrowing.
uint rr_exact_r16(double x)
{
    uint lo, hi;
    asuint(x, lo, hi);
    const uint sign = (hi >> 16u) & 0x8000u;
    const uint eb = (hi >> 20u) & 0x7ffu;
    const uint mh = hi & 0xfffffu;
    if (eb == 0x7ffu) return sign | ((mh | lo) != 0u ? 0x7e00u : 0x7c00u);
    const int e = int(eb) - 1023;
    if (e > 15) return sign | 0x7c00u;
    if (e < -25) return sign;
    uint q, rem, halfway;
    if (e >= -14) {
        q = (uint(e + 15) << 10u) | (mh >> 10u);
        rem = mh & 0x3ffu;
        halfway = 0x200u;
    } else {
        const uint shiftHi = uint(-e - 4); // full significand shift is 28-e, minus low word's 32 bits
        const uint significandHi = mh | 0x100000u;
        q = significandHi >> shiftHi;
        rem = significandHi & ((1u << shiftHi) - 1u);
        halfway = 1u << (shiftHi - 1u);
    }
    const bool up = rem > halfway || (rem == halfway && (lo != 0u || (q & 1u) != 0u));
    return sign | (q + uint(up)); // carries cover subnormal->normal, binade and infinity boundaries
}

// Finite codes only; caller detects the two NaN codes without relying on payload propagation.
double rr_exact_d8(uint code)
{
    const uint e = (code >> 3u) & 15u, m = code & 7u;
    const uint scaled = e == 0u ? m : ((8u + m) << (e - 1u));
    const double v = double(scaled) * double(0.001953125f);
    return (code & 128u) != 0u ? -v : v;
}

// Original fragment-layout gather based on own gpu/layout.py. All 32 lanes must
// participate. The loops and all WaveReadLaneAt calls stay wave-uniform even
// when a particular output has NaN/infinite C or a NaN operand.
v8f rr_exact_step2(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    const uint lane = WaveGetLaneIndex(), col = lane & 15u, hf = lane >> 4u;
    v8f result;
    [loop] for (uint v = 0; v < 8u; ++v) {
        const uint row = 8u * hf + v;
        precise double sum = double(0.0f);
        bool hasNan = (asuint(c.v[v]) & 0x7fffffffu) > 0x7f800000u;
        [loop] for (uint k = 0; k < 32u; ++k) {
            const uint2 af = k < 16u ? a0 : a1, bf = k < 16u ? b0 : b1;
            const uint j = k & 7u, halfOffset = 16u * ((k >> 3u) & 1u);
            const uint aw = j < 4u ? af.x : af.y, bw = j < 4u ? bf.x : bf.y;
            const uint aa = (WaveReadLaneAt(aw, halfOffset + row) >> (8u * (j & 3u))) & 255u;
            const uint bb = (WaveReadLaneAt(bw, halfOffset + col) >> (8u * (j & 3u))) & 255u;
            hasNan = hasNan || (aa & 127u) == 127u || (bb & 127u) == 127u;
            precise double product = rr_exact_d8(aa) * rr_exact_d8(bb);
            sum = sum + product;
        }
        if (hasNan) result.v[v] = f16tof32(0x7e00u);
        else if ((asuint(c.v[v]) & 0x7fffffffu) == 0x7f800000u) result.v[v] = c.v[v];
        else {
            sum = sum + double(0.0f);
            precise double total = double(c.v[v]) + sum;
            result.v[v] = f16tof32(total == double(0.0f) ? 0u : rr_exact_r16(total));
        }
    }
    return result;
}
