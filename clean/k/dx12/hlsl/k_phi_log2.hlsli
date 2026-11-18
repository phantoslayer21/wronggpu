// SPDX-License-Identifier: MIT
// The HDR phi argument is RN32(12 * maxNum(Y, 0) + 1), hence in [1, +inf].
#pragma once

float ki_phi_log2(float x)
{
    const uint bits = asuint(x);
    [branch] if (bits >= 0x3f800000u && bits < 0x7f800000u) {
        const float m0 = asfloat((bits & 0x7fffffu) | 0x3f800000u);
        const bool fold = m0 >= 1.5f;
        const float m = fold ? m0 * 0.5f : m0;
        const float e = (float)((int)(bits >> 23) - 127 + (int)fold);
        const int j = (int)round((m - 1.0f) * 128.0f) + 32;
        const float inv = asfloat(KF_LOG_INV[j]);
        const float p = kf_opq(m * inv);
        const float t = p - 1.0f;
        const float dt = kf_fma(m, inv, -p);

        // log2(1 + t + dt) through t^5, with the first two terms of its derivative for dt.
        const float lead = kf_opq(t * KF2_INV_LN2_H);
        const float lead_tail = kf_fma(t, KF2_INV_LN2_H, -lead);
        const float q = kf_fma(t, kf_fma(t, kf_fma(t, 0.2885390082f, -0.3606737602f),
                                       0.4808983470f), -0.7213475204f);
        const float dt_scaled = kf_fma(dt, kf_fma(-t, KF2_INV_LN2_H, KF2_INV_LN2_H),
                                       kf_fma(t, KF2_INV_LN2_L, lead_tail));
        const float corr = kf_fma(t * t, q, dt_scaled + asfloat(KF_LOG_L[j]));
        const kf2 v = kf2_fast_two_sum(asfloat(KF_LOG_H[j]), lead);
        const kf2 s = kf2_fast_two_sum(e, v.h);
        const kf2 r = kf2_fast_two_sum(s.h, (v.l + corr) + s.l);

        // The exhaustive CPU certificate bounds the mantissa approximation and both additions.
        // Use the smaller half-gap on the lower side of a binary power. Strict compare rejects ties.
        const uint h = asuint(r.h);
        if (h == 0u) return 0.0f;  // x == 1
        const uint exponent = h >> 23;
        const uint down = (asuint(r.l) & ((h & 0x7fffffu) - 1u)) >> 31;
        const float half_gap = asfloat((exponent - 24u - down) << 23);
        const float error = abs(v.h) * asfloat(0x31000000u) + r.h * asfloat(0x2a000000u);
        [branch] if (abs(r.l) + error < half_gap) return r.h;
    }
    return kf_log2(x);
}
