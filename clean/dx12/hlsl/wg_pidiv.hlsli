// SPDX-License-Identifier: GPL-3.0-only
// The pre-exposure division of the staged native D3D12 route, done by the dedicated _pf kernels (input_encoder_pf,
// emo/input_encoder_mo_pf, and the post_tf _pf buffer forms) when pi != 1 instead of a separate color_divide launch.
// Include after wg_platform.hlsli.
//
// Per channel the value is native_color_div (native_helpers.hlsl: wg_div plus the exact repair of the one tiny quotient
// that rounds to the smallest normal): the staged route's color_divide bits. Like native_direct.hlsli's
// native_direct_color_div3 (the direct route's per-frame form, same proof: clean/runtime/native_direct_pi.h) three
// operations on the lanes whose |a| lies in the window [lo, hi) (or is zero / an FP32 denormal), the reference division on
// the others. The window is native_direct_pi_window(pi) of the host, computed here from the pre-exposure's bits (the host
// folds the division only for a pi whose window is valid: hi != 0): y = RN(1 / pi) is the kernel's own correctly rounded
// quotient, lo = 2^max(-79, B - 125), hi = 2^(B + 127) (+inf for B > 0), B the unbiased exponent of pi. Alpha is not touched
// by the caller (color_divide copies it as raw bits).
#pragma once

// native_color_div (native_helpers.hlsl)
float pidiv_ref(float a, float b)
{
    const float r = wg_div(a, b);
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23u) & 255u, eb = (ub >> 23u) & 255u;
    if ((asuint(r) & 0x7fffffffu) == 0u && ea != 0u && eb != 255u && eb == ea + 127u &&
        (ua & 0x7fffffu) == 0x7fffffu && (ub & 0x7fffffu) == 0u)
        return asfloat(((ua ^ ub) & 0x80000000u) | 0x00800000u);
    return r;
}

struct PiDiv { float b, y, lo, hi; };   // pi, RN(1 / pi), the window's bounds (|a| in [lo, hi) takes the fast form)

PiDiv pidiv_make(uint bits)
{
    PiDiv d;
    d.b = asfloat(bits);
    d.y = wg_div(1.0f, d.b);
    const uint eb = (bits >> 23u) & 255u;
    const bool ok = and(bits < 0x80000000u, and(eb >= 1u, eb <= 252u));
    const int B = (int)eb - 127;
    const int L = max(B - 125, -79);
    d.lo = asfloat((uint)(L + 127) << 23u);
    d.hi = ok ? (B > 0 ? asfloat(0x7f800000u) : asfloat((uint)(B + 254) << 23u)) : 0.0f;
    return d;
}

// c / pi per channel. z: a zero the compiler cannot see (the loop count, so that the three exact-fallback copies stay one
// rolled loop; the caller xors the result with it to keep the removed store's rounding boundary)
float3 pidiv3(float3 c, PiDiv d, uint z)
{
    const float3 q0 = c * d.y;
    const float3 rn = mad(d.b, q0, -c);
    float3 q = mad(-rn, d.y, q0);
    const float3 m = abs(c);
    const bool3 fast = and(m < d.hi, or(m >= d.lo, m < asfloat(0x00800000u)));
    [branch] if (!all(fast)) {
        float3 a = c;
        uint3 f = uint3(fast);
        [loop] for (uint i = 0u; i < 3u + z; ++i) {   // channel i in .x; rotated after each
            const float r = f.x != 0u ? q.x : pidiv_ref(a.x, d.b);
            q = float3(q.y, q.z, r);
            a = float3(a.y, a.z, a.x);
            f = uint3(f.y, f.z, f.x);
        }
    }
    return q;
}
