// SPDX-License-Identifier: MIT
// Direct-form (POST_OUT == 3) helpers of post.hlsl: the same values as native_direct.hlsli's colour fetch and output
// scale, with less vector work. Include after native_direct.hlsli and post.hlsl's g_z.
//
// 1. Colour by (column, row) of the dense render rectangle: native_direct_coord divides the dense index again by nd_rw
//    (an integer division per texel); when the dense rectangle is the port's P_rw x P_rh (always on the direct route) the
//    clamped (column, row) are those of the division and the index bound check cannot fire.
// 2. Division by the frame's pre-exposure (wave-uniform divisor): native_direct.hlsli's native_direct_color_div3 (the
//    per-frame reciprocal form, its own exact fallback lanes) at both colour sites (post.hlsl tf_put, pd_normalize).
// 3. Output scale: native_direct_scale's repair test only matters for a zero product of a normal a; its full test runs
//    only when some lane of the wave has one.
#pragma once

// native_direct_raw_color(ui) of dense texel ui = row * P_rw + col (col, row: the clamped column and row) with one texture
// load for both routes: (col, row) itself when the dense rectangle is the port's (wave-uniform), otherwise the
// coordinate of native_direct_coord's division and its bound check (an index past the rectangle gives +0: the load's
// value is not used there). One load site instead of two, so that the kernel's code holds one copy of it per call.
float4 pd_raw_color(uint ui, int col, int row, bool rc)
{
    int2 p = int2(col, row);
    bool ok = true;
    [branch] if (!rc) {
        const uint y = ui / nd_rw;
        p = int2((int)(ui - y * nd_rw), (int)y);
        ok = ui < nd_rw * nd_rh;
    }
    const float3 c = native_direct_color_tex.Load(int3((int)nd_rx + p.x, (int)nd_ry + p.y, 0)).rgb;
    return ok ? float4(c, 0.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// the removed color_divide store's arithmetic boundary (as native_direct_normalized_color)
float3 pd_xor(float3 rgb) { return asfloat(asuint(rgb) ^ (nd_rw >> 31u)); }

// native_direct_color_div3 (native_direct.hlsli) with its exact fallback (native_direct_color_div_ref per channel on the
// lanes outside the window) in one rolled loop over the three channels instead of three inlined copies: the same values,
// less code at each of the kernel's division sites (tf_put and every fallback-colour site)
float3 pd_div3(float3 c)
{
    const float b = nd_color_pi, y = nd_pi_rcp;
    const float3 q0 = c * y;
    const float3 rn = mad(b, q0, -c);
    float3 q = mad(-rn, y, q0);
    const float3 m = abs(c);
    const bool3 fast = and(m < nd_pi_hi, or(m >= nd_pi_lo, m < asfloat(0x00800000u)));
    [branch] if (!all(fast)) {
        float3 a = c;
        uint3 f = uint3(fast);
        [loop] for (uint i = 0u; i < 3u + g_z; ++i) {   // channel i in .x; rotated after each
            const float r = f.x != 0u ? q.x : native_direct_color_div_ref(a.x, b);
            q = float3(q.y, q.z, r);
            a = float3(a.y, a.z, a.x);
            f = uint3(f.y, f.z, f.x);
        }
    }
    return q;
}

// The pre-exposure is not 1 (the division and the output scale run). rc, pi1: the caller's copy (post.hlsl post_body):
// pi1 = the pre-exposure is 1; the (rc, !pi1) copy runs only when it is not 1, so there the test is a literal too
bool pd_div_on(bool rc, bool pi1) { return pi1 ? false : (rc ? true : nd_color_pi != 1.0f); }

float4 pd_normalize(float4 c, bool rc, bool pi1)   // rc, pi1: literals of the caller's copy (pd_div_on)
{
    if (!pd_div_on(rc, pi1)) return c;   // original runtime skips color_divide when pi == 1
    // native_direct.hlsli's per-frame reciprocal form (3 operations per channel, its own exact fallback lanes; big-async
    // 9885bf4): native_direct_color_div_ref's bits
    return float4(pd_xor(pd_div3(c.xyz)), c.w);
}

// native_direct_scale(a, b) of the three output channels where the plain product a * b might differ from it: its repair
// applies only to a zero product (of a normal a), so the lanes with a zero product (conservative: a zero a too) are
// recomputed by the caller (pd_scale_exact) and the plain products serve every other lane
bool pd_scale_check(float3 r)
{
    const float m = asfloat(0x00800000u);   // FLT_MIN: |r| below it is a zero product (FTZ); NaN compares false
    return or(or(abs(r.x) < m, abs(r.y) < m), abs(r.z) < m);
}
float3 pd_scale_exact(float3 a, float b)
{
    float3 t = a;
    [loop] for (uint i = 0u; i < 3u + g_z; ++i) t = float3(t.y, t.z, native_direct_scale(t.x, b));
    return t;
}
