// SPDX-License-Identifier: MIT
// Frame generation (issue #96): the Full blend (presets A and B) with net2's input tensors, as
// clean/fg/reference/stages_full.py g4_full_blend, bit for bit. Built only when --only names it.
//
// One workgroup per 16 x 16 tile of the padded frame Wp x Hp (tiles_x = Wp / 16), one lane per pixel in Morton order.
// Frame pixels write X0, X1 (4 x float16) and Xw (float16 0 / 1); every lane's single-precision values (zero outside
// the frame) then go through a lane tree, and the first lane of each k x k block writes its net2 element:
// Ib, Ic (4 x float16, the means), Ia (float16, 1 if any hole). Only cross-wave sums use shared memory.
// Inputs: Cp, Cc, Up, Uc (frame, 4 x float16), W (scatter grid W1 x H1, 2 x float16), Qc (Wd x Hd, 2 x float16), MA, MB
// (scatter grid, 64-bit words). ui != 0: preset B (UI recomposition with Up / Uc), else preset A (Up / Uc not read).
//
// @wg-kernel fg_full_blend co=fg_full.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u
#include "wg_platform.hlsli"
#include "fg_full.hlsli"

WG_ARGS_BEGIN
    uint cp, cc, wimg, qc, ma, mb, up, uc;  // inputs (arena byte offsets)
    uint x0, x1, xw, ib, ic, ia;            // outputs
    int Wo, Ho, W1, H1, Wd, Hd, k, tiles_x, ui;
WG_ARGS_END

groupshared float fgb_s[9 * 8];

float4 fgb_mean(float4 a, uint shift)
{
    // k is a power of two in [1, 16].
    const float scale = asfloat((127u - 2u * shift) << 23u);
    // The general division quiets NaNs without changing the sign or payload.
    return select((asuint(a) & 0x7fffffffu) > 0x7f800000u, asfloat(asuint(a) | 0x00400000u), a * scale);
}

float fgb_add(float a, float b)
{
    // Preserve the lower Morton lane's payload when both operands are NaNs.
    return isnan(a) ? a : a + b;
}

WG_KERNEL(fg_full_blend)
{
    FGF_SETUP();
    const uint l = wg_lane_id();
    const uint t = wg_block_id();
    const uint ty = t / (uint)tiles_x;
    const int2 origin = int2(t - ty * (uint)tiles_x, ty) * 16;
    const int px = origin.x + (int)fgf_mx(l);
    const int py = origin.y + (int)fgf_my(l);
    float v[9];
    [unroll] for (uint c = 0; c < 9u; ++c) v[c] = 0.0f;
    const bool in_bounds = px < Wo && py < Ho;
    if (in_bounds) {
        const float u = fgf_pixel_centre(px, Wo), vv = fgf_pixel_centre(py, Ho);
        const float tie = asfloat(0xB4000000u);                         // -2^-23
        const int ix = clamp((int)round(mad(u, (float)W1, -0.5f) + tie), 0, W1 - 1);
        const int iy = clamp((int)round(mad(vv, (float)H1, -0.5f) + tie), 0, H1 - 1);
        const uint offset = (uint)(iy * W1 + ix) * 8u;
        const uint2 wa = wg_ld64(ma, offset);
        const uint2 wb = wg_ld64(mb, offset);
        const bool emA = wa.x == 0x7FFFFFFFu && wa.y == 0x7FFFFFFFu;
        const bool emB = wb.x == 0x7FFFFFFFu && wb.y == 0x7FFFFFFFu;
        const float ax = emA ? 0.0f : wg_h2f_hi(wa.x), ay = emA ? 0.0f : wg_h2f_lo(wa.x);
        const float bx = emB ? 0.0f : wg_h2f_hi(wb.x), by = emB ? 0.0f : wg_h2f_lo(wb.x);
        const float eA = emA ? 1.0f : 0.0f, eB = emB ? 1.0f : 0.0f;
        const float ub = u + bx, vb = vv + by, ua = u + ax, va = vv + ay;
        const int pi = py * Wo + px;
        const float3 P0 = fgf_ld4h(cp, pi).xyz, Q0 = fgf_ld4h(cc, pi).xyz;
        const float3 cP = fgf_bil4h(cp, fgf_tap_uv(ub, vb, Wo, Ho)).xyz;
        const float3 cQ = fgf_bil4h(cc, fgf_tap_uv(ua, va, Wo, Ho)).xyz;
        const float2 w = fgf_bil2h(wimg, fgf_tap_uv(u, vv, W1, H1));
        const float2 qu = fgf_bil2h(qc, fgf_tap_uv(u, vv, Wd, Hd));
        // Reuse the co-located quality when the warped coordinates are identical.
        float qbx = qu.x, qay = qu.y;
        [branch] if (ub != u || vb != vv) qbx = fgf_bil2h(qc, fgf_tap_uv(ub, vb, Wd, Hd)).x;
        [branch] if (ua != u || va != vv) qay = fgf_bil2h(qc, fgf_tap_uv(ua, va, Wd, Hd)).y;
        float3 x0rgb = mad(w.x, mad(eB, cQ - cP, cP) - P0, P0);
        float x0a = mad(w.x, qbx - qu.x, qu.x);
        float3 x1rgb = mad(w.y, mad(eA, cP - cQ, cQ) - Q0, Q0);
        float x1a = mad(w.y, qay - qu.y, qu.y);
        bool hole = emA && emB && (!(w.x < 0.5f) || !(w.y < 0.5f));     // unordered: NaN counts as >= 0.5
        if (ui != 0) {
            const float4 Up = fgf_ld4h(up, pi), Uc = fgf_ld4h(uc, pi);
            x0rgb = mad(x0rgb, 1.0f - Up.w, Up.xyz);
            x1rgb = mad(x1rgb, 1.0f - Uc.w, Uc.xyz);
            const float e = fgf_sat(wg_div(abs(Up.z - Uc.z) + (abs(Up.x - Uc.x) + abs(Up.y - Uc.y)), 3.0f));
            const float eps = asfloat(0x34000000u);                     // 2^-23
            if (Up.w > eps) x0a = e;
            if (Uc.w > eps) x1a = e;
            hole = hole && Up.w < eps && Uc.w < eps;
        }
        v[0] = x0rgb.x; v[1] = x0rgb.y; v[2] = x0rgb.z; v[3] = x0a;
        v[4] = x1rgb.x; v[5] = x1rgb.y; v[6] = x1rgb.z; v[7] = x1a;
        v[8] = hole ? 1.0f : 0.0f;
    }
    {
        v8f pv;
        [unroll] for (uint c = 0; c < 8u; ++c) pv.v[c] = fgf_opq(v[c]);
        const uint4 h = wg_to_f16_ags(pv);
        if (in_bounds) {
            const uint pi8 = (uint)(py * Wo + px) * 8u;
            wg_st64(x0, pi8, h.xy);
            wg_st64(x1, pi8, h.zw);
        }
    }
    fgf_store_morton_flag(xw, px, py, Wo, Ho, v[8] > 0.0f);
    const uint shift = (uint)firstbithigh((uint)k);
    const uint kk = (uint)(k * k);
    const uint lane = wg_wave_lane();
    // Each surviving lane adds the higher half in Morton order at every stride.
    if (kk >= 4u) {
        [unroll] for (uint c = 0; c < 9u; ++c) {
            v[c] = fgb_add(v[c], QuadReadAcrossX(v[c]));
            v[c] = fgb_add(v[c], QuadReadAcrossY(v[c]));
        }
    }
    if (kk > 4u) {
        [unroll] for (uint c = 0; c < 9u; ++c) {
            v[c] = fgb_add(v[c], WaveReadLaneAt(v[c], lane ^ 4u));
            v[c] = fgb_add(v[c], WaveReadLaneAt(v[c], lane ^ 8u));
        }
    }
    [branch] if (kk > 32u) {
        [unroll] for (uint c = 0; c < 9u; ++c)
            v[c] = fgb_add(v[c], WaveReadLaneAt(v[c], lane ^ 16u));
        if (lane == 0u) {
            [unroll] for (uint c = 0; c < 9u; ++c) fgb_s[c * 8u + (l >> 5)] = v[c];
        }
        wg_barrier_lds();
        if (l < 32u) {
            // The first wave continues the same tree over the eight partial sums.
            const uint part = lane & 7u;
            [unroll] for (uint c = 0; c < 9u; ++c) v[c] = fgb_s[c * 8u + part];
            [loop] for (uint s = 1u; s < (kk >> 5); s <<= 1u) {
                [unroll] for (uint c = 0; c < 9u; ++c) v[c] = fgb_add(v[c], WaveReadLaneAt(v[c], lane ^ s));
            }
        }
    }
    const uint leader_lane = kk > 32u ? l * 32u : l;
    if ((kk <= 32u || l < 8u) && (leader_lane & (kk - 1u)) == 0u) {
        const int bx = origin.x + (int)fgf_mx(leader_lane);
        const int by = origin.y + (int)fgf_my(leader_lane);
        const int e = (by >> shift) * ((tiles_x * 16) >> shift) + (bx >> shift);
        fgf_st4h(ib, e, fgb_mean(float4(v[0], v[1], v[2], v[3]), shift));
        fgf_st4h(ic, e, fgb_mean(float4(v[4], v[5], v[6], v[7]), shift));
        wg_st16(ia, (uint)e * 2u, v[8] > 0.0f ? 0x3c00u : 0u);
    }
}
