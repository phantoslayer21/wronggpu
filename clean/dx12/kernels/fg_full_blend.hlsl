// SPDX-License-Identifier: MIT
// Frame generation (issue #96): the Full blend (presets A and B) with net2's input tensors, as
// clean/fg/reference/stages_full.py g4_full_blend, bit for bit. Built only when --only names it.
//
// One workgroup per 16 x 16 tile of the padded frame Wp x Hp (tiles_x = Wp / 16), one lane per pixel in Morton order.
// Frame pixels write X0, X1 (4 x float16) and Xw (float16 0 / 1); every lane's single-precision values (zero outside
// the frame) then go through a lane tree in shared memory, and the first lane of each k x k block writes its net2
// element: Ib, Ic (4 x float16, the means), Ia (float16, 1 if any hole).
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

groupshared float fgb_s[9 * 256];

WG_KERNEL(fg_full_blend)
{
    FGF_SETUP();
    const uint l = wg_lane_id();
    const uint t = wg_block_id();
    const int px = (int)((t % (uint)tiles_x) * 16u + fgf_mx(l));
    const int py = (int)((t / (uint)tiles_x) * 16u + fgf_my(l));
    float v[9];
    [unroll] for (uint c = 0; c < 9u; ++c) v[c] = 0.0f;
    if (px < Wo && py < Ho) {
        const float u = fgf_centre(px, Wo), vv = fgf_centre(py, Ho);
        const float tie = asfloat(0xB4000000u);                         // -2^-23
        const int ix = clamp((int)round(mad(u, (float)W1, -0.5f) + tie), 0, W1 - 1);
        const int iy = clamp((int)round(mad(vv, (float)H1, -0.5f) + tie), 0, H1 - 1);
        const uint2 wa = wg_ld64(ma, (uint)(iy * W1 + ix) * 8u);
        const uint2 wb = wg_ld64(mb, (uint)(iy * W1 + ix) * 8u);
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
        const float2 qb = fgf_bil2h(qc, fgf_tap_uv(ub, vb, Wd, Hd));
        const float2 qa = fgf_bil2h(qc, fgf_tap_uv(ua, va, Wd, Hd));
        float3 x0rgb = mad(w.x, mad(eB, cQ - cP, cP) - P0, P0);
        float x0a = mad(w.x, qb.x - qu.x, qu.x);
        float3 x1rgb = mad(w.y, mad(eA, cP - cQ, cQ) - Q0, Q0);
        float x1a = mad(w.y, qa.y - qu.y, qu.y);
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
        fgf_st4h(x0, pi, float4(x0rgb, x0a));
        fgf_st4h(x1, pi, float4(x1rgb, x1a));
        fgf_st1h(xw, pi, hole ? 1.0f : 0.0f);
        v[0] = x0rgb.x; v[1] = x0rgb.y; v[2] = x0rgb.z; v[3] = x0a;
        v[4] = x1rgb.x; v[5] = x1rgb.y; v[6] = x1rgb.z; v[7] = x1a;
        v[8] = hole ? 1.0f : 0.0f;
    }
    [unroll] for (uint c2 = 0; c2 < 9u; ++c2) fgb_s[c2 * 256u + l] = v[c2];
    const uint kk = (uint)(k * k);
    [loop] for (uint s = 1u; s < kk; s <<= 1u) {
        wg_barrier_lds();
        if ((l & (2u * s - 1u)) == 0u) {
            [unroll] for (uint c3 = 0; c3 < 9u; ++c3) fgb_s[c3 * 256u + l] = fgb_s[c3 * 256u + l] + fgb_s[c3 * 256u + l + s];
        }
    }
    wg_barrier_lds();
    if ((l & (kk - 1u)) == 0u) {
        const int e = (py / k) * (tiles_x * 16 / k) + px / k;
        const float n = (float)kk;
        fgf_st4h(ib, e, float4(wg_div(fgb_s[l], n), wg_div(fgb_s[256u + l], n), wg_div(fgb_s[512u + l], n),
                               wg_div(fgb_s[768u + l], n)));
        fgf_st4h(ic, e, float4(wg_div(fgb_s[1024u + l], n), wg_div(fgb_s[1280u + l], n), wg_div(fgb_s[1536u + l], n),
                               wg_div(fgb_s[1792u + l], n)));
        fgf_st1h(ia, e, fgb_s[2048u + l] > 0.0f ? 1.0f : 0.0f);
    }
}
