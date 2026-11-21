// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: R3, downsample to the net1 grid with the UI estimate and the colour alpha
// (clean/fg/reference/stages_common.py downsample_ui). One lane per d x d block (d = FG_D, 2 or 4) of the frame:
// every frame pixel of the block writes Cc (and Uc in case U, Lc with Lite flag 2 in cases A / H); blocks inside the
// net1 grid write Dc (block mean of the source colour), Ld (luma-change minima: min, or the mean of the four 2 x 2
// minima for d = 4) and Um (any UI flag, cases A / H). Pixels outside the frame contribute zeros. D3D12 only.
// flags: bit 0 F1 (UI-recomposition path or Lite flag 1), 1 UI alpha active, 2 hudless active, 3 Lite flag 1,
// 4 Lite flag 2.
//
// @wg-kernel fg_lite_downsample2 co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u defines=FG_D=2
// @wg-kernel fg_lite_downsample4 co=- entry=fg_lite_downsample block=256 wave=32 args=p,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,p,u,u,u,u,u,u,u,u,u,u,u,u,u,u,p,p,p,p,p,p,p,u,u,u,u,u defines=FG_D=4
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

WG_ARGS_BEGIN
    uint colour, cw, cbx, cby;
    uint hud, hw, hh;
    FG_REGION_ARGS(h)
    uint alpha, aw, ah, akind, ach;
    FG_REGION_ARGS(a)
    uint cp, cc, uc, lc, dc, ld, um;
    uint wo, ho, wd, hd, flags;
WG_ARGS_END

float3 fg_opq3(float3 v) { return float3(fg_opq(v.x), fg_opq(v.y), fg_opq(v.z)); }
float fg_r16(float x) { return wg_h2f(wg_f2h(x)); }

// one frame pixel: writes Cc / Uc / Lc, returns the block source colour, the luma change and the UI flag
void fg_r3_pixel(uint x, uint y, out float3 s, out float diff, out bool flag)
{
    const bool F1 = (flags & 1u) != 0u, A = (flags & 2u) != 0u, Hd = (flags & 4u) != 0u;
    const bool L1 = (flags & 8u) != 0u, L2 = (flags & 16u) != 0u;
    const uint caseU = F1 ? 1u : 0u, caseA = (!F1 && A) ? 1u : 0u, caseH = (!F1 && !A && Hd) ? 1u : 0u;
    const float u = fg_centre(x, wo), v = fg_centre(y, ho);
    const uint pi = y * wo + x;
    const float4 c = fg_texel(colour, 4u, (cby + y) * cw + cbx + x);
    float4 hc = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (caseU != 0u || caseH != 0u || (caseA != 0u && L2))
    {
        const float2 XY = fg_region_xy(FG_REGION(h), u, v, false);
        hc = L2 ? fg_read_linear(hud, 4u, (int)hw, (int)hh, XY.x, XY.y) : fg_read_nearest(hud, 4u, (int)hw, (int)hh, XY.x, XY.y);
    }
    float al = 0.0f;
    if (caseU != 0u || caseA != 0u)
    {
        const float2 XY = fg_region_xy(FG_REGION(a), u, v, false);
        al = fg_read_nearest(alpha, akind, (int)aw, (int)ah, XY.x, XY.y)[ach];
    }
    float4 ccv;
    flag = false;
    if (caseU != 0u)
    {
        const float ap = L1 ? (fg_finite(al) ? fg_sat(al) : 1.0f) : al;
        const float apm1 = ap - 1.0f;
        const float3 U = float3(mad(hc.x, apm1, c.x), mad(hc.y, apm1, c.y), mad(hc.z, apm1, c.z));
        const bool nf = !fg_finite3(U);
        const float mn = min(min(U.x, U.y), U.z);
        const bool low = U.x < c.x - (hc.x + hc.x) || U.y < c.y - (hc.y + hc.y) || U.z < c.z - (hc.z + hc.z);
        const bool bad = ap == 1.0f || nf || (!nf && mn <= -1.0f) || low;
        const bool zero = L1 && ap == 0.0f;
        float4 ucv = bad ? float4(c.xyz, 1.0f) : float4(U, ap);
        ucv = zero ? float4(0.0f, 0.0f, 0.0f, 0.0f) : ucv;
        fg_st_h4(uc, pi, ucv);
        ccv = float4(hc.xyz, ap);
        s = hc.xyz;
    }
    else if (caseA != 0u)
    {
        flag = al > asfloat(0x3ECCCCCDu);
        ccv = float4(c.xyz, L2 ? (flag ? 1.0f : 0.0f) : al);
        if (L2) fg_st_h4(lc, pi, float4(hc.xyz, 0.0f));
        s = c.xyz;
    }
    else if (caseH != 0u)
    {
        const float3 dd = abs(c.xyz - hc.xyz);
        flag = max(max(dd.x, dd.y), dd.z) > asfloat(0x3D8F5C29u);
        ccv = float4(c.xyz, L2 ? (flag ? 1.0f : 0.0f) : 0.0f);
        if (L2) fg_st_h4(lc, pi, float4(hc.xyz, 0.0f));
        s = c.xyz;
    }
    else
    {
        ccv = float4(c.xyz, 0.0f);
        s = c.xyz;
    }
    fg_st_h4(cc, pi, ccv);
    const float3 c16 = float3(fg_r16(ccv.x), fg_r16(ccv.y), fg_r16(ccv.z));
    const float3 p = fg_texel(cp, 4u, pi).xyz;
    diff = abs(fg_luma(c16) - fg_luma(p));
}

WG_KERNEL(fg_lite_downsample)
{
    FG_SETUP();
    const uint nbx = (wo + FG_D - 1u) / FG_D, nby = (ho + FG_D - 1u) / FG_D;
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= nbx * nby) return;
    const uint X = idx % nbx, Y = idx / nbx;
    float3 s[FG_D * FG_D];
    float df[FG_D * FG_D];
    bool any = false;
    [unroll] for (uint j = 0u; j < FG_D; ++j)
        [unroll] for (uint i = 0u; i < FG_D; ++i)
        {
            const uint x = FG_D * X + i, y = FG_D * Y + j;
            float3 sv = float3(0.0f, 0.0f, 0.0f);
            float dv = 0.0f;
            bool fv = false;
            if (x < wo && y < ho) fg_r3_pixel(x, y, sv, dv, fv);
            s[j * FG_D + i] = sv;
            df[j * FG_D + i] = dv;
            any = any || fv;
        }
    if (X >= wd || Y >= hd) return;
    float3 dcv;
    float ldv;
#if FG_D == 2
    dcv = float3(fg_mul(fg_opq(fg_opq(s[0].x + s[1].x) + fg_opq(s[2].x + s[3].x)), 0.25f),
                 fg_mul(fg_opq(fg_opq(s[0].y + s[1].y) + fg_opq(s[2].y + s[3].y)), 0.25f),
                 fg_mul(fg_opq(fg_opq(s[0].z + s[1].z) + fg_opq(s[2].z + s[3].z)), 0.25f));
    ldv = min(min(df[0], df[1]), min(df[2], df[3]));
#else
    // colour: pairs over x + 1, then y + 1, then x + 2, then y + 2
    float3 a[8];       // a(i, j), i in {0, 2} -> 0 / 1, j = 0..3: a[j * 2 + i / 2]
    [unroll] for (uint jj = 0u; jj < 4u; ++jj)
        [unroll] for (uint ii = 0u; ii < 2u; ++ii)
            a[jj * 2u + ii] = fg_opq3(s[jj * 4u + 2u * ii] + s[jj * 4u + 2u * ii + 1u]);
    float3 b[4];       // b(i, j), j in {0, 2}
    [unroll] for (uint jb = 0u; jb < 2u; ++jb)
        [unroll] for (uint ib = 0u; ib < 2u; ++ib)
            b[jb * 2u + ib] = fg_opq3(a[(2u * jb) * 2u + ib] + a[(2u * jb + 1u) * 2u + ib]);
    const float3 c0 = fg_opq3(b[0] + b[1]), c2 = fg_opq3(b[2] + b[3]);
    const float3 e = fg_opq3(c0 + c2);
    dcv = float3(fg_mul(e.x, 0.0625f), fg_mul(e.y, 0.0625f), fg_mul(e.z, 0.0625f));
    float m1[8];
    [unroll] for (uint jm = 0u; jm < 4u; ++jm)
        [unroll] for (uint im = 0u; im < 2u; ++im)
            m1[jm * 2u + im] = min(df[jm * 4u + 2u * im], df[jm * 4u + 2u * im + 1u]);
    float m2[4];
    [unroll] for (uint jn = 0u; jn < 2u; ++jn)
        [unroll] for (uint in_ = 0u; in_ < 2u; ++in_)
            m2[jn * 2u + in_] = min(m1[(2u * jn) * 2u + in_], m1[(2u * jn + 1u) * 2u + in_]);
    const float s0 = fg_opq(m2[0] + m2[1]), s2 = fg_opq(m2[2] + m2[3]);
    ldv = fg_mul(fg_opq(s0 + s2), 0.25f);
#endif
    const uint ni = Y * wd + X;
    fg_st_h4(dc, ni, float4(dcv, 0.0f));
    fg_st_h1(ld, ni, ldv);
    const bool F1 = (flags & 1u) != 0u, A = (flags & 2u) != 0u, Hd = (flags & 4u) != 0u;
    if (!F1 && (A || Hd)) fg_st_h1(um, ni, any ? 1.0f : 0.0f);
}
