// SPDX-License-Identifier: MIT
// Frame generation (issue #96), light path: G2 intermediate-time scatter and G3 blend weights (clean/fg/reference/
// stages_common.py interm_scatter, interm_inputs, blend_weights). D3D12 only.
//
// fg_lite_interm: one lane per scatter-grid pixel; pass 0 (A): displacement (1 - t) (Mv W1, Mv H1) and keys from Zc,
//   written unless both the truncated-key and the displacement test fail; pass 1 (B): displacement t F and keys from
//   Zp, written when the displacement test passes. Footprint 2 at a depth edge (key range over 3 x 3 > tau), else from
//   the spread of the warped 3 x 3 positions (2..5 per axis). Signed 64-bit atomic minimum (SM 6.6) of (key bits << 32
//   | float16 of the vector back to the source frame, normalised).
// fg_lite_weights: W = (sigma(NA at the MB vector - NA co-located), sigma(NB at the MA vector - NB co-located)).
//
// @wg-kernel fg_lite_interm co=- block=256 wave=32 args=u,p,p,p,u,u,u,u,u,u,u defines=FG_K=1
// @wg-kernel fg_lite_weights co=- block=256 wave=32 args=p,p,p,p,p,u,u,u,u defines=FG_K=2
#include "wg_platform.hlsli"
#include "fg_lite_common.hlsli"

#if FG_K == 1
WG_ARGS_BEGIN
    uint pass_, src, zimg, target, w1, h1, tbits, s_k, inverted, t2, tau;
WG_ARGS_END

float2 fg_disp(int x, int y)
{
    const float t = asfloat(tbits);
    if (pass_ == 0u)
    {
        const float omt = 1.0f - t;
        const float4 m = fg_read_int(src, 5u, (int)w1, (int)h1, x, y);
        return float2(fg_mul(omt, fg_mul(m.x, (float)w1)), fg_mul(omt, fg_mul(m.y, (float)h1)));
    }
    const float4 f = fg_read_int(src, 2u, (int)w1, (int)h1, x, y);
    return float2(fg_mul(t, f.x), fg_mul(t, f.y));
}

float fg_key_at(int x, int y)
{
    const uint Z = wg_ld32(zimg, (uint)(clamp(y, 0, (int)h1 - 1) * (int)w1 + clamp(x, 0, (int)w1 - 1)) * 4u);
    return fg_depth_key(fg_unpack_depth(Z, 0.0f, 0.0f).x, asfloat(s_k), inverted);
}

int fg_radius(float qmax, float qmin, float qp)
{
    const float a = max(abs(qmax - qp), abs(qmin - qp));
    return clamp((int)fg_med3(ceil(a + 0.5f), 2.0f, 5.0f), 2, 5);
}

WG_KERNEL(fg_lite_interm)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const int x = (int)(idx % w1), y = (int)(idx / w1);
    const float tauf = asfloat(tau);
    const float kp = fg_key_at(x, y);
    const float2 dp = fg_disp(x, y);
    float kmax = -FG_INF, kmin = FG_INF, qxmax = -10000.0f, qymax = -10000.0f, qxmin = 10000.0f, qymin = 10000.0f;
    [unroll] for (int j = -1; j <= 1; ++j)
        [unroll] for (int i = -1; i <= 1; ++i)
        {
            const float kn = fg_key_at(x + i, y + j);
            const float2 dn = fg_disp(x + i, y + j);
            kmax = max(kmax, kn);
            kmin = min(kmin, kn);
            const float qx = fg_opq((float)(x + i) + 0.5f) + dn.x, qy = fg_opq((float)(y + j) + 0.5f) + dn.y;
            qxmax = max(qxmax, qx); qxmin = min(qxmin, qx);
            qymax = max(qymax, qy); qymin = min(qymin, qy);
        }
    const float qpx = fg_opq((float)x + 0.5f) + dp.x, qpy = fg_opq((float)y + 0.5f) + dp.y;
    const bool edge = (kmax - kp) > tauf || (kp - kmin) > tauf;
    const int rx = edge ? 2 : fg_radius(qxmax, qxmin, qpx);
    const int ry = edge ? 2 : fg_radius(qymax, qymin, qpy);
    const uint hi = asuint(kp);
    const uint lo = (wg_f2h(fg_adiv(-dp.x, (float)w1)) << 16) | wg_f2h(fg_adiv(-dp.y, (float)h1));
    const float thr = max(fg_adiv(fg_sq(dp.x, dp.y), asfloat(t2)), 1.0f);
    const int sx = fg_sgn(dp.x), sy = fg_sgn(dp.y);
    const int bx = x + (int)dp.x, by = y + (int)dp.y;
    [loop] for (int jj = 0; jj < ry; ++jj)
        [loop] for (int ii = 0; ii < rx; ++ii)
        {
            const int tx = bx + ii * sx, ty = by + jj * sy;
            if (tx < 0 || tx >= (int)w1 || ty < 0 || ty >= (int)h1) continue;
            const int nx = x + ii * sx, ny = y + jj * sy;
            const float2 nd = fg_disp(nx, ny);
            const float dd = fg_sq(nd.x - dp.x, nd.y - dp.y);
            bool ok;
            if (pass_ == 0u)
            {
                const float kt = (float)(uint)fg_key_at(nx, ny);
                ok = !(!(abs(kt - kp) < 3.0f) && !(dd < thr));
            }
            else
                ok = dd < thr;
            if (ok) fg_atomic_min64(target, (uint)(ty * (int)w1 + tx), hi, lo);
        }
}
#endif

#if FG_K == 2
WG_ARGS_BEGIN
    uint ma, mb, na, nb, wout, w1, h1, wd, hd;
WG_ARGS_END

WG_KERNEL(fg_lite_weights)
{
    FG_SETUP();
    const uint idx = wg_block_id() * 256u + wg_lane_id();
    if (idx >= w1 * h1) return;
    const uint x = idx % w1, y = idx / w1;
    const float u = fg_centre(x, w1), v = fg_centre(y, h1);
    const uint mi = fg_index_rint(v, h1) * w1 + fg_index_rint(u, w1);
    const float3 a = fg_vec64(ma, mi), b = fg_vec64(mb, mi);
    const uint ji = fg_index_rint(v, hd) * wd + fg_index_rint(u, wd);
    const float A1 = fg_read_linear(na, 2u, (int)wd, (int)hd, u + b.x, v + b.y).x;
    const float A0 = fg_texel(na, 2u, ji).y;
    const float B1 = fg_read_linear(nb, 2u, (int)wd, (int)hd, u + a.x, v + a.y).x;
    const float B0 = fg_texel(nb, 2u, ji).y;
    fg_st_h2(wout, idx, fg_sigmoid(A1, A0), fg_sigmoid(B1, B0));
}
#endif
