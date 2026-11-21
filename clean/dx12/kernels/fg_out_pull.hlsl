// SPDX-License-Identifier: MIT
// Frame generation (issue #96): the Full path's output pull and middle pull, as clean/fg/reference/stages_full.py
// g6_output_pull and g6_output_pull_mid, bit for bit. Built only when --only names them.
//
// fg_out_pull: one workgroup per 16 x 16 frame tile (tiles_x = ceil(Wo / 16)), one lane per pixel in Morton order.
// Per frame pixel: the hole rule on Xw's 5 x 5 neighbourhood (outside the frame counts as need), net2's output N2
// (W2 x H2, 8 x float16) sampled at (x / k, y / k) with the logistic of channel 4 per tap from the table sigtab
// (65,536 float32, indexed by the float16 bits), X0 / X1 read bilinearly at the pixel plus k times net2's two vectors,
// c = sat(fma(sigma, A, (1 - sigma) B)), holes black. Writes out (4 x float16, alpha 1, row pitch out_pitch, offset
// (out_x, out_y)), the fine flag image (float16 0 / 1) and output pyramid levels 1..4 (4 x float16) through a lane tree.
// fg_out_pull_mid: one workgroup per 16 x 16 tile of level 4 (tiles_x = ceil(w4 / 16)): levels 5..8 from level 4's
// stored values (partials (r w, g w, b w, w)).
//
// @wg-kernel fg_out_pull co=fg_full.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,u,u,u
// @wg-kernel fg_out_pull_mid co=fg_full.co block=256 wave=32 args=p,p,p,p,p,u,u,u,u,u defines=FGP_MID=1
#include "wg_platform.hlsli"
#include "fg_full.hlsli"

#ifndef FGP_MID
#define FGP_MID 0
#endif

groupshared float4 fgp_s[256];

// two lane-tree steps (x then y) at strides s and 2 s
void fgp_steps(uint l, uint s)
{
    wg_barrier_lds();
    if ((l & (2u * s - 1u)) == 0u) fgp_s[l] = fgp_s[l] + fgp_s[l + s];
    wg_barrier_lds();
    if ((l & (4u * s - 1u)) == 0u) fgp_s[l] = fgp_s[l] + fgp_s[l + 2u * s];
    wg_barrier_lds();
}

int fgp_size(int n, uint lvl) { return (n + (1 << lvl) - 1) >> lvl; }

WG_ARGS_BEGIN
#if FGP_MID
    uint lv4, lv5, lv6, lv7, lv8;
    int w4, h4, tiles_x, Wo, Ho;
#else
    uint x0, x1, xw, n2, sigtab, outp, fine, lv1, lv2, lv3, lv4;
    int Wo, Ho, W2, H2, k, tiles_x, out_pitch, out_x, out_y;
    uint rk_bits;   // 1 / k (float bits)
#endif
WG_ARGS_END

#if FGP_MID
WG_KERNEL(fg_out_pull_mid)
{
    FGF_SETUP();
    const uint l = wg_lane_id();
    const uint t = wg_block_id();
    const int ex = (int)((t % (uint)tiles_x) * 16u + fgf_mx(l));
    const int ey = (int)((t / (uint)tiles_x) * 16u + fgf_my(l));
    float4 acc = 0.0f;
    if (ex < w4 && ey < h4) {
        const float4 e = fgf_ld4h(lv4, ey * w4 + ex);
        acc = float4(e.xyz * e.w, e.w);
    }
    fgp_s[l] = acc;
    [unroll] for (uint m = 1u; m <= 4u; ++m) {
        fgp_steps(l, 1u << (2u * m - 2u));
        const uint lvl = m + 4u;
        if ((l & ((1u << (2u * m)) - 1u)) == 0u) {
            const int X = ex >> m, Y = ey >> m;
            if (X < fgp_size(Wo, lvl) && Y < fgp_size(Ho, lvl)) {
                const uint dst = m == 1u ? lv5 : (m == 2u ? lv6 : (m == 3u ? lv7 : lv8));
                fgf_st4h(dst, Y * fgp_size(Wo, lvl) + X, fgf_level(fgp_s[l], (float)(1u << (2u * m))));
            }
        }
    }
}
#else
// net2's bilinear rule: fma(1 - fy, fma(1 - fx, q00, fx q10), fy fma(1 - fx, q01, fx q11))
float fgp_bil(float fx, float fy, float gx, float gy, float q00, float q10, float q01, float q11)
{
    return mad(gy, mad(gx, q00, fx * q10), fy * mad(gx, q01, fx * q11));
}

WG_KERNEL(fg_out_pull)
{
    FGF_SETUP();
    const uint l = wg_lane_id();
    const uint t = wg_block_id();
    const int px = (int)((t % (uint)tiles_x) * 16u + fgf_mx(l));
    const int py = (int)((t / (uint)tiles_x) * 16u + fgf_my(l));
    float4 acc = 0.0f;
    if (px < Wo && py < Ho) {
        // hole: every 3 x 3 window containing the pixel holds a pixel with Xw > 0 (outside the frame: need)
        uint m = 0u;
        [unroll] for (int dy = 0; dy < 5; ++dy) {
            [unroll] for (int dx = 0; dx < 5; ++dx) {
                const int qx = px + dx - 2, qy = py + dy - 2;
                const bool inside = qx >= 0 && qy >= 0 && qx < Wo && qy < Ho;
                const float need = fgf_ld1h(xw, clamp(qy, 0, Ho - 1) * Wo + clamp(qx, 0, Wo - 1));
                m |= ((!inside || need > 0.0f) ? 1u : 0u) << (uint)(dy * 5 + dx);
            }
        }
        bool hole = true;
        [unroll] for (int cy = 0; cy < 3; ++cy) {
            [unroll] for (int cx = 0; cx < 3; ++cx) {
                const uint wmask = (7u << (uint)(cy * 5 + cx)) | (7u << (uint)(cy * 5 + 5 + cx)) | (7u << (uint)(cy * 5 + 10 + cx));
                hole = hole && (m & wmask) != 0u;
            }
        }
        // net2's output at (x / k, y / k)
        const float rk = asfloat(rk_bits);
        const float s = min((float)px * rk, (float)(W2 - 1)), tt = min((float)py * rk, (float)(H2 - 1));
        const float sf = floor(s), tf = floor(tt);
        const int i0 = (int)sf, j0 = (int)tf, i1 = min(i0 + 1, W2 - 1), j1 = min(j0 + 1, H2 - 1);
        const float fx = s - sf, fy = tt - tf, gx = 1.0f - fx, gy = 1.0f - fy;
        const uint4 d00 = wg_ld128(n2, (uint)(j0 * W2 + i0) * 16u), d10 = wg_ld128(n2, (uint)(j0 * W2 + i1) * 16u);
        const uint4 d01 = wg_ld128(n2, (uint)(j1 * W2 + i0) * 16u), d11 = wg_ld128(n2, (uint)(j1 * W2 + i1) * 16u);
        const float sigma = fgp_bil(fx, fy, gx, gy, wg_ldf(sigtab, (d00.z & 0xFFFFu) * 4u), wg_ldf(sigtab, (d10.z & 0xFFFFu) * 4u),
                                    wg_ldf(sigtab, (d01.z & 0xFFFFu) * 4u), wg_ldf(sigtab, (d11.z & 0xFFFFu) * 4u));
        const float vax = fgp_bil(fx, fy, gx, gy, wg_h2f_lo(d00.x), wg_h2f_lo(d10.x), wg_h2f_lo(d01.x), wg_h2f_lo(d11.x));
        const float vay = fgp_bil(fx, fy, gx, gy, wg_h2f_hi(d00.x), wg_h2f_hi(d10.x), wg_h2f_hi(d01.x), wg_h2f_hi(d11.x));
        const float vbx = fgp_bil(fx, fy, gx, gy, wg_h2f_lo(d00.y), wg_h2f_lo(d10.y), wg_h2f_lo(d01.y), wg_h2f_lo(d11.y));
        const float vby = fgp_bil(fx, fy, gx, gy, wg_h2f_hi(d00.y), wg_h2f_hi(d10.y), wg_h2f_hi(d01.y), wg_h2f_hi(d11.y));
        // the two candidates re-sampled at the pixel plus k times the vectors (pixel units)
        const float kf = (float)k;
        const float3 A = fgf_bil4h(x0, fgf_tap(vax * kf + (float)px, vay * kf + (float)py, Wo, Ho)).xyz;
        const float3 B = fgf_bil4h(x1, fgf_tap(vbx * kf + (float)px, vby * kf + (float)py, Wo, Ho)).xyz;
        float3 c = fgf_sat3(mad(sigma, A, (1.0f - sigma) * B));
        if (isnan(abs(c.z * (c.x * c.y)))) c = float3(0.0f, 1.0f, 1.0f);
        if (hole) c = 0.0f;
        fgf_st4h(outp, (py + out_y) * out_pitch + px + out_x, float4(c, 1.0f));
        fgf_st1h(fine, py * Wo + px, hole ? 1.0f : 0.0f);
        acc = float4(c, hole ? 0.0f : 1.0f);
    }
    fgp_s[l] = acc;
    [unroll] for (uint lvl = 1u; lvl <= 4u; ++lvl) {
        fgp_steps(l, 1u << (2u * lvl - 2u));
        if ((l & ((1u << (2u * lvl)) - 1u)) == 0u) {
            const int X = px >> lvl, Y = py >> lvl;
            if (X < fgp_size(Wo, lvl) && Y < fgp_size(Ho, lvl)) {
                const uint dst = lvl == 1u ? lv1 : (lvl == 2u ? lv2 : (lvl == 3u ? lv3 : lv4));
                fgf_st4h(dst, Y * fgp_size(Wo, lvl) + X, fgf_level(fgp_s[l], (float)(1u << (2u * lvl))));
            }
        }
    }
}
#endif
