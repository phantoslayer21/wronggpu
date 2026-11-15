// SPDX-License-Identifier: MIT
// D3D12 ports of model K's legacy exact input-stage kernels (clean/k/kernels/k_io.hip k_in_color, k_in_reset, k_in_base;
// HIP build k_io_exact.co = WGK_IN_EXACT, wave64, 128 lanes per workgroup; spec 42). WrongGPU's own code. The runtime
// takes them when the tiled input stage does not apply: a render size above 0.68 x the output (native AA, DLAA-like
// ratios up to 1.0), or a jitter outside [-0.5, 0.5], and for a base size outside the tiled guard.
//
// k_in_color: per render pixel (i, j) the colour tap values (Y, Co, Cg, phi(Y)) into cpre and the HDR weight and the
// own sample's channel 0 into cpre2. k_in_reset: a reset frame's x0 from the own sample's channel 0 only. k_in_base: one
// lane per display position, 32 base positions per workgroup; the four display positions of a base position meet in LDS
// for the hidden-history channel. Same arithmetic, same order and same roundings as HIP, with the exact path of
// WGK_IN_EXACT (hlsl/k_in_common.hlsli: correctly rounded log2 / 2^x / divisions and reciprocals, FP16 rounding to
// nearest even); where the D3D12 compiler could merge two of HIP's roundings (a product by a uniform feeding an add or
// a subtraction) the intermediate goes through kf_opq, exactly where the tiled kernel (k_in.hlsl) does. A luminance
// weight of LDR is 0 in cpre2 (HIP), and the LDR sum is a plain addition (HIP's A + v), not a fused multiply-add.
// The direct forms (rs=direct, the native executor: colour / depth / motion from the game's textures) are the same
// source with WG_DIRECT_TEXTURES; k_in_reset reads only buffers and has no direct form.
// FP32 denormals are flushed here (HIP keeps them), as in every D3D12 port.
//
// @wg-kernel kx/k_in_color co=k_io_exact.co entry=k_in_color block=128 wave=64 args=s200:p4.b128.p5
// @wg-kernel kx/k_in_reset co=k_io_exact.co entry=k_in_reset block=128 wave=64 args=s200:p4.b128.p5
// @wg-kernel kx/k_in_base co=k_io_exact.co entry=k_in_base block=128 wave=64 args=s200:p4.b128.p5
// @wg-kernel kx/k_in_color_direct co=k_io_exact.co entry=k_in_color block=128 wave=64 args=s200:p4.b128.p5 rs=direct defines=WG_DIRECT_TEXTURES=1
// @wg-kernel kx/k_in_base_direct co=k_io_exact.co entry=k_in_base block=128 wave=64 args=s200:p4.b128.p5 rs=direct defines=WG_DIRECT_TEXTURES=1
#include "wg_platform.hlsli"
#ifndef WG_DIRECT_TEXTURES
#define WG_DIRECT_TEXTURES 0
#endif
#if WG_DIRECT_TEXTURES
#include "native_direct.hlsli"
#endif
#include "k_fp.hlsli"
#include "k_frame.hlsli"

WG_ARGS_BEGIN
    K_ARGS_FRAME_IN
    K_ARGS_IN_PTRS
WG_ARGS_END

float3 kf_opq3(float3 x) { return asfloat(asuint(x) ^ kf_opq_z_); }

#include "k_in_common.hlsli"

groupshared float4 lds_hid[128];

// Per render pixel: the operations k_in_base would do for a colour tap or an own sample at clamped position (i, j)
WG_KERNEL(k_in_color)
{
    WG_SETUP();
    KF_SETUP();
    KInParams P0;
    K_LOAD_IN_PARAMS(P0)
    const KInParams P = ki_params(P0);
    const KFrameIn F = P.f;
    const int id = (int)(wg_block_id() * 128u + wg_lane_id());
    if (id >= F.rw * F.rh) return;
    const int i = id % F.rw, j = id / F.rw;
    const bool hdr = (F.flags & KIO_HDR) != 0;
    const float4 c = kio_texel(F, F.cbx + i, F.cby + j);
    const float3 rgb = ki_rgb(F, c);
    float Y, Co, Cg;
    kio_ycocg(rgb.x, rgb.y, rgb.z, Y, Co, Cg);
    wg_st128(P.cpre, (uint)id * 16u, asuint(float4(Y, Co, Cg, ki_phi(Y, hdr))));
    const float Yc = kio_luma_fma(rgb.x, rgb.y, rgb.z);
    wg_stf(P.cpre2, (uint)id * 8u, hdr ? kf_max(ki_rcp(Y + 1.0f), kf_bits(0x38D1B717u)) : 0.0f);
    wg_stf(P.cpre2, (uint)id * 8u + 4u, ki_S(ki_T(F.lam * ki_L(Yc, hdr))));
}

// A reset frame: channel 1 equals channel 0 and channels 2 / 3 are zero (spec 42 sections 6.1, 6.5 and 7)
WG_KERNEL(k_in_reset)
{
    WG_SETUP();
    KF_SETUP();
    KInParams P;
    K_LOAD_IN_PARAMS(P)
    const KFrameIn F = P.f;
    const int lane = (int)wg_lane_id();
    const int id = (int)(wg_block_id() * 32u) + (lane >> 2), k = lane & 3;
    if (id >= F.Bw * F.Bh) return;
    const int bx = id % F.Bw, by = id / F.Bw;
    const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
    const KiHost hk = ki_host(F);
    const int fx = 2 * u + 1 - (v & 1), fy = 2 * v + (u & 1);
    const int rx = (int)kf_R(F.jx + kf_fma((float)fx + 0.5f, hk.a_x, -0.5f));
    const int ry = (int)kf_R(F.jy + kf_fma((float)fy + 0.5f, hk.a_y, -0.5f));
    const float ch0 = wg_ldf(P.cpre2, (((uint)kio_tap(ry, 0, 0, F.rh) * (uint)F.rw + (uint)kio_tap(rx, 0, 0, F.rw)) * 2u + 1u) * 4u);
    ki_store_x0(P.x0, id, k, ch0, ch0, 0.0f, 0.0f);
}

// One lane per display position: lanes 4b .. 4b + 3 are the display positions k = 0..3 of base position b
WG_KERNEL(k_in_base)
{
    WG_SETUP();
    KF_SETUP();
    KInParams P0;
    K_LOAD_IN_PARAMS(P0)
    const KInParams P = ki_params(P0);
    const KFrameIn F = P.f;
    const int lane = (int)wg_lane_id();
    const int id = (int)(wg_block_id() * 32u) + (lane >> 2), k = lane & 3;
    const bool active = id < F.Bw * F.Bh;   // inactive lanes still reach the barrier
    const int bx = active ? id % F.Bw : 0, by = active ? id / F.Bw : 0;
    const bool hdr = (F.flags & KIO_HDR) != 0;
    const KiHost hk = ki_host(F);
    const float c1e4 = kf_bits(0x38D1B717u);
    float ch0 = 0.0f, ch1 = 0.0f, ch3 = 0.0f;
    float4 hid = 0.0f;
    if (active) {
        const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
        // colour neighbourhood
        const int cx = (int)kf_R(F.jx + kf_fma((float)(2 * u + 1), hk.a_x, -0.5f));
        const int cy = (int)kf_R(F.jy + kf_fma((float)(2 * v + 1), hk.a_y, -0.5f));
        float3 A = 0.0f, lo = 0.0f, hi = 0.0f;
        [unroll] for (int ddy = -1; ddy <= 1; ++ddy) {
            [unroll] for (int ddx = -1; ddx <= 1; ++ddx) {
                // the tap's values from k_in_color (render pixel kio_tap(.., 0, n) of the subrect)
                const uint t = (uint)kio_tap(cy, ddy, 0, F.rh) * (uint)F.rw + (uint)kio_tap(cx, ddx, 0, F.rw);
                const float4 cv = wg_ld4f(P.cpre, t * 16u);
                if (hdr) {
                    const float om = wg_ldf(P.cpre2, t * 8u);
                    A = float3(kf_fma(om, cv.x, A.x), kf_fma(om, cv.y, A.y), kf_fma(om, cv.z, A.z));
                } else {
                    A = float3(A.x + cv.x, A.y + cv.y, A.z + cv.z);   // HIP's A + v (no weight)
                }
                const float3 val = cv.xyz * cv.w;
                const bool first = ddx == -1 && ddy == -1;
                lo = first ? val : min(lo, val);
                hi = first ? val : max(hi, val);
            }
        }
        float3 mu = A * kf_bits(0x3DE38E39u);
        if (hdr) {
            const float den = kf_max(1.0f - mu.x, c1e4);
            mu = float3(ki_div(mu.x, den), ki_div(mu.y, den), ki_div(mu.z, den));
        }
        const float pm = ki_phi(mu.x, hdr);
        const float3 mp = kf_opq3(mu * pm);
        const float3 eta = 0.5f * (hi - lo);
        const float3 nu = float3(kf_fma(hi.x - lo.x, 8.0f, 1.0f), kf_fma(hi.y - lo.y, 8.0f, 1.0f), kf_fma(hi.z - lo.z, 8.0f, 1.0f));
        const int oq = (1 - (v & 1)) + 2 * (u & 1);   // (2u + 1 - (v mod 2), 2v + (u mod 2)) as a fine position index
        // fine positions in the order (2u, 2v), (2u+1, 2v), (2u, 2v+1), (2u+1, 2v+1); each scored as it is produced
        bool valid = (F.flags & KIO_RESET) == 0;
        float D = 0.0f, own_hx = 0.0f, own_nx = 0.0f, own_ny = 0.0f;
        [unroll] for (int q = 0; q < 4; ++q) {
            const KiFine fq = ki_fine(P, hk, 2 * u + (q & 1), 2 * v + (q >> 1));
            valid = valid && fq.valid;
            const float ph = ki_phi(fq.h.x, hdr);
            const float3 g = max(abs(mp - kf_opq3(fq.h * ph)) - eta, 0.0f);
            const float3 dl = float3(ki_div(g.x, nu.x), ki_div(g.y, nu.y), ki_div(g.z, nu.z));
            const float score = kf_min(kf_fma(dl.z, 0.5f, kf_fma(dl.x, 2.0f, 0.5f * dl.y)), 1.0f);
            D = D + score;
            if (q == oq) {
                own_hx = fq.h.x;
                own_nx = fq.nx;
                own_ny = fq.ny;
            }
        }
        D = D * 0.25f;
        // own sample: channel 0 from k_in_color's cpre2
        const int ofx = 2 * u + 1 - (v & 1), ofy = 2 * v + (u & 1);
        const int rx = (int)kf_R(F.jx + kf_fma((float)ofx + 0.5f, hk.a_x, -0.5f));
        const int ry = (int)kf_R(F.jy + kf_fma((float)ofy + 0.5f, hk.a_y, -0.5f));
        ch0 = wg_ldf(P.cpre2, (((uint)kio_tap(ry, 0, 0, F.rh) * (uint)F.rw + (uint)kio_tap(rx, 0, 0, F.rw)) * 2u + 1u) * 4u);
        const float4 hs = kio_bilinear4(P.Hh, F.Bw, F.Bh, own_nx, own_ny);
        hid = valid ? hs : 0.0f;
        // channel 1 is channel 0's function of Yh, and Yh = Yc without a valid history: then it equals channel 0
        ch1 = valid ? ki_S(ki_T(F.lam * ki_L(own_hx, hdr))) : ch0;
        const float d3 = valid ? D : 0.0f;
        ch3 = kf_finite(d3) ? d3 : 0.0f;
    }
    lds_hid[lane] = hid;
    wg_barrier_lds();
    if (!active) return;
    // slot k, channel 2: 0.25 ((s0 + s3) + (s1 + s2)), s_m = channel k of slot m's hidden-history sample
    const int g4 = lane & ~3;
    float hsum = (lds_hid[g4][k] + lds_hid[g4 + 3][k]) + (lds_hid[g4 + 1][k] + lds_hid[g4 + 2][k]);
    hsum = hsum * 0.25f;
    ki_store_x0(P.x0, id, k, ch0, ch1, kf_finite(hsum) ? hsum : 0.0f, ch3);
}
