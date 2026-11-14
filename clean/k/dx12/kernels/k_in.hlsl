// SPDX-License-Identifier: MIT
// D3D12 port of model K's exact tiled input-stage kernels (clean/k/kernels/k_io.hip ki_tiled_body, HIP build
// k_io_exact.co = WGK_IN_EXACT, wave64; spec 42 sections 3-8). WrongGPU's own code.
//
// One workgroup of 256 lanes per 8 x 8 base positions: lane 4 b + k is display position k of base position b. The
// workgroup first stages the 24 x 24 render-colour footprint of its tile in LDS (Y, Co, Cg, phi(Y) and the HDR weight
// omega per render pixel), then each lane does its display position's 4 fine positions (reprojected colour history,
// motion with the nearest-depth rule, hidden-history position), the 3 x 3 colour neighbourhood (mean, box, phi), the
// disagreement score, the own sample's luma channel, and the hidden-history sample; the 4 display positions of a base
// position exchange their hidden-history samples by two lane swaps (xor 1, xor 2: within groups of 4 lanes, so the
// wave size does not matter; wave64 as the HIP build) and add them in HIP's order. Writes x0 [Bh][Bw][16] FP16.
// Reset frames: only the own sample's channel 0 (channels 1 = 0 's, 2 and 3 zero), no LDS.
//
// Arithmetic: exactly the HIP kernel's operations in its order (k_io.hip has contraction off and every fma written
// out), with the exact path of WGK_IN_EXACT: log2 / 2^x correctly rounded (kf_log2 / kf_exp2, k_fp.hlsli, with their
// binary64 fallback: extended doubles), every division and reciprocal correctly rounded (wg_div; HIP's a / b and
// 1.0f / x), FP16 rounding to nearest even (wg_f2h; never the driver's (half), which rounds toward zero). Where the
// D3D12 compiler could merge two of HIP's roundings (a product by the uniform kappa or rho, a constant chain), the
// intermediate goes through kf_opq (k_fp.hlsli). FP32 denormals are flushed here (HIP keeps them): none occur in the
// port checks' frames (clean/k/dx12/k_port_check.py).
//
// Kernel declared here: kx/k_in_tiled (KInParams; computes KiHost on the GPU). The forms with KiHost in the arguments
// (kx/k_in_tiled_uniform, K_UNIFORM) and the forms reading the game's textures (_direct, WG_DIRECT_TEXTURES) are in this
// source for the runtime's use but are not declared yet: they are declared and checked with the runtime wiring.
// The KInParams pointers cpre / cpre2 are null in the tiled launches (runtime_k.cpp), as in HIP unused here.
// @wg-kernel kx/k_in_tiled co=k_io_exact.co block=256 wave=64 args=s200:p4.b128.p5
#include "wg_platform.hlsli"
#ifndef K_UNIFORM
#define K_UNIFORM 0
#endif
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
#if K_UNIFORM
    K_ARGS_HOST
#endif
WG_ARGS_END

float3 kf_opq3(float3 x) { return asfloat(asuint(x) ^ kf_opq_z_); }

// ---- the exact input stage's arithmetic (k_io.hip with KI_FAST = 0) --------------------------------------------------
float ki_log2(float x) { return kf_log2(x); }
float ki_exp2(float x) { return kf_exp2(x); }
float ki_rcp(float x) { return kf_div(1.0f, x); }
float ki_div(float a, float b) { return kf_div(a, b); }
// spec 42 decode factor D(t)
float ki_decode(float t)
{
    if (t < kf_bits(0x3F7FBE88u)) return ki_rcp(1.0f - t);
    const float e = kf_opq((t + kf_bits(0x3C4FAE29u)) * 9.0f) * kf_bits(0x3FB8AA3Bu);
    return ki_div(ki_exp2(e) - 1.0f, 9.0f * t);
}
float ki_phi(float Y, bool hdr)
{
    if (hdr) {
        const float Yp = kf_max(Y, 0.0f);
        return ki_div(ki_log2(kf_fma(Yp, 12.0f, 1.0f)) * kf_bits(0x3F317218u), Yp + kf_bits(0x358637BDu));
    }
    return kf_fma(Y, 3.0f, 1.0f);
}
float ki_S(float z)
{
    const float zp = kf_min(kf_max(z, 0.0f), 1.0f);   // HIP's NaN test is implied: maxNum(NaN, 0) = 0
    if (!(zp > kf_bits(0x3B4D2E1Cu))) return kf_bits(0x414EB852u) * zp;
    const float p = ki_exp2(ki_log2(zp) * kf_bits(0x3ED55555u));         // log2 and 2^x each rounded once
    return kf_fma(p, kf_bits(0x3F870A3Du), kf_bits(0xBD6147AEu));
}
// z is often a product (lambda L) that HIP rounds before z + 1. No kf_opq: the gfx1201 ISA keeps the product apart
// (G1's precaution link, dropped in speed round 1 after the ISA check: same v_mul / v_fma counts without it)
float ki_T(float z)
{
    return z * kf_max(ki_rcp(z + 1.0f), kf_bits(0x38D1B717u));
}
float ki_L(float Y, bool hdr) { return hdr ? Y : ki_div(Y, kf_max(1.0f - kf_opq(Y), kf_bits(0x3E4CCCCDu))); }

struct KiFine {  // spec 42 section 5 per fine position
    bool valid;
    float3 h;      // reprojected history (Y, Co, Cg)
    float nx, ny;  // hidden-history position (normalized)
};

KiFine ki_fine(KInParams P, KiHost k, int fx, int fy)
{
    const KFrameIn F = P.f;
    const int W16 = kio_ceil16(F.W), H16 = kio_ceil16(F.H);
    const float px = (float)fx + 0.5f, py = (float)fy + 0.5f;
    float mvx, mvy;
    if ((F.flags & KIO_MVLO) != 0) {
        const int ix = (int)kf_R(F.jx + kf_fma(px, k.a_x, -0.5f)), iy = (int)kf_R(F.jy + kf_fma(py, k.a_y, -0.5f));
        int dx, dy;
        kio_nearest(F, ix, iy, dx, dy);
        kio_motion(F, kio_mtap(ix, dx, F.mbx, F.rw), kio_mtap(iy, dy, F.mby, F.rh), mvx, mvy);
    } else {
        const int mx = (int)kf_R(kf_fma(px, k.mp_x, -0.5f)), my = (int)kf_R(kf_fma(py, k.mp_y, -0.5f));
        kio_motion(F, kio_tap(mx, 0, F.mbx, F.W), kio_tap(my, 0, F.mby, F.H), mvx, mvy);
    }
    // rounded before px + m (HIP); no kf_opq needed: the ISA keeps the products (precaution link dropped, ISA-checked)
    // the texture's motion scale folded into the factor and its offset in raw units, as the buffer form gets them in
    // WgRtFrame (hlsl/k_frame.hlsli kio_motion; issue #26): direct forms b1's ms / mo, KIO_M16 rmsx ..
#if WG_DIRECT_TEXTURES
    const bool dfold = true;
    const float dmsx = nd_msx, dmsy = nd_msy, dmox = nd_mox, dmoy = nd_moy;
#else
    const bool dfold = (F.flags & KIO_M16) != 0;
    const float dmsx = F.rmsx, dmsy = F.rmsy, dmox = F.rmox, dmoy = F.rmoy;
#endif
    float spx = k.sp_x, spy = k.sp_y, ofx = F.mox, ofy = F.moy;
    [branch] if (dfold) {
        const bool dmvlo = (F.flags & KIO_MVLO) != 0;
        spx = ki_dsp(F.sx, dmsx, F.W, dmvlo ? F.rw : F.W);
        spy = ki_dsp(F.sy, dmsy, F.H, dmvlo ? F.rh : F.H);
        ofx = ki_dmo(F.mox, dmox, dmsx);
        ofy = ki_dmo(F.moy, dmoy, dmsy);
    }
    const float mx = spx * (mvx + ofx), my = spy * (mvy + ofy);
    const float wx = px + mx, wy = py + my;
    KiFine r;
    const float ux = wx * k.xi_x, uy = wy * k.xi_y;
    r.valid = ux >= 0.0f && ux <= 1.0f && uy >= 0.0f && uy <= 1.0f;
    const float4 s = kio_bilinear4(P.Hc, W16, H16, wx * k.h_x, wy * k.h_y);
    if ((F.flags & KIO_HDR) != 0) {
        const float f = ki_decode(s.x);
        r.h = F.rho * kf_opq3(s.xyz * f);
    } else {
        r.h = s.xyz;
    }
    r.nx = ((float)(4 * (fx >> 2) + 2) + mx) * k.g_x;   // fx, fy >= 0: fx / 4 = fx >> 2
    r.ny = ((float)(4 * (fy >> 2) + 2) + my) * k.g_y;
    return r;
}

// kappa max(RGB, 0), each product rounded (kf_opq: no merging with the 0.25 / 0.5 factors that follow)
float3 ki_rgb(KFrameIn F, float4 c) { return kf_opq3(F.kappa * max(c.xyz, 0.0f)); }

struct KiTiledTap {
    float4 cv;  // Y, Co, Cg, phi(Y)
    float om;   // HDR weight max(1 / (Y + 1), 1e-4); 1 for LDR (A = fma(1, v, A) is HIP's A + v, one rounding of the sum)
};
KiTiledTap ki_tiled_tap_from_raw(KFrameIn F, float4 c)
{
    const float3 rgb = ki_rgb(F, c);
    float Y, Co, Cg;
    kio_ycocg(rgb.x, rgb.y, rgb.z, Y, Co, Cg);
    const bool hdr = (F.flags & KIO_HDR) != 0;
    KiTiledTap r;
    r.cv = float4(Y, Co, Cg, ki_phi(Y, hdr));
    r.om = hdr ? kf_max(ki_rcp(Y + 1.0f), kf_bits(0x38D1B717u)) : 1.0f;
    return r;
}
KiTiledTap ki_tiled_tap(KFrameIn F, int rx, int ry) { return ki_tiled_tap_from_raw(F, kio_color(F, rx, ry)); }
float ki_tiled_ch0(KFrameIn F, int rx, int ry)
{
    const float3 rgb = ki_rgb(F, kio_color(F, rx, ry));
    const bool hdr = (F.flags & KIO_HDR) != 0;
    const float Yc = kio_luma_fma(rgb.x, rgb.y, rgb.z);
    return ki_S(ki_T(F.lam * ki_L(Yc, hdr)));
}
// HIP ki_tiled_transpose: lane bit 0 / bit 1 swaps through ds_bpermute (wg_xor = WaveReadLaneAt(lane ^ d))
float4 ki_tiled_transpose(float4 v, int lane)
{
    const bool b0 = (lane & 1) != 0, b1 = (lane & 2) != 0;
    const float x = wg_xor(b0 ? v.x : v.y, 1u), y = wg_xor(b0 ? v.z : v.w, 1u);
    const float4 a = float4(b0 ? x : v.x, b0 ? v.y : x, b0 ? y : v.z, b0 ? v.w : y);
    const float z = wg_xor(b1 ? a.x : a.z, 2u), w = wg_xor(b1 ? a.y : a.w, 2u);
    return float4(b1 ? z : a.x, b1 ? w : a.y, b1 ? a.z : z, b1 ? a.w : w);
}
void ki_store_x0(uint x0, int id, int k, float c0, float c1, float c2, float c3)
{
    wg_st64(x0, ((uint)id * 16u + 4u * (uint)k) * 2u,
            uint2(kh_round_bits(c0) | (kh_round_bits(c1) << 16), kh_round_bits(c2) | (kh_round_bits(c3) << 16)));
}

groupshared float4 tile_cv[24 * 24];
groupshared float tile_om[24 * 24];

#if K_UNIFORM
WG_KERNEL(k_in_tiled_uniform)
#else
WG_KERNEL(k_in_tiled)
#endif
{
    WG_SETUP();
    KF_SETUP();
    KInParams P0;
    K_LOAD_IN_PARAMS(P0)
    const KInParams P = ki_params(P0);
    const KFrameIn F = P.f;
#if K_UNIFORM
    KiHost hk;
    K_LOAD_HOST(hk)
#else
    const KiHost hk = ki_host(F);
#endif
    const int lane = (int)wg_lane_id();
    const int tiles_x = (F.Bw + 7) / 8, tile = (int)wg_block_id();
    const int bx0 = (tile % tiles_x) * 8, by0 = (tile / tiles_x) * 8;
    const int k = lane & 3, local = lane >> 2;
    const int bx = bx0 + (local & 7), by = by0 + (local >> 3);
    const bool active = bx < F.Bw && by < F.Bh;
    const int id = by * F.Bw + bx;
    const bool hdr = (F.flags & KIO_HDR) != 0;
    const float c1e4 = kf_bits(0x38D1B717u);
    float ch0 = 0.0f, ch1 = 0.0f, ch3 = 0.0f;
    float4 hid = 0.0f;
    if ((F.flags & KIO_RESET) != 0) {
        if (active) {
            const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
            const int ofx = 2 * u + 1 - (v & 1), ofy = 2 * v + (u & 1);
            const int rx = (int)kf_R(F.jx + kf_fma((float)ofx + 0.5f, hk.a_x, -0.5f));
            const int ry = (int)kf_R(F.jy + kf_fma((float)ofy + 0.5f, hk.a_y, -0.5f));
            const float h = ki_tiled_ch0(F, rx, ry);
            ki_store_x0(P.x0, id, k, h, h, 0.0f, 0.0f);
        }
        return;
    }
    const int tx0 = ki_max(ki_min((int)kf_R(F.jx + kf_fma((float)(4 * bx0 + 1), hk.a_x, -0.5f)) - 1, F.rw - 1), 0);
    const int ty0 = ki_max(ki_min((int)kf_R(F.jy + kf_fma((float)(4 * by0 + 1), hk.a_y, -0.5f)) - 1, F.rh - 1), 0);
    for (int q = lane; q < 24 * 24; q += 256) {
        const int raw_x = tx0 + q % 24, raw_y = ty0 + q / 24;
        const int rx = ki_min(raw_x, F.rw - 1), ry = ki_min(raw_y, F.rh - 1);
        const KiTiledTap value = ki_tiled_tap_from_raw(F, kio_color(F, rx, ry));
        tile_cv[q] = value.cv;
        tile_om[q] = value.om;
    }
    wg_barrier_lds();
    if (active) {
        const int u = 2 * bx + (k & 1), v = 2 * by + (k >> 1);
        // colour neighbourhood first, then each fine position scored as it is produced (the same per-position
        // operations and the same order of the D sum as HIP; nothing of a fine position stays live but the own
        // sample's Yh, nx, ny)
        const int cx = (int)kf_R(F.jx + kf_fma((float)(2 * u + 1), hk.a_x, -0.5f));
        const int cy = (int)kf_R(F.jy + kf_fma((float)(2 * v + 1), hk.a_y, -0.5f));
        float3 A = 0.0f, lo = 0.0f, hi = 0.0f;
        [unroll] for (int ddy = -1; ddy <= 1; ++ddy) {
            [unroll] for (int ddx = -1; ddx <= 1; ++ddx) {
                const int rx = kio_tap(cx, ddx, 0, F.rw), ry = kio_tap(cy, ddy, 0, F.rh);
                const int lx = rx - tx0, ly = ry - ty0;
                const bool cached = (uint)lx < 24u && (uint)ly < 24u;
                float4 cv;
                float om;
                [branch] if (cached) {
                    cv = tile_cv[ly * 24 + lx];
                    om = tile_om[ly * 24 + lx];
                } else {  // defensive, as HIP: a footprint wider than the tile (the runtime's guard prevents it)
                    const KiTiledTap fb = ki_tiled_tap(F, rx, ry);
                    cv = fb.cv;
                    om = fb.om;
                }
                A = float3(kf_fma(om, cv.x, A.x), kf_fma(om, cv.y, A.y), kf_fma(om, cv.z, A.z));   // LDR: om = 1
                const float3 val = cv.xyz * cv.w;
                const bool first = ddx == -1 && ddy == -1;
                lo = first ? val : min(lo, val);
                hi = first ? val : max(hi, val);
            }
        }
        float3 mu = A * kf_bits(0x3DE38E39u);   // rounded before 1 - mu (the ISA keeps the products: no kf_opq)
        if (hdr) {
            const float den = kf_max(1.0f - mu.x, c1e4);
            mu = float3(ki_div(mu.x, den), ki_div(mu.y, den), ki_div(mu.z, den));
        }
        const float pm = ki_phi(mu.x, hdr);
        const float3 mp = kf_opq3(mu * pm);
        const float3 eta = 0.5f * (hi - lo);
        const float3 nu = float3(kf_fma(hi.x - lo.x, 8.0f, 1.0f), kf_fma(hi.y - lo.y, 8.0f, 1.0f), kf_fma(hi.z - lo.z, 8.0f, 1.0f));
        const int oq = (1 - (v & 1)) + 2 * (u & 1);  // (2u + 1 - (v mod 2), 2v + (u mod 2)) as a fine position index
        // fine positions of the display position, in the order (2u, 2v), (2u+1, 2v), (2u, 2v+1), (2u+1, 2v+1)
        bool valid = true;
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
        // own sample
        const int ofx = 2 * u + 1 - (v & 1), ofy = 2 * v + (u & 1);
        const int rx = (int)kf_R(F.jx + kf_fma((float)ofx + 0.5f, hk.a_x, -0.5f));
        const int ry = (int)kf_R(F.jy + kf_fma((float)ofy + 0.5f, hk.a_y, -0.5f));
        ch0 = ki_tiled_ch0(F, rx, ry);
        const float4 hs = kio_bilinear4(P.Hh, F.Bw, F.Bh, own_nx, own_ny);
        hid = valid ? hs : 0.0f;
        // channel 1 is channel 0's function of Yh, and Yh = Yc without a valid history: then it equals channel 0
        ch1 = valid ? ki_S(ki_T(F.lam * ki_L(own_hx, hdr))) : ch0;
        const float d3 = valid ? D : 0.0f;
        ch3 = kf_finite(d3) ? d3 : 0.0f;
    }
    const float4 hs = ki_tiled_transpose(hid, lane);
    if (!active) return;
    float hsum = (hs.x + hs.w) + (hs.y + hs.z);
    hsum = hsum * 0.25f;
    ki_store_x0(P.x0, id, k, ch0, ch1, kf_finite(hsum) ? hsum : 0.0f, ch3);
}
