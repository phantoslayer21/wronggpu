// SPDX-License-Identifier: MIT
// The exact input stage's scalar arithmetic and per-fine-position reprojection, shared by the tiled kernels (k_in.hlsl)
// and the legacy kernels (k_in_legacy.hlsl). WrongGPU's own code. Include after the kernel's argument block,
// k_fp.hlsli and k_frame.hlsli, and after kf_opq3 is defined.
#pragma once

// ---- the exact input stage's arithmetic (k_io.hip with KI_FAST = 0) --------------------------------------------------
#ifndef KI_HWMATH
#define KI_HWMATH 0
#endif
#if KI_HWMATH
// CLOSE-APPROXIMATION variant (opt-in, never the default): the transcendental unit instead of the correctly rounded
// sequences: v_log_f32 / v_exp_f32 (about 1 FP32 ulp), v_rcp_f32 and a product for the divisions. The results meet the
// input stage's FP16 roundings, so they differ from the exact form only where an FP32 result lies within about one
// FP32 ulp of an FP16 rounding midpoint.
float ki_log2(float x) { return log2(x); }
float ki_exp2(float x) { return exp2(x); }
float ki_rcp(float x) { return rcp(x); }
// a rcp(b), branch free. For |b| >= 2^126, 1 / b would flush to zero (e.g. the score's g / nu for a colour range near
// FLT_MAX, where the quotient is about 1/8): there b is scaled by s = 2^-64 first and the product scaled back,
// (a rcp(b s)) s, which cannot overflow (|a / b| <= 2^2 there) and flushes only where a / b itself is below 2^-126.
// s = 1 elsewhere (the same value as a rcp(b)); b = 0, +-inf and NaN give the IEEE quotient's special value.
float ki_div(float a, float b)
{
    const float s = abs(b) < 8.5070592e+37f ? 1.0f : 5.4210109e-20f;   // 2^126, 2^-64
    return (a * rcp(b * s)) * s;
}
#else
float ki_log2(float x) { return kf_log2(x); }
float ki_exp2(float x) { return kf_exp2(x); }
float ki_rcp(float x) { return kf_div(1.0f, x); }
float ki_div(float a, float b) { return kf_div(a, b); }
#endif
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
void ki_store_x0(uint x0, int id, int k, float c0, float c1, float c2, float c3)
{
    wg_st64(x0, ((uint)id * 16u + 4u * (uint)k) * 2u,
            uint2(kh_round_bits(c0) | (kh_round_bits(c1) << 16), kh_round_bits(c2) | (kh_round_bits(c3) << 16)));
}
