// SPDX-License-Identifier: MIT
// D3D12 ports of model K's output stage (spec 43): k_out_color, k_out_pixel_c (compact pixel head, the default)
// and k_out_pixel (k_head=full). The forms that read the game's textures and write its output texture (_direct,
// rs=direct, WG_DIRECT_TEXTURES) are declared beside them (see k_in.hlsl). WrongGPU's own code.
// Basis: our HIP kernels in clean/k/kernels/k_io.hip (HIP build k_io.co, wave64, 128 lanes per workgroup), exact forms
// only (FAST = false; the k_fast=1 "_f" kernels are not ported and keep running on HIP). Same names, same argument
// struct (KOutParams, see hlsl/k_out.hlsli for the s-token and why each skipped field is dead), same launch geometry
// and lane <-> pixel mapping (k_out_pixel: 64 x 2 pixels per workgroup, one row per wave64, the F2 group exchange of
// Ybar through LDS between the two waves).
//
// Bit-identity notes (every deviation from the HIP source and why it gives the same bits):
//  * FP32 steps: one IEEE operation each as written (build_dx12.py exact mode), mad() where HIP has kf_fma; products
//    whose factors the driver could regroup (a shared or uniform factor: kappa, e, q, Yo, kk, d, kinv, rho) pass an
//    opaque identity (kof) so each is rounded as HIP rounds it.
//  * FP32 -> FP16: ko_f2h (hlsl/k_out.hlsli): the hardware v_cvt_f16_f32 (round to nearest even, as kh_round) through
//    the driver channel's op 0x1C with its input through the opaque zero (KO_HW16 = 0: the software wg_f2h, same
//    bits); the D3D12 casts round toward zero and are used only for values that are FP16 numbers already (exact).
//  * FP16 steps: native half arithmetic (v_*_f16, nearest even, FP16 denormals kept, as HIP's _Float16), the HIP
//    kernel's packed pairs as packed 32-bit words (kp_*); every FP16 product that a later add could absorb passes the
//    opaque zero (koh / kp_mul: no v_fma / v_fma_mix fusion), and so does every FP16 arithmetic result that is
//    converted to FP32 (cbar, the 5.4 sums, beta, beta_c, the selector denominator: the driver folds an FP16 fma / add
//    followed by the conversion into one v_fma_mix_f32, with no FP16 rounding in between). The E products dv dv,
//    (dv dv) tA and du du feed only a multiply, a multiplicand or an fma addend (no add can absorb them) and go
//    without the opaque (kp_mul_n / kp_mul_s_n; RGA: the same v_mul_f16 / v_fma_f16 / v_fma_mix counts, every square
//    kept). Wave64 kernels get no v_pk_* from this driver; wave32 (with a 16 x 2 tile per wave, F2 by ballot) measured
//    slower (721 vs 640 us), as did wave64 with a 32 x 2 tile per wave (y no longer wave-uniform: +10 us).
//  * Divisions: ko_div (= wg_div, HIP's correctly rounded '/', on every input; wg_div's own steps without its operand
//    scaling where the exponents allow it, wg_div itself elsewhere); kh_div = FP32 quotient rounded once to FP16.
//  * Hardware estimates as HIP: rcp() = v_rcp_f32 (ko_div_axis, ko_selector_normalize, the bounded reciprocals with
//    their Newton step), exp2(half) = v_exp_f16 (ko_exp2h; checked in the RGA listing: v_exp_f16, no FP32 detour).
//  * kio_encode / kio_decode far branches: the binary64 functions (hlsl/k_out.hlsli, equal to HIP's kf_ln / kf_exp2).
//  * HIP's fast addressing forms (KO_ADDR_FAST / KO_WIN_FAST / KO_TAP_FAST: the same loads without per-tap clamps
//    where nothing is clamped) as [branch] forms with the clamped general forms for the border lanes; KO_SOFT_MAX's
//    maximum test as one unsigned maximum of the widened window's FP32 bits (below 0x3F7D8000 = FP16 bits below
//    0x3bec, the same order on every input) (below 0x3bec = every tap in [0, 0.99): no soft
//    limit and unclamped row cubics, HIP's ko_cubic_unit; any other window: soft limit and clamped cubics, the same
//    values as HIP's per-tap decision).
//  * Loops whose first iteration differs are peeled (DXC kept the 12-tap loop with an n == 0 test rolled, the taps then
//    lived in indexed register arrays); the non-plain best-E retry recomputes E from opaque inputs (merged with the
//    main loop's E it kept all of them live: 143 VGPRs).
//  * FP32 denormals flush on D3D12 (wg_math.hlsli); HIP keeps them. The stage's values (FP16 data, FP32 colour) do
//    not produce FP32 denormals; the port check covers the runtime's cases.
//
// @wg-kernel k_out_color co=k_io.co block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56
// @wg-kernel k_out_pixel_c co=k_io.co block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 defines=KO_COMPACT=1
// @wg-kernel k_out_pixel co=k_io.co entry=k_out_pixel_c block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 defines=KO_COMPACT=0
// @wg-kernel k_out_color_direct co=k_io.co entry=k_out_color block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 rs=direct defines=WG_DIRECT_TEXTURES=1
// @wg-kernel k_out_pixel_c_direct co=k_io.co entry=k_out_pixel_c block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 rs=direct defines=KO_COMPACT=1,WG_DIRECT_TEXTURES=1
// @wg-kernel k_out_pixel_direct co=k_io.co entry=k_out_pixel_c block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 rs=direct defines=KO_COMPACT=0,WG_DIRECT_TEXTURES=1
// The __a2 forms: the opt-in close approximation a2 (runtime option k_approx=a2, never the default): divisions on
// v_rcp_f32 with one quotient correction, two reciprocals without their Newton step, every opaque barrier kept
// (hlsl/k_out.hlsli KO_APPROX); not byte-identical to HIP.
// @wg-kernel k_out_color__a2 co=k_io.co entry=k_out_color block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 defines=KO_APPROX=1
// @wg-kernel k_out_pixel_c__a2 co=k_io.co entry=k_out_pixel_c block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 defines=KO_COMPACT=1,KO_APPROX=1
// @wg-kernel k_out_color_direct__a2 co=k_io.co entry=k_out_color block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 rs=direct defines=WG_DIRECT_TEXTURES=1,KO_APPROX=1
// @wg-kernel k_out_pixel_c_direct__a2 co=k_io.co entry=k_out_pixel_c block=128 wave=64 args=s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 rs=direct defines=KO_COMPACT=1,WG_DIRECT_TEXTURES=1,KO_APPROX=1
#include "wg_platform.hlsli"
#if defined(WG_DIRECT_TEXTURES)
#include "native_direct.hlsli"
#endif
#include "k_out.hlsli"

#ifndef KO_COMPACT
#define KO_COMPACT 1
#endif
// The game output's alpha (issue #26): the colour input's alpha of the current frame, spatial only. It is cbar3, the
// fourth channel of the 4.3 bilinear below: the stored tap's A = RN16(the colour texel's fourth value, exactly as the
// game supplies it: no pre-exposure, no exposure scale, no clamp) at the same four render taps as Y / Co / Cg (tap
// positions clamped to the render rectangle), with the same four FP16 weights: RN16(w0 a0), then three FP16 fused
// multiply-adds in the tap order. No history, nothing from the network; HDR and LDR alike. Every output form stores it
// unchanged: the RGBA16F rows as its FP16 bits, the direct texture store and the dense RGBA32F form as its exact FP32
// widening (the dense RGB32F form has no fourth value). HIP's k_io.hip writes the same; the interop's passes carry the
// input alpha into the buffer forms' colour (convert.cpp, model K).

// Per render pixel: the stored colour tap (4.2) and the filter-tap value T (5.3) (HIP k_out_color)
WG_KERNEL(k_out_color)
{
    WG_SETUP();
    const int id = (int)(wg_block_id() * 128u + wg_lane_id());
    if (id >= f_rw * f_rh)
        return;
    const int i = id % f_rw, j = id / f_rw;
    uint2 tap;
    uint tT;
    ko_color_tap(kio_texel(i, j), tap, tT);
    wg_st64(p_ctap, (uint)id * 8u, tap);
    wg_st16(p_ctapT, (uint)id * 2u, tT);
}

groupshared uint lds_ybar[128];

// bilinear sample of the colour history (k_io.hip ko_bilinear4_load / _math, clamp to edge)
float4 ko_bilinear_hc(int w, int h, float cx, float cy)
{
    const float px = cx * (float)w - 0.5f, py = cy * (float)h - 0.5f;
    const float fx0 = floor(px), fy0 = floor(py);
    const float fx = px - fx0, fy = py - fy0;
    int x0 = (int)max(min(fx0, (float)w), -1.0f);
    int y0 = (int)max(min(fy0, (float)h), -1.0f);
    x0 = fx0 != fx0 ? 0 : x0;
    y0 = fy0 != fy0 ? 0 : y0;
    uint2 q00, q10, q01, q11;
    // HIP KO_ADDR_FAST: all four corners inside (no clamp): two 128-bit row loads instead of four 64-bit loads
    [branch] if ((uint)x0 <= (uint)(w - 2) && (uint)y0 <= (uint)(h - 2)) {
        const uint voff = ((uint)y0 * (uint)w + (uint)x0) * 8u, r1 = p_Hc + (uint)w * 8u;
        const uint4 row0 = wg_ld128(p_Hc, voff);
        const uint4 row1 = wg_ld128(r1, voff);
        q00 = row0.xy; q10 = row0.zw;
        q01 = row1.xy; q11 = row1.zw;
    } else {
        const int xa = ko_clampi(x0, w), xb = ko_clampi(x0 + 1, w);
        const int ya = ko_clampi(y0, h), yb = ko_clampi(y0 + 1, h);
        q00 = wg_ld64(p_Hc, ((uint)ya * (uint)w + (uint)xa) * 8u);
        q10 = wg_ld64(p_Hc, ((uint)ya * (uint)w + (uint)xb) * 8u);
        q01 = wg_ld64(p_Hc, ((uint)yb * (uint)w + (uint)xa) * 8u);
        q11 = wg_ld64(p_Hc, ((uint)yb * (uint)w + (uint)xb) * 8u);
    }
    const float omfy = 1.0f - fy;
    const float4 v00 = float4(f16tof32(q00.x), wg_h2f_hi(q00.x), f16tof32(q00.y), wg_h2f_hi(q00.y));
    const float4 v10 = float4(f16tof32(q10.x), wg_h2f_hi(q10.x), f16tof32(q10.y), wg_h2f_hi(q10.y));
    const float4 v01 = float4(f16tof32(q01.x), wg_h2f_hi(q01.x), f16tof32(q01.y), wg_h2f_hi(q01.y));
    const float4 v11 = float4(f16tof32(q11.x), wg_h2f_hi(q11.x), f16tof32(q11.y), wg_h2f_hi(q11.y));
    const float4 t0 = v00 + (v10 - v00) * fx;
    const float4 t1 = v01 + (v11 - v01) * fx;
    return t0 * omfy + t1 * fy;
}

// E of the 12 taps (index 3 l + tc) for a sub-sample pair at (us, vs), in the spec's operation order (HIP each_E); FP16
// pairs as packed words (kp_*). lines[l] = v0 + (half)l to avoid holding 4 extra half VGPRs.
void ko_each_E(uint us, uint vs, half v0, half cols[4], half tA, half tB, half tC, out uint E[12])
{
    [unroll] for (int l = 0; l < 4; ++l) {
        const uint dv = kp_rsub_s(v0 + (half)l, vs);
        const uint t2v = kp_mul_s_n(kp_mul_n(dv, dv), tA);
        const uint t3v = kp_mul_s(dv, tB);
        uint du = kp_rsub_s(cols[l], us);
        uint mm = kp_fma(du, t3v, t2v);
        [unroll] for (int tc = 0; tc < 3; ++tc) {
            if (tc > 0) {
                du = kp_add_s(du, (half)1);
                mm = kp_add(mm, t3v);
            }
            E[3 * l + tc] = kp_fma_s(kp_mul_n(du, du), tC, mm);
        }
    }
}

// 3.6 luma history value of the four sub-samples from the (soft-limited) 5 x 5 window: 10 unique row cubics computed
// one column j at a time (5 VGPRs rc[5] instead of 10, dropping 6 duplicate row cubics = 30 FMAs, 100% bit-identical).
void ko_ellv(float Lw[5][5], KoAxis axx, KoAxis axy, bool unit, out float ellv[4])
{
    [unroll] for (int j = 0; j < 2; ++j) {
        float rc[5];
        [unroll] for (int r = 0; r < 5; ++r)
            rc[r] = unit ? ko_cubic_unit(Lw[r][j], Lw[r][j + 1], Lw[r][j + 2], Lw[r][j + 3], axx)
                         : ko_cubic(Lw[r][j], Lw[r][j + 1], Lw[r][j + 2], Lw[r][j + 3], axx);
        [unroll] for (int i = 0; i < 2; ++i) {
            const float val = ko_cubic(rc[i], rc[i + 1], rc[i + 2], rc[i + 3], axy);
            const float lo = min(min(Lw[i + 1][j + 1], Lw[i + 1][j + 2]), min(Lw[i + 2][j + 1], Lw[i + 2][j + 2]));
            const float hi = max(max(Lw[i + 1][j + 1], Lw[i + 1][j + 2]), max(Lw[i + 2][j + 1], Lw[i + 2][j + 2]));
            ellv[j + 2 * i] = min(max(val, lo), hi);
        }
    }
}

#if defined(WG_DIRECT_TEXTURES)
// the game output of a pixel: with output_rne16 the HIP output-copy bytes of an RGBA16F target (RGB rounded to nearest
// even as KIO_OUT_RGBA16F writes them), stored as the exact FP16 values (the typed store keeps them); else the FP32
// values (the HIP path's dense f32 output goes through the Converter's typed store). Alpha: A (cbar3), an FP16 value,
// as its exact FP32 widening in both cases
void ko_store_out(int x, int y, float R, float G, float B, half A)
{
    float3 v = float3(R, G, B);
    if (nd_output_rne16 != 0u)
        v = float3(f16tof32(ko_f2h(R)), f16tof32(ko_f2h(G)), f16tof32(ko_f2h(B)));
    native_direct_output[uint2((uint)x, (uint)y)] = float4(v, (float)A);
}
#endif

// spec 43 output stage per pixel (HIP ko_pixel<COMPACT, false>)
WG_KERNEL(k_out_pixel_c)
{
    WG_SETUP();
    const int W16 = kio_ceil16(f_W), H16 = kio_ceil16(f_H);
    const int lane = (int)wg_lane_id(), tiles_x = (W16 + 63) / 64;
    const int tile = (int)wg_block_id();
    const int x_raw = (tile % tiles_x) * 64 + (lane & 63);
    const bool active = x_raw < W16;
    const int x = active ? x_raw : W16 - 1, y = (tile / tiles_x) * 2 + wg_uniform(lane >> 6);
    const bool hdr = (f_flags & KIO_HDR) != 0, mvlo = (f_flags & KIO_MVLO) != 0;
    const float px = (float)x + 0.5f, py = WaveReadLaneFirst((float)y + 0.5f);
    const float rho = WaveReadLaneFirst(ko_rho());

    // 3.1 network channels
    const int u = (int)trunc(f_s * (float)x), v = WaveReadLaneFirst((int)trunc(f_s * (float)y));
    const int k = (u & 1) + 2 * (v & 1);
    const int bxn = ki_min(u >> 1, f_Bw - 1), byn = WaveReadLaneFirst(ki_min(v >> 1, f_Bh - 1));
#if KO_COMPACT
    const uint2 nvw = wg_ld64(p_N, (((uint)(2 * byn + (k >> 1)) * (2u * (uint)f_Bw) + (uint)(2 * bxn + (k & 1))) * 4u) * 2u);
    const half nv0 = kh_bits(nvw.x & 0xffffu), nv1 = kh_bits(nvw.x >> 16), nv9 = kh_bits(nvw.y & 0xffffu);
#else
    const uint ncb = (((uint)byn * (uint)f_Bw + (uint)bxn) * 40u + 10u * (uint)k) * 2u;
    const half nv0 = kh_bits(wg_ld16(p_N, ncb)), nv1 = kh_bits(wg_ld16(p_N, ncb + 2u)), nv9 = kh_bits(wg_ld16(p_N, ncb + 18u));
#endif

    // 4.1 local origin, 4.3 current-colour taps
    const float o_x = kf_R(mad((float)(16 * (x / 16)) + 0.5f, u_rhox, u_jmx)) - 4.0f;
    const float o_y = WaveReadLaneFirst(kf_R(mad((float)(16 * (y / 16)) + 0.5f, u_rhoy, u_jmy)) - 4.0f);
    const float Pxf = mad(px, u_rhox, u_jmx) - o_x, Pyf = WaveReadLaneFirst(mad(py, u_rhoy, u_jmy) - o_y);
    const half Ux = kh_round(Pxf), Uy = kh_bits(WaveReadLaneFirst(ko_f2h(Pyf)));
    uint cbar_lo, cbar_hi;
    {   // ko_cbar_load / ko_cbar_math
        const half flx = floor(Ux), fly = floor(Uy);
        const half fx = kh_sat(Ux - flx), fy = kh_sat(Uy - fly);
        const half one = (half)1;
        half w[4];
        w[0] = kh_mul(one - fx, one - fy);
        w[1] = kh_mul(fx, one - fy);
        w[2] = kh_mul(one - fx, fy);
        w[3] = kh_mul(fx, fy);
        const int bx = (int)(o_x + (float)flx), by = WaveReadLaneFirst((int)(o_y + (float)fly));
        uint2 tq[4];
        // HIP KO_ADDR_FAST: the 2 x 2 taps inside the render image (no clamp): two 128-bit row loads
        [branch] if (bx >= 0 && bx <= f_rw - 2 && by >= 0 && by <= f_rh - 2) {
            const uint voff = ((uint)by * (uint)f_rw + (uint)bx) * 8u;
            const uint4 cr0 = wg_ld128(p_ctap, voff);
            const uint4 cr1 = wg_ld128(p_ctap + (uint)f_rw * 8u, voff);
            tq[0] = cr0.xy; tq[1] = cr0.zw;
            tq[2] = cr1.xy; tq[3] = cr1.zw;
        } else {
            [unroll] for (int q = 0; q < 4; ++q) {
                const int tx = ko_clampi(bx + (q & 1), f_rw), ty = ko_clampi(by + (q >> 1), f_rh);
                tq[q] = wg_ld64(p_ctap, ((uint)ty * (uint)f_rw + (uint)tx) * 8u);
            }
        }
        uint lo = 0u, hi = 0u;
        [unroll] for (int q = 0; q < 4; ++q) {
            const uint2 t = tq[q];
            lo = q == 0 ? kp_mul_s(t.x, w[q]) : kp_fma_s(t.x, w[q], lo);
            hi = q == 0 ? kp_mul_s(t.y, w[q]) : kp_fma_s(t.y, w[q], hi);
        }
        // FP16 results converted to FP32 below pass the opaque (kp_opq / koh): the driver folds an FP16 fma / add
        // followed by the conversion into one v_fma_mix_f32 (no FP16 rounding in between)
        cbar_lo = kp_opq(lo);
        cbar_hi = kp_opq(hi);
    }

    // 3.3 motion
    float mvx, mvy;
    if (mvlo) {
        const int ix = (int)kf_R(mad(px, u_rhox, u_jmx)), iy = WaveReadLaneFirst((int)kf_R(mad(py, u_rhoy, u_jmy)));
        int dx, dy;
        ko_nearest(ix, iy, dx, dy);
        ko_motion(ko_clampi(ix + dx, f_rw), ko_clampi(iy + dy, f_rh), mvx, mvy);
    } else {
        const int mx = (int)kf_R(mad(px, 1.0f, -0.5f)), my = WaveReadLaneFirst((int)kf_R(mad(py, 1.0f, -0.5f)));
        ko_motion(ko_clampi(mx, f_W), ko_clampi(my, f_H), mvx, mvy);
    }
    // the texture's motion scale folded into the factor and its offset in raw units, as the buffer form gets them in
    // WgRtFrame (hlsl/k_out.hlsli ko_motion; issue #26): direct forms b1's ms / mo, KIO_M16 f_rmsx ..
#if defined(WG_DIRECT_TEXTURES)
    const bool dfold = true;
    const float dmsx = nd_msx, dmsy = nd_msy, dmox = nd_mox, dmoy = nd_moy;
#else
    const bool dfold = (f_flags & KIO_M16) != 0;
    const float dmsx = f_rmsx, dmsy = f_rmsy, dmox = f_rmox, dmoy = f_rmoy;
#endif
    float spx = u_spx, spy = u_spy, ofx = f_mox, ofy = f_moy;
    [branch] if (dfold) {
        spx = ko_dsp(f_sx, dmsx, u_rhox, mvlo);
        spy = ko_dsp(f_sy, dmsy, u_rhoy, mvlo);
        ofx = ko_dmo(f_mox, dmox, dmsx);
        ofy = ko_dmo(f_moy, dmoy, dmsy);
    }
    const float wx = mad(spx, mvx + ofx, px), wy = mad(spy, mvy + ofy, py);

    // 3.4 validity
    bool valid = (f_flags & KIO_RESET) == 0 && wx * u_invW >= 0.0f && wx * u_invW <= 1.0f && wy * u_invH >= 0.0f &&
                 wy * u_invH <= 1.0f;

    // F2: Ybar of the aligned 2 x 2 group through LDS (lanes lane, lane ^ 1, lane ^ 64, lane ^ 65)
    lds_ybar[lane] = cbar_lo & 0xffffu;

    // 3.6 luma history: the 5 x 5 window of the four sub-samples (25 two-byte loads on interior fast path, 10 unique
    // row cubics in ko_ellv; Lmax16 < 0x3becu is exact-equivalent to Lmax32 < 0x3F7D8000u on all 65,536 FP16 patterns;
    // computed before Hh so Hh[4] does not stay live across Lw[5][5])
    const KoAxis axx = ko_axis(wx), axy = ko_axis(wy);
    const int LW = 2 * W16, LH = 2 * H16;
    float Lw[5][5];
    uint Lmax16 = 0u;
    [branch] if ((uint)(axx.n - 1) <= (uint)(LW - 5) && (uint)(axy.n - 1) <= (uint)(LH - 5)) {
        const uint voff = ((uint)(axy.n - 1) * (uint)LW + (uint)(axx.n - 1)) * 2u;
        [unroll] for (int r = 0; r < 5; ++r)
            [unroll] for (int c = 0; c < 5; ++c) {
                const uint h = wg_ld16(p_L + (uint)(r * LW) * 2u + 2u * (uint)c, voff);
                Lmax16 = max(Lmax16, h);
                Lw[r][c] = f16tof32(h);
            }
    } else {
        [unroll] for (int r = 0; r < 5; ++r) {
            const uint rb = (uint)ko_clampi(axy.n - 1 + r, LH) * (uint)LW;
            [unroll] for (int c = 0; c < 5; ++c) {
                const uint h = wg_ld16(p_L, (rb + (uint)ko_clampi(axx.n - 1 + c, LW)) * 2u);
                Lmax16 = max(Lmax16, h);
                Lw[r][c] = f16tof32(h);
            }
        }
    }
    float ellv[4];
    [branch] if (Lmax16 < 0x3becu) {
        ko_ellv(Lw, axx, axy, true, ellv);
    } else {
        [unroll] for (int r = 0; r < 5; ++r)
            [unroll] for (int c = 0; c < 5; ++c) Lw[r][c] = ko_ell(kh_exact(Lw[r][c]));
        ko_ellv(Lw, axx, axy, false, ellv);
    }

    // 3.5 colour history (after Lw[5][5] is dead)
    float4 Hh = ko_bilinear_hc(W16, H16, wx * u_invW16, wy * u_invH16);

    // 3.2 blend logit (|t| <= 3 keeps 9 t^2 + 27 in [27, 108]: ko_div_inrange is exact without a wg_div cold branch)
    const float t = 0.5f * (float)nv9;
    float alpha;
    if (t < -3.0f) {
        alpha = 0.0f;
    } else if (t > 3.0f) {
        alpha = 1.0f;
    } else {
        const float t2 = t * t;
        const float g = (t == t) ? ko_div_inrange(t * (t2 + 27.0f), mad(t2, 9.0f, 27.0f)) : t;
        alpha = mad(g, 0.5f, 0.5f);
    }

    // 3.8 / 3.9 history luma
    float Hl[4];
    if (hdr) {
        bool near = true;
        [unroll] for (int s = 0; s < 4; ++s) near = near && ellv[s] >= 0.0f && ellv[s] < kKioDecodeNear;
        if (near) {
            float lin[4];
            [unroll] for (int s = 0; s < 4; ++s) {
                lin[s] = rho * kof(ellv[s] * ko_rcp_output_bounded(1.0f - ellv[s]));
                near = near && lin[s] >= 0.0f && lin[s] < kKioEncodeNear;
            }
            if (near)
                [unroll] for (int s = 0; s < 4; ++s) Hl[s] = lin[s] * ko_rcp_output_bounded(lin[s] + 1.0f);
        }
        [branch] if (!near) {
            [loop] for (int s = 0; s < 4 + (int)ko_z; ++s) {
                const float lin = rho * kof(ellv[s] * kio_decode(ellv[s]));
                Hl[s] = lin * kio_encode(lin);
            }
        }
    } else {
        [unroll] for (int s = 0; s < 4; ++s) Hl[s] = rho * ellv[s];
    }
    // 3.10 confidence
    const float chk = mad(alpha, Hh.z, mad(Hh.x, Hh.y, mad(Hl[0], Hl[1], Hl[2] * Hl[3])));
    valid = valid && kf_finite(chk);
    const float alphap = valid ? alpha : 1.0f, bp = valid ? Hh.w : 1.0f;
    if (!valid) {
        [unroll] for (int s = 0; s < 4; ++s) Hl[s] = 0.0f;
        Hh = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float m0 = mad(alphap, 1.0f - bp, bp);
    const float m1 = mad(bp, 1.0f - alphap * alphap, alphap * alphap);
    const bool F0 = (float)kh_bits(0x251fu) > m0;   // (float)(h16)0.02f
    const bool F1 = m0 < 0.25f && alpha < 0.25f;
    // Pack mh16 and age16 together so m0, m1, and alpha are dead before 5.1-5.6
    const uint ma_r = ko_f2h(m0) | (ko_f2h(min(max(ko_div(m1, m1 + 1.0f), 0.01f), 1.0f)) << 16);

    // F2 (Ybar of the group, stored above)
    GroupMemoryBarrierWithGroupSync();
    const int g0 = lane & ~65;
    const half c08 = kh_bits(0x3a66u);   // (h16)0.8f
    const bool F2 = kh_bits(lds_ybar[g0]) > c08 || kh_bits(lds_ybar[g0 + 1]) > c08 ||
                    kh_bits(lds_ybar[g0 + 64]) > c08 || kh_bits(lds_ybar[g0 + 65]) > c08;

    // 5.1 table index
    int qs[2];
    [unroll] for (int c = 0; c < 2; ++c) {
        const half z = ko_selector_normalize(c == 0 ? nv0 : nv1);
        const int qv = (int)(float)round(kh_mul(z, (half)8)) + 8;
        qs[c] = ki_max(ki_min(qv, 15), 0);
    }
    const int tix = (16 * qs[0] + qs[1]) + 256 * ((int)F0 + 2 * (int)F1 + 4 * (int)F2);
    const uint4 ent = wg_mem.Load4((p_table + (uint)tix * 16u) & ~15u);
    const half tA = kh_bits(ent.x & 0xffffu), tC = kh_bits(ent.x >> 16), tB = kh_bits(ent.y & 0xffffu);
    const half tS = kh_bits(ent.y >> 16), tK = kh_bits(ent.z >> 16), tk6 = kh_bits(ent.w & 0xffffu);
    const bool O = (ent.z & 0xffffu) != 0u;

    // 5.2 line coordinates
    half Xs[2], Ys[2];
    [unroll] for (int s = 0; s < 2; ++s) {
        Xs[s] = kh_round(mad((float)(2 * x + s) + 0.5f, u_rho2x, u_jmx) - o_x);
        Ys[s] = kh_bits(WaveReadLaneFirst(ko_f2h(mad((float)(2 * y + s) + 0.5f, u_rho2y, u_jmy) - o_y)));
    }
    const half uP = O ? Uy : Ux, vP = O ? Ux : Uy;

    // 5.3 taps (keep T[12] as half so driver doesn't unpack, drop lines[4] via v0 + (half)l, drop us2/vs2/gam[4])
    const half one = (half)1;
    const half v0 = floor(vP) - one;
    half cl = uP + kh_mul(v0 - vP, tS);
    half cols[4];
    half T[12];
    const float o_ln = O ? o_x : o_y, o_cl = O ? o_y : o_x;
    const int n_ln = O ? f_rw : f_rh, n_cl = O ? f_rh : f_rw;
    int as_[4];
    bool tfast = abs(o_cl) < 32768.0f;
    int cb[4];
    [unroll] for (int l = 0; l < 4; ++l) {
        if (l > 0) cl = cl + tS;
        const half ln_l = v0 + (half)l;
        const half col_l = round(cl) - one;
        cols[l] = col_l;
        as_[l] = ko_clampi((int)(o_ln + (float)ln_l), n_ln);
        cb[l] = (int)(o_cl + (float)col_l);
        tfast = tfast && abs((float)col_l) <= 2045.0f && cb[l] >= 0 && cb[l] <= n_cl - 3;
    }
    uint toff[12];
    [branch] if (tfast) {
        const uint st = O ? (uint)f_rw * 2u : 2u;
        [unroll] for (int l2 = 0; l2 < 4; ++l2) {
            const uint b = ((uint)(O ? cb[l2] : as_[l2]) * (uint)f_rw + (uint)(O ? as_[l2] : cb[l2])) * 2u;
            [unroll] for (int tc = 0; tc < 3; ++tc) toff[3 * l2 + tc] = b + (uint)tc * st;
        }
    } else {
        [unroll] for (int l2 = 0; l2 < 4; ++l2) {
            const half col_l = cols[l2];
            [unroll] for (int tc = 0; tc < 3; ++tc) {
                const half colv = col_l + (half)tc;
                const int c = ko_clampi((int)(o_cl + (float)colv), n_cl);
                const uint row = (uint)(O ? c : as_[l2]), col = (uint)(O ? as_[l2] : c);
                toff[3 * l2 + tc] = (row * (uint)f_rw + col) * 2u;
            }
        }
    }
    [unroll] for (int n2 = 0; n2 < 12; ++n2) T[n2] = kh_bits(wg_ld16(p_ctapT, toff[n2]));

    // 5.4 - 5.6 filtered values, 6.1 blend weights (sub-samples (i, 0), (i, 1) as the halves of half2 values)
    const half kux = kh_bits(u_kuxkvy & 0xffffu), kvy = kh_bits(u_kuxkvy >> 16);
    const half k_u = O ? kvy : kux, k_v = O ? kux : kvy;
    half Fv[4];
    const uint xs2 = kp_pack(Xs[0], Xs[1]);
    const uint m12 = 0xca00ca00u;   // (h16)-12 twice
    bool plain = true;
    uint gam2[2];
    [unroll] for (int i2 = 0; i2 < 2; ++i2) {
        const uint ys2 = kp_splat(Ys[i2]);
        const uint us = O ? ys2 : xs2, vs = O ? xs2 : ys2;
        uint E[12];
        ko_each_E(us, vs, v0, cols, tA, tB, tC, E);
        uint gg = kh2_u(exp2(kh2_h(E[0])));   // ko_exp2h2: two v_exp_f16
        uint sgT = kp_mul_s(gg, T[0]), sg = gg, maxE = E[0];
        [unroll] for (int n = 1; n < 12; ++n) {
            gg = kh2_u(exp2(kh2_h(E[n])));
            sgT = kp_fma_s(gg, T[n], sgT);
            sg = kp_add(gg, sg);
            maxE = kp_max(maxE, E[n]);
        }
        const uint sgTo = kp_opq(sgT), sgo = kp_opq(sg);   // (converted to FP32 by kh_div)
        Fv[2 * i2] = kh_div(kp_lo(sgTo), kp_lo(sgo));
        Fv[2 * i2 + 1] = kh_div(kp_hi(sgTo), kp_hi(sgo));
        plain = plain && kp_lo(maxE) >= kp_lo(m12) && kp_hi(maxE) >= kp_hi(m12) && kp_lo(sg) == kp_lo(sg) &&
                kp_hi(sg) == kp_hi(sg);
        const uint dvf = kp_sub(kp_rni(vs), vs);
        const uint up = kp_fma_s(dvf, tS, us);
        const uint duf = kp_sub(kp_rni(up), up);
        const half c45 = kh_bits(0x4480u);   // (h16)4.5
        const uint wu = kp_sat_nz(kp_fma_ss(kp_abs(duf), k_u, c45));
        const uint wv = kp_sat_nz(kp_fma_ss(kp_abs(dvf), k_v, c45));
        gam2[i2] = kp_mul(kp_fma_ss(wv, tK, tk6), wu);
    }
    [branch] if (!plain) {
        [unroll] for (int i3 = 0; i3 < 2; ++i3) {
            const uint ys2_o = kp_splat(Ys[i3]);
            const uint us_o = (O ? ys2_o : xs2) ^ (ko_z * 0x10001u), vs_o = (O ? xs2 : ys2_o) ^ (ko_z * 0x10001u);
            uint E[12];
            ko_each_E(us_o, vs_o, v0, cols, tA, tB, tC, E);
            uint bestE = E[0];
            half bTl = T[0], bTh = T[0];
            [unroll] for (int n = 1; n < 12; ++n) {
                const uint d = kp_sub(E[n], bestE);
                const bool kl = (d & 0xffffu) - 0x8000u <= 0x7c00u, kh = (d >> 16) - 0x8000u <= 0x7c00u;
                bestE = (kl ? bestE & 0xffffu : E[n] & 0xffffu) | (kh ? bestE & 0xffff0000u : E[n] & 0xffff0000u);
                bTl = kl ? bTl : T[n];
                bTh = kh ? bTh : T[n];
            }
            if (kp_lo(bestE) < kp_lo(m12)) Fv[2 * i3] = bTl;
            if (kp_hi(bestE) < kp_hi(m12)) Fv[2 * i3 + 1] = bTh;
        }
    }
    const half gbar = kh_mul((half)0.25, (kp_lo(gam2[0]) + kp_lo(gam2[1])) + (kp_hi(gam2[0]) + kp_hi(gam2[1])));
    const half mh = kh_bits(ma_r & 0xffffu);
    const half MU = kh_bits(0x27aeu), NU = kh_bits(0x3c20u);   // (h16)0.03f, (h16)(1.0 / 0.97)
    const half rr = kh_bits(0x2e66u) + kh_mul(kh_bits(0x3b33u), mh);   // 0.1 + 0.9 mh
    const half d0 = kh_mul(mh - MU, NU);
    const half d1 = kh_mul(rr - MU, NU);
    const half beta_c = koh(kh_sat(mad(gbar, rr - d1, d1)));   // (converted to FP32 in 6.5)

    // 6.2 guard
    half Ybar = kp_lo(cbar_lo), Cobar = kp_hi(cbar_lo), Cgbar = kp_lo(cbar_hi);
    const half cbar3 = kp_hi(cbar_hi);
    const float guard = kof(kof(kof((float)Ybar * (float)Fv[0]) * (float)Fv[1]) * (float)Fv[2]) * (float)Fv[3];
    if (!kf_finite(guard)) {
        [unroll] for (int s = 0; s < 4; ++s) Fv[s] = (half)0;
        Ybar = (half)0;
        Cobar = (half)0;
        Cgbar = (half)0;
    }

    // 6.3 / 6.4 new luma, luma history out
    uint beta2[2];
    beta2[0] = kp_sat_nz(kp_fma_ss(gam2[0], mh - d0, d0));
    beta2[1] = kp_sat_nz(kp_fma_ss(gam2[1], mh - d0, d0));
    beta2[0] = kp_opq(beta2[0]);   // (converted to FP32 below)
    beta2[1] = kp_opq(beta2[1]);
    float Nn[4];
    uint Lo[4];
    [unroll] for (int s2 = 0; s2 < 4; ++s2) {
        const half beta = (s2 & 1) ? kp_hi(beta2[s2 >> 1]) : kp_lo(beta2[s2 >> 1]);
        Nn[s2] = mad((float)beta, (float)Fv[s2] - Hl[s2], Hl[s2]);
        const float c99 = kf_bits(0x3F7D70A4u);
        const float lv = Nn[s2] < c99 ? Nn[s2] : mad(Nn[s2] - c99, 1000.0f, c99);
        Lo[s2] = ko_f2h(lv);
    }
    if (active) {
        [unroll] for (int r2 = 0; r2 < 2; ++r2)
            wg_st32(p_Lout, ((uint)(2 * y + r2) * (uint)LW + (uint)(2 * x)) * 2u, Lo[2 * r2] | (Lo[2 * r2 + 1] << 16));
    }

    // 6.5 colour
    const float bc = (float)beta_c;
    const float Yc = mad(Hh.x, 1.0f - bc, bc * (float)Ybar);
    const float Coc = mad(Hh.y, 1.0f - bc, bc * (float)Cobar);
    const float Cgc = mad(Hh.z, 1.0f - bc, bc * (float)Cgbar);
    const float eps = kf_bits(0x3727C5ACu);
    const float q = ko_rcp_output_bounded(max(Yc, eps));
    const float rCo = kof(Coc * q), rCg = kof(Cgc * q);
    const float Yo = 0.25f * (Nn[3] + (Nn[2] + (Nn[0] + Nn[1])));
    const float Coo = kof(rCo * Yo), Cgo = kof(rCg * Yo);

    // 6.6 colour history out
    const float Yp = max(Yo, 0.0f);
    const float s1 = (Cgo - eps < -Yp) ? eps - Cgo : Yp;
    const float s2 = ((Coo - Cgo) - eps < -s1) ? (Cgo - Coo) + eps : s1;
    const float s3 = ((-Coo - Cgo) - eps < -s2) ? (Coo + Cgo) + eps : s2;
    const float kk = s3 <= Yp ? 1.0f : ko_div(Yp, s3);
    if (!active)
        return;
    wg_st64(p_Hcout, ((uint)y * (uint)W16 + (uint)x) * 8u,
            uint2(ko_f2h(Yp) | (ko_f2h(Coo * kk) << 16), ko_f2h(Cgo * kk) | ((ma_r >> 16) << 16)));

    // 6.7 game output
    if (x < f_W && y < f_H) {
        float Yg = Yo, Cog = Coo, Cgg = Cgo;
        if (hdr) {
            float d;
            [branch] if (Yo < kKioDecodeNear) d = ko_rcp_refined(1.0f - Yo);
            else d = kio_decode_far(Yo);
            const float kinv = WaveReadLaneFirst(ko_kinv());
            Yg = kinv * kof(Yo * d);
            Cog = kinv * kof(Coo * d);
            Cgg = kinv * kof(Cgo * d);
        }
        const float R = (Yg + Cog) - Cgg, G = Cgg + mad(Cog, kof(0.0f), Yg), B = (Yg - Cog) - Cgg;
#if defined(WG_DIRECT_TEXTURES)
        ko_store_out(x, y, R, G, B, cbar3);
#else
        if (p_omode == KIO_OUT_RGBA16F) {
            wg_st64(p_out, ((uint)y * (uint)p_opitch + (uint)x) * 8u, uint2(ko_f2h(R) | (ko_f2h(G) << 16), ko_f2h(B) | (kh_u(cbar3) << 16)));
        } else if (p_omode == KIO_OUT_RGB32F) {
            const uint o = ((uint)y * (uint)p_opitch + (uint)x) * 12u;
            wg_stf(p_out, o, R);
            wg_stf(p_out, o + 4u, G);
            wg_stf(p_out, o + 8u, B);
        } else {
            wg_st128(p_out, ((uint)y * (uint)p_opitch + (uint)x) * 16u, asuint(float4(R, G, B, (float)cbar3)));
        }
#endif
    }
}
