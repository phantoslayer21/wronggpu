// SPDX-License-Identifier: GPL-3.0-only
// D3D12 port of Model L's reconstruction kernel (issue #89): l_post_recon of l_post.co (clean/l/kernels/l_post.hip, HDR),
// byte for byte. Arguments, buffers, layouts and grid are Model M's post_recon (clean/kernels/post.hip): grid
// ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes; lane (l, k) of wave k handles D column x0 + l and the
// row pair k0 + k. w = the post blob (w[0] = s). Model L's changes against post_recon: the 4 x 4 colour gather in the
// model's precision (FP16 features, offsets, exponent and accumulators), linear-colour history candidates
// L = D(h) r_prev (r_prev = 1 / e_prev) combined as fma(w1, L1, w2 L2) e, and the render pixel's colour as the
// fallback.
//
// The operations follow HIP's ISA of l_post_recon (llvm-objdump of l_post.co), not only its source:
// - E_in(e c) and the B / A codes k / 255 (an IEEE division: wg_div) are rounded to FP16 by v_cvt_f16_f32 (wg_f2h);
//   the kernel shape A, B, C = (_Float16)(k * const) is one rounding (v_fma_mixlo_f16 k, const, 0: wg_mul_f16);
// - the gather's FP16 products and sums are v_mul_f16 / v_add_f16, the exponent and the accumulators v_fma_f16 (mad on
//   half). The driver fuses an FP16 multiply with a following add (wg_math.hlsli), so each FP16 product passes through
//   opqh (an integer xor with a zero it cannot see); the accumulators start from an opaque +0 (fma(w, f, +0) as HIP);
// - an FP16 result that is widened to FP32 (eps for the exponential, the accumulators for the normalization) passes
//   through opqh first: otherwise the driver computes its last fma in FP32 (v_fma_mix_f32) and drops the FP16 rounding;
// - the tap weight w = 2^eps (v_exp_f32) is flushed to +0 below 2^-126 and rounded to FP16 by v_cvt_f16_f32 (wg_f2h);
// - the full history's texel is (_Float16)(kEncK log2(..)) in one rounding (v_fma_mixlo_f16: wg_mul_f16); the display
//   value is clamp(E(o), 0, 2.45) rounded by v_cvt_f16_f32 (wg_f2h);
// - r_prev = 1 / e_prev is one IEEE division per launch (wg_div), then L = D(h) r_prev (HIP: the same product; the
//   spec's division as a product with its reciprocal is free within the post tolerance, spec answer Q5).
// Model M's DXILs are untouched: this file only adds the kernel (bound to l_post.co).
// l_post_recon_p12 (LP_P12 = 1): the same kernel on the pack12 coefficients of l_coef_pack12: A = image A8 (8 bytes per
// render pixel: codes A0 A1 A2 B0 | B1 B2, C2 FP16), C = image C4 (4 bytes per render pixel: C0, C1 FP16), B unused.
//
// LDR forms (LP_LDR = 1; l_post.hip's l_post_recon_ldr / l_post_recon_p12_ldr, spec preset L part 3 item 6, Q1, Q2):
// identity transfer (the colour feature is the driver's FP16 conversion of the colour, a NaN kept; no decode of the
// history candidates or of C), no exposure (not read: e = e_prev = 1), the full history the driver's FP16 conversion of
// H^, the display image h(code RN32(1 / N)) of the 11 / 11 / 10-bit code floor(clamp(fused(O^ N + 1/2), 0, N)) (a NaN
// gives 0), the reduced codes rint(clamp(mean, 0, 1) 1023).
//
// @wg-kernel l_post_recon co=l_post.co block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p
// @wg-kernel l_post_recon_p12 co=l_post.co entry=l_post_recon_p12 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1
// @wg-kernel l_post_recon_ldr co=l_post.co entry=l_post_recon_ldr block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_LDR=1
// @wg-kernel l_post_recon_p12_ldr co=l_post.co entry=l_post_recon_p12_ldr block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_LDR=1
//
// LP_C12 = 1 (Q11 part C / 0.6.5 hybrid_full: runtime WG_L_POST=c12, and WG_L_FORM=hybrid on native D3D12; port of
// l_post__lc12.co = l_post.hip with WG_LP_C1 = 1, WG_CR_C2 = 1, tag "lc12"): the gather exponent in the machine-code form
// eps = fma(h(u v), C, fma(h(v v), B, h(h(u u) A))) (C1) and the Catmull-Rom weights' fused forms (C2: f2 = f f,
// f3 = f f2, w0 = fused(-(f + f3), 1/2, f2), w1 = fused(f3, 1.5, -RN(2.5 f2)) + 1, w3 = RN(RN(f3 - f2) 1/2),
// w2 = ((1 - w0) - w1) - w3). The default entries above are unchanged (LP_C12 = 0).
// @wg-kernel lc12/l_post_recon co=l_post__lc12.co entry=l_post_recon block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_C12=1
// @wg-kernel lc12/l_post_recon_p12 co=l_post__lc12.co entry=l_post_recon_p12 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_C12=1
// @wg-kernel lc12/l_post_recon_ldr co=l_post__lc12.co entry=l_post_recon_ldr block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_LDR=1,LP_C12=1
// @wg-kernel lc12/l_post_recon_p12_ldr co=l_post__lc12.co entry=l_post_recon_p12_ldr block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_LDR=1,LP_C12=1
//
// The wave64 form (LP_W64 = 1; runtime WG_L_POST_W64, default off, 1 = this port; 0 = the lc12 entries above): the same kernel (the same
// operations on the same values in the same order, every FP16 rounding kept) in 64-lane waves, tag w64_lc12 (the same
// code object l_post__lc12.co as lc12: the tag only selects the port). The 256-lane workgroup is unchanged (lane (l, k):
// column l & 31 of the row pair lane >> 5), so a wave holds two row pairs. Differences against the lc12 entries, each
// value-preserving: LP_CODE 1 (the FP16 code table is built from one multiplication by the FP32 1 / 255 and the driver's
// conversion instead of an IEEE division per lane: f16(f32(k) f32(1 / 255)) = f16(RN32(k / 255)) for k = 0 ... 255,
// checked for all 256 codes against the division), LP_RPUNI 1 (r_prev = 1 / e_prev is one IEEE division read from the first
// lane: the operand is wave-uniform, so every lane's quotient is the same value), LP_QX 1 (the lane x + 1 of the 2 x 2 block
// is the quad neighbour: QuadReadAcrossX is the lane ^ 1 exchange of wg_xor(v, 1), DPP instead of ds_bpermute), LP_WTEX
// (the wave's first lane of the staging rounds: the wave size in place of 32).
// @wg-kernel w64_lc12/l_post_recon co=l_post__lc12.co entry=l_post_recon block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_C12=1,LP_W64=1,LP_CODE=1,LP_RPUNI=1,LP_QX=1
// @wg-kernel w64_lc12/l_post_recon_p12 co=l_post__lc12.co entry=l_post_recon_p12 block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_C12=1,LP_W64=1,LP_CODE=1,LP_RPUNI=1,LP_QX=1
// @wg-kernel w64_lc12/l_post_recon_ldr co=l_post__lc12.co entry=l_post_recon_ldr block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_LDR=1,LP_C12=1,LP_W64=1,LP_CODE=1,LP_RPUNI=1,LP_QX=1
// @wg-kernel w64_lc12/l_post_recon_p12_ldr co=l_post__lc12.co entry=l_post_recon_p12_ldr block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_LDR=1,LP_C12=1,LP_W64=1,LP_CODE=1,LP_RPUNI=1,LP_QX=1
//
// LP_MORIG = 1 (WG_M_POST=orig on native D3D12: Model M's original reconstruction, 3 x 3 gather + encoded-domain
// history candidates, with C1/C2 machine-code forms; port of l_post__morig.co = l_post.hip with WG_LP_M, WG_LP_C1, WG_CR_C2).
// @wg-kernel morig/l_post_recon co=l_post__morig.co entry=l_post_recon block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_C12=1,LP_MORIG=1
// @wg-kernel morig/l_post_recon_p12 co=l_post__morig.co entry=l_post_recon_p12 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p defines=LP_P12=1,LP_C12=1,LP_MORIG=1
#include "wg_platform.hlsli"
#ifndef LP_P12
#define LP_P12 0
#endif
#ifndef LP_C12
#define LP_C12 0
#endif
#ifndef LP_LDR
#define LP_LDR 0
#endif
#ifndef LP_MORIG
#define LP_MORIG 0
#endif
#ifndef LP_W64
#define LP_W64 0   // 1: the wave64 form (see the header)
#endif
#ifndef LP_CODE
#define LP_CODE 0   // 1: the FP16 code table from f32(k) f32(1 / 255) (exact for the 256 codes); 0: the IEEE division
#endif
#ifndef LP_RPUNI
#define LP_RPUNI 0   // 1: r_prev's division from the first lane's operand
#endif
#ifndef LP_QX
#define LP_QX 0   // 1: the 2 x 2 block's neighbour lane by QuadReadAcrossX
#endif
#if LP_W64 != (WG_WAVE == 64)
#error "l_post.hlsl: LP_W64 and the wave size (wave=) must agree"
#endif

WG_ARGS_BEGIN
    uint color, A, B, C, hist_prev, m_near, m_center, m_far, w_p, hist_out, reduced, display;   // pointers
    int P_rw, P_rh, P_dw, P_dh, P_reset, P_pad0, P_pad1, P_pad2;                                  // WgPostParams
    float P_rx, P_ry, P_jx, P_jy, P_e, P_e_prev, P_ratio, P_pad3;
    uint expo;
WG_ARGS_END

#define kThreads 256
#define kTileX 32
#define kTilePairs 8
#include "l_post_core.hlsli"

#if LP_P12 && LP_LDR
WG_KERNEL(l_post_recon_p12_ldr)
#elif LP_LDR
WG_KERNEL(l_post_recon_ldr)
#elif LP_P12
WG_KERNEL(l_post_recon_p12)
#else
WG_KERNEL(l_post_recon)
#endif
{
    WG_SETUP();
    g_z = wg_grid_ >> 31;
#if LP_LDR
    g_e = 1.0f;   // LDR: no exposure (the exposure pair is not read)
    g_ep = 1.0f;
    g_rp = 1.0f;
#else
    g_e = P_e;
    g_ep = P_e_prev;
    if (expo != 0u) {
        const float e0 = wg_ldf(expo, 0u), e1 = wg_ldf(expo, 4u);
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_ep = e1 == 0.0f ? 1.0f : e1;
    }
#if LP_RPUNI
    // g_ep is wave-uniform (a root constant or one loaded word): every lane's quotient is the first lane's
    g_rp = asfloat(WaveReadLaneFirst(asuint(wg_div(1.0f, g_ep))));
#else
    g_rp = wg_div(1.0f, g_ep);
#endif
#endif
    const int lane = (int)wg_lane_id();
    const int tiles_x = (P_dw + kTileX - 1) / kTileX;
    const int tile = (int)wg_block_id();
    const int x0 = (tile % tiles_x) * kTileX;
    const int k0 = (tile / tiles_x) * kTilePairs;
    const int npairs = (P_dh + 1) / 2;
    const float s = wg_ldf(w_p, 0u);
    code16_build(lane);
    GroupMemoryBarrierWithGroupSync();

    // render footprint of the tile: window columns ax - 1 ... ax + 2 of its first and last column, window rows of its
    // first and last row pair
    const int xl = mini(x0 + kTileX - 1, P_dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
    const int c0 = floor_i(lpx(x0) - 0.5f) - 1, c1 = floor_i(lpx(xl) - 0.5f) + (kLWin - 2);
    const int r0 = floor_i(lpy(2 * k0) - 0.5f) - 1, r1 = floor_i(lpy(2 * kl) - 0.5f) + (kLWin - 2);
    const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
    const bool staged = ncol > 0 && nrow > 0 && ncol <= kStageCap && nrow <= kStageCap && ncol * nrow <= kStageCap;
    if (staged) {   // wave-uniform rounds (the driver's conversion in load_texel_hw); lanes past the end repeat the last
        const int ntex = ncol * nrow;
        const int wl = wg_uniform(lane & ~(kTexWave - 1));   // this wave's first lane (uniform)
        [loop] for (int base = 0; base < ntex; base += kThreads) {
          [branch] if (base + wl < ntex) {   // a wave wholly past the end has nothing to store
            const int i = base + lane, ii = mini(i, ntex - 1);
            const int r = ii / ncol, c = ii - r * ncol;
            const int idx = clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1);
            uint t[kTexDw];
            load_texel_hw(idx, t);
            if (i < ntex) {
                [unroll] for (int d = 0; d < kTexDw; ++d) sh_tex[i * kTexDw + d] = t[d];
            }
          }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const bool active = and(x0 + (lane & 31) < P_dw, k0 + (lane >> 5) < npairs);
    const int x = mini(x0 + (lane & 31), P_dw - 1);
    const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
    const bool has1 = y0 + 1 < P_dh;
    const float px = lpx(x), py0 = lpy(y0), py1 = lpy(y0 + 1);
    const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

    // gather offsets: dx = h((ax + 1/2) - px), d_y = h((b_y + 1/2) - py(y)) minus 1 (FP16) when the row's own b_y differs
    // from the pair's b; u = h(dx + i), v = h(d_y + j); taps i = -1 ... 2 (outer), j = -1 ... 2 (inner)
    const int b1 = floor_i(py1 - 0.5f);
    v8f ov;   // the three offsets in one conversion
    ov.v[0] = opq(((float)ax + 0.5f) - px);
    ov.v[1] = opq(((float)b + 0.5f) - py0);
    ov.v[2] = opq(((float)b1 + 0.5f) - py1);
    [unroll] for (int oi = 3; oi < 8; ++oi) ov.v[oi] = 0.0f;
    const uint4 oh16 = wg_to_f16_ags(ov);
    const half dx = h_bits(oh16.x), dy0 = h_bits(oh16.x >> 16);
    half dy1 = h_bits(oh16.y);
    if (b1 != b) dy1 = dy1 - (half)1.0f;
    half v0[kLWin], v1[kLWin], vv0[kLWin], vv1[kLWin];
    [unroll] for (int jj = 0; jj < kLWin; ++jj) {
        v0[jj] = dy0 + (half)(float)(jj - 1);
        v1[jj] = dy1 + (half)(float)(jj - 1);
        vv0[jj] = mulh(v0[jj], v0[jj]);
        vv1[jj] = mulh(v1[jj], v1[jj]);
    }
    // taps column by column (i outer, j inner, each gather in that order as before); per column the eight weights (four
    // rows, two pixels) are rounded to FP16 by one driver conversion
    LGather g0 = gather_zero(), g1 = gather_zero();
    // the staged texel's LDS index without per-tap multiplies: one base per lane, the row stride and the column
    // step wave-uniform (integer identity: ((b + j - r0) ncol + (ax + i - c0)) kTexDw)
    const int tb0 = ((b - 1 - r0) * ncol + (ax - 1 - c0)) * kTexDw, rstride = ncol * kTexDw;
    // the hot loop reads the staged texels; the unstaged fallback (a tile whose render footprint does not fit) is its own loop
    [branch] if (staged) {
        [loop] for (int i = -1; i <= (kLWin - 2); ++i)
            gather_col<true>(i, dx, tb0, rstride, b, ax, v0, v1, vv0, vv1, g0, g1);
    } else {
        [loop] for (int i = -1; i <= kLWin - 2; ++i)
            gather_col<false>(i, dx, tb0, rstride, b, ax, v0, v1, vv0, vv1, g0, g1);
    }

    // the two rows of the pair (wave-uniform: every lane runs both; a missing second row is not stored and adds +0 as
    // before); the full history and display values of a row in one driver conversion
    float r[3] = {0.0f, 0.0f, 0.0f};
    const int pitch = (P_dw + 63) & ~63, plane = pitch * P_dh;
    [loop] for (int row = 0; row < 2; ++row) {
        const bool second = row == 1;
        LGather g = g0;
        if (second) g = g1;
        float hh[3], oo[3];
        const MotRaw mot = motion_load(px, second ? py1 : py0);
        finish_pixel(g, x, y0 + row, mot, s, hh, oo);
        v8f cv;
#if !LP_LDR && !LP_MORIG
        float el[3];
#endif
        [unroll] for (int ci = 0; ci < 3; ++ci) {
#if LP_LDR
            // h(H^); the display code floor(clamp(fused(O^ N + 1/2), 0, N)), then h(code RN32(1 / N)) (one rounding or
            // two give the same FP16 value for every code 0 ... N)
            const float n = ci == 2 ? 1023.0f : 2047.0f;
            const float k = asfloat(ci == 2 ? 0x3a802008u : 0x3a001002u);
            cv.v[ci] = opq(hh[ci]);
            cv.v[3 + ci] = opq(floor(clampf(mad(oo[ci], n, 0.5f), 0.0f, n)) * k);
#elif LP_MORIG
            // Spec 06 draft 6: full-history store rounds FP32 -> FP16 toward zero (RTZ; f32tof16 -> v_cvt_pkrtz_f16_f32,
            // matching HIP WG_LP_M wg_h2). Display channels stay RNE via wg_to_f16_ags.
            cv.v[ci] = 0.0f;
            cv.v[3 + ci] = opq(clampf(enc_e(oo[ci]), 0.0f, kDisplayMax));
#else
            el[ci] = enc_log(hh[ci]);
            cv.v[ci] = mul_odd(kEncK, el[ci]);
            cv.v[3 + ci] = opq(clampf(enc_e(oo[ci]), 0.0f, kDisplayMax));
#endif
        }
        cv.v[6] = 0.0f;
        cv.v[7] = 0.0f;
        const uint4 hc = wg_to_f16_ags(cv);
        const int y = y0 + row;
        if (and(active, or(!second, has1))) {
#if LP_LDR
            const uint q0 = hc.x & 0xffffu, q1 = hc.x >> 16, q2 = hc.y & 0xffffu;
            wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(q0 | (q1 << 16), q2));
#elif LP_MORIG
            const uint q01 = wg_h2(opq(enc_e(hh[0])), opq(enc_e(hh[1])));
            const uint q2 = f32tof16(opq(enc_e(hh[2])));
            wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(q01, q2));
#else
            const uint q0 = mul_zero_k(el[0]) ? 0u : (hc.x & 0xffffu);
            const uint q1 = mul_zero_k(el[1]) ? 0u : (hc.x >> 16);
            const uint q2 = mul_zero_k(el[2]) ? 0u : (hc.y & 0xffffu);
            wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(q0 | (q1 << 16), q2));
#endif
            const int di = y * pitch + x;
            wg_st16(display, (uint)di * 2u, hc.y >> 16);
            wg_st16(display, (uint)(plane + di) * 2u, hc.z & 0xffffu);
            wg_st16(display, (uint)(2 * plane + di) * 2u, hc.z >> 16);
        }
        // reduced history: mean of H^ over the 2 x 2 block (x, x + 1) x (y0, y0 + 1): r = h0 + h1 (h1 = 0 without a
        // second row)
        [unroll] for (int ri = 0; ri < 3; ++ri) r[ri] = second ? r[ri] + (has1 ? hh[ri] : 0.0f) : hh[ri];
    }
    // lane x + 1 is lane ^ 1 (same wave)
    float o[3];
#if LP_QX
    [unroll] for (int i = 0; i < 3; ++i) o[i] = QuadReadAcrossX(r[i]);   // lane ^ 1
#else
    [unroll] for (int i = 0; i < 3; ++i) o[i] = wg_xor(r[i], 1u);
#endif
    if (active && has1 && (x & 1) == 0 && x + 1 < P_dw) {
        uint code = 0u;
        [unroll] for (int i = 0; i < 3; ++i) {
            const float m = opq((r[i] + o[i]) * 0.25f);
#if LP_LDR
            const float v = clampf(m, 0.0f, 1.0f);
#else
            const float v = clampf(opq(enc_e(m)) * kInv245, 0.0f, 1.0f);
#endif
            code |= wg_f2u(round(v * 1023.0f)) << (10 * i);
        }
        wg_st32(reduced, (uint)((y0 / 2) * (P_dw / 2) + x / 2) * 4u, code);
    }
}
