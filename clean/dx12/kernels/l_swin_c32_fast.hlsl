// SPDX-License-Identifier: GPL-3.0-only
// Model L's fast-mode width-32 Swin blocks with TWO windows per wave (runtime l_mode=fast, native D3D12 only; launched in
// place of a base kernel with the same arguments, runtime l_fast_alt; the base kernels, their "_fast" twins and every
// kernel l_mode=hybrid runs are not touched). The launch shape of swin_c64w2.hlsl: the launch keeps HIP's geometry
// (8 windows per launched 256-lane workgroup, grid = ceil(windows / 8)), the D3D12 device runs each launched workgroup as
// one workgroup of 128 lanes (manifest "group" 128, "gscale" 1), and wave v does windows 8 g + 2 v and 8 g + 2 v + 1:
// every staged weight fragment is read from LDS once for both windows, every bias / norm / linear weight is loaded once
// for both, a unit barrier waits for 4 waves instead of 8, and the two windows are independent chains for the matrix
// unit and the VALU. LDS: the same 16 KB staging (2 x 8 KB units: 4 workgroups per CU = 32 windows per CU, as the
// one-window kernels at 96 VGPRs), which needs <= 192 VGPRs (wave32, 128 lanes = 2 waves per SIMD per workgroup).
//
// The arithmetic is swin_block.hlsl's M-form chain (WMMA FP8 / FP16, the same packs, norms, soft cap and order of the
// products) except where fast mode drops the reference's bit-exactness:
//   - GELU (LF_GELU): 1 = the clamped cubic GELU(x) = (x + |x| sat(P(|x|))) / 2, P(t) = c0 + c1 t + c2 t^2 + c3 t^3
//     fitted (weighted minimax) to the model's GELU E(t) = 2 GELU(t) / t - 1, evaluated in packed FP16 (per pair of
//     values: one v_cvt_pk_rtz_f16_f32 of |x|, three v_pk_fma_f16, the last one saturated; then per value one
//     v_fma_mix_f32 2 GELU = |x| sat(P) + x in FP32): three full-rate VALU operations per value, no exponential, no
//     reciprocal, no table load. The halving is not an operation: the W2 products accumulate onto 2 x the residual
//     (the accumulator doubled with b2, halved at the output: exact power-of-two scalings; FP8(2 g) = 2 FP8(g) for
//     every normal FP8 value). |error| <= 0.0031, RMS 0.0014 for N(0, 1.5) inputs against the model's GELU (FP32
//     evaluation of the same cubic: 0.0020), 85 % of the FP8 codes unchanged (test_l_fast_numerics.py PolyGelu).
//     P >= 1 from t = 3.3 on and never falls back below 1, so GELU(x) = x for x >= 3.3 and 0 for x <= -3.3.
//     RX 9070 XT (2B-3): the earlier quick GELU x / (1 + 2^(-2.53 x)) (an exponential and a reciprocal per value: the
//     quarter-rate transcendental unit) was 44 / 56 us slower than the table form, hence the polynomial.
//     2 = the model's GELU bytes by the arena table (swin_block.hlsl's sw_gelu_pre / sw_gelu_fin with SW_GELU_PAT 13333,
//     the base kernels' form: no exponential, one 8-byte table load per value; both windows' loads issued before either
//     window's fmas); 0 = HIP's transcendental chain (the same bytes, two exponential-unit operations per value; not
//     declared: the table form is the exact A/B of the polynomial).
//   (the "_w2t_fast" kernels are the "_w2p_fast" ones with LF_GELU 2: the A/B of the GELU approximation alone)
//   - softmax: the scores are clamped by the soft cap (|c| <= 32.37, t = c - c^3 / (3 cap^2) in [-21.6, 21.6], times
//     tau log2(e) = 0.51: exponents in [-11, 11]), so 2^z needs no maximum subtracted (no overflow: the sum of 16 terms
//     is below 2^15; no underflow): the max reduction, its swizzle and the negated product are gone. The weights are
//     the same values up to the rounding of the exponentials.
//   - the decoder block's coefficient epilogue (SW_OUT_COEF12): one pixel per LANE instead of per lane pair: the
//     channels 0..12 of a token sit in its hh = 0 lane for both windows, so the hh = 1 lane takes the second window's
//     (13 v_permlanex16 + 13 selects) and every lane evaluates Model L's coefficient functions once; in FP32 from the
//     block's FP32 output (no FP16 rounding of y2, no RN16 of the sigmas, tanh as 1 - 2 / (2^(2 t log2 e) + 1) for
//     every t: no polynomial branch), the FP16 products by the native conversion with the scale folded in.
//
// @wg-kernel l_swin_block_c32_enc_lin_w2p_fast co=- block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=4,SW_SRC=3,WG_LD_A16=1,WG_REV_ORDER=1,LF_NAME=l_swin_block_c32_enc_lin_w2p_fast,LF_GELU=1
// @wg-kernel l_swin_block_c32_enc_lin_w2t_fast co=- block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=4,SW_SRC=3,WG_LD_A16=1,WG_REV_ORDER=1,LF_NAME=l_swin_block_c32_enc_lin_w2t_fast,LF_GELU=2,SW_GELU_PAT=13333
// @wg-kernel l_swin_block_c32_dec_coef12_add_w2p_fast co=- block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=4,SW_SRC=2,SW_OUT_F16=1,SW_OUT_COEF12=1,WG_LD_A16=1,LF_NAME=l_swin_block_c32_dec_coef12_add_w2p_fast,LF_GELU=1
// @wg-kernel l_swin_block_c32_dec_coef12_add_w2t_fast co=- block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=4,SW_SRC=2,SW_OUT_F16=1,SW_OUT_COEF12=1,WG_LD_A16=1,LF_NAME=l_swin_block_c32_dec_coef12_add_w2t_fast,LF_GELU=2,SW_GELU_PAT=13333
//
// Arguments as the kernels they replace: l_swin_block_c32_enc_lin_nc (x = the FP8 q halo map in its 20-byte form, xadd and
// out_p unused / the output map, w, H, W, ox, oy, enc0.Win1, enc0.Win2) and l_swin_block_c32_dec_coef12_add_nc (y, S, out
// unused, w, H, W, ox, oy, then the store's A, C, latent, dec0.coef, rw, rh). l_swin_c64_fast.hlsl instantiates the same
// body at C = 64 (Model L's middle blocks).
#ifndef LF_GELU
#define LF_GELU 1
#endif
#ifndef LF_NAME
#error "l_swin_c32_fast.hlsl: LF_NAME (the kernel's name)"
#endif
#define SW_ENTRY_OVERRIDE lf_sw_one_window_unused   // swin_block.hlsl's entry under another name (not compiled: not the entry)
#ifndef SW_GELU_PAT
#define SW_GELU_PAT 10000   // swin_block.hlsl's GELU knobs: no table of its own in this kernel (no groupshared table)
#endif
#include "swin_block.hlsl"

#if (THREADS != 128 && THREADS != 256) || SW_HYB || SW_MEGA || !SW_SAFE || (SW_OUT_F16 && !SW_OUT_COEF12) || (SW_SRC != 0 && SW_SRC != 2 && SW_SRC != 3) || (SW_ECL > 0 && SW_SRC != 0)
#error "l_swin_c32_fast.hlsl: 128 or 256 lanes, the M-form inner packs; output q8 or the coefficient store; source 0 (with or without the fused expand), 2 or 3"
#endif
#if THREADS == 256 && (SW_C != 96 || SW_OUT_COEF12)
#error "l_swin_c32_fast.hlsl: one window per wave (256 lanes) is the width-96 form (l_swin_c96_fast.hlsl)"
#endif
#if SW_OUT_COEF12 && (SW_C != 32 || !SW_ADD || SW_SRC != 2)
#error "l_swin_c32_fast.hlsl: the coefficient store is Model L's decoder block (C = 32, the skip, the coarse source)"
#endif
#ifndef LF_PM
#define LF_PM 0
#endif
#if LF_PM && (SW_C != 96 || SW_SRC != 0 || SW_ADD || SW_ECL > 0 || !defined(LF_PM_ARGS))
#error "LF_PM: enc2's block with enc1's patch merge (C = 96, source 0; aux and aux2 appended to the arguments: l_swin_c96_fast.hlsl)"
#endif
#define LF_LIN (SW_SRC == 3)   // the encoder block's input linears (Model L's enc0)
// LF_W2_ONE (the coefficient store at C = 32): the store reads channels 0..12 only, so W2 runs one output tile, tile 0
// remapped to channels 0..15 in order (rows 8..15: tile 1's rows 0..7, the residual from lane t, the weight fragment from
// lane l - 8): half the W2 WMMAs, the same products and the same accumulation order for every channel stored
#define LF_W2_ONE (SW_OUT_COEF12 && SW_C == 32 && KC == 2)
#if LF_PM
#define LF_PM_W (16 * 6 * 256)   // the 24 KB of merge fragments: both staging buffers
#if LF_PM_W > 2 * MAXU
#error "LF_PM: the merge fragments must fit the staging buffers"
#endif
// fused_swin.hlsl fz_stage_w: the 24 KB of fragments at src into LDS byte 0 (6 16-byte chunks per lane), then a barrier
void lf_stage_w(uint src, uint tid)
{
    uint4 v[6];
    [unroll] for (uint i = 0; i < 6; ++i) v[i] = wg_ld128(src, (tid + i * 256u) * 16u);
    [unroll] for (uint i2 = 0; i2 < 6; ++i2) sw_lds_st128((tid + i2 * 256u) * 16u, v[i2]);
    wg_barrier_lds();
}
#endif
#if THREADS == 128
#define NWW 2   // windows per wave: 128-lane workgroups (the launch shape of swin_c64w2.hlsl)
#else
#define NWW 1   // 256-lane workgroups, one window per wave (the launch shape of the width-96 base kernels)
#endif

// ---- GELU: q8 of one 16-row tile (8 values per lane) ----------------------------------------------------------------
// P(t) of the clamped cubic (test_l_fast_numerics.py PolyGelu.C: the same float32 values)
#if LF_GELU == 1
static const float kLfP0 = asfloat(0xbd6c7ea1u), kLfP1 = asfloat(0x3f878bedu), kLfP2 = asfloat(0xbeb6e5c7u),
                   kLfP3 = asfloat(0x3d25ab15u);   // -0.05773795, 1.0589577, -0.35722181, 0.040446360
#define LF_G2 1   // the MLP's W2 products accumulate onto 2 x the residual (q8 of 2 GELU)
#else
#define LF_G2 0
#endif
#if LF_GELU == 2 && (SW_GELU != 3 || SW_GELU_PAT != 13333 || SW_GL_LDS)
#error "LF_GELU 2: swin_block.hlsl's arena table for every tile (SW_GELU 3, SW_GELU_PAT 13333)"
#endif
uint2 lf_gelu_q8(v8f h)
{
    v8f y;
#if LF_GELU == 1
    // q8(2 GELU(h)) (LF_G2) in operations the driver can pair (VOPD): |h| by an AND with an opaque mask (an abs source
    // modifier would make every operation VOP3), the Horner steps as v_fmaak (one shared literal per step), the clamp as
    // v_min (P < 0 only below t = 0.055, where GELU's error stays under 0.0013 without the lower clamp), 2 GELU by v_fmac.
    // Written stage by stage: the 8 independent chains interleave.
    const uint am = 0x7fffffffu ^ (wg_grid_ >> 31);
    float t[8], p[8];
    [unroll] for (uint i = 0; i < 8; ++i) t[i] = asfloat(asuint(h.v[i]) & am);
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) p[i1] = mad(kLfP3, t[i1], kLfP2);
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) p[i2] = mad(p[i2], t[i2], kLfP1);
    [unroll] for (uint i3 = 0; i3 < 8; ++i3) p[i3] = min(mad(p[i3], t[i3], kLfP0), 1.0f);
    [unroll] for (uint i4 = 0; i4 < 8; ++i4) y.v[i4] = mad(t[i4], p[i4], h.v[i4]);
#else
    float t[8];   // HIP's chain (swin_block.hlsl gelu_group / sw_gelu_fin digit 0)
    [unroll] for (uint i = 0; i < 8; ++i) t[i] = h.v[i] * h.v[i];
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) t[i1] = mad(t[i1], asfloat(0xbdcd6502u), asfloat(0xc013b8edu));
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) t[i2] = h.v[i2] * t[i2];
    [unroll] for (uint i3 = 0; i3 < 8; ++i3) t[i3] = wg_exp2(t[i3]);
    [unroll] for (uint i4 = 0; i4 < 8; ++i4) t[i4] = 1.0f + t[i4];
    [unroll] for (uint i5 = 0; i5 < 8; ++i5) t[i5] = wg_rcp(t[i5]);
    [unroll] for (uint i6 = 0; i6 < 8; ++i6) y.v[i6] = h.v[i6] * t[i6];
#endif
    return wg_pack_q8_nc_ags(y);
}

// ---- softmax of one head's scores (lane (t, h): S(t, s) of 8 keys; the other 8 in lane t ^ 16) ----------------------
// c = clamp(S + B, -cap, cap); t = fma(c^3, -1 / (3 cap^2), c); p = FP16(2^(t tau log2 e) / sum) (no max: see the header)
uint4 lf_softmax(v8f s, v8f bias)
{
    v8f c, tc, e;
    [unroll] for (uint va0 = 0; va0 < 8; ++va0) c.v[va0] = s.v[va0] + bias.v[va0];
    [unroll] for (uint v = 0; v < 8; ++v) c.v[v] = min(max(c.v[v], asfloat(0xc20179dbu)), asfloat(0x420179dbu));
    [unroll] for (uint v1 = 0; v1 < 8; ++v1) tc.v[v1] = c.v[v1] * c.v[v1];
    [unroll] for (uint v2 = 0; v2 < 8; ++v2) tc.v[v2] = mad(c.v[v2] * tc.v[v2], asfloat(0xb9a6cc38u), c.v[v2]);
    [unroll] for (uint v4 = 0; v4 < 8; ++v4) e.v[v4] = wg_exp2(tc.v[v4] * asfloat(0x3f028297u));
    float sum = sw_hsum8(e);
    sum = sum + wg_xor16(sum);
    const float rs = wg_rcp(sum);
    v8f pm;
    [unroll] for (uint v6 = 0; v6 < 8; ++v6) pm.v[v6] = mad(e.v[v6], rs, 0.0f);   // the driver's v_fma_mix (one rounding)
    return wg_to_f16_ags(pm);
}

#if LF_LIN
// sw_load_halo_lin's TT_LINF32 form (swin_block.hlsl: the two input linears in FP32, x1 = FP8(relu(b + W q)),
// x2 = b2 + W2 x1) with the linears' weights loaded once per wave for both windows (lin_blob outside)
void lf_halo_lin(uint base, int pr, int pc, uint hh, v8f b1[2], uint2 f1[2][2], v8f b2[2], uint2 f2[2][2], out v8f f[KC])
{
    const uint pix = (uint)((pr + 2) * (W + 4) + pc + 2);
    uint2 xq[2];
    const uint plane1 = (uint)((H + 4) * (W + 4)) * 16u;
    if (hh != 0u) {
        xq[0] = uint2(wg_ld32(base, plane1 + pix * 4u), 0u);
        xq[1] = uint2(0u, 0u);
    } else {
        const uint4 v = wg_ld128(base, pix * 16u);
        xq[0] = v.xy;
        xq[1] = v.zw;
    }
    uint2 x1[2];
    [unroll] for (uint m1 = 0; m1 < 2u; ++m1) {
        v8f r = lin_acc(b1[m1], f1[m1], xq);
        [unroll] for (uint j = 0; j < 8u; ++j) r.v[j] = wg_fmax(r.v[j], 0.0f);
        x1[m1] = wg_pack_q8_ags(r);
    }
    [unroll] for (uint k8 = 0; k8 < KC; ++k8) f[k8] = lin_acc(b2[k8], f2[k8], x1);
}
#endif

#if SW_OUT_COEF12
// Model L's coefficient functions of one pixel (l_coef_px.hlsli's values in FP32; see the header)
float lf_sigma(float x) { return wg_rcp(1.0f + wg_exp2(x * asfloat(0xBFB8AA3Bu))); }          // 1 / (1 + 2^(-x log2 e))
float lf_tanh(float x) { return mad(wg_rcp(wg_exp2(x * asfloat(0x4038AA3Bu)) + 1.0f), -2.0f, 1.0f); }   // 1 - 2 / (e^2x + 1)
uint lf_unorm8(float v) { return wg_f2u(mad(v, 255.0f, 0.5f)); }                              // v in [0, 1]
uint lf_snorm8(float v) { return (uint)wg_f2i(wg_rint(v * 127.0f)) & 0xffu; }                  // v in [-1, 1]
void lf_coef_px(float a[13], float S2, float T, uint pA, uint pC, uint pL, uint i)
{
    const uint cA = lf_unorm8(lf_sigma(a[0])) | (lf_unorm8(lf_sigma(a[1])) << 8) | (lf_unorm8(lf_sigma(a[2])) << 16);
    const float m = max(max(a[3], a[4]), a[5]);
    const float e0 = wg_exp2((a[3] - m) * asfloat(0x3FB8AA3Bu)), e1 = wg_exp2((a[4] - m) * asfloat(0x3FB8AA3Bu)),
                e2 = wg_exp2((a[5] - m) * asfloat(0x3FB8AA3Bu));
    const float rs = wg_rcp(e2 + (e0 + e1));
    const uint cB = lf_unorm8(e0 * rs) | (lf_unorm8(e1 * rs) << 8) | (lf_unorm8(e2 * rs) << 16);
    // C: (S g0)^2, (S g1)^2, (S g0)(S g1) tanh(a8 T), the S^2 and the round-to-nearest bias of the truncating pack folded
    // into S2 (= S^2 (1 + 2^-11.7))
    const float g0 = lf_sigma(clamp(a[6], -20.0f, 20.0f)), g1 = lf_sigma(clamp(a[7], -20.0f, 20.0f));
    const float rho = lf_tanh(a[8] * T);
    const float sg0 = S2 * g0;
    const uint c0 = f32tof16(sg0 * g0) | (f32tof16((S2 * g1) * g1) << 16);
    const uint c1 = f32tof16((sg0 * g1) * rho);
    const uint cL = lf_snorm8(lf_tanh(a[9])) | (lf_snorm8(lf_tanh(a[10])) << 8) | (lf_snorm8(lf_tanh(a[11])) << 16) |
                    (lf_snorm8(lf_tanh(a[12])) << 24);
    wg_st64(pA, i * 8u, uint2(cA | (cB << 24), (cB >> 8) | (c1 << 16)));
    wg_st32(pC, i * 4u, c0);
    wg_st32(pL, i * 4u, cL);
}
#endif

WG_KERNEL(LF_NAME)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
    const int nwx = (W + ox + 3) >> 2, nwy = (H + oy + 3) >> 2, nwin = nwx * nwy;
    const uint wave_id = (uint)wg_uniform((int)(tid >> 5));

    Stager stage;
    st_fetch(stage, w + L_head(0), HEAD_BYTES / 16, tid);   // unit 0

    // Gather: token t of each window, mirrored at the borders.
    int dst[NWW];
#if SW_OUT_COEF12
    int prw[NWW], pcw[NWW];
#endif
    v8f acc[NWW][KC];
#if LF_LIN
    v8f lb1[2], lb2[2];
    uint2 lf1[2][2], lf2[2][2];
    lin_blob(linW1, lane, hh, lb1, lf1);
    lin_blob(linW2, lane, hh, lb2, lf2);
#else
    uint2 xa[NWW][KC];
#endif
#if SW_ADD
    uint2 xb[NWW][KC];
#endif
#if SW_ECL > 0
    const uint lowp = sw_win_low0(nwx, nwin, lane, (int)wg_block_id());
    fused_expand(w + L_TAIL, MAXU, nwx, nwin, tid, (int)wg_block_id(), lowp);
    uint eslot[NWW];
#endif
    [unroll] for (uint i = 0; i < NWW; ++i) {
        const int win0 = (int)(wg_block_id() * (THREADS / 32 * NWW) + wave_id * NWW + i);
        const int win = win0 < nwin ? win0 : nwin - 1;
        const int wy = win / nwx, wx = win - wy * nwx;
        const int pr = 4 * wy - oy + (int)(t >> 2), pc = 4 * wx - ox + (int)(t & 3u);
        const bool real = win0 < nwin && pr >= 0 && pr < H && pc >= 0 && pc < W;
        dst[i] = real ? pr * W + pc : -1;
#if SW_OUT_COEF12
        prw[i] = pr; pcw[i] = pc;
#endif
        const int mr = sw_mirror(pr, H), mc = sw_mirror(pc, W);
#if SW_ECL > 0
        const uint wb = wave_id * NWW + i;
        const uint lpw = WaveReadLaneAt(lowp, wb);
        const int low = ((mr >> 1) - (int)(lpw & 0xffffu)) * 2 + ((mc >> 1) - (int)(lpw >> 16));
        eslot[i] = wb * 16u + (uint)low * 4u + (uint)(((mr & 1) << 1) | (mc & 1));
#elif LF_LIN
        lf_halo_lin(x, pr, pc, hh, lb1, lf1, lb2, lf2, acc[i]);
#elif LF_PM
        {   // fused_swin.hlsl SW_PM (FZ_PMT 1): enc1's patch merge of this lane's (mirrored) token, the bias as the first
            // accumulator, the 16 K groups in order, the clamping FP8 pack (patch_merge_c64's bytes)
            uint2 zf[16];
            const uint zo = (uint)((2 * mr + (int)hh) * (2 * W) + 2 * mc) * 64u;
            [unroll] for (uint q = 0; q < 8; ++q) {
                const uint4 v = wg_ld128(x, zo + 16u * q);
                zf[2 * q] = v.xy; zf[2 * q + 1] = v.zw;
            }
            lf_stage_w(aux + 4u * 96u, tid);
            const uint bm = aux + hh * 32u, fbm = sw_lane_base(0u, lane);
            [unroll] for (uint mp = 0; mp < KC; ++mp) {
                uint2 fa[16];
                [unroll] for (uint g = 0; g < 16; ++g) fa[g] = sw_lfrag_b(fbm, mp * 16u + g);
                v8f m = wg_load8f(bm, mp * 64u);
                [unroll] for (uint g2 = 0; g2 < 16; ++g2) m = wg_wmma_fp8(fa[g2], zf[g2], m);
                xa[i][mp] = sw_pack_q8_v2(m);
            }
            wg_barrier_lds();   // every wave is done with the merge fragments before unit 0 overwrites them
        }
#elif SW_SRC == 2
        load_frags_g(x, sw_coarse_src(pr, pc, H, W, hh), xa[i]);
#else
        load_frags_g(x, (uint)(mr * W + mc) * SW_C + hh * HALF, xa[i]);
#endif
#if SW_ADD
        load_frags_g(xadd, (uint)(mr * W + mc) * SW_C + hh * HALF, xb[i]);
#endif
    }
    v8f ng[KC], nb[KC];
    {
        const uint g1 = w + L_G1 + hh * (HALF * 4), bp = w + L_BP + hh * 32u;
        [unroll] for (uint k = 0; k < KC; ++k) { ng[k] = wg_load8f(g1, 32u * k); nb[k] = wg_load8f(bp, 64u * k); }
    }
    st_commit(stage, 0u, HEAD_BYTES / 16, tid);
    wg_barrier_lds();
#if SW_ECL > 0
    [unroll] for (uint ie = 0; ie < NWW; ++ie)
        [unroll] for (uint je = 0; je < KC / 2; ++je) {
            const uint4 v = sw_lds_ld128(MAXU + eslot[ie] * SW_C + hh * HALF + 16u * je);
            xa[ie][2 * je] = v.xy; xa[ie][2 * je + 1] = v.zw;
        }
    wg_barrier_lds();   // read before the buffer they live in is refilled
#endif
    uint buf = 0u;

    // Steps 1-2: a = q8(RMSNorm(x; gamma1)); the residual accumulator starts as x + bp.
    uint2 af[NWW][KC];
    [unroll] for (uint i1 = 0; i1 < NWW; ++i1) {
#if !LF_LIN
        [unroll] for (uint k = 0; k < KC; ++k) {
            acc[i1][k] = sw_unpack_fp8(xa[i1][k]);
#if SW_ADD
            acc[i1][k] = sw_add8(acc[i1][k], sw_unpack_fp8(xb[i1][k]));
#endif
        }
#endif
        const float r = norm_r(acc[i1]);
        [unroll] for (uint k2 = 0; k2 < KC; ++k2) {
            v8f d;
            [unroll] for (uint v = 0; v < 8; ++v) d.v[v] = acc[i1][k2].v[v] * r * ng[k2].v[v];
            af[i1][k2] = pack_in(d);
            acc[i1][k2] = sw_add8(acc[i1][k2], nb[k2]);
        }
    }

    // Steps 3-7, one head at a time (unrolled); one staged unit per head (QKV fragments, then the head's slice of Wp).
    [unroll] for (uint hd = 0; hd < SW_HEADS; ++hd) {
        const bool last = hd + 1u == SW_HEADS;
        if (last) st_fetch(stage, w + L_mlp(0), MLP_BYTES / 16, tid);
        else st_fetch(stage, w + L_head(hd + 1u), HEAD_BYTES / 16, tid);
        const uint fb = sw_lane_base(buf * MAXU, lane);
        const v8f bias = wg_load8f(w + L_BIAS, ((hd * 16u + t) * 2u + hh) * 32u);
        uint2 uf[NWW][2];
        uint4 vt[NWW][2];
        [unroll] for (uint mt = 0; mt < 2; ++mt) {
            uint2 fq[KC], fv[KC];
            [unroll] for (uint kc = 0; kc < KC; ++kc) {
                fq[kc] = sw_lfrag_b(fb, (mt * KC + kc) * 2u);
                fv[kc] = sw_lfrag_b(fb, (mt * KC + kc) * 2u + 1u);
            }
            v8f accq[NWW], accv[NWW];
            [unroll] for (uint i = 0; i < NWW; ++i) { accq[i] = sw_zero8(); accv[i] = sw_zero8(); }
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (uint i2 = 0; i2 < NWW; ++i2) {
                    accq[i2] = wg_wmma_fp8(fq[kc2], af[i2][kc2], accq[i2]);
                    accv[i2] = wg_wmma_fp8(af[i2][kc2], fv[kc2], accv[i2]);
                }
            [unroll] for (uint i3 = 0; i3 < NWW; ++i3) {
                uf[i3][mt] = pack_in(accq[i3]);
                vt[i3][mt] = wg_to_f16_ags(accv[i3]);
            }
        }
        uint2 of[NWW][2];
        [unroll] for (uint i4 = 0; i4 < NWW; ++i4) {
            v8f s = wg_wmma_fp8(uf[i4][0], uf[i4][0], sw_zero8());   // S = U^T U
            s = wg_wmma_fp8(uf[i4][1], uf[i4][1], s);
            const uint4 p = lf_softmax(s, bias);
            [unroll] for (uint dt = 0; dt < 2; ++dt) of[i4][dt] = pack_in(wg_wmma_f16(vt[i4][dt], p, sw_zero8()));
        }
        // x1 += Wp[:, head hd] q8(o_hd)
        uint2 fp[KC][2];
        [unroll] for (uint m = 0; m < KC; ++m) {
            fp[m][0] = sw_lfrag_b(fb, QKV_FRAGS + m * 2u);
            fp[m][1] = sw_lfrag_b(fb, QKV_FRAGS + m * 2u + 1u);
        }
        [unroll] for (uint m2 = 0; m2 < KC; ++m2)
            [unroll] for (uint i5 = 0; i5 < NWW; ++i5) {
                acc[i5][m2] = wg_wmma_fp8(fp[m2][0], of[i5][0], acc[i5][m2]);
                acc[i5][m2] = wg_wmma_fp8(fp[m2][1], of[i5][1], acc[i5][m2]);
            }
        if (last) st_commit(stage, (buf ^ 1u) * MAXU, MLP_BYTES / 16, tid);
        else st_commit(stage, (buf ^ 1u) * MAXU, HEAD_BYTES / 16, tid);
        wg_barrier_lds();
        buf ^= 1u;
    }

    // Step 8: m = q8(RMSNorm(x1; gamma2)), then b2 into the residual accumulator.
    uint2 mf[NWW][KC];
    {
        const uint g2 = w + L_G2 + hh * (HALF * 4), b2 = w + L_B2 + hh * 32u;
        float r[NWW];
        [unroll] for (uint i = 0; i < NWW; ++i) r[i] = norm_r(acc[i]);
        [unroll] for (uint k = 0; k < KC; ++k) {
            const v8f g = wg_load8f(g2, 32u * k), b = wg_load8f(b2, 64u * k);
#if LF_G2
            const v8f b2x = sw_add8(b, b);   // 2 (x1 + b2): once for both windows
#endif
            [unroll] for (uint i1 = 0; i1 < NWW; ++i1) {
                v8f d;
                [unroll] for (uint v = 0; v < 8; ++v) d.v[v] = acc[i1][k].v[v] * r[i1] * g.v[v];
                mf[i1][k] = pack_in(d);
#if LF_G2
                [unroll] for (uint v2 = 0; v2 < 8; ++v2) acc[i1][k].v[v2] = mad(acc[i1][k].v[v2], 2.0f, b2x.v[v2]);
#else
                acc[i1][k] = sw_add8(acc[i1][k], b);
#endif
            }
        }
    }

#if LF_W2_ONE
    [unroll] for (uint iw = 0; iw < NWW; ++iw)
        [unroll] for (uint v = 0; v < 8; ++v) {
            const float x = wg_xor16(acc[iw][1].v[v]);   // lane t's tile 1 row v: channel 8 + v
            acc[iw][0].v[v] = hh != 0u ? x : acc[iw][0].v[v];
        }
    const bool w2_hi = (lane & 8u) != 0u;   // fragment rows 8..15 of the remapped tile
#endif
    // Steps 9-10: x2 = x1 + W2 q8(GELU(W1 m + b1)) + b2, GROUP hidden tiles of 16 at a time (both windows per group: the
    // other window's chain is the independent work under each GELU).
    const uint b1 = w + L_B1 + hh * 32u;
    [unroll] for (uint u = 0; u < NMLP; ++u) {
        const bool more = u + 1u < NMLP;
        if (more) st_fetch(stage, w + L_mlp(u + 1u), MLP_BYTES / 16, tid);
        [unroll] for (uint j0 = 0; j0 < HPU; j0 += GROUP) {
            const uint fb = sw_lane_base(buf * MAXU, lane) + j0 * (2u * KC * 32u);
            v8f hcur[NWW][GROUP];
            {
                uint2 f1[GROUP][KC];
                [unroll] for (uint g = 0; g < GROUP; ++g)
                    [unroll] for (uint kc = 0; kc < KC; ++kc) f1[g][kc] = sw_lfrag_b(fb, g * 2u * KC + kc);
                v8f bh[GROUP];
                [unroll] for (uint g2 = 0; g2 < GROUP; ++g2) bh[g2] = wg_load8f(b1, (u * HPU + j0 + g2) * 64u);
                [unroll] for (uint i = 0; i < NWW; ++i) [unroll] for (uint g1 = 0; g1 < GROUP; ++g1) hcur[i][g1] = bh[g1];
                [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                    [unroll] for (uint i2 = 0; i2 < NWW; ++i2)
                        [unroll] for (uint g3 = 0; g3 < GROUP; ++g3) hcur[i2][g3] = wg_wmma_fp8(f1[g3][kc2], mf[i2][kc2], hcur[i2][g3]);
            }
            uint2 f2[GROUP][KC];
            [unroll] for (uint g4 = 0; g4 < GROUP; ++g4)
                [unroll] for (uint m = 0; m < KC; ++m) f2[g4][m] = sw_lfrag_b(fb, g4 * 2u * KC + KC + m);
            uint2 hf[NWW][GROUP];
#if LF_GELU == 2
            SwGeluPre gp[NWW];
            [unroll] for (uint ia = 0; ia < NWW; ++ia) gp[ia] = sw_gelu_pre(hcur[ia]);
            [unroll] for (uint ib = 0; ib < NWW; ++ib) sw_gelu_fin(hcur[ib], gp[ib], hf[ib]);
#else
            [unroll] for (uint ib = 0; ib < NWW; ++ib)
                [unroll] for (uint gb = 0; gb < GROUP; ++gb) hf[ib][gb] = lf_gelu_q8(hcur[ib][gb]);
#endif
#if LF_W2_ONE
            [unroll] for (uint g5 = 0; g5 < GROUP; ++g5) {
                const uint2 t1 = uint2(wg_xor_u(f2[g5][1].x, 8u), wg_xor_u(f2[g5][1].y, 8u));
                f2[g5][0] = w2_hi ? t1 : f2[g5][0];
            }
            [unroll] for (uint i6 = 0; i6 < NWW; ++i6)
                [unroll] for (uint g6 = 0; g6 < GROUP; ++g6) acc[i6][0] = wg_wmma_fp8(f2[g6][0], hf[i6][g6], acc[i6][0]);
#else
            [unroll] for (uint m2 = 0; m2 < KC; ++m2)
                [unroll] for (uint i6 = 0; i6 < NWW; ++i6)
                    [unroll] for (uint g6 = 0; g6 < GROUP; ++g6) acc[i6][m2] = wg_wmma_fp8(f2[g6][m2], hf[i6][g6], acc[i6][m2]);
#endif
            if (more && j0 == 0u) st_commit(stage, (buf ^ 1u) * MAXU, MLP_BYTES / 16, tid);   // the other buffer is free
        }
        if (more) {
            wg_barrier_lds();
            buf ^= 1u;
        }
    }

#if SW_OUT_COEF12
    // Model L's coefficient store, one pixel per lane: lane (t, 0) takes window 0's token t, lane (t, 1) window 1's (its
    // channels 0..12 from lane (t, 0): acc[1][0] = channels 0..7, acc[1][1] = 8..15 of that lane)
    {
        float a[13];
        const uint ch[13] = {0u, 1u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 2u, 3u, 4u, 5u};   // L_DEC_VALUES
        [unroll] for (uint q = 0; q < 13u; ++q) {
#if LF_W2_ONE
            // tile 0 = channels 0..7 (lane t) and 8..15 (lane t + 16): the window's own half, the other from lane t ^ 16
            const uint vq = ch[q] & 7u;
            const float own = hh != 0u ? acc[1][0].v[vq] : acc[0][0].v[vq];
            const float recv = wg_xor16(hh != 0u ? acc[0][0].v[vq] : acc[1][0].v[vq]);
            a[q] = ((ch[q] < 8u) == (hh == 0u)) ? own : recv;
#else
            const float own = acc[0][ch[q] >> 3].v[ch[q] & 7u];
            const float other = wg_xor16(acc[1][ch[q] >> 3].v[ch[q] & 7u]);
            a[q] = hh != 0u ? other : own;
#endif
#if LF_G2
            a[q] = a[q] * 0.5f;   // the doubled accumulator (exact)
#endif
        }
        const float S = wg_ldf(coefW, 320u), T = wg_ldf(coefW, 324u);
        const float S2 = S * S * 1.00034f;   // the truncating FP16 pack: x (1 + 2^-11.7) has the nearest rounding's mean error
        const int pr = hh != 0u ? prw[1] : prw[0], pc = hh != 0u ? pcw[1] : pcw[0];
        const bool real = (hh != 0u ? dst[1] : dst[0]) >= 0;
        if (real && pr < coefRh && pc < coefRw)
            lf_coef_px(a, S2, T, (uint)wg_uniform((int)coefA), (uint)wg_uniform((int)coefC), (uint)wg_uniform((int)coefL),
                       (uint)(pr * coefRw + pc));
    }
#else
    // Output q8(x2), real window members only.
    [unroll] for (uint io = 0; io < NWW; ++io) {
#if LF_G2
        [unroll] for (uint kh = 0; kh < KC; ++kh)
            [unroll] for (uint vh = 0; vh < 8; ++vh) acc[io][kh].v[vh] = acc[io][kh].v[vh] * 0.5f;   // exact
#endif
        uint2 o[KC];
        float am = 0.0f;
        [unroll] for (uint k0 = 0; k0 < KC; ++k0) am = sw_absmax8(am, acc[io][k0]);
        if (sw_in448(am)) {
            [unroll] for (uint k = 0; k < KC; ++k) o[k] = wg_pack_q8_nc_ags(acc[io][k]);
        } else {
            [unroll] for (uint k = 0; k < KC; ++k) o[k] = sw_pack_q8_v2(acc[io][k]);
        }
        if (dst[io] >= 0) {
            const uint ob = (uint)dst[io] * SW_C + hh * HALF;
            [unroll] for (uint i = 0; i < KC / 2; ++i) wg_st128(out_p, ob + 16u * i, uint4(o[2 * i], o[2 * i + 1]));
        }
    }
#endif
}
