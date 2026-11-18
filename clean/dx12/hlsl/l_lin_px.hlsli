// SPDX-License-Identifier: MIT
// Model L's two input linears for one pixel per lane (issue #89; clean/l/kernels/l_ops.hip linear_32_32<RELU = true>,
// then <RELU = false>), D3D12 only, shared by l_linear_32_32_fused (kernels/l_ops_linear.hlsl) and the encoder block
// with the linears as its prologue (kernels/swin_block.hlsl, SW_SRC 3). Lane (t, h) of a 16-pixel tile: its pixel's q
// fragments xf[kc] = q's bytes at pix * 32 + 16 h + 8 kc (FP8); output tile m = x2's channels 16 h + 8 m .. + 7.
// Blob (enc0.Win1 / enc0.Win2): bias tile vector (rows R_L(32)), then fragments [mt < 2][kc < 2], rows R_L(32), K order
// K_L(32) (clean/l/FORMATS.md).
#ifndef L_LIN_PX_HLSLI
#define L_LIN_PX_HLSLI

// the blob's bias tile vectors and weight fragments of lane (t, h)
void lin_blob(uint wb, uint lane, uint hh, out v8f bias[2], out uint2 fr[2][2])
{
    [unroll] for (uint mt = 0; mt < 2u; ++mt) {
        bias[mt] = wg_load8f(wb, (hh * 8u + mt * 16u) * 4u);
        [unroll] for (uint kc2 = 0; kc2 < 2u; ++kc2) fr[mt][kc2] = wg_ld64(wb, 128u + (mt * 2u + kc2) * 256u + lane * 8u);
    }
}

// output tile m before the rounding: b + W x
v8f lin_acc(v8f bias, uint2 fr[2], uint2 xf[2])
{
    v8f acc = bias;
    acc = wg_wmma_fp8(fr[0], xf[0], acc);
    return wg_wmma_fp8(fr[1], xf[1], acc);
}

// Both linears: x2's FP16 tiles y[m] (8 halves each) of the lane's pixel from its q fragments xf. x1 stays in
// registers: the relu linear's lane (t, h) holds x1's channels 16 h + 8 m .. + 7 as the two FP8 fragments that the f16
// linear's lane (t, h) takes as its K blocks kc = m. The conversions by the driver's instructions (wg_platform.hlsli:
// the same bits as the software wg_rn16 / wg_pack_q8; wave-uniform control flow only): RN16 = v_cvt_f16_f32 then the
// exact widening, the pack v_cvt_pk_fp8_f32 after the clamp; FP16(RN16(acc)) = FP16(acc), one conversion (rounding to
// FP16 is idempotent, NaN payloads included).
void lin_pair(uint2 xf[2], v8f bias[2], uint2 fr[2][2], v8f bias2[2], uint2 fr2[2][2], out uint4 y[2])
{
    uint2 x1[2];
    [unroll] for (uint m1 = 0; m1 < 2u; ++m1) {
        const uint4 h = wg_to_f16_ags(lin_acc(bias[m1], fr[m1], xf));
        v8f r;
        [unroll] for (uint j = 0; j < 4u; ++j) {
            // f16tof32 directly, not wg_h2f_lo: its mask costs a v_mov_b16 per FP16 pair (see hyb_h2f_lo in wg_hyb.hlsli)
            r.v[2 * j] = wg_fmax(f16tof32(h[j]), 0.0f);
            r.v[2 * j + 1] = wg_fmax(wg_h2f_hi(h[j]), 0.0f);
        }
        x1[m1] = wg_pack_q8_ags(r);
    }
    [unroll] for (uint m = 0; m < 2u; ++m) y[m] = wg_to_f16_ags(lin_acc(bias2[m], fr2[m], x1));
}

#endif
