// SPDX-License-Identifier: MIT
// D3D12 ports of Model L's input linears (issue #89): l_linear_32_32_relu and l_linear_32_32_f16 of l_ops.co
// (clean/l/kernels/l_ops.hip, linear_32_32<RELU>), byte for byte. Maps and blobs: clean/l/FORMATS.md.
//
// One 32 -> 32 linear on a pixel map: acc = b + W x (the bias tile vector as the accumulator start, one WMMA per K block
// of 16 inputs, two K blocks), then RN16. relu: y = q8(max(0, acc)), FP8 out; f16: y = acc, FP16 out. 256 lanes; wave i
// handles pixels 16 i .. 16 i + 15 of its workgroup's 128, lane (t, h) pixel 16 i + t, output channels 16 h + 8 mt + v.
// Blob: bias tile vector (rows R_L(32)), then fragments [mt < 2][kc < 2], rows R_L(32), K order K_L(32).
// No branch at all: a lane past the end of the map (HIP: `continue`, no store) computes pixel npix - 1 (its input is
// clamped there, as HIP's) and stores it to pixel npix - 1, the same bytes the lane of that pixel stores. A branch around
// the store let the D3D12 compiler sink the matrix instructions into it, where they ran with part of the wave masked
// off: wrong values in a partly filled wave (a 35-pixel map, pixels 32..34; found by l_d3d12_ops_check.py).
//
// l_linear_32_32_fused (D3D12 only): the two in one kernel, x1 kept in registers. The relu linear's lane (t, h) stores
// x1's channels 16 h + 8 m .. + 7 (m = 0, 1) as the two FP8 fragments that the f16 linear's lane (t, h) loads as its
// K blocks kc = m: the fragments pass in registers, the same bytes, and the x1 map is neither written nor read. x is
// the q map, y the x2 map: two different buffers (the pair runs q -> x1 -> x2 with x2 over q's buffer; here x2 would
// overwrite q that other workgroups have not read yet). Arguments: x, y, Win1, Win2, npix. Its conversions are the
// driver's instructions (RN16 and the final FP16 by v_cvt_f16_f32, the FP8 pack by v_cvt_pk_fp8_f32 after the clamp:
// wg_platform.hlsli, the same bits as the software forms the pair keeps); the software forms were most of the pair's
// instructions. Measured at 1080p Quality: 0.095 ms against the pair's 0.217 ms (the fused kernel with the software
// forms: 0.236 ms).
//
// @wg-kernel l_linear_32_32_relu co=l_ops.co block=256 wave=32 args=p,p,p,u defines=LIN_RELU=1
// @wg-kernel l_linear_32_32_f16 co=l_ops.co block=256 wave=32 args=p,p,p,u defines=LIN_RELU=0
// @wg-kernel l_linear_32_32_fused co=- block=256 wave=32 args=p,p,p,p,u defines=LIN_RELU=0,LIN_FUSED=1
#include "wg_platform.hlsli"

#ifndef LIN_FUSED
#define LIN_FUSED 0
#endif

WG_ARGS_BEGIN
    uint x, y, w;   // pointers: x [npix][32] FP8, y [npix][32] FP8 or FP16, w the blob (enc0.Win1 / enc0.Win2)
#if LIN_FUSED
    uint w2;        // the f16 linear's blob (enc0.Win2; w = enc0.Win1)
#endif
    int npix;
WG_ARGS_END

#if LIN_FUSED   // (helpers of the fused kernel only: the pair keeps its own code and DXIL)
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

#endif

#if LIN_FUSED
WG_KERNEL(l_linear_32_32_fused)
#elif LIN_RELU
WG_KERNEL(l_linear_32_32_relu)
#else
WG_KERNEL(l_linear_32_32_f16)
#endif
{
    WG_SETUP();
    const uint tid = wg_lane_id(), lane = tid & 31u, hh = (tid >> 4) & 1u;
    const int pix = (int)(wg_block_id() * (WG_BLOCK / WG_WAVE) + (tid >> 5)) * 16 + (int)(tid & 15u);
    const int pk = select(pix < npix, pix, npix - 1);
    // every load the matrix instructions need, before the first of them
    uint2 xf[2];
    [unroll] for (uint kc = 0; kc < 2u; ++kc) xf[kc] = wg_ld64(x, (uint)pk * 32u + hh * 16u + 8u * kc);
    v8f bias[2];
    uint2 fr[2][2];
#if LIN_FUSED
    lin_blob(w, lane, hh, bias, fr);
    v8f bias2[2];
    uint2 fr2[2][2];
    lin_blob(w2, lane, hh, bias2, fr2);
    // the relu linear: x1's fragments (the bytes its kernel stores at pk * 32 + hh * 16 + 8 m). The conversions by the
    // driver's instructions (wg_platform.hlsli: the same bits as the pair's software wg_rn16 / wg_pack_q8; no branch in
    // this kernel): RN16 = v_cvt_f16_f32 then the exact widening, the pack v_cvt_pk_fp8_f32 after the clamp.
    uint2 x1[2];
    [unroll] for (uint m1 = 0; m1 < 2u; ++m1) {
        const uint4 h = wg_to_f16_ags(lin_acc(bias[m1], fr[m1], xf));
        v8f r;
        [unroll] for (uint j = 0; j < 4u; ++j) {
            r.v[2 * j] = wg_fmax(wg_h2f_lo(h[j]), 0.0f);
            r.v[2 * j + 1] = wg_fmax(wg_h2f_hi(h[j]), 0.0f);
        }
        x1[m1] = wg_pack_q8_ags(r);
    }
    // the f16 linear on them (its kernel's loads of x1 at pk * 32 + hh * 16 + 8 kc): FP16(RN16(acc)) = FP16(acc), one
    // conversion (rounding to FP16 is idempotent, NaN payloads included)
    [unroll] for (uint m = 0; m < 2u; ++m)
        wg_st128(y, ((uint)pk * 32u + hh * 16u + 8u * m) * 2u, wg_to_f16_ags(lin_acc(bias2[m], fr2[m], x1)));
#else
    // (the pair's own code, unchanged)
    [unroll] for (uint mt = 0; mt < 2u; ++mt) {
        bias[mt] = wg_load8f(w, (hh * 8u + mt * 16u) * 4u);
        [unroll] for (uint kc2 = 0; kc2 < 2u; ++kc2) fr[mt][kc2] = wg_ld64(w, 128u + (mt * 2u + kc2) * 256u + lane * 8u);
    }
    [unroll] for (uint m = 0; m < 2u; ++m) {
        v8f acc = bias[m];
        acc = wg_wmma_fp8(fr[m][0], xf[0], acc);
        acc = wg_wmma_fp8(fr[m][1], xf[1], acc);
        [unroll] for (uint i = 0; i < 8u; ++i) acc.v[i] = wg_rn16(acc.v[i]);
#if LIN_RELU
        [unroll] for (uint j = 0; j < 8u; ++j) acc.v[j] = wg_fmax(acc.v[j], 0.0f);
        wg_st64(y, (uint)pk * 32u + hh * 16u + 8u * m, wg_pack_q8(acc));
#else
        wg_st128(y, ((uint)pk * 32u + hh * 16u + 8u * m) * 2u, wg_to_f16(acc));
#endif
    }
#endif
}
