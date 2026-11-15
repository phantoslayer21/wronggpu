// SPDX-License-Identifier: MIT
// Model L post: the history texel's FP16 encode through the driver's conversion fold, shared by
// clean/dx12/kernels/l_post.hlsl (l_post_recon_p12) and its start-up probe clean/dx12/kernels/l_fold_probe.hlsl, so
// that both compile the same expression. WrongGPU's own code. Include after wg_platform.hlsli.
//
// q_i = (_Float16)(kEncK x_i), one rounding of the exact product, as HIP's v_fma_mixlo_f16 x, kEncK, 0 (+0 for a zero
// product). Written as mad(x, kEncK, 0) feeding the driver's round-to-nearest-even matrix conversion, which the gfx1201
// D3D12 compiler folds into exactly that instruction (v_fma_mixlo_f16 / v_fma_mixhi_f16, as Model M's post). Without
// the fold the FP32 product would be rounded twice (a value whose FP32 product lands on an FP16 midpoint moves by one
// FP16 ulp). Guards: build_dx12.py's ISA check on the kernel (isa= in its declaration: the six sites, kEncK as the
// literal with a 0 addend) and two start-up steps of the native device, both of which must pass: the probe
// (l_fold_probe, clean/runtime/fold_probe.h: this expression on adversarial inputs) and the differential self-test of
// the real post shader (its fold port against its no-fold port on the crafted frame of clean/runtime/fold_selftest.h,
// raw bytes) with the check of the fold port's own values. Otherwise, or on any error, the device runs the no-fold
// port of the kernel.
#pragma once

#ifndef LP_MUT
#define LP_MUT 0
#endif
#ifndef LP_MUTSITE
#define LP_MUTSITE 0   // 1 ... 6: negative control (never shipped): site LP_MUTSITE - 1 = 3 row + channel rounded twice
#endif
static const float kEncK = asfloat(0x3dddf184u);        // 0.36f * 0.30102999566398120f

// one history code's conversion input: mad(x, kEncK, 0), which the driver folds into the conversion (one rounding),
// or (twice, a compile-time constant of the negative controls) the FP32 product made opaque first (two roundings)
float lp_enc_in(float x, uint z, bool twice)
{
    return twice ? asfloat(asuint(kEncK * x) ^ z) : mad(x, kEncK, 0.0f);
}

uint4 lp_encode_conv(float x0, float x1, float x2, float d0, float d1, float d2, uint z, int twice_ch)
{
    v8f cv;
    cv.v[0] = lp_enc_in(x0, z, LP_MUT == 3 || twice_ch == 0);
    cv.v[1] = lp_enc_in(x1, z, LP_MUT == 3 || twice_ch == 1);
    cv.v[2] = lp_enc_in(x2, z, LP_MUT == 3 || twice_ch == 2);
    cv.v[3] = d0;
    cv.v[4] = d1;
    cv.v[5] = d2;
    cv.v[6] = 0.0f;
    cv.v[7] = 0.0f;
    return wg_to_f16_ags(cv);
}

// the 8-value conversion of the history codes (x0 .. x2: the FP32 log2 arguments) and the three display values
// d0 .. d2 (already rounded to FP32, passed as opaque values by the caller); z = an opaque zero (the mutants only);
// row: the row of the pair (wave-uniform; the single-site mutants only)
uint4 lp_encode_fold(float x0, float x1, float x2, float d0, float d1, float d2, uint z, int row = 0)
{
#if LP_MUTSITE
    [branch] if (row == (LP_MUTSITE - 1) / 3)
        return lp_encode_conv(x0, x1, x2, d0, d1, d2, z, (LP_MUTSITE - 1) % 3);
#endif
    return lp_encode_conv(x0, x1, x2, d0, d1, d2, z, -1);
}
