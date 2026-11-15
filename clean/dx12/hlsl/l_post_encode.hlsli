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
// raw bytes). Otherwise, or on any error, the device runs the no-fold port of the kernel.
#pragma once

#ifndef LP_MUT
#define LP_MUT 0
#endif
#ifndef LP_ENCCONTROL
#define LP_ENCCONTROL 0   // 1: the device guard's sensitivity control port (needs=control): two roundings
#endif
static const float kEncK = asfloat(0x3dddf184u);        // 0.36f * 0.30102999566398120f

// the 8-value conversion of the history codes (x0 .. x2: the FP32 log2 arguments) and the three display values
// d0 .. d2 (already rounded to FP32, passed as opaque values by the caller); z = an opaque zero (mutant 3 only)
uint4 lp_encode_fold(float x0, float x1, float x2, float d0, float d1, float d2, uint z)
{
    v8f cv;
#if LP_MUT == 3 || LP_ENCCONTROL   // two roundings, the FP32 product made opaque before the conversion: the mutant
                                   // (LP_MUT 3, never shipped) and the guard's control port (never selected to run)
    cv.v[0] = asfloat(asuint(kEncK * x0) ^ z);
    cv.v[1] = asfloat(asuint(kEncK * x1) ^ z);
    cv.v[2] = asfloat(asuint(kEncK * x2) ^ z);
#else
    cv.v[0] = mad(x0, kEncK, 0.0f);
    cv.v[1] = mad(x1, kEncK, 0.0f);
    cv.v[2] = mad(x2, kEncK, 0.0f);
#endif
    cv.v[3] = d0;
    cv.v[4] = d1;
    cv.v[5] = d2;
    cv.v[6] = 0.0f;
    cv.v[7] = 0.0f;
    return wg_to_f16_ags(cv);
}
