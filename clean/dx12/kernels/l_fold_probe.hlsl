// SPDX-License-Identifier: MIT
// Start-up probe of the driver's conversion fold used by Model L's l_post_recon_p12 (hlsl/l_post_encode.hlsli): the
// device runs it once (clean/runtime/fold_probe.h holds the inputs and the expected bytes, computed from the exact
// products) and takes the no-fold port of the kernel when any history code differs. WrongGPU's own code.
// Buffer: lane l reads six FP32 values at byte 24 l (x0 x1 x2 = log2 arguments of the history codes, d0 d1 d2 = display
// values) and writes the three converted dwords at byte 768 + 12 l (q0 | q1 << 16, q2 | h(d0) << 16, h(d1) | h(d2) << 16).
// @wg-kernel l_fold_probe co=- block=32 wave=32 args=p
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint dst;
WG_ARGS_END
#include "l_post_encode.hlsli"

WG_KERNEL(l_fold_probe)
{
    WG_SETUP();
    const uint lane = wg_lane_id();
    const uint z = wg_grid_ >> 31;   // 0, opaque to the compiler
    const uint in0 = lane * 24u;
    const float x0 = wg_ldf(dst, in0), x1 = wg_ldf(dst, in0 + 4u), x2 = wg_ldf(dst, in0 + 8u);
    const float d0 = asfloat(asuint(wg_ldf(dst, in0 + 12u)) ^ z), d1 = asfloat(asuint(wg_ldf(dst, in0 + 16u)) ^ z);
    const float d2 = asfloat(asuint(wg_ldf(dst, in0 + 20u)) ^ z);
    const uint4 hc = lp_encode_fold(x0, x1, x2, d0, d1, d2, z);
    const uint o = 768u + lane * 12u;
    wg_st32(dst, o, hc.x);
    wg_st32(dst, o + 4u, hc.y);
    wg_st32(dst, o + 8u, hc.z);
}
