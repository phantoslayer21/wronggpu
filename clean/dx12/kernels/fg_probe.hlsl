// SPDX-License-Identifier: MIT
// D3D12 port of frame generation's matrix probe (issue #96; clean/fg/kernels/fg_probe.hip, run by
// clean/fg/runner/fg_wmma_probe.py): fg_probe_mma16 (v_wmma_f16_16x16x16_f16 through the driver channel, format F16 ->
// F16) and fg_probe_mma32 (v_wmma_f32_16x16x16_f16, format F16 -> F32, with the float16 accumulator in between), the
// same fragments, memory layout and outputs as the HIP kernels. The exact reference (fg_probe_exact) runs on HIP only.
// Built only when --only names them (fg_ kernels are in no model's set).
//
// @wg-kernel fg_probe_mma16 co=fg_probe.co block=256 wave=32 args=p,p,p,p,p,u,u defines=FG_PROBE_FORM=16
// @wg-kernel fg_probe_mma32 co=fg_probe.co block=256 wave=32 args=p,p,p,p,p,u,u defines=FG_PROBE_FORM=32
#include "wg_platform.hlsli"

#define FG_FN
#include "../../fg/kernels/fg_probe_mid.h"

WG_ARGS_BEGIN
    uint A, B, C, D, F;  // pointers (arena byte offsets)
    int tiles;
    int L;
WG_ARGS_END

// this lane's A row (8 consecutive halves) and B column (8 halves, stride 16) of chunk l of tile t, as v8h (uint4)
void fg_probe_frags(uint t, uint l, uint lane, out uint4 fa, out uint4 fb)
{
    const uint r = lane & 15u, h = lane >> 4;
    const uint base = (t * (uint)L + l) * 512u;
    fa = wg_ld128(A, base + (r * 16u + 8u * h) * 2u);
    uint v[8];
    [unroll] for (uint j = 0; j < 8; ++j) v[j] = wg_ld16(B, base + ((8u * h + j) * 16u + r) * 2u);
    fb = uint4(v[0] | (v[1] << 16), v[2] | (v[3] << 16), v[4] | (v[5] << 16), v[6] | (v[7] << 16));
}

#if FG_PROBE_FORM == 16
WG_KERNEL(fg_probe_mma16)
{
    WG_SETUP();
    const uint lane = wg_lane_id() & 31u;
    const uint t = wg_block_id() * 8u + (wg_lane_id() >> 5);
    if (t >= (uint)tiles) return;                                   // wave-uniform
    const uint r = lane & 15u, h = lane >> 4;
    uint c[8];
    [unroll] for (uint v = 0; v < 8; ++v) c[v] = wg_ld16(C, (t * 256u + (8u * h + v) * 16u + r) * 2u);
    uint4 acc = uint4(c[0] | (c[1] << 16), c[2] | (c[3] << 16), c[4] | (c[5] << 16), c[6] | (c[7] << 16));
    for (uint l = 0; l < (uint)L; ++l) {
        uint4 fa, fb;
        fg_probe_frags(t, l, lane, fa, fb);
        wg_frag d = wg_ags_mma(WG_AGS_F16_F16, wg_frag4(fa), wg_frag4(fb), wg_frag4(acc));
        acc = uint4(d.v[0], d.v[1], d.v[2], d.v[3]);
    }
    const uint o[4] = { acc.x, acc.y, acc.z, acc.w };
    [unroll] for (uint v = 0; v < 8; ++v)
        wg_st16(D, (t * 256u + (8u * h + v) * 16u + r) * 2u, (o[v >> 1] >> ((v & 1u) * 16u)) & 0xffffu);
}
#else
WG_KERNEL(fg_probe_mma32)
{
    WG_SETUP();
    const uint lane = wg_lane_id() & 31u;
    const uint t = wg_block_id() * 8u + (wg_lane_id() >> 5);
    if (t >= (uint)tiles) return;                                   // wave-uniform
    const uint r = lane & 15u, h = lane >> 4;
    uint acc[8], flag[8];
    v8f d = wg_splat(0.0f);
    [unroll] for (uint v = 0; v < 8; ++v) {
        acc[v] = wg_ld16(C, (t * 256u + (8u * h + v) * 16u + r) * 2u);
        flag[v] = 0u;
    }
    for (uint l = 0; l < (uint)L; ++l) {
        uint4 fa, fb;
        fg_probe_frags(t, l, lane, fa, fb);
        v8f c;
        [unroll] for (uint v = 0; v < 8; ++v) c.v[v] = wg_h2f(acc[v]);
        d = wg_wmma_f16(fa, fb, c);
        [unroll] for (uint v = 0; v < 8; ++v) {
            flag[v] |= fg_mid16(asuint(d.v[v]));
            acc[v] = wg_f2h(d.v[v]);
        }
    }
    // Stores through two pointers at one per-lane offset were merged by the driver, keeping only the last (README
    // "Driver exceptions"; seen here too: D kept its sentinel). Remedy: the pointers read through wg_uniform().
    const uint pd = (uint)wg_uniform((int)D), pf = (uint)wg_uniform((int)F);
    [unroll] for (uint v = 0; v < 8; ++v) wg_st32(pd, (t * 256u + (8u * h + v) * 16u + r) * 4u, asuint(d.v[v]));
    [unroll] for (uint v = 0; v < 8; ++v) wg_st32(pf, (t * 256u + (8u * h + v) * 16u + r) * 4u, (flag[v] << 16) | acc[v]);
}
#endif
