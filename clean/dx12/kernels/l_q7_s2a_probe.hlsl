// SPDX-License-Identifier: MIT
// D3D12 port of the S2a primitive's measurement probe (clean/l/kernels/l_q7_s2a_probe.hip; the primitive in
// clean/dx12/hlsl/l_q7_op8.hlsli), not used by the runtime: built only when named (probe=1), compared with the HIP
// probe and the exact model by clean/l/runner/l_q7_s2a_check.py. Mutant ports: _m1 ... _m4 (l_q7_s2a_probe.hip).
//
// @wg-kernel l_q7_s2a_probe co=l_q7_s2a_probe.co block=32 wave=32 args=p,p,p,p,p,p,p probe=1
// @wg-kernel l_q7_s2a_probe_m1 co=l_q7_s2a_probe.co entry=l_q7_s2a_probe block=32 wave=32 args=p,p,p,p,p,p,p probe=1 defines=Q7_S2A_MUTANT=1
// @wg-kernel l_q7_s2a_probe_m2 co=l_q7_s2a_probe.co entry=l_q7_s2a_probe block=32 wave=32 args=p,p,p,p,p,p,p probe=1 defines=Q7_S2A_MUTANT=2
// @wg-kernel l_q7_s2a_probe_m3 co=l_q7_s2a_probe.co entry=l_q7_s2a_probe block=32 wave=32 args=p,p,p,p,p,p,p probe=1 defines=Q7_S2A_MUTANT=3
// @wg-kernel l_q7_s2a_probe_m4 co=l_q7_s2a_probe.co entry=l_q7_s2a_probe block=32 wave=32 args=p,p,p,p,p,p,p probe=1 defines=Q7_S2A_MUTANT=4
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint a, w, w16, st, wsum, out_, path;
WG_ARGS_END

groupshared uint g_m[32];

#include "l_q7_common.hlsli"
#include "l_q7_op8.hlsli"

WG_KERNEL(l_q7_s2a_probe)
{
    WG_SETUP();
    uint lane = wg_lane_id(), t = lane & 15u, hf = lane >> 4;
    uint base = 16u * wg_block_id();
    uint i = base + t;
    uint2 a0 = wg_ld64(w, i * 32u + 8u * hf), a1 = wg_ld64(w, i * 32u + 16u + 8u * hf);
    uint2 b0 = wg_ld64(a, i * 32u + 8u * hf), b1 = wg_ld64(a, i * 32u + 16u + 8u * hf);
    uint m = q7_maxcode4(0u, b0.x);
    m = q7_maxcode4(m, b0.y);
    m = q7_maxcode4(m, b1.x);
    m = q7_maxcode4(m, b1.y);
    g_m[lane] = m;
    wg_barrier_lds();
#if Q7_S2A_MUTANT == 3
    float amax = q7_amax(g_m[lane], g_m[lane]);
#else
    float amax = q7_amax(g_m[t], g_m[t + 16u]);
#endif
    bool diag = (t >> 3) == hf;
    float sti = wg_ldf(st, 4u * i);
#if Q7_S2A_MUTANT == 4
    v8f c, zero;
    [unroll] for (uint j = 0; j < 8; ++j) {
        c.v[j] = (8u * hf + j == t) ? sti : 0.0f;
        zero.v[j] = 0.0f;
    }
    v8f lo = wg_wmma_fp8(a0, b0, c);
    v8f hi = wg_wmma_fp8(a1, b1, zero);
    v8f S;
    [unroll] for (uint j2 = 0; j2 < 8; ++j2) S.v[j2] = lo.v[j2] + hi.v[j2];
    float start = 0.0f;
#else
    v8f S = q7_tile_sum(a0, b0, a1, b1);
    float start = sti;
#endif
    float s = S.v[0];
    [unroll] for (uint j3 = 1; j3 < 8; ++j3) s = (t & 7u) == j3 ? S.v[j3] : s;
    if (diag) {
        float ws = wg_ldf(wsum, 4u * i);
        float y;
        if (q7_bounded(ws, amax)) {
            y = q7_join(s, start);
        } else {
            float av[32], wv[32];
            q7_load8(a, i * 32u, av);
            q7_load16(w16, i * 64u, wv);
            y = op8(sti, av, wv);
        }
        wg_stf(out_, 4u * i, y);
        wg_st32(path, 4u * i, q7_bounded(ws, amax) ? 1u : 0u);
    }
}
