// SPDX-License-Identifier: MIT
// Exhaustive probe of the hybrid block form's element-wise helpers (hlsl/wg_hyb.hlsli), not used by the runtime:
// every thread takes 8 consecutive FP16 bit patterns i = 8 t .. 8 t + 7 (t = global thread, 8192 threads = all 65,536)
// and writes, at byte offsets of `dst` (the expected bytes come from clean/l/reference/hyb_tables.py and oracle.py;
// driver: clean/l/runner/l_hyb_probe_check.py):
//   [0, 65536)            hyb_gelu_q8 (h = i)                     FP8 bytes
//   [65536, 196608)       hyb_softmax_e8 (s = i)                  e as FP16 bits (u16)
//   [196608, 327680)      hyb_rsq16 (S' bits = i)                 FP16 bits (u16; compared where S' >= 2^-13)
//   [327680, 458752)      hyb_rcp16 (sum bits = i)                FP16 bits (u16; compared where both are normal)
//   [458752, 720896)      hyb_merge_clamp (x = i as FP16 value)   float32 bits
// and, for the chain-end double rounding, nf float32 values at `src` (nf a multiple of 8): hyb_q8_of_chain bytes at
// [720896, 720896 + nf).
// @wg-kernel l_hyb_probe co=- probe=1 block=256 wave=32 args=p,p,u
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint src, dst;
    uint nf;
WG_ARGS_END
#include "wg_gelu_lut.hlsli"
#include "wg_hyb.hlsli"

WG_KERNEL(l_hyb_probe)
{
    WG_SETUP();
    const uint t = wg_block_id() * 256u + wg_lane_id();
    const uint i0 = 8u * t;
    uint4 h;
    h.x = i0 | ((i0 + 1u) << 16);
    h.y = (i0 + 2u) | ((i0 + 3u) << 16);
    h.z = (i0 + 4u) | ((i0 + 5u) << 16);
    h.w = (i0 + 6u) | ((i0 + 7u) << 16);
    // GELU16 -> FP8
    const uint2 g = hyb_gelu_q8(h);
    wg_st32(dst, i0, g.x);
    wg_st32(dst, i0 + 4u, g.y);
    // softmax weight of the 8 scores
    const v8f s = hyb_h2f8(h);
    const uint4 e = hyb_rn16x8(hyb_softmax_e8(s));
    wg_st32(dst, 65536u + 2u * i0, e.x);
    wg_st32(dst, 65536u + 2u * i0 + 4u, e.y);
    wg_st32(dst, 65536u + 2u * i0 + 8u, e.z);
    wg_st32(dst, 65536u + 2u * i0 + 12u, e.w);
    // rsqrt / rcp tables (the 16-bit results packed in pairs) and the merge clamp
    [unroll] for (uint j = 0; j < 8; j += 2) {
        const uint a = i0 + j, b = i0 + j + 1u;
        wg_st32(dst, 196608u + 2u * a, wg_f2h_bits(hyb_rsq16(a)) | (wg_f2h_bits(hyb_rsq16(b)) << 16));
        wg_st32(dst, 327680u + 2u * a, wg_f2h_bits(hyb_rcp16(a)) | (wg_f2h_bits(hyb_rcp16(b)) << 16));
    }
    [unroll] for (uint k = 0; k < 8; ++k) wg_st32(dst, 458752u + 4u * (i0 + k), asuint(hyb_merge_clamp(s.v[k])));
    // chain-end double rounding on the given float32 values (8 per thread; the conversion runs in wave-uniform control
    // flow on a clamped index, only the stores depend on the range)
    const uint ic = min(i0, nf - 8u);
    v8f x;
    [unroll] for (uint m = 0; m < 8; ++m) x.v[m] = wg_ldf(src, 4u * (ic + m));
    const uint2 q = hyb_q8_of_chain(x);
    if (i0 < nf) {
        wg_st32(dst, 720896u + i0, q.x);
        wg_st32(dst, 720896u + i0 + 4u, q.y);
    }
}
