// SPDX-License-Identifier: MIT
// Probe kernels of the gfx11 (RDNA 3) platform helpers (wg_platform.hlsli with WG_GFX 11), one entry point each, run by
// test_gfx11_helpers.py through gfx11_probe_host.cpp. Lane l of a 32-lane workgroup; every array is in the arena at the
// byte offset of its argument. Fragments are 8 halves per lane (uint4), accumulators 8 floats per lane stored element-major:
// element i of lane l at (i * 32 + l) * 4, matrix e at + e * 1024.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint pa, pb, pc, po, n, pa2, pb2;
WG_ARGS_END

uint4 load_frag(uint base, uint e, uint l) { return wg_ld128(base, (e * 32u + l) * 16u); }
v8f load_acc(uint base, uint e, uint l)
{
    v8f c;
    [unroll] for (uint i = 0; i < 8u; ++i) c.v[i] = wg_ldf(base, ((e * 8u + i) * 32u + l) * 4u);
    return c;
}
void store_acc(uint base, uint e, uint l, v8f d)
{
    [unroll] for (uint i = 0; i < 8u; ++i) wg_stf(base, ((e * 8u + i) * 32u + l) * 4u, d.v[i]);
}

// One raw instruction (K 0 .. 7), operands already replicated across the halves.
WG_KERNEL(k8)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_wmma_f16_k8(load_frag(pa, e, l), load_frag(pb, e, l), load_acc(pc, e, l)));
}

// K = 16: two chained instructions from replicated fragments (K 0 .. 7 at pa / pb, K 8 .. 15 at pa2 / pb2).
WG_KERNEL(k16)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_wmma_f16_k16(load_frag(pa, e, l), load_frag(pb, e, l), load_frag(pa2, e, l),
                                            load_frag(pb2, e, l), load_acc(pc, e, l)));
}

// HIP's wg_wmma_f16 shape: lane l row / column l % 16, half h holds K 8 h .. 8 h + 7.
WG_KERNEL(f16_gfx12_shape)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_wmma_f16(load_frag(pa, e, l), load_frag(pb, e, l), load_acc(pc, e, l)));
}

// The driver's F32 -> F16 conversion (0x32) of 8 floats per lane: out = 4 dwords per lane.
WG_KERNEL(convert)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    wg_st128(po, l * 16u, wg_to_f16_ags(load_acc(pa, 0u, l)));
}

// wg_xor16 of one float per lane and of a uint4 (wg_xor16_4); the accumulator row table of the lane.
WG_KERNEL(xor16)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    wg_stf(po, l * 4u, wg_xor16(wg_ldf(pa, l * 4u)));
    wg_st128(po, 128u + l * 16u, wg_xor16_4(wg_ld128(pb, l * 16u)));
}
WG_KERNEL(d_row)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [unroll] for (uint i = 0; i < 8u; ++i) wg_st32(po, (i * 32u + l) * 4u, wg_d_row(i));
}

// wg_xor16 after a matrix instruction (the exchange of the accumulator rows 2 i and 2 i + 1).
WG_KERNEL(xor16_after_mma)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    v8f d = wg_wmma_f16_k8(load_frag(pa, 0u, l), load_frag(pb, 0u, l), load_acc(pc, 0u, l));
    [unroll] for (uint i = 0; i < 8u; ++i) d.v[i] += wg_xor16(d.v[i]);
    store_acc(po, 0u, l, d);
}

// Rounding of FP32 -> FP16 in a shader that contains a matrix instruction (RADV): f32tof16, a (half) cast, the form the
// gfx11 branch of wg_gelu_lut_off2 uses, and the driver's conversion. 8 floats per lane at pa; out: three regions of
// 8 dwords per lane at po (f32tof16, cast, asfloat16(f32tof16)), then the convert (4 dwords per lane) at +3072 and the
// matrix result at +4096.
WG_KERNEL(round_modes)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    const v8f c = load_acc(pa, 0u, l);
    const v8f z = wg_wmma_f16_k8(uint4(0, 0, 0, 0), uint4(0, 0, 0, 0), c);   // keeps the shader a matrix shader
    const uint4 r = wg_to_f16_ags(c);
    wg_st128(po, 3u * 1024u + l * 16u, r);
    [unroll] for (uint k = 0; k < 8u; ++k) {
        wg_st32(po, (k * 32u + l) * 4u, f32tof16(c.v[k]));
        const half h = (half)c.v[k];
        wg_st32(po, 1024u + (k * 32u + l) * 4u, (uint)asuint16(h));
        const half g = asfloat16((uint16_t)f32tof16(c.v[k]));
        wg_st32(po, 2048u + (k * 32u + l) * 4u, (uint)asuint16(g));
        wg_stf(po, 4096u + (k * 32u + l) * 4u, z.v[k]);
    }
}
