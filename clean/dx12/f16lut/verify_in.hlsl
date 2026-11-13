// SPDX-License-Identifier: MIT
// D3D12 side of the encoder colour-table proof (run_in.py). Root constants: obase, e0 bits, pi bits, mode.
// mode 0, for every FP16 bit pattern i (lane i of a 65536-lane grid) and one parameter set (e0 = expo[0], pi):
//   orig[i] = the direct encoder's own colour values (v, gc) of the texel value f16tof32(i): kernels/encoder.hlsl, copied
//             (U.e, nd_color_xy's division with the opaque-xor boundary, pixel_values' channel loop)
//   gen[i]  = f16lut_enc_entry (hlsl/wg_f16lut_in.hlsli), what f16lut_build_direct writes into the table
//   written at obase + 8 i (orig) and obase + 524288 + 8 i (gen).
// mode 1 (the index the encoder computes): out[i] = f16tof32(i) bits, out[65536 + i] = f32tof16(f16tof32(i)); and lane i
//   converts the 256 FP32 NaN patterns (i << 8 | k) (bit 23 = sign, 23-bit mantissa != 0) with f32tof16 and counts the
//   results that are not an FP16 NaN: out[131072 + i].
#include "wg_platform.hlsli"
#include "native_direct.hlsli"
#include "wg_f16lut.hlsli"
#include "wg_f16lut_in.hlsli"

WG_ARGS_BEGIN
    uint obase, e_bits, pi_bits, mode;
WG_ARGS_END

// ---- copied from kernels/encoder.hlsl ----
#define kG asfloat(0x3dddf182u)
#define k1em4 asfloat(0x38d1b717u)
// ------------------------------------------

WG_KERNEL(f16lut_in_verify)
{
    WG_SETUP();
    const uint z = wg_grid_ >> 31;   // an opaque zero (the encoder's nd_rw >> 31)
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (mode == 1u) {
        const float x = f16tof32(i);
        wg_mem.Store(obase + 4u * i, asuint(x));
        wg_mem.Store(obase + 262144u + 4u * i, f32tof16(x));
        uint bad = 0u;
        for (uint k = 0; k < 256u; ++k) {
            const uint v = (i << 8) | k, m = v & 0x7fffffu;
            if (m == 0u) continue;   // +-inf
            const uint h = f32tof16(asfloat(((v >> 23) << 31) | 0x7f800000u | m));
            bad += ((h & 0x7c00u) != 0x7c00u || (h & 0x3ffu) == 0u) ? 1u : 0u;
        }
        wg_mem.Store(obase + 524288u + 4u * i, bad);
        return;
    }
    const float e0 = asfloat(e_bits), pi = asfloat(pi_bits);
    // encoder_main: U.e
    const float e = e0 == 0.0f ? 1.0f : e0;
    // nd_color_xy (the texel value is f16tof32(i)): the division and the removed store's boundary
    float4 cc = float4(f16tof32(i), f16tof32(i), f16tof32(i), 0.0f);
    if (pi != 1.0f) {
        const float3 rgb = float3(native_direct_color_div_ref(cc.x, pi), native_direct_color_div_ref(cc.y, pi),
                                  native_direct_color_div_ref(cc.z, pi));
        cc = float4(asfloat(asuint(rgb) ^ z), cc.w);
    }
    // pixel_values' channel loop
    float v[3], gc[3];
    [unroll] for (uint ch = 0; ch < 3; ++ch) {
        float ec = e * cc[ch];
        float r = kG * wg_log2(wg_fmax(mad(100.0f, wg_fmin(ec, 32752.0f), 1.0f), k1em4));
        float g = ec > 32752.0f ? kG * wg_log2(mad(100.0f, ec, 1.0f)) : r;
        bool nan = ec != ec;
        v[ch] = nan ? 0.0f : r;
        gc[ch] = nan ? 0.0f : g;
    }
    uint2 gen = f16lut_enc_entry(i, e0, pi, z);
#if defined(F16LUT_NEGATIVE)
    gen.x ^= i == 0x3c00u ? 1u : 0u;   // sensitivity: exactly one input per set changed
#endif
    // the three channels must agree (one chain); a disagreement is reported as a difference of orig
    const bool same = and(and(asuint(v[0]) == asuint(v[1]), asuint(v[0]) == asuint(v[2])),
                          and(asuint(gc[0]) == asuint(gc[1]), asuint(gc[0]) == asuint(gc[2])));
    wg_mem.Store2(obase + 8u * i, same ? uint2(asuint(v[0]), asuint(gc[0])) : uint2(0xdeadbeefu, 0xdeadbeefu));
    wg_mem.Store2(obase + 524288u + 8u * i, gen);
}
