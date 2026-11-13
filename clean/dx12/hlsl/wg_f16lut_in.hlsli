// SPDX-License-Identifier: MIT
// Exact per-frame 64K table of the direct encoder's colour chain (round 2 of the f16lut track; wg_f16lut.hlsli has the
// post output table). The game's colour arrives as an FP16 texture (RGBA16F, R11G11B10F, RG16F, R16F: every value a
// texel load returns is an FP16 value, so it is f16tof32 of its own f32tof16 bits; f16lut/run.py proves the round trip for
// all 65536 patterns and that every FP32 NaN maps to an FP16 NaN). Each colour channel value c then enters a chain that
// depends on c and two per-frame scalars only, the exposure e (expo[0], 0 -> 1) and the pre-exposure pi:
//     c' = pi == 1 ? c : native_direct_color_div(c, pi)   (with the removed color_divide store's opaque-xor boundary)
//     ec = e c'; r = kG log2(max(fma(100, min(ec, 32752), 1), 1e-4)); g = ec > 32752 ? kG log2(fma(100, ec, 1)) : r;
//     (v, gc) = ec is NaN ? (0, 0) : (r, g)            (encoder.hlsl pixel_values: values 0..2 and the history fallback)
// so (v, gc) is a table over the 65536 bit patterns of c. f16lut_build_direct (kernels/exposure.hlsl: one lane per
// pattern, launched after the exposure step and before the encoder in every direct frame) writes it, the post output table
// (wg_f16lut.hlsli: it needs dec_scale, also known after the exposure step) and both headers; the encoder uses the table
// when its header names the encoder's own e and pi and the host says the colour format is FP16 (nd_color_f16), else it
// runs the chain. (The post staging chain enc_in(e c') measured slower as a table: three scattered 4-byte gathers per
// staged texel cost more than the chain, post +12 us.) Buffer (runtime.cpp f16lut, the direct kernels' colour pointer):
//   [0, 16) output-table header (wg_f16lut.hlsli), [16, 32) input header (magic, e bits, pi bits, 1),
//   [256, +256 KB) output table, [256 + 256 KB, +512 KB) encoder table ((v, gc) FP32 pairs).
// Proof: clean/dx12/f16lut/run_in.py (all 65536 c for a grid of (e, pi) against a copy of the encoder's chain and a HIP
// oracle of color_divide + encoder.hip's pixel_values colour values).
// Needs: wg_platform.hlsli, native_direct.hlsli and wg_f16lut.hlsli included before it.
#pragma once

#define WG_F16LUT_IN_MAGIC 0x4e493146u
#define WG_F16LUT_IN 16u
#define WG_F16LUT_ENC (256u + 65536u * 4u)
#define WG_F16LUT_ALL_BYTES (256u + 65536u * 12u)

// b1 word 14 (NativeDirectParams.padding[3]; words 11..13 hold the pre-exposure division window): 1 = the colour texture's SRV format is FP16 (RGBA16F, R11G11B10F, RG16F,
// R16F): every texel value is an FP16 value.
#define nd_color_f16 nd_pad4

#define kF16lutG asfloat(0x3dddf182u)      // encoder.hlsl kG

// nd_color_xy's division of a raw texel channel (z: an opaque zero, nd_rw >> 31: the removed store's boundary)
float f16lut_div(float c, float pi, uint z)
{
    if (pi == 1.0f) return c;
    return asfloat(asuint(native_direct_color_div_ref(c, pi)) ^ z);   // native_direct_color_div3's bits (generator only)
}

// encoder.hlsl pixel_values: (v, gc) of one channel value c (already divided)
float2 f16lut_enc_vg(float e, float c)
{
    float ec = e * c;
    float r = kF16lutG * wg_log2(wg_fmax(mad(100.0f, wg_fmin(ec, 32752.0f), 1.0f), kF16lut1em4));
    float g = ec > 32752.0f ? kF16lutG * wg_log2(mad(100.0f, ec, 1.0f)) : r;
    bool nan = ec != ec;
    return float2(nan ? 0.0f : r, nan ? 0.0f : g);
}

// The encoder table entry of FP16 pattern i: (v, gc) bits of the texel value f16tof32(i) (e0 = expo[0])
uint2 f16lut_enc_entry(uint i, float e0, float pi, uint z)
{
    const float e = e0 == 0.0f ? 1.0f : e0;
    return asuint(f16lut_enc_vg(e, f16lut_div(f16tof32(i), pi, z)));
}

// The encoder table may be used for (e, pi): the colour is FP16 and the header names e and pi (wave-uniform).
bool f16lut_in_valid(uint tab, uint4 h, float e, float pi)
{
    return and(and(tab != 0u, nd_color_f16 != 0u),
               and(h.x == WG_F16LUT_IN_MAGIC, and(h.y == asuint(e), h.z == asuint(pi))));
}

// One lane per FP16 pattern i (i < 65536): every table entry of i; lane 0 writes the headers (the consumers run in later
// dispatches). e = expo[0] (0 -> 1), dec_scale = expo[16].
void f16lut_build(uint tab, uint i, float e0, float dec_scale, float pi, uint rne16, uint z)
{
    const float e = e0 == 0.0f ? 1.0f : e0;
    wg_mem.Store(tab + WG_F16LUT_DEC + 4u * i, asuint(f16lut_out(i, dec_scale, pi, rne16 != 0u, z)));
    const bool f16 = nd_color_f16 != 0u;
    if (f16) wg_mem.Store2(tab + WG_F16LUT_ENC + 8u * i, f16lut_enc_entry(i, e0, pi, z));
    if (i == 0u) {
        const bool finite = and((asuint(dec_scale) & 0x7f800000u) != 0x7f800000u, (asuint(pi) & 0x7f800000u) != 0x7f800000u);
        wg_mem.Store4(tab, finite ? uint4(WG_F16LUT_MAGIC, asuint(dec_scale), asuint(pi), rne16) : uint4(0u, 0u, 0u, 0u));
        wg_mem.Store4(tab + WG_F16LUT_IN, f16 ? uint4(WG_F16LUT_IN_MAGIC, asuint(e), asuint(pi), 1u) : uint4(0u, 0u, 0u, 0u));
    }
}
