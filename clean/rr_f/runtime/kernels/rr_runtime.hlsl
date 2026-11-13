// SPDX-License-Identifier: MIT
// WrongGPU RR runtime (clean/rr_f/runtime): the passes around the preset-F kernels that touch the game's resources.
// Original code; conventions of clean/dx12 (wg_platform.hlsli, one arena at u0, root constants b0). Texture access:
// one SRV at t0 and one typed UAV at u2 (the runtime's second descriptor table).
//
//   rr_rt_ingest   one game resource (Texture2D<float4> t0, the API read conversion) -> an arena image in a format
//                  the preset-F kernels read, region [0, w) x [0, h), channels c0 .. c0 + ch - 1:
//                    mode 0  f32 [h][w][ch] (optional linear conversion high = low * scale + bias, wg_rr_adapter.h
//                            WgRrImage.conversion == 1): the input stage's / exposure's f32 images (52.1.2), and the
//                            resolve's f32 surfaces (54b.N)
//                    mode 1  f16 [h][w][ch]: binary16 resources (the value read is a binary16 value: exact)
//                    mode 2  unorm8 x 4 [h][w] (one dword): 8-bit unorm resources as their codes (54b.N reads codes)
//                    mode 3  RGB10A2 [h][w] (one dword): 10-10-10-2 unorm resources as their codes
//                    mode 4  f32 [h][w][ch] of the 54b.N unorm read code / (2^n - 1), one IEEE division, of the
//                            code under the hardware's value (maxc a9 for channels 0..2, a10 for channel 3)
//                  Unorm codes are recovered from the API value: code = nearest integer to value * maxc (the API's
//                  unorm decode is within far less than half a code of code / maxc).
//   rr_rt_fill     n dwords := value (zero-filled (re)creation of the images of page 53.8 / 53.9).
//   rr_rt_store    an arena image written by the resolve (54b.N store conversion done there) -> the game's output
//                  resource through the typed UAV u2 (the destination format's own conversion: exact for binary16 /
//                  binary32 values into float formats and for codes into unorm formats); srgb: the linear value is
//                  encoded to sRGB before the store (UNORM view of an sRGB output; WrongGPU's choice, approximate).
//   rr_rt_selftest one wave: an fp8 WMMA and a driver f32 -> f16 conversion through the driver channel (the runtime's
//                  check that the channel works on the game's device).
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23;
WG_ARGS_END

Texture2D<float4> rr_tex : register(t0);
RWTexture2D<float4> rr_out : register(u2);

uint rr_rt_lane() { return wg_block_id() * WG_BLOCK + wg_lane_id(); }

uint rr_code(float v, uint maxc)
{
    const float s = (isnan(v) ? 0.0f : min(max(v, 0.0f), 1.0f)) * (float)maxc;
    return min(maxc, (uint)(s + 0.5f));
}

// ---- ingest: a0 dst, a1 w, a2 h, a3 ch, a4 conversion, a5 scale, a6 bias (f32 bits), a7 mode, a8 first channel,
//      a9 maxc of channels 0..2, a10 maxc of channel 3 (mode 4) ---------------------------------------------------
WG_KERNEL(rr_rt_ingest)
{
    WG_SETUP();
    const uint i = rr_rt_lane();
    if (i >= a1 * a2) return;
    const uint x = i % a1, y = i / a1;
    float4 v = rr_tex.Load(int3((int)x, (int)y, 0));
    if (a4 == 1u) v = v * asfloat(a5) + asfloat(a6);
    if (a7 == 2u) {
        wg_st32(a0, i * 4u, rr_code(v.x, 255u) | (rr_code(v.y, 255u) << 8) | (rr_code(v.z, 255u) << 16) | (rr_code(v.w, 255u) << 24));
        return;
    }
    if (a7 == 3u) {
        wg_st32(a0, i * 4u, rr_code(v.x, 1023u) | (rr_code(v.y, 1023u) << 10) | (rr_code(v.z, 1023u) << 20) | (rr_code(v.w, 3u) << 30));
        return;
    }
    [unroll] for (uint k = 0; k < 4u; ++k) {
        if (k >= a3) break;
        const uint c = a8 + k;
        const float t = c == 0u ? v.x : (c == 1u ? v.y : (c == 2u ? v.z : v.w));
        if (a7 == 1u) {
            wg_st16(a0, (i * a3 + k) * 2u, f32tof16(t));
        } else if (a7 == 4u) {
            const uint m = c == 3u ? a10 : a9;
            wg_st32(a0, (i * a3 + k) * 4u, asuint(wg_div((float)rr_code(t, m), (float)m)));
        } else {
            wg_st32(a0, (i * a3 + k) * 4u, asuint(t));
        }
    }
}

// ---- fill: a0 dst, a1 dwords, a2 value -----------------------------------------------------------------------------
WG_KERNEL(rr_rt_fill)
{
    WG_SETUP();
    const uint i = rr_rt_lane();
    if (i < a1) wg_st32(a0, i * 4u, a2);
}

// ---- store: a0 src, a1 w, a2 h (region), a3 src pitch (texels), a4 fmt (0 f32, 1 f16, 2 unorm8 x 4, 3 RGB10A2),
//      a5 channels (fmt 0 / 1), a6 ox, a7 oy (destination origin), a8 srgb --------------------------------------------
float rr_srgb(float v)
{
    v = isnan(v) ? 0.0f : min(max(v, 0.0f), 1.0f);
    return v <= 0.0031308f ? v * 12.92f : 1.055f * pow(v, 0.41666666f) - 0.055f;
}
WG_KERNEL(rr_rt_store)
{
    WG_SETUP();
    const uint i = rr_rt_lane();
    if (i >= a1 * a2) return;
    const uint x = i % a1, y = i / a1, e = y * a3 + x;
    float4 o = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (a4 == 2u) {
        const uint w = wg_ld32(a0, e * 4u);
        o = float4(wg_div((float)(w & 255u), 255.0f), wg_div((float)((w >> 8) & 255u), 255.0f),
                   wg_div((float)((w >> 16) & 255u), 255.0f), wg_div((float)(w >> 24), 255.0f));
    } else if (a4 == 3u) {
        const uint w = wg_ld32(a0, e * 4u);
        o = float4(wg_div((float)(w & 1023u), 1023.0f), wg_div((float)((w >> 10) & 1023u), 1023.0f),
                   wg_div((float)((w >> 20) & 1023u), 1023.0f), wg_div((float)(w >> 30), 3.0f));
    } else {
        [unroll] for (uint c = 0; c < 4u; ++c) {
            if (c >= a5) break;
            o[c] = a4 == 1u ? f16tof32(wg_ld16(a0, (e * a5 + c) * 2u)) : asfloat(wg_ld32(a0, (e * a5 + c) * 4u));
        }
    }
    if (a8 != 0u) o = float4(rr_srgb(o.x), rr_srgb(o.y), rr_srgb(o.z), o.w);
    rr_out[uint2(a6 + x, a7 + y)] = o;
}

// ---- pass-through (WrongGPU's fail-safe for a failed evaluate, not part of preset F): the colour's render rectangle
//      at (a0, a1), a2 x a3 texels (t0), bilinear to the output rectangle at (a4, a5), a6 x a7 pixels (u2);
//      a8 / a9 = a2 / a6, a3 / a7 as f32 bits (host-computed); non-finite values -> 0 ----------------------------------
WG_KERNEL(rr_rt_pass)
{
    WG_SETUP();
    const uint i = rr_rt_lane();
    if (i >= a6 * a7) return;
    const uint x = i % a6, y = i / a6;
    const float u = ((float)x + 0.5f) * asfloat(a8) - 0.5f;
    const float v = ((float)y + 0.5f) * asfloat(a9) - 0.5f;
    const float fu = floor(u), fv = floor(v);
    const float tu = u - fu, tv = v - fv;
    const int xm = (int)a2 - 1, ym = (int)a3 - 1;
    const int x0 = clamp((int)fu, 0, xm), x1 = clamp((int)fu + 1, 0, xm);
    const int y0 = clamp((int)fv, 0, ym), y1 = clamp((int)fv + 1, 0, ym);
    const int ox = (int)a0, oy = (int)a1;
    const float4 c00 = rr_tex.Load(int3(ox + x0, oy + y0, 0)), c10 = rr_tex.Load(int3(ox + x1, oy + y0, 0));
    const float4 c01 = rr_tex.Load(int3(ox + x0, oy + y1, 0)), c11 = rr_tex.Load(int3(ox + x1, oy + y1, 0));
    float4 c = lerp(lerp(c00, c10, tu), lerp(c01, c11, tu), tv);
    [unroll] for (uint k = 0; k < 4u; ++k)
        if (isnan(c[k]) || isinf(c[k])) c[k] = 0.0f;
    rr_out[uint2(a4 + x, a5 + y)] = c;
}

// ---- self-test: a0 out (8 f32 + 4 dwords per lane) -----------------------------------------------------------------
WG_KERNEL(rr_rt_selftest)
{
    WG_SETUP();
    const uint lane = wg_lane_id();
    const uint2 one = uint2(0x38383838u, 0x38383838u);      // e4m3 1.0 in every byte
    const v8f d = wg_wmma_fp8(one, one, wg_splat(0.5f));     // 16 products of 1 x 1 + 0.5 = 16.5
    const uint4 h = wg_to_f16_ags(d);                          // 16.5 = 0x4C20
    [unroll] for (uint k = 0; k < 8u; ++k) wg_st32(a0, lane * 48u + 4u * k, asuint(d.v[k]));
    wg_st128(a0, lane * 48u + 32u, h);
}
