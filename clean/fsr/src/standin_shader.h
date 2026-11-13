// SPDX-License-Identifier: MIT
#pragma once
// HLSL source of the stand-in upscaler (compiled at run time with D3DCompile, target cs_5_0). Written for WrongGPU
// from the definitions in spec 20 and clean/FORMATS.md; the root constants match StandInConstants in
// standin_d3d12.cpp register for register.
namespace wgfsr {

constexpr const char kStandInShader[] = R"HLSL(
cbuffer Constants : register(b0)
{
    uint outW, outH, renderX, renderY;
    uint renderW, renderH, view, filterMode;
    float jitterX, jitterY, mvScaleX, mvScaleY;
    float mvOffX, mvOffY, warpWeight, gain;
    uint flags, motionW, motionH, pad0;
};

// flags
#define F_JITTER_COMP    1u
#define F_MOTION_OUTRES  2u
#define F_HISTORY_VALID  4u
#define F_OUTPUT_SRGB    8u
#define F_DEPTH_INVERTED 16u
#define F_RESET          32u

Texture2D<float4>   colorTex  : register(t0);
Texture2D<float>    depthTex  : register(t1);
Texture2D<float2>   motionTex : register(t2);
Texture2D<float4>   histTex   : register(t3);
RWTexture2D<float4> outTex    : register(u0);
RWTexture2D<float4> histOut   : register(u1);

int2 ClampRender(int2 p)
{
    return clamp(p, int2(renderX, renderY), int2(renderX + renderW - 1, renderY + renderH - 1));
}

float4 LoadColor(int2 p) { return colorTex.Load(int3(ClampRender(p), 0)); }

// t: position in texels of the color texture, texel centers at i + 0.5.
float4 Bilinear(float2 t)
{
    float2 q = t - 0.5;
    float2 b = floor(q);
    float2 f = q - b;
    int2 i = int2(b);
    float4 a = LoadColor(i), c = LoadColor(i + int2(1, 0)), d = LoadColor(i + int2(0, 1)), e = LoadColor(i + int2(1, 1));
    return lerp(lerp(a, c, f.x), lerp(d, e, f.x), f.y);
}

float Lanczos2(float x)
{
    x = abs(x);
    if (x < 1e-5) return 1.0;
    if (x >= 2.0) return 0.0;
    float px = 3.14159265 * x;
    return 2.0 * sin(px) * sin(0.5 * px) / (px * px);
}

// Lanczos-2 (4x4 taps) with the result clamped to the four nearest texels (no ringing).
float4 Lanczos(float2 t)
{
    float2 q = t - 0.5;
    float2 b = floor(q);
    float2 f = q - b;
    int2 i = int2(b);
    float4 sum = 0;
    float wsum = 0;
    float4 lo = 1e30, hi = -1e30;
    [unroll] for (int y = -1; y <= 2; ++y)
    {
        float wy = Lanczos2(float(y) - f.y);
        [unroll] for (int x = -1; x <= 2; ++x)
        {
            float w = Lanczos2(float(x) - f.x) * wy;
            float4 c = LoadColor(i + int2(x, y));
            sum += w * c;
            wsum += w;
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) { lo = min(lo, c); hi = max(hi, c); }
        }
    }
    return clamp(sum / wsum, lo, hi);
}

float2 RenderPerOutput() { return float2(renderW, renderH) / float2(outW, outH); }

// The color at the output pixel q. With jitter compensation the color texture is read where the scene point under
// the output pixel center was sampled: the image moved by j, so texel position = render position + jitter (FORMATS.md).
float4 Upscaled(uint2 q)
{
    float2 t = float2(renderX, renderY) + (float2(q) + 0.5) * RenderPerOutput();
    if (flags & F_JITTER_COMP) t += float2(jitterX, jitterY);
    return filterMode == 1 ? Lanczos(t) : Bilinear(t);
}

int2 RenderPixelUnder(uint2 q)
{
    return ClampRender(int2(float2(renderX, renderY) + (float2(q) + 0.5) * RenderPerOutput()));
}

// Motion in render pixels (previous = current + motion) for the output pixel q.
float2 MotionRender(uint2 q)
{
    int2 p = (flags & F_MOTION_OUTRES) ? int2(min(q, uint2(motionW - 1, motionH - 1))) : RenderPixelUnder(q);
    float2 raw = motionTex.Load(int3(p, 0));
    return raw * float2(mvScaleX, mvScaleY) + float2(mvOffX, mvOffY);
}

float4 LoadHistory(int2 p) { return histTex.Load(int3(clamp(p, int2(0, 0), int2(outW - 1, outH - 1)), 0)); }

float4 HistoryBilinear(float2 pos)
{
    float2 q = pos - 0.5;
    float2 b = floor(q);
    float2 f = q - b;
    int2 i = int2(b);
    float4 a = LoadHistory(i), c = LoadHistory(i + int2(1, 0)), d = LoadHistory(i + int2(0, 1)), e = LoadHistory(i + int2(1, 1));
    return lerp(lerp(a, c, f.x), lerp(d, e, f.x), f.y);
}

float3 LinearToSrgb(float3 c)
{
    c = saturate(c);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= outW || id.y >= outH) return;
    uint2 q = id.xy;
    float4 result;
    if (view == 1 || view == 2)
    {
        // warp / warpdiff: the previous frame's image moved along the motion vectors.
        float4 cur = Upscaled(q);
        float2 prev = float2(q) + 0.5 + MotionRender(q) / RenderPerOutput();
        bool valid = (flags & F_HISTORY_VALID) != 0 && (flags & F_RESET) == 0 && all(prev >= 0.0) &&
                     all(prev <= float2(outW, outH));
        float4 h = valid ? HistoryBilinear(prev) : cur;
        if (view == 1)
        {
            result = lerp(cur, h, valid ? warpWeight : 0.0);
            histOut[q] = result;
        }
        else
        {
            result = float4(saturate(abs(cur.rgb - h.rgb) * gain), 1.0);
            histOut[q] = cur;
        }
    }
    else if (view == 3)
    {
        // motion: red = x, green = y; 0.5 is no motion, gain/64 per render pixel.
        float2 m = MotionRender(q);
        result = float4(saturate(0.5 + m * gain / 64.0), 0.5, 1.0);
    }
    else if (view == 4)
    {
        // depth: near is bright; gamma 1/gain spreads the usual depth range.
        float d = depthTex.Load(int3(RenderPixelUnder(q), 0));
        float v = (flags & F_DEPTH_INVERTED) ? d : 1.0 - d;
        result = float4(pow(saturate(v), 1.0 / gain).xxx, 1.0);
    }
    else
    {
        result = Upscaled(q);
        if (view == 5)
        {
            // jitter: a 129x129 box in the top-left corner with a dot at the jitter position (+-0.5 render pixel
            // at the box edges; x right, y down).
            int2 o = int2(q) - 64;
            if (all(abs(o) <= 64))
            {
                int2 dot = int2(round(float2(jitterX, jitterY) * 128.0));
                bool axis = o.x == 0 || o.y == 0;
                bool border = abs(o.x) == 64 || abs(o.y) == 64;
                bool hit = all(abs(o - dot) <= 3);
                result = hit ? float4(1, 0.1, 0.1, 1) : (axis || border) ? float4(0.6, 0.6, 0.6, 1) : float4(0.05, 0.05, 0.05, 1);
            }
        }
    }
    if (flags & F_OUTPUT_SRGB) result.rgb = LinearToSrgb(result.rgb);
    outTex[q] = result;
}
)HLSL";

}  // namespace wgfsr
