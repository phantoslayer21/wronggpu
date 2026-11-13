// SPDX-License-Identifier: MIT
// Probe build only (WG_RR_PROBE): the plain bilinear upscale the probe writes into the game's output, and the
// 16 x 16 mean of the normal input. Compiled by tools/build.py with fxc (cs_5_1; the root signature is embedded).
// Constants: rect0 = source rectangle (x, y, w, h), rect1 = destination rectangle (upscale) or patch origin
// (patch), misc.x = bit 0 decode sRGB on read, bit 1 encode sRGB on write; misc.y = readback slot; misc.z = marker.

#define WG_PROBE_RS "RootConstants(num32BitConstants=12, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0))"

cbuffer ProbeConstants : register(b0) {
  uint4 rect0;
  uint4 rect1;
  uint4 misc;
};

Texture2D<float4> Src : register(t0);

float3 srgb_to_linear(float3 c) {
  return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float3 linear_to_srgb(float3 c) {
  c = saturate(c);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// One texel of the source rectangle, clamped to its edges.
float4 fetch(int2 p) {
  int2 lo = int2(rect0.xy);
  int2 hi = int2(rect0.xy + rect0.zw) - 1;
  float4 v = Src.Load(int3(clamp(p, lo, hi), 0));
  if (misc.x & 1u) v.rgb = srgb_to_linear(v.rgb);
  return v;
}

#ifdef WG_PROBE_UPSCALE
RWTexture2D<float4> Dst : register(u0);

[RootSignature(WG_PROBE_RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rect1.z || id.y >= rect1.w) return;
  // pixel centres: s = (d + 0.5) * src / dst - 0.5, in source pixels relative to the source rectangle
  float2 s = (float2(id.xy) + 0.5) * (float2(rect0.zw) / float2(rect1.zw)) - 0.5;
  float2 f = floor(s);
  float2 t = s - f;
  int2 i = int2(f) + int2(rect0.xy);
  float4 a = fetch(i);
  float4 b = fetch(i + int2(1, 0));
  float4 c = fetch(i + int2(0, 1));
  float4 d = fetch(i + int2(1, 1));
  float4 v = lerp(lerp(a, b, t.x), lerp(c, d, t.x), t.y);
  if (misc.x & 2u) v.rgb = linear_to_srgb(v.rgb);
  Dst[rect1.xy + id.xy] = v;
}
#endif

#ifdef WG_PROBE_PATCH
RWByteAddressBuffer Out : register(u0);
groupshared float4 g_sum[256];

[RootSignature(WG_PROBE_RS)]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_GroupThreadID, uint gi : SV_GroupIndex) {
  g_sum[gi] = fetch(int2(rect1.xy) + int2(tid.xy));
  GroupMemoryBarrierWithGroupSync();
  for (uint k = 128; k > 0; k >>= 1) {
    if (gi < k) g_sum[gi] += g_sum[gi + k];
    GroupMemoryBarrierWithGroupSync();
  }
  if (gi == 0) {
    uint o = misc.y * 32;
    Out.Store4(o, asuint(g_sum[0] / 256.0));
    Out.Store2(o + 16, uint2(misc.z, 256));
  }
}
#endif
