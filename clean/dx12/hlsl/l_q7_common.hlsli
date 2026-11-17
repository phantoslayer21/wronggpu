// SPDX-License-Identifier: MIT
// The FP16 form's exact arithmetic for the D3D12 ports of Model L's q7 kernels (clean/l/kernels/l_q7.hip, l_q7w.hip):
// one rounding to FP16 (software, nearest even), exact FP16 additions (error-free sums, the tail picking the side at
// an FP16 midpoint), and one FP8 op over 32 terms computed exactly. Porting notes: clean/dx12/kernels/l_q7.hlsl.
#pragma once

float h16(float v) { return wg_rn16(v); }
uint bits16(float v) { return wg_f2h(v); }
float from16(uint b) { return f16tof32(b & 0xffffu); }
float fp8v(uint code) { return wg_e4m3_to_f32(code & 0xffu); }
bool fin(float v) { return (asuint(v) & 0x7f800000u) != 0x7f800000u; }
// Output stores: the plain codes. This driver generates NaNs with the sign set where HIP does not; the checks compare
// with only a NaN's sign bit masked (the accepted port exception), since a canonical NaN would cost about 3 VALU per
// stored dword (RGA ISA, measured 2026-11-15) and real data has no NaN.
uint out8(float v) { return wg_f32_to_e4m3(v); }
uint out16(float v) { return wg_f2h(v); }
int q7_mirror(int p, int n)
{
    p = p < 0 ? -p : p;
    return p >= n ? 2 * (n - 1) - p : p;
}

void two_sum(float a, float b, out float s, out float e)
{
    s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

float rn16_pair(float z, float t)
{
    if (!fin(z) || t == 0.0f) return h16(z);
    uint b = asuint(z);
    bool mid;
    if (abs(z) >= asfloat(0x38800000u)) {
        mid = (b & 0x1fffu) == 0x1000u;
    } else {
        float q = z * 33554432.0f;
        int qi = (int)q;
        mid = (float)qi == q && (qi & 1) != 0;
    }
    if (!mid) return h16(z);
    bool up = t > 0.0f;
    uint nb = (up == (z > 0.0f)) ? b + 1u : b - 1u;
    return h16(asfloat(nb));
}

float add16(float a, float b)
{
    float s, e;
    two_sum(a, b, s, e);
    return rn16_pair(s, e);
}
float mul16(float a, float b) { return h16(a * b); }
float fma16(float a, float b, float c) { return add16(a * b, c); }

float gelu16(float h)
{
    float y = min(max(h, -2.0f), 2.0f);
    const float A = 0.412109375f, B = 0.0810546875f;
    return mul16(h, add16(0.5f, mul16(y, add16(A, -mul16(B, abs(y))))));
}
