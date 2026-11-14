// SPDX-License-Identifier: MIT
// Frame generation (issue #96), the matrix probe (fg_probe.hip, clean/dx12/kernels/fg_probe.hlsl): is a float32 value
// (bits) exactly halfway between two adjacent float16 values? Only there can RN16(RN32(s)) differ from RN16(s) for an
// exact sum s, so a float32-accumulate path is exact once these values are redone exactly. Integer code, the same on
// both back ends; FG_FN is defined by the including file.
//   float16 normal range (2^-14 <= |x| < 2^16, which includes 65520, the midpoint above 65504): the 13 low fraction
//   bits are exactly 0x1000;
//   float16 subnormal range (|x| < 2^-14): x = (odd integer) * 2^-25.
// A float32 sum of float16 products is never a float32 subnormal (its magnitude is 0 or at least 2^-48).
#ifndef FG_PROBE_MID_H
#define FG_PROBE_MID_H

FG_FN uint fg_mid16(uint x) {
    const uint a = x & 0x7FFFFFFFu;
    const uint e = a >> 23;                                  // biased exponent
    if (e == 255u || e == 0u) return 0u;                     // inf, NaN, zero (no float32 subnormals here)
    if (e >= 113u) return (e <= 142u && (a & 0x1FFFu) == 0x1000u) ? 1u : 0u;
    // x = M * 2^(e - 150), M = 1.f in 24 bits; x * 2^25 = M * 2^(e - 125) is an odd integer exactly when M's lowest
    // set bit is bit s = 125 - e (s >= 13 here; s > 23 means x < 2^-25: never a midpoint)
    const uint s = 125u - e;
    if (s > 23u) return 0u;
    const uint M = (a & 0x7FFFFFu) | 0x800000u;
    return (M & ((2u << s) - 1u)) == (1u << s) ? 1u : 0u;
}

#endif
