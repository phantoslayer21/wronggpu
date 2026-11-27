// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): cheap float16 helpers of the fast path with the same bits as the exact ones of
// fg_exact16.h, shared by HIP (fg_net1_mma.hip) and the D3D12 ports. The including file defines FG_FN, FG_F2H(float) ->
// float16 bits (nearest even), FG_H2F(bits) -> float (exact), FG_SEL(c, a, b), and includes fg_exact16.h first.
#ifndef FG_FAST16_H
#define FG_FAST16_H

// net1's leaky ReLU, y = max(x, h(s * x)) with the delivered slope (0x211E): fg_leaky16's result for every input. The
// product of two float16 values is exact in single precision (22 significant bits; never a single-precision subnormal),
// so one float16 rounding of it is h(s * x). The only possible difference to the integer path is the sign of a zero
// product, and the max (by fg_key16, which treats -0 and +0 as equal, keeping x) discards it. NaN stays as it is.
// Checked on all 65,536 inputs (runner/fg_net1_mma.py, leaky check).
FG_FN uint fg_leaky_fast16(uint x) {
    const uint t = FG_F2H(FG_H2F(FG_LEAKY_SLOPE16) * FG_H2F(x));
    const uint y = FG_SEL(fg_key16(t) > fg_key16(x), t, x);
    return FG_SEL(fg_is_nan16(x) != 0u, x, y);
}

// net1's skip add of an upscaled value u and a skip value a, as fg_add_f16 writes it: h(u + a), NaN as 0x7E00
FG_FN uint fg_add_fast16(uint u, uint a) {
    const uint s = FG_F2H(FG_H2F(u) + FG_H2F(a));
    return FG_SEL(fg_is_nan16(s) != 0u, FG_NAN16, s);
}

#endif
