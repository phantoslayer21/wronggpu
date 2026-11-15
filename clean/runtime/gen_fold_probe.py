# SPDX-License-Identifier: MIT
"""Writes clean/runtime/fold_probe.h: inputs and expected bytes of the start-up probe of the driver's conversion fold
(clean/dx12/kernels/l_fold_probe.hlsl, hlsl/l_post_encode.hlsli). WrongGPU's own code.

    python clean/runtime/gen_fold_probe.py [out.h]

Lane l reads x0 x1 x2 (the FP32 log2 arguments of the three history codes) and d0 d1 d2 (display values) and writes
q_i = (_Float16)(kEncK x_i) with ONE rounding of the exact product (+0 for a zero product: the fold's +0 addend) and
h(d_i) (round to nearest even). Expected values are computed from exact rationals (fractions.Fraction), never from
FP arithmetic. The x inputs are adversarial: values whose FP32 product kEncK x rounds exactly onto an FP16 midpoint
while the exact product is not that midpoint (residual of both signs; a second rounding moves them by one FP16 ulp),
FP16-subnormal products, +-0, NaN, the encode's floor log2(1e-4) and ordinary values; the d inputs: FP16 midpoints and
their FP32 neighbours, +-0, NaN, values beyond the FP16 range.
"""
import os
import struct
import sys
from fractions import Fraction

import numpy as np

K_BITS = 0x3dddf184   # kEncK


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def rn32(fr):
    """Round a Fraction to the nearest FP32 (ties to even): float() is exact for these products (<= 48 bits)."""
    return np.float32(float(fr))


def rn16_bits(fr):
    """Round a Fraction to FP16 bits, nearest even, subnormals kept, overflow to infinity; sign of zero kept."""
    if fr == 0:
        return 0
    s = 0x8000 if fr < 0 else 0
    a = abs(fr)
    if a >= Fraction(65520):
        return s | 0x7c00
    # exponent: 2^e <= a < 2^(e+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -14)                     # subnormal range uses the exponent -14 grid
    q = a / (Fraction(2) ** (e - 10))   # in units of the ulp
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n >= 2048:                       # carried into the next binade
        n //= 2
        e += 1
        if e > 15:
            return s | 0x7c00
    if n < 1024:                        # subnormal
        return s | n
    return s | ((e + 15) << 10) | (n - 1024)


def h16_value(b):
    """The exact value of FP16 bits b (None for infinity / NaN)."""
    if (b & 0x7c00) == 0x7c00:
        return None
    return Fraction(float(np.frombuffer(struct.pack("<H", b), np.float16)[0]))


def midpoint_xs(K, count, seed):
    """x (FP32) with RN32(K x) an FP16 midpoint m and K x != m, residual signs alternating."""
    rng = np.random.default_rng(seed)
    out, want_up = [], True
    while len(out) < count:
        h = int(rng.integers(0x0001, 0x4400))          # FP16 values up to ~4 (products of the encode)
        if rng.random() < 0.5:
            h |= 0x8000
        v = h16_value(h)
        if v is None:
            continue
        ulp = Fraction(1, 2 ** 24) if (h & 0x7c00) == 0 else Fraction(2) ** (((h >> 10) & 31) - 25)
        m = v + (ulp / 2 if v >= 0 else -ulp / 2)
        x0 = rn32(m / K)
        for dx in (0, 1, -1, 2, -2):
            x = f32((bits32(x0) + dx) & 0xffffffff)
            p = Fraction(x) * K
            if rn32(p) == np.float32(float(m)) and p != m and ((p > m) == want_up):
                out.append(x)
                want_up = not want_up
                break
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fold_probe.h")
    K = Fraction(f32(K_BITS))
    xs = midpoint_xs(K, 60, 20261115)
    nan = f32(0x7fc00000)
    floor = float(np.log2(np.float32(1e-4)).astype(np.float32))
    xs += [0.0, -0.0, nan, floor, 0.0, 1.0, -1.0, 20.0, float(np.float32(3.0e-4)), float(np.float32(-2.0e-4)),
           float(np.float32(1e-6)), -float(np.float32(1e-6))]
    rng = np.random.default_rng(7)
    while len(xs) < 96:
        xs.append(float(np.float32(rng.uniform(-13.3, 21.0))))
    ds = []
    for h in rng.integers(0x0400, 0x7bff, 40):
        v = h16_value(int(h))
        ulp = Fraction(2) ** ((((int(h) >> 10) & 31)) - 25)
        m = v + ulp / 2
        ds.append(float(np.float32(float(m))))                                  # an exact FP16 midpoint (ties to even)
        ds.append(f32(bits32(float(m)) + 1))                                    # just above
    ds += [0.0, -0.0, nan, 65519.0, 65520.0, 1e6, f32(0x33800000), f32(0x33000001)]
    while len(ds) < 96:
        ds.append(float(np.float32(rng.uniform(0.0, 8.0))))
    xs, ds = xs[:96], ds[:96]
    inputs, expect, nanmask = [], [], []
    for lane in range(32):
        x = xs[3 * lane:3 * lane + 3]
        d = ds[3 * lane:3 * lane + 3]
        inputs += [bits32(v) for v in x + d]
        q, nm = [], []
        for v in x:
            if v != v:
                q.append(0x7e00); nm.append(True)
            else:
                p = Fraction(v) * K
                q.append(0 if p == 0 else rn16_bits(p)); nm.append(False)
        for v in d:
            if v != v:
                q.append(0x7e00); nm.append(True)
            elif v == 0:   # a plain conversion keeps the sign of a zero
                q.append(0x8000 if struct.pack("<f", v)[3] & 0x80 else 0); nm.append(False)
            else:
                q.append(rn16_bits(Fraction(v))); nm.append(False)
        expect += [q[0] | (q[1] << 16), q[2] | (q[3] << 16), q[4] | (q[5] << 16)]
        nanmask += [(1 if nm[0] else 0) | (2 if nm[1] else 0), (1 if nm[2] else 0) | (2 if nm[3] else 0),
                    (1 if nm[4] else 0) | (2 if nm[5] else 0)]
    with open(out, "w", newline="\n") as f:
        f.write("// SPDX-License-Identifier: MIT\n// Generated by clean/runtime/gen_fold_probe.py - do not edit.\n")
        f.write("// Start-up probe of the driver's conversion fold (clean/dx12/kernels/l_fold_probe.hlsl): inputs (32 lanes x\n"
                "// 6 FP32 bits) and expected output dwords (32 x 3); nan_mask bit 0 / 1 = the low / high FP16 half must be a\n"
                "// NaN (any sign and payload: the accepted port exception), the other halves must match bit for bit.\n")
        f.write("#pragma once\n#include <cstdint>\n#include <cstring>\n\nnamespace fold_probe {\n")
        f.write("constexpr unsigned kInBytes = 768, kOutOffset = 768, kOutBytes = 384, kBytes = 1152;\n")
        f.write("constexpr uint32_t kInput[192] = {\n")
        for i in range(0, 192, 6):
            f.write("    " + ", ".join(f"0x{v:08x}u" for v in inputs[i:i + 6]) + ",\n")
        f.write("};\nconstexpr uint32_t kExpect[96] = {\n")
        for i in range(0, 96, 6):
            f.write("    " + ", ".join(f"0x{v:08x}u" for v in expect[i:i + 6]) + ",\n")
        f.write("};\nconstexpr uint8_t kNanMask[96] = {\n")
        for i in range(0, 96, 24):
            f.write("    " + ", ".join(str(v) for v in nanmask[i:i + 24]) + ",\n")
        f.write("};\n")
        f.write("""inline void fill(uint8_t* buffer) {
  std::memset(buffer, 0, kBytes);
  std::memcpy(buffer, kInput, sizeof kInput);
}
inline bool half_ok(uint32_t got, uint32_t want, bool nan) {
  if (nan) return (got & 0x7c00u) == 0x7c00u && (got & 0x03ffu) != 0u;
  return got == want;
}
// index of the first differing output half (0 .. 191), or -1 when every half matches
inline int first_mismatch(const uint32_t* out) {
  for (unsigned i = 0; i < 96; ++i) {
    if (!half_ok(out[i] & 0xffffu, kExpect[i] & 0xffffu, (kNanMask[i] & 1u) != 0)) return (int)(2 * i);
    if (!half_ok(out[i] >> 16, kExpect[i] >> 16, (kNanMask[i] & 2u) != 0)) return (int)(2 * i + 1);
  }
  return -1;
}
}  // namespace fold_probe
""")
    # self-check of rn16_bits against numpy's double -> half conversion (correctly rounded from a double; every value
    # here is an exact double)
    chk = np.random.default_rng(3)
    for v in list(chk.uniform(-70000, 70000, 2000)) + list(chk.uniform(-1e-4, 1e-4, 2000)):
        nb = int(np.array([v]).astype(np.float16).view(np.uint16)[0])
        assert rn16_bits(Fraction(float(v))) == nb or (v == 0), (v, nb)
    print("wrote", out, "midpoint x:", sum(1 for v in xs[:60]), "lanes 32")


if __name__ == "__main__":
    main()
