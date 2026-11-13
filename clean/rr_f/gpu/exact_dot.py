# SPDX-License-Identifier: MIT
"""Original standalone exact spec-51.1 dot32 D3D12 baseline and synthetic qualification.

Run with a private WG_RR_BUILD and --out. Every GPU
run acquires the shared GPU lock and requires D3D12 binary64 support. No learned
inputs, matrix hardware claim, integration claim, or performance claim is made.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from clean.rr_f.gpu import build
from clean.rr_f.reference import fmt


def _codes(a, name):
    a = np.asarray(a)
    if a.dtype != np.uint8 or a.ndim != 2 or a.shape[1] != 32:
        raise ValueError(f"{name} must be uint8 [N,32]")
    return a


def dot32(a, b, c_bits, workdir=None):
    """Return uint16 [N] spec51 dot results and host log; c_bits is uint16 [N]."""
    a, b = _codes(a, "A"), _codes(b, "B")
    c_bits = np.asarray(c_bits)
    if b.shape != a.shape or c_bits.dtype != np.uint16 or c_bits.shape != (len(a),):
        raise ValueError("B must match A; C must be uint16 [N]")
    if not len(a):
        raise ValueError("at least one dot is required")
    inp = np.zeros((len(a), 68), np.uint8)
    inp[:, :32], inp[:, 32:64] = a, b
    inp[:, 64:68] = c_bits.astype(np.uint32).view(np.uint8).reshape(-1, 4)
    return _dispatch(inp, len(a), "rr_exact_dot32", workdir)


def round64(values, workdir=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a nonempty one-dimensional array")
    return _dispatch(values, len(values), "rr_exact_round64", workdir)


def _dispatch(inp, n, entry, workdir):
    ar = build.Arena()
    pin = ar.add(inp)
    pout = ar.reserve(n * 4)
    dx = build.compile_kernel("rr_exact_dot.hlsl", entry, {})
    img, log = build.run(ar, {0: (dx, 3)}, [(0, (n + 127) // 128, [pin, pout, n])],
                         lock=True, require_double=True, workdir=workdir)
    return img[pout:pout + n * 4].view(np.uint32).astype(np.uint16), log


def oracle(a, b, c):
    # Exact binary64 sum; the integer oracle below independently checks a subset.
    with np.errstate(invalid="ignore"):
        part = np.sum(fmt.d8(a) * fmt.d8(b), axis=1) + 0.0
        return fmt.to_f16_bits(fmt.round_f16(fmt.from_f16_bits(c) + part))


def integer_oracle(a, b, c):
    """Independent Python unbounded-integer oracle: all finite values scaled by 2^24."""
    c = int(c)
    def scaled8(code):
        code = int(code)
        e, m = (code >> 3) & 15, code & 7
        v = m if e == 0 else (8 + m) << (e - 1)
        return -v if code & 128 else v
    if (c & 0x7fff) > 0x7c00 or any((int(v) & 127) == 127 for v in (*a, *b)):
        return 0x7e00
    if (c & 0x7fff) == 0x7c00:
        return c
    ce, cm = (c >> 10) & 31, c & 1023
    ci = cm if ce == 0 else (1024 + cm) << (ce - 1)
    total = (-ci if c & 0x8000 else ci) + sum(scaled8(x) * scaled8(y) * 64 for x, y in zip(a, b))
    sign = 0x8000 if total < 0 else 0
    mag = abs(total)
    shift = max(0, mag.bit_length() - 11)
    q = mag >> shift
    if shift:
        rem = mag & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        q += rem > halfway or (rem == halfway and q & 1)
    rounded = q << shift
    if rounded >= 65536 << 24:
        return sign | 0x7c00
    if rounded < 1024:
        return sign | rounded
    exponent = rounded.bit_length() - 11
    return sign | ((exponent + 1) << 10) | ((rounded >> exponent) - 1024)


def same_bits(actual, expected):
    nan = ((actual & 0x7fff) > 0x7c00) & ((expected & 0x7fff) > 0x7c00)
    return (actual == expected) | nan


def cases(seed=51, random_count=65536):
    """Deterministic synthetic finite and nonfinite cases, with stable family boundaries."""
    rng = np.random.default_rng(seed)
    data, families = [], []
    def add(name, a, b, c):
        n = len(c)
        data.append((np.asarray(a, np.uint8), np.asarray(b, np.uint8), np.asarray(c, np.uint16)))
        families.append((name, n))
    pair_a = np.repeat(np.arange(256, dtype=np.uint8), 256)
    pair_b = np.tile(np.arange(256, dtype=np.uint8), 256)
    for cb in (0, 1, 0x8001):
        a = np.zeros((65536, 32), np.uint8)
        b = a.copy()
        a[:, 0], b[:, 0] = pair_a, pair_b
        add(f"all_65536_code_pairs_bias_{cb:04x}", a, b, np.full(65536, cb, np.uint16))
    a = np.zeros((65536, 32), np.uint8)
    add("all_f16_start_patterns_zero_dot", a, a.copy(), np.arange(65536, dtype=np.uint16))
    finite = np.array([v for v in range(256) if v & 127 != 127], np.uint8)
    a = rng.choice(finite, (random_count, 32))
    b = rng.choice(finite, (random_count, 32))
    c = rng.integers(0, 65536, random_count, dtype=np.uint16)
    add("random_finite_codes_all_bias_classes", a, b, c)
    n = 32768
    a = rng.choice(finite, (n, 32))
    b = rng.choice(finite, (n, 32))
    a[:, 16:] = a[:, :16]
    b[:, 16:] = b[:, :16] ^ 128
    add("exact_cancellation_all_bias_classes", a, b, rng.integers(0, 65536, n, dtype=np.uint16))
    # 1 + 2^-11 +/- 2^-24, at even/odd half midpoint, both signs.
    a = np.zeros((12, 32), np.uint8)
    b = a.copy()
    a[:, 0], b[:, 0] = 0x38, 0x38
    a[:, 1], b[:, 1] = 0x10, 0x08
    a[6:, 2], b[6:, 2] = 0x10, 0x10
    c = np.tile(np.array([0, 1, 0x8001], np.uint16), 4)
    a[3:6] ^= 128
    a[9:12] ^= 128
    add("double_round_midpoint_adversaries", a, b, c)
    # Large cancelling partial sums around tiny residuals and half midpoints.
    # Each row permutes 14 cancelling pairs plus three small terms and one zero.
    n = 8192
    a = np.zeros((n, 32), np.uint8)
    b = a.copy()
    a[:, :14] = rng.choice(finite, (n, 14))
    b[:, :14] = rng.choice(finite, (n, 14))
    a[:, 14:28] = a[:, :14]
    b[:, 14:28] = b[:, :14] ^ 128
    a[:, 28:30], b[:, 28:30] = [0x38, 0x10], [0x38, 0x08]
    a[n//2:, 30], b[n//2:, 30] = 0x10, 0x10
    order = np.argsort(rng.random((n, 32)), axis=1)
    a, b = np.take_along_axis(a, order, 1), np.take_along_axis(b, order, 1)
    a[::2] ^= 128
    c = np.resize(np.array([0, 1, 0x8001], np.uint16), n)
    add("permuted_large_cancellation_at_half_midpoints", a, b, c)
    a = np.full((16, 32), 0x7e, np.uint8)
    b = a.copy()
    b[8:] ^= 128
    f16 = lambda sign, exponent, fraction=0: (sign << 15) | (exponent << 10) | fraction   # binary16 bits from fields
    c_cases = [f16(0, 0), f16(0, 0, 1), f16(0, 30, 1023), f16(1, 0, 1),   # 0, smallest subnormal, max, -smallest
               f16(1, 30, 1023), f16(0, 31), f16(1, 31), f16(0, 31, 512)]   # -max, +-infinity, quiet NaN
    add("maximum_magnitude", a, b, np.tile(np.array(c_cases, np.uint16), 2))
    # Each sign of NaN in each slot of each input, with finite/infinite C.
    a = np.zeros((256, 32), np.uint8)
    b = a.copy()
    c = np.zeros(256, np.uint16)
    for i in range(256):
        (a if i < 128 else b)[i, i % 32] = 127 | ((i // 32 % 2) << 7)
        c[i] = 0x7c00 if i // 64 % 2 else 0
    add("nan_codes_all_slots", a, b, c)
    return tuple(np.concatenate([d[i] for d in data]) for i in range(3)), families


def rounding_cases():
    # All adjacent nonnegative finite f16 pairs plus the overflow threshold;
    # nextafter64 distinguishes each exact tie from both immediately adjacent doubles.
    h = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    mids = np.concatenate(((h[:-1] + h[1:]) * 0.5, [65520.0]))
    near = np.stack((np.nextafter(mids, -np.inf), mids, np.nextafter(mids, np.inf)), axis=1).ravel()
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan,
                        np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
                        np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0), 1e300, -1e300])
    return np.concatenate((near, -near, fmt.ALL_F16, special))


def run_suite(out, seed=51, random_count=65536):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (a, b, c), families = cases(seed, random_count)
    expected = oracle(a, b, c)
    # Independent integer checks include all code-pair combinations at zero bias,
    # nonfinite start patterns, plus representative random/cancellation/adversaries.
    indices = np.unique(np.concatenate((np.arange(65536), np.arange(len(c) - 512, len(c)),
                                       np.random.default_rng(seed + 1).integers(0, len(c), 4096))))
    integers = np.array([integer_oracle(a[i], b[i], c[i]) for i in indices], np.uint16)
    if not np.all(same_bits(integers, expected[indices])):
        raise AssertionError("integer and binary64 CPU oracles disagree")
    actual, dotlog = dot32(a, b, c, str(out / "dot-run"))
    (out / "dot-host.log").write_text(dotlog, encoding="utf-8")
    summaries, start = [], 0
    for name, n in families:
        ok = same_bits(actual[start:start+n], expected[start:start+n])
        summaries.append({"family": name, "count": n, "mismatches": int(np.count_nonzero(~ok))})
        start += n
    values = rounding_cases()
    with np.errstate(over="ignore", invalid="ignore"):
        rounded_expected = values.astype(np.float16).view(np.uint16)
    rounded, roundlog = round64(values, str(out / "round-run"))
    (out / "round-host.log").write_text(roundlog, encoding="utf-8")
    round_ok = same_bits(rounded, rounded_expected)
    dot_ok = same_bits(actual, expected)
    # Report an observed double-round counterexample count as evidence the direct converter matters.
    with np.errstate(over="ignore", invalid="ignore"):
        sum64 = np.sum(fmt.d8(a) * fmt.d8(b), axis=1) + fmt.from_f16_bits(c)
        narrowed = sum64.astype(np.float32).astype(np.float16).view(np.uint16)
    double_round_mismatches = int(np.count_nonzero(~same_bits(narrowed, expected)))
    for name, arr in (("dot-actual", actual), ("dot-expected", expected),
                      ("round-actual", rounded), ("round-expected", rounded_expected)):
        arr.tofile(out / (name + ".u16"))
    report = {"seed": seed, "dot_count": len(c), "families": summaries,
              "integer_oracle_count": len(indices), "integer_oracle_mismatches": 0,
              "rounding_count": len(values), "rounding_mismatches": int(np.count_nonzero(~round_ok)),
              "dot_mismatches": int(np.count_nonzero(~dot_ok)),
              "double_to_float_to_half_dot_mismatches": double_round_mismatches,
              "first_dot_failures": np.flatnonzero(~dot_ok)[:16].tolist(),
              "first_rounding_failures": np.flatnonzero(~round_ok)[:16].tolist(),
              "input_sha256": hashlib.sha256(a.tobytes() + b.tobytes() + c.tobytes()).hexdigest(),
              "round_input_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
              "exact_scaled_bound": (32 * 448**2 + 65504) * 2**24,
              "limitations": "Standalone synthetic dot32 only; no full network, learned data, WMMA correction or performance qualification."}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return not (report["dot_mismatches"] or report["rounding_mismatches"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--random-count", type=int, default=65536)
    args = p.parse_args()
    raise SystemExit(0 if run_suite(args.out, args.seed, args.random_count) else 1)


if __name__ == "__main__":
    main()
