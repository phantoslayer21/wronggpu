# SPDX-License-Identifier: MIT
"""GPU self-test with synthetic data: checks the hardware facts the Swin kernels rely on.

    python clean/runner/selftest.py --build <build_dir> --rocm <ROCm>

Prints a short report (numbers and pass/fail only). What it checks:
  1. WMMA FP8 and FP16 register layouts, inferred with one-hot operands:
     - A operand: lane l holds row l % 16; B operand: lane l holds column l % 16;
     - A and B use the same K index for the same (lane half, element) slot (any permutation is fine);
     - accumulator: element v of lane half h holds row D_ROW[h][v] (the weight packing adapts to any permutation;
       the inferred table can be passed to Runner(d_row=...) or the WG_D_ROW environment variable).
  2. FP8 inputs of WMMA are E4M3FN (1.0 encoded as 0x38).
  3. The kernels' FP8 conversions equal the reference q8 and the E4M3FN decoding (bit-exact): the software
     functions, and the pack/unpack functions the kernels use (hardware instructions unless built with --soft-fp8)
     in every byte position; the inner pack (no NaN pass-through) must give identical bytes for all non-NaN values.
  3b. The kernels' GELU (default: fitted sigmoid form): equal to its own formula within 1e-4, within 1e-3 of the exact
      GELU, and no result more than one FP8 step away from the exact GELU after FP8 rounding.
  4. Informational: hardware FP8 conversions (v_cvt_pk_fp8_f32 / v_cvt_f32_fp8) against the reference.
  5. If the second design was built (build.py --variants v2 or all: swin__v2.co), its conversion helpers
     (probe_v2_conv): packed FP8 decoding (v_cvt_pk_f32_fp8) in every byte position, the FP8 packing with full and
     upper-only clamp, its FP16 conversion (v_fma_mixlo/hi_f16, must equal round-to-nearest-even), and its GELU + q8
     (no result more than one FP8 step from the exact GELU).
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ctypes  # noqa: E402

import numpy as np  # noqa: E402

from clean.reference.fp8 import DECODE, fp8_encode, q8  # noqa: E402
from clean.runner.packing import D_ROW_LAYOUTS  # noqa: E402

BLOCK = 128
FP8_ONE = 0x38
F16_ONE = 0x3C00


class _Dev:
    def __init__(self, backend, build_dir, rocm_dir, code_object="probe.co", emu_library="wg_emu.so", dev=None):
        if backend == "hip":
            from clean.runner.hip import Hip
            self.dev = dev or Hip(rocm_dir)
            with open(os.path.join(build_dir, code_object), "rb") as f:
                self.module = self.dev.load_module(f.read())
        else:
            from clean.runner.emu import Emu
            self.dev = Emu(os.path.join(build_dir, emu_library))
            self.module = None

    def call(self, kernel, inputs, outputs, scalars=(), threads=None):
        """inputs: list of bytes; outputs: list of byte sizes. Returns list of output bytes."""
        d = self.dev
        bufs = []
        try:
            for data in inputs:
                p = d.malloc(len(data))
                d.upload(p, data)
                bufs.append(p)
            for n in outputs:
                p = d.malloc(n)
                d.memset(p, 0, n)
                bufs.append(p)
            args = [ctypes.c_void_p(p) for p in bufs] + [ctypes.c_int(s) for s in scalars]
            d.launch(d.get_function(self.module, kernel), -(-threads // BLOCK), BLOCK, args)
            d.sync()
            return [d.download(p, n) for p, n in zip(bufs[len(inputs):], outputs)]
        finally:
            for p in bufs:
                d.free(p)


def _run_wmma(dev, kind, a, b):
    """a, b: [probes, 32 lanes, 8 elements] (uint8 codes for fp8, uint16 bit patterns for f16) -> D [probes, 32, 8]."""
    n = a.shape[0]
    pad = (-n) % (BLOCK // 32)
    if pad:
        a = np.concatenate([a, np.zeros((pad,) + a.shape[1:], a.dtype)])
        b = np.concatenate([b, np.zeros((pad,) + b.shape[1:], b.dtype)])
    kernel = "probe_wmma_fp8" if kind == "fp8" else "probe_wmma_f16"
    (d,) = dev.call(kernel, [a.tobytes(), b.tobytes()], [a.shape[0] * 32 * 32], threads=a.shape[0] * 32)
    return np.frombuffer(d, dtype=np.float32).reshape(-1, 32, 8)[:n]


def probe_wmma(dev, kind):
    dt = np.uint8 if kind == "fp8" else np.uint16
    one = FP8_ONE if kind == "fp8" else F16_ONE
    slots = [(L, J) for L in range(32) for J in range(8)]
    ones = np.full((32, 8), one, dt)

    # rows of A: one-hot A x all-ones B -> row(L) lights up in every column
    a = np.zeros((len(slots), 32, 8), dt)
    for i, (L, J) in enumerate(slots):
        a[i, L, J] = one
    d_rows = _run_wmma(dev, kind, a, np.broadcast_to(ones, a.shape).copy())
    # columns of B: all-ones A x one-hot B -> column(L) lights up for every row
    d_cols = _run_wmma(dev, kind, np.broadcast_to(ones, a.shape).copy(), a)
    # K pairing: one-hot A x B whose slot (h, j) holds a distinct value 1 + q/16 (q = 8h + j)
    if kind == "fp8":
        # slot (h, j) -> code 0x38 + j (1.0 .. 1.875) for h = 0, 0x40 + j (2.0 .. 3.75) for h = 1
        tag = np.array([[0x38 + j for j in range(8)], [0x40 + j for j in range(8)]], dt)
        values = {float(DECODE[c]): (h, j) for h in range(2) for j, c in enumerate(tag[h])}
    else:
        vals = [[1.0 + (8 * h + j) / 16.0 for j in range(8)] for h in range(2)]
        tag = np.array([[np.float16(v).view(np.uint16) for v in row] for row in vals], dt)
        values = {vals[h][j]: (h, j) for h in range(2) for j in range(8)}
    b = np.zeros((32, 8), dt)
    for L in range(32):
        b[L] = tag[L // 16]
    d_pair = _run_wmma(dev, kind, a, np.broadcast_to(b, a.shape).copy())

    res = {"kind": kind}
    lit_value = float(np.max(d_rows))
    res["one_times_one"] = lit_value
    arow = {}   # lane L -> (h, v) positions holding its row, per column n
    a_ok = True
    for i, (L, J) in enumerate(slots):
        nz = np.argwhere(d_rows[i] != 0)                # (lane, v)
        cols = sorted(set(int(l) % 16 for l, _ in nz))
        hv = sorted(set((int(l) // 16, int(v)) for l, v in nz))
        if cols != list(range(16)) or len(hv) != 1 or len(nz) != 16:
            a_ok = False
            continue
        arow.setdefault(L % 16, set()).add(hv[0])
    rows_ok = a_ok and len(arow) == 16 and all(len(s) == 1 for s in arow.values())
    d_row = None
    if rows_ok:
        table = [[None] * 8 for _ in range(2)]
        for r, s in arow.items():
            h, v = next(iter(s))
            table[h][v] = r
        if all(x is not None for row in table for x in row):
            d_row = table
    res["a_row_is_lane_mod_16"] = bool(rows_ok and d_row is not None)
    res["d_row"] = d_row
    res["d_row_name"] = next((k for k, t in D_ROW_LAYOUTS.items() if d_row is not None and t.tolist() == d_row), None)

    b_ok = True
    for i, (L, J) in enumerate(slots):
        nz = np.argwhere(d_cols[i] != 0)
        lanes = sorted(set(int(l) for l, _ in nz))
        if lanes != [L % 16, L % 16 + 16] or len(nz) != 16:
            b_ok = False
    res["b_col_is_lane_mod_16"] = b_ok

    sym = True
    if rows_ok and d_row is not None:
        pos = {r: hv for r, s in arow.items() for hv in s}
        for i, (L, J) in enumerate(slots):
            h, v = pos[L % 16]
            got = float(d_pair[i, h * 16 + 3, v])        # any column; take n = 3
            if values.get(got) != (L // 16, J):
                sym = False
    else:
        sym = False
    res["k_slots_symmetric"] = sym
    res["pass"] = bool(res["a_row_is_lane_mod_16"] and b_ok and sym and lit_value == 1.0)
    return res


def probe_fp8(dev, backend):
    edge = [0.0, -0.0, 448.0, 449.0, 463.9, 464.0, 465.0, 500.0, 1e6, np.inf, -np.inf, np.nan, 2.0 ** -9,
            2.0 ** -10, 3 * 2.0 ** -11, 2.0 ** -6, 2.0 ** -6 * 1.0625, 2.0 ** -7 * 1.5, 1.0625, 1.1875, 240.0, 256.0]
    mids = [(DECODE[i] + DECODE[i + 1]) / 2 for i in range(0x7E)]
    rng = np.random.default_rng(0)
    xs = np.array(edge + mids + [-m for m in mids] + list(rng.standard_normal(4000) * 20), dtype=np.float32)
    n = len(xs)
    want = fp8_encode(xs.astype(np.float64))
    enc, dec = dev.call("probe_fp8_sw", [xs.tobytes()], [4 * n, 4 * 256], scalars=[n], threads=max(n, 256))
    enc = np.frombuffer(enc, np.uint32)
    dec = np.frombuffer(dec, np.float32)
    nan_codes = np.isnan(xs)
    sw_enc_mismatch = int(np.sum((enc != want) & ~nan_codes)) + int(np.sum(nan_codes & ((enc & 0x7F) != 0x7F)))
    ref_dec = DECODE.astype(np.float32)
    sw_dec_mismatch = int(np.sum(~((dec == ref_dec) | (np.isnan(dec) & np.isnan(ref_dec)))))
    res = {"sw_encode_mismatches": sw_enc_mismatch, "sw_decode_mismatches": sw_dec_mismatch, "values": n,
           "pass": sw_enc_mismatch == 0 and sw_dec_mismatch == 0}
    if backend == "hip":
        e, es, hd = dev.call("probe_fp8_hw", [xs.tobytes()], [4 * n, 4 * n, 4 * 256], scalars=[n],
                             threads=max(n, 256))
        e, es, hd = (np.frombuffer(e, np.uint32), np.frombuffer(es, np.uint32), np.frombuffer(hd, np.float32))
        finite = np.isfinite(xs)
        res["hw_decode_mismatches_vs_e4m3fn"] = int(np.sum(~((hd == ref_dec) | (np.isnan(hd) & np.isnan(ref_dec)))))
        res["hw_encode_mismatches"] = int(np.sum((e != want) & finite))
        res["hw_encode_clamped_mismatches"] = int(np.sum((es != want) & finite))
        bad = np.nonzero((es != want) & finite)[0][:8]
        res["hw_encode_clamped_examples"] = [[float(xs[i]), int(es[i]), int(want[i])] for i in bad]

    # The pack/unpack functions the Swin kernels actually use, every value in every byte position.
    m = n - n % 8
    xs8 = np.concatenate([xs[:m], np.roll(xs[:m], 3)]).reshape(-1, 8)
    want8 = fp8_encode(xs8.astype(np.float64))
    groups = 256
    codes = ((np.arange(groups)[:, None] + 37 * np.arange(8)[None, :]) % 256).astype(np.uint8)
    k = max(len(xs8), groups)
    xs8p = np.zeros((k, 8), np.float32)
    xs8p[:len(xs8)] = xs8
    codesp = np.zeros((k, 8), np.uint8)
    codesp[:groups] = codes
    packed, unpacked, inner = dev.call("probe_fp8_pack", [xs8p.tobytes(), codesp.tobytes()], [8 * k, 32 * k, 8 * k],
                                       scalars=[k], threads=k)
    packed = np.frombuffer(packed, np.uint8).reshape(k, 8)[:len(xs8)]
    inner = np.frombuffer(inner, np.uint8).reshape(k, 8)[:len(xs8)]
    unpacked = np.frombuffer(unpacked, np.float32).reshape(k, 8)[:groups]
    nan8 = np.isnan(xs8)
    pack_bad = int(np.sum((packed != want8) & ~nan8)) + int(np.sum(nan8 & ((packed & 0x7F) != 0x7F)))
    ref_unpack = ref_dec[codes]
    unpack_bad = int(np.sum(~((unpacked == ref_unpack) | (np.isnan(unpacked) & np.isnan(ref_unpack)))))
    # the inner pack (no NaN pass-through): identical bytes for every non-NaN input, including +-inf and out of range
    inner_bad = int(np.sum((inner != want8) & ~nan8))
    res["kernel_pack_inner_mismatches"] = inner_bad
    pack_bad += 0
    res["kernel_pack_mismatches"] = pack_bad
    res["kernel_unpack_mismatches"] = unpack_bad
    res["kernel_pack_values"] = int(xs8.size)
    res["pass"] = bool(res["pass"] and pack_bad == 0 and unpack_bad == 0 and inner_bad == 0)
    res.update(probe_gelu(dev))
    res["pass"] = bool(res["pass"] and res["gelu_max_abs_err_vs_formula"] <= 1e-4 and
                       res["gelu_max_abs_err_vs_exact"] <= 1e-3 and res["gelu_fp8_flips_over_1_step"] == 0)
    return res


def probe_gelu(dev):
    """The GELU of the kernels (the default sigmoid form, WG_GELU_MODE 2) against its own formula in float64, and
    against the exact GELU (erf) in FP32 and after FP8 rounding."""
    import math
    xs = np.concatenate([np.linspace(-14, 14, 20001), np.random.default_rng(1).standard_normal(8000) * 3]).astype(
        np.float32)
    n = len(xs)
    (y,) = dev.call("probe_gelu", [xs.tobytes()], [4 * n], scalars=[n], threads=n)
    y = np.frombuffer(y, np.float32).astype(np.float64)
    x64 = xs.astype(np.float64)
    exact = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x64])
    with np.errstate(over="ignore"):
        sig = x64 / (1.0 + np.exp(-x64 * (1.59989588569617 + 0.06951594437094213 * x64 * x64)))
    ca, cb = fp8_encode(y).astype(int), fp8_encode(exact).astype(int)
    oa, ob = np.where(ca & 0x80, -(ca & 0x7F), ca), np.where(cb & 0x80, -(cb & 0x7F), cb)
    steps = np.abs(oa - ob)
    return {"gelu_values": n, "gelu_max_abs_err_vs_formula": float(np.max(np.abs(y - sig) / (1.0 + np.abs(sig)))),
            "gelu_max_abs_err_vs_exact": float(np.max(np.abs(y - exact))),
            "gelu_fp8_differs_vs_exact": int(np.sum(steps > 0)), "gelu_fp8_flips_over_1_step": int(np.sum(steps > 1))}


def probe_v2(dev):
    """The conversion helpers of the second design (kernels/swin_v2.hip, probe_v2_conv): packed FP8 decoding
    (v_cvt_pk_f32_fp8) in every byte position, the clamped and upper-clamped FP8 packing, and its GELU + q8."""
    import math
    rng = np.random.default_rng(2)
    edge = np.array([0.0, -0.0, 448.0, 449.0, 463.9, 464.0, 500.0, 1e6, np.inf, -np.inf, 2.0 ** -9, 2.0 ** -10,
                     3 * 2.0 ** -11, 2.0 ** -6, 1.0625, 240.0, -0.17, -3.0, -20.0], np.float32)
    mids = np.array([(DECODE[i] + DECODE[i + 1]) / 2 for i in range(0x7E)], np.float32)
    xs = np.concatenate([edge, mids, -mids, rng.standard_normal(4000).astype(np.float32) * 4,
                         np.linspace(-12, 12, 4000, dtype=np.float32)])
    xs = xs[: len(xs) - len(xs) % 8].reshape(-1, 8)
    xs = np.concatenate([xs, np.roll(xs, 3, axis=1)])
    k = len(xs)
    codes = ((np.arange(k)[:, None] * 8 + 37 * np.arange(8)[None, :]) % 256).astype(np.uint8)
    unp, packed, upper, gel, half = dev.call("probe_v2_conv", [xs.tobytes(), codes.tobytes()],
                                             [32 * k, 8 * k, 8 * k, 8 * k, 16 * k], scalars=[k], threads=k)
    half = np.frombuffer(half, np.float16).reshape(k, 8)
    want_half = xs.astype(np.float16)  # round to nearest even
    res_half = int(np.sum(~((half == want_half) | (np.isnan(half) & np.isnan(want_half)))))
    unp = np.frombuffer(unp, np.float32).reshape(k, 8)
    packed, upper, gel = (np.frombuffer(b, np.uint8).reshape(k, 8) for b in (packed, upper, gel))
    ref_dec = DECODE.astype(np.float32)[codes]
    want = fp8_encode(xs.astype(np.float64))
    res = {"values": int(xs.size), "f16_mismatches": res_half}
    res["unpack_mismatches"] = int(np.sum(~((unp == ref_dec) | (np.isnan(unp) & np.isnan(ref_dec)))))
    res["pack_mismatches"] = int(np.sum(packed != want))
    ok_upper = xs >= -448
    res["pack_upper_mismatches"] = int(np.sum((upper != want) & ok_upper))
    exact = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) if np.isfinite(v) else 0.0
                      for v in xs.astype(np.float64).ravel()]).reshape(xs.shape)
    finite = np.isfinite(xs)
    ca, cb = gel.astype(int), fp8_encode(exact).astype(int)
    oa, ob = np.where(ca & 0x80, -(ca & 0x7F), ca), np.where(cb & 0x80, -(cb & 0x7F), cb)
    steps = np.abs(oa - ob)[finite]
    res["gelu_fp8_differs"] = int(np.sum(steps > 0))
    res["gelu_fp8_flips_over_1_step"] = int(np.sum(steps > 1))
    res["pass"] = bool(res["unpack_mismatches"] == 0 and res["pack_mismatches"] == 0 and res["f16_mismatches"] == 0 and
                       res["pack_upper_mismatches"] == 0 and res["gelu_fp8_flips_over_1_step"] == 0)
    return res


def selftest(build_dir, rocm_dir="", backend="hip"):
    dev = _Dev(backend, build_dir, rocm_dir)
    report = {"device": dev.dev.device_name}
    report["wmma_fp8"] = probe_wmma(dev, "fp8")
    report["wmma_f16"] = probe_wmma(dev, "f16")
    report["fp8_conversion"] = probe_fp8(dev, backend)
    d8, d16 = report["wmma_fp8"]["d_row"], report["wmma_f16"]["d_row"]
    report["d_row_consistent_fp8_f16"] = d8 is not None and d8 == d16
    report["pass"] = bool(report["wmma_fp8"]["pass"] and report["wmma_f16"]["pass"] and
                          report["fp8_conversion"]["pass"] and report["d_row_consistent_fp8_f16"])
    # Second design (kernels/swin_v2.hip), when it was built: its own conversion helpers.
    v2_file = "swin__v2.co" if backend == "hip" else "wg_emu__v2.so"
    if os.path.exists(os.path.join(build_dir, v2_file)):
        v2 = _Dev(backend, build_dir, rocm_dir, code_object=v2_file, emu_library=v2_file,
                  dev=dev.dev if backend == "hip" else None)
        report["v2_conversion"] = probe_v2(v2)
        report["pass"] = bool(report["pass"] and report["v2_conversion"]["pass"])
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--backend", default="hip", choices=["hip", "emu"])
    args = ap.parse_args(argv)
    report = selftest(args.build, args.rocm, args.backend)
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
