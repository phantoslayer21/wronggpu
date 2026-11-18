# SPDX-License-Identifier: MIT
"""Exhaustive GPU check of the gfx11 FP32 -> E4M3 conversions of kernels/wg_platform.h over all 2^32 FP32 bit patterns.

    python clean/runner/q8_exhaustive.py --build <gfx11 build with probe.co> --rocm <ROCm> [--chunks 0:256]

A chunk is the 2^24 patterns that share their top byte (sign and the top 7 exponent bits); the kernel probe_q8_sweep
(kernels/probe.hip) converts one chunk per launch with the functions the kernels use, and the result is compared with a
NumPy reference written from the definition of the format (round to nearest even on the E4M3 grid, saturating at +-448,
NaN -> sign|0x7f, FP32 denormals -> a signed zero), never from the kernel code. A chunk takes seconds, the whole range
about twenty minutes; GPU buffers stay under 200 MB.

What each function must give (NaN inputs and inputs beyond +-464 are outside the contract of the `nc` forms):
  wg_f32_to_e4m3, wg_pack_q8   the code byte of the definition
  wg_pack_q8_inner             the same, except that a NaN gives sign|0x7e
  wg_pack_q8_nc                the same for |x| <= 464 (not NaN)
  wg_q8_to_f16                 the F16 bits of the E4M3 value (the sign of a zero kept; any F16 NaN for a NaN)
  wg_q8_to_f16_inner           the same, except that a NaN gives +-0
  wg_q8_to_f16_nc              the same for |x| <= 464 (not NaN)
Exit status 0 if there is no mismatch.
"""
import argparse
import ctypes
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CHUNK = 1 << 24          # patterns per launch
GROUP = 8                # values per lane (one v8f)
BLOCK = 256
LIMIT_NC = np.float32(464.0)   # the |x| bound of the `nc` forms


def positive_values():
    """Values of the positive E4M3 codes 0..0x7e, from the format: bias 7, 3 mantissa bits, subnormal step 2^-9."""
    c = np.arange(0x7F)
    e, m = c >> 3, c & 7
    return np.where(e == 0, m * 2.0 ** -9, (1.0 + m / 8.0) * np.exp2(e.astype(np.float64) - 7.0))


VALUES = positive_values()


def reference_codes(bits):
    """E4M3 code (uint8) of each FP32 bit pattern, by the definition: round |x| to a multiple of the spacing of its
    binade (at least 2^-9) with ties to even, saturate at 448, keep the sign; NaN -> sign|0x7f."""
    bits = np.asarray(bits, np.uint32)
    with np.errstate(all="ignore"):   # a signalling NaN raises the invalid flag when widened
        x = bits.view(np.float32).astype(np.float64)
        a = np.abs(x)
        _, e = np.frexp(a)                               # a = m 2^e, m in [0.5, 1); 0 for zero and infinity
        spacing = np.ldexp(1.0, np.maximum(e - 1, -6) - 3)
        r = np.where(np.isinf(a), 448.0, np.rint(a / spacing) * spacing)   # a / spacing is exact in double
    r = np.minimum(np.where(np.isnan(a), 0.0, r), 448.0)
    code = np.searchsorted(VALUES, r).astype(np.uint8)  # r is an E4M3 value: exact position
    sign = ((bits >> 24) & 0x80).astype(np.uint8)
    return np.where(np.isnan(x), 0x7F, code) | sign


def _half_table():
    """F16 bits of every E4M3 code (exact; the NaN code gives the quiet F16 NaN)."""
    table = np.zeros(256, np.uint16)
    for c in range(256):
        h = np.float16(VALUES[c & 0x7F]).view(np.uint16) if (c & 0x7F) != 0x7F else 0x7E00
        table[c] = h | (0x8000 if c & 0x80 else 0)
    return table


HALVES = _half_table()


def reference_halves(code):
    """F16 bits of the E4M3 codes."""
    return HALVES[code]


def is_f16_nan(h):
    return ((h & 0x7C00) == 0x7C00) & ((h & 0x03FF) != 0)


FORMS = ("wg_f32_to_e4m3", "wg_pack_q8", "wg_pack_q8_inner", "wg_pack_q8_nc", "wg_q8_to_f16", "wg_q8_to_f16_inner",
         "wg_q8_to_f16_nc")


def mismatches(bits, got):
    """{form: boolean array of the patterns that differ from the contract} for the outputs of one chunk."""
    code = reference_codes(bits)
    sign = (bits >> 24).astype(np.uint8) & 0x80
    x = bits.view(np.float32)
    nan = np.isnan(x)
    in_domain = np.abs(x) <= LIMIT_NC                      # False for NaN
    half = reference_halves(code)
    inner = np.where(nan, 0x7E | sign, code)
    bad = {"wg_f32_to_e4m3": got["wg_f32_to_e4m3"] != code, "wg_pack_q8": got["wg_pack_q8"] != code,
           "wg_pack_q8_inner": got["wg_pack_q8_inner"] != inner,
           "wg_pack_q8_nc": in_domain & (got["wg_pack_q8_nc"] != code)}
    h = got["wg_q8_to_f16"]
    bad["wg_q8_to_f16"] = np.where(nan, ~is_f16_nan(h), h != half)
    h = got["wg_q8_to_f16_inner"]
    bad["wg_q8_to_f16_inner"] = np.where(nan, (h & 0x7FFF) != 0, h != half)
    bad["wg_q8_to_f16_nc"] = in_domain & (got["wg_q8_to_f16_nc"] != half)
    return bad


def classify(bits):
    a = bits & 0x7FFFFFFF
    return np.select([a == 0, a < 0x00800000, a < 0x3C800000, a < 0x43E00000, a < 0x7F800000, a == 0x7F800000],
                     [0, 1, 2, 3, 4, 5], 6)


CLASSES = ("zero", "fp32 denormal", "below 2^-6", "2^-6..448", "448..inf (finite)", "inf", "NaN")


class Sweep:
    def __init__(self, build_dir, rocm_dir):
        from clean.runner.hip import Hip
        self.dev = Hip(rocm_dir)
        with open(os.path.join(build_dir, "probe.co"), "rb") as f:
            self.func = self.dev.get_function(self.dev.load_module(f.read()), "probe_q8_sweep")
        n = CHUNK
        self.sizes = [n, n, n, n, 2 * n, 2 * n, 2 * n]       # bytes per output: 1 per value, 2 for the F16 forms
        self.bufs = [self.dev.malloc(s) for s in self.sizes]

    def close(self):
        for p in self.bufs:
            self.dev.free(p)

    def chunk(self, index):
        """Outputs of the functions for the patterns index << 24 ... : {form: uint8 or uint16 array}."""
        base = index * CHUNK
        for p, s in zip(self.bufs, self.sizes):
            self.dev.memset(p, 0x5A, s)   # an unwritten output shows as a mismatch
        args = [ctypes.c_uint(base)] + [ctypes.c_void_p(p) for p in self.bufs]
        self.dev.launch(self.func, CHUNK // GROUP // BLOCK, BLOCK, args)
        self.dev.sync()
        out = {}
        for form, p, s in zip(FORMS, self.bufs, self.sizes):
            raw = self.dev.download(p, s)
            out[form] = np.frombuffer(raw, np.uint8 if s == CHUNK else np.uint16)
        return out


def check(build_dir, rocm_dir, chunks, log=print):
    """Run the chunks; returns {form: {class: mismatch count}}, first mismatching patterns, and the pattern count."""
    sweep = Sweep(build_dir, rocm_dir)
    total = {f: np.zeros(len(CLASSES), np.int64) for f in FORMS}
    first = {}
    done = 0
    try:
        for index in chunks:
            t0 = time.time()
            bits = (np.uint64(index) * np.uint64(CHUNK) + np.arange(CHUNK, dtype=np.uint64)).astype(np.uint32)
            bad = mismatches(bits, sweep.chunk(index))
            cls = classify(bits)
            line = []
            for form, b in bad.items():
                if b.any():
                    total[form] += np.bincount(cls[b], minlength=len(CLASSES))
                    first.setdefault(form, int(bits[np.argmax(b)]))
                    line.append(f"{form} {int(b.sum())}")
            done += CHUNK
            log(f"chunk 0x{index:02x} {time.time() - t0:5.1f} s  " + ("; ".join(line) or "no mismatch"))
    finally:
        sweep.close()
    return total, first, done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="a gfx11 build folder with probe.co")
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--chunks", default="0:256", help="first:last+1 of the chunk indices (top byte of the pattern)")
    args = ap.parse_args(argv)
    lo, hi = (int(v, 0) for v in args.chunks.split(":"))
    total, first, done = check(args.build, args.rocm, range(lo, hi))
    bad = 0
    print(f"patterns checked: {done}")
    for form in FORMS:
        n = int(total[form].sum())
        bad += n
        detail = {c: int(v) for c, v in zip(CLASSES, total[form]) if v}
        print(f"{form:22s} mismatches {n}" + (f"  {detail}  first 0x{first[form]:08x}" if n else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
