# SPDX-License-Identifier: MIT
"""Host proof of the GELU T5 table (hlsl/wg_gelu_t5.hlsli, runtime/gelu_t5_table.h) against the arena table T16
(runtime/gelu_lut_table.h): for every FP32 bit pattern w (all 2^32, NaN encodings excepted), the T5 entry j = w >> 18
is the T16 entry of slot f16_rtz(w) >> 5 that the arena-table path uses (same slot, so the same (a, b), the same fma and
the same FP8 byte). A NaN w only comes from a NaN h, whose fma(h, a, b) is NaN with any entry.

    python clean/dx12/gelu_lut/t5_check.py [--clang "C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe"]

f16_rtz here is independent of gelu_t5_table.h's bit formula: numpy's round-to-nearest-even FP16 cast, stepped one code
toward zero wherever its magnitude exceeds |w| (finite overflow -> 0x7bff, infinity stays infinity, FP32 denormals give
a signed zero). With --clang, the runtime's own builder (wg::gelu_t5_expand) is compiled and its 131072 bytes must equal
the table derived here. Exit status 0 only if every check passes.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'runtime')
CHUNK = 1 << 24


def t16_table():
    src = open(os.path.join(RUNTIME, 'gelu_lut_table.h')).read()
    runs = re.findall(r'\{(\d+), (\d+), (0x[0-9a-f]+)u, (0x[0-9a-f]+)u\}', src)
    t = np.zeros((2048, 2), np.uint32)
    for first, count, a, b in runs:
        t[int(first):int(first) + int(count)] = (int(a, 16), int(b, 16))
    assert sum(int(r[1]) for r in runs) == 2048, 'gelu_lut_table.h: runs do not cover 2048 slots'
    return t


def f16_rtz(u):
    """FP16 codes (uint32) of FP32 bit patterns u (uint32 array, no NaN), rounded toward zero."""
    x = u.view(np.float32)
    with np.errstate(over='ignore'):
        c = x.astype(np.float16).view(np.uint16).astype(np.uint32)
    back = c.astype(np.uint16).view(np.float16).astype(np.float32)
    step = np.abs(back) > np.abs(x)              # RNE rounded away from zero (or overflowed to infinity): one code back
    return np.where(step, c - 1, c)              # sign-magnitude code: magnitude - 1 keeps the sign bit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clang', help='clang++ to compile the runtime builder (runtime/gelu_t5_table.h) and compare')
    args = ap.parse_args()
    t16 = t16_table()
    # T5 from each bucket's first member (bucket j = all w with w >> 18 == j); NaN buckets: the infinity slot
    j = np.arange(16384, dtype=np.uint32)
    first = j << 18
    nan_bucket = ((first >> 23) & 0xff) == 0xff
    slot5 = np.where(nan_bucket, ((first >> 31) << 10) | (0x7c00 >> 5), f16_rtz(np.where(nan_bucket, 0, first)) >> 5)
    t5 = t16[slot5]
    bad = checked = nan = 0
    for k in range((1 << 32) // CHUNK):
        u = np.arange(k * CHUNK, (k + 1) * CHUNK, dtype=np.uint64).astype(np.uint32)
        isnan = ((u >> 23) & 0xff) == 0xff
        isnan &= (u & 0x7fffff) != 0
        v = u[~isnan]
        s = f16_rtz(v) >> 5
        mism = s != slot5[v >> 18]
        bad += int(mism.sum())
        ent = np.any(t16[s] != t5[v >> 18], axis=1)
        bad += int(ent.sum())
        checked += v.size
        nan += int(isnan.sum())
        if mism.any() or ent.any():
            w = v[mism | ent][:5]
            print('MISMATCH at', [hex(int(x)) for x in w])
    print('T5 vs T16: %d FP32 encodings checked, %d NaN encodings skipped, %d mismatches (slot or entry)' %
          (checked, nan, bad))
    print('T5: 16384 entries, %d distinct (a, b), %d NaN buckets' % (len({tuple(r) for r in t5.tolist()}),
                                                                       int(nan_bucket.sum())))
    ok = bad == 0 and checked + nan == 1 << 32
    if args.clang:
        with tempfile.TemporaryDirectory() as d:
            cpp = os.path.join(d, 'dump.cpp')
            exe = os.path.join(d, 'dump.exe')
            out = os.path.join(d, 't5.bin')
            open(cpp, 'w').write('#include <cstdio>\n#include "gelu_t5_table.h"\nint main(int, char** v) {\n'
                                 '  static uint32_t t[2 * wg::kGeluT5Entries]; wg::gelu_t5_expand(t);\n'
                                 '  FILE* f = std::fopen(v[1], "wb"); if (!f) return 1;\n'
                                 '  return std::fwrite(t, 1, sizeof t, f) == sizeof t && std::fclose(f) == 0 ? 0 : 1;\n}\n')
            subprocess.run([args.clang, '-std=c++17', '-O1', '-D_CRT_SECURE_NO_WARNINGS', '-I', RUNTIME, cpp, '-o', exe],
                           check=True)
            subprocess.run([exe, out], check=True)
            rt = np.fromfile(out, '<u4').reshape(16384, 2)
        same = np.array_equal(rt, t5)
        print('runtime builder wg::gelu_t5_expand: %s (%d differing entries)' %
              ('IDENTICAL' if same else 'DIFFERENT', int(np.any(rt != t5, axis=1).sum())))
        ok = ok and same
    print('t5_check: ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
