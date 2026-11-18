# SPDX-License-Identifier: MIT
"""Raw-byte comparison of a D3D12 result against HIP's, with a distinct sentinel per backend (model K's D3D12 tests).

Before a frame every output and intermediate buffer is filled with a sentinel byte: 0xA5 on HIP, 0x5A on D3D12. After
it, the two buffers are compared as 4-byte words of raw bit patterns (no float conversion: an FP16 NaN keeps its
payload, -0 stays -0). A buffer whose size is not a multiple of 4 (ctapT is 2 bytes a render pixel, so an odd render
size gives one) has its 1 to 3 trailing bytes compared byte for byte under the same rules, so every byte is checked.
A unit that differs is a failure, except a unit that neither side wrote (HIP's sentinel on one side and D3D12's on the
other): those are counted, and a caller that knows the buffer is written in full allows none.
A write that D3D12 dropped leaves its sentinel where HIP stored a value, and a write that only D3D12 made leaves HIP's
sentinel: both fail, which the negative controls in test_k_raw_compare.py show.
"""
import numpy as np

HIP_SENTINEL = 0xA5
D3D12_SENTINEL = 0x5A


def sentinel_word(byte):
    return np.uint32(byte * 0x01010101)


def _compare_units(what, h, d, hip_sentinel, d3d12_sentinel, unit, base, width):
    """h, d: arrays of equal length of one unit (words or bytes); base: index of the first unit in the buffer (bytes, for
    the trailing bytes). Returns (units written by neither side, units compared); raises on a differing unit."""
    differ = h != d
    untouched = differ & (h == hip_sentinel) & (d == d3d12_sentinel)
    bad = differ & ~untouched
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(f"{what}: {int(bad.sum())} of {len(h)} {unit}s differ; first at {unit} {base + i}: "
                             f"HIP {int(h[i]):0{width}x}, D3D12 {int(d[i]):0{width}x}")
    return int(untouched.sum()), len(h)


def compare_raw(what, hip, d3d12, allow_untouched=0, hip_sentinel=HIP_SENTINEL, d3d12_sentinel=D3D12_SENTINEL):
    """hip, d3d12: bytes-like of equal length, any length. Raises AssertionError on a differing word or byte, or on more
    than allow_untouched units written by neither side; returns the number of such units (words and trailing bytes)."""
    hip, d3d12 = bytes(hip), bytes(d3d12)
    if len(hip) != len(d3d12):
        raise AssertionError(f"{what}: sizes {len(hip)} and {len(d3d12)} (need equal)")
    whole = len(hip) - len(hip) % 4
    untouched, total = _compare_units(
        what, np.frombuffer(hip[:whole], np.uint32), np.frombuffer(d3d12[:whole], np.uint32),
        sentinel_word(hip_sentinel), sentinel_word(d3d12_sentinel), "word", 0, 8)
    if whole < len(hip):
        n, m = _compare_units(
            what, np.frombuffer(hip[whole:], np.uint8), np.frombuffer(d3d12[whole:], np.uint8),
            hip_sentinel, d3d12_sentinel, "byte", whole, 2)
        untouched, total = untouched + n, total + m
    if untouched > allow_untouched:
        raise AssertionError(f"{what}: {untouched} of {total} units (words, trailing bytes) written by neither side "
                             f"(allowed {allow_untouched})")
    return untouched
