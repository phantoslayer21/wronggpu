# SPDX-License-Identifier: MIT
"""Raw-byte comparison of a D3D12 result against HIP's, with a distinct sentinel per backend (model K's D3D12 tests).

Before a frame every output and intermediate buffer is filled with a sentinel byte: 0xA5 on HIP, 0x5A on D3D12. After
it, the two buffers are compared as 4-byte words of raw bit patterns (no float conversion: an FP16 NaN keeps its
payload, -0 stays -0). A word that differs is a failure, except a word that neither side wrote (HIP's sentinel on one
side and D3D12's on the other): those are counted, and a caller that knows the buffer is written in full allows none.
A write that D3D12 dropped leaves its sentinel where HIP stored a value, and a write that only D3D12 made leaves HIP's
sentinel: both fail, which the negative controls in test_k_raw_compare.py show.
"""
import numpy as np

HIP_SENTINEL = 0xA5
D3D12_SENTINEL = 0x5A


def sentinel_word(byte):
    return np.uint32(byte * 0x01010101)


def compare_raw(what, hip, d3d12, allow_untouched=0, hip_sentinel=HIP_SENTINEL, d3d12_sentinel=D3D12_SENTINEL):
    """hip, d3d12: bytes-like of equal length (a multiple of 4). Raises AssertionError on a differing word or on more
    than allow_untouched words written by neither; returns the number of such words."""
    hip, d3d12 = bytes(hip), bytes(d3d12)
    if len(hip) != len(d3d12) or len(hip) % 4:
        raise AssertionError(f"{what}: sizes {len(hip)} and {len(d3d12)} (need equal, a multiple of 4)")
    h, d = np.frombuffer(hip, np.uint32), np.frombuffer(d3d12, np.uint32)
    differ = h != d
    untouched = differ & (h == sentinel_word(hip_sentinel)) & (d == sentinel_word(d3d12_sentinel))
    bad = differ & ~untouched
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(f"{what}: {int(bad.sum())} of {len(h)} words differ; first at word {i}: HIP {h[i]:08x}, "
                             f"D3D12 {d[i]:08x}")
    n = int(untouched.sum())
    if n > allow_untouched:
        raise AssertionError(f"{what}: {n} of {len(h)} words written by neither side (allowed {allow_untouched})")
    return n
