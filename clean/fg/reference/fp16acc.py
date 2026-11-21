# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): exact float16 chunk accumulation, as a CPU reference.

Written for this project. net1's convolutions accumulate in float16, starting at the bias: each chunk of 16 products
is summed exactly and added to the accumulator, which is rounded to float16 once per chunk (round to nearest, ties to
even). This module computes acc' = RN16(acc + sum of 16 products a_i * b_i) exactly, for arrays of accumulators.

How: every finite float16 value is an integer multiple of 2^-24, |m| < 2^40. Each m is split into a high and a low
20-bit part, so the three partial dot products (high*high, high*low + low*high, low*low) are exact in int64. The
exact total, in units of 2^-48, is reduced to units of 2^-25 (half the smallest subnormal) with a sticky bit, and
rounded to float16 with integer arithmetic. No floating-point rounding happens before the single float16 rounding.

Non-finite operands (an accumulator or factor that is inf or NaN) follow IEEE arithmetic: those elements are computed
in float64 and rounded to float16 (inf - inf and 0 * inf give NaN). What the original's matrix unit does with them is
not specified; finite inputs are exact.
"""
import numpy as np

_S24 = float(1 << 24)
_LOW = (1 << 20) - 1


def h(x):
    """Round to float16 (nearest, ties to even) and return float64."""
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def _ints(x):
    """Finite float16 values (as float64) -> exact integers m with x = m * 2^-24."""
    return np.round(np.asarray(x, dtype=np.float64) * _S24).astype(np.int64)


def _split(m):
    return m >> 20, m & _LOW


def round16_parts(hh, mid, ll):
    """RN16 of the exact value (hh * 2^40 + mid * 2^20 + ll) * 2^-48, with ll >= 0. Returns float64 float16 values."""
    hh = np.asarray(hh, dtype=np.int64)
    mid = np.asarray(mid, dtype=np.int64)
    ll = np.asarray(ll, dtype=np.int64)
    big = np.abs(hh) > (1 << 30)                       # |value| >= 2^22: infinite whatever the rest adds
    hhc = np.clip(hh, -(1 << 30), 1 << 30)
    # value * 2^25 = hh * 2^17 + mid2 / 8 + llr / 2^23, with mid2 = mid + (ll >> 20) and 0 <= llr < 2^20
    l1 = ll >> 20
    llr = ll & _LOW
    mid2 = mid + l1
    m1 = mid2 >> 3
    midr = mid2 & 7
    q = (hhc << 17) + m1                               # floor of the value in units of 2^-25
    sticky = (midr != 0) | (llr != 0)                  # the fraction (midr * 2^20 + llr) / 2^23 is not zero
    neg = q < 0
    mag = np.where(neg, np.where(sticky, -q - 1, -q), q)
    # quantum in units of 2^-25: 2 for subnormals (value < 2^-14 = 2^11 units), else 2^(e - 10) in binade 2^e units
    _, ex = np.frexp(np.maximum(mag, 1).astype(np.float64))
    e = ex.astype(np.int64) - 1
    qexp = np.where(mag < (1 << 11), 1, np.maximum(e - 10, 1))
    quantum = np.left_shift(np.int64(1), qexp)
    r = mag % quantum
    down = mag - r
    half = quantum >> 1
    odd = ((down // quantum) & 1) == 1
    up = (r > half) | ((r == half) & (sticky | odd))
    res = down + np.where(up, quantum, 0)
    val = res.astype(np.float64) * 2.0 ** -25
    val = np.where((val >= 65536.0) | big, np.inf, val)
    sign_neg = np.where(big, hh < 0, neg)
    return np.where(sign_neg, -val, val)


def acc_dot16(acc, a, b, axis=-1):
    """RN16(acc + sum over `axis` of a * b), exact for finite float16 inputs. a and b broadcast against each other;
    the reduced axis must have length <= 16; acc broadcasts against the result."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    if a.shape[axis] > 16:
        raise ValueError("a chunk has at most 16 products")
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)
    ah, al = _split(_ints(np.where(fin_a, a, 0.0)))
    bh, bl = _split(_ints(np.where(fin_b, b, 0.0)))
    hh = np.sum(ah * bh, axis=axis)
    mid = np.sum(ah * bl + al * bh, axis=axis)
    ll = np.sum(al * bl, axis=axis)
    fin_acc = np.isfinite(acc)
    m_acc = _ints(np.where(fin_acc, acc, 0.0))
    hh, mid, ll, m_acc = np.broadcast_arrays(hh, mid, ll, m_acc)
    out = round16_parts(hh, mid + (m_acc << 4), ll)
    bad = ~(np.all(fin_a & fin_b, axis=axis) & fin_acc)
    if np.any(bad):
        with np.errstate(invalid="ignore", over="ignore"):
            slow = h(acc + np.sum(a * b, axis=axis))
        out = np.where(bad, slow, out)
    return out


def _exact_matmul(p, q):
    """p @ q for int64 matrices whose every partial sum stays below 2^53 in magnitude: computed in float64 (each product
    and each partial sum is an integer below 2^53, so exact in any summation order) and returned as int64."""
    return np.rint(p.astype(np.float64) @ q.astype(np.float64)).astype(np.int64)


def acc_matmul16(acc, a, b):
    """RN16(acc + a @ b) per element, exact for finite float16 inputs: a [n, k], b [k, m] with k <= 16, acc [n, m] (or
    broadcasting to it). The same result as acc_dot16 over the k axis, computed with three matrix products of the
    20-bit parts (each partial sum below 2^45, so float64 matrix products are exact). Non-finite operands as in
    acc_dot16."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("a [n, k] and b [k, m] expected")
    if a.shape[1] > 16:
        raise ValueError("a chunk has at most 16 products")
    acc = np.broadcast_to(np.asarray(acc, dtype=np.float64), (a.shape[0], b.shape[1]))
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)
    ah, al = _split(_ints(np.where(fin_a, a, 0.0)))
    bh, bl = _split(_ints(np.where(fin_b, b, 0.0)))
    hh = _exact_matmul(ah, bh)
    mid = _exact_matmul(ah, bl) + _exact_matmul(al, bh)
    ll = _exact_matmul(al, bl)
    fin_acc = np.isfinite(acc)
    m_acc = _ints(np.where(fin_acc, acc, 0.0))
    out = round16_parts(hh, mid + (m_acc << 4), ll)
    bad = ~(np.all(fin_a, axis=1)[:, None] & np.all(fin_b, axis=0)[None, :] & fin_acc)
    if np.any(bad):
        with np.errstate(invalid="ignore", over="ignore"):
            slow = h(acc + a @ b)
        out = np.where(bad, slow, out)
    return out
