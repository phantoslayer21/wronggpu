# SPDX-License-Identifier: MIT
"""The model K activation of spec 40 ("The activation function (exact)"), as a NumPy oracle.

    t = min(max(x, -2), 2)
    p = f16(a - f16(b |t|))        a = 0.412109375 (FP16 0x3698), b = 0.0810546875 (FP16 0x2d30)
    q = f16(0.5 + f16(t p))
    act(x) = f16(x q)

`act(x)` uses NumPy float16 arithmetic: every float16 ufunc converts its operands to float32, computes there and rounds
the result to float16 once; for one +, - or * of two FP16 values this is the correctly rounded FP16 result (float32
has 24 >= 2 * 11 + 2 significand bits, so rounding twice gives the same result as rounding once).
`act_exact_bits(b)` evaluates the same formula on one bit pattern with exact rational arithmetic and the rounding of
clean/k/reference/f16.py; the exhaustive test compares the two over all 65536 inputs.
"""
from fractions import Fraction

import numpy as np

from clean.k.reference.f16 import decode_exact, round_f16_exact

A = np.float16(0.412109375)
B = np.float16(0.0810546875)
assert A.view(np.uint16) == 0x3698 and B.view(np.uint16) == 0x2D30
HALF = np.float16(0.5)
TWO = np.float16(2.0)


def act(x):
    """Elementwise activation on FP16 values (any shape); returns float16. NaN in -> NaN out."""
    x = np.asarray(x, dtype=np.float16)
    with np.errstate(invalid="ignore", over="ignore"):
        t = np.minimum(np.maximum(x, -TWO), TWO)
        p = A - B * np.abs(t)
        q = HALF + t * p
        return (x * q).astype(np.float16)


def act_exact_bits(b):
    """Exact evaluation for one FP16 bit pattern; returns the result's bit pattern (NaN results as 0x7E00)."""
    kind = decode_exact(b)
    if kind[0] == "nan":
        return 0x7E00
    if kind[0] == "inf":
        x_inf_sign = kind[1]
        t = Fraction(2 * x_inf_sign)
        xval = None
    else:
        _, mag, s = kind
        xval = mag * s
        t = min(max(xval, Fraction(-2)), Fraction(2))
    a = Fraction(float(A))  # exact: A and B are dyadic
    bq = Fraction(float(B))

    def val(bits_):
        k = decode_exact(bits_)
        assert k[0] == "num"
        return k[1] * k[2]

    bt = val(round_f16_exact(bq * abs(t)))
    p = val(round_f16_exact(a - bt))
    tp = val(round_f16_exact(t * p))  # the sign of a zero t p does not matter: q = 0.5 then
    q = val(round_f16_exact(Fraction(1, 2) + tp))
    if xval is None:  # x = +-inf
        if q == 0:
            return 0x7E00  # inf * 0
        return (0x7C00 if (kind[1] > 0) == (q > 0) else 0xFC00)
    # x q = 0: the sign is sign(x) * sign(q), and a zero q (= 0.5 + (-0.5)) is +0 under round-to-nearest
    zero_sign = (-1 if kind[2] < 0 else 1) * (1 if q >= 0 else -1)
    return round_f16_exact(xval * q, sign_if_zero=zero_sign)
