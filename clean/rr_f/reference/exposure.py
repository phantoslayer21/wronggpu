# SPDX-License-Identifier: MIT
"""Independent CPU auto-exposure pass from spec 58 (DRAFT 6).

The caller supplies all nine published symbols and the already API-decoded f32
colour image. No build-specific constants, format decoder, lifecycle transitions,
GPU implementation, or game exposure inputs are supplied here. The caller writes
``stored`` to EX[par] as part of the exposure pass, without frame rollback, and
supplies the combined reset flag (both exposure memories cleared before reset).
"""
from dataclasses import dataclass, fields
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import operator

import numpy as np

from .exact import F32, is_midpoint_close, round_frac


_F32_TINY = np.float32(2.0 ** -126)


def _scalar32(value, name):
    """Require a scalar already representable in f32, without hidden rounding."""
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real f32 scalar")
    with np.errstate(all="ignore"):
        result = np.float32(a)
    if not np.isnan(result) and float(a) != float(result):
        raise ValueError(f"{name} is not exactly representable in f32")
    return result


@dataclass(frozen=True)
class ExposureSymbols:
    """Mandatory caller-supplied f32 symbols; validates precisely spec 58.1.

    Python floats are accepted only if already exactly representable in f32.
    Convert a decimal input explicitly with np.float32 if rounding is intended.
    """
    X_LWrb: np.float32
    X_LWg: np.float32
    X_LMAX: np.float32
    X_KEY: np.float32
    X_MFL: np.float32
    X_EFL: np.float32
    X_CL: np.float32
    X_CLn: np.float32
    X_NRM: np.float32

    def __post_init__(self):
        for field in fields(self):
            value = _scalar32(getattr(self, field.name), field.name)
            if not np.isfinite(value) or (value != 0 and abs(value) < _F32_TINY):
                raise ValueError(f"{field.name} must be finite and not subnormal")
            object.__setattr__(self, field.name, value)
        if self.X_LMAX < 0:
            raise ValueError("X_LMAX must be nonnegative")
        for name in ("X_KEY", "X_MFL", "X_EFL"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.X_CLn > self.X_CL:
            raise ValueError("X_CLn must not exceed X_CL")
        if Fraction(float(self.X_KEY)) / Fraction(float(self.X_MFL)) >= 65520:
            raise ValueError("X_KEY / X_MFL must be below binary16 overflow")


@dataclass(frozen=True)
class ExposureResult:
    """Four f32 pass outputs, next f16 storage value, and f32 diagnostics."""
    E: np.float32
    Ep: np.float32
    invE: np.float32
    X3: np.float32
    stored: np.float16
    E_raw: np.float32
    mean: np.float32


def _ftz(value):
    a = np.asarray(value, dtype=np.float32)
    return np.where(np.abs(a) < _F32_TINY, np.copysign(np.float32(0), a), a)


def _mul(a, b):
    with np.errstate(all="ignore"):
        return _ftz(np.multiply(_ftz(a), _ftz(b), dtype=np.float32))


def _add(a, b):
    with np.errstate(all="ignore"):
        return _ftz(np.add(_ftz(a), _ftz(b), dtype=np.float32))


def _rcp(value):
    value = float(_ftz(value))
    if math.isnan(value):
        return np.float32(np.nan)
    if value == 0:
        return np.float32(math.copysign(math.inf, value))
    if math.isinf(value):
        return np.float32(math.copysign(0.0, value))
    return np.float32(_ftz(round_frac(1 / Fraction(value), F32)))


def _log2(value):
    """Correct f32 rounding with a guarded high-precision logarithm."""
    value = float(_ftz(value))
    if math.isnan(value) or value < 0:
        return np.float32(np.nan)
    if value == 0:
        return np.float32(-np.inf)
    if math.isinf(value):
        return np.float32(np.inf)
    mantissa, exponent = math.frexp(value)
    if mantissa == 0.5:
        return np.float32(exponent - 1)
    with localcontext() as context:
        context.prec = 110
        approximation = Fraction(Decimal(value).ln() / Decimal(2).ln())
    # Powers of two are exact above. Other f32 inputs have irrational logarithms;
    # refuse an undecidable midpoint instead of claiming a rounded result.
    if is_midpoint_close(abs(approximation), F32, Fraction(1, 10 ** 90)):
        raise ArithmeticError("log2 result too close to an f32 rounding boundary")
    return np.float32(_ftz(round_frac(approximation, F32)))


def _exact_mean(values):
    """Exact nonnegative f32 sum and quotient, rounded once to f32, then FTZ.

    Each finite f32 is an integer multiple of 2^-149. In blocks of <= 2^16
    values, group integer significands by exponent. Each bincount addition is
    exact in binary64: its nonnegative integer sum is < 2^40. Python integer
    addition combines all blocks without size/order-dependent roundoff. This
    exceeds 58.2 step 2's accuracy bound at every supported array size.
    """
    a = np.asarray(values, dtype=np.float32).reshape(-1)
    if a.size == 0 or np.any(~np.isfinite(a)) or np.any(a < 0):
        raise ValueError("mean requires nonempty finite nonnegative f32 values")
    total = 0
    for start in range(0, a.size, 1 << 16):
        bits = a[start:start + (1 << 16)].view(np.uint32)
        exponent = ((bits >> 23) & 255).astype(np.int32)
        mantissa = (bits & 0x7fffff) | np.where(exponent != 0, 0x800000, 0).astype(np.uint32)
        bins = np.bincount(exponent, weights=mantissa.astype(np.float64), minlength=255)
        for e in np.flatnonzero(bins):
            total += int(bins[e]) << max(int(e) - 1, 0)
    quotient = Fraction(total, int(a.size) << 149)
    return np.float32(_ftz(round_frac(quotient, F32)))


def _pair(value, name):
    try:
        if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) for v in value):
            raise ValueError
        return tuple(operator.index(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain two integers") from None


def auto_exposure(colour, *, symbols, render_size, Eprev, reset, origin=(0, 0)):
    """Evaluate spec 58.2 once, without mutating input or external state.

    ``colour`` is an H x W x C (C >= 3) float32 array after the resource view's
    standard conversion, including any sRGB decode. ``render_size`` and
    ``origin`` are (width, height) and (x, y). Zero sizes are rejected; there is
    no creation-size fallback. ``Eprev`` is an already-read f32 scalar (normally
    an exact conversion of stored binary16); arbitrary NaN/negative/infinite
    values use the branches explicitly specified by 58.2. Reset overrides it
    with 1. The returned ``stored`` preserves binary16 subnormals and zero.
    """
    if not isinstance(symbols, ExposureSymbols):
        raise TypeError("symbols must be ExposureSymbols")
    c = np.asarray(colour)
    if c.dtype != np.float32 or c.ndim != 3 or c.shape[2] < 3:
        raise ValueError("colour must be an API-decoded float32 H x W x C array, C >= 3")
    width, height = _pair(render_size, "render_size")
    ox, oy = _pair(origin, "origin")
    if width <= 0 or height <= 0 or ox < 0 or oy < 0:
        raise ValueError("render size must be positive and origin nonnegative")
    if ox + width > c.shape[1] or oy + height > c.shape[0]:
        raise ValueError("render rectangle lies outside colour")
    if not isinstance(reset, (bool, np.bool_)):
        raise ValueError("reset must be boolean")
    previous = _scalar32(Eprev, "Eprev")
    if reset:
        previous = np.float32(1)
    s = symbols
    rgb = c[oy:oy + height, ox:ox + width, :3]
    # Chosen permitted path: separate f32 products and left-associated additions.
    luminance = _add(_add(_mul(s.X_LWrb, rgb[..., 0]),
                         _mul(s.X_LWg, rgb[..., 1])), _mul(s.X_LWrb, rgb[..., 2]))
    w = np.fmin(np.fmax(luminance, np.float32(0)), s.X_LMAX)
    mean = _exact_mean(w)
    raw = np.float32(_mul(s.X_KEY, _rcp(np.fmax(mean, s.X_MFL))))
    with np.errstate(all="ignore"):
        stored = np.float16(raw)
    exposure = np.float32(stored) if stored != 0 else np.float32(1)
    previous = previous if previous != 0 else np.float32(1)
    inverse = _rcp(exposure)
    change = np.float32(0)
    if not reset:
        ratio = _mul(np.fmax(exposure, s.X_EFL), _rcp(np.fmax(_ftz(previous), s.X_EFL)))
        clamped = np.fmin(np.fmax(_log2(ratio), s.X_CLn), s.X_CL)
        change = np.float32(_mul(clamped, s.X_NRM))
    return ExposureResult(exposure, previous, inverse, change, stored, raw, mean)
