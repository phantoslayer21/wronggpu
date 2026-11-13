# SPDX-License-Identifier: MIT
"""NumPy reference of spec/07-auto-exposure.md (stage "exposure" of spec/90-test-interface.md): the clean team's oracle.

    e = exposure(color, pre_exposure=1.0, alpha=0.18)     # float, FP16-exact

Arithmetic in float64 (spec 07: the mean "summed in FP32 or wider"; e rounded to FP16 at the end and saturated at
65504, the largest finite FP16 value).
"""
import numpy as np


def sample_grid(rw, rh):
    """(n_x, n_y, X [n_x], Y [n_y]): X_i = floor(i (rw - 1) / (n_x - 1) + 1/2), exactly (integer arithmetic)."""
    nx, ny = max(256, rw // 2), max(256, rh // 2)
    i, j = np.arange(nx, dtype=np.int64), np.arange(ny, dtype=np.int64)
    X = (2 * i * (rw - 1) + (nx - 1)) // (2 * (nx - 1))
    Y = (2 * j * (rh - 1) + (ny - 1)) // (2 * (ny - 1))
    return nx, ny, X, Y


def log_luminance(c, pre_exposure):
    """z per sample: ln(L + 1e-6) for finite L > 0, else 0, with L = (R/4 + G/2 + B/4) / pi."""
    c = np.asarray(c, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        L = (0.25 * c[..., 0] + 0.5 * c[..., 1] + 0.25 * c[..., 2]) / float(pre_exposure)
        ok = np.isfinite(L) & (L > 0)
        return np.where(ok, np.log(np.where(ok, L, 1.0) + 1e-6), 0.0)


def exposure(color, pre_exposure=1.0, alpha=0.18):
    color = np.asarray(color)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"color must be [rh, rw, 3], got {color.shape}")
    rh, rw = color.shape[:2]
    nx, ny, X, Y = sample_grid(rw, rh)
    z = log_luminance(color[Y][:, X], pre_exposure)
    m = float(np.mean(z))
    K = alpha / (1.0 - alpha)
    with np.errstate(over="ignore"):
        return float(min(np.float16(K * np.exp(-m)), np.float16(65504.0)))   # spec 07: saturate at 65504


def random_color(rng, rw, rh, special=True):
    """Synthetic HDR frame; special=True adds NaN, +-Inf, negative and zero pixels."""
    c = rng.lognormal(-1.5, 2.0, size=(rh, rw, 3))
    if special:
        flat = c.reshape(-1, 3)
        n = flat.shape[0]
        idx = rng.choice(n, size=min(n, max(8, n // 20)), replace=False)
        flat[idx[0::5]] = np.nan
        flat[idx[1::5], 1] = np.inf
        flat[idx[2::5], 0] = -np.inf
        flat[idx[3::5]] = -rng.uniform(0, 1, size=(len(idx[3::5]), 3))
        flat[idx[4::5]] = 0.0
    return c.astype(np.float32)
