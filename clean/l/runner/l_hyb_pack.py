# SPDX-License-Identifier: MIT
"""Weights of the HYBRID block kernels (SW_HYB / HYB = true): the Swin block section of the M-form kernels
(clean/runner/packing.py pack_block, the layout their fragments load) filled with the original's values, as the oracle
uses them (clean/l/reference/oracle.py _block_tokens_q7):
  gamma1, gamma2 -> g' = RN16(float32(gamma) sqrtf(C))  (gamma_q7; the kernels multiply r g' themselves)
  bp, b1, b2, B  -> FP16 values (stored as float32 in the header: exact)
  Wqk, Wv, Wp, W1, W2 -> FP8 values (q8 of the natural value)"""
import numpy as np

from clean.l.reference import oracle as O
from clean.reference.fp8 import q8
from clean.runner import packing as P


def hyb_block_values(p):
    """The block's parameters in the values the hybrid kernels read (same keys as the M-form packer)."""
    out = dict(p)
    out["gamma1"] = O.gamma_q7(p["gamma1"]).astype(np.float32)
    out["gamma2"] = O.gamma_q7(p["gamma2"]).astype(np.float32)
    for n in ("bp", "b1", "b2", "B"):
        out[n] = O.f16(p[n]).astype(np.float32)
    for n in ("Wqk", "Wv", "Wp", "W1", "W2"):
        out[n] = q8(np.asarray(p[n], np.float64)).astype(np.float32)
    return out


def pack_block_hyb(p, C, heads, mlp, drow="gfx12"):
    """The packed block section (bytes) for the hybrid kernels."""
    return P.pack_block(hyb_block_values(p), C, heads, mlp, P.resolve_d_row(drow))


def pack_merge_hyb(Wm, bm, C, C_out, drow="gfx12"):
    """The merge section (bytes) for patch_merge_c*_hyb: packing.pack_merge's layout, bm as FP16 values (float32), Wm
    plain FP8 values."""
    return P.pack_merge(q8(np.asarray(Wm, np.float64)).astype(np.float32), O.f16(bm).astype(np.float32), C, C_out,
                        P.resolve_d_row(drow))


def pack_expand_hyb(We, be, C, C_low, drow="gfx12"):
    """The expand section (bytes) for expand_c*_hyb: packing.pack_expand's layout, be as FP16 values (float32), We plain
    FP8 values."""
    return P.pack_expand(q8(np.asarray(We, np.float64)).astype(np.float32), O.f16(be).astype(np.float32), C, C_low,
                         P.resolve_d_row(drow))
