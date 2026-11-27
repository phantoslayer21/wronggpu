# SPDX-License-Identifier: GPL-3.0-only
"""Model L's HYBRID form on the fast kernels (WG_L_FORM=hybrid, D3D12): every block stage's parameters in the values
the hybrid kernels read, packed by the unchanged M-form packers (clean/runner/packing.py pack_stage for the ten shared
stages, clean/l/runner/l_pack.py pack_model_l for Model L's own launches). Reference of the runtime's C++
(clean/l/runtime/l_hyb_pack.h, byte for byte: clean/l/runtime/tests/test_l_runtime_hyb_pack.py).

Values (as clean/l/reference/oracle.py uses them in the FP16 form):
- the Swin block section: l_hyb_pack.hyb_block_values (gamma1 / gamma2 -> RN16(float32(gamma) sqrtf(C)); bp, b1, b2,
  B -> FP16 values; Wqk, Wv, Wp, W1, W2 -> FP8 values);
- the merge (enc1, enc2: Wm, bm) and the expansion (dec2, dec1: We, be), as run_stage_q7 reads them: the weights q8,
  the biases f16.
Stages other than these blocks (enc0, dec0: Model L's linears, projections, coefficient scalars) keep their values.
"""
import numpy as np

from clean.l.reference import oracle as O
from clean.l.runner.l_hyb_pack import hyb_block_values
from clean.reference.fp8 import q8

BLOCK_STAGES = ("enc0_block", "dec0_block", "enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1")
LIN_W = ("Wm", "We")
LIN_B = ("bm", "be")


def hyb_stage_values(p):
    """One block stage's tensors in the hybrid values (same keys; float32 arrays)."""
    out = hyb_block_values(p)
    for n in LIN_W:
        if n in p:
            out[n] = q8(np.asarray(p[n], np.float64)).astype(np.float32)
    for n in LIN_B:
        if n in p:
            out[n] = O.f16(p[n]).astype(np.float32)
    return out


def hyb_params(params):
    """The whole model's stage map with every block stage in the hybrid values."""
    return {s: (hyb_stage_values(v) if s in BLOCK_STAGES else v) for s, v in params.items()}
