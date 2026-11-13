# SPDX-License-Identifier: MIT
"""Build variants of enc0 / dec0 (runner/perf_variants.py) that are meant to give identical outputs: built into the
host emulator with their defines and compared bit for bit with the default build. Ablations (wrong on purpose) and
the v1 baselines are not compared here (v1: see clean/PERF.md).

Run: python -m unittest clean.tests.test_perf_variants_emu
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clean.reference import decoder as Dc
from clean.reference import encoder as E
from clean.runner import build
from clean.runner import perf_variants as PV
from clean.tests.test_kernels_emu import build_dir


def cases(stage):
    out = []
    for i, (R, kw) in enumerate([((40, 24), {}), ((37, 21), {"reset": True}), ((70, 40), {})]):
        rng = np.random.default_rng(6100 + i)
        if stage == "enc0":
            out.append((E.random_inputs(rng, rw=R[0], rh=R[1], frame=i, **kw), E.random_params(rng)))
        else:
            out.append((Dc.random_inputs(rng, rw=R[0], rh=R[1], **kw), Dc.random_params(rng)))
    return out


class TestPerfVariantsEmu(unittest.TestCase):
    def test_identical_variants(self):
        from clean.runner.stage_runner import Runner
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        base = Runner(build_dir(), "", backend="emu")
        for stage in ("enc0", "dec0"):
            want = [base.run(stage, base.pack_weights(stage, p), x, None) for x, p in cases(stage)]
            for name, (src, defines, ablation) in PV.VARIANTS[stage].items():
                if name in ("base", "v1") or ablation:
                    continue
                d = tempfile.mkdtemp(prefix="wg_emu_pv_")
                r = subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu.so"), defines),
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                variant = Runner(d, "", backend="emu")
                for (x, p), w in zip(cases(stage), want):
                    if any(PV.input_f16(stage, name)):  # reads FP16 inputs: the default build on rounded inputs
                        x = PV.round_inputs(x, *PV.input_f16(stage, name))
                        w = base.run(stage, base.pack_weights(stage, p), x, None)
                    with self.subTest(stage=stage, variant=name):
                        # "<stage>@<variant>": the variant's launch plan (input formats), here with the emulator
                        # library of that variant
                        got = variant.run(f"{stage}@{name}", variant.pack_weights(stage, p), x, None)
                        for k in w:
                            np.testing.assert_array_equal(got[k], w[k], err_msg=f"{stage}@{name} {k}")


if __name__ == "__main__":
    unittest.main()
