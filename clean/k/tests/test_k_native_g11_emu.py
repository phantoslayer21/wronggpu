# SPDX-License-Identifier: MIT
"""Model K's native RDNA 3 stage kernels (clean/k/dx12/kernels/k_block_g11.hlsl) on the CPU: the kernel text run by the
emulation of clean/dx12/emu (every lane a coroutine of the lane emulator; the matrix instruction is the emulator's model
of the measured gfx11 F16 one, which sums exactly), against the NumPy reference and against the RDNA 4 ports of
k_block.hlsl run the same way (gfx12 model, also exact sums).

Because both models sum each product exactly, a native stage and its RDNA 4 port must give the same bytes here, for
every output (block, merged, head, compact head, hidden history), at every size and offset: any difference is a layout,
addressing or exchange error of the native form. Against NumPy the limits are spec 49's stage limits (mean 2.5e-3,
max 3e-2 of the reference RMS), as test_k_kernels_emu.py. Negative controls: a wrong row table (WG_A_ROW_MUT) must
break both. The arithmetic of the card (sum order, the shortcut's bits) is the GPU tests' job (GPU-TESTS.md of the
native RDNA 3 work).

Needs Linux and clang++ (as test_k_kernels_emu.py).
"""
import ctypes
import os
import sys
import tempfile
import unittest

import numpy as np

from clean.k.reference import network as N
from clean.k.reference.f16 import f16
from clean.k.reference.layout import STAGE, cdiv
from clean.k.runner import k_launch as KL
from clean.k.runner import k_stages

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "clean", "dx12"))
from emu import hlsl_emu as H  # noqa: E402

PORTS = os.path.join(ROOT, "clean", "k", "dx12", "kernels", "k_block.hlsl")
NATIVE = os.path.join(ROOT, "clean", "k", "dx12", "kernels", "k_block_g11.hlsl")
# launch name (the HIP / RDNA 4 port name) -> the declarations of one library (same defines)
GROUPS = [["k_stage_e2", "k_stage2_e0", "k_stage2_e1", "k_stage_d2", "k_stage2_d1", "k_stage_d1"],
          ["k_stage_e0"], ["k_stage_e1"], ["k_stage2_d0_hp", "k_stage2_d0_hidden"],
          ["k_stage_e2__rodd", "k_stage_d2__rodd", "k_stage2_d1__rodd"]]
LIMITS = dict(mean=2.5e-3, max=3e-2)
_DEVS = {}


def _device(form, extra=None):
    """The emulated device with every stage kernel of `form` ("ports": k_block.hlsl gfx12, "native": k_block_g11.hlsl
    gfx11) under their launch names."""
    key = (form, tuple(sorted((extra or {}).items())))
    if key not in _DEVS:
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("the lane emulator needs Linux (ucontext)")
        try:
            H.compiler()
        except RuntimeError as e:
            raise unittest.SkipTest(str(e))
        out = tempfile.mkdtemp(prefix="wg_k_native_", dir=os.environ.get("WG_TEST_TMP"))
        kernels = {}
        for names in GROUPS:
            if form == "ports":
                lib, ent = H.build(PORTS, names, out, gfx=12, extra_defines=extra)
                kernels.update({n: (lib, ent[n]) for n in names})
            else:
                for n in names:   # one library each: the native declarations carry per-kernel knobs
                    d = n.replace("__rodd", "") + ("__rodd" if "__rodd" in n else "") + "__g11"
                    lib, ent = H.build(NATIVE, [d], out, gfx=11, extra_defines=extra)
                    kernels[n] = (lib, ent[d])
        _DEVS[key] = H.HlslEmu(kernels)
    return _DEVS[key]


def _runner(dev):
    from clean.runner.stage_runner import Runner
    r = Runner.__new__(Runner)   # the spec 90 Runner's model K plans on the emulated device (no build folder)
    r.model, r.backend, r.d_row, r.dev, r.module, r._funcs, r._extra_modules = "K", "hlsl", "gfx12", dev, None, {}, {}
    r.build_dir, r.variant = "", "base"
    return r


def stage_inputs(st, lv, rng):
    if st["kind"] == "dec":
        x = f16(rng.standard_normal((cdiv(lv[1], 2), cdiv(lv[0], 2), st["C_prev"])) * 2)
        return x, f16(rng.standard_normal((lv[1], lv[0], st["C"])))
    cin = 16 if st.get("embed") else st["C"]
    return f16(rng.standard_normal((lv[1], lv[0], cin))), None


def run_stage(dev, name, suffix, p, x, off, skip):
    dev.reset()
    r = _runner(dev)
    return r.run(name + suffix, r.pack_weights(name, p), x, off, skip=skip)


def run_d0(dev, kernel, p, x, off, skip):
    """d0's two-window kernels with the hidden-history pointer (the D3D12 ports' form): the raw bytes of the block map,
    the head buffer (full head [H, W, 40] or the compact pixel head [2H][2W][4]) and the hidden history [H, W, 4], all
    filled with 0x7F first (unwritten bytes stay so)."""
    dev.reset()
    r = _runner(dev)
    plan = k_stages.prepare(dev, "d0@w2", r.pack_weights("d0", p), x, off, skip)
    lv = (skip.shape[1], skip.shape[0])
    n = lv[0] * lv[1]
    head = dev.malloc(max(n * 40, 16 * n) * 2)
    hidden = dev.malloc(n * 8)
    name, grid, threads, args = plan["launches"][0]
    prm = args[0]
    prm.head = head
    for b, nb in ((plan["bufs"]["block"], plan["sizes"]["block"]), (head, max(n * 40, 16 * n) * 2), (hidden, n * 8)):
        dev.memset(b, 0x7F, nb)
    dev.launch(kernel, grid, threads, [prm, ctypes.c_void_p(hidden)])
    return {"block": dev.download(plan["bufs"]["block"], plan["sizes"]["block"]),
            "head": dev.download(head, max(n * 40, 16 * n) * 2), "hidden": dev.download(hidden, n * 8)}


SIZES = [{0: (21, 13), 1: (11, 7), 2: (9, 8)}, {0: (5, 3), 1: (3, 2), 2: (3, 2)}, {0: (17, 9), 1: (9, 5), 2: (5, 3)}]
OFFSETS = [(4, 2), (0, 6), (6, 4)]
NARROW = [("e0", ""), ("e0", "@w2"), ("e1", ""), ("e1", "@w2"), ("e2", ""), ("d2", ""), ("d1", ""), ("d1", "@w2")]


class Checks(unittest.TestCase):
    def assert_close(self, got, ref, what):
        got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
        self.assertEqual(got.shape, ref.shape, what)
        self.assertTrue(np.all(np.isfinite(got)), f"{what}: non-finite outputs")
        rms = np.sqrt(np.mean(ref ** 2))
        d = np.abs(got - ref)
        self.assertLessEqual(d.mean() / rms, LIMITS["mean"], what)
        self.assertLessEqual(d.max() / rms, LIMITS["max"], what)


class TestNativeStages(Checks):
    """Every narrow stage, both window counts, three size sets (odd maps, partial windows, maps smaller than a window)
    and three offsets: native against NumPy and byte for byte against the RDNA 4 port."""

    def test_native_against_reference_and_port(self):
        P = N.random_network_params(5)
        native, ports = _device("native"), _device("ports")
        cases = 0
        for si, sizes in enumerate(SIZES):
            for name, suffix in NARROW:
                st = STAGE[name]
                off = OFFSETS[(si + len(name) + len(suffix)) % len(OFFSETS)]
                rng = np.random.default_rng(100 * si + len(name))
                x, skip = stage_inputs(st, sizes[st["level"]], rng)
                got = run_stage(native, name, suffix, P[name], x, off, skip)
                port = run_stage(ports, name, suffix, P[name], x, off, skip)
                ref = N.run_stage(st, P[name], x, off, skip, sizes[st["level"]]) if skip is not None else \
                    N.run_stage(st, P[name], x, off)
                for k in got:
                    what = f"{name}{suffix} {sizes[st['level']]} {off} {k}"
                    self.assert_close(got[k], ref[k], what)
                    np.testing.assert_array_equal(got[k].view(np.uint32), port[k].view(np.uint32), what)
                    cases += 1
        self.assertEqual(cases, 3 * 13)

    def test_round_to_odd_forms(self):
        P = N.random_network_params(6)
        native, ports = _device("native"), _device("ports")
        for name, kern, suffix in (("e2", "k_stage_e2", ""), ("d2", "k_stage_d2", ""), ("d1", "k_stage2_d1", "@w2")):
            st = STAGE[name]
            x, skip = stage_inputs(st, SIZES[0][st["level"]], np.random.default_rng(9))
            for dev in (native, ports):   # the __rodd kernels under the default names for this run
                dev.kernels[kern], dev.kernels[kern + "__plain"] = dev.kernels[kern + "__rodd"], dev.kernels[kern]
            try:
                got = run_stage(native, name, suffix, P[name], x, (2, 4), skip)
                port = run_stage(ports, name, suffix, P[name], x, (2, 4), skip)
            finally:
                for dev in (native, ports):
                    dev.kernels[kern] = dev.kernels.pop(kern + "__plain")
            for k in got:
                np.testing.assert_array_equal(got[k].view(np.uint32), port[k].view(np.uint32), f"{name} rodd {k}")


class TestNativeD0(Checks):
    """d0's two-window head stages (full head + hidden history, compact pixel head + hidden history)."""

    def test_d0_heads_and_hidden(self):
        P = N.random_network_params(7)
        st = STAGE["d0"]
        native, ports = _device("native"), _device("ports")
        for si, sizes in enumerate(SIZES):
            lv = sizes[0]
            x, skip = stage_inputs(st, lv, np.random.default_rng(40 + si))
            off = OFFSETS[si]
            ref = N.run_stage(st, P["d0"], x, off, skip, lv)
            for kern in ("k_stage2_d0_hidden", "k_stage2_d0_hp"):
                got = run_d0(native, kern, P["d0"], x, off, skip)
                port = run_d0(ports, kern, P["d0"], x, off, skip)
                for k in got:
                    self.assertEqual(got[k], port[k], f"d0 {kern} {lv} {off} {k}")
                self.assertNotEqual(got["hidden"], b"\x7f" * len(got["hidden"]), "hidden history not written")
                if kern == "k_stage2_d0_hidden":
                    n = lv[0] * lv[1]
                    head = np.frombuffer(got["head"][:n * 80], np.float16).reshape(lv[1], lv[0], 40)
                    self.assert_close(head.astype(np.float32), ref["head"], f"d0 head {lv} {off}")


class TestNativeKnobsAndControls(Checks):
    def _e2_d1(self, dev):
        P = N.random_network_params(8)
        out = {}
        for name, suffix in (("e2", ""), ("d1", "@w2"), ("e0", "@w2")):
            st = STAGE[name]
            x, skip = stage_inputs(st, SIZES[0][st["level"]], np.random.default_rng(11))
            out[name] = (run_stage(dev, name, suffix, P[name], x, (4, 2), skip),
                         N.run_stage(st, P[name], x, (4, 2), skip, SIZES[0][st["level"]]) if skip is not None else
                         N.run_stage(st, P[name], x, (4, 2)))
        return out

    def test_knobs_keep_the_bytes(self):
        """Numeric knobs change registers, loads and scheduling only (the declarations' own knobs switched off or others
        added): the same bytes."""
        base = self._e2_d1(_device("native"))
        for knob in ({"KN_NB_HOIST": 0}, {"KB_GRPLATE": 1}, {"KB_W2LATE": 1}, {"KB_X1LDS": 1}, {"KN_A16": 0},
                     {"KN_ZSTREAM": 0, "KB_GRPLATE": 0}, {"KN_XPIN": 1, "KN_HEAD_ROLL": 1}, {"KN_VSTREAM": 1, "KN_NB_HOIST": 2},
                     {"KN_MERGE_ROLL": 1, "KN_MERGE_STEP": 2}, {"KN_AB": 1}, {"KN_AB": 2}):
            other = self._e2_d1(_device("native", knob))
            for name in base:
                for k in base[name][0]:
                    np.testing.assert_array_equal(base[name][0][k].view(np.uint32), other[name][0][k].view(np.uint32),
                                                  f"{knob} {name} {k}")

    def test_negative_control_wrong_row_table(self):
        """A wrong row table (WG_A_ROW_MUT: rows 2 and 3 swapped) must put every stage far outside the limits."""
        bad = self._e2_d1(_device("native", {"WG_A_ROW_MUT": 1}))
        for name, (got, ref) in bad.items():
            k = "block"
            rms = np.sqrt(np.mean(np.asarray(ref[k], np.float64) ** 2))
            worst = np.max(np.abs(np.asarray(got[k], np.float64) - ref[k])) / rms
            self.assertGreater(worst, 10 * LIMITS["max"], name)


if __name__ == "__main__":
    unittest.main()
