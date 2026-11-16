# SPDX-License-Identifier: MIT
"""Model K's per-frame runtime on the D3D12 devices against the HIP backend, frame by frame, byte for byte (issue #15).

The same frames (a synthetic model K, random FP16 parameters; clean/k/tests/test_k_runtime_gpu.py) go through two
runtimes of the same build: HIP (backend 0) and a D3D12 device, and after every frame the game output (the raw bytes of
the RGBA16F rows, or the dense RGB float32) and the K buffers must be identical, compared as raw 4-byte words
(clean/k/tests/raw_compare.py):
- the output buffer and the network's intermediates (the input stage's x0, the skip maps S0..S4, the ping-pong maps A
  and B, the output colour taps ctap / ctapT, the head map or compact pixel head, exposure.pair, k.scalars) are filled
  with a sentinel before every frame: 0xA5 on HIP, 0x5A on D3D12. A write that one side did not make leaves its
  sentinel and fails the comparison; the output region and every intermediate must be written in full (a word neither side wrote fails);
- the first frame too: option k_fill (a test hook of the runtime) fills every buffer a frame writes and both history
  copies with the backend's sentinel after allocation, so first-frame and reset-only writes are proven;
- the exposure buffers and the histories k.Hc / k.L / k.Hh (the state a frame carries to the next) are compared raw;
- a buffer that does not exist fails the test (no skipping);
- negative controls: a dropped write in the real output (its first word set back to the D3D12 sentinel) must be caught.
The devices:
- the hybrid device (backend 2: the ports of dx12/manifest.json run on D3D12, any other kernel on HIP, one arena):
  every kernel the frame launches must have run on D3D12 (the launch trace), so that an unported kernel cannot hide
  behind HIP running on both sides;
- the native device (backend 3, no HIP).
Sequences: a reset frame, steady frames with a pan, a game reset, a render-size change (the history is kept); HDR with a
value, with the game's exposure texel, and with auto-exposure, and LDR; output rows RGBA16F and dense RGB; render size
equal to the output (native AA, DLAA 1.0: the legacy input kernels), a jitter outside [-0.5, 0.5], and input_tiled=0.
Refusals (one line, create or frame): an approximation option on the native device, an input-copy frame (raw_inputs).
Softmax weight rounding (native device; runtime_k.cpp "create-time self-check"): with nothing set the mix form passes
the check (smx=mix, no note); with the developer hook k_smx_test=mix_fail the round-to-odd ports are taken (smx=rodd,
exactly one note of create, which the executor logs) and the frames match HIP with every softmax stage on its __rodd port,
in both e4 / d4 shapes (1440p Quality: NG = 4; 4K Performance: NG = 2); with both forms failed, or with a corrupt stage
shader (a softmax stage's __rodd port, the fixed-weight stage e0), create refuses in one line. Each create leg runs in its own process. The hook exists only in a runtime library
built with clean/runner/build.py --dev-diagnostics: these tests skip on another build.

GPU only: WG_K_GPU_BUILD=<full build dir: clean/runner/build.py with the runtime library, and clean/dx12/build_dx12.py
--model-k (or --only the K kernels and exposure_auto_diet) into the same folder>, ROCM_PATH; run each test under the
machine's GPU lock.
Run: python -m unittest clean.k.tests.test_k_runtime_d3d12
"""
import ctypes
import json
import os
import subprocess
import sys
import unittest

import numpy as np

# read once per process, at the first launch of the first runtime
TRACE = os.path.join(os.environ.get("WG_K_GPU_BUILD", "."), "k_d3d12_launch_trace.txt")
os.environ.setdefault("WG_DX12_KERNELS", "all")
os.environ.setdefault("WG_LAUNCH_TRACE", TRACE)

from clean.k.reference.network import random_network_params  # noqa: E402
from clean.k.tests.raw_compare import D3D12_SENTINEL, HIP_SENTINEL, compare_raw, sentinel_word  # noqa: E402
from clean.k.tests.test_k_runtime_gpu import (KRuntimeExt, gpu_build, inputs, pitch256,  # noqa: E402
                                              write_k_model)

OUT, MAX_RENDER = (200, 120), (130, 78)
# written in full by every frame: compared raw with a sentinel before the frame (the head buffer is added per frame)
WRITTEN = ["exposure.pair", "k.scalars", "k.x0", "k.S0", "k.S1", "k.S2", "k.S3", "k.S4", "k.A", "k.B", "k.ctap",
           "k.ctapT"]
# the state a frame carries to the next (the histories) and the last auto-exposure value: compared raw, not poked
STATE = ["exposure.measured", "k.Hc", "k.L", "k.Hh"]
# kernels a HDR / LDR frame of the default path launches (the D3D12 ports must run every one of them in the hybrid device)
STAGES = ["k_stage2_e0", "k_stage2_e1", "k_stage_e2", "k_stagew2_e3", "k_stagew4_e4", "k_stagew4_m", "k_stagew4_d4",
          "k_stagew2_d3", "k_stage_d2", "k_stage2_d1", "k_stage2_d0_hp"]
ALWAYS = ["k_expo_scalars", "k_out_pixel_c"]
TILED = ["kx/k_in_tiled", "kx/k_in_tiled_uniform"]
LEGACY = ["kx/k_in_color", "kx/k_in_base"]   # (+ kx/k_in_reset on a reset frame)


def trace_since(pos):
    if not os.path.exists(TRACE):
        return [], pos
    with open(TRACE, "rb") as f:
        f.seek(pos)
        data = f.read()
    return data.decode(errors="replace").splitlines(), pos + len(data)


def poke(rt, ptr, nbytes, byte):
    rt.upload(ptr, np.full(nbytes, byte, np.uint8))


def out_bytes(rt, half):
    """The game output's defined bytes as the frame wrote them: RGBA16F rows (8 bytes a pixel, the pitch padding left
    out) or dense RGB float32."""
    W, H = rt.W, rt.H
    if half:
        rows = rt.download(rt.bufs["out"], rt.half_pitch * H, np.uint8).reshape(H, rt.half_pitch)
        return np.ascontiguousarray(rows[:, :8 * W]).tobytes()
    return rt.download(rt.bufs["out"], W * H * 12, np.uint8).tobytes()


class TestKRuntimeD3D12(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = gpu_build()
        if not os.path.exists(os.path.join(cls.build, "dx12", "manifest.json")):
            raise unittest.SkipTest("no dx12/manifest.json in WG_K_GPU_BUILD (build_dx12.py --model-k)")
        cls.params = random_network_params(11)
        cls.mdir = os.path.join(cls.build, "k_runtime_d3d12_test")
        cls.wgcm, cls.map = write_k_model(cls.mdir, cls.params)

    def make(self, backend, hdr, auto, options, out, max_render):
        return KRuntimeExt(self.build, self.mdir, self.map, out, max_render, hdr, False, auto, options, backend=backend)

    def sequence(self, backend, hdr, mode, half, frames, options="", seed=0, out=OUT, max_render=MAX_RENDER):
        """The frames [(rw, rh, reset, E, pi[, jitter])] through HIP and `backend`; raises on the first difference."""
        auto = mode == 2
        # k_fill: every buffer a frame writes and both history copies start with the backend's own sentinel (HIP 0xA5,
        # D3D12 0x5A), so the first frame's and a reset frame's writes are proven too, not only the later frames'
        sep = ";" if options else ""
        hip = self.make(0, hdr, auto, f"{options}{sep}k_fill={HIP_SENTINEL}", out, max_render)
        dx = self.make(backend, hdr, auto, f"{options}{sep}k_fill={D3D12_SENTINEL}", out, max_render)
        head = "k.N" if "k_head=full" in options else "k.Np"
        pos = os.path.getsize(TRACE) if os.path.exists(TRACE) else 0
        on_dx, on_hip = {}, {}
        try:
            for i, fr in enumerate(frames):
                rw, rh, reset, E, pi = fr[:5]
                x = inputs(np.random.default_rng(seed * 100 + i), rw, rh, hdr, reset, jitter=fr[5] if len(fr) > 5 else None)
                x["pre_exposure"], x["E_cur"] = pi, float(np.float32(E))
                if i == 2:
                    x["mv_scale"], x["mv_offset"] = (0.75, 1.25), (0.125, -0.25)
                # sentinels before each frame: the output buffer, and from the second frame on the intermediates again
                # (the first frame's come from k_fill at create)
                for rt, byte in ((hip, HIP_SENTINEL), (dx, D3D12_SENTINEL)):
                    poke(rt, rt.bufs["out"], rt.bufs_size["out"], byte)
                    if i:
                        for name in WRITTEN + [head]:
                            p, n = rt.buffer_info(name)
                            poke(rt, p, n, byte)
                _, _ = hip.frame(x, mode, half=half)
                _, pos = trace_since(pos)
                _, _ = dx.frame(x, mode, half=half)
                lines, pos = trace_since(pos)
                for ln in lines:
                    parts = ln.split()
                    if parts:
                        # where: "dx12" (hybrid device) or "d3d12" (native device), else HIP
                        tgt = on_dx if parts[-1] in ("dx12", "d3d12") else on_hip
                        tgt[parts[0]] = tgt.get(parts[0], 0) + 1
                want, got = out_bytes(hip, half), out_bytes(dx, half)
                compare_raw(f"frame {i}: output", want, got)
                if i == 0:   # negative control: a dropped write in the real output is caught
                    dropped = bytearray(got)
                    dropped[:4] = sentinel_word(D3D12_SENTINEL).tobytes()
                    with self.assertRaises(AssertionError, msg="a dropped output write was not detected"):
                        compare_raw("control", want, bytes(dropped))
                for name in WRITTEN + STATE + [head]:
                    # a buffer that does not exist raises RuntimeError here: that is a failure, not a skip
                    w, g = hip.buffer(name, np.uint8), dx.buffer(name, np.uint8)
                    # every compared buffer is written in full by a frame: a word neither side wrote fails too
                    compare_raw(f"frame {i}: {name}", w, g)
        finally:
            del hip, dx
        return on_dx, on_hip

    FRAMES = [(130, 78, True, 1.0, 1.0), (130, 78, False, 1.0, 1.0), (130, 78, False, 1.25, 0.5),
              (100, 60, False, 1.0, 1.0), (100, 60, True, 0.75, 1.0)]

    def check_hybrid(self, hdr, mode, half, options="", frames=None, inputs_ran=TILED, **size):
        on_dx, on_hip = self.sequence(2, hdr, mode, half, frames or self.FRAMES, options, **size)
        want = STAGES + ALWAYS + ["k_out_color"] + (["exposure_auto_diet"] if mode == 2 and hdr else [])
        if size.get("out", OUT)[0] >= 3840:   # above 24 windows e4 and d4 take their NG = 2 kernels
            want = [{"k_stagew4_e4": "k_stagew2_e4", "k_stagew4_d4": "k_stagew2_d4"}.get(k, k) for k in want]
        if "k_head=full" in options:   # the head map and k_out_pixel instead of the compact pixel head
            want = [{"k_stage2_d0_hp": "k_stage2_d0_hidden", "k_out_pixel_c": "k_out_pixel"}.get(k, k) for k in want]
        never = [k for k in want if not on_dx.get(k)]
        self.assertFalse(never, f"never ran on D3D12 in the hybrid device: {never}; D3D12: {sorted(on_dx)}")
        # the input stage: the tiled kernel (the host's frame values by default, else the GPU's own), or the legacy kernels
        self.assertTrue(any(on_dx.get(k) for k in inputs_ran), f"none of {inputs_ran} ran on D3D12: {sorted(on_dx)}")
        k_on_hip = sorted(k for k in on_hip if k.startswith(("k_", "kx/")))
        self.assertFalse(k_on_hip, f"model K kernels that ran on HIP in the hybrid device: {k_on_hip}")
        return on_dx

    def test_hybrid_hdr_value_rows(self):
        self.check_hybrid(True, 0, True)

    def test_hybrid_hdr_texel_dense(self):
        self.check_hybrid(True, 1, False)

    def test_hybrid_hdr_auto_exposure(self):
        self.check_hybrid(True, 2, True)

    def test_hybrid_ldr_rows(self):
        self.check_hybrid(False, 0, True)

    def test_hybrid_head_full(self):
        self.check_hybrid(True, 0, True, "k_head=full")

    def test_hybrid_input_without_host_uniforms(self):
        """k_in_uniforms=0: the tiled input kernel that computes its frame values on the GPU."""
        on_dx, _ = self.sequence(2, True, 0, True, self.FRAMES, "k_in_uniforms=0")
        self.assertTrue(on_dx.get("kx/k_in_tiled"), sorted(on_dx))
        self.assertFalse(on_dx.get("kx/k_in_tiled_uniform"))

    # the legacy input kernels: a render size equal to the output (native AA, DLAA 1.0; above 0.68 x the output the tiled
    # kernel's guard refuses), a jitter outside [-0.5, 0.5], and input_tiled=0
    DLAA = dict(out=OUT, max_render=OUT)
    FRAMES_DLAA = [(200, 120, True, 1.0, 1.0), (200, 120, False, 1.0, 1.0), (200, 120, False, 1.25, 0.5),
                   (160, 96, False, 1.0, 1.0)]
    FRAMES_JITTER = [(130, 78, True, 1.0, 1.0), (130, 78, False, 1.0, 1.0, (0.75, -0.8)),
                     (130, 78, False, 1.0, 1.0, (-0.9, 0.6)), (130, 78, False, 1.0, 1.0, (0.5, -0.5))]

    def test_hybrid_dlaa_ratio_one(self):
        on_dx = self.check_hybrid(True, 0, True, frames=self.FRAMES_DLAA, inputs_ran=LEGACY, **self.DLAA)
        self.assertTrue(on_dx.get("kx/k_in_reset"), sorted(on_dx))

    def test_hybrid_ldr_dlaa_dense(self):
        self.check_hybrid(False, 1, False, frames=self.FRAMES_DLAA, inputs_ran=LEGACY, **self.DLAA)

    def test_hybrid_jitter_outside_guard(self):
        on_dx = self.check_hybrid(True, 0, True, frames=self.FRAMES_JITTER, inputs_ran=LEGACY)
        self.assertTrue(on_dx.get("kx/k_in_tiled_uniform") or on_dx.get("kx/k_in_tiled"), sorted(on_dx))

    def test_hybrid_input_not_tiled(self):
        self.check_hybrid(True, 0, True, "input_tiled=0", inputs_ran=LEGACY)

    # the sizes of a game: 1440p Quality (2560 x 1440 from 1707 x 960) and 4K Performance (3840 x 2160 from 1920 x 1080,
    # where e4 / d4 run their NG = 2 kernels); a reset frame, a steady frame with a pan, and a smaller render size
    QUALITY = dict(out=(2560, 1440), max_render=(1707, 960))
    FRAMES_Q = [(1707, 960, True, 1.0, 1.0), (1707, 960, False, 1.0, 1.0), (1500, 840, False, 1.0, 1.0)]

    def test_hybrid_1440p_quality(self):
        self.check_hybrid(True, 0, True, frames=self.FRAMES_Q, **self.QUALITY)

    def test_hybrid_4k_performance(self):
        self.check_hybrid(True, 0, True, frames=[(1920, 1080, True, 1.0, 1.0), (1920, 1080, False, 1.0, 1.0)],
                          out=(3840, 2160), max_render=(1920, 1080))

    def test_native_1440p_quality(self):
        self.sequence(3, True, 0, True, self.FRAMES_Q, **self.QUALITY)

    def test_native_hdr_value_rows(self):
        self.sequence(3, True, 0, True, self.FRAMES)

    def test_native_hdr_auto_exposure_dense(self):
        self.sequence(3, True, 2, False, self.FRAMES)

    def test_native_ldr_rows(self):
        self.sequence(3, False, 0, True, self.FRAMES)

    def test_native_dlaa_ratio_one(self):
        self.sequence(3, True, 0, True, self.FRAMES_DLAA, **self.DLAA)

    def test_native_jitter_outside_guard(self):
        self.sequence(3, True, 0, True, self.FRAMES_JITTER)

    def test_native_input_not_tiled(self):
        self.sequence(3, True, 0, True, self.FRAMES, "input_tiled=0")

    def test_fill_hook_reaches_both_backends(self):
        """k_fill: the runtime's own input copy of the depth (never written when the caller passes its buffers) keeps
        the backend's sentinel through a frame: HIP 0xA5 and D3D12 0x5A, so the first-frame check is not vacuous."""
        for backend in (2, 3):
            with self.subTest(backend=backend):
                hip = self.make(0, True, False, f"k_fill={HIP_SENTINEL}", OUT, MAX_RENDER)
                dx = self.make(backend, True, False, f"k_fill={D3D12_SENTINEL}", OUT, MAX_RENDER)
                try:
                    x = inputs(np.random.default_rng(3), 130, 78, True, True)
                    x["pre_exposure"], x["E_cur"] = 1.0, 1.0
                    hip.frame(x, 0, half=True)
                    dx.frame(x, 0, half=True)
                    want, got = hip.buffer("input.depth", np.uint8), dx.buffer("input.depth", np.uint8)
                    self.assertTrue((want == HIP_SENTINEL).all() and (got == D3D12_SENTINEL).all())
                    with self.assertRaises(AssertionError):   # and the comparison tells the two apart
                        compare_raw("control", want, got)
                finally:
                    del hip, dx

    # refusals: one line, naming the cause; the game keeps its own upscaler (no HIP route behind the device's back)
    def test_native_refuses_approximation_options(self):
        for options in ("network=fused16", "k_fast=1", "k_in_exact=0"):
            with self.subTest(options=options):
                with self.assertRaises(RuntimeError) as cm:
                    self.make(3, True, False, options, OUT, MAX_RENDER)
                msg = str(cm.exception)
                self.assertNotIn("\n", msg.strip())
                self.assertIn("HIP", msg)

    # ---- softmax weight rounding: the create-time self-check and its round-to-odd ports ----------------------------
    def leg(self, options, build=None):
        """One native create in its own process: {"ok", "smx", "notes", "error"}."""
        r = subprocess.run([sys.executable, "-m", "clean.k.tests.test_k_runtime_d3d12", "--softmax-leg", self.mdir,
                            self.map, options, build or self.build], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def need_dev_build(self):
        got = self.leg("k_smx_test=report")
        if not got["ok"] and "developer builds only" in got["error"]:
            self.skipTest("the softmax test hook needs a runtime built with --dev-diagnostics")
        self.assertTrue(got["ok"], got)

    def test_native_softmax_switch_is_logged(self):
        """The round-to-odd softmax ports are exact too, but taking them is never silent: one note of create."""
        self.need_dev_build()
        plain = self.leg("")
        self.assertTrue(plain["ok"], plain)
        self.assertEqual((plain["smx"], plain["notes"]), ("mix", []))
        fb = self.leg("k_smx_test=mix_fail")
        self.assertTrue(fb["ok"], fb)
        self.assertEqual(fb["smx"], "rodd(hook)")
        self.assertEqual(len(fb["notes"]), 1, fb)
        self.assertIn("round-to-odd", fb["notes"][0])
        self.assertIn("test hook", fb["notes"][0])

    def test_native_softmax_refusals(self):
        """Neither form shown exact, or a corrupt stage shader (a softmax stage's __rodd port, a fixed-weight stage): create
        refuses in one line naming it (the game keeps its own upscaler)."""
        import shutil
        import tempfile
        self.need_dev_build()
        both = self.leg("k_smx_test=both_fail")
        self.assertFalse(both["ok"], both)
        self.assertIn("not exact on this driver", both["error"])
        self.assertNotIn("\n", both["error"].strip())
        # a copy of the build (its top-level files and dx12/) with one flipped byte in one stage shader: a softmax stage's
        # __rodd port and a fixed-weight stage (e0); create prepares every selectable kernel, so it refuses in one line
        for victim in ("k_stagew2_e3__rodd", "k_stage2_e0"):
            with self.subTest(victim=victim):
                tmp = tempfile.mkdtemp(prefix="k_smx_corrupt_")
                try:
                    for f in os.listdir(self.build):
                        src = os.path.join(self.build, f)
                        if os.path.isfile(src):
                            shutil.copy2(src, tmp)
                    shutil.copytree(os.path.join(self.build, "dx12"), os.path.join(tmp, "dx12"))
                    path = os.path.join(tmp, "dx12", victim + ".dxil")
                    data = bytearray(open(path, "rb").read())
                    data[len(data) // 2] ^= 0x40
                    open(path, "wb").write(bytes(data))
                    bad = self.leg("", build=tmp)
                    self.assertFalse(bad["ok"], bad)
                    self.assertIn(victim, bad["error"])
                    self.assertNotIn("\n", bad["error"].strip())
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

    def check_fallback_frames(self, frames, big, **size):
        """Forced fallback: the frames match HIP (raw outputs and intermediates) and every softmax stage ran its __rodd
        port on D3D12, none its mix form."""
        self.need_dev_build()
        on_dx, _ = self.sequence(3, True, 0, True, frames, "k_smx_test=mix_fail", **size)
        ng = "k_stagew2_" if big else "k_stagew4_"
        want = ["k_stage_e2", "k_stagew2_e3", ng + "e4", "k_stagew4_m", ng + "d4", "k_stagew2_d3", "k_stage_d2",
                "k_stage2_d1"]
        self.assertEqual([k for k in want if not on_dx.get(k + "__rodd")], [], on_dx)
        self.assertEqual([k for k in want if on_dx.get(k)], [], on_dx)

    def test_native_fallback_frames_1440p_quality(self):
        self.check_fallback_frames(self.FRAMES_Q, False, **self.QUALITY)

    def test_native_fallback_frames_4k_performance(self):
        self.check_fallback_frames([(1920, 1080, True, 1.0, 1.0), (1920, 1080, False, 1.0, 1.0)], True,
                                   out=(3840, 2160), max_render=(1920, 1080))

    def test_input_copy_frames_refused_on_d3d12(self):
        """raw_inputs (the interop's HIP input copies) has no checked D3D12 port: refused, not run unchecked."""
        for backend in (2, 3):
            with self.subTest(backend=backend):
                rt = self.make(backend, True, False, "", OUT, MAX_RENDER)
                try:
                    rw, rh = MAX_RENDER
                    x = inputs(np.random.default_rng(1), rw, rh, True, True)
                    x["pre_exposure"], x["E_cur"] = 1.0, 1.0
                    crow = np.zeros((rh, pitch256(8 * rw) // 2), np.float16)
                    crow[:, :4 * rw] = x["color"].astype(np.float16).reshape(rh, 4 * rw)
                    mrow = np.zeros((rh, pitch256(4 * rw) // 2), np.float16)
                    with self.assertRaises(RuntimeError) as cm:
                        rt.frame(x, 0, half=True, raw=(crow, mrow, (1.0, 1.0), (0.0, 0.0), 2))
                    self.assertIn("raw_inputs", str(cm.exception))
                    self.assertNotIn("\n", str(cm.exception).strip())
                finally:
                    del rt


def softmax_leg(mdir, map_path, options, build):
    """Child of the softmax tests: one native create; prints ok, the smx field, the notes and the error as JSON."""
    try:
        rt = KRuntimeExt(build, mdir, map_path, OUT, MAX_RENDER, True, False, False, options, backend=3)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "smx": "", "notes": [], "error": str(e)}))
        return
    rt.lib.wg_rt_notes.argtypes = [ctypes.c_void_p]
    rt.lib.wg_rt_notes.restype = ctypes.c_char_p
    kern = rt.lib.wg_rt_kernels(rt.rt).decode().split()
    smx = [kv.split("=", 1)[1] for kv in kern if kv.startswith("smx=")]
    notes = [n for n in rt.lib.wg_rt_notes(rt.rt).decode().split("\n") if n]
    print(json.dumps({"ok": True, "smx": smx[0] if smx else "", "notes": notes, "error": ""}))


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--softmax-leg":
        softmax_leg(*sys.argv[2:6])
    else:
        unittest.main()
