# SPDX-License-Identifier: MIT
"""The compact pixel head (k_stage2_d0_hp / k_stage2h_d0_hp -> k_out_pixel_c) against the 40-channel head map
(k_stage2_d0_hidden / k_stage2h_d0_hidden -> k_out_pixel): the same head values in another layout, so every output and
history bit must be the same.

GPU only: WG_K_GPU_BUILD=<kernel build dir>, under the shared GPU lock.
"""
import ctypes
import os
import unittest

import numpy as np

from clean.k.reference.network import random_network_params
from clean.k.runner import k_io, k_launch, k_pack
from clean.k.runner.k_bench import synthetic_frame
from clean.k.runner.k_frame import KFrame
from clean.k.tests.test_k_kernels_emu import device



@unittest.skipUnless(os.environ.get("WG_K_GPU_BUILD"), "GPU only: set WG_K_GPU_BUILD and use the shared GPU lock")
class TestCompactHead(unittest.TestCase):
    def test_kernel_layout(self):
        """d0 alone, both accumulation forms: the compact map is the head map's channels 0, 1, 9 per slot (bits), the
        hidden history is the same, and positions outside the compact map are not written. 17 x 13 tokens: mirrored
        borders and an odd number of windows (the last G2 group has one)."""
        dev, funcs = device()
        rng = np.random.default_rng(7)
        w, h = 17, 13
        cw, ch = (w + 1) // 2, (h + 1) // 2
        x = rng.standard_normal((ch, cw, 64)).astype(np.float16)
        skip = rng.standard_normal((h, w, 32)).astype(np.float16)
        original = random_network_params(44)["d0"]
        special = np.resize(np.array([np.inf, -np.inf, np.nan, 0, -0.0, 1, -1, 65504, 2 ** -24], np.float16), 40)
        cases = [("random", original), ("special", dict(original, Wh=np.zeros_like(original["Wh"]), bh=special))]
        for label, params in cases:
            for full, compact in (("k_stage2_d0_hidden", "k_stage2_d0_hp"), ("k_stage2h_d0_hidden", "k_stage2h_d0_hp")):
                with self.subTest(case=label, kernel=compact), np.errstate(all="ignore"):
                    blob = k_pack.pack_stage("d0", params)
                    owned = []

                    def alloc(data=None, nbytes=None, fill=None):
                        raw = data.tobytes() if isinstance(data, np.ndarray) else data
                        n = max(256, len(raw) if raw is not None else nbytes)
                        ptr = dev.malloc(n)
                        owned.append(ptr)
                        if raw is not None:
                            dev.upload(ptr, raw)
                        if fill is not None:
                            dev.memset(ptr, fill, n)
                        return ptr
                    try:
                        xp, sp, wp = alloc(x), alloc(skip), alloc(blob)
                        nfull, ncomp = w * h * 80, w * h * 32
                        head = alloc(nbytes=nfull, fill=0x59)
                        comp = alloc(nbytes=ncomp + 256, fill=0xA6)  # 256 guard bytes behind the compact map
                        hh0, hh1 = alloc(nbytes=w * h * 8, fill=0x39), alloc(nbytes=w * h * 8, fill=0xC7)
                        for kernel, dst, hid in ((full, head, hh0), (compact, comp, hh1)):
                            _, grid, threads, args = k_launch.stage_fused_launch(
                                "d0", wp, {"in": xp, "skip": sp, "head": dst}, {"in": (cw, ch), "level": (w, h)},
                                (4, 4), windows=2)
                            dev.launch(funcs(kernel), grid, threads, args + [ctypes.c_void_p(hid)])
                        dev.sync()
                        hm = np.frombuffer(dev.download(head, nfull), np.uint16).reshape(h, w, 40)
                        raw = np.frombuffer(dev.download(comp, ncomp + 256), np.uint8)
                        got = raw[:ncomp].view(np.uint16).reshape(2 * h, 2 * w, 4)
                        np.testing.assert_array_equal(got, k_io.compact_head(hm))
                        self.assertTrue(np.all(raw[ncomp:] == 0xA6), "wrote behind the compact map")
                        np.testing.assert_array_equal(np.frombuffer(dev.download(hh0, w * h * 8), np.uint16),
                                                      np.frombuffer(dev.download(hh1, w * h * 8), np.uint16))
                    finally:
                        for ptr in reversed(owned):
                            dev.free(ptr)

    def test_output_kernel_all_variants(self):
        """k_out_pixel_c on the compact head against k_out_pixel on the head map, the output stage alone on the frames
        of test_k_io_emu (bright colours, luma history above the soft limit, logits beyond +-3, motion out of the
        frame, negative channels): all flag variants and a reset; out, L and Hc bit for bit."""
        from clean.k.tests.test_k_io_emu import frame, runner, same_bits
        r = runner()
        rng = np.random.default_rng(21)
        for hdr in (True, False):
            for mvlo in (True, False):
                for inv in (True, False):
                    for reset in (False, True):
                        if reset and not (hdr and mvlo):
                            continue
                        x = frame(rng, hdr, mvlo, inv, reset)
                        a = r.run("output", b"", x)
                        b = r.run("output@compact", b"", x)
                        for k in ("out", "L", "Hc"):
                            ok = same_bits(a[k], b[k])
                            self.assertTrue(ok.all(), f"hdr={hdr} mvlo={mvlo} inv={inv} reset={reset} {k}: "
                                                      f"{int((~ok).sum())} values differ")

    def test_frames(self):
        """Whole frames with and without the compact head on the same inputs: the game output and the three
        histories after every frame, bit for bit; all flag variants at a small odd size, and 1440p Quality."""
        dev, funcs = device()
        params = random_network_params(321)
        cases = [(v, (203, 117), (101, 64), dict(hdr=h, mv_lowres=m, depth_inverted=i), 4)
                 for v in ("fused", "fused16") for h in (True, False) for m in (True, False) for i in (True, False)]
        cases += [(v, (2560, 1440), (1707, 960), dict(hdr=True, mv_lowres=True, depth_inverted=True), 3)
                  for v in ("fused", "fused16")]
        rng = np.random.default_rng(5)
        for variant, size, render, flags, frames in cases:
            with self.subTest(variant=variant, size=size, flags=flags):
                a = KFrame(dev, funcs, size, render, params, flags, variant=variant, compact_head=False)
                b = KFrame(dev, funcs, size, render, params, flags, variant=variant, compact_head=True)
                try:
                    for k in range(frames):
                        x = synthetic_frame(rng, size, render, flags)
                        x.update(E_cur=0.8 + 0.17 * k, E_aux=1.2, jitter=(0.13 - 0.06 * k, -0.27 + 0.10 * k),
                                 reset=k == 0 or k == 2)
                        na = [n for _, ls in a._launches(a._frame_dict(x))[0] for n, _, _, _ in ls]
                        nb = [n for _, ls in b._launches(b._frame_dict(x))[0] for n, _, _, _ in ls]
                        self.assertIn("k_out_pixel", na)
                        self.assertNotIn("k_out_pixel_c", na)
                        self.assertIn("k_out_pixel_c", nb)
                        self.assertNotIn("k_out_pixel", nb)
                        self.assertIn("k_stage2_d0_hp" if variant == "fused" else "k_stage2h_d0_hp", nb)
                        oa, ob = a.frame(x), b.frame(x)
                        np.testing.assert_array_equal(oa["out"].view(np.uint32), ob["out"].view(np.uint32),
                                                      err_msg=f"frame {k}")
                        ha, hb = a.histories(), b.histories()
                        for key in ("Hc", "L", "Hh"):
                            np.testing.assert_array_equal(ha[key].view(np.uint32), hb[key].view(np.uint32),
                                                          err_msg=f"{key} frame {k}")
                finally:
                    a.release()
                    b.release()

    def test_plans(self):
        """Paths without the G2 d0 epilogue keep k_out_pixel and the head map; a debug frame returns the head map."""
        dev, funcs = device()
        flags = dict(hdr=True, mv_lowres=True, depth_inverted=True)
        params = random_network_params(3)
        rng = np.random.default_rng(1)
        for variant in ("fused_w1", "fused16_w1", "fused1"):
            kf = KFrame(dev, funcs, (65, 49), (43, 33), params, flags, variant=variant, base=(17, 13),
                        compact_head=True)
            try:
                x = synthetic_frame(rng, (65, 49), (43, 33), flags)
                names = [n for _, ls in kf._launches(kf._frame_dict(x))[0] for n, _, _, _ in ls]
                self.assertIn("k_out_pixel", names)
                self.assertIn("k_out_hidden", names)
                self.assertNotIn("k_out_pixel_c", names)
            finally:
                kf.release()
        a = KFrame(dev, funcs, (65, 49), (43, 33), params, flags, base=(17, 13), compact_head=False)
        b = KFrame(dev, funcs, (65, 49), (43, 33), params, flags, base=(17, 13), compact_head=True)
        try:
            x = synthetic_frame(rng, (65, 49), (43, 33), flags)
            oa, ob = a.frame(x, debug=True), b.frame(x, debug=True)
            np.testing.assert_array_equal(oa["N"].view(np.uint32), ob["N"].view(np.uint32))
            np.testing.assert_array_equal(oa["out"].view(np.uint32), ob["out"].view(np.uint32))
        finally:
            a.release()
            b.release()


if __name__ == "__main__":
    unittest.main()
