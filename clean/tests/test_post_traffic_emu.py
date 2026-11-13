# SPDX-License-Identifier: MIT
"""post_traffic_* (timing-only memory-traffic kernels, runner/post_traffic.py) in the host lane emulator: every layout
runs at a small size, writes its whole history, display and output, and nothing past the end of its buffers.

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.runner import post as PS
from clean.runner import post_traffic as T
from clean.tests.test_kernels_emu import EmuTest


class TestTrafficEmu(EmuTest):
    def test_layouts(self):
        dev = self.runner.dev
        sizes = (13, 7, 30, 17, 20, 11)   # rw, rh, dw, dh, ow, oh
        rw, rh, dw, dh, ow, oh = sizes
        n = dw * dh
        pad = 64
        for name, (_, fk, hist, disp, outb) in T.LAYOUTS.items():
            with self.subTest(layout=name):
                nbytes = {"color": rw * rh * 16, "A": rw * rh * 4, "B": rw * rh * 4, "C": rw * rh * 8,
                          "m_near": rw * rh * 4, "m_center": rw * rh * 4, "m_far": rw * rh * 4,
                          "hist_prev": n * hist, "hist_out": n * hist, "reduced": (n + 3) // 4 * 4,
                          "display": n * 6, "out": ow * oh * outb}
                bufs = {k: dev.malloc(v + pad) for k, v in nbytes.items()}
                rng = np.random.default_rng(5)
                for k, v in nbytes.items():
                    if k in ("hist_out", "reduced", "display", "out"):
                        dev.memset(bufs[k], 0x5A, v + pad)
                    else:
                        dev.upload(bufs[k], rng.integers(0, 256, v + pad, dtype=np.uint8).tobytes())
                for kname, grid, threads, args in T.traffic_launches(sizes, name, bufs):
                    dev.launch(PS._function(self.runner, kname), grid, threads, args)
                for k in ("hist_out", "out") + (("display",) if disp else ()):
                    raw = np.frombuffer(dev.download(bufs[k], nbytes[k] + pad), np.uint8)
                    self.assertTrue(np.all(raw[nbytes[k]:] == 0x5A), f"{k}: written past the end")
                    words = raw[:nbytes[k]].view(np.uint16)
                    self.assertGreater(np.mean(words != 0x5A5A), 0.9, f"{k}: not written")
                if not disp:
                    raw = np.frombuffer(dev.download(bufs["display"], 16), np.uint8)
                    self.assertTrue(np.all(raw == 0x5A), "fused layout wrote the display")
                for p in bufs.values():
                    dev.free(p)

    def test_run_small(self):
        rep = T.run(self.runner, (13, 7, 30, 17, 20, 11), repeats=1, rounds=1, reference=False,
                    flush_bytes=1 << 12)
        self.assertEqual(set(rep["layouts"]), set(T.LAYOUTS))
        self.assertTrue(all("cold_ms" in v for v in rep["layouts"].values()))
