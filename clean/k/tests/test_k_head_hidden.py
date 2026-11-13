# SPDX-License-Identifier: MIT
"""Exact d0 head/hidden fusion against the unchanged standalone hidden kernel."""
import ctypes
import os
import unittest

import numpy as np

from clean.k.reference.network import random_network_params
from clean.k.runner import k_io, k_launch, k_pack
from clean.k.runner.k_bench import synthetic_frame
from clean.k.runner.k_frame import KFrame
from clean.k.tests.test_k_io_emu import same_bits
from clean.k.tests.test_k_kernels_emu import device


@unittest.skipUnless(os.environ.get('WG_K_GPU_BUILD'), 'GPU code-object regression; run under shared GPU lock')
class TestHeadHiddenFusion(unittest.TestCase):
    def test_frames_and_fallback_plans(self):
        dev, funcs = device()
        flags = dict(hdr=True, mv_lowres=True, depth_inverted=True)
        params = random_network_params(123)
        cases = [('fused', (65, 49), (43, 33), (17, 13), 8),
                 ('fused', (2560, 1440), (1707, 960), None, 8),
                 ('fused_w1', (65, 49), (43, 33), (17, 13), 2),
                 ('fused16', (65, 49), (43, 33), (17, 13), 8),
                 ('fused16', (2560, 1440), (1707, 960), None, 4),
                 ('fused16_w1', (65, 49), (43, 33), (17, 13), 2)]
        rng = np.random.default_rng(91)
        for variant, size, render, base, frames in cases:
            with self.subTest(variant=variant, size=size):
                # the 40-channel head map (not the compact pixel head): this test reads it back for k_out_hidden
                kf = KFrame(dev, funcs, size, render, params, flags, variant=variant, base=base, compact_head=False)
                nbytes = kf.Bw * kf.Bh * 8
                ref = dev.malloc(nbytes)
                try:
                    for phase in range(frames):
                        x = synthetic_frame(rng, size, render, flags)
                        x.update(E_cur=0.8 + 0.17 * phase, E_aux=1.2, pre_exposure=0.9,
                                 jitter=(0.13 - 0.06 * phase, -0.27 + 0.10 * phase), reset=phase == 0)
                        parts, _ = kf._launches(kf._frame_dict(x))
                        names = [n for _, launches in parts for n, _, _, _ in launches]
                        if variant in ('fused', 'fused16'):
                            self.assertIn('k_stage2_d0_hidden' if variant == 'fused' else 'k_stage2h_d0_hidden', names)
                            self.assertNotIn('k_out_hidden', names)
                            self.assertEqual(len(parts[1][1]), 11)
                            self.assertEqual(len(parts[2][1]), 1)
                        else:
                            self.assertNotIn('k_stage2_d0_hidden', names)
                            self.assertNotIn('k_stage2h_d0_hidden', names)
                            self.assertIn('k_out_hidden', names)
                            self.assertEqual(len(parts[1][1]), 11)
                            self.assertEqual(len(parts[2][1]), 2)
                        result = kf.frame(x)
                        self.assertEqual(result['f'], phase)
                        emitted = dev.download(kf.bufs[f'Hh{kf.parity}'], nbytes)
                        dev.memset(ref, 0xA5, nbytes)
                        p = k_io.KOutParams(f=k_io.KFrameIn(Bw=kf.Bw, Bh=kf.Bh),
                                           N=kf.net.bufs['OUT'], Hhout=ref)
                        dev.launch(funcs('k_out_hidden'), (kf.Bw * kf.Bh + 127) // 128, 128, [p])
                        dev.sync()
                        expected = dev.download(ref, nbytes)
                        np.testing.assert_array_equal(np.frombuffer(emitted, np.uint16),
                                                      np.frombuffer(expected, np.uint16),
                                                      err_msg=f'{variant} {size} phase {phase}')
                finally:
                    dev.free(ref)
                    kf.release()

    def test_cross_lane_large_and_nonfinite_heads(self):
        dev, funcs = device()
        rng = np.random.default_rng(32)
        w, h = 17, 13  # 3x3 windows: odd final G2 group and mirrored borders
        cw, ch = (w + 1) // 2, (h + 1) // 2
        x = rng.standard_normal((ch, cw, 64)).astype(np.float16)
        skip = rng.standard_normal((h, w, 32)).astype(np.float16)
        original = random_network_params(44)['d0']
        patterns = [np.arange(40, dtype=np.float16) / np.float16(16),
                    np.resize(np.array([65504, -65504, 2**-24, -0.0, 1], np.float16), 40),
                    np.resize(np.array([np.inf, -np.inf, np.nan, 0, -0.0, 1, -1], np.float16), 40)]
        for index, bias in enumerate(patterns):
            with self.subTest(pattern=index), np.errstate(all='ignore'):
                params = dict(original, Wh=np.zeros_like(original['Wh']), bh=bias)
                blob = k_pack.pack_stage('d0', params)
                owned = []
                def alloc(data=None, nbytes=None):
                    raw = data.tobytes() if isinstance(data, np.ndarray) else data
                    ptr = dev.malloc(max(256, len(raw) if raw is not None else nbytes))
                    owned.append(ptr)
                    if raw is not None:
                        dev.upload(ptr, raw)
                    return ptr
                try:
                    xp, sp, wp = alloc(x), alloc(skip), alloc(blob)
                    head0, head1 = alloc(nbytes=w*h*80), alloc(nbytes=w*h*80)
                    hh0, hh1 = alloc(nbytes=w*h*8), alloc(nbytes=w*h*8)
                    dev.memset(head0, 0x59, w*h*80)
                    dev.memset(head1, 0xA6, w*h*80)
                    dev.memset(hh0, 0x39, w*h*8)
                    dev.memset(hh1, 0xC7, w*h*8)
                    launches = [k_launch.stage_fused_launch('d0', wp,
                                {'in': xp, 'skip': sp, 'head': dst},
                                {'in': (cw, ch), 'level': (w, h)}, (4, 4), windows=2)
                                for dst in (head0, head1)]
                    name, grid, threads, args = launches[0]
                    dev.launch(funcs(name), grid, threads, args)
                    _, grid, threads, args = launches[1]
                    dev.launch(funcs('k_stage2_d0_hidden'), grid, threads,
                               args + [ctypes.c_void_p(hh1)])
                    p = k_io.KOutParams(f=k_io.KFrameIn(Bw=w, Bh=h), N=head0, Hhout=hh0)
                    dev.launch(funcs('k_out_hidden'), (w*h+127)//128, 128, [p])
                    dev.sync()
                    a = np.frombuffer(dev.download(head0, w*h*80), np.float16)
                    b = np.frombuffer(dev.download(head1, w*h*80), np.float16)
                    self.assertTrue(same_bits(a, b).all(), 'head changed')
                    a = np.frombuffer(dev.download(hh0, w*h*8), np.float16)
                    b = np.frombuffer(dev.download(hh1, w*h*8), np.float16)
                    self.assertTrue(same_bits(a, b).all(), 'hidden changed')
                    # Existing reference convention compares NaNs as equivalent;
                    # signed zeros and every finite/Inf bit remain significant.
                finally:
                    for ptr in reversed(owned):
                        dev.free(ptr)


if __name__ == '__main__':
    unittest.main()
