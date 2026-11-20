# SPDX-License-Identifier: MIT
"""The gfx11 (RDNA 3) platform helpers of hlsl/wg_gfx11.hlsli (WG_GFX 11), run headless on the card through
vkd3d-proton under Wine and compared with numpy and with a small HIP kernel on the same card (the native
v_wmma_f32_16x16x16_f16 of gfx1100): fragment layout of the raw F16 matrix instruction, the K = 16 step as two chained
instructions, the accumulator row table, the F32 -> F16 conversion and the half-wave exchange. The layout and the
numbers are those measured on the card (RX 7900 XT, vkd3d-proton with extended emulation, RADV).

The card tests need a prefix and tools; each test is skipped without them (the compile checks need only DXC):
  WG_GFX11_WORK      scratch folder (never a memory-backed one) for the DXIL, the host and the data
  WG_GFX11_WRUN      runner script: `<script> -- <exe> <args>` runs a Windows program headless under Wine with the
                     right prefix, holds the GPU lock and refuses any display (a window must never open)
  WG_GFX11_VKD3D     folder with d3d12.dll and d3d12core.dll of a vkd3d-proton built with extended emulation
  WG_GFX11_MINGW     llvm-mingw g++ command (builds gfx11_probe_host.cpp; or WG_GFX11_HOST = a built host.exe)
  WG_DXC_DIR         the pinned DXC (as build_dx12.py), and
  WG_GFX11_ASM       the dxil_asm tool (the one build_dx12.py builds)
  WG_GFX11_HIP_RUN   command prefix that runs a program on the card under the GPU lock (HIP reference tests; hipcc
                     on the PATH or WG_GFX11_HIPCC)
"""
import os
import shlex
import shutil
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
sys.path.insert(0, DX12)
import build_dx12 as B  # noqa: E402

WORK = os.environ.get("WG_GFX11_WORK")
WRUN = os.environ.get("WG_GFX11_WRUN")
VKD3D = os.environ.get("WG_GFX11_VKD3D")
MINGW = os.environ.get("WG_GFX11_MINGW")
HOST = os.environ.get("WG_GFX11_HOST")
ASM = os.environ.get("WG_GFX11_ASM")
HIP_RUN = os.environ.get("WG_GFX11_HIP_RUN")
HIPCC = os.environ.get("WG_GFX11_HIPCC", "hipcc")
HAVE_DXC = bool(WORK and ASM and os.path.exists(os.path.join(B.DXC_DIR, "dxc.exe")) and os.path.exists(ASM))
HAVE_CARD = bool(HAVE_DXC and WRUN and VKD3D and (MINGW or HOST))
SRC = os.path.join(HERE, "gfx11_helpers.hlsl")
KERNELS = ("k8", "k16", "f16_gfx12_shape", "convert", "xor16", "d_row", "xor16_after_mma", "round_modes")


def compile_kernel(entry, gfx=11, src=SRC, out=None, defines=None):
    out = out or os.path.join(WORK, f"{os.path.basename(src)[:-5]}_{entry}_g{gfx}.dxil")
    defs = {"WG_BLOCK": 32, "WG_WAVE": 32, **({"WG_GFX": gfx} if gfx != 12 else {}), **(defines or {})}
    B.compile_hlsl(src, entry, out, defines=defs, includes=[HERE], asm_tool=ASM)
    return out


# ---- the fragment model (numpy): the measured layout ---------------------------------------------------------
def frag_replicated(A, B_, C):
    """Logical A (N, 16, 8), B (N, 8, 16), C (N, 16, 16) -> lane fragments of the raw instruction, K 0 .. 7: lane l holds
    row (A) / column (B) l % 16, so both halves hold the same values (A: the instruction reads row r from lane
    16 (r & 1) + r, B: lane l itself). Fragments (N, 32, 8) f16 and the accumulator (N, 8, 32): element i of lane l is
    row 2 i + l / 16, column l % 16."""
    n = A.shape[0]
    fa = np.zeros((n, 32, 8), np.float16)
    fb = np.zeros((n, 32, 8), np.float16)
    for l in range(32):
        fa[:, l] = A[:, l % 16]
        fb[:, l] = B_[:, :, l % 16]
    return fa, fb, acc_to_lanes(C)


def acc_to_lanes(C):
    fc = np.zeros((C.shape[0], 8, 32), np.float32)
    for i in range(8):
        for h in range(2):
            fc[:, i, 16 * h:16 * h + 16] = C[:, 2 * i + h, :]
    return fc


def lanes_to_acc(D):
    out = np.zeros((D.shape[0], 16, 16), np.float32)
    for i in range(8):
        for h in range(2):
            out[:, 2 * i + h, :] = D[:, i, 16 * h:16 * h + 16]
    return out


def frag_gfx12_shape(A):
    """Logical A (N, 16, 16) -> the gfx12 shape (lane l: row l % 16, half h holds K 8 h .. 8 h + 7); for B pass the
    transpose: lane l then holds column l % 16."""
    n = A.shape[0]
    f = np.zeros((n, 32, 8), np.float16)
    for l in range(32):
        f[:, l] = A[:, l % 16, 8 * (l // 16):8 * (l // 16) + 8]
    return f


def exact(A, B_, C):
    return (np.einsum("nik,nkj->nij", A.astype(np.float64), B_.astype(np.float64)) + C).astype(np.float32)


def bits_equal(x, y):
    return float(np.mean(x.astype(np.float32).view(np.uint32) == y.astype(np.float32).view(np.uint32)))


class Card:
    """Compile once, run on the card through the Wine runner."""
    host = None

    @classmethod
    def setup(cls):
        os.makedirs(WORK, exist_ok=True)
        if HOST:
            cls.host = HOST
        else:
            cls.host = os.path.join(WORK, "gfx11_probe_host.exe")
            if not os.path.exists(cls.host) or os.path.getmtime(cls.host) < os.path.getmtime(
                    os.path.join(HERE, "gfx11_probe_host.cpp")):
                subprocess.run([*shlex.split(MINGW), "-O1", "-o", cls.host, os.path.join(HERE, "gfx11_probe_host.cpp")],
                               check=True)
        for f in ("d3d12.dll", "d3d12core.dll"):   # next to the host: the native override of Wine's own
            dst = os.path.join(os.path.dirname(cls.host), f)
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(VKD3D, f), dst)
        cls.dxil = {k: compile_kernel(k) for k in KERNELS}

    @classmethod
    def run(cls, kernel, arena, args, groups=1, tag=None, out_bytes=None):
        """arena: bytes -> the arena after the run (np.uint8), or None with the host's text when the run failed."""
        nb = max((len(arena) + 4095) & ~4095, (out_bytes or 0) + 4095 & ~4095)
        fi, fo = os.path.join(WORK, f"{tag or kernel}.in"), os.path.join(WORK, f"{tag or kernel}.out")
        with open(fi, "wb") as f:
            f.write(arena)
        if os.path.exists(fo):
            os.remove(fo)
        r = subprocess.run([WRUN, "--", cls.host, cls.dxil[kernel], fi, fo, str(nb), str(groups), *[hex(a) for a in args]],
                           capture_output=True, text=True, timeout=1800, cwd=WORK)
        txt = r.stdout + r.stderr
        ok = "DONE" in txt and os.path.exists(fo)
        return (np.fromfile(fo, dtype=np.uint8) if ok else None), txt

    @classmethod
    def mma(cls, kernel, fa, fb, fc, fa2=None, fb2=None):
        """Matrix kernels (k8, k16, f16_gfx12_shape): the layout of lib.py of the P0 probes. Returns D (N, 8, 32)."""
        n = fa.shape[0]
        parts = [fa.tobytes(), fb.tobytes(), fc.astype(np.float32).tobytes(), bytes(n * 1024)]
        o = [0, n * 512, n * 1024, n * 2048, n]
        if fa2 is not None:
            o += [n * 3072, n * 3072 + n * 512]
            parts += [fa2.tobytes(), fb2.tobytes()]
        out, txt = cls.run(kernel, b"".join(parts), o, tag=kernel)
        if out is None:
            raise AssertionError(f"{kernel}: no result\n{txt[-400:]}")
        return out[n * 2048:n * 3072].view(np.float32).reshape(n, 8, 32)


def hip_ref(A, B_, C):
    """D = A x B + C (K = 16) by the native instruction of the gfx1100 card. A (N,16,16), B (N,16,16) f16, C f32."""
    exe = os.path.join(WORK, "gfx11_hip_ref")
    src = os.path.join(HERE, "gfx11_hip_ref.hip")
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.run([HIPCC, "--offload-arch=gfx1100", "-O1", "-w", "-lamdhip64", "-o", exe, src], check=True)
    n = A.shape[0]
    fi, fo = os.path.join(WORK, "hip_ref.in"), os.path.join(WORK, "hip_ref.out")
    with open(fi, "wb") as f:
        f.write(np.int32(n).tobytes() + A.astype(np.float16).tobytes() + B_.astype(np.float16).tobytes()
                + C.astype(np.float32).tobytes())
    r = subprocess.run([*shlex.split(HIP_RUN), exe, fi, fo], capture_output=True, text=True, timeout=1800)
    assert "gfx1100" in r.stderr, r.stderr
    return np.fromfile(fo, np.float32).reshape(n, 16, 16)


def need_hip(fn):
    return unittest.skipUnless(HAVE_CARD and HIP_RUN, "no card / HIP reference configured (module doc)")(fn)


@unittest.skipUnless(HAVE_CARD, "no card configured (module doc)")
class TestGfx11HelpersOnCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Card.setup()
        cls.rng = np.random.default_rng(20261115)

    def k8_inputs(self, gen, n=64, with_c=False):
        A, B_ = gen((n, 16, 8)), gen((n, 8, 16))
        C = self.rng.normal(0, 1, (n, 16, 16)).astype(np.float32) if with_c else np.zeros((n, 16, 16), np.float32)
        return A.astype(np.float16), B_.astype(np.float16), C

    def test_k8_layout_exact_on_positive_integers(self):
        ints = lambda s: self.rng.integers(0, 5, s)
        A, B_, C = self.k8_inputs(ints)
        D = lanes_to_acc(Card.mma("k8", *frag_replicated(A, B_, C)))
        self.assertEqual(bits_equal(D, exact(A, B_, C)), 1.0)

    def test_k8_unread_lanes_do_not_matter(self):
        # the instruction reads row r of A from lane 16 (r & 1) + r only: a fragment with every other lane zero is the same
        A, B_, C = self.k8_inputs(lambda s: self.rng.normal(0, 1, s), with_c=True)
        fa, fb, fc = frag_replicated(A, B_, C)
        fa_min = np.zeros_like(fa)
        for r in range(16):
            fa_min[:, r + 16 * (r & 1)] = fa[:, r + 16 * (r & 1)]
        full = Card.mma("k8", fa, fb, fc)
        minimal = Card.mma("k8", fa_min, fb, fc)
        self.assertEqual(bits_equal(minimal, full), 1.0)

    def test_k8_close_to_float64_on_random_data(self):
        A, B_, C = self.k8_inputs(lambda s: self.rng.normal(0, 1, s), with_c=True)
        D = lanes_to_acc(Card.mma("k8", *frag_replicated(A, B_, C)))
        self.assertLess(float(np.abs(D - exact(A, B_, C)).max()), 1e-5)

    def test_gfx12_shaped_fragments_are_wrong_for_the_raw_instruction(self):
        # a probe that can fail: the gfx12 shape (half = K block) fed to the raw instruction does not give A x B
        A, B_, C = self.k8_inputs(lambda s: self.rng.integers(0, 5, s))
        A2, B2, _ = self.k8_inputs(lambda s: self.rng.integers(0, 5, s))   # K 8 .. 15: other values, so a mix-up shows
        A16 = np.concatenate([A, A2], axis=2)
        B16 = np.concatenate([B_, B2], axis=1)
        D = lanes_to_acc(Card.mma("k8", frag_gfx12_shape(A16), frag_gfx12_shape(B16.transpose(0, 2, 1)),
                                  np.zeros((64, 8, 32), np.float32)))
        self.assertLess(bits_equal(D, exact(A, B_, C)), 0.75)   # about half of the rows read the right K block

    @need_hip
    def test_k8_matches_hip_bit_for_bit(self):
        A, B_, C = self.k8_inputs(lambda s: self.rng.normal(0, 1, s), with_c=True)
        D = lanes_to_acc(Card.mma("k8", *frag_replicated(A, B_, C)))
        A16 = np.concatenate([A, np.zeros_like(A)], axis=2)
        B16 = np.concatenate([B_, np.zeros_like(B_)], axis=1)
        self.assertEqual(bits_equal(D, hip_ref(A16, B16, C)), 1.0)

    def k16(self, A, B_, C):
        """The K = 16 step as two chained instructions: -> (D, the HIP result, the exact one)."""
        fa0, fb0, fc = frag_replicated(A[:, :, :8], B_[:, :8, :], C)
        fa1, fb1, _ = frag_replicated(A[:, :, 8:], B_[:, 8:, :], C)
        return lanes_to_acc(Card.mma("k16", fa0, fb0, fc, fa1, fb1)), exact(A, B_, C)

    def test_k16_chain_exact_on_positive_integers(self):
        A = self.rng.integers(0, 5, (64, 16, 16)).astype(np.float16)
        B_ = self.rng.integers(0, 5, (64, 16, 16)).astype(np.float16)
        C = np.zeros((64, 16, 16), np.float32)
        D, X = self.k16(A, B_, C)
        self.assertEqual(bits_equal(D, X), 1.0)

    def test_k16_chain_second_half_matters(self):
        # K 8 .. 15 only: the chain must add the second instruction's product to the first's result
        A = np.zeros((64, 16, 16), np.float16)
        B_ = np.zeros((64, 16, 16), np.float16)
        A[:, :, 8:] = self.rng.integers(1, 5, (64, 16, 8))
        B_[:, 8:, :] = self.rng.integers(1, 5, (64, 8, 16))
        D, X = self.k16(A, B_, np.zeros((64, 16, 16), np.float32))
        self.assertEqual(bits_equal(D, X), 1.0)
        self.assertGreater(float(np.abs(X).min()), 0.0)

    def test_k16_close_to_float64_on_random_data(self):
        A = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float16)
        B_ = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float16)
        C = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float32)
        D, X = self.k16(A, B_, C)
        self.assertLess(float(np.abs(D - X).max()), 1e-5)

    @need_hip
    def test_k16_chain_matches_the_hip_instruction_bit_for_bit(self):
        for gen in (lambda s: self.rng.integers(-4, 5, s), lambda s: self.rng.normal(0, 1, s)):
            A, B_ = gen((64, 16, 16)).astype(np.float16), gen((64, 16, 16)).astype(np.float16)
            C = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float32)
            D, _ = self.k16(A, B_, C)
            self.assertEqual(bits_equal(D, hip_ref(A, B_, C)), 1.0)

    @need_hip
    def test_wg_wmma_f16_in_the_gfx12_shape_matches_hip(self):
        A = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float16)
        B_ = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float16)
        C = self.rng.normal(0, 1, (64, 16, 16)).astype(np.float32)
        D = lanes_to_acc(Card.mma("f16_gfx12_shape", frag_gfx12_shape(A), frag_gfx12_shape(B_.transpose(0, 2, 1)),
                                  acc_to_lanes(C)))
        self.assertEqual(bits_equal(D, hip_ref(A, B_, C)), 1.0)

    def test_wg_wmma_f16_in_the_gfx12_shape_close_to_float64(self):
        A = self.rng.integers(0, 5, (64, 16, 16)).astype(np.float16)
        B_ = self.rng.integers(0, 5, (64, 16, 16)).astype(np.float16)
        C = np.zeros((64, 16, 16), np.float32)
        D = lanes_to_acc(Card.mma("f16_gfx12_shape", frag_gfx12_shape(A), frag_gfx12_shape(B_.transpose(0, 2, 1)),
                                  acc_to_lanes(C)))
        self.assertEqual(bits_equal(D, exact(A, B_, C)), 1.0)

    def test_convert_is_round_to_nearest_even(self):
        r = np.random.default_rng(5)
        x = np.concatenate([
            r.normal(0, 1, 100) * 2.0 ** r.integers(-14, 10, 100),
            1 + np.array([3, 1, 5, 7, 9, 1023, 2047, 2049]) * 2.0 ** -12,   # not ties
            1 + np.array([1, 3, 5, 7]) * 2.0 ** -11,                          # exact ties
            -(1 + np.array([3, 1, 5, 7]) * 2.0 ** -12),
            r.uniform(-70000, 70000, 40), 2.0 ** r.uniform(-26, -14, 30),    # overflow, subnormal halves
            [0.0, -0.0, np.inf, -np.inf, 65504, 65519.9, 65520, 1e-8],
        ]).astype(np.float32)
        x = np.resize(x, 256).astype(np.float32).reshape(8, 32)               # element i of lane l at (i, l)
        out, txt = Card.run("convert", x.tobytes() + bytes(4096), [0, 0, 0, 4096, 1, 0, 0])
        self.assertIsNotNone(out, txt[-400:])
        h = out[4096:4096 + 512].view(np.uint32).reshape(32, 4).view(np.uint16).reshape(32, 8).T
        want = x.astype(np.float16).view(np.uint16)
        self.assertEqual(int(np.sum(h != want)), 0)

    def test_rounding_modes_in_a_matrix_shader(self):
        # RADV: (half) is round to nearest even, f32tof16 toward zero (wg_gelu_lut_off2 expects the latter); the driver's
        # conversion is round to nearest even
        r = np.random.default_rng(8)
        x = np.concatenate([r.normal(0, 1, 120) * 2.0 ** r.integers(-14, 10, 120),
                            1 + np.array([3, 1, 5, 7, 9, 1023, 2047, 2049]) * 2.0 ** -12,
                            1 + np.array([1, 3, 5, 7]) * 2.0 ** -11, r.uniform(-60000, 60000, 60),
                            2.0 ** r.uniform(-26, -14, 30), [0.0, -0.0, 65504.0, 1e-8]]).astype(np.float32)
        x = np.resize(x, 256).astype(np.float32).reshape(8, 32)
        out, txt = Card.run("round_modes", x.tobytes() + bytes(4096 - 1024) + bytes(8192), [0, 0, 0, 4096, 1, 0, 0],
                            out_bytes=4096 + 5120)
        self.assertIsNotNone(out, txt[-400:])
        reg = lambda i: out[4096 + 1024 * i:4096 + 1024 * (i + 1)].view(np.uint32).reshape(8, 32)
        rne = x.astype(np.float16)
        away = np.abs(rne.astype(np.float64)) > np.abs(x.astype(np.float64))
        rtz = np.where(away, np.nextafter(rne, np.float16(0)), rne).view(np.uint16)
        rne = rne.view(np.uint16)
        self.assertGreater(int(np.sum(rne != rtz)), 20)   # the data tells the two modes apart
        self.assertTrue(np.array_equal(reg(0) & 0xffff, rtz), "f32tof16 is round toward zero")
        self.assertTrue(np.array_equal(reg(1) & 0xffff, rne), "(half) is round to nearest even")
        self.assertTrue(np.array_equal(reg(2) & 0xffff, rtz), "asfloat16(f32tof16) is round toward zero")
        h = out[4096 + 3072:4096 + 3072 + 512].view(np.uint32).reshape(32, 4).view(np.uint16).reshape(32, 8).T
        self.assertTrue(np.array_equal(h, rne), "the driver's convert is round to nearest even")

    def test_xor16_reads_the_other_half(self):
        v = self.rng.normal(0, 1, 32).astype(np.float32)
        w = self.rng.integers(0, 2 ** 32, (32, 4), dtype=np.uint64).astype(np.uint32)
        out, txt = Card.run("xor16", v.tobytes() + bytes(4096 - 128) + w.tobytes(), [0, 4096, 0, 8192, 1, 0, 0],
                            out_bytes=8192 + 128 + 512)
        self.assertIsNotNone(out, txt[-400:])
        lanes = np.arange(32) ^ 16
        self.assertTrue(np.array_equal(out[8192:8192 + 128].view(np.float32), v[lanes]))
        self.assertTrue(np.array_equal(out[8192 + 128:8192 + 128 + 512].view(np.uint32).reshape(32, 4), w[lanes]))

    def test_d_row_is_2i_plus_half(self):
        out, txt = Card.run("d_row", bytes(4096), [0, 0, 0, 0, 1, 0, 0])
        self.assertIsNotNone(out, txt[-400:])
        got = out[:1024].view(np.uint32).reshape(8, 32)
        want = 2 * np.arange(8)[:, None] + (np.arange(32)[None, :] // 16)
        self.assertTrue(np.array_equal(got, want))

    def test_xor16_after_a_matrix_instruction_sums_rows_2i_and_2i_plus_1(self):
        A, B_, C = self.k8_inputs(lambda s: self.rng.normal(0, 1, s), n=1, with_c=True)
        fa, fb, fc = frag_replicated(A, B_, C)
        out, txt = Card.run("xor16_after_mma", fa.tobytes() + fb.tobytes() + fc.tobytes() + bytes(4096),
                            [0, 512, 1024, 2048, 1, 0, 0])
        self.assertIsNotNone(out, txt[-400:])
        G = exact(A, B_, C).astype(np.float64)[0]
        D = out[2048:2048 + 1024].view(np.float32).reshape(8, 32)
        want = np.zeros((8, 32))
        for i in range(8):
            for h in range(2):
                want[i, 16 * h:16 * h + 16] = G[2 * i + h] + G[2 * i + 1 - h]
        self.assertLess(float(np.abs(D - want).max()), 1e-4)


@unittest.skipUnless(HAVE_DXC, "needs WG_GFX11_WORK, WG_DXC_DIR, WG_GFX11_ASM (module doc)")
class TestGfx11Compile(unittest.TestCase):
    def kernel_file(self, body):
        path = os.path.join(WORK, "compile_check.hlsl")
        os.makedirs(WORK, exist_ok=True)
        with open(path, "w") as f:
            f.write('#include "wg_gfx11.hlsli"\nWG_ARGS_BEGIN\n    uint po;\nWG_ARGS_END\nWG_KERNEL(k)\n{\n'
                    "    WG_SETUP();\n" + body + "\n}\n")
        return path

    def test_there_is_no_fp8_instruction_on_gfx11(self):
        for body in ("v8f z = wg_wmma_fp8(uint2(0, 0), uint2(0, 0), wg_splat(0.0f)); wg_stf(po, 0u, z.v[0]);",
                     "uint2 q = wg_pack_q8_ags(wg_splat(1.0f)); wg_st64(po, 0u, q);",
                     "wg_st32(po, 0u, WG_AGS_FP8_F32);"):
            with self.assertRaises(SystemExit, msg=body):
                compile_kernel("k", 11, self.kernel_file(body), out=os.path.join(WORK, "compile_check.dxil"))

    def test_the_same_source_builds_on_gfx12(self):
        body = "v8f z = wg_wmma_fp8(uint2(0, 0), uint2(0, 0), wg_splat(0.0f)); wg_stf(po, 0u, z.v[0]);"
        compile_kernel("k", 12, self.kernel_file(body), out=os.path.join(WORK, "compile_check.dxil"))

    def test_software_pack_builds_on_gfx11(self):
        body = "uint2 q = wg_pack_q8(wg_splat(1.0f)); wg_st64(po, 0u, q);"
        compile_kernel("k", 11, self.kernel_file(body), out=os.path.join(WORK, "compile_check.dxil"))

    def test_unsupported_family_is_refused(self):
        body = "wg_stf(po, 0u, 1.0f);"
        with self.assertRaises(SystemExit):
            compile_kernel("k", 10, self.kernel_file(body), out=os.path.join(WORK, "compile_check.dxil"))


if __name__ == "__main__":
    unittest.main()
