# SPDX-License-Identifier: MIT
"""The native RDNA 3 K = 16 operands of hlsl/wg_gfx11.hlsli (slices S1 / S2 of the native RDNA 3 work): one-hot probes of
the row map and the operand forms, run on the CPU through the emulation of the kernel text (clean/dx12/emu: the lane
emulator's model of the measured gfx11 F16 instruction), with negative controls (a wrong row table, the v^T fix left
out). The card runs of the same kernels (tests/gfx11_k16_probe.hlsl) come later (the GPU test list of the native work).

Needs Linux and clang++ (as the emulator tests of model K). The translator checks (one cooperative-matrix multiply per
K = 16 step, no lane exchange) need WG_GFX11_DXIL_SPIRV = the dxil-spirv command line tool built with the K = 16 patch,
spirv-dis, DXC (WG_DXC_DIR, as build_dx12.py) and WG_GFX11_ASM (build_dx12.py's dxil_asm tool); they are skipped
without them. WG_GFX11_DXIL_SPIRV_STOCK (the tool without the patch) adds the refusal check.
"""
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
sys.path.insert(0, DX12)
import build_dx12 as B  # noqa: E402
from emu import hlsl_emu as H  # noqa: E402

SRC = os.path.join(HERE, "gfx11_k16_probe.hlsl")
NAMES = ["nat_ldA", "nat_b16", "nat_vt", "nat_vt_nofix", "k16n", "k16_chain", "k16_xchg", "k64n"]
_LIBS = {}


def work_dir():
    d = os.environ.get("WG_TEST_TMP")
    return tempfile.mkdtemp(prefix="wg_gfx11_native_", dir=d)


def device(extra=None):
    key = tuple(sorted((extra or {}).items()))
    if key not in _LIBS:
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("the lane emulator needs Linux (ucontext)")
        try:
            H.compiler()
        except RuntimeError as e:
            raise unittest.SkipTest(str(e))
        lib, ent = H.build(SRC, NAMES, work_dir(), gfx=11, extra_defines=extra)
        _LIBS[key] = H.HlslEmu({n: (lib, ent[n]) for n in NAMES}, arena_bytes=8 << 20)
    return _LIBS[key]


# ---- the layouts (numpy) -------------------------------------------------------------------------------------------
def tile_gfx12(M):
    """Logical M (N, 16, 16) as rows of A (or, with M = B^T, columns of B) -> tiles (N, 32, 8) in the gfx12 fragment order:
    slot 16 h + r holds row r, K 8 h .. 8 h + 7."""
    t = np.zeros((M.shape[0], 32, 8), M.dtype)
    for s in range(32):
        t[:, s] = M[:, s % 16, 8 * (s // 16):8 * (s // 16) + 8]
    return t


def acc_gfx12(C):
    """Logical C (N, 16, 16) -> (N, 32, 8) lane-major: lane l, element i = row 8 h + i, column l % 16."""
    a = np.zeros((C.shape[0], 32, 8), np.float32)
    for l in range(32):
        a[:, l] = C[:, 8 * (l // 16):8 * (l // 16) + 8, l % 16]
    return a


def arena_tiles(A, Bm, C):
    """The probe arena of the tile kernels (nat_*): pa, pb (gfx12 tiles), pc (gfx12 accumulator), po (output, 0x7F)."""
    n = A.shape[0]
    parts = [tile_gfx12(A.astype(np.float16)).tobytes(), tile_gfx12(np.swapaxes(Bm, 1, 2).astype(np.float16)).tobytes(),
             acc_gfx12(C).tobytes(), b"\x7f" * (n * 1024)]
    return b"".join(parts), [0, n * 512, n * 1024, n * 2048, n]


def arena_channel(A, Bm, C, gfx12_shape=False):
    """The probe arena of the channel kernels (k16n, k16_chain, k16_xchg, k64n): the instruction's own operands (lane l:
    K 0 .. 7 at 32 l, K 8 .. 15 at 32 l + 16; gfx12_shape: lane l's K 8 h .. 8 h + 7 first) and its accumulator layout
    (element i of lane l: row 2 i + l / 16)."""
    n = A.shape[0]
    fa = np.zeros((n, 32, 16), np.float16)
    fb = np.zeros((n, 32, 16), np.float16)
    for l in range(32):
        if gfx12_shape:
            fa[:, l, :8] = A[:, l % 16, 8 * (l // 16):8 * (l // 16) + 8]
            fb[:, l, :8] = Bm[:, 8 * (l // 16):8 * (l // 16) + 8, l % 16]
        else:
            fa[:, l] = A[:, l % 16]
            fb[:, l] = Bm[:, :, l % 16]
    ca = np.zeros((n, 32, 8), np.float32)
    for l in range(32):
        for i in range(8):
            ca[:, l, i] = C[:, 2 * i + l // 16, l % 16]
    return fa.tobytes() + fb.tobytes() + ca.tobytes() + b"\x7f" * (n * 1024), [0, n * 1024, 2 * n * 1024, 3 * n * 1024, n]


def channel_logical(raw):
    """Channel kernels' output (N, 32, 8) -> logical D (N, 16, 16)."""
    D = np.zeros((raw.shape[0], 16, 16), raw.dtype)
    for l in range(32):
        for i in range(8):
            D[:, 2 * i + l // 16, l % 16] = raw[:, l, i]
    return D


def emu_arena(dev, kernel, arena, offs):
    """Run a probe kernel on the emulated device over one arena laid out at offsets offs (pa, pb, pc, po, n)."""
    dev.reset()
    base = dev.malloc(len(arena))
    dev.upload(base, arena)
    dev.launch(kernel, 1, 32, [ctypes.c_void_p(base + o) for o in offs[:4]] + [ctypes.c_uint(offs[4])])
    return np.frombuffer(dev.download(base, len(arena)), np.uint8)


def run(dev, kernel, A, Bm, C, out_f16=False):
    """A, Bm (N, 16, 16) f16 logical, C (N, 16, 16) f32 -> the raw output of a tile kernel (N cases)."""
    n = A.shape[0]
    arena, offs = arena_tiles(A, Bm, C)
    out = emu_arena(dev, kernel, arena, offs)[offs[3]:offs[3] + n * (512 if out_f16 else 1024)].tobytes()
    return np.frombuffer(out, np.float16 if out_f16 else np.float32).reshape(n, 32, 8)


def onehot_cases():
    """A = E(r, k) with B[k][c] = c + 1 (every row and K position of A), and B = E(k, c) with A[r][k] = r + 1 (every K
    position and column of B): 512 cases whose products are exact."""
    A, Bm = [], []
    for r in range(16):
        for k in range(16):
            a = np.zeros((16, 16)); a[r, k] = 1
            b = np.zeros((16, 16)); b[k, :] = np.arange(1, 17)
            A.append(a); Bm.append(b)
    for k in range(16):
        for c in range(16):
            a = np.zeros((16, 16)); a[:, k] = np.arange(1, 17)
            b = np.zeros((16, 16)); b[k, c] = 1
            A.append(a); Bm.append(b)
    return np.array(A, np.float16), np.array(Bm, np.float16)


class TestRowMapTable(unittest.TestCase):
    def test_row_map_gives_the_gfx12_rows(self):
        """wg_a_row composed with the measured instruction (output element i of half h is row 2 i + h, read from lane
        16 (row & 1) + row) puts gfx12 row 8 h + i in element i of half h."""
        a_row = lambda p: 8 * (p & 1) + (p >> 1)
        self.assertEqual(sorted(a_row(p) for p in range(16)), list(range(16)))
        for h in range(2):
            for i in range(8):
                self.assertEqual(a_row(2 * i + h), 8 * h + i)


class TestOneHotEmu(unittest.TestCase):
    """The operand forms on the emulated gfx11 instruction (exact sums), every row / K / column position."""

    @classmethod
    def setUpClass(cls):
        cls.A, cls.B = onehot_cases()
        cls.C0 = np.zeros((cls.A.shape[0], 16, 16), np.float32)
        cls.want = acc_gfx12(np.einsum("nik,nkj->nij", cls.A.astype(np.float64), cls.B.astype(np.float64)))

    def test_ldA16_ldB16(self):
        got = run(device(), "nat_ldA", self.A, self.B, self.C0)
        np.testing.assert_array_equal(got, self.want)

    def test_b16_from_registers(self):
        got = run(device(), "nat_b16", self.A, self.B, self.C0)
        np.testing.assert_array_equal(got, self.want)

    def test_accumulator_start_in_gfx12_layout(self):
        rng = np.random.default_rng(7)
        C = rng.integers(-8, 8, self.C0.shape).astype(np.float32)
        got = run(device(), "nat_ldA", self.A, self.B, C)
        np.testing.assert_array_equal(got, self.want + acc_gfx12(C))

    def test_vt_form_gives_the_gfx12_fragment(self):
        """v^T's form: A's rows in the instruction's own order, then wg_rows_to_gfx12: the FP16 result in the gfx12
        fragment order (slot 16 h + c: rows 8 h .. 8 h + 7 of column c)."""
        D = np.einsum("nik,nkj->nij", self.A.astype(np.float64), self.B.astype(np.float64))
        got = run(device(), "nat_vt", self.A, self.B, self.C0, out_f16=True)
        np.testing.assert_array_equal(got, tile_gfx12(np.swapaxes(D, 1, 2).astype(np.float16)))

    def test_negative_control_vt_without_the_fix(self):
        D = np.einsum("nik,nkj->nij", self.A.astype(np.float64), self.B.astype(np.float64))
        got = run(device(), "nat_vt_nofix", self.A, self.B, self.C0, out_f16=True)
        self.assertFalse(np.array_equal(got, tile_gfx12(np.swapaxes(D, 1, 2).astype(np.float16))))

    def test_negative_control_wrong_row_table(self):
        """WG_A_ROW_MUT (rows 2 and 3 of the table swapped): the probe must catch it, on exactly the rows it moves."""
        got = run(device({"WG_A_ROW_MUT": 1}), "nat_ldA", self.A, self.B, self.C0)
        bad = np.argwhere(np.any(got != self.want, axis=2))
        self.assertGreater(len(bad), 0)
        # gfx12 rows held by physical rows 2 and 3: 1 and 9 -> elements 1 of lanes 0..15 and 16..31
        self.assertEqual(sorted({8 * (l // 16) + 1 for _, l in bad}), [1, 9])

    def test_k16_channel_equals_the_chain_and_the_exchange(self):
        """On the emulated instruction (exact sums) one K = 16 call, two chained K = 8 calls and HIP's exchanging wrapper
        give the same accumulator; the GPU decides whether the bits agree (GPU-TESTS.md)."""
        rng = np.random.default_rng(3)
        n = 8
        A = rng.integers(-4, 5, (n, 16, 16)).astype(np.float16)
        Bm = rng.integers(-4, 5, (n, 16, 16)).astype(np.float16)
        C = rng.integers(-8, 8, (n, 16, 16)).astype(np.float32)
        dev = device()

        def native(kernel, gfx12_shape=False):
            arena, offs = arena_channel(A, Bm, C, gfx12_shape)
            out = emu_arena(dev, kernel, arena, offs)[offs[3]:offs[3] + n * 1024].tobytes()
            return np.frombuffer(out, np.float32).reshape(n, 32, 8)

        want = np.zeros((n, 32, 8), np.float32)
        D = np.einsum("nik,nkj->nij", A.astype(np.float64), Bm.astype(np.float64)) + C
        for l in range(32):
            for i in range(8):
                want[:, l, i] = D[:, 2 * i + l // 16, l % 16]
        np.testing.assert_array_equal(native("k16n"), want)
        np.testing.assert_array_equal(native("k16_chain"), want)
        np.testing.assert_array_equal(native("k16_xchg", gfx12_shape=True), want)


# ---- the card (GPU-TESTS.md of the native RDNA 3 work): the same probes through vkd3d-proton with the K = 16 patch -----
sys.path.insert(0, HERE)
import test_gfx11_helpers as GH  # noqa: E402  (the card harness of the gfx11 helpers: its env variables, module doc)
CARD = GH.HAVE_CARD


def card_arena(kernel, arena, offs):
    out, txt = GH.Card.run(kernel, arena, offs, tag=kernel, out_bytes=len(arena))
    if out is None:
        raise AssertionError(f"{kernel}: no result\n{txt[-400:]}")
    return out


@unittest.skipUnless(CARD, "no card configured (test_gfx11_helpers.py module doc; needs the K = 16 patched DLLs)")
class TestNativeOnCard(unittest.TestCase):
    """GPU: the K = 16 opcode and the operand forms on the RX 7900 XT. WG_GFX11_VKD3D must hold the K = 16 patched
    vkd3d-proton (a stock build refuses the opcode)."""

    @classmethod
    def setUpClass(cls):
        GH.Card.setup()
        for e in NAMES:
            GH.Card.dxil[e] = GH.compile_kernel(e, src=SRC)
        cls.rng = np.random.default_rng(20261116)

    def channel(self, kernel, A, Bm, C, gfx12_shape=False):
        n = A.shape[0]
        arena, offs = arena_channel(A, Bm, C, gfx12_shape)
        raw = card_arena(kernel, arena, offs)[offs[3]:offs[3] + n * 1024].tobytes()
        return channel_logical(np.frombuffer(raw, np.float32).reshape(n, 32, 8))

    def data(self, n, specials=False):
        A = self.rng.normal(0, 1, (n, 16, 16))
        Bm = self.rng.normal(0, 1, (n, 16, 16))
        A[: n // 4] *= 2.0 ** self.rng.integers(-12, 12, (n // 4, 16, 1))      # cancellation across K
        C = self.rng.normal(0, 4, (n, 16, 16)).astype(np.float32)
        if specials:   # infinities and NaNs in both operand roles, signed zeros
            for M in (A, Bm):
                pick = self.rng.random(M.shape) < 0.01
                M[pick] = self.rng.choice([np.inf, -np.inf, np.nan, 0.0, -0.0, 65504.0], int(pick.sum()))
        return A.astype(np.float16), Bm.astype(np.float16), C

    def test_k16_opcode_equals_the_chain_bit_for_bit(self):
        for specials in (False, True):
            A, Bm, C = self.data(256, specials)
            one = self.channel("k16n", A, Bm, C)
            chain = self.channel("k16_chain", A, Bm, C)
            same = (one.view(np.uint32) == chain.view(np.uint32)) | (np.isnan(one) & np.isnan(chain))
            self.assertEqual(float(np.mean(same)), 1.0, f"specials={specials}")

    def test_k16_opcode_matches_hip(self):
        if not GH.HIP_RUN:
            self.skipTest("WG_GFX11_HIP_RUN not set")
        A, Bm, C = self.data(64)
        self.assertEqual(GH.bits_equal(self.channel("k16n", A, Bm, C), GH.hip_ref(A, Bm, C)), 1.0)

    def test_k16_opcode_close_to_float64(self):
        A, Bm, C = self.data(64)
        D = self.channel("k16n", A, Bm, C)
        ref = np.einsum("nik,nkj->nij", A.astype(np.float64), Bm.astype(np.float64)) + C
        self.assertLess(float(np.max(np.abs(D - ref) / (np.abs(ref) + 1.0))), 1e-3)

    def test_operand_forms_one_hot(self):
        A, Bm = onehot_cases()
        C0 = np.zeros((A.shape[0], 16, 16), np.float32)
        want = acc_gfx12(np.einsum("nik,nkj->nij", A.astype(np.float64), Bm.astype(np.float64)))
        n = A.shape[0]
        for kernel in ("nat_ldA", "nat_b16"):
            arena, offs = arena_tiles(A, Bm, C0)
            got = np.frombuffer(card_arena(kernel, arena, offs)[offs[3]:offs[3] + n * 1024].tobytes(),
                                np.float32).reshape(n, 32, 8)
            np.testing.assert_array_equal(got, want, kernel)
        arena, offs = arena_tiles(A, Bm, C0)
        got = np.frombuffer(card_arena("nat_vt", arena, offs)[offs[3]:offs[3] + n * 512].tobytes(),
                            np.float16).reshape(n, 32, 8)
        D = np.einsum("nik,nkj->nij", A.astype(np.float64), Bm.astype(np.float64))
        np.testing.assert_array_equal(got, tile_gfx12(np.swapaxes(D, 1, 2).astype(np.float16)))

    def test_operand_forms_equal_the_chain_on_random_data(self):
        """nat_ldA (row-mapped A, B column) gives the chain's bits on the same logical data: the same K order."""
        A, Bm, C = self.data(128)
        n = A.shape[0]
        arena, offs = arena_tiles(A, Bm, C)
        got = np.frombuffer(card_arena("nat_ldA", arena, offs)[offs[3]:offs[3] + n * 1024].tobytes(),
                            np.float32).reshape(n, 32, 8)
        chain = acc_gfx12(self.channel("k16_chain", A, Bm, C))
        self.assertEqual(GH.bits_equal(got, chain), 1.0)


# ---- S2: the translator (CPU: DXC + the dxil-spirv command line tool) --------------------------------------------------
DXIL_SPIRV = os.environ.get("WG_GFX11_DXIL_SPIRV")
SPIRV_DIS = os.environ.get("WG_GFX11_SPIRV_DIS") or shutil.which("spirv-dis")


def _have_translator():
    asm = os.environ.get("WG_GFX11_ASM")
    return bool(DXIL_SPIRV and os.path.exists(DXIL_SPIRV) and SPIRV_DIS and asm and os.path.exists(asm) and
                os.path.exists(os.path.join(B.DXC_DIR, "dxc.exe")))


def spirv_ops(entry, dxil_dir, tool=None, config="wmma_rdna3_raw_fp8"):
    """Compile `entry` of the probe for gfx11, translate it (the K = 16 patched tool, or `tool`) with
    DXIL_SPIRV_CONFIG=`config` (the shortcut's, "" for none), return the SPIR-V instruction names, or None when the
    translation is refused."""
    dxil = os.path.join(dxil_dir, f"{entry}.dxil")
    if not os.path.exists(dxil):
        decl = {d["name"]: d for d in B.kernel_decls(SRC)}[entry]
        B.compile_hlsl(SRC, decl["entry"], dxil, defines={"WG_BLOCK": 32, "WG_WAVE": 32, "WG_GFX": 11},
                       includes=[HERE], asm_tool=os.environ.get("WG_GFX11_ASM"))
    return translate_ops(dxil, tool or DXIL_SPIRV, config)


def translate_ops(dxil, tool, config="wmma_rdna3_raw_fp8"):
    """SPIR-V instruction names of one DXIL file (vkd3d-proton's raw buffers: SSBOs), None when refused."""
    spv = dxil + f".{os.path.basename(os.path.dirname(os.path.dirname(tool)))}.{config or 'noconf'}.spv"
    env = dict(os.environ)
    env.pop("DXIL_SPIRV_CONFIG", None)
    if config:
        env["DXIL_SPIRV_CONFIG"] = config
    if os.path.exists(spv):
        os.remove(spv)
    r = subprocess.run([tool, dxil, "--output", spv, "--ssbo-uav", "--ssbo-srv"], capture_output=True, text=True,
                       env=env)
    if r.returncode != 0 or not os.path.exists(spv):
        return None
    dis = subprocess.run([SPIRV_DIS, "--raw-id", spv], capture_output=True, text=True, check=True).stdout
    return re.findall(r"\bOp[A-Za-z0-9]+", dis)


@unittest.skipUnless(_have_translator(), "no patched dxil-spirv / spirv-dis / DXC configured (module doc)")
class TestTranslatorK16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = work_dir()

    def count(self, entry, op, config="wmma_rdna3_raw_fp8"):
        ops = spirv_ops(entry, self.dir, config=config)
        self.assertIsNotNone(ops, f"{entry}: the patched translator refused it")
        return sum(1 for o in ops if o == op)

    def test_one_multiply_per_k16_step_and_no_exchange(self):
        for config in ("wmma_rdna3_raw_fp8", ""):   # with and without the shortcut's configuration
            self.assertEqual(self.count("k16n", "OpCooperativeMatrixMulAddKHR", config), 1)
            self.assertEqual(self.count("k64n", "OpCooperativeMatrixMulAddKHR", config), 4)
            for e in ("k16n", "k64n", "nat_ldA"):
                ops = spirv_ops(e, self.dir, config=config)
                self.assertFalse([o for o in ops if o.startswith("OpGroupNonUniformShuffle")], (e, config))

    def test_stock_translator_refuses_the_opcode(self):
        """WG_GFX11_DXIL_SPIRV_STOCK (the translator without the K = 16 patch): the opcode is unknown, so it must refuse
        the native entries (a PSO fails to build instead of running something else) and still take the chain."""
        stock = os.environ.get("WG_GFX11_DXIL_SPIRV_STOCK")
        if not stock:
            self.skipTest("WG_GFX11_DXIL_SPIRV_STOCK not set")
        self.assertIsNone(spirv_ops("k16n", self.dir, tool=stock))
        self.assertIsNotNone(spirv_ops("k16_chain", self.dir, tool=stock))

    def test_controls_have_their_exchanges(self):
        """Negative controls of the count: the chained form has two multiplies, the exchanging wrapper its shuffles,
        and wg_b16 its one exchange per dword."""
        self.assertEqual(self.count("k16_chain", "OpCooperativeMatrixMulAddKHR"), 2)
        self.assertGreater(self.count("k16_xchg", "OpGroupNonUniformShuffleXor"), 0)
        self.assertEqual(self.count("nat_b16", "OpGroupNonUniformShuffleXor"), 4)


if __name__ == "__main__":
    unittest.main()
