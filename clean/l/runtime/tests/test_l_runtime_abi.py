# SPDX-License-Identifier: MIT
"""Model L's code objects against what the runtime launches (#89), without running them: l_enc_values.co, l_ops.co,
l_swin.co and l_post.co are compiled (clean/l/runner/l_c32_check.py compile_kernels; HIP SDK clang, compile only),
and each kernel the runtime frame launches (clean/runtime/runtime.cpp, clean/l/FORMATS.md) must be exported with the
arguments the runtime passes: their number and sizes (a device address 8 bytes, an int 4, the encoder parameters
WgEncParams 116, the decoder parameters WgDecParams 56, the reconstruction parameters WgPostParams 64). The kernels'
own metadata (the AMDGPU metadata note of each code object, amdhsa.kernels) is read here; nothing is loaded or
executed. The argument kinds are the frame trace's table
(clean/tests/runtime_frame_trace.cpp), so the recorder that checks the runtime's arguments and the kernels agree.

Skipped without the HIP SDK (ROCM_PATH or C:\\Program Files\\AMD\\ROCm\\7.2).

Run (repository root):
    python -m unittest discover -s clean/l/runtime/tests -t clean/l/runtime/tests -p test_l_runtime_abi.py
"""
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
SIZES = {"p": 8, "i": 4, "E": 116, "D": 56, "P": 64}
# the kernels the Model L frame launches, by code object (the lev tag of l_enc_values.co is the runtime's)
LAUNCHED = {"l_enc_values.co": ["input_encoder"],
            "l_ops.co": ["l_linear_32_32_relu", "l_linear_32_32_f16", "l_project_gather", "l_dec_project", "l_coef"],
            "l_swin.co": ["l_swin_block_c32_enc", "l_swin_block_c32_enc_nc", "l_swin_block_c32_dec",
                          "l_swin_block_c32_dec_nc"],
            "l_post.co": ["l_post_recon"]}
_BUILD = None


def rocm():
    for d in (os.environ.get("ROCM_PATH"), r"C:\Program Files\AMD\ROCm\7.2"):
        if d and os.path.isdir(d):
            return d
    return None


def build():
    global _BUILD
    if _BUILD is None:
        r = rocm()
        if not r:
            raise unittest.SkipTest("no HIP SDK (ROCM_PATH)")
        sys.path.insert(0, ROOT)
        from clean.l.runner import l_c32_check as C32
        d = tempfile.mkdtemp(prefix="wg_labi_")
        C32.compile_kernels(d, r)
        _BUILD = d
    return _BUILD


def kinds_table():
    """{kernel name or prefix*: kinds} from the frame trace program's table."""
    with open(os.path.join(ROOT, "clean", "tests", "runtime_frame_trace.cpp"), encoding="utf-8") as f:
        text = f.read()
    return dict(re.findall(r'\{"([A-Za-z0-9_*]+)", "([piEDP]+)"\}', text))


def kinds_of(table, kernel):
    for name, kinds in table.items():
        if name.endswith("*") and kernel.startswith(name[:-1]) or name == kernel:
            return kinds
    return None


# ---- the AMDGPU metadata note: ELF64 notes, NT_AMDGPU_METADATA (32, owner "AMDGPU"), MessagePack ----
def msgpack(b, i=0):
    """(value, next index) of one MessagePack object (the types the metadata uses)."""
    t = b[i]
    if t <= 0x7F:
        return t, i + 1
    if 0x80 <= t <= 0x8F or t in (0xDE, 0xDF):
        n, i = (t & 0x0F, i + 1) if t <= 0x8F else (struct.unpack_from(">H" if t == 0xDE else ">I", b, i + 1)[0],
                                                     i + (3 if t == 0xDE else 5))
        out = {}
        for _ in range(n):
            k, i = msgpack(b, i)
            v, i = msgpack(b, i)
            out[k] = v
        return out, i
    if 0x90 <= t <= 0x9F or t in (0xDC, 0xDD):
        n, i = (t & 0x0F, i + 1) if t <= 0x9F else (struct.unpack_from(">H" if t == 0xDC else ">I", b, i + 1)[0],
                                                     i + (3 if t == 0xDC else 5))
        out = []
        for _ in range(n):
            v, i = msgpack(b, i)
            out.append(v)
        return out, i
    if 0xA0 <= t <= 0xBF or t in (0xD9, 0xDA, 0xDB):
        if t <= 0xBF:
            n, i = t & 0x1F, i + 1
        else:
            w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[t]
            n, i = int.from_bytes(b[i + 1:i + 1 + w], "big"), i + 1 + w
        return b[i:i + n].decode(), i + n
    fixed = {0xC0: (None, 1), 0xC2: (False, 1), 0xC3: (True, 1)}
    if t in fixed:
        return fixed[t][0], i + fixed[t][1]
    ints = {0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q", 0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"}
    if t in ints:
        return struct.unpack_from(ints[t], b, i + 1)[0], i + 1 + struct.calcsize(ints[t])
    if t >= 0xE0:
        return t - 0x100, i + 1
    raise ValueError("MessagePack type 0x%02x not handled" % t)


def kernels_of(path):
    """{kernel name: [argument sizes]} from a code object's AMDGPU metadata."""
    with open(path, "rb") as f:
        e = f.read()
    assert e[:4] == b"\x7fELF" and e[4] == 2, path
    shoff, shentsize, shnum = struct.unpack_from("<Q", e, 0x28)[0], *struct.unpack_from("<HH", e, 0x3A)
    for k in range(shnum):
        sh_type, = struct.unpack_from("<I", e, shoff + k * shentsize + 4)
        if sh_type != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", e, shoff + k * shentsize + 0x18)
        i = off
        while i < off + size:
            namesz, descsz, ntype = struct.unpack_from("<III", e, i)
            name = e[i + 12:i + 12 + namesz].rstrip(b"\0")
            desc = i + 12 + (namesz + 3) // 4 * 4
            if ntype == 32 and name == b"AMDGPU":
                meta, _ = msgpack(e, desc)
                return {kd[".name"]: [a[".size"] for a in kd.get(".args", []) if not a[".value_kind"].startswith(
                    "hidden_")] for kd in meta["amdhsa.kernels"]}
            i = desc + (descsz + 3) // 4 * 4
    raise AssertionError("no AMDGPU metadata note in " + path)


class Abi(unittest.TestCase):
    def test_launched_kernels(self):
        b = build()
        table = kinds_table()
        for co, names in LAUNCHED.items():
            exported = kernels_of(os.path.join(b, co))
            for k in names:
                self.assertIn(k, exported, "%s does not export %s" % (co, k))
                kinds = kinds_of(table, k)
                self.assertIsNotNone(kinds, k)
                self.assertEqual(exported[k], [SIZES[c] for c in kinds], "%s in %s" % (k, co))

    def test_reader(self):
        """The metadata reader on Model M's own encoder: input_encoder takes 12 addresses and WgEncParams."""
        b = build()
        self.assertEqual(kernels_of(os.path.join(b, "l_enc_values.co"))["input_encoder"], [8] * 12 + [116])


if __name__ == "__main__":
    unittest.main()
