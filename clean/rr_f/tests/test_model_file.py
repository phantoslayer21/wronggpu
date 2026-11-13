# SPDX-License-Identifier: MIT
"""Loading converted preset-F model files (pages 56.4, 57.7; 57-T6 runtime round trip on synthetic data).

Standard library only (the NumPy stage step runs only where NumPy is installed). The round-trip tests convert the
spec team's synthetic 57-T8 files with the C++ converter; they are skipped when the converter or the synthetic
library stand-ins are not found. Paths: WGC_CONVERT_EXE (default clean/converter/build-plain/wronggpu-convert[.exe]),
WGC_SPEC_DATA (default data/), WGC_SYNTH_BINS (no default: the synthetic library stand-ins are not in the
repository; without them those tests are skipped).
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

from clean.rr_f.model import model_file as mf

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _spec_data():
    env = os.environ.get("WGC_SPEC_DATA")
    if env:
        return env
    return os.path.join(ROOT, "data")


def _exe():
    env = os.environ.get("WGC_CONVERT_EXE")
    if env:
        return env
    base = os.path.join(ROOT, "clean", "converter", "build-plain", "wronggpu-convert")
    return base + ".exe" if os.path.exists(base + ".exe") else base


SYNTH_BINS = os.environ.get("WGC_SYNTH_BINS", "")
CODES = {v: k for k, v in mf.DTYPES.items()}


def write_container(groups, key=b"K" * 32, map_sha=b"M" * 32, version=2, digests=None):
    """An independent writer of the container of OUTPUT-FORMAT.md: groups = [(name, [(tname, dtype, shape, data)])]."""
    def name(s):
        b = s.encode()
        return struct.pack("<H", len(b)) + b

    def index(offsets):
        ix = b""
        k = 0
        for gi, (g, tensors) in enumerate(groups):
            dg = digests[gi] if digests else hashlib.sha256(key + b"".join(t[3] for t in tensors)).digest()
            ix += name(g) + dg + struct.pack("<I", len(tensors))
            for (t, dt, shape, data) in tensors:
                ix += name(t) + bytes([CODES[dt], len(shape)]) + struct.pack("<%dI" % len(shape), *shape)
                ix += struct.pack("<QQ", offsets[k] if offsets else 0, len(data))
                k += 1
        return ix

    align = lambda v: (v + 63) // 64 * 64
    size0 = len(index(None))
    pos = align(88 + size0)
    offsets = []
    for _, tensors in groups:
        for t in tensors:
            offsets.append(pos)
            pos = align(pos + len(t[3]))
    ix = index(offsets)
    out = bytearray(b"WGCMODEL" + struct.pack("<II", version, len(groups)) + key + map_sha + struct.pack("<Q", len(ix)))
    out += ix
    k = 0
    for _, tensors in groups:
        for t in tensors:
            out += b"\0" * (offsets[k] - len(out))
            out += t[3]
            k += 1
    out += b"\0" * (pos - len(out))
    return bytes(out) + hashlib.sha256(bytes(out)).digest()


def groups_of(model):
    return [(g.name, [(t.name, t.dtype, t.shape, t.data) for t in g.tensors.values()]) for g in model.groups.values()]


class Inventory(unittest.TestCase):
    """57 section 11 item 1: the page-56.4 / 57.7 inventory (written from the page text) equals the shipped map."""

    def test_inventory_equals_shipped_map(self):
        path = os.path.join(_spec_data(), "preset-F.v1.map.json")
        if not os.path.exists(path):
            self.skipTest("shipped map not found")
        with open(path) as f:
            m = json.load(f)
        inv = mf.expected_inventory()
        self.assertEqual([g["group"] for g in m["groups"]], list(inv))
        for g in m["groups"]:
            got = {t["name"]: (tuple(t["shape"]), t["dtype"]) for t in g["tensors"]}
            self.assertEqual(got, inv[g["group"]], g["group"])

    def test_element_counts_of_56_4_4(self):
        inv = mf.expected_inventory()
        want = {"IN0": (20576, 6435), "E1": (78016, 6691), "E2": (160032, 12995), "E3": (262528, 13219),
                "E4": (430560, 25699), "E5": (328160, 25539), "D4-F": (430080, 26176), "D4-R": (328320, 25540),
                "D3-F": (262144, 13568), "D3-R": (180736, 13060), "D2-F": (159744, 13248), "D2-R": (110976, 12868),
                "D1-F": (69632, 6784), "D1-R": (45312, 6532), "D0-F": (22528, 6464), "D0-R": (14336, 6336)}
        tot = [0, 0]
        for part, (e4, f16) in want.items():
            n = {"e4m3": 0, "float16": 0}
            for name, (shape, dt) in inv[part].items():
                if dt in n and name != "g":  # 56.4.4 counts f16 without g
                    c = 1
                    for d in shape:
                        c *= d
                    n[dt] += c
            self.assertEqual((n["e4m3"], n["float16"]), (e4, f16), part)
            tot[0] += e4
            tot[1] += f16
        self.assertEqual(tot, [2903680, 221154])


class Container(unittest.TestCase):
    def tiny(self):
        return [("constants", [("X", "float32", (1,), struct.pack("<f", 1.5)),
                               ("SH", "uint32", (2,), struct.pack("<II", 2, 4))]),
                ("IN0", [("Wq", "e4m3", (2, 3), bytes(range(6))), ("g", "float16", (1,), struct.pack("<e", 0.5))])]

    def test_parse_and_digest(self):
        data = write_container(self.tiny())
        m = mf.read_model_file(data)
        self.assertEqual(list(m.groups), ["constants", "IN0"])
        self.assertEqual(m.groups["IN0"].tensors["Wq"].shape, (2, 3))
        self.assertEqual(m.groups["constants"].tensors["X"].values(), [1.5])

    def test_refusals(self):
        good = write_container(self.tiny())
        cases = {
            "trailer": good[:-1] + bytes([good[-1] ^ 1]),
            "magic": b"X" + good[1:],
            "short": good[:50],
        }
        for label, data in cases.items():
            with self.assertRaises(mf.ModelFileError, msg=label):
                mf.read_model_file(data)
        with self.assertRaises(mf.ModelFileError) as cm:
            mf.read_model_file(write_container(self.tiny(), version=1))
        self.assertEqual(cm.exception.code, "CONTAINER_VERSION")
        bad = write_container(self.tiny(), digests=[b"\0" * 32, b"\0" * 32])
        with self.assertRaises(mf.ModelFileError) as cm:
            mf.read_model_file(bad)
        self.assertEqual(cm.exception.code, "DIGEST_MISMATCH")
        r = mf.admit_model_file(good)
        self.assertEqual(r.decision, mf.REJECTED)  # not the preset-F inventory
        self.assertEqual(r.issues[0][:2], ("TENSOR_SET", "IN0"))


def _have_round_trip():
    return (os.path.exists(_exe()) and
            all(os.path.exists(os.path.join(SYNTH_BINS, "synth-common-v%d.bin" % v)) for v in (0, 1)) and
            os.path.exists(os.path.join(_spec_data(), "rr-f-synth", "synth-r0.map.json")))


@unittest.skipUnless(_have_round_trip(), "converter or synthetic library stand-ins not found")
class RoundTrip(unittest.TestCase):
    """57-T6 on synthetic data: the runtime loads what the converter wrote (both build variants)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rrf-model-file-")
        cls.map_path = os.path.join(_spec_data(), "rr-f-synth", "synth-r0.map.json")
        tags = [struct.pack("<e", float(v)) for v in range(16)]
        w = b"".join(tags[e & 15] for e in range(6386372 // 2))
        cls.files = {}
        for v in (0, 1):
            with open(os.path.join(SYNTH_BINS, "synth-common-v%d.bin" % v), "rb") as f:
                dll = os.path.join(cls.tmp, "file-v%d.bin" % v)
                with open(dll, "wb") as o:
                    o.write(w + f.read())
            out = os.path.join(cls.tmp, "out-v%d" % v)
            p = subprocess.run([_exe(), "--map", cls.map_path, "--dll", dll, "--out", out], capture_output=True)
            if p.returncode != 0:
                raise RuntimeError("conversion failed: %s" % p.stdout.decode())
            cls.files[v] = os.path.join(out, "model.wgcm")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def admit(self, v, **kw):
        return mf.admit_model_file(self.files[v], map_path=self.map_path, **kw)

    def test_both_variants_admitted(self):
        for v, variant in ((0, "B1"), (1, "B2")):
            r = self.admit(v)
            self.assertEqual(r.decision, mf.ADMITTED, r.issues)
            self.assertTrue(r.player_ready)
            self.assertEqual(r.model.variant, variant)
            self.assertFalse(r.model.attention_sum_may_overflow)

    def test_also_without_the_map(self):
        r = mf.admit_model_file(self.files[1])
        self.assertEqual(r.decision, mf.ADMITTED, r.issues)

    def test_accessors(self):
        m = self.admit(1).model
        for parity in (0, 1):
            sh = m.shifts(parity)
            self.assertEqual(len(sh), 11)
            for sx, sy in sh.values():  # SH carries `require: even, below_8`
                self.assertTrue(sx in (0, 2, 4, 6) and sy in (0, 2, 4, 6))
        for part in mf.PARTS:
            c = m.stage_constants(part)
            self.assertEqual(set(c) - {"f", "g"}, {"ra", "m"})
            self.assertEqual("f" in c, part.endswith("-F"))
            self.assertEqual("g" in c, part in mf.SINGLES or part.endswith("-R"))
        self.assertEqual(set(m.window_constants()), set(mf.WINDOW_SYMBOLS))
        self.assertEqual(len(m.tensor("D0-R", "K1").values()), 1)
        self.assertIsInstance(m.symbol("C_SV4"), float)
        with self.assertRaises(KeyError):
            self.admit(0).model.symbol("C_SV4")

    def test_tampered_file_refused(self):
        model = mf.read_model_file(self.files[0])
        groups = groups_of(model)
        name, tensors = groups[3]
        t = tensors[0]
        tensors[0] = (t[0], t[1], t[2], bytes([t[3][0] ^ 1]) + t[3][1:])
        # Keep the stored digests: the data no longer gives them (57.7.3 read-back).
        stored = [g.digest for g in model.groups.values()]
        data = write_container(groups, key=model.key, map_sha=model.map_sha256, digests=stored)
        r = mf.admit_model_file(data, map_path=self.map_path)
        self.assertEqual(r.decision, mf.REJECTED)
        self.assertEqual(r.issues[0][:2], ("DIGEST_MISMATCH", name))
        # Recomputed digests: the file is consistent, but the map's digest no longer matches.
        data = write_container(groups, key=model.key, map_sha=model.map_sha256)
        r = mf.admit_model_file(data, map_path=self.map_path)
        self.assertEqual(r.issues[0][:2], ("DIGEST_MISMATCH", name))

    def test_missing_and_extra_groups(self):
        model = mf.read_model_file(self.files[1])
        groups = groups_of(model)
        # An earlier file without constants_b2 and without any record of it: accepted as B1 (57.7.3).
        no_b2 = [g for g in groups if g[0] != "constants_b2"]
        r = mf.admit_model_file(write_container(no_b2, key=model.key), map_path=self.map_path)
        self.assertEqual((r.decision, r.model.variant), (mf.ADMITTED, "B1"))
        no_e3 = [g for g in groups if g[0] != "E3"]
        r = mf.admit_model_file(write_container(no_e3, key=model.key), map_path=self.map_path)
        self.assertEqual(r.issues[0][:2], ("MISSING_GROUP", "E3"))
        extra = groups + [("filter", [("output_taps", "float32", (5,), b"\0" * 20)])]
        r = mf.admit_model_file(write_container(extra, key=model.key), map_path=self.map_path)
        self.assertEqual(r.issues[0][0], "UNKNOWN_GROUP")
        reshaped = [(g, [(t, dt, (shape[0] * shape[1],) + tuple(shape[2:]) if t == "Wm" else shape, d)
                         for (t, dt, shape, d) in ts]) for g, ts in groups]
        r = mf.admit_model_file(write_container(reshaped, key=model.key))
        self.assertEqual(r.issues[0][0], "TENSOR_SET")

    def test_reference_stage_runs_with_loaded_weights(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy not installed: the reference stage step of 57-T6 is not run here")
        from clean.rr_f.reference.stages import weight_shapes
        m = self.admit(1).model
        for part in mf.PARTS:
            if part == "D0-R":
                continue
            w = m.stage_weights_numpy(part)
            for name, (shape, _t) in weight_shapes(part).items():
                self.assertEqual(w[name].shape, shape, part + "." + name)


if __name__ == "__main__":
    unittest.main()
