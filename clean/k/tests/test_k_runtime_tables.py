# SPDX-License-Identifier: MIT
"""The runtime's C++ model K pieces against the Python reference:

- the filter-shape table and level (clean/k/runtime/k_filter_table.cpp, spec 43 draft 2 section 7) against
  clean/k/reference/outstage.py filter_table / filter_level and fp.py trunc16, bit for bit, all 16 sets;
- the model.wgcm loader (clean/k/runtime/model_k_load.cpp, spec 48 section 7) on a converted synthetic preset-K DLL
  (clean/converter/tests/synth.py k_shipped_like_case): every tensor equals the converter's --raw output, and damaged
  files and a wrong map are refused.

Both go through clean/k/tests/k_runtime_tables_dump.cpp, compiled here with the first available compiler: $CXX,
clang++ / g++ on PATH, ROCm's clang++, then MSVC cl (through vcvars64.bat; WG_K_RT_CXX=msvc picks it first).
Skipped when there is none.
Build outputs go to $WG_K_RT_BUILD if set, else to a temporary folder. The converter is $WG_CONVERTER_EXE if set, else
built from clean/converter/src with the same compiler.

Run: python -m unittest clean.k.tests.test_k_runtime_tables
"""
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clean.k.reference import fp as FP
from clean.k.reference import outstage as OS

HERE = os.path.dirname(os.path.abspath(__file__))
K_DIR = os.path.dirname(HERE)
CLEAN = os.path.dirname(K_DIR)
REPO = os.path.dirname(CLEAN)
RUNTIME_K = os.path.join(K_DIR, "runtime")
RUNTIME = os.path.join(CLEAN, "runtime")
CONV = os.path.join(CLEAN, "converter")
KMAP = os.path.join(REPO, "data", "preset-K.v1.map.json")

DUMP_SRCS = [os.path.join(HERE, "k_runtime_tables_dump.cpp"),
             os.path.join(RUNTIME_K, "k_filter_table.cpp"),
             os.path.join(RUNTIME_K, "model_k_load.cpp"),
             os.path.join(RUNTIME_K, "model_k.cpp"),
             os.path.join(RUNTIME_K, "k_pack.cpp"),
             os.path.join(CONV, "src", "sha256.cpp"),
             os.path.join(CONV, "src", "json.cpp")]
CONV_SRCS = [os.path.join(CONV, "src", n + ".cpp") for n in
             ("sha256", "json", "numeric", "map", "locate", "lz4block", "convert", "convert5", "ops5", "output", "pe_version",
              "api", "main")]   # the WGC_LIB_SOURCES of clean/converter/CMakeLists.txt + main
ROCM_ENV_DROP = ("LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIPFIRE_ROCM_PATH")
EXE = ".exe" if os.name == "nt" else ""


# ---- compiler -------------------------------------------------------------------------------------------------------

def _gcc_style(cxx):
    def build(srcs, out, includes):
        cmd = [cxx, "-std=c++17", "-O2", "-ffp-contract=off", "-D_CRT_SECURE_NO_WARNINGS"]
        cmd += ["-I" + i for i in includes] + srcs + ["-o", out]
        env = {k: v for k, v in os.environ.items() if k not in ROCM_ENV_DROP}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError("compile failed:\n" + r.stdout + r.stderr)
    return build


def _msvc(vcvars):
    def build(srcs, out, includes):
        objdir = os.path.join(os.path.dirname(out), "obj_" + os.path.splitext(os.path.basename(out))[0])
        os.makedirs(objdir, exist_ok=True)
        bat = os.path.join(os.path.dirname(out), "build_" + os.path.splitext(os.path.basename(out))[0] + ".bat")
        incs = " ".join('/I"%s"' % i for i in includes)
        files = " ".join('"%s"' % s for s in srcs)
        with open(bat, "w") as f:
            f.write('@echo off\r\nset "PATH=C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer;%PATH%"\r\n')
            f.write('call "%s" >nul || exit /b 1\r\n' % vcvars)
            f.write('cl /nologo /std:c++17 /O2 /EHsc /fp:precise /utf-8 /D_CRT_SECURE_NO_WARNINGS %s %s /Fo"%s\\\\" '
                    '/Fe"%s" || exit /b 1\r\n' % (incs, files, objdir, out))
        r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("compile failed:\n" + r.stdout + r.stderr)
    return build


def find_compiler():
    """(description, build function) of the first available C++17 compiler, or None. WG_K_RT_CXX=msvc forces MSVC."""
    force_msvc = os.environ.get("WG_K_RT_CXX", "").lower() == "msvc"
    cxx = None if force_msvc else os.environ.get("CXX")
    if cxx:
        return cxx, _gcc_style(cxx)
    for name in () if force_msvc else ("clang++", "g++"):
        p = shutil.which(name)
        if p:
            return p, _gcc_style(p)
    if os.name == "nt":
        for p in [] if force_msvc else sorted(glob.glob(r"C:\Program Files\AMD\ROCm\*\bin\clang++.exe"), reverse=True):
            return p, _gcc_style(p)
        vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        if os.path.exists(vswhere):
            r = subprocess.run([vswhere, "-latest", "-property", "installationPath"], capture_output=True, text=True)
            vcvars = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
            if r.returncode == 0 and r.stdout.strip() and os.path.exists(vcvars):
                return "MSVC cl (" + vcvars + ")", _msvc(vcvars)
    return None


_STATE = {}


def build_dir():
    if "dir" not in _STATE:
        d = os.environ.get("WG_K_RT_BUILD")
        if d:
            os.makedirs(d, exist_ok=True)
        else:
            d = tempfile.mkdtemp(prefix="wg_k_rt_")
        _STATE["dir"] = d
    return _STATE["dir"]


def compiler():
    if "cc" not in _STATE:
        _STATE["cc"] = find_compiler()
    if _STATE["cc"] is None:
        raise unittest.SkipTest("no C++ compiler (clang++, g++, ROCm clang++ or MSVC)")
    return _STATE["cc"]


def dump_exe():
    if "dump" not in _STATE:
        exe = os.path.join(build_dir(), "k_runtime_tables_dump" + EXE)
        compiler()[1](DUMP_SRCS, exe, [RUNTIME_K, RUNTIME])
        _STATE["dump"] = exe
    return _STATE["dump"]


def converter_exe():
    if "conv" not in _STATE:
        exe = os.environ.get("WG_CONVERTER_EXE")
        if not exe:
            exe = os.path.join(build_dir(), "wronggpu-convert" + EXE)
            compiler()[1](CONV_SRCS, exe, [os.path.join(CONV, "src")])
        _STATE["conv"] = exe
    return _STATE["conv"]


def run_dump(*args):
    return subprocess.run([dump_exe()] + [str(a) for a in args], capture_output=True, text=True)


# ---- filter-shape table -----------------------------------------------------------------------------------------------

class TestFilterTable(unittest.TestCase):
    def test_all_sets_bit_exact(self):
        """All 16 sets (levels 0..7, theta 1 and 0) equal outstage.filter_table bit for bit."""
        out = os.path.join(build_dir(), "filter_table.bin")
        r = run_dump("table", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = np.fromfile(out, np.uint16).reshape(16, 2048, 8)
        for s in range(16):
            level, theta = s // 2, 1 - s % 2
            ref = OS.filter_table(level, theta=theta).view(np.uint16)
            bad = np.argwhere(got[s] != ref)
            self.assertEqual(len(bad), 0, "set %d (level %d, theta %d): %d values differ, first %s: %04x vs %04x" % (
                s, level, theta, len(bad), bad[:1].tolist(), *(
                    (int(got[s][tuple(bad[0])]), int(ref[tuple(bad[0])])) if len(bad) else (0, 0))))

    def test_level(self):
        rng = np.random.default_rng(43)
        outs = [(1920, 1080), (2560, 1440), (3840, 2160), (1280, 720), (1000, 777), (7680, 4320), (640, 360),
                (3440, 1440), (1366, 768), (17, 9)]
        cases = []
        for W, H in outs:
            for s in np.linspace(0.5, 4.0, 141):  # render scale factors, incl. above the output size
                cases.append((max(1, int(W / s)), max(1, int(H / s)), W, H))
            for _ in range(200):
                cases.append((int(rng.integers(1, 2 * W)), int(rng.integers(1, 2 * H)), W, H))
        for rw in range(1, 200):  # small sizes: the clamp at 7 and the rounding boundaries
            cases.append((rw, rw, 256, 256))
            cases.append((rw * 3, rw * 2, 600, 400))
        src = os.path.join(build_dir(), "levels_in.txt")
        dst = os.path.join(build_dir(), "levels_out.txt")
        with open(src, "w") as f:
            f.write("".join("%d %d %d %d\n" % c for c in cases))
        r = run_dump("level", src, dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(dst) as f:
            got = [int(x) for x in f.read().split()]
        ref = [OS.filter_level(*c) for c in cases]
        self.assertEqual(len(got), len(cases))
        bad = [(c, g, w) for c, g, w in zip(cases, got, ref) if g != w]
        self.assertEqual(bad, [], "%d of %d levels differ" % (len(bad), len(cases)))
        self.assertEqual(set(ref), set(range(8)), "the cases cover every level")

    def test_trunc16(self):
        """binary32 -> binary16 truncation (spec 43 section 7.8) on random and boundary bit patterns."""
        rng = np.random.default_rng(78)
        rand = rng.integers(0, 2 ** 32, 400_000, dtype=np.uint64).astype(np.uint32)
        # magnitudes around the FP16 range: exponents 2^-26 ... 2^17
        near = (rng.integers(0x32800000, 0x48000000, 400_000, dtype=np.uint64).astype(np.uint32) |
                (rng.integers(0, 2, 400_000, dtype=np.uint64).astype(np.uint32) << np.uint32(31)))
        edges = FP.f32_bits(np.array([0.0, -0.0, 65504.0, 65504.5, 65519.0, 65519.99, 65520.0, 1e9, np.inf,
                                      -np.inf, np.nan, 2.0 ** -14, 2.0 ** -14 * (1 - 2 ** -20), 2.0 ** -24,
                                      2.0 ** -24 * 0.999, 2.0 ** -25, 1e-45, 1.0, 1.0 + 2 ** -11, 1.0 - 2 ** -24,
                                      -1.0 - 2 ** -11, 3.0e-5, 226.0, -0.1], np.float32))
        bits = np.concatenate([edges, rand, near]).astype(np.uint32)
        src = os.path.join(build_dir(), "trunc_in.bin")
        dst = os.path.join(build_dir(), "trunc_out.bin")
        bits.tofile(src)
        r = run_dump("trunc", src, dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = np.fromfile(dst, np.uint16)
        with np.errstate(invalid="ignore"):  # NaN inputs
            ref = FP.trunc16(FP.from_f32_bits(bits)).view(np.uint16)
        bad = np.flatnonzero(got != ref)
        self.assertEqual(len(bad), 0, "%d differ, first %08x -> %04x vs %04x" % (
            len(bad), *((int(bits[bad[0]]), int(got[bad[0]]), int(ref[bad[0]])) if len(bad) else (0, 0, 0))))


# ---- model.wgcm loader ------------------------------------------------------------------------------------------------

def parse_wgcm(data):
    """Index of a model.wgcm with the byte positions of its fields (clean/converter/OUTPUT-FORMAT.md)."""
    (isize,) = struct.unpack_from("<Q", data, 80)
    pos, groups = 88, []
    (ngroups,) = struct.unpack_from("<I", data, 12)
    for _ in range(ngroups):
        (n,) = struct.unpack_from("<H", data, pos)
        name = data[pos + 2: pos + 2 + n].decode()
        pos += 2 + n
        dpos = pos
        pos += 32
        (nt,) = struct.unpack_from("<I", data, pos)
        pos += 4
        tensors = []
        for _ in range(nt):
            (n,) = struct.unpack_from("<H", data, pos)
            tname = data[pos + 2: pos + 2 + n].decode()
            pos += 2 + n
            dtype, rank = data[pos], data[pos + 1]
            shape_pos = pos + 2
            pos += 2 + 4 * rank
            off, size = struct.unpack_from("<QQ", data, pos)
            pos += 16
            tensors.append({"name": tname, "dtype": dtype, "shape_pos": shape_pos, "rank": rank, "off": off,
                            "size": size})
        groups.append({"name": name, "digest_pos": dpos, "tensors": tensors})
    assert pos == 88 + isize
    return groups


def synthetic_converted():
    """A synthetic preset-K DLL (clean/converter/tests/synth.py k_shipped_like_case: the shipped map's layout with our
    own random content) converted with --raw, once per process: {"case", "work", "map", "dll", "out", "wgcm"}. Also
    used by test_k_runtime_pack.py."""
    if "synth" not in _STATE:
        if not os.path.exists(KMAP):
            raise unittest.SkipTest("preset-K map not found")
        conv = converter_exe()
        sys.path.insert(0, os.path.join(CONV, "tests"))
        try:
            from synth import k_shipped_like_case
        finally:
            sys.path.pop(0)
        work = os.path.join(build_dir(), "loader")
        if os.path.exists(work):
            shutil.rmtree(work)
        os.makedirs(work)
        # small magnitudes (|x| < 2^-3) keep every op finite
        case = k_shipped_like_case(KMAP, seed=1, max_exp_field=0x0B)
        dll = os.path.join(work, "fake.dll")
        map_path = os.path.join(work, "map.json")
        with open(dll, "wb") as f:
            f.write(case.file)
        with open(map_path, "wb") as f:
            f.write(json.dumps(case.map, indent=1).encode())
        out = os.path.join(work, "out")
        r = subprocess.run([conv, "--map", map_path, "--dll", dll, "--out", out, "--raw"], capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise RuntimeError("converter failed (%d):\n%s%s" % (r.returncode, r.stdout, r.stderr))
        _STATE["synth"] = {"case": case, "work": work, "map": map_path, "dll": dll, "out": out,
                           "wgcm": os.path.join(out, "model.wgcm")}
    return _STATE["synth"]


class TestKLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(KMAP):
            raise unittest.SkipTest("preset-K map not found")
        dump_exe()
        syn = synthetic_converted()
        cls.case, cls.work, cls.map, cls.out, cls.wgcm = syn["case"], syn["work"], syn["map"], syn["out"], syn["wgcm"]
        with open(cls.wgcm, "rb") as f:
            cls.data = f.read()

    def load(self, wgcm, map_path=None, out_dir=None):
        args = ["load", wgcm, map_path or self.map] + ([out_dir] if out_dir else [])
        r = run_dump(*args)
        return r.returncode, (r.stdout + r.stderr).strip()

    def variant(self, name, data, map_obj=None):
        """Writes a damaged copy (and optionally its own map); returns (wgcm path, map path)."""
        d = os.path.join(self.work, name)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "model.wgcm")
        with open(p, "wb") as f:
            f.write(bytes(data))
        mp = self.map
        if map_obj is not None:
            mp = os.path.join(d, "map.json")
            with open(mp, "wb") as f:  # binary: the header holds the SHA-256 of these exact bytes
                f.write(json.dumps(map_obj, indent=1).encode())
        return p, mp

    def resign(self, data, map_obj):
        """Recomputes every group digest (stored and in map_obj), the map hash in the header and the trailer, as a
        converter would for this content. Returns the new map bytes."""
        key = bytes(data[16:48])
        groups = parse_wgcm(bytes(data))
        for g in groups:
            h = hashlib.sha256(key)
            for t in g["tensors"]:
                h.update(bytes(data[t["off"]: t["off"] + t["size"]]))
            data[g["digest_pos"]: g["digest_pos"] + 32] = h.digest()
            for st in map_obj["stages"]:
                if st["stage"] == g["name"]:
                    st["digest"] = h.hexdigest()
        mbytes = json.dumps(map_obj, indent=1).encode()
        data[48:80] = hashlib.sha256(mbytes).digest()
        data[-32:] = hashlib.sha256(bytes(data[:-32])).digest()

    def tensor(self, group, name):
        for g in parse_wgcm(self.data):
            if g["name"] == group:
                for t in g["tensors"]:
                    if t["name"] == name:
                        return t
        raise KeyError(group + "/" + name)

    # -- accepted file

    def test_specs_match_map(self):
        """The loader's tensor list (spec 48 section 7.2) is the map's, names, shapes and order."""
        r = run_dump("specs")
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout)
        with open(KMAP) as f:
            m = json.load(f)
        want = {st["stage"]: [[t["name"], t["shape"]] for t in st["tensors"]] for st in m["stages"]
                if st["stage"] != "net"}
        self.assertEqual(got, want)
        net = [st for st in m["stages"] if st["stage"] == "net"][0]["tensors"]
        self.assertEqual([(t["name"], t["shape"], t["dtype"]) for t in net], [("window_offsets", [8, 2], "int32")])

    def test_load_equals_raw(self):
        loaded = os.path.join(self.work, "loaded")
        if os.path.exists(loaded):
            shutil.rmtree(loaded)
        code, msg = self.load(self.wgcm, out_dir=loaded)
        self.assertEqual((code, msg), (0, "ok"))
        raw = os.path.join(self.out, "raw")
        files = sorted(os.path.relpath(p, raw) for p in glob.glob(os.path.join(raw, "*", "*.bin")))
        self.assertEqual(sorted(os.path.relpath(p, loaded) for p in glob.glob(os.path.join(loaded, "*", "*.bin"))),
                         files)
        self.assertEqual(len(files), sum(len(st["tensors"]) for st in self.case.map["stages"]))
        for rel in files:
            with open(os.path.join(raw, rel), "rb") as f:
                want = f.read()
            with open(os.path.join(loaded, rel), "rb") as f:
                got = f.read()
            self.assertEqual(got, want, rel)
        # and the synthetic case's own expected values
        for stage, tensors in self.case.expected.items():
            for name, data in tensors:
                with open(os.path.join(loaded, stage, name + ".bin"), "rb") as f:
                    self.assertEqual(f.read(), data, stage + "/" + name)
        wo = np.fromfile(os.path.join(loaded, "net", "window_offsets.bin"), np.int32)
        self.assertEqual(wo.shape, (16,))
        self.assertTrue(np.all((wo >= 0) & (wo <= 7)))

    # -- refused files

    def test_flipped_byte(self):
        t = self.tensor("d2", "W1")
        for label, pos in (("tensor data", t["off"] + 13), ("index", 88 + 3), ("key", 20), ("trailer", -1)):
            d = bytearray(self.data)
            d[pos] ^= 0x10
            p, _ = self.variant("flip_" + label.replace(" ", "_"), d)
            code, msg = self.load(p)
            self.assertEqual(code, 1, label)
            self.assertIn("fails its integrity check", msg, label)

    def test_flipped_byte_with_valid_trailer(self):
        """Tensor data changed and the trailer recomputed: the keyed stage digest catches it."""
        t = self.tensor("m", "Wk")
        d = bytearray(self.data)
        d[t["off"] + 101] ^= 0x01
        d[-32:] = hashlib.sha256(bytes(d[:-32])).digest()
        p, _ = self.variant("flip_resigned_trailer", d)
        code, msg = self.load(p)
        self.assertEqual(code, 1)
        self.assertIn("stage m does not match the map's digest", msg)

    def test_truncated(self):
        for n in (len(self.data) - 1, len(self.data) // 2, 100, 0):
            p, _ = self.variant("trunc_%d" % n, self.data[:n])
            code, msg = self.load(p)
            self.assertEqual(code, 1, n)
            self.assertIn("model data damaged", msg, n)

    def test_wrong_map(self):
        code, msg = self.load(self.wgcm, KMAP)  # the shipped map: other source hashes and digests
        self.assertEqual(code, 1)
        self.assertIn("converted with a different map", msg)
        m = json.loads(json.dumps(self.case.map))
        m["stages"][3]["digest"] = "00" * 32  # same layout, one digest changed: other map bytes
        p, mp = self.variant("other_map", self.data, m)
        code, msg = self.load(p, mp)
        self.assertEqual(code, 1)
        self.assertIn("converted with a different map", msg)
        code, msg = self.load(self.wgcm, os.path.join(self.work, "no-such-map.json"))
        self.assertEqual(code, 1)
        self.assertIn("converter map missing", msg)

    def test_map_digest_differs(self):
        """A map whose digest for one stage differs, with the file's header and trailer made to match it."""
        m = json.loads(json.dumps(self.case.map))
        d = bytearray(self.data)
        self.resign(d, m)
        m["stages"][5]["digest"] = "ab" * 32
        mbytes = json.dumps(m, indent=1).encode()
        d[48:80] = hashlib.sha256(mbytes).digest()
        d[-32:] = hashlib.sha256(bytes(d[:-32])).digest()
        p, mp = self.variant("map_digest", d, m)
        code, msg = self.load(p, mp)
        self.assertEqual(code, 1)
        self.assertIn("stage m does not match the map's digest", msg)

    def test_window_offset_out_of_range(self):
        t = self.tensor("net", "window_offsets")
        for v in (8, -1):
            d = bytearray(self.data)
            struct.pack_into("<i", d, t["off"] + 4 * 5, v)
            m = json.loads(json.dumps(self.case.map))
            self.resign(d, m)
            p, mp = self.variant("offset_%d" % v, d, m)
            code, msg = self.load(p, mp)
            self.assertEqual(code, 1, v)
            self.assertIn("window_offsets has a value outside 0..7", msg)

    def test_non_finite(self):
        t = self.tensor("e3", "bo")
        d = bytearray(self.data)
        struct.pack_into("<f", d, t["off"] + 8, float("inf"))
        m = json.loads(json.dumps(self.case.map))
        self.resign(d, m)
        p, mp = self.variant("non_finite", d, m)
        code, msg = self.load(p, mp)
        self.assertEqual(code, 1)
        self.assertIn("e3/bo has a non-finite value", msg)

    def test_wrong_shape(self):
        """e0/We stored as [16][32] instead of [32][16] (same size, valid digests)."""
        t = self.tensor("e0", "We")
        d = bytearray(self.data)
        struct.pack_into("<II", d, t["shape_pos"], 16, 32)
        m = json.loads(json.dumps(self.case.map))
        self.resign(d, m)
        p, mp = self.variant("wrong_shape", d, m)
        code, msg = self.load(p, mp)
        self.assertEqual(code, 1)
        self.assertIn("e0/We has the wrong shape or type", msg)

    def test_wrong_dtype(self):
        """net/window_offsets marked float32 (valid digests): refused."""
        t = self.tensor("net", "window_offsets")
        d = bytearray(self.data)
        d[t["shape_pos"] - 2] = 0
        m = json.loads(json.dumps(self.case.map))
        self.resign(d, m)
        p, mp = self.variant("wrong_dtype", d, m)
        code, msg = self.load(p, mp)
        self.assertEqual(code, 1)
        self.assertIn("window_offsets, int32 [8][2]", msg)


if __name__ == "__main__":
    unittest.main()
