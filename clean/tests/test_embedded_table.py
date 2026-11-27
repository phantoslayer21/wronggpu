# SPDX-License-Identifier: GPL-3.0-only
"""The runtime library's built-in file table (clean/runner/build.py write_embedded -> embedded_files.cpp, read by
clean/runtime/embedded.cpp): every file the runtime's own lookup (wg_read_file) returns has the SHA-256 of its source
file. A host harness (clean/tests/embed_dump.cpp) walks the compiled table and writes each file back out.

    python -m unittest clean.tests.test_embedded_table          # CPU: synthetic files (every byte value, empty file,
                                                                # quotes / backslashes / "??" in text) + a mutant
    python -m clean.tests.test_embedded_table --build <dir> [--model-l] [--no-model-k] [--cxx <clang++>]
                                                                # the table of a real build folder (count, bytes)"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from clean.runner import build  # noqa: E402


def source_of(out_dir, name):
    """The file a table entry was made from (write_embedded's naming)."""
    if name == "manifest.json":
        return os.path.join(out_dir, "dx12", "manifest.json")
    if name.endswith(".dxil"):
        return os.path.join(out_dir, "dx12", name)
    m = re.match(r"^([A-Za-z0-9]+)\.(.+\.co)$", name)
    if m and not os.path.isfile(os.path.join(out_dir, name)):
        p = os.path.join(out_dir, m.group(1), m.group(2))   # an extra architecture's code object
        if os.path.isfile(p):
            return p
    return os.path.join(out_dir, name)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dump(table_cpp, cxx=None, opt="-O1"):
    """Compiles embedded.cpp + table_cpp + the harness and returns {name: bytes} of the runtime's lookup."""
    tmp = tempfile.mkdtemp(prefix="wg_embed_")
    try:
        exe = os.path.join(tmp, "embed_dump" + (".exe" if os.name == "nt" else ""))
        cxx = cxx or os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("no clang++ / g++ (the table is built by clang++ or g++ only)")
        cmd = [cxx, "-std=c++17", opt, "-DWG_EMBEDDED_TABLE", "-I", os.path.join(ROOT, "clean", "runtime")]
        cmd += ["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else []
        cmd += [os.path.join(ROOT, "clean", "runtime", "embedded.cpp"), table_cpp,
                os.path.join(ROOT, "clean", "tests", "embed_dump.cpp"), "-o", exe]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("harness build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        empty, out = os.path.join(tmp, "empty"), os.path.join(tmp, "out")
        os.makedirs(empty)
        os.makedirs(out)
        r = subprocess.run([exe, empty, out], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("harness failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        count = int(r.stdout.splitlines()[0].split()[1])
        got = {}
        for line in r.stdout.splitlines()[1:]:
            name = line.rsplit(" ", 1)[0]
            with open(os.path.join(out, name), "rb") as f:
                got[name] = f.read()
        assert count == len(got), (count, len(got))
        return got
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_folder(out_dir, got):
    """{name: (ok, size)} against the source files; raises on a name the folder cannot account for."""
    res = {}
    for name, data in got.items():
        src = source_of(out_dir, name)
        res[name] = (hashlib.sha256(data).hexdigest() == sha(src), len(data))
    return res


def table_names(path):
    with open(path) as f:
        return re.findall(r'\{"([^"]+)", f\d+, \d+\}', f.read())


class EmbeddedTable(unittest.TestCase):
    def _folder(self):
        d = tempfile.mkdtemp(prefix="wg_embed_src_")
        self.addCleanup(shutil.rmtree, d, True)
        rng = random.Random(20261115)
        files = {
            "a.co": bytes(range(256)) * 3 + b"\x00\x7f\xff\x22\x5c",   # every byte value, size not a multiple of 8
            "b.co": b"",                                                 # empty
            "c.json": b'{"k": "a\\\\b \\"q\\" ??= ??/ ??\' \\x41B \\101 1"}\n',   # quotes, backslashes, trigraphs
            "d.co": bytes(rng.getrandbits(8) for _ in range(4099)),
        }
        for n, data in files.items():
            with open(os.path.join(d, n), "wb") as f:
                f.write(data)
        os.makedirs(os.path.join(d, "dx12"))
        with open(os.path.join(d, "dx12", "k.dxil"), "wb") as f:
            f.write(bytes(rng.getrandbits(8) for _ in range(1000)))
        with open(os.path.join(d, "dx12", "manifest.json"), "w") as f:
            json.dump({"format": 1, "kernels": [{"name": "k", "file": "k.dxil"}]}, f)
        return d, list(files)

    def _write(self, d, names):
        with mock.patch.object(build, "EMBED_FILES", names), mock.patch.object(build, "EXTRA_ARCHS", []):
            return build.write_embedded(d)

    def test_every_file_byte_identical(self):
        d, names = self._folder()
        path = self._write(d, names)
        got = dump(path)
        res = check_folder(d, got)
        # the named files, the primary architecture's copies of the code objects, the manifest and its DXIL
        expect = set(names) | {build.ARCH + "." + n for n in names if n.endswith(".co")} | {"manifest.json", "k.dxil"}
        self.assertEqual(set(res), expect)
        self.assertTrue(all(ok for ok, _ in res.values()), res)
        self.assertEqual(sorted(table_names(path)), sorted(expect))   # the table line keeps its format

    def test_mutant_caught(self):
        """one byte of one literal changed: the comparison must fail for exactly that file"""
        d, names = self._folder()
        path = self._write(d, names)
        with open(path) as f:
            text = f.read()
        i = text.index("static const unsigned char f3[] =")   # d.co (random bytes)
        j = text.index('"', i) + 1
        c = text[j]
        text = text[:j] + ("\\001" if c != "\\" else "A") + text[j + (4 if c == "\\" else 1):]
        with open(path, "w") as f:
            f.write(text)
        res = check_folder(d, dump(path))
        self.assertEqual(sorted(n for n, (ok, _) in res.items() if not ok), ["d.co"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="a build folder (its code objects, dx12/manifest.json)")
    ap.add_argument("--model-l", action="store_true")
    ap.add_argument("--no-model-k", action="store_true")
    ap.add_argument("--cxx", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    build.set_model_l(a.model_l)
    build.set_model_k(not a.no_model_k)
    path = build.write_embedded(a.build)
    got = dump(path, a.cxx, "-O2")
    res = check_folder(a.build, got)
    bad = sorted(n for n, (ok, _) in res.items() if not ok)
    rec = {"files": len(res), "bytes": sum(s for _, s in res.values()), "mismatches": bad,
           "table": {n: {"bytes": s, "sha256_match": ok} for n, (ok, s) in sorted(res.items())}}
    print("files %d, bytes %d, SHA-256 mismatches %d" % (rec["files"], rec["bytes"], len(bad)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
