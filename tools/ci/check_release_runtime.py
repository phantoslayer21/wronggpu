# SPDX-License-Identifier: GPL-3.0-only
"""CI (R1): the package gate, run on the REAL runtime library built from this tree, not on made-up files.

    python tools/ci/check_release_runtime.py --out <folder> --cxx <clang++>

Builds the runtime library twice with clean/runner/build.py (--runtime-only --no-embed: the C++ runtime without the GPU code
objects, which CI cannot build): in the RELEASE configuration (what the package gets: Model L and Model K on) and as a
developer build (--dev-diagnostics). Then checks, with the functions the packager uses (clean/tests/check_no_dev_diagnostics.py):

  * the release library says dev-diagnostics=0, carries none of the developer-diagnostic code texts, has a build manifest
    with dev_diagnostics false and a matching SHA-256, and the real payload packager (clean/installer/tools/make_payload.py)
    accepts it;
  * the developer library carries those texts (so the scan means something), says dev-diagnostics=1, has a manifest saying
    dev_diagnostics true, and the packager refuses it; with its marker forged to =0 it is still refused (the code texts and
    its manifest); a release library stripped of its marker, or changed after its build, is refused.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "clean", "tests"))
import check_no_dev_diagnostics as chk  # noqa: E402

BUILD = os.path.join(ROOT, "clean", "runner", "build.py")
PACKAGER = os.path.join(ROOT, "clean", "installer", "tools", "make_payload.py")
FAILURES = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILURES.append(what)


def build(out, cxx, dev):
    cmd = [sys.executable, BUILD, "--out", out, "--runtime-only", "--no-embed", "--model-l", "--runtime-cxx", cxx]
    if dev:
        cmd.append("--dev-diagnostics")
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit("the runtime build failed (%d)" % r.returncode)
    dll = os.path.join(out, "wronggpu_runtime.dll" if sys.platform == "win32" else "libwg_runtime.so")
    with open(dll, "rb") as f:
        data = f.read()
    with open(os.path.splitext(dll)[0] + ".build.json", "rb") as f:
        manifest = f.read()
    return dll, data, manifest


def pack(dll_name, data, manifest):
    """Runs the real packager on a payload folder holding the library (and its manifest) next to a front end."""
    d = tempfile.mkdtemp(prefix="wg_gate_")
    payload = os.path.join(d, "payload")
    os.makedirs(payload)
    with open(os.path.join(payload, "amd_fidelityfx_dx12.dll"), "wb") as f:
        f.write(b"front end")
    with open(os.path.join(payload, dll_name), "wb") as f:
        f.write(data)
    if manifest is not None:
        with open(os.path.join(payload, os.path.splitext(dll_name)[0] + ".build.json"), "wb") as f:
            f.write(manifest)
    with open(os.path.join(d, "map.json"), "wb") as f:
        f.write(b"{}")
    r = subprocess.run([sys.executable, PACKAGER, "--payload", payload, "--map", os.path.join(d, "map.json"), "--version", "0.0.1",
                        "--out", os.path.join(d, "out")], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cxx", required=True, help="the host C++ compiler (clang++)")
    a = ap.parse_args()
    name = "wronggpu_runtime.dll"

    rel_dll, rel, rel_m = build(os.path.join(a.out, "release"), a.cxx, False)
    dev_dll, dev, dev_m = build(os.path.join(a.out, "dev"), a.cxx, True)

    ok, text = chk.check_runtime_with_manifest(name, rel, rel_m)
    check(ok, "release library: " + text)
    check(not chk.dev_only_strings_in(rel), "release library carries none of the developer-diagnostic code texts")
    check(chk.main([os.path.dirname(rel_dll)]) == 0, "check_no_dev_diagnostics on the release build folder")
    code, out = pack(name, rel, rel_m)
    check(code == 0, "the payload packager accepts the release library (%s)" % ("ok" if code == 0 else out[-300:]))

    check(bool(chk.dev_only_strings_in(dev)), "the developer library carries the developer-diagnostic code texts (the scan means something)")
    ok, text = chk.check_runtime_with_manifest(name, dev, dev_m)
    check(not ok and "dev-diagnostics=1" in text, "developer library refused: " + text)
    code, out = pack(name, dev, dev_m)
    check(code != 0 and "dev-diagnostics=1" in out, "the payload packager refuses the developer library")
    forged = dev.replace(b"wg-build dev-diagnostics=1", b"wg-build dev-diagnostics=0")
    check(forged != dev, "the marker of the developer library was forged to =0")
    ok, text = chk.check_runtime_with_manifest(name, forged, dev_m)
    check(not ok, "forged marker: refused (" + text + ")")
    code, out = pack(name, forged, dev_m)
    check(code != 0, "the payload packager refuses the forged developer library")
    code, out = pack(name, forged, None)
    check(code != 0 and "forged marker" in out, "forged marker, no manifest: refused by the code-text scan")
    stripped = rel.replace(b"wg-build dev-diagnostics=0", b"wg-build dev-diagnostics=?")
    code, out = pack(name, stripped, rel_m)
    check(code != 0 and "marker" in out, "a stripped release library is refused")
    code, out = pack(name, rel + b"\0", rel_m)
    check(code != 0 and "does not match its build manifest" in out, "a release library changed after its build is refused")
    code, out = pack(name, rel, None)
    check(code != 0 and "no build manifest" in out, "a release library without its manifest is refused")

    print("release sha256 %s" % hashlib.sha256(rel).hexdigest()[:16])
    if FAILURES:
        print("FAILED: %d check(s)" % len(FAILURES))
        return 1
    print("package gate: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
