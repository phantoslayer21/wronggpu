# SPDX-License-Identifier: MIT
"""R1: a release build or a package never holds the developer diagnostics (executor=hip, the frame capture).

The runtime library says which kind of build it is: the text "wg-build dev-diagnostics=0" (release) or
"wg-build dev-diagnostics=1" (a developer build made with --dev-diagnostics; clean/runtime/interop/backend.cpp
wgBuildMarker). This checks libraries and packages for it:

    python clean/tests/check_no_dev_diagnostics.py <path> [...]

The marker is a statement of the build; the gate does not rest on it alone. Two more layers: (1) a library built without
WG_DEV_DIAGNOSTICS has no code or text for the diagnostics at all, so none of the texts that exist only in that code may be
found in it (DEV_ONLY_STRINGS: a developer build whose marker was forged to =0 still carries them); (2) the payload packager
also demands the library's BUILD MANIFEST (clean/runner/build.py writes <library>.build.json: the build's configuration and the
SHA-256 of the library) and accepts a library only when its SHA-256 matches a manifest that says dev_diagnostics = false
and records that the D3D12 kernels it embeds were checked against their sources ("dx12_kernels": null, or provenance
"checked": clean/runner/build.py refuses test-only, bypassed, define-built or stale kernels before embedding them)
(check_runtime_with_manifest). A loose D3D12 kernel manifest (dx12/manifest.json) in a payload or package fails when it is
test-only or holds a bypassed, define-built or unprovenanced entry (dx12_manifest_problem).

A path is a runtime library (wronggpu_runtime*.dll), a folder (searched for libraries, zip files and executables), a zip
file (every member), or an executable (the launcher embeds the payload's files as they are, so its bytes hold the marker).
Exit status 0 only when every runtime library found says dev-diagnostics=0, no other file says =1, and each path gave
evidence of at least one "dev-diagnostics=0" library: a library with no marker (an older build), a developer build, or a
path with nothing to check fails. This is the gate of the payload packager (clean/installer/tools/make_payload.py, which
refuses the payload) and of the release script on the final package (tools/release/check_package_diagnostics.py); the same
functions are used by clean/tests/test_dev_diagnostics_off.py in CI.
"""
import hashlib
import json
import os
import re
import sys
import zipfile

MARKER = re.compile(rb"wg-build dev-diagnostics=([01])")

# Texts that exist only in the developer diagnostics' code (clean/runtime/interop/executor_select.h, backend.cpp, all behind
# WG_DEV_DIAGNOSTICS): a release build's library contains none of them.
DEV_ONLY_STRINGS = (
    b"diagnostic: executor=hip selects the HIP executor",
    b"diagnostic: frame capture forces the HIP executor",
    b"frame capture armed: create ",
    b"diagnostic: hybrid tables off",
    # the check routes of Model L / M (runtime.cpp, behind WG_DEV_DIAGNOSTICS): the mutant and twin code objects
    b"/l_q7__hybmut.co",
    b"/l_q7w__hybmut.co",
    b"(WG_L_FORM=hybrid_twin: l_swin_hyb.co, swin_hyb.co)", b"WG_DEV_BNECK_ABORT",
    # the check routes' selector values (WG_L_FORM / WG_M_FORM): a release build does not know their names
    b"hybrid_mutant", b"hybrid_twin", b"__hybmut",
)

# Check-only code objects (mutants, probes, the HIP twins of the D3D12 hybrid ports, the fused-square scan's mutants): never
# in a release payload, a package or a runtime library's built-in files.
# (The HIP self-test's probe.co and the D3D12 device's native_driver_probe are not check objects.)
PAYLOAD_CHECK_NAMES = re.compile(r"(mut\b|mut\.co|_mut\.dxil|_mutdump\.dxil|_dump\.dxil|_tmo\.dxil|_scanmut|_probe(__mut)?\.co|l_hyb_probe|(^|[\\/]|gfx\w+\.)l?_?swin_hyb\.co)",
                                 re.I)


def check_object_names_in(data):
    """Built-in check-object names in a runtime library (embedded files are listed by name)."""
    return [t.decode() for t in (b"l_q7__hybmut.co", b"l_q7w__hybmut.co", b"l_post_c12_probe", b"l_swin_hyb.co",
                                 b"swin_hyb.co", b"_scanmut", b"_mut.dxil", b"_mutdump.dxil", b"_dump.dxil",
                                 b"_tmo.dxil", b"l_hyb_probe", b'_hyb_mut"', b'_hyb_tmo"', b'_hyb_dump"') if t in data]


# Switches of the test programs (never in a release artifact: a file that names one is a test build)
TEST_HARNESS_STRINGS = (b"WG_TEST_JITTER_OUT",)


def test_harness_strings_in(data):
    return [t.decode("ascii") for t in TEST_HARNESS_STRINGS if t in data]


def dev_only_strings_in(data):
    return [t.decode("ascii") for t in DEV_ONLY_STRINGS if t in data]


def _base(name):
    """Last component of a payload path: the packager writes Windows paths (backslash), which os.path.basename
    does not split on Linux."""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def is_runtime_name(name):
    n = _base(name).lower()
    return n.startswith("wronggpu_runtime") and n.endswith(".dll")


def check_bytes(data):
    """(True, text) when the library data says dev-diagnostics=0 once and never =1 (a runtime library)."""
    found = MARKER.findall(data)
    if not found:
        return False, "no 'wg-build dev-diagnostics=' marker (a build older than R1, or not the runtime library)"
    if any(v == b"1" for v in found):
        return False, "built with WG_DEV_DIAGNOSTICS (dev-diagnostics=1)"
    carried = dev_only_strings_in(data)
    if carried:
        return False, "says dev-diagnostics=0 but carries developer-diagnostic code (%r): a forged marker" % carried[0]
    objs = check_object_names_in(data)
    if objs:
        return False, "carries check-only code objects (%r)" % objs[0]
    harness = test_harness_strings_in(data)
    if harness:
        return False, "carries a test program's switch (%r)" % harness[0]
    return True, "dev-diagnostics=0"


def check_runtime_with_manifest(name, data, manifest_bytes):
    """(ok, text) for a runtime library the packager is about to embed: the marker and the negative scan (check_bytes) AND
    the build manifest next to the library: present, format 1, for this file, SHA-256 equal to the library's, and
    dev_diagnostics false. Fails closed on anything else (no manifest, unreadable, another library's, a library that was
    changed after its build: stripped, signed, replaced)."""
    ok, text = check_bytes(data)
    if not ok:
        return False, text
    if manifest_bytes is None:
        return False, "no build manifest (%s.build.json next to the library): build it with clean/runner/build.py" % os.path.splitext(_base(name))[0]
    try:
        m = json.loads(manifest_bytes.decode("utf-8"))
        assert isinstance(m, dict)
    except Exception:
        return False, "the build manifest is not readable JSON"
    if m.get("format") != 1:
        return False, "the build manifest has an unknown format"
    if str(m.get("file", "")).lower() != _base(name).lower():
        return False, "the build manifest is for another file (%s)" % m.get("file")
    if str(m.get("sha256", "")).lower() != hashlib.sha256(data).hexdigest():
        return False, "the library does not match its build manifest (SHA-256): changed after its build, or another build's file"
    if m.get("dev_diagnostics") is not False:
        return False, "the build manifest says dev_diagnostics is not false"
    if "dx12_kernels" not in m:
        return False, ("the build manifest does not record the check of the D3D12 kernels the library embeds (a build "
                       "older than that check): rebuild it with clean/runner/build.py")
    dk = m["dx12_kernels"]
    if dk is not None and not (isinstance(dk, dict) and dk.get("provenance") == "checked"):
        return False, "the build manifest says the library's D3D12 kernels were not checked against their sources"
    return True, "dev-diagnostics=0, matches its build manifest"


def dx12_manifest_problem(name, data):
    """None, or why a file named manifest.json that is a D3D12 kernel manifest (clean/dx12/build_dx12.py: format, kernels)
    must not ship: test-only (or older than the flag), or an entry built past the softmax self-check reference, with
    extra defines, or without provenance."""
    if _base(name).lower() != "manifest.json":
        return None
    try:
        doc = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not (isinstance(doc, dict) and isinstance(doc.get("kernels"), list) and "format" in doc):
        return None
    if doc.get("test_only") is not False:
        return "a test-only D3D12 kernel manifest (or one older than the test-only flag)"
    for e in doc["kernels"]:
        if not isinstance(e, dict) or e.get("smx_bypass") or e.get("defines") or "source_digest" not in e or "defines" not in e:
            return "a D3D12 kernel manifest with a bypassed, define-built or unprovenanced entry (%s)" % (
                e.get("name") if isinstance(e, dict) else e)
    return None


def check_artifact(name, data):
    """(ok, text, evidence) for one file by its name: a runtime library must carry the =0 marker; any other file only
    must not say =1. evidence: the file says dev-diagnostics=0 (so a package check knows it looked at a runtime)."""
    found = MARKER.findall(data)
    zero = any(v == b"0" for v in found)
    if is_runtime_name(name):
        ok, text = check_bytes(data)
        return ok, text, ok
    if PAYLOAD_CHECK_NAMES.search(_base(name)):
        return False, "a check-only code object (mutant, probe or HIP twin) in a release payload", False
    why = dx12_manifest_problem(name, data)
    if why:
        return False, why, False
    if any(v == b"1" for v in found):
        return False, "carries a developer runtime (dev-diagnostics=1)", False
    carried = dev_only_strings_in(data)
    if carried:
        return False, "carries developer-diagnostic code (%r)" % carried[0], False
    harness = test_harness_strings_in(data)
    if harness:
        return False, "carries a test program's switch (%r)" % harness[0], False
    return True, "dev-diagnostics=0" if zero else "nothing to check", zero


def artifacts(path):
    """[(label, name, data)] of one path: the file itself, a folder's libraries / zips / executables, a zip's members."""
    out = []
    if os.path.isdir(path):
        for dirpath, _, names in os.walk(path):
            for n in sorted(names):
                low = n.lower()
                if is_runtime_name(n) or low.endswith((".zip", ".exe", ".asi")) or low == "manifest.json":
                    out += artifacts(os.path.join(dirpath, n))
                elif PAYLOAD_CHECK_NAMES.search(n):   # a check-only object anywhere in the folder fails the check
                    out.append((os.path.join(dirpath, n), n, b""))
        return out
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                out.append((path + "!" + info.filename, info.filename, z.read(info)))
        return out
    with open(path, "rb") as f:
        out.append((path, path, f.read()))
    return out


def check_path(path):
    """[(label, ok, text, evidence)]; an empty list or no evidence is a failure of the caller to find a library."""
    results = []
    for label, name, data in artifacts(path):
        ok, text, evidence = check_artifact(name, data)
        if ok and text == "nothing to check":
            continue  # a file that is neither a runtime nor says anything: not listed
        results.append((label, ok, text, evidence))
    return results


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    bad = False
    for path in argv:
        results = check_path(path)
        if not any(e for _, _, _, e in results):
            print(f"FAIL {path}: no runtime library with a 'dev-diagnostics=0' marker found")
            bad = True
        for label, ok, text, _ in results:
            print(("ok   " if ok else "FAIL ") + f"{label}: {text}")
            bad = bad or not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
