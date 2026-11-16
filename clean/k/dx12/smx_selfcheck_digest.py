# SPDX-License-Identifier: MIT
"""The source digest of model K's softmax self-check reference (clean/k/runtime/k_smx_selfcheck.h). WrongGPU's own code.

The reference hashes in k_smx_selfcheck.h are valid only for the shaders and the synthetic data they were generated from.
The digest is a SHA-256 over:
- the @wg-kernel definition lines of every self-check kernel (both forms: the names, code objects and defines);
- the text of every source file those kernels compile from (their .hlsl file and its #include closure in
  clean/k/dx12/hlsl and clean/dx12/hlsl; an include generated into the build folder is covered by its generator,
  clean/k/dx12/gen_k_fp_tables.py);
- the runtime regions marked "SMX-SELFCHECK-SOURCE BEGIN" / "END" in clean/k/runtime/runtime_k.cpp and model_k.cpp (the
  cases, the synthetic data, the block parameters, the kernel-name rule and the window-offset rule);
- the extra preprocessor defines of a build (build_dx12.py --define).
gen_k_smx_selfcheck.py writes it into the header (kSourceDigest); build_dx12.py and build.py refuse to build when the
current digest differs from the header's (WG_DX12_SMX_DIGEST=0 lets a negative-control build through only into a
test-only output, build_dx12.py --test-only, which build.py never embeds); clean/k/tests/test_k_smx_selfcheck_digest.py checks the committed header and that a changed byte is caught.
"""
import hashlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
HEADER = os.path.join("clean", "k", "runtime", "k_smx_selfcheck.h")
KERNEL_DIR = os.path.join("clean", "k", "dx12", "kernels")
INCLUDE_DIRS = [os.path.join("clean", "k", "dx12", "hlsl"), os.path.join("clean", "dx12", "hlsl")]
GENERATED_BY = os.path.join("clean", "k", "dx12", "gen_k_fp_tables.py")
RUNTIME_FILES = [os.path.join("clean", "k", "runtime", "runtime_k.cpp"), os.path.join("clean", "k", "runtime", "model_k.cpp")]
# the softmax stage kernels the self-check runs (runtime_k.cpp kSmxCases), both forms
BASE = ["k_stage_e2", "k_stagew2_e3", "k_stagew2_d3", "k_stage_d2", "k_stage2_d1", "k_stagew4_e4", "k_stagew4_d4",
        "k_stagew2_e4", "k_stagew2_d4", "k_stagew4_m"]
KERNELS = sorted(BASE + [k + "__rodd" for k in BASE])
DECL = re.compile(r"^//\s*@wg-kernel\s+(\S+)\s.*$", re.M)
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
REGION = re.compile(r"// SMX-SELFCHECK-SOURCE BEGIN.*?// SMX-SELFCHECK-SOURCE END", re.S)


def _text(root, rel):
    with open(os.path.join(root, rel), "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def _closure(root, rel, seen, missing):
    if rel in seen:
        return
    seen.add(rel)
    for inc in INCLUDE.findall(_text(root, rel).decode("utf-8", "replace")):
        for d in [os.path.dirname(rel)] + INCLUDE_DIRS:
            cand = os.path.normpath(os.path.join(d, inc))
            if os.path.isfile(os.path.join(root, cand)):
                _closure(root, cand, seen, missing)
                break
        else:
            missing.add(inc)


def digest(root=ROOT, defines=()):
    """The digest (64 hex digits) of the self-check's shader sources, definitions, runtime regions and extra defines."""
    lines, files, missing = {}, set(), set()
    kdir = os.path.join(root, KERNEL_DIR)
    for fn in sorted(os.listdir(kdir)):
        if not fn.endswith(".hlsl"):
            continue
        rel = os.path.join(KERNEL_DIR, fn)
        for m in DECL.finditer(_text(root, rel).decode("utf-8", "replace")):
            if m.group(1) in KERNELS:
                lines[m.group(1)] = m.group(0).strip()
                _closure(root, rel, files, missing)
    absent = sorted(set(KERNELS) - set(lines))
    if absent:
        raise ValueError("self-check kernels not declared: %s" % absent)
    h = hashlib.sha256()
    h.update(b"kernels\n")
    for k in sorted(lines):
        h.update(lines[k].encode() + b"\n")
    h.update(b"sources\n")
    for rel in sorted(files, key=lambda r: r.replace(os.sep, "/")):
        h.update(rel.replace(os.sep, "/").encode() + b"\n" + _text(root, rel) + b"\n")
    if missing:   # includes generated into the build folder: their generator
        h.update(b"generated " + ",".join(sorted(missing)).encode() + b"\n" + _text(root, GENERATED_BY) + b"\n")
    h.update(b"runtime\n")
    for rel in RUNTIME_FILES:
        for r in REGION.findall(_text(root, rel).decode("utf-8", "replace")):
            h.update(r.encode() + b"\n")
    h.update(b"defines\n" + "\n".join(sorted(defines)).encode() + b"\n")
    return h.hexdigest()


def header_digest(root=ROOT):
    """kSourceDigest of the header, or None."""
    p = os.path.join(root, HEADER)
    if not os.path.isfile(p):
        return None
    m = re.search(r'kSourceDigest\[\]\s*=\s*"([0-9a-f]{64})"', _text(root, HEADER).decode("utf-8", "replace"))
    return m.group(1) if m else None


def stale(root=ROOT, defines=()):
    """None when the header matches the current sources and defines, else the reason (one line)."""
    want, got = header_digest(root), digest(root, defines)
    if want == got:
        return None
    return ("clean/k/runtime/k_smx_selfcheck.h was generated from other shader sources or defines (header %s, this build "
            "%s): regenerate it with clean/k/dx12/gen_k_smx_selfcheck.py" % (want, got))
