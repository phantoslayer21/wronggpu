# SPDX-License-Identifier: MIT
"""The runtime's own loading of the converter's model.wgcm (src/rr_model.cpp, src/rr_pack.cpp) against the Python packing
(tools/pack_model.py pack() of model/model_file.py's logical model): the same device image (blob for blob), the same
blob names and offsets, the same scalars; rho of page 55.3 against head/fp.py tanh32 (the binary64 path with its guard,
and every entry through the double-double path); refusals of damaged files. CPU only (the DLL's test exports
WrongGPU_RrModelDumpV1 / WrongGPU_RrRhoV1 do no GPU work).

    python -m clean.rr_f.runtime.tests.test_model_pack [--dll <wronggpu_rr_runtime.dll>] [--model <model.wgcm> ...]
"""
import argparse
import ctypes
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_DLL = os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "build")), "rr-f-rt-int", "out", "wronggpu_rr_runtime.dll")
DEFAULT_MODELS = (os.environ.get("WG_RR_MODEL_V0", ""),
                  os.environ.get("WG_RR_MODEL", ""))


def load_dll(path):
    d = ctypes.CDLL(path)
    d.WrongGPU_RrModelDumpV1.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    d.WrongGPU_RrModelDumpV1.restype = ctypes.c_uint32
    d.WrongGPU_RrRhoV1.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    d.WrongGPU_RrRhoV1.restype = ctypes.c_uint32
    return d


def cpp_dump(dll, model, tmp):
    out = os.path.join(tmp, "dump")
    rc = dll.WrongGPU_RrModelDumpV1(model, out, 0)
    txt = open(out + ".txt", encoding="utf-8").read().splitlines()
    if rc != 0:
        return None, txt
    blobs, scalars, info = {}, {}, {}
    for line in txt:
        f = line.split()
        if f[0] == "blob":
            blobs[f[1]] = (int(f[2]), int(f[3]))
        elif f[0] == "scalar":
            scalars[f[1]] = int(f[2])
        else:
            info[f[0]] = f[1]
    img = np.fromfile(out + ".img", np.uint8)
    return dict(blobs=blobs, scalars=scalars, info=info, image=img), txt


def check_model(dll, model, tmp):
    from clean.rr_f.runtime.tools import pack_model
    from clean.rr_f.head import fp
    rows = []
    got, txt = cpp_dump(dll, model, tmp)
    if got is None:
        return [("load", False, " ".join(txt))]
    m = pack_model.wgcm_logical(model)
    data, entries, sc = pack_model.pack(m)
    # the Python container's device image: after the tables, padded to 256
    nb, ns = len(entries), len(sc)
    off = 64 + nb * 64 + ns * 52
    off = (off + 255) // 256 * 256
    py_img = np.frombuffer(data[off:len(data) - 32], np.uint8)
    rows.append(("blob names", set(entries) == set(got["blobs"]) - {"in.LOGT", "in.EXPT"},
                 f"{len(entries)} python, {len(got['blobs'])} runtime"))
    bad = []
    for n, (o, size) in entries.items():
        go, gs = got["blobs"].get(n, (None, None))
        if gs != size or go != o or not np.array_equal(got["image"][go:go + gs], py_img[o:o + size]):
            bad.append(n)
    rows.append(("blob bytes and offsets", not bad, f"{len(entries) - len(bad)} / {len(entries)} equal" +
                 (f"; differing {bad[:8]}" if bad else "")))
    rows.append(("device image", np.array_equal(got["image"][:py_img.size], py_img),
                 f"{py_img.size} bytes python, {got['image'].size} runtime (+ the input stage's ff tables)"))
    sbad = [n for n in sc if got["scalars"].get(n) != (sc[n] & 0xffffffff)]
    rows.append(("scalars", not sbad and set(sc) == set(got["scalars"]), f"{len(sc)} python, {len(got['scalars'])} runtime" +
                 (f"; differing {sbad[:8]}" if sbad else "")))
    rows.append(("variant", (got["info"].get("flags") == "1") == (m["variant"] == "B2"), f"{m['variant']}"))
    # rho: the guarded binary64 path and the double-double path against head/fp.py (correctly rounded tanh)
    s2 = np.float32(m["groups"]["D0-R"]["s2"].reshape(-1)[0])
    want = fp.tanh32((fp.all_f16().astype(np.float32) * s2).astype(np.float32)).astype(np.float32)
    for force in (0, 1):
        buf = (ctypes.c_float * 65536)()
        und = dll.WrongGPU_RrRhoV1(int(np.asarray(s2).view(np.uint32)), buf, force)
        r = np.frombuffer(buf, np.float32)
        same = (r.view(np.uint32) == want.view(np.uint32)) | (np.isnan(r) & np.isnan(want))
        rows.append((f"rho ({'double-double, every entry' if force else 'binary64 + guard'})", bool(same.all()),
                     f"{int(same.sum())} / 65536 equal; {und} entries the binary64 value did not decide"))
    return rows


def damaged(dll, model, tmp):
    rows = []
    data = bytearray(open(model, "rb").read())
    cases = []
    d1 = bytearray(data)
    d1[-100] ^= 1                                              # tensor data: whole-file hash
    cases.append(("whole-file hash", d1))
    import hashlib
    import struct
    first = (88 + struct.unpack_from("<Q", data, 80)[0] + 63) // 64 * 64   # the first tensor's data
    d2 = bytearray(data)
    d2[first] ^= 1                                             # tensor data with a fixed-up trailer: the group digest
    d2[-32:] = hashlib.sha256(bytes(d2[:-32])).digest()
    cases.append(("group digest (57.7.3)", d2))
    d3 = bytearray(data)
    d3[8] = 1                                                  # container version
    d3[-32:] = hashlib.sha256(bytes(d3[:-32])).digest()
    cases.append(("container version", d3))
    for name, d in cases:
        p = os.path.join(tmp, "bad.wgcm")
        open(p, "wb").write(bytes(d))
        got, txt = cpp_dump(dll, p, tmp)
        rows.append((f"refuses: {name}", got is None, " ".join(txt)[:120]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dll", default=DEFAULT_DLL)
    ap.add_argument("--model", action="append")
    a = ap.parse_args(argv)
    dll = load_dll(a.dll)
    ok = True
    with tempfile.TemporaryDirectory(prefix="rr_pack_") as tmp:
        for model in a.model or DEFAULT_MODELS:
            print(model)
            rows = check_model(dll, model, tmp) + damaged(dll, model, tmp)
            for name, passed, info in rows:
                print(f"  {'ok  ' if passed else 'FAIL'} {name}: {info}")
                ok = ok and passed
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
