#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end tests of wronggpu-convert on synthetic files (spec 08 sections 4-9).

Written for this project; Python standard library only. Never uses a real model file: every "DLL"
here is random bytes with our own fake source blocks, and every map is built for those blocks with
hashes and digests computed by the independent reference in synth.py.

    python run_tests.py --exe <path to wronggpu-convert> [--work <scratch dir>] [--map <shipped map>]
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases  # noqa: E402
from synth import keyed_sha256, shipped_like_case  # noqa: E402

DEFAULT_MAP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "preset-M.v1.map.json"))
DEFAULT_KMAP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "preset-K.v1.map.json"))
DEFAULT_LMAP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "preset-L.v1.map.json"))
MSG_NO_PRESET_L = "This DLSS file has no preset-L model."  # the message's start (its minimum version: api.cpp)
DEFAULT_T5 = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "k48-capped-softmax-test.json"))
DEFAULT_T7 = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "lz4-block-test.json"))
DTYPE_CODE = {"float32": 0, "uint8": 1, "int32": 2}

EXIT = {"ok": 0, "usage": 1, "output": 2, "unreadable": 10, "not_model": 11, "no_preset_m": 12, "no_preset_k": 12,
        "unsupported": 13, "map_damaged": 20, "map_misfit": 21, "bad_value": 22, "self_check": 23}

MSG = {
    "unreadable": "This file cannot be read",
    "not_model": "This is not a DLSS model file",
    "no_preset_m": "This DLSS file has no preset-M model. Version 310.5 or newer is needed.",
    "no_preset_k": "This DLSS file has no preset-K model. Version 310.1 or newer is needed.",
    "unsupported": "This DLSS build is not supported yet (model data found, but other parts differ).",
    "map_damaged": "converter data is damaged",
    "map_misfit": "does not fit",
    "bad_value": "not a valid number",
    "self_check": "Conversion self-check failed.",
}
SUCCESS = "The model was converted successfully."
# Spec 08 section 8: not an error; one more line for the player after the success message.
FILTER_NOTE = "The output filter of this DLSS build is not known to this version; WrongGPU's own filter will be used."


class Runner:
    def __init__(self, exe, work):
        self.exe = exe
        self.work = work
        self.failures = []
        self.passed = 0

    def path(self, *parts):
        p = os.path.join(self.work, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def write(self, name, data):
        p = self.path(name)
        with open(p, "wb") as f:
            f.write(data if isinstance(data, bytes) else json.dumps(data, indent=1).encode())
        return p

    def run(self, map_path, dlls, out, raw=True, extra=()):
        cmd = [self.exe, "--map", map_path, "--out", out]
        for d in dlls:
            cmd += ["--dll", d]
        if raw:
            cmd.append("--raw")
        cmd += list(extra)
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
        else:
            self.failures.append(name + (": " + detail if detail else ""))
            print("FAIL", name, detail)


# ---------------------------------------------------------------- output checks

def read_container(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"WGCMODEL", "magic"
    body, trailer = data[:-32], data[-32:]
    assert hashlib.sha256(body).digest() == trailer, "trailer"
    version, nstages = struct.unpack_from("<II", data, 8)
    assert version == 1
    key = data[16:48]
    map_sha = data[48:80]
    (index_size,) = struct.unpack_from("<Q", data, 80)
    pos = 88
    stages = []
    for _ in range(nstages):
        (n,) = struct.unpack_from("<H", data, pos)
        name = data[pos + 2: pos + 2 + n].decode()
        pos += 2 + n
        digest = data[pos: pos + 32]
        pos += 32
        (nt,) = struct.unpack_from("<I", data, pos)
        pos += 4
        tensors = []
        for _ in range(nt):
            (n,) = struct.unpack_from("<H", data, pos)
            tname = data[pos + 2: pos + 2 + n].decode()
            pos += 2 + n
            dtype, rank = data[pos], data[pos + 1]
            pos += 2
            shape = list(struct.unpack_from("<%dI" % rank, data, pos))
            pos += 4 * rank
            off, size = struct.unpack_from("<QQ", data, pos)
            pos += 16
            assert off % 64 == 0
            tensors.append((tname, dtype, shape, data[off: off + size]))
        stages.append((name, digest, tensors))
    assert pos == 88 + index_size, "index size"
    return key, map_sha, stages


def check_output(r, label, case, out_dir, map_bytes, raw=True):
    """Compares everything the converter wrote with the reference."""
    try:
        key, map_sha, stages = read_container(os.path.join(out_dir, "model.wgcm"))
    except Exception as exc:  # noqa: BLE001
        r.check(label + ": container readable", False, repr(exc))
        return
    r.check(label + ": container key", key == case.key)
    r.check(label + ": container map hash", map_sha == hashlib.sha256(map_bytes).digest())
    # A group read from a source with variants (filter) exists only when a variant was found: it is then in
    # case.expected, and its digest is the entry of that variant (case.found).
    map_stages = [st for st in case.map["stages"] if st["stage"] in case.expected]
    exp_stages = [st["stage"] for st in map_stages]
    r.check(label + ": stage order", [s[0] for s in stages] == exp_stages, "%s vs %s" % ([s[0] for s in stages], exp_stages))
    if raw:
        for st in case.map["stages"]:
            if st["stage"] not in case.expected:
                r.check(label + ": no raw folder for the absent group " + st["stage"],
                        not os.path.exists(os.path.join(out_dir, "raw", st["stage"])))
    for (name, digest, tensors), st in zip(stages, map_stages):
        want = st["digest"] if isinstance(st["digest"], str) else st["digest"][case.found[name]]
        r.check(label + ": digest " + name, digest.hex() == want)
        concat = b"".join(t[3] for t in tensors)
        r.check(label + ": recomputed digest " + name, keyed_sha256(key, concat).hex() == want)
        exp = case.expected[name]
        r.check(label + ": tensor list " + name, [t[0] for t in tensors] == [t[0] for t in exp])
        for (tname, dtype, shape, data), (ename, edata), mt in zip(tensors, exp, st["tensors"]):
            ok = data == edata and shape == mt["shape"] and dtype == DTYPE_CODE[mt["dtype"]]
            detail = ""
            if data != edata and len(data) == len(edata):
                step = 1 if mt["dtype"] == "uint8" else 4
                bad = [i for i in range(0, len(data), step) if data[i:i + step] != edata[i:i + step]]
                detail = "%d elements differ, first at %d: got %s want %s" % (
                    len(bad), bad[0] // step, data[bad[0]:bad[0] + step].hex(), edata[bad[0]:bad[0] + step].hex())
            r.check("%s: values %s/%s" % (label, name, tname), ok, detail)
            if raw:
                rp = os.path.join(out_dir, "raw", name, tname + ".bin")
                try:
                    with open(rp, "rb") as f:
                        r.check("%s: raw %s/%s" % (label, name, tname), f.read() == edata)
                except OSError as exc:
                    r.check("%s: raw %s/%s" % (label, name, tname), False, repr(exc))


def snapshot(d):
    snap = {}
    for root, dirs, files in os.walk(d):
        for fn in files:
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                snap[os.path.relpath(p, d)] = hashlib.sha256(f.read()).hexdigest()
        for dn in dirs:
            snap[os.path.relpath(os.path.join(root, dn), d) + "/"] = "dir"
    return snap


def logged_offsets(stderr):
    found = {}
    for m in re.finditer(r"source (\w+) at offsets? (\d+)", stderr):
        found[m.group(1)] = int(m.group(2))
    return found


# ---------------------------------------------------------------- tests

def test_feature(r):
    t0 = time.time()
    case = cases.feature_case()
    print("feature case built in %.1f s" % (time.time() - t0))
    dll = r.write("feature/fake.dll", case.file)
    mp = r.write("feature/map.json", case.map)
    out = r.path("feature/out", "x")[:-2]
    code, so, se = r.run(mp, [dll], out)
    r.check("feature: exit 0", code == 0, so + se)
    with open(mp, "rb") as f:
        check_output(r, "feature", case, out, f.read())
    offs = logged_offsets(se)
    r.check("feature: search order (W even before odd, D odd, K at gap 25)",
            offs == case.placements, "%s vs %s" % (offs, case.placements))
    r.check("feature: version logged", "310.99.1.0" in se, se)
    r.check("feature: log has no values", "0x3fa00000" not in se.lower())
    # Several thread counts give the same result and the same offsets.
    for threads in ("1", "3"):
        out2 = r.path("feature/out-t" + threads, "x")[:-2]
        code, so, se = r.run(mp, [dll], out2, extra=("--threads", threads))
        r.check("feature: threads " + threads, code == 0 and logged_offsets(se) == case.placements, so + se)
        try:
            with open(os.path.join(out, "model.wgcm"), "rb") as a, open(os.path.join(out2, "model.wgcm"), "rb") as b:
                same = a.read() == b.read()
        except OSError:
            same = False
        r.check("feature: threads %s identical output" % threads, same)


def test_shipped_like(r, shipped_map):
    """The shipped map's exact layout, with our own random source content. Source F (map version 4): variant 1 at a
    multiple of 16, variant 2 at an odd offset, and no block at all."""
    with open(shipped_map, "rb") as f:
        has_f = "F" in json.load(f)["sources"]
    r.check("shipped map: version 4 with source F and the group filter", has_f)
    for seed, w_off, f_present, f_shift in ((1, 4096, 0, 0), (2, 4096 + 6, 1, 1), (3, 4096 + 5, None, 0)):
        t0 = time.time()
        case = shipped_like_case(shipped_map, seed=seed, w_offset=w_off, f_present=f_present, f_shift=f_shift)
        label = "shipped-like(seed %d, W at %d)" % (seed, w_off)
        dll = r.write("shipped/%d/fake.dll" % seed, case.file)
        mp = r.write("shipped/%d/map.json" % seed, case.map)
        out = r.path("shipped/%d/out" % seed, "x")[:-2]
        code, so, se = r.run(mp, [dll], out)
        r.check(label + ": exit 0", code == 0, so + se)
        with open(mp, "rb") as f:
            check_output(r, label, case, out, f.read())
        r.check(label + ": offsets", logged_offsets(se) == case.placements, "%s vs %s" % (logged_offsets(se), case.placements))
        if has_f:
            check_filter_report(r, label, case, f_present, so, se)
            r.check(label + ": every group, or every group but filter",
                    ("all %d stage digests match" % (len(case.map["stages"]) - (f_present is None))) in se, se)
            r.check(label + ": F at a multiple of 16, or one byte after",
                    f_present is None or case.placements["F"] % 16 == f_shift)
        print("%s: %.1f s" % (label, time.time() - t0))


def test_shipped_map_validates(r, shipped_map):
    """The real map must pass validation; with our fake file it must report "no preset-M" (W absent)."""
    import random
    rng = random.Random(9)
    with open(shipped_map, "rb") as f:
        m = json.load(f)
    lw = m["sources"]["W"]["length"]
    out = r.path("shipped-real/out", "x")[:-2]
    small = r.write("shipped-real/small.dll", rng.randbytes(lw - 1))
    code, so, se = r.run(shipped_map, [small], out)
    r.check("real map: smaller than W -> not a model file", code == EXIT["not_model"] and MSG["not_model"] in so, so + se)
    big = r.write("shipped-real/nomodel.dll", rng.randbytes(lw + 100_000))
    code, so, se = r.run(shipped_map, [big], out)
    r.check("real map: valid, W absent -> no preset-M", code == EXIT["no_preset_m"] and MSG["no_preset_m"] in so, so + se)
    r.check("real map: nothing written", not os.path.exists(os.path.join(out, "model.wgcm")))


def test_l_real_map(r, lmap):
    """Model L (#89): the delivered preset-L map (data/preset-L.v1.map.json; map version 1, W, D, K; no "preset" field,
    so it is loaded with --preset L). It validates; our own random file gives preset L's message (and, without
    --preset, preset M's, as for any version-1 map); a mismatching --preset is refused; and its exact layout converts
    our own random data at two placements, every tensor of every stage checked against the reference."""
    import random
    with open(lmap, "rb") as f:
        m = json.load(f)
    r.check("preset-L map: version 1, sources W, D and K, no \"preset\" field",
            m["version"] == 1 and sorted(m["sources"]) == ["D", "K", "W"] and "preset" not in m, (m["version"], sorted(m["sources"])))
    rng = random.Random(29)
    lw = m["sources"]["W"]["length"]
    out = r.path("l-real/out", "x")[:-2]
    big = r.write("l-real/nomodel.dll", rng.randbytes(lw + 100_000))
    code, so, se = r.run(lmap, [big], out, extra=("--preset", "L"))
    r.check("preset-L map --preset L: valid, W absent -> exit 12, preset L's message", code == EXIT["no_preset_m"] and MSG_NO_PRESET_L in so, so + se[-400:])
    r.check("...the log names the preset", "preset L" in se, se[-400:])
    code, so, se = r.run(lmap, [big], out)
    r.check("preset-L map without --preset: read as version 1 (preset M's message)", code == EXIT["no_preset_m"] and MSG["no_preset_m"] in so, so)
    for asked in ("K", "F"):
        code, so, se = r.run(lmap, [big], out, extra=("--preset", asked))
        r.check("preset-L map --preset %s: converter data damaged" % asked, code == EXIT["map_damaged"] and MSG["map_damaged"] in so, so)
    code, so, se = r.run(lmap, [big], out, extra=("--preset", "Q"))
    r.check("--preset Q (unknown): usage error", code == EXIT["usage"] and "unknown preset" in so, so)
    r.check("preset-L map: nothing written", not os.path.exists(os.path.join(out, "model.wgcm")))
    for seed, w_off in ((21, 4096), (22, 4096 + 6)):
        t0 = time.time()
        case = shipped_like_case(lmap, seed=seed, w_offset=w_off)
        label = "preset-L layout(seed %d, W at %d)" % (seed, w_off)
        dll = r.write("l-real/%d/fake.dll" % seed, case.file)
        mp = r.write("l-real/%d/map.json" % seed, case.map)
        o = r.path("l-real/%d/out" % seed, "x")[:-2]
        code, so, se = r.run(mp, [dll], o, extra=("--preset", "L"))
        r.check(label + ": exit 0", code == 0 and SUCCESS in so, so + se[-600:])
        with open(mp, "rb") as f:
            check_output(r, label, case, o, f.read())
        r.check(label + ": offsets", logged_offsets(se) == case.placements, "%s vs %s" % (logged_offsets(se), case.placements))
        r.check(label + ": every stage digest matches", ("all %d stage digests match" % len(case.map["stages"])) in se, se[-400:])
        print("%s: %.1f s" % (label, time.time() - t0))


def test_preset_l(r):
    """Model L (#89): a map that names preset L ("preset": "L", map version 1) next to a preset-M map, both reading one
    synthetic library file in which L's parameter block lies directly before M's. Each map converts its own block, and
    only it; a file without a preset's block fails with that preset's message (exit 12)."""
    case_l, case_m, l_only, m_only = cases.preset_pair_case()
    lp = r.write("preset-l/map-L.json", case_l.map)
    mp = r.write("preset-l/map-M.json", case_m.map)
    both = r.write("preset-l/both.dll", case_l.file)
    lo = r.write("preset-l/l-only.dll", l_only)
    mo = r.write("preset-l/m-only.dll", m_only)
    for label, c, p in (("preset L", case_l, lp), ("preset M next to L", case_m, mp)):
        out = r.path("preset-l", "out-" + c.name, "x")[:-2]
        code, so, se = r.run(p, [both], out)
        r.check(label + ": exit 0", code == 0 and SUCCESS in so, so + se[-600:])
        with open(p, "rb") as f:
            check_output(r, label, c, out, f.read())
        r.check(label + ": W found where it was placed", ("source W at offset %d " % c.placements["W"]) in se, se[-800:])
    r.check("preset L: its extra stage is in the L output only",
            "inenc_extra" in case_l.expected and "inenc_extra" not in case_m.expected and
            os.path.isdir(os.path.join(r.work, "preset-l", "out-pair-L", "raw", "inenc_extra")) and
            not os.path.exists(os.path.join(r.work, "preset-l", "out-pair-M", "raw", "inenc_extra")))
    msg_l = "This DLSS file has no preset-L model. Version 310.9 or newer is needed."
    for label, p, f, want in (("preset-L map, a file with only preset M's block", lp, mo, msg_l),
                              ("preset-M map, a file with only preset L's block", mp, lo, MSG["no_preset_m"])):
        code, so, se = r.run(p, [f], r.path("preset-l", "out-fail", "x")[:-2])
        r.check(label + ": exit 12 with that preset's message", code == EXIT["no_preset_m"] and want in so, "code %d, out %r" % (code, so))
    # the preset name decides only the message: the same L map without its field reads as preset M, and converts the same
    plain = copy.deepcopy(case_l.map)
    del plain["preset"]
    pp = r.write("preset-l/map-L-no-field.json", plain)
    code, so, se = r.run(pp, [mo], r.path("preset-l", "out-fail2", "x")[:-2])
    r.check("the L map without its \"preset\" field: read as preset M (its message)", code == 12 and MSG["no_preset_m"] in so, so)
    out = r.path("preset-l", "out-no-field", "x")[:-2]
    code, so, se = r.run(pp, [both], out)
    r.check("...and it converts the same tensors", code == 0, so + se[-400:])
    with open(pp, "rb") as f:
        check_output(r, "L map without the field", case_l, out, f.read())
    # a preset name that does not fit the map version, or is not known: converter data damaged, before any file is read
    for bad in ("K", "F", "X", "l"):
        m2 = copy.deepcopy(case_l.map)
        m2["preset"] = bad
        code, so, se = r.run(r.write("preset-l/bad-%s.json" % bad, m2), [both], r.path("preset-l", "out-bad", "x")[:-2])
        r.check("\"preset\": %r in a version-1 map: converter data damaged" % bad, code == EXIT["map_damaged"] and MSG["map_damaged"] in so, so)


def test_l_shaped(r, m_map_path):
    """Model L (#89): an L-shaped map made from the shipped preset-M map's layout (cases.l_shaped_map: the encoder's
    output linear 64 x 128, a stage with two linears and a C = 32 Swin block before enc0, a C = 32 Swin block before
    dec0; "preset": "L"), filled with our own random data like the shipped-like cases. Every tensor of every stage is
    checked against the reference, at two placements of the block; a file without the block gives preset L's message."""
    with open(m_map_path, "rb") as f:
        lmap = cases.l_shaped_map(json.load(f))
    mp_in = r.write("l-shaped/layout.json", lmap)
    names = [st["stage"] for st in lmap["stages"]]
    r.check("L-shaped map: enc0_pre before enc0, dec0_pre before dec0, no filter group",
            names.index("enc0_pre") + 1 == names.index("enc0") and names.index("dec0_pre") + 1 == names.index("dec0") and "filter" not in names, names)
    for seed, w_off in ((7, 4096), (8, 4096 + 2)):
        t0 = time.time()
        case = shipped_like_case(mp_in, seed=seed, w_offset=w_off)
        label = "L-shaped(seed %d, W at %d)" % (seed, w_off)
        dll = r.write("l-shaped/%d/fake.dll" % seed, case.file)
        mp = r.write("l-shaped/%d/map.json" % seed, case.map)
        out = r.path("l-shaped/%d/out" % seed, "x")[:-2]
        code, so, se = r.run(mp, [dll], out)
        r.check(label + ": exit 0", code == 0 and SUCCESS in so, so + se[-600:])
        with open(mp, "rb") as f:
            check_output(r, label, case, out, f.read())
        r.check(label + ": offsets", logged_offsets(se) == case.placements, "%s vs %s" % (logged_offsets(se), case.placements))
        r.check(label + ": every stage digest matches", ("all %d stage digests match" % len(case.map["stages"])) in se, se[-400:])
        enc0 = dict((t[0], t) for t in case.expected["enc0"])
        r.check(label + ": the encoder's output linear is 64 x 128 (float32)", len(enc0["W"][1]) == 64 * 128 * 4)
        r.check(label + ": the C = 32 blocks have their 11 records",
                [t[0] for t in case.expected["dec0_pre"]] == ["gamma1", "gamma2", "Wqk", "Wv", "B", "Wp", "bp", "W1", "b1", "W2", "b2"] and
                len(case.expected["enc0_pre"]) == 15)
        print("%s: %.1f s" % (label, time.time() - t0))
    # the same layout against a file without its block: preset L's message (exit 12)
    case = shipped_like_case(mp_in, seed=9)
    mp = r.write("l-shaped/9/map.json", case.map)
    other = shipped_like_case(mp_in, seed=10)  # another random block of the same layout: not this map's
    code, so, se = r.run(mp, [r.write("l-shaped/9/other.dll", other.file)], r.path("l-shaped/9/out", "x")[:-2])
    r.check("L-shaped map, a file without its block: exit 12 with preset L's message",
            code == EXIT["no_preset_m"] and "This DLSS file has no preset-L model." in so, "code %d, out %r" % (code, so))


def mutate(m, fn):
    m2 = copy.deepcopy(m)
    fn(m2)
    return m2


def test_errors(r):
    base = cases.mini_case()
    mp = r.write("err/map.json", base.map)
    dll = r.write("err/fake.dll", base.file)
    out = r.path("err/out", "x")[:-2]

    # A first successful conversion; every failure below must leave it exactly as it is.
    code, so, se = r.run(mp, [dll], out)
    r.check("mini: exit 0", code == 0, so + se)
    with open(mp, "rb") as f:
        check_output(r, "mini", base, out, f.read())
    before = snapshot(out)

    def expect(label, code_name, map_obj=None, dlls=None, map_path=None, msg=None):
        p = map_path or (r.write("err/%s.json" % re.sub(r"\W+", "_", label), map_obj) if map_obj is not None else mp)
        code, so, se = r.run(p, dlls or [dll], out)
        want = msg if msg is not None else MSG.get(code_name, "")
        r.check(label + " -> " + code_name, code == EXIT[code_name] and want in so, "code %d, out %r, log %r" % (code, so, se[-400:]))
        r.check(label + ": earlier output untouched", snapshot(out) == before)
        return so, se

    M = base.map
    # --- section 9 / 08-T4: corrupted maps, refused before any output
    expect("unknown op", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1]["op"].update(name="f17")))
    expect("bad perm name", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][0]["axes"][1].update(perm="p5")))
    expect("shape/axes mismatch", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][0]["axes"].pop()))
    expect("wrong digest", "self_check", mutate(M, lambda m: m["stages"][1].update(
        digest=("0" if m["stages"][1]["digest"][0] != "0" else "1") + m["stages"][1]["digest"][1:])))
    # --- section 8: every other map condition
    expect("bad format", "map_damaged", mutate(M, lambda m: m.update(format="wronggpu-converter-mop")))
    expect("bad version", "map_damaged", mutate(M, lambda m: m.update(version=2)))
    expect("version 5", "map_damaged", mutate(M, lambda m: m.update(version=5)))
    expect("version 0", "map_damaged", mutate(M, lambda m: m.update(version=0)))
    expect("digest list in a version-1 map", "map_damaged", mutate(M, lambda m: m["stages"][0].update(digest=[m["stages"][0]["digest"]])))
    expect("version as text", "map_damaged", mutate(M, lambda m: m.update(version="1")))
    expect("unknown locate", "map_damaged", mutate(M, lambda m: m["sources"]["D"].update(locate="scan")))
    expect("unknown element", "map_damaged", mutate(M, lambda m: m["sources"]["D"].update(element="u8")))
    expect("unknown dtype", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(dtype="float16")))
    expect("unknown channel order", "map_damaged", mutate(M, lambda m: m.update(channel_order="interleaved")))
    expect("perm block does not divide axis", "map_damaged",
           mutate(M, lambda m: m["stages"][0]["tensors"][1]["axes"][0].update(perm="p4") or m["stages"][0]["tensors"][1].update(shape=[6]) or m["stages"][0]["tensors"][1]["axes"][0].update(stride=1)))
    expect("perm not a permutation", "map_damaged", mutate(M, lambda m: m["perms"].update(p4=[0, 1, 1, 3])))
    expect("perm out of range", "map_damaged", mutate(M, lambda m: m["perms"].update(p4=[0, 1, 2, 4])))
    expect("op does not fit element", "map_damaged",
           mutate(M, lambda m: m["stages"][0]["tensors"][2].update(op={"name": "fp8"}, dtype="float32")))
    expect("op does not fit dtype", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][2].update(dtype="float32")))
    expect("f16_scaled without div", "map_damaged", mutate(M, lambda m: m["stages"][1]["tensors"][0]["op"].pop("div")))
    expect("bad hex", "map_damaged", mutate(M, lambda m: m["stages"][1]["tensors"][0]["op"].update(mul="0xZZ")))
    expect("bad sha256", "map_damaged", mutate(M, lambda m: m["sources"]["W"].update(sha256="abc")))
    expect("pair without gaps", "map_damaged", mutate(M, lambda m: m["sources"]["K"].pop("gaps")))
    expect("key from a keyed source", "map_damaged", mutate(M, lambda m: m["key"].update(**{"from": "K"})))
    expect("unknown source in tensor", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(src="X")))
    expect("duplicate tensor", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"].append(copy.deepcopy(m["stages"][0]["tensors"][1]))))
    expect("tensor names equal ignoring case", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(name="w")))
    expect("stage names equal ignoring case", "map_damaged", mutate(M, lambda m: m["stages"][1].update(stage="ENC0")))
    expect("unsafe tensor name", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(name="../b")))
    expect("17 axes", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(shape=[1] * 16 + [8], axes=[{"stride": 0}] * 16 + [{"stride": 1}])))
    # 16 axes are allowed: [1]*15 + [8] with stride-0 axes gives the same bytes, so the digest still matches.
    m16 = r.write("err/rank16.json", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(shape=[1] * 15 + [8], axes=[{"stride": 0}] * 15 + [{"stride": 1}])))
    code, so, se = r.run(m16, [dll], r.path("err/out16", "x")[:-2])
    r.check("16 axes converts", code == 0, so + se)
    expect("zero axis length", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(shape=[0])))
    for path in (("format",), ("version",), ("sources",), ("sources", "W"), ("sources", "D"), ("sources", "K"),
                 ("sources", "W", "fnv1a64"), ("sources", "D", "sha256"), ("sources", "K", "length"),
                 ("key",), ("key", "label"), ("perms",), ("stages",), ("model",), ("channel_order",)):
        def drop(m, path=path):
            obj = m
            for k in path[:-1]:
                obj = obj[k]
            del obj[path[-1]]
        expect("missing " + ".".join(path), "map_damaged", mutate(M, drop))
    for field in ("name", "shape", "dtype", "src", "op", "base", "axes"):
        expect("missing tensor " + field, "map_damaged", mutate(M, lambda m, f=field: m["stages"][2]["tensors"][0].pop(f)))
    expect("missing stride", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1]["axes"][0].pop("stride")))
    expect("missing op name", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1]["op"].pop("name")))
    expect("missing stage digest", "map_damaged", mutate(M, lambda m: m["stages"][0].pop("digest")))
    expect("missing u16_shr shift", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][2]["op"].pop("shift")))
    r.write("err/broken.json", b'{"format": "wronggpu-converter-map", "version": 1,')
    expect("invalid JSON", "map_damaged", map_path=os.path.join(r.work, "err/broken.json"))
    expect("map file missing", "map_damaged", map_path=os.path.join(r.work, "err/no-such-map.json"))
    # Index outside its source.
    expect("index above source", "map_misfit", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(base=4096 - 7)))
    expect("index below 0 (negative stride)", "map_misfit",
           mutate(M, lambda m: m["stages"][0]["tensors"][1].update(base=3) or m["stages"][0]["tensors"][1]["axes"][0].update(stride=-1)))
    expect("K element index 2", "map_misfit", mutate(M, lambda m: m["stages"][0]["tensors"][4].update(base=2)))
    # Decoded values that are not finite / do not fit (a correct map never produces them).
    w = bytearray(base.sources["W"])
    w[2 * 96: 2 * 96 + 2] = struct.pack("<H", 0x7C00)          # +inf where b (f16) reads
    w[2 * 5: 2 * 5 + 2] = struct.pack("<H", 0x7E01)            # NaN where W (fp8) reads
    w[2 * 300: 2 * 300 + 2] = struct.pack("<H", 0x7800)        # 32768 * sqrt(8) overflows binary16 (B)
    w[2 * 400: 2 * 400 + 2] = struct.pack("<H", 0x5A00)        # e^192 overflows binary32 (s)
    w = bytes(w)
    import synth
    for label, mut in (
        ("f16 of +inf", lambda m: m["stages"].__setitem__(slice(None), [
            {"stage": "x", "digest": "0" * 64, "tensors": [cases.tensor("b", [1], "float32", "W", {"name": "f16"}, 96, [cases.lin(0)])]}])),
        ("fp8 of NaN", lambda m: m["stages"].__setitem__(slice(None), [
            {"stage": "x", "digest": "0" * 64, "tensors": [cases.tensor("W", [1], "float32", "W", {"name": "fp8"}, 5, [cases.lin(0)])]}])),
        ("f16_scaled overflow", lambda m: m["stages"].__setitem__(slice(None), [
            {"stage": "x", "digest": "0" * 64, "tensors": [cases.tensor("B", [1], "float32", "W", {"name": "f16_scaled", "mul": cases.SQRT8, "div": "0x3f800000"}, 300, [cases.lin(0)])]}])),
        ("exp overflow", lambda m: m["stages"].__setitem__(slice(None), [
            {"stage": "x", "digest": "0" * 64, "tensors": [cases.tensor("s", [1], "float32", "W", {"name": "exp"}, 400, [cases.lin(0)])]}])),
        ("u32_as_f32 NaN", lambda m: m["stages"].__setitem__(slice(None), [
            {"stage": "x", "digest": "0" * 64, "tensors": [cases.tensor("s", [1], "float32", "K", {"name": "u32_as_f32"}, 0, [cases.lin(0)])]}])),
    ):
        m2 = mutate(M, mut)
        k = struct.pack("<II", 0x7FC00001, 0x3F800000) if "u32" in label else base.sources["K"]
        m2["sources"]["W"] = synth.source_entry("W", "w", "f16le", w)
        key = synth.sha256(m2["key"]["label"].encode() + w)
        m2["sources"]["K"] = synth.source_entry("K", "k", "u32le", k, locate="pair", gaps=[10], keyed=True, key=key)
        f = bytearray(base.file)
        f[base.placements["W"]: base.placements["W"] + len(w)] = w
        f[base.placements["K"]: base.placements["K"] + 4] = k[:4]
        f[base.placements["K"] + 10: base.placements["K"] + 14] = k[4:]
        d2 = r.write("err/%s.dll" % label.replace(" ", "_"), bytes(f))
        expect(label, "bad_value", m2, dlls=[d2])
    expect("u16_shr result above 255", "map_misfit", mutate(M, lambda m: m["stages"][0]["tensors"][2]["op"].update(shift=7)))

    # --- 08-T3: damaged copies of a supported file
    W, D, K = base.placements["W"], base.placements["D"], base.placements["K"]
    lw = len(base.sources["W"])
    for label, data, code_name in (
        ("bit flipped in W (prefix)", cases.flip_bit(base.file, W + 3, 5), "no_preset_m"),
        ("bit flipped in W (middle)", cases.flip_bit(base.file, W + lw // 2, 0), "no_preset_m"),
        ("bit flipped in W (last byte)", cases.flip_bit(base.file, W + lw - 1, 7), "no_preset_m"),
        ("bit flipped in D", cases.flip_bit(base.file, D + 4000, 2), "unsupported"),
        ("bit flipped in D prefix", cases.flip_bit(base.file, D + 1, 2), "unsupported"),
        ("bit flipped in K (first word)", cases.flip_bit(base.file, K + 2, 1), "unsupported"),
        ("bit flipped in K (second word)", cases.flip_bit(base.file, K + 13, 1), "unsupported"),
        ("truncated inside W", base.file[: W + lw - 100], "no_preset_m"),
        ("truncated below the W length", base.file[: lw - 1], "not_model"),
        ("empty file", b"", "not_model"),
    ):
        d2 = r.write("err/%s.dll" % re.sub(r"\W+", "_", label), data)
        expect(label, code_name, dlls=[d2])
    # Version shown with the "no preset-M" message.
    old = cases.mini_case(version=(310, 4, 0, 0))
    lw_mini = len(old.sources["W"])
    damaged = cases.flip_bit(old.file, old.placements["W"] + lw_mini // 3, 4)
    d2 = r.write("err/old-version.dll", damaged)
    so, se = expect("no preset-M with a version resource", "no_preset_m", dlls=[d2])
    r.check("version shown to the player", "This file is version 310.4.0.0." in so, so)
    # Unreadable files.
    expect("missing file", "unreadable", dlls=[os.path.join(r.work, "err", "no-such.dll")])
    expect("directory", "unreadable", dlls=[os.path.join(r.work, "err")])
    # Leftovers: no staging folders after all those failures.
    r.check("no staging folders left", not any(n.startswith(".wronggpu-convert-staging") for n in os.listdir(out)))

    # --- several files: tried in order; the first that converts is used, every rejection reported
    nomodel = r.write("multi/nomodel.dll", cases.flip_bit(base.file, W + 10, 0))
    out_multi = r.path("multi/out", "x")[:-2]
    code, so, se = r.run(mp, [os.path.join(r.work, "multi", "missing.dll"), nomodel, dll], out_multi)
    r.check("several files: third converts", code == 0, so + se)
    r.check("several files: rejections reported",
            MSG["unreadable"] in so and MSG["no_preset_m"] in so and "converted successfully" in so, so)
    with open(mp, "rb") as f:
        check_output(r, "several files", base, out_multi, f.read())
    code, so, se = r.run(mp, [nomodel, os.path.join(r.work, "multi", "missing.dll")], r.path("multi/out2", "x")[:-2])
    r.check("several files, none converts: first file's code", code == EXIT["no_preset_m"], so + se)
    r.check("several files, none converts: both reasons", MSG["unreadable"] in so and MSG["no_preset_m"] in so, so)
    r.check("several files, none converts: nothing written", not os.path.exists(r.path("multi/out2", "model.wgcm")))
    bad_map = r.write("multi/bad.json", mutate(M, lambda m: m["stages"][1].update(digest="ab" * 32)))
    code, so, se = r.run(bad_map, [dll, dll], r.path("multi/out3", "x")[:-2])
    r.check("self-check failure stops at the first file", code == EXIT["self_check"] and se.count("rejected") == 1, so + se)

    # --- output problems and re-runs
    blocker = r.write("outfile/not-a-dir", b"x")
    code, so, se = r.run(mp, [dll], blocker)
    r.check("output folder is a file -> output error", code == EXIT["output"], so + se)
    code, so, se = r.run(mp, [dll], out)  # convert again over existing output (replacing it)
    r.check("re-run over existing output", code == 0 and snapshot(out) == before, so + se)
    out_noraw = r.path("noraw/out", "x")[:-2]
    code, so, se = r.run(mp, [dll], out_noraw, raw=False)
    r.check("without --raw: only the model file", code == 0 and sorted(os.listdir(out_noraw)) == ["model.wgcm"], str(os.listdir(out_noraw)))
    # Usage errors.
    code, so, se = r.run(mp, [], out)
    r.check("no --dll -> usage", code == EXIT["usage"])
    code, so, se = r.run(mp, [dll], out, extra=("--bogus",))
    r.check("unknown option -> usage", code == EXIT["usage"])


# ---------------------------------------------------------------- source F (spec 08 sections 4.3, 4.4; 08-T6, 08-T7)

def check_filter_report(r, label, case, variant, so, se):
    """What the converter tells the player and the log about source F; and that neither shows weights or text."""
    lines = so.splitlines()
    if variant is None:
        r.check(label + ": player note after the success line", lines == [SUCCESS, FILTER_NOTE], so)
        r.check(label + ": log says F was not found and the group is left out",
                "source F not found" in se and "no group filter" in se and FILTER_NOTE in se, se)
    else:
        r.check(label + ": no player note", lines == [SUCCESS], so)
        r.check(label + ": log names the variant",
                "source F at offset %d (variant %d of 2)" % (case.placements["F"], variant + 1) in se and
                "filter: variant %d" % (variant + 1) in se, se)
    text = (so + se).lower()
    shown = [v for v in case.variants if any(("%08x" % x) in text for x in v["elements"]) or
             v["text"][:24].decode().lower() in text or v["text"][v["fields"][0] - 30: v["fields"][0]].decode().lower() in text]
    r.check(label + ": output and log show no weights and no decoded text", not shown)


def test_filter(r, t7_path):
    import synth
    if os.path.exists(t7_path):
        n, bad = synth.check_lz4_vectors(t7_path)
        r.check("08-T7: the reference decoder agrees with the %d spec vectors" % n, n > 0 and not bad, str(bad))
    else:
        print("note: 08-T7 vectors not found at %s" % t7_path)

    def convert(label, sub, case, variant):
        """A conversion that must succeed: with the group filter of `variant`, or without it (None)."""
        dll = r.write("filter/%s/fake.dll" % sub, case.file)
        mp = r.write("filter/%s/map.json" % sub, case.map)
        out = r.path("filter/%s/out" % sub, "x")[:-2]
        code, so, se = r.run(mp, [dll], out)
        r.check(label + ": exit 0", code == 0, so + se)
        with open(mp, "rb") as f:
            check_output(r, label, case, out, f.read())
        r.check(label + ": offsets", logged_offsets(se) == case.placements, "%s vs %s" % (logged_offsets(se), case.placements))
        check_filter_report(r, label, case, variant, so, se)
        if variant is not None:
            want = b"".join(struct.pack("<I", x) for x in case.variants[variant]["elements"])
            try:
                with open(os.path.join(out, "raw", "filter", "output_taps.bin"), "rb") as f:
                    got = f.read()
            except OSError:
                got = b""
            r.check(label + ": raw/filter/output_taps.bin is five little-endian float32", got == want and len(got) == 20)
        return dll, mp, out

    def with_file(case, data, **changes):
        c = copy.copy(case)
        c.file = data
        for k, v in changes.items():
            setattr(c, k, v)
        return c

    # --- found: each variant, every kind of offset, map order, a decoy, the end of the file
    v1 = cases.f_mini_case(present=0, f_offset=70_000)
    assert v1.placements["F"] % 16 == 0
    dll1, mp, out1 = convert("F variant 1 at a multiple of 16", "v1", v1, 0)
    v2 = cases.f_mini_case(present=1, f_offset=70_001)
    dll2, _, _ = convert("F variant 2 at an odd offset", "v2", v2, 1)
    r.check("F: one map for all these files", v1.map == v2.map)
    convert("F variant 1 at an even offset after variant 2's block (map order decides)", "both",
            cases.f_mini_case(present=0, f_offset=100_006, other_offset=70_000), 0)
    convert("F variant 2 after a decoy of its first 64 bytes", "decoy",
            cases.f_mini_case(present=1, f_offset=80_002, decoy_offset=68_000), 1)
    end = cases.f_mini_case(present=0, at_end=True)
    convert("F variant 1 as the last bytes of the file", "end", end, 0)

    # --- 08-T6: not found is not an error: no block; damaged blocks; a truncated file; hashes under another key
    none = cases.f_mini_case(present=None)
    r.check("F: the same map without a block", none.map == v1.map and "filter" not in none.expected)
    convert("F absent", "none", none, None)
    F, n = v1.placements["F"], len(v1.variants[0]["block"])
    last = bytearray(v1.file)
    last[F + n - 1] = (last[F + n - 1] + 1) & 0xFF
    for sub, label, data in (
        ("flip-head", "one bit flipped in the block's first 64 bytes", cases.flip_bit(v1.file, F + 3, 6)),
        ("flip-mid", "one bit flipped inside the block", cases.flip_bit(v1.file, F + n // 2, 0)),
        ("flip-last", "one bit flipped in the block's last byte", cases.flip_bit(v1.file, F + n - 1, 7)),
        ("last-byte", "the block's last byte changed", bytes(last)),
        ("cut-1", "file truncated one byte before the end of the block", end.file[:-1]),
        ("cut-head", "file truncated inside the block's first 64 bytes", end.file[: end.placements["F"] + 40]),
    ):
        convert("F: " + label + " -> converted without the group", sub, with_file(none, data), None)
    other_key = synth.sha256(b"another key")
    wrong = copy.deepcopy(v1.map)
    wrong["sources"]["F"] = synth.f_source_entry("fake filter text block", v1.variants, other_key)
    convert("F: block hashes made with another key -> not found", "wrong-key", with_file(none, v1.file, map=wrong), None)
    unkeyed = copy.deepcopy(v1.map)
    for entry, v in zip(unkeyed["sources"]["F"]["variants"], v1.variants):
        entry["fnv1a64"] = "0x%016x" % synth.fnv1a64(v["block"][:64])
        entry["sha256"] = synth.sha256(v["block"]).hex()
    convert("F: unkeyed block hashes -> not found", "unkeyed", with_file(none, v1.file, map=unkeyed), None)
    # A version-1 map (no F) converts as before, also on a file that has the block; so does version 4 without F.
    plain = cases.mini_case()
    old = with_file(plain, v1.file)
    mp_old = r.write("filter/v1map/map.json", plain.map)
    out = r.path("filter/v1map/out", "x")[:-2]
    code, so, se = r.run(mp_old, [dll1], out)
    r.check("version-1 map on a file with the block: converts as before", code == 0 and so.splitlines() == [SUCCESS] and
            "source F" not in se, so + se)
    with open(mp_old, "rb") as f:
        check_output(r, "version-1 map on a file with the block", old, out, f.read())
    v4_plain = with_file(plain, v1.file, map=mutate(plain.map, lambda m: m.update(version=4)))
    mp4 = r.write("filter/v4plain/map.json", v4_plain.map)
    out = r.path("filter/v4plain/out", "x")[:-2]
    code, so, se = r.run(mp4, [dll1], out)
    r.check("version-4 map without source F: converts", code == 0 and so.splitlines() == [SUCCESS], so + se)
    with open(mp4, "rb") as f:
        check_output(r, "version-4 map without source F", v4_plain, out, f.read())
    # Converting a file without the block over output that has the group removes the group (model file and raw).
    dll_none = os.path.join(r.work, "filter", "none", "fake.dll")
    code, so, se = r.run(mp, [dll_none], out1)
    r.check("re-run without the block over output with the group: exit 0", code == 0, so + se)
    with open(mp, "rb") as f:
        check_output(r, "re-run without the block", none, out1, f.read())
    code, so, se = r.run(mp, [dll1], out1)
    r.check("re-run with the block again: exit 0", code == 0, so + se)
    with open(mp, "rb") as f:
        check_output(r, "re-run with the block again", v1, out1, f.read())
    r.check("without --raw: only the model file",
            r.run(mp, [dll1], r.path("filter/noraw/out", "x")[:-2], raw=False)[0] == 0 and
            sorted(os.listdir(os.path.join(r.work, "filter", "noraw", "out"))) == ["model.wgcm"])

    # --- errors: everything below must leave out1 exactly as it is
    before = snapshot(out1)

    def expect(label, code_name, map_obj, dlls=None, msg=None):
        p = r.write("filter/err/%s.json" % re.sub(r"\W+", "_", label), map_obj)
        code, so, se = r.run(p, dlls or [dll1], out1)
        want = msg if msg is not None else MSG.get(code_name, "")
        r.check("F: " + label + " -> " + code_name, code == EXIT[code_name] and want in so,
                "code %d, out %r, log %r" % (code, so, se[-400:]))
        r.check("F: " + label + ": earlier output untouched", snapshot(out1) == before)
        r.check("F: " + label + ": no weights or text shown",
                not any(("%08x" % x) in (so + se).lower() for v in v1.variants for x in v["elements"]))
        return p

    M = v1.map
    FV = lambda m, i=0: m["sources"]["F"]["variants"][i]  # noqa: E731
    FS = lambda m: m["stages"][-1]  # noqa: E731
    flip = lambda h: ("0" if h[0] != "0" else "1") + h[1:]  # noqa: E731

    def also_converts(label, p, dll, sub):
        code, so, se = r.run(p, [dll], r.path("filter/err-ok/%s/out" % sub, "x")[:-2])
        r.check("F: " + label, code == 0, so + se)

    # 08-T6: a map that does not fit the block it finds (the block hashes still match).
    p = expect("wrong decoded_sha256", "map_damaged", mutate(M, lambda m: FV(m).update(decoded_sha256=flip(FV(m)["decoded_sha256"]))))
    also_converts("wrong decoded_sha256 of variant 1 does not matter for a file with variant 2", p, dll2, "dsha")
    p = expect("wrong filter digest", "self_check", mutate(M, lambda m: FS(m)["digest"].__setitem__(0, flip(FS(m)["digest"][0]))))
    also_converts("wrong filter digest of variant 1 does not matter for a file with variant 2", p, dll2, "digest")
    expect("wrong filter digest, variant 2", "self_check",
           mutate(M, lambda m: FS(m)["digest"].__setitem__(1, flip(FS(m)["digest"][1]))), dlls=[dll2])
    expect("field one byte late (not hexadecimal)", "map_damaged", mutate(M, lambda m: FV(m)["fields"].__setitem__(2, FV(m)["fields"][2] + 1)))
    expect("field one byte early (not hexadecimal)", "map_damaged", mutate(M, lambda m: FV(m)["fields"].__setitem__(0, FV(m)["fields"][0] - 1)))
    expect("field in plain text", "map_damaged", mutate(M, lambda m: FV(m)["fields"].__setitem__(4, 3)))
    expect("field of variant 2 not hexadecimal", "map_damaged",
           mutate(M, lambda m: FV(m, 1)["fields"].__setitem__(1, FV(m, 1)["fields"][1] + 3)), dlls=[dll2])
    expect("decoded_length one too small", "map_damaged", mutate(M, lambda m: FV(m).update(decoded_length=FV(m)["decoded_length"] - 1)))
    expect("decoded_length one too large", "map_damaged", mutate(M, lambda m: FV(m).update(decoded_length=FV(m)["decoded_length"] + 1)))
    # Blocks whose hashes match the map but which are not valid blocks (our own bytes).
    for label, block in (
        ("matching block with distance 0", bytes([0x40]) + b"abcd" + b"\x00\x00" + bytes([0x50]) + b"final"),
        ("matching block with a distance beyond the output", bytes([0x40]) + b"abcd" + b"\x09\x00" + bytes([0x50]) + b"final"),
        ("matching block that ends inside a literal run", bytes([0xF0, 200]) + b"too short"),
        ("matching block that ends inside the distance", bytes([0x40]) + b"abcd" + b"\x04"),
        ("matching block of random bytes", bytes(random_bytes_for_test(4000))),
        ("matching literal-only block that is too long", synth.lz4_block_encode(v1.variants[0]["text"] + b"x", literal_only=True)),
        ("matching block that decodes to other text", synth.lz4_block_encode(v1.variants[0]["text"][::-1])),
    ):
        bad = {"text": v1.variants[0]["text"], "block": block, "fields": v1.variants[0]["fields"]}
        m2 = mutate(M, lambda m: m["sources"]["F"]["variants"].__setitem__(0, synth.f_variant_entry(bad, v1.key)))
        f2 = bytearray(v1.file)
        f2[F: F + n] = bytes(n)
        f2[F: F + len(block)] = block
        expect(label, "map_damaged", m2, dlls=[r.write("filter/err/%s.dll" % re.sub(r"\W+", "_", label), bytes(f2))])
    # A field that is hexadecimal but no number: an installer error like every other non-finite value.
    nan = cases.f_mini_case(present=None, elements0=[0x3F000000, 0x7FC00000, 0x3E800000, 0x80000000, 0x00000001])
    f2 = bytearray(nan.file)
    f2[F: F + len(nan.variants[0]["block"])] = nan.variants[0]["block"]
    expect("u32_as_f32 of a NaN field", "bad_value", nan.map, dlls=[r.write("filter/err/nan.dll", bytes(f2))])

    # Section 5 and 8: invalid version-4 maps, refused before any file is read.
    for label, fn in (
        ("version 5", lambda m: m.update(version=5)),
        ("hex32 in a version-1 map", lambda m: m.update(version=1)),
        ("unknown element", lambda m: m["sources"]["F"].update(element="hex64")),
        ("unknown encoding", lambda m: m["sources"]["F"].update(encoding="lz5-block")),
        ("encoding not text", lambda m: m["sources"]["F"].update(encoding=4)),
        ("missing encoding", lambda m: m["sources"]["F"].pop("encoding")),
        ("F not optional", lambda m: m["sources"]["F"].update(optional=False)),
        ("missing optional", lambda m: m["sources"]["F"].pop("optional")),
        ("F not keyed", lambda m: m["sources"]["F"].update(keyed=False)),
        ("missing keyed", lambda m: m["sources"]["F"].pop("keyed")),
        ("F as a pair source", lambda m: m["sources"]["F"].update(locate="pair", gaps=[4])),
        ("unknown locate for F", lambda m: m["sources"]["F"].update(locate="scan")),
        ("missing locate for F", lambda m: m["sources"]["F"].pop("locate")),
        ("missing what for F", lambda m: m["sources"]["F"].pop("what")),
        ("missing variants", lambda m: m["sources"]["F"].pop("variants")),
        ("variants empty", lambda m: m["sources"]["F"].update(variants=[])),
        ("variants not a list", lambda m: m["sources"]["F"].update(variants={"a": 1})),
        ("variant not an object", lambda m: m["sources"]["F"]["variants"].__setitem__(1, 7)),
        ("17 variants", lambda m: m["sources"]["F"].update(variants=[FV(m)] * 17)),
        ("variant length 0", lambda m: FV(m).update(length=0)),
        ("variant length as text", lambda m: FV(m).update(length="8609")),
        ("decoded_length 0", lambda m: FV(m).update(decoded_length=0)),
        ("decoded_length beyond the limit", lambda m: FV(m).update(decoded_length=(1 << 26) + 1)),
        ("bad variant fnv1a64", lambda m: FV(m).update(fnv1a64="0xZZ")),
        ("bad variant sha256", lambda m: FV(m).update(sha256="abc")),
        ("bad decoded_sha256", lambda m: FV(m, 1).update(decoded_sha256=FV(m, 1)["decoded_sha256"][:-1])),
        ("fields empty", lambda m: FV(m).update(fields=[])),
        ("fields not a list", lambda m: FV(m).update(fields=9536)),
        ("negative field", lambda m: FV(m)["fields"].__setitem__(0, -1)),
        ("field as text", lambda m: FV(m)["fields"].__setitem__(0, "12")),
        ("field past the decoded text", lambda m: FV(m)["fields"].__setitem__(3, FV(m)["decoded_length"] - 7)),
        ("variants with different numbers of fields", lambda m: FV(m, 1)["fields"].pop()),
        ("hex32 source not named F", lambda m: m["sources"].update(G=m["sources"].pop("F"))),
        ("encoding on W", lambda m: m["sources"]["W"].update(encoding="lz4-block")),
        ("unknown encoding on D", lambda m: m["sources"]["D"].update(encoding="none")),
        ("variants on D", lambda m: m["sources"]["D"].update(variants=m["sources"]["F"]["variants"])),
        ("optional K", lambda m: m["sources"]["K"].update(optional=True)),
        ("optional not true or false", lambda m: m["sources"]["D"].update(optional="no")),
        ("filter digest as text", lambda m: FS(m).update(digest=FS(m)["digest"][0])),
        ("filter digest with one entry", lambda m: FS(m)["digest"].pop()),
        ("filter digest with three entries", lambda m: FS(m)["digest"].append(FS(m)["digest"][0])),
        ("bad hex in the filter digest", lambda m: FS(m)["digest"].__setitem__(1, "xyz")),
        ("filter digest entry not text", lambda m: FS(m)["digest"].__setitem__(1, 5)),
        ("missing filter digest", lambda m: FS(m).pop("digest")),
        ("digest list for a group without variants", lambda m: m["stages"][0].update(digest=[m["stages"][0]["digest"]] * 2)),
        ("group with tensors from F and from W", lambda m: FS(m)["tensors"].append(cases.tensor("b", [8], "float32", "W", {"name": "f16"}, 96, [cases.lin(1)]))),
        ("tensor from F in another group", lambda m: m["stages"][0]["tensors"].append(copy.deepcopy(FS(m)["tensors"][0]))),
        ("f16 on F", lambda m: FS(m)["tensors"][0].update(op={"name": "f16"})),
        ("u32_as_f32 from F as uint8", lambda m: FS(m)["tensors"][0].update(dtype="uint8")),
        ("key from F", lambda m: m["key"].update(**{"from": "F"})),
        ("missing W in a version-4 map", lambda m: m["sources"].pop("W")),
        ("base64 in a version-4 map", lambda m: m["sources"]["D"].update(element="base64", segments=[[0, 4]])),
        ("capped_softmax in a version-4 map", lambda m: m["stages"][1]["tensors"][1].update(
            shape=[1, 8, 8], axes=[{"stride": 0}, {"stride": 8}, {"stride": 1}], op=dict(cases.CS_OP))),
        ("int32 in a version-4 map", lambda m: FS(m)["tensors"][0].update(dtype="int32", op={"name": "i32"})),
    ):
        expect(label, "map_damaged", mutate(M, fn))
    expect("F index 5", "map_misfit", mutate(M, lambda m: FS(m)["tensors"][0].update(base=1)))
    expect("six elements from F", "map_misfit", mutate(M, lambda m: FS(m)["tensors"][0].update(shape=[6])))
    expect("F index below 0", "map_misfit", mutate(M, lambda m: FS(m)["tensors"][0]["axes"][0].update(stride=-1)))
    # Informational fields that a variant source may carry are ignored (top-level length and hashes are not used).
    extra = r.write("filter/extra/map.json", mutate(M, lambda m: (m["sources"]["F"].update(length=1, note="x"), FV(m).update(note="y"))))
    code, so, se = r.run(extra, [dll1], r.path("filter/extra/out", "x")[:-2])
    r.check("F: unknown fields in the source and in a variant are ignored", code == 0, so + se)
    r.check("F: no staging folders left", not any(nm.startswith(".wronggpu-convert-staging") for nm in os.listdir(out1)))
    # Several files (spec 08 section 8): the first that converts completely with the group is used; if none has it,
    # the first that converts completely. The note is printed only when the file used has no group.
    with open(mp, "rb") as f:
        map_bytes = f.read()
    nomodel = r.write("filter/multi/nomodel.dll", cases.flip_bit(v1.file, v1.placements["W"] + 10, 0))
    # A second file without the block whose other bytes differ: which of the two was used shows in the log only.
    dll_none2 = r.write("filter/multi/none2.dll", none.file + b"other trailing bytes of our own")

    def several(label, sub, dlls, case, used, note):
        out = r.path("filter/multi/" + sub, "x")[:-2]
        code, so, se = r.run(mp, dlls, out)
        tail = [SUCCESS, FILTER_NOTE] if note else [SUCCESS]
        r.check("several files, " + label + ": exit 0 and the player lines", code == 0 and so.splitlines()[-len(tail):] == tail and
                (note or FILTER_NOTE not in so), so + se)
        r.check("several files, " + label + ": the log names the file that was used",
                "using " + used in se and "from " + used in se, se)
        check_output(r, "several files, " + label, case, out, map_bytes)
        return so, se

    so, se = several("[rejected, without the block, with it]", "a", [nomodel, dll_none, dll1], v1, dll1, False)
    r.check("several files: the rejection is still reported and the passed-over file is logged",
            MSG["no_preset_m"] in so and dll_none + " converts without an optional group" in se, so + se)
    several("[without the block, with it]", "b", [dll_none, dll1], v1, dll1, False)
    so, se = several("[with the block, without it]", "c", [dll1, dll_none], v1, dll1, False)
    r.check("several files: nothing after a file with the group is read", "file: " + dll_none not in se, se)
    several("[variant 2, variant 1]", "d", [dll2, dll1], v2, dll2, False)
    so, se = several("[two without the block]", "e", [dll_none, dll_none2], none, dll_none, True)
    r.check("several files, two without the block: both were tried, the first is the one written",
            "file: " + dll_none2 in se and "using " + dll_none2 not in se and se.count("written to") == 1, se)
    several("[without the block, rejected]", "f", [dll_none, nomodel], none, dll_none, True)
    # An installer error in a later file keeps nothing, also when an earlier file converted without the group.
    bad_f = r.write("filter/multi/bad-map.json", mutate(v1.map, lambda m: m["stages"][-1]["digest"].__setitem__(0, "ab" * 32)))
    out_bad = r.path("filter/multi/g", "x")[:-2]
    code, so, se = r.run(bad_f, [dll_none, dll1], out_bad)
    r.check("several files, installer error after a file without the block: refused, nothing written",
            code == EXIT["self_check"] and not os.path.exists(os.path.join(out_bad, "model.wgcm")) and
            not any(nm.startswith(".wronggpu-convert-staging") for nm in os.listdir(out_bad)), so + se)


def random_bytes_for_test(n):
    import random
    return random.Random(4242).randbytes(n)


# ---------------------------------------------------------------- preset K (spec 48, map versions 2 and 3)

def test_k_t5(r, t5_path):
    """48-T5 against the independent reference (the converter's own result is checked in test_k_feature)."""
    import synth
    r.check("48-T5: reference matches the spec vectors", synth.check_capped_softmax_vectors(t5_path) == 0)


def test_k_feature(r, t5_path):
    t0 = time.time()
    case = cases.k_feature_case(t5_path)
    print("k feature case built in %.1f s" % (time.time() - t0))
    dll = r.write("kfeature/fake.dll", case.file)
    mp = r.write("kfeature/map.json", case.map)
    out = r.path("kfeature/out", "x")[:-2]
    code, so, se = r.run(mp, [dll], out)
    r.check("k feature: exit 0", code == 0, so + se)
    with open(mp, "rb") as f:
        check_output(r, "k feature", case, out, f.read())
    # 48-T5 directly: the converter's capped_softmax of the spec's x equals the spec's y, bit for bit.
    want = b"".join(struct.pack("<I", int(y, 16)) for row in case.t5["y"] for y in row)
    try:
        with open(os.path.join(out, "raw", "cs", "t5.bin"), "rb") as f:
            got = f.read()
    except OSError:
        got = b""
    r.check("48-T5: converter output equals the spec vectors", got == want,
            "%d of %d differ" % (sum(got[i:i + 4] != want[i:i + 4] for i in range(0, len(want), 4)), len(want) // 4))
    try:
        with open(os.path.join(out, "raw", "net", "window_offsets.bin"), "rb") as f:
            wo = f.read()
    except OSError:
        wo = b""
    r.check("k feature: net/window_offsets.bin is little-endian int32, row-major", wo == case.sources["O"])
    offs = logged_offsets(se)
    r.check("k feature: search (T even after a decoy, O odd after a decoy)", offs == case.placements,
            "%s vs %s" % (offs, case.placements))
    r.check("k feature: version logged", "310.98.2.0" in se, se)
    for threads in ("1", "5"):
        out2 = r.path("kfeature/out-t" + threads, "x")[:-2]
        code, so, se = r.run(mp, [dll], out2, extra=("--threads", threads))
        r.check("k feature: threads " + threads, code == 0 and logged_offsets(se) == case.placements, so + se)
        try:
            with open(os.path.join(out, "model.wgcm"), "rb") as a, open(os.path.join(out2, "model.wgcm"), "rb") as b:
                same = a.read() == b.read()
        except OSError:
            same = False
        r.check("k feature: threads %s identical output" % threads, same)


def test_k_shipped_like(r, kmap):
    """The shipped preset-K map's exact layout, with our own random T and O."""
    from synth import k_shipped_like_case
    for seed, t_off in ((1, 4096), (2, 4096 + 6), (3, 4096 + 5)):
        t0 = time.time()
        case = k_shipped_like_case(kmap, seed=seed, t_offset=t_off, o_offset=t_off + 4_400_000 + seed)
        label = "k shipped-like(seed %d, T at %d)" % (seed, t_off)
        dll = r.write("kshipped/%d/fake.dll" % seed, case.file)
        mp = r.write("kshipped/%d/map.json" % seed, case.map)
        out = r.path("kshipped/%d/out" % seed, "x")[:-2]
        code, so, se = r.run(mp, [dll], out)
        r.check(label + ": exit 0", code == 0, so + se)
        with open(mp, "rb") as f:
            check_output(r, label, case, out, f.read())
        r.check(label + ": offsets", logged_offsets(se) == case.placements, "%s vs %s" % (logged_offsets(se), case.placements))
        print("%s: %.1f s" % (label, time.time() - t0))


def test_k_shipped_map_validates(r, kmap):
    """The real preset-K map validates; with fake files it reports "not a model file" / "no preset-K"."""
    import random
    rng = random.Random(19)
    with open(kmap, "rb") as f:
        m = json.load(f)
    r.check("real K map: version 3", m["version"] == 3)
    lt = m["sources"]["T"]["length"]
    out = r.path("kshipped-real/out", "x")[:-2]
    small = r.write("kshipped-real/small.dll", rng.randbytes(lt - 1))
    code, so, se = r.run(kmap, [small], out)
    r.check("real K map: smaller than T -> not a model file", code == EXIT["not_model"] and MSG["not_model"] in so, so + se)
    big = r.write("kshipped-real/nomodel.dll", rng.randbytes(lt + 100_000))
    code, so, se = r.run(kmap, [big], out)
    r.check("real K map: valid, T absent -> no preset-K", code == EXIT["no_preset_k"] and MSG["no_preset_k"] in so, so + se)
    r.check("real K map: nothing written", not os.path.exists(os.path.join(out, "model.wgcm")))
    # Version 2 of the same map (no O, no net group) is still read.
    m2 = copy.deepcopy(m)
    m2["version"] = 2
    del m2["sources"]["O"]
    m2["stages"] = [s for s in m2["stages"] if s["stage"] != "net"]
    code, so, se = r.run(r.write("kshipped-real/v2.json", m2), [big], out)
    r.check("real K map as version 2: valid, T absent -> no preset-K", code == EXIT["no_preset_k"], so + se)


def test_k_errors(r):
    import synth
    base = cases.k_mini_case()
    mp = r.write("kerr/map.json", base.map)
    dll = r.write("kerr/fake.dll", base.file)
    out = r.path("kerr/out", "x")[:-2]
    code, so, se = r.run(mp, [dll], out)
    r.check("k mini: exit 0", code == 0, so + se)
    with open(mp, "rb") as f:
        check_output(r, "k mini", base, out, f.read())
    before = snapshot(out)

    def expect(label, code_name, map_obj=None, dlls=None, msg=None):
        p = r.write("kerr/%s.json" % re.sub(r"\W+", "_", label), map_obj) if map_obj is not None else mp
        code, so, se = r.run(p, dlls or [dll], out)
        want = msg if msg is not None else MSG.get(code_name, "")
        r.check("K: " + label + " -> " + code_name, code == EXIT[code_name] and want in so,
                "code %d, out %r, log %r" % (code, so, se[-400:]))
        r.check("K: " + label + ": earlier output untouched", snapshot(out) == before)
        return so, se

    M = base.map
    T = lambda m: m["sources"]["T"]  # noqa: E731
    # --- 48-T4 and 08-T4 analogues: corrupted maps, refused before any output.
    expect("segment past the end of T", "map_damaged", mutate(M, lambda m: T(m)["segments"][-1].__setitem__(
        1, ((T(m)["length"] - T(m)["segments"][-1][0]) // 4 + 1) * 4)))
    expect("segment offset past the end of T", "map_damaged", mutate(M, lambda m: T(m)["segments"][0].__setitem__(0, T(m)["length"] + 8)))
    expect("segment length not a multiple of 4", "map_damaged", mutate(M, lambda m: T(m)["segments"][1].__setitem__(1, T(m)["segments"][1][1] - 2)))
    expect("segment of length 0", "map_damaged", mutate(M, lambda m: T(m)["segments"][1].__setitem__(1, 0)))
    expect("negative segment offset", "map_damaged", mutate(M, lambda m: T(m)["segments"][1].__setitem__(0, -4)))
    expect("segment not a pair", "map_damaged", mutate(M, lambda m: T(m)["segments"][1].append(4)))
    expect("segments empty", "map_damaged", mutate(M, lambda m: T(m).update(segments=[])))
    expect("segments missing", "map_damaged", mutate(M, lambda m: T(m).pop("segments")))
    expect("stride leading outside W", "map_misfit", mutate(M, lambda m: m["stages"][0]["tensors"][0]["axes"][0].update(stride=100)))
    expect("negative stride below 0", "map_misfit", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(base=100)))
    expect("O index 16", "map_misfit", mutate(M, lambda m: m["stages"][2]["tensors"][0].update(base=1)))
    expect("unknown op", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1]["op"].update(name="softmax")))
    expect("wrong net digest", "self_check", mutate(M, lambda m: m["stages"][2].update(digest="00" * 32)))
    expect("version 4", "map_damaged", mutate(M, lambda m: m.update(version=4)))
    expect("version 1 with T", "map_damaged", mutate(M, lambda m: m.update(version=1)))
    expect("version 2 with O", "map_damaged", mutate(M, lambda m: m.update(version=2)))
    expect("version 2 with an int32 tensor", "map_damaged",
           mutate(M, lambda m: (m.update(version=2), m["sources"].pop("O"), m["stages"][2]["tensors"][0].update(src="W"))))
    expect("missing O", "map_damaged", mutate(M, lambda m: m["sources"].pop("O")))
    expect("missing T", "map_damaged", mutate(M, lambda m: m["sources"].pop("T")))
    expect("extra source", "map_damaged", mutate(M, lambda m: m["sources"].update(D=copy.deepcopy(m["sources"]["O"]))))
    expect("O not keyed", "map_damaged", mutate(M, lambda m: m["sources"]["O"].pop("keyed")))
    expect("O as f16le", "map_damaged", mutate(M, lambda m: m["sources"]["O"].update(element="f16le")))
    expect("T keyed", "map_damaged", mutate(M, lambda m: T(m).update(keyed=True)))
    expect("key from O", "map_damaged", mutate(M, lambda m: m["key"].update(**{"from": "O"})))
    expect("tensor indexes T", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][0].update(src="T")))
    expect("i32 on W", "map_damaged", mutate(M, lambda m: m["stages"][2]["tensors"][0].update(src="W")))
    expect("f16 on O", "map_damaged", mutate(M, lambda m: m["stages"][2]["tensors"][0].update(op={"name": "f16"}, dtype="float32")))
    expect("i32 as float32", "map_damaged", mutate(M, lambda m: m["stages"][2]["tensors"][0].update(dtype="float32")))
    expect("capped_softmax on [2][8][4]", "map_damaged",
           mutate(M, lambda m: m["stages"][0]["tensors"][1].update(shape=[2, 8, 4]) or m["stages"][0]["tensors"][1]["axes"][2].update(stride=-1)))
    expect("capped_softmax on [64]", "map_damaged",
           mutate(M, lambda m: m["stages"][0]["tensors"][1].update(shape=[64], axes=[{"stride": 1}], base=0)))
    for field in ("cap", "div", "sub"):
        expect("capped_softmax without " + field, "map_damaged", mutate(M, lambda m, f=field: m["stages"][0]["tensors"][1]["op"].pop(f)))
    expect("capped_softmax cap NaN", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1]["op"].update(cap="0x7fc00000")))
    expect("capped_softmax as uint8", "map_damaged", mutate(M, lambda m: m["stages"][0]["tensors"][1].update(dtype="uint8")))
    for path in (("sources", "T", "fnv1a64"), ("sources", "T", "length"), ("sources", "O", "sha256"), ("sources", "O", "locate")):
        def drop(m, path=path):
            obj = m
            for k in path[:-1]:
                obj = obj[k]
            del obj[path[-1]]
        expect("missing " + ".".join(path), "map_damaged", mutate(M, drop))

    # --- 48-T3: damaged copies of a supported file; 48-T2 analogue: no T.
    t0, o0 = base.placements["T"], base.placements["O"]
    segs = M["sources"]["T"]["segments"]
    inside = t0 + segs[1][0] + 17
    outside = t0 + segs[1][0] - 2  # junk between segments 0 and 1
    for label, data, code_name in (
        ("bit flipped in T inside a segment", cases.flip_bit(base.file, inside, 1), "no_preset_k"),
        ("bit flipped in T outside the segments", cases.flip_bit(base.file, outside, 0), "no_preset_k"),
        ("bit flipped in T's first 64 bytes", cases.flip_bit(base.file, t0 + 2, 3), "no_preset_k"),
        ("truncated inside T", base.file[: t0 + len(base.sources["T"]) - 3], "no_preset_k"),
        ("bit flipped in O", cases.flip_bit(base.file, o0 + 9, 6), "unsupported"),  # spec 48 draft 4 section 8
        ("bit flipped in O's last byte", cases.flip_bit(base.file, o0 + 63, 0), "unsupported"),
        ("smaller than T", base.file[: len(base.sources["T"]) - 1], "not_model"),
    ):
        expect(label, code_name, dlls=[r.write("kerr/%s.dll" % re.sub(r"\W+", "_", label), data)])
    old = cases.k_mini_case(file_version=(310, 0, 9, 0))
    nofile = cases.flip_bit(old.file, old.placements["T"] + 100, 2)
    so, se = expect("no T, with a version resource", "no_preset_k", dlls=[r.write("kerr/old.dll", nofile)])
    r.check("K: version shown to the player", "This file is version 310.0.9.0." in so, so)

    # --- Base64 problems in a block whose hashes match (an installer error, like an index outside W).
    def with_text(label, text, code_name):
        m2 = copy.deepcopy(M)
        m2["sources"]["T"] = synth.t_source_entry("fake text block", text, m2["sources"]["T"]["segments"])
        key = synth.sha256(m2["key"]["label"].encode() + text)
        m2["sources"]["O"] = synth.source_entry("O", "fake offset table", "i32le", base.sources["O"], keyed=True, key=key)
        f = bytearray(base.file)
        f[t0: t0 + len(text)] = text
        expect(label, code_name, m2, dlls=[r.write("kerr/%s.dll" % re.sub(r"\W+", "_", label), bytes(f))])

    tx = bytearray(base.sources["T"])
    tx[segs[1][0] + 40] = ord("!")
    with_text("bad Base64 character", bytes(tx), "map_misfit")
    tx = bytearray(base.sources["T"])
    tx[segs[0][0] + 8] = ord("=")
    with_text("'=' inside a segment", bytes(tx), "map_misfit")
    # Segment 0 holds 1001 bytes (1336 characters ending in one '='). 1000 bytes give the same length (ending in
    # "=="), and an odd total.
    import base64 as b64
    tx = bytearray(base.sources["T"])
    w0 = base64_decode_segment(base.sources["T"], segs[0])
    enc = b64.b64encode(w0[:1000])
    assert len(w0) == 1001 and len(enc) == segs[0][1]
    tx[segs[0][0]: segs[0][0] + len(enc)] = enc
    with_text("odd decoded total", bytes(tx), "map_misfit")
    # An index beyond the decoded W (2000 elements) but within the bound the map allows before decoding (2001: the
    # padding is only known after decoding).
    m3 = mutate(M, lambda m: m["stages"][1]["tensors"][0].update(base=2000, shape=[1], axes=[{"stride": 0}]))
    expect("index between the decoded W and its bound", "map_misfit", m3)
    # Non-finite results: capped_softmax of an infinite input; a zero sum.
    tx = bytearray(base.sources["T"])
    w0 = bytearray(base64_decode_segment(base.sources["T"], segs[0]))
    w0[20:22] = struct.pack("<H", 0x7C00)
    enc = b64.b64encode(bytes(w0))
    tx[segs[0][0]: segs[0][0] + len(enc)] = enc
    m4 = mutate(M, lambda m: m["stages"][0]["tensors"][1].update(base=10, axes=[{"stride": 64}, {"stride": 8}, {"stride": 1}]))
    m4["stages"] = m4["stages"][:1]
    m4["stages"][0]["tensors"] = m4["stages"][0]["tensors"][1:]  # only the capped_softmax tensor reads the inf
    text4 = bytes(tx)
    m4["sources"]["T"] = synth.t_source_entry("fake text block", text4, segs)
    key4 = synth.sha256(m4["key"]["label"].encode() + text4)
    m4["sources"]["O"] = synth.source_entry("O", "fake offset table", "i32le", base.sources["O"], keyed=True, key=key4)
    f4 = bytearray(base.file)
    f4[t0: t0 + len(text4)] = text4
    expect("capped_softmax of +inf", "bad_value", m4, dlls=[r.write("kerr/inf.dll", bytes(f4))])
    expect("capped_softmax with a zero sum", "bad_value",
           mutate(M, lambda m: m["stages"][0]["tensors"][1]["op"].update(sub="0x43c80000")))  # e^(g - 400) = 0
    r.check("K: no staging folders left", not any(n.startswith(".wronggpu-convert-staging") for n in os.listdir(out)))

    # --- a version-2 map (no O, no net) still converts
    v2 = cases.k_mini_case(version=2)
    mp2 = r.write("kerr/v2/map.json", v2.map)
    out2 = r.path("kerr/v2/out", "x")[:-2]
    code, so, se = r.run(mp2, [r.write("kerr/v2/fake.dll", v2.file)], out2)
    r.check("K: version-2 map converts", code == 0, so + se)
    with open(mp2, "rb") as f:
        check_output(r, "k mini v2", v2, out2, f.read())


def base64_decode_segment(text, seg):
    import base64
    return base64.b64decode(text[seg[0]: seg[0] + seg[1]], validate=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", default=os.path.join(HERE, "..", "build", "test-work"))
    ap.add_argument("--map", default=DEFAULT_MAP, help="the shipped map (only its layout is used)")
    ap.add_argument("--kmap", default=DEFAULT_KMAP, help="the shipped preset-K map (only its layout is used)")
    ap.add_argument("--lmap", default=DEFAULT_LMAP, help="the shipped preset-L map (only its layout is used)")
    ap.add_argument("--t5", default=DEFAULT_T5, help="spec 48-T5 capped_softmax vectors")
    ap.add_argument("--t7", default=DEFAULT_T7, help="spec 08-T7 block decompression vectors")
    ap.add_argument("--quick", action="store_true", help="skip the large cases")
    args = ap.parse_args()
    if os.path.exists(args.work):
        shutil.rmtree(args.work)
    os.makedirs(args.work)
    r = Runner(os.path.abspath(args.exe), os.path.abspath(args.work))
    t0 = time.time()
    test_errors(r)
    test_preset_l(r)
    test_filter(r, args.t7)
    test_k_errors(r)
    if os.path.exists(args.map):
        test_shipped_map_validates(r, args.map)
        if not args.quick:
            test_shipped_like(r, args.map)
            test_l_shaped(r, args.map)
    else:
        print("note: shipped map not found at %s; skipping the layout tests" % args.map)
    if os.path.exists(args.kmap):
        test_k_shipped_map_validates(r, args.kmap)
        if not args.quick:
            test_k_shipped_like(r, args.kmap)
    else:
        print("note: preset-K map not found at %s; skipping its layout tests" % args.kmap)
    if os.path.exists(args.lmap):
        test_l_real_map(r, args.lmap)
    else:
        print("note: preset-L map not found at %s; skipping its layout tests" % args.lmap)
    if os.path.exists(args.t5):
        test_k_t5(r, args.t5)
        if not args.quick:
            test_k_feature(r, args.t5)
    else:
        print("note: 48-T5 vectors not found at %s; skipping the K feature case" % args.t5)
    if not args.quick:
        test_feature(r)
    print("%d checks passed, %d failed (%.1f s)" % (r.passed, len(r.failures), time.time() - t0))
    if r.failures:
        for f in r.failures:
            print("  FAILED:", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
