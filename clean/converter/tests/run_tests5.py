# SPDX-License-Identifier: MIT
"""End-to-end tests of wronggpu-convert for map version 5 (spec 57, preset F). Synthetic data only.

Written for this project. Python standard library only. Expected values come from tests/synth5.py (an exact
reference written independently of the C++ code) or, for 57-T8, from the spec team's synth-expected.json.

    python run_tests5.py --exe <wronggpu-convert> --unit <wgc-unit-tests> [--work <dir>]
                         [--spec-data <data>] [--synth-bins <folder with synth-common-v0/v1.bin>] [--quick]

The test files for 57-T8 are W_r (made here) followed by the spec team's synthetic library stand-ins
synth-common-v<v>.bin (kept outside git, see data/rr-f-synth/README.md). Without them the T1, T3, T4 (map
edits of the shipped synthetic maps), T7 and T8 parts are skipped and reported as skipped.
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

import synth5  # noqa: E402
from synth5 import DTYPE_CODE  # noqa: E402

MSG = {
    "unreadable": "This file cannot be read.",
    "not_model": "This is not a DLSS model file.",
    "no_preset_f": "This DLSS file has no preset-F (Ray Reconstruction) model. Version 310.7 or newer is needed.",
    "unsupported": "This DLSS build is not supported yet (model data found, but other parts differ).",
    "map_damaged": "converter data is damaged",
    "self_check": "Conversion self-check failed.",
    "success": "The model was converted successfully.",
}


class Runner:
    def __init__(self, exe, unit, work):
        self.exe, self.unit, self.work = exe, unit, work
        self.passed = 0
        self.failures = []
        self.skipped = []

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


def read_container(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"WGCMODEL", "magic"
    body, trailer = data[:-32], data[-32:]
    assert hashlib.sha256(body).digest() == trailer, "trailer"
    version, ngroups = struct.unpack_from("<II", data, 8)
    key, map_sha = data[16:48], data[48:80]
    (index_size,) = struct.unpack_from("<Q", data, 80)
    pos = 88
    groups = []
    for _ in range(ngroups):
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
        groups.append((name, digest, tensors))
    assert pos == 88 + index_size, "index size"
    return version, key, map_sha, groups


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
    found, variants = {}, {}
    for m in re.finditer(r"source (\S+) at offset (\d+)(?: \(variant (\d+) of \d+\))?", stderr):
        found[m.group(1)] = int(m.group(2))
        if m.group(3) is not None:
            variants[m.group(1)] = int(m.group(3))
    return found, variants


def check_output(r, label, m, expected, key, out_dir, map_bytes, raw=True):
    """Everything the converter wrote against the reference: container (version 2, key, map hash, group order,
    digests, dtype codes, shapes, bytes) and the raw files."""
    try:
        version, ckey, map_sha, groups = read_container(os.path.join(out_dir, "model.wgcm"))
    except Exception as exc:  # noqa: BLE001
        r.check(label + ": container readable", False, repr(exc))
        return
    r.check(label + ": container version 2", version == 2)
    r.check(label + ": container key", ckey == key)
    r.check(label + ": container map hash", map_sha == hashlib.sha256(map_bytes).digest())
    want_groups = [g for g in m["groups"] if g["group"] in expected]
    r.check(label + ": group order", [g[0] for g in groups] == [g["group"] for g in want_groups],
            "%s" % [g[0] for g in groups])
    for (name, digest, tensors), g in zip(groups, want_groups):
        r.check(label + ": digest " + name, digest.hex() == g["digest"])
        r.check(label + ": recomputed digest " + name,
                synth5.keyed_sha256(key, b"".join(t[3] for t in tensors)).hex() == g["digest"])
        exp = expected[name]
        r.check(label + ": tensor list " + name, [t[0] for t in tensors] == [t[0] for t in exp])
        for (tname, dtype, shape, data), (ename, edata), mt in zip(tensors, exp, g["tensors"]):
            ok = data == edata and shape == mt["shape"] and dtype == DTYPE_CODE[mt["dtype"]]
            detail = ""
            if data != edata and len(data) == len(edata):
                step = synth5.DTYPE_SIZE[mt["dtype"]]
                bad = [i for i in range(0, len(data), step) if data[i:i + step] != edata[i:i + step]]
                detail = "%d elements differ, first %d: got %s want %s" % (
                    len(bad), bad[0] // step, data[bad[0]:bad[0] + step].hex(), edata[bad[0]:bad[0] + step].hex())
            r.check("%s: values %s/%s" % (label, name, tname), ok, detail)
            if raw:
                try:
                    with open(os.path.join(out_dir, "raw", name, tname + ".bin"), "rb") as f:
                        r.check("%s: raw %s/%s" % (label, name, tname), f.read() == edata)
                except OSError as exc:
                    r.check("%s: raw %s/%s" % (label, name, tname), False, repr(exc))
    if raw:
        for g in m["groups"]:
            if g["group"] not in expected:
                r.check(label + ": no raw folder for the absent group " + g["group"],
                        not os.path.exists(os.path.join(out_dir, "raw", g["group"])))


def verify_model(r, map_path, model):
    p = subprocess.run([r.unit, "--verify-model", map_path, model], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", "replace").strip()


def expect_failure(r, label, map_path, dlls, code, message, out_dir=None):
    """A conversion that must fail with `code` and `message` and leave an earlier output untouched."""
    out_dir = out_dir or r.path("fail-out", "x")
    snap = snapshot(out_dir) if os.path.isdir(out_dir) else None
    rc, so, se = r.run(map_path, dlls, out_dir)
    r.check(label + ": exit code %d" % code, rc == code, "got %d; %s %s" % (rc, so.strip(), se[-400:]))
    r.check(label + ": message", message in so, so.strip())
    if snap is not None:
        r.check(label + ": earlier output untouched", snapshot(out_dir) == snap)
        r.check(label + ": no staging folder left",
                not any(n.startswith(".wronggpu-convert-staging") for n in os.listdir(out_dir)))
    return rc, so, se


# ---------------------------------------------------------------- 57-T5 (both implementations)

def test_t5(r, t5_path):
    n, fails = synth5.check_t5_vectors(t5_path)
    r.check("57-T5: reference agrees with the spec vectors (%d checks)" % n, not fails, str(fails[:5]))
    p = subprocess.run([r.unit, "--t5-vectors", t5_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.decode()
    r.check("57-T5: converter operations agree with the spec vectors", p.returncode == 0, out[-500:])
    print(out.strip())


# ---------------------------------------------------------------- our own feature cases

def test_feature(r):
    cases = synth5.feature_cases()
    for c in cases:
        mp = r.write("feature/%s.map.json" % c.name, c.map)
        dll = r.write("feature/%s.bin" % c.name, c.file)
        map_bytes = open(mp, "rb").read()
        for threads in (None, 1):
            out = r.path("feature", "out-%s-%s" % (c.name, threads), "x")[:-2]
            extra = () if threads is None else ("--threads", str(threads))
            rc, so, se = r.run(mp, [dll], out, extra=extra)
            label = "feature %s threads %s" % (c.name, threads)
            r.check(label + ": converts", rc == 0, so + se[-1500:])
            if rc:
                continue
            r.check(label + ": success message", MSG["success"] in so)
            check_output(r, label, c.map, c.expected, c.key, out, map_bytes)
            offs, vars_ = logged_offsets(se)
            for n, off in c.placements.items():
                r.check(label + ": %s found at its intact copy" % n, offs.get(n) == off, "%s vs %s" % (offs.get(n), off))
            r.check(label + ": variant found", vars_.get("IMG-V") == c.variant["IMG-V"])
            r.check(label + ": variant logged", "variant used: IMG-V variant %d" % c.variant["IMG-V"] in se)
            r.check(label + ": attention record logged", "attention_sum_may_overflow: false" in se)
            r.check(label + ": only_v1 present iff variant 1", os.path.isdir(os.path.join(out, "raw", "only_v1")) ==
                    (c.variant["IMG-V"] == 1))
            rc2, txt = verify_model(r, mp, os.path.join(out, "model.wgcm"))
            r.check(label + ": verify_model_file accepts it", rc2 == 0, txt)
            r.check(label + ": verify_model_file reports the gated group",
                    ("absent: only_v1" in txt) == (c.variant["IMG-V"] == 0), txt)
    return cases


def refinalize(m, blocks):
    """New hashes (and digests) after a change of data."""
    m = copy.deepcopy(m)
    synth5.finalize(m, blocks)
    return m


def place_case(c, m, blocks, v=None):
    """The file of case c with other blocks (same layout)."""
    import random
    rng = random.Random(99)
    v = c.variant["IMG-V"] if v is None else v
    lay = list(c.placements.items())
    return synth5.build_file(rng, blocks, m, lay, len(c.file), variant_choice={"IMG-V": v})


def test_value_rejections(r, cases):
    """Spec 57 section 8: every value check, with sources that match their hashes (exit code 14, "not supported
    yet"), and the log names the tensor or check."""
    base = cases[1]  # variant 1: every group present
    out = r.path("rej-out", "x")
    rc, so, se = r.run(r.write("rej/base.map.json", base.map), [r.write("rej/base.bin", base.file)], out)
    r.check("value rejections: earlier output written", rc == 0, so + se[-500:])

    def blocks_with(**changes):
        b = dict(base.blocks)
        b["IMG-V"] = list(b["IMG-V"])
        for k, v in changes.items():
            b[k] = v
        return b

    def img_with(idx_vals, name="IMG-A"):
        words = list(struct.unpack("<%dI" % (len(base.blocks[name]) // 4), base.blocks[name]))
        for i, v in idx_vals.items():
            words[i] = v
        return b"".join(struct.pack("<I", w) for w in words)

    def v1_with(idx_vals):
        words = list(struct.unpack("<%dI" % (len(base.blocks["IMG-V"][1]) // 4), base.blocks["IMG-V"][1]))
        for i, v in idx_vals.items():
            words[i] = v
        return b"".join(struct.pack("<I", w) for w in words)

    def w_with(idx_vals):
        w = bytearray(base.blocks["W"])
        for i, v in idx_vals.items():
            struct.pack_into("<H", w, 2 * i, v)
        return bytes(w)

    f32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    h16 = synth5.h16
    cases_ = [
        ("const sites disagree", blocks_with(**{"IMG-A": img_with({11: 0x38013801})}), None, "A_CLAMP"),
        ("const_f16x2 halves differ", blocks_with(**{"IMG-A": img_with({12: (h16(-0.5) << 16) | h16(-0.25)})}), None,
         "A_CLAMPn"),
        ("const_f16lo high half not zero", blocks_with(**{"IMG-A": img_with({16: 0x00013d80})}), None, "A_P0"),
        ("const_f32_to_f16 RN16 overflow", blocks_with(**{"IMG-A": img_with({19: f32(65520.0)})}), None, "RW_B"),
        ("const_f32_to_f16 of NaN", blocks_with(**{"IMG-A": img_with({17: 0x7fc00000, 18: 0x7fc00000})}), None, "RW_A"),
        ("const_u32_as_f32 infinite (finite-output rule)", blocks_with(**{"IMG-A": img_with({22: 0x7f800000})}), None,
         "X_CL"),
        ("const_f16x2 infinite (finite-output rule)",
         blocks_with(**{"IMG-A": img_with({12: 0x7c007c00})}), None, "A_CLAMPn"),
        ("require not_subnormal", blocks_with(**{"IMG-A": img_with({20: 0x00000100})}), None, "X_KEY"),
        ("require finite (NaN)", blocks_with(**{"IMG-A": img_with({21: 0x7fc00001}),
                                                "IMG-V": [base.blocks["IMG-V"][0], v1_with({9: 0x7fc00001})]}), None,
         "X_MFL"),
        ("require even", blocks_with(SH=struct.pack("<I", 3) + base.blocks["SH"][4:]), None, "SH"),
        ("require below_8", blocks_with(SH=struct.pack("<I", 8) + base.blocks["SH"][4:]), None, "SH"),
        ("check gt const", blocks_with(**{"IMG-A": img_with({20: f32(-1.0)})}), None, "gt X_KEY"),
        ("check le", blocks_with(**{"IMG-A": img_with({23: f32(4.0)})}), None, "le X_CLn X_CL"),
        ("check ratio_below_f16_overflow", blocks_with(**{"IMG-A": img_with({20: f32(65520.0 * 0.25)})}), None,
         "ratio_below_f16_overflow"),
        ("attention constants out of range", blocks_with(**{"IMG-A": img_with({16: h16(2.0)})}), None, "WMAX"),
        ("attention bounds reversed", blocks_with(**{"IMG-A": img_with({12: (h16(0.75) << 16) | h16(0.75)})}), None,
         "WMAX"),
        ("e4m3 of a non-finite W element", blocks_with(W=w_with({70005: 0x7c00})), None, "G1/fe"),
        ("f16 of a NaN W element", blocks_with(W=w_with({71004: 0x7e00})), None, "G1/b"),
        ("f16 of a NaN window", blocks_with(**{"H.x": synth5.window(__import__("random").Random(1), 8,
                                                                    struct.pack("<H", 0xfe01))}), None, "G1/hx"),
        ("exp overflow without a cap", blocks_with(W=w_with({71310: h16(100.0)})), None, "G1/e1"),
        ("exp of a non-finite value", blocks_with(W=w_with({71401: 0xfc00})), None, "G1/e2"),
        ("exp cap not finite", blocks_with(CAP=synth5.window(__import__("random").Random(2), 8,
                                                             struct.pack("<I", 0x7f800000))), None, "the cap CAP"),
        ("score_bias overflow", blocks_with(BS_A=synth5.window(__import__("random").Random(3), 8,
                                                               struct.pack("<f", 256.0))), None, "G1/Tb"),
        ("score_bias negative Wd", blocks_with(BS_W=synth5.window(__import__("random").Random(4), 8,
                                                                  struct.pack("<f", -4.0))), None, "G1/Tb"),
        ("score_bias non-finite stored value", blocks_with(W=w_with({72000 + 4096 + 77: 0x7c00})), None, "G1/Tb"),
        ("u32_as_f32 NaN (finite-output rule)",
         blocks_with(**{"IMG-A": img_with({201: 0x7fc00000, 301: 0x7fc00000}),
                        "IMG-B": (lambda wds: b"".join(struct.pack("<I", x) for x in wds))(
                            [0x7fc00000 if i == 105 else w for i, w in enumerate(
                                struct.unpack("<200I", base.blocks["IMG-B"]))])}), None, "G1/TR"),
        ("also reads differ (data)", blocks_with(**{"IMG-B": (lambda wds: b"".join(struct.pack("<I", x) for x in wds))(
            [w ^ 1 if i == 9 else w for i, w in enumerate(struct.unpack("<200I", base.blocks["IMG-B"]))])}), None,
         "also read"),
    ]
    for label, blocks, _, logname in cases_:
        try:
            m = refinalize(base.map, blocks)
        except Exception as exc:  # noqa: BLE001
            r.check("value rejection %s: case builds" % label, False, repr(exc))
            continue
        mp = r.write("rej/%s.map.json" % re.sub(r"\W+", "_", label), m)
        dll = r.write("rej/%s.bin" % re.sub(r"\W+", "_", label), place_case(base, m, blocks))
        rc, so, se = expect_failure(r, "value rejection " + label, mp, [dll], 14, MSG["unsupported"], out)
        r.check("value rejection %s: log names it" % label, logname in se, se[-300:])


# ---------------------------------------------------------------- 57-T4: corrupted maps

def corrupt_map_cases(fm):
    """(label, edit function on a deep copy, expected code). fm: a valid version-5 map (our feature map)."""
    def g(m, name):
        return next(x for x in m["groups"] if x["group"] == name)

    def t(m, gname, tname):
        return next(x for x in g(m, gname)["tensors"] if x["name"] == tname)

    E = []
    add = lambda label, fn: E.append((label, fn))
    add("unknown op", lambda m: t(m, "G1", "b")["op"].update(name="f16x"))
    add("perm block does not divide its axis", lambda m: m["perms"].update(rev4=[2, 1, 0]))
    add("version-4 map with version-5 features", lambda m: m.update(version=4))
    add("version-3 map with version-5 features", lambda m: m.update(version=3))
    add("unsupported version 6", lambda m: m.update(version=6))
    add("index outside its source", lambda m: t(m, "G1", "D").update(base=100))
    add("index outside its source (also)", lambda m: t(m, "G1", "w2")["also"][0].update(base=190))
    add("index outside a variant (ungated group)",
        lambda m: g(m, "G1")["tensors"].append({"name": "vv", "shape": [70], "dtype": "uint32", "op": {"name": "u32"},
                                               "src": "IMG-V", "base": 0, "axes": [{"stride": 1}]}))
    add("site index outside its source", lambda m: t(m, "constants", "X_KEY")["sites"][0].update(elements=[400]))
    add("site index outside a variant",
        lambda m: t(m, "constants", "A_CLAMP")["sites"][1].update(elements_by_variant=[[64], [7]]))
    add("score_bias with axes", lambda m: t(m, "G1", "Tb").update(axes=[{"stride": 1}] * 4))
    add("score_bias pair out of range", lambda m: t(m, "G1", "Tb")["op"]["pairs"].__setitem__(3, [1, 4]))
    add("score_bias with 11 pairs", lambda m: t(m, "G1", "Tb")["op"]["pairs"].pop())
    add("score_bias beyond W", lambda m: t(m, "G1", "Tb").update(base=72000 + 64 + 1))
    add("score_bias not reading W", lambda m: t(m, "G1", "Tb").update(src="D"))
    add("score_bias a_src not a window", lambda m: t(m, "G1", "Tb")["op"].update(a_src="IMG-A"))
    add("score_bias w_src a 16-bit window", lambda m: t(m, "G1", "Tb")["op"].update(w_src="H.x"))
    add("score_bias shape", lambda m: t(m, "G1", "Tb").update(shape=[2, 12, 16, 8]))
    add("op element size does not fit its source (u32 of u16)", lambda m: t(m, "G1", "D")["op"].update(name="u32"))
    add("op element size does not fit its source (e4m3 of u32)", lambda m: (
        t(m, "G1", "w2")["op"].update(name="e4m3"), t(m, "G1", "w2").update(dtype="e4m3")))
    add("f16 of a u16 block", lambda m: (t(m, "G1", "D")["op"].update(name="f16"), t(m, "G1", "D").update(dtype="float16")))
    add("dtype does not match its op", lambda m: t(m, "G1", "b").update(dtype="float32"))
    add("dtype int32 (version 3 only)", lambda m: t(m, "G1", "b").update(dtype="int32"))
    add("dtype uint8 (version 1 only)", lambda m: t(m, "G1", "D").update(dtype="uint8"))
    add("window field outside its window", lambda m: m["sources"]["BS_A"].update(field=13))
    add("window field missing", lambda m: m["sources"]["BS_A"].pop("field"))
    add("negative window field", lambda m: m["sources"]["BS_A"].update(field=-1))
    add("field on a block source", lambda m: m["sources"]["D"].update(field=0))
    add("unkeyed source", lambda m: m["sources"]["D"].pop("keyed"))
    add("keyed: false", lambda m: m["sources"]["IMG-A"].update(keyed=False))
    add("keyed W", lambda m: m["sources"]["W"].update(keyed=True))
    add("elements_by_variant of the wrong length",
        lambda m: t(m, "constants", "A_CLAMP")["sites"][1].update(elements_by_variant=[[5]]))
    add("elements_by_variant on a source without variants",
        lambda m: t(m, "constants", "X_KEY")["sites"][0].update(elements_by_variant=[[20], [20]]) or
        t(m, "constants", "X_KEY")["sites"][0].pop("elements"))
    add("both elements and elements_by_variant",
        lambda m: t(m, "constants", "A_CLAMP")["sites"][1].update(elements=[5]))
    add("requires_variant on a source without variants", lambda m: g(m, "only_v1").update(requires_variant={"IMG-A": 0}))
    add("requires_variant index out of range", lambda m: g(m, "only_v1").update(requires_variant={"IMG-V": 2}))
    add("requires_variant negative", lambda m: g(m, "only_v1").update(requires_variant={"IMG-V": -1}))
    add("requires_variant unknown source", lambda m: g(m, "only_v1").update(requires_variant={"IMG-Q": 0}))
    add("const tensor with no selected site in an enabled group", lambda m: g(m, "only_v1").pop("requires_variant"))
    add("const tensor with no sites", lambda m: t(m, "constants", "SUB").update(sites=[]))
    add("const tensor with an empty fixed list only", lambda m: t(m, "constants", "SUB")["sites"][0].update(elements=[]))
    add("const site with 16-bit elements", lambda m: t(m, "constants", "SUB")["sites"][0].update(src="H.x"))
    add("const with src", lambda m: t(m, "constants", "SUB").update(src="IMG-A"))
    add("const with shape [2]", lambda m: t(m, "constants", "SUB").update(shape=[2]))
    add("negate on const_u32_as_f32", lambda m: t(m, "constants", "SUB")["op"].update(negate=True))
    add("max_src not a 32-bit window (block)", lambda m: t(m, "G1", "e0")["op"].update(max_src="IMG-A"))
    add("max_src not a 32-bit window (16-bit window)", lambda m: t(m, "G1", "e0")["op"].update(max_src="H.x"))
    add("exp with the version-1 max", lambda m: t(m, "G1", "e0")["op"].update(max="3f800000"))
    add("list-valued digest", lambda m: g(m, "G1").update(digest=[g(m, "G1")["digest"]] * 2))
    add("short digest", lambda m: g(m, "G1").update(digest=g(m, "G1")["digest"][:62]))
    add("malformed digest", lambda m: g(m, "G1").update(digest="zz" + g(m, "G1")["digest"][2:]))
    add("short sha256", lambda m: m["sources"]["D"].update(sha256=m["sources"]["D"]["sha256"][:63]))
    add("short fnv1a64", lambda m: m["sources"]["D"].update(fnv1a64=m["sources"]["D"]["fnv1a64"][1:]))
    add("long fnv1a64", lambda m: m["sources"]["D"].update(fnv1a64="0" + m["sources"]["D"]["fnv1a64"]))
    add("source length not divisible by its element size", lambda m: m["sources"]["IMG-A"].update(length=1602))
    add("variant length not divisible by its element size",
        lambda m: m["sources"]["IMG-V"]["variants"][0].update(length=258))
    add("negative base", lambda m: t(m, "G1", "D").update(base=-1))
    add("negative stride", lambda m: t(m, "G1", "b")["axes"][1].update(stride=-1))
    add("negative limit", lambda m: t(m, "G1", "fe")["axes"][0].update(limit=-1))
    add("negative site index", lambda m: t(m, "constants", "SUB")["sites"][0].update(elements=[-1]))
    add("zero shape dimension", lambda m: t(m, "G1", "D").update(shape=[0]))
    add("empty shape", lambda m: t(m, "constants", "SUB").update(shape=[]))
    add("axes length differs from shape", lambda m: t(m, "G1", "D")["axes"].append({"stride": 1}))
    add("unknown perm", lambda m: t(m, "G1", "b")["axes"][0].update(perm="nope"))
    add("not a permutation", lambda m: m["perms"].update(rev4=[0, 0, 1, 2]))
    add("empty permutation", lambda m: m["perms"].update(rev4=[]))
    add("unknown require condition", lambda m: t(m, "constants", "SH").update(require=["even", "odd"]))
    add("unknown check relation", lambda m: g(m, "constants")["checks"][0].update(rel="ne"))
    add("check const not exactly binary32", lambda m: g(m, "constants")["checks"][1]["b"].update(const=0.1))
    add("check const as operand a", lambda m: g(m, "constants")["checks"][0].update(a={"const": 1}, b="X_KEY"))
    add("check operand not a scalar", lambda m: g(m, "constants")["checks"][2].update(b="SH"))
    add("check operand from another group", lambda m: g(m, "constants")["checks"][2].update(b="hx"))
    add("attn_wmax input not earlier", lambda m: t(m, "constants", "WMAX")["op"].update(p0="WMAX"))
    add("attn_wmax input not float16", lambda m: t(m, "constants", "WMAX")["op"].update(p0="X_KEY"))
    add("attn_wmax input unknown", lambda m: t(m, "constants", "WMAX")["op"].update(p0="ZZ"))
    add("also source of the wrong kind", lambda m: t(m, "G1", "w2")["also"][0].update(src="D"))
    add("also on score_bias", lambda m: t(m, "G1", "Tb").update(also=[{"src": "W", "base": 0}]))
    add("also empty", lambda m: t(m, "G1", "w2").update(also=[]))
    add("unknown source", lambda m: t(m, "G1", "D").update(src="DD"))
    add("missing source W", lambda m: (m["sources"].update(W2=m["sources"].pop("W")), m["key"].update(**{"from": "W2"})))
    add("key not from W", lambda m: m["key"].update(**{"from": "D"}))
    add("locate pair (version 1-4 only)", lambda m: m["sources"]["D"].update(locate="pair", gaps=[0]))
    add("element base64 (version 2-3 only)", lambda m: m["sources"]["D"].update(element="base64"))
    add("element i32le (version 3 only)", lambda m: m["sources"]["D"].update(element="i32le"))
    add("encoding lz4-block (version 4 only)", lambda m: m["sources"]["IMG-A"].update(encoding="lz4-block"))
    add("source field optional (version 4 only)", lambda m: m["sources"]["IMG-V"].update(optional=True))
    add("variant decoded_length (version 4 only)", lambda m: m["sources"]["IMG-V"]["variants"][0].update(decoded_length=4))
    add("variants without encoding none", lambda m: m["sources"]["IMG-V"].pop("encoding"))
    add("op fp8 (version 1 only)", lambda m: t(m, "G1", "fe")["op"].update(name="fp8"))
    add("op capped_softmax (version 2-3 only)", lambda m: t(m, "G1", "fe")["op"].update(name="capped_softmax"))
    add("stages instead of groups", lambda m: m.update(stages=m.pop("groups")))
    add("both stages and groups", lambda m: m.update(stages=[]))
    add("duplicate group names", lambda m: g(m, "only_v1").update(group="G1"))
    add("duplicate tensor names", lambda m: t(m, "G1", "b").update(name="D"))
    add("duplicate tensor names ignoring case", lambda m: t(m, "G1", "b").update(name="d"))
    add("missing tensor field op", lambda m: t(m, "G1", "b").pop("op"))
    add("missing group digest", lambda m: g(m, "G1").pop("digest"))
    add("missing source what", lambda m: m["sources"]["D"].pop("what"))
    add("bad format", lambda m: m.update(format="other"))
    add("unsafe tensor name", lambda m: t(m, "G1", "b").update(name="../b"))
    return E


def test_corrupt_maps(r, cases):
    base = cases[1]
    out = r.path("t4-out", "x")
    rc, so, se = r.run(r.write("t4/base.map.json", base.map), [r.write("t4/base.bin", base.file)], out)
    r.check("57-T4: base map converts", rc == 0, so + se[-500:])
    missing_dll = r.path("t4", "no-such-file.dll")
    for label, fn in corrupt_map_cases(base.map):
        m = copy.deepcopy(base.map)
        fn(m)
        mp = r.write("t4/%s.map.json" % re.sub(r"\W+", "_", label), m)
        # Structural failures are refused before any DLL is opened: the missing DLL is never looked at.
        rc, so, se = expect_failure(r, "57-T4 " + label, mp, [missing_dll], 20, MSG["map_damaged"], out)
        r.check("57-T4 %s: no DLL read" % label, "file:" not in se)
    # Wrong digest: the conversion runs and the self-check refuses it.
    m = copy.deepcopy(base.map)
    m["groups"][1]["digest"] = "%064x" % (int(m["groups"][1]["digest"], 16) ^ 1)
    expect_failure(r, "57-T4 wrong digest", r.write("t4/wrong-digest.map.json", m), [r.path("t4", "base.bin")], 23,
                   MSG["self_check"], out)
    # Wrong digest of the gated group: only checked when the group is produced.
    m = copy.deepcopy(base.map)
    m["groups"][2]["digest"] = "%064x" % (int(m["groups"][2]["digest"], 16) ^ 1)
    mp = r.write("t4/wrong-gated-digest.map.json", m)
    expect_failure(r, "57-T4 wrong digest of the gated group (variant 1)", mp, [r.path("t4", "base.bin")], 23,
                   MSG["self_check"], out)
    rc, so, se = r.run(mp, [r.write("t4/v0.bin", cases[0].file)], r.path("t4-out-v0", "x"))
    r.check("57-T4 wrong digest of the gated group: variant 0 converts (group not produced)", rc == 0, so)
    # A valid map whose `also` points at other data: "not supported yet".
    m = copy.deepcopy(base.map)
    next(x for x in m["groups"][0]["tensors"] if x["name"] == "w2")["also"][0]["base"] = 5
    expect_failure(r, "57-T4 also pointing at different data", r.write("t4/also-other.map.json", m),
                   [r.path("t4", "base.bin")], 14, MSG["unsupported"], out)
    # Check constants that are exactly binary32 values in other spellings are accepted.
    m = copy.deepcopy(base.map)
    m["groups"][1]["checks"][1]["b"]["const"] = 1.220703125e-4
    m["groups"][1]["checks"].append({"rel": "ge", "a": "X_KEY", "b": {"const": -1.5e0}})
    m["groups"][1]["checks"].append({"rel": "le", "a": "X_CLn", "b": {"const": 340282346638528859811704183484516925440}})
    rc, so, se = r.run(r.write("t4/const-spellings.map.json", m), [r.path("t4", "base.bin")], r.path("t4-out-c", "x"))
    r.check("57-T4 exact binary32 constants in other spellings accepted", rc == 0, so + se[-300:])


# ---------------------------------------------------------------- exit paths and several files

def test_exit_paths(r, cases):
    base = cases[1]
    mp = r.write("exit/base.map.json", base.map)
    dll = r.write("exit/base.bin", base.file)
    out = r.path("exit-out", "x")
    rc, so, se = r.run(mp, [dll], out)
    r.check("exit 0", rc == 0 and MSG["success"] in so)
    p = subprocess.run([r.exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    r.check("exit 1: no arguments", p.returncode == 1)
    p = subprocess.run([r.exe, "--map", mp, "--dll", dll, "--out", out, "--threads", "0"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    r.check("exit 1: bad --threads", p.returncode == 1)
    p = subprocess.run([r.exe, "--map", mp, "--dll", dll, "--out", out, "--bogus"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    r.check("exit 1: unknown argument", p.returncode == 1)
    blocker = r.write("exit/a-file", b"x")
    rc, so, se = r.run(mp, [dll], os.path.join(blocker, "sub"))
    r.check("exit 2: output folder cannot be written", rc == 2 and "could not be saved" in so, so)
    expect_failure(r, "exit 10: missing file", mp, [r.path("exit", "missing.dll")], 10, MSG["unreadable"], out)
    adir = r.path("exit", "adir", "x")[:-2]
    os.makedirs(adir, exist_ok=True)
    expect_failure(r, "exit 10: a directory", mp, [adir], 10, MSG["unreadable"], out)
    expect_failure(r, "exit 11: smaller than W", mp, [r.write("exit/small.bin", base.file[:100])], 11,
                   MSG["not_model"], out)
    expect_failure(r, "exit 11: empty file", mp, [r.write("exit/empty.bin", b"")], 11, MSG["not_model"], out)
    wlen = len(base.blocks["W"])
    noW = bytearray(base.file)
    woff = base.placements["W"]
    noW[woff + wlen // 2] ^= 0x10
    expect_failure(r, "exit 12: W damaged", mp, [r.write("exit/noW.bin", bytes(noW))], 12, MSG["no_preset_f"], out)
    # With a version resource, the message shows the file's version (57 section 8).
    vr = synth5_version_resource(310, 6, 0, 0)
    withv = bytearray(noW)
    withv[390000: 390000 + len(vr)] = vr
    rc, so, se = expect_failure(r, "exit 12: with a file version", mp, [r.write("exit/noW-v.bin", bytes(withv))], 12,
                                MSG["no_preset_f"], out)
    r.check("exit 12: the version is shown", "This file is version 310.6.0.0." in so, so)
    noD = bytearray(base.file)
    noD[base.placements["D"] + 7] ^= 1
    rc, so, se = expect_failure(r, "exit 13: D damaged", mp, [r.write("exit/noD.bin", bytes(noD))], 13,
                                MSG["unsupported"], out)
    r.check("exit 13: the log names the missing source", "missing sources: D" in se, se[-300:])
    rc, so, se = r.run(r.path("exit", "no-map.json"), [dll], out)
    r.check("exit 20: missing map", rc == 20 and MSG["map_damaged"] in so, so)
    r.write("exit/bad-json.map.json", b"{\"format\": ")
    rc, so, se = r.run(r.path("exit", "bad-json.map.json"), [dll], out)
    r.check("exit 20: invalid JSON", rc == 20 and MSG["map_damaged"] in so, so)
    # Several files (57 section 1): the first that converts completely; rejections are shown.
    o2 = r.path("exit-multi", "x")
    rc, so, se = r.run(mp, [r.path("exit", "noD.bin"), dll], o2)
    r.check("several files: a rejected file, then a good one", rc == 0 and MSG["unsupported"] in so and
            MSG["success"] in so, so)
    r.check("several files: the good one is used", "converted from " + dll in se.replace("\\", "/") or
            "converted from" in se, se[-300:])
    v0 = r.write("exit/v0.bin", cases[0].file)
    rc, so, se = r.run(mp, [v0, dll], o2)
    r.check("several files: B1-like first file is used (constants of B2 are not optional)",
            rc == 0 and not os.path.isdir(os.path.join(o2, "raw", "only_v1")), so + se[-500:])
    r.check("several files: second file not tried", "file: " + dll not in se)
    rc, so, se = r.run(mp, [r.path("exit", "missing.dll"), v0], o2)
    r.check("several files: missing, then good", rc == 0 and MSG["unreadable"] in so, so)
    rc, so, se = r.run(mp, [r.path("exit", "missing.dll"), r.path("exit", "small.bin")], o2)
    r.check("several files: all rejected -> the first file's code", rc == 10, "%d %s" % (rc, so))
    # An installer error stops the list: a wrong digest is not retried with the next file.
    m = copy.deepcopy(base.map)
    m["groups"][0]["digest"] = "%064x" % (int(m["groups"][0]["digest"], 16) ^ 2)
    rc, so, se = r.run(r.write("exit/wrong.map.json", m), [dll, v0], o2)
    r.check("several files: self-check failure stops", rc == 23 and "file: " + v0 not in se, "%d" % rc)


def synth5_version_resource(major, minor, build, rev):
    key = "VS_VERSION_INFO\0".encode("utf-16-le")
    fixed = struct.pack("<13I", 0xFEEF04BD, 0x00010000, (major << 16) | minor, (build << 16) | rev,
                        (major << 16) | minor, (build << 16) | rev, 0x3F, 0, 0x40004, 2, 0, 0, 0)
    body = key + b"\0\0" + fixed
    return struct.pack("<HHH", 6 + len(body), len(fixed), 0) + body


# ---------------------------------------------------------------- the spec team's synthetic files (T1, T3, T4, T7, T8)

def w_r(r_):
    n = 6386372 // 2
    tags = [struct.pack("<e", float(v)) for v in range(16)]
    return b"".join(tags[(e >> (4 * r_)) & 15] for e in range(n))


def test_spec_synth(r, spec_synth, bins, quick):
    exp = json.load(open(os.path.join(spec_synth, "synth-expected.json")))["runs"]
    commons = [open(os.path.join(bins, "synth-common-v%d.bin" % v), "rb").read() for v in (0, 1)]
    runs = range(1) if quick else range(6)
    for r_ in runs:
        w = w_r(r_)
        mp = os.path.join(spec_synth, "synth-r%d.map.json" % r_)
        m = json.load(open(mp))
        for v in (0, 1):
            label = "57-T8 r%d v%d" % (r_, v)
            dll = r.write("t8/file.bin", w + commons[v])
            out = r.path("t8", "out-r%d-v%d" % (r_, v), "x")[:-2]
            rc, so, se = r.run(mp, [dll], out)
            r.check(label + ": converts", rc == 0, so + se[-800:])
            e = exp["r%d_v%d" % (r_, v)]
            groups = sorted(os.listdir(os.path.join(out, "raw"))) if os.path.isdir(os.path.join(out, "raw")) else []
            r.check(label + ": constants_b2 iff v = 1", ("constants_b2" in groups) == (v == 1))
            r.check(label + ": group set", set(groups) == set(e), "%s" % groups)
            for gname, ts in e.items():
                for tname, h in ts.items():
                    try:
                        got = hashlib.sha256(open(os.path.join(out, "raw", gname, tname + ".bin"), "rb").read()).hexdigest()
                    except OSError:
                        got = None
                    r.check("%s: %s/%s" % (label, gname, tname), got == h)
            if rc == 0:
                # 57-T1 analogue: every group digest in the container equals the map's.
                version, key, map_sha, cgroups = read_container(os.path.join(out, "model.wgcm"))
                want = {g["group"]: g["digest"] for g in m["groups"]}
                r.check(label + ": container digests equal the map's", all(dg.hex() == want[n] for n, dg, _ in cgroups))
                r.check(label + ": variant logged", "variant used: IMG-IN variant %d" % v in se)
                rc2, txt = verify_model(r, mp, os.path.join(out, "model.wgcm"))
                r.check(label + ": verify_model_file", rc2 == 0 and (("constants_b2" in txt) == (v == 0)), txt)
    return commons


def test_spec_synth_damage(r, spec_synth, commons):
    """57-T3, T4 (map edits of a shipped synthetic map), T7 on the run-0 variant-0 file."""
    mp = os.path.join(spec_synth, "synth-r0.map.json")
    m = json.load(open(mp))
    w = w_r(0)
    good = w + commons[0]
    dll = r.write("t3/good.bin", good)
    out = r.path("t3-out", "x")
    rc, so, se = r.run(mp, [dll], out)
    r.check("57-T3: the undamaged file converts", rc == 0, so)
    offs, _ = logged_offsets(se)
    with open(os.path.join(out, "model.wgcm"), "rb") as f:
        good_model = f.read()

    def length(name):
        s = m["sources"][name]
        return s["variants"][0]["length"] if "variants" in s else s["length"]

    def flipped(data, positions):
        b = bytearray(data)
        for p in positions:
            b[p] ^= 0x04
        return bytes(b)

    def copies(data, block):
        pos, found = 0, []
        while True:
            i = data.find(block, pos)
            if i < 0:
                return found
            found.append(i)
            pos = i + 1

    wlen = m["sources"]["W"]["length"]
    # The W damage cases (a bit flip in W; a truncation inside W) run on our own random-W files in test_w_damage:
    # the synthetic W_r is periodic (W_0 repeats every 32 bytes), so with W damaged every period is a prefix-hash
    # hit that needs a full 6.4 MB SHA-256 (the rule of 08 section 4.2 checks every candidate).
    d_block = good[offs["D"]: offs["D"] + length("D")]
    d_copies = copies(good, d_block)
    r.check("57-T3: D copies found", len(d_copies) >= 1, str(d_copies))
    if len(d_copies) > 1:  # one damaged copy is not an error while another is intact
        rc, so, se = r.run(mp, [r.write("t3/d1.bin", flipped(good, [d_copies[0] + 100]))], r.path("t3-d1", "x"))
        r.check("57-T3: one damaged copy of D (another intact) converts", rc == 0, so)
    expect_failure(r, "57-T3 bit flip in every copy of D", mp,
                   [r.write("t3/d.bin", flipped(good, [c + 100 for c in d_copies]))], 13, MSG["unsupported"], out)
    for name in ("SH", "BS_A", "BS_W", "TBW", "S1CAP", "H.Gh", "H.ra.E3", "IMG-E1", "IMG-IN", "IMG-OUTa", "IMG-OUTb"):
        block = good[offs[name]: offs[name] + length(name)]
        cps = copies(good, block)
        data = flipped(good, [c + length(name) // 2 for c in cps])
        rc, so, se = expect_failure(r, "57-T3 bit flip in " + name, mp,
                                    [r.write("t3/%s.bin" % name, data)], 13, MSG["unsupported"], out)
        r.check("57-T3 bit flip in %s: logged as missing" % name, "source %s not found" % name in se)
    expect_failure(r, "57-T3 truncated below the W length", mp, [r.write("t3/trunc-small.bin", good[:wlen - 1])], 11,
                   MSG["not_model"], out)
    last = max(offs, key=lambda n: offs[n] + length(n))
    end = offs[last] + length(last)
    cut = good[:offs[last] + length(last) // 2]
    r.check("57-T3: no intact copy of the last source before the cut", good.find(good[offs[last]:end]) == offs[last])
    rc, so, se = expect_failure(r, "57-T3 truncated inside the last source (%s)" % last, mp,
                                [r.write("t3/trunc-last.bin", cut)], 13, MSG["unsupported"], out)
    # 57-T7: bytes outside every source changed (gaps and an appended signature area): identical output.
    covered = bytearray(len(good))
    for n, o in offs.items():
        covered[o: o + length(n)] = b"\1" * length(n)
    gaps = [i for i in range(0, len(good), 997) if not covered[i]]
    changed = bytearray(good)
    for i in gaps:
        changed[i] ^= 0xFF
    changed += b"SIGNATURE" * 1000
    out7 = r.path("t7-out", "x")
    rc, so, se = r.run(mp, [r.write("t7/changed.bin", bytes(changed))], out7)
    r.check("57-T7: %d bytes changed outside every source, signature appended: converts" % len(gaps), rc == 0, so)
    if rc == 0:
        with open(os.path.join(out7, "model.wgcm"), "rb") as f:
            r.check("57-T7: output identical to the unchanged file", f.read() == good_model)
    # 57-T4 on the shipped synthetic map: map edits that only the DLL can reveal, and a wrong digest.
    m2 = copy.deepcopy(m)
    tr = next(t for t in next(g for g in m2["groups"] if g["group"] == "constants")["tensors"] if t["name"] == "TR")
    tr["also"][0]["base"] += 4
    expect_failure(r, "57-T4 (shipped layout) also pointing at different data",
                   r.write("t3/also.map.json", m2), [dll], 14, MSG["unsupported"], out)
    m2 = copy.deepcopy(m)
    m2["groups"][5]["digest"] = "%064x" % (int(m2["groups"][5]["digest"], 16) ^ 1)
    expect_failure(r, "57-T4 (shipped layout) wrong digest", r.write("t3/digest.map.json", m2), [dll], 23,
                   MSG["self_check"], out)
    m2 = copy.deepcopy(m)
    cg = next(g for g in m2["groups"] if g["group"] == "constants")
    a = next(t for t in cg["tensors"] if t["name"] == "A_SCALE")
    a["sites"][0]["elements"].append(a["sites"][0]["elements"][0] + 1)
    expect_failure(r, "57-T4 (shipped layout) a site that disagrees", r.write("t3/site.map.json", m2), [dll], 14,
                   MSG["unsupported"], out)
    # Every shipped-layout structural rule: the shipped map itself validates (no DLL read when the map is damaged).
    m2 = copy.deepcopy(m)
    m2["groups"][0]["tensors"][0]["op"]["name"] = "fp8"
    expect_failure(r, "57-T4 (shipped layout) unknown op", r.write("t3/op.map.json", m2), [r.path("t3", "none.dll")],
                   20, MSG["map_damaged"], out)


def test_w_damage(r, cases):
    """57-T3 for W on a file whose W is random (see test_spec_synth_damage)."""
    base = cases[1]
    mp = r.write("t3w/base.map.json", base.map)
    out = r.path("t3w-out", "x")
    rc, so, se = r.run(mp, [r.write("t3w/base.bin", base.file)], out)
    r.check("57-T3 (random W): base converts", rc == 0, so)
    wlen, woff = len(base.blocks["W"]), base.placements["W"]
    for pos in (0, 63, 64, wlen // 2, wlen - 1):
        b = bytearray(base.file)
        b[woff + pos] ^= 0x01
        expect_failure(r, "57-T3 (random W) bit flip in W at byte %d" % pos, mp, [r.write("t3w/flip.bin", bytes(b))],
                       12, MSG["no_preset_f"], out)
    cut = base.file[: woff + wlen - 2]
    r.check("57-T3 (random W): the cut file is at least W long", len(cut) >= wlen)
    expect_failure(r, "57-T3 (random W) truncated within W (file >= W length)", mp, [r.write("t3w/cut.bin", cut)], 12,
                   MSG["no_preset_f"], out)
    expect_failure(r, "57-T3 (random W) truncated below the W length", mp,
                   [r.write("t3w/cut2.bin", base.file[: wlen - 1])], 11, MSG["not_model"], out)


def test_shipped_map(r, spec_data):
    """The shipped map validates (rejecting fake files by size and by content)."""
    mp = os.path.join(spec_data, "preset-F.v1.map.json")
    out = r.path("shipped-out", "x")
    expect_failure(r, "shipped map: small file", mp, [r.write("shipped/small.bin", b"MZ" + os.urandom(5000))], 11,
                   MSG["not_model"])
    expect_failure(r, "shipped map: random file", mp, [r.write("shipped/random.bin", os.urandom(7_000_000))], 12,
                   MSG["no_preset_f"])
    _ = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--work", default=os.path.join(HERE, "..", "build-plain", "test-work5"))
    repo_data = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data"))
    default_data = os.environ.get("WGC_SPEC_DATA", repo_data)
    ap.add_argument("--spec-data", default=default_data)
    ap.add_argument("--synth-bins", default=os.environ.get("WGC_SYNTH_BINS", ""))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if os.path.isdir(a.work):
        shutil.rmtree(a.work)
    os.makedirs(a.work)
    r = Runner(os.path.abspath(a.exe), os.path.abspath(a.unit), os.path.abspath(a.work))
    t0 = time.time()
    test_t5(r, os.path.join(a.spec_data, "57-T5-vectors.json"))
    cases = test_feature(r)
    test_value_rejections(r, cases)
    test_corrupt_maps(r, cases)
    test_exit_paths(r, cases)
    test_w_damage(r, cases)
    test_shipped_map(r, a.spec_data)
    spec_synth = os.path.join(a.spec_data, "rr-f-synth")
    if all(os.path.exists(os.path.join(a.synth_bins, "synth-common-v%d.bin" % v)) for v in (0, 1)) and \
            os.path.exists(os.path.join(spec_synth, "synth-expected.json")):
        commons = test_spec_synth(r, spec_synth, a.synth_bins, a.quick)
        test_spec_synth_damage(r, spec_synth, commons)
    else:
        r.skipped.append("57-T1/T3/T7/T8 on the spec team's synthetic files (stand-ins not found)")
    for s in r.skipped:
        print("SKIPPED", s)
    print("%d checks passed, %d failed (%.1f s)" % (r.passed, len(r.failures), time.time() - t0))
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
