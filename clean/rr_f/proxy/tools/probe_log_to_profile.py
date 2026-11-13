# SPDX-License-Identifier: MIT
"""Probe log -> compact machine-readable game profile (the game's RR traffic as the probe build saw it).

    python clean/rr_f/proxy/tools/probe_log_to_profile.py <probe.log> [--summary <benchmark-summary.json>]
           [--game cp2077] [--out clean/rr_f/proxy/tests/data/<game>-<version>-traffic.json]

Input: a probe log written by the probe build (format: src/probe_log.cpp / README "The probe log"):
`hh:mm:ss.mmm f=<frame> <key>: <values> ; <extra>`, frames counted at <new-frame-token>, frames 0..600 in full,
afterwards a record only when its compared part (everything but seq / tid / frame index / jitter) changes.

Output: one JSON file (schema wronggpu.rr-proxy.game-profile.v1) with the structures the game passed, the call order,
tags (type, format, state, extent, lifetime), options, constants flags and their changes, jitter / camera samples,
normal samples, call totals and timing. The replay test (tests/test_replay_profile.cpp) drives the proxy core with
it. Every value is copied or counted from the log; "derived" fields say how they were computed.

Leak-scan rules: the log holds the RR feature-function names and other vendor function names; this tool never
copies a function name. For feature 1001 it maps the requested name to its role through binding/rr-functions.json
(names[i] -> roles[i]); for other features the name is withheld (only the feature id and the result are kept).
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)

LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+) f=(\d+) (.*)$")

# Public DXGI_FORMAT numbers (Windows SDK dxgiformat.h) of the formats a game may tag.
DXGI = {
    0: "UNKNOWN", 2: "R32G32B32A32_FLOAT", 9: "R16G16B16A16_TYPELESS", 10: "R16G16B16A16_FLOAT",
    11: "R16G16B16A16_UNORM", 13: "R16G16B16A16_SNORM", 16: "R32G32_FLOAT", 24: "R10G10B10A2_UNORM",
    26: "R11G11B10_FLOAT", 27: "R8G8B8A8_TYPELESS", 28: "R8G8B8A8_UNORM", 29: "R8G8B8A8_UNORM_SRGB",
    34: "R16G16_FLOAT", 35: "R16G16_UNORM", 37: "R16G16_SNORM", 39: "R32_TYPELESS", 40: "D32_FLOAT",
    41: "R32_FLOAT", 44: "R24G8_TYPELESS", 45: "D24_UNORM_S8_UINT", 49: "R8G8_UNORM", 53: "R16_TYPELESS",
    54: "R16_FLOAT", 55: "D16_UNORM", 56: "R16_UNORM", 61: "R8_UNORM", 87: "B8G8R8A8_UNORM",
}
# D3D12_RESOURCE_STATES bits (public d3d12.h).
STATES = [(0x1, "VERTEX_AND_CONSTANT_BUFFER"), (0x2, "INDEX_BUFFER"), (0x4, "RENDER_TARGET"),
          (0x8, "UNORDERED_ACCESS"), (0x10, "DEPTH_WRITE"), (0x20, "DEPTH_READ"),
          (0x40, "NON_PIXEL_SHADER_RESOURCE"), (0x80, "PIXEL_SHADER_RESOURCE"), (0x100, "STREAM_OUT"),
          (0x200, "INDIRECT_ARGUMENT"), (0x400, "COPY_DEST"), (0x800, "COPY_SOURCE")]
# D3D12_RESOURCE_FLAGS bits.
RFLAGS = [(0x1, "ALLOW_RENDER_TARGET"), (0x2, "ALLOW_DEPTH_STENCIL"), (0x4, "ALLOW_UNORDERED_ACCESS"),
          (0x8, "DENY_SHADER_RESOURCE")]
# Spec 62.2 roles of the buffer types (our short names) and whether the RR path / preset F uses them.
BUFFER = {
    0: ("hardware depth", "one-of depth"), 1: ("motion vectors", "required"), 3: ("scaling input colour", "required"),
    4: ("scaling output colour", "required (written)"), 5: ("normals", "required with packing 0"),
    6: ("roughness", "required with packing 0"), 7: ("diffuse albedo", "required"), 8: ("specular albedo", "required"),
    10: ("specular motion vectors", "optional, read"), 11: ("disocclusion mask", "optional, read"),
    13: ("exposure", "not used by preset F"), 14: ("normals + roughness", "required with packing 1"),
    29: ("current-colour-bias hint mask", "not used"), 39: ("colour before particles", "optional: the layer (replaces 51)"),
    40: ("colour before transparency", "optional, read"), 41: ("colour before fog", "optional: G1 (replaces 58)"),
    42: ("specular hit distance", "optional, read (specular-motion reconstruction)"),
    49: ("linear depth", "one-of depth, preferred"), 51: ("transparency layer", "optional: the layer"),
    52: ("transparency opacity", "optional, with a layer"), 58: ("subsurface-scattering guide", "optional: G1"),
    64: ("depth-of-field guide", "optional, read"),
}
PREF_BITS = {0: "command-list state tracking disabled", 1: "no debug text", 2: "manual hooking",
             3: "allow over-the-air updates", 4: "bypass OS version check", 5: "DXGI factory proxy",
             6: "load downloaded plugins", 7: "frame-based tagging (2.14.1 only)"}


def bits(v, table):
    return [n for b, n in table if v & b]


def kv(text):
    """'a=1 b=x,y c=...' -> dict (values kept as strings)."""
    out = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def ms(m):
    return ((int(m.group(1)) * 60 + int(m.group(2))) * 60 + int(m.group(3))) * 1000 + int(m.group(4))


def num(s):
    try:
        return int(s, 0)
    except ValueError:
        return float(s)


def floats(s):
    return [float(x) for x in s.split(",")]


def parse(path):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            body = m.group(6)
            key, _, rest = body.partition(": ")
            val, _, extra = rest.partition(" ; ")
            recs.append({"line": n, "ms": ms(m), "frame": int(m.group(5)), "key": key, "val": val, "extra": extra})
    return recs


def rr_function_roles():
    with open(os.path.join(PROXY, "binding", "rr-functions.json")) as f:
        t = json.load(f)
    return dict(zip(t["names"], t["roles"]))


def build(recs, summary, game, log_path, log_sha, nlines):
    roles_by_name = rr_function_roles()
    prof = {"schema": "wronggpu.rr-proxy.game-profile.v1", "game": game}
    src = {"log": log_path, "log_sha256": log_sha, "log_lines": nlines}
    start = next((r for r in recs if r["key"] == "probe start"), None)
    if start:
        src["probe"] = start["val"]
    hook = [r for r in recs if r["key"] == "hook"]
    src["hook_mode"] = bool(hook)
    if summary:
        d = summary.get("Data", {})
        src["benchmark"] = {k: d.get(k) for k in ("gameVersion", "renderWidth", "renderHeight", "averageFps", "minFps",
                                                  "maxFps", "time", "frameNumber", "rayTracedPathTracingEnabled",
                                                  "DRSEnabled", "verticalSync", "gpuName", "gpuDriverVersion")}
    prof["source"] = src
    unseen = [r for r in recs if r["key"] == "init unseen"]
    prof["probe_caveats"] = []
    if unseen:
        prof["probe_caveats"].append(
            {"line": unseen[0]["line"], "what": "hook mode: the probe adopted an unseen <init> at plugin load (flags "
             "taken as 0) before the game's real <init>; the real <init> calls were then seen and logged; the "
             "adoption is a probe-build artefact, not a game call"})
    prof["probe_caveats"].append(
        {"what": "frames are counted at <new-frame-token> (several per frame index); the full-log window of 600 frames "
         "ended during loading (frame index 149), so every RR-phase record is the first one plus changes only"})

    # ---- init sessions and the calls before the first frame token
    inits = [r for r in recs if r["key"] == "init"]
    sessions = []
    for r in inits:
        v = kv(r["val"])
        flags = int(v["flags"], 0)
        sessions.append({"line": r["line"], "sdk": v["sdk"], "magic": int(v["magic"], 0), "flags": flags,
                         "flag_bits": {str(b): PREF_BITS[b] for b in range(8) if flags >> b & 1},
                         "features": [int(x) for x in v["features"].strip("[]").split(",") if x],
                         "render_api": int(v["render_api"]), "engine": int(v["engine"]),
                         "log_level": int(v["log_level"]), "app_id": int(v["app_id"]),
                         "plugin_paths": int(v["plugin_paths"]), "prefs_version": int(v["prefs_version"]),
                         "rc": int(v["rc"])})
    prof["api_minor"] = int(sessions[0]["sdk"].split(".")[1]) if sessions else None
    prof["init_sessions"] = sessions

    calls = []
    for r in recs:
        if not r["key"].startswith("call "):
            continue
        e = kv(r["extra"])
        if "seq" not in e:
            continue
        head = r["key"][5:].split()
        c = {"seq": int(e["seq"]), "line": r["line"], "frame": r["frame"], "role": head[0],
             "rc": int(kv(r["val"])["rc"].split()[0]), "tid": int(e.get("tid", 0))}
        kk = kv(" ".join(head[1:]))
        if "feature" in kk:
            c["feature"] = int(kk["feature"])
        if "value" in kk:
            c["value"] = int(kk["value"])
        if "count" in kk:
            c["count"] = int(kk["count"])
        if "vp" in kk:
            c["viewport"] = int(kk["vp"])
        if "name" in kk:  # names are never copied (leak-scan rule): RR names map to their role
            if c.get("feature") == 1001:
                c["function_role"] = roles_by_name.get(kk["name"], "unknown")
            else:
                c["function_role"] = None
        if "frame_index" in e:
            c["frame_index"] = int(e["frame_index"])
        calls.append(c)
    first_token_seq = min((c["seq"] for c in calls if c["role"] == "new-frame-token"), default=None)
    setup = []
    for c in sorted(calls, key=lambda c: c["seq"]):
        if first_token_seq is not None and c["seq"] >= first_token_seq:
            break
        s = {k: c[k] for k in ("seq", "line", "role", "feature", "value", "function_role", "rc") if k in c}
        setup.append(s)
    for r in recs:  # set-device and the adapter details
        if r["key"] == "set-device":
            prof["set_device"] = dict(kv(r["val"]), line=r["line"])
        if r["key"] == "is-supported adapter":
            prof["is_supported_adapter"] = dict(kv(r["val"]), line=r["line"])
    prof["setup_calls"] = setup
    prof["get_function_calls"] = [
        {k: c[k] for k in ("seq", "line", "frame", "feature", "function_role", "rc") if k in c}
        for c in calls if c["role"] == "get-function"]

    # ---- frame tokens and constants in the full-log window
    tok = [r for r in recs if r["key"] == "new-frame-token"]
    ft = next((r for r in recs if r["key"] == "frame-token first"), None)
    per_index = {}
    eq = True
    given = True
    for r in tok:
        e = kv(r["extra"])
        v = kv(r["val"])
        given = given and v.get("index_argument") == "given"
        ia, it = int(e["index_argument_value"]), int(e["index_from_table"])
        eq = eq and ia == it
        per_index.setdefault(it, []).append(r["frame"])
    consts_full = [r for r in recs if r["key"].startswith("constants vp") and r["frame"] <= 600]
    cidx = {}
    for r in consts_full:
        cidx.setdefault(int(kv(r["extra"])["frame_index"]), []).append(r["frame"])
    full_indices = sorted(per_index)
    complete = [i for i in full_indices if i != full_indices[-1]]  # the last index may continue past frame 600
    counts = {}
    for i in complete:
        counts.setdefault(str(len(per_index[i])), []).append(i)
    counts = {k: (v if len(v) <= 3 else "%d indices (%d..%d)" % (len(v), v[0], v[-1])) for k, v in counts.items()}
    first_tok_of = {i: min(per_index[i]) for i in complete}
    on_first = all(cidx.get(i, [None])[0] == first_tok_of[i] for i in complete)
    ft_v = kv(ft["val"]) if ft else {}
    prof["frame_tokens"] = {
        "token_layout": {"line": ft["line"] if ft else None,
                         "guid_at_16_is_frame_token_guid": ft_v.get("matches_frame_token_guid") == "1",
                         "version_at_32": int(ft_v.get("version_at_32", 0))},
        "index_argument": "given" if given else "sometimes null",
        "index_from_table_equals_argument_frames_1_600": eq,
        "loading_window": {"frames": [tok[0]["frame"], tok[-1]["frame"]] if tok else None,
                           "frame_indices": [full_indices[0], full_indices[-1]] if full_indices else None,
                           "tokens_per_index": counts,
                           "last_index_cut_by_window": full_indices[-1] if full_indices else None,
                           "constants_per_index": sorted({len(cidx.get(i, [])) for i in complete}),
                           "constants_on_first_token_of_index": on_first,
                           "index_step_per_new_index": 1},
    }

    # ---- constants: flags, version, changes over the whole run
    crec = [r for r in recs if r["key"].startswith("constants vp")]
    changes, last = [], None
    for r in crec:
        v = kv(r["val"])
        cur = {"reset": int(v["reset"]), "mv_scale": floats(v["mv_scale"]),
               "camera_motion_included": int(v["camera_motion_included"]), "mv_jittered": int(v["mv_jittered"]),
               "depth_inverted": int(v["depth_inverted"]), "mv_3d": int(v["mv_3d"]),
               "mv_dilated": int(v["mv_dilated"]), "orthographic": int(v["orthographic"]),
               "version": int(v["version"])}
        if cur != last:
            changes.append(dict(cur, frame_index=int(kv(r["extra"])["frame_index"]), line=r["line"]))
            last = cur
    prof["constants"] = {"viewport": 0, "changes": changes,
                         "note": "a change list over the whole run: before frame 600 every call was logged, afterwards "
                                 "only changes; the values hold from frame_index until the next entry"}

    # ---- the first RR frame: tags, options, evaluate
    tags = []
    for r in recs:
        m = re.match(r"^tag vp=(\d+) type=(\d+)$", r["key"])
        if not m:
            continue
        v = kv(r["val"])
        t = {"order": len(tags), "line": r["line"], "viewport": int(m.group(1)), "type": int(m.group(2)),
             "role": BUFFER.get(int(m.group(2)), ("?", "?"))[0], "rr_use": BUFFER.get(int(m.group(2)), ("?", "?"))[1],
             "lifetime": int(v["lifetime"]), "extent": [int(x) for x in v["extent"].split(",")]}
        if "resource" in v and v["resource"] == "null":
            t["resource"] = None
        else:
            st = int(v["state"], 0)
            nf = int(v["desc_format"])
            t["resource"] = {"kind": int(v["kind"]), "desc_format": int(v["tag_format"]),
                             "desc_size": [int(x) for x in v["res"].split("x")], "state": st,
                             "state_names": bits(st, STATES), "desc_mips": int(v["mips"]),
                             "desc_layers": int(v["layers"]), "version": int(v["res_version"]),
                             "native": {"format": nf, "format_name": DXGI.get(nf, f"DXGI_{nf}"),
                                        "size": [int(x) for x in v["desc"].split("x")],
                                        "dimension": int(v["desc_dim"]), "samples": int(v["desc_samples"]),
                                        "flags": int(v["desc_flags"], 0),
                                        "flag_names": bits(int(v["desc_flags"], 0), RFLAGS),
                                        "layers": int(v["desc_layers"]), "mips": int(v["desc_mips"])}}
        tags.append(t)
    chains = [r for r in recs if r["key"].startswith("chain tag ")]
    ext = any("->" in r["val"] for r in chains)
    tag_calls = [c for c in calls if c["role"] == "tag"]
    prof["tags"] = {"api": "tag (viewport tags; no local tags, no frame-scoped tags)",
                    "count_per_call": sorted({c.get("count") for c in tag_calls}),
                    "chained_extension_seen": ext, "list": tags}

    o = next((r for r in recs if r["key"].startswith("rr-options vp")), None)
    if o:
        v = kv(o["val"])
        prof["rr_options"] = {"line": o["line"], "mode": int(v["mode"]),
                              "output": [int(x) for x in v["output"].split("x")], "sharpness": float(v["sharpness"]),
                              "pre_exposure": float(v["pre_exposure"]), "exposure_scale": float(v["exposure_scale"]),
                              "hdr": int(v["hdr"]), "invert_x": int(v["invert_x"]), "invert_y": int(v["invert_y"]),
                              "normal_roughness_packing": int(v["normal_roughness_mode"]),
                              "version": int(v["version"]), "alpha_upscaling": int(v["alpha_upscaling"]),
                              "presets": [int(x) for x in v["presets"].split(",")],
                              "release_rc": int(v["release_rc"]),
                              "changed_during_run": sum(1 for r in recs if r["key"].startswith("rr-options vp")) > 1}
    ev = next((r for r in recs if r["key"].startswith("evaluate vp")), None)
    if ev:
        v = kv(ev["val"])
        ins = [r for r in recs if r["key"].startswith("chain evaluate.input")]
        prof["evaluate"] = {"line": ev["line"], "render": [int(x) for x in v["render"].split("x")],
                            "output": [int(x) for x in v["output"].split("x")],
                            "output_rect": [int(x) for x in v["output_rect"].split(",")], "mode": int(v["mode"]),
                            "options_given": int(v["options"]), "constants_for_token": int(v["constants"]),
                            "normal_type": int(v["normal_type"]), "normal_format": int(v["normal_format"]),
                            "roughness": v["roughness_format"], "colour_format": int(v["colour_format"]),
                            "output_format": int(v["output_format"]), "local_tags": int(v["locals"]),
                            "inputs": [r["val"].split("{")[0] for r in ins],
                            "command_list": v["command_list"],
                            "changed_during_run": sum(1 for r in recs if r["key"].startswith("evaluate vp")) > 1}
    up = [r for r in recs if r["key"].startswith("upscale vp")]
    if up:
        prof["probe_upscale"] = {"line": up[0]["line"], "rc": int(kv(up[0]["val"])["rc"]),
                                 "first_ms": up[0]["ms"] - ev["ms"] if ev else None}

    # ---- per-frame call order of the RR phase (first two evaluates; seq numbers)
    evs = sorted((c for c in calls if c["role"] == "evaluate"), key=lambda c: c["seq"])
    watch = [r for r in recs if r["key"] == "colour-extent-watch"]
    order = {}
    if evs:
        e1 = evs[0]["seq"]
        t1 = [c for c in tag_calls]
        tag_seq0 = t1[0]["seq"] if t1 else None
        rr_opt = [c for c in calls if c["role"] == "rr-options"]
        gf = [c for c in calls if c["role"] == "get-function" and c.get("feature") == 1001]
        seqs2 = [int(kv(r["extra"])["seq"]) for r in watch]
        tag_frame = next((r["frame"] for r in recs if r["line"] == tags[-1]["line"]), None) if tags else None
        opt_frame = o["frame"] if o else None
        order = {
            "first_rr_frame": {
                "tag_seq_first": tag_seq0, "tag_calls": len(tags),
                "tag_seq_last_derived": tag_seq0 + len(tags) - 1 if tag_seq0 is not None else None,
                "new_frame_tokens_between_tags_and_options": (opt_frame - tag_frame)
                if tag_frame is not None and opt_frame is not None else None,
                "get_function_rr_options_seq": gf[0]["seq"] if gf else None,
                "rr_options_seq": rr_opt[0]["seq"] if rr_opt else None, "evaluate_seq": e1},
            "calls_between_evaluate_1_and_2": (seqs2[1] - seqs2[0]) if len(seqs2) > 1 else None,
            "tokens_between_evaluate_1_and_2": (watch[1]["frame"] - watch[0]["frame"]) if len(watch) > 1 else None,
            "per_frame": ["new-frame-token (first of the frame index)", "constants (frame index N)",
                          "tag x %d (count=1 each, order of tags.list)" % len(tags),
                          "new-frame-token (same frame index; the token later passed to evaluate)",
                          "get-function(1001, rr-options role) - first RR frame only", "rr-options",
                          "evaluate(1001, token of index N, inputs=[viewport], count=1)"],
            "per_frame_note": "constants -> tags -> token -> options -> evaluate is read from the first RR frame "
                              "(log order and seq numbers); other new-frame-token calls of the same index (about 9 "
                              "per index in the RR phase) and their exact positions are not logged after frame 600",
        }
    prof["call_order"] = order

    # ---- jitter / camera samples
    cams = [r for r in recs if r["key"].startswith("camera vp")]
    js = []
    for i, r in enumerate(cams):
        v, e = kv(r["val"]), kv(r["extra"])
        js.append({"evaluate": i + 1, "line": r["line"], "frame_index": int(e["frame_index"]),
                   "jitter": floats(e["jitter"]), "v2c20": float(v["v2c20"]), "v2c21": float(v["v2c21"])})
    prof["jitter_camera"] = {"samples": js,
                             "camera_pair_changed_after_evaluate_2": len(cams) > 2,
                             "note": "after frame 600 the camera record (and the jitter in it) is logged only when "
                                     "the camera-view-to-clip pair [2][0], [2][1] changes; no change was logged, so "
                                     "the pair stayed 0, 0 for every evaluate and only these jitter samples exist"}

    # ---- normal samples
    ns = []
    for r in recs:
        if not r["key"].startswith("normal-patch sample="):
            continue
        v = kv(r["val"])
        mean = floats(v["mean"])
        w2v = floats(v["world_to_view"])
        view = [sum(mean[k] * w2v[4 * k + j] for k in range(3)) for j in range(3)]
        ns.append({"sample": int(r["key"].split("=")[1]), "line": r["line"], "evaluate": int(v["eval"]),
                   "type": int(v["type"]), "format": int(v["format"]),
                   "origin": [int(x) for x in v["origin"].split(",")], "size": v["size"], "mean": mean,
                   "world_to_view": w2v, "derived_view_xyz_if_world_space": [round(x, 4) for x in view]})
    prof["normal_samples"] = ns

    # ---- totals and timing
    summ = [r for r in recs if r["key"] == "summary at shutdown"]
    if summ:
        prof["totals_at_shutdown"] = dict({k: int(x) for k, x in kv(summ[-1]["val"]).items()}, line=summ[-1]["line"])
        prof["sessions_ended_by_shutdown"] = len(summ)
    timing = {}
    t0 = recs[0]["ms"]
    ev_c = [r for r in recs if r["key"].startswith("call evaluate")]
    if inits:
        timing["init_ms"] = [r["ms"] - next(x["ms"] for x in recs if x["key"] == "init entered" and
                                              x["line"] < r["line"] and x["line"] > (inits[i - 1]["line"] if i else 0))
                             for i, r in enumerate(inits)]
    if ft:
        timing["plugin_load_to_first_token_ms"] = ft["ms"] - t0
    if ev and ft:
        timing["first_token_to_first_evaluate_ms"] = ev["ms"] - ft["ms"]
    if len(watch) > 1:
        timing["evaluate_1_to_2_ms"] = watch[1]["ms"] - watch[0]["ms"]
    if len(ns) > 1:
        gaps = [(b["evaluate"] - a["evaluate"], b_r - a_r) for a, b, a_r, b_r in
                zip(ns, ns[1:], [x["ms"] for x in recs if x["key"].startswith("normal-patch")],
                    [x["ms"] for x in recs if x["key"].startswith("normal-patch")][1:])]
        per = sorted(round(dt / de, 2) for de, dt in gaps)
        nsr = [x for x in recs if x["key"].startswith("normal-patch")]
        timing["ms_per_evaluate_from_normal_samples"] = {
            "mean": round((nsr[-1]["ms"] - nsr[0]["ms"]) / (ns[-1]["evaluate"] - ns[0]["evaluate"]), 3),
            "min": per[0], "median": per[len(per) // 2], "max": per[-1],
            "intervals_over_25ms_per_evaluate": [{"from_evaluate": ns[i]["evaluate"], "ms_per_evaluate":
                                                  round(gaps[i][1] / gaps[i][0], 2)}
                                                 for i in range(len(gaps)) if gaps[i][1] / gaps[i][0] > 25]}
        tok_per = [(b["frame"] - a["frame"]) / (bn["evaluate"] - an["evaluate"]) for a, b, an, bn in
                   zip(nsr, nsr[1:], ns, ns[1:])]
        timing["tokens_per_evaluate_from_normal_samples"] = [round(min(tok_per), 2), round(max(tok_per), 2)]
    if summ and ev:
        tot = prof["totals_at_shutdown"]
        timing["rr_phase_ms"] = summ[-1]["ms"] - ev["ms"]
        timing["rr_phase_ms_per_evaluate_upper_bound"] = round((summ[-1]["ms"] - ev["ms"]) / tot["evaluate"], 3)
        tokens_rr = tot["new-frame-token"] - (ev["frame"] - 1)
        timing["tokens_per_evaluate_rr_phase"] = round(tokens_rr / tot["evaluate"], 3)
    prof["timing"] = timing
    if evs:
        prof["rr_phase"] = {"first_evaluate_frame_index": evs[0].get("frame_index"),
                            "evaluates": prof.get("totals_at_shutdown", {}).get("evaluate"),
                            "evaluate_rc_nonzero_records": sum(1 for c in calls if c["role"] == "evaluate" and c["rc"]),
                            "frame_index_step_per_evaluate":
                                (js[1]["frame_index"] - js[0]["frame_index"]) if len(js) > 1 else None,
                            "constants_after_first_evaluate_derived":
                                prof["totals_at_shutdown"]["constants"] - evs[0].get("frame_index", 0)
                                if "totals_at_shutdown" in prof else None,
                            "note": "constants_after_first_evaluate_derived = total constants - first RR frame index "
                                    "(one constants call per frame index before it); it matches evaluates + 1"}
    return prof


def dump(obj, ind=0):
    """JSON with one line per leaf list / small object (keeps numeric lines short for the leak scan)."""
    sp = " " * ind
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        flat = json.dumps(obj, separators=(", ", ": "))
        if len(flat) <= 260 and not any(isinstance(v, (dict, list)) and v and isinstance(v, dict) for v in obj.values()):
            return flat
        items = [f'{sp} {json.dumps(k)}: {dump(v, ind + 1)}' for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + f"\n{sp}}}"
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return json.dumps(obj)
        items = [f"{sp} {dump(v, ind + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + f"\n{sp}]"
    return json.dumps(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--summary")
    ap.add_argument("--game", default="cp2077")
    ap.add_argument("--out")
    a = ap.parse_args()
    raw = open(a.log, "rb").read()
    recs = parse(a.log)
    summary = json.load(open(a.summary)) if a.summary else None
    prof = build(recs, summary, a.game, os.path.basename(a.log), hashlib.sha256(raw).hexdigest(),
                 raw.count(b"\n"))
    if summary:
        prof["source"]["summary"] = os.path.basename(a.summary)
        prof["source"]["summary_sha256"] = hashlib.sha256(open(a.summary, "rb").read()).hexdigest()
    text = dump(prof) + "\n"
    if a.out:
        with open(a.out, "w", newline="\n") as f:
            f.write(text)
        print(f"wrote {a.out} ({len(text)} bytes)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
