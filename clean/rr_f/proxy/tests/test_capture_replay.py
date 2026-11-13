# SPDX-License-Identifier: MIT
"""Tests of the offline RR runner on captures (tools/rr_capture_replay.py, CAPTURE.md).

    python -m unittest test_capture_replay                       (from clean/rr_f/proxy/tests; CPU only)
    python clean/rr_f/proxy/tests/test_capture_replay.py --gpu [--out DIR] [--model FILE] [--warp DIR]

CPU tests: a synthetic, game-like capture (game A's formats: R32 typeless depth, RGBA16F colour / motion / normals +
roughness / colour before particles, RGBA8 albedos, R32F hit distance, R16F guide, R8 hint, the output only described)
written by an independent writer of the capture format; every decoder against the values written; the proxy's part
(routing of 62.2, creation decisions of 61.3, the test host's frame file parsed back by an independent reader: depth
bits unchanged as R32_FLOAT, origins, data); tonemap, PNG writer, resampling. With WG_RR_PROXY_BUILD (set by
tools/build.py) or --warp, the capture the probe ASI wrote on WARP in test_hook_dll is decoded and compared with the
patterns that test uploaded.

--gpu (end to end, the real GPU under the GPU lock (WG_GPU_LOCK, BUILDING.md)): the runner on synthetic captures (8 frames 192x108 ->
288x162; 3 frames at game A's 1706x960 -> 2560x1440) and on the WARP capture, with the synthetic model
(default: WG_RR_MODEL): every evaluate rc 0 and completed, the PNGs
there with the right sizes, the summary written.
"""
import json
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import rr_capture_replay as R  # noqa: E402

SYNTH_MODEL = os.environ.get("WG_RR_MODEL", "")


# ------------------------------------------------------------------------------------------------- capture writer
def write_input(path, btype, fmt, texels, state, extent=(0, 0, 0, 0), tag_format=None, lifetime=1, pad=0):
    """An input file as the probe writes it: 64-byte header, rows (optionally `pad` extra bytes per row)."""
    t = np.ascontiguousarray(texels, np.uint8)
    h, w = t.shape[:2]
    row = t.reshape(h, -1)
    if pad:
        row = np.concatenate([row, np.full((h, pad), 0xAB, np.uint8)], axis=1)
    hdr = R.MAGIC + struct.pack("<14I", 64, btype, fmt, w, h, row.shape[1], fmt if tag_format is None else tag_format,
                                state, extent[0], extent[1], extent[2], extent[3], lifetime, 0)
    with open(path, "wb") as fh:
        fh.write(hdr + row.tobytes())
    return dict(type=btype, file=os.path.basename(path), format=fmt, width=w, height=h, row_pitch=row.shape[1],
                tag_format=fmt if tag_format is None else tag_format, state=state, lifetime=lifetime, kind=0,
                extent=dict(left=extent[0], top=extent[1], width=extent[2], height=extent[3]), tag_size=[0, 0])


def as_bytes(a, dtype):
    a = np.ascontiguousarray(np.asarray(a).astype(dtype))
    return a.view(np.uint8).reshape(a.shape[0], a.shape[1], -1)


def perspective(w, h, fov_y=1.0, near=0.1):
    f = 1.0 / math.tan(fov_y / 2)
    m = np.zeros((4, 4), np.float32)   # row-major, reversed infinite depth (depth inverted)
    m[0, 0], m[1, 1], m[2, 3], m[3, 2] = f * h / w, f, 1.0, near
    return m


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def scene(x, y, rw, rh):
    """A simple HDR picture in render pixels: sky gradient, checker floor, two bright disks, a thin bright line."""
    u, v = x / rw, y / rh
    sky = np.stack([0.3 + 0.4 * (1 - v), 0.5 + 0.5 * (1 - v), 1.2 + 0.8 * (1 - v)], axis=-1)
    check = ((np.floor(x / 16) + np.floor(y / 16)) % 2)[..., None]
    floor = 0.08 + 0.5 * check * np.array([0.9, 0.7, 0.5])
    img = np.where((v > 0.45)[..., None], floor, sky)
    for cx, cy, r, c in ((0.3, 0.3, 0.08, (6.0, 5.0, 3.0)), (0.7, 0.62, 0.05, (0.5, 3.0, 8.0))):
        d = np.hypot((u - cx) * rw / rh, v - cy)
        img = np.where((d < r)[..., None], np.array(c), img)
    line = (np.abs(y - 0.8 * rh - 0.15 * (x - rw / 2)) < 0.6)[..., None]
    return np.where(line, np.array([4.0, 4.0, 4.0]), img)


def synthetic_capture(folder, frames, rw, rh, wo, ho, start=300, seed=7, pan=1.5, noise=0.15):
    """Writes `frames` consecutive frame folders shaped like game A's (types and formats of the probe run)."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    v2c = perspective(rw, rh)
    truth = []
    for k in range(frames):
        n = start + k
        d = os.path.join(folder, f"frame_{n:06d}")
        os.makedirs(d, exist_ok=True)
        jit = (halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5)
        cam = pan * k
        ys, xs = np.mgrid[0:rh, 0:rw].astype(np.float64)
        sx, sy = xs + 0.5 + jit[0] + cam, ys + 0.5 + jit[1]
        colour = scene(sx, sy, rw, rh) * (1 + noise * rng.standard_normal((rh, rw, 1)))
        c4 = np.concatenate([colour, np.ones((rh, rw, 1))], axis=2)
        mv = np.zeros((rh, rw, 4))
        mv[..., 0] = -pan / rw          # units of the render size (scale 1, 1); a point moves left as the camera pans
        depth = np.where(ys / rh > 0.45, 0.02 + 0.5 * (ys / rh - 0.45), 0.0).astype(np.float32)   # inverted: sky 0
        nrm = np.zeros((rh, rw, 4))
        nrm[..., 2] = np.where(ys / rh > 0.45, 1.0, -1.0)
        nrm[..., 3] = 0.4                                                                    # roughness
        alb = np.clip(scene(sx, sy, rw, rh) / 6.0, 0, 1)
        dif = np.concatenate([alb, np.ones((rh, rw, 1))], axis=2)
        spec = np.full((rh, rw, 4), 0.04)
        hit = np.where(ys / rh > 0.45, 5.0, 0.0).astype(np.float32)
        guide = np.zeros((rh, rw), np.float16)
        hint = np.zeros((rh, rw), np.uint8)
        inputs = [
            write_input(os.path.join(d, "input_00.raw"), 0, 39, as_bytes(depth[..., None], np.float32), 0xC0),
            write_input(os.path.join(d, "input_01.raw"), 1, 10, as_bytes(mv, np.float16), 0xC0, (0, 0, rw, rh)),
            write_input(os.path.join(d, "input_03.raw"), 3, 10, as_bytes(c4, np.float16), 0x8, (0, 0, rw, rh), pad=8),
            write_input(os.path.join(d, "input_07.raw"), 7, 28, as_bytes(np.rint(dif * 255), np.uint8), 0x8),
            write_input(os.path.join(d, "input_08.raw"), 8, 28, as_bytes(np.rint(spec * 255), np.uint8), 0x8),
            write_input(os.path.join(d, "input_14.raw"), 14, 10, as_bytes(nrm, np.float16), 0x8),
            write_input(os.path.join(d, "input_29.raw"), 29, 61, hint[..., None], 0x8),
            write_input(os.path.join(d, "input_39.raw"), 39, 10, as_bytes(c4, np.float16), 0x8),
            write_input(os.path.join(d, "input_42.raw"), 42, 41, as_bytes(hit[..., None], np.float32), 0x8),
            write_input(os.path.join(d, "input_58.raw"), 58, 54, as_bytes(guide[..., None], np.float16), 0x8),
        ]
        w2v = np.eye(4, dtype=np.float32)
        w2v[3, 0] = -0.01 * cam
        meta = {
            "format": "wronggpu-rr-capture", "version": 1, "evaluate": n, "frame_index": 3654 + n, "viewport": 0,
            "api_minor": 7, "preference_flags": 0x69, "render": {"x": 0, "y": 0, "width": rw, "height": rh},
            "output": {"type": 4, "tag_format": 10, "format": 10, "width": wo, "height": ho, "rect": [0, 0, wo, ho],
                       "state": 8},
            "options": {"given": True, "version": 3, "mode": 3, "output_width": wo, "output_height": ho,
                        "sharpness": 0.0, "pre_exposure": 1.0, "exposure_scale": 1.0, "colour_is_hdr": 1,
                        "indicator_invert_x": 0, "indicator_invert_y": 0, "normal_roughness_packing": 1,
                        "world_to_view": [float(x) for x in w2v.reshape(-1)],
                        "view_to_world": [float(x) for x in np.linalg.inv(w2v).reshape(-1)], "alpha_upscaling": 0,
                        "presets": [0] * 6},
            "constants": {"version": 2, "view_to_clip": [float(x) for x in v2c.reshape(-1)],
                          "clip_to_view": [0.0] * 16, "clip_to_lens_clip": [0.0] * 16, "clip_to_prev_clip": [0.0] * 16,
                          "prev_clip_to_clip": [0.0] * 16, "jitter": [float(np.float32(j)) for j in jit],
                          "mv_scale": [1.0, 1.0], "pinhole_offset": [0.0, 0.0], "camera_pos": [0.0] * 3,
                          "camera_up": [0.0, 0.0, 1.0], "camera_right": [1.0, 0.0, 0.0], "camera_fwd": [0.0, 1.0, 0.0],
                          "near_plane": 0.1, "far_plane": float("inf"), "fov_y": 1.0, "aspect": rw / rh,
                          "mv_invalid_value": 0.0, "depth_inverted": 1, "camera_motion_included": 1, "mv_3d": 0,
                          "reset": 0, "orthographic": 0, "mv_dilated": 0, "mv_jittered": 0,
                          "min_depth_separation": 0.0},
            "inputs": inputs, "not_copied": [],
        }
        meta["camera"] = {"jitter": meta["constants"]["jitter"], "view_to_clip": meta["constants"]["view_to_clip"],
                          "world_to_view": meta["options"]["world_to_view"],
                          "view_to_world": meta["options"]["view_to_world"], "mv_scale": [1.0, 1.0], "reset": 0}
        with open(os.path.join(d, "frame.json"), "w") as fh:
            json.dump(meta, fh)
        truth.append(dict(colour=c4.astype(np.float16).astype(np.float32), depth=depth, dif=dif, hit=hit))
    return truth


def parse_frame_file(path):
    """Independent reader of the test host's frame input (RRF2)."""
    with open(path, "rb") as fh:
        b = fh.read()
    magic, rw, rh, reset, settings, lin = struct.unpack_from("<6I", b, 0)
    jit = struct.unpack_from("<2f", b, 24)
    ms = struct.unpack_from("<2f", b, 32)
    v2c = struct.unpack_from("<16f", b, 40)
    w2v = struct.unpack_from("<16f", b, 104)
    lox, loy, n = struct.unpack_from("<3I", b, 168)
    pos, roles = 180, {}
    bpp = {2: 16, 10: 8, 16: 8, 41: 4, 28: 4, 29: 4, 24: 4, 26: 4, 34: 4, 87: 4, 54: 2, 49: 2, 61: 1}
    for _ in range(n):
        role, fmt, w, h, ox, oy, state, conv = struct.unpack_from("<8I", b, pos)
        sc, bias = struct.unpack_from("<2f", b, pos + 32)
        size = w * h * bpp[fmt]
        roles[role] = dict(format=fmt, w=w, h=h, ox=ox, oy=oy, state=state, conv=conv, scale=sc, bias=bias,
                           data=b[pos + 40:pos + 40 + size])
        pos += 40 + size
    assert pos == len(b), (pos, len(b))
    return dict(magic=magic, render=(rw, rh), reset=reset, settings=settings, linear=lin, jitter=jit, mv_scale=ms,
                v2c=v2c, w2v=w2v, layer_origin=(lox, loy), roles=roles)


def warp_capture_dir():
    b = os.environ.get("WG_RR_PROXY_BUILD")
    if not b:
        return None
    p = os.path.join(b, "2.7", "hook-probe", "stage", "wronggpu-rr-capture")
    return p if os.path.isdir(p) else None


def trunc_half(x):
    """binary32 -> binary16 by mantissa truncation, as test_hook_dll's to_half (normal, positive values only)."""
    b = np.asarray(x, np.float32).view(np.uint32)
    e = ((b >> 23) & 0xFF).astype(np.int64) - 127 + 15
    h = ((e << 10) | ((b & 0x7FFFFF) >> 13)).astype(np.uint16)
    return h.view(np.float16)


# ------------------------------------------------------------------------------------------------- CPU tests
class Formats(unittest.TestCase):
    def roundtrip(self, fmt, texels, expect, depth=False, pad=0):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.raw")
            write_input(p, 3, fmt, texels, 0x8, pad=pad)
            h, data = R.read_input_file(p)
            v = R.decode(h, data, depth=depth)
            np.testing.assert_array_equal(v, np.asarray(expect, np.float32))
            return h, data

    def test_decoders(self):
        f16 = np.array([[[1.5, -2.0, 0.25, 65504.0], [0.0, np.inf, -0.0, 3.0]]], np.float16)
        self.roundtrip(10, as_bytes(f16, np.float16), f16.astype(np.float32), pad=16)
        self.roundtrip(9, as_bytes(f16, np.float16), f16.astype(np.float32))           # typeless: float member
        u8 = np.array([[[0, 51, 255, 128]]], np.uint8)
        self.roundtrip(28, u8, u8 / np.float32(255))
        self.roundtrip(61, u8[..., :1], u8[..., :1] / np.float32(255))
        f32 = np.array([[[0.75], [1e-30]]], np.float32)
        self.roundtrip(39, as_bytes(f32, np.float32), f32, depth=True)                  # R32 typeless depth = float
        self.roundtrip(41, as_bytes(f32, np.float32), f32)
        r16 = np.array([[[0.5], [-1.25]]], np.float16)
        self.roundtrip(54, as_bytes(r16, np.float16), r16.astype(np.float32))
        u16 = np.array([[[0], [65535], [32768]]], np.uint16)
        self.roundtrip(55, as_bytes(u16, np.uint16), u16 / np.float32(65535), depth=True)
        d24 = np.array([[[0xFF000000 | 0xFFFFFF], [0x00800000]]], np.uint32)
        exp = (np.array([[[0xFFFFFF], [0x800000]]], np.float64) / 16777215.0).astype(np.float32)
        self.roundtrip(45, as_bytes(d24, np.uint32), exp, depth=True)
        p10 = np.array([[1023 | (512 << 10) | (0 << 20) | (3 << 30)]], np.uint32)
        exp = np.array([[[1.0, 512 / 1023, 0.0, 1.0]]], np.float32)
        self.roundtrip(24, as_bytes(p10[..., None], np.uint32), exp)
        # R11G11B10: 1.0 = exponent 15, mantissa 0 in each field
        one = (15 << 6) | ((15 << 6) << 11) | ((15 << 5) << 22)
        self.roundtrip(26, as_bytes(np.array([[[one]]], np.uint32), np.uint32), np.ones((1, 1, 3), np.float32))

    def test_host_formats(self):
        f32 = np.array([[[0.75], [0.125]]], np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "z.raw")
            write_input(p, 0, 39, as_bytes(f32, np.float32), 0xC0)
            h, data = R.read_input_file(p)
            fmt, b = R.for_host(h, data, depth=True)
            self.assertEqual(fmt, 41)
            self.assertEqual(b, as_bytes(f32, np.float32).tobytes())                    # the same bits
            write_input(p, 3, 11, as_bytes(np.array([[[0, 65535, 0, 65535]]], np.uint16), np.uint16), 0x8)
            h, data = R.read_input_file(p)
            fmt, b = R.for_host(h, data)                                                 # RGBA16 unorm -> RGBA32F
            self.assertEqual(fmt, 2)
            np.testing.assert_array_equal(np.frombuffer(b, np.float32), [0, 1, 0, 1])
            write_input(p, 7, 27, np.array([[[1, 2, 3, 4]]], np.uint8), 0x8)            # typeless RGBA8 -> unorm
            h, data = R.read_input_file(p)
            self.assertEqual(R.for_host(h, data), (28, bytes([1, 2, 3, 4])))


class ProxyPart(unittest.TestCase):
    def test_load_route_create_frame_file(self):
        with tempfile.TemporaryDirectory() as d:
            truth = synthetic_capture(d, 3, 48, 27, 72, 40, start=10)
            os.makedirs(os.path.join(d, "frame_000099"))          # incomplete (no frame.json): ignored
            frames = R.load_capture(d)
            self.assertEqual([f["meta"]["evaluate"] for f in frames], [10, 11, 12])
            uses, lin, lo = R.route(frames[0])
            self.assertEqual(uses, [(0, 3), (2, 0), (3, 1), (4, 7), (5, 8), (6, 14), (9, 42), (11, 58), (14, 39)])
            self.assertFalse(lin)
            self.assertEqual(lo, (0, 0))
            c = R.creation(frames[0])
            self.assertEqual(c, dict(output=(72, 40), render=(48, 27), packing=1, hdr=1, depth_inverted=1,
                                     mv_jittered=0, hw_depth=1, alpha=0))
            self.assertEqual(R.checks(frames[0], c), [])
            p = os.path.join(d, "f.bin")
            R.frame_file(frames[1], p)
            ff = parse_frame_file(p)
            self.assertEqual(ff["magic"], 0x32465252)
            self.assertEqual(ff["render"], (48, 27))
            self.assertEqual((ff["reset"], ff["linear"], ff["layer_origin"]), (0, 0, (0, 0)))
            self.assertEqual(ff["mv_scale"], (1.0, 1.0))
            self.assertAlmostEqual(ff["jitter"][0], halton(2, 2) - 0.5, places=6)
            self.assertEqual(ff["v2c"], tuple(np.float32(frames[1]["meta"]["constants"]["view_to_clip"])))
            self.assertEqual(ff["w2v"][12], np.float32(-0.015))
            self.assertEqual(sorted(ff["roles"]), [0, 2, 3, 4, 5, 6, 9, 11, 14])
            z = ff["roles"][2]
            self.assertEqual((z["format"], z["w"], z["h"], z["state"]), (41, 48, 27, 0xC0))
            np.testing.assert_array_equal(np.frombuffer(z["data"], np.float32).reshape(27, 48), truth[1]["depth"])
            col = ff["roles"][0]
            self.assertEqual(col["format"], 10)
            np.testing.assert_array_equal(np.frombuffer(col["data"], np.float16).reshape(27, 48, 4).astype(np.float32),
                                          truth[1]["colour"])                     # the row padding dropped
            self.assertEqual(ff["roles"][4]["format"], 28)
            self.assertEqual(ff["roles"][9]["format"], 41)
            self.assertEqual(ff["roles"][11]["format"], 54)
            # a no-GPU run writes every frame file and the job
            out = os.path.join(d, "out")
            s = R.run(d, None, out, gpu=False, previews=True, log=lambda *_: None)
            self.assertEqual(s["evaluates"], [10, 11, 12])
            with open(os.path.join(out, "work", "job.txt")) as fh:
                job = fh.read().splitlines()
            self.assertEqual(job[3], "create 72 40 48 27 1 1 1 0 1 0")
            self.assertEqual(sum(1 for l in job if l.startswith("frame ")), 3)
            self.assertTrue(os.path.isfile(os.path.join(out, "frame_000010_in00_depth.png")))

    def test_missing_required(self):
        with tempfile.TemporaryDirectory() as d:
            synthetic_capture(d, 1, 16, 9, 24, 14)
            os.remove(os.path.join(d, "frame_000300", "input_07.raw"))
            with open(os.path.join(d, "frame_000300", "frame.json")) as fh:
                m = json.load(fh)
            m["inputs"] = [e for e in m["inputs"] if e["type"] != 7]
            with open(os.path.join(d, "frame_000300", "frame.json"), "w") as fh:
                json.dump(m, fh)
            frames = R.load_capture(d)
            with self.assertRaises(R.CaptureError):
                R.route(frames[0])


class Conventions(unittest.TestCase):
    """The replay's convention switches (quality investigation): defaults = the release proxy's frame file, bit for
    bit; each switch changes exactly its own part."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.truth = synthetic_capture(self.d.name, 2, 48, 27, 72, 40, start=10)
        self.frames = R.load_capture(self.d.name)

    def tearDown(self):
        self.d.cleanup()

    def ff(self, **kw):
        p = os.path.join(self.d.name, "f.bin")
        R.frame_file(self.frames[1], p, conv=R.conventions(**kw))
        with open(p, "rb") as fh:
            raw = fh.read()
        return parse_frame_file(p), raw

    def test_defaults_identical(self):
        p = os.path.join(self.d.name, "g.bin")
        R.frame_file(self.frames[1], p)
        with open(p, "rb") as fh:
            ref = fh.read()
        self.assertEqual(self.ff()[1], ref)
        self.assertEqual(R.changed(R.conventions()), [])
        self.assertEqual(R.creation(self.frames[0], R.conventions()), R.creation(self.frames[0]))

    def test_routing_switches(self):
        roles = lambda **kw: sorted(self.ff(**kw)[0]["roles"])
        self.assertNotIn(R.ROLE["LAYER"], roles(layer="drop"))
        self.assertNotIn(R.ROLE["G1"], roles(g1="drop"))
        self.assertNotIn(R.ROLE["SPECULAR_HIT_DIST"], roles(hitdist="drop"))
        f, _ = self.ff(albedo="swap")
        self.assertEqual(f["roles"][R.ROLE["DIFFUSE_ALBEDO"]]["data"], self.ff()[0]["roles"][R.ROLE["SPECULAR_ALBEDO"]]["data"])
        with self.assertRaises(R.CaptureError):
            R.conventions(layer="sideways")

    def test_value_switches(self):
        base = self.ff()[0]
        f, _ = self.ff(layer="additive", colour_scale=2.0)
        lay = np.frombuffer(f["roles"][R.ROLE["LAYER"]]["data"], np.float32).reshape(27, 48, 4)
        self.assertEqual(f["roles"][R.ROLE["LAYER"]]["format"], 2)
        np.testing.assert_array_equal(lay[..., 3], 0)
        np.testing.assert_array_equal(lay[..., :3], 2 * self.truth[1]["colour"][..., :3])
        col = np.frombuffer(f["roles"][R.ROLE["COLOUR"]]["data"], np.float32).reshape(27, 48, 4)
        np.testing.assert_array_equal(col[..., :3], 2 * self.truth[1]["colour"][..., :3])
        f, _ = self.ff(jitter_sign="-1,1", mv_scale="0.5,-1")
        self.assertEqual(f["jitter"], (np.float32(-base["jitter"][0]), np.float32(base["jitter"][1])))
        self.assertEqual(f["mv_scale"], (0.5, -1.0))
        # view-space normals: [x y z 0] W2V; roughness kept unless switched
        w2v = np.array(self.frames[1]["meta"]["options"]["world_to_view"], np.float64).reshape(4, 4)
        w2v[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        self.frames[1]["meta"]["options"]["world_to_view"] = list(w2v.reshape(-1))
        f, _ = self.ff(normals="view", roughness="invert")
        n = np.frombuffer(f["roles"][R.ROLE["NORMALS"]]["data"], np.float32).reshape(27, 48, 4)
        n0 = np.frombuffer(base["roles"][R.ROLE["NORMALS"]]["data"], np.float16).reshape(27, 48, 4).astype(np.float32)
        np.testing.assert_array_equal(n[..., 0], n0[..., 1])          # world y -> view x with this matrix
        np.testing.assert_array_equal(n[..., 2], n0[..., 0])
        np.testing.assert_allclose(n[..., 3], 1 - n0[..., 3], rtol=0, atol=1e-7)
        f, _ = self.ff(albedo="srgb", hitdist=0.5)
        a = np.frombuffer(f["roles"][R.ROLE["DIFFUSE_ALBEDO"]]["data"], np.float32).reshape(27, 48, 4)
        np.testing.assert_allclose(a[..., :3], R.srgb_to_linear(np.rint(self.truth[1]["dif"][..., :3] * 255) / 255),
                                   rtol=0, atol=1e-6)
        hd = np.frombuffer(f["roles"][R.ROLE["SPECULAR_HIT_DIST"]]["data"], np.float32).reshape(27, 48)
        np.testing.assert_array_equal(hd, 0.5 * self.truth[1]["hit"])
        c = R.creation(self.frames[0], R.conventions(depth_inverted="0", mv_jittered="1"))
        self.assertEqual((c["depth_inverted"], c["mv_jittered"]), (0, 1))

    def test_diagnostic_probes(self):
        """The diagnostic probes (not conventions) replace exactly the channels they name."""
        role = lambda f, r, c: np.frombuffer(f["roles"][R.ROLE[r]]["data"], np.float32).reshape(27, 48, c)
        col = self.truth[1]["colour"][..., :3]
        f, _ = self.ff(colour_probe="rotate")
        np.testing.assert_array_equal(role(f, "COLOUR", 4)[..., :3], col[..., [1, 2, 0]])
        f, _ = self.ff(colour_probe="zero_b")
        c = role(f, "COLOUR", 4)
        np.testing.assert_array_equal(c[..., 2], 0)
        np.testing.assert_array_equal(c[..., :2], col[..., :2])
        f, _ = self.ff(albedo="grey")
        a = role(f, "DIFFUSE_ALBEDO", 4)
        np.testing.assert_array_equal(a[..., 0], a[..., 1])
        np.testing.assert_array_equal(a[..., 0], a[..., 2])
        f, _ = self.ff(albedo="one")
        np.testing.assert_array_equal(role(f, "SPECULAR_ALBEDO", 4)[..., :3], 1)
        n0 = np.frombuffer(self.ff()[0]["roles"][R.ROLE["NORMALS"]]["data"], np.float16).reshape(27, 48, 4)
        f, _ = self.ff(normals="flat", roughness="half")
        n = role(f, "NORMALS", 4)
        np.testing.assert_array_equal(n[..., :3], np.broadcast_to(np.float32([0, 0, 1]), n[..., :3].shape))
        np.testing.assert_array_equal(n[..., 3], 0.5)
        f, _ = self.ff(normals="zero")
        n = role(f, "NORMALS", 4)
        np.testing.assert_array_equal(n[..., :3], 0)
        np.testing.assert_array_equal(n[..., 3], n0[..., 3].astype(np.float32))

    def test_crop_rect(self):
        """Both halves of <n>_crop.png come from one rectangle; the requested one is clipped to the output."""
        self.assertEqual(R.crop_rect(2560, 1440), (640, 360, 1280, 720))
        self.assertEqual(R.crop_rect(2560, 1440, (2500, -5, 200, 100)), (2500, 0, 60, 100))
        with tempfile.TemporaryDirectory() as d:
            synthetic_capture(d, 1, 48, 27, 72, 40)
            fr = R.load_capture(d)[0]
            cin = R.colour_input(fr)
            near = R.tonemap(R.nearest(cin, 72, 40))
            for crop in (None, (10, 5, 30, 20)):
                x, y, w, h = R.crop_rect(72, 40, crop)
                # an output equal to the nearest-upscaled input gives two identical halves of the crop rectangle
                img = R.crop_image(cin, near, 1.0, crop)
                self.assertEqual(img.shape, (h, 2 * w + 4, 3))
                np.testing.assert_array_equal(img[:, :w], img[:, w + 4:])
                np.testing.assert_array_equal(img[:, :w], near[y:y + h, x:x + w])


class Images(unittest.TestCase):
    def test_tonemap(self):
        t = R.tonemap(np.array([[[0.0, 1.0, 3.0], [np.nan, 0.0, 0.0], [-1.0, 1e9, np.inf]]], np.float32))
        self.assertEqual(t[0, 0].tolist(), [0, 188, 225])       # 1 -> 0.5 -> sRGB 0.7354; 3 -> 0.75 -> 0.8808
        self.assertEqual(t[0, 1].tolist(), [255, 0, 255])
        self.assertEqual(t[0, 2].tolist(), [255, 0, 255])
        self.assertEqual(R.tonemap(np.array([[[-1.0, 1e9, 0.0]]]))[0, 0].tolist(), [0, 255, 0])

    def test_png_and_resampling(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            R.write_png(p, img)
            np.testing.assert_array_equal(R.read_png(p), img)
        f = rng.random((6, 4, 3)).astype(np.float32)
        np.testing.assert_allclose(R.bilinear(f, 4, 6), f, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(R.nearest(f, 8, 12)[::2, ::2], f)
        self.assertTrue(np.allclose(R.bilinear(np.full((3, 3, 3), 2.0, np.float32), 9, 7), 2.0))
        self.assertEqual(R.side_by_side(img, img).shape, (7, 14, 3))


class WarpCapture(unittest.TestCase):
    """The capture the probe ASI wrote in test_hook_dll (WARP): decoded against the uploaded patterns."""

    def test_warp(self):
        d = warp_capture_dir()
        if not d:
            self.skipTest("no test_hook_dll capture (WG_RR_PROXY_BUILD unset or not built yet)")
        check_warp_capture(self, d)


def check_warp_capture(tc, d):
    frames = R.load_capture(d)
    tc.assertEqual([f["meta"]["evaluate"] for f in frames], [4, 6, 8])
    i = np.arange(40 * 24 * 4)
    colour = trunc_half((i % 977).astype(np.float32) / np.float32(64) + np.float32(0.125)).astype(np.float32)
    colour = colour.reshape(24, 40, 4)
    depth = np.array([np.float32(1.0) / (np.float32(1.0) + np.float32(k)) for k in range(32 * 18)], np.float32)
    alb = np.array([(k * 7 + 3) & 255 for k in range(32 * 18 * 4)], np.uint8).reshape(18, 32, 4)
    for f in frames:
        h, data = f["inputs"][3]
        np.testing.assert_array_equal(R.decode(h, data), colour)
        h, data = f["inputs"][0]
        tc.assertEqual(h["format"], 39)
        np.testing.assert_array_equal(R.decode(h, data, depth=True).reshape(-1), depth)
        h, data = f["inputs"][7]
        np.testing.assert_array_equal(R.decode(h, data), alb / np.float32(255))
        tc.assertEqual(R.input_rect(f["inputs"][3][0]), (4, 2, 32, 18))
    c = R.creation(frames[0])
    tc.assertEqual((c["output"], c["render"], c["packing"], c["depth_inverted"]), ((64, 36), (32, 18), 1, 1))
    uses, _, _ = R.route(frames[0])
    tc.assertEqual(uses, [(0, 3), (2, 0), (3, 1), (4, 7), (5, 8), (6, 14)])


# ------------------------------------------------------------------------------------------------- GPU end to end
def gpu_selftest(out, model, warp=None):
    os.makedirs(out, exist_ok=True)
    report = []
    cases = [("synthetic-192x108", 8, 192, 108, 288, 162), ("synthetic-1440p", 3, 1706, 960, 2560, 1440)]
    ok = True
    for name, n, rw, rh, wo, ho in cases:
        cap = os.path.join(out, name, "capture")
        synthetic_capture(cap, n, rw, rh, wo, ho)
        s = R.run(cap, model, os.path.join(out, name, "replay"), previews=(name == "synthetic-192x108"))
        good = check_replay(s, os.path.join(out, name, "replay"), n, wo, ho, rw, rh)
        ok = ok and good
        report.append((name, s, good))
    if warp:
        s = R.run(warp, model, os.path.join(out, "warp-capture", "replay"), previews=True)
        good = check_replay(s, os.path.join(out, "warp-capture", "replay"), 3, 64, 36, 32, 18)
        ok = ok and good
        report.append(("warp-capture", s, good))
    lines = []
    for name, s, good in report:
        fr = s["frames"]
        lines.append(f"{name}: {'PASS' if good else 'FAIL'}: {len(fr)} frames, rc {sorted(set(x['rc'] for x in fr))}, "
                     f"completed {sum(x['completed'] for x in fr)}/{len(fr)}, finite "
                     f"{min(x['output_finite_fraction'] for x in fr):.4f}..{max(x['output_finite_fraction'] for x in fr):.4f}")
    with open(os.path.join(out, "selftest.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return ok


def check_replay(s, out, n, wo, ho, rw, rh):
    good = len(s["frames"]) == n and all(x["rc"] == 0 and x["completed"] == 1 for x in s["frames"])
    for x in s["frames"]:
        e = x["evaluate"]
        for suffix, shape in (("input", (rh, rw, 3)), ("rr", (ho, wo, 3)), ("side", (ho, 2 * wo + 4, 3)),
                              ("crop", (ho // 2, 2 * (wo // 2) + 4, 3))):
            p = os.path.join(out, f"frame_{e:06d}_{suffix}.png")
            if not os.path.isfile(p) or R.read_png(p).shape != shape:
                print(f"bad or missing {p}")
                good = False
    good = good and os.path.isfile(os.path.join(out, "summary.json"))
    return good


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.environ.get("WG_BUILD_ROOT", "build"), "rr-f-capture-out", "selftest"))
    ap.add_argument("--model", default=SYNTH_MODEL)
    ap.add_argument("--warp", help="a capture folder written by test_hook_dll (WARP)")
    a, rest = ap.parse_known_args()
    if a.warp:
        os.environ["WG_RR_PROXY_BUILD"] = ""
    r = unittest.main(argv=[sys.argv[0]] + rest, exit=False, verbosity=2)
    ok = r.result.wasSuccessful()
    if a.warp:
        tc = unittest.TestCase()
        check_warp_capture(tc, a.warp)
        print("WARP capture decoded: OK")
    if a.gpu:
        ok = gpu_selftest(a.out, a.model, a.warp or warp_capture_dir()) and ok
    sys.exit(0 if ok else 1)
