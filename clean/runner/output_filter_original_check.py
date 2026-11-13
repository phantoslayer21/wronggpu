# SPDX-License-Identifier: MIT
"""Whole-frame GPU check of the output filter "original" (spec 06 draft 4; runtime option output_filter): synthetic
Model M parameters that carry made-up output-filter weights (clean/runner/synthetic_model.py --output-taps), synthetic
frames (resets, render-size changes, both exposure paths, pre-exposure).

1. Selection: no option = "original" when the model has the weights, else "analytic" plus a note; `original` without
   weights and unknown values are refused; wave32 post kernels (no `original` build) fall back with a note.
2. State: every stage output and state image of an `original` frame is bit-identical to the `analytic` run of the same
   frames (only the final resampling differs).
3. Output against clean/reference/post.py output_filter_original, on the display image of an unfused run of the same
   frames (the fused kernels never write it): the filtered value q is recovered from the output
   (q = 0.36 log10(1 + 100 out e / pi)) and must equal the reference's FP16 value within 1e-4 relative; two
   neighbouring FP16 values are at least 4.9e-4 apart, so this is "the same FP16 value" for every channel of every
   pixel. As a control the same test against the same weights WITHOUT the FP16 steps must fail on many values.
4. Every other path of the `original` build gives the same output bits as the default one (true fusion): the frame
   without stage outputs, the banded two-kernel sequence, other true-fusion kernels, the unbanded filters, the fused
   kernels; RGBA16F forms (the _o16 kernels, and the interop's output copy through wg_rt_frame_io: RGBA16F rows with
   pi folded in) equal the FP16 rounding of that output.
5. With --baseline-build (a build of the runtime from before this change, same code objects): output_filter=analytic,
   and no option on a model without weights, give the same bits as that build for every stage output and the output;
   likewise quality5 and quality9.
6. With --large: frames at 1706x960 -> 2560x1440 against the reference (test 3).

No learned model data is used. Run under the shared GPU lock:

    python the GPU lock (WG_GPU_LOCK, BUILDING.md) -- python -m clean.runner.output_filter_original_check --build <dir> --rocm <ROCm>
        --models <empty work dir> --json <report.json> [--baseline-build <dir>] [--large]
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from clean.reference import encoder
from clean.reference import post as P
from clean.runner import synthetic_model as SM
from clean.runner.pipeline import Pipeline

OUT = (127, 73)
RENDERS = ((80, 49), (80, 49), (64, 40), (127, 73), (63, 37))
SEED = 61013
Q_TOL = 1e-4          # relative; neighbouring FP16 values differ by 4.9e-4 ... 9.8e-4
# Named buffers that the default kernels do not write (they hold whatever the device memory held before): the
# bottleneck megakernel swin_bneck6 writes b5 only, and the fused dec1 + dec0 keeps dec1's map on the chip. (The
# display image of a frame whose post kernel filters itself is not written either: see display_written.)
STALE = ("b0", "b1", "b2", "b3", "b4", "dec1")
# The previous-frame state on a reset frame (the first frame, a game reset, a render-size change): not read by the
# frame, and after a size change partly never written at the new size.
ON_RESET = ("state.latent_prev", "state.motion_history", "state.history_prev")
up, vp = ctypes.c_size_t, ctypes.c_void_p


def frame_input(index, render, pi=None, reset=None, seed=SEED):
    rw, rh = render
    rng = np.random.default_rng([seed, index])
    s = encoder.random_inputs(rng, rw=rw, rh=rh, fw=((rw + 31) // 32) * 32, fh=((rh + 31) // 32) * 32)
    if pi is None:
        pi = 1.0 if index % 3 else 1.375
    return {"color": s["color"], "depth": s["depth"], "motion": s["motion"], "mv_scale": s["mv_scale"],
            "jitter": s["jitter"], "exposure": 0.75 + (index % 24) / 24, "pre_exposure": pi,
            "reset": index in (0, 5, 12) if reset is None else reset}


def stream(n, pi=None):
    return [frame_input(i, RENDERS[i % len(RENDERS)], pi) for i in range(n)]


def make(args, model, auto, max_render=OUT, out=OUT, **kernels):
    return Pipeline(args.build, args.rocm, str(model), max_render, out,
                    {"depth_inverted": False, "auto_exposure": auto}, model_format="raw", check_packing=False,
                    kernels=kernels or None)


def close(*pipelines):
    for p in pipelines:
        if p is not None and getattr(p, "rt", None):
            p.lib.wg_rt_destroy(p.rt)
            p.rt = None


def compare_tree(a, b, skip=("out", "filter", "post.display"), prefix="", reset=None):
    """Bit identity of two debug trees (every stage output and state image of a frame, apart from `skip` and STALE);
    returns the number of values compared."""
    values = 0
    if reset is None:
        reset = bool(a["state"]["info"]["reset"])
    for name in a:
        path = prefix + name
        if path in skip or path in STALE or (reset and path in ON_RESET):
            continue
        x, y = a[name], b[name]
        if isinstance(x, dict):
            values += compare_tree(x, y, skip, path + ".", reset)
        elif isinstance(x, np.ndarray):
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                raise AssertionError("stage output or state differs: " + path)
            values += x.size
        elif x is None or y is None:
            if x is not y:
                raise AssertionError("stage output present in one run only: " + path)
        elif x != y:
            raise AssertionError("frame metadata differs: " + path)
    return values


def display_written(pipeline):
    """The last frame wrote the display image (its post kernel did not run the filter itself)."""
    return pipeline.kernels()["filter_last"] != "fused"


def recovered_q(out, e, pi):
    """The filtered value behind an output value: Dec inverted (float64)."""
    v = out.astype(np.float64) * (float(e) / float(pi))
    return P.E_SCALE * np.log1p(100.0 * v) / np.log(10.0)


def unrounded_filter(display, output_size, taps):
    """Control: the same FP16 weights and clamp, but float64 sums without the FP16 rounding of each step."""
    w16 = P.filter_taps_f16(taps)

    def one(img, n_out):
        x = np.arange(n_out)
        idx = np.clip(P.filter_center(x)[:, None] + np.arange(-2, 3)[None, :], 0, img.shape[0] - 1)
        t = img[idx]
        w = np.where((x & 1)[:, None] == 0, w16[None, :], w16[None, ::-1]).reshape((n_out, 5) + (1,) * (img.ndim - 1))
        return P.clamp((t * w).sum(axis=1), t.min(axis=1), t.max(axis=1))

    ow, oh = output_size
    img = np.asarray(display, np.float64)
    return one(one(img.transpose(1, 0, 2), ow).transpose(1, 0, 2), oh)


def check_reference(out, display, e, pi, taps, output_size, control=True):
    """(values, mismatches, max relative error of q, control mismatches) of one f32 output frame."""
    if not np.isfinite(out).all():
        raise AssertionError("non-finite output")
    qg = recovered_q(out, e, pi)
    values = mism = ctrl = 0
    worst = 0.0
    for c in range(3):   # per channel: bounded memory at 1440p
        q = P.encoded_filter_original(display[..., c:c + 1], output_size, taps).astype(np.float64)[..., 0]
        qc = np.clip(q, P.DEC_LO, P.DEC_HI)
        rel = np.abs(qg[..., c] - qc) / qc
        values += rel.size
        mism += int((rel > Q_TOL).sum())
        worst = max(worst, float(rel.max()))
        if control:
            qu = np.clip(unrounded_filter(display[..., c:c + 1], output_size, taps)[..., 0], P.DEC_LO, P.DEC_HI)
            ctrl += int((np.abs(qg[..., c] - qu) / qu > Q_TOL).sum())
    return values, mism, worst, ctrl


def half_of(out):
    """What the RGBA16F forms hold: the f32 output saturated to FP16's finite range and rounded to nearest even."""
    return np.clip(out, -65504.0, 65504.0).astype(np.float16)


class WgRtExtIo(ctypes.Structure):
    _fields_ = [("color", up), ("depth", up), ("motion", up), ("out", up), ("expo_src", up),
                ("exposure_mode", ctypes.c_int), ("raw_inputs", ctypes.c_int), ("color_pitch", ctypes.c_int),
                ("depth_pitch", ctypes.c_int), ("motion_pitch", ctypes.c_int), ("motion_channels", ctypes.c_int),
                ("ms", ctypes.c_float * 2), ("mo", ctypes.c_float * 2), ("out_half_pitch", ctypes.c_int)]


class OutputCopy:
    """A second runtime with external buffers (the path of the D3D12 interop's HIP executor) whose output is the
    interop's output copy: RGBA16F rows of `pitch` bytes with pi folded in (wg_rt_frame_io, out_half_pitch)."""

    def __init__(self, ref, pitch):
        lib = self.lib = ref.lib
        lib.wg_rt_create_external.argtypes = [vp, ctypes.POINTER(vp), ctypes.c_char_p, ctypes.c_int]
        lib.wg_rt_alloc.argtypes = [vp, up, ctypes.POINTER(up)]
        lib.wg_rt_upload.argtypes = [vp, up, vp, up]
        lib.wg_rt_frame_io.argtypes = [vp, vp, ctypes.POINTER(WgRtExtIo)]
        self.ref, self.pitch = ref, int(pitch)
        rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(2048)
        if lib.wg_rt_create_external(ctypes.byref(ref._cfg), ctypes.byref(rt), err, len(err)) != 0:
            raise RuntimeError(err.value.decode())
        self.rt = rt
        R = ref.max_rw * ref.max_rh
        self.bufs = {k: self.alloc(n) for k, n in {"color": R * 16, "depth": R * 4, "motion": R * 8,
                                                   "out": self.pitch * ref.oh, "expo": 16}.items()}

    def alloc(self, n):
        p = up()
        if self.lib.wg_rt_alloc(self.rt, n, ctypes.byref(p)) != 0:
            raise RuntimeError(self.lib.wg_rt_error(self.rt).decode())
        return p.value

    def upload(self, name, a):
        a = np.ascontiguousarray(a)
        if self.lib.wg_rt_upload(self.rt, self.bufs[name], a.ctypes.data, a.nbytes) != 0:
            raise RuntimeError(self.lib.wg_rt_error(self.rt).decode())

    def frame(self, x):
        rh, rw = x["color"].shape[:2]
        rgba = np.zeros((rh, rw, 4), np.float32)
        rgba[..., :3] = x["color"]
        self.upload("color", rgba)
        self.upload("depth", np.asarray(x["depth"], np.float32))
        self.upload("motion", np.asarray(x["motion"], np.float32))
        fr = self.ref._upload(x)   # builds the WgRtFrame (the upload into the reference runtime is not used)
        b = self.bufs
        io = WgRtExtIo(color=b["color"], depth=b["depth"], motion=b["motion"], out=b["out"], expo_src=b["expo"],
                       exposure_mode=-1, raw_inputs=0, out_half_pitch=self.pitch)
        if self.lib.wg_rt_frame_io(self.rt, ctypes.addressof(fr), ctypes.byref(io)) != 0:
            raise RuntimeError(self.lib.wg_rt_error(self.rt).decode())
        rows = np.empty(self.pitch * self.ref.oh // 2, np.float16)
        if self.lib.wg_rt_download(self.rt, b["out"], rows.ctypes.data, rows.nbytes) != 0:
            raise RuntimeError(self.lib.wg_rt_error(self.rt).decode())
        return rows.reshape(self.ref.oh, self.pitch // 2)[:, :4 * self.ref.ow].reshape(self.ref.oh, self.ref.ow, 4)

    def kernels(self):
        return dict(kv.split("=", 1) for kv in self.lib.wg_rt_kernels(self.rt).decode().split())

    def close(self):
        if self.rt:
            self.lib.wg_rt_destroy(self.rt)
            self.rt = None
        close(self.ref)


# Other paths of the `original` build with 3 x f32 output: bit-identical to the default path.
F32_PATHS = {
    "banded (post_tf=0)": {"options": "post_tf=0"},
    "true fusion, post_diet=0": {"options": "post_diet=0"},
    "true fusion, tf_vdiet=0": {"options": "tf_vdiet=0"},
    "true fusion, post_trim=0": {"options": "post_trim=0"},
    "unbanded post_filter_c1_s4": {"post": "post_recon2_r", "filter": "post_filter_c1_s4"},
    "unbanded post_filter": {"post": "post_recon2_r", "filter": "post_filter"},
    "post_fused_h8": {"post": "post_fused_h8"},
    "post_fused3_h6": {"post": "post_fused3_h6"},
}
# RGBA16F output (pre-exposure 1 only): the FP16 rounding of the default path's output.
O16_PATHS = {
    "post_filter_c1_s4_o16": {"post": "post_recon2_r", "filter": "post_filter_c1_s4_o16"},
    "post_fused_h8_o16": {"post": "post_fused_h8_o16"},
    "post_fused3_h6_o16": {"post": "post_fused3_h6_o16"},
}
# Interop output copy (RGBA16F rows, pi folded in): kernels dict, row pitch in bytes (None: 8 ow, dense).
COPY_PATHS = {
    "output copy, true fusion": ({}, 1280),
    "output copy, banded": ({"options": "post_tf=0"}, 1280),
    "output copy, post_filter_h": ({"post": "post_recon2_r", "filter": "post_filter_c1_s4"}, 1280),
    "output copy, dense rows": ({"post": "post_recon2_r", "filter": "post_filter_c1_s4"}, None),
}


def merged(kernels, option):
    k = dict(kernels)
    k["options"] = ";".join(v for v in (k.get("options"), option) if v)
    return k


def selection_checks(args, plain_model, taps_model, taps):
    """Test 1."""
    rec = {}
    p = make(args, plain_model, False)
    k = p.kernels()
    assert k["output_filter"] == "analytic" and k["output_taps"] == "none", k
    assert len(p.notes()) == 1 and "analytic" in p.notes()[0] and "no output filter weights" in p.notes()[0], p.notes()
    assert p.output_taps() is None
    rec["no weights, no option"] = {"output_filter": k["output_filter"], "note": p.notes()[0]}
    close(p)
    for opt in ("default", "analytic"):
        p = make(args, plain_model, False, options="output_filter=" + opt)
        assert p.kernels()["output_filter"] == "analytic", p.kernels()
        assert (len(p.notes()) == 1) == (opt == "default"), p.notes()   # asked for by name: nothing to note
        close(p)
    for opt, msg in (("original", "no output filter weights"), ("lanczos", "output_filter must be")):
        try:
            close(make(args, plain_model, False, options="output_filter=" + opt))
        except RuntimeError as ex:
            assert msg in str(ex), str(ex)
            rec["no weights, output_filter=" + opt] = str(ex).replace("runtime create failed: ", "")
        else:
            raise AssertionError("output_filter=" + opt + " was accepted without weights")
    for opt, want in ((None, "original"), ("default", "original"), ("original", "original"), ("analytic", "analytic"),
                      ("quality5", "quality5"), ("quality9", "quality9")):
        p = make(args, taps_model, False, **({"options": "output_filter=" + opt} if opt else {}))
        k = p.kernels()
        assert k["output_filter"] == want and k["output_taps"] == "model" and p.notes() == [], (opt, k, p.notes())
        assert np.array_equal(p.output_taps(), taps)
        close(p)
    rec["weights"] = "no option, default, original -> original; analytic, quality5, quality9 as asked; no note"
    # wave32 post kernels (post.co): there is no `original` build of them
    p = make(args, taps_model, False, post_wave=32)
    k = p.kernels()
    assert k["output_filter"] == "analytic" and k["post_wave"] == "32" and len(p.notes()) == 1, (k, p.notes())
    rec["weights, post_wave=32"] = {"output_filter": k["output_filter"], "note": p.notes()[0]}
    close(p)
    try:
        close(make(args, taps_model, False, post_wave=32, options="output_filter=original"))
    except RuntimeError as ex:
        assert "post_w64__orig.co" in str(ex), str(ex)
        rec["weights, post_wave=32, output_filter=original"] = str(ex).replace("runtime create failed: ", "")
    else:
        raise AssertionError("output_filter=original was accepted with wave32 post kernels")
    return rec


def original_checks(args, model, taps, auto, frames):
    """Tests 2, 3 and 4 for one model and exposure path."""
    xs, xs1 = stream(frames), stream(frames, pi=1.0)
    rec = {"automatic_exposure": auto, "taps": [float(v) for v in taps], "frames": frames}
    base = make(args, model, auto, options="output_filter=analytic")
    aux = make(args, model, auto, options="output_filter=quality9")       # unfused: writes the display image
    orig, plain = make(args, model, auto), make(args, model, auto)
    outs, post_kernels = [], set()
    values = mism = ctrl = state = 0
    worst = 0.0
    try:
        for i, x in enumerate(xs):
            old, got, dis = base.frame(x, debug=True), orig.frame(x, debug=True), aux.frame(x, debug=True)
            k = orig.kernels()
            assert k["output_filter"] == "original", k
            post_kernels.add(k["post_last"] + " / " + k["filter_last"])
            state += compare_tree(old, got)                # test 2
            compare_tree(old, dis)                         # the display's run has the same state too
            n, m, w, c = check_reference(got["out"], dis["post"]["display"], got["exposure"]["e"], x["pre_exposure"],
                                         taps, OUT)                      # test 3
            values, mism, ctrl, worst = values + n, mism + m, ctrl + c, max(worst, w)
            if plain.frame(x)["out"].tobytes() != got["out"].tobytes():
                raise AssertionError(("output differs between the frame with and without stage outputs", i))
            outs.append(got["out"])
    finally:
        close(base, aux, plain)
    if not any(s.startswith("post_tf") for s in post_kernels):
        raise AssertionError("the default path did not run a true-fusion kernel: " + str(sorted(post_kernels)))
    if mism:
        raise AssertionError(f"{mism} of {values} output values are not the reference's FP16 value (max rel {worst})")
    if ctrl < values // 100:
        raise AssertionError(f"control: only {ctrl} of {values} values tell the FP16 steps from unrounded sums")
    rec.update({"state_values_bit_identical_to_analytic": state, "output_values": values,
                "values_not_the_reference_fp16_value": mism, "max_relative_error_of_q": worst,
                "control_values_off_without_fp16_steps": ctrl, "default_path_kernels": sorted(post_kernels)})
    paths = {}
    try:
        for name, kern in F32_PATHS.items():                             # test 4, 3 x f32
            p, q = make(args, model, auto, **kern), make(args, model, auto, **merged(kern, "output_filter=analytic"))
            seen = set()
            for i, x in enumerate(xs):
                got = p.frame(x, debug=True)
                try:
                    compare_tree(q.frame(x, debug=True), got)
                except AssertionError as ex:
                    raise AssertionError((name, i, p.kernels()["post_last"], str(ex))) from None
                seen.add(p.kernels()["post_last"] + " / " + p.kernels()["filter_last"])
                if got["out"].tobytes() != outs[i].tobytes():
                    raise AssertionError((name, "output differs from the default path", i,
                                          int(np.count_nonzero(got["out"] != outs[i]))))
            assert p.kernels()["output_filter"] == "original"
            paths[name] = sorted(seen)
            close(p, q)
        orig1 = make(args, model, auto)
        outs1 = [orig1.frame(x)["out"] for x in xs1]
        close(orig1)
        for name, kern in O16_PATHS.items():                             # test 4, RGBA16F
            p = make(args, model, auto, **kern)
            for i, x in enumerate(xs1):
                got = p.frame(x)["out"]
                assert p.kernels()["out"] == "rgba16f", p.kernels()
                if got.astype(np.float16).tobytes() != half_of(outs1[i]).tobytes():
                    raise AssertionError((name, "RGBA16F output is not the FP16 rounding of the default output", i))
            paths[name] = [p.kernels()["post_last"] + " / " + p.kernels()["filter_last"]]
            close(p)
        for name, (kern, pitch) in COPY_PATHS.items():                   # test 4, interop output copy
            dense = pitch is None
            c = OutputCopy(make(args, model, auto, **kern), 8 * OUT[0] if dense else pitch)
            seen = set()
            for i, x in enumerate(xs1 if dense else xs):
                got = c.frame(x)
                want = half_of((outs1 if dense else outs)[i])
                seen.add(c.kernels()["post_last"] + " / " + c.kernels()["filter_last"])
                if got[..., :3].tobytes() != want.tobytes() or not (got[..., 3] == 1).all():
                    raise AssertionError((name, "output copy is not the FP16 rounding of the default output", i,
                                          int(np.count_nonzero(got[..., :3] != want))))
            assert c.kernels()["output_filter"] == "original"
            paths[name] = sorted(seen)
            c.close()
    finally:
        close(orig)
    rec["paths_bit_identical_to_the_default_path"] = paths
    return rec


def baseline_checks(args, plain_model, taps_model, frames):
    """Test 5: the analytic path (and quality5 / quality9) against a build from before this change."""
    xs = stream(frames)
    old_args = argparse.Namespace(**{**vars(args), "build": args.baseline_build})
    cases = [("no weights: no option (-> analytic) vs old default", plain_model, {}, {}),
             ("no weights: output_filter=analytic vs old default", plain_model, {"options": "output_filter=analytic"},
              {}),
             ("weights: output_filter=analytic vs old default", taps_model, {"options": "output_filter=analytic"}, {}),
             ("weights: output_filter=analytic, post_tf=0 vs old post_tf=0", taps_model,
              {"options": "post_tf=0;output_filter=analytic"}, {"options": "post_tf=0"}),
             ("weights: analytic, unbanded post_recon2_r + post_filter_c1_s4 vs old", taps_model,
              {"post": "post_recon2_r", "filter": "post_filter_c1_s4", "options": "output_filter=analytic"},
              {"post": "post_recon2_r", "filter": "post_filter_c1_s4"}),
             ("weights: output_filter=quality5 vs old quality5", taps_model, {"options": "output_filter=quality5"},
              {"options": "output_filter=quality5"}),
             ("weights: output_filter=quality9 vs old quality9", taps_model, {"options": "output_filter=quality9"},
              {"options": "output_filter=quality9"})]
    rec = {}
    for auto in (False, True):
        for name, model, new_k, old_k in cases:
            new, old = make(args, model, auto, **new_k), make(old_args, model, auto, **old_k)
            values = 0
            seen = set()
            try:
                for x in xs:
                    a, b = old.frame(x, debug=True), new.frame(x, debug=True)
                    # everything, the output included
                    values += compare_tree(a, b, skip=() if display_written(new) else ("post.display",))
                    seen.add(new.kernels()["post_last"] + " / " + new.kernels()["filter_last"])
            finally:
                close(new, old)
            rec[f"{name} (auto exposure {auto})"] = {"values_bit_identical": values, "kernels": sorted(seen)}
    return rec


def large_check(args, model, taps, frames=3):
    """Test 6: 1706x960 -> 2560x1440."""
    R, O = (1706, 960), (2560, 1440)
    orig, aux = make(args, model, False, R, O), make(args, model, False, R, O, options="output_filter=quality9")
    values = mism = ctrl = 0
    worst, kern = 0.0, set()
    try:
        for i in range(frames):
            x = frame_input(1000 + i, R, pi=1.0, reset=i == 0)
            got = orig.frame(x)["out"]
            kern.add(orig.kernels()["post_last"] + " / " + orig.kernels()["filter_last"])
            dis = aux.frame(x, debug=True)
            n, m, w, c = check_reference(got, dis["post"]["display"], dis["exposure"]["e"], 1.0, taps, O,
                                         control=i == 0)
            values, mism, ctrl, worst = values + n, mism + m, ctrl + c, max(worst, w)
    finally:
        close(orig, aux)
    if mism:
        raise AssertionError(f"1440p: {mism} of {values} output values are not the reference's FP16 value")
    return {"render": list(R), "output": list(O), "frames": frames, "output_values": values,
            "values_not_the_reference_fp16_value": mism, "max_relative_error_of_q": worst,
            "control_values_off_without_fp16_steps_frame_0": ctrl, "kernels": sorted(kern)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--models", required=True, help="work directory for the synthetic models of this test")
    ap.add_argument("--json", required=True)
    ap.add_argument("--baseline-build", help="a runtime build from before this change (test 5)")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--large", action="store_true", help="also 1706x960 -> 2560x1440 (test 6)")
    args = ap.parse_args()
    root = Path(args.models)
    if (root / "model.wgcm").exists():
        raise ValueError("refusing to write synthetic parameters into a converted-model directory")
    # made-up weights: a seeded perturbation of the analytic set, and a hand-written set with strong negative lobes
    tap_sets = {"random": SM.output_taps("random", SEED), "lobes": SM.output_taps([-0.2, 0.3, 0.9, 0.4, -0.3]),
                "analytic values": SM.output_taps("analytic")}
    models = {}
    for name, taps in [("plain", None)] + list(tap_sets.items()):
        models[name] = root / name.replace(" ", "_")
        SM.write_raw(str(models[name]), SM.synthetic_tensors(SEED, taps=taps))
    report = {"scope": "full Model M GPU runtime, synthetic parameters, synthetic frames, made-up filter weights",
              "learned_model_or_game_qualification": False,
              "runtime_sha256": hashlib.sha256((Path(args.build) / "wronggpu_runtime.dll").read_bytes()).hexdigest()
              if os.name == "nt" else None}
    report["selection"] = selection_checks(args, models["plain"], models["random"], tap_sets["random"])
    report["original"] = []
    for name, auto in (("random", False), ("random", True), ("lobes", True), ("analytic values", False)):
        r = original_checks(args, models[name], tap_sets[name], auto, args.frames)
        r["tap_set"] = name
        report["original"].append(r)
        print(f"original, taps {name}, auto exposure {auto}: {r['output_values']} output values, "
              f"{r['values_not_the_reference_fp16_value']} off, max rel {r['max_relative_error_of_q']:.2e}; control "
              f"{r['control_values_off_without_fp16_steps']}; {len(r['paths_bit_identical_to_the_default_path'])} "
              f"other paths identical", flush=True)
    if args.baseline_build:
        report["analytic_vs_baseline_build"] = baseline_checks(args, models["plain"], models["random"], args.frames)
        print(f"baseline: {len(report['analytic_vs_baseline_build'])} comparisons bit-identical", flush=True)
    if args.large:
        report["large"] = large_check(args, models["random"], tap_sets["random"])
        print("1440p:", json.dumps(report["large"]), flush=True)
    report["passed"] = True
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in ("passed", "selection", "runtime_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
