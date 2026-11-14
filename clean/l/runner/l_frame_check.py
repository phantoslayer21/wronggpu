# SPDX-License-Identifier: MIT
"""Model L (issue #89): the runtime's Model L frame on HIP against the CPU reference, on synthetic frames, with a
synthetic Model L or (--model) the converter's model.wgcm. Model M's frame on the same harness is the control.

    python clean/l/runner/l_frame_check.py --build <dir> --rocm <ROCm> --prepare     # CPU
    python clean/l/runner/l_frame_check.py --build <dir> --rocm <ROCm> --prepare --model <model.wgcm> [--map <map>]
    python clean/l/runner/l_frame_check.py --build <dir> --rocm <ROCm> --run         # GPU, inside a GPU slot
    python clean/l/runner/l_frame_check.py --build <dir> --compare --json out.json   # CPU

<dir> first holds Model M's code objects (python -m clean.runner.build --out <dir> --rocm <ROCm> --no-runtime
--no-model-k). --prepare adds Model L's code objects (l_c32_check.compile_kernels), the runtime library with Model L
(WG_MODEL_L; build.py has no Model L row yet) and the synthetic models: Model M in raw/, Model L as L/model.wgcm with
its synthetic preset-L map (the runtime loads it through l::load_wgcm_model). With --model, Model L is that file
instead (WrongGPU.exe model convert --preset=L writes it), checked by the runtime against --map (default: the shipped
data/preset-L.v1.map.json), and the reference gets the values the file holds; real weights, synthetic frames. The model
file stays in <dir> and is never part of the repository. --run runs the frame sequence SEQUENCE
through clean/runner/pipeline.py (backend HIP, every stage output read back) for Model L (runtime option model=L) and
Model M (post_pack=none: the base coefficient format and post kernel Model L uses), and keeps the outputs in <dir>.
Each model runs twice: with the bottleneck as six block launches (swin_bneck=none), the run the stages are checked on,
since the default bottleneck megakernel writes only b5's output and leaves b0 ... b4's buffers unwritten; and with the
runtime's default kernels, whose outputs must be byte-identical to the first run's (b0 ... b4 aside, and on the first
frame the previous frame's state buffers, which no frame has written and a reset frame does not read).
Only --run uses the GPU; the HIP runtime is loaded with the environment of clean/runner/hip.py.

--compare checks every frame on the runtime's own stage inputs, as clean/tests/test_runtime.py does in the emulator,
but against the float64 references with the kernels' GELU (sigmoid):
- the frame state: frame counter, parity, reset and the Swin window offsets; the latent, motion history and full
  history handed over from the previous frame (byte for byte);
- enc0: Model L's encoder (clean/l/reference/oracle.py encode) or Model M's (clean/reference/encoder.py), features under
  Model M's per-stage rule (l_c32_check.passes), the motion surfaces at most 1 FP16 ulp and 99.5 % exact;
- the ten Swin stages (clean/reference/swin.py run_stage), each output under the per-stage rule;
- dec0: Model L's decoder (oracle decode) or Model M's (clean/reference/decoder.py) on the runtime's dec1 output: the
  A, B and latent codes and the C values, each at least 99.5 % identical (l_c32_check's end-to-end bound);
- post: Model L's reconstruction (l_post_recon) against clean/l/reference/recon.py reconstruct, Model M's against
  clean/reference/post.py reconstruct: history and display under test_post_emu.py's FP16 acceptance for Model M, and
  for Model L under Model L's post acceptance (clean/l/runner/l_post_check.py, provisional: values more than one ulp
  off at most 0.01 % over the run and 0.05 % per frame, for the hardware-log2 reason given there), the reduced
  history codes within 1 and 99 % exact; the filter (when
  the post kernel writes a display image) as test_post_emu.py, against recon.output_filter / post.output_filter;
- exposure: the runtime's value and the reference's, reported (not a pass condition).
Output: counts only, no timing. The summary gives per stage the values differing, the largest difference (FP8 steps,
FP16 ulps or codes, as the stage's metric has it), the non-finite values on each side and the largest finite magnitude
on each side (saturation shows as one side stopping where the other goes on).

Native D3D12 (runtime backend 3), after --run:

    python clean/l/runner/l_frame_check.py --build <dir> --run --backend d3d12   # GPU, inside a GPU slot
    python clean/l/runner/l_frame_check.py --build <dir> --compare-d3d12 --json out.json

<dir>/dx12 holds the D3D12 ports (python clean/dx12/build_dx12.py --out <dir>). The same frames run on D3D12 with the
runtime's default kernels there (Model L: its encoder and decoder ports, the pack12 store l_coef_pack12, Model M's
pack12 post path; Model M: its native frame), and every stage output both backends write is compared with the HIP
run's ("default"), byte for byte: the D3D12 ports reproduce their HIP kernels byte for byte, and Model M's pack12 post
path the banded one bit for bit. The pack12 images (A: A0 A1 A2 B0 | B1 B2 C2, C: C0 C1; clean/FORMATS.md) are
rearranged into the base layout's bytes and compared with the HIP run's A, B and C.

Byte identity (every "byte for byte" above): the RAW defined bytes of every output (RAW_OUTPUTS, the sizes wg_rt_buffer
reports for the frame), never decoded values. Before every frame each run fills the outputs with its own sentinel byte
(SENTINELS: HIP stages 0xC3, HIP default 0xA5, D3D12 stages 0x3C, D3D12 default 0x5A; wg_rt_fill), except the buffers
the next frame reads before it writes them (CARRIED: the reduced history and the exposure pair; and from the second
frame on the history, latent and motion surface the last frame wrote). A byte one run leaves unwritten therefore
differs, and a missing output fails. The outputs a configuration does not write by design (b0 ... b4 under the
bottleneck megakernel, the maps the D3D12 fusions keep on chip, the display image of the fused post, the previous
frame's state on the first frame) are listed per comparison with the reason (raw_compare `unwritten`), and there every
byte must still hold that run's sentinel. Negative controls (record level): one history write and one reduced write
dropped from a D3D12 run (the 4th FP16 of one history pixel, bits 30-31 of one reduced word: the bytes a decoded
comparison does not see) must each be caught.
"""
import argparse
import copy
import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_post_check as LP  # noqa: E402 (Model L's post acceptance)

L_SOURCES = [os.path.join("..", "l", "runtime", "l_model.cpp"), os.path.join("..", "l", "runtime", "l_pack.cpp")]
SEED = 89
MAX_RENDER = (96, 64)
OUTPUT = (120, 72)
# (render size, game reset): aligned (96 x 64), not aligned (80 x 48, 77 x 45), resets by the game, by a size change
# and on the first frame; both parities on every size
SEQUENCE = [((96, 64), False), ((96, 64), False), ((96, 64), True), ((96, 64), False), ((96, 64), False),
            ((80, 48), False), ((80, 48), False), ((80, 48), False), ((77, 45), False), ((77, 45), True),
            ((77, 45), False)]
PRE_EXPOSURE = 1.25
DEFAULT_L_MAP = os.path.join(ROOT, "data", "preset-L.v1.map.json")


def model_dir(build):
    return os.path.join(build, "lframe_model")


def l_map(build):
    return os.path.join(model_dir(build), "maps", "preset-L.v1.map.json")


# ---- --prepare (CPU) ------------------------------------------------------------------------------------------------

def _sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prepare(build, rocm, model=None, map_path=None):
    from clean.l.reference import oracle as L
    from clean.l.runner import l_c32_check as C32
    from clean.runner import build as B
    from clean.runner import synthetic_model as SM
    from clean.runner.hip import clean_process_environment
    from clean.runner.pipeline import logical_tensors
    sys.path.insert(0, os.path.join(ROOT, "clean", "l", "runtime", "tests"))
    import test_l_runtime_load as LL

    clean_process_environment(rocm)
    C32.compile_kernels(build, rocm)
    # the runtime library with Model L (and without Model K, as the code objects of --no-model-k)
    B.set_model("K", False)
    B.RUNTIME_SOURCES.extend(L_SOURCES)
    plain = B.runtime_command
    B.runtime_command = lambda cxx, out, embedded=None: (lambda c: c[:1] + ["-DWG_MODEL_L"] + c[1:])(
        plain(cxx, out, embedded))
    B.build_runtime(B.find_host_cxx(rocm), build, embed=False)
    # the HIP device loads "<arch>.<name>" (clean/runtime/device.cpp; build.py's write_embedded makes these copies of
    # the files it builds in): the same for every code object here, Model L's included
    for n in os.listdir(build):
        if n.endswith(".co") and not n.startswith("gfx"):
            shutil.copyfile(os.path.join(build, n), os.path.join(build, f"{B.ARCH}.{n}"))

    d = model_dir(build)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "maps"))
    SM.write_raw(d, SM.synthetic_tensors(1))
    if model:
        # a converter-written model.wgcm (WrongGPU.exe model convert --preset=L) and the map it was converted with:
        # the runtime checks the file against that map; the reference gets the values the file holds
        from clean.runner.pipeline import read_wgcm
        os.makedirs(os.path.join(d, "L"))
        shutil.copyfile(model, os.path.join(d, "L", "model.wgcm"))
        shutil.copyfile(map_path or DEFAULT_L_MAP, l_map(build))
        raw_l = read_wgcm(model)
        source = {"kind": "converter", "model_sha256": _sha256(model), "map_sha256": _sha256(l_map(build))}
    else:
        raw_l = LL.raw_model(SEED)
        synthetic_map = SM.write_wgcm(os.path.join(d, "L"), raw_l, order=list(LL.specs()), map_file=None)
        shutil.copyfile(synthetic_map, l_map(build))
        source = {"kind": "synthetic", "seed": SEED}
    with open(os.path.join(d, "source.json"), "w") as f:
        json.dump(source, f, indent=1)
    # the reference's parameters: the values the file holds, in the oracle's shapes and types
    p = L.random_params(np.random.default_rng(SEED))
    params_l = {}
    for stage, tensors in raw_l.items():
        params_l[stage] = {}
        for name, a in tensors.items():
            if stage in p and name in p[stage]:
                ref = np.asarray(p[stage][name])
                params_l[stage][name] = np.asarray(a).reshape(ref.shape).astype(ref.dtype)
            else:
                params_l[stage][name] = np.asarray(a)
    params_m = logical_tensors(SM.read_raw(d))
    with open(os.path.join(d, "params.pkl"), "wb") as f:
        pickle.dump({"L": params_l, "M": params_m}, f)
    print("prepared", build)


# ---- --run (GPU) ----------------------------------------------------------------------------------------------------

def frame_inputs(rng, rw, rh, reset, special=True):
    from clean.reference import encoder as En
    x = En.random_inputs(rng, rw=rw, rh=rh, fw=-(-rw // 32) * 32, fh=-(-rh // 32) * 32, special=special)
    return {"color": x["color"], "depth": x["depth"], "motion": x["motion"], "mv_scale": x["mv_scale"],
            "jitter": x["jitter"], "exposure": 1.0, "pre_exposure": PRE_EXPOSURE, "reset": reset}


# ---- raw outputs and sentinels (byte identity) -------------------------------------------------------------------------

# every output compared byte for byte (the runtime's wg_rt_buffer names; Model L's scratch maps l.* are not outputs)
RAW_OUTPUTS = (("exposure.pair", "enc0.features", "enc0.m_near", "enc0.m_far", "enc0.m_center", "enc1.skip",
                "enc1.merged", "enc2.skip", "enc2.merged") + tuple("b%d.out" % i for i in range(6)) +
               ("dec2.out", "dec1.out", "dec0.A", "dec0.B", "dec0.C", "dec0.latent", "post.history", "post.reduced",
                "post.display", "filter.out", "state.latent_prev", "state.motion_history", "state.history_prev"))
# read by the next frame before it writes them: never filled (the encoder reads the reduced history, the exposure step
# the pair); and from the second frame on, the last frame's history, latent and motion surface (the next frame's
# history_prev, latent_prev and motion_history)
CARRIED = ("exposure.pair", "post.reduced")
NEXT_READS = ("post.history", "dec0.latent", "enc0.m_near")
SENTINELS = {"stages": 0xC3, "default": 0xA5, "d3d12_stages": 0x3C, "d3d12": 0x5A}
BNECK_UNWRITTEN = ("b0.out", "b1.out", "b2.out", "b3.out", "b4.out")   # the megakernel writes only b5's output


def fill_sentinels(p, value, first):
    """Every output this frame writes set to the run's sentinel (before the frame)."""
    for n in RAW_OUTPUTS:
        if n in CARRIED or (not first and n in NEXT_READS):
            continue
        p.fill(n, value)


def raw_outputs(p):
    """{output: its defined bytes} after a frame. Pack12 frames (Model L on D3D12, Model M's native frame): image A
    8 bytes and image C 4 bytes per render pixel, B unused (wg_rt_buffer reports the base layout's 8 bytes of C)."""
    rw, rh = p.info()["render_size"]
    pack12 = "pack12" in p.kernels()["dec0"]
    out = {}
    for n in RAW_OUTPUTS:
        b = p.raw(n)
        if pack12 and n == "dec0.B":
            b = b[:0]
        elif pack12 and n == "dec0.C":
            b = b[:rw * rh * 4]
        elif n == "post.display" and b.size:
            # three FP16 planes of rows display_pitch(dw) values apart: the image's bytes (the row padding is written
            # by no kernel)
            from clean.runner import post as Po
            b = b.reshape(3, p.dh, Po.display_pitch(p.dw) * 2)[:, :, :p.dw * 2].reshape(-1).copy()
        out[n] = b
    return out


def pack12_to_base(raw, rw, rh):
    """A record frame's raw outputs with dec0's pack12 images rearranged into the base layout's bytes (A, B: three
    codes and a zero byte; C: C0 C1 C2 as FP16, then two zero bytes), as the HIP run's base-format outputs hold them."""
    a = raw["dec0.A"][:rw * rh * 8].reshape(rh * rw, 8)
    c = raw["dec0.C"][:rw * rh * 4].reshape(rh * rw, 4)
    z = np.zeros((rh * rw, 1), np.uint8)
    out = dict(raw)
    out["dec0.A"] = np.concatenate([a[:, 0:3], z], 1).reshape(-1)
    out["dec0.B"] = np.concatenate([a[:, 3:6], z], 1).reshape(-1)
    out["dec0.C"] = np.concatenate([c, a[:, 6:8], z, z], 1).reshape(-1)
    return out


def raw_compare(ra, rb, sa, sb, unwritten=None, skip=(), convert_b=False):
    """Per frame, every output of RAW_OUTPUTS: the raw defined bytes of record ra against rb (sentinels sa, sb).
    unwritten(frame, name) -> None or the side(s) that do not write it by design ("a", "b", "both"): there every byte of
    that side must still be its sentinel. skip: outputs not compared (the caller gives the reason). convert_b: rb's
    pack12 images rearranged into the base layout first (pack12_to_base). An output missing on either side, of another
    size, or with any differing byte fails the frame."""
    res = []
    for k, (fa, fb) in enumerate(zip(ra["frames"], rb["frames"])):
        rh, rw = fa["inputs"]["color"].shape[:2]
        xa, xb = fa.get("raw", {}), fb.get("raw", {})
        if convert_b:
            xb = pack12_to_base(xb, rw, rh)
        d, notes = {}, {}
        for n in RAW_OUTPUTS:
            if n in skip:
                continue
            u, v = xa.get(n), xb.get(n)
            side = unwritten(k, n) if unwritten else None
            if side:
                ok = (side not in ("a", "both") or (u is not None and bool(np.all(u == sa)))) and \
                     (side not in ("b", "both") or (v is not None and bool(np.all(v == sb))))
                d[n] = 0 if ok else "written, expected unwritten (%s)" % side
                continue
            if u is None or v is None:
                d[n] = "missing"
            elif u.size != v.size:
                d[n] = "size %d vs %d" % (u.size, v.size)
            else:
                d[n] = int(np.sum(u != v))
                both = int(np.sum((u == sa) & (v == sb)))
                if both:
                    notes[n] = {"neither_wrote_bytes": both}
        res.append({"frame": k, "render": [rw, rh], "differ": d, "notes": notes,
                    "identical": all(x == 0 for x in d.values())})
    return res


def raw_summary(frames):
    return {"frames": len(frames), "identical_frames": sum(f["identical"] for f in frames),
            "differing": sorted({n for f in frames for n, v in f["differ"].items() if v != 0}),
            "outputs_compared": len({n for f in frames for n in f["differ"]})}


def drop_write(rec, kind, frame):
    """A copy of a run's record with ONE write dropped at `frame` (negative control): "history": the 4th FP16 of one
    pixel of post.history back to the run's sentinel (what a dropped store leaves there); "reduced": bits 30-31 of one
    word of post.reduced flipped (the reduced history is not sentinel-filled: a dropped store leaves the last frame's
    word, here one differing only in the bits a decoded comparison drops); "reduced_frame": the whole post.reduced
    write dropped (the last frame's words stay); "missing": post.history absent from the record."""
    out = copy.deepcopy(rec)
    raw = out["frames"][frame]["raw"]
    if kind == "missing":
        del raw["post.history"]
    elif kind == "reduced_frame":
        raw["post.reduced"] = rec["frames"][frame - 1]["raw"]["post.reduced"].copy()
    elif kind == "history":
        nb = int(rec["kernels"].get("history_bytes", 8))
        h = raw["post.history"]
        px = (h.size // nb) // 2 + 17
        h[px * nb + nb - 2: px * nb + nb] = rec["sentinel"]
    else:
        r = raw["post.reduced"].view(np.uint32)
        r[r.size // 2 + 5] ^= np.uint32(0xC0000000)
    return out


def negative_controls(ra, rb, sa, sb, unwritten, skip=(), convert_b=False, frame=5):
    """The record-level negative controls on comparison (ra, rb): each must make the comparison fail."""
    res = {}
    for kind, name in (("history", "post.history"), ("reduced", "post.reduced"), ("reduced_frame", "post.reduced"),
                       ("missing", "post.history")):
        f = raw_compare(ra, drop_write(rb, kind, frame), sa, sb, unwritten, skip, convert_b)
        res[kind] = {"frame": frame, "caught": not f[frame]["identical"], "differ": f[frame]["differ"].get(name)}
    return res


def run_d3d12(build, special=True, stages=False):
    """The frame sequence on native D3D12 (backend 3), also the raw pack12 images: "d3d12" with the default kernels,
    "d3d12_stages" with the D3D12 fusions off (WG_D3D12_FUSION=none), so that every stage output is written except b0 ...
    b4 (the bottleneck runs as its megakernel there: the plain C = 128 block has no D3D12 port). One configuration
    per process: the runtime library reads WG_D3D12_FUSION from the environment when it is loaded."""
    if stages:
        os.environ["WG_D3D12_FUSION"] = "none"  # before the runtime library is loaded
    from clean.runner.pipeline import Pipeline
    d = model_dir(build)
    setups = {"L": dict(map_path=l_map(build), check_packing=False, kernels={"options": "model=L"}),
              "M": dict(check_packing=True, kernels={})}
    flags = {"L": {"depth_inverted": False, "auto_exposure": True, "hdr": True},  # as the HIP run's
             "M": {"depth_inverted": False, "auto_exposure": True}}
    for model, kw in setups.items():
        for config in ("d3d12_stages",) if stages else ("d3d12",):
            p = Pipeline(build, "", d, MAX_RENDER, OUTPUT, flags[model], backend="d3d12", **kw)
            rng = np.random.default_rng(SEED)
            frames = []
            for i, ((rw, rh), reset) in enumerate(SEQUENCE):
                fr = frame_inputs(rng, rw, rh, reset, special)
                fill_sentinels(p, SENTINELS[config], i == 0)
                out = p.frame(fr, debug=True)
                frames.append({"inputs": fr, "out": out, "raw": raw_outputs(p)})
            rec = {"model": model, "config": config, "sentinel": SENTINELS[config], "kernels": p.kernels(),
                   "notes": p.notes(),
                   "model_source": p.model_source, "dw": p.dw, "dh": p.dh, "ow": p.ow, "oh": p.oh,
                   "taps": p.output_taps(), "frames": frames}
            with open(os.path.join(build, f"lframe_{model}_{config}.pkl"), "wb") as f:
                pickle.dump(rec, f)
            print(model, config, "frames", len(frames), "post", rec["kernels"].get("post"), "pack",
                  rec["kernels"].get("post_pack"), "bneck", rec["kernels"].get("swin_bneck"))
            del p


def unpack12(raw_a, raw_c, rw, rh):
    """The pack12 images -> A, B codes [rh, rw, 3] (uint8) and C [rh, rw, 3] (FP16 as float32)."""
    a = np.frombuffer(raw_a.tobytes()[:rw * rh * 8], np.uint8).reshape(rh, rw, 8)
    c = np.frombuffer(raw_c.tobytes()[:rw * rh * 4], np.uint8).reshape(rh, rw, 4)
    A = a[..., 0:3]
    B = np.stack([a[..., 3], a[..., 4], a[..., 5]], -1)
    C = np.stack([c[..., 0:2].copy().view(np.float16)[..., 0], c[..., 2:4].copy().view(np.float16)[..., 0],
                  a[..., 6:8].copy().view(np.float16)[..., 0]], -1).astype(np.float32)
    return A, B, C


POST_OUTPUTS = ("post.history", "post.reduced", "state.history_prev", "out")


def compare_backends(hip, dx, same_post=True):
    """Per frame: the raw defined bytes of every output, HIP run against D3D12 run (raw_compare; the D3D12 pack12
    images in the base layout). same_post False (the backends run different reconstruction kernels): the post outputs
    are left out. Not written by design: b0 ... b4 on both (the bottleneck megakernels), on D3D12 the maps its fusions
    keep on chip (enc1's merged map; Model M's dec2 / dec1 outputs, which its fused decoder reads on chip) and the display
    image (the fused pack12 post writes none), on the first frame the previous frame's state on both."""
    fused_onchip = ("enc1.merged",) + (("dec2.out", "dec1.out") if hip["model"] == "M" else ())

    def unwritten(k, n):
        if n in BNECK_UNWRITTEN or (k == 0 and n in UNWRITTEN_ON_FIRST):
            return "both"
        if n in fused_onchip or n == "post.display":
            return "b"
        return None
    skip = () if same_post else POST_OUTPUTS_RAW
    return raw_compare(hip, dx, hip["sentinel"], dx["sentinel"], unwritten, skip, convert_b=True)


def compare_d3d12_configs(st, df):
    """D3D12 default kernels (df) against the run with the fusions off (st), raw defined bytes. Not written by design:
    b0 ... b4 on both (the megakernel runs in both), the display image on both (the fused pack12 post), enc1's merged
    map with the defaults (fusion pm keeps it on chip), on the first frame the previous frame's state on both."""
    def unwritten(k, n):
        if n in BNECK_UNWRITTEN or n == "post.display" or (k == 0 and n in UNWRITTEN_ON_FIRST):
            return "both"
        if n == "enc1.merged":
            return "b"
        return None
    return raw_compare(st, df, st["sentinel"], df["sentinel"], unwritten), unwritten


# the post outputs (left out where the two runs have different reconstruction kernels)
POST_OUTPUTS_RAW = ("post.history", "post.reduced", "post.display", "filter.out", "state.history_prev")


def fill_pack12(rec):
    """A copy of a D3D12 record with dec0's A, B, C taken from the unpacked pack12 images (the debug view has none);
    with the true-fusion post (one launch, reconstruction and filter) no display image, which that kernel does not
    write."""
    out = copy.deepcopy(rec)
    tf = "post_tf" in str(rec["kernels"].get("post_last", ""))
    for fd in out["frames"]:
        if tf:
            fd["out"]["post"]["display"] = None
        rh, rw = fd["inputs"]["color"].shape[:2]
        A, B, C = unpack12(fd["raw"]["dec0.A"], fd["raw"]["dec0.C"], rw, rh)
        fd["out"]["dec0"].update(A=(A / 255.0).astype(np.float32), B=(B / 255.0).astype(np.float32), C=C)
    return out


def check_m_post(dx, params):
    """Model L's D3D12 frame runs Model M's fused pack12 post: its history and reduced codes against Model M's post
    reference (clean/reference/post.py) on the frame's own inputs (the unpacked pack12 coefficients), under
    test_post_emu.py's acceptance as in --compare."""
    from clean.reference import post as Po
    s = float(np.asarray(params["post"]["s"]).reshape(-1)[0])
    D = (dx["dw"], dx["dh"])
    res, prev, prev_size = [], None, None
    for k, fd in enumerate(dx["frames"]):
        fr, out = fd["inputs"], fd["out"]
        rh, rw = fr["color"].shape[:2]
        reset = prev is None or bool(fr["reset"]) or prev_size != (rw, rh)
        A, B, C = unpack12(fd["raw"]["dec0.A"], fd["raw"]["dec0.C"], rw, rh)
        xp = {"color": (fr["color"] / np.float32(fr["pre_exposure"])).astype(np.float32),
              "exposure": out["exposure"]["e"], "exposure_prev": out["exposure"]["e_prev"],
              "A": (A / 255.0).astype(np.float32), "B": (B / 255.0).astype(np.float32), "C": C,
              "history": out["state"]["history_prev"] if not reset else np.zeros((D[1], D[0], 3), np.float32),
              "m_near": out["enc0"]["m_near"], "m_center": out["enc0"]["m_center"], "m_far": out["enc0"]["m_far"],
              "jitter": fr["jitter"], "reset": reset, "recon_size": D}
        ref = Po.reconstruct(xp, s)
        h = post_fp16(out["post"]["history"], ref["history"])
        dc = codes(out["post"]["reduced"], ref["reduced"], 1023.0)
        dc["pass"] = dc["max_code_diff"] <= 1 and dc["differ"] <= 0.01 * dc["values"]
        res.append({"frame": k, "history": h, "reduced": dc, "pass": h["pass"] and dc["pass"]})
        prev, prev_size = out, (rw, rh)
    return res


def compare_d3d12(build, json_path):
    with open(os.path.join(model_dir(build), "params.pkl"), "rb") as f:
        params = pickle.load(f)
    res = {}
    for model in ("L", "M"):
        recs = {}
        for config in ("default", "d3d12"):
            with open(os.path.join(build, f"lframe_{model}_{config}.pkl"), "rb") as f:
                recs[config] = pickle.load(f)
        # the HIP run's reconstruction: Model M's (bit-identical to the D3D12 fused pack12 post) or Model L's own
        hip_post = recs["default"]["kernels"].get("l_post") or recs["default"]["kernels"].get("post_last")
        same_post = hip_post != "l_post_recon"
        frames = compare_backends(recs["default"], recs["d3d12"], same_post)
        res[model] = {"kernels_d3d12": recs["d3d12"]["kernels"], "frames": frames,
                      "outputs_compared": sorted({n for f in frames for n in f["differ"]}),
                      "identical_frames": sum(f["identical"] for f in frames),
                      "differing": sorted({k for f in frames for k, v in f["differ"].items() if v != 0}),
                      "post_compared_with_hip": same_post}
        if same_post:  # the record-level negative controls on this comparison
            dflt = recs["default"]
            res[model]["negative_controls"] = negative_controls(
                dflt, recs["d3d12"], dflt["sentinel"], recs["d3d12"]["sentinel"],
                lambda k, n: ("both" if n in BNECK_UNWRITTEN or (k == 0 and n in UNWRITTEN_ON_FIRST) else
                              "b" if n in ("enc1.merged", "dec2.out", "dec1.out", "post.display") else None),
                convert_b=True)
        if not same_post:
            # Model L: HIP runs l_post_recon, D3D12 Model M's fused pack12 post, and each frame's encoder reads the
            # previous frame's reconstruction, so the two runs part after the first frame. The D3D12 frame is checked
            # on its own: every stage of the run with all outputs written against the reference on that run's inputs
            # (as --compare), and the default kernels against that run byte for byte.
            with open(os.path.join(build, f"lframe_{model}_d3d12_stages.pkl"), "rb") as f:
                st = fill_pack12(pickle.load(f))
            per_stage = compare_model(st, params[model], post_ref="M", bneck_unit=True)
            fused, unw = compare_d3d12_configs(st, recs["d3d12"])
            res[model]["d3d12_per_stage"] = {"frames": per_stage, "summary": summary(per_stage),
                                             "passed": sum(m["pass"] for m in per_stage)}
            res[model]["d3d12_default_vs_stages"] = dict(raw_summary(fused), frames=fused)
            res[model]["negative_controls"] = negative_controls(st, recs["d3d12"], st["sentinel"],
                                                                recs["d3d12"]["sentinel"], unw)
            mp = check_m_post(recs["d3d12"], params[model])
            res[model]["m_post_reference"] = {"frames": mp, "passed": sum(f["pass"] for f in mp)}
    n_frames = {m: len(res[m]["frames"]) for m in ("L", "M")}

    def model_pass(m):
        r = res[m]
        caught = all(v["caught"] for v in r["negative_controls"].values())
        if r["post_compared_with_hip"]:
            return r["identical_frames"] == n_frames[m] and caught
        return (r["d3d12_per_stage"]["passed"] == n_frames[m] and r["m_post_reference"]["passed"] == n_frames[m] and
                r["d3d12_default_vs_stages"]["identical_frames"] == n_frames[m] and caught)
    res["pass"] = all(model_pass(m) for m in ("L", "M"))
    with open(json_path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for model in ("L", "M"):
        print(model, "D3D12 against HIP (raw bytes): frames identical", res[model]["identical_frames"], "of",
              len(res[model]["frames"]), "outputs compared", len(res[model]["outputs_compared"]), "differing",
              res[model]["differing"])
        print("  negative controls (one dropped write):", json.dumps(res[model]["negative_controls"]))
        if "d3d12_per_stage" in res[model]:
            r = res[model]
            print("  (HIP runs l_post_recon, D3D12 Model M's fused pack12 post: the runs part after frame 0)")
            print("  D3D12 per stage against the reference (fusions off; b0-b5 as one unit):",
                  r["d3d12_per_stage"]["passed"],
                  "of", len(r["frames"]), "frames pass;", "failing:",
                  sorted({s for f in r["d3d12_per_stage"]["frames"] for s, ok in f["stage_pass"].items() if not ok}))
            print("  D3D12 default kernels against that run (raw bytes): identical", r["d3d12_default_vs_stages"][
                "identical_frames"], "of", len(r["frames"]), "differing", r["d3d12_default_vs_stages"]["differing"])
            print("  its post against Model M's post reference (default kernels):", r["m_post_reference"]["passed"],
                  "of", len(r["frames"]), "frames pass")
    return 0 if res["pass"] else 1


def run(build, rocm, special=True):
    from clean.runner.hip import clean_process_environment
    clean_process_environment(rocm)
    from clean.runner.pipeline import Pipeline
    d = model_dir(build)
    # Model L: HDR colour (its reconstruction kernel l_post_recon is HDR; the runtime refuses Model L without it)
    setups = {"L": dict(map_path=l_map(build), check_packing=False, kernels={"options": "model=L"}),
              "M": dict(check_packing=True, kernels={"pack": "none"})}
    flags = {"L": {"depth_inverted": False, "auto_exposure": True, "hdr": True},
             "M": {"depth_inverted": False, "auto_exposure": True}}
    for model, kw in setups.items():
        # "stages": the bottleneck as six block launches, so that b0 ... b4 are written (the default megakernel writes
        # only b5's output); "default": the runtime's default kernels, which must give the same bytes
        for config in ("stages", "default"):
            k = dict(kw, kernels=dict(kw["kernels"], **({"bneck": "none"} if config == "stages" else {})))
            p = Pipeline(build, rocm, d, MAX_RENDER, OUTPUT, flags[model], **k)
            rng = np.random.default_rng(SEED)
            frames = []
            for i, ((rw, rh), reset) in enumerate(SEQUENCE):
                fr = frame_inputs(rng, rw, rh, reset, special)
                fill_sentinels(p, SENTINELS[config], i == 0)
                out = p.frame(fr, debug=True)
                frames.append({"inputs": fr, "out": out, "raw": raw_outputs(p)})
            rec = {"model": model, "config": config, "sentinel": SENTINELS[config], "kernels": p.kernels(),
                   "notes": p.notes(),
                   "model_source": p.model_source, "dw": p.dw, "dh": p.dh, "ow": p.ow, "oh": p.oh,
                   "taps": p.output_taps(), "frames": frames}
            with open(os.path.join(build, f"lframe_{model}_{config}.pkl"), "wb") as f:
                pickle.dump(rec, f)
            print(model, config, "frames", len(frames), "bneck", rec["kernels"].get("swin_bneck"), "post",
                  rec["kernels"].get("post"))
            del p


# ---- --compare (CPU) ------------------------------------------------------------------------------------------------

def _f16_steps(a, b):
    ia = np.asarray(a, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ib = np.asarray(b, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    return np.abs(np.where(ia < 0, -(ia & 0x7FFF), ia) - np.where(ib < 0, -(ib & 0x7FFF), ib))


def fp8_stage(got, ref):
    from clean.l.runner import l_c32_check as C32
    m = C32.metrics(got, ref)
    m.update(_magnitudes(got, ref))
    m["pass"] = bool(C32.passes(m))
    return m


def _magnitudes(got, ref):
    """Largest magnitude of the finite values on each side and the reference's non-finite count: with real weights a
    side that stops growing where the other goes on shows saturation, a non-finite reference an overflow in the model."""
    g, r = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    fg, fr = np.isfinite(g), np.isfinite(r)
    return {"ref_nonfinite": int(np.sum(~fr)), "got_absmax": float(np.abs(g[fg]).max()) if fg.any() else 0.0,
            "ref_absmax": float(np.abs(r[fr]).max()) if fr.any() else 0.0}


def motion_surface(got, ref):
    ref = np.asarray(ref, np.float32)
    fin = np.isfinite(ref)
    u = _f16_steps(got[fin], ref[fin])
    same = bool(np.array_equal(np.asarray(got, np.float32)[~fin], ref[~fin], equal_nan=True))
    return {"values": int(u.size), "differ": int(np.sum(u > 0)), "max_f16_steps": int(u.max()) if u.size else 0,
            "pass": same and (int(u.max()) if u.size else 0) <= 1 and float(np.mean(u == 0)) >= 0.995}


def codes(got, ref, scale):
    g, r = np.rint(np.asarray(got, np.float64) * scale), np.rint(np.asarray(ref, np.float64) * scale)
    dc = np.abs(g - r)
    return {"values": int(dc.size), "differ": int(np.sum(dc > 0)), "max_code_diff": int(dc.max()) if dc.size else 0}


def same_values(got, ref):
    g, r = np.asarray(got, np.float32), np.asarray(ref, np.float32)
    dif = ~((g == r) | (np.isnan(g) & np.isnan(r)))
    return {"values": int(dif.size), "differ": int(np.sum(dif)),
            "max_f16_steps": int(_f16_steps(g[dif], r[dif]).max()) if dif.any() else 0}


def post_fp16(got, ref, model="M"):
    finite = np.isfinite(got)
    s = _f16_steps(got, ref)
    d = np.abs(np.where(finite, got, 0.0).astype(np.float64) - ref)
    m = {"values": int(s.size), "differ": int(np.sum(s > 0)), "max_f16_steps": int(s.max()) if s.size else 0,
         "nonfinite": int(np.sum(~finite)), "frac_over_1": float(np.mean(s > 1)), "over_1": int(np.sum(s > 1)),
         "frac_diff": float(np.mean(s > 0)),
         "mean_rel_rms": float(d.mean() / max(np.sqrt(np.mean(np.square(ref, dtype=np.float64))), 1e-30))}
    m.update(_magnitudes(got, ref))
    if model == "L":
        # Model L's post acceptance (clean/l/runner/l_post_check.py, provisional): per frame at most 0.05 % more than
        # one ulp off; the run-level 0.01 % is checked in compare()
        m["pass"] = LP.l_post_rule(m["values"], m["nonfinite"], m["over_1"], m["differ"], m["mean_rel_rms"])
    else:
        m["pass"] = (m["nonfinite"] == 0 and m["frac_over_1"] <= 1e-4 and m["mean_rel_rms"] <= 1e-3
                     and m["frac_diff"] <= 0.01)
    return m


def compare_model(rec, params, post_ref=None, bneck_unit=False):
    """post_ref "M": the post against Model M's reference whatever the model (Model L's native D3D12 frame runs Model
    M's fused pack12 post for now). bneck_unit: b0 ... b5 as one unit (a bottleneck megakernel writes only b5): the
    reference chained from the runtime's enc2.merged, b5 under the per-stage rule."""
    from clean.l.reference import oracle as L
    from clean.l.reference import table as T
    from clean.reference import decoder as MD
    from clean.reference import encoder as ME
    from clean.reference import exposure as Ex
    from clean.l.reference import recon as R
    from clean.reference import post as Po
    from clean.reference import swin as S
    model = rec["model"]
    table = copy.deepcopy(T.TABLE)
    table["consts"]["default"]["gelu"] = "sigmoid"   # the kernels' GELU (clean/l/runner/l_c32_check.py CONSTS)
    D = (rec["dw"], rec["dh"])
    res, prev, prev_jitter, n, f = [], None, None, 0, 0
    prev_size = None
    for k, fd in enumerate(rec["frames"]):
        fr, out = fd["inputs"], fd["out"]
        rh, rw = fr["color"].shape[:2]
        fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
        size_change = prev_size is not None and prev_size != (rw, rh)
        eff_reset = prev is None or bool(fr["reset"]) or size_change
        jp = tuple(float(v) for v in (fr["jitter"] if eff_reset else prev_jitter))
        info = out["state"]["info"]
        m = {"frame": k, "render": [rw, rh], "reset": eff_reset, "parity": f}
        offsets_ok = all(tuple(info["offsets"][s]) == tuple(S.window_offset(s, f)) for s in T.SHARED_STAGES)
        m["state"] = {"n_parity_reset": (info["n"], info["parity"], bool(info["reset"])) == (n, f, eff_reset),
                      "jitter_prev": bool(np.array_equal(np.float32(info["jitter_prev"]), np.float32(jp))),
                      "offsets": offsets_ok}
        if prev is not None and not size_change:
            m["state"]["latent_handed_over"] = bool(np.array_equal(out["state"]["latent_prev"],
                                                                   prev["dec0"]["latent"]))
            m["state"]["motion_history_handed_over"] = bool(np.array_equal(out["state"]["motion_history"],
                                                                           prev["enc0"]["m_near"]))
            m["state"]["history_handed_over"] = bool(np.array_equal(out["state"]["history_prev"],
                                                                    prev["post"]["history"], equal_nan=True))
        m["state"]["pass"] = all(m["state"].values())
        pi = np.float32(fr["pre_exposure"])
        cnet = (fr["color"] / pi).astype(np.float32)
        e_ref = Ex.exposure(fr["color"], fr["pre_exposure"], 0.18)
        m["exposure"] = {"runtime": float(out["exposure"]["e"]), "reference": float(e_ref),
                         "identical": float(out["exposure"]["e"]) == float(e_ref)}

        # enc0 on the runtime's own inputs (the state it read)
        reduced = prev["post"]["reduced"] if prev is not None else np.zeros((D[1] // 2, D[0] // 2, 3), np.float32)
        x0 = {"color": cnet, "depth": fr["depth"], "motion": fr["motion"], "mv_scale": fr["mv_scale"],
              "mv_bias": np.zeros(2, np.float32), "jitter": fr["jitter"], "jitter_prev": np.array(jp),
              "exposure": out["exposure"]["e"], "exposure_prev": out["exposure"]["e_prev"], "reset": eff_reset,
              "frame": n, "depth_inverted": False, "feature_size": (fw, fh), "recon_size": D,
              "reduced_history": reduced, "latent": out["state"]["latent_prev"],
              "motion_history": out["state"]["motion_history"]}
        if eff_reset:
            x0["latent"] = np.zeros_like(x0["latent"])
            x0["motion_history"] = np.zeros_like(x0["motion_history"])
            x0["reduced_history"] = np.zeros_like(reduced)
        ref = L.encode(x0, params, f, t=table) if model == "L" else ME.encode(x0, params["enc0"])
        m["enc0"] = {"features": fp8_stage(out["enc0"]["features"], ref["features"])}
        for s in ("m_near", "m_far", "m_center"):
            m["enc0"][s] = motion_surface(out["enc0"][s], ref[s])

        # the ten Swin stages
        x, skips = out["enc0"]["features"], {}
        for s in T.SHARED_STAGES:
            off = S.window_offset(s, f)
            kind = S.STAGES[s]["kind"]
            skip = skips.get("enc2" if s == "dec2" else "enc1") if kind == "dec" else None
            ref = S.run_stage(s, params[s], x, off, skip, gelu="sigmoid")
            if bneck_unit and s in ("b0", "b1", "b2", "b3", "b4"):
                x = ref["out"]  # the reference's own chain (the runtime does not write b0 ... b4)
                continue
            m[s] = {key: fp8_stage(out[s][key], ref[key]) for key in out[s]}
            if kind == "enc":
                skips[s] = out[s]["skip"]
                x = out[s]["merged"]
            else:
                x = out[s]["out"]

        # dec0 on the runtime's dec1 output
        xd = {"features": out["dec1"]["out"], "motion": out["enc0"]["m_near"],
              "latent_prev": out["state"]["latent_prev"] if not eff_reset else np.zeros((rh, rw, 4), np.float32),
              "jitter": fr["jitter"], "jitter_prev": np.array(jp), "reset": eff_reset, "feature_size": (fw, fh),
              "recon_size": D}
        ref = L.decode(xd, params, f, t=table) if model == "L" else MD.decode(xd, params["dec0"])
        dec = out["dec0"]
        m["dec0"] = {"codes_A": codes(dec["A"], ref["A"], 255.0), "codes_B": codes(dec["B"], ref["B"], 255.0),
                     "codes_L": codes(dec["latent"], ref["latent"], 127.0), "C": same_values(dec["C"], ref["C"])}
        m["dec0"]["pass"] = all(v["differ"] <= 0.005 * v["values"] for v in m["dec0"].values())

        # post and filter on the runtime's own inputs
        xp = {"color": cnet, "exposure": out["exposure"]["e"], "exposure_prev": out["exposure"]["e_prev"],
              "A": dec["A"], "B": dec["B"], "C": dec["C"],
              "history": out["state"]["history_prev"] if not eff_reset else np.zeros((D[1], D[0], 3), np.float32),
              "m_near": out["enc0"]["m_near"], "m_center": out["enc0"]["m_center"], "m_far": out["enc0"]["m_far"],
              "jitter": fr["jitter"], "reset": eff_reset, "recon_size": D}
        s_post = float(np.asarray(params["post"]["s"]).reshape(-1)[0])
        l_recon = model == "L" and post_ref != "M"
        ref = R.reconstruct(xp, s_post) if l_recon else Po.reconstruct(xp, s_post)
        m["post"] = {"history": post_fp16(out["post"]["history"], ref["history"], model)}
        if out["post"]["display"] is not None:
            m["post"]["display"] = post_fp16(out["post"]["display"], ref["display"], model)
        dc = codes(out["post"]["reduced"], ref["reduced"], 1023.0)
        dc["pass"] = dc["max_code_diff"] <= 1 and dc["differ"] <= 0.01 * dc["values"]
        m["post"]["reduced"] = dc
        if out["post"]["display"] is not None:
            disp, e = out["post"]["display"], out["exposure"]["e"]
            if rec["taps"] is not None:
                fo = Po.output_filter_original(disp, e, (rec["ow"], rec["oh"]), rec["taps"])
            elif l_recon:
                fo = R.output_filter(disp, e, (rec["ow"], rec["oh"]))
            else:
                fo = Po.output_filter(disp, e, (rec["ow"], rec["oh"]))
            fo = (np.asarray(fo, np.float32) * pi).astype(np.float32)
            close = np.isclose(out["out"], fo, rtol=1e-5, atol=2e-9)
            m["filter"] = {"values": int(close.size), "outside": int(np.sum(~close)), "pass": bool(close.all())}

        stage_pass = {s: (all(v["pass"] for v in m[s].values() if isinstance(v, dict) and "pass" in v)
                          if isinstance(m[s], dict) else True)
                      for s in ["enc0"] + list(T.SHARED_STAGES) + ["post"] if s in m}
        stage_pass["dec0"] = m["dec0"]["pass"]
        stage_pass["state"] = m["state"]["pass"]
        if "filter" in m:
            stage_pass["filter"] = m["filter"]["pass"]
        m["stage_pass"] = stage_pass
        m["pass"] = all(stage_pass.values())
        res.append(m)
        prev, prev_jitter, prev_size = out, np.asarray(fr["jitter"], np.float32), (rw, rh)
        n, f = (n + 1) % 8, 1 - f
    return res


def summary(frames):
    """Per stage: frames passing, values compared and values differing (FP8 / FP16 steps or codes) over all frames."""
    out = {}
    for m in frames:
        for stage, part in m.items():
            if stage in ("frame", "render", "reset", "parity", "pass", "stage_pass", "exposure"):
                continue
            s = out.setdefault(stage, {"frames": 0, "passed": 0, "values": 0, "differ": 0, "max_diff": 0,
                                       "nonfinite": 0, "ref_nonfinite": 0, "got_absmax": 0.0, "ref_absmax": 0.0})
            s["frames"] += 1
            s["passed"] += int(m["stage_pass"].get(stage, True))
            for v in part.values() if isinstance(part, dict) else []:
                if isinstance(v, dict):
                    s["values"] += v.get("values", 0)
                    s["differ"] += v.get("differ", v.get("outside", 0))
                    # the largest difference: FP8 steps (per-stage rule), FP16 ulps or codes, as the metric has it
                    s["max_diff"] = max([s["max_diff"]] + [int(x) for k, x in v.items() if k.startswith("max_")])
                    s["nonfinite"] += v.get("nonfinite", 0)
                    s["ref_nonfinite"] += v.get("ref_nonfinite", 0)
                    s["got_absmax"] = max(s["got_absmax"], v.get("got_absmax", 0.0))
                    s["ref_absmax"] = max(s["ref_absmax"], v.get("ref_absmax", 0.0))
            if stage == "filter":
                s["values"] += part["values"]
                s["differ"] += part["outside"]
    out["exposure_identical"] = sum(int(m["exposure"]["identical"]) for m in frames)
    return out


UNWRITTEN_ON_FIRST = ("state.latent_prev", "state.motion_history", "state.history_prev")


def compare(build, json_path):
    with open(os.path.join(model_dir(build), "params.pkl"), "rb") as f:
        params = pickle.load(f)
    res = {}
    for model in ("L", "M"):
        recs = {}
        for config in ("stages", "default"):
            with open(os.path.join(build, f"lframe_{model}_{config}.pkl"), "rb") as f:
                recs[config] = pickle.load(f)
        frames = compare_model(recs["stages"], params[model])
        # the default kernels against the six-launch bottleneck, raw defined bytes: b0 ... b4 are not written by the
        # default megakernel; on the first frame the previous frame's state is written by neither (a reset frame does
        # not read it)
        def unwritten(k, n):
            if n in BNECK_UNWRITTEN:
                return "b"
            return "both" if k == 0 and n in UNWRITTEN_ON_FIRST else None
        st_, df_ = recs["stages"], recs["default"]
        same = raw_compare(st_, df_, st_["sentinel"], df_["sentinel"], unwritten)
        neg = negative_controls(st_, df_, st_["sentinel"], df_["sentinel"], unwritten)
        res[model] = {"kernels": recs["default"]["kernels"], "kernels_stages": recs["stages"]["kernels"],
                      "model_source": recs["default"]["model_source"], "notes": recs["default"]["notes"],
                      "frames": frames, "summary": summary(frames),
                      "default_identical": dict(raw_summary(same), frames=same), "negative_controls": neg,
                      "pass": (all(m["pass"] for m in frames) and all(f["identical"] for f in same) and
                               all(v["caught"] for v in neg.values()))}
        if model == "L":  # Model L's run-level limit on post values more than one ulp off (provisional)
            res[model]["post_run"] = {k: LP.l_post_run([m["post"][k] for m in frames if k in m["post"]])
                                      for k in ("history", "display")}
            res[model]["pass"] = res[model]["pass"] and all(r["pass"] for r in res[model]["post_run"].values())
    res["pass"] = res["L"]["pass"] and res["M"]["pass"]
    src = os.path.join(model_dir(build), "source.json")
    if os.path.exists(src):
        with open(src) as f:
            res["L"]["source"] = json.load(f)
        print("Model L source:", json.dumps(res["L"]["source"]))
    with open(json_path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for model in ("L", "M"):
        print(model, "pass" if res[model]["pass"] else "FAIL", json.dumps(res[model]["summary"]))
        print("  default kernels against the six-launch bottleneck (raw bytes):",
              json.dumps({k: v for k, v in res[model]["default_identical"].items() if k != "frames"}),
              "negative controls:", json.dumps(res[model]["negative_controls"]))
        for m in res[model]["frames"]:
            if not m["pass"]:
                print("  frame", m["frame"], "failing:", [s for s, ok in m["stage_pass"].items() if not ok])
    return 0 if res["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--compare", action="store_true")
    g.add_argument("--compare-d3d12", action="store_true")
    ap.add_argument("--backend", choices=("hip", "d3d12", "d3d12-stages"), default="hip",
                    help="--run: HIP (both bottleneck forms), native D3D12 with its default kernels, or native D3D12 "
                         "with its fusions off (d3d12-stages; run both D3D12 forms before --compare-d3d12)")
    ap.add_argument("--json")
    ap.add_argument("--plain", action="store_true",
                    help="--run: frames without random_inputs' special values (NaN / huge / negative colours, NaN "
                         "depths, motion leaving the frame)")
    ap.add_argument("--model", help="--prepare: a converter-written Model L model.wgcm in place of the synthetic model")
    ap.add_argument("--map", help="--prepare with --model: the map it was converted with (default: the shipped "
                                  "data/preset-L.v1.map.json)")
    a = ap.parse_args(argv)
    if (a.model or a.map) and not a.prepare:
        ap.error("--model and --map go with --prepare")
    if a.map and not a.model:
        ap.error("--map goes with --model")
    if a.prepare:
        prepare(a.build, a.rocm, a.model, a.map)
        return 0
    if a.run:
        if a.backend.startswith("d3d12"):
            run_d3d12(a.build, special=not a.plain, stages=a.backend == "d3d12-stages")
        else:
            run(a.build, a.rocm, special=not a.plain)
        return 0
    if a.compare_d3d12:
        return compare_d3d12(a.build, a.json or os.path.join(a.build, "lframe_d3d12.json"))
    return compare(a.build, a.json or os.path.join(a.build, "lframe_check.json"))


if __name__ == "__main__":
    sys.exit(main())
