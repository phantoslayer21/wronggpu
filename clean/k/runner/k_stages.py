# SPDX-License-Identifier: MIT
"""Model K stages of the spec 90 runner (clean/runner/stage_runner.py): weight packing and launch plans.

    r = Runner(build_dir, rocm_dir, model="K")       # stage names without the "k." prefix, or
    r = Runner(build_dir, rocm_dir)                  # and "k.<stage>" with model M's runner
    w = r.pack_weights("e2", tensors)                # tensor names: clean/k/reference/network.py
    out = r.run("e2", w, x, (ox, oy))                # {"block": [H, W, 64], "merged": [H/2, W/2, 96]}
    out = r.run("d3", w, x, (ox, oy), skip=S3)       # {"block": [H3, W3, 96]}; x = d4's block output (level 4)
    ms = r.time("e2", w, x, (ox, oy))

Maps are float32 arrays [H, W, C] holding FP16 values (inputs are rounded to FP16; outputs are FP16-exact).

Stages:
  e0 .. e4, m, d4 .. d0   spec 41 stages. x: the stage input (e0: the 16-channel network input at level 0; e1..e4,
                          m: the previous merged map at the block level; dec: the coarser map). skip: S<level> (dec,
                          required; its size is the block level's size). Outputs: "block" (all), "merged" (enc),
                          "head" (d0, 40 channels). offset: (ox, oy) of spec 44.
  block:<C>:<heads>:<mix> one block on x [H, W, C]; tensors as a stage's block part. Output "out".
  lin:<mode>:<K>x<N>      one linear layer (mode plain / merge / expand); tensors {"W": [N, K], "b": [N] (optional)};
                          x: the input map or {"x": map, "act": bool, "relu": bool, "out_size": (W, H) (expand)};
                          skip: residual / skip map at the output (optional). Output "out".
  win                     window partition round trip (K3): x = {"x": map [H, W, C], "offset": (ox, oy), "margin": k,
                          "border": "mirror" | "clamp" | "zero"} -> {"windows": [nw, S*S, C], "back": [H, W, C]}.
  act                     the activation on all 65536 FP16 inputs, scalar and packed form, compared with the NumPy
                          oracle: {"scalar", "packed": uint16 [65536], "mismatch_scalar", "mismatch_packed": int}.
  input                   spec 42: x = the frame dict of clean/k/reference/instage.py (color, motion, depth, flags,
                          jitter, exposure values, Hc, Hh, ...) -> {"x0": [Bh, Bw, 16]}.
  output                  spec 43: x = the frame dict of clean/k/reference/outstage.py (N, color, motion, depth, flags,
                          L, Hc, ...; "table" optional) -> {"out": [H, W, 4] RGBA, "L", "Hc", "Hh"}.
Stages run as one fused kernel (k_stage_<name>) when the offset allows it (even offsets, as spec 44 gives); suffix
"@split" runs the separate kernels instead (embedding / expand, block, merge / head), "@g1" the same with the linear
kernels at 16 tokens per wave (k_linear__g1.co). Odd offsets always use the separate kernels. "@w2" (e0, d0): the
fused kernel with two windows per workgroup (k_stage2_*). "@wide2" / "@wide4" / "@wide" (e3, e4, m, d4, d3; even
offsets): the wide kernel k_stagew<NG>_* (k_block_wide.hip) with NG = 2 / 4 / the network's choice.
These accumulate in FP32 (spec 41 as written). A further suffix "@f16" ("e2@f16", "e0@w2@f16", "e3@wide2@f16"; even
offsets) runs the FP16-accumulating form of the fused or wide kernel instead (k_stageh_* with m included,
k_stage2h_*, k_stagew<NG>h_*: the spec 49 speed approximation of clean/k/README.md, opt-in).
"""
import numpy as np

from clean.k.reference.layout import STAGE, cdiv, window_grid
from clean.k.runner import k_launch as KL
from clean.k.runner import k_pack

KERNEL_CODE_OBJECTS = KL.KERNEL_CODE_OBJECTS
STAGE_NAMES = list(STAGE)


def _f16_bytes(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).astype(np.float16)).tobytes()


def _decode_f16(raw, shape):
    return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)


def _decode_u16(raw, shape):
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()


def _split(stage):
    """"e2" -> ("e2", "fused"); "e0@w2" -> ("e0", "fused2"); "e2@split" -> ("e2", "g2"); "e2@g1" / "lin:...@g1" ->
    (.., "g1"); "lin:..." -> "g2". A further "@f16" (fused and wide forms only) is kept as a suffix: "e2@f16" ->
    ("e2", "fused@f16"), "e3@wide2@f16" -> ("e3", "wide2@f16"), "e0@w2@f16" -> ("e0", "fused2@f16")."""
    base, _, variant = stage.partition("@")
    variant, _, f16 = variant.partition("@") if variant != "f16" else ("", "", "f16")
    if f16 not in ("", "f16"):
        raise ValueError(f"unknown model K stage suffix in {stage!r}")
    if variant == "split":
        variant = "g2"
    if variant == "w2":
        variant = "fused2"
    variant = variant or ("fused" if base in STAGE else "g2")
    if f16:
        if not (variant.startswith("fused") or variant.startswith("wide")) or base not in STAGE:
            raise ValueError(f"{stage!r}: @f16 applies to the fused and wide stage kernels only")
        variant += "@f16"
    return base, variant


def pack(stage, tensors, d_row="gfx12"):
    """Runner.pack_weights for "k.<...>" stages (stage without the "k." prefix)."""
    base, _ = _split(stage)
    if base in STAGE:
        return k_pack.pack_stage(base, tensors, d_row)
    kind, _, rest = base.partition(":")
    if kind == "block":
        C, heads, mix = rest.split(":")
        return k_pack.pack_block(tensors, int(C), int(heads), mix, d_row)
    if kind == "lin":
        return k_pack.pack_linear(tensors["W"], tensors.get("b"), d_row)
    if kind in ("act", "win", "input", "output"):
        return b""
    raise ValueError(f"unknown model K stage {stage!r}")


class _Plan:
    def __init__(self, dev):
        self.dev = dev
        self.plan = {"bufs": {}, "sizes": {}, "launches": [], "outputs": [], "decoders": {}}

    def alloc(self, name, nbytes, data=None):
        nbytes = max(int(nbytes), 256)
        self.plan["bufs"][name] = self.dev.malloc(nbytes)
        self.plan["sizes"][name] = nbytes
        if data is not None:
            self.dev.upload(self.plan["bufs"][name], data)
        return self.plan["bufs"][name]

    def output(self, name, shape, decoder=_decode_f16):
        self.plan["outputs"].append((name, tuple(shape)))
        self.plan["decoders"][name] = decoder

    def release(self):
        for p in self.plan["bufs"].values():
            self.dev.free(p)
        self.plan["bufs"].clear()


def _map(a, what):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"{what} must be [H, W, C], got {a.shape}")
    return a


def prepare(dev, stage, weights, x, offset=None, skip=None):
    """Launch plan of a "k.<...>" stage (stage without the prefix), in the format of Runner._prepare."""
    base, variant = _split(stage)
    P = _Plan(dev)
    try:
        if base in STAGE:
            _prepare_stage(P, STAGE[base], weights, x, offset, skip, variant)
        elif base.startswith("block:"):
            _, C, heads, mix = base.split(":")
            _prepare_block(P, int(C), int(heads), mix, weights, x, offset)
        elif base.startswith("lin:"):
            _prepare_lin(P, base, weights, x, skip, variant)
        elif base == "win":
            _prepare_win(P, x)
        elif base == "act":
            _prepare_act(P)
        elif base == "output":
            _prepare_output(P, x, compact=variant == "compact")  # "output@compact": k_out_pixel_c on the compact head
        elif base == "input":
            _prepare_input(P, x, exact=variant == "exact")  # "input@exact": the bit-exact build
        else:
            raise ValueError(f"unknown model K stage {stage!r}")
    except Exception:
        P.release()
        raise
    return P.plan


def _prepare_stage(P, st, weights, x, offset, skip, variant):
    if len(weights) != k_pack.stage_sections(st)["_size"]:
        raise ValueError(f"{st['name']}: weight blob has {len(weights)} bytes, expected "
                         f"{k_pack.stage_sections(st)['_size']}")
    x = _map(x, "x")
    C = st["C"]
    if offset is None:
        raise ValueError(f"{st['name']}: the window offset (ox, oy) is required (spec 44)")
    offset = tuple(int(v) for v in offset)
    if st["kind"] == "dec":
        if skip is None:
            raise ValueError(f"{st['name']}: the skip map S{st['level']} is required")
        skip = _map(skip, "skip")
        lv = (skip.shape[1], skip.shape[0])
        if skip.shape[2] != C or x.shape[2] != st["C_prev"] or (cdiv(lv[0], 2), cdiv(lv[1], 2)) != (x.shape[1],
                                                                                                     x.shape[0]):
            raise ValueError(f"{st['name']}: x {x.shape} / skip {skip.shape} do not fit the stage")
    else:
        lv = (x.shape[1], x.shape[0])
        cin = 16 if st.get("embed") else C
        if x.shape[2] != cin:
            raise ValueError(f"{st['name']}: x must have {cin} channels, got {x.shape[2]}")
    n_lv = lv[0] * lv[1]
    bufs = {"in": P.alloc("in", x.size * 2, _f16_bytes(x))}
    wptr = P.alloc("weights", len(weights), bytes(weights))
    if st["kind"] == "dec":
        bufs["skip"] = P.alloc("skip", skip.size * 2, _f16_bytes(skip))
    f16 = variant.endswith("@f16")
    variant = variant[:-4] if f16 else variant
    if variant.startswith("wide"):
        sizes = {"in": (x.shape[1], x.shape[0]), "level": lv}
        ng = int(variant[4:]) if variant[4:] else KL.WIDE_DEFAULT[st["name"]]
        bufs["block"] = P.alloc("block", n_lv * C * 2)
        P.output("block", (lv[1], lv[0], C))
        if st["kind"] == "enc":
            mw, mh = cdiv(lv[0], 2), cdiv(lv[1], 2)
            bufs["merged"] = P.alloc("merged", mw * mh * st["C_next"] * 2)
            P.output("merged", (mh, mw, st["C_next"]))
        P.plan["launches"] = [KL.stage_fused_launch(st["name"], wptr, bufs, sizes, offset, wide=ng, f16=f16)]
        return
    if f16 and not KL.fusable(st["name"], offset):
        raise ValueError(f"{st['name']}@f16: the FP16-accumulating kernels need an even window offset, got {offset}")
    windows = 2 if variant == "fused2" else 1
    fused = variant in ("fused", "fused2") and KL.fusable(st["name"], offset)  # else the separate kernels
    if variant in ("fused", "fused2"):
        variant = "g2"
    if not fused and st.get("embed"):
        bufs["tmp"] = P.alloc("tmp", n_lv * C * 2)
    if not fused and st["kind"] == "dec":
        pw, ph = KL.padded_size(lv, offset)
        bufs["tmp"] = P.alloc("tmp", pw * ph * C * 2)
    bufs["block"] = P.alloc("block", n_lv * C * 2)
    P.output("block", (lv[1], lv[0], C))
    if st["kind"] == "enc":
        mw, mh = cdiv(lv[0], 2), cdiv(lv[1], 2)
        bufs["merged"] = P.alloc("merged", mw * mh * st["C_next"] * 2)
        P.output("merged", (mh, mw, st["C_next"]))
    if st.get("head"):
        bufs["head"] = P.alloc("head", n_lv * 40 * 2)
        P.output("head", (lv[1], lv[0], 40))
    sizes = {"in": (x.shape[1], x.shape[0]), "level": lv}
    if fused:
        P.plan["launches"] = [KL.stage_fused_launch(st["name"], wptr, bufs, sizes, offset, windows, f16=f16)]
    else:
        P.plan["launches"] = KL.stage_launches(st["name"], wptr, bufs, sizes, offset, variant)


def _prepare_block(P, C, heads, mix, weights, x, offset):
    x = _map(x, "x")
    if x.shape[2] != C:
        raise ValueError(f"block: x must have {C} channels")
    size = (x.shape[1], x.shape[0])
    xin = P.alloc("in", x.size * 2, _f16_bytes(x))
    wptr = P.alloc("weights", len(weights), bytes(weights))
    out = P.alloc("out", x.size * 2)
    P.output("out", x.shape)
    P.plan["launches"] = [KL.block_launch(C, heads, mix, xin, out, wptr, size, tuple(int(v) for v in offset))]


def _prepare_lin(P, base, weights, x, skip, variant):
    _, mode, shape = base.split(":")
    k, n = (int(v) for v in shape.split("x"))
    opts = x if isinstance(x, dict) else {"x": x}
    xm = _map(opts["x"], "x")
    in_size = (xm.shape[1], xm.shape[0])
    if mode == "plain":
        out_size, cout = in_size, n
    elif mode == "merge":
        out_size, cout = (cdiv(in_size[0], 2), cdiv(in_size[1], 2)), n
    else:
        out_size, cout = tuple(opts.get("out_size") or (2 * in_size[0], 2 * in_size[1])), n // 4
    xin = P.alloc("in", xm.size * 2, _f16_bytes(xm))
    wptr = P.alloc("weights", len(weights), bytes(weights))
    out = P.alloc("out", out_size[0] * out_size[1] * cout * 2)
    res = P.alloc("res", out_size[0] * out_size[1] * cout * 2, _f16_bytes(skip)) if skip is not None else None
    P.output("out", (out_size[1], out_size[0], cout))
    P.plan["launches"] = [KL.lin_blob_launch(mode, k, n, xin, wptr, out, in_size, out_size, res=res,
                                             act=bool(opts.get("act")), relu=bool(opts.get("relu")),
                                             variant=variant)]


def _prepare_win(P, x):
    xm = _map(x["x"], "x")
    h, w, c = xm.shape
    if c % 8:
        raise ValueError("win: C must be a multiple of 8")
    offset = tuple(int(v) for v in x.get("offset", (0, 0)))
    margin = int(x.get("margin", 0))
    border = x.get("border", "mirror")
    nwx, nwy = window_grid(w, h, *offset)
    s = 8 + 2 * margin
    src = P.alloc("in", xm.size * 2, _f16_bytes(xm))
    wins = P.alloc("windows", nwx * nwy * s * s * c * 2)
    back = P.alloc("back", xm.size * 2)
    P.output("windows", (nwx * nwy, s * s, c))
    P.output("back", (h, w, c))
    P.plan["launches"] = [KL.win_launch("k_win_gather", src, wins, (w, h), c, offset, margin, border),
                          KL.win_launch("k_win_scatter", wins, back, (w, h), c, offset, margin, border)]


def _prepare_act(P):
    a = P.alloc("scalar", 65536 * 2)
    b = P.alloc("packed", 65536 * 2)
    P.output("scalar", (65536,), _decode_u16)
    P.output("packed", (65536,), _decode_u16)
    P.plan["launches"] = [("k_act_table", 512, 128, [_ptr(a)]), ("k_act_table2", 256, 128, [_ptr(b)])]

    def finish(out):
        from clean.k.reference.act import act
        from clean.k.reference.f16 import all_f16
        ref = act(all_f16())
        for name in ("scalar", "packed"):
            got = out[name].view(np.float16)
            same = (got.view(np.uint16) == ref.view(np.uint16)) | (np.isnan(got) & np.isnan(ref))
            out[f"mismatch_{name}"] = int(np.count_nonzero(~same))
        return out
    P.plan["finish"] = finish


def _prepare_output(P, x, compact=False):
    """compact: the output kernel k_out_pixel_c reading the compact pixel head (k_io.compact_head of N) instead of
    k_out_pixel on the head map; no k_out_hidden then (it needs the head map), so "Hh" is not written."""
    from clean.k.runner import k_io
    from clean.k.reference.outstage import ceil16
    fin_bufs = k_io.upload_frame(P.alloc, x)
    fin = k_io.frame_in(x, fin_bufs)
    W, H, W16, H16 = fin.W, fin.H, ceil16(fin.W), ceil16(fin.H)
    N = np.asarray(x["N"], dtype=np.float32)
    if N.shape != (fin.Bh, fin.Bw, 40):
        raise ValueError(f"output: N must be [{fin.Bh}, {fin.Bw}, 40], got {N.shape}")
    nbytes = k_io.compact_head(N.astype(np.float16)).tobytes() if compact else k_io.f16_bytes(N)
    bufs = {"N": P.alloc("N", len(nbytes), nbytes),
            "L": P.alloc("L", 4 * W16 * H16 * 2, k_io.f16_bytes(np.asarray(x["L"]).reshape(2 * H16, 2 * W16))),
            "Hc": P.alloc("Hc", W16 * H16 * 8, k_io.f16_bytes(np.asarray(x["Hc"]).reshape(H16, W16, 4))),
            "table": P.alloc("table", 2048 * 16, np.ascontiguousarray(k_io.table_for(x)).tobytes()),
            "exp2tab": P.alloc("exp2tab", 65536 * 2, k_io.exp2_table().tobytes()),
            "out": P.alloc("out", W * H * 16), "Lout": P.alloc("Lout", 4 * W16 * H16 * 2),
            "Hcout": P.alloc("Hcout", W16 * H16 * 8), "Hhout": P.alloc("Hhout", fin.Bw * fin.Bh * 8)}
    for name, nbytes in k_io.output_scratch_bytes(fin.rw, fin.rh).items():
        bufs[name] = P.alloc(name, nbytes)
    P.output("out", (H, W, 4), lambda raw, shape: np.frombuffer(raw, np.float32).reshape(shape).copy())
    P.plan["outputs"] += [("Lout", (2 * H16, 2 * W16)), ("Hcout", (H16, W16, 4)), ("Hhout", (fin.Bh, fin.Bw, 4))]
    for nm in ("Lout", "Hcout", "Hhout"):
        P.plan["decoders"][nm] = _decode_f16
    P.plan["launches"] = k_io.output_launches(fin, bufs, {})
    if compact:
        P.plan["launches"] = [("k_out_pixel_c",) + ln[1:] if ln[0] == "k_out_pixel" else ln
                              for ln in P.plan["launches"] if ln[0] != "k_out_hidden"]

    def finish(out):
        return {"out": out["out"], "L": out["Lout"], "Hc": out["Hcout"], "Hh": out["Hhout"]}
    P.plan["finish"] = finish


def _prepare_input(P, x, exact=False):
    from clean.k.runner import k_io
    from clean.k.reference.outstage import ceil16
    fin_bufs = k_io.upload_frame(P.alloc, x)
    fin = k_io.frame_in(x, fin_bufs)
    W16, H16 = ceil16(fin.W), ceil16(fin.H)
    bufs = {"Hc": P.alloc("Hc", W16 * H16 * 8, k_io.f16_bytes(np.asarray(x["Hc"]).reshape(H16, W16, 4))),
            "Hh": P.alloc("Hh", fin.Bw * fin.Bh * 8, k_io.f16_bytes(np.asarray(x["Hh"]).reshape(fin.Bh, fin.Bw, 4))),
            "x0": P.alloc("x0", fin.Bw * fin.Bh * 32)}
    for name, nbytes in k_io.input_scratch_bytes(fin.rw, fin.rh).items():
        bufs[name] = P.alloc(name, nbytes)
    P.output("x0", (fin.Bh, fin.Bw, 16))
    P.plan["launches"] = k_io.input_launches(fin, bufs, exact=exact)


def _ptr(p):
    import ctypes
    return ctypes.c_void_p(p)
