# SPDX-License-Identifier: MIT
"""Convention variants of the offline RR replay (quality investigation): runs tools/rr_capture_replay.py once per
variant on the same capture, then writes a contact sheet per variant and one metric table.

    python clean/rr_f/proxy/tools/rr_replay_variants.py <capture folder> --model <model.wgcm> --out <dir>
        [--frames 300:307] [--frame 306] [--reference 300:306] [--only NAME,...] [--sheets-only] [--head strict]

Each variant changes ONE input convention against the release proxy's behaviour (the switches of
rr_capture_replay.CONVENTIONS); "base" is the release proxy itself. Everything is replay-side: the runtime and the
release proxy are not changed. The model is handed to the runtime by path only.

Per variant <out>\\<name>\\ holds the replay's PNGs and summary.json and frame_<F>_rr.npy (float16 output of the
analysed frame F). <out>\\sheets\\<name>.png: the RR output of frame F at half size, and below it two crops at full
output resolution (CROPS); <out>\\sheets\\reference.png the same for the reference (the temporal mean of the
reference frames' colour, bilinear to the output size: the camera is static there, so it is a denoised input).
<out>\\metrics.md / metrics.json: per variant, frame F:
  lum out / ref       mean linear luminance of the RR output and of the reference, whole frame
  tm R G B (clear)    mean tonemapped colour (0..255) where the layer (type 39) alpha < 0.05 ("clear" region)
  blue shift          (B/(R+G+B) of the output) - (that of the reference), clear region, tonemapped
  edge HP             mean |l - box5(l)| of the tonemapped luminance on the geometry edges of frame F (depth jumps
                      > 5 % or normal turns > 60 degrees, dilated 1 output pixel), clear region, output / reference
  edge blue           mean of (B - (R + G) / 2) on those edges minus the same on a ring 4..6 pixels away, output minus
                      reference (positive = blue outlines)
  PSNR ref            PSNR of the tonemapped output against the tonemapped reference (dB), whole frame and clear region
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rr_capture_replay as R  # noqa: E402

# name: (conventions, what it tests)
VARIANTS = {
    "base": ({}, "release proxy routing (spec defaults)"),
    "layer_drop": (dict(layer="drop"), "type 39 not routed (no layer, plain resolve)"),
    "layer_additive": (dict(layer="additive"), "type 39 as layer with alpha forced 0 (pure addition)"),
    "layer_straight": (dict(layer="straight"), "type 39 rgb multiplied by its alpha (not premultiplied)"),
    "normals_view": (dict(normals="view"), "normals rotated into view space"),
    "rough_invert": (dict(roughness="invert"), "roughness := 1 - 4th channel"),
    "albedo_srgb": (dict(albedo="srgb"), "both albedos sRGB-decoded"),
    "albedo_swap": (dict(albedo="swap"), "diffuse and specular albedo swapped"),
    "hitdist_drop": (dict(hitdist="drop"), "type 42 not routed (smv off, m2 = own motion)"),
    "g1_drop": (dict(g1="drop"), "type 58 (G1 guide) not routed"),
    "colour_x4": (dict(colour_scale=4.0), "colour and layer x4 (output / 4): auto-exposure invariance"),
    "colour_x0.25": (dict(colour_scale=0.25), "colour and layer x0.25 (output x 4)"),
    "depth_inv0": (dict(depth_inverted="0"), "creation: depth not inverted"),
    "mv_jittered1": (dict(mv_jittered="1"), "creation: motion vectors jittered"),
    "mv_neg": (dict(mv_scale="-1,-1"), "motion vectors negated"),
    "mv_zero": (dict(mv_scale="0,0"), "motion vectors zero"),
    "jitter_neg": (dict(jitter_sign="-1,-1"), "jitter negated (x and y)"),
    "jitter_negx": (dict(jitter_sign="-1,1"), "jitter x negated"),
    "jitter_negy": (dict(jitter_sign="1,-1"), "jitter y negated"),
    # diagnostics (not conventions): how the output colour follows the input colour
    "probe_grey": (dict(colour_probe="grey"), "diagnostic: colour input R = G = B = its luminance"),
    "probe_swap_rb": (dict(colour_probe="swap_rb"), "diagnostic: colour input with R and B exchanged"),
    "probe_all_grey": (dict(colour_probe="grey", albedo="grey", layer="drop"),
                       "diagnostic: colour and both albedos grey (luminance), no layer"),
    "probe_rotate": (dict(colour_probe="rotate"), "diagnostic: colour input (r, g, b) := (g, b, r)"),
    "probe_zero_r": (dict(colour_probe="zero_r"), "diagnostic: colour input R := 0"),
    "probe_zero_g": (dict(colour_probe="zero_g"), "diagnostic: colour input G := 0"),
    "probe_zero_b": (dict(colour_probe="zero_b"), "diagnostic: colour input B := 0"),
}
# full-resolution crops of the sheets (output pixels x, y, w, h): ceiling grid / tiles (outlines, colour) and the
# centre of the room (the layer)
CROPS = ((1700, 0, 800, 450), (700, 420, 800, 450))


def box(img, r):
    """Box mean of radius r (window 2r+1, edges clamped) of a 2D float image."""
    p = np.pad(img.astype(np.float64), r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k)).astype(np.float32)


def dilate(mask, r):
    return box(mask.astype(np.float32), r) > 1e-6


def geometry_edges(frame, wo, ho):
    """Depth jumps > 5 % or normal turns > 60 degrees between 4-neighbours, render resolution -> output (nearest)."""
    h, d = frame["inputs"][R.BT["DEPTH"]] if R.BT["DEPTH"] in frame["inputs"] else frame["inputs"][R.BT["LINEAR_DEPTH"]]
    z = R.decode(h, d, depth=True)[..., 0].astype(np.float64)
    lz = np.log(np.maximum(np.abs(z), 1e-12))
    nt = R.BT["NORMAL_ROUGHNESS"] if R.BT["NORMAL_ROUGHNESS"] in frame["inputs"] else R.BT["NORMALS"]
    n = R.decode(*frame["inputs"][nt])[..., :3].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)
    e = np.zeros(z.shape, bool)
    for ax in (0, 1):
        dz = np.abs(np.diff(lz, axis=ax)) > np.log(1.05)
        dn = (n * np.roll(n, -1, axis=ax)).sum(2)
        dn = (dn < np.cos(np.radians(60)))
        dn = dn[:-1] if ax == 0 else dn[:, :-1]
        m = dz | dn
        if ax == 0:
            e[:-1] |= m
            e[1:] |= m
        else:
            e[:, :-1] |= m
            e[:, 1:] |= m
    return R.nearest(e[..., None].astype(np.float32), wo, ho)[..., 0] > 0.5


def clear_region(frame, wo, ho):
    t = R.BT["COLOUR_BEFORE_PARTICLES"]
    if t not in frame["inputs"]:
        return np.ones((ho, wo), bool)
    a = R.decode(*frame["inputs"][t])[..., 3]
    return R.nearest(a[..., None], wo, ho)[..., 0] < 0.05


def metrics(out_lin, ref_lin, edges, ring, clear):
    to, tr = R.tonemap(out_lin).astype(np.float32), R.tonemap(ref_lin).astype(np.float32)
    lum = lambda t: 0.2126 * t[..., 0] + 0.7152 * t[..., 1] + 0.0722 * t[..., 2]
    lo, lr = lum(to), lum(tr)
    hp = lambda l: np.abs(l - box(l, 2))
    e = edges & clear
    g = ring & clear
    bl = lambda t: t[..., 2] - 0.5 * (t[..., 0] + t[..., 1])
    frac = lambda t: (t[..., 2].sum() / max(t[..., :3].sum(), 1e-9))
    c = clear
    return dict(
        lum_out=R.luminance(out_lin), lum_ref=R.luminance(ref_lin),
        tm_rgb_clear=[round(float(to[c][:, k].mean()), 1) for k in range(3)],
        tm_rgb_clear_ref=[round(float(tr[c][:, k].mean()), 1) for k in range(3)],
        blue_shift=float(frac(to[c]) - frac(tr[c])),
        edge_hp_ratio=float(hp(lo)[e].mean() / max(hp(lr)[e].mean(), 1e-9)),
        edge_hp_out=float(hp(lo)[e].mean()), edge_hp_ref=float(hp(lr)[e].mean()),
        edge_blue=float((bl(to)[e].mean() - bl(to)[g].mean()) - (bl(tr)[e].mean() - bl(tr)[g].mean())),
        psnr_ref=R.psnr(to, tr), psnr_ref_clear=R.psnr(to[c], tr[c]),
    )


def label(img, text):
    """A text strip above img (PIL when present, else a plain strip)."""
    strip = np.full((28, img.shape[1], 3), 24, np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(strip)
        ImageDraw.Draw(im).text((6, 7), text, fill=(235, 235, 235))
        strip = np.asarray(im)
    except Exception:
        pass
    return np.concatenate([strip, img], axis=0)


def sheet(lin, text):
    t = R.tonemap(lin)
    ho, wo = t.shape[:2]
    half = t[::2, ::2]
    crops = [t[y:y + h, x:x + w] for x, y, w, h in CROPS]
    row = R.side_by_side(crops[0], crops[1])
    pad = np.full((row.shape[0], max(half.shape[1] - row.shape[1], 0), 3), 32, np.uint8)
    row = np.concatenate([row, pad], axis=1)[:, :half.shape[1]]
    w = max(half.shape[1], row.shape[1])
    return label(np.concatenate([half, np.full((4, w, 3), 32, np.uint8), row], axis=0), text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture")
    ap.add_argument("--model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="300:307")
    ap.add_argument("--frame", type=int, default=306, help="the analysed evaluate")
    ap.add_argument("--reference", default="300:306", help="evaluates averaged into the reference (static camera)")
    ap.add_argument("--only", help="comma list of variant names (default: all)")
    ap.add_argument("--extra", action="append", default=[],
                    help="an extra variant NAME:key=value[;key=value] (e.g. combo:layer=drop;jitter_sign=-1,-1)")
    ap.add_argument("--sheets-only", action="store_true", help="no GPU runs: sheets and metrics of existing runs")
    ap.add_argument("--head", default="strict", choices=("strict", "fast"))
    a = ap.parse_args()
    variants = dict(VARIANTS)
    for e in a.extra:
        name, _, kv = e.partition(":")
        variants[name] = (dict(x.split("=", 1) for x in kv.split(";") if x), "combination: " + kv)
    names = a.only.split(",") if a.only else list(variants)
    f0, f1 = (int(x) for x in a.frames.split(":"))
    r0, r1 = (int(x) for x in a.reference.split(":"))
    os.makedirs(os.path.join(a.out, "sheets"), exist_ok=True)
    for name in names:
        kw, what = variants[name]
        d = os.path.join(a.out, name)
        if not a.sheets_only:
            print(f"=== {name}: {what}", flush=True)
            R.run(a.capture, a.model, d, first=f0, last=f1, head=a.head, conv=R.conventions(**kw), raw_frames={a.frame})
    frames = R.load_capture(a.capture, r0, r1)
    fr = R.load_capture(a.capture, a.frame, a.frame)[0]
    wo, ho = R.output_size(fr)
    ref = R.bilinear(np.mean([R.colour_input(f) for f in frames], axis=0), wo, ho)
    edges = dilate(geometry_edges(fr, wo, ho), 1)
    ring = dilate(edges, 6) & ~dilate(edges, 3)
    clear = clear_region(fr, wo, ho)
    R.write_png(os.path.join(a.out, "sheets", "reference.png"),
                sheet(ref, f"reference: mean colour of {r0}..{r1}, bilinear (denoised input)"))
    R.write_png(os.path.join(a.out, "sheets", "masks.png"),
                np.stack([edges * 255, ring * 160, clear * 96], axis=2).astype(np.uint8)[::2, ::2])
    rows = {}
    for name in names:
        kw, what = variants[name]
        p = os.path.join(a.out, name, f"frame_{a.frame:06d}_rr.npy")
        if not os.path.isfile(p):
            print(f"{name}: no {p}")
            continue
        o = np.load(p).astype(np.float32)
        m = metrics(o, ref, edges, ring, clear)
        m["what"] = what
        m["conventions"] = R.changed(R.conventions(**kw))
        rows[name] = m
        R.write_png(os.path.join(a.out, "sheets", f"{name}.png"), sheet(o, f"{name}: {what}"))
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(dict(frame=a.frame, reference=[r0, r1], crops=CROPS, variants=rows), fh, indent=1)
    lines = [f"Frame {a.frame}; reference = mean colour of {r0}..{r1} (bilinear); 'clear' = layer alpha < 0.05 "
             f"({clear.mean() * 100:.0f} % of the frame); edges {edges.mean() * 100:.1f} % of the frame.", "",
             "| variant | lum out / ref | tm R G B clear (ref) | blue shift | edge HP out/ref | edge blue | PSNR ref "
             "(clear) dB |", "|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        lines.append(f"| {name} | {m['lum_out']:.4f} / {m['lum_ref']:.4f} | {' '.join(str(x) for x in m['tm_rgb_clear'])}"
                     f" ({' '.join(str(x) for x in m['tm_rgb_clear_ref'])}) | {m['blue_shift']:+.4f} | "
                     f"{m['edge_hp_ratio']:.2f} | {m['edge_blue']:+.2f} | {m['psnr_ref']:.2f} ({m['psnr_ref_clear']:.2f}) |")
    with open(os.path.join(a.out, "metrics.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
