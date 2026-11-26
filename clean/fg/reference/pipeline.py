# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the per-call schedule of the light path (preset C) and its CPU executor, written for
this project.

The schedule (Host) is shared: Host.call() turns one evaluate call into a list of operations, and an executor runs
them, this module's LitePipeline on the CPU (stages_common.py, lite.py) and clean/fg/runner/fg_lite_d3d12_check.py on
the D3D12 device. Words: a game frame is everything between two last-index calls; the first call of a game frame runs
the once-per-game-frame stages; history means at least one completed game frame since creation or the last reset.

Once per game frame (first call): reset check, history clears (no history), menu detection, R3 downsample / UI
estimate, R4 motion and depth, R6 previous-to-current scatter, unpack, hole rule, pull and push, R7 hole mask, R8 the
quality image (no history) or net1's input, net1 and its output (history).
Once per call (index i = 1 .. N, t = i / (N + 1)): with history the two intermediate-time scatters, the blend weights,
the light blend and the menu output; without history the generated frame is the current colour.
History advances, and the current / previous image pairs swap, only on the last-index call; N changes nothing else.

Refused here (not part of the light path): the full path (presets A / B) and the distortion field.
"""
import numpy as np

from clean.fg.reference import config as CFG
from clean.fg.reference import lite as LITE
from clean.fg.reference import stages_common as S

N_MAX = 5           # largest multi-frame count (6x)


class Settings:
    """Create-time and layered settings with their defaults; lite=True is preset C."""

    def __init__(self, lite=True, ui_recomposition=False, menu_detection=False, lite_l1=False, lite_l2=True,
                 ignore_distortion=False, ignore_hudless=False, ignore_ui=False, alpha_threshold=0.7, fraction=0.7,
                 kmax=600.0, tau=40.0, s_k=1.0, extra_index=False):
        self.lite, self.ui_recomposition, self.menu_detection = lite, ui_recomposition, menu_detection
        self.lite_l1, self.lite_l2 = lite_l1, lite_l2
        self.ignore_distortion, self.ignore_hudless, self.ignore_ui = ignore_distortion, ignore_hudless, ignore_ui
        self.alpha_threshold, self.fraction = np.float32(alpha_threshold), np.float32(fraction)
        self.kmax, self.tau, self.s_k = np.float32(kmax), np.float32(tau), np.float32(s_k)
        self.extra_index = extra_index


def derived_flags(st, hudless, ui4, ui1, distortion=False):
    """The six derived flags from which inputs are supplied."""
    dist = bool(distortion) and not st.ignore_distortion
    hud = bool(hudless) and not st.ignore_hudless
    uia = (bool(ui4) or bool(ui1)) and not st.ignore_ui
    uirp = (not st.lite) and hud and uia and st.ui_recomposition
    lf1 = st.lite and st.lite_l1 and hud and uia
    lf2 = st.lite and st.lite_l2 and hud and not lf1
    return {"distortion": dist, "hudless": hud, "ui_alpha": uia, "ui_recomp": uirp, "lite1": lf1, "lite2": lf2}


def r3_flags(f):
    return {"F1": f["ui_recomp"] or f["lite1"], "A": f["ui_alpha"], "Hd": f["hudless"], "L1": f["lite1"],
            "L2": f["lite2"]}


class Geometry:
    """Grid sizes of one frame size: the net1 image grid (d = 4 above 1440 lines, else 2), its cell grid with the
    centring pad, the scatter grid (d' = 4 above 2880 lines, else 2) and its pyramid. d / dp override for tests."""

    def __init__(self, wo, ho, d=None, dp=None):
        if not (4 <= wo <= 8192 and 4 <= ho <= 7680):
            raise ValueError("frame size outside 4 x 4 .. 8192 x 7680")
        self.wo, self.ho = int(wo), int(ho)
        self.d = d if d is not None else (4 if ho > 1440 else 2)
        self.wd, self.hd = wo // self.d, ho // self.d
        cw, ch = self.wd // 2, self.hd // 2
        self.W0, self.H0 = 32 * (-(-cw // 32)), 32 * (-(-ch // 32))
        self.pad_l, self.pad_t = (self.W0 - cw) // 2, (self.H0 - ch) // 2
        self.dp = dp if dp is not None else (4 if ho > 2880 else 2)
        self.w1, self.h1 = wo // self.dp, ho // self.dp
        self.levels = S.pyramid_levels(self.w1, self.h1)
        self.top = S.pull_top(self.levels)


class Host:
    """The control state of one frame-generation instance: counters, the previous game frame's flags, whether a menu
    detector ran, and the frozen light-path flags of the current game frame."""

    def __init__(self, settings, geom):
        if not settings.lite:
            raise NotImplementedError("the full path (presets A / B) is not part of this reference")
        self.st, self.g = settings, geom
        self.history = 0
        self.since_last = 0
        self.prev_flags = None
        self.flags = None
        self.detect_ran_prev = False
        self.hist = False
        self.case = None

    def call(self, n, i, hudless=False, ui4=False, ui1=False, reset=False, distortion=False, out_subrect=False):
        """The operations of one call (list of (name, params)), and whether this was the last index."""
        CFG.check_multi_frame(n, i, n_max=N_MAX, extra_index=self.st.extra_index)
        t = CFG.intermediate_time(n, i)
        last = CFG.last_index(i, n + 1 if self.st.extra_index else n)
        ops = []
        first = self.since_last == 0
        if first:
            f = derived_flags(self.st, hudless, ui4, ui1, distortion)
            if f["distortion"]:
                raise NotImplementedError("the distortion field is not part of this reference")
            self.flags = f
            if reset or (self.prev_flags is not None and f != self.prev_flags):
                self.history, self.prev_flags = 0, None
            self.hist = self.history > 0
            rf = r3_flags(f)
            self.case = S.r3_case(rf)
            if not self.hist:
                ops.append(("clear", {"role": "Cp", "value": (0.0, 0.0, 0.0, 1.0)}))
                ops.append(("clear", {"role": "Kp", "value": (1.0,)}))
                ops.append(("clear", {"role": "Zp", "value": (0,)}))
                ops.append(("clear", {"role": "m", "value": (0.5,)}))
                if rf["F1"]:
                    ops.append(("clear", {"role": "Uc", "value": (0.0, 0.0, 0.0, 1.0)}))
                    ops.append(("clear", {"role": "Up", "value": (0.0, 0.0, 0.0, 1.0)}))
            base = 0 if self.hist else 1
            kind = None
            if self.st.menu_detection:
                kind = "alpha" if ui4 else ("hudless" if hudless else None)
            if kind:
                ops.append(("detect", {"kind": kind, "hysteresis_off": (not self.hist) and not self.detect_ran_prev,
                                       "base": base, "channel": 0 if ui1 else 3}))
            else:
                ops.append(("set_disable", {"value": base}))
            self.detect_ran_prev = kind is not None
            ops.append(("downsample", {"flags": rf, "case": self.case}))
            ops.append(("mvec", {}))
            ops.append(("fill64", {"role": "S"}))
            ops.append(("p2c_scatter", {}))
            ops.append(("p2c_unpack", {}))
            ops.append(("p2c_inpaint", {}))
            ops.append(("p2c_pull", {"top": self.g.top}))
            ops.append(("p2c_push", {"top": self.g.top}))
            ops.append(("hole_mask", {}))
            if self.hist:
                ops.append(("net1_input", {}))
                ops.append(("net1", {}))
                ops.append(("net1_output", {"ui_forcing": (f["hudless"] or f["ui_alpha"]) and not rf["F1"]}))
            else:
                ops.append(("quality_init", {}))
        f = self.flags
        wrote = False
        if self.hist:
            ops.append(("fill64", {"role": "MA"}))
            ops.append(("fill64", {"role": "MB"}))
            ops.append(("interm", {"t": t}))
            ops.append(("weights", {}))
            if out_subrect:
                ops.append(("prefill", {}))
            ops.append(("lite_blend", {"t": t, "flag1": f["lite1"], "flag2": f["lite2"]}))
            wrote = True
            if self.st.menu_detection and (ui4 or hudless):
                ops.append(("menu_output", {"channel": 0 if ui1 else 3}))
        elif out_subrect:
            ops.append(("prefill", {}))
            wrote = True
        if not wrote:
            ops.append(("final_copy", {}))
        if last:
            self.history += 1
            self.since_last = 0
            self.prev_flags = f
            pairs = [("Cc", "Cp"), ("Dc", "Dp"), ("Zc", "Zp"), ("Kc", "Kp"), ("Qc", "Qp"), ("Hc", "Hp")]
            if f["lite2"]:
                pairs.append(("Lc", "Lp"))
            if f["ui_recomp"] or f["lite1"]:
                pairs.append(("Uc", "Up"))
            if self.case in ("A", "H"):
                pairs.append(("Um", "Ump"))
            ops.append(("swap", {"pairs": pairs}))
        else:
            self.since_last += 1
        return ops, {"t": t, "last": last, "first": first, "history": self.hist}


# ---- the CPU executor --------------------------------------------------------------------------------------------

def alloc(shape, dtype, fill=0):
    """An array of the given shape and type with every byte = fill."""
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return np.full(n, fill, np.uint8).view(dtype).reshape(shape)


def image_shapes(g):
    """role -> (shape, dtype) of every internal image."""
    fr, nd, sc = (g.ho, g.wo), (g.hd, g.wd), (g.h1, g.w1)
    f16, f32, u32 = np.float16, np.float32, np.uint32
    out = {}
    for r in ("Cc", "Cp", "Lc", "Lp", "Uc", "Up"):
        out[r] = (fr + (4,), f16)
    for r in ("Dc", "Dp"):
        out[r] = (nd + (4,), f16)
    for r in ("Ld", "m", "Um", "Ump", "Hc", "Hp", "Om"):
        out[r] = (nd + (1,), f16)
    for r in ("Qc", "Qp", "NA", "NB"):
        out[r] = (nd + (2,), f16)
    out["Mv"] = (sc + (2,), f32)
    out["Vf"] = (sc + (2,), f32)
    for r in ("Zc", "Zp", "Fl", "G"):
        out[r] = (sc, u32)
    for r in ("Kc", "Kp", "E"):
        out[r] = (sc + (1,), f16)
    for r in ("F", "W"):
        out[r] = (sc + (2,), f16)
    for r in ("S", "MA", "MB"):
        out[r] = ((g.h1 * g.w1,), np.int64)
    for l in range(1, g.top + 1):
        out["P%d" % l] = (g.levels[l][::-1] + (3,), f16)
    out["cells"] = ((g.H0, g.W0, 48), f16)
    out["n1out"] = ((g.H0, g.W0, 16), f16)
    return out


class CallInputs:
    """One call's application inputs. Textures are arrays [h, w, c] (colour, hudless, ui4: float16 x 4; ui1: float16
    x 1; depth: float32 x 1; motion: float32 x 2), each with a stages_common.Region. out_size / out_offset: the
    generated frame's texture size and the base offset of its Wo x Ho extent."""

    def __init__(self, colour, creg, depth, dreg, motion, mreg, clip_to_prev, prev_to_clip, n=1, i=1, hudless=None,
                 hreg=None, ui4=None, u4reg=None, ui1=None, u1reg=None, depth_inverted=False, mvec_scale=(1.0, 1.0),
                 reset=False, out_size=None, out_offset=(0, 0)):
        self.colour, self.creg, self.depth, self.dreg, self.motion, self.mreg = colour, creg, depth, dreg, motion, mreg
        self.clip_to_prev = np.asarray(clip_to_prev, np.float32).reshape(16)
        self.prev_to_clip = np.asarray(prev_to_clip, np.float32).reshape(16)
        self.n, self.i = n, i
        self.hudless, self.hreg, self.ui4, self.u4reg, self.ui1, self.u1reg = hudless, hreg, ui4, u4reg, ui1, u1reg
        self.depth_inverted, self.mvec_scale, self.reset = bool(depth_inverted), mvec_scale, reset
        self.out_size, self.out_offset = out_size, out_offset

    def alpha_source(self):
        """(texture, region, channel) of the UI alpha: the 1-channel input when supplied, else the 4-channel one."""
        if self.ui1 is not None:
            return self.ui1, self.u1reg, 0
        if self.ui4 is not None:
            return self.ui4, self.u4reg, 3
        return None, None, 0


def exact_net1(params):
    """net1 as the bit-exact CPU reference (clean/fg/reference/net1.py): cells [H0, W0, 48] -> [H0, W0, 16]."""
    from clean.fg.reference import net1 as N1

    def run(cells):
        out = N1.run(cells.astype(np.float64).transpose(2, 0, 1), params)["out"]
        return np.asarray(out, np.float64).astype(np.float16).transpose(1, 2, 0).copy()
    return run


class LitePipeline:
    """The light path on the CPU. net1: a callable cells [H0, W0, 48] float16 -> [H0, W0, 16] float16. fill: the byte
    every internal image starts with (0, or the GPU check's sentinel)."""

    def __init__(self, settings, geom, net1, fill=0):
        self.st, self.g, self.net1 = settings, geom, net1
        self.host = Host(settings, geom)
        self.img = {r: alloc(s, t, fill) for r, (s, t) in image_shapes(geom).items()}
        self.detres = np.zeros(4, np.uint32)
        self.disable = 0
        self.fill = fill
        self.log = []

    def evaluate(self, c):
        """Runs one call; returns {"out": generated frame (out texture), "disable": the generation-disabled word,
        "ops": the schedule, "info": ...}."""
        g, st, I = self.g, self.st, self.img
        if (c.creg.ew, c.creg.eh) != (g.wo, g.ho):
            raise ValueError("the colour region's extent must be the frame size")
        osz = c.out_size or (c.colour.shape[1], c.colour.shape[0])
        ox, oy = c.out_offset
        subrect = (ox, oy) != (0, 0) or tuple(osz) != (g.wo, g.ho)
        out = alloc((osz[1], osz[0], 4), np.float16, self.fill)
        ops, info = self.host.call(c.n, c.i, hudless=c.hudless is not None, ui4=c.ui4 is not None,
                                   ui1=c.ui1 is not None, reset=c.reset, out_subrect=subrect)
        steps = S.depth_steps(c.dreg)
        alpha_img, areg, ach = c.alpha_source()
        for name, p in ops:
            if name == "clear":
                a = I[p["role"]]
                if a.dtype == np.uint32:
                    a[...] = np.uint32(p["value"][0])
                else:
                    a[...] = np.asarray(p["value"], a.dtype)
            elif name == "detect":
                if p["kind"] == "alpha":
                    cnt = S.detect_count_alpha(alpha_img, areg, ach, st.alpha_threshold)
                else:
                    cnt = S.detect_count_hudless(c.colour, c.creg, c.hudless, c.hreg)
                det = S.detect_decide(p["kind"], cnt, st.fraction)
                self.detres = S.detect_update(self.detres, p["kind"], det, p["hysteresis_off"])
                self.disable = p["base"] | (1 if self.detres[0] != 0 else 0)
                self.log.append(("detect", p["kind"], cnt, det))
            elif name == "set_disable":
                self.disable = p["value"]
            elif name == "downsample":
                S.downsample_ui(c.colour, c.creg, c.hudless, c.hreg, alpha_img, areg, ach, I["Cp"], p["flags"], g.d,
                                g.wd, g.hd, I)
            elif name == "mvec":
                S.mvec_process(c.depth, c.dreg, c.motion, c.mreg, g.w1, g.h1, c.depth_inverted, st.s_k,
                               c.mvec_scale[0], c.mvec_scale[1], I)
            elif name == "fill64":
                I[p["role"]][...] = S.EMPTY64
            elif name == "p2c_scatter":
                S.p2c_scatter(I["Zc"], I["Mv"], c.clip_to_prev, g.ho, st.s_k, c.depth_inverted, steps, I["S"], I["Kc"])
            elif name == "p2c_unpack":
                S.p2c_unpack(I["S"], I["Zp"], I["Kp"], I["Mv"], c.prev_to_clip, st.s_k, c.depth_inverted, steps,
                             st.kmax, st.tau, I)
            elif name == "p2c_inpaint":
                S.p2c_inpaint(I["Fl"], I["G"])
            elif name == "p2c_pull":
                for l, a in S.p2c_pull(I["G"], I["Vf"], p["top"]).items():
                    I["P%d" % l][...] = a
            elif name == "p2c_push":
                lv = {l: I["P%d" % l].copy() for l in range(1, p["top"] + 1)}
                S.p2c_push(lv, p["top"], I["G"], I["F"])
                for l in range(1, p["top"] + 1):
                    I["P%d" % l][...] = lv[l]
            elif name == "hole_mask":
                S.hole_mask(I["F"], g.wd, g.hd, I["Hc"])
            elif name == "quality_init":
                S.quality_init(I["Dc"], I["Qc"])
            elif name == "net1_input":
                S.net1_input(I["Ld"], I["Dc"], I["Dp"], I["F"], I["Mv"], I["E"], I["Hc"], I["Hp"], I["Qp"], I["m"],
                             g.W0, g.H0, g.pad_l, g.pad_t, I)
            elif name == "net1":
                I["n1out"][...] = self.net1(I["cells"].copy())
            elif name == "net1_output":
                S.net1_output(I["n1out"], I["Om"], I["Um"], I["Ump"], g.wd, g.hd, g.pad_l, g.pad_t, p["ui_forcing"], I)
            elif name == "interm":
                da, ka, db, kb = S.interm_inputs(I["Mv"], I["Zc"], I["F"], I["Zp"], p["t"], st.s_k, c.depth_inverted)
                _, T2 = S.thresholds(g.ho)
                S.interm_scatter(da, ka, T2, st.tau, True, I["MA"])
                S.interm_scatter(db, kb, T2, st.tau, False, I["MB"])
            elif name == "weights":
                S.blend_weights(I["MA"], I["MB"], I["NA"], I["NB"], g.w1, g.h1, I["W"])
            elif name == "prefill":
                S.copy4(out, 0, 0, c.colour, 0, 0, osz[0], osz[1])
            elif name == "lite_blend":
                LITE.lite_blend(I["Cp"], I["Cc"], I["MA"], I["MB"], I["W"], g.w1, g.h1, p["t"], p["flag1"], p["flag2"],
                                I["Up"], I["Uc"], I["Lp"], I["Lc"], out, ox, oy)
            elif name == "menu_output":
                S.menu_output(self.detres, c.colour, c.creg, c.hudless, c.hreg, alpha_img, areg, ach, out, ox, oy,
                              g.wo, g.ho)
            elif name == "final_copy":
                S.copy4(out, ox, oy, c.colour, c.creg.bx, c.creg.by, g.wo, g.ho)
            elif name == "swap":
                for a, b in p["pairs"]:
                    I[a], I[b] = I[b], I[a]
            else:
                raise ValueError("unknown operation %r" % name)
        return {"out": out, "disable": int(self.disable), "ops": ops, "info": info}
