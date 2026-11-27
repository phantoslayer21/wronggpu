# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): net2 (the 50-blob network, sub-nets A and B) end to end, as a CPU reference with the
delivered arithmetic.

Written for this project from the spec side's part 3 (net2). Everything here is fixed; nothing is an argument.
Tensors are float16 values held in float64 arrays, laid out [H, W, C] (channel innermost), as the original stores them.

- Grid: from the frame size Wo x Ho (4..8192 x 4..7680), k is the smallest power of two with floor(Ho / k) <= 480, and
  W2 = 16 ceil(Wo / 16k), H2 = 16 ceil(Ho / 16k) (grid()). Sub-net A runs at W2 / 2, / 4, / 8; sub-net B at / 2, / 4.
- Inputs: Ia (1 channel), Ib and Ic (4 channels each), all W2 x H2, and one single-precision scalar f.
- 28 launches (NET2_LAUNCHES, as data): input merge, L1..L3 (convolution, activation, 2 x 2 average pool), the residual
  block L4..L11, L12 (2x upsample, convolution 64 -> 96, three groups of 32), L13 (three heads sharing one 8-output
  convolution, each on its own 2x-upsampled group), a weightless 2x upsample, the warp-merge, L14, L15, the residual block
  L16..L23, L24 (2x upsample, 32 -> 48, three groups of 16), L25 (three heads plus a residual on the upsampled A heads).
- Convolution (all 3 x 3, stride 1, zero border outside the conv's own input grid, cross-correlation over the natural
  weights [out, in, ky, kx]): float16 accumulator started at the bias; chunks taken tap by tap in row-major (ky, kx)
  order and, inside a tap, by ascending blocks of 16 input channels (a last block of fewer channels is one shorter
  chunk); each chunk's dot product is exact and is added to the accumulator with one float16 rounding.
- Activation: leaky y = max(h(s v), v) with s = h(0.01) (bits 0x211F), then a clamp to [-6, 6], except: L12 group 0
  (leaky only), L13 head 0 (nothing), L13 heads 1 and 2 (clamp only), L24 (leaky only), L25 head 0 (nothing), L25 heads
  1 and 2 (clamp only).
- Average pool: p = h(h(h(a00 + a01) + a10) + a11), then h(p * 0.25).
- 2x bilinear upsample, half-pixel centres: per axis s = (o - 0.5) * 0.5 in single precision, i0 = trunc(s),
  i1 = trunc(s + 1), both n - 1 when i1 >= n and both 0 when s < 0, w1 = h(s - i0), w0 = h((i0 - s) + 1); then
  top = fma(wx1, P01, h(wx0 P00)), bottom likewise, out = fma(wy1, bottom, h(wy0 top)), where fma rounds once.
- Residual units: t = h(act(conv(x)) + 0), x' = h(act(conv(t)) + x): the clamp comes before the add, nothing after it.
- Warp-merge: Ib and Ic sampled bilinearly (clamp to edge) at the pixel plus twice sub-net A's two vectors (warp_sample).
- Output: one 8-channel W2 x H2 tensor: h(c0 + 2 U4) (one rounding), h(clamp(c1) + U1), h(clamp(c2) + U3).

Signed zeros: products, sums and fused multiply-adds outside the convolution chunks follow IEEE (an exact zero sum is -0
only when both terms are -0). A convolution chunk is exact integer arithmetic (fp16acc), so an exactly zero
accumulator is +0, as in net1. Non-finite values follow IEEE where the formulas define them; the warp's sample position
must be finite (an infinite or NaN vector component has no defined pixel).
"""
import numpy as np

from clean.fg import layout as L
from clean.fg.reference.fp16acc import acc_matmul16, h

LEAKY_SLOPE = float(np.uint16(0x211F).view(np.float16))   # h(0.01) = 0.01000213623046875
CLAMP = 6.0
MAX_ROW_BAND = 32768                                       # pixels per matrix product in conv3 (memory bound)
FRAME_W = (4, 8192)
FRAME_H = (4, 7680)


# ---- grid ---------------------------------------------------------------------------------------------------------

def grid(wo, ho):
    """(k, W2, H2) for a frame of wo x ho pixels."""
    if not (FRAME_W[0] <= wo <= FRAME_W[1] and FRAME_H[0] <= ho <= FRAME_H[1]):
        raise ValueError("frame %d x %d outside %d..%d x %d..%d" % (wo, ho, FRAME_W[0], FRAME_W[1], *FRAME_H))
    k = 1
    while ho // k > 480:
        k *= 2
    return k, 16 * -(-wo // (16 * k)), 16 * -(-ho // (16 * k))


def padding(wo, ho):
    """(right, bottom) padding of the frame to a multiple of 16 k (left and top are 0)."""
    k, _, _ = grid(wo, ho)
    q = 16 * k
    return q * -(-wo // q) - wo, q * -(-ho // q) - ho


def extents(W2, H2):
    """{buffer: (H, W)} of every net2 buffer on a W2 x H2 grid."""
    if W2 % 16 or H2 % 16:
        raise ValueError("W2 and H2 are multiples of 16")
    e = {}
    for d, names in ((1, ("x", "U4", "U1", "U3", "y", "out")), (2, ("a1", "A4", "A1", "A3", "b1", "e0", "e1", "e2")),
                     (4, ("a2", "g0", "g1", "g2", "s0", "s1", "s2", "s3", "s4", "tb1", "tb2", "tb3", "tb4")),
                     (8, ("r0", "r1", "r2", "r3", "r4", "t1", "t2", "t3", "t4"))):
        for n in names:
            e[n] = (H2 // d, W2 // d)
    return e


# ---- float16 operations -------------------------------------------------------------------------------------------

def f16(x):
    return h(x)


def mul16(a, b):
    """h(a b): the product of two float16 values is exact in float64, so this is one rounding (IEEE signs)."""
    with np.errstate(invalid="ignore", over="ignore"):
        return h(np.asarray(a, np.float64) * np.asarray(b, np.float64))


def add16(a, b):
    """h(a + b): the sum of two float16 values is exact in float64."""
    with np.errstate(invalid="ignore", over="ignore"):
        return h(np.asarray(a, np.float64) + np.asarray(b, np.float64))


def fma16(a, b, c):
    """RN16(a b + c) with one rounding, for float16 a, b, c (IEEE signed zeros). The product is exact in float64; the
    sum's float64 error (two-sum) turns its rounding into round-to-odd, which a float16 rounding then makes exact."""
    a, b, c = np.broadcast_arrays(*(np.asarray(v, np.float64) for v in (a, b, c)))
    with np.errstate(invalid="ignore", over="ignore"):
        p = a * b
        s = p + c
        bb = s - p
        err = (p - (s - bb)) + (c - bb)
    fix = np.isfinite(s) & np.isfinite(err) & (err != 0) & ((s.view(np.int64) & 1) == 0)
    if np.any(fix):
        s = np.where(fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    return h(s)


def leaky(x):
    """max(h(s x), x) with s = LEAKY_SLOPE. h(s x) has x's sign, so the max never compares zeros of different signs."""
    x = np.asarray(x, np.float64)
    return np.maximum(mul16(LEAKY_SLOPE, x), x)


def clamp6(x):
    """min(max(x, -6), 6); NaN stays NaN."""
    return np.minimum(np.maximum(np.asarray(x, np.float64), -CLAMP), CLAMP)


def act(x):
    return clamp6(leaky(x))


POST = {"act": act, "leaky": leaky, "clamp": clamp6, "none": lambda v: np.asarray(v, np.float64)}


def plus_zero(x):
    """h(x + 0): -0 becomes +0, everything else is unchanged."""
    return add16(x, 0.0)


# ---- layers -------------------------------------------------------------------------------------------------------

def conv3(x, w, b):
    """3 x 3 convolution, stride 1, zero border: x [H, W, cin], canonical weights w [cout, 3, 3, cw] (cw >= cin; input
    channels cin..cw - 1 read 0), bias [cout]. Float16 accumulator from the bias; chunks per tap (ky, kx) in row-major
    order, inside a tap ascending blocks of 16 channels (the last block may be shorter); each chunk exact, one rounding.
    Returns the accumulator [H, W, cout] (no activation)."""
    x = h(x)
    w = h(w)
    b = h(b)
    H, W, cin = x.shape
    cout, kh, kw, cw = w.shape
    if (kh, kw) != (3, 3) or cin > cw:
        raise ValueError("3 x 3 weights with at least the input's channels expected")
    xp = np.zeros((H + 2, W + 2, cw))
    xp[1:H + 1, 1:W + 1, :cin] = x
    out = np.empty((H * W, cout))
    rows = max(1, MAX_ROW_BAND // W)
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        n = (y1 - y0) * W
        acc = np.broadcast_to(b[None, :], (n, cout)).copy()
        for ky in range(3):
            for kx in range(3):
                xs = xp[y0 + ky:y1 + ky, kx:kx + W, :].reshape(n, cw)
                for c0 in range(0, cw, 16):
                    c1 = min(cw, c0 + 16)
                    acc = acc_matmul16(acc, xs[:, c0:c1], w[:, ky, kx, c0:c1].T)
        out[y0 * W:y1 * W] = acc
    return out.reshape(H, W, cout)


def pool2(a):
    """2 x 2 average pool, stride 2, floor sizes: p = h(h(h(a00 + a01) + a10) + a11), out = h(p * 0.25)."""
    a = h(a)
    H, W, _ = a.shape
    Ho, Wo = H // 2, W // 2
    t = [a[dy:2 * Ho:2, dx:2 * Wo:2] for dy in (0, 1) for dx in (0, 1)]
    p = add16(add16(add16(t[0], t[1]), t[2]), t[3])
    return mul16(p, 0.25)


def up_axis(n_in):
    """(i0, i1, w0, w1) per output index (2 n_in of them) of the half-pixel 2x rule, in single precision."""
    f32 = np.float32
    i0s, i1s, w0s, w1s = [], [], [], []
    for o in range(2 * n_in):
        s = f32(f32(f32(o) - f32(0.5)) * f32(0.5))
        i0 = int(np.trunc(s))
        i1 = int(np.trunc(f32(s + f32(1))))
        if i1 >= n_in:
            i0 = i1 = n_in - 1
        if s < 0:
            i0 = i1 = 0
        w1 = float(h(f32(s - f32(i0))))
        w0 = float(h(f32(f32(f32(i0) - s) + f32(1))))
        i0s.append(i0); i1s.append(i1); w0s.append(w0); w1s.append(w1)
    return np.array(i0s), np.array(i1s), np.array(w0s), np.array(w1s)


def up2(p):
    """2x bilinear upsample, half-pixel centres: p [h, w, C] -> [2h, 2w, C]; top = fma(wx1, P01, h(wx0 P00)), bottom
    likewise on row y1, out = fma(wy1, bottom, h(wy0 top))."""
    p = h(p)
    hi, wi, _ = p.shape
    x0, x1, wx0, wx1 = up_axis(wi)
    y0, y1, wy0, wy1 = up_axis(hi)
    wx0, wx1 = wx0[None, :, None], wx1[None, :, None]

    def row(r):
        return fma16(wx1, r[:, x1], mul16(wx0, r[:, x0]))
    top = row(p[y0])
    bot = row(p[y1])
    return fma16(wy1[:, None, None], bot, mul16(wy0[:, None, None], top))


def f_half(f):
    """The scalar f (single precision) rounded to float16, nearest even."""
    with np.errstate(over="ignore"):
        return float(np.float32(f).astype(np.float16))


def merge_in(ib, ic, ia, f):
    """Launch 1: x [H2, W2, 10] = Ib 0..3, Ic 0..3, Ia, h(f)."""
    ib, ic, ia = h(ib), h(ic), h(ia)
    H, W, _ = ib.shape
    return np.concatenate([ib, ic, ia, np.full((H, W, 1), f_half(f))], axis=2)


def warp_sample(img, ux, uy):
    """Bilinear sample of img [H, W, C] at (x + 2 ux, y + 2 uy) per pixel (x, y), clamp to edge, float16 steps:
    px = fma(2, ux, h(x)); X0 = floor(px), fx = h(px - X0); x0, x1 = X0, X0 + 1 clamped to [0, W - 1] (y likewise);
    gx = h(1 - fx); w00 = h(gx gy), w10 = h(fx gy), w01 = h(gx fy), w11 = h(fx fy);
    value = fma(w11, P11, fma(w01, P10, fma(w10, P01, h(w00 P00)))), P01 = (y0, x1), P10 = (y1, x0)."""
    img = h(img)
    H, W, _ = img.shape
    ux, uy = h(ux), h(uy)
    gy_, gx_ = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    px = fma16(2.0, ux, h(gx_))
    py = fma16(2.0, uy, h(gy_))
    if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        raise ValueError("a warp sample position is not finite")
    X0 = np.floor(px)
    Y0 = np.floor(py)
    fx = h(px - X0)
    fy = h(py - Y0)
    X0 = X0.astype(np.int64)
    Y0 = Y0.astype(np.int64)
    x0, x1 = np.clip(X0, 0, W - 1), np.clip(X0 + 1, 0, W - 1)
    y0, y1 = np.clip(Y0, 0, H - 1), np.clip(Y0 + 1, 0, H - 1)
    gx = h(1.0 - fx)
    gy = h(1.0 - fy)
    w00, w10, w01, w11 = (mul16(a, c)[:, :, None] for a, c in ((gx, gy), (fx, gy), (gx, fy), (fx, fy)))
    P00, P01, P10, P11 = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
    acc = fma16(w10, P01, mul16(w00, P00))
    acc = fma16(w01, P10, acc)
    return fma16(w11, P11, acc)


def warp_merge(ib, ic, ia, u4, u1, u3, f):
    """Launch 16: y [H2, W2, 18] = Ib warped by (U4 0, U4 1), Ic warped by (U4 2, U4 3), U4 (not doubled), U1, Ia,
    U3, h(f)."""
    ib, ic, ia, u4, u1, u3 = (h(v) for v in (ib, ic, ia, u4, u1, u3))
    H, W, _ = ib.shape
    wb = warp_sample(ib, u4[:, :, 0], u4[:, :, 1])
    wc = warp_sample(ic, u4[:, :, 2], u4[:, :, 3])
    return np.concatenate([wb, wc, u4, u1, ia, u3, np.full((H, W, 1), f_half(f))], axis=2)


# ---- the launches -------------------------------------------------------------------------------------------------

def _block(first, src, temps, outs):
    """A residual block: four units of two launches (flag 0, then flag 1 with the unit's input as skip)."""
    steps, x = [], src
    for u in range(4):
        la, lb = "net2_%02d" % (first + 2 * u), "net2_%02d" % (first + 2 * u + 1)
        steps.append({"op": "conv", "layer": la, "in": x, "out": temps[u], "post": "act", "res": "zero"})
        steps.append({"op": "conv", "layer": lb, "in": temps[u], "out": outs[u], "post": "act", "res": x})
        x = outs[u]
    return steps


def _launches():
    s = [{"op": "merge", "out": "x"},
         {"op": "conv", "layer": "net2_01", "in": "x", "out": "a1", "post": "act", "pool": True},
         {"op": "conv", "layer": "net2_02", "in": "a1", "out": "a2", "post": "act", "pool": True},
         {"op": "conv", "layer": "net2_03", "in": "a2", "out": "r0", "post": "act", "pool": True}]
    s += _block(4, "r0", ["t1", "t2", "t3", "t4"], ["r1", "r2", "r3", "r4"])
    s += [{"op": "split", "layer": "net2_12", "in": "r4", "out": ["g0", "g1", "g2"],
           "post": ["leaky", "act", "act"]},
          {"op": "heads", "layer": "net2_13", "in": ["g0", "g1", "g2"], "out": ["A4", "A1", "A3"],
           "keep": [(0, 4), (4, 5), (5, 8)], "post": ["none", "clamp", "clamp"]},
          {"op": "up", "in": ["A4", "A1", "A3"], "out": ["U4", "U1", "U3"]},
          {"op": "warp", "out": "y"},
          {"op": "conv", "layer": "net2_14", "in": "y", "out": "b1", "post": "act", "pool": True},
          {"op": "conv", "layer": "net2_15", "in": "b1", "out": "s0", "post": "act", "pool": True}]
    s += _block(16, "s0", ["tb1", "tb2", "tb3", "tb4"], ["s1", "s2", "s3", "s4"])
    s += [{"op": "split", "layer": "net2_24", "in": "s4", "out": ["e0", "e1", "e2"],
           "post": ["leaky", "leaky", "leaky"]},
          {"op": "heads", "layer": "net2_25", "in": ["e0", "e1", "e2"], "out": ["out"],
           "keep": [(0, 4), (4, 5), (5, 8)], "post": ["none", "clamp", "clamp"],
           "res": [("U4", 2.0), ("U1", 1.0), ("U3", 1.0)]}]
    return s


NET2_LAUNCHES = _launches()   # 28 launches; the network output is buffer "out" [H2, W2, 8]


def run_launch(st, buf, params, inputs):
    """Runs one launch on the buffers `buf`; returns {buffer: array} of what it writes."""
    op = st["op"]
    if op == "merge":
        return {"x": merge_in(inputs["Ib"], inputs["Ic"], inputs["Ia"], inputs["f"])}
    if op == "warp":
        return {"y": warp_merge(inputs["Ib"], inputs["Ic"], inputs["Ia"], buf["U4"], buf["U1"], buf["U3"],
                                inputs["f"])}
    if op == "up":
        return {o: up2(buf[i]) for i, o in zip(st["in"], st["out"])}
    w, b = params[st["layer"]]
    if op == "conv":
        v = POST[st["post"]](conv3(buf[st["in"]], w, b))
        res = st.get("res")
        if res == "zero":
            v = plus_zero(v)
        elif res is not None:
            v = add16(v, buf[res])
        if st.get("pool"):
            v = pool2(v)
        return {st["out"]: v}
    if op == "split":
        c = conv3(up2(buf[st["in"]]), w, b)
        g = c.shape[2] // len(st["out"])
        return {o: POST[p](c[:, :, j * g:(j + 1) * g]) for j, (o, p) in enumerate(zip(st["out"], st["post"]))}
    if op == "heads":
        parts = []
        for j, (src, (lo, hi), p) in enumerate(zip(st["in"], st["keep"], st["post"])):
            v = POST[p](conv3(up2(buf[src]), w, b)[:, :, lo:hi])
            if "res" in st:
                rn, m = st["res"][j]
                v = fma16(m, buf[rn], v)
            parts.append(v)
        if len(st["out"]) == 1:
            return {st["out"][0]: np.concatenate(parts, axis=2)}
        return dict(zip(st["out"], parts))
    raise ValueError("unknown launch %r" % op)


def check_inputs(ib, ic, ia):
    H, W = np.shape(ia)[:2]
    if H % 16 or W % 16:
        raise ValueError("net2's grid is a multiple of 16 on both axes (grid())")
    for n, v, c in (("Ib", ib, 4), ("Ic", ic, 4), ("Ia", ia, 1)):
        if np.shape(v) != (H, W, c):
            raise ValueError("%s: shape %s, expected %s" % (n, np.shape(v), (H, W, c)))


def run(ib, ic, ia, f, params, launches=NET2_LAUNCHES):
    """All 28 launches on inputs Ib, Ic [H2, W2, 4], Ia [H2, W2, 1], f (single precision); params {layer: (canonical
    weights [out, 3, 3, in], bias)} (forward.py's parameters). Returns every buffer."""
    check_inputs(ib, ic, ia)
    inputs = {"Ib": h(ib), "Ic": h(ic), "Ia": h(ia), "f": np.float32(f)}
    buf = dict(inputs)
    for st in launches:
        buf.update(run_launch(st, buf, params, inputs))
    return buf


def forward(ib, ic, ia, f, params):
    """net2's output [H2, W2, 8]."""
    return run(ib, ic, ia, f, params)["out"]


def check_launch_layers():
    """Every net2 layer is used, in upload order (L13 and L25 once each for their three heads)."""
    names = []
    for st in NET2_LAUNCHES:
        if "layer" in st:
            names.append(st["layer"])
    return names == [l.name for l in L.NET2]
