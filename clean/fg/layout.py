# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the weight layout of its two convolution networks, as structure only.

Written for this project from the spec side's frame-generation scoping (parts 1 and 2). No weights, no file bytes:
only layer shapes and the byte sizes of the blobs that hold them. One weight set serves every configuration (both FG
presets, every multi-frame count and index, every optional input), and the same blobs are in every 310.x library, so
this one table describes them all.

Network 1 ("net1"): a six-level encoder-decoder on a 2 x 2-folded grid, 17 blobs. Each blob is
    [weights: out * K float16][a zero area of the same size][bias: out * 16 float16]
where K = taps * input channels, and the bias holds the output channels in groups of 8, each group stored 16 times
(group g, copy r, channel j at element g * 128 + r * 8 + j of the bias area). A converter takes copy 0 and skips the
zero area.

Network 2 ("net2"): two small encoder-decoders (sub-nets A and B), 25 layers, all 3 x 3, each a weight blob of
9 * in * out float16 followed by a bias blob of out float16 (50 blobs).

The blob order below is the order the library uploads them, which is also their data order. The order of the weights
inside each blob is given by the converter map (data/fg.v1.map.json, revision v1.1: natural order by explicit offsets;
clean/fg/reference/natural.py). net1's blob 2 is a 1 x 1 convolution. OPEN (not delivered yet; see FORMATS.md): which
launch reads which blob and the grid sizes of both networks relative to the output. Nothing here guesses them.
"""
from dataclasses import dataclass

F16 = 2                 # bytes per element
NET1_BIAS_GROUP = 8     # output channels per bias group
NET1_BIAS_COPIES = 16   # copies of each bias group


@dataclass(frozen=True)
class Net1Layer:
    index: int       # 1-based, upload order
    out: int         # output channels (= distinct bias values)
    k: int           # taps * input channels
    taps: int        # 9 (3 x 3) or 1 (1 x 1)

    @property
    def cin(self):
        """Input channels."""
        return self.k // self.taps

    @property
    def weight_elements(self):
        return self.out * self.k

    @property
    def bias_elements(self):
        """Elements of the stored (replicated) bias area."""
        return self.out * NET1_BIAS_COPIES

    @property
    def blob_bytes(self):
        return F16 * (2 * self.weight_elements + self.bias_elements)

    @property
    def zero_offset(self):
        """First element of the zero area."""
        return self.weight_elements

    @property
    def bias_offset(self):
        """First element of the bias area."""
        return 2 * self.weight_elements

    @property
    def params(self):
        """Weights and distinct biases (the zero area and the bias copies excluded)."""
        return self.weight_elements + self.out

    @property
    def name(self):
        return "net1_%02d" % self.index


@dataclass(frozen=True)
class Net2Layer:
    index: int       # 1-based, upload order
    subnet: str      # "A" or "B"
    cin: int         # stored input channels
    cin_used: int    # input channels that carry data (layer 1 of A stores 16, uses 10)
    out: int
    # The grid the layer runs on, as the divisor d of the network grid W2 x H2 (1: W2, 2: W2/2, 4: W2/4, 8: W2/8);
    # None where it is not given (sub-net A's last layer: OPEN).
    level: object = None
    upscaled: bool = False   # its output is upscaled 2x (bilinear) to the next level up

    @property
    def weight_elements(self):
        return 9 * self.cin * self.out

    @property
    def weight_bytes(self):
        return F16 * self.weight_elements

    @property
    def bias_bytes(self):
        return F16 * self.out

    @property
    def params(self):
        return self.weight_elements + self.out

    @property
    def name(self):
        return "net2_%02d" % self.index


def _net1():
    rows = [  # (out, K, taps)
        (32, 432, 9),     # 3 x 3, 48 -> 32 (level 0)
        (64, 32, 1),      # 32 -> 64, 1 x 1 (map v1.1)
        (128, 576, 9),
        (128, 1152, 9),
        (256, 1152, 9),
        (512, 256, 1),    # bottom
        (256, 512, 1),    # bottom
        (256, 256, 1),
        (128, 256, 1),
        (128, 128, 1),
        (128, 128, 1),
        (128, 128, 1),
        (64, 128, 1),
        (64, 64, 1),
        (32, 64, 1),
        (32, 32, 1),
        (16, 32, 1),      # network output
    ]
    return tuple(Net1Layer(i + 1, o, k, t) for i, (o, k, t) in enumerate(rows))


def _net2():
    # (sub-net, in stored, in used, out, level divisor, output upscaled)
    rows = [("A", 16, 10, 32, 2, False), ("A", 32, 32, 32, 4, False), ("A", 32, 32, 64, 8, False)]
    rows += [("A", 64, 64, 64, 8, False)] * 8                    # the eight-layer block
    rows += [("A", 64, 64, 96, 8, True), ("A", 32, 32, 8, None, False)]
    rows += [("B", 18, 18, 16, 2, False), ("B", 16, 16, 32, 4, False)]
    rows += [("B", 32, 32, 32, 4, False)] * 8                    # the eight-layer block
    rows += [("B", 32, 32, 48, 4, True), ("B", 16, 16, 8, 1, False)]
    return tuple(Net2Layer(i + 1, s, c, cu, o, lv, up) for i, (s, c, cu, o, lv, up) in enumerate(rows))


NET1 = _net1()
NET2 = _net2()


@dataclass(frozen=True)
class Blob:
    name: str      # net1_NN, net2_NN_w, net2_NN_b
    nbytes: int
    layer: object  # the Net1Layer / Net2Layer it belongs to


def blobs():
    """All weight blobs in upload (= data) order: the 17 of net1, then net2's weight and bias blob per layer."""
    out = [Blob(l.name, l.blob_bytes, l) for l in NET1]
    for l in NET2:
        out.append(Blob(l.name + "_w", l.weight_bytes, l))
        out.append(Blob(l.name + "_b", l.bias_bytes, l))
    return out


def net1_bias_index(channel):
    """Element of copy 0 of an output channel's bias, relative to the bias area."""
    g, j = divmod(channel, NET1_BIAS_GROUP)
    return g * NET1_BIAS_GROUP * NET1_BIAS_COPIES + j


def net2_macs_per_pixel(subnet, *, used_inputs=True, open_level=None):
    """Multiply-adds of one sub-net per pixel of the network grid W2: each layer's 9 * in * out at its own grid
    (divided by d * d). used_inputs: count only the input channels that carry data. open_level: the divisor to assume
    for a layer whose level is OPEN (required when the sub-net has one). Computed, not measured."""
    total = 0
    for l in NET2:
        if l.subnet != subnet:
            continue
        d = l.level if l.level is not None else open_level
        if d is None:
            raise ValueError("%s: level OPEN; pass open_level" % l.name)
        total += 9 * (l.cin_used if used_inputs else l.cin) * l.out / (d * d)
    return total


def totals():
    net1 = sum(l.blob_bytes for l in NET1)
    net2 = sum(l.weight_bytes + l.bias_bytes for l in NET2)
    return {
        "net1_blobs": len(NET1), "net2_blobs": 2 * len(NET2),
        "net1_bytes": net1, "net2_bytes": net2, "bytes": net1 + net2,
        "net1_params": sum(l.params for l in NET1), "net2_params": sum(l.params for l in NET2),
    }
