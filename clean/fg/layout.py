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

The blob order below is the order the library uploads them, which is also their data order. OPEN (not delivered yet;
see FORMATS.md): the in-blob axis order of the weights (tap, input and output channel, any tiling), which launch reads
which blob, whether net1's blob 2 is a 1 x 1 or a 2 x 2 convolution, the grid sizes of both networks relative to the
output, and the converter map itself. Nothing here guesses them.
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
    taps: object     # 9 (3 x 3), 1 (1 x 1), or None where the reading is OPEN

    @property
    def cin(self):
        """Input channels, or None while the tap count is OPEN."""
        return None if self.taps is None else self.k // self.taps

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
        (64, 32, None),   # 32 -> 64: 1 x 1, or 2 x 2 over 8 channels (OPEN)
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
    rows = [("A", 16, 10, 32), ("A", 32, 32, 32), ("A", 32, 32, 64)]
    rows += [("A", 64, 64, 64)] * 8                              # the eight-layer block
    rows += [("A", 64, 64, 96), ("A", 32, 32, 8)]
    rows += [("B", 18, 18, 16), ("B", 16, 16, 32)]
    rows += [("B", 32, 32, 32)] * 8                              # the eight-layer block
    rows += [("B", 32, 32, 48), ("B", 16, 16, 8)]
    return tuple(Net2Layer(i + 1, s, c, cu, o) for i, (s, c, cu, o) in enumerate(rows))


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


def totals():
    net1 = sum(l.blob_bytes for l in NET1)
    net2 = sum(l.weight_bytes + l.bias_bytes for l in NET2)
    return {
        "net1_blobs": len(NET1), "net2_blobs": 2 * len(NET2),
        "net1_bytes": net1, "net2_bytes": net2, "bytes": net1 + net2,
        "net1_params": sum(l.params for l in NET1), "net2_params": sum(l.params for l in NET2),
    }
