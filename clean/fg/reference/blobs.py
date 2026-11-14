# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): splitting the networks' weight blobs into their parts, as a CPU reference.

Written for this project from the blob layout in clean/fg/layout.py. Input: the bytes of one blob (float16, little
endian). Output: flat float16 arrays in the order they are stored, plus checks of what the layout says must hold.

- net1 blob: [weights: out * K][zero area: out * K][bias: out * 16, groups of 8 channels each stored 16 times].
  split_net1 returns the weights, and the out distinct biases (copy 0 of each group); it checks that the zero area
  holds only zeros (+0 or -0) and that the 16 copies of every group are bit-identical, unless told not to.
- net2: a weight blob of 9 * in * out values and a bias blob of out values; split_net2 checks the sizes.

The order of the weights inside a blob (tap, input and output channel, any tiling) is OPEN, so the weights stay flat in
stored order; reshaping them into clean/fg/reference/ops.py's canonical order waits for it.
"""
import numpy as np

from clean.fg import layout as L


class BlobError(ValueError):
    """A blob that does not have the layout's size or structure."""


def _halves(data, what):
    if len(data) % 2:
        raise BlobError("%s: odd byte count %d" % (what, len(data)))
    return np.frombuffer(bytes(data), dtype="<f2")


def split_net1(layer, data, *, check=True):
    """(weights [out * K] float16, bias [out] float16) of one net1 blob. check: verify the zero area and the bias
    copies (BlobError on a mismatch)."""
    if len(data) != layer.blob_bytes:
        raise BlobError("%s: %d bytes, the layout says %d" % (layer.name, len(data), layer.blob_bytes))
    h = _halves(data, layer.name)
    n = layer.weight_elements
    weights = h[:n].copy()
    zero = h[layer.zero_offset:layer.bias_offset]
    stored = h[layer.bias_offset:].view("<u2").reshape(layer.out // L.NET1_BIAS_GROUP, L.NET1_BIAS_COPIES,
                                                       L.NET1_BIAS_GROUP)
    if check:
        if np.any(zero.view("<u2") & 0x7FFF):
            raise BlobError("%s: the zero area holds a non-zero value" % layer.name)
        if np.any(stored != stored[:, :1, :]):
            raise BlobError("%s: the copies of a bias group differ" % layer.name)
    bias = stored[:, 0, :].reshape(layer.out).view("<f2").copy()
    return weights, bias


def split_net2(layer, weight_data, bias_data):
    """(weights [9 * in * out] float16, bias [out] float16) of one net2 layer's two blobs."""
    if len(weight_data) != layer.weight_bytes:
        raise BlobError("%s_w: %d bytes, the layout says %d" % (layer.name, len(weight_data), layer.weight_bytes))
    if len(bias_data) != layer.bias_bytes:
        raise BlobError("%s_b: %d bytes, the layout says %d" % (layer.name, len(bias_data), layer.bias_bytes))
    return _halves(weight_data, layer.name + "_w").copy(), _halves(bias_data, layer.name + "_b").copy()


def split_all(blobs, *, check=True):
    """{layer name: (weights, bias)} for every layer of both networks. blobs: {blob name: bytes} with the names of
    layout.blobs() (net1_NN, net2_NN_w, net2_NN_b). A missing or extra blob is a BlobError."""
    want = {b.name for b in L.blobs()}
    if set(blobs) != want:
        missing, extra = sorted(want - set(blobs)), sorted(set(blobs) - want)
        raise BlobError("blob set differs: missing %s, extra %s" % (missing[:4], extra[:4]))
    out = {}
    for l in L.NET1:
        out[l.name] = split_net1(l, blobs[l.name], check=check)
    for l in L.NET2:
        out[l.name] = split_net2(l, blobs[l.name + "_w"], blobs[l.name + "_b"])
    return out


def build_net1(layer, weights, bias):
    """The inverse of split_net1: a blob from weights [out * K] and bias [out] (zero area of +0, 16 copies of each
    bias group). For synthetic tests."""
    w = np.asarray(weights, dtype="<f2").reshape(-1)
    b = np.asarray(bias, dtype="<f2").reshape(-1)
    if w.size != layer.weight_elements or b.size != layer.out:
        raise BlobError("%s: wrong sizes" % layer.name)
    groups = b.reshape(-1, 1, L.NET1_BIAS_GROUP)
    rep = np.repeat(groups, L.NET1_BIAS_COPIES, axis=1).reshape(-1)
    return w.tobytes() + np.zeros(layer.weight_elements, dtype="<f2").tobytes() + rep.tobytes()
