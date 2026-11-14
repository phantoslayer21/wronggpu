# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): run net1 (clean/fg/reference/net1.py) on the weights of a converter output file and
record a digest of its output for a fixed synthetic input.

    python -m clean.fg.tools.net1_digest --model <model.wgcm> [--record <file.json>]

The input is synthetic and fixed (seeded random values in [0, 1), SIZE x SIZE pixels; value 9 is 1 - value 8 as in
net1's input packing). The record holds the SHA-256 of the output (float16, little endian, [4, SIZE, SIZE] row-major),
its shape, the number of non-finite values and the model file's SHA-256: digests only, never weights or values.
"""
import argparse
import hashlib
import json
import sys

import numpy as np

from clean.fg.reference import forward as F

SIZE = 64        # pixels per side: the smallest grid with a cell on every level
SEED = 20261114


def fixed_input():
    rng = np.random.default_rng(SEED)
    v = rng.uniform(0.0, 1.0, (10, SIZE, SIZE)).astype(np.float16).astype(np.float64)
    v[9] = (1.0 - v[8]).astype(np.float16)
    return v


def digest(model_path):
    params = F.params_from_model_file(model_path)
    out = F.net1_forward(fixed_input(), params).astype("<f2")
    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "model_sha256": model_sha,
        "input": {"seed": SEED, "size": SIZE},
        "output_shape": list(out.shape),
        "nonfinite": int(np.count_nonzero(~np.isfinite(out))),
        "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a converter output file of the preset-FG map (model.wgcm)")
    ap.add_argument("--record", default=None, help="write the record (JSON) here as well")
    a = ap.parse_args(argv)
    text = json.dumps(digest(a.model), indent=1, sort_keys=True)
    print(text)
    if a.record:
        with open(a.record, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
