# SPDX-License-Identifier: MIT
"""Model L's copies of the converter map's facts agree with the map (data/preset-L.v*.map.json, spec-team output).

Run: python -m unittest discover -s clean/l/tests -t . -p test_l_map.py

The map is the source of truth. The table (clean/l/reference/table.py) and the coefficient kernel
(clean/l/kernels/l_ops.hip) hold copies of its value positions, and this test fails when they drift apart. The first
map (v1) still carried Model M's value positions, so the check needs v1.1 or later.
"""
import json
import os
import re
import unittest

from clean.l.reference import table as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "data")


def newest_l_map():
    """Path of the newest preset-L map at version 1.1 or later, else None."""
    best = None
    for f in os.listdir(DATA):
        m = re.fullmatch(r"preset-L\.v(\d+)(?:\.(\d+))?\.map\.json", f)
        if m:
            v = (int(m.group(1)), int(m.group(2) or 0))
            if v >= (1, 1) and (best is None or v > best[0]):
                best = (v, os.path.join(DATA, f))
    return best[1] if best else None


def kernel_values():
    with open(os.path.join(ROOT, "clean", "l", "kernels", "l_ops.hip"), encoding="utf-8") as f:
        m = re.search(r"kLValues\[13\]\s*=\s*\{([^}]*)\}", f.read())
    return tuple(int(v) for v in m.group(1).split(","))


class MapCopiesTest(unittest.TestCase):
    def test_kernel_values_are_the_table(self):
        self.assertEqual(kernel_values(), T.L_DEC_VALUES)

    def test_table_values_are_the_map(self):
        path = newest_l_map()
        if path is None:
            self.skipTest("no preset-L map v1.1 or later in data/")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(tuple(m["perms"]["dec0_values"]), T.L_DEC_VALUES)
        self.assertEqual(m.get("preset", "L"), "L")

    def test_table_shapes_are_the_map(self):
        path = newest_l_map() or os.path.join(DATA, "preset-L.v1.map.json")
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        shapes = T.param_shapes()
        for st in m["stages"]:
            if st["stage"] not in shapes:
                continue
            got = {r["name"]: tuple(r["shape"]) for r in st["tensors"]}
            want = {k: (v if v else (1,)) for k, v in shapes[st["stage"]].items()}   # scalars: () here, [1] there
            self.assertEqual(got, want, st["stage"])


if __name__ == "__main__":
    unittest.main()
