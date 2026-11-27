# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (#96): the delivered converter map data/fg.v1.map.json against the layout table
(clean/fg/layout.py) and the blob splitter (clean/fg/reference/blobs.py). Structure only: the map holds no weights.

    python -m unittest clean.fg.tests.test_fg_map
"""
import hashlib
import json
import os
import unittest

from clean.fg import layout as L

MAP = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "fg.v1.map.json")
MAP_SHA256 = "b32d9195ab76e14f9c4e318bc55e5d8def83698e0a10d7581689fedf6e1c7ad5"   # the spec side's v1.1, unchanged


def load():
    with open(MAP, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw)


class DeliveredMap(unittest.TestCase):
    def test_digest(self):
        raw, _ = load()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), MAP_SHA256)
        self.assertNotIn(b"\r\n", raw)

    def test_header(self):
        _, m = load()
        self.assertEqual((m["format"], m["version"], m["preset"], m["channel_order"]),
                         ("wronggpu-converter-map", 1, "FG", "natural"))
        self.assertEqual(m["key"]["from"], "A01")
        self.assertEqual(m["perms"], {})

    def test_sources_follow_the_layout(self):
        _, m = load()
        names = list(m["sources"])
        blobs = L.blobs()
        self.assertEqual(len(names), len(blobs))
        for name, b in zip(names, blobs):
            s = m["sources"][name]
            self.assertEqual((s["element"], s["locate"], s["length"]), ("f16le", "block", b.nbytes), name)
            self.assertNotIn("keyed", s)
        self.assertEqual(len({s["sha256"] for s in m["sources"].values()}), len(names))

    def test_stages_follow_the_layout(self):
        _, m = load()
        st = m["stages"]
        self.assertEqual(len(st), len(L.NET1) + len(L.NET2))
        for s, l in zip(st[:17], L.NET1):
            w, b = s["tensors"]
            want = [l.out, l.cin, 3, 3] if l.taps == 9 else [l.out, l.cin]
            self.assertEqual((w["name"], w["shape"], w["base"]), ("W", want, 0), l.name)
            self.assertTrue(all(set(a) == {"offsets"} for a in w["axes"]), l.name)    # natural view by offsets
            self.assertEqual((b["name"], b["shape"], b["base"]), ("b", [l.out], l.bias_offset))
            self.assertEqual(b["axes"], [{"offsets": [L.net1_bias_index(c) for c in range(l.out)]}])
            self.assertEqual({w["src"], b["src"]}, {"A%02d" % l.index})
        for s, l in zip(st[17:], L.NET2):
            w, b = s["tensors"]
            self.assertEqual((w["name"], w["src"], w["shape"]), ("W", "B%02dw" % l.index, [l.out, l.cin_used, 3, 3]))
            self.assertTrue(all(set(a) == {"offsets"} for a in w["axes"]), l.name)
            self.assertEqual((b["name"], b["src"], b["shape"], b["axes"]), ("b", "B%02db" % l.index, [l.out], [{"stride": 1}]))
        for s in st:
            for t in s["tensors"]:
                self.assertEqual((t["dtype"], t["op"]), ("float32", {"name": "f16"}))


if __name__ == "__main__":
    unittest.main()
