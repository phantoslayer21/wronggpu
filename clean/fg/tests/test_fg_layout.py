# SPDX-License-Identifier: MIT
"""Frame generation (#96): the weight layout table (clean/fg/layout.py) against the totals the spec side gives for it.

    python -m unittest clean.fg.tests.test_fg_layout
"""
import unittest

from clean.fg import layout as L


class LayoutTotals(unittest.TestCase):
    def test_counts_and_bytes(self):
        t = L.totals()
        self.assertEqual((t["net1_blobs"], t["net2_blobs"]), (17, 50))
        self.assertEqual(len(L.blobs()), 67)
        self.assertEqual(t["net1_bytes"], 3905024)
        self.assertEqual(t["net2_bytes"], 963552)
        self.assertEqual(t["bytes"], 4868576)

    def test_params(self):
        t = L.totals()
        self.assertEqual(t["net1_params"], 959792)
        self.assertEqual(sum(l.weight_elements for l in L.NET1), 957440)
        self.assertEqual(sum(l.out for l in L.NET1), 2352)
        self.assertEqual(t["net2_params"], 481776)

    def test_net1_blob_sizes(self):
        want = [56320, 10240, 299008, 593920, 1187840, 540672, 532480, 270336, 135168, 69632, 69632, 69632, 34816,
                18432, 9216, 5120, 2560]
        self.assertEqual([l.blob_bytes for l in L.NET1], want)
        for l in L.NET1:
            self.assertEqual(l.out % L.NET1_BIAS_GROUP, 0)
            if l.taps is not None:
                self.assertEqual(l.k % l.taps, 0)
        self.assertIsNone(L.NET1[1].taps)  # blob 2: 1 x 1 or 2 x 2, OPEN
        self.assertEqual(L.NET1[0].cin, 48)

    def test_net2_blob_sizes(self):
        want_w = [9216, 18432, 36864] + [73728] * 8 + [110592, 4608, 5184, 9216] + [18432] * 8 + [27648, 2304]
        self.assertEqual([l.weight_bytes for l in L.NET2], want_w)
        self.assertEqual([l.bias_bytes for l in L.NET2], [2 * l.out for l in L.NET2])
        self.assertEqual([l.subnet for l in L.NET2].count("A"), 13)
        self.assertEqual(L.NET2[0].cin_used, 10)

    def test_blob_order(self):
        names = [b.name for b in L.blobs()]
        self.assertEqual(names[:17], ["net1_%02d" % i for i in range(1, 18)])
        self.assertEqual(names[17:21], ["net2_01_w", "net2_01_b", "net2_02_w", "net2_02_b"])
        self.assertEqual(names[-1], "net2_25_b")

    def test_net1_bias_copy0(self):
        self.assertEqual([L.net1_bias_index(c) for c in range(10)], [0, 1, 2, 3, 4, 5, 6, 7, 128, 129])
        l = L.NET1[0]
        self.assertEqual(L.net1_bias_index(l.out - 1) + 15 * 8, l.bias_elements - 1)  # last channel's last copy ends the blob


if __name__ == "__main__":
    unittest.main()
