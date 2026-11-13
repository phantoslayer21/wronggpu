# SPDX-License-Identifier: MIT
"""The spec 51 vectors: the generator reproduces the committed manifest and SHA256SUMS byte for byte (line endings
normalised), and the reference reproduces every stored output when the files are read back."""
import os
import shutil
import tempfile
import unittest

from clean.rr_f.tests import check_vectors, gen_vectors
from clean.rr_f.tests.vectors_io import DEFAULT_DIR

CR = bytes([13])
LF = bytes([10])


def _text(path):
    with open(path, "rb") as f:
        return f.read().replace(CR + LF, LF)


class TestVectors(unittest.TestCase):
    def test_regenerated_equal_to_committed(self):
        tmp = tempfile.mkdtemp(prefix="rr51_")
        try:
            out = os.path.join(tmp, "vectors")
            gen_vectors.main(["--out", out])
            for name in ("manifest.json", "SHA256SUMS"):
                self.assertEqual(_text(os.path.join(DEFAULT_DIR, name)), _text(os.path.join(out, name)), name)
            self.assertEqual(check_vectors.main(["--dir", out]), 0)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
