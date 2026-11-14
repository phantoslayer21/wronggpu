# SPDX-License-Identifier: MIT
"""The D3D12 kernel ports' struct arguments (clean/runtime/struct_args.h): the token s<N>:<fields> and the packing of a
HIP struct with pointers into root constants. No GPU: clean/tests/test_struct_args.cpp is built with the host C++
compiler (CXX, clang++ or g++; on Windows MSVC cl after them; skipped without one) and run.

Run: python -m unittest clean.tests.test_struct_args
"""
import subprocess
import unittest

from clean.tests import host_cxx


class StructArgs(unittest.TestCase):
    def test_parse_and_pack(self):
        exe = host_cxx.build("test_struct_args", ["clean/tests/test_struct_args.cpp"])
        r = subprocess.run([exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
