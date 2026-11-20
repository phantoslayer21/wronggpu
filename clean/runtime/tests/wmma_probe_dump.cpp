// SPDX-License-Identifier: MIT
// Writes the driver-channel probe's input buffer and its expected results (runtime/wmma_probe.h) to files, for a test that
// runs the probe's shader on a card another way than the runtime does (clean/dx12/tests/test_gfx11_probe.py).
//   c++ -std=c++17 wmma_probe_dump.cpp -o wmma_probe_dump && wmma_probe_dump <gfx11|gfx12> <buffer.bin> <expected.bin>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../wmma_probe.h"

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  namespace P = wgrt::wmma_probe;
  const bool g11 = std::strcmp(argv[1], "gfx11") == 0;
  if (!g11 && std::strcmp(argv[1], "gfx12") != 0) return 2;
  std::vector<uint8_t> buffer(g11 ? P::gfx11::kBytes : P::kBytes);
  std::vector<float> expected(g11 ? P::gfx11::kResults : P::kResults);
  if (g11) P::gfx11::fill(buffer.data()); else P::fill(buffer.data());
  for (unsigned i = 0; i < expected.size(); ++i) expected[i] = g11 ? P::gfx11::expected(i) : P::expected(i);
  FILE* f = std::fopen(argv[2], "wb");
  FILE* e = std::fopen(argv[3], "wb");
  if (!f || !e) return 1;
  std::fwrite(buffer.data(), 1, buffer.size(), f);
  std::fwrite(expected.data(), sizeof(float), expected.size(), e);
  std::fclose(f);
  std::fclose(e);
  std::printf("tolerance %.9g\n", g11 ? P::gfx11::kTolerance : 0.0f);   // 0: exact (gfx12)
  return 0;
}
