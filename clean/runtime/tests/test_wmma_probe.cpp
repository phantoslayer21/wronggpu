// SPDX-License-Identifier: MIT
// The driver-channel probe (runtime/wmma_probe.h) on any host (no GPU, no D3D12). WrongGPU's own code.
//   clang++ -std=c++17 -O2 test_wmma_probe.cpp -o test_wmma_probe.exe && test_wmma_probe.exe
//   (or: cl /nologo /std:c++17 /O2 /EHsc test_wmma_probe.cpp)
// A model of the matrix instruction reads the probe's buffer with a fragment layout, computes D = A x B + C and writes
// D back in that layout. With the layout of wmma_probe.h the probe must pass; with every wrong layout below (lanes,
// elements or halves swapped, A / B / the accumulator transposed, A and B swapped, the other accumulator row table) it
// must fail. The probe used up to 0.6.0 (all-ones A, all-twos B, C = the lane) is run through the same model to show
// that it passes all of those wrong layouts. Exit status 0 when every case passed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../wmma_probe.h"

namespace {

namespace P = wgrt::wmma_probe;

int failures = 0;

void expect(const std::string& name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

float f16_value(uint16_t h) {
  const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -24) : std::ldexp(static_cast<float>(1024 + m), e - 25);
  return s ? -v : v;
}

float fp8_value(uint8_t b) {  // E4M3, bias 7 (no infinities; 0x7f / 0xff are NaN, never used here)
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(static_cast<float>(8 + m), e - 10);
  return s ? -v : v;
}

// Where the matrix instruction takes element j (A, B) or v (C, D) of lane l from: (row, column) of the matrix.
using Map = std::function<void(unsigned, unsigned, unsigned&, unsigned&)>;
struct Layout {
  std::string name;
  Map a = P::a_entry, b = P::b_entry, d = P::d_entry;
  bool swap_ab = false;  // the instruction reads the A operand from the B registers and B from A
};

// The model of the instruction: one pass for the FP16 operands, one for the FP8 ones.
void run(const uint8_t* buffer, const Layout& hw, float* results) {
  for (unsigned p = 0; p < 2; ++p) {
    double a[16][16] = {}, b[16][16] = {}, c[16][16] = {};
    uint32_t a_at = p ? P::kInAFp8 : P::kInAF16, b_at = p ? P::kInBFp8 : P::kInBF16;
    if (hw.swap_ab) std::swap(a_at, b_at);
    for (unsigned l = 0; l < 32; ++l)
      for (unsigned j = 0; j < 8; ++j) {
        auto element = [&](uint32_t at) {
          if (p) return fp8_value(buffer[at + 8 * l + j]);
          uint16_t h;
          std::memcpy(&h, buffer + at + 16 * l + 2 * j, 2);
          return f16_value(h);
        };
        unsigned m, k, n;
        hw.a(l, j, m, k);
        a[m][k] = element(a_at);
        hw.b(l, j, k, n);
        b[k][n] = element(b_at);
        hw.d(l, j, m, n);
        float cv;
        std::memcpy(&cv, buffer + P::kInC + 4 * (32 * j + l), 4);
        c[m][n] = cv;
      }
    for (unsigned l = 0; l < 32; ++l)
      for (unsigned v = 0; v < 8; ++v) {
        unsigned m, n;
        hw.d(l, v, m, n);
        double sum = c[m][n];
        for (unsigned k = 0; k < 16; ++k) sum += a[m][k] * b[k][n];
        results[256 * p + 32 * v + l] = static_cast<float>(sum);
      }
  }
}

// The probe up to 0.6.0: A all 1.0, B all 2.0 (the same bits as its shader constants 0x3c00 / 0x40 ...), C = the lane
// in every element; expected 32 + lane for every float.
void old_fill(uint8_t* buffer) {
  std::memset(buffer, 0, P::kBytes);
  for (unsigned l = 0; l < 32; ++l)
    for (unsigned j = 0; j < 8; ++j) {
      const uint16_t one = 0x3c00, two = 0x4000;
      std::memcpy(buffer + P::kInAF16 + 16 * l + 2 * j, &one, 2);
      std::memcpy(buffer + P::kInBF16 + 16 * l + 2 * j, &two, 2);
      buffer[P::kInAFp8 + 8 * l + j] = 0x38;
      buffer[P::kInBFp8 + 8 * l + j] = 0x40;
      const float c = static_cast<float>(l);
      std::memcpy(buffer + P::kInC + 4 * (32 * j + l), &c, 4);
    }
}
bool old_pass(const float* r) {
  for (unsigned i = 0; i < P::kResults; ++i)
    if (r[i] != static_cast<float>(32 + i % 32)) return false;
  return true;
}

// Results of the new and the old probe under one layout: new_bad = first mismatch (-1 = passes), old_ok.
struct Outcome { int new_bad; bool old_ok; };
Outcome probe(const Layout& hw) {
  static uint8_t fresh[P::kBytes], old[P::kBytes];
  static bool filled = false;
  if (!filled) { P::fill(fresh); old_fill(old); filled = true; }
  float r[P::kResults];
  run(fresh, hw, r);
  const int bad = P::first_mismatch(r);
  run(old, hw, r);
  return {bad, old_pass(r)};
}

// A layout with operand X's slots permuted: the instruction takes element j of lane l from where the model has
// element s(l, j). which: 'a', 'b' or 'd'.
Layout permuted(const std::string& name, char which, std::function<std::pair<unsigned, unsigned>(unsigned, unsigned)> s) {
  Layout x;
  x.name = name;
  Map base = which == 'a' ? x.a : which == 'b' ? x.b : x.d;
  Map map = [base, s](unsigned l, unsigned j, unsigned& r, unsigned& c) {
    const auto lj = s(l, j);
    base(lj.first, lj.second, r, c);
  };
  (which == 'a' ? x.a : which == 'b' ? x.b : x.d) = map;
  return x;
}
Layout transposed(const std::string& name, char which) {
  Layout x;
  x.name = name;
  Map base = which == 'a' ? x.a : which == 'b' ? x.b : x.d;
  (which == 'a' ? x.a : which == 'b' ? x.b : x.d) = [base](unsigned l, unsigned j, unsigned& r, unsigned& c) {
    base(l, j, c, r);
  };
  return x;
}

}  // namespace

int main() {
  // value encodings: exact for every table value; the old probe's constants
  bool enc = P::f16_bits(1) == 0x3c00 && P::f16_bits(2) == 0x4000 && P::fp8_bits(1) == 0x38 && P::fp8_bits(2) == 0x40;
  for (int x = -16; x <= 16; ++x)
    enc = enc && f16_value(P::f16_bits(x)) == static_cast<float>(x) && fp8_value(P::fp8_bits(x)) == static_cast<float>(x);
  expect("FP16 / FP8 E4M3 bits of the integers -16 .. 16 are exact", enc);

  // the table properties the probe relies on
  std::set<long> products;
  std::set<float> expected;
  long largest = 0;
  bool nonzero = true;
  for (unsigned m = 0; m < 16; ++m)
    for (unsigned n = 0; n < 16; ++n) {
      long s = 0;
      for (unsigned k = 0; k < 16; ++k) s += P::kA[m][k] * P::kB[k][n];
      products.insert(s);
      largest = std::max(largest, std::labs(s));
      nonzero = nonzero && P::kA[m][n] != 0 && P::kB[m][n] != 0;
    }
  for (unsigned i = 0; i < 256; ++i) expected.insert(P::expected(i));
  expect("all 256 products A x B are distinct (" + std::to_string(products.size()) + ")", products.size() == 256);
  expect("all 256 expected results of one pass are distinct (" + std::to_string(expected.size()) + ")",
         expected.size() == 256);
  expect("no zero operand", nonzero);
  // |A x B| + C < 2^11 with C a multiple of 2^-9: every value and partial sum fits 20 bits, exact in FP32
  expect("results exact in FP32 (largest |A x B| = " + std::to_string(largest) + " < 2048)", largest < 2048);
  for (unsigned i = 0; i < 256; ++i)
    if (P::expected(i) != P::expected(256 + i)) { expect("FP16 and FP8 passes expect the same values", false); break; }

  // the right layout passes, both probes
  Layout model;
  model.name = "model";
  Outcome right = probe(model);
  expect("layout of wmma_probe.h: new probe passes", right.new_bad < 0);
  expect("layout of wmma_probe.h: old probe passes", right.old_ok);

  // WG_TEST_DRIVER_PROBE_FAIL=1 (device_d3d12.cpp) still reports float 7
  {
    uint8_t buffer[P::kBytes];
    P::fill(buffer);
    float r[P::kResults];
    run(buffer, model, r);
    r[7] = -1.0f;
    expect("a corrupted float 7 is reported at float 7", P::first_mismatch(r) == 7);
  }

  // named wrong layouts: the new probe fails each one, the old probe passes each one
  std::vector<Layout> wrong;
  using LJ = std::pair<unsigned, unsigned>;
  wrong.push_back(permuted("A lane halves swapped (K 0-7 <-> 8-15 in A only)", 'a', [](unsigned l, unsigned j) { return LJ(l ^ 16, j); }));
  wrong.push_back(permuted("B lane halves swapped", 'b', [](unsigned l, unsigned j) { return LJ(l ^ 16, j); }));
  wrong.push_back(permuted("C/D lane halves swapped (rows 0-7 <-> 8-15)", 'd', [](unsigned l, unsigned j) { return LJ(l ^ 16, j); }));
  wrong.push_back(permuted("A neighbouring lanes swapped (rows 2i <-> 2i+1)", 'a', [](unsigned l, unsigned j) { return LJ(l ^ 1, j); }));
  wrong.push_back(permuted("B neighbouring lanes swapped (columns)", 'b', [](unsigned l, unsigned j) { return LJ(l ^ 1, j); }));
  wrong.push_back(permuted("C/D neighbouring lanes swapped (columns)", 'd', [](unsigned l, unsigned j) { return LJ(l ^ 1, j); }));
  wrong.push_back(permuted("A elements reversed in each lane", 'a', [](unsigned l, unsigned j) { return LJ(l, 7 - j); }));
  wrong.push_back(permuted("B elements reversed in each lane", 'b', [](unsigned l, unsigned j) { return LJ(l, 7 - j); }));
  wrong.push_back(permuted("C/D elements reversed in each lane", 'd', [](unsigned l, unsigned j) { return LJ(l, 7 - j); }));
  wrong.push_back(permuted("A FP16 halves of each dword swapped", 'a', [](unsigned l, unsigned j) { return LJ(l, j ^ 1); }));
  wrong.push_back(permuted("A bytes of each dword reversed", 'a', [](unsigned l, unsigned j) { return LJ(l, j ^ 3); }));
  wrong.push_back(permuted("B bytes of each dword reversed", 'b', [](unsigned l, unsigned j) { return LJ(l, j ^ 3); }));
  wrong.push_back(transposed("A transposed", 'a'));
  wrong.push_back(transposed("B transposed", 'b'));
  wrong.push_back(transposed("C/D transposed", 'd'));
  {
    Layout x;
    x.name = "A and B swapped";
    x.swap_ab = true;
    wrong.push_back(x);
    Layout y;
    y.name = "accumulator rows 2v + h (the other D_ROW table, clean/runner/packing.py gfx11)";
    y.d = [](unsigned l, unsigned v, unsigned& m, unsigned& n) { m = 2 * v + l / 16; n = l % 16; };
    wrong.push_back(y);
  }
  for (const Layout& x : wrong) {
    const Outcome o = probe(x);
    expect(x.name + ": new probe fails" + (o.new_bad >= 0 ? " (float " + std::to_string(o.new_bad) + ")" : ""),
           o.new_bad >= 0);
    expect(x.name + ": old probe passes it", o.old_ok);
  }

  // exhaustive: every pair of lanes and every pair of element slots swapped, in A, in B and in C/D
  for (char which : {'a', 'b', 'd'}) {
    int lanes_caught = 0, lanes_old = 0, lanes_total = 0, elems_caught = 0, elems_old = 0, elems_total = 0;
    for (unsigned l1 = 0; l1 < 32; ++l1)
      for (unsigned l2 = l1 + 1; l2 < 32; ++l2) {
        const Outcome o = probe(permuted("", which, [l1, l2](unsigned l, unsigned j) {
          return LJ(l == l1 ? l2 : l == l2 ? l1 : l, j);
        }));
        ++lanes_total; lanes_caught += o.new_bad >= 0; lanes_old += o.old_ok;
      }
    for (unsigned j1 = 0; j1 < 8; ++j1)
      for (unsigned j2 = j1 + 1; j2 < 8; ++j2) {
        const Outcome o = probe(permuted("", which, [j1, j2](unsigned l, unsigned j) {
          return LJ(l, j == j1 ? j2 : j == j2 ? j1 : j);
        }));
        ++elems_total; elems_caught += o.new_bad >= 0; elems_old += o.old_ok;
      }
    const std::string op = which == 'a' ? "A" : which == 'b' ? "B" : "C/D";
    expect(op + ": new probe fails all " + std::to_string(lanes_total) + " lane-pair swaps (" +
               std::to_string(lanes_caught) + "); old probe passes " + std::to_string(lanes_old),
           lanes_caught == lanes_total);
    expect(op + ": new probe fails all " + std::to_string(elems_total) + " element-pair swaps (" +
               std::to_string(elems_caught) + "); old probe passes " + std::to_string(elems_old),
           elems_caught == elems_total);
  }

  // A K order that A and B share does not change any product: the probe cannot see it and need not
  {
    Layout x;
    x.name = "the same K halves swap in A and B";
    x.a = [](unsigned l, unsigned j, unsigned& m, unsigned& k) { P::a_entry(l ^ 16, j, m, k); m = l % 16; };
    x.b = [](unsigned l, unsigned j, unsigned& k, unsigned& n) { P::b_entry(l ^ 16, j, k, n); n = l % 16; };
    expect(x.name + " (no product changes): new probe passes", probe(x).new_bad < 0);
  }

  std::printf("%s\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED");
  return failures ? 1 : 0;
}
