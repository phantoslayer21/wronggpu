// SPDX-License-Identifier: GPL-3.0-only
// CPU test of clean/runtime/struct_args.h: the parser of the manifest's struct argument token s<N>:<fields> and the packer
// that turns a HIP struct (64-bit device pointers inside) into root constants. Prints "ok" and exits 0, or the first
// failure and exits 1.
#include "../runtime/struct_args.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace wgrt;

static int failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL line %d: %s\n", __LINE__, #cond);                 \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

static bool parse(const char* t, unsigned* bytes, unsigned* dwords, std::vector<StructField>* f) {
  return parse_struct_arg(t, bytes, dwords, f);
}

int main() {
  unsigned bytes = 0, dwords = 0;
  std::vector<StructField> f;
  CHECK(parse("s184:p19.b32", &bytes, &dwords, &f) && bytes == 184 && dwords == 27 && f.size() == 2);
  CHECK(parse("s32:p1.x8.b16", &bytes, &dwords, &f) && bytes == 32 && dwords == 5 && f.size() == 3);
  CHECK(parse("s16:p2", &bytes, &dwords, &f) && dwords == 2);
  const char* bad[] = {"", "s", "s8", "s8:", "s8:p", "s8:q1", "s16:p1", "s16:p1.b4", "s12:p1.b3", "s12:p1.b4.x", "s0:p0",
                       "x8:p1", "s8:p1.", "s8:.p1", "s8p1", "s1234567:p1", "s12:p1.x3.b1", "s8:p1.p1"};
  for (const char* b : bad) CHECK(!parse(b, &bytes, &dwords, &f));

  // pack: p2.b8.x8.p1 = 16 + 8 + 8 + 8 = 40 bytes -> 2 + 2 + 0 + 1 root constants
  CHECK(parse("s40:p2.b8.x8.p1", &bytes, &dwords, &f) && dwords == 5);
  unsigned char src[40];
  for (int i = 0; i < 40; ++i) src[i] = (unsigned char)(0xA0 + i);
  auto put64 = [&](int at, uint64_t v) { for (int k = 0; k < 8; ++k) src[at + k] = (unsigned char)(v >> (8 * k)); };
  put64(0, 0x1000);   // translated by the callback
  put64(8, 0);        // null stays 0 and is not translated
  put64(32, 0x2468);
  uint32_t out[8] = {};
  int calls = 0;
  int n = pack_struct_arg(f, src, out, [&](uint64_t p, uint32_t* off) {
    ++calls;
    *off = (uint32_t)(p - 0x1000) + 64;
    return true;
  });
  CHECK(n == 5);
  CHECK(calls == 2);
  CHECK(out[0] == 64 && out[1] == 0);
  CHECK(out[2] == ((uint32_t)src[16] | (uint32_t)src[17] << 8 | (uint32_t)src[18] << 16 | (uint32_t)src[19] << 24));
  CHECK(out[3] == ((uint32_t)src[20] | (uint32_t)src[21] << 8 | (uint32_t)src[22] << 16 | (uint32_t)src[23] << 24));
  CHECK(out[4] == (0x2468 - 0x1000) + 64);  // the skipped bytes 24..31 produce nothing
  // a pointer that cannot be translated fails the whole argument
  CHECK(pack_struct_arg(f, src, out, [](uint64_t, uint32_t*) { return false; }) == -1);
  if (failures == 0) std::printf("ok\n");
  return failures ? 1 : 0;
}
