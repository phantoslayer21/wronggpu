// SPDX-License-Identifier: MIT
// clean/runtime/post_tf_align.h: an A8 or history pointer that is not 8-byte aligned is refused naming it; null, 8- and
// 256-byte aligned pointers pass. Built and run by clean/tests/test_post_tf_align.py with the host compiler (no GPU).
#include <cstdio>
#include <cstring>

#include "clean/runtime/post_tf_align.h"

int main() {
  int bad = 0;
  const unsigned long long base = 0x100000ull;   // an arena allocation (256-byte aligned)
  const unsigned long long ok_vals[] = {0ull, base, base + 8u, base + 256u, base + 4096u};
  for (unsigned long long a : ok_vals)
    for (unsigned long long h : ok_vals)
      if (const char* n = post_tf_misaligned(a, h)) {
        std::printf("FAIL aligned pair %llx %llx refused as %s\n", a, h, n);
        ++bad;
      }
  const unsigned offs[] = {1u, 2u, 4u, 6u};
  for (unsigned o : offs) {
    const char* n = post_tf_misaligned(base + o, base);
    if (n && std::strcmp(n, "A8") == 0) std::printf("A8 + %u refused naming it\n", o);
    else { std::printf("FAIL A8 + %u\n", o); ++bad; }
    n = post_tf_misaligned(base, base + o);
    if (n && std::strcmp(n, "hist_prev") == 0) std::printf("hist_prev + %u refused naming it\n", o);
    else { std::printf("FAIL hist_prev + %u\n", o); ++bad; }
  }
  return bad ? 1 : 0;
}
