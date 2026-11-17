// SPDX-License-Identifier: MIT
// Writes runtime/hyb_tables.h's hyb_fill() bytes to the file given (test_hyb_tables_cpp.py compares them with
// clean/l/reference/hyb_tables.py). With a second argument "downward" the generation runs with the thread in FE_DOWNWARD
// (the bytes must not change) and the program fails (exit 3) unless that mode is restored afterwards.
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hyb_tables.h"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  std::vector<uint8_t> t(wg::kHybBytes);
  const bool down = argc > 2 && std::strcmp(argv[2], "downward") == 0;
  if (down && std::fesetround(FE_DOWNWARD) != 0) return 4;
  wg::hyb_fill(t.data());
  if (down) {
    const bool restored = std::fegetround() == FE_DOWNWARD;
    std::fesetround(FE_TONEAREST);
    if (!restored) return 3;
  }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f) return 1;
  std::fwrite(t.data(), 1, t.size(), f);
  std::fclose(f);
  return 0;
}
