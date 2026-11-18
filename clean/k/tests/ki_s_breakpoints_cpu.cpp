// SPDX-License-Identifier: MIT
// Dump the production table for the exhaustive Python oracle.
#include "../runtime/k_s_breakpoints.h"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  auto* out = std::fopen(argv[1], "wb");
  if (!out) return 2;
  const auto table = wgrt::ki_s_breakpoints::make_table();
  const size_t count = std::fwrite(table.data(), sizeof table[0], table.size(), out);
  const int rc = std::fclose(out);
  return count == table.size() && rc == 0 ? 0 : 2;
}
