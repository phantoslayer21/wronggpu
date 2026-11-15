// SPDX-License-Identifier: MIT
// clean/runtime/target_family.h for clean/tests/test_target_family.py: prints "<name> <family>" per argument, "-" when
// the target is refused.
#include <cstdio>

#include "../runtime/target_family.h"

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string f = wgrt::target_family(argv[i]);
    std::printf("%s %s\n", argv[i], f.empty() ? "-" : f.c_str());
  }
  return 0;
}
