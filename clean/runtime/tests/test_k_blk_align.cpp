// SPDX-License-Identifier: MIT
// The developer-build alignment check of Model K's stage parameters (clean/k/runtime/k_blk_align.h). WrongGPU's own code.
// Built by clean/tests/test_k_blk_align.py with -I <dir holding kblk_struct.h>: that file is KBlkParams as runtime_k.cpp
// declares it, copied out by the test, so every pointer field of the real struct is exercised. Exit status 0 when every
// case passed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using DevPtr = uintptr_t;
#include "kblk_struct.h"
#include "../../k/runtime/k_blk_align.h"

namespace {

int failures = 0;

void expect(const std::string& name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

// every pointer field of KBlkParams, by offset (kblk_struct.h's KBLK_FIELDS list, generated with the struct)
struct Field { const char* name; size_t off; };
const Field kFields[] = {KBLK_FIELDS};

}  // namespace

int main() {
  KBlkParams p;
  std::memset(&p, 0, sizeof p);
  expect("all null: no pointer named", k_blk_misaligned(p) == nullptr);
  const DevPtr base = 0x10000;
  for (size_t i = 0; i < sizeof kFields / sizeof kFields[0]; ++i)
    *reinterpret_cast<DevPtr*>(reinterpret_cast<char*>(&p) + kFields[i].off) = base + 256 * i;
  expect("all 256-byte aligned: none named", k_blk_misaligned(p) == nullptr);
  for (size_t i = 0; i < sizeof kFields / sizeof kFields[0]; ++i) {
    DevPtr* f = reinterpret_cast<DevPtr*>(reinterpret_cast<char*>(&p) + kFields[i].off);
    const DevPtr keep = *f;
    *f = keep + 16;
    expect(std::string("16-byte step in ") + kFields[i].name + ": accepted", k_blk_misaligned(p) == nullptr);
    for (DevPtr low : {1u, 4u, 8u, 12u}) {
      *f = keep + low;   // a misaligned base: the A16 kernels would read keep + 0 instead
      const char* bad = k_blk_misaligned(p);
      expect(std::string(kFields[i].name) + " + " + std::to_string(low) + ": refused naming it",
             bad && std::strcmp(bad, kFields[i].name) == 0);
    }
    *f = keep;
  }
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
