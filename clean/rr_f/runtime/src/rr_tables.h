// SPDX-License-Identifier: MIT
// Model-independent tables the converter's model file does not carry, generated at build time by
// clean/rr_f/runtime/tools/build.py from the clean reference (the same arrays tools/pack_model.py packs):
//   kRrSigTable  binary16 sigmoid of 51.4 for every binary16 bit pattern (reference/scalar.py sig_tables()[0])
//   kRrHeadTab   the D0-R head's sigmoid / tanh binary16 tables (head/dx12/run_dx12.py tables(): sig then tanh16)
#pragma once
#include <stdint.h>

namespace wgrr_rt {
extern const uint16_t kRrSigTable[65536];
extern const uint16_t kRrHeadTab[];
extern const uint32_t kRrHeadTabCount;
}  // namespace wgrr_rt
