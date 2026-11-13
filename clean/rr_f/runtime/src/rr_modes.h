// SPDX-License-Identifier: MIT
// RR modes of the runtime, derived from the registry (clean/wgshared/include/wg_registry_gen.h, round i15).
//
// The registry's mode rows (kWgModeRows) name each mode's kernel variants by word (head exact / relaxed, input exact /
// lite / split, body relaxed / exact, output default / fast, resolve hw / exact). This file maps the words to the
// runtime's variant bits (the four groups of variant_group() in rr_runtime.cpp) and input-features selector, so adding a
// mode that reuses existing variants is one registry row; a mode with a new variant word adds its kernels and one entry
// in row_known() (rr_modes.cpp; checked once at start, a row with an unknown word is refused and logged). The registry's
// tables are C `static const` arrays (not constant expressions), so the build-time checks are on the counts.
#pragma once
#include <stdint.h>

#include "../../../wgshared/include/wg_registry_gen.h"

namespace wgrr_rt {

// the runtime's variant groups (bit g of a variant set = the group's alternative kernels run)
enum { VG_HEAD = 0, VG_BODY = 1, VG_OUT = 2, VG_RES = 3 };

typedef int Mode;  // WG_RR_MODE_* (registry slot WG_OPT_RR_MODE values)
constexpr int kModeCount = WG_RR_MODE_COUNT;
constexpr Mode kDefaultMode = WG_RR_MODE_ACCURATE;

// variant set of a mode (head relaxed -> VG_HEAD, body exact -> VG_BODY, output fast -> VG_OUT, resolve exact -> VG_RES)
unsigned mode_vbits(Mode m);
// input-features selector of a mode: 0 exact, 1 lite, 2 split (input_features_kernel in rr_runtime.cpp)
int mode_input(Mode m);
// true when every registry mode row uses only variant words this runtime has kernels for; else `why` names the row
// (the runtime then refuses that mode: it falls back to the default mode with a log line)
bool modes_self_check(const char** why);
// the registry's name of a mode (log words)
inline const char* mode_name(Mode m) { return (m >= 0 && m < kModeCount) ? kWgRrModeNames[m] : "?"; }
// name -> mode (case-sensitive lower-case words as the registry writes them); false when unknown
bool parse_mode(const char* text, Mode* out);

}  // namespace wgrr_rt
