// SPDX-License-Identifier: MIT
// RR modes from the registry (rr_modes.h). The checks below tie the registry's mode table to the runtime's variants.
#include "rr_modes.h"

#include <string.h>

namespace wgrr_rt {

static_assert(kModeCount == sizeof(kWgModeRows) / sizeof(kWgModeRows[0]), "registry mode rows vs WG_RR_MODE_COUNT");
static_assert(kModeCount == sizeof(kWgRrModeNames) / sizeof(kWgRrModeNames[0]), "registry mode names vs WG_RR_MODE_COUNT");
static_assert(kModeCount >= 3 && kModeCount <= 31, "the three base modes exist; a mode is a slot value of the shared block");
static_assert(WG_RR_MODE_EXACT == 0 && WG_RR_MODE_ACCURATE == 1 && WG_RR_MODE_FAST == 2, "base mode ids are append-only");
static_assert(VG_HEAD == 0 && VG_BODY == 1 && VG_OUT == 2 && VG_RES == 3, "variant groups of rr_runtime.cpp");

namespace {
bool row_known(const WgModeRow& r) {
  return (r.head == WG_VARIANT_HEAD_EXACT || r.head == WG_VARIANT_HEAD_RELAXED) &&
         (r.input == WG_VARIANT_INPUT_EXACT || r.input == WG_VARIANT_INPUT_LITE || r.input == WG_VARIANT_INPUT_SPLIT) &&
         (r.body == WG_VARIANT_BODY_RELAXED || r.body == WG_VARIANT_BODY_EXACT) &&
         (r.output == WG_VARIANT_OUTPUT_DEFAULT || r.output == WG_VARIANT_OUTPUT_FAST) &&
         (r.resolve == WG_VARIANT_RESOLVE_HW || r.resolve == WG_VARIANT_RESOLVE_EXACT);
}
unsigned row_vbits(const WgModeRow& r) {
  return ((r.head == WG_VARIANT_HEAD_RELAXED ? 1u : 0u) << VG_HEAD) | ((r.body == WG_VARIANT_BODY_EXACT ? 1u : 0u) << VG_BODY) |
         ((r.output == WG_VARIANT_OUTPUT_FAST ? 1u : 0u) << VG_OUT) | ((r.resolve == WG_VARIANT_RESOLVE_EXACT ? 1u : 0u) << VG_RES);
}
int row_input(const WgModeRow& r) {
  return r.input == WG_VARIANT_INPUT_LITE ? 1 : r.input == WG_VARIANT_INPUT_SPLIT ? 2 : 0;
}
bool valid(Mode m) { return m >= 0 && m < kModeCount && row_known(kWgModeRows[m]); }
}  // namespace

unsigned mode_vbits(Mode m) { return row_vbits(kWgModeRows[valid(m) ? m : kDefaultMode]); }
int mode_input(Mode m) { return row_input(kWgModeRows[valid(m) ? m : kDefaultMode]); }

bool modes_self_check(const char** why) {
  for (int m = 0; m < kModeCount; ++m)
    if (!row_known(kWgModeRows[m])) {
      if (why) *why = kWgModeRows[m].name;
      return false;
    }
  // the three modes of round s10 keep their variant sets (before i15: exact = body + resolve, accurate = resolve,
  // fast = head + output; input lite for fast only; clean/rr-mp1-head, owner decision 2026-11-01: accurate = head +
  // resolve, i.e. accurate runs the relaxed fp32 head head.fast.rx, gate-checked): a registry edit cannot silently
  // change their kernels
  if (mode_vbits(WG_RR_MODE_EXACT) != ((1u << VG_BODY) | (1u << VG_RES)) ||
      mode_vbits(WG_RR_MODE_ACCURATE) != ((1u << VG_HEAD) | (1u << VG_RES)) ||
      mode_vbits(WG_RR_MODE_FAST) != ((1u << VG_HEAD) | (1u << VG_OUT)) || mode_input(WG_RR_MODE_EXACT) != 0 ||
      mode_input(WG_RR_MODE_ACCURATE) != 0 || mode_input(WG_RR_MODE_FAST) != 1) {
    if (why) *why = "base modes changed";
    return false;
  }
  return true;
}

bool parse_mode(const char* text, Mode* out) {
  if (!text || !*text) return false;
  for (int m = 0; m < kModeCount; ++m)
    if (strcmp(text, kWgRrModeNames[m]) == 0 && valid(m)) {
      *out = m;
      return true;
    }
  return false;
}

}  // namespace wgrr_rt
