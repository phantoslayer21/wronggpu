// SPDX-License-Identifier: MIT
// The player's install switch for RR: the launcher's "Use WrongGPU" row of a game's Ray Reconstruction section
// (launcher spec BE-wide\rr-notes.md), registry key rr_enabled (clean/wgshared/registry, slot WG_OPT_RR_ENABLED).
// Off = the RR files stay installed but WrongGPU does not take RR: the plugin reports RR unsupported, so the game uses
// its own path (RrProxy::set_rr_off). Read once, when the plugin's core starts (the game asks <is-supported> before it
// creates RR, so a later change applies at the next game start, as the launcher writes it while the game is closed).
//
// Where it is read, first hit wins:
//   1. <data dir>\wronggpu.ini, [wronggpu] rr_enabled (the file the launcher writes; data dir = WG_DATA_DIR or
//      <exe dir>\WrongGPU\);
//   2. the plugin's own wronggpu-rr.ini next to the game exe: rr_enabled, or the older 60.9 key wronggpu_rr;
//   3. on.
// Values as the registry parses them: 0 / 1, off / on, false / true, no / yes. An unreadable value is ignored (the next
// source applies).
#pragma once
#include <string>
#include <vector>

#include "../include/wg_settings_reader.h"

namespace wgrr {

struct RrEnabled {
  bool on = true;
  std::string source = "default";  // log words: where the value came from
};

inline RrEnabled read_rr_enabled(const std::wstring& data_dir_ini, const std::wstring& own_ini) {
  RrEnabled r;
  const WgOptionDef& o = kWgOptions[WG_OPT_RR_ENABLED];
  std::vector<wgss::IniEntry> ini;
  int32_t v = 1;
  if (wgss::read_ini(data_dir_ini, &ini) && wgss::ini_value(ini, o, &v)) {
    r.on = v != 0;
    r.source = wgss::utf8(data_dir_ini);
    return r;
  }
  if (wgss::read_ini(own_ini, &ini) && wgss::ini_value(ini, o, &v)) {
    r.on = v != 0;
    r.source = wgss::utf8(own_ini);
  }
  return r;
}

// The plugin's usual call: the shared settings file of this game and the plugin's own ini next to the game exe.
inline RrEnabled read_rr_enabled() {
  return read_rr_enabled(wgss::data_dir() + L"wronggpu.ini", wgss::exe_dir() + L"wronggpu-rr.ini");
}

inline std::string rr_off_reason(const RrEnabled& e) {
  return "WrongGPU RR is switched off (rr_enabled=0 in " + e.source +
         "); the game uses its own path. Turn \"Use WrongGPU\" back on in the WrongGPU launcher to use it.";
}

}  // namespace wgrr
