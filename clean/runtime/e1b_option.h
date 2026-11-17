// SPDX-License-Identifier: MIT
// The runtime option e1b (runtime.cpp e1b_resolve; clean/FORMATS.md "Runtime option e1b"). "on" asks for E1b, "off"
// turns it off. Any other value is most likely a typo in a player's ini, so it never fails create: E1b stays off and
// *note gets one line for the log. A valid value clears *note, so the last e1b key decides.
#pragma once
#include <string>

inline bool e1b_option_on(const std::string& value, std::string* note) {
  if (note) note->clear();
  if (value == "on") return true;
  if (value != "off" && note) *note = "e1b: unknown value '" + value + "', using off";
  return false;
}

// The e1b key of a runtime options string ("key=value" items separated by ';' or ','; other keys are ignored).
// No e1b key: off, no note.
inline void e1b_scan(const std::string& options, bool* on, std::string* note) {
  *on = false;
  note->clear();
  for (size_t a = 0; a <= options.size();) {
    size_t b = options.find_first_of(";,", a);
    if (b == std::string::npos) b = options.size();
    const std::string kv = options.substr(a, b - a);
    a = b + 1;
    const size_t eq = kv.find('=');
    if (eq != std::string::npos && kv.compare(0, eq, "e1b") == 0 && eq == 3) *on = e1b_option_on(kv.substr(eq + 1), note);
  }
}

// The kernel report's e1b item (only when the option asked for E1b): the route chosen at create, never a side effect.
inline std::string e1b_report(bool on, const std::string& reason) {
  return on ? std::string(" e1b=on") : " e1b=off (" + reason + ")";
}
