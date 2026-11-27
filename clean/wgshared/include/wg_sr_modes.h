// SPDX-License-Identifier: GPL-3.0-only
// The SR models' modes (registry options sr_m_mode / sr_l_mode / sr_k_mode, slots 12..14: fast, the default, or exact)
// and the words every WrongGPU program uses for them: the key in the game's state folder (<game>\wronggpu-setup\
// settings.ini: m_mode / l_mode / k_mode, written by the launcher and by the in-game card of WrongGPU.asi) and the
// runtime option the SR front end passes to the runtime for each mode. Pure C++, no Windows.
//
// Slot owners (wg_shared_settings.h): the in-game card of WrongGPU.asi owns these three slots (as the RR plugin's panel
// owns rr_mode); it writes them whether or not it is the elected writer. No other module writes them.
#ifndef WG_SR_MODES_H
#define WG_SR_MODES_H

#include <cstdint>
#include <string>

#include "wg_registry_gen.h"

static_assert(WG_SR_M_MODE_FAST == 0 && WG_SR_M_MODE_EXACT == 1 && WG_SR_L_MODE_FAST == 0 && WG_SR_L_MODE_EXACT == 1 &&
                  WG_SR_K_MODE_FAST == 0 && WG_SR_K_MODE_EXACT == 1,
              "the three SR mode options share one value order (fast, exact)");

#define WG_SR_MODE_FAST 0
#define WG_SR_MODE_EXACT 1

struct WgSrModeKey {
    char model;                // 'M', 'L', 'K'
    uint32_t slot;             // the registry option (WG_OPT_SR_*_MODE)
    const char* state_key;     // its key in wronggpu-setup\settings.ini
    const char* fast_option;   // runtime option of mode fast
    const char* exact_option;  // runtime option of mode exact
};

static const WgSrModeKey kWgSrModeKeys[3] = {
    {'M', WG_OPT_SR_M_MODE, "m_mode", "m_mode=fast", "m_mode=exact"},
    {'L', WG_OPT_SR_L_MODE, "l_mode", "l_mode=fast", "l_mode=hybrid"},  // Model L's exact form is its hybrid route
    {'K', WG_OPT_SR_K_MODE, "k_mode", "k_mode=fast", "k_mode=exact"},   // the K runtime maps fast to k_approx=a1,a2
};

// The mode row of model 'M' / 'K' / 'L' (either case); nullptr for anything else.
inline const WgSrModeKey* wg_sr_mode_of(char model) {
    if (model >= 'a' && model <= 'z') model = static_cast<char>(model - 'a' + 'A');
    for (const WgSrModeKey& k : kWgSrModeKeys)
        if (k.model == model) return &k;
    return nullptr;
}

// "fast" -> 0, "exact" -> 1 (either case, spaces around ignored); -1 for anything else.
inline int32_t wg_parse_sr_mode(const std::string& text) {
    size_t a = 0, b = text.size();
    while (a < b && (text[a] == ' ' || text[a] == '\t')) ++a;
    while (b > a && (text[b - 1] == ' ' || text[b - 1] == '\t' || text[b - 1] == '\r' || text[b - 1] == '\n')) --b;
    std::string v = text.substr(a, b - a);
    for (char& c : v)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (v == "fast") return WG_SR_MODE_FAST;
    if (v == "exact") return WG_SR_MODE_EXACT;
    return -1;
}

inline const char* wg_sr_mode_name(int32_t mode) { return mode == WG_SR_MODE_EXACT ? "exact" : "fast"; }
inline const char* wg_sr_mode_title(int32_t mode) { return mode == WG_SR_MODE_EXACT ? "Exact" : "Fast"; }

#endif  // WG_SR_MODES_H
