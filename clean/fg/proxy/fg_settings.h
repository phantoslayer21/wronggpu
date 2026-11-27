// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8a): the per-game FG settings keys.
//
// They live with the other per-game settings in <game folder>\wronggpu-setup\settings.ini (clean/wgshared/include/
// wg_state.h: key=value lines, the launcher writes them under the state lock). Four keys, all optional:
//
//   fg=off|on                                   default off: WrongGPU never takes frame generation unless the player
//                                               turns it on (and the route qualifies, fg_proxy_core.h)
//   fg_preset=A|B|C                             default A
//   fg_multiplier=2|3|4|5|6                     default 2 (2x: one generated image per game frame; N = multiplier - 1)
//   fg_presenter=auto|hwfq|dxgi_tearing|dxgi_vsync   default auto
//
// Strict: a value is taken only in exactly the spelling above (no case folding, no leading zeros or signs, no spaces
// inside). A missing key takes its default; an invalid value also leaves the default in place AND is reported, so the
// launcher can show it and a reader never acts on a guess. Keys that are not FG keys are not looked at: the SR / RR keys
// and everything else in the file stay exactly as they are (FgApplySettings changes only the four FG keys).
// Platform-free (no Windows headers): works on the key/value list wgstate::Settings::kv holds.
// Written for this project.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace wgfgp {

inline constexpr const char* kFgKeyEnable = "fg";
inline constexpr const char* kFgKeyPreset = "fg_preset";
inline constexpr const char* kFgKeyMultiplier = "fg_multiplier";
inline constexpr const char* kFgKeyPresenter = "fg_presenter";

inline constexpr uint32_t kFgMinMultiplier = 2;
inline constexpr uint32_t kFgMaxMultiplier = 6;

enum class FgPresenterChoice : uint32_t { Auto = 0, Hwfq = 1, DxgiTearing = 2, DxgiVsync = 3 };  // WgFgPresenterMode

struct FgSettings {
    bool enabled = false;          // fg
    uint32_t preset = 0;           // fg_preset: WgFgPreset (A = 0, B = 1, C = 2)
    uint32_t multiplier = 2;       // fg_multiplier: 2..6
    FgPresenterChoice presenter = FgPresenterChoice::Auto;  // fg_presenter
};

struct FgSettingsIssue {
    std::string key;
    std::string value;
    std::string why;
};

using FgKeyValues = std::vector<std::pair<std::string, std::string>>;

bool FgIsSettingsKey(const std::string& key);
// One key's value into `s` (only that field changes). false: not an FG key, or a value outside the key's spelling list;
// `why` (optional) says which.
bool FgParseSetting(const std::string& key, const std::string& value, FgSettings* s, std::string* why);
// The FG settings from a key/value list (the first line of a key counts, as in wg_state.h). Missing keys keep their
// defaults; invalid values keep the default and are listed in `issues` (optional). Other keys are ignored.
FgSettings FgSettingsFromKeyValues(const FgKeyValues& kv, std::vector<FgSettingsIssue>* issues);
// The four keys in their canonical spelling, in the order above.
FgKeyValues FgSettingsToKeyValues(const FgSettings& s);
// Writes the four keys into `kv`: an existing key's value is replaced in place, a missing one is appended; every other
// entry stays where it is with its value. false (kv unchanged): `s` is not valid.
bool FgApplySettings(const FgSettings& s, FgKeyValues* kv);
bool FgSettingsValid(const FgSettings& s);

// N, the generated images per game frame, of a valid multiplier (2..6 -> 1..5); 0 for anything else.
uint32_t FgGeneratedCountOf(uint32_t multiplier);
const char* FgPresetSpelling(uint32_t preset);           // "A" / "B" / "C"; nullptr outside A..C
const char* FgPresenterSpelling(FgPresenterChoice p);    // nullptr outside the four

}  // namespace wgfgp
