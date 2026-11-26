// SPDX-License-Identifier: MIT
// WrongGPU launcher core: frame generation's per-game settings and its library (issue #96, F8c). Strictly off by default.
//
// The settings: four keys of the game's profile (profile.ini, `set <game> key=value`) that the launcher writes into the
// game's state folder, <install dir>\wronggpu-setup\settings.ini (clean/wgshared/include/wg_state.h), where WrongGPU's
// frame generation reads them. The keys, their spellings and their defaults are those of the frame generation proxy's
// reader (clean/fg/proxy/fg_settings.h):
//
//   fg=off|on                                        default off
//   fg_preset=A|B|C                                  default A
//   fg_multiplier=2|3|4|5|6                          default 2
//   fg_presenter=auto|hwfq|dxgi_tearing|dxgi_vsync   default auto
//
// Strict: a value is taken only in exactly that spelling (no case folding, no leading zeros or signs, nothing inside the
// value but its own characters; spaces around it are dropped, as for every line of the files). A missing key takes its
// default; an invalid one in settings.ini also keeps the default and is reported, never guessed at.
//
// Off by default, and nothing changes for a game that was never opted in: the keys are profile settings only in a build
// with the frame generation choice (CMake WRONGGPU_FG, off by default; profile.cpp), a default value is never written into
// profile.ini, and settings.ini gets the four keys only once the profile turns a key away from its default, or when the file
// already holds one of them (so that turning frame generation off again is written down too).
//
// The library: frame generation's data is converted on this PC from the frame generation library of one of the player's own
// games (nvngx_dlssg.dll, version 310.1 or newer) with the preset-FG converter map (data/fg.v1.map.json) into
// %LOCALAPPDATA%\WrongGPU\model\FG\model.wgcm ("model convert --preset=FG"; context.h SelectPreset). An older library (3.x)
// has no frame-generation model and is refused by its version before anything is read from it.
// Written for this project.
#pragma once
#include "util.h"

namespace wg {

struct Profile;
struct Record;
struct Context;

// ---- the settings ------------------------------------------------------------------------------------------------
inline constexpr const char* kFgKeyEnable = "fg";
inline constexpr const char* kFgKeyPreset = "fg_preset";
inline constexpr const char* kFgKeyMultiplier = "fg_multiplier";
inline constexpr const char* kFgKeyPresenter = "fg_presenter";
inline constexpr const char* const kFgKeys[] = {kFgKeyEnable, kFgKeyPreset, kFgKeyMultiplier, kFgKeyPresenter};

bool IsFgKey(const string& key);
// The allowed spellings ('|' separated) and the default of an FG key; nullptr for any other key.
const char* FgValues(const string& key);
const char* FgDefault(const string& key);
// One FG key's value checked in its exact spelling (spaces around it dropped). True with *normal = the value; false with
// *why (the allowed spellings named). False for a key that is not an FG key.
bool ValidateFgValue(const string& key, const string& value, string* normal, wstring* why);

// The refusal of an FG key in a build without the frame generation choice (WRONGGPU_FG off).
inline const wchar_t* const kNoFgChoice =
    L"Frame generation isn't part of this build of WrongGPU: it comes with a release that runs it. Frame generation stays off.";

// The FG settings a key/value list holds (wg_state.h Settings::kv: a key's first line counts). Every FG key is in `v`: its
// valid value, else its default; each invalid value is listed in `issues` ("settings.ini: ..."). `present`: the list has at
// least one FG key (valid or not).
struct FgState {
    std::map<string, string> v;
    std::vector<wstring> issues;
    bool present = false;
};
FgState FgStateOf(const std::vector<std::pair<string, string>>& kv);

// The FG keys the launcher writes into settings.ini for profile `p`, given the file's keys as they are now (`now`) and
// whether frame generation's data on this PC is ready. Empty (nothing written, the file stays as it is) while every FG key
// of the profile is at its default and the file holds no FG key. Otherwise all four keys in their exact spelling, in the
// order above; fg is written as off while the data is not ready, whatever the profile says (the profile keeps the choice).
std::vector<std::pair<string, string>> FgStateKeys(const Profile& p, const std::vector<std::pair<string, string>>& now, bool dataReady);
// The profile turns frame generation on, or moves another FG key away from its default.
bool FgOptedIn(const Profile& p);

// ---- the library -------------------------------------------------------------------------------------------------
inline constexpr const wchar_t* kFgLibraryName = L"nvngx_dlssg.dll";
inline constexpr unsigned kFgMinMajor = 310, kFgMinMinor = 1;  // the first version with the frame-generation model

// Whether a file version ("310.2.1.0", as pe.h FileVersion gives it) may hold the frame-generation model: 310.1 or newer.
// An empty or unparsable version: true (the converter decides from the file's bytes). False with *why in words.
bool FgVersionSupported(const wstring& version, wstring* why = nullptr);

// One frame generation library checked by its version, before anything else is read from it.
struct FgLibrary {
    enum Verdict { MISSING, TOO_OLD, CANDIDATE } verdict = MISSING;
    wstring path, version, why;
    bool usable() const { return verdict == CANDIDATE; }
};
FgLibrary CheckFgLibrary(const wstring& path);
// The frame generation libraries in a game's folder (its sub-folders too, as the converter's search does), the usable
// ones first (in the order found), then the others. Empty: the game has none.
std::vector<FgLibrary> GameFgLibraries(const wstring& gameDir);

// What `set <game> fg=on` needs before it is saved: READY (the data on this PC is ready and the game has a frame generation
// library), CONVERT (the data is not ready: `library` is the game's own library to convert it from), NO_LIBRARY (the game
// has no frame generation library: it does not offer DLSS Frame Generation) or TOO_OLD (the game's libraries are all older
// than 310.1, and the data is not ready). *why: the player's words for the last two.
enum class FgEnable { READY, CONVERT, NO_LIBRARY, TOO_OLD };
FgEnable CheckFgEnable(const wstring& gameDir, bool dataReady, wstring* library, wstring* why);

// The player's line for frame generation turned on while its data on this PC is not ready (`why`: the reason).
inline wstring FgNotReady(const wstring& why) {
    return L"Frame generation's data on this PC is not ready (" + why + L"). Convert it first: WrongGPU.exe model convert --preset=FG "
           L"(optionally --model=<the nvngx_dlssg.dll of one of your games, version 310.1 or newer>).";
}

// ---- the files (a build with them: tools/make_payload.py --fg, CMake WRONGGPU_FG with WRONGGPU_FG_DIR) ------------------
// The frame generation plugin (an .asi OptiScaler loads from its plugins folder, beside WrongGPU's own plugin and WrongGPU's RR
// plugin) and its kernel files, which it reads from wronggpu-setup\fg-dxil next to itself. An install puts the plugin beside
// WrongGPU's plugin (plugins\, or the plugin's own folder next to Ultimate ASI Loader) and the kernels in that folder's
// wronggpu-setup\fg-dxil: every one an entry of the install record (staging, rollback, uninstall byte for byte). Only in a game
// with a frame generation library (GameFgLibraries) and an OptiScaler setup; the plugin does nothing until fg=on is written
// into the game's state folder (FgStateKeys).
inline constexpr const wchar_t* kFgPluginName = L"WrongGPU-FG.asi";
inline constexpr const wchar_t* kFgPayloadPlugin = L"plugins\\WrongGPU-FG.asi";  // as make_payload.py embeds them
inline constexpr const wchar_t* kFgPayloadDxilDir = L"fg-dxil";                  // ...fg-dxil\<name>.dxil
inline constexpr const wchar_t* kFgDxilDir = L"wronggpu-setup\\fg-dxil";         // beside the installed plugin
inline constexpr const wchar_t* kFgLog = L"wronggpu-fg.log";                     // the plugin's log, in the game's (exe) folder
// WrongGPU's frame generation plugin, WrongGPU-FG.asi in any folder (inline: the restorer and the coexistence checks use it).
// It is WrongGPU's own: never "another setup's OptiScaler plugin" (coexist.h ForeignPlugins), so it never keeps WrongGPU's RR out.
inline bool IsFgPlugin(const wstring& rel) { return IEq(BaseName(rel), kFgPluginName); }
// The install placed frame generation's plugin (a copy entry of WrongGPU-FG.asi).
bool RecordHasFg(const Record& m);
// This build has frame generation's files (its plugin among them).
bool FgPayloadAvailable(Context& ctx);
// Where an install puts one embedded FG file (`payloadRel`: plugins\WrongGPU-FG.asi or fg-dxil\<name>.dxil), given where
// WrongGPU's plugin goes (`pluginRel`, relative to the install folder: plugins\WrongGPU.asi or the plugin's own folder's):
// <its folder>\WrongGPU-FG.asi and <its folder>\wronggpu-setup\fg-dxil\<name>.dxil. Empty for any other file.
wstring FgInstallRel(const wstring& pluginRel, const wstring& payloadRel);
// The game has a frame generation library (nvngx_dlssg.dll) in its folder: the exe's folder, the install folder or the game's
// root (sub-folders too). The game offers DLSS Frame Generation, which WrongGPU's frame generation takes over.
bool GameHasFgLibrary(const wstring& exeDir, const wstring& dir, const wstring& root);
// OptiScaler's own frame generation turned on in its settings (OptiScaler.ini [FrameGen]: FGInput, FGOutput or the older FGType
// set to anything but auto, nofg or empty; Enabled=true): it would take the same frames as WrongGPU's. The launcher never writes
// those keys (OptiScaler's defaults leave its frame generation off; a value there is the player's choice in OptiScaler's menu):
// with fg=on it says so. Empty: nothing in the way. OptiFgConflict reads <dir>\OptiScaler.ini.
wstring OptiFgConflictText(const string& optiIni);
wstring OptiFgConflict(const wstring& dir);

}  // namespace wg
