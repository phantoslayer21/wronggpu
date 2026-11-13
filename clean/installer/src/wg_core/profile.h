// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the per-game settings profile (build/launcher-design.md section 4, the Basic tier of M1).
// The profile (<install dir>\wronggpu-setup\profile.ini) is the source of truth for a game's settings; the launcher
// writes wronggpu.ini (front-end keys and runtime_options) and OptiScaler.ini from it, validating every value first
// (a bad runtime value makes the runtime's create fail and AMD's FSR takes over). Written for this project.
#pragma once
#include "settings.h"

namespace wg {

// wronggpu.ini [fsr] key, runtime_options item, OptiScaler.ini key, the RR plugin's WrongGPU\wronggpu.ini [wronggpu] key
// (Ray Reconstruction games only, rr.h: RenderRrSettings writes them; wronggpu.ini and OptiScaler.ini never hold them)
enum class Layer { Fe, Rt, Opti, Rr };
enum class Tier { Basic, Advanced, Diagnostics };

// One setting of the profile. Values are the profile's own words ("on", "off", ...); Render maps them to the file.
struct SettingDef {
    const char* key;           // profile key (profile.ini, `set <game> key=value`)
    Layer layer;
    Tier tier;
    const char* def;           // default (the front end's / runtime's own default: never written to the game)
    const char* values;        // allowed values, '|' separated; "#rate" accepts a number in (0, 1) as well
    const wchar_t* label;      // UI label
    const wchar_t* help;       // one line for the UI
};
extern const SettingDef kSettings[];
extern const size_t kSettingCount;
const SettingDef* FindSetting(const string& key);

// Checks and normalises one value ("d3d12" -> "dx12", "0" -> "off", "0.50" -> "0.5"); false with *why when it is not
// allowed.
bool ValidateSetting(const SettingDef& d, const string& value, string* normal, wstring* why);

struct Profile {
    std::map<string, string> v;        // every key of kSettings (defaults filled in) plus unknown keys kept as read
    string get(const string& key) const;
    bool isDefault(const string& key) const;
};
Profile DefaultProfile();
// profile.ini: "[profile]" with key=value lines; comments and unknown keys are kept on rewrite. Invalid values fall
// back to the default (listed in *problems).
bool LoadProfile(const wstring& path, Profile& p, std::vector<wstring>* problems = nullptr);
string ProfileText(const Profile& p);
void SaveProfile(const wstring& path, const Profile& p);
wstring ProfilePath(const wstring& installDir);
// `set` and the UI: one key; false with *why for an unknown key or a bad value.
bool SetSetting(Profile& p, const string& key, const string& value, wstring* why);

// ---- wronggpu.ini --------------------------------------------------------------------------------------------
// The runtime_options items in order (split at ';', trimmed, empty items dropped).
std::vector<std::pair<string, string>> SplitOptions(const string& options);
string JoinOptions(const std::vector<std::pair<string, string>>& items);  // "k=v;k=v", no spaces around ';' or '='
// What the front end would use for each managed key of the file: the last runtime_options line's items, then the
// runtime keys written on lines of their own (appended by the front end, so they win), the last value of a key wins.
// Keys not in the file get their defaults. Values are normalised (ValidateSetting); a value that does not validate
// is returned as it is (the outside-edit check shows it).
std::map<string, string> EffectiveSettings(const string& wronggpuIni, const string& optiIni);
// Writes the profile into the file texts: front-end keys only when not default or when the file differs; the managed
// runtime items removed from runtime_options (and their own lines) and the non-default ones appended in a fixed
// order; unmanaged items and every other line stay as they are. OptiScaler keys only where the key already exists.
// false with *why when a profile value does not validate (nothing is rendered then).
bool RenderWronggpuIni(const Profile& p, const string& in, string& out, wstring* why);
bool RenderOptiIni(const Profile& p, const string& in, string& out, std::vector<wstring>* notes = nullptr);
// The profile a 0.5.x install implies: the effective values of its wronggpu.ini and OptiScaler.ini (defaults and the
// stale defaults of earlier versions dropped; upscaler=standin counts as on). Only keys of the settings schema
// (kSettings) are taken: developer and test keys (runtime_options items such as timing_log, capture_dir, model_dir,
// watchdog; diagnostics lines such as log_dispatches = 1) never become profile settings; each is listed in *notes as
// "ignored: key=value (...)" (kIgnoredPrefix), and an executor value outside the schema as "not imported: ...".
Profile ImportProfile(const string& wronggpuIni, const string& optiIni, std::vector<wstring>* notes = nullptr);
// The developer / test keys of a wronggpu.ini that an import ignores ("timing_log=1", "log_dispatches=1"): runtime_options
// items that are not schema keys (model=, the installer's own choice, excepted) and diagnostics lines with a value the
// installers never write.
std::vector<string> IgnoredImportKeys(const string& wronggpuIni);
inline const wchar_t* const kIgnoredPrefix = L"ignored: ";

// A key whose value in the files differs from both its default and the profile: changed outside the launcher.
struct OutsideEdit { string key, fileValue, profileValue; };
std::vector<OutsideEdit> OutsideEdits(const Profile& p, const string& wronggpuIni, const string& optiIni);

}  // namespace wg
