// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the per-game settings profile (build/launcher-design.md section 4, the Basic tier of M1).
// The profile (<install dir>\wronggpu-setup\profile.ini) is the source of truth for a game's settings; the launcher
// writes wronggpu.ini (front-end keys and runtime_options) and OptiScaler.ini from it, validating every value first
// (a bad runtime value makes the runtime's create fail and AMD's FSR takes over). Written for this project.
#pragma once
#include "fg.h"
#include "settings.h"
#include "../../../wgshared/include/wg_model_entries.h"

namespace wg {

// wronggpu.ini [fsr] key, runtime_options item, OptiScaler.ini key, the RR plugin's WrongGPU\wronggpu.ini [wronggpu] key
// (Ray Reconstruction games only, rr.h: RenderRrSettings writes them; wronggpu.ini and OptiScaler.ini never hold them)
// Model: the model the game starts with (WRONGGPU_MODEL_L / WRONGGPU_MODEL_K builds only): wronggpu.ini's runtime_options
// model item and the OptiScaler entry OptiScaler starts with (UpscalerIndex); with M nothing is changed in either file
// (except that Model K's or Model L's own item and entry go back to Model M's when the profile leaves them).
// Fg: frame generation (fg.h; WRONGGPU_FG builds only): the game's state folder, settings.ini, once the player opts in.
// Mode: each SR model's mode, Fast (the default) or Exact (m_mode, and k_mode / l_mode in builds with those models;
// clean/wgshared/include/wg_sr_modes.h): the game's state folder, settings.ini, which the front end reads at each create and
// WrongGPU.asi's card in OptiScaler's menu writes too (FoldSrFromGame brings that back into the profile); never wronggpu.ini.
enum class Layer { Fe, Rt, Opti, Rr, Model, Fg, Mode };
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

// R1: every model runs on the Direct3D 12 runtime in a game; HIP is an explicit diagnostic (executor=hip in wronggpu.ini)
// and the frame capture only. These choices exist on the HIP runtime only, so the launcher does not offer them on an RX
// 9000-series card (it shows NotOnDx12Text() instead): executor=hip, the output filters Quality 5 and Quality 9, and the
// sparkle guard. A profile or an ini that already holds one still starts the game: the runtime uses the default output
// filter or turns the sparkle guard off, with one log line, and never starts HIP for it.
bool NeedsHipRuntime(const string& key, const string& value);
// executor=hip is not a game setting (R1: nothing in a game starts HIP; the diagnostic exists in developer builds only). A
// profile.ini or a wronggpu.ini that holds it loses it when it is read (LoadProfile, ImportProfile; the next Apply writes
// wronggpu.ini without it), and this sentence says so: install, repair and update log it.
inline const wchar_t* const kRetiredHipNote =
    L"executor=hip removed: HIP is a developer diagnostic, not a game setting; every model runs on Direct3D 12";
const wchar_t* NotOnDx12Text();

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
// The refusal of the model setting in a build without the Model L choice (WRONGGPU_MODEL_L off).
inline const wchar_t* const kNoModelChoice =
    L"Model L isn't part of this build of WrongGPU: it comes with a release that runs Model L. The game stays on Model M.";
// The indexes of Model K's and Model L's entries in the list of upscalers WrongGPU's front end gives OptiScaler (its
// UpscalerIndex), from the one shared definition (clean/wgshared/include/wg_model_entries.h): Model M, then Model K (a
// build with Model K or Preset K: the front end of the same payload lists it), then Model L.
#if WRONGGPU_PRESET_K || WRONGGPU_MODEL_K
inline constexpr bool kListsModelK = true;
#else
inline constexpr bool kListsModelK = false;
#endif
inline const char* const kModelKEntryIndex = wgentries::IndexText(wgentries::ModelEntryIndex('K', true, true));
inline const char* const kModelLEntryIndex = wgentries::IndexText(wgentries::ModelEntryIndex('L', kListsModelK, true));

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
// The model a wronggpu.ini text runs: its last runtime_options model item, 'M', 'K' or 'L'; 0 without one.
char ModelOfWronggpuIni(const string& text);
// The OptiScaler.ini text with its UpscalerIndex set to `model`'s entry of WrongGPU's list: 0 Model M, kModelKEntryIndex
// (a build with Model K or Preset K), kModelLEntryIndex (a build with the L choice); a model this build cannot choose: 0.
// [WrongGPU] of OptiScaler 10's custom slot when the file has that key or section, else [FSR]; whatever value was there
// (AMD's version, another one, none: the key is created). The launcher's model choice sets it after the record
// (launcher.cpp ApplyProfile); it is for display only, the front end never takes it for a choice.
string SetOptiModelEntry(const string& text, char model);
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

// The SR model modes of this build's profile (Layer::Mode): m_mode, then k_mode / l_mode where the build has those models.
// `model`: 'm', 'k', 'l' (a profile's model value) -> its mode key; nullptr for a model this build has no mode for.
const char* ModeKeyOf(const string& model);

// A key whose value in the files differs from both its default and the profile: changed outside the launcher.
struct OutsideEdit { string key, fileValue, profileValue; };
std::vector<OutsideEdit> OutsideEdits(const Profile& p, const string& wronggpuIni, const string& optiIni);

}  // namespace wg
