// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the player's OptiScaler: its package, the OptiScaler.ini keys for WrongGPU, the proxy name,
// WrongGPU's OptiScaler plugin (WrongGPU.asi). Written for this project.
#pragma once
#include <cstring>

#include "context.h"
#include "record.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// OptiScaler (player-supplied). For games with DLSS or XeSS (and FSR games too), OptiScaler takes the game's upscaler
// inputs and hands them to WrongGPU through AMD's FidelityFX API. The player downloads OptiScaler (the release .7z, a
// .zip, or the folder they unpacked it to); WrongGPU never ships it. Layout (proven in Cyberpunk 2077 with OptiScaler
// 0.9.4, DLSS Quality): OptiScaler.dll as a proxy DLL (dxgi.dll, ...) next to the game exe with the rest of its
// release; WrongGPU's FFX DLL under its own name (kOptiFrontEnd: OptiScaler hooks every amd_fidelityfx_dx12.dll it
// sees, so under that name its upscaler calls would loop into its own hook and crash the game); OptiScaler.ini set up
// for WrongGPU (kOptiIni). The game's own FSR files are left as they are. Unpacking uses Windows' own tar.exe.
// ------------------------------------------------------------------------------------------------
const wchar_t* const kOptiFrontEnd = L"wronggpu_ffx_dx12.dll";
// The proxy names OptiScaler can load as, in the order the proxy picker tries them (coexist.h kProxyOrder, design
// section 5).
inline const wchar_t* const kOptiProxies[] = {L"dxgi.dll", L"winmm.dll", L"version.dll", L"wininet.dll", L"winhttp.dll", L"dbghelp.dll"};
const wchar_t* const kOptiTested = L"0.9.4";
inline const wchar_t* const kOptiFactoryWrappingGames[] = {L"007FirstLight.exe"};  // see DoInstall
struct OptiKey { const char* section; const char* key; const char* value; };
// OptiScaler.ini keys of a game's experimental opt-in setting (profile.cpp kSettings; off by default, an install writes
// none of them). Marvel's Spider-Man 2's Ray Reconstruction (attended run G10, 2026-11-04): OptiScaler's GPU-name spoof
// (vendor and device id) on with the device id tested there, fakenvapi off; later runs showed the game unstable with it
// (2026-11-04), so it is the opt-in `set Spider-Man2.exe sm2_rr_experimental=on`. ApplyProfile (launcher.cpp
// RenderGameOptiKeys) writes them into the installed OptiScaler.ini when the setting is on and journals them (inikey
// lines); off puts the values the install wrote back (only where the value is still ours). The installed OptiScaler.ini
// is a recorded install entry with a backup of the player's own, so an uninstall takes them away either way. No GPU name
// is written (OptiScaler's own default name is used). OptiScaler's spoofing for the game interface is the player's step
// in its menu (the release notes), as is the game's launch option. section null: the section where this OptiScaler.ini
// has the key; a key the file does not have is not written.
struct OptiGameKey { const wchar_t* exe; const char* setting; const char* section; const char* key; const char* value; };
inline const OptiGameKey kOptiGameKeys[] = {
    {L"Spider-Man2.exe", "sm2_rr_experimental", "Spoofing", "Dxgi", "true"},
    {L"Spider-Man2.exe", "sm2_rr_experimental", "Spoofing", "SpoofedDeviceId", "0x2684"},
    {L"Spider-Man2.exe", "sm2_rr_experimental", nullptr, "UseFakenvapi", "false"},
};
// The game a setting of kOptiGameKeys belongs to (nullptr: no such setting).
inline const wchar_t* OptiGameSettingExe(const char* setting) {
    for (auto& g : kOptiGameKeys) if (!strcmp(g.setting, setting)) return g.exe;
    return nullptr;
}
// The steps a game's opt-in Ray Reconstruction needs from the player, shown when the setting writes its keys.
struct GameNotice { const wchar_t* exe; const wchar_t* notice; };
inline const GameNotice kGameNotices[] = {
    {L"Spider-Man2.exe",
     L"Marvel's Spider-Man 2 (experimental Ray Reconstruction, not recommended: the game often freezes or crashes with it): "
     L"turn on OptiScaler's spoofing for the game interface in its menu, remove the Steam launch option that turns the game "
     L"interface off if you set it, keep ray-traced reflections off, and choose DLSS with Ray Reconstruction in the game "
     L"(the release notes have the steps)."},
};
// The plugin is a payload file of its own (plugins\WrongGPU.asi, built with the front end), installed by the hosted
// (OptiScaler) route for every game: OptiScaler loads it from the "plugins" folder next to itself when
// [Plugins] LoadAsiPlugins=true. On the OptiScaler builds it recognises, it puts "WrongGPU" into OptiScaler's main
// upscaler list as the default entry and leaves OptiScaler's FSR 3.X(/4) entry to AMD's upscaler; on any other build it
// changes nothing (one line in wronggpu-asi.log) and WrongGPU stays the first entry of OptiScaler's FFX box, as before.
// Dx12Upscaler=fsr31 and FfxDx12SRPath (kOptiIni) stay: the plugin's entry runs through OptiScaler's FidelityFX slot.
//
// Install record: the plugin is a "copy" entry and the plugins folder (when new) a "dir" entry; OptiScaler.ini is
// installed as a whole file with a backup of the player's, so an uninstall removes the plugin and its folder and
// puts the player's OptiScaler.ini back byte for byte. The plugin's own files (wronggpu-asi.log, wronggpu-asi.ini:
// the entry last selected) are written into the OptiScaler folder, named wronggpu-*, and go with RemoveRuntimeFiles.
// OptiScaler.ini keys for the plugin (entries of kOptiIni). Path=auto: OptiScaler's "plugins" folder next to itself.
#define WRONGGPU_ASI_INI {"Plugins", "LoadAsiPlugins", "true"}, {"Plugins", "Path", "auto"},

const wchar_t* const kAsiPlugin = L"plugins\\WrongGPU.asi";

// A payload file only the hosted (OptiScaler) route installs.
inline bool OptiOnlyPayload(const std::wstring& rel) { return _wcsicmp(rel.c_str(), kAsiPlugin) == 0; }

// The entry the plugin starts on (0.5.0, an installer with Preset K): the plugin's wronggpu-asi.ini in the OptiScaler
// folder, in the plugin's own format (FormatSelection, clean/fsr/asi/asi_thunk.h). With the plugin hooked, its entry
// decides over wronggpu.ini's model=, so without this file a Preset K install would start on Model M (the plugin's
// default). Preset K: "selection=wronggpu-k" (WrongGPU (Model K)); Model M: "selection=wronggpu". Written after the
// install succeeded and not recorded: it is one of the plugin's own files (wronggpu-*), which an uninstall removes
// (RemoveRuntimeFiles), as the restore of an earlier install removes that install's copy before this one is made.
inline void SeedAsiSelection(Context& ctx, const std::wstring& dir, bool presetK) {
    const std::string text = std::string("; WrongGPU OptiScaler plugin: the entry last selected in OptiScaler's upscaler list\r\n"
                                         "; (wronggpu or amd). Deleting this file selects WrongGPU.\r\nselection=") +
                             (presetK ? "wronggpu-k" : "wronggpu") + "\r\n";
    const std::wstring path = Join(dir, L"wronggpu-asi.ini");
    WriteFileAtomic(path, text.data(), text.size());
    LogLine(ctx, std::wstring(L"OptiScaler plugin: starts on WrongGPU (Model ") + (presetK ? L"K" : L"M") + L") (" + path + L")");
}
// OptiScaler.ini for WrongGPU; FfxDx12SRPath (empty here) gets the full path of kOptiFrontEnd at install time.
inline const OptiKey kOptiIni[] = {
    WRONGGPU_ASI_INI
    {"Upscalers", "Dx12Upscaler", "fsr31"},          // the game's inputs go to the FidelityFX API: WrongGPU
    {"Libraries", "FfxDx12SRPath", ""},              // WrongGPU's DLL is the upscaler
    {"FSR", "Fsr4Update", "false"},                  // OptiScaler must not patch WrongGPU's DLL as FSR 4 (crash)
    {"FSR", "UpscalerIndex", "0"},                   // the first version WrongGPU's DLL lists: WrongGPU (AMD's follow)
    {"CAS", "Enabled", "false"},                     // no extra sharpening on top of Model M
    {"Sharpness", "OverrideSharpness", "false"},
    {"OutputScaling", "Enabled", "false"},           // the game's own render and output sizes
    {"UpscaleRatio", "UpscaleRatioOverrideEnabled", "false"},
    {"Log", "LogToFile", "true"},                    // OptiScaler.log, collected with Collect logs
    {"Log", "LogLevel", "2"},
};

// OptiScaler 10 (the WrongGPU edition; its package has an OptiScaler\ folder): WrongGPU's front end goes into
// OptiScaler's custom upscaler slot instead of FfxDx12SRPath, so OptiScaler lists it as an upscaler of its own named
// WrongGPU (not as an FSR version) and selects it by its own index, which AMD's FSR 4 provider cannot move. The front
// end shares the runtime and wronggpu.ini next to the exe (clean/fsr settings_bridge.cpp). No menu plugin
// (WrongGPU.asi) with this layout.
const wchar_t* const kOptiSlotDir = L"OptiScaler\\custom\\WrongGPU\\ffx";
const wchar_t* const kOptiSlotFrontEnd = L"OptiScaler\\custom\\WrongGPU\\ffx\\amd_fidelityfx_upscaler_dx12.dll";
inline const OptiKey kOptiSlotIni[] = {
    {"Upscalers", "Dx12Upscaler", "WrongGPU"},       // the custom slot's folder name
    {"Libraries", "FfxDx12SRPath", "auto"},          // OptiScaler's own FidelityFX DLLs for its FSR entries
    {"WrongGPU", "UpscalerIndex", "0"},              // the slot's first version: WrongGPU Model M
};
inline bool IsOptiFrontEnd(const std::wstring& rel) {
    return _wcsicmp(rel.c_str(), kOptiFrontEnd) == 0 || _wcsicmp(rel.c_str(), kOptiSlotFrontEnd) == 0;
}

wstring OptiTempDir(Context& ctx);
int RunHiddenWait(const wstring& exe, const wstring& args, DWORD timeoutMs);
OptiPackage PrepareOptiScaler(Context& ctx, const wstring& chosen);
wstring OptiProxyFor(const wstring& dir);
void RequireOptiScaler(Context& ctx);
wstring OptiProblem(Context& ctx);
}  // namespace wg
