// SPDX-License-Identifier: MIT
// WrongGPU launcher core: constants shared by the modules (names, the FidelityFX DLLs WrongGPU replaces, other
// mods, anti-cheat, exit codes). Moved unchanged from setup.cpp. Written for this project.
#pragma once
#include "base.h"

#include <cwchar>

namespace wg {

const wchar_t* const kAppName = L"WrongGPU";
const wchar_t* const kStateDirName = L"wronggpu-setup";           // <install dir>\wronggpu-setup: record, backup, uninstaller
// <install dir>\WrongGPU-originals\<time>\: the player's originals a restore could not put back because another file is at
// their path now (record.h, the file-ownership rule). Never removed by the launcher; the searches below a game skip it.
const wchar_t* const kKeptDirName = L"WrongGPU-originals";
// The FidelityFX DLLs WrongGPU replaces (spec 20, 30). The game's file is kept next to WrongGPU's under the original
// name, and WrongGPU's DLL forwards to it. The FFX API DLL is AMD's (version resource); the FSR 3.0 component DLLs are
// built by the game's developer and are recognised by their exports (as the front end does).
struct FfxSlot { const wchar_t* name; const wchar_t* original; bool amdBuilt; const char* exports[2]; const wchar_t* what; };
inline const FfxSlot kSlots[] = {
    {L"amd_fidelityfx_dx12.dll", L"amd_fidelityfx_dx12.wronggpu-original.dll", true, {"ffxCreateContext", "ffxDispatch"}, L"FSR 3.1 API"},
    {L"ffx_fsr3_x64.dll", L"ffx_fsr3_x64.wronggpu-original.dll", false, {"ffxFsr3ContextCreate", "ffxFsr3ContextDispatchUpscale"}, L"FSR 3.0"},
    {L"ffx_fsr3upscaler_x64.dll", L"ffx_fsr3upscaler_x64.wronggpu-original.dll", false, {"ffxFsr3UpscalerContextCreate", "ffxFsr3UpscalerContextDispatch"}, L"FSR 3.0 upscaler"},
};
// The Apps & features root (HKCU, UninstallHive) the entries go under, ending in a backslash.
#ifdef WRONGGPU_TESTING
// Test builds never touch the real one: a test root of their own per %LOCALAPPDATA% (each test run, and a sandbox, sets its
// own), so one run never counts another run's games as still installed. Its name holds the FNV-1a hash of that folder
// (backslashes, no trailing one, ASCII lower case; over its UTF-16 code units), as the test suites compute it
// (tests/run_installer_tests.py uninstall_root). Read each time: the elevated worker sets LOCALAPPDATA from its job.
inline wstring UninstallRootFor(wstring lad) {
    for (auto& c : lad) c = c == L'/' ? L'\\' : (c >= L'A' && c <= L'Z' ? (wchar_t)(c - L'A' + L'a') : c);
    while (!lad.empty() && lad.back() == L'\\') lad.pop_back();
    uint32_t h = 2166136261u;
    for (wchar_t c : lad) h = (h ^ (uint32_t)(uint16_t)c) * 16777619u;
    wchar_t b[80];
    swprintf(b, 80, L"Software\\WrongGPU-CandidateTests\\%08X\\Uninstall\\", h);
    return b;
}
inline wstring UninstallRoot() {
    wstring v(GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0), L'\0');
    v.resize(v.empty() ? 0 : GetEnvironmentVariableW(L"LOCALAPPDATA", &v[0], (DWORD)v.size()));
    return UninstallRootFor(v);
}
#else
inline wstring UninstallRoot() { return L"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\"; }
#endif
const wchar_t* const kRegPrefix = L"WrongGPU-";
const wchar_t* const kTmpSuffix = L".wronggpu-tmp";
const wchar_t* const kHipDownload = L"https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html";
// Older installs: WrongGPU 0.1.x / 0.2.x, published as "DLSS 4.5 AMD" (state folder dlss45-amd next to the game exe).
const wchar_t* const kLegacyStateDir = L"dlss45-amd";
const wchar_t* const kLegacyKeyPrefix = L"DLSS45AMD-";
const wchar_t* const kLegacyKey = L"DLSS45AMD";  // single entry of the Cyberpunk-only 0.1.0 installer
inline const wchar_t* const kLegacyLoaderNames[] = {L"version.dll", L"winmm.dll", L"winhttp.dll", L"dbghelp.dll", L"nvngx.dll", L"nvapi64.dll"};
inline const wchar_t* const kLegacyFiles[] = {L"dlss45-amd-runtime.dll", L"dlss45-amd.pak", L"dlss45-amd-notices.txt"};

// Other upscaler mods that take over the same calls and cannot run next to WrongGPU. Their marker files, and proxy DLLs
// that name them, are moved into the backup if the player agrees (restored on uninstall). (List from the 0.2.0 installer.)
inline const wchar_t* const kProxyNames[] = {L"dxgi.dll", L"winmm.dll", L"version.dll", L"dbghelp.dll", L"d3d12.dll", L"wininet.dll", L"winhttp.dll", L"OptiScaler.asi"};
struct OtherMod { const wchar_t* name; const char* signature; const wchar_t* files[4]; };
inline const OtherMod kOtherMods[] = {
    {L"OptiScaler", "optiscaler", {L"OptiScaler.ini", L"OptiScaler.log", L"OptiScaler.dll", nullptr}},
    // detected by its signature and its two file names; never named in what players read (owner 2026-11-04)
    {L"Another upscaler mod", "dlssnr_on_amd", {L"dlssnr_on_amd.ini", L"dlssnr_on_amd.log", nullptr, nullptr}},
};
// Anti-cheat marker files and folders (list from the 0.2.0 installer).
inline const wchar_t* const kAntiCheat[] = {L"EasyAntiCheat", L"EasyAntiCheat_EOS.sys", L"start_protected_game.exe", L"EasyAntiCheat_EOS_Setup.exe",
                                     L"EasyAntiCheat_Setup.exe", L"BattlEye", L"BEService_x64.exe", L"BEService.exe",
                                     L"EAAntiCheat.GameServiceLauncher.exe", L"EAAntiCheat.GameServiceLauncher.dll", L"EAAntiCheat.Installer.exe",
                                     L"GameGuard", L"mhyprot2.sys", L"vgk.sys", L"vgc.exe", L"PnkBstrA.exe", L"xigncode", L"nProtect",
                                     L"anogs.dll", L"atvi-randgrid_sc.exe", L"randgrid.sys", L"NeacSafe64.sys", L"NeacClient.exe",
                                     L"AntiCheatExpert", L"ACE-BASE.sys", L"SGuard64.exe", L"TenProtect", L"EAAntiCheat"};
// Online games whose anti-cheat is built into the game (no marker files): matched by executable name, or by a word in
// the store folder name. They are never supported.
struct DeniedGame { const wchar_t* exe; const wchar_t* folder; const wchar_t* why; };
inline const DeniedGame kDenied[] = {
    {nullptr, L"Call of Duty", L"Call of Duty (Ricochet anti-cheat)"},
    {L"cod.exe", nullptr, L"Call of Duty (Ricochet anti-cheat)"},
    {L"BlackOpsColdWar.exe", nullptr, L"Call of Duty (Ricochet anti-cheat)"},
    {L"ModernWarfare.exe", nullptr, L"Call of Duty (Ricochet anti-cheat)"},
    {L"Marvel-Win64-Shipping.exe", nullptr, L"Marvel Rivals (NetEase anti-cheat)"},
    {L"Overwatch.exe", nullptr, L"Overwatch (Warden anti-cheat)"},
    {L"cs2.exe", nullptr, L"Counter-Strike 2 (VAC)"},
    {L"FortniteClient-Win64-Shipping.exe", nullptr, L"Fortnite (anti-cheat)"},
    {L"r5apex.exe", nullptr, L"Apex Legends (anti-cheat)"},
    {L"r5apex_dx12.exe", nullptr, L"Apex Legends (anti-cheat)"},
    {L"TslGame.exe", nullptr, L"PUBG (anti-cheat)"},
    {L"bf6.exe", nullptr, L"Battlefield 6 (EA Javelin anti-cheat)"},
    {L"bf2042.exe", nullptr, L"Battlefield 2042 (EA anti-cheat)"},
    {L"Discovery.exe", nullptr, L"THE FINALS (anti-cheat)"},
    {L"DeltaForceClient-Win64-Shipping.exe", nullptr, L"Delta Force (ACE anti-cheat)"},
    {L"NarakaBladepoint.exe", nullptr, L"NARAKA: BLADEPOINT (anti-cheat)"},
    {L"EscapeFromTarkov.exe", nullptr, L"Escape from Tarkov (BattlEye)"},
    {L"RainbowSix.exe", nullptr, L"Rainbow Six Siege (BattlEye)"},
    {L"destiny2.exe", nullptr, L"Destiny 2 (BattlEye)"},
    {L"VALORANT-Win64-Shipping.exe", nullptr, L"VALORANT (Vanguard)"},
};
// Validated games: tested end to end with this build (spec 30: Cyberpunk 2077). All others are experimental.
struct KnownGame { const wchar_t* exe; const wchar_t* name; };
inline const KnownGame kValidated[] = {{L"Cyberpunk2077.exe", L"Cyberpunk 2077"}};

// Exit codes of silent mode (/S); also documented in README.md.
enum ExitCode {
    EXIT_OK = 0,
    EXIT_FAILED = 1,           // install or uninstall failed; an install was rolled back
    EXIT_REFUSED = 2,          // a requirement is not met (game, FSR DLL, Windows, GPU, HIP runtime, game running, ...)
    EXIT_EXPERIMENTAL = 3,     // experimental game: pass /EXPERIMENTAL
    EXIT_ANTICHEAT = 4,        // anti-cheat found: never supported
    EXIT_NO_MODEL = 5,         // no DLSS model file converted
    EXIT_OTHER_MODS = 6,       // other upscaler mods found: pass /MOVEMODS to move them aside (with backup)
    EXIT_OLD_INSTALL = 7,      // an older WrongGPU / "DLSS 4.5 AMD" install: pass /REMOVEOLD to uninstall it first
    EXIT_DAMAGED = 8,          // the installer itself is damaged (payload or converter data)
    EXIT_NOT_INSTALLED = 9,    // uninstall: no install record in that folder
    EXIT_USAGE = 10,           // bad command line
};

}  // namespace wg
