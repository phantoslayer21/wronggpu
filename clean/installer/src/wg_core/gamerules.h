// SPDX-License-Identifier: MIT
// WrongGPU launcher core: what a few games need beyond the usual install (build/launcher/QUIET-R1.md "RE9"): REFramework,
// without which RE Requiem stops any OptiScaler (its own integrity check; WrongGPU never works around it and never
// installs, downloads or ships REFramework), WrongGPU's OptiScaler edition where the game's Ray Reconstruction needs the
// host to restore the game's command-list state (plugin interface 5), and REFramework's mirror of the game folder's DLLs
// (_storage_\, which the game then loads them from). Written for this project.
#pragma once
#include "util.h"

namespace wg {

// ---- per-game rules (by the exe's name: what a game needs cannot be read from its files) -----------------------------
struct GameRule {
    const wchar_t* exe;        // the game's exe name
    const wchar_t* name;       // the game's short name in sentences ("RE Requiem")
    bool needsReFramework;     // the game stops OptiScaler unless the player's REFramework is installed
    bool needsHostRestore;     // its Ray Reconstruction needs OptiScaler's plugin interface 5: WrongGPU's OptiScaler edition
    bool rrOnlyPathTracing;    // RR comes only with the game's path tracing: with WrongGPU's RR off the game offers no path
                               // tracing, only its own ray tracing levels
    bool rrKernelsInGame;      // release 0.6.0 polish: the launcher can't prepare RR's kernels ahead (rrprep.h): the game makes
                               // its D3D12 device through AMD's AGS library under an application name of its own, and the AMD
                               // driver keeps its shader cache under that name, so the launcher's helper (a program named like
                               // the game) fills another cache. The game prepares them at its first start (about a minute
                               // with a fresh driver; OptiScaler's overlay says "preparing" meanwhile)
    // rc5: its RR only through this game's experimental opt-in (a profile setting, `set` only; off by default): while it is
    // off the window shows RR off and not switchable, with rrHeldOffText, and nothing prepares or asks about RR there
    const char* rrExperimentalSetting = nullptr;
    const wchar_t* rrHeldOffText = nullptr;
    // rc5: the game's own menu has no Ray Reconstruction option: WrongGPU's RR isn't offered (DetectRr: not used; no RR
    // plugin installed, a repair or update takes an installed one out), the window says this line instead
    const wchar_t* rrNotOffered = nullptr;
};
const GameRule* GameRuleFor(const wstring& exe);  // null: the game has no rule

// ---- store launch exes that are never the game ---------------------------------------------------------------------------
// A store's manifest can name a launcher stub as the game's launch exe (Epic's LaunchExecutable, GOG's exe): CD PROJEKT RED's
// REDprelauncher.exe starts Cyberpunk 2077's bin\x64\Cyberpunk2077.exe. The store scan takes such an exe as no hint, so the
// exe discovery the Steam path uses finds the game (DetectGames). By file name; kept short and explicit.
inline const wchar_t* const kLauncherStubs[] = {L"REDprelauncher.exe"};
bool IsLauncherStub(const wstring& exe);

// ---- REFramework (the player's; only detected) -------------------------------------------------------------------------
const wchar_t* const kReFrameworkName = L"REFramework";
const wchar_t* const kReFrameworkPage = L"https://github.com/praydog/REFramework";  // its public page (Learn more)
const wchar_t* const kReFrameworkDll = L"dinput8.dll";
struct ReFrameworkState { bool found = false; wstring evidence; };
// REFramework's loader: its version resource or its own text names REFramework (never decided by the file name alone).
bool IsReFrameworkDll(const wstring& path);
// Next to the game exe: dinput8.dll that is REFramework (the evidence also names its reframework\ folder when there).
ReFrameworkState DetectReFramework(const wstring& exeDir);
// The sentence for a game that needs REFramework and does not have it (the core's refusal; the window says it its own way).
wstring NeedsReFrameworkText(const GameRule& rule);

// ---- REFramework's mirror (_storage_\ next to the exe) -----------------------------------------------------------------
// REFramework copies every DLL of the game's root folder into _storage_\ and the game loads them from there. An install
// next to it writes its root DLLs there too (record entries: the file that was there backed up and put back); an
// uninstall also removes the copies REFramework made of WrongGPU's files when they are byte for byte the ones installed.
const wchar_t* const kMirrorDir = L"_storage_";
inline bool IsMirrorRel(const wstring& rel) { return StartsWithI(rel, wstring(kMirrorDir) + L"\\"); }
// A file REFramework mirrors: a DLL directly in the install folder (rel without a folder).
inline bool MirroredRel(const wstring& rel) { return rel.find(L'\\') == wstring::npos && EndsWithI(rel, L".dll"); }
inline wstring MirrorRel(const wstring& rel) { return wstring(kMirrorDir) + L"\\" + rel; }

}  // namespace wg
