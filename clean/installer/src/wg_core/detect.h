// SPDX-License-Identifier: MIT
// WrongGPU launcher core: game discovery (Steam, Epic, GOG), the game executable and its folders, the FidelityFX DLLs
// WrongGPU replaces (the game's, ours or another mod's) and the inspection of one game. Written for this project.
#pragma once
#include "coexist.h"
#include "context.h"
#include "record.h"

namespace wg {

struct StoreGame { wstring name, root, hint, source; };
struct Found { wstring name, exe, source; bool validated = false, opti = false; };  // opti: DLSS / XeSS only (needs OptiScaler)
enum FfxKind { FFX_NONE, FFX_GAME, FFX_OURS, FFX_FOREIGN };  // FFX_GAME: AMD's API DLL, or the game's FSR 3.0 build
// The state of one FidelityFX DLL in the install folder.
struct SlotState { const FfxSlot* slot = nullptr; FfxKind kind = FFX_NONE; wstring info; bool inPayload = false; };
struct GameTarget {
    wstring exe, exeDir, dir, root, name;  // dir: the install folder (where the game's FidelityFX DLLs are)
    bool validated = false;
    std::vector<SlotState> slots;  // the three FidelityFX DLLs in dir
    bool anySlot() const { for (auto& st : slots) if (st.kind != FFX_NONE) return true; return false; }
    bool installed = false;  // a WrongGPU record exists in dir
    Record record;
    bool presetK = false;  // installed, and an earlier build set it up to start on Preset K (StartsOnPresetK)
    wstring antiCheat;
    Conflicts conflicts;
    CoexistReport coexist;  // the rules table (design section 5): every tool found, its decision, the proxy name
};

wstring UnescapeJsonLike(wstring s);
wstring QuotedAt(const wstring& line, int n);
wstring KeyValue(const wstring& text, const wchar_t* key);
wstring RegString(HKEY root, const wchar_t* key, const wchar_t* value, REGSAM extra = 0);
wstring GameRoot(const wstring& exeDir);
bool LooksLikeHelperExe(const wstring& exe);
wstring UnrealShipping(const wstring& root);
// A root starter's game of the same name: <root>\<Project>\Binaries\Win64|WinGDK\<name>.exe or <root>\bin\x64\<name>.exe (rc6:
// Hogwarts Legacy's root HogwartsLegacy.exe). FindGameExe's hint (a store's launch exe, an exe added by hand) goes there.
wstring SameNameGameExe(const wstring& exe);
wstring FindGameExe(const wstring& root, const wstring& hint = L"");
const KnownGame* Validated(const wstring& exe);
std::vector<StoreGame> ScanLibraries();
StoreGame EpicItemGame(const wstring& manifestText);  // one Epic manifest (.item): name, InstallLocation, LaunchExecutable hint
// rc6 L7: the AppName of the Epic manifest (%ProgramData%\Epic\EpicGamesLauncher\Data\Manifests\*.item) whose InstallLocation
// holds exe; empty: not an Epic install (or a name that can't go into an address). Play starts such a game through Epic.
wstring EpicAppName(const wstring& exe);
// The hint DetectGames gives FindGameExe for a store's launch exe: empty for a launcher stub (gamerules.h kLauncherStubs) or
// when the folder holds a known game's exe the launch exe doesn't lead to (the discovery decides), else the launch exe.
wstring StoreExe(const wstring& root, const wstring& hint);
std::vector<Found> DetectGames(const std::vector<StoreGame>& roots);
wstring ResolveGameExe(wstring in);
bool IsSlotName(const wstring& name);
wstring SignatureInfo(const wstring& path);
FfxKind ClassifySlot(const FfxSlot& slot, const wstring& path, const Record* rec, wstring& info);
std::vector<SlotState> InspectSlots(Context& ctx, const wstring& dir, const Record* rec);
wstring GameDisplayName(const wstring& exe);
// An install an earlier build (0.5.0, 0.5.1) set up to start on Preset K: the OptiScaler plugin's choice in dir
// (wronggpu-asi.ini, selection=wronggpu-k), else the last model item of wronggpu.ini's runtime_options (model=K), else the
// record's model file in the Preset K cache. A build without Preset K switches such a game to Model M when it updates or
// repairs it, and the launcher asks first.
bool StartsOnPresetK(const wstring& dir, const Record* rec);
// The Steam app id of a game in a Steam library (its appmanifest_<id>.acf in the steamapps folder that holds the game's
// folder, matched by installdir; any depth of the exe below steamapps\common\<installdir>); empty elsewhere.
wstring SteamAppId(const wstring& exe);
wstring FindInstallDir(const wstring& exe);
GameTarget InspectGame(Context& ctx, const wstring& exe, const wstring& displayName = L"");
}  // namespace wg
