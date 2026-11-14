// SPDX-License-Identifier: MIT
// WrongGPU launcher core: other mods, tools and older installs next to a game (build/launcher-design.md section 5,
// milestone M2): the installer's conflict rules (FindConflicts: what an install moves aside with the player's OK), the
// rules table of the launcher (AssessCoexistence: every tool recognised by its fingerprint, with a decision) and the
// proxy picker (where OptiScaler goes). Owner decision 2026-10-17: a player's own OptiScaler is not adopted in place;
// the player is asked to switch to the OptiScaler setup the launcher installs, their files are backed up (recorded as
// "moved") and put back byte for byte on uninstall. Written for this project.
#pragma once
#include "fingerprint.h"
#include "record.h"

namespace wg {

struct Conflicts {
    std::vector<wstring> mods;       // names of other upscaler mods found (the player's OptiScaler)
    std::vector<wstring> modFiles;   // their files in the install folder (moved into the backup with the player's OK)
    std::vector<wstring> outside;    // mod files outside the install folder: cannot be handled automatically
    wstring legacyRecordDir;         // folder with an older install's record (byte-exact restore with /REMOVEOLD)
    std::vector<wstring> legacyFiles;// an older install's loose files without a record (moved into the backup)
    bool modsFound() const { return !mods.empty(); }
    bool oldFound() const { return !legacyRecordDir.empty() || !legacyFiles.empty(); }
};

bool IsLegacyLoader(const wstring& path);
// The player's OptiScaler (its marker files, and every DLL a game loads by name whose fingerprint is OptiScaler) and
// older WrongGPU installs. The DLSS-NR on AMD mod is never in here: it is refused and never touched (AssessCoexistence).
Conflicts FindConflicts(const wstring& dir, const wstring& exeDir);

// ---- the rules table (design section 5) ------------------------------------------------------------------------------
enum class Decision {
    Managed,    // ours (record + hash): WrongGPU may update or remove it
    AskSwitch,  // the player's OptiScaler: install only with the player's OK (switch; their files backed up as moved)
    Chain,      // stays where it is; WrongGPU works next to it (OptiScaler takes another proxy name)
    Warn,       // stays; a warning for the player
    Inform,     // shown only
    Refuse,     // WrongGPU is not installed in this game (no override)
    Migrate,    // an older WrongGPU / DLSS 4.5 AMD install: removed first (byte for byte) with the player's OK
};
const wchar_t* DecisionName(Decision d);  // "managed", "ask-switch", "chain", "warn", "inform", "refuse", "migrate"
bool ParseDecision(const wstring& s, Decision& d);

struct Finding {
    Tool tool = Tool::Unknown;
    Decision decision = Decision::Inform;
    wstring what;                 // "ReShade 6.3.3 as dxgi.dll"
    std::vector<wstring> files;   // its files, relative to the exe folder (processes: their exe names)
    wstring evidence;             // the fingerprint ("version resource: ReShade", "text \"dxvk\"", "file neuralhub.ini")
    wstring message;              // what WrongGPU does about it, for the player
    int code = 0;                 // Refuse: the exit code of an install
};

struct ProxyChoice {
    wstring name;                 // OptiScaler's DLL name next to the exe (empty: no name is free)
    wstring reason;               // why this name (recorded as the record's proxy line)
    bool untested = false;        // next to ReShade / Special K: an untested combination
};

struct CoexistReport {
    std::vector<Finding> findings;
    ProxyChoice proxy;
    bool asiLoader = false;       // Ultimate ASI Loader: WrongGPU's plugin goes to its own folder (kAsiPluginOwnDir)
    bool hipTool = false;         // another HIP-based tool: version.dll is never taken
    wstring playersOptiProxy;     // the name the player's OptiScaler loads as (reused by a switch)
    std::vector<wstring> foreignPlugins;  // 0.6.1: other setups' .asi files in OptiScaler's plugin folders (ForeignPlugins)
    bool foreignPluginsBlockRr = false;   // ...and the game uses Ray Reconstruction: WrongGPU's RR stays out unless they are moved aside
    GraphicsApi api;
    const Finding* refusal() const;   // the first Refuse finding, or null
    bool has(Tool t) const;
    bool needsSwitch() const;
    std::vector<wstring> Warnings() const;  // the messages of Chain and Warn findings (shown before and after an install)
};

// Proxy names in the order the picker tries them, and names it never takes (other tools' usual names).
inline const wchar_t* const kProxyOrder[] = {L"dxgi.dll", L"winmm.dll", L"version.dll", L"wininet.dll", L"winhttp.dll", L"dbghelp.dll"};
inline const wchar_t* const kNeverProxy[] = {L"d3d12.dll", L"dinput8.dll", L"dwmapi.dll", L"xinput1_3.dll", L"xinput1_4.dll", L"xinput9_1_0.dll"};
// The plugin's folder when Ultimate ASI Loader is present (it would load plugins\WrongGPU.asi a second time).
const wchar_t* const kAsiPluginOwnDir = L"wronggpu-plugins";
// 0.6.1: the folders OptiScaler loads plugins from (next to it, in its own OptiScaler\ folder, or WrongGPU's own folder).
inline const wchar_t* const kPluginDirs[] = {L"plugins", L"OptiScaler\\plugins", L"wronggpu-plugins"};
// The .asi files in those folders next to the exe that are not WrongGPU's install's own (the record copied them there with
// that hash), relative to exeDir. Another setup's plugin may hook Ray Reconstruction as WrongGPU's RR plugin does: by default
// WrongGPU's RR stays out of such a game (with a warning); with the player's OK (Consent::movePlugins) they are moved aside
// (backed up, put back on uninstall) and WrongGPU's RR goes in. Never by a plugin's name alone. recordDir: the folder the
// record's paths are relative to (the install folder; empty: exeDir); a record of a folder that exeDir is not in names none.
std::vector<wstring> ForeignPlugins(const wstring& exeDir, const Record* ours, const wstring& recordDir = L"");
// 0.6.1: what an install does with those plugins (exeDir-relative, from ForeignPlugins) in a Ray Reconstruction game. Each
// one the player agreed to move (`agreed`, as the question listed them) or that an earlier install moved aside with the
// player's OK (`movedBefore`: its record's disable entries, lower-case path relative to dir -> the hash moved; byte for byte
// that file) goes into moveAside, as a path relative to the install folder dir. Any other one is in `blocking`: WrongGPU's RR
// stays out. When exeDir is not dir or a folder below it nothing can be moved: every plugin is blocking (outside: true).
struct PluginPlan {
    std::vector<wstring> moveAside;  // relative to dir
    std::vector<wstring> blocking;   // relative to exeDir
    bool outside = false;            // a blocking plugin is outside the install folder
};
// `agreed`: (path relative to exeDir, the SHA-256 it had when the player agreed); a plugin counts as agreed only while its
// bytes are still those.
PluginPlan PlanPluginsAside(const wstring& exeDir, const wstring& dir, const std::vector<wstring>& plugins,
                            const std::vector<std::pair<wstring, wstring>>& agreed, const std::map<wstring, wstring>& movedBefore);
// 0.6.1: WrongGPU's RR plugin exports this marker (clean/rr_f/proxy/include/wg_rr_settings_contract.h,
// WG_RR_CONTRACT_EXPORT). A plugin file that exports it is WrongGPU's own RR plugin whatever its name or hash (a copy another
// package placed): never "another setup's plugin", never moved aside. Read from the file's PE export table, never by
// loading it.
const char* const kRrContractExport = "WrongGPU_GetSettingsContract";
bool ExportsRrContract(const wstring& path);
// Those copies in OptiScaler's plugin folders that the install record doesn't name with that hash (relative to exeDir): the
// rules table shows them as "WrongGPU RR (from another package)".
std::vector<wstring> OurPluginsFromOtherPackages(const wstring& exeDir, const Record* ours);
// The warning when they keep WrongGPU's RR out, and the line of a plugin moved aside with the OK.
wstring ForeignPluginsText(const std::vector<wstring>& plugins);

// Every finding for one game (exe) and its install folder (dir): fingerprints of the DLLs next to the exe, the tools'
// folders and marker files, running overlays, anti-cheat, the graphics API, older installs. ours: the game's install
// record (its files are not "other tools"); null when there is none.
CoexistReport AssessCoexistence(Context& ctx, const wstring& exe, const wstring& dir, const Record* ours);
// The proxy picker (design section 5): our earlier install's name, else the first name of kProxyOrder that is free,
// never a name a recognised tool, an unknown file or a game's own Windows DLL holds (version.dll never when hipTool);
// the name the player's OptiScaler had (a switch moves it aside) only when no other is free. The reason says what
// was skipped.
ProxyChoice PickProxy(const wstring& exeDir, const Record* ours, const wstring& playersOptiProxy, bool hipTool, bool untested);
// Process names of this PC (the overlays rule). WRONGGPU_TESTING builds see only WRONGGPU_TEST_PROCESSES ("a.exe;b.exe";
// none when it is not set), so the tests do not depend on what runs on the PC.
std::vector<wstring> RunningProcesses();
// The overlay or injector a process name belongs to (null: none) and its decision and message.
const wchar_t* OverlayFor(const wstring& process, Decision& d, const wchar_t** message);
}  // namespace wg
