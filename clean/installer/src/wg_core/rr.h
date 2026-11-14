// SPDX-License-Identifier: MIT
// WrongGPU launcher core: Ray Reconstruction (RR) in the games that use it (build/launcher/ui-options/BE-wide/rr-notes.md).
// A game "uses Ray Reconstruction" when it ships the DLSS Ray Reconstruction files (the game interface's RR plugin or the
// RR model file), or is one of the few games known to call it directly. For those games a WrongGPU install also places
// the RR files of this build: WrongGPU's RR plugin (an .asi OptiScaler loads from its plugins folder, beside WrongGPU's
// own plugin), the RR runtime next to the game exe, wronggpu-rr.ini (rr_model: the RR data) and the plugin's settings
// file WrongGPU\wronggpu.ini ([wronggpu] rr_enabled, rr_mode, rr_exact_allowed; the plugin saves rr_mode there when the
// player switches it in OptiScaler's menu). Every one is an entry of the install record (staging, rollback, uninstall
// byte for byte); the two settings files go at uninstall even after the plugin rewrote them.
// The RR data (preset F) is converted once per PC from the player's own RR model file with WrongGPU's converter, as the
// model data is (model.h), into %LOCALAPPDATA%\WrongGPU\model\F\model.wgcm: shared by every game (a game whose own file is
// too old uses it too). The file is never copied or changed, and never downloaded or shipped. Written for this project.
#pragma once
#include "convert_ignore.h"
#include "detect.h"
#include "profile.h"

namespace wg {

// The RR files of an install (relative to the install folder; the plugin's folder is the one of WrongGPU's plugin).
const wchar_t* const kRrPluginName = L"WrongGPU-RR.asi";
// release 0.6.0 (owner 2026-11-02: one runtime DLL): the RR runtime is part of wronggpu_runtime.dll (the payload's runtime,
// beside the Model M interface); no RR runtime file of its own. kRrLegacyRuntime: the file of the first 0.6.0 build and
// of the RR test builds, which make_payload.py refuses in an RR folder (an install record that has it removes it as usual).
const wchar_t* const kRrRuntime = L"wronggpu_runtime.dll";
const wchar_t* const kRrLegacyRuntime = L"wronggpu_rr_runtime.dll";
const wchar_t* const kRrIni = L"wronggpu-rr.ini";
const wchar_t* const kRrSettings = L"WrongGPU\\wronggpu.ini";
const wchar_t* const kRrDataDir = L"WrongGPU";  // the plugin's own folder next to the exe (its settings file)
const wchar_t* const kRrLog = L"wronggpu-rr.log";
// The RR payload's files as make_payload.py embeds them (relative to the payload's RR folder): the general RR plugin, builds
// of it for one version of the game interface it hooks (plugins\<major>.<minor>\WrongGPU-RR.asi: the plugin works only with
// the interface version it was built for). Never OptiScaler or any file of it (owner 2026-11-02: the OptiScaler fork is GPL and
// not part of WrongGPU's download): make_payload.py refuses kRrPayloadEdition, the place an earlier build embedded WrongGPU's
// edition of OptiScaler; the launcher downloads that edition from its own project instead (optiedition.h).
const wchar_t* const kRrPayloadPlugin = L"plugins\\WrongGPU-RR.asi";
const wchar_t* const kRrPayloadEdition = L"optiscaler-edition\\OptiScaler.dll";
// (inline: the restorer uses them without the rest of this module)
inline bool IsRrPlugin(const wstring& rel) { return IEq(BaseName(rel), kRrPluginName); }  // WrongGPU-RR.asi in any folder
// wronggpu-rr.ini or WrongGPU\wronggpu.ini (the plugin rewrites them)
inline bool IsRrSettingsFile(const wstring& rel) { return IEq(rel, kRrIni) || IEq(rel, kRrSettings); }
inline bool RecordHasRr(const Record& m) {  // the install placed the RR files
    for (auto& e : m.e) if (e.action == L"copy" && IsRrPlugin(e.rel)) return true;
    return false;
}

// ---- which games ---------------------------------------------------------------------------------------------------
// direct: the game creates Ray Reconstruction through the RR library itself, not through the game interface: the RR library
// is there and the game interface's RR plugin is not (the layout of Unreal Engine's plugin: Hogwarts Legacy, LEGO Batman,
// Borderlands 4), or a known game that does so. WrongGPU's RR plugin gets such a game's RR only through WrongGPU's edition
// of OptiScaler (its plugin interface 4 or newer hands the game's RR calls to the plugin); with another OptiScaler the
// game keeps its own ray tracing.
struct RrUse { bool uses = false, direct = false, notOffered = false; wstring why; };  // notOffered: WrongGPU's RR not offered here (why: the line)  // why: the file found ("bin\x64\nvngx_dlssd.dll") or "known game"
// Looks next to the exe, then through the game's folder (bounded walk). Cached per exe for this process (a library scan
// reads each game once); Forget clears the cache (Rescan).
RrUse DetectRr(const wstring& exe);
RrUse DetectRrUncached(const wstring& exe);
void ForgetRrDetection();
// The OptiScaler of an install is WrongGPU's edition (optiedition.h: the pinned DLL), read from the install record: the copy
// entry of OptiScaler's proxy DLL has the edition DLL's hash. A game whose RR is direct needs it (RrUse::direct).
bool RecordHasEdition(const Record& m);
// The plain reason, for the log and the result's warnings, why WrongGPU's RR isn't offered in a direct-RR game with an
// OptiScaler that is not the edition (`game`: the game's name).
wstring RrDirectNoEditionText(const wstring& game);

// ---- this build's RR files -------------------------------------------------------------------------------------------
bool RrPayloadAvailable(Context& ctx);  // the RR plugin is built in (the RR runtime is part of wronggpu_runtime.dll)
// The general plugin or a build of it for one interface version (plugins\2.8\WrongGPU-RR.asi); the version it is for
// ("2.8"; empty: the general one).
bool IsRrPayloadPlugin(const wstring& rel);
wstring RrPluginVersion(const wstring& rel);
// The game interface the RR plugin hooks (its file name comes with the RR payload: ctx.data->rrInterface): the file
// version of the game's copy next to the exe, else of the first one in the game's folder (a bounded walk). Empty: none (a
// game that calls Ray Reconstruction directly), or this build names no interface file. where: the file read.
wstring GameInterfaceVersion(Context& ctx, const wstring& exe, wstring* where = nullptr);
// Which plugin of `rels` (payload names) goes into a game whose interface has `version` ("2.8.0.0"): the build for its
// major.minor when there is one, else the general plugin. Empty: none.
wstring PickRrPlugin(const std::vector<wstring>& rels, const wstring& version);
// The payload file of that plugin (null: no RR plugin in this build); *forVersion: "2.8" for a build of its own, else empty.
const PayloadFile* RrPluginFor(Context& ctx, const wstring& version, wstring* forVersion = nullptr);

// ---- the RR data (preset F) ----------------------------------------------------------------------------------------
struct RrDataState {
    bool ok = false;          // a usable file: reused, or converted now (written by WriteRrData)
    bool reused = false;
    bool available = true;    // false: this build has no RR converter data
    wstring dll, origin, version, error;
    std::vector<wstring> rejections;  // "path: reason" for every file tried
    IgnoreReport ignored;             // the files the ignore lists left out of the search, and the lists' warnings (convert_ignore.h)
    // The data on this PC checked against the ignore lists before it is reused (as ModelState::Trust): LISTED is never
    // reused; UNKNOWN (no SHA-256 on record) is converted again from an allowed file, else kept with a line in `notes`.
    enum Trust { TRUSTED, LISTED, UNKNOWN } trust = TRUSTED;
    std::vector<wstring> notes;       // lines for the player (the install's warnings)
    wstring dllSha256;                // the SHA-256 of `dll` (written to model-source.txt)
    std::shared_ptr<wgc::Converted> converted;
};
bool LoadFMap(Context& ctx);
wstring RrDataDir();          // %LOCALAPPDATA%\WrongGPU\model\F
wstring RrDataFile();         // ...\F\model.wgcm (named in every RR game's wronggpu-rr.ini)
bool RrDataValid(Context& ctx, wstring* why = nullptr);
// A swap of WriteRrData that stopped between its two renames (the process ended) left the data in use in F.old and no F:
// it goes back (the next swap would delete F.old first; the games name F's file). Only where no swap can run meanwhile:
// the converters themselves (FindRrData, WriteRrData) and the window's start; never from a read (RrDataValid).
void RecoverRrData(Context& ctx);
// model-source.txt beside it: the file it came from (empty when unknown)
wstring RrDataSource();
// The file on this PC when it is valid (unless `fresh`); else `userDll` when given, then this game's RR model files, then
// the library games' (each converted in memory until one converts completely).
RrDataState FindRrData(Context& ctx, const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries, bool fresh = false);
// Writes a fresh conversion beside the folder in use, checks it there and moves it into place (a valid earlier file is
// never lost); a reused file is checked again. False with *why.
bool WriteRrData(Context& ctx, const RrDataState& m, wstring* why);
// An installed RR game whose settings file could not be written (its folder, the reason): a protected folder, written
// again by an elevated Apply (the window), else RR stays off there and the player is told.
struct RrSettingsFailure { wstring dir, why; };
// Settings' Convert again / Choose a DLSS file... and a game page's Convert: a fresh conversion (`dll` alone when given: no
// other file is tried; else the RR model files of the launcher's games, then of `libraries`), written while the data in
// use stays until the new one is checked; then every installed RR game's settings file follows (RR turns on where its
// profile wants it; the ones that could not be written in *notWritten). False with *why. *ignored: the files the ignore lists
// left out (convert_ignore.h; a chosen file on a list is not converted either) and the lists' warnings.
bool ConvertRrData(Context& ctx, const wstring& dll, const std::vector<StoreGame>& libraries, wstring* why, std::vector<RrSettingsFailure>* notWritten = nullptr,
                   IgnoreReport* ignored = nullptr);
// A managed game that uses the RR data and runs now (its name; empty: none): the data cannot be replaced meanwhile.
wstring RunningRrGame();

// ---- settings ------------------------------------------------------------------------------------------------------
// WrongGPU\wronggpu.ini from the profile (rr_enabled, rr_mode; rr_exact_allowed = 1 only while the mode is exact: the
// plugin runs Exact only with it). RR stays off (rr_enabled=0) while the RR data is not ready, whatever the profile says;
// other lines and keys (the plugin's) stay as they are.
string RenderRrSettings(const Profile& p, bool dataReady, const string& in);
string RrIniText(const wstring& dataFile);  // wronggpu-rr.ini as an install writes it
struct RrFileState { bool present = false, enabled = true; string mode; };
RrFileState ParseRrSettings(const string& text);
RrFileState ReadRrSettings(const wstring& dir);
// The mode the player chose in the game (OptiScaler's menu saves it into WrongGPU\wronggpu.ini) becomes the profile's:
// the page shows it and an update keeps it. True when it changed `p`.
bool FoldRrFromGame(const wstring& dir, Profile& p);
// WrongGPU\wronggpu.ini of an installed RR game written again for the RR data's state (after a conversion: RR turns on
// where the profile wants it; `dataReady`: RrDataValid, read once by the caller). The record's hash follows. 1: written;
// 0: nothing to write (not an RR install, the same text); -1: the folder refused it (*why).
int RefreshRrSettings(Context& ctx, const wstring& dir, bool dataReady, wstring* why = nullptr);
// Every managed RR game's settings file (but `except`'s, which its own operation writes) brought to the RR data's state:
// after the data was converted, the games installed before it get RR. Returns the ones that could not be written.
std::vector<RrSettingsFailure> RefreshAllRrSettings(Context& ctx, const wstring& except = L"");

}  // namespace wg
