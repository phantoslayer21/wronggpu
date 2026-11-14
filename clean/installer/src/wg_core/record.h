// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the install record (v1), the "Apps & features" entry and the restore / uninstall engine.
// Written for this project.
#pragma once
#include "context.h"
#include "optikeys.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// Install record. Plain UTF-8 text, tab separated:
//   WRONGGPU-RECORD 1
//   key <TAB> value                      (version, game = install folder, exe = full path, status, regkey, model;
//                                         model-m = what this install did with the model M file: converted, reused,
//                                         replaced or kept, with the reason and where the output filter comes from)
//   action <TAB> existed(0/1) <TAB> originalSha256|- <TAB> installedSha256|- <TAB> relativePath
// Actions: copy    = a payload file written at the path (the file that was there, if any, is in the backup)
//          rename  = the game's FidelityFX DLL, kept under <name>.wronggpu-original.dll (installedSha256 = its hash)
//          disable = a file moved into the backup (another mod's file, or an older install's loose file)
//          dir     = a folder the installer created (removed on uninstall if empty)
// The 0.1.x/0.2.x record (DLSS45AMD-MANIFEST, actions copy/disable) is read by the same parser.
//
// Record v2 (WRONGGPU-RECORD 2, the launcher; build/launcher-design.md section 2): the same lines, and
//   payload <TAB> version <TAB> runtimeSha256|- <TAB> frontEndSha256|- <TAB> pluginSha256|- [<TAB> rrPluginSha256]
//                                    (what this install wrote; the sixth field, rc6: the build of the Ray Reconstruction
//                                    plugin it placed, RrPluginFor, written only by an install with the RR files)
//   proxy <TAB> name <TAB> reason                    (OptiScaler's proxy DLL name and why it was chosen)
//   owner <TAB> action <TAB> relativePath <TAB> owner   (one per action line)
//   inikey <TAB> file <TAB> section <TAB> key <TAB> original <TAB> ours   (the launcher's OptiScaler.ini journal, optikeys.h;
//                                    a value is written as "=value"; an original "absent" means the key was added)
//   coexist <TAB> tool <TAB> fingerprint <TAB> decision   (what the install found next to the game and decided, coexist.h)
//   rrconfig <TAB> path <TAB> flags <TAB> previous <TAB> folder   (rrconfig.h: the game's Engine.ini outside the game folder that
//                                    the player's rr_config_fix=on changed; flags: c created, s section added, w was read-only,
//                                    r set read-only, b backup kept; previous: "absent" or "=value" of the key before; folder:
//                                    the topmost folder the fix created, or "-")
//   runtime <TAB> folder <TAB> pattern   (0.6.1: files the install's own programs write while the game runs, by WrongGPU's own
//                                    unique name: in that folder (relative to the install folder, "." for it) the files matching
//                                    pattern, which starts with "wronggpu-"; the only files an uninstall removes by name)
// Copy entries of settings files (wronggpu.ini, OptiScaler.ini, the RR plugin's): installedSha256 is the hash of what the
// launcher last wrote there (every write of the launcher updates it).
// Owners: wg (WrongGPU's payload and its folders), wg-optibundle (OptiScaler files copied from the player's package:
// ours to remove), adopted (a player's file used but never deleted), moved (a player's or the game's file moved or
// renamed aside and put back byte for byte). Action lines keep their five fields, so a 0.5.x uninstaller still reads a
// v2 record (it skips the lines it does not know). Lines of a later version are kept as they are.
// ------------------------------------------------------------------------------------------------
struct Entry { wstring action, rel; bool existed = false; wstring orig, inst, owner; };
// The rrconfig line (rrconfig.h): what "Fix it for me" did to the game's Engine.ini, so that =off and the uninstall undo it.
struct RrConfigFix {
    bool present = false;
    wstring path;                 // the Engine.ini (absolute: the player's folder, resolved when it was fixed)
    bool created = false, sectionAdded = false, wasReadOnly = false, setReadOnly = false, backup = false;
    bool prevAbsent = true;       // the key was not in the section before
    string prev;                  // else its value before
    wstring folder;               // the topmost folder the fix created ("" none)
};
// A runtime line's folder for the game root (GameRoot of the record's exe folder) when it is above the install folder.
inline const wchar_t* const kRuntimeGameRoot = L"<game-root>";
struct Record {
    wstring version, game, exe, status, regkey, model, modelM;
    std::vector<Entry> e;
    bool legacy = false;
    int format = 1;                                   // 1: 0.5.x setup, 2: the launcher
    wstring payloadVersion, runtimeSha, frontEndSha, pluginSha, proxy, proxyReason;
    wstring rrPluginSha;                              // the payload line's sixth field (empty: no RR files, or a record before rc6)
    std::vector<wstring> extra;                       // lines this version does not know (written back unchanged)
    std::vector<IniKeyChange> inikeys;                // the OptiScaler.ini journal
    struct CoexistLine { wstring tool, fingerprint, decision; };
    std::vector<CoexistLine> coexist;
    RrConfigFix rrconfig;
    struct RuntimeLine { wstring folder, pattern; };  // folder: relative to the install folder, or kRuntimeGameRoot
    std::vector<RuntimeLine> runtime;                 // the runtime lines (an older record has none: nothing removed by name)
    // rc6 L1 (in memory only): the record was read from RecordPath(dir) but named another folder (the game folder was moved:
    // Steam's "Move install folder", another drive letter): LoadRecord rebased game, exe and the journal's paths to dir, and
    // this is the folder it named. SaveRebasedRecord writes the rebased record.
    wstring movedFrom;
};
// Fills in the owner of every entry that has none (a v1 record, or a new install): see the list above.
void InferOwners(Context& ctx, Record& m);
// Sets the payload line from the embedded payload (version and the hashes of the runtime, front end and plugin; with the RR
// files also the RR plugin build this payload has for the game, RrPluginFor of its game interface version).
void SetPayloadLine(Context& ctx, Record& m);
// rc5: the record's payload line names other files than this launcher's payload (runtime, front end, menu plugin hashes,
// and since rc6 the RR plugin's: compared as SetPayloadLine writes them for this record; a record before rc6 without the
// sixth field is read by its RR plugin entry's installed hash), or it has none: the game holds another build's files. Read
// together with the version: at the same version this is an update (the window's Update / Repair), never a write here.
bool PayloadDiffers(Context& ctx, const Record& m);
bool RecordCopies(const Record& m, const wstring& rel);  // the record has a copy entry for rel
// owned-files.json: the public, read-only list of the files WrongGPU owns in this folder (for mod managers).
string OwnedFilesJson(const Record& m);
void WriteOwnedFiles(const wstring& dir, const Record& m);
// v1 -> v2 in place: the old file is kept as install.record.v1. false when there is no v1 record.
bool UpgradeRecord(Context& ctx, const wstring& dir, std::vector<wstring>* notes = nullptr);
// Every file the record says WrongGPU wrote (owner wg / wg-optibundle, copy or rename), checked against its installed
// hash. Settings files (wronggpu.ini, OptiScaler.ini) count only when missing. Empty: the install is complete.
std::vector<wstring> VerifyInstall(const wstring& dir, const Record& m);
// The launcher's list of managed game folders (%LOCALAPPDATA%\WrongGPU\managed-games.txt): with one Apps & features
// entry for all games, an uninstaller reads it to know whether another game still uses the shared model data.
std::vector<wstring> ManagedGames();
void AddManagedGame(const wstring& dir);
void RemoveManagedGame(const wstring& dir);

wstring StateDir(const wstring& dir);
wstring RecordPath(const wstring& dir);
wstring BackupPath(const wstring& dir, const wstring& rel);
wstring LegacyStateDir(const wstring& dir);
wstring LegacyRecordPath(const wstring& dir);
wstring LegacyBackupPath(const wstring& dir, const wstring& rel);
// Reads a record. One read from <dir>\wronggpu-setup\install.record that names another folder is the record of dir all the
// same (a moved game folder): rebased to dir in memory (Record::movedFrom), never refused for it.
bool LoadRecord(const wstring& path, Record& m);
void SaveRecord(const wstring& path, const Record& m);
// rc6 L1: a rebased record (movedFrom) written in its folder: OptiScaler.ini's paths into the old folder (its front end, its
// plugins folder) pointed at the new one, the record and owned-files.json saved, the game's own Apps & features entry (a
// record with a regkey) pointed at the new folder, and on the managed list the new folder in the old one's place (the old
// one stays while its drive is not connected). Best effort, never throws (a folder this process may not write keeps its
// record as it is, rebased again at each read); true when written. movedFrom is cleared then.
bool SaveRebasedRecord(Context& ctx, const wstring& dir, Record& m);
wstring Target(const wstring& dir, const wstring& rel);
bool HostedRecord(const Record& m);
// ---- the file-ownership rule (0.6.1; README "What the launcher removes") ----------------------------------------------
// Every restore, uninstall, repair, update, move out of a wrong folder and cleanup deletes or overwrites a file only when
// its current bytes are WrongGPU's: the hash the record says WrongGPU last wrote there, or a WrongGPU release's (size and
// SHA-256). Not by its name, a neighbouring file, an ini key or because it is a log. The one exception: the game's FSR DLL
// slots, where WrongGPU's DLL is recognised by its version resource (ClassifySlot), because the game may update that DLL.
// The other exception the rule names: files with WrongGPU's own unique names (wronggpu-* created by the install, the RR
// plugin's files in its WrongGPU folder) that the record lists as created by WrongGPU.
// Anything else at a recorded path stays as it is; a player's original that cannot be put back because of it is kept in
// <install dir>\WrongGPU-originals\<time>\ (KeptOriginal), and the result says "partly" (kPartlyRestored).
enum class Holds { Missing, Ours, Original, Other };
// What the path of entry `e` holds now (h: its current SHA-256; empty when missing).
Holds WhatIsThere(Context& ctx, const wstring& dir, const Entry& e, const wstring& h);
// The warnings of files a restore could not put back or had to leave begin with this ("partly" in the result).
extern const wchar_t* const kPartlyRestored;
bool AnyPartly(const std::vector<wstring>& warnings);
// The player's original of `rel` (from `source`, its SHA-256 `sha`) kept in the folder that survives: an earlier kept copy
// with these bytes is used again, else a new <time> generation. Checked byte for byte; throws on failure. Returns its path.
wstring KeptOriginal(const wstring& dir, const wstring& rel, const wstring& source, const wstring& sha);
// The OptiScaler.ini in `dir` is the one WrongGPU last wrote (its hash is the record's). The restore goes by this alone.
bool OptiIniWrittenByUs(const wstring& dir, const Record& m);
// The OptiScaler setup next to the game is still WrongGPU's: the OptiScaler DLL the record installed is there under its name
// with its hash, or the ini is the one WrongGPU last wrote. Used only to decide what the scan reports and asks about (the
// switch question) and which menu-saved values carry over; never to delete or overwrite a file.
bool OptiSetupIsOurs(const wstring& dir, const Record& m);
// Test hook: called at the named steps of DoRestore ("restored", "kept", "record-removed"); a test stops the restore there
// (as a killed process would). Test builds also stop the process at the step named by WRONGGPU_TEST_STOP_AT.
void SetRestoreStepHook(std::function<void(const char*)> hook);
// The hive the Apps & features entries go to (HKEY_CURRENT_USER unless set: the elevated worker of another account sets
// the player's hive, HKEY_USERS\<sid>).
HKEY UninstallHive();
void SetUninstallHive(HKEY hive);
wstring RegKeyFor(const wstring& dir, const wchar_t* prefix = kRegPrefix);
void RegisterUninstall(Context& ctx, const Record& m, const wstring& state, const wstring& displayName);
std::vector<wstring> RegisteredInstalls();
// A game other than dir that still has WrongGPU (an Apps & features entry or the launcher's list, with its record; a stale
// entry, its folder gone by hand, is none; one on a drive that is not connected counts). Empty: none, so dir's uninstall is
// the last one.
wstring OtherGameWithWrongGPU(const wstring& dir);
// The drive or network share a folder is on cannot be reached (unplugged, offline, not ready): its root does not answer.
// A folder whose drive answers but that is gone was deleted.
bool DriveUnreachable(const wstring& dir);
// A launcher uses the shared data in %LOCALAPPDATA%\WrongGPU (the model and RR data): installed for this user, or its
// settings (launcher.ini) are there. Then the last game's uninstall leaves that data; the launcher's own uninstall removes it.
// A portable launcher also keeps it (its settings are next to it, not there): the portable copies that used the shared data
// note themselves in it (portable-launchers.txt), and one that is still there with its marker counts. The note is only
// added to a shared folder that is there already (never made for it).
bool LauncherKeepsSharedData();
// Only the portable part of that: a portable copy noted in portable-launchers.txt is still there with its marker. The
// launcher's own uninstall keeps the shared data for it too (SelfUninstall), and its question says so.
bool PortableLauncherUsesSharedData();
void NotePortableLauncher(const wstring& exe);
void UnregisterUninstall(const Record& m);
// The files the record's runtime lines name (WrongGPU's own unique names, not in the record's entries): removed. Returns
// how many.
size_t RemoveRuntimeFiles(const wstring& dir, const Record& m);
// The runtime lines for an install in `dir` with its game program in exeDir (folders inside dir only), from the record's
// dir entries too: call after those are in m.
void SetRuntimeLines(Record& m, const wstring& dir, const wstring& exeDir);
// The shared folder (appData) is moved aside, to a free name made unique with a process-wide sequence number, and deleted by a
// hidden cmd.exe; its ignore-files.txt is put back first (convert_ignore.h PutIgnoreListBack). Returns PutIgnoreListBack's line
// for the player ("kept ignore-files.txt ...", the list kept as ignore-files.txt.kept-<n>, or left in the moved folder) or "".
// A list that could not leave the moved folder is never deleted: the rest of that folder goes at once, the folder stays with
// the list, and the hidden cmd.exe does not delete it. The move: a sharing violation (32) or an access denied (5: a scanner holding a file in the
// folder) is tried again, at most kCleanupMoveTries times (waits from kCleanupMoveFirstWaitMs doubling: 50, 100, 200, 400 ms),
// each wait logged with the error; any other error, or the last try, leaves the folder in place ("Shared data cleanup deferred
// ... (error N: ...)").
constexpr int kCleanupMoveTries = 5;
constexpr unsigned long kCleanupMoveFirstWaitMs = 50;
// Test seam: the call StartCleanup moves the folder with (MoveFileExW when unset). The tests put in one that fails with
// a chosen error for the first tries; nullptr puts MoveFileExW back.
using CleanupMoveFn = BOOL (WINAPI*)(LPCWSTR from, LPCWSTR to, DWORD flags);
void SetCleanupMoveForTest(CleanupMoveFn fn);
wstring StartCleanup(Context& ctx, const wstring& state, const wstring& self, const wstring& appData);
// The restore of an install, by the file-ownership rule above. Before anything changes it checks every recorded file
// (refused, "Nothing was changed", when one cannot be read or replaced: another program holds it, or a backup is missing).
// Then, per entry: WrongGPU's file goes (or the player's original from the backup takes its place); the original already
// there stays; anything else stays as it is, and a backed-up original that cannot go back is kept (KeptOriginal) with a
// kPartlyRestored warning naming both. Each file is checked again right before it is changed. The record is deleted
// before the backup, so a stop at any point leaves either a record that a second run completes, or every original at its
// path or in the kept folder. keepState (update, repair, a move): the state folder stays (record and backup cleared).
// keptWarnings: the warning of each recorded file left in place, by its path (the update path replaces the warning of a
// file it then moves aside with the player's OK).
std::vector<wstring> DoRestore(Context& ctx, const wstring& dir, Ui& ui, bool failedInstall, bool keepState,
                               std::map<wstring, wstring>* keptWarnings = nullptr);
// rc6 L4: the player's things in the state folder that a failed install or update never removes when they were there before
// it: profile.ini (the game's settings), previous\ (the version Roll back goes to) and rr-config\ (the Engine.ini fix's backup).
bool StateHasSettings(const wstring& dir);
// The state folder emptied of everything else (the record, the backup, the uninstaller, logs); removed when nothing is left.
void RemoveStateKeepingSettings(const wstring& dir);
std::vector<wstring> LegacyRestore(Context& ctx, const wstring& dir, Ui& ui);
}  // namespace wg
