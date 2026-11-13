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
// 0.6.1: the OptiScaler.ini in `dir` is still the one of WrongGPU's setup: its keys point at WrongGPU's files (FfxDx12SRPath
// at WrongGPU's front end, or Dx12Upscaler at WrongGPU's custom upscaler slot, whichever the record installed). OptiScaler's
// menu saving it keeps them; a file whose keys point elsewhere is another setup's (it replaced or rewrote WrongGPU's): never
// removed or replaced without the player's OK. A record with neither front end: true (nothing to compare).
bool OptiIniIsOurs(const wstring& dir, const Record& m);
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
size_t RemoveRuntimeFiles(const wstring& dir, const wstring& exeDir, const Record& m, bool settings);
void StartCleanup(Context& ctx, const wstring& state, const wstring& self, const wstring& appData);
// keptWarnings (0.6.1): the warning of each recorded file left in place because it changed since the install, by its path (the
// update path replaces the warning of a file it then moves aside with the player's OK).
std::vector<wstring> DoRestore(Context& ctx, const wstring& dir, Ui& ui, bool failedInstall, bool keepState,
                               std::map<wstring, wstring>* keptWarnings = nullptr);
// rc6 L4: the player's things in the state folder that a failed install or update never removes when they were there before
// it: profile.ini (the game's settings), previous\ (the version Roll back goes to) and rr-config\ (the Engine.ini fix's backup).
bool StateHasSettings(const wstring& dir);
// The state folder emptied of everything else (the record, the backup, the uninstaller, logs); removed when nothing is left.
void RemoveStateKeepingSettings(const wstring& dir);
std::vector<wstring> LegacyRestore(Context& ctx, const wstring& dir, Ui& ui);
}  // namespace wg
