// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the engine setting Ray Reconstruction needs in Borderlands 4 and Hogwarts Legacy (Brief 16,
// owner 2026-11-04). Both games create Ray Reconstruction themselves only with one line in the player's own Engine.ini
// (rrconfig_line.h), section [SystemSettings]; Borderlands 4 deletes that file a few seconds after every start unless it is
// read-only. The file is in the player's profile, outside the game folder:
//   Borderlands 4:   <Documents known folder>\My Games\Borderlands 4\Saved\Config\<sub>\Engine.ini
//   Hogwarts Legacy: %LOCALAPPDATA%\Hogwarts Legacy\Saved\Config\<sub>\Engine.ini
// <sub>: Windows or WindowsNoEditor, the one that holds the game's GameUserSettings.ini. The game makes it at its first start;
// before that the sub-folder isn't known (Hogwarts Legacy uses WindowsNoEditor, Borderlands 4 Windows), so nothing is made in
// a guessed one: "Fix it for me" then keeps the choice (rr_config_fix=on, no rrconfig line yet) and the line goes in once the
// folder is there (the next Apply: the launcher applies it when it opens or shows the game's page, and Repair/Update) (UI
// review r2b #1).
//
// The per-game setting rr_config_fix (profile.cpp; ask by default: the launcher asks once after an install, the CLI says
// how) decides. on ("Fix it for me", ApplyRrConfig): the file backed up into the game's wronggpu-setup folder (or noted as
// absent), the line added only where it is missing (under the existing [SystemSettings], or that section added; a key with
// another value set to the line's), Borderlands 4's file made read-only, all of it in the install record (rrconfig line).
// off or the uninstall (RevertRrConfig): the line taken out again (or the key's earlier value put back), the section if it
// was added and is empty now, read-only cleared unless the file had it before, the file deleted if the fix created it and
// nothing else is in it; a file that is gone comes back from the backup. Repair keeps the record and applies it again.
//
// Test builds (WRONGGPU_TESTING) resolve the Documents folder only from WRONGGPU_TEST_DOCUMENTS and LocalAppData only from
// a LOCALAPPDATA that is a test root: without them the fix refuses, so a test never touches a real profile. Written for this
// project.
#pragma once
#include "context.h"
#include "record.h"

namespace wg {

struct RrConfigGame {
    const wchar_t* exe;        // the game's exe name (the launcher's match)
    const wchar_t* name;       // its name for the player
    bool documents;            // the Documents known folder (else LocalAppData)
    const wchar_t* folder;     // below that: the game's Saved\Config folder
    bool readOnly;             // the game deletes the file unless it is read-only
};
const RrConfigGame* RrConfigGameFor(const wstring& exe);
wstring RrConfigGameExes();    // "Borderlands4.exe and HogwartsLegacy.exe" (a refusal names them)

// The game's Engine.ini (the folder chosen as above, nothing created); empty with *why when its root is not known. *known:
// a sub-folder holds GameUserSettings.ini (else the path is the Windows guess, not to be written).
wstring RrConfigFile(Context& ctx, const RrConfigGame& g, wstring* why, bool* known = nullptr);
// The player's Documents folder (ctx.playerDocuments, else the known folder; test builds: WRONGGPU_TEST_DOCUMENTS only).
wstring PlayerDocumentsDir(Context& ctx);

// Player text: the question after an install ("title\n\nbody"), the do-it-yourself steps with the line, and the line itself.
// protectedFolder: the game is in a folder that needs administrator permission, so Fix it for me brings Windows' prompt: said
// in the offer (r4 #9). beforeFirstStart (with protectedFolder): the game hasn't made its settings folder yet, so the prompt
// comes later, when the player clicks Fix it now on its page after that first start: said so (r5 #9)
wstring RrConfigNotice(const RrConfigGame& g, bool protectedFolder = false, bool beforeFirstStart = false);
// What the line is for, in one sentence: the question and the game page's card lead with it (UI review r2 #8).
wstring RrConfigWhy(const RrConfigGame& g);
wstring RrConfigSteps(const RrConfigGame& g, const wstring& file);
wstring RrConfigLineW();
// What the player's Engine.ini has (the game page's file check, UI review #2): -1 its folder is not known (*file empty) or not
// there yet (no sub-folder holds GameUserSettings.ini: the game makes it at its first start; the sub-folder is a guess until
// then: r2 #7, r2b #1), 0 no line (or no file), 1 the line in [SystemSettings], 2 the line there but the file not read-only in
// a game that deletes it unless it is (Borderlands 4), 3 the file is there but couldn't be read (*why: another program has it
// open, access denied, saved as UTF-16; r3 #8). Read-only: nothing is created or changed.
int RrConfigLineState(Context& ctx, const RrConfigGame& g, wstring* file, wstring* why = nullptr);

// The text edits alone: `in` with the line added (fix says what was done; *changed: the text differs; first: the first
// apply, whose key found in the section is the player's), and the fix's edits taken out again.
string RrConfigAddText(const string& in, RrConfigFix& fix, bool* changed, bool first);
string RrConfigRemoveText(const string& in, const RrConfigFix& fix);
// The bytes of a file the fix creates: "[SystemSettings]", the line, each ending in CRLF.
string RrConfigNewFile();

// on: makes the game's Engine.ini have the line (again: a repair, a second =on) and records it in m.rrconfig; false when
// nothing could be done (*warnings says why) or nothing can be done yet: the game's settings folder isn't there (a note
// "<game>: " + kRrConfigWaitNote, nothing created, m.rrconfig not present: the next Apply adds it, r2b #1). A line added
// before in a guessed sub-folder that turns out not to be the game's (an earlier build) moves to the game's.
// The game folder's backup goes to wronggpu-setup\rr-config\Engine.ini.
bool ApplyRrConfig(Context& ctx, const wstring& dir, Record& m, std::vector<wstring>& notes, std::vector<wstring>& warnings);
extern const wchar_t kRrConfigWaitNote[];  // "the Ray Reconstruction setting will be added after you start the game once"
// rr_config_fix=on and nothing applied yet (m.rrconfig not present): 1 the game's settings folder is there now (an Apply adds
// the line: due), -1 not yet ("Will be added after you start the game once"), 0 otherwise (applied, not on, no root known).
int RrConfigWaiting(Context& ctx, const Record& m, const string& fixSetting);
// off / uninstall: undoes m.rrconfig (nothing when it is not present) and clears it. backupBytes: the backup's content when
// the caller holds it (a repair that failed half way), else it is read from the game's wronggpu-setup folder.
void RevertRrConfig(Context& ctx, const wstring& dir, Record& m, std::vector<wstring>& notes, std::vector<wstring>& warnings,
                    const string* backupBytes = nullptr);
wstring RrConfigBackupPath(const wstring& dir);

}  // namespace wg
