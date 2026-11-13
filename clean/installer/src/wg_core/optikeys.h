// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the OptiScaler tab of a game's page (build/launcher-design.md sections 3 and 4, milestone
// M2): which OptiScaler.ini keys WrongGPU manages (locked: shown, never edited by the player through the launcher),
// which the player may edit (only keys that already exist in that OptiScaler build's ini), byte-exact edits of single
// values, and the "inikey" journal of the install record: every key the launcher changed in the installed
// OptiScaler.ini, with its value before, so an edit can be reverted byte for byte, an update carries the player's
// choices over to the new ini, and a value OptiScaler's own menu saved over ours is shown as an outside edit.
// Written for this project.
#pragma once
#include "util.h"

namespace wg {

// ---- single values, byte for byte ---------------------------------------------------------------------------------------
// The first line of `key` in `section` (both compared ignoring case, as OptiScaler reads its ini). The value is the text
// after '=' without the spaces around it (no inline comments: OptiScaler's ini has none).
struct IniSpan { bool found = false; size_t lineBegin = 0, lineEnd = 0, valueBegin = 0, valueEnd = 0; };
IniSpan IniFind(const string& text, const string& section, const string& key);
string IniValueAt(const string& text, const IniSpan& s);
string IniReplaceValue(const string& text, const IniSpan& s, const string& value);  // only the value's bytes change
struct IniEntry { string section, key, value; };
std::vector<IniEntry> IniEntries(const string& text);  // every key line (first occurrence of each section/key), in order

// ---- the keys of the OptiScaler tab ------------------------------------------------------------------------------------
enum class OptiKind { Bool, Int, Number, Key };
struct OptiKeyDef { const char* section; const char* key; OptiKind kind; const wchar_t* label; const wchar_t* help; };
extern const OptiKeyDef kOptiEditable[];
extern const size_t kOptiEditableCount;
struct OptiLockedKey { const char* section; const char* key; const wchar_t* why; };
extern const OptiLockedKey kOptiLocked[];
extern const size_t kOptiLockedCount;
const OptiKeyDef* FindOptiKey(const string& section, const string& key);
bool IsOptiLocked(const string& section, const string& key, const wchar_t** why = nullptr);
// true/false/auto, a number or auto, a key code (decimal or 0x hex) or auto; normalised (lower case booleans).
bool ValidateOptiValue(const OptiKeyDef& d, const string& value, string* normal, wstring* why);
// "Section.Key" -> section and key.
bool SplitOptiName(const string& name, string& section, string& key);

struct OptiRow { string section, key, value; bool locked = false; const OptiKeyDef* def = nullptr; const wchar_t* why = nullptr; };
// The tab: the locked keys this ini has, then the editable keys it has (keys it lacks are not offered).
std::vector<OptiRow> OptiTabRows(const string& ini);
// Edits ("Section.Key", value) of editable keys that exist; false with *why (nothing changed then).
bool ApplyOptiEdits(const string& in, const std::vector<std::pair<string, string>>& edits, string& out, wstring* why);

// ---- the inikey journal (record lines "inikey <file> <section> <key> <original|absent> <ours>") ------------------------
struct IniKeyChange { wstring file; string section, key; bool absent = false; string original, ours; };
// Adds the keys whose value differs between `before` and `after` (keys in both) to the journal of `file`: a new key
// keeps before's value as its original; a key that is back at its original leaves the journal.
void JournalChanges(std::vector<IniKeyChange>& journal, const wstring& file, const string& before, const string& after);
// The originals put back (all keys of `file`); notes name keys whose value was not ours any more.
string RevertJournal(const string& text, const std::vector<IniKeyChange>& journal, const wstring& file, std::vector<wstring>* notes);
// Keys of the journal whose value in the file is not ours any more (OptiScaler's menu saved its ini, or a hand edit).
struct OptiOutside { string section, key, fileValue, ours; };
std::vector<OptiOutside> JournalOutside(const string& text, const std::vector<IniKeyChange>& journal, const wstring& file);
// Update / repair: the journal's values (not the locked keys) written into a freshly installed ini where those keys
// exist; the new journal is relative to the fresh file.
string ReplayJournal(const string& fresh, const std::vector<IniKeyChange>& old, const wstring& file, std::vector<IniKeyChange>& journal,
                     std::vector<wstring>* notes);

}  // namespace wg
