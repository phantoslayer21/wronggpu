// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the ignore lists of the converter's file search (README "Ignore lists"). WrongGPU converts only
// the vendor DLL files that came with the player's own games, never a copy another program's installer placed in a game.
// Such an installer names the files it placed in a list the search always reads (the well-known lists below); the command
// line adds lists and entries of its own (--ignore=, --ignore-list=). A file on a list is never converted: it is skipped
// before the conversion is tried, and named with the reason in the log and the command's output. An empty list (or none)
// leaves the search exactly as it is. Written for this project.
//
// List format (version 1; UTF-8, with or without a BOM, or UTF-16 LE with its BOM; CRLF or LF):
//   wronggpu-ignore-files 1                     optional format version line (first line by convention)
//   # a comment                                 a line whose first non-blank character is '#'; blank lines are fine
//   C:\Games\Some Game\bin\nvngx_dlssd.dll      a full path (drive or UNC; quotes around it are allowed)
//   C:\Games\Other\nvngx_dlssd.dll  sha256=<64 hex digits>   a path and the file's SHA-256 (whitespace before sha256=)
// A path alone matches the candidate that is the same file: the same fully resolved final path (case, '/' or '\', '.', '..',
// symbolic links and junctions resolved), or the same file identity on its volume (a hard link, any other way to it). A path
// that does not exist now is compared as written, normalised. A path with a hash matches only when BOTH match: the same file
// at that path AND that SHA-256 (the entry names a file as it was placed; the same path later holding other bytes, and a copy
// with the same bytes elsewhere, are not skipped). A hash alone is not a list entry: it would skip every copy of a vendor
// DLL with the same bytes, also the player's own, so a line "sha256=<hex>" in a list is skipped with a warning. The command line alone takes one
// (--ignore=sha256=<hex>): a deliberate choice of the player, which matches the candidate with that SHA-256 wherever it is.
// A line that is not understood (and a path that is a folder) is skipped with a warning and the rest of the list is used. A
// newer format version is read with these rules (its lines that are not understood are skipped with a warning). "\\?\" and
// "\\?\UNC\" prefixes are read as the plain path.
// A list that is there but can't be read, or can't be decoded (UTF-16 big-endian, UTF-16 without a BOM, bytes that are not
// UTF-8), fails closed: nothing is converted or reused (after kIgnoreReadTries on a sharing violation). So does a candidate
// whose SHA-256 can't be computed while a hash entry applies to it (it is left out). A conversion on this PC is checked
// against the lists (the source path text and SHA-256 its model-source.txt records, as recorded: the path is not opened
// again) before it is reused (model.cpp FindModel, rr.cpp FindRrData). The uninstalls and remove-data keep the list
// (PutIgnoreListBack): a player's list is never deleted.
#pragma once
#include "context.h"

namespace wg {

const wchar_t* const kIgnoreListName = L"ignore-files.txt";
const wchar_t* const kIgnoreFormatTag = L"wronggpu-ignore-files";
const int kIgnoreFormatVersion = 1;
// A list that is there but can't be read fails closed (nothing is converted or reused); a sharing violation is waited for.
const int kIgnoreReadTries = 5, kIgnoreReadWaitMs = 200;
// What the uninstalls and remove-data say when they keep the list (PutIgnoreListBack).
const wchar_t* const kKeptIgnoreListText = L"kept ignore-files.txt (your/integrators' ignore list)";

// A file's identity on its volume (FILE_ID_INFO: the volume serial number and the 128-bit file id).
struct FileIdent {
    unsigned long long volume = 0;
    unsigned char id[16] = {};
    bool valid = false;
    bool operator==(const FileIdent& o) const;
};

struct IgnoreEntry {
    wstring text;      // the entry as written (for the messages)
    wstring path;      // the full path as written, normalised ("" for a hash-only entry: --ignore=sha256= only)
    wstring resolved;  // the file's fully resolved final path when it exists now, else `path`
    FileIdent ident;   // the file's identity when it exists now
    wstring sha256;    // lower-case hex, "" when none; with a path, it must match too
    wstring where;     // "<list file>, line <n>", or "--ignore"
};
struct IgnoreLists {
    std::vector<IgnoreEntry> entries;
    std::vector<wstring> warnings;  // lines skipped (each with its list, line and why)
    std::vector<wstring> lists;     // the list files read
    wstring unreadable;             // a list that is there but could not be read: the player's message (the search fails closed)
    bool empty() const { return entries.empty(); }
};
// What a search left out: "path: reason" per candidate skipped, and the warnings of the lists it read. `unreadable`: a list
// could not be read (nothing is converted or reused; the command line's exit code is 10).
struct IgnoreReport {
    std::vector<wstring> skipped, warnings;
    wstring unreadable;
};

// One list line (trimmed by the caller or not). True: an entry in `e`, or nothing for a blank / comment / version line
// (`e.text` empty). False: a line that is not understood (`why`). allowHashAlone: "sha256=<hex>" without a path is an entry
// (--ignore= only; in a list it is not understood).
bool ParseIgnoreLine(const wstring& line, IgnoreEntry& e, wstring& why, int* version = nullptr, bool allowHashAlone = false);
// A whole list's bytes (`source`: its name in the messages). Bad lines go to into.warnings and are skipped. False: the bytes
// can't be decoded (UTF-16 big-endian or without a BOM, not UTF-8; `why`); nothing is added then.
bool ParseIgnoreText(const string& bytes, const wstring& source, IgnoreLists& into, wstring* why = nullptr);
// One list file's bytes, the one bounded reader of every list (a sharing violation is tried kIgnoreReadTries times,
// kIgnoreReadWaitMs apart; a file of kIgnoreListMaxBytes or more is refused before anything is read). False: `why`.
const long long kIgnoreListMaxBytes = 16ll << 20;
bool ReadIgnoreListBytes(const wstring& file, string& out, wstring& why);
// Reads (ReadIgnoreListBytes) and parses one list file. False: it could not be read or decoded (`why`); nothing is added then.
bool ReadIgnoreList(const wstring& file, IgnoreLists& into, wstring* why = nullptr);
// The well-known lists, in the order they are read: %LOCALAPPDATA%\WrongGPU\ignore-files.txt, then ignore-files.txt in the
// launcher's own data folder when that is another folder (a portable launcher's WrongGPU-data, a --data= folder).
std::vector<wstring> WellKnownIgnoreLists(const wstring& ownDataDir);
// The fully resolved final path of an existing file or folder (links followed; no \\?\ prefix); "" when it can't be opened.
wstring FinalPath(const wstring& p, FileIdent* ident = nullptr);
// Why `candidate` is on a list ("" : it is not): "on the ignore list <file>, line <n> (path ...)" or "named by --ignore=<entry>
// (SHA-256 ...)"; a candidate whose SHA-256 can't be computed while a hash entry applies to it gets "unreadable, ..." (left
// out). An empty list never opens or reads the file; a file is hashed only when an entry that needs its hash applies.
wstring IgnoreReason(const IgnoreLists& l, const wstring& candidate);
// The same for a conversion on this PC: the source path and SHA-256 its model-source.txt recorded, compared as recorded.
// The recorded path is never opened again: what is at that path now (the file replaced, or the path made a hard link or a
// symbolic link to another file) says nothing about the file that was converted. The recorded path's text, normalised
// (case, '/' or '\', '.', '..'), is compared with the entry's path as written and as it resolves now. A path entry needs the
// recorded path; a path + hash entry the recorded path and the recorded SHA-256 (an old record without a SHA-256: the path
// text alone, its bytes can't be checked); a hash entry (--ignore=sha256=) the recorded SHA-256. "" : not on a list (or
// nothing recorded to compare).
wstring RecordedSourceReason(const IgnoreLists& l, const wstring& source, const wstring& sha);
// The player's line when files were left out of a search (rep.skipped not empty): the first file with its list and line
// (or --ignore=), the number of the others and where the log is. "" when nothing was left out.
wstring IgnoredExample(const Context& ctx, const IgnoreReport& rep);
// A "key<TAB>value" line of a model-source.txt ("" when the file or the key is not there).
wstring SourceField(const wstring& file, const wchar_t* key);
// The lists of one search: the well-known lists, then this run's (ctx.ignoreListFiles, read from ctx.ignoreListBytes when
// the command line read them already, and ctx.ignoreEntries). Their warnings
// are logged (`what`: the log prefix, "model" or "RR data") and added to rep->warnings; a list that is there but can't be
// read sets l.unreadable and rep->unreadable (the caller converts and reuses nothing then).
IgnoreLists LoadConvertIgnore(Context& ctx, const wchar_t* what, IgnoreReport* rep);
// For each path: true when it is skipped (logged, and "path: reason" added to rep.skipped). An empty list skips nothing.
std::vector<bool> IgnoredMask(Context& ctx, const IgnoreLists& l, const std::vector<wstring>& paths, const wchar_t* what, IgnoreReport& rep);
// What PutIgnoreListBack did with the list of the folder moved aside.
struct ListKept {
    wstring said;    // the player's line, also logged: kKeptIgnoreListText + ": <path>" when it is back in place; another line when
                     // it was kept beside another program's new list (".kept-<n>") or left in the moved folder; "" : no list
    wstring leftIn;  // not "": the list could not leave the moved folder and is kept there, at this path. The caller deletes
                     // the rest of the folder but never this file, and never the folder's whole tree (no recursive delete).
    bool inPlace = false;  // back at <shared>\ignore-files.txt as it was
};
// The shared folder (%LOCALAPPDATA%\WrongGPU) was moved aside to `moved` to be deleted: its ignore-files.txt goes back into
// `shared` (made again for it), as the uninstalls and remove-data keep it. A player's list is never deleted: when another
// program made <shared>\ignore-files.txt again meanwhile, the list is kept beside it as ignore-files.txt.kept-<n> (the first
// free n); when it can't be moved or copied out at all, it stays in the moved folder (ListKept::leftIn). A list that is a
// link (a reparse point) is moved as the link (a copy copies the link, never what it points to). Each case is said in
// ListKept::said for the player's result.
ListKept PutIgnoreListBack(Context& ctx, const wstring& moved, const wstring& shared);
// Test seam: called by PutIgnoreListBack after the folder was moved aside, before the list is put back (`shared`: where it
// goes). The tests make the folder again with another list in it, or put a file where it should be. nullptr: none.
// Test builds (WRONGGPU_TESTING) also read WRONGGPU_TEST_LIST_PUTBACK ("exists" or "blocked") when no seam is set, for the
// command-line suites (tests/run_convert_ignore_tests.py).
using PutBackHookFn = void (*)(const wstring& shared);
void SetPutBackHookForTest(PutBackHookFn fn);

}  // namespace wg
