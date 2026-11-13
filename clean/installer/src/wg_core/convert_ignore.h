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
//   sha256=<64 hex digits>                      a hash alone: the file with this SHA-256 is skipped wherever it is
// A path matches the candidate that is the same file: the same fully resolved final path (case, '/' or '\', '.', '..',
// symbolic links and junctions resolved), or the same file identity on its volume (a hard link, any other way to it).
// A path that does not exist now is compared as written, normalised. A hash matches the candidate whose SHA-256 it is,
// wherever that file is; a line with both matches by either. A line that is not understood is skipped with a warning and
// the rest of the list is used. A newer format version is read with these rules (its lines that are not understood are
// skipped with a warning). "\\?\" and "\\?\UNC\" prefixes are read as the plain path.
// A list that is there but can't be read fails closed: nothing is converted or reused (after kIgnoreReadTries on a sharing
// violation). A conversion on this PC is checked against the lists (the source path and SHA-256 its model-source.txt
// records) before it is reused (model.cpp FindModel, rr.cpp FindRrData). The uninstalls and remove-data keep the list
// (PutIgnoreListBack).
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
    wstring path;      // the full path as written, normalised ("" for a hash-only entry)
    wstring resolved;  // the file's fully resolved final path when it exists now, else `path`
    FileIdent ident;   // the file's identity when it exists now
    wstring sha256;    // lower-case hex, "" when none
    wstring where;     // "<list file>, line <n>", or "--ignore"
};
struct IgnoreLists {
    std::vector<IgnoreEntry> entries;
    std::vector<wstring> warnings;  // lines skipped (each with its list, line and why)
    std::vector<wstring> lists;     // the list files read
    wstring unreadable;             // a list that is there but could not be read: the player's message (the search fails closed)
    bool empty() const { return entries.empty(); }
    bool hasHashes() const;
};
// What a search left out: "path: reason" per candidate skipped, and the warnings of the lists it read. `unreadable`: a list
// could not be read (nothing is converted or reused; the command line's exit code is 10).
struct IgnoreReport {
    std::vector<wstring> skipped, warnings;
    wstring unreadable;
};

// One list line (trimmed by the caller or not). True: an entry in `e`, or nothing for a blank / comment / version line
// (`e.text` empty). False: a line that is not understood (`why`).
bool ParseIgnoreLine(const wstring& line, IgnoreEntry& e, wstring& why, int* version = nullptr);
// A whole list's bytes (`source`: its name in the messages). Bad lines go to into.warnings and are skipped.
void ParseIgnoreText(const string& bytes, const wstring& source, IgnoreLists& into);
// Reads and parses one list file (a sharing violation is tried kIgnoreReadTries times, kIgnoreReadWaitMs apart). False: it
// could not be read (`why`); nothing is added then.
bool ReadIgnoreList(const wstring& file, IgnoreLists& into, wstring* why = nullptr);
// The well-known lists, in the order they are read: %LOCALAPPDATA%\WrongGPU\ignore-files.txt, then ignore-files.txt in the
// launcher's own data folder when that is another folder (a portable launcher's WrongGPU-data, a --data= folder).
std::vector<wstring> WellKnownIgnoreLists(const wstring& ownDataDir);
// The fully resolved final path of an existing file or folder (links followed; no \\?\ prefix); "" when it can't be opened.
wstring FinalPath(const wstring& p, FileIdent* ident = nullptr);
// Why `candidate` is on a list ("" : it is not): "on the ignore list <file>, line <n> (path ...)" or "named by --ignore=<entry>
// (SHA-256 ...)". An empty list never opens or reads the file.
wstring IgnoreReason(const IgnoreLists& l, const wstring& candidate);
// The same for a conversion on this PC: the source path and SHA-256 its model-source.txt recorded (the file is not read
// again; it may have changed or gone since). "" : not on a list (or nothing recorded to compare).
wstring RecordedSourceReason(const IgnoreLists& l, const wstring& source, const wstring& sha);
// A "key<TAB>value" line of a model-source.txt ("" when the file or the key is not there).
wstring SourceField(const wstring& file, const wchar_t* key);
// The lists of one search: the well-known lists, then this run's (ctx.ignoreListFiles, ctx.ignoreEntries). Their warnings
// are logged (`what`: the log prefix, "model" or "RR data") and added to rep->warnings; a list that is there but can't be
// read sets l.unreadable and rep->unreadable (the caller converts and reuses nothing then).
IgnoreLists LoadConvertIgnore(Context& ctx, const wchar_t* what, IgnoreReport* rep);
// For each path: true when it is skipped (logged, and "path: reason" added to rep.skipped). An empty list skips nothing.
std::vector<bool> IgnoredMask(Context& ctx, const IgnoreLists& l, const std::vector<wstring>& paths, const wchar_t* what, IgnoreReport& rep);
// The shared folder (%LOCALAPPDATA%\WrongGPU) was moved aside to `moved` to be deleted: its ignore-files.txt goes back into
// `shared` (made again for it), as the uninstalls and remove-data keep it. Returns kKeptIgnoreListText and the path, or ""
// when there was no list.
wstring PutIgnoreListBack(const wstring& moved, const wstring& shared);

}  // namespace wg
