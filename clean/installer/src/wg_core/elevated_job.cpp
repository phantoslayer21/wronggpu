// SPDX-License-Identifier: MIT
// The elevated worker's side of "WrongGPU.exe apply" (elevated_job.h): the job and batch files are bound to the
// launcher's request by SHA-256, results go to paths the worker works out itself, and the log stays in the player's
// folders. Written for this project.
#include "elevated_job.h"

#include <atomic>
#include <memory>

namespace wg {

namespace {

struct HandleCloser { void operator()(HANDLE h) const { if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h); } };
using Held = std::unique_ptr<void, HandleCloser>;
Held Hold(HANDLE h) { return Held(h == INVALID_HANDLE_VALUE ? nullptr : h); }

// GetFinalPathNameByHandle's path without its \\?\ prefix (\\?\UNC\server\share -> \\server\share).
wstring FinalPath(HANDLE h) {
    wstring buf(1024, L'\0');
    DWORD n = GetFinalPathNameByHandleW(h, buf.data(), (DWORD)buf.size(), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (n >= buf.size()) { buf.resize(n + 1); n = GetFinalPathNameByHandleW(h, buf.data(), (DWORD)buf.size(), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS); }
    if (!n || n >= buf.size()) return L"";
    buf.resize(n);
    if (buf.rfind(L"\\\\?\\UNC\\", 0) == 0) return L"\\\\" + buf.substr(8);
    if (buf.rfind(L"\\\\?\\", 0) == 0) return buf.substr(4);
    return buf;
}

// The long form of a path (8.3 short names such as RUNNER~1 expanded), as far as it exists; the rest kept as written.
wstring LongOf(const wstring& p) {
    wstring buf(1024, L'\0');
    DWORD n = GetLongPathNameW(p.c_str(), buf.data(), (DWORD)buf.size());
    if (n >= buf.size()) { buf.resize(n + 1); n = GetLongPathNameW(p.c_str(), buf.data(), (DWORD)buf.size()); }
    if (n && n < buf.size()) { buf.resize(n); return buf; }
    const wstring parent = Parent(p);
    return parent.empty() || parent == p ? p : Join(LongOf(parent), BaseName(p));
}

bool Under(const wstring& path, const wstring& root) {
    return !root.empty() && (IEq(path, root) || StartsWithI(path, root.back() == L'\\' ? root : root + L'\\'));
}

// A file or folder opened as itself (FILE_FLAG_OPEN_REPARSE_POINT): refused when it is a reparse point, a file with more
// than one name (hard links) or the wrong kind. No delete sharing: it cannot be renamed, deleted or replaced while held.
bool OpenPlain(const wstring& path, bool folder, DWORD access, DWORD disposition, Held& out, wstring& finalPath, wstring& why) {
    const DWORD flags = FILE_FLAG_OPEN_REPARSE_POINT | (folder ? FILE_FLAG_BACKUP_SEMANTICS : FILE_ATTRIBUTE_NORMAL);
    Held h = Hold(CreateFileW(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, disposition, flags, nullptr));
    if (!h) { why = L"cannot open " + path + L" (" + ErrText(GetLastError()) + L")"; return false; }
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(h.get(), &info)) { why = L"cannot read the attributes of " + path; return false; }
    if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) { why = path + L" is a link or junction (reparse point)"; return false; }
    if (!!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != folder) { why = path + (folder ? L" is not a folder" : L" is a folder"); return false; }
    if (!folder && info.nNumberOfLinks > 1) { why = path + L" has more than one name (hard link)"; return false; }
    finalPath = FinalPath(h.get());
    if (finalPath.empty()) { why = L"cannot resolve the path of " + path; return false; }
    out = std::move(h);
    return true;
}

// Reads a file once, as itself, and returns its bytes and the real folder it is in.
bool ReadOnce(const wstring& path, string& bytes, wstring& realFolder, wstring& why) {
    Held h;
    wstring fin;
    if (!OpenPlain(path, false, GENERIC_READ, OPEN_EXISTING, h, fin, why)) return false;
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h.get(), &size) || size.QuadPart > (1 << 20)) { why = path + L" is not a job file (size)"; return false; }
    bytes.resize((size_t)size.QuadPart);
    size_t done = 0;
    while (done < bytes.size()) {
        DWORD got = 0;
        if (!ReadFile(h.get(), bytes.data() + done, (DWORD)(bytes.size() - done), &got, nullptr) || !got) { why = L"cannot read " + path; return false; }
        done += got;
    }
    realFolder = Parent(fin);
    return true;
}

bool SameHash(const wstring& a, const wstring& b) { return a.size() == 64 && IEq(a, b); }

// The folder results are written to: the folder named on the command line, opened as itself (not a reparse point), must
// be the real folder the file was read from; it is held open while the worker runs.
bool HoldFolder(const wstring& named, const wstring& folder, Held& held, wstring& why) {
    wstring fin;
    if (!OpenPlain(named, true, FILE_LIST_DIRECTORY | FILE_TRAVERSE, OPEN_EXISTING, held, fin, why)) return false;
    if (!IEq(fin, folder)) { why = L"the folder " + named + L" resolves to " + fin + L", not " + folder; held.reset(); return false; }
    return true;
}

// A result file in the held folder: a new temporary file there, then moved over the result name (which, when it is there
// already, must be a plain file with one name). Never written through an existing name.
bool WriteResult(const wstring& path, const string& text, wstring& why) {
    if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES) {
        Held old;
        wstring fin;
        if (!OpenPlain(path, false, FILE_READ_ATTRIBUTES, OPEN_EXISTING, old, fin, why)) return false;
    }
    static std::atomic<unsigned> seq{0};
    const wstring tmp = path + L".tmp-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(seq++);
    Held h = Hold(CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!h) { why = L"cannot write " + tmp + L" (" + ErrText(GetLastError()) + L")"; return false; }
    DWORD wrote = 0;
    const bool ok = WriteFile(h.get(), text.data(), (DWORD)text.size(), &wrote, nullptr) && wrote == text.size();
    FlushFileBuffers(h.get());
    h.reset();
    if (!ok || !MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        why = L"cannot write " + path + L" (" + ErrText(GetLastError()) + L")";
        DeleteFileW(tmp.c_str());
        return false;
    }
    return true;
}

// log=: a file under one of the roots (each resolved to its real path), its folder and the file opened as themselves and
// held while the job runs. An empty log= stays empty (no log); so does one whose folder is not there (noLog).
bool HoldLog(const wstring& log, const std::vector<wstring>& roots, Held& folderHeld, Held& fileHeld, bool& noLog, wstring& why) {
    noLog = false;
    if (log.empty()) return true;
    const wstring full = LongOf(FullPath(log)), folder = Parent(full);
    std::vector<wstring> real;
    for (auto& r : roots) {
        if (r.empty()) continue;
        Held h = Hold(CreateFileW(r.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                  FILE_FLAG_BACKUP_SEMANTICS, nullptr));
        if (h) real.push_back(FinalPath(h.get()));
    }
    wstring folderFinal, fileFinal;
    bool inside = false;
    for (auto& r : roots) inside = inside || (!r.empty() && Under(full, LongOf(FullPath(r))));
    if (!inside) { why = L"the log file " + log + L" is not in the player's data folders"; return false; }
    if (GetFileAttributesW(folder.c_str()) == INVALID_FILE_ATTRIBUTES) {
        const DWORD e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) { noLog = true; return true; }
    }
    if (!OpenPlain(folder, true, FILE_LIST_DIRECTORY | FILE_TRAVERSE, OPEN_EXISTING, folderHeld, folderFinal, why)) return false;
    inside = false;
    for (auto& r : real) inside = inside || Under(folderFinal, r);
    if (!inside) { why = L"the log folder " + folder + L" resolves to " + folderFinal + L", outside the player's data folders"; return false; }
    if (!OpenPlain(full, false, FILE_APPEND_DATA, OPEN_ALWAYS, fileHeld, fileFinal, why)) return false;
    if (!IEq(Parent(fileFinal), folderFinal)) { why = L"the log file " + log + L" resolves to " + fileFinal; return false; }
    return true;
}

bool NameAfter(const wstring& path, const wchar_t* prefix, wstring& rest) {
    const wstring name = BaseName(path), pre = prefix;
    if (name.size() <= pre.size() + 4 || !StartsWithI(name, pre) || !EndsWithI(name, L".txt")) return false;
    rest = name.substr(pre.size(), name.size() - pre.size() - 4);
    return true;
}

}  // namespace

wstring BatchResultPath(const wstring& batch, size_t n) {
    wstring stamp;
    return NameAfter(batch, L"batch-", stamp) ? Join(Parent(batch), L"result-" + stamp + L"-" + std::to_wstring(n) + L".txt") : L"";
}
wstring JobResultPath(const wstring& job) {
    wstring name;
    return NameAfter(job, L"job-", name) ? Join(Parent(job), L"result-" + name + L".txt") : L"";
}

WorkerRun RunWorkerFiles(const wstring& batch, const wstring& job, const wstring& sha256, const WorkerRunFn& run) {
    WorkerRun out;
    auto refuse = [&out](const wstring& why) {
        out.refused = true;
        out.last.code = EXIT_USAGE;
        out.last.message = L"The elevated step did not run: " + why + L".";
        return out;
    };
    const bool isBatch = !batch.empty();
    const wstring given = isBatch ? batch : job;
    if (given.empty() || sha256.empty()) return refuse(L"apply needs --batch=<file> or --job=<file>, and --sha256=<its SHA-256>");
    string text;
    wstring folder, why;
    if (!ReadOnce(given, text, folder, why)) return refuse(why);
    if (!SameHash(Sha256Bytes(text.data(), text.size()), sha256))
        return refuse(BaseName(given) + L" is not the file the launcher wrote for this request (SHA-256 mismatch)");
    Held folderHeld;
    if (!HoldFolder(Parent(FullPath(given)), folder, folderHeld, why)) return refuse(why);
    const wstring real = Join(folder, BaseName(given));

    // (job file, its SHA-256 from the batch; empty for the single job: its text is already checked) and its result path
    struct Item { wstring file, sha, result; };
    std::vector<Item> items;
    bool stop = false;
    if (isBatch) {
        std::vector<std::pair<wstring, wstring>> jobs;
        if (!ParseBatch(text, jobs, stop, &why)) return refuse(L"bad batch file: " + why);
        for (size_t i = 0; i < jobs.size(); ++i) items.push_back({jobs[i].first, jobs[i].second, BatchResultPath(real, i + 1)});
    } else {
        items.push_back({real, L"", JobResultPath(real)});
    }
    if (items.empty() || items[0].result.empty()) return refuse(L"the file name is not batch-<stamp>.txt or job-<name>.txt");
    // <data>\logs: next to the jobs folder named on the command line (its real path is checked when the log is opened)
    const std::vector<wstring> logRoots0 = {Join(Parent(Parent(FullPath(given))), L"logs")};

    bool failed = false;
    for (auto& it : items) {
        OpResult res;
        if (stop && failed) {
            res.code = kOpNotRun;
            res.message = L"not run: an earlier game failed";
        } else {
            string jt = isBatch ? string() : text;
            wstring jf, jwhy;
            WorkerJob w;
            Held logFolder, logFile;
            if (isBatch && !ReadOnce(it.file, jt, jf, jwhy)) { res.code = EXIT_USAGE; res.message = L"Not run: " + jwhy + L"."; }
            else if (isBatch && !SameHash(Sha256Bytes(jt.data(), jt.size()), it.sha)) {
                res.code = EXIT_USAGE;
                res.message = L"Not run: " + BaseName(it.file) + L" is not the job file the launcher wrote for this request (SHA-256 mismatch).";
            } else if (!ParseJob(jt, w.r, w.localAppData, w.logFile, &jwhy, &w.userSid, &w.perGameEntry, &w.documents)) {
                res.code = EXIT_USAGE; res.message = L"Not run: bad job file: " + jwhy + L".";
            } else {
                std::vector<wstring> roots = logRoots0;
                roots.push_back(w.localAppData);
                bool noLog = false;
                if (!HoldLog(w.logFile, roots, logFolder, logFile, noLog, jwhy)) { res.code = EXIT_USAGE; res.message = L"Not run: " + jwhy + L"."; }
                else {
                    if (noLog) w.logFile.clear();
                    res = run(w);
                }
            }
            ++out.ran;  // taken up (as before: a job refused here counts too; only the ones stop=1 skips do not)
            if (res.code) { failed = true; out.last = res; }
        }
        wstring wwhy;
        WriteResult(it.result, ResultText(res), wwhy);
        if (!isBatch) out.last = res;
    }
    out.jobs = (int)items.size();
    return out;
}

}  // namespace wg
