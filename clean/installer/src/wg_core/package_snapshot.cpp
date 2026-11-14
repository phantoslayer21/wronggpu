// SPDX-License-Identifier: MIT
// The OptiScaler package's digest and the elevated worker's copy of it (package_snapshot.h). Written for this project.
#include "package_snapshot.h"

#include <sddl.h>
#include <bcrypt.h>

#include <algorithm>
#include <memory>

#include "checks.h"    // IsElevated
#include "launcher.h"  // CurrentUserSid
#include "secure_fs.h"
#include "sha256.h"

#pragma comment(lib, "bcrypt.lib")

namespace wg {

namespace {

const size_t kMaxFiles = 20000;
const unsigned long long kMaxBytes = 1ull << 30;  // 1 GiB: far more than an OptiScaler release

struct Source {
    bool archive = false;
    wstring root;   // the folder (folder kinds), or the archive's folder
    wstring file;   // the archive (archive kind)
};

bool SourceOf(const wstring& chosen, Source& s, wstring& why) {
    const wstring full = FullPath(chosen);
    if (IsDir(full)) { s.root = full; return true; }
    if (IsFile(full) && IEq(BaseName(full), L"OptiScaler.dll")) { s.root = Parent(full); return true; }
    if (IsFile(full) && (EndsWithI(full, L".7z") || EndsWithI(full, L".zip"))) { s.archive = true; s.file = full; s.root = Parent(full); return true; }
    why = L"the OptiScaler package " + chosen + L" is not an archive (.7z or .zip), a folder or OptiScaler.dll";
    return false;
}

// The files of a folder and its subfolders, relative to it (links left out: util.h ListEntries, the same on the launcher's
// side and through the elevated worker's guard), sorted.
bool ListPackage(const wstring& root, std::vector<wstring>& files, wstring& why) {
    std::vector<wstring> stack{L""};
    while (!stack.empty()) {
        const wstring rel = stack.back();
        stack.pop_back();
        std::vector<DirEntry> entries;
        const wstring dir = rel.empty() ? root : Join(root, rel);
        if (!ListEntries(dir, entries) && GetLastError() != ERROR_FILE_NOT_FOUND) {
            why = L"cannot list " + dir + L" (" + ErrText(GetLastError()) + L")";
            return false;
        }
        for (auto& en : entries) {
            if (en.attrs & FILE_ATTRIBUTE_REPARSE_POINT) continue;
            const wstring r = rel.empty() ? en.name : Join(rel, en.name);
            if (en.attrs & FILE_ATTRIBUTE_DIRECTORY) stack.push_back(r);
            else files.push_back(r);
            if (files.size() > kMaxFiles) { why = L"the OptiScaler package has more files than a package can have"; return false; }
        }
    }
    std::sort(files.begin(), files.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return true;
}

struct HandleCloser { void operator()(HANDLE h) const { if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h); } };
using Held = std::unique_ptr<void, HandleCloser>;

// One file read once (no other writer while it is read): its SHA-256, and its bytes written to `copyTo` when given.
bool HashFile(const wstring& path, const wstring& copyTo, unsigned long long& total, wstring& sha, wstring& why) {
    Held in(OpenForRead(path, FILE_SHARE_READ));
    if (in.get() == INVALID_HANDLE_VALUE) { in.release(); why = L"cannot read " + path + L" (" + ErrText(GetLastError()) + L")"; return false; }
    Held out;
    if (!copyTo.empty()) {
        out.reset(CreateNewFile(copyTo));
        if (out.get() == INVALID_HANDLE_VALUE) { out.release(); why = L"cannot write " + copyTo + L" (" + ErrText(GetLastError()) + L")"; return false; }
    }
    wgc::Sha256 h;
    std::vector<char> buf(1 << 20);
    for (;;) {
        DWORD got = 0;
        if (!ReadFile(in.get(), buf.data(), (DWORD)buf.size(), &got, nullptr)) { why = L"cannot read " + path + L" (" + ErrText(GetLastError()) + L")"; return false; }
        if (!got) break;
        total += got;
        if (total > kMaxBytes) { why = L"the OptiScaler package is larger than a package can be"; return false; }
        h.update(buf.data(), got);
        DWORD wrote = 0;
        if (out && (!WriteFile(out.get(), buf.data(), got, &wrote, nullptr) || wrote != got)) { why = L"cannot write " + copyTo; return false; }
    }
    sha = HexW(h.finish());
    return true;
}

// The manifest of the package (and its copy in `copyDir` when given) and the manifest's SHA-256.
bool Digest(const Source& s, const wstring& copyDir, wstring& digest, wstring& copy, wstring& why) {
    std::vector<wstring> files;
    wstring base = s.root;
    if (s.archive) files.push_back(BaseName(s.file));
    else if (!ListPackage(s.root, files, why)) return false;
    if (files.empty()) { why = L"the OptiScaler package " + s.root + L" has no files"; return false; }
    string manifest = string("; WrongGPU OptiScaler package\r\nkind=") + (s.archive ? "archive" : "folder") + "\r\n";
    unsigned long long total = 0;
    const wstring into = copyDir.empty() ? L"" : (s.archive ? copyDir : Join(copyDir, L"package"));
    if (!into.empty() && !s.archive) EnsureDir(into);
    for (auto& rel : files) {
        wstring target;
        if (!into.empty()) {
            target = Join(into, rel);
            if (Parent(target) != into) EnsureDir(Parent(target));
        }
        wstring sha;
        if (!HashFile(Join(base, rel), target, total, sha, why)) return false;
        manifest += "file=" + ToUtf8(rel) + "\r\nsha256=" + ToUtf8(sha) + "\r\n";
    }
    digest = Sha256Bytes(manifest.data(), manifest.size());
    copy = into.empty() ? L"" : (s.archive ? Join(into, files[0]) : into);
    return true;
}

}  // namespace

wstring RandomHex() {
    unsigned char b[16] = {};
    if (BCryptGenRandom(nullptr, b, sizeof(b), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) Fail(L"no random numbers from Windows", EXIT_FAILED);
    static const wchar_t* hex = L"0123456789abcdef";
    wstring s;
    for (unsigned char c : b) { s.push_back(hex[c >> 4]); s.push_back(hex[c & 15]); }
    return s;
}

bool PackageDigest(const wstring& chosen, wstring& digest, wstring& why) {
    Source s;
    wstring copy;
    return SourceOf(chosen, s, why) && Digest(s, L"", digest, copy, why);
}

bool SnapshotPackage(const wstring& chosen, const wstring& expected, const wstring& intoDir, wstring& copy, wstring& why) {
    Source s;
    wstring digest;
    if (expected.size() != 64) { why = L"the job has no digest for the OptiScaler package " + chosen; return false; }
    try {
        if (!SourceOf(chosen, s, why) || !Digest(s, intoDir, digest, copy, why)) return false;
    } catch (const SetupError& e) { why = e.msg; return false; }
    if (!IEq(digest, expected)) { why = L"the OptiScaler package " + chosen + L" is not the one chosen for this request (its files changed)"; return false; }
    return true;
}

bool MakePrivateFolder(const wstring& parentIn, wstring& path, wstring& why) {
    wstring parent = parentIn;
    if (parent.empty()) {
        wchar_t win[MAX_PATH] = {};
        const UINT n = GetWindowsDirectoryW(win, MAX_PATH);
        if (!n || n >= MAX_PATH) { why = L"cannot find the Windows folder"; return false; }
        parent = Join(win, L"Temp");
    }
    // administrators and SYSTEM only, nothing inherited from the parent; this account too when not elevated (tests)
    wstring sddl = L"D:P(A;OICI;FA;;;BA)(A;OICI;FA;;;SY)";
    if (!IsElevated()) {
        const wstring me = CurrentUserSid();
        if (me.empty()) { why = L"cannot read this account's SID"; return false; }
        sddl += L"(A;OICI;FA;;;" + me + L")";
    }
    PSECURITY_DESCRIPTOR sd = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &sd, nullptr)) {
        why = L"cannot make the folder's permissions (" + ErrText(GetLastError()) + L")";
        return false;
    }
    // made relative to the parent's held handle, the parent opened without following a link (secure_fs.h)
    bool ok = false;
    DWORD e = 0;
    FsGuard guard;
    wstring refused;
    if (!guard.AddRoot(parent, refused)) {
        LocalFree(sd);
        why = L"cannot make a private folder in " + parent + L": " + refused;
        return false;
    }
    for (int i = 0; i < 8 && !ok; ++i) {
        path = Join(NormalPath(parent), L"WrongGPU-elevated-" + RandomHex());
        try { ok = guard.CreateNewDir(path, sd, e); }
        catch (const SetupError& x) { refused = x.msg; break; }
        if (!ok && e != ERROR_ALREADY_EXISTS && e != ERROR_FILE_EXISTS) break;
    }
    if (!ok) why = L"cannot make a private folder in " + parent + L" (" + (refused.empty() ? ErrText(e) : refused) + L")";
    LocalFree(sd);
    return ok;
}

}  // namespace wg
