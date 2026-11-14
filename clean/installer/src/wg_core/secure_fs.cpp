// SPDX-License-Identifier: MIT
// The elevated worker's file operations through held handles (secure_fs.h). Written for this project.
//
// The NT open (NtCreateFile) is used for one thing Win32 has no call for: opening a name relative to an open folder's
// handle (OBJECT_ATTRIBUTES.RootDirectory). FILE_OPEN_REPARSE_POINT is the NT form of FILE_FLAG_OPEN_REPARSE_POINT, and
// FILE_DIRECTORY_FILE with FILE_OPEN_FOR_BACKUP_INTENT the NT form of opening a folder with FILE_FLAG_BACKUP_SEMANTICS.
// Everything else is Win32: GetFileInformationByHandleEx, SetFileInformationByHandle (rename, delete, attributes),
// GetFinalPathNameByHandleW.
#include "secure_fs.h"

#include <winternl.h>

#include "package_snapshot.h"  // RandomHex

#include <atomic>

namespace wg {

namespace {

// ---- NT open ------------------------------------------------------------------------------------------------------------
constexpr ULONG kFileOpen = 1, kFileCreate = 2, kFileOpenIf = 3;                     // CreateDisposition
constexpr ULONG kDirectoryFile = 0x1, kSyncIo = 0x20, kNonDirectoryFile = 0x40,       // CreateOptions
                kBackupIntent = 0x4000, kOpenReparsePoint = 0x200000;
constexpr ULONG kCaseInsensitive = 0x40;                                             // OBJ_CASE_INSENSITIVE
constexpr ULONG_PTR kFileCreated = 2;                                                // IO_STATUS_BLOCK.Information
constexpr ULONG kShareNoDelete = FILE_SHARE_READ | FILE_SHARE_WRITE;
constexpr ULONG kShareAll = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
constexpr ACCESS_MASK kDirAccess = FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES;

using NtCreateFileFn = LONG(NTAPI*)(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES, PIO_STATUS_BLOCK, PLARGE_INTEGER, ULONG, ULONG, ULONG, ULONG, PVOID, ULONG);
using NtSetInformationFileFn = LONG(NTAPI*)(HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG, ULONG);
using StatusToErrorFn = ULONG(NTAPI*)(LONG);
struct NtCalls { NtCreateFileFn create = nullptr; NtSetInformationFileFn setInfo = nullptr; StatusToErrorFn toError = nullptr; };
const NtCalls& Nt() {
    static const NtCalls calls = [] {
        NtCalls c;
        if (HMODULE m = GetModuleHandleW(L"ntdll.dll")) {
            c.create = reinterpret_cast<NtCreateFileFn>(reinterpret_cast<void*>(GetProcAddress(m, "NtCreateFile")));
            c.setInfo = reinterpret_cast<NtSetInformationFileFn>(reinterpret_cast<void*>(GetProcAddress(m, "NtSetInformationFile")));
            c.toError = reinterpret_cast<StatusToErrorFn>(reinterpret_cast<void*>(GetProcAddress(m, "RtlNtStatusToDosError")));
        }
        return c;
    }();
    return calls;
}

struct H {
    HANDLE h = nullptr;
    H() = default;
    explicit H(HANDLE x) : h(x == INVALID_HANDLE_VALUE ? nullptr : x) {}
    ~H() { if (h) CloseHandle(h); }
    H(H&& o) noexcept : h(o.h) { o.h = nullptr; }
    H& operator=(H&& o) noexcept { if (this != &o) { if (h) CloseHandle(h); h = o.h; o.h = nullptr; } return *this; }
    H(const H&) = delete;
    H& operator=(const H&) = delete;
    HANDLE get() const { return h; }
    HANDLE release() { HANDLE x = h; h = nullptr; return x; }
    void reset() { if (h) CloseHandle(h); h = nullptr; }
    explicit operator bool() const { return h != nullptr; }
};

// Opens `name` relative to the folder `dir` (dir null: `name` is an NT path), never following a reparse point at `name`
// itself unless `follow`. 0 or the Win32 error. `created`: whether it was created.
DWORD OpenAt(HANDLE dir, const wstring& name, ACCESS_MASK access, ULONG share, ULONG disposition, ULONG options, H& out,
             bool* created = nullptr, PSECURITY_DESCRIPTOR sd = nullptr, bool follow = false) {
    if (!Nt().create || !Nt().toError) return ERROR_PROC_NOT_FOUND;
    if (name.empty() || name.size() > 32000) return ERROR_INVALID_NAME;
    UNICODE_STRING us;
    us.Buffer = const_cast<PWSTR>(name.c_str());
    us.Length = us.MaximumLength = (USHORT)(name.size() * sizeof(wchar_t));
    OBJECT_ATTRIBUTES oa;
    oa.Length = sizeof(oa);
    oa.RootDirectory = dir;
    oa.ObjectName = &us;
    oa.Attributes = kCaseInsensitive;
    oa.SecurityDescriptor = sd;
    oa.SecurityQualityOfService = nullptr;
    IO_STATUS_BLOCK io{};
    HANDLE h = nullptr;
    const LONG st = Nt().create(&h, access | SYNCHRONIZE, &oa, &io, nullptr, FILE_ATTRIBUTE_NORMAL, share, disposition,
                                options | kSyncIo | (follow ? 0 : kOpenReparsePoint), nullptr, 0);
    if (st < 0) return Nt().toError(st);
    out = H(h);
    if (created) *created = io.Information == kFileCreated;
    return 0;
}

// ---- reparse points -----------------------------------------------------------------------------------------------------
// Reparse points that only store a file differently and lead nowhere else: Windows' file compression (WOF),
// deduplication, OneDrive's files on demand (the cloud tags and the older OneDrive tag). Any other reparse point (a
// junction, a symbolic link, and every tag that redirects a name) is a link.
bool StoresOnly(DWORD tag) {
    if (IsReparseTagNameSurrogate(tag)) return false;
    return tag == 0x80000017u /*WOF*/ || tag == 0x80000013u /*DEDUP*/ || (tag & 0xFFFF0FFFu) == 0x9000001Au /*CLOUD_n*/ || tag == 0x80000021u /*ONEDRIVE*/;
}
struct Tag { DWORD attrs = 0, tag = 0; };
bool TagOf(HANDLE h, Tag& t) {
    FILE_ATTRIBUTE_TAG_INFO i{};
    if (!GetFileInformationByHandleEx(h, FileAttributeTagInfo, &i, sizeof(i))) return false;
    t.attrs = i.FileAttributes;
    t.tag = i.ReparseTag;
    return true;
}
bool IsLink(const Tag& t) { return (t.attrs & FILE_ATTRIBUTE_REPARSE_POINT) && !StoresOnly(t.tag); }
bool SameFile(HANDLE a, HANDLE b) {
    FILE_ID_INFO x{}, y{};
    if (GetFileInformationByHandleEx(a, FileIdInfo, &x, sizeof(x)) && GetFileInformationByHandleEx(b, FileIdInfo, &y, sizeof(y)))
        return x.VolumeSerialNumber == y.VolumeSerialNumber && memcmp(&x.FileId, &y.FileId, sizeof(x.FileId)) == 0;
    BY_HANDLE_FILE_INFORMATION p{}, q{};
    return GetFileInformationByHandle(a, &p) && GetFileInformationByHandle(b, &q) && p.dwVolumeSerialNumber == q.dwVolumeSerialNumber &&
           p.nFileIndexHigh == q.nFileIndexHigh && p.nFileIndexLow == q.nFileIndexLow;
}

// A refusal inside this file; the public operations turn it into SetupError (or AddRoot's `why`).
struct Refusal { wstring msg; };
[[noreturn]] void Refuse(const wstring& m) { throw Refusal{m}; }
[[noreturn]] void LinkRefused(const wstring& path) {
    Refuse(path + L" is a link (a junction, a symbolic link or another reparse point that leads elsewhere); the elevated step does not act through links");
}

bool SameName(const wstring& a, const wstring& b) {
    return a.size() == b.size() && CompareStringOrdinal(a.c_str(), (int)a.size(), b.c_str(), (int)b.size(), TRUE) == CSTR_EQUAL;
}

// The final path of `h` must be `expected`.
void CheckFinal(HANDLE h, const wstring& expected) {
    const wstring got = FinalPathOfHandle(h);
    if (got.empty()) Refuse(L"Windows cannot say where " + expected + L" is; the elevated step stops");
    if (!SameName(got, expected)) Refuse(expected + L" is not where its path says (Windows reports " + got + L"); the elevated step stops");
}

// ---- paths --------------------------------------------------------------------------------------------------------------
struct Parsed {
    wstring norm;                 // NormalPath
    wstring dosRoot, ntRoot;      // "C:\" and "\??\C:\", or "\\server\share\" and "\??\UNC\server\share\"
    std::vector<wstring> parts;   // the components below the root
};
bool GoodPart(const wstring& s) {
    if (s.empty() || s == L"." || s == L"..") return false;
    for (wchar_t c : s) if (c < 32 || wcschr(L"<>:\"/\\|?*", c)) return false;  // no stream names, no wildcards
    return true;
}
bool Parse(const wstring& in, Parsed& p) {
    p = Parsed();
    p.norm = NormalPath(in);
    if (p.norm.empty()) return false;
    wstring rest;
    if (p.norm[1] == L':') {
        p.dosRoot = p.norm.substr(0, 2) + L"\\";
        p.ntRoot = L"\\??\\" + p.dosRoot;
        rest = p.norm.size() > 3 ? p.norm.substr(3) : wstring();
    } else {
        const size_t server = p.norm.find(L'\\', 2), share = server == wstring::npos ? wstring::npos : p.norm.find(L'\\', server + 1);
        if (server == wstring::npos || server == 2) return false;
        const wstring head = share == wstring::npos ? p.norm : p.norm.substr(0, share);
        if (head.size() <= server + 1) return false;
        p.dosRoot = head + L"\\";
        p.ntRoot = L"\\??\\UNC\\" + head.substr(2) + L"\\";
        rest = share == wstring::npos ? wstring() : p.norm.substr(share + 1);
    }
    size_t at = 0;
    while (at < rest.size()) {
        size_t e = rest.find(L'\\', at);
        if (e == wstring::npos) e = rest.size();
        const wstring part = rest.substr(at, e - at);
        if (!GoodPart(part)) return false;
        p.parts.push_back(part);
        at = e + 1;
    }
    return true;
}
wstring PathOf(const Parsed& p, size_t n) {
    wstring s = p.dosRoot;
    for (size_t i = 0; i < n; ++i) s += (i ? L"\\" : L"") + p.parts[i];
    return n ? s : (p.dosRoot.size() == 3 ? p.dosRoot : p.dosRoot.substr(0, p.dosRoot.size() - 1));
}

// The folder `name` below `dir`, held without delete sharing; a link refused; a reparse point that only stores the folder
// differently reopened as the folder itself (checked to be the same).
DWORD OpenDirAt(HANDLE dir, const wstring& name, const wstring& path, H& out) {
    DWORD e = OpenAt(dir, name, kDirAccess, kShareNoDelete, kFileOpen, kDirectoryFile | kBackupIntent, out);
    if (e == ERROR_ACCESS_DENIED) e = OpenAt(dir, name, FILE_TRAVERSE | FILE_READ_ATTRIBUTES, kShareNoDelete, kFileOpen, kDirectoryFile | kBackupIntent, out);
    if (e) return e;
    Tag t;
    if (!TagOf(out.get(), t)) { out.reset(); return GetLastError(); }
    if (IsLink(t)) { out.reset(); LinkRefused(path); }
    if ((t.attrs & FILE_ATTRIBUTE_REPARSE_POINT) && dir) {
        H again;
        if (DWORD e2 = OpenAt(dir, name, kDirAccess, kShareNoDelete, kFileOpen, kDirectoryFile | kBackupIntent, again, nullptr, nullptr, true)) return e2;
        if (!SameFile(out.get(), again.get())) Refuse(path + L" changed while it was opened; the elevated step stops");
        out = std::move(again);
    }
    return 0;
}

// The volume root and the folders parts[0..n), each held; 0 or the Win32 error (a link: Refusal).
DWORD OpenChain(const Parsed& p, size_t n, std::vector<H>& chain) {
    H vol;
    DWORD e = OpenAt(nullptr, p.ntRoot, kDirAccess, kShareNoDelete, kFileOpen, kDirectoryFile | kBackupIntent, vol);
    if (e == ERROR_ACCESS_DENIED) e = OpenAt(nullptr, p.ntRoot, FILE_TRAVERSE | FILE_READ_ATTRIBUTES, kShareNoDelete, kFileOpen, kDirectoryFile | kBackupIntent, vol);
    if (e) return e;
    chain.push_back(std::move(vol));
    for (size_t i = 0; i < n; ++i) {
        H d;
        if ((e = OpenDirAt(chain.back().get(), p.parts[i], PathOf(p, i + 1), d))) return e == ERROR_DIRECTORY ? ERROR_PATH_NOT_FOUND : e;
        chain.push_back(std::move(d));
    }
    return 0;
}

// The entries of the open folder `h`.
DWORD ListHandle(HANDLE h, std::vector<FsGuard::Entry>& out) {
    std::vector<unsigned char> buf(64 * 1024);
    bool first = true;
    for (;;) {
        if (!GetFileInformationByHandleEx(h, first ? FileFullDirectoryRestartInfo : FileFullDirectoryInfo, buf.data(), (DWORD)buf.size())) {
            const DWORD e = GetLastError();
            return e == ERROR_NO_MORE_FILES || e == ERROR_FILE_NOT_FOUND ? 0 : e;
        }
        first = false;
        size_t at = 0;
        for (;;) {
            const auto* fi = reinterpret_cast<const FILE_FULL_DIR_INFO*>(buf.data() + at);
            const wstring name(fi->FileName, fi->FileNameLength / sizeof(wchar_t));
            if (name != L"." && name != L"..") {
                DWORD attrs = fi->FileAttributes;
                if ((attrs & FILE_ATTRIBUTE_REPARSE_POINT) && StoresOnly(fi->EaSize)) attrs &= ~FILE_ATTRIBUTE_REPARSE_POINT;  // EaSize: the tag
                out.push_back({name, attrs, (unsigned long long)fi->EndOfFile.QuadPart});
            }
            if (!fi->NextEntryOffset) break;
            at += fi->NextEntryOffset;
        }
    }
}

// Deletes the open object `h` (it must have DELETE access): its name goes at once where the volume allows it.
DWORD Dispose(HANDLE h) {
    struct { ULONG flags; } ex{0x1 | 0x2};  // FILE_DISPOSITION_INFO_EX: DELETE | POSIX_SEMANTICS
    if (SetFileInformationByHandle(h, (FILE_INFO_BY_HANDLE_CLASS)21 /*FileDispositionInfoEx*/, &ex, sizeof(ex))) return 0;
    const DWORD e = GetLastError();
    if (e != ERROR_INVALID_PARAMETER && e != ERROR_NOT_SUPPORTED && e != ERROR_INVALID_FUNCTION) return e;
    FILE_DISPOSITION_INFO d{TRUE};
    return SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof(d)) ? 0 : GetLastError();
}
// Renames the open object `h` (DELETE access) to `leaf` in the open folder `dir`. NtSetInformationFile with
// FileRenameInformation: SetFileInformationByHandle(FileRenameInfo) refuses a root directory handle (ERROR_INVALID_PARAMETER),
// so the rename relative to the held folder takes the NT call with the same structure (FILE_RENAME_INFO's layout).
DWORD RenameHandle(HANDLE h, HANDLE dir, const wstring& leaf, bool replace) {
    if (!Nt().setInfo || !Nt().toError) return ERROR_PROC_NOT_FOUND;
    std::vector<unsigned char> buf(sizeof(FILE_RENAME_INFO) + leaf.size() * sizeof(wchar_t));
    auto* ri = reinterpret_cast<FILE_RENAME_INFO*>(buf.data());
    ri->ReplaceIfExists = replace ? TRUE : FALSE;
    ri->RootDirectory = dir;
    ri->FileNameLength = (DWORD)(leaf.size() * sizeof(wchar_t));
    memcpy(ri->FileName, leaf.data(), leaf.size() * sizeof(wchar_t));
    IO_STATUS_BLOCK io{};
    const LONG st = Nt().setInfo(h, &io, ri, (ULONG)buf.size(), 10 /*FileRenameInformation*/);
    return st < 0 ? Nt().toError(st) : 0;
}
DWORD SetAttrsOf(HANDLE h, DWORD attrs) {
    FILE_BASIC_INFO bi{};
    attrs &= FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_TEMPORARY |
             FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED;
    bi.FileAttributes = attrs ? attrs : FILE_ATTRIBUTE_NORMAL;
    return SetFileInformationByHandle(h, FileBasicInfo, &bi, sizeof(bi)) ? 0 : GetLastError();
}
bool WriteAllTo(HANDLE h, const void* data, size_t size, DWORD& err) {
    const char* p = static_cast<const char*>(data);
    size_t done = 0;
    while (done < size) {
        DWORD wrote = 0;
        if (!WriteFile(h, p + done, (DWORD)std::min<size_t>(size - done, 1u << 24), &wrote, nullptr) || !wrote) { err = GetLastError(); return false; }
        done += wrote;
    }
    return true;
}

wstring LongFullPath(const wstring& p) {
    DWORD n = GetFullPathNameW(p.c_str(), 0, nullptr, nullptr);
    if (!n) return L"";
    wstring b(n, L'\0');
    n = GetFullPathNameW(p.c_str(), n, &b[0], nullptr);
    if (!n || n >= b.size()) return L"";
    b.resize(n);
    return b;
}

std::atomic<FsGuard*> g_active{nullptr};

}  // namespace

// ---- paths (public) -----------------------------------------------------------------------------------------------------
bool ReparseTagIsLink(DWORD tag) { return !StoresOnly(tag); }

wstring NormalPath(const wstring& in) {
    wstring s = in;
    for (auto& c : s) if (c == L'/') c = L'\\';
    if (s.rfind(L"\\\\?\\UNC\\", 0) == 0) s = L"\\\\" + s.substr(8);
    else if (s.rfind(L"\\\\?\\", 0) == 0) s = s.substr(4);
    const bool drive = s.size() >= 3 && iswalpha(s[0]) && s[1] == L':' && s[2] == L'\\';
    const bool unc = s.size() > 4 && s[0] == L'\\' && s[1] == L'\\' && s[2] != L'?' && s[2] != L'.' && s[2] != L'\\';
    if (!drive && !unc) return L"";
    s = LongFullPath(s);
    if (s.size() < 3 || (s[1] == L':' && s[2] != L'\\') || s.rfind(L"\\\\?\\", 0) == 0 || s.rfind(L"\\\\.\\", 0) == 0) return L"";
    while (s.size() > 3 && s.back() == L'\\') s.pop_back();
    if (s[1] == L':') s[0] = (wchar_t)towupper(s[0]);
    return s;
}

wstring FinalPathOfHandle(HANDLE h) {
    DWORD n = GetFinalPathNameByHandleW(h, nullptr, 0, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (!n) return L"";
    wstring b(n + 1, L'\0');
    n = GetFinalPathNameByHandleW(h, &b[0], (DWORD)b.size(), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (!n || n >= b.size()) return L"";
    b.resize(n);
    if (b.rfind(L"\\\\?\\UNC\\", 0) == 0) b = L"\\\\" + b.substr(8);
    else if (b.rfind(L"\\\\?\\", 0) == 0) b = b.substr(4);
    while (b.size() > 3 && b.back() == L'\\') b.pop_back();
    if (b.size() >= 2 && b[1] == L':') b[0] = (wchar_t)towupper(b[0]);
    return b;
}

wstring BoundPath(const wstring& p) {
    if (p.empty()) return p;
    const wstring norm = NormalPath(p);
    if (norm.empty()) return p;
    H h(CreateFileW(norm.c_str(), FILE_READ_ATTRIBUTES, kShareAll, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr));
    if (h) {
        const wstring f = FinalPathOfHandle(h.get());
        return f.empty() ? norm : f;
    }
    const wstring parent = Parent(norm);
    if (norm.size() <= 3 || parent.empty() || parent.size() >= norm.size() || (parent.size() == 2 && parent[1] == L':')) return norm;
    return Join(BoundPath(parent), BaseName(norm));
}

// ---- the guard ----------------------------------------------------------------------------------------------------------
struct FsGuard::Impl {
    struct Root { Parsed p; std::vector<H> chain; };
    std::vector<std::unique_ptr<Root>> roots;

    // `path` resolved to its parent folder's open handle: below the nearest root's held handle, else (reading only) below
    // the volume root. 0 or the error (a folder on the way missing; ERROR_INVALID_NAME: not an absolute path).
    struct Walk {
        Parsed p;
        std::vector<H> chain;      // the folders opened for this walk
        HANDLE parent = nullptr;   // the leaf's folder; the object itself when `leaf` is empty
        wstring leaf;              // empty: the path is a root (or a volume root)
        const Root* root = nullptr;
    };
    DWORD WalkTo(const wstring& path, bool write, Walk& w) const {
        if (!Parse(path, w.p)) {
            if (write) Refuse((path.empty() ? wstring(L"(no path)") : path) + L" is not an absolute path to a file or folder");
            return ERROR_INVALID_NAME;
        }
        for (auto& r : roots) {
            if (!SameName(r->p.dosRoot, w.p.dosRoot) || r->p.parts.size() > w.p.parts.size()) continue;
            bool in = true;
            for (size_t i = 0; in && i < r->p.parts.size(); ++i) in = SameName(r->p.parts[i], w.p.parts[i]);
            if (in && (!w.root || r->p.parts.size() > w.root->p.parts.size())) w.root = r.get();
        }
        if (!w.root && write) Refuse(w.p.norm + L" is outside the folders this job may change");
        size_t depth = 0;
        HANDLE at = nullptr;
        if (w.root) { depth = w.root->p.parts.size(); at = w.root->chain.back().get(); }
        else {
            if (DWORD e = OpenChain(w.p, 0, w.chain)) return e;
            at = w.chain.back().get();
        }
        if (depth == w.p.parts.size()) { w.parent = at; return 0; }
        for (size_t i = depth; i + 1 < w.p.parts.size(); ++i) {
            H d;
            if (DWORD e = OpenDirAt(at, w.p.parts[i], PathOf(w.p, i + 1), d)) return e == ERROR_FILE_NOT_FOUND || e == ERROR_DIRECTORY ? ERROR_PATH_NOT_FOUND : e;
            at = d.get();
            w.chain.push_back(std::move(d));
        }
        w.parent = at;
        w.leaf = w.p.parts.back();
        return 0;
    }
    // The leaf of `w`, never through a link, its final path checked; `data`: a reparse point that only stores the file
    // differently is reopened as the file itself (checked to be the same) so that its contents read as usual.
    DWORD OpenLeaf(const Walk& w, ACCESS_MASK access, ULONG share, ULONG options, H& out, bool data) const {
        if (DWORD e = OpenAt(w.parent, w.leaf, access, share, kFileOpen, options | kBackupIntent, out)) return e;
        Tag t;
        if (!TagOf(out.get(), t)) { const DWORD e = GetLastError(); out.reset(); return e; }
        if (IsLink(t)) { out.reset(); LinkRefused(w.p.norm); }
        if (data && (t.attrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
            H again;
            if (DWORD e = OpenAt(w.parent, w.leaf, access, share, kFileOpen, options | kBackupIntent, again, nullptr, nullptr, true)) return e;
            if (!SameFile(out.get(), again.get())) Refuse(w.p.norm + L" changed while it was opened; the elevated step stops");
            out = std::move(again);
        }
        CheckFinal(out.get(), w.p.norm);
        return 0;
    }
    // A new file next to the leaf of `w` (`leaf` + suffix), for writing; an old one of that name (left by an interrupted
    // run) is deleted first.
    DWORD CreateTemp(const Walk& w, H& out) const {
        const wstring name = w.leaf + kTmpSuffix;
        DWORD e = OpenAt(w.parent, name, FILE_GENERIC_WRITE | DELETE | FILE_READ_ATTRIBUTES, 0, kFileCreate, kNonDirectoryFile, out);
        if (e == ERROR_ALREADY_EXISTS || e == ERROR_FILE_EXISTS) {
            H old;
            if (!OpenAt(w.parent, name, DELETE | FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES, kShareAll, kFileOpen, kNonDirectoryFile, old)) {
                Tag t;
                if (TagOf(old.get(), t) && IsLink(t)) LinkRefused(w.p.norm + kTmpSuffix);
                SetAttrsOf(old.get(), FILE_ATTRIBUTE_NORMAL);
                Dispose(old.get());
            }
            old.reset();
            e = OpenAt(w.parent, name, FILE_GENERIC_WRITE | DELETE | FILE_READ_ATTRIBUTES, 0, kFileCreate, kNonDirectoryFile, out);
        }
        if (!e) CheckFinal(out.get(), w.p.norm + kTmpSuffix);
        return e;
    }
    // The existing leaf of `w` (if any) made replaceable: a link refused, read-only cleared.
    void PrepareTarget(const Walk& w) const {
        H old;
        if (OpenAt(w.parent, w.leaf, FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES, kShareAll, kFileOpen, kBackupIntent, old)) return;
        Tag t;
        if (!TagOf(old.get(), t)) return;
        if (IsLink(t)) LinkRefused(w.p.norm);
        if (t.attrs & FILE_ATTRIBUTE_READONLY) SetAttrsOf(old.get(), FILE_ATTRIBUTE_NORMAL);
    }
    // Moves the written temporary file `tmp` over the leaf of `w`.
    DWORD Commit(const Walk& w, H& tmp) const {
        FlushFileBuffers(tmp.get());
        try { PrepareTarget(w); } catch (const Refusal&) { Dispose(tmp.get()); throw; }
        if (DWORD e = RenameHandle(tmp.get(), w.parent, w.leaf, true)) { Dispose(tmp.get()); return e; }
        CheckFinal(tmp.get(), w.p.norm);
        return 0;
    }
    void RemoveTreeAt(HANDLE dir, const wstring& path, const wstring& keep) const {
        std::vector<FsGuard::Entry> entries;
        if (ListHandle(dir, entries)) return;
        for (auto& en : entries) {
            const wstring child = Join(path, en.name);
            if (en.attrs & FILE_ATTRIBUTE_REPARSE_POINT) continue;  // a link: never followed, never removed
            H c;
            if (OpenAt(dir, en.name, DELETE | FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES, kShareAll, kFileOpen, kBackupIntent, c)) continue;
            Tag t;
            if (!TagOf(c.get(), t) || IsLink(t)) continue;
            if (t.attrs & FILE_ATTRIBUTE_DIRECTORY) RemoveTreeAt(c.get(), child, keep);
            else if (!keep.empty() && SameName(child, keep)) continue;
            else if (t.attrs & FILE_ATTRIBUTE_READONLY) SetAttrsOf(c.get(), FILE_ATTRIBUTE_NORMAL);
            Dispose(c.get());
        }
    }
};

FsGuard::FsGuard() : impl_(new Impl) {}
FsGuard::~FsGuard() {
    FsGuard* self = this;
    g_active.compare_exchange_strong(self, nullptr);
}

bool FsGuard::AddRoot(const wstring& path, wstring& why) {
    auto r = std::make_unique<Impl::Root>();
    if (!Parse(path, r->p)) { why = path + L" is not an absolute path to a folder"; return false; }
    try {
        if (DWORD e = OpenChain(r->p, r->p.parts.size(), r->chain)) {
            why = L"the folder " + r->p.norm + L" cannot be opened (" + ErrText(e) + L")";
            return false;
        }
        CheckFinal(r->chain.back().get(), r->p.norm);
    } catch (const Refusal& x) {
        why = x.msg;
        return false;
    }
    impl_->roots.push_back(std::move(r));
    return true;
}
std::vector<wstring> FsGuard::Roots() const {
    std::vector<wstring> out;
    for (auto& r : impl_->roots) out.push_back(r->p.norm);
    return out;
}

// Every public operation: a Refusal becomes SetupError (EXIT_REFUSED).
#define WG_GUARDED(...) try { __VA_ARGS__ } catch (const Refusal& x) { Fail(x.msg, EXIT_REFUSED); }

DWORD FsGuard::Attributes(const wstring& path, unsigned long long* size, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, false, w))) return INVALID_FILE_ATTRIBUTES;
        H h;
        HANDLE target = w.parent;
        if (!w.leaf.empty()) {
            if ((err = impl_->OpenLeaf(w, FILE_READ_ATTRIBUTES, kShareAll, 0, h, false))) return INVALID_FILE_ATTRIBUTES;
            target = h.get();
        }
        Tag t;
        if (!TagOf(target, t)) { err = GetLastError(); return INVALID_FILE_ATTRIBUTES; }
        if (size) {
            FILE_STANDARD_INFO si{};
            *size = GetFileInformationByHandleEx(target, FileStandardInfo, &si, sizeof(si)) ? (unsigned long long)si.EndOfFile.QuadPart : 0;
        }
        err = 0;
        return t.attrs & ~(DWORD)FILE_ATTRIBUTE_REPARSE_POINT;  // (a link was refused above)
    )
}

HANDLE FsGuard::OpenRead(const wstring& path, DWORD share, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, false, w))) return INVALID_HANDLE_VALUE;
        if (w.leaf.empty()) { err = ERROR_ACCESS_DENIED; return INVALID_HANDLE_VALUE; }
        H h;
        if ((err = impl_->OpenLeaf(w, FILE_GENERIC_READ, share, kNonDirectoryFile, h, true))) return INVALID_HANDLE_VALUE;
        return h.release();
    )
}

HANDLE FsGuard::CreateNew(const wstring& path, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, true, w))) return INVALID_HANDLE_VALUE;
        if (w.leaf.empty()) Refuse(w.p.norm + L" is one of the job's folders, not a file");
        H h;
        if ((err = OpenAt(w.parent, w.leaf, FILE_GENERIC_WRITE | DELETE | FILE_READ_ATTRIBUTES, 0, kFileCreate, kNonDirectoryFile, h))) return INVALID_HANDLE_VALUE;
        CheckFinal(h.get(), w.p.norm);
        return h.release();
    )
}

bool FsGuard::CreateNewDir(const wstring& path, PSECURITY_DESCRIPTOR sd, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, true, w))) return false;
        if (w.leaf.empty()) { err = ERROR_ALREADY_EXISTS; return false; }
        H h;
        if ((err = OpenAt(w.parent, w.leaf, FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES, kShareAll, kFileCreate, kDirectoryFile | kBackupIntent, h, nullptr, sd))) return false;
        CheckFinal(h.get(), w.p.norm);
        return true;
    )
}

void FsGuard::EnsureDir(const wstring& dir, std::vector<wstring>* created) {
    WG_GUARDED(
        Impl::Walk w;
        if (!Parse(dir, w.p)) Refuse((dir.empty() ? wstring(L"(no path)") : dir) + L" is not an absolute path to a folder");
        // the deepest root above it (outside every root: refused), then each missing folder made below the one before
        impl_->WalkTo(dir, true, w);
        const Impl::Root* r = w.root;
        HANDLE at = r->chain.back().get();
        std::vector<H> chain;
        for (size_t i = r->p.parts.size(); i < w.p.parts.size(); ++i) {
            const wstring here = PathOf(w.p, i + 1);
            H d;
            bool made = false;
            DWORD e = OpenAt(at, w.p.parts[i], kDirAccess, kShareNoDelete, kFileOpenIf, kDirectoryFile | kBackupIntent, d, &made);
            if (e == ERROR_DIRECTORY || e == ERROR_FILE_EXISTS || e == ERROR_ALREADY_EXISTS) {
                H f;
                if (!OpenAt(at, w.p.parts[i], FILE_READ_ATTRIBUTES, kShareAll, kFileOpen, kBackupIntent, f)) {
                    Tag t;
                    if (TagOf(f.get(), t) && IsLink(t)) LinkRefused(here);
                }
                Fail(L"A file is in the way of the folder " + here + L". Remove or rename it and run setup again.");
            }
            if (e) Fail(L"Cannot create folder " + here + L": " + ErrText(e));
            Tag t;
            if (!TagOf(d.get(), t) || IsLink(t)) LinkRefused(here);
            CheckFinal(d.get(), here);
            if (made && created) created->push_back(here);
            at = d.get();
            chain.push_back(std::move(d));
        }
    )
}

void FsGuard::WriteAtomic(const wstring& path, const void* data, size_t size) {
    WG_GUARDED(
        Impl::Walk w;
        if (DWORD e = impl_->WalkTo(path, true, w)) Fail(L"Cannot write " + path + L": " + ErrText(e));
        if (w.leaf.empty()) Refuse(w.p.norm + L" is one of the job's folders, not a file");
        H tmp;
        if (DWORD e = impl_->CreateTemp(w, tmp)) Fail(L"Cannot write " + path + L": " + ErrText(e));
        DWORD e = 0;
        if (!WriteAllTo(tmp.get(), data, size, e)) { Dispose(tmp.get()); Fail(L"Write failed for " + path + L": " + ErrText(e)); }
        if ((e = impl_->Commit(w, tmp))) Fail(L"Cannot replace " + path + L": " + ErrText(e) + L". Is the file in use?");
    )
}

bool FsGuard::Copy(const wstring& src, const wstring& dst, DWORD& err) {
    WG_GUARDED(
        H in(OpenRead(src, FILE_SHARE_READ | FILE_SHARE_DELETE, err));
        if (!in) return false;
        Impl::Walk w;
        if ((err = impl_->WalkTo(dst, true, w))) return false;
        if (w.leaf.empty()) Refuse(w.p.norm + L" is one of the job's folders, not a file");
        H tmp;
        if ((err = impl_->CreateTemp(w, tmp))) return false;
        std::vector<char> buf(1 << 20);
        for (;;) {
            DWORD got = 0;
            if (!ReadFile(in.get(), buf.data(), (DWORD)buf.size(), &got, nullptr)) { err = GetLastError(); Dispose(tmp.get()); return false; }
            if (!got) break;
            if (!WriteAllTo(tmp.get(), buf.data(), got, err)) { Dispose(tmp.get()); return false; }
        }
        // as CopyFileW: the source's attributes and last write time
        FILE_BASIC_INFO bi{};
        if (GetFileInformationByHandleEx(in.get(), FileBasicInfo, &bi, sizeof(bi))) {
            FILE_BASIC_INFO set{};
            set.LastWriteTime = bi.LastWriteTime;
            set.FileAttributes = bi.FileAttributes & (FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_ARCHIVE);
            if (!set.FileAttributes) set.FileAttributes = FILE_ATTRIBUTE_NORMAL;
            FlushFileBuffers(tmp.get());
            SetFileInformationByHandle(tmp.get(), FileBasicInfo, &set, sizeof(set));
        }
        if ((err = impl_->Commit(w, tmp))) return false;
        return true;
    )
}

bool FsGuard::Unlink(const wstring& path, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, true, w))) return false;
        if (w.leaf.empty()) Refuse(w.p.norm + L" is one of the job's folders; it is never deleted");
        H h;
        if ((err = impl_->OpenLeaf(w, DELETE | FILE_READ_ATTRIBUTES, kShareAll, kNonDirectoryFile, h, false))) return false;
        return !(err = Dispose(h.get()));
    )
}

bool FsGuard::RemoveDir(const wstring& path, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, true, w))) return false;
        if (w.leaf.empty()) Refuse(w.p.norm + L" is one of the job's folders; it is never removed");
        H h;
        if ((err = impl_->OpenLeaf(w, DELETE | FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES, kShareAll, kDirectoryFile, h, false))) return false;
        return !(err = Dispose(h.get()));
    )
}

void FsGuard::RemoveTree(const wstring& dir, const wstring& keepFile) {
    WG_GUARDED(
        Impl::Walk w;
        if (impl_->WalkTo(dir, true, w)) return;
        if (w.leaf.empty()) Refuse(w.p.norm + L" is one of the job's folders; it is never removed");
        H h;
        if (impl_->OpenLeaf(w, DELETE | FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES, kShareAll, kDirectoryFile, h, false)) return;
        const wstring keep = keepFile.empty() ? wstring() : NormalPath(keepFile);
        impl_->RemoveTreeAt(h.get(), w.p.norm, keep);
        Dispose(h.get());
    )
}

bool FsGuard::Move(const wstring& from, const wstring& to, bool replace, bool copyAllowed, DWORD& err) {
    WG_GUARDED(
        Impl::Walk f, t;
        if ((err = impl_->WalkTo(from, true, f))) return false;
        if (f.leaf.empty()) Refuse(f.p.norm + L" is one of the job's folders; it is never moved");
        H s;
        if ((err = impl_->OpenLeaf(f, DELETE | FILE_READ_ATTRIBUTES, kShareAll, 0, s, false))) return false;
        if ((err = impl_->WalkTo(to, true, t))) return false;
        if (t.leaf.empty()) Refuse(t.p.norm + L" is one of the job's folders");
        {   // a link at the target is refused (a rename would replace the link itself, but nothing is done with links)
            H old;
            if (!OpenAt(t.parent, t.leaf, FILE_READ_ATTRIBUTES, kShareAll, kFileOpen, kBackupIntent, old)) {
                Tag tg;
                if (TagOf(old.get(), tg) && IsLink(tg)) LinkRefused(t.p.norm);
                if (!replace) { err = ERROR_ALREADY_EXISTS; return false; }
            }
        }
        err = RenameHandle(s.get(), t.parent, t.leaf, replace);
        if (err == ERROR_NOT_SAME_DEVICE && copyAllowed) {
            Tag tg;
            if (!TagOf(s.get(), tg) || (tg.attrs & FILE_ATTRIBUTE_DIRECTORY)) return false;
            s.reset();
            if (!Copy(from, to, err)) return false;
            return Unlink(from, err);
        }
        if (err) return false;
        CheckFinal(s.get(), t.p.norm);
        return true;
    )
}

bool FsGuard::SetAttributes(const wstring& path, DWORD attrs, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(path, true, w))) return false;
        if (w.leaf.empty()) return true;  // a root's own attributes are not changed
        H own;
        if ((err = impl_->OpenLeaf(w, FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES, kShareAll, 0, own, false))) return false;
        return !(err = SetAttrsOf(own.get(), attrs));
    )
}

bool FsGuard::List(const wstring& dir, std::vector<Entry>& out, DWORD& err) {
    WG_GUARDED(
        Impl::Walk w;
        if ((err = impl_->WalkTo(dir, false, w))) return false;
        H h;
        HANDLE target = w.parent;
        if (!w.leaf.empty()) {
            if ((err = impl_->OpenLeaf(w, kDirAccess, kShareAll, kDirectoryFile, h, true))) return false;
            target = h.get();
        }
        return !(err = ListHandle(target, out));
    )
}

#undef WG_GUARDED

// ---- the converter's files ----------------------------------------------------------------------------------------------
static wstring PrivateSub(const wstring& privateDir, const wchar_t* what) {
    if (privateDir.empty()) Fail(L"the elevated step has no private folder for the converter's files");
    return Join(privateDir, wstring(what) + RandomHex());
}
ConverterFile::ConverterFile(const wstring& file, const wstring& privateDir) : path_(file) {
    if (!ActiveFsGuard() || file.empty()) return;
    dir_ = PrivateSub(privateDir, L"converter-in-");
    EnsureDir(dir_);
    path_ = Join(dir_, BaseName(file));
    CopyPath(file, path_);  // (not there: the converter then finds no file, as it would have)
}
ConverterFile::~ConverterFile() {
    if (dir_.empty()) return;
    try { RemoveTree(dir_); } catch (const SetupError&) {}
}
ConverterDir::ConverterDir(const wstring& dir, const wstring& privateDir) : dir_(dir), path_(dir) {
    if (ActiveFsGuard()) path_ = PrivateSub(privateDir, L"converter-out-");
}
ConverterDir::~ConverterDir() {
    if (path_ == dir_) return;
    try { RemoveTree(path_); } catch (const SetupError&) {}
}
void ConverterDir::Publish() {
    if (path_ == dir_) return;
    EnsureDir(dir_);
    for (auto& f : ListFiles(path_, L"*")) CopyChecked(f, Join(dir_, BaseName(f)));
}

FsGuard* ActiveFsGuard() { return g_active.load(); }
ScopedFsGuard::ScopedFsGuard(FsGuard* g) : prev_(g_active.exchange(g)) {}
ScopedFsGuard::~ScopedFsGuard() { g_active.store(prev_); }

}  // namespace wg
