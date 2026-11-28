// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU launcher core: WrongGPU's edition of OptiScaler, downloaded from its own project's release (optiedition.h).
// Written for this project. Windows' own WinHTTP and the core's SHA-256 only.
#include "optiedition.h"

#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

namespace wg {

// Known older WrongGPU edition DLL SHA-256 hashes (upgraded in place to the pinned edition, never treated as foreign).
static const wchar_t* const kPriorEditionDllSha256[] = {
    L"ff6f06ca2f8e761016f41313e2533fd176623a7a85f9e9169c9fc003b227826d",  // WrongGPU edition v5 (wronggpu-edition-20261102.1)
};

EditionInfo PinnedEdition() {
    EditionInfo e;
    e.url = L"https://github.com/phantoslayer21/OptiScaler/releases/download/wronggpu-edition-20261127.1/"
            L"OptiScaler_v10.0.0-pre1_20261127_WrongGPU-edition-v6.zip";
    e.page = kEditionPage;
    e.file = L"OptiScaler_v10.0.0-pre1_20261127_WrongGPU-edition-v6.zip";
    e.size = 185383587ull;
    e.sha256 = L"1876b742ac5b20b35eee161df223ec02c9cdb71274c7a1e3a7028a9bfcf7fca4";
    e.dllSha256 = L"d377dfb47a4d635c8fa7adc71845cee5018d0507f054cd12a74e1039709727ab";
    e.version = L"10.0.0-pre1 (WrongGPU edition v6)";
#ifdef WRONGGPU_TESTING
    // tests: their own server and stand-in archive, or no download at all
    const wstring t = EnvVar(L"WRONGGPU_TEST_EDITION");
    std::vector<wstring> parts;
    for (size_t a = 0;;) {
        const size_t b = t.find(L'|', a);
        parts.push_back(t.substr(a, b == wstring::npos ? wstring::npos : b - a));
        if (b == wstring::npos) break;
        a = b + 1;
    }
    if (parts.size() == 4 || parts.size() == 5) {
        e.url = parts[0]; e.size = _wtoi64(parts[1].c_str()); e.sha256 = Lower(parts[2]); e.dllSha256 = Lower(parts[3]);
        e.file = (parts.size() == 5 && !parts[4].empty()) ? parts[4] : L"edition-test.zip";
    } else {
        e.url.clear();
        if (!t.empty()) e.dllSha256.clear();
    }
    // the stand-in's OptiScaler.dll counts as the edition also without a download (WRONGGPU_TEST_EDITION_DLL)
    const wstring dll = EnvVar(L"WRONGGPU_TEST_EDITION_DLL");
    if (!dll.empty()) e.dllSha256 = Lower(dll);
#endif
    return e;
}

wstring EditionCacheDir(const wstring& dataDir) { return Join(dataDir, L"optiscaler"); }

static bool Matches(const wstring& path, const EditionInfo& e) {
    return IsFile(path) && FileSize(path) == e.size && IEq(Sha256File(path), e.sha256);
}

bool EditionCached(const wstring& dataDir, wstring* path) {
    const EditionInfo e = PinnedEdition();
    if (dataDir.empty() || e.sha256.empty()) return false;
    const wstring p = Join(EditionCacheDir(dataDir), e.file);
    if (!PathWithin(p, dataDir, true) || !Matches(p, e)) return false;
    if (path) *path = p;
    return true;
}

bool IsEditionDll(const wstring& optiDll) {
    const EditionInfo e = PinnedEdition();
    return !e.dllSha256.empty() && IsFile(optiDll) && IEq(Sha256File(optiDll), e.dllSha256);
}

bool IsPriorEditionHash(const wstring& sha256) {
    if (sha256.empty()) return false;
    const EditionInfo cur = PinnedEdition();
    if (!cur.dllSha256.empty() && IEq(sha256, cur.dllSha256)) return false;
    for (auto h : kPriorEditionDllSha256)
        if (IEq(sha256, h)) return true;
#ifdef WRONGGPU_TESTING
    const wstring extra = EnvVar(L"WRONGGPU_TEST_PRIOR_EDITION_DLL");
    if (!extra.empty() && IEq(sha256, extra)) return true;
#endif
    return false;
}

bool IsPriorEditionDll(const wstring& optiDll) {
    return IsFile(optiDll) && IsPriorEditionHash(Sha256File(optiDll));
}

// One GET into an exclusively created staging file. Status is zero if no answer was received.
static bool Download(const EditionInfo& e, PinnedFile& file, const wstring& dest, const wstring& dataDir, const std::function<bool(unsigned long long, unsigned long long)>& progress,
                     DWORD& status, bool& cancelled, wstring& err) {
    status = 0;
    URL_COMPONENTS uc{}; uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {}, path[4096] = {};
    uc.lpszHostName = host; uc.dwHostNameLength = 256; uc.lpszUrlPath = path; uc.dwUrlPathLength = 4096;
    if (!WinHttpCrackUrl(e.url.c_str(), 0, 0, &uc)) { err = L"address not readable"; return false; }
    bool secure = uc.nScheme == INTERNET_SCHEME_HTTPS;
#ifdef WRONGGPU_TESTING
    const bool local = IEq(host, L"127.0.0.1") || IEq(host, L"localhost");
    if (!secure && !local) { err = L"only https:// (or a test server on 127.0.0.1)"; return false; }
#else
    if (!secure) { err = L"only https://"; return false; }
#endif
    HINTERNET s = WinHttpOpen(L"WrongGPU-launcher", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) { err = L"WinHttpOpen " + ErrText(GetLastError()); return false; }
    WinHttpSetTimeouts(s, 15000, 15000, 30000, 60000);
    DWORD proto = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | 0x00002000 /* TLS 1.3 */;
    if (!WinHttpSetOption(s, WINHTTP_OPTION_SECURE_PROTOCOLS, &proto, sizeof proto)) {
        proto = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
        WinHttpSetOption(s, WINHTTP_OPTION_SECURE_PROTOCOLS, &proto, sizeof proto);
    }
    bool ok = false;
    if (HINTERNET c = WinHttpConnect(s, host, uc.nPort, 0)) {
        if (HINTERNET rq = WinHttpOpenRequest(c, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0)) {
            if (WinHttpSendRequest(rq, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(rq, nullptr)) {
                DWORD len = sizeof(status);
                WinHttpQueryHeaders(rq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &len, WINHTTP_NO_HEADER_INDEX);
                if (status == 200) {
                    DWORD error = 0;
                    if (!PathWithin(dest, dataDir, true)) err = L"unsafe staging path " + dest;
                    else if (!file.Create(dest, &error)) err = L"cannot create " + dest + L": " + ErrText(error);
                    else {
                        std::vector<char> buf(1 << 20);
                        unsigned long long done = 0;
                        DWORD got = 0;
                        bool failed = false;
                        if (progress && !progress(0, e.size)) { cancelled = true; failed = true; }
                        while (!failed) {
                            if (!WinHttpReadData(rq, buf.data(), (DWORD)buf.size(), &got)) { err = L"connection lost: " + ErrText(GetLastError()); failed = true; break; }
                            if (!got) break;
                            done += got;
                            if (done > e.size) { err = L"more data than the pinned size"; failed = true; break; }
                            if (!file.WriteChunk(buf.data(), got, &error)) { err = L"cannot write " + dest + L": " + ErrText(error); failed = true; break; }
                            if (progress && !progress(done, e.size)) { cancelled = true; failed = true; }
                        }
                        if (!failed && !file.Flush(&error)) { err = ErrText(error); failed = true; }
                        if (!failed && done != e.size) err = L"incomplete download (" + std::to_wstring(done) + L" of " + std::to_wstring(e.size) + L" bytes)";
                        ok = !failed && done == e.size;
                    }
                } else {
                    err = L"HTTP status " + std::to_wstring(status);
                }
            } else {
                err = L"no answer: " + ErrText(GetLastError());
            }
            WinHttpCloseHandle(rq);
        } else err = L"WinHttpOpenRequest " + ErrText(GetLastError());
        WinHttpCloseHandle(c);
    } else err = L"WinHttpConnect " + ErrText(GetLastError());
    WinHttpCloseHandle(s);
    return ok;
}

EditionResult GetEdition(Context& ctx, const wstring& dataDir, const std::function<bool(unsigned long long, unsigned long long)>& progress) {
    EditionResult r;
    try {
        const EditionInfo e = PinnedEdition();
        if (e.url.empty() || dataDir.empty()) {
            r.error = L"WrongGPU couldn't download its OptiScaler edition.";
            r.detail = e.url.empty() ? L"download off in this test build (WRONGGPU_TEST_EDITION not set)" : L"no data folder";
            return r;
        }
        const wstring dir = EditionCacheDir(dataDir);
        const wstring final_ = Join(dir, e.file);
        if (!PathWithin(final_, dataDir, true) || !IEq(e.file, BaseName(e.file)))
            Fail(L"The OptiScaler cache path is unsafe: " + final_);
        if (Matches(final_, e)) {
            r.path = final_; r.cached = true;
            LogLine(ctx, L"OptiScaler: WrongGPU edition " + e.version + L" from the launcher's cache (" + final_ + L", sha256 checked)");
            return r;
        }
        if (IsFile(final_)) {  // a copy that doesn't match (changed, cut short): never used
            LogLine(ctx, L"OptiScaler: the cached edition " + final_ + L" doesn't match the pinned file; downloading it again");
        }
        EnsureDir(dir);
        struct CacheDirectory {
            HANDLE handle;
            ~CacheDirectory() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
        } cache{CreateFileW(dir.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                            FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr)};
        if (cache.handle == INVALID_HANDLE_VALUE || !IEq(ResolvedHandlePath(cache.handle), ResolvedPath(dir)) ||
            !PathWithin(dir, dataDir, true))
            Fail(L"The OptiScaler cache directory cannot be secured: " + dir);
        const wstring part = final_ + L"." + std::to_wstring(GetCurrentProcessId()) + L"-" +
                             std::to_wstring(GetTickCount64()) + L".part";
        if (!PathWithin(part, dataDir, true)) Fail(L"The OptiScaler download path is unsafe: " + part);
        LogLine(ctx, L"OptiScaler: downloading WrongGPU's edition " + e.version + L" from " + e.url);
        struct StagingFile {
            PinnedFile file;
            ~StagingFile() { if (file.IsOpen()) file.Delete(); }
        } staging;
        DWORD status = 0;
        wstring err;
        if (!Download(e, staging.file, part, dataDir, progress, status, r.cancelled, err)) {
            r.detail = err;
            r.error = r.cancelled ? L"The OptiScaler download was cancelled."
                      : status == 0 ? L"WrongGPU couldn't download its OptiScaler edition. Check your internet connection."
                                    : L"WrongGPU couldn't download its OptiScaler edition (the release page answered " + std::to_wstring(status) + L").";
            LogLine(ctx, L"OptiScaler: download failed: " + err);
            return r;
        }
        if (!PathWithin(final_, dataDir, true)) Fail(L"The OptiScaler cache path changed during download.");
        const wstring sha = staging.file.Sha256();
        if (!IEq(sha, e.sha256)) {
            r.mismatch = true;
            r.error = L"The OptiScaler download wasn't the file WrongGPU expects, so it wasn't used.";
            r.detail = L"sha256 " + sha + L", expected " + e.sha256;
            LogLine(ctx, L"OptiScaler: download refused: " + r.detail);
            return r;
        }
        DWORD error = 0;
        PinnedFile previous;
        const bool exists = previous.Open(final_, true, &error);
        const bool missing = error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND;
        wstring originalLeft;
        if (!(exists ? previous.ReplaceWith(staging.file, final_, &error, &originalLeft) : missing && staging.file.RenameNew(final_, &error))) {
            r.detail = L"cannot rename " + part + L": " + ErrText(error);
            if (!originalLeft.empty()) r.detail += L". The original is kept at " + originalLeft;
            r.error = L"WrongGPU couldn't save its OptiScaler edition in its data folder.";
            LogLine(ctx, L"OptiScaler: " + r.detail);
            return r;
        }
        staging.file.Close();
        r.path = final_;
        LogLine(ctx, L"OptiScaler: WrongGPU edition " + e.version + L" downloaded and checked (sha256 " + e.sha256.substr(0, 8) + L"): " + final_);
    } catch (const SetupError& x) {
        r.path.clear(); r.error = L"WrongGPU couldn't download its OptiScaler edition."; r.detail = x.msg;
    } catch (...) {
        r.path.clear(); r.error = L"WrongGPU couldn't download its OptiScaler edition."; r.detail = L"unexpected failure";
    }
    return r;
}

}  // namespace wg
