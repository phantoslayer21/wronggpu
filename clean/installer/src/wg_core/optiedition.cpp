// SPDX-License-Identifier: MIT
// WrongGPU launcher core: WrongGPU's edition of OptiScaler, downloaded from its own project's release (optiedition.h).
// Written for this project. Windows' own WinHTTP and the core's SHA-256 only.
#include "optiedition.h"

#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

namespace wg {

EditionInfo PinnedEdition() {
    EditionInfo e;
    e.url = L"https://github.com/phantoslayer21/OptiScaler/releases/download/wronggpu-edition-20261102.1/"
            L"OptiScaler_v10.0.0-pre1_20261030_WrongGPU-edition-v5.zip";
    e.page = kEditionPage;
    e.file = L"OptiScaler_v10.0.0-pre1_20261030_WrongGPU-edition-v5.zip";
    e.size = 185500996ull;
    e.sha256 = L"c285e7e3c029924dccf196c1d1c3b63ef499669df3cdd330161d62119eef90fb";
    e.dllSha256 = L"ff6f06ca2f8e761016f41313e2533fd176623a7a85f9e9169c9fc003b227826d";
    e.version = L"10.0.0-pre1 (WrongGPU edition v5)";
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
    if (parts.size() == 4) {
        e.url = parts[0]; e.size = _wtoi64(parts[1].c_str()); e.sha256 = Lower(parts[2]); e.dllSha256 = Lower(parts[3]);
        e.file = L"edition-test.zip";
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
    if (!Matches(p, e)) return false;
    if (path) *path = p;
    return true;
}

bool IsEditionDll(const wstring& optiDll) {
    const EditionInfo e = PinnedEdition();
    return !e.dllSha256.empty() && IsFile(optiDll) && IEq(Sha256File(optiDll), e.dllSha256);
}

// One GET into `dest` (truncated first). status: the HTTP status (0: no answer). False: no complete body (err says why).
static bool Download(const EditionInfo& e, const wstring& dest, const std::function<bool(unsigned long long, unsigned long long)>& progress,
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
    HANDLE f = INVALID_HANDLE_VALUE;
    if (HINTERNET c = WinHttpConnect(s, host, uc.nPort, 0)) {
        // GitHub answers a release asset with a redirect to its download host (https to https: WinHTTP follows it)
        if (HINTERNET rq = WinHttpOpenRequest(c, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0)) {
            if (WinHttpSendRequest(rq, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(rq, nullptr)) {
                DWORD len = sizeof(status);
                WinHttpQueryHeaders(rq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &len, WINHTTP_NO_HEADER_INDEX);
                if (status == 200) {
                    f = CreateFileW(dest.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
                    if (f == INVALID_HANDLE_VALUE) err = L"cannot write " + dest + L": " + ErrText(GetLastError());
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
                            DWORD w = 0;
                            if (!WriteFile(f, buf.data(), got, &w, nullptr) || w != got) { err = L"cannot write " + dest + L": " + ErrText(GetLastError()); failed = true; break; }
                            if (progress && !progress(done, e.size)) { cancelled = true; failed = true; break; }
                        }
                        CloseHandle(f); f = INVALID_HANDLE_VALUE;
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
        if (Matches(final_, e)) {
            r.path = final_; r.cached = true;
            LogLine(ctx, L"OptiScaler: WrongGPU edition " + e.version + L" from the launcher's cache (" + final_ + L", sha256 checked)");
            return r;
        }
        if (IsFile(final_)) {  // a copy that doesn't match (changed, cut short): never used
            LogLine(ctx, L"OptiScaler: the cached edition " + final_ + L" doesn't match the pinned file; downloading it again");
            ForceDelete(final_);
        }
        EnsureDir(dir);
        const wstring part = final_ + L".part";
        LogLine(ctx, L"OptiScaler: downloading WrongGPU's edition " + e.version + L" from " + e.url);
        DWORD status = 0;
        wstring err;
        const bool got = Download(e, part, progress, status, r.cancelled, err);
        if (!got) {
            if (IsFile(part)) ForceDelete(part);
            r.detail = err;
            r.error = r.cancelled ? L"The OptiScaler download was cancelled."
                      : status == 0 ? L"WrongGPU couldn't download its OptiScaler edition. Check your internet connection."
                                    : L"WrongGPU couldn't download its OptiScaler edition (the release page answered " + std::to_wstring(status) + L").";
            LogLine(ctx, L"OptiScaler: download failed: " + err);
            return r;
        }
        const wstring sha = Sha256File(part);
        if (!IEq(sha, e.sha256)) {
            ForceDelete(part);
            r.mismatch = true;
            r.error = L"The OptiScaler download wasn't the file WrongGPU expects, so it wasn't used.";
            r.detail = L"sha256 " + sha + L", expected " + e.sha256;
            LogLine(ctx, L"OptiScaler: download refused: " + r.detail);
            return r;
        }
        if (!MoveFileExW(part.c_str(), final_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            r.detail = L"cannot rename " + part + L": " + ErrText(GetLastError());
            ForceDelete(part);
            r.error = L"WrongGPU couldn't save its OptiScaler edition in its data folder.";
            LogLine(ctx, L"OptiScaler: " + r.detail);
            return r;
        }
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
