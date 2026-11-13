// SPDX-License-Identifier: MIT
// WrongGPU launcher core: Ray Reconstruction's kernels prepared by the launcher (rrprep.h). Written for this project.
// Windows' own DXGI (loaded at run time), process and pipe functions only.
#include "rrprep.h"

#include "gamerules.h"

#include <dxgi1_6.h>
#include <shlobj.h>

#include <cstdio>
#include <sstream>

namespace wg {

namespace {

const wchar_t* const kRunTime = L"wronggpu_runtime.dll";

wstring PrepDir(const wstring& dataDir) { return Join(dataDir, L"rr-prepare"); }
wstring PrepFile(const wstring& dataDir) { return Join(PrepDir(dataDir), L"prepared.txt"); }

struct PrepLine { wstring name, driver, sha; int passes = 0; };

std::vector<PrepLine> ReadPrepared(const wstring& dataDir) {
    std::vector<PrepLine> out;
    string text;
    if (!ReadAll(PrepFile(dataDir), text)) return out;
    std::istringstream in(text);
    string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<string> f;
        for (size_t a = 0;;) {
            const size_t b = line.find('\t', a);
            f.push_back(line.substr(a, b == string::npos ? string::npos : b - a));
            if (b == string::npos) break;
            a = b + 1;
        }
        if (f.size() != 4) continue;
        PrepLine p;
        p.name = FromUtf8(f[0]);
        p.driver = FromUtf8(f[1]);
        p.sha = FromUtf8(f[2]);
        p.passes = atoi(f[3].c_str());
        out.push_back(p);
    }
    return out;
}

void WritePrepared(const wstring& dataDir, const PrepLine& now) {
    std::vector<PrepLine> all = ReadPrepared(dataDir);
    bool found = false;
    for (auto& p : all)
        if (IEq(p.name, now.name)) { p = now; found = true; }
    if (!found) all.push_back(now);
    string text;
    for (auto& p : all) text += ToUtf8(Lower(p.name)) + "\t" + ToUtf8(p.driver) + "\t" + ToUtf8(p.sha) + "\t" + std::to_string(p.passes) + "\n";
    EnsureDir(PrepDir(dataDir));
    WriteFileAtomic(PrepFile(dataDir), text.data(), text.size());
}

// the environment for the helper: this one, with LOCALAPPDATA set to the user's real local app data folder (the driver
// keeps its shader cache there; a launcher started with another LOCALAPPDATA, as the tests' sandbox does, must still
// prepare the cache the game finds) and the runtime's log in the launcher's data folder (never a new file in the game)
std::vector<wchar_t> HelperEnvironment(const wstring& log) {
    wstring local;
    PWSTR p = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_DONT_VERIFY, nullptr, &p)) && p) local = p;
    if (p) CoTaskMemFree(p);
    std::vector<wchar_t> env;
    LPWCH block = GetEnvironmentStringsW();
    for (LPWCH s = block; s && *s; s += wcslen(s) + 1) {
        const wstring kv = s;
        if ((!local.empty() && StartsWithI(kv, L"LOCALAPPDATA=")) || StartsWithI(kv, L"WG_RR_LOG=") || StartsWithI(kv, L"WG_RR_MODE=")) continue;
        env.insert(env.end(), kv.begin(), kv.end());
        env.push_back(0);
    }
    if (block) FreeEnvironmentStringsW(block);
    for (const wstring& kv : {local.empty() ? wstring() : L"LOCALAPPDATA=" + local, L"WG_RR_LOG=" + log}) {
        if (kv.empty()) continue;
        env.insert(env.end(), kv.begin(), kv.end());
        env.push_back(0);
    }
    env.push_back(0);
    return env;
}

}  // namespace

wstring RrDriverVersion() {
    HMODULE m = LoadSystemDll(L"dxgi.dll");
    if (!m) return L"";
    typedef HRESULT(WINAPI * Create)(REFIID, void**);
    auto create = (Create)GetProcAddress(m, "CreateDXGIFactory1");
    IDXGIFactory1* f = nullptr;
    if (!create || FAILED(create(__uuidof(IDXGIFactory1), (void**)&f)) || !f) return L"";
    wstring out;
    IDXGIAdapter1* a = nullptr;
    for (UINT i = 0; out.empty() && f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        LARGE_INTEGER v = {};
        if (d.VendorId == 0x1002 && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(a->CheckInterfaceSupport(__uuidof(IDXGIDevice), &v))) {
            wchar_t b[64];
            swprintf(b, 64, L"%u.%u.%u.%u", (unsigned)HIWORD(v.HighPart), (unsigned)LOWORD(v.HighPart), (unsigned)HIWORD(v.LowPart), (unsigned)LOWORD(v.LowPart));
            out = b;
        }
        a->Release();
    }
    f->Release();
    return out;
}

wstring RrRuntimeSha(Context& ctx) {
    if (!ctx.data) return L"";
    for (size_t i = 0; i < ctx.data->count; ++i)
        if (IEq(ctx.data->files[i].rel, kRunTime)) return ctx.data->files[i].sha256;
    return L"";
}

int RrPassesFor(const string& mode) { return mode.empty() || mode == "accurate" ? 1 : 2; }

bool RrKernelsInGame(const wstring& exe) {
    const GameRule* r = GameRuleFor(exe);
    return r && r->rrKernelsInGame;
}

const wchar_t* RrPrepStateName(RrPrepState s) {
    switch (s) {
        case RrPrepState::Ready: return L"ready";
        case RrPrepState::NotPrepared: return L"not-prepared";
        case RrPrepState::DriverChanged: return L"driver-changed";
        case RrPrepState::RuntimeChanged: return L"runtime-changed";
        case RrPrepState::MorePasses: return L"more-passes";
        case RrPrepState::InGame: return L"in-game";
    }
    return L"?";
}

RrPrepState RrPrepStateOf(Context& ctx, const wstring& dataDir, const wstring& exe, int passes, wstring* why) {
    if (RrKernelsInGame(exe)) {
        if (why) *why = L"the game prepares them at its first start (its graphics driver cache is its own)";
        return RrPrepState::InGame;
    }
    const wstring name = BaseName(exe), driver = RrDriverVersion(), sha = RrRuntimeSha(ctx);
    for (auto& p : ReadPrepared(dataDir)) {
        if (!IEq(p.name, name)) continue;
        if (p.driver != driver) { if (why) *why = L"the graphics driver changed since (" + p.driver + L", now " + driver + L")"; return RrPrepState::DriverChanged; }
        if (!IEq(p.sha, sha)) { if (why) *why = L"WrongGPU's runtime changed since"; return RrPrepState::RuntimeChanged; }
        if (p.passes < passes) { if (why) *why = L"prepared for the default Ray Reconstruction quality only"; return RrPrepState::MorePasses; }
        return RrPrepState::Ready;
    }
    if (why) *why = L"not prepared yet";
    return RrPrepState::NotPrepared;
}

bool RrPrepared(Context& ctx, const wstring& dataDir, const wstring& exe, int passes, wstring* why) {
    return RrPrepStateOf(ctx, dataDir, exe, passes, why) == RrPrepState::Ready;
}

int PrepareRr(Context& ctx, const wstring& dataDir, const wstring& exe, const wstring& runtimeDll, int passes,
              const std::function<bool(const RrPrepProgress&)>& progress, wstring* why) {
    if (RrKernelsInGame(exe)) {
        if (why) *why = L"the game prepares them at its first start (its graphics driver cache is its own)";
        LogLine(ctx, L"Ray Reconstruction: " + BaseName(exe) + L" prepares its kernels at its first start (the graphics driver keeps the game's shader cache under the game's own "
                     L"application name, which the launcher's helper can't reach); nothing prepared ahead");
        return 4;
    }
    const wstring driver = RrDriverVersion();
    if (driver.empty()) { if (why) *why = L"no AMD graphics card found"; return 2; }
    if (!IsFile(runtimeDll)) { if (why) *why = L"the runtime isn't there (" + runtimeDll + L")"; return 3; }
    const wstring run = Join(PrepDir(dataDir), L"run");
    EnsureDir(run);
    const wstring helper = Join(run, BaseName(exe));
    if (!CopyFileW(SelfPath().c_str(), helper.c_str(), FALSE)) {
        if (why) *why = L"the helper could not be written (" + ErrText(GetLastError()) + L")";
        return 3;
    }
    const wstring log = Join(PrepDir(dataDir), L"prepare.log");
    LogLine(ctx, L"Ray Reconstruction: preparing the kernels for " + BaseName(exe) + L" (driver " + driver + L", " + std::to_wstring(passes) +
                     L" pass(es)); the helper runs as " + helper + L", runtime log " + log);
    SECURITY_ATTRIBUTES sa = {sizeof sa, nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { DeleteFileW(helper.c_str()); if (why) *why = L"no pipe"; return 3; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si = {sizeof si};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi = {};
    wstring cmd = L"\"" + helper + L"\" --rr-prepare \"" + runtimeDll + L"\" " + std::to_wstring(passes);
    std::vector<wchar_t> cl(cmd.begin(), cmd.end());
    cl.push_back(0);
    std::vector<wchar_t> env = HelperEnvironment(log);
    const BOOL ok = CreateProcessW(helper.c_str(), cl.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | BELOW_NORMAL_PRIORITY_CLASS,
                                   env.data(), run.c_str(), &si, &pi);
    CloseHandle(wr);
    if (!ok) {
        const DWORD e = GetLastError();
        CloseHandle(rd);
        DeleteFileW(helper.c_str());
        if (why) *why = L"the helper did not start (" + ErrText(e) + L")";
        return 3;
    }
    const ULONGLONG t0 = GetTickCount64();
    RrPrepProgress st;
    string buf;
    bool cancelled = false;
    for (;;) {
        DWORD avail = 0;
        while (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail) {
            char tmp[4096];
            DWORD got = 0;
            if (!ReadFile(rd, tmp, (DWORD)std::min<DWORD>(avail, sizeof tmp), &got, nullptr) || !got) break;
            buf.append(tmp, got);
            for (size_t nl; (nl = buf.find('\n')) != string::npos; buf.erase(0, nl + 1)) {
                const string line = buf.substr(0, nl);
                unsigned d = 0, t = 0, ps = 0;
                char k[128] = {};
                if (sscanf_s(line.c_str(), "progress %u %u %u %127s", &d, &t, &ps, k, (unsigned)sizeof k) >= 3) {
                    st.done = d; st.total = t; st.pass = ps; st.kernel = FromUtf8(k);
                }
            }
        }
        if (!cancelled && progress && !progress(st)) {
            cancelled = true;
            TerminateProcess(pi.hProcess, 1);
        }
        if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) {
            DWORD avail2 = 0;
            if (!(PeekNamedPipe(rd, nullptr, 0, nullptr, &avail2, nullptr) && avail2)) break;
        }
    }
    DWORD code = 3;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    CloseHandle(rd);
    for (int i = 0; i < 20 && IsFile(helper) && !DeleteFileW(helper.c_str()); ++i) Sleep(50);
    const ULONGLONG ms = GetTickCount64() - t0;
    if (cancelled) {
        LogLine(ctx, L"Ray Reconstruction: preparing cancelled after " + std::to_wstring(ms / 1000) + L" s (" + std::to_wstring(st.done) + L" of " +
                         std::to_wstring(st.total) + L" kernels; the driver keeps them, the next run goes on from there)");
        return 1;
    }
    if (code == 0) {
        PrepLine p;
        p.name = BaseName(exe);
        p.driver = driver;
        p.sha = RrRuntimeSha(ctx);
        p.passes = passes;
        WritePrepared(dataDir, p);
        LogLine(ctx, L"Ray Reconstruction: prepared for " + p.name + L" in " + std::to_wstring(ms / 1000) + L"." + std::to_wstring(ms % 1000 / 100) + L" s (" +
                         std::to_wstring(st.done) + L" kernels)");
        return 0;
    }
    if (code == 2) { if (why) *why = L"no Radeon RX 9000 series graphics card (Ray Reconstruction needs one)"; return 2; }
    if (why) *why = L"the helper ended with " + std::to_wstring(code) + L" (details: " + log + L")";
    LogLine(ctx, L"Ray Reconstruction: preparing failed: " + (why ? *why : wstring()));
    return 3;
}

int RunRrPrepareHelper(int argc, wchar_t** argv) {
    if (argc < 4) return 9;
    HMODULE m = LoadLibraryExW(argv[2], nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m) return 8;
    typedef int (*Cb)(void*, uint32_t, uint32_t, uint32_t, const char*, uint32_t, uint32_t);
    typedef uint32_t (*Prep)(uint32_t, Cb, void*);
    auto prep = (Prep)GetProcAddress(m, "WrongGPU_RrPrepareKernelsV1");
    if (!prep) return 7;
    const Cb cb = [](void*, uint32_t done, uint32_t total, uint32_t pass, const char* kernel, uint32_t ms, uint32_t) -> int {
        char line[256];
        const int n = snprintf(line, sizeof line, "progress %u %u %u %s %u\n", done, total, pass, kernel ? kernel : "-", ms);
        DWORD w = 0;
        if (n > 0) WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), line, (DWORD)n, &w, nullptr);
        return 0;
    };
    return (int)prep((uint32_t)_wtoi(argv[3]), cb, nullptr);
}

}  // namespace wg
