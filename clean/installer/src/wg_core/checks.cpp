// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the requirement checks (spec 30 "Requirements check"): Windows, the GPU, the HIP runtime
// (with its child-process probe), anti-cheat, a running game, write access. Written for this project.
#include "checks.h"
#include <tlhelp32.h>
#include <dxgi.h>
#include "optiscaler.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// System requirements
// ------------------------------------------------------------------------------------------------
WinInfo WindowsInfo() {
    using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
    auto fn = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion")));
    OSVERSIONINFOW v{}; v.dwOSVersionInfoSize = sizeof(v);
    WinInfo w;
    if (!fn || fn(&v) != 0) { w.text = L"unknown"; return w; }
    wchar_t b[64]; swprintf(b, 64, L"%lu.%lu.%lu", v.dwMajorVersion, v.dwMinorVersion, v.dwBuildNumber);
    w.text = b;
    DWORD val = 0, sz = sizeof(val);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"UBR", RRF_RT_REG_DWORD, nullptr, &val, &sz) == ERROR_SUCCESS) w.text += L"." + std::to_wstring(val);
    // This installer is a 64-bit program, so it only runs on 64-bit Windows; Windows 10 and 11 report 10.0.
    w.ok = v.dwMajorVersion >= 10 && v.dwBuildNumber >= 10240;
    w.text = (v.dwBuildNumber >= 22000 ? L"Windows 11 (" : v.dwMajorVersion >= 10 ? L"Windows 10 (" : L"Windows (") + w.text + L", 64-bit)";
    return w;
}

bool IsRdna4Name(const wstring& name) { return Contains(name, L"RX 90") || Contains(name, L"PRO R97") || Contains(name, L"PRO R96"); }
GpuInfo DetectGpu() {
    GpuInfo best;
    HMODULE dxgi = LoadSystemDll(L"dxgi.dll");
    if (!dxgi) return best;
    using CreateFn = HRESULT(WINAPI*)(REFIID, void**);
    auto create = reinterpret_cast<CreateFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    IDXGIFactory1* f = nullptr;
    if (!create || FAILED(create(__uuidof(IDXGIFactory1), (void**)&f)) || !f) return best;
    IDXGIAdapter1* a = nullptr;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        LARGE_INTEGER umd{};
        wstring driver;
        if (SUCCEEDED(a->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd))) {
            wchar_t b[64];
            swprintf(b, 64, L"%u.%u.%u.%u", HIWORD(umd.HighPart), LOWORD(umd.HighPart), HIWORD(umd.LowPart), LOWORD(umd.LowPart));
            driver = b;
        }
        a->Release();
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        wstring name = d.Description;
        if (best.name.empty()) { best.name = name; best.driver = driver; }
        if (d.VendorId != 0x1002) continue;
        bool rdna4 = IsRdna4Name(name);
        if (!best.amd || (rdna4 && !best.rdna4)) { best.amd = true; best.rdna4 = rdna4; best.name = name; best.driver = driver; }
    }
    f->Release();
    return best;
}

// The AMD HIP runtime (amdhip64_7.dll) the clean runtime loads (spec 30 answer 7): from the system search path first
// (System32, where the AMD Software: Adrenalin Edition driver installs it; then the PATH folders), then the newest
// C:\Program Files\AMD\ROCm\<version>\bin (HIP SDK), then HIP_PATH. WRONGGPU_HIP_RUNTIME overrides (development; read
// only by a WRONGGPU_TESTING build).
const wchar_t* const kHipDll = L"amdhip64_7.dll";
const wchar_t* const kDriverPage = L"https://www.amd.com/en/support/download/drivers.html";
const char* const kSupportedArch[] = {"gfx1201", "gfx1200"};  // RDNA 4: the RX 9070 family and the RX 9060 family
// Folder names like "7.2" or "6.4.1", and release versions ("0.6.1", "v0.6.1-rc1"), compared as versions: the numbers of
// the version (digits and dots after any leading letters such as "v") first; at equal numbers a pre-release (a suffix
// such as "-rc1", "-beta2" or "rc1") comes before the release itself, two pre-releases by their words ("beta" < "rc"), then
// by their numbers ("rc2" < "rc10"). A build suffix ("+abc") is no pre-release and doesn't count.
bool VersionLess(const wstring& a, const wstring& b) {
    auto parts = [](const wstring& s) { std::vector<long> v; long cur = -1; for (wchar_t c : s) { if (iswdigit(c)) cur = (cur < 0 ? 0 : cur * 10) + (c - L'0'); else if (cur >= 0) { v.push_back(cur); cur = -1; } } if (cur >= 0) v.push_back(cur); return v; };
    struct Split { wstring core, pre; };
    auto split = [](const wstring& s) {
        size_t i = 0;
        while (i < s.size() && !iswdigit(s[i])) ++i;            // "v0.6.1": the leading letters
        size_t j = i;
        while (j < s.size() && (iswdigit(s[j]) || s[j] == L'.')) ++j;
        Split r{s.substr(i, j - i), s.substr(j)};
        while (!r.pre.empty() && (r.pre.back() == L' ' || r.pre.back() == L'\t' || r.pre.back() == L'\r' || r.pre.back() == L'\n')) r.pre.pop_back();
        if (!r.pre.empty() && r.pre[0] == L'+') r.pre.clear();  // build metadata
        if (r.core.empty()) r = {s, L""};                       // no version number at all: the old reading of every digit group
        return r;
    };
    const Split x = split(a), y = split(b);
    const auto px = parts(x.core), py = parts(y.core);
    if (px != py) return px < py;
    if (x.pre.empty() || y.pre.empty()) return !x.pre.empty() && y.pre.empty();  // "0.6.1-rc1" < "0.6.1"
    auto letters = [](const wstring& s) { wstring t; for (wchar_t c : s) if (!iswdigit(c)) t += c; return t; };
    if (const int c = _wcsicmp(letters(x.pre).c_str(), letters(y.pre).c_str())) return c < 0;  // "-beta" < "-rc"
    return parts(x.pre) < parts(y.pre);
}
std::vector<wstring> HipCandidates() {
    std::vector<wstring> c;
#ifdef WRONGGPU_TESTING
    const wstring mocks = EnvVar(L"WRONGGPU_TEST_HIP_CANDIDATES");
    if (!mocks.empty()) {
        for (size_t i = 0; i < mocks.size();) {
            const size_t end = mocks.find(L';', i);
            c.push_back(mocks.substr(i, end == wstring::npos ? end : end - i));
            i = end == wstring::npos ? mocks.size() : end + 1;
        }
        return c;
    }
    // 0.6.1: the development override only in a WRONGGPU_TESTING build; a release build never reads it (a developer's
    // mock HIP library set at install time was recorded in hip-runtime.txt)
    const wstring env = EnvVar(L"WRONGGPU_HIP_RUNTIME");
    if (!env.empty()) c.push_back(env);
#endif
    wchar_t sys[MAX_PATH];
    if (GetSystemDirectoryW(sys, MAX_PATH)) c.push_back(Join(sys, kHipDll));
    wstring path = EnvVar(L"PATH");
    if (path.empty()) {  // PATH can be longer than EnvVar's buffer
        DWORD n = GetEnvironmentVariableW(L"PATH", nullptr, 0);
        if (n) { path.resize(n); path.resize(GetEnvironmentVariableW(L"PATH", path.data(), n)); }
    }
    for (size_t i = 0; i <= path.size();) {
        size_t j = path.find(L';', i);
        if (j == wstring::npos) j = path.size();
        wstring d = path.substr(i, j - i);
        if (d.size() > 2 && d[1] == L':') c.push_back(Join(d, kHipDll));  // absolute folders only
        i = j + 1;
    }
    auto dirs = ListDirs(L"C:\\Program Files\\AMD\\ROCm");
    std::sort(dirs.begin(), dirs.end(), [](const wstring& a, const wstring& b) { return VersionLess(BaseName(b), BaseName(a)); });  // newest first
    for (auto& d : dirs) c.push_back(Join(d, wstring(L"bin\\") + kHipDll));
    wstring hipPath = EnvVar(L"HIP_PATH");
    if (!hipPath.empty()) c.push_back(Join(Join(hipPath, L"bin"), kHipDll));
    std::vector<wstring> out;
    std::set<wstring> seen;
    for (auto& p : c) if (IsFile(p) && seen.insert(Lower(FullPath(p))).second) out.push_back(FullPath(p));
    return out;
}
// The architecture name ("gfx1201", ...) in a HIP device-properties structure. The structure's layout differs between
// HIP versions, so instead of fixed offsets the buffer is searched for the gcnArchName text: "gfx" followed by digits.
string FindArchName(const uint8_t* p, size_t n) {
    for (size_t i = 0; i + 4 < n; ++i) {
        if (p[i] != 'g' || p[i + 1] != 'f' || p[i + 2] != 'x' || !isdigit(p[i + 3]) || (i > 0 && isalnum(p[i - 1]))) continue;
        string a = "gfx";
        for (size_t k = i + 3; k < n && isalnum(p[k]) && a.size() < 16; ++k) a += (char)p[k];
        return a;
    }
    return "";
}
bool IsSupportedArch(const string& a) { for (auto s : kSupportedArch) if (a == s) return true; return false; }
// Child-process probe (/HIPPROBE=<dll>): load the HIP runtime, read each device's name and architecture, and create a
// context on a supported device (public HIP API: hipInit, hipGetDeviceCount, hipDeviceGetName,
// hipGetDevicePropertiesR0600 / hipGetDeviceProperties, hipSetDevice, hipMalloc, hipFree). Running it in a separate
// process keeps a misbehaving driver from taking the installer down. Lines "device<TAB>name<TAB>arch" and
// "use<TAB>name<TAB>arch" go to standard output. Exit: 0 ok, 11 cannot load, 12 hipInit failed, 13 no device,
// 14 no RX 9000-series device, 15 no context, 16 only RX 9000-series devices that are not supported yet (other gfx12).
int HipProbe(Context& ctx, const wstring& dll) {
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    auto emit = [&](const string& line) { if (out && out != INVALID_HANDLE_VALUE) { DWORD w; WriteFile(out, line.data(), (DWORD)line.size(), &w, nullptr); } };
    HMODULE h = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) { LogLine(ctx, L"HIP probe: cannot load " + dll + L": " + ErrText(GetLastError())); return 11; }
    using InitFn = int (*)(unsigned);
    using CountFn = int (*)(int*);
    using NameFn = int (*)(char*, int, int);
    using PropsFn = int (*)(void*, int);
    using SetFn = int (*)(int);
    using MallocFn = int (*)(void**, size_t);
    using FreeFn = int (*)(void*);
    auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(h, n)); };
    auto init = reinterpret_cast<InitFn>(sym("hipInit"));
    auto count = reinterpret_cast<CountFn>(sym("hipGetDeviceCount"));
    auto name = reinterpret_cast<NameFn>(sym("hipDeviceGetName"));
    auto props = reinterpret_cast<PropsFn>(sym("hipGetDevicePropertiesR0600"));
    if (!props) props = reinterpret_cast<PropsFn>(sym("hipGetDeviceProperties"));
    auto set = reinterpret_cast<SetFn>(sym("hipSetDevice"));
    auto alloc = reinterpret_cast<MallocFn>(sym("hipMalloc"));
    auto release = reinterpret_cast<FreeFn>(sym("hipFree"));
    if (!init || !count || !name || !props || !set || !alloc || !release) { LogLine(ctx, L"HIP probe: missing HIP functions in " + dll); return 11; }
    if (init(0) != 0) { LogLine(ctx, L"HIP probe: hipInit failed"); return 12; }
    int n = 0;
    if (count(&n) != 0 || n <= 0) { LogLine(ctx, L"HIP probe: no HIP device"); return 13; }
    bool rdna4 = false;
    std::vector<uint8_t> buf(64 * 1024);  // far larger than any hipDeviceProp_t
    for (int i = 0; i < n; ++i) {
        char nm[256] = {};
        if (name(nm, (int)sizeof(nm) - 1, i) != 0) continue;
        std::fill(buf.begin(), buf.end(), uint8_t(0));
        const string arch = props(buf.data(), i) == 0 ? FindArchName(buf.data(), buf.size()) : "";
        LogLine(ctx, L"HIP probe: device " + std::to_wstring(i) + L": " + FromUtf8(nm) + L" (" + (arch.empty() ? wstring(L"architecture unknown") : FromUtf8(arch)) + L")");
        emit(string("device\t") + nm + "\t" + arch + "\n");
        if (arch.rfind("gfx12", 0) == 0) rdna4 = true;
        if (!IsSupportedArch(arch)) continue;
        void* p = nullptr;
        if (set(i) == 0 && alloc(&p, 256) == 0 && p) {
            release(p);
            LogLine(ctx, L"HIP probe: context created on " + FromUtf8(nm));
            emit(string("use\t") + nm + "\t" + arch + "\n");
            return 0;
        }
        LogLine(ctx, L"HIP probe: cannot create a context on " + FromUtf8(nm));
        return 15;
    }
    return rdna4 ? 16 : 14;
}
static std::mutex probeJobLock;
static HANDLE probeJob = nullptr;
static HANDLE ProbeJob() {  // made once, kept for the process's life (its handle closing at exit ends what is in it)
    std::lock_guard<std::mutex> lock(probeJobLock);
    if (!probeJob) {
        probeJob = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (probeJob && !SetInformationJobObject(probeJob, JobObjectExtendedLimitInformation, &li, sizeof(li))) { CloseHandle(probeJob); probeJob = nullptr; }
    }
    return probeJob;
}
void StopHipProbes() {
    std::lock_guard<std::mutex> lock(probeJobLock);
    if (!probeJob) return;
    TerminateJobObject(probeJob, 97);
    // until no process is left in it (at most a second: a probe stuck in the driver goes once its call returns)
    for (int i = 0; i < 50; ++i) {
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION ai{};
        if (!QueryInformationJobObject(probeJob, JobObjectBasicAccountingInformation, &ai, sizeof(ai), nullptr) || ai.ActiveProcesses == 0) break;
        Sleep(20);
    }
}
HipInfo CheckHip(Context& ctx) {
    HipInfo r;
    r.ran = true;
    auto cands = HipCandidates();
    if (cands.empty()) { r.text = L"AMD HIP runtime (amdhip64_7.dll) not found"; return r; }
    // The child's environment without LLVM_PATH (a stray LLVM_PATH can make HIP's module loading crash).
    std::vector<wchar_t> envBlock;
    if (LPWCH env = GetEnvironmentStringsW()) {
        for (LPWCH p = env; *p; p += wcslen(p) + 1)
            if (!StartsWithI(p, L"LLVM_PATH=")) envBlock.insert(envBlock.end(), p, p + wcslen(p) + 1);
        FreeEnvironmentStringsW(env);
    }
    envBlock.push_back(0);
    if (envBlock.size() == 1) envBlock.push_back(0);
    for (auto& dll : cands) {
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        HANDLE rd = nullptr, wr = nullptr;
        if (!CreatePipe(&rd, &wr, &sa, 0)) continue;
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
        wstring cmd = L"\"" + SelfPath() + L"\" \"/HIPPROBE=" + dll + L"\" \"/LOG=" + ctx.logFile + L"\"";
        STARTUPINFOW si{}; si.cb = sizeof(si); si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = wr; si.hStdError = wr; si.hStdInput = nullptr;
        PROCESS_INFORMATION pi{};
        std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
        DWORD code = 99;
        string output;
        if (CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED, envBlock.data(), nullptr, &si,
                           &pi)) {
            // in the probes' job before it runs a line (it ends with this process; StopHipProbes ends it at once)
            if (HANDLE job = ProbeJob()) AssignProcessToJobObject(job, pi.hProcess);
            ResumeThread(pi.hThread);
            CloseHandle(wr); wr = nullptr;
            const ULONGLONG deadline = GetTickCount64() + 60000;
            for (;;) {
                if (GetTickCount64() >= deadline) {
                    TerminateProcess(pi.hProcess, 98);
                    WaitForSingleObject(pi.hProcess, 1000);
                    code = 98;
                    break;
                }
                char chunk[512]; DWORD available = 0, got = 0;
                if (PeekNamedPipe(rd, nullptr, 0, nullptr, &available, nullptr) && available &&
                    ReadFile(rd, chunk, (DWORD)std::min<size_t>(sizeof(chunk), available), &got, nullptr) && got) {
                    if (output.size() < 65536) output.append(chunk, got);
                    continue;
                }
                if (WaitForSingleObject(pi.hProcess, 20) == WAIT_OBJECT_0) {
                    // The child can write its final device/use lines while we wait for exit.
                    while (PeekNamedPipe(rd, nullptr, 0, nullptr, &available, nullptr) && available &&
                           ReadFile(rd, chunk, (DWORD)std::min<size_t>(sizeof(chunk), available), &got, nullptr) && got)
                        if (output.size() < 65536) output.append(chunk, got);
                    GetExitCodeProcess(pi.hProcess, &code);
                    break;
                }
            }
            CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
        }
        if (wr) CloseHandle(wr);
        CloseHandle(rd);
        r.dll = dll; r.code = (int)code;
        r.device.clear(); r.arch.clear();  // each candidate must report its own device
        wstring devices;
        for (size_t p = 0; p < output.size();) {
            size_t e = output.find('\n', p);
            string line = output.substr(p, e == string::npos ? string::npos : e - p);
            p = e == string::npos ? output.size() : e + 1;
            size_t t1 = line.find('\t'), t2 = t1 == string::npos ? string::npos : line.find('\t', t1 + 1);
            if (t2 == string::npos) continue;
            const wstring kind = FromUtf8(line.substr(0, t1)), nm = FromUtf8(line.substr(t1 + 1, t2 - t1 - 1)), ar = FromUtf8(line.substr(t2 + 1));
            if (kind == L"use" || (kind == L"device" && r.device.empty())) { r.device = nm; r.arch = ar; }
            if (kind == L"device") devices += (devices.empty() ? L"" : L", ") + nm + L" (" + (ar.empty() ? wstring(L"?") : ar) + L")";
        }
        LogLine(ctx, L"HIP check: " + dll + L" -> probe code " + std::to_wstring(code) + (devices.empty() ? L"" : L"; devices: " + devices));
        if (code == 0) { r.ok = true; r.text = L"AMD HIP runtime works (" + dll + L")"; return r; }
        if (code == 14 || code == 16) {
            r.text = L"HIP could not validate a supported architecture using " + dll + L": " + (devices.empty() ? wstring(L"no architecture reported") : devices);
            continue;  // another installed runtime may recognise this GPU
        }
        r.text = L"AMD HIP runtime " + dll + L" does not work (probe code " + std::to_wstring(code) + L")";
    }
    return r;
}

PlayWatch WatchPlayed(unsigned long long sinceMs, bool seenRunning, bool runningNow) {
    if (runningNow) return PlayWatch::Running;
    if (sinceMs < kPlayGraceMs) return PlayWatch::Starting;
    if (!seenRunning && sinceMs < kPlayStartWaitMs) return PlayWatch::Starting;
    return PlayWatch::Closed;
}

bool GameRunning(const wstring& exe) {
    if (exe.empty()) return false;
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (s == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W pe{}; pe.dwSize = sizeof(pe);
    bool running = false;
    for (BOOL ok = Process32FirstW(s, &pe); ok && !running; ok = Process32NextW(s, &pe)) {
        if (!IEq(pe.szExeFile, BaseName(exe))) continue;
        HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
        if (!p) { running = true; continue; }  // cannot tell: assume it is this game
        wchar_t path[MAX_PATH * 2]; DWORD n = (DWORD)std::size(path);
        if (!QueryFullProcessImageNameW(p, 0, path, &n) || IEq(path, exe)) running = true;
        CloseHandle(p);
    }
    CloseHandle(s);
    return running;
}
bool IsElevated() {
    HANDLE t = nullptr; TOKEN_ELEVATION e{}; DWORD n = 0;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &t)) return false;
    bool r = GetTokenInformation(t, TokenElevation, &e, sizeof(e), &n) && e.TokenIsElevated;
    CloseHandle(t);
    return r;
}
// 0 = writable, otherwise the Win32 error.
DWORD WriteTest(const wstring& dir) {
    wstring p = Join(dir, L".wronggpu-write-test");
    HANDLE h = CreateFileW(p.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (h == INVALID_HANDLE_VALUE) return GetLastError();
    CloseHandle(h);
    return 0;
}

// ------------------------------------------------------------------------------------------------
// Anti-cheat: markers in the game's own folders only (never in a library folder that holds other games)
// ------------------------------------------------------------------------------------------------
static bool IsVolumeRoot(const wstring& d) {
    if (StartsWithI(d, L"\\\\?\\")) return IsVolumeRoot(d.substr(4));
    if (d.size() <= 3 || d.find_first_of(L"\\/") == wstring::npos) return true;  // "C:", "C:\", ""
    if ((d[0] == L'\\' || d[0] == L'/') && (d[1] == L'\\' || d[1] == L'/')) {     // \\server\share
        size_t n = 0;
        for (size_t i = 2; i < d.size(); ++i) n += d[i] == L'\\' || d[i] == L'/';
        return n <= 1;
    }
    return false;
}
// A folder that holds games (a store library or a games folder), or a drive root.
bool LibraryContainer(const wstring& dir) {
    if (IsVolumeRoot(dir)) return true;
    const wstring n = Lower(BaseName(dir)), p = Lower(BaseName(Parent(dir)));
    static const wchar_t* const names[] = {L"steamapps", L"steamlibrary", L"epic games", L"gog games", L"xboxgames", L"program files",
                                           L"program files (x86)", L"games", L"my games", L"steam games", L"rockstar games", L"ea games",
                                           L"origin games", L"ubisoft game launcher"};
    for (auto s : names) if (n == s) return true;  // ("games" covers GOG Galaxy\Games and Ubisoft Game Launcher\games)
    if (n == L"common" && p == L"steamapps") return true;
    if (n == L"library" && p == L"amazon games") return true;
    return IsDir(Join(dir, L"steamapps")) || IsDir(Join(dir, L"SteamLibrary"));
}
// The folders a game keeps its executable in below its own folder (bin\x64, bin\x64_dx12, Bin64, Binaries\Win64, ...).
static bool BinaryFolder(const wstring& name) {
    const wstring n = Lower(name);
    static const wchar_t* const exact[] = {L"bin", L"bin32", L"bin64", L"binaries", L"x64", L"x86", L"win64", L"win32", L"wingdk", L"win_x64",
                                           L"win_x86", L"retail", L"shipping"};
    for (auto e : exact) if (n == e) return true;
    for (auto p : {L"bin_", L"x64_", L"x86_", L"win64_", L"win32_"}) if (n.rfind(p, 0) == 0) return true;
    return false;
}
// The game's own top folder: up from the exe's folder through its binaries folders (at most three levels; Unreal:
// <root>\<Project>\Binaries\Win64 -> <root> when <root> has Engine\ or the bootstrapper exe, else <Project>), and
// never past the folder directly below a library container. An exe that sits in a container itself: that folder.
wstring GameTopFolder(const wstring& exeDir) {
    if (LibraryContainer(exeDir)) return exeDir;
    wstring top = exeDir;
    const wstring leaf = Lower(BaseName(exeDir)), up = Lower(BaseName(Parent(exeDir)));
    if ((leaf == L"win64" || leaf == L"wingdk") && up == L"binaries") {
        const wstring project = Parent(Parent(exeDir)), root = Parent(project);
        top = !LibraryContainer(root) && (IsDir(Join(root, L"Engine")) || !ListFiles(root, L"*.exe").empty()) ? root : project;
    } else {
        for (int i = 0; i < 3 && BinaryFolder(BaseName(top)); ++i) top = Parent(top);
    }
    for (wstring d = exeDir;; d = Parent(d)) {
        if (IEq(d, top) || LibraryContainer(Parent(d))) return d;
    }
}
wstring AntiCheatMessage(const wstring& found) {
    return L"Anti-cheat found: " + found + L" - games with anti-cheat are never supported";
}

// Games with built-in anti-cheat by name (from the 0.2.0 installer); anti-cheat markers in the exe's folder and each
// folder up to the game's top folder (GameTopFolder), plus two levels below the top folder and below the store layout's
// game root when that lies inside it. Returns the marker's full path (or the denied game's reason), empty for none.
wstring FindAntiCheat(const wstring& exe) {
    const wstring exeDir = Parent(exe), root = GameRoot(exeDir), top = GameTopFolder(exeDir);
    for (auto& d : kDenied) {
        if (d.exe && IEq(BaseName(exe), d.exe)) return d.why;
        if (d.folder && Contains(BaseName(root), d.folder)) return d.why;
    }
    for (wstring dir = exeDir; !dir.empty(); dir = Parent(dir)) {
        for (auto m : kAntiCheat) if (Exists(Join(dir, m))) return Join(dir, m);
        if (IEq(dir, top)) break;
    }
    if (LibraryContainer(top)) return L"";  // the exe sits in a library folder: the other games below it are not this game's
    std::vector<wstring> bases{top};
    if (!IEq(root, top) && StartsWithI(root, top + L"\\")) bases.push_back(root);
    for (const wstring& base : bases) {
        std::vector<std::pair<wstring, int>> stack{{base, 0}};
        while (!stack.empty()) {
            auto [d, depth] = stack.back(); stack.pop_back();
            for (auto& sub : ListDirs(d)) {
                for (auto m : kAntiCheat) if (IEq(BaseName(sub), m)) return sub;
                if (depth < 1) stack.push_back({sub, depth + 1});
            }
        }
    }
    return L"";
}

// ------------------------------------------------------------------------------------------------
// Requirement checks (spec 30 "Requirements check", before anything is written)
// ------------------------------------------------------------------------------------------------
Checks RunChecks(Context& ctx, const GameTarget& t, SystemInfo& sys, const ModelState* model, bool force) {
    Checks c;
    const bool have = !t.exe.empty() && IsFile(t.exe);
    auto set = [&](int i, CheckState s, const wstring& text, int code = 0) { c.st[i] = s; c.text[i] = text; c.code[i] = code; };
    if (!have) set(L_GAME, CK_FAIL, L"No game executable found - choose the game's .exe or its folder");
    else set(L_GAME, CK_OK, L"Game executable: " + BaseName(t.exe));
    if (!kDirectRoute || ctx.opti.chosen()) {  // OptiScaler takes the game's DLSS / XeSS / FSR inputs; the game's FSR files are not touched
        if (!have) set(L_FFX, CK_NA, L"Game inputs through OptiScaler (DLSS, XeSS or FSR)");
        else if (!ctx.opti.ok()) set(L_FFX, CK_FAIL, OptiProblem(ctx));
        else {
            wstring what;
            if (!FindFilesNamed(t.root, L"nvngx_dlss.dll", 30000, 8).empty()) what += L"DLSS";
            if (!FindFilesNamed(t.root, L"libxess.dll", 30000, 8).empty()) what += what.empty() ? L"XeSS" : L" / XeSS";
            if (t.anySlot()) what += what.empty() ? L"FSR" : L" / FSR";
            const wstring head = L"OptiScaler " + ctx.opti.version + L" hands the game's " + (what.empty() ? wstring(L"upscaler") : what) + L" inputs to WrongGPU";
            if (what.empty()) set(L_FFX, CK_WARN, head + L" - no DLSS, XeSS or FSR files found in this game");
            else set(L_FFX, CK_OK, head);
        }
    } else if (!have) set(L_FFX, CK_NA, L"AMD FSR 3 (amd_fidelityfx_dx12.dll or ffx_fsr3_x64.dll / ffx_fsr3upscaler_x64.dll)");
    else if (!t.anySlot()) set(L_FFX, CK_FAIL, L"No AMD FSR 3 DLL found - for a game with DLSS or XeSS, click \"OptiScaler...\" and choose your OptiScaler download");
    else {
        wstring found, foreign, orphan, missingInPayload;
        int installable = 0;
        for (auto& st : t.slots) {
            if (st.kind == FFX_NONE) continue;
            if (st.kind == FFX_FOREIGN && foreign.empty()) foreign = wstring(st.slot->name) + L" (" + st.info + L")";
            if (st.kind == FFX_OURS && !t.installed && orphan.empty()) orphan = st.slot->name;
            if ((st.kind == FFX_GAME || (st.kind == FFX_OURS && t.installed)) && st.inPayload) ++installable;
            if (!st.inPayload) missingInPayload += (missingInPayload.empty() ? L"" : L", ") + wstring(st.slot->name);
            found += (found.empty() ? L"" : L", ") + wstring(st.slot->name) +
                     (st.kind == FFX_OURS ? L" (WrongGPU)" : st.slot->amdBuilt ? L" (AMD, " + st.info.substr(st.info.find(L"version")) + L")" : L" (the game's FSR 3.0 build)");
        }
        if (IsFile(Join(t.dir, L"wronggpu-fsr-install.json")))  // a test install made with clean/fsr/install/install-wronggpu-fsr.ps1
            set(L_FFX, CK_FAIL, L"WrongGPU was installed here with the test script (wronggpu-fsr-install.json) - remove it first: install-wronggpu-fsr.ps1 -Action Uninstall");
        else if (!foreign.empty()) set(L_FFX, CK_FAIL, foreign + L" is not the game's FSR DLL - another mod replaced it; verify the game files first", EXIT_OTHER_MODS);
        else if (!orphan.empty()) set(L_FFX, CK_FAIL, L"WrongGPU's " + orphan + L" is here without an install record - verify the game files first");
        else if (!installable) set(L_FFX, CK_FAIL, L"This installer has no WrongGPU replacement for " + missingInPayload);
        else set(L_FFX, CK_OK, (t.installed ? L"WrongGPU is installed here already; it will be updated (" : L"AMD FSR found: ") + found + (t.installed ? L")" : L""));
    }
    if (!have) set(L_AC, CK_NA, L"Anti-cheat");
    else if (!t.antiCheat.empty()) set(L_AC, CK_FAIL, AntiCheatMessage(t.antiCheat), EXIT_ANTICHEAT);
    else set(L_AC, CK_OK, L"No anti-cheat found");
    if (sys.win.ok) set(L_WIN, CK_OK, sys.win.text);
    else set(L_WIN, CK_FAIL, L"64-bit Windows 10 or 11 is needed (found " + sys.win.text + L")");
    // GPU (spec 30 answer 8): the adapter name is a fast pre-filter; the HIP device architecture decides (gfx1201, gfx1200).
    const wstring hipFix = L"update the AMD Software: Adrenalin Edition driver (" + wstring(kDriverPage) + L"), or install the AMD HIP SDK for Windows (" + kHipDownload + L")";
    if (!sys.gpu.rdna4) {
        set(L_GPU, CK_FAIL, L"WrongGPU needs an AMD Radeon RX 9000-series graphics card." + (sys.gpu.name.empty() ? wstring() : L" (found " + sys.gpu.name + L")"));
        set(L_HIP, CK_NA, L"AMD HIP runtime (checked once an RX 9000-series card is found)");
    } else {
        if (!sys.hip.ran) sys.hip = CheckHip(ctx);
        const HipInfo& h = sys.hip;
        if (h.ok) {
            set(L_GPU, CK_OK, h.device + L" (" + h.arch + L")");
            set(L_HIP, CK_OK, h.text);
        } else if (ctx.hipOptional) {  // the launcher: Model M runs on the D3D12 executor without HIP
            set(L_GPU, CK_OK, sys.gpu.name + L" (D3D12 executor)");
            set(L_HIP, CK_WARN, L"HIP runtime not usable (" + h.text + L") - not needed for Model M on the D3D12 executor; "
                                L"needed only for the HIP executor, sparkle guard and frame capture");
        } else if (h.code == 14 || h.code == 16) {
            set(L_GPU, CK_WARN, sys.gpu.name + L" detected by DXGI; no HIP candidate validated gfx1200 or gfx1201. Installation is blocked.");
            set(L_HIP, CK_FAIL, h.text + L" - " + hipFix + L". Collect logs for all attempted runtime paths and architectures.");
        } else {
            set(L_GPU, CK_WARN, sys.gpu.name + L" (architecture not checked: the HIP runtime does not work)");
            set(L_HIP, CK_FAIL, h.text + L" - " + hipFix);
        }
    }
    if (force && c.st[L_GPU] != CK_OK) c.text[L_GPU] += L" - check skipped (/FORCE)";
    if (force && c.st[L_HIP] != CK_OK) c.text[L_HIP] += L" - check skipped (/FORCE)";
    if (!model || model->status == ModelState::NONE) set(L_MODEL, CK_NA, L"DLSS model file (nvngx_dlss.dll 310.5 or newer)");
    else if (model->status == ModelState::SEARCHING) set(L_MODEL, CK_NA, L"Looking for a DLSS model file and converting it...");
    else if (model->ok()) set(L_MODEL, CK_OK, L"Model data: " + (model->reused ? wstring(L"already prepared on this PC")
                                                                             : BaseName(model->dll) + L" " + (model->version.empty() ? wstring() : model->version + L" ") +
                                                                                   L"from " + model->origin + L", converted on this PC"));
    // model->code: an ignore-list case (convert_ignore.h) with its own words: a list that can't be read (10), every file found
    // listed, the model data on this PC converted from a listed file (5)
    else set(L_MODEL, CK_FAIL, model->damaged || model->code ? model->error : L"No usable DLSS model file (nvngx_dlss.dll 310.5 or newer) - click \"Model file...\"",
             model->damaged ? EXIT_DAMAGED : model->code ? model->code : EXIT_NO_MODEL);
    if (!have) set(L_CONFLICT, CK_NA, L"Other upscaler mods and older installs");
    else if (const Finding* no = t.coexist.refusal(); no && no->tool != Tool::AntiCheat)  // the rules table: no override
        set(L_CONFLICT, CK_FAIL, no->what + L": " + no->message, no->code ? no->code : EXIT_REFUSED);
    else if (!t.conflicts.outside.empty()) set(L_CONFLICT, CK_FAIL, L"Files of another mod or an older install are outside the FSR folder (" + BaseName(t.conflicts.outside[0]) + L") - remove them first", EXIT_OTHER_MODS);
    else if (t.conflicts.modsFound() || t.conflicts.oldFound()) {
        wstring what;
        for (auto& m : t.conflicts.mods) what += (what.empty() ? L"" : L", ") + m;
        if (t.conflicts.oldFound()) what += (what.empty() ? L"" : L", ") + wstring(L"an older WrongGPU (DLSS 4.5 AMD) install");
        set(L_CONFLICT, CK_WARN, L"Found " + what + L" - needs your OK below", t.conflicts.modsFound() ? EXIT_OTHER_MODS : EXIT_OLD_INSTALL);
    } else set(L_CONFLICT, CK_OK, L"No other upscaler mod or older install found");
    if (!have) set(L_IDLE, CK_NA, L"Game is not running");
    else if (GameRunning(t.exe)) set(L_IDLE, CK_FAIL, BaseName(t.exe) + L" is running - close the game first");
    else set(L_IDLE, CK_OK, L"Game is not running");
    if (have && (t.anySlot() || ctx.opti.chosen()) && Contains(t.dir, L"\\WindowsApps\\"))
        set(L_FFX, CK_FAIL, L"This game is installed in the protected WindowsApps folder (Xbox app / Microsoft Store), which does not allow mods");
    return c;
}

}  // namespace wg
