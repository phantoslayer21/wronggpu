// SPDX-License-Identifier: MIT
// The SR front end as OptiScaler 10's custom FFX upscaler (DESIGN-i15 8, spec 65), end to end on WARP, driven by our
// own OptiScaler-style loader (fake_opti_loader.h). The game folder is this test's own folder:
//   <exe dir>\OptiScaler.ini                                   [WrongGPU] UpscalerIndex
//   <exe dir>\OptiScaler\custom\WrongGPU\ffx\amd_fidelityfx_upscaler_dx12.dll   the built package's DLL
//            ...\ffx\amd_fidelityfx_dx12.wronggpu-original.dll  AMD's test double (the AMD entries of the list)
//   <exe dir>\OptiScaler\custom\WrongGPU M\ffx\ + wronggpu-entry.ini        one folder per choice (fallback)
//   <exe dir>\OptiScaler\custom\WrongGPU K\ffx\ + wronggpu-entry.ini
//   <exe dir>\WrongGPU\                                        the data folder (wronggpu.ini, wronggpu_runtime.dll = test double)
//   <exe dir>\OptiScaler\plugins\canary.txt, <exe dir>\OptiScaler\custom\Other\ffx\canary.txt   must never be touched
// Checks: the version list (the registry's two SR entries, then AMD's; i20: no RR entries), the entries write sr_model
// and never rr_mode (the RR plugin's box owns it), a reader opened here sees every write (new
// generation), wronggpu.ini is persisted, re-create on a change, AMD's entry is forwarded, the OptiScaler.ini
// UpscalerIndex poll applies a change within the poll budget, one folder per choice lists one entry each and applies
// it, and the DLL's own file accesses (an import-table canary on every loaded instance) stay in its folder and the
// data folder (plus the read-only host ini).
// Arguments: <package slot DLL> <AMD test double DLL> <runtime test double DLL>
#define WG_SETTINGS_IO_IMPLEMENTATION
#include "wg_settings_io.h"

#include "fake_opti_loader.h"

#include <d3d12.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static int g_checks = 0, g_failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(c)) {                                                                  \
            ++g_failures;                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
        }                                                                            \
    } while (0)

// ---- file-access canary: the loaded DLL's imports of the file / library functions, recorded -----------------------

struct Access {
    std::string op;
    std::wstring path;
    bool write;
};
static std::mutex g_accessMutex;
static std::vector<Access> g_access;

static void Rec(const char* op, const wchar_t* path, bool write) {
    std::lock_guard<std::mutex> lock(g_accessMutex);
    g_access.push_back({op, path ? path : L"", write});
}
static void RecA(const char* op, const char* path, bool write) { Rec(op, wg::widen(path ? path : "").c_str(), write); }

template <class F>
static F Orig(const char* name) {
    static HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    return reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(k32, name)));
}

static bool Writes(DWORD access, DWORD disposition) {
    return (access & (GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA | FILE_APPEND_DATA | DELETE)) || disposition == CREATE_ALWAYS ||
           disposition == CREATE_NEW || disposition == TRUNCATE_EXISTING;
}

static HANDLE WINAPI HkCreateFileW(LPCWSTR n, DWORD a, DWORD s, LPSECURITY_ATTRIBUTES sa, DWORD d, DWORD f, HANDLE t) {
    Rec("CreateFileW", n, Writes(a, d));
    return Orig<decltype(&CreateFileW)>("CreateFileW")(n, a, s, sa, d, f, t);
}
static HANDLE WINAPI HkCreateFileA(LPCSTR n, DWORD a, DWORD s, LPSECURITY_ATTRIBUTES sa, DWORD d, DWORD f, HANDLE t) {
    RecA("CreateFileA", n, Writes(a, d));
    return Orig<decltype(&CreateFileA)>("CreateFileA")(n, a, s, sa, d, f, t);
}
static HMODULE WINAPI HkLoadLibraryW(LPCWSTR n) {
    Rec("LoadLibraryW", n, false);
    return Orig<decltype(&LoadLibraryW)>("LoadLibraryW")(n);
}
static HMODULE WINAPI HkLoadLibraryA(LPCSTR n) {
    RecA("LoadLibraryA", n, false);
    return Orig<decltype(&LoadLibraryA)>("LoadLibraryA")(n);
}
static HMODULE WINAPI HkLoadLibraryExW(LPCWSTR n, HANDLE h, DWORD f) {
    Rec("LoadLibraryExW", n, false);
    return Orig<decltype(&LoadLibraryExW)>("LoadLibraryExW")(n, h, f);
}
static HMODULE WINAPI HkLoadLibraryExA(LPCSTR n, HANDLE h, DWORD f) {
    RecA("LoadLibraryExA", n, false);
    return Orig<decltype(&LoadLibraryExA)>("LoadLibraryExA")(n, h, f);
}
static DWORD WINAPI HkGetFileAttributesW(LPCWSTR n) {
    Rec("GetFileAttributesW", n, false);
    return Orig<decltype(&GetFileAttributesW)>("GetFileAttributesW")(n);
}
static BOOL WINAPI HkGetFileAttributesExW(LPCWSTR n, GET_FILEEX_INFO_LEVELS l, LPVOID i) {
    Rec("GetFileAttributesExW", n, false);
    return Orig<decltype(&GetFileAttributesExW)>("GetFileAttributesExW")(n, l, i);
}
static HANDLE WINAPI HkFindFirstFileExW(LPCWSTR n, FINDEX_INFO_LEVELS l, LPVOID d, FINDEX_SEARCH_OPS o, LPVOID f, DWORD x) {
    Rec("FindFirstFileExW", n, false);
    return Orig<decltype(&FindFirstFileExW)>("FindFirstFileExW")(n, l, d, o, f, x);
}
static HANDLE WINAPI HkFindFirstFileW(LPCWSTR n, LPWIN32_FIND_DATAW d) {
    Rec("FindFirstFileW", n, false);
    return Orig<decltype(&FindFirstFileW)>("FindFirstFileW")(n, d);
}
static BOOL WINAPI HkMoveFileExW(LPCWSTR a, LPCWSTR b, DWORD f) {
    Rec("MoveFileExW", a, true);
    Rec("MoveFileExW", b, true);
    return Orig<decltype(&MoveFileExW)>("MoveFileExW")(a, b, f);
}
static BOOL WINAPI HkDeleteFileW(LPCWSTR n) {
    Rec("DeleteFileW", n, true);
    return Orig<decltype(&DeleteFileW)>("DeleteFileW")(n);
}
static BOOL WINAPI HkCreateDirectoryW(LPCWSTR n, LPSECURITY_ATTRIBUTES sa) {
    Rec("CreateDirectoryW", n, true);
    return Orig<decltype(&CreateDirectoryW)>("CreateDirectoryW")(n, sa);
}
static BOOL WINAPI HkCopyFileW(LPCWSTR a, LPCWSTR b, BOOL f) {
    Rec("CopyFileW", a, false);
    Rec("CopyFileW", b, true);
    return Orig<decltype(&CopyFileW)>("CopyFileW")(a, b, f);
}

static void* HookFor(const char* name) {
    static const struct { const char* name; void* hook; } kHooks[] = {
        {"CreateFileW", reinterpret_cast<void*>(&HkCreateFileW)},       {"CreateFileA", reinterpret_cast<void*>(&HkCreateFileA)},
        {"LoadLibraryW", reinterpret_cast<void*>(&HkLoadLibraryW)},     {"LoadLibraryA", reinterpret_cast<void*>(&HkLoadLibraryA)},
        {"LoadLibraryExW", reinterpret_cast<void*>(&HkLoadLibraryExW)}, {"LoadLibraryExA", reinterpret_cast<void*>(&HkLoadLibraryExA)},
        {"GetFileAttributesW", reinterpret_cast<void*>(&HkGetFileAttributesW)},
        {"GetFileAttributesExW", reinterpret_cast<void*>(&HkGetFileAttributesExW)},
        {"FindFirstFileExW", reinterpret_cast<void*>(&HkFindFirstFileExW)}, {"FindFirstFileW", reinterpret_cast<void*>(&HkFindFirstFileW)},
        {"MoveFileExW", reinterpret_cast<void*>(&HkMoveFileExW)},       {"DeleteFileW", reinterpret_cast<void*>(&HkDeleteFileW)},
        {"CreateDirectoryW", reinterpret_cast<void*>(&HkCreateDirectoryW)}, {"CopyFileW", reinterpret_cast<void*>(&HkCopyFileW)},
    };
    for (const auto& h : kHooks)
        if (std::strcmp(h.name, name) == 0) return h.hook;
    return nullptr;
}

// Points the module's import slots of the functions above at the recorders. Returns the number of slots patched.
static int PatchImports(HMODULE m) {
    auto* base = reinterpret_cast<BYTE*>(m);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return 0;
    int n = 0;
    for (auto* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); d->Name; ++d) {
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            auto* ibn = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            void* hook = HookFor(reinterpret_cast<const char*>(ibn->Name));
            if (!hook) continue;
            DWORD old = 0;
            VirtualProtect(&slots->u1.Function, sizeof(void*), PAGE_READWRITE, &old);
            slots->u1.Function = reinterpret_cast<ULONG_PTR>(hook);
            VirtualProtect(&slots->u1.Function, sizeof(void*), old, &old);
            ++n;
        }
    }
    return n;
}

// ---- helpers --------------------------------------------------------------------------------------------------------

static std::wstring Lower(std::wstring s) {
    for (wchar_t& c : s) c = static_cast<wchar_t>(towlower(c));
    for (wchar_t& c : s) if (c == L'/') c = L'\\';
    return s;
}

static bool Under(const std::wstring& path, const std::wstring& dir) {
    const std::wstring p = Lower(path), d = Lower(dir) + L"\\";
    return p.compare(0, d.size(), d) == 0;
}

static bool Copy(const std::wstring& from, const std::wstring& to) { return CopyFileW(from.c_str(), to.c_str(), FALSE) != 0; }

static void MakeDirs(const std::wstring& dir) {
    for (size_t i = 3; i <= dir.size(); ++i)
        if (i == dir.size() || dir[i] == L'\\') CreateDirectoryW(dir.substr(0, i).c_str(), nullptr);
}

static void Remove(const std::wstring& path) { DeleteFileW(path.c_str()); }

static std::string ReadText(const std::wstring& path) {
    std::string t;
    wg::read_file(path, &t);
    return t;
}

static bool WaitFor(const std::function<bool()>& cond, int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!cond()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

static ID3D12Device* WarpDevice() {
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) return nullptr;
    IDXGIAdapter* warp = nullptr;
    ID3D12Device* device = nullptr;
    if (SUCCEEDED(factory->EnumWarpAdapter(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&warp))))
        D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device));
    if (warp) warp->Release();
    factory->Release();
    return device;
}

static int EntryIndex(const FakeOptiLoader& l, uint64_t id) {
    for (size_t i = 0; i < l.ids.size(); ++i)
        if (l.ids[i] == id) return int(i);
    return -1;
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 4) {
        std::printf("usage: test_custom_slot <package slot DLL> <AMD test double> <runtime test double>\n");
        return 2;
    }
    const std::wstring package = argv[1], fakeAmd = argv[2], fakeRuntime = argv[3];
    const std::wstring root = wg::exe_dir();
    const std::wstring opti = root + L"\\OptiScaler", data = root + L"\\WrongGPU", hostIni = root + L"\\OptiScaler.ini";
    const std::wstring scratchLocal = root + L"\\localappdata";  // the DLL's AMD-provider lookup reads %LOCALAPPDATA%; never the real one here
    SetEnvironmentVariableW(L"LOCALAPPDATA", scratchLocal.c_str());
    SetEnvironmentVariableW(L"WG_DATA_DIR", nullptr);  // the default rule: <exe dir>\WrongGPU
    SetEnvironmentVariableW(L"WG_RUNTIME_OPTIONS", nullptr);
    SetEnvironmentVariableW(L"WG_RUNTIME_MODEL", nullptr);
    SetEnvironmentVariableW(L"WG_RR_MODE", nullptr);

    // ---- the game folder (files of an earlier run removed one by one) ----
    const wchar_t* folders[] = {L"WrongGPU", L"WrongGPU M", L"WrongGPU K"};
    for (const wchar_t* f : folders) {
        const std::wstring slot = opti + L"\\custom\\" + f + L"\\ffx";
        MakeDirs(slot);
        for (const wchar_t* leaf : {L"\\wronggpu-fsr.log", L"\\wronggpu-fsr.log.old", L"\\wronggpu-entry.ini", L"\\wronggpu.ini",
                                    L"\\wronggpu-diagnostics-amd_fidelityfx_dx12.jsonl"})
            Remove(slot + leaf);
        CHECK(Copy(package, slot + L"\\" + FakeOptiLoader::kSlotFile));
        CHECK(Copy(fakeAmd, slot + L"\\amd_fidelityfx_dx12.wronggpu-original.dll"));
    }
    MakeDirs(data);
    MakeDirs(scratchLocal);
    for (const wchar_t* leaf : {L"\\wronggpu.ini", L"\\wronggpu.ini.tmp"}) Remove(data + leaf);
    Remove(root + L"\\wronggpu.ini");
    CHECK(Copy(fakeRuntime, data + L"\\wronggpu_runtime.dll"));
    MakeDirs(opti + L"\\plugins");
    MakeDirs(opti + L"\\custom\\Other\\ffx");
    CHECK(wg::write_file_atomic(opti + L"\\plugins\\canary.txt", "canary\r\n"));
    CHECK(wg::write_file_atomic(opti + L"\\custom\\Other\\ffx\\canary.txt", "canary\r\n"));
    CHECK(wg::write_file_atomic(hostIni, "[Upscalers]\r\nDx12Upscaler=WrongGPU\r\n[WrongGPU]\r\nUpscalerIndex=auto\r\n"));
    // The fallback's sidecars.
    CHECK(wg::write_file_atomic(opti + L"\\custom\\WrongGPU M\\ffx\\wronggpu-entry.ini", "entry=0x5747000000000001\r\n"));
    CHECK(wg::write_file_atomic(opti + L"\\custom\\WrongGPU K\\ffx\\wronggpu-entry.ini", "entry=0x5747000000000002\r\n"));

    ID3D12Device* device = WarpDevice();
    if (!device) {
        std::printf("SKIP: no WARP D3D12 device\n");
        return 77;
    }

    // ---- the version-list layout: custom\WrongGPU\ffx ----
    FakeOptiLoader main;
    main.root = root;
    main.folder = L"WrongGPU";
    CHECK(main.Load());
    if (!main.module) return 1;
    CHECK(PatchImports(main.module) > 0);
    CHECK(main.QueryVersions(device));
    // The registry's entries this build offers (the SR models; i20: the RR modes are the RR plugin's own box; a build
    // without Model K lists Model M alone, Model K being the registry's last row), then AMD's (the test double).
    const uint32_t kOwn = WG_ENTRY_COUNT - (WRONGGPU_MODEL_K ? 0 : 1);
    CHECK(main.ids.size() == size_t(kOwn) + 1);
    for (uint32_t i = 0; i < kOwn && i < main.ids.size(); ++i)
        CHECK(main.ids[i] == kWgEntries[i].id && main.names[i] == kWgEntries[i].label);
    if (main.ids.size() == size_t(kOwn) + 1)
        CHECK(main.ids[kOwn] == 0xFA4E && main.names[kOwn] == "fake AMD FSR 3.1");
    CHECK(!main.names.empty() && main.names[0].size() > 7 &&
          main.names[0].compare(main.names[0].size() - 7, 7, "##3.1.0") == 0);  // the host's API level, hidden after ##
    CHECK(main.IndexFromIni() == -1);

    // A reader of our own on the same block (this process's name).
    wg::SharedSettings reader;
    std::string log;
    CHECK(reader.open(WG_SS_WRITER_TEST, data + L"\\wronggpu.ini", &log) == wg::SharedSettings::Status::Opened);
    wg::Snapshot snap;
    CHECK(reader.snapshot(&snap));
    CHECK(snap.writer == WG_SS_WRITER_SR_FRONTEND && (snap.flags & WG_SS_FLAG_HOSTED_BY_OPTISCALER));
    CHECK(!reader.is_writer() && !reader.write_slot(WG_OPT_SR_MODEL, WG_SR_MODEL_K));  // a non-writer never writes
    CHECK(wg::file_exists(data + L"\\wronggpu.ini"));  // defaults written into the data folder
    CHECK(ReadText(data + L"\\wronggpu.ini").find("[fsr]") != std::string::npos);

    auto value = [&](uint32_t slot) {
        wg::Snapshot s;
        reader.snapshot(&s);
        return s.is_set(slot) ? s.values[slot] : -1;
    };
    auto choose = [&](uint64_t id, const char* expectedProvider) {
        const int index = EntryIndex(main, id);
        CHECK(index >= 0);
        wg::Snapshot before;
        reader.snapshot(&before);
        CHECK(main.Select(index, device, 1280, 720, 2560, 1440) == FFX_API_RETURN_OK && main.context);
        wg::Snapshot after;
        reader.snapshot(&after);
        uint64_t providerId = 0;
        const std::string provider = main.ProviderName(&providerId);
        CHECK(provider == expectedProvider);
        if (provider != expectedProvider) std::printf("  provider \"%s\", expected \"%s\"\n", provider.c_str(), expectedProvider);
        return std::make_pair(before.generation, after.generation);
    };

#if WRONGGPU_MODEL_K
    // Model K: sr_model=K written (a new generation), persisted to wronggpu.ini.
    auto g = choose(0x5747000000000002ull, "WrongGPU Model K");
    CHECK(g.second == g.first + 2 && value(WG_OPT_SR_MODEL) == WG_SR_MODEL_K);
    CHECK(ReadText(data + L"\\wronggpu.ini").find("sr_model = K") != std::string::npos);
    // Model M: sr_model=M; rr_mode is never written by the SR front end (the RR plugin's box owns it).
    const int32_t rrMode = value(WG_OPT_RR_MODE);
    g = choose(0x5747000000000001ull, "WrongGPU Model M");
    CHECK(g.second == g.first + 2 && value(WG_OPT_SR_MODEL) == WG_SR_MODEL_M && value(WG_OPT_RR_MODE) == rrMode);
    CHECK(ReadText(data + L"\\wronggpu.ini").find("sr_model = M") != std::string::npos);
    // Model K again, then M: each choice is a new generation.
    g = choose(0x5747000000000002ull, "WrongGPU Model K");
    CHECK(g.second == g.first + 2 && value(WG_OPT_SR_MODEL) == WG_SR_MODEL_K && value(WG_OPT_RR_MODE) == rrMode);
    g = choose(0x5747000000000001ull, "WrongGPU Model M");
    CHECK(g.second == g.first + 2 && value(WG_OPT_SR_MODEL) == WG_SR_MODEL_M);
#else
    // Model M (the only model of this build): sr_model=M written (a new generation), persisted; rr_mode is never
    // written by the SR front end (the RR plugin's box owns it).
    const int32_t rrMode = value(WG_OPT_RR_MODE);
    auto g = choose(0x5747000000000001ull, "WrongGPU Model M");
    CHECK(g.second == g.first + 2 && value(WG_OPT_SR_MODEL) == WG_SR_MODEL_M && value(WG_OPT_RR_MODE) == rrMode);
    CHECK(ReadText(data + L"\\wronggpu.ini").find("sr_model = M") != std::string::npos);
#endif
    // The same entry again (the host re-creates at a resolution change): nothing rewritten.
    g = choose(0x5747000000000001ull, "WrongGPU Model M");
    CHECK(g.second == g.first);
    // AMD's entry: forwarded to AMD's DLL, nothing written.
    {
        wg::Snapshot before;
        reader.snapshot(&before);
        CHECK(main.Select(int(kOwn), device, 1280, 720, 2560, 1440) == FFX_API_RETURN_OK && main.context);
        wg::Snapshot after;
        reader.snapshot(&after);
        CHECK(after.generation == before.generation);
        HMODULE amd = GetModuleHandleW((main.SlotDir() + L"\\amd_fidelityfx_dx12.wronggpu-original.dll").c_str());
        auto fakeLog = amd ? reinterpret_cast<const char* (*)()>(reinterpret_cast<void*>(GetProcAddress(amd, "fakeLog"))) : nullptr;
        CHECK(fakeLog && std::string(fakeLog()).find("create ctx=") != std::string::npos);
    }
    main.Release();

    // The host's persisted choice: OptiScaler.ini [WrongGPU] UpscalerIndex, polled (file time at most once per second).
    auto pollTo = [&](int index, uint32_t slot, int32_t expected) {
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(main.WriteIndex(index));
        const bool ok = WaitFor([&] { return value(slot) == expected; }, 1500);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("poll: UpscalerIndex=%d applied after %.0f ms\n", index, ms);
        CHECK(ok);
        return ms;
    };
    double worst = 0;
#if WRONGGPU_MODEL_K
    worst = std::max(worst, pollTo(EntryIndex(main, 0x5747000000000002ull), WG_OPT_SR_MODEL, WG_SR_MODEL_K));
    worst = std::max(worst, pollTo(EntryIndex(main, 0x5747000000000001ull), WG_OPT_SR_MODEL, WG_SR_MODEL_M));
    worst = std::max(worst, pollTo(EntryIndex(main, 0x5747000000000002ull), WG_OPT_SR_MODEL, WG_SR_MODEL_K));
    const int32_t lastModel = WG_SR_MODEL_K;
#else
    worst = std::max(worst, pollTo(EntryIndex(main, 0x5747000000000001ull), WG_OPT_SR_MODEL, WG_SR_MODEL_M));
    const int32_t lastModel = WG_SR_MODEL_M;
#endif
    std::printf("poll: worst %.0f ms (poll budget: file time checked at most once per second + 200 ms tick)\n", worst);
    CHECK(worst <= 1300.0);
    // An AMD index: nothing applied.
    {
        CHECK(main.WriteIndex(int(kOwn)));
        std::this_thread::sleep_for(std::chrono::milliseconds(1400));
        CHECK(value(WG_OPT_RR_MODE) == rrMode && value(WG_OPT_SR_MODEL) == lastModel);
    }

    // ---- one folder per choice: two more instances of the same DLL, each with its sidecar ----
    FakeOptiLoader modelM, modelK;
    modelM.root = modelK.root = root;
    modelM.folder = L"WrongGPU M";
    modelK.folder = L"WrongGPU K";
#if WRONGGPU_MODEL_K
    for (FakeOptiLoader* l : {&modelM, &modelK}) {
#else
    for (FakeOptiLoader* l : {&modelM}) {  // (no Model K folder in a build without Model K)
#endif
        CHECK(l->Load());
        if (!l->module) return 1;
        CHECK(l->module != main.module);
        CHECK(PatchImports(l->module) > 0);
        CHECK(l->QueryVersions(device));
        CHECK(l->ids.size() == 1);
    }
    CHECK(modelM.ids.size() == 1 && modelM.ids[0] == 0x5747000000000001ull && modelM.names[0] == "WrongGPU Model M##3.1.0");
#if WRONGGPU_MODEL_K
    CHECK(modelK.ids.size() == 1 && modelK.ids[0] == 0x5747000000000002ull && modelK.names[0] == "WrongGPU Model K##3.1.0");
#endif
    // The main folder is on the last polled model; then the folders apply their entries.
    CHECK(modelM.Select(0, device, 1280, 720, 2560, 1440) == FFX_API_RETURN_OK && modelM.ProviderName() == "WrongGPU Model M");
    CHECK(value(WG_OPT_SR_MODEL) == WG_SR_MODEL_M && value(WG_OPT_RR_MODE) == rrMode);
#if WRONGGPU_MODEL_K
    CHECK(modelK.Select(0, device, 1280, 720, 2560, 1440) == FFX_API_RETURN_OK && modelK.ProviderName() == "WrongGPU Model K");
    CHECK(value(WG_OPT_SR_MODEL) == WG_SR_MODEL_K && value(WG_OPT_RR_MODE) == rrMode);
#endif
    wg::Snapshot last;
    reader.snapshot(&last);
    CHECK(last.flags & WG_SS_FLAG_FOLDER_ENTRY);
    modelM.Release();
    modelK.Release();

    // ---- logs: the registry's words ----
    const std::wstring mainLog = main.SlotDir() + L"\\wronggpu-fsr.log";
#if WRONGGPU_MODEL_K  // (without Model K the poll never changes the model, so it logs nothing)
    CHECK(WaitFor([&] { return ReadText(mainLog).find("settings: sr_model=K (optiscaler.ini)") != std::string::npos; }, 3000));
#endif
    const std::string text = ReadText(mainLog);
    CHECK(text.find("custom slot: loaded from OptiScaler's custom folder \"WrongGPU\"") != std::string::npos);
    CHECK(text.find("the host chose WrongGPU Model M (0x5747000000000001): settings: sr_model=M (menu)") != std::string::npos);
    CHECK(text.find(": settings: rr_mode=") == std::string::npos);  // the host never chooses an RR mode here
#if WRONGGPU_MODEL_K
    CHECK(text.find("the host chose WrongGPU Model K (0x5747000000000002): settings: sr_model=K (menu)") != std::string::npos);
#endif
    CHECK(text.find("this SR front end writes the shared settings") != std::string::npos);
    CHECK(text.find(WRONGGPU_MODEL_K ? ";model=K" : ";model=M") != std::string::npos);
    const std::string folderLog = ReadText(modelM.SlotDir() + L"\\wronggpu-fsr.log");
    CHECK(folderLog.find("one folder per choice") != std::string::npos);
    CHECK(WaitFor([&] { return ReadText(modelM.SlotDir() + L"\\wronggpu-fsr.log").find("settings: sr_model=M (folder)") != std::string::npos; }, 3000));

    // ---- the canary: every file / library access of the three instances ----
    std::vector<std::wstring> allowed = {main.SlotDir(), modelM.SlotDir(), modelK.SlotDir(), data};
    wchar_t windows[MAX_PATH] = {};
    GetWindowsDirectoryW(windows, MAX_PATH);
    size_t outside = 0, total = 0, writes = 0, hostReads = 0, bare = 0, probes = 0;
    {
        std::lock_guard<std::mutex> lock(g_accessMutex);
        total = g_access.size();
        for (const Access& a : g_access) {
            const std::wstring p = Lower(a.path);
            const bool hasDir = p.find(L'\\') != std::wstring::npos;
            CHECK(p.find(L"\\optiscaler\\plugins") == std::wstring::npos && p.find(L"\\custom\\other") == std::wstring::npos);
            if (!hasDir) {
                ++bare;  // a library by name (the system's search)
                continue;
            }
            bool ok = false;
            for (const std::wstring& d : allowed) ok = ok || Under(a.path, d);
            if (a.write) ++writes;
            if (!ok && !a.write && Lower(a.path) == Lower(hostIni)) ok = true, ++hostReads;
            if (!ok && !a.write && Under(a.path, windows)) ok = true;
            // The data folder itself (does it exist?) and the old place of wronggpu.ini next to the exe (existence
            // probes of the data folder rule, read-only).
            if (!ok && !a.write && (Lower(a.path) == Lower(data) || Lower(a.path) == Lower(root + L"\\wronggpu.ini"))) ok = true, ++probes;
            // The AMD-provider lookup (pre-existing, read-only): %LOCALAPPDATA%\WrongGPU\compare.txt.
            if (!ok && !a.write && Lower(a.path) == Lower(scratchLocal + L"\\WrongGPU\\compare.txt")) ok = true;
            // ... and its search of the game folder for AMD's SDK DLLs (folder listings only; custom and plugins skipped).
            if (!ok && !a.write && a.op == "FindFirstFileExW" && Under(a.path, root)) ok = true;
            if (!ok) {
                ++outside;
                std::printf("canary: %s %ls%s\n", a.op.c_str(), a.path.c_str(), a.write ? " (write)" : "");
            }
        }
    }
    std::printf("canary: %zu accesses (%zu writes, %zu reads of the host ini, %zu data-folder-rule probes, %zu libraries by name), "
                "%zu outside\n", total, writes, hostReads, probes, bare, outside);
    // Every write, listed (they must all be in the DLL's folder or the data folder).
    {
        std::lock_guard<std::mutex> lock(g_accessMutex);
        std::vector<std::wstring> seen;
        for (const Access& a : g_access) {
            if (!a.write) continue;
            const std::wstring p = Lower(a.path);
            if (std::find(seen.begin(), seen.end(), p) != seen.end()) continue;
            seen.push_back(p);
            std::printf("canary: write %s %ls\n", a.op.c_str(), a.path.c_str());
        }
    }
    CHECK(total > 0 && writes > 0 && hostReads > 0 && outside == 0);

    // ---- the churn of 2026-10-30 (Cyberpunk): WrongGPU's upscaler cannot be created (no model) in a custom folder
    //      without an AMD DLL of WrongGPU's own, and the host asks for its upscaler again every frame. Each create must
    //      fail cleanly (no context: the host's own upscaler takes over), only the first one may try the runtime, the
    //      log says it once, and no D3D12 vtable may be patched for a create that failed. ----
    {
        SetEnvironmentVariableW(L"WG_FAKE_RUNTIME", L"split_fail_create");  // a split backend whose create fails
        const std::wstring churnSlot = opti + L"\\custom\\WrongGPU Churn\\ffx";
        MakeDirs(churnSlot);
        for (const wchar_t* leaf : {L"\\wronggpu-fsr.log", L"\\wronggpu-fsr.log.old", L"\\wronggpu-diagnostics-amd_fidelityfx_dx12.jsonl",
                                    L"\\amd_fidelityfx_dx12.wronggpu-original.dll"})
            Remove(churnSlot + leaf);
        CHECK(Copy(package, churnSlot + L"\\" + FakeOptiLoader::kSlotFile));
        // The vtables a hook installation would patch, before.
        auto table = [](void* object, int n) {
            void** vt = *reinterpret_cast<void***>(object);
            return std::vector<void*>(vt, vt + n);
        };
        ID3D12CommandQueue* queue = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                                       reinterpret_cast<void**>(&allocator))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, __uuidof(ID3D12GraphicsCommandList),
                                                  reinterpret_cast<void**>(&list))));
        const std::vector<void*> deviceBefore = table(device, 40);
        const std::vector<void*> queueBefore = queue ? table(queue, 18) : std::vector<void*>();
        const std::vector<void*> listBefore = list ? table(list, 50) : std::vector<void*>();

        FakeOptiLoader churn;
        churn.root = root;
        churn.folder = L"WrongGPU Churn";
        CHECK(churn.Load());
        CHECK(churn.QueryVersions(device));
        CHECK(churn.ids.size() == size_t(kOwn));  // no AMD entries: none from the host's FidelityFX DLLs
        const int m = EntryIndex(churn, 0x5747000000000001ull);
        CHECK(m >= 0);
        int failed = 0;
        for (int frame = 0; frame < 50; ++frame) {
            const ffxReturnCode_t rc = churn.Select(m, device, 1280, 720, 2560, 1440);
            failed += rc != FFX_API_RETURN_OK && churn.context == nullptr;
        }
        CHECK(failed == 50);
        CHECK(table(device, 40) == deviceBefore);
        if (queue) CHECK(table(queue, 18) == queueBefore);
        if (list) CHECK(table(list, 50) == listBefore);
        // The log is written by the DLL's writer thread: wait for the last line this run must produce.
        CHECK(WaitFor([&] { return ReadText(churnSlot + L"\\wronggpu-fsr.log").find("custom slot: create 10 since the failure") != std::string::npos; },
                      5000));
        const std::string churnLog = ReadText(churnSlot + L"\\wronggpu-fsr.log");
        auto count = [&](const char* needle) {
            size_t n = 0;
            for (size_t at = churnLog.find(needle); at != std::string::npos; at = churnLog.find(needle, at + 1)) ++n;
            return n;
        };
        std::printf("churn: 50 creates, %zu ffxCreateContext lines, %zu runtime attempts, %zu remembered-failure summaries\n",
                    count("ffxCreateContext"), count("create: interface="), count("custom slot: create "));
        CHECK(count("U1 upscaler ERROR: fake runtime: no HIP device found") == 1);  // one real attempt
        CHECK(count("create: interface=") == 1);
        CHECK(count("this failure is remembered for the session") == 1);
        CHECK(count("no context, the host's own upscaler takes over") == 1);
        CHECK(count("custom slot: create ") == 4);                    // repeats 1, 2, 3, 10 of 49
        CHECK(count("ffxCreateContext") == 1);
        CHECK(count("d3d12 hooks: installing") == 0 && count("patched") == 0);
        CHECK(count("goes to AMD FSR") == 0 && count("AMD provider") == 0);
        CHECK(count("custom slot: when WrongGPU's upscaler cannot be created, the create fails") == 1);
        if (list) list->Release();
        if (allocator) allocator->Release();
        if (queue) queue->Release();
        SetEnvironmentVariableW(L"WG_FAKE_RUNTIME", nullptr);
    }

    device->Release();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
