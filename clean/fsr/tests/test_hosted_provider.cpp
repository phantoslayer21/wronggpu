// SPDX-License-Identifier: MIT
// Hosted WrongGPU (wronggpu_ffx_dx12.dll, as OptiScaler loads it) with AMD's versions from the game's own DLLs. Own
// host mock and test double (fake_original.cpp); no third-party code. Run from one of two folders (CMakeLists.txt):
//   game: the test, wronggpu_ffx_dx12.dll and the test double as amd_fidelityfx_dx12.dll (the game's own FidelityFX
//         DLL), nothing else: no installer copy, no FSR 4 DLL. AMD's version is listed after WrongGPU's, selecting it
//         creates, dispatches and destroys on that same DLL, and the log names it. Also the module checks of
//         CheckProviderModule: a redirected load and a hooked export are refused.
//   none: the test and wronggpu_ffx_dx12.dll only: WrongGPU alone, and the log says why.
// %LOCALAPPDATA% points at an empty folder, so an FSR 4 DLL the installer found on this PC is never used.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "../src/frontend.h"
#include "../src/win_util.h"

namespace wronggpu { extern const char* const kLogTag = "test_hosted_provider"; }  // (nothing here writes a log)

namespace {

int failures = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++failures;                                                \
        }                                                              \
    } while (0)

std::wstring Here() {
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring dir(path);
    return dir.substr(0, dir.find_last_of(L'\\'));
}

std::string ReadText(const std::wstring& path) {
    std::string text;
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return text;
    char buffer[4096];
    DWORD n = 0;
    while (ReadFile(f, buffer, sizeof buffer, &n, nullptr) && n) text.append(buffer, n);
    CloseHandle(f);
    return text;
}

// The log is written by a background thread: waits up to 5 s for `part`.
bool LogHas(const std::wstring& log, const char* part) {
    for (int i = 0; i < 100; ++i) {
        if (ReadText(log).find(part) != std::string::npos) return true;
        Sleep(50);
    }
    std::printf("log %ls lacks \"%s\"\n", log.c_str(), part);
    return false;
}

extern "C" ffxReturnCode_t HostHook(ffxContext*, ffxQueryDescHeader*) { return FFX_API_RETURN_OK; }

// CheckProviderModule on the test double: as loaded, under a path it was not loaded from, with a hooked export.
void TestModuleChecks(const std::wstring& fake) {
    HMODULE m = LoadLibraryExW(fake.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    CHECK(m != nullptr);
    if (!m) return;
    std::string detail;
    CHECK(wgwin::CheckProviderModule(m, fake, detail) == wgfsr::ProbeResult::Ok);
    detail.clear();
    // A LoadLibrary hook handed out this module for another file
    CHECK(wgwin::CheckProviderModule(m, Here() + L"\\wronggpu_ffx_dx12.dll", detail) == wgfsr::ProbeResult::Redirected);
    CHECK(detail.find("loading it gave") != std::string::npos);
    // The module that holds CheckProviderModule (here the test exe, in the game WrongGPU's DLL) handed back for an AMD
    // name: refused even when it is the file that was asked for
    detail.clear();
    HMODULE exe = GetModuleHandleW(nullptr);
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(exe, exePath, MAX_PATH);
    const wgfsr::ProbeResult exeResult = wgwin::CheckProviderModule(exe, exePath, detail);
    CHECK(exeResult == wgfsr::ProbeResult::Ours && detail.find("WrongGPU's own DLL") != std::string::npos);
    // A host's proxy module (OptiScaler as dxgi.dll) handed back for an AMD name
    detail.clear();
    HMODULE dxgi = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    CHECK(dxgi && wgwin::CheckProviderModule(dxgi, Here() + L"\\amd_fidelityfx_dx12.dll", detail) == wgfsr::ProbeResult::Redirected);
    CHECK(detail.find("dxgi.dll") != std::string::npos);
    if (dxgi) FreeLibrary(dxgi);
    // An inline hook: ffxQuery starts with mov rax, <HostHook>; jmp rax
    auto* code = reinterpret_cast<unsigned char*>(reinterpret_cast<void*>(GetProcAddress(m, "ffxQuery")));
    unsigned char saved[12];
    std::memcpy(saved, code, sizeof saved);
    DWORD old = 0;
    CHECK(VirtualProtect(code, sizeof saved, PAGE_EXECUTE_READWRITE, &old));
    unsigned char patch[12] = {0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xE0};
    void* hook = reinterpret_cast<void*>(&HostHook);
    std::memcpy(patch + 2, &hook, sizeof hook);
    std::memcpy(code, patch, sizeof patch);
    detail.clear();
    CHECK(wgwin::CheckProviderModule(m, fake, detail) == wgfsr::ProbeResult::Hooked);
    CHECK(detail.find("ffxQuery goes to test_hosted_provider.exe") != std::string::npos);
    std::memcpy(code, saved, sizeof saved);
    VirtualProtect(code, sizeof saved, old, &old);
    FlushInstructionCache(GetCurrentProcess(), code, sizeof saved);
    detail.clear();
    CHECK(wgwin::CheckProviderModule(m, fake, detail) == wgfsr::ProbeResult::Ok);
    FreeLibrary(m);
}

}  // namespace

int main(int argc, char** argv) {
    const bool game = argc > 1 && std::strcmp(argv[1], "game") == 0;
    const std::wstring dir = Here();
    // An empty %LOCALAPPDATA%: no compare.txt of the installer
    const std::wstring appdata = dir + L"\\localappdata";
    CreateDirectoryW(appdata.c_str(), nullptr);
    SetEnvironmentVariableW(L"LOCALAPPDATA", appdata.c_str());
    const std::wstring log = dir + L"\\wronggpu-fsr.log";
    DeleteFileW(log.c_str());
    const std::wstring fake = dir + L"\\amd_fidelityfx_dx12.dll";
    CHECK(game == (GetFileAttributesW(fake.c_str()) != INVALID_FILE_ATTRIBUTES));
    CHECK(GetFileAttributesW((dir + L"\\wronggpu-amd-compare.dll").c_str()) == INVALID_FILE_ATTRIBUTES);
    if (game) TestModuleChecks(fake);

    HMODULE host = LoadLibraryW((dir + L"\\wronggpu_ffx_dx12.dll").c_str());
    CHECK(host != nullptr);
    if (!host) return 1;
    auto query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(host, "ffxQuery"));
    auto create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(host, "ffxCreateContext"));
    auto destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(host, "ffxDestroyContext"));
    auto dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(host, "ffxDispatch"));
    if (!query || !create || !destroy || !dispatch) return 2;
    const uint64_t own = uint64_t(wgfsr::OwnEntryCount());  // the registry's entries (SR models, RR modes)
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.outputCount = &count;
    CHECK(query(nullptr, &q.header) == FFX_API_RETURN_OK);
    uint64_t ids[16]{};
    const char* names[16]{};
    q.versionIds = ids;
    q.versionNames = names;
    count = 16;
    CHECK(query(nullptr, &q.header) == FFX_API_RETURN_OK);
    CHECK(ids[0] == wgfsr::kWrongGpuProviderVersionId);
    if (!game) {
        CHECK(count == own);
        CHECK(LogHas(log, "FSR 4 / AMD versions not available: no AMD FidelityFX DLL was found"));
        CHECK(LogHas(log, "FSR 4 / AMD versions not available: no usable AMD FidelityFX DLL"));
    } else {
        CHECK(count == own + 1 && ids[own] == 0xFA4E && names[own] && std::strcmp(names[own], "fake AMD FSR 3.1") == 0);
        CHECK(LogHas(log, "AMD provider: chose "));
        CHECK(LogHas(log, "amd_fidelityfx_dx12.dll (the game's amd_fidelityfx_dx12.dll)"));
        CHECK(LogHas(log, "AMD's versions: fake AMD FSR 3.1 (0xfa4e)"));
        // The host selects AMD's version: create, dispatch and destroy all reach the game's DLL
        ffxOverrideVersion version{};
        version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        version.versionId = ids[own];
        ffxCreateContextDescUpscale up{};
        up.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        up.header.pNext = &version.header;
        ffxContext context = nullptr;
        CHECK(create(&context, &up.header, nullptr) == FFX_API_RETURN_OK && context);
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        CHECK(dispatch(&context, &d.header) == FFX_API_RETURN_OK);
        CHECK(destroy(&context, nullptr) == FFX_API_RETURN_OK && !context);
        HMODULE amd = GetModuleHandleW(fake.c_str());
        auto fakeLog = amd ? reinterpret_cast<const char* (*)()>(reinterpret_cast<void*>(GetProcAddress(amd, "fakeLog"))) : nullptr;
        CHECK(fakeLog != nullptr);
        if (fakeLog) {
            const std::string calls = fakeLog();
            CHECK(calls.find("create ctx=") != std::string::npos);
            CHECK(calls.find("dispatch ctx=") != std::string::npos);
            CHECK(calls.find("destroy ctx=") != std::string::npos);
        }
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS: hosted, %s\n", game ? "only the game's amd_fidelityfx_dx12.dll: AMD's version listed and run on it; redirected and hooked modules refused"
                                           : "no AMD DLL: WrongGPU alone, the log says FSR 4 / AMD versions not available");
    return 0;
}
