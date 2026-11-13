// SPDX-License-Identifier: MIT
// In-process test of the WrongGPU OptiScaler plugin's module identification (clean/fsr/asi/opti_module.h) with the
// real, built WrongGPU.asi. Written for this project.
//
//   test_asi_module <WrongGPU.asi> opti [hooked|not-hooked]   (OptiScaler binary: WRONGGPU_OPTISCALER_BIN; unset: 77)
//     A temporary folder gets a copy of that binary named dxgi.dll and the plugin as plugins\WrongGPU.asi, as an
//     install does. The copy is mapped with LoadLibraryEx(DONT_RESOLVE_DLL_REFERENCES): no OptiScaler code runs (no
//     DllMain, no imports). The plugin is loaded and its InitializeASI called from this program (so the caller is not
//     OptiScaler: the plugin has to find the module itself). Expected (default "hooked"): WrongGpuAsiQuery reports
//     hooked, and wronggpu-asi.log names the dxgi.dll copy as the host module.
//   test_asi_module <WrongGPU.asi> system
//     Only the system dxgi.dll is loaded: not hooked, the system DLL and this program are skipped.
// Both: the plugin never scans its own image (the 0.5.0 in-game bug: "host module ...\plugins\WrongGPU.asi").
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                                                        \
    do {                                                                                                   \
        ++g_checks;                                                                                        \
        if (!(cond)) { ++g_failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); }      \
    } while (0)

static std::wstring Widen(const std::string& s) {
    std::wstring w(s.size() + 1, L'\0');
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], int(w.size()));
    w.resize(n > 0 ? size_t(n - 1) : 0);
    return w;
}

static std::string Narrow(const std::wstring& w) {
    std::string s(w.size() * 3 + 1, '\0');
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], int(s.size()), nullptr, nullptr);
    s.resize(n > 0 ? size_t(n - 1) : 0);
    return s;
}

static std::string Lower(std::string s) {
    for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: test_asi_module <WrongGPU.asi> opti [hooked|not-hooked] | system\n");
        return 2;
    }
    const std::wstring plugin = Widen(argv[1]);
    const std::string mode = argv[2];
    const bool expectHooked = mode == "opti" && !(argc > 3 && std::strcmp(argv[3], "not-hooked") == 0);
    std::wstring opti;
    if (mode == "opti") {
        const char* bin = std::getenv("WRONGGPU_OPTISCALER_BIN");
        if (!bin || !*bin || GetFileAttributesW(Widen(bin).c_str()) == INVALID_FILE_ATTRIBUTES) {
            std::printf("SKIP: WRONGGPU_OPTISCALER_BIN is not set or not readable\n");
            return 77;
        }
        opti = Widen(bin);
    } else if (mode != "system") {
        std::printf("unknown mode %s\n", mode.c_str());
        return 2;
    }

    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    const std::wstring dir = std::wstring(tmp) + L"wg-asi-module-" + std::to_wstring(GetCurrentProcessId());
    CreateDirectoryW(dir.c_str(), nullptr);
    CreateDirectoryW((dir + L"\\plugins").c_str(), nullptr);
    const std::wstring asi = dir + L"\\plugins\\WrongGPU.asi", proxy = dir + L"\\dxgi.dll";
    CHECK(CopyFileW(plugin.c_str(), asi.c_str(), FALSE));

    HMODULE host = nullptr;
    if (mode == "opti") {
        CHECK(CopyFileW(opti.c_str(), proxy.c_str(), FALSE));
        host = LoadLibraryExW(proxy.c_str(), nullptr, DONT_RESOLVE_DLL_REFERENCES);  // mapped, never run
        CHECK(host != nullptr);
    } else {
        host = LoadLibraryW(L"dxgi.dll");  // the system's
        CHECK(host != nullptr);
    }
    HMODULE p = LoadLibraryW(asi.c_str());
    CHECK(p != nullptr);
    using PfnInit = void (*)();
    using PfnQuery = int (*)(int*, char*);
    auto init = p ? reinterpret_cast<PfnInit>(reinterpret_cast<void*>(GetProcAddress(p, "InitializeASI"))) : nullptr;
    auto query = p ? reinterpret_cast<PfnQuery>(reinterpret_cast<void*>(GetProcAddress(p, "WrongGpuAsiQuery"))) : nullptr;
    CHECK(init && query);
    int hooked = -1;
    char model = 0;
    if (init && query) {
        init();
        query(&hooked, &model);
    }

    std::ifstream f(dir + L"\\wronggpu-asi.log", std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    f.close();
    const std::string log = ss.str();
    std::printf("---- wronggpu-asi.log\n%s----\n", log.c_str());
    const std::string lowLog = Lower(log), lowAsi = Lower(Narrow(asi)), lowProxy = Lower(Narrow(proxy));

    // The plugin never scans itself: its own path may appear only as a skipped candidate.
    size_t at = 0;
    bool scannedSelf = false;
    while ((at = lowLog.find(lowAsi + " (", at)) != std::string::npos) {
        const size_t eol = lowLog.find('\n', at);
        if (lowLog.substr(at, eol - at).find("skipped, the plugin itself") == std::string::npos) scannedSelf = true;
        at = eol == std::string::npos ? lowLog.size() : eol;
    }
    CHECK(!scannedSelf);

    if (expectHooked) {
        CHECK(hooked == 1);
        CHECK(lowLog.find("host module " + lowProxy + " (") != std::string::npos);
        CHECK(log.find("menu recognised") != std::string::npos && log.find("hooked: ") != std::string::npos);
        CHECK(model == 'M');
    } else {
        CHECK(hooked == 0);
        CHECK(log.find("not hooked") != std::string::npos);
        if (mode == "system") CHECK(log.find("skipped, a system DLL") != std::string::npos);
    }

    if (p) FreeLibrary(p);
    if (host) FreeLibrary(host);
    DeleteFileW(asi.c_str());
    DeleteFileW(proxy.c_str());
    DeleteFileW((dir + L"\\wronggpu-asi.log").c_str());
    RemoveDirectoryW((dir + L"\\plugins").c_str());
    RemoveDirectoryW(dir.c_str());
    std::printf("%s: hooked=%d; %d checks, %d failures\n", mode.c_str(), hooked, g_checks, g_failures);
    return g_failures ? 1 : 0;
}
