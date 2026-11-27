// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): a test host built under the Cyberpunk 2077 exe's file name, so the
// shipped plugin (WrongGPU-FG.asi) selects that title's binding. No game interface module is loaded in this process:
//   fg_cp2077_asi_host <path of WrongGPU-FG.asi> on    fg=on: the binding is selected and logged; no FG model or
//                                                       kernel folder: nothing is installed or patched
//   fg_cp2077_asi_host <path of WrongGPU-FG.asi> off   fg=off: nothing is read, no transport
// The settings file comes from WG_STATE_GAME_DIR (set by ctest before the process starts); the host empties the
// plugin's log file there first (a file of this test's own case folder).
// Written for this project.
#include <cstdio>
#include <cstring>
#include <string>

#include "../win32/fg_asi_transport.h"

namespace {

struct WgFgAsiStatus {
    uint32_t struct_size, result, settings_enabled, transport_created, patched_slots, init_calls;
};

int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failed;                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

std::wstring LogPath() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetEnvironmentVariableW(L"WG_STATE_GAME_DIR", buf, MAX_PATH * 2);
    if (!n || n >= MAX_PATH * 2) return L"";
    return std::wstring(buf, n) + L"\\wronggpu-fg.log";
}

std::string ReadAll(const std::wstring& path) {
    std::string s;
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return s;
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(h, buf, sizeof buf, &got, nullptr) && got) s.append(buf, got);
    CloseHandle(h);
    return s;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    using namespace wgfgp::win32;
    if (argc != 3) return 2;
    const std::wstring leg = argv[2];
    const std::wstring log = LogPath();
    CHECK(!log.empty());
    if (!log.empty()) {
        HANDLE h = CreateFileW(log.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
    HMODULE asi = LoadLibraryW(argv[1]);
    CHECK(asi != nullptr);
    if (!asi) return 1;
    auto init = reinterpret_cast<void (*)()>(GetProcAddress(asi, "InitializeASI"));
    auto query = reinterpret_cast<int (*)(WgFgAsiStatus*)>(GetProcAddress(asi, "WrongGPU_FgQuery"));
    auto shutdown = reinterpret_cast<void (*)()>(GetProcAddress(asi, "WrongGPU_FgShutdown"));
    CHECK(init && query && shutdown);
    if (!init || !query || !shutdown) return 1;
    init();
    WgFgAsiStatus st = {};
    st.struct_size = sizeof(st);
    CHECK(query(&st) == 1);
    CHECK(st.patched_slots == 0 && st.init_calls == 1);
    const std::string text = ReadAll(log);
    if (leg == L"on") {
        // the binding was taken; the case folder (LOCALAPPDATA redirected by ctest) has no FG model and the plugin's
        // folder no kernel folder, so the session host's precheck refuses before any transport exists
        CHECK(st.settings_enabled == 1 && st.transport_created == 0);
        CHECK(st.result == kFgRuntimeMissing);
        CHECK(text.find("title binding cp2077-2.14.1 (Cyberpunk 2077)") != std::string::npos);
        CHECK(text.find("no title binding") == std::string::npos);
        CHECK(text.find("no hooks installed") != std::string::npos);
    } else if (leg == L"off") {
        CHECK(st.result == kFgInstallOff && st.settings_enabled == 0 && st.transport_created == 0);
        CHECK(text.find("title binding") == std::string::npos);  // nothing was even selected
    } else {
        CHECK(false);
    }
    shutdown();
    std::printf("cp2077 asi leg %ls: result %u (%s); %d checks, %d failed\n", leg.c_str(), st.result,
                FgInstallResultWord(st.result), g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
