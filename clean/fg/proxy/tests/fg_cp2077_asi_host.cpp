// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): a test host built under the Cyberpunk 2077 exe's file name, so the
// shipped plugin (WrongGPU-FG.asi) selects that title's binding. No game interface module is loaded in this process:
//   fg_cp2077_asi_host <path of WrongGPU-FG.asi> on    fg=on: the binding is selected and logged; no FG model or
//                                                       kernel folder: nothing is installed or patched
//   fg_cp2077_asi_host <path of WrongGPU-FG.asi> off   fg=off: nothing is read, no transport
// The settings file comes from WG_STATE_GAME_DIR (set by ctest before the process starts); the host empties the
// plugin's log file there first (a file of this test's own case folder).
// Written for this project.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>
#include <vector>

#include "../win32/fg_asi_transport.h"
#include "../../../wgshared/include/wg_host_menu_abi.h"

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

int (*g_query)(WgFgAsiStatus*) = nullptr;
bool g_allow_claim = false;
int g_claims = 0, g_sections = 0, g_releases = 0;
bool g_claimed = false;

struct DrawEntry {
    std::string title;
    WgHostDrawFn draw;
    void* user;
};
std::vector<DrawEntry> g_draw_entries;

bool RegisterCard(WgHostMenuPlugin*, const char* title, WgHostDrawFn draw, void* user) {
    g_draw_entries.push_back({title, draw, user});
    return draw != nullptr;
}

bool RegisterSection(WgHostMenuPlugin*, WgHostDrawFn draw, void* user) {
    g_draw_entries.push_back({"Frame Gen", draw, user});
    ++g_sections;
    return draw != nullptr;
}

bool Report(WgHostMenuPlugin*, const char* name, const char* state, const char*, const char*, float, float, bool) {
    if (!name) {
        CHECK(g_claimed);
        g_claimed = false;
        ++g_releases;
        return true;
    }
    if (!g_claimed && state && std::strcmp(state, "Waiting") == 0 && ++g_claims == 1) {
        WgFgAsiStatus st = {};
        st.struct_size = sizeof(st);
        CHECK(g_query(&st) == 1 && st.transport_created == 0 && st.patched_slots == 0);
    }
    g_claimed = g_allow_claim;
    return g_claimed;
}

WgHostMenuApi HostV6() {
    WgHostMenuApi api{};
    api.size = sizeof(api);
    api.version = 6;
    api.section = [](const char*) {};
    api.text = [](const char*) {};
    api.text_disabled = api.text;
    api.text_colored = [](float, float, float, float, const char*) {};
    api.item_tooltip = api.text;
    api.combo = [](const char*, int32_t*, const char* const*, int32_t) { return false; };
    api.checkbox = [](const char*, bool*) { return false; };
    api.register_custom_card = &RegisterCard;
    api.register_frame_gen_section = &RegisterSection;
    api.report_frame_generation = &Report;
    return api;
}

int TestRegistration(bool v6, int count, wchar_t** paths) {
    auto api = HostV6();
    if (!v6) {
        api.version = 5;
        api.size = offsetof(WgHostMenuApi, report_frame_generation);
    }
    std::vector<std::string> expected;
    for (int i = 0; i < count; ++i) {
        const auto leaf = std::filesystem::path(paths[i]).filename();
        const HMODULE module = LoadLibraryW(paths[i]);
        CHECK(module);
        if (!module) return 1;
        const auto init = reinterpret_cast<WgHostMenuInitFn>(GetProcAddress(module, WG_HOST_MENU_INIT_EXPORT));
        CHECK(init);
        if (!init) return 1;
        if (leaf == L"WrongGPU-FG.asi") {
            if (v6) {
                const auto prepare = reinterpret_cast<WgHostPrepareAsiFn>(GetProcAddress(module, WG_HOST_MENU_PREPARE_ASI_EXPORT));
                CHECK(prepare);
                if (!prepare) return 1;
                prepare();
            }
            const auto asi_init = reinterpret_cast<void (*)()>(GetProcAddress(module, "InitializeASI"));
            CHECK(asi_init);
            if (!asi_init) return 1;
            asi_init();  // the fixture settings have FG off; no transport or hooks
            expected.push_back(v6 ? "Frame Gen" : "WrongGPU Frame Generation");
        } else if (leaf == L"WrongGPU-RR.asi") {
            expected.push_back("WrongGPU");
        } else {
            CHECK(leaf == L"WrongGPU.asi");
            expected.push_back("WrongGPU Upscaler");
        }
        CHECK(init(&api, reinterpret_cast<WgHostMenuPlugin*>(module)));
    }

    std::sort(expected.begin(), expected.end());
    std::vector<std::string> actual;
    for (const auto& entry : g_draw_entries) {
        actual.push_back(entry.title);
        CHECK(entry.draw);
        if (entry.draw) entry.draw(entry.user);
    }
    std::sort(actual.begin(), actual.end());
    CHECK(actual == expected);
    CHECK(g_claims == 0);
    std::printf("card registration v%d, %d plugins: %d checks, %d failed\n", v6 ? 6 : 5, count, g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

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
    if (argc >= 3 && std::wcscmp(argv[1], L"--registration") == 0)
        return TestRegistration(std::wcscmp(argv[2], L"6") == 0, argc - 3, argv + 3);
    if (argc != 3 && argc != 4) return 2;
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
    const bool hosted = leg == L"claim" || leg == L"conflict" || leg == L"truncated";
    if (hosted) {
        const auto prepare = reinterpret_cast<WgHostPrepareAsiFn>(GetProcAddress(asi, WG_HOST_MENU_PREPARE_ASI_EXPORT));
        CHECK(prepare);
        if (!prepare) return 1;
        prepare();
    }
    if (leg == L"nonplugin") {
        if (argc != 4) return 2;
        const HMODULE loader = LoadLibraryW(argv[3]);
        CHECK(loader);
        if (!loader) return 1;
        auto load_asi = reinterpret_cast<DWORD (*)(void (*)())>(GetProcAddress(loader, "LoadLegacyAsi"));
        CHECK(load_asi);
        if (!load_asi) return 1;
        CHECK(load_asi(init) == GetCurrentThreadId());
        FreeLibrary(loader);
    } else {
        init();
    }
    WgFgAsiStatus st = {};
    st.struct_size = sizeof(st);
    CHECK(query(&st) == 1);
    CHECK(st.patched_slots == 0 && st.init_calls == 1);
    auto api = HostV6();
    if (hosted) {
        CHECK(st.settings_enabled == 1 && st.transport_created == 0 && ReadAll(log).find("title binding") == std::string::npos);
        const auto plugin_init = reinterpret_cast<WgHostMenuInitFn>(GetProcAddress(asi, WG_HOST_MENU_INIT_EXPORT));
        CHECK(plugin_init);
        if (!plugin_init) return 1;
        g_query = query;
        g_allow_claim = leg == L"claim";
        if (leg == L"truncated") {
            api.size = offsetof(WgHostMenuApi, report_frame_generation);
            api.register_custom_card = [](WgHostMenuPlugin*, const char*, WgHostDrawFn, void*) { return true; };
        }
        const auto plugin = reinterpret_cast<WgHostMenuPlugin*>(&api);
        CHECK(plugin_init(&api, plugin));
        CHECK(plugin_init(&api, plugin));  // a repeated host call reuses the first registration and claim
        CHECK(g_claims == (leg == L"truncated" ? 0 : 1));
        CHECK(g_sections == (leg == L"truncated" ? 0 : 1));
        CHECK(g_releases == (leg == L"claim" ? 1 : 0) && !g_claimed);
        CHECK(query(&st) == 1 && st.transport_created == 0 && st.patched_slots == 0);
    }
    const std::string text = ReadAll(log);
    if (leg == L"on" || leg == L"claim" || leg == L"nonplugin") {
        // the binding was taken; the case folder (LOCALAPPDATA redirected by ctest) has no FG model and the plugin's
        // folder no kernel folder, so the session host's precheck refuses before any transport exists
        CHECK(st.settings_enabled == 1 && st.transport_created == 0);
        CHECK(st.result == kFgRuntimeMissing);
        CHECK(text.find("title binding cp2077-2.14.1 (Cyberpunk 2077)") != std::string::npos);
        CHECK(text.find("no title binding") == std::string::npos);
        CHECK(text.find("no hooks installed") != std::string::npos);
    } else if (leg == L"conflict" || leg == L"truncated") {
        CHECK(text.find("title binding") == std::string::npos);
        CHECK(text.find(leg == L"conflict" ? "could not reserve" : "no exclusive legacy FG route") != std::string::npos);
    } else if (leg == L"off") {
        CHECK(st.result == kFgInstallOff && st.settings_enabled == 0 && st.transport_created == 0);
        CHECK(text.find("title binding") == std::string::npos);  // nothing was even selected
    } else {
        CHECK(false);
    }
    shutdown();
    CHECK(g_releases == (leg == L"claim" ? 1 : 0));
    std::printf("cp2077 asi leg %ls: result %u (%s); %d checks, %d failed\n", leg.c_str(), st.result,
                FgInstallResultWord(st.result), g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
