// SPDX-License-Identifier: MIT
// CPU-only WrongGPU diagnostic tests. No InitializeASI call or foreign image inspection.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cstddef>
#include <thread>
#include "../asi/asi_status.h"
#include "../asi/asi_thunk.h"

static int failures = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("FAIL line %d: %s\n", __LINE__, #x); } } while (0)
static_assert(sizeof(WrongGpuAsiStatusV1) == 48, "Fixed v1 diagnostic layout");
static_assert(offsetof(WrongGpuAsiStatusV1, rr_blockers) == 40, "Fixed field offsets");
static_assert(WRONGGPU_ASI_ABI == 1u, "Legacy ABI unchanged");

static WrongGpuAsiStatusV1 Request() {
    WrongGpuAsiStatusV1 s{};
    s.size = sizeof(s); s.version = WRONGGPU_ASI_STATUS_VERSION;
    return s;
}
template<class Query> static void CheckProtocol(Query query) {
    CHECK(query(nullptr, 0) == WG_ASI_STATUS_INVALID_ARGUMENT);
    CHECK(query(nullptr, sizeof(WrongGpuAsiStatusV1)) == WG_ASI_STATUS_INVALID_ARGUMENT);
    for (uint32_t capacity : {0u, 1u, 7u, 8u, 47u}) {
        auto s = Request(); const auto before = s;
        CHECK(query(&s, capacity) == WG_ASI_STATUS_INVALID_SIZE);
        CHECK(std::memcmp(&s, &before, sizeof(s)) == 0);
    }
    for (uint32_t size : {0u, 7u, 8u, 47u, 49u, UINT32_MAX}) {
        auto s = Request(); s.size = size; const auto before = s;
        CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_INVALID_SIZE);
        CHECK(std::memcmp(&s, &before, sizeof(s)) == 0);
    }
    for (uint32_t version : {0u, 2u, UINT32_MAX}) {
        auto s = Request(); s.version = version; const auto before = s;
        CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_UNSUPPORTED_VERSION);
        CHECK(std::memcmp(&s, &before, sizeof(s)) == 0);
    }
    struct Extended { WrongGpuAsiStatusV1 s; uint32_t sentinel; } ext{Request(), 0x12345678};
    ext.s.size = sizeof(ext);
    CHECK(query(&ext.s, sizeof(ext)) == WG_ASI_STATUS_OK);
    CHECK(ext.s.size == sizeof(ext.s) && ext.s.version == 1);
    CHECK(ext.sentinel == 0x12345678);
    CHECK(ext.s.loaded == 1 && ext.s.rr_ready == 0 && ext.s.reserved == 0);
    CHECK(ext.s.effective_backend == WG_ASI_EFFECTIVE_BACKEND_UNKNOWN);
    CHECK(ext.s.rr_blockers == (WG_ASI_RR_MISSING_INTERFACE | WG_ASI_RR_MISSING_F_CONTRACTS |
                               WG_ASI_RR_BACKEND_UNAVAILABLE | WG_ASI_RR_HOST_UNQUALIFIED));
}

static void TestFake() {
    wgasi::DiagnosticState state;
    wgasi::PluginState plugin;
    auto query = [&](WrongGpuAsiStatusV1* s, uint32_t n) {
        return state.Query(s, n, plugin.hooked.load(), plugin.model.load());
    };
    CheckProtocol(query);
    auto s = Request(); CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_OK);
    CHECK(s.initialization == WG_ASI_INIT_NOT_ATTEMPTED && s.hooked == 0);
    CHECK(s.menu_state == WG_ASI_MENU_UNAVAILABLE && s.menu_reason == WG_ASI_REASON_NOT_INITIALIZED);
    CHECK(s.selected_model == 'M');
    state.Begin(); CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_OK);
    CHECK(s.initialization == WG_ASI_INIT_IN_PROGRESS && s.menu_reason == WG_ASI_REASON_INITIALIZING);
    for (uint32_t reason : {WG_ASI_REASON_HOST_NOT_RECOGNIZED, WG_ASI_REASON_STUB_ALLOCATION_FAILED,
                            WG_ASI_REASON_PATCH_FAILED, WG_ASI_REASON_MODULE_PIN_FAILED}) {
        state.Outcome(WG_ASI_MENU_UNAVAILABLE, reason); state.Complete();
        CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_OK);
        CHECK(s.initialization == WG_ASI_INIT_COMPLETED && s.hooked == 0 && s.menu_reason == reason);
        CHECK(s.rr_ready == 0);
    }
    plugin.hooked.store(true);
    for (char model : {char(0), 'M', 'K'}) {
        plugin.model.store(model);
        state.Outcome(WG_ASI_MENU_PARTIAL, WG_ASI_REASON_INCOMPLETE_PATCHES);
        CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_OK);
        CHECK(s.hooked == 1 && s.menu_state == WG_ASI_MENU_PARTIAL);
        CHECK(s.menu_reason == WG_ASI_REASON_INCOMPLETE_PATCHES && s.selected_model == uint32_t(model));
        state.Outcome(WG_ASI_MENU_PATCHED, WG_ASI_REASON_NONE);
        CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_OK);
        CHECK(s.menu_state == WG_ASI_MENU_PATCHED && s.menu_reason == WG_ASI_REASON_NONE);
        CHECK(s.rr_ready == 0 && s.effective_backend == WG_ASI_EFFECTIVE_BACKEND_UNKNOWN);
    }
    std::atomic<bool> good{true};
    auto reader = [&] {
        for (int i = 0; i < 10000; ++i) {
            auto v = Request();
            if (query(&v, sizeof(v)) != WG_ASI_STATUS_OK || v.rr_ready || v.effective_backend ||
                (v.selected_model != 'M' && v.selected_model != 'K')) good.store(false);
        }
    };
    std::thread a(reader), b(reader);
    for (int i = 0; i < 10000; ++i) plugin.model.store(i % 2 ? 'M' : 'K');
    a.join(); b.join(); CHECK(good.load());
}

int main(int argc, char** argv) {
    TestFake();
    if (argc != 2) { std::puts("Pass the absolute path to the owned WrongGPU.asi build"); return 2; }
    HMODULE module = LoadLibraryA(argv[1]); CHECK(module != nullptr);
    if (module) {
        auto legacy = reinterpret_cast<PfnWrongGpuAsiQuery>(GetProcAddress(module, WRONGGPU_ASI_QUERY));
        auto query = reinterpret_cast<PfnWrongGpuAsiQueryStatusV1>(GetProcAddress(module, WRONGGPU_ASI_STATUS_QUERY));
        CHECK(legacy != nullptr && query != nullptr);
        CHECK(GetProcAddress(module, "InitializeASI") != nullptr); // Resolve only; never invoke.
        if (legacy && query) {
            CheckProtocol(query);
            CHECK(legacy(nullptr, nullptr) == 1);
            int hooked = -1; char model = '?';
            CHECK(legacy(&hooked, nullptr) == 1 && hooked == 0);
            CHECK(legacy(nullptr, &model) == 1 && model == 'M');
            CHECK(legacy(&hooked, &model) == 1 && hooked == 0 && model == 'M');
            for (int i = 0; i < 3; ++i) {
                auto s = Request(); CHECK(query(&s, sizeof(s)) == WG_ASI_STATUS_OK);
                CHECK(s.loaded == 1 && s.initialization == WG_ASI_INIT_NOT_ATTEMPTED && s.hooked == 0);
                CHECK(s.menu_state == WG_ASI_MENU_UNAVAILABLE && s.menu_reason == WG_ASI_REASON_NOT_INITIALIZED);
                CHECK(s.selected_model == 'M' && s.rr_ready == 0);
            }
        }
        CHECK(FreeLibrary(module) != 0); // No initialization or patches ever occurred.
        CHECK(GetModuleHandleA(argv[1]) == nullptr); // Query-only loading did not permanently pin it.
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
