// SPDX-License-Identifier: MIT
// Original regression: a host creates many distinct command-list wrapper tables.
// Calls the real production patcher and dispatch lookup; no graphics device is needed.
#include "../src/d3d12_hooks.cpp"
#include <array>
#include <memory>
#include <thread>
namespace wronggpu { const char* const kLogTag = "vtable growth regression"; }

namespace wgwin {
struct GrowthObject { void** vtable; };
using GrowthMethod = int (*)(GrowthObject*);
VtblFamily growthFamily;
int GrowthIdentity(GrowthObject*) { return 0; }
int GrowthNativeA(GrowthObject*) { return 31; }
int GrowthNativeB(GrowthObject*) { return 47; }
int GrowthHook(GrowthObject* object) {
    auto original = Orig<GrowthMethod>(growthFamily, object, 1);
    return original ? original(object) : -1;
}
int RunGrowthRegression() {
    std::vector<std::unique_ptr<void*[]>> tables;
    std::vector<SlotHook> hooks{{1, reinterpret_cast<void*>(&GrowthHook)}};
    for (int i = 0; i < 256; ++i) {
        auto table = std::make_unique<void*[]>(kMaxSlots);
        table[0] = reinterpret_cast<void*>(&GrowthIdentity);
        table[1] = reinterpret_cast<void*>(i % 2 ? &GrowthNativeB : &GrowthNativeA);
        {
            std::lock_guard<std::mutex> guard(g_patchMutex);
            if (!Patch(growthFamily, table.get(), hooks, "growth regression")) {
                std::printf("FAIL: host wrapper table %d rejected; every table must be tracked\n", i + 1);
                return 1;
            }
        }
        GrowthObject object{table.get()};
        if (reinterpret_cast<GrowthMethod>(table[1])(&object) != (i % 2 ? 47 : 31)) return 2;
        tables.push_back(std::move(table));
    }
    // Readers of old records must remain valid while new records are published.
    std::atomic<bool> stop{false};
    std::atomic<int> wrong{0};
    std::thread reader([&] {
        while (!stop.load()) {
            for (int i = 0; i < 256; ++i) {
                GrowthObject object{tables[i].get()};
                if (GrowthHook(&object) != (i % 2 ? 47 : 31)) ++wrong;
            }
        }
    });
    std::vector<std::unique_ptr<void*[]>> newTables;
    bool added = true;
    for (int i = 0; i < 256; ++i) {
        auto table = std::make_unique<void*[]>(kMaxSlots);
        table[0] = reinterpret_cast<void*>(&GrowthIdentity);
        table[1] = reinterpret_cast<void*>(&GrowthNativeA);
        std::lock_guard<std::mutex> guard(g_patchMutex);
        added = Patch(growthFamily, table.get(), hooks, "concurrent growth") && added;
        newTables.push_back(std::move(table));
    }
    stop.store(true); reader.join();
    if (!added || wrong.load() || growthFamily.count.load() != 512) return 3;
    // A host-owned source table can disappear. Clone recovery must use saved metadata,
    // rather than dereference the old, now inaccessible host table.
    VtblFamily retired;
    auto source = static_cast<void**>(VirtualAlloc(nullptr, sizeof(void*) * kMaxSlots, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!source) return 4;
    source[0] = reinterpret_cast<void*>(&GrowthIdentity);
    source[1] = reinterpret_cast<void*>(&GrowthNativeA);
    if (!Patch(retired, source, hooks, "retired source")) return 5;
    std::array<void*, kMaxSlots> clone{};
    std::memcpy(clone.data(), source, sizeof(void*) * kMaxSlots);
    VirtualFree(source, 0, MEM_RELEASE);
    if (!Patch(retired, clone.data(), hooks, "clone after source freed")) return 6;
    GrowthObject cloned{clone.data()};
    if (Original(retired, &cloned, 1) != reinterpret_cast<void*>(&GrowthNativeA)) return 7;
    std::puts("PASS: 512 distinct tables; originals remain correct across concurrent growth");
    std::puts("PASS: inherited hook recovered after host source table was freed");
    return 0;
}
}
int main() { return wgwin::RunGrowthRegression(); }
