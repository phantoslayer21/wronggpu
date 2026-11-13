// SPDX-License-Identifier: MIT
// Original regression for cloned COM vtables inheriting WrongGPU's installed hooks.
// Exercises the production patcher without creating a GPU device.
#include "../src/d3d12_hooks.cpp"
namespace wronggpu { const char* const kLogTag = "vtable clone test"; }

namespace wgwin {
VtblFamily cloneTestFamily;
int cloneTestCalls = 0;
struct CloneTestObject { void** vtable; };
using CloneTestMethod = int (*)(CloneTestObject*);
int CloneTestIdentity(CloneTestObject*) { return 0; }
int CloneTestOtherIdentity(CloneTestObject*) { return 1; }
int CloneTestNative(CloneTestObject*) { ++cloneTestCalls; return 37; }
int CloneTestExternal(CloneTestObject* self) { return CloneTestNative(self) + 1; }
int CloneTestHook(CloneTestObject* self) { return Orig<CloneTestMethod>(cloneTestFamily, self, 1)(self); }
int RunCloneTest() {
    void* base[kMaxSlots] = {};
    base[0] = reinterpret_cast<void*>(&CloneTestIdentity);
    base[1] = reinterpret_cast<void*>(&CloneTestNative);
    std::vector<SlotHook> hooks{{1, reinterpret_cast<void*>(&CloneTestHook)}};
    if (!Patch(cloneTestFamily, base, hooks, "test")) return 1;
    void* cloned[kMaxSlots] = {};
    std::memcpy(cloned, base, sizeof base);
    CloneTestObject object{cloned};
    if (!Patch(cloneTestFamily, cloned, hooks, "test clone")) return 2;
    if (Original(cloneTestFamily, &object, 1) != reinterpret_cast<void*>(&CloneTestNative)) {
        std::puts("FAIL: cloned vtable chains the hook to itself");
        return 3;
    }
    const int value = reinterpret_cast<CloneTestMethod>(cloned[1])(&object);
    if (value != 37 || cloneTestCalls != 1) return 4;
    void* unknown[kMaxSlots] = {};
    std::memcpy(unknown, cloned, sizeof cloned);
    unknown[0] = reinterpret_cast<void*>(&CloneTestOtherIdentity);
    if (Patch(cloneTestFamily, unknown, hooks, "unknown clone") || cloneTestFamily.count.load() != 2) return 5;
    void* external[kMaxSlots] = {};
    external[0] = base[0];
    external[1] = reinterpret_cast<void*>(&CloneTestExternal);
    if (!Patch(cloneTestFamily, external, hooks, "external chain")) return 6;
    CloneTestObject externalObject{external};
    if (reinterpret_cast<CloneTestMethod>(external[1])(&externalObject) != 38 || cloneTestCalls != 2) return 7;
    void* ambiguous[kMaxSlots] = {};
    std::memcpy(ambiguous, cloned, sizeof cloned);
    if (Patch(cloneTestFamily, ambiguous, hooks, "ambiguous clone") || cloneTestFamily.count.load() != 3) return 8;
    std::puts("PASS: inherited hook calls the native method once");
    std::puts("PASS: external hooks stay chained; unknown and ambiguous copies are refused");
    return 0;
}
}
int main() { return wgwin::RunCloneTest(); }
