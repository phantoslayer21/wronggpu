// SPDX-License-Identifier: MIT
// WrongGPU.asi: an OptiScaler plugin (its own DLL, our own code only) that makes WrongGPU appear as its own entry
// ("WrongGPU"; with Model K: "WrongGPU Model M", "WrongGPU Model K") in OptiScaler's MAIN DX12 upscaler dropdown,
// leaving OptiScaler's "FSR 3.X(/4)" entry to mean AMD's upscaler again. WrongGPU is the default entry; the player's
// choice is remembered in wronggpu-asi.ini next to the plugin.
//
// How: OptiScaler calls our exported InitializeASI() after loading us (see its LoadAsiPlugins). The module that
// called us is OptiScaler itself (its image holds the menu code), so we identify it by InitializeASI's return
// address, resolve the menu anchors by content (opti_resolver.h), and redirect the `call` at each Selectable site in
// AddDx12Backends to a thunk of ours (asi_thunk.h) through a small absolute-jump stub placed within reach of the
// module. The thunk draws our two items and, on a click, returns "clicked" for OptiScaler's own FSR 3.X entry so
// OptiScaler's unchanged code performs the backend switch; we write none of OptiScaler's data. We remember which
// model the player chose and hand it to the front end through WrongGpuAsiQuery. If any anchor is missing or
// ambiguous (a different OptiScaler build), we patch nothing, report "not hooked", and log why.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "asi_abi.h"
#include "asi_lifetime.h"
#include "asi_log.h"
#include "asi_patch.h"
#include "asi_status.h"
#include "asi_thunk.h"
#include "opti_module.h"
#include "opti_resolver.h"

#if defined(_MSC_VER)
#include <intrin.h>
#define WG_RETURN_ADDRESS() _ReturnAddress()
#else
#define WG_RETURN_ADDRESS() __builtin_return_address(0)
#endif

namespace {

using wgasi::AsiLog;

wgasi::PluginState g_state;
wgasi::DiagnosticState g_diagnostics;

// The real ImGui functions resolved in OptiScaler's image.
using PfnSelectable = bool(*)(const char* label, bool selected, int flags, const void* size);
using PfnBeginCombo = bool(*)(const char* label, const char* preview, int flags);
PfnSelectable g_realSelectable = nullptr;
PfnBeginCombo g_realBeginCombo = nullptr;

std::once_flag g_initOnce;

// Pin only this module, by an address in our own code. Windows keeps a pinned module loaded
// until process termination; there is no supported unpin, including after patch failure.
// Kept out of DllMain and query exports so a private ABI-only load remains unloadable.
bool PinOwnModule() noexcept {
    HMODULE pinned = nullptr;
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                              reinterpret_cast<LPCWSTR>(&PinOwnModule), &pinned) != 0;
}

// The thunks OptiScaler's patched call sites reach (through the near stub). extern "C": the Microsoft x64 calling
// convention, matching ImGui's compiled Selectable / BeginCombo.
// The remembered choice: wronggpu-asi.ini in PluginDataDir() (asi_thunk.h, ParseSelection / FormatSelection).
std::wstring SelectionPath() { return wgasi::PluginDataDir() + L"wronggpu-asi.ini"; }

char LoadSelection() {
    std::string text;
    HANDLE h = CreateFileW(SelectionPath().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        char buf[512];
        DWORD got = 0;
        if (ReadFile(h, buf, sizeof buf, &got, nullptr)) text.assign(buf, got);
        CloseHandle(h);
    }
    return wgasi::ParseSelection(text);
}

void SaveSelection(char model) {
    const std::string text = wgasi::FormatSelection(model);
    HANDLE h = CreateFileW(SelectionPath().c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(h, text.data(), DWORD(text.size()), &written, nullptr);
    CloseHandle(h);
}

extern "C" bool WgSelectableThunk(const char* label, bool selected, int flags, const void* size) {
    const char before = g_state.model.load();
    const bool clicked = wgasi::SelectableThunkLogic(
        [](const char* l, bool s, int f, const void* z) { return g_realSelectable(l, s, f, z); },
        label, selected, flags, size, g_state);
    const char after = g_state.model.load();
    if (after != before) {  // a click on another entry of the FSR 3.X group: remember it (rare; on the menu thread)
        SaveSelection(after);
        AsiLog::Get().Writef("selected: %s", after == wgasi::kModelAmd ? label : wgasi::LabelOf(after));
    }
    return clicked;
}
extern "C" bool WgBeginComboThunk(const char* label, const char* preview, int flags) {
    return wgasi::BeginComboThunkLogic(
        [](const char* l, const char* p, int f) { return g_realBeginCombo(l, p, f); }, label, preview, flags, g_state);
}

uint32_t SizeOfImage(const uint8_t* base) {
    uint32_t peoff;
    std::memcpy(&peoff, base + 0x3C, 4);
    uint32_t sz;
    std::memcpy(&sz, base + peoff + 24 + 56, 4);
    return sz;
}

// A page of executable memory within +-2GB of `anchor` (so a rel32 call from the module can reach a stub in it).
uint8_t* AllocNear(const uint8_t* anchor) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t gran = si.dwAllocationGranularity;
    const uint64_t a = reinterpret_cast<uint64_t>(anchor);
    for (uint64_t delta = gran; delta < (uint64_t(1) << 31); delta += gran) {
        for (int dir = 0; dir < 2; ++dir) {
            const uint64_t addr = dir ? a + delta : a - delta;
            if (addr < gran) continue;
            void* p = VirtualAlloc(reinterpret_cast<void*>(addr & ~(gran - 1)), 0x1000,
                                   MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
            if (p) return static_cast<uint8_t*>(p);
        }
    }
    return nullptr;
}

// Write a `mov rax, imm64; jmp rax` stub at `dst` jumping to `fn`. 12 bytes.
void WriteAbsJmp(uint8_t* dst, const void* fn) {
    dst[0] = 0x48; dst[1] = 0xB8;
    uint64_t t = reinterpret_cast<uint64_t>(fn);
    std::memcpy(dst + 2, &t, 8);
    dst[10] = 0xFF; dst[11] = 0xE0;
}

// Capture every original protection before any call-site write. Refuse operands spanning different
// VirtualQuery regions; no guessed protection restoration and no dynamic allocation after publication.
struct CallWriter {
    uint8_t* image;
    const wgasi::CallPatchTransaction& transaction;
    std::array<DWORD, wgasi::CallPatchTransaction::kCapacity> protections{};

    bool Prepare() {
        for (size_t i = 0; i < transaction.count(); ++i) {
            const uint8_t* operand = image + transaction.patch(i).offset + 1;
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(operand, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT ||
                (info.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return false;
            const uintptr_t within = reinterpret_cast<uintptr_t>(operand) -
                                     reinterpret_cast<uintptr_t>(info.BaseAddress);
            if (within > info.RegionSize || info.RegionSize - within < 4) return false;
            const DWORD executable = info.Protect & 0xff;
            if (executable != PAGE_EXECUTE_READ && executable != PAGE_EXECUTE_READWRITE &&
                executable != PAGE_EXECUTE_WRITECOPY) return false;
            protections[i] = info.Protect;
        }
        return true;
    }
    wgasi::CallPatchTransaction::WriteResult operator()(size_t i, const uint8_t* bytes,
                                                       const uint8_t* expected) noexcept {
        uint8_t* site = image + transaction.patch(i).offset;
        if (std::memcmp(site, expected, 5)) return {false, 0};
        DWORD ignored = 0;
        if (!VirtualProtect(site + 1, 4, PAGE_EXECUTE_READWRITE, &ignored)) return {false, 0};
        // Protection changes are a scheduling window: recheck before the actual byte stores.
        // This narrows the window but cannot establish exclusion from other threads.
        if (std::memcmp(site, expected, 5)) {
            VirtualProtect(site + 1, 4, protections[i], &ignored);
            return {false, 0};
        }
        std::memcpy(site + 1, bytes, 4);
        const bool protectedAgain = VirtualProtect(site + 1, 4, protections[i], &ignored) != 0;
        const bool flushed = FlushInstructionCache(GetCurrentProcess(), site, 5) != 0;
        return {protectedAgain && flushed, 0x0f};
    }
};
// Find OptiScaler's module (opti_module.h: the caller of InitializeASI, then modules whose version resource names
// OptiScaler, then modules loaded under one of its install names; never the plugin, the executable or a system DLL)
// and patch its DX12 upscaler menu. The first candidate the resolver recognises is used; every candidate tried and why
// it was a candidate is logged.
void Hook(const void* caller) {
    AsiLog& log = AsiLog::Get();
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&Hook), &self);
    std::vector<std::string> skipped;
    const std::vector<wgasi::ModuleCandidate> candidates = wgasi::OptiScalerCandidates(caller, self, &skipped);
    for (const std::string& s : skipped) log.Write("host module candidate " + s);
    const uint8_t* optiBase = nullptr;
    wgasi::Resolved r;
    for (const wgasi::ModuleCandidate& c : candidates) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(c.module);
        wgasi::PeImage img(base, SizeOfImage(base));
        wgasi::Resolved rc;
        std::string why;
        if (wgasi::Resolve(img, rc, [&](const std::string& s) { why = s; }) && rc.ok) {
            log.Write("host module " + wgasi::Utf8(c.path) + " (" + c.why + "): OptiScaler's DX12 upscaler menu recognised");
            optiBase = base;
            r = rc;
            break;
        }
        log.Write("host module candidate " + wgasi::Utf8(c.path) + " (" + c.why + "): not OptiScaler's menu" +
                  (why.empty() ? std::string() : " (" + why + ")"));
    }
    if (!optiBase) {
        g_diagnostics.Outcome(WG_ASI_MENU_UNAVAILABLE, WG_ASI_REASON_HOST_NOT_RECOGNIZED);
        log.Writef("not hooked: OptiScaler's DX12 upscaler menu was not recognised in %zu candidate module(s) (no "
                   "OptiScaler loaded, or this OptiScaler build differs). WrongGPU stays in the FFX sub-box; nothing "
                   "was patched.", candidates.size());
        return;
    }
    g_realSelectable = reinterpret_cast<PfnSelectable>(const_cast<uint8_t*>(optiBase) + r.selectable);
    g_realBeginCombo = r.beginCombo ? reinterpret_cast<PfnBeginCombo>(const_cast<uint8_t*>(optiBase) + r.beginCombo) : nullptr;

    // Selection I/O and all allocations happen before any call site can enter our thunks.
    g_state.model.store(LoadSelection());
    uint8_t* stubs = AllocNear(optiBase + r.addDx12Lo);
    if (!stubs) {
        g_diagnostics.Outcome(WG_ASI_MENU_UNAVAILABLE, WG_ASI_REASON_STUB_ALLOCATION_FAILED);
        log.Write("not hooked: could not allocate a stub page within reach of OptiScaler"); return;
    }
    WriteAbsJmp(stubs, reinterpret_cast<void*>(&WgSelectableThunk));
    WriteAbsJmp(stubs + 16, reinterpret_cast<void*>(&WgBeginComboThunk));

    uint8_t* image = const_cast<uint8_t*>(optiBase);
    wgasi::CallPatchTransaction transaction(image, SizeOfImage(optiBase));
    bool prepared = !r.selectableSites.empty();
    for (uint32_t site : r.selectableSites)
        if (!transaction.Add(site, r.selectable, reinterpret_cast<uintptr_t>(stubs))) prepared = false;
    const bool previewAvailable = r.beginComboSite && g_realBeginCombo;
    if (previewAvailable && !transaction.Add(r.beginComboSite, r.beginCombo,
                                             reinterpret_cast<uintptr_t>(stubs + 16))) prepared = false;
    CallWriter writer{image, transaction};
    if (!prepared || !writer.Prepare() || !FlushInstructionCache(GetCurrentProcess(), stubs, 28)) {
        VirtualFree(stubs, 0, MEM_RELEASE); // No published target exists yet.
        g_diagnostics.Outcome(WG_ASI_MENU_UNAVAILABLE, WG_ASI_REASON_PATCH_FAILED);
        log.Write("not hooked: call-patch preparation failed; no call site was written");
        return;
    }
    // No host call site can reach a thunk before its owning module is permanently resident.
    // This is deliberately after host recognition and preparation, but before any host write.
    const auto publication = wgasi::PublishPinnedCalls(transaction, PinOwnModule, writer);
    if (!publication.modulePinned) {
        VirtualFree(stubs, 0, MEM_RELEASE); // No writer was invoked; nothing was published.
        g_diagnostics.Outcome(WG_ASI_MENU_UNAVAILABLE, WG_ASI_REASON_MODULE_PIN_FAILED);
        log.Write("not hooked: could not pin WrongGPU.asi for process lifetime; no call site was written");
        return;
    }
    const auto result = publication.patch;
    g_state.hooked.store(result.committed());
    if (!result.committed()) {
        const bool rollbackFailed = result.state == wgasi::CallPatchTransaction::State::RollbackFailed;
        g_diagnostics.Outcome(rollbackFailed ? WG_ASI_MENU_PARTIAL : WG_ASI_MENU_UNAVAILABLE,
                              rollbackFailed ? WG_ASI_REASON_ROLLBACK_FAILED : WG_ASI_REASON_PATCH_FAILED);
        if (result.attempted == 0) VirtualFree(stubs, 0, MEM_RELEASE);
        // Once attempted, retain stubs even after restored bytes: another thread may already be in a stub.
        // Our module stays pinned too. No host lifetime, thread exclusion or in-flight call drain
        // is established here; pinning is not a safe live-unhook/unload implementation.
        log.Write(rollbackFailed ? "not hooked: call-patch rollback failed; bytes/protection/cache state may be partial; stubs retained" :
                  "not hooked: patch transaction refused or failed; all attempted own writes restored (if any)");
        return;
    }
    g_diagnostics.Outcome(previewAvailable ? WG_ASI_MENU_PATCHED : WG_ASI_MENU_PARTIAL,
                          previewAvailable ? WG_ASI_REASON_NONE : WG_ASI_REASON_INCOMPLETE_PATCHES);
    log.Writef("hooked: all %zu prepared call sites patched; preview relabel %s. Selected: %s",
               transaction.count(), previewAvailable ? "on" : "unavailable",
               g_state.model.load() == wgasi::kModelAmd ? "AMD (FSR 3.X)" : wgasi::LabelOf(g_state.model.load()));
}

}  // namespace

// The return address is taken here, in InitializeASI's own frame (not inside the call_once lambda, whose return
// address is the plugin's own std::call_once code: the 0.5.0 in-game bug), and is only the first candidate.
extern "C" __declspec(dllexport) __declspec(noinline) void InitializeASI() {
    const void* caller = WG_RETURN_ADDRESS();
    std::call_once(g_initOnce, [caller] {
        g_diagnostics.Begin();
        AsiLog::Get().Write("InitializeASI");
        Hook(caller);
        g_diagnostics.Complete();
    });
}

extern "C" __declspec(dllexport) int WrongGpuAsiQuery(int* hooked, char* model) {
    if (hooked) *hooked = g_state.hooked.load() ? 1 : 0;
    if (model) *model = g_state.model.load();
    return int(WRONGGPU_ASI_ABI);
}

extern "C" __declspec(dllexport) int WrongGpuAsiQueryStatusV1(WrongGpuAsiStatusV1* status, uint32_t caller_size) {
    return g_diagnostics.Query(status, caller_size, g_state.hooked.load(), g_state.model.load());
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(instance);
        AsiLog::Get().Write("WrongGPU.asi loaded");
    }
    return TRUE;
}
