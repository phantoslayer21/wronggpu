// SPDX-License-Identifier: MIT
// D3D12 hooks for the same-frame split (see d3d12_hooks.h). Written from Microsoft's public D3D12 documentation and
// the Windows SDK / MinGW-w64 declarations of the interfaces (vtable positions in d3d12_vtbl.h, checked at build time).
#include "d3d12_hooks.h"

#include "log_win.h"

#include <d3d12.h>
#include <algorithm>
#if defined(_MSC_VER)
#include <intrin.h>
#define WG_RETURN_ADDRESS() _ReturnAddress()
#else
#define WG_RETURN_ADDRESS() __builtin_return_address(0)
#endif

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <memory>
#include <new>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "d3d12_vtbl.h"
#include "param_map.h"
#include "version.h"
#include "statistics_query.h"
#if defined(WRONGGPU_SPLIT_PROGRESS_QA)
#include "split_progress.h"
#endif

namespace wgwin {
namespace {

using wronggpu::Log;

// ---------------------------------------------------------------------------------------------------------------
// Settings and log

struct Settings {
    int mode = 1;                 // 0 off, 1 auto, 2 on
    uint32_t timeoutMs = 2000;    // split_timeout_ms
    bool logAll = true;           // log_dispatches
    uint32_t logMaxMegabytes = 512;
};
Settings g_settings;

void HLog(const char* fmt, ...) {
    char text[3072];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(text, sizeof text, fmt, args);
    va_end(args);
    Log::Get().Write("d3d12 hooks: %s", text);
}

// Per-frame lines (trace, splits): every one while the log is small, then one in 600 (as for dispatch lines).
bool ShouldLogFrame(uint64_t n) {
    if (!g_settings.logAll) return n < 3;
    if (Log::Get().Bytes() > (uint64_t(g_settings.logMaxMegabytes) << 20)) return n % 600 == 0;
    return true;
}

uint64_t NowUs() {
    static const uint64_t freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return uint64_t(f.QuadPart);
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return uint64_t(c.QuadPart) / freq * 1000000u + uint64_t(c.QuadPart) % freq * 1000000u / freq;
}

#if defined(WRONGGPU_SPLIT_PROGRESS_QA)
std::atomic<bool> g_progressReady{false};
void ProgressLog(void*, const char* text) { HLog("%s",text); }
#endif

const char* ListTypeName(uint32_t t) {
    switch (t) {
    case D3D12_COMMAND_LIST_TYPE_DIRECT: return "DIRECT";
    case D3D12_COMMAND_LIST_TYPE_BUNDLE: return "BUNDLE";
    case D3D12_COMMAND_LIST_TYPE_COMPUTE: return "COMPUTE";
    case D3D12_COMMAND_LIST_TYPE_COPY: return "COPY";
    default: return "other";
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Vtable patching. Each family (device, queue, command list) may have more than one vtable (e.g. different classes
// in the driver or a layer); each patched vtable keeps the entries it had before, and a hook finds its originals
// through the object's vtable pointer.

constexpr int kMaxSlots = WG_VT_LIST_SLOTS;

struct PatchedVtbl {
    void** vtbl = nullptr;
    void* identity = nullptr;  // saved while the host-owned table is accessible
    std::atomic<void*> orig[kMaxSlots]{};
    PatchedVtbl* next = nullptr;
};

struct VtblFamily {
    std::atomic<int> count{0};
    // Append-only records stay at stable addresses. Hook callbacks never take the
    // patch mutex, and growing this list cannot relocate records used by readers.
    std::atomic<PatchedVtbl*> head{nullptr};
    ~VtblFamily() {
        auto* entry = head.load(std::memory_order_relaxed);
        while (entry) { auto* next = entry->next; delete entry; entry = next; }
    }
};

VtblFamily g_deviceVt, g_queueVt, g_listVt;
std::mutex g_patchMutex;

void* Original(VtblFamily& family, const void* self, int slot) {
    void** vt = *static_cast<void** const*>(self);
    PatchedVtbl* oldest = nullptr;
    for (auto* entry = family.head.load(std::memory_order_acquire); entry; entry = entry->next) {
        if (entry->vtbl == vt && entry->identity == vt[0]) return entry->orig[slot].load(std::memory_order_acquire);
        oldest = entry;
    }
    return oldest ? oldest->orig[slot].load(std::memory_order_acquire) : nullptr;
}

template <class F>
F Orig(VtblFamily& family, const void* self, int slot) {
    return reinterpret_cast<F>(Original(family, self, slot));
}

struct SlotHook {
    int slot;
    void* hook;
};

bool IsPatched(VtblFamily& family, void** vtbl) {
    for (auto* entry = family.head.load(std::memory_order_acquire); entry; entry = entry->next)
        if (entry->vtbl == vtbl && entry->identity == vtbl[0]) return true;
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Tables WrongGPU must not write to. A class's vtable is part of a module image and is shared by every object of the
// class. Some hook layers instead give each command list its own table: a copy of the class table (taken when the list
// is created) in which they replace entries with their own functions. Such a table
//   - holds WrongGPU's hooks in the entries the layer left alone when the copy was taken after WrongGPU patched the
//     class table ("inherited"), and the layer's functions in the others ("foreign");
//   - forwards the layer's functions to what the class table held when the copy was taken: with WrongGPU's hooks in
//     the class table that is WrongGPU's hook, so writing WrongGPU's hook over the layer's entry and chaining to the
//     layer makes hook -> layer -> hook -> ... without end (a stack overflow in the game's next call into the list);
//     with the layer's own saved originals instead, the calls made before WrongGPU's hooks were written never reached
//     WrongGPU, so the bindings a split would restore are incomplete;
//   - is new for every list, so it can only be patched when the list first reaches a dispatch, after its Reset.
// Either way such a list cannot be tracked, and resetting it for a split would lose the game's state or recurse. The
// rule: a table outside every module image that carries foreign entries is left untouched, and its list is never
// split (WG_ERROR_NOT_READY from the split; the front end gives the context to AMD's FSR, frontend.cpp).
struct TableFacts {
    bool inImage = false;  // the table is part of a loaded module
    int foreign = 0;       // hooked slots whose entry is neither WrongGPU's hook nor in the module of the first entry
    int inherited = 0;     // hooked slots that already hold WrongGPU's hook
};

constexpr bool OwnedByAnotherLayer(const TableFacts& f) { return !f.inImage && f.foreign > 0; }

constexpr char kForeignTableReason[] =
    "the command list has its own vtable (outside every module) with another hook layer's entries: it cannot be tracked or split safely";

HMODULE ModuleOf(const void* address) {
    HMODULE m = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       static_cast<LPCWSTR>(address), &m);
    return m;
}

// ---------------------------------------------------------------------------------------------------------------
// Wrapper objects (clean/fe-wrapped-list; Spider-Man 2 run G8). Another layer (a game interface's interposer, a capture
// tool) can hand the game, and so WrongGPU's dispatch, a command-list object of its own class: a separate object whose
// class vtable is in that layer's module and whose methods forward to an inner list of the real device (its GetDevice
// may return the layer's device wrapper). Patching that class table too hooks every call twice (on the wrapper, then on
// the inner list) and tracks one logical list as two; a split of it closes and resets the wrapper and puts it on a
// queue made from the wrapper's device. The rules:
//   - a list class table is WrongGPU's to patch only in the modules of the lists the context's device itself creates
//     (PatchListOfType at installation). A class table in any other module is a wrapper's: never patched, never split.
//     Its calls reach the inner list, whose table is patched: the hooks run once per call, on the list that is tracked;
//   - a split happens only on a list whose device is one an upscaler context was created on (Install without a list)
//     and is the D3D12 runtime's own object (vtable in D3D12Core.dll, d3d12.dll or D3D12SDKLayers.dll). Otherwise the
//     split is refused (one log line) and the front end gives the context its safe path (AMD's FSR, else the
//     stand-in, ListSplitBlocked); WrongGPU's queue is never created on any other device.
constexpr int kMaxKnown = 16;
std::atomic<HMODULE> g_listModules[kMaxKnown];       // modules of the class tables of the context devices' own lists
std::atomic<int> g_listModuleCount{0};
std::atomic<void*> g_contextDevices[kMaxKnown];      // devices upscaler contexts were created on
std::atomic<int> g_contextDeviceCount{0};
std::atomic<void*> g_refusedTables[kMaxKnown];       // wrapper class tables whose split was refused (BlockedImpl)
std::atomic<int> g_refusedTableCount{0};
HMODULE g_testRuntimeModule = nullptr;               // tests: the module of their fake D3D12 runtime

constexpr char kWrapperReason[] =
    "the command list is another layer's wrapper object (its class vtable is not in the module of the device's own command lists): "
    "WrongGPU leaves it untouched and does not split it";
constexpr char kNotOurDeviceReason[] =
    "the command list's device is not the D3D12 runtime device an upscaler was created on (a layer's device wrapper, or another device): "
    "WrongGPU does not split it";

template <class T>
bool Known(const std::atomic<T>* list, const std::atomic<int>& count, T value) {
    const int n = count.load(std::memory_order_acquire);
    for (int i = 0; i < n && i < kMaxKnown; ++i)
        if (list[i].load(std::memory_order_acquire) == value) return true;
    return false;
}
template <class T>
void Remember(std::atomic<T>* list, std::atomic<int>& count, T value) {  // under g_patchMutex
    if (!value || Known(list, count, value)) return;
    const int n = count.load(std::memory_order_relaxed);
    for (int i = 0; i < n && i < kMaxKnown; ++i) {  // a slot Forget emptied (bug hunt F3)
        T empty{};
        if (list[i].compare_exchange_strong(empty, value, std::memory_order_acq_rel)) return;
    }
    if (n >= kMaxKnown) return;
    list[n].store(value, std::memory_order_release);
    count.store(n + 1, std::memory_order_release);
}
// Bug hunt F3: an entry no longer wanted (a removed device): its slot is emptied, for Remember to reuse. Lock-free.
template <class T>
void Forget(std::atomic<T>* list, const std::atomic<int>& count, T value) {
    const int n = count.load(std::memory_order_acquire);
    for (int i = 0; i < n && i < kMaxKnown; ++i) {
        T expected = value;
        list[i].compare_exchange_strong(expected, T{}, std::memory_order_acq_rel);
    }
}

// The D3D12 runtime's own modules (the runtime's objects have their class tables there).
bool RuntimeModule(HMODULE m) {
    if (!m) return false;
    if (m == g_testRuntimeModule) return true;
    wchar_t path[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(m, path, MAX_PATH);
    if (!n || n >= MAX_PATH) return false;
    const wchar_t* base = std::wcsrchr(path, L'\\');
    base = base ? base + 1 : path;
    return _wcsicmp(base, L"d3d12core.dll") == 0 || _wcsicmp(base, L"d3d12.dll") == 0 || _wcsicmp(base, L"d3d12sdklayers.dll") == 0;
}

// A list class table in a module image that is not one of the context devices' own list modules.
bool WrapperTable(void** vt) {
    if (!g_listModuleCount.load(std::memory_order_acquire)) return false;
    const HMODULE m = ModuleOf(vt);
    return m && !Known(g_listModules, g_listModuleCount, m);
}

// A device an upscaler context was created on, and the D3D12 runtime's own object.
bool ProvenDevice(void* device) {
    return device && Known(g_contextDevices, g_contextDeviceCount, device) && RuntimeModule(ModuleOf(*static_cast<void**>(device)));
}

TableFacts Inspect(void** vtbl, const std::vector<SlotHook>& hooks) {
    TableFacts f;
    f.inImage = ModuleOf(vtbl) != nullptr;
    const HMODULE first = ModuleOf(vtbl[0]);
    for (const SlotHook& h : hooks) {
        if (vtbl[h.slot] == h.hook) ++f.inherited;
        else if (ModuleOf(vtbl[h.slot]) != first) ++f.foreign;
    }
    return f;
}

// Patches `vtbl` (under g_patchMutex). The originals are published before the first hook is written.
bool Patch(VtblFamily& family, void** vtbl, const std::vector<SlotHook>& hooks, const char* what) {
    if (IsPatched(family, vtbl)) return true;
    std::unique_ptr<PatchedVtbl> record(new (std::nothrow) PatchedVtbl);
    if (!record) { HLog("ERROR: cannot allocate %s hook metadata; table %p not patched", what, vtbl); return false; }
    PatchedVtbl& e = *record;
    e.vtbl = vtbl;
    e.identity = vtbl[0];
    int inherited = 0;
    for (const SlotHook& h : hooks) {
        e.orig[h.slot].store(vtbl[h.slot], std::memory_order_relaxed);
        if (vtbl[h.slot] != h.hook) continue;
        // A host can copy an already patched vtable into a per-object table.
        // Saving our own hook as its original would make the next call recurse.
        // Recover only an unambiguous original with the same COM identity method.
        void* resolved = nullptr;
        for (auto* entry = family.head.load(std::memory_order_acquire); entry; entry = entry->next) {
            const PatchedVtbl& source = *entry;
            void* candidate = source.orig[h.slot].load(std::memory_order_acquire);
            if (source.identity != e.identity || !candidate || candidate == h.hook) continue;
            if (resolved && resolved != candidate) {
                HLog("ERROR: ambiguous inherited %s hook in slot %d; vtable %p not patched", what, h.slot, vtbl);
                return false;
            }
            resolved = candidate;
        }
        if (!resolved) {
            HLog("ERROR: no original for inherited %s hook in slot %d; vtable %p not patched", what, h.slot, vtbl);
            return false;
        }
        e.orig[h.slot].store(resolved, std::memory_order_relaxed);
        ++inherited;
    }
    e.next = family.head.load(std::memory_order_relaxed);
    family.head.store(record.release(), std::memory_order_release);
    family.count.fetch_add(1, std::memory_order_release);
    int chained = 0;
    for (const SlotHook& h : hooks) {
        DWORD old = 0;
        if (!VirtualProtect(&vtbl[h.slot], sizeof(void*), PAGE_READWRITE, &old)) {
            HLog("ERROR: VirtualProtect of %s vtable entry %d failed (error %lu)", what, h.slot, GetLastError());
            continue;
        }
        if (e.orig[h.slot].load(std::memory_order_acquire) != vtbl[h.slot] && vtbl[h.slot] != h.hook)
            e.orig[h.slot].store(vtbl[h.slot], std::memory_order_release);  // changed meanwhile: chain to the newest external hook
        InterlockedExchangePointer(&vtbl[h.slot], h.hook);
        DWORD unused = 0;
        VirtualProtect(&vtbl[h.slot], sizeof(void*), old, &unused);
        // Entries outside the module that owns most of the vtable were patched by someone else before us.
        HMODULE entryModule = nullptr, firstModule = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(e.orig[h.slot].load(std::memory_order_acquire)), &entryModule);
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(vtbl[0]), &firstModule);
        if (entryModule != firstModule) ++chained;
    }
    HLog("%s vtable %p patched: %zu entries%s", what, static_cast<void*>(vtbl), hooks.size(),
         chained ? (" (" + std::to_string(chained) + " of them already pointed outside the vtable's module: chained to them)").c_str() : "");
    if (inherited) HLog("%s vtable %p: recovered %d inherited hooks without self-chaining", what, vtbl, inherited);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Command-list bindings (everything a Reset clears that the game may expect to survive an upscale dispatch).

struct RootArg {
    uint8_t kind = 0;  // 0 none, 1 descriptor table, 2 CBV, 3 SRV, 4 UAV, 5 32-bit constants
    uint64_t value = 0;
    std::vector<uint32_t> constants;
    std::vector<uint8_t> constantSet;
};

constexpr UINT kRootParams = 64;

struct Pipe {
    bool sigSet = false;
    ID3D12RootSignature* sig = nullptr;
    RootArg args[kRootParams];
};

struct Bindings {
    bool heapsSet = false;
    UINT numHeaps = 0;
    ID3D12DescriptorHeap* heaps[2] = {};
    bool psoSet = false;
    ID3D12PipelineState* pso = nullptr;
    bool stateObjectSet = false;
    void* stateObject = nullptr;
    Pipe gfx, cmp;
    bool topologySet = false;
    D3D12_PRIMITIVE_TOPOLOGY topology{};
    bool ibSet = false, ibNull = false;
    D3D12_INDEX_BUFFER_VIEW ib{};
    uint32_t vbSet = 0, vbNull = 0;
    D3D12_VERTEX_BUFFER_VIEW vb[32] = {};
    uint32_t soSet = 0, soNull = 0;
    D3D12_STREAM_OUTPUT_BUFFER_VIEW so[4] = {};
    bool rtSet = false;
    UINT numRt = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE rt[8] = {};
    BOOL rtSingle = FALSE;
    bool dsvSet = false;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
    bool viewportsSet = false;
    UINT numViewports = 0;
    D3D12_VIEWPORT viewports[16] = {};
    bool scissorsSet = false;
    UINT numScissors = 0;
    D3D12_RECT scissors[16] = {};
    bool blendSet = false, blendNull = false;
    FLOAT blend[4] = {};
    bool stencilSet = false;
    UINT stencil = 0;
    bool boundsSet = false;
    FLOAT boundsMin = 0, boundsMax = 1;
    bool samplesSet = false;
    UINT samplesPerPixel = 0, samplePixels = 0;
    std::vector<D3D12_SAMPLE_POSITION> samples;
    bool viewMaskSet = false;
    UINT viewMask = 0;
    bool shadingRateSet = false, combinersNull = false;
    UINT shadingRate = 0;
    UINT combiners[2] = {};
    bool shadingImageSet = false;
    void* shadingImage = nullptr;
    bool predicationSet = false;
    ID3D12Resource* predicationBuffer = nullptr;
    UINT64 predicationOffset = 0;
    D3D12_PREDICATION_OP predicationOp{};
};

struct Lane;

struct PendingSplit {
    Lane* lane = nullptr;
    uint64_t ticket = 0;
    ID3D12Fence* resume = nullptr;
    uint64_t resumeValue = 0;
    uint64_t id = 0;
    uint64_t timeUs = 0;
    bool released = false;  // order ticket already signalled from the CPU (timeout)
    ID3D12Fence* queryDone = nullptr;
    uint64_t queryDoneValue = 0;
};

struct OwnTable;

struct ListState {
    bool known = false;  // Reset or creation seen since the hooks were installed
    std::shared_ptr<OwnTable> own;      // the list records into its own vtable, which WrongGPU took over (see OwnTable)
    const PatchedVtbl* cls = nullptr;   // the class table the list pointed at before its Reset switched it to its own
    ID3D12CommandAllocator* allocator = nullptr;
    int openQueries = 0;
    std::vector<StatisticsQuery> statistics;
    bool renderPass = false;
    Bindings b;
    std::vector<PendingSplit> pending;  // under g_splitMutex
    // "list facts" diagnostics (one-time log lines about lists whose vtable is not a patched class table).
    uint8_t createdBy = 0;         // 0: creation not seen at this address, 1: CreateCommandList, 2: CreateCommandList1
    void** createdVtbl = nullptr;  // the vtable pointer when the creation call returned
    void** resetVtbl = nullptr;    // the vtable pointer at the last Reset WrongGPU saw
    bool watch = false;            // that vtable was not a patched one: count the hook calls of this recording
    uint32_t watchCalls = 0;
    HMODULE watchCaller[4] = {};   // modules the first 256 of those calls came from
    uint32_t watchCallerCalls[4] = {};
    bool notOurDevice = false;     // a split was refused: the list's device is not a proven context device (BlockedImpl)
};

SRWLOCK g_stateLock = SRWLOCK_INIT;
std::unordered_map<const void*, ListState*> g_states;
thread_local const void* t_lastList = nullptr;
thread_local ListState* t_lastState = nullptr;

ListState& StateOf(const void* list) {
    if (t_lastList == list && t_lastState) return *t_lastState;
    ListState* s = nullptr;
    AcquireSRWLockShared(&g_stateLock);
    auto it = g_states.find(list);
    if (it != g_states.end()) s = it->second;
    ReleaseSRWLockShared(&g_stateLock);
    if (!s) {
        AcquireSRWLockExclusive(&g_stateLock);
        ListState*& slot = g_states[list];
        if (!slot) slot = new ListState();  // kept for the process lifetime; reused when an address is reused
        s = slot;
        ReleaseSRWLockExclusive(&g_stateLock);
    }
    t_lastList = list;
    t_lastState = s;
    return *s;
}

ListState* FindState(const void* list) {
    AcquireSRWLockShared(&g_stateLock);
    auto it = g_states.find(list);
    ListState* s = it != g_states.end() ? it->second : nullptr;
    ReleaseSRWLockShared(&g_stateLock);
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// "list facts": what a list with its own vtable is, from one run's log. No behaviour depends on any of this.

std::string ModuleName(HMODULE m) {
    if (!m) return "no module";
    wchar_t path[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameW(m, path, MAX_PATH);
    const wchar_t* base = path;
    for (DWORD i = 0; i < n; ++i)
        if (path[i] == L'\\' || path[i] == L'/') base = path + i + 1;
    std::string s;
    for (; *base; ++base) s += *base < 128 ? char(*base) : '?';
    return s.empty() ? "?" : s;
}

// "D3D12Core.dll+0x1234", or the kind of memory when the address is in no module.
std::string Where(const void* address) {
    char text[96];
    if (HMODULE m = ModuleOf(address)) {
        std::snprintf(text, sizeof text, "+0x%llx", static_cast<unsigned long long>(static_cast<const char*>(address) - reinterpret_cast<const char*>(m)));
        return ModuleName(m) + text;
    }
    MEMORY_BASIC_INFORMATION mi{};
    if (!VirtualQuery(address, &mi, sizeof mi)) return "unmapped";
    std::snprintf(text, sizeof text, "%s memory (protection 0x%lx, allocation %p)",
                  mi.Type == MEM_PRIVATE ? "private" : mi.Type == MEM_MAPPED ? "mapped" : mi.Type == MEM_IMAGE ? "image" : "free",
                  static_cast<unsigned long>(mi.Protect), mi.AllocationBase);
    return text;
}

// A hook call on a watched list (ListState::watch): how many, and from which modules the first 256 came.
void WatchCall(ListState& s, const void* caller) {
    if (++s.watchCalls > 256) return;
    const HMODULE m = ModuleOf(caller);
    for (int i = 0; i < 4; ++i) {
        if (s.watchCallerCalls[i] && s.watchCaller[i] != m) continue;
        s.watchCaller[i] = m;
        ++s.watchCallerCalls[i];
        return;
    }
}

std::atomic<int> g_factLines{0};  // budget of creation / change lines

// After a creation call returned `object`.
void NoteCreated(int path, D3D12_COMMAND_LIST_TYPE type, const void* allocator, const void* pso, UINT flags, REFIID riid,
                 void* object, const void* caller) {
    ListState& s = StateOf(object);
    void** vt = *static_cast<void***>(object);
    s.createdBy = uint8_t(path);
    s.createdVtbl = vt;
    s.resetVtbl = nullptr;
    s.watch = false;
    s.own.reset();  // a new object at this address: nothing kept from the one before
    s.cls = nullptr;
    s.notOurDevice = false;
    static std::atomic<int> perPath[3], outside{0};
    const bool image = ModuleOf(vt) != nullptr;
    if (image ? perPath[path].fetch_add(1) >= 4 : outside.fetch_add(1) >= 6) return;
    HLog("list facts: %p created by CreateCommandList%s (type %s, allocator %p, pso %p, flags 0x%x, riid {%08lx-%04x-%04x-...}) called from %s: "
         "vtable %p in %s%s",
         object, path == 2 ? "1" : "", ListTypeName(uint32_t(type)), allocator, pso, flags, static_cast<unsigned long>(riid.Data1), riid.Data2,
         riid.Data3, Where(caller).c_str(), static_cast<void*>(vt), Where(vt).c_str(),
         IsPatched(g_listVt, vt) ? " (a table WrongGPU patched)" : "");
}

// At a Reset WrongGPU's hook saw (after the list's own Reset returned).
void NoteReset(const void* self, ListState& s, bool owned) {
    void** vt = *static_cast<void** const*>(self);
    s.watch = !owned && !IsPatched(g_listVt, vt);
    if (s.watch) {
        s.watchCalls = 0;
        for (int i = 0; i < 4; ++i) s.watchCaller[i] = nullptr, s.watchCallerCalls[i] = 0;
    }
    if (vt != s.resetVtbl && s.resetVtbl && g_factLines.fetch_add(1) < 12)
        HLog("list facts: %p: vtable pointer changed between two Resets: %p -> %p in %s", self, static_cast<void*>(s.resetVtbl),
             static_cast<void*>(vt), Where(vt).c_str());
    else if (s.createdBy && !s.resetVtbl && vt != s.createdVtbl && g_factLines.fetch_add(1) < 12)
        HLog("list facts: %p: vtable pointer changed between creation and the first Reset seen: %p -> %p in %s", self,
             static_cast<void*>(s.createdVtbl), static_cast<void*>(vt), Where(vt).c_str());
    s.resetVtbl = vt;
}

// Everything known about the list of a dispatch whose vtable is another layer's own table: the first such list, and
// the first three whose creation or Reset WrongGPU saw (those came after the hooks and say the most). Dispatches that
// AMD's FSR upscales after the fallback are described too.
std::atomic<bool> g_describeOpen{true};
void DescribeForeignList(const void* list, const std::vector<SlotHook>& hooks) {
    static std::atomic<int> plain{0}, informed{0};
    void** vt = *static_cast<void** const*>(list);
    ListState* s = FindState(list);
    const bool known = s && (s->createdBy || s->watch);
    if ((known ? informed : plain).fetch_add(1) >= (known ? 3 : 1)) {
        if (informed.load() >= 3 && plain.load() >= 1) g_describeOpen.store(false);
        return;
    }
    HLog("list facts: dispatch list %p: vtable %p in %s, %lld bytes from the list object (object in %s); creation %s, vtable at creation %p, "
         "at the last Reset seen %p; Reset seen: %s, allocator %p",
         list, static_cast<void*>(vt), Where(vt).c_str(), static_cast<long long>(reinterpret_cast<const char*>(vt) - static_cast<const char*>(list)),
         Where(list).c_str(), !s || !s->createdBy ? "not seen (before the hooks, or not through ID3D12Device::CreateCommandList / CreateCommandList1 of the hooked device class)"
                              : s->createdBy == 2 ? "by CreateCommandList1" : "by CreateCommandList",
         s ? static_cast<void*>(s->createdVtbl) : nullptr, s ? static_cast<void*>(s->resetVtbl) : nullptr, s && s->known ? "yes" : "no",
         s ? static_cast<void*>(s->allocator) : nullptr);
    // The patched class table this one resembles most, and every entry that differs from it.
    const PatchedVtbl* best = nullptr;
    int bestEqual = -1;
    for (auto* entry = g_listVt.head.load(std::memory_order_acquire); entry; entry = entry->next) {
        if (!ModuleOf(entry->vtbl)) continue;
        int equal = 0;
        for (int i = 0; i < kMaxSlots; ++i) equal += entry->vtbl[i] == vt[i];
        if (equal > bestEqual) bestEqual = equal, best = entry;
    }
    if (!best) {
        HLog("list facts: no patched class table to compare with");
        return;
    }
    std::vector<std::pair<HMODULE, std::string>> groups;
    int ours = 0;
    for (int i = 0; i < kMaxSlots; ++i) {
        if (vt[i] == best->vtbl[i]) continue;
        bool hooked = false, inherited = false;
        void* original = nullptr;
        for (const SlotHook& h : hooks)
            if (h.slot == i) hooked = true, inherited = vt[i] == h.hook, original = best->orig[i].load(std::memory_order_acquire);
        if (inherited) {
            ++ours;
            continue;
        }
        const HMODULE m = ModuleOf(vt[i]);
        char number[48];
        if (m) std::snprintf(number, sizeof number, "+0x%llx", static_cast<unsigned long long>(static_cast<const char*>(vt[i]) - reinterpret_cast<const char*>(m)));
        else std::snprintf(number, sizeof number, "@%p", vt[i]);
        const std::string item = " " + std::to_string(i) + (hooked ? "*" : "") + number + (vt[i] == original ? "=class original" : "");
        size_t g = 0;
        while (g < groups.size() && groups[g].first != m) ++g;
        if (g == groups.size()) groups.emplace_back(m, m ? std::string() : " [" + Where(vt[i]) + "]");
        groups[g].second += item;
    }
    HLog("list facts: compared with class table %p (%s): %d of %d entries equal; first entry (QueryInterface) in %s; differing entries below as "
         "slot+offset, * = a slot WrongGPU hooks (in the class table those hold WrongGPU's hooks)",
         static_cast<void*>(best->vtbl), Where(best->vtbl).c_str(), bestEqual, kMaxSlots, Where(vt[0]).c_str());
    for (const auto& g : groups)
        HLog("list facts: entries in %s:%s", g.first ? ModuleName(g.first).c_str() : "no module (slot@address)", g.second.c_str());
    if (s) {
        std::string callers;
        for (int i = 0; i < 4; ++i)
            if (s->watchCallerCalls[i]) callers += " " + ModuleName(s->watchCaller[i]) + " x" + std::to_string(s->watchCallerCalls[i]);
        HLog("list facts: WrongGPU's hooks were called %u times on this list since its last Reset seen (%s); the first 256 came from:%s",
             s->watchCalls, s->watch ? "counted: its vtable was not a patched one at that Reset" : "not counted", callers.empty() ? " -" : callers.c_str());
    }
}

void MergeBundle(Bindings& to, const Bindings& from) {
    if (from.heapsSet) {
        to.heapsSet = true;
        to.numHeaps = from.numHeaps;
        std::memcpy(to.heaps, from.heaps, sizeof to.heaps);
    }
    if (from.psoSet) to.psoSet = true, to.pso = from.pso;
    if (from.stateObjectSet) to.stateObjectSet = true, to.stateObject = from.stateObject;
    for (int p = 0; p < 2; ++p) {
        const Pipe& f = p ? from.cmp : from.gfx;
        Pipe& t = p ? to.cmp : to.gfx;
        if (f.sigSet && f.sig != t.sig) {
            t.sigSet = true;
            t.sig = f.sig;
            for (RootArg& a : t.args) a = RootArg{};
        }
        for (UINT i = 0; i < kRootParams; ++i)
            if (f.args[i].kind) t.args[i] = f.args[i];
    }
    if (from.topologySet) to.topologySet = true, to.topology = from.topology;
    if (from.ibSet) to.ibSet = true, to.ibNull = from.ibNull, to.ib = from.ib;
    for (int i = 0; i < 32; ++i)
        if (from.vbSet & (1u << i)) {
            to.vbSet |= 1u << i;
            to.vbNull = (to.vbNull & ~(1u << i)) | (from.vbNull & (1u << i));
            to.vb[i] = from.vb[i];
        }
    if (from.stencilSet) to.stencilSet = true, to.stencil = from.stencil;
    if (from.blendSet) {
        to.blendSet = true;
        to.blendNull = from.blendNull;
        std::memcpy(to.blend, from.blend, sizeof to.blend);
    }
    if (from.boundsSet) to.boundsSet = true, to.boundsMin = from.boundsMin, to.boundsMax = from.boundsMax;
    if (from.viewMaskSet) to.viewMaskSet = true, to.viewMask = from.viewMask;
}

// ---------------------------------------------------------------------------------------------------------------
// Function types of the hooked methods (x64: one calling convention; `this` first).

using CmdList = ID3D12GraphicsCommandList;
using PfnClose = HRESULT(STDMETHODCALLTYPE*)(CmdList*);
using PfnReset = HRESULT(STDMETHODCALLTYPE*)(CmdList*, ID3D12CommandAllocator*, ID3D12PipelineState*);
using PfnClearState = void(STDMETHODCALLTYPE*)(CmdList*, ID3D12PipelineState*);
using PfnTopology = void(STDMETHODCALLTYPE*)(CmdList*, D3D12_PRIMITIVE_TOPOLOGY);
using PfnViewports = void(STDMETHODCALLTYPE*)(CmdList*, UINT, const D3D12_VIEWPORT*);
using PfnScissors = void(STDMETHODCALLTYPE*)(CmdList*, UINT, const D3D12_RECT*);
using PfnBlend = void(STDMETHODCALLTYPE*)(CmdList*, const FLOAT*);
using PfnStencil = void(STDMETHODCALLTYPE*)(CmdList*, UINT);
using PfnPso = void(STDMETHODCALLTYPE*)(CmdList*, ID3D12PipelineState*);
using PfnBundle = void(STDMETHODCALLTYPE*)(CmdList*, CmdList*);
using PfnHeaps = void(STDMETHODCALLTYPE*)(CmdList*, UINT, ID3D12DescriptorHeap* const*);
using PfnRootSig = void(STDMETHODCALLTYPE*)(CmdList*, ID3D12RootSignature*);
using PfnTable = void(STDMETHODCALLTYPE*)(CmdList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE);
using PfnConstant = void(STDMETHODCALLTYPE*)(CmdList*, UINT, UINT, UINT);
using PfnConstants = void(STDMETHODCALLTYPE*)(CmdList*, UINT, UINT, const void*, UINT);
using PfnRootView = void(STDMETHODCALLTYPE*)(CmdList*, UINT, D3D12_GPU_VIRTUAL_ADDRESS);
using PfnIndexBuffer = void(STDMETHODCALLTYPE*)(CmdList*, const D3D12_INDEX_BUFFER_VIEW*);
using PfnVertexBuffers = void(STDMETHODCALLTYPE*)(CmdList*, UINT, UINT, const D3D12_VERTEX_BUFFER_VIEW*);
using PfnSoTargets = void(STDMETHODCALLTYPE*)(CmdList*, UINT, UINT, const D3D12_STREAM_OUTPUT_BUFFER_VIEW*);
using PfnRenderTargets = void(STDMETHODCALLTYPE*)(CmdList*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL, const D3D12_CPU_DESCRIPTOR_HANDLE*);
using PfnBeginQuery = void(STDMETHODCALLTYPE*)(CmdList*, ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT);
using PfnEndQuery = void(STDMETHODCALLTYPE*)(CmdList*, ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT);
using PfnPredication = void(STDMETHODCALLTYPE*)(CmdList*, ID3D12Resource*, UINT64, D3D12_PREDICATION_OP);
using PfnDepthBounds = void(STDMETHODCALLTYPE*)(CmdList*, FLOAT, FLOAT);
using PfnSamplePositions = void(STDMETHODCALLTYPE*)(CmdList*, UINT, UINT, D3D12_SAMPLE_POSITION*);
using PfnViewMask = void(STDMETHODCALLTYPE*)(CmdList*, UINT);
// ID3D12GraphicsCommandList4/5 (not declared by every SDK the front end builds with): pointers as void*, enums as UINT.
using PfnBeginRenderPass = void(STDMETHODCALLTYPE*)(CmdList*, UINT, const void*, const void*, UINT);
using PfnEndRenderPass = void(STDMETHODCALLTYPE*)(CmdList*);
using PfnStateObject = void(STDMETHODCALLTYPE*)(CmdList*, void*);
using PfnShadingRate = void(STDMETHODCALLTYPE*)(CmdList*, UINT, const UINT*);
using PfnShadingImage = void(STDMETHODCALLTYPE*)(CmdList*, void*);
using PfnExecute = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
using PfnCreateList = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator*,
                                                   ID3D12PipelineState*, REFIID, void**);


// ---------------------------------------------------------------------------------------------------------------
// Lists that record into their own vtable. With some drivers the D3D12 runtime gives every command list a table
// of its own inside the list object: a closed list points at the class table; the list's Reset switches it to its
// own table, in which the runtime has written the driver's entry points for the hot methods (draws, root arguments,
// input assembly, ...) next to the class's methods. Calls made through that table reach neither the class table nor
// WrongGPU's hooks in it (seen in 007 First Light and The Witcher 3 on an AMD driver: 28 of the 41 hooked entries
// in amdxc64.dll, no hook call between Reset and the dispatch).
//
// WrongGPU therefore takes the list's own table over when it appears: after every Reset it sees (its Reset hook is
// in the class table and, from then on, in the list's table) it writes its hooks into the table and keeps, per list,
// what each entry held (the driver's or the class's method) as the original to call. The runtime may write the
// table again at each Reset, so this is repeated then; the entries are checked again before a split. A list whose
// table was taken over only at a dispatch (its Reset came before the hooks) is not split that frame: it is tracked
// from its next Reset. Lists whose vtable is a patched class table never come here.
std::vector<SlotHook> ListHooks(CmdList* list, bool* has1, bool* has4, bool* has5);  // below, with the hooks

struct OwnTable {
    void** vetted = nullptr;    // the table whose memory and identity were checked
    void** rejected = nullptr;  // a table that failed those checks (left untouched)
    void** table = nullptr;     // the table that holds WrongGPU's hooks now
    const PatchedVtbl* cls = nullptr;  // the class table with the same IUnknown entries
    std::vector<SlotHook> hooks;
    void* orig[kMaxSlots] = {};
    bool tracked = false;       // taken over at the Reset that began this recording: every call since was seen
};

std::atomic<bool> g_ownTables{false};   // some list records into a table WrongGPU took over
std::atomic<bool> g_ownLoop{false};     // an entry of such a table called back into WrongGPU's hook without end
std::atomic<int> g_ownStrikes{0};       // splits refused in a row because a table was rewritten while its list recorded
std::atomic<int> g_ownNotReady{0};      // splits refused in a row because a table was not tracked yet
std::atomic<int> g_ownLines{0};

constexpr char kOwnLoopReason[] =
    "an entry of the command list's own vtable calls back into WrongGPU's hook: such lists are left alone and never split";
constexpr char kOwnRewrittenReason[] =
    "the command list's own vtable is rewritten while the list records: its calls cannot be tracked";
constexpr char kOwnNeverReason[] =
    "the command lists record into their own vtables and none could be tracked from its Reset in 300 dispatches";

bool OwnIntact(const OwnTable& o) {
    for (const SlotHook& h : o.hooks)
        if (o.table[h.slot] != h.hook) return false;
    return true;
}

// The patched class table whose IUnknown entries `vt` shares (the one that equals it in most entries), or null.
const PatchedVtbl* ClassOf(void** vt, const PatchedVtbl* preferred) {
    if (preferred && preferred->vtbl[0] == vt[0] && preferred->vtbl[1] == vt[1] && preferred->vtbl[2] == vt[2]) return preferred;
    const PatchedVtbl* best = nullptr;
    int bestEqual = -1;
    for (auto* entry = g_listVt.head.load(std::memory_order_acquire); entry; entry = entry->next) {
        if (entry->vtbl[0] != vt[0] || entry->vtbl[1] != vt[1] || entry->vtbl[2] != vt[2] || !ModuleOf(entry->vtbl)) continue;
        int equal = 0;
        for (int i = 0; i < kMaxSlots; ++i) equal += entry->vtbl[i] == vt[i];
        if (equal > bestEqual) bestEqual = equal, best = entry;
    }
    return best;
}

// Writes WrongGPU's hooks into the list's own table `vt` (again). False: the table is not one WrongGPU can take
// over (nothing was written), or a loop was seen before.
bool Adopt(CmdList* self, ListState& s, void** vt, bool atReset) {
    if (g_ownLoop.load(std::memory_order_relaxed)) return false;
    if (!s.own) s.own.reset(new (std::nothrow) OwnTable);
    if (!s.own) return false;
    OwnTable& o = *s.own;
    if (o.rejected == vt) return false;
    const char* refused = nullptr;
    if (o.vetted != vt) {  // the first time for this table: the checks that cost
        o.table = nullptr;
        o.tracked = false;
        std::memset(o.orig, 0, sizeof o.orig);
        MEMORY_BASIC_INFORMATION mi{};
        const char* begin = reinterpret_cast<const char*>(vt);
        if (ModuleOf(vt)) refused = "it is part of a module";
        else if (!VirtualQuery(vt, &mi, sizeof mi) || mi.State != MEM_COMMIT || (mi.Protect != PAGE_READWRITE && mi.Protect != PAGE_EXECUTE_READWRITE) ||
                 begin + kMaxSlots * sizeof(void*) > static_cast<const char*>(mi.BaseAddress) + mi.RegionSize)
            refused = "its memory is not plain writable memory";
        else if (!(o.cls = ClassOf(vt, s.cls))) refused = "no patched class table has its IUnknown entries";
        else o.hooks = ListHooks(self, nullptr, nullptr, nullptr);
    }
    // Every entry must be WrongGPU's hook (then the original is the class's or the one kept before) or code of a module.
    void* fresh[kMaxSlots];
    int rewritten = 0, inherited = 0;
    for (size_t i = 0; !refused && i < o.hooks.size(); ++i) {
        const SlotHook& h = o.hooks[i];
        void* cur = vt[h.slot];
        fresh[h.slot] = o.orig[h.slot];
        if (cur == h.hook) {
            if (!fresh[h.slot]) {
                fresh[h.slot] = o.cls->orig[h.slot].load(std::memory_order_acquire);
                ++inherited;
                if (!fresh[h.slot] || fresh[h.slot] == h.hook) refused = "a copied hook has no class original";
            }
        } else {
            ++rewritten;
            if (cur != fresh[h.slot] && (!cur || !ModuleOf(cur))) refused = "an entry does not point into a module";
            fresh[h.slot] = cur;
        }
    }
    if (refused) {
        o.rejected = vt;
        o.vetted = o.table = nullptr;
        if (g_ownLines.fetch_add(1) < 6)
            HLog("own table: list %p records into its own vtable %p (%s), which WrongGPU does not take over: %s", static_cast<void*>(self),
                 static_cast<void*>(vt), Where(vt).c_str(), refused);
        return false;
    }
    const bool first = o.table != vt;
    for (const SlotHook& h : o.hooks) {
        o.orig[h.slot] = fresh[h.slot];
        if (vt[h.slot] != h.hook) InterlockedExchangePointer(&vt[h.slot], h.hook);
    }
    o.vetted = o.table = vt;
    o.tracked = atReset;
    g_ownTables.store(true, std::memory_order_release);
    static std::atomic<int> said{0};
    if (first && said.fetch_add(1) < 4) {
        std::string modules;
        std::vector<HMODULE> seen;
        for (const SlotHook& h : o.hooks) {
            const HMODULE m = ModuleOf(o.orig[h.slot]);
            if (std::find(seen.begin(), seen.end(), m) != seen.end()) continue;
            seen.push_back(m);
            int n = 0;
            for (const SlotHook& k : o.hooks) n += ModuleOf(o.orig[k.slot]) == m;
            modules += (modules.empty() ? "" : ", ") + std::to_string(n) + " in " + ModuleName(m);
        }
        HLog("own table: list %p records into its own vtable %p (%lld bytes from the list object, class table %p): taken over %s; "
             "%zu hooks written, originals kept per list: %s (%d entries were copies of WrongGPU's class-table hooks)%s",
             static_cast<void*>(self), static_cast<void*>(vt), static_cast<long long>(reinterpret_cast<const char*>(vt) - reinterpret_cast<const char*>(self)),
             static_cast<void*>(o.cls->vtbl), atReset ? "at the list's Reset: tracked from here" : "at a dispatch: tracked from the list's next Reset",
             o.hooks.size(), modules.c_str(), inherited, said.load() == 4 ? " (further lists: not logged)" : "");
    }
    (void)rewritten;
    return true;
}

// Are less than 48 KiB of this thread's stack left? (The only sign of an entry that calls back into the hook; threads
// that record command lists may have small stacks, so the margin is only what the handling below needs.)
bool StackNearlyExhausted() {
    thread_local uintptr_t low = 0;
    char here;
    if (!low) {
        MEMORY_BASIC_INFORMATION mi{};
        low = VirtualQuery(&here, &mi, sizeof mi) && mi.AllocationBase ? reinterpret_cast<uintptr_t>(mi.AllocationBase) : 1;
    }
    return reinterpret_cast<uintptr_t>(&here) - low < (48u << 10);
}

// The original of `slot` for the list `self`: by its vtable (a patched class table), else from the list's own table.
void* ListOriginal(const void* self, int slot) {
    void** vt = *static_cast<void** const*>(self);
    PatchedVtbl* oldest = nullptr;
    for (auto* entry = g_listVt.head.load(std::memory_order_acquire); entry; entry = entry->next) {
        if (entry->vtbl == vt && entry->identity == vt[0]) return entry->orig[slot].load(std::memory_order_acquire);
        oldest = entry;
    }
    if (g_ownTables.load(std::memory_order_relaxed)) {
        ListState& s = StateOf(self);
        if (s.own && s.own->table == vt && s.own->orig[slot]) {
            OwnTable& o = *s.own;
            if (!StackNearlyExhausted()) return o.orig[slot];
            // The entry WrongGPU chains to came back to this hook again and again: give every entry back, never take
            // such a table over again, and end the chain in the class's method.
            for (const SlotHook& h : o.hooks)
                if (vt[h.slot] == h.hook && o.orig[h.slot]) InterlockedExchangePointer(&vt[h.slot], o.orig[h.slot]);
            o.table = o.vetted = nullptr;
            o.rejected = vt;
            o.tracked = false;
            s.known = false;
            if (!g_ownLoop.exchange(true)) HLog("ERROR: own table: %s (list %p, vtable %p, slot %d)", kOwnLoopReason, self, static_cast<void*>(vt), slot);
            if (o.cls) return o.cls->orig[slot].load(std::memory_order_acquire);
        }
    }
    return oldest ? oldest->orig[slot].load(std::memory_order_acquire) : nullptr;
}

template <class F>
F ListOrig(const void* self, int slot) {
    return reinterpret_cast<F>(ListOriginal(self, slot));
}

// After a Reset of `self` returned (the game's, through WrongGPU's hook, or the split's): `before` is the vtable
// pointer the list had when the Reset was called.
void AfterReset(CmdList* self, ListState& s, void** before) {
    void** vt = *reinterpret_cast<void***>(self);
    bool owned = false;
    if (IsPatched(g_listVt, vt)) {
        if (s.own) s.own->tracked = false;
    } else {
        if (before != vt)
            for (auto* entry = g_listVt.head.load(std::memory_order_acquire); entry; entry = entry->next)
                if (entry->vtbl == before) s.cls = entry;
        owned = Adopt(self, s, vt, true);
    }
    NoteReset(self, s, owned);
}

#define ORIG(Pfn, slot) ListOrig<Pfn>(self, slot)

// Re-applies `b` to a freshly reset list, through the original entries (the tracked state stays as it is).
int Restore(CmdList* self, const Bindings& b) {
    int n = 0;
    if (b.heapsSet && b.numHeaps) ORIG(PfnHeaps, WG_VT_LIST_SET_DESCRIPTOR_HEAPS)(self, b.numHeaps, b.heaps), ++n;
    if (b.gfx.sigSet) ORIG(PfnRootSig, WG_VT_LIST_SET_GRAPHICS_ROOT_SIGNATURE)(self, b.gfx.sig), ++n;
    if (b.cmp.sigSet) ORIG(PfnRootSig, WG_VT_LIST_SET_COMPUTE_ROOT_SIGNATURE)(self, b.cmp.sig), ++n;
    if (b.psoSet) ORIG(PfnPso, WG_VT_LIST_SET_PIPELINE_STATE)(self, b.pso), ++n;
    if (b.stateObjectSet) {
        if (auto f = ORIG(PfnStateObject, WG_VT_LIST_SET_PIPELINE_STATE1)) f(self, b.stateObject), ++n;
    }
    for (int p = 0; p < 2; ++p) {
        const Pipe& pipe = p ? b.cmp : b.gfx;
        const bool g = p == 0;
        for (UINT i = 0; i < kRootParams; ++i) {
            const RootArg& a = pipe.args[i];
            switch (a.kind) {
            case 1: {
                D3D12_GPU_DESCRIPTOR_HANDLE h{a.value};
                ORIG(PfnTable, g ? WG_VT_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE : WG_VT_LIST_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE)(self, i, h);
                ++n;
                break;
            }
            case 2: ORIG(PfnRootView, g ? WG_VT_LIST_SET_GRAPHICS_ROOT_CBV : WG_VT_LIST_SET_COMPUTE_ROOT_CBV)(self, i, a.value), ++n; break;
            case 3: ORIG(PfnRootView, g ? WG_VT_LIST_SET_GRAPHICS_ROOT_SRV : WG_VT_LIST_SET_COMPUTE_ROOT_SRV)(self, i, a.value), ++n; break;
            case 4: ORIG(PfnRootView, g ? WG_VT_LIST_SET_GRAPHICS_ROOT_UAV : WG_VT_LIST_SET_COMPUTE_ROOT_UAV)(self, i, a.value), ++n; break;
            case 5: {
                const auto set = ORIG(PfnConstants, g ? WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS : WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANTS);
                size_t k = 0;
                while (k < a.constants.size()) {
                    if (!a.constantSet[k]) {
                        ++k;
                        continue;
                    }
                    size_t e = k;
                    while (e < a.constants.size() && a.constantSet[e]) ++e;
                    set(self, i, UINT(e - k), a.constants.data() + k, UINT(k));
                    ++n;
                    k = e;
                }
                break;
            }
            default: break;
            }
        }
    }
    if (b.topologySet) ORIG(PfnTopology, WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY)(self, b.topology), ++n;
    if (b.ibSet) ORIG(PfnIndexBuffer, WG_VT_LIST_IA_SET_INDEX_BUFFER)(self, b.ibNull ? nullptr : &b.ib), ++n;
    for (UINT i = 0; i < 32; ++i)
        if (b.vbSet & (1u << i)) ORIG(PfnVertexBuffers, WG_VT_LIST_IA_SET_VERTEX_BUFFERS)(self, i, 1, (b.vbNull & (1u << i)) ? nullptr : &b.vb[i]), ++n;
    for (UINT i = 0; i < 4; ++i)
        if (b.soSet & (1u << i)) ORIG(PfnSoTargets, WG_VT_LIST_SO_SET_TARGETS)(self, i, 1, (b.soNull & (1u << i)) ? nullptr : &b.so[i]), ++n;
    if (b.rtSet)
        ORIG(PfnRenderTargets, WG_VT_LIST_OM_SET_RENDER_TARGETS)(self, b.numRt, b.numRt ? b.rt : nullptr, b.rtSingle, b.dsvSet ? &b.dsv : nullptr), ++n;
    if (b.viewportsSet) ORIG(PfnViewports, WG_VT_LIST_RS_SET_VIEWPORTS)(self, b.numViewports, b.viewports), ++n;
    if (b.scissorsSet) ORIG(PfnScissors, WG_VT_LIST_RS_SET_SCISSOR_RECTS)(self, b.numScissors, b.scissors), ++n;
    if (b.blendSet) ORIG(PfnBlend, WG_VT_LIST_OM_SET_BLEND_FACTOR)(self, b.blendNull ? nullptr : b.blend), ++n;
    if (b.stencilSet) ORIG(PfnStencil, WG_VT_LIST_OM_SET_STENCIL_REF)(self, b.stencil), ++n;
    if (b.boundsSet) {
        if (auto f = ORIG(PfnDepthBounds, WG_VT_LIST_OM_SET_DEPTH_BOUNDS)) f(self, b.boundsMin, b.boundsMax), ++n;
    }
    if (b.samplesSet) {
        if (auto f = ORIG(PfnSamplePositions, WG_VT_LIST_SET_SAMPLE_POSITIONS))
            f(self, b.samplesPerPixel, b.samplePixels, b.samples.empty() ? nullptr : const_cast<D3D12_SAMPLE_POSITION*>(b.samples.data())), ++n;
    }
    if (b.viewMaskSet) {
        if (auto f = ORIG(PfnViewMask, WG_VT_LIST_SET_VIEW_INSTANCE_MASK)) f(self, b.viewMask), ++n;
    }
    if (b.shadingRateSet) {
        if (auto f = ORIG(PfnShadingRate, WG_VT_LIST_RS_SET_SHADING_RATE)) f(self, b.shadingRate, b.combinersNull ? nullptr : b.combiners), ++n;
    }
    if (b.shadingImageSet) {
        if (auto f = ORIG(PfnShadingImage, WG_VT_LIST_RS_SET_SHADING_RATE_IMAGE)) f(self, b.shadingImage), ++n;
    }
    if (b.predicationSet) ORIG(PfnPredication, WG_VT_LIST_SET_PREDICATION)(self, b.predicationBuffer, b.predicationOffset, b.predicationOp), ++n;
    return n;
}

// ---------------------------------------------------------------------------------------------------------------
// Split lanes: WrongGPU's own queues, one order fence each. A lane carries the pending splits of one list at a time,
// so a lane never waits for a list the game submits after another lane's list (no cross-list deadlock).

struct Lane {
    uint32_t id = 0;
    ID3D12Device* device = nullptr;
    D3D12_COMMAND_LIST_TYPE type{};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* order = nullptr;
    uint64_t ticket = 0;
    const void* list = nullptr;  // list whose splits are pending on this lane
    int pending = 0;
};

std::mutex g_splitMutex;            // lanes, pending splits, queues, trace
std::vector<Lane*> g_lanes;
std::atomic<int> g_pendingTotal{0};
std::atomic<bool> g_installed{false};
bool g_installFailed = false;
uint64_t g_splitIds = 0;
uint64_t g_refused = 0;
uint32_t g_laneIds = 0;
constexpr size_t kLanesPerType = 4;

// Bug hunt F3: WrongGPU's queues and fences of a device the game lost (device removed: after a TDR it makes a new one)
// kept that device alive, and its entry filled the known-device list (16), after which every split was refused. When a
// device gets its first lane of a type, the idle lanes of other devices that report removal are released, and those
// devices leave the list. Under g_splitMutex. Lanes with splits pending stay (their splits point at them).
void DropRemovedLanes(ID3D12Device* current) {
    for (size_t i = 0; i < g_lanes.size();) {
        Lane* l = g_lanes[i];
        if (l->device == current || l->pending || l->device->GetDeviceRemovedReason() == S_OK) {
            ++i;
            continue;
        }
        HLog("WrongGPU split queue W%u released: its device %p was removed", l->id, static_cast<void*>(l->device));
        Forget(g_contextDevices, g_contextDeviceCount, static_cast<void*>(l->device));
        if (l->order) l->order->Release();
        if (l->queue) l->queue->Release();
        delete l;
        g_lanes.erase(g_lanes.begin() + std::ptrdiff_t(i));
    }
}

// Releases the pending splits of `s` from the CPU (the game reset the list without submitting it, or the timeout).
// Under g_splitMutex. keepEntries: leave them (marked released) so a later submission still waits for resume.
void ReleaseFromCpu(const void* list, ListState& s, const char* why, bool keepEntries) {
    for (PendingSplit& p : s.pending) {
        if (!p.released) {
            p.lane->order->Signal(p.ticket);
#if defined(WRONGGPU_SPLIT_PROGRESS_QA)
            if(g_progressReady.load()) ProcessSplitProgress().CpuReleased(p.id);
#endif
            p.released = true;
            HLog("split S%llu of list %p released from the CPU: %s (its first half runs now; WrongGPU's order ticket %llu on W%u)",
                 static_cast<unsigned long long>(p.id), list, why, static_cast<unsigned long long>(p.ticket), p.lane->id);
        }
    }
    if (!keepEntries) {
        for (PendingSplit& p : s.pending) {
            if (--p.lane->pending == 0) p.lane->list = nullptr;
            g_pendingTotal.fetch_sub(1);
        }
        s.pending.clear();
    }
}

void CheckTimeouts(uint64_t now) {
    if (!g_pendingTotal.load()) return;
    for (Lane* lane : g_lanes) {
        if (!lane->pending || !lane->list) continue;
        ListState* s = FindState(lane->list);
        if (!s) continue;
        for (PendingSplit& p : s->pending)
            if (!p.released && now - p.timeUs > uint64_t(g_settings.timeoutMs) * 1000u) {
                ReleaseFromCpu(lane->list, *s, "the game did not submit the list within split_timeout_ms", true);
                break;
            }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Submission trace (20-T4): queues seen, and the FSR dispatch lists until they are submitted.

struct QueueInfo {
    const void* queue = nullptr;
    uint32_t id = 0;
    uint32_t type = 0;
    int priority = 0;
    uint64_t calls = 0;
    uint64_t lists = 0;
};

struct Counts {
    uint64_t calls = 0, lists = 0;
};

struct DispatchNote {
    const void* list = nullptr;
    uint64_t seq = 0;
    uint32_t notes = 0;  // times noted (ffx_fsr3 and ffx_fsr3upscaler both see the same dispatch)
    DWORD thread = 0;
    uint64_t timeUs = 0;
    std::vector<Counts> at;  // per queue (index = id - 1) when the dispatch was recorded
};

std::vector<QueueInfo> g_queues;
std::vector<DispatchNote> g_notes;  // not yet submitted, oldest first (at most 64)
std::atomic<int> g_noteCount{0};    // g_notes.size(), readable without the lock
std::unordered_map<const void*, const void*> g_lastQueue;  // FSR list -> queue it was last executed on
std::vector<Counts> g_atPreviousSubmit;
uint64_t g_dispatchSeq = 0;
uint64_t g_tracedSubmits = 0;

QueueInfo& QueueOf(ID3D12CommandQueue* q) {
    for (QueueInfo& qi : g_queues)
        if (qi.queue == q) return qi;
    QueueInfo qi;
    qi.queue = q;
    qi.id = uint32_t(g_queues.size() + 1);
    const D3D12_COMMAND_QUEUE_DESC d = q->GetDesc();
    qi.type = uint32_t(d.Type);
    qi.priority = d.Priority;
    g_queues.push_back(qi);
    HLog("queue Q%u seen: %p %s, priority %d, flags 0x%x", qi.id, static_cast<void*>(q), ListTypeName(qi.type), d.Priority, unsigned(d.Flags));
    return g_queues.back();
}

std::vector<Counts> Snapshot() {
    std::vector<Counts> s(g_queues.size());
    for (size_t i = 0; i < g_queues.size(); ++i) s[i] = {g_queues[i].calls, g_queues[i].lists};
    return s;
}

std::string Since(const std::vector<Counts>& then) {
    uint64_t calls = 0, lists = 0;
    std::string per;
    for (size_t i = 0; i < g_queues.size(); ++i) {
        const Counts t = i < then.size() ? then[i] : Counts{};
        const uint64_t c = g_queues[i].calls - t.calls, l = g_queues[i].lists - t.lists;
        if (!c) continue;
        calls += c;
        lists += l;
        char part[96];
        std::snprintf(part, sizeof part, "%sQ%u %s: %llu/%llu", per.empty() ? "" : ", ", g_queues[i].id, ListTypeName(g_queues[i].type),
                      static_cast<unsigned long long>(c), static_cast<unsigned long long>(l));
        per += part;
    }
    char text[128];
    std::snprintf(text, sizeof text, "%llu lists in %llu calls", static_cast<unsigned long long>(lists), static_cast<unsigned long long>(calls));
    return std::string(text) + (per.empty() ? "" : " (calls/lists: " + per + ")");
}

// ---------------------------------------------------------------------------------------------------------------
// Hooks: command list

#define STATE                      \
    ListState& s = StateOf(self);  \
    if (s.watch) WatchCall(s, WG_RETURN_ADDRESS())

HRESULT STDMETHODCALLTYPE HClose(CmdList* self) { return ORIG(PfnClose, WG_VT_LIST_CLOSE)(self); }

HRESULT STDMETHODCALLTYPE HReset(CmdList* self, ID3D12CommandAllocator* allocator, ID3D12PipelineState* pso) {
    void** const before = *reinterpret_cast<void***>(self);
    const HRESULT hr = ORIG(PfnReset, WG_VT_LIST_RESET)(self, allocator, pso);
    if (FAILED(hr)) return hr;
    STATE;
    if (g_pendingTotal.load() || g_noteCount.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(g_splitMutex);
        if (!s.pending.empty()) ReleaseFromCpu(self, s, "the game reset the list without submitting it", false);
        for (size_t k = 0; k < g_notes.size(); ++k)
            if (g_notes[k].list == self) {
                HLog("trace: FSR list %p (dispatch D%llu) was reset without being submitted", static_cast<void*>(self),
                     static_cast<unsigned long long>(g_notes[k].seq));
                g_notes.erase(g_notes.begin() + std::ptrdiff_t(k));
                g_noteCount.store(int(g_notes.size()));
                break;
            }
    }
    s.known = true;
    s.allocator = allocator;
    if (!InternalQueryCommands()) { s.openQueries = 0; s.statistics.clear(); }
    s.renderPass = false;
    s.b = Bindings{};
    if (pso) s.b.psoSet = true, s.b.pso = pso;
    AfterReset(self, s, before);
    return hr;
}

void STDMETHODCALLTYPE HClearState(CmdList* self, ID3D12PipelineState* pso) {
    ORIG(PfnClearState, WG_VT_LIST_CLEAR_STATE)(self, pso);
    STATE;
    s.b = Bindings{};
    if (pso) s.b.psoSet = true, s.b.pso = pso;
}

void STDMETHODCALLTYPE HTopology(CmdList* self, D3D12_PRIMITIVE_TOPOLOGY t) {
    ORIG(PfnTopology, WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY)(self, t);
    STATE;
    s.b.topologySet = true;
    s.b.topology = t;
}

void STDMETHODCALLTYPE HViewports(CmdList* self, UINT n, const D3D12_VIEWPORT* v) {
    ORIG(PfnViewports, WG_VT_LIST_RS_SET_VIEWPORTS)(self, n, v);
    STATE;
    s.b.viewportsSet = true;
    s.b.numViewports = v ? (n < 16 ? n : 16) : 0;
    if (s.b.numViewports) std::memcpy(s.b.viewports, v, s.b.numViewports * sizeof *v);
}

void STDMETHODCALLTYPE HScissors(CmdList* self, UINT n, const D3D12_RECT* r) {
    ORIG(PfnScissors, WG_VT_LIST_RS_SET_SCISSOR_RECTS)(self, n, r);
    STATE;
    s.b.scissorsSet = true;
    s.b.numScissors = r ? (n < 16 ? n : 16) : 0;
    if (s.b.numScissors) std::memcpy(s.b.scissors, r, s.b.numScissors * sizeof *r);
}

void STDMETHODCALLTYPE HBlend(CmdList* self, const FLOAT* f) {
    ORIG(PfnBlend, WG_VT_LIST_OM_SET_BLEND_FACTOR)(self, f);
    STATE;
    s.b.blendSet = true;
    s.b.blendNull = !f;
    if (f) std::memcpy(s.b.blend, f, sizeof s.b.blend);
}

void STDMETHODCALLTYPE HStencil(CmdList* self, UINT v) {
    ORIG(PfnStencil, WG_VT_LIST_OM_SET_STENCIL_REF)(self, v);
    STATE;
    s.b.stencilSet = true;
    s.b.stencil = v;
}

void STDMETHODCALLTYPE HPso(CmdList* self, ID3D12PipelineState* p) {
    ORIG(PfnPso, WG_VT_LIST_SET_PIPELINE_STATE)(self, p);
    STATE;
    s.b.psoSet = true;
    s.b.pso = p;
}

void STDMETHODCALLTYPE HBundle(CmdList* self, CmdList* bundle) {
    ORIG(PfnBundle, WG_VT_LIST_EXECUTE_BUNDLE)(self, bundle);
    STATE;
    if (bundle) MergeBundle(s.b, StateOf(bundle).b);
    t_lastList = nullptr;  // StateOf(bundle) replaced the per-thread cache entry; keep it simple
}

void STDMETHODCALLTYPE HHeaps(CmdList* self, UINT n, ID3D12DescriptorHeap* const* h) {
    ORIG(PfnHeaps, WG_VT_LIST_SET_DESCRIPTOR_HEAPS)(self, n, h);
    STATE;
    s.b.heapsSet = true;
    s.b.numHeaps = h ? (n < 2 ? n : 2) : 0;
    for (UINT i = 0; i < s.b.numHeaps; ++i) s.b.heaps[i] = h[i];
}

void SetSig(Pipe& p, ID3D12RootSignature* sig) {
    if (!p.sigSet || p.sig != sig)
        for (RootArg& a : p.args) a = RootArg{};  // a different root signature invalidates the arguments
    p.sigSet = true;
    p.sig = sig;
}

void STDMETHODCALLTYPE HComputeSig(CmdList* self, ID3D12RootSignature* sig) {
    ORIG(PfnRootSig, WG_VT_LIST_SET_COMPUTE_ROOT_SIGNATURE)(self, sig);
    STATE;
    SetSig(s.b.cmp, sig);
}

void STDMETHODCALLTYPE HGraphicsSig(CmdList* self, ID3D12RootSignature* sig) {
    ORIG(PfnRootSig, WG_VT_LIST_SET_GRAPHICS_ROOT_SIGNATURE)(self, sig);
    STATE;
    SetSig(s.b.gfx, sig);
}

void SetView(Pipe& p, UINT i, uint8_t kind, uint64_t value) {
    if (i >= kRootParams) return;
    RootArg& a = p.args[i];
    a.kind = kind;
    a.value = value;
    a.constants.clear();
    a.constantSet.clear();
}

void SetConstants(Pipe& p, UINT i, UINT count, const void* data, UINT offset) {
    if (i >= kRootParams || !data || offset + count > 64 * 4) return;
    RootArg& a = p.args[i];
    if (a.kind != 5) {
        a = RootArg{};
        a.kind = 5;
    }
    if (a.constants.size() < offset + count) {
        a.constants.resize(offset + count);
        a.constantSet.resize(offset + count);
    }
    std::memcpy(a.constants.data() + offset, data, count * sizeof(uint32_t));
    std::memset(a.constantSet.data() + offset, 1, count);
}

void STDMETHODCALLTYPE HComputeTable(CmdList* self, UINT i, D3D12_GPU_DESCRIPTOR_HANDLE h) {
    ORIG(PfnTable, WG_VT_LIST_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE)(self, i, h);
    STATE;
    SetView(s.b.cmp, i, 1, h.ptr);
}
void STDMETHODCALLTYPE HGraphicsTable(CmdList* self, UINT i, D3D12_GPU_DESCRIPTOR_HANDLE h) {
    ORIG(PfnTable, WG_VT_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE)(self, i, h);
    STATE;
    SetView(s.b.gfx, i, 1, h.ptr);
}
void STDMETHODCALLTYPE HComputeConstant(CmdList* self, UINT i, UINT v, UINT offset) {
    ORIG(PfnConstant, WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANT)(self, i, v, offset);
    STATE;
    SetConstants(s.b.cmp, i, 1, &v, offset);
}
void STDMETHODCALLTYPE HGraphicsConstant(CmdList* self, UINT i, UINT v, UINT offset) {
    ORIG(PfnConstant, WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANT)(self, i, v, offset);
    STATE;
    SetConstants(s.b.gfx, i, 1, &v, offset);
}
void STDMETHODCALLTYPE HComputeConstants(CmdList* self, UINT i, UINT n, const void* d, UINT offset) {
    ORIG(PfnConstants, WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANTS)(self, i, n, d, offset);
    STATE;
    SetConstants(s.b.cmp, i, n, d, offset);
}
void STDMETHODCALLTYPE HGraphicsConstants(CmdList* self, UINT i, UINT n, const void* d, UINT offset) {
    ORIG(PfnConstants, WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS)(self, i, n, d, offset);
    STATE;
    SetConstants(s.b.gfx, i, n, d, offset);
}
#define ROOT_VIEW_HOOK(Name, slot, pipe, kind)                                    \
    void STDMETHODCALLTYPE Name(CmdList* self, UINT i, D3D12_GPU_VIRTUAL_ADDRESS va) {  \
        ORIG(PfnRootView, slot)(self, i, va);                                     \
        STATE;                                                                    \
        SetView(s.b.pipe, i, kind, va);                                           \
    }
ROOT_VIEW_HOOK(HComputeCbv, WG_VT_LIST_SET_COMPUTE_ROOT_CBV, cmp, 2)
ROOT_VIEW_HOOK(HGraphicsCbv, WG_VT_LIST_SET_GRAPHICS_ROOT_CBV, gfx, 2)
ROOT_VIEW_HOOK(HComputeSrv, WG_VT_LIST_SET_COMPUTE_ROOT_SRV, cmp, 3)
ROOT_VIEW_HOOK(HGraphicsSrv, WG_VT_LIST_SET_GRAPHICS_ROOT_SRV, gfx, 3)
ROOT_VIEW_HOOK(HComputeUav, WG_VT_LIST_SET_COMPUTE_ROOT_UAV, cmp, 4)
ROOT_VIEW_HOOK(HGraphicsUav, WG_VT_LIST_SET_GRAPHICS_ROOT_UAV, gfx, 4)

void STDMETHODCALLTYPE HIndexBuffer(CmdList* self, const D3D12_INDEX_BUFFER_VIEW* v) {
    ORIG(PfnIndexBuffer, WG_VT_LIST_IA_SET_INDEX_BUFFER)(self, v);
    STATE;
    s.b.ibSet = true;
    s.b.ibNull = !v;
    if (v) s.b.ib = *v;
}

void STDMETHODCALLTYPE HVertexBuffers(CmdList* self, UINT start, UINT n, const D3D12_VERTEX_BUFFER_VIEW* v) {
    ORIG(PfnVertexBuffers, WG_VT_LIST_IA_SET_VERTEX_BUFFERS)(self, start, n, v);
    STATE;
    for (UINT k = 0; k < n && start + k < 32; ++k) {
        const uint32_t bit = 1u << (start + k);
        s.b.vbSet |= bit;
        if (v) {
            s.b.vbNull &= ~bit;
            s.b.vb[start + k] = v[k];
        } else {
            s.b.vbNull |= bit;
        }
    }
}

void STDMETHODCALLTYPE HSoTargets(CmdList* self, UINT start, UINT n, const D3D12_STREAM_OUTPUT_BUFFER_VIEW* v) {
    ORIG(PfnSoTargets, WG_VT_LIST_SO_SET_TARGETS)(self, start, n, v);
    STATE;
    for (UINT k = 0; k < n && start + k < 4; ++k) {
        const uint32_t bit = 1u << (start + k);
        s.b.soSet |= bit;
        if (v) {
            s.b.soNull &= ~bit;
            s.b.so[start + k] = v[k];
        } else {
            s.b.soNull |= bit;
        }
    }
}

void STDMETHODCALLTYPE HRenderTargets(CmdList* self, UINT n, const D3D12_CPU_DESCRIPTOR_HANDLE* rt, BOOL single, const D3D12_CPU_DESCRIPTOR_HANDLE* dsv) {
    ORIG(PfnRenderTargets, WG_VT_LIST_OM_SET_RENDER_TARGETS)(self, n, rt, single, dsv);
    STATE;
    s.b.rtSet = true;
    s.b.numRt = rt ? (n < 8 ? n : 8) : 0;
    s.b.rtSingle = single;
    if (s.b.numRt) std::memcpy(s.b.rt, rt, (single ? 1 : s.b.numRt) * sizeof *rt);
    s.b.dsvSet = dsv != nullptr;
    if (dsv) s.b.dsv = *dsv;
}

void STDMETHODCALLTYPE HBeginQuery(CmdList* self, ID3D12QueryHeap* h, D3D12_QUERY_TYPE t, UINT i) {
    ORIG(PfnBeginQuery, WG_VT_LIST_BEGIN_QUERY)(self, h, t, i);
    if (InternalQueryCommands()) return;
    STATE;
    ++s.openQueries;
    if (IsStatisticsQuery(t)) {
        StatisticsQuery q{h,t,i};
        s.statistics.push_back(q);
        StatisticsQueryBegin(q);
    }
}

void STDMETHODCALLTYPE HEndQuery(CmdList* self, ID3D12QueryHeap* h, D3D12_QUERY_TYPE t, UINT i) {
    ORIG(PfnEndQuery, WG_VT_LIST_END_QUERY)(self, h, t, i);
    if (InternalQueryCommands()) return;
    STATE;
    if (t != D3D12_QUERY_TYPE_TIMESTAMP && s.openQueries > 0) --s.openQueries;
    for (auto it=s.statistics.begin();it!=s.statistics.end();++it)
        if(it->heap==h && it->type==t && it->index==i) {s.statistics.erase(it);break;}
}

QueryMethods Methods(CmdList* self) {
    return {ORIG(QueryBeginFn,WG_VT_LIST_BEGIN_QUERY),ORIG(QueryEndFn,WG_VT_LIST_END_QUERY),
            ORIG(QueryResolveFn,WG_VT_LIST_RESOLVE_QUERY_DATA)};
}

void STDMETHODCALLTYPE HResolveQuery(CmdList* self,ID3D12QueryHeap* heap,D3D12_QUERY_TYPE type,UINT first,UINT count,ID3D12Resource* target,UINT64 offset) {
    auto m=Methods(self);
    if(InternalQueryCommands() || !IsStatisticsQuery(type) || !HasStatisticsPrefix(heap,type,first,count)) {
        m.resolve(self,heap,type,first,count,target,offset);return;
    }
    STATE;
    if(!PrepareStatisticsQueries(s.statistics)) {
        HLog("ERROR: cannot prepare active statistics queries for result aggregation");
        return;
    }
    const Bindings saved=s.b;
    const auto active=s.statistics;
    SuspendStatisticsQueries(self,active,m);
    for(UINT k=0;k<count;++k) ResolveStatisticsQuery(self,{heap,type,first+k},target,
                               offset+k*(unsigned(type)==10?112:88),m);
    Restore(self,saved); s.b=saved;
    ResumeStatisticsQueries(self,active,m);
}

void STDMETHODCALLTYPE HPredication(CmdList* self, ID3D12Resource* r, UINT64 offset, D3D12_PREDICATION_OP op) {
    ORIG(PfnPredication, WG_VT_LIST_SET_PREDICATION)(self, r, offset, op);
    STATE;
    s.b.predicationSet = true;
    s.b.predicationBuffer = r;
    s.b.predicationOffset = offset;
    s.b.predicationOp = op;
}

void STDMETHODCALLTYPE HDepthBounds(CmdList* self, FLOAT lo, FLOAT hi) {
    ORIG(PfnDepthBounds, WG_VT_LIST_OM_SET_DEPTH_BOUNDS)(self, lo, hi);
    STATE;
    s.b.boundsSet = true;
    s.b.boundsMin = lo;
    s.b.boundsMax = hi;
}

void STDMETHODCALLTYPE HSamplePositions(CmdList* self, UINT perPixel, UINT pixels, D3D12_SAMPLE_POSITION* p) {
    ORIG(PfnSamplePositions, WG_VT_LIST_SET_SAMPLE_POSITIONS)(self, perPixel, pixels, p);
    STATE;
    s.b.samplesSet = true;
    s.b.samplesPerPixel = perPixel;
    s.b.samplePixels = pixels;
    if (p && perPixel * pixels <= 64) s.b.samples.assign(p, p + perPixel * pixels);
    else s.b.samples.clear();
}

void STDMETHODCALLTYPE HViewMask(CmdList* self, UINT m) {
    ORIG(PfnViewMask, WG_VT_LIST_SET_VIEW_INSTANCE_MASK)(self, m);
    STATE;
    s.b.viewMaskSet = true;
    s.b.viewMask = m;
}

void STDMETHODCALLTYPE HBeginRenderPass(CmdList* self, UINT n, const void* rts, const void* ds, UINT flags) {
    ORIG(PfnBeginRenderPass, WG_VT_LIST_BEGIN_RENDER_PASS)(self, n, rts, ds, flags);
    STATE;
    s.renderPass = true;
}

void STDMETHODCALLTYPE HEndRenderPass(CmdList* self) {
    ORIG(PfnEndRenderPass, WG_VT_LIST_END_RENDER_PASS)(self);
    STATE;
    s.renderPass = false;
}

void STDMETHODCALLTYPE HStateObject(CmdList* self, void* so) {
    ORIG(PfnStateObject, WG_VT_LIST_SET_PIPELINE_STATE1)(self, so);
    STATE;
    s.b.stateObjectSet = true;
    s.b.stateObject = so;
}

void STDMETHODCALLTYPE HShadingRate(CmdList* self, UINT rate, const UINT* combiners) {
    ORIG(PfnShadingRate, WG_VT_LIST_RS_SET_SHADING_RATE)(self, rate, combiners);
    STATE;
    s.b.shadingRateSet = true;
    s.b.shadingRate = rate;
    s.b.combinersNull = !combiners;
    if (combiners) s.b.combiners[0] = combiners[0], s.b.combiners[1] = combiners[1];
}

void STDMETHODCALLTYPE HShadingImage(CmdList* self, void* image) {
    ORIG(PfnShadingImage, WG_VT_LIST_RS_SET_SHADING_RATE_IMAGE)(self, image);
    STATE;
    s.b.shadingImageSet = true;
    s.b.shadingImage = image;
}

#undef STATE

// ---------------------------------------------------------------------------------------------------------------
// Hooks: device and queue

HRESULT STDMETHODCALLTYPE HCreateCommandList(ID3D12Device* self, UINT mask, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator,
                                             ID3D12PipelineState* pso, REFIID riid, void** out) {
    const HRESULT hr = Orig<PfnCreateList>(g_deviceVt, self, WG_VT_DEVICE_CREATE_COMMAND_LIST)(self, mask, type, allocator, pso, riid, out);
    if (SUCCEEDED(hr) && out && *out) {
        ListState& s = StateOf(*out);  // the same object pointer for every ID3D12CommandList interface (single inheritance)
        s.known = true;
        s.allocator = allocator;
        s.openQueries = 0;
        s.renderPass = false;
        s.b = Bindings{};
        if (pso) s.b.psoSet = true, s.b.pso = pso;
        NoteCreated(1, type, allocator, pso, 0, riid, *out, WG_RETURN_ADDRESS());
    }
    return hr;
}

// ID3D12Device4::CreateCommandList1 (a list created closed, without an allocator): observed for the "list facts"
// lines only; such a list is tracked from its first Reset as before.
using PfnCreateList1 = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE, UINT, REFIID, void**);
HRESULT STDMETHODCALLTYPE HCreateCommandList1(ID3D12Device* self, UINT mask, D3D12_COMMAND_LIST_TYPE type, UINT flags, REFIID riid, void** out) {
    const HRESULT hr = Orig<PfnCreateList1>(g_deviceVt, self, WG_VT_DEVICE4_CREATE_COMMAND_LIST1)(self, mask, type, flags, riid, out);
    if (SUCCEEDED(hr) && out && *out) NoteCreated(2, type, nullptr, nullptr, flags, riid, *out, WG_RETURN_ADDRESS());
    return hr;
}

struct Step {
    int kind;  // 0 execute [begin, end), 1 signal, 2 wait
    UINT begin, end;
    ID3D12Fence* fence;
    uint64_t value;
};

void STDMETHODCALLTYPE HExecuteCommandLists(ID3D12CommandQueue* self, UINT n, ID3D12CommandList* const* lists) {
    const auto execute = Orig<PfnExecute>(g_queueVt, self, WG_VT_QUEUE_EXECUTE_COMMAND_LISTS);
    if (!g_installed.load(std::memory_order_relaxed) || !lists) return execute(self, n, lists);
    std::vector<Step> steps;
    std::vector<std::string> lines;
    {
        std::lock_guard<std::mutex> lock(g_splitMutex);
        const uint64_t now = NowUs();
        CheckTimeouts(now);
        QueueInfo& q = QueueOf(self);
        const uint64_t call = q.calls + 1;
        UINT begin = 0;
        for (UINT i = 0; i < n; ++i) {
            const void* list = lists[i];
            // Trace: is this an FSR dispatch list?
            for (size_t k = 0; k < g_notes.size(); ++k) {
                if (g_notes[k].list != list) continue;
                const DispatchNote note = g_notes[k];
                g_notes.erase(g_notes.begin() + std::ptrdiff_t(k));
                g_noteCount.store(int(g_notes.size()));
                g_lastQueue[list] = self;
                ++g_tracedSubmits;
                if (ShouldLogFrame(note.seq)) {
                    char head[512];
                    std::snprintf(head, sizeof head,
                                  "submit: FSR list %p (dispatch D%llu, recorded on thread %lu, %.2f ms ago) in ExecuteCommandLists #%llu on Q%u "
                                  "(%s, priority %d, thread %lu), position %u of %u",
                                  list, static_cast<unsigned long long>(note.seq), note.thread, double(now - note.timeUs) / 1000.0,
                                  static_cast<unsigned long long>(call), q.id, ListTypeName(q.type), q.priority, GetCurrentThreadId(), i + 1, n);
                    lines.push_back(std::string(head) + "; submitted after the dispatch was recorded, before this call: " + Since(note.at) +
                                    "; since the previous FSR list was submitted: " + Since(g_atPreviousSubmit) +
                                    (i ? "; earlier in this call: " + std::to_string(i) + " lists" : ""));
                }
                g_atPreviousSubmit = Snapshot();
                break;
            }
            // Splits: release the first halves at this list's place in the game's order.
            if (!g_pendingTotal.load()) continue;
            ListState* s = FindState(list);
            if (!s || s->pending.empty()) continue;
            if (i > begin) steps.push_back({0, begin, i, nullptr, 0});
            for (PendingSplit& p : s->pending) {
#if defined(WRONGGPU_SPLIT_PROGRESS_QA)
                if(g_progressReady.load()) ProcessSplitProgress().Submitted(p.id,self);
#endif
                if (!p.released) steps.push_back({1, 0, 0, p.lane->order, p.ticket});
                if (p.queryDone) steps.push_back({2,0,0,p.queryDone,p.queryDoneValue});
                steps.push_back({2, 0, 0, p.resume, p.resumeValue});
                if (ShouldLogFrame(p.id)) {
                    char text[384];
                    std::snprintf(text, sizeof text,
                                  "split S%llu released in ExecuteCommandLists #%llu on Q%u, position %u of %u: %s wait for resume %p >= %llu, then the second half",
                                  static_cast<unsigned long long>(p.id), static_cast<unsigned long long>(call), q.id, i + 1, n,
                                  p.released ? "(order already released from the CPU)" : "signal WrongGPU's order ticket, then",
                                  static_cast<void*>(p.resume), static_cast<unsigned long long>(p.resumeValue));
                    lines.push_back(text);
                }
                if (--p.lane->pending == 0) p.lane->list = nullptr;
                g_pendingTotal.fetch_sub(1);
            }
            s->pending.clear();
            begin = i;
        }
        if (begin < n) steps.push_back({0, begin, n, nullptr, 0});
        q.calls += 1;
        q.lists += n;
    }
    for (const Step& st : steps) {
        if (st.kind == 0) execute(self, st.end - st.begin, lists + st.begin);
        else if (st.kind == 1) self->Signal(st.fence, st.value);
        else self->Wait(st.fence, st.value);
    }
    for (const std::string& line : lines) HLog("%s", line.c_str());
}

// ---------------------------------------------------------------------------------------------------------------
// Installation

// IIDs of ID3D12GraphicsCommandList4/5 (Windows SDK d3d12.h); the MinGW-w64 headers do not declare these interfaces.
#if defined(_MSC_VER) && defined(__ID3D12GraphicsCommandList5_INTERFACE_DEFINED__)
const GUID kIidList4 = __uuidof(ID3D12GraphicsCommandList4);
const GUID kIidList5 = __uuidof(ID3D12GraphicsCommandList5);
#else
const GUID kIidList4 = {0x8754318e, 0xd3a9, 0x4541, {0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74}};
const GUID kIidList5 = {0x55050859, 0x4024, 0x474c, {0x87, 0xf5, 0x64, 0x72, 0xea, 0xee, 0x44, 0xea}};
#endif

bool Supports(IUnknown* o, const GUID& iid) {
    void* p = nullptr;
    if (FAILED(o->QueryInterface(iid, &p)) || !p) return false;
    static_cast<IUnknown*>(p)->Release();
    return true;
}

// WrongGPU's hooks for the interfaces `list` implements.
std::vector<SlotHook> ListHooks(CmdList* list, bool* has1 = nullptr, bool* has4 = nullptr, bool* has5 = nullptr) {
    std::vector<SlotHook> h = {
        {WG_VT_LIST_CLOSE, reinterpret_cast<void*>(&HClose)},
        {WG_VT_LIST_RESET, reinterpret_cast<void*>(&HReset)},
        {WG_VT_LIST_CLEAR_STATE, reinterpret_cast<void*>(&HClearState)},
        {WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY, reinterpret_cast<void*>(&HTopology)},
        {WG_VT_LIST_RS_SET_VIEWPORTS, reinterpret_cast<void*>(&HViewports)},
        {WG_VT_LIST_RS_SET_SCISSOR_RECTS, reinterpret_cast<void*>(&HScissors)},
        {WG_VT_LIST_OM_SET_BLEND_FACTOR, reinterpret_cast<void*>(&HBlend)},
        {WG_VT_LIST_OM_SET_STENCIL_REF, reinterpret_cast<void*>(&HStencil)},
        {WG_VT_LIST_SET_PIPELINE_STATE, reinterpret_cast<void*>(&HPso)},
        {WG_VT_LIST_EXECUTE_BUNDLE, reinterpret_cast<void*>(&HBundle)},
        {WG_VT_LIST_SET_DESCRIPTOR_HEAPS, reinterpret_cast<void*>(&HHeaps)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_SIGNATURE, reinterpret_cast<void*>(&HComputeSig)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_SIGNATURE, reinterpret_cast<void*>(&HGraphicsSig)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE, reinterpret_cast<void*>(&HComputeTable)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE, reinterpret_cast<void*>(&HGraphicsTable)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANT, reinterpret_cast<void*>(&HComputeConstant)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANT, reinterpret_cast<void*>(&HGraphicsConstant)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANTS, reinterpret_cast<void*>(&HComputeConstants)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS, reinterpret_cast<void*>(&HGraphicsConstants)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_CBV, reinterpret_cast<void*>(&HComputeCbv)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_CBV, reinterpret_cast<void*>(&HGraphicsCbv)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_SRV, reinterpret_cast<void*>(&HComputeSrv)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_SRV, reinterpret_cast<void*>(&HGraphicsSrv)},
        {WG_VT_LIST_SET_COMPUTE_ROOT_UAV, reinterpret_cast<void*>(&HComputeUav)},
        {WG_VT_LIST_SET_GRAPHICS_ROOT_UAV, reinterpret_cast<void*>(&HGraphicsUav)},
        {WG_VT_LIST_IA_SET_INDEX_BUFFER, reinterpret_cast<void*>(&HIndexBuffer)},
        {WG_VT_LIST_IA_SET_VERTEX_BUFFERS, reinterpret_cast<void*>(&HVertexBuffers)},
        {WG_VT_LIST_SO_SET_TARGETS, reinterpret_cast<void*>(&HSoTargets)},
        {WG_VT_LIST_OM_SET_RENDER_TARGETS, reinterpret_cast<void*>(&HRenderTargets)},
        {WG_VT_LIST_BEGIN_QUERY, reinterpret_cast<void*>(&HBeginQuery)},
        {WG_VT_LIST_END_QUERY, reinterpret_cast<void*>(&HEndQuery)},
        {WG_VT_LIST_RESOLVE_QUERY_DATA,reinterpret_cast<void*>(&HResolveQuery)},
        {WG_VT_LIST_SET_PREDICATION, reinterpret_cast<void*>(&HPredication)},
    };
    const bool v1 = Supports(list, __uuidof(ID3D12GraphicsCommandList1));
    const bool v4 = v1 && Supports(list, kIidList4);
    const bool v5 = v4 && Supports(list, kIidList5);
    if (v1) {
        h.push_back({WG_VT_LIST_OM_SET_DEPTH_BOUNDS, reinterpret_cast<void*>(&HDepthBounds)});
        h.push_back({WG_VT_LIST_SET_SAMPLE_POSITIONS, reinterpret_cast<void*>(&HSamplePositions)});
        h.push_back({WG_VT_LIST_SET_VIEW_INSTANCE_MASK, reinterpret_cast<void*>(&HViewMask)});
    }
    if (v4) {
        h.push_back({WG_VT_LIST_BEGIN_RENDER_PASS, reinterpret_cast<void*>(&HBeginRenderPass)});
        h.push_back({WG_VT_LIST_END_RENDER_PASS, reinterpret_cast<void*>(&HEndRenderPass)});
        h.push_back({WG_VT_LIST_SET_PIPELINE_STATE1, reinterpret_cast<void*>(&HStateObject)});
    }
    if (v5) {
        h.push_back({WG_VT_LIST_RS_SET_SHADING_RATE, reinterpret_cast<void*>(&HShadingRate)});
        h.push_back({WG_VT_LIST_RS_SET_SHADING_RATE_IMAGE, reinterpret_cast<void*>(&HShadingImage)});
    }
    if (has1) *has1 = v1;
    if (has4) *has4 = v4;
    if (has5) *has5 = v5;
    return h;
}

uint64_t g_foreignTables = 0;  // lists refused by the rule above (under g_patchMutex)

// True when the list's vtable carries WrongGPU's hooks afterwards (it did already, or it was patched now). False:
// the table was left untouched (another layer's per-object table, or no unambiguous originals) and the list must
// not be split.
bool PatchList(CmdList* list, bool deviceOwn = false) {
    void** vt = *reinterpret_cast<void***>(list);
    if (IsPatched(g_listVt, vt)) return true;
    if (!deviceOwn && WrapperTable(vt)) {
        static std::atomic<void*> logged[kMaxKnown];  // one line per wrapper class table
        static std::atomic<int> loggedCount{0};
        const bool first = !Known(logged, loggedCount, static_cast<void*>(vt));
        if (first) Remember(logged, loggedCount, static_cast<void*>(vt));
        if (first)
            HLog("command list %p: vtable %p in %s is another layer's class table (a wrapper object around a list of the device; its "
                 "calls reach that list, whose table WrongGPU patched): left untouched, its calls are hooked once, through the list "
                 "it wraps; it is never split",
                 static_cast<void*>(list), static_cast<void*>(vt), Where(vt).c_str());
        return false;
    }
    bool v1 = false, v4 = false, v5 = false;
    const std::vector<SlotHook> h = ListHooks(list, &v1, &v4, &v5);
    const TableFacts facts = Inspect(vt, h);
    if (OwnedByAnotherLayer(facts)) {
        if (++g_foreignTables <= 3 || g_foreignTables % 600 == 0)
            HLog("command list %p: vtable %p is its own table outside every module, and %d of the %zu entries WrongGPU hooks belong to "
                 "another hook layer (%d more are copies of WrongGPU's hooks): left untouched, this list is not split%s",
                 static_cast<void*>(list), static_cast<void*>(vt), facts.foreign, h.size(), facts.inherited,
                 g_foreignTables == 3 ? " (further lists of this kind: one in 600 logged)" : "");
        return false;
    }
    if (!Patch(g_listVt, vt, h, "ID3D12GraphicsCommandList")) return false;
    if (deviceOwn) Remember(g_listModules, g_listModuleCount, ModuleOf(vt));
    HLog("command list interfaces of vtable %p: ID3D12GraphicsCommandList1 %s, 4 %s, 5 %s", static_cast<void*>(vt), v1 ? "yes" : "no",
         v4 ? "yes" : "no", v5 ? "yes" : "no");
    return true;
}

// The reason a dispatch into `list` can never be split, or null. Reads only (any thread).
const char* BlockedImpl(void* listPtr) {
    if (!listPtr || !g_installed.load()) return nullptr;
    auto* list = static_cast<CmdList*>(listPtr);
    void** vt = *reinterpret_cast<void***>(list);
    // clean/fe-wrapped-list: a split already refused for the list's device or its wrapper class (not before: a backend that
    // never splits, e.g. the D3D12 executor, keeps the context)
    if (Known(g_refusedTables, g_refusedTableCount, static_cast<void*>(vt))) return kWrapperReason;
    if (ListState* st = FindState(list); st && st->notOurDevice) return kNotOurDeviceReason;
    if (IsPatched(g_listVt, vt)) return nullptr;
    // A list that records into its own vtable: the split takes the table over and answers WG_ERROR_NOT_READY until the
    // list is tracked from a Reset (the stand-in upscales those frames). Only what cannot get better is a reason here.
    if (g_ownLoop.load(std::memory_order_relaxed)) return kOwnLoopReason;
    if (g_ownStrikes.load(std::memory_order_relaxed) >= 3) return kOwnRewrittenReason;
    if (g_ownNotReady.load(std::memory_order_relaxed) >= 300) return kOwnNeverReason;
    ListState* state = FindState(list);
    if (state && state->own && (state->own->table == vt || state->own->vetted == vt)) return nullptr;
    if (!(state && state->own && state->own->rejected == vt) && !ModuleOf(vt) && ClassOf(vt, nullptr)) return nullptr;
    const std::vector<SlotHook> hooks = ListHooks(list);
    if (!OwnedByAnotherLayer(Inspect(vt, hooks))) return nullptr;
    DescribeForeignList(list, hooks);
    return kForeignTableReason;
}

void PatchQueueOfType(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type) {
    D3D12_COMMAND_QUEUE_DESC d{};
    d.Type = type;
    ID3D12CommandQueue* q = nullptr;
    if (FAILED(device->CreateCommandQueue(&d, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&q))) || !q) {
        HLog("note: no %s queue for the vtable", ListTypeName(type));
        return;
    }
    Patch(g_queueVt, *reinterpret_cast<void***>(q), {{WG_VT_QUEUE_EXECUTE_COMMAND_LISTS, reinterpret_cast<void*>(&HExecuteCommandLists)}},
          "ID3D12CommandQueue");
    q->Release();
}

void PatchListOfType(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type) {
    ID3D12CommandAllocator* a = nullptr;
    CmdList* list = nullptr;
    if (SUCCEEDED(device->CreateCommandAllocator(type, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&a))) && a &&
        SUCCEEDED(device->CreateCommandList(0, type, a, nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&list))) && list) {
        PatchList(list, true);  // a list the device itself made: its class table's module is a list module
        list->Close();
    } else {
        HLog("note: no %s command list for the vtable", ListTypeName(type));
    }
    if (list) list->Release();
    if (a) a->Release();
}

// An upscaler context was created on `device` (Install without a list): splits are allowed on its lists when it is the
// D3D12 runtime's own object (ProvenDevice). Under g_patchMutex.
void NoteContextDevice(void* device) {
    if (!device || Known(g_contextDevices, g_contextDeviceCount, device)) return;
    Remember(g_contextDevices, g_contextDeviceCount, device);
    void** vt = *static_cast<void***>(device);
    if (!RuntimeModule(ModuleOf(vt)))
        HLog("note: the upscaler's device %p has its vtable %p in %s, not in the D3D12 runtime (a layer's device wrapper): command "
             "lists of it are not split (the D3D12 executor is not affected)",
             device, static_cast<void*>(vt), Where(vt).c_str());
}

bool Install(void* devicePtr, void* listPtr, const char* reason) {
    if (g_installed.load()) {
        std::lock_guard<std::mutex> lock(g_patchMutex);
        if (!listPtr) NoteContextDevice(devicePtr);
        if (listPtr) PatchList(static_cast<CmdList*>(listPtr));  // a list of a class not seen yet
        return true;
    }
    std::lock_guard<std::mutex> lock(g_patchMutex);
    if (!listPtr) NoteContextDevice(devicePtr);
    if (g_installed.load()) return true;
    if (g_installFailed || !devicePtr) return false;
    auto* device = static_cast<ID3D12Device*>(devicePtr);
    HLog("installing WrongGPU's D3D12 vtable hooks (%s) on device %p", reason, devicePtr);
    std::vector<SlotHook> deviceHooks{{WG_VT_DEVICE_CREATE_COMMAND_LIST, reinterpret_cast<void*>(&HCreateCommandList)}};
    const GUID iidDevice4 = {0xe865df17, 0xa9ee, 0x46f9, {0xa4, 0x63, 0x30, 0x98, 0x31, 0x5a, 0xa2, 0xe5}};  // ID3D12Device4
    if (Supports(device, iidDevice4)) deviceHooks.push_back({WG_VT_DEVICE4_CREATE_COMMAND_LIST1, reinterpret_cast<void*>(&HCreateCommandList1)});
    Patch(g_deviceVt, *reinterpret_cast<void***>(device), deviceHooks, "ID3D12Device");
    PatchQueueOfType(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    PatchQueueOfType(device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    PatchQueueOfType(device, D3D12_COMMAND_LIST_TYPE_COPY);
    PatchListOfType(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    PatchListOfType(device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    PatchListOfType(device, D3D12_COMMAND_LIST_TYPE_BUNDLE);
    if (listPtr) PatchList(static_cast<CmdList*>(listPtr));  // after the device's own lists (the list modules)
    if (!g_queueVt.count.load() || !g_listVt.count.load()) {
        HLog("ERROR: hooks incomplete (queue vtables %d, command list vtables %d): no split, no trace", g_queueVt.count.load(), g_listVt.count.load());
        g_installFailed = true;
        return false;
    }
    // The vtables now point into this DLL: keep it loaded for the life of the process.
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, reinterpret_cast<LPCWSTR>(&Install), &self);
#if defined(WRONGGPU_SPLIT_PROGRESS_QA)
    g_progressReady.store(StartSplitProgressWorker(&ProgressLog,nullptr,&NowUs));
    HLog("[split-progress-qa] read-only CPU observer %s; no extra GPU commands, waits or signals",g_progressReady.load()?"started":"unavailable");
#endif
    g_installed.store(true);
    HLog("installed (device %d, queue %d, command list %d vtables); split timeout %u ms. Command lists already being recorded are tracked "
         "from their next Reset; until then a split of them answers WG_ERROR_NOT_READY",
         g_deviceVt.count.load(), g_queueVt.count.load(), g_listVt.count.load(), g_settings.timeoutMs);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// The split service

void Refuse(uint64_t& counter, const void* list, const char* code, const char* why) {
    ++counter;
    if (counter <= 20 || counter % 600 == 0)
        HLog("split of list %p refused (%s): %s%s", list, code, why, counter == 20 ? " (further refusals: one in 600 logged)" : "");
}

// Is everything the list recorded since its Reset known, so that it can be split? WG_OK, or the refusal (logged).
// ownTable: the list is tracked through its own vtable.
WgResult TrackList(CmdList* list, bool& ownTable) {
    ownTable = false;
    if (void** vt = *reinterpret_cast<void***>(list); !IsPatched(g_listVt, vt) && WrapperTable(vt)) {
        Remember(g_refusedTables, g_refusedTableCount, static_cast<void*>(vt));
        const std::string why = std::string(kWrapperReason) + " (vtable " + Where(vt) + "); the context gets AMD's FSR, else the stand-in";
        Refuse(g_refused, list, "WG_ERROR_NOT_READY", why.c_str());
        return WG_ERROR_NOT_READY;
    }
    if (void** vt = *reinterpret_cast<void***>(list); !IsPatched(g_listVt, vt)) {
        ListState& own = StateOf(list);
        const bool was = own.own && own.own->table == vt && own.own->tracked;
        if (was && OwnIntact(*own.own)) {
            ownTable = true;  // tracked through its own table since its Reset
        } else if (Adopt(list, own, vt, false)) {
            // Taken over (again) now: calls made before this were not seen, so not this frame.
            if (was) g_ownStrikes.fetch_add(1);
            g_ownNotReady.fetch_add(1);
            own.known = false;
            Refuse(g_refused, list, "WG_ERROR_NOT_READY",
                   was ? "the list's own vtable was rewritten while the list recorded; taken over again, tracked from the list's next Reset"
                       : "the list records into its own vtable, which WrongGPU took over now: tracked from the list's next Reset");
            return WG_ERROR_NOT_READY;
        } else if (!ModuleOf(vt)) {
            // A table of its own that WrongGPU cannot take over: never written to, never split.
            const char* why = BlockedImpl(list);
            Refuse(g_refused, list, "WG_ERROR_NOT_READY", why ? why : "the list's own vtable could not be taken over");
            return WG_ERROR_NOT_READY;
        }
    }
    if (!ownTable && !PatchList(list)) {
        // Nothing was written and the list is still open and unchanged. NOT_READY: the backend draws nothing into it
        // and the stand-in upscales the frame when the front end could not give the context to AMD's FSR.
        Refuse(g_refused, list, "WG_ERROR_NOT_READY",
               BlockedImpl(list) ? kForeignTableReason : "WrongGPU's hooks could not be written to the list's vtable");
        return WG_ERROR_NOT_READY;
    }
    return WG_OK;
}

WgResult SplitImpl(const WgFrame* frame, const WgSplitRequest* request) {
    std::lock_guard<std::mutex> once(g_patchMutex);  // one split at a time (splits are rare: one per frame)
    const void* listPtr = frame ? frame->command_list : nullptr;
    if (!frame || !request || !listPtr || request->struct_size < sizeof(WgSplitRequest) || !request->done_fence || !request->resume_fence) {
        Refuse(g_refused, listPtr, "WG_ERROR_PARAMETER", "missing frame, command list or fence");
        return WG_ERROR_PARAMETER;
    }
    if (!g_installed.load()) {
        Refuse(g_refused, listPtr, "WG_ERROR_UNSUPPORTED", "WrongGPU's D3D12 hooks are not installed");
        return WG_ERROR_UNSUPPORTED;
    }
    auto* list = static_cast<CmdList*>(const_cast<void*>(listPtr));
    bool ownTable = false;
    if (const WgResult tracked = TrackList(list, ownTable); tracked != WG_OK) return tracked;
    ListState& s = StateOf(list);
    if (!s.known || !s.allocator) {
        Refuse(g_refused, list, "WG_ERROR_NOT_READY", "the list's Reset has not been seen since the hooks were installed");
        return WG_ERROR_NOT_READY;
    }
    if (s.openQueries > int(s.statistics.size())) {
        Refuse(g_refused, list, "WG_ERROR_UNSUPPORTED", "a query is open (BeginQuery without EndQuery)");
        return WG_ERROR_UNSUPPORTED;
    }
    const auto statistics=s.statistics;
    if(!statistics.empty() && !PrepareStatisticsQueries(statistics)) {
        Refuse(g_refused,list,"WG_ERROR_UNSUPPORTED","pipeline-statistics continuation could not be prepared");
        return WG_ERROR_UNSUPPORTED;
    }
    if (s.renderPass) {
        Refuse(g_refused, list, "WG_ERROR_UNSUPPORTED", "inside a render pass");
        return WG_ERROR_UNSUPPORTED;
    }
    const D3D12_COMMAND_LIST_TYPE type = list->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        Refuse(g_refused, list, "WG_ERROR_UNSUPPORTED", "command list type is neither DIRECT nor COMPUTE");
        return WG_ERROR_UNSUPPORTED;
    }
    ID3D12Device* device = nullptr;
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device))) || !device) {
        Refuse(g_refused, list, "WG_ERROR_DEVICE", "GetDevice failed");
        return WG_ERROR_DEVICE;
    }
    device->Release();  // the game keeps its device alive
    if (!ProvenDevice(device)) {
        s.notOurDevice = true;
        const std::string why = std::string(kNotOurDeviceReason) + " (device " + Where(*reinterpret_cast<void**>(device)) +
                                "); no WrongGPU queue is made on it; the context gets AMD's FSR, else the stand-in";
        Refuse(g_refused, list, "WG_ERROR_NOT_READY", why.c_str());
        return WG_ERROR_NOT_READY;
    }

    // Choose a lane: the one already carrying this list's splits, else a free one, else a new one.
    Lane* lane = nullptr;
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(g_splitMutex);
        CheckTimeouts(NowUs());
        size_t ofType = 0;
        for (Lane* l : g_lanes) {
            if (l->device != device || l->type != type) continue;
            ++ofType;
            if (l->list == list) lane = l;
        }
        for (size_t i = 0; !lane && i < g_lanes.size(); ++i)
            if (g_lanes[i]->device == device && g_lanes[i]->type == type && !g_lanes[i]->pending) lane = g_lanes[i];
        if (!lane && ofType == 0) DropRemovedLanes(device);  // a device's first lane of the type: a new device? (bug hunt F3)
        if (!lane && ofType < kLanesPerType) {
            auto* l = new Lane();
            l->id = ++g_laneIds;
            l->device = device;
            l->type = type;
            D3D12_COMMAND_QUEUE_DESC d{};
            d.Type = type;
            if (FAILED(device->CreateCommandQueue(&d, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&l->queue))) ||
                FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&l->order)))) {
                if (l->queue) l->queue->Release();
                delete l;
                Refuse(g_refused, list, "WG_ERROR_DEVICE", "creating WrongGPU's queue or order fence failed");
                return WG_ERROR_DEVICE;
            }
            l->queue->SetName(L"WrongGPU split queue");
            g_lanes.push_back(l);
            lane = l;
            HLog("WrongGPU split queue W%u created: %s, on device %p", l->id, ListTypeName(type), static_cast<void*>(device));
        }
        if (!lane) {
            Refuse(g_refused, list, "WG_ERROR_NOT_READY", "4 other lists of this type have splits waiting for their submission");
            return WG_ERROR_NOT_READY;
        }
        lane->list = list;
        ++lane->pending;  // reserved; registered below
        g_pendingTotal.fetch_add(1);
        id = ++g_splitIds;
    }
    auto unreserve = [&] {
        std::lock_guard<std::mutex> lock(g_splitMutex);
        if (--lane->pending == 0) lane->list = nullptr;
        g_pendingTotal.fetch_sub(1);
    };

    // End and resolve each statistics segment before closing. The API forbids an
    // active query spanning command lists; later resolution adds both segments.
    const Bindings beforeQueries=s.b;
    if(!statistics.empty()) {
        SuspendStatisticsQueries(list,statistics,Methods(list));
        Restore(list,beforeQueries);s.b=beforeQueries;
    }
    // First half: close and queue it behind the order ticket; the GPU signals done after it.
    HRESULT hr = ListOrig<PfnClose>(list, WG_VT_LIST_CLOSE)(list);
    if (FAILED(hr)) {
        unreserve();
        Refuse(g_refused, list, "WG_ERROR_DEVICE", ("Close failed (0x" + std::to_string(unsigned(hr)) + "); the list is in an error state").c_str());
        return WG_ERROR_DEVICE;
    }
    const uint64_t ticket = ++lane->ticket;
    auto* done = static_cast<ID3D12Fence*>(request->done_fence);
    auto* resume = static_cast<ID3D12Fence*>(request->resume_fence);
    lane->queue->Wait(lane->order, ticket);
    static std::atomic<bool> firstSubmit{false};
    if (!firstSubmit.exchange(true)) {  // written to disk now: a crash in the submission still leaves this line
        HLog("about to submit the first split: list %p, its first half on WrongGPU's queue W%u (device %p)", static_cast<void*>(list),
             lane->id, static_cast<void*>(device));
        Log::Get().Flush();
    }
    ID3D12CommandList* first = list;
    Orig<PfnExecute>(g_queueVt, lane->queue, WG_VT_QUEUE_EXECUTE_COMMAND_LISTS)(lane->queue, 1, &first);
    lane->queue->Signal(done, request->done_value);

    // Second half: the same list object, reset with its allocator, the game's bindings restored.
    {
        InternalQueryScope scope;
        void** const before = *reinterpret_cast<void***>(list);
        hr = ListOrig<PfnReset>(list, WG_VT_LIST_RESET)(list, s.allocator, nullptr);
        // The runtime may have written the list's own table again: hooks and originals first, then the bindings.
        if (SUCCEEDED(hr) && ownTable) {
            AfterReset(list, s, before);
            if (!(s.own && s.own->tracked)) {
                g_ownStrikes.store(3);
                HLog("ERROR: split S%llu: the list's own vtable could not be taken over again after the split's Reset; lists of this kind are "
                     "not split any more", static_cast<unsigned long long>(id));
            }
        }
    }
    if (FAILED(hr)) {
        // The first half is queued: keep it in the game's order; the list cannot be reopened.
        std::lock_guard<std::mutex> lock(g_splitMutex);
        s.pending.push_back({lane, ticket, resume, request->resume_value, id, NowUs(), false});
        HLog("ERROR: split S%llu: Reset of list %p after the split failed (0x%08lx); the game's further commands into it will fail",
             static_cast<unsigned long long>(id), static_cast<const void*>(list), static_cast<unsigned long>(hr));
        return WG_ERROR_DEVICE;
    }
    const int restored = Restore(list, s.b);
    if(!statistics.empty()) ResumeStatisticsQueries(list,statistics,Methods(list));
    {
        std::lock_guard<std::mutex> lock(g_splitMutex);
        s.pending.push_back({lane, ticket, resume, request->resume_value, id, NowUs(), false,
                             statistics.empty()?nullptr:done,request->done_value});
#if defined(WRONGGPU_SPLIT_PROGRESS_QA)
        // Register under the same submission lock, so an Execute/CPU-release annotation cannot precede this record.
        if(g_progressReady.load() && !ProcessSplitProgress().Track(id,list,device,lane->order,ticket,done,request->done_value,
                                                                resume,request->resume_value,NowUs()) && ShouldLogFrame(id))
            HLog("[split-progress-qa] S%llu observer record dropped (capacity/allocation)",static_cast<unsigned long long>(id));
#endif
    }
    if(!statistics.empty() && ShouldLogFrame(id)) HLog("split S%llu: continued %zu pipeline-statistics queries with exact result aggregation",
                    static_cast<unsigned long long>(id),statistics.size());
    if (ownTable) {
        g_ownNotReady.store(0);
        if (s.own && s.own->tracked) g_ownStrikes.store(0);
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            HLog("own table: split S%llu is the first split of a list that records into its own vtable (tracked through that table since "
                 "the list's Reset)", static_cast<unsigned long long>(id));
    }
    if (ShouldLogFrame(id))
        HLog("split S%llu of list %p (%s): first half on W%u behind order ticket %llu, done %p = %llu; list reset with allocator %p, "
             "%d bindings restored; second half waits for resume %p >= %llu",
             static_cast<unsigned long long>(id), static_cast<const void*>(list), ListTypeName(type), lane->id,
             static_cast<unsigned long long>(ticket), static_cast<void*>(done), static_cast<unsigned long long>(request->done_value),
             static_cast<void*>(s.allocator), restored, static_cast<void*>(resume), static_cast<unsigned long long>(request->resume_value));
    return WG_OK;
}

WgResult WGAPI_Split(void*, const WgFrame* frame, const WgSplitRequest* request) {
    try {
        return SplitImpl(frame, request);
    } catch (...) {
        HLog("ERROR: exception in split_command_list");
        return WG_ERROR;
    }
}

const WgFrontEndServices g_services = {sizeof(WgFrontEndServices), nullptr, &WGAPI_Split};

void NoteImpl(void* listPtr, WgFrame* frame) {
    if (!listPtr) return;
    auto* list = static_cast<CmdList*>(listPtr);
    if (g_settings.mode == 2 && !g_installed.load()) {
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device))) && device) {
            device->Release();
            Install(device, list, "d3d12_hooks = on");
        }
    }
    if (frame) frame->command_list_type = uint32_t(list->GetType());
    if (!g_installed.load()) return;
    if (g_describeOpen.load(std::memory_order_relaxed)) {
        void** table = *reinterpret_cast<void***>(list);
        if (!IsPatched(g_listVt, table)) {
            const std::vector<SlotHook> hooks = ListHooks(list);
            if (OwnedByAnotherLayer(Inspect(table, hooks))) DescribeForeignList(list, hooks);
        }
    }
    static std::atomic<int> firstLists{0};
    if (firstLists.load(std::memory_order_relaxed) < 2 && firstLists.fetch_add(1) < 2) {
        void** vt = *reinterpret_cast<void***>(list);
        ListState* s = FindState(list);
        HLog("list facts: dispatch list %p: vtable %p in %s%s; creation %s", listPtr, static_cast<void*>(vt), Where(vt).c_str(),
             IsPatched(g_listVt, vt) ? " (a table WrongGPU patched)" : "",
             !s || !s->createdBy ? "not seen" : s->createdBy == 2 ? "by CreateCommandList1" : "by CreateCommandList");
    }
    std::lock_guard<std::mutex> lock(g_splitMutex);
    if (frame) {
        auto it = g_lastQueue.find(list);
        frame->command_queue = it != g_lastQueue.end() ? const_cast<void*>(it->second) : nullptr;
    }
    for (DispatchNote& n : g_notes)
        if (n.list == list) {
            ++n.notes;
            return;
        }
    DispatchNote n;
    n.list = list;
    n.seq = ++g_dispatchSeq;
    n.notes = 1;
    n.thread = GetCurrentThreadId();
    n.timeUs = NowUs();
    n.at = Snapshot();
    if (g_notes.size() >= 64) {
        HLog("note: 64 FSR lists recorded but not submitted; the oldest (%p, D%llu) is dropped from the trace", g_notes.front().list,
             static_cast<unsigned long long>(g_notes.front().seq));
        g_notes.erase(g_notes.begin());
    }
    g_notes.push_back(std::move(n));
    g_noteCount.store(int(g_notes.size()));
}

// ---------------------------------------------------------------------------------------------------------------
// One owner per process

struct SharedApi {
    uint32_t size;
    uint32_t layout;
    char version[32];
    bool (*install)(void* device, void* list, const char* reason);
    void (*note)(void* list, WgFrame* frame);
    const WgFrontEndServices* services;
    bool (*installed)();
    const char* owner;
    const char* (*blocked)(void* list);  // layout 2
};

bool InstalledImpl() { return g_installed.load(); }

constexpr uint32_t kSharedLayout = 2;
const SharedApi g_mine = {sizeof(SharedApi), kSharedLayout, WRONGGPU_FSR_VERSION, &Install, &NoteImpl, &g_services, &InstalledImpl, nullptr,
                          &BlockedImpl};

const SharedApi* Owner() {
    static const SharedApi* owner = [] () -> const SharedApi* {
        wchar_t name[96];
        std::swprintf(name, 96, L"Local\\WrongGPU-FSR-D3D12Hooks-%lu", GetCurrentProcessId());
        HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(void*), name);
        void* view = h ? MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(void*)) : nullptr;
        if (!view) {
            HLog("ERROR: per-process hook registry not available (error %lu); this DLL owns its own hooks", GetLastError());
            return &g_mine;
        }
        // Handle and view stay open for the life of the process.
        void* prev = InterlockedCompareExchangePointer(static_cast<void* volatile*>(view), const_cast<SharedApi*>(&g_mine), nullptr);
        const auto* o = prev ? static_cast<const SharedApi*>(prev) : &g_mine;
        if (o != &g_mine && (o->size != sizeof(SharedApi) || o->layout != kSharedLayout || std::strcmp(o->version, WRONGGPU_FSR_VERSION) != 0)) {
            HLog("ERROR: the D3D12 hooks belong to another WrongGPU DLL of a different build (%s); no split or trace from this one",
                 o->size == sizeof(SharedApi) ? o->version : "?");
            return nullptr;
        }
        if (o == &g_mine) {
            HMODULE self = nullptr;  // other DLLs call into this one: keep it loaded
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, reinterpret_cast<LPCWSTR>(&Owner), &self);
        }
        return o;
    }();
    return owner;
}

}  // namespace

void ConfigureHooks(const wgfsr::Config& config) {
    g_settings.mode = config.d3d12Hooks == "off" ? 0 : config.d3d12Hooks == "on" ? 2 : 1;
    g_settings.timeoutMs = config.splitTimeoutMs;
    g_settings.logAll = config.logDispatches;
    g_settings.logMaxMegabytes = config.logMaxMegabytes;
}

const WgFrontEndServices* SplitServices(const WgUpscalerBackend* backend, void* device) {
    if (g_settings.mode == 0 || !device || wgfsr::BackendVersion(backend) < 2) return nullptr;
    const bool needs = (wgfsr::BackendFlags(backend) & WG_BACKEND_NEEDS_SPLIT) != 0;
    const SharedApi* o = Owner();
    if (!o) return nullptr;
    if (needs && !o->install(device, nullptr, "a backend sets WG_BACKEND_NEEDS_SPLIT")) return nullptr;
    return o->installed() ? o->services : nullptr;
}

const WgFrontEndServices* SplitServicesDeferred(const WgUpscalerBackend* backend, void* device) {
    if (g_settings.mode == 0 || !device || wgfsr::BackendVersion(backend) < 2) return nullptr;
    const bool needs = (wgfsr::BackendFlags(backend) & WG_BACKEND_NEEDS_SPLIT) != 0;
    const SharedApi* o = Owner();
    if (!o) return nullptr;
    if (o->installed()) return o->services;
    // Not installed yet: offered to a backend that needs the split (ActivateSplitServices installs the hooks after its
    // create succeeded); an installation that failed before is not tried again.
    if (!needs) return nullptr;
    if (o == &g_mine && g_installFailed) return nullptr;
    return o->services;
}

bool ActivateSplitServices(const WgUpscalerBackend* backend, void* device) {
    if (g_settings.mode == 0 || !device || wgfsr::BackendVersion(backend) < 2) return true;
    if (!(wgfsr::BackendFlags(backend) & WG_BACKEND_NEEDS_SPLIT)) return true;
    const SharedApi* o = Owner();
    if (!o) return false;
    return o->install(device, nullptr, "a backend sets WG_BACKEND_NEEDS_SPLIT") && o->installed();
}

const char* ListSplitBlocked(void* commandList) {
    if (g_settings.mode == 0 || !commandList) return nullptr;
    const SharedApi* o = Owner();
    return o && o->installed() ? o->blocked(commandList) : nullptr;
}

void NoteDispatch(void* commandList, WgFrame* frame) {
    if (g_settings.mode == 0 || !commandList) return;
    if (g_settings.mode == 1 && !g_installed.load()) {
        // auto: hooks only for a backend that needs them; another DLL may own them already.
        const SharedApi* o = Owner();
        if (!o || !o->installed()) {
            if (frame) frame->command_list_type = uint32_t(static_cast<CmdList*>(commandList)->GetType());
            return;
        }
        o->note(commandList, frame);
        return;
    }
    const SharedApi* o = Owner();
    if (o) o->note(commandList, frame);
}

}  // namespace wgwin
