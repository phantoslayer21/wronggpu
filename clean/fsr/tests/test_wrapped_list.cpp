// SPDX-License-Identifier: MIT
// Command lists that record into their own vtable (d3d12_hooks.cpp, "Lists that record into their own vtable"), on a
// fake of the runtime's behaviour seen in 007 First Light: a closed list points at the class table, its Reset switches
// it to a table inside the list object in which the runtime has written the driver's entry points. Also the tables
// WrongGPU must leave alone, and the safety net for an entry that calls back into the hook. Case F (clean/fe-wrapped-list):
// another layer's wrapper object (class vtable in the layer's module, every call forwarded to an inner list, a device
// wrapper from GetDevice) and the device proof of the split. Runs the production hooks, patcher and split tracking on
// fake COM objects; no device, no GPU.
#include "../src/d3d12_hooks.cpp"
namespace wronggpu { const char* const kLogTag = "wrapped list test"; }

namespace wgwin {

int g_failures = 0;
#define EXPECT(cond)                                                              \
    do {                                                                          \
        if (!(cond)) {                                                            \
            ++g_failures;                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                         \
    } while (0)

// The runtime's list object: its own table 1480 bytes into the object.
struct RuntimeList {
    void** vtable;
    char state[1472];
    void* own[kMaxSlots];
};
void* g_classTable[kMaxSlots];  // static storage: inside this module's image, like a class's vtable
void* g_classOriginal[kMaxSlots];
int g_classTopology = 0, g_driverTopology = 0, g_classHeaps = 0, g_resets = 0, g_closes = 0;

CmdList* AsList(void* object) { return static_cast<CmdList*>(object); }
RuntimeList* AsRuntime(CmdList* list) { return reinterpret_cast<RuntimeList*>(list); }
bool DriverSlot(int slot) { return slot >= 20 && slot <= 45 && slot != WG_VT_LIST_EXECUTE_BUNDLE && slot != WG_VT_LIST_SET_DESCRIPTOR_HEAPS; }

HRESULT STDMETHODCALLTYPE FakeQuery(CmdList*, REFIID, void** out) {
    if (out) *out = nullptr;
    return E_NOINTERFACE;  // ID3D12GraphicsCommandList only: 33 hooked entries
}
ULONG STDMETHODCALLTYPE FakeAddRef(CmdList*) { return 2; }
ULONG STDMETHODCALLTYPE FakeRelease(CmdList*) { return 1; }
void STDMETHODCALLTYPE ClassOther(CmdList*) {}
void STDMETHODCALLTYPE DriverOther(CmdList*) {}
void STDMETHODCALLTYPE ClassTopology(CmdList*, D3D12_PRIMITIVE_TOPOLOGY) { ++g_classTopology; }
void STDMETHODCALLTYPE DriverTopology(CmdList*, D3D12_PRIMITIVE_TOPOLOGY) { ++g_driverTopology; }
void STDMETHODCALLTYPE ClassHeaps(CmdList*, UINT, ID3D12DescriptorHeap* const*) { ++g_classHeaps; }
HRESULT STDMETHODCALLTYPE ClassClose(CmdList* self) {
    ++g_closes;
    AsRuntime(self)->vtable = g_classTable;  // a closed list points at the class table
    return S_OK;
}
HRESULT STDMETHODCALLTYPE ClassReset(CmdList* self, ID3D12CommandAllocator*, ID3D12PipelineState*) {
    ++g_resets;
    RuntimeList* list = AsRuntime(self);
    for (int slot = 0; slot < kMaxSlots; ++slot)  // the runtime writes the driver's entry points again
        if (DriverSlot(slot))
            list->own[slot] = slot == WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY ? reinterpret_cast<void*>(&DriverTopology) : reinterpret_cast<void*>(&DriverOther);
    list->vtable = list->own;
    return S_OK;
}

// A list as the runtime makes it: its own table is a copy of the class table as that is at the moment.
std::unique_ptr<RuntimeList> NewList() {
    std::unique_ptr<RuntimeList> list(new RuntimeList());
    std::memcpy(list->own, g_classTable, sizeof g_classTable);
    list->vtable = g_classTable;
    // As the creation hook does: nothing is kept from an object that was at this address before.
    NoteCreated(1, D3D12_COMMAND_LIST_TYPE_DIRECT, nullptr, nullptr, 0, __uuidof(ID3D12GraphicsCommandList), list.get(), WG_RETURN_ADDRESS());
    return list;
}

template <class F>
F Method(RuntimeList& list, int slot) { return reinterpret_cast<F>(list.vtable[slot]); }
void GameReset(RuntimeList& list, void* allocator) {
    Method<PfnReset>(list, WG_VT_LIST_RESET)(AsList(&list), static_cast<ID3D12CommandAllocator*>(allocator), nullptr);
}
void GameTopology(RuntimeList& list) { Method<PfnTopology>(list, WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY)(AsList(&list), D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST); }
void GameHeaps(RuntimeList& list) {
    static ID3D12DescriptorHeap* const heaps[1] = {reinterpret_cast<ID3D12DescriptorHeap*>(&g_classHeaps)};
    Method<PfnHeaps>(list, WG_VT_LIST_SET_DESCRIPTOR_HEAPS)(AsList(&list), 1, heaps);
}
void GameClose(RuntimeList& list) { Method<PfnClose>(list, WG_VT_LIST_CLOSE)(AsList(&list)); }
bool Hooked(const RuntimeList& list, const std::vector<SlotHook>& hooks) {
    for (const SlotHook& h : hooks)
        if (list.own[h.slot] != h.hook) return false;
    return true;
}
WgResult Track(RuntimeList& list, bool* ownTable = nullptr) {
    bool own = false;
    const WgResult r = TrackList(AsList(&list), own);
    if (ownTable) *ownTable = own;
    return r;
}

// A layer's entry that forwards to whatever the class table holds (WrongGPU's hook): the loop.
void STDMETHODCALLTYPE LayerTopology(CmdList* self, D3D12_PRIMITIVE_TOPOLOGY t) {
    reinterpret_cast<PfnTopology>(g_classTable[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY])(self, t);
}

// The same hazard on the bare patcher (what 0.4.1 / 0.4.5-rc1 did to such a table).
VtblFamily g_loopFamily;
int g_loopDepth = 0;
struct LoopObject { void** vtable; };
using LoopMethod = int (*)(LoopObject*);
int LoopNative(LoopObject*) { return 37; }
int LoopHook(LoopObject* self) {
    if (++g_loopDepth > 8) return -1;  // the test's stand-in for the stack overflow
    return Orig<LoopMethod>(g_loopFamily, self, 1)(self);
}
int LoopLayer(LoopObject* self) { return LoopHook(self) + 1; }

// ---- case F: wrapper objects and the device proof (clean/fe-wrapped-list)
struct FakeDevice {
    void** vtable;
    int queues;
};
struct InnerList {  // a list of the device's own class
    void** vtable;
    FakeDevice* device;
    int topology, resets, closes;
};
struct WrapperList {  // another layer's object: its own class, every call forwarded to the inner list
    void** vtable;
    InnerList* inner;
    FakeDevice* device;
};
void* g_innerClass[kMaxSlots];
void* g_deviceClass[64];
InnerList* AsInner(CmdList* l) { return reinterpret_cast<InnerList*>(l); }
WrapperList* AsWrapper(CmdList* l) { return reinterpret_cast<WrapperList*>(l); }
CmdList* InnerOf(CmdList* l) { return reinterpret_cast<CmdList*>(AsWrapper(l)->inner); }
template <class F>
F InnerMethod(CmdList* wrapper, int slot) { return reinterpret_cast<F>(AsWrapper(wrapper)->inner->vtable[slot]); }

HRESULT STDMETHODCALLTYPE InnerGetDevice(CmdList* self, REFIID, void** out) { *out = AsInner(self)->device; return S_OK; }
D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE InnerGetType(CmdList*) { return D3D12_COMMAND_LIST_TYPE_DIRECT; }
HRESULT STDMETHODCALLTYPE InnerClose(CmdList* self) { ++AsInner(self)->closes; return S_OK; }
HRESULT STDMETHODCALLTYPE InnerReset(CmdList* self, ID3D12CommandAllocator*, ID3D12PipelineState*) { ++AsInner(self)->resets; return S_OK; }
void STDMETHODCALLTYPE InnerTopology(CmdList* self, D3D12_PRIMITIVE_TOPOLOGY) { ++AsInner(self)->topology; }

HRESULT STDMETHODCALLTYPE WrapGetDevice(CmdList* self, REFIID, void** out) { *out = AsWrapper(self)->device; return S_OK; }
D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE WrapGetType(CmdList* self) {
    using Pfn = D3D12_COMMAND_LIST_TYPE(STDMETHODCALLTYPE*)(CmdList*);
    return InnerMethod<Pfn>(self, 8)(InnerOf(self));
}
HRESULT STDMETHODCALLTYPE WrapClose(CmdList* self) { return InnerMethod<PfnClose>(self, WG_VT_LIST_CLOSE)(InnerOf(self)); }
HRESULT STDMETHODCALLTYPE WrapReset(CmdList* self, ID3D12CommandAllocator* a, ID3D12PipelineState* p) {
    return InnerMethod<PfnReset>(self, WG_VT_LIST_RESET)(InnerOf(self), a, p);
}
void STDMETHODCALLTYPE WrapTopology(CmdList* self, D3D12_PRIMITIVE_TOPOLOGY t) {
    InnerMethod<PfnTopology>(self, WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY)(InnerOf(self), t);
}

HRESULT STDMETHODCALLTYPE DeviceQuery(FakeDevice*, REFIID, void** out) { if (out) *out = nullptr; return E_NOINTERFACE; }
ULONG STDMETHODCALLTYPE DeviceAddRef(FakeDevice*) { return 2; }
ULONG STDMETHODCALLTYPE DeviceRelease(FakeDevice*) { return 1; }
void STDMETHODCALLTYPE DeviceOther(FakeDevice*) {}
HRESULT STDMETHODCALLTYPE DeviceCreateQueue(FakeDevice* self, const void*, REFIID, void** out) {
    ++self->queues;  // never succeeds here: the split stops at the first queue it tries to make
    if (out) *out = nullptr;
    return E_FAIL;
}
void FillDeviceTable(void** t) {
    for (int i = 0; i < 64; ++i) t[i] = reinterpret_cast<void*>(&DeviceOther);
    t[0] = reinterpret_cast<void*>(&DeviceQuery);
    t[1] = reinterpret_cast<void*>(&DeviceAddRef);
    t[2] = reinterpret_cast<void*>(&DeviceRelease);
    t[8] = reinterpret_cast<void*>(&DeviceCreateQueue);  // ID3D12Device::CreateCommandQueue
}

void RunWrapper(void* allocator) {
    HMODULE layer = LoadLibraryW(L"test_wrapper_layer.dll");  // built next to this test
    EXPECT(layer != nullptr);
    if (!layer) return;
    using TableFn = void** (*)();
    void** wrapperTable = reinterpret_cast<TableFn>(reinterpret_cast<void*>(GetProcAddress(layer, "WgTestWrapperListTable")))();
    void** wrapperDeviceTable = reinterpret_cast<TableFn>(reinterpret_cast<void*>(GetProcAddress(layer, "WgTestWrapperDeviceTable")))();
    for (int i = 0; i < kMaxSlots; ++i) {
        g_innerClass[i] = reinterpret_cast<void*>(&ClassOther);
        wrapperTable[i] = reinterpret_cast<void*>(&ClassOther);
    }
    for (void** t : {static_cast<void**>(g_innerClass), wrapperTable}) {
        t[0] = reinterpret_cast<void*>(&FakeQuery);
        t[1] = reinterpret_cast<void*>(&FakeAddRef);
        t[2] = reinterpret_cast<void*>(&FakeRelease);
    }
    g_innerClass[7] = reinterpret_cast<void*>(&InnerGetDevice);  // ID3D12DeviceChild::GetDevice
    g_innerClass[8] = reinterpret_cast<void*>(&InnerGetType);    // ID3D12CommandList::GetType
    g_innerClass[WG_VT_LIST_CLOSE] = reinterpret_cast<void*>(&InnerClose);
    g_innerClass[WG_VT_LIST_RESET] = reinterpret_cast<void*>(&InnerReset);
    g_innerClass[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] = reinterpret_cast<void*>(&InnerTopology);
    wrapperTable[7] = reinterpret_cast<void*>(&WrapGetDevice);
    wrapperTable[8] = reinterpret_cast<void*>(&WrapGetType);
    wrapperTable[WG_VT_LIST_CLOSE] = reinterpret_cast<void*>(&WrapClose);
    wrapperTable[WG_VT_LIST_RESET] = reinterpret_cast<void*>(&WrapReset);
    wrapperTable[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] = reinterpret_cast<void*>(&WrapTopology);
    FillDeviceTable(g_deviceClass);
    FillDeviceTable(wrapperDeviceTable);

    FakeDevice real{g_deviceClass, 0};            // the D3D12 runtime's device (this test's module stands for the runtime)
    FakeDevice other{g_deviceClass, 0};           // a runtime device no upscaler was created on
    FakeDevice wrapDevice{wrapperDeviceTable, 0};  // the layer's device wrapper
    InnerList inner{g_innerClass, &real, 0, 0, 0};
    WrapperList wrap{wrapperTable, &inner, &wrapDevice};

    // The upscaler's context was created on the real device; one made on the wrapper device proves nothing.
    EXPECT(Install(&real, nullptr, "test: the context's device"));
    EXPECT(ProvenDevice(&real) && !ProvenDevice(&other) && !ProvenDevice(&wrapDevice));
    EXPECT(Install(&wrapDevice, nullptr, "test: a context made on the wrapper device") && !ProvenDevice(&wrapDevice));

    // The inner list's class (the device's own lists' module) is patched; the wrapper's class table never is.
    EXPECT(!WrapperTable(g_innerClass) && WrapperTable(wrapperTable));
    EXPECT(PatchList(AsList(&inner)) && IsPatched(g_listVt, g_innerClass));
    void* snapshot[kMaxSlots];
    std::memcpy(snapshot, wrapperTable, sizeof snapshot);
    const int tables = g_listVt.count.load();
    EXPECT(!PatchList(AsList(&wrap)) && !IsPatched(g_listVt, wrapperTable));
    EXPECT(Install(&real, &wrap, "test: a dispatch list of a class not seen yet"));
    EXPECT(std::memcmp(snapshot, wrapperTable, sizeof snapshot) == 0 && g_listVt.count.load() == tables);

    // The game records through the wrapper: each call reaches WrongGPU's hooks once, on the inner list (the one tracked).
    reinterpret_cast<PfnReset>(wrapperTable[WG_VT_LIST_RESET])(AsList(&wrap), static_cast<ID3D12CommandAllocator*>(allocator), nullptr);
    reinterpret_cast<PfnTopology>(wrapperTable[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY])(AsList(&wrap), D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ListState* tracked = FindState(&inner);
    EXPECT(tracked && tracked->known && tracked->allocator == allocator && tracked->b.topologySet);
    EXPECT(inner.resets == 1 && inner.topology == 1);
    EXPECT(FindState(&wrap) == nullptr);  // no hook ran on the wrapper itself: one logical list, tracked once

    // A split of the wrapper: refused before anything is read or written; no queue on any device. Before a split was
    // asked the front end keeps the context (a backend that never splits, the D3D12 executor, is not affected).
    WgFrame frame{};
    frame.command_list = &wrap;
    WgSplitRequest request{};
    request.struct_size = sizeof request;
    request.done_fence = &g_resets;
    request.resume_fence = &g_closes;
    EXPECT(BlockedImpl(&wrap) == nullptr);
    EXPECT(SplitImpl(&frame, &request) == WG_ERROR_NOT_READY);
    EXPECT(std::memcmp(snapshot, wrapperTable, sizeof snapshot) == 0 && FindState(&wrap) == nullptr);
    EXPECT(inner.closes == 0 && inner.resets == 1 && g_lanes.empty() && wrapDevice.queues == 0 && real.queues == 0);
    EXPECT(BlockedImpl(&wrap) == kWrapperReason && BlockedImpl(&inner) == nullptr);  // the front end's safe path from now on

    // A list of the device's own class whose GetDevice gives the wrapper device, or a device no upscaler was made on:
    // refused before any lane or queue is made.
    InnerList strays[2] = {{g_innerClass, &wrapDevice, 0, 0, 0}, {g_innerClass, &other, 0, 0, 0}};
    for (InnerList& stray : strays) {
        reinterpret_cast<PfnReset>(g_innerClass[WG_VT_LIST_RESET])(AsList(&stray), static_cast<ID3D12CommandAllocator*>(allocator), nullptr);
        frame.command_list = &stray;
        EXPECT(BlockedImpl(&stray) == nullptr);
        EXPECT(SplitImpl(&frame, &request) == WG_ERROR_NOT_READY);
        EXPECT(g_lanes.empty() && stray.device->queues == 0 && stray.closes == 0 && stray.resets == 1);
        EXPECT(BlockedImpl(&stray) == kNotOurDeviceReason);
    }
    // Control: the same list on the proven device gets as far as making WrongGPU's queue on that device (the fake fails it).
    InnerList good{g_innerClass, &real, 0, 0, 0};
    reinterpret_cast<PfnReset>(g_innerClass[WG_VT_LIST_RESET])(AsList(&good), static_cast<ID3D12CommandAllocator*>(allocator), nullptr);
    frame.command_list = &good;
    EXPECT(SplitImpl(&frame, &request) == WG_ERROR_DEVICE && real.queues == 1 && wrapDevice.queues == 0 && other.queues == 0);
    EXPECT(BlockedImpl(&good) == nullptr && good.closes == 0);
}

void RunRemovedDevice();  // bug hunt F3 (below)
void Run() {
    g_testRuntimeModule = GetModuleHandleW(nullptr);  // this test's fake runtime objects live in its own module
    // The rule for tables WrongGPU cannot take over.
    EXPECT(OwnedByAnotherLayer(TableFacts{false, 28, 13}) && OwnedByAnotherLayer(TableFacts{false, 28, 0}));
    EXPECT(!OwnedByAnotherLayer(TableFacts{true, 28, 0}) && !OwnedByAnotherLayer(TableFacts{false, 0, 41}) && !OwnedByAnotherLayer(TableFacts{true, 0, 0}));

    // The class table.
    for (void*& entry : g_classTable) entry = reinterpret_cast<void*>(&ClassOther);
    g_classTable[0] = reinterpret_cast<void*>(&FakeQuery);
    g_classTable[1] = reinterpret_cast<void*>(&FakeAddRef);
    g_classTable[2] = reinterpret_cast<void*>(&FakeRelease);
    g_classTable[WG_VT_LIST_CLOSE] = reinterpret_cast<void*>(&ClassClose);
    g_classTable[WG_VT_LIST_RESET] = reinterpret_cast<void*>(&ClassReset);
    g_classTable[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] = reinterpret_cast<void*>(&ClassTopology);
    g_classTable[WG_VT_LIST_SET_DESCRIPTOR_HEAPS] = reinterpret_cast<void*>(&ClassHeaps);
    std::memcpy(g_classOriginal, g_classTable, sizeof g_classTable);

    // Lists the game made before WrongGPU's hooks: A is closed, B is being recorded.
    auto a = NewList(), b = NewList();
    ClassReset(AsList(b.get()), nullptr, nullptr);
    g_resets = 0;

    // The hooks are installed: the class table is patched.
    auto probe = NewList();
    const std::vector<SlotHook> hooks = ListHooks(AsList(probe.get()));
    EXPECT(hooks.size() == 33);
    EXPECT(PatchList(AsList(probe.get()), true) && IsPatched(g_listVt, g_classTable) && g_listVt.count.load() == 1);
    g_installed.store(true);
    void* allocator = &g_resets;

    // A: the game's Reset comes through the class table; the runtime switches the list to its own table with the
    // driver's entries, and WrongGPU takes that table over before the game's next call.
    GameReset(*a, allocator);
    EXPECT(g_resets == 1 && a->vtable == a->own && Hooked(*a, hooks));
    ListState& sa = StateOf(a.get());
    EXPECT(sa.known && sa.allocator == allocator && sa.own && sa.own->table == a->own && sa.own->tracked && !sa.watch);
    EXPECT(sa.own->orig[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] == reinterpret_cast<void*>(&DriverTopology));
    EXPECT(sa.own->orig[WG_VT_LIST_SET_DESCRIPTOR_HEAPS] == reinterpret_cast<void*>(&ClassHeaps));
    EXPECT(g_listVt.count.load() == 1);  // no table is added to the class tables: nothing grows with the lists
    GameTopology(*a);
    GameHeaps(*a);
    EXPECT(g_driverTopology == 1 && g_classTopology == 0 && g_classHeaps == 1);  // each call reaches the entry it had, once
    EXPECT(sa.b.topologySet && sa.b.heapsSet);                                   // and WrongGPU saw it
    EXPECT(BlockedImpl(a.get()) == nullptr);
    bool own = false;
    EXPECT(Track(*a, &own) == WG_OK && own);
    // The split: Close and Reset through the originals; the runtime writes its table again; the hooks are back before
    // the bindings are restored.
    EXPECT(ListOrig<PfnClose>(a.get(), WG_VT_LIST_CLOSE)(AsList(a.get())) == S_OK && a->vtable == g_classTable && g_closes == 1);
    void** before = a->vtable;
    EXPECT(ListOrig<PfnReset>(a.get(), WG_VT_LIST_RESET)(AsList(a.get()), nullptr, nullptr) == S_OK && g_resets == 2 && !Hooked(*a, hooks));
    AfterReset(AsList(a.get()), sa, before);
    EXPECT(Hooked(*a, hooks) && sa.own->tracked);
    EXPECT(Restore(AsList(a.get()), sa.b) == 2 && g_driverTopology == 2 && g_classHeaps == 2);
    // The next frames: the same list again.
    for (int frame = 0; frame < 3; ++frame) {
        GameClose(*a);
        GameReset(*a, allocator);
        GameTopology(*a);
        EXPECT(Hooked(*a, hooks) && Track(*a, &own) == WG_OK && own);
    }
    EXPECT(g_driverTopology == 5 && g_classTopology == 0);
    // The runtime rewrites the table while the list records: not split, taken over again, tracked from the next Reset.
    a->own[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] = reinterpret_cast<void*>(&DriverTopology);
    EXPECT(Track(*a, &own) == WG_ERROR_NOT_READY && !own && Hooked(*a, hooks) && !sa.own->tracked && !sa.known);
    EXPECT(g_ownStrikes.load() == 1 && BlockedImpl(a.get()) == nullptr);
    GameClose(*a);
    GameReset(*a, allocator);
    EXPECT(Track(*a, &own) == WG_OK && own);

    // B: its Reset came before the hooks. The first dispatch takes its table over (no split that frame); its calls
    // are seen from then on, and from its next Reset it is tracked.
    ListState& sb = StateOf(b.get());
    EXPECT(BlockedImpl(b.get()) == nullptr && !sb.own);
    EXPECT(Track(*b, &own) == WG_ERROR_NOT_READY && Hooked(*b, hooks) && sb.own && !sb.own->tracked && !sb.known);
    GameTopology(*b);
    EXPECT(g_driverTopology == 6 && sb.b.topologySet);
    EXPECT(Track(*b, &own) == WG_ERROR_NOT_READY);
    GameClose(*b);
    GameReset(*b, allocator);
    EXPECT(Track(*b, &own) == WG_OK && own && sb.known);

    // C: a list made after the hooks. Its own table starts as a copy of the patched class table: the entries the
    // runtime leaves alone hold WrongGPU's hooks, and their originals are the class's.
    auto c = NewList();
    EXPECT(c->own[WG_VT_LIST_SET_DESCRIPTOR_HEAPS] != reinterpret_cast<void*>(&ClassHeaps));
    GameReset(*c, allocator);
    ListState& sc = StateOf(c.get());
    EXPECT(Hooked(*c, hooks) && sc.own && sc.own->tracked);
    EXPECT(sc.own->orig[WG_VT_LIST_SET_DESCRIPTOR_HEAPS] == reinterpret_cast<void*>(&ClassHeaps));
    const int heaps = g_classHeaps;
    GameHeaps(*c);
    EXPECT(g_classHeaps == heaps + 1 && Track(*c, &own) == WG_OK && own);

    // D: a table with entries that point into no module (another party's generated code): left untouched, never
    // split, and a lasting reason for the front end (AMD's FSR takes the context).
    void* thunks = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (int frame = 0; frame < 3; ++frame) {
        auto d = NewList();
        std::memcpy(d->own, g_classOriginal, sizeof g_classOriginal);
        for (int slot = 0; slot < kMaxSlots; ++slot)
            if (DriverSlot(slot)) d->own[slot] = thunks;
        d->vtable = d->own;
        void* snapshot[kMaxSlots];
        std::memcpy(snapshot, d->own, sizeof snapshot);
        EXPECT(Track(*d, &own) == WG_ERROR_NOT_READY && !own);
        EXPECT(std::memcmp(snapshot, d->own, sizeof snapshot) == 0);
        const char* why = BlockedImpl(d.get());
        EXPECT(why && std::strstr(why, "another hook layer"));
    }
    EXPECT(g_listVt.count.load() == 1);

    // F: another layer's wrapper object, and the device proof of the split (above).
    RunWrapper(allocator);
    RunRemovedDevice();

    // E: an entry that forwards to the class table's entry (WrongGPU's hook). Taken over, the first call would never
    // return; the stack guard gives every entry back, ends the chain in the class's method, and such lists are left
    // alone from then on (a lasting reason for the front end).
    auto e = NewList();
    std::memcpy(e->own, g_classOriginal, sizeof g_classOriginal);
    e->own[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] = reinterpret_cast<void*>(&LayerTopology);
    e->vtable = e->own;
    EXPECT(Track(*e, &own) == WG_ERROR_NOT_READY && Hooked(*e, hooks));
    const int topology = g_classTopology;
    GameTopology(*e);
    EXPECT(g_ownLoop.load() && g_classTopology == topology + 1);
    EXPECT(e->own[WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY] == reinterpret_cast<void*>(&LayerTopology) && !Hooked(*e, hooks));
    GameTopology(*e);  // layer -> class hook -> class method: no loop without WrongGPU's hook in the list's table
    EXPECT(g_classTopology == topology + 2);
    EXPECT(BlockedImpl(e.get()) == kOwnLoopReason && Track(*e, &own) == WG_ERROR_NOT_READY && !Hooked(*e, hooks));
    auto f = NewList();
    GameReset(*f, allocator);
    EXPECT(!Hooked(*f, hooks) && BlockedImpl(f.get()) == kOwnLoopReason);  // nothing is taken over any more

    // The bare patcher on such a table: hook -> layer -> hook -> ...
    static void* loopClass[kMaxSlots];
    loopClass[0] = reinterpret_cast<void*>(&FakeQuery);
    loopClass[1] = reinterpret_cast<void*>(&LoopNative);
    const std::vector<SlotHook> loopHooks{{1, reinterpret_cast<void*>(&LoopHook)}};
    EXPECT(Patch(g_loopFamily, loopClass, loopHooks, "loop class"));
    void* layerTable[kMaxSlots] = {};
    layerTable[0] = loopClass[0];
    layerTable[1] = reinterpret_cast<void*>(&LoopLayer);
    LoopObject layered{layerTable};
    g_loopDepth = 0;
    EXPECT(reinterpret_cast<LoopMethod>(layerTable[1])(&layered) == 38 && g_loopDepth == 1);
    EXPECT(Patch(g_loopFamily, layerTable, loopHooks, "loop layer table"));
    g_loopDepth = 0;
    reinterpret_cast<LoopMethod>(layerTable[1])(&layered);
    EXPECT(g_loopDepth > 8);
}


// Bug hunt F3: the idle lanes of a removed device are released (its queue and fence: the device is no longer kept) and
// the device leaves the known-device list, whose freed slot is reused; lanes of live devices, and lanes with splits
// pending, stay.
struct FakeCom {
    void** vtable;
    int releases = 0;
    HRESULT removed = S_OK;
};
ULONG STDMETHODCALLTYPE ComRelease(FakeCom* self) { return ULONG(++self->releases); }
HRESULT STDMETHODCALLTYPE ComRemoved(FakeCom* self) { return self->removed; }
void RunRemovedDevice() {
    static void* table[64];
    for (void*& e : table) e = reinterpret_cast<void*>(&DeviceOther);
    table[2] = reinterpret_cast<void*>(&ComRelease);
    table[37] = reinterpret_cast<void*>(&ComRemoved);  // ID3D12Device::GetDeviceRemovedReason
    FakeCom lost{table}, alive{table}, current{table}, queues[4] = {{table}, {table}, {table}, {table}}, fences[4] = {{table}, {table}, {table}, {table}};
    lost.removed = DXGI_ERROR_DEVICE_REMOVED;
    auto lane = [&](FakeCom& device, int k, int pending) {
        auto* l = new Lane();
        l->id = ++g_laneIds;
        l->device = reinterpret_cast<ID3D12Device*>(&device);
        l->queue = reinterpret_cast<ID3D12CommandQueue*>(&queues[k]);
        l->order = reinterpret_cast<ID3D12Fence*>(&fences[k]);
        l->pending = pending;
        return l;
    };
    std::lock_guard<std::mutex> patch(g_patchMutex);
    const int before = g_contextDeviceCount.load();
    for (int i = before; i < kMaxKnown - 1; ++i) Remember(g_contextDevices, g_contextDeviceCount, reinterpret_cast<void*>(uintptr_t(0x1000 + 16 * i)));
    Remember(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&lost));  // the list is full now
    EXPECT(g_contextDeviceCount.load() == kMaxKnown && Known(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&lost)));
    Remember(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&current));
    EXPECT(!Known(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&current)));  // full: the old behaviour
    {
        std::lock_guard<std::mutex> split(g_splitMutex);
        const size_t lanes = g_lanes.size();
        g_lanes.push_back(lane(lost, 0, 0));
        g_lanes.push_back(lane(lost, 1, 0));
        g_lanes.push_back(lane(lost, 2, 1));  // a split pending: stays
        g_lanes.push_back(lane(alive, 3, 0));
        DropRemovedLanes(reinterpret_cast<ID3D12Device*>(&current));
        EXPECT(g_lanes.size() == lanes + 2);
        EXPECT(queues[0].releases == 1 && fences[0].releases == 1 && queues[1].releases == 1 && fences[1].releases == 1);
        EXPECT(queues[2].releases == 0 && queues[3].releases == 0 && fences[3].releases == 0);
        while (g_lanes.size() > lanes) {
            delete g_lanes.back();
            g_lanes.pop_back();
        }
    }
    EXPECT(!Known(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&lost)));
    Remember(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&current));  // the freed slot
    EXPECT(Known(g_contextDevices, g_contextDeviceCount, static_cast<void*>(&current)));
    for (int i = 0; i < kMaxKnown; ++i) {  // leave the list as it was
        void* v = g_contextDevices[i].load();
        if (i >= before) g_contextDevices[i].store(nullptr);
        (void)v;
    }
    g_contextDeviceCount.store(before);
}

}  // namespace wgwin

int main() {
    wgwin::Run();
    if (wgwin::g_failures) return 1;
    std::puts("PASS: lists that record into their own vtable are tracked through it; other tables are left alone; wrapper objects "
              "are hooked once through the list they wrap and never split; no split or queue off the proven device");
    return 0;
}
