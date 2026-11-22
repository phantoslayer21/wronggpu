// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): CPU tests of the Cyberpunk 2077 title binding (interface build 2.14.1)
// and the 2.14 native codec, through the Win32 transport.
//
//  - the binding data (generated from bindings/cp2077-2.14.1.json): validity, structure versions and sizes, buffer
//    types, UI range, the factory import, selection by exe name;
//  - the codec on its own: GUID text, the chain walk (any order, unknown structures skipped, bounded), the options
//    checks and decoding read from raw bytes at the layout's offsets, the state written per version, tags and
//    constants;
//  - a whole game session through the stand-in interface (fg_mock_interposer.dll, imported by fg_mock_game.dll): the
//    binding's own structure GUIDs, buffer types and FG feature-function names with the stand-in's export names. Support
//    query and function lookups for 1000 answered; super resolution, ray reconstruction, the latency features, NIS and
//    unknown ids forwarded with the game's own arguments; the factory created through the interface export with the
//    swap-chain hook put on it; the chain claimed; tags 0 / 1 / 2 and the constants assembled; every game present
//    handed to the core (the backbuffer as colour input) and an executor before the presenter; restore;
//  - fg=off (nothing patched, the factory untouched) and FG not taken (pass-through wrapper, forwarded calls).
// Every value is synthetic. Built with CI's flags (/W4 /WX /permissive-, static runtime).
// Written for this project.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../win32/fg_asi_transport.h"
#include "../win32/fg_sl214_codec.h"
#include "../win32/fg_swapchain_hook.h"
#include "../win32/fg_title_bindings.h"
#include "fg_mock_interposer.h"

using namespace wgfgp;
using namespace wgfgp::win32;

namespace {

int g_checks = 0, g_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failed;                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

constexpr uint32_t kFeatFg = WG_FEAT_FRAME_GENERATION;
constexpr uint32_t kFeatSr = WG_FEAT_SUPER_RESOLUTION;
constexpr uint32_t kFeatNis = WG_FEAT_IMAGE_SCALING;
constexpr uint32_t kFeatRr = WG_FEAT_RAY_RECONSTRUCTION;
constexpr uint32_t kFeatLatency = WG_FEAT_LATENCY;
constexpr uint32_t kFeatPcLatency = WG_FEAT_PC_LATENCY;
constexpr uint32_t kAmdPciVendor = 4098;
const char* const kLeaf = "fg_mock_interposer.dll";
const char* const kMockRoleNames[kFgRoleCount] = {"fgmock_is_supported", "fgmock_requirements", "fgmock_get_function",
                                                  "fgmock_tag",          "fgmock_tag_for_frame", "fgmock_constants",
                                                  "fgmock_evaluate",     "fgmock_free"};
const char* const kMockFactory = "fgmock_create_factory2";

std::vector<std::string> g_log;
void CollectLog(void*, const char* line) { g_log.push_back(line ? line : ""); }

std::wstring ExeDir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
    std::wstring w(buf, n);
    return w.substr(0, w.find_last_of(L"\\/"));
}

// ---- raw native objects (bytes at the layout's offsets) -------------------------------------------------------------

template <typename T>
void Put(void* base, size_t off, T v) {
    std::memcpy(static_cast<uint8_t*>(base) + off, &v, sizeof v);
}
template <typename T>
T Get(const void* base, size_t off) {
    T v;
    std::memcpy(&v, static_cast<const uint8_t*>(base) + off, sizeof v);
    return v;
}

struct alignas(8) Raw120 {
    uint8_t b[120];
};
struct alignas(8) Raw88 {
    uint8_t b[88];
};
struct alignas(8) RawHeader {  // an unknown structure in a chain
    uint8_t b[48];
};

void SetHeader(void* p, const WgGuid& type, uint64_t version, const void* next) {
    Put<const void*>(p, 0, next);
    std::memcpy(static_cast<uint8_t*>(p) + 8, type.b, 16);
    Put<uint64_t>(p, 24, version);
}

WgGuid SyntheticGuid(uint8_t seed) {
    WgGuid g = {};
    for (int i = 0; i < 16; ++i) g.b[i] = static_cast<uint8_t>(seed * 17 + i * 3 + 1);
    return g;
}

// options v5 as both titles build them (defaults: tri-state +104 = 2, flags 8), then mode and count
void MakeOptions(Raw120& o, const FgSl214Layout& l, uint32_t mode, uint32_t frames, const void* next) {
    std::memset(&o, 0, sizeof o);
    SetHeader(&o, l.options_guid, l.options_version, next);
    Put<uint32_t>(&o, 32, mode);
    Put<uint32_t>(&o, 36, frames);
    Put<uint32_t>(&o, 40, 8u);
    Put<uint8_t>(&o, 104, 2);
}

void MakeState(Raw88& s, const FgSl214Layout& l, uint64_t version) {
    std::memset(&s, 0xcd, sizeof s);  // sentinel: what the codec must not touch stays 0xcd
    SetHeader(&s, l.state_guid, version, nullptr);
}

WgViewport MakeViewport(const FgSl214Layout& l, uint32_t id) {
    WgViewport v = {};
    v.h.type = l.viewport_guid;
    v.h.version = 1;
    v.id = id;
    return v;
}

WgResource MakeResource(const FgSl214Layout& l, void* native, uint32_t w, uint32_t h, uint32_t fmt) {
    WgResource r = {};
    r.h.type = l.resource_guid;
    r.h.version = 1;
    r.kind = WG_RES_TEX2D;
    r.native = native;
    r.state = 0x80;
    r.width = w;
    r.height = h;
    r.native_format = fmt;
    r.mip_levels = 1;
    r.array_layers = 1;
    return r;
}

WgTag MakeTag(const FgSl214Layout& l, uint32_t type, WgResource* res, uint32_t lifetime) {
    WgTag t = {};
    t.h.type = l.tag_guid;
    t.h.version = 1;
    t.resource = res;
    t.buffer_type = type;
    t.lifetime = lifetime;
    return t;
}

WgConstants MakeConstants(const FgSl214Layout& l, float seed, uint8_t reset) {
    WgConstants c = {};
    c.h.type = l.constants_guid;
    c.h.version = 2;
    for (int i = 0; i < 16; ++i) c.view_to_clip[i] = seed + 0.125f * static_cast<float>(i);
    c.jitter[0] = 0.25f * seed;
    c.jitter[1] = -0.375f;
    c.mv_scale[0] = 0.5f;
    c.mv_scale[1] = -0.5f;
    c.depth_inverted = WG_TRI_TRUE;
    c.camera_motion_included = WG_TRI_TRUE;
    c.reset = reset;
    return c;
}

// ---- fake DXGI objects ---------------------------------------------------------------------------------------------

class FakeBuffer final : public IUnknown {
public:
    long refs = 1;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** ppv) override {
        if (ppv) *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(++refs); }
    ULONG STDMETHODCALLTYPE Release() override { return static_cast<ULONG>(--refs); }
};

class FakeSwapChain final : public IDXGISwapChain4 {
public:
    long refs = 1;
    std::vector<HRESULT> results;  // consumed by Present / Present1 in order; empty: S_OK
    uint32_t presents = 0, tests = 0, getbuffers = 0;
    UINT backbuffer_index = 0;
    FakeBuffer buffers[3];
    HWND hwnd = reinterpret_cast<HWND>(static_cast<uintptr_t>(0x5a));

    HRESULT Next() {
        if (results.empty()) return S_OK;
        const HRESULT hr = results.front();
        results.erase(results.begin());
        return hr;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
            riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) || riid == __uuidof(IDXGISwapChain2) ||
            riid == __uuidof(IDXGISwapChain3) || riid == __uuidof(IDXGISwapChain4)) {
            *ppv = static_cast<IDXGISwapChain4*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(++refs); }
    ULONG STDMETHODCALLTYPE Release() override { return static_cast<ULONG>(--refs); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE Present(UINT, UINT flags) override {
        if (flags & DXGI_PRESENT_TEST) {
            ++tests;
            return S_OK;
        }
        ++presents;
        return Next();
    }
    HRESULT STDMETHODCALLTYPE GetBuffer(UINT index, REFIID, void** s) override {
        ++getbuffers;
        if (!s) return E_POINTER;
        if (index >= 3) {
            *s = nullptr;
            return DXGI_ERROR_INVALID_CALL;
        }
        buffers[index].AddRef();
        *s = static_cast<IUnknown*>(&buffers[index]);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL, IDXGIOutput*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* f, IDXGIOutput** t) override {
        if (f) *f = FALSE;
        if (t) *t = nullptr;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT, UINT, UINT, DXGI_FORMAT, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput**) override { return DXGI_ERROR_UNSUPPORTED; }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS*) override { return DXGI_ERROR_FRAME_STATISTICS_DISJOINT; }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* c) override {
        if (c) *c = presents;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* d) override {
        if (!d) return E_POINTER;
        *d = DXGI_SWAP_CHAIN_DESC1{};
        d->Width = 96;
        d->Height = 48;
        d->Format = DXGI_FORMAT_R10G10B10A2_UNORM;
        d->BufferCount = 3;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* h) override {
        if (h) *h = hwnd;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID, void**) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE Present1(UINT, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
        if (flags & DXGI_PRESENT_TEST) {
            ++tests;
            return S_OK;
        }
        ++presents;
        return Next();
    }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** o) override {
        if (o) *o = nullptr;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT*, UINT*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT*) override { return S_OK; }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override { return nullptr; }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F*) override { return S_OK; }
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override { return backbuffer_index; }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE, UINT* s) override {
        if (s) *s = 1;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT*, IUnknown* const*) override {
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE, UINT, void*) override { return S_OK; }
};

class FakeFactory final : public IDXGIFactory2 {
public:
    std::vector<FakeSwapChain*> made;
    long refs = 1;
    ~FakeFactory() {
        for (FakeSwapChain* s : made) delete s;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIFactory) ||
            riid == __uuidof(IDXGIFactory1) || riid == __uuidof(IDXGIFactory2)) {
            *ppv = static_cast<IDXGIFactory2*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(++refs); }
    ULONG STDMETHODCALLTYPE Release() override { return static_cast<ULONG>(--refs); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT, IDXGIAdapter**) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE, IDXGIAdapter**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT, IDXGIAdapter1**) override { return DXGI_ERROR_NOT_FOUND; }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return TRUE; }
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*,
                                                     IDXGISwapChain1** out) override {
        FakeSwapChain* s = new FakeSwapChain();
        made.push_back(s);
        *out = s;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(IUnknown*, IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                           IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE, LUID*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND, UINT, DWORD*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE, DWORD*) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND, UINT, DWORD*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE, DWORD*) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                            IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }
};

// Calls through the method table, as a game does (a volatile pointer: no devirtualised call).
HRESULT CreateThroughTable(IDXGIFactory2* f, IDXGISwapChain1** out) {
    IDXGIFactory2* volatile vf = f;
    DXGI_SWAP_CHAIN_DESC1 desc = {};
    return vf->CreateSwapChainForHwnd(nullptr, reinterpret_cast<HWND>(static_cast<uintptr_t>(0x5a)), &desc, nullptr,
                                      nullptr, out);
}

// The presenter behind the bridge: records when the core saw the interval, presents the game image (plus `generated`).
class TestPresenter final : public IFgPresentCallback {
public:
    FgAsiTransport* transport = nullptr;
    uint32_t generated = 0;
    uint32_t begins = 0, presents = 0, retries = 0, detached = 0;
    uint64_t submits_at_begin = 0, submits_at_present = 0;
    void BeginInterval(const FgPresentCall&) override {
        ++begins;
        submits_at_begin = transport ? transport->present_stats().submits : 0;
    }
    HRESULT PresentInterval(const FgPresentCall& call, IFgInnerPresenter& inner) override {
        ++presents;
        if (call.retry) ++retries;
        submits_at_present = transport ? transport->present_stats().submits : 0;
        if (!call.retry)
            for (uint32_t i = 0; i < generated; ++i) inner.Present(0, 0);
        return inner.Present(call.sync_interval, call.flags);
    }
    void Detached() override { ++detached; }
};

// The executor: keeps every packet it accepted.
struct Executor {
    uint32_t submits = 0;
    WgFgPacket last = {};
    WgFgSessionDesc session = {};
};
int32_t ExecutorSubmit(void* self, const WgFgPacket* p, const WgFgSessionDesc* s) {
    Executor* e = static_cast<Executor*>(self);
    ++e->submits;
    e->last = *p;
    e->session = *s;
    return 0;
}

// ---- the stand-in game ---------------------------------------------------------------------------------------------

struct GameModule {
    HMODULE h = nullptr;
    WgResultCode (*is_supported)(uint32_t, const WgAdapterInfo*) = nullptr;
    WgResultCode (*requirements)(uint32_t, WgRequirements*) = nullptr;
    WgResultCode (*get_function)(uint32_t, const char*, void**) = nullptr;
    WgResultCode (*tag)(const WgViewport*, const WgTag*, uint32_t, void*) = nullptr;
    WgResultCode (*constants)(const WgConstants*, const WgFrameToken*, const WgViewport*) = nullptr;
    WgResultCode (*evaluate)(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*) = nullptr;
    HRESULT (*create_factory2)(UINT, REFIID, void**) = nullptr;
    void (*slot_values)(void**) = nullptr;

    template <typename F>
    void bind(F& f, const char* name) {
        f = reinterpret_cast<F>(GetProcAddress(h, name));
    }
    bool load(const wchar_t* leaf) {
        h = LoadLibraryW((ExeDir() + L"\\" + leaf).c_str());
        if (!h) return false;
        bind(is_supported, "game_is_supported");
        bind(requirements, "game_requirements");
        bind(get_function, "game_get_function");
        bind(tag, "game_tag");
        bind(constants, "game_constants");
        bind(evaluate, "game_evaluate");
        bind(create_factory2, "game_create_factory2");
        bind(slot_values, "game_slot_values");
        return is_supported && requirements && get_function && tag && constants && evaluate && create_factory2 &&
               slot_values;
    }
    std::vector<void*> slots() const {
        std::vector<void*> v(FG_MOCK_GAME_SLOTS, nullptr);
        slot_values(v.data());
        return v;
    }
};

void* GenuineExport(const char* name) { return reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA(kLeaf), name)); }

bool SlotsAreGenuine(const GameModule& g) {
    const auto v = g.slots();
    for (uint32_t i = 0; i < kFgRoleCount; ++i)
        if (v[i] != GenuineExport(kMockRoleNames[i])) return false;
    return v[9] == GenuineExport(kMockFactory);
}

void SetMockResults() {
    FgMockRecord* r = fgmock_record();
    for (uint32_t i = 0; i < kMockFnCount; ++i) r->result[i] = static_cast<int32_t>(40 + i);  // distinct, synthetic
}

void MakeTaken(FgProxy* core) {
    WgFgAdapterIdentity id = {};
    id.struct_size = sizeof(id);
    id.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    id.vendor_id = kAmdPciVendor;
    id.device_id = 1;
    id.luid = {77, 0};
    id.rdna4 = 1;
    id.probes_passed = 1;
    core->set_adapter(id);
    core->set_runtime_facts(true, true);
}

// The Cyberpunk 2077 binding with the stand-in's file name, version and export names standing in for the game's
// interface module; everything else (feature id, function names, GUIDs, buffer types, UI range) is the binding's.
FgAsiConfig MakeCpConfig(FgSl214Codec* codec, FgPresentOwners* owners, bool enabled, IFgPresentCallback* presenter) {
    const FgTitleBinding b = FgCp2077Binding();
    FgAsiConfig c;
    c.settings.enabled = enabled;
    c.settings.multiplier = 6;
    FgApplyBinding(b, &c);
    c.imports.interposer_leaf = kLeaf;
    c.imports.version_major = 3;
    c.imports.version_minor = 1;
    for (uint32_t i = 0; i < kFgRoleCount; ++i) c.imports.role_names[i] = kMockRoleNames[i];
    c.imports.factory2_name = kMockFactory;
    c.core.binding.version_major = 3;
    c.core.binding.version_minor = 1;
    c.core.modes_qualified = true;
    c.codec = codec;
    c.owners = owners;
    c.log = &CollectLog;
    c.presenter = presenter;
    return c;
}

// ---- the binding data ----------------------------------------------------------------------------------------------

void TestBindingData() {
    const FgTitleBinding b = FgCp2077Binding();
    CHECK(b.valid && b.imports.qualified && b.core.qualified);
    CHECK(b.id == "cp2077-2.14.1" && b.title == "Cyberpunk 2077" && b.exe_leaf == "Cyberpunk2077.exe");
    CHECK(b.fg_feature == 1000u);
    CHECK(b.imports.version_major == 2 && b.imports.version_minor == 14);
    CHECK(b.core.version_major == 2 && b.core.version_minor == 14);
    CHECK(!b.imports.interposer_leaf.empty());
    for (uint32_t i = 0; i < kFgRoleCount; ++i) {
        CHECK(!b.imports.role_names[i].empty());
        for (uint32_t k = 0; k < i; ++k) CHECK(b.imports.role_names[i] != b.imports.role_names[k]);
    }
    CHECK(b.imports.factory2_name == "CreateDXGIFactory2");
    CHECK(!b.imports.fn_options.empty() && !b.imports.fn_state.empty() && b.imports.fn_options != b.imports.fn_state);
    CHECK(b.core.fn_names.size() == 2 && b.core.fn_names[0] == b.imports.fn_options && b.core.fn_names[1] == b.imports.fn_state);
    CHECK(b.imports.abi_max_generated == 5 && b.core.ui_max_generated == 5 && b.core.presets_mask == WG_FG_PRESETS_ALL);
    // the FG inputs the game can tag: depth 0, motion vectors 1, HUD-less colour 2; the backbuffer comes from the chain
    CHECK(b.core.native_type[WG_FG_ROLE_DEPTH] == 0 && b.core.native_type[WG_FG_ROLE_MOTION] == 1 &&
          b.core.native_type[WG_FG_ROLE_HUDLESS] == 2);
    CHECK(b.core.native_type[WG_FG_ROLE_BACKBUFFER] == 53 && b.core.native_type[WG_FG_ROLE_UI_ALPHA] == 69 &&
          b.core.native_type[WG_FG_ROLE_DISTORTION] == 50 && b.core.native_type[WG_FG_ROLE_NO_WARP] == 54);
    const FgSl214Layout& l = b.layout;
    CHECK(FgSl214LayoutValid(l));
    CHECK(l.options_version == 5 && l.options_size == 120 && sizeof(WgNatFgOptions) == 120);
    CHECK(l.state_version == 4 && l.state_size == 88 && sizeof(WgNatFgState) == 88);
    CHECK(l.min_size == 100 && !l.dynamic_supported && l.constants_min_version == 1);
    CHECK(l.tag_lifetime_mask == (1u << WG_LIFE_UNTIL_PRESENT));
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) CHECK(l.optional_decl[i] == WG_FG_DECL_MAY);
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r) CHECK(l.native_type[r] == b.core.native_type[r]);
    CHECK(!FgGuidZero(l.requirements_guid));
    // selection by exe name (case-insensitive), nothing for any other exe
    FgTitleBinding s;
    CHECK(FgSelectShippedBinding(L"Cyberpunk2077.exe", &s) && s.id == b.id);
    CHECK(FgSelectShippedBinding(L"CYBERPUNK2077.EXE", &s));
    CHECK(!FgSelectShippedBinding(L"HogwartsLegacy.exe", &s));
    CHECK(!FgSelectShippedBinding(L"Cyberpunk2077.exe.bak", &s));
    CHECK(!FgSelectShippedBinding(L"", &s));
    CHECK(!FgSelectShippedBinding(L"Cyberpunk2077.exe", nullptr));
    // applying it to a configuration
    FgAsiConfig c;
    FgApplyBinding(b, &c);
    CHECK(c.imports.qualified && c.core.binding.qualified && c.core.ids.fg == 1000u && c.imports.factory2_name == b.imports.factory2_name);
}

// ---- the codec on its own ------------------------------------------------------------------------------------------

void TestGuidAndLayout() {
    WgGuid g = {};
    CHECK(FgParseGuid("00112233-4455-6677-8899-aabbccddeeff", &g));
    const uint8_t want[16] = {0x33, 0x22, 0x11, 0x00, 0x55, 0x44, 0x77, 0x66, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    CHECK(std::memcmp(g.b, want, 16) == 0);
    CHECK(FgParseGuid("00112233-4455-6677-8899-AABBCCDDEEFF", &g) && std::memcmp(g.b, want, 16) == 0);
    CHECK(!FgParseGuid("00112233-4455-6677-8899-aabbccddeef", &g));    // short
    CHECK(!FgParseGuid("00112233-4455-6677-8899-aabbccddeeff0", &g));  // long
    CHECK(!FgParseGuid("0011223304455-6677-8899-aabbccddeeff", &g));
    CHECK(!FgParseGuid("00112233-4455-6677-8899-aabbccddeegg", &g));
    CHECK(!FgParseGuid(nullptr, &g) && !FgParseGuid("00112233-4455-6677-8899-aabbccddeeff", nullptr));
    CHECK(FgGuidZero(WgGuid{}) && !FgGuidZero(g));
    // a layout with a missing or duplicated GUID, or other sizes, is not valid
    FgSl214Layout l = FgCp2077Binding().layout;
    FgSl214Layout bad = l;
    bad.constants_guid = WgGuid{};
    CHECK(!FgSl214LayoutValid(bad));
    bad = l;
    bad.state_guid = bad.options_guid;
    CHECK(!FgSl214LayoutValid(bad));
    bad = l;
    bad.options_size = 112;
    CHECK(!FgSl214LayoutValid(bad));
    bad = l;
    bad.state_size = 96;
    CHECK(!FgSl214LayoutValid(bad));
}

void TestChainWalk() {
    const FgSl214Layout l = FgCp2077Binding().layout;
    WgViewport vp = MakeViewport(l, 5);
    Raw120 o;
    RawHeader unknown;
    std::memset(&unknown, 0, sizeof unknown);
    // options -> unknown -> viewport
    SetHeader(&unknown, SyntheticGuid(9), 1, &vp);
    MakeOptions(o, l, 1, 2, &unknown);
    CHECK(FgFindInChain(&o, l.options_guid) == reinterpret_cast<const WgHeader*>(&o));
    CHECK(FgFindInChain(&o, l.viewport_guid) == &vp.h);
    CHECK(FgFindInChain(&o, l.state_guid) == nullptr);
    // viewport -> unknown -> options: found from the viewport too (any order)
    Raw120 o2;
    RawHeader unknown2;
    std::memset(&unknown2, 0, sizeof unknown2);
    WgViewport vp2 = MakeViewport(l, 6);
    MakeOptions(o2, l, 1, 2, nullptr);
    SetHeader(&unknown2, SyntheticGuid(10), 1, &o2);
    vp2.h.next = reinterpret_cast<WgHeader*>(&unknown2);
    CHECK(FgFindInChain(&vp2, l.options_guid) == reinterpret_cast<const WgHeader*>(&o2));
    CHECK(FgFindInChain(&unknown2, l.viewport_guid) == nullptr);  // behind, never before
    // an unknown structure first: skipped, not taken for the one asked for
    CHECK(FgFindInChain(&unknown, l.viewport_guid) == &vp.h);
    CHECK(FgFindInChain(&unknown, l.options_guid) == nullptr);
    CHECK(FgFindInChain(nullptr, l.options_guid) == nullptr);
    // a cycle and an over-long chain end the walk
    RawHeader a, b2;
    std::memset(&a, 0, sizeof a);
    std::memset(&b2, 0, sizeof b2);
    SetHeader(&a, SyntheticGuid(1), 1, &b2);
    SetHeader(&b2, SyntheticGuid(2), 1, &a);
    CHECK(FgFindInChain(&a, l.options_guid) == nullptr);
    std::vector<RawHeader> chain(kFgMaxChainLinks + 2);
    for (size_t i = 0; i < chain.size(); ++i)
        SetHeader(&chain[i], SyntheticGuid(static_cast<uint8_t>(20 + (i % 7))), 1, i + 1 < chain.size() ? &chain[i + 1] : nullptr);
    SetHeader(&chain.back(), l.viewport_guid, 1, nullptr);
    CHECK(FgFindInChain(&chain[0], l.viewport_guid) == nullptr);  // 33 links behind: past the bound
    CHECK(FgFindInChain(&chain[1], l.viewport_guid) == reinterpret_cast<const WgHeader*>(&chain.back()));  // 32: inside
    // the codec's lookups
    FgSl214Codec codec(l);
    CHECK(codec.find_options(&o) == reinterpret_cast<const WgNatFgOptions*>(&o));
    CHECK(codec.find_viewport(&o, nullptr) == &vp);
    Raw120 lone;
    MakeOptions(lone, l, 1, 2, nullptr);
    WgViewport explicit_vp = MakeViewport(l, 8);
    CHECK(codec.find_viewport(&lone, nullptr) == nullptr);
    CHECK(codec.find_viewport(&lone, &explicit_vp) == &explicit_vp);
    WgViewport wrong = explicit_vp;
    wrong.h.type = SyntheticGuid(3);
    CHECK(codec.find_viewport(&lone, &wrong) == nullptr);
    Put<uint64_t>(&lone, 24, 4);  // an older options version: not read
    CHECK(codec.find_options(&lone) == nullptr);
}

void TestOptionsCodec() {
    const FgSl214Layout l = FgCp2077Binding().layout;
    FgSl214Codec codec(l);
    WgViewport vp = MakeViewport(l, 7);
    Raw120 o;
    MakeOptions(o, l, kFgNatModeOn, 3, &vp);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_OK);
    // count 0: 25, also with the mode off; above the maximum: 38; at the maximum: accepted
    Put<uint32_t>(&o, 36, 0u);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_E_INVALID_PARAMETER);
    Put<uint32_t>(&o, 32, kFgNatModeOff);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_E_INVALID_PARAMETER);
    Put<uint32_t>(&o, 36, 6u);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_E_INVALID_STATE);
    Put<uint32_t>(&o, 36, 4u);
    CHECK(codec.check_options(nullptr, &o, 3) == WG_E_INVALID_STATE);
    CHECK(codec.check_options(nullptr, &o, 4) == WG_OK);
    Put<uint32_t>(&o, 36, 5u);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_OK);
    // modes: 0 off, 1 on, 2 auto accepted; 3 dynamic refused while the binding reports no dynamic support; 4 refused
    for (uint32_t m : {0u, 1u, 2u}) {
        Put<uint32_t>(&o, 32, m);
        CHECK(codec.check_options(nullptr, &o, 5) == WG_OK);
    }
    Put<uint32_t>(&o, 32, kFgNatModeDynamic);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_E_INVALID_PARAMETER);
    Put<uint32_t>(&o, 32, 4u);
    CHECK(codec.check_options(nullptr, &o, 5) == WG_E_INVALID_PARAMETER);
    FgSl214Layout dyn = l;
    dyn.dynamic_supported = true;
    FgSl214Codec dcodec(dyn);
    Put<uint32_t>(&o, 32, kFgNatModeDynamic);
    CHECK(dcodec.check_options(nullptr, &o, 5) == WG_OK);
    // the options or the viewport missing: 20
    Raw120 lone;
    MakeOptions(lone, l, 1, 2, nullptr);
    CHECK(codec.check_options(nullptr, &lone, 5) == WG_E_MISSING_INPUT_PARAMETER);
    CHECK(codec.check_options(&vp, &lone, 5) == WG_OK);
    CHECK(codec.check_options(&vp, nullptr, 5) == WG_E_MISSING_INPUT_PARAMETER);
    Raw120 wrong;
    MakeOptions(wrong, l, 1, 2, &vp);
    SetHeader(&wrong, SyntheticGuid(4), 5, &vp);  // not the options GUID
    CHECK(codec.check_options(&vp, &wrong, 5) == WG_E_MISSING_INPUT_PARAMETER);
    Raw120 older;
    MakeOptions(older, l, 1, 2, &vp);
    Put<uint64_t>(&older, 24, 4);  // a version without every member of the 120-byte layout
    CHECK(codec.check_options(&vp, &older, 5) == WG_E_MISSING_INPUT_PARAMETER);
    CHECK(codec.counters().options_refused >= 9);

    // decoding, from bytes at the layout's offsets
    MakeOptions(o, l, kFgNatModeOn, 3, &vp);
    WgFgOptions d = {};
    CHECK(codec.options(&vp, &o, &d));
    CHECK(d.enable == 1 && d.generated_count == 3 && d.preset == WG_FG_PRESET_NOT_SET && d.color_space == WG_FG_CS_SDR);
    CHECK(d.dynamic_resolution == 0 && d.output_width == 0 && d.render_max_width == 0);
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) CHECK(d.optional_decl[i] == WG_FG_DECL_MAY);
    Put<uint8_t>(&o, 112, 1);  // UI recomposition exactly 1: preset B
    CHECK(codec.options(&vp, &o, &d) && d.preset == WG_FG_PRESET_B);
    Put<uint8_t>(&o, 112, 2);  // anything else: the player's preset
    CHECK(codec.options(&vp, &o, &d) && d.preset == WG_FG_PRESET_NOT_SET);
    Put<uint32_t>(&o, 32, kFgNatModeOff);
    CHECK(codec.options(&vp, &o, &d) && d.enable == 0 && d.generated_count == 3);
    Put<uint32_t>(&o, 32, kFgNatModeAuto);
    CHECK(codec.options(&vp, &o, &d) && d.enable == 1);
    Put<uint32_t>(&o, 40, 8u | kFgNatFlagDynamicResolution);
    Put<uint32_t>(&o, 44, 640u);
    Put<uint32_t>(&o, 48, 360u);
    Put<uint32_t>(&o, 56, 1280u);
    Put<uint32_t>(&o, 60, 720u);
    Put<uint32_t>(&o, 64, 2560u);
    Put<uint32_t>(&o, 68, 1440u);
    Put<uint32_t>(&o, 72, 24u);
    Put<float>(&o, 116, 90.5f);  // the dynamic target rate (mode 3): carried in the layout, not used while refused
    CHECK(codec.options(&vp, &o, &d));
    CHECK(d.dynamic_resolution == 1 && d.render_min_width == 640 && d.render_min_height == 360);
    CHECK(d.render_max_width == 1280 && d.render_max_height == 720);
    CHECK(d.output_width == 2560 && d.output_height == 1440 && d.output_format == 24);
    const WgNatFgOptions* typed = codec.find_options(&o);
    CHECK(typed && typed->dynamic_target_fps == 90.5f && typed->ui_recomposition == 2 && typed->latency_matrices == 2 &&
          typed->flags == (8u | kFgNatFlagDynamicResolution) && typed->frames == 3);
    CHECK(!codec.options(&vp, &lone, nullptr));
    CHECK(!codec.options(&vp, &wrong, &d));
}

void TestStateCodec() {
    const FgSl214Layout l = FgCp2077Binding().layout;
    FgSl214Codec codec(l);
    WgFgState st = {};
    st.struct_size = sizeof(st);
    st.taken = 1;
    st.enabled = 1;
    st.effective_count = 3;
    st.max_generated = 5;
    st.generate_requests = 4;
    st.base_presents = 9;
    Raw88 s;
    MakeState(s, l, 4);
    CHECK(codec.write_state_for(st, nullptr, &s));
    CHECK(Get<uint64_t>(&s, 32) == 0 && Get<uint32_t>(&s, 40) == 0 && Get<uint32_t>(&s, 44) == 100);
    CHECK(Get<uint32_t>(&s, 48) == 4 && Get<uint32_t>(&s, 52) == 5);
    CHECK(Get<uint8_t>(&s, 56) == 0xcd && Get<uint8_t>(&s, 57) == 0);  // +56 is never written
    CHECK(Get<uint64_t>(&s, 64) == 0 && Get<uint64_t>(&s, 72) == 0 && Get<uint8_t>(&s, 80) == 0);
    CHECK(Get<uint8_t>(&s, 81) == 0xcd);
    // version 1: only +32..+51
    MakeState(s, l, 1);
    CHECK(codec.write_state(st, &s));
    CHECK(Get<uint32_t>(&s, 44) == 100 && Get<uint32_t>(&s, 48) == 4);
    CHECK(Get<uint32_t>(&s, 52) == 0xcdcdcdcdu && Get<uint8_t>(&s, 57) == 0xcd && Get<uint8_t>(&s, 64) == 0xcd &&
          Get<uint8_t>(&s, 80) == 0xcd);
    // version 2: + the maximum and the v-sync byte; version 3: + the fence members; not the dynamic byte
    MakeState(s, l, 2);
    CHECK(codec.write_state(st, &s) && Get<uint32_t>(&s, 52) == 5 && Get<uint8_t>(&s, 64) == 0xcd);
    MakeState(s, l, 3);
    CHECK(codec.write_state(st, &s) && Get<uint64_t>(&s, 64) == 0 && Get<uint8_t>(&s, 80) == 0xcd);
    // dynamic support from the binding
    FgSl214Layout dyn = l;
    dyn.dynamic_supported = true;
    FgSl214Codec dcodec(dyn);
    MakeState(s, l, 4);
    CHECK(dcodec.write_state(st, &s) && Get<uint8_t>(&s, 80) == 1);
    // the presented count: game image + generated while generating, 1 otherwise, 0 before any present
    st.generate_requests = 0;
    MakeState(s, l, 4);
    CHECK(codec.write_state(st, &s) && Get<uint32_t>(&s, 48) == 1);
    st.base_presents = 0;
    CHECK(codec.write_state(st, &s) && Get<uint32_t>(&s, 48) == 0);
    // status: a colour size below the minimum in the passed options (bit 1); the v-sync byte with the mode off
    WgViewport vp = MakeViewport(l, 7);
    Raw120 o;
    MakeOptions(o, l, kFgNatModeOff, 1, &vp);
    Put<uint32_t>(&o, 64, 64u);
    Put<uint32_t>(&o, 68, 480u);
    MakeState(s, l, 4);
    CHECK(codec.write_state_for(st, &o, &s));
    CHECK(Get<uint32_t>(&s, 40) == kFgNatStatusResolutionLow && Get<uint8_t>(&s, 57) == 1);
    Put<uint32_t>(&o, 64, 640u);
    Put<uint32_t>(&o, 32, kFgNatModeOn);
    CHECK(codec.write_state_for(st, &o, &s) && Get<uint32_t>(&s, 40) == 0 && Get<uint8_t>(&s, 57) == 0);
    Put<uint32_t>(&o, 40, 8u | kFgNatFlagDynamicResolution);
    Put<uint32_t>(&o, 44, 99u);
    Put<uint32_t>(&o, 48, 400u);
    CHECK(codec.write_state_for(st, &o, &s) && Get<uint32_t>(&s, 40) == kFgNatStatusResolutionLow);
    // status bit 8: a tri-state left unset in the last constants
    WgConstants c = MakeConstants(l, 1.0f, WG_TRI_UNSET);
    WgFgConstants dc = {};
    CHECK(codec.constants(&c, nullptr, &vp, &dc) && codec.unset_constants() == 4u);
    CHECK(codec.write_state(st, &s) && (Get<uint32_t>(&s, 40) & kFgNatStatusConstantUnset));
    c.reset = WG_TRI_FALSE;
    CHECK(codec.constants(&c, nullptr, &vp, &dc) && codec.unset_constants() == 0u);
    CHECK(codec.write_state(st, &s) && Get<uint32_t>(&s, 40) == 0);
    // not a state structure: nothing written
    Raw88 wrong;
    MakeState(wrong, l, 4);
    SetHeader(&wrong, SyntheticGuid(5), 4, nullptr);
    CHECK(!codec.write_state(st, &wrong) && Get<uint32_t>(&wrong, 44) == 0xcdcdcdcdu);
    MakeState(wrong, l, 0);
    CHECK(!codec.write_state(st, &wrong));
    CHECK(!codec.write_state(st, nullptr));
}

// The FG call lines' words for the options and state structures (synthetic structures; read only, never written).
void TestDescribeCodec() {
    const FgSl214Layout l = FgCp2077Binding().layout;
    FgSl214Codec codec(l);
    WgViewport vp = MakeViewport(l, 7);
    Raw120 o;
    MakeOptions(o, l, kFgNatModeOn, 3, &vp);
    Put<uint8_t>(&o, 112, 1);
    CHECK(codec.describe_options(&o) == "mode 1 (on), frames 3, flags 0x8, ui recomposition 1");
    MakeOptions(o, l, 9, 1, nullptr);
    CHECK(codec.describe_options(&o) == "mode 9 (unknown), frames 1, flags 0x8, ui recomposition 0");
    CHECK(codec.describe_options(&vp) == "options not in the chain");
    CHECK(codec.describe_options(nullptr) == "options not in the chain");
    Raw88 s;
    MakeState(s, l, 4);
    Put<uint32_t>(&s, 40, 0x8u);
    Put<uint32_t>(&s, 52, 2u);
    CHECK(codec.describe_state(&s) == "frames max 2, status 0x8, state v4");
    MakeState(s, l, 1);
    Put<uint32_t>(&s, 40, 0u);
    CHECK(codec.describe_state(&s) == "status 0x0, state v1 (no frames max)");
    CHECK(Get<uint32_t>(&s, 52) == 0xcdcdcdcdu);  // describing writes nothing
    CHECK(codec.describe_state(nullptr) == "state not in the chain");
}

void TestTagsAndConstantsCodec() {
    const FgSl214Layout l = FgCp2077Binding().layout;
    FgSl214Codec codec(l);
    WgViewport vp = MakeViewport(l, 7);
    FakeBuffer depth, motion, hud, colour, other;
    WgResource rd = MakeResource(l, &depth, 64, 32, 40);
    WgResource rm = MakeResource(l, &motion, 64, 32, 34);
    WgResource rh = MakeResource(l, &hud, 128, 64, 24);
    WgResource rc = MakeResource(l, &colour, 64, 32, 10);
    WgResource rbuf = MakeResource(l, &other, 64, 32, 0);
    rbuf.kind = WG_RES_BUFFER;
    WgResource rwrong = rd;
    rwrong.h.type = SyntheticGuid(6);
    WgTag tags[9] = {MakeTag(l, 0, &rd, WG_LIFE_UNTIL_PRESENT),  MakeTag(l, 1, &rm, WG_LIFE_UNTIL_PRESENT),
                     MakeTag(l, 2, &rh, WG_LIFE_UNTIL_PRESENT),  MakeTag(l, 3, &rc, WG_LIFE_UNTIL_PRESENT),
                     MakeTag(l, 0, &rd, WG_LIFE_ONLY_DURING_TAG), MakeTag(l, 1, &rm, WG_LIFE_UNTIL_EVALUATE),
                     MakeTag(l, 1, &rbuf, WG_LIFE_UNTIL_PRESENT), MakeTag(l, 2, &rwrong, WG_LIFE_UNTIL_PRESENT),
                     MakeTag(l, 0, nullptr, WG_LIFE_UNTIL_PRESENT)};
    tags[2].extent = WgExtent{4, 8, 96, 48};  // top, left, width, height
    WgTag wrong_tag = tags[0];
    wrong_tag.h.type = SyntheticGuid(7);
    WgFgTag out[10] = {};
    CHECK(codec.tags(nullptr, &vp, tags, 9, out));
    CHECK(out[0].native_type == 0 && out[0].texture.resource == &depth && out[0].texture.format == 40 &&
          out[0].texture.width == 64 && out[0].texture.height == 32 && out[0].texture.state == 0x80);
    CHECK(out[0].texture.frame_token == 1 && out[0].texture.viewport_id == 7);
    CHECK(out[0].texture.rect.x == 0 && out[0].texture.rect.y == 0 && out[0].texture.rect.width == 64 &&
          out[0].texture.rect.height == 32);
    CHECK(out[1].native_type == 1 && out[1].texture.resource == &motion);
    CHECK(out[2].native_type == 2 && out[2].texture.resource == &hud && out[2].texture.rect.x == 8 &&
          out[2].texture.rect.y == 4 && out[2].texture.rect.width == 96 && out[2].texture.rect.height == 48);
    CHECK(out[3].native_type == WG_FG_NATIVE_TYPE_UNBOUND);  // a super resolution input: not FG's
    CHECK(out[4].native_type == WG_FG_NATIVE_TYPE_UNBOUND);  // valid only during the call: would need a copy
    CHECK(out[5].native_type == WG_FG_NATIVE_TYPE_UNBOUND);  // valid until evaluate: FG is never evaluated
    CHECK(out[6].native_type == WG_FG_NATIVE_TYPE_UNBOUND);  // not a 2D texture
    CHECK(out[7].native_type == WG_FG_NATIVE_TYPE_UNBOUND);  // the resource's GUID is not the binding's
    CHECK(out[8].native_type == WG_FG_NATIVE_TYPE_UNBOUND);  // a removal
    for (int i = 0; i < 9; ++i) CHECK(out[i].struct_size == sizeof(WgFgTag));
    CHECK(codec.tags(nullptr, &vp, &wrong_tag, 1, out) && out[0].native_type == WG_FG_NATIVE_TYPE_UNBOUND);
    const FgSl214Counters k = codec.counters();
    CHECK(k.tags_seen == 10 && k.tags_bound == 3 && k.tags_lifetime_refused == 2 && k.tags_not_texture == 1 &&
          k.tags_wrong_guid == 2);
    WgViewport wrong_vp = vp;
    wrong_vp.h.type = SyntheticGuid(8);
    CHECK(!codec.tags(nullptr, &wrong_vp, tags, 1, out));
    CHECK(!codec.tags(nullptr, &vp, nullptr, 1, out));
    CHECK(codec.tags(nullptr, nullptr, tags, 1, out) && out[0].texture.viewport_id == 0);
    // the frame clock: everything up to the next present carries presents + 1
    CHECK(codec.frame_for_present() == 1);
    codec.note_present();
    CHECK(codec.presents() == 1 && codec.frame_for_present() == 2);
    CHECK(codec.tags(nullptr, &vp, tags, 1, out) && out[0].texture.frame_token == 2);

    // constants
    WgConstants c1 = MakeConstants(l, 1.0f, WG_TRI_TRUE);
    WgFgConstants d = {};
    CHECK(codec.constants(&c1, nullptr, &vp, &d));
    CHECK(d.frame_token == 2 && d.viewport_id == 7 && d.jitter_x == 0.25f && d.jitter_y == -0.375f);
    CHECK(d.motion_scale_x == 0.5f && d.motion_scale_y == -0.5f && d.color_space == WG_FG_CS_SDR);
    CHECK(d.flags == (WG_FG_FLAG_RESET | WG_FG_FLAG_DEPTH_REVERSED));
    CHECK(d.camera_current[3] == c1.view_to_clip[3] && d.camera_previous[3] == c1.view_to_clip[3]);
    codec.note_present();
    WgConstants c2 = MakeConstants(l, 2.0f, WG_TRI_FALSE);
    c2.depth_inverted = WG_TRI_FALSE;
    CHECK(codec.constants(&c2, nullptr, &vp, &d));
    CHECK(d.frame_token == 3 && d.flags == 0 && d.camera_current[5] == c2.view_to_clip[5] &&
          d.camera_previous[5] == c1.view_to_clip[5]);
    WgConstants c2b = MakeConstants(l, 2.5f, WG_TRI_FALSE);  // a second call in the same interval
    CHECK(codec.constants(&c2b, nullptr, &vp, &d) && d.camera_previous[5] == c1.view_to_clip[5]);
    codec.note_present();
    WgConstants c3 = MakeConstants(l, 3.0f, WG_TRI_TRUE);  // a reset: no previous matrix across it
    CHECK(codec.constants(&c3, nullptr, &vp, &d) && d.camera_previous[5] == c3.view_to_clip[5]);
    WgConstants bad = c3;
    bad.h.type = SyntheticGuid(9);
    CHECK(!codec.constants(&bad, nullptr, &vp, &d));
    bad = c3;
    bad.h.version = 0;
    CHECK(!codec.constants(&bad, nullptr, &vp, &d));
    CHECK(!codec.constants(&c3, nullptr, &wrong_vp, &d));
    CHECK(!codec.constants(nullptr, nullptr, &vp, &d));
    CHECK(codec.counters().constants_refused == 4);

    // requirements (not asked by this title; answered from the binding): D3D12, one viewport, depth + motion required
    WgRequirements rq = {};
    rq.h.type = l.requirements_guid;
    WgFgCapability cap = {};
    cap.supported = 1;
    CHECK(codec.write_requirements(cap, &rq));
    CHECK(rq.flags == 2u && rq.max_viewports == 1 && rq.required_tag_count == 2 && rq.required_tags &&
          rq.required_tags[0] == 0 && rq.required_tags[1] == 1);
    cap.supported = 0;
    CHECK(codec.write_requirements(cap, &rq) && rq.required_tag_count == 0 && rq.required_tags == nullptr);
    rq.h.type = SyntheticGuid(11);
    CHECK(!codec.write_requirements(cap, &rq) && !codec.write_requirements(cap, nullptr));
}

// ---- the game session -----------------------------------------------------------------------------------------------

struct Frame {
    FakeBuffer depth, motion, hud, sr;
    WgResource rd, rm, rh, rs;
    WgTag td, tm, th, ts;
    void make(const FgSl214Layout& l, uint32_t lifetime) {
        rd = MakeResource(l, &depth, 64, 32, 40);
        rm = MakeResource(l, &motion, 64, 32, 34);
        rh = MakeResource(l, &hud, 96, 48, 24);
        rs = MakeResource(l, &sr, 64, 32, 10);
        td = MakeTag(l, 0, &rd, lifetime);
        tm = MakeTag(l, 1, &rm, lifetime);
        th = MakeTag(l, 2, &rh, WG_LIFE_UNTIL_PRESENT);
        ts = MakeTag(l, 3, &rs, WG_LIFE_UNTIL_PRESENT);
    }
};

// One tag per call, as the game tags; every call must reach the stand-in with the game's own arguments.
void TagOne(const GameModule& game, const WgViewport* vp, const WgTag* t, void* cmd) {
    FgMockRecord* r = fgmock_record();
    const uint32_t before = r->calls[kMockTag];
    CHECK(game.tag(vp, t, 1, cmd) == r->result[kMockTag]);
    CHECK(r->calls[kMockTag] == before + 1 && r->last[kMockTag].a == vp && r->last[kMockTag].b == t &&
          r->last[kMockTag].c == cmd && r->last[kMockTag].count == 1);
}

void SendConstants(const GameModule& game, const WgConstants* c, const WgViewport* vp) {
    FgMockRecord* r = fgmock_record();
    const WgFrameToken* token = reinterpret_cast<const WgFrameToken*>(&r->result[0]);  // opaque: never read
    CHECK(game.constants(c, token, vp) == r->result[kMockConstants]);
    CHECK(r->last[kMockConstants].a == c && r->last[kMockConstants].b == token && r->last[kMockConstants].c == vp);
}

void TestCp2077Session(const GameModule& game) {
    const FgTitleBinding b = FgCp2077Binding();
    const FgSl214Layout& l = b.layout;
    FgSl214Codec codec(l);
    FgPresentOwners owners;
    TestPresenter presenter;
    FakeFactory factory;
    fgmock_reset();
    SetMockResults();
    fgmock_set_factory(&factory);
    FgMockRecord* r = fgmock_record();
    FgAsiTransport t;
    presenter.transport = &t;
    CHECK(t.install(MakeCpConfig(&codec, &owners, true, &presenter)) == kFgInstalled);
    size_t factory_slots = 0;
    for (const FgPatch& p : t.patches())
        if (p.role == kFgSlotCreateFactory2 && p.module == game.h) {
            ++factory_slots;
            CHECK(p.handler == FgAsiTransport::handler_for(kFgSlotCreateFactory2) && !p.delay_load);
        }
    CHECK(factory_slots == 1);
    CHECK(game.slots()[9] == FgAsiTransport::handler_for(kFgSlotCreateFactory2));
    CHECK(t.genuine(kFgSlotCreateFactory2) == GenuineExport(kMockFactory));
    CHECK(FgRoleWord(kFgSlotCreateFactory2) == std::string("create-factory"));

    WgFgLuid luid = {77, 0};
    WgAdapterInfo adapter = {};
    adapter.luid = &luid;
    adapter.luid_size = sizeof(luid);
    // not taken yet: 1000 reaches the genuine interface with the game's arguments
    uint32_t calls = r->calls[kMockIsSupported];
    CHECK(game.is_supported(kFeatFg, &adapter) == r->result[kMockIsSupported] && r->calls[kMockIsSupported] == calls + 1);
    void* fn = nullptr;
    CHECK(game.get_function(kFeatFg, b.imports.fn_options.c_str(), &fn) == WG_E_FEATURE_MISSING && fn == nullptr);
    CHECK(r->last[kMockGetFunction].feature == kFeatFg && r->last[kMockGetFunction].b == &fn);

    MakeTaken(t.core());
    Executor ex;
    WgFgExecutorV1 exv = {sizeof(WgFgExecutorV1), WG_FG_ADAPTER_ABI_VERSION, &ex, &ExecutorSubmit};
    t.core()->set_executor(&exv);
    CHECK(t.core()->fg_taken() && t.core()->advertised_max_generated() == 5);

    // support: 1000 answered here; SR, RR, latency, PC latency, NIS and an unknown id unchanged
    calls = r->calls[kMockIsSupported];
    CHECK(game.is_supported(kFeatFg, &adapter) == WG_OK && r->calls[kMockIsSupported] == calls);
    for (uint32_t f : {kFeatSr, kFeatRr, kFeatLatency, kFeatNis, kFeatPcLatency, 4321u}) {
        CHECK(game.is_supported(f, &adapter) == r->result[kMockIsSupported]);
        CHECK(r->last[kMockIsSupported].feature == f && r->last[kMockIsSupported].a == &adapter);
    }
    CHECK(r->calls[kMockIsSupported] == calls + 6);
    // functions: the binding's two FG names give WrongGPU's; every other feature's lookup is the genuine one
    void* fn_opt = nullptr;
    void* fn_st = nullptr;
    calls = r->calls[kMockGetFunction];
    CHECK(game.get_function(kFeatFg, b.imports.fn_options.c_str(), &fn_opt) == WG_OK &&
          fn_opt == FgAsiTransport::options_handler());
    CHECK(game.get_function(kFeatFg, b.imports.fn_state.c_str(), &fn_st) == WG_OK &&
          fn_st == FgAsiTransport::state_handler());
    void* fn_unknown = nullptr;
    CHECK(game.get_function(kFeatFg, "fgmock_fn_unknown", &fn_unknown) == WG_E_FEATURE_MISSING && !fn_unknown);
    CHECK(r->calls[kMockGetFunction] == calls);
    void* fn_sr = nullptr;
    CHECK(game.get_function(kFeatSr, FG_MOCK_FN_SR_OPTIONS, &fn_sr) == r->result[kMockGetFunction] &&
          fn_sr == fgmock_feature_fn(kMockSrOptions));
    for (uint32_t f : {kFeatRr, kFeatLatency, kFeatPcLatency}) {
        void* o = nullptr;
        CHECK(game.get_function(f, "fgmock_fn_none", &o) == WG_E_FEATURE_MISSING);
        CHECK(r->last[kMockGetFunction].feature == f && r->last[kMockGetFunction].b == &o);
    }
    CHECK(r->calls[kMockGetFunction] == calls + 4);

    // the factory: the genuine creation's object returned unchanged; the creation hook goes on it, once
    IDXGIFactory2* f2 = nullptr;
    CHECK(game.create_factory2(0, __uuidof(IDXGIFactory2), reinterpret_cast<void**>(&f2)) == S_OK && f2 == &factory);
    CHECK(r->calls[kMockCreateFactory2] == 1 && r->last[kMockCreateFactory2].b == &f2);
    CHECK(FgSwapChainHook::Installed());
    FgPresentBridgeStats ps = t.present_stats();
    CHECK(ps.factories == 1 && ps.factory_hooked == 1);
    IDXGIFactory* f1 = nullptr;
    CHECK(game.create_factory2(1, __uuidof(IDXGIFactory), reinterpret_cast<void**>(&f1)) == S_OK &&
          f1 == static_cast<IDXGIFactory*>(&factory));
    CHECK(r->last[kMockCreateFactory2].feature == 1 && t.present_stats().factories == 2);
    fgmock_set_factory(nullptr);  // a failing creation: its result unchanged, nothing hooked
    IDXGIFactory2* none = nullptr;
    CHECK(game.create_factory2(0, __uuidof(IDXGIFactory2), reinterpret_cast<void**>(&none)) == E_FAIL && !none);
    fgmock_set_factory(&factory);

    // the chain the game creates on that factory: wrapped, claimed for the bridge (the swap-chain wrapper owner)
    IDXGISwapChain1* sc = nullptr;
    CHECK(CreateThroughTable(f2, &sc) == S_OK && sc);
    FgSwapChain* w = sc ? FgSwapChain::FromInterface(sc) : nullptr;
    CHECK(w && w->Attached() && factory.made.size() == 1 && w->Inner() == factory.made[0]);
    if (!w || factory.made.empty()) return;
    FakeSwapChain* inner = factory.made[0];
    CHECK(owners.owner_of(w->Id()) == WG_FG_OWNER_SWAPCHAIN_WRAPPER);
    ps = t.present_stats();
    CHECK(ps.chains_created == 1 && ps.chains_attached == 1);

    // state before any options; then the options checks; then accepted
    const auto set_options = reinterpret_cast<FgOptionsFn>(fn_opt);
    const auto get_state = reinterpret_cast<FgStateFn>(fn_st);
    WgViewport vp = MakeViewport(l, 7);
    Raw120 opt;
    MakeOptions(opt, l, kFgNatModeOn, 3, &vp);
    Raw88 st;
    MakeState(st, l, 4);
    CHECK(get_state(&vp, &st, &opt) == WG_OK);
    CHECK(Get<uint32_t>(&st, 52) == 5 && Get<uint32_t>(&st, 44) == 100 && Get<uint8_t>(&st, 80) == 0 &&
          Get<uint32_t>(&st, 40) == 0 && Get<uint64_t>(&st, 32) == 0);
    CHECK(r->calls[kMockFgState] == 0);
    Put<uint32_t>(&opt, 36, 0u);
    CHECK(set_options(&vp, &opt) == WG_E_INVALID_PARAMETER);
    Put<uint32_t>(&opt, 32, kFgNatModeOff);
    CHECK(set_options(&vp, &opt) == WG_E_INVALID_PARAMETER);
    Put<uint32_t>(&opt, 32, kFgNatModeOn);
    Put<uint32_t>(&opt, 36, 6u);
    CHECK(set_options(&vp, &opt) == WG_E_INVALID_STATE);
    Put<uint32_t>(&opt, 32, kFgNatModeDynamic);
    Put<uint32_t>(&opt, 36, 3u);
    CHECK(set_options(&vp, &opt) == WG_E_INVALID_PARAMETER);  // dynamic: not reported as supported
    Raw120 lone;
    MakeOptions(lone, l, kFgNatModeOn, 3, nullptr);
    CHECK(set_options(nullptr, &lone) == WG_E_MISSING_INPUT_PARAMETER);
    Raw120 wrong;
    MakeOptions(wrong, l, kFgNatModeOn, 3, &vp);
    SetHeader(&wrong, l.state_guid, 5, &vp);  // another structure where the options should be
    CHECK(set_options(&vp, &wrong) == WG_E_MISSING_INPUT_PARAMETER);
    Put<uint32_t>(&opt, 32, kFgNatModeOn);
    CHECK(set_options(&vp, &opt) == WG_OK);
    CHECK(set_options(&vp, &opt) == WG_OK);  // a repeated call is accepted
    CHECK(r->calls[kMockFgOptions] == 0);
    const WgFgState cs = t.core()->state();
    CHECK(cs.enabled == 1 && cs.effective_count == 3 && cs.effective_preset == WG_FG_PRESET_A);
    Raw88 st_wrong;
    MakeState(st_wrong, l, 4);
    SetHeader(&st_wrong, l.options_guid, 4, nullptr);
    CHECK(get_state(&vp, &st_wrong, nullptr) == WG_E_INVALID_PARAMETER);

    // frame 1: depth, motion vectors and HUD-less colour (one tag per call), a super resolution input, the constants
    void* cmd = &opt;
    Frame f1r;
    f1r.make(l, WG_LIFE_UNTIL_PRESENT);
    TagOne(game, &vp, &f1r.ts, cmd);
    TagOne(game, &vp, &f1r.td, cmd);
    TagOne(game, &vp, &f1r.tm, cmd);
    TagOne(game, &vp, &f1r.th, cmd);
    WgConstants c1 = MakeConstants(l, 1.0f, WG_TRI_TRUE);
    SendConstants(game, &c1, &vp);
    CHECK(t.core()->frames_kept() == 1);
    // the game presents: the interval goes to the core first (the presented buffer as colour), then the presenter
    inner->backbuffer_index = 1;
    CHECK(sc->Present(1, 0) == S_OK);
    CHECK(ex.submits == 1);
    CHECK(ex.last.color.resource == static_cast<IUnknown*>(&inner->buffers[1]));
    CHECK(ex.last.color.width == 96 && ex.last.color.height == 48 && ex.last.color.format == DXGI_FORMAT_R10G10B10A2_UNORM);
    CHECK(ex.last.motion.resource == &f1r.motion && ex.last.depth.resource == &f1r.depth);
    CHECK((ex.last.optional_present & 1u) && ex.last.optional[0].resource == &f1r.hud);
    CHECK(ex.last.frame_token == 1 && ex.last.viewport_id == 7 && ex.last.generated_count == 3 && ex.last.generated_index == 0);
    CHECK(ex.last.jitter_x == c1.jitter[0] && ex.last.motion_scale_y == -0.5f);
    CHECK(ex.last.flags == (WG_FG_FLAG_RESET | WG_FG_FLAG_DEPTH_REVERSED));
    CHECK(ex.last.adapter_luid.low == 77 && ex.session.viewport_id == 7 && ex.session.swapchain_id == w->Id());
    ps = t.present_stats();
    CHECK(ps.submits == 1 && ps.generate == 1 && ps.last.status == WG_FG_PRESENT_GENERATE);
    CHECK(ps.last_desc.owner == WG_FG_OWNER_SWAPCHAIN_WRAPPER && ps.last_desc.window_id == 0x5a &&
          ps.last_desc.backbuffer.resource == static_cast<IUnknown*>(&inner->buffers[1]));
    CHECK(presenter.begins == 1 && presenter.presents == 1 && presenter.submits_at_begin == 0 &&
          presenter.submits_at_present == 1);
    CHECK(inner->presents == 1 && inner->buffers[1].refs == 1);  // the buffer reference given back
    CHECK(codec.presents() == 1);

    // frame 2: new inputs; the previous camera matrix is frame 1's
    Frame f2r;
    f2r.make(l, WG_LIFE_UNTIL_PRESENT);
    TagOne(game, &vp, &f2r.td, cmd);
    TagOne(game, &vp, &f2r.tm, cmd);
    TagOne(game, &vp, &f2r.th, cmd);
    WgConstants c2 = MakeConstants(l, 2.0f, WG_TRI_FALSE);
    SendConstants(game, &c2, &vp);
    inner->backbuffer_index = 2;
    presenter.generated = 3;  // the presenter's own generated images, through the inner presenter
    CHECK(sc->Present(1, 0) == S_OK);
    CHECK(ex.submits == 2 && ex.last.frame_token == 2 && ex.last.color.resource == static_cast<IUnknown*>(&inner->buffers[2]));
    CHECK(ex.last.motion.resource == &f2r.motion && ex.last.depth.resource == &f2r.depth);
    CHECK(ex.last.camera_current[5] == c2.view_to_clip[5] && ex.last.camera_previous[5] == c1.view_to_clip[5]);
    CHECK(ex.last.flags == WG_FG_FLAG_DEPTH_REVERSED);
    CHECK(inner->presents == 1 + 4 && presenter.presents == 2);
    presenter.generated = 0;

    // frame 3: only a super resolution input: the game image without generation, still presented once
    Frame f3r;
    f3r.make(l, WG_LIFE_UNTIL_PRESENT);
    TagOne(game, &vp, &f3r.ts, cmd);
    WgConstants c3 = MakeConstants(l, 3.0f, WG_TRI_FALSE);
    SendConstants(game, &c3, &vp);
    CHECK(sc->Present(1, 0) == S_OK);
    ps = t.present_stats();
    CHECK(ex.submits == 2 && ps.real_only == 1 && ps.last.status == WG_FG_PRESENT_REAL_ONLY);
    CHECK(inner->presents == 6 && presenter.presents == 3);

    // frame 4: depth and motion valid only during the tag call: not kept (no copy at this stage): no generation
    Frame f4r;
    f4r.make(l, WG_LIFE_ONLY_DURING_TAG);
    TagOne(game, &vp, &f4r.td, cmd);
    TagOne(game, &vp, &f4r.tm, cmd);
    WgConstants c4 = MakeConstants(l, 4.0f, WG_TRI_FALSE);
    SendConstants(game, &c4, &vp);
    CHECK(sc->Present(1, 0) == S_OK);
    CHECK(ex.submits == 2 && t.present_stats().real_only == 2 && codec.counters().tags_lifetime_refused == 2);

    // frame 5: a present that is still drawing keeps the interval open; the retry does not submit again
    Frame f5r;
    f5r.make(l, WG_LIFE_UNTIL_PRESENT);
    TagOne(game, &vp, &f5r.td, cmd);
    TagOne(game, &vp, &f5r.tm, cmd);
    WgConstants c5 = MakeConstants(l, 5.0f, WG_TRI_FALSE);
    SendConstants(game, &c5, &vp);
    inner->results = {DXGI_ERROR_WAS_STILL_DRAWING};
    CHECK(sc->Present(0, DXGI_PRESENT_DO_NOT_WAIT) == DXGI_ERROR_WAS_STILL_DRAWING);
    CHECK(sc->Present(0, DXGI_PRESENT_DO_NOT_WAIT) == S_OK);
    CHECK(ex.submits == 3 && ex.last.frame_token == 5 && t.present_stats().submits == 5 && presenter.retries == 1);
    CHECK((ex.last.optional_present & 1u) == 0);  // no HUD-less this frame: the optional input is simply absent
    // a test present: never an interval
    const uint64_t submits = t.present_stats().submits;
    CHECK(sc->Present(0, DXGI_PRESENT_TEST) == S_OK && t.present_stats().submits == submits && inner->tests == 1);

    // evaluate: super resolution and ray reconstruction keep their genuine path; 1000 never reaches it
    const WgHeader* inputs[1] = {&vp.h};
    const WgFrameToken* token = reinterpret_cast<const WgFrameToken*>(&c5);
    calls = r->calls[kMockEvaluate];
    CHECK(game.evaluate(kFeatSr, token, inputs, 1, nullptr) == r->result[kMockEvaluate]);
    CHECK(r->last[kMockEvaluate].feature == kFeatSr && r->last[kMockEvaluate].a == token &&
          r->last[kMockEvaluate].b == inputs && r->last[kMockEvaluate].count == 1);
    CHECK(game.evaluate(kFeatRr, token, inputs, 1, nullptr) == r->result[kMockEvaluate] &&
          r->last[kMockEvaluate].feature == kFeatRr);
    CHECK(game.evaluate(kFeatFg, token, inputs, 1, nullptr) == WG_OK && r->calls[kMockEvaluate] == calls + 2);
    const FgTransportStats ts = t.stats();
    CHECK(ts.codec_failures == 1 && ts.option_calls >= 8 && ts.state_calls == 2);  // the wrong-GUID state above

    // restore: the creation hook off, the chain's owner given back (the presenter told), every slot genuine
    t.restore();
    CHECK(!FgSwapChainHook::Installed());
    CHECK(!w->Attached() && presenter.detached == 1);
    CHECK(owners.owner_of(w->Id()) == WG_FG_OWNER_NONE);
    CHECK(SlotsAreGenuine(game));
    const uint32_t inner_before = inner->presents;
    CHECK(sc->Present(0, 0) == S_OK && inner->presents == inner_before + 1 && ex.submits == 3);
    CHECK(sc->Release() == 0 && inner->refs == 0);
    IDXGISwapChain1* raw = nullptr;  // a chain created after restore is not wrapped
    CHECK(CreateThroughTable(f2, &raw) == S_OK && raw == factory.made.back() && FgSwapChain::FromInterface(raw) == nullptr);
    f2->Release();
    f1->Release();
    fgmock_set_factory(nullptr);
}

// fg=off: nothing read, nothing patched; the factory and every call keep the genuine path.
void TestOffGate(const GameModule& game) {
    const FgSl214Layout l = FgCp2077Binding().layout;
    FgSl214Codec codec(l);
    FgPresentOwners owners;
    FakeFactory factory;
    fgmock_reset();
    SetMockResults();
    fgmock_set_factory(&factory);
    FgAsiTransport t;
    const uint64_t factories_before = t.present_stats().factories;
    CHECK(t.install(MakeCpConfig(&codec, &owners, false, nullptr)) == kFgInstallOff);
    CHECK(!t.active() && t.patch_count() == 0 && t.core() == nullptr && t.interface_module() == nullptr);
    CHECK(SlotsAreGenuine(game));
    IDXGIFactory2* f2 = nullptr;
    CHECK(game.create_factory2(0, __uuidof(IDXGIFactory2), reinterpret_cast<void**>(&f2)) == S_OK && f2 == &factory);
    CHECK(fgmock_record()->calls[kMockCreateFactory2] == 1 && t.present_stats().factories == factories_before);
    CHECK(!FgSwapChainHook::Installed());
    IDXGISwapChain1* sc = nullptr;
    CHECK(CreateThroughTable(f2, &sc) == S_OK && FgSwapChain::FromInterface(sc) == nullptr);
    FgMockRecord* r = fgmock_record();
    CHECK(game.is_supported(kFeatFg, nullptr) == r->result[kMockIsSupported] && r->calls[kMockIsSupported] == 1);
    CHECK(codec.counters().options_checked == 0 && codec.counters().tags_seen == 0);
    if (sc) sc->Release();
    if (f2) f2->Release();
    fgmock_set_factory(nullptr);
}

// fg=on with the binding, but FG not taken (the adapter facts are not there): chains are wrapped as pass-throughs,
// every call forwards, the state call reaches the genuine FG function with all three of the game's arguments.
void TestNotTaken(const GameModule& game) {
    const FgTitleBinding b = FgCp2077Binding();
    FgSl214Codec codec(b.layout);
    FgPresentOwners owners;
    FakeFactory factory;
    fgmock_reset();
    SetMockResults();
    fgmock_set_factory(&factory);
    FgMockRecord* r = fgmock_record();
    FgAsiTransport t;
    FgAsiConfig cfg = MakeCpConfig(&codec, &owners, true, nullptr);
    // the stand-in serves its own FG function names: here they stand in for the binding's
    cfg.imports.fn_options = FG_MOCK_FN_FG_OPTIONS;
    cfg.imports.fn_state = FG_MOCK_FN_FG_STATE;
    cfg.core.binding.fn_names = {FG_MOCK_FN_FG_OPTIONS, FG_MOCK_FN_FG_STATE};
    CHECK(t.install(cfg) == kFgInstalled);
    CHECK(t.core() && !t.core()->fg_taken() && t.core()->route_reason() == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    IDXGIFactory2* f2 = nullptr;
    CHECK(game.create_factory2(0, __uuidof(IDXGIFactory2), reinterpret_cast<void**>(&f2)) == S_OK && f2 == &factory);
    CHECK(FgSwapChainHook::Installed());
    IDXGISwapChain1* sc = nullptr;
    CHECK(CreateThroughTable(f2, &sc) == S_OK && sc);
    FgSwapChain* w = sc ? FgSwapChain::FromInterface(sc) : nullptr;
    CHECK(w && !w->Attached());
    const FgPresentBridgeStats ps = t.present_stats();
    CHECK(ps.chains_created == 1 && ps.chains_attached == 0);
    if (sc) {
        CHECK(sc->Present(0, 0) == S_OK && factory.made[0]->presents == 1 && t.present_stats().submits == 0);
    }
    CHECK(game.is_supported(kFeatFg, nullptr) == r->result[kMockIsSupported] && r->calls[kMockIsSupported] == 1);
    // the options / state functions (kept by a game from an earlier lookup) forward with every argument
    r->result[kMockGetFunction] = WG_OK;
    const auto get_state = reinterpret_cast<FgStateFn>(FgAsiTransport::state_handler());
    const auto set_options = reinterpret_cast<FgOptionsFn>(FgAsiTransport::options_handler());
    WgViewport vp = MakeViewport(b.layout, 3);
    Raw120 opt;
    MakeOptions(opt, b.layout, kFgNatModeOn, 0, &vp);  // even an invalid count: not WrongGPU's to check
    Raw88 st;
    MakeState(st, b.layout, 4);
    CHECK(set_options(&vp, &opt) == r->result[kMockFgOptions]);
    CHECK(r->last[kMockFgOptions].a == &vp && r->last[kMockFgOptions].b == &opt);
    CHECK(get_state(&vp, &st, &opt) == r->result[kMockFgState]);
    CHECK(r->last[kMockFgState].a == &vp && r->last[kMockFgState].b == &st && r->last[kMockFgState].c == &opt);
    CHECK(Get<uint32_t>(&st, 44) == 0xcdcdcdcdu);  // nothing written by WrongGPU
    CHECK(codec.counters().options_checked == 0 && codec.counters().states_written == 0);
    t.restore();
    CHECK(!FgSwapChainHook::Installed() && SlotsAreGenuine(game));
    if (sc) sc->Release();
    if (f2) f2->Release();
    fgmock_set_factory(nullptr);
}

}  // namespace

int main() {
    TestBindingData();
    TestGuidAndLayout();
    TestChainWalk();
    TestOptionsCodec();
    TestStateCodec();
    TestDescribeCodec();
    TestTagsAndConstantsCodec();
    GameModule game;
    CHECK(game.load(L"fg_mock_game.dll"));
    if (game.h) {
        TestOffGate(game);
        TestNotTaken(game);
        TestCp2077Session(game);
        TestOffGate(game);  // again after a full session: nothing left behind
    }
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
