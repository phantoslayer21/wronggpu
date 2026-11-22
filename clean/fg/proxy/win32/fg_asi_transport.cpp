// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the Win32 hook transport. See fg_asi_transport.h.
// Written for this project.
#include "fg_asi_transport.h"

#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2  // K32EnumProcessModules from kernel32: no extra library
#endif
#include <psapi.h>
#include <string.h>
#include <d3d12.h>

#include <exception>

// Test-only mutants (CMakeLists.txt builds the Win32 suite against each; every one must fail):
//  1 install-off gate ignored  2 a forwarded call loses an argument  3 scan budget ignored  7 restore leaves slots
#ifndef WG_FG_WIN32_MUTANT
#define WG_FG_WIN32_MUTANT 0
#endif
// Test-only mutants of the title-binding suite (fg_sl214_codec.cpp has the others): 6 the factory's swap-chain hook is
// never installed  7 the present bridge never hands the interval to the core
#ifndef WG_FG_CP2077_MUTANT
#define WG_FG_CP2077_MUTANT 0
#endif

namespace wgfgp {
namespace win32 {

namespace g = wgrr::guard;

// ---- what the handlers route through -------------------------------------------------------------------------------

struct FgAsiTransport::Impl {
    FgProxy* core = nullptr;
    IFgNativeCodec* codec = nullptr;
    uint32_t fg_feature = WG_FEAT_FRAME_GENERATION;
    std::string fn_options, fn_state;
    IFgPresentCallback* presenter = nullptr;  // behind the present bridge (null: the bridge presents itself)
    FgPresentRouteFn present_route = nullptr;
    void* present_route_ctx = nullptr;
    FgFactoryNoteFn factory_note = nullptr;
    void* factory_note_ctx = nullptr;
    FgPresentOwners* owners = nullptr;
    FgLogFn log = nullptr;
    void* log_ctx = nullptr;
    std::atomic<bool> hooked_factory{false};  // this transport installed the swap-chain creation hook
};

namespace {

constexpr unsigned kMaxModules = 1024;   // as the RR hook's load scan
constexpr unsigned kMaxDescriptors = 4096;
constexpr uint32_t kMaxDecodedTags = 4096;

std::atomic<FgAsiTransport*> g_owner{nullptr};       // the active transport (one per process)
std::atomic<FgAsiTransport::Impl*> g_route{nullptr};  // null: every handler forwards to the genuine export
// the roles, then kFgSlotCreateFactory2; kept after restore: a handler entered late still forwards
std::atomic<void*> g_genuine[kFgRoleCount + 1] = {};
std::atomic<void*> g_genuine_options{nullptr}, g_genuine_state{nullptr};  // resolved lazily through get-function
std::atomic<uint32_t> g_fg_feature{WG_FEAT_FRAME_GENERATION};
std::atomic<long> g_in_flight{0};

struct Counters {
    std::atomic<uint64_t> calls[kFgRoleCount] = {};
    std::atomic<uint64_t> forwarded[kFgRoleCount] = {};
    std::atomic<uint64_t> fg_answered{0}, option_calls{0}, state_calls{0}, codec_failures{0}, scan_entries{0};
};
Counters g_ctr;

// the present bridge's view (FgPresentBridgeStats)
struct BridgeCounters {
    std::atomic<uint64_t> factories{0}, factory_hooked{0}, chains_created{0}, chains_attached{0}, submits{0},
        generate{0}, real_only{0}, forwarded{0}, refused{0}, failures{0};
    std::mutex mu;
    WgFgPresentResult last = {};
    WgFgPresentDesc last_desc = {};
};
BridgeCounters g_bridge_ctr;

void ResetBridgeCounters() {
    g_bridge_ctr.factories = 0;
    g_bridge_ctr.factory_hooked = 0;
    g_bridge_ctr.chains_created = 0;
    g_bridge_ctr.chains_attached = 0;
    g_bridge_ctr.submits = 0;
    g_bridge_ctr.generate = 0;
    g_bridge_ctr.real_only = 0;
    g_bridge_ctr.forwarded = 0;
    g_bridge_ctr.refused = 0;
    g_bridge_ctr.failures = 0;
    std::lock_guard<std::mutex> lock(g_bridge_ctr.mu);
    g_bridge_ctr.last = WgFgPresentResult{};
    g_bridge_ctr.last_desc = WgFgPresentDesc{};
}

void ResetCounters() {
    for (uint32_t i = 0; i < kFgRoleCount; ++i) {
        g_ctr.calls[i] = 0;
        g_ctr.forwarded[i] = 0;
    }
    g_ctr.fg_answered = 0;
    g_ctr.option_calls = 0;
    g_ctr.state_calls = 0;
    g_ctr.codec_failures = 0;
    g_ctr.scan_entries = 0;
}

// The native arguments of the call being routed, for the chain trampolines (the core's chain carries only part of them).
struct CallFrame {
    uint32_t role = 0;
    const WgViewport* vp = nullptr;
    const WgFrameToken* token = nullptr;
    void* cmd = nullptr;
    const void* aux = nullptr;   // the state call's options argument
    bool forwarded = false;      // a trampoline called the genuine export
    WgResultCode fwd_result = WG_OK;
};
thread_local CallFrame* t_frame = nullptr;

class FrameScope {
public:
    explicit FrameScope(CallFrame* f) : prev_(t_frame) { t_frame = f; }
    ~FrameScope() { t_frame = prev_; }
    FrameScope(const FrameScope&) = delete;
    FrameScope& operator=(const FrameScope&) = delete;

private:
    CallFrame* prev_;
};

class InFlight {
public:
    InFlight() { g_in_flight.fetch_add(1); }
    ~InFlight() { g_in_flight.fetch_sub(1); }
    InFlight(const InFlight&) = delete;
    InFlight& operator=(const InFlight&) = delete;
};

template <typename F>
F Genuine(uint32_t role) {
    return reinterpret_cast<F>(g_genuine[role].load());
}

void NoteForward(uint32_t role, WgResultCode r) {
    g_ctr.forwarded[role].fetch_add(1);
    if (t_frame) {
        t_frame->forwarded = true;
        t_frame->fwd_result = r;
    }
}

// ---- forwarding (the genuine export, the game's own arguments) -------------------------------------------------------

WgResultCode FwdIsSupported(uint32_t f, const WgAdapterInfo* a) {
    auto fn = Genuine<FgIsSupportedFn>(kFgRoleIsSupported);
    const WgResultCode r = fn ? fn(f, a) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleIsSupported, r);
    return r;
}
WgResultCode FwdRequirements(uint32_t f, WgRequirements* q) {
    auto fn = Genuine<FgRequirementsFn>(kFgRoleRequirements);
    const WgResultCode r = fn ? fn(f, q) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleRequirements, r);
    return r;
}
WgResultCode FwdGetFunction(uint32_t f, const char* n, void** o) {
    auto fn = Genuine<FgGetFunctionFn>(kFgRoleGetFunction);
    const WgResultCode r = fn ? fn(f, n, o) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleGetFunction, r);
    return r;
}
WgResultCode FwdTag(const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
    auto fn = Genuine<FgTagFn>(kFgRoleTag);
#if WG_FG_WIN32_MUTANT == 2
    c = nullptr;  // test-only mutant: a forwarded call loses an argument
#endif
    const WgResultCode r = fn ? fn(v, t, n, c) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleTag, r);
    return r;
}
WgResultCode FwdTagForFrame(const WgFrameToken* k, const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
    auto fn = Genuine<FgTagForFrameFn>(kFgRoleTagForFrame);
    const WgResultCode r = fn ? fn(k, v, t, n, c) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleTagForFrame, r);
    return r;
}
WgResultCode FwdConstants(const WgConstants* c, const WgFrameToken* k, const WgViewport* v) {
    auto fn = Genuine<FgConstantsFn>(kFgRoleConstants);
    const WgResultCode r = fn ? fn(c, k, v) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleConstants, r);
    return r;
}
WgResultCode FwdEvaluate(uint32_t f, const WgFrameToken* k, const WgHeader* const* in, uint32_t n, void* c) {
    auto fn = Genuine<FgEvaluateFn>(kFgRoleEvaluate);
    const WgResultCode r = fn ? fn(f, k, in, n, c) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleEvaluate, r);
    return r;
}
WgResultCode FwdFree(uint32_t f, const WgViewport* v) {
    auto fn = Genuine<FgFreeFn>(kFgRoleFree);
    const WgResultCode r = fn ? fn(f, v) : WG_E_MISSING_OR_INVALID_API;
    NoteForward(kFgRoleFree, r);
    return r;
}

// The genuine FG feature function `name`, fetched once through the genuine get-function (never through ours).
void* GenuineFeatureFunction(std::atomic<void*>& cache, const std::string& name) {
    void* p = cache.load();
    if (p) return p;
    auto get = Genuine<FgGetFunctionFn>(kFgRoleGetFunction);
    if (!get || name.empty()) return nullptr;
    void* out = nullptr;
    if (get(g_fg_feature.load(), name.c_str(), &out) != WG_OK || !out) return nullptr;
    cache.store(out);
    return out;
}

// Fallback names when no route is active (a game kept our feature-function pointer past restore).
std::string g_last_fn_options, g_last_fn_state;  // written only by install() before routing starts

WgResultCode FwdOptions(const WgViewport* v, const void* o, const std::string& name) {
    auto fn = reinterpret_cast<FgOptionsFn>(GenuineFeatureFunction(g_genuine_options, name));
    const WgResultCode r = fn ? fn(v, o) : WG_E_FEATURE_MISSING;
    if (t_frame) {
        t_frame->forwarded = true;
        t_frame->fwd_result = r;
    }
    return r;
}
WgResultCode FwdState(const WgViewport* v, void* s, const void* o, const std::string& name) {
    auto fn = reinterpret_cast<FgStateFn>(GenuineFeatureFunction(g_genuine_state, name));
    const WgResultCode r = fn ? fn(v, s, o) : WG_E_FEATURE_MISSING;
    if (t_frame) {
        t_frame->forwarded = true;
        t_frame->fwd_result = r;
    }
    return r;
}

// ---- the core's next link (WgFgChainV1): back to the genuine export with the frame's full arguments -----------------

int32_t ChainIsSupported(void*, uint32_t feature, const void* native_adapter) {
    return FwdIsSupported(feature, static_cast<const WgAdapterInfo*>(native_adapter));
}
int32_t ChainGetFunction(void*, uint32_t feature, const char* name, void** out) {
    return FwdGetFunction(feature, name, out);
}
int32_t ChainSetOptions(void*, uint32_t, uint32_t, const void* native_options) {
    FgAsiTransport::Impl* r = g_route.load();
    return FwdOptions(t_frame ? t_frame->vp : nullptr, native_options, r ? r->fn_options : g_last_fn_options);
}
int32_t ChainGetState(void*, uint32_t, uint32_t, void* native_state) {
    FgAsiTransport::Impl* r = g_route.load();
    return FwdState(t_frame ? t_frame->vp : nullptr, native_state, t_frame ? t_frame->aux : nullptr,
                    r ? r->fn_state : g_last_fn_state);
}
int32_t ChainTag(void*, const void* native_tags, uint32_t count) {
    const WgTag* t = static_cast<const WgTag*>(native_tags);
    if (t_frame && t_frame->role == kFgRoleTagForFrame)
        return FwdTagForFrame(t_frame->token, t_frame->vp, t, count, t_frame->cmd);
    return FwdTag(t_frame ? t_frame->vp : nullptr, t, count, t_frame ? t_frame->cmd : nullptr);
}
int32_t ChainConstants(void*, const void* native_constants) {
    return FwdConstants(static_cast<const WgConstants*>(native_constants), t_frame ? t_frame->token : nullptr,
                        t_frame ? t_frame->vp : nullptr);
}
// A forwarded present: the swap-chain hook presents the game image itself (fg_swapchain_hook.h); nothing to do here.
int32_t ChainPresent(void*, const WgFgPresentDesc*) { return WG_OK; }

bool DecodeLuid(const WgAdapterInfo* a, WgFgLuid* out) {
    if (!a || !a->luid || a->luid_size < sizeof(WgFgLuid) || a->vk_physical_device) return false;
    memcpy(out, a->luid, sizeof(WgFgLuid));
    return true;
}

uint32_t ViewportId(const WgViewport* v) { return v ? v->id : 0u; }

// ---- handlers (the game's signatures; written into the import slots) ------------------------------------------------
// Each: count, route (or forward when no route is active), and on any exception forward unchanged unless the genuine
// export was already reached for this call (then its result stands: a call is never forwarded twice).

WgResultCode H_IsSupported(uint32_t feature, const WgAdapterInfo* a) {
    InFlight in;
    g_ctr.calls[kFgRoleIsSupported].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdIsSupported(feature, a);
    CallFrame f;
    f.role = kFgRoleIsSupported;
    FrameScope s(&f);
    try {
        WgFgLuid luid = {};
        const bool have = DecodeLuid(a, &luid);
        const WgResultCode res = r->core->is_supported(feature, have ? &luid : nullptr, a);
        if (!f.forwarded) g_ctr.fg_answered.fetch_add(1);
        return res;
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : FwdIsSupported(feature, a);
    }
}

WgResultCode H_Requirements(uint32_t feature, WgRequirements* q) {
    InFlight in;
    g_ctr.calls[kFgRoleRequirements].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdRequirements(feature, q);
    CallFrame f;
    f.role = kFgRoleRequirements;
    FrameScope s(&f);
    try {
        if (!r->core->owns_feature(feature)) return FwdRequirements(feature, q);
        g_ctr.fg_answered.fetch_add(1);
        if (!q) return WG_E_MISSING_INPUT_PARAMETER;
        // FG is ours: the genuine feature never answers (one owner per feature)
        if (!r->codec->write_requirements(r->core->capability(), q)) {
            g_ctr.codec_failures.fetch_add(1);
            return WG_E_INVALID_PARAMETER;
        }
        return WG_OK;
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : WG_E_INVALID_PARAMETER;
    }
}

WgResultCode H_GetFunction(uint32_t feature, const char* name, void** out) {
    InFlight in;
    g_ctr.calls[kFgRoleGetFunction].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdGetFunction(feature, name, out);
    CallFrame f;
    f.role = kFgRoleGetFunction;
    FrameScope s(&f);
    try {
        const WgResultCode res = r->core->get_function(feature, name, out);
        if (!f.forwarded) g_ctr.fg_answered.fetch_add(1);
        return res;
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : FwdGetFunction(feature, name, out);
    }
}

// Tags: decoded only while FG is taken (the core only observes them then); always forwarded by the core.
WgResultCode RouteTags(FgAsiTransport::Impl* r, CallFrame& f, const WgTag* t, uint32_t n) {
    std::vector<WgFgTag> decoded;
    const WgFgTag* dp = nullptr;
    if (t && n && n <= kMaxDecodedTags && r->core->fg_taken()) {
        decoded.assign(n, WgFgTag{});
        for (auto& d : decoded) {
            d.struct_size = sizeof(WgFgTag);
            d.native_type = WG_FG_NATIVE_TYPE_UNBOUND;
        }
        if (r->codec->tags(f.token, f.vp, t, n, decoded.data())) dp = decoded.data();
        else g_ctr.codec_failures.fetch_add(1);
    }
    return r->core->tag(dp, n, t);
}

WgResultCode H_Tag(const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
    InFlight in;
    g_ctr.calls[kFgRoleTag].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdTag(v, t, n, c);
    CallFrame f;
    f.role = kFgRoleTag;
    f.vp = v;
    f.cmd = c;
    FrameScope s(&f);
    try {
        return RouteTags(r, f, t, n);
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : FwdTag(v, t, n, c);
    }
}

WgResultCode H_TagForFrame(const WgFrameToken* k, const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
    InFlight in;
    g_ctr.calls[kFgRoleTagForFrame].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdTagForFrame(k, v, t, n, c);
    CallFrame f;
    f.role = kFgRoleTagForFrame;
    f.vp = v;
    f.token = k;
    f.cmd = c;
    FrameScope s(&f);
    try {
        return RouteTags(r, f, t, n);
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : FwdTagForFrame(k, v, t, n, c);
    }
}

WgResultCode H_Constants(const WgConstants* c, const WgFrameToken* k, const WgViewport* v) {
    InFlight in;
    g_ctr.calls[kFgRoleConstants].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdConstants(c, k, v);
    CallFrame f;
    f.role = kFgRoleConstants;
    f.vp = v;
    f.token = k;
    FrameScope s(&f);
    try {
        WgFgConstants d = {};
        const WgFgConstants* dp = nullptr;
        if (c && r->core->fg_taken()) {
            d.struct_size = sizeof(d);
            d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            if (r->codec->constants(c, k, v, &d)) dp = &d;
            else g_ctr.codec_failures.fetch_add(1);
        }
        return r->core->constants(dp, c);
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : FwdConstants(c, k, v);
    }
}

// FG is present-driven on this entry: generation runs once per game present through the chain's present owner, so an
// FG evaluate (and an FG free) while FG is taken is acknowledged here and never reaches the genuine FG feature.
WgResultCode H_Evaluate(uint32_t feature, const WgFrameToken* k, const WgHeader* const* in_, uint32_t n, void* c) {
    InFlight in;
    g_ctr.calls[kFgRoleEvaluate].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    try {
        if (r && r->core->owns_feature(feature)) {
            g_ctr.fg_answered.fetch_add(1);
            return WG_OK;
        }
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
    }
    return FwdEvaluate(feature, k, in_, n, c);
}

WgResultCode H_Free(uint32_t feature, const WgViewport* v) {
    InFlight in;
    g_ctr.calls[kFgRoleFree].fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    try {
        if (r && r->core->owns_feature(feature)) {
            g_ctr.fg_answered.fetch_add(1);
            return WG_OK;
        }
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
    }
    return FwdFree(feature, v);
}

// The FG feature functions handed out by the core's get-function while FG is taken.
WgResultCode H_FgOptions(const WgViewport* v, const void* native) {
    InFlight in;
    g_ctr.option_calls.fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdOptions(v, native, g_last_fn_options);
    CallFrame f;
    f.role = kFgRoleGetFunction;
    f.vp = v;
    FrameScope s(&f);
    try {
        if (r->core->owns_feature(r->fg_feature)) {
            // the interface's own checks, in its order, before anything is decoded (FG is ours: one owner)
            const WgResultCode chk = r->codec->check_options(v, native, r->core->advertised_max_generated());
            if (chk != WG_OK) {
                g_ctr.fg_answered.fetch_add(1);
                return chk;
            }
        }
        WgFgOptions d = {};
        const WgFgOptions* dp = nullptr;
        if (native) {
            d.struct_size = sizeof(d);
            d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            if (r->codec->options(v, native, &d)) dp = &d;
            else g_ctr.codec_failures.fetch_add(1);
        }
        return r->core->set_options(r->fg_feature, ViewportId(v), dp, native);
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : WG_E_INVALID_PARAMETER;
    }
}

WgResultCode H_FgState(const WgViewport* v, void* native, const void* native_options) {
    InFlight in;
    g_ctr.state_calls.fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdState(v, native, native_options, g_last_fn_state);
    CallFrame f;
    f.role = kFgRoleGetFunction;
    f.vp = v;
    f.aux = native_options;
    FrameScope s(&f);
    try {
        WgFgState st = {};
        st.struct_size = sizeof(st);
        st.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        const WgResultCode res = r->core->get_state(r->fg_feature, ViewportId(v), &st, native);
        if (f.forwarded || res != WG_OK) return res;
        if (!native) return WG_E_MISSING_INPUT_PARAMETER;
        if (!r->codec->write_state_for(st, native_options, native)) {
            g_ctr.codec_failures.fetch_add(1);
            return WG_E_INVALID_PARAMETER;
        }
        return WG_OK;
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : WG_E_INVALID_PARAMETER;
    }
}

// ---- swap chain: the present bridge, the creation sink and the factory handler --------------------------------------
// Process-lifetime objects (a chain or a creation call that outlives the transport never sees a freed object); every
// entry reads the active route and does nothing of its own while there is none.

// The callback a claimed chain presents through: the game interval goes to the core first (the backbuffer as colour
// input, the codec's frame token), then the presenter (or, without one, the game image straight to the real chain).
class PresentBridge final : public IFgPresentCallback {
public:
    void BeginInterval(const FgPresentCall& call) override {
        InFlight in;
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->BeginInterval(call);
    }
    HRESULT PresentInterval(const FgPresentCall& call, IFgInnerPresenter& inner) override {
        InFlight in;
        FgAsiTransport::Impl* r = g_route.load();
        // once per interval: a retry never submits again
        if (r && !call.retry && WG_FG_CP2077_MUTANT != 7) Submit(*r, call, inner);
        if (r && r->presenter) return r->presenter->PresentInterval(call, inner);
        return call.present1 ? inner.Present1(call.sync_interval, call.flags, call.params)
                             : inner.Present(call.sync_interval, call.flags);
    }
    void BeforeResize() override {
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->BeforeResize();
    }
    void AfterResize(HRESULT hr) override {
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->AfterResize(hr);
    }
    void BeforeFullscreen(BOOL fs) override {
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->BeforeFullscreen(fs);
    }
    void AfterFullscreen(BOOL fs, HRESULT hr) override {
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->AfterFullscreen(fs, hr);
    }
    void ColorSpaceChanged(DXGI_COLOR_SPACE_TYPE cs) override {
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->ColorSpaceChanged(cs);
    }
    bool GetBuffer(UINT b, REFIID riid, void** out, HRESULT* hr) override {
        FgAsiTransport::Impl* r = g_route.load();
        return r && r->presenter && r->presenter->GetBuffer(b, riid, out, hr);
    }
    bool CurrentBackBufferIndex(UINT* i) override {
        FgAsiTransport::Impl* r = g_route.load();
        return r && r->presenter && r->presenter->CurrentBackBufferIndex(i);
    }
    bool FrameLatencyWaitableObject(HANDLE* h) override {
        FgAsiTransport::Impl* r = g_route.load();
        return r && r->presenter && r->presenter->FrameLatencyWaitableObject(h);
    }
    void Detached() override {
        FgAsiTransport::Impl* r = g_route.load();
        if (r && r->presenter) r->presenter->Detached();
    }

private:
    static void Submit(FgAsiTransport::Impl& r, const FgPresentCall& call, IFgInnerPresenter& inner) {
        g_bridge_ctr.submits.fetch_add(1);
        try {
            WgFgPresentDesc d = {};
            d.struct_size = sizeof(d);
            d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
            uint32_t vp = 0;
            if (r.core->session_viewport(&vp)) d.viewport_id = vp;
            d.frame_token = r.codec->frame_for_present();
            d.swapchain_id = call.swapchain_id;
            d.swapchain_epoch = call.swapchain_epoch;
            d.queue_epoch = 0;
            d.luid = r.core->capability().adapter.luid;
            IDXGISwapChain4* sc = inner.Inner();
            if (sc) {
                HWND hwnd = nullptr;
                if (SUCCEEDED(sc->GetHwnd(&hwnd))) d.window_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(hwnd));
                DXGI_SWAP_CHAIN_DESC1 sd = {};
                if (SUCCEEDED(sc->GetDesc1(&sd))) {
                    d.backbuffer.format = static_cast<uint32_t>(sd.Format);
                    d.backbuffer.width = sd.Width;
                    d.backbuffer.height = sd.Height;
                    d.backbuffer.rect = WgFgRect{0, 0, sd.Width, sd.Height};
                }
                // the buffer being presented (the chain keeps its own reference; the core only reads the pointer); a
                // presenter that serves the game-facing buffers (a facade) names the one the game rendered
                void* buf = nullptr;
                UINT fi = 0;
                HRESULT fhr = E_FAIL;
                if (r.presenter && r.presenter->CurrentBackBufferIndex(&fi) &&
                    r.presenter->GetBuffer(fi, __uuidof(ID3D12Resource), &buf, &fhr) && SUCCEEDED(fhr) && buf) {
                    d.backbuffer.resource = buf;
                    static_cast<IUnknown*>(buf)->Release();
                } else if (SUCCEEDED(sc->GetBuffer(sc->GetCurrentBackBufferIndex(), __uuidof(ID3D12Resource), &buf)) &&
                           buf) {
                    d.backbuffer.resource = buf;
                    static_cast<IUnknown*>(buf)->Release();
                }
            }
            d.backbuffer.state = 0;  // D3D12_RESOURCE_STATE_PRESENT
            d.backbuffer.frame_token = d.frame_token;
            d.backbuffer.viewport_id = d.viewport_id;
            WgFgPresentResult res = {};
            res.struct_size = sizeof(res);
            res.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            if (r.present_route)
                r.present_route(r.present_route_ctx, r.core, &d, &res);
            else
                r.core->present(&d, &res);
            r.codec->note_present();
            switch (res.status) {
                case WG_FG_PRESENT_GENERATE: g_bridge_ctr.generate.fetch_add(1); break;
                case WG_FG_PRESENT_REAL_ONLY: g_bridge_ctr.real_only.fetch_add(1); break;
                case WG_FG_PRESENT_FORWARDED: g_bridge_ctr.forwarded.fetch_add(1); break;
                default: g_bridge_ctr.refused.fetch_add(1); break;
            }
            std::lock_guard<std::mutex> lock(g_bridge_ctr.mu);
            g_bridge_ctr.last = res;
            g_bridge_ctr.last_desc = d;
        } catch (...) {
            g_bridge_ctr.failures.fetch_add(1);  // the interval is still presented below
        }
    }
};

PresentBridge g_present_bridge;

// Each chain created through the hooked factory: claimed for the bridge while FG is taken, else a pass-through.
class ChainSink final : public IFgSwapChainSink {
public:
    void Created(FgSwapChain* chain, HWND, const DXGI_SWAP_CHAIN_DESC1*) override {
        InFlight in;
        g_bridge_ctr.chains_created.fetch_add(1);
        FgAsiTransport::Impl* r = g_route.load();
        if (!r || !chain) return;
        if (chain->Attach(&g_present_bridge, r->core, r->owners, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_NONE) {
            g_bridge_ctr.chains_attached.fetch_add(1);
            if (r->presenter) r->presenter->Attached(chain->Inner(), chain->Queue());
        }
    }
    UINT ExtraCreateFlags(HWND, const DXGI_SWAP_CHAIN_DESC1& desc) override {
        InFlight in;
        FgAsiTransport::Impl* r = g_route.load();
        // only while the chain would be claimed (FG taken): a pass-through chain is created exactly as the game asked
        if (!r || !r->presenter || !r->core || !r->core->fg_taken()) return 0;
        return r->presenter->ExtraSwapChainFlags(desc);
    }
};

ChainSink g_chain_sink;

// The factory's swap-chain creation hook, once per process (another one already installed: left alone).
void HookFactory(FgAsiTransport::Impl& r, IUnknown* unknown) {
#if WG_FG_CP2077_MUTANT == 6
    (void)r;
    (void)unknown;
    return;
#else
    if (FgSwapChainHook::Installed()) return;
    IDXGIFactory2* f2 = nullptr;
    if (FAILED(unknown->QueryInterface(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&f2))) || !f2) return;
    if (FgSwapChainHook::Install(f2, &g_chain_sink, r.log, r.log_ctx)) {
        r.hooked_factory.store(true);
        g_bridge_ctr.factory_hooked = 1;
    }
    f2->Release();
#endif
}

// The game's factory creation through the interface module: the genuine export first, its result unchanged.
HRESULT WINAPI H_CreateFactory2(UINT flags, REFIID riid, void** out) {
    InFlight in;
    g_bridge_ctr.factories.fetch_add(1);
    auto fn = reinterpret_cast<FgCreateFactory2Fn>(g_genuine[kFgSlotCreateFactory2].load());
    const HRESULT hr = fn ? fn(flags, riid, out) : E_NOINTERFACE;
    FgAsiTransport::Impl* r = g_route.load();
    if (FAILED(hr) || !r || !out || !*out) return hr;
    try {
        HookFactory(*r, static_cast<IUnknown*>(*out));
        if (r->factory_note) r->factory_note(r->factory_note_ctx, r->core, static_cast<IUnknown*>(*out));
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
    }
    return hr;
}

// ---- patching -------------------------------------------------------------------------------------------------------

// One import slot, atomically, the page protection restored (as the RR hook's patch_slot).
bool PatchSlot(void** slot, void* value) {
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
    InterlockedExchangePointer(slot, value);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(void*), old, &ignored);
    return true;
}

std::wstring Widen(const std::string& s) {
    std::wstring w;
    for (char c : s) w.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    return w;
}

std::string Narrow(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) s.push_back(c < 128 ? static_cast<char>(c) : '?');
    return s;
}

std::string ModuleLeaf(HMODULE m) {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(m, buf, MAX_PATH);
    if (!n || n >= MAX_PATH) return "?";
    std::wstring w(buf, n);
    const size_t slash = w.find_last_of(L"\\/");
    return Narrow(slash == std::wstring::npos ? w : w.substr(slash + 1));
}

HMODULE SelfModule() {
    HMODULE m = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&H_IsSupported), &m);
    return m;
}

// The delay-load descriptors of `module` (bounded as the import walk): f(descriptor) until it returns false.
template <typename F>
void ForEachDelayDescriptor(const g::PeView& pe, g::Reader& r, g::Budget& b, F&& f) {
    uint32_t rva = 0, size = 0;
    if (!pe.dir(IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT, &rva, &size)) return;
    uint32_t count = size / sizeof(IMAGE_DELAYLOAD_DESCRIPTOR);
    if (count > kMaxDescriptors) count = kMaxDescriptors;
    for (uint32_t k = 0; k < count && b.step(); ++k) {
        const uint8_t* dp = pe.at(rva + (uint64_t)k * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR), sizeof(IMAGE_DELAYLOAD_DESCRIPTOR));
        IMAGE_DELAYLOAD_DESCRIPTOR d;
        if (!dp || !r.get(dp, &d) || !d.DllNameRVA) return;
        if (!d.Attributes.RvaBased) continue;  // pre-VC7 VA-based descriptors: not produced by any supported game
        if (!f(d)) return;
    }
}

bool DelayImports(const void* module, const char* leaf, g::Budget& b) {
    g::Reader r;
    g::PeView pe;
    if (!pe.open(module, r)) return false;
    bool found = false;
    ForEachDelayDescriptor(pe, r, b, [&](const IMAGE_DELAYLOAD_DESCRIPTOR& d) {
        char dll[64];
        if (pe.str(d.DllNameRVA, dll, sizeof dll, r) && _stricmp(dll, leaf) == 0) found = true;
        return !found;
    });
    return found;
}

// The game's own interface file next to the exe, for an exe that delay-loads it (HOOK-MODE.md step 2).
HMODULE LoadNextToExe(const std::string& leaf) {
    wchar_t exe[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (!n || n >= MAX_PATH) return nullptr;
    std::wstring path(exe, n);
    const size_t slash = path.find_last_of(L"\\/");
    path = (slash == std::wstring::npos ? std::wstring() : path.substr(0, slash + 1)) + Widen(leaf);
    if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) return nullptr;
    return LoadLibraryW(path.c_str());
}

}  // namespace

// ---- public helpers -------------------------------------------------------------------------------------------------

const char* FgRoleWord(uint32_t role) {
    static const char* const kWords[kFgRoleCount] = {"is-supported", "requirements", "get-function", "tag",
                                                     "tag-for-frame", "constants", "evaluate", "free"};
    if (role == kFgSlotCreateFactory2) return "create-factory";
    return role < kFgRoleCount ? kWords[role] : "?";
}

const char* FgInstallResultWord(uint32_t r) {
    switch (r) {
        case kFgInstalled: return "installed";
        case kFgInstallOff: return "fg=off (nothing read, nothing patched)";
        case kFgNoBinding: return "no qualified title binding (nothing patched)";
        case kFgInterfaceNotFound: return "game interface module not found (nothing patched)";
        case kFgVersionUnreadable: return "game interface version unreadable (nothing patched)";
        case kFgVersionMismatch: return "game interface version is not the binding's (nothing patched)";
        case kFgAttachRefused: return "adapter core refused the chain (nothing patched)";
        case kFgScanOverBudget: return "scan over its budget (nothing patched)";
        case kFgNothingToPatch: return "no importer names a hooked role (nothing patched)";
        case kFgAlreadyActive: return "another transport is active (nothing changed)";
        case kFgNullArgument: return "null argument";
        case kFgRuntimeMissing: return "the FG model or kernels are missing or invalid (nothing patched)";
        default: return "?";
    }
}

bool FgModuleFileVersion(HMODULE m, uint32_t* major, uint32_t* minor) {
    if (!m || !major || !minor) return false;
    wchar_t path[MAX_PATH * 2];
    const DWORD pn = GetModuleFileNameW(m, path, MAX_PATH * 2);
    if (!pn || pn >= MAX_PATH * 2) return false;
    DWORD ignored = 0;
    const DWORD size = GetFileVersionInfoSizeW(path, &ignored);
    if (!size) return false;
    std::vector<uint8_t> blk(size);
    if (!GetFileVersionInfoW(path, 0, size, blk.data())) return false;
    VS_FIXEDFILEINFO* f = nullptr;
    UINT n = 0;
    if (!VerQueryValueW(blk.data(), L"\\", reinterpret_cast<void**>(&f), &n) || !f || n < sizeof(VS_FIXEDFILEINFO))
        return false;
    if (f->dwSignature != VS_FFI_SIGNATURE) return false;
    *major = HIWORD(f->dwFileVersionMS);
    *minor = LOWORD(f->dwFileVersionMS);
    return true;
}

std::vector<FgSlot> FgCollectSlots(const void* module, const char* leaf, const std::vector<std::string>& names,
                                   g::Budget& budget) {
    std::vector<FgSlot> out;
    if (!module || !leaf) return out;
    bool delay = false;
    auto take = [&](const char* name, void** slot) {
        for (uint32_t i = 0; i < names.size(); ++i)
            if (!names[i].empty() && names[i] == name) out.push_back({slot, i, delay});
    };
    g::Reader r;
    g::import_slots(module, leaf, r, budget, take);
    g::PeView pe;
    if (budget.over() || !pe.open(module, r)) return out;
    delay = true;
    ForEachDelayDescriptor(pe, r, budget, [&](const IMAGE_DELAYLOAD_DESCRIPTOR& d) {
        char dll[64];
        if (pe.str(d.DllNameRVA, dll, sizeof dll, r) && _stricmp(dll, leaf) == 0)
            g::thunks(pe, r, budget, d.ImportNameTableRVA, d.ImportAddressTableRVA, take);
        return !budget.over();
    });
    return out;
}

// ---- the transport --------------------------------------------------------------------------------------------------

FgAsiTransport::FgAsiTransport() = default;
FgAsiTransport::~FgAsiTransport() { restore(); }

void FgAsiTransport::log(const std::string& line) const {
    if (log_fn_) log_fn_(log_ctx_, line.c_str());
}

uint32_t FgAsiTransport::finish(uint32_t result, const std::string& line) {
    result_ = result;
    log("fg transport: " + std::string(FgInstallResultWord(result)) + (line.empty() ? "" : ": " + line));
    return result;
}

void* FgAsiTransport::handler_for(uint32_t role) {
    switch (role) {
        case kFgRoleIsSupported: return reinterpret_cast<void*>(&H_IsSupported);
        case kFgRoleRequirements: return reinterpret_cast<void*>(&H_Requirements);
        case kFgRoleGetFunction: return reinterpret_cast<void*>(&H_GetFunction);
        case kFgRoleTag: return reinterpret_cast<void*>(&H_Tag);
        case kFgRoleTagForFrame: return reinterpret_cast<void*>(&H_TagForFrame);
        case kFgRoleConstants: return reinterpret_cast<void*>(&H_Constants);
        case kFgRoleEvaluate: return reinterpret_cast<void*>(&H_Evaluate);
        case kFgRoleFree: return reinterpret_cast<void*>(&H_Free);
        case kFgSlotCreateFactory2: return reinterpret_cast<void*>(&H_CreateFactory2);
        default: return nullptr;
    }
}
IFgSwapChainSink* FgAsiTransport::swapchain_sink() { return &g_chain_sink; }
IFgPresentCallback* FgAsiTransport::present_bridge() { return &g_present_bridge; }

FgPresentBridgeStats FgAsiTransport::present_stats() const {
    FgPresentBridgeStats s;
    s.factories = g_bridge_ctr.factories.load();
    s.factory_hooked = g_bridge_ctr.factory_hooked.load();
    s.chains_created = g_bridge_ctr.chains_created.load();
    s.chains_attached = g_bridge_ctr.chains_attached.load();
    s.submits = g_bridge_ctr.submits.load();
    s.generate = g_bridge_ctr.generate.load();
    s.real_only = g_bridge_ctr.real_only.load();
    s.forwarded = g_bridge_ctr.forwarded.load();
    s.refused = g_bridge_ctr.refused.load();
    s.failures = g_bridge_ctr.failures.load();
    std::lock_guard<std::mutex> lock(g_bridge_ctr.mu);
    s.last = g_bridge_ctr.last;
    s.last_desc = g_bridge_ctr.last_desc;
    return s;
}
void* FgAsiTransport::options_handler() { return reinterpret_cast<void*>(&H_FgOptions); }
void* FgAsiTransport::state_handler() { return reinterpret_cast<void*>(&H_FgState); }

void* FgAsiTransport::genuine(uint32_t role) const {
    return role <= kFgSlotCreateFactory2 ? g_genuine[role].load() : nullptr;
}

FgTransportStats FgAsiTransport::stats() const {
    FgTransportStats s;
    for (uint32_t i = 0; i < kFgRoleCount; ++i) {
        s.calls[i] = g_ctr.calls[i].load();
        s.forwarded[i] = g_ctr.forwarded[i].load();
    }
    s.fg_answered = g_ctr.fg_answered.load();
    s.option_calls = g_ctr.option_calls.load();
    s.state_calls = g_ctr.state_calls.load();
    s.codec_failures = g_ctr.codec_failures.load();
    s.scan_entries = g_ctr.scan_entries.load();
    return s;
}

uint32_t FgAsiTransport::install(const FgAsiConfig& cfg) {
    try {
        log_fn_ = cfg.log;
        log_ctx_ = cfg.log_ctx;
        if (active_) return finish(kFgAlreadyActive, "this transport is already installed");
        // INSTALL-OFF GATE: before anything is read from the process
        if (!cfg.settings.enabled && WG_FG_WIN32_MUTANT != 1) return finish(kFgInstallOff, "");
        if (!cfg.imports.qualified || !cfg.core.binding.qualified || !cfg.codec || cfg.imports.interposer_leaf.empty())
            return finish(kFgNoBinding, "");
        FgAsiTransport* expected = nullptr;
        if (!g_owner.compare_exchange_strong(expected, this)) return finish(kFgAlreadyActive, "");

        // the claim is released on every refusal below
        struct OwnerRelease {
            FgAsiTransport* self;
            bool keep = false;
            ~OwnerRelease() {
                if (!keep) {
                    FgAsiTransport* me = self;
                    g_owner.compare_exchange_strong(me, nullptr);
                }
            }
        } owner_release{this};

        // find the interface module (already loaded, or the game's own file for an exe that delay-loads it)
        const std::string& leaf = cfg.imports.interposer_leaf;
        HMODULE iface = GetModuleHandleW(Widen(leaf).c_str());
        bool loaded = false;
        if (!iface && cfg.load_delay_linked) {
            g::Budget b(cfg.scan_max_entries, cfg.scan_max_ms);
            if (DelayImports(GetModuleHandleW(nullptr), leaf.c_str(), b)) {
                iface = LoadNextToExe(leaf);
                loaded = iface != nullptr;
            }
        }
        if (!iface) return finish(kFgInterfaceNotFound, leaf);
        iface_ = iface;
        loaded_by_us_ = loaded;
        log(std::string("fg transport: game interface ") + (loaded ? "loaded for the exe's delay-load import: "
                                                                   : "found already loaded: ") + ModuleLeaf(iface));

        // version gate
        uint32_t major = 0, minor = 0;
        const FgVersionReader rv = cfg.read_version ? cfg.read_version : &FgModuleFileVersion;
        if (!rv(iface, &major, &minor)) return finish(kFgVersionUnreadable, "");
        if (major != cfg.imports.version_major || minor != cfg.imports.version_minor)
            return finish(kFgVersionMismatch, std::to_string(major) + "." + std::to_string(minor) + ", binding " +
                                                  std::to_string(cfg.imports.version_major) + "." +
                                                  std::to_string(cfg.imports.version_minor));

        // the genuine exports, by name (the address the slots hold; a host's inline patch on them stays in the chain)
        for (uint32_t i = 0; i < kFgRoleCount; ++i) {
            const std::string& nm = cfg.imports.role_names[i];
            g_genuine[i].store(nm.empty() ? nullptr : reinterpret_cast<void*>(GetProcAddress(iface, nm.c_str())));
        }
        const std::string& fac = cfg.imports.factory2_name;
        g_genuine[kFgSlotCreateFactory2].store(fac.empty() ? nullptr
                                                           : reinterpret_cast<void*>(GetProcAddress(iface, fac.c_str())));
        ResetBridgeCounters();
        g_genuine_options.store(nullptr);
        g_genuine_state.store(nullptr);
        g_fg_feature.store(cfg.core.ids.fg);
        g_last_fn_options = cfg.imports.fn_options;
        g_last_fn_state = cfg.imports.fn_state;
        ResetCounters();

        // the core, attached to the genuine interface through the trampolines
        FgProxyConfig pc = cfg.core;
        pc.settings = cfg.settings;
        auto core = std::unique_ptr<FgProxy>(new FgProxy(pc, cfg.owners, cfg.log, cfg.log_ctx));
        WgFgChainV1 chain = {};
        chain.struct_size = sizeof(chain);
        chain.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        chain.self = this;
        chain.max_generated = cfg.imports.abi_max_generated;
        chain.is_supported = &ChainIsSupported;
        chain.get_function = &ChainGetFunction;
        chain.set_options = &ChainSetOptions;
        chain.get_state = &ChainGetState;
        chain.tag = &ChainTag;
        chain.constants = &ChainConstants;
        chain.present = &ChainPresent;
        const uint32_t ar = core->attach_interposer(&chain, major, minor);
        if (ar != WG_FG_REASON_NONE) return finish(kFgAttachRefused, "reason " + std::to_string(ar));
        if (!cfg.imports.fn_options.empty()) core->set_feature_function(cfg.imports.fn_options, options_handler());
        if (!cfg.imports.fn_state.empty()) core->set_feature_function(cfg.imports.fn_state, state_handler());

        // bounded scan of every loaded importer (the interface module and this module skipped); nothing is patched
        // until the whole scan finished inside its budget
        std::vector<std::string> names;
        for (uint32_t i = 0; i < kFgRoleCount; ++i)
            names.push_back(g_genuine[i].load() ? cfg.imports.role_names[i] : std::string());
        names.push_back(g_genuine[kFgSlotCreateFactory2].load() ? fac : std::string());  // index kFgSlotCreateFactory2
        HMODULE mods[kMaxModules];
        DWORD needed = 0;
        if (!K32EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &needed)) needed = 0;
        unsigned count = needed / sizeof(HMODULE);
        if (count > kMaxModules) count = kMaxModules;
        // this module is skipped, unless it is the exe itself (a transport linked into the exe, as in the tests)
        const HMODULE self = SelfModule() == GetModuleHandleW(nullptr) ? nullptr : SelfModule();
        g::Budget budget(cfg.scan_max_entries, cfg.scan_max_ms);
        struct Found {
            HMODULE module;
            FgSlot slot;
        };
        std::vector<Found> found;
        std::vector<HMODULE> held;  // a reference on each scanned module until the patches are written
        struct HeldRelease {
            std::vector<HMODULE>* v;
            ~HeldRelease() {
                for (HMODULE h : *v) FreeLibrary(h);
            }
        } held_release{&held};
        for (unsigned i = 0; i < count && budget.step(); ++i) {
            if (mods[i] == iface || mods[i] == self) continue;
            HMODULE h = nullptr;
            if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(mods[i]), &h) ||
                h != mods[i]) {
                if (h) FreeLibrary(h);
                continue;  // unloaded meanwhile: nothing found there
            }
            held.push_back(h);
            for (const FgSlot& s : FgCollectSlots(h, leaf.c_str(), names, budget)) found.push_back({h, s});
        }
        g_ctr.scan_entries.store(budget.entries());
        if (budget.over() && WG_FG_WIN32_MUTANT != 3)
            return finish(kFgScanOverBudget, std::to_string(budget.entries()) + " entries, " +
                                                 std::to_string(static_cast<unsigned>(budget.elapsed_ms())) + " ms");
        if (found.empty()) return finish(kFgNothingToPatch, "");

        // route first, then redirect: a slot is live the moment it is written
        impl_.reset(new Impl());
        impl_->core = core.get();
        impl_->codec = cfg.codec;
        impl_->fg_feature = cfg.core.ids.fg;
        impl_->fn_options = cfg.imports.fn_options;
        impl_->fn_state = cfg.imports.fn_state;
        impl_->presenter = cfg.presenter;
        impl_->present_route = cfg.present_route;
        impl_->present_route_ctx = cfg.present_route_ctx;
        impl_->factory_note = cfg.factory_note;
        impl_->factory_note_ctx = cfg.factory_note_ctx;
        impl_->owners = cfg.owners;
        impl_->log = cfg.log;
        impl_->log_ctx = cfg.log_ctx;
        core_ = std::move(core);
        g_route.store(impl_.get());
        for (const Found& f : found) {
            void* h = handler_for(f.slot.name_index);
            void* original = nullptr;
            if (!g::safe_get(f.slot.slot, &original) || original == h) continue;
            if (PatchSlot(f.slot.slot, h)) {
                patches_.push_back({f.slot.slot, original, h, f.slot.name_index, f.slot.delay_load, f.module});
                log(std::string("fg transport: patched ") + (f.slot.delay_load ? "delay-load" : "import") + " slot " +
                    FgRoleWord(f.slot.name_index) + " in " + ModuleLeaf(f.module));
            } else {
                log(std::string("fg transport: FAILED to patch slot ") + FgRoleWord(f.slot.name_index) + " in " +
                    ModuleLeaf(f.module));
            }
        }
        active_ = true;
        owner_release.keep = true;
        return finish(kFgInstalled, std::to_string(patches_.size()) + " slot(s) redirected");
    } catch (...) {
        // nothing half-done stays routed: back to the genuine interface for every call
        restore();
        return finish(kFgNullArgument, "exception during install; nothing stays patched");
    }
}

void FgAsiTransport::restore() {
    if (impl_) {
        // the creation hook this transport installed comes off; chains claimed for the bridge give their present owner
        // back (they stay wrapped as pass-throughs; the presenter hears Detached while the route still names it)
        if (impl_->hooked_factory.load()) FgSwapChainHook::Uninstall();
        FgSwapChain::DetachCallback(&g_present_bridge);
    }
    if (g_route.load() == impl_.get() && impl_) g_route.store(nullptr);
    for (auto it = patches_.rbegin(); it != patches_.rend(); ++it) {
        void* now = nullptr;
        if (g::safe_get(it->slot, &now) && now == it->handler && WG_FG_WIN32_MUTANT != 7)
            PatchSlot(it->slot, it->original);
    }
    patches_.clear();
    // calls already inside a handler finish before the core goes (bounded wait)
    for (int i = 0; i < 2000 && g_in_flight.load() > 0; ++i) Sleep(1);
    // a chain a creation call attached meanwhile (its sink call has finished: in-flight is 0, the route is gone)
    if (impl_) FgSwapChain::DetachCallback(&g_present_bridge);
    if (core_) core_->detach();
    core_.reset();
    impl_.reset();
    if (active_) {
        FgAsiTransport* me = this;
        g_owner.compare_exchange_strong(me, nullptr);
    }
    active_ = false;
}

}  // namespace win32
}  // namespace wgfgp
