// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the Win32 hook transport. See fg_asi_transport.h.
// Written for this project.
#include "fg_asi_transport.h"

#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2  // K32EnumProcessModules from kernel32: no extra library
#endif
#include <psapi.h>
#include <string.h>

#include <exception>

// Test-only mutants (CMakeLists.txt builds the Win32 suite against each; every one must fail):
//  1 install-off gate ignored  2 a forwarded call loses an argument  3 scan budget ignored  7 restore leaves slots
#ifndef WG_FG_WIN32_MUTANT
#define WG_FG_WIN32_MUTANT 0
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
};

namespace {

constexpr unsigned kMaxModules = 1024;   // as the RR hook's load scan
constexpr unsigned kMaxDescriptors = 4096;
constexpr uint32_t kMaxDecodedTags = 4096;

std::atomic<FgAsiTransport*> g_owner{nullptr};       // the active transport (one per process)
std::atomic<FgAsiTransport::Impl*> g_route{nullptr};  // null: every handler forwards to the genuine export
std::atomic<void*> g_genuine[kFgRoleCount] = {};      // kept after restore: a handler entered late still forwards
std::atomic<void*> g_genuine_options{nullptr}, g_genuine_state{nullptr};  // resolved lazily through get-function
std::atomic<uint32_t> g_fg_feature{WG_FEAT_FRAME_GENERATION};
std::atomic<long> g_in_flight{0};

struct Counters {
    std::atomic<uint64_t> calls[kFgRoleCount] = {};
    std::atomic<uint64_t> forwarded[kFgRoleCount] = {};
    std::atomic<uint64_t> fg_answered{0}, option_calls{0}, state_calls{0}, codec_failures{0}, scan_entries{0};
};
Counters g_ctr;

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
WgResultCode FwdState(const WgViewport* v, void* s, const std::string& name) {
    auto fn = reinterpret_cast<FgStateFn>(GenuineFeatureFunction(g_genuine_state, name));
    const WgResultCode r = fn ? fn(v, s) : WG_E_FEATURE_MISSING;
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
    return FwdState(t_frame ? t_frame->vp : nullptr, native_state, r ? r->fn_state : g_last_fn_state);
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

WgResultCode H_FgState(const WgViewport* v, void* native) {
    InFlight in;
    g_ctr.state_calls.fetch_add(1);
    FgAsiTransport::Impl* r = g_route.load();
    if (!r) return FwdState(v, native, g_last_fn_state);
    CallFrame f;
    f.role = kFgRoleGetFunction;
    f.vp = v;
    FrameScope s(&f);
    try {
        WgFgState st = {};
        st.struct_size = sizeof(st);
        st.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        const WgResultCode res = r->core->get_state(r->fg_feature, ViewportId(v), &st, native);
        if (f.forwarded || res != WG_OK) return res;
        if (!native) return WG_E_MISSING_INPUT_PARAMETER;
        if (!r->codec->write_state(st, native)) {
            g_ctr.codec_failures.fetch_add(1);
            return WG_E_INVALID_PARAMETER;
        }
        return WG_OK;
    } catch (...) {
        g_ctr.codec_failures.fetch_add(1);
        return f.forwarded ? f.fwd_result : WG_E_INVALID_PARAMETER;
    }
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
        default: return nullptr;
    }
}
void* FgAsiTransport::options_handler() { return reinterpret_cast<void*>(&H_FgOptions); }
void* FgAsiTransport::state_handler() { return reinterpret_cast<void*>(&H_FgState); }

void* FgAsiTransport::genuine(uint32_t role) const { return role < kFgRoleCount ? g_genuine[role].load() : nullptr; }

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
    if (g_route.load() == impl_.get() && impl_) g_route.store(nullptr);
    for (auto it = patches_.rbegin(); it != patches_.rend(); ++it) {
        void* now = nullptr;
        if (g::safe_get(it->slot, &now) && now == it->handler && WG_FG_WIN32_MUTANT != 7)
            PatchSlot(it->slot, it->original);
    }
    patches_.clear();
    // calls already inside a handler finish before the core goes (bounded wait)
    for (int i = 0; i < 2000 && g_in_flight.load() > 0; ++i) Sleep(1);
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
