// SPDX-License-Identifier: MIT
#include "frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ffx_describe.h"
#include "diagnostics.h"
#include "wg_settings_io.h"  // pure part: describe() (the registry's log words)

namespace wgfsr {

struct FrontEnd::UpscaleContext {
    enum class Mode { Own, Amd };
    FrontEnd* owner = nullptr;
    bool unsupportedSaid = false;  // the upscaler answered WG_ERROR_UNSUPPORTED (handed to AMD's FSR, or the stand-in)
    uint32_t id = 0;
    Mode mode = Mode::Own;
    UpscaleCreateParams create;
    void* device = nullptr;
    bool hasOverride = false;
    uint64_t overrideId = 0;
    const char* modelProviderName = nullptr;
    uint64_t modelProviderId = kWrongGpuProviderVersionId;
    char selectedModel = 0;
    const char* modelOption = nullptr;  // the registry model's runtime option ("model=K") when an entry or sr_model chose it
    ffxApiMessage fpMessage = nullptr;
    ffxContext amd = nullptr;     // AMD's context when mode == Amd
    const OriginalApi* amdApi = nullptr; // the exact DLL that created this context
    bool amdFromSaved = false;    // AMD context created by the fallback (default allocator), not by the game's call
    const WgUpscalerBackend* backend = nullptr;
    WgUpscaler* upscaler = nullptr;
    WgCreateInfo createInfo{};    // as given to the backend (for the one-frame stand-in fallback)
    const WgUpscalerBackend* fallbackBackend = nullptr;
    WgUpscaler* fallback = nullptr;  // stand-in for frames the backend is not ready for (WG_ERROR_NOT_READY)
    std::string options;          // kept alive for WgCreateInfo::options
    MapState map;
    uint64_t dispatches = 0;
    std::string failReason;       // why WrongGPU is not upscaling this context (the upscaler's first error line), for the overlay
    bool compared = false;        // the last upscale went to AMD for a comparison: the next one of the other upscaler resets
    bool amdUnavailable = false;  // AMD's context for comparisons could not be created
    bool splitBlocked = false;    // a dispatch list that can never be split was seen (and said once in the log)
    std::string amdName;          // "AMD FSR 3.1.4": AMD's provider of `amd`, for the overlay
    AmdComparison comparison;     // AMD's upscaler from compareApi (FSR 4 with AMD's SDK 2 DLL)
    // The upscaler's create did not finish within create_wait_ms (async_create.h): AMD's FSR upscales (mode Amd) until
    // a dispatch finds it finished and takes the upscaler over. createInfo / options / this context stay alive for it.
    std::shared_ptr<PendingCreate> pending;
    const WgUpscalerBackend* pendingBackend = nullptr;
    std::atomic<bool> creating{false};  // the create runs on its thread: its first error line goes to createError
    std::string createError;
    bool resetNext = false;       // the upscaler's next frame discards its history (a frame went past it, or it was parked)
    bool restartNext = false;     // ... and starts over as a newly created upscaler (reset = 2): it was parked
    uint32_t badRun = 0;          // frames in a row the upscaler could not take (kBadFrameRun)
    bool amdLast = false;         // the last frame was one of AMD's FSR for such a frame (AMD's history continues)
    bool broken = false;          // the upscaler failed for good: it is not parked
    uint64_t brokenFrames = 0;    // bug hunt F1: frames since the upscaler failed for good (it is not called again)
    uint32_t resizes = 0;         // bug hunt H: upscalers created again for the frames' output size (kMaxResizes)
    bool resizeFailed = false;    // ... and such a create failed: no more attempts for this context
    std::mutex mutex;             // one call at a time per context

    // This object (parked with its upscaler, stand-in, create info and options: the upscaler holds pointers into it)
    // becomes the context of a new create with the same key; `fresh` has what that create's chain said.
    void Renew(const UpscaleContext& fresh) {
        id = fresh.id;
        mode = Mode::Own;
        create = fresh.create;
        hasOverride = fresh.hasOverride;
        overrideId = fresh.overrideId;
        modelProviderName = fresh.modelProviderName;
        modelProviderId = fresh.modelProviderId;
        selectedModel = fresh.selectedModel;
        modelOption = fresh.modelOption;
        fpMessage = fresh.fpMessage;
        amd = nullptr;
        amdApi = nullptr;
        amdFromSaved = false;
        map = MapState{};
        dispatches = 0;
        failReason.clear();
        compared = false;
        amdUnavailable = false;
        splitBlocked = false;
        amdName.clear();
        comparison = AmdComparison{};
        createError.clear();
        resetNext = true;  // as a newly created upscaler: no history, counters from 0 (WgFrame::reset = 2)
        restartNext = true;
        badRun = 0;
        amdLast = false;
    }
};

namespace {

void UpscalerLog(void* user, int level, const char* message) {
    auto* u = static_cast<FrontEnd::UpscaleContext*>(user);
    const char* tag = level == WG_LOG_ERROR ? "ERROR" : level == WG_LOG_WARN ? "warning" : "info";
    char prefix[64];
    std::snprintf(prefix, sizeof prefix, "  U%u upscaler %s: ", u->id, tag);
    if (level == WG_LOG_ERROR && message) {
        std::string& reason = u->creating ? u->createError : u->failReason;
        if (reason.empty()) reason = message;
    }
    u->owner->LogLine(std::string(prefix) + (message ? message : ""));
    // clean/fe-wrapped-list: "about to ..." comes right before a step that may take the process down (HIP start-up, the
    // shared-memory import): on disk now, so the log says where it stopped
    if (message && std::strncmp(message, "about to ", 9) == 0) u->owner->FlushLog();
}

const char* RcName(ffxReturnCode_t rc) { return ReturnCodeName(rc); }

// The front end's calls in progress on this thread. A host that routes AMD's DLL back into WrongGPU (OptiScaler's
// custom slot: the game's FidelityFX files are OptiScaler's way in, and their upscale calls go to the selected
// upscaler, this DLL) turns an AMD fallback call into a new WrongGPU call on the same thread, inside the first one:
// Cyberpunk on 2026-10-30 went 36 creates deep (each AMD dispatch created the next context) until the thread's stack
// gave out. Re-entry is refused for upscaler creates (never legitimate inside another call of this front end) and for
// any call on a context this thread is already inside of (its lock is held: a second call would deadlock on it).
struct ThreadCalls {
    int depth = 0;
    std::vector<const void*> busy;  // WrongGPU contexts this thread is inside a call of
};
thread_local ThreadCalls t_calls;

struct EntryScope {
    const void* context = nullptr;
    EntryScope() { ++t_calls.depth; }
    ~EntryScope() {
        --t_calls.depth;
        if (context) t_calls.busy.pop_back();
    }
    bool Nested() const { return t_calls.depth > 1; }
    // Marks `c` as in use by this thread; false (nothing marked) when this thread is already inside a call of it.
    bool Enter(const void* c) {
        if (std::find(t_calls.busy.begin(), t_calls.busy.end(), c) != t_calls.busy.end()) return false;
        t_calls.busy.push_back(c);
        context = c;
        return true;
    }
};

// Lines held back while a create is decided (FrontEnd::Log): a custom-slot create that only repeats a failure
// remembered for the session logs one summary line at growing intervals instead of its whole create log.
struct HeldLog {
    std::vector<std::string> lines;
    bool drop = false;  // a repeat between two log points: nothing of this create is logged
};
thread_local HeldLog* t_held = nullptr;

bool LogPoint(uint64_t n) {  // 1, 2, 3, 10, 100, 1000, ...
    if (n <= 3) return true;
    while (n % 10 == 0) n /= 10;
    return n == 1;
}

bool OwnModelK(const std::string& options) {
    const char* name = RuntimeProviderName(options, std::getenv("WG_RUNTIME_OPTIONS"));
    const WgEntryDef* k = nullptr;
    for (int m = 0; m < WG_SR_MODEL_COUNT; ++m)
        if (std::strcmp(kWgModelRows[m].name, "K") == 0) k = SrEntryOfModel(m);
    return k && std::strcmp(name, EntryProviderName(*k)) == 0;
}

// A frame whose sizes do not fit its textures (a resolution change of the game: the stated render size is the old one,
// the textures are the new ones, or the reverse) made usable for the stand-in: the regions are cut to the textures.
// False when the stand-in cannot upscale it either (no command list, color or output).
bool UsableForStandIn(WgFrame& f) {
    if (!f.command_list || !f.color.resource || !f.output.resource) return false;
    auto cut = [](uint32_t& size, uint32_t texture) {
        if (texture && (!size || size > texture)) size = texture;
    };
    cut(f.render_width, f.color.width);
    cut(f.render_height, f.color.height);
    cut(f.output_width, f.output.width);
    cut(f.output_height, f.output.height);
    return f.render_width && f.render_height && f.output_width && f.output_height;
}

// A hosted caller can select our version even when configuration/startup chooses AMD. Remove only our private
// version descriptor. Shadow known prefix nodes so caller memory and allocator identity remain unchanged; an
// opaque suffix stays attached verbatim. We cannot copy an unknown prefix whose size is not in the public API.
// useVersion != 0: the last private descriptor is replaced by one naming AMD's version `useVersion` instead of being
// removed (the OptiScaler plugin's AMD entry while the host's FFX sub-box names WrongGPU).
ffxReturnCode_t CreateAmdWithoutPrivateVersion(const OriginalApi& api, ffxContext* context,
                                              ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb,
                                              uint64_t useVersion = 0) {
    const ffxApiHeader* lastPrivate = nullptr;
    size_t length = 0;
    for (const ffxApiHeader* h = desc; h; h = h->pNext) {
        if (++length > 64) return FFX_API_RETURN_ERROR_PARAMETER;
        if (h->type == FFX_API_DESC_TYPE_OVERRIDE_VERSION &&
            IsWrongGpuProvider(reinterpret_cast<const ffxOverrideVersion*>(h)->versionId)) lastPrivate = h;
    }
    if (!lastPrivate) return api.createContext(context, desc, memCb);
    struct Shadow {
        ffxCreateContextDescUpscale upscale{};
        BackendDx12DescView backend{};
        ffxOverrideVersion version{};
        ffxApiHeader* header = nullptr;
    };
    std::vector<Shadow> nodes;
    nodes.reserve(length);  // keep every header address stable
    for (const ffxApiHeader* h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_DESC_TYPE_OVERRIDE_VERSION &&
            IsWrongGpuProvider(reinterpret_cast<const ffxOverrideVersion*>(h)->versionId)) {
            if (h == lastPrivate) {
                if (useVersion) {
                    nodes.emplace_back();
                    nodes.back().version = *reinterpret_cast<const ffxOverrideVersion*>(h);
                    nodes.back().version.versionId = useVersion;
                    nodes.back().header = &nodes.back().version.header;
                }
                break;
            }
            continue;
        }
        nodes.emplace_back();
        auto& node = nodes.back();
        switch (h->type) {
        case FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE:
            node.upscale = *reinterpret_cast<const ffxCreateContextDescUpscale*>(h);
            node.header = &node.upscale.header;
            break;
        case kCreateBackendDx12:
            node.backend = *reinterpret_cast<const BackendDx12DescView*>(h);
            node.header = &node.backend.header;
            break;
        case FFX_API_DESC_TYPE_OVERRIDE_VERSION:
            node.version = *reinterpret_cast<const ffxOverrideVersion*>(h);
            node.header = &node.version.header;
            break;
        default: return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
        }
    }
    for (size_t i = 0; i < nodes.size(); ++i)
        nodes[i].header->pNext = i + 1 < nodes.size() ? nodes[i + 1].header : lastPrivate->pNext;
    return api.createContext(context, nodes.empty() ? const_cast<ffxApiHeader*>(lastPrivate->pNext) : nodes[0].header, memCb);
}

}  // namespace

#if WRONGGPU_MODEL_K
FrontEnd::FrontEnd(Platform platform, Config config) : platform_(std::move(platform)), config_(std::move(config)), providerName_(RuntimeProviderName(config_.runtimeOptions, std::getenv("WG_RUNTIME_OPTIONS"))) {}
#else
FrontEnd::FrontEnd(Platform platform, Config config) : platform_(std::move(platform)), config_(std::move(config)) {
    if (const std::string note = IgnoreModelK(config_); !note.empty()) Log(note);  // (config.h; one log line)
    providerName_ = RuntimeProviderName(config_.runtimeOptions, std::getenv("WG_RUNTIME_OPTIONS"));
}
#endif

FrontEnd::~FrontEnd() {
    // Contexts the game never destroyed: the process is ending; GPU objects are left to the runtime.
    std::lock_guard lock(mutex_);
    for (auto& entry : own_)
        if (!entry.first->pending) delete entry.first;  // one whose create still runs is left to the process exit
    own_.clear();
    std::lock_guard park(parkMutex_);
    delete parked_;  // the same for a parked one: its upscaler is not destroyed here
    parked_ = nullptr;
}

// ---------------------------------------------------------------------------------------------------------------
// Parked upscalers (frontend.h)

bool FrontEnd::HasParked() {
    std::lock_guard lock(parkMutex_);
    return parked_ != nullptr;
}

bool FrontEnd::OtherUpscalerLocked() const {
    if (!own_.empty()) return true;
    for (const auto& f : forwarded_)
        if (f.second.upscale) return true;
    return false;
}

bool FrontEnd::CanPark(const UpscaleContext& u) {
    if (!config_.contextParkMs || !platform_.timer || !u.upscaler || !u.backend || u.pending) return false;
    // Only an upscaler that upscaled this context to the end: not one AMD's FSR replaced, not one that failed.
    if (u.mode != UpscaleContext::Mode::Own || u.broken || u.splitBlocked) return false;
    // sync=pipelined (a runtime option) shows the previous frame's result: a new context must not start with a frame of
    // the old one.
    const char* env = std::getenv("WG_RUNTIME_OPTIONS");
    if (u.options.find("sync=pipelined") != std::string::npos || (env && std::strstr(env, "sync=pipelined"))) return false;
    return !(platform_.deviceLost && platform_.deviceLost(u.device));
}

void FrontEnd::Retire(UpscaleContext* u, const std::string& tag, const char* why) {
    const auto t0 = std::chrono::steady_clock::now();
    if (u->upscaler) u->backend->destroy(u->upscaler);
    u->upscaler = nullptr;
    if (u->fallback) u->fallbackBackend->destroy(u->fallback);
    u->fallback = nullptr;
    char ms[32];
    std::snprintf(ms, sizeof ms, "%.1f", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    Log(tag + " parked upscaler of U" + std::to_string(u->id) + " destroyed in " + ms + " ms: " + why);
    delete u;
}

void FrontEnd::Park(UpscaleContext* u, const std::string& tag) {
    bool arm = false;
    {
        std::lock_guard lock(parkMutex_);
        if (parked_) Retire(parked_, tag, "another context was destroyed and is parked in its place");
        parked_ = u;
        parkedAt_ = std::chrono::steady_clock::now();
        parkedUntil_ = parkedAt_ + std::chrono::milliseconds(config_.contextParkMs);
        arm = !parkTimerArmed_;
        parkTimerArmed_ = true;
    }
    if (arm && !platform_.timer(config_.contextParkMs, [this] { return ParkTimer(); })) {
        {
            std::lock_guard lock(parkMutex_);
            parkTimerArmed_ = false;
        }
        DropParked(tag, "no timer could be started to end its time");
    }
}

uint32_t FrontEnd::ParkTimer() {
    std::lock_guard lock(parkMutex_);
    if (parked_) {
        const auto now = std::chrono::steady_clock::now();
        if (now < parkedUntil_)  // parked later than the one this timer was started for
            return uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(parkedUntil_ - now).count()) + 1;
        UpscaleContext* u = parked_;
        parked_ = nullptr;
        Retire(u, "timer:", ("no context with the same parameters was created within " + std::to_string(config_.contextParkMs) +
                              " ms (context_park_ms)").c_str());
    }
    parkTimerArmed_ = false;
    return 0;
}

void FrontEnd::DropParked(const std::string& tag, const char* why) {
    std::lock_guard lock(parkMutex_);
    if (!parked_) return;
    UpscaleContext* u = parked_;
    parked_ = nullptr;
    Retire(u, tag, why);
}

FrontEnd::UpscaleContext* FrontEnd::TakeParked(const WgUpscalerBackend* backend, const UpscaleContext& fresh, const std::string& tag,
                                               double* parkedMs) {
    std::lock_guard lock(parkMutex_);  // (waits for a parked upscaler the timer is destroying right now)
    UpscaleContext* p = parked_;
    if (!p) return nullptr;
    parked_ = nullptr;
    const WgCreateInfo& a = p->createInfo;
    const WgCreateInfo& b = fresh.createInfo;
    const char* why = nullptr;
    if (p->backend != backend || a.struct_size != b.struct_size || a.interface_version != b.interface_version || a.services != b.services)
        why = "the new context has another upscaler implementation";
    else if (a.d3d12_device != b.d3d12_device) why = "the new context has another D3D12 device";
    else if (a.max_render_width != b.max_render_width || a.max_render_height != b.max_render_height ||
             a.output_width != b.output_width || a.output_height != b.output_height)
        why = "the new context has other sizes";
    else if (a.flags != b.flags || p->create.flags != fresh.create.flags) why = "the new context has other flags";
    else if (p->options != fresh.options) why = "the new context has other options or another model";
    else if (platform_.deviceLost && platform_.deviceLost(b.d3d12_device)) why = "the D3D12 device was removed";
    if (!why) {
        *parkedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - parkedAt_).count();
        return p;
    }
    Retire(p, tag, why);
    return nullptr;
}

void FrontEnd::Log(const std::string& line) {
    if (t_held) {
        if (!t_held->drop) t_held->lines.push_back(line);
        return;
    }
    if (platform_.log) platform_.log(line);
}

const OriginalApi* FrontEnd::Original() { return platform_.original ? platform_.original() : nullptr; }

uint64_t FrontEnd::NewestAmdVersionLocked() const {
    uint64_t best = 0;
    unsigned bestV[3] = {};
    bool any = false;
    for (const auto& [id, api] : versionApi_) {
        if (!api) continue;
        unsigned v[3] = {};
        auto name = versionNames_.find(id);
        if (name != versionNames_.end()) {
            const char* p = name->second.c_str();
            while (*p && (*p < '0' || *p > '9')) ++p;
            for (int i = 0; i < 3 && *p; ++i) {
                char* end = nullptr;
                v[i] = unsigned(std::strtoul(p, &end, 10));
                p = end;
                if (*p != '.') break;
                ++p;
            }
        }
        if (!any || std::lexicographical_compare(bestV, bestV + 3, v, v + 3)) {
            any = true;
            best = id;
            std::copy(v, v + 3, bestV);
        }
    }
    return best;
}

const OriginalApi* FrontEnd::UpscaleFallbackApi() {
    if (const OriginalApi* api = Original()) return api;
    // Custom slot: the comparison DLL is one of the host's FidelityFX DLLs (config.h Config::customSlot): never.
    if (!config_.hosted || config_.customSlot || !platform_.compareApi) return nullptr;
    const OriginalApi* api = platform_.compareApi();
    // Only the upscale fallback can use the standalone AMD SDK DLL. Other
    // effect/global calls retain their existing original-DLL forwarding route.
    return api && api->createContext && api->destroyContext && api->configure && api->query && api->dispatch ? api : nullptr;
}

size_t FrontEnd::OwnContextCount() {
    std::lock_guard lock(mutex_);
    return own_.size();
}

FrontEnd::UpscaleContext* FrontEnd::FindOwn(ffxContext* context) {
    if (!context || !*context) return nullptr;
    std::lock_guard lock(mutex_);
    auto* candidate = static_cast<UpscaleContext*>(*context);
    return own_.count(candidate) ? candidate : nullptr;
}

std::string FrontEnd::Name(ffxContext* context) {
    if (!context) return "(no context pointer)";
    if (!*context) return "null";
    if (UpscaleContext* u = FindOwn(context)) {
        char text[96];
        std::snprintf(text, sizeof text, "U%u(%s)", u->id,
                      u->mode == UpscaleContext::Mode::Amd ? "AMD FSR" : (u->backend ? u->backend->name : "?"));
        return text;
    }
    std::lock_guard lock(mutex_);
    auto it = forwarded_.find(*context);
    return it != forwarded_.end() ? it->second.name : "unknown " + Pointer(*context);
}

bool FrontEnd::ShouldLogDispatch(uint64_t dispatchCount) {
    if (!config_.logDispatches) return dispatchCount < 3;
    if (platform_.logBytes && platform_.logBytes() > uint64_t(config_.logMaxMegabytes) << 20) {
        if (!thinning_.exchange(true))
            Log("log is larger than log_max_mb: from now on only every 600th dispatch of a context is logged");
        return dispatchCount % 600 == 0;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// ffxCreateContext

ffxReturnCode_t FrontEnd::CreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    const EntryScope scope;
    const uint64_t call = ++calls_;
    bool upscale = false;
    for (const ffxApiHeader* h = desc; h; h = h->pNext)
        upscale = upscale || h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    if (upscale && scope.Nested()) {
        // An upscaler create inside another call of this front end on the same thread (see ThreadCalls): refused, so a
        // host routing AMD's DLL back to WrongGPU cannot build an endless chain of contexts.
        const uint64_t n = ++reentered_;
        if (LogPoint(n))
            Log("#" + std::to_string(call) + " ffxCreateContext " + DescribeChain(desc) + " -> ERROR_RUNTIME_ERROR: re-entered on this "
                "thread from inside another FidelityFX call of WrongGPU (the AMD DLL it called leads back to WrongGPU); refused to stop a "
                "loop of contexts (refusal " + std::to_string(n) + ")");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // Custom slot with a create failure remembered for the session: this create's log is held until it is known whether
    // it only repeats that failure (then one summary line now and then) or tries something new (then logged in full).
    HeldLog held;
    struct Hold {
        FrontEnd& fe;
        HeldLog& log;
        bool on = false;
        ~Hold() {
            if (!on) return;
            t_held = nullptr;
            if (!log.drop)
                for (const std::string& l : log.lines) fe.Log(l);
        }
    } hold{*this, held};
    if (upscale && config_.customSlot && context && desc) {
        std::lock_guard lock(mutex_);
        hold.on = !failedCreates_.empty();
    }
    if (hold.on) t_held = &held;
    Log("#" + std::to_string(call) + " ffxCreateContext " + DescribeChain(desc) + " memCb=" + Pointer(memCb));
    if (!context || !desc) {
        Log("#" + std::to_string(call) + " -> ERROR_PARAMETER (null context pointer or description)");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (upscale) return CreateUpscale(context, desc, memCb, call);

    const OriginalApi* original = Original();
    if (!original) {
        Log("#" + std::to_string(call) + " -> NO_PROVIDER (AMD's original DLL is not available)");
        return FFX_API_RETURN_NO_PROVIDER;
    }
    const ffxReturnCode_t rc = original->createContext(context, desc, memCb);
    if (rc == FFX_API_RETURN_OK && *context) {
        const char* type = StructTypeName(desc->type);
        const std::string name = std::string("amd:") + (type ? type : Hex(desc->type).c_str()) + "#" + std::to_string(nextId_++);
        std::lock_guard lock(mutex_);
        forwarded_[*context] = Forwarded{name, 0};
    }
    Log("#" + std::to_string(call) + " -> " + RcName(rc) + " (forwarded to AMD) context=" + Name(context));
    return rc;
}

ffxReturnCode_t FrontEnd::CreateUpscale(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb,
                                        uint64_t call) {
    const std::string tag = "#" + std::to_string(call);
    // A parked upscaler lives until the next upscaler context is created: that one takes it over, or it is destroyed.
    struct ParkGuard {
        FrontEnd& fe;
        const std::string& tag;
        bool settled = false;
        ~ParkGuard() {
            if (!settled) fe.DropParked(tag, "an upscaler context that does not use it was created");
        }
    } parkGuard{*this, tag};
    auto* u = new UpscaleContext;
    u->owner = this;
    u->modelProviderName = providerName_;
    u->id = nextId_++;
    for (const ffxApiHeader* h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
            auto* c = reinterpret_cast<const ffxCreateContextDescUpscale*>(h);
            u->create.flags = c->flags;
            u->create.maxRenderSize = c->maxRenderSize;
            u->create.maxUpscaleSize = c->maxUpscaleSize;
            u->fpMessage = c->fpMessage;
        } else if (h->type == kCreateBackendDx12) {
            u->device = reinterpret_cast<const BackendDx12DescView*>(h)->device;
        } else if (h->type == FFX_API_DESC_TYPE_OVERRIDE_VERSION) {
            u->hasOverride = true;
            u->overrideId = reinterpret_cast<const ffxOverrideVersion*>(h)->versionId;
        } else {
            Log(tag + " note: chained " + (StructTypeName(h->type) ? StructTypeName(h->type) : Hex(h->type)) +
                " is not used by WrongGPU (kept for AMD if the context falls back)");
        }
    }

    // OptiScaler ASI plugin (clean/fsr/asi): when it has hooked the host's MAIN upscaler list, its entry decides.
    // "WrongGPU" -> WrongGPU, whatever the FFX sub-box names. "FSR 3.X(/4)" -> AMD: the version the FFX sub-box names
    // when that is one of AMD's, else AMD's newest listed version (FSR 4 when the AMD provider has it).
    char asiModel = 0;
    const bool asiHooked = config_.hosted && platform_.asiPlugin && platform_.asiPlugin(&asiModel);
    bool forceAmd = false;
    if (asiHooked) {
        if (asiModel == 'M' || asiModel == 'K') {
#if WRONGGPU_MODEL_K
            const char model = asiModel;
#else
            const char model = 'M';  // a plugin built with Model K, with a front end without it: Model M
#endif
            u->selectedModel = model;
            u->modelProviderId = model == 'K' ? kWrongGpuKProviderVersionId : kWrongGpuProviderVersionId;
            u->modelProviderName = model == 'K' ? "WrongGPU Model K" : "WrongGPU Model M";
            u->hasOverride = false;  // an AMD version in the FFX sub-box must not route this context to AMD
            Log(tag + " OptiScaler plugin: " + u->modelProviderName + " selected in the host's upscaler list");
        } else {
            forceAmd = true;
            Log(tag + " OptiScaler plugin: AMD FSR selected in the host's upscaler list");
        }
    }

    // One folder per choice (registry option entry_source = folders): this DLL's folder IS the choice; the host's
    // override can only name it (it is the one entry listed), or be absent.
    const char* entrySource = "menu";
    if (config_.hosted && !asiHooked && platform_.menu.folderEntry && OwnEntryOfId(platform_.menu.folderEntry) &&
        (!u->hasOverride || IsWrongGpuProvider(u->overrideId))) {
        u->hasOverride = true;
        u->overrideId = platform_.menu.folderEntry;
        entrySource = "folder";
    }
    // The host's choice of upscaler version (QUERY_GET_VERSIONS, ListVersions); with the plugin, only for its AMD entry.
    if (config_.hosted && (forceAmd || (!asiHooked && u->hasOverride))) {
        if (!forceAmd && IsWrongGpuProvider(u->overrideId)) {
            // One of WrongGPU's registry entries (an SR model, or an RR mode: the SR context keeps the current model).
            ApplyEntry(*OwnEntryOfId(u->overrideId), *u, tag, entrySource);
        } else {
            const OriginalApi* api = nullptr;
            UpscaleStatus status;
            status.state = UpscaleState::Amd;
            status.device = u->device;
            uint64_t amdId = u->hasOverride ? u->overrideId : 0;
            {
                std::lock_guard lock(mutex_);
                auto it = versionApi_.find(amdId);
                if (forceAmd && (!u->hasOverride || it == versionApi_.end())) {
                    amdId = NewestAmdVersionLocked();  // the sub-box names WrongGPU (or nothing): AMD's newest
                    it = versionApi_.find(amdId);
                }
                if (it != versionApi_.end()) api = it->second;
                auto name = versionNames_.find(amdId);
                status.upscaler = name != versionNames_.end() ? name->second : "AMD FSR";
                status.reason = "The host selected " + status.upscaler;
            }
            if (api) {
                const uint64_t chosen = amdId;
                const bool substitute = !u->hasOverride || amdId != u->overrideId;
                delete u;
                const ffxReturnCode_t rc = substitute ? CreateAmdWithoutPrivateVersion(*api, context, desc, memCb, chosen)
                                                      : api->createContext(context, desc, memCb);
                if (rc == FFX_API_RETURN_OK && *context) {
                    const std::string name = "amd:upscale#" + std::to_string(nextId_++);
                    std::lock_guard lock(mutex_);
                    forwarded_[*context] = Forwarded{name, 0, api, status, true};
                }
                Log(tag + " -> " + RcName(rc) + " (the host chose AMD's upscaler version " + Hex(chosen) + "; forwarded) context=" +
                    Name(context));
                return rc;
            }
            if (!forceAmd)
                Log(tag + " note: the host asked for upscaler version " + Hex(u->overrideId) + ", which WrongGPU did not list; WrongGPU upscales");
        }
        u->hasOverride = false;  // WrongGPU's own id means nothing to AMD's DLL (fallback, comparison)
    }
    // No entry chose the model (direct route, no override, an id WrongGPU did not list): sr_model when it is set in
    // the shared block, the environment or wronggpu.ini; otherwise runtime_options / the runtime's default as before.
    if (!forceAmd && !u->selectedModel && !u->modelOption) {
        int32_t model = -1;
        std::string described;
        if (ConfiguredModel(&model, &described)) {
            const WgEntryDef* sr = SrEntryOfModel(model);
            u->selectedModel = kWgModelRows[model].name[0];
            u->modelOption = kWgModelRows[model].runtime_option;
            u->modelProviderId = sr->id;
            u->modelProviderName = EntryProviderName(*sr);
            Log(tag + " settings: " + described);
        }
    }

    std::string why;
    bool repeatedFailure = false;  // custom slot: a failure remembered for the session, not tried again
    if (forceAmd) {
        why = "OptiScaler plugin: AMD FSR selected";
    } else if (config_.upscaler == UpscalerChoice::Amd) {
        why = "upscaler = amd in wronggpu.ini";
    } else if (u->create.flags & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE) {
        why = "non-linear (perceptual) colour (FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE): AMD FSR handles it (spec 20, ANSWERS item 5)";
    } else if (!u->device) {
        why = "no CREATE_BACKEND_DX12 description (D3D12 device) in the chain";
    } else if (!u->create.maxRenderSize.width || !u->create.maxRenderSize.height || !u->create.maxUpscaleSize.width ||
               !u->create.maxUpscaleSize.height) {
        why = "maxRenderSize or maxUpscaleSize is 0";
    } else if (const WgUpscalerBackend* backend = platform_.backend ? platform_.backend(config_, why) : nullptr) {
        WgCreateInfo info{};
        MapCreate(u->create, u->device, info);
        u->options = config_.BackendOptions();
        if (u->selectedModel || u->modelOption) {
            u->options += u->modelOption ? std::string(";") + u->modelOption : std::string(";model=") + u->selectedModel;
            // An inherited environment override must never make the selected provider run another model.
            if (std::strcmp(RuntimeProviderName(u->options, std::getenv("WG_RUNTIME_OPTIONS")), u->modelProviderName) != 0) {
                Log(tag + " -> ERROR_RUNTIME_ERROR: WG_RUNTIME_OPTIONS conflicts with the host's selected model");
                delete u;
                return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
            }
        }
        ApplyModelDir(*u, tag);
        info.options = u->options.c_str();
        info.log = &UpscalerLog;
        info.log_user = u;
        // The split service only (the hooks it needs are installed after the create succeeded: ActivateSplit).
        AdaptCreateInfo(backend, platform_.services ? platform_.services(backend, u->device) : nullptr, info);
        u->createInfo = info;
        // Custom slot: the same upscaler with the same options failed earlier in this session: no new attempt.
        std::string remembered;
        uint64_t repeats = 0;
        if (config_.customSlot) {
            std::lock_guard lock(mutex_);
            auto it = failedCreates_.find(FailureKey(backend, *u));
            if (it != failedCreates_.end()) {
                remembered = it->second.reason;
                repeats = ++it->second.repeats;
            }
        }
        if (!remembered.empty()) {
            if (t_held) {  // only the summary below, now and then (LogPoint); between those, nothing
                t_held->lines.clear();
                t_held->drop = !LogPoint(repeats);
            }
            why = std::string("upscaler \"") + backend->name + "\" failed earlier in this session with the same options (" + remembered +
                  "); it is not created again until the game restarts";
            u->failReason = remembered;
            repeatedFailure = true;
            if (LogPoint(repeats))
                Log(tag + " custom slot: create " + std::to_string(repeats) + " since the failure: " + why);
        } else if (t_held) {
            // Something new: the held lines go out now, in order, before the create's own lines.
            std::vector<std::string> lines = std::move(t_held->lines);
            t_held->lines.clear();
            t_held = nullptr;
            for (const std::string& l : lines) Log(l);
        }
        if (remembered.empty()) Log(tag + " WrongGPU upscaler \"" + backend->name + "\" create: " + DescribeCreateInfo(info));
        double parkedMs = 0;
        UpscaleContext* warm = remembered.empty() ? TakeParked(backend, *u, tag, &parkedMs) : nullptr;
        if (remembered.empty()) parkGuard.settled = true;
        if (!remembered.empty()) {
            // (nothing created: `why` says so; the fallback below decides)
        } else if (warm) {
            // The parked context object stays the upscaler's (its log callback and options point into it): it becomes
            // this context. Nothing is created; the first frame discards the history, as a new upscaler's would.
            char ms[32];
            std::snprintf(ms, sizeof ms, "%.0f", parkedMs);
            const uint32_t was = warm->id;
            warm->Renew(*u);
            delete u;
            u = warm;
            Log(tag + " took over the parked upscaler of U" + std::to_string(was) + " (same device, sizes, flags, options and model; parked " +
                ms + " ms ago): nothing is created, the first frame starts without history");
        } else {
            // On a worker thread: the game's thread waits create_wait_ms at most (async_create.h). Without the fallback to
            // AMD there is nothing else to upscale with, so it waits until the create finishes.
            // Custom slot without an AMD DLL of its own there is nothing to upscale with meanwhile: it waits as with
            // fallback_to_amd = 0.
            const bool amdMeanwhile = config_.fallbackToAmd && (!config_.customSlot || UpscaleFallbackApi());
            u->creating = true;
            auto pending = PendingCreate::Start(backend, u->createInfo);
            if (pending->Wait(amdMeanwhile ? config_.createWaitMs : UINT32_MAX)) {
                u->creating = false;
                if (u->failReason.empty()) u->failReason = u->createError;
                const WgResult r = pending->Result();
                u->upscaler = pending->Take();
                if (r == WG_OK && u->upscaler) {
                    if (ActivateSplit(*u, backend, tag)) {
                        u->backend = backend;
                    } else {
                        // As before hooks were installed after the create: an upscaler without the split service (the
                        // runtime decides whether it can run without it).
                        backend->destroy(u->upscaler);
                        u->upscaler = nullptr;
                        u->createInfo.services = nullptr;
                        Log(tag + " U" + std::to_string(u->id) + ": created again without the command-list split service");
                        u->creating = true;
                        WgUpscaler* again = nullptr;
                        const WgResult r2 = backend->create(&u->createInfo, &again);
                        u->creating = false;
                        if (r2 == WG_OK && again) {
                            u->upscaler = again;
                            u->backend = backend;
                        } else {
                            if (again) backend->destroy(again);
                            if (u->failReason.empty()) u->failReason = u->createError;
                            why = std::string("upscaler \"") + backend->name + "\" create without the split service failed: " + WgResultText(r2);
                        }
                    }
                } else {
                    why = std::string("upscaler \"") + backend->name + "\" create failed: " + WgResultText(r);
                    u->upscaler = nullptr;
                }
                if (!u->backend && config_.customSlot) {
                    const std::string reason = u->createError.empty() ? why : why + ": " + u->createError;
                    {
                        std::lock_guard lock(mutex_);
                        failedCreates_[FailureKey(backend, *u)] = RememberedFailure{reason, 0};
                    }
                    Log(tag + " custom slot: this failure is remembered for the session: a create of upscaler \"" + backend->name +
                        "\" with the same options fails at once from now on, without a new attempt (logged now and then)");
                }
            } else {
                u->pending = pending;
                u->pendingBackend = backend;
                why = std::string("upscaler \"") + backend->name + "\" is still starting after " + std::to_string(config_.createWaitMs) +
                      " ms (create_wait_ms); it takes over when it has started";
            }
        }
    } else if (why.empty()) {
        why = "no upscaler implementation available";
    }

    if (!u->backend) {
        if (config_.customSlot && !u->pending && !UpscaleFallbackApi()) {
            // Custom slot, no AMD DLL of WrongGPU's own: a clean failure, and the host's own upscaler takes over. Never
            // the host's FidelityFX DLLs (config.h Config::customSlot): they would call this DLL again.
            if (!repeatedFailure)
                Log(tag + " -> ERROR_RUNTIME_ERROR: " + why + "; hosted in OptiScaler's custom slot without an AMD DLL of WrongGPU's own "
                    "(amd_fidelityfx_dx12.wronggpu-original.dll next to this DLL): no context, the host's own upscaler takes over");
            delete u;
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        if (config_.upscaler != UpscalerChoice::Amd && !config_.fallbackToAmd) {
            Log(tag + " -> ERROR_RUNTIME_ERROR: " + why + " (fallback_to_amd = 0)");
            delete u;
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        const OriginalApi* original = UpscaleFallbackApi();
        if (!original) {
            Log(tag + " -> NO_PROVIDER: " + why + "; AMD's original DLL is not available either");
            if (u->pending) {
                // A bounded create can still use u's options and log callback.
                // Keep them alive and destroy any late result before retiring u.
                const WgUpscalerBackend* backend = u->pendingBackend;
                PendingCreate::Abandon(u->pending, [u, backend](WgUpscaler* upscaler) {
                    if (upscaler) backend->destroy(upscaler);
                    delete u;
                });
            } else {
                delete u;
            }
            return FFX_API_RETURN_NO_PROVIDER;
        }
        Log(tag + " upscaler context goes to AMD FSR" + (u->pending ? " for now" : "") + ": " + why);
        if (!u->pending && (u->failReason.empty() || config_.upscaler == UpscalerChoice::Amd)) u->failReason = why;
        const ffxReturnCode_t rc = config_.hosted ? CreateAmdWithoutPrivateVersion(*original, &u->amd, desc, memCb)
                                               : original->createContext(&u->amd, desc, memCb);
        if (rc != FFX_API_RETURN_OK) {
            Log(tag + " -> " + RcName(rc) + " (AMD's create failed)");
            if (u->pending) {
                const WgUpscalerBackend* backend = u->pendingBackend;
                PendingCreate::Abandon(u->pending, [u, backend](WgUpscaler* upscaler) {
                    if (upscaler) backend->destroy(upscaler);
                    delete u;
                });
            } else {
                delete u;
            }
            return rc;
        }
        u->amdApi = original;
        u->mode = UpscaleContext::Mode::Amd;
    }
    {
        std::lock_guard lock(mutex_);
        own_[u] = true;
    }
    *context = u;
    Log(tag + " -> OK context=" + Name(context) + (u->mode == UpscaleContext::Mode::Amd ? " amd=" + Pointer(u->amd) : ""));
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t FrontEnd::CreateAmdFromSaved(UpscaleContext& u, uint64_t call, bool takeOver) {
    if (u.amd) {  // created earlier for a comparison
        if (takeOver) u.mode = UpscaleContext::Mode::Amd;
        return FFX_API_RETURN_OK;
    }
    const OriginalApi* original = UpscaleFallbackApi();
    if (!original) return FFX_API_RETURN_NO_PROVIDER;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags = u.create.flags;
    create.maxRenderSize = u.create.maxRenderSize;
    create.maxUpscaleSize = u.create.maxUpscaleSize;
    create.fpMessage = u.fpMessage;
    BackendDx12DescView backend{};
    backend.header.type = kCreateBackendDx12;
    backend.device = u.device;
    ffxOverrideVersion version{};
    version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version.versionId = u.overrideId;
    create.header.pNext = &backend.header;
    backend.header.pNext = u.hasOverride ? &version.header : nullptr;
    Log("#" + std::to_string(call) + " creating AMD FSR context for U" + std::to_string(u.id) + ": " + DescribeChain(&create.header));
    const ffxReturnCode_t rc = original->createContext(&u.amd, &create.header, nullptr);
    if (rc == FFX_API_RETURN_OK) {
        u.amdApi = original;
        u.amdFromSaved = true;
        if (takeOver) u.mode = UpscaleContext::Mode::Amd;
    }
    Log("#" + std::to_string(call) + " AMD FSR create -> " + RcName(rc));
    return rc;
}

void FrontEnd::CheckPending(UpscaleContext& u, uint64_t call) {
    const std::string tag = "#" + std::to_string(call) + " U" + std::to_string(u.id);
    if (!u.pending->Done()) {
        for (const double seconds : {10.0, 30.0, 60.0, 300.0})
            if (u.pending->PassedThreshold(seconds))
                Log(tag + " note: WrongGPU's upscaler is still starting after " + std::to_string(int(seconds)) + " s; AMD FSR upscales meanwhile");
        return;
    }
    u.creating = false;
    const WgResult r = u.pending->Result();
    WgUpscaler* upscaler = u.pending->Take();
    char seconds[32];
    std::snprintf(seconds, sizeof seconds, "%.1f s", u.pending->Seconds());
    if (r == WG_OK && upscaler) {
        u.upscaler = upscaler;
        if (!ActivateSplit(u, u.pendingBackend, tag)) {
            // AMD's FSR keeps upscaling this context (as after a failed create); nothing of the hooks is left patched.
            u.pendingBackend->destroy(upscaler);
            u.upscaler = nullptr;
            u.failReason = "WrongGPU's D3D12 hooks could not be installed for the command-list split";
            Log(tag + " ERROR: upscaler \"" + u.pendingBackend->name + "\" started after " + seconds + ", but " + u.failReason +
                "; AMD FSR keeps upscaling");
            u.pending.reset();
            return;
        }
        u.backend = u.pendingBackend;
        u.mode = UpscaleContext::Mode::Own;
        u.compared = true;  // the network starts with a reset frame; AMD keeps its history for comparisons
        u.failReason.clear();
        Log(tag + ": WrongGPU's upscaler \"" + u.backend->name + "\" started after " + seconds + "; it upscales from this frame on");
    } else {
        u.failReason = u.createError.empty() ? std::string("upscaler \"") + u.pendingBackend->name + "\" create failed: " + WgResultText(r)
                                             : u.createError;
        Log(tag + " ERROR: upscaler \"" + u.pendingBackend->name + "\" create failed after " + seconds + ": " + WgResultText(r) +
            "; AMD FSR keeps upscaling");
        if (config_.customSlot) {
            std::lock_guard lock(mutex_);
            failedCreates_[FailureKey(u.pendingBackend, u)] = RememberedFailure{u.failReason, 0};
        }
    }
    u.pending.reset();
}

bool FrontEnd::ActivateSplit(UpscaleContext& u, const WgUpscalerBackend* backend, const std::string& tag) {
    if (!u.createInfo.services || !platform_.activateServices) return true;
    if (platform_.activateServices(backend, u.device)) return true;
    Log(tag + " note U" + std::to_string(u.id) + ": WrongGPU's D3D12 hooks could not be installed (the \"d3d12 hooks\" lines say why)");
    return false;
}

// Bug hunt F1 (a): the upscaler failed for good (u.broken) and AMD's FSR did not take the context over (none, or
// fallback_to_amd = 0): it is not called again (each call would fail and log again, every frame). In OptiScaler's custom
// slot the error goes back to the host, which makes a new context (Dispatch's run of failures decides when that stops);
// elsewhere the stand-in upscales the frame, or the frame fails. Logged at LogPoint counts.
ffxReturnCode_t FrontEnd::BrokenFrame(UpscaleContext& u, WgFrame& frame, const std::string& tag, uint64_t frameIndex) {
    const uint64_t n = ++u.brokenFrames;
    const std::string what = "frame " + std::to_string(frameIndex) + ": WrongGPU's upscaler failed earlier in this context (" + u.failReason +
                             "); it is not called again (" + std::to_string(n) + " frame(s) since)";
    if (!config_.customSlot && UsableForStandIn(frame)) {
        if (!u.fallbackBackend && platform_.standIn) u.fallbackBackend = platform_.standIn(config_);
        if (u.fallbackBackend && u.fallbackBackend != u.backend) {
            if (!u.fallback) {
                WgCreateInfo info = u.createInfo;
                AdaptCreateInfo(u.fallbackBackend, nullptr, info);
                if (u.fallbackBackend->create(&info, &u.fallback) != WG_OK) u.fallback = nullptr;
            }
            AdaptFrame(u.fallbackBackend, frame);
            if (u.fallback && u.fallbackBackend->dispatch(u.fallback, &frame) == WG_OK) {
                if (LogPoint(n)) Log(tag + " note U" + std::to_string(u.id) + ": " + what + "; the stand-in upscales");
                if (platform_.afterDispatch) platform_.afterDispatch(frame, true);
                Report(u, UpscaleState::StandIn, &frame);
                return FFX_API_RETURN_OK;
            }
        }
    }
    if (LogPoint(n)) Log(tag + " ERROR U" + std::to_string(u.id) + ": " + what + " -> ERROR_RUNTIME_ERROR");
    Report(u, UpscaleState::Failed, nullptr, u.failReason + "; AMD fallback is unavailable");
    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
}

// Bug hunt H: the game gives an output size (upscaleSize) other than the one the upscaler was created for (an upscaler's
// output size is fixed at create: every such frame was refused with WG_ERROR_PARAMETER). Within the context's
// maxUpscaleSize the upscaler is created again for the frame's size, here on the game's thread (as a create without
// AMD's FSR to upscale meanwhile); the outer FidelityFX context stays, the new upscaler starts without history. At most
// kMaxResizes times per context (a game that changes the size every few frames gets the old behaviour after that), and
// never again after a failed create (the old upscaler stays).
void FrontEnd::ResizeForFrame(UpscaleContext& u, WgFrame& frame, const std::string& tag) {
    constexpr uint32_t kMaxResizes = 8;
    if (!u.backend || !u.upscaler || u.pending || u.broken || u.resizeFailed) return;
    const uint32_t w = frame.output_width, h = frame.output_height;
    if ((w == u.createInfo.output_width && h == u.createInfo.output_height) || !w || !h) return;
    if (w > u.create.maxUpscaleSize.width || h > u.create.maxUpscaleSize.height) return;  // (refused as before)
    const std::string was = std::to_string(u.createInfo.output_width) + "x" + std::to_string(u.createInfo.output_height);
    const std::string now = std::to_string(w) + "x" + std::to_string(h);
    if (u.resizes >= kMaxResizes) {
        if (u.resizes++ == kMaxResizes)
            Log(tag + " note U" + std::to_string(u.id) + ": output size " + now + " again after " + std::to_string(kMaxResizes) +
                " sizes; the upscaler is not created again for this context (frames of another size are refused)");
        return;
    }
    WgCreateInfo info = u.createInfo;
    info.output_width = w;
    info.output_height = h;
    WgUpscaler* fresh = nullptr;
    u.creating = true;
    const WgResult r = u.backend->create(&info, &fresh);
    u.creating = false;
    if (r != WG_OK || !fresh) {
        if (fresh) u.backend->destroy(fresh);
        u.resizeFailed = true;
        Log(tag + " note U" + std::to_string(u.id) + ": output size " + now + " (the upscaler's: " + was + "): creating the upscaler for it failed: " +
            WgResultText(r) + "; frames of that size are refused (the stand-in or AMD's FSR takes them)");
        return;
    }
    u.backend->destroy(u.upscaler);
    u.upscaler = fresh;
    u.createInfo = info;
    if (u.fallback && u.fallbackBackend) {  // the stand-in is made again for the new size when it is needed
        u.fallbackBackend->destroy(u.fallback);
        u.fallback = nullptr;
    }
    ++u.resizes;
    frame.reset = 2;  // a new upscaler: no history
    u.resetNext = u.restartNext = false;
    Log(tag + " note U" + std::to_string(u.id) + ": output size " + now + " (the upscaler was created for " + was + ", the context's maximum is " +
        std::to_string(u.create.maxUpscaleSize.width) + "x" + std::to_string(u.create.maxUpscaleSize.height) +
        "): the upscaler is created again for it; the context stays");
}

std::string FrontEnd::FailureKey(const WgUpscalerBackend* backend, const UpscaleContext& u) const {
    return std::string(backend && backend->name ? backend->name : "?") + "|" + u.options;
}

void FrontEnd::ApplyModelDir(UpscaleContext& u, const std::string& tag) {
    bool inOptions = false, inEnv = false;
    OptionValue(u.options, "model_dir", &inOptions);
    const char* env = std::getenv("WG_RUNTIME_OPTIONS");
    if (env) OptionValue(env, "model_dir", &inEnv);
    const bool k = OwnModelK(u.options);
    bool kDir = false;
    if (k) {
        bool a = false, b = false;
        OptionValue(u.options, "model_k_dir", &a);
        if (env) OptionValue(env, "model_k_dir", &b);
        kDir = a || b;
    }
    std::string line;
    if (inOptions || inEnv || kDir) {
        line = std::string("model: ") + (kDir ? "model_k_dir" : "model_dir") + " from " + (inEnv ? "WG_RUNTIME_OPTIONS" : "runtime_options") +
               " decides (the front end's model lookup is not used)";
    } else {
        if (!platform_.modelPlaces) return;
        const ModelChoice c = ChooseModelDir(platform_.modelPlaces(), k, platform_.fileExists);
        u.options += c.option;
        const char* model = k ? "Model K" : "Model M";
        if (!c.option.empty())
            line = std::string("model: ") + model + " from " + c.file + " (passed to the runtime as model_dir); looked at: " + c.searched;
        else if (c.index >= 0)
            line = std::string("model: ") + model + " is not in the game's folders; the runtime looks in its own default place; looked at: " + c.searched;
        else
            line = std::string("model: no converted ") + model + " found; looked at: " + c.searched;
    }
    {
        std::lock_guard lock(mutex_);
        if (modelLogged_.count(line)) return;
        modelLogged_[line] = true;
    }
    Log(tag + " " + line);
}

void FrontEnd::Fail(UpscaleContext& u, const std::string& why, uint64_t call, bool error) {
    Log("#" + std::to_string(call) + (error ? " ERROR U" : " note U") + std::to_string(u.id) + ": " + why);
    if (u.failReason.empty()) u.failReason = why;
    if (!config_.fallbackToAmd) return;
    if (CreateAmdFromSaved(u, call) == FFX_API_RETURN_OK && u.upscaler) {
        // Dispatch can leave resources referenced by commands the host has not
        // submitted yet. A handoff is not a GPU completion boundary. Retain the
        // network and stand-in until the host tears down this context normally.
        Log("#" + std::to_string(call) + " U" + std::to_string(u.id) + " now runs on AMD FSR for the rest of its life");
    }
}

void FrontEnd::Report(const UpscaleContext& u, UpscaleState state, const WgFrame* frame, const std::string& failureReason,
                      const std::string& amdReason) {
    if (!platform_.status) return;
    UpscaleStatus s;
    s.state = state;
    s.upscaler = state == UpscaleState::Compare || state == UpscaleState::Amd ? (u.amdName.empty() ? std::string("AMD FSR") : u.amdName)
                                                : u.backend ? u.backend->name : "";
    if (state == UpscaleState::Network && config_.upscaler == UpscalerChoice::Network)
        s.upscaler = std::string("WrongGPU network ") + u.modelProviderName;
    if (state == UpscaleState::Failed) {
        s.upscaler = "Upscaling unavailable";
        s.reason = !failureReason.empty() ? failureReason : !u.failReason.empty() ? u.failReason : "No upscaler produced this frame";
    }
    if (state == UpscaleState::Amd && !amdReason.empty())
        s.reason = amdReason;
    else if (state == UpscaleState::Amd)
        s.reason = u.pending ? "WrongGPU's upscaler is starting (" + std::to_string(int(u.pending->Seconds())) + " s)"
                   : u.failReason.empty() ? "WrongGPU's upscaler failed (see wronggpu-fsr.log)" : u.failReason;
    if (!s.reason.empty() &&
        (state == UpscaleState::Failed || (state == UpscaleState::Amd && !u.pending && (!u.failReason.empty() || !amdReason.empty()))))
        s.reason += " [" + std::string(wgdiag::Classify(s.reason).code) + "]";
    if (frame) {
        s.renderWidth = frame->render_width;
        s.renderHeight = frame->render_height;
        s.outputWidth = frame->output_width;
        s.outputHeight = frame->output_height;
    } else {
        s.renderWidth = u.create.maxRenderSize.width;
        s.renderHeight = u.create.maxRenderSize.height;
        s.outputWidth = u.create.maxUpscaleSize.width;
        s.outputHeight = u.create.maxUpscaleSize.height;
    }
    s.device = u.device;
    platform_.status(s);
}

// ---------------------------------------------------------------------------------------------------------------
// ffxDestroyContext

ffxReturnCode_t FrontEnd::DestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    EntryScope scope;
    const uint64_t call = ++calls_;
    const std::string tag = "#" + std::to_string(call);
    Log(tag + " ffxDestroyContext context=" + Name(context) + " memCb=" + Pointer(memCb));
    if (!context || !*context) {
        Log(tag + " -> ERROR_PARAMETER");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (UpscaleContext* u = FindOwn(context)) {
        if (!scope.Enter(u)) {  // destroyed from inside a call of itself on this thread (see ThreadCalls)
            ++reentered_;
            Log(tag + " -> ERROR_RUNTIME_ERROR: U" + std::to_string(u->id) + " is destroyed from inside a call of itself on this thread; refused");
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        bool others = false;  // another upscaler context is in use: the host switched upscalers (frontend.h, parked)
        {
            std::lock_guard lock(mutex_);
            own_.erase(u);
            others = OtherUpscalerLocked();
        }
        ffxReturnCode_t rc = FFX_API_RETURN_OK;
        std::shared_ptr<PendingCreate> pending;
        bool park = false;
        bool switched = false;  // it could have been parked, but another upscaler context is in use
        {
            std::lock_guard lock(u->mutex);
            pending = std::move(u->pending);
            // A working upscaler is parked for the next context with the same parameters (frontend.h), with its
            // stand-in; AMD's contexts always go now.
            park = !pending && CanPark(*u);
            switched = park && others;
            park = park && !others;
            if (!park) {
                if (u->upscaler) u->backend->destroy(u->upscaler);
                u->upscaler = nullptr;
                if (u->fallback) u->fallbackBackend->destroy(u->fallback);
                u->fallback = nullptr;
            }
            DestroyComparison(u->comparison);
            if (u->amd) {
                const OriginalApi* original = u->amdApi;
                rc = original ? original->destroyContext(&u->amd, u->amdFromSaved ? nullptr : memCb) : ffxReturnCode_t(FFX_API_RETURN_NO_PROVIDER);
                if (park) u->amd = nullptr;
            }
        }
        Log(tag + " -> " + RcName(rc) + " (U" + std::to_string(u->id) + " destroyed after " + std::to_string(u->dispatches) +
            " dispatches" + (pending && !pending->Done() ? "; its upscaler is still starting and is destroyed when it has" : "") +
            (park ? "; its upscaler is parked for " + std::to_string(config_.contextParkMs) +
                        " ms (context_park_ms) for a context with the same parameters"
                  : switched ? "; its upscaler is destroyed now, not parked: another upscaler context is in use (the host switched "
                               "upscalers)"
                             : std::string()) +
            ")");
        *context = nullptr;
        if (park) {
            Park(u, tag);
        } else if (pending) {  // the create's thread may still use the context (create info, log callback)
            const WgUpscalerBackend* backend = u->pendingBackend;
            PendingCreate::Abandon(pending, [u, backend](WgUpscaler* upscaler) {
                if (upscaler) backend->destroy(upscaler);
                delete u;
            });
        } else {
            delete u;
        }
        return rc;
    }
    const OriginalApi* original = ApiOf(*context);
    if (!original) {
        Log(tag + " -> NO_PROVIDER");
        return FFX_API_RETURN_NO_PROVIDER;
    }
    void* key = *context;
    const ffxReturnCode_t rc = original->destroyContext(context, memCb);
    {
        std::lock_guard lock(mutex_);
        forwarded_.erase(key);
    }
    Log(tag + " -> " + RcName(rc) + " (forwarded to AMD)");
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// ffxConfigure

ffxReturnCode_t FrontEnd::Configure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    EntryScope scope;
    const uint64_t call = ++calls_;
    const std::string tag = "#" + std::to_string(call);
    if (!desc) {
        Log(tag + " ffxConfigure context=" + Name(context) + " (no description) -> ERROR_PARAMETER");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // Bug hunt F2: frame generation configures every frame: like queries, only the first calls of each description type
    // are logged unless log_dispatches = 1; failed calls always (with their description).
    uint64_t seen;
    {
        std::lock_guard lock(mutex_);
        seen = configureCounts_[desc->type]++;
    }
    const bool log = seen < 3 || config_.logDispatches;
    const std::string head = tag + " ffxConfigure context=" + Name(context) + " ";
    if (log) Log(head + DescribeChain(desc));
    auto result = [&](ffxReturnCode_t rc, const std::string& who) {
        if (log || rc != FFX_API_RETURN_OK) {
            if (!log) Log(head + DescribeChain(desc));
            Log(tag + " -> " + RcName(rc) + who);
        }
        return rc;
    };
    UpscaleContext* u = FindOwn(context);
    if (u && !scope.Enter(u)) {
        ++reentered_;
        if (!log) Log(head + DescribeChain(desc));
        Log(tag + " -> ERROR_RUNTIME_ERROR: U" + std::to_string(u->id) + " re-entered on this thread from inside a call of itself; refused");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    if (u && u->mode == UpscaleContext::Mode::Own) {
        std::lock_guard lock(u->mutex);
        ffxReturnCode_t rc = FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
        if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE) {
            rc = FFX_API_RETURN_OK;  // FSR tuning keys; WrongGPU's upscaler has no equivalent
        } else if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1) {
            u->fpMessage = reinterpret_cast<const ffxConfigureDescGlobalDebug1*>(desc)->fpMessage;
            rc = FFX_API_RETURN_OK;
        }
        return result(rc, " (WrongGPU)");
    }
    const OriginalApi* original = u ? u->amdApi : !context ? Original() : ApiOf(*context);
    if (!original) return result(FFX_API_RETURN_NO_PROVIDER, "");
    const ffxReturnCode_t rc = u ? original->configure(&u->amd, desc) : original->configure(context, desc);
    return result(rc, " (forwarded to AMD)");
}

// ---------------------------------------------------------------------------------------------------------------
// ffxQuery

ffxReturnCode_t FrontEnd::QueryOwn(UpscaleContext& u, ffxQueryDescHeader* desc) {
    switch (desc->type) {
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
        auto* q = reinterpret_cast<ffxQueryGetProviderVersion*>(desc);
        q->versionId = u.modelProviderId;
        q->versionName = config_.upscaler == UpscalerChoice::Network ? u.modelProviderName
                            : u.backend ? u.backend->name : "WrongGPU";
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(desc);
        const float ratio = UpscaleRatioFromQualityMode(q->qualityMode);
        if (!q->pOutUpscaleRatio || ratio == 0.0f) return FFX_API_RETURN_ERROR_PARAMETER;
        *q->pOutUpscaleRatio = ratio;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(desc);
        const float ratio = UpscaleRatioFromQualityMode(q->qualityMode);
        if (!q->pOutRenderWidth || !q->pOutRenderHeight || ratio == 0.0f) return FFX_API_RETURN_ERROR_PARAMETER;
        *q->pOutRenderWidth = std::max<uint32_t>(1u, uint32_t(float(q->displayWidth) / ratio));
        *q->pOutRenderHeight = std::max<uint32_t>(1u, uint32_t(float(q->displayHeight) / ratio));
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(desc);
        if (!q->pOutPhaseCount) return FFX_API_RETURN_ERROR_PARAMETER;
        *q->pOutPhaseCount = JitterPhaseCount(q->renderWidth, q->displayWidth);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(desc);
        if (!q->pOutX || !q->pOutY) return FFX_API_RETURN_ERROR_PARAMETER;
        JitterOffset(q->index, q->phaseCount, q->pOutX, q->pOutY);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsage*>(desc);
        if (!q->gpuMemoryUsageUpscaler) return FFX_API_RETURN_ERROR_PARAMETER;
        q->gpuMemoryUsageUpscaler->totalUsageInBytes =
            u.backend && u.backend->gpu_memory_bytes && u.upscaler ? u.backend->gpu_memory_bytes(u.upscaler) : 0;
        q->gpuMemoryUsageUpscaler->aliasableUsageInBytes = 0;
        return FFX_API_RETURN_OK;
    }
    default:
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

const OriginalApi* FrontEnd::ApiOf(void* context) {
    {
        std::lock_guard lock(mutex_);
        auto it = forwarded_.find(context);
        if (it != forwarded_.end() && it->second.api) return it->second.api;
    }
    return Original();
}

bool FrontEnd::ConfiguredModel(int32_t* model, std::string* described) {
    if (!platform_.menu.get) return false;
    bool isDefault = true;
    std::string text;
    const int32_t value = platform_.menu.get(WG_OPT_SR_MODEL, &text, &isDefault);
    if (isDefault || !SrEntryOfModel(value)) return false;
    *model = value;
    if (described) *described = text;
    return true;
}

void FrontEnd::ApplyEntry(const WgEntryDef& e, UpscaleContext& u, const std::string& tag, const char* source) {
    int32_t model = e.slot == WG_OPT_SR_MODEL ? e.value : -1;
    std::string described;
    if (model < 0 && ConfiguredModel(&model, &described))
        Log(tag + " settings: " + described + " (the SR context's model)");
    if (const WgEntryDef* sr = model >= 0 ? SrEntryOfModel(model) : nullptr) {
        u.selectedModel = kWgModelRows[model].name[0];
        u.modelOption = kWgModelRows[model].runtime_option;
        u.modelProviderId = sr->id;
        u.modelProviderName = EntryProviderName(*sr);
    } else {
        // An RR entry with no configured model: the model of runtime_options / the runtime's default, as before.
        for (int i = 0; i < OwnEntryCount(); ++i)
            if (OwnEntry(i)->slot == WG_OPT_SR_MODEL && std::strcmp(EntryProviderName(*OwnEntry(i)), u.modelProviderName) == 0)
                u.modelProviderId = OwnEntry(i)->id;
    }
    const bool written = platform_.menu.set && platform_.menu.set(e.slot, e.value, source);
    wg::Source word = std::strcmp(source, "folder") == 0 ? wg::Source::Sidecar : wg::Source::Menu;
    Log(tag + " the host chose " + EntryProviderName(e) + " (" + Hex(e.id) + "): settings: " + wg::describe(e.slot, e.value, word) +
        (written ? std::string() : platform_.menu.set ? " (not written: another module writes the shared settings)" : std::string()) +
        (e.slot == WG_OPT_SR_MODEL ? std::string() : std::string("; the SR context runs ") + u.modelProviderName));
}

ffxReturnCode_t FrontEnd::ListVersions(ffxQueryDescGetVersions& q) {
    if (!q.outputCount) return FFX_API_RETURN_ERROR_PARAMETER;
    // Listed names start with a version number the host can read (frontend.h, kWrongGpuListedNameM). WrongGPU stays
    // listed first also when the OptiScaler plugin (clean/fsr/asi) has hooked the host's main list: the host reads the
    // DLL's version from the first name, and the sub-box indexes are the same with and without the plugin.
    // WrongGPU's entries come from the registry (config.h OwnEntry: SR models, then RR modes); one folder per choice
    // (MenuSettings::folderEntry) lists that folder's entry alone.
    std::vector<uint64_t> ids;
    std::vector<const char*> names;
    std::vector<const OriginalApi*> apis;
    const WgEntryDef* folder = platform_.menu.folderEntry ? OwnEntryOfId(platform_.menu.folderEntry) : nullptr;
    for (int i = 0; i < OwnEntryCount(); ++i) {
        if (folder && OwnEntry(i) != folder) continue;
        ids.push_back(OwnEntry(i)->id);
        names.push_back(OwnEntry(i)->label);
        apis.push_back(nullptr);
    }
    if (folder) {
        {
            std::lock_guard lock(mutex_);
            versionNames_[folder->id] = folder->label;
        }
        if (*q.outputCount != 0 && (q.versionIds || q.versionNames)) {
            if (q.versionIds) q.versionIds[0] = folder->id;
            if (q.versionNames) q.versionNames[0] = folder->label;
        }
        *q.outputCount = 1;
        return FFX_API_RETURN_OK;
    }
    // A DLL that answers by calling back into WrongGPU (a host mod's hook) only adds nothing, never loops.
    static thread_local bool listing = false;
    const OriginalApi* candidates[2] = {nullptr, nullptr};
    struct Guard {
        bool& flag;
        bool was;
        ~Guard() { flag = was; }
    } guard{listing, listing};
    listing = true;
    if (!guard.was) {
        // Custom slot: AMD's entries only from an AMD DLL of WrongGPU's own (Original()), never from the host's
        // FidelityFX DLLs (config.h Config::customSlot: choosing such an entry would route the context back here).
        candidates[0] = platform_.compareApi && !config_.customSlot ? platform_.compareApi() : nullptr;
        candidates[1] = Original();
    }
    for (int c = 0; c < 2; ++c) {
        const OriginalApi* api = candidates[c];
        if (!api || !api->query || !api->createContext || (c == 1 && candidates[0] && candidates[0]->query == api->query)) continue;
        uint64_t count = 0;
        ffxQueryDescGetVersions v{};
        v.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        v.createDescType = q.createDescType;
        v.device = q.device;
        v.outputCount = &count;
        if (api->query(nullptr, &v.header) != FFX_API_RETURN_OK || count == 0) continue;
        uint64_t vid[16] = {};
        const char* vname[16] = {};
        count = std::min<uint64_t>(count, 16);
        v.versionIds = vid;
        v.versionNames = vname;
        if (api->query(nullptr, &v.header) != FFX_API_RETURN_OK) continue;
        for (uint64_t i = 0; i < count && i < 16; ++i) {
            if (std::find(ids.begin(), ids.end(), vid[i]) != ids.end()) continue;
            ids.push_back(vid[i]);
            names.push_back(vname[i] ? vname[i] : "AMD FSR");
            apis.push_back(api);
        }
    }
    {
        std::lock_guard lock(mutex_);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (apis[i]) versionApi_[ids[i]] = apis[i];  // only AMD ids route a forwarded context to a DLL
            versionNames_[ids[i]] = names[i];
        }
    }
    if (!guard.was && !versionsLogged_.exchange(true)) {  // once: which AMD DLL gives AMD's versions, and which they are
        std::string amd;
        const OriginalApi* provider = nullptr;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!apis[i]) continue;
            if (!provider) provider = apis[i];
            amd += std::string(amd.empty() ? "" : ", ") + names[i] + " (" + Hex(ids[i]) + ")";
        }
        if (provider)
            Log("AMD provider for the host's upscaler list: " + (provider->source.empty() ? std::string("AMD's original DLL") : provider->source) +
                "; AMD's versions: " + amd);
        else
            Log(std::string("FSR 4 / AMD versions not available: ") +
                (candidates[0] || candidates[1] ? "the AMD FidelityFX DLL listed no upscaler versions for this device"
                 : config_.customSlot ? "custom slot: no AMD DLL of WrongGPU's own next to this DLL (the game's FidelityFX DLLs are not used here)"
                                      : "no usable AMD FidelityFX DLL (see the \"AMD provider\" lines above)") +
                "; the host and the game see only WrongGPU");
    }
    const uint64_t total = ids.size();
    if (*q.outputCount == 0 || (!q.versionIds && !q.versionNames)) {
        *q.outputCount = total;
        return FFX_API_RETURN_OK;
    }
    const uint64_t n = std::min(*q.outputCount, total);
    for (uint64_t i = 0; i < n; ++i) {
        if (q.versionIds) q.versionIds[i] = ids[i];
        if (q.versionNames) q.versionNames[i] = names[i];
    }
    *q.outputCount = n;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t FrontEnd::Query(ffxContext* context, ffxQueryDescHeader* desc) {
    EntryScope scope;
    const uint64_t call = ++calls_;
    const std::string tag = "#" + std::to_string(call);
    if (!desc) {
        Log(tag + " ffxQuery context=" + Name(context) + " (no description) -> ERROR_PARAMETER");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // Games query the jitter (and more) every frame: like dispatches, only the first calls of each query type are logged
    // unless log_dispatches = 1; failed queries always.
    uint64_t seen;
    {
        std::lock_guard lock(mutex_);
        seen = queryCounts_[desc->type]++;
    }
    const bool log = seen < 3 || config_.logDispatches;
    const std::string head = tag + " ffxQuery context=" + Name(context) + " " + (log ? DescribeChain(desc) : std::string());
    if (log) Log(head);
    UpscaleContext* u = FindOwn(context);
    if (u && !scope.Enter(u)) {
        ++reentered_;
        if (!log) Log(head + DescribeChain(desc));
        Log(tag + " -> ERROR_RUNTIME_ERROR: U" + std::to_string(u->id) + " re-entered on this thread from inside a call of itself; refused");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    ffxReturnCode_t rc;
    const char* who;
    if (config_.hosted && !u && (!context || !*context) && desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS &&
        reinterpret_cast<ffxQueryDescGetVersions*>(desc)->createDescType == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
        rc = ListVersions(*reinterpret_cast<ffxQueryDescGetVersions*>(desc));
        who = "WrongGPU, then AMD's upscaler versions";
    } else if (u && u->mode == UpscaleContext::Mode::Own) {
        std::lock_guard lock(u->mutex);
        rc = QueryOwn(*u, desc);
        who = "WrongGPU";
    } else if (const OriginalApi* original = u ? u->amdApi : !context || !*context ? Original() : ApiOf(*context)) {
        rc = u ? original->query(&u->amd, desc) : original->query(context, desc);
        who = "forwarded to AMD";
    } else {
        rc = FFX_API_RETURN_NO_PROVIDER;
        who = "AMD's original DLL is not available";
    }
    if (log || rc != FFX_API_RETURN_OK) {
        const std::string outputs = rc == FFX_API_RETURN_OK ? DescribeQueryOutputs(desc) : std::string();
        if (!log) Log(head + DescribeChain(desc));
        Log(tag + " -> " + RcName(rc) + " (" + who + ")" + (outputs.empty() ? "" : " " + outputs));
    }
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// ffxDispatch

ffxReturnCode_t FrontEnd::DispatchUpscale(UpscaleContext& u, const ffxDispatchDescUpscale& d, uint64_t call) {
    const std::string tag = "#" + std::to_string(call);
    const uint64_t frameIndex = u.dispatches++;
    const bool log = ShouldLogDispatch(frameIndex);
    WgFrame frame;
    std::string problem, notes;
    if (d.flags & (FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB | FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ)) {
        // Perceptual input goes to AMD's FSR (spec 20, ANSWERS item 5): this frame, and the context for good when the
        // game keeps flagging it (kBadFrameRun).
        return BadFrame(u, d, call, frameIndex, "non-linear (sRGB/PQ) colour flagged by the game: AMD FSR handles it", nullptr, true,
                        WG_ERROR_UNSUPPORTED);
    }
    const bool mapped = MapDispatch(d, u.create, config_, platform_.resourceInfo, u.map, frameIndex, frame, problem, notes);
    const bool resetAsked = frame.reset != 0;
    if (mapped && u.compared) {  // back from AMD (comparison): the history is old
        frame.reset = 1;
        u.compared = false;
    }
    // A frame went past the upscaler (stand-in, AMD's FSR), or it was parked: its history is old until it takes a frame.
    if (mapped && u.resetNext) frame.reset = u.restartNext ? 2u : 1u;
    if (frame.command_list && platform_.noteDispatch) platform_.noteDispatch(frame.command_list, &frame);
    if (log || (!mapped && u.badRun == 0)) Log(tag + " WrongGPU frame: " + DescribeFrame(frame) + (notes.empty() ? "" : " notes: " + notes));
    WgResult r = WG_ERROR_PARAMETER;
    if (mapped) {
        // A list the front end cannot split (another hook layer owns its vtable): a backend that needs the split would
        // never get a frame through it. AMD's FSR takes the context over; when it cannot (no AMD DLL, fallback_to_amd
        // = 0) the split answers WG_ERROR_NOT_READY and the stand-in upscales each frame.
        if (!u.splitBlocked && platform_.splitBlocked && u.backend && (BackendFlags(u.backend) & WG_BACKEND_NEEDS_SPLIT)) {
            if (const char* why = platform_.splitBlocked(frame.command_list)) {
                u.splitBlocked = true;
                Fail(u, "frame " + std::to_string(frameIndex) + ": " + why + "; AMD FSR upscales this context instead of WrongGPU", call, false);
                if (u.mode == UpscaleContext::Mode::Amd) {
                    const ffxReturnCode_t rc = u.amdApi->dispatch(&u.amd, &d.header);
                    Log(tag + " -> " + RcName(rc) + " (forwarded to AMD)");
                    Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Amd : UpscaleState::Failed, nullptr,
                           "AMD FSR dispatch failed: " + std::string(RcName(rc)));
                    return rc;
                }
                Log(tag + " note U" + std::to_string(u.id) + ": AMD FSR is not available for this context; the stand-in upscales its frames");
            }
        }
        if (u.broken) return BrokenFrame(u, frame, tag, frameIndex);  // bug hunt F1: the failed upscaler is not called again
        ResizeForFrame(u, frame, tag);
        if (platform_.diagnosticFrame) platform_.diagnosticFrame(u.id, frame);
        if (!u.fallbackBackend && platform_.standIn) u.fallbackBackend = platform_.standIn(config_);
        std::string note;
        r = DispatchWithFallback(u.backend, u.upscaler, frame, u.fallbackBackend, u.createInfo, &u.fallback, note);
        if (platform_.diagnosticResult) platform_.diagnosticResult(u.id, frameIndex, r);
        if (!note.empty()) Log(tag + " note U" + std::to_string(u.id) + ": " + note);
        if (r == WG_OK) {
            if (note.empty()) u.resetNext = u.restartNext = false;  // the upscaler took the frame (with the reset, if one was due)
            if (u.badRun) {
                Log(tag + " note U" + std::to_string(u.id) + ": frame " + std::to_string(frameIndex) + ": WrongGPU upscales again after " +
                    std::to_string(u.badRun) + " frame(s) it could not take (history reset)");
                u.badRun = 0;
            }
            u.amdLast = false;
            if (inputRefusedAny_.load(std::memory_order_relaxed)) {  // a taken frame ends a run of input refusals (bug hunt A)
                std::lock_guard lock(mutex_);
                inputRefusedRun_ = 0;
                inputRefusedKey_.clear();
                inputRefusedAny_.store(false, std::memory_order_relaxed);
            }
            if (platform_.afterDispatch) platform_.afterDispatch(frame, !note.empty());
            if (log) Log(tag + " -> OK (WrongGPU)");
            Report(u, note.empty() ? UpscaleState::Network : UpscaleState::StandIn, &frame);
            return FFX_API_RETURN_OK;
        }
        problem = std::string("upscaler \"") + u.backend->name + "\" dispatch failed: " + WgResultText(r);
        if (platform_.diagnosticFailure) platform_.diagnosticFailure(u.id, frameIndex, u.device);
        // clean/fe-wrapped-list: WG_ERROR_UNSUPPORTED from the upscaler itself: it does not take this context's frames
        // (the network runtime with hip_switch=off when its D3D12 executor refuses a frame, e.g. native resolution).
        // AMD's FSR takes the context over now; without it the stand-in upscales such frames (below) and the context
        // stays the upscaler's. Said once per context.
        if (r == WG_ERROR_UNSUPPORTED && !u.unsupportedSaid) {
            u.unsupportedSaid = true;
            Fail(u, "frame " + std::to_string(frameIndex) + ": " + problem + " (the upscaler does not take this frame); AMD FSR upscales this context instead of WrongGPU",
                 call, false);
            if (u.mode == UpscaleContext::Mode::Amd) {
                ffxDispatchDescUpscale amd = d;
                amd.reset = true;  // AMD's history starts here
                const ffxReturnCode_t rc = u.amdApi->dispatch(&u.amd, &amd.header);
                Log(tag + " -> " + RcName(rc) + " (forwarded to AMD)");
                Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Amd : UpscaleState::Failed, nullptr,
                       "AMD FSR dispatch failed: " + std::string(RcName(rc)));
                return rc;
            }
            Log(tag + " note U" + std::to_string(u.id) + ": AMD FSR is not available for this context; the stand-in upscales the frames the upscaler does not take");
        }
    }
    // Inputs the upscaler cannot take (the mapping's checks, or the upscaler's own: WG_ERROR_PARAMETER,
    // WG_ERROR_UNSUPPORTED) are one frame's matter (BadFrame); every other result means the upscaler is broken.
    if (!mapped || r == WG_ERROR_PARAMETER || r == WG_ERROR_UNSUPPORTED) {
        frame.reset = resetAsked ? 1u : 0u;  // (for the stand-in: as the game gave it)
        return BadFrame(u, d, call, frameIndex, problem, &frame, false, mapped ? r : WG_ERROR_PARAMETER);
    }
    u.broken = true;
    Fail(u, "frame " + std::to_string(frameIndex) + ": " + problem, call);
    // 0.6.0 follow-up (a tester's CONTROL Resonant: 393 contexts in 3 minutes, each failing its first frame the same way):
    // the game's inputs can never be taken (WG_ERROR_INPUT). In OptiScaler's custom slot the host makes a new context
    // after every failed dispatch. After kInputRefusedRun such refusals in a row with the same words and no frame taken
    // between them (bug hunt A: one menu or loading frame with an odd texture is not enough), remembered for the session
    // as a failed create, so the next create fails at once and OptiScaler's own upscaler takes over (said once). Before
    // that, the host's next context is a new attempt. Other failures keep today's behaviour.
    // Bug hunt F1 (b): every other failure of the upscaler (device, HIP, WG_ERROR) the same way, with the same scoping.
    if (config_.customSlot && u.backend) {
        bool first = false;
        uint32_t run = 0;
        {
            std::lock_guard lock(mutex_);
            const std::string key = FailureKey(u.backend, u) + "|" + problem;
            run = key == inputRefusedKey_ ? inputRefusedRun_ + 1 : 1;
            inputRefusedKey_ = key;
            inputRefusedRun_ = run;
            inputRefusedAny_.store(true, std::memory_order_relaxed);
            if (run >= kInputRefusedRun) first = failedCreates_.emplace(FailureKey(u.backend, u), RememberedFailure{u.failReason, 0}).second;
        }
        if (first)
            Log(tag + (r == WG_ERROR_INPUT ? " custom slot: the game's inputs are refused for good (" : " custom slot: WrongGPU's upscaler fails the same way (") +
                problem + ", " + std::to_string(run) + " contexts in a row); this failure is remembered for the session: a create of upscaler \"" +
                u.backend->name + "\" with the same options fails at once from now on, so OptiScaler's own upscaler takes over");
    }
    if (u.mode == UpscaleContext::Mode::Amd) {
        const ffxReturnCode_t rc = u.amdApi->dispatch(&u.amd, &d.header);
        Log(tag + " -> " + RcName(rc) + " (forwarded to AMD after the failure)");
        Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Amd : UpscaleState::Failed, nullptr,
               u.failReason + "; AMD FSR dispatch failed: " + RcName(rc));
        return rc;
    }
    Report(u, UpscaleState::Failed, nullptr, u.failReason + "; AMD fallback is unavailable");
    Log(tag + " -> ERROR_RUNTIME_ERROR");
    return r == WG_ERROR_PARAMETER ? FFX_API_RETURN_ERROR_PARAMETER : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
}

ffxReturnCode_t FrontEnd::BadFrame(UpscaleContext& u, const ffxDispatchDescUpscale& d, uint64_t call, uint64_t frameIndex,
                                   const std::string& problem, WgFrame* frame, bool amdOnly, WgResult result) {
    const std::string tag = "#" + std::to_string(call);
    const std::string what = "frame " + std::to_string(frameIndex) + ": " + problem;
    const uint32_t run = ++u.badRun;
    if (!frame && d.commandList && platform_.noteDispatch) platform_.noteDispatch(d.commandList, nullptr);
    if (run == kBadFrameRun) {
        // Not one frame's matter any more: AMD's FSR takes the context over, as after a failure. (Without AMD's FSR the
        // frames go on as below.)
        Fail(u, what + " (" + std::to_string(kBadFrameRun) + " frames in a row)", call, !amdOnly);
        if (u.mode == UpscaleContext::Mode::Amd) {
            ffxDispatchDescUpscale amd = d;
            if (!u.amdLast) amd.reset = true;
            const ffxReturnCode_t rc = u.amdApi->dispatch(&u.amd, &amd.header);
            Log(tag + " -> " + RcName(rc) + " (forwarded to AMD)");
            Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Amd : UpscaleState::Failed, nullptr,
                   u.failReason + "; AMD FSR dispatch failed: " + RcName(rc));
            return rc;
        }
    }
    u.resetNext = true;  // this frame goes past the upscaler: its next one starts without history
    // 1. The stand-in, on the frame as it is or cut to its textures.
    if (!amdOnly && frame && UsableForStandIn(*frame)) {
        if (!u.fallbackBackend && platform_.standIn) u.fallbackBackend = platform_.standIn(config_);
        if (u.fallbackBackend && u.fallbackBackend != u.backend) {
            if (!u.fallback) {
                WgCreateInfo info = u.createInfo;
                AdaptCreateInfo(u.fallbackBackend, nullptr, info);
                if (u.fallbackBackend->create(&info, &u.fallback) != WG_OK) u.fallback = nullptr;
            }
            AdaptFrame(u.fallbackBackend, *frame);
            if (u.fallback && u.fallbackBackend->dispatch(u.fallback, frame) == WG_OK) {
                if (run == 1)
                    Log(tag + " note U" + std::to_string(u.id) + ": " + what + ": the stand-in upscales this frame (" +
                        std::to_string(frame->render_width) + "x" + std::to_string(frame->render_height) + "); the context stays WrongGPU's (AMD FSR takes it over after " +
                        std::to_string(kBadFrameRun) + " such frames in a row)");
                u.amdLast = false;
                if (platform_.afterDispatch) platform_.afterDispatch(*frame, true);
                Report(u, UpscaleState::StandIn, frame);
                return FFX_API_RETURN_OK;
            }
        }
    }
    // 2. AMD's FSR, for this frame only: the context made from the saved create parameters, kept for later ones.
    if (config_.fallbackToAmd && CreateAmdFromSaved(u, call, false) == FFX_API_RETURN_OK) {
        ffxDispatchDescUpscale amd = d;
        if (!u.amdLast) amd.reset = true;  // AMD's history is old, or never started
        u.amdLast = true;
        const ffxReturnCode_t rc = u.amdApi->dispatch(&u.amd, &amd.header);
        if (run == 1 || rc != FFX_API_RETURN_OK)
            Log(tag + " note U" + std::to_string(u.id) + ": " + what + ": AMD FSR upscales this frame -> " + RcName(rc) +
                "; the context stays WrongGPU's (AMD FSR takes it over after " + std::to_string(kBadFrameRun) + " such frames in a row)");
        Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Amd : UpscaleState::Failed, nullptr,
               what + "; AMD FSR dispatch failed: " + RcName(rc), what);
        return rc;
    }
    // 3. Nobody: the frame fails, the context stays.
    if (run == 1 || ShouldLogDispatch(frameIndex))
        Log(tag + (amdOnly ? " note U" : " ERROR U") + std::to_string(u.id) + ": " + what + ": no upscaler for this frame (" +
            (config_.fallbackToAmd ? "AMD FSR is not available" : "fallback_to_amd = 0") + ") -> " +
            (result == WG_ERROR_PARAMETER ? "ERROR_PARAMETER" : "ERROR_RUNTIME_ERROR"));
    Report(u, UpscaleState::Failed, nullptr, what + "; AMD fallback is unavailable");
    return result == WG_ERROR_PARAMETER ? FFX_API_RETURN_ERROR_PARAMETER : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
}

// The player compares with AMD's FSR: the frame goes to an AMD context made for the comparison (once, from the saved
// create parameters); WrongGPU's upscaler keeps its resources and resets when the player switches back.
ffxReturnCode_t FrontEnd::DispatchCompare(UpscaleContext& u, const ffxDispatchDescUpscale& d, uint64_t call) {
    const std::string tag = "#" + std::to_string(call);
    if (!u.comparison.context && !u.comparison.unavailable) {
        const OriginalApi* api = platform_.compareApi ? platform_.compareApi() : nullptr;
        std::string why;
        if (api && CreateComparison(u.comparison, api, u.create, u.device, u.fpMessage, why)) {
            u.amdName = u.comparison.name;
            Log(tag + " U" + std::to_string(u.id) + " compares with " + u.comparison.name + " (" + why + ")");
        } else {
            u.comparison.unavailable = true;
            if (api) Log(tag + " note U" + std::to_string(u.id) + ": comparison DLL not usable: " + why + "; the game's AMD FSR compares");
        }
    }
    if (u.comparison.context) {
        const bool reset = !u.compared;
        u.compared = true;
        ++u.dispatches;
        const ffxReturnCode_t rc = DispatchComparison(u.comparison, d, reset);
        if (ShouldLogDispatch(u.dispatches) || rc != FFX_API_RETURN_OK) Log(tag + " -> " + RcName(rc) + " (" + u.comparison.name + ", comparison)");
        Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Compare : UpscaleState::Failed, nullptr,
               "AMD comparison dispatch failed: " + std::string(RcName(rc)));
        return rc;
    }
    if (!u.amd && !u.amdUnavailable && CreateAmdFromSaved(u, call, false) != FFX_API_RETURN_OK) {
        u.amdUnavailable = true;
        Log(tag + " note U" + std::to_string(u.id) + ": no AMD FSR context for the comparison; WrongGPU keeps upscaling");
    }
    if (u.amd && u.amdName.empty()) {  // which AMD upscaler compares (FSR 3.1, FSR 4 ...), for the overlay
        ffxQueryGetProviderVersion provider{};
        provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        const char* name = u.amdApi->query(&u.amd, &provider.header) == FFX_API_RETURN_OK ? provider.versionName : nullptr;
        u.amdName = std::string("AMD FSR") + (name && *name ? std::string(" ") + name : std::string());
        Log(tag + " U" + std::to_string(u.id) + " compares with " + u.amdName);
    }
    if (!u.amd) return DispatchUpscale(u, d, call);
    ffxDispatchDescUpscale amd = d;
    if (!u.compared) amd.reset = true;  // AMD's history is old (or never started)
    u.compared = true;
    ++u.dispatches;
    const ffxReturnCode_t rc = u.amdApi->dispatch(&u.amd, &amd.header);
    if (ShouldLogDispatch(u.dispatches) || rc != FFX_API_RETURN_OK) Log(tag + " -> " + RcName(rc) + " (AMD, comparison)");
    Report(u, rc == FFX_API_RETURN_OK ? UpscaleState::Compare : UpscaleState::Failed, nullptr,
           "AMD comparison dispatch failed: " + std::string(RcName(rc)));
    return rc;
}

ffxReturnCode_t FrontEnd::Dispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    EntryScope scope;
    const uint64_t call = ++calls_;
    const std::string tag = "#" + std::to_string(call);
    UpscaleContext* u = FindOwn(context);
    if (u && !scope.Enter(u)) {
        // This context's dispatch (or another call of it) is in progress on this thread and holds its lock: a host that
        // routes AMD's DLL back to WrongGPU dispatched it again from inside (see ThreadCalls).
        const uint64_t n = ++reentered_;
        if (LogPoint(n))
            Log(tag + " ffxDispatch context=" + Name(context) + " -> ERROR_RUNTIME_ERROR: re-entered on this thread from inside a call of "
                "the same context; refused (refusal " + std::to_string(n) + ")");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    uint64_t count = u ? u->dispatches : 0;
    if (!u && context && *context) {
        std::lock_guard lock(mutex_);
        auto it = forwarded_.find(*context);
        if (it != forwarded_.end()) count = it->second.dispatches++;
    }
    const bool log = ShouldLogDispatch(count) || !desc;
    if (log) Log(tag + " ffxDispatch context=" + Name(context) + " " + DescribeChain(desc));
    if (!desc) {
        Log(tag + " -> ERROR_PARAMETER");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (u && desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
        std::lock_guard lock(u->mutex);
        if (u->pending) CheckPending(*u, call);
    }
    const GpuTimed timed(platform_.gpuTimer, desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE
                                                 ? reinterpret_cast<const ffxDispatchDescUpscale*>(desc)->commandList
                                                 : nullptr);
    if (u && u->mode == UpscaleContext::Mode::Own) {
        std::lock_guard lock(u->mutex);
        if (u->mode == UpscaleContext::Mode::Own) {
            if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
                const auto& d = *reinterpret_cast<const ffxDispatchDescUpscale*>(desc);
                if (!config_.customSlot && platform_.compareAmd && platform_.compareAmd()) return DispatchCompare(*u, d, call);
                return DispatchUpscale(*u, d, call);
            }
            if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK) {
                // The reactive mask only feeds AMD's upscaler; WrongGPU's does not read it.
                if (log) Log(tag + " -> OK (WrongGPU: reactive mask not needed, nothing recorded)");
                return FFX_API_RETURN_OK;
            }
            Log(tag + " -> ERROR_UNKNOWN_DESCTYPE (WrongGPU)");
            return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
        }
    }
    const OriginalApi* original = u ? u->amdApi : !context || !*context ? Original() : ApiOf(*context);
    if (!original) {
        Log(tag + " -> NO_PROVIDER");
        return FFX_API_RETURN_NO_PROVIDER;
    }
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE && platform_.noteDispatch)
        platform_.noteDispatch(reinterpret_cast<const ffxDispatchDescUpscale*>(desc)->commandList, nullptr);
    if (u) {
        std::lock_guard lock(u->mutex);
        ++u->dispatches;
        const ffxReturnCode_t rc = original->dispatch(&u->amd, desc);
        if (log || rc != FFX_API_RETURN_OK) Log(tag + " -> " + RcName(rc) + " (forwarded to AMD)");
        if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE)
            Report(*u, rc == FFX_API_RETURN_OK ? UpscaleState::Amd : UpscaleState::Failed, nullptr,
                   "AMD FSR dispatch failed: " + std::string(RcName(rc)));
        return rc;
    }
    const ffxReturnCode_t rc = original->dispatch(context, desc);
    if (log || rc != FFX_API_RETURN_OK) Log(tag + " -> " + RcName(rc) + " (forwarded to AMD)");
    if (rc == FFX_API_RETURN_OK && platform_.status && context && *context &&
        desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
        UpscaleStatus status;
        {
            std::lock_guard lock(mutex_);
            auto it = forwarded_.find(*context);
            if (it != forwarded_.end()) status = it->second.status;
        }
        if (status.device) {
            const auto& d = *reinterpret_cast<const ffxDispatchDescUpscale*>(desc);
            status.renderWidth = d.renderSize.width;
            status.renderHeight = d.renderSize.height;
            status.outputWidth = d.upscaleSize.width;
            status.outputHeight = d.upscaleSize.height;
            platform_.status(status);
        }
    }
    return rc;
}

}  // namespace wgfsr
