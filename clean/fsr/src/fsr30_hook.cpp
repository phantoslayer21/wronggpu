// SPDX-License-Identifier: MIT
#include "fsr30_hook.h"

#include <cstring>
#include <cstdio>

#include "ffx_describe.h"
#include "fsr30_map.h"

namespace wgfsr {
namespace fsr30 {

struct Hook::Context {
    Hook* hook = nullptr;  // for the upscaler's log callback
    uint32_t id = 0;
    std::string kind;  // "upscaler" or "fsr3"
    UpscaleCreateParams create;
    MapState map;
    const WgUpscalerBackend* backend = nullptr;
    WgUpscaler* upscaler = nullptr;
    WgCreateInfo createInfo{};       // as given to the backend (for the one-frame stand-in fallback)
    std::string options;             // kept alive for createInfo.options
    const WgUpscalerBackend* fallbackBackend = nullptr;
    WgUpscaler* fallback = nullptr;  // stand-in for frames the backend is not ready for (WG_ERROR_NOT_READY)
    bool failed = false;   // the stand-in failed once: AMD only from then on
    std::string failReason;  // why (the upscaler's first error line, else the front end's), for the overlay
    void* device = nullptr;  // the game's ID3D12Device (from the first dispatch's command list)
    // Created with FFX_FSR3_ENABLE_UPSCALING_ONLY (ffx_fsr3 context), or created by such an ffx_fsr3 context (upscaler
    // context): no frame generation reads the upscaler's dilated depth / motion / previous depth, so AMD's upscaler
    // need not run (ANSWERS.md, spec 20, FSR 3.0 item 12).
    bool upscalingOnly = false;
    int innerSeen = -1;    // ffx_fsr3 context: did WrongGPU's upscaler DLL see the upscale inside AMD's call (-1 unknown)
    bool compared = false; // the last upscale went to AMD for a comparison: the next one of the other upscaler resets
    AmdComparison comparison;  // AMD's FidelityFX API upscaler for comparisons (compareApi)
    uint64_t frames = 0;   // upscale dispatches
    // The upscaler's create did not finish within create_wait_ms (async_create.h): AMD's FSR upscales until a dispatch
    // finds it finished. createInfo / options / this context stay alive for it.
    std::shared_ptr<PendingCreate> pending;
    const WgUpscalerBackend* pendingBackend = nullptr;
    std::atomic<bool> creating{false};  // the create runs on its thread: its first error line goes to createError
    std::string createError;
    std::mutex mutex;
};

namespace {

thread_local uint64_t t_seen = 0;
thread_local int t_upscalingOnlyHint = -1;  // set by ffx_fsr3 around ffxFsr3ContextCreate: -1 none, 0 no, 1 yes

std::string Rc(ErrorCode rc) {
    char text[64];
    std::snprintf(text, sizeof text, "%s (0x%08x)", ErrorName(rc), unsigned(rc));
    return text;
}

}  // namespace

Hook::Hook(Platform platform, Config config, std::string tag)
    : platform_(std::move(platform)), config_(std::move(config)), tag_(tag.empty() ? std::string() : "[" + tag + "] ") {
#if !WRONGGPU_MODEL_K  // (config.h: model=K is dropped with one log line, Model M runs)
    if (const std::string note = IgnoreModelK(config_); !note.empty()) Log(note);
#endif
}

Hook::~Hook() {  // contexts the game never destroyed: the process is ending
    for (auto& entry : contexts_)
        if (entry.second->pending) (void)entry.second.release();  // its create's thread may still use it
}

uint64_t Hook::SeenOnThisThread() { return t_seen; }

void Hook::HintUpscalingOnly(int hint) { t_upscalingOnlyHint = hint; }

size_t Hook::ContextCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.size();
}

void Hook::Log(const std::string& line) {
    if (platform_.log) platform_.log(tag_ + line);
}

bool Hook::ShouldLog(const char* function) {
    uint64_t n;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        n = callCounts_[function]++;
    }
    if (n < 3) return true;  // the first calls of every function are always logged
    if (!config_.logDispatches) return false;
    if (platform_.logBytes && platform_.logBytes() > uint64_t(config_.logMaxMegabytes) << 20) {
        if (!thinning_.exchange(true)) Log("log is larger than log_max_mb: from now on only every 600th call of a function is logged");
        return n % 600 == 0;
    }
    return true;
}

std::string Hook::Begin(const char* function, const char* caller, uint64_t& call) {
    call = ++calls_;
    return "#" + std::to_string(call) + " " + function + " from " + (caller && *caller ? caller : "?");
}

Hook::Context* Hook::Find(void* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = contexts_.find(ctx);
    return it == contexts_.end() ? nullptr : it->second.get();
}

void Hook::Register(void* ctx, const UpscaleCreateParams& params, const char* kind) {
    auto c = std::make_unique<Context>();
    c->hook = this;
    c->id = nextId_++;
    c->kind = kind;
    c->create = params;
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_[ctx] = std::move(c);  // a context memory block reused by the game replaces the old entry
}

void Hook::Unregister(void* ctx) {
    std::unique_ptr<Context> c;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = contexts_.find(ctx);
        if (it == contexts_.end()) return;
        c = std::move(it->second);
        contexts_.erase(it);
    }
    std::shared_ptr<PendingCreate> pending;
    {
        std::lock_guard<std::mutex> lock(c->mutex);
        pending = std::move(c->pending);
        if (c->upscaler) c->backend->destroy(c->upscaler);
        c->upscaler = nullptr;
        if (c->fallback) c->fallbackBackend->destroy(c->fallback);
        c->fallback = nullptr;
        DestroyComparison(c->comparison);
    }
    if (pending) {  // the create's thread may still use the context (create info, log callback)
        Context* raw = c.release();
        const WgUpscalerBackend* backend = raw->pendingBackend;
        PendingCreate::Abandon(pending, [raw, backend](WgUpscaler* upscaler) {
            if (upscaler) backend->destroy(upscaler);
            delete raw;
        });
    }
}

// Who upscales a context, for the log: "WrongGPU stand-in" (the wording the test plan uses), the network runtime's
// name, or "the WrongGPU upscaler" before one was created.
std::string Hook::Label(const Context& c) {
    if (!c.backend || !c.backend->name) return "the WrongGPU upscaler";
    const std::string name = c.backend->name;
    return name == "WrongGPU stand-in upscaler" ? "WrongGPU stand-in" : name;
}

bool Hook::StandInActive() const { return config_.upscaler != UpscalerChoice::Amd && config_.fsr3Route != "off"; }

bool Hook::TakePending(Context& c, const std::string& tag) {
    const std::string who = " C" + std::to_string(c.id);
    if (!c.pending->Done()) {
        for (const double seconds : {10.0, 30.0, 60.0, 300.0})
            if (c.pending->PassedThreshold(seconds))
                Log(tag + " note" + who + ": WrongGPU's upscaler is still starting after " + std::to_string(int(seconds)) + " s; AMD FSR upscales meanwhile");
        Report(c, UpscaleState::Amd, nullptr, c.device);
        return false;
    }
    c.creating = false;
    const WgResult r = c.pending->Result();
    WgUpscaler* upscaler = c.pending->Take();
    char seconds[32];
    std::snprintf(seconds, sizeof seconds, "%.1f s", c.pending->Seconds());
    const WgUpscalerBackend* backend = c.pendingBackend;
    c.pending.reset();
    if (r != WG_OK || !upscaler) {
        Log(tag + " ERROR" + who + ": upscaler \"" + backend->name + "\" create failed after " + seconds + ": " + WgResultText(r));
        if (c.failReason.empty()) c.failReason = c.createError.empty() ? "create failed: " + WgResultText(r) : c.createError;
        Report(c, UpscaleState::Amd, nullptr, c.device);
        return false;
    }
    c.upscaler = upscaler;
    c.backend = backend;
    c.compared = true;  // the network starts with a reset frame
    Log(tag + who + ": WrongGPU's upscaler \"" + backend->name + "\" started after " + seconds + "; it upscales from this frame on");
    return true;
}

bool Hook::RunStandIn(Context& c, const ffxDispatchDescUpscale& dispatch, const std::string& tag) {
    if (!c.upscaler && c.pending && !TakePending(c, tag)) return false;
    if (!c.upscaler) {
        std::string why;
        void* device = platform_.deviceFromCommandList && dispatch.commandList ? platform_.deviceFromCommandList(dispatch.commandList) : nullptr;
        const WgUpscalerBackend* backend = device && platform_.backend ? platform_.backend(config_, why) : nullptr;
        if (!device) why = "no D3D12 device from the command list";
        if (device) c.device = device;
        if (!backend) {
            Log(tag + " ERROR C" + std::to_string(c.id) + ": no upscaler: " + (why.empty() ? "unavailable" : why));
            if (c.failReason.empty()) c.failReason = "no upscaler: " + (why.empty() ? std::string("unavailable") : why);
            Report(c, UpscaleState::Amd, nullptr, device);
            return false;
        }
        WgCreateInfo info{};
        MapCreate(c.create, device, info);
        c.options = config_.BackendOptions();
        info.options = c.options.c_str();
        info.log_user = &c;
        info.log = [](void* user, int level, const char* message) {
            auto* ctx = static_cast<Context*>(user);
            const char* what = level == WG_LOG_ERROR ? "ERROR" : level == WG_LOG_WARN ? "warning" : "info";
            if (level == WG_LOG_ERROR && message) {
                std::string& reason = ctx->creating ? ctx->createError : ctx->failReason;
                if (reason.empty()) reason = message;
            }
            ctx->hook->Log("  C" + std::to_string(ctx->id) + " upscaler " + what + ": " + (message ? message : ""));
            if (message && std::strncmp(message, "about to ", 9) == 0 && ctx->hook->platform_.flushLog) ctx->hook->platform_.flushLog();
        };
        AdaptCreateInfo(backend, platform_.services ? platform_.services(backend, device) : nullptr, info);
        c.createInfo = info;
        Log(tag + " WrongGPU upscaler \"" + backend->name + "\" create for C" + std::to_string(c.id) + ": " + DescribeCreateInfo(info));
        // On a worker thread: the game's thread waits create_wait_ms at most (async_create.h); AMD's FSR upscales the
        // frames until the create has finished.
        c.creating = true;
        auto pending = PendingCreate::Start(backend, c.createInfo);
        if (!pending->Wait(config_.createWaitMs)) {
            c.pending = pending;
            c.pendingBackend = backend;
            Log(tag + " note C" + std::to_string(c.id) + ": upscaler \"" + backend->name + "\" is still starting after " +
                std::to_string(config_.createWaitMs) + " ms (create_wait_ms); AMD FSR upscales until it has started");
            Report(c, UpscaleState::Amd, nullptr, device);
            return false;
        }
        c.creating = false;
        if (c.failReason.empty()) c.failReason = c.createError;
        const WgResult r = pending->Result();
        c.upscaler = pending->Take();
        if (r != WG_OK || !c.upscaler) {
            c.upscaler = nullptr;
            Log(tag + " ERROR C" + std::to_string(c.id) + ": upscaler \"" + backend->name + "\" create failed: " + WgResultText(r));
            if (c.failReason.empty()) c.failReason = "create failed: " + WgResultText(r);
            Report(c, UpscaleState::Amd, nullptr, device);
            return false;
        }
        c.backend = backend;
    }
    WgFrame frame;
    std::string problem, notes;
    const uint64_t index = c.frames;
    const bool mapped = MapDispatch(dispatch, c.create, config_, platform_.resourceInfo, c.map, index, frame, problem, notes);
    // the first frames of a context, every frame with log_dispatches = 1, and any frame that cannot be mapped
    if (index < 3 || config_.logDispatches || !mapped)
        Log(tag + " WrongGPU frame: " + DescribeFrame(frame) + (notes.empty() ? "" : " notes: " + notes));
    if (mapped && c.compared) {  // back from AMD (comparison): the history is old
        frame.reset = 1;
        c.compared = false;
    }
    if (!mapped) {
        Log(tag + " ERROR C" + std::to_string(c.id) + ": frame " + std::to_string(index) + ": " + problem);
        if (c.failReason.empty()) c.failReason = problem;
        Report(c, UpscaleState::Amd, nullptr, c.device);
        return false;
    }
    if (platform_.noteDispatch) platform_.noteDispatch(frame.command_list, &frame);
    if (platform_.diagnosticFrame) platform_.diagnosticFrame(c.id, frame);
    if (!c.fallbackBackend && platform_.standIn) c.fallbackBackend = platform_.standIn(config_);
    std::string note;
    const WgResult r = DispatchWithFallback(c.backend, c.upscaler, frame, c.fallbackBackend, c.createInfo, &c.fallback, note);
    if (platform_.diagnosticResult) platform_.diagnosticResult(c.id, index, r);
    if (!note.empty()) Log(tag + " note C" + std::to_string(c.id) + ": " + note);
    if (r != WG_OK) {
        if (platform_.diagnosticFailure) platform_.diagnosticFailure(c.id, index, c.device);
        Log(tag + " ERROR C" + std::to_string(c.id) + ": upscaler \"" + c.backend->name + "\" dispatch failed: " + WgResultText(r));
        if (c.failReason.empty()) c.failReason = "dispatch failed: " + WgResultText(r);
        Report(c, UpscaleState::Amd, nullptr, c.device);
        return false;
    }
    if (platform_.afterDispatch) platform_.afterDispatch(frame, !note.empty());
    Report(c, note.empty() ? UpscaleState::Network : UpscaleState::StandIn, &frame, c.device);
    return true;
}

bool Hook::RunComparison(Context& c, ffxDispatchDescUpscale dispatch, const std::string& tag, bool log) {
    if (!c.comparison.context && !c.comparison.unavailable) {
        const OriginalApi* api = platform_.compareApi ? platform_.compareApi() : nullptr;
        if (!c.device && platform_.deviceFromCommandList && dispatch.commandList) c.device = platform_.deviceFromCommandList(dispatch.commandList);
        std::string why;
        if (api && CreateComparison(c.comparison, api, c.create, c.device, nullptr, why)) {
            Log(tag + " C" + std::to_string(c.id) + " compares with " + c.comparison.name + " (" + why + ")");
        } else {
            c.comparison.unavailable = true;
            Log(tag + " note C" + std::to_string(c.id) + ": " + (api ? "comparison DLL not usable: " + why : std::string("no comparison DLL")) +
                "; AMD's FSR 3.0 compares");
        }
    }
    if (!c.comparison.context) return false;
    if (!dispatch.upscaleSize.width || !dispatch.upscaleSize.height) dispatch.upscaleSize = c.create.maxUpscaleSize;
    const bool reset = !c.compared;
    c.compared = true;
    const ffxReturnCode_t rc = DispatchComparison(c.comparison, dispatch, reset);
    if (log || rc != FFX_API_RETURN_OK) Log(tag + " -> " + std::to_string(rc) + " (" + c.comparison.name + ", comparison)");
    Report(c, UpscaleState::Compare, nullptr, nullptr);
    return true;
}

void Hook::Report(const Context& c, UpscaleState state, const WgFrame* frame, void* device) {
    if (!platform_.status) return;
    UpscaleStatus s;
    s.state = state;
    s.upscaler = state == UpscaleState::Compare ? (c.comparison.context ? c.comparison.name : std::string("AMD FSR 3"))
                 : c.backend ? c.backend->name : "";
    if (state == UpscaleState::Amd)
        s.reason = !StandInActive() ? "upscaler = amd in wronggpu.ini"
                   : c.pending ? "WrongGPU's upscaler is starting (" + std::to_string(int(c.pending->Seconds())) + " s)"
                   : c.failReason.empty() ? "WrongGPU's upscaler failed (see wronggpu-fsr.log)" : c.failReason;
    if (frame) {
        s.renderWidth = frame->render_width;
        s.renderHeight = frame->render_height;
        s.outputWidth = frame->output_width;
        s.outputHeight = frame->output_height;
    } else {
        s.renderWidth = c.create.maxRenderSize.width;
        s.renderHeight = c.create.maxRenderSize.height;
        s.outputWidth = c.create.maxUpscaleSize.width;
        s.outputHeight = c.create.maxUpscaleSize.height;
    }
    s.device = device ? device : c.device;
    platform_.status(s);
}

// ---------------------------------------------------------------------------------------------------------------
// ffx_fsr3upscaler_x64.dll

#define UP_ORIGINAL(fn)                                                                    \
    const UpscalerOriginal* o = platform_.upscaler ? platform_.upscaler() : nullptr;      \
    if (!o || !o->fn) {                                                                    \
        Log(head + " -> not available in AMD's DLL");                                      \
        return
#define F3_ORIGINAL(fn)                                                                    \
    const Fsr3Original* o = platform_.fsr3 ? platform_.fsr3() : nullptr;                  \
    if (!o || !o->fn) {                                                                    \
        Log(head + " -> not available in AMD's DLL");                                      \
        return

ErrorCode Hook::UpContextCreate(void* ctx, const UpscalerContextDescription* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerContextCreate", caller, call);
    UP_ORIGINAL(contextCreate) ERROR_INVALID_POINTER;
    }
    const bool decode = !o->sdk11 && desc;
    Log(head + " context=" + Pointer(ctx) + " " + (decode ? Describe(*desc) : "desc=" + Pointer(desc) + " (SDK 1.1 layout, not decoded)"));
    const int hint = t_upscalingOnlyHint;
    const ErrorCode rc = o->contextCreate(ctx, desc);
    if (rc == OK && decode) Register(ctx, CreateParams(*desc), "upscaler");
    Context* c = Find(ctx);
    if (c && rc == OK) c->upscalingOnly = hint == 1;
    Log("#" + std::to_string(call) + " -> " + Rc(rc) + (c && rc == OK ? " context C" + std::to_string(c->id) : "") +
        (c && c->upscalingOnly ? " (created by an ffx_fsr3 context with UPSCALING_ONLY: AMD's upscaler is not needed)" : ""));
    return rc;
}

ErrorCode Hook::UpGetSharedResourceDescriptions(void* ctx, void* out, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerGetSharedResourceDescriptions", caller, call);
    UP_ORIGINAL(getSharedResourceDescriptions) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->getSharedResourceDescriptions(ctx, out);
    Log(head + " context=" + Pointer(ctx) + " out=" + Pointer(out) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::UpContextDispatch(void* ctx, const UpscalerDispatchDescription* desc, const char* caller) {
    ++t_seen;
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerContextDispatch", caller, call);
    const std::string tag = "#" + std::to_string(call);
    const bool log = ShouldLog("ffxFsr3UpscalerContextDispatch");
    UP_ORIGINAL(contextDispatch) ERROR_INVALID_POINTER;
    }
    const bool decode = !o->sdk11 && desc;
    if (decode && platform_.noteDispatch) platform_.noteDispatch(desc->commandList, nullptr);
    const GpuTimed timed(platform_.gpuTimer, decode ? desc->commandList : nullptr);
    Context* c = decode ? Find(ctx) : nullptr;
    if (log) {
        Log(head + " context=" + Pointer(ctx) + (c ? " (C" + std::to_string(c->id) + ")" : "") + " " +
            (decode ? Describe(*desc) : "desc=" + Pointer(desc) + (o->sdk11 ? " (SDK 1.1 layout, not decoded)" : "")));
    }
    const bool route = c && StandInActive() && (config_.fsr3Route == "auto" || config_.fsr3Route == "upscaler");
    if (route && platform_.compareAmd && platform_.compareAmd()) {  // the player compares with AMD's FSR
        std::lock_guard<std::mutex> lock(c->mutex);
        if (RunComparison(*c, ApiDispatch(*desc), tag, log)) return OK;
        UpscalerDispatchDescription d = *desc;
        if (!c->compared) d.reset = true;  // AMD's history is old (or never started)
        c->compared = true;
        if (!c->device && platform_.deviceFromCommandList && desc->commandList) c->device = platform_.deviceFromCommandList(desc->commandList);
        const ErrorCode rc = o->contextDispatch(ctx, &d);
        if (log || rc != OK) Log(tag + " -> " + Rc(rc) + " (AMD, comparison)");
        Report(*c, UpscaleState::Compare, nullptr, nullptr);
        return rc;
    }
    if (!route) {
        const ErrorCode rc = o->contextDispatch(ctx, desc);
        if (log || rc != OK) Log(tag + " -> " + Rc(rc) + " (AMD)");
        if (c && !StandInActive()) {
            if (!c->device && platform_.deviceFromCommandList && desc->commandList) c->device = platform_.deviceFromCommandList(desc->commandList);
            Report(*c, UpscaleState::Amd, nullptr, nullptr);
        }
        return rc;
    }
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->failed) {
        const ErrorCode rc = o->contextDispatch(ctx, desc);
        if (log || rc != OK) Log(tag + " -> " + Rc(rc) + " (AMD; " + Label(*c) + " failed earlier on C" + std::to_string(c->id) + ")");
        Report(*c, UpscaleState::Amd, nullptr, nullptr);
        return rc;
    }
    const bool shared = !IsNull(desc->dilatedDepth) || !IsNull(desc->dilatedMotionVectors) || !IsNull(desc->reconstructedPrevNearestDepth);
    const bool amdFirst = config_.fsr3AmdFirst == "always" || (config_.fsr3AmdFirst == "auto" && shared && !c->upscalingOnly);
    if (amdFirst) {
        const ErrorCode rc = o->contextDispatch(ctx, desc);
        if (rc != OK) {
            Log(tag + " -> " + Rc(rc) + " (AMD failed; " + Label(*c) + " not run)");
            return rc;
        }
    }
    const bool ok = RunStandIn(*c, ApiDispatch(*desc), tag);
    ++c->frames;
    if (!ok && c->pending) {  // WrongGPU's upscaler is still starting: AMD's FSR upscales this frame
        if (!amdFirst) {
            const ErrorCode rc = o->contextDispatch(ctx, desc);
            if (log || rc != OK) Log(tag + " -> " + Rc(rc) + " (AMD while WrongGPU's upscaler starts)");
            return rc;
        }
        if (log) Log(tag + " -> FFX_OK (AMD's output kept while WrongGPU's upscaler starts)");
        return OK;
    }
    if (!ok) {
        c->failed = true;
        if (!amdFirst) {
            const ErrorCode rc = o->contextDispatch(ctx, desc);
            Log(tag + " -> " + Rc(rc) + " (AMD after " + Label(*c) + " failed; C" + std::to_string(c->id) + " stays on AMD)");
            return rc;
        }
        Log(tag + " -> FFX_OK (AMD's output kept; C" + std::to_string(c->id) + " stays on AMD)");
        return OK;
    }
    if (log)
        Log(tag + " -> FFX_OK (" + std::string(amdFirst ? "AMD for the frame-generation resources, then " : "") + Label(*c) +
            (shared && !amdFirst ? "; AMD's upscaler skipped: no frame generation" : "") + ")");
    return OK;
}

ErrorCode Hook::UpContextGenerateReactiveMask(void* ctx, const UpscalerGenerateReactiveDescription* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerContextGenerateReactiveMask", caller, call);
    const bool log = ShouldLog("ffxFsr3UpscalerContextGenerateReactiveMask");
    UP_ORIGINAL(contextGenerateReactiveMask) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->contextGenerateReactiveMask(ctx, desc);
    if (log || rc != OK) Log(head + " context=" + Pointer(ctx) + " " + (!o->sdk11 && desc ? Describe(*desc) : "desc=" + Pointer(desc)) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::UpContextDestroy(void* ctx, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerContextDestroy", caller, call);
    UP_ORIGINAL(contextDestroy) ERROR_INVALID_POINTER;
    }
    Context* c = Find(ctx);
    const std::string info = c ? " (C" + std::to_string(c->id) + ", " + std::to_string(c->frames) + " upscale dispatches)" : "";
    Unregister(ctx);
    const ErrorCode rc = o->contextDestroy(ctx);
    Log(head + " context=" + Pointer(ctx) + info + " -> " + Rc(rc));
    return rc;
}

float Hook::UpGetUpscaleRatioFromQualityMode(int32_t mode, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerGetUpscaleRatioFromQualityMode", caller, call);
    UP_ORIGINAL(getUpscaleRatioFromQualityMode) 0.0f;
    }
    const float r = o->getUpscaleRatioFromQualityMode(mode);
    if (ShouldLog("ffxFsr3UpscalerGetUpscaleRatioFromQualityMode")) Log(head + " qualityMode=" + std::to_string(mode) + " -> " + std::to_string(r));
    return r;
}

ErrorCode Hook::UpGetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerGetRenderResolutionFromQualityMode", caller, call);
    UP_ORIGINAL(getRenderResolutionFromQualityMode) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->getRenderResolutionFromQualityMode(w, h, dw, dh, mode);
    if (ShouldLog("ffxFsr3UpscalerGetRenderResolutionFromQualityMode"))
        Log(head + " display=" + std::to_string(dw) + "x" + std::to_string(dh) + " qualityMode=" + std::to_string(mode) + " -> " + Rc(rc) +
            " render=" + (w ? std::to_string(*w) : "null") + "x" + (h ? std::to_string(*h) : "null"));
    return rc;
}

int32_t Hook::UpGetJitterPhaseCount(int32_t rw, int32_t dw, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerGetJitterPhaseCount", caller, call);
    UP_ORIGINAL(getJitterPhaseCount) 1;
    }
    const int32_t n = o->getJitterPhaseCount(rw, dw);
    if (ShouldLog("ffxFsr3UpscalerGetJitterPhaseCount"))
        Log(head + " renderWidth=" + std::to_string(rw) + " displayWidth=" + std::to_string(dw) + " -> " + std::to_string(n));
    return n;
}

ErrorCode Hook::UpGetJitterOffset(float* x, float* y, int32_t index, int32_t count, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerGetJitterOffset", caller, call);
    UP_ORIGINAL(getJitterOffset) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->getJitterOffset(x, y, index, count);
    if (ShouldLog("ffxFsr3UpscalerGetJitterOffset"))
        Log(head + " index=" + std::to_string(index) + " phaseCount=" + std::to_string(count) + " -> " + Rc(rc) + " x=" +
            (x ? std::to_string(*x) : "null") + " y=" + (y ? std::to_string(*y) : "null"));
    return rc;
}

bool Hook::UpResourceIsNull(const void* resourceCopy, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerResourceIsNull", caller, call);
    UP_ORIGINAL(resourceIsNull) true;
    }
    const bool r = o->resourceIsNull(resourceCopy);
    if (ShouldLog("ffxFsr3UpscalerResourceIsNull")) Log(head + " -> " + (r ? "true" : "false"));
    return r;
}

ErrorCode Hook::UpContextGetGpuMemoryUsage(void* ctx, void* usage, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerContextGetGpuMemoryUsage", caller, call);
    UP_ORIGINAL(contextGetGpuMemoryUsage) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->contextGetGpuMemoryUsage(ctx, usage);
    Log(head + " context=" + Pointer(ctx) + " -> " + Rc(rc));
    return rc;
}

uint32_t Hook::UpGetEffectVersion(const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerGetEffectVersion", caller, call);
    UP_ORIGINAL(getEffectVersion) 0;
    }
    const uint32_t v = o->getEffectVersion();
    Log(head + " -> " + Hex(v));
    return v;
}

ErrorCode Hook::UpSetConstant(void* ctx, int32_t key, void* value, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerSetConstant", caller, call);
    UP_ORIGINAL(setConstant) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->setConstant(ctx, key, value);
    Log(head + " context=" + Pointer(ctx) + " key=" + std::to_string(key) + " value=" + Pointer(value) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::UpSetGlobalDebugMessage(void* fn, uint32_t level, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3UpscalerSetGlobalDebugMessage", caller, call);
    UP_ORIGINAL(setGlobalDebugMessage) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->setGlobalDebugMessage(fn, level);
    Log(head + " fn=" + Pointer(fn) + " level=" + Hex(level) + " -> " + Rc(rc));
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// ffx_fsr3_x64.dll

ErrorCode Hook::F3DispatchFrameGeneration(const FrameGenerationDispatchDescription* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3DispatchFrameGeneration", caller, call);
    const bool log = ShouldLog("ffxFsr3DispatchFrameGeneration");
    F3_ORIGINAL(dispatchFrameGeneration) ERROR_INVALID_POINTER;
    }
    if (log) Log(head + " " + (!o->sdk11 && desc ? Describe(*desc) : "desc=" + Pointer(desc)));
    const ErrorCode rc = o->dispatchFrameGeneration(desc);
    if (log || rc != OK) Log("#" + std::to_string(call) + " -> " + Rc(rc) + " (AMD)");
    return rc;
}

ErrorCode Hook::F3ContextCreate(void* ctx, Fsr3ContextDescription* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ContextCreate", caller, call);
    F3_ORIGINAL(contextCreate) ERROR_INVALID_POINTER;
    }
    const bool decode = !o->sdk11 && desc;
    Log(head + " context=" + Pointer(ctx) + " " + (decode ? Describe(*desc) : "desc=" + Pointer(desc) + " (SDK 1.1 layout, not decoded)"));
    const bool upscalingOnly = decode && (desc->flags & FSR3_FLAG_UPSCALING_ONLY);
    // AMD's ffx_fsr3 creates its upscaler context inside this call, on this thread: tell WrongGPU's upscaler DLL.
    if (platform_.hintUpscalingOnly) platform_.hintUpscalingOnly(decode ? (upscalingOnly ? 1 : 0) : -1);
    const ErrorCode rc = o->contextCreate(ctx, desc);
    if (platform_.hintUpscalingOnly) platform_.hintUpscalingOnly(-1);
    if (rc == OK && decode) Register(ctx, CreateParams(*desc), "fsr3");
    Context* c = Find(ctx);
    if (c && rc == OK) c->upscalingOnly = upscalingOnly;
    Log("#" + std::to_string(call) + " -> " + Rc(rc) + (c && rc == OK ? " context C" + std::to_string(c->id) : ""));
    return rc;
}

ErrorCode Hook::F3ContextDispatchUpscale(void* ctx, const Fsr3DispatchUpscaleDescription* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ContextDispatchUpscale", caller, call);
    const std::string tag = "#" + std::to_string(call);
    const bool log = ShouldLog("ffxFsr3ContextDispatchUpscale");
    F3_ORIGINAL(contextDispatchUpscale) ERROR_INVALID_POINTER;
    }
    const bool decode = !o->sdk11 && desc;
    if (decode && platform_.noteDispatch) platform_.noteDispatch(desc->commandList, nullptr);
    const GpuTimed timed(platform_.gpuTimer, decode ? desc->commandList : nullptr);
    Context* c = decode ? Find(ctx) : nullptr;
    if (log)
        Log(head + " context=" + Pointer(ctx) + (c ? " (C" + std::to_string(c->id) + ")" : "") + " " +
            (decode ? Describe(*desc) : "desc=" + Pointer(desc) + (o->sdk11 ? " (SDK 1.1 layout, not decoded)" : "")));
    // With frame generation, AMD's call always runs: frame generation needs FSR 3.0's own upscaler outputs. WrongGPU's
    // ffx_fsr3upscaler DLL may see (and take over) the upscale inside this call. With UPSCALING_ONLY and the upscale
    // known to happen at this level (not in the upscaler DLL), AMD's call is skipped and only the stand-in runs.
    const bool fsr3Level = config_.fsr3Route == "fsr3" || (config_.fsr3Route == "auto" && c && c->innerSeen == 0);
    const bool compare = c && StandInActive() && platform_.compareAmd && platform_.compareAmd();
    if (c && StandInActive() && c->upscalingOnly && fsr3Level && config_.fsr3AmdFirst != "always" && !compare) {
        std::lock_guard<std::mutex> lock(c->mutex);
        if (!c->failed) {
            const bool ok = RunStandIn(*c, ApiDispatch(*desc), tag);
            ++c->frames;
            if (ok) {
                if (log) Log(tag + " -> FFX_OK (" + Label(*c) + "; AMD's ffxFsr3ContextDispatchUpscale skipped: UPSCALING_ONLY)");
                return OK;
            }
            c->failed = true;
            const ErrorCode rc = o->contextDispatchUpscale(ctx, desc);
            Log(tag + " -> " + Rc(rc) + " (AMD after " + Label(*c) + " failed; C" + std::to_string(c->id) + " stays on AMD)");
            return rc;
        }
    }
    if (compare && c->upscalingOnly && fsr3Level && config_.fsr3AmdFirst != "always") {
        std::lock_guard<std::mutex> lock(c->mutex);
        if (RunComparison(*c, ApiDispatch(*desc), tag, log)) return OK;
    }
    const uint64_t before = platform_.upscalerSeenOnThread ? platform_.upscalerSeenOnThread() : 0;
    Fsr3DispatchUpscaleDescription amdDesc = *desc;
    if (compare && fsr3Level) {  // AMD upscales at this level for the comparison; its history is old at the switch
        std::lock_guard<std::mutex> lock(c->mutex);
        if (!c->compared) amdDesc.reset = true;
        c->compared = true;
    }
    const ErrorCode rc = o->contextDispatchUpscale(ctx, &amdDesc);
    const uint64_t inner = (platform_.upscalerSeenOnThread ? platform_.upscalerSeenOnThread() : 0) - before;
    if (c && rc == OK && c->innerSeen < 0) {
        c->innerSeen = inner > 0 ? 1 : 0;
        Log(tag + " C" + std::to_string(c->id) + ": AMD's ffx_fsr3 " +
            (inner ? "calls the upscaler DLL (the upscale is taken over there)" : "has the upscaler built in (the upscale is taken over after its call)"));
    }
    if (rc != OK || !c || !StandInActive()) {
        if (log || rc != OK) Log(tag + " -> " + Rc(rc) + " (AMD; " + std::to_string(inner) + " upscaler dispatches seen by WrongGPU's ffx_fsr3upscaler inside)");
        return rc;
    }
    const bool here = config_.fsr3Route == "fsr3" || (config_.fsr3Route == "auto" && inner == 0);
    if (compare) {
        if (here) Report(*c, UpscaleState::Compare, nullptr, nullptr);
        if (log) Log(tag + " -> " + Rc(rc) + " (AMD, comparison)");
        return rc;
    }
    if (!here) {
        if (log) Log(tag + " -> " + Rc(rc) + " (" + std::to_string(inner) + " upscaler dispatches inside, handled in ffx_fsr3upscaler)");
        return rc;
    }
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->failed) {
        if (log) Log(tag + " -> " + Rc(rc) + " (AMD; " + Label(*c) + " failed earlier on C" + std::to_string(c->id) + ")");
        Report(*c, UpscaleState::Amd, nullptr, nullptr);
        return rc;
    }
    const bool ok = RunStandIn(*c, ApiDispatch(*desc), tag);
    ++c->frames;
    if (!ok) {
        c->failed = true;
        Log(tag + " -> " + Rc(rc) + " (AMD's output kept; C" + std::to_string(c->id) + " stays on AMD)");
        return rc;
    }
    if (log) Log(tag + " -> " + Rc(rc) + " (AMD, then " + Label(*c) + " over upscaleOutput)");
    return rc;
}

ErrorCode Hook::F3SkipPresent(void* ctx, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3SkipPresent", caller, call);
    F3_ORIGINAL(skipPresent) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->skipPresent(ctx);
    if (ShouldLog("ffxFsr3SkipPresent") || rc != OK) Log(head + " context=" + Pointer(ctx) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::F3ContextGenerateReactiveMask(void* ctx, const UpscalerGenerateReactiveDescription* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ContextGenerateReactiveMask", caller, call);
    const bool log = ShouldLog("ffxFsr3ContextGenerateReactiveMask");
    F3_ORIGINAL(contextGenerateReactiveMask) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->contextGenerateReactiveMask(ctx, desc);
    if (log || rc != OK) Log(head + " context=" + Pointer(ctx) + " " + (!o->sdk11 && desc ? Describe(*desc) : "desc=" + Pointer(desc)) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::F3ConfigureFrameGeneration(void* ctx, const FrameGenerationConfig* config, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ConfigureFrameGeneration", caller, call);
    const bool log = ShouldLog("ffxFsr3ConfigureFrameGeneration");
    F3_ORIGINAL(configureFrameGeneration) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->configureFrameGeneration(ctx, config);
    if (log || rc != OK)
        Log(head + " context=" + Pointer(ctx) + " " + (!o->sdk11 && config ? Describe(*config) : "config=" + Pointer(config)) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::F3ContextDestroy(void* ctx, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ContextDestroy", caller, call);
    F3_ORIGINAL(contextDestroy) ERROR_INVALID_POINTER;
    }
    Context* c = Find(ctx);
    const std::string info = c ? " (C" + std::to_string(c->id) + ", " + std::to_string(c->frames) + " WrongGPU upscaler frames)" : "";
    Unregister(ctx);
    const ErrorCode rc = o->contextDestroy(ctx);
    Log(head + " context=" + Pointer(ctx) + info + " -> " + Rc(rc));
    return rc;
}

float Hook::F3GetUpscaleRatioFromQualityMode(int32_t mode, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3GetUpscaleRatioFromQualityMode", caller, call);
    F3_ORIGINAL(getUpscaleRatioFromQualityMode) 0.0f;
    }
    const float r = o->getUpscaleRatioFromQualityMode(mode);
    if (ShouldLog("ffxFsr3GetUpscaleRatioFromQualityMode")) Log(head + " qualityMode=" + std::to_string(mode) + " -> " + std::to_string(r));
    return r;
}

ErrorCode Hook::F3GetRenderResolutionFromQualityMode(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t mode, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3GetRenderResolutionFromQualityMode", caller, call);
    F3_ORIGINAL(getRenderResolutionFromQualityMode) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->getRenderResolutionFromQualityMode(w, h, dw, dh, mode);
    if (ShouldLog("ffxFsr3GetRenderResolutionFromQualityMode"))
        Log(head + " display=" + std::to_string(dw) + "x" + std::to_string(dh) + " qualityMode=" + std::to_string(mode) + " -> " + Rc(rc) +
            " render=" + (w ? std::to_string(*w) : "null") + "x" + (h ? std::to_string(*h) : "null"));
    return rc;
}

int32_t Hook::F3GetJitterPhaseCount(int32_t rw, int32_t dw, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3GetJitterPhaseCount", caller, call);
    F3_ORIGINAL(getJitterPhaseCount) 1;
    }
    const int32_t n = o->getJitterPhaseCount(rw, dw);
    if (ShouldLog("ffxFsr3GetJitterPhaseCount"))
        Log(head + " renderWidth=" + std::to_string(rw) + " displayWidth=" + std::to_string(dw) + " -> " + std::to_string(n));
    return n;
}

ErrorCode Hook::F3GetJitterOffset(float* x, float* y, int32_t index, int32_t count, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3GetJitterOffset", caller, call);
    F3_ORIGINAL(getJitterOffset) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->getJitterOffset(x, y, index, count);
    if (ShouldLog("ffxFsr3GetJitterOffset"))
        Log(head + " index=" + std::to_string(index) + " phaseCount=" + std::to_string(count) + " -> " + Rc(rc) + " x=" +
            (x ? std::to_string(*x) : "null") + " y=" + (y ? std::to_string(*y) : "null"));
    return rc;
}

bool Hook::F3ResourceIsNull(const void* resourceCopy, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ResourceIsNull", caller, call);
    F3_ORIGINAL(resourceIsNull) true;
    }
    const bool r = o->resourceIsNull(resourceCopy);
    if (ShouldLog("ffxFsr3ResourceIsNull")) Log(head + " -> " + (r ? "true" : "false"));
    return r;
}

ErrorCode Hook::F3ContextGetGpuMemoryUsage(void* ctx, void* a, void* b, void* c, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ContextGetGpuMemoryUsage", caller, call);
    F3_ORIGINAL(contextGetGpuMemoryUsage) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->contextGetGpuMemoryUsage(ctx, a, b, c);
    Log(head + " context=" + Pointer(ctx) + " -> " + Rc(rc));
    return rc;
}

ErrorCode Hook::F3ContextDispatchFrameGenerationPrepare(void* ctx, const void* desc, const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3ContextDispatchFrameGenerationPrepare", caller, call);
    const bool log = ShouldLog("ffxFsr3ContextDispatchFrameGenerationPrepare");
    F3_ORIGINAL(contextDispatchFrameGenerationPrepare) ERROR_INVALID_POINTER;
    }
    const ErrorCode rc = o->contextDispatchFrameGenerationPrepare(ctx, desc);
    if (log || rc != OK) Log(head + " context=" + Pointer(ctx) + " desc=" + Pointer(desc) + " -> " + Rc(rc));
    return rc;
}

uint32_t Hook::F3GetEffectVersion(const char* caller) {
    uint64_t call;
    const std::string head = Begin("ffxFsr3GetEffectVersion", caller, call);
    F3_ORIGINAL(getEffectVersion) 0;
    }
    const uint32_t v = o->getEffectVersion();
    Log(head + " -> " + Hex(v));
    return v;
}

}  // namespace fsr30
}  // namespace wgfsr
