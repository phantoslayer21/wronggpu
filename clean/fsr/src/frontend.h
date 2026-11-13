// SPDX-License-Identifier: MIT
#pragma once
// The FidelityFX API front end (spec 20, integration point 1): routes the five exported calls. Upscaler contexts
// are handled by a WrongGPU upscaler (stand-in or network) behind wronggpu_upscaler.h; every other context type and
// every call WrongGPU does not handle is forwarded unchanged to AMD's original DLL. Every call is logged.
//
// No Windows or D3D12 code here: the DLL glue (dllmain.cpp) and the unit tests provide a Platform.
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>

#include "../include/wronggpu_upscaler.h"
#include "amd_compare.h"
#include "async_create.h"
#include "config.h"
#include "status.h"
#include "ffx_headers.h"
#include "param_map.h"
#include "wg_registry_gen.h"  // clean/wgshared: the registry's options, version-list entries and models

namespace wgfsr {

// The shared settings block and wronggpu.ini as the front end sees them (src/settings_bridge.h implements it in the
// DLL; unit tests may leave it empty: every option then reads as its registry default and nothing is written).
struct MenuSettings {
    // The option's current value (block slot > environment > wronggpu.ini > registry default) and, in *described, its
    // log text "key=value (source)" (wg_describe). *isDefault: no source set it. Empty: registry default.
    std::function<int32_t(uint32_t slot, std::string* described, bool* isDefault)> get;
    // A choice from the host's menu (an entry chosen at create, or the host ini's UpscalerIndex): writes the slot of the
    // shared block and persists it to wronggpu.ini. Only the elected writer writes; returns false (and writes nothing)
    // otherwise. `source` is the log word ("menu", "optiscaler.ini", "folder"). Empty: nothing is written.
    std::function<bool(uint32_t slot, int32_t value, const char* source)> set;
    // One-folder-per-choice fallback (registry option entry_source = folders): the version-list id from the sidecar
    // wronggpu-entry.ini next to the DLL; 0: the normal version list.
    uint64_t folderEntry = 0;
};

struct Platform {
    std::function<const OriginalApi*()> original;  // loads AMD's DLL on first use; null if it cannot be loaded
    // Resolves the upscaler implementation for this configuration; null (with the reason) when unavailable.
    std::function<const WgUpscalerBackend*(const Config&, std::string& why)> backend;
    std::function<void(const std::string&)> log;   // one line, no newline
    std::function<void()> flushLog;                // writes the queued log lines to disk now; may be empty (clean/fe-wrapped-list:
                                                   // after an upscaler's "about to ..." line, before a step that may crash)
    std::function<uint64_t()> logBytes;            // current log size, may be empty
    ResourceInfoFn resourceInfo = nullptr;
    // Interface version 2 (same-frame split). services: the split service a backend gets in its create info; null when
    // unavailable. It installs nothing: activateServices installs the D3D12 hooks (for a backend that sets
    // WG_BACKEND_NEEDS_SPLIT), and the front end calls it only after that backend's create succeeded, so a failed
    // create (or AMD's FSR) never leaves a vtable patched; false: the hooks could not be installed (the upscaler is
    // then created again without the service, as when none was offered). Empty activateServices: nothing to install.
    // noteDispatch: every upscale dispatch with
    // its command list (also those forwarded to AMD): submission trace, hooks with d3d12_hooks = on, and the frame's
    // command_queue / command_list_type (frame may be null). standIn: the stand-in, used for one frame when a backend
    // answers WG_ERROR_NOT_READY.
    std::function<const WgFrontEndServices*(const WgUpscalerBackend*, void* device)> services;
    std::function<bool(const WgUpscalerBackend*, void* device)> activateServices;
    // Where the converted model is looked for, in order (config.h ModelPlace: wronggpu.ini model_dir, the game's data
    // folder, the runtime's default), and whether a file exists. Empty: the runtime's default alone (no model_dir).
    std::function<std::vector<ModelPlace>()> modelPlaces;
    std::function<bool(const std::string& path)> fileExists;
    std::function<void(void* commandList, WgFrame* frame)> noteDispatch;
    // The reason an upscale dispatch into `commandList` can never be split (d3d12_hooks.h, ListSplitBlocked), or null.
    // Asked for a backend that sets WG_BACKEND_NEEDS_SPLIT: the context then goes to AMD's FSR. May be empty.
    std::function<const char*(void* commandList)> splitBlocked;
    std::function<void(uint64_t context, const WgFrame&)> diagnosticFrame;
    std::function<void(uint64_t context, uint64_t frame, void* device)> diagnosticFailure;
    std::function<void(uint64_t context, uint64_t frame, WgResult)> diagnosticResult;
    std::function<const WgUpscalerBackend*(const Config&)> standIn;
    // Every upscale dispatch WrongGPU's upscaler recorded (WG_OK), with its frame (frame capture, capture.h); standIn:
    // the one-frame stand-in upscaled it. May be empty.
    std::function<void(const WgFrame& frame, bool standIn)> afterDispatch;
    // Every upscale dispatch of a WrongGPU context: what upscaled it (for the in-game overlay); may be empty.
    std::function<void(const UpscaleStatus&)> status;
    GpuTimerFn gpuTimer;  // brackets every upscale dispatch (WrongGPU's and AMD's) for the overlay; may be empty
    CompareFn compareAmd;  // the player switched to AMD's FSR (comparison): WrongGPU's contexts dispatch to AMD;
                           // may be empty
    // AMD's DLL for comparisons (compare_dll; FSR 4 with AMD's FidelityFX SDK 2 DLL); null or empty: AMD's original.
    std::function<const OriginalApi*()> compareApi;
    // Parked upscalers (Config::contextParkMs). timer: runs `fn` on another thread after `ms` milliseconds, and again
    // after the number of milliseconds it returns, until it returns 0; the DLL stays loaded until then, and nothing of
    // it runs at process exit. False: not started (fn is never called). Empty: no context is ever parked (nothing
    // could end a parked upscaler's short life).
    std::function<bool(uint32_t ms, std::function<uint32_t()> fn)> timer;
    // The D3D12 device was removed (device lost): nothing of it is parked or taken over. May be empty.
    std::function<bool(void* device)> deviceLost;
    // OptiScaler ASI plugin bridge (clean/fsr/asi): when WrongGPU.asi is loaded and has hooked OptiScaler's menu,
    // WrongGPU is its own entry in the host's MAIN upscaler list, not a version in the FFX sub-box. Returns true when
    // hooked (then the version query lists AMD only and this plugin, not an FFX override, chooses the model) and sets
    // *model to 0 (AMD FSR), 'M' or 'K'. Null or returns false: no plugin / unknown host build; hosted mode as before.
    std::function<bool(char* model)> asiPlugin;
    // Registry options and the menu's choices (shared settings block). Empty members: defaults, nothing written.
    MenuSettings menu;
};

// A frame WrongGPU's upscaler cannot take although nothing is broken (inputs that do not fit each other during a
// resolution change, a missing input, a colour flag) is one frame of the stand-in or of AMD's FSR; only this many of
// them in a row hand the context to AMD's FSR for good.
constexpr uint32_t kBadFrameRun = 30;

// Provider version reported for WrongGPU's own upscaler contexts (QUERY_GET_PROVIDER_VERSION), and WrongGPU's entry in
// the upscaler versions a hosted front end lists (QUERY_GET_VERSIONS, Config::hosted).
// The ids are the registry's entry ids (clean/wgshared/registry/registry.json: WG_ENTRY_ID_BASE | option id << 8 |
// value index + 1), which keep these two; test_frontend checks they equal kWgEntries[0] and [1].
constexpr uint64_t kWrongGpuProviderVersionId = 0x5747000000000001ull;  // "WG", interface 1
constexpr uint64_t kWrongGpuKProviderVersionId = 0x5747000000000002ull; // same interface, Model K
constexpr const char* kWrongGpuVersionName = "WrongGPU Model M";

// Any of WrongGPU's own listed entries (config.h, OwnEntryOfId: SR models and RR modes from the registry).
inline bool IsWrongGpuProvider(uint64_t id) { return OwnEntryOfId(id) != nullptr; }

// The names of WrongGPU's two entries in the hosted version list (QUERY_GET_VERSIONS, Config::hosted). AMD's own
// entries are named by their version alone ("4.1.1", "3.1.5", "2.3.4"), and callers read numbers out of the names:
// the host (OptiScaler 0.9.4, 0.9.5.4-pre4 and its master of October 2026 alike) takes the first "a.b.c" in the FIRST
// listed name as "the FidelityFX version" of the whole DLL, answers a game's QUERY_GET_PROVIDER_VERSION with it and
// asks again on every use while it reads 0.0.0; from the name of the selected entry it decides which FSR 3.1 / FSR 4
// controls and work-arounds apply. A name without a number left that host with 0.0.0 (the game was then answered
// with the frame-generation DLL's number, or 0.0.0) and with the number of whichever entry was selected before. So
// each name carries the level of the FidelityFX upscaler API WrongGPU answers to, hidden after "##" (owner
// 2026-10-31: no FSR version on show; the host's menu draws nothing after "##" but still reads the number there):
//   - 3.1.0: a 3.1 level (the host still treats the entry as FSR 3.1 or later, e.g. for the output size), below
//     3.1.1, so the host's FSR 3.1 presets and manual-tuning panels stay hidden; the key-value configure calls the
//     host sends only from 3.1.1 / 3.1.4 (velocity / reactive / shading factors) are FSR tuning WrongGPU ignores;
//   - below 4.0.0: the host must never take WrongGPU for FSR 4 (FSR 4 colour-space handling, FSR 4 model options).
// The text before "##" says who upscales and holds no a.b.c number. The ids select the entry; a context's own
// provider name (QUERY_GET_PROVIDER_VERSION) is the text alone.
#define WRONGGPU_LISTED_API_LEVEL "3.1.0"
constexpr const char* kWrongGpuListedNameM = "WrongGPU Model M##" WRONGGPU_LISTED_API_LEVEL;
#if WRONGGPU_MODEL_K  // (model_k.h; without it the Model K id above is not WrongGPU's: an unknown override)
constexpr const char* kWrongGpuListedNameK = "WrongGPU Model K##" WRONGGPU_LISTED_API_LEVEL;
#endif

class FrontEnd {
public:
    FrontEnd(Platform platform, Config config);
    ~FrontEnd();

    ffxReturnCode_t CreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb);
    ffxReturnCode_t DestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb);
    ffxReturnCode_t Configure(ffxContext* context, const ffxConfigureDescHeader* desc);
    ffxReturnCode_t Query(ffxContext* context, ffxQueryDescHeader* desc);
    ffxReturnCode_t Dispatch(ffxContext* context, const ffxDispatchDescHeader* desc);

    const Config& GetConfig() const { return config_; }
    void LogLine(const std::string& line) { Log(line); }
    void FlushLog() {
        if (platform_.flushLog) platform_.flushLog();
    }
    size_t OwnContextCount();  // for tests
    bool HasParked();          // for tests

    struct UpscaleContext;

private:
    // Parked upscalers. A destroyed context whose upscaler worked to the end is not torn down: the context object and
    // its upscaler (and its stand-in) wait in parked_ for contextParkMs. A create with the same key (backend, device,
    // sizes, flags, options with the model, services) takes them over and starts with a reset frame; the timer, a
    // create with another key and a second parked context destroy them. At process exit nothing is done (as for
    // contexts the game never destroyed). Only a context destroyed while no other upscaler context is in use is parked
    // (0.5.1): a host that switches upscalers (OptiScaler's menu) creates the new context first and destroys the old one
    // after it, and nothing would take that one over; it is destroyed at once instead of holding its GPU memory.
    bool CanPark(const UpscaleContext& u);
    void Park(UpscaleContext* u, const std::string& tag);
    // The parked context if it fits `fresh` (which has its create info and options), else null and no parked context
    // any more.
    UpscaleContext* TakeParked(const WgUpscalerBackend* backend, const UpscaleContext& fresh, const std::string& tag, double* parkedMs);
    void DropParked(const std::string& tag, const char* why);
    void Retire(UpscaleContext* u, const std::string& tag, const char* why);  // really destroys a parked context
    uint32_t ParkTimer();
    // One frame WrongGPU's upscaler cannot take (kBadFrameRun): the stand-in upscales it (`frame` usable) or AMD's FSR
    // does, and the context stays WrongGPU's; `amdOnly`: never the stand-in (perceptual colour).
    ffxReturnCode_t BadFrame(UpscaleContext& u, const ffxDispatchDescUpscale& d, uint64_t call, uint64_t frameIndex,
                             const std::string& problem, WgFrame* frame, bool amdOnly, WgResult result);
    UpscaleContext* FindOwn(ffxContext* context);
    std::string Name(ffxContext* context);  // "U1(standin)", "amd:0x..." for logs
    void Log(const std::string& line);
    const OriginalApi* Original();
    const OriginalApi* UpscaleFallbackApi();
    // The AMD upscaler version with the highest "a.b.c" in its listed name (ListVersions); 0 when none. mutex_ held.
    uint64_t NewestAmdVersionLocked() const;
    ffxReturnCode_t CreateUpscale(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb,
                                  uint64_t call);
    // AMD's context for `u` from its saved create parameters; takeOver: `u` runs on AMD from now on (else only for
    // comparisons).
    ffxReturnCode_t CreateAmdFromSaved(UpscaleContext& u, uint64_t call, bool takeOver = true);
    ffxReturnCode_t DispatchCompare(UpscaleContext& u, const ffxDispatchDescUpscale& d, uint64_t call);
    ffxReturnCode_t DispatchUpscale(UpscaleContext& u, const ffxDispatchDescUpscale& d, uint64_t call);
    ffxReturnCode_t QueryOwn(UpscaleContext& u, ffxQueryDescHeader* desc);
    // Hosted (Config::hosted): the upscaler versions WrongGPU offers, its own first, then those of AMD's DLLs (the
    // comparison DLL, the game's original), each remembered with the DLL that listed it.
    ffxReturnCode_t ListVersions(ffxQueryDescGetVersions& q);
    // The host chose WrongGPU's entry `e` for upscaler context `u` (at create): an SR entry sets the context's model
    // and writes sr_model; an RR entry keeps the current sr_model for the context and writes rr_mode (the RR runtime
    // switches at its next frame). Both through Platform::menu (writer only), logged as "settings: key=value (menu)".
    void ApplyEntry(const WgEntryDef& e, UpscaleContext& u, const std::string& tag, const char* source);
    // The configured SR model (sr_model: block > environment > ini) for a context no entry chose; false: not set
    // anywhere (the model then comes from runtime_options / the runtime's default, as before the registry).
    bool ConfiguredModel(int32_t* model, std::string* described);
    // AMD's DLL behind a forwarded context (the one that created it), else AMD's original.
    const OriginalApi* ApiOf(void* context);
    // `u` runs on AMD while its upscaler's create runs (u.pending): takes the upscaler over when the create has finished.
    void CheckPending(UpscaleContext& u, uint64_t call);
    // After a successful create of `u`'s upscaler: installs the D3D12 hooks its split service needs
    // (Platform::activateServices). False: they could not be installed (nothing of them is left patched).
    bool ActivateSplit(UpscaleContext& u, const WgUpscalerBackend* backend, const std::string& tag);
    // The model folder for `u`'s options (Platform::modelPlaces): appends ";model_dir=<dir>" when the model is found
    // in a place before the runtime's own default; nothing when runtime_options / WG_RUNTIME_OPTIONS name model_dir.
    void ApplyModelDir(UpscaleContext& u, const std::string& tag);
    // Custom slot (Config::customSlot): a create of the same upscaler with the same options failed earlier in this
    // session. Key: backend name + options. Guarded by mutex_.
    std::string FailureKey(const WgUpscalerBackend* backend, const UpscaleContext& u) const;
    void ResizeForFrame(UpscaleContext& u, WgFrame& frame, const std::string& tag);  // bug hunt H
    ffxReturnCode_t BrokenFrame(UpscaleContext& u, WgFrame& frame, const std::string& tag, uint64_t frameIndex);  // bug hunt F1
    void Fail(UpscaleContext& u, const std::string& why, uint64_t call, bool error = true);
    // Platform::status for one upscale dispatch of `u` (frame: the mapped frame when WrongGPU upscaled it, else null).
    // amdReason: why AMD's FSR upscaled this one frame of a context that stays WrongGPU's (state Amd).
    void Report(const UpscaleContext& u, UpscaleState state, const WgFrame* frame, const std::string& failureReason = {},
                const std::string& amdReason = {});
    bool ShouldLogDispatch(uint64_t dispatchCount);

    Platform platform_;
    Config config_;
    const char* providerName_ = kWrongGpuVersionName;
    std::atomic<uint64_t> calls_{0};
    std::atomic<uint32_t> nextId_{1};
    std::mutex mutex_;                               // guards the maps below
    std::map<UpscaleContext*, bool> own_;            // WrongGPU's upscaler contexts
    struct Forwarded {
        std::string name;      // for the log
        uint64_t dispatches;   // for log thinning
        const OriginalApi* api = nullptr;  // the DLL that created it (null: AMD's original)
        UpscaleStatus status;  // hosted AMD upscale contexts: actual provider and game device
        bool upscale = false;  // an upscaler context (hosted: the host chose one of AMD's versions)
    };
    // Another upscaler context is in use (WrongGPU's or a forwarded AMD one): a context destroyed now is not parked, the
    // host has switched upscalers (0.5.1). mutex_ held.
    bool OtherUpscalerLocked() const;
    std::map<void*, Forwarded> forwarded_;           // AMD contexts created through us
    std::map<uint64_t, const OriginalApi*> versionApi_;  // hosted: upscaler version id -> AMD DLL that lists it
    std::map<uint64_t, std::string> versionNames_;  // copy names while the version list is valid
    std::atomic<bool> versionsLogged_{false};  // ListVersions logged the AMD provider and its versions
    std::map<uint64_t, uint64_t> queryCounts_;       // ffxQuery calls per description type (log thinning)
    std::map<uint64_t, uint64_t> configureCounts_;   // ffxConfigure calls per description type (log thinning, bug hunt F2)
    std::atomic<bool> thinning_{false};
    struct RememberedFailure {
        std::string reason;     // the first failure's reason (logged once in full)
        uint64_t repeats = 0;   // creates refused since, without a new attempt
    };
    std::map<std::string, RememberedFailure> failedCreates_;  // custom slot: FailureKey -> the failure (mutex_)
    // Custom slot, WG_ERROR_INPUT (0.6.0 follow-up, bug hunt A): remembered for the session only after kInputRefusedRun refusals
    // in a row with the same words (each a new context of the host's, no frame taken between them); a taken frame ends the run,
    // so a menu or loading frame with an odd texture does not turn WrongGPU off for the session.
    static constexpr uint32_t kInputRefusedRun = 3;
    std::string inputRefusedKey_;                  // FailureKey + the refusal's words (mutex_)
    uint32_t inputRefusedRun_ = 0;                 // refusals in a row with that key (mutex_)
    std::atomic<bool> inputRefusedAny_{false};     // inputRefusedRun_ > 0 (the taken-frame path's cheap check)
    std::map<std::string, bool> modelLogged_;      // ApplyModelDir: each distinct choice logged once (mutex_)
    std::atomic<uint64_t> reentered_{0};           // calls refused because they re-entered this front end on one thread
    std::mutex parkMutex_;                           // guards the three below; held while a parked context is destroyed
    UpscaleContext* parked_ = nullptr;
    std::chrono::steady_clock::time_point parkedAt_{}, parkedUntil_{};
    bool parkTimerArmed_ = false;
};

}  // namespace wgfsr
