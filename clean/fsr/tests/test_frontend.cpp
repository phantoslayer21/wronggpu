// SPDX-License-Identifier: MIT
// Unit tests of the FidelityFX front end without D3D12: API routing and forwarding (fake AMD DLL), the FSR ->
// network parameter mapping, the fallback to AMD, the configuration file and the call log.
// Builds on any host (Linux g++/clang, MinGW, MSVC): see clean/fsr/CMakeLists.txt and tests/run-tests.sh.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "../src/config.h"
#include "../src/ffx_describe.h"
#include "../src/frontend.h"
#include "../src/param_map.h"
#include "wg_settings_io.h"  // pure part: describe()

using namespace wgfsr;

// WrongGPU's entries at the start of the hosted version list: the registry's kWgEntries this build offers (Model M,
// Model K when it is built in (CMake WG_MODEL_K, src/model_k.h), then the RR modes). AMD's entries follow from here.
static const int kOwn = OwnEntryCount();

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond)                                                                  \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(cond)) {                                                               \
            ++g_failures;                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
        }                                                                            \
    } while (0)
#define CHECK_NEAR(a, b) CHECK(std::fabs(double(a) - double(b)) < 1e-6)

// ---------------------------------------------------------------------------------------------------------------
// Fake AMD DLL: records every call.

struct AmdCall {
    std::string fn;
    void* context;  // the context value passed (or created)
    uint64_t type;  // first desc type
    std::vector<uint64_t> chain;
    const void* memCb;
    ffxCreateContextDescUpscale upscale{};  // copy when a CREATE_UPSCALE is in the chain
    void* device = nullptr;
    uint64_t overrideId = 0;
};
static std::vector<AmdCall> g_amd;
static int g_amdNext = 1;
static ffxReturnCode_t g_amdCreateResult = FFX_API_RETURN_OK;
static ffxReturnCode_t g_amdDispatchResult = FFX_API_RETURN_OK;

static AmdCall Record(const char* fn, void* context, const ffxApiHeader* desc, const void* memCb) {
    AmdCall c;
    c.fn = fn;
    c.context = context;
    c.type = desc ? desc->type : 0;
    c.memCb = memCb;
    for (const ffxApiHeader* h = desc; h; h = h->pNext) {
        c.chain.push_back(h->type);
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) c.upscale = *reinterpret_cast<const ffxCreateContextDescUpscale*>(h);
        if (h->type == kCreateBackendDx12) c.device = reinterpret_cast<const BackendDx12DescView*>(h)->device;
        if (h->type == FFX_API_DESC_TYPE_OVERRIDE_VERSION) c.overrideId = reinterpret_cast<const ffxOverrideVersion*>(h)->versionId;
    }
    return c;
}

static ffxReturnCode_t AmdCreate(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    // AMD cannot resolve the private version id advertised by our hosted front end.
    if (Record("create", nullptr, desc, memCb).overrideId == kWrongGpuProviderVersionId) {
        g_amd.push_back(Record("create", nullptr, desc, memCb));
        return FFX_API_RETURN_NO_PROVIDER;
    }
    if (g_amdCreateResult != FFX_API_RETURN_OK) {
        g_amd.push_back(Record("create", nullptr, desc, memCb));
        return g_amdCreateResult;
    }
    *context = reinterpret_cast<ffxContext>(uintptr_t(0xA000 + 0x10 * g_amdNext++));
    g_amd.push_back(Record("create", *context, desc, memCb));
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t AmdDestroy(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    g_amd.push_back(Record("destroy", *context, nullptr, memCb));
    *context = nullptr;
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t g_amdConfigureResult = FFX_API_RETURN_OK;
static ffxReturnCode_t AmdConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    g_amd.push_back(Record("configure", context ? *context : nullptr, desc, nullptr));
    return g_amdConfigureResult;
}
static uint64_t g_versionIds[2] = {0x1234, 0x5678};
static const char* g_versionNames[2] = {"FSR 3.1.4 (fake)", "FSR 3.1.0 (fake)"};
static ffxReturnCode_t AmdQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    g_amd.push_back(Record("query", context ? *context : nullptr, desc, nullptr));
    if (desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) {
        auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
        const uint64_t capacity = *q->outputCount;
        *q->outputCount = 2;
        for (uint64_t i = 0; i < 2 && i < capacity; ++i) {
            if (q->versionIds) q->versionIds[i] = g_versionIds[i];
            if (q->versionNames) q->versionNames[i] = g_versionNames[i];
        }
    }
    if (desc->type == FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION) {
        auto* q = reinterpret_cast<ffxQueryGetProviderVersion*>(desc);
        q->versionId = 0x1234;
        q->versionName = "FSR 3.1.4 (fake)";
    }
    return FFX_API_RETURN_OK;
}
static bool g_amdLastReset = false;  // reset flag of the last upscale dispatch AMD saw
static ffxReturnCode_t AmdDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    g_amd.push_back(Record("dispatch", context ? *context : nullptr, desc, nullptr));
    if (desc && desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) g_amdLastReset = reinterpret_cast<const ffxDispatchDescUpscale*>(desc)->reset;
    return g_amdDispatchResult;
}
static OriginalApi g_original{AmdCreate, AmdDestroy, AmdConfigure, AmdQuery, AmdDispatch};

// ---------------------------------------------------------------------------------------------------------------
// Fake upscaler behind wronggpu_upscaler.h.

struct FakeUpscaler {
    int id;
};
static WgCreateInfo g_createInfo{};
static std::string g_createOptions;
static std::vector<WgFrame> g_frames;
static std::atomic<int> g_destroyed{0};
static bool g_failCreate = false;
static int g_failDispatchAt = -1;
static int g_notReadyUntil = 0;  // FakeDispatch answers WG_ERROR_NOT_READY for frames below this count
static int g_parameterErrorAt = -1;  // FakeDispatch answers WG_ERROR_PARAMETER for this frame (inputs it cannot take)
static int g_unsupportedFrom = -1;   // FakeDispatch answers WG_ERROR_UNSUPPORTED from this frame on (frames it does not take)
static int g_inputErrorFrom = -1;    // FakeDispatch answers WG_ERROR_INPUT from this frame on (inputs it never takes)
static std::set<int> g_inputErrorAt; // ...and at these frames (a menu or loading frame with an odd texture)
static int g_deviceErrorFrom = -1;   // FakeDispatch answers WG_ERROR_DEVICE from this frame on (bug hunt F1)
static bool g_aboutTo = false;       // FakeCreate logs an "about to ..." line (the log is flushed after it)
static int g_flushes = 0;            // Platform::flushLog calls
static int g_created = 0;            // FakeCreate calls that made an upscaler
// Platform::timer and Platform::deviceLost for parked upscalers: the test runs the timer's function itself.
static bool g_useTimer = false, g_timerStarts = true, g_deviceLost = false;
static std::function<uint32_t()> g_timerFn;
static int g_timersStarted = 0;
static std::atomic<bool> g_createGate{true};  // FakeCreate (on the front end's create thread) waits while false
static std::atomic<int> g_createCalls{0};  // every FakeCreate call (made an upscaler or not)
static WgResult FakeCreate(const WgCreateInfo* info, WgUpscaler** out) {
    ++g_createCalls;
    g_createInfo = *info;
    g_createOptions = info->options;
    while (!g_createGate) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (g_aboutTo) info->log(info->log_user, WG_LOG_INFO, "about to start the HIP runtime (test)");
    info->log(info->log_user, WG_LOG_INFO, "fake upscaler created");
    if (g_failCreate) return WG_ERROR_DEVICE;
    *out = reinterpret_cast<WgUpscaler*>(new FakeUpscaler{1});
    ++g_created;
    return WG_OK;
}
static WgResult FakeDispatch(WgUpscaler*, const WgFrame* frame) {
    g_frames.push_back(*frame);
    if (int(g_frames.size()) - 1 == g_parameterErrorAt) return WG_ERROR_PARAMETER;
    if (g_unsupportedFrom >= 0 && int(g_frames.size()) - 1 >= g_unsupportedFrom) return WG_ERROR_UNSUPPORTED;
    if (g_inputErrorFrom >= 0 && int(g_frames.size()) - 1 >= g_inputErrorFrom) return WG_ERROR_INPUT;
    if (g_inputErrorAt.count(int(g_frames.size()) - 1)) return WG_ERROR_INPUT;
    if (g_deviceErrorFrom >= 0 && int(g_frames.size()) - 1 >= g_deviceErrorFrom) return WG_ERROR_DEVICE;
    if (g_failDispatchAt >= 0 && int(g_frames.size()) - 1 == g_failDispatchAt) return WG_ERROR_DEVICE;
    if (int(g_frames.size()) <= g_notReadyUntil) return WG_ERROR_NOT_READY;
    return WG_OK;
}
static void FakeDestroy(WgUpscaler* u) {
    ++g_destroyed;
    delete reinterpret_cast<FakeUpscaler*>(u);
}
static uint64_t FakeMemory(WgUpscaler*) { return 12345; }
static const WgUpscalerBackend g_backend{sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "fake upscaler",
                                         FakeCreate, FakeDispatch, FakeDestroy, FakeMemory, WG_BACKEND_NEEDS_SPLIT};
// The same backend as a version 1 DLL would return it (struct without backend_flags).
static const WgUpscalerBackend g_backendV1{offsetof(WgUpscalerBackend, backend_flags), 1, "fake upscaler v1",
                                           FakeCreate, FakeDispatch, FakeDestroy, FakeMemory, 0xFFFFFFFFu};
static const WgUpscalerBackend* g_useBackend = &g_backend;

// Fake stand-in for the one-frame fallback, and the fake split service.
static std::vector<WgFrame> g_standInFrames;
static WgCreateInfo g_standInCreate{};
static int g_standInDestroyed = 0;
static WgResult StandInCreate(const WgCreateInfo* info, WgUpscaler** out) {
    g_standInCreate = *info;
    *out = reinterpret_cast<WgUpscaler*>(new FakeUpscaler{2});
    return WG_OK;
}
static WgResult StandInDispatch(WgUpscaler*, const WgFrame* frame) {
    g_standInFrames.push_back(*frame);
    return WG_OK;
}
static void StandInDestroy(WgUpscaler* u) {
    ++g_standInDestroyed;
    delete reinterpret_cast<FakeUpscaler*>(u);
}
static const WgUpscalerBackend g_standIn{sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "fake stand-in",
                                         StandInCreate, StandInDispatch, StandInDestroy, nullptr, 0};
static WgFrontEndServices g_services{sizeof(WgFrontEndServices), nullptr, nullptr};
static std::vector<std::pair<const WgUpscalerBackend*, void*>> g_servicesAsked;
static std::vector<std::pair<void*, bool>> g_notes;  // noteDispatch calls: command list, with frame

// Fake D3D12 resource descriptions.
static std::map<void*, ResourceInfo> g_resources;
static bool FakeResourceInfo(void* resource, ResourceInfo* out) {
    auto it = g_resources.find(resource);
    if (it == g_resources.end()) return false;
    *out = it->second;
    return true;
}

static std::vector<std::string> g_log;
static std::mutex g_logMutex;  // the upscaler's create logs from the front end's create thread
static bool g_originalAvailable = true;
static const OriginalApi* g_originalApi = &g_original;
static const OriginalApi* g_compareApi = nullptr;
static const char* g_splitBlocked = nullptr;  // Platform::splitBlocked's answer
static int g_splitBlockedAsked = 0;
static bool g_compare = false;                 // Platform::compareAmd
static std::vector<UpscaleStatus> g_statuses;  // Platform::status
static std::vector<std::pair<void*, bool>> g_timed;  // Platform::gpuTimer

// Fake WrongGPU.asi bridge (clean/fsr/asi): g_asiHooked/g_asiModel stand in for the plugin's WrongGpuAsiQuery.
static bool g_asiPresent = false;  // the plugin bridge is wired at all
static bool g_asiHooked = false;   // and it reports a successful hook
static char g_asiModel = 0;        // 0 (AMD), 'M', 'K'

// Fake shared settings (Platform::menu, settings_bridge.h in the DLL): values set "somewhere" (block / env / ini),
// the writes of the menu's choices, whether this module is the writer, the folder sidecar's entry.
static bool g_menuOn = false;
static std::map<uint32_t, int32_t> g_menuValues;
static std::vector<std::tuple<uint32_t, int32_t, std::string>> g_menuWrites;
static bool g_menuWriter = true;
static uint64_t g_folderEntry = 0;

// Platform::activateServices (the D3D12 hooks after a successful create): on, its answer, its calls, and how many
// creates had been made when it was called.
static bool g_activateOn = false;
static bool g_activateResult = true;
static int g_activateCalls = 0;
static int g_createCallsAtActivate = -1;
// Platform::modelPlaces / fileExists (model lookup order).
static bool g_modelOn = false;
static std::vector<ModelPlace> g_modelPlaces;
static std::vector<std::string> g_modelFiles;  // files that "exist"
static std::vector<std::string> g_modelProbes;  // fileExists calls

static std::unique_ptr<FrontEnd> MakeFrontEnd(const Config& config) {
    Platform p;
    if (g_menuOn) {
        p.menu.get = [](uint32_t slot, std::string* described, bool* isDefault) {
            auto it = g_menuValues.find(slot);
            const int32_t v = it != g_menuValues.end() ? it->second : kWgOptions[slot].default_value;
            if (isDefault) *isDefault = it == g_menuValues.end();
            if (described) *described = wg::describe(slot, v, it != g_menuValues.end() ? wg::Source::Ini : wg::Source::Default);
            return v;
        };
        p.menu.set = [](uint32_t slot, int32_t value, const char* source) {
            if (!g_menuWriter) return false;
            g_menuWrites.emplace_back(slot, value, source);
            g_menuValues[slot] = value;
            return true;
        };
        p.menu.folderEntry = g_folderEntry;
    }
    p.original = []() -> const OriginalApi* { return g_originalAvailable ? g_originalApi : nullptr; };
    if (g_asiPresent)
        p.asiPlugin = [](char* model) { if (model) *model = g_asiModel; return g_asiHooked; };
    p.compareApi = []() -> const OriginalApi* { return g_compareApi; };
    p.backend = [](const Config&, std::string&) -> const WgUpscalerBackend* { return g_useBackend; };
    p.services = [](const WgUpscalerBackend* b, void* device) -> const WgFrontEndServices* {
        g_servicesAsked.emplace_back(b, device);
        return &g_services;
    };
    if (g_activateOn)
        p.activateServices = [](const WgUpscalerBackend*, void*) {
            ++g_activateCalls;
            g_createCallsAtActivate = g_createCalls;
            return g_activateResult;
        };
    if (g_modelOn) {
        p.modelPlaces = [] { return g_modelPlaces; };
        p.fileExists = [](const std::string& path) {
            g_modelProbes.push_back(path);
            return std::find(g_modelFiles.begin(), g_modelFiles.end(), path) != g_modelFiles.end();
        };
    }
    p.noteDispatch = [](void* list, WgFrame* frame) {
        g_notes.emplace_back(list, frame != nullptr);
        if (frame) frame->command_list_type = 2;  // as the D3D12 side would fill it in
    };
    p.standIn = [](const Config&) -> const WgUpscalerBackend* { return &g_standIn; };
    p.splitBlocked = [](void*) -> const char* {
        ++g_splitBlockedAsked;
        return g_splitBlocked;
    };
    p.compareAmd = [] { return g_compare; };
    p.status = [](const UpscaleStatus& s) { g_statuses.push_back(s); };
    p.gpuTimer = [](void* list, bool end) { g_timed.emplace_back(list, end); };
    p.log = [](const std::string& line) {
        std::lock_guard<std::mutex> lock(g_logMutex);
        g_log.push_back(line);
    };
    p.logBytes = []() -> uint64_t { return 0; };
    p.flushLog = [] {
        std::lock_guard<std::mutex> lock(g_logMutex);
        ++g_flushes;
    };
    p.resourceInfo = &FakeResourceInfo;
    if (g_useTimer) {
        p.timer = [](uint32_t, std::function<uint32_t()> fn) {
            if (!g_timerStarts) return false;
            g_timerFn = std::move(fn);
            ++g_timersStarted;
            return true;
        };
        p.deviceLost = [](void*) { return g_deviceLost; };
    }
    return std::make_unique<FrontEnd>(p, config);
}

static void Reset() {
    g_amd.clear();
    g_frames.clear();
    g_log.clear();
    g_destroyed = 0;
    g_failCreate = false;
    g_failDispatchAt = -1;
    g_notReadyUntil = 0;
    g_parameterErrorAt = -1;
    g_unsupportedFrom = -1;
    g_inputErrorFrom = -1;
    g_inputErrorAt.clear();
    g_deviceErrorFrom = -1;
    g_aboutTo = false;
    g_flushes = 0;
    g_created = 0;
    g_useTimer = false;
    g_timerStarts = true;
    g_deviceLost = false;
    g_timerFn = nullptr;
    g_timersStarted = 0;
    g_createGate = true;
    g_useBackend = &g_backend;
    g_standInFrames.clear();
    g_standInCreate = WgCreateInfo{};
    g_standInDestroyed = 0;
    g_servicesAsked.clear();
    g_notes.clear();
    g_amdCreateResult = FFX_API_RETURN_OK;
    g_amdDispatchResult = FFX_API_RETURN_OK;
    g_originalAvailable = true;
    g_originalApi = &g_original;
    g_compareApi = nullptr;
    g_splitBlocked = nullptr;
    g_splitBlockedAsked = 0;
    g_asiPresent = false;
    g_asiHooked = false;
    g_asiModel = 0;
    g_createInfo = WgCreateInfo{};
    g_compare = false;
    g_statuses.clear();
    g_timed.clear();
    g_menuOn = false;
    g_menuValues.clear();
    g_menuWrites.clear();
    g_menuWriter = true;
    g_folderEntry = 0;
    g_createCalls = 0;
    g_activateOn = false;
    g_activateResult = true;
    g_activateCalls = 0;
    g_createCallsAtActivate = -1;
    g_modelOn = false;
    g_modelPlaces.clear();
    g_modelFiles.clear();
    g_modelProbes.clear();
}

static void* const kDevice = reinterpret_cast<void*>(uintptr_t(0xD3D0));
static void* const kColor = reinterpret_cast<void*>(uintptr_t(0xC010));
static void* const kDepth = reinterpret_cast<void*>(uintptr_t(0xDE70));
static void* const kMotion = reinterpret_cast<void*>(uintptr_t(0x3070));
static void* const kOutput = reinterpret_cast<void*>(uintptr_t(0x0070));
static void* const kExposure = reinterpret_cast<void*>(uintptr_t(0xE490));
static void* const kCommandList = reinterpret_cast<void*>(uintptr_t(0xC3D1));
static int g_memCbStorage;
static const ffxAllocationCallbacks* const kMemCb = reinterpret_cast<const ffxAllocationCallbacks*>(&g_memCbStorage);
static void TestMessage(uint32_t, const wchar_t*) {}

struct UpscaleCreate {
    ffxCreateContextDescUpscale upscale{};
    BackendDx12DescView backend{};
    ffxOverrideVersion version{};
    UpscaleCreate(uint32_t flags, bool withOverride = false) {
        upscale.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        upscale.flags = flags;
        upscale.maxRenderSize = {1706, 960};
        upscale.maxUpscaleSize = {2560, 1440};
        upscale.fpMessage = &TestMessage;
        backend.header.type = kCreateBackendDx12;
        backend.device = kDevice;
        upscale.header.pNext = &backend.header;
        if (withOverride) {
            version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
            version.versionId = 0x5678;
            backend.header.pNext = &version.header;
        }
    }
};

static FfxApiResource Res(void* p, uint32_t ffxFormat, uint32_t w, uint32_t h, uint32_t state) {
    FfxApiResource r{};
    r.resource = p;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = ffxFormat;
    r.description.width = w;
    r.description.height = h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.state = state;
    return r;
}

static ffxDispatchDescUpscale MakeDispatch() {
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = kCommandList;
    d.color = Res(kColor, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, 2560, 1440, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Res(kDepth, FFX_API_SURFACE_FORMAT_R32_FLOAT, 2560, 1440, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = Res(kMotion, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, 2560, 1440, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    d.output = Res(kOutput, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, 2560, 1440, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = {0.25f, -0.375f};
    d.motionVectorScale = {1706.0f, 960.0f};
    d.renderSize = {1706, 960};
    d.upscaleSize = {0, 0};
    d.frameTimeDelta = 16.6f;
    d.preExposure = 1.0f;
    d.cameraNear = 0.1f;
    d.cameraFar = 1000.0f;
    return d;
}

static size_t CountLog(const char* needle) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    size_t n = 0;
    for (const auto& line : g_log)
        if (line.find(needle) != std::string::npos) ++n;
    return n;
}

// ---------------------------------------------------------------------------------------------------------------

static void TestOwnUpscalerPath() {
    Reset();
    // The game's color texture is typeless; the depth is a D32 depth buffer; motion is typed.
    g_resources[kColor] = {dxgi::R16G16B16A16_TYPELESS, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_TYPELESS, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R11G11B10_FLOAT, 2560, 1440};
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(ctx != nullptr);
    CHECK(g_amd.empty());  // nothing reached AMD
    CHECK(fe->OwnContextCount() == 1);
    CHECK(g_createInfo.d3d12_device == kDevice);
    CHECK(g_createInfo.max_render_width == 1706 && g_createInfo.max_render_height == 960);
    CHECK(g_createInfo.output_width == 2560 && g_createInfo.output_height == 1440);
    CHECK(g_createInfo.flags == (WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED | WG_CREATE_AUTO_EXPOSURE));
    CHECK(g_createOptions.find("view=upscale") != std::string::npos);
    CHECK(CountLog("fake upscaler created") == 1);

    ffxDispatchDescUpscale d = MakeDispatch();
    d.preExposure = 0.0f;  // invalid: becomes 1
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 1);
    const WgFrame& f = g_frames[0];
    CHECK(f.command_list == kCommandList);
    CHECK(f.color.resource == kColor && f.color.format == dxgi::R16G16B16A16_FLOAT);
    CHECK(f.depth.format == dxgi::R32_FLOAT);  // R32_TYPELESS + FFX R32_FLOAT
    CHECK(f.motion.format == dxgi::R16G16_FLOAT);
    CHECK(f.output.format == dxgi::R11G11B10_FLOAT);  // the resource's own typed format wins
    CHECK(f.color.state == d3d12state::NON_PIXEL_SHADER_RESOURCE);
    CHECK(f.motion.state == (d3d12state::NON_PIXEL_SHADER_RESOURCE | d3d12state::PIXEL_SHADER_RESOURCE));
    CHECK(f.output.state == d3d12state::UNORDERED_ACCESS);
    CHECK(f.render_width == 1706 && f.render_height == 960 && f.render_x == 0 && f.render_y == 0);
    CHECK(f.output_width == 2560 && f.output_height == 1440);  // upscaleSize 0 -> maxUpscaleSize
    CHECK_NEAR(f.jitter[0], 0.25f);   // network jitter = +FSR jitterOffset (spec 20 draft 2)
    CHECK_NEAR(f.jitter[1], -0.375f);
    CHECK_NEAR(f.motion_scale[0], 1706.0f);
    CHECK_NEAR(f.motion_scale[1], 960.0f);
    CHECK_NEAR(f.motion_offset[0], 0.0f);
    CHECK(f.exposure_mode == WG_EXPOSURE_COMPUTE);  // no texture, auto-exposure flag
    CHECK_NEAR(f.pre_exposure, 1.0f);
    CHECK(f.reset == 0);
    CHECK(f.frame_index == 0);
    CHECK(CountLog("preExposure <= 0") == 1);

    // Second frame: exposure texture, reset, explicit upscale size, other jitter.
    d = MakeDispatch();
    d.exposure = Res(kExposure, FFX_API_SURFACE_FORMAT_R32_FLOAT, 1, 1, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.reset = true;
    d.preExposure = 2.5f;
    d.upscaleSize = {2560, 1440};
    d.jitterOffset = {-0.1f, 0.2f};
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 2);
    CHECK(g_frames[1].exposure_mode == WG_EXPOSURE_TEXTURE);
    CHECK(g_frames[1].exposure.resource == kExposure && g_frames[1].exposure.format == dxgi::R32_FLOAT);
    CHECK(g_frames[1].reset == 1);
    CHECK_NEAR(g_frames[1].pre_exposure, 2.5f);
    CHECK(g_frames[1].frame_index == 1);
    CHECK_NEAR(g_frames[1].jitter[0], -0.1f);
    CHECK_NEAR(g_frames[1].motion_offset[0], 0.0f);  // no jitter-cancellation flag

    // Queries answered by WrongGPU for its own context.
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    CHECK(fe->Query(&ctx, &pv.header) == FFX_API_RETURN_OK);
    CHECK(pv.versionId == kWrongGpuProviderVersionId && std::strcmp(pv.versionName, "WrongGPU Model M") == 0);
    int32_t phases = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount pc{};
    pc.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    pc.renderWidth = 1706;
    pc.displayWidth = 2560;
    pc.pOutPhaseCount = &phases;
    CHECK(fe->Query(&ctx, &pc.header) == FFX_API_RETURN_OK);
    CHECK(phases == 18);
    float jx = 9, jy = 9;
    ffxQueryDescUpscaleGetJitterOffset jo{};
    jo.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
    jo.index = 0;
    jo.phaseCount = 18;
    jo.pOutX = &jx;
    jo.pOutY = &jy;
    CHECK(fe->Query(&ctx, &jo.header) == FFX_API_RETURN_OK);
    CHECK_NEAR(jx, 0.0f);
    CHECK_NEAR(jy, 1.0 / 3.0 - 0.5);
    uint32_t rw = 0, rh = 0;
    ffxQueryDescUpscaleGetRenderResolutionFromQualityMode rr{};
    rr.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
    rr.displayWidth = 2560;
    rr.displayHeight = 1440;
    rr.qualityMode = FFX_UPSCALE_QUALITY_MODE_QUALITY;
    rr.pOutRenderWidth = &rw;
    rr.pOutRenderHeight = &rh;
    CHECK(fe->Query(&ctx, &rr.header) == FFX_API_RETURN_OK);
    CHECK(rw == 1706 && rh == 960);
    FfxApiEffectMemoryUsage mem{};
    ffxQueryDescUpscaleGetGPUMemoryUsage mq{};
    mq.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
    mq.gpuMemoryUsageUpscaler = &mem;
    CHECK(fe->Query(&ctx, &mq.header) == FFX_API_RETURN_OK);
    CHECK(mem.totalUsageInBytes == 12345);

    // Configure key/value and reactive-mask generation stay inside WrongGPU.
    ffxConfigureDescUpscaleKeyValue kv{};
    kv.header.type = FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE;
    kv.key = FFX_API_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR;
    CHECK(fe->Configure(&ctx, &kv.header) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscaleGenerateReactiveMask rm{};
    rm.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK;
    CHECK(fe->Dispatch(&ctx, &rm.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.empty());

    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(ctx == nullptr);
    CHECK(g_destroyed == 1);
    CHECK(fe->OwnContextCount() == 0);
    CHECK(g_amd.empty());

    // Every call is in the log: 1 create + 2 dispatch + 5 query + 1 configure + 1 dispatch + 1 destroy.
    CHECK(CountLog(" ffxCreateContext ") == 1);
    CHECK(CountLog(" ffxDispatch ") == 3);
    CHECK(CountLog(" ffxQuery ") == 5);
    CHECK(CountLog(" ffxConfigure ") == 1);
    CHECK(CountLog(" ffxDestroyContext ") == 1);
    CHECK(CountLog("WrongGPU frame: frame=1 ") == 1);
    CHECK(CountLog("jitterOffset=(-0.1,0.2)") == 1);
}

// Bug hunt H: a frame whose output size (upscaleSize) is smaller than the create's maxUpscaleSize: the upscaler is created
// again for that size (the context stays), its first frame without history; the same size again creates nothing; a
// size over the maximum is not; after 8 sizes no more creates.
static void TestOutputResize() {
    Reset();
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
    CHECK(g_createCalls == 1 && g_createInfo.output_width == 2560 && g_createInfo.output_height == 1440);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_createCalls == 1 && g_frames.size() == 1 && g_frames[0].output_width == 2560);
    d.upscaleSize = {1920, 1080};
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_createCalls == 2 && g_createInfo.output_width == 1920 && g_createInfo.output_height == 1080);
    CHECK(g_destroyed == 1);  // the old upscaler
    CHECK(g_frames.size() == 2 && g_frames[1].output_width == 1920 && g_frames[1].output_height == 1080 && g_frames[1].reset == 2);
    CHECK(CountLog("output size 1920x1080 (the upscaler was created for 2560x1440") == 1);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_createCalls == 2 && g_frames.size() == 3 && g_frames[2].reset == 0);  // the same size: nothing created
    d.upscaleSize = {3840, 2160};  // over the context's maximum: not created (passed on as before)
    fe->Dispatch(&ctx, &d.header);
    CHECK(g_createCalls == 2);
    d.upscaleSize = {0, 0};  // back to the maximum
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_createCalls == 3 && g_createInfo.output_width == 2560 && g_frames.back().output_width == 2560);
    for (uint32_t k = 0; k < 10; ++k) {  // 10 more sizes: 6 more creates (8 in all), then none
        d.upscaleSize = {1000 + 10 * k, 600 + 10 * k};
        fe->Dispatch(&ctx, &d.header);
    }
    CHECK(g_createCalls == 1 + 8);
    CHECK(CountLog("the upscaler is not created again for this context") == 1);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 0);
    CHECK(g_destroyed == 9);
}

// Bug hunt F1 (a): an upscaler that failed for good, without AMD's FSR to take the context over, is not called again:
// the stand-in (or an error) for the later frames, logged at 1, 2, 3, 10... frames, not two ERROR lines per frame.
static void TestBrokenNotCalledAgain() {
    Reset();
    g_originalAvailable = false;
    g_failDispatchAt = 1;
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
    ffxDispatchDescUpscale d = MakeDispatch();
    for (int i = 0; i < 30; ++i) fe->Dispatch(&ctx, &d.header);
    std::printf("broken: upscaler frames %zu, stand-in frames %zu, lines %zu\n", g_frames.size(), g_standInFrames.size(),
                CountLog("it is not called again"));
    CHECK(g_frames.size() == 2);                         // frame 0 taken, frame 1 failed, never called again
    CHECK(CountLog("it is not called again") == 4);      // 28 frames: lines at 1, 2, 3 and 10
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 0);
}

static void TestForwarding() {
    Reset();
    auto fe = MakeFrontEnd(Config{});
    // Frame generation context: forwarded unchanged, AMD's context handed back as is.
    ffxCreateContextDescFrameGeneration fg{};
    fg.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    fg.displaySize = {2560, 1440};
    fg.maxRenderSize = {2560, 1440};
    BackendDx12DescView backend{};
    backend.header.type = kCreateBackendDx12;
    backend.device = kDevice;
    fg.header.pNext = &backend.header;
    ffxContext fgctx = nullptr;
    CHECK(fe->CreateContext(&fgctx, &fg.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == 1 && g_amd[0].fn == "create" && g_amd[0].type == FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION);
    CHECK(g_amd[0].chain.size() == 2 && g_amd[0].memCb == kMemCb);
    CHECK(fgctx == g_amd[0].context);
    CHECK(fe->OwnContextCount() == 0);

    ffxConfigureDescFrameGeneration cfg{};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.frameGenerationEnabled = true;
    CHECK(fe->Configure(&fgctx, &cfg.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "configure" && g_amd.back().context == fgctx);
    // Bug hunt F2: frame generation configures every frame: the first 3 of the type are logged, later ones only when
    // they fail (and every call still reaches AMD)
    {
        const size_t lines = CountLog(" ffxConfigure "), results = CountLog("(forwarded to AMD)");
        const size_t calls = g_amd.size();
        for (int i = 0; i < 50; ++i) CHECK(fe->Configure(&fgctx, &cfg.header) == FFX_API_RETURN_OK);
        CHECK(g_amd.size() == calls + 50);
        CHECK(CountLog(" ffxConfigure ") == lines + 2 && CountLog("(forwarded to AMD)") == results + 2);  // calls 2 and 3 of the type
        g_amdConfigureResult = FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        CHECK(fe->Configure(&fgctx, &cfg.header) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
        g_amdConfigureResult = FFX_API_RETURN_OK;
        CHECK(CountLog(" ffxConfigure ") == lines + 3 && CountLog("-> ERROR_RUNTIME_ERROR (forwarded to AMD)") >= 1);  // a failure: always
    }

    ffxDispatchDescFrameGenerationPrepare prep{};
    prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    prep.frameID = 7;
    CHECK(fe->Dispatch(&fgctx, &prep.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "dispatch" && g_amd.back().context == fgctx &&
          g_amd.back().type == FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE);

    // Global (null-context) queries and configures go to AMD.
    uint64_t count = 2;
    uint64_t ids[2] = {};
    const char* names[2] = {};
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = kDevice;
    versions.outputCount = &count;
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "query" && g_amd.back().context == nullptr);
    CHECK(count == 2 && ids[1] == 0x5678 && std::strcmp(names[0], "FSR 3.1.4 (fake)") == 0);
    CHECK(CountLog("name=\"FSR 3.1.4 (fake)\"") == 1);
    ffxConfigureDescGlobalDebug1 dbg{};
    dbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
    CHECK(fe->Configure(nullptr, &dbg.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "configure" && g_amd.back().context == nullptr);

    CHECK(fe->DestroyContext(&fgctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "destroy" && g_amd.back().memCb == kMemCb && fgctx == nullptr);

    // Without AMD's DLL, forwarded calls fail cleanly.
    g_originalAvailable = false;
    ffxContext none = nullptr;
    CHECK(fe->CreateContext(&none, &fg.header, nullptr) == FFX_API_RETURN_NO_PROVIDER);
    CHECK(none == nullptr);
}

static void TestCreateFallback() {
    Reset();
    g_failCreate = true;
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_DEPTH_INVERTED);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == 1 && g_amd[0].fn == "create" && g_amd[0].type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE);
    CHECK(g_amd[0].memCb == kMemCb);  // the game's own chain and allocator
    void* amdctx = g_amd[0].context;
    CHECK(ctx != nullptr && ctx != amdctx);  // WrongGPU wrapper in AMD mode
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.empty());
    CHECK(g_amd.back().fn == "dispatch" && g_amd.back().context == amdctx);
    int32_t phases = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount pc{};
    pc.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    pc.renderWidth = 1280;
    pc.displayWidth = 2560;
    pc.pOutPhaseCount = &phases;
    CHECK(fe->Query(&ctx, &pc.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "query" && g_amd.back().context == amdctx);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "destroy" && g_amd.back().context == amdctx && g_amd.back().memCb == kMemCb);
    CHECK(CountLog("upscaler context goes to AMD FSR") == 1);

    // fallback_to_amd = 0: the create fails instead.
    Reset();
    g_failCreate = true;
    Config noFallback;
    noFallback.fallbackToAmd = false;
    auto fe2 = MakeFrontEnd(noFallback);
    ctx = nullptr;
    CHECK(fe2->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
    CHECK(ctx == nullptr && g_amd.empty());

    // A create without the D3D12 backend description cannot run WrongGPU: AMD gets it.
    Reset();
    auto fe3 = MakeFrontEnd(Config{});
    c.upscale.header.pNext = nullptr;
    ctx = nullptr;
    CHECK(fe3->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == 1 && CountLog("no CREATE_BACKEND_DX12") == 1);
    CHECK(fe3->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

static void TestFrameFallback() {
    Reset();
    g_failDispatchAt = 1;
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // frame 0 by WrongGPU
    CHECK(g_amd.empty());
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // frame 1 fails -> AMD context created, dispatched
    CHECK(g_destroyed == 0);
    CHECK(g_amd.size() == 2);
    CHECK(g_amd[0].fn == "create" && g_amd[0].memCb == nullptr);
    CHECK(g_amd[0].upscale.flags == FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    CHECK(g_amd[0].upscale.maxRenderSize.width == 1706 && g_amd[0].upscale.maxUpscaleSize.height == 1440);
    CHECK(g_amd[0].upscale.fpMessage == &TestMessage);
    CHECK(g_amd[0].device == kDevice && g_amd[0].overrideId == 0x5678);
    CHECK(g_amd[1].fn == "dispatch" && g_amd[1].context == g_amd[0].context);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // frame 2: AMD directly
    CHECK(g_frames.size() == 2);
    CHECK(g_amd.size() == 3);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_destroyed == 1);
    CHECK(g_amd.back().fn == "destroy" && g_amd.back().memCb == nullptr);  // matches the fallback's allocator

    // A frame with missing inputs goes to AMD's FSR for that frame (the stand-in cannot write it either); the context
    // stays WrongGPU's (TestBadFrames).
    Reset();
    auto fe2 = MakeFrontEnd(Config{});
    ctx = nullptr;
    CHECK(fe2->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    d = MakeDispatch();
    d.output.resource = nullptr;
    CHECK(fe2->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.empty() && g_amd.size() == 2 && CountLog("no output") >= 1);
    CHECK(g_amd[0].fn == "create" && g_amd[1].fn == "dispatch" && g_amdLastReset);
    CHECK(CountLog("now runs on AMD FSR for the rest of its life") == 0);
    CHECK(fe2->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// One frame the upscaler cannot take does not end the network for its context (The Witcher 3 with dynamic resolution:
// during a resolution step one dispatch states a render size that does not fit its textures): the stand-in upscales
// that frame, cut to the textures, or AMD's FSR does when the stand-in cannot; the next frame is the upscaler's again,
// with a reset. Only kBadFrameRun such frames in a row hand the context to AMD's FSR.
static void TestBadFrames() {
    Reset();
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    c.upscale.maxRenderSize = {2560, 1440};
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscale good = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 1 && g_frames[0].reset == 0);
    // The textures are already the new size, the render size is still the old one.
    ffxDispatchDescUpscale bad = MakeDispatch();
    void* const smallColor = reinterpret_cast<void*>(uintptr_t(0xC020));
    g_resources[smallColor] = {dxgi::R16G16B16A16_FLOAT, 1280, 720};
    bad.color = Res(smallColor, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, 1280, 720, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 1 && g_amd.empty());
    CHECK(g_standInFrames.size() == 1 && g_standInFrames[0].render_width == 1280 && g_standInFrames[0].render_height == 720);
    CHECK(g_statuses.back().state == UpscaleState::StandIn);
    CHECK(CountLog("note U1: frame 1: render size 1706x960 exceeds the color texture 1280x720: the stand-in upscales this frame") == 1);
    CHECK(CountLog(" ERROR ") == 0 && CountLog("for the rest of its life") == 0);
    // The next frame is the upscaler's, without history; the one after it has it again.
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 2 && g_frames[1].reset == 1);
    CHECK(g_statuses.back().state == UpscaleState::Network);
    CHECK(CountLog("note U1: frame 2: WrongGPU upscales again after 1 frame(s)") == 1);
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 3 && g_frames[2].reset == 0);

    // Inputs the upscaler itself refuses (WG_ERROR_PARAMETER): the same.
    g_parameterErrorAt = 3;
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 4 && g_standInFrames.size() == 2);
    CHECK(g_statuses.back().state == UpscaleState::StandIn && g_amd.empty());
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 5 && g_frames[4].reset == 1);

    // A run of them: one line when it starts, one when it ends; kBadFrameRun - 1 frames do not end the network.
    const size_t standIns = g_standInFrames.size();
    for (uint32_t i = 0; i + 1 < kBadFrameRun; ++i) CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK);
    CHECK(g_standInFrames.size() == standIns + kBadFrameRun - 1 && g_amd.empty());
    CHECK(CountLog("the stand-in upscales this frame") == 3);  // this run, and the two single frames before
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.back().reset == 1);
    CHECK(CountLog("WrongGPU upscales again after 29 frame(s)") == 1);
    // kBadFrameRun in a row: AMD's FSR takes the context over, for good.
    const size_t frames = g_frames.size();
    for (uint32_t i = 0; i < kBadFrameRun; ++i) CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == 2 && g_amd[0].fn == "create" && g_amd[1].fn == "dispatch" && g_amdLastReset);
    CHECK(CountLog("(30 frames in a row)") == 1 && CountLog("now runs on AMD FSR for the rest of its life") == 1);
    CHECK(g_statuses.back().state == UpscaleState::Amd);
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_amd.size() == 3 && g_frames.size() == frames);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && g_destroyed == 1);

    // The stand-in cannot write the frame either (no output): AMD's FSR for that frame, from the saved create
    // parameters, with a reset; the context stays WrongGPU's and AMD's context is kept for the next such frame.
    Reset();
    fe = MakeFrontEnd(Config{});
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK);
    bad = MakeDispatch();
    bad.output.resource = nullptr;
    CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == 2 && g_amd[0].fn == "create" && g_amd[0].memCb == nullptr && g_amd[1].fn == "dispatch" && g_amdLastReset);
    CHECK(g_statuses.back().state == UpscaleState::Amd && g_statuses.back().reason.find("no output") != std::string::npos);
    CHECK(CountLog("note U1: frame 1: no output: AMD FSR upscales this frame -> OK") == 1);
    CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK && g_amd.size() == 3 && !g_amdLastReset);  // AMD's history goes on
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 2 && g_frames[1].reset == 1);
    CHECK(g_statuses.back().state == UpscaleState::Network);
    CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK && g_amd.size() == 4 && g_amd[3].fn == "dispatch" && g_amdLastReset);
    CHECK(CountLog("for the rest of its life") == 0);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "destroy" && g_amd.back().memCb == nullptr && g_destroyed == 1);

    // Without AMD's FSR (fallback_to_amd = 0) such a frame fails and the context stays; a frame the stand-in can
    // upscale is still upscaled.
    Reset();
    Config alone;
    alone.fallbackToAmd = false;
    fe = MakeFrontEnd(alone);
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_ERROR_PARAMETER && g_statuses.back().state == UpscaleState::Failed);
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 1 && g_amd.empty());
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);

    // A single frame flagged as perceptual colour is AMD's; the next one is the upscaler's again.
    Reset();
    fe = MakeFrontEnd(Config{});
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK);
    bad = MakeDispatch();
    bad.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
    CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK && g_amd.size() == 2 && g_standInFrames.empty());
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 2 && g_frames[1].reset == 1);
    for (uint32_t i = 0; i < kBadFrameRun + 2; ++i) CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK);
    CHECK(CountLog("now runs on AMD FSR for the rest of its life") == 1 && CountLog(" ERROR ") == 0);
    CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK && g_frames.size() == 2);  // AMD's for good
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// Parked upscalers (frontend.h): a destroyed context's upscaler waits context_park_ms for a create with the same key.
static void TestParkedContexts() {
    Reset();
    g_useTimer = true;
    Config config;
    config.contextParkMs = 40;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 1);
    ffxDispatchDescUpscale d = MakeDispatch();
    g_notReadyUntil = 1;  // (the first frame is the stand-in's: the context has one, and it is parked with the upscaler)
    for (int i = 0; i < 3; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 3 && g_standInFrames.size() == 1 && g_frames[2].reset == 0);
    const ffxContext first = ctx;
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && ctx == nullptr);
    CHECK(g_destroyed == 0 && g_standInDestroyed == 0 && fe->HasParked() && fe->OwnContextCount() == 0);
    CHECK(g_timersStarted == 1 && CountLog("its upscaler is parked for 40 ms") == 1);
    // A destroyed context is no context: calls with its old pointer are not WrongGPU's.
    ffxContext stale = first;
    CHECK(fe->Dispatch(&stale, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 3 && g_amd.back().fn == "dispatch");
    g_amd.clear();

    // The same create again: nothing is created, the context has a new number and starts without history.
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && ctx != nullptr);
    CHECK(g_created == 1 && g_destroyed == 0 && !fe->HasParked() && fe->OwnContextCount() == 1);
    CHECK(CountLog("took over the parked upscaler of U1") == 1 && CountLog("-> OK context=U2(fake upscaler)") == 1);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 4);
    // (reset = 2: the upscaler also starts over as a newly created one)
    CHECK(g_frames[3].reset == 2 && g_frames[3].frame_index == 0 && g_statuses.back().state == UpscaleState::Network);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames[4].reset == 0 && g_frames[4].frame_index == 1);
    CHECK(g_amd.empty());
    ffxQueryDescUpscaleGetGPUMemoryUsage memory{};
    FfxApiEffectMemoryUsage usage{};
    memory.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
    memory.gpuMemoryUsageUpscaler = &usage;
    CHECK(fe->Query(&ctx, &memory.header) == FFX_API_RETURN_OK && usage.totalUsageInBytes == 12345);

    // The timer: before the time is over the parked upscaler stays (the timer asks to be called again); after it, the
    // upscaler and its stand-in are destroyed, on the timer's thread.
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && fe->HasParked() && g_timersStarted == 1);
    CHECK(g_timerFn() > 0 && fe->HasParked() && g_destroyed == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    CHECK(g_timerFn() == 0 && !fe->HasParked() && g_destroyed == 1 && g_standInDestroyed == 1);
    CHECK(CountLog("timer: parked upscaler of U2 destroyed") == 1);
    CHECK(g_timerFn() == 0);  // nothing parked: nothing to do

    // Too late: an ordinary create. Then a create with another key: the parked upscaler is destroyed first.
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 2);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.back().reset == 0);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && fe->HasParked() && g_timersStarted == 2);
    UpscaleCreate other(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE);
    CHECK(fe->CreateContext(&ctx, &other.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_created == 3 && g_destroyed == 2 && !fe->HasParked() && CountLog("the new context has other flags") == 1);
    // Two contexts destroyed one after the other (0.5.1): the first goes at once, another upscaler context is in use (a
    // host switching upscalers creates the new one first); the last one is parked.
    ffxContext second = nullptr;
    CHECK(fe->CreateContext(&second, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 4);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && g_destroyed == 3 && !fe->HasParked());
    CHECK(CountLog("its upscaler is destroyed now, not parked: another upscaler context is in use (the host switched upscalers)") == 1);
    CHECK(fe->DestroyContext(&second, nullptr) == FFX_API_RETURN_OK && g_destroyed == 3 && fe->HasParked());
    CHECK(CountLog("another context was destroyed and is parked in its place") == 0);
    // Other sizes, another device: not the parked one.
    UpscaleCreate sized(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED);
    sized.upscale.maxRenderSize = {1280, 720};
    CHECK(fe->CreateContext(&ctx, &sized.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 5 && g_destroyed == 4);
    CHECK(CountLog("the new context has other sizes") == 1);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && fe->HasParked());
    UpscaleCreate moved(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED);
    moved.upscale.maxRenderSize = {1280, 720};
    moved.backend.device = reinterpret_cast<void*>(uintptr_t(0xD3D8));
    CHECK(fe->CreateContext(&ctx, &moved.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 6 && g_destroyed == 5);
    CHECK(CountLog("the new context has another D3D12 device") == 1);
    // A removed device: its upscaler is not parked; a parked one is not taken over.
    g_deviceLost = true;
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && !fe->HasParked() && g_destroyed == 6);
    g_deviceLost = false;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 7);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && fe->HasParked());
    g_deviceLost = true;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 8 && g_destroyed == 7);
    CHECK(CountLog("the D3D12 device was removed") == 1);
    g_deviceLost = false;
    // A context AMD's FSR took over (its upscaler failed) is not parked.
    g_failDispatchAt = int(g_frames.size());
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_amd.back().fn == "dispatch");
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && !fe->HasParked() && g_destroyed == 8);
    // An upscaler context that goes to AMD's FSR ends a parked one too.
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 9);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && fe->HasParked());
    UpscaleCreate perceptual(FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE);
    CHECK(fe->CreateContext(&ctx, &perceptual.upscale.header, nullptr) == FFX_API_RETURN_OK && !fe->HasParked() && g_destroyed == 9);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
    // No timer thread: the upscaler is destroyed at once.
    g_timerStarts = false;
    CHECK(g_timerFn() == 0);  // (the timer of the earlier parked contexts ends)
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK && g_created == 10);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && !fe->HasParked() && g_destroyed == 10);
    CHECK(CountLog("no timer could be started") == 1);

    // context_park_ms = 0, or a platform without a timer (every other test here): contexts are destroyed as before.
    Reset();
    g_useTimer = true;
    config.contextParkMs = 0;
    fe = MakeFrontEnd(config);
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && g_destroyed == 1 && !fe->HasParked() && g_timersStarted == 0);
    // sync=pipelined (the frame shown is the previous one's result): never parked.
    Reset();
    g_useTimer = true;
    config.contextParkMs = 3000;
    config.runtimeOptions = "sync=pipelined";
    fe = MakeFrontEnd(config);
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && g_destroyed == 1 && !fe->HasParked());
    // The front end ends with a parked upscaler (process exit): the upscaler is left alone.
    Reset();
    g_useTimer = true;
    config.runtimeOptions.clear();
    fe = MakeFrontEnd(config);
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK && fe->HasParked());
    fe.reset();
    CHECK(g_destroyed == 0);
}

// A host switching upscalers (OptiScaler's menu, 0.5.1): the new context is created first, the old one destroyed after
// it. With another upscaler context in use (WrongGPU's or AMD's through the host's version choice) the old upscaler is
// destroyed at once, never parked; the last context, destroyed alone, is parked as before.
static void TestHostedSwitchNotParked() {
    Reset();
    g_useTimer = true;
    Config config;
    config.hosted = true;
    config.contextParkMs = 3000;
    auto fe = MakeFrontEnd(config);
    uint64_t count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = kDevice;
    versions.outputCount = &count;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK && count == kOwn + 2);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    own.version.versionId = kWrongGpuProviderVersionId;
    UpscaleCreate amd(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    amd.version.versionId = g_versionIds[0];
    ffxDispatchDescUpscale d = MakeDispatch();
    // WrongGPU -> AMD: AMD's context first, then WrongGPU's destroyed: not parked.
    ffxContext a = nullptr, b = nullptr;
    CHECK(fe->CreateContext(&a, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && g_created == 1);
    CHECK(fe->Dispatch(&a, &d.header) == FFX_API_RETURN_OK);
    CHECK(fe->CreateContext(&b, &amd.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    CHECK(fe->DestroyContext(&a, kMemCb) == FFX_API_RETURN_OK && g_destroyed == 1 && !fe->HasParked() && g_timersStarted == 0);
    CHECK(CountLog("its upscaler is destroyed now, not parked: another upscaler context is in use (the host switched upscalers)") == 1);
    // AMD -> WrongGPU -> WrongGPU (another context of WrongGPU's, e.g. the other model): the same.
    CHECK(fe->CreateContext(&a, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && g_created == 2);
    CHECK(fe->DestroyContext(&b, kMemCb) == FFX_API_RETURN_OK && g_amd.back().fn == "destroy");
    CHECK(fe->Dispatch(&a, &d.header) == FFX_API_RETURN_OK);
    CHECK(fe->CreateContext(&b, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && g_created == 3);
    CHECK(fe->DestroyContext(&a, kMemCb) == FFX_API_RETURN_OK && g_destroyed == 2 && !fe->HasParked());
    CHECK(CountLog("another upscaler context is in use (the host switched upscalers)") == 2);
    // The last one, destroyed with no other upscaler context (a resize: destroy, then create): parked.
    CHECK(fe->Dispatch(&b, &d.header) == FFX_API_RETURN_OK);
    CHECK(fe->DestroyContext(&b, kMemCb) == FFX_API_RETURN_OK && g_destroyed == 2 && fe->HasParked() && g_timersStarted == 1);
    CHECK(fe->CreateContext(&a, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && g_created == 3 && !fe->HasParked());
    CHECK(fe->DestroyContext(&a, kMemCb) == FFX_API_RETURN_OK);
}

// The upscaler's create takes longer than create_wait_ms (Borderlands 4: HIP's start blocked while the game's thread
// waited for it): the game's create returns with AMD's FSR upscaling; the first dispatch after the create finished takes
// the upscaler over (with a reset frame). A context destroyed while its create runs is cleaned up when it finishes.
static void TestSlowCreate() {
    Reset();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    Config config;
    config.createWaitMs = 20;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    ffxContext ctx = nullptr;
    g_createGate = false;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(ctx != nullptr && g_amd.size() == 1 && g_amd[0].fn == "create" && g_amd[0].memCb == kMemCb);
    CHECK(CountLog("is still starting after 20 ms") == 1 && CountLog("goes to AMD FSR for now") == 1);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // AMD while the create runs
    CHECK(g_frames.empty() && g_amd.back().fn == "dispatch" && g_amd.back().context == g_amd[0].context);
    CHECK(!g_statuses.empty() && g_statuses.back().state == UpscaleState::Amd &&
          g_statuses.back().reason.find("is starting") != std::string::npos);
    g_createGate = true;
    for (int i = 0; i < 400 && g_frames.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    }
    CHECK(g_frames.size() == 1 && g_frames[0].reset == 1);  // taken over, starting from a reset
    CHECK(CountLog("started after") == 1 && g_statuses.back().state == UpscaleState::Network);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 2 && g_frames[1].reset == 0);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_destroyed == 1 && g_amd.back().fn == "destroy" && g_amd.back().memCb == kMemCb);  // the game's allocator

    // Destroyed while its create runs: the upscaler is destroyed when the create returns.
    Reset();
    auto fe2 = MakeFrontEnd(config);
    ctx = nullptr;
    g_createGate = false;
    CHECK(fe2->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(fe2->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(ctx == nullptr && g_destroyed == 0 && CountLog("still starting and is destroyed when it has") == 1);
    g_createGate = true;
    for (int i = 0; i < 400 && g_destroyed == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(g_destroyed == 1);

    // The create fails after the wait: AMD keeps upscaling, the reason is the upscaler's error.
    Reset();
    auto fe3 = MakeFrontEnd(config);
    ctx = nullptr;
    g_createGate = false;
    g_failCreate = true;
    CHECK(fe3->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    g_createGate = true;
    for (int i = 0; i < 400 && CountLog("create failed after") == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(fe3->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    }
    CHECK(CountLog("create failed after") == 1 && g_frames.empty());
    CHECK(fe3->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.empty() && g_amd.back().fn == "dispatch");
    CHECK(g_statuses.back().state == UpscaleState::Amd && g_statuses.back().reason.find("create failed") != std::string::npos);
    CHECK(fe3->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
}

// Comparison (compare_key): while the player compares, WrongGPU's context dispatches to an AMD context made once from
// the saved create parameters (AMD resets at the switch); switching back resets WrongGPU's history. Every upscale
// dispatch is bracketed by the GPU timer.
static void TestCompare() {
    Reset();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // WrongGPU
    CHECK(g_frames.size() == 1 && g_amd.empty());
    CHECK(g_statuses.size() == 1 && g_statuses[0].state == UpscaleState::Network);
    g_compare = true;
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // AMD: context created, queried, dispatched with reset
    CHECK(g_frames.size() == 1);
    CHECK(g_amd.size() == 3 && g_amd[0].fn == "create" && g_amd[0].overrideId == 0x5678 && g_amd[1].fn == "query");
    CHECK(g_amd[2].fn == "dispatch" && g_amd[2].context == g_amd[0].context && g_amdLastReset);
    CHECK(g_statuses.back().state == UpscaleState::Compare && g_statuses.back().upscaler == "AMD FSR FSR 3.1.4 (fake)");
    d.reset = false;
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // AMD again, no new context
    CHECK(g_amd.size() == 4 && g_amd[3].fn == "dispatch" && !g_amdLastReset);
    g_compare = false;
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // back to WrongGPU: history reset
    CHECK(g_frames.size() == 2 && g_frames[1].reset == 1);
    CHECK(g_statuses.back().state == UpscaleState::Network);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 3 && g_frames[2].reset == 0);
    CHECK(g_timed.size() == 10);  // five dispatches, begin and end each
    for (size_t i = 0; i < g_timed.size(); ++i) CHECK(g_timed[i].first == kCommandList && g_timed[i].second == (i % 2 == 1));
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "destroy" && g_amd.back().memCb == nullptr);
}

// Issue #59: the reactive-mask generation call of a WrongGPU context goes to AMD's context when one exists (frames of
// the context that went to AMD with the game's own reactive resource need this frame's mask): in a comparison and
// after a frame the stand-in could not take. Without an AMD context there is nothing to send it to (OK, nothing
// recorded, as before). A failure of AMD's call is the call's result.
static void TestReactiveMaskForwarded() {
    auto maskCalls = [] {
        size_t n = 0;
        for (const AmdCall& c : g_amd) n += c.fn == "dispatch" && c.type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK;
        return n;
    };
    ffxDispatchDescUpscaleGenerateReactiveMask mask{};
    mask.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK;

    // Comparison.
    Reset();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    {
        auto fe = MakeFrontEnd(Config{});
        UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        ffxDispatchDescUpscale d = MakeDispatch();
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // WrongGPU
        CHECK(fe->Dispatch(&ctx, &mask.header) == FFX_API_RETURN_OK && g_amd.empty());  // no AMD context yet
        g_compare = true;
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // AMD's context is made and dispatched
        CHECK(g_amd.size() == 3 && maskCalls() == 0);
        const size_t before = g_amd.size();
        CHECK(fe->Dispatch(&ctx, &mask.header) == FFX_API_RETURN_OK);
        CHECK(g_amd.size() == before + 1 && maskCalls() == 1 && g_amd.back().context == g_amd[0].context);
        g_amdDispatchResult = FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        CHECK(fe->Dispatch(&ctx, &mask.header) == FFX_API_RETURN_ERROR_RUNTIME_ERROR && maskCalls() == 2);
        g_amdDispatchResult = FFX_API_RETURN_OK;
        g_compare = false;  // back to WrongGPU: AMD's context still exists, the call still reaches it (next frames may go to AMD)
        CHECK(fe->Dispatch(&ctx, &mask.header) == FFX_API_RETURN_OK && maskCalls() == 3);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }

    // A frame the stand-in cannot take (no output): AMD's context is made from the saved create parameters.
    Reset();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    {
        auto fe = MakeFrontEnd(Config{});
        UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        ffxDispatchDescUpscale good = MakeDispatch();
        CHECK(fe->Dispatch(&ctx, &good.header) == FFX_API_RETURN_OK);
        CHECK(fe->Dispatch(&ctx, &mask.header) == FFX_API_RETURN_OK && g_amd.empty());  // no AMD context yet
        ffxDispatchDescUpscale bad = MakeDispatch();
        bad.output.resource = nullptr;
        CHECK(fe->Dispatch(&ctx, &bad.header) == FFX_API_RETURN_OK);
        CHECK(g_amd.size() == 2 && g_amd[0].fn == "create" && g_amd[1].fn == "dispatch" && maskCalls() == 0);
        CHECK(fe->Dispatch(&ctx, &mask.header) == FFX_API_RETURN_OK);
        CHECK(g_amd.size() == 3 && maskCalls() == 1 && g_amd.back().context == g_amd[0].context);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
}

// Interface version 2: a version 2 backend gets the services and the version 2 frame; a version 1 backend gets the
// version 1 structures; WG_ERROR_NOT_READY runs the stand-in for that frame only and keeps the context on the backend.
static void TestInterfaceVersions() {
    Reset();
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(0);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_createInfo.interface_version == 2 && g_createInfo.struct_size == sizeof(WgCreateInfo));
    CHECK(g_createInfo.services == &g_services);
    CHECK(g_servicesAsked.size() == 1 && g_servicesAsked[0].first == &g_backend && g_servicesAsked[0].second == kDevice);
    CHECK(CountLog("interface=2") == 1 && CountLog("split=available") == 1);
    g_notReadyUntil = 2;  // frames 0 and 1 not ready
    ffxDispatchDescUpscale d = MakeDispatch();
    for (int i = 0; i < 3; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 3 && g_standInFrames.size() == 2 && g_amd.empty() && g_destroyed == 0);
    CHECK(g_frames[0].struct_size == sizeof(WgFrame) && g_frames[0].command_list_type == 2);
    CHECK(g_standInFrames[0].command_list == kCommandList && g_standInFrames[1].frame_index == 1);
    CHECK(g_standInCreate.services == nullptr && g_standInCreate.interface_version == 2);  // the stand-in never splits
    CHECK(g_standInCreate.max_render_width == 1706 && std::string(g_standInCreate.options) == g_createOptions);
    CHECK(CountLog("stand-in for this frame") == 2);
    CHECK(g_notes.size() == 3 && g_notes[0].first == kCommandList && g_notes[0].second);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_destroyed == 1 && g_standInDestroyed == 1);

    Reset();
    g_useBackend = &g_backendV1;
    auto fe1 = MakeFrontEnd(Config{});
    ctx = nullptr;
    CHECK(fe1->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_createInfo.interface_version == 1 && g_createInfo.struct_size == offsetof(WgCreateInfo, services));
    CHECK(g_createInfo.services == nullptr && g_servicesAsked.size() == 1);
    CHECK(BackendFlags(&g_backendV1) == 0 && BackendVersion(&g_backendV1) == 1 && BackendFlags(&g_backend) == WG_BACKEND_NEEDS_SPLIT);
    d = MakeDispatch();
    CHECK(fe1->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 1 && g_frames[0].struct_size == offsetof(WgFrame, command_queue));
    CHECK(fe1->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);

    // Upscale dispatches forwarded to AMD are noted too (submission trace with upscaler = amd).
    Reset();
    Config amd;
    amd.upscaler = UpscalerChoice::Amd;
    auto fe2 = MakeFrontEnd(amd);
    ctx = nullptr;
    CHECK(fe2->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    d = MakeDispatch();
    CHECK(fe2->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_notes.size() == 1 && g_notes[0].first == kCommandList && !g_notes[0].second);
    CHECK(fe2->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// Non-linear (perceptual) colour goes to AMD's FSR (ANSWERS.md, spec 20 item 5): at create (context flag) and per
// frame (dispatch flags), logged as a note, not an error.
static void TestNonLinearColour() {
    Reset();
    auto fe = MakeFrontEnd(Config{});
    UpscaleCreate c(FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == 1 && g_amd[0].fn == "create" && g_createInfo.d3d12_device == nullptr);
    CHECK(CountLog("non-linear (perceptual) colour") == 1);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);

    Reset();
    auto fe2 = MakeFrontEnd(Config{});
    UpscaleCreate linear(0);
    CHECK(fe2->CreateContext(&ctx, &linear.upscale.header, nullptr) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe2->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // WrongGPU
    d.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ;
    CHECK(fe2->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);  // AMD from here on
    CHECK(g_frames.size() == 1 && g_amd.size() == 2 && g_amd[0].fn == "create" && g_amd[1].fn == "dispatch");
    CHECK(CountLog("note U1: frame 1: non-linear (sRGB/PQ) colour") == 1 && CountLog(" ERROR ") == 0);
    CHECK(fe2->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

static void TestAmdMode() {
    Reset();
    Config amd;
    amd.upscaler = UpscalerChoice::Amd;
    auto fe = MakeFrontEnd(amd);
    UpscaleCreate c(0);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(g_createInfo.d3d12_device == nullptr);  // the upscaler was never asked
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().fn == "dispatch" && g_frames.empty());
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    CHECK(fe->Query(&ctx, &pv.header) == FFX_API_RETURN_OK);
    CHECK(std::strcmp(pv.versionName, "FSR 3.1.4 (fake)") == 0);
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

static void TestMappingDetails() {
    Reset();
    Config config;
    UpscaleCreateParams create;
    create.maxRenderSize = {1280, 720};
    create.maxUpscaleSize = {2560, 1440};
    MapState state;
    WgFrame f;
    std::string problem, notes;

    // Display-resolution motion vectors: scale converted from display pixels to render pixels.
    create.flags = FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    ffxDispatchDescUpscale d = MakeDispatch();
    d.renderSize = {1280, 720};
    d.motionVectorScale = {2560.0f, -1440.0f};
    CHECK(MapDispatch(d, create, config, nullptr, state, 0, f, problem, notes));
    CHECK_NEAR(f.motion_scale[0], 1280.0f);
    CHECK_NEAR(f.motion_scale[1], -720.0f);

    // Jitter-cancelled motion vectors: offset = current - previous FSR jitter (zero on the first frame).
    create.flags = FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
    state = MapState{};
    d.jitterOffset = {0.25f, -0.25f};
    CHECK(MapDispatch(d, create, config, nullptr, state, 0, f, problem, notes));
    CHECK_NEAR(f.motion_offset[0], 0.0f);
    d.jitterOffset = {-0.125f, 0.375f};
    CHECK(MapDispatch(d, create, config, nullptr, state, 1, f, problem, notes));
    CHECK_NEAR(f.motion_offset[0], -0.375f);
    CHECK_NEAR(f.motion_offset[1], 0.625f);

    // Convention switches from the .ini.
    config.jitterSign[0] = -1.0f;
    config.motionSign[1] = -1.0f;
    create.flags = 0;
    CHECK(MapDispatch(d, create, config, nullptr, state, 2, f, problem, notes));
    CHECK_NEAR(f.jitter[0], 0.125f);
    CHECK_NEAR(f.jitter[1], 0.375f);
    CHECK_NEAR(f.motion_scale[1], 1440.0f);

    // Exposure without auto-exposure: value 1.
    CHECK(f.exposure_mode == WG_EXPOSURE_VALUE && f.exposure_value == 1.0f);

    // Validation.
    Config plain;
    d = MakeDispatch();
    d.renderSize = {4000, 960};
    CHECK(!MapDispatch(d, create, plain, nullptr, state, 3, f, problem, notes));
    CHECK(problem.find("exceeds the color texture") != std::string::npos);
    d = MakeDispatch();
    d.commandList = nullptr;
    CHECK(!MapDispatch(d, create, plain, nullptr, state, 3, f, problem, notes));

    // Formats and states.
    CHECK(ViewFormat(dxgi::R8G8B8A8_TYPELESS, dxgi::R8G8B8A8_UNORM_SRGB) == dxgi::R8G8B8A8_UNORM_SRGB);
    CHECK(ViewFormat(dxgi::R8G8B8A8_TYPELESS, dxgi::UNKNOWN) == dxgi::R8G8B8A8_UNORM);
    CHECK(ViewFormat(dxgi::R10G10B10A2_TYPELESS, dxgi::R16G16_FLOAT) == dxgi::R10G10B10A2_UNORM);  // foreign family
    CHECK(ViewFormat(dxgi::D32_FLOAT, dxgi::R32_FLOAT) == dxgi::R32_FLOAT);
    CHECK(ViewFormat(dxgi::R24G8_TYPELESS, dxgi::R32_UINT) == dxgi::R24_UNORM_X8_TYPELESS);  // FFX's D24S8 mapping ignored
    CHECK(ViewFormat(dxgi::D24_UNORM_S8_UINT, dxgi::UNKNOWN) == dxgi::R24_UNORM_X8_TYPELESS);
    CHECK(ViewFormat(dxgi::UNKNOWN, dxgi::R16G16B16A16_TYPELESS) == dxgi::R16G16B16A16_FLOAT);
    CHECK(DxgiFromFfxSurfaceFormat(FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT) == dxgi::R11G11B10_FLOAT);
    CHECK(DxgiFromFfxSurfaceFormat(FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB) == dxgi::B8G8R8A8_UNORM_SRGB);
    CHECK(D3D12StatesFromFfx(FFX_API_RESOURCE_STATE_GENERIC_READ) == (d3d12state::COPY_SOURCE | d3d12state::NON_PIXEL_SHADER_RESOURCE));
    CHECK(D3D12StatesFromFfx(FFX_API_RESOURCE_STATE_RENDER_TARGET) == d3d12state::RENDER_TARGET);
    CHECK(D3D12StatesFromFfx(FFX_API_RESOURCE_STATE_COMMON) == d3d12state::COMMON);

    // Helper queries (AMD documentation values).
    CHECK_NEAR(UpscaleRatioFromQualityMode(FFX_UPSCALE_QUALITY_MODE_BALANCED), 1.7f);
    CHECK(UpscaleRatioFromQualityMode(99) == 0.0f);
    CHECK(JitterPhaseCount(1506, 2560) == 23);
    CHECK(JitterPhaseCount(1280, 2560) == 32);
    CHECK(JitterPhaseCount(853, 2560) == 72);
    CHECK(JitterPhaseCount(1707, 2560) == 17);  // AMD's answer in Cyberpunk 2077 (20-T0)
    CHECK(JitterPhaseCount(1706, 2560) == 18);
    float x, y;
    JitterOffset(1, 8, &x, &y);  // Halton index 2: (1/4, 2/3)
    CHECK_NEAR(x, 0.25 - 0.5);
    CHECK_NEAR(y, 2.0 / 3.0 - 0.5);
    JitterOffset(-7, 8, &x, &y);  // wraps to index 1
    CHECK_NEAR(x, -0.25);
}

static void TestConfig() {
    std::string warnings;
    Config c = ParseConfig(
        "[wronggpu]\n upscaler = network ; comment\nview=warpdiff\nfilter = lanczos\njitter_sign_x = -1\n"
        "motion_sign_y=-1\nwarp_weight=0.5\nlog_dispatches = off\nlog_max_mb=64\nfallback_to_amd=no\n"
        "runtime_dll = my_runtime.dll\nbogus = 1\nview = sideways\njitter_sign_y = 2\n"
        "[runtime]\nmodel = somewhere\nview = depth\n[fsr]\ndebug_gain = 2\n",
        &warnings);
    CHECK(c.upscaler == UpscalerChoice::Network);
    CHECK(c.view == "warpdiff" && c.filter == "lanczos");
    CHECK(c.jitterSign[0] == -1.0f && c.jitterSign[1] == 1.0f);
    CHECK(c.motionSign[0] == 1.0f && c.motionSign[1] == -1.0f);
    CHECK_NEAR(c.warpWeight, 0.5f);
    CHECK(!c.logDispatches && c.logMaxMegabytes == 64 && !c.fallbackToAmd);
    CHECK(c.runtimeDll == "my_runtime.dll");
    CHECK(warnings.find("unknown key \"bogus\"") != std::string::npos);
    CHECK(warnings.find("bad value \"sideways\"") != std::string::npos);
    CHECK(warnings.find("bad value \"2\"") != std::string::npos);
    CHECK(c.BackendOptions().find("view=warpdiff;filter=lanczos") == 0);
    std::string none;
    Config d = ParseConfig(DefaultConfigText(), &none);
    CHECK(none.empty());
    // The default file a player gets runs Model M (network); a missing upscaler key also means network.
    CHECK(d.upscaler == UpscalerChoice::Network && d.view == "upscale" && d.jitterSign[0] == 1.0f && d.jitterSign[1] == 1.0f);
    CHECK(Config{}.upscaler == UpscalerChoice::Network);
    // Sections of other WrongGPU components are skipped without warnings; [fsr] is read.
    CHECK(c.view == "warpdiff" && warnings.find("\"model\"") == std::string::npos);
    CHECK_NEAR(c.debugGain, 2.0f);
    // Interface version 2 settings.
    CHECK(d.d3d12Hooks == "auto" && !d.splitTest && d.splitTimeoutMs == 2000);
    CHECK(d.BackendOptions().find("split_test=0") != std::string::npos);
    std::string w2;
    Config e = ParseConfig("d3d12_hooks = ON\nsplit_test = 1\nsplit_timeout_ms = 250\nd3d12_hooks_x = 1\nsplit_timeout_ms = 5\n", &w2);
    CHECK(e.d3d12Hooks == "on" && e.splitTest && e.splitTimeoutMs == 250);
    CHECK(e.BackendOptions().find("split_test=1") != std::string::npos);
    CHECK(e.Describe().find("d3d12_hooks=on split_timeout_ms=250") != std::string::npos);
    CHECK(w2.find("unknown key \"d3d12_hooks_x\"") != std::string::npos && w2.find("bad value \"5\"") != std::string::npos);

    // runtime_options: appended to the options string after the front end's keys.
    const std::string base = "view=upscale;filter=bilinear;jitter_compensation=1;warp_weight=0.9;debug_gain=4;split_test=0";
    CHECK(d.runtimeOptions.empty() && d.BackendOptions() == base);  // empty (default text has "runtime_options =")
    std::string w3;
    Config one = ParseConfig("runtime_options = timing_log=1\n", &w3);
    CHECK(w3.empty() && one.runtimeOptions == "timing_log=1" && one.BackendOptions() == base + ";timing_log=1");
    Config several = ParseConfig("[fsr]\nruntime_options = timing_log=1;sync=pipelined;share_path=C:\\Temp\\wg share ; comment\n", &w3);
    CHECK(w3.empty() && several.runtimeOptions == "timing_log=1;sync=pipelined;share_path=C:\\Temp\\wg share");
    CHECK(several.BackendOptions() == base + ";timing_log=1;sync=pipelined;share_path=C:\\Temp\\wg share");
    CHECK(several.Describe().find(base + ";timing_log=1;sync=pipelined;share_path=C:\\Temp\\wg share jitter_sign=") != std::string::npos);
    Config messy = ParseConfig("runtime_options = a=1;;b=2; c=3;\n", &w3);  // empty items dropped, items trimmed
    CHECK(messy.BackendOptions() == base + ";a=1;b=2;c=3");
    Config cut = ParseConfig("runtime_options = a=1 ;b=2\n", &w3);  // " ;" starts a comment, as documented
    CHECK(cut.runtimeOptions == "a=1");
    Config blank = ParseConfig("runtime_options =\n# a comment line\n; another\nview = depth # trailing comment\n", &w3);
    CHECK(w3.empty() && blank.runtimeOptions.empty() && blank.view == "depth");
    CHECK(NormalizeOptions("") == "" && NormalizeOptions(" x = 1 ") == "x = 1");
    // Runtime options on lines of their own (a player's wronggpu.ini, 0.5.1): appended to runtime_options after it,
    // each with a warning naming the line; unknown keys still only warn; other sections stay skipped.
    std::string w4;
    Config loose = ParseConfig("[fsr]\nruntime_options=model=K\nnetwork=fused\nk_fast=0\nhandover=fence\nnot_a_key = 3\n"
                               "[runtime]\nfp8_stages = e0\n",
                               &w4);
    CHECK(loose.runtimeOptions == "model=K;network=fused;k_fast=0;handover=fence");
    CHECK(loose.BackendOptions() == base + ";model=K;network=fused;k_fast=0;handover=fence");
    CHECK(w4.find("wronggpu.ini line 3: \"network = fused\" is a runtime option on a line of its own: appended to runtime_options as "
                  "network=fused") != std::string::npos);
    CHECK(w4.find("wronggpu.ini line 4: \"k_fast = 0\"") != std::string::npos);
    CHECK(w4.find("wronggpu.ini line 5: \"handover = fence\"") != std::string::npos);
    CHECK(w4.find("wronggpu.ini line 6: unknown key \"not_a_key\" (ignored)") != std::string::npos);
    CHECK(w4.find("fp8_stages") == std::string::npos);
    // A line before runtime_options, and keys of the runtime's other parts (output filter, pre-exposure, timing).
    std::string w5;
    Config before = ParseConfig("output_filter = quality9\n[wronggpu]\npre_exposure=scale\nruntime_options = timing_log=1\n"
                                "motion_outres = point\n",
                                &w5);
    CHECK(before.runtimeOptions == "timing_log=1;output_filter=quality9;pre_exposure=scale;motion_outres=point");
    CHECK(w5.find("line 1: \"output_filter = quality9\"") != std::string::npos && w5.find("line 5: \"motion_outres = point\"") != std::string::npos);
    CHECK(w5.find("unknown key") == std::string::npos);
    // The front end's own keys are never taken as runtime options.
    std::string w6;
    Config own = ParseConfig("view = depth\nfilter = lanczos\n", &w6);
    CHECK(w6.empty() && own.runtimeOptions.empty() && own.view == "depth" && own.filter == "lanczos");
    // A long path is passed whole (no fixed-size buffer).
    Config longPath;
    longPath.runtimeOptions = "share_path=" + std::string(600, 'p');
    CHECK(longPath.BackendOptions().size() == base.size() + 1 + 11 + 600 && longPath.Describe().find(std::string(600, 'p') + " jitter_sign") != std::string::npos);
}

static void TestLogThinning() {
    Reset();
    Config quiet;
    quiet.logDispatches = false;
    auto fe = MakeFrontEnd(quiet);
    UpscaleCreate c(0);
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &c.upscale.header, nullptr) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscale d = MakeDispatch();
    for (int i = 0; i < 10; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 10);
    CHECK(CountLog(" ffxDispatch ") == 3);  // first three only
    CHECK(fe->DestroyContext(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// Hosted by another upscaler mod (Config::hosted, OptiScaler): WrongGPU is listed first among the upscaler versions,
// the host's override decides: WrongGPU's id gives WrongGPU's context, an AMD id gives AMD's context (forwarded, every
// later call to the DLL that listed it).
static void TestHosted() {
    Reset();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    Config config;
    config.hosted = true;
    auto fe = MakeFrontEnd(config);
    uint64_t count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = kDevice;
    versions.outputCount = &count;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK);
    CHECK(count == kOwn + 2);  // WrongGPU's (M, and K when built in) + AMD's two
    uint64_t ids[16] = {};
    const char* names[16] = {};
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK);
    CHECK(count == kOwn + 2 && ids[0] == kWrongGpuProviderVersionId && std::strcmp(names[0], kWrongGpuListedNameM) == 0);
    // The label follows the number the host reads (TestHostVersionHandling).
    CHECK(std::strcmp(names[0], "WrongGPU Model M##3.1.0") == 0);
#if WRONGGPU_MODEL_K
    CHECK(ids[1] == kWrongGpuKProviderVersionId && std::strcmp(names[1], kWrongGpuListedNameK) == 0);
    CHECK(std::strstr(names[1], "WrongGPU Model K"));
#else  // nothing of Model K in the list
    for (uint64_t i = 0; i < count; ++i) CHECK(ids[i] != kWrongGpuKProviderVersionId && !std::strstr(names[i], "Model K"));
#endif
    CHECK(ids[kOwn] == g_versionIds[0] && ids[kOwn + 1] == g_versionIds[1] && std::strcmp(names[kOwn], g_versionNames[0]) == 0);
    count = 1;  // a smaller capacity: only the first entries
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK && count == 1 && ids[0] == kWrongGpuProviderVersionId);

    // The host picks WrongGPU: WrongGPU's own context; the override is not passed on.
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    own.version.versionId = kWrongGpuProviderVersionId;
    ffxContext ctx = nullptr;
    const size_t amdCalls = g_amd.size();
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(fe->OwnContextCount() == 1 && g_amd.size() == amdCalls);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 0);

    // The host picks AMD's second version: AMD's context, created with that override, every call forwarded to it.
    UpscaleCreate amd(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    amd.version.versionId = g_versionIds[1];
    ffxContext actx = nullptr;
    CHECK(fe->CreateContext(&actx, &amd.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(fe->OwnContextCount() == 0 && g_amd.back().fn == "create" && g_amd.back().overrideId == g_versionIds[1] && actx == g_amd.back().context);
    CHECK(fe->Dispatch(&actx, &d.header) == FFX_API_RETURN_OK && g_amd.back().fn == "dispatch" && g_amd.back().context == actx);
    CHECK(g_frames.size() == 1);
    CHECK(g_statuses.size() == 2 && g_statuses.back().state == UpscaleState::Amd &&
          g_statuses.back().upscaler == g_versionNames[1] && g_statuses.back().device == kDevice &&
          g_statuses.back().renderWidth == d.renderSize.width && g_statuses.back().outputWidth == d.upscaleSize.width);
    CHECK(fe->DestroyContext(&actx, kMemCb) == FFX_API_RETURN_OK && g_amd.back().fn == "destroy");

    // An id WrongGPU never listed: WrongGPU upscales.
    UpscaleCreate other(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    other.version.versionId = 0xBAD;
    ffxContext octx = nullptr;
    CHECK(fe->CreateContext(&octx, &other.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    CHECK(fe->Dispatch(&octx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_statuses.size() == 3 && g_statuses.back().state == UpscaleState::Network);
    CHECK(fe->DestroyContext(&octx, kMemCb) == FFX_API_RETURN_OK);

    // Not hosted: the version query goes to AMD unchanged (TestForwarding) and an override does not stop WrongGPU.
}

static void TestHostedFallbackLifetime() {
    Reset();
    Config config;
    config.hosted = true;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
    own.version.versionId = kWrongGpuProviderVersionId;
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    auto d = MakeDispatch();
    g_notReadyUntil = 1;
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_standInFrames.size() == 1);
    g_failDispatchAt = 1;
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_destroyed == 0 && g_standInDestroyed == 0);
    CHECK(g_amd.back().fn == "dispatch");
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_destroyed == 1 && g_standInDestroyed == 1);
}

static void TestHostedAmdOverride() {
    Reset();
    Config config;
    config.hosted = true;
    config.upscaler = UpscalerChoice::Amd;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
    own.version.versionId = kWrongGpuProviderVersionId;
    // An opaque extension after the removed override must survive unchanged.
    ffxApiHeader extension{};
    extension.type = 0xABCDEF;
    own.version.header.pNext = &extension;
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().overrideId == 0 && g_amd.back().memCb == kMemCb);
    CHECK((g_amd.back().chain == std::vector<uint64_t>{FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, kCreateBackendDx12, extension.type}));
    CHECK(own.backend.header.pNext == &own.version.header && own.version.header.pNext == &extension);
    CHECK(own.version.versionId == kWrongGpuProviderVersionId);
    if (ctx) CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && g_amd.back().memCb == kMemCb);

    // Private override first: the opaque tail and both public create descriptions remain intact.
    own.version.header.pNext = &own.upscale.header;
    own.backend.header.pNext = &extension;
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.version.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().overrideId == 0 && g_amd.back().device == kDevice && g_amd.back().upscale.fpMessage == &TestMessage);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);

    // Unknown prefix cannot safely be copied: fail explicitly without forwarding or modifying the caller.
    extension.pNext = &own.version.header;
    own.backend.header.pNext = nullptr;
    const size_t before = g_amd.size();
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &extension, kMemCb) == FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE);
    CHECK(g_amd.size() == before && !ctx && extension.pNext == &own.version.header);

    // Startup fallback follows the same path; it must not pass our id to AMD either.
    Reset();
    config.upscaler = UpscalerChoice::Network;
    fe = MakeFrontEnd(config);
    g_failCreate = true;
    UpscaleCreate failed(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
    failed.version.versionId = kWrongGpuProviderVersionId;
    CHECK(fe->CreateContext(&ctx, &failed.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.back().overrideId == 0 && g_amd.back().memCb == kMemCb);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
}

static int g_wrongOwnerCalls = 0;
static ffxReturnCode_t WrongOwnerCreate(ffxContext*, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*) { ++g_wrongOwnerCalls; return FFX_API_RETURN_NO_PROVIDER; }
static ffxReturnCode_t WrongOwnerDestroy(ffxContext*, const ffxAllocationCallbacks*) { ++g_wrongOwnerCalls; return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
static ffxReturnCode_t WrongOwnerConfigure(ffxContext*, const ffxConfigureDescHeader*) { ++g_wrongOwnerCalls; return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
static ffxReturnCode_t WrongOwnerQuery(ffxContext*, ffxQueryDescHeader*) { ++g_wrongOwnerCalls; return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
static ffxReturnCode_t WrongOwnerDispatch(ffxContext*, const ffxDispatchDescHeader*) { ++g_wrongOwnerCalls; return FFX_API_RETURN_ERROR_RUNTIME_ERROR; }
static const OriginalApi g_wrongOwner{WrongOwnerCreate, WrongOwnerDestroy, WrongOwnerConfigure, WrongOwnerQuery, WrongOwnerDispatch};

static std::atomic<bool> g_quietCreateFinished{false};
static WgResult QuietDelayedCreate(const WgCreateInfo*, WgUpscaler** out) {
    while (!g_createGate.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Avoid dereferencing the existing dangling log callback in the red build;
    // the result still has to be destroyed exactly once when abandoned.
    *out = reinterpret_cast<WgUpscaler*>(new FakeUpscaler{1});
    g_quietCreateFinished.store(true);
    return WG_OK;
}
static const WgUpscalerBackend g_quietDelayedBackend{sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION,
    "quiet delayed create", QuietDelayedCreate, FakeDispatch, FakeDestroy, FakeMemory, WG_BACKEND_NEEDS_SPLIT};

static void TestNoProviderDelayedCreateCleanup() {
    Reset();
    g_originalAvailable = false;
    g_compareApi = nullptr;
    g_useBackend = &g_quietDelayedBackend;
    g_createGate = false;
    g_quietCreateFinished = false;
    Config config; config.hosted = true; config.createWaitMs = 1;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
    own.version.versionId = kWrongGpuProviderVersionId;
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_NO_PROVIDER);
    CHECK(!ctx && g_destroyed.load() == 0);
    g_createGate = true;
    for (int i = 0; i < 200 && g_destroyed.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(g_quietCreateFinished.load());
    CHECK(g_destroyed.load() == 1);
}

static void TestHostedSeparateAmdFallback() {
    for (bool startupFailure : {true, false}) {
        Reset();
        g_originalAvailable = false;
        g_compareApi = &g_original;
        Config config; config.hosted = true;
        auto fe = MakeFrontEnd(config);
        UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        own.version.versionId = kWrongGpuProviderVersionId;
        g_failCreate = startupFailure;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        if (!ctx) continue; // retain the precise red signal without crashing the test
        auto dispatch = MakeDispatch();
        if (!startupFailure) {
            g_failDispatchAt = 0;
            CHECK(fe->Dispatch(&ctx, &dispatch.header) == FFX_API_RETURN_OK);
            CHECK(g_destroyed == 0); // commands may still reference network resources
        }
        if (g_amd.empty()) {
            CHECK(!g_amd.empty());
            CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
            continue;
        }
        CHECK(g_amd.back().fn == (startupFailure ? "create" : "dispatch"));
        CHECK(g_amd[0].overrideId == 0);
        CHECK(g_amd[0].memCb == (startupFailure ? kMemCb : nullptr));
        // Another original appears later. It must never receive the existing
        // context created by the separate AMD DLL.
        g_wrongOwnerCalls = 0;
        g_originalAvailable = true;
        g_originalApi = &g_wrongOwner;
        CHECK(fe->Dispatch(&ctx, &dispatch.header) == FFX_API_RETURN_OK);
        ffxQueryGetProviderVersion provider{};
        provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        CHECK(fe->Query(&ctx, &provider.header) == FFX_API_RETURN_OK && provider.versionName);
        ffxConfigureDescGlobalDebug1 configure{};
        configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
        CHECK(fe->Configure(&ctx, &configure.header) == FFX_API_RETURN_OK);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && !ctx);
        CHECK(g_wrongOwnerCalls == 0);
        CHECK(g_amd.back().memCb == (startupFailure ? kMemCb : nullptr));
        CHECK(g_destroyed == (startupFailure ? 0 : 1));
    }
}

static void TestFailedFrameProviderTruth() {
    for (int failure = 0; failure < 4; ++failure) {
        Reset();
        Config config; config.hosted = true;
        auto fe = MakeFrontEnd(config);
        UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        own.version.versionId = kWrongGpuProviderVersionId;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        g_failDispatchAt = 0;
        if (failure == 0 || failure == 3) g_originalAvailable = false;
        if (failure == 1) g_amdCreateResult = FFX_API_RETURN_NO_PROVIDER;
        if (failure == 2) g_amdDispatchResult = FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        auto dispatch = MakeDispatch();
        if (failure == 3) dispatch.flags = FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB;
        CHECK(fe->Dispatch(&ctx, &dispatch.header) != FFX_API_RETURN_OK);
        CHECK(!g_statuses.empty());
        if (!g_statuses.empty()) {
            const auto& status = g_statuses.back();
            CHECK(status.state != UpscaleState::Amd && status.state != UpscaleState::Network &&
                  status.state != UpscaleState::StandIn && status.state != UpscaleState::Compare);
            CHECK(!status.reason.empty());
        }
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    for (bool comparison : {false, true}) {
        Reset();
        Config config; config.hosted = true;
        if (!comparison) config.upscaler = UpscalerChoice::Amd;
        else { g_compare = true; g_compareApi = &g_original; }
        auto fe = MakeFrontEnd(config);
        UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        own.version.versionId = kWrongGpuProviderVersionId;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        auto dispatch = MakeDispatch();
        CHECK(fe->Dispatch(&ctx, &dispatch.header) == FFX_API_RETURN_OK);
        CHECK(g_statuses.back().state == (comparison ? UpscaleState::Compare : UpscaleState::Amd));
        g_amdDispatchResult = FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        CHECK(fe->Dispatch(&ctx, &dispatch.header) == FFX_API_RETURN_ERROR_RUNTIME_ERROR);
        CHECK(g_statuses.back().state == UpscaleState::Failed && !g_statuses.back().reason.empty());
        g_amdDispatchResult = FFX_API_RETURN_OK;
        CHECK(fe->Dispatch(&ctx, &dispatch.header) == FFX_API_RETURN_OK);
        CHECK(g_statuses.back().state == (comparison ? UpscaleState::Compare : UpscaleState::Amd));
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
}

static void TestHostedLookupReentry() {
    Config config;
    config.hosted = true;
    FrontEnd* current = nullptr;
    int lookups = 0;
    Platform p;
    p.compareApi = [&]() -> const OriginalApi* {
        ++lookups;
        uint64_t count = 0;
        ffxQueryDescGetVersions q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        q.outputCount = &count;
        CHECK(current->Query(nullptr, &q.header) == FFX_API_RETURN_OK && count == kOwn);
        return &g_original;
    };
    FrontEnd fe(p, config);
    current = &fe;
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.outputCount = &count;
    CHECK(fe.Query(nullptr, &q.header) == FFX_API_RETURN_OK && count == kOwn + 2);
    CHECK(lookups == 1);
}

static void TestRuntimeProviderName() {
#if WRONGGPU_MODEL_K
    for (const char* model : {"M", "K"}) {
#else  // (the configured model=K is dropped, TestWithoutModelK; the host's choice of Model M is all there is)
    for (const char* model : {"M"}) {
#endif
        Reset();
        g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
        Config c;
        c.hosted = true;
        // Deliberately configure the opposite default: the in-game choice must win.
        c.runtimeOptions = std::string("model=") + (model[0] == 'K' ? "M" : "K");
        auto f = MakeFrontEnd(c);
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        create.version.versionId = model[0] == 'K' ? kWrongGpuKProviderVersionId : kWrongGpuProviderVersionId;
        ffxContext ctx = nullptr;
        CHECK(f->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        auto frame = MakeDispatch();
        CHECK(f->Dispatch(&ctx, &frame.header) == FFX_API_RETURN_OK);
        ffxQueryGetProviderVersion q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        CHECK(f->Query(&ctx, &q.header) == FFX_API_RETURN_OK);
        const std::string expected = std::string("WrongGPU Model ") + model;
        CHECK(q.versionName && expected == q.versionName);
        CHECK(q.versionId == create.version.versionId);
        CHECK(std::string(g_createInfo.options).rfind(std::string("model=") + model) == std::string(g_createInfo.options).size() - 7);
        CHECK(!g_statuses.empty() && g_statuses.back().upscaler.find(expected) != std::string::npos);
        // After a frame failure, version queries must report the actual AMD fallback.
        g_failDispatchAt = int(g_frames.size());
        CHECK(f->Dispatch(&ctx, &frame.header) == FFX_API_RETURN_OK);
        CHECK(f->Query(&ctx, &q.header) == FFX_API_RETURN_OK);
        CHECK(q.versionName && std::strcmp(q.versionName, "FSR 3.1.4 (fake)") == 0);
        CHECK(!g_statuses.empty() && g_statuses.back().state == UpscaleState::Amd);
        CHECK(f->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    CHECK(std::strcmp(RuntimeProviderName("", nullptr), "WrongGPU Model M") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=M;network=fused", "model = m"), "WrongGPU Model M") == 0);
#if !WRONGGPU_MODEL_K
    // Model K is no model of this build: named like any other value the runtime refuses.
    CHECK(std::strcmp(RuntimeProviderName("model=K", nullptr), "WrongGPU") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=M", "model=k"), "WrongGPU") == 0);
    Config config;
    config.hosted = true;
    config.runtimeOptions = "model=K";
    FrontEnd frontend(Platform{}, config);
    uint64_t count = 2, ids[2] = {};
    const char* names[2] = {};
    ffxQueryDescGetVersions query{};
    query.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    query.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    query.outputCount = &count;
    query.versionIds = ids;
    query.versionNames = names;
    CHECK(frontend.Query(nullptr, &query.header) == FFX_API_RETURN_OK);
    // Model M alone: the list holds only the SR models (RR modes have their own box since 10-31) and this build has no
    // Model K (OwnEntry(1) is null here; the test read it and crashed in a build without Model K).
    CHECK(count == 1 && ids[0] == kWrongGpuProviderVersionId && std::strcmp(names[0], kWrongGpuListedNameM) == 0);
    CHECK(OwnEntryCount() == 1 && OwnEntry(1) == nullptr);
#else
    CHECK(std::strcmp(RuntimeProviderName("model=K;network=fused", nullptr), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model = k", nullptr), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=M;model=K", nullptr), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=K", "model=M"), "WrongGPU Model M") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=M", "model=k"), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=M", " model = K "), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=M", "model=K "), "WrongGPU") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=K", "timing_log=1"), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=K", "model=M;model=K"), "WrongGPU Model K") == 0);
    CHECK(std::strcmp(RuntimeProviderName("model=invalid", "model=K"), "WrongGPU") == 0);
    Config config;
    config.hosted = true;
    config.runtimeOptions = "model=K";
    FrontEnd frontend(Platform{}, config);
    uint64_t count = 2, ids[2] = {};
    const char* names[2] = {};
    ffxQueryDescGetVersions query{};
    query.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    query.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    query.outputCount = &count;
    query.versionIds = ids;
    query.versionNames = names;
    CHECK(frontend.Query(nullptr, &query.header) == FFX_API_RETURN_OK);
    CHECK(count == 2 && ids[0] == kWrongGpuProviderVersionId && ids[1] == kWrongGpuKProviderVersionId);
    CHECK(std::strcmp(names[0], kWrongGpuListedNameM) == 0 && std::strcmp(names[1], kWrongGpuListedNameK) == 0);
#endif
}

// The items a build without Model K drops (config.h, WithoutModelK): the same in every build.
static void TestWithoutModelKItems() {
    bool removed = true;
    CHECK(WithoutModelK("", &removed).empty() && !removed);
    CHECK(WithoutModelK("timing_log=1;model=M", &removed) == "timing_log=1;model=M" && !removed);
    CHECK(WithoutModelK("model=Kx;model_k_dir=C:\\K", &removed) == "model=Kx;model_k_dir=C:\\K" && !removed);
    CHECK(WithoutModelK("model=K", &removed).empty() && removed);
    CHECK(WithoutModelK("timing_log=1;model=K;sync=pipelined", &removed) == "timing_log=1;sync=pipelined" && removed);
    CHECK(WithoutModelK(" model = k ,graph=1", &removed) == "graph=1" && removed);
    CHECK(WithoutModelK("model=M;model=K", &removed) == "model=M" && removed);
    CHECK(WithoutModelK("model=K;model=M;model=k", nullptr) == "model=M");
}

#if !WRONGGPU_MODEL_K
static void SetOptionsEnvironment(const char* value) {
#ifdef _WIN32
    _putenv_s("WG_RUNTIME_OPTIONS", value);
#else
    if (*value) setenv("WG_RUNTIME_OPTIONS", value, 1);
    else unsetenv("WG_RUNTIME_OPTIONS");
#endif
}

// A build without Model K (CMake WG_MODEL_K=OFF): model=K from wronggpu.ini or the environment is dropped with one
// log line and Model M runs; the Model K id is an override WrongGPU did not list, like any other unknown id.
static void TestWithoutModelK() {
    const char* const kNote = "Model K is not part of this build; Model M is used";
    auto runs = [&](FrontEnd& f, const UpscaleCreate& create, const char* optionKept) {
        ffxContext ctx = nullptr;
        UpscaleCreate copy = create;
        copy.upscale.header.pNext = &copy.backend.header;
        if (copy.backend.header.pNext) copy.backend.header.pNext = &copy.version.header;
        const size_t amd = g_amd.size();
        CHECK(f.CreateContext(&ctx, &copy.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        CHECK(f.OwnContextCount() == 1 && g_amd.size() == amd);  // WrongGPU's own context, nothing forwarded
        auto frame = MakeDispatch();
        CHECK(f.Dispatch(&ctx, &frame.header) == FFX_API_RETURN_OK);
        const std::string options = g_createInfo.options ? g_createInfo.options : "";
        CHECK(options.find("model=K") == std::string::npos && options.find("model=k") == std::string::npos);
        CHECK(!optionKept || options.find(optionKept) != std::string::npos);
        ffxQueryGetProviderVersion q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        CHECK(f.Query(&ctx, &q.header) == FFX_API_RETURN_OK && q.versionName && std::strcmp(q.versionName, "WrongGPU Model M") == 0);
        CHECK(q.versionId == kWrongGpuProviderVersionId);
        CHECK(!g_statuses.empty() && g_statuses.back().state == UpscaleState::Network &&
              g_statuses.back().upscaler.find("Model M") != std::string::npos);
        CHECK(f.DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    };
    const char* saved = std::getenv("WG_RUNTIME_OPTIONS");
    const std::string savedValue = saved ? saved : "";
    SetOptionsEnvironment("");
    {  // wronggpu.ini: runtime_options = ...;model=K
        Reset();
        g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
        std::string warnings;
        Config c = ParseConfig("runtime_options = timing_log=1;model=K\n", &warnings);
        CHECK(warnings.empty() && c.runtimeOptions == "timing_log=1;model=K");
        auto f = MakeFrontEnd(c);
        CHECK(CountLog(kNote) == 1 && CountLog("runtime_options in wronggpu.ini") == 1 && CountLog("WG_RUNTIME_OPTIONS") == 0);
        runs(*f, UpscaleCreate(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE), "timing_log=1");
        CHECK(CountLog(kNote) == 1);  // one line, not one per context
    }
    {  // the environment: dropped from the process environment before the runtime library is loaded and reads it
        Reset();
        g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
        SetOptionsEnvironment("timing_log=1;model=K");
        auto f = MakeFrontEnd(Config());
        const char* now = std::getenv("WG_RUNTIME_OPTIONS");
        CHECK(now && std::strcmp(now, "timing_log=1") == 0);
        CHECK(CountLog(kNote) == 1 && CountLog("(WG_RUNTIME_OPTIONS)") == 1);
        runs(*f, UpscaleCreate(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE), nullptr);
        SetOptionsEnvironment("model=k");  // alone: the variable goes
        Reset();
        Config both;
        both.runtimeOptions = "model=K";
        auto g = MakeFrontEnd(both);
        now = std::getenv("WG_RUNTIME_OPTIONS");
        CHECK(!now || !*now);
        CHECK(CountLog(kNote) == 1 && CountLog("runtime_options in wronggpu.ini and WG_RUNTIME_OPTIONS") == 1);
        SetOptionsEnvironment("");
    }
    {  // nothing to drop: no line
        Reset();
        Config c;
        c.runtimeOptions = "model=M;timing_log=1";
        auto f = MakeFrontEnd(c);
        CHECK(CountLog("Model K") == 0);
    }
    {  // a hosted create that names the Model K id: as for any id WrongGPU did not list (logged, WrongGPU upscales)
        CHECK(!IsWrongGpuProvider(kWrongGpuKProviderVersionId) && IsWrongGpuProvider(kWrongGpuProviderVersionId));
        size_t notes[2] = {};
        const uint64_t unknown[2] = {kWrongGpuKProviderVersionId, 0xDEAD0001ull};
        for (int i = 0; i < 2; ++i) {
            Reset();
            g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
            Config c;
            c.hosted = true;
            auto f = MakeFrontEnd(c);
            uint64_t count = 0;
            ffxQueryDescGetVersions list{};
            list.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
            list.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
            list.device = kDevice;
            list.outputCount = &count;
            CHECK(f->Query(nullptr, &list.header) == FFX_API_RETURN_OK && count == uint64_t(kOwn + 2));
            UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
            create.version.versionId = unknown[i];
            runs(*f, create, nullptr);
            notes[i] = CountLog("which WrongGPU did not list; WrongGPU upscales");
            CHECK(CountLog("the host chose") == 0 && CountLog("Model K") == 0);
        }
        CHECK(notes[0] == 1 && notes[1] == 1);
    }
    SetOptionsEnvironment(savedValue.c_str());
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// A host's version handling (OptiScaler 0.9.4, 0.9.5.4-pre4 and master read the same way; this simulation is written
// from that behaviour, it is not the host's code). The host keeps one "FidelityFX version" per DLL it loaded: the
// first "a.b.c" found in the FIRST name the DLL lists for upscalers, asked with no device. While that reads 0.0.0 the
// host asks again on every use and falls through to the next DLL (upscaler DLL, then the frame-generation DLL). It
// answers a game's QUERY_GET_PROVIDER_VERSION itself from that number, and it reads the number of the entry the
// player selected to decide which FSR 3.1 / FSR 4 controls and work-arounds apply.
struct HostVersion {
    unsigned major = 0, minor = 0, patch = 0;
    bool AtLeast(unsigned a, unsigned b, unsigned c) const {
        return major != a ? major > a : minor != b ? minor > b : patch >= c;
    }
};

// The first place in `name` where three unsigned numbers separated by dots can be read; false: no such place.
static bool HostParseVersion(const char* name, HostVersion& out) {
    for (const char* p = name; p && *p; ++p) {
        if (*p < '0' || *p > '9') continue;
        unsigned v[3] = {};
        const char* q = p;
        int parts = 0;
        while (parts < 3 && *q >= '0' && *q <= '9') {
            char* end = nullptr;
            v[parts++] = unsigned(std::strtoul(q, &end, 10));
            q = end;
            if (parts == 3 || *q != '.') break;
            ++q;
        }
        if (parts == 3) {
            out.major = v[0];
            out.minor = v[1];
            out.patch = v[2];
            return true;
        }
    }
    return false;
}

struct HostModule {
    PfnFfxQuery query = nullptr;
    HostVersion version;
    int queries = 0;  // ffxQuery calls the host made into this DLL
    // The host's cached version of this DLL: asked (count, then names, no device) only while it reads 0.x.
    HostVersion Version() {
        if (version.major == 0 && query) {
            uint64_t count = 0;
            ffxQueryDescGetVersions q{};
            q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
            q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
            q.outputCount = &count;
            ++queries;
            if (query(nullptr, &q.header) == FFX_API_RETURN_OK && count > 0) {
                std::vector<uint64_t> ids(count);
                std::vector<const char*> names(count);
                q.versionIds = ids.data();
                q.versionNames = names.data();
                ++queries;
                if (query(nullptr, &q.header) == FFX_API_RETURN_OK) HostParseVersion(names[0], version);
            }
        }
        return version;
    }
};

// What the host answers a game that asks for the provider version: the upscaler DLL's number, else (while that
// reads 0.x) the frame-generation DLL's, packed as the host packs it.
struct HostProviderAnswer {
    uint64_t id = 0;
    std::string name;
    bool fromFrameGeneration = false;
};
static HostProviderAnswer HostProviderVersion(HostModule& upscaler, HostModule& frameGeneration) {
    HostProviderAnswer a;
    HostVersion v = upscaler.Version();
    if (v.major == 0) {
        v = frameGeneration.Version();
        a.fromFrameGeneration = true;
    }
    a.id = (uint64_t(v.major) << 22 | uint64_t(v.minor) << 12 | v.patch) & 0xFFFFFFFFu;
    a.name = std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch);
    return a;
}

static FrontEnd* g_hostedFrontEnd = nullptr;
static ffxReturnCode_t HostedQuery(ffxContext* context, ffxQueryDescHeader* desc) { return g_hostedFrontEnd->Query(context, desc); }
static ffxReturnCode_t FrameGenerationQuery(ffxContext*, ffxQueryDescHeader* desc) {
    static const char* name = "3.1.6";
    auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
    const uint64_t capacity = *q->outputCount;
    *q->outputCount = 1;
    if (capacity && q->versionIds) q->versionIds[0] = 0xF6;
    if (capacity && q->versionNames) q->versionNames[0] = name;
    return FFX_API_RETURN_OK;
}

static void TestHostVersionHandling() {
    // AMD's DLLs name their entries by the version alone; the fake AMD DLL does so for this test.
    const char* const savedNames[2] = {g_versionNames[0], g_versionNames[1]};
    g_versionNames[0] = "3.1.5";
    g_versionNames[1] = "2.3.4";

    // The parser itself.
    HostVersion parsed;
    CHECK(HostParseVersion("4.1.1", parsed) && parsed.major == 4 && parsed.minor == 1 && parsed.patch == 1);
    CHECK(HostParseVersion("FSR 3.1.4 (fake)", parsed) && parsed.major == 3 && parsed.minor == 1 && parsed.patch == 4);
    CHECK(HostParseVersion("v2 then 2.3.4", parsed) && parsed.major == 2 && parsed.minor == 3 && parsed.patch == 4);
    CHECK(!HostParseVersion("3.1", parsed) && !HostParseVersion("", parsed));

    // The defect of 0.4.1 / 0.4.5-rc1: a first name without a number reads 0.0.0; the host then answers the game from
    // the frame-generation DLL instead and asks the upscaler DLL again on every use.
    HostVersion none;
    CHECK(!HostParseVersion(kWrongGpuVersionName, none) && none.major == 0);
    CHECK(!HostParseVersion("WrongGPU Model K", none));

    // With AMD's DLL as the host's upscaler DLL (the reference behaviour).
    Reset();
    HostModule amd{AmdQuery}, amdFrameGeneration{FrameGenerationQuery};
    const HostProviderAnswer reference = HostProviderVersion(amd, amdFrameGeneration);
    CHECK(reference.name == "3.1.5" && !reference.fromFrameGeneration && amd.queries == 2 && amdFrameGeneration.queries == 0);

    // With WrongGPU's DLL as the host's upscaler DLL.
    Reset();
    Config config;
    config.hosted = true;
    auto fe = MakeFrontEnd(config);
    g_hostedFrontEnd = fe.get();
    HostModule wronggpu{HostedQuery}, frameGeneration{FrameGenerationQuery};
    const HostProviderAnswer answer = HostProviderVersion(wronggpu, frameGeneration);
    // From the upscaler DLL, with as many queries as with AMD's DLL. The number is WrongGPU's own 3.1 level (3.1.0,
    // hidden after "##" in the first name; owner 2026-10-31), not AMD's 3.1.5: an FSR 3.1 level all the same.
    CHECK(answer.name == "3.1.0" && answer.id == ((3ull << 22) | (1ull << 12)) && !answer.fromFrameGeneration);
    CHECK(wronggpu.queries == 2 && frameGeneration.queries == 0);
    // Cached: the host does not ask again.
    HostProviderVersion(wronggpu, frameGeneration);
    CHECK(wronggpu.queries == 2 && frameGeneration.queries == 0);
    // Every step the host compares against: an FSR 3.1 DLL, below 3.1.1 (the host's FSR 3.1 presets and manual
    // tuning stay hidden; the key-value tuning it sends from 3.1.1 / 3.1.4 is not for WrongGPU), never FSR 4.
    const HostVersion v = wronggpu.Version();
    CHECK(v.major == 3 && v.minor == 1 && v.patch == 0);
    CHECK(v.AtLeast(3, 1, 0) && !v.AtLeast(3, 1, 1) && !v.AtLeast(3, 1, 4) && v.major > 2);
    CHECK(!v.AtLeast(4, 0, 0) && !v.AtLeast(4, 0, 2) && !v.AtLeast(4, 1, 1) && v.major < 4);

    // The host's own list (with a device): both WrongGPU entries read the same number in front of their labels, the
    // ids are WrongGPU's, AMD's entries follow unchanged. A parser that reads only the start of a name gets it too.
    uint64_t count = 0;
    ffxQueryDescGetVersions list{};
    list.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    list.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    list.device = kDevice;
    list.outputCount = &count;
    CHECK(fe->Query(nullptr, &list.header) == FFX_API_RETURN_OK && count == kOwn + 2);
    uint64_t ids[16] = {};
    const char* names[16] = {};
    list.versionIds = ids;
    list.versionNames = names;
    CHECK(fe->Query(nullptr, &list.header) == FFX_API_RETURN_OK && count == kOwn + 2);
    CHECK(ids[0] == 0x5747000000000001ull && ids[kOwn] == g_versionIds[0] && ids[kOwn + 1] == g_versionIds[1]);
    CHECK(kOwn == 1 || ids[1] == 0x5747000000000002ull);
    CHECK(std::strcmp(names[kOwn], "3.1.5") == 0 && std::strcmp(names[kOwn + 1], "2.3.4") == 0);
    for (int i = 0; i < kOwn; ++i) {
        HostVersion selected;
        selected.major = 4;  // the number of an FSR 4 entry selected before must not survive
        CHECK(HostParseVersion(names[i], selected) && selected.major == 3 && selected.minor == 1 && selected.patch == 0);
        // WrongGPU's label first, then the API level hidden after "##" (the host's menu draws only the label); no
        // other maker's product is named and no number is shown.
        const char* hidden = std::strstr(names[i], "##");
        CHECK(hidden && std::strcmp(hidden, "##3.1.0") == 0 && std::strncmp(names[i], "WrongGPU", 8) == 0);
        // (the registry's labels: "WrongGPU Model M", "WrongGPU Model K", "WrongGPU RR exact", ...)
        CHECK(hidden && std::string(names[i], hidden) == EntryProviderName(*OwnEntry(i)));
    }
    CHECK(std::strcmp(names[0], "WrongGPU Model M##3.1.0") == 0);
    CHECK(WRONGGPU_MODEL_K == 0 || std::strcmp(names[1], "WrongGPU Model K##3.1.0") == 0);
    // The selection still works by id, and a context's own provider name stays the label alone (the last SR model).
    const uint64_t lastModel = WRONGGPU_MODEL_K ? kWrongGpuKProviderVersionId : kWrongGpuProviderVersionId;
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    own.version.versionId = lastModel;
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    ffxQueryGetProviderVersion provider{};
    provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    CHECK(fe->Query(&ctx, &provider.header) == FFX_API_RETURN_OK && provider.versionId == lastModel && provider.versionName &&
          std::strcmp(provider.versionName, WRONGGPU_MODEL_K ? "WrongGPU Model K" : "WrongGPU Model M") == 0);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);

    g_hostedFrontEnd = nullptr;
    g_versionNames[0] = savedNames[0];
    g_versionNames[1] = savedNames[1];
}

// A dispatch list the front end can never split (another hook layer's own vtable, d3d12_hooks.cpp): the context of a
// backend that needs the split goes to AMD's FSR with one line in the log; without AMD the backend keeps being asked
// (its split answers WG_ERROR_NOT_READY and the stand-in draws); a backend that needs no split is not affected.
// clean/fe-wrapped-list: frames the upscaler does not take (WG_ERROR_UNSUPPORTED, the network runtime with
// hip_switch=off at native resolution): AMD's FSR takes the context over at once; without it the stand-in upscales
// those frames and the context stays WrongGPU's. And an upscaler's "about to ..." line flushes the log.
static void TestUnsupportedFrames() {
    Reset();
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    Config config;
    config.hosted = true;
    g_aboutTo = true;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    own.version.versionId = kWrongGpuProviderVersionId;
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    CHECK(CountLog("about to start the HIP runtime (test)") == 1 && g_flushes == 1);  // written to disk right after the line
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1 && g_statuses.back().state == UpscaleState::Network);
    g_unsupportedFrom = 1;
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 2 && g_standInFrames.empty());
    CHECK(!g_amd.empty() && g_amd.back().fn == "dispatch" && g_statuses.back().state == UpscaleState::Amd);
    CHECK(CountLog("(the upscaler does not take this frame); AMD FSR upscales this context instead of WrongGPU") == 1);
    CHECK(CountLog("now runs on AMD FSR for the rest of its life") == 1);
    for (int i = 0; i < 3; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_amd.back().fn == "dispatch");
    CHECK(g_frames.size() == 2 && g_statuses.back().state == UpscaleState::Amd);  // the upscaler is not asked again
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);

    // No AMD DLL: the stand-in upscales the frames the upscaler does not take; the context stays WrongGPU's; said once.
    Reset();
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_originalAvailable = false;
    fe = MakeFrontEnd(config);
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    CHECK(g_flushes == 0);  // nothing logged "about to ...": no flush
    g_unsupportedFrom = 0;
    for (int i = 0; i < 3; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 3 && g_standInFrames.size() == 3 && g_amd.empty());
    CHECK(g_statuses.back().state == UpscaleState::StandIn);
    CHECK(CountLog("AMD FSR is not available for this context; the stand-in upscales the frames the upscaler does not take") == 1);
    CHECK(CountLog("(the upscaler does not take this frame)") == 1);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
}

static void TestSplitBlocked() {
    Reset();
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    Config config;
    config.hosted = true;
    auto fe = MakeFrontEnd(config);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
    own.version.versionId = kWrongGpuProviderVersionId;
    ffxContext ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    ffxDispatchDescUpscale d = MakeDispatch();
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1);  // a list that can be split
    CHECK(g_splitBlockedAsked == 1 && g_statuses.back().state == UpscaleState::Network);
    g_splitBlocked = "the command list has its own vtable";
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 1);  // the backend was not asked to draw into that list
    CHECK(g_amd.back().fn == "dispatch" && g_statuses.back().state == UpscaleState::Amd);
    CHECK(g_statuses.back().reason.find("the command list has its own vtable") != std::string::npos);
    CHECK(CountLog("the command list has its own vtable; AMD FSR upscales this context instead of WrongGPU") == 1);
    CHECK(CountLog("now runs on AMD FSR for the rest of its life") == 1);
    const int asked = g_splitBlockedAsked;
    for (int i = 0; i < 3; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_amd.back().fn == "dispatch");
    CHECK(g_frames.size() == 1 && g_splitBlockedAsked == asked && g_statuses.back().state == UpscaleState::Amd);
    CHECK(CountLog("AMD FSR upscales this context instead of WrongGPU") == 1);  // said once
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);

    // No AMD DLL: the context stays WrongGPU's, the backend is asked every frame, the reason is said once.
    Reset();
    g_originalAvailable = false;
    g_splitBlocked = "the command list has its own vtable";
    fe = MakeFrontEnd(config);
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
    g_notReadyUntil = 100;  // as the split answers for such a list
    for (int i = 0; i < 3; ++i) CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_frames.size() == 3 && g_standInFrames.size() == 3 && g_amd.empty());
    CHECK(CountLog("AMD FSR upscales this context instead of WrongGPU") == 1);
    CHECK(CountLog("AMD FSR is not available for this context; the stand-in upscales its frames") == 1);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);

    // A backend that does not need the split is never asked.
    Reset();
    g_useBackend = &g_backendV1;
    g_splitBlocked = "the command list has its own vtable";
    fe = MakeFrontEnd(config);
    ctx = nullptr;
    CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1 && g_splitBlockedAsked == 0);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
}

// OptiScaler ASI plugin hooked (clean/fsr/asi): WrongGPU is an entry of the host's MAIN list. The FFX version list is
// the same as without the plugin (WrongGPU first: the host reads the DLL's version from the first name, and the
// sub-box indexes stay put). The plugin's entry decides: "WrongGPU" runs WrongGPU whatever the sub-box names; the
// host's FSR 3.X(/4) entry runs AMD: the sub-box's AMD version, else AMD's newest listed version.
static void TestAsiPluginHooked() {
    Reset();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    Config config;
    config.hosted = true;
    config.upscaler = UpscalerChoice::Network;
    g_asiPresent = true;
    g_asiHooked = true;
    g_asiModel = 'M';
    auto fe = MakeFrontEnd(config);
    const uint64_t own = uint64_t(kOwn);

    uint64_t count = 8, ids[8] = {};
    const char* names[8] = {};
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = kDevice;
    versions.outputCount = &count;
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK);
    CHECK(count == own + 2 && ids[0] == kWrongGpuProviderVersionId && ids[own] == g_versionIds[0] && ids[own + 1] == g_versionIds[1]);
    CHECK(std::strcmp(names[0], kWrongGpuListedNameM) == 0);

    auto ownRun = [&](uint64_t overrideId, const char* provider, uint64_t providerId) {
        const size_t amdBefore = g_amd.size();
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/overrideId != 0);
        create.version.versionId = overrideId;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        CHECK(fe->OwnContextCount() == 1 && g_amd.size() == amdBefore);
        ffxQueryGetProviderVersion pv{};
        pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        CHECK(fe->Query(&ctx, &pv.header) == FFX_API_RETURN_OK);
        CHECK(pv.versionId == providerId && pv.versionName && std::strcmp(pv.versionName, provider) == 0);
        ffxDispatchDescUpscale d = MakeDispatch();
        const size_t frames = g_frames.size();
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == frames + 1 && g_amd.size() == amdBefore);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    };
    // "WrongGPU": WrongGPU upscales with the sub-box at WrongGPU, at an AMD version, or with no override at all.
    ownRun(kWrongGpuProviderVersionId, "WrongGPU Model M", kWrongGpuProviderVersionId);
    ownRun(g_versionIds[0], "WrongGPU Model M", kWrongGpuProviderVersionId);
    ownRun(0, "WrongGPU Model M", kWrongGpuProviderVersionId);
    g_asiModel = 'K';
#if WRONGGPU_MODEL_K
    ownRun(kWrongGpuProviderVersionId, "WrongGPU Model K", kWrongGpuKProviderVersionId);
#else
    ownRun(kWrongGpuProviderVersionId, "WrongGPU Model M", kWrongGpuProviderVersionId);  // no Model K in this build
#endif

    // The host's FSR 3.X(/4) entry: AMD upscales.
    g_asiModel = 0;
    auto amdRun = [&](uint64_t overrideId, bool withOverride, uint64_t expectAmd) {
        const size_t amdBefore = g_amd.size();
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, withOverride);
        create.version.versionId = overrideId;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        CHECK(g_amd.size() == amdBefore + 1 && g_amd.back().fn == "create" && g_amd.back().overrideId == expectAmd);
        const size_t frames = g_frames.size();
        ffxDispatchDescUpscale d = MakeDispatch();
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == frames && g_amd.back().fn == "dispatch");
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
        CHECK(fe->OwnContextCount() == 0);
    };
    amdRun(g_versionIds[1], true, g_versionIds[1]);             // the sub-box names an AMD version: that one
    amdRun(kWrongGpuProviderVersionId, true, g_versionIds[0]);  // the sub-box names WrongGPU: AMD's newest (3.1.4)
    CHECK(CountLog("OptiScaler plugin: AMD FSR selected") == 2);
    CHECK(CountLog("WrongGPU upscales") == 0);
}

// Plugin present but not hooked (unknown OptiScaler build): exactly hosted mode without a plugin -- the FFX sub-box
// lists WrongGPU first and its selection decides.
static void TestAsiPluginNotHooked() {
    Reset();
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    Config config;
    config.hosted = true;
    g_asiPresent = true;
    g_asiHooked = false;
    g_asiModel = 'M';  // ignored while not hooked
    auto fe = MakeFrontEnd(config);
    uint64_t count = 8, ids[8] = {};
    const char* names[8] = {};
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = kDevice;
    versions.outputCount = &count;
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK);
    const uint64_t own = uint64_t(kOwn);
    CHECK(count == own + 2 && ids[0] == kWrongGpuProviderVersionId);
    // The sub-box names AMD's version: forwarded to AMD although the (unhooked) plugin says M.
    UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
    create.version.versionId = g_versionIds[0];
    ffxContext ctx = nullptr;
    const size_t amdBefore = g_amd.size();
    CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK);
    CHECK(g_amd.size() == amdBefore + 1 && g_amd.back().overrideId == g_versionIds[0] && fe->OwnContextCount() == 0);
    CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    CHECK(CountLog("OptiScaler plugin") == 0);
}

// The installer's Preset K with the plugin hooked (0.5.0): wronggpu.ini says runtime_options = model=K, OptiScaler.ini
// UpscalerIndex=1, and the installer seeds the plugin's wronggpu-asi.ini with selection=wronggpu-k, so the plugin
// reports 'K' and the context runs Model K. The plugin's entry decides over the ini: without that seed its default
// entry (Model M) would run, and a later choice of Model M in OptiScaler's list runs Model M.
static void TestAsiPluginPresetK() {
#if WRONGGPU_MODEL_K
    const bool present = g_asiPresent, hooked = g_asiHooked;
    const char saved = g_asiModel;
    for (char plugin : {'K', 'M'}) {
        Reset();
        g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
        std::string warnings;
        Config c = ParseConfig("runtime_options = model=K\n", &warnings);
        CHECK(warnings.empty());
        c.hosted = true;
        c.upscaler = UpscalerChoice::Network;
        g_asiPresent = true;
        g_asiHooked = true;
        g_asiModel = plugin;
        auto fe = MakeFrontEnd(c);
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        create.version.versionId = kWrongGpuKProviderVersionId;  // the FFX sub-box at UpscalerIndex=1
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
        auto frame = MakeDispatch();
        CHECK(fe->Dispatch(&ctx, &frame.header) == FFX_API_RETURN_OK);
        ffxQueryGetProviderVersion q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        CHECK(fe->Query(&ctx, &q.header) == FFX_API_RETURN_OK);
        const std::string expected = std::string("WrongGPU Model ") + plugin;
        CHECK(q.versionName && expected == q.versionName);
        CHECK(q.versionId == (plugin == 'K' ? kWrongGpuKProviderVersionId : kWrongGpuProviderVersionId));
        const std::string options = g_createInfo.options ? g_createInfo.options : "";
        CHECK(options.size() >= 7 && options.compare(options.size() - 7, 7, std::string("model=") + plugin) == 0);
        CHECK(CountLog((std::string("OptiScaler plugin: ") + expected + " selected").c_str()) == 1);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    g_asiPresent = present;
    g_asiHooked = hooked;
    g_asiModel = saved;
#endif
}

// i15: WrongGPU's version-list entries come from the registry (clean/wgshared). i20 (owner 2026-10-31): only the SR
// models; the RR modes are a box of their own in the RR plugin's menu tab. An entry sets the context's model and
// writes sr_model; one folder per choice lists its one entry and applies it without an override; a module that is not
// the writer writes nothing.
static void TestRegistryEntries() {
    // The registry keeps the two historical ids and labels in front (the host reads the API level from the first).
    CHECK(kWgEntries[0].id == kWrongGpuProviderVersionId && std::strcmp(kWgEntries[0].label, kWrongGpuListedNameM) == 0);
#if WRONGGPU_MODEL_K
    CHECK(kWgEntries[1].id == kWrongGpuKProviderVersionId && std::strcmp(kWgEntries[1].label, kWrongGpuListedNameK) == 0);
#endif
    CHECK(WG_ENTRY_COUNT == WG_SR_MODEL_COUNT);  // no RR entries
    CHECK(kOwn == WG_ENTRY_COUNT - (WRONGGPU_MODEL_K ? 0 : 1));
    CHECK(std::strcmp(EntryProviderName(kWgEntries[0]), "WrongGPU Model M") == 0);
    for (int i = 0; i < kOwn; ++i) {
        const WgEntryDef* e = OwnEntry(i);
        CHECK(e && std::strstr(e->label, "##" WG_API_LEVEL_PREFIX) != nullptr && e->slot == WG_OPT_SR_MODEL);
    }
    const WgEntryDef* modelM = SrEntryOfModel(WG_SR_MODEL_M);
    CHECK(modelM && modelM->id == kWrongGpuProviderVersionId);
    if (!modelM) return;
    CHECK(ModelOfName("k") == (WRONGGPU_MODEL_K ? WG_SR_MODEL_K : -1) && ModelOfName("M") == WG_SR_MODEL_M && ModelOfName("Q") == -1);

    Reset();
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_menuOn = true;
    Config config;
    config.hosted = true;
    auto fe = MakeFrontEnd(config);
    uint64_t count = 16, ids[16] = {};
    const char* names[16] = {};
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = kDevice;
    versions.outputCount = &count;
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(fe->Query(nullptr, &versions.header) == FFX_API_RETURN_OK && count == uint64_t(kOwn + 2));
    for (int i = 0; i < kOwn; ++i) CHECK(ids[i] == OwnEntry(i)->id && std::strcmp(names[i], OwnEntry(i)->label) == 0);
    CHECK(ids[kOwn] == g_versionIds[0] && ids[kOwn + 1] == g_versionIds[1]);
    for (uint64_t i = 0; i < count; ++i) CHECK(std::strstr(names[i], "WrongGPU RR") == nullptr);

    auto run = [&](uint64_t id, bool withOverride) -> std::string {
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, withOverride);
        create.version.versionId = id;
        ffxContext ctx = nullptr;
        const size_t amdBefore = g_amd.size();
        CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        CHECK(fe->OwnContextCount() == 1 && g_amd.size() == amdBefore);  // never AMD: the entry is WrongGPU's
        ffxQueryGetProviderVersion pv{};
        pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        CHECK(fe->Query(&ctx, &pv.header) == FFX_API_RETURN_OK);
        const std::string provider = pv.versionName ? pv.versionName : "";
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
        return provider;
    };
    // Model M chosen: sr_model=M written (menu), the context runs Model M.
    CHECK(run(modelM->id, true) == "WrongGPU Model M");
    CHECK(g_menuWrites.size() == 1 && std::get<0>(g_menuWrites[0]) == WG_OPT_SR_MODEL &&
          std::get<1>(g_menuWrites[0]) == WG_SR_MODEL_M && std::get<2>(g_menuWrites[0]) == "menu");
    CHECK(CountLog("the host chose WrongGPU Model M (0x5747000000000001): settings: sr_model=M (menu)") == 1);
#if WRONGGPU_MODEL_K
    // Model K chosen: sr_model=K written, the context runs Model K.
    CHECK(run(kWrongGpuKProviderVersionId, true) == "WrongGPU Model K");
    CHECK(g_menuWrites.size() == 2 && std::get<0>(g_menuWrites[1]) == WG_OPT_SR_MODEL && std::get<1>(g_menuWrites[1]) == WG_SR_MODEL_K);
    CHECK(g_createOptions.size() >= 8 && g_createOptions.compare(g_createOptions.size() - 8, 8, ";model=K") == 0);
    CHECK(g_menuValues[WG_OPT_SR_MODEL] == WG_SR_MODEL_K);
    // No override at all (the host passes none): sr_model as configured (K), nothing written.
    CHECK(run(0, false) == "WrongGPU Model K" && g_menuWrites.size() == 2);
    CHECK(CountLog("settings: sr_model=K (ini)") >= 1);
#endif
    // Every entry the registry has: its option = its value, and nothing writes rr_mode (the RR plugin's box owns it).
    for (int i = 0; i < kOwn; ++i) {
        const WgEntryDef* e = OwnEntry(i);
        CHECK(!run(e->id, true).empty());
        CHECK(!g_menuWrites.empty() && std::get<0>(g_menuWrites.back()) == e->slot && std::get<1>(g_menuWrites.back()) == e->value);
        CHECK(CountLog((std::string("the host chose ") + EntryProviderName(*e) + " (").c_str()) >= 1);
    }
    for (const auto& w : g_menuWrites) CHECK(std::get<0>(w) != WG_OPT_RR_MODE);
    std::printf("registry entries: %d own entries listed and chosen\n", kOwn);
    // Not the writer: the entry still runs, nothing is written, the log says so.
    g_menuWriter = false;
    CHECK(!run(modelM->id, true).empty());
    CHECK(CountLog("(not written: another module writes the shared settings)") == 1);
    g_menuWriter = true;
    // An AMD id is still AMD's (forwarded, nothing written).
    {
        const size_t writes = g_menuWrites.size();
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        create.version.versionId = g_versionIds[0];
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 0);
        CHECK(g_amd.back().fn == "create" && g_amd.back().overrideId == g_versionIds[0] && g_menuWrites.size() == writes);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    // A context that falls back to AMD never shows AMD WrongGPU's entry id (it is removed from the chain).
    {
        g_failCreate = true;
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        create.version.versionId = modelM->id;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK);
        CHECK(!g_amd.empty() && g_amd.back().fn == "create" && g_amd.back().overrideId != modelM->id);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
        g_failCreate = false;
    }

    // One folder per choice: the folder of Model M lists that one entry (no AMD versions) and applies it without an
    // override, with the source word "folder".
    Reset();
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_menuOn = true;
    g_folderEntry = modelM->id;
    auto folder = MakeFrontEnd(config);
    count = 0;
    versions.versionIds = nullptr;
    versions.versionNames = nullptr;
    CHECK(folder->Query(nullptr, &versions.header) == FFX_API_RETURN_OK && count == 1);
    count = 16;
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(folder->Query(nullptr, &versions.header) == FFX_API_RETURN_OK && count == 1 && ids[0] == modelM->id &&
          std::strcmp(names[0], "WrongGPU Model M##3.1.0") == 0);
    {
        UpscaleCreate create(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, false);
        ffxContext ctx = nullptr;
        CHECK(folder->CreateContext(&ctx, &create.upscale.header, kMemCb) == FFX_API_RETURN_OK && folder->OwnContextCount() == 1);
        CHECK(g_menuWrites.size() == 1 && std::get<0>(g_menuWrites[0]) == WG_OPT_SR_MODEL &&
              std::get<1>(g_menuWrites[0]) == WG_SR_MODEL_M && std::get<2>(g_menuWrites[0]) == "folder");
        CHECK(CountLog("settings: sr_model=M (folder)") == 1);
        CHECK(folder->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    // The [wronggpu] keys of the registry are no unknown keys of the front end's ini reader; the default text has both.
    std::string warnings;
    ParseConfig("[wronggpu]\nsr_model = K\nrr_mode = fast\ndiag_env_1 =\n[fsr]\nupscaler = network\n", &warnings);
    CHECK(warnings.empty());
    ParseConfig("[wronggpu]\nnot_a_key = 1\n", &warnings);
    CHECK(warnings.find("unknown key \"not_a_key\"") != std::string::npos);
    const std::string text = DefaultConfigText();
    CHECK(text.find("[wronggpu]\r\n") != std::string::npos && text.find("rr_mode = accurate\r\n") != std::string::npos &&
          text.find("sr_model = M\r\n") != std::string::npos && text.find("[fsr]\r\n") != std::string::npos &&
          text.find("upscaler = network\r\n") != std::string::npos);
}

// ---------------------------------------------------------------------------------------------------------------
// A host like OptiScaler's custom slot as the Cyberpunk log of 2026-10-30 shows it: the FidelityFX DLL WrongGPU found
// in the game is the host's way in, and an upscale dispatch through it goes to the host's selected upscaler, WrongGPU:
// the host asks WrongGPU for its versions, creates a context with WrongGPU's version id, configures and dispatches it.
// An AMD fallback through that DLL therefore re-enters WrongGPU on the same thread, each time with a new context.

struct HostContext {
    ffxContext feature = nullptr;  // the host's upscaler (WrongGPU's context) for this game context
};
static FrontEnd* g_hostFe = nullptr;
static int g_hostDepth = 0, g_hostMaxDepth = 0;
static const int kHostCap = 40;  // the test's stop; the real host ran until the thread's stack gave out (36 deep)
static int g_hostCreates = 0, g_hostDispatches = 0, g_hostDestroys = 0;
static int g_hostFeatureCreates = 0, g_hostFeatureFailures = 0;
static std::vector<HostContext*> g_hostContexts;

// The host's frame for one game upscale: its feature (WrongGPU) is created when it has none (versions, create with
// WrongGPU's id, the FSR tuning keys), then dispatched. A failed create: the host's own upscaler does the frame.
static ffxReturnCode_t HostUpscale(ffxContext* feature) {
    if (!*feature) {
        uint64_t count = 0;
        ffxQueryDescGetVersions q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        q.device = kDevice;
        q.outputCount = &count;
        g_hostFe->Query(nullptr, &q.header);
        std::vector<uint64_t> ids(size_t(count ? count : 1));
        std::vector<const char*> names(size_t(count ? count : 1));
        q.versionIds = ids.data();
        q.versionNames = names.data();
        g_hostFe->Query(nullptr, &q.header);
        UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        c.version.versionId = kWrongGpuProviderVersionId;
        ++g_hostFeatureCreates;
        if (g_hostFe->CreateContext(feature, &c.upscale.header, nullptr) != FFX_API_RETURN_OK || !*feature) {
            ++g_hostFeatureFailures;
            *feature = nullptr;
            return FFX_API_RETURN_OK;
        }
        ffxConfigureDescUpscaleKeyValue kv{};
        kv.header.type = FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE;
        g_hostFe->Configure(feature, &kv.header);
    }
    auto d = MakeDispatch();
    return g_hostFe->Dispatch(feature, &d.header);
}
static ffxReturnCode_t HostCreate(ffxContext* context, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*) {
    ++g_hostCreates;
    auto* h = new HostContext;
    g_hostContexts.push_back(h);
    *context = h;
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t HostDispatch(ffxContext* context, const ffxDispatchDescHeader*) {
    ++g_hostDispatches;
    if (g_hostDepth >= kHostCap) return FFX_API_RETURN_OK;
    ++g_hostDepth;
    g_hostMaxDepth = std::max(g_hostMaxDepth, g_hostDepth);
    HostUpscale(&static_cast<HostContext*>(*context)->feature);
    --g_hostDepth;
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t HostDestroy(ffxContext* context, const ffxAllocationCallbacks*) {
    ++g_hostDestroys;
    auto* h = static_cast<HostContext*>(*context);
    if (h->feature) g_hostFe->DestroyContext(&h->feature, nullptr);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t HostConfigure(ffxContext*, const ffxConfigureDescHeader*) { return FFX_API_RETURN_OK; }
static ffxReturnCode_t HostQuery(ffxContext*, ffxQueryDescHeader* desc) {
    if (desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) *reinterpret_cast<ffxQueryDescGetVersions*>(desc)->outputCount = 0;
    return FFX_API_RETURN_OK;
}
static OriginalApi g_hostApi{HostCreate, HostDestroy, HostConfigure, HostQuery, HostDispatch};

// An AMD DLL whose dispatch dispatches WrongGPU's context again (the same context, same thread).
static ffxContext* g_againContext = nullptr;
static ffxReturnCode_t g_againResult = FFX_API_RETURN_OK;
static ffxReturnCode_t AgainDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    AmdDispatch(context, desc);
    if (g_againContext) {
        auto d = MakeDispatch();
        g_againResult = g_hostFe->Dispatch(g_againContext, &d.header);
    }
    return FFX_API_RETURN_OK;
}
static OriginalApi g_againApi{AmdCreate, AmdDestroy, AmdConfigure, AmdQuery, AgainDispatch};

static void ResetHost() {
    for (HostContext* h : g_hostContexts) delete h;
    g_hostContexts.clear();
    g_hostFe = nullptr;
    g_hostDepth = g_hostMaxDepth = 0;
    g_hostCreates = g_hostDispatches = g_hostDestroys = 0;
    g_hostFeatureCreates = g_hostFeatureFailures = 0;
}

// The churn of 2026-10-30, custom slot, WrongGPU's upscaler failing (no model) and no AMD DLL of WrongGPU's own: the
// create fails cleanly (the host's own upscaler takes over), the host's FidelityFX DLL is never called, and the
// host asking again every frame costs no new attempt and almost no log.
static void TestCustomSlotFailedCreate() {
    Reset();
    ResetHost();
    g_originalAvailable = false;  // no amd_fidelityfx_dx12.wronggpu-original.dll next to the DLL
    g_compareApi = &g_hostApi;    // what the provider search finds in the game: the host's way in
    g_failCreate = true;
    g_activateOn = true;
    Config config;
    config.hosted = true;
    config.customSlot = true;
    auto fe = MakeFrontEnd(config);
    g_hostFe = fe.get();
    ffxContext feature = nullptr;
    for (int frame = 0; frame < 100; ++frame) HostUpscale(&feature);
    CHECK(feature == nullptr && fe->OwnContextCount() == 0);
    CHECK(g_hostFeatureCreates == 100 && g_hostFeatureFailures == 100);  // the host asked every frame
    CHECK(g_createCalls == 1);                                          // one real attempt
    CHECK(g_hostCreates == 0 && g_hostDispatches == 0 && g_hostMaxDepth == 0);  // the host's DLL is never called
    CHECK(g_amd.empty());
    CHECK(g_activateCalls == 0);  // no hooks for a failed create
    // The version list: WrongGPU's own entries only (no AMD entries from the host's DLL).
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.outputCount = &count;
    CHECK(fe->Query(nullptr, &q.header) == FFX_API_RETURN_OK && count == uint64_t(kOwn));
    // Logged once, then a summary line at repeats 1, 2, 3 and 10 (of 99).
    CHECK(CountLog("ffxCreateContext") == 1 && CountLog("WrongGPU upscaler \"fake upscaler\" create:") == 1);
    CHECK(CountLog("this failure is remembered for the session") == 1);
    CHECK(CountLog("no context, the host's own upscaler takes over") == 1);
    CHECK(CountLog("custom slot: create ") == 4 && CountLog("custom slot: create 10 since the failure") == 1);
    // Another model is something new: one attempt of its own.
#if WRONGGPU_MODEL_K
    {
        UpscaleCreate k(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        k.version.versionId = kWrongGpuKProviderVersionId;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &k.upscale.header, nullptr) == FFX_API_RETURN_ERROR_RUNTIME_ERROR && !ctx);
        CHECK(g_createCalls == 2 && CountLog("this failure is remembered for the session") == 2);
        CHECK(fe->CreateContext(&ctx, &k.upscale.header, nullptr) == FFX_API_RETURN_ERROR_RUNTIME_ERROR && g_createCalls == 2);
    }
#endif
    // A working upscaler (the model was there) in a fresh session: created, hooks after the create.
    Reset();
    ResetHost();
    g_originalAvailable = false;
    g_compareApi = &g_hostApi;
    g_activateOn = true;
    auto fe2 = MakeFrontEnd(config);
    g_hostFe = fe2.get();
    feature = nullptr;
    for (int frame = 0; frame < 3; ++frame) HostUpscale(&feature);
    CHECK(feature != nullptr && fe2->OwnContextCount() == 1 && g_createCalls == 1 && g_frames.size() == 3);
    CHECK(g_activateCalls == 1 && g_createCallsAtActivate == 1);
    CHECK(g_hostCreates == 0 && g_hostDispatches == 0);
    CHECK(fe2->DestroyContext(&feature, nullptr) == FFX_API_RETURN_OK && !feature && fe2->OwnContextCount() == 0);
    ResetHost();
}

// 0.6.0 follow-up (a tester's CONTROL Resonant): custom slot, the upscaler refusing the game's inputs for good
// (WG_ERROR_INPUT: a colour texture with mips before the runtime took them) and the host making a new context after
// every failed dispatch (393 in 3 minutes): one create, one dispatch, then the remembered failure makes every new
// create fail at once and the host's own upscaler takes over; said once. Another failure (WG_ERROR_DEVICE) keeps
// today's behaviour: the host gets a new context of WrongGPU's after it.
static void TestCustomSlotInputRefused() {
    auto session = [](int frames, int* errors, int inputFrom, int failAt, std::set<int> inputAt = {}, int deviceFrom = -1) {
        Reset();
        ResetHost();
        g_inputErrorFrom = inputFrom;
        g_inputErrorAt = inputAt;
        g_deviceErrorFrom = deviceFrom;
        g_failDispatchAt = failAt;
        g_originalAvailable = false;  // no AMD DLL of WrongGPU's own: the custom slot's usual case
        g_compareApi = &g_hostApi;
        g_activateOn = true;
        Config config;
        config.hosted = true;
        config.customSlot = true;
        auto fe = MakeFrontEnd(config);
        g_hostFe = fe.get();
        ffxContext feature = nullptr;
        *errors = 0;
        for (int frame = 0; frame < frames; ++frame)
            if (HostUpscale(&feature) != FFX_API_RETURN_OK) {  // the host: a new context after a failed dispatch
                ++*errors;
                if (feature) fe->DestroyContext(&feature, nullptr);
                feature = nullptr;
            }
        if (feature) fe->DestroyContext(&feature, nullptr);
        g_hostFe = nullptr;
        return fe;
    };
    int errors = 0;
    {
        auto fe = session(50, &errors, 0, -1);
        CHECK(g_createCalls == 3);                            // three real creates (bug hunt A: three refusals in a row)
        CHECK(g_frames.size() == 3 && errors == 3);           // three dispatches reached the upscaler, three failed frames
        CHECK(g_hostFeatureCreates == 50 && g_hostFeatureFailures == 47);  // then the host's own upscaler
        CHECK(CountLog("the game's inputs are refused for good") == 1 && CountLog("WG_ERROR_INPUT") >= 1);
        CHECK(CountLog("3 contexts in a row") == 1);
        CHECK(fe->OwnContextCount() == 0);
    }
    // bug hunt A: a refused menu frame, then the game's frames: the host's next context upscales them
    {
        auto fe = session(20, &errors, -1, -1, {0});
        CHECK(errors == 1 && g_createCalls == 2 && g_frames.size() == 20);
        CHECK(CountLog("the game's inputs are refused for good") == 0);
    }
    // ...two refused contexts in a row are still not enough
    {
        auto fe = session(20, &errors, -1, -1, {0, 1});
        CHECK(errors == 2 && g_createCalls == 3 && g_frames.size() == 20 && CountLog("the game's inputs are refused for good") == 0);
    }
    // ...refusals with taken frames between them do not add up (each run is one long)
    {
        auto fe = session(30, &errors, -1, -1, {0, 5, 10, 15, 20});
        CHECK(errors == 5 && g_createCalls == 6 && g_frames.size() == 30 && CountLog("the game's inputs are refused for good") == 0);
    }
    // bug hunt F1 (b): a hard failure (device) of every new context: remembered after 3 in a row, as input refusals
    {
        auto fe = session(50, &errors, -1, -1, {}, 0);
        CHECK(g_createCalls == 3 && g_frames.size() == 3 && errors == 3);
        CHECK(g_hostFeatureCreates == 50 && g_hostFeatureFailures == 47);
        CHECK(CountLog("WrongGPU's upscaler fails the same way") == 1 && CountLog("the game's inputs are refused for good") == 0);
    }
    // a failure that is no input refusal: the host gets a new context of WrongGPU's, as before
    {
        auto fe = session(3, &errors, -1, 0);
        CHECK(errors == 1 && g_createCalls == 2 && CountLog("the game's inputs are refused for good") == 0);
    }
    ResetHost();
}

// The old shape of the churn where the AMD fallback still goes through the host's DLL (hosted, not the custom slot):
// the host's re-entry is refused, so it stops after one level instead of a new context per level, and the host's
// release reaches the destroy path.
static void TestHostReentryStopped() {
    Reset();
    ResetHost();
    g_originalAvailable = false;
    g_compareApi = &g_hostApi;
    g_failCreate = true;
    Config config;
    config.hosted = true;  // another host mod: AMD fallback through the comparison DLL, which leads back here
    auto fe = MakeFrontEnd(config);
    g_hostFe = fe.get();
    ffxContext feature = nullptr;
    HostUpscale(&feature);
    CHECK(feature != nullptr && fe->OwnContextCount() == 1);
    CHECK(g_hostCreates == 1 && g_hostDispatches == 1);
    CHECK(g_hostMaxDepth == 1);                                    // without the guard: kHostCap levels deep
    CHECK(g_hostFeatureCreates == 2 && g_hostFeatureFailures == 1);  // the nested create was refused
    CHECK(g_createCalls == 1);
    CHECK(CountLog("re-entered on this thread from inside another FidelityFX call") == 1);
    for (int frame = 1; frame < 20; ++frame) HostUpscale(&feature);
    CHECK(fe->OwnContextCount() == 1 && g_createCalls == 1 && g_hostCreates == 1 && g_hostDispatches == 20 && g_hostMaxDepth == 1);
    CHECK(CountLog("re-entered on this thread from inside another FidelityFX call") == 4);  // refusals 1, 2, 3 and 10 of 20
    // The host releases its feature: WrongGPU's destroy runs and destroys AMD's (the host's) context.
    CHECK(fe->DestroyContext(&feature, nullptr) == FFX_API_RETURN_OK && !feature);
    CHECK(g_hostDestroys == 1 && fe->OwnContextCount() == 0);
    ResetHost();

    // A dispatch of a context from inside its own dispatch on the same thread (an AMD DLL that calls back) is refused:
    // the context's lock is held by the outer call.
    Reset();
    g_originalApi = &g_againApi;
    g_failCreate = true;
    auto fe2 = MakeFrontEnd(Config{});
    g_hostFe = fe2.get();
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    ffxContext ctx = nullptr;
    CHECK(fe2->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
    g_againContext = &ctx;
    g_againResult = FFX_API_RETURN_OK;
    auto d = MakeDispatch();
    CHECK(fe2->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(g_againResult == FFX_API_RETURN_ERROR_RUNTIME_ERROR && CountLog("re-entered on this thread from inside a call of the same context") == 1);
    g_againContext = nullptr;
    CHECK(fe2->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && fe2->OwnContextCount() == 0);
    ResetHost();
}

// Custom slot with an AMD DLL of WrongGPU's own next to it: a failed create gives an AMD context per create request
// (one real attempt for the session), and every release reaches the destroy path.
static void TestCustomSlotDestroyPath() {
    Reset();
    g_failCreate = true;
    Config config;
    config.hosted = true;
    config.customSlot = true;
    auto fe = MakeFrontEnd(config);
    int amdCreates = 0, amdDestroys = 0;
    for (int cycle = 0; cycle < 5; ++cycle) {
        UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
        own.version.versionId = kWrongGpuProviderVersionId;
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
        CHECK(fe->OwnContextCount() == 1);
        auto d = MakeDispatch();
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_amd.back().fn == "dispatch");
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && !ctx);
        CHECK(fe->OwnContextCount() == 0 && g_amd.back().fn == "destroy");
    }
    for (const AmdCall& c : g_amd) {
        amdCreates += c.fn == "create";
        amdDestroys += c.fn == "destroy";
    }
    CHECK(amdCreates == 5 && amdDestroys == 5);
    CHECK(g_createCalls == 1);
    CHECK(CountLog("destroyed after 1 dispatches") == 5);  // every release is logged and reaches AMD's destroy
    // WrongGPU's upscaler works: destroyed with its context (no parking).
    Reset();
    config.contextParkMs = 0;
    auto fe2 = MakeFrontEnd(config);
    UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, true);
    own.version.versionId = kWrongGpuProviderVersionId;
    ffxContext ctx = nullptr;
    CHECK(fe2->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
    auto d = MakeDispatch();
    CHECK(fe2->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1);
    CHECK(fe2->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && !ctx);
    CHECK(g_destroyed == 1 && fe2->OwnContextCount() == 0 && CountLog("destroyed after 1 dispatches") == 1);
}

// The D3D12 hooks (Platform::activateServices) only after a successful create: never on a failed create or for AMD's
// FSR, once after the create that worked, after a slow create only when it has finished well; when they cannot be
// installed, the upscaler is created again without the split service (nothing left patched).
static void TestHooksAfterCreate() {
    g_resources.clear();
    g_resources[kColor] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    g_resources[kDepth] = {dxgi::R32_FLOAT, 2560, 1440};
    g_resources[kMotion] = {dxgi::R16G16_FLOAT, 2560, 1440};
    g_resources[kOutput] = {dxgi::R16G16B16A16_FLOAT, 2560, 1440};
    UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
    auto d = MakeDispatch();
    {  // failed create: AMD's FSR, no hooks
        Reset();
        g_activateOn = true;
        g_failCreate = true;
        auto fe = MakeFrontEnd(Config{});
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_amd.back().fn == "dispatch");
        CHECK(g_createCalls == 1 && g_activateCalls == 0 && g_servicesAsked.size() == 1);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
        CHECK(g_activateCalls == 0);
    }
    {  // AMD's FSR chosen: no create, no hooks
        Reset();
        g_activateOn = true;
        Config amd;
        amd.upscaler = UpscalerChoice::Amd;
        auto fe = MakeFrontEnd(amd);
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        CHECK(g_createCalls == 0 && g_activateCalls == 0);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    {  // a create that works: hooks once, after it
        Reset();
        g_activateOn = true;
        auto fe = MakeFrontEnd(Config{});
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
        CHECK(g_activateCalls == 1 && g_createCallsAtActivate == 1 && g_created == 1);
        CHECK(g_createInfo.services == &g_services);
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1);
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
    }
    {  // hooks not installable: created again without the split service
        Reset();
        g_activateOn = true;
        g_activateResult = false;
        auto fe = MakeFrontEnd(Config{});
        ffxContext ctx = nullptr;
        CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
        CHECK(g_activateCalls == 1 && g_createCalls == 2 && g_destroyed == 1 && g_createInfo.services == nullptr);
        CHECK(CountLog("created again without the command-list split service") == 1);
        CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK && g_frames.size() == 1);  // WrongGPU's upscaler
        CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK && g_destroyed == 2);
    }
    {  // slow create: hooks only when it has finished, and only if it worked
        for (bool fail : {false, true}) {
            Reset();
            g_activateOn = true;
            Config slow;
            slow.createWaitMs = 20;
            auto fe = MakeFrontEnd(slow);
            ffxContext ctx = nullptr;
            g_createGate = false;
            g_failCreate = fail;
            CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
            CHECK(g_activateCalls == 0);
            g_createGate = true;
            for (int i = 0; i < 400 && CountLog("started after") == 0 && CountLog("create failed after") == 0; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                CHECK(fe->Dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
            }
            CHECK(g_activateCalls == (fail ? 0 : 1));
            CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
        }
    }
}

// Where the model is looked for: wronggpu.ini model_dir, then the game's data folder, then the runtime's default
// (not probed, nothing passed); runtime_options' model_dir decides over all of it.
static void TestModelLookup() {
    const std::vector<ModelPlace> places = {{"K:\\ini\\m\\", "wronggpu.ini model_dir", false},
                                            {"G:\\game\\WrongGPU\\model", "the game's data folder", false},
                                            {"L:\\local\\WrongGPU\\model", "%LOCALAPPDATA%", true}};
    std::vector<std::string> files;
    int probes = 0;
    auto exists = [&](const std::string& f) {
        ++probes;
        return std::find(files.begin(), files.end(), f) != files.end();
    };
    files = {"G:\\game\\WrongGPU\\model\\model.wgcm"};
    ModelChoice m = ChooseModelDir(places, false, exists);
    CHECK(m.index == 1 && m.option == ";model_dir=G:\\game\\WrongGPU\\model" && m.file == "G:\\game\\WrongGPU\\model\\model.wgcm");
    CHECK(m.searched.find("K:\\ini\\m\\model.wgcm (wronggpu.ini model_dir: not there)") != std::string::npos);
    files.push_back("K:\\ini\\m\\model.wgcm");
    m = ChooseModelDir(places, false, exists);
    CHECK(m.index == 0 && m.option == ";model_dir=K:\\ini\\m");  // the ini's folder first; its trailing '\' dropped
    files.clear();
    probes = 0;
    m = ChooseModelDir(places, false, exists);
    CHECK(m.index == 2 && m.option.empty() && m.file == "L:\\local\\WrongGPU\\model\\model.wgcm" && probes == 2);  // default: not probed
    files = {"G:\\game\\WrongGPU\\model\\K\\model.wgcm"};
    CHECK(ChooseModelDir(places, true, exists).index == 1 && ChooseModelDir(places, false, exists).index == 2);
    CHECK(ChooseModelDir({{"C:\\a;b", "x", false}}, false, [](const std::string&) { return true; }).index == -1);
    CHECK(ChooseModelDir({{"C:\\none", "x", false}}, false, [](const std::string&) { return false; }).index == -1);

    // Through the front end.
    auto created = [&](Config config) {
        auto fe = MakeFrontEnd(config);
        UpscaleCreate c(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE);
        for (int i = 0; i < 2; ++i) {
            ffxContext ctx = nullptr;
            CHECK(fe->CreateContext(&ctx, &c.upscale.header, kMemCb) == FFX_API_RETURN_OK && ctx);
            CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
        }
        return g_createOptions;
    };
    Reset();
    g_modelOn = true;
    g_modelPlaces = places;
    g_modelFiles = {"G:\\game\\WrongGPU\\model\\model.wgcm"};
    Config none;
    none.contextParkMs = 0;
    std::string options = created(none);
    CHECK(options.size() > 33 && options.substr(options.size() - 33) == ";model_dir=G:\\game\\WrongGPU\\model");
    CHECK(CountLog("model: Model M from G:\\game\\WrongGPU\\model\\model.wgcm (passed to the runtime as model_dir)") == 1);  // once for two creates
    Reset();
    g_modelOn = true;
    g_modelPlaces = places;
    g_modelFiles = {"G:\\game\\WrongGPU\\model\\model.wgcm"};
    Config explicitDir = none;
    explicitDir.runtimeOptions = "model_dir=X:\\mine";
    options = created(explicitDir);
    CHECK(options.find("model_dir=X:\\mine") != std::string::npos && options.find("G:\\game") == std::string::npos);
    CHECK(CountLog("model_dir from runtime_options decides") == 1 && g_modelProbes.empty());
    Reset();
    g_modelOn = true;
    g_modelPlaces = places;
    options = created(none);
    CHECK(options.find("model_dir") == std::string::npos && CountLog("is not in the game's folders") == 1);

    // wronggpu.ini: model_dir is the front end's own key now (no warning, not a loose runtime option).
    std::string warnings;
    Config parsed = ParseConfig("[fsr]\nmodel_dir = K:\\Wrong GPU\\model\n", &warnings);
    CHECK(parsed.modelDir == "K:\\Wrong GPU\\model" && warnings.empty() && parsed.runtimeOptions.empty());
    parsed = ParseConfig("[fsr]\nmodel_dir = a;b\n", &warnings);
    CHECK(parsed.modelDir.empty() && warnings.find("bad value") != std::string::npos);
    CHECK(std::string(DefaultFsrSectionText()).find("model_dir =") != std::string::npos);
}

int main() {
    TestCustomSlotFailedCreate();
    TestCustomSlotInputRefused();
    TestOutputResize();
    TestBrokenNotCalledAgain();
    TestHostReentryStopped();
    TestCustomSlotDestroyPath();
    TestHooksAfterCreate();
    TestModelLookup();
    TestRegistryEntries();
    TestUnsupportedFrames();
    TestSplitBlocked();
    TestRuntimeProviderName();
    TestWithoutModelKItems();
#if !WRONGGPU_MODEL_K
    TestWithoutModelK();
#endif
    TestAsiPluginHooked();
    TestAsiPluginNotHooked();
    TestAsiPluginPresetK();
    {
        Config profile;
        const auto original = profile.BackendOptions();
        CHECK(ApplyProcessCompatibility(profile, "C:\\Games\\Crimson Desert\\CrimsonDesert.exe"));
        CHECK(profile.runtimeOptions == "ready_wait=fence" && profile.BackendOptions() == original + ";ready_wait=fence");
        CHECK(!ApplyProcessCompatibility(profile, "CrimsonDesert.exe"));
        Config other;
        CHECK(!ApplyProcessCompatibility(other, "OtherGame.exe") && other.runtimeOptions.empty());
        CHECK(!ApplyProcessCompatibility(other, "CrimsonDesertLauncher.exe"));
        other.runtimeOptions = "timing_log=1;sync=inline";
        CHECK(ApplyProcessCompatibility(other, "CRIMSONDESERT.EXE") && other.runtimeOptions == "timing_log=1;sync=inline;ready_wait=fence");
        other.runtimeOptions = "timing_log=1;READY_WAIT = memory;watchdog=1";
        CHECK(!ApplyProcessCompatibility(other, "CrimsonDesert.exe") && other.runtimeOptions == "timing_log=1;READY_WAIT = memory;watchdog=1");
        other.upscaler = UpscalerChoice::Amd; other.runtimeOptions.clear();
        CHECK(!ApplyProcessCompatibility(other, "CrimsonDesert.exe") && other.runtimeOptions.empty());
        other.upscaler = UpscalerChoice::Network; other.runtimeDll = "preset_k.dll";
        CHECK(!ApplyProcessCompatibility(other, "CrimsonDesert.exe"));
    }
    {
        Config cyberpunk;
        CHECK(ApplyProcessCompatibility(cyberpunk, "N:\\Epic Games\\Cyberpunk2077\\bin\\x64\\Cyberpunk2077.exe"));
        CHECK(cyberpunk.runtimeOptions == "ready_wait=fence");
        Config control;
        control.runtimeOptions = "timing_log=1";
        CHECK(ApplyProcessCompatibility(control, "C:\\Games\\CONTROL Resonant\\CONTROLResonant.exe"));
        CHECK(control.runtimeOptions == "timing_log=1;ready_wait=fence");
        CHECK(!ApplyProcessCompatibility(control, "CONTROLResonant.exe"));
        Config explicitMode;
        explicitMode.runtimeOptions = "ready_wait=memory;timing_log=1";
        CHECK(!ApplyProcessCompatibility(explicitMode, "Cyberpunk2077.exe"));
        CHECK(explicitMode.runtimeOptions == "ready_wait=memory;timing_log=1");
        CHECK(!ApplyProcessCompatibility(explicitMode, "ControlResonantLauncher.exe"));
    }
    TestFailedFrameProviderTruth();
    TestNoProviderDelayedCreateCleanup();
    TestHostedSeparateAmdFallback();
    TestHostedAmdOverride();
    TestHostedFallbackLifetime();
    TestHostedLookupReentry();
    TestOwnUpscalerPath();
    TestForwarding();
    TestCreateFallback();
    TestFrameFallback();
    TestBadFrames();
    TestParkedContexts();
    TestHostedSwitchNotParked();
    TestCompare();
    TestReactiveMaskForwarded();
    TestSlowCreate();
    TestInterfaceVersions();
    TestAmdMode();
    TestNonLinearColour();
    TestMappingDetails();
    TestConfig();
    TestLogThinning();
    TestHosted();
    TestHostVersionHandling();
    if (const char* dump = std::getenv("WG_TEST_DUMP_LOG")) {
        (void)dump;
        for (const auto& line : g_log) std::printf("%s\n", line.c_str());
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
