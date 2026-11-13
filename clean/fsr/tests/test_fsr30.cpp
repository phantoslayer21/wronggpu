// SPDX-License-Identifier: MIT
// Unit tests of the FSR 3.0 component forwarders (fsr30_hook.h) without Windows or D3D12: forwarding and logging of
// every export, the FSR 3.0 -> network parameter mapping, where the stand-in takes over (upscaler level, ffx_fsr3
// level, AMD first for frame generation), failure handling and the SDK 1.1 guard. Two Hook objects play the two DLLs.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../src/fsr30_hook.h"
#include "../src/fsr30_map.h"

using namespace wgfsr;
using namespace wgfsr::fsr30;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                             \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                       \
    } while (0)
#define CHECK_NEAR(a, b) CHECK(std::fabs(double(a) - double(b)) < 1e-6)

// ---------------------------------------------------------------------------------------------------------------
// Fake AMD DLLs: a list of calls in order.

static std::vector<std::string> g_amd;
static ErrorCode g_amdDispatchResult = OK;
static Hook* g_upscalerHook = nullptr;  // "WrongGPU's ffx_fsr3upscaler_x64.dll", for the fake ffx_fsr3 to call into
static bool g_fsr3CallsUpscaler = false;

static ErrorCode UpCreate(void*, const UpscalerContextDescription*) { g_amd.push_back("up.create"); return OK; }
static ErrorCode UpShared(void*, void*) { g_amd.push_back("up.shared"); return OK; }
static ErrorCode UpDispatch(void*, const UpscalerDispatchDescription* d) {
    g_amd.push_back(std::string("up.dispatch") + (d->dilatedDepth.resource ? "+shared" : ""));
    return g_amdDispatchResult;
}
static ErrorCode UpReactive(void*, const UpscalerGenerateReactiveDescription*) { g_amd.push_back("up.reactive"); return OK; }
static ErrorCode UpDestroy(void*) { g_amd.push_back("up.destroy"); return OK; }
static float UpRatio(int32_t) { g_amd.push_back("up.ratio"); return 1.5f; }
static ErrorCode UpRenderRes(uint32_t* w, uint32_t* h, uint32_t dw, uint32_t dh, int32_t) {
    g_amd.push_back("up.renderres");
    *w = dw * 2 / 3;
    *h = dh * 2 / 3;
    return OK;
}
static int32_t UpPhases(int32_t, int32_t) { g_amd.push_back("up.phases"); return 17; }
static ErrorCode UpJitter(float* x, float* y, int32_t, int32_t) {
    g_amd.push_back("up.jitter");
    *x = 0.125f;
    *y = -0.25f;
    return OK;
}
static bool UpIsNull(const void* copy) {
    g_amd.push_back("up.isnull");
    return static_cast<const Resource*>(copy)->resource == nullptr;
}

static char g_shared[3];
static ErrorCode F3Create(void* ctx, Fsr3ContextDescription* d) {
    g_amd.push_back("f3.create");
    if (g_fsr3CallsUpscaler) {
        UpscalerContextDescription u{};
        u.flags = d->flags & kSharedUpscaleFlags;
        u.maxRenderSize = d->maxRenderSize;
        u.displaySize = d->upscaleOutputSize;
        return g_upscalerHook->UpContextCreate(static_cast<char*>(ctx) + 64, &u, "ffx_fsr3_x64.wronggpu-original.dll");
    }
    return OK;
}
static ErrorCode F3DispatchUpscale(void* ctx, const Fsr3DispatchUpscaleDescription* d) {
    g_amd.push_back("f3.dispatchUpscale");
    if (g_fsr3CallsUpscaler) {
        UpscalerDispatchDescription u{};
        u.commandList = d->commandList;
        u.color = d->color;
        u.depth = d->depth;
        u.motionVectors = d->motionVectors;
        u.output = d->upscaleOutput;
        u.dilatedDepth.resource = &g_shared[0];
        u.dilatedMotionVectors.resource = &g_shared[1];
        u.reconstructedPrevNearestDepth.resource = &g_shared[2];
        u.jitterOffset = d->jitterOffset;
        u.motionVectorScale = d->motionVectorScale;
        u.renderSize = d->renderSize;
        u.preExposure = d->preExposure;
        return g_upscalerHook->UpContextDispatch(static_cast<char*>(ctx) + 64, &u, "ffx_fsr3_x64.wronggpu-original.dll");
    }
    return OK;
}
static ErrorCode F3Destroy(void* ctx) {
    g_amd.push_back("f3.destroy");
    if (g_fsr3CallsUpscaler) return g_upscalerHook->UpContextDestroy(static_cast<char*>(ctx) + 64, "ffx_fsr3_x64.wronggpu-original.dll");
    return OK;
}
static ErrorCode F3FrameGen(const FrameGenerationDispatchDescription*) { g_amd.push_back("f3.framegen"); return OK; }
static ErrorCode F3Configure(void*, const FrameGenerationConfig*) { g_amd.push_back("f3.configure"); return OK; }
static ErrorCode F3Skip(void*) { g_amd.push_back("f3.skip"); return OK; }
static int32_t F3Phases(int32_t, int32_t) { g_amd.push_back("f3.phases"); return 18; }
static uint32_t EffectVersion() { return 0x00C01001u; }

static UpscalerOriginal MakeUpscalerOriginal(bool sdk11 = false) {
    UpscalerOriginal o;
    o.contextCreate = UpCreate;
    o.getSharedResourceDescriptions = UpShared;
    o.contextDispatch = UpDispatch;
    o.contextGenerateReactiveMask = UpReactive;
    o.contextDestroy = UpDestroy;
    o.getUpscaleRatioFromQualityMode = UpRatio;
    o.getRenderResolutionFromQualityMode = UpRenderRes;
    o.getJitterPhaseCount = UpPhases;
    o.getJitterOffset = UpJitter;
    o.resourceIsNull = UpIsNull;
    if (sdk11) o.getEffectVersion = EffectVersion;
    o.sdk11 = sdk11;
    return o;
}

static Fsr3Original MakeFsr3Original() {
    Fsr3Original o;
    o.contextCreate = F3Create;
    o.contextDispatchUpscale = F3DispatchUpscale;
    o.contextDestroy = F3Destroy;
    o.dispatchFrameGeneration = F3FrameGen;
    o.configureFrameGeneration = F3Configure;
    o.skipPresent = F3Skip;
    o.getJitterPhaseCount = F3Phases;
    return o;
}

// ---------------------------------------------------------------------------------------------------------------
// Fake upscaler (stand-in).

static std::vector<WgFrame> g_frames;
static std::vector<WgCreateInfo> g_creates;
static int g_destroyed = 0;
static int g_failDispatchAt = -1;
static int g_notReadyAt = -1;  // frame answered with WG_ERROR_NOT_READY
static WgResult FakeCreate(const WgCreateInfo* info, WgUpscaler** out) {
    g_creates.push_back(*info);
    info->log(info->log_user, WG_LOG_INFO, "fake stand-in created");
    *out = reinterpret_cast<WgUpscaler*>(new int(1));
    return WG_OK;
}
static WgResult FakeDispatch(WgUpscaler*, const WgFrame* f) {
    g_frames.push_back(*f);
    g_amd.push_back("standin");
    if (int(g_frames.size()) - 1 == g_notReadyAt) return WG_ERROR_NOT_READY;
    return int(g_frames.size()) - 1 == g_failDispatchAt ? WG_ERROR_DEVICE : WG_OK;
}
static void FakeDestroy(WgUpscaler* u) {
    ++g_destroyed;
    delete reinterpret_cast<int*>(u);
}
// Named like the real stand-in, so the log says "WrongGPU stand-in" as in game; g_network is the same fake under
// another name (the network runtime).
static const WgUpscalerBackend g_backend{sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "WrongGPU stand-in upscaler", FakeCreate,
                                         FakeDispatch, FakeDestroy, nullptr, 0};
static const WgUpscalerBackend g_network{sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "WrongGPU network (fake)", FakeCreate,
                                         FakeDispatch, FakeDestroy, nullptr, WG_BACKEND_NEEDS_SPLIT};
static const WgUpscalerBackend* g_useBackend = &g_backend;
// The one-frame fallback for WG_ERROR_NOT_READY.
static int g_fallbackFrames = 0, g_fallbackDestroyed = 0;
static WgResult FallbackCreate(const WgCreateInfo*, WgUpscaler** out) {
    *out = reinterpret_cast<WgUpscaler*>(new int(2));
    return WG_OK;
}
static WgResult FallbackDispatch(WgUpscaler*, const WgFrame*) {
    ++g_fallbackFrames;
    g_amd.push_back("fallback");
    return WG_OK;
}
static void FallbackDestroy(WgUpscaler* u) {
    ++g_fallbackDestroyed;
    delete reinterpret_cast<int*>(u);
}
static const WgUpscalerBackend g_fallback{sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "fake fallback", FallbackCreate,
                                          FallbackDispatch, FallbackDestroy, nullptr, 0};
static int g_notes = 0;

static std::vector<std::string> g_log;
static void* const kDevice = reinterpret_cast<void*>(uintptr_t(0xD3D0));
static void* const kList = reinterpret_cast<void*>(uintptr_t(0xC3D1));
static char g_color, g_depth, g_motion, g_output;

static UpscalerOriginal g_upOriginal;
static Fsr3Original g_f3Original;

static Platform MakePlatform(bool upscalerSide) {
    Platform p;
    if (upscalerSide) p.upscaler = [] { return &g_upOriginal; };
    else p.fsr3 = [] { return &g_f3Original; };
    p.backend = [](const Config&, std::string&) -> const WgUpscalerBackend* { return g_useBackend; };
    p.log = [](const std::string& line) { g_log.push_back(line); };
    p.deviceFromCommandList = [](void* list) -> void* { return list == kList ? kDevice : nullptr; };
    p.upscalerSeenOnThread = [] { return Hook::SeenOnThisThread(); };
    p.hintUpscalingOnly = [](int hint) { Hook::HintUpscalingOnly(hint); };
    p.standIn = [](const Config&) -> const WgUpscalerBackend* { return &g_fallback; };
    p.noteDispatch = [](void* list, WgFrame*) { g_notes += list == kList; };
    return p;
}

static void Reset() {
    g_amd.clear();
    g_frames.clear();
    g_creates.clear();
    g_log.clear();
    g_destroyed = 0;
    g_failDispatchAt = -1;
    g_notReadyAt = -1;
    g_useBackend = &g_backend;
    g_fallbackFrames = g_fallbackDestroyed = 0;
    g_notes = 0;
    g_amdDispatchResult = OK;
    g_fsr3CallsUpscaler = false;
    g_upOriginal = MakeUpscalerOriginal();
    g_f3Original = MakeFsr3Original();
}

static Resource Res(void* p, uint32_t format, uint32_t w, uint32_t h, uint32_t state) {
    Resource r{};
    r.resource = p;
    r.description.type = 2;  // TEXTURE2D
    r.description.format = format;
    r.description.width = w;
    r.description.height = h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.state = state;
    const char16_t name[] = u"Color";
    std::memcpy(r.name, name, sizeof name);
    return r;
}

static UpscalerContextDescription UpCreateDesc() {
    UpscalerContextDescription d{};
    d.flags = FLAG_HIGH_DYNAMIC_RANGE | FLAG_DEPTH_INVERTED | FLAG_TEXTURE1D_USAGE;
    d.maxRenderSize = {1707, 960};
    d.displaySize = {2560, 1440};
    d.backendInterface.device = reinterpret_cast<void*>(uintptr_t(0xFFD0));  // an FfxDevice, not used by WrongGPU
    return d;
}

static UpscalerDispatchDescription UpDispatchDesc(bool shared) {
    UpscalerDispatchDescription d{};
    d.commandList = kList;
    d.color = Res(&g_color, SF_R16G16B16A16_FLOAT, 2560, 1440, 4 /*COMPUTE_READ*/);
    d.depth = Res(&g_depth, SF_R32_FLOAT, 2560, 1440, 4);
    d.motionVectors = Res(&g_motion, SF_R16G16_FLOAT, 2560, 1440, 4);
    d.output = Res(&g_output, SF_R11G11B10_FLOAT, 2560, 1440, 2 /*UNORDERED_ACCESS*/);
    if (shared) {
        d.dilatedDepth.resource = &g_shared[0];
        d.dilatedMotionVectors.resource = &g_shared[1];
        d.reconstructedPrevNearestDepth.resource = &g_shared[2];
    }
    d.jitterOffset = {0.25f, -0.375f};
    d.motionVectorScale = {1707.0f, 960.0f};
    d.renderSize = {1707, 960};
    d.preExposure = 1.0f;
    d.frameTimeDelta = 16.0f;
    return d;
}

static size_t CountLog(const char* needle) {
    size_t n = 0;
    for (const auto& line : g_log) n += line.find(needle) != std::string::npos;
    return n;
}

static std::string Calls() {
    std::string s;
    for (const auto& c : g_amd) s += c + " ";
    return s;
}

static Config StandIn() {
    Config c;
    c.upscaler = UpscalerChoice::StandIn;
    return c;
}

// ---------------------------------------------------------------------------------------------------------------

static void TestLogOnly() {
    Reset();
    Config amd;
    amd.upscaler = UpscalerChoice::Amd;
    Hook up(MakePlatform(true), amd, "ffx_fsr3upscaler");
    std::vector<uint32_t> ctx(64);
    UpscalerContextDescription cd = UpCreateDesc();
    CHECK(up.UpContextCreate(ctx.data(), &cd, "Cyberpunk2077.exe") == OK);
    UpscalerDispatchDescription d = UpDispatchDesc(false);
    CHECK(up.UpContextDispatch(ctx.data(), &d, "Cyberpunk2077.exe") == OK);
    CHECK(up.UpContextDestroy(ctx.data(), "Cyberpunk2077.exe") == OK);
    CHECK(Calls() == "up.create up.dispatch up.destroy ");
    CHECK(g_creates.empty() && g_frames.empty());
    CHECK(CountLog("[ffx_fsr3upscaler] #1 ffxFsr3UpscalerContextCreate from Cyberpunk2077.exe") == 1);
    CHECK(CountLog("flags=HDR|DEPTH_INVERTED|TEXTURE1D_USAGE maxRenderSize=1707x960 displaySize=2560x1440") == 1);
    CHECK(CountLog("jitterOffset=(0.25,-0.375) motionVectorScale=(1707,960) renderSize=1707x960") == 1);
    CHECK(CountLog("R16G16B16A16_FLOAT 2560x1440x1") == 1);
    CHECK(CountLog("\"Color\"") >= 1);
    CHECK(CountLog("-> FFX_OK (0x00000000) (AMD)") == 1);

    // Helpers are forwarded and logged with their results.
    float x = 0, y = 0;
    uint32_t w = 0, h = 0;
    CHECK(up.UpGetJitterPhaseCount(1707, 2560, "game") == 17);
    CHECK(up.UpGetJitterOffset(&x, &y, 3, 17, "game") == OK && x == 0.125f && y == -0.25f);
    CHECK(up.UpGetRenderResolutionFromQualityMode(&w, &h, 2560, 1440, 1, "game") == OK && w == 1706);
    CHECK_NEAR(up.UpGetUpscaleRatioFromQualityMode(1, "game"), 1.5f);
    Resource r = Res(&g_color, SF_R32_FLOAT, 1, 1, 1);
    CHECK(!up.UpResourceIsNull(&r, "game"));
    CHECK(CountLog("ffxFsr3UpscalerGetJitterPhaseCount from game renderWidth=1707 displayWidth=2560 -> 17") == 1);
    CHECK(CountLog("x=0.125000 y=-0.250000") == 1);
    // SDK 1.1-only exports are missing in a 3.0.x DLL: logged, error returned.
    CHECK(up.UpGetEffectVersion("game") == 0);
    CHECK(CountLog("ffxFsr3UpscalerGetEffectVersion from game -> not available in AMD's DLL") == 1);
}

static void TestUpscalerRoute() {
    Reset();
    Hook up(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    std::vector<uint32_t> ctx(64);
    UpscalerContextDescription cd = UpCreateDesc();
    CHECK(up.UpContextCreate(ctx.data(), &cd, "game") == OK);
    CHECK(up.ContextCount() == 1);

    // No frame-generation resources: only the stand-in runs.
    UpscalerDispatchDescription d = UpDispatchDesc(false);
    CHECK(up.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "up.create standin ");
    CHECK(g_creates.size() == 1 && g_creates[0].d3d12_device == kDevice);
    CHECK(g_creates[0].flags == (WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED));  // TEXTURE1D_USAGE is not a network flag
    CHECK(g_creates[0].output_width == 2560 && g_creates[0].max_render_width == 1707);
    const WgFrame& f = g_frames[0];
    CHECK(f.command_list == kList && f.color.resource == &g_color && f.output.resource == &g_output);
    CHECK(f.color.format == 10 /*R16G16B16A16_FLOAT*/ && f.depth.format == 41 && f.motion.format == 34 && f.output.format == 26);
    CHECK(f.color.state == 0x40 && f.output.state == 0x8);
    CHECK_NEAR(f.jitter[0], 0.25f);  // + jitterOffset (spec 20 draft 2)
    CHECK_NEAR(f.jitter[1], -0.375f);
    CHECK_NEAR(f.motion_scale[0], 1707.0f);
    CHECK(f.render_width == 1707 && f.output_width == 2560 && f.output_height == 1440);
    CHECK(CountLog("WrongGPU frame: frame=0 R=1707x960") == 1);
    CHECK(CountLog("-> FFX_OK (WrongGPU stand-in)") == 1);
    CHECK(CountLog("C1 upscaler info: fake stand-in created") == 1);

    // With frame-generation resources: AMD first, then the stand-in on the same command list.
    g_amd.clear();
    d = UpDispatchDesc(true);
    CHECK(up.UpContextDispatch(ctx.data(), &d, "ffx_fsr3_x64.wronggpu-original.dll") == OK);
    CHECK(Calls() == "up.dispatch+shared standin ");
    CHECK(g_frames.size() == 2 && g_frames[1].frame_index == 1);
    CHECK(CountLog("AMD for the frame-generation resources, then WrongGPU stand-in") == 1);

    CHECK(up.UpContextDestroy(ctx.data(), "game") == OK);
    CHECK(g_destroyed == 1 && up.ContextCount() == 0);
    CHECK(CountLog("(C1, 2 upscale dispatches)") == 1);

    // fsr3_amd_first = always / never.
    for (const char* mode : {"always", "never"}) {
        Reset();
        Config c = StandIn();
        c.fsr3AmdFirst = mode;
        Hook h(MakePlatform(true), c, "ffx_fsr3upscaler");
        CHECK(h.UpContextCreate(ctx.data(), &cd, "game") == OK);
        UpscalerDispatchDescription plain = UpDispatchDesc(false), shared = UpDispatchDesc(true);
        CHECK(h.UpContextDispatch(ctx.data(), &plain, "game") == OK);
        CHECK(h.UpContextDispatch(ctx.data(), &shared, "game") == OK);
        if (std::string(mode) == "always") CHECK(Calls() == "up.create up.dispatch standin up.dispatch+shared standin ");
        else CHECK(Calls() == "up.create standin standin ");
        CHECK(h.UpContextDestroy(ctx.data(), "game") == OK);
    }
}

// WG_ERROR_NOT_READY (interface version 2): the fallback upscales that frame, the context stays on the backend.
static void TestNotReady() {
    Reset();
    g_notReadyAt = 0;
    Hook up(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    std::vector<uint32_t> ctx(64);
    UpscalerContextDescription cd = UpCreateDesc();
    CHECK(up.UpContextCreate(ctx.data(), &cd, "game") == OK);
    UpscalerDispatchDescription d = UpDispatchDesc(false);
    CHECK(up.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(up.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "up.create standin fallback standin ");
    CHECK(g_fallbackFrames == 1 && CountLog("stand-in for this frame") == 1 && CountLog("ERROR C1") == 0);
    CHECK(g_notes == 4);  // each dispatch noted before routing and again with its frame
    CHECK(up.UpContextDestroy(ctx.data(), "game") == OK);
    CHECK(g_destroyed == 1 && g_fallbackDestroyed == 1);
}

static void TestFailures() {
    // The stand-in fails on frame 1: AMD renders that frame and every later one.
    Reset();
    g_failDispatchAt = 1;
    Hook up(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    std::vector<uint32_t> ctx(64);
    UpscalerContextDescription cd = UpCreateDesc();
    CHECK(up.UpContextCreate(ctx.data(), &cd, "game") == OK);
    UpscalerDispatchDescription d = UpDispatchDesc(false);
    for (int i = 0; i < 3; ++i) CHECK(up.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "up.create standin standin up.dispatch up.dispatch ");
    CHECK(CountLog("ERROR C1: upscaler \"WrongGPU stand-in upscaler\" dispatch failed: WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)") == 1);
    CHECK(CountLog("C1 stays on AMD") == 1);
    CHECK(up.UpContextDestroy(ctx.data(), "game") == OK);

    // A frame without a usable command list (no device): AMD renders it.
    Reset();
    Hook up2(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    CHECK(up2.UpContextCreate(ctx.data(), &cd, "game") == OK);
    d = UpDispatchDesc(false);
    d.commandList = reinterpret_cast<void*>(uintptr_t(0xBAD));
    CHECK(up2.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "up.create up.dispatch ");
    CHECK(CountLog("no D3D12 device from the command list") == 1);

    // AMD's own dispatch fails while it runs first: its error is returned, the stand-in does not run.
    Reset();
    g_amdDispatchResult = ERROR_BACKEND_API_ERROR;
    Hook up3(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    CHECK(up3.UpContextCreate(ctx.data(), &cd, "game") == OK);
    d = UpDispatchDesc(true);
    CHECK(up3.UpContextDispatch(ctx.data(), &d, "game") == ERROR_BACKEND_API_ERROR);
    CHECK(Calls() == "up.create up.dispatch+shared ");

    // A dispatch on a context WrongGPU did not see created is only forwarded.
    Reset();
    Hook up4(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    std::vector<uint32_t> other(64);
    d = UpDispatchDesc(false);
    CHECK(up4.UpContextDispatch(other.data(), &d, "game") == OK);
    CHECK(Calls() == "up.dispatch ");

    // The network runtime fails on frame 1: the log names it, says why in words, and AMD takes over.
    Reset();
    g_useBackend = &g_network;
    g_failDispatchAt = 1;
    Hook up5(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    CHECK(up5.UpContextCreate(ctx.data(), &cd, "game") == OK);
    d = UpDispatchDesc(false);
    for (int i = 0; i < 3; ++i) CHECK(up5.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "up.create standin standin up.dispatch up.dispatch ");
    CHECK(CountLog("-> FFX_OK (WrongGPU network (fake))") == 1);
    CHECK(CountLog("dispatch failed: WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)") == 1);
    CHECK(CountLog("(AMD after WrongGPU network (fake) failed; C1 stays on AMD)") == 1);
    CHECK(CountLog("(AMD; WrongGPU network (fake) failed earlier on C1)") == 1);
    CHECK(CountLog("WrongGPU stand-in") == 0);
    CHECK(up5.UpContextDestroy(ctx.data(), "game") == OK);
    CHECK(CountLog("(C1, 2 upscale dispatches)") == 1);
}

static void TestSdk11Guard() {
    Reset();
    g_upOriginal = MakeUpscalerOriginal(true);
    Hook up(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    std::vector<uint32_t> ctx(64);
    UpscalerContextDescription cd = UpCreateDesc();
    CHECK(up.UpContextCreate(ctx.data(), &cd, "game") == OK);
    UpscalerDispatchDescription d = UpDispatchDesc(false);
    CHECK(up.UpContextDispatch(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "up.create up.dispatch ");
    CHECK(g_creates.empty() && up.ContextCount() == 0);
    CHECK(CountLog("(SDK 1.1 layout, not decoded)") == 2);
    CHECK(up.UpGetEffectVersion("game") == 0x00C01001u);
}

static void TestFsr3Level() {
    // ffx_fsr3 calls the upscaler DLL (dynamically linked FSR 3.0): the upscaler level takes over, ffx_fsr3 does not.
    Reset();
    g_fsr3CallsUpscaler = true;
    Hook up(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    Hook f3(MakePlatform(false), StandIn(), "ffx_fsr3");
    g_upscalerHook = &up;
    std::vector<uint32_t> ctx(256);
    Fsr3ContextDescription cd{};
    cd.flags = FLAG_DEPTH_INVERTED | FSR3_FLAG_ASYNC_WORKLOAD_SUPPORT;
    cd.maxRenderSize = {1707, 960};
    cd.upscaleOutputSize = {2560, 1440};
    cd.displaySize = {2560, 1440};
    CHECK(f3.F3ContextCreate(ctx.data(), &cd, "game") == OK);
    CHECK(f3.ContextCount() == 1 && up.ContextCount() == 1);
    Fsr3DispatchUpscaleDescription d{};
    d.commandList = kList;
    d.color = Res(&g_color, SF_R16G16B16A16_FLOAT, 2560, 1440, 4);
    d.depth = Res(&g_depth, SF_R32_FLOAT, 2560, 1440, 4);
    d.motionVectors = Res(&g_motion, SF_R16G16_FLOAT, 2560, 1440, 4);
    d.upscaleOutput = Res(&g_output, SF_R16G16B16A16_FLOAT, 2560, 1440, 2);
    d.jitterOffset = {-0.1f, 0.2f};
    d.motionVectorScale = {1707.0f, 960.0f};
    d.renderSize = {1707, 960};
    d.preExposure = 1.0f;
    CHECK(f3.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create up.create f3.dispatchUpscale up.dispatch+shared standin ");
    CHECK(g_frames.size() == 1);
    CHECK(CountLog("handled in ffx_fsr3upscaler") == 1);
    CHECK(CountLog("ffxFsr3UpscalerContextDispatch from ffx_fsr3_x64.wronggpu-original.dll") == 1);
    FrameGenerationConfig fg{};
    fg.frameGenerationEnabled = true;
    CHECK(f3.F3ConfigureFrameGeneration(ctx.data(), &fg, "game") == OK);
    CHECK(CountLog("frameGenerationEnabled=1") == 1);
    FrameGenerationDispatchDescription fgd{};
    fgd.numInterpolatedFrames = 1;
    CHECK(f3.F3DispatchFrameGeneration(&fgd, "ffx_backend_dx12_x64.dll") == OK);
    CHECK(CountLog("numInterpolatedFrames=1") == 1);
    CHECK(f3.F3ContextDestroy(ctx.data(), "game") == OK);
    CHECK(f3.ContextCount() == 0 && up.ContextCount() == 0 && g_destroyed == 1);

    // ffx_fsr3 has the upscaler built in: after AMD's upscale, the ffx_fsr3 level runs the stand-in.
    Reset();
    g_fsr3CallsUpscaler = false;
    Hook f3b(MakePlatform(false), StandIn(), "ffx_fsr3");
    CHECK(f3b.F3ContextCreate(ctx.data(), &cd, "game") == OK);
    CHECK(f3b.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create f3.dispatchUpscale standin ");
    CHECK(g_creates.size() == 1 && g_creates[0].output_width == 2560 && g_creates[0].flags == WG_CREATE_DEPTH_INVERTED);
    CHECK_NEAR(g_frames[0].jitter[0], -0.1f);
    CHECK(g_frames[0].output.resource == &g_output);
    CHECK(CountLog("AMD, then WrongGPU stand-in over upscaleOutput") == 1);
    CHECK(f3b.F3ContextDestroy(ctx.data(), "game") == OK);

    // fsr3_route = fsr3 with a dynamically linked ffx_fsr3: only the ffx_fsr3 level runs the stand-in.
    Reset();
    g_fsr3CallsUpscaler = true;
    Config route = StandIn();
    route.fsr3Route = "fsr3";
    Hook up2(MakePlatform(true), route, "ffx_fsr3upscaler");
    Hook f3c(MakePlatform(false), route, "ffx_fsr3");
    g_upscalerHook = &up2;
    CHECK(f3c.F3ContextCreate(ctx.data(), &cd, "game") == OK);
    CHECK(f3c.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create up.create f3.dispatchUpscale up.dispatch+shared standin ");
    CHECK(g_frames.size() == 1 && g_creates.size() == 1);
    CHECK(f3c.F3ContextDestroy(ctx.data(), "game") == OK);

    // fsr3_route = off: everything goes to AMD.
    Reset();
    g_fsr3CallsUpscaler = true;
    Config off = StandIn();
    off.fsr3Route = "off";
    Hook up3(MakePlatform(true), off, "ffx_fsr3upscaler");
    Hook f3d(MakePlatform(false), off, "ffx_fsr3");
    g_upscalerHook = &up3;
    CHECK(f3d.F3ContextCreate(ctx.data(), &cd, "game") == OK);
    CHECK(f3d.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create up.create f3.dispatchUpscale up.dispatch+shared ");
    CHECK(f3d.F3ContextDestroy(ctx.data(), "game") == OK);
    g_upscalerHook = nullptr;
}

// FSR 3 contexts created with UPSCALING_ONLY: nothing reads the frame-generation resources, so AMD's upscaler does
// not run at all (ANSWERS.md, spec 20, FSR 3.0 item 12).
static void TestUpscalingOnly() {
    Fsr3ContextDescription cd{};
    cd.flags = FLAG_DEPTH_INVERTED | FLAG_AUTO_EXPOSURE | FSR3_FLAG_UPSCALING_ONLY;
    cd.maxRenderSize = {1707, 960};
    cd.upscaleOutputSize = {2560, 1440};
    cd.displaySize = {2560, 1440};
    Fsr3DispatchUpscaleDescription d{};
    d.commandList = kList;
    d.color = Res(&g_color, SF_R16G16B16A16_FLOAT, 2560, 1440, 4);
    d.depth = Res(&g_depth, SF_R32_FLOAT, 2560, 1440, 4);
    d.motionVectors = Res(&g_motion, SF_R16G16B16A16_FLOAT, 2560, 1440, 4);
    d.upscaleOutput = Res(&g_output, SF_R16G16B16A16_FLOAT, 2560, 1440, 2);
    d.motionVectorScale = {1707.0f, 960.0f};
    d.renderSize = {1707, 960};
    d.preExposure = 1.0f;
    std::vector<uint32_t> ctx(256);

    // Cyberpunk 2077's path: ffx_fsr3 calls the upscaler DLL, which gets the three resources anyway; only the stand-in.
    Reset();
    g_fsr3CallsUpscaler = true;
    Hook up(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    Hook f3(MakePlatform(false), StandIn(), "ffx_fsr3");
    g_upscalerHook = &up;
    CHECK(f3.F3ContextCreate(ctx.data(), &cd, "Cyberpunk2077.exe") == OK);
    CHECK(f3.F3ContextDispatchUpscale(ctx.data(), &d, "Cyberpunk2077.exe") == OK);
    CHECK(f3.F3ContextDispatchUpscale(ctx.data(), &d, "Cyberpunk2077.exe") == OK);
    CHECK(Calls() == "f3.create up.create f3.dispatchUpscale standin f3.dispatchUpscale standin ");
    CHECK(CountLog("(created by an ffx_fsr3 context with UPSCALING_ONLY: AMD's upscaler is not needed)") == 1);
    CHECK(CountLog("AMD's upscaler skipped: no frame generation") == 2);
    CHECK(CountLog("AMD's ffx_fsr3 calls the upscaler DLL") == 1);
    CHECK(f3.F3ContextDestroy(ctx.data(), "Cyberpunk2077.exe") == OK);
    // The hint is consumed: an upscaler context the game creates itself afterwards is not UPSCALING_ONLY.
    std::vector<uint32_t> direct(64);
    UpscalerContextDescription ud = UpCreateDesc();
    g_amd.clear();
    CHECK(up.UpContextCreate(direct.data(), &ud, "game") == OK);
    UpscalerDispatchDescription shared = UpDispatchDesc(true);
    CHECK(up.UpContextDispatch(direct.data(), &shared, "game") == OK);
    CHECK(Calls() == "up.create up.dispatch+shared standin ");
    CHECK(up.UpContextDestroy(direct.data(), "game") == OK);

    // Frame generation on (no UPSCALING_ONLY): AMD's upscaler still runs first.
    Reset();
    g_fsr3CallsUpscaler = true;
    Hook up2(MakePlatform(true), StandIn(), "ffx_fsr3upscaler");
    Hook f3b(MakePlatform(false), StandIn(), "ffx_fsr3");
    g_upscalerHook = &up2;
    Fsr3ContextDescription fg = cd;
    fg.flags &= ~FSR3_FLAG_UPSCALING_ONLY;
    CHECK(f3b.F3ContextCreate(ctx.data(), &fg, "game") == OK);
    CHECK(f3b.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create up.create f3.dispatchUpscale up.dispatch+shared standin ");
    CHECK(f3b.F3ContextDestroy(ctx.data(), "game") == OK);

    // ffx_fsr3 with the upscaler built in: the first frame finds that out (AMD runs), later frames skip AMD's call.
    Reset();
    g_fsr3CallsUpscaler = false;
    Hook f3c(MakePlatform(false), StandIn(), "ffx_fsr3");
    CHECK(f3c.F3ContextCreate(ctx.data(), &cd, "game") == OK);
    for (int i = 0; i < 3; ++i) CHECK(f3c.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create f3.dispatchUpscale standin standin standin ");
    CHECK(CountLog("AMD's ffxFsr3ContextDispatchUpscale skipped: UPSCALING_ONLY") == 2);
    CHECK(f3c.F3ContextDestroy(ctx.data(), "game") == OK);

    // fsr3_amd_first = always keeps AMD running.
    Reset();
    g_fsr3CallsUpscaler = true;
    Config always = StandIn();
    always.fsr3AmdFirst = "always";
    Hook up3(MakePlatform(true), always, "ffx_fsr3upscaler");
    Hook f3d(MakePlatform(false), always, "ffx_fsr3");
    g_upscalerHook = &up3;
    CHECK(f3d.F3ContextCreate(ctx.data(), &cd, "game") == OK);
    CHECK(f3d.F3ContextDispatchUpscale(ctx.data(), &d, "game") == OK);
    CHECK(Calls() == "f3.create up.create f3.dispatchUpscale up.dispatch+shared standin ");
    CHECK(f3d.F3ContextDestroy(ctx.data(), "game") == OK);
    g_upscalerHook = nullptr;
}

static void TestMapping() {
    CHECK(ApiSurfaceFormat(SF_R32G32_FLOAT) == FFX_API_SURFACE_FORMAT_R32G32_FLOAT);  // index 5 in 3.0, 6 in the FFX API
    CHECK(ApiSurfaceFormat(SF_R10G10B10A2_UNORM) == FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM);
    CHECK(ApiSurfaceFormat(SF_R32_FLOAT) == FFX_API_SURFACE_FORMAT_R32_FLOAT);
    CHECK(ApiSurfaceFormat(99) == FFX_API_SURFACE_FORMAT_UNKNOWN);
    for (uint32_t f = 1; f < SF_COUNT; ++f) CHECK(ApiSurfaceFormat(f) != FFX_API_SURFACE_FORMAT_UNKNOWN);
    Fsr3ContextDescription c{};
    c.flags = 0xFFFFFFFFu;
    c.displaySize = {3840, 2160};
    c.upscaleOutputSize = {0, 0};
    const UpscaleCreateParams p = CreateParams(c);
    CHECK(p.flags == 0x7F && p.maxUpscaleSize.width == 3840);  // no upscaleOutputSize: the display size
    CHECK(DescribeFsr3Flags(FSR3_FLAG_UPSCALING_ONLY | FLAG_HIGH_DYNAMIC_RANGE) == "HDR|UPSCALING_ONLY");
    CHECK(std::string(ErrorName(ERROR_INVALID_ARGUMENT)) == "FFX_ERROR_INVALID_ARGUMENT");
}

#if !WRONGGPU_MODEL_K
// A build without Model K (CMake WG_MODEL_K=OFF): the FSR 3.0 hooks drop model=K from the settings as the FSR 3.1
// front end does (config.h, IgnoreModelK), with one log line.
static void TestWithoutModelK() {
    Reset();
    Config config;
    config.runtimeOptions = "timing_log=1;model=K";
    {
        Hook hook(MakePlatform(true), config, "");
        size_t notes = 0;
        for (const auto& line : g_log) notes += line.find("Model K is not part of this build; Model M is used") != std::string::npos;
        CHECK(notes == 1 && g_log.size() == 1);
    }
    Reset();
    config.runtimeOptions = "timing_log=1;model=M";
    {
        Hook hook(MakePlatform(true), config, "");
        CHECK(g_log.empty());
    }
    CHECK(IgnoreModelK(config).empty() && config.runtimeOptions == "timing_log=1;model=M");
    config.runtimeOptions = "model=k;timing_log=1";
    CHECK(!IgnoreModelK(config).empty() && config.runtimeOptions == "timing_log=1");
}
#endif

int main() {
#if !WRONGGPU_MODEL_K
    TestWithoutModelK();
#endif
    TestLogOnly();
    TestUpscalerRoute();
    TestNotReady();
    TestFailures();
    TestSdk11Guard();
    TestFsr3Level();
    TestUpscalingOnly();
    TestMapping();
    if (std::getenv("WG_TEST_DUMP_LOG"))
        for (const auto& line : g_log) std::printf("%s\n", line.c_str());
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
