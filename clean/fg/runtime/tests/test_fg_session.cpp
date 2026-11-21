// SPDX-License-Identifier: MIT
// Frame generation (issue #96, F7a): end-to-end CPU tests of the FG session. A fake game drives the game adapter core
// (clean/fg/proxy) on a fake game interface; the core hands assembled game frames to the session through its executor
// hand-off; the session records into a fake kernel executor and presents through the deterministic fake presenter
// (clean/fg/presenter) on a fake clock with fake GPU fences. Every value is synthetic. Built with CI's flags
// (/W4 /WX /permissive-). The same file is also built against deliberately broken sessions (WG_FG_SESSION_MUTANT=k,
// CMakeLists.txt); each of those builds must fail.
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fg_session.h"
#include "../../presenter/fg_fake_presenter.h"
#include "../../proxy/fg_proxy_core.h"
#if defined(WG_FG_SESSION_WIN)
#include "fg_session_win.h"
#endif

using namespace wgfg;

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

bool SameBits(float a, float b) {
    uint32_t x = 0, y = 0;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

// ---- fake game interface (the next link behind the adapter core) ----------------------------------------------------

constexpr int32_t kChainRc = 55;  // what the fake genuine interface answers: a forwarded call returns exactly this
constexpr uint64_t kChain = 0xC0FFEEull;
constexpr WgFgLuid kLuid = {0x2222u, 0};
constexpr uint32_t kNtBackbuffer = 900, kNtMotion = 901, kNtDepth = 902, kNtHudless = 903, kNtUiAlpha = 904,
                   kNtDistortion = 905, kNtNoWarp = 906;

struct FakeChain {
    int calls[8] = {};  // per feature slot: 0 fg, 1 sr, 2 rr, 3 latency, 4 pc latency
    int tags = 0, consts = 0, presents = 0;
    const void* last_native = nullptr;
    static FakeChain* Of(void* s) { return static_cast<FakeChain*>(s); }
    static int Slot(uint32_t f) {
        switch (f) {
            case WG_FEAT_FRAME_GENERATION: return 0;
            case WG_FEAT_SUPER_RESOLUTION: return 1;
            case WG_FEAT_RAY_RECONSTRUCTION: return 2;
            case WG_FEAT_LATENCY: return 3;
            case WG_FEAT_PC_LATENCY: return 4;
            default: return 5;
        }
    }
    static int32_t IsSupported(void* s, uint32_t f, const void* a) {
        Of(s)->calls[Slot(f)]++;
        Of(s)->last_native = a;
        return kChainRc;
    }
    static int32_t GetFunction(void* s, uint32_t f, const char*, void** out) {
        Of(s)->calls[Slot(f)]++;
        if (out) *out = nullptr;
        return kChainRc;
    }
    static int32_t SetOptions(void* s, uint32_t f, uint32_t, const void* o) {
        Of(s)->calls[Slot(f)]++;
        Of(s)->last_native = o;
        return kChainRc;
    }
    static int32_t GetState(void* s, uint32_t f, uint32_t, void* st) {
        Of(s)->calls[Slot(f)]++;
        Of(s)->last_native = st;
        return kChainRc;
    }
    static int32_t Tag(void* s, const void*, uint32_t) {
        Of(s)->tags++;
        return kChainRc;
    }
    static int32_t Constants(void* s, const void*) {
        Of(s)->consts++;
        return kChainRc;
    }
    static int32_t Present(void* s, const WgFgPresentDesc*) {
        Of(s)->presents++;
        return kChainRc;
    }
    WgFgChainV1 table() {
        WgFgChainV1 t = {};
        t.struct_size = sizeof(t);
        t.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        t.self = this;
        t.max_generated = 5;
        t.is_supported = &IsSupported;
        t.get_function = &GetFunction;
        t.set_options = &SetOptions;
        t.get_state = &GetState;
        t.tag = &Tag;
        t.constants = &Constants;
        t.present = &Present;
        return t;
    }
};

struct LogSink {
    std::vector<std::string> lines;
    static void Fn(void* ctx, const char* line) { static_cast<LogSink*>(ctx)->lines.emplace_back(line); }
    int count(const char* needle) const {
        int n = 0;
        for (const auto& l : lines)
            if (l.find(needle) != std::string::npos) ++n;
        return n;
    }
};

// synthetic resources: distinct addresses only, never dereferenced
char g_res[16];
void* Res(int i) { return &g_res[i]; }
int g_fn_a = 0, g_fn_b = 0;

// ---- fake kernel executor --------------------------------------------------------------------------------------------

struct ExecEvent {
    char kind = 0;  // 'S' shared, 'G' generated index, 'C' copy, 'R' real endpoint
    uint64_t token = 0;
    uint64_t interval = 0;
    uint32_t index = 0, n = 0;
    float t = 0.0f;
    bool last = false;
    FgPreset preset = FgPreset::A;
    FgPresetGraph graph = {};
    FgCopyReason reason = FgCopyReason::MenuDetected;
    bool real_only = false;
    bool has_previous = false;
    uint32_t slot = 0;
    bool slot_reserved = false;
};

class FakeKernelExecutor final : public IFgKernelExecutor {
public:
    FakeKernelExecutor(FgFakeClock* clock, FgFakeGpu* gpu) : clock_(clock), gpu_(gpu) {}

    FgExecResult Configure(const FgCreateDesc& desc, uint32_t banks) override {
        ++configures;
        configured_w = desc.output_width;
        configured_h = desc.output_height;
        configured_banks = banks;
        if (!fence_) fence_ = gpu_->CreateFence();
        if (!late_fence_) late_fence_ = gpu_->CreateFence();
        return refuse_configure ? FgExecResult::Refused : FgExecResult::Ok;
    }
    FgExecResult RecordSharedGameFrame(const FgSharedDispatchDesc& d, FgSharedResult* out) override {
        if (Fail('S')) return fail_result;
        ExecEvent e;
        e.kind = 'S';
        e.token = d.frame_token;
        e.interval = d.interval_id;
        e.n = d.generated_count;
        e.preset = d.preset;
        e.graph = d.graph;
        e.has_previous = d.has_previous;
        events.push_back(e);
        if (d.packet == nullptr || d.packet->frame_token != d.frame_token) ++bad_packets;
        for (uint64_t m : menu_tokens)
            if (m == d.frame_token) out->menu.detected = true;
        Signal(0);
        return FgExecResult::Ok;
    }
    FgExecResult RecordGeneratedIndex(const FgIndexDispatchDesc& d, FgReadyFence* ready) override {
        if (Fail('G')) return fail_result;
        ExecEvent e;
        e.kind = 'G';
        e.token = d.frame_token;
        e.interval = d.interval_id;
        e.index = d.index;
        e.n = d.generated_count;
        e.t = d.t;
        e.last = d.is_last_index;
        e.preset = d.preset;
        e.graph = d.graph;
        e.slot = d.slot.slot;
        e.slot_reserved = d.slot.reserved && !d.slot.real && d.slot.index == d.index;
        events.push_back(e);
        if (d.frame_token == late_token) {
            // the GPU finishes this image after its deadline
            *ready = FgReadyFence{late_fence_, ++late_value_};
            gpu_->SignalAt(late_fence_, late_value_, d.target.deadline + 1000);
        } else {
            *ready = Signal(latency);
        }
        if (d.frame_token == stall_token && d.index == 1) {
            // the CPU stalls while recording index 1: every later index of this interval is past its deadline
            clock_->Set(d.target.target + stall_slot);
        }
        return FgExecResult::Ok;
    }
    FgExecResult RecordMenuCopy(const FgCopyDispatchDesc& d, FgReadyFence* ready) override {
        if (Fail('C')) return fail_result;
        ExecEvent e;
        e.kind = 'C';
        e.token = d.frame_token;
        e.interval = d.interval_id;
        e.index = d.index;
        e.n = d.generated_count;
        e.last = d.is_last_index;
        e.reason = d.reason;
        e.slot = d.slot.slot;
        e.slot_reserved = d.slot.reserved && !d.slot.real;
        events.push_back(e);
        *ready = Signal(latency);
        return FgExecResult::Ok;
    }
    FgExecResult RecordRealEndpoint(const FgRealDispatchDesc& d, FgReadyFence* ready) override {
        if (Fail('R')) return fail_result;
        ExecEvent e;
        e.kind = 'R';
        e.token = d.frame_token;
        e.interval = d.interval_id;
        e.real_only = d.real_only;
        e.slot = d.slot.slot;
        e.slot_reserved = d.slot.reserved && d.slot.real;
        events.push_back(e);
        *ready = Signal(latency);
        return FgExecResult::Ok;
    }
    uint64_t LastComputeValue() const override { return value_; }
    uint64_t CompletedComputeValue() const override {
        return compute_stalled ? 0 : gpu_->Completed(fence_, clock_->Now());
    }

    int count(char kind, uint64_t token) const {
        int n = 0;
        for (const ExecEvent& e : events)
            if (e.kind == kind && e.token == token) ++n;
        return n;
    }
    int count(char kind) const {
        int n = 0;
        for (const ExecEvent& e : events)
            if (e.kind == kind) ++n;
        return n;
    }

    // knobs
    FgTicks latency = 5000;
    std::vector<uint64_t> menu_tokens;
    uint64_t late_token = 0;
    uint64_t stall_token = 0;
    FgTicks stall_slot = 0;
    bool refuse_configure = false;
    bool compute_stalled = false;  // the compute timeline reports nothing completed (history banks stay read)
    char fail_kind = 0;        // 'S', 'G', 'C', 'R' or '*' (every record)
    int fail_after = 0;        // calls of that kind that still succeed first
    FgExecResult fail_result = FgExecResult::Refused;

    // observations
    std::vector<ExecEvent> events;
    int configures = 0;
    uint32_t configured_w = 0, configured_h = 0, configured_banks = 0;
    int bad_packets = 0;

private:
    bool Fail(char kind) {
        if (fail_kind == 0) return false;
        if (fail_kind != '*' && fail_kind != kind) return false;
        if (fail_after > 0) {
            --fail_after;
            return false;
        }
        return true;
    }
    FgReadyFence Signal(FgTicks after) {
        ++value_;
        gpu_->SignalAt(fence_, value_, clock_->Now() + after);
        return FgReadyFence{fence_, value_};
    }

    FgFakeClock* clock_;
    FgFakeGpu* gpu_;
    std::uint32_t fence_ = 0, late_fence_ = 0;
    uint64_t value_ = 0, late_value_ = 0;
};

// ---- fake presenter source and a deliberately broken presenter -------------------------------------------------------

// Hands out a reservation without the real image's slot: a presenter the session must refuse, not work around.
class BrokenReservationPresenter final : public IFgPresenter {
public:
    explicit BrokenReservationPresenter(std::unique_ptr<FgFakePresenter> inner) : inner_(std::move(inner)) {}
    FgPresenterProbe Probe() const override { return inner_->Probe(); }
    FgStatus Attach(std::uint64_t c, std::uint64_t e) override { return inner_->Attach(c, e); }
    FgStatus BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) override {
        const FgStatus s = inner_->BeginInterval(plan, out);
        if (s == FgStatus::Ok && out && out->count > 0) out->slots[out->count - 1].real = false;
        return s;
    }
    FgStatus SubmitInterval(const FgIntervalSubmission& s) override { return inner_->SubmitInterval(s); }
    FgStatus PresentRealOnly(const FgIntervalPlan& p, const FgRealImage& r) override {
        return inner_->PresentRealOnly(p, r);
    }
    FgStatus RevokeGenerated(std::uint64_t id, FgDropReason why) override { return inner_->RevokeGenerated(id, why); }
    FgStatus Quiesce() override { return inner_->Quiesce(); }
    FgStatus CompleteResize(bool ok, std::uint64_t e) override { return inner_->CompleteResize(ok, e); }
    FgStatus OnDeviceLost() override { return inner_->OnDeviceLost(); }
    FgStatus Detach() override { return inner_->Detach(); }
    FgPresenterState State() const override { return inner_->State(); }
    FgPresentStats Stats() const override { return inner_->Stats(); }
    FgFakePresenter* inner() { return inner_.get(); }

private:
    std::unique_ptr<FgFakePresenter> inner_;
};

class FakePresenterSource final : public IFgPresenterSource {
public:
    FakePresenterSource(FgFakeClock* clock, FgFakeGpu* gpu, FgFakeChainRegistry* registry)
        : clock_(clock), gpu_(gpu), registry_(registry) {
        config.slot_count = 16;
        config.real_reserve = 1;
        config.max_pending_intervals = 2;
    }
    FgStatus CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out, std::string* why) override {
        ++creates;
        last_desc = desc;
        if (refuse != FgStatus::Ok) {
            if (why) *why = "fake source refused";
            return refuse;
        }
        std::unique_ptr<FgFakePresenter> p(new FgFakePresenter(config, clock_, gpu_, registry_));
        last = p.get();
        if (broken_reservation) {
            out->reset(new BrokenReservationPresenter(std::move(p)));
        } else {
            *out = std::move(p);
        }
        return FgStatus::Ok;
    }
    FgFakeConfig config;
    FgStatus refuse = FgStatus::Ok;
    bool broken_reservation = false;
    int creates = 0;
    FgCreateDesc last_desc;
    FgFakePresenter* last = nullptr;

private:
    FgFakeClock* clock_;
    FgFakeGpu* gpu_;
    FgFakeChainRegistry* registry_;
};

struct MenuSink {
    std::vector<FgMenuDecision> decisions;
    std::vector<bool> copies;
    static void Fn(void* ctx, const FgMenuDecision& d, bool copy) {
        MenuSink* m = static_cast<MenuSink*>(ctx);
        m->decisions.push_back(d);
        m->copies.push_back(copy);
    }
};

FgTicks ClockFn(void* ctx) { return static_cast<FgFakeClock*>(ctx)->Now(); }

// named create-time jobs (no capturing lambdas)
struct OkJob {
    FgRefusal operator()() const { return FgRefusal::None; }
};
struct RefusingJob {
    FgRefusal operator()() const { return FgRefusal::Net1PipelineMissing; }
};
struct ThrowingJob {
    FgRefusal operator()() const { throw std::runtime_error("synthetic create failure"); }
};

// ---- the rig: fake game -> adapter core -> session -> fake executor -> fake presenter -------------------------------

struct FrameRecord {
    uint64_t token = 0;
    uint32_t proxy_status = 0;
    uint32_t proxy_reason = 0;
    uint32_t proxy_real_only = 0;
    FgSessionResult session;
};

wgfgp::FgProxyConfig QualifiedConfig(uint32_t multiplier) {
    wgfgp::FgProxyConfig c;
    c.settings.enabled = true;
    c.settings.multiplier = multiplier;
    c.binding.qualified = true;
    c.binding.version_major = 2;
    c.binding.version_minor = 14;
    const uint32_t nt[WG_FG_ROLE_COUNT] = {kNtBackbuffer, kNtMotion, kNtDepth, kNtHudless, kNtUiAlpha, kNtDistortion,
                                           kNtNoWarp};
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r) c.binding.native_type[r] = nt[r];
    c.binding.fn_names = {"fgTestSetOptions", "fgTestGetState"};
    c.binding.ui_max_generated = 5;
    c.modes_qualified = true;  // the release gate is false; these tests exercise the taken route
    return c;
}

struct Rig {
    FgFakeClock clock;
    FgFakeGpu gpu;
    FgFakeChainRegistry registry;
    FakeChain chain;
    FakeKernelExecutor exec{&clock, &gpu};
    FakePresenterSource source{&clock, &gpu, &registry};
    LogSink proxy_log, session_log;
    MenuSink menu;
    wgfgp::FgPresentOwners owners;
    std::unique_ptr<wgfgp::FgProxy> proxy;
    FgSession session;
    FgSessionDesc desc;
    std::vector<FrameRecord> frames;
    uint32_t out_w = 1600, out_h = 900;
    uint32_t render_w = 0, render_h = 0;  // 0: half the output size
    uint64_t sc_epoch = 1;
    FgTicks period = 200000;  // 50 game frames per second
    FgTicks step = 1000;      // presenter wakeups every 0.1 ms
    FgTicks t = 10000000;
    uint64_t token = 0;
    uint32_t preset = WG_FG_PRESET_A;
    uint32_t count = 1;

    explicit Rig(uint32_t n = 1, uint32_t p = WG_FG_PRESET_A) : preset(p), count(n) {
        clock.Set(t);
        // the player's multiplier setting is 6x (the cap); the game asks for its own count through its options
        proxy.reset(new wgfgp::FgProxy(QualifiedConfig(6), &owners, &LogSink::Fn, &proxy_log));
        WgFgChainV1 ct = chain.table();
        proxy->attach_interposer(&ct, 2, 14);
        WgFgAdapterIdentity a = {};
        a.struct_size = sizeof(a);
        a.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        a.vendor_id = 0x1002;
        a.device_id = 0x1234;
        a.luid = kLuid;
        a.rdna4 = 1;
        a.probes_passed = 1;
        proxy->set_adapter(a);
        proxy->set_device(Res(10), Res(11));
        proxy->set_runtime_facts(true, true);
        proxy->set_feature_function("fgTestSetOptions", &g_fn_a);
        proxy->set_feature_function("fgTestGetState", &g_fn_b);
        SetOptions(n, p);
        proxy->claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER);

        desc.create = CreateDesc();
        desc.executor = &exec;
        desc.presenter_source = &source;
        desc.history_banks = 3;
        desc.clock = &ClockFn;
        desc.clock_ctx = &clock;
        desc.log = &LogSink::Fn;
        desc.log_ctx = &session_log;
        desc.on_menu = &MenuSink::Fn;
        desc.menu_ctx = &menu;
    }

    WgFgOptions Options(uint32_t n, uint32_t p) const {
        WgFgOptions o = {};
        o.struct_size = sizeof(o);
        o.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        o.enable = 1;
        o.preset = p;
        o.generated_count = n;
        for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) o.optional_decl[i] = WG_FG_DECL_MAY;
        o.color_space = WG_FG_CS_SDR;
        o.output_width = out_w;
        o.output_height = out_h;
        o.output_format = 28;
        o.render_min_width = 640;
        o.render_min_height = 360;
        o.render_max_width = out_w;
        o.render_max_height = out_h;
        o.dynamic_resolution = 1;
        return o;
    }
    int32_t SetOptions(uint32_t n, uint32_t p) {
        count = n;
        preset = p;
        WgFgOptions o = Options(n, p);
        return proxy->set_options(WG_FEAT_FRAME_GENERATION, 1, &o, &o);
    }
    FgCreateDesc CreateDesc() const {
        FgCreateDesc c;
        c.device = Res(10);
        c.queue = Res(11);
        c.adapter_luid.low = kLuid.low;
        c.adapter_luid.high = kLuid.high;
        c.swapchain_id = kChain;
        c.window_id = 0x77;
        c.queue_epoch = 1;
        c.swapchain_epoch = sc_epoch;
        c.viewport_id = 1;
        c.output_width = out_w;
        c.output_height = out_h;
        c.output_format = 28;
        c.render_min_width = 640;
        c.render_min_height = 360;
        c.render_max_width = out_w;
        c.render_max_height = out_h;
        c.dynamic_resolution = 1;
        c.preset = static_cast<FgPreset>(preset);
        c.max_generated = 5;
        for (uint32_t i = 0; i < kFgOptCount; ++i) c.optional[i] = FgOptionalDecl::May;
        return c;
    }
    FgSessionResult Create() {
        const FgSessionResult r = session.Create(desc);
        if (r.error == FgSessionError::None) {
            WgFgExecutorV1 h = session.ExecutorHandOff();
            proxy->set_executor(&h);
        }
        return r;
    }
    FgFakePresenter* presenter() const { return source.last; }

    WgFgTag Tag(uint32_t nt, void* res, uint64_t tok) const {
        WgFgTag g = {};
        g.struct_size = sizeof(g);
        g.native_type = nt;
        g.texture.resource = res;
        g.texture.format = 10;
        g.texture.width = out_w / 2;
        g.texture.height = out_h / 2;
        g.texture.frame_token = tok;
        g.texture.viewport_id = 1;
        g.texture.producer_epoch = 3;
        g.texture.rect = {0, 0, out_w / 2, out_h / 2};
        return g;
    }
    void Feed(uint64_t tok, uint32_t flags) {
        WgFgTag tags[2] = {Tag(kNtMotion, Res(1), tok), Tag(kNtDepth, Res(2), tok)};
        proxy->tag(tags, 2, tags);
        WgFgConstants c = {};
        c.struct_size = sizeof(c);
        c.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        c.frame_token = tok;
        c.viewport_id = 1;
        c.flags = flags | WG_FG_FLAG_MOTION_RENDER_RES;
        for (int i = 0; i < 16; ++i) {
            c.camera_current[i] = 0.125f * static_cast<float>(i);
            c.camera_previous[i] = -0.25f * static_cast<float>(i);
        }
        c.jitter_x = 0.25f;
        c.jitter_y = -0.25f;
        c.motion_scale_x = 1.0f;
        c.motion_scale_y = 1.0f;
        c.render_rect = {0, 0, render_w ? render_w : out_w / 2, render_h ? render_h : out_h / 2};
        c.output_rect = {0, 0, out_w, out_h};
        proxy->constants(&c, &c);
    }
    WgFgPresentDesc PresentDesc(uint64_t tok) const {
        WgFgPresentDesc p = {};
        p.struct_size = sizeof(p);
        p.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        p.owner = WG_FG_OWNER_NATIVE_ADAPTER;
        p.viewport_id = 1;
        p.frame_token = tok;
        p.swapchain_id = kChain;
        p.window_id = 0x77;
        p.queue_epoch = 1;
        p.swapchain_epoch = sc_epoch;
        p.luid = kLuid;
        p.backbuffer.resource = Res(0);
        p.backbuffer.format = 28;
        p.backbuffer.width = out_w;
        p.backbuffer.height = out_h;
        p.backbuffer.rect = {0, 0, out_w, out_h};
        return p;
    }
    // Advances the clock to `until` waking the presenter every `step`.
    void PumpUntil(FgTicks until) {
        FgTicks x = clock.Now();
        while (x < until) {
            x += step;
            clock.Set(x);
            if (presenter() && presenter()->State() != FgPresenterState::Detached) presenter()->Pump();
        }
    }
    // One game frame: tags and constants, the present through the adapter core, then the presenter runs until the
    // next game frame is due.
    FrameRecord& Frame(uint32_t flags = 0) {
        ++token;
        if (clock.Now() < t) clock.Set(t);
        Feed(token, flags);
        WgFgPresentDesc d = PresentDesc(token);
        FrameRecord rec;
        rec.token = token;
        WgFgPresentResult pr = {};
        FgSessionPresent(proxy.get(), &session, &d, &pr, &rec.session);
        rec.proxy_status = pr.status;
        rec.proxy_reason = pr.reason;
        rec.proxy_real_only = pr.real_only;
        frames.push_back(rec);
        t += period;
        PumpUntil(t);
        return frames.back();
    }
    void Frames(int n, uint32_t flags = 0) {
        for (int i = 0; i < n; ++i) Frame(flags);
    }
    int RealShown(uint64_t tok) const {
        int n = 0;
        for (const FgFakeDisplayRecord& d : presenter()->Displayed())
            if (d.real && d.frame_id == tok) ++n;
        return n;
    }
    int GeneratedShown(uint64_t tok) const {
        int n = 0;
        for (const FgFakeDisplayRecord& d : presenter()->Displayed())
            if (!d.real && d.frame_id == tok) ++n;
        return n;
    }
    bool HistoryOk() const {
        std::string why;
        const bool ok = FgCheckHistoryInvariants(session.History()->Snapshot(), &why);
        if (!ok) std::printf("history invariant: %s\n", why.c_str());
        return ok;
    }
    // Lets every queued image reach the display, then releases the chain.
    bool Shutdown() {
        PumpUntil(t + 4 * period);
        FgSessionResult r = session.Destroy();
        for (int i = 0; i < 64 && r.error == FgSessionError::PresenterBusy; ++i) {
            PumpUntil(clock.Now() + period);
            r = session.Destroy();
        }
        return r.error == FgSessionError::None && registry.Owner(kChain) == nullptr &&
               session.State() == FgSessionState::None;
    }
};

// The index records of one game frame, in record order.
std::vector<ExecEvent> IndexEvents(const FakeKernelExecutor& e, uint64_t token, char kind) {
    std::vector<ExecEvent> out;
    for (const ExecEvent& ev : e.events)
        if (ev.kind == kind && ev.token == token) out.push_back(ev);
    return out;
}

// Record order of one game frame: shared first (once), then indices 1..N in order, then the real endpoint.
bool FrameOrderOk(const FakeKernelExecutor& e, uint64_t token, uint32_t n, char index_kind) {
    std::string seq;
    std::vector<uint32_t> idx;
    for (const ExecEvent& ev : e.events) {
        if (ev.token != token) continue;
        seq.push_back(ev.kind);
        if (ev.kind == index_kind) idx.push_back(ev.index);
    }
    std::string want = "S" + std::string(n, index_kind) + "R";
    if (seq != want) {
        std::printf("frame %llu record order '%s', want '%s'\n", static_cast<unsigned long long>(token), seq.c_str(),
                    want.c_str());
        return false;
    }
    for (uint32_t i = 0; i < idx.size(); ++i)
        if (idx[i] != i + 1) return false;
    return true;
}

// ---- tests ---------------------------------------------------------------------------------------------------------

constexpr int kWarmup = 6;  // frames until the cadence estimate is stable and history is primed

// Every count (2x..6x) with every preset, end to end.
void TestEndToEndCountsAndPresets() {
    const uint32_t presets[3] = {WG_FG_PRESET_A, WG_FG_PRESET_B, WG_FG_PRESET_C};
    for (uint32_t n = 1; n <= 5; ++n) {
        for (uint32_t p : presets) {
            Rig rig(n, p);
            CHECK(rig.Create().error == FgSessionError::None);
            CHECK(rig.session.State() == FgSessionState::Active);
            const int frames = 24;
            rig.Frames(frames);
            FgPresetGraph want_graph = {};
            CHECK(FgPresetGraphOf(static_cast<FgPreset>(p), &want_graph));
            int generated_frames = 0;
            bool all_ok = true;
            for (const FrameRecord& f : rig.frames) {
                all_ok = all_ok && f.proxy_status == WG_FG_PRESENT_GENERATE;
                all_ok = all_ok && f.session.error == FgSessionError::None;
                // the shared work exactly once per game frame
                all_ok = all_ok && rig.exec.count('S', f.token) == 1;
                // every game frame reached the display exactly once
                all_ok = all_ok && rig.RealShown(f.token) == 1;
                const std::vector<ExecEvent> g = IndexEvents(rig.exec, f.token, 'G');
                if (f.session.scheduled == 0) {
                    all_ok = all_ok && g.empty();
                    continue;
                }
                ++generated_frames;
                all_ok = all_ok && f.session.frame == FgFrameStatus::Generated;
                all_ok = all_ok && g.size() == n && f.session.generated_recorded == n;
                all_ok = all_ok && FrameOrderOk(rig.exec, f.token, n, 'G');
                for (uint32_t i = 0; i < g.size(); ++i) {
                    const ExecEvent& e = g[i];
                    all_ok = all_ok && e.index == i + 1 && e.n == n;
                    all_ok = all_ok && SameBits(e.t, FgIndexTime(i + 1, n));
                    all_ok = all_ok && SameBits(e.t, static_cast<float>(i + 1) / (static_cast<float>(n) + 1.0f));
                    all_ok = all_ok && e.last == (i + 1 == n);
                    all_ok = all_ok && e.preset == static_cast<FgPreset>(p);
                    all_ok = all_ok && e.graph.full_blend == want_graph.full_blend &&
                             e.graph.ui_recompose == want_graph.ui_recompose &&
                             e.graph.lite_blend == want_graph.lite_blend;
                    all_ok = all_ok && e.slot_reserved;
                }
                all_ok = all_ok && rig.GeneratedShown(f.token) == static_cast<int>(n);
                all_ok = all_ok && f.session.history_advanced;
            }
            CHECK(all_ok);
            CHECK(generated_frames >= frames - kWarmup);
            CHECK(rig.exec.count('S') == frames);
            CHECK(rig.exec.count('G') == generated_frames * static_cast<int>(n));
            CHECK(rig.exec.count('C') == 0);
            CHECK(rig.exec.bad_packets == 0);
            CHECK(rig.session.Counters().history_advances == static_cast<uint64_t>(generated_frames));
            CHECK(rig.session.Counters().game_frames == static_cast<uint64_t>(frames));
            CHECK(rig.session.History()->primes() == 1);
            CHECK(rig.session.History()->rotations() == static_cast<uint64_t>(frames - 1));
            CHECK(rig.HistoryOk());
            const FgPresentStats st = rig.presenter()->Stats();
            CHECK(st.dropped_late == 0 && st.dropped_pressure == 0 && st.cancelled == 0);
            CHECK(st.displayed_generated == static_cast<uint64_t>(generated_frames) * n);
            CHECK(rig.chain.presents == 0);  // the taken route never hands the present to the next link
            CHECK(rig.Shutdown());
        }
    }
}

// 2x -> 6x -> 3x in one session: the count follows the game at game-frame boundaries and history is never reset.
void TestDynamicCountNoHistoryReset() {
    Rig rig(1, WG_FG_PRESET_B);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    const uint64_t epoch0 = rig.session.History()->epoch();
    const uint64_t primes0 = rig.session.History()->primes();
    CHECK(primes0 == 1);
    CHECK(rig.SetOptions(5, WG_FG_PRESET_B) == WG_OK);
    rig.Frames(10);
    CHECK(rig.SetOptions(2, WG_FG_PRESET_B) == WG_OK);
    rig.Frames(10);
    CHECK(rig.session.History()->epoch() == epoch0);   // no Invalidate
    CHECK(rig.session.History()->primes() == primes0); // 0 history frames reset
    CHECK(rig.session.History()->rotations() == 29);
    CHECK(rig.session.Counters().count_changes == 2);
    CHECK(rig.HistoryOk());
    // every frame's indices use that frame's frozen count; each block reaches its new count
    int seen[6] = {};
    bool ok = true;
    for (const FrameRecord& f : rig.frames) {
        ok = ok && rig.exec.count('S', f.token) == 1 && rig.RealShown(f.token) == 1;
        const std::vector<ExecEvent> g = IndexEvents(rig.exec, f.token, 'G');
        if (g.empty()) continue;
        const uint32_t n = g.front().n;
        ok = ok && n == f.session.scheduled && g.size() == n;
        for (uint32_t i = 0; i < g.size(); ++i)
            ok = ok && g[i].n == n && SameBits(g[i].t, FgIndexTime(i + 1, n)) && g[i].last == (i + 1 == n);
        if (n <= 5) seen[n]++;
    }
    CHECK(ok);
    CHECK(seen[1] >= 3 && seen[5] >= 3 && seen[2] >= 3);
    // the last frame of each block generates at that block's count
    CHECK(IndexEvents(rig.exec, 10, 'G').size() == 1);
    CHECK(IndexEvents(rig.exec, 20, 'G').size() == 5);
    CHECK(IndexEvents(rig.exec, 30, 'G').size() == 2);
    CHECK(rig.Shutdown());
}

// Preset A -> C -> B: the preset is taken from the game frame and frozen for all of its indices.
void TestPresetSwitchAtFrameBoundary() {
    Rig rig(3, WG_FG_PRESET_A);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(8);
    CHECK(rig.SetOptions(3, WG_FG_PRESET_C) == WG_OK);
    rig.Frames(4);
    CHECK(rig.SetOptions(3, WG_FG_PRESET_B) == WG_OK);
    rig.Frames(4);
    CHECK(rig.session.Counters().preset_changes == 2);
    CHECK(rig.session.History()->primes() == 1);
    bool ok = true;
    for (const FrameRecord& f : rig.frames) {
        const std::vector<ExecEvent> g = IndexEvents(rig.exec, f.token, 'G');
        const FgPreset want = f.token <= 8 ? FgPreset::A : f.token <= 12 ? FgPreset::C : FgPreset::B;
        for (const ExecEvent& e : g) ok = ok && e.preset == want && e.graph.lite_blend == (want == FgPreset::C);
        for (const ExecEvent& e : IndexEvents(rig.exec, f.token, 'S')) ok = ok && e.preset == want;
    }
    CHECK(ok);
    CHECK(IndexEvents(rig.exec, 9, 'G').size() == 3);   // the switch frame itself generates, with the new preset
    CHECK(IndexEvents(rig.exec, 13, 'G').size() == 3);
    CHECK(rig.Shutdown());
}

// Menu detected on frame k: frames k and k + 1 show exact game-frame copies, k + 2 generates again; history keeps
// rotating (nothing re-primes).
void TestMenuLatch() {
    Rig rig(4, WG_FG_PRESET_A);
    const uint64_t k = 12;
    rig.exec.menu_tokens = {k};
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(20);
    CHECK(rig.exec.count('G', k) == 0 && rig.exec.count('G', k + 1) == 0);
    const std::vector<ExecEvent> ck = IndexEvents(rig.exec, k, 'C');
    const std::vector<ExecEvent> ck1 = IndexEvents(rig.exec, k + 1, 'C');
    CHECK(ck.size() == 4 && ck1.size() == 4);
    bool reasons = true;
    for (uint32_t i = 0; i < ck.size(); ++i) {
        reasons = reasons && ck[i].reason == FgCopyReason::MenuDetected && ck[i].index == i + 1;
        reasons = reasons && ck[i].last == (i + 1 == 4);
    }
    for (const ExecEvent& e : ck1) reasons = reasons && e.reason == FgCopyReason::MenuLatched;
    CHECK(reasons);
    CHECK(FrameOrderOk(rig.exec, k, 4, 'C'));
    CHECK(FrameOrderOk(rig.exec, k + 1, 4, 'C'));
    CHECK(rig.exec.count('G', k + 2) == 4 && rig.exec.count('C', k + 2) == 0);
    CHECK(rig.exec.count('G', k - 1) == 4);
    CHECK(rig.frames[k - 1].session.menu_detected && rig.frames[k - 1].session.menu_copy);
    CHECK(!rig.frames[k].session.menu_detected && rig.frames[k].session.menu_copy);
    CHECK(!rig.frames[k + 1].session.menu_copy);
    CHECK(rig.session.Counters().menu_detections == 1);
    CHECK(rig.session.Counters().menu_copy_frames == 2);
    // copies are shown like generated images: pacing never changes
    CHECK(rig.GeneratedShown(k) == 4 && rig.GeneratedShown(k + 1) == 4);
    CHECK(rig.RealShown(k) == 1 && rig.RealShown(k + 1) == 1 && rig.RealShown(k + 2) == 1);
    // history is not corrupted or reset by the menu frames
    CHECK(rig.session.History()->primes() == 1);
    CHECK(rig.session.History()->rotations() == 19);
    CHECK(rig.HistoryOk());
    // the decision callback saw every frame's decision; copy flags only on k, k + 1
    CHECK(rig.menu.decisions.size() == 20);
    bool cb = true;
    for (size_t i = 0; i < rig.menu.decisions.size(); ++i) {
        const uint64_t tok = rig.menu.decisions[i].frame_token;
        cb = cb && rig.menu.decisions[i].detected == (tok == k);
        cb = cb && rig.menu.copies[i] == (tok == k || tok == k + 1);
    }
    CHECK(cb);
    CHECK(rig.Shutdown());

    // two detections in a row extend the latch by one frame
    Rig r2(2, WG_FG_PRESET_B);
    r2.exec.menu_tokens = {10, 11};
    CHECK(r2.Create().error == FgSessionError::None);
    r2.Frames(16);
    CHECK(r2.exec.count('C', 10) == 2 && r2.exec.count('C', 11) == 2 && r2.exec.count('C', 12) == 2);
    CHECK(r2.exec.count('G', 13) == 2);
    CHECK(r2.session.History()->primes() == 1);
    CHECK(r2.Shutdown());

    // the game's own menu flag: the adapter core shows the frame real-only; the session still presents it
    Rig r3(2, WG_FG_PRESET_A);
    CHECK(r3.Create().error == FgSessionError::None);
    r3.Frames(10);
    const FrameRecord flagged = r3.Frame(WG_FG_FLAG_MENU);
    CHECK(flagged.proxy_status == WG_FG_PRESENT_REAL_ONLY && flagged.proxy_real_only == WG_FG_REAL_ONLY_MENU);
    CHECK(flagged.session.error == FgSessionError::None && flagged.session.frame == FgFrameStatus::RealOnly);
    CHECK(r3.exec.count('S', flagged.token) == 0 && r3.exec.count('G', flagged.token) == 0);
    CHECK(r3.RealShown(flagged.token) == 1);
    r3.Frames(10);
    CHECK(r3.exec.count('G', r3.token) == 2);  // generation resumes
    CHECK(r3.HistoryOk());
    CHECK(r3.Shutdown());
}

// Super resolution, ray reconstruction and the latency features in the same game session reach the game interface
// untouched, and never reach the session.
void TestOtherFeaturesPassThrough() {
    Rig rig(2, WG_FG_PRESET_A);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(8);
    const FgSessionCounters before = rig.session.Counters();
    const size_t events_before = rig.exec.events.size();
    const uint32_t feats[4] = {WG_FEAT_SUPER_RESOLUTION, WG_FEAT_RAY_RECONSTRUCTION, WG_FEAT_LATENCY,
                               WG_FEAT_PC_LATENCY};
    int forwarded = 0;
    for (uint32_t f : feats) {
        static int native_a = 0, native_o = 0, native_s = 0;
        WgFgLuid luid = kLuid;
        if (rig.proxy->is_supported(f, &luid, &native_a) == kChainRc && rig.chain.last_native == &native_a) ++forwarded;
        void* fn = &g_fn_a;
        if (rig.proxy->get_function(f, "fgTestSetOptions", &fn) == kChainRc && fn == nullptr) ++forwarded;
        WgFgOptions o = rig.Options(2, WG_FG_PRESET_A);
        if (rig.proxy->set_options(f, 1, &o, &native_o) == kChainRc && rig.chain.last_native == &native_o) ++forwarded;
        WgFgState st = {};
        st.struct_size = sizeof(st);
        if (rig.proxy->get_state(f, 1, &st, &native_s) == kChainRc && rig.chain.last_native == &native_s) ++forwarded;
    }
    CHECK(forwarded == 16);
    for (int s = 1; s <= 4; ++s) CHECK(rig.chain.calls[s] == 4);
    CHECK(rig.chain.calls[0] == 0);  // FG itself is answered by the core
    const FgSessionCounters after = rig.session.Counters();
    CHECK(after.game_frames == before.game_frames && after.shared_records == before.shared_records);
    CHECK(rig.exec.events.size() == events_before);
    // the shared tag / constants calls are always forwarded too
    CHECK(rig.chain.tags == 8 && rig.chain.consts == 8);
    rig.Frames(4);  // and generation goes on unchanged after them
    CHECK(rig.exec.count('G', rig.token) == 2);
    CHECK(rig.chain.presents == 0);
    CHECK(rig.Shutdown());
}

// Resize mid-session: admission stops, old work drains, the new epoch primes, and every game frame is shown.
void TestResize() {
    Rig rig(3, WG_FG_PRESET_A);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    rig.PumpUntil(rig.t + 2 * rig.period);  // the last interval reaches the display
    rig.t = rig.clock.Now();
    CHECK(rig.session.BeginResize().error == FgSessionError::None);
    CHECK(rig.session.State() == FgSessionState::Resizing);
    // presents during the resize are refused explicitly (the transport keeps the game's own path)
    CHECK(rig.session.OnGameFrameOnly(999, 0, 0).error == FgSessionError::WrongState);
    rig.out_w = 1920;
    rig.out_h = 1080;
    rig.sc_epoch = 2;
    FgSessionResult r = rig.session.CompleteResize(true, rig.CreateDesc());
    for (int i = 0; i < 16 && r.error == FgSessionError::PresenterBusy; ++i) {
        rig.PumpUntil(rig.clock.Now() + rig.period);
        r = rig.session.CompleteResize(true, rig.CreateDesc());
    }
    CHECK(r.error == FgSessionError::None);
    CHECK(rig.session.State() == FgSessionState::Active);
    CHECK(rig.session.PresentEpoch() == 2);
    CHECK(rig.exec.configured_w == 1920 && rig.exec.configured_h == 1080);
    CHECK(rig.presenter()->Drained());
    CHECK(rig.presenter()->LeasesHeld() == 0);
    CHECK(rig.session.Counters().resizes == 1);
    rig.t = rig.clock.Now();
    CHECK(rig.SetOptions(3, WG_FG_PRESET_A) == WG_OK);  // the game re-sends its options for the new size
    rig.Frames(12);
    bool shown = true;
    for (const FrameRecord& f : rig.frames) shown = shown && rig.RealShown(f.token) == 1;
    CHECK(shown);  // every game frame, before and after the resize, reached the display once
    CHECK(rig.presenter()->Stats().cancelled == 0);
    CHECK(rig.exec.count('G', rig.token) == 3);  // generation resumed on the new epoch
    CHECK(rig.session.History()->primes() == 1);  // the new history primed once
    CHECK(rig.HistoryOk());
    CHECK(rig.Shutdown());

    // a failed swap-chain resize is passed through: the session refuses, no fallback
    Rig r2(2, WG_FG_PRESET_A);
    CHECK(r2.Create().error == FgSessionError::None);
    r2.Frames(8);
    r2.PumpUntil(r2.t + 2 * r2.period);
    CHECK(r2.session.BeginResize().error == FgSessionError::None);
    FgSessionResult f = r2.session.CompleteResize(false, r2.CreateDesc());
    for (int i = 0; i < 16 && f.error == FgSessionError::PresenterBusy; ++i) {
        r2.PumpUntil(r2.clock.Now() + r2.period);
        f = r2.session.CompleteResize(false, r2.CreateDesc());
    }
    CHECK(f.error == FgSessionError::ResizeFailed && f.presenter == FgStatus::ResizeFailed);
    CHECK(r2.session.State() == FgSessionState::Refused);
    CHECK(r2.session.OnGameFrameOnly(100, 0, 0).error == FgSessionError::SessionRefused);
    CHECK(r2.Shutdown());
}

// Device loss: everything is dropped at once, every later call refuses, nothing more is recorded.
void TestDeviceLost() {
    Rig rig(4, WG_FG_PRESET_C);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    CHECK(rig.session.OnDeviceLost().error == FgSessionError::DeviceLost);
    CHECK(rig.session.State() == FgSessionState::Lost);
    CHECK(rig.presenter()->State() == FgPresenterState::Lost);
    CHECK(rig.presenter()->LeasesHeld() == 0);
    const size_t events = rig.exec.events.size();
    const FrameRecord& f = rig.Frame();
    CHECK(f.proxy_status == WG_FG_PRESENT_REAL_ONLY && f.proxy_reason == WG_FG_REASON_EXECUTOR_REFUSED);
    CHECK(f.session.error == FgSessionError::DeviceLost);
    CHECK(rig.exec.events.size() == events);
    CHECK(rig.session.BeginResize().error == FgSessionError::DeviceLost);
    CHECK(rig.Shutdown());

    // the executor reports the loss while recording an index
    Rig r2(3, WG_FG_PRESET_A);
    CHECK(r2.Create().error == FgSessionError::None);
    r2.Frames(10);
    r2.exec.fail_kind = 'G';
    r2.exec.fail_after = 1;
    r2.exec.fail_result = FgExecResult::DeviceLost;
    const FrameRecord& g = r2.Frame();
    CHECK(g.session.error == FgSessionError::ExecutorDeviceLost);
    CHECK(r2.session.State() == FgSessionState::Lost);
    CHECK(r2.presenter()->State() == FgPresenterState::Lost);
    CHECK(r2.Shutdown());
}

// Leaving the display's variable-refresh window suspends generation (game frames only, at their own targets) and
// re-entering resumes it, without losing a game frame or resetting history.
void TestVrrExitAndReentry() {
    Rig rig(5, WG_FG_PRESET_A);
    rig.desc.scheduler.display.min_interval = 27778;  // 360 Hz panel ceiling
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    CHECK(rig.exec.count('G', rig.token) == 5);
    rig.period = 100000;  // 100 game frames per second: 6x would need 600 Hz
    rig.Frames(20);
    int out_of_window = 0;
    for (size_t i = 10; i < rig.frames.size(); ++i)
        if (rig.frames[i].session.plan_real_only == FgRealOnlyReason::OutOfVrrWindow) ++out_of_window;
    CHECK(out_of_window >= 10);
    CHECK(rig.exec.count('G', rig.token) == 0);
    rig.period = 200000;  // back inside the window
    rig.Frames(30);
    CHECK(rig.exec.count('G', rig.token) == 5);
    bool shown = true, shared = true;
    for (const FrameRecord& f : rig.frames) {
        shown = shown && rig.RealShown(f.token) == 1;
        shared = shared && rig.exec.count('S', f.token) == 1;
    }
    CHECK(shown);   // no game frame lost while suspended
    CHECK(shared);  // the shared work kept history inputs current
    CHECK(rig.session.History()->primes() == 1);
    CHECK(rig.HistoryOk());
    CHECK(rig.presenter()->Stats().real_only_intervals[static_cast<uint32_t>(FgRealOnlyReason::OutOfVrrWindow)] >= 10);
    CHECK(rig.Shutdown());
}

// Late images: the GPU finishing after the deadline drops those images (the game frame stays); the CPU falling
// behind while recording cancels the stale suffix of the interval.
void TestLateDrops() {
    Rig rig(5, WG_FG_PRESET_A);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    rig.exec.late_token = 11;
    rig.Frames(6);
    CHECK(rig.exec.count('G', 11) == 5);
    CHECK(rig.GeneratedShown(11) == 0);
    CHECK(rig.RealShown(11) == 1);
    CHECK(rig.presenter()->Stats().dropped_late == 5);
    CHECK(rig.GeneratedShown(12) == 5);
    rig.PumpUntil(rig.t + 4 * rig.period);
    CHECK(rig.presenter()->LeasesHeld() == 0);  // the held leases came back once their fences completed

    rig.t = rig.clock.Now();
    rig.exec.stall_token = 18;
    rig.exec.stall_slot = rig.period / 6 + 1000;
    rig.Frames(6);
    const FrameRecord& stalled = rig.frames[17];
    CHECK(stalled.token == 18);
    CHECK(stalled.session.error == FgSessionError::None);
    CHECK(stalled.session.generated_recorded == 1);
    CHECK(stalled.session.stale_cancelled == 4);
    CHECK(stalled.session.history_advanced);
    CHECK(rig.RealShown(18) == 1);
    CHECK(rig.GeneratedShown(18) <= 1);
    CHECK(rig.exec.count('G', 19) == 5 || rig.exec.count('G', 20) == 5);
    CHECK(rig.session.Counters().stale_cancelled == 4);
    bool shown = true;
    for (const FrameRecord& f : rig.frames) shown = shown && rig.RealShown(f.token) == 1;
    CHECK(shown);
    CHECK(rig.session.History()->primes() == 1);
    CHECK(rig.HistoryOk());
    CHECK(rig.Shutdown());
}

// A broken executor: the session refuses explicitly and never falls back; the game frame of the failing interval is
// still shown when it can be recorded.
void TestExecutorRefusalNoFallback() {
    {  // shared work refused
        Rig rig(3, WG_FG_PRESET_A);
        CHECK(rig.Create().error == FgSessionError::None);
        rig.Frames(10);
        const uint64_t gen_before = rig.presenter()->Stats().generated_queued;
        rig.exec.fail_kind = 'S';
        const FrameRecord& f = rig.Frame();
        CHECK(rig.presenter()->Stats().generated_queued == gen_before);
        CHECK(f.session.error == FgSessionError::ExecutorRefused);
        CHECK(f.proxy_status == WG_FG_PRESENT_REAL_ONLY && f.proxy_reason == WG_FG_REASON_EXECUTOR_REFUSED);
        CHECK(rig.session.State() == FgSessionState::Refused);
        CHECK(rig.session.RefusedWith() == FgSessionError::ExecutorRefused);
        CHECK(rig.exec.count('G', f.token) == 0);
        CHECK(rig.exec.count('R', f.token) == 1);
        CHECK(rig.RealShown(f.token) == 1);  // the game frame of the failing interval was still shown
        const size_t events = rig.exec.events.size();
        rig.Frames(5);
        CHECK(rig.exec.events.size() == events);  // nothing more is recorded, by this or any other route
        for (size_t i = 11; i < rig.frames.size(); ++i)
            CHECK(rig.frames[i].session.error == FgSessionError::SessionRefused);
        CHECK(rig.session_log.count("refused") == 1);  // one log line for the transition
        CHECK(rig.Shutdown());
    }
    {  // an index refused part-way: nothing generated is shown for that interval, the game frame is
        Rig rig(5, WG_FG_PRESET_B);
        CHECK(rig.Create().error == FgSessionError::None);
        rig.Frames(10);
        const uint64_t gen_before = rig.presenter()->Stats().generated_queued;
        rig.exec.fail_kind = 'G';
        rig.exec.fail_after = 2;
        const FrameRecord& f = rig.Frame();
        CHECK(f.session.error == FgSessionError::ExecutorRefused);
        CHECK(rig.session.State() == FgSessionState::Refused);
        CHECK(rig.exec.count('G', f.token) == 2);
        CHECK(rig.RealShown(f.token) == 1);
        CHECK(rig.GeneratedShown(f.token) == 0);
        CHECK(rig.presenter()->Stats().generated_queued == gen_before);
        CHECK(rig.Shutdown());
    }
    {  // everything refused, the game frame too: the session shows nothing and refuses
        Rig rig(2, WG_FG_PRESET_C);
        CHECK(rig.Create().error == FgSessionError::None);
        rig.Frames(10);
        rig.exec.fail_kind = '*';
        const FrameRecord& f = rig.Frame();
        CHECK(f.session.error == FgSessionError::ExecutorRefused || f.session.error == FgSessionError::SessionRefused);
        CHECK(rig.session.State() == FgSessionState::Refused);
        CHECK(rig.RealShown(f.token) == 0);
        CHECK(rig.presenter()->State() == FgPresenterState::Quiescing);
        CHECK(rig.Shutdown());
    }
    {  // a presenter that loses the real image's slot in its reservation is refused, not worked around
        Rig rig(2, WG_FG_PRESET_A);
        rig.source.broken_reservation = true;
        CHECK(rig.Create().error == FgSessionError::None);
        rig.Frames(10);
        bool refused = false;
        for (const FrameRecord& f : rig.frames)
            refused = refused || f.session.error == FgSessionError::PlanMismatch;
        CHECK(refused);
        CHECK(rig.session.State() == FgSessionState::Refused);
        CHECK(rig.session.RefusedWith() == FgSessionError::PlanMismatch);
        CHECK(rig.exec.count('G') == 0);
        CHECK(rig.Shutdown());
    }
}

// Create: every failure is explicit and leaves nothing held.
void TestCreateRefusals() {
    {
        Rig rig;
        rig.desc.create.device = nullptr;
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::CreateInvalid && r.refusal == FgRefusal::NullDevice);
        CHECK(rig.session.State() == FgSessionState::None && rig.source.creates == 0);
    }
    {
        Rig rig;
        rig.desc.executor = nullptr;
        CHECK(rig.Create().error == FgSessionError::ExecutorMissing);
    }
    {
        Rig rig;
        rig.desc.model_path = "fg-session-test-no-such-model.bin";
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::ModelInvalid && r.refusal == FgRefusal::ModelMissing);
        CHECK(rig.source.creates == 0 && rig.exec.configures == 0);
    }
    {
        Rig rig;
        rig.desc.create_work = FgCopiedJob<RefusingJob>(RefusingJob());
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::WorkerFailed && r.refusal == FgRefusal::Net1PipelineMissing);
    }
    {
        Rig rig;
        rig.desc.create_work = FgCopiedJob<ThrowingJob>(ThrowingJob());
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::WorkerFailed && r.refusal == FgRefusal::WorkerInitFailed);
        CHECK(rig.session.State() == FgSessionState::None);
    }
    {
        Rig rig;
        rig.desc.create_work = FgCopiedJob<OkJob>(OkJob());
        CHECK(rig.Create().error == FgSessionError::None);
        rig.Frames(8);
        CHECK(rig.exec.count('G', rig.token) == 1);
        CHECK(rig.Shutdown());
    }
    {
        Rig rig;
        rig.exec.refuse_configure = true;
        CHECK(rig.Create().error == FgSessionError::ExecutorConfigure);
        CHECK(rig.source.creates == 0);
    }
    {
        Rig rig;
        rig.source.refuse = FgStatus::Unsupported;
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::PresenterSource && r.presenter == FgStatus::Unsupported);
        CHECK(rig.session.State() == FgSessionState::None);
        CHECK(rig.session_log.count("presenter refused") == 1);
    }
    {
        Rig rig;
        rig.source.config.probe.supported = false;
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::PresenterAttach);
        CHECK(rig.registry.Owner(kChain) == nullptr);
    }
    {  // one owner per chain: a second session on the same chain is refused
        Rig rig;
        CHECK(rig.Create().error == FgSessionError::None);
        FakePresenterSource second_source(&rig.clock, &rig.gpu, &rig.registry);
        FgSessionDesc d2 = rig.desc;
        d2.presenter_source = &second_source;
        FgSession second;
        const FgSessionResult r = second.Create(d2);
        CHECK(r.error == FgSessionError::PresenterAttach && r.presenter == FgStatus::Refused);
        CHECK(rig.Shutdown());
    }
    {  // calls on a session that was never created
        FgSession s;
        FgFramePacket p;
        CHECK(s.OnPresent(p).error == FgSessionError::WrongState);
        CHECK(s.Destroy().error == FgSessionError::WrongState);
        CHECK(s.OnDeviceLost().error == FgSessionError::WrongState);
    }
}

// A game frame history refuses (here: a render size below the session's bounds, which the adapter core does not
// check) is still shown, real-only, by the session; the next good frames prime and generate again.
void TestFrameRefusedStillShown() {
    Rig rig(2, WG_FG_PRESET_A);
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    rig.render_w = 320;
    rig.render_h = 180;
    const FrameRecord bad = rig.Frame();
    CHECK(bad.proxy_status == WG_FG_PRESENT_GENERATE);
    CHECK(bad.session.error == FgSessionError::FrameRefused);
    CHECK(bad.session.refusal == FgRefusal::RenderRectOutsideBounds);
    CHECK(bad.session.frame == FgFrameStatus::Refused);
    CHECK(rig.exec.count('S', bad.token) == 0 && rig.exec.count('G', bad.token) == 0);
    CHECK(rig.exec.count('R', bad.token) == 1);
    CHECK(rig.RealShown(bad.token) == 1);
    CHECK(rig.session.State() == FgSessionState::Active);  // a frame refusal is not a session refusal
    CHECK(rig.session.Counters().frame_refusals == 1);
    rig.render_w = rig.render_h = 0;
    rig.Frames(10);
    CHECK(rig.exec.count('G', rig.token) == 2);
    bool shown = true;
    for (const FrameRecord& f : rig.frames) shown = shown && rig.RealShown(f.token) == 1;
    CHECK(shown);
    CHECK(rig.HistoryOk());
    CHECK(rig.Shutdown());
}

// History cannot rotate while compute still reads its banks (HistoryBusy): the frame's generated slots get exact
// game-frame copies (Priming), the next frame primes (shared work without a previous frame, copies again), and the
// one after generates; every game frame is shown and the bank invariants hold throughout.
void TestHistoryBusyPrimingCopies() {
    Rig rig(3, WG_FG_PRESET_A);
    rig.desc.history_banks = 2;
    CHECK(rig.Create().error == FgSessionError::None);
    rig.Frames(10);
    rig.exec.compute_stalled = true;
    const FrameRecord busy = rig.Frame();
    rig.exec.compute_stalled = false;
    CHECK(busy.session.error == FgSessionError::None);
    CHECK(busy.session.history_real_only == FgFrameRealOnly::HistoryBusy);
    CHECK(rig.HistoryOk());
    rig.Frames(3);
    const uint64_t b = busy.token;
    CHECK(rig.exec.count('S', b) == 0);  // nothing accepted into a bank: no shared work
    const std::vector<ExecEvent> c0 = IndexEvents(rig.exec, b, 'C');
    CHECK(c0.size() == 3 && rig.exec.count('G', b) == 0);
    bool priming = true;
    for (const ExecEvent& e : c0) priming = priming && e.reason == FgCopyReason::Priming;
    CHECK(priming);
    CHECK(rig.frames[b].session.history_real_only == FgFrameRealOnly::Prime);
    const std::vector<ExecEvent> s1 = IndexEvents(rig.exec, b + 1, 'S');
    CHECK(s1.size() == 1 && !s1.front().has_previous);
    CHECK(rig.exec.count('C', b + 1) == 3 && rig.exec.count('G', b + 1) == 0);
    CHECK(rig.exec.count('G', b + 2) == 3);
    const std::vector<ExecEvent> s2 = IndexEvents(rig.exec, b + 2, 'S');
    CHECK(s2.size() == 1 && s2.front().has_previous);
    bool shown = true;
    for (const FrameRecord& f : rig.frames) shown = shown && rig.RealShown(f.token) == 1;
    CHECK(shown);
    CHECK(rig.GeneratedShown(b) == 3 && rig.GeneratedShown(b + 1) == 3);  // pacing kept with copies
    CHECK(rig.session.History()->primes() == 2);
    CHECK(rig.HistoryOk());
    CHECK(rig.Shutdown());
}

// The hand-off rejects null arguments and an explicit per-index packet without touching the session.
void TestHandOffArguments() {
    Rig rig(2);
    CHECK(rig.Create().error == FgSessionError::None);
    WgFgExecutorV1 h = rig.session.ExecutorHandOff();
    CHECK(h.struct_size == sizeof(WgFgExecutorV1) && h.abi_version == WG_FG_ADAPTER_ABI_VERSION && h.submit);
    WgFgPacket p = {};
    WgFgSessionDesc s = {};
    CHECK(h.submit(nullptr, &p, &s) != 0);
    CHECK(h.submit(h.self, nullptr, &s) != 0);
    CHECK(h.submit(h.self, &p, nullptr) != 0);
    FgFramePacket idx;
    idx.generated_count = 2;
    idx.generated_index = 1;
    const FgSessionResult r = rig.session.OnPresent(idx);
    CHECK(r.error == FgSessionError::BadArgument && r.refusal == FgRefusal::BadGeneratedIndex);
    CHECK(rig.session.State() == FgSessionState::Active);
    CHECK(rig.exec.events.empty());
    CHECK(rig.Shutdown());
}

#if defined(WG_FG_SESSION_WIN)
// The Windows presenter source: the chosen back end is created or refused with the factory's reason; never another.
void TestWindowsPresenterSourceRefuses() {
    FgPresenterCapabilities none;  // nothing supported, nothing probed: no device is touched
    none.adapter_probed = true;
    none.presentation_probed = true;
    none.tearing_probed = true;
    FgPresenterFactoryDesc base;
    base.caps = &none;
    const uint32_t modes[4] = {WG_FG_PRESENTER_AUTO, WG_FG_PRESENTER_HWFQ, WG_FG_PRESENTER_DXGI_TEARING,
                               WG_FG_PRESENTER_DXGI_VSYNC};
    for (uint32_t m : modes) {
        FgWinPresenterSource src(base);
        Rig rig(2);
        rig.desc.presenter_source = &src;
        rig.desc.create.presenter_mode = m;
        const FgSessionResult r = rig.Create();
        CHECK(r.error == FgSessionError::PresenterSource);
        CHECK(!src.LastResult().Ok());
        FgPresenterChoice want = FgPresenterChoice::Auto;
        CHECK(FgPresenterChoiceOfMode(m, &want));
        CHECK(src.LastSelection().requested == want);
        // an explicit choice is refused or kept, never replaced by another back end; Auto picks by its fixed rule
        // (fixed refresh, no composition capabilities: DXGI flip with vsync), whose creation then fails on its own
        const FgPresenterChoice chosen = src.LastSelection().chosen;
        if (m == WG_FG_PRESENTER_AUTO)
            CHECK(chosen == FgPresenterChoice::Auto || chosen == FgPresenterChoice::DxgiFlipVsync);
        else
            CHECK(chosen == FgPresenterChoice::Auto || chosen == want);
        CHECK(src.LastResult().refusal != FgWinRefusal::None);
        CHECK(rig.session.State() == FgSessionState::None);
    }
    {  // tearing supported but no swap chain: the DXGI back end refuses, the composition one is not tried
        FgPresenterCapabilities tear = none;
        tear.tearing_supported = true;
        FgPresenterFactoryDesc b2;
        b2.caps = &tear;
        FgWinPresenterSource src(b2);
        Rig rig(2);
        rig.desc.presenter_source = &src;
        rig.desc.create.presenter_mode = WG_FG_PRESENTER_DXGI_TEARING;
        CHECK(rig.Create().error == FgSessionError::PresenterSource);
        CHECK(src.LastSelection().chosen == FgPresenterChoice::DxgiFlipTearing);
        CHECK(!src.LastResult().Ok());
        CHECK(src.LastResult().refusal != FgWinRefusal::None);
    }
    {  // an unknown presenter mode
        FgWinPresenterSource src(base);
        Rig rig(2);
        rig.desc.presenter_source = &src;
        rig.desc.create.presenter_mode = 9;
        CHECK(rig.Create().error == FgSessionError::PresenterSource);
        CHECK(src.LastResult().refusal == FgWinRefusal::UnknownChoice);
    }
}
#endif

void Run(const char* name, void (*test)()) {
    const int before = g_failed;
    test();
    std::printf("%s: %s\n", name, g_failed == before ? "ok" : "FAILED");
    std::fflush(stdout);
}

}  // namespace

int main() {
    Run("TestEndToEndCountsAndPresets", &TestEndToEndCountsAndPresets);
    Run("TestDynamicCountNoHistoryReset", &TestDynamicCountNoHistoryReset);
    Run("TestPresetSwitchAtFrameBoundary", &TestPresetSwitchAtFrameBoundary);
    Run("TestMenuLatch", &TestMenuLatch);
    Run("TestOtherFeaturesPassThrough", &TestOtherFeaturesPassThrough);
    Run("TestResize", &TestResize);
    Run("TestDeviceLost", &TestDeviceLost);
    Run("TestVrrExitAndReentry", &TestVrrExitAndReentry);
    Run("TestLateDrops", &TestLateDrops);
    Run("TestExecutorRefusalNoFallback", &TestExecutorRefusalNoFallback);
    Run("TestCreateRefusals", &TestCreateRefusals);
    Run("TestHandOffArguments", &TestHandOffArguments);
    Run("TestFrameRefusedStillShown", &TestFrameRefusedStillShown);
    Run("TestHistoryBusyPrimingCopies", &TestHistoryBusyPrimingCopies);
#if defined(WG_FG_SESSION_WIN)
    Run("TestWindowsPresenterSourceRefuses", &TestWindowsPresenterSourceRefuses);
#endif
    std::printf("fg_session: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
