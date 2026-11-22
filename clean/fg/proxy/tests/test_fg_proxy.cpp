// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8a): CPU tests of the FG game-adapter core and the FG settings keys, against
// a fake game interface (super resolution, ray reconstruction, frame generation and the latency features in one
// session), a fake host and a fake executor. Every value is synthetic. Built with CI's flags (/W4 /WX /permissive-).
// The same file is also built against deliberately broken cores (WG_FG_PROXY_MUTANT=k, CMakeLists.txt); each of those
// builds must fail.
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../fg_proxy_core.h"
#include "../fg_settings.h"

#if defined(_WIN32)
#include "wg_state.h"
#endif

#if defined(WG_FG_HAVE_RUNTIME_CONTRACT)
#include "../fg_contract_bridge.h"
#endif

using namespace wgfgp;

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

// ---- fakes ----------------------------------------------------------------------------------------------------------

constexpr uint32_t kFeatFg = WG_FEAT_FRAME_GENERATION;
constexpr uint32_t kFeatSr = WG_FEAT_SUPER_RESOLUTION;
constexpr uint32_t kFeatRr = WG_FEAT_RAY_RECONSTRUCTION;
constexpr uint32_t kFeatLatency = WG_FEAT_LATENCY;
constexpr uint32_t kFeatPcLatency = WG_FEAT_PC_LATENCY;
constexpr uint32_t kFeatUnknown = 4242;
constexpr int32_t kChainRc = 55;  // what the fake genuine interface answers: a forwarded call returns exactly this

// The genuine game interface behind the adapter: counts every call per feature and keeps the native arguments.
struct FakeChain {
    std::map<uint32_t, int> supported, getfn, options, state;
    int tags = 0, consts = 0, presents = 0;
    const void* native_adapter = nullptr;
    const void* native_options = nullptr;
    void* native_state = nullptr;
    const void* native_tags = nullptr;
    const void* native_consts = nullptr;
    const WgFgPresentDesc* present_desc = nullptr;
    const char* name = nullptr;
    uint32_t tag_count = 0;

    static FakeChain* Of(void* s) { return static_cast<FakeChain*>(s); }
    static int32_t IsSupported(void* s, uint32_t f, const void* a) {
        Of(s)->supported[f]++;
        Of(s)->native_adapter = a;
        return kChainRc;
    }
    static int32_t GetFunction(void* s, uint32_t f, const char* n, void** out) {
        Of(s)->getfn[f]++;
        Of(s)->name = n;
        if (out) *out = nullptr;
        return kChainRc;
    }
    static int32_t SetOptions(void* s, uint32_t f, uint32_t, const void* o) {
        Of(s)->options[f]++;
        Of(s)->native_options = o;
        return kChainRc;
    }
    static int32_t GetState(void* s, uint32_t f, uint32_t, void* st) {
        Of(s)->state[f]++;
        Of(s)->native_state = st;
        return kChainRc;
    }
    static int32_t Tag(void* s, const void* t, uint32_t n) {
        Of(s)->tags++;
        Of(s)->native_tags = t;
        Of(s)->tag_count = n;
        return kChainRc;
    }
    static int32_t Constants(void* s, const void* c) {
        Of(s)->consts++;
        Of(s)->native_consts = c;
        return kChainRc;
    }
    static int32_t Present(void* s, const WgFgPresentDesc* p) {
        Of(s)->presents++;
        Of(s)->present_desc = p;
        return kChainRc;
    }
    WgFgChainV1 table(uint32_t max_generated = 5) {
        WgFgChainV1 t = {};
        t.struct_size = sizeof(t);
        t.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        t.self = this;
        t.max_generated = max_generated;
        t.is_supported = &IsSupported;
        t.get_function = &GetFunction;
        t.set_options = &SetOptions;
        t.get_state = &GetState;
        t.tag = &Tag;
        t.constants = &Constants;
        t.present = &Present;
        return t;
    }
    int feature_calls(uint32_t f) const {
        int n = 0;
        for (const auto* m : {&supported, &getfn, &options, &state}) {
            auto it = m->find(f);
            if (it != m->end()) n += it->second;
        }
        return n;
    }
};

struct FakeExecutor {
    int submits = 0;
    int32_t rc = 0;
    WgFgPacket last = {};
    WgFgSessionDesc session = {};
    static int32_t Submit(void* s, const WgFgPacket* p, const WgFgSessionDesc* d) {
        FakeExecutor* e = static_cast<FakeExecutor*>(s);
        ++e->submits;
        e->last = *p;
        e->session = *d;
        return e->rc;
    }
    WgFgExecutorV1 table() {
        WgFgExecutorV1 t = {};
        t.struct_size = sizeof(t);
        t.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        t.self = this;
        t.submit = &Submit;
        return t;
    }
};

struct FakeHost {
    int registers = 0, unregisters = 0;
    int32_t register_rc = 0;
    const WgFgProviderV1* provider = nullptr;
    static int32_t Register(void* s, const WgFgProviderV1* p) {
        FakeHost* h = static_cast<FakeHost*>(s);
        ++h->registers;
        if (h->register_rc == 0) h->provider = p;
        return h->register_rc;
    }
    static int32_t Unregister(void* s, const WgFgProviderV1* p) {
        FakeHost* h = static_cast<FakeHost*>(s);
        ++h->unregisters;
        if (h->provider == p) h->provider = nullptr;
        return 0;
    }
    WgFgHostV1 table(uint32_t max_generated = 5) {
        WgFgHostV1 t = {};
        t.struct_size = sizeof(t);
        t.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        t.self = this;
        t.max_generated = max_generated;
        t.register_provider = &Register;
        t.unregister_provider = &Unregister;
        return t;
    }
};

struct LogSink {
    std::vector<std::string> lines;
    static void Fn(void* ctx, const char* line) { static_cast<LogSink*>(ctx)->lines.emplace_back(line); }
};

// synthetic resources: distinct addresses only, never dereferenced
char g_res[32];
void* Res(int i) { return &g_res[i]; }
int g_fn_a = 0, g_fn_b = 0;  // stand-ins for the feature-function trampolines

// synthetic buffer-type numbers of a test binding
constexpr uint32_t kNtBackbuffer = 900, kNtMotion = 901, kNtDepth = 902, kNtHudless = 903, kNtUiAlpha = 904,
                   kNtDistortion = 905, kNtNoWarp = 906, kNtOther = 950;
constexpr WgFgLuid kLuid = {0x1111u, 0};
constexpr uint64_t kChain = 0xC0FFEEull;

FgProxyConfig QualifiedConfig() {
    FgProxyConfig c;
    c.settings.enabled = true;
    c.settings.multiplier = 6;
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

WgFgAdapterIdentity Rdna4() {
    WgFgAdapterIdentity a = {};
    a.struct_size = sizeof(a);
    a.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    a.vendor_id = 0x1002;
    a.device_id = 0x1234;
    a.luid = kLuid;
    a.rdna4 = 1;
    a.probes_passed = 1;
    return a;
}

WgFgOptions Options(uint32_t count = WG_FG_COUNT_NOT_SET, uint32_t preset = WG_FG_PRESET_NOT_SET) {
    WgFgOptions o = {};
    o.struct_size = sizeof(o);
    o.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    o.enable = 1;
    o.preset = preset;
    o.generated_count = count;
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) o.optional_decl[i] = WG_FG_DECL_MAY;
    o.color_space = WG_FG_CS_SDR;
    o.output_width = 1600;
    o.output_height = 900;
    o.output_format = 28;
    o.render_min_width = 800;
    o.render_min_height = 450;
    o.render_max_width = 1600;
    o.render_max_height = 900;
    o.dynamic_resolution = 1;
    return o;
}

WgFgTag Tag(uint32_t native_type, void* res, uint64_t token, uint32_t vp = 1) {
    WgFgTag t = {};
    t.struct_size = sizeof(t);
    t.native_type = native_type;
    t.texture.resource = res;
    t.texture.format = 10;
    t.texture.width = 800;
    t.texture.height = 450;
    t.texture.frame_token = token;
    t.texture.viewport_id = vp;
    t.texture.producer_epoch = 3;
    t.texture.rect = {0, 0, 800, 450};
    return t;
}

WgFgConstants Consts(uint64_t token, uint32_t vp = 1, uint32_t flags = 0) {
    WgFgConstants c = {};
    c.struct_size = sizeof(c);
    c.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    c.frame_token = token;
    c.viewport_id = vp;
    c.flags = flags;
    for (int i = 0; i < 16; ++i) {
        c.camera_current[i] = 0.25f * static_cast<float>(i);
        c.camera_previous[i] = -0.5f * static_cast<float>(i);
    }
    c.jitter_x = 0.125f;
    c.jitter_y = -0.375f;
    c.motion_scale_x = 2.0f;
    c.motion_scale_y = -3.0f;
    c.render_rect = {0, 0, 800, 450};
    c.output_rect = {0, 0, 1600, 900};
    return c;
}

WgFgPresentDesc Present(uint64_t token, uint32_t owner = WG_FG_OWNER_NATIVE_ADAPTER, uint32_t vp = 1) {
    WgFgPresentDesc p = {};
    p.struct_size = sizeof(p);
    p.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    p.owner = owner;
    p.viewport_id = vp;
    p.frame_token = token;
    p.swapchain_id = kChain;
    p.window_id = 0x77;
    p.queue_epoch = 1;
    p.swapchain_epoch = 1;
    p.luid = kLuid;
    p.backbuffer.resource = Res(0);
    p.backbuffer.format = 28;
    p.backbuffer.width = 1600;
    p.backbuffer.height = 900;
    p.backbuffer.rect = {0, 0, 1600, 900};
    return p;
}

// one fully qualified adapter on the fake interface
struct Rig {
    FakeChain chain;
    FakeExecutor exec;
    LogSink log;
    FgPresentOwners owners;
    FgProxy proxy;
    explicit Rig(const FgProxyConfig& cfg, bool attach = true) : proxy(cfg, &owners, &LogSink::Fn, &log) {
        if (!attach) return;
        WgFgChainV1 t = chain.table();
        proxy.attach_interposer(&t, 2, 14);
        proxy.set_adapter(Rdna4());
        proxy.set_runtime_facts(true, true);
        WgFgExecutorV1 e = exec.table();
        proxy.set_executor(&e);
        proxy.set_feature_function("fgTestSetOptions", &g_fn_a);
        proxy.set_feature_function("fgTestGetState", &g_fn_b);
    }
    // tags (motion, depth and the optional roles in `optional_roles`) and constants for one frame
    void feed(uint64_t token, uint32_t optional_mask = 0, uint32_t flags = 0, uint32_t vp = 1) {
        std::vector<WgFgTag> tags = {Tag(kNtMotion, Res(1), token, vp), Tag(kNtDepth, Res(2), token, vp)};
        const uint32_t nt[4] = {kNtHudless, kNtUiAlpha, kNtDistortion, kNtNoWarp};
        for (uint32_t i = 0; i < 4; ++i)
            if (optional_mask & (1u << i)) tags.push_back(Tag(nt[i], Res(3 + static_cast<int>(i)), token, vp));
        proxy.tag(tags.data(), static_cast<uint32_t>(tags.size()), &tags);
        WgFgConstants c = Consts(token, vp, flags);
        proxy.constants(&c, &c);
    }
    WgFgPresentResult present(uint64_t token, uint32_t owner = WG_FG_OWNER_NATIVE_ADAPTER) {
        WgFgPresentDesc d = Present(token, owner);
        WgFgPresentResult r = {};
        proxy.present(&d, &r);
        return r;
    }
    int log_lines(const char* needle) const {
        int n = 0;
        for (const auto& l : log.lines)
            if (l.find(needle) != std::string::npos) ++n;
        return n;
    }
};

// Drives every feature-scoped and shared call once for `feature` with distinct native pointers; returns how many of
// them reached the fake interface with the game's own arguments and result.
int DriveFeature(Rig& rig, uint32_t feature) {
    int forwarded = 0;
    static int native_adapter = 0, native_options = 0, native_state = 0;
    rig.chain.native_adapter = rig.chain.native_options = nullptr;
    rig.chain.native_state = nullptr;
    WgFgLuid luid = kLuid;
    if (rig.proxy.is_supported(feature, &luid, &native_adapter) == kChainRc && rig.chain.native_adapter == &native_adapter)
        ++forwarded;
    void* fn = &g_fn_a;
    if (rig.proxy.get_function(feature, "fgTestSetOptions", &fn) == kChainRc && fn == nullptr) ++forwarded;
    WgFgOptions o = Options();
    if (rig.proxy.set_options(feature, 1, &o, &native_options) == kChainRc &&
        rig.chain.native_options == &native_options)
        ++forwarded;
    WgFgState st = {};
    st.struct_size = sizeof(st);
    if (rig.proxy.get_state(feature, 1, &st, &native_state) == kChainRc && rig.chain.native_state == &native_state)
        ++forwarded;
    return forwarded;
}

// ---- settings -------------------------------------------------------------------------------------------------------

void TestSettingsDefaultsAndParse() {
    FgSettings d;
    CHECK(!d.enabled && d.preset == 0 && d.multiplier == 2 && d.presenter == FgPresenterChoice::Auto);
    CHECK(FgSettingsValid(d));
    std::vector<FgSettingsIssue> issues;
    FgSettings e = FgSettingsFromKeyValues({}, &issues);
    CHECK(!e.enabled && e.preset == 0 && e.multiplier == 2 && e.presenter == FgPresenterChoice::Auto && issues.empty());

    FgSettings s = FgSettingsFromKeyValues(
        {{"rr_mode", "quality"}, {"fg", "on"}, {"fg_preset", "C"}, {"fg_multiplier", "6"}, {"fg_presenter", "hwfq"}},
        &issues);
    CHECK(s.enabled && s.preset == 2 && s.multiplier == 6 && s.presenter == FgPresenterChoice::Hwfq && issues.empty());
    s = FgSettingsFromKeyValues({{"fg_presenter", "dxgi_tearing"}}, nullptr);
    CHECK(s.presenter == FgPresenterChoice::DxgiTearing);
    s = FgSettingsFromKeyValues({{"fg_presenter", "dxgi_vsync"}, {"fg", "off"}}, nullptr);
    CHECK(s.presenter == FgPresenterChoice::DxgiVsync && !s.enabled);

    // strict spellings: every one of these is refused, reported, and leaves the default
    const char* bad_fg[] = {"On", "ON", "1", "true", "yes", "", "of", "onn"};
    for (const char* v : bad_fg) {
        issues.clear();
        s = FgSettingsFromKeyValues({{"fg", v}}, &issues);
        CHECK(!s.enabled && issues.size() == 1 && issues[0].key == "fg" && issues[0].value == v && !issues[0].why.empty());
    }
    const char* bad_preset[] = {"a", "b", "c", "D", "0", "AB", ""};
    for (const char* v : bad_preset) {
        issues.clear();
        s = FgSettingsFromKeyValues({{"fg_preset", v}}, &issues);
        CHECK(s.preset == 0 && issues.size() == 1);
    }
    const char* bad_mult[] = {"1", "7", "0", "02", "+3", "3.0", "-2", "x2", "2x", "", "10"};
    for (const char* v : bad_mult) {
        issues.clear();
        s = FgSettingsFromKeyValues({{"fg_multiplier", v}}, &issues);
        CHECK(s.multiplier == 2 && issues.size() == 1);
    }
    const char* bad_presenter[] = {"Auto", "HWFQ", "vsync", "tearing", "dxgi", "dxgi-vsync", ""};
    for (const char* v : bad_presenter) {
        issues.clear();
        s = FgSettingsFromKeyValues({{"fg_presenter", v}}, &issues);
        CHECK(s.presenter == FgPresenterChoice::Auto && issues.size() == 1);
    }
    // a key's first line counts; other keys are never looked at
    issues.clear();
    s = FgSettingsFromKeyValues({{"fg", "on"}, {"fg", "off"}, {"fgx", "on"}, {"model", "on"}}, &issues);
    CHECK(s.enabled && issues.empty());
    std::string why;
    FgSettings t;
    CHECK(!FgParseSetting("rr_mode", "on", &t, &why) && why == "not a frame generation key");
    CHECK(!FgParseSetting("fg", "on", nullptr, &why));
    CHECK(FgIsSettingsKey("fg") && FgIsSettingsKey("fg_presenter") && !FgIsSettingsKey("rr_mode") &&
          !FgIsSettingsKey("FG"));

    for (uint32_t m = 0; m < 10; ++m)
        CHECK(FgGeneratedCountOf(m) == (m >= 2 && m <= 6 ? m - 1 : 0u));
    CHECK(std::strcmp(FgPresetSpelling(1), "B") == 0 && FgPresetSpelling(3) == nullptr);
    CHECK(FgPresenterSpelling(static_cast<FgPresenterChoice>(4)) == nullptr);
}

void TestSettingsRoundTrip() {
    int combos = 0;
    for (int en = 0; en < 2; ++en)
        for (uint32_t p = 0; p < 3; ++p)
            for (uint32_t m = 2; m <= 6; ++m)
                for (uint32_t pr = 0; pr < 4; ++pr) {
                    FgSettings s;
                    s.enabled = en != 0;
                    s.preset = p;
                    s.multiplier = m;
                    s.presenter = static_cast<FgPresenterChoice>(pr);
                    std::vector<FgSettingsIssue> issues;
                    const FgKeyValues kv = FgSettingsToKeyValues(s);
                    const FgSettings back = FgSettingsFromKeyValues(kv, &issues);
                    CHECK(kv.size() == 4 && issues.empty());
                    CHECK(back.enabled == s.enabled && back.preset == s.preset && back.multiplier == s.multiplier &&
                          back.presenter == s.presenter);
                    ++combos;
                }
    CHECK(combos == 120);

    // apply: FG keys replaced in place or appended; every other entry stays where it is with its value
    FgKeyValues kv = {{"rr_mode", "balanced"}, {"fg_preset", "B"}, {"model", "m"}, {"rr_enabled", "1"}};
    FgSettings s;
    s.enabled = true;
    s.preset = 2;
    s.multiplier = 4;
    s.presenter = FgPresenterChoice::DxgiVsync;
    CHECK(FgApplySettings(s, &kv));
    CHECK(kv.size() == 7);
    CHECK(kv[0] == std::make_pair(std::string("rr_mode"), std::string("balanced")));
    CHECK(kv[1] == std::make_pair(std::string("fg_preset"), std::string("C")));
    CHECK(kv[2] == std::make_pair(std::string("model"), std::string("m")));
    CHECK(kv[3] == std::make_pair(std::string("rr_enabled"), std::string("1")));
    CHECK(kv[4].first == "fg" && kv[4].second == "on");
    CHECK(kv[5].first == "fg_multiplier" && kv[5].second == "4");
    CHECK(kv[6].first == "fg_presenter" && kv[6].second == "dxgi_vsync");

    // an invalid settings value is never written
    FgSettings bad;
    bad.multiplier = 7;
    FgKeyValues before = kv;
    CHECK(!FgApplySettings(bad, &kv) && kv == before);
    CHECK(FgSettingsToKeyValues(bad).empty() && !FgSettingsValid(bad));
    bad = FgSettings{};
    bad.preset = 3;
    CHECK(!FgSettingsValid(bad));
    CHECK(!FgApplySettings(FgSettings{}, nullptr));

#if defined(_WIN32)
    // through the per-game settings file's own parser (wg_state.h): the FG keys are valid key=value lines there, and an
    // existing file's other keys, comments and line ends come back byte for byte
    const std::string file = "; a player comment\r\nrr_mode=balanced\r\n[section]\r\nmodel=m\r\n";
    wgstate::Settings st = wgstate::ParseSettings(file);
    CHECK(!st.corrupt && st.kv.size() == 2);
    CHECK(FgSettingsFromKeyValues(st.kv, nullptr).enabled == false);
    FgKeyValues want = st.kv;
    CHECK(FgApplySettings(s, &want));
    for (const auto& p : want)
        if (FgIsSettingsKey(p.first)) CHECK(wgstate::Set(st, p.first, p.second));
    const std::string out = wgstate::RenderSettings(st);
    CHECK(out.rfind(file, 0) == 0);  // the old text is the unchanged start
    const wgstate::Settings again = wgstate::ParseSettings(out);
    std::vector<FgSettingsIssue> issues;
    const FgSettings back = FgSettingsFromKeyValues(again.kv, &issues);
    CHECK(issues.empty() && back.enabled && back.preset == 2 && back.multiplier == 4 &&
          back.presenter == FgPresenterChoice::DxgiVsync);
    CHECK(wgstate::Get(again, "rr_mode") && *wgstate::Get(again, "rr_mode") == "balanced");
    for (const auto& p : FgSettingsToKeyValues(back)) CHECK(wgstate::ValidKey(p.first) && wgstate::ValidValue(p.second));
#endif
}

// ---- install-off gate, release gate, route order ----------------------------------------------------------------------

void TestDefaultOffForwardsEverything() {
    FgProxyConfig cfg = QualifiedConfig();
    cfg.settings = FgSettings{};  // the shipped default: fg=off
    Rig rig(cfg);
    CHECK(rig.proxy.route_reason() == WG_FG_REASON_INSTALL_OFF);
    CHECK(!rig.proxy.fg_taken() && !rig.proxy.owns_feature(kFeatFg));
    const uint32_t feats[] = {kFeatFg, kFeatSr, kFeatRr, kFeatLatency, kFeatPcLatency, kFeatUnknown};
    for (uint32_t f : feats) {
        CHECK(DriveFeature(rig, f) == 4);
        CHECK(rig.chain.feature_calls(f) == 4);
    }
    CHECK(rig.chain.name && std::strcmp(rig.chain.name, "fgTestSetOptions") == 0);
    // shared calls forwarded, nothing kept
    rig.feed(1, 0xF);
    CHECK(rig.chain.tags == 1 && rig.chain.consts == 1 && rig.chain.tag_count == 6);
    CHECK(rig.proxy.frames_kept() == 0);
    // present forwarded with the caller's own descriptor and result; no chain claimed
    CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_INSTALL_OFF);
    CHECK(rig.owners.owner_of(kChain) == WG_FG_OWNER_NONE);
    WgFgPresentDesc d = Present(1);
    WgFgPresentResult r = {};
    CHECK(rig.proxy.present(&d, &r) == kChainRc);
    CHECK(rig.chain.presents == 1 && rig.chain.present_desc == &d);
    CHECK(r.status == WG_FG_PRESENT_FORWARDED && r.forwarded_presents == 1 && r.base_presents == 0);
    CHECK(rig.exec.submits == 0);
    // capability: truthful adapter, nothing advertised
    const WgFgCapability c = rig.proxy.capability();
    CHECK(c.supported == 0 && c.max_generated == 0 && c.presets_mask == 0 && c.reason == WG_FG_REASON_INSTALL_OFF);
    CHECK(c.adapter.vendor_id == 0x1002 && c.adapter.device_id == 0x1234 && c.adapter.luid.low == kLuid.low);
    CHECK(rig.proxy.advertised_max_generated() == 0);
    // one diagnostic for the decision, not one per call
    CHECK(rig.log_lines("not taken: install switch off") == 1);
}

void TestReleaseGateForwardsEverything() {
    FgProxyConfig cfg = QualifiedConfig();
    // the shipped gate is closed (a developer build with WRONGGPU_FG_DEV_QUALIFIED opens it; the closed gate's
    // behaviour is still checked there)
    CHECK(kFgModesQualified == (WG_FG_MODES_QUALIFIED != 0));
#if !WG_FG_MODES_QUALIFIED
    CHECK(!kFgModesQualified);
#endif
    cfg.modes_qualified = false;
    Rig rig(cfg);
    CHECK(rig.proxy.route_reason() == WG_FG_REASON_ROUTE_NOT_QUALIFIED);
    for (int i = 0; i < 5; ++i) CHECK(DriveFeature(rig, kFeatFg) == 4);
    CHECK(rig.chain.feature_calls(kFeatFg) == 20);
    const WgFgCapability c = rig.proxy.capability();
    CHECK(c.supported == 0 && c.max_generated == 0 && c.reason == WG_FG_REASON_ROUTE_NOT_QUALIFIED);
    CHECK(c.adapter.rdna4 == 1 && c.adapter.probes_passed == 1);
    WgFgPresentResult r = rig.present(1);
    CHECK(r.status == WG_FG_PRESENT_FORWARDED && rig.chain.presents == 1);
    CHECK(rig.log_lines("not taken: modes not qualified") == 1);
}

void TestRouteFactOrder() {
    {
        Rig rig(QualifiedConfig());
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_NONE && rig.proxy.fg_taken());
        CHECK(rig.log_lines("frame generation taken") == 1);
    }
    {
        FgProxyConfig cfg = QualifiedConfig();
        cfg.binding.qualified = false;
        Rig rig(cfg);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_BINDING_NOT_QUALIFIED);
    }
    {
        Rig rig(QualifiedConfig(), false);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_NOT_ATTACHED);
    }
    // the adapter: another vendor, not RDNA 4, probes failed, no LUID, never set
    auto adapter_case = [](void (*mutate)(WgFgAdapterIdentity*)) -> uint32_t {
        Rig rig(QualifiedConfig());
        WgFgAdapterIdentity a = Rdna4();
        mutate(&a);
        rig.proxy.set_adapter(a);
        return rig.proxy.route_reason();
    };
    CHECK(adapter_case([](WgFgAdapterIdentity* a) { a->vendor_id = 0x8086; }) == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    CHECK(adapter_case([](WgFgAdapterIdentity* a) { a->rdna4 = 0; }) == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    CHECK(adapter_case([](WgFgAdapterIdentity* a) { a->probes_passed = 0; }) == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    CHECK(adapter_case([](WgFgAdapterIdentity* a) { a->luid = WgFgLuid{0, 0}; }) == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    CHECK(adapter_case([](WgFgAdapterIdentity* a) { a->struct_size = 8; }) == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    {
        Rig rig(QualifiedConfig());
        rig.proxy.set_runtime_facts(false, true);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_MODEL_NOT_VALIDATED);
        rig.proxy.set_runtime_facts(true, false);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_PRESENTER_NOT_QUALIFIED);
        rig.proxy.set_runtime_facts(true, true);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_NONE);
        // one line per transition: taken, not taken (model), not taken (presenter), taken
        CHECK(rig.log_lines("frame generation taken") == 2);
        CHECK(rig.log_lines("not taken: model not validated") == 2);  // also once while the rig was set up
        CHECK(rig.log_lines("not taken: presenter not qualified") == 1);
    }
    {
        // first failure wins: install off beats everything else
        FgProxyConfig cfg = QualifiedConfig();
        cfg.settings.enabled = false;
        cfg.binding.qualified = false;
        Rig rig(cfg, false);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_INSTALL_OFF);
    }
    {
        // an entry whose ABI offers no generated count never takes FG
        Rig rig(QualifiedConfig(), false);
        WgFgChainV1 t = rig.chain.table(0);
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_NONE);
        rig.proxy.set_adapter(Rdna4());
        rig.proxy.set_runtime_facts(true, true);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_BAD_GENERATED_COUNT);
        CHECK(rig.proxy.capability().max_generated == 0);
    }
}

// ---- feature-scoped routing -----------------------------------------------------------------------------------------

void TestFeatureScopedRouting() {
    Rig rig(QualifiedConfig());
    CHECK(rig.proxy.fg_taken());
    // the other features of the same session reach the interface with the game's own arguments and results
    const uint32_t others[] = {kFeatSr, kFeatRr, kFeatLatency, kFeatPcLatency, kFeatUnknown};
    for (uint32_t f : others) {
        CHECK(!rig.proxy.owns_feature(f));
        CHECK(DriveFeature(rig, f) == 4);
        CHECK(rig.chain.feature_calls(f) == 4);
    }
    // frame generation is answered here: the interface never sees it
    CHECK(rig.proxy.owns_feature(kFeatFg));
    CHECK(DriveFeature(rig, kFeatFg) == 0);
    CHECK(rig.chain.feature_calls(kFeatFg) == 0);
    WgFgLuid luid = kLuid;
    CHECK(rig.proxy.is_supported(kFeatFg, &luid, nullptr) == WG_OK);
    CHECK(rig.proxy.is_supported(kFeatFg, nullptr, nullptr) == WG_OK);
    WgFgLuid other = {0x2222u, 0};
    CHECK(rig.proxy.is_supported(kFeatFg, &other, nullptr) == WG_E_ADAPTER_NOT_SUPPORTED);
    void* fn = nullptr;
    CHECK(rig.proxy.get_function(kFeatFg, "fgTestSetOptions", &fn) == WG_OK && fn == &g_fn_a);
    CHECK(rig.proxy.get_function(kFeatFg, "fgTestGetState", &fn) == WG_OK && fn == &g_fn_b);
    fn = &g_fn_a;
    CHECK(rig.proxy.get_function(kFeatFg, "fgTestNotBound", &fn) == WG_E_FEATURE_MISSING && fn == nullptr);
    CHECK(rig.proxy.get_function(kFeatFg, nullptr, &fn) == WG_E_INVALID_PARAMETER);
    CHECK(rig.chain.feature_calls(kFeatFg) == 0);
    CHECK(rig.log_lines("'fgTestNotBound' refused") == 1);
    // the same name for super resolution is not ours
    CHECK(rig.proxy.get_function(kFeatSr, "fgTestSetOptions", &fn) == kChainRc);
    // shared calls: always forwarded, also while FG is taken (the other features need every tag and constant)
    const int tags_before = rig.chain.tags, consts_before = rig.chain.consts;
    rig.feed(5, 0x3);
    CHECK(rig.chain.tags == tags_before + 1 && rig.chain.consts == consts_before + 1 && rig.chain.tag_count == 4);
    WgFgTag foreign = Tag(kNtOther, Res(9), 5);
    CHECK(rig.proxy.tag(&foreign, 1, &foreign) == kChainRc && rig.chain.native_tags == &foreign);
    WgFgState st = {};
    st.struct_size = sizeof(st);
    CHECK(rig.proxy.get_state(kFeatFg, 1, &st, nullptr) == WG_OK && st.taken == 1 && st.max_generated == 5);
    st.struct_size = 4;
    CHECK(rig.proxy.get_state(kFeatFg, 1, &st, nullptr) == WG_E_INVALID_PARAMETER);
    CHECK(rig.proxy.get_state(kFeatFg, 1, nullptr, nullptr) == WG_E_MISSING_INPUT_PARAMETER);
}

// ---- present owner ----------------------------------------------------------------------------------------------------

void TestSinglePresentOwner() {
    Rig rig(QualifiedConfig());
    WgFgOptions o = Options(1);
    CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
    CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
    CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);  // again: same owner
    CHECK(rig.owners.claim(kChain, WG_FG_OWNER_HOST_WRAPPER) == WG_FG_REASON_PRESENT_OWNED);
    CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_PRESENT_OWNED);
    CHECK(rig.log_lines("present owner refused for swap-chain wrapper") == 1);
    CHECK(rig.owners.owner_of(kChain) == WG_FG_OWNER_NATIVE_ADAPTER);
    CHECK(rig.owners.claim(0, WG_FG_OWNER_HOST_WRAPPER) == WG_FG_REASON_NULL_ARGUMENT);
    CHECK(rig.owners.claim(1, WG_FG_OWNER_NONE) == WG_FG_REASON_NULL_ARGUMENT);
    CHECK(rig.owners.claim(1, 9) == WG_FG_REASON_NULL_ARGUMENT);

    // a second presenter's present is refused and consumes nothing
    rig.feed(1);
    WgFgPresentResult r = rig.present(1, WG_FG_OWNER_HOST_WRAPPER);
    CHECK(r.status == WG_FG_PRESENT_REFUSED && r.reason == WG_FG_REASON_PRESENT_OWNED && r.base_presents == 0);
    r = rig.present(1, WG_FG_OWNER_NONE);
    CHECK(r.status == WG_FG_PRESENT_REFUSED && r.reason == WG_FG_REASON_NOT_PRESENT_OWNER);
    // the owner's present of the same interval generates once
    r = rig.present(1);
    CHECK(r.status == WG_FG_PRESENT_GENERATE && r.base_presents == 1 && rig.exec.submits == 1);
    // the same interval again (any owner) is refused: never consumed twice
    r = rig.present(1);
    CHECK(r.status == WG_FG_PRESENT_REFUSED && r.reason == WG_FG_REASON_TOKEN_NOT_MONOTONIC && r.base_presents == 1);
    CHECK(rig.exec.submits == 1);
    WgFgPresentDesc d = Present(1);
    CHECK(rig.proxy.present(&d, nullptr) == WG_E_INVALID_STATE);
    // an older interval too
    rig.feed(3);
    CHECK(rig.present(3).status == WG_FG_PRESENT_GENERATE);
    CHECK(rig.present(2).reason == WG_FG_REASON_TOKEN_NOT_MONOTONIC);

    // a second adapter in the same process (another entry) cannot take the chain
    LogSink log2;
    FgProxy second(QualifiedConfig(), &rig.owners, &LogSink::Fn, &log2);
    FakeHost host;
    WgFgHostV1 ht = host.table();
    CHECK(second.register_with_host(&ht) == WG_FG_REASON_NONE);
    second.set_adapter(Rdna4());
    second.set_runtime_facts(true, true);
    CHECK(second.fg_taken());
    CHECK(second.claim_present(kChain, WG_FG_OWNER_HOST_WRAPPER) == WG_FG_REASON_PRESENT_OWNED);
    // a chain without an owner: nobody presents it through an adapter
    WgFgPresentDesc unowned = Present(10, WG_FG_OWNER_HOST_WRAPPER);
    unowned.swapchain_id = 0xBEEF;
    WgFgPresentResult ur = {};
    CHECK(second.present(&unowned, &ur) == WG_E_INVALID_STATE && ur.reason == WG_FG_REASON_NOT_PRESENT_OWNER);

    // release hands it over
    CHECK(!rig.proxy.release_present(kChain, WG_FG_OWNER_HOST_WRAPPER));
    CHECK(rig.proxy.release_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER));
    CHECK(second.claim_present(kChain, WG_FG_OWNER_HOST_WRAPPER) == WG_FG_REASON_NONE);
    CHECK(rig.owners.owner_of(kChain) == WG_FG_OWNER_HOST_WRAPPER);
    // detach (or destruction) releases a claimed chain
    second.detach();
    CHECK(rig.owners.owner_of(kChain) == WG_FG_OWNER_NONE);
    CHECK(host.unregisters == 1);
}

// ---- N = 1..5 and presets --------------------------------------------------------------------------------------------

void TestMultiFrameAndPresets() {
    {
        Rig rig(QualifiedConfig());
        CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
        CHECK(rig.proxy.capability().max_generated == 5 && rig.proxy.capability().presets_mask == 0x7);
        uint64_t token = 1;
        for (uint32_t n = 1; n <= 5; ++n) {
            for (uint32_t p = 0; p < 3; ++p) {
                WgFgOptions o = Options(n, p);
                CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
                rig.feed(token);
                const WgFgPresentResult r = rig.present(token);
                CHECK(r.status == WG_FG_PRESENT_GENERATE && r.effective_count == n && r.effective_preset == p);
                CHECK(rig.exec.last.generated_count == n && rig.exec.last.preset == p &&
                      rig.exec.last.generated_index == 0);
                CHECK(rig.exec.session.max_generated == 5 && rig.exec.session.preset == p);
                ++token;
            }
        }
        CHECK(rig.exec.submits == 15);
        // above the range: clamped to 5, one line per new request
        WgFgOptions o = Options(9);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
        CHECK(rig.proxy.state().effective_count == 5);
        CHECK(rig.log_lines("requested generated count 9 clamped to 5") == 1);
        // a preset outside A..C is refused and the previous options stay
        o = Options(2, 3);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_E_INVALID_PARAMETER);
        CHECK(rig.proxy.state().effective_count == 5);
        CHECK(rig.log_lines("options refused: preset not offered") == 1);
        o = Options(2);
        o.optional_decl[1] = 3;
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_E_INVALID_PARAMETER);
        o = Options(2);
        o.struct_size = 16;
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_E_INVALID_PARAMETER);
        o = Options(2);
        o.abi_version = 2;
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_E_INVALID_PARAMETER);
        CHECK(rig.proxy.set_options(kFeatFg, 1, nullptr, nullptr) == WG_E_MISSING_INPUT_PARAMETER);
        // one viewport per session
        o = Options(2);
        CHECK(rig.proxy.set_options(kFeatFg, 2, &o, nullptr) == WG_E_INVALID_STATE);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
        CHECK(rig.log_lines("options accepted again") == 1);
    }
    {
        // the player's multiplier caps N; unset options take the player's preset and multiplier
        FgProxyConfig cfg = QualifiedConfig();
        cfg.settings.multiplier = 3;
        cfg.settings.preset = 1;
        Rig rig(cfg);
        WgFgOptions o = Options(5);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK && rig.proxy.state().effective_count == 2);
        o = Options();
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
        CHECK(rig.proxy.state().effective_count == 2 && rig.proxy.state().effective_preset == 1);
        CHECK(rig.proxy.capability().max_generated == 5);  // the advertised range is the route's, not the player's cap
    }
    {
        // the game's own UI range and the entry ABI's maximum bound what is advertised
        FgProxyConfig cfg = QualifiedConfig();
        cfg.binding.ui_max_generated = 3;
        Rig rig(cfg);
        CHECK(rig.proxy.capability().max_generated == 3 && rig.proxy.advertised_max_generated() == 3);
        WgFgOptions o = Options(5);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK && rig.proxy.state().effective_count == 3);
    }
    {
        Rig rig(QualifiedConfig(), false);
        WgFgChainV1 t = rig.chain.table(2);
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_NONE);
        rig.proxy.set_adapter(Rdna4());
        rig.proxy.set_runtime_facts(true, true);
        CHECK(rig.proxy.capability().max_generated == 2);
        WgFgChainV1 big = rig.chain.table(9);
        Rig rig2(QualifiedConfig(), false);
        CHECK(rig2.proxy.attach_interposer(&big, 2, 14) == WG_FG_REASON_NONE);
        rig2.proxy.set_adapter(Rdna4());
        rig2.proxy.set_runtime_facts(true, true);
        CHECK(rig2.proxy.capability().max_generated == 5);  // never above 5
    }
    {
        // a binding without the Lite preset refuses C
        FgProxyConfig cfg = QualifiedConfig();
        cfg.binding.presets_mask = 0x3;
        Rig rig(cfg);
        CHECK(rig.proxy.capability().presets_mask == 0x3);
        WgFgOptions o = Options(1, WG_FG_PRESET_C);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_E_INVALID_PARAMETER);
        o = Options(1, WG_FG_PRESET_B);
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
    }
    {
        // the game's FG switch off: frames are shown real-only (generation off), nothing is submitted
        Rig rig(QualifiedConfig());
        CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
        WgFgOptions o = Options(3);
        o.enable = 0;
        CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
        rig.feed(1);
        const WgFgPresentResult r = rig.present(1);
        CHECK(r.status == WG_FG_PRESENT_REAL_ONLY && r.real_only == WG_FG_REAL_ONLY_DISABLED && rig.exec.submits == 0);
        CHECK(rig.proxy.state().enabled == 0 && rig.proxy.state().effective_count == 0);
    }
}

// ---- tag-to-packet assembly ------------------------------------------------------------------------------------------

void TestPacketAssembly() {
    FgProxyConfig cfg = QualifiedConfig();
    cfg.settings.presenter = FgPresenterChoice::Hwfq;
    Rig rig(cfg);
    static int device = 0, queue = 0;
    rig.proxy.set_device(&device, &queue);
    WgFgOptions o = Options(4, WG_FG_PRESET_B);
    o.color_space = WG_FG_CS_HDR10_PQ;
    CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
    CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
    WgFgSessionDesc before = {};
    CHECK(!rig.proxy.session_desc(&before));  // no present latched yet
    rig.feed(7, 0xF, WG_FG_FLAG_RESET | WG_FG_FLAG_DEPTH_REVERSED);
    const WgFgPresentResult r = rig.present(7);
    CHECK(r.status == WG_FG_PRESENT_GENERATE && r.reason == WG_FG_REASON_NONE && rig.exec.submits == 1);
    const WgFgPacket& p = rig.exec.last;
    CHECK(p.frame_token == 7 && p.viewport_id == 1 && p.adapter_luid.low == kLuid.low);
    CHECK(p.queue_epoch == 1 && p.swapchain_epoch == 1);
    CHECK(p.color.resource == Res(0) && p.color.frame_token == 7 && p.color.width == 1600);
    CHECK(p.motion.resource == Res(1) && p.depth.resource == Res(2) && p.motion.frame_token == 7);
    CHECK(p.optional_present == 0xF);
    for (int i = 0; i < 4; ++i) CHECK(p.optional[i].resource == Res(3 + i));
    CHECK(p.render_rect.width == 800 && p.output_rect.width == 1600);
    CHECK(p.jitter_x == 0.125f && p.jitter_y == -0.375f && p.motion_scale_x == 2.0f && p.motion_scale_y == -3.0f);
    CHECK(p.camera_current[5] == 1.25f && p.camera_previous[3] == -1.5f);
    CHECK(p.flags == (WG_FG_FLAG_RESET | WG_FG_FLAG_DEPTH_REVERSED | WG_FG_FLAG_HDR));
    CHECK(p.preset == WG_FG_PRESET_B && p.generated_count == 4 && p.generated_index == 0);
    const WgFgSessionDesc& s = rig.exec.session;
    CHECK(s.device == &device && s.queue == &queue && s.swapchain_id == kChain && s.window_id == 0x77);
    CHECK(s.viewport_id == 1 && s.output_width == 1600 && s.output_format == 28 && s.render_max_height == 900);
    CHECK(s.preset == WG_FG_PRESET_B && s.max_generated == 5 && s.hdr == 1);
    CHECK(s.presenter_mode == WG_FG_PRESENTER_HWFQ);
    CHECK(rig.proxy.frames_kept() == 0);  // the presented frame's inputs are let go
    CHECK(rig.log_lines("frames: generating") == 1);

    // the backbuffer from a tag when the present carries none
    rig.feed(8);
    WgFgTag bb = Tag(kNtBackbuffer, Res(20), 8);
    rig.proxy.tag(&bb, 1, &bb);
    WgFgPresentDesc d = Present(8);
    d.backbuffer.resource = nullptr;
    WgFgPresentResult r2 = {};
    CHECK(rig.proxy.present(&d, &r2) == WG_OK && r2.status == WG_FG_PRESENT_GENERATE);
    CHECK(rig.exec.last.color.resource == Res(20));

    // inputs for future frames wait; at most kFgMaxFramesKept sets are kept
    for (uint64_t t = 100; t < 100 + 2 * kFgMaxFramesKept; ++t) rig.feed(t);
    CHECK(rig.proxy.frames_kept() == kFgMaxFramesKept);
}

struct FailCase {
    const char* what;
    uint32_t reason;
    uint32_t real_only;
};

void TestAssemblyFailures() {
    auto fresh = [](Rig& rig, uint32_t decl_hudless = WG_FG_DECL_MAY, uint32_t decl_nowarp = WG_FG_DECL_MAY) {
        WgFgOptions o = Options(2);
        o.optional_decl[0] = decl_hudless;
        o.optional_decl[3] = decl_nowarp;
        rig.proxy.set_options(kFeatFg, 1, &o, nullptr);
        rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER);
    };
    // stale frame token: the inputs are of an older frame than the one presented
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        rig.feed(4);
        const WgFgPresentResult r = rig.present(5);
        CHECK(r.status == WG_FG_PRESENT_REAL_ONLY && r.reason == WG_FG_REASON_STALE_TAG && rig.exec.submits == 0);
        CHECK(r.base_presents == 1 && r.real_only_presents == 1);
        // tags of an interval already presented are not kept
        rig.feed(5);
        CHECK(rig.proxy.frames_kept() == 0);
        // five stale presents in a row: one diagnostic; generating again: one more
        for (uint64_t t = 7; t < 17; t += 2) {
            rig.feed(t - 1);
            CHECK(rig.present(t).reason == WG_FG_REASON_STALE_TAG);
        }
        CHECK(rig.log_lines("real-only (stale frame token)") == 1);
        rig.feed(20);
        CHECK(rig.present(20).status == WG_FG_PRESENT_GENERATE);
        rig.feed(19);  // older than the presented interval
        rig.feed(21);
        CHECK(rig.present(21).status == WG_FG_PRESENT_GENERATE);
        CHECK(rig.log_lines("frames: generating") == 1);
    }
    // missing inputs, one at a time
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        WgFgTag depth = Tag(kNtDepth, Res(2), 1);
        rig.proxy.tag(&depth, 1, &depth);
        WgFgConstants c = Consts(1);
        rig.proxy.constants(&c, &c);
        CHECK(rig.present(1).reason == WG_FG_REASON_MISSING_MOTION);
        WgFgTag motion = Tag(kNtMotion, Res(1), 2);
        rig.proxy.tag(&motion, 1, &motion);
        c = Consts(2);
        rig.proxy.constants(&c, &c);
        CHECK(rig.present(2).reason == WG_FG_REASON_STALE_TAG);  // depth was tagged, but only for frame 1
        WgFgTag both[2] = {Tag(kNtMotion, Res(1), 3), Tag(kNtDepth, Res(2), 3)};
        rig.proxy.tag(both, 2, both);
        CHECK(rig.present(3).reason == WG_FG_REASON_MISSING_CONSTANTS);
        rig.feed(4);
        WgFgPresentDesc d = Present(4);
        d.backbuffer.resource = nullptr;
        WgFgPresentResult r = {};
        rig.proxy.present(&d, &r);
        CHECK(r.reason == WG_FG_REASON_MISSING_COLOR);
        CHECK(rig.exec.submits == 0 && rig.proxy.state().real_only_presents == 4);
    }
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        WgFgTag motion = Tag(kNtMotion, Res(1), 1);
        rig.proxy.tag(&motion, 1, &motion);
        WgFgConstants c = Consts(1);
        rig.proxy.constants(&c, &c);
        CHECK(rig.present(1).reason == WG_FG_REASON_MISSING_DEPTH);
    }
    // conflicting tags: one role, one frame, two resources
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        rig.feed(1);
        WgFgTag again = Tag(kNtMotion, Res(11), 1);
        rig.proxy.tag(&again, 1, &again);
        CHECK(rig.present(1).reason == WG_FG_REASON_CONFLICTING_TAG);
        // the same resource tagged twice is no conflict
        rig.feed(2);
        rig.feed(2);
        CHECK(rig.present(2).status == WG_FG_PRESENT_GENERATE);
    }
    // optional inputs against their declarations
    {
        Rig rig(QualifiedConfig());
        fresh(rig, WG_FG_DECL_ALWAYS, WG_FG_DECL_NEVER);
        rig.feed(1, 0x0);
        CHECK(rig.present(1).reason == WG_FG_REASON_MISSING_PROMISED_OPTIONAL);
        rig.feed(2, 0x1 | 0x8);
        CHECK(rig.present(2).reason == WG_FG_REASON_UNDECLARED_OPTIONAL);
        rig.feed(3, 0x1 | 0x2);
        CHECK(rig.present(3).status == WG_FG_PRESENT_GENERATE && rig.exec.last.optional_present == 0x3);
    }
    // viewport, LUID, epochs, menu
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        rig.feed(1, 0, 0, 2);  // inputs of viewport 2 are not this session's
        CHECK(rig.proxy.frames_kept() == 0);
        WgFgPresentDesc d = Present(1, WG_FG_OWNER_NATIVE_ADAPTER, 2);
        WgFgPresentResult r = {};
        rig.proxy.present(&d, &r);
        CHECK(r.status == WG_FG_PRESENT_REAL_ONLY && r.reason == WG_FG_REASON_VIEWPORT_MISMATCH);
        rig.feed(2);
        d = Present(2);
        d.luid.low = 0x9999;
        rig.proxy.present(&d, &r);
        CHECK(r.reason == WG_FG_REASON_LUID_MISMATCH);
        rig.feed(3);
        CHECK(rig.present(3).status == WG_FG_PRESENT_GENERATE);
        rig.feed(4);
        d = Present(4);
        d.swapchain_epoch = 2;  // the chain was resized / recreated
        rig.proxy.present(&d, &r);
        CHECK(r.reason == WG_FG_REASON_SWAPCHAIN_EPOCH_MISMATCH && r.real_only == WG_FG_REAL_ONLY_EPOCH_CHANGE);
        rig.feed(5);
        d = Present(5);
        d.swapchain_epoch = 2;  // the new epoch generates again
        rig.proxy.present(&d, &r);
        CHECK(r.status == WG_FG_PRESENT_GENERATE && rig.exec.last.swapchain_epoch == 2);
        rig.feed(6);
        d = Present(6);
        d.swapchain_epoch = 2;
        d.queue_epoch = 5;
        rig.proxy.present(&d, &r);
        CHECK(r.reason == WG_FG_REASON_QUEUE_EPOCH_MISMATCH && r.real_only == WG_FG_REAL_ONLY_EPOCH_CHANGE);
        rig.feed(7, 0, WG_FG_FLAG_MENU);
        d = Present(7);
        d.swapchain_epoch = 2;
        d.queue_epoch = 5;
        rig.proxy.present(&d, &r);
        CHECK(r.status == WG_FG_PRESENT_REAL_ONLY && r.real_only == WG_FG_REAL_ONLY_MENU && r.reason == 0);
    }
    // executor missing / refusing
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        rig.proxy.set_executor(nullptr);
        rig.feed(1);
        CHECK(rig.present(1).reason == WG_FG_REASON_NO_EXECUTOR);
        WgFgExecutorV1 e = rig.exec.table();
        e.struct_size = 8;
        rig.proxy.set_executor(&e);
        rig.feed(2);
        CHECK(rig.present(2).reason == WG_FG_REASON_NO_EXECUTOR);
        e = rig.exec.table();
        rig.proxy.set_executor(&e);
        rig.exec.rc = 22;
        rig.feed(3);
        CHECK(rig.present(3).reason == WG_FG_REASON_EXECUTOR_REFUSED);
        CHECK(rig.log_lines("executor refused the frame") == 1);
    }
    // bad present descriptors are refused without consuming an interval
    {
        Rig rig(QualifiedConfig());
        fresh(rig);
        WgFgPresentResult r = {};
        CHECK(rig.proxy.present(nullptr, &r) == WG_E_MISSING_INPUT_PARAMETER && r.reason == WG_FG_REASON_NULL_ARGUMENT);
        WgFgPresentDesc d = Present(1);
        d.struct_size = 8;
        CHECK(rig.proxy.present(&d, &r) == WG_E_INVALID_PARAMETER && r.reason == WG_FG_REASON_STRUCT_SIZE);
        d = Present(1);
        d.abi_version = 0;
        CHECK(rig.proxy.present(&d, &r) == WG_E_INVALID_PARAMETER && r.reason == WG_FG_REASON_ABI_VERSION);
        CHECK(r.base_presents == 0);
        rig.feed(1);
        CHECK(rig.present(1).status == WG_FG_PRESENT_GENERATE);
        // malformed tags and constants are not observed (and still forwarded)
        WgFgTag small = Tag(kNtMotion, Res(1), 2);
        small.struct_size = 4;
        rig.proxy.tag(&small, 1, &small);
        WgFgConstants cs = Consts(2);
        cs.abi_version = 7;
        rig.proxy.constants(&cs, &cs);
        CHECK(rig.proxy.frames_kept() == 0);
        CHECK(rig.proxy.tag(nullptr, 0, nullptr) == kChainRc);
    }
}

// ---- entries: null host, double attach, version gate, host provider -------------------------------------------------

void TestEntries() {
    {
        Rig rig(QualifiedConfig(), false);
        CHECK(rig.proxy.attach_interposer(nullptr, 2, 14) == WG_FG_REASON_NULL_HOST);
        WgFgChainV1 t = rig.chain.table();
        t.present = nullptr;
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_NULL_HOST);
        t = rig.chain.table();
        t.struct_size = 16;
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_STRUCT_SIZE);
        t = rig.chain.table();
        t.abi_version = 2;
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_ABI_VERSION);
        t = rig.chain.table();
        CHECK(rig.proxy.attach_interposer(&t, 2, 13) == WG_FG_REASON_VERSION_MISMATCH);
        CHECK(rig.proxy.attach_interposer(&t, 3, 14) == WG_FG_REASON_VERSION_MISMATCH);
        CHECK(rig.proxy.entry_kind() == WG_FG_ENTRY_NONE);
        CHECK(rig.log_lines("interface version differs from the binding") == 2);
        // calls with no entry attached reach nothing and claim nothing
        CHECK(rig.proxy.is_supported(kFeatSr, nullptr, nullptr) == WG_E_NOT_INITIALISED);
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_NONE);
        CHECK(rig.proxy.entry_kind() == WG_FG_ENTRY_INTERPOSER);
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_DOUBLE_ATTACH);
        FakeHost host;
        WgFgHostV1 ht = host.table();
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_DOUBLE_ATTACH && host.registers == 0);
        rig.proxy.detach();
        CHECK(rig.proxy.entry_kind() == WG_FG_ENTRY_NONE);
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_NONE && host.registers == 1);
    }
    {
        // an unqualified binding: no version gate (nothing will be taken), everything forwards
        FgProxyConfig cfg = QualifiedConfig();
        cfg.binding.qualified = false;
        Rig rig(cfg, false);
        WgFgChainV1 t = rig.chain.table();
        CHECK(rig.proxy.attach_interposer(&t, 9, 9) == WG_FG_REASON_NONE);
        CHECK(DriveFeature(rig, kFeatFg) == 4);
    }
    {
        // host provider entry
        Rig rig(QualifiedConfig(), false);
        CHECK(rig.proxy.register_with_host(nullptr) == WG_FG_REASON_NULL_HOST);
        FakeHost host;
        WgFgHostV1 ht = host.table();
        ht.register_provider = nullptr;
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_NULL_HOST);
        ht = host.table();
        ht.struct_size = 8;
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_STRUCT_SIZE);
        ht = host.table();
        host.register_rc = 1;
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_HOST_REFUSED && host.registers == 1);
        CHECK(rig.proxy.entry_kind() == WG_FG_ENTRY_NONE && rig.log_lines("the host refused the registration") == 1);
        host.register_rc = 0;
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_NONE && host.provider == rig.proxy.provider());
        CHECK(host.provider->struct_size == sizeof(WgFgProviderV1) && host.provider->abi_version == 1);
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_DOUBLE_ATTACH && host.registers == 2);
        WgFgChainV1 t = rig.chain.table();
        CHECK(rig.proxy.attach_interposer(&t, 2, 14) == WG_FG_REASON_DOUBLE_ATTACH);

        // not taken yet (no adapter): the provider refuses FG so the host keeps its own path
        const WgFgProviderV1* pv = host.provider;
        WgFgOptions o = Options(3, WG_FG_PRESET_C);
        CHECK(pv->set_options(pv->self, 1, &o) == WG_E_FEATURE_NOT_SUPPORTED);
        WgFgState st = {};
        st.struct_size = sizeof(st);
        CHECK(pv->get_state(pv->self, &st) == WG_E_FEATURE_NOT_SUPPORTED);
        WgFgPresentDesc d = Present(1, WG_FG_OWNER_HOST_WRAPPER);
        WgFgPresentResult r = {};
        CHECK(pv->present(pv->self, &d, &r) == WG_OK && r.status == WG_FG_PRESENT_FORWARDED);
        WgFgCapability cap = {};
        cap.struct_size = sizeof(cap);
        CHECK(pv->get_capability(pv->self, &cap) == WG_OK && cap.supported == 0 &&
              cap.reason == WG_FG_REASON_UNSUPPORTED_ADAPTER && cap.max_generated == 0);
        cap.struct_size = 4;
        CHECK(pv->get_capability(pv->self, &cap) == WG_E_INVALID_PARAMETER);
        CHECK(pv->get_capability(nullptr, &cap) == WG_E_MISSING_INPUT_PARAMETER);

        // taken: the whole frame through the provider table
        rig.proxy.set_adapter(Rdna4());
        rig.proxy.set_runtime_facts(true, true);
        WgFgExecutorV1 e = rig.exec.table();
        rig.proxy.set_executor(&e);
        cap.struct_size = sizeof(cap);
        CHECK(pv->get_capability(pv->self, &cap) == WG_OK && cap.supported == 1 && cap.max_generated == 5);
        CHECK(pv->set_options(pv->self, 1, &o) == WG_OK);
        CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_HOST_WRAPPER) == WG_FG_REASON_NONE);
        WgFgTag tags[2] = {Tag(kNtMotion, Res(1), 2), Tag(kNtDepth, Res(2), 2)};
        CHECK(pv->tag(pv->self, tags, 2) == WG_OK);
        WgFgConstants c = Consts(2);
        CHECK(pv->constants(pv->self, &c) == WG_OK);
        d = Present(2, WG_FG_OWNER_HOST_WRAPPER);
        CHECK(pv->present(pv->self, &d, &r) == WG_OK && r.status == WG_FG_PRESENT_GENERATE);
        CHECK(rig.exec.last.generated_count == 3 && rig.exec.last.preset == WG_FG_PRESET_C);
        CHECK(pv->get_state(pv->self, &st) == WG_OK && st.generate_requests == 1 && st.base_presents == 1);
        CHECK(rig.chain.tags == 0 && rig.chain.presents == 0);  // no interface link on this entry
        rig.proxy.detach();
        CHECK(host.unregisters == 1 && host.provider == nullptr);
        CHECK(rig.owners.owner_of(kChain) == WG_FG_OWNER_NONE);
    }
    {
        // install switch off: nothing is registered with the host at all
        FgProxyConfig cfg = QualifiedConfig();
        cfg.settings.enabled = false;
        Rig rig(cfg, false);
        FakeHost host;
        WgFgHostV1 ht = host.table();
        CHECK(rig.proxy.register_with_host(&ht) == WG_FG_REASON_INSTALL_OFF);
        CHECK(host.registers == 0 && host.provider == nullptr && rig.proxy.entry_kind() == WG_FG_ENTRY_NONE);
    }
    {
        // a hand-built invalid settings value keeps FG off
        FgProxyConfig cfg = QualifiedConfig();
        cfg.settings.multiplier = 9;
        Rig rig(cfg);
        CHECK(rig.proxy.route_reason() == WG_FG_REASON_INSTALL_OFF);
        CHECK(rig.log_lines("settings not valid") == 1);
    }
}

#if defined(WG_FG_HAVE_RUNTIME_CONTRACT)
// The adapter's packet and session through the bridge into the runtime contract, checked by the runtime's own rules.
void TestRuntimeContractBridge() {
    Rig rig(QualifiedConfig());
    WgFgOptions o = Options(5, WG_FG_PRESET_C);
    CHECK(rig.proxy.set_options(kFeatFg, 1, &o, nullptr) == WG_OK);
    CHECK(rig.proxy.claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
    static int device = 0, queue = 0;
    rig.proxy.set_device(&device, &queue);
    rig.feed(3, 0x5);
    CHECK(rig.present(3).status == WG_FG_PRESENT_GENERATE);
    const wgfg::FgCreateDesc create = ToRuntime(rig.exec.session);
    const wgfg::FgFramePacket packet = ToRuntime(rig.exec.last);
    CHECK(wgfg::FgValidateCreate(create) == wgfg::FgRefusal::None);
    CHECK(wgfg::FgValidatePacketShape(create, packet) == wgfg::FgRefusal::None);
    CHECK(packet.generated_count == 5 && packet.preset == wgfg::FgPreset::C && packet.optional_present == 0x5);
    CHECK(packet.motion.resource == Res(1) && packet.optional[2].resource == Res(5));
    CHECK(create.max_generated == 5 && create.presenter_mode == 0);
    CHECK(kFgModesQualified == wgfg::kFgModesQualified);
#if !WG_FG_MODES_QUALIFIED
    CHECK(!wgfg::kFgModesQualified);
#endif
}
#endif

}  // namespace

int main() {
    TestSettingsDefaultsAndParse();
    TestSettingsRoundTrip();
    TestDefaultOffForwardsEverything();
    TestReleaseGateForwardsEverything();
    TestRouteFactOrder();
    TestFeatureScopedRouting();
    TestSinglePresentOwner();
    TestMultiFrameAndPresets();
    TestPacketAssembly();
    TestAssemblyFailures();
    TestEntries();
#if defined(WG_FG_HAVE_RUNTIME_CONTRACT)
    TestRuntimeContractBridge();
    std::printf("runtime contract bridge: checked\n");
#else
    std::printf("runtime contract bridge: not built (clean/fg/runtime/fg_contract.h not present)\n");
#endif
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
