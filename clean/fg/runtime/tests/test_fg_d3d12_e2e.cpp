// SPDX-License-Identifier: MIT
// Frame generation (issue #96, F7b): tests of the D3D12 executor (fg_d3d12_executor.h) and its CPU side
// (fg_exec_plan.h). Every value is synthetic. Built with CI's flags (/W4 /WX /permissive-).
//
//   test_fg_d3d12_e2e                       CPU checks (ctest fg_d3d12_e2e): geometry, regions, thresholds and the
//                                           float16 rounding against values of the CPU reference, the logistic table's
//                                           digest, the schedule (once per game frame / per index / last index, history
//                                           resets), every executor refusal that needs no GPU, and a session whose
//                                           executor refuses (no presenter is created, nothing falls back)
//   test_fg_d3d12_e2e --gpu --dxil <dir>    adds the headless GPU leg (no window): a synthetic game drives the game
//                                           adapter core, the FG session and this executor on the D3D12 device with
//                                           the composition back end (unbound surface) and the DXGI back end
//                                           (composition swap chain): preset C 2x and 6x, preset A 2x, a menu frame,
//                                           and a broken kernel folder that must be refused
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "fg_d3d12_executor.h"
#include "fg_exec_plan.h"
#include "fg_session.h"
#include "../../converter/src/sha256.h"
#include "../presenter/fg_dxgi_presenter.h"
#include "../presenter/fg_win_comp_presenter.h"
#include "../proxy/fg_proxy_core.h"

using Microsoft::WRL::ComPtr;
using namespace wgfg;

namespace {

int g_checks = 0, g_failed = 0;
const char* g_case = "";

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            ++g_failed;                                                               \
            std::printf("FAIL [%s] %s:%d: %s\n", g_case, __FILE__, __LINE__, #cond);  \
        }                                                                             \
    } while (0)

std::vector<std::string> OpNames(const std::vector<FgExecOp>& ops) {
    std::vector<std::string> v;
    for (const FgExecOp& o : ops) v.emplace_back(FgOpName(o.op));
    return v;
}
bool SameNames(const std::vector<FgExecOp>& ops, std::initializer_list<const char*> want) {
    const std::vector<std::string> got = OpNames(ops);
    if (got.size() != want.size()) return false;
    size_t i = 0;
    for (const char* w : want)
        if (got[i++] != w) return false;
    return true;
}

// ---- CPU checks ------------------------------------------------------------------------------------------------------

void TestGeometry() {
    g_case = "geometry";
    // rows from clean/fg/reference/pipeline.py Geometry and stages_full.py (net2_k, padded, n_levels)
    const uint32_t rows[4][17] = {
        {160, 90, 2, 80, 45, 64, 32, 12, 5, 2, 80, 45, 9, 4, 1, 160, 96},
        {320, 180, 2, 160, 90, 96, 64, 8, 9, 2, 160, 90, 9, 4, 1, 320, 192},
        {1920, 1080, 2, 960, 540, 480, 288, 0, 9, 2, 960, 540, 11, 8, 4, 1920, 1088},
        {3840, 2160, 4, 960, 540, 480, 288, 0, 9, 2, 1920, 1080, 12, 8, 8, 3840, 2176},
    };
    const uint32_t out_levels[4] = {9, 10, 12, 12};
    for (int i = 0; i < 4; ++i) {
        const uint32_t* r = rows[i];
        FgExecGeometry g;
        std::string why;
        CHECK(FgExecMakeGeometry(r[0], r[1], &g, &why));
        CHECK(g.d == r[2] && g.wd == r[3] && g.hd == r[4] && g.W0 == r[5] && g.H0 == r[6] && g.pad_l == r[7] &&
              g.pad_t == r[8] && g.dp == r[9] && g.w1 == r[10] && g.h1 == r[11]);
        CHECK(g.lw.size() == r[12] && g.top == r[13] && g.k == r[14] && g.Wp == r[15] && g.Hp == r[16]);
        CHECK(g.out_levels == out_levels[i] && g.out_coarsest == (out_levels[i] > 9 ? 8u : 4u));
        CHECK(g.W2 == g.Wp / g.k && g.H2 == g.Hp / g.k);
    }
    FgExecGeometry g;
    std::string why;
    CHECK(!FgExecMakeGeometry(3, 90, &g, &why) && !why.empty());
    CHECK(!FgExecMakeGeometry(8193, 90, &g, &why));
    CHECK(!FgExecMakeGeometry(320, 7681, &g, &why));
}

void TestRegionsAndThresholds() {
    g_case = "regions";
    // stages_common.py Region(...).floats() as float bits
    const uint32_t in[3][6] = {{164, 94, 2, 2, 160, 90}, {320, 180, 0, 0, 320, 180}, {1000, 700, 13, 7, 961, 541}};
    const uint32_t want[3][10] = {
        {0x43200000, 0x42b40000, 0x40000000, 0x40000000, 0x3bc7ce0c, 0x3c2e4c41, 0x3c79c18f, 0x3f7c18f9, 0x3cd9df52,
         0x3f793104},
        {0x43a00000, 0x43340000, 0x0, 0x0, 0x3b4ccccd, 0x3bb60b61, 0x3acccccd, 0x3f7f9999, 0x3b360b61, 0x3f7f49f5},
        {0x44704000, 0x44074000, 0x41500000, 0x40e00000, 0x3a83126f, 0x3abb3ee7, 0x3c5d2f1c, 0x3f79374c, 0x3c2f8af9,
         0x3f483a83},
    };
    for (int i = 0; i < 3; ++i) {
        FgExecRegion r;
        CHECK(FgExecMakeRegion(in[i][0], in[i][1], in[i][2], in[i][3], in[i][4], in[i][5], &r));
        uint32_t a[10];
        r.Args(a);
        bool same = true;
        for (int j = 0; j < 10; ++j) same = same && a[j] == want[i][j];
        CHECK(same);
    }
    FgExecRegion r;
    CHECK(!FgExecMakeRegion(100, 100, 50, 0, 51, 10, &r));
    CHECK(!FgExecMakeRegion(100, 100, 0, 0, 0, 10, &r));
    float t1 = 0, t2 = 0;
    FgExecThresholds(1080, &t1, &t2);
    CHECK(FgExecFloatBits(t1) == 0x44a20000u && FgExecFloatBits(t2) == 0x42e947aeu);
}

void TestHalf() {
    g_case = "float16";
    bool round_trip = true;
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7C00u) == 0x7C00u && (h & 0x3FFu)) continue;  // NaN payloads
        round_trip = round_trip && FgExecHalfBits(FgExecHalfToFloat(static_cast<uint16_t>(h))) == h;
    }
    CHECK(round_trip);
    CHECK(FgExecHalfBits(1.0f + std::ldexp(1.0f, -11)) == 0x3C00);      // tie to even
    CHECK(FgExecHalfBits(1.0f + 3.0f * std::ldexp(1.0f, -11)) == 0x3C02);
    CHECK(FgExecHalfBits(65520.0f) == 0x7C00 && FgExecHalfBits(65519.0f) == 0x7BFF);
    CHECK(FgExecHalfBits(std::ldexp(1.0f, -25)) == 0x0000);              // tie to even (zero)
    CHECK(FgExecHalfBits(3.0f * std::ldexp(1.0f, -26)) == 0x0001);
    CHECK(FgExecHalfBits(-0.0f) == 0x8000 && (FgExecHalfBits(NAN) & 0x7C00) == 0x7C00);
}

void TestSigTable() {
    g_case = "logistic table";
    std::vector<float> t;
    FgExecSigTable(&t);
    CHECK(t.size() == 65536);
    const wgc::Digest d = wgc::Sha256::hash(t.data(), t.size() * 4);
    // digest of clean/fg/reference/stages_full.py sig_table() as little-endian float32
    const uint8_t want[32] = {0x09, 0x90, 0x7d, 0x7f, 0xee, 0x13, 0x7a, 0xe4, 0x69, 0x92, 0x9d, 0xb6, 0x89, 0x55, 0x82, 0x5b,
                              0xff, 0xf2, 0x15, 0xb4, 0x80, 0xe8, 0x2f, 0xed, 0xed, 0xe3, 0x52, 0x11, 0xed, 0x1a, 0x67, 0xb2};
    CHECK(std::memcmp(d.data(), want, 32) == 0);
}

void TestSchedule() {
    g_case = "schedule";
    FgExecSettings st;
    st.menu_detection = true;
    FgExecGeometry g;
    std::string why;
    FgExecMakeGeometry(160, 90, &g, &why);
    FgExecHost h;
    h.Configure(st, g);
    std::vector<FgExecOp> ops;
    // first game frame: no history (clears, detection with hysteresis off, base 1, the quality image)
    h.BeginFrame(true, false, true, false, false, true, &ops);
    CHECK(SameNames(ops, {"clear", "clear", "clear", "clear", "detect", "downsample", "mvec", "fill64", "p2c_scatter",
                          "p2c_unpack", "p2c_inpaint", "p2c_pull", "p2c_push", "hole_mask", "quality_init"}));
    CHECK(ops[4].detect_kind == 1 && ops[4].hysteresis_off && ops[4].base == 1);
    CHECK(ops[0].role == kRoleCp && ops[0].period == 4 && ops[0].value[3] == 0x3C00 && ops[2].role == kRoleZp &&
          ops[2].period == 2 && ops[3].value[0] == 0x3800);
    CHECK(!h.Frame().history && h.Frame().flags.lite2 && h.Frame().r3_case == 'H');
    ops.clear();
    h.IndexOps(0.5f, true, false, false, &ops);
    CHECK(SameNames(ops, {"final_copy"}));
    std::vector<uint32_t> pairs;
    h.EndFrame(&pairs);
    CHECK(pairs.size() == 16);  // six pairs, Lc / Lp (Lite flag 2), Um / Ump (case H)
    // second game frame: history, net1
    ops.clear();
    h.BeginFrame(true, false, true, false, false, false, &ops);
    CHECK(SameNames(ops, {"detect", "downsample", "mvec", "fill64", "p2c_scatter", "p2c_unpack", "p2c_inpaint",
                          "p2c_pull", "p2c_push", "hole_mask", "net1_input", "net1", "net1_output"}));
    CHECK(ops[0].base == 0 && !ops[0].hysteresis_off && ops.back().ui_forcing);
    // six indices of a 6x frame share it; each index runs its own scatter / weights / blend / menu output
    for (uint32_t i = 1; i <= 5; ++i) {
        ops.clear();
        h.IndexOps(FgIndexTime(i, 5), true, false, false, &ops);
        CHECK(SameNames(ops, {"fill64", "fill64", "interm", "weights", "lite_blend", "menu_output"}));
        CHECK(FgExecFloatBits(ops[2].t) == FgExecFloatBits(FgIndexTime(i, 5)) && ops[4].flag2 && !ops[4].flag1);
    }
    CHECK(h.InFrame() && h.HistoryFrames() == 1);
    pairs.clear();
    h.EndFrame(&pairs);
    CHECK(h.HistoryFrames() == 2 && !h.InFrame());
    // flags change (hudless gone): history restarts
    ops.clear();
    h.BeginFrame(true, false, false, false, false, false, &ops);
    CHECK(!h.Frame().history && OpNames(ops).front() == "clear" && OpNames(ops).back() == "quality_init");
    CHECK(ops[4].op == FgOp::SetDisable && ops[4].base == 1);  // no detector without hudless / UI alpha
    pairs.clear();
    h.EndFrame(&pairs);
    // the Full path (preset A): the same shared stages, then the Full blend, net2 and the output pyramid per index
    ops.clear();
    h.BeginFrame(false, false, false, false, false, false, &ops);
    CHECK(h.Frame().history && !h.Frame().flags.lite2);
    ops.clear();
    h.IndexOps(0.5f, false, false, false, &ops);
    CHECK(SameNames(ops, {"fill64", "fill64", "interm", "weights", "full_blend", "net2", "out_pyramid"}));
    // preset B with hudless and UI alpha: UI recomposition (R3 case U)
    pairs.clear();
    h.EndFrame(&pairs);
    ops.clear();
    h.BeginFrame(false, true, true, true, false, false, &ops);
    CHECK(h.Frame().flags.ui_recomp && h.Frame().r3_case == 'U' && !h.Frame().history);
    CHECK(ops[4].op == FgOp::Clear && ops[5].op == FgOp::Clear && ops[6].op == FgOp::Detect && ops[6].detect_kind == 0);
}

struct CountingSource final : public IFgPresenterSource {
    int calls = 0;
    FgStatus CreatePresenter(const FgCreateDesc&, std::unique_ptr<IFgPresenter>* out, std::string* why) override {
        ++calls;
        if (out) out->reset();
        if (why) *why = "test source";
        return FgStatus::Unsupported;
    }
};
FgTicks g_fake_now = 1000000;
FgTicks FakeClock(void*) { return g_fake_now += 1000; }

FgCreateDesc CpuCreate(uint32_t w, uint32_t h) {
    FgCreateDesc c;
    static char fake[4];
    c.device = &fake[0];  // synthetic: distinct addresses, never dereferenced (the checks come first)
    c.queue = &fake[1];
    c.adapter_luid.low = 7;
    c.swapchain_id = 0x51;
    c.window_id = 0x52;
    c.queue_epoch = 1;
    c.swapchain_epoch = 1;
    c.viewport_id = 1;
    c.output_width = w;
    c.output_height = h;
    c.output_format = 10;
    c.render_min_width = c.render_max_width = w;
    c.render_min_height = c.render_max_height = h;
    c.preset = FgPreset::C;
    c.max_generated = 5;
    for (uint32_t i = 0; i < kFgOptCount; ++i) c.optional[i] = FgOptionalDecl::May;
    return c;
}

void WriteFile(const std::string& path, const char* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(data, static_cast<std::streamsize>(n));
}

void TestRefusals(const FgExecWeights& weights) {
    g_case = "refusals";
    FgD3d12ExecutorDesc d;
    d.dxil_dir = "fg_f7b_no_such_folder";
    {
        FgD3d12Executor e(d);  // no weights
        CHECK(e.Configure(CpuCreate(320, 180), 3) == FgExecResult::Refused &&
              e.LastRefusal() == FgExecRefusal::WeightsMissing);
        FgSharedResult sr;
        FgFramePacket p;
        FgSharedDispatchDesc sd;
        sd.packet = &p;
        CHECK(e.RecordSharedGameFrame(sd, &sr) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::NotConfigured);
        FgReadyFence rf;
        CHECK(e.RecordGeneratedIndex(FgIndexDispatchDesc(), &rf) == FgExecResult::Refused);
        CHECK(e.RecordMenuCopy(FgCopyDispatchDesc(), &rf) == FgExecResult::Refused);
        CHECK(e.RecordRealEndpoint(FgRealDispatchDesc(), &rf) == FgExecResult::Refused);
        CHECK(e.LastComputeValue() == 0 && e.CompletedComputeValue() == 0);
    }
    d.weights = &weights;
    {
        FgD3d12Executor e(d);
        CHECK(e.Configure(CpuCreate(2, 2), 3) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::SizeUnsupported);
        CHECK(e.Configure(CpuCreate(8200, 90), 3) == FgExecResult::Refused &&
              e.LastRefusal() == FgExecRefusal::SizeUnsupported);
        FgCreateDesc c = CpuCreate(320, 180);
        c.device = nullptr;
        CHECK(e.Configure(c, 3) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::NullDevice);
        c = CpuCreate(320, 180);
        c.queue = nullptr;
        CHECK(e.Configure(c, 3) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::NullQueue);
        CHECK(e.Configure(CpuCreate(320, 180), 3) == FgExecResult::Refused &&
              e.LastRefusal() == FgExecRefusal::DxilDirMissing);
        c = CpuCreate(320, 180);
        c.optional[kFgOptDistortion] = FgOptionalDecl::Always;
        CHECK(e.Configure(c, 3) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::DistortionUnsupported);
    }
    // a kernel folder with a broken first kernel, and an empty one
    CreateDirectoryA("fg_f7b_scratch_broken", nullptr);
    CreateDirectoryA("fg_f7b_scratch_empty", nullptr);
    const char junk[64] = "not a kernel container";
    WriteFile("fg_f7b_scratch_broken\\fg_lite_fill16.dxil", junk, sizeof(junk));
    {
        d.dxil_dir = "fg_f7b_scratch_broken";
        FgD3d12Executor e(d);
        CHECK(e.Configure(CpuCreate(320, 180), 3) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::DxilInvalid);
        d.dxil_dir = "fg_f7b_scratch_empty";
        FgD3d12Executor e2(d);
        CHECK(e2.Configure(CpuCreate(320, 180), 3) == FgExecResult::Refused &&
              e2.LastRefusal() == FgExecRefusal::DxilMissing);
        CHECK(e2.LastWhy().find("fg_lite_fill16") != std::string::npos);
    }
    // the Full path without net2's weights
    {
        FgExecWeights no2 = weights;
        no2.has_net2 = false;
        FgD3d12ExecutorDesc d2 = d;
        d2.weights = &no2;
        FgD3d12Executor e(d2);
        CHECK(e.Configure(CpuCreate(320, 180), 3) == FgExecResult::Refused && e.LastRefusal() == FgExecRefusal::WeightsMissing);
        d2.enable_full = false;
        FgD3d12Executor e2(d2);
        CHECK(e2.Configure(CpuCreate(320, 180), 3) == FgExecResult::Refused &&
              e2.LastRefusal() == FgExecRefusal::DxilMissing);  // gets past the weights: only the kernels are missing
    }
    // a model report that did not pass
    FgExecWeights w;
    std::string why;
    const uint8_t b[4] = {};
    CHECK(!FgExecWeightsFromModel(b, sizeof(b), FgModelReport(), &w, &why) && !why.empty());
    // a session whose executor refuses: the create fails before any presenter exists
    {
        d.dxil_dir = "fg_f7b_no_such_folder";
        FgD3d12Executor e(d);
        CountingSource src;
        FgExecutorPresenterSource bound(&src, &e);
        FgSession s;
        FgSessionDesc sd;
        sd.create = CpuCreate(320, 180);
        sd.executor = &e;
        sd.presenter_source = &bound;
        sd.clock = &FakeClock;
        const FgSessionResult r = s.Create(sd);
        CHECK(r.error == FgSessionError::ExecutorConfigure && r.executor == FgExecResult::Refused);
        CHECK(src.calls == 0 && s.State() == FgSessionState::None);
    }
}

// ---- GPU leg ---------------------------------------------------------------------------------------------------------

constexpr uint32_t kW = 320, kH = 180;
constexpr int32_t kChainRc = 55;
constexpr uint64_t kChain = 0xF7B0ull;
constexpr uint32_t kNtBackbuffer = 900, kNtMotion = 901, kNtDepth = 902, kNtHudless = 903, kNtUiAlpha = 904,
                   kNtDistortion = 905, kNtNoWarp = 906;

struct FakeChain {
    int presents = 0;
    static int32_t IsSupported(void*, uint32_t, const void*) { return kChainRc; }
    static int32_t GetFunction(void*, uint32_t, const char*, void** out) {
        if (out) *out = nullptr;
        return kChainRc;
    }
    static int32_t SetOptions(void*, uint32_t, uint32_t, const void*) { return kChainRc; }
    static int32_t GetState(void*, uint32_t, uint32_t, void*) { return kChainRc; }
    static int32_t Tag(void*, const void*, uint32_t) { return kChainRc; }
    static int32_t Constants(void*, const void*) { return kChainRc; }
    static int32_t Present(void* s, const WgFgPresentDesc*) {
        static_cast<FakeChain*>(s)->presents++;
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
int g_fn_a = 0, g_fn_b = 0;

FgTicks RealClock(void*) { return FgInterruptNow(); }
bool g_debug_layer = false;
int g_log_lines = 0;
void PrintLog(void*, const char* line) {
    if (g_log_lines++ < 40) std::printf("      log: %s\n", line);
}

struct Gpu {
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> info;  // debug layer only
    uint64_t value = 0;
    HANDLE event = nullptr;
    LUID luid = {};
    // Errors / corruption messages of the debug layer since the last call (printed); -1 without the layer.
    int64_t DebugErrors() {
        if (!info) return -1;
        int64_t errors = 0;
        const UINT64 n = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < n; ++i) {
            SIZE_T len = 0;
            info->GetMessage(i, nullptr, &len);
            std::vector<char> buf(len);
            D3D12_MESSAGE* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
            if (FAILED(info->GetMessage(i, m, &len))) continue;
            if (m->Severity == D3D12_MESSAGE_SEVERITY_ERROR || m->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
                if (errors < 8) std::printf("      debug layer: %s\n", m->pDescription);
                ++errors;
            }
        }
        info->ClearStoredMessages();
        return errors;
    }
    std::string name;
    ~Gpu() {
        if (event) CloseHandle(event);
    }
    bool Init() {
        ComPtr<IDXGIFactory6> f;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)))) return false;
        for (UINT i = 0; f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) !=
                         DXGI_ERROR_NOT_FOUND;
             ++i) {
            DXGI_ADAPTER_DESC1 d;
            adapter->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                luid = d.AdapterLuid;
                char buf[128] = {};
                WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, buf, sizeof(buf) - 1, nullptr, nullptr);
                name = buf;
                break;
            }
            adapter.Reset();
        }
        if (g_debug_layer) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
        }
        if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
            return false;
        if (g_debug_layer) device.As(&info);
        D3D12_COMMAND_QUEUE_DESC q = {};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&alloc))) ||
            FAILED(device->CreateCommandList(0, q.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
            FAILED(list->Close()) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return false;
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return event != nullptr;
    }
    bool Wait() {
        queue->Signal(fence.Get(), ++value);
        if (fence->GetCompletedValue() >= value) return true;
        fence->SetEventOnCompletion(value, event);
        return WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    }
};

uint32_t Hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
float Noise(uint32_t frame, uint32_t i, uint32_t c) {
    return static_cast<float>(Hash(frame * 0x9E3779B1u + i * 4u + c) & 0xFFFFu) / 65535.0f - 0.5f;
}

// The synthetic game's textures and one frame of content (a panning background, a moving block, a UI panel, an
// optional full-screen menu overlay, a depth plane, motion in pixels, the translation matrices).
struct Game {
    Gpu* gpu = nullptr;
    ComPtr<ID3D12Resource> colour, hudless, ui, depth, motion;
    DXGI_FORMAT colour_format = DXGI_FORMAT_R16G16B16A16_FLOAT;  // the chain's format (B8G8R8A8 or RGBA16F)
    ComPtr<ID3D12Resource> upload;
    D3D12_RESOURCE_STATES colour_state = D3D12_RESOURCE_STATE_COPY_SOURCE;  // colour left in a copy state on purpose
    D3D12_RESOURCE_STATES other_state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    float clip_to_prev[16] = {}, prev_to_clip[16] = {};

    ComPtr<ID3D12Resource> Tex(DXGI_FORMAT f) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = kW;
        rd.Height = kH;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = f;
        rd.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> r;
        gpu->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&r));
        return r;
    }
    bool Init(Gpu* g, DXGI_FORMAT cf) {
        colour_format = cf;
        gpu = g;
        colour = Tex(cf);
        hudless = Tex(DXGI_FORMAT_R16G16B16A16_FLOAT);
        ui = Tex(DXGI_FORMAT_R16G16B16A16_FLOAT);
        depth = Tex(DXGI_FORMAT_R32_FLOAT);
        motion = Tex(DXGI_FORMAT_R16G16_FLOAT);
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 5ull * 3072 * kH;  // five textures, rows of 3072 bytes (320 x 8 rounded up to 256)
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        gpu->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                             IID_PPV_ARGS(&upload));
        for (int i = 0; i < 16; ++i) clip_to_prev[i] = prev_to_clip[i] = 0.0f;
        clip_to_prev[0] = clip_to_prev[5] = clip_to_prev[15] = 1.0f;
        clip_to_prev[11] = 0.0625f;
        std::memcpy(prev_to_clip, clip_to_prev, sizeof(prev_to_clip));
        const float tx = 12.0f / static_cast<float>(kW);
        clip_to_prev[12] = tx;
        prev_to_clip[12] = -tx;
        return colour && hudless && ui && depth && motion && upload;
    }
    // Writes frame `f` (menu: the overlay) into the textures and waits for the copies.
    bool Upload(uint32_t f, bool menu, bool first) {
        uint8_t* base = nullptr;
        if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&base)))) return false;
        const uint32_t p8 = 3072, p4 = 1280, region = p8 * kH;  // row pitches: 8-byte and 4-byte texels
        uint16_t* col = reinterpret_cast<uint16_t*>(base);
        uint16_t* hud = reinterpret_cast<uint16_t*>(base + region);
        uint16_t* uim = reinterpret_cast<uint16_t*>(base + 2 * region);
        float* dep = reinterpret_cast<float*>(base + 3 * region);
        uint16_t* mot = reinterpret_cast<uint16_t*>(base + 4 * region);
        const float pan = 6.0f * static_cast<float>(f);
        const uint32_t bx0 = static_cast<uint32_t>(kW * 0.3f) + 2 * f, by0 = static_cast<uint32_t>(kH * 0.35f) + f;
        const uint32_t bw = kW / 5, bh = kH / 4;
        for (uint32_t y = 0; y < kH; ++y)
            for (uint32_t x = 0; x < kW; ++x) {
                const uint32_t i = y * kW + x;
                const float fx = static_cast<float>(x), fy = static_cast<float>(y);
                float rgb[3] = {0.5f + 0.4f * std::sin((fx + pan) * 0.11f) * std::cos(fy * 0.07f),
                                0.5f + 0.3f * std::cos((fx + pan) * 0.05f + fy * 0.09f),
                                0.4f + 0.3f * std::sin((fy - pan) * 0.13f)};
                for (int c = 0; c < 3; ++c) rgb[c] += 0.1f * Noise(f, i, c);
                const bool block = x >= bx0 && x < bx0 + bw && y >= by0 && y < by0 + bh;
                if (block) {
                    rgb[0] = 0.9f + 0.06f * Noise(f, i, 0);
                    rgb[1] = 0.2f;
                    rgb[2] = 0.15f;
                }
                float alpha = 0.0f, uc[3] = {0, 0, 0};
                const bool panel = x >= kW * 0.62f && x < kW * 0.62f + kW / 4 && y >= kH * 0.08f && y < kH * 0.08f + kH / 5;
                if (panel) {
                    alpha = ((x / 3 + y / 2) % 3 == 0) ? 1.0f : 0.75f;
                    uc[0] = alpha == 1.0f ? 1.0f : 0.1f;
                    uc[1] = alpha == 1.0f ? 1.0f : 0.8f;
                    uc[2] = alpha == 1.0f ? 1.0f : 0.9f;
                }
                if (menu && x >= kW * 0.05f && x < kW * 0.95f && y >= kH * 0.03f && y < kH * 0.97f) {
                    alpha = 0.9f;
                    uc[0] = 0.05f;
                    uc[1] = 0.05f;
                    uc[2] = 0.2f;
                }
                const bool c8 = colour_format == DXGI_FORMAT_B8G8R8A8_UNORM;
                uint16_t* cp = col + (y * (c8 ? p4 : p8) / 2) + (c8 ? 0 : 4 * x);
                uint8_t* cb = reinterpret_cast<uint8_t*>(col) + y * p4 + 4 * x;
                uint16_t* hp = hud + (y * p8 / 2) + 4 * x;
                uint16_t* up = uim + (y * p8 / 2) + 4 * x;
                for (int c = 0; c < 3; ++c) {
                    hp[c] = FgExecHalfBits(rgb[c]);
                    const float v = uc[c] * alpha + rgb[c] * (1.0f - alpha);
                    if (c8)
                        cb[2 - c] = static_cast<uint8_t>(std::floor(std::fmin(std::fmax(v, 0.0f), 1.0f) * 255.0f + 0.5f));
                    else
                        cp[c] = FgExecHalfBits(v);
                    up[c] = FgExecHalfBits(uc[c] * alpha);
                }
                hp[3] = 0x3C00;
                if (c8)
                    cb[3] = 255;
                else
                    cp[3] = 0x3C00;
                up[3] = FgExecHalfBits(alpha);
                const float dplane = 0.3f + 0.3f * fy / kH;
                dep[y * p4 / 4 + x] = 1.0f - (block ? 0.012f + 0.008f * fx / kW : dplane);
                const float mx = block ? -4.0f : 6.0f, my = block ? -2.0f : 0.0f;
                mot[y * p4 / 2 + 2 * x] = FgExecHalfBits(mx + 0.4f * Noise(f, i, 3));
                mot[y * p4 / 2 + 2 * x + 1] = FgExecHalfBits(my + 0.4f * Noise(f, i, 4));
            }
        upload->Unmap(0, nullptr);
        gpu->alloc->Reset();
        gpu->list->Reset(gpu->alloc.Get(), nullptr);
        ID3D12Resource* texs[5] = {colour.Get(), hudless.Get(), ui.Get(), depth.Get(), motion.Get()};
        const DXGI_FORMAT fmts[5] = {colour_format, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                     DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16_FLOAT};
        const uint32_t pitches[5] = {colour_format == DXGI_FORMAT_B8G8R8A8_UNORM ? p4 : p8, p8, p8, p4, p4};
        for (int t = 0; t < 5; ++t) {
            const D3D12_RESOURCE_STATES st = t == 0 ? colour_state : other_state;
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = texs[t];
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = first ? D3D12_RESOURCE_STATE_COMMON : st;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            gpu->list->ResourceBarrier(1, &b);
            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = texs[t];
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = upload.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint.Offset = static_cast<uint64_t>(t) * region;
            src.PlacedFootprint.Footprint = {fmts[t], kW, kH, 1, pitches[t]};
            gpu->list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            b.Transition.StateAfter = st;
            gpu->list->ResourceBarrier(1, &b);
        }
        gpu->list->Close();
        ID3D12CommandList* l[] = {gpu->list.Get()};
        gpu->queue->ExecuteCommandLists(1, l);
        return gpu->Wait();
    }
};

// A test presenter source: the composition back end on an unbound surface, or the DXGI back end on a composition swap
// chain bound to no visual. Nothing is shown.
struct HeadlessSource final : public IFgPresenterSource {
    Gpu* gpu = nullptr;
    bool dxgi = false;
    ComPtr<IDXGISwapChain4> chain;
    std::string last;
    FgStatus CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out, std::string* why) override {
        out->reset();
        if (!dxgi) {
            FgCompPresenterDesc cd;
            cd.device = gpu->device.Get();
            cd.bind_window = false;
            cd.width = desc.output_width;
            cd.height = desc.output_height;
            cd.format = static_cast<DXGI_FORMAT>(desc.output_format);
            cd.variable_refresh = true;
            cd.max_generated = desc.max_generated;
            // an unbound surface is never composed, so no slot ever comes back from the display: the run must fit
            // the back end's 32 slots (the presenter's own headless test shows the same)
            cd.slot_count = 32;
            std::unique_ptr<FgWinCompPresenter> p;
            const FgWinResult r = FgWinCompPresenter::Create(cd, &p);
            last = std::string("composition: ") + FgWinRefusalName(r.refusal);
            if (!r.Ok()) {
                if (why) *why = last;
                return r.status;
            }
            *out = std::move(p);
            return FgStatus::Ok;
        }
        ComPtr<IDXGIFactory2> f;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)))) return FgStatus::Unsupported;
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = desc.output_width;
        sd.Height = desc.output_height;
        sd.Format = static_cast<DXGI_FORMAT>(desc.output_format);
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        ComPtr<IDXGISwapChain1> sc1;
        if (FAILED(f->CreateSwapChainForComposition(gpu->queue.Get(), &sd, nullptr, &sc1)) || FAILED(sc1.As(&chain)))
            return FgStatus::Unsupported;
        chain->SetMaximumFrameLatency(2);
        FgDxgiPresenterDesc dd;
        dd.device = gpu->device.Get();
        dd.queue = gpu->queue.Get();
        dd.swap_chain = chain.Get();
        dd.pacing = FgDxgiPacing::Tearing;
        dd.max_generated = desc.max_generated;
        dd.slot_count = desc.max_generated + 5;
        std::unique_ptr<FgDxgiPresenter> p;
        const FgWinResult r = FgDxgiPresenter::Create(dd, &p);
        last = std::string("dxgi tearing: ") + FgWinRefusalName(r.refusal);
        if (!r.Ok()) {
            if (why) *why = last;
            return r.status;
        }
        *out = std::move(p);
        return FgStatus::Ok;
    }
};

wgfgp::FgProxyConfig QualifiedConfig() {
    wgfgp::FgProxyConfig c;
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
    c.modes_qualified = true;  // the release gate is false; this test exercises the taken route
    return c;
}

uint16_t Unorm8(uint16_t h) {
    float v = FgExecHalfToFloat(h);
    v = std::fmin(std::fmax(v, 0.0f), 1.0f);  // fmin / fmax: NaN gives the other operand (0)
    return static_cast<uint16_t>(std::floor(std::fma(v, 255.0f, 0.5f)));
}

struct RunResult {
    bool ran = false;
    uint64_t frames = 0, generated = 0, copies = 0, real = 0, menu_copies = 0;
};

// One configuration end to end. Returns what happened; every check is inside.
RunResult RunPipeline(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, FgPreset preset, uint32_t n,
                      bool dxgi, DXGI_FORMAT slot_format, int frames, int menu_frame) {
    RunResult rr;
    FgD3d12ExecutorDesc ed;
    ed.dxil_dir = dxil;
    ed.weights = &weights;
    ed.settings.menu_detection = true;
    ed.keep_outputs = true;
    FgD3d12Executor exec(ed);
    HeadlessSource headless;
    headless.gpu = &gpu;
    headless.dxgi = dxgi;
    FgExecutorPresenterSource source(&headless, &exec);
    Game game;
    CHECK(game.Init(&gpu, slot_format));

    FakeChain chain;
    wgfgp::FgPresentOwners owners;
    std::unique_ptr<wgfgp::FgProxy> proxy(new wgfgp::FgProxy(QualifiedConfig(), &owners, &PrintLog, nullptr));
    WgFgChainV1 ct = chain.table();
    proxy->attach_interposer(&ct, 2, 14);
    WgFgAdapterIdentity a = {};
    a.struct_size = sizeof(a);
    a.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    a.vendor_id = 0x1002;
    a.device_id = 0x7550;
    a.luid.low = gpu.luid.LowPart;
    a.luid.high = gpu.luid.HighPart;
    a.rdna4 = 1;
    a.probes_passed = 1;
    proxy->set_adapter(a);
    proxy->set_device(gpu.device.Get(), gpu.queue.Get());
    proxy->set_runtime_facts(true, true);
    proxy->set_feature_function("fgTestSetOptions", &g_fn_a);
    proxy->set_feature_function("fgTestGetState", &g_fn_b);
    WgFgOptions o = {};
    o.struct_size = sizeof(o);
    o.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    o.enable = 1;
    o.preset = static_cast<uint32_t>(preset);
    o.generated_count = n;
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) o.optional_decl[i] = WG_FG_DECL_MAY;
    o.color_space = WG_FG_CS_SDR;
    o.output_width = kW;
    o.output_height = kH;
    o.output_format = static_cast<uint32_t>(slot_format);
    o.render_min_width = o.render_max_width = kW;
    o.render_min_height = o.render_max_height = kH;
    proxy->set_options(WG_FEAT_FRAME_GENERATION, 1, &o, &o);
    proxy->claim_present(kChain, WG_FG_OWNER_NATIVE_ADAPTER);

    FgSession session;
    FgSessionDesc sd;
    FgCreateDesc& c = sd.create;
    c.device = gpu.device.Get();
    c.queue = gpu.queue.Get();
    c.adapter_luid.low = gpu.luid.LowPart;
    c.adapter_luid.high = gpu.luid.HighPart;
    c.swapchain_id = kChain;
    c.window_id = 0x77;
    c.queue_epoch = 1;
    c.swapchain_epoch = 1;
    c.viewport_id = 1;
    c.output_width = kW;
    c.output_height = kH;
    c.output_format = static_cast<uint32_t>(slot_format);
    c.render_min_width = c.render_max_width = kW;
    c.render_min_height = c.render_max_height = kH;
    c.preset = preset;
    c.max_generated = 5;
    for (uint32_t i = 0; i < kFgOptCount; ++i) c.optional[i] = FgOptionalDecl::May;
    sd.executor = &exec;
    sd.presenter_source = &source;
    sd.history_banks = 3;
    sd.clock = &RealClock;
    sd.log = &PrintLog;
    g_log_lines = 0;
    sd.scheduler.display.min_interval = 1;
    sd.scheduler.display.max_interval = 10000000;
    sd.scheduler.max_lead = 2000000;
    sd.scheduler.first_image_budget = 600000;
    const FgSessionResult cr = session.Create(sd);
    std::printf("    create: %s (executor %s: %s; presenter %s)\n", FgSessionErrorName(cr.error),
                FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str(), headless.last.c_str());
    if (cr.error != FgSessionError::None) {
        // a back end the headless system cannot make is an explicit refusal (reported), never a substitute
        CHECK(cr.error == FgSessionError::PresenterSource);
        return rr;
    }
    rr.ran = true;
    WgFgExecutorV1 h = session.ExecutorHandOff();
    proxy->set_executor(&h);

    const FgTicks period = 333333;  // 30 game frames per second
    int errors = 0;
    bool menu_seen = false;
    for (int f = 0; f < frames; ++f) {
        const uint64_t tok = static_cast<uint64_t>(f) + 1;
        const bool menu = f == menu_frame;
        CHECK(game.Upload(static_cast<uint32_t>(f), menu, f == 0));
        WgFgTag tags[4] = {};
        ID3D12Resource* res[4] = {game.motion.Get(), game.depth.Get(), game.hudless.Get(), game.ui.Get()};
        const uint32_t nts[4] = {kNtMotion, kNtDepth, kNtHudless, kNtUiAlpha};
        const uint32_t fm[4] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                DXGI_FORMAT_R16G16B16A16_FLOAT};
        for (int t = 0; t < 4; ++t) {
            tags[t].struct_size = sizeof(WgFgTag);
            tags[t].native_type = nts[t];
            tags[t].texture.resource = res[t];
            tags[t].texture.format = fm[t];
            tags[t].texture.width = kW;
            tags[t].texture.height = kH;
            tags[t].texture.state = static_cast<uint32_t>(game.other_state);
            tags[t].texture.frame_token = tok;
            tags[t].texture.viewport_id = 1;
            tags[t].texture.producer_epoch = 3;
            tags[t].texture.rect = {0, 0, kW, kH};
        }
        proxy->tag(tags, 4, tags);
        WgFgConstants k = {};
        k.struct_size = sizeof(k);
        k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        k.frame_token = tok;
        k.viewport_id = 1;
        std::memcpy(k.camera_current, game.clip_to_prev, sizeof(k.camera_current));
        std::memcpy(k.camera_previous, game.prev_to_clip, sizeof(k.camera_previous));
        k.motion_scale_x = 1.0f;
        k.motion_scale_y = 1.0f;
        k.render_rect = {0, 0, kW, kH};
        k.output_rect = {0, 0, kW, kH};
        proxy->constants(&k, &k);
        WgFgPresentDesc pd = {};
        pd.struct_size = sizeof(pd);
        pd.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        pd.owner = WG_FG_OWNER_NATIVE_ADAPTER;
        pd.viewport_id = 1;
        pd.frame_token = tok;
        pd.swapchain_id = kChain;
        pd.window_id = 0x77;
        pd.queue_epoch = 1;
        pd.swapchain_epoch = 1;
        pd.luid.low = gpu.luid.LowPart;
        pd.luid.high = gpu.luid.HighPart;
        pd.backbuffer.resource = game.colour.Get();
        pd.backbuffer.format = static_cast<uint32_t>(slot_format);
        pd.backbuffer.width = kW;
        pd.backbuffer.height = kH;
        pd.backbuffer.state = static_cast<uint32_t>(game.colour_state);
        pd.backbuffer.frame_token = tok;
        pd.backbuffer.viewport_id = 1;
        pd.backbuffer.rect = {0, 0, kW, kH};
        WgFgPresentResult pr = {};
        FgSessionResult r;
        FgSessionPresent(proxy.get(), &session, &pd, &pr, &r);
        if (r.error != FgSessionError::None) {
            ++errors;
            std::printf("    frame %llu: session %s, executor %s (%s), presenter %s; core status %u reason %u "
                        "real-only %u\n",
                        static_cast<unsigned long long>(tok), FgSessionErrorName(r.error),
                        FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str(), FgStatusName(r.presenter),
                        pr.status, pr.reason, pr.real_only);
        }
        if (menu) menu_seen = r.menu_detected;
        ++rr.frames;
        Sleep(static_cast<DWORD>(period / 10000));
    }
    CHECK(errors == 0);
    CHECK(exec.WaitIdle(5000));
    Sleep(200);
    IFgPresenter* pres = session.Presenter();
    const FgPresentStats st = pres ? pres->Stats() : FgPresentStats();
    const FgSessionCounters& sc = session.Counters();
    const FgExecCounters& ec = exec.Counters();
    std::printf("    session: game %llu, shared %llu, indices %llu, copies %llu, real %llu, menu detections %llu, "
                "menu copy frames %llu, stale %llu\n",
                static_cast<unsigned long long>(sc.game_frames), static_cast<unsigned long long>(sc.shared_records),
                static_cast<unsigned long long>(sc.index_records), static_cast<unsigned long long>(sc.copy_records),
                static_cast<unsigned long long>(sc.real_endpoints), static_cast<unsigned long long>(sc.menu_detections),
                static_cast<unsigned long long>(sc.menu_copy_frames), static_cast<unsigned long long>(sc.stale_cancelled));
    std::printf("    executor: dispatches %llu, submissions %llu, captures %llu, menu waits %llu, arena %llu bytes, "
                "fence %llu/%llu\n",
                static_cast<unsigned long long>(ec.dispatches), static_cast<unsigned long long>(ec.submissions),
                static_cast<unsigned long long>(ec.captures), static_cast<unsigned long long>(ec.menu_waits),
                static_cast<unsigned long long>(exec.ArenaBytes()),
                static_cast<unsigned long long>(exec.CompletedComputeValue()),
                static_cast<unsigned long long>(exec.LastComputeValue()));
    std::printf("    presenter: real queued %llu, generated queued %llu, displayed %llu+%llu, late %llu, pressure %llu, "
                "cancelled %llu, refused calls %llu\n",
                static_cast<unsigned long long>(st.real_queued), static_cast<unsigned long long>(st.generated_queued),
                static_cast<unsigned long long>(st.displayed_real), static_cast<unsigned long long>(st.displayed_generated),
                static_cast<unsigned long long>(st.dropped_late), static_cast<unsigned long long>(st.dropped_pressure),
                static_cast<unsigned long long>(st.cancelled), static_cast<unsigned long long>(st.refused_calls));
    CHECK(sc.shared_records == static_cast<uint64_t>(frames) && ec.shared == static_cast<uint64_t>(frames));
    CHECK(sc.index_records > 0 && ec.indices == sc.index_records);
    CHECK(exec.CompletedComputeValue() == exec.LastComputeValue());
    // every generated / copied image reached the display path or was dropped late by the presenter (explicitly)
    CHECK(st.generated_queued + st.dropped_late >= sc.index_records + sc.copy_records);
    CHECK(st.refused_calls == 0);
    if (preset == FgPreset::C) CHECK(st.generated_queued > 0);  // the Full smoke run may be dropped late
    if (menu_frame >= 0) {
        CHECK(menu_seen && sc.menu_detections >= 1 && sc.menu_copy_frames >= 1);
    }
    // the images the executor wrote, read back
    std::vector<FgExecOutputRecord> outs;
    CHECK(exec.TakeOutputs(&outs));
    uint64_t gen_ok = 0, copy_ok = 0, slot_ok = 0, menu_tok_copies = 0;
    bool finite = true;
    for (const FgExecOutputRecord& rec : outs) {
        const size_t px = static_cast<size_t>(rec.width) * rec.height;
        if (rec.kind == 'g') {
            ++rr.generated;
            size_t nonzero = 0;
            for (size_t i = 0; i < px; ++i)
                for (int ch = 0; ch < 3; ++ch) {
                    const uint16_t v = rec.image[4 * i + ch];
                    if ((v & 0x7C00) == 0x7C00) finite = false;
                    if (v & 0x7FFF) ++nonzero;
                }
            if (nonzero > px) ++gen_ok;
        } else {
            if (rec.kind == 'm') ++rr.copies;
            if (rec.kind == 'r') ++rr.real;
            if (rec.frame_token == static_cast<uint64_t>(menu_frame) + 1 && rec.kind == 'm') ++menu_tok_copies;
            // an exact copy of the game frame (its colour texture is the frame)
            if (rec.colour_width == rec.width && rec.colour_height == rec.height && rec.image == rec.colour) ++copy_ok;
        }
        // the slot holds the image in its format
        bool same = true;
        const uint8_t* s = rec.slot.data();
        for (uint32_t y = 0; y < rec.height && same; ++y)
            for (uint32_t x = 0; x < rec.width && same; ++x) {
                const uint16_t* p = &rec.image[(static_cast<size_t>(y) * rec.width + x) * 4];
                const uint8_t* q = s + static_cast<size_t>(y) * rec.slot_pitch;
                if (rec.slot_format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                    same = std::memcmp(q + 8ull * x, p, 8) == 0;
                } else if (rec.slot_format == DXGI_FORMAT_B8G8R8A8_UNORM) {
                    same = q[4 * x] == Unorm8(p[2]) && q[4 * x + 1] == Unorm8(p[1]) && q[4 * x + 2] == Unorm8(p[0]) &&
                           q[4 * x + 3] == Unorm8(p[3]);
                }
            }
        if (same) ++slot_ok;
    }
    rr.menu_copies = menu_tok_copies;
    std::printf("    outputs: %zu (generated %llu finite+nonzero %llu; copies %llu, real %llu, exact copies %llu; "
                "slot bytes match %llu; menu-frame copies %llu)\n",
                outs.size(), static_cast<unsigned long long>(rr.generated), static_cast<unsigned long long>(gen_ok),
                static_cast<unsigned long long>(rr.copies), static_cast<unsigned long long>(rr.real),
                static_cast<unsigned long long>(copy_ok), static_cast<unsigned long long>(slot_ok),
                static_cast<unsigned long long>(menu_tok_copies));
    CHECK(rr.generated > 0 && gen_ok == rr.generated && finite);
    CHECK(copy_ok == rr.copies + rr.real);
    CHECK(slot_ok == outs.size());
    if (menu_frame >= 0) CHECK(menu_tok_copies == n);
    // every slot comes back to the DXGI back end once the display path is done (the unbound composition surface
    // never composes, so its slots stay with the display: see HeadlessSource)
    if (FgDxgiPresenter* dp = dynamic_cast<FgDxgiPresenter*>(pres)) {
        uint32_t free_slots = 0;
        for (int i = 0; i < 50 && free_slots != dp->Info().slot_count; ++i) {
            Sleep(20);
            free_slots = dp->FreeSlots();
        }
        std::printf("    dxgi slots free after the run: %u / %u\n", free_slots, dp->Info().slot_count);
        CHECK(free_slots == dp->Info().slot_count);
    }
    const int64_t dbg = gpu.DebugErrors();
    if (dbg >= 0) {
        std::printf("    debug layer errors: %lld\n", static_cast<long long>(dbg));
        CHECK(dbg == 0);
    }
    // shutdown
    std::fflush(stdout);
    FgSessionResult dr = session.Destroy();
    for (int i = 0; i < 100 && dr.error == FgSessionError::PresenterBusy; ++i) {
        Sleep(20);
        dr = session.Destroy();
    }
    CHECK(dr.error == FgSessionError::None && session.State() == FgSessionState::None);
    std::printf("    destroyed: %s\n", FgSessionErrorName(dr.error));
    std::fflush(stdout);
    proxy.reset();
    // the queue's last presents and copies finish before the swap chain goes away
    CHECK(gpu.Wait());
    headless.chain.Reset();
    return rr;
}

bool CopyDir(const std::string& from, const std::string& to, const char* broken, bool truncate_pso) {
    CreateDirectoryA(to.c_str(), nullptr);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((from + "\\*.dxil").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return false;
    do {
        std::ifstream in(from + "\\" + fd.cFileName, std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (std::strcmp(fd.cFileName, broken) == 0) {
            if (truncate_pso && b.size() > 64)
                b.resize(b.size() / 2);  // a container header with half its parts: the driver must refuse it
            else
                b.assign(64, 'x');
        }
        WriteFile(to + "\\" + fd.cFileName, b.data(), b.size());
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return true;
}

void TestBrokenKernels(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, const std::string& scratch) {
    g_case = "gpu-broken-kernels";
    for (int v = 0; v < 2; ++v) {
        const std::string dir = scratch + (v ? "\\dxil_half" : "\\dxil_junk");
        CHECK(CopyDir(dxil, dir, "fg_lite_blend.dxil", v == 1));
        FgD3d12ExecutorDesc ed;
        ed.dxil_dir = dir;
        ed.weights = &weights;
        FgD3d12Executor exec(ed);
        HeadlessSource headless;
        headless.gpu = &gpu;
        FgExecutorPresenterSource source(&headless, &exec);
        FgSession s;
        FgSessionDesc sd;
        sd.create = CpuCreate(kW, kH);
        sd.create.device = gpu.device.Get();
        sd.create.queue = gpu.queue.Get();
        sd.executor = &exec;
        sd.presenter_source = &source;
        sd.clock = &RealClock;
        const FgSessionResult r = s.Create(sd);
        std::printf("    %s: session %s, executor %s (%s)\n", v ? "half container" : "junk file",
                    FgSessionErrorName(r.error), FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str());
        CHECK(r.error == FgSessionError::ExecutorConfigure);
        CHECK(exec.LastRefusal() == (v ? FgExecRefusal::PipelineFailed : FgExecRefusal::DxilInvalid));
        CHECK(exec.LastWhy().find("fg_lite_blend") != std::string::npos);
        CHECK(headless.last.empty() && s.State() == FgSessionState::None);  // no presenter was ever created
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool gpu = false;
    std::string dxil, scratch = ".", only;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) gpu = true;
        if (std::strcmp(argv[i], "--dxil") == 0 && i + 1 < argc) dxil = argv[++i];
        if (std::strcmp(argv[i], "--scratch") == 0 && i + 1 < argc) scratch = argv[++i];
        if (std::strcmp(argv[i], "--debug-layer") == 0) g_debug_layer = true;
        if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
    }
    FgExecWeights weights;
    FgExecSyntheticWeights(11, &weights);
    TestGeometry();
    TestRegionsAndThresholds();
    TestHalf();
    TestSigTable();
    TestSchedule();
    TestRefusals(weights);
    std::printf("cpu: %d checks, %d failed\n", g_checks, g_failed);
    if (gpu) {
        Gpu g;
        g_case = "gpu-init";
        CHECK(!dxil.empty());
        const bool ok = g.Init();
        CHECK(ok);
        if (ok && !dxil.empty()) {
            std::printf("gpu: %s\n", g.name.c_str());
            struct Cfg {
                const char* name;
                FgPreset preset;
                uint32_t n;
                bool dxgi;
                DXGI_FORMAT fmt;
                int menu;
                int frames;
            };
            const Cfg cfgs[] = {
                {"gpu-lite-2x-composition", FgPreset::C, 1, false, DXGI_FORMAT_B8G8R8A8_UNORM, 8, 14},
                {"gpu-lite-6x-composition", FgPreset::C, 5, false, DXGI_FORMAT_B8G8R8A8_UNORM, 6, 8},
                {"gpu-full-a-2x-composition", FgPreset::A, 1, false, DXGI_FORMAT_B8G8R8A8_UNORM, -1, 14},
                {"gpu-lite-2x-dxgi", FgPreset::C, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 14},
                {"gpu-lite-6x-dxgi", FgPreset::C, 5, true, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 14},
                {"gpu-full-a-2x-dxgi", FgPreset::A, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, -1, 14},
            };
            int ran_comp = 0, ran_dxgi = 0;
            for (const Cfg& c : cfgs) {
                if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
                g_case = c.name;
                std::printf("  %s\n", c.name);
                const RunResult r = RunPipeline(g, dxil, weights, c.preset, c.n, c.dxgi, c.fmt, c.frames, c.menu);
                if (r.ran) (c.dxgi ? ran_dxgi : ran_comp)++;
            }
            g_case = "gpu-backends";
            std::printf("  configurations run: composition %d / 3, dxgi %d / 3\n", ran_comp, ran_dxgi);
            if (only.empty()) CHECK(ran_comp == 3 && ran_dxgi == 3);  // both back ends ran every configuration
            if (only.empty()) TestBrokenKernels(g, dxil, weights, scratch);
        }
    }
    std::printf("fg_d3d12_e2e: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
