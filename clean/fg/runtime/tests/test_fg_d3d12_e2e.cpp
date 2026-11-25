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
//   test_fg_d3d12_e2e --bench --dxil <dir>  reports steady-state 1440p GPU stage timings for presets A and C
//       [--gen N]       N = 1..5 generated images per game frame (2x .. 6x), really recorded (default 1)
//       [--preset AC]   which presets (default AC)       [--timed N] [--warmup N]
//       [--kprof]       per-launch profile: kernel, workgroups, time per call (serialises the queue a little)
//       [--batch 0|1] [--hoist 0|1] [--dualq 0|1] [--fuse 0|1] [--fuse2 0|1|MASK]   Turbo v2 levers for paired A / B runs
//                       (default: on, or the environment WG_FG_BATCH / _HOIST / _DUALQ / _FUSE / _FUSE2 = 0; --fuse2 takes
//                       a mask of the Wave 1B levers: 1 net1 cells, 2 net2 pairs, 4 upwarp in L14, 8 blend weights,
//                       16 pyramid tail, 32 X0 / X1 as RGB)
//       [--turbo-fuse2 MASK]  the Wave 1B mask the --only turbo legs run against all levers off (default 63: every lever)
//       The report has the per-stage sums and the honest multi-queue numbers: the wall-clock GPU span and the busy
//       time per game frame over the timed window (overlap between queues is counted once there, twice in the sums).
//   --reference-dxil <dir>  compare direct-SRV output bytes against the captured path built with these kernels
//                                           adapter core, the FG session and this executor on the D3D12 device with
//                                           the composition back end (unbound surface) and the DXGI back end
//                                           (composition swap chain): preset C 2x and 6x, presets A / B 2x, a menu
//                                           frame, net2's exact path, generation switched off for some frames
//                                           (real-only from the back buffer), the swap chain released right after
//                                           teardown, net2's fast path against the exact one (tolerance, determinism,
//                                           a negative control, GPU time at 1080p / 1440p), and a broken kernel folder
//                                           that must be refused
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
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

// Turbo v2: the order in which the launches of several indices sharing one command list are recorded, and the shape of an
// index's ops that the hoisted work (the first three) and the batch rely on.
void TestInterleave() {
    g_case = "interleave";
    using Key = FgLaunchKey;
    std::vector<FgLaunchSlot> out;
    FgInterleaveLaunches({}, &out);
    CHECK(out.empty());
    // one frame: its own order and barrier flags
    const std::vector<Key> plan = {{1, true}, {2, false}, {3, true}, {4, true}};
    FgInterleaveLaunches({plan}, &out);
    CHECK(out.size() == 4);
    for (uint32_t j = 0; j < out.size(); ++j)
        CHECK(out[j].frame == 0 && out[j].index == j && out[j].barrier == plan[j].barrier && out[j].mark_stage);
    // three frames of the same plan: launch j of every frame in turn, one barrier only before frame 0's launch j and
    // only where the plan needs one; the stage mark only on frame 0
    FgInterleaveLaunches({plan, plan, plan}, &out);
    CHECK(out.size() == 12);
    bool order = true;
    for (uint32_t j = 0; j < 4; ++j)
        for (uint32_t f = 0; f < 3; ++f) {
            const FgLaunchSlot& x = out[j * 3 + f];
            order = order && x.frame == f && x.index == j && x.mark_stage == (f == 0) &&
                    x.barrier == (f == 0 && plan[j].barrier);
        }
    CHECK(order);
    // every launch of every frame appears exactly once
    int seen[3][4] = {};
    for (const FgLaunchSlot& x : out) ++seen[x.frame][x.index];
    bool once = true;
    for (auto& row : seen)
        for (int n : row) once = once && n == 1;
    CHECK(once);
    // frames of different plans (a different kernel, a different barrier need, a different length) are not interleaved:
    // frame after frame with the frame's own barrier flags
    const std::vector<Key> other_kernel = {{1, true}, {2, false}, {9, true}, {4, true}};
    const std::vector<Key> other_barrier = {{1, true}, {2, true}, {3, true}, {4, true}};
    const std::vector<Key> shorter = {{1, true}, {2, false}};
    for (const auto& frames : {std::vector<std::vector<Key>>{plan, other_kernel},
                               std::vector<std::vector<Key>>{plan, other_barrier},
                               std::vector<std::vector<Key>>{plan, shorter}}) {
        FgInterleaveLaunches(frames, &out);
        size_t total = 0, at = 0;
        for (const auto& f : frames) total += f.size();
        CHECK(out.size() == total);
        bool sequential = out.size() == total;
        for (uint32_t f = 0; f < frames.size(); ++f)
            for (uint32_t j = 0; sequential && j < frames[f].size(); ++j, ++at)
                sequential = out[at].frame == f && out[at].index == j && out[at].barrier == frames[f][j].barrier &&
                             out[at].mark_stage;
        CHECK(sequential);
    }
    // the hoisted work is the first three ops of an index with history (fills of MA / MB, then the intermediate-time
    // scatters), for every index and preset; a game frame without history has none
    for (const bool lite : {true, false}) {
        FgExecSettings st;
        FgExecGeometry g;
        std::string why;
        FgExecMakeGeometry(160, 90, &g, &why);
        FgExecHost h;
        h.Configure(st, g);
        std::vector<FgExecOp> ops;
        h.BeginFrame(lite, false, true, false, false, true, &ops);
        std::vector<uint32_t> pairs;
        h.EndFrame(&pairs);
        ops.clear();
        h.BeginFrame(lite, false, true, false, false, false, &ops);
        CHECK(h.Frame().history);
        for (uint32_t i = 1; i <= 5; ++i) {
            std::vector<FgExecOp> idx;
            h.IndexOps(FgIndexTime(i, 5), true, false, false, &idx);
            CHECK(idx.size() >= 5 && idx[0].op == FgOp::Fill64 && idx[0].role == kRoleMA && idx[1].op == FgOp::Fill64 &&
                  idx[1].role == kRoleMB && idx[2].op == FgOp::Interm && idx[3].op == FgOp::Weights);
            const float t = FgIndexTime(i, 5);
            CHECK(std::memcmp(&idx[2].t, &t, sizeof(float)) == 0);
        }
    }
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

// net2's fast path: the 27-launch plan (variants as clean/fg/runner/fg_net2_mma.py variant() picks them for the same
// grids), the kernel lists per path, and the A-fragment swizzle on constructed weights.
void TestNet2FastPlan() {
    g_case = "net2-fast-plan";
    std::vector<FgNet2FastLaunch> p;
    CHECK(FgNet2FastPlan(480, 272, &p) && p.size() == 27);
    const char* const want480[27] = {
        "fg_net2_merge",       "fg_net2_mconv_m2b1",  "fg_net2_mconv_m2b2",  "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b4s",
        "fg_net2_mconv_m1b4s", "fg_net2_mconv_m1b4s", "fg_net2_mconv_m1b4s", "fg_net2_mconv_m1b4s", "fg_net2_mconv_m1b4s",
        "fg_net2_mconv_m1b4s", "fg_net2_mconv_m1b4s", "fg_net2_mconv_m3b4u", "fg_net2_mconv_m1b2u", "fg_net2_upwarp",
        "fg_net2_mconv_m1b2",  "fg_net2_mconv_m2b1",  "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b2s",
        "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b2s",
        "fg_net2_mconv_m3b2u", "fg_net2_mconv_m1b1u"};
    bool names = p.size() == 27;
    for (size_t i = 0; names && i < 27; ++i) names = std::strcmp(p[i].kernel, want480[i]) == 0;
    CHECK(names);
    if (p.size() == 27) {
        CHECK(p[0].groups == 5100 && p[1].groups == 30 * 17 && p[13].layer == 13 && p[13].groups == 15 * 9 * 3);
        CHECK(p[14].layer == 26 && p[14].groups == 510 && p[26].layer == 25 && p[26].groups == 30 * 17 * 3);
        CHECK(p[12].H == 68 && p[12].W == 120 && p[25].H == 136 && p[25].W == 240);
    }
    // a small grid takes the 128-lane one-tile variants where the layer has one (L2, L14)
    CHECK(FgNet2FastPlan(48, 32, &p) && p.size() == 27);
    if (p.size() == 27) {
        CHECK(std::strcmp(p[2].kernel, "fg_net2_mconv_m1b2s") == 0 && std::strcmp(p[15].kernel, "fg_net2_mconv_m1b2s") == 0);
        CHECK(std::strcmp(p[1].kernel, "fg_net2_mconv_m2b1") == 0 && std::strcmp(p[26].kernel, "fg_net2_mconv_m1b1u") == 0);
    }
    CHECK(!FgNet2FastPlan(470, 272, &p) && !FgNet2FastPlan(480, 0, &p));
    // the kernel lists: only the chosen path's net2 kernels (the input merge is in both)
    const std::vector<std::string> fast = FgD3d12ExecutorKernelNames(true, FgNet2Path::Fast);
    const std::vector<std::string> exact = FgD3d12ExecutorKernelNames(true, FgNet2Path::Exact);
    const std::vector<std::string> lite = FgD3d12ExecutorKernelNames(false, FgNet2Path::Fast);
    auto has = [](const std::vector<std::string>& v, const char* n) {
        for (const std::string& s : v)
            if (s == n) return true;
        return false;
    };
    CHECK(has(fast, "fg_net2_mconv_m3b4u") && has(fast, "fg_net2_upwarp") && has(fast, "fg_net2_merge") &&
          !has(fast, "fg_net2_conv") && !has(fast, "fg_net2_warp"));
    CHECK(has(exact, "fg_net2_conv") && has(exact, "fg_net2_up2") && has(exact, "fg_net2_merge") &&
          !has(exact, "fg_net2_mconv_m1b1") && !has(exact, "fg_net2_upwarp"));
    CHECK(fast.size() == 36 + 13 + 1 && exact.size() == 36 + 4 && lite.size() == 31);
    CHECK(!has(lite, "fg_full_blend") && !has(lite, "fg_net2_merge") && has(lite, "fg_mpair_d0"));
    // swizzle: 20 outputs (2 tiles, zero-padded to 32) x 18 channels (2 blocks, padded to 32); w[o][t][c] = o*1000 +
    // t*32 + c as float16 bits (distinct constructed values)
    const uint32_t cout = 20, cw = 18;
    std::vector<uint8_t> w(2ull * cout * 9 * cw), b(2ull * cout), frag, bp;
    for (uint32_t o = 0; o < cout; ++o)
        for (uint32_t t = 0; t < 9; ++t)
            for (uint32_t c = 0; c < cw; ++c) {
                const uint16_t v = static_cast<uint16_t>(o * 1000 + t * 32 + c + 1);
                const size_t i = (static_cast<size_t>(o) * 9 + t) * cw + c;
                w[2 * i] = static_cast<uint8_t>(v & 0xFF);
                w[2 * i + 1] = static_cast<uint8_t>(v >> 8);
            }
    for (uint32_t o = 0; o < cout; ++o) b[2 * o] = static_cast<uint8_t>(o + 1);
    CHECK(FgNet2SwizzleFast(w, b, cout, cw, &frag, &bp));
    CHECK(frag.size() == 2ull * 2 * 9 * 2 * 2 * 16 * 8 && bp.size() == 64);
    bool sw = frag.size() == 2ull * 2 * 9 * 2 * 2 * 16 * 8;
    for (uint32_t mt = 0; sw && mt < 2; ++mt)
        for (uint32_t t = 0; t < 9; ++t)
            for (uint32_t cb = 0; cb < 2; ++cb)
                for (uint32_t hh = 0; hh < 2; ++hh)
                    for (uint32_t m = 0; m < 16; ++m)
                        for (uint32_t j = 0; j < 8; ++j) {
                            const size_t at = ((((static_cast<size_t>(mt) * 9 + t) * 2 + cb) * 2 + hh) * 16 + m) * 8 + j;
                            const uint32_t o = 16 * mt + m, c = 16 * cb + 8 * hh + j;
                            const uint16_t got = static_cast<uint16_t>(frag[2 * at] | (frag[2 * at + 1] << 8));
                            const uint16_t want = (o < cout && c < cw) ? static_cast<uint16_t>(o * 1000 + t * 32 + c + 1)
                                                                       : uint16_t(0);
                            if (got != want) sw = false;
                        }
    CHECK(sw);
    CHECK(bp[0] == 1 && bp[38] == 20 && bp[40] == 0 && bp[62] == 0);
    CHECK(!FgNet2SwizzleFast(w, b, cout, 40, &frag, &bp));  // 3 blocks: no variant
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

struct PipelineClock {
    FgTicks arrival = 0;
    static FgTicks Read(void* ctx) {
        auto& clock = *static_cast<PipelineClock*>(ctx);
        if (clock.arrival == 0) return FgInterruptNow();
        const FgTicks arrival = clock.arrival;
        clock.arrival = 0;
        return arrival;
    }
};
bool g_debug_layer = false;
int g_turbo_fuse2 = 63;  // the Wave 1B mask TestTurboExact runs against all levers off (--turbo-fuse2 MASK)
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
    uint32_t width = kW, height = kH;
    ComPtr<ID3D12Resource> colour, hudless, ui, depth, motion;
    DXGI_FORMAT colour_format = DXGI_FORMAT_R16G16B16A16_FLOAT;  // the chain's format (B8G8R8A8 or RGBA16F)
    ComPtr<ID3D12Resource> upload;
    D3D12_RESOURCE_STATES colour_state = D3D12_RESOURCE_STATE_COPY_SOURCE;  // colour left in a copy state on purpose
    D3D12_RESOURCE_STATES other_state = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    float clip_to_prev[16] = {}, prev_to_clip[16] = {};

    ComPtr<ID3D12Resource> Tex(DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = width;
        rd.Height = height;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = f;
        rd.SampleDesc.Count = 1;
        rd.Flags = flags;
        ComPtr<ID3D12Resource> r;
        gpu->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&r));
        return r;
    }
    bool Init(Gpu* g, DXGI_FORMAT cf, uint32_t w = kW, uint32_t h = kH) {
        colour_format = cf;
        gpu = g;
        width = w;
        height = h;
        const uint32_t pitch8 = (width * 8 + 255) & ~255u;
        colour = Tex(cf, colour_state == D3D12_RESOURCE_STATE_RENDER_TARGET
                            ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_NONE);
        hudless = Tex(DXGI_FORMAT_R16G16B16A16_FLOAT);
        ui = Tex(DXGI_FORMAT_R16G16B16A16_FLOAT);
        depth = Tex(DXGI_FORMAT_R32_FLOAT);
        motion = Tex(DXGI_FORMAT_R16G16_FLOAT);
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 5ull * pitch8 * height;
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
        const float tx = 12.0f / static_cast<float>(width);
        clip_to_prev[12] = tx;
        prev_to_clip[12] = -tx;
        return colour && hudless && ui && depth && motion && upload;
    }
    FgFramePacket Packet() const {
        const auto input = [&](ID3D12Resource* tex, D3D12_RESOURCE_STATES state) {
            FgTextureRef ref;
            ref.resource = tex;
            ref.format = tex->GetDesc().Format;
            ref.state = state;
            ref.width = width;
            ref.height = height;
            ref.rect = {0, 0, width, height};
            return ref;
        };
        FgFramePacket packet;
        packet.color = input(colour.Get(), colour_state);
        packet.depth = input(depth.Get(), other_state);
        packet.motion = input(motion.Get(), other_state);
        packet.optional[kFgOptHudless] = input(hudless.Get(), other_state);
        packet.optional[kFgOptUiAlpha] = input(ui.Get(), other_state);
        packet.optional_present = FgOptBit(kFgOptHudless) | FgOptBit(kFgOptUiAlpha);
        packet.flags = kFgFlagDepthReversed;
        std::memcpy(packet.camera_current, clip_to_prev, sizeof(clip_to_prev));
        std::memcpy(packet.camera_previous, prev_to_clip, sizeof(prev_to_clip));
        return packet;
    }
    // Writes frame `f` (menu: the overlay) into the textures and waits for the copies.
    bool Upload(uint32_t f, bool menu, bool first) {
        uint8_t* base = nullptr;
        if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&base)))) return false;
        const uint32_t p8 = (width * 8 + 255) & ~255u, p4 = (width * 4 + 255) & ~255u, region = p8 * height;
        uint16_t* col = reinterpret_cast<uint16_t*>(base);
        uint16_t* hud = reinterpret_cast<uint16_t*>(base + region);
        uint16_t* uim = reinterpret_cast<uint16_t*>(base + 2 * region);
        float* dep = reinterpret_cast<float*>(base + 3 * region);
        uint16_t* mot = reinterpret_cast<uint16_t*>(base + 4 * region);
        const float pan = 6.0f * static_cast<float>(f);
        const uint32_t bx0 = static_cast<uint32_t>(width * 0.3f) + 2 * f, by0 = static_cast<uint32_t>(height * 0.35f) + f;
        const uint32_t bw = width / 5, bh = height / 4;
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const uint32_t i = y * width + x;
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
                const bool panel = x >= width * 0.62f && x < width * 0.62f + width / 4 && y >= height * 0.08f && y < height * 0.08f + height / 5;
                if (panel) {
                    alpha = ((x / 3 + y / 2) % 3 == 0) ? 1.0f : 0.75f;
                    uc[0] = alpha == 1.0f ? 1.0f : 0.1f;
                    uc[1] = alpha == 1.0f ? 1.0f : 0.8f;
                    uc[2] = alpha == 1.0f ? 1.0f : 0.9f;
                }
                if (menu && x >= width * 0.05f && x < width * 0.95f && y >= height * 0.03f && y < height * 0.97f) {
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
                const float dplane = 0.3f + 0.3f * fy / height;
                dep[y * p4 / 4 + x] = 1.0f - (block ? 0.012f + 0.008f * fx / width : dplane);
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
            src.PlacedFootprint.Footprint = {fmts[t], width, height, 1, pitches[t]};
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
        // Fit two complete intervals and the real reserve while the headless display thread drains.
        dd.slot_count = 2 * (desc.max_generated + 1) + 1;
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
// off_from .. off_to - 1: game frames with frame generation switched off in the game's options (the adapter core
// answers real-only, reason disabled): the session presents them real-only through OnGameFrameOnly, and the executor
// copies the game's back buffer into a reserved slot.
RunResult RunPipeline(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, FgPreset preset, uint32_t n,
                      bool dxgi, DXGI_FORMAT slot_format, int frames, int menu_frame, FgNet2Path net2 = FgNet2Path::Fast,
                      int off_from = -1, int off_to = -1) {
    RunResult rr;
    FgD3d12ExecutorDesc ed;
    ed.dxil_dir = dxil;
    ed.weights = &weights;
    ed.net2 = net2;
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

    PipelineClock clock;
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
    sd.clock = &PipelineClock::Read;
    sd.clock_ctx = &clock;
    sd.log = &PrintLog;
    g_log_lines = 0;
    sd.scheduler.display.min_interval = 1;
    sd.scheduler.display.max_interval = 10000000;
    sd.scheduler.max_lead = 2000000;
    sd.scheduler.first_image_budget = dxgi ? 300000 : 600000;
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

    const FgTicks period = dxgi ? 500000 : 333333;
    const FgTicks first_arrival = FgInterruptNow() + period;
    int errors = 0;
    bool menu_seen = false;
    uint32_t menu_recorded = 0;
    int off_frames = 0;
    for (int f = 0; f < frames; ++f) {
        const uint64_t tok = static_cast<uint64_t>(f) + 1;
        const bool menu = f == menu_frame;
        if (f == off_from || f == off_to) {
            o.enable = f == off_from ? 0u : 1u;
            proxy->set_options(WG_FEAT_FRAME_GENERATION, 1, &o, &o);
        }
        if (f >= off_from && f < off_to) ++off_frames;
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
        // Fix the synthetic arrival cadence. The presenter and deadline checks still use wall time.
        clock.arrival = first_arrival + f * period;
        while (FgInterruptNow() < clock.arrival) Sleep(1);
        FgSessionPresent(proxy.get(), &session, &pd, &pr, &r);
        clock.arrival = 0;
        if (r.error != FgSessionError::None) {
            ++errors;
            std::printf("    frame %llu: session %s, executor %s (%s), presenter %s; core status %u reason %u "
                        "real-only %u\n",
                        static_cast<unsigned long long>(tok), FgSessionErrorName(r.error),
                        FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str(), FgStatusName(r.presenter),
                        pr.status, pr.reason, pr.real_only);
        }
        if (menu) {
            menu_seen = r.menu_detected;
            menu_recorded = r.copies_recorded;
            if (menu_recorded != n)
                std::printf("    menu frame: scheduled %u, copied %u, unreserved %u, stale %u\n",
                            r.scheduled, menu_recorded, r.not_reserved, r.stale_cancelled);
        }
        ++rr.frames;
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
    std::printf("    real-only from the back buffer: %llu (frames with generation off: %d), net2 fast %llu exact %llu\n",
                static_cast<unsigned long long>(ec.game_colour_copies), off_frames,
                static_cast<unsigned long long>(ec.net2_fast), static_cast<unsigned long long>(ec.net2_exact));
    // every game frame with generation on had its shared work; each one with it off was copied from the back buffer
    const uint64_t on_frames = static_cast<uint64_t>(frames - off_frames);
    CHECK(sc.shared_records == on_frames && ec.shared == on_frames);
    CHECK(ec.game_colour_copies == static_cast<uint64_t>(off_frames));
    if (preset != FgPreset::C) CHECK((net2 == FgNet2Path::Fast ? ec.net2_fast : ec.net2_exact) > 0);
    CHECK((net2 == FgNet2Path::Fast ? ec.net2_exact : ec.net2_fast) == 0);
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
    if (menu_frame >= 0) CHECK(menu_recorded == n && menu_tok_copies == menu_recorded);
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
    // the DXGI back end drained the queue past its last copy and present when it was destroyed (FgDxgiPresenter's
    // destructor), so the swap chain goes away at once, without a wait here (under the debug layer this used to crash)
    headless.chain.Reset();
    CHECK(gpu.Wait());
    const int64_t dbg2 = gpu.DebugErrors();
    if (dbg2 >= 0) {
        std::printf("    debug layer errors after teardown: %lld\n", static_cast<long long>(dbg2));
        CHECK(dbg2 == 0);
    }
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

// net2 alone inside a configured executor (FgD3d12Executor::Net2SelfTest) on synthetic smooth inputs: one run, a
// second run (deterministic: same bytes), and optionally `runs` timed recordings.
struct Net2Run {
    bool ok = false;
    std::vector<uint16_t> out;
    bool deterministic = false;
    double gpu_ms = -1;
};
Net2Run RunNet2(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, FgNet2Path path, uint32_t w,
                uint32_t h, uint32_t runs, bool half_last) {
    Net2Run r;
    FgD3d12ExecutorDesc ed;
    ed.dxil_dir = dxil;
    ed.weights = &weights;
    ed.net2 = path;
    FgD3d12Executor exec(ed);
    FgCreateDesc c = CpuCreate(w, h);
    c.device = gpu.device.Get();
    c.queue = gpu.queue.Get();
    c.adapter_luid.low = gpu.luid.LowPart;
    c.adapter_luid.high = gpu.luid.HighPart;
    c.output_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    c.preset = FgPreset::A;
    if (exec.Configure(c, 3) != FgExecResult::Ok) {
        std::printf("    configure: %s (%s)\n", FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str());
        return r;
    }
    std::vector<uint16_t> again;
    double ms = -1;
    if (exec.Net2SelfTest(7, 0.4375f, 0, half_last, &r.out, nullptr) != FgExecResult::Ok ||
        exec.Net2SelfTest(7, 0.4375f, runs, false, &again, &ms) != FgExecResult::Ok) {
        std::printf("    self-test: %s (%s)\n", FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str());
        return r;
    }
    r.deterministic = half_last || again == r.out;
    r.gpu_ms = ms;
    r.ok = true;
    return r;
}

struct Net2Diff {
    size_t values = 0, nonfinite = 0, unwritten = 0, over_step = 0;
    double max_abs = 0, max_ref = 0, rel = 0;
};
// fast against exact: the largest absolute difference over the exact output's largest magnitude, values more than one
// float16 step apart, values the fast run left at the 0x5A5A fill, non-finite values
Net2Diff CompareNet2(const std::vector<uint16_t>& fast, const std::vector<uint16_t>& exact) {
    Net2Diff d;
    d.values = std::min(fast.size(), exact.size());
    for (size_t i = 0; i < d.values; ++i) {
        const float a = FgExecHalfToFloat(fast[i]), b = FgExecHalfToFloat(exact[i]);
        if (fast[i] == 0x5A5A && exact[i] != 0x5A5A) ++d.unwritten;
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++d.nonfinite;
            continue;
        }
        d.max_abs = std::max(d.max_abs, static_cast<double>(std::fabs(a - b)));
        d.max_ref = std::max(d.max_ref, static_cast<double>(std::fabs(b)));
        const int sa = (fast[i] & 0x8000) ? -(fast[i] & 0x7FFF) : (fast[i] & 0x7FFF);
        const int sb = (exact[i] & 0x8000) ? -(exact[i] & 0x7FFF) : (exact[i] & 0x7FFF);
        if (std::abs(sa - sb) > 1) ++d.over_step;
    }
    d.rel = d.max_ref > 0 ? d.max_abs / d.max_ref : 0;
    return d;
}

// The fast (tier B, matrix unit) and the exact net2 paths inside the executor: same synthetic inputs and weights, the
// fast output within tolerance of the exact one (tier B differs only in the chunk sums), both deterministic, every
// value written; a fast run whose last launch covers half its workgroups must be caught; GPU time of both paths at the
// 1080p and 1440p net2 grids (information; device timestamps around net2's launches after one warm-up).
void TestNet2Paths(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights) {
    struct Size {
        const char* label;
        uint32_t w, h, fast_runs, exact_runs;
    };
    const Size sizes[] = {{"320x180", 320, 180, 0, 0}, {"1080p", 1920, 1080, 20, 3}, {"1440p", 2560, 1440, 20, 3}};
    const double kTolerance = 3e-2;  // largest |fast - exact| over the exact output's largest magnitude
    for (const Size& s : sizes) {
        const std::string case_name = std::string("gpu-net2-") + s.label;
        g_case = case_name.c_str();
        std::printf("  net2 fast vs exact, %s output\n", s.label);
        const Net2Run fast = RunNet2(gpu, dxil, weights, FgNet2Path::Fast, s.w, s.h, s.fast_runs, false);
        const Net2Run exact = RunNet2(gpu, dxil, weights, FgNet2Path::Exact, s.w, s.h, s.exact_runs, false);
        CHECK(fast.ok && exact.ok);
        if (!fast.ok || !exact.ok) continue;
        CHECK(fast.deterministic && exact.deterministic);
        const Net2Diff d = CompareNet2(fast.out, exact.out);
        std::printf("    values %zu: max |fast - exact| %.5g, max |exact| %.5g, rel %.3g; > 1 float16 step %zu (%.3f %%); "
                    "unwritten %zu, non-finite %zu; deterministic fast %d exact %d\n",
                    d.values, d.max_abs, d.max_ref, d.rel, d.over_step,
                    d.values ? 100.0 * static_cast<double>(d.over_step) / static_cast<double>(d.values) : 0.0,
                    d.unwritten, d.nonfinite, fast.deterministic ? 1 : 0, exact.deterministic ? 1 : 0);
        CHECK(d.values > 0 && d.unwritten == 0 && d.nonfinite == 0 && d.max_ref > 0 && d.rel <= kTolerance);
        if (s.fast_runs || s.exact_runs)
            std::printf("    GPU time (UNCHECKED timing, information only): fast %.3f ms (%u runs), exact %.3f ms "
                        "(%u runs)\n",
                        fast.gpu_ms, s.fast_runs, exact.gpu_ms, s.exact_runs);
        if (s.fast_runs == 0) {
            // negative control: the fast chain's last launch on half of its workgroups must be caught
            const Net2Run bad = RunNet2(gpu, dxil, weights, FgNet2Path::Fast, s.w, s.h, 0, true);
            CHECK(bad.ok);
            const Net2Diff b = CompareNet2(bad.out, exact.out);
            std::printf("    control (last launch on half its workgroups): unwritten %zu, rel %.3g -> %s\n", b.unwritten,
                        b.rel, (b.unwritten > 0 || b.rel > kTolerance) ? "caught" : "MISSED");
            CHECK(b.unwritten > 0 || b.rel > kTolerance);
        }
    }
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

// The motion input: x and y come from the first two channels, so a two-channel and a four-channel format are accepted;
// one channel, a depth format and an unknown format are not.
void TestMotionFormats() {
    g_case = "motion-formats";
    CHECK(FgD3d12MotionFormatAccepted(DXGI_FORMAT_R16G16_FLOAT));
    CHECK(FgD3d12MotionFormatAccepted(DXGI_FORMAT_R16G16_TYPELESS));
    CHECK(FgD3d12MotionFormatAccepted(DXGI_FORMAT_R32G32_FLOAT));
    CHECK(FgD3d12MotionFormatAccepted(DXGI_FORMAT_R16G16B16A16_FLOAT));
    CHECK(FgD3d12MotionFormatAccepted(DXGI_FORMAT_R16G16B16A16_TYPELESS));
    CHECK(FgD3d12MotionFormatAccepted(DXGI_FORMAT_R32G32B32A32_FLOAT));
    CHECK(!FgD3d12MotionFormatAccepted(DXGI_FORMAT_R16_FLOAT));
    CHECK(!FgD3d12MotionFormatAccepted(DXGI_FORMAT_R32_FLOAT));
    CHECK(!FgD3d12MotionFormatAccepted(DXGI_FORMAT_D32_FLOAT));
    CHECK(!FgD3d12MotionFormatAccepted(DXGI_FORMAT_UNKNOWN));
    CHECK(!FgD3d12MotionFormatAccepted(DXGI_FORMAT_BC1_UNORM));
}

struct ExecutorSlots final : IFgSlotTarget {
    ComPtr<ID3D12Resource> textures[8];  // the generated indices' slots and the real endpoint's (benchmarks use up to 6)
    ID3D12Resource* SlotResource(uint32_t slot) const override {
        return slot < 8 ? textures[slot].Get() : nullptr;
    }
    uint32_t RegisterFence(ID3D12Fence*, std::string*) override { return 1; }
    bool ReserveRealSlot(uint32_t* slot) override {
        *slot = 1;
        return true;
    }
};

bool CopiesMatchUpload(Gpu& gpu, const Game& game, const ExecutorSlots& slots) {
    const UINT pitch = (game.width * 4 + 255) & ~255u;
    const UINT64 bytes = static_cast<UINT64>(pitch) * game.height;
    const UINT64 stride = (bytes + 511) & ~511ull;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = stride + bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    if (FAILED(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&readback))) ||
        FAILED(gpu.alloc->Reset()) || FAILED(gpu.list->Reset(gpu.alloc.Get(), nullptr)))
        return false;
    for (uint32_t i = 0; i < 2; ++i) {
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = slots.textures[i].Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Offset = i * stride;
        dst.PlacedFootprint.Footprint = {game.colour_format, game.width, game.height, 1, pitch};
        // A COMMON texture is implicitly promoted for this read-only copy and decays at the list boundary.
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(gpu.list->Close())) return false;
    ID3D12CommandList* lists[] = {gpu.list.Get()};
    gpu.queue->ExecuteCommandLists(1, lists);
    if (!gpu.Wait()) return false;
    void* expected = nullptr;
    void* actual = nullptr;
    if (FAILED(game.upload->Map(0, nullptr, &expected))) return false;
    const bool mapped = SUCCEEDED(readback->Map(0, nullptr, &actual));
    bool exact = mapped;
    if (mapped) {
        const auto* images = static_cast<const uint8_t*>(actual);
        const auto* upload = static_cast<const uint8_t*>(expected);
        for (UINT y = 0; exact && y < game.height; ++y) {
            const UINT64 row = static_cast<UINT64>(pitch) * y;
            exact = std::memcmp(images + row, upload + row, game.width * 4) == 0 &&
                    std::memcmp(images + stride + row, upload + row, game.width * 4) == 0;
        }
    }
    const D3D12_RANGE none = {0, 0};
    if (mapped) readback->Unmap(0, &none);
    game.upload->Unmap(0, &none);
    return exact;
}

void TestCopyPaths(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, bool supplied_compute) {
    g_case = supplied_compute ? "gpu-copy-paths-supplied-compute" : "gpu-copy-paths-direct";
    constexpr DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
    Game game;
    game.colour_state = supplied_compute ? D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_RENDER_TARGET;
    game.other_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const bool game_ready = game.Init(&gpu, format) && game.Upload(0, false, true);
    CHECK(game_ready);
    if (!game_ready) return;
    ExecutorSlots slots;
    slots.textures[0] = game.Tex(format);
    slots.textures[1] = game.Tex(supplied_compute ? format : DXGI_FORMAT_B8G8R8A8_TYPELESS);
    CHECK(slots.textures[0] && slots.textures[1]);
    if (!slots.textures[0] || !slots.textures[1]) return;

    ComPtr<ID3D12CommandQueue> queue = gpu.queue;
    if (supplied_compute) {
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        queue.Reset();
        const HRESULT hr = gpu.device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue));
        CHECK(SUCCEEDED(hr));
        if (FAILED(hr)) return;
    }
    FgD3d12ExecutorDesc ed;
    ed.dxil_dir = dxil;
    ed.weights = &weights;
    ed.enable_full = false;
    FgD3d12Executor exec(ed);
    FgCreateDesc create = CpuCreate(kW, kH);
    create.device = gpu.device.Get();
    create.queue = queue.Get();
    const auto ok = [&](FgExecResult s) {
        if (s != FgExecResult::Ok) std::printf("    executor: %s\n", exec.LastWhy().c_str());
        CHECK(s == FgExecResult::Ok);
        return s == FgExecResult::Ok;
    };
    if (!ok(exec.Configure(create, 3))) return;
    const bool bound = exec.BindSlotTarget(&slots);
    CHECK(bound);
    if (!bound) return;

    const FgFramePacket packet = game.Packet();
    for (uint64_t token = 1; token <= 2; ++token) {
        FgSharedDispatchDesc shared;
        shared.frame_token = token;
        shared.preset = FgPreset::C;
        shared.has_previous = token > 1;
        shared.generates = true;
        shared.packet = &packet;
        FgSharedResult result;
        const uint64_t before = exec.LastComputeValue();
        if (!ok(exec.RecordSharedGameFrame(shared, &result))) return;
        CHECK(exec.LastComputeValue() == before + 1 && !result.menu.detected);
    }
    FgReadyFence ready;
    FgIndexDispatchDesc gen;
    gen.frame_token = 2;
    gen.index = gen.generated_count = 1;
    gen.is_last_index = true;
    gen.preset = FgPreset::C;
    gen.t = 0.5f;
    gen.slot.reserved = true;
    if (!ok(exec.RecordGeneratedIndex(gen, &ready))) return;
    const uint64_t dispatches = exec.Counters().dispatches;
    const bool uncaptured = exec.Counters().captures == 0;
    const uint64_t menu_cost = supplied_compute ? 2 + static_cast<uint64_t>(uncaptured) : 0;
    // Reuse the generated slot for the menu copy while compute may still be writing it.
    FgCopyDispatchDesc menu;
    menu.frame_token = 2;
    menu.slot.reserved = true;
    if (!ok(exec.RecordMenuCopy(menu, &ready))) return;
    CHECK(exec.Counters().dispatches == dispatches + menu_cost);

    FgRealDispatchDesc real;
    real.frame_token = 2;
    real.slot.slot = 1;
    real.slot.reserved = true;
    if (!ok(exec.RecordRealEndpoint(real, &ready))) return;
    CHECK(exec.Counters().dispatches == dispatches + menu_cost + (supplied_compute ? 2 : 0));
    real.frame_token = 3;
    real.real_only = true;
    real.game_colour = packet.color;
    if (!ok(exec.RecordRealEndpoint(real, &ready))) return;
    CHECK(exec.Counters().dispatches == dispatches + menu_cost + (supplied_compute ? 5 : 0));
    CHECK(exec.Counters().game_colour_copies == 1);
    const bool idle = exec.WaitIdle(5000);
    CHECK(idle && exec.CompletedComputeValue() >= ready.value);
    if (idle) CHECK(CopiesMatchUpload(gpu, game, slots));
    const int64_t debug_errors = gpu.DebugErrors();
    if (debug_errors >= 0) CHECK(debug_errors == 0);
}

// Upload shader-readable images directly in their declared format. The depth
// fixture uses an R32 typeless resource with the D32_FLOAT depth-stencil cast.
enum class FixtureRole { Colour, Depth, Motion, Alpha };

bool UploadFixture(Gpu& gpu, ID3D12Resource* tex, FixtureRole role, uint32_t frame) {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    const auto d = tex->GetDesc();
    UINT64 size = 0;
    gpu.device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &size);
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = size;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload;
    uint8_t* bytes = nullptr;
    if (FAILED(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr, IID_PPV_ARGS(&upload))) ||
        FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&bytes))))
        return false;
    for (uint32_t y = 0; y < d.Height; ++y) {
        uint8_t* row = bytes + fp.Footprint.RowPitch * y;
        for (uint32_t x = 0; x < d.Width; ++x) {
            const uint32_t code = Hash(x + 419u * y + 611953u * frame);
            const uint16_t half[4] = {static_cast<uint16_t>(0x3000u + (code & 0xbffu)),
                                      static_cast<uint16_t>(0x3000u + ((code >> 7) & 0xbffu)),
                                      static_cast<uint16_t>(0x3000u + ((code >> 13) & 0xbffu)),
                                      static_cast<uint16_t>(0x3400u + ((code >> 21) & 0x7ffu))};
            if (role == FixtureRole::Depth) {
                const float z = 0.1f + 0.8f * static_cast<float>(code & 0xffffu) / 65535.0f;
                std::memcpy(row + 4u * x, &z, 4);
            } else if (role == FixtureRole::Motion) {
                const float m[2] = {3.0f + 8.0f * Noise(frame, x + y * 419u, 0),
                                    2.0f + 6.0f * Noise(frame, x + y * 419u, 1)};
                const uint16_t h[2] = {FgExecHalfBits(m[0]), FgExecHalfBits(m[1])};
                if (d.Format == DXGI_FORMAT_R32G32_FLOAT) std::memcpy(row + 8u * x, m, 8);
                else std::memcpy(row + 4u * x, h, 4);
            } else if (d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                std::memcpy(row + 8u * x, half, 8);
            } else if (d.Format == DXGI_FORMAT_R8_UNORM) {
                row[x] = static_cast<uint8_t>(code >> 24u);
            } else {
                const uint32_t packed = d.Format == DXGI_FORMAT_R11G11B10_FLOAT
                    ? ((half[0] >> 4u) | ((half[1] >> 4u) << 11u) | ((half[2] >> 5u) << 22u)) : code;
                std::memcpy(row + 4u * x, &packed, 4);
            }
        }
    }
    upload->Unmap(0, nullptr);
    if (FAILED(gpu.alloc->Reset()) || FAILED(gpu.list->Reset(gpu.alloc.Get(), nullptr))) return false;
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {tex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST};
    gpu.list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = tex;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = fp;
    gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    gpu.list->ResourceBarrier(1, &b);
    if (FAILED(gpu.list->Close())) return false;
    ID3D12CommandList* lists[] = {gpu.list.Get()};
    gpu.queue->ExecuteCommandLists(1, lists);
    return gpu.Wait();
}

bool SlotsMatch(Gpu& gpu, const ExecutorSlots& actual, const ExecutorSlots& expected, uint32_t count = 2) {
    // Read both sets of textures through identical footprints and compare only the texel bytes.
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[8][2] = {};
    UINT64 row_bytes[8] = {};
    UINT rows[8] = {};
    for (uint32_t i = 0; i < count; ++i) {
        const auto d = actual.textures[i]->GetDesc();
        for (auto& dest : fp[i]) {
            UINT64 size = 0;
            gpu.device->GetCopyableFootprints(&d, 0, 1, (bd.Width + 511u) & ~511ull, &dest,
                                               &rows[i], &row_bytes[i], &size);
            bd.Width = dest.Offset + size;
        }
    }
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> readback;
    if (FAILED(gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&readback))) ||
        FAILED(gpu.alloc->Reset()) || FAILED(gpu.list->Reset(gpu.alloc.Get(), nullptr)))
        return false;
    for (uint32_t i = 0; i < count; ++i) {
        ID3D12Resource* srcs[] = {actual.textures[i].Get(), expected.textures[i].Get()};
        for (uint32_t j = 0; j < 2; ++j) {
            D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
            src.pResource = srcs[j];
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = readback.Get();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = fp[i][j];
            gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    }
    if (FAILED(gpu.list->Close())) return false;
    ID3D12CommandList* lists[] = {gpu.list.Get()};
    gpu.queue->ExecuteCommandLists(1, lists);
    if (!gpu.Wait()) return false;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    const auto* bytes = static_cast<const uint8_t*>(mapped);
    bool exact = true;
    for (uint32_t i = 0; i < count; ++i)
        for (uint32_t y = 0; y < rows[i] && exact; ++y)
            exact = std::memcmp(bytes + fp[i][0].Offset + y * fp[i][0].Footprint.RowPitch,
                                bytes + fp[i][1].Offset + y * fp[i][1].Footprint.RowPitch,
                                static_cast<size_t>(row_bytes[i])) == 0;
    const D3D12_RANGE none = {0, 0};
    readback->Unmap(0, &none);
    return exact;
}

void TestInputSrvPaths(Gpu& gpu, const std::string& dxil, const std::string& reference, const FgExecWeights& weights) {
    g_case = "gpu-input-srv-formats";
    std::ifstream shader(dxil + "/fg_lite_mvec_srv.dxil", std::ios::binary);
    if (!shader) {
        CHECK(reference.empty());
        return;
    }
    struct FormatCase {
        DXGI_FORMAT colour, alpha, motion, slot;
        FgPreset preset;
        uint32_t height;
    };
    const FormatCase cases[] = {
        {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT,
         DXGI_FORMAT_R16G16B16A16_FLOAT, FgPreset::B, 89},
        {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R32G32_FLOAT,
         DXGI_FORMAT_R8G8B8A8_UNORM, FgPreset::C, 91},
        {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16_FLOAT,
         DXGI_FORMAT_R8G8B8A8_UNORM, FgPreset::C, 93},
        {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R32G32_FLOAT,
         DXGI_FORMAT_B8G8R8A8_UNORM, FgPreset::A, 95},
        {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16_FLOAT,
         DXGI_FORMAT_R10G10B10A2_UNORM, FgPreset::C, 97},
        // the Full path packs the slot rows in the output pyramid (RGBA8 and RGB10A2 slots, odd widths)
        {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16G16_FLOAT,
         DXGI_FORMAT_R8G8B8A8_UNORM, FgPreset::A, 99},
        {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R32G32_FLOAT,
         DXGI_FORMAT_R10G10B10A2_UNORM, FgPreset::B, 101},
        {DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R32G32_FLOAT,
         DXGI_FORMAT_R16G16B16A16_FLOAT, FgPreset::C, 1441},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16G16_FLOAT,
         DXGI_FORMAT_R16G16B16A16_FLOAT, FgPreset::C, 1443},
    };
    for (const auto& c : cases) {
        Game game;
        game.gpu = &gpu;
        game.width = 131;
        game.height = c.height;
        game.colour_state = game.other_state = D3D12_RESOURCE_STATE_COMMON;
        game.colour = game.Tex(c.colour);
        game.hudless = game.Tex(c.colour);
        game.ui = game.Tex(c.alpha);
        game.depth = game.Tex(DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        game.motion = game.Tex(c.motion);
        const bool ready = game.colour && game.hudless && game.ui && game.depth && game.motion;
        CHECK(ready);
        if (!ready) continue;
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
        dsv.Format = DXGI_FORMAT_D32_FLOAT;
        dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        D3D12_DESCRIPTOR_HEAP_DESC dh = {};
        dh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dh.NumDescriptors = 1;
        ComPtr<ID3D12DescriptorHeap> heap;
        if (FAILED(gpu.device->CreateDescriptorHeap(&dh, IID_PPV_ARGS(&heap)))) {
            CHECK(false);
            return;
        }
        gpu.device->CreateDepthStencilView(game.depth.Get(), &dsv, heap->GetCPUDescriptorHandleForHeapStart());
        for (int i = 0; i < 16; i += 5) game.clip_to_prev[i] = game.prev_to_clip[i] = 1.0f;

        ExecutorSlots test_slots, ref_slots;
        FgD3d12ExecutorDesc ed;
        ed.dxil_dir = dxil;
        ed.weights = &weights;
        ed.enable_full = c.preset != FgPreset::C;
        FgD3d12Executor test(ed);
        ed.dxil_dir = reference.empty() ? dxil : reference;
        ed.keep_outputs = true;
        FgD3d12Executor ref(ed);
        FgCreateDesc create = CpuCreate(game.width, game.height);
        create.device = gpu.device.Get();
        create.queue = gpu.queue.Get();
        create.preset = c.preset;
        auto ok = [&](FgD3d12Executor& exec, FgExecResult status) {
            if (status != FgExecResult::Ok)
                std::printf("    input SRV test: %s\n", exec.LastWhy().c_str());
            CHECK(status == FgExecResult::Ok);
            return status == FgExecResult::Ok;
        };
        for (auto* slots : {&test_slots, &ref_slots})
            for (auto& tex : slots->textures) tex = game.Tex(c.slot);
        if (!ok(test, test.Configure(create, 3)) || !ok(ref, ref.Configure(create, 3))) return;
        if (!test.BindSlotTarget(&test_slots) || !ref.BindSlotTarget(&ref_slots)) {
            CHECK(false);
            return;
        }
        for (uint64_t token = 1; token <= 2; ++token) {
            const auto f = static_cast<uint32_t>(token);
            const bool uploaded = UploadFixture(gpu, game.colour.Get(), FixtureRole::Colour, f) &&
                                  UploadFixture(gpu, game.hudless.Get(), FixtureRole::Colour, f + 2) &&
                                  UploadFixture(gpu, game.ui.Get(), FixtureRole::Alpha, f + 4) &&
                                  UploadFixture(gpu, game.depth.Get(), FixtureRole::Depth, f) &&
                                  UploadFixture(gpu, game.motion.Get(), FixtureRole::Motion, f);
            CHECK(uploaded);
            if (!uploaded) return;
            auto packet = game.Packet();
            packet.frame_token = token;
            packet.motion_scale_x = 8.0f;
            packet.motion_scale_y = 8.0f;
            if (c.colour != DXGI_FORMAT_R8G8B8A8_UNORM_SRGB && c.colour != DXGI_FORMAT_R10G10B10A2_UNORM)
                packet.optional[kFgOptHudless].rect = {1, 3, game.width - 2, game.height - 5};
            packet.optional[kFgOptUiAlpha].rect = {2, 1, game.width - 4, game.height - 2};
            FgSharedDispatchDesc shared;
            shared.frame_token = token;
            shared.preset = c.preset;
            shared.has_previous = token == 2;
            shared.generates = true;
            shared.packet = &packet;
            FgSharedResult result;
            const uint64_t captures = test.Counters().captures;
            if (!ok(test, test.RecordSharedGameFrame(shared, &result)) ||
                !ok(ref, ref.RecordSharedGameFrame(shared, &result))) return;
            CHECK(test.Counters().captures == captures);
            FgIndexDispatchDesc gen;
            gen.frame_token = token;
            gen.index = gen.generated_count = 1;
            gen.is_last_index = true;
            gen.preset = c.preset;
            gen.t = 0.5f;
            gen.slot.reserved = true;
            FgRealDispatchDesc real;
            real.frame_token = token;
            real.slot.slot = 1;
            real.slot.reserved = true;
            FgReadyFence fence;
            for (auto* exec : {&test, &ref}) {
                if (!ok(*exec, exec->RecordGeneratedIndex(gen, &fence)) ||
                    !ok(*exec, exec->RecordRealEndpoint(real, &fence))) return;
            }
            const bool idle = test.WaitIdle(5000) && ref.WaitIdle(5000);
            CHECK(idle);
            if (!idle) return;
            const bool exact = SlotsMatch(gpu, test_slots, ref_slots);
            if (!exact)
                std::printf("    input SRV mismatch: colour %u, alpha %u, motion %u, slot %u, frame %llu\n",
                            c.colour, c.alpha, c.motion, c.slot, static_cast<unsigned long long>(token));
            CHECK(exact);
        }
    }
    const int64_t errors = gpu.DebugErrors();
    if (errors >= 0) CHECK(errors == 0);
}

// Turbo v2: every lever on (batched indices, hoisted per-index work on the second queue, the split shared work, the fused
// and retiled kernels) must give the same bytes as every lever off, in every generated slot and the real one, over several
// game frames (history, swaps), the presets A / B / C and N = 1 .. 5. Both executors read the same game textures and run
// the same kernel folder (`dxil`: the Turbo kernels sit beside the ones they replace).
// wave1b_only: both executors run the Wave 1 levers (batch, hoist, dualq, fuse), only the Wave 1B levers (fuse2) differ:
// the fused kernels against exactly the kernels they replace.
void TestTurboExact(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, bool wave1b_only = false) {
    g_case = wave1b_only ? "gpu-turbo-exact-wave1b" : "gpu-turbo-exact";
    {
        std::ifstream probe(dxil + "/fg_lite_interm_both.dxil", std::ios::binary);
        if (!probe) std::printf("  turbo: the Turbo v2 kernels are not in %s (batching, hoisting and the split only)\n", dxil.c_str());
        std::ifstream probe2(dxil + "/fg_net2_mpair_res_c64.dxil", std::ios::binary);
        if (!probe2) std::printf("  turbo: the Wave 1B kernels are not in %s (fuse2 has nothing to run)\n", dxil.c_str());
    }
    struct Case {
        FgPreset preset;
        uint32_t n, width, height;
        DXGI_FORMAT colour;
    };
    const Case cases[] = {
        {FgPreset::C, 1, kW, kH, DXGI_FORMAT_B8G8R8A8_UNORM},   {FgPreset::C, 2, 131, 89, DXGI_FORMAT_B8G8R8A8_UNORM},
        {FgPreset::C, 5, kW, kH, DXGI_FORMAT_R16G16B16A16_FLOAT}, {FgPreset::A, 1, kW, kH, DXGI_FORMAT_B8G8R8A8_UNORM},
        {FgPreset::A, 3, 131, 89, DXGI_FORMAT_B8G8R8A8_UNORM},   {FgPreset::A, 5, kW, kH, DXGI_FORMAT_B8G8R8A8_UNORM},
        {FgPreset::B, 2, kW, kH, DXGI_FORMAT_R16G16B16A16_FLOAT},
    };
    for (const Case& c : cases) {
        char name[96];
        std::snprintf(name, sizeof(name), "preset %c N=%u %ux%u", c.preset == FgPreset::A ? 'A' : c.preset == FgPreset::B ? 'B' : 'C',
                      c.n, c.width, c.height);
        Game game;
        const bool ready = game.Init(&gpu, c.colour, c.width, c.height) && game.Upload(1, false, true);
        CHECK(ready);
        if (!ready) return;
        FgD3d12ExecutorDesc ed;
        ed.dxil_dir = dxil;
        ed.weights = &weights;
        ed.enable_full = c.preset != FgPreset::C;
        ed.batch = ed.hoist = ed.dualq = ed.fuse = 1;
        ed.fuse2 = g_turbo_fuse2;   // 63: every Wave 1B lever (default of the test), --turbo-fuse2 1: the executor's default levers
        FgD3d12Executor test(ed);
        ed.batch = ed.hoist = ed.dualq = ed.fuse = wave1b_only ? 1 : 0;
        ed.fuse2 = 0;
        FgD3d12Executor ref(ed);
        FgCreateDesc create = CpuCreate(c.width, c.height);
        create.device = gpu.device.Get();
        create.queue = gpu.queue.Get();
        create.preset = c.preset;
        create.max_generated = c.n;
        const auto ok = [&](FgD3d12Executor& exec, FgExecResult status) {
            if (status != FgExecResult::Ok) std::printf("    turbo %s: %s (%s)\n", name, FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str());
            CHECK(status == FgExecResult::Ok);
            return status == FgExecResult::Ok;
        };
        ExecutorSlots test_slots, ref_slots;
        for (uint32_t i = 0; i <= c.n; ++i) {
            test_slots.textures[i] = game.Tex(c.colour);
            ref_slots.textures[i] = game.Tex(c.colour);
        }
        if (!ok(test, test.Configure(create, 3)) || !ok(ref, ref.Configure(create, 3))) return;
        if (!test.BindSlotTarget(&test_slots) || !ref.BindSlotTarget(&ref_slots)) {
            CHECK(false);
            return;
        }
        std::printf("  turbo%s %s: batch %d hoist %d dualq %d banks %u fuse2 0x%x (reference: batch %d hoist %d dualq %d fuse2 0x%x)\n",
                    wave1b_only ? " (wave 1B)" : "", name, test.BatchActive(), test.HoistActive(), test.DualQueueActive(),
                    test.ScratchBanks(), test.Fuse2Active(), ref.BatchActive(), ref.HoistActive(), ref.DualQueueActive(),
                    ref.Fuse2Active());
        CHECK(ref.Fuse2Active() == 0);
        if (!wave1b_only) CHECK(!ref.BatchActive() && !ref.HoistActive() && !ref.DualQueueActive());
        bool exact_all = true;
        for (uint64_t token = 1; token <= 4; ++token) {
            // (the textures already hold frame 1's state from the upload above: `first` would claim COMMON again)
            if (!game.Upload(static_cast<uint32_t>(token), false, false)) {
                CHECK(false);
                return;
            }
            FgFramePacket packet = game.Packet();
            packet.frame_token = token;
            packet.motion_scale_x = 8.0f;
            packet.motion_scale_y = 8.0f;
            FgSharedDispatchDesc shared;
            shared.frame_token = token;
            shared.preset = c.preset;
            shared.has_previous = token > 1;
            shared.generated_count = c.n;
            shared.generates = token > 1;
            shared.packet = &packet;
            FgSharedResult result;
            if (!ok(test, test.RecordSharedGameFrame(shared, &result)) ||
                !ok(ref, ref.RecordSharedGameFrame(shared, &result)))
                return;
            FgReadyFence fences[kFgMaxGenerated + 1];
            if (token > 1) {
                FgIndexDispatchDesc gens[kFgMaxGenerated];
                for (uint32_t k = 0; k < c.n; ++k) {
                    gens[k].frame_token = token;
                    gens[k].index = k + 1;
                    gens[k].generated_count = c.n;
                    gens[k].is_last_index = k + 1 == c.n;
                    gens[k].preset = c.preset;
                    gens[k].t = FgIndexTime(k + 1, c.n);
                    gens[k].slot.slot = k;
                    gens[k].slot.reserved = true;
                }
                uint32_t done = 0;
                const FgExecResult batched = c.n > 1 ? test.RecordGeneratedBatch(gens, c.n, fences, &done)
                                                     : test.RecordGeneratedIndex(gens[0], &fences[0]);
                if (!ok(test, batched)) return;
                for (uint32_t k = 0; k < c.n; ++k)
                    if (!ok(ref, ref.RecordGeneratedIndex(gens[k], &fences[k]))) return;
            }
            FgRealDispatchDesc real;
            real.frame_token = token;
            real.slot.slot = c.n;
            real.slot.reserved = true;
            if (!ok(test, test.RecordRealEndpoint(real, &fences[c.n])) ||
                !ok(ref, ref.RecordRealEndpoint(real, &fences[c.n])))
                return;
            const bool idle = test.WaitIdle(30000) && ref.WaitIdle(30000);
            CHECK(idle);
            if (!idle) return;
            const bool exact = SlotsMatch(gpu, test_slots, ref_slots, c.n + 1);
            if (!exact) std::printf("    turbo %s: slots differ at game frame %llu\n", name, static_cast<unsigned long long>(token));
            exact_all = exact_all && exact;
            CHECK(exact);
        }
        if (exact_all) std::printf("    turbo %s: 4 game frames, 0 byte differences\n", name);
    }
    const int64_t errors = gpu.DebugErrors();
    if (errors >= 0) CHECK(errors == 0);
}

struct BenchOptions {
    uint32_t gen = 1, warmup = 8, timed = 30;
    std::string presets = "AC";
    bool kprof = false;
    int batch = -1, hoist = -1, dualq = -1, fuse = -1, fuse2 = -1;  // -1: the environment (default on)
};

bool RunBench(Gpu& gpu, const std::string& dxil, const FgExecWeights& weights, FgPreset preset,
              const BenchOptions& bo) {
    constexpr uint32_t width = 2560, height = 1440;
    constexpr uint32_t timeout = 30000;
    const uint32_t gen_count = bo.gen, warmup = bo.warmup, timed = bo.timed;
    const char label = preset == FgPreset::A ? 'A' : 'C';
    Game game;
    const DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
    if (!game.Init(&gpu, format, width, height) || !game.Upload(0, false, true)) return false;
    ExecutorSlots slots;
    for (uint32_t i = 0; i <= gen_count; ++i) {  // slot i - 1 holds index i, slot gen_count the real endpoint
        slots.textures[i] = game.Tex(format);
        if (!slots.textures[i]) return false;
    }

    FgD3d12ExecutorDesc ed;
    ed.dxil_dir = dxil;
    ed.weights = &weights;
    ed.enable_full = preset == FgPreset::A;
    ed.timing = true;
    ed.kernel_profile = bo.kprof;
    ed.batch = bo.batch;
    ed.hoist = bo.hoist;
    ed.dualq = bo.dualq;
    ed.fuse = bo.fuse;
    ed.fuse2 = bo.fuse2;
    FgD3d12Executor exec(ed);
    const auto ok = [&](FgExecResult s) {
        if (s == FgExecResult::Ok) return true;
        std::printf("bench %c: %s (%s)\n", label, FgExecRefusalName(exec.LastRefusal()), exec.LastWhy().c_str());
        return false;
    };
    FgCreateDesc create = CpuCreate(width, height);
    create.device = gpu.device.Get();
    create.queue = gpu.queue.Get();
    create.preset = preset;
    create.max_generated = gen_count;
    if (!ok(exec.Configure(create, 3)) || !exec.BindSlotTarget(&slots)) return false;

    FgFramePacket packet = game.Packet();

    FgExecTimings timing;
    std::vector<FgExecKernelProfileRow> profile;
    for (uint32_t i = 0; i < warmup + timed; ++i) {
        packet.frame_token = i + 1;
        FgSharedDispatchDesc shared;
        shared.frame_token = packet.frame_token;
        shared.preset = preset;
        shared.has_previous = i != 0;
        shared.generated_count = gen_count;
        shared.generates = true;
        shared.packet = &packet;
        FgSharedResult result;
        if (!ok(exec.RecordSharedGameFrame(shared, &result))) return false;
        FgIndexDispatchDesc gens[kFgMaxGenerated];
        FgReadyFence ready[kFgMaxGenerated + 1];
        for (uint32_t k = 0; k < gen_count; ++k) {
            FgIndexDispatchDesc& gen = gens[k];
            gen.frame_token = packet.frame_token;
            gen.index = k + 1;
            gen.generated_count = gen_count;
            gen.is_last_index = k + 1 == gen_count;
            gen.preset = preset;
            gen.t = FgIndexTime(k + 1, gen_count);
            gen.slot.slot = k;
            gen.slot.reserved = true;
        }
        if (gen_count > 1) {  // the executor falls back to one index at a time when batching is off
            uint32_t done = 0;
            if (!ok(exec.RecordGeneratedBatch(gens, gen_count, ready, &done)) || done != gen_count) return false;
        } else if (!ok(exec.RecordGeneratedIndex(gens[0], &ready[0]))) {
            return false;
        }
        FgRealDispatchDesc real;
        real.frame_token = packet.frame_token;
        real.slot.slot = gen_count;
        real.slot.reserved = true;
        if (!ok(exec.RecordRealEndpoint(real, &ready[gen_count]))) return false;
        if (i + 1 == warmup) {
            if (!exec.TakeTimings(&timing, timeout)) {
                std::printf("bench %c: warmup timestamps unavailable\n", label);
                return false;
            }
            if (bo.kprof && !exec.TakeKernelProfile(&profile, timeout)) {
                std::printf("bench %c: warmup kernel profile unavailable\n", label);
                return false;
            }
        }
    }
    if (!exec.TakeTimings(&timing, timeout)) {
        std::printf("bench %c: timed timestamps unavailable\n", label);
        return false;
    }
    const char* stages[] = {"capture", "shared_pre_net1", "net1", "shared_post_net1", "gen_pre_net2",
                            "net2", "out_pyramid", "pack_copy", "real_endpoint"};
    static_assert(sizeof(stages) / sizeof(*stages) == static_cast<size_t>(FgExecStage::Count));
    std::printf("bench preset %c %ux (gen %u) %ux%u (%u warmup, %u timed, mean GPU ms per game frame)\n", label,
                gen_count + 1, gen_count, width, height, warmup, timed);
    std::printf("  levers: batch %s, hoist %s, dualq %s, scratch banks %u, fuse2 mask 0x%x\n", exec.BatchActive() ? "on" : "off",
                exec.HoistActive() ? "on" : "off", exec.DualQueueActive() ? "on" : "off", exec.ScratchBanks(),
                exec.Fuse2Active());
    double shared = 0.0, gen = 0.0;
    for (size_t i = 0; i < timing.ms.size(); ++i) {
        const double mean = timing.ms[i] / timed;
        std::printf("  %-18s %.6f\n", stages[i], mean);
        if (i <= static_cast<size_t>(FgExecStage::SharedPostNet1)) shared += mean;
        else if (i <= static_cast<size_t>(FgExecStage::PackCopy)) gen += mean;
    }
    // The stage totals sum GPU execution intervals: work that two queues overlap is counted twice and queue waits
    // are not counted. The span and busy lines are measured on one clock over the whole timed window.
    std::printf("  %-18s %.6f\n  %-18s %.6f\n  %-18s %.6f\n", "total_shared", shared, "total_gen", gen,
                "sum_of_stages", shared + gen + timing.ms[static_cast<size_t>(FgExecStage::RealEndpoint)] / timed);
    std::printf("  %-18s %.6f   (first start to last end of the window / frames, idle gaps included)\n",
                "gpu_span_per_frame", timing.span_ms / timed);
    std::printf("  %-18s %.6f   (time at least one queue ran / frames; %u recordings)\n", "gpu_busy_per_frame",
                timing.busy_ms / timed, timing.recordings);
    if (bo.kprof) {
        if (!exec.TakeKernelProfile(&profile, timeout)) {
            std::printf("bench %c: kernel profile unavailable\n", label);
            return false;
        }
        std::printf("  per-launch profile (kernel#ordinal, workgroups per call, calls and mean us per game frame):\n");
        double total = 0.0;
        for (const FgExecKernelProfileRow& row : profile) {
            const double calls = static_cast<double>(row.calls) / timed;
            const double wg = row.calls ? static_cast<double>(row.groups) / static_cast<double>(row.calls) : 0.0;
            const double per_call = row.calls ? row.ms * 1000.0 / static_cast<double>(row.calls) : 0.0;
            std::printf("    %-34s #%-2u wg %8.0f  x%-5.2f %9.2f us/call %9.2f us/frame\n", row.name.c_str(),
                        row.ordinal, wg, calls, per_call, row.ms * 1000.0 / timed);
            total += row.ms * 1000.0 / timed;
        }
        std::printf("    %-34s %55.2f us/frame\n", "(sum of launches)", total);
    }
    return gpu.DebugErrors() <= 0;
}

int main(int argc, char** argv) {
    bool gpu = false, bench = false;
    std::string dxil, reference_dxil, scratch = ".", only;
    BenchOptions bo;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) gpu = true;
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        // --gen N: N generated images per game frame (1..5: 2x .. 6x), really recorded and timed
        if (std::strcmp(argv[i], "--gen") == 0 && i + 1 < argc) bo.gen = static_cast<uint32_t>(std::atoi(argv[++i]));
        if (std::strcmp(argv[i], "--timed") == 0 && i + 1 < argc) bo.timed = static_cast<uint32_t>(std::atoi(argv[++i]));
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) bo.warmup = static_cast<uint32_t>(std::atoi(argv[++i]));
        if (std::strcmp(argv[i], "--preset") == 0 && i + 1 < argc) bo.presets = argv[++i];
        if (std::strcmp(argv[i], "--kprof") == 0) bo.kprof = true;
        // paired A / B levers inside one process (default: the environment, WG_FG_BATCH / _HOIST / _DUALQ = 0 off)
        if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) bo.batch = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--hoist") == 0 && i + 1 < argc) bo.hoist = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--dualq") == 0 && i + 1 < argc) bo.dualq = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--fuse") == 0 && i + 1 < argc) bo.fuse = std::atoi(argv[++i]);
        if (std::strcmp(argv[i], "--fuse2") == 0 && i + 1 < argc) bo.fuse2 = static_cast<int>(std::strtol(argv[++i], nullptr, 0));
        if (std::strcmp(argv[i], "--dxil") == 0 && i + 1 < argc) dxil = argv[++i];
        if (std::strcmp(argv[i], "--reference-dxil") == 0 && i + 1 < argc) reference_dxil = argv[++i];
        if (std::strcmp(argv[i], "--scratch") == 0 && i + 1 < argc) scratch = argv[++i];
        if (std::strcmp(argv[i], "--debug-layer") == 0) g_debug_layer = true;
        if (std::strcmp(argv[i], "--turbo-fuse2") == 0 && i + 1 < argc) g_turbo_fuse2 = static_cast<int>(std::strtol(argv[++i], nullptr, 0));
        if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
    }
    FgExecWeights weights;
    FgExecSyntheticWeights(11, &weights);
    if (bench) {
        Gpu g;
        if (dxil.empty() || !g.Init()) {
            std::printf("bench requires --dxil <dir> and a D3D12 device\n");
            return 1;
        }
        std::printf("gpu: %s\n", g.name.c_str());
        if (bo.gen < 1 || bo.gen > 5) {
            std::printf("--gen takes 1..5\n");
            return 1;
        }
        for (char c : bo.presets) {
            if ((c == 'A' || c == 'a') && !RunBench(g, dxil, weights, FgPreset::A, bo)) return 1;
            if ((c == 'C' || c == 'c') && !RunBench(g, dxil, weights, FgPreset::C, bo)) return 1;
        }
        if (!gpu) return 0;
    }
    TestGeometry();
    TestRegionsAndThresholds();
    TestHalf();
    TestSigTable();
    TestSchedule();
    TestInterleave();
    TestNet2FastPlan();
    TestRefusals(weights);
    TestMotionFormats();
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
                FgNet2Path net2;
                int off_from, off_to;
            };
            const FgNet2Path kF = FgNet2Path::Fast, kE = FgNet2Path::Exact;
            const Cfg cfgs[] = {
                {"gpu-lite-2x-composition", FgPreset::C, 1, false, DXGI_FORMAT_B8G8R8A8_UNORM, 8, 14, kF, -1, -1},
                {"gpu-lite-6x-composition", FgPreset::C, 5, false, DXGI_FORMAT_B8G8R8A8_UNORM, 6, 8, kF, -1, -1},
                {"gpu-full-a-2x-composition", FgPreset::A, 1, false, DXGI_FORMAT_B8G8R8A8_UNORM, -1, 14, kF, -1, -1},
                {"gpu-lite-2x-real-only-composition", FgPreset::C, 1, false, DXGI_FORMAT_B8G8R8A8_UNORM, -1, 12, kF, 4,
                 7},
                {"gpu-lite-2x-dxgi", FgPreset::C, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 14, kF, -1, -1},
                {"gpu-lite-6x-dxgi", FgPreset::C, 5, true, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 14, kF, -1, -1},
                {"gpu-full-a-2x-dxgi", FgPreset::A, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, -1, 14, kF, -1, -1},
                {"gpu-full-b-2x-dxgi", FgPreset::B, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, -1, 14, kF, -1, -1},
                {"gpu-full-a-2x-exact-dxgi", FgPreset::A, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, -1, 10, kE, -1, -1},
                {"gpu-lite-6x-real-only-dxgi", FgPreset::C, 5, true, DXGI_FORMAT_R16G16B16A16_FLOAT, -1, 16, kF, 5,
                 9},
                {"gpu-full-a-2x-real-only-dxgi", FgPreset::A, 1, true, DXGI_FORMAT_R16G16B16A16_FLOAT, -1, 16, kF, 0,
                 4},
            };
            int ran_comp = 0, ran_dxgi = 0, want_comp = 0, want_dxgi = 0;
            for (const Cfg& c : cfgs) {
                (c.dxgi ? want_dxgi : want_comp)++;
                if (!only.empty() && std::string(c.name).find(only) == std::string::npos) continue;
                g_case = c.name;
                std::printf("  %s (net2 %s)\n", c.name, FgNet2PathName(c.net2));
                const RunResult r = RunPipeline(g, dxil, weights, c.preset, c.n, c.dxgi, c.fmt, c.frames, c.menu,
                                                c.net2, c.off_from, c.off_to);
                if (r.ran) (c.dxgi ? ran_dxgi : ran_comp)++;
            }
            g_case = "gpu-backends";
            std::printf("  configurations run: composition %d / %d, dxgi %d / %d\n", ran_comp, want_comp, ran_dxgi,
                        want_dxgi);
            if (only.empty()) CHECK(ran_comp == want_comp && ran_dxgi == want_dxgi);  // every configuration ran
            if (only.empty() || only.find("net2") != std::string::npos) TestNet2Paths(g, dxil, weights);
            if (only.empty() || only.find("copy-paths") != std::string::npos) {
                TestCopyPaths(g, dxil, weights, false);
                TestCopyPaths(g, dxil, weights, true);
            }
            if (only.empty() || only.find("input-srv") != std::string::npos)
                TestInputSrvPaths(g, dxil, reference_dxil, weights);
            if (only.empty() || only.find("turbo") != std::string::npos) {
                TestTurboExact(g, dxil, weights);
                TestTurboExact(g, dxil, weights, true);
            }
            if (only.empty()) TestBrokenKernels(g, dxil, weights, scratch);
        }
    }
    std::printf("fg_d3d12_e2e: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
