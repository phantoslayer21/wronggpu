// SPDX-License-Identifier: GPL-3.0-only
// Tests of the Windows presenter back ends (clean/fg/presenter): the shared display-slot engine, the pacing
// arithmetic, the back-end selector and its refusals, the worker thread and the timer, all on the CPU with synthetic
// values. Nothing here opens a window.
//
//   test_fg_win_presenters          CPU checks (CI)
//   test_fg_win_presenters --gpu    also headless GPU checks: the capability probe with a device, the composition
//                                   presenter on an unbound surface (zero-copy shared displayable slots, shared-fence
//                                   handoff, presents with target times) and the DXGI presenter on a composition swap
//                                   chain bound to nothing. No window is created and nothing reaches the screen.
#include <algorithm>
#include <atomic>
#include <d3d12sdklayers.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "../fg_presenter_factory.h"
#include "../fg_present_engine.h"
#include "../fg_target_scheduler.h"
#include "../fg_win_common.h"

using namespace wgfg;
using Microsoft::WRL::ComPtr;

namespace {

int g_checks = 0;
int g_failures = 0;
bool g_debug_layer = false;
const char* g_case = "";

void Check(bool ok, const char* what, int line) {
    static const char* last = nullptr;
    if (last != g_case) {
        last = g_case;
        std::printf("case %s\n", g_case);
    }
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL [%s] line %d: %s\n", g_case, line, what);
    }
}
#define CHECK(x) Check((x), #x, __LINE__)

constexpr FgTicks kMs = 10000;
constexpr FgTicks kRefresh360 = 27778;  // 360 Hz panel

// --- synthetic pieces ----------------------------------------------------------------------------------------------

// Fences whose completed values the test sets.
class ScriptFences final : public IFgFenceSource {
public:
    std::uint32_t Create() {
        values_.push_back(0);
        return static_cast<std::uint32_t>(values_.size());
    }
    void Set(std::uint32_t f, std::uint64_t v) {
        if (f >= 1 && f <= values_.size()) values_[f - 1] = v;
    }
    bool FenceValid(std::uint32_t f) const override { return f >= 1 && f <= values_.size(); }
    std::uint64_t FenceCompleted(std::uint32_t f) const override { return FenceValid(f) ? values_[f - 1] : 0; }

private:
    std::vector<std::uint64_t> values_;
};

FgIntervalPlan MakePlan(std::uint64_t epoch, std::uint64_t id, std::uint64_t frame, FgTicks t_base, FgTicks period,
                        std::uint32_t n, FgTicks margin = kMs) {
    FgIntervalPlan p;
    p.epoch = epoch;
    p.interval_id = id;
    p.real_frame_id = frame;
    p.requested_count = n;
    p.generated_count = n;
    p.real_only_reason = n == 0 ? FgRealOnlyReason::NotPrimed : FgRealOnlyReason::None;
    p.t_base = t_base;
    p.period = period;
    p.margin = margin;
    p.target_count = n + 1;
    FgBuildTargets(t_base, period, n, margin, p.targets);
    return p;
}

FgEngineConfig EngineConfig(std::uint32_t slots = kFgMaxGenerated + 4, FgTicks refresh = kRefresh360) {
    FgEngineConfig c;
    c.probe = FgPresenterProbe{FgPresenterBackend::Fake, true, true, kFgMaxGenerated, FgStatus::Ok};
    c.slot_count = slots;
    c.refresh_period = refresh;
    return c;
}

// A display path that shows outbox presents at their target times (no earlier than `now`) and releases a superseded
// buffer one step later, the way an available event follows a flip. It also checks the engine's ownership rule: a
// slot the display holds is never handed to the executor.
struct SimDisplay {
    FgPresentEngine* engine;
    std::vector<FgEnginePresent> queue;  // submitted, not yet shown
    std::vector<std::uint32_t> owned;    // slots the display references (queued, on screen, being released)
    std::vector<std::uint64_t> shown_ids;
    std::vector<FgTicks> shown_targets;
    std::vector<std::uint32_t> retired_pending;  // superseded / skipped, released at the next step
    bool ordered = true;
    std::uint64_t last_id = 0;

    void Own(std::uint32_t s) {
        for (std::uint32_t o : owned)
            if (o == s) return;
        owned.push_back(s);
    }
    void Disown(std::uint32_t s) {
        for (std::size_t i = 0; i < owned.size(); ++i)
            if (owned[i] == s) {
                owned.erase(owned.begin() + static_cast<std::ptrdiff_t>(i));
                return;
            }
    }
    bool Owns(std::uint32_t s) const {
        for (std::uint32_t o : owned)
            if (o == s) return true;
        return false;
    }
    void Submit() {
        std::vector<FgEnginePresent> out;
        engine->TakeOutbox(&out);
        for (const FgEnginePresent& p : out) {
            if (p.present_id <= last_id) ordered = false;
            last_id = p.present_id;
            queue.push_back(p);
            Own(p.slot);
        }
    }
    // Cancel on the display: drop queued presents with id >= from (their buffers are released next step).
    void Cancel(std::uint64_t from) {
        if (from == 0) return;
        std::vector<FgEnginePresent> keep;
        for (const FgEnginePresent& p : queue) {
            if (p.present_id >= from) retired_pending.push_back(p.slot);
            else keep.push_back(p);
        }
        queue.swap(keep);
    }
    void Step(FgTicks now) {
        // release last step's retired buffers (available event)
        for (std::uint32_t s : retired_pending) {
            if (engine->SlotState(s) == FgEngineSlotState::Retired) {
                engine->OnSlotReleased(s, false);
                Disown(s);
            }
        }
        retired_pending.clear();
        std::size_t i = 0;
        int shown_slot = -1;
        for (; i < queue.size(); ++i) {
            const FgEnginePresent& p = queue[i];
            if (p.target > now) break;
            if (engine->SlotPresentId(p.slot) != p.present_id) continue;  // cancelled on the engine side
            for (std::size_t k = 0; k < engine->SlotCount(); ++k)
                if (engine->SlotState(static_cast<std::uint32_t>(k)) == FgEngineSlotState::Displayed)
                    shown_slot = static_cast<int>(k);
            engine->OnPresentDisplayed(p.present_id, p.target);
            shown_ids.push_back(p.present_id);
            shown_targets.push_back(p.target);
            if (shown_slot >= 0 && static_cast<std::uint32_t>(shown_slot) != p.slot)
                retired_pending.push_back(static_cast<std::uint32_t>(shown_slot));
        }
        queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(i));
    }
};

// --- pacing arithmetic ---------------------------------------------------------------------------------------------

void TestPacingMath() {
    g_case = "pacing-math";
    FgRefreshGrid grid{1000, kRefresh360};
    FgTicks vb = 0, sub = 0;
    CHECK(FgSnapToRefreshGrid(1000 + 3 * kRefresh360, grid, &vb, &sub) == FgStatus::Ok);
    CHECK(vb == 1000 + 3 * kRefresh360 && sub == vb - kRefresh360 / 2);
    // just below / above a half period: the nearest vblank, never the other side
    CHECK(FgSnapToRefreshGrid(1000 + 3 * kRefresh360 + kRefresh360 / 2 - 1, grid, &vb, &sub) == FgStatus::Ok);
    CHECK(vb == 1000 + 3 * kRefresh360);
    CHECK(FgSnapToRefreshGrid(1000 + 3 * kRefresh360 + kRefresh360 / 2, grid, &vb, &sub) == FgStatus::Ok);
    CHECK(vb == 1000 + 4 * kRefresh360);  // tie goes to the later vblank
    // a target before the origin
    CHECK(FgSnapToRefreshGrid(1000 - 2 * kRefresh360 + 10, grid, &vb, &sub) == FgStatus::Ok);
    CHECK(vb == 1000 - 2 * kRefresh360);
    // jitter of up to just under half a refresh around a vblank-centred target never changes the vblank
    for (FgTicks j = -(kRefresh360 / 2) + 1; j < kRefresh360 / 2; j += 997) {
        CHECK(FgSnapToRefreshGrid(1000 + 7 * kRefresh360 + j, grid, &vb, &sub) == FgStatus::Ok);
        CHECK(vb == 1000 + 7 * kRefresh360);
    }
    FgRefreshGrid none;
    CHECK(FgSnapToRefreshGrid(5, none, &vb, &sub) == FgStatus::InvalidArgument);
    CHECK(FgSnapToRefreshGrid(5, grid, nullptr, &sub) == FgStatus::InvalidArgument);
    FgRefreshGrid distant{std::numeric_limits<FgTicks>::max() - 10, kRefresh360};
    CHECK(FgSnapToRefreshGrid(std::numeric_limits<FgTicks>::min() + 10, distant, &vb, &sub) == FgStatus::Overflow);

    CHECK(FgRefreshMultiple(3 * kRefresh360, kRefresh360, kRefresh360 / 8) == 3);
    CHECK(FgRefreshMultiple(3 * kRefresh360 + 100, kRefresh360, kRefresh360 / 8) == 3);
    CHECK(FgRefreshMultiple(41667, kRefresh360, kRefresh360 / 8) == 0);  // 240 fps on 360 Hz: 1.5 refreshes
    CHECK(FgRefreshMultiple(33333, kRefresh360, kRefresh360 / 8) == 0);  // 300 fps on 360 Hz: 1.2 refreshes
    CHECK(FgRefreshMultiple(kRefresh360, kRefresh360, 0) == 1);
    CHECK(FgRefreshMultiple(0, kRefresh360, 10) == 0 && FgRefreshMultiple(100, 0, 10) == 0);

    // 60 fps game frames on a 360 Hz panel: 2x (3 refreshes), 3x (2), 6x (1) fit; 4x and 5x do not
    const FgTicks p60 = 166667;
    CHECK(FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 1), kRefresh360));
    CHECK(FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 2), kRefresh360));
    CHECK(!FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 3), kRefresh360));
    CHECK(!FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 4), kRefresh360));
    CHECK(FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 5), kRefresh360));
    CHECK(FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 0), kRefresh360));  // real-only always fits
    CHECK(!FgPlanFitsRefresh(MakePlan(1, 1, 1, 0, p60, 1), 0));

    CHECK(FgTimerDue(100 * kMs, 50 * kMs, 10 * kMs) == -40 * kMs);
    CHECK(FgTimerDue(100 * kMs, 95 * kMs, 10 * kMs) == 0);
    CHECK(FgTimerDue(100 * kMs, 200 * kMs, 0) == 0);
    CHECK(FgMinSlotsFor(1) == 4 && FgMinSlotsFor(5) == 8);
}

// --- engine --------------------------------------------------------------------------------------------------------

void TestEngineBasicFlow() {
    g_case = "engine-flow";
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    FgPresentEngine e(EngineConfig(), &fences);
    CHECK(e.State() == FgPresenterState::Detached);
    FgIntervalReservation res;
    CHECK(e.BeginInterval(MakePlan(1, 1, 1, 0, 166667, 1), 0, &res) == FgStatus::WrongState);
    CHECK(e.Attach(1) == FgStatus::Ok);
    CHECK(e.Attach(1) == FgStatus::WrongState);

    const FgTicks t0 = 1000 * kMs;
    // real-only first (priming), then a 3x interval
    FgIntervalPlan r0 = MakePlan(1, 1, 1, t0, 166667, 0);
    CHECK(e.PresentRealOnly(r0, FgRealImage{1, 1, FgReadyFence{f, 1}}) == FgStatus::Ok);
    e.Pump(t0);
    CHECK(e.OutboxEmpty());  // fence not done: nothing goes out
    fences.Set(f, 1);
    e.Pump(t0);
    std::vector<FgEnginePresent> out;
    e.TakeOutbox(&out);
    CHECK(out.size() == 1 && out[0].real && out[0].target == r0.targets[0].target && out[0].present_id == 1);

    FgIntervalPlan p = MakePlan(1, 2, 2, r0.targets[0].target, 166667, 2);
    CHECK(e.BeginInterval(p, t0, &res) == FgStatus::Ok);
    CHECK(res.count == 3 && res.slots[0].reserved && res.slots[1].reserved && res.slots[2].reserved &&
          res.slots[2].real);
    // all three slots differ and none is the one queued for the real-only image
    CHECK(res.slots[0].slot != res.slots[1].slot && res.slots[1].slot != res.slots[2].slot &&
          res.slots[0].slot != res.slots[2].slot);
    for (int i = 0; i < 3; ++i) CHECK(res.slots[i].slot != out[0].slot);
    FgIntervalSubmission sub;
    sub.epoch = 1;
    sub.interval_id = 2;
    sub.count = 3;
    for (std::uint32_t i = 0; i < 3; ++i) {
        sub.ready[i] = FgReadyFence{f, 2 + i};
        sub.produced[i] = true;
    }
    CHECK(e.SubmitInterval(sub, t0) == FgStatus::Ok);
    out.clear();
    e.TakeOutbox(&out);
    CHECK(out.empty());
    fences.Set(f, 2);  // index 1 ready, index 2 not: order is kept
    e.Pump(t0);
    e.TakeOutbox(&out);
    CHECK(out.size() == 1 && out[0].index == 1 && !out[0].real);
    fences.Set(f, 4);
    e.Pump(t0);
    e.TakeOutbox(&out);
    CHECK(out.size() == 3 && out[1].index == 2 && out[2].real && out[2].index == 3);
    CHECK(out[0].target < out[1].target && out[1].target < out[2].target);
    CHECK(out[0].present_id < out[1].present_id && out[1].present_id < out[2].present_id);
    const FgPresentStats st = e.Stats();
    CHECK(st.game_presents == 2 && st.real_queued == 2 && st.generated_queued == 2);
    CHECK(st.refused_calls == 2);  // BeginInterval before Attach, the second Attach
}

// Runs 40 intervals of 60 fps game frames with n generated images each through the engine and a display whose
// buffer release lags one refresh behind the flip (as an available event does). Returns the engine's statistics.
FgPresentStats RunOwnership(std::uint32_t slots, std::uint32_t n, std::uint32_t* displayed, bool* overlap,
                            bool* ordered, bool* monotonic) {
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    fences.Set(f, 1000000);
    FgPresentEngine e(EngineConfig(slots), &fences);
    e.Attach(1);
    SimDisplay disp{&e, {}, {}, {}, {}, {}, true, 0};
    const FgTicks period = 6 * kRefresh360;  // 60 fps on a 360 Hz panel: 2x, 3x and 6x land on whole refreshes
    FgTicks t = 100 * kMs;
    FgTicks now = t - period;
    std::uint64_t fence_value = 1;
    *overlap = false;
    for (std::uint64_t j = 1; j <= 40; ++j) {
        FgIntervalPlan p = MakePlan(1, j, j, t, period, n);
        FgIntervalReservation res;
        if (e.BeginInterval(p, now, &res) != FgStatus::Ok) break;
        for (std::uint32_t i = 0; i < res.count; ++i)
            if (res.slots[i].reserved && disp.Owns(res.slots[i].slot)) *overlap = true;
        FgIntervalSubmission sub;
        sub.epoch = 1;
        sub.interval_id = j;
        sub.count = res.count;
        for (std::uint32_t i = 0; i < res.count; ++i) {
            sub.ready[i] = FgReadyFence{f, fence_value};
            sub.produced[i] = res.slots[i].reserved;
        }
        ++fence_value;
        e.SubmitInterval(sub, now);
        disp.Submit();
        const FgTicks until = p.targets[n].target;  // the display runs until the real target, in refresh steps
        while (now < until) {
            now += kRefresh360;
            disp.Step(now);
        }
        t = p.targets[n].target;
    }
    *displayed = static_cast<std::uint32_t>(disp.shown_ids.size());
    *ordered = disp.ordered;
    *monotonic = true;
    for (std::size_t i = 1; i < disp.shown_targets.size(); ++i)
        *monotonic = *monotonic && disp.shown_targets[i] > disp.shown_targets[i - 1];

    // a Retired slot is not free until the display releases it: resize waits (Busy), never a fake success
    std::uint64_t cancel = 0;
    e.Quiesce(now, &cancel);
    std::vector<std::uint32_t> awaiting;
    e.SlotsAwaitingRelease(&awaiting);
    bool retired = false;
    for (std::uint32_t s : awaiting) retired = retired || e.SlotState(s) == FgEngineSlotState::Retired;
    if (retired) CHECK(e.CompleteResize(true, 2) == FgStatus::Busy);
    disp.Cancel(cancel);
    for (std::uint32_t s : awaiting)
        if (e.SlotState(s) == FgEngineSlotState::Retired) e.OnSlotReleased(s, false);
    CHECK(e.Drained());
    CHECK(e.CompleteResize(true, 2) == FgStatus::Ok);
    CHECK(e.Epoch() == 2 && e.FreeSlots() == slots);
    return e.Stats();
}

void TestEngineOwnership() {
    g_case = "engine-ownership";
    std::uint32_t displayed = 0;
    bool overlap = true, ordered = false, monotonic = false;
    // N + 4 slots (the composition default): every image shown, no drop, no shared buffer
    for (std::uint32_t n : {1u, 2u, 5u}) {
        const FgPresentStats st = RunOwnership(n + 4, n, &displayed, &overlap, &ordered, &monotonic);
        std::printf("  ownership n=%u slots=%u: displayed %u, late %llu, pressure %llu, collisions %llu, refused %llu\n",
                    n, n + 4, displayed, static_cast<unsigned long long>(st.dropped_late),
                    static_cast<unsigned long long>(st.dropped_pressure),
                    static_cast<unsigned long long>(st.display_collisions),
                    static_cast<unsigned long long>(st.refused_calls));
        CHECK(!overlap);  // the executor never got a buffer the display still referenced
        CHECK(ordered && monotonic);
        CHECK(displayed >= 40 * (n + 1) - (n + 1));  // the last interval may still be queued
        CHECK(st.dropped_late == 0 && st.dropped_pressure == 0 && st.display_collisions == 0);
        CHECK(st.refused_calls <= 1);  // only the expected Busy of the resize while a buffer was still retired
    }
    // N + 3 is the floor for a display that releases at once; with the one-refresh release lag of an available
    // event, the real image's reserve makes generated images yield (Pressure), they are never shown late or doubled
    const FgPresentStats lag = RunOwnership(FgMinSlotsFor(1), 1, &displayed, &overlap, &ordered, &monotonic);
    CHECK(!overlap && ordered && monotonic);
    CHECK(lag.dropped_pressure > 0 && lag.dropped_late == 0 && lag.display_collisions == 0);
    CHECK(lag.displayed_real >= 39);
}

void TestEngineNoSlotUntilReleased() {
    g_case = "engine-release";
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    fences.Set(f, 100);
    FgPresentEngine e(EngineConfig(4), &fences);
    CHECK(e.Attach(1) == FgStatus::Ok);
    const FgTicks period = 166667;
    FgTicks t = 0;
    std::vector<FgEnginePresent> out;
    // four real-only frames fill every slot (nothing displayed yet)
    for (std::uint64_t j = 1; j <= 4; ++j) {
        FgIntervalPlan p = MakePlan(1, j, j, t, period, 0);
        CHECK(e.PresentRealOnly(p, FgRealImage{1, j, FgReadyFence{f, 1}}) == FgStatus::Ok);
        t = p.targets[0].target;
    }
    e.Pump(0);
    e.TakeOutbox(&out);
    CHECK(out.size() == 4 && e.FreeSlots() == 0);
    FgIntervalPlan p5 = MakePlan(1, 5, 5, t, period, 0);
    CHECK(e.PresentRealOnly(p5, FgRealImage{1, 5, FgReadyFence{f, 1}}) == FgStatus::NoSlot);
    // displayed: the first goes on screen; still no slot (the display holds it)
    CHECK(e.OnPresentDisplayed(out[0].present_id, out[0].target));
    CHECK(e.FreeSlots() == 0);
    // the second supersedes the first: Retired, still not free
    CHECK(e.OnPresentDisplayed(out[1].present_id, out[1].target));
    CHECK(e.SlotState(out[0].slot) == FgEngineSlotState::Retired && e.FreeSlots() == 0);
    CHECK(!e.OnSlotReleased(out[1].slot, false));  // on screen: not releasable without a copy
    CHECK(e.OnSlotReleased(out[0].slot, false));
    CHECK(e.FreeSlots() == 1);
    CHECK(e.PresentRealOnly(p5, FgRealImage{1, 5, FgReadyFence{f, 1}}) == FgStatus::Ok);
    // a consumed-by-copy release frees a queued or displayed slot at once (the DXGI path)
    CHECK(e.OnSlotReleased(out[1].slot, true));
    CHECK(e.FreeSlots() == 1);
}

void TestEngineLateAndLeases() {
    g_case = "engine-late";
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    FgPresentEngine e(EngineConfig(), &fences);
    CHECK(e.Attach(1) == FgStatus::Ok);
    const FgTicks t0 = 500 * kMs;
    FgIntervalPlan p = MakePlan(1, 1, 1, t0, 166667, 2);
    FgIntervalReservation res;
    CHECK(e.BeginInterval(p, t0 - kMs, &res) == FgStatus::Ok);
    FgIntervalSubmission sub;
    sub.epoch = 1;
    sub.interval_id = 1;
    sub.count = 3;
    for (std::uint32_t i = 0; i < 3; ++i) {
        sub.ready[i] = FgReadyFence{f, 10 + i};
        sub.produced[i] = true;
    }
    CHECK(e.SubmitInterval(sub, t0) == FgStatus::Ok);
    CHECK(e.NextDeadline() == p.targets[0].deadline);
    // index 1 misses its deadline: dropped, its lease held until its fence completes
    e.Pump(p.targets[0].deadline + 1);
    CHECK(e.Stats().dropped_late == 1 && e.LeasesHeld() == 1);
    CHECK(e.SlotState(res.slots[0].slot) == FgEngineSlotState::Draining);
    fences.Set(f, 10);
    e.Pump(p.targets[0].deadline + 2);
    CHECK(e.LeasesHeld() == 0 && e.SlotState(res.slots[0].slot) == FgEngineSlotState::Free);
    // the real image is never dropped, however late
    fences.Set(f, 12);
    e.Pump(p.targets[2].target + 100 * kMs);
    std::vector<FgEnginePresent> out;
    e.TakeOutbox(&out);
    CHECK(out.size() == 1 && out[0].real);
    CHECK(e.Stats().dropped_late == 2);  // index 2 passed its deadline too
    // a deadline already past at reservation is refused right there (no slot spent)
    FgIntervalPlan q = MakePlan(1, 2, 2, p.targets[2].target, 166667, 1);
    CHECK(e.BeginInterval(q, q.targets[0].deadline + 1, &res) == FgStatus::Ok);
    CHECK(!res.slots[0].reserved && res.slots[0].refused == FgDropReason::Late && res.slots[1].reserved);
    CHECK(e.Stats().real_only_intervals[static_cast<std::uint32_t>(FgRealOnlyReason::Pressure)] == 0);
    CHECK(e.Stats().real_only_intervals[static_cast<std::uint32_t>(FgRealOnlyReason::Late)] == 1);
}

void TestEngineCancelAndRevoke() {
    g_case = "engine-cancel";
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    fences.Set(f, 100);
    FgPresentEngine e(EngineConfig(), &fences);
    CHECK(e.Attach(1) == FgStatus::Ok);
    const FgTicks t0 = 500 * kMs;
    FgIntervalPlan p = MakePlan(1, 1, 1, t0, 166667, 3);
    FgIntervalReservation res;
    CHECK(e.BeginInterval(p, t0, &res) == FgStatus::Ok);
    FgIntervalSubmission sub;
    sub.epoch = 1;
    sub.interval_id = 1;
    sub.count = 4;
    for (std::uint32_t i = 0; i < 4; ++i) {
        sub.ready[i] = FgReadyFence{f, 1};
        sub.produced[i] = true;
    }
    CHECK(e.SubmitInterval(sub, t0) == FgStatus::Ok);
    std::vector<FgEnginePresent> out;
    e.TakeOutbox(&out);
    CHECK(out.size() == 4);
    // stale suffix from the second generated image: real re-presented with a new id, generated retired
    std::uint64_t from = 0;
    CHECK(e.CancelPresentsFrom(out[1].present_id, t0, &from) == 3);
    CHECK(from == out[1].present_id);
    std::vector<FgEnginePresent> again;
    e.TakeOutbox(&again);
    CHECK(again.size() == 1 && again[0].real && again[0].represent && again[0].present_id > out[3].present_id &&
          again[0].slot == out[3].slot && again[0].target == out[3].target);
    CHECK(e.SlotState(out[1].slot) == FgEngineSlotState::Retired &&
          e.SlotState(out[2].slot) == FgEngineSlotState::Retired);
    CHECK(e.Stats().cancelled == 3 && e.Stats().real_represented == 1);
    // the cancelled buffers come back only through the display's release
    const std::uint32_t free_before = e.FreeSlots();
    CHECK(e.OnSlotReleased(out[1].slot, false) && e.OnSlotReleased(out[2].slot, false));
    CHECK(e.FreeSlots() == free_before + 2);
    // revoke: the queued generated image of the interval is cancelled as a suffix, the real one stays
    from = 0;
    CHECK(e.RevokeGenerated(1, FgDropReason::Revoked, t0, &from) == FgStatus::Ok);
    CHECK(from == out[0].present_id);
    std::vector<FgEnginePresent> re;
    e.TakeOutbox(&re);
    CHECK(re.size() == 1 && re[0].real);
    CHECK(e.Stats().revoked == 1);
    CHECK(e.RevokeGenerated(99, FgDropReason::Revoked, t0, &from) == FgStatus::InvalidArgument);
    CHECK(e.RevokeGenerated(1, FgDropReason::None, t0, &from) == FgStatus::InvalidArgument);
}

void TestEngineFeedback() {
    g_case = "engine-feedback";
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    fences.Set(f, 100);
    FgPresentEngine e(EngineConfig(), &fences);
    CHECK(e.Attach(1) == FgStatus::Ok);
    FgIntervalPlan p = MakePlan(1, 1, 1, 0, 166667, 2);
    FgIntervalReservation res;
    CHECK(e.BeginInterval(p, 0, &res) == FgStatus::Ok);
    FgIntervalSubmission sub;
    sub.epoch = 1;
    sub.interval_id = 1;
    sub.count = 3;
    for (std::uint32_t i = 0; i < 3; ++i) {
        sub.ready[i] = FgReadyFence{f, 1};
        sub.produced[i] = true;
    }
    CHECK(e.SubmitInterval(sub, 0) == FgStatus::Ok);
    std::vector<FgEnginePresent> out;
    e.TakeOutbox(&out);
    CHECK(out.size() == 3);
    CHECK(!e.OnPresentDropped(out[2].present_id, FgDropReason::Late));  // a real image is never dropped
    const std::uint64_t producer_late = e.Stats().generated_late_pre_queue;
    const std::uint64_t post_queue = e.Stats().generated_dropped_post_queue;
    CHECK(e.OnPresentDropped(out[0].present_id, FgDropReason::Late));
    CHECK(e.Stats().generated_late_pre_queue == producer_late);
    CHECK(e.Stats().generated_dropped_post_queue == post_queue + 1);
    CHECK(e.SlotState(out[0].slot) == FgEngineSlotState::Free && e.Stats().dropped_late == 1);
    CHECK(e.OnPresentSkipped(out[1].present_id) && e.Stats().display_skipped == 1);
    CHECK(!e.OnPresentSkipped(out[1].present_id));  // already gone
    CHECK(!e.OnPresentDisplayed(12345, 0));         // unknown id
    CHECK(e.OnPresentDisplayed(out[2].present_id, out[2].target + 3 * kMs));
    CHECK(e.Stats().max_slip == 3 * kMs && e.Stats().displayed_real == 1);
    // two images shown within half a refresh: a collision (two flips in one scan-out)
    FgIntervalPlan q = MakePlan(1, 2, 2, p.targets[2].target, 166667, 1);
    CHECK(e.BeginInterval(q, 0, &res) == FgStatus::Ok);
    sub.interval_id = 2;
    sub.count = 2;
    CHECK(e.SubmitInterval(sub, 0) == FgStatus::Ok);
    out.clear();
    e.TakeOutbox(&out);
    CHECK(out.size() == 2);
    CHECK(e.OnPresentDisplayed(out[0].present_id, out[0].target));
    CHECK(e.OnPresentDisplayed(out[1].present_id, out[0].target + kRefresh360 / 2 - 1));
    CHECK(e.Stats().display_collisions == 1);
}

void TestEngineLifecycle() {
    g_case = "engine-lifecycle";
    ScriptFences fences;
    const std::uint32_t f = fences.Create();
    fences.Set(f, 100);
    {
        FgPresentEngine e(EngineConfig(), &fences);
        CHECK(e.Attach(1) == FgStatus::Ok);
        std::uint32_t slot = 99;
        CHECK(e.ReserveRealOnlySlot(&slot) == FgStatus::Ok && slot < e.SlotCount());
        CHECK(e.HasRealOnlyReservation());
        std::uint32_t slot2 = 0;
        CHECK(e.ReserveRealOnlySlot(&slot2) == FgStatus::WrongState);
        FgIntervalPlan r = MakePlan(1, 1, 1, 0, 166667, 0);
        CHECK(e.PresentRealOnly(r, FgRealImage{1, 1, FgReadyFence{f, 1}}) == FgStatus::Ok);
        e.Pump(0);
        std::vector<FgEnginePresent> out;
        e.TakeOutbox(&out);
        CHECK(out.size() == 1 && out[0].slot == slot && !e.HasRealOnlyReservation());
        // a reservation dropped by Quiesce
        CHECK(e.ReserveRealOnlySlot(&slot2) == FgStatus::Ok);
        std::uint64_t from = 0;
        CHECK(e.Quiesce(0, &from) == FgStatus::Ok && from == out[0].present_id);
        CHECK(!e.HasRealOnlyReservation() && e.SlotState(slot2) == FgEngineSlotState::Free);
        CHECK(e.Detach(0, &from) == FgStatus::Busy);  // the cancelled present's buffer is not released yet
        CHECK(e.OnSlotReleased(slot, false));
        CHECK(e.Detach(0, &from) == FgStatus::Ok && e.State() == FgPresenterState::Detached);
    }
    {
        FgPresentEngine e(EngineConfig(), &fences);
        CHECK(e.Attach(1) == FgStatus::Ok);
        CHECK(e.OnDeviceLost() == FgStatus::Ok && e.State() == FgPresenterState::Lost);
        FgIntervalReservation res;
        CHECK(e.BeginInterval(MakePlan(1, 1, 1, 0, 166667, 1), 0, &res) == FgStatus::DeviceLost);
        std::uint64_t from = 0;
        CHECK(e.Quiesce(0, &from) == FgStatus::DeviceLost);
        CHECK(e.Detach(0, &from) == FgStatus::Ok);
    }
    {
        FgPresentEngine e(EngineConfig(), &fences);
        CHECK(e.Attach(1) == FgStatus::Ok);
        e.MarkFailed();
        CHECK(e.State() == FgPresenterState::Failed);
        FgIntervalReservation res;
        CHECK(e.BeginInterval(MakePlan(1, 1, 1, 0, 166667, 1), 0, &res) == FgStatus::Refused);
        std::uint64_t from = 0;
        CHECK(e.Detach(0, &from) == FgStatus::Ok);
    }
    {
        FgEngineConfig c = EngineConfig();
        c.probe.supported = false;
        c.probe.refusal = FgStatus::Unsupported;
        FgPresentEngine e(c, &fences);
        CHECK(e.Attach(1) == FgStatus::Unsupported);
    }
    {
        FgEngineConfig c = EngineConfig();
        c.probe.max_generated = 2;
        FgPresentEngine e(c, &fences);
        CHECK(e.Attach(1) == FgStatus::Ok);
        FgIntervalReservation res;
        CHECK(e.BeginInterval(MakePlan(1, 1, 1, 0, 166667, 3), 0, &res) == FgStatus::Unsupported);
        // a broken plan is refused (validated like every presenter does)
        FgIntervalPlan bad = MakePlan(1, 1, 1, 0, 166667, 2);
        bad.targets[0].target += 1;
        CHECK(e.BeginInterval(bad, 0, &res) == FgStatus::InvalidArgument);
        FgIntervalPlan stale = MakePlan(7, 1, 1, 0, 166667, 2);
        CHECK(e.BeginInterval(stale, 0, &res) == FgStatus::StaleEpoch);
        CHECK(e.Stats().refused_calls == 3);
    }
}

// Scheduler + engine + simulated display for 2x..6x at cadences that fit a 360 Hz panel.
void TestCadenceMatrix() {
    g_case = "cadence-matrix";
    struct Case {
        std::uint32_t fps;
        std::uint32_t n;
    };
    // output rates below the 360 Hz ceiling (exactly 360 is outside the scheduler's margin and stays real-only)
    const Case cases[] = {{60, 1}, {60, 2}, {60, 4}, {90, 1}, {120, 1}, {90, 2}, {72, 3}, {48, 5}, {45, 3}};
    for (const Case& c : cases) {
        ScriptFences fences;
        const std::uint32_t f = fences.Create();
        fences.Set(f, ~0ull >> 1);
        FgSchedulerConfig sc;
        FgTargetScheduler sched(sc);
        FgPresentEngine e(EngineConfig(FgMinSlotsFor(c.n) + 1), &fences);
        CHECK(sched.BeginEpoch(1) == FgStatus::Ok);
        CHECK(e.Attach(1) == FgStatus::Ok);
        SimDisplay disp{&e, {}, {}, {}, {}, {}, true, 0};
        const FgTicks period = (10000000 + c.fps / 2) / c.fps;
        FgTicks now = 1000 * kMs;
        std::uint64_t generated_plans = 0;
        std::uint64_t value = 1;
        bool all_ok = true;
        for (std::uint64_t j = 1; j <= 60; ++j) {
            FgGameFrame gf;
            gf.epoch = 1;
            gf.frame_id = j;
            gf.arrival = now;
            gf.requested_count = c.n;
            FgIntervalPlan plan;
            if (sched.Commit(gf, &plan) != FgStatus::Ok) {
                all_ok = false;
                break;
            }
            if (plan.generated_count == 0) {
                all_ok = all_ok && e.PresentRealOnly(plan, FgRealImage{1, j, FgReadyFence{f, value}}) == FgStatus::Ok;
            } else {
                ++generated_plans;
                FgIntervalReservation res;
                all_ok = all_ok && e.BeginInterval(plan, now, &res) == FgStatus::Ok;
                FgIntervalSubmission sub;
                sub.epoch = 1;
                sub.interval_id = plan.interval_id;
                sub.count = res.count;
                for (std::uint32_t i = 0; i < res.count; ++i) {
                    sub.ready[i] = FgReadyFence{f, value};
                    sub.produced[i] = res.slots[i].reserved;
                }
                all_ok = all_ok && e.SubmitInterval(sub, now) == FgStatus::Ok;
            }
            ++value;
            e.Pump(now);
            disp.Submit();
            const FgTicks next = now + period;
            for (FgTicks t = now + kMs; t <= next; t += kMs) {
                e.Pump(t);
                disp.Submit();
                disp.Step(t);
            }
            now = next;
        }
        const FgPresentStats st = e.Stats();
        CHECK(all_ok);
        CHECK(generated_plans > 40);
        CHECK(st.dropped_late == 0 && st.dropped_pressure == 0 && st.refused_calls == 0);
        CHECK(st.display_collisions == 0);
        CHECK(disp.ordered);
        bool even = true;
        for (std::size_t i = 1; i < disp.shown_targets.size(); ++i)
            even = even && disp.shown_targets[i] > disp.shown_targets[i - 1];
        CHECK(even);
        if (!all_ok || st.dropped_late || st.dropped_pressure || generated_plans <= 40)
            std::printf("  cadence %u fps x%u: plans %llu late %llu pressure %llu refused %llu\n", c.fps, c.n + 1,
                        static_cast<unsigned long long>(generated_plans),
                        static_cast<unsigned long long>(st.dropped_late),
                        static_cast<unsigned long long>(st.dropped_pressure),
                        static_cast<unsigned long long>(st.refused_calls));
    }
}

// --- selection and creation refusals -----------------------------------------------------------------------------

FgPresenterCapabilities AllCaps() {
    FgPresenterCapabilities c;
    c.adapter_probed = true;
    c.adapter.queried = true;
    c.adapter.hw_flip_queue_enabled = true;
    c.adapter.displayable_supported = true;
    c.presentation_probed = true;
    c.presentation_supported = true;
    c.independent_flip_supported = true;
    c.d3d11_displayable = true;
    c.tearing_probed = true;
    c.tearing_supported = true;
    return c;
}

void TestSelection() {
    g_case = "selection";
    FgDisplayDesc vrr{true, 0};
    FgDisplayDesc fixed{false, kRefresh360};
    FgPresenterCapabilities all = AllCaps();
    FgPresenterSelection s = FgSelectPresenter(FgPresenterChoice::Auto, all, vrr);
    CHECK(s.status == FgStatus::Ok && s.chosen == FgPresenterChoice::DxgiFlipTearing);
    s = FgSelectPresenter(FgPresenterChoice::Auto, all, fixed);
    CHECK(s.status == FgStatus::Ok && s.chosen == FgPresenterChoice::WindowsCompositionHwfq);
    FgPresenterCapabilities no_hwfq = all;
    no_hwfq.adapter.hw_flip_queue_enabled = false;
    s = FgSelectPresenter(FgPresenterChoice::Auto, no_hwfq, fixed);
    CHECK(s.status == FgStatus::Ok && s.chosen == FgPresenterChoice::DxgiFlipVsync);
    FgPresenterCapabilities no_tear = all;
    no_tear.tearing_supported = false;
    s = FgSelectPresenter(FgPresenterChoice::Auto, no_tear, vrr);
    CHECK(s.status == FgStatus::Ok && s.chosen == FgPresenterChoice::WindowsCompositionHwfq);
    FgPresenterCapabilities nothing = no_tear;
    nothing.independent_flip_supported = false;
    s = FgSelectPresenter(FgPresenterChoice::Auto, nothing, vrr);
    CHECK(s.status == FgStatus::Unsupported && s.refusal == FgWinRefusal::NoTearing &&
          s.chosen == FgPresenterChoice::Auto);

    // explicit choices are checked and refused with the first missing capability, never replaced
    s = FgSelectPresenter(FgPresenterChoice::WindowsCompositionHwfq, no_hwfq, fixed);
    CHECK(s.status == FgStatus::Unsupported && s.refusal == FgWinRefusal::NoHardwareFlipQueue &&
          s.chosen == FgPresenterChoice::Auto);
    FgPresenterCapabilities c = all;
    c.presentation_supported = false;
    CHECK(FgSelectPresenter(FgPresenterChoice::WindowsCompositionHwfq, c, fixed).refusal ==
          FgWinRefusal::NoPresentationManager);
    c = all;
    c.independent_flip_supported = false;
    CHECK(FgSelectPresenter(FgPresenterChoice::WindowsCompositionHwfq, c, fixed).refusal ==
          FgWinRefusal::NoIndependentFlip);
    c = all;
    c.d3d11_displayable = false;
    CHECK(FgSelectPresenter(FgPresenterChoice::WindowsCompositionHwfq, c, fixed).refusal ==
          FgWinRefusal::NotDisplayable);
    c = all;
    c.presentation_probed = false;
    CHECK(FgSelectPresenter(FgPresenterChoice::WindowsCompositionHwfq, c, fixed).refusal == FgWinRefusal::Probe);
    s = FgSelectPresenter(FgPresenterChoice::DxgiFlipTearing, no_tear, vrr);
    CHECK(s.status == FgStatus::Unsupported && s.refusal == FgWinRefusal::NoTearing);
    c = all;
    c.tearing_probed = false;
    CHECK(FgSelectPresenter(FgPresenterChoice::DxgiFlipTearing, c, vrr).refusal == FgWinRefusal::NoTearing);
    s = FgSelectPresenter(FgPresenterChoice::DxgiFlipVsync, nothing, fixed);
    CHECK(s.status == FgStatus::Ok && s.chosen == FgPresenterChoice::DxgiFlipVsync);
    s = FgSelectPresenter(static_cast<FgPresenterChoice>(9), all, fixed);
    CHECK(s.status == FgStatus::InvalidArgument && s.refusal == FgWinRefusal::UnknownChoice);
    FgDisplayDesc broken{false, -5};
    CHECK(FgSelectPresenter(FgPresenterChoice::Auto, all, broken).refusal == FgWinRefusal::RefreshUnknown);
    CHECK(std::strcmp(FgPresenterChoiceName(FgPresenterChoice::DxgiFlipTearing), "dxgi-flip-tearing") == 0);
    CHECK(std::strcmp(FgWinRefusalName(FgWinRefusal::NoIndependentFlip), "no-independent-flip") == 0);
}

void TestCreationRefusals() {
    g_case = "creation-refusals";
    const FgPresenterCapabilities all = AllCaps();
    FgCreatedPresenter made;
    FgPresenterFactoryDesc d;
    FgWinResult r;

    d.choice = static_cast<FgPresenterChoice>(7);
    r = FgCreatePresenter(d, &made);
    CHECK(r.status == FgStatus::InvalidArgument && r.refusal == FgWinRefusal::UnknownChoice && !made.presenter);

    d = FgPresenterFactoryDesc();  // no caps, no device: nothing to probe
    r = FgCreatePresenter(d, &made);
    CHECK(r.status == FgStatus::InvalidArgument && r.refusal == FgWinRefusal::NullDevice && !made.presenter);

    d = FgPresenterFactoryDesc();
    d.caps = &all;
    d.choice = FgPresenterChoice::WindowsCompositionHwfq;
    d.width = 64;
    d.height = 64;
    d.hwnd = nullptr;
    r = FgCreatePresenter(d, &made);
    CHECK(r.status == FgStatus::InvalidArgument && r.refusal == FgWinRefusal::InvalidWindow && !made.presenter);
    CHECK(made.selection.chosen == FgPresenterChoice::WindowsCompositionHwfq);
    d.hwnd = reinterpret_cast<HWND>(static_cast<std::uintptr_t>(0x7fff0123));  // not a window
    r = FgCreatePresenter(d, &made);
    CHECK(r.refusal == FgWinRefusal::InvalidWindow && !made.presenter);
    d.hwnd = GetDesktopWindow();  // a window of another process
    r = FgCreatePresenter(d, &made);
    CHECK(r.status != FgStatus::Ok && !made.presenter);
    CHECK(r.refusal == FgWinRefusal::ForeignWindow || r.refusal == FgWinRefusal::NullDevice);

    // Auto on a fixed-refresh display picks composition; its refusal comes back, DXGI is NOT tried
    d.choice = FgPresenterChoice::Auto;
    d.hwnd = nullptr;
    r = FgCreatePresenter(d, &made);
    CHECK(r.refusal == FgWinRefusal::InvalidWindow && !made.presenter && !made.dxgi && !made.composition);
    CHECK(made.selection.chosen == FgPresenterChoice::WindowsCompositionHwfq);

    // unsupported mode: tearing requested where the probe says no
    FgPresenterCapabilities no_tear = all;
    no_tear.tearing_supported = false;
    d = FgPresenterFactoryDesc();
    d.caps = &no_tear;
    d.choice = FgPresenterChoice::DxgiFlipTearing;
    r = FgCreatePresenter(d, &made);
    CHECK(r.status == FgStatus::Unsupported && r.refusal == FgWinRefusal::NoTearing && !made.presenter);
    // DXGI with nothing to present on
    d.choice = FgPresenterChoice::DxgiFlipVsync;
    r = FgCreatePresenter(d, &made);
    CHECK(r.status == FgStatus::InvalidArgument && r.refusal == FgWinRefusal::NullDevice && !made.presenter);

    // direct back-end argument checks (decided before any device is touched)
    std::unique_ptr<FgWinCompPresenter> comp;
    FgCompPresenterDesc cd;
    cd.bind_window = false;
    cd.width = 64;
    cd.height = 64;
    cd.hwnd = GetDesktopWindow();
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::InvalidWindow && !comp);
    cd.hwnd = nullptr;
    cd.max_generated = 6;
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::BadCount && !comp);
    cd.max_generated = 5;
    cd.slot_count = 7;  // below N + 3
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::SlotCount && !comp);
    cd.slot_count = 33;
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::SlotCount && !comp);
    cd.slot_count = 0;
    cd.width = 0;
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::BadSize && !comp);
    cd.width = 64;
    cd.format = DXGI_FORMAT_R32_FLOAT;
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::BadFormat && !comp);
    cd.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    CHECK(FgWinCompPresenter::Create(cd, &comp).refusal == FgWinRefusal::NullDevice && !comp);

    std::unique_ptr<FgDxgiPresenter> dx;
    FgDxgiPresenterDesc dd;
    dd.pacing = static_cast<FgDxgiPacing>(5);
    CHECK(FgDxgiPresenter::Create(dd, &dx).refusal == FgWinRefusal::UnknownChoice && !dx);
    dd.pacing = FgDxgiPacing::Tearing;
    dd.slot_count = 3;
    CHECK(FgDxgiPresenter::Create(dd, &dx).refusal == FgWinRefusal::SlotCount && !dx);
    dd.slot_count = 0;
    dd.max_generated = 0;
    CHECK(FgDxgiPresenter::Create(dd, &dx).refusal == FgWinRefusal::BadCount && !dx);
    dd.max_generated = 5;
    CHECK(FgDxgiPresenter::Create(dd, &dx).refusal == FgWinRefusal::NullDevice && !dx);
}

// --- worker thread, timer, clocks, read-only probe -----------------------------------------------------------------

class CountingBody final : public IFgWorkerBody {
public:
    std::atomic<int> wakes{0};
    std::atomic<int> runs{0};
    void Run(HANDLE stop, HANDLE wake) override {
        ++runs;
        HANDLE hs[2] = {stop, wake};
        for (;;) {
            const DWORD w = WaitForMultipleObjects(2, hs, FALSE, 1000);
            if (w == WAIT_OBJECT_0) return;
            if (w == WAIT_OBJECT_0 + 1) ++wakes;
        }
    }
};

class ReturningBody final : public IFgWorkerBody {
public:
    void Run(HANDLE, HANDLE) override {}
};

void TestWorkerLifecycle() {
    g_case = "worker";
    CountingBody body;
    {
        FgWorkerThread w;
        CHECK(!w.Running() && !w.Joinable());
        CHECK(!w.Start(nullptr, L"x"));
        CHECK(w.Start(&body, L"wg-fg-test-worker"));
        CHECK(!w.Start(&body, L"again"));  // already running
        for (int i = 0; i < 5; ++i) {
            w.Wake();
            Sleep(2);
        }
        w.Stop();
        CHECK(!w.Running() && !w.Joinable());
        w.Stop();  // idempotent
        CHECK(body.wakes.load() >= 1 && body.runs.load() == 1);
        CHECK(w.Start(&body, L"wg-fg-test-worker-2"));  // restart after stop
        CHECK(w.Running());
    }  // destroyed while running: joins
    CHECK(body.runs.load() == 2);
    ReturningBody ret;
    FgWorkerThread w2;
    CHECK(w2.Start(&ret, nullptr));
    for (int i = 0; i < 200 && w2.Running(); ++i) Sleep(1);
    CHECK(!w2.Running());  // the body returned on its own
    w2.Stop();
    CHECK(!w2.Joinable());
    // many short lifetimes
    bool ok = true;
    for (int i = 0; i < 20; ++i) {
        FgWorkerThread w3;
        ok = ok && w3.Start(&body, L"wg-fg-short");
        w3.Wake();
    }
    CHECK(ok);
}

void TestTimerAndClock() {
    g_case = "timer-clock";
    FgHighResTimer timer;
    CHECK(timer.Valid());
    if (timer.Valid()) {
        const FgTicks t0 = FgInterruptNow();
        CHECK(timer.ArmAt(t0 + 3 * kMs));
        CHECK(WaitForSingleObject(timer.Handle(), 2000) == WAIT_OBJECT_0);
        const FgTicks elapsed = FgInterruptNow() - t0;
        CHECK(elapsed >= 2 * kMs && elapsed < 1000 * kMs);
        CHECK(timer.ArmAt(FgInterruptNow() - kMs));  // already past: fires at once
        CHECK(WaitForSingleObject(timer.Handle(), 500) == WAIT_OBJECT_0);
        timer.Cancel();
    }
    FgClockMap map;
    LARGE_INTEGER q = {};
    QueryPerformanceCounter(&q);
    const FgTicks from_qpc = map.FromQpc(q.QuadPart);
    const FgTicks now = FgInterruptNow();
    CHECK(std::llabs(now - from_qpc) < 20 * kMs);
    CHECK(FgInterruptNow() >= now);
    CHECK(FgIsDeviceLostHr(DXGI_ERROR_DEVICE_REMOVED) && FgIsDeviceLostHr(static_cast<HRESULT>(0x88810001L)) &&
          !FgIsDeviceLostHr(E_INVALIDARG) && !FgIsDeviceLostHr(S_OK));
    FgD3D12FenceTable table;
    CHECK(table.Add(nullptr) == 0 && !table.FenceValid(1) && table.FenceCompleted(1) == 0 && table.Count() == 0);
    FgChainRegistry& reg = FgChainRegistry::Process();
    int a = 0, b = 0;
    CHECK(reg.Claim(424242, &a) && reg.Claim(424242, &a) && !reg.Claim(424242, &b));
    reg.Release(424242, &b);
    CHECK(reg.Owner(424242) == &a);
    reg.Release(424242, &a);
    CHECK(reg.Owner(424242) == nullptr && reg.Claim(424242, &b));
    reg.Release(424242, &b);
}

void TestReadOnlyProbe() {
    g_case = "probe-read-only";
    std::vector<LUID> luids;
    const FgWinResult e = FgEnumAdapterLuids(&luids);
    std::printf("  adapters (D3DKMT, read only): enum %s, count %zu\n", e.Ok() ? "ok" : "failed", luids.size());
    for (const LUID& l : luids) {
        FgPresenterCapabilities caps;
        const FgWinResult r = FgProbeAdapterCapabilities(l, &caps);
        std::printf("  adapter %08lx:%08lx: query %s (0x%08lx), hw flip queue support %u enabled %d, displayable %d, "
                    "hw scheduling %d, tearing %d\n",
                    static_cast<unsigned long>(l.HighPart), static_cast<unsigned long>(l.LowPart),
                    r.Ok() ? "ok" : "refused", static_cast<unsigned long>(caps.adapter.status),
                    caps.adapter.hw_flip_queue_support_state, caps.adapter.hw_flip_queue_enabled,
                    caps.adapter.displayable_supported, caps.adapter.hw_sch_enabled, caps.tearing_supported);
        CHECK(caps.adapter_probed);
        CHECK(!caps.adapter.hw_flip_queue_enabled || caps.adapter.queried);  // enabled implies answered
        CHECK(r.Ok() == caps.adapter.queried);
    }
    FgPresenterCapabilities none;
    CHECK(FgProbePresenterCapabilities(nullptr, &none).refusal == FgWinRefusal::NullDevice);
    LUID bogus = {0x7ffffff0, 0x7ffffff0};
    FgAdapterFlipCaps bad;
    const FgWinResult br = FgQueryAdapterFlipCaps(bogus, &bad);
    CHECK(!br.Ok() && !bad.queried && br.refusal == FgWinRefusal::AdapterNotFound);
}

// --- headless GPU checks (--gpu only) ------------------------------------------------------------------------------

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> debug;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12DescriptorHeap> rtv;
    ComPtr<ID3D12Fence> fence;  // shared: the executor's producer fence
    std::uint64_t value = 0;
    HANDLE event = nullptr;
    ~Gpu() {
        if (event) CloseHandle(event);
    }
    bool Init() {
        ComPtr<IDXGIFactory6> f;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f)))) return false;
        ComPtr<IDXGIAdapter1> a;
        for (UINT i = 0; f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&a)) !=
                         DXGI_ERROR_NOT_FOUND;
             ++i) {
            DXGI_ADAPTER_DESC1 d;
            a->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            a.Reset();
        }
        if (g_debug_layer) {
            ComPtr<ID3D12Debug> layer;
            if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)))) return false;
            layer->EnableDebugLayer();
        }
        if (!a || FAILED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return false;
        if (g_debug_layer && FAILED(device.As(&debug))) return false;
        D3D12_COMMAND_QUEUE_DESC q = {};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&alloc))) ||
            FAILED(device->CreateCommandList(0, q.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
            FAILED(list->Close()) || FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))))
            return false;
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return event != nullptr;
    }
    bool DebugLayerClean() const {
        if (!debug) return !g_debug_layer;
        bool clean = true;
        for (UINT64 index = 0; index < debug->GetNumStoredMessages(); ++index) {
            SIZE_T bytes = 0;
            if (FAILED(debug->GetMessage(index, nullptr, &bytes))) return false;
            std::vector<char> buffer(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(buffer.data());
            if (FAILED(debug->GetMessage(index, message, &bytes))) return false;
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::printf("debug layer: %s\n", message->pDescription);
                clean = false;
            }
        }
        return clean;
    }
    // The executor's work for one slot: clear it to a constant grey, leave it in COMMON, signal the fence.
    std::uint64_t Render(ID3D12Resource* target, float grey) {
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        device->CreateRenderTargetView(target, nullptr, rtv->GetCPUDescriptorHandleForHeapStart());
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {target, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_RENDER_TARGET};
        list->ResourceBarrier(1, &b);
        const float c[4] = {grey, grey, grey, 1.0f};
        list->ClearRenderTargetView(rtv->GetCPUDescriptorHandleForHeapStart(), c, 0, nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        list->ResourceBarrier(1, &b);
        list->Close();
        ID3D12CommandList* l[] = {list.Get()};
        queue->ExecuteCommandLists(1, l);
        queue->Signal(fence.Get(), ++value);
        // one list / allocator: wait so the next Render may reset them (test executor only)
        fence->SetEventOnCompletion(value, event);
        WaitForSingleObject(event, 2000);
        return value;
    }
};

// Drives `frames` game frames at `period` through the scheduler into a back end with slot access (composition or
// DXGI). Returns false when a call failed unexpectedly.
template <class P>
bool DriveFrames(P* presenter, Gpu* gpu, std::uint32_t fence_id, std::uint32_t n, FgTicks period, int frames,
                 std::uint64_t* generated_plans, std::uint64_t* refused) {
    FgSchedulerConfig sc;
    sc.display.min_interval = 1;           // headless: the synthetic cadence decides, not a panel
    sc.display.max_interval = 10000000;
    sc.max_lead = 2000000;
    // this test's executor renders each slot with its own CPU wait: give the first image a budget that covers it,
    // so late drops measure the presenter, not the test's own executor
    sc.first_image_budget = 80000;
    FgTargetScheduler sched(sc);
    if (sched.BeginEpoch(1) != FgStatus::Ok) return false;
    *generated_plans = 0;
    *refused = 0;
    bool ok = true;
    // game-frame arrivals paced by the high-resolution timer: Sleep(1) jitter (about 1 ms) would make the cadence
    // estimate miss whole-refresh multiples, which fixed-refresh back ends rightly refuse
    FgHighResTimer pacer;
    int reasons_printed = 0;
    FgTicks next = FgInterruptNow() + 5 * kMs;
    for (int j = 1; j <= frames; ++j) {
        if (pacer.Valid() && pacer.ArmAt(next)) WaitForSingleObject(pacer.Handle(), 1000);
        while (FgInterruptNow() < next) SwitchToThread();
        FgGameFrame gf;
        gf.epoch = 1;
        gf.frame_id = static_cast<std::uint64_t>(j);
        gf.arrival = FgInterruptNow();
        gf.requested_count = n;
        FgIntervalPlan plan;
        if (sched.Commit(gf, &plan) != FgStatus::Ok) return false;
        if (plan.generated_count == 0 && j > 4 && reasons_printed < 3) {
            ++reasons_printed;
            std::printf("  frame %d: real-only (%s), period %lld\n", j, FgRealOnlyReasonName(plan.real_only_reason),
                        static_cast<long long>(plan.period));
        }
        if (plan.generated_count == 0) {
            std::uint32_t slot = 0;
            const FgStatus rs = presenter->ReserveRealSlot(&slot);
            if (rs != FgStatus::Ok) {
                if (*refused < 3) std::printf("  frame %d: ReserveRealSlot -> %s\n", j, FgStatusName(rs));
                ++*refused;
            } else {
                const std::uint64_t v = gpu->Render(presenter->SlotResource(slot), 0.02f);
                ok = ok && presenter->PresentRealOnly(plan, FgRealImage{1, gf.frame_id, FgReadyFence{fence_id, v}}) ==
                               FgStatus::Ok;
            }
        } else {
            ++*generated_plans;
            FgIntervalReservation res;
            const FgStatus bs = presenter->BeginInterval(plan, &res);
            if (bs != FgStatus::Ok) {
                if (*refused < 3)
                    std::printf("  frame %d: BeginInterval(N=%u, period %lld) -> %s\n", j, plan.generated_count,
                                static_cast<long long>(plan.period), FgStatusName(bs));
                ++*refused;
            } else {
                FgIntervalSubmission sub;
                sub.epoch = 1;
                sub.interval_id = plan.interval_id;
                sub.count = res.count;
                for (std::uint32_t i = 0; i < res.count; ++i) {
                    if (!res.slots[i].reserved) continue;
                    const std::uint64_t v =
                        gpu->Render(presenter->SlotResource(res.slots[i].slot), res.slots[i].real ? 0.02f : 0.03f);
                    sub.ready[i] = FgReadyFence{fence_id, v};
                    sub.produced[i] = true;
                }
                ok = ok && presenter->SubmitInterval(sub) == FgStatus::Ok;
            }
        }
        next += period;
    }
    return ok;
}

void TestGpuProbe(Gpu& gpu) {
    g_case = "gpu-probe";
    FgPresenterCapabilities caps;
    const FgWinResult r = FgProbePresenterCapabilities(gpu.device.Get(), &caps);
    std::printf("  probe: %s; hw flip queue enabled %d (support state %u), presentation %d, independent flip %d, "
                "displayable %d, tearing %d\n",
                r.Ok() ? "ok" : FgWinRefusalName(r.refusal), caps.adapter.hw_flip_queue_enabled,
                caps.adapter.hw_flip_queue_support_state, caps.presentation_supported, caps.independent_flip_supported,
                caps.d3d11_displayable, caps.tearing_supported);
    CHECK(r.Ok() && caps.presentation_probed && caps.tearing_probed && caps.adapter_probed);
    const FgPresenterSelection fixed = FgSelectPresenter(FgPresenterChoice::Auto, caps, FgDisplayDesc{false, 0});
    const FgPresenterSelection vrr = FgSelectPresenter(FgPresenterChoice::Auto, caps, FgDisplayDesc{true, 0});
    std::printf("  auto: fixed refresh -> %s, variable refresh -> %s\n", FgPresenterChoiceName(fixed.chosen),
                FgPresenterChoiceName(vrr.chosen));
    // Auto with the real probe on a missing window: the chosen back end's refusal, nothing else is tried
    FgPresenterFactoryDesc d;
    d.device = gpu.device.Get();
    d.width = 64;
    d.height = 64;
    d.choice = FgPresenterChoice::WindowsCompositionHwfq;
    FgCreatedPresenter made;
    const FgWinResult cr = FgCreatePresenter(d, &made);
    CHECK(!cr.Ok() && !made.presenter && !made.dxgi);
    std::printf("  composition without a window: %s / %s\n", FgStatusName(cr.status), FgWinRefusalName(cr.refusal));
}

void TestGpuComposition(Gpu& gpu) {
    g_case = "gpu-composition";
    for (int variant = 0; variant < 2; ++variant) {
        const bool vrr = variant == 0;
        FgCompPresenterDesc cd;
        cd.device = gpu.device.Get();
        cd.bind_window = false;  // headless: an unbound surface, nothing is shown
        cd.width = 256;
        cd.height = 144;
        cd.variable_refresh = vrr;
        cd.max_generated = 2;
        cd.slot_count = 12;
        std::unique_ptr<FgWinCompPresenter> p;
        const FgWinResult r = FgWinCompPresenter::Create(cd, &p);
        std::printf("  composition (%s): create %s / %s hr 0x%08lx\n", vrr ? "variable refresh" : "fixed refresh",
                    FgStatusName(r.status), FgWinRefusalName(r.refusal), static_cast<unsigned long>(r.hr));
        if (!vrr && !r.Ok()) {
            // fixed refresh needs the compositor's vblank phase; without a live output it may be missing. That is an
            // explicit refusal, never a silent unsnapped mode.
            CHECK(r.refusal == FgWinRefusal::RefreshUnknown && !p);
            continue;
        }
        CHECK(r.Ok() && p);
        if (!p) continue;
        const FgCompPresenterInfo info = p->Info();
        std::printf("  slots %u, D3D11 misc flags 0x%x, simultaneous access %d, grid period %lld\n", info.slot_count,
                    info.d3d11_misc_flags, info.simultaneous_access, static_cast<long long>(info.grid.period));
        CHECK(info.slot_count == 12 && (info.d3d11_misc_flags & D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE) &&
              (info.d3d11_misc_flags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE) && info.simultaneous_access);
        CHECK(info.independent_flip_supported && info.d3d11_displayable && info.adapter.hw_flip_queue_enabled);
        // fences: an unshared fence is refused, the shared one is accepted
        ComPtr<ID3D12Fence> unshared;
        CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&unshared))));
        FgWinResult why;
        CHECK(p->RegisterFence(unshared.Get(), &why) == 0 && why.refusal == FgWinRefusal::SharedHandle);
        const std::uint32_t fid = p->RegisterFence(gpu.fence.Get(), &why);
        CHECK(fid >= 1 && why.Ok());
        CHECK(p->Attach(0xC0FFEE, 1) == FgStatus::Ok);
        CHECK(p->State() == FgPresenterState::Attached);
        std::uint64_t gen = 0, refused = 0;
        // about 30 fps game frames, 2x; on the fixed grid exactly 6 refreshes per image
        const FgTicks base = vrr ? 333333 : info.grid.period * 12;
        const bool ok = DriveFrames(p.get(), &gpu, fid, 1, base, 12, &gen, &refused);
        Sleep(150);
        const FgPresentStats st = p->Stats();
        std::printf("  %s: game %llu, real queued %llu, generated queued %llu, displayed %llu+%llu, skipped %llu, "
                    "late %llu, pressure %llu, refused %llu (plans %llu, refused here %llu), free slots %u\n",
                    vrr ? "vrr" : "fixed", static_cast<unsigned long long>(st.game_presents),
                    static_cast<unsigned long long>(st.real_queued),
                    static_cast<unsigned long long>(st.generated_queued),
                    static_cast<unsigned long long>(st.displayed_real),
                    static_cast<unsigned long long>(st.displayed_generated),
                    static_cast<unsigned long long>(st.display_skipped),
                    static_cast<unsigned long long>(st.dropped_late),
                    static_cast<unsigned long long>(st.dropped_pressure),
                    static_cast<unsigned long long>(st.refused_calls), static_cast<unsigned long long>(gen),
                    static_cast<unsigned long long>(refused), p->FreeSlots());
        CHECK(ok);
        // headless: an unbound surface is never displayed. Depending on timing the presentation manager answers
        // with Skipped statistics (a newer present was ready) or with nothing. A buffer comes back only through such
        // a statistic plus IsAvailable; without one the pool runs dry and the presenter refuses (back-pressure).
        // Either way no buffer is handed out while the display path may still own it.
        CHECK(st.real_queued + st.generated_queued >= 4);
        CHECK(st.generated_queued >= 2);  // at least one generating interval reached the display path
        CHECK(st.display_skipped >= 1 || refused > 0);
        CHECK(st.real_queued + st.generated_queued <= info.slot_count + st.display_skipped);
        CHECK(st.displayed_real + st.displayed_generated == 0);
        CHECK(st.refused_calls == refused);  // every refusal the driver saw is counted, nothing else
        CHECK(p->State() == FgPresenterState::Attached);
        // quiesce -> drained -> detach; another owner cannot claim the chain meanwhile
        CHECK(FgChainRegistry::Process().Owner(0xC0FFEE) == p.get());
        CHECK(p->Quiesce() == FgStatus::Ok);
        const bool idle = p->WaitIdle(3000);
        std::printf("  drained after quiesce: %d, free slots %u\n", idle, p->FreeSlots());
        CHECK(idle);
        FgStatus ds = p->Detach();
        for (int i = 0; i < 100 && ds == FgStatus::Busy; ++i) {
            Sleep(10);
            ds = p->Detach();
        }
        CHECK(ds == FgStatus::Ok && FgChainRegistry::Process().Owner(0xC0FFEE) == nullptr);
        // the buffers went to the compositor: a new attach gets a fresh pool
        CHECK(p->Attach(0xC0FFEE, 5) == FgStatus::Ok && p->FreeSlots() == info.slot_count);
        std::uint32_t slot = 0;
        CHECK(p->ReserveRealSlot(&slot) == FgStatus::Ok && p->SlotResource(slot) != nullptr);
        CHECK(p->Quiesce() == FgStatus::Ok && p->WaitIdle(1000));
        CHECK(p->SetResizeTarget(320, 180) == FgStatus::Ok);
        CHECK(p->CompleteResize(true, 6) == FgStatus::Ok && p->FreeSlots() == info.slot_count);
        CHECK(p->Detach() == FgStatus::Ok);
    }
}

void TestGpuDxgi(Gpu& gpu) {
    g_case = "gpu-dxgi";
    ComPtr<IDXGIFactory2> f;
    CHECK(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f))));
    if (!f) return;
    auto make_chain = [&gpu, &f](DXGI_SWAP_EFFECT effect, UINT flags, ComPtr<IDXGISwapChain4>* out) -> HRESULT {
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = 256;
        sd.Height = 144;
        sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = effect;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        sd.Flags = flags;
        ComPtr<IDXGISwapChain1> sc1;
        // a composition swap chain bound to no visual: presents go nowhere visible
        HRESULT hr = f->CreateSwapChainForComposition(gpu.queue.Get(), &sd, nullptr, &sc1);
        if (SUCCEEDED(hr)) hr = sc1.As(out);
        return hr;
    };
    ComPtr<IDXGISwapChain4> seq;
    if (SUCCEEDED(make_chain(DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
                             &seq))) {
        FgDxgiPresenterDesc dd;
        dd.device = gpu.device.Get();
        dd.queue = gpu.queue.Get();
        dd.swap_chain = seq.Get();
        dd.pacing = FgDxgiPacing::Vsync;
        std::unique_ptr<FgDxgiPresenter> p;
        CHECK(FgDxgiPresenter::Create(dd, &p).refusal == FgWinRefusal::SwapChainNotFlipDiscard && !p);
    }
    ComPtr<IDXGISwapChain4> nowait;
    if (SUCCEEDED(make_chain(DXGI_SWAP_EFFECT_FLIP_DISCARD, 0, &nowait))) {
        FgDxgiPresenterDesc dd;
        dd.device = gpu.device.Get();
        dd.queue = gpu.queue.Get();
        dd.swap_chain = nowait.Get();
        dd.pacing = FgDxgiPacing::Vsync;
        std::unique_ptr<FgDxgiPresenter> p;
        CHECK(FgDxgiPresenter::Create(dd, &p).refusal == FgWinRefusal::SwapChainNoWaitable && !p);
    }
    ComPtr<IDXGISwapChain4> sc;
    const HRESULT hr = make_chain(DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT, &sc);
    CHECK(SUCCEEDED(hr));
    if (FAILED(hr)) return;
    sc->SetMaximumFrameLatency(2);
    {
        ComPtr<IDXGISwapChain4> vrr_chain;
        CHECK(SUCCEEDED(make_chain(DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
                                   &vrr_chain)));
        if (vrr_chain) {
            CHECK(SUCCEEDED(vrr_chain->SetMaximumFrameLatency(2)));
            FgDxgiPresenterDesc dd;
            dd.device = gpu.device.Get();
            dd.queue = gpu.queue.Get();
            dd.swap_chain = vrr_chain.Get();
            dd.pacing = FgDxgiPacing::Vsync;
            dd.variable_refresh = true;
            dd.refresh_period = 27778;
            dd.min_interval = FgVrrMinInterval(dd.refresh_period, true);
            dd.max_generated = 1;
            std::unique_ptr<FgDxgiPresenter> vrr;
            CHECK(FgDxgiPresenter::Create(dd, &vrr).Ok());
            if (vrr) {
                CHECK(vrr->Attach(0xBEEE, 1) == FgStatus::Ok);
                FgWinResult why;
                const auto fence = vrr->RegisterFence(gpu.fence.Get(), &why);
                CHECK(fence != 0);
                uint64_t generated = 0, refused = 0;
                CHECK(DriveFrames(vrr.get(), &gpu, fence, 1, 147000, 20, &generated, &refused));
                CHECK(generated >= 10 && refused == 0);
                CHECK(vrr->WaitIdle(1000));
                const auto timing = vrr->Info();
                const auto display = vrr->ReadDisplayCadence(false);
                CHECK(timing.presents >= 20 && timing.start_gap_samples > 0);
                CHECK(timing.start_gap_under_refresh == 0);
                CHECK(display.submitted > 0);
                std::printf("  DXGI VRR vsync: presents=%llu observed=%llu unresolved=%llu pending=%llu "
                            "start_gap_us p50=%lld p95=%lld under_refresh=%llu\n",
                            static_cast<unsigned long long>(timing.presents),
                            static_cast<unsigned long long>(display.real + display.generated),
                            static_cast<unsigned long long>(display.unresolved),
                            static_cast<unsigned long long>(display.pending),
                            static_cast<long long>(timing.start_gap_p50 / 10),
                            static_cast<long long>(timing.start_gap_p95 / 10),
                            static_cast<unsigned long long>(timing.start_gap_under_refresh));
                CHECK(vrr->Detach() == FgStatus::Ok);
            }
        }
    }
    {
        FgDxgiPresenterDesc dd;
        dd.device = gpu.device.Get();
        dd.queue = gpu.queue.Get();
        dd.swap_chain = sc.Get();
        dd.pacing = FgDxgiPacing::Tearing;  // chain lacks the tearing flag: refused, not downgraded to VSync
        std::unique_ptr<FgDxgiPresenter> p;
        const FgWinResult r = FgDxgiPresenter::Create(dd, &p);
        CHECK(!p && (r.refusal == FgWinRefusal::SwapChainNoTearingFlag || r.refusal == FgWinRefusal::NoTearing));
    }
    FgDxgiPresenterDesc dd;
    dd.device = gpu.device.Get();
    dd.queue = gpu.queue.Get();
    dd.swap_chain = sc.Get();
    dd.pacing = FgDxgiPacing::Vsync;
    dd.max_generated = 1;
    std::unique_ptr<FgDxgiPresenter> p;
    const FgWinResult r = FgDxgiPresenter::Create(dd, &p);
    std::printf("  dxgi vsync: create %s / %s\n", FgStatusName(r.status), FgWinRefusalName(r.refusal));
    if (!r.Ok()) {
        CHECK(r.refusal == FgWinRefusal::RefreshUnknown && !p);  // no compositor grid: explicit refusal
        return;
    }
    const FgDxgiPresenterInfo info = p->Info();
    std::printf("  chain %ux%u, %u buffers, slots %u, grid period %lld\n", info.width, info.height, info.buffer_count,
                info.slot_count, static_cast<long long>(info.grid.period));
    // the default slot count: FgMinSlotsFor(N), at least 8 (fg_dxgi_presenter.cpp)
    const std::uint32_t default_slots = FgMinSlotsFor(1) < 8 ? 8u : FgMinSlotsFor(1);
    CHECK(info.slot_count == default_slots && info.buffer_count == 3 && info.width == 256);
    FgWinResult why;
    const std::uint32_t fid = p->RegisterFence(gpu.fence.Get(), &why);
    CHECK(fid >= 1);
    CHECK(p->Attach(0xBEEF, 1) == FgStatus::Ok);
    // 2x: one image per two refreshes of the compositor's grid (a whole-refresh cadence, as VSync mode requires)
    const FgTicks base = info.grid.period * 4;
    std::uint64_t gen = 0, refused = 0;
    const bool ok = DriveFrames(p.get(), &gpu, fid, 1, base, 16, &gen, &refused);
    Sleep(static_cast<DWORD>(base * 2 / kMs + 50));
    const FgPresentStats st = p->Stats();
    const FgDxgiPresenterInfo after = p->Info();
    std::printf("  presents %llu, latency timeouts %llu, real queued %llu, generated queued %llu, displayed %llu+%llu, "
                "late %llu, pressure %llu, refused %llu (plans %llu, refused here %llu)\n",
                static_cast<unsigned long long>(after.presents),
                static_cast<unsigned long long>(after.latency_timeouts),
                static_cast<unsigned long long>(st.real_queued), static_cast<unsigned long long>(st.generated_queued),
                static_cast<unsigned long long>(st.displayed_real),
                static_cast<unsigned long long>(st.displayed_generated),
                static_cast<unsigned long long>(st.dropped_late), static_cast<unsigned long long>(st.dropped_pressure),
                static_cast<unsigned long long>(st.refused_calls), static_cast<unsigned long long>(gen),
                static_cast<unsigned long long>(refused));
    CHECK(ok && after.presents >= 4 && st.displayed_real >= 4);
    CHECK(gen >= 4 && st.generated_queued >= 2 && st.displayed_generated >= 1);
    CHECK(after.presents == st.displayed_real + st.displayed_generated);  // one Present per shown image
    // resize: quiesce, the worker lets go of the buffers, ResizeBuffers succeeds, the new epoch takes the new size
    CHECK(p->Quiesce() == FgStatus::Ok);
    CHECK(p->WaitIdle(3000));
    CHECK(!p->HoldsSwapChainBuffers());
    std::printf("  dxgi resize after quiesce and queue drain\n");
    const HRESULT rr = sc->ResizeBuffers(0, 320, 180, DXGI_FORMAT_UNKNOWN,
                                         DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
    CHECK(SUCCEEDED(rr));
    CHECK(p->CompleteResize(SUCCEEDED(rr), 2) == FgStatus::Ok);
    CHECK(p->Info().width == 320 && p->HoldsSwapChainBuffers() && p->FreeSlots() == default_slots);
    const HANDLE latency = sc->GetFrameLatencyWaitableObject();
    CHECK(latency != nullptr);
    if (latency) {
        while (WaitForSingleObject(latency, 0) == WAIT_OBJECT_0) {}
        const FgTicks now = FgInterruptNow();
        const FgIntervalPlan real_plan = MakePlan(2, 1, 17, now, 10 * kMs, 0);
        std::uint32_t slot = 0;
        CHECK(p->ReserveRealSlot(&slot) == FgStatus::Ok);
        if (ID3D12Resource* target = p->SlotResource(slot)) {
            const std::uint64_t value = gpu.Render(target, 0.5f);
            CHECK(p->PresentRealOnly(real_plan, FgRealImage{2, 17, {fid, value}}) == FgStatus::Ok);
            CHECK(p->WaitIdle(1000));
            CHECK(p->Info().occlusion_retired_real == 1);
            CHECK(p->Stats().display_skipped_real == 1);
        } else {
            CHECK(false);
        }
        CloseHandle(latency);
    }
    CHECK(p->Quiesce() == FgStatus::Ok && p->WaitIdle(3000));
    FgStatus ds = p->Detach();
    for (int i = 0; i < 100 && ds == FgStatus::Busy; ++i) {
        Sleep(10);
        ds = p->Detach();
    }
    CHECK(ds == FgStatus::Ok);
    CHECK(p->Attach(0xBEEF, 3) == FgStatus::WrongState);  // buffers released: CompleteResize would have to retake them
}

// RC2 (A2 / A4 / N-B3): real-retire callback fires strictly once per real frame (displayed, skipped, cancelled by
// quiesce, NOT on CancelSuffix re-present); truthful funnel counters never double-count; 20 slots + 3 pending
// intervals sustain steady 2-interval 6x pipeline with 0 pressure drops.
void TestDxgiWorkerLeaseAndJitter() {
    g_case = "dxgi worker lease and +2 ms ready jitter";
    ScriptFences fences;
    const std::uint32_t fence = fences.Create();
    FgEngineConfig config = EngineConfig(6);
    config.pending_late_tolerance = 40000;
    FgPresentEngine engine(config, &fences);
    CHECK(engine.Attach(1) == FgStatus::Ok);
    std::uint32_t retired_real = 0;
    engine.SetRealRetireCallback([](void* context, std::uint64_t, std::uint64_t, bool) {
        ++*static_cast<std::uint32_t*>(context);
    }, &retired_real);
    std::vector<std::uint8_t> trace;
    engine.SetTraceCallback([](void* context, std::uint64_t, std::uint64_t, std::uint32_t, bool,
                               std::uint8_t kind, FgTicks, FgTicks) {
        static_cast<std::vector<std::uint8_t>*>(context)->push_back(kind);
    }, &trace);

    const FgTicks now = 100 * kMs;
    const FgIntervalPlan plan = MakePlan(1, 1, 1, now, 20 * kMs, 1);
    FgIntervalReservation reservation;
    CHECK(engine.BeginInterval(plan, now, &reservation) == FgStatus::Ok);
    FgIntervalSubmission submission;
    submission.epoch = 1;
    submission.interval_id = 1;
    submission.count = 2;
    submission.produced[0] = submission.produced[1] = true;
    submission.ready[0] = {fence, 1};
    submission.ready[1] = {fence, 2};
    CHECK(engine.SubmitInterval(submission, now) == FgStatus::Ok);
    fences.Set(fence, 1);
    const FgTicks ready = plan.targets[0].target + 2 * kMs;
    engine.Pump(ready);
    std::vector<FgEnginePresent> out;
    engine.TakeOutbox(&out);
    CHECK(out.size() == 1 && !out[0].real);
    if (out.size() != 1) return;
    const FgEnginePresent generated = out[0];
    CHECK(engine.LeasePresent(generated.slot, generated.present_id));

    // Game-thread operations while the worker is outside the presenter lock doing its native copy/present.
    std::uint64_t cancel_from = 0;
    CHECK(engine.CancelPresentsFrom(generated.present_id, ready, &cancel_from) == 0);
    engine.RevokeGenerated(plan.interval_id, FgDropReason::Revoked, ready, &cancel_from);
    CHECK(!engine.OnSlotReleased(generated.slot, false));
    std::uint32_t another_slot = 0;
    CHECK(engine.ReserveRealOnlySlot(&another_slot) == FgStatus::Ok && another_slot != generated.slot);
    engine.ReleaseRealOnlySlot();
    engine.FinishLeasedPresent(generated.slot, ready, ready, true);
    CHECK(engine.OnSlotReleased(generated.slot, true));

    fences.Set(fence, 2);
    engine.Pump(ready);
    out.clear();
    engine.TakeOutbox(&out);
    CHECK(out.size() == 1 && out[0].real);
    if (out.size() != 1) return;
    const FgEnginePresent real = out[0];
    CHECK(engine.LeasePresent(real.slot, real.present_id));
    CHECK(engine.Quiesce(ready, &cancel_from) == FgStatus::Ok);
    CHECK(!engine.Drained() && retired_real == 0 && !engine.OnSlotReleased(real.slot, false));
    CHECK(engine.Detach(ready, &cancel_from) == FgStatus::Busy);
    engine.FinishLeasedPresent(real.slot, real.target, real.target, true);
    CHECK(retired_real == 0 && !engine.Drained());
    CHECK(engine.OnSlotReleased(real.slot, true));
    CHECK(retired_real == 1 && engine.Drained());
    CHECK(engine.Stats().generated_submitted == 1 && engine.Stats().dropped_late == 0);
    CHECK(!engine.Stats().display_time_measured);
    CHECK(std::count(trace.begin(), trace.end(), 6u) == 2);
}

void TestRc2EngineContracts() {
    g_case = "rc2-engine-contracts";
    {
        struct RetireLog {
            int count = 0;
            int displayed = 0;
            std::vector<std::uint64_t> frames;
            static void Fn(void* ctx, std::uint64_t frame_id, std::uint64_t, bool disp) {
                RetireLog* l = static_cast<RetireLog*>(ctx);
                ++l->count;
                if (disp) ++l->displayed;
                l->frames.push_back(frame_id);
            }
        } log;
        ScriptFences fences;
        const std::uint32_t f = fences.Create();
        fences.Set(f, 100);
        FgPresentEngine e(EngineConfig(12), &fences);
        e.SetRealRetireCallback(&RetireLog::Fn, &log);
        CHECK(e.Attach(1) == FgStatus::Ok);
        const FgTicks t0 = 100 * kMs;
        // 1) displayed real frame -> retires once (displayed=true)
        FgIntervalPlan p1 = MakePlan(1, 1, 101, t0, 166667, 0);
        CHECK(e.PresentRealOnly(p1, FgRealImage{1, 101, {f, 1}}) == FgStatus::Ok);
        e.Pump(t0);
        std::vector<FgEnginePresent> out;
        e.TakeOutbox(&out);
        CHECK(out.size() == 1 && log.count == 0);
        CHECK(e.OnPresentDisplayed(out[0].present_id, out[0].target));
        CHECK(log.count == 1 && log.displayed == 1 && log.frames.back() == 101);
        CHECK(e.OnSlotReleased(out[0].slot, true));
        CHECK(log.count == 1);  // releasing the slot does NOT retire again!

        // 2) skipped real frame -> retires once (displayed=false)
        FgIntervalPlan p2 = MakePlan(1, 2, 102, p1.targets[0].target, 166667, 0);
        CHECK(e.PresentRealOnly(p2, FgRealImage{1, 102, {f, 2}}) == FgStatus::Ok);
        e.Pump(t0);
        out.clear();
        e.TakeOutbox(&out);
        CHECK(out.size() == 1);
        CHECK(e.OnPresentSkipped(out[0].present_id));
        CHECK(log.count == 2 && log.displayed == 1 && log.frames.back() == 102);
        CHECK(e.OnSlotReleased(out[0].slot, false));
        CHECK(log.count == 2);

        // 3) CancelSuffix re-presents the real frame -> must NOT retire until the re-presented real frame displays
        FgIntervalPlan p3 = MakePlan(1, 3, 103, p2.targets[0].target, 166667, 1);
        FgIntervalReservation res;
        CHECK(e.BeginInterval(p3, t0, &res) == FgStatus::Ok);
        FgIntervalSubmission sub;
        sub.epoch = 1;
        sub.interval_id = 3;
        sub.count = 2;
        sub.produced[0] = sub.produced[1] = true;
        sub.ready[0] = FgReadyFence{f, 3};
        sub.ready[1] = FgReadyFence{f, 4};
        CHECK(e.SubmitInterval(sub, t0) == FgStatus::Ok);
        out.clear();
        e.TakeOutbox(&out);
        CHECK(out.size() == 2);
        std::uint64_t cancel_from = 0;
        CHECK(e.CancelPresentsFrom(out[0].present_id, t0, &cancel_from) == 2);
        CHECK(log.count == 2);  // NOT retired by CancelSuffix!
        out.clear();
        e.TakeOutbox(&out);
        CHECK(out.size() == 1 && out[0].real);
        CHECK(e.OnPresentDisplayed(out[0].present_id, out[0].target));
        CHECK(log.count == 3 && log.displayed == 2 && log.frames.back() == 103);

        // 4) Quiesce cancels a queued real frame -> retires once
        FgIntervalPlan p4 = MakePlan(1, 4, 104, p3.targets[1].target, 166667, 0);
        CHECK(e.PresentRealOnly(p4, FgRealImage{1, 104, {f, 5}}) == FgStatus::Ok);
        e.Pump(t0);
        out.clear();
        e.TakeOutbox(&out);
        CHECK(out.size() == 1);
        CHECK(e.Quiesce(t0, &cancel_from) == FgStatus::Ok);
        CHECK(log.count == 4 && log.displayed == 2 && log.frames.back() == 104);
        CHECK(e.OnSlotReleased(out[0].slot, false));
        CHECK(log.count == 4);
    }
    {
        // A4: truthful funnel counters never double-count pre-queue vs post-queue drops
        ScriptFences fences;
        const std::uint32_t f = fences.Create();
        FgPresentEngine e(EngineConfig(12), &fences);
        CHECK(e.Attach(1) == FgStatus::Ok);
        const FgTicks t0 = 500 * kMs;
        // interval 1: index 1 past deadline at reservation (pre-queue late), index 2 expires while still pending
        FgIntervalPlan p = MakePlan(1, 1, 1, t0, 166667, 2);
        FgIntervalReservation res;
        CHECK(e.BeginInterval(p, p.targets[0].deadline + 1, &res) == FgStatus::Ok);
        CHECK(!res.slots[0].reserved && res.slots[1].reserved && res.slots[2].reserved);
        FgIntervalSubmission sub;
        sub.epoch = 1;
        sub.interval_id = 1;
        sub.count = 3;
        sub.produced[1] = sub.produced[2] = true;
        sub.ready[1] = FgReadyFence{f, 2};
        sub.ready[2] = FgReadyFence{f, 3};
        CHECK(e.SubmitInterval(sub, p.targets[0].deadline + 1) == FgStatus::Ok);
        fences.Set(f, 3);
        e.Pump(p.targets[1].deadline + 1);  // index 2 expires while still pending; real goes out
        std::vector<FgEnginePresent> out;
        e.TakeOutbox(&out);
        CHECK(out.size() == 1 && out[0].real);
        CHECK(e.OnPresentSubmitted(out[0].present_id, p.targets[2].target));
        const FgPresentStats st = e.Stats();
        CHECK(st.generated_requested == 2 && st.generated_scheduled == 2 && st.generated_reserved == 1);
        CHECK(st.generated_dropped_pre_queue == 2 && st.generated_dropped_post_queue == 0);
        CHECK(st.dropped_late == 2 && st.dropped_pressure == 0);
        CHECK(st.dropped_late + st.dropped_pressure == st.generated_dropped_pre_queue + st.generated_dropped_post_queue);
        CHECK(st.real_submitted == 1 && st.generated_submitted == 0);
    }
    {
        // N-B3: steady 2-interval pipeline at 6x (N=5) with 20 slots and max_pending_intervals=3 -> 0 dropped_pressure
        ScriptFences fences;
        const std::uint32_t f = fences.Create();
        FgEngineConfig ec = EngineConfig(20);
        ec.max_pending_intervals = 3;
        FgPresentEngine e(ec, &fences);
        CHECK(e.Attach(1) == FgStatus::Ok);
        const FgTicks period = 166667;
        FgTicks t = 100 * kMs;
        std::uint64_t fv = 0;
        std::vector<FgEnginePresent> out;
        for (std::uint64_t j = 1; j <= 30; ++j) {
            // In a 2-interval playout pipeline, interval j (t_base = t) arrives at ~t - period (before interval j-1's
            // generated deadlines, since interval j-1 has t_base = t - period).
            const FgTicks arrival = t - period;
            FgIntervalPlan p = MakePlan(1, j, j, t, period, 5);
            FgIntervalReservation res;
            CHECK(e.BeginInterval(p, arrival, &res) == FgStatus::Ok);
            FgIntervalSubmission sub;
            sub.epoch = 1;
            sub.interval_id = j;
            sub.count = res.count;
            for (std::uint32_t i = 0; i < res.count; ++i) {
                CHECK(res.slots[i].reserved);
                sub.produced[i] = true;
                sub.ready[i] = FgReadyFence{f, ++fv};
            }
            CHECK(e.SubmitInterval(sub, arrival) == FgStatus::Ok);
            // keep 2 intervals pending simultaneously: retire interval j-1 while j remains pending
            if (j >= 2) {
                fences.Set(f, (j - 1) * 6);
                e.Pump(arrival);
                e.TakeOutbox(&out);
                for (const FgEnginePresent& pr : out) {
                    CHECK(e.OnPresentDisplayed(pr.present_id, pr.target));
                    CHECK(e.OnSlotReleased(pr.slot, true));
                }
                out.clear();
            }
            t = p.targets[5].target;
        }
        fences.Set(f, fv);
        e.Pump(t - period);
        e.TakeOutbox(&out);
        for (const FgEnginePresent& pr : out) {
            CHECK(e.OnPresentDisplayed(pr.present_id, pr.target));
            CHECK(e.OnSlotReleased(pr.slot, true));
        }
        const FgPresentStats st = e.Stats();
        CHECK(st.dropped_pressure == 0 && st.dropped_late == 0);
        CHECK(st.displayed_real == 30 && st.displayed_generated == 150);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash must not swallow the log
    bool gpu = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) gpu = true;
        if (std::strcmp(argv[i], "--debug-layer") == 0) g_debug_layer = true;
    }
    TestPacingMath();
    TestEngineBasicFlow();
    TestEngineOwnership();
    TestEngineNoSlotUntilReleased();
    TestEngineLateAndLeases();
    TestEngineCancelAndRevoke();
    TestEngineFeedback();
    TestEngineLifecycle();
    TestRc2EngineContracts();
    TestDxgiWorkerLeaseAndJitter();
    TestCadenceMatrix();
    TestSelection();
    TestCreationRefusals();
    TestWorkerLifecycle();
    TestTimerAndClock();
    TestReadOnlyProbe();
    if (gpu) {
        Gpu g;
        g_case = "gpu-init";
        const bool init = g.Init();
        CHECK(init);
        if (init) {
            TestGpuProbe(g);
            TestGpuComposition(g);
            TestGpuDxgi(g);
            CHECK(g.DebugLayerClean());
        }
    }
    std::printf("test_fg_win_presenters%s: %d checks, %d failure(s)\n", gpu ? " --gpu" : "", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
