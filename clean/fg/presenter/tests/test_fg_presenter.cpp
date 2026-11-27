// SPDX-License-Identifier: GPL-3.0-only
// CPU tests of the frame generation presentation boundary: the target scheduler (fg_target_scheduler.h) driving the
// deterministic fake presenter (fg_fake_presenter.h). Synthetic cadences only; no GPU, no window.
//
// Covered: exact target spacing and strict order for 2x..6x at several real cadences on a 48-360 Hz display, the
// interval estimator, count changes latched at game frames, menu / reset transitions, late drops with held leases,
// suffix cancellation and the reserved real path, the variable-refresh window with hysteresis, resize / epoch /
// device loss, refusal without fallback, and mutants (broken plans, a broken spacing oracle) that must be caught.
#include <deque>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include "../fg_fake_presenter.h"
#include "../fg_present_engine.h"
#include "../fg_target_scheduler.h"

using namespace wgfg;

namespace {

int g_checks = 0;
int g_failures = 0;
const char* g_case = "";

void Check(bool ok, const char* what, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL [%s] line %d: %s\n", g_case, line, what);
    }
}
#define CHECK(x) Check((x), #x, __LINE__)

constexpr FgTicks kSecond = 10000000;  // 100 ns ticks
constexpr std::uint64_t kChain = 77;

// Arrival i of a cadence at `fps` that starts at t0: t0 + round(i * 1 s / fps).
FgTicks At(FgTicks t0, std::uint64_t i, std::uint32_t fps) {
    return t0 + (static_cast<FgTicks>(i) * kSecond + fps / 2) / fps;
}

// Independent spacing oracle: target[k] == t_base + floor((2kT + d) / 2d), d = N + 1 (halves up), the real endpoint
// at t_base + T, and every target within half a tick of the exact rational time.
bool ExactSpacingOk(const FgIntervalPlan& p) {
    const FgTicks d = static_cast<FgTicks>(p.generated_count) + 1;
    for (std::uint32_t k = 1; k <= p.generated_count; ++k) {
        const FgTicks expect = p.t_base + (2 * static_cast<FgTicks>(k) * p.period + d) / (2 * d);
        if (p.targets[k - 1].target != expect) return false;
        const long double exact = static_cast<long double>(p.t_base) +
                                  static_cast<long double>(k) * static_cast<long double>(p.period) /
                                      static_cast<long double>(d);
        if (std::fabs(static_cast<long double>(p.targets[k - 1].target) - exact) > 0.5L) return false;
    }
    return p.targets[p.generated_count].target == p.t_base + p.period && p.targets[p.generated_count].real;
}

FgSchedulerConfig DefaultScheduler() {
    FgSchedulerConfig c;  // 48-360 Hz window, 1 ms margin, 2 ms first-image budget, 50 ms lead cap
    return c;
}

FgFakeConfig DefaultFake() {
    FgFakeConfig c;
    c.min_flip_interval = 27778;  // the 360 Hz panel cannot flip faster
    return c;
}

// Scheduler + fake presenter + a synthetic executor: generated index k of an interval is ready at
// arrival + ready_offset[k] (negative: never, until released), the real image at arrival.
struct Rig {
    FgSchedulerConfig scfg;
    FgFakeConfig pcfg;
    FgFakeClock clock;
    FgFakeGpu gpu;
    FgFakeChainRegistry registry;
    FgTargetScheduler scheduler;
    FgFakePresenter presenter;
    std::uint32_t gen_fence[kFgMaxTargets + 1] = {};
    std::uint64_t gen_value[kFgMaxTargets + 1] = {};
    std::uint32_t real_fence = 0;
    std::uint64_t real_value = 0;
    FgTicks ready_offset[kFgMaxTargets + 1] = {0, 2000, 4000, 6000, 8000, 10000, 12000};
    std::uint64_t epoch = 1;
    std::uint64_t frame_id = 0;
    std::vector<FgIntervalPlan> plans;
    std::vector<FgStatus> statuses;
    FgTicks step = 1000;

    Rig(const FgSchedulerConfig& s, const FgFakeConfig& p)
        : scfg(s), pcfg(p), scheduler(s), presenter(p, &clock, &gpu, &registry) {
        for (std::uint32_t k = 1; k <= kFgMaxTargets; ++k) gen_fence[k] = gpu.CreateFence();
        real_fence = gpu.CreateFence();
        clock.Set(kSecond);
        attach_status = presenter.Attach(kChain, epoch);
        epoch_status = scheduler.BeginEpoch(epoch);
        plans.reserve(1024);
        statuses.reserve(1024);
    }
    FgStatus attach_status;
    FgStatus epoch_status;

    void RunUntil(FgTicks t) {
        while (clock.Now() < t) {
            const FgTicks d = t - clock.Now();
            clock.Advance(d < step ? d : step);
            presenter.Pump();
        }
    }

    FgStatus Frame(FgTicks arrival, std::uint32_t n, bool enabled = true, bool reset = false) {
        RunUntil(arrival);
        FgGameFrame f;
        f.epoch = epoch;
        f.frame_id = ++frame_id;
        f.arrival = arrival;
        f.requested_count = n;
        f.generation_enabled = enabled;
        f.reset = reset;
        FgIntervalPlan plan;
        FgStatus s = scheduler.Commit(f, &plan);
        if (s == FgStatus::Ok) {
            plans.push_back(plan);
            s = Present(plan, arrival);
        }
        statuses.push_back(s);
        return s;
    }

    FgStatus Present(const FgIntervalPlan& plan, FgTicks arrival) {
        ++real_value;
        gpu.SignalAt(real_fence, real_value, arrival);
        if (plan.generated_count == 0) {
            FgRealImage img;
            img.epoch = plan.epoch;
            img.frame_id = plan.real_frame_id;
            img.ready = FgReadyFence{real_fence, real_value};
            return presenter.PresentRealOnly(plan, img);
        }
        FgIntervalReservation res;
        FgStatus s = presenter.BeginInterval(plan, &res);
        if (s != FgStatus::Ok) return s;
        FgIntervalSubmission sub;
        sub.epoch = res.epoch;
        sub.interval_id = res.interval_id;
        sub.count = res.count;
        for (std::uint32_t i = 0; i < res.count; ++i) {
            const FgPresentSlot& ps = res.slots[i];
            if (!ps.reserved) continue;
            sub.produced[i] = true;
            if (ps.real) {
                sub.ready[i] = FgReadyFence{real_fence, real_value};
                continue;
            }
            const std::uint32_t k = ps.index;
            const std::uint64_t v = ++gen_value[k];
            sub.ready[i] = FgReadyFence{gen_fence[k], v};
            if (ready_offset[k] >= 0) gpu.SignalAt(gen_fence[k], v, arrival + ready_offset[k]);
        }
        return presenter.SubmitInterval(sub);
    }

    // Completes every generated fence value issued so far (a stalled executor finishing).
    void ReleaseStalled(FgTicks when) {
        for (std::uint32_t k = 1; k <= kFgMaxTargets; ++k) gpu.SignalAt(gen_fence[k], gen_value[k], when);
    }

    std::uint32_t CountReason(FgRealOnlyReason r) const {
        std::uint32_t n = 0;
        for (const FgIntervalPlan& p : plans)
            if (p.generated_count == 0 && p.real_only_reason == r) ++n;
        return n;
    }
    std::uint32_t Generating() const {
        std::uint32_t n = 0;
        for (const FgIntervalPlan& p : plans)
            if (p.generated_count > 0) ++n;
        return n;
    }
    const FgIntervalPlan* PlanFor(std::uint64_t interval_id) const {
        for (const FgIntervalPlan& p : plans)
            if (p.interval_id == interval_id) return &p;
        return nullptr;
    }
};

bool AllOk(const Rig& r) {
    for (FgStatus s : r.statuses)
        if (s != FgStatus::Ok) return false;
    return true;
}

// Displayed targets and actual flips strictly increasing; present ids increasing.
bool DisplayMonotonic(const Rig& r) {
    const std::vector<FgFakeDisplayRecord>& d = r.presenter.Displayed();
    for (std::size_t i = 1; i < d.size(); ++i)
        if (d[i].target <= d[i - 1].target || d[i].actual <= d[i - 1].actual || d[i].present_id <= d[i - 1].present_id)
            return false;
    return true;
}

// Every displayed image of an interval matches its plan: generated images belong to a generating plan with that
// index and target; the display order inside an interval is 1..N, then the real image.
bool DisplayMatchesPlans(const Rig& r) {
    for (const FgFakeDisplayRecord& rec : r.presenter.Displayed()) {
        const FgIntervalPlan* p = r.PlanFor(rec.interval_id);
        if (!p || rec.epoch != p->epoch) return false;
        const std::uint32_t idx = rec.real ? p->generated_count : rec.index - 1;
        if (idx >= p->target_count || p->targets[idx].target != rec.target || p->targets[idx].real != rec.real)
            return false;
    }
    return true;
}

std::uint32_t DisplayedGeneratedOf(const Rig& r, std::uint64_t interval_id) {
    std::uint32_t n = 0;
    for (const FgFakeDisplayRecord& rec : r.presenter.Displayed())
        if (!rec.real && rec.interval_id == interval_id) ++n;
    return n;
}

std::uint32_t DisplayedRealOf(const Rig& r, std::uint64_t frame_id) {
    std::uint32_t n = 0;
    for (const FgFakeDisplayRecord& rec : r.presenter.Displayed())
        if (rec.real && rec.frame_id == frame_id) ++n;
    return n;
}

// --- Cases ---------------------------------------------------------------------------------------------------------

void TestIntermediateTime() {
    g_case = "intermediate time";
    for (std::uint32_t n = 1; n <= kFgMaxGenerated; ++n) {
        float previous = 0.0f;
        for (std::uint32_t k = 1; k <= n; ++k) {
            const float t = FgIntermediateTime(k, n);
            // one correctly rounded single-precision quotient (the double quotient rounds to the same float here)
            CHECK(t == static_cast<float>(static_cast<double>(k) / static_cast<double>(n + 1)));
            CHECK(t > previous && t < 1.0f);
            previous = t;
        }
    }
    CHECK(FgIntermediateTime(1, 1) == 0.5f);
    CHECK(FgIntermediateTime(3, 5) == 0.5f);
    CHECK(FgIntermediateTime(2, 3) == 0.5f);
}

void TestCheckedArithmetic() {
    g_case = "checked arithmetic";
    FgTicks out = 0;
    CHECK(FgScaledOffset(1, 3, 2, &out) == FgStatus::Ok && out == 2);  // 1.5 -> 2
    CHECK(FgScaledOffset(1, 1, 2, &out) == FgStatus::Ok && out == 1);  // 0.5 -> 1
    CHECK(FgScaledOffset(1, 1, 3, &out) == FgStatus::Ok && out == 0);  // 0.33 -> 0
    CHECK(FgScaledOffset(2, 1, 3, &out) == FgStatus::Ok && out == 1);  // 0.67 -> 1
    CHECK(FgScaledOffset(0, 12345, 6, &out) == FgStatus::Ok && out == 0);
    // a synthetic sweep against the independent half-up formula
    std::uint64_t lcg = 0x2545F4914F6CDD1DULL;
    bool sweep_ok = true;
    for (int i = 0; i < 20000; ++i) {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        const FgTicks period = static_cast<FgTicks>((lcg >> 20) % 3000000);
        for (std::uint32_t d = 1; d <= kFgMaxTargets; ++d) {
            for (std::uint32_t k = 0; k <= d; ++k) {
                const FgTicks expect = (2 * static_cast<FgTicks>(k) * period + d) / (2 * static_cast<FgTicks>(d));
                if (FgScaledOffset(k, period, d, &out) != FgStatus::Ok || out != expect) sweep_ok = false;
            }
        }
    }
    CHECK(sweep_ok);
    const FgTicks big = std::numeric_limits<FgTicks>::max();
    CHECK(FgScaledOffset(3, big / 3 + 1, 4, &out) == FgStatus::Overflow);
    CHECK(FgScaledOffset(1, big, 6, &out) == FgStatus::Ok && out == big / 6 + ((big % 6) * 2 >= 6 ? 1 : 0));
    CHECK(FgScaledOffset(1, -5, 2, &out) == FgStatus::InvalidArgument);
    CHECK(FgScaledOffset(1, 5, 0, &out) == FgStatus::InvalidArgument);
    CHECK(FgCheckedAdd(big, 1, &out) == FgStatus::Overflow);
    CHECK(FgCheckedAdd(std::numeric_limits<FgTicks>::min(), -1, &out) == FgStatus::Overflow);
    CHECK(FgCheckedAdd(big - 1, 1, &out) == FgStatus::Ok && out == big);
    FgFrameTarget t[kFgMaxTargets];
    CHECK(FgBuildTargets(0, 5, 5, 0, t) == FgStatus::InvalidArgument);       // T < N + 1: no strict order
    CHECK(FgBuildTargets(0, 600, 6, 0, t) == FgStatus::InvalidArgument);     // N > 5
    CHECK(FgBuildTargets(big - 100, 1000, 3, 0, t) == FgStatus::Overflow);  // targets past the clock's range
    CHECK(FgBuildTargets(1000, 600, 5, -1, t) == FgStatus::InvalidArgument);
    CHECK(FgBuildTargets(1000, 6, 5, 0, t) == FgStatus::Ok);
    bool strict = true;
    for (std::uint32_t i = 1; i < 6; ++i) strict = strict && t[i].target > t[i - 1].target;
    CHECK(strict && t[0].target > 1000 && t[5].target == 1006);
}

void TestEstimator() {
    g_case = "interval estimator";
    using Ev = FgIntervalEstimator::Event;
    FgIntervalEstimator e;
    const FgTicks t0 = kSecond;
    CHECK(e.Push(At(t0, 0, 60)) == Ev::First);
    CHECK(!e.Stable() && e.Period() == 0);
    for (std::uint64_t i = 1; i <= 3; ++i) CHECK(e.Push(At(t0, i, 60)) == Ev::Sample);
    CHECK(e.Stable());
    CHECK(e.Period() == 166666 || e.Period() == 166667);
    for (std::uint64_t i = 4; i < 30; ++i) e.Push(At(t0, i, 60));
    CHECK(e.Period() == 166666 || e.Period() == 166667);

    // jitter of up to +-3 % stays inside the median
    FgIntervalEstimator j;
    FgTicks t = t0;
    j.Push(t);
    const FgTicks jitter[8] = {5000, -4000, 3000, -5000, 1000, 4000, -2000, 0};
    bool jitter_ok = true;
    for (int i = 0; i < 64; ++i) {
        t += 166667 + jitter[i % 8];
        if (j.Push(t) != Ev::Sample) jitter_ok = false;
    }
    CHECK(jitter_ok && j.Period() >= 166667 - 5000 && j.Period() <= 166667 + 5000);

    // a single hitch is an outlier and leaves the estimate alone
    const FgTicks before = e.Period();
    FgTicks last = At(t0, 29, 60);
    last += 2 * 166667;
    CHECK(e.Push(last) == Ev::Outlier);
    CHECK(e.Period() == before);
    last += 166667;
    CHECK(e.Push(last) == Ev::Sample);
    CHECK(e.Period() == before);

    // a cadence change (60 -> 120 fps): two outliers, then a discontinuity with the new period
    CHECK(e.Push(last + 83333) == Ev::Outlier);
    CHECK(e.Push(last + 166667) == Ev::Outlier);
    CHECK(e.Push(last + 250000) == Ev::Discontinuity);
    CHECK(e.Stable() && e.Period() >= 83333 && e.Period() <= 83334);
    last += 250000;

    // a long gap, a duplicate present and a backwards clock all restart the window
    CHECK(e.Push(last + 3000000) == Ev::Discontinuity);
    CHECK(!e.Stable());
    last += 3000000;
    CHECK(e.Push(last) == Ev::Discontinuity);
    CHECK(e.Push(last - 10) == Ev::Discontinuity);
}

// 2x..6x at 45 / 50 / 60 / 90 / 120 fps on a 48-360 Hz panel. A combination generates exactly when its image time
// T / (N + 1) lies inside the window less the margin; otherwise every interval is real-only (out of window).
void TestSpacingMatrix() {
    g_case = "spacing matrix";
    const std::uint32_t fps_list[5] = {45, 50, 60, 90, 120};
    std::uint32_t combos_generating = 0;
    std::uint32_t combos_suspended = 0;
    for (std::uint32_t fps : fps_list) {
        for (std::uint32_t n = 1; n <= kFgMaxGenerated; ++n) {
            Rig r(DefaultScheduler(), DefaultFake());
            const int frames = 150;
            for (int i = 0; i < frames; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), fps), n);
            r.RunUntil(At(kSecond, static_cast<std::uint64_t>(frames), fps) + 400000);
            CHECK(AllOk(r));
            const double image_time = static_cast<double>(kSecond) / fps / (n + 1);
            const FgDisplayWindow& w = r.scfg.display;
            const bool in_window = image_time >= static_cast<double>(w.min_interval + w.margin) &&
                                   image_time <= static_cast<double>(w.max_interval - w.margin);
            const FgPresentStats st = r.presenter.Stats();
            CHECK(DisplayMonotonic(r));
            CHECK(DisplayMatchesPlans(r));
            CHECK(st.max_slip == 0);
            CHECK(st.game_presents == static_cast<std::uint64_t>(frames));
            CHECK(st.displayed_real == static_cast<std::uint64_t>(frames));  // every game frame shown once
            if (in_window) {
                ++combos_generating;
                // warm-up: a few real-only frames (no cadence yet, then one anchor), then every interval generates
                CHECK(r.Generating() >= static_cast<std::uint32_t>(frames - 6));
                bool exact = true;
                bool counts = true;
                bool steady = true;
                bool all_admitted = true;
                const FgTicks nominal = kSecond / fps;
                for (const FgIntervalPlan& p : r.plans) {
                    if (p.generated_count == 0) continue;
                    exact = exact && ExactSpacingOk(p);
                    counts = counts && p.generated_count == n && p.requested_count == n;
                    steady = steady && p.period >= nominal - nominal / 128 - 1 && p.period <= nominal + nominal / 128 + 1;
                    for (std::uint32_t k = 0; k < n; ++k) all_admitted = all_admitted && p.targets[k].admitted;
                }
                CHECK(exact);
                CHECK(counts);
                CHECK(steady);
                CHECK(all_admitted);
                CHECK(st.dropped_late == 0 && st.dropped_pressure == 0 && st.cancelled == 0);
                CHECK(st.displayed_generated == static_cast<std::uint64_t>(r.Generating()) * n);
                // the display shows N generated images between consecutive real ones
                bool pattern = true;
                for (const FgIntervalPlan& p : r.plans)
                    if (p.generated_count > 0) pattern = pattern && DisplayedGeneratedOf(r, p.interval_id) == n;
                CHECK(pattern);
            } else {
                ++combos_suspended;
                CHECK(r.Generating() == 0);
                CHECK(st.displayed_generated == 0);
                CHECK(r.CountReason(FgRealOnlyReason::OutOfVrrWindow) >= static_cast<std::uint32_t>(frames - 6));
            }
            if (g_failures) {
                std::printf("  (fps %u, N %u, in window %d)\n", fps, n, in_window ? 1 : 0);
                return;
            }
        }
    }
    // 45 and 50 fps: all of 2x..6x; 60: up to 5x; 90: up to 3x; 120: 2x
    CHECK(combos_generating == 5 + 5 + 4 + 2 + 1);
    CHECK(combos_suspended == 25 - combos_generating);
}

// 2x -> 4x -> 6x -> 2x -> 5x at 50 fps: each plan uses the count requested with its own game frame; an increase may
// cost one re-anchored real-only interval (more lead), never a mixed interval.
void TestDynamicCount() {
    g_case = "dynamic count";
    Rig r(DefaultScheduler(), DefaultFake());
    const std::uint32_t schedule[5] = {1, 3, 5, 1, 4};
    const int seg = 60;
    for (int i = 0; i < 5 * seg; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 50), schedule[i / seg]);
    r.RunUntil(At(kSecond, 5 * seg, 50) + 400000);
    CHECK(AllOk(r));
    CHECK(r.plans.size() == static_cast<std::size_t>(5 * seg));
    bool latched = true;
    for (std::size_t i = 0; i < r.plans.size(); ++i) {
        const FgIntervalPlan& p = r.plans[i];
        const std::uint32_t want = schedule[i / seg];
        latched = latched && p.requested_count == want && (p.generated_count == 0 || p.generated_count == want);
        if (p.generated_count > 0) latched = latched && ExactSpacingOk(p);
    }
    CHECK(latched);
    for (int s = 1; s < 5; ++s) {
        std::uint32_t real_only = 0;
        std::uint32_t generating = 0;
        for (int i = s * seg; i < (s + 1) * seg; ++i) {
            const FgIntervalPlan& p = r.plans[static_cast<std::size_t>(i)];
            if (p.generated_count == 0) {
                ++real_only;
                CHECK(p.real_only_reason == FgRealOnlyReason::Reanchor);
            } else {
                ++generating;
            }
        }
        CHECK(real_only <= 1);
        CHECK(real_only == 0 || schedule[s] > schedule[s - 1]);  // only an increase needs more lead
        CHECK(generating >= static_cast<std::uint32_t>(seg - 1));
    }
    const FgPresentStats st = r.presenter.Stats();
    CHECK(DisplayMonotonic(r));
    CHECK(DisplayMatchesPlans(r));
    CHECK(st.max_slip == 0 && st.dropped_late == 0 && st.cancelled == 0);
    bool per_interval = true;
    for (const FgIntervalPlan& p : r.plans)
        per_interval = per_interval && DisplayedGeneratedOf(r, p.interval_id) == p.generated_count;
    CHECK(per_interval);
}

// Menu frames and a camera cut: the affected intervals present only their real image, history re-primes, and the
// game's presents each show exactly one real image.
void TestMenuAndReset() {
    g_case = "menu and reset";
    {
        Rig r(DefaultScheduler(), DefaultFake());
        const int frames = 80;
        for (int i = 0; i < frames; ++i) {
            const bool menu = i >= 30 && i < 40;
            const bool reset = i == 55;
            r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3, !menu, reset);
        }
        r.RunUntil(At(kSecond, frames, 60) + 400000);
        CHECK(AllOk(r));
        for (int i = 30; i < 40; ++i) {
            const FgIntervalPlan& p = r.plans[static_cast<std::size_t>(i)];
            CHECK(p.generated_count == 0 && p.real_only_reason == FgRealOnlyReason::MenuDisabled);
            CHECK(DisplayedGeneratedOf(r, p.interval_id) == 0);
        }
        CHECK(r.plans[40].generated_count == 0 && r.plans[40].real_only_reason == FgRealOnlyReason::NotPrimed);
        CHECK(r.plans[41].generated_count == 3);
        CHECK(r.plans[29].generated_count == 3);
        CHECK(r.plans[55].generated_count == 0 && r.plans[55].real_only_reason == FgRealOnlyReason::Reset);
        CHECK(r.plans[56].generated_count == 3);  // the cut frame primed history
        // the cadence held through the menu: no re-anchor, targets continue one period apart
        CHECK(!r.plans[35].reanchored);
        CHECK(r.plans[41].reanchored && r.plans[41].generated_count > 0);
        const FgPresentStats st = r.presenter.Stats();
        CHECK(st.real_only_intervals[static_cast<int>(FgRealOnlyReason::MenuDisabled)] == 10);
        CHECK(st.real_only_intervals[static_cast<int>(FgRealOnlyReason::Reset)] == 1);
        CHECK(st.game_presents == static_cast<std::uint64_t>(frames));
        bool one_real_each = true;
        for (const FgIntervalPlan& p : r.plans) one_real_each = one_real_each && DisplayedRealOf(r, p.real_frame_id) == 1;
        CHECK(one_real_each);
        CHECK(DisplayMonotonic(r) && DisplayMatchesPlans(r));
        CHECK(st.max_slip == 0);
    }
    {
        // menu frames that keep history valid (configuration): generation resumes on the first enabled frame
        FgSchedulerConfig c = DefaultScheduler();
        c.menu_reprimes_history = false;
        Rig r(c, DefaultFake());
        for (int i = 0; i < 50; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3, !(i >= 20 && i < 25));
        CHECK(AllOk(r));
        CHECK(r.plans[24].real_only_reason == FgRealOnlyReason::MenuDisabled);
        CHECK(r.plans[25].generated_count == 3);
    }
    {
        // a late menu decision for an interval already submitted and queued: its generated images are revoked, the
        // real image stays (re-presented as a fresh endpoint) and is shown exactly once, at its target
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 20; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        const FgTicks a = At(kSecond, 20, 60);
        r.Frame(a, 3);
        const FgIntervalPlan victim = r.plans.back();
        CHECK(victim.generated_count == 3);
        r.RunUntil(a + 9000);  // all three generated and the real image are queued, none shown yet
        CHECK(r.presenter.QueuedCount() >= 4);
        CHECK(DisplayedGeneratedOf(r, victim.interval_id) == 0);
        const FgPresentStats before = r.presenter.Stats();
        CHECK(r.presenter.RevokeGenerated(victim.interval_id, FgDropReason::Revoked) == FgStatus::Ok);
        for (int i = 21; i < 40; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        r.RunUntil(At(kSecond, 40, 60) + 400000);
        CHECK(AllOk(r));
        const FgPresentStats st = r.presenter.Stats();
        CHECK(DisplayedGeneratedOf(r, victim.interval_id) == 0);
        CHECK(DisplayedRealOf(r, victim.real_frame_id) == 1);
        CHECK(st.revoked - before.revoked == 3);
        CHECK(st.real_represented - before.real_represented >= 1);
        bool real_on_time = false;
        for (const FgFakeDisplayRecord& d : r.presenter.Displayed())
            if (d.real && d.frame_id == victim.real_frame_id) real_on_time = d.actual == victim.targets[3].target;
        CHECK(real_on_time);
        CHECK(DisplayMonotonic(r) && st.max_slip == 0);
        CHECK(r.presenter.RevokeGenerated(999999, FgDropReason::Revoked) == FgStatus::InvalidArgument);
    }
}

// Late generated images are dropped (leases held until their fences complete), a stale suffix is cancelled with
// the real image kept, and a stalled executor never blocks the game frames.
void TestLateDropAndReservation() {
    g_case = "late drop and real reservation";
    {
        Rig r(DefaultScheduler(), DefaultFake());
        r.ready_offset[2] = 80000;  // index 2 of 4x at 60 fps finishes 8 ms after arrival: past its deadline
        const int frames = 60;
        for (int i = 0; i < frames; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        CHECK(AllOk(r));
        // at some point inside an interval, index 2 was dropped while its fence was still running: lease held
        const FgTicks a = At(kSecond, frames, 60);
        r.Frame(a, 3);
        const FgIntervalPlan& p = r.plans.back();
        CHECK(p.generated_count == 3);
        r.RunUntil(p.targets[1].deadline + 1000);
        CHECK(r.presenter.LeasesHeld() >= 1);
        r.RunUntil(a + 81000);
        CHECK(r.presenter.LeasesHeld() == 0);
        r.RunUntil(a + 600000);
        const FgPresentStats st = r.presenter.Stats();
        CHECK(st.dropped_late == static_cast<std::uint64_t>(r.Generating()));
        bool shape = true;
        for (const FgFakeDropRecord& d : r.presenter.Dropped()) shape = shape && d.index == 2 && d.reason == FgDropReason::Late;
        CHECK(shape);
        for (const FgIntervalPlan& q : r.plans)
            if (q.generated_count == 3) CHECK(DisplayedGeneratedOf(r, q.interval_id) == 2);
        CHECK(st.displayed_real == static_cast<std::uint64_t>(frames + 1));
        CHECK(st.max_slip == 0 && DisplayMonotonic(r));
    }
    {
        // suffix cancellation: everything queued from the first queued present is stale; the real image survives
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 20; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        const FgTicks a = At(kSecond, 20, 60);
        r.Frame(a, 3);
        const FgIntervalPlan victim = r.plans.back();
        r.RunUntil(a + 9000);
        const std::uint64_t first = r.presenter.FirstQueuedPresentId();
        CHECK(first != 0);
        const std::uint32_t queued = r.presenter.QueuedCount();
        CHECK(r.presenter.CancelPresentsFrom(first) == queued);
        CHECK(r.presenter.QueuedCount() >= 1);  // the real endpoint(s), re-presented
        r.RunUntil(a + 600000);
        CHECK(DisplayedGeneratedOf(r, victim.interval_id) == 0);
        CHECK(DisplayedRealOf(r, victim.real_frame_id) == 1);
        const FgPresentStats st = r.presenter.Stats();
        CHECK(st.cancelled == queued && st.real_represented >= 1);
        CHECK(DisplayMonotonic(r) && st.max_slip == 0);
    }
    {
        // the executor stalls completely: generated leases pile up and their slots stay held, generation is refused
        // for pressure, and every game frame still gets its slot and reaches the display on time
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 10; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        for (std::uint32_t k = 1; k <= kFgMaxTargets; ++k) r.ready_offset[k] = -1;
        for (int i = 10; i < 40; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        CHECK(AllOk(r));  // no NoSlot: the real path never waits for generation
        FgPresentStats st = r.presenter.Stats();
        CHECK(st.dropped_pressure > 0);
        CHECK(r.presenter.LeasesHeld() > 0);
        const std::uint64_t generated_before = st.displayed_generated;
        r.RunUntil(At(kSecond, 40, 60));
        st = r.presenter.Stats();
        CHECK(st.displayed_generated == generated_before);
        CHECK(st.displayed_real >= 39);
        CHECK(st.max_slip == 0);
        // the executor recovers: leases come back and generation resumes
        r.ReleaseStalled(r.clock.Now());
        for (std::uint32_t k = 1; k <= kFgMaxTargets; ++k) r.ready_offset[k] = static_cast<FgTicks>(k) * 2000;
        r.presenter.Pump();
        for (int i = 40; i < 70; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        r.RunUntil(At(kSecond, 70, 60) + 400000);
        CHECK(AllOk(r));
        st = r.presenter.Stats();
        CHECK(r.presenter.LeasesHeld() == 0);
        CHECK(st.displayed_generated > generated_before + 60);
        CHECK(st.displayed_real == 70);
        CHECK(DisplayMonotonic(r) && st.max_slip == 0);
    }
    {
        // a stall starts, then the game presents two frames with no display flip in between (an early frame after a
        // normal one): stalled leases took every slot generation may use, and only the reserved slot lets the second
        // game frame in
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 10; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        for (std::uint32_t k = 1; k <= kFgMaxTargets; ++k) r.ready_offset[k] = -1;
        r.Frame(At(kSecond, 10, 60), 3);
        r.Frame(At(kSecond, 11, 60), 3);
        const std::size_t shown = r.presenter.Displayed().size();
        r.Frame(At(kSecond, 11, 60) + 41667, 3);  // early: nothing flipped since the previous present
        CHECK(r.presenter.Displayed().size() == shown);
        for (int i = 13; i < 30; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        r.RunUntil(At(kSecond, 30, 60) + 400000);
        CHECK(AllOk(r));  // never NoSlot for a game frame
        const FgPresentStats st = r.presenter.Stats();
        CHECK(st.dropped_pressure > 0);
        CHECK(st.displayed_real == 30);
        CHECK(st.max_slip == 0 && DisplayMonotonic(r));
    }
}

// 4x on a 48-360 Hz panel: 60 fps fits; 100 fps leaves the window (suspend at once); 88 fps is inside the margin but
// not the hysteresis band (stays suspended); 75 fps re-enters after the re-entry run; 88 fps then keeps generating;
// 92 fps leaves again. Also the low end (below 48 Hz per image) and a check that without the guard the panel slips.
void TestVrrWindow() {
    g_case = "vrr window";
    {
        Rig r(DefaultScheduler(), DefaultFake());
        struct Phase {
            std::uint32_t fps;
            int frames;
        };
        const Phase phases[6] = {{60, 60}, {100, 40}, {88, 40}, {75, 40}, {88, 40}, {92, 20}};
        FgTicks t0 = kSecond;
        std::size_t begin[7] = {};
        for (int ph = 0; ph < 6; ++ph) {
            begin[ph] = r.plans.size();
            for (int i = 0; i < phases[ph].frames; ++i)
                r.Frame(At(t0, static_cast<std::uint64_t>(i + 1), phases[ph].fps), 3);
            t0 = At(t0, static_cast<std::uint64_t>(phases[ph].frames), phases[ph].fps);
        }
        begin[6] = r.plans.size();
        r.RunUntil(t0 + 400000);
        CHECK(AllOk(r));
        auto generating = [&r](std::size_t from, std::size_t to) {
            std::uint32_t n = 0;
            for (std::size_t i = from; i < to; ++i) n += r.plans[i].generated_count > 0 ? 1u : 0u;
            return n;
        };
        auto switches = [&r](std::size_t from, std::size_t to) {
            std::uint32_t n = 0;
            for (std::size_t i = from + 1; i < to; ++i)
                n += (r.plans[i].generated_count > 0) != (r.plans[i - 1].generated_count > 0) ? 1u : 0u;
            return n;
        };
        CHECK(generating(begin[0], begin[1]) >= 54);
        // 100 fps: suspended once the estimate has the new cadence (two outliers, then the re-seeded window)
        CHECK(generating(begin[1] + 3, begin[2]) == 0);
        std::uint32_t out_of_window = 0;
        for (std::size_t i = begin[1]; i < begin[2]; ++i)
            out_of_window += r.plans[i].real_only_reason == FgRealOnlyReason::OutOfVrrWindow ? 1u : 0u;
        CHECK(out_of_window >= 35);
        // 88 fps after a suspension: inside the margin, outside the hysteresis band: stays suspended, no flapping
        CHECK(generating(begin[2], begin[3]) == 0);
        // 75 fps: re-entry after the re-entry run (8 intervals), then steady
        const std::uint32_t g75 = generating(begin[3], begin[4]);
        CHECK(g75 >= 25 && g75 <= 40 - 8);
        CHECK(switches(begin[3], begin[4]) == 1);
        // 88 fps while generating: inside the margin, keeps generating (the band works both ways)
        CHECK(generating(begin[4] + 6, begin[5]) == static_cast<std::uint32_t>(begin[5] - begin[4] - 6));
        // 92 fps: outside the raw window, suspended as soon as the median crosses it
        CHECK(generating(begin[5] + 6, begin[6]) == 0);
        const FgPresentStats st = r.presenter.Stats();
        CHECK(st.max_slip == 0);
        CHECK(DisplayMonotonic(r) && DisplayMatchesPlans(r));
        bool none_in_suspended = true;
        for (const FgIntervalPlan& p : r.plans)
            if (p.generated_count == 0) none_in_suspended = none_in_suspended && DisplayedGeneratedOf(r, p.interval_id) == 0;
        CHECK(none_in_suspended);
    }
    {
        // low end: 2x at 20 fps holds each image 25 ms (below 48 Hz): suspended; at 30 fps (60 Hz images) it runs
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 30; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 20), 1);
        CHECK(AllOk(r));
        CHECK(r.Generating() == 0 && r.CountReason(FgRealOnlyReason::OutOfVrrWindow) >= 25);
        Rig q(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 30; ++i) q.Frame(At(kSecond, static_cast<std::uint64_t>(i), 30), 1);
        CHECK(AllOk(q));
        CHECK(q.Generating() >= 24);
    }
    {
        // adaptive count: 6x requested at 75 fps runs as 4x (the largest that fits) and reports both counts
        FgSchedulerConfig c = DefaultScheduler();
        c.adaptive_count = true;
        Rig r(c, DefaultFake());
        for (int i = 0; i < 40; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 75), 5);
        CHECK(AllOk(r));
        bool four = true;
        for (const FgIntervalPlan& p : r.plans)
            if (p.generated_count > 0) four = four && p.generated_count == 3 && p.requested_count == 5;
        CHECK(four && r.Generating() >= 34);
        CHECK(r.presenter.Stats().max_slip == 0);
    }
    {
        // negative control: the same 4x at 100 fps with the window guard disabled stutters on the 360 Hz panel
        FgSchedulerConfig c = DefaultScheduler();
        c.display.min_interval = 1;
        c.display.margin = 0;
        c.display.hysteresis = 0;
        Rig r(c, DefaultFake());
        for (int i = 0; i < 40; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 100), 3);
        r.RunUntil(At(kSecond, 40, 100) + 400000);
        CHECK(r.Generating() > 0);
        CHECK(r.presenter.Stats().max_slip > 0);
    }
}

// Resize: quiesce cancels unshown presents, completion waits for old work (Busy, never a fake success), the new
// epoch primes real-only, old-epoch plans are refused; a failed resize is passed through; device loss tears down.
void TestResizeAndDeviceLoss() {
    g_case = "resize, epoch and device loss";
    {
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 20; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        r.ready_offset[3] = -1;  // index 3 of the next interval is still running when the resize starts
        const FgTicks a = At(kSecond, 20, 60);
        r.Frame(a, 3);
        const FgIntervalPlan stale = r.plans.back();
        r.RunUntil(a + 5000);
        CHECK(r.presenter.QueuedCount() > 0);
        const std::size_t shown_before = r.presenter.Displayed().size();
        CHECK(r.presenter.Quiesce() == FgStatus::Ok);
        CHECK(r.presenter.State() == FgPresenterState::Quiescing);
        CHECK(r.presenter.QueuedCount() == 0);
        // admission is stopped
        FgIntervalReservation res;
        CHECK(r.presenter.BeginInterval(stale, &res) == FgStatus::WrongState);
        CHECK(r.presenter.Detach() == FgStatus::Busy);  // old work in flight on a healthy device
        CHECK(!r.presenter.Drained());
        CHECK(r.presenter.CompleteResize(true, 2) == FgStatus::Busy);
        CHECK(r.presenter.State() == FgPresenterState::Quiescing);
        r.RunUntil(a + 300000);
        CHECK(r.presenter.Displayed().size() == shown_before);  // nothing unshown reached the display
        r.ReleaseStalled(r.clock.Now());
        r.presenter.Pump();
        CHECK(r.presenter.Drained());
        CHECK(r.presenter.CompleteResize(true, 1) == FgStatus::InvalidArgument);  // the epoch must advance
        CHECK(r.presenter.CompleteResize(true, 2) == FgStatus::Ok);
        CHECK(r.presenter.State() == FgPresenterState::Attached);
        CHECK(r.scheduler.BeginEpoch(2) == FgStatus::Ok);
        CHECK(r.scheduler.BeginEpoch(2) == FgStatus::InvalidArgument);
        // old-epoch work is refused on both sides
        CHECK(r.presenter.BeginInterval(stale, &res) == FgStatus::StaleEpoch);
        FgGameFrame old;
        old.epoch = 1;
        old.frame_id = 1000;
        old.arrival = r.clock.Now();
        old.requested_count = 3;
        FgIntervalPlan plan;
        CHECK(r.scheduler.Commit(old, &plan) == FgStatus::StaleEpoch);
        // the new epoch primes real-only, then generates
        r.epoch = 2;
        r.ready_offset[3] = 6000;
        const FgTicks t0 = r.clock.Now() + 100000;
        const std::size_t first_new = r.plans.size();
        for (int i = 0; i < 30; ++i) r.Frame(At(t0, static_cast<std::uint64_t>(i), 60), 3);
        r.RunUntil(At(t0, 30, 60) + 400000);
        bool new_ok = true;
        for (std::size_t i = first_new; i < r.statuses.size(); ++i) new_ok = new_ok && r.statuses[i] == FgStatus::Ok;
        CHECK(new_ok);
        CHECK(r.plans[first_new].generated_count == 0 &&
              r.plans[first_new].real_only_reason == FgRealOnlyReason::NotPrimed);
        std::uint32_t gen_new = 0;
        for (std::size_t i = first_new; i < r.plans.size(); ++i) gen_new += r.plans[i].generated_count > 0 ? 1u : 0u;
        CHECK(gen_new >= 24);
        bool epochs = true;
        const std::vector<FgFakeDisplayRecord>& d = r.presenter.Displayed();
        for (std::size_t i = shown_before; i < d.size(); ++i) epochs = epochs && d[i].epoch == 2;
        CHECK(epochs);
        CHECK(DisplayedGeneratedOf(r, stale.interval_id) == 0);
        CHECK(r.presenter.Stats().epochs == 2);
        CHECK(r.presenter.Detach() == FgStatus::Ok);  // everything shown: detach drains at once
        CHECK(r.registry.Owner(kChain) == nullptr);
    }
    {
        // a failed resize is passed through: state Failed, every present refused, no success anywhere
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 10; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        r.RunUntil(At(kSecond, 10, 60));
        CHECK(r.presenter.Quiesce() == FgStatus::Ok);
        r.RunUntil(At(kSecond, 10, 60) + 100000);
        CHECK(r.presenter.CompleteResize(false, 2) == FgStatus::ResizeFailed);
        CHECK(r.presenter.State() == FgPresenterState::Failed);
        const std::size_t shown = r.presenter.Displayed().size();
        const FgPresentStats before = r.presenter.Stats();
        FgIntervalPlan plan = r.plans.back();
        FgRealImage img;
        img.epoch = plan.epoch;
        img.frame_id = plan.real_frame_id;
        img.ready = FgReadyFence{r.real_fence, r.real_value};
        CHECK(r.presenter.PresentRealOnly(plan, img) == FgStatus::Refused);
        FgIntervalReservation res;
        CHECK(r.presenter.BeginInterval(plan, &res) == FgStatus::Refused);
        CHECK(r.presenter.CompleteResize(true, 3) == FgStatus::WrongState);
        r.RunUntil(r.clock.Now() + 200000);
        CHECK(r.presenter.Displayed().size() == shown);
        CHECK(r.presenter.Stats().game_presents == before.game_presents);
        CHECK(r.presenter.Detach() == FgStatus::Ok);
        CHECK(r.registry.Owner(kChain) == nullptr);
    }
    {
        // device loss: immediate teardown, stale fences ignored, refusals until detach, the chain freed
        Rig r(DefaultScheduler(), DefaultFake());
        for (int i = 0; i < 15; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        const FgTicks a = At(kSecond, 15, 60);
        r.ready_offset[1] = -1;
        r.Frame(a, 3);
        r.RunUntil(a + 3000);
        const std::size_t shown = r.presenter.Displayed().size();
        CHECK(r.presenter.OnDeviceLost() == FgStatus::Ok);
        r.scheduler.OnDeviceLost();
        CHECK(r.presenter.State() == FgPresenterState::Lost);
        r.ReleaseStalled(r.clock.Now());  // a stale completion after the loss
        r.RunUntil(a + 500000);
        CHECK(r.presenter.Displayed().size() == shown);
        CHECK(r.Frame(a + 600000, 3) == FgStatus::DeviceLost);  // the scheduler refuses
        FgIntervalPlan plan = r.plans.back();
        FgIntervalReservation res;
        CHECK(r.presenter.BeginInterval(plan, &res) == FgStatus::DeviceLost);
        CHECK(r.presenter.Quiesce() == FgStatus::DeviceLost);
        CHECK(r.presenter.CompleteResize(true, 2) == FgStatus::DeviceLost);
        CHECK(r.presenter.Detach() == FgStatus::Ok);
        CHECK(r.presenter.State() == FgPresenterState::Detached);
        CHECK(r.registry.Owner(kChain) == nullptr);
        // a new presenter can own the chain on the recreated device
        FgFakePresenter next(DefaultFake(), &r.clock, &r.gpu, &r.registry);
        CHECK(next.Attach(kChain, 2) == FgStatus::Ok);
        CHECK(r.scheduler.BeginEpoch(2) == FgStatus::Ok);
    }
}

// A refused back end or a second owner refuses with a reason; nothing is presented through another route.
void TestRefusal() {
    g_case = "refusal without fallback";
    {
        FgFakeConfig pc = DefaultFake();
        pc.probe.supported = false;
        pc.probe.max_generated = 0;
        pc.probe.refusal = FgStatus::Unsupported;
        Rig r(DefaultScheduler(), pc);
        CHECK(r.attach_status == FgStatus::Unsupported);
        CHECK(r.presenter.State() == FgPresenterState::Detached);
        CHECK(r.registry.Owner(kChain) == nullptr);
        for (int i = 0; i < 10; ++i) CHECK(r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3) == FgStatus::WrongState);
        r.RunUntil(At(kSecond, 10, 60) + 400000);
        CHECK(r.presenter.Displayed().empty());
        const FgPresentStats st = r.presenter.Stats();
        CHECK(st.game_presents == 0 && st.real_queued == 0 && st.generated_queued == 0);
        CHECK(st.refused_calls == 11);
    }
    {
        Rig r(DefaultScheduler(), DefaultFake());
        CHECK(r.attach_status == FgStatus::Ok);
        FgFakePresenter second(DefaultFake(), &r.clock, &r.gpu, &r.registry);
        CHECK(second.Attach(kChain, 1) == FgStatus::Refused);  // one visible owner per chain
        CHECK(second.Attach(kChain + 1, 1) == FgStatus::Ok);   // another chain is fine
        CHECK(r.presenter.Attach(kChain, 1) == FgStatus::WrongState);
        for (int i = 0; i < 10; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        CHECK(AllOk(r));
        // a generating interval without its real image is refused, and so is a real-only plan through BeginInterval
        const FgTicks a = At(kSecond, 10, 60);
        r.RunUntil(a);
        FgGameFrame f;
        f.epoch = 1;
        f.frame_id = ++r.frame_id;
        f.arrival = a;
        f.requested_count = 3;
        FgIntervalPlan plan;
        CHECK(r.scheduler.Commit(f, &plan) == FgStatus::Ok && plan.generated_count == 3);
        FgIntervalReservation res;
        CHECK(r.presenter.BeginInterval(plan, &res) == FgStatus::Ok);
        FgIntervalSubmission sub;
        sub.epoch = res.epoch;
        sub.interval_id = res.interval_id;
        sub.count = res.count;
        for (std::uint32_t i = 0; i + 1 < res.count; ++i) {
            sub.produced[i] = res.slots[i].reserved;
            sub.ready[i] = FgReadyFence{r.gen_fence[i + 1], ++r.gen_value[i + 1]};
        }
        CHECK(r.presenter.SubmitInterval(sub) == FgStatus::InvalidArgument);  // the real image is missing
        sub.produced[res.count - 1] = true;
        sub.ready[res.count - 1] = FgReadyFence{999, 1};
        CHECK(r.presenter.SubmitInterval(sub) == FgStatus::InvalidArgument);  // an unknown fence
        // the reservation is still open: a second interval or a real-only present is refused meanwhile
        FgIntervalPlan real_only = plan;
        real_only.generated_count = 0;
        CHECK(r.presenter.PresentRealOnly(real_only, FgRealImage{1, plan.real_frame_id, {r.real_fence, 1}}) ==
              FgStatus::WrongState);
        ++r.real_value;
        r.gpu.SignalAt(r.real_fence, r.real_value, a);
        sub.ready[res.count - 1] = FgReadyFence{r.real_fence, r.real_value};
        for (std::uint32_t i = 0; i + 1 < res.count; ++i) r.gpu.SignalAt(r.gen_fence[i + 1], r.gen_value[i + 1], a + 2000);
        CHECK(r.presenter.SubmitInterval(sub) == FgStatus::Ok);
        CHECK(r.presenter.SubmitInterval(sub) == FgStatus::WrongState);  // submitted once
        r.RunUntil(a + 600000);
        CHECK(DisplayedGeneratedOf(r, plan.interval_id) == 3 && DisplayedRealOf(r, plan.real_frame_id) == 1);
    }
}

// Mutants: plans a broken scheduler could produce must fail validation and be refused by the presenter without any
// state change; the independent spacing oracle must catch a cumulative-rounding schedule; the scheduler's commit is
// transactional.
void TestMutations() {
    g_case = "mutations";
    Rig r(DefaultScheduler(), DefaultFake());
    for (int i = 0; i < 12; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 50), 5);
    CHECK(AllOk(r));
    const FgTicks a = At(kSecond, 12, 50);
    r.RunUntil(a);
    FgGameFrame f;
    f.epoch = 1;
    f.frame_id = ++r.frame_id;
    f.arrival = a;
    f.requested_count = 5;
    FgIntervalPlan good;
    CHECK(r.scheduler.Commit(f, &good) == FgStatus::Ok);
    CHECK(good.generated_count == 5 && FgValidatePlan(good) == FgStatus::Ok && ExactSpacingOk(good));
    const FgTicks d = 6;

    std::vector<FgIntervalPlan> mutants;
    FgIntervalPlan m = good;
    std::swap(m.targets[0].target, m.targets[1].target);  // order violated
    mutants.push_back(m);
    m = good;  // cumulative rounding: k * round(T / 6)
    for (std::uint32_t k = 1; k <= 5; ++k) m.targets[k - 1].target = m.t_base + k * ((m.period + d / 2) / d);
    for (std::uint32_t k = 1; k <= 5; ++k) m.targets[k - 1].deadline = m.targets[k - 1].target - m.margin;
    const bool cumulative_differs = m.targets[4].target != good.targets[4].target ||
                                    m.targets[3].target != good.targets[3].target;
    CHECK(cumulative_differs);       // this synthetic period separates the two roundings
    CHECK(!ExactSpacingOk(m));       // the test's own oracle catches it
    mutants.push_back(m);
    m = good;  // truncation instead of rounding
    for (std::uint32_t k = 1; k <= 5; ++k) {
        m.targets[k - 1].target = m.t_base + (static_cast<FgTicks>(k) * m.period) / d;
        m.targets[k - 1].deadline = m.targets[k - 1].target - m.margin;
    }
    mutants.push_back(m);
    m = good;
    m.targets[2].target += 1;  // one tick off
    m.targets[2].deadline += 1;
    mutants.push_back(m);
    m = good;
    m.targets[5].target += 1;  // real endpoint not at t_base + T
    mutants.push_back(m);
    m = good;
    m.targets[1].deadline = m.targets[1].target;  // margin ignored
    mutants.push_back(m);
    m = good;
    m.targets[0].t = 1.0f / 5.0f;  // k / N instead of k / (N + 1)
    mutants.push_back(m);
    m = good;
    m.target_count = 5;  // count mismatch
    mutants.push_back(m);
    m = good;
    m.generated_count = 6;
    m.requested_count = 6;
    m.target_count = 7;
    mutants.push_back(m);
    m = good;
    m.real_only_reason = FgRealOnlyReason::Reset;  // a generating plan with a real-only reason
    mutants.push_back(m);
    m = good;
    m.targets[5].admitted = false;  // the real endpoint dropped
    mutants.push_back(m);
    m = good;
    m.targets[5].real = false;
    mutants.push_back(m);

    const FgPresentStats before = r.presenter.Stats();
    const std::uint32_t free_before = r.presenter.FreeSlots();
    int caught = 0;
    for (const FgIntervalPlan& mp : mutants) {
        FgIntervalReservation res;
        const bool invalid = FgValidatePlan(mp) != FgStatus::Ok;
        const bool refused = r.presenter.BeginInterval(mp, &res) == FgStatus::InvalidArgument;
        caught += invalid && refused ? 1 : 0;
    }
    CHECK(caught == static_cast<int>(mutants.size()));
    CHECK(r.presenter.Stats().game_presents == before.game_presents);
    CHECK(r.presenter.FreeSlots() == free_before);

    // a valid plan that does not continue the timeline (t_base before the last committed real target) is refused
    FgIntervalPlan back = r.plans[r.plans.size() - 2];
    back.interval_id = good.interval_id;
    CHECK(FgValidatePlan(back) == FgStatus::Ok);
    FgIntervalReservation res;
    CHECK(r.presenter.BeginInterval(back, &res) == FgStatus::InvalidArgument);
    // ... and so is one that jumps ahead without saying it re-anchored (a gap in the timeline)
    FgIntervalPlan jump = good;
    jump.t_base += 1000;
    for (std::uint32_t i = 0; i < jump.target_count; ++i) {
        jump.targets[i].target += 1000;
        jump.targets[i].deadline += 1000;
    }
    CHECK(FgValidatePlan(jump) == FgStatus::Ok);
    CHECK(r.presenter.BeginInterval(jump, &res) == FgStatus::InvalidArgument);
    CHECK(r.presenter.Stats().game_presents == before.game_presents);
    // the genuine plan is still accepted afterwards
    CHECK(r.Present(good, a) == FgStatus::Ok);
    r.plans.push_back(good);

    // transactional commit: a refused frame leaves the timeline as it was
    FgGameFrame dup = f;  // same frame id
    dup.arrival = a + 200000;
    FgIntervalPlan none;
    CHECK(r.scheduler.Commit(dup, &none) == FgStatus::InvalidArgument);
    FgGameFrame bad = f;
    bad.frame_id = ++r.frame_id;
    bad.arrival = a + 200000;
    bad.requested_count = 6;
    CHECK(r.scheduler.Commit(bad, &none) == FgStatus::InvalidArgument);
    CHECK(r.scheduler.LastRealTarget() == good.targets[5].target);
    r.RunUntil(a + 200000);
    bad.requested_count = 5;
    FgIntervalPlan after;
    CHECK(r.scheduler.Commit(bad, &after) == FgStatus::Ok);
    CHECK(after.t_base == good.targets[5].target && !after.reanchored && after.generated_count == 5);
    CHECK(r.Present(after, a + 200000) == FgStatus::Ok);
}

// RC2 (N-B2 / O-A4 / O-A7 / M1): 2-3 ms steady jitter at 6x/60 fps produces 0 reanchors; LatencyCap steps N down
// instead of dropping straight to 0; SetMaxGeneratedCap clamps & restores cleanly; 360 Hz VRR ceiling (~349 Hz)
// prevents above-refresh targets while 80 fps 4x runs cleanly.
void TestFrameStartWaitableCadence() {
    g_case = "frame-start waitable cadence";
    constexpr FgTicks kMs = 10000;
    for (const FgTicks render : {2 * kMs, 4 * kMs, 8 * kMs, 16 * kMs}) {
        for (const FgTicks budget : {2 * kMs, 4 * kMs, 8 * kMs}) {
            for (std::uint32_t requested = 0; requested <= 5; ++requested) {
                for (std::size_t latency = 1; latency <= 3; ++latency) {
                    FgSchedulerConfig config = DefaultScheduler();
                    config.first_image_budget = budget;
                    config.bridge_and_submit_margin = 2500;
                    config.max_lead = 280000;
                    config.display.min_interval = 28619;
                    config.display.margin = 0;
                    config.adaptive_count = true;
                    FgTargetScheduler scheduler(config);
                    CHECK(scheduler.BeginEpoch(1) == FgStatus::Ok);
                    std::deque<FgTicks> pending;
                    FgTicks handoff = kSecond;
                    FgTicks warm_handoff = 0;
                    for (std::uint64_t frame = 1; frame <= 160; ++frame) {
                        const FgTicks arrival = handoff + render;
                        handoff = arrival;
                        if (pending.size() == latency && pending.front() > handoff) handoff = pending.front();
                        while (!pending.empty() && pending.front() <= handoff) pending.pop_front();
                        FgGameFrame game{1, frame, arrival, requested, true, false, handoff};
                        FgIntervalPlan plan;
                        CHECK(scheduler.Commit(game, &plan) == FgStatus::Ok);
                        pending.push_back(plan.targets[0].target);
                        CHECK(pending.size() <= latency);
                        if (requested == 0 && frame > 2 &&
                            render >= config.display.min_interval)
                            CHECK(plan.targets[0].target - handoff == config.real_min_lead);
                        if (frame == 40) warm_handoff = handoff;
                    }
                    const FgTicks expected = (std::max)({render, budget, config.display.min_interval});
                    if ((handoff - warm_handoff) * 100 > expected * 120 * 105)
                        std::printf("  R=%lld B=%lld N=%u L=%zu P=%lld bound=%lld\n",
                                    static_cast<long long>(render), static_cast<long long>(budget), requested, latency,
                                    static_cast<long long>((handoff - warm_handoff) / 120),
                                    static_cast<long long>(expected));
                    CHECK((handoff - warm_handoff) * 100 <= expected * 120 * 105);
                }
            }
        }
    }

    // R3B-5: degraded case where the interval's first generated image is late-dropped. Admission must end when the
    // first image is submitted OR dropped (trace kind 1, 3, 4, or 7), not hold an L=1 caller until the real image.
    {
        struct ScriptedFence final : IFgFenceSource {
            std::uint64_t done = 0;
            bool FenceValid(std::uint32_t f) const override { return f == 1; }
            std::uint64_t FenceCompleted(std::uint32_t f) const override { return f == 1 ? done : 0; }
        } fence;
        struct AdmissionTrace {
            std::deque<std::uint64_t> pending;
            std::uint64_t drop_releases = 0;
            std::uint64_t submit_releases = 0;
            static void OnTrace(void* ctx, std::uint64_t interval_id, std::uint64_t frame_id, std::uint32_t, bool,
                                std::uint8_t kind, FgTicks, FgTicks) {
                auto* self = static_cast<AdmissionTrace*>(ctx);
                if (!self || interval_id == 0) return;
                if (kind == 1 || kind == 3 || kind == 4 || kind == 7) {
                    for (auto it = self->pending.begin(); it != self->pending.end(); ++it) {
                        if (*it == frame_id) {
                            self->pending.erase(it);
                            if (kind == 3 || kind == 4) ++self->drop_releases;
                            else ++self->submit_releases;
                            break;
                        }
                    }
                }
            }
        } trace;

        FgEngineConfig ecfg;
        ecfg.probe.backend = FgPresenterBackend::DxgiFlip;
        ecfg.probe.supported = true;
        ecfg.probe.per_image_target_time = true;
        ecfg.probe.max_generated = 5;
        ecfg.slot_count = 20;
        ecfg.max_pending_intervals = 3;
        ecfg.pending_late_tolerance = 20000;  // 2.0 ms
        FgPresentEngine engine(ecfg, &fence);
        engine.SetTraceCallback(&AdmissionTrace::OnTrace, &trace);
        CHECK(engine.Attach(1) == FgStatus::Ok);

        FgSchedulerConfig scfg = DefaultScheduler();
        scfg.first_image_budget = 4 * kMs;
        scfg.bridge_and_submit_margin = 2500;
        scfg.max_lead = 280000;
        scfg.display.min_interval = 28619;
        scfg.display.margin = 0;
        scfg.adaptive_count = true;
        FgTargetScheduler scheduler(scfg);
        CHECK(scheduler.BeginEpoch(1) == FgStatus::Ok);

        const FgTicks render = 16 * kMs;  // 62.5 fps base, 4x (N=3 -> 4 ms spacing)
        FgTicks handoff = kSecond;
        FgTicks prev_release = 0;
        FgTicks warm_handoff = 0;
        std::uint64_t fence_val = 0;
        for (std::uint64_t frame = 1; frame <= 50; ++frame) {
            const FgTicks arrival = handoff + render;
            handoff = prev_release > arrival ? prev_release : arrival;
            FgGameFrame game{1, frame, arrival, 3u, true, false, handoff};
            FgIntervalPlan plan;
            CHECK(scheduler.Commit(game, &plan) == FgStatus::Ok);
            trace.pending.push_back(frame);
            if (plan.generated_count == 0) {
                const std::uint64_t rv = ++fence_val;
                fence.done = rv;
                FgRealImage ri{1, frame, {1, rv}};
                CHECK(engine.PresentRealOnly(plan, ri) == FgStatus::Ok);
                engine.Pump(plan.targets[0].target);
                std::vector<FgEnginePresent> out;
                engine.TakeOutbox(&out);
                for (const auto& p : out) {
                    CHECK(engine.LeasePresent(p.slot, p.present_id));
                    engine.FinishLeasedPresent(p.slot, p.target, p.target, true);
                    CHECK(engine.OnSlotReleased(p.slot, true));
                }
                prev_release = plan.targets[0].target;
            } else {
                FgIntervalReservation res;
                CHECK(engine.BeginInterval(plan, handoff, &res) == FgStatus::Ok);
                FgIntervalSubmission sub;
                sub.epoch = 1;
                sub.interval_id = plan.interval_id;
                sub.count = res.count;
                const std::uint64_t gen1_val = ++fence_val;
                const std::uint64_t rest_val = ++fence_val;
                for (std::uint32_t i = 0; i < res.count; ++i) {
                    sub.produced[i] = res.slots[i].reserved;
                    sub.ready[i] = {1, i == 0 ? gen1_val : rest_val};
                }
                CHECK(engine.SubmitInterval(sub, handoff) == FgStatus::Ok);
                // First generated image misses PendingDeadline while real endpoint is far later (~target[3]):
                // Pump at target[0] + pending_late_tolerance + 1 must drop index 1 (kind 3) and release admission
                // long before the real endpoint target[3]!
                const FgTicks drop_at = plan.targets[0].target + ecfg.pending_late_tolerance + 1;
                CHECK(drop_at < plan.targets[plan.generated_count].target);
                engine.Pump(drop_at);
                CHECK(trace.pending.empty());  // released on late drop without waiting for real image!
                prev_release = drop_at;
                // Complete remaining images of the interval so slots recycle cleanly.
                fence.done = rest_val;
                engine.Pump(plan.targets[plan.generated_count].target);
                std::vector<FgEnginePresent> out;
                engine.TakeOutbox(&out);
                for (const auto& p : out) {
                    if (!p.real && frame == 20 && p.index == 2) {
                        // Also prove post-queue drop (kind 4) releases admission cleanly.
                        trace.pending.push_back(frame);
                        CHECK(engine.OnPresentDropped(p.present_id, FgDropReason::Late));
                        CHECK(trace.pending.empty());
                    } else {
                        CHECK(engine.LeasePresent(p.slot, p.present_id));
                        engine.FinishLeasedPresent(p.slot, p.target, p.target, true);
                        CHECK(engine.OnSlotReleased(p.slot, true));
                    }
                }
            }
            CHECK(trace.pending.empty());
            if (frame == 10) warm_handoff = handoff;
        }
        CHECK(trace.drop_releases >= 40);
        // Degraded L=1 cadence stays at max(render, budget + late_tolerance) (~16 ms), NOT stretched to real target!
        CHECK((handoff - warm_handoff) <= render * 40 * 105 / 100);
    }
}

void TestTearingPlanHeadNearCeiling() {
    g_case = "tearing plan head near VRR ceiling";
    constexpr FgTicks kMinInterval = 28619;   // ~349.4 Hz ceiling on 360 Hz VRR display
    constexpr FgTicks kBaseLateTol = 20000;   // 2.0 ms DXGI tearing tolerance
    // Test both 320 Hz (80 fps @ 4x) and 345 Hz (86.25 fps @ 4x) with 0.3-1.0 ms issue jitter.
    const FgTicks slots[2] = {31250, 28986};  // 320 Hz, 345 Hz
    const FgTicks jitters[6] = {3000, 5000, 7500, 10000, 4000, 9000};  // 0.3 ms .. 1.0 ms
    for (const FgTicks slot : slots) {
        FgTicks last_issue = 0;
        FgTicks t0 = kSecond;
        std::uint32_t dropped = 0;
        for (int i = 0; i < 60; ++i) {
            const FgTicks target = t0 + static_cast<FgTicks>(i) * slot;
            const FgTicks next_target = target + slot;
            const bool real = (i % 4) == 3;
            const FgTicks now = target + jitters[i % 6];
            const FgTearingPlanDecision d =
                FgPlanTearingImage(target, real, now, last_issue, kMinInterval, kBaseLateTol, true, next_target);
            if (d.drop_late) {
                ++dropped;
            } else {
                const FgTicks issued = now > d.issue_at ? now : d.issue_at;
                if (last_issue != 0) CHECK(issued - last_issue >= kMinInterval);
                last_issue = issued;
            }
        }
        // Ready generated images with 0.3-1.0 ms jitter near the VRR ceiling must NOT be dropped.
        CHECK(dropped == 0);
    }
    // Genuinely late generated image (> 1.5 ms late at 345 Hz) IS dropped; real image is NEVER dropped.
    {
        const FgTicks target = kSecond;
        const FgTicks next_target = target + 28986;
        const FgTearingPlanDecision gen_late =
            FgPlanTearingImage(target, false, target + 15000, target - kMinInterval, kMinInterval, kBaseLateTol, true,
                               next_target);
        CHECK(gen_late.drop_late);
        const FgTearingPlanDecision real_late =
            FgPlanTearingImage(target, true, target + 50000, target - kMinInterval, kMinInterval, kBaseLateTol, true,
                               next_target);
        CHECK(!real_late.drop_late && real_late.issue_at >= target);
    }
}

void TestRc2SchedulerPacing() {
    g_case = "rc2 scheduler pacing";
    {
        // N-B2: +-2.5 ms jitter at 6x / 60 fps on 360 Hz panel -> 0 reanchors after warm-up, 0 late drops
        FgSchedulerConfig sc = DefaultScheduler();
        sc.first_image_budget = 40000;        // 4.0 ms
        sc.bridge_and_submit_margin = 2500;   // 0.25 ms (DXGI tearing)
        sc.max_lead = 280000;                 // 28 ms (O-A7)
        sc.display.min_interval = 27778;      // 360 Hz
        sc.display.margin = 0;                // 6x at 60 fps = 27778 ticks exactly
        sc.adaptive_count = true;
        Rig r(sc, DefaultFake());
        const FgTicks jitter[8] = {25000, -25000, 20000, -20000, 15000, -15000, 22000, -22000};
        FgTicks t = kSecond;
        for (int i = 0; i < 120; ++i) {
            t += 166667 + (i >= 10 ? jitter[i % 8] : 0);
            r.Frame(t, 5);
        }
        r.RunUntil(t + 400000);
        CHECK(AllOk(r));
        std::uint32_t reanchors_steady = 0;
        for (std::size_t i = 10; i < r.plans.size(); ++i)
            if (r.plans[i].reanchored) ++reanchors_steady;
        CHECK(reanchors_steady == 0);
        CHECK(r.Generating() >= 110);
        CHECK(r.presenter.Stats().dropped_late == 0);
    }
    {
        // N-B2: LatencyCap step-down (N=5 at 60 fps needs 178.9 ms/10 -> 17.89 ms; max_lead=14 ms steps N down to 1,
        // NOT straight to 0!)
        FgSchedulerConfig sc = DefaultScheduler();
        sc.first_image_budget = 40000;  // 4.0 ms
        sc.max_lead = 140000;           // 14.0 ms: N=5 needs 17.89 ms, N=2 needs 15.11 ms, N=1 needs 12.33 ms
        sc.adaptive_count = true;
        Rig r(sc, DefaultFake());
        for (int i = 0; i < 30; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 5);
        r.RunUntil(At(kSecond, 30, 60) + 400000);
        CHECK(AllOk(r));
        CHECK(r.Generating() >= 24);
        CHECK(r.plans.back().generated_count == 1 && r.plans.back().requested_count == 5);
        CHECK(r.scheduler.LastReductionReason() == FgRealOnlyReason::LatencyCap);
        CHECK(r.CountReason(FgRealOnlyReason::LatencyCap) == 0);  // never dropped to real-only!
    }
    {
        // O-A7: SetMaxGeneratedCap steps down and restores cleanly
        FgSchedulerConfig sc = DefaultScheduler();
        sc.first_image_budget = 40000;
        sc.max_lead = 280000;
        sc.adaptive_count = true;
        Rig r(sc, DefaultFake());
        for (int i = 0; i < 15; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        CHECK(r.plans.back().generated_count == 3);
        r.scheduler.SetMaxGeneratedCap(2);
        for (int i = 15; i < 25; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        CHECK(r.plans.back().generated_count == 2);
        CHECK(r.scheduler.LastReductionReason() == FgRealOnlyReason::LatencyCap);
        r.scheduler.SetMaxGeneratedCap(0);
        for (int i = 25; i < 35; ++i) r.Frame(At(kSecond, static_cast<std::uint64_t>(i), 60), 3);
        CHECK(r.plans.back().generated_count == 3);
    }
    {
        // O-A4 / M1: 360 Hz panel with +3% VRR ceiling (~349 Hz): 90 fps 4x (360 Hz) steps down to 3x (270 Hz);
        // 80 fps 4x (320 Hz) runs full 4x with 0 slip.
        FgSchedulerConfig sc = DefaultScheduler();
        sc.display.min_interval = 27778 + 27778 / 33;  // ~28620 ticks (~349 Hz ceiling)
        sc.display.margin = 500;
        sc.display.max_interval = 208333;              // 48 Hz floor
        sc.adaptive_count = true;
        Rig r90(sc, DefaultFake());
        for (int i = 0; i < 30; ++i) r90.Frame(At(kSecond, static_cast<std::uint64_t>(i), 90), 3);
        r90.RunUntil(At(kSecond, 30, 90) + 400000);
        CHECK(AllOk(r90));
        CHECK(r90.plans.back().generated_count == 2);  // stepped down from 4x to 3x
        CHECK(r90.scheduler.LastReductionReason() == FgRealOnlyReason::OutOfVrrWindow);
        CHECK(r90.presenter.Stats().max_slip == 0);

        Rig r80(sc, DefaultFake());
        for (int i = 0; i < 30; ++i) r80.Frame(At(kSecond, static_cast<std::uint64_t>(i), 80), 3);
        r80.RunUntil(At(kSecond, 30, 80) + 400000);
        CHECK(AllOk(r80));
        CHECK(r80.plans.back().generated_count == 3);  // full 4x (320 Hz <= 349 Hz)
        CHECK(r80.presenter.Stats().max_slip == 0);
    }
}

}  // namespace

int main() {
    TestIntermediateTime();
    TestCheckedArithmetic();
    TestEstimator();
    TestSpacingMatrix();
    TestDynamicCount();
    TestMenuAndReset();
    TestLateDropAndReservation();
    TestVrrWindow();
    TestResizeAndDeviceLoss();
    TestRefusal();
    TestMutations();
    TestRc2SchedulerPacing();
    TestFrameStartWaitableCadence();
    TestTearingPlanHeadNearCeiling();
    std::printf("test_fg_presenter: %d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
