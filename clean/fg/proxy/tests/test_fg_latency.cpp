// SPDX-License-Identifier: GPL-3.0-only
#include <cstdio>

#include "../fg_latency_pacer.h"

using namespace wgfg;

namespace {

int failed = 0;

void Check(bool pass, const char* description, int line) {
    if (pass) return;
    std::printf("line %d: %s\n", line, description);
    ++failed;
}

#define CHECK(x) Check((x), #x, __LINE__)

struct FakeClock {
    FgTicks now = 1000000;
    unsigned pre_input_waits = 0, present_waits = 0;

    void SleepUntil(FgTicks wake) {
        if (wake > now) {
            ++pre_input_waits;
            now = wake;
        }
    }

    void Present(bool paced) {
        if (!paced) ++present_waits;
    }
};

FgIntervalPlan Plan(uint32_t multiplier, FgTicks real) {
    FgIntervalPlan plan;
    plan.epoch = 1;
    plan.interval_id = 2;
    plan.real_frame_id = 20;
    plan.period = 180000;
    plan.margin = 10000;
    plan.generated_count = multiplier - 1;
    plan.target_count = multiplier;
    plan.targets[0].target = real - plan.period + plan.period / multiplier;
    plan.targets[plan.generated_count].target = real;
    return plan;
}

void Prime(FgLatencyPacer& pacer, const FgIntervalPlan& plan, FgTicks cpu, FgTicks gpu) {
    constexpr FgTicks simulation = 780000;
    pacer.Marker(100, FgLatencyMarker::SimulationStart, simulation);
    pacer.Marker(101, FgLatencyMarker::SimulationStart, simulation + 100);
    pacer.Marker(100, FgLatencyMarker::RenderSubmitEnd, simulation + cpu);
    pacer.Marker(100, FgLatencyMarker::PresentStart, simulation + cpu + 100);
    CHECK(!pacer.BeginPresent(20, 100, simulation + cpu + 200));
    pacer.ObservePlan(plan, simulation + cpu + 200);
    pacer.ObserveReady(21, plan.interval_id, simulation + cpu + gpu + 10000);  // unrelated fence
    pacer.ObserveReady(20, plan.interval_id, simulation + cpu + gpu);
    pacer.SetGenerating(true);
}

void TestPacing(uint32_t multiplier, FgTicks cpu, FgTicks gpu) {
    FgLatencyPacer pacer;
    FakeClock clock;
    const auto plan = Plan(multiplier, 1180000);
    Prime(pacer, plan, cpu, gpu);
    const FgTicks expected = 1180000 + plan.period / multiplier - cpu - gpu - plan.margin - 5000;
    const auto request = pacer.RequestSleep(102, clock.now);
    CHECK(request.handled);
    CHECK(request.wake == (expected > clock.now ? expected : 0));
    clock.SleepUntil(request.wake);
    CHECK(clock.now < 1180000 + plan.period);
    const auto duplicate = pacer.RequestSleep(102, clock.now);
    CHECK(duplicate.handled && duplicate.wake == 0);

    pacer.Marker(103, FgLatencyMarker::PresentStart, clock.now);
    CHECK(!pacer.BeginPresent(21, 103, clock.now));
    pacer.Marker(102, FgLatencyMarker::SimulationStart, clock.now);
    pacer.Marker(102, FgLatencyMarker::PresentStart, clock.now + cpu);
    clock.now += cpu;
    clock.Present(pacer.BeginPresent(22, 102, clock.now));
    CHECK(clock.pre_input_waits == (expected > 1000000 ? 1u : 0u));
    CHECK(clock.present_waits == 0);
    CHECK(!pacer.BeginPresent(23, 102, clock.now));  // marker cannot be reused
}

void TestTransitions() {
    FgLatencyPacer pacer;
    CHECK(!pacer.RequestSleep(99, 1000000).handled);
    Prime(pacer, Plan(2, 1180000), 30000, 50000);
    pacer.SetGenerating(false);  // the menu-copy latch or a disabled FG frame
    CHECK(!pacer.RequestSleep(102, 1000000).handled);
    pacer.SetGenerating(true);
    CHECK(!pacer.RequestSleep(102, 1000000).handled);  // same token already went downstream
    CHECK(pacer.RequestSleep(101, 1000000).wake == 0);

    pacer.Reset();  // resize, detach or device loss
    pacer.ObserveReady(20, 2, 1020000);
    CHECK(!pacer.RequestSleep(103, 1000000).handled);
    Prime(pacer, Plan(2, 1180000), 30000, 50000);
    CHECK(pacer.RequestSleep(104, 1000000, true).wake == 0);
    CHECK(pacer.RequestSleep(104, 1000000).wake > 1000000);  // no ownership assigned to a suppressed attempt
    CHECK(pacer.RequestSleep(109, 4800000).wake == 0);  // stale plan
    pacer.SetOptions(false, 0);
    CHECK(pacer.RequestSleep(105, 1000000).wake == 0);
    pacer.Marker(106, FgLatencyMarker::SimulationStart, 1000000);
    CHECK(pacer.RequestSleep(106, 1000001).wake == 0);  // too late in the frame

    pacer.Reset();
    pacer.SetOptions(true, 0);
    Prime(pacer, Plan(6, 1700000), 30000, 50000);
    CHECK(pacer.RequestSleep(107, 900000).wake == 900000 + FgLatencyPacer::kMaxSleep);
    CHECK(pacer.RequestSleep(107, 1400000, false, false).handled);  // a toggle cannot hand a paced frame to a second owner
    pacer.SetOptions(false, 25000);
    CHECK(pacer.RequestSleep(108, 1400000).wake == 1650000);
}

}  // namespace

int main() {
    CHECK(FgDecodeLatencyMarker(0) == FgLatencyMarker::SimulationStart);
    CHECK(FgDecodeLatencyMarker(3) == FgLatencyMarker::RenderSubmitEnd);
    CHECK(FgDecodeLatencyMarker(4) == FgLatencyMarker::PresentStart);
    CHECK(FgDecodeLatencyMarker(7) == FgLatencyMarker::Other);
    for (uint32_t multiplier = 2; multiplier <= 6; ++multiplier) {
        TestPacing(multiplier, 30000, 50000);
        TestPacing(multiplier, 175000, 20000);
    }
    TestTransitions();
    return failed ? 1 : 0;
}
