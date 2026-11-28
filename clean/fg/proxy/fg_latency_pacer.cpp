// SPDX-License-Identifier: GPL-3.0-only
#include "fg_latency_pacer.h"

#include <algorithm>
#include <limits>

namespace wgfg {
namespace {

constexpr FgTicks kStale = 2500000;  // 250 ms
constexpr FgTicks kSafetyMargin = 5000;  // 0.5 ms

FgTicks Estimate(FgTicks previous, FgTicks sample) {
    // React immediately to increased work. Decay slowly as the workload gets lighter.
    return previous <= sample ? sample : previous - (previous - sample + 3) / 4;
}

}  // namespace

FgLatencyMarker FgDecodeLatencyMarker(uint32_t marker) {
    switch (marker) {
        case 0: return FgLatencyMarker::SimulationStart;
        case 3: return FgLatencyMarker::RenderSubmitEnd;
        case 4: return FgLatencyMarker::PresentStart;
        default: return FgLatencyMarker::Other;
    }
}

FgLatencyPacer::Frame& FgLatencyPacer::FindOrStart(uint32_t native) {
    Frame& frame = frames_[native % frames_.size()];
    if (!frame.used || frame.native != native) {
        frame = {};
        frame.used = true;
        frame.native = native;
    }
    return frame;
}

void FgLatencyPacer::ClearTiming() {
    completions_ = {};
    presenting_ = {};
    presenting_frame_ = 0;
    last_plan_ = next_real_due_ = margin_ = cpu_ = gpu_ = last_ready_ = last_wake_ = 0;
}

void FgLatencyPacer::Reset() {
    frames_ = {};
    generating_ = false;
    ClearTiming();
}

bool FgLatencyPacer::SetOptions(bool enabled, uint32_t frame_limit_us) {
    const auto limit = static_cast<FgTicks>(frame_limit_us) * 10;
    if (enabled == enabled_ && limit == frame_limit_) return false;
    enabled_ = enabled;
    frame_limit_ = limit;
    return true;
}

void FgLatencyPacer::SetGenerating(bool generating) {
    if (!generating && generating_) ClearTiming();
    generating_ = generating;
}

void FgLatencyPacer::Marker(uint32_t native_frame, FgLatencyMarker marker, FgTicks now) {
    if (marker == FgLatencyMarker::Other || now <= 0) return;
    Frame& frame = FindOrStart(native_frame);
    switch (marker) {
        case FgLatencyMarker::SimulationStart:
            if (!frame.simulation) frame.simulation = now;
            break;
        case FgLatencyMarker::RenderSubmitEnd: frame.submit = now; break;
        case FgLatencyMarker::PresentStart: frame.present = now; break;
        case FgLatencyMarker::Other: break;
    }
}

FgLatencyPacer::Sleep FgLatencyPacer::RequestSleep(uint32_t native_frame, FgTicks now, bool another_call_waiting,
                                                   bool allow_local) {
    if (now <= 0) return {};
    Frame& frame = FindOrStart(native_frame);
    if (frame.sleep_at && now - frame.sleep_at > kStale) {
        frame = {};
        frame.used = true;
        frame.native = native_frame;
    }
    if (frame.sleep_at) return {!frame.forwarded, 0};
    if (!generating_ || !allow_local) {
        frame.sleep_at = now;
        frame.forwarded = true;
        return {};
    }
    if (frame.simulation || another_call_waiting) return {true, 0};
    frame.sleep_at = now;
    if (now < last_plan_ || now - last_plan_ > kStale || now > std::numeric_limits<FgTicks>::max() - kMaxSleep)
        return {true, 0};

    const bool prediction = enabled_ && cpu_ > 0 && gpu_ > 0 && next_real_due_ > 0 &&
                            now >= last_ready_ && now - last_ready_ <= kStale;
    if (!prediction && frame_limit_ == 0) return {true, 0};
    FgTicks wake = prediction ? next_real_due_ - cpu_ - gpu_ - margin_ : 0;
    if (frame_limit_ && last_wake_ && now - last_wake_ < kStale &&
        last_wake_ <= std::numeric_limits<FgTicks>::max() - kMaxSleep)
        wake = std::max(wake, last_wake_ + std::min(frame_limit_, kMaxSleep));
    last_wake_ = std::clamp(wake, now, now + kMaxSleep);
    frame.paced = true;
#if defined(WG_FG_LATENCY_MUTANT) && WG_FG_LATENCY_MUTANT == 1
    return {true, std::numeric_limits<FgTicks>::max()};
#elif defined(WG_FG_LATENCY_MUTANT) && WG_FG_LATENCY_MUTANT == 2
    return {true, last_wake_ + 100000};
#else
    return {true, last_wake_ > now ? last_wake_ : 0};
#endif
}

bool FgLatencyPacer::BeginPresent(uint64_t frame, std::optional<uint32_t> native_frame, FgTicks now) {
    presenting_frame_ = frame;
    presenting_ = {};
    if (!native_frame) return false;
    Frame& entry = frames_[*native_frame % frames_.size()];
    if (!entry.used || entry.native != *native_frame || !entry.present ||
        entry.present > now || now - entry.present > kMaxSleep)
        return false;
    presenting_ = entry;
    entry = {};
#if defined(WG_FG_LATENCY_MUTANT) && WG_FG_LATENCY_MUTANT == 3
    return false;
#else
    return presenting_.paced;
#endif
}

void FgLatencyPacer::ObservePlan(const FgIntervalPlan& plan, FgTicks arrival) {
    if (plan.generated_count == 0 || plan.target_count != plan.generated_count + 1 ||
        plan.generated_count > kFgMaxGenerated || plan.period <= 0 || plan.period > kStale)
        return;
    const FgTicks real = plan.targets[plan.generated_count].target;
    const FgTicks first = plan.targets[0].target;
    if (arrival <= 0 || first <= 0 || first > real || real - first > plan.period || plan.margin < 0 ||
        plan.margin > kStale || real < arrival || real - arrival > kStale ||
        real > std::numeric_limits<FgTicks>::max() - plan.period)
        return;
    last_plan_ = arrival;
    next_real_due_ = real + plan.period;
    margin_ = real - first + plan.margin + kSafetyMargin;
    if (presenting_frame_ != plan.real_frame_id || presenting_.simulation <= 0 ||
        presenting_.submit <= presenting_.simulation || presenting_.present < presenting_.submit ||
        arrival - presenting_.simulation > kStale)
        return;
    cpu_ = Estimate(cpu_, presenting_.submit - presenting_.simulation);
    completions_[plan.interval_id % completions_.size()] = {plan.real_frame_id, plan.interval_id, presenting_.submit};
}

void FgLatencyPacer::ObserveReady(uint64_t frame, uint64_t interval, FgTicks now) {
    Completion& sample = completions_[interval % completions_.size()];
    if (sample.interval != interval || sample.frame != frame || sample.submit <= 0 ||
        now <= sample.submit || now - sample.submit > kStale)
        return;
    if (now >= last_ready_) {
        gpu_ = Estimate(gpu_, now - sample.submit);
        last_ready_ = now;
    }
    sample = {};
}

}  // namespace wgfg
