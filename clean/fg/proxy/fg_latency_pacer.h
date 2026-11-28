// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "../presenter/fg_presenter.h"

namespace wgfg {

enum class FgLatencyMarker { Other, SimulationStart, RenderSubmitEnd, PresentStart };
FgLatencyMarker FgDecodeLatencyMarker(uint32_t marker);

// The caller serializes observations and supplies interrupt-time ticks. The pacer owns no clock, wait or resources.
class FgLatencyPacer {
public:
    static constexpr FgTicks kMaxSleep = 500000;  // 50 ms

    struct Sleep {
        bool handled = false;
        FgTicks wake = 0;  // 0 when this call should return immediately
    };

    void Reset();
    bool SetOptions(bool enabled, uint32_t frame_limit_us);
    void SetGenerating(bool generating);
    bool Generating() const { return generating_; }
    void Marker(uint32_t native_frame, FgLatencyMarker marker, FgTicks now);
    Sleep RequestSleep(uint32_t native_frame, FgTicks now, bool another_call_waiting = false,
                       bool allow_local = true);

    // Attach the game's marker record to the existing presentation token on the thread that calls Present.
    // True means this frame already got its timing from the pre-input call.
    bool BeginPresent(uint64_t frame, std::optional<uint32_t> native_frame, FgTicks now);
    void ObservePlan(const FgIntervalPlan& plan, FgTicks arrival);
    void ObserveReady(uint64_t frame, uint64_t interval, FgTicks now);

private:
    struct Frame {
        bool used = false;
        uint32_t native = 0;
        FgTicks simulation = 0, submit = 0, present = 0;
        FgTicks sleep_at = 0;
        bool paced = false, forwarded = false;
    };

    struct Completion {
        uint64_t frame = 0, interval = 0;
        FgTicks submit = 0;
    };

    Frame& FindOrStart(uint32_t native);
    void ClearTiming();

    std::array<Frame, 16> frames_{};
    std::array<Completion, 16> completions_{};
    Frame presenting_{};
    uint64_t presenting_frame_ = 0;
    bool generating_ = false;
    bool enabled_ = true;
    FgTicks frame_limit_ = 0;
    FgTicks last_plan_ = 0, next_real_due_ = 0, margin_ = 0;
    FgTicks cpu_ = 0, gpu_ = 0, last_ready_ = 0;
    FgTicks last_wake_ = 0;
};

}  // namespace wgfg
