// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include "fg_presenter.h"

#include <algorithm>
#include <array>

namespace wgfg {

struct FgDisplayPresent {
    FgTicks target;
    FgTicks issue;
    bool real;
    bool gameplay;
};

struct FgDisplaySummary {
    std::uint64_t submitted = 0, real = 0, generated = 0, unseen = 0, unresolved = 0, pending = 0;
    std::uint64_t gameplay_real = 0, gameplay_generated = 0, gameplay_intervals = 0;
    std::uint64_t interval_samples = 0, short_intervals = 0, untimed = 0;
    std::uint64_t disjoint = 0, unsupported = 0, read_errors = 0;
    FgTicks gameplay_time = 0;
    FgTicks interval_p50 = 0, interval_p95 = 0, interval_p99 = 0, interval_max = 0;
    FgTicks latency_p50 = 0, latency_p95 = 0;

    double GameplayMultiplier() const {
        return gameplay_real ? 1.0 + static_cast<double>(gameplay_generated) / gameplay_real : 0.0;
    }
    double GameplayFps() const {
        return gameplay_time > 0 ? static_cast<double>(gameplay_intervals) * 10000000.0 / gameplay_time : 0.0;
    }
};

enum class FgDisplayFeedbackError { Disjoint, Unsupported, ReadError };

// Bounded memory for arbitrarily long sessions. Defaults use 100 us bins saturated at +/-100 ms.
template<FgTicks Bin = 1000, FgTicks First = -1000, FgTicks Last = 1000>
class FgDisplayHistogram {
public:
    void Add(FgTicks value) {
        const auto bin = std::clamp(value / Bin, First, Last);
        ++bins_[static_cast<std::size_t>(bin - First)];
        ++count_;
    }

    FgTicks Percentile(unsigned percent) const {
        if (count_ == 0) return 0;
        const auto rank = (count_ * percent + 99) / 100;
        std::uint64_t cumulative = 0;
        for (std::size_t i = 0; i < bins_.size(); ++i) {
            cumulative += bins_[i];
            if (cumulative >= rank) return (static_cast<FgTicks>(i) + First) * Bin;
        }
        return Last * Bin;
    }

private:
    std::array<std::uint64_t, Last - First + 1> bins_{};
    std::uint64_t count_ = 0;
};

class FgDisplayCadence {
public:
    explicit FgDisplayCadence(FgTicks refresh) : short_limit_(refresh - refresh / 10) {}

    void Submitted() { Both([](Samples& s) { ++s.total.submitted; }); }
    void Unseen(const FgDisplayPresent& present) {
        Both([](Samples& s) { ++s.total.unseen; });
        if (!present.gameplay) last_gameplay_ = false;
    }
    void Unresolved(std::size_t count) {
        Both([=](Samples& s) { s.total.unresolved += count; });
        if (count != 0) BreakTiming();
    }
    void BreakTiming() {
        last_ = 0;
        last_gameplay_ = false;
    }
    void Error(FgDisplayFeedbackError error) {
        Both([=](Samples& s) {
            if (error == FgDisplayFeedbackError::Disjoint) ++s.total.disjoint;
            else if (error == FgDisplayFeedbackError::Unsupported) ++s.total.unsupported;
            else ++s.total.read_errors;
        });
    }

    void Displayed(const FgDisplayPresent& present, FgTicks displayed) {
        // A backwards feedback timestamp starts a new contiguous run.
        if (displayed < last_) BreakTiming();
        Both([&](Samples& s) {
            auto& t = s.total;
            ++(present.real ? t.real : t.generated);
            if (present.gameplay) ++(present.real ? t.gameplay_real : t.gameplay_generated);
            if (displayed <= 0) {
                ++t.untimed;
                return;
            }
            s.latency.Add(displayed - present.target);
            if (last_ == 0) return;
            const FgTicks interval = displayed - last_;
            s.intervals.Add(interval);
            ++t.interval_samples;
            t.interval_max = std::max(t.interval_max, interval);
            if (interval < short_limit_) ++t.short_intervals;
            if (last_gameplay_ && present.gameplay) {
                ++t.gameplay_intervals;
                t.gameplay_time += interval;
            }
        });
        last_ = std::max<FgTicks>(0, displayed);
        last_gameplay_ = displayed > 0 && present.gameplay;
    }

    FgDisplaySummary Session() const { return session_.Summary(); }
    FgDisplaySummary TakeWindow() {
        const auto result = window_.Summary();
        window_ = {};
        return result;
    }

private:
    struct Samples {
        FgDisplaySummary total;
        FgDisplayHistogram<> intervals, latency;

        FgDisplaySummary Summary() const {
            auto result = total;
            result.interval_p50 = intervals.Percentile(50);
            result.interval_p95 = intervals.Percentile(95);
            result.interval_p99 = intervals.Percentile(99);
            result.latency_p50 = latency.Percentile(50);
            result.latency_p95 = latency.Percentile(95);
            return result;
        }
    };

    template<class Update>
    void Both(Update update) {
        update(session_);
        update(window_);
    }

    Samples session_, window_;
    FgTicks short_limit_ = 0;
    FgTicks last_ = 0;
    bool last_gameplay_ = false;
};

// DXGI reports only the latest ID at each poll. Older tracked IDs may have been
// skipped or displayed between polls. A disjoint epoch cannot resolve pending IDs.
class FgDxgiDisplayCadence {
public:
    explicit FgDxgiDisplayCadence(FgTicks refresh) : cadence_(refresh) {}

    void Submitted(std::uint32_t id, const FgDisplayPresent& present) {
        cadence_.Submitted();
        if (have_id_ && !Before(last_id_, id)) {
            cadence_.Unresolved(1);
            return;
        }
        if (count_ == entries_.size()) {
            Pop();
            cadence_.Unresolved(1);
        }
        entries_[(head_ + count_) % entries_.size()] = {id, present};
        ++count_;
        last_id_ = id;
        have_id_ = true;
    }

    void MissingId() {
        cadence_.Submitted();
        cadence_.Unresolved(1);
    }

    void ObserveLatest(std::uint32_t id, FgTicks displayed) {
        while (count_ && Before(entries_[head_].id, id)) {
            cadence_.Unseen(entries_[head_].present);
            Pop();
        }
        if (count_ && entries_[head_].id == id) {
            cadence_.Displayed(entries_[head_].present, displayed);
            Pop();
        }
    }

    void Discontinuity(FgDisplayFeedbackError error) {
        cadence_.Error(error);
        if (error != FgDisplayFeedbackError::ReadError) EndEpoch();
    }

    void EndEpoch() {
        cadence_.Unresolved(count_);
        cadence_.BreakTiming();
        head_ = count_ = 0;
        have_id_ = false;
    }

    FgDisplaySummary Session() const {
        auto result = cadence_.Session();
        result.pending = count_;
        return result;
    }
    FgDisplaySummary TakeWindow() {
        auto result = cadence_.TakeWindow();
        result.pending = count_;
        return result;
    }

private:
    struct Entry {
        std::uint32_t id;
        FgDisplayPresent present;
    };
    static bool Before(std::uint32_t a, std::uint32_t b) { return a != b && b - a < 0x80000000u; }
    void Pop() {
        head_ = (head_ + 1) % entries_.size();
        --count_;
    }

    FgDisplayCadence cadence_;
    std::array<Entry, 64> entries_{};
    std::size_t head_ = 0, count_ = 0;
    std::uint32_t last_id_ = 0;
    bool have_id_ = false;
};

}  // namespace wgfg
