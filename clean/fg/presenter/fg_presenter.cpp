// SPDX-License-Identifier: MIT
// Names of the presentation boundary's status values, for logs and test output.
#include "fg_presenter.h"

namespace wgfg {

const char* FgStatusName(FgStatus s) {
    switch (s) {
        case FgStatus::Ok: return "ok";
        case FgStatus::InvalidArgument: return "invalid-argument";
        case FgStatus::WrongState: return "wrong-state";
        case FgStatus::StaleEpoch: return "stale-epoch";
        case FgStatus::NoSlot: return "no-slot";
        case FgStatus::Busy: return "busy";
        case FgStatus::Unsupported: return "unsupported";
        case FgStatus::Refused: return "refused";
        case FgStatus::DeviceLost: return "device-lost";
        case FgStatus::Overflow: return "overflow";
        case FgStatus::ResizeFailed: return "resize-failed";
    }
    return "unknown";
}

const char* FgRealOnlyReasonName(FgRealOnlyReason r) {
    switch (r) {
        case FgRealOnlyReason::None: return "none";
        case FgRealOnlyReason::GenerationOff: return "generation-off";
        case FgRealOnlyReason::NotPrimed: return "not-primed";
        case FgRealOnlyReason::MenuDisabled: return "menu-disabled";
        case FgRealOnlyReason::Reset: return "reset";
        case FgRealOnlyReason::OutOfVrrWindow: return "out-of-vrr-window";
        case FgRealOnlyReason::Reanchor: return "reanchor";
        case FgRealOnlyReason::LatencyCap: return "latency-cap";
        case FgRealOnlyReason::Pressure: return "pressure";
        case FgRealOnlyReason::EpochChanged: return "epoch-changed";
        case FgRealOnlyReason::DeviceLost: return "device-lost";
    }
    return "unknown";
}

const char* FgDropReasonName(FgDropReason r) {
    switch (r) {
        case FgDropReason::None: return "none";
        case FgDropReason::Late: return "late";
        case FgDropReason::Pressure: return "pressure";
        case FgDropReason::Cancelled: return "cancelled";
        case FgDropReason::Revoked: return "revoked";
        case FgDropReason::Epoch: return "epoch";
    }
    return "unknown";
}

}  // namespace wgfg
