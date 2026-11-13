// SPDX-License-Identifier: MIT
#pragma once
// Original WrongGPU status bookkeeping. No host calls, resource exchange, or backend inference.
#include <atomic>
#include "asi_abi.h"

namespace wgasi {
struct DiagnosticState {
    std::atomic<uint32_t> initialization{WG_ASI_INIT_NOT_ATTEMPTED};
    std::atomic<uint32_t> menu{WG_ASI_MENU_UNAVAILABLE};
    std::atomic<uint32_t> reason{WG_ASI_REASON_NOT_INITIALIZED};

    void Begin() {
        initialization.store(WG_ASI_INIT_IN_PROGRESS);
        reason.store(WG_ASI_REASON_INITIALIZING);
    }
    void Outcome(uint32_t menuState, uint32_t why) {
        menu.store(menuState);
        reason.store(why);
    }
    void Complete() { initialization.store(WG_ASI_INIT_COMPLETED); }

    int Query(WrongGpuAsiStatusV1* out, uint32_t callerSize, bool hooked, char model) const {
        if (!out) return WG_ASI_STATUS_INVALID_ARGUMENT;
        // Do not read even the header unless the separately supplied capacity covers it.
        if (callerSize < 2 * sizeof(uint32_t)) return WG_ASI_STATUS_INVALID_SIZE;
        if (out->size < sizeof(*out) || out->size > callerSize || callerSize < sizeof(*out))
            return WG_ASI_STATUS_INVALID_SIZE;
        if (out->version != WRONGGPU_ASI_STATUS_VERSION) return WG_ASI_STATUS_UNSUPPORTED_VERSION;
        WrongGpuAsiStatusV1 value{};
        value.size = sizeof(value);
        value.version = WRONGGPU_ASI_STATUS_VERSION;
        value.loaded = 1;
        value.initialization = initialization.load();
        value.hooked = hooked ? 1u : 0u;
        value.menu_state = menu.load();
        value.menu_reason = reason.load();
        value.selected_model = static_cast<unsigned char>(model);
        value.effective_backend = WG_ASI_EFFECTIVE_BACKEND_UNKNOWN;
        value.rr_ready = 0;
        value.rr_blockers = WG_ASI_RR_MISSING_INTERFACE | WG_ASI_RR_MISSING_F_CONTRACTS |
                            WG_ASI_RR_BACKEND_UNAVAILABLE | WG_ASI_RR_HOST_UNQUALIFIED;
        *out = value;
        return WG_ASI_STATUS_OK;
    }
};
} // namespace wgasi
