// SPDX-License-Identifier: MIT
#pragma once
// Original WrongGPU publication gate: keep our code/state resident before a host can call it.
// Pin is supplied by the platform adapter; tests inject it without loading or patching a host.
#include "asi_patch.h"

namespace wgasi {
struct PinnedCallResult {
    bool modulePinned = false;
    CallPatchTransaction::Result patch{};
};

// Caller completes host recognition, plan and writer preflight before reaching this gate.
// Pin must retain OUR module until process termination, without modifying host call sites.
// Failure never invokes the writer. A successful pin is intentionally never undone, including
// when Apply rejects or rolls back: another thread could already have entered published code.
// This does not pin the host, exclude foreign threads, or make call-site writes atomic.
template<class Pin, class Writer>
PinnedCallResult PublishPinnedCalls(CallPatchTransaction& transaction, Pin&& pin, Writer&& writer) {
    static_assert(noexcept(pin()), "Module pin must not throw");
    if (!pin()) return {};
    return {true, transaction.Apply(writer)};
}
} // namespace wgasi
