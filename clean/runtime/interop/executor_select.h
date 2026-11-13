// SPDX-License-Identifier: MIT
// Which executor runs a context (backend.cpp Context::create): the decision alone, free of D3D12, so that
// tests/test_executor_select.cpp checks it on any host. WrongGPU's own code.
//
// Option `executor` (wronggpu.ini runtime_options, an [fsr] line, WG_RUNTIME_OPTIONS):
//  - not named (or `auto`): Model M on an RDNA 4 adapter (../rdna4.h) runs on the D3D12 executor (executor_d3d12.cpp),
//    everything else on the HIP executor, with one log line saying why (other GPU, model K, the sparkle guard or the
//    frame capture, which run on HIP only). The D3D12 executor can still refuse the context at create (its
//    driver-channel probe, the network's create, a context its kernels do not take): then the HIP executor runs it,
//    again with one line (backend.cpp). A frame it does not take later (e.g. above the checked render size; native AA
//    and every ratio between 1.0 and 1.5 run on it since Brief 15) goes back to the front end
//    (AMD FSR, else its stand-in; hip_switch=off, the default since the owner's decision of 2026-11-04); with
//    hip_switch=on the context moves to the HIP executor for good (backend.cpp switch_to_hip; one line).
//  - `hip`: the HIP executor, always.
//  - `dx12` (or `d3d12`): the D3D12 executor on any adapter (the device refuses unsupported ones and the HIP executor
//    takes the context); model K, the sparkle guard and the frame capture keep the HIP executor (warnings, as before);
//    frames it does not take get the bilinear fallback (no switch).
//  - `d3d12-test`: the offline test executor.
#pragma once

#include <cstdint>
#include <string>

#include "../rdna4.h"

namespace wgi {

struct ExecutorRequest {
  std::string executor = "auto";  // the option's value ("auto" when not named)
  bool model_k = false;           // the runtime options select model K
  bool adapter_known = false;     // the game device's adapter could be read (DXGI)
  uint32_t vendor_id = 0;
  bool software = false;          // DXGI_ADAPTER_FLAG_SOFTWARE
  std::string adapter;            // DXGI description
  bool sparkle_guard = false;     // option sparkle_guard on
  bool capture = false;           // option capture_dir set
};

struct ExecutorChoice {
  std::string executor;     // hip | dx12 | d3d12-test; empty: unknown option value (create fails)
  bool by_default = false;  // chosen by the default rule (option not named): frames the D3D12 executor does not take
                            // move the context to the HIP executor
  bool warn = false;        // log `why` as a warning (else information)
  std::string why;          // the one log line about the choice; empty: nothing to log here
};

inline ExecutorChoice choose_executor(const ExecutorRequest& r) {
  ExecutorChoice c;
  const std::string& e = r.executor;
  if (e == "hip" || e == "d3d12-test") {
    c.executor = e;
    return c;
  }
  if (e == "dx12" || e == "d3d12") {  // forced (the messages of the option before it had a default)
    const char* why = r.model_k ? nullptr
                      : r.sparkle_guard ? "the sparkle guard runs on HIP only"
                      : r.capture       ? "the frame capture runs on HIP only"
                                        : nullptr;
    c.executor = r.model_k || why ? "hip" : "dx12";
    c.warn = c.executor == "hip";
    if (r.model_k) c.why = "executor=dx12: model K has no D3D12 port; the HIP executor runs it";
    else if (why) c.why = std::string("executor=dx12 not available for this context: ") + why + "; the HIP executor runs it";
    return c;
  }
  if (e != "auto" && !e.empty()) return c;  // unknown value
  c.by_default = true;
  const std::string gpu = r.adapter_known ? "the adapter " + (r.adapter.empty() ? std::string("(no name)") : r.adapter)
                                          : std::string("the adapter (not identified)");
  const char* why = nullptr;
  std::string text;
  if (r.model_k) {
    why = "model K has no D3D12 port";
  } else if (!r.adapter_known || !wgrt::rdna4_adapter(r.vendor_id, r.software, r.adapter)) {
    text = gpu + " is not an RDNA 4 Radeon RX 9070 / RX 9060, where the D3D12 executor runs";
    why = text.c_str();
  } else if (r.sparkle_guard) {
    why = "the sparkle guard runs on HIP only";
  } else if (r.capture) {
    why = "the frame capture runs on HIP only";
  }
  if (why) {
    c.executor = "hip";
    c.why = std::string("executor hip (default): ") + why + " (executor=dx12 / executor=hip in the options force a choice)";
    return c;
  }
  c.executor = "dx12";  // the success line is logged once the D3D12 executor has taken the context
  return c;
}

// clean/fe-wrapped-list (Brief 12): what a context on the default D3D12 executor does when that executor does not take a
// frame mid-session (a render size above the one checked at create). Option `hip_switch`:
//  - `on`: the HIP executor runs this frame and every later one of the context (switch_to_hip; the 0.6.0 behaviour);
//  - `off`: no switch. The frame is answered WG_ERROR_UNSUPPORTED with nothing recorded; the front end hands the context
//    to AMD's FSR, else its stand-in upscales such frames. The context stays on the D3D12 executor.
// Not named: kHipSwitchDefault, off (owner 2026-11-04, Brief 14: no mid-session HIP; oversize frames go to
// AMD FSR, else the stand-in). Another value: the default, with a note. Forced executors (executor=dx12 /
// executor=hip) are not affected.
inline constexpr bool kHipSwitchDefault = false;

struct RefusedFramePolicy {
  bool switch_to_hip = kHipSwitchDefault;
  std::string note;  // a log line about the option's value; empty: nothing to say
};

inline RefusedFramePolicy refused_frame_policy(const std::string& option) {
  RefusedFramePolicy p;
  if (option.empty()) return p;
  if (option == "on" || option == "1" || option == "true") {
    p.switch_to_hip = true;
  } else if (option == "off" || option == "0" || option == "false") {
    p.switch_to_hip = false;
  } else {
    p.note = "hip_switch=" + option + " not recognised (on or off); using " + (kHipSwitchDefault ? "on" : "off");
  }
  return p;
}

// The one warning line of a context with hip_switch=off whose D3D12 executor does not take a frame.
inline std::string no_switch_line(uint64_t frame, const std::string& why) {
  return "frame " + std::to_string(frame) + ": executor dx12 (default on RDNA 4) does not take this frame (" + why +
         "); hip_switch=off: the context does not move to the HIP executor; such frames go back to the front end "
         "(AMD FSR, else its stand-in)";
}

// After choose_executor gave dx12: the D3D12 executor did not take the context at create (`err`: the driver-channel
// probe, the network's create on the D3D12 device, a context its kernels do not take). The HIP executor runs it; this
// is the one warning line about it.
inline std::string dx12_refused_line(bool by_default, const std::string& err) {
  return std::string(by_default ? "executor dx12 (default on RDNA 4)" : "executor=dx12") +
         " could not take this context (" + err + "); the HIP executor runs it";
}

}  // namespace wgi
