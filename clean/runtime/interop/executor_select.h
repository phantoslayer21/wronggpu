// SPDX-License-Identifier: MIT
// Which executor runs a context (backend.cpp Context::create): the decision alone, free of D3D12, so that
// tests/test_executor_select.cpp checks it on any host. WrongGPU's own code.
//
// Option `executor` (wronggpu.ini runtime_options, an [fsr] line, WG_RUNTIME_OPTIONS):
//  - not named (or `auto`): every model (M, L, K) on an RDNA 4 adapter (../rdna4.h) runs on the D3D12 executor
//    (executor_d3d12.cpp). On an adapter that is not an RDNA 4 Radeon RX 9000-series card, or one that cannot be
//    identified, the context is not created: the front end's own upscaler (AMD FSR, else its stand-in) takes the
//    game, with one warning line (R1, oversight 2026-11-15: the networks' kernels are RX 9000-only, a HIP attempt
//    only fails at kernel load). The sparkle guard (M, L) and the frame capture, which run on HIP only, select the HIP
//    executor with a warning line. The D3D12 executor can still refuse the context at create
//    (its driver-channel probe, the network's create, a context its kernels do not take): then the HIP executor runs
//    it, again with one line (backend.cpp; model K: no HIP route, the context is not created and the game keeps its own
//    upscaler, dx12_refused_line_k). Model K is taken only when every kernel the context can launch has a
//    D3D12 port in the build's dx12/manifest.json and its options select the exact default path (KRuntime::create on
//    backend 3 checks both before it loads the model; clean/k/runtime/runtime_k.h); else the refusal line names what
//    is missing. A frame it does not take later (e.g. above the checked render size; native AA
//    and every ratio between 1.0 and 1.5 run on it since Brief 15) goes back to the front end
//    (AMD FSR, else its stand-in; hip_switch=off, the default since the owner's decision of 2026-11-04); with
//    hip_switch=on the context moves to the HIP executor for good (backend.cpp switch_to_hip; one line).
//  - `hip`: the HIP executor, always: a diagnostic, not the in-game runtime (owner 2026-11-14: every model runs on the
//    D3D12 runtime in a game); one warning line says so.
//  - `dx12` (or `d3d12`): the D3D12 executor on any adapter (the device refuses unsupported ones and the HIP executor
//    takes the context; model K as above); the sparkle guard and the frame capture keep the HIP executor (warnings, as
//    before) except for model K; frames it does not take get the bilinear fallback (no switch).
//  - Model K (REVIEW-215b, oversight 2026-11-14): never moved to HIP by the sparkle guard (HIP-only: it is turned off
//    for K, one line, K runs on D3D12), never by hip_switch (ignored for K, one line if set), never by a refused frame
//    (the frame goes to the front end's own upscaler, AMD FSR, else its stand-in, the same path as model M's refused
//    frames; a create-level refusal ends the context with one line and the game keeps its own upscaler). The frame
//    capture and executor=hip stay opt-in diagnostics with a loud line.
//  - `d3d12-test`: the offline test executor.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "../rdna4.h"

namespace wgi {

struct ExecutorRequest {
  std::string executor = "auto";  // the option's value ("auto" when not named)
  bool model_k = false;           // the runtime options select model K (D3D12 executor: ports checked at its create)
  // The game device's adapter, from sources a layer in the game cannot rewrite (rdna4_win.h read_adapter_identity):
  bool adapter_known = false;     // the SYSTEM's DXGI described the adapter (never the game's dxgi.dll)
  uint32_t vendor_id = 0;
  bool software = false;          // DXGI_ADAPTER_FLAG_SOFTWARE
  std::string adapter;            // that DXGI description
  bool kmt_known = false;         // the kernel-mode driver's PCI ids were read (they decide first)
  uint32_t kmt_vendor = 0, kmt_device = 0;
  bool sparkle_guard = false;     // option sparkle_guard on
  bool capture = false;           // option capture_dir set
};

struct ExecutorChoice {
  std::string executor;     // hip | dx12 | d3d12-test | fsr (no executor: create ends, the front end's own upscaler
                            // takes the game); empty: unknown option value (create fails)
  bool by_default = false;  // chosen by the default rule (option not named): frames the D3D12 executor does not take
                            // move the context to the HIP executor
  bool warn = false;        // log `why` as a warning (else information)
  bool sparkle_off = false; // model K: the sparkle guard (HIP only) is turned off; the context stays on D3D12
  std::string why;          // the one log line about the choice; empty: nothing to log here
};

// The one line when "auto" meets an adapter that is not an RDNA 4 Radeon RX 9000-series card or is not identified (the
// launcher shows the same sentence): no HIP attempt, no kernel load, the front end's own upscaler takes the game.
inline std::string needs_rx9000_line(bool adapter_known, const std::string& adapter) {
  return "WrongGPU's models need an RX 9000-series GPU; using AMD FSR (" +
         (adapter_known ? "adapter " + (adapter.empty() ? std::string("(no name)") : adapter) : std::string("adapter not identified")) +
         ")";
}

// The one line when model K meets the sparkle guard: the guard is HIP-only, so it is off for K.
inline std::string k_sparkle_off_line() {
  return "model K: the sparkle guard runs on HIP only and is turned off for model K; model K runs on the D3D12 executor";
}

// rdna4.h's identity rule on a request.
inline wgrt::AdapterIdentity adapter_identity(const ExecutorRequest& r) {
  wgrt::AdapterIdentity id;
  id.dxgi_known = r.adapter_known;
  id.dxgi_vendor = r.vendor_id;
  id.software = r.software;
  id.description = r.adapter;
  id.kmt_known = r.kmt_known;
  id.kmt_vendor = r.kmt_vendor;
  id.kmt_device = r.kmt_device;
  return id;
}

inline ExecutorChoice choose_executor(const ExecutorRequest& r) {
  ExecutorChoice c;
  const std::string& e = r.executor;
  if (e == "d3d12-test") {
    c.executor = e;
    return c;
  }
  if (e == "hip") {  // a diagnostic for every model: loud, never the default
    c.executor = e;
    c.warn = true;
    c.why = "diagnostic: executor=hip selects the HIP executor; it is not the in-game runtime (the default and "
            "executor=dx12 run on the D3D12 executor)";
    return c;
  }
  if (e == "dx12" || e == "d3d12") {  // forced (the messages of the option before it had a default)
    if (r.model_k && r.capture) {
      c.executor = "hip";
      c.warn = true;
      c.why = "diagnostic: frame capture forces the HIP executor (model K runs on the D3D12 executor without it)";
      return c;
    }
    if (r.model_k && r.sparkle_guard) {
      c.executor = "dx12";
      c.sparkle_off = true;
      c.warn = true;
      c.why = k_sparkle_off_line();
      return c;
    }
    const char* why = r.sparkle_guard ? "the sparkle guard runs on HIP only"
                      : r.capture     ? "the frame capture runs on HIP only"
                                      : nullptr;
    c.executor = why ? "hip" : "dx12";
    c.warn = c.executor == "hip";
    if (why) c.why = std::string("executor=dx12 not available for this context: ") + why + "; the HIP executor runs it";
    return c;
  }
  if (e != "auto" && !e.empty()) return c;  // unknown value
  c.by_default = true;
  const char* why = nullptr;
  if (!wgrt::rdna4_identity(adapter_identity(r))) {
    c.executor = "fsr";  // every model: the kernels are RX 9000-only; the front end's own upscaler takes the game
    c.warn = true;
    char ids[32] = "";
    if (!r.adapter_known && r.kmt_known) std::snprintf(ids, sizeof ids, "PCI %04X:%04X", (unsigned)r.kmt_vendor, (unsigned)r.kmt_device);
    c.why = needs_rx9000_line(r.adapter_known || r.kmt_known, r.adapter_known ? r.adapter : std::string(ids));
    return c;
  } else if (r.model_k && r.capture) {   // an opt-in diagnostic, never on by default
    c.executor = "hip";
    c.warn = true;
    c.why = "diagnostic: frame capture forces the HIP executor (model K runs on the D3D12 executor without it)";
    return c;
  } else if (r.model_k && r.sparkle_guard) {   // the guard never moves model K to HIP: it is turned off
    c.executor = "dx12";
    c.sparkle_off = true;
    c.warn = true;
    c.why = k_sparkle_off_line();
    return c;
  } else if (r.sparkle_guard) {
    why = "the sparkle guard runs on HIP only";
  } else if (r.capture) {
    why = "the frame capture runs on HIP only";
  }
  if (why) {  // models M, L: the sparkle guard and the frame capture are HIP only (opt-in): a warning, never silent
    c.executor = "hip";
    c.warn = true;
    c.why = std::string("diagnostic: executor hip (default): ") + why +
            " (executor=dx12 / executor=hip in the options force a choice)";
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

// Model K ignores hip_switch (it never runs on HIP in a game): the option's line, empty when the option is not set.
inline std::string k_hip_switch_note(const std::string& option) {
  if (option.empty()) return "";
  return "model K ignores hip_switch=" + option +
         ": a frame the D3D12 executor does not take goes to the front end's own upscaler, never to HIP";
}

// What a context does with a frame the D3D12 executor did not take (nothing of it recorded).
//  kNoSwitch      the frame is answered WG_ERROR_UNSUPPORTED, the context stays usable: the front end takes it from there
//                 (AMD's FSR when it exists, else its stand-in; the same path for every model);
//  kSwitchToHip   the default executor with hip_switch=on moves the context to HIP (never for model K);
//  kBilinear      executor=dx12 (forced): the frame gets the bilinear fallback (never for model K).
// Model K has no state of its own here: every refused K frame is answered WG_ERROR_UNSUPPORTED, exactly as model M's
// refused frames are with hip_switch=off, and the front end does the rest (REVIEW-215c, oversight 2026-11-15).
enum class RefusedAction { kNoSwitch, kSwitchToHip, kBilinear };

inline RefusedAction refused_frame_action(bool model_k, bool by_default, bool hip_switch) {
  if (model_k) return RefusedAction::kNoSwitch;
  if (by_default) return hip_switch ? RefusedAction::kSwitchToHip : RefusedAction::kNoSwitch;
  return RefusedAction::kBilinear;
}

// A frame the executor did not take (nothing recorded) carries this prefix in its error (executor_d3d12.cpp, convert.cpp:
// no output texture, a render size outside the configured maximum, an output texture without unordered access, ...).
inline bool frame_refused(const std::string& err) { return err.rfind("frame not taken", 0) == 0; }

inline std::string k_refusal_line(uint64_t frame, const std::string& why) {
  return "frame " + std::to_string(frame) + ": model K: the D3D12 executor does not take this frame (" + why +
         "); the frame goes to the front end's own upscaler (AMD FSR, else its stand-in), never to the HIP executor";
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
// Model K has no HIP route in a game (owner 2026-11-14: models run on the D3D12 runtime, HIP is a test reference and an
// opt-in diagnostic): the context is not created, the one line says why, and the game keeps its own upscaler.
inline std::string dx12_refused_line_k(bool by_default, const std::string& err) {
  return std::string("model K: ") + (by_default ? "executor dx12 (default on RDNA 4)" : "executor=dx12") +
         " could not take this context (" + err + "); model K has no HIP route in a game (executor=hip selects it as a "
         "diagnostic): the game keeps its own upscaler";
}

inline std::string dx12_refused_line(bool by_default, const std::string& err) {
  return std::string(by_default ? "executor dx12 (default on RDNA 4)" : "executor=dx12") +
         " could not take this context (" + err + "); the HIP executor runs it";
}

}  // namespace wgi
