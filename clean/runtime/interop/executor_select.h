// SPDX-License-Identifier: MIT
// Which executor runs a context (backend.cpp Context::create): the decision alone, free of D3D12, so that
// tests/test_executor_select.cpp checks it on any host. WrongGPU's own code.
//
// Option `executor` (wronggpu.ini runtime_options, an [fsr] line, WG_RUNTIME_OPTIONS). R1 (owner 2026-11-14, oversight
// 2026-11-15): in a game every model (M, L, K) runs on the D3D12 executor and NOTHING in a game starts HIP. So:
//  - not named (or `auto`): the D3D12 executor (executor_d3d12.cpp) on an RDNA 4 adapter (../rdna4.h: the PCI vendor and
//    device id from the kernel-mode driver, never a description a layer in the game can rewrite, failing closed). On any
//    other adapter, or one whose ids cannot be read, the context is not created: one warning line (the sentence of
//    wgrt::needs_rx9000_sentence with the adapter's description and ids) and the front end's own upscaler (AMD FSR, else
//    its stand-in) takes the game, whatever fallback_to_amd says (a hard incompatibility, not a runtime failure); no HIP
//    attempt, no kernel load. When the D3D12 executor refuses the context at create (its driver-channel probe, device
//    preparation, the network's create, a context its kernels do not take; a build without its D3D12 kernel set; model K:
//    every kernel must have a port in the build's dx12/manifest.json and its options select the exact default path), the
//    context is not created either: one error line and the front end's upscaler; never the HIP executor. The sparkle guard
//    (HIP only) is turned off for every model with one line; the model stays on D3D12.
//  - `dx12` (or `d3d12`): the D3D12 executor on any adapter (the device refuses unsupported ones, and the context is not
//    created).
//  - A frame the D3D12 executor does not take later (e.g. above the checked render size) is answered WG_ERROR_UNSUPPORTED
//    for every model and every executor choice and goes back to the front end (AMD FSR, else its stand-in): never a
//    substitute image made here, never a move to HIP. The 0.6.0 option hip_switch=on is ignored for every model (one line).
//  - `executor=hip` (the HIP executor, a test reference) and the frame capture (option capture_dir, HIP only) are developer
//    diagnostics: IGNORED in a release build, with one warning line ("ignored: diagnostic key, release build"); the context
//    stays on D3D12. They work only in a build made with WG_DEV_DIAGNOSTICS (clean/runner/build.py --dev-diagnostics),
//    which no release build and no package is: tests/check_no_dev_diagnostics.py checks the built library for the marker.
//  - `d3d12-test`: the offline test executor.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../rdna4.h"

namespace wgi {

// A developer build (WG_DEV_DIAGNOSTICS): executor=hip and the frame capture work. Never in a release build.
#ifdef WG_DEV_DIAGNOSTICS
inline constexpr bool kDevDiagnostics = true;
#else
inline constexpr bool kDevDiagnostics = false;
#endif

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
  bool wine_fallback = false;     // Wine only: the driver's query is not implemented there; the system DXGI's ids (#248)
  uint32_t wine_vendor = 0, wine_device = 0;
  bool sparkle_guard = false;     // option sparkle_guard on
  bool capture = false;           // option capture_dir set
  std::string hip_switch;         // the 0.6.0 option hip_switch (ignored for every model: hip_switch_note)
};

struct ExecutorChoice {
  std::string executor;     // hip | dx12 | d3d12-test | fsr (no executor: create ends, the front end's own upscaler
                            // takes the game); empty: unknown option value (create fails)
  bool by_default = false;  // chosen by the default rule (option not named): frames the D3D12 executor does not take
                            // move the context to the HIP executor
  bool warn = false;        // log `why` as a warning (else information)
  bool sparkle_off = false; // the sparkle guard (HIP only) is turned off, whatever the model; the context stays on D3D12
  std::string why;          // the one log line about the choice; empty: nothing to log here
  std::string hip_note;     // the one line when the option hip_switch asks for a move to HIP (ignored, any model)
  std::string adapter_note; // the one line when the Wine fallback identified the adapter (rdna4.h, #248); else empty
  std::vector<std::string> ignored;  // release build: one warning line per diagnostic key that was ignored (executor=hip,
                                     // capture_dir); the context stays on D3D12 (the backend also drops capture_dir)
};

// A diagnostic key of a developer build asked for in a release build.
inline std::string ignored_key_line(const std::string& key) {
  return key + " ignored: diagnostic key, release build; the context runs on the D3D12 executor (a game never starts the "
               "HIP runtime)";
}

// The one line when "auto" meets an adapter that is not an RDNA 4 Radeon RX 9000-series card or cannot be identified (the
// launcher shows the same sentence): no HIP attempt, no kernel load, the front end's own upscaler takes the game. The
// description (the system DXGI's) and the driver's ids are in it, so that a board missing from the id list shows in a log.
inline std::string needs_rx9000_line(const wgrt::AdapterIdentity& id) {
  return std::string(wgrt::needs_rx9000_sentence()) + " (" + wgrt::describe_identity(id) + ")";
}

// The one line when the sparkle guard is on: it runs on HIP only, so it is off, whatever the model; the model stays on D3D12.
inline std::string sparkle_off_line() {
  return "the sparkle guard runs on HIP only and is turned off; the model runs on the D3D12 executor";
}

// The one line for the frame capture (an opt-in diagnostic that needs the HIP executor). Developer builds only: the text is
// not in a release build's library at all (the package gate looks for it: clean/tests/check_no_dev_diagnostics.py).
#ifdef WG_DEV_DIAGNOSTICS
inline std::string capture_hip_line() {
  return "diagnostic: frame capture forces the HIP executor (the model runs on the D3D12 executor without it)";
}
#endif

// rdna4.h's identity of a request.
inline wgrt::AdapterIdentity adapter_identity(const ExecutorRequest& r) {
  wgrt::AdapterIdentity id;
  id.dxgi_known = r.adapter_known;
  id.dxgi_vendor = r.vendor_id;
  id.software = r.software;
  id.description = r.adapter;
  id.kmt_known = r.kmt_known;
  id.kmt_vendor = r.kmt_vendor;
  id.kmt_device = r.kmt_device;
  id.wine_fallback = r.wine_fallback;
  id.wine_vendor = r.wine_vendor;
  id.wine_device = r.wine_device;
  return id;
}

inline std::string hip_switch_note(const std::string& option);  // below

// `dev`: whether the developer diagnostics are on. They can only be on in a build made with WG_DEV_DIAGNOSTICS (a release
// build has no code or text for them whatever is passed), so the tests pass both values and the two build configurations.
inline ExecutorChoice choose_executor(const ExecutorRequest& r, bool devRequested = kDevDiagnostics) {
  const bool dev = kDevDiagnostics && devRequested;
  ExecutorChoice c;
  c.hip_note = hip_switch_note(r.hip_switch);  // the same for every model: the option is ignored
  c.adapter_note = wgrt::wine_fallback_line(adapter_identity(r));
  std::string e = r.executor;
  bool capture = r.capture;
  if (!dev) {  // R1: a release build never starts HIP, whatever the game's files say
    if (e == "hip") {
      c.ignored.push_back(ignored_key_line("executor=hip"));
      e = "auto";
    }
    if (capture) {
      c.ignored.push_back(ignored_key_line("capture_dir"));
      capture = false;
    }
  }
  if (e == "d3d12-test") {
    c.executor = e;
    return c;
  }
#ifdef WG_DEV_DIAGNOSTICS
  if (e == "hip") {  // developer build only: a diagnostic for every model, loud, never the default
    c.executor = e;
    c.warn = true;
    c.why = "diagnostic: executor=hip selects the HIP executor; it is not the in-game runtime (the default and "
            "executor=dx12 run on the D3D12 executor)";
    return c;
  }
#endif
  const bool forced = e == "dx12" || e == "d3d12";
  if (!forced && e != "auto" && !e.empty()) return c;  // unknown value
  if (!forced) {
    c.by_default = true;
    if (!wgrt::rdna4_identity(adapter_identity(r))) {
      c.executor = "fsr";  // every model: the kernels are RX 9000-only; the front end's own upscaler takes the game
      c.warn = true;
      c.why = needs_rx9000_line(adapter_identity(r));
      return c;
    }
  }
#ifdef WG_DEV_DIAGNOSTICS
  if (capture) {  // developer build only: an opt-in diagnostic, never on by default
    c.executor = "hip";
    c.warn = true;
    c.why = capture_hip_line();
    return c;
  }
#endif
  c.executor = "dx12";  // the success line is logged once the D3D12 executor has taken the context
  if (r.sparkle_guard) {  // the guard is HIP only: it is off, the model stays on D3D12
    c.sparkle_off = true;
    c.warn = true;
    c.why = sparkle_off_line();
  }
  return c;
}

// What a context on the default D3D12 executor does when that executor does not take a frame mid-session (a render size
// above the one checked at create): the frame is answered WG_ERROR_UNSUPPORTED with nothing recorded; the front end hands
// the context to AMD's FSR, else its stand-in upscales such frames. The context stays on the D3D12 executor: no model
// moves to HIP in a game (R1). The 0.6.0 option `hip_switch=on` (a model M or L context moved to HIP for good) is ignored
// for every model, with one line; `off` (the default since the owner's decision of 2026-11-04) is the behaviour.
// executor=hip is the explicit diagnostic and is not affected.
inline std::string hip_switch_note(const std::string& option) {
  if (option.empty() || option == "off" || option == "0" || option == "false") return "";
  return "hip_switch=" + option +
         " is ignored: a frame the D3D12 executor does not take goes to the front end's own upscaler, never to HIP "
         "(executor=hip is the explicit diagnostic)";
}

// What a context does with a frame the D3D12 executor did not take (nothing of it recorded): the frame is answered
// WG_ERROR_UNSUPPORTED, the context stays usable, and the front end takes it from there (AMD's FSR when it exists, else its
// stand-in). The same for every model and for a forced executor=dx12: never an image made here (no bilinear substitute),
// never a move to HIP (REVIEW-215c, oversight 2026-11-15; review finding on #225).
inline constexpr bool kRefusedFrameGoesToFrontEnd = true;

// H3 (R1): the output filters Quality 5 and Quality 9 (runtime option output_filter=quality5 | quality9) exist on the HIP
// executor only; the D3D12 executor refuses them at create. A context on the D3D12 executor never goes to HIP because of
// them: the option is dropped, the default filter runs, and `*note` is the one log line. `options` is the network
// runtime's "key=value;..." string; the last output_filter wins there, so a later quality value drops every output_filter
// item (an earlier quality value followed by another filter is only a leftover and is dropped too). Nothing else changes.
inline std::string drop_hip_only_filter(const std::string& options, std::string* note) {
  struct Item {
    std::string text, key, value;
  };
  auto trim = [](const std::string& t) {
    const size_t a = t.find_first_not_of(" \t"), b = t.find_last_not_of(" \t");
    return a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
  };
  std::vector<Item> items;
  size_t pos = 0;
  while (pos <= options.size()) {
    size_t end = options.find(';', pos);
    if (end == std::string::npos) end = options.size();
    Item it;
    it.text = options.substr(pos, end - pos);
    pos = end + 1;
    const size_t eq = it.text.find('=');
    it.key = trim(it.text.substr(0, eq));
    it.value = eq == std::string::npos ? std::string() : trim(it.text.substr(eq + 1));
    if (!it.key.empty()) items.push_back(it);
  }
  std::string last;
  for (const Item& it : items)
    if (it.key == "output_filter") last = it.value;
  if (last != "quality5" && last != "quality9") return options;
  std::string out;
  for (const Item& it : items) {
    if (it.key == "output_filter") continue;
    out += (out.empty() ? "" : ";") + it.text;
  }
  if (note)
    *note = "output_filter=" + last + " is not available on the D3D12 runtime yet; the default output filter is used "
            "(the HIP executor is not started for it)";
  return out;
}

// A frame the executor did not take (nothing recorded) carries this prefix in its error (executor_d3d12.cpp, convert.cpp:
// no output texture, a render size outside the configured maximum, an output texture without unordered access, ...).
inline bool frame_refused(const std::string& err) { return err.rfind("frame not taken", 0) == 0; }

// Model L hybrid: the frame after one whose bottleneck wait timed out on the GPU (runtime.cpp kBneckTimeoutRefusal;
// hyb_bneck.hlsl). The timed-out frame and the frames queued after it were skipped on the GPU (the device's frame guard
// predicates the network and the executor's output pass alike), and each of them got the executor's fallback instead:
// under the INVERSE predication on the same word, a simple upscale of that frame's colour into that frame's output
// texture (Converter::record_output with no result, or through the route's texture of ours), so every such frame,
// the first one of a context included, shows an image whatever output texture the game passed. This frame (the one
// whose start sees the timeout count move) records nothing that writes (only the game textures' state changes,
// restored) and takes the path of a frame the executor did not take: the backend logs one line and answers
// WG_ERROR_UNSUPPORTED, so the front end's stand-in (AMD FSR, else its own upscale) renders it; the context is kept
// (never failed_), and the next frame is a reset (the runtime starts over, and the backend asks for the reset too).
inline bool frame_timeout_refused(const std::string& err) {
  return err.rfind("frame refused (bottleneck wait timeout): ", 0) == 0;
}

inline std::string k_refusal_line(uint64_t frame, const std::string& why) {
  return "frame " + std::to_string(frame) + ": model K: the D3D12 executor does not take this frame (" + why +
         "); the frame goes to the front end's own upscaler (AMD FSR, else its stand-in), never to the HIP executor";
}

// The one warning line of a context whose default D3D12 executor does not take a frame.
inline std::string no_switch_line(uint64_t frame, const std::string& why) {
  return "frame " + std::to_string(frame) + ": executor dx12 (default on RDNA 4) does not take this frame (" + why +
         "); the context does not move to the HIP executor; such frames go back to the front end "
         "(AMD FSR, else its stand-in)";
}

// After choose_executor gave dx12: the D3D12 executor did not take the context at create (`err`: the driver-channel probe,
// device preparation, the network's create on the D3D12 device, a context its kernels do not take). Every model has the
// same answer (R1: no HIP route in a game): the context is not created, this is the one line, and the front end's own
// upscaler (AMD FSR, else its stand-in) keeps the game.
inline std::string dx12_refused_line(bool by_default, const std::string& err) {
  return std::string(by_default ? "executor dx12 (default on RDNA 4)" : "executor=dx12") +
         " could not take this context (" + err + "); there is no HIP route in a game (executor=hip selects the HIP "
         "executor as a diagnostic): the front end's own upscaler (AMD FSR) keeps the game";
}

}  // namespace wgi
