// SPDX-License-Identifier: MIT
// Executor choice (interop/executor_select.h): the default rule and the forcing options, on any host (no GPU, no
// D3D12). WrongGPU's own code.
//   clang++ -std=c++17 test_executor_select.cpp -o test_executor_select.exe && test_executor_select.exe
// Exit status 0 when every case passed. The interop test (test_interop_d3d12 dx12same with WG_TEST_DX12_DEFAULT=1)
// checks the same rule through the runtime DLL on the GPU.
#include <cstdio>
#include <string>

#include "../executor_select.h"

namespace {

int failures = 0;

void expect(const char* name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name);
  if (!ok) ++failures;
}

wgi::ExecutorRequest gpu(const char* adapter, uint32_t vendor = 0x1002, bool software = false) {
  wgi::ExecutorRequest r;
  r.adapter_known = true;
  r.adapter = adapter;
  r.vendor_id = vendor;
  r.software = software;
  return r;
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

}  // namespace

int main() {
  using wgi::choose_executor;
  using wgi::ExecutorChoice;
  using wgi::ExecutorRequest;

  // RDNA 4 adapter recognition (the native device's rule, ../rdna4.h)
  expect("rdna4: RX 9070 XT", wgrt::rdna4_adapter(0x1002, false, "AMD Radeon RX 9070 XT"));
  expect("rdna4: RX 9070", wgrt::rdna4_adapter(0x1002, false, "AMD Radeon RX 9070"));
  expect("rdna4: RX 9070 GRE", wgrt::rdna4_adapter(0x1002, false, "AMD Radeon RX 9070 GRE"));
  expect("rdna4: RX 9060 XT", wgrt::rdna4_adapter(0x1002, false, "AMD Radeon RX 9060 XT"));
  expect("not rdna4: RX 7900 XTX", !wgrt::rdna4_adapter(0x1002, false, "AMD Radeon RX 7900 XTX"));
  expect("not rdna4: RX 6800", !wgrt::rdna4_adapter(0x1002, false, "AMD Radeon RX 6800"));
  expect("not rdna4: integrated Radeon", !wgrt::rdna4_adapter(0x1002, false, "AMD Radeon(TM) Graphics"));
  expect("not rdna4: other vendor with the name", !wgrt::rdna4_adapter(0x10DE, false, "RX 9070"));
  expect("not rdna4: software adapter", !wgrt::rdna4_adapter(0x1002, true, "AMD Radeon RX 9070 XT"));
  expect("not rdna4: WARP", !wgrt::rdna4_adapter(0x1414, true, "Microsoft Basic Render Driver"));

  // default (option not named): RDNA 4 + model M + supported context -> dx12, by the default rule, nothing logged yet
  {
    ExecutorChoice c = choose_executor(gpu("AMD Radeon RX 9070 XT"));
    expect("default: RDNA 4 -> dx12", c.executor == "dx12" && c.by_default && c.why.empty());
    ExecutorRequest r = gpu("AMD Radeon RX 9060 XT");
    r.executor = "auto";
    c = choose_executor(r);
    expect("executor=auto: RDNA 4 (9060) -> dx12", c.executor == "dx12" && c.by_default);
    r.executor = "";
    expect("executor= (empty): as not named", choose_executor(r).executor == "dx12");
  }
  // R1 (oversight 2026-11-15): auto on an adapter that is not an RDNA 4 Radeon RX 9000-series card goes straight to the
  // front end's own upscaler (AMD FSR, else its stand-in), for every model, with one warning line: no executor, no HIP
  // attempt, no kernel load
  {
    const char* const sentence = "WrongGPU's models need an RX 9000-series GPU; using AMD FSR";
    ExecutorChoice c = choose_executor(gpu("AMD Radeon RX 7900 XTX"));
    expect("auto: RDNA 3 -> the front end's upscaler (fsr), not hip", c.executor == "fsr" && c.by_default && c.warn);
    expect("auto: RDNA 3 line is the sentence and names the adapter",
           c.why.rfind(sentence, 0) == 0 && has(c.why, "RX 7900 XTX") && c.why.find('\n') == std::string::npos);
    c = choose_executor(gpu("AMD Radeon RX 6800"));
    expect("auto: RDNA 2 -> fsr", c.executor == "fsr" && has(c.why, sentence));
    c = choose_executor(gpu("Other Vendor Graphics 4070", 0x10DE));
    expect("auto: another vendor -> fsr", c.executor == "fsr" && has(c.why, sentence));
    c = choose_executor(gpu("AMD Radeon(TM) Graphics"));
    expect("auto: an integrated Radeon -> fsr", c.executor == "fsr" && has(c.why, sentence));
    c = choose_executor(gpu("Microsoft Basic Render Driver", 0x1414, true));
    expect("auto: WARP -> fsr", c.executor == "fsr" && has(c.why, sentence));
    c = choose_executor(gpu("AMD Radeon RX 9070 XT", 0x1002, true));
    expect("auto: software adapter named RX 9070 -> fsr", c.executor == "fsr");
    ExecutorRequest r;  // adapter not identified (DXGI failed)
    c = choose_executor(r);
    expect("auto: unidentified adapter -> fsr with the same sentence",
           c.executor == "fsr" && c.warn && c.why.rfind(sentence, 0) == 0 && has(c.why, "adapter not identified"));
    r.model_k = true;
    c = choose_executor(r);
    expect("auto: unidentified adapter + model K -> fsr", c.executor == "fsr" && has(c.why, sentence));
    r = gpu("AMD Radeon RX 7900 XTX");
    r.sparkle_guard = true;
    r.capture = true;
    c = choose_executor(r);
    expect("auto: RDNA 3 with the sparkle guard or the capture -> fsr too (the kernels are RX 9000-only)",
           c.executor == "fsr");
    r = gpu("AMD Radeon RX 7900 XTX");
    r.executor = "";
    expect("executor= (empty) on RDNA 3: as auto", choose_executor(r).executor == "fsr");
    expect("the line helper: sentence first, then the adapter",
           wgi::needs_rx9000_line(true, "AMD Radeon RX 7900 XTX") ==
               "WrongGPU's models need an RX 9000-series GPU; using AMD FSR (adapter AMD Radeon RX 7900 XTX)");
    expect("the line helper: unidentified",
           wgi::needs_rx9000_line(false, "") ==
               "WrongGPU's models need an RX 9000-series GPU; using AMD FSR (adapter not identified)");
  }
  // RDNA 4: auto is unchanged for every model (the selector cannot see L: M and L share the not-K request)
  {
    for (const char* name : {"AMD Radeon RX 9070 XT", "AMD Radeon RX 9070", "AMD Radeon RX 9070 GRE", "AMD Radeon RX 9060 XT"}) {
      ExecutorRequest r = gpu(name);
      ExecutorChoice c = choose_executor(r);
      expect((std::string("RDNA 4 auto, model M or L -> dx12: ") + name).c_str(),
             c.executor == "dx12" && c.by_default && c.why.empty() && !c.warn);
      r.model_k = true;
      c = choose_executor(r);
      expect((std::string("RDNA 4 auto, model K -> dx12: ") + name).c_str(),
             c.executor == "dx12" && c.by_default && c.why.empty() && !c.warn);
    }
  }
  // default with model K on RDNA 4 -> dx12 by the default rule (the D3D12 executor's create checks K's ports and
  // refuses the context when one is missing: dx12_refused_line, then the HIP executor); on other GPUs -> hip
  {
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    r.model_k = true;
    ExecutorChoice c = choose_executor(r);
    expect("default: model K on RDNA 4 -> dx12 (by default)", c.executor == "dx12" && c.by_default && c.why.empty());
    r = gpu("AMD Radeon RX 7900 XTX");
    r.model_k = true;
    c = choose_executor(r);
    expect("default: model K on RDNA 3 -> the front end's upscaler", c.executor == "fsr" && has(c.why, "RX 9000-series"));
    r = gpu("AMD Radeon RX 9070 XT");
    r.model_k = true;
    r.capture = true;
    c = choose_executor(r);
    expect("default: model K + frame capture -> hip", c.executor == "hip" && has(c.why, "frame capture"));
  }
  // default with what has no D3D12 port -> hip, one line with the reason
  {
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    ExecutorChoice c;
    r.sparkle_guard = true;
    c = choose_executor(r);
    expect("default: sparkle guard -> hip, with a warning (model M, L: an opt-in HIP feature)",
           c.executor == "hip" && c.warn && has(c.why, "sparkle guard") && has(c.why, "diagnostic"));
    r = gpu("AMD Radeon RX 9070 XT");
    r.capture = true;
    c = choose_executor(r);
    expect("default: frame capture -> hip, with a warning (an opt-in diagnostic)",
           c.executor == "hip" && c.warn && has(c.why, "frame capture") && has(c.why, "diagnostic"));
  }
  // forced: executor=hip on RDNA 4 -> hip, nothing logged, no switch rule
  {
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    r.executor = "hip";
    ExecutorChoice c = choose_executor(r);
    expect("executor=hip on RDNA 4 -> hip, as a loud diagnostic",
           c.executor == "hip" && !c.by_default && c.warn && has(c.why, "diagnostic: executor=hip"));
    r.model_k = true;
    c = choose_executor(r);
    expect("executor=hip + model K -> hip, the same diagnostic line (never a game route)",
           c.executor == "hip" && !c.by_default && c.warn && has(c.why, "not the in-game runtime"));
  }
  // explicit executor=hip stays a loud diagnostic on any adapter (the front end's HIP attempt is the user's own request)
  {
    ExecutorRequest r = gpu("AMD Radeon RX 7900 XTX");
    r.executor = "hip";
    ExecutorChoice c = choose_executor(r);
    expect("executor=hip on RDNA 3 -> hip, the loud diagnostic line",
           c.executor == "hip" && !c.by_default && c.warn && has(c.why, "diagnostic: executor=hip"));
  }
  // forced: executor=dx12 (or d3d12) -> dx12 on any adapter (the device refuses unsupported ones), not by default
  {
    ExecutorRequest r = gpu("AMD Radeon RX 7900 XTX");
    r.executor = "dx12";
    ExecutorChoice c = choose_executor(r);
    expect("executor=dx12 on RDNA 3 -> dx12 (forced)", c.executor == "dx12" && !c.by_default && c.why.empty());
    r = gpu("AMD Radeon RX 9070 XT");
    r.executor = "dx12";
    c = choose_executor(r);
    expect("executor=dx12 on RDNA 4 -> dx12, not by default (no switch to HIP)", c.executor == "dx12" && !c.by_default);
    r.executor = "d3d12";
    expect("executor=d3d12 = dx12", choose_executor(r).executor == "dx12");
    r.executor = "dx12";
    r.model_k = true;
    c = choose_executor(r);
    expect("executor=dx12 + model K -> dx12 (forced; ports checked at the executor's create)",
           c.executor == "dx12" && !c.by_default && c.why.empty());
    {
      const std::string e = "D3D12 runtime: model K: the D3D12 build has no port of k_out_color_direct";
      expect("model K port missing (default): the refusal line names it",
             has(wgi::dx12_refused_line_k(true, e), "no port of k_out_color_direct") &&
                 has(wgi::dx12_refused_line_k(true, e), "model K has no HIP route") &&
                 has(wgi::dx12_refused_line_k(true, e), "the game keeps its own upscaler") &&
                 !has(wgi::dx12_refused_line_k(true, e), "the HIP executor runs it") &&
                 wgi::dx12_refused_line_k(true, e).find('\n') == std::string::npos);
    }
    r.model_k = false;
    r.sparkle_guard = true;
    c = choose_executor(r);
    expect("executor=dx12 + sparkle guard -> hip with the earlier warning",
           c.executor == "hip" && c.warn &&
               c.why == "executor=dx12 not available for this context: the sparkle guard runs on HIP only; the HIP "
                        "executor runs it");
    r.sparkle_guard = false;
    r.capture = true;
    c = choose_executor(r);
    expect("executor=dx12 + capture -> hip", c.executor == "hip" && c.warn && has(c.why, "frame capture"));
  }
  // the D3D12 executor refuses the context at create (driver-channel probe failure, ...): the HIP executor, one line
  {
    const std::string e = "D3D12 device: native D3D12 borrowed driver-channel probe mismatch at float 7";
    const std::string d = wgi::dx12_refused_line(true, e), f = wgi::dx12_refused_line(false, e);
    expect("probe failure (default): one line naming the default, the reason and the HIP executor",
           has(d, "executor dx12 (default on RDNA 4) could not take this context") && has(d, "probe mismatch") &&
               has(d, "the HIP executor runs it"));
    expect("probe failure (executor=dx12): the earlier line",
           f == "executor=dx12 could not take this context (" + e + "); the HIP executor runs it");
  }
  // clean/fe-wrapped-list: hip_switch (what the default D3D12 executor does with a frame it does not take mid-session)
  {
    using wgi::refused_frame_policy;
    expect("hip_switch not named -> the default: off (owner 2026-11-04, no mid-session HIP)",
           !wgi::kHipSwitchDefault && !refused_frame_policy("").switch_to_hip && refused_frame_policy("").note.empty());
    expect("hip_switch=on -> switch", refused_frame_policy("on").switch_to_hip && refused_frame_policy("1").switch_to_hip);
    expect("hip_switch=off -> no switch", !refused_frame_policy("off").switch_to_hip && !refused_frame_policy("0").switch_to_hip &&
                                              refused_frame_policy("off").note.empty());
    const wgi::RefusedFramePolicy odd = refused_frame_policy("maybe");
    expect("hip_switch=maybe -> the default with a note",
           odd.switch_to_hip == wgi::kHipSwitchDefault && has(odd.note, "hip_switch=maybe not recognised"));
    const std::string l = wgi::no_switch_line(2, "frame not taken (nothing recorded): native direct frame geometry");
    expect("the hip_switch=off line names the frame, the reason, no HIP and the front end's fallback",
           has(l, "frame 2: executor dx12 (default on RDNA 4) does not take this frame") && has(l, "native direct frame") &&
               has(l, "does not move to the HIP executor") && has(l, "AMD FSR, else its stand-in"));
  }
  // REVIEW-215b P1-a: model K's executor selection
  {
    using wgi::RefusedAction;
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    r.model_k = true;
    ExecutorChoice c = choose_executor(r);
    expect("K auto on RDNA 4 -> dx12 (by default), nothing to log", c.executor == "dx12" && c.by_default && c.why.empty());
    r.sparkle_guard = true;
    c = choose_executor(r);
    expect("K auto + sparkle guard -> dx12 with the guard turned off, one line",
           c.executor == "dx12" && c.by_default && c.sparkle_off && c.warn && has(c.why, "turned off for model K") &&
               has(c.why, "runs on the D3D12 executor"));
    r.executor = "dx12";
    c = choose_executor(r);
    expect("K executor=dx12 + sparkle guard -> dx12, guard off", c.executor == "dx12" && c.sparkle_off && c.warn);
    r = gpu("AMD Radeon RX 9070 XT");
    r.model_k = true;
    r.sparkle_guard = false;
    r.capture = true;
    c = choose_executor(r);
    expect("K auto + capture -> hip with the loud diagnostic line",
           c.executor == "hip" && c.warn && c.why == "diagnostic: frame capture forces the HIP executor (model K runs on the "
                                                    "D3D12 executor without it)");
    r.executor = "dx12";
    c = choose_executor(r);
    expect("K executor=dx12 + capture -> hip, the same diagnostic line",
           c.executor == "hip" && c.warn && has(c.why, "diagnostic: frame capture forces the HIP executor"));
    r.capture = false;
    r.executor = "auto";
    r.sparkle_guard = true;
    r.model_k = false;
    c = choose_executor(r);
    expect("model M + sparkle guard -> hip as before (the guard rule is K's exception only)",
           c.executor == "hip" && !c.sparkle_off && has(c.why, "sparkle guard"));
    r = gpu("AMD Radeon RX 7900 XTX");
    r.model_k = true;
    c = choose_executor(r);
    expect("K auto on a non-RDNA 4 adapter: the front end's upscaler, one line",
           c.executor == "fsr" && has(c.why, "RX 9000-series GPU; using AMD FSR"));
    r = wgi::ExecutorRequest();
    r.model_k = true;
    c = choose_executor(r);
    expect("K auto on an unidentified adapter: the front end's upscaler, one line",
           c.executor == "fsr" && has(c.why, "not identified"));

    // hip_switch: ignored for K, one line when it is set
    expect("K: no line when hip_switch is not set", wgi::k_hip_switch_note("").empty());
    expect("K: hip_switch=on is ignored with one line",
           has(wgi::k_hip_switch_note("on"), "model K ignores hip_switch=on") &&
               has(wgi::k_hip_switch_note("on"), "never to HIP"));

    // REVIEW-215b P1-b: a frame the D3D12 executor did not take. The three paths of the review and the messages the
    // executor gives them: no output texture, a render size outside the configured maximum, an unsupported frame (an
    // output texture without unordered access); each with hip_switch off and on, by default and forced.
    const char* const refusals[] = {
        "frame not taken (nothing recorded): no output texture",
        "frame not taken (nothing recorded): model K direct frame: render size outside the configured maximum",
        "frame not taken: output texture has no unordered access",
    };
    for (const char* why : refusals) {
      const std::string w = why;
      expect(("refused frame recognised: " + w).c_str(), wgi::frame_refused(w));
      for (bool by_default : {true, false}) {
        for (bool hip_switch : {false, true}) {
          const std::string tag = w.substr(0, 40) + (by_default ? " default" : " forced") + (hip_switch ? " switch=on" : " switch=off");
          // no counter, no state: every refused K frame is answered the same way (WG_ERROR_UNSUPPORTED), whatever came before
          expect(("K refused frame -> WG_ERROR_UNSUPPORTED, the front end's upscaler: " + tag).c_str(),
                 wgi::refused_frame_action(true, by_default, hip_switch) == RefusedAction::kNoSwitch);
        }
      }
    }
    expect("a frame that is not a refusal is not a 'frame not taken'",
           !wgi::frame_refused("input refused: texture format") && !wgi::frame_refused("model K: kernel failed"));
    // the other models keep their rules
    expect("M default + hip_switch=on -> switch to HIP (the 0.6.0 option)",
           wgi::refused_frame_action(false, true, true) == RefusedAction::kSwitchToHip);
    expect("M default + hip_switch=off -> front end's upscaler",
           wgi::refused_frame_action(false, true, false) == RefusedAction::kNoSwitch);
    expect("M executor=dx12 -> bilinear fallback", wgi::refused_frame_action(false, false, true) == RefusedAction::kBilinear);
    const std::string rl = wgi::k_refusal_line(7, "frame not taken (nothing recorded): no output texture");
    expect("the refusal line names the frame, the reason, the front end's upscaler and 'never to the HIP executor'",
           has(rl, "frame 7: model K") && has(rl, "no output texture") && has(rl, "AMD FSR, else its stand-in") &&
               has(rl, "never to the HIP executor") && rl.find('\n') == std::string::npos);
    expect("K and M (hip_switch=off) take the same path: the same action",
           wgi::refused_frame_action(true, true, true) == wgi::refused_frame_action(false, true, false));
  }
  // R1 (oversight 2026-11-15): the GPU's identity comes from the system's DXGI and the kernel-mode driver's ids, never from
  // a description a layer in the game (OptiScaler's GPU spoofing) presents. A real RX 9000 keeps WrongGPU whatever name it
  // is shown under; another board, or no identity at all, goes to the front end's upscaler with the one line.
  {
    const char* const sentence = "WrongGPU's models need an RX 9000-series GPU; using AMD FSR";
    for (uint32_t id : {0x7550u, 0x7551u, 0x7590u, 0x7591u})
      expect("RDNA 4 device id", wgrt::rdna4_device_id(id));
    for (uint32_t id : {0x744Cu, 0x73BFu, 0x7480u, 0x164Eu, 0x13C0u, 0u})
      expect("not an RDNA 4 device id", !wgrt::rdna4_device_id(id));
    auto spoof = [](const char* shown, uint32_t shown_vendor, uint32_t kvendor, uint32_t kdevice) {
      ExecutorRequest r = gpu(shown, shown_vendor);
      r.kmt_known = true;
      r.kmt_vendor = kvendor;
      r.kmt_device = kdevice;
      return r;
    };
    // a SPOOFED name on a real RX 9000 id: still the D3D12 executor, WrongGPU active, nothing to log (every model)
    ExecutorRequest r = spoof("Other Vendor Graphics 4090", 0x10DE, 0x1002, 0x7550);
    ExecutorChoice c = choose_executor(r);
    expect("spoofed other-vendor name on a real RX 9070 id -> dx12", c.executor == "dx12" && c.by_default && c.why.empty());
    r.model_k = true;
    expect("the same, model K -> dx12", choose_executor(r).executor == "dx12");
    r = spoof("AMD Radeon RX 7900 XTX", 0x1002, 0x1002, 0x7590);
    expect("spoofed RDNA 3 name on a real RX 9060 id -> dx12", choose_executor(r).executor == "dx12");
    r = spoof("Microsoft Basic Render Driver", 0x1414, 0x1002, 0x7551);
    r.software = true;
    expect("a software-looking description on a real RDNA 4 id -> dx12 (the driver's id decides)", choose_executor(r).executor == "dx12");
    // the driver's ids name the board; the description reported by the system's DXGI agrees
    r = spoof("AMD Radeon RX 9070 XT", 0x1002, 0x1002, 0x7550);
    expect("real RX 9070 XT: dx12", choose_executor(r).executor == "dx12");
    // a real non-RDNA-4 id: the front end's upscaler with the one line
    r = spoof("AMD Radeon RX 7900 XTX", 0x1002, 0x1002, 0x744C);
    c = choose_executor(r);
    expect("real RDNA 3 id -> fsr with the line", c.executor == "fsr" && c.warn && c.why.rfind(sentence, 0) == 0);
    r = spoof("Other Vendor Graphics 4070", 0x10DE, 0x10DE, 0x2786);
    expect("real other-vendor id -> fsr", choose_executor(r).executor == "fsr");
    // an unidentified adapter: no description, no driver ids
    c = choose_executor(ExecutorRequest());
    expect("nothing known -> fsr with the line and 'not identified'",
           c.executor == "fsr" && c.why.rfind(sentence, 0) == 0 && has(c.why, "adapter not identified"));
    // only the driver's ids known (the system DXGI could not be read)
    r = ExecutorRequest();
    r.kmt_known = true;
    r.kmt_vendor = 0x1002;
    r.kmt_device = 0x744C;
    c = choose_executor(r);
    expect("only a real RDNA 3 id known -> fsr, the line names the ids", c.executor == "fsr" && has(c.why, "PCI 1002:744C"));
    r.kmt_device = 0x7550;
    expect("only a real RX 9070 id known -> dx12", choose_executor(r).executor == "dx12");
    // the pure rule
    wgrt::AdapterIdentity id;
    expect("identity: nothing known -> not RDNA 4", !wgrt::rdna4_identity(id));
    id.dxgi_known = true;
    id.dxgi_vendor = 0x1002;
    id.description = "AMD Radeon RX 9060 XT";
    expect("identity: the system DXGI's description alone (no driver ids) -> RDNA 4", wgrt::rdna4_identity(id));
    id.software = true;
    expect("identity: a software adapter is not RDNA 4", !wgrt::rdna4_identity(id));
    // forced executors are not affected by the identity (a forced dx12 on a real RX 9000 under a spoof stays dx12)
    r = spoof("Other Vendor Graphics 4090", 0x10DE, 0x1002, 0x7550);
    r.executor = "dx12";
    expect("executor=dx12 under a spoof on a real RX 9000 -> dx12", choose_executor(r).executor == "dx12" && !choose_executor(r).by_default);
    r.executor = "hip";
    expect("executor=hip under a spoof -> the loud diagnostic", choose_executor(r).executor == "hip" && choose_executor(r).warn);
  }
  // test executor and unknown values
  {
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    r.executor = "d3d12-test";
    expect("executor=d3d12-test", choose_executor(r).executor == "d3d12-test");
    r.executor = "vulkan";
    expect("unknown executor -> empty (create fails)", choose_executor(r).executor.empty());
  }
  std::printf(failures ? "%d FAILED\n" : "ALL TESTS PASSED\n", failures);
  return failures ? 1 : 0;
}
