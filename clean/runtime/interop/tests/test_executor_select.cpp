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
  // default on other GPUs -> hip, one information line naming the adapter
  {
    ExecutorChoice c = choose_executor(gpu("AMD Radeon RX 7900 XTX"));
    expect("default: RDNA 3 -> hip", c.executor == "hip" && c.by_default && !c.warn);
    expect("default: RDNA 3 line names the adapter and the reason",
           has(c.why, "executor hip (default)") && has(c.why, "RX 7900 XTX") && has(c.why, "not an RDNA 4"));
    c = choose_executor(gpu("Microsoft Basic Render Driver", 0x1414, true));
    expect("default: WARP -> hip", c.executor == "hip" && !c.why.empty());
    c = choose_executor(gpu("AMD Radeon RX 9070 XT", 0x1002, true));
    expect("default: software adapter named RX 9070 -> hip", c.executor == "hip");
    ExecutorRequest r;  // adapter not identified (DXGI failed)
    c = choose_executor(r);
    expect("default: unknown adapter -> hip", c.executor == "hip" && has(c.why, "not identified"));
  }
  // default with what has no D3D12 port -> hip, one line with the reason
  {
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    r.model_k = true;
    ExecutorChoice c = choose_executor(r);
    expect("default: model K -> hip", c.executor == "hip" && has(c.why, "model K"));
    r = gpu("AMD Radeon RX 9070 XT");
    r.sparkle_guard = true;
    c = choose_executor(r);
    expect("default: sparkle guard -> hip", c.executor == "hip" && has(c.why, "sparkle guard"));
    r = gpu("AMD Radeon RX 9070 XT");
    r.capture = true;
    c = choose_executor(r);
    expect("default: frame capture -> hip", c.executor == "hip" && has(c.why, "frame capture"));
  }
  // forced: executor=hip on RDNA 4 -> hip, nothing logged, no switch rule
  {
    ExecutorRequest r = gpu("AMD Radeon RX 9070 XT");
    r.executor = "hip";
    ExecutorChoice c = choose_executor(r);
    expect("executor=hip on RDNA 4 -> hip", c.executor == "hip" && !c.by_default && c.why.empty());
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
    expect("executor=dx12 + model K -> hip with the earlier warning",
           c.executor == "hip" && c.warn && c.why == "executor=dx12: model K has no D3D12 port; the HIP executor runs it");
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
