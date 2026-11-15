// SPDX-License-Identifier: MIT
// Executor choice (interop/executor_select.h) and the GPU identity rule it stands on (rdna4.h): the default rule, the
// forcing options and the refusal paths, on any host (no GPU, no D3D12). WrongGPU's own code.
//   clang++ -std=c++17 test_executor_select.cpp -o test_executor_select.exe && test_executor_select.exe
// Exit status 0 when every case passed. clean/tests/test_executor_select.py builds and runs it in the Python tests (CI).
// The interop test (test_interop_d3d12 dx12same with WG_TEST_DX12_DEFAULT=1) checks the same rule through the runtime DLL on
// the GPU.
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "../executor_select.h"

namespace {

int failures = 0;

void expect(const char* name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name);
  if (!ok) ++failures;
}
void expect(const std::string& name, bool ok) { expect(name.c_str(), ok); }

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// A request for an adapter as the lookup found it (what backend.cpp builds from wgrt::lookup_adapter)
wgi::ExecutorRequest request_for(const wgrt::AdapterIdentity& id) {
  wgi::ExecutorRequest r;
  r.adapter_known = id.dxgi_known;
  r.vendor_id = id.dxgi_vendor;
  r.software = id.software;
  r.adapter = id.description;
  r.kmt_known = id.kmt_known;
  r.kmt_vendor = id.kmt_vendor;
  r.kmt_device = id.kmt_device;
  return r;
}

// An adapter: the description the system DXGI gives (shown vendor / name) and the driver's ids
wgrt::AdapterIdentity adapter(const char* shown_name, uint32_t shown_vendor, uint32_t kvendor, uint32_t kdevice,
                              bool software = false) {
  wgrt::AdapterIdentity id;
  id.dxgi_known = true;
  id.dxgi_vendor = shown_vendor;
  id.software = software;
  id.description = shown_name;
  id.kmt_known = true;
  id.kmt_vendor = kvendor;
  id.kmt_device = kdevice;
  return id;
}

// A real board: its own name and ids
wgi::ExecutorRequest board(const char* name, uint32_t vendor, uint32_t device, bool software = false) {
  return request_for(adapter(name, vendor, vendor, device, software));
}

struct RealBoard {
  const char* name;
  uint32_t vendor, device;
};
const RealBoard kRdna4[] = {{"AMD Radeon RX 9070 XT", 0x1002, 0x7550},
                            {"AMD Radeon RX 9070 GRE", 0x1002, 0x7551},
                            {"AMD Radeon RX 9060 XT", 0x1002, 0x7590},
                            {"AMD Radeon RX 9060", 0x1002, 0x7591}};
const RealBoard kOthers[] = {{"AMD Radeon RX 7900 XTX", 0x1002, 0x744C},
                             {"AMD Radeon RX 6800", 0x1002, 0x73BF},
                             {"AMD Radeon(TM) Graphics", 0x1002, 0x15BF},
                             {"Other Vendor Graphics 4070", 0x10DE, 0x2786}};

}  // namespace

int main() {
  using wgi::choose_executor;
  using wgi::ExecutorChoice;
  using wgi::ExecutorRequest;
  const char* const sentence = "WrongGPU's models need an RX 9000-series GPU; using AMD FSR";

  // ---- the identity rule (rdna4.h): the driver's vendor and device id, nothing else, failing closed -------------------
  for (const RealBoard& b : kRdna4) expect(std::string("RDNA 4 device id: ") + b.name, wgrt::rdna4_device_id(b.device));
  for (const RealBoard& b : kOthers)
    expect(std::string("not an RDNA 4 board: ") + b.name,
           !wgrt::rdna4_identity(adapter(b.name, b.vendor, b.vendor, b.device)));
  for (const RealBoard& b : kRdna4)
    expect(std::string("RDNA 4 identity: ") + b.name, wgrt::rdna4_identity(adapter(b.name, b.vendor, b.vendor, b.device)));
  {
    expect("a spoofed other-vendor name on a real RX 9070 id is RDNA 4",
           wgrt::rdna4_identity(adapter("Other Vendor Graphics 4090", 0x10DE, 0x1002, 0x7550)));
    expect("a spoofed software-looking description on a real RDNA 4 id is RDNA 4",
           wgrt::rdna4_identity(adapter("Microsoft Basic Render Driver", 0x1414, 0x1002, 0x7551, true)));
    expect("a supported-looking name on a real RDNA 3 id is NOT RDNA 4 (the description never accepts)",
           !wgrt::rdna4_identity(adapter("AMD Radeon RX 9070 XT", 0x1002, 0x1002, 0x744C)));
    expect("a supported-looking name on a real other-vendor id is NOT RDNA 4",
           !wgrt::rdna4_identity(adapter("AMD Radeon RX 9070 XT", 0x1002, 0x10DE, 0x2786)));
    expect("an RDNA 4 device id under another vendor's id is NOT RDNA 4", !wgrt::rdna4_identity(adapter("x", 0x1002, 0x10DE, 0x7550)));
    wgrt::AdapterIdentity id = adapter("AMD Radeon RX 9070 XT", 0x1002, 0x1002, 0x7550);
    id.kmt_known = false;
    expect("the driver's ids unreadable: NOT RDNA 4 even with an RX 9070 description (fail closed)", !wgrt::rdna4_identity(id));
    id.dxgi_known = false;
    expect("nothing known: NOT RDNA 4", !wgrt::rdna4_identity(id) && !wgrt::rdna4_identity(wgrt::AdapterIdentity()));
    expect("the unknown-id AMD board is named in the line",
           has(wgrt::describe_identity(adapter("AMD Radeon RX 9070 XT", 0x1002, 0x1002, 0x7592)), "outside the RDNA 4 id list") &&
               has(wgrt::describe_identity(adapter("x", 0x1002, 0x1002, 0x7592)), "PCI 1002:7592"));
    expect("describe: unidentified", wgrt::describe_identity(wgrt::AdapterIdentity()) == "adapter not identified");
    expect("describe: description and ids",
           wgrt::describe_identity(adapter("AMD Radeon RX 7900 XTX", 0x1002, 0x1002, 0x744C)) ==
               "adapter AMD Radeon RX 7900 XTX, PCI 1002:744C; an AMD board outside the RDNA 4 id list");
  }

  // ---- the lookup with injectable sources: the game device's LUID selects the adapter -----------------------------
  {
    struct Fake {
      std::map<std::pair<uint32_t, int32_t>, wgrt::AdapterRecord> dxgi;
      std::map<std::pair<uint32_t, int32_t>, std::pair<uint32_t, uint32_t>> ids;
      std::vector<std::pair<uint32_t, int32_t>> asked_dxgi, asked_driver;
      bool dxgi_fails = false, driver_fails = false;
      wgrt::AdapterSources sources() {
        wgrt::AdapterSources s;
        s.find_adapter = [this](const wgrt::AdapterLuid& l, wgrt::AdapterRecord* out) {
          asked_dxgi.push_back({l.low, l.high});
          auto it = dxgi.find({l.low, l.high});
          if (dxgi_fails || it == dxgi.end()) return false;
          *out = it->second;
          return true;
        };
        s.driver_ids = [this](const wgrt::AdapterLuid& l, uint32_t* v, uint32_t* d) {
          asked_driver.push_back({l.low, l.high});
          auto it = ids.find({l.low, l.high});
          if (driver_fails || it == ids.end()) return false;
          *v = it->second.first;
          *d = it->second.second;
          return true;
        };
        return s;
      }
    };
    auto record = [](uint32_t low, int32_t high, uint32_t vendor, bool software, const char* name) {
      wgrt::AdapterRecord r;
      r.luid.low = low;
      r.luid.high = high;
      r.vendor = vendor;
      r.software = software;
      r.description = name;
      return r;
    };
    Fake f;  // two adapters: WARP (LUID 1:0) and an RX 9070 XT (LUID 2:0) that a layer in the game shows as another vendor
    f.dxgi[{1, 0}] = record(1, 0, 0x1414, true, "Microsoft Basic Render Driver");
    f.dxgi[{2, 0}] = record(2, 0, 0x10DE, false, "Other Vendor Graphics 4090");
    f.ids[{1, 0}] = {0x1414, 0x008C};
    f.ids[{2, 0}] = {0x1002, 0x7550};
    wgrt::AdapterLuid warp, gpu2, nobody;
    warp.low = 1;
    gpu2.low = 2;
    nobody.low = 9;
    nobody.high = 3;
    wgrt::AdapterIdentity id = wgrt::lookup_adapter(gpu2, f.sources());
    expect("lookup: the game's LUID picks the RX 9000 among two adapters, with its own ids and description",
           id.kmt_known && id.kmt_device == 0x7550 && id.dxgi_known && id.description == "Other Vendor Graphics 4090" &&
               wgrt::rdna4_identity(id));
    id = wgrt::lookup_adapter(warp, f.sources());
    expect("lookup: the other LUID picks the software adapter, not RDNA 4",
           id.kmt_known && id.kmt_vendor == 0x1414 && id.software && !wgrt::rdna4_identity(id));
    expect("lookup: the same LUID went to both sources", f.asked_dxgi == f.asked_driver &&
                                                         f.asked_dxgi.size() == 2 && f.asked_dxgi[0].first == 2 &&
                                                         f.asked_dxgi[1].first == 1);
    id = wgrt::lookup_adapter(nobody, f.sources());
    expect("lookup: a LUID neither source knows is not identified", !id.dxgi_known && !id.kmt_known && !wgrt::rdna4_identity(id));
    f.driver_fails = true;
    id = wgrt::lookup_adapter(gpu2, f.sources());
    expect("lookup: the driver query failing leaves the board unidentified and refused (fail closed)",
           id.dxgi_known && !id.kmt_known && !wgrt::rdna4_identity(id));
    f.driver_fails = false;
    f.dxgi_fails = true;
    id = wgrt::lookup_adapter(gpu2, f.sources());
    expect("lookup: the system DXGI failing does not matter, the driver's ids decide",
           !id.dxgi_known && id.kmt_known && wgrt::rdna4_identity(id));
    f.dxgi_fails = false;
    f.ids[{2, 0}] = {0x1002, 0x744C};  // a conflicting pair: the description says RX 9070 XT, the ids say RDNA 3
    f.dxgi[{2, 0}] = record(2, 0, 0x1002, false, "AMD Radeon RX 9070 XT");
    id = wgrt::lookup_adapter(gpu2, f.sources());
    expect("lookup: a supported-looking description with RDNA 3 ids is refused", !wgrt::rdna4_identity(id));

    // the same cases through the executor choice and the one line (every model)
    f.ids[{2, 0}] = {0x1002, 0x7550};
    f.dxgi[{2, 0}] = record(2, 0, 0x10DE, false, "Other Vendor Graphics 4090");
    for (bool model_k : {false, true}) {
      const char* tag = model_k ? " (model K)" : " (model M or L)";
      ExecutorRequest r = request_for(wgrt::lookup_adapter(gpu2, f.sources()));
      r.model_k = model_k;
      ExecutorChoice c = choose_executor(r);
      expect(std::string("spoofed name on a real RX 9000 id: the D3D12 executor, nothing logged") + tag,
             c.executor == "dx12" && c.by_default && c.why.empty() && !c.warn);
      r = request_for(wgrt::lookup_adapter(warp, f.sources()));
      r.model_k = model_k;
      c = choose_executor(r);
      expect(std::string("software adapter: the front end's upscaler with the sentence and the ids") + tag,
             c.executor == "fsr" && c.warn && c.why.rfind(sentence, 0) == 0 && has(c.why, "PCI 1414:008C"));
      f.driver_fails = true;
      r = request_for(wgrt::lookup_adapter(gpu2, f.sources()));
      r.model_k = model_k;
      c = choose_executor(r);
      expect(std::string("driver ids unreadable: the front end's upscaler (fail closed), no ids in the line") + tag,
             c.executor == "fsr" && c.why.rfind(sentence, 0) == 0 && !has(c.why, "PCI "));
      f.driver_fails = false;
      f.dxgi_fails = f.driver_fails = true;
      r = request_for(wgrt::lookup_adapter(gpu2, f.sources()));
      r.model_k = model_k;
      c = choose_executor(r);
      expect(std::string("nothing readable: the front end's upscaler, 'adapter not identified'") + tag,
             c.executor == "fsr" && has(c.why, "adapter not identified") && c.why.rfind(sentence, 0) == 0);
      f.dxgi_fails = f.driver_fails = false;
    }
    // the Ray Reconstruction runtime uses the same lookup, rule and sentence (rr_runtime.cpp rdna4_luid)
    f.driver_fails = true;
    id = wgrt::lookup_adapter(gpu2, f.sources());
    expect("RR: ids unreadable -> refused before anything is created, with the shared sentence",
           !wgrt::rdna4_identity(id) && std::string(wgrt::needs_rx9000_sentence()) == sentence &&
               wgi::needs_rx9000_line(id).rfind(sentence, 0) == 0);
    f.driver_fails = false;
    id = wgrt::lookup_adapter(gpu2, f.sources());
    expect("RR: the game's RX 9000 (spoofed name) is accepted", wgrt::rdna4_identity(id));
  }

  // ---- the default rule on real boards, every model -----------------------------------------------------------------
  for (bool model_k : {false, true}) {
    const char* tag = model_k ? " (model K)" : " (model M or L)";
    for (const RealBoard& b : kRdna4) {
      ExecutorRequest r = board(b.name, b.vendor, b.device);
      r.model_k = model_k;
      ExecutorChoice c = choose_executor(r);
      expect(std::string("RDNA 4 auto -> dx12, by default, nothing logged: ") + b.name + tag,
             c.executor == "dx12" && c.by_default && c.why.empty() && !c.warn);
      r.executor = "auto";
      expect(std::string("executor=auto: ") + b.name + tag, choose_executor(r).executor == "dx12");
      r.executor = "";
      expect(std::string("executor= (empty): ") + b.name + tag, choose_executor(r).executor == "dx12");
    }
    for (const RealBoard& b : kOthers) {
      ExecutorRequest r = board(b.name, b.vendor, b.device);
      r.model_k = model_k;
      ExecutorChoice c = choose_executor(r);
      expect(std::string("other board auto -> the front end's upscaler with the sentence: ") + b.name + tag,
             c.executor == "fsr" && c.by_default && c.warn && c.why.rfind(sentence, 0) == 0 && has(c.why, b.name) &&
                 c.why.find('\n') == std::string::npos);
      r.sparkle_guard = r.capture = true;
      expect(std::string("... also with the sparkle guard or the capture set: ") + b.name + tag,
             choose_executor(r).executor == "fsr");
    }
    ExecutorRequest r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550, true);  // software flag with an RDNA 4 id: the id decides
    r.model_k = model_k;
    expect(std::string("a software flag does not decide, the ids do") + tag, choose_executor(r).executor == "dx12");
    r = ExecutorRequest();
    r.model_k = model_k;
    ExecutorChoice c = choose_executor(r);
    expect(std::string("an unidentified adapter -> the front end's upscaler") + tag,
           c.executor == "fsr" && c.warn && has(c.why, "adapter not identified"));
    expect(std::string("the supported-looking description with no ids -> the front end's upscaler") + tag, [&] {
      ExecutorRequest q;
      q.model_k = model_k;
      q.adapter_known = true;
      q.vendor_id = 0x1002;
      q.adapter = "AMD Radeon RX 9070 XT";
      return choose_executor(q).executor == "fsr";
    }());
    expect(std::string("the supported-looking description on RDNA 3 ids -> the front end's upscaler") + tag, [&] {
      ExecutorRequest q = request_for(adapter("AMD Radeon RX 9070 XT", 0x1002, 0x1002, 0x744C));
      q.model_k = model_k;
      return choose_executor(q).executor == "fsr";
    }());
  }

  // ---- diagnostics and the sparkle guard (every model, both configurations) ------------------------------------------
  // `dev` = a developer build (WG_DEV_DIAGNOSTICS: executor=hip and the frame capture work, as loud diagnostics); a release
  // build IGNORES both with one line each and the context stays on D3D12 (R1: nothing in a game starts HIP).
#ifdef WG_DEV_DIAGNOSTICS
  expect("this test binary is a developer build: kDevDiagnostics is on", wgi::kDevDiagnostics);
#else
  expect("this test binary is a release build: kDevDiagnostics is off", !wgi::kDevDiagnostics);
#endif
  for (bool model_k : {false, true})
    for (bool dev : {true, false}) {
      // (a release build has no developer code at all: dev=true then behaves as a release build; the developer routes are
      // asserted by the binary built with WG_DEV_DIAGNOSTICS, which test_executor_select.py also runs)
      const bool devEff = dev && wgi::kDevDiagnostics;
      const std::string tag = std::string(model_k ? " (model K)" : " (model M or L)") + (devEff ? " [dev]" : " [release]");
      auto choose = [&](const ExecutorRequest& q) { return choose_executor(q, dev); };
      auto only_ignored = [](const ExecutorChoice& c, const char* key) {
        return c.ignored.size() == 1 && has(c.ignored[0], key) && has(c.ignored[0], "ignored: diagnostic key, release build") &&
               c.ignored[0].find('\n') == std::string::npos;
      };
      ExecutorRequest r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
      r.model_k = model_k;
      r.executor = "hip";
      ExecutorChoice c = choose(r);
      if (devEff)
        expect("executor=hip: the HIP executor as a loud diagnostic" + tag,
               c.executor == "hip" && !c.by_default && c.warn && has(c.why, "diagnostic: executor=hip") &&
                   has(c.why, "not the in-game runtime") && c.ignored.empty());
      else
        expect("executor=hip is IGNORED: the D3D12 executor by default, one line, no HIP" + tag,
               c.executor == "dx12" && c.by_default && c.why.empty() && only_ignored(c, "executor=hip"));
      r = board("AMD Radeon RX 7900 XTX", 0x1002, 0x744C);
      r.model_k = model_k;
      r.executor = "hip";
      c = choose(r);
      if (devEff)
        expect("executor=hip on another board: the same diagnostic" + tag,
               c.executor == "hip" && c.warn && has(c.why, "diagnostic: executor=hip"));
      else
        expect("executor=hip on another board is ignored: no executor, the front end's upscaler (the card sentence)" + tag,
               c.executor == "fsr" && c.warn && c.why.rfind(sentence, 0) == 0 && only_ignored(c, "executor=hip"));
      r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
      r.model_k = model_k;
      r.capture = true;
      c = choose(r);
      if (devEff)
        expect("auto + frame capture: HIP with the loud diagnostic line" + tag,
               c.executor == "hip" && c.warn && has(c.why, "diagnostic: frame capture") && c.ignored.empty());
      else
        expect("auto + frame capture is IGNORED: the D3D12 executor, one line, no HIP" + tag,
               c.executor == "dx12" && c.by_default && c.why.empty() && only_ignored(c, "capture_dir"));
      r.executor = "dx12";
      c = choose(r);
      if (devEff)
        expect("executor=dx12 + frame capture: the same diagnostic (HIP)" + tag, c.executor == "hip" && has(c.why, "diagnostic: frame capture"));
      else
        expect("executor=dx12 + capture_dir is ignored: the D3D12 executor, one line" + tag,
               c.executor == "dx12" && !c.by_default && only_ignored(c, "capture_dir"));
      r.executor = "hip";
      c = choose(r);
      if (devEff)
        expect("executor=hip + capture (developer build): HIP" + tag, c.executor == "hip");
      else
        expect("executor=hip + capture_dir in a release build: both ignored, two lines, D3D12" + tag,
               c.executor == "dx12" && c.ignored.size() == 2 && has(c.ignored[0], "executor=hip") && has(c.ignored[1], "capture_dir"));
      r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
      r.model_k = model_k;
      r.sparkle_guard = true;
      c = choose(r);
      expect("auto + sparkle guard: the D3D12 executor, the guard turned off, one line (both configurations)" + tag,
             c.executor == "dx12" && c.by_default && c.sparkle_off && c.warn && c.why == wgi::sparkle_off_line() &&
                 has(c.why, "turned off") && has(c.why, "D3D12") && c.ignored.empty());
      r.executor = "dx12";
      c = choose(r);
      expect("executor=dx12 + sparkle guard: the same" + tag, c.executor == "dx12" && c.sparkle_off && !c.by_default);
      r.capture = true;
      c = choose(r);
      if (devEff)
        expect("the sparkle guard with the capture (developer build): the capture's diagnostic wins" + tag, c.executor == "hip");
      else
        expect("the sparkle guard with capture_dir (release): D3D12, the guard off, the key ignored" + tag,
               c.executor == "dx12" && c.sparkle_off && only_ignored(c, "capture_dir"));
      r = board("AMD Radeon RX 7900 XTX", 0x1002, 0x744C);
      r.model_k = model_k;
      r.executor = "dx12";
      c = choose(r);
      expect("executor=dx12 on another board: dx12 (the device refuses it at create)" + tag,
             c.executor == "dx12" && !c.by_default && c.why.empty() && c.ignored.empty());
      r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
      r.model_k = model_k;
      r.executor = "d3d12";
      expect("executor=d3d12 = dx12" + tag, choose(r).executor == "dx12");
      r.executor = "d3d12-test";
      expect("executor=d3d12-test" + tag, choose(r).executor == "d3d12-test");
      r.executor = "vulkan";
      expect("unknown executor -> empty (create fails)" + tag, choose(r).executor.empty());
      r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
      r.model_k = model_k;
      expect("a plain RDNA 4 auto context has nothing ignored and nothing to log" + tag,
             choose(r).executor == "dx12" && choose(r).ignored.empty() && choose(r).why.empty());
    }

  // ---- the D3D12 executor refuses the context at create: every model, the front end's upscaler, never HIP -------------
  {
    const std::string e = "D3D12 device: native D3D12 borrowed driver-channel probe mismatch at float 7";
    const std::string d = wgi::dx12_refused_line(true, e), f = wgi::dx12_refused_line(false, e);
    expect("create refusal (default): one line with the reason, no HIP route, the front end's upscaler",
           has(d, "executor dx12 (default on RDNA 4) could not take this context") && has(d, "probe mismatch") &&
               has(d, "there is no HIP route in a game") && has(d, "the front end's own upscaler (AMD FSR) keeps the game") &&
               !has(d, "the HIP executor runs it") && d.find('\n') == std::string::npos);
    expect("create refusal (executor=dx12): the same", has(f, "executor=dx12 could not take this context") && has(f, "no HIP route"));
    const std::string k = "D3D12 runtime: model K: the D3D12 build has no port of k_out_color_direct";
    expect("model K port missing: the same line names it", has(wgi::dx12_refused_line(true, k), "no port of k_out_color_direct"));
  }

  // ---- clean/fe-wrapped-list: hip_switch, and a frame the D3D12 executor does not take -----------------------------
  {
    expect("hip_switch not set, off, 0 or false -> no line", wgi::hip_switch_note("").empty() &&
                                                                 wgi::hip_switch_note("off").empty() &&
                                                                 wgi::hip_switch_note("0").empty() &&
                                                                 wgi::hip_switch_note("false").empty());
    for (const char* v : {"on", "1", "true", "maybe"})
      expect(std::string("hip_switch=") + v + " is ignored with one line, no HIP",
             has(wgi::hip_switch_note(v), std::string("hip_switch=") + v + " is ignored") &&
                 has(wgi::hip_switch_note(v), "never to HIP") && wgi::hip_switch_note(v).find('\n') == std::string::npos);
    const std::string l = wgi::no_switch_line(2, "frame not taken (nothing recorded): native direct frame geometry");
    expect("the refused-frame line names the frame, the reason, no HIP and the front end's fallback",
           has(l, "frame 2: executor dx12 (default on RDNA 4) does not take this frame") && has(l, "native direct frame") &&
               has(l, "does not move to the HIP executor") && has(l, "AMD FSR, else its stand-in"));
    const char* const refusals[] = {
        "frame not taken (nothing recorded): no output texture",
        "frame not taken (nothing recorded): model K direct frame: render size outside the configured maximum",
        "frame not taken: output texture has no unordered access",
    };
    for (const char* why : refusals) {
      const std::string w = why;
      expect("refused frame recognised: " + w, wgi::frame_refused(w));
      expect("a refused frame (every model, default and forced executor) goes back to the front end: " + w.substr(0, 40),
             wgi::kRefusedFrameGoesToFrontEnd);
    }
    expect("a frame that is not a refusal is not a 'frame not taken'",
           !wgi::frame_refused("input refused: texture format") && !wgi::frame_refused("model K: kernel failed"));
    expect("no bilinear or other substitute exists in the selector any more (a forced executor=dx12 included)",
           wgi::kRefusedFrameGoesToFrontEnd);
    const std::string rl = wgi::k_refusal_line(7, "frame not taken (nothing recorded): no output texture");
    expect("the refusal line names the frame, the reason, the front end's upscaler and 'never to the HIP executor'",
           has(rl, "frame 7: model K") && has(rl, "no output texture") && has(rl, "AMD FSR, else its stand-in") &&
               has(rl, "never to the HIP executor") && rl.find('\n') == std::string::npos);
  }

  // ---- H3 (R1): the HIP-only output filters give the default filter on the D3D12 runtime, never HIP --------------------
  {
    std::string note;
    expect("no output_filter: nothing changes, no line",
           wgi::drop_hip_only_filter("post_diet=0;k_input=16", &note) == "post_diet=0;k_input=16" && note.empty());
    expect("default / original / analytic stay as they are",
           wgi::drop_hip_only_filter("output_filter=default", &note) == "output_filter=default" &&
               wgi::drop_hip_only_filter("a=1;output_filter=original;b=2", &note) == "a=1;output_filter=original;b=2" &&
               wgi::drop_hip_only_filter("output_filter=analytic", &note) == "output_filter=analytic" && note.empty());
    for (const char* q : {"quality5", "quality9"}) {
      note.clear();
      const std::string out = wgi::drop_hip_only_filter(std::string("a=1;output_filter=") + q + ";b=2", &note);
      expect(std::string(q) + ": the item is dropped, the others stay, one line, no HIP",
             out == "a=1;b=2" && has(note, std::string("output_filter=") + q + " is not available on the D3D12 runtime yet") &&
                 has(note, "default output filter") && note.find('\n') == std::string::npos);
    }
    note.clear();
    expect("spaces around the key and value are read as the runtime reads them",
           wgi::drop_hip_only_filter("a=1; output_filter = quality9 ;b=2", &note) == "a=1;b=2" && !note.empty());
    note.clear();
    expect("the last output_filter wins: quality5 then analytic stays analytic",
           wgi::drop_hip_only_filter("output_filter=quality5;output_filter=analytic", &note) ==
               "output_filter=quality5;output_filter=analytic" && note.empty());
    note.clear();
    expect("the last output_filter wins: analytic then quality9 gives the default filter",
           wgi::drop_hip_only_filter("output_filter=analytic;x=1;output_filter=quality9", &note) == "x=1" && !note.empty());
    note.clear();
    expect("only the option's name is read: a value that merely contains it is not touched",
           wgi::drop_hip_only_filter("model_dir=C:\\output_filter=quality5", &note) == "model_dir=C:\\output_filter=quality5" &&
               note.empty());
    expect("an empty option string gives an empty one", wgi::drop_hip_only_filter("", nullptr).empty());
  }

  // ---- R1 (oversight): hip_switch=on is ignored for models M, L and K alike ----------------------------------------
  // The selector sees a model only as "is it K"; M and L are the same request. For each of the three: on an RX 9000 board
  // the default stays the D3D12 executor with one line about the option, a refused frame never moves to HIP, and the
  // option changes nothing about the choice (the same executor, default flag and log text with and without it).
  {
    struct Model {
      const char* name;
      bool k;
    };
    for (const Model m : {Model{"M", false}, Model{"L", false}, Model{"K", true}}) {
      for (const char* value : {"on", "1", "true"}) {
        wgi::ExecutorRequest r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
        r.model_k = m.k;
        wgi::ExecutorRequest plain = r;
        r.hip_switch = value;
        const wgi::ExecutorChoice c = choose_executor(r);
        const wgi::ExecutorChoice p = choose_executor(plain);
        const std::string tag = std::string("model ") + m.name + " hip_switch=" + value;
        expect(tag + ": stays on the D3D12 executor by default", c.executor == "dx12" && c.by_default);
        expect(tag + ": one line, no HIP", has(c.hip_note, (std::string("hip_switch=") + value + " is ignored").c_str()) &&
                                              has(c.hip_note, "never to HIP") && c.hip_note.find('\n') == std::string::npos);
        expect(tag + ": the choice is the one without the option",
               c.executor == p.executor && c.by_default == p.by_default && c.why == p.why && c.warn == p.warn &&
                   c.sparkle_off == p.sparkle_off && p.hip_note.empty());
        expect(tag + ": a refused frame goes to the front end, not to HIP", wgi::kRefusedFrameGoesToFrontEnd);
      }
      for (const char* value : {"off", "0", "false", ""}) {
        wgi::ExecutorRequest r = board("AMD Radeon RX 9070 XT", 0x1002, 0x7550);
        r.model_k = m.k;
        r.hip_switch = value;
        expect(std::string("model ") + m.name + " hip_switch=" + value + ": nothing to say",
               choose_executor(r).hip_note.empty());
      }
    }
  }

  std::printf(failures ? "%d FAILED\n" : "ALL TESTS PASSED\n", failures);
  return failures ? 1 : 0;
}
