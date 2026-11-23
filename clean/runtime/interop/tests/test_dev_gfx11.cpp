// SPDX-License-Identifier: MIT
// The developer-only opening of RDNA 3 (WG_DEV_GFX11, clean/runtime/rdna4.h) on any host (no GPU, no D3D12). WrongGPU's own
// code. The same file is built twice and both builds must pass:
//   clang++ -std=c++17 test_dev_gfx11.cpp -o test_dev_gfx11.exe                  (a normal build: nothing changes)
//   clang++ -std=c++17 -DWG_DEV_GFX11 test_dev_gfx11.cpp -o test_dev_gfx11_dev.exe  (the opening: the RX 7900 series id)
// clean/tests/test_dev_gfx11.py builds and runs both. Exit status 0 when every case passed.
#include <cstdio>
#include <string>

#include "../../target_family.h"
#include "../executor_select.h"

namespace {

int failures = 0;

void expect(const std::string& name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name.c_str());
  if (!ok) ++failures;
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// An adapter as the kernel-mode driver reports it (the ids decide; the name is for log lines only).
wgrt::AdapterIdentity driver(const char* name, uint32_t vendor, uint32_t device) {
  wgrt::AdapterIdentity id;
  id.dxgi_known = true;
  id.dxgi_vendor = vendor;
  id.description = name;
  id.kmt_known = true;
  id.kmt_vendor = vendor;
  id.kmt_device = device;
  return id;
}

// An adapter identified the Wine way (the driver's query is not implemented there; the system DXGI's ids stand in).
wgrt::AdapterIdentity wine(const char* name, uint32_t vendor, uint32_t device) {
  wgrt::AdapterIdentity id;
  id.dxgi_known = true;
  id.dxgi_vendor = vendor;
  id.description = name;
  id.wine_fallback = true;
  id.wine_vendor = vendor;
  id.wine_device = device;
  return id;
}

wgi::ExecutorRequest request_for(const wgrt::AdapterIdentity& id) {
  wgi::ExecutorRequest r;
  r.adapter_known = id.dxgi_known;
  r.vendor_id = id.dxgi_vendor;
  r.adapter = id.description;
  r.kmt_known = id.kmt_known;
  r.kmt_vendor = id.kmt_vendor;
  r.kmt_device = id.kmt_device;
  r.wine_fallback = id.wine_fallback;
  r.wine_vendor = id.wine_vendor;
  r.wine_device = id.wine_device;
  return r;
}

// The rule the native device asks (device_d3d12.cpp, both paths) and the executor choice asks (executor_select.h)
bool taken(const wgrt::AdapterIdentity& id) { return wgrt::rdna4_identity(id) || wgrt::dev_gfx11_identity(id); }

}  // namespace

int main() {
  using wgrt::kDevGfx11;
  const char* const sentence = "WrongGPU's models need an RX 9000-series GPU; using AMD FSR";
#ifdef WG_DEV_GFX11
  expect("the macro makes kDevGfx11 true", kDevGfx11);
#else
  expect("without the macro kDevGfx11 is false", !kDevGfx11);
#endif

  // The id read on the development card (clean/dx12 notes: 1002:744C, RX 7900 XT; the XTX and GRE share the id)
  const wgrt::AdapterIdentity card = driver("AMD Radeon RX 7900 XT", 0x1002, 0x744C);
  expect("rdna4_identity never takes the RX 7900 series id, with or without the macro", !wgrt::rdna4_identity(card));
  expect("dev_gfx11_identity takes the RX 7900 series id exactly when the macro is on",
         wgrt::dev_gfx11_identity(card) == kDevGfx11);
  expect("the RDNA 4 ids are taken in both builds", taken(driver("AMD Radeon RX 9070 XT", 0x1002, 0x7550)) &&
                                                    !wgrt::dev_gfx11_identity(driver("x", 0x1002, 0x7550)));

  // Everything else stays refused in both builds: other RDNA 3 and older boards, the iGPU of the development machine
  // (1002:13C0), the workstation id next to the opened one (7448, never measured here), another vendor's id
  struct Other { const char* name; uint32_t vendor, device; };
  const Other others[] = {{"AMD Radeon RX 7800 XT", 0x1002, 0x747E},   {"AMD Radeon PRO W7900", 0x1002, 0x7448},
                          {"AMD Radeon RX 7600", 0x1002, 0x7480},       {"AMD Radeon RX 6800", 0x1002, 0x73BF},
                          {"AMD Radeon(TM) Graphics", 0x1002, 0x13C0},  {"Other Vendor 744C", 0x10DE, 0x744C},
                          {"AMD Radeon RX 7900 XT", 0x1002, 0x744D},    {"AMD Radeon RX 7900 XT", 0x1002, 0x744B}};
  for (const Other& o : others) {
    const wgrt::AdapterIdentity id = driver(o.name, o.vendor, o.device);
    expect(std::string("refused in both builds: ") + o.name, !taken(id) &&
           wgi::choose_executor(request_for(id)).executor == "fsr");
  }
  expect("the name never accepts: a 7900 XT name on another id is refused",
         !taken(driver("AMD Radeon RX 7900 XT", 0x1002, 0x73BF)));
  expect("the ids never read: refused even with the name (fails closed)", [&] {
    wgrt::AdapterIdentity id;
    id.dxgi_known = true;
    id.dxgi_vendor = 0x1002;
    id.description = "AMD Radeon RX 7900 XT";
    return !taken(id) && !wgrt::dev_gfx11_identity(id);
  }());

  // The executor choice
  {
    const wgi::ExecutorChoice c = wgi::choose_executor(request_for(card));
    if (kDevGfx11)
      expect("macro on: the board takes the D3D12 executor by default", c.executor == "dx12" && c.by_default && !c.warn);
    else
      expect("macro off: the board goes to the front end's upscaler with the sentence, as before",
             c.executor == "fsr" && c.by_default && c.warn && c.why.rfind(sentence, 0) == 0 && has(c.why, "PCI 1002:744C") &&
                 has(c.why, "outside the RDNA 4 id list"));
    wgi::ExecutorRequest r = request_for(card);
    r.executor = "dx12";
    expect("executor=dx12 is the same in both builds (the device decides)", wgi::choose_executor(r).executor == "dx12");
    r.executor = "hip";
    expect("executor=hip is ignored in a release build in both (R1)", wgi::choose_executor(r, false).executor ==
           (kDevGfx11 ? "dx12" : "fsr"));
  }

  // Wine: the system DXGI's ids stand in only when the driver's query is not implemented; the same rule
  {
    const wgrt::AdapterIdentity w = wine("AMD Radeon RX 7900 XT (RADV NAVI31)", 0x1002, 0x744C);
    expect("Wine: the fallback ids are taken exactly when the macro is on", taken(w) == kDevGfx11);
    const std::string line = wgrt::wine_fallback_line(w);
    expect("Wine: the fallback line is said exactly when the board is taken", line.empty() == !kDevGfx11);
    if (kDevGfx11) expect("Wine: the line names the adapter and the ids", has(line, "PCI 1002:744C"));
    expect("describe_identity: the 'outside the RDNA 4 id list' phrase is gone only for a taken board",
           has(wgrt::describe_identity(w), "outside the RDNA 4 id list") == !kDevGfx11);
    expect("Wine: a board the opening does not take keeps the phrase",
           has(wgrt::describe_identity(wine("x", 0x1002, 0x747E)), "outside the RDNA 4 id list"));
    wgrt::AdapterIdentity both = wine("x", 0x1002, 0x744C);  // the driver's ids, when read, decide and the DXGI's never override
    both.kmt_known = true;
    both.kmt_vendor = 0x1002;
    both.kmt_device = 0x747E;
    expect("the driver's ids decide over the Wine fallback's: a refused board stays refused", !taken(both));
    both.kmt_device = 0x7550;
    expect("... and an RDNA 4 board stays taken", taken(both) && !wgrt::dev_gfx11_identity(both));
  }

  // The kernel folder and the target name
  expect("the target the opening reports is a gfx11 target of the family map", wgrt::target_family(wgrt::dev_gfx11_arch()) == "gfx11");
  expect("the RDNA 4 device (no target name) reads the folder dx12", std::string(wgrt::dx12_folder("")) == "dx12");
  expect("a gfx12 target reads the folder dx12", std::string(wgrt::dx12_folder("gfx1201")) == "dx12");
  expect("a gfx1100 target reads dx12-gfx11 exactly when the macro is on",
         std::string(wgrt::dx12_folder(wgrt::dev_gfx11_arch())) == (kDevGfx11 ? "dx12-gfx11" : "dx12"));
  expect("another gfx11 target (gfx1101) is not opened", std::string(wgrt::dx12_folder("gfx1101")) == "dx12");

  std::printf(failures ? "%d FAILED\n" : "ALL TESTS PASSED\n", failures);
  return failures ? 1 : 0;
}
