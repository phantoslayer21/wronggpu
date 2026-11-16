// SPDX-License-Identifier: MIT
// The adapters WrongGPU's networks run on: AMD RDNA 4 (gfx1200 / gfx1201) Radeon RX 9070 and RX 9060 series.
// WrongGPU's own code. One rule for every place that asks: the native D3D12 device (device_d3d12.cpp, borrowed and own
// device), the interop's executor choice (interop/executor_select.h, backend.cpp) and the Ray Reconstruction runtime
// (rr_f/runtime). The driver-channel WMMA ports are qualified only on this target; the device's driver-channel probe
// (native_driver_probe) still has to pass on it at create.
//
// Identity is decided by the adapter's PCI VENDOR and DEVICE id as the kernel-mode driver reports them, and by nothing
// else (R1, oversight 2026-11-15; R1-AUDIT H8). A description or a vendor id read through the game's dxgi.dll can be
// rewritten by a layer in the game process (OptiScaler's GPU spoofing), and a name tells nothing about the silicon, so no
// name ever accepts or rejects an adapter; the system DXGI's description is read only to be shown in a log line. The rule
// FAILS CLOSED: an adapter whose ids could not be read, or whose ids are not RDNA 4's, is not RDNA 4, and the caller hands
// the game to the front end's upscaler (AMD FSR) with one line that carries the ids, so that an RDNA 4 board missing from
// the list below shows up in a log and can be added. (The AMD AGS library's family query would be a second, family-level
// source; it is not part of this project's build, see the notes of the R1 pull request.)
//
// Wine and Proton (#248): Wine does not implement the driver's id query (D3DKMTQueryAdapterInfo type 31 answers
// STATUS_NOT_IMPLEMENTED), so under Wine the rule above would refuse every board. Only in that case, the query answering
// exactly STATUS_NOT_IMPLEMENTED in a process that runs under Wine (ntdll exports wine_get_version), the system DXGI's
// vendor and device id of the adapter with the same LUID stand in for the driver's, against the same vendor and the same
// id list. Any other failure, and the same status outside Wine, fails closed as before. Windows never takes this path.
//
// Developer-only opening of RDNA 3 (WG_DEV_GFX11, OFF in every normal build, never defined by clean/runner/build.py): the
// D3D12 device, the executor choice and the Wine log line also accept the RX 7900 series board id (Navi 31, gfx1100) and
// the device then runs the gfx11 kernel set (dx12-gfx11, clean/dx12/build_dx12.py --gfx 11). rdna4_identity itself never
// changes (the Ray Reconstruction runtime has no gfx11 kernels); the places that open the device ask
// dev_gfx11_identity as well, which is false without the macro, so a normal build behaves exactly as before.
//
// Header only and portable: the lookup takes its sources as functions (AdapterSources), so that tests run on any host
// with fakes; rdna4_win.h supplies the system's.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace wgrt {

// RDNA 4 PCI device ids (Navi 48: RX 9070 XT / 9070 / 9070 GRE; Navi 44: RX 9060 XT / 9060), the Ray Reconstruction
// runtime's list, now the only one. AMD's vendor id is 0x1002.
inline bool rdna4_device_id(uint32_t device) {
  return device == 0x7550 || device == 0x7551 || device == 0x7590 || device == 0x7591;
}

// WG_DEV_GFX11: the developer opening of RDNA 3 (see the header comment). A constant, so that both builds are compiled and
// tested; without the macro every function below that mentions it answers as it did before it existed.
#ifdef WG_DEV_GFX11
inline constexpr bool kDevGfx11 = true;
#else
inline constexpr bool kDevGfx11 = false;
#endif

// The second developer-only choice inside the opening, the SHORTCUT: the board runs the UNCHANGED gfx12 kernel set (dx12) and
// the gfx12 accumulator table, the FP8 matrix operations being emulated by dxil-spirv (the process runs with
// DXIL_SPIRV_CONFIG=wmma_rdna3_raw_fp8 in a patched vkd3d-proton). Chosen by the environment variable
// WG_DEV_GFX11_SHORTCUT=1, which is read only in a build made with WG_DEV_GFX11 (without the macro this function is
// constant false and the library never asks the environment). A mechanism for the maintainer to replace.
#ifdef WG_DEV_GFX11
inline bool dev_gfx11_shortcut() {
  const char* v = std::getenv("WG_DEV_GFX11_SHORTCUT");
  return v && v[0] == '1' && v[1] == 0;
}
#else
inline bool dev_gfx11_shortcut() { return false; }
#endif

// The third developer-only choice, NATIVE K on top of the shortcut: the board runs the shortcut (above) and, for Model K,
// launches the native RDNA 3 kernel <name>__g11 of the dx12-gfx11 set wherever that manifest has it (K-NATIVE-1.md 5.1 (a));
// every other kernel still runs as its gfx12 port through the shortcut. Chosen by WG_DEV_GFX11_NATIVE_K=1 together with
// WG_DEV_GFX11_SHORTCUT=1 (its base; the device refuses the one without the other), read only in a build made with
// WG_DEV_GFX11. One build times both: the same run without WG_DEV_GFX11_NATIVE_K is the shortcut. The value picks the kernels:
// WG_DEV_GFX11_NATIVE_K=1 all of them, WG_DEV_GFX11_NATIVE_K=<comma list of port names> only those (per-kernel timing found
// native faster on some stages and slower on others).
#ifdef WG_DEV_GFX11
// The value of WG_DEV_GFX11_NATIVE_K: "1" = every kernel the manifest has a __g11 form of; a comma list of port names
// (k_stage2_d0_hp,k_stage_e2) = only those; unset, empty or "0" = off.
inline std::string dev_gfx11_native_k_value() {
  const char* v = std::getenv("WG_DEV_GFX11_NATIVE_K");
  return v ? std::string(v) : std::string();
}
inline bool dev_gfx11_native_k_on(const std::string& value) { return !value.empty() && value != "0"; }
inline bool dev_gfx11_native_k() { return dev_gfx11_native_k_on(dev_gfx11_native_k_value()); }
inline bool dev_gfx11_native_name_taken(const std::string& name) {  // a native kernel's own name is never looked up again
  return name.size() > 5 && name.compare(name.size() - 5, 5, "__g11") == 0;
}
// Does `value` select the port `name`: "1" selects every name, a list only the names in it (whole names, no prefixes).
inline bool dev_gfx11_native_k_selects(const std::string& value, const std::string& name) {
  if (!dev_gfx11_native_k_on(value)) return false;
  if (value == "1") return true;
  for (size_t start = 0; start <= value.size();) {
    size_t end = value.find(',', start);
    if (end == std::string::npos) end = value.size();
    if (value.compare(start, end - start, name) == 0 && end - start == name.size()) return true;
    start = end + 1;
  }
  return false;
}
// The kernel that runs for `name`: `name__g11` when `value` selects it and the manifest has it (`has` answers for a name),
// else `name`.
template <class Has>
inline std::string dev_gfx11_native_kernel(const std::string& value, const std::string& name, Has has) {
  if (dev_gfx11_native_name_taken(name) || !dev_gfx11_native_k_selects(value, name)) return name;
  const std::string g11 = name + "__g11";
  return has(g11) ? g11 : name;
}
#else
inline bool dev_gfx11_native_k() { return false; }
template <class Has>
inline std::string dev_gfx11_native_kernel(const std::string&, const std::string& name, Has) { return name; }
#endif

// RDNA 3 PCI device id the opening takes: Navi 31 (1002:744C, which the RX 7900 XT, XTX and GRE share; gfx1100), the id
// read on the development card. No other RDNA 3 id: gfx1101 and later boards were never measured here.
inline bool gfx11_device_id(uint32_t device) { return device == 0x744C; }

// The target the D3D12 device reports for an adapter of the opening (runtime.cpp takes the accumulator row table of the
// target's family from it, as for a HIP device).
inline const char* dev_gfx11_arch() { return "gfx1100"; }

// The folder of a build's D3D12 kernel set (<build>/<folder>/manifest.json): dx12 (RDNA 4), dx12-gfx11 for a target of the
// opening. `arch` is the device's target name ("" for RDNA 4).
inline const char* dx12_folder(const std::string& arch) { return kDevGfx11 && arch == dev_gfx11_arch() ? "dx12-gfx11" : "dx12"; }

struct AdapterLuid {
  uint32_t low = 0;
  int32_t high = 0;
  bool operator==(const AdapterLuid& o) const { return low == o.low && high == o.high; }
};

// An adapter the system's DXGI lists (its description is for log lines only).
struct AdapterRecord {
  AdapterLuid luid;
  uint32_t vendor = 0;
  uint32_t device = 0;  // DXGI's DeviceId (read only for the Wine fallback)
  bool software = false;
  std::string description;
};

// What is known about one adapter.
struct AdapterIdentity {
  bool dxgi_known = false;  // the system's DXGI listed the adapter (never the game's dxgi.dll): for log lines only
  uint32_t dxgi_vendor = 0;
  bool software = false;
  std::string description;
  bool kmt_known = false;  // the kernel-mode driver's ids were read: the only evidence that decides
  uint32_t kmt_vendor = 0, kmt_device = 0;
  // Wine only: the driver's query answered STATUS_NOT_IMPLEMENTED under Wine, and the system DXGI's ids were read instead
  bool wine_fallback = false;
  uint32_t wine_vendor = 0, wine_device = 0;
};

// NTSTATUS 0xC0000002, STATUS_NOT_IMPLEMENTED: Wine's answer to the driver's id query.
constexpr int32_t kStatusNotImplemented = -1073741822;
// NTSTATUS 0xC0000001, STATUS_UNSUCCESSFUL: what rdna4_win.h's query reports when it could not ask at all (no thunks, the
// adapter not opened), so that only the query's own STATUS_NOT_IMPLEMENTED can lead to the Wine fallback.
constexpr int32_t kStatusUnsuccessful = -1073741823;

// The sources (injectable; rdna4_win.h: the system's own DXGI by its full path, and D3DKMTQueryAdapterInfo).
struct AdapterSources {
  std::function<bool(const AdapterLuid&, AdapterRecord*)> find_adapter;                       // system DXGI, by LUID
  std::function<bool(const AdapterLuid&, uint32_t* vendor, uint32_t* device)> driver_ids;  // kernel-mode driver, by LUID
  // The driver's query with its own status (0: the ids were read); when set, it is asked instead of driver_ids.
  std::function<int32_t(const AdapterLuid&, uint32_t* vendor, uint32_t* device)> driver_query;
  // The Wine fallback's two sources, asked only when driver_query answered kStatusNotImplemented: whether the process runs
  // under Wine, then the system DXGI's vendor and device id of the adapter with this LUID.
  std::function<bool()> under_wine;
  std::function<bool(const AdapterLuid&, uint32_t* vendor, uint32_t* device)> dxgi_ids;
};

// The adapter with this LUID (the game device's): the same LUID goes to both sources, nothing is guessed; what a source
// cannot answer stays unknown.
inline AdapterIdentity lookup_adapter(const AdapterLuid& luid, const AdapterSources& s) {
  AdapterIdentity id;
  AdapterRecord rec;
  if (s.find_adapter && s.find_adapter(luid, &rec)) {
    id.dxgi_known = true;
    id.dxgi_vendor = rec.vendor;
    id.software = rec.software;
    id.description = rec.description;
  }
  if (s.driver_query) {
    uint32_t v = 0, d = 0;
    const int32_t status = s.driver_query(luid, &v, &d);
    if (status == 0) {
      id.kmt_known = true;
      id.kmt_vendor = v;
      id.kmt_device = d;
    } else if (status == kStatusNotImplemented && s.under_wine && s.under_wine() && s.dxgi_ids) {
      uint32_t wv = 0, wd = 0;
      if (s.dxgi_ids(luid, &wv, &wd)) {
        id.wine_fallback = true;
        id.wine_vendor = wv;
        id.wine_device = wd;
      }
    }
  } else if (s.driver_ids) {
    id.kmt_known = s.driver_ids(luid, &id.kmt_vendor, &id.kmt_device);
  }
  return id;
}

// An RDNA 4 Radeon RX 9000-series board: the driver's vendor and device id, nothing else; unknown ids are not RDNA 4.
// Under Wine, when the driver's query is not implemented there, the system DXGI's ids under the same rule (#248).
inline bool rdna4_identity(const AdapterIdentity& a) {
  if (a.kmt_known) return a.kmt_vendor == 0x1002 && rdna4_device_id(a.kmt_device);
  return a.wine_fallback && a.wine_vendor == 0x1002 && rdna4_device_id(a.wine_device);
}

// The adapter is an RX 7900 series board and the developer opening is on (WG_DEV_GFX11): the same rule as rdna4_identity (the
// driver's ids, else the Wine fallback's, vendor and device id only), against the RDNA 3 id. Always false without the macro.
inline bool dev_gfx11_identity(const AdapterIdentity& a) {
  if (!kDevGfx11) return false;
  if (a.kmt_known) return a.kmt_vendor == 0x1002 && gfx11_device_id(a.kmt_device);
  return a.wine_fallback && a.wine_vendor == 0x1002 && gfx11_device_id(a.wine_device);
}

// The adapter as a log line shows it: the system DXGI's description when it has one, the driver's ids when known, and the
// words "not identified" when neither is.
inline std::string describe_identity(const AdapterIdentity& a) {
  char ids[32] = "";
  if (a.kmt_known) std::snprintf(ids, sizeof ids, "PCI %04X:%04X", (unsigned)a.kmt_vendor, (unsigned)a.kmt_device);
  std::string s;
  if (a.dxgi_known) s = "adapter " + (a.description.empty() ? std::string("(no name)") : a.description);
  if (a.kmt_known) s += (s.empty() ? "adapter " : ", ") + std::string(ids);
  const bool wine = !a.kmt_known && a.wine_fallback;
  if (wine) {
    std::snprintf(ids, sizeof ids, "PCI %04X:%04X", (unsigned)a.wine_vendor, (unsigned)a.wine_device);
    s += (s.empty() ? "adapter " : ", ") + std::string(ids) + " (Wine: the system DXGI's ids)";
  }
  if (s.empty()) s = "adapter not identified";
  const uint32_t vendor = wine ? a.wine_vendor : a.kmt_vendor, device = wine ? a.wine_device : a.kmt_device;
  if ((a.kmt_known || wine) && vendor == 0x1002 && !rdna4_device_id(device) && !dev_gfx11_identity(a))
    s += "; an AMD board outside the RDNA 4 id list";
  return s;
}

// The one line when an adapter was accepted by the Wine fallback (#248); empty otherwise.
inline std::string wine_fallback_line(const AdapterIdentity& a) {
  if (a.kmt_known || !a.wine_fallback || !(rdna4_identity(a) || dev_gfx11_identity(a))) return std::string();
  return "Wine: the driver's adapter id query is not implemented; the adapter was identified from the system DXGI's ids (" +
         describe_identity(a) + ")";
}

// The sentence every refusal of a GPU that is not RDNA 4 (or not identified) starts with: the runtime's log line when it
// hands the game to AMD FSR, the Ray Reconstruction runtime's, and the launcher's (clean/installer wg_core
// NeedsRx9000Sentence is the same words).
inline const char* needs_rx9000_sentence() { return "WrongGPU's models need an RX 9000-series GPU; using AMD FSR"; }

}  // namespace wgrt
