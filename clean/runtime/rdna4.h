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
// Header only and portable: the lookup takes its sources as functions (AdapterSources), so that tests run on any host
// with fakes; rdna4_win.h supplies the system's.
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace wgrt {

// RDNA 4 PCI device ids (Navi 48: RX 9070 XT / 9070 / 9070 GRE; Navi 44: RX 9060 XT / 9060), the Ray Reconstruction
// runtime's list, now the only one. AMD's vendor id is 0x1002.
inline bool rdna4_device_id(uint32_t device) {
  return device == 0x7550 || device == 0x7551 || device == 0x7590 || device == 0x7591;
}

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
  if ((a.kmt_known || wine) && vendor == 0x1002 && !rdna4_device_id(device)) s += "; an AMD board outside the RDNA 4 id list";
  return s;
}

// The one line when an adapter was accepted by the Wine fallback (#248); empty otherwise.
inline std::string wine_fallback_line(const AdapterIdentity& a) {
  if (a.kmt_known || !a.wine_fallback || !rdna4_identity(a)) return std::string();
  return "Wine: the driver's adapter id query is not implemented; the adapter was identified from the system DXGI's ids (" +
         describe_identity(a) + ")";
}

// The sentence every refusal of a GPU that is not RDNA 4 (or not identified) starts with: the runtime's log line when it
// hands the game to AMD FSR, the Ray Reconstruction runtime's, and the launcher's (clean/installer wg_core
// NeedsRx9000Sentence is the same words).
inline const char* needs_rx9000_sentence() { return "WrongGPU's models need an RX 9000-series GPU; using AMD FSR"; }

}  // namespace wgrt
