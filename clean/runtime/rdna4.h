// SPDX-License-Identifier: MIT
// The adapters Model M's D3D12 path runs on: AMD RDNA 4 (gfx1200 / gfx1201) Radeon RX 9070 and RX 9060 series.
// WrongGPU's own code. One rule for every place that asks: the native D3D12 device (device_d3d12.cpp, borrowed and
// own device) refuses other adapters, and the interop's executor choice (interop/executor_select.h) makes the D3D12
// executor the default on exactly these. The driver-channel WMMA ports are qualified only on this target; the
// device's driver-channel probe (native_driver_probe) still has to pass on it at create.
// Identification as DXGI reports the adapter: vendor 0x1002, a hardware adapter (not DXGI_ADAPTER_FLAG_SOFTWARE), and
// a description naming an RX 9070 or RX 9060 board (RX 9070 XT / 9070 / 9070 GRE: gfx1201; RX 9060 XT / 9060: gfx1200).
#pragma once

#include <cstdint>
#include <string>

namespace wgrt {

inline bool rdna4_adapter(uint32_t vendor_id, bool software, const std::string& description) {
  return vendor_id == 0x1002 && !software &&
         (description.find("RX 9070") != std::string::npos || description.find("RX 9060") != std::string::npos);
}

// RDNA 4 PCI device ids (Navi 48: RX 9070 XT / 9070 / 9070 GRE; Navi 44: RX 9060 XT / 9060), the same list as the
// Ray Reconstruction runtime's (clean/rr_f/runtime/src/rr_runtime.cpp).
inline bool rdna4_device_id(uint32_t device) {
  return device == 0x7550 || device == 0x7551 || device == 0x7590 || device == 0x7591;
}

// What is known about an adapter, from sources another layer in the game process cannot rewrite (rdna4_win.h reads
// them): the system's own DXGI (description, vendor id, software flag) and the kernel-mode driver's vendor and device id.
// OptiScaler's GPU spoofing rewrites what the GAME's dxgi.dll reports, so a description read there is never used.
struct AdapterIdentity {
  bool dxgi_known = false;  // the system's DXGI described the adapter
  uint32_t dxgi_vendor = 0;
  bool software = false;
  std::string description;
  bool kmt_known = false;  // the kernel-mode driver's ids were read
  uint32_t kmt_vendor = 0, kmt_device = 0;
};

// An RDNA 4 Radeon RX 9000-series board: the driver's own ids first (a real RX 9000 is one whatever name a layer
// presents, a spoofed name on another board is not one), else the system DXGI's description (rdna4_adapter). Nothing
// known: not identified, so not RDNA 4.
inline bool rdna4_identity(const AdapterIdentity& a) {
  if (a.kmt_known && a.kmt_vendor == 0x1002 && rdna4_device_id(a.kmt_device)) return true;
  if (a.dxgi_known) return rdna4_adapter(a.dxgi_vendor, a.software, a.description) && (!a.kmt_known || a.kmt_vendor == 0x1002);
  return false;
}

}  // namespace wgrt
