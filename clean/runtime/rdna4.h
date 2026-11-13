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

}  // namespace wgrt
