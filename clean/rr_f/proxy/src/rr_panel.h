// SPDX-License-Identifier: MIT
// The RR plugin's box in OptiScaler's menu (OptiScaler builds with the plugin interface, wg_host_menu_abi.h): a
// "WrongGPU" card in the Custom tab whose Ray reconstruction box chooses the RR mode while the game runs RR (owner
// 2026-10-31: RR mode is a box of its own, not an entry of the host's version list; the box is there only while RR
// runs), and the RR name and GPU time in the host's overlay. Exact is offered
// only when the launcher allowed it before the game started (rr_exact_allowed, latched at start); without it a mode of
// exact runs accurate. The choice goes to the shared settings block (the RR runtime reads it at the next frame) and
// to WrongGPU\wronggpu.ini (the next start keeps it).
#pragma once

#include <stdint.h>

#include <string>

#include "../include/wg_host_menu_abi.h"

namespace wgrr {
class Settings;
namespace panel {

// The modes the box offers, in box order: exact (only when allowed), accurate, fast. Returns the count.
int offered_modes(bool exact_allowed, int32_t* out);
// The mode that runs for a chosen one: exact without the launcher's permission runs accurate.
int32_t effective_mode(int32_t mode, bool exact_allowed);
// The title the box shows for a mode ("Exact", "Accurate", "Fast").
const char* mode_title(int32_t mode);

typedef void (*LogFn)(const std::string& line);
typedef bool (*ActiveFn)();
typedef float (*GpuMsFn)();
typedef bool (*PreparingFn)();

// From InitializeASI, before the RR core is built: opens the shared settings block, latches rr_exact_allowed and
// turns a disallowed exact into accurate (in the block, not in the file). `rr_active` tells the box whether RR runs,
// `gpu_ms` gives the RR runtime's GPU time per frame (0 when unknown).
// `preparing` (release 0.6.0): the RR runtime still creates its frame kernels (first start, cold driver cache).
void start(LogFn log, ActiveFn rr_active, GpuMsFn gpu_ms, PreparingFn preparing = nullptr);
// Whether exact may run in this process (rr_exact_allowed at start).
bool exact_allowed();
// rr_mode as it stands, the box's own precedence: the shared settings block's slot (when set), else WG_RR_MODE, else
// WrongGPU\wronggpu.ini (read only while the block is unavailable), else the registry default. The plugin hands the
// same value to the RR runtime at start (clean/060-rr-plugin-hunt).
struct ModeSetting {
  int32_t mode = 0;     // a registry mode id (the registry default when nothing sets it)
  std::string source;   // log words: "shared settings, wronggpu.ini", "WG_RR_MODE", "wronggpu.ini", "default"
  bool set = false;     // a source sets it (false: the registry default applies)
  bool shared = false;  // the shared settings block was readable
};
ModeSetting resolved_mode();
// OptiPluginInit: the box goes into the host's Custom tab (owner 2026-10-31: no tab of its own; a host without
// Custom tab cards gets a tab). False when the host's table lacks what the box needs.
bool menu_init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin);
// The plugin's RR settings (wronggpu-rr.ini, read once): the box shows the opt-in reflection speckle filter as a
// checkbox while they are bound (the change applies from the next frame and is saved to that file).
void bind_settings(Settings* settings);
// After every RR frame the core accepted: tells the host "WrongGPU RR <mode>" and the GPU time (its overlay), or
// "WrongGPU RR (preparing...)" while the runtime still creates its frame kernels (the frames pass through meanwhile).
void frame_done();

}  // namespace panel
}  // namespace wgrr
