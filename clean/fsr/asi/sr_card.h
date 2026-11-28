// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// WrongGPU.asi's card in OptiScaler's menu (the host's plugin interface, clean/wgshared/include/wg_host_menu_abi.h; the
// same C ABI and registration pattern as the RR plugin's box, clean/rr_f/proxy/src/rr_panel.cpp): "WrongGPU Upscaler"
// on the host's Custom tab (a tab of its own on hosts without cards). It names the SR model that runs and offers its
// mode, Fast (the default) or Exact, plus the other models' modes. A change applies at once and is saved:
//   1. the shared settings block's slot (sr_m_mode / sr_l_mode / sr_k_mode; the card owns these slots,
//      wg_settings_io.h SharedSettings::write_owned_slot): the SR front end creates its upscaler again with the new mode
//      at the next frame (frontend.cpp SwitchModeForFrame);
//   2. the game's state folder, <game>\wronggpu-setup\settings.ini (m_mode / l_mode / k_mode = fast | exact), under the
//      game's state lock, written atomically (wgstate::SetSettings): the launcher shows the choice, and the next start
//      keeps it.
#include <cstdint>
#include <functional>
#include <string>

#include "../../wgshared/include/wg_host_menu_abi.h"
#include "asi_abi.h"

namespace wgasi {
namespace card {

struct Hooks {
    std::function<std::wstring()> gameDir;  // the game folder whose wronggpu-setup holds settings.ini
    std::function<char()> activeModel;      // 'M', 'K', 'L': the model that runs (or the saved choice); 0: AMD's FSR runs
    std::function<void(const std::string&)> log;
    // rc3: what the front end reports it actually runs (asi_abi.h WrongGpuAsiReportSrV1); false until a front end reported
    // (an older one never does: the card then names activeModel as before). The status lines come from it: running
    // (the model and mode of the last frame), switching, idle, and the last model that could not start.
    std::function<bool(WrongGpuAsiSrV1*)> srStatus;
    // What a front end reported through WrongGpuAsiShowModelV1: 'M', 'K', 'L', 0 (AMD's FSR), or -1 when no front end
    // reported anything yet (the SR front end has not run: the card then says "Not in use", never "Running").
    std::function<int()> frontEndShown;
};

// OptiPluginInit's work: registers the card. False (nothing registered) when the host's table lacks what it needs.
bool Init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin, Hooks hooks);

// The mode of model 'M' / 'K' / 'L' as the card shows it (0 fast, 1 exact): the shared block's live slot, else the game's
// settings.ini (re-read at most once a second), else the option's environment variable, else fast.
int32_t CurrentMode(char model);

// The player picked `mode` for `model`: the block's slot, then settings.ini (wronggpu-setup). Returns the log line.
std::string Choose(char model, int32_t mode);

}  // namespace card
}  // namespace wgasi
