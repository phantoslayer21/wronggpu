// SPDX-License-Identifier: MIT
#pragma once
// Front-end side of the WrongGPU ASI plugin interface (clean/fsr/asi/asi_abi.h): find WrongGPU.asi if it is loaded
// in the process and ask it which upscaler entry the player selected in OptiScaler's menu. No link to the plugin.
#include "model_precedence.h"

namespace wgwin {

// Fills *choice: the plugin's saved entry (0 AMD FSR, 'M', 'K', 'L'), whether the player chose it, and when
// (WrongGpuAsiQueryChoiceV1; a plugin without it: WrongGpuAsiQuery, with Model M as its default entry and K / AMD as
// explicit choices with no recorded time). Returns true only when the plugin is present and has hooked the host's
// menu (so the plugin's choice, weighed against the launcher's, decides the model). False otherwise.
bool QueryAsiPlugin(wgfsr::PluginModelChoice* choice);

// Tells the plugin which model runs (its menu marks that entry). Nothing happens without the plugin or the export.
void ShowAsiModel(char model);

}  // namespace wgwin
