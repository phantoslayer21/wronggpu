// SPDX-License-Identifier: MIT
#pragma once
// Front-end side of the WrongGPU ASI plugin interface (clean/fsr/asi/asi_abi.h): find WrongGPU.asi if it is loaded
// in the process and ask it which upscaler entry the player selected in OptiScaler's menu. No link to the plugin.
namespace wgwin {

// Sets *model to 0 (AMD FSR), 'M' or 'K'. Returns true only when the plugin is present and has hooked the host's
// menu (so the front end should list AMD-only versions and let the plugin choose the model). False otherwise.
bool QueryAsiPlugin(char* model);

}  // namespace wgwin
