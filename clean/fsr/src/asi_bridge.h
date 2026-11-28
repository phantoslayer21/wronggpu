// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Front-end side of the WrongGPU ASI plugin interface (clean/fsr/asi/asi_abi.h): find WrongGPU.asi if it is loaded
// in the process and ask it which upscaler entry the player selected in OptiScaler's menu. No link to the plugin.
#include <cstdint>
#include <string>

namespace wgwin {

// Sets *model to 0 (AMD FSR) or a WrongGPU model ('M', 'K'; which WrongGPU model runs is the model choice record's).
// Returns true only when the plugin is present and has hooked the host's menu. False otherwise.
bool QueryAsiPlugin(char* model);

// Tells the plugin which model runs (its menu marks that entry). Nothing happens without the plugin or the export.
void ShowAsiModel(char model);

// Tells the plugin where the game's model choice record is (src/model_choice_record.h). Nothing without the plugin.
void SetAsiChoiceRecord(const std::wstring& path);

// rc3: tells the plugin what the front end actually runs (asi_abi.h WrongGpuAsiReportSrV1: the card's status lines).
// Nothing happens without the plugin or the export (an older plugin).
void ReportAsiSr(uint32_t state, char running, char pending, int32_t runningMode, char errorModel, uint32_t errorWidth,
                 uint32_t errorHeight, const std::string& errorReason);

}  // namespace wgwin
