// SPDX-License-Identifier: MIT
#pragma once
// The front end's side of WrongGPU's shared settings (clean/wgshared): where the data folder and wronggpu.ini are, the
// shared settings block (opened or created at start, claimed as the single writer when OptiScaler hosts this DLL), the
// registry options the front end reads, the one-folder-per-choice sidecar, and the poll of OptiScaler.ini's
// [<custom folder>] UpscalerIndex. All of it goes through clean/wgshared/include/wg_settings_io.h, the one
// implementation every WrongGPU DLL uses (its Win32 part is compiled in settings_bridge.cpp).
#include <cstdint>
#include <string>

#include "frontend.h"

namespace wgwin {

// Where this DLL finds its settings and its runtime (computed once).
//   - custom slot: loaded from ...\custom\<folder>\ffx\<file> (OptiScaler 10's custom upscaler folder);
//   - data folder: <game exe folder>\WrongGPU (WG_DATA_DIR overrides it), the one place of wronggpu.ini and the
//     runtimes (DESIGN-i15 section 9);
//   - the old place (next to this DLL, or next to the exe in the custom slot) is still read when the data folder has
//     no wronggpu.ini, with a log line.
struct SettingsPlace {
    std::wstring modulePath;   // this DLL
    std::wstring dataDir;      // wg::data_dir()
    std::wstring ini;          // the wronggpu.ini used (read, written with defaults when absent, persisted to)
    std::wstring runtimeDir;   // the folder runtime_dll is loaded from
    bool customSlot = false;
    std::wstring folder;       // the custom folder's name (the OptiScaler.ini section to poll), custom slot only
    std::wstring hostIni;      // OptiScaler.ini next to OptiScaler's folder, custom slot only
    std::wstring sidecar;      // wronggpu-entry.ini next to this DLL
    bool oldPlace = false;     // ini from the old place (next to the DLL / the exe)
    std::string note;          // one log line about the choice ("" when nothing to say)
};
const SettingsPlace& Place();

// Opens or creates the shared settings block (seeded from Place().ini) and, when `hosted` (an upscaler mod loaded this
// DLL under its own name, e.g. OptiScaler's custom slot), claims it as the single writer. Reads the sidecar and sets
// up the OptiScaler.ini poll. Logs the options the front end reads ("settings: key=value (source)"). Once per process.
void InitSettings(bool hosted);

// The registry options and the menu's choices for the front end (Platform::menu).
wgfsr::MenuSettings Menu();

// A registry option's value for this module (block > environment > ini > default) and its log text.
int32_t Setting(uint32_t slot, std::string* described = nullptr, bool* isDefault = nullptr);

// The OptiScaler.ini [<folder>] UpscalerIndex poll (mtime at most once per second); applies a changed index as if it
// had been chosen in the menu (writer only). True while there is something to poll (custom slot and writer).
bool PollHostIni();
bool HostIniPollActive();

// For tests and logs.
bool IsSettingsWriter();

}  // namespace wgwin
