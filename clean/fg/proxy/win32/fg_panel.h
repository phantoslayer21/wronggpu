// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation: WrongGPU-FG.asi's "WrongGPU Frame Generation" card in OptiScaler's menu (the host menu
// interface, clean/wgshared/include/wg_host_menu_abi.h; the same interface the RR plugin's box and WrongGPU.asi's upscaler
// card use).
//
//  - Where: a card at the top of OptiScaler's Custom tab (register_custom_card); an OptiScaler without cards gets a tab
//    of its own (register_tab).
//  - Status line: Off (fg=off), not active (why the hooks are not in place), waiting for the game, generating (2x..6x,
//    the preset, the presenter) or real frames only (the reason), or stopped for the run.
//  - With fg=on: Preset (A / B / C) and Multiplier (2x..6x). A change is handed to the adapter core at once (Hooks::apply,
//    FgProxy::update_live_settings: taken on the game's thread at its next frame, the session keeps running) and saved
//    in the game's state folder (<game>\wronggpu-setup\settings.ini, wgstate::SetSettings: under the state lock,
//    written atomically), so the WrongGPU Launcher shows the same values.
//  - Threads: the card is drawn on OptiScaler's render thread, which can be a present inside the FG presenter's own
//    worker. The status hook and the live hook take no lock the present path holds (FgAsiSessionHost::LiveStatus and
//    FgProxy::update_live_settings are lock-free); this module's own lock guards only its settings copy.
// Written for this project.
#pragma once
#include <stdint.h>

#include <string>

#include "../../../wgshared/include/wg_host_menu_abi.h"
#include "../fg_settings.h"
#include "fg_asi_session.h"

namespace wgfgp {
namespace win32 {
namespace panel {

// What the card shows (Hooks::status). Plain values.
struct Status {
    bool enabled = false;          // fg=on at the game's start
    bool installed = false;        // the transport routes the game's calls (import slots or host routes)
    uint32_t install_result = 0;   // FgInstallResult (fg_asi_transport.h) while not installed
    bool have_host = false;        // a session host exists (a binding was taken and the runtime checked)
    FgAsiLiveStatus live;          // the session host's live status (have_host)
};

struct Hooks {
    Status (*status)() = nullptr;                                 // lock-free
    bool (*apply)(uint32_t preset, uint32_t multiplier) = nullptr;  // live, lock-free (null: next start only)
    std::wstring (*game_dir)() = nullptr;                         // the folder holding wronggpu-setup\settings.ini
    void (*log)(const char* line) = nullptr;                      // wronggpu-fg.log
};

// Registers the card; `start` is the FG settings read at the game's start. false: the host's interface lacks what the
// card needs or refused it (logged).
bool Init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin, const Hooks& hooks, const FgSettings& start);

// What the menu's combos do (also called by the tests): the live hook, then settings.ini. Returns the log line;
// empty: not a valid value (nothing changes).
std::string ChoosePreset(uint32_t preset);
std::string ChooseMultiplier(uint32_t multiplier);
FgSettings Current();  // the card's settings copy

// The status line the card shows for `s` (also used by the tests).
std::string StatusLine(const Status& s, const FgSettings& current);

}  // namespace panel
}  // namespace win32
}  // namespace wgfgp
