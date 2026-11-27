// SPDX-License-Identifier: GPL-3.0-only
// FG controls in OptiScaler's Frame Gen tab with plugin API 6, or a Custom card on older hosts.
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
    bool enabled = false;          // current WrongGPU FG switch
    bool installed = false;        // the transport routes the game's calls (import slots or host routes)
    uint32_t install_result = 0;   // FgInstallResult (fg_asi_transport.h) while not installed
    bool have_host = false;        // a session host exists (a binding was taken and the runtime checked)
    bool host_conflict = false;
    bool deferred = false;
    FgAsiLiveStatus live;          // the session host's live status (have_host)
};

struct Hooks {
    Status (*status)() = nullptr;                                 // lock-free
    bool (*apply)(uint32_t preset, uint32_t multiplier) = nullptr;  // live, lock-free (null: next start only)
    bool (*enable)(bool enabled) = nullptr;                       // live on an installed route
    std::wstring (*game_dir)() = nullptr;                         // the folder holding wronggpu-setup\settings.ini
    void (*log)(const char* line) = nullptr;                      // wronggpu-fg.log
};

// Registers the card; `start` is the FG settings read at the game's start. false: the host's interface lacks what the
// card needs or refused it (logged).
bool Init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin, const Hooks& hooks, const FgSettings& start);

bool HasFrameGenApi(const WgHostMenuApi* api);
bool ClaimOutput(const WgHostMenuApi* api, WgHostMenuPlugin* plugin);
void ReleaseOutput();
// Take this before reading live state, or before routing a present that will publish afterward.
uint64_t NextReportSequence() noexcept;
bool Publish(const Status& status, uint64_t sequence, bool immediate = false) noexcept;
// Deterministic clock entry point for unit tests.
bool PublishAt(const Status& status, uint64_t sequence, bool immediate, uint64_t now_ms) noexcept;

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
