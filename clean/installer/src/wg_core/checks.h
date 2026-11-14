// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the requirement checks (spec 30 "Requirements check"): Windows, the GPU, the HIP runtime
// (with its child-process probe), anti-cheat, a running game, write access. Written for this project.
#pragma once
#include "detect.h"
#include "model.h"

namespace wg {

struct WinInfo { bool ok = false; wstring text; };
struct GpuInfo { bool amd = false, rdna4 = false; wstring name, driver; };
struct HipInfo { bool ran = false, ok = false; int code = -1; wstring dll, device, arch, text; };
enum CheckState { CK_OK, CK_FAIL, CK_WARN, CK_NA };
enum { L_GAME, L_FFX, L_AC, L_WIN, L_GPU, L_HIP, L_MODEL, L_CONFLICT, L_IDLE, L_COUNT };
struct Checks {
    CheckState st[L_COUNT]{};
    wstring text[L_COUNT];
    int code[L_COUNT]{};  // exit code if this line blocks
    // force (/FORCE, tests and development): the GPU and HIP lines do not block.
    bool blocking(int i, bool force) const { return (st[i] == CK_FAIL || st[i] == CK_NA) && !(force && (i == L_GPU || i == L_HIP)); }
    bool ok(bool force) const { for (int i = 0; i < L_COUNT; ++i) if (blocking(i, force)) return false; return true; }
    int firstFailure(bool force, wstring& why) const {
        for (int i = 0; i < L_COUNT; ++i) if (blocking(i, force)) { why = text[i]; return code[i] ? code[i] : EXIT_REFUSED; }
        return EXIT_OK;
    }
};
struct SystemInfo { WinInfo win; GpuInfo gpu; HipInfo hip; };

WinInfo WindowsInfo();
bool IsRdna4Name(const wstring& name);
// Under Wine: the name DetectGpu gives an RDNA 4 card from Linux's PCI ids when Wine's DXGI names it otherwise; empty
// when Linux reports no RDNA 4 card (or outside Wine, where Z:\sys does not exist). drmRoot: the folder holding card0,
// card1, ... (Linux's /sys/class/drm; tests point it at a made-up folder). Every card is looked at, in number order.
wstring LinuxRdna4Name(const wstring& drmRoot = L"Z:\\sys\\class\\drm");
GpuInfo DetectGpu();
bool VersionLess(const wstring& a, const wstring& b);
std::vector<wstring> HipCandidates();
string FindArchName(const uint8_t* p, size_t n);
bool IsSupportedArch(const string& a);
int HipProbe(Context& ctx, const wstring& dll);
HipInfo CheckHip(Context& ctx);
// The HIP probes CheckHip runs (child processes of this exe, writing to this process's log) are in one job that ends them
// when this process ends, however it ends. StopHipProbes ends the running ones now and waits a moment for them to go: the
// window calls it as it closes, before it deletes the first run's temporary log, so no probe writes it again or keeps a
// downloaded exe open (a HIP runtime that hangs).
void StopHipProbes();
bool GameRunning(const wstring& exe);
// rc6: what the running-game watch (app/ui_play.cpp) makes of a game Play started, `sinceMs` after its Play. Its store may close
// and start it again at once (the first kPlayGraceMs: starting, not looked at), and a store's own launcher may start it only
// when the player says so there (Cyberpunk 2077 through Steam: REDlauncher, 28 s after Play): not seen running yet, it is
// starting for up to kPlayStartWaitMs. Once seen running, not running is closed (its entry, Last run included, read again).
enum class PlayWatch { Starting, Running, Closed };
constexpr unsigned long long kPlayGraceMs = 15000, kPlayStartWaitMs = 180000;
PlayWatch WatchPlayed(unsigned long long sinceMs, bool seenRunning, bool runningNow);
bool IsElevated();
DWORD WriteTest(const wstring& dir);
// Anti-cheat (see checks.cpp): LibraryContainer = a folder that holds games (store library, games folder, drive root);
// GameTopFolder = the game's own top folder above its exe; FindAntiCheat = the marker's full path or a denied game's
// reason, empty for none; AntiCheatMessage = the refusal text ("Anti-cheat found: <full path> - ...").
bool LibraryContainer(const wstring& dir);
wstring GameTopFolder(const wstring& exeDir);
wstring FindAntiCheat(const wstring& exe);
wstring AntiCheatMessage(const wstring& found);
Checks RunChecks(Context& ctx, const GameTarget& t, SystemInfo& sys, const ModelState* model, bool force);
}  // namespace wg
