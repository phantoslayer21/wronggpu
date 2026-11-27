// SPDX-License-Identifier: GPL-3.0-only
// Frame generation's files in an install (wg_core/fg.h, 0.7.0): WrongGPU-FG.asi is WrongGPU's own plugin, never another setup's
// (coexist.h ForeignPlugins: it never keeps WrongGPU's Ray Reconstruction out, recorded or not), where an install puts the
// plugin and its kernels (beside WrongGPU's plugin, the kernels in wronggpu-setup\fg-dxil there), the record's frame generation
// entry, the frame generation library check, OptiScaler's own frame generation in OptiScaler.ini, and (a build with the frame
// generation choice) the state folder's frame generation keys folded back into the profile. Synthetic files in the temp
// folder only. Written for this project. Run by ctest ("fg-files").
#include <iostream>
#include <stdexcept>

#include "wg_core/coexist.h"
#include "wg_core/fg.h"
#include "wg_core/launcher.h"
#include "wg_core/profile.h"
#include "wg_core/record.h"

using namespace wg;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}
wstring TempFolder() {
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    wstring dir = Join(tmp, L"fg-files-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return dir;
}
void Put(const wstring& path, const string& text) {
    EnsureDir(Parent(path));
    WriteFileAtomic(path, text.data(), text.size());
}
bool Has(const std::vector<wstring>& v, const wstring& x) {
    for (auto& s : v) if (IEq(s, x)) return true;
    return false;
}
}  // namespace

int main() {
    const wstring tmp = TempFolder();
    try {
        // ---- the plugin's identity --------------------------------------------------------------------------------------
        require(IsFgPlugin(L"plugins\\WrongGPU-FG.asi") && IsFgPlugin(L"wronggpu-plugins\\wronggpu-fg.asi") && IsFgPlugin(L"WrongGPU-FG.asi"),
                "WrongGPU-FG.asi is frame generation's plugin in any folder, any case");
        require(!IsFgPlugin(L"plugins\\WrongGPU.asi") && !IsFgPlugin(L"plugins\\WrongGPU-RR.asi") && !IsFgPlugin(L"plugins\\OtherFG.asi"),
                "WrongGPU.asi, WrongGPU-RR.asi and another setup's plugin are not");

        // ---- ForeignPlugins: WrongGPU-FG.asi is WrongGPU's own, recorded or not (0.7.0 coexistence) ---------------------------
        {
            const wstring exeDir = Join(tmp, L"game");
            Put(Join(exeDir, L"plugins\\WrongGPU-FG.asi"), "MZ synthetic frame generation plugin stand-in");
            Put(Join(exeDir, L"plugins\\OtherSetup.asi"), "MZ synthetic plugin of another setup");
            Put(Join(exeDir, L"OptiScaler\\plugins\\WrongGPU-FG.asi"), "MZ synthetic frame generation plugin stand-in");
            const std::vector<wstring> none = ForeignPlugins(exeDir, nullptr);
            require(Has(none, L"plugins\\OtherSetup.asi") && !Has(none, L"plugins\\WrongGPU-FG.asi") && !Has(none, L"OptiScaler\\plugins\\WrongGPU-FG.asi") &&
                        none.size() == 1,
                    "ForeignPlugins without a record: another setup's plugin is listed, WrongGPU-FG.asi (either plugin folder) never");
            Record m;
            Entry e;
            e.action = L"copy";
            e.rel = L"plugins\\WrongGPU-FG.asi";
            e.inst = L"0000000000000000000000000000000000000000000000000000000000000000";  // (bytes changed since: still not foreign)
            m.e.push_back(e);
            const std::vector<wstring> rec = ForeignPlugins(exeDir, &m);
            require(!Has(rec, L"plugins\\WrongGPU-FG.asi") && Has(rec, L"plugins\\OtherSetup.asi"),
                    "ForeignPlugins with a record: WrongGPU-FG.asi is not foreign whatever its hash; the other setup's plugin still is");
            const PluginPlan plan = PlanPluginsAside(exeDir, exeDir, ForeignPlugins(exeDir, nullptr), {}, {});
            require(!Has(plan.blocking, L"plugins\\WrongGPU-FG.asi") && !Has(plan.moveAside, L"plugins\\WrongGPU-FG.asi"),
                    "the plugins plan never blocks on WrongGPU-FG.asi nor moves it aside");
            require(!Has(OurPluginsFromOtherPackages(exeDir, nullptr), L"plugins\\WrongGPU-FG.asi"),
                    "WrongGPU-FG.asi is not an RR plugin of another package either");
            DeleteFileW(Join(exeDir, L"plugins\\OtherSetup.asi").c_str());
            require(ForeignPlugins(exeDir, nullptr).empty(), "only WrongGPU-FG.asi in the plugin folders: nothing foreign, nothing keeps RR out");
        }

        // ---- where an install puts each file --------------------------------------------------------------------------------
        require(FgInstallRel(L"plugins\\WrongGPU.asi", L"plugins\\WrongGPU-FG.asi") == L"plugins\\WrongGPU-FG.asi" &&
                    FgInstallRel(L"plugins\\WrongGPU.asi", L"fg-dxil\\fg_exec_pack.dxil") == L"plugins\\wronggpu-setup\\fg-dxil\\fg_exec_pack.dxil",
                "plugins\\: the plugin beside WrongGPU.asi, its kernels in plugins\\wronggpu-setup\\fg-dxil (where the plugin reads them)");
        require(FgInstallRel(L"wronggpu-plugins\\WrongGPU.asi", L"plugins\\WrongGPU-FG.asi") == L"wronggpu-plugins\\WrongGPU-FG.asi" &&
                    FgInstallRel(L"wronggpu-plugins\\WrongGPU.asi", L"fg-dxil\\mock.dxil") == L"wronggpu-plugins\\wronggpu-setup\\fg-dxil\\mock.dxil",
                "the plugin's own folder (Ultimate ASI Loader): the same layout there");
        require(FgInstallRel(L"plugins\\WrongGPU.asi", L"fg-dxil\\manifest.json").empty() && FgInstallRel(L"plugins\\WrongGPU.asi", L"other\\x.dxil").empty() &&
                    FgInstallRel(L"plugins\\WrongGPU.asi", L"fg-dxil\\.dxil").empty() && FgInstallRel(L"plugins\\WrongGPU.asi", L"WrongGPU-FG.asi").empty(),
                "anything else: no place (never installed)");

        // ---- the record ------------------------------------------------------------------------------------------------------
        {
            Record m;
            require(!RecordHasFg(m), "an empty record has no frame generation files");
            Entry e;
            e.action = L"copy";
            e.rel = L"plugins\\wronggpu-setup\\fg-dxil\\mock.dxil";
            m.e.push_back(e);
            require(!RecordHasFg(m), "kernels alone are not the plugin");
            e.action = L"disable";
            e.rel = L"plugins\\WrongGPU-FG.asi";
            m.e.push_back(e);
            require(!RecordHasFg(m), "a disable entry is not an install of it");
            e.action = L"copy";
            m.e.push_back(e);
            require(RecordHasFg(m), "a copy entry of WrongGPU-FG.asi: the install placed frame generation's files");
        }

        // ---- the game's frame generation library --------------------------------------------------------------------------
        {
            const wstring root = Join(tmp, L"fg-game"), exeDir = Join(root, L"bin\\x64");
            Put(Join(exeDir, L"Game.exe"), "mock game exe");
            require(!GameHasFgLibrary(exeDir, exeDir, root), "no nvngx_dlssg.dll: no frame generation library");
            Put(Join(root, L"engine\\plugins\\streamline\\nvngx_dlssg.dll"), "MZ synthetic stand-in without a version resource");
            require(GameHasFgLibrary(exeDir, exeDir, root) && !GameHasFgLibrary(exeDir, exeDir, L""),
                    "a frame generation library anywhere in the game's root (a sub-folder): found; only under the exe's folder: not there");
            require(GameHasFgLibrary(L"", L"", root), "the root alone is enough");
        }

        // ---- OptiScaler's own frame generation ----------------------------------------------------------------------------
        require(OptiFgConflictText("").empty() && OptiFgConflictText("[FrameGen]\r\nFGInput=auto\r\nFGOutput=nofg\r\nEnabled=false\r\n").empty() &&
                    OptiFgConflictText("[FrameGen]\nFGType=auto\n").empty() && OptiFgConflictText("[Upscalers]\nFGInput=dlssg\n").empty(),
                "OptiScaler's defaults (auto, nofg, Enabled=false) and keys of another section: nothing in the way");
        {
            const wstring w = OptiFgConflictText("[FrameGen]\nFGInput=dlssg\nFGOutput=fsrfg\n");
            require(w.find(L"FGInput=dlssg") != wstring::npos && w.find(L"FGOutput=fsrfg") != wstring::npos && w.find(L"OptiScaler's menu") != wstring::npos,
                    "OptiScaler's frame generation set: both keys named, with what to do");
            require(!OptiFgConflictText("[FrameGen]\nEnabled=true\n").empty() && !OptiFgConflictText("[FrameGen]\nFGType=optifg\n").empty(),
                    "Enabled=true, or the older FGType set: said too");
            const wstring dir = Join(tmp, L"opti");
            Put(Join(dir, L"OptiScaler.ini"), "[FrameGen]\r\nFGInput=nukems\r\n");
            require(OptiFgConflict(dir).find(L"FGInput=nukems") != wstring::npos && OptiFgConflict(Join(tmp, L"no-such")).empty(),
                    "OptiFgConflict reads the game's OptiScaler.ini (none: nothing)");
        }

#if WRONGGPU_FG
        // ---- outside changes in the state folder folded into the profile (FoldFgFromGame) --------------------------------------
        {
            const wstring dir = Join(tmp, L"fold");
            EnsureDir(Join(dir, L"wronggpu-setup"));
            Put(Join(dir, L"wronggpu-setup\\settings.ini"), "fg=on\r\nfg_preset=B\r\nfg_multiplier=3\r\nfg_presenter=dxgi_tearing\r\n");
            Profile p = DefaultProfile();
            require(FoldFgFromGame(dir, p) && p.get("fg") == "on" && p.get("fg_preset") == "B" && p.get("fg_multiplier") == "3" &&
                        p.get("fg_presenter") == "dxgi_tearing" && FgOptedIn(p),
                    "fg=on, B, 3, dxgi_tearing changed outside the launcher: all four in the profile");
            require(!FoldFgFromGame(dir, p), "folded again: nothing changes");
            Put(Join(dir, L"wronggpu-setup\\settings.ini"), "fg_preset=b\r\nfg_multiplier=7\r\nfg_presenter= hwfq \r\n");
            require(FoldFgFromGame(dir, p) && p.get("fg_preset") == "B" && p.get("fg_multiplier") == "3" && p.get("fg_presenter") == "hwfq",
                    "invalid spellings (b, 7) never taken; a valid one with spaces around it is");
            // frame generation's data: this test's own LOCALAPPDATA (never the PC's), first without it, then with a file there
            SetEnvironmentVariableW(L"LOCALAPPDATA", Join(tmp, L"localappdata").c_str());
            Put(Join(dir, L"wronggpu-setup\\settings.ini"), "fg=off\r\n");
            require(!FoldFgFromGame(dir, p) && p.get("fg") == "on",
                    "fg=off while frame generation's data is not on this PC: the launcher's own hold, the profile keeps fg=on");
            Put(Join(Join(ModelBaseDir(), L"FG"), L"model.wgcm"), "synthetic stand-in");
            require(FoldFgFromGame(dir, p) && p.get("fg") == "off", "fg=off with the data on this PC: the player's choice, taken");
        }
#endif
        RemoveTree(tmp);
        std::cout << "fg-files: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& x) {
        RemoveTree(tmp);
        std::cerr << "FAILED after " << checks << " checks: " << x.what() << "\n";
        return 1;
    } catch (const SetupError& x) {
        RemoveTree(tmp);
        std::wcerr << L"FAILED after " << checks << L" checks: " << x.msg << L"\n";
        return 1;
    }
}
