// SPDX-License-Identifier: MIT
// WrongGPU installer: the state the wizard (app/wizard.cpp) and the command line (app/main.cpp) share. The engine is
// wg_core (its wg::Context is App::ctx), the widgets are wg_ui. Moved from setup.cpp. Written for this project.
#pragma once
#include <map>
#include <vector>

#include "wg_core/install.h"
#include "wg_core/logs.h"
#include "wg_ui/kit.h"
#include "wizard_policy.h"

namespace wg::app {

enum PageId { PG_WELCOME, PG_GAME, PG_REVIEW, PG_PROGRESS, PG_FINISH, PG_UCONFIRM, PG_ROUTE, PG_COUNT };

struct App {
    Context ctx;  // the engine's state (wg_core): log file, converter map, OptiScaler package, embedded payload
    HINSTANCE hinst{}; HWND hwnd{};
    ui::Kit kit;  // the widget kit (wg_ui): DPI scale, fonts, brushes, control backgrounds, hover, wordmark
    bool hipCheck = false;
    bool uninstall = false, collect = false, list = false, fsr4 = false, noScan = false, force = false, silent = false, go = false, elevatedRelaunch = false;
    bool experimentalOk = false, moveMods = false, removeOld = false;
    wstring input, shotDir, outZip, userDll;
    wg_wizard::Route route = wg_wizard::Route::OptiScaler;
    OptiPackage optiChoice; // retained when the user visits the advanced direct route
    ModelState model;
    int modelGeneration = 0;
    wstring modelFor;  // the target the current model search was started for
    PageId page = PG_WELCOME;
    std::vector<HWND> ctl[PG_COUNT];
    HWND btnBack{}, btnNext{}, btnCancel{}, btnCollect{}, btnModel{}, btnOpti{}, edPath{}, lbGames{}, stStatus{}, pbar{}, stStep{}, edLog{}, edReview{}, chkConsent{}, chkConflict{}, stFHead{}, stFBody{}, chkOpen{};
    HWND chkPresetM{}, chkPresetK{}, edChecks{}, chkDirect{}, stRoute{}, btnOptiFolder{};
    SystemInfo sys; GameTarget target; Checks checks;
    std::map<wstring, GameTarget> cache;  // InspectGame results by exe path
    std::vector<Found> found;
    std::vector<StoreGame> libraries;
    bool scanning = false, working = false, finishOk = false;
    wstring finishTitle, finishSub;
};
extern App g;  // app/main.cpp

int RunSilent();      // /S (app/main.cpp)
int RunWizard();      // the wizard window, or the "logs collected" message of /COLLECT (app/wizard.cpp)
bool NeedsConsent();  // an experimental game: the player confirms (app/wizard.cpp)

}  // namespace wg::app
