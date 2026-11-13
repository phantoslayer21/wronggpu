// SPDX-License-Identifier: MIT
#include "../src/wizard_policy.h"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace wg_wizard;
    int checks = 0;
    auto require = [&](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
        ++checks;
    };
    require(!RouteReady(Route::OptiScaler, false), "Missing or rejected package must block the default route");
    require(RouteReady(Route::OptiScaler, true), "Accepted package allows game selection");
#if WRONGGPU_DIRECT_ROUTE
    require(RouteReady(Route::DirectFsr, false), "Advanced route does not require OptiScaler");
    require(RouteReady(Route::DirectFsr, true), "Retained package must not block direct FSR");
#else  // an installer without the direct route (CMake WRONGGPU_DIRECT_ROUTE=OFF): OptiScaler is required
    require(!kDirectRoute && !IsDirect(Route::DirectFsr), "There is no direct route");
    require(!RouteReady(Route::DirectFsr, false), "Nothing continues without an accepted OptiScaler package");
#endif
    require(!RouteReady(Route::OptiScaler, false), "Returning to default must restore the missing-package gate");
    const std::wstring archive = L"C:\\Users\\Player Name\\Downloads\\OptiScaler release.7z";
    require(RouteArguments(Route::OptiScaler, archive) == L" \"/OPTISCALER=" + archive + L"\"", "Elevation must carry the exact package path as one argument");
    const std::wstring folder = L"K:\\Jeux\\Cr\u00e9ation\\OptiScaler";
    require(RouteArguments(Route::OptiScaler, folder).find(folder) != std::wstring::npos, "Unicode folder survives elevation");
    require(RouteArguments(Route::OptiScaler, L"").empty(), "No package must not become a direct install on elevation");
    require(RouteArguments(Route::OptiScaler, L"C:\\bad\"path").empty(), "Reject invalid quote-bearing paths");
#if WRONGGPU_DIRECT_ROUTE
    require(RouteArguments(Route::DirectFsr, archive) == L" /DIRECTFSR", "Direct route must not carry the remembered package");
#else
    require(RouteArguments(Route::DirectFsr, archive).empty(), "No relaunch asks for a direct install");
#endif
    require(PlayInstructions(Route::OptiScaler).find(L"DLSS or XeSS") != std::wstring::npos, "Hosted finish page explains the input setting");
    require(PlayInstructions(Route::OptiScaler).find(L"Model M") != std::wstring::npos, "Hosted finish page names the selected provider");
#if WRONGGPU_PRESET_K
    require(PlayInstructions(Route::OptiScaler, true).find(L"WrongGPU (Model K)") != std::wstring::npos, "Preset K finish page names K");
    require(PlayInstructions(Route::OptiScaler, true).find(L"Select FSR WrongGPU (Model M)") == std::wstring::npos, "K must not instruct selection of M");
#else  // an installer without Preset K (CMake WRONGGPU_PRESET_K=OFF): no page names it
    for (Route route : {Route::OptiScaler, Route::DirectFsr})
        for (bool k : {false, true}) {
            const std::wstring text = PlayInstructions(route, k);
            require(text.find(L"Model K") == std::wstring::npos && text.find(L"Preset K") == std::wstring::npos, "No finish page names Preset K");
        }
    // the in-game entry by its one name, as OptiScaler's menu and the release notes have it (UI review #7)
    require(PlayInstructions(Route::OptiScaler, true).find(L"Select WrongGPU Model M") != std::wstring::npos, "The hosted finish page names WrongGPU Model M");
#endif
#if WRONGGPU_DIRECT_ROUTE
    require(PlayInstructions(Route::DirectFsr).find(L"Choose AMD FSR 3") != std::wstring::npos, "Direct finish page explains the FSR setting");
    require(PlayInstructions(Route::DirectFsr).find(L"Choose DLSS") == std::wstring::npos, "Direct finish page must not describe the hosted route");
    require(std::wstring(ReviewMethod(Route::DirectFsr)).find(L"Direct FSR") != std::wstring::npos && std::wstring(ReviewMethod(Route::OptiScaler)).find(L"OptiScaler") != std::wstring::npos,
            "The review names the selected method");
    require(std::wstring(RouteTitle(Route::DirectFsr)) != RouteTitle(Route::OptiScaler) && std::wstring(GameSubtitle(Route::DirectFsr)) != GameSubtitle(Route::OptiScaler),
            "The method and game pages follow the selected method");
#else  // no page offers, names or describes another method than OptiScaler
    for (Route route : {Route::OptiScaler, Route::DirectFsr}) {
        require(PlayInstructions(route) == PlayInstructions(Route::OptiScaler) && PlayInstructions(route).find(L"Choose DLSS or XeSS") != std::wstring::npos,
                "The finish page is the hosted one");
        for (const std::wstring text : {std::wstring(WelcomeIntro()), std::wstring(RouteStep()), std::wstring(RouteTitle(route)), std::wstring(RouteSubtitle()),
                                        std::wstring(RouteBadge()), std::wstring(DirectChoice()), std::wstring(DirectCard()), std::wstring(DirectSelected()),
                                        std::wstring(GameSubtitle(route)), std::wstring(ReviewMethod(route)), PlayInstructions(route)})
            for (const wchar_t* word : {L"irect FSR", L"irect inst", L"irectly", L"dvanced", L"ecommended", L"RECOMMENDED", L"AMD FSR 3", L"without OptiScaler", L"method"})
                require(text.find(word) == std::wstring::npos, "No wizard text offers, names or describes the direct route or a choice of method");
    }
    require(std::wstring(DirectChoice()).empty() && std::wstring(DirectCard()).empty() && std::wstring(DirectSelected()).empty(), "The direct route has no texts");
    require(std::wstring(ReviewMethod(Route::OptiScaler)).find(L"OptiScaler") != std::wstring::npos && std::wstring(RouteTitle(Route::OptiScaler)).find(L"OptiScaler") != std::wstring::npos,
            "The method page and the review name OptiScaler");
    require(std::wstring(OptiRequired()).find(L"OptiScaler is required: pass /OPTISCALER=<") == 0, "The requirement message of a silent install");
#endif
    std::cout << checks << " wizard policy checks passed\n";
}
