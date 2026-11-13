// SPDX-License-Identifier: MIT
#pragma once
#include <string>

// Preset K in this installer (CMake option WRONGGPU_PRESET_K, default ON; OFF: Model M only, no Preset K wording).
#ifndef WRONGGPU_PRESET_K
#define WRONGGPU_PRESET_K 1
#endif
// The direct FSR route in this installer (CMake option WRONGGPU_DIRECT_ROUTE, default ON; OFF: OptiScaler is required:
// no method choice and no wording of the direct route, /DIRECTFSR is an unknown option; setup_route.h has the rest).
#ifndef WRONGGPU_DIRECT_ROUTE
#define WRONGGPU_DIRECT_ROUTE 1
#endif

// UI policy only. Silent installations keep their existing explicit command-line contract.
namespace wg_wizard {
enum class Route { OptiScaler, DirectFsr };
constexpr bool kDirectRoute = WRONGGPU_DIRECT_ROUTE != 0;
// Whether `route` is the direct route in an installer that has it (never in one without).
inline bool IsDirect(Route route) {
#if WRONGGPU_DIRECT_ROUTE
    return route == Route::DirectFsr;
#else
    (void)route;
    return false;
#endif
}
inline bool RouteReady(Route route, bool packageReady) {
    return IsDirect(route) || packageReady;
}
inline std::wstring RouteArguments(Route route, const std::wstring& source) {
    // Windows paths cannot contain a quote. Reject rather than constructing a broken relaunch.
#if WRONGGPU_DIRECT_ROUTE
    if (route == Route::DirectFsr) return L" /DIRECTFSR";
#else
    if (route != Route::OptiScaler) return L"";
#endif
    if (source.empty() || source.find(L'"') != std::wstring::npos) return L"";
    return L" \"/OPTISCALER=" + source + L"\"";
}
inline std::wstring PlayInstructions(Route route, bool presetK = false) {
#if WRONGGPU_DIRECT_ROUTE
    if (route != Route::OptiScaler)
        return L"1. Start the game in DirectX 12 mode.\r\n"
               L"2. Choose AMD FSR 3 or newer in the game's graphics settings, then pick a quality mode.\r\n"
               L"3. The WrongGPU panel shows the active upscaler. Insert toggles it; Home compares with AMD FSR.\r\n"
               L"4. If neural upscaling fails, AMD FSR may take over. Fallback is not neural upscaling running.\r\n";
#else
    (void)route;
#endif
#if !WRONGGPU_PRESET_K
    (void)presetK;
    return L"1. Start the game in DirectX 12 mode.\r\n"
           L"2. Choose DLSS or XeSS in the game's graphics settings, then pick a quality mode.\r\n"
           L"3. Press Insert to open OptiScaler. Select WrongGPU Model M in its upscaler list.\r\n"
           L"4. Frame generation support depends on the game and OptiScaler configuration.\r\n";
#else
    return std::wstring( L"1. Start the game in DirectX 12 mode.\r\n"
           L"2. Choose DLSS or XeSS in the game's graphics settings, then pick a quality mode.\r\n"
           L"3. Press Insert to open OptiScaler. Select FSR ") + (presetK ? L"WrongGPU (Model K)" : L"WrongGPU (Model M)") + L" in its upscaler list.\r\n"
           L"   Switch between WrongGPU (Model M) and WrongGPU (Model K) there while playing.\r\n"
           L"   Each model needs its matching local model cache; setup prepares both when matching files are available.\r\n"
           L"4. Frame generation support depends on the game and OptiScaler configuration.\r\n";
#endif
}

// The wizard's texts that name the install method. Without the direct route there is one method: nothing is called
// recommended or advanced, and the texts of the direct route are not in the program.
#if WRONGGPU_DIRECT_ROUTE
inline const wchar_t* WelcomeIntro() { return L"Install WrongGPU through OptiScaler, then choose a quality mode in your game. Direct FSR installation is available as an advanced option."; }
inline const wchar_t* RouteStep() { return L"Install method"; }
inline const wchar_t* RouteTitle(Route route) { return route == Route::DirectFsr ? L"Direct FSR installation" : L"Set up OptiScaler"; }
inline const wchar_t* RouteSubtitle() { return L"OptiScaler is the recommended way to install WrongGPU"; }
inline const wchar_t* RouteBadge() { return L"RECOMMENDED"; }
inline const wchar_t* DirectChoice() { return L"Use direct FSR installation instead (advanced)"; }
inline const wchar_t* DirectCard() { return L"Replaces supported FSR 3 DLLs directly. Use this route only when you specifically want to install without OptiScaler."; }
inline const wchar_t* DirectSelected() { return L"Direct FSR selected. Only games with a supported FSR 3 DLL can use this route."; }
inline const wchar_t* GameSubtitle(Route route) { return route == Route::OptiScaler ? L"Install through OptiScaler; the game's FSR files stay in place" : L"Advanced direct installation: requires AMD FSR 3 or newer"; }
inline const wchar_t* ReviewMethod(Route route) { return route == Route::OptiScaler ? L"INSTALL METHOD: OptiScaler (recommended)\r\n\r\n" : L"INSTALL METHOD: Direct FSR (advanced)\r\n\r\n"; }
#else
inline const wchar_t* WelcomeIntro() { return L"Install WrongGPU through OptiScaler, then choose a quality mode in your game."; }
inline const wchar_t* RouteStep() { return L"OptiScaler"; }
inline const wchar_t* RouteTitle(Route) { return L"Set up OptiScaler"; }
inline const wchar_t* RouteSubtitle() { return L"WrongGPU is installed through OptiScaler"; }
inline const wchar_t* RouteBadge() { return L"REQUIRED"; }
inline const wchar_t* DirectChoice() { return L""; }
inline const wchar_t* DirectCard() { return L""; }
inline const wchar_t* DirectSelected() { return L""; }
inline const wchar_t* GameSubtitle(Route) { return L"Install through OptiScaler; the game's FSR files stay in place"; }
inline const wchar_t* ReviewMethod(Route) { return L"INSTALL METHOD: OptiScaler\r\n\r\n"; }
// What a silent install without /OPTISCALER= is told (exit 2, before anything is written).
inline const wchar_t* OptiRequired() { return L"OptiScaler is required: pass /OPTISCALER=<its release archive, folder or OptiScaler.dll>"; }
#endif
}  // namespace wg_wizard
