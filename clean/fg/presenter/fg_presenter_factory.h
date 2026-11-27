// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Presenter back-end selection: an explicit choice (or Auto), a capability probe that reports exactly what the
// system supports, and creation that refuses with a reason instead of falling back.
//
//  - WindowsCompositionHwfq: fg_win_comp_presenter.h. Needs the adapter's hardware flip queue
//    (D3DKMT_WDDM_3_0_CAPS.HwFlipQueueEnabled), IsPresentationSupportedWithIndependentFlip and
//    D3D11_FEATURE_DISPLAYABLE.
//  - DxgiFlipTearing: fg_dxgi_presenter.h, Present(0, DXGI_PRESENT_ALLOW_TEARING) at each target. Needs
//    DXGI_FEATURE_PRESENT_ALLOW_TEARING.
//  - DxgiFlipVsync: fg_dxgi_presenter.h, tear-free Present(1, 0) on whole-refresh cadences.
//  - Auto picks one by a fixed rule from the probe and the display description (variable refresh with tearing:
//    DxgiFlipTearing; fixed refresh: WindowsCompositionHwfq when every one of its capabilities is there, else
//    DxgiFlipVsync; variable refresh without tearing: WindowsCompositionHwfq) and reports which and why. It is a
//    selection made once before creation: if the chosen back end then fails to create, the error is returned and no
//    other back end is tried.
#include <memory>

#include "fg_dxgi_presenter.h"
#include "fg_win_comp_presenter.h"

namespace wgfg {

enum class FgPresenterChoice : std::uint8_t {
    Auto = 0,
    WindowsCompositionHwfq,
    DxgiFlipTearing,
    DxgiFlipVsync,
};
const char* FgPresenterChoiceName(FgPresenterChoice c);

struct FgPresenterCapabilities {
    bool adapter_probed = false;
    FgAdapterFlipCaps adapter;
    bool presentation_probed = false;
    bool presentation_supported = false;      // IPresentationFactory::IsPresentationSupported
    bool independent_flip_supported = false;  // IPresentationFactory::IsPresentationSupportedWithIndependentFlip
    bool d3d11_displayable = false;           // D3D11_FEATURE_DISPLAYABLE.DisplayableTexture
    bool tearing_probed = false;
    bool tearing_supported = false;           // DXGI_FEATURE_PRESENT_ALLOW_TEARING
};

// Read only, no device: the adapter's D3DKMT capability bits and DXGI tearing support.
FgWinResult FgProbeAdapterCapabilities(LUID luid, FgPresenterCapabilities* caps);
// Everything: the read-only part plus a temporary Direct3D 11 device on the D3D12 device's adapter for the
// presentation factory and the displayable feature (created and released here: this touches the GPU driver).
FgWinResult FgProbePresenterCapabilities(ID3D12Device* device, FgPresenterCapabilities* caps);

struct FgDisplayDesc {
    bool variable_refresh = false;  // the output runs variable refresh for this window
    FgTicks refresh_period = 0;     // fixed refresh period (0: measured from the compositor by the back end)
};

struct FgPresenterSelection {
    FgPresenterChoice requested = FgPresenterChoice::Auto;
    FgPresenterChoice chosen = FgPresenterChoice::Auto;  // Auto only when refused
    FgStatus status = FgStatus::Unsupported;
    FgWinRefusal refusal = FgWinRefusal::Probe;
};

// Pure: the decision from a probe result. Explicit choices are checked and refused, never replaced.
FgPresenterSelection FgSelectPresenter(FgPresenterChoice requested, const FgPresenterCapabilities& caps,
                                       const FgDisplayDesc& display);

struct FgPresenterFactoryDesc {
    FgPresenterChoice choice = FgPresenterChoice::Auto;
    FgDisplayDesc display;
    const FgPresenterCapabilities* caps = nullptr;  // null: probed here (needs device)
    ID3D12Device* device = nullptr;
    // composition
    HWND hwnd = nullptr;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
    // DXGI
    ID3D12CommandQueue* queue = nullptr;
    IDXGISwapChain4* swap_chain = nullptr;
    std::uint32_t max_generated = kFgMaxGenerated;
    std::uint32_t slot_count = 0;
};

struct FgCreatedPresenter {
    std::unique_ptr<IFgPresenter> presenter;
    FgWinCompPresenter* composition = nullptr;  // typed view of presenter (or null)
    FgDxgiPresenter* dxgi = nullptr;            // typed view of presenter (or null)
    FgPresenterSelection selection;
};

// Selects and creates. On any refusal out->presenter stays null and the result names the reason.
FgWinResult FgCreatePresenter(const FgPresenterFactoryDesc& desc, FgCreatedPresenter* out);

}  // namespace wgfg
