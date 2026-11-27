// SPDX-License-Identifier: GPL-3.0-only
// Presenter back-end selection: see fg_presenter_factory.h.
#include "fg_presenter_factory.h"

namespace wgfg {

using Microsoft::WRL::ComPtr;

const char* FgPresenterChoiceName(FgPresenterChoice c) {
    switch (c) {
        case FgPresenterChoice::Auto: return "auto";
        case FgPresenterChoice::WindowsCompositionHwfq: return "windows-composition-hwfq";
        case FgPresenterChoice::DxgiFlipTearing: return "dxgi-flip-tearing";
        case FgPresenterChoice::DxgiFlipVsync: return "dxgi-flip-vsync";
    }
    return "?";
}

namespace {

bool KnownChoice(FgPresenterChoice c) {
    return c == FgPresenterChoice::Auto || c == FgPresenterChoice::WindowsCompositionHwfq ||
           c == FgPresenterChoice::DxgiFlipTearing || c == FgPresenterChoice::DxgiFlipVsync;
}

// The first missing composition capability, in the order the back end checks them (None: all present).
FgWinRefusal CompositionMissing(const FgPresenterCapabilities& c) {
    if (!c.adapter_probed || !c.adapter.queried || !c.adapter.hw_flip_queue_enabled)
        return FgWinRefusal::NoHardwareFlipQueue;
    if (!c.presentation_probed) return FgWinRefusal::Probe;
    if (!c.presentation_supported) return FgWinRefusal::NoPresentationManager;
    if (!c.independent_flip_supported) return FgWinRefusal::NoIndependentFlip;
    if (!c.d3d11_displayable) return FgWinRefusal::NotDisplayable;
    return FgWinRefusal::None;
}

FgPresenterSelection Chosen(FgPresenterChoice requested, FgPresenterChoice chosen) {
    FgPresenterSelection s;
    s.requested = requested;
    s.chosen = chosen;
    s.status = FgStatus::Ok;
    s.refusal = FgWinRefusal::None;
    return s;
}

FgPresenterSelection Refused(FgPresenterChoice requested, FgStatus status, FgWinRefusal why) {
    FgPresenterSelection s;
    s.requested = requested;
    s.chosen = FgPresenterChoice::Auto;
    s.status = status;
    s.refusal = why;
    return s;
}

void ProbeTearing(FgPresenterCapabilities* caps) {
    ComPtr<IDXGIFactory5> f5;
    BOOL tearing = FALSE;
    if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f5)))) {
        caps->tearing_probed = true;
        if (FAILED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
            tearing = FALSE;
    }
    caps->tearing_supported = tearing != FALSE;
}

}  // namespace

FgWinResult FgProbeAdapterCapabilities(LUID luid, FgPresenterCapabilities* caps) {
    if (!caps) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::Probe);
    *caps = FgPresenterCapabilities();
    caps->adapter_probed = true;
    const FgWinResult r = FgQueryAdapterFlipCaps(luid, &caps->adapter);
    ProbeTearing(caps);
    return r;
}

FgWinResult FgProbePresenterCapabilities(ID3D12Device* device, FgPresenterCapabilities* caps) {
    if (!caps) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::Probe);
    *caps = FgPresenterCapabilities();
    if (!device) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
    const LUID luid = device->GetAdapterLuid();
    FgProbeAdapterCapabilities(luid, caps);  // its own failure is recorded in caps->adapter

    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, hr);
    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, hr);
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    ComPtr<ID3D11Device> dev;
    hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels, 2,
                           D3D11_SDK_VERSION, &dev, nullptr, nullptr);
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::D3D11DeviceFailed, hr);
    caps->presentation_probed = true;
    ComPtr<IPresentationFactory> pf;
    hr = FgDCompCreatePresentationFactory(dev.Get(), IID_PPV_ARGS(&pf));
    if (SUCCEEDED(hr)) {
        caps->presentation_supported = pf->IsPresentationSupported() != 0;
        caps->independent_flip_supported = pf->IsPresentationSupportedWithIndependentFlip() != 0;
    }
    ComPtr<ID3D11Device5> d5;
    D3D11_FEATURE_DATA_DISPLAYABLE disp = {};
    if (SUCCEEDED(dev.As(&d5)) && SUCCEEDED(d5->CheckFeatureSupport(D3D11_FEATURE_DISPLAYABLE, &disp, sizeof(disp))))
        caps->d3d11_displayable = disp.DisplayableTexture != FALSE;
    return FgWinOk();
}

FgPresenterSelection FgSelectPresenter(FgPresenterChoice requested, const FgPresenterCapabilities& caps,
                                       const FgDisplayDesc& display) {
    if (!KnownChoice(requested)) return Refused(requested, FgStatus::InvalidArgument, FgWinRefusal::UnknownChoice);
    if (display.refresh_period < 0) return Refused(requested, FgStatus::InvalidArgument, FgWinRefusal::RefreshUnknown);
    const FgWinRefusal comp = CompositionMissing(caps);
    const bool tearing = caps.tearing_probed && caps.tearing_supported;
    switch (requested) {
        case FgPresenterChoice::WindowsCompositionHwfq:
            return comp == FgWinRefusal::None ? Chosen(requested, requested)
                                              : Refused(requested, FgStatus::Unsupported, comp);
        case FgPresenterChoice::DxgiFlipTearing:
            return tearing ? Chosen(requested, requested)
                           : Refused(requested, FgStatus::Unsupported, FgWinRefusal::NoTearing);
        case FgPresenterChoice::DxgiFlipVsync:
            return Chosen(requested, requested);  // the back end refuses when it cannot find the refresh grid
        case FgPresenterChoice::Auto:
            break;
    }
    if (display.variable_refresh) {
        if (tearing) return Chosen(requested, FgPresenterChoice::DxgiFlipTearing);
        if (comp == FgWinRefusal::None) return Chosen(requested, FgPresenterChoice::WindowsCompositionHwfq);
        return Refused(requested, FgStatus::Unsupported, FgWinRefusal::NoTearing);
    }
    if (comp == FgWinRefusal::None) return Chosen(requested, FgPresenterChoice::WindowsCompositionHwfq);
    return Chosen(requested, FgPresenterChoice::DxgiFlipVsync);
}

FgWinResult FgCreatePresenter(const FgPresenterFactoryDesc& desc, FgCreatedPresenter* out) {
    if (!out) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::None);
    out->presenter.reset();
    out->composition = nullptr;
    out->dxgi = nullptr;
    out->selection = FgPresenterSelection();
    out->selection.requested = desc.choice;
    if (!KnownChoice(desc.choice)) {
        out->selection = Refused(desc.choice, FgStatus::InvalidArgument, FgWinRefusal::UnknownChoice);
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::UnknownChoice);
    }
    FgPresenterCapabilities probed;
    const FgPresenterCapabilities* caps = desc.caps;
    if (!caps) {
        if (!desc.device) {
            out->selection = Refused(desc.choice, FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
            return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
        }
        const FgWinResult pr = FgProbePresenterCapabilities(desc.device, &probed);
        (void)pr;  // a failed part is visible in the capability bits; the selection names what is missing
        caps = &probed;
    }
    out->selection = FgSelectPresenter(desc.choice, *caps, desc.display);
    if (out->selection.status != FgStatus::Ok)
        return FgWinFail(out->selection.status, out->selection.refusal);

    if (out->selection.chosen == FgPresenterChoice::WindowsCompositionHwfq) {
        FgCompPresenterDesc cd;
        cd.device = desc.device;
        cd.hwnd = desc.hwnd;
        cd.width = desc.width;
        cd.height = desc.height;
        cd.format = desc.format;
        cd.hdr = desc.hdr;
        cd.variable_refresh = desc.display.variable_refresh;
        cd.refresh_period = desc.display.refresh_period;
        cd.max_generated = desc.max_generated;
        cd.slot_count = desc.slot_count;
        std::unique_ptr<FgWinCompPresenter> p;
        const FgWinResult r = FgWinCompPresenter::Create(cd, &p);
        if (!r.Ok()) return r;  // no other back end is tried
        out->composition = p.get();
        out->presenter = std::move(p);
        return FgWinOk();
    }
    FgDxgiPresenterDesc dd;
    dd.device = desc.device;
    dd.queue = desc.queue;
    dd.swap_chain = desc.swap_chain;
    dd.pacing = out->selection.chosen == FgPresenterChoice::DxgiFlipTearing ? FgDxgiPacing::Tearing
                                                                             : FgDxgiPacing::Vsync;
    dd.refresh_period = desc.display.refresh_period;
    dd.max_generated = desc.max_generated;
    dd.slot_count = desc.slot_count;
    std::unique_ptr<FgDxgiPresenter> p;
    const FgWinResult r = FgDxgiPresenter::Create(dd, &p);
    if (!r.Ok()) return r;  // no other back end is tried
    out->dxgi = p.get();
    out->presenter = std::move(p);
    return FgWinOk();
}

}  // namespace wgfg
