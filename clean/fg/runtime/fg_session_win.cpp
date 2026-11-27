// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96, F7a): the Windows presenter source of an FG session. See fg_session_win.h.
#include "fg_session_win.h"

#include <utility>

namespace wgfg {

bool FgPresenterChoiceOfMode(uint32_t presenter_mode, FgPresenterChoice* out) {
    if (!out) return false;
    switch (presenter_mode) {
        case WG_FG_PRESENTER_AUTO: *out = FgPresenterChoice::Auto; return true;
        case WG_FG_PRESENTER_HWFQ: *out = FgPresenterChoice::WindowsCompositionHwfq; return true;
        case WG_FG_PRESENTER_DXGI_TEARING: *out = FgPresenterChoice::DxgiFlipTearing; return true;
        case WG_FG_PRESENTER_DXGI_VSYNC: *out = FgPresenterChoice::DxgiFlipVsync; return true;
        default: return false;
    }
}

FgStatus FgWinPresenterSource::CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out,
                                               std::string* why) {
    if (!out) return FgStatus::InvalidArgument;
    out->reset();
    selection_ = FgPresenterSelection();
    FgPresenterFactoryDesc fd = base_;
    if (!FgPresenterChoiceOfMode(desc.presenter_mode, &fd.choice)) {
        last_ = FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::UnknownChoice);
        if (why) *why = FgWinRefusalName(last_.refusal);
        return last_.status;
    }
    fd.width = desc.output_width;
    fd.height = desc.output_height;
    fd.format = static_cast<DXGI_FORMAT>(desc.output_format);
    fd.max_generated = desc.max_generated;
    FgCreatedPresenter created;
    last_ = FgCreatePresenter(fd, &created);
    selection_ = created.selection;
    if (!last_.Ok() || !created.presenter) {
        if (last_.Ok()) last_ = FgWinFail(FgStatus::Unsupported, FgWinRefusal::Probe);
        if (why) *why = std::string(FgPresenterChoiceName(fd.choice)) + ": " + FgWinRefusalName(last_.refusal);
        return last_.status;
    }
    *out = std::move(created.presenter);
    return FgStatus::Ok;
}

}  // namespace wgfg
