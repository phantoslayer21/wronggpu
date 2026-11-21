// SPDX-License-Identifier: MIT
#pragma once
// Frame generation (issue #96, F7a): the Windows presenter source of an FG session. It creates the back end the
// session's presenter mode names (fg_presenter_factory.h: Auto, the composition back end with the hardware flip queue,
// DXGI flip with tearing, DXGI flip with vsync) for the session's size and N range, and refuses with the factory's
// reason when that back end cannot be made. It never tries a second back end. Windows only.
#include <string>

#include "../presenter/fg_presenter_factory.h"
#include "fg_session.h"

namespace wgfg {

// The adapter contract's presenter mode (WgFgPresenterMode / fg_settings.h) as a factory choice; false outside the four.
bool FgPresenterChoiceOfMode(uint32_t presenter_mode, FgPresenterChoice* out);

class FgWinPresenterSource final : public IFgPresenterSource {
public:
    // `base` carries what the session description does not: the window, the device / queue / swap chain interfaces,
    // the display description, optional pre-probed capabilities and the slot count. Size, format, max_generated and
    // the choice are taken from the session description at creation.
    explicit FgWinPresenterSource(const FgPresenterFactoryDesc& base) : base_(base) {}

    FgStatus CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out, std::string* why) override;

    const FgWinResult& LastResult() const { return last_; }
    const FgPresenterSelection& LastSelection() const { return selection_; }

private:
    FgPresenterFactoryDesc base_;
    FgWinResult last_;
    FgPresenterSelection selection_;
};

}  // namespace wgfg
