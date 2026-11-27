// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation (issue #96): the network grid sizes of both networks, from the frame (output) size Wo x Ho.
// One create-time size table; all divisions are integer divisions.
//
// net1 (17-blob U-Net on a 2 x 2-folded grid):
//   d = 4 if Ho > 1440 else 2;  Wd x Hd = floor(Wo / d) x floor(Ho / d);
//   W0 = 32 * ceil(floor(Wd / 2) / 32),  H0 = 32 * ceil(floor(Hd / 2) / 32)   (cells; the pixel grid is 2 W0 x 2 H0);
//   centring pad in cells: left = floor((W0 - floor(Wd / 2)) / 2), right = W0 - left - floor(Wd / 2); same for y.
// net2: k = the smallest power of two with floor(Ho / k) <= 480;  W2 = 16 * ceil(Wo / (16 k)), H2 likewise;
//   right / bottom pad in frame pixels = 16 k * ceil(Wo / (16 k)) - Wo (likewise Ho); left = top = 0.
// Accepted frame sizes: 4 <= Wo <= 8192 and 4 <= Ho <= 7680.
#include <cstdint>

namespace wgfg {

inline constexpr uint32_t kFgMinFrame = 4, kFgMaxFrameW = 8192, kFgMaxFrameH = 7680;

constexpr bool FgOutputSizeSupported(uint32_t wo, uint32_t ho) {
    return wo >= kFgMinFrame && wo <= kFgMaxFrameW && ho >= kFgMinFrame && ho <= kFgMaxFrameH;
}

struct FgNet1Grid {
    uint32_t d = 0;                 // downsample divisor
    uint32_t wd = 0, hd = 0;        // downsampled frame
    uint32_t w0 = 0, h0 = 0;        // level-0 cells (multiples of 32)
    uint32_t pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;  // in cells
};

struct FgNet2Grid {
    uint32_t k = 0;                 // power-of-two divisor
    uint32_t w2 = 0, h2 = 0;        // network grid (multiples of 16)
    uint32_t pad_right = 0, pad_bottom = 0;  // in frame pixels
};

struct FgGrids {
    FgNet1Grid net1;
    FgNet2Grid net2;
};

constexpr uint32_t FgCeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

constexpr FgNet1Grid FgNet1GridFor(uint32_t wo, uint32_t ho) {
    FgNet1Grid g{};
    g.d = ho > 1440 ? 4u : 2u;
    g.wd = wo / g.d;
    g.hd = ho / g.d;
    const uint32_t hw = g.wd / 2, hh = g.hd / 2;
    g.w0 = 32 * FgCeilDiv(hw, 32);
    g.h0 = 32 * FgCeilDiv(hh, 32);
    g.pad_left = (g.w0 - hw) / 2;
    g.pad_right = g.w0 - g.pad_left - hw;
    g.pad_top = (g.h0 - hh) / 2;
    g.pad_bottom = g.h0 - g.pad_top - hh;
    return g;
}

constexpr FgNet2Grid FgNet2GridFor(uint32_t wo, uint32_t ho) {
    FgNet2Grid g{};
    uint32_t k = 1;
    while (ho / k > 480) k *= 2;
    g.k = k;
    const uint32_t cw = FgCeilDiv(wo, 16 * k), ch = FgCeilDiv(ho, 16 * k);
    g.w2 = 16 * cw;
    g.h2 = 16 * ch;
    g.pad_right = 16 * k * cw - wo;
    g.pad_bottom = 16 * k * ch - ho;
    return g;
}

// Both grids; false (out untouched) for a frame size outside the accepted range.
constexpr bool FgGridsFor(uint32_t wo, uint32_t ho, FgGrids* out) {
    if (!FgOutputSizeSupported(wo, ho)) return false;
    out->net1 = FgNet1GridFor(wo, ho);
    out->net2 = FgNet2GridFor(wo, ho);
    return true;
}

}  // namespace wgfg
