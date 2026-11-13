// SPDX-License-Identifier: MIT
// Format classes of spec 62.5. Numbers: the public DXGI_FORMAT enum of the Windows SDK (dxgiformat.h);
// tests/test_proxy_core.cpp checks this table against the SDK's named constants.
#include "formats.h"

namespace wgrr {

namespace {
enum : uint32_t { F = WG_FMT_FLOAT, U = WG_FMT_UNORM, S = WG_FMT_SNORM, R = WG_FMT_SRGB, X = WG_FMT_UNSUPPORTED,
                  DF = WG_FMT_DEPTH_FLOAT, DU = WG_FMT_DEPTH_UNORM };

struct Entry { uint32_t cls; uint32_t depth_cls; };  // class for ordinary inputs, class for depth inputs

// Index = DXGI format number, 0..115.
const Entry kTable[] = {
  {X, X},   // 0 unknown (resolved from the native resource before classification)
  {F, F},   // 1 R32G32B32A32_TYPELESS -> FLOAT
  {F, F}, {X, X}, {X, X},             // 2 FLOAT, 3 UINT, 4 SINT
  {F, F}, {F, F}, {X, X}, {X, X},     // 5 R32G32B32_TYPELESS, 6 FLOAT, 7 UINT, 8 SINT
  {F, F}, {F, F}, {U, U}, {X, X}, {S, S}, {X, X},  // 9 R16G16B16A16_TYPELESS, 10 FLOAT, 11 UNORM, 12 UINT, 13 SNORM, 14 SINT
  {F, F}, {F, F}, {X, X}, {X, X},     // 15 R32G32_TYPELESS, 16 FLOAT, 17 UINT, 18 SINT
  {F, DF},  // 19 R32G8X24_TYPELESS: float member R32_FLOAT_X8X24; depth family D32_FLOAT_S8X24
  {X, DF},  // 20 D32_FLOAT_S8X24_UINT (depth format)
  {F, DF},  // 21 R32_FLOAT_X8X24_TYPELESS (depth read view, float)
  {X, X},   // 22 X32_TYPELESS_G8X24_UINT (stencil, integer)
  {U, U}, {U, U}, {X, X},             // 23 R10G10B10A2_TYPELESS -> UNORM, 24 UNORM, 25 UINT
  {F, F},   // 26 R11G11B10_FLOAT
  {U, U}, {U, U}, {R, R}, {X, X}, {S, S}, {X, X},  // 27 R8G8B8A8_TYPELESS, 28 UNORM, 29 SRGB, 30 UINT, 31 SNORM, 32 SINT
  {F, F}, {F, F}, {U, U}, {X, X}, {S, S}, {X, X},  // 33 R16G16_TYPELESS, 34 FLOAT, 35 UNORM, 36 UINT, 37 SNORM, 38 SINT
  {F, DF},  // 39 R32_TYPELESS: float member R32_FLOAT; depth family D32_FLOAT
  {X, DF},  // 40 D32_FLOAT
  {F, F}, {X, X}, {X, X},             // 41 R32_FLOAT, 42 UINT, 43 SINT
  {U, DU},  // 44 R24G8_TYPELESS: no float member, unorm member R24_UNORM_X8; depth family D24_UNORM_S8
  {X, DU},  // 45 D24_UNORM_S8_UINT
  {U, DU},  // 46 R24_UNORM_X8_TYPELESS (depth read view, unorm)
  {X, X},   // 47 X24_TYPELESS_G8_UINT (stencil, integer)
  {U, U}, {U, U}, {X, X}, {S, S}, {X, X},  // 48 R8G8_TYPELESS, 49 UNORM, 50 UINT, 51 SNORM, 52 SINT
  {F, DU},  // 53 R16_TYPELESS: float member R16_FLOAT; depth family D16_UNORM
  {F, F},   // 54 R16_FLOAT
  {X, DU},  // 55 D16_UNORM
  {U, U}, {X, X}, {S, S}, {X, X},     // 56 R16_UNORM, 57 UINT, 58 SNORM, 59 SINT
  {U, U}, {U, U}, {X, X}, {S, S}, {X, X},  // 60 R8_TYPELESS, 61 UNORM, 62 UINT, 63 SNORM, 64 SINT
  {U, U},   // 65 A8_UNORM
  {U, U},   // 66 R1_UNORM
  {F, F},   // 67 R9G9B9E5_SHAREDEXP (float, shared exponent)
  {U, U}, {U, U},                     // 68 R8G8_B8G8_UNORM, 69 G8R8_G8B8_UNORM
  {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X},  // 70-78 BC1-BC3
  {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X},                          // 79-84 BC4-BC5
  {U, U}, {U, U}, {U, U}, {U, U}, {U, U},  // 85 B5G6R5, 86 B5G5R5A1, 87 B8G8R8A8, 88 B8G8R8X8, 89 R10G10B10_XR_BIAS_A2
  {U, U}, {R, R}, {U, U}, {R, R},     // 90 B8G8R8A8_TYPELESS, 91 SRGB, 92 B8G8R8X8_TYPELESS, 93 SRGB
  {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X},                          // 94-99 BC6H, BC7
  {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X},  // 100-108 video
  {X, X}, {X, X}, {X, X}, {X, X}, {X, X}, {X, X},                          // 109-114 video / palette
  {U, U},   // 115 B4G4R4A4_UNORM
};
static_assert(sizeof(kTable) / sizeof(kTable[0]) == 116, "DXGI 0..115");
}  // namespace

uint32_t format_class(uint32_t f, bool depth_input) {
  if (f == 191) return WG_FMT_UNORM;  // A4B4G4R4_UNORM
  if (f >= sizeof(kTable) / sizeof(kTable[0])) return WG_FMT_UNSUPPORTED;
  return depth_input ? kTable[f].depth_cls : kTable[f].cls;
}

}  // namespace wgrr
