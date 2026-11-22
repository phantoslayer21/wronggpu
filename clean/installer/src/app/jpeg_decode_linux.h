// SPDX-License-Identifier: MIT
// Linux: decodes a JPEG file (Steam's hero, header and capsule art) with libjpeg-turbo, where Windows uses WIC.
#pragma once
#include <cstdint>
#include <vector>

namespace wg::app::lui {

// Reads the JPEG at `utf8Path` into 8-bit BGRA bytes (alpha 255: Cairo's ARGB32 on a little-endian machine, premultiplied
// trivially). libjpeg's own DCT scaling shrinks the image by 2, 4 or 8 while the result stays at least `minW` pixels wide, so
// a big hero is not decoded at full size for a small tile. Fails (false, outputs untouched) on a missing or damaged file, a
// side over 8192 pixels, or a colour model other than grey or YCbCr/RGB (CMYK art does not occur in Steam's cache).
bool DecodeJpegFile(const char* utf8Path, unsigned minW, int& w, int& h, std::vector<uint8_t>& bgra);

// Whether the file starts with the JPEG signature (FF D8 FF); the extension is not trusted.
bool LooksLikeJpeg(const char* utf8Path);

}  // namespace wg::app::lui
