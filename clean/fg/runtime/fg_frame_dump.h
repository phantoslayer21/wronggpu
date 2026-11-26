// SPDX-License-Identifier: MIT
#pragma once
// Frame generation (issue #96, F9 developer diagnostics): the CPU side of the executor's frame dumps. A presenter
// slot's bytes (rows of `pitch`, one of the four slot formats the executor writes) become 8-bit RGB, written as a
// binary PPM (P6). Float16 slots are clamped to [0, 1] and scaled; no transfer function is applied (a quick look, not a
// colour-exact capture). No Windows calls, no external libraries.
#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace wgfg {

// DXGI format numbers of the slot formats (dxgiformat.h), so this header needs no Windows header.
enum FgDumpFormat : uint32_t {
    kFgDumpRgba16Float = 10,
    kFgDumpRgb10A2 = 24,
    kFgDumpRgba8 = 28,
    kFgDumpRgba8Srgb = 29,
    kFgDumpBgra8 = 87,
    kFgDumpBgra8Srgb = 91,
};

// Bytes per pixel of a slot format the dump reads; 0: not one of them.
inline uint32_t FgDumpBytesPerPixel(uint32_t format) {
    switch (format) {
        case kFgDumpRgba16Float: return 8;
        case kFgDumpRgb10A2: case kFgDumpRgba8: case kFgDumpRgba8Srgb: case kFgDumpBgra8: case kFgDumpBgra8Srgb: return 4;
        default: return 0;
    }
}

// IEEE half -> float (subnormals kept, infinities and NaN as such).
inline float FgDumpHalfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t e = 113, m = man;  // 127 - 15 + 1
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

inline uint8_t FgDumpUnit8(float v) {
    if (!(v > 0.0f)) return 0;  // NaN and negatives
    if (v >= 1.0f) return 255;
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

// Slot bytes -> tightly packed RGB8 (w * h * 3). False: unknown format, a zero size or a pitch too small for a row.
inline bool FgDumpToRgb8(const uint8_t* src, uint32_t pitch, uint32_t w, uint32_t h, uint32_t format,
                         std::vector<uint8_t>* rgb) {
    const uint32_t bpp = FgDumpBytesPerPixel(format);
    if (!src || !rgb || !bpp || !w || !h || static_cast<uint64_t>(pitch) < static_cast<uint64_t>(bpp) * w) return false;
    rgb->assign(static_cast<size_t>(w) * h * 3, 0);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* row = src + static_cast<size_t>(y) * pitch;
        uint8_t* o = rgb->data() + static_cast<size_t>(y) * w * 3;
        for (uint32_t x = 0; x < w; ++x, o += 3) {
            const uint8_t* p = row + static_cast<size_t>(x) * bpp;
            switch (format) {
                case kFgDumpRgba16Float: {
                    uint16_t c[3];
                    std::memcpy(c, p, sizeof c);
                    for (int i = 0; i < 3; ++i) o[i] = FgDumpUnit8(FgDumpHalfToFloat(c[i]));
                    break;
                }
                case kFgDumpRgb10A2: {
                    uint32_t v;
                    std::memcpy(&v, p, sizeof v);
                    o[0] = static_cast<uint8_t>((v >> 2) & 0xFFu);
                    o[1] = static_cast<uint8_t>((v >> 12) & 0xFFu);
                    o[2] = static_cast<uint8_t>((v >> 22) & 0xFFu);
                    break;
                }
                case kFgDumpBgra8: case kFgDumpBgra8Srgb:
                    o[0] = p[2];
                    o[1] = p[1];
                    o[2] = p[0];
                    break;
                default:  // RGBA8
                    o[0] = p[0];
                    o[1] = p[1];
                    o[2] = p[2];
                    break;
            }
        }
    }
    return true;
}

// The binary PPM bytes of a tightly packed RGB8 image: "P6\n<w> <h>\n255\n" then the pixels.
inline std::string FgDumpPpmHeader(uint32_t w, uint32_t h) {
    return "P6\n" + std::to_string(w) + " " + std::to_string(h) + "\n255\n";
}

inline bool FgDumpWritePpm(const std::string& path, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgb) {
    if (!w || !h || rgb.size() != static_cast<size_t>(w) * h * 3) return false;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    const std::string head = FgDumpPpmHeader(w, h);
    f.write(head.data(), static_cast<std::streamsize>(head.size()));
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(f);
}

}  // namespace wgfg
