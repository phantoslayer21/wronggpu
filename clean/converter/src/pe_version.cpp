// SPDX-License-Identifier: MIT
// See pe_version.h. The VS_VERSIONINFO structure starts with three 16-bit fields (wLength,
// wValueLength, wType), then the UTF-16LE key L"VS_VERSION_INFO" with its terminator, padding to a
// 32-bit boundary, then VS_FIXEDFILEINFO, whose first field is the signature 0xFEEF04BD and whose
// third and fourth fields are dwFileVersionMS and dwFileVersionLS. Instead of walking the PE resource
// directory, this looks for the key text and checks the signature after it, which is enough for a
// display-only value.
#include "pe_version.h"

#include <cstring>

namespace wgc {

namespace {

std::uint32_t rd32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

}  // namespace

std::string read_file_version(const std::uint8_t* file, std::uint64_t size) {
    static const char key[] = "VS_VERSION_INFO";
    std::uint8_t pattern[32];  // UTF-16LE key including the terminator
    for (int i = 0; i < 16; ++i) {
        pattern[2 * i] = static_cast<std::uint8_t>(key[i]);
        pattern[2 * i + 1] = 0;
    }
    for (std::uint64_t o = 6; o + 32 + 4 + 16 <= size; o += 2) {
        if (file[o] != 'V' || std::memcmp(file + o, pattern, 32) != 0) continue;
        const std::uint64_t start = o - 6;  // start of VS_VERSIONINFO
        const std::uint64_t value = start + 40;  // 6 + 32 bytes, padded to a multiple of 4 from start
        if (value + 16 > size) break;
        if (rd32(file + value) != 0xFEEF04BDu) continue;
        const std::uint32_t ms = rd32(file + value + 8), ls = rd32(file + value + 12);
        return std::to_string(ms >> 16) + "." + std::to_string(ms & 0xFFFF) + "." + std::to_string(ls >> 16) + "." +
               std::to_string(ls & 0xFFFF);
    }
    return std::string();
}

}  // namespace wgc
