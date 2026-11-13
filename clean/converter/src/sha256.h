// SPDX-License-Identifier: MIT
// SHA-256 as specified in FIPS 180-4 (written from the standard for this project).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace wgc {

using Digest = std::array<std::uint8_t, 32>;

class Sha256 {
public:
    Sha256();
    void update(const void* data, std::size_t size);
    Digest finish();  // the object must not be used again afterwards

    static Digest hash(const void* data, std::size_t size);

private:
    void block(const std::uint8_t* p);

    std::uint32_t h_[8];
    std::uint8_t buf_[64];
    std::size_t used_ = 0;
    std::uint64_t total_ = 0;  // bytes
};

// Keyed SHA-256 of spec 08 section 4.1: SHA-256(key || data).
Digest keyed_sha256(const Digest& key, const void* data, std::size_t size);

std::string to_hex(const Digest& d);
// Parses exactly 64 hex digits (either case). Returns false on any other input.
bool parse_digest_hex(const std::string& s, Digest& out);

}  // namespace wgc
