// SPDX-License-Identifier: MIT
// Block decompression of spec 08 section 4.4 (the public LZ4 block format: no frame header, no checksum).
// Written for this project from the seven steps of that section; the step numbers below are the spec's.
#include "lz4block.h"

namespace wgc {

namespace {

// Steps 2 and 6: adds further bytes to `value` until a byte other than 255 has been read (it is added too).
// A value above `limit` can only describe an output longer than decoded_length, so it is refused at once; that
// also keeps the sum from overflowing.
bool extend(const std::uint8_t* in, std::size_t n, std::size_t& pos, std::size_t limit, std::size_t& value) {
    for (;;) {
        if (pos >= n) return false;  // the input ends inside the length
        const std::uint8_t b = in[pos++];
        value += b;
        if (value > limit) return false;
        if (b != 255) return true;
    }
}

}  // namespace

bool lz4_block_decode(const std::uint8_t* in, std::size_t n, std::size_t decoded_length,
                      std::vector<std::uint8_t>& out) {
    out.clear();
    out.reserve(decoded_length);
    std::size_t pos = 0;
    while (pos < n) {  // "repeat until the input is used up"
        // Step 1.
        const std::uint8_t t = in[pos++];
        std::size_t literals = t >> 4, match = t & 15;
        // Step 2.
        if (literals == 15 && !extend(in, n, pos, decoded_length, literals)) return false;
        // Step 3.
        if (literals > n - pos) return false;                     // the input ends inside the literals
        if (literals > decoded_length - out.size()) return false;  // output too long
        out.insert(out.end(), in + pos, in + pos + literals);
        pos += literals;
        // Step 4: the last sequence has literals only.
        if (pos == n) break;
        // Step 5.
        if (n - pos < 2) return false;  // the input ends inside the distance
        const std::size_t distance = static_cast<std::size_t>(in[pos]) | (static_cast<std::size_t>(in[pos + 1]) << 8);
        pos += 2;
        if (distance == 0 || distance > out.size()) return false;
        // Step 6.
        if (match == 15 && !extend(in, n, pos, decoded_length, match)) return false;
        match += 4;
        if (match > decoded_length - out.size()) return false;  // output too long
        // Step 7: one byte at a time, so a range that overlaps the bytes just written repeats a pattern.
        for (std::size_t i = 0; i < match; ++i) {
            const std::uint8_t b = out[out.size() - distance];
            out.push_back(b);
        }
    }
    return out.size() == decoded_length;
}

}  // namespace wgc
