// SPDX-License-Identifier: MIT
// Block decompression of spec 08 section 4.4 (the public LZ4 block format: no frame header, no checksum).
// Written for this project from the seven steps of that section.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wgc {

// Decodes the n bytes at `in` into `out` (cleared first). Returns true only if the whole input was used and the
// output has exactly decoded_length bytes. Returns false for every malformed block: input that ends inside a
// sequence, distance 0, a distance greater than the output so far, an output that would become longer than
// decoded_length, or a final size other than decoded_length. Never reads outside [in, in + n) and never lets
// `out` grow beyond decoded_length bytes. On failure the content of `out` is unspecified.
bool lz4_block_decode(const std::uint8_t* in, std::size_t n, std::size_t decoded_length,
                      std::vector<std::uint8_t>& out);

}  // namespace wgc
