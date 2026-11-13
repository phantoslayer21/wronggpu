// SPDX-License-Identifier: MIT
// Locating the sources in the DLL (spec 08 sections 4.2 and 4.3). Written for this project.
#pragma once

#include <cstdint>
#include <vector>

#include "map.h"
#include "sha256.h"

namespace wgc {

struct Found {
    bool found = false;
    std::uint64_t offset = 0;   // block: start; pair: first 4 bytes
    std::uint64_t offset2 = 0;  // pair: second 4 bytes (offset + gap)
    std::vector<std::uint8_t> bytes;  // the source content (length bytes)
};

struct SearchOptions {
    unsigned threads = 0;  // 0 = hardware concurrency
};

// key is required when s.keyed (and ignored otherwise). Not for a source with variants.
Found locate_source(const std::uint8_t* file, std::uint64_t size, const Source& s, const Digest* key,
                    const SearchOptions& opt);

struct FoundVariant {
    int variant = -1;          // index into Source::variants of the variant that matched; -1 if none did
    std::uint64_t offset = 0;  // start of its block
    std::vector<std::uint8_t> bytes;  // the block (the variant's length bytes)
};

// Section 4.3: tries the variants of s in map order, each with the block search of section 4.2. The first variant
// with a match is used; the others are not searched.
FoundVariant locate_variant(const std::uint8_t* file, std::uint64_t size, const Source& s, const Digest* key,
                            const SearchOptions& opt);

// One block to find by the block search of spec 08 section 4.2 (map version 5: a block, a window or one variant of a
// code image; spec 57 section 4).
struct BlockTarget {
    std::uint64_t length = 0;
    std::uint64_t fnv = 0;  // of the first min(64, length) bytes
    Digest sha{};
    bool keyed = false;
};

// Finds every target as locate_source would find it alone (pass order: offsets that are multiples of 16, then the
// other even offsets, then the odd ones; the lowest offset within a pass; every candidate checked against its
// full SHA-256), sharing one scan among the targets with the same prefix length and key (spec 57 section 4,
// "Search reuse"). Result i belongs to target i.
std::vector<Found> locate_blocks(const std::uint8_t* file, std::uint64_t size, const std::vector<BlockTarget>& targets,
                                 const Digest* key, const SearchOptions& opt);

// Initial FNV state for a source: the offset basis, XOR the key's first 8 bytes when keyed.
std::uint64_t fnv_initial_state(bool keyed, const Digest* key);

}  // namespace wgc
