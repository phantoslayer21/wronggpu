// SPDX-License-Identifier: MIT
// Decoding the tensors (spec 08 sections 4.3, 5.1, 6, 7) and checking the stage digests (7.3).
// Written for this project.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "map.h"
#include "sha256.h"

namespace wgc {

struct TensorOut {
    const Tensor* tensor = nullptr;
    std::vector<std::uint8_t> bytes;  // row-major little-endian (float32: 4 bytes per element)
};

struct StageOut {
    const Stage* stage = nullptr;
    std::vector<TensorOut> tensors;
    Digest digest{};  // keyed SHA-256 of the concatenated tensor bytes
    int variant = -1;  // for a group read from a source with variants: the variant found (0-based)
};

// SourceDamaged: a block of a source with variants matched its hashes but does not decode, does not match
// decoded_sha256, or has a field that is not hexadecimal (spec 08 section 4.3: "converter data damaged").
// Map version 5 (spec 57 section 8) adds ValueRejected (a constant's sites disagree, a forbidden non-finite value, an
// RN16 overflow, `also` reads that differ, a failed `require` or check, rejected attention constants: the player
// message "not supported yet") and MapDamaged (a defensive bounds failure at run time: "converter data damaged").
enum class DecodeErrorKind { None, NonFinite, OutOfRange, DigestMismatch, SourceDamaged, ValueRejected, MapDamaged };

struct DecodeError {
    DecodeErrorKind kind = DecodeErrorKind::None;
    std::string detail;  // stage, tensor and element index; never values or source bytes
};

// The key of section 4.1: SHA-256(label || bytes of the key source).
Digest make_key(const std::string& label, const std::vector<std::uint8_t>& key_source_bytes);

// RFC 4648 section 4 Base64 decoding of n bytes of text, appended to out. n must be a multiple of 4; '=' padding
// only in the last quad. Returns false on any other character or form (bits below the padding are ignored).
bool decode_base64(const std::uint8_t* text, std::size_t n, std::vector<std::uint8_t>& out);

// Spec 08 section 4.3 for the located block of one variant: decodes it (section 4.4) to the variant's
// decoded_length, checks decoded_sha256 and reads the listed fields (eight hexadecimal digits each, most
// significant first). On success `elements` holds one little-endian unsigned 32-bit integer per field. The decoded
// text is kept nowhere. On failure `why` names the step (never content) and `elements` is empty.
bool decode_variant_block(const Variant& v, const std::vector<std::uint8_t>& block, const Digest& key,
                          std::vector<std::uint8_t>& elements, std::string& why);

// sources[i] holds the bytes of map.sources[i]; for a source with variants, the elements decode_variant_block
// gave, and variants[i] is the variant that was found (0-based) or -1 if none was (variants[i] is ignored for
// every other source). Builds the element arrays (W from the segments of a Base64 source T), decodes every stage,
// computes each digest and compares it with the map. A group whose source has no found variant is left out of
// `out`. Returns false on the first error; `out` is then incomplete. A Base64 failure or an odd decoded length is
// reported as OutOfRange (spec 48 section 8: an installer error, like an index outside W).
bool convert_all(const Map& map, const std::vector<std::vector<std::uint8_t>>& sources,
                 const std::vector<int>& variants, const Digest& key, std::vector<StageOut>& out, DecodeError& err);

// What a version-5 conversion records for the log besides the groups (spec 57 section 6.4).
struct Convert5Info {
    bool have_wmax = false;                   // an attn_wmax tensor was produced
    bool attention_sum_may_overflow = false;  // 64 * WMAX >= 65520
};

// Map version 5 (spec 57 sections 5.3, 6 and 7). sources[i] holds the bytes of map.sources[i] as found (a window: its
// whole block; a source with variants: the block of the variant found, variants[i]). Decodes every group whose
// `requires_variant` is met, in map order, applies the value checks and, after all groups, compares every digest.
// Groups whose gate is not met are left out of `out`. Returns false on the first error; `out` is then incomplete.
bool convert_all_v5(const Map& map, const std::vector<std::vector<std::uint8_t>>& sources,
                    const std::vector<int>& variants, const Digest& key, std::vector<StageOut>& out, DecodeError& err,
                    Convert5Info* info = nullptr);

// Whether a version-5 group is produced for the variants found (its `requires_variant` is met).
bool group_enabled(const Stage& g, const std::vector<int>& variants);

}  // namespace wgc
