// SPDX-License-Identifier: MIT
// The converter map of spec 08 section 5 (map versions 1 and 4), spec 48 section 5 (versions 2 and 3) and spec 57
// section 5 (version 5, preset F): parsing and validation. Written for this project.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sha256.h"

namespace wgc {

// Base64: the text source T of spec 48 (its segments decode to the binary16 array W). I32: signed 32-bit.
// Hex32: the source F of spec 08 section 4.3 (a compressed text block; its listed fields give unsigned 32-bit
// elements).
enum class Element { F16, U16, U32, Base64, I32, Hex32 };
// Window (map version 5, spec 57 section 4): a block found by the block search whose one element is the value at
// byte `field` of the block.
enum class Locate { Block, Pair, Window };
// The first eight are the operations of spec 08 and 48 (map versions 1-4). The others are spec 57 section 6 (map
// version 5); there `f16` is F16Bits (the bit pattern unchanged, float16 output) and `exp` is Exp5 (float32
// output, optional cap from a data word): the map version decides (57 section 5.1).
enum class OpKind {
    F16, FP8, F16Scaled, Exp, U32AsF32, U16Shr, CappedSoftmax, I32,
    E4M3, F16Bits, U16, U32, Exp5, ScoreBias, ConstU32AsF32, ConstF16x2, ConstF16lo, ConstF32ToF16, AttnWmax
};
// Float16 .. E4M3: map version 5 only (spec 56 section 56.4.1, spec 57 section 7.3).
enum class DType { Float32, UInt8, Int32, Float16, UInt16, UInt32, E4M3 };

// Bytes per element of a dtype (row-major little-endian, spec 57 section 7.3).
inline std::uint64_t dtype_size(DType d) {
    switch (d) {
        case DType::UInt8: case DType::E4M3: return 1;
        case DType::Float16: case DType::UInt16: return 2;
        default: return 4;
    }
}

// The dtype code of the model-file container (OUTPUT-FORMAT.md).
inline std::uint8_t dtype_code(DType d) {
    switch (d) {
        case DType::Float32: return 0;
        case DType::UInt8: return 1;
        case DType::Int32: return 2;
        case DType::Float16: return 3;
        case DType::UInt16: return 4;
        case DType::UInt32: return 5;
        default: return 6;  // E4M3
    }
}

inline bool is_const_op(OpKind k) {
    return k == OpKind::ConstU32AsF32 || k == OpKind::ConstF16x2 || k == OpKind::ConstF16lo || k == OpKind::ConstF32ToF16;
}
// Operations that read their source through the index formula (`src`, `base`, `axes`).
inline bool is_indexed_op(OpKind k) {
    return !is_const_op(k) && k != OpKind::ScoreBias && k != OpKind::AttnWmax;
}

// `require` conditions of spec 57 section 6.3 (bit mask).
enum Require : unsigned { kReqFinite = 1, kReqNotSubnormal = 2, kReqEven = 4, kReqBelow8 = 8 };

struct Segment {
    std::uint64_t offset = 0;  // bytes, relative to the start of the source
    std::uint64_t length = 0;  // bytes, a multiple of 4
};

// One block that a source with variants may be (spec 08 section 4.3; spec 57 section 4: without decoding).
struct Variant {
    std::uint64_t length = 0;  // bytes of the (compressed) block
    std::uint64_t fnv = 0;     // keyed, of the first min(64, length) bytes
    Digest sha{};              // keyed, of the whole block
    // Map version 4 only (Hex32): the decoding step.
    std::uint64_t decoded_length = 0;  // bytes of the decoded text
    Digest decoded_sha{};              // keyed, of the decoded text
    std::vector<std::uint64_t> fields;  // byte offset of the 8 hexadecimal digits of each element
};

// A byte string located in the DLL (spec 08 section 4; spec 48 section 4; spec 57 section 4).
struct Source {
    std::string name;  // key in "sources" (W, D, K, F; T, O; preset F: W, D, SH, BS_A, ..., IMG-...)
    std::string what;
    Element element = Element::F16;
    std::uint64_t length = 0;  // bytes (0 for a source with variants: each variant has its own)
    Locate locate = Locate::Block;
    std::vector<std::uint64_t> gaps;  // pair only
    std::vector<Segment> segments;    // Base64 only, in map order
    std::uint64_t field = 0;          // window only: byte offset of its one element inside the block
    bool keyed = false;
    std::uint64_t fnv = 0;
    Digest sha{};
    // The blocks to try, in map order. Map version 4 (Hex32, source F): such a source is optional and decoded;
    // when no variant is found, the conversion succeeds without the output groups that read it. Map version 5
    // (a code image, spec 57 section 4): no decoding and not optional; the variant found is recorded.
    bool optional = false;
    std::vector<Variant> variants;

    bool has_variants() const { return !variants.empty(); }
    std::uint64_t element_size() const { return element == Element::U32 || element == Element::I32 ? 4 : 2; }
    // Map version 5: the number of addressable elements (a window has exactly one; a source with variants has
    // the count of the given variant).
    std::uint64_t elements(int variant = -1) const {
        if (locate == Locate::Window) return 1;
        if (has_variants()) return variant < 0 ? 0 : variants[static_cast<std::size_t>(variant)].length / element_size();
        return length / element_size();
    }
};

// An element array that tensors index ("src"). For map versions 1 and 4 every source is one (W, D, K, F; the
// elements of F are the 32-bit integers its hexadecimal fields write). For versions 2 and 3 the Base64 source T
// gives the binary16 array W (its decoded segments, spec 48 section 4), and O is one.
struct Array {
    std::string name;
    int source = -1;                 // index into Map::sources
    Element element = Element::F16;  // F16, U16, U32 or I32
    // Number of elements. For W from T it is an upper bound until T is decoded (the exact count depends on the
    // Base64 padding); the decoder checks every index again against the decoded length.
    std::uint64_t max_elements = 0;

    std::uint64_t element_size() const { return element == Element::U32 || element == Element::I32 ? 4 : 2; }
};

struct Op {
    OpKind kind = OpKind::F16;
    float mul = 1.0f, div = 1.0f;  // f16_scaled; div also capped_softmax
    bool has_max = false;          // exp
    float max = 0.0f;
    int shift = 0;                 // u16_shr
    float cap = 0.0f, sub = 0.0f;  // capped_softmax
    // Map version 5 (spec 57 section 6).
    int max_src = -1;              // exp: index into Map::sources of the 32-bit window that caps y, or -1
    int a_src = -1, w_src = -1;    // score_bias: the 32-bit windows A and Wd (indices into Map::sources)
    std::vector<std::pair<int, int>> pairs;  // score_bias: the 12 block pairs (I, J)
    bool negate = false;           // const_f32_to_f16
    int wmax_in[4] = {-1, -1, -1, -1};  // attn_wmax: clamp_lo, clamp_hi, p1, p0 (tensor indices in the group)
};

struct Axis {
    std::int64_t length = 0;
    std::int64_t stride = 0;
    int perm = -1;  // index into Map::perms, or -1
    std::int64_t limit = -1;  // map version 5 (57 section 5.3): elements with pi(i) >= limit are zero; -1 = none
};

// Map version 5: a second read of the same elements from another source and base (`also`).
struct AlsoRead {
    int src = -1;  // index into Map::arrays
    std::int64_t base = 0;
};

// Map version 5: one site list of a const operation (`sites`).
struct Site {
    int src = -1;  // index into Map::sources (32-bit elements)
    bool by_variant = false;
    std::vector<std::uint64_t> elements;                 // `elements` (every variant)
    std::vector<std::vector<std::uint64_t>> per_variant;  // `elements_by_variant` (one list per variant)

    const std::vector<std::uint64_t>& selected(int variant) const {
        return by_variant ? per_variant[static_cast<std::size_t>(variant)] : elements;
    }
};

struct Tensor {
    std::string name;
    std::vector<std::int64_t> shape;
    DType dtype = DType::Float32;
    int src = -1;  // index into Map::arrays (-1 for the const operations and attn_wmax of map version 5)
    Op op;
    std::int64_t base = 0;
    std::vector<Axis> axes;
    std::uint64_t count = 0;  // number of elements
    // Map version 5.
    std::vector<AlsoRead> also;
    std::vector<Site> sites;
    unsigned require = 0;  // Require bits
    bool allow_nonfinite = false;

    std::uint64_t byte_size() const { return count * dtype_size(dtype); }
};

// Map version 5: a group check of spec 57 section 6.3.
enum class Rel { Ge, Gt, Le, RatioBelowF16Overflow };
struct Check {
    Rel rel = Rel::Ge;
    int a = -1;              // tensor index in the group
    int b = -1;              // tensor index in the group, or -1 for a constant
    double b_const = 0.0;    // exactly a binary32 value (checked when the map is read)
    std::string text;        // for the log: "rel a b"
};

struct Stage {
    std::string name;
    Digest digest{};  // unused when variant_source >= 0
    // A group whose tensors all come from one source with variants (spec 08 section 5, "stages"): that source's
    // index in Map::sources and one digest per variant, in the same order. The group exists in the output only
    // when a variant was found; its digest is then the entry of that variant.
    int variant_source = -1;
    std::vector<Digest> variant_digests;
    std::vector<Tensor> tensors;
    // Map version 5 (a "group"): checks after all its tensors, and `requires_variant` (source index, variant):
    // the group is produced only when every listed source was found as that variant (spec 57 section 5.2).
    std::vector<Check> checks;
    std::vector<std::pair<int, int>> requires_variant;

    bool optional() const { return variant_source >= 0; }
    bool gated() const { return !requires_variant.empty(); }
    int find_tensor(const std::string& n) const {
        for (std::size_t i = 0; i < tensors.size(); ++i)
            if (tensors[i].name == n) return static_cast<int>(i);
        return -1;
    }
};

struct Perm {
    std::string name;
    std::vector<std::int64_t> table;
};

// A preset the converter knows, by name: the map versions its maps may have, and the player's message when a DLL has no
// model of that preset. A map names its preset in the optional top-level field "preset"; a map without it gets the preset
// its version had before the field existed (versions 1 and 4: M; 2 and 3: K; 5: F), so the shipped maps load as before.
struct PresetInfo {
    const char* name;              // "M", "K", "F", "L"
    unsigned versions;             // bit v set: map version v is allowed for this preset
    const char* no_model_message;  // api.cpp kMsgNoPreset*: the DLL has no model of this preset
    bool allows(int version) const { return version >= 0 && version < 32 && (versions >> version & 1u) != 0; }
};
// The known presets, in a fixed order; find_preset: by name (exact, upper case), nullptr when unknown.
const std::vector<PresetInfo>& known_presets();
const PresetInfo* find_preset(const std::string& name);
// The preset a map without a "preset" field has: the one its version always meant ("" for no known version). The first
// row of known_presets that allows the version: M before L, so a map without the field is never preset L.
std::string preset_for_version(int version);

struct Map {
    // 1 or 4 (preset M, spec 08; 4 adds the source F), 2 or 3 (preset K, spec 48), 5 (preset F, spec 57: its
    // output groups are kept in `stages` too)
    int version = 0;
    std::string model;
    // The preset, by name ("M", "K", "F"): the map's "preset" field, or preset_for_version(version) without one.
    // load_map checks it is a known preset that allows the map's version.
    std::string preset_name;
    std::vector<Source> sources;  // in file order
    std::vector<Array> arrays;    // what tensors index
    int key_source = -1;
    std::string key_label;
    std::vector<Perm> perms;
    std::vector<Stage> stages;
    Digest file_sha256{};  // SHA-256 of the map file bytes (for logs and the output header)

    int find_source(const std::string& name) const;
    int find_array(const std::string& name) const;
    // The preset the player messages name, as one letter (the first of preset_name; without one, by the version as before:
    // 'M' for versions 1 and 4, 'K' for versions 2 and 3, 'F' for 5).
    char preset() const {
        if (!preset_name.empty()) return preset_name[0];
        const std::string p = preset_for_version(version);
        return p.empty() ? 'M' : p[0];
    }
    // The preset's entry in known_presets() (nullptr only for a map load_map did not accept).
    const PresetInfo* preset_info() const { return find_preset(preset_name.empty() ? preset_for_version(version) : preset_name); }
};

enum class MapErrorKind {
    None,
    Damaged,      // any structural problem (section 8: "converter data damaged")
    IndexRange,   // an index outside its source (section 8: "the map does not fit this build")
};

struct MapError {
    MapErrorKind kind = MapErrorKind::None;
    std::string detail;  // for the log
};

// Parses and fully validates a map, including every index range (source lengths are part of the map).
bool load_map(const std::string& file_bytes, Map& out, MapError& err);

// Hex helpers shared with the tests: optional 0x prefix, then 1..max_digits hex digits.
bool parse_hex_u64(const std::string& s, int max_digits, std::uint64_t& out);

}  // namespace wgc
