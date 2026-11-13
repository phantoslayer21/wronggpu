// SPDX-License-Identifier: MIT
// The converter map of spec 08 section 5 (versions 1 and 4), spec 48 section 5 (versions 2 and 3) and spec 57
// section 5 (version 5): parsing and validation. Written for this project.
//
// Version 1 (preset M): sources W, D, K; tensors index the sources themselves.
// Version 2 (preset K): one source T, Base64 text with segments; tensors index W, the binary16 array its segments
// decode to. Adds negative strides (already accepted in version 1) and the op capped_softmax.
// Version 3 (preset K): as 2 plus the keyed block source O (int32 elements), the op i32 and the dtype int32.
// Version 4 (preset M): as 1 plus the optional source F (element hex32, encoding lz4-block, a list of variants;
// spec 08 section 4.3) and, for a group read from it, a list of digests (one per variant). It does not include the
// preset-K additions of versions 2 and 3.
// Version 5 (preset F, spec 57 section 5): its own feature set (windows, code images with undecoded variants,
// the ops of 57 section 6, axis limits, also, sites, require, checks, requires_variant, "groups"); none of the
// version 1-4-only features. All rules of 57 section 5.4 are checked here, before any DLL is read (load_v5).
#include "map.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <set>

#include "api.h"
#include "json.h"
#include "numeric.h"

namespace wgc {

int Map::find_source(const std::string& name) const {
    for (std::size_t i = 0; i < sources.size(); ++i)
        if (sources[i].name == name) return static_cast<int>(i);
    return -1;
}

int Map::find_array(const std::string& name) const {
    for (std::size_t i = 0; i < arrays.size(); ++i)
        if (arrays[i].name == name) return static_cast<int>(i);
    return -1;
}

bool parse_hex_u64(const std::string& s, int max_digits, std::uint64_t& out) {
    std::size_t i = 0;
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) i = 2;
    const std::size_t n = s.size() - i;
    if (n == 0 || n > static_cast<std::size_t>(max_digits)) return false;
    std::uint64_t v = 0;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return false;
        v = (v << 4) | static_cast<std::uint64_t>(d);
    }
    out = v;
    return true;
}

namespace {

// Limits that no plausible map reaches; they keep a damaged map from exhausting memory.
const std::uint64_t kMaxTensorElements = std::uint64_t(1) << 26;
const std::uint64_t kMaxTotalElements = std::uint64_t(1) << 28;
const std::uint64_t kMaxSourceLength = std::uint64_t(1) << 32;
const std::uint64_t kMaxDecodedLength = std::uint64_t(1) << 26;  // the decoded text of one variant
const std::size_t kMaxVariants = 16;
const std::size_t kMaxFields = 4096;
const std::size_t kMaxRank = 16;
const std::size_t kMaxSiteElements = std::size_t(1) << 20;  // map version 5: all site elements of one tensor

// ---- exact decimal -> binary32 (spec 57 section 5.4: a {"const": n} operand is exactly a binary32 value) ----

// A small unsigned big integer (base 2^32, little-endian limbs) for exact comparisons of decimal literals with
// binary32 values.
struct BigUint {
    std::vector<std::uint32_t> limb;

    void mul_small(std::uint32_t m) {
        std::uint64_t carry = 0;
        for (std::uint32_t& x : limb) {
            const std::uint64_t v = std::uint64_t(x) * m + carry;
            x = static_cast<std::uint32_t>(v);
            carry = v >> 32;
        }
        if (carry) limb.push_back(static_cast<std::uint32_t>(carry));
    }
    void add_small(std::uint32_t a) {
        std::uint64_t carry = a;
        for (std::size_t i = 0; i < limb.size() && carry; ++i) {
            const std::uint64_t v = std::uint64_t(limb[i]) + carry;
            limb[i] = static_cast<std::uint32_t>(v);
            carry = v >> 32;
        }
        if (carry) limb.push_back(static_cast<std::uint32_t>(carry));
    }
    void trim() {
        while (!limb.empty() && limb.back() == 0) limb.pop_back();
    }
    bool operator==(BigUint o) const {
        BigUint a = *this;
        a.trim();
        o.trim();
        return a.limb == o.limb;
    }
};

// The exact value of a JSON number literal as binary32. Returns false if the literal is not exactly a finite
// binary32 value (or is unreasonably long).
bool json_number_exact_f32(const std::string& lit, float& out) {
    std::size_t i = 0;
    bool neg = false;
    if (i < lit.size() && lit[i] == '-') neg = true, ++i;
    std::string digits;
    long long exp10 = 0;
    for (; i < lit.size() && lit[i] >= '0' && lit[i] <= '9'; ++i) digits.push_back(lit[i]);
    if (i < lit.size() && lit[i] == '.') {
        for (++i; i < lit.size() && lit[i] >= '0' && lit[i] <= '9'; ++i) digits.push_back(lit[i]), --exp10;
    }
    if (i < lit.size() && (lit[i] == 'e' || lit[i] == 'E')) {
        ++i;
        bool eneg = false;
        if (i < lit.size() && (lit[i] == '+' || lit[i] == '-')) eneg = lit[i] == '-', ++i;
        long long e = 0;
        std::size_t start = i;
        for (; i < lit.size() && lit[i] >= '0' && lit[i] <= '9'; ++i) {
            e = e * 10 + (lit[i] - '0');
            if (e > 100000) return false;
        }
        if (i == start) return false;
        exp10 += eneg ? -e : e;
    }
    if (i != lit.size() || digits.empty() || digits.size() > 1000) return false;
    std::size_t nz = digits.find_first_not_of('0');
    if (nz == std::string::npos) {  // zero (keep the sign of a written -0)
        out = neg ? -0.0f : 0.0f;
        return true;
    }
    digits = digits.substr(nz);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back(), ++exp10;
    if (exp10 < -200 || exp10 > 60) return false;  // outside every finite binary32 value's exact decimal form
    // Candidate: the nearest double (strtod of the normalised literal); it must be a binary32 value, and the
    // literal must equal it exactly: D * 10^E == k * 2^q, compared as integers.
    const std::string norm = digits + "e" + std::to_string(exp10);
    const double d = std::strtod(norm.c_str(), nullptr);
    if (!std::isfinite(d) || d == 0.0 || static_cast<double>(static_cast<float>(d)) != d) return false;
    int q;
    const double frac = std::frexp(d, &q);                // d = frac * 2^q, frac in [0.5, 1)
    std::uint64_t k = static_cast<std::uint64_t>(std::ldexp(frac, 53));  // exact: d has <= 53 bits
    q -= 53;
    while ((k & 1) == 0) k >>= 1, ++q;
    BigUint lhs, rhs;
    for (char c : digits) {
        lhs.mul_small(10);
        lhs.add_small(static_cast<std::uint32_t>(c - '0'));
    }
    rhs.limb = {static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(k >> 32)};
    for (long long e = exp10; e > 0; --e) lhs.mul_small(10);
    for (long long e = exp10; e < 0; ++e) rhs.mul_small(10);
    for (int e = q; e > 0; --e) rhs.mul_small(2);
    for (int e = q; e < 0; ++e) lhs.mul_small(2);
    if (!(lhs == rhs)) return false;
    out = static_cast<float>(neg ? -d : d);
    return true;
}

class Loader {
public:
    explicit Loader(MapError& err) : err_(err) {}

    bool damaged(const std::string& what) {
        err_.kind = MapErrorKind::Damaged;
        err_.detail = what;
        return false;
    }
    bool out_of_range(const std::string& what) {
        err_.kind = MapErrorKind::IndexRange;
        err_.detail = what;
        return false;
    }

    // Required-field accessors. `where` names the object for the log.
    const Json* need(const Json& obj, const char* key, const std::string& where) {
        const Json* v = obj.find(key);
        if (!v) damaged(where + ": missing field \"" + key + "\"");
        return v;
    }
    bool need_string(const Json& obj, const char* key, const std::string& where, std::string& out) {
        const Json* v = need(obj, key, where);
        if (!v) return false;
        if (!v->is_string()) return damaged(where + ": field \"" + key + "\" must be text");
        out = v->text;
        return true;
    }
    bool need_int(const Json& obj, const char* key, const std::string& where, std::int64_t& out) {
        const Json* v = need(obj, key, where);
        if (!v) return false;
        if (!v->as_int64(out)) return damaged(where + ": field \"" + key + "\" must be an integer");
        return true;
    }
    bool need_object(const Json& obj, const char* key, const std::string& where, const Json*& out) {
        out = need(obj, key, where);
        if (!out) return false;
        if (!out->is_object()) return damaged(where + ": field \"" + key + "\" must be an object");
        return true;
    }
    bool need_array(const Json& obj, const char* key, const std::string& where, const Json*& out) {
        out = need(obj, key, where);
        if (!out) return false;
        if (!out->is_array()) return damaged(where + ": field \"" + key + "\" must be a list");
        return true;
    }
    bool need_f32_bits(const Json& obj, const char* key, const std::string& where, float& out) {
        std::string s;
        if (!need_string(obj, key, where, s)) return false;
        std::uint64_t bits;
        if (!parse_hex_u64(s, 8, bits)) return damaged(where + ": \"" + key + "\" is not a binary32 bit pattern");
        out = bits_to_float(static_cast<std::uint32_t>(bits));
        return true;
    }

    // Names become file and folder names under --raw, so they are restricted to a safe set and must be
    // unique ignoring ASCII case (Windows file names are case-insensitive).
    static std::string fold(std::string s) {
        for (char& c : s)
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        return s;
    }

    // Names become file names under --raw, so they are restricted to a safe set.
    static bool safe_name(const std::string& s) {
        if (s.empty() || s.size() > 64 || s[0] == '.') return false;
        for (char c : s) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                            c == '_' || c == '-' || c == '.';
            if (!ok) return false;
        }
        return true;
    }

    bool load(const Json& root, Map& m) {
        if (!root.is_object()) return damaged("the map is not a JSON object");
        std::string format;
        if (!need_string(root, "format", "map", format)) return false;
        if (format != "wronggpu-converter-map") return damaged("unknown format \"" + format + "\"");
        std::int64_t version;
        if (!need_int(root, "version", "map", version)) return false;
        if (version < 1 || version > 5) return damaged("unsupported map version " + std::to_string(version));
        m.version = version_ = static_cast<int>(version);
        k_ = version_ == 2 || version_ == 3;
        if (!need_string(root, "model", "map", m.model)) return false;
        // The preset, by name: the optional field "preset", else the one this version always meant (the shipped maps).
        if (const Json* pj = root.find("preset")) {
            if (!pj->is_string()) return damaged("map: field \"preset\" must be text");
            const PresetInfo* p = find_preset(pj->text);
            if (!p) return damaged("map: unknown preset \"" + pj->text + "\"");
            if (!p->allows(version_))
                return damaged("map: preset " + pj->text + " does not use map version " + std::to_string(version_));
            m.preset_name = p->name;
        } else {
            m.preset_name = preset_for_version(version_);
        }
        std::string order;
        if (!need_string(root, "channel_order", "map", order)) return false;
        if (order != "natural") return damaged("unknown channel_order \"" + order + "\"");
        if (version_ == 5) return load_v5(root, m);
        // Spec 57 section 5.1: a map of versions 1-4 that uses a version-5 feature is invalid.
        if (root.find("groups")) return damaged("map: \"groups\" is a version-5 feature");

        const Json* sources;
        if (!need_object(root, "sources", "map", sources)) return false;
        for (const auto& kv : sources->members) {
            if (!load_source(kv.first, kv.second, m)) return false;
        }
        if (!k_) {
            for (const char* required : {"W", "D", "K"}) {
                if (m.find_source(required) < 0) return damaged(std::string("sources: missing source ") + required);
            }
        } else if (!check_k_sources(m)) {
            return false;
        }
        build_arrays(m);

        const Json* key;
        if (!need_object(root, "key", "map", key)) return false;
        if (!need_string(*key, "label", "key", m.key_label)) return false;
        std::string from;
        if (!need_string(*key, "from", "key", from)) return false;
        m.key_source = m.find_source(from);
        if (m.key_source < 0) return damaged("key: unknown source \"" + from + "\"");
        const Source& ks = m.sources[static_cast<std::size_t>(m.key_source)];
        if (ks.locate != Locate::Block || ks.keyed)
            return damaged("key: source \"" + from + "\" must be an unkeyed block source");
        if (k_ && ks.name != "T") return damaged("key: must be made from source T");

        const Json* perms;
        if (!need_object(root, "perms", "map", perms)) return false;
        for (const auto& kv : perms->members) {
            if (!load_perm(kv.first, kv.second, m)) return false;
        }

        const Json* stages;
        if (!need_array(root, "stages", "map", stages)) return false;
        if (stages->items.empty()) return damaged("stages: the list is empty");
        std::set<std::string> stage_names;
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < stages->items.size(); ++i) {
            Stage st;
            if (!load_stage(stages->items[i], i, m, st, total)) return false;
            if (!stage_names.insert(fold(st.name)).second) return damaged("stages: duplicate stage \"" + st.name + "\"");
            m.stages.push_back(std::move(st));
        }
        return true;
    }

    bool load_source(const std::string& name, const Json& j, Map& m) {
        const std::string where = "sources." + name;
        if (!safe_name(name)) return damaged(where + ": bad source name");
        if (!j.is_object()) return damaged(where + ": must be an object");
        Source s;
        s.name = name;
        if (!need_string(j, "what", where, s.what)) return false;
        std::string element, locate;
        if (!need_string(j, "element", where, element)) return false;
        if (element == "f16le") s.element = Element::F16;
        else if (element == "u16le") s.element = Element::U16;
        else if (element == "u32le") s.element = Element::U32;
        else if (element == "base64" && k_) s.element = Element::Base64;
        else if (element == "i32le" && m.version == 3) s.element = Element::I32;
        else if (element == "hex32" && m.version == 4) s.element = Element::Hex32;
        else return damaged(where + ": unknown element \"" + element + "\"");
        if (j.find("field")) return damaged(where + ": \"field\" is a version-5 feature");
        if (const Json* enc = j.find("encoding"))
            if (enc->is_string() && enc->text == "none") return damaged(where + ": encoding none is a version-5 feature");
        if (s.element == Element::Hex32) {
            if (!load_variant_source(j, where, s)) return false;
            m.sources.push_back(std::move(s));
            return true;
        }
        if (m.version == 4) {
            // What only a source with variants may have (spec 08 section 5). Earlier versions ignore these fields.
            if (const Json* enc = j.find("encoding")) {
                if (!enc->is_string() || enc->text != "lz4-block")
                    return damaged(where + ": unknown encoding" + (enc->is_string() ? " \"" + enc->text + "\"" : ""));
                return damaged(where + ": encoding lz4-block needs element hex32");
            }
            if (j.find("variants")) return damaged(where + ": variants need element hex32");
            if (const Json* opt = j.find("optional")) {
                if (!opt->is_bool()) return damaged(where + ": optional must be true or false");
                if (opt->boolean) return damaged(where + ": only a source with variants may be optional");
            }
        }
        std::int64_t length;
        if (!need_int(j, "length", where, length)) return false;
        if (length <= 0 || static_cast<std::uint64_t>(length) > kMaxSourceLength)
            return damaged(where + ": bad length");
        s.length = static_cast<std::uint64_t>(length);
        if (s.element == Element::Base64) {
            if (!load_segments(j, where, s)) return false;
        } else {
            const std::uint64_t size = (s.element == Element::U32 || s.element == Element::I32) ? 4 : 2;
            if (s.length % size != 0) return damaged(where + ": length is not a whole number of elements");
        }
        if (!need_string(j, "locate", where, locate)) return false;
        if (locate == "block") {
            s.locate = Locate::Block;
        } else if (locate == "pair") {
            s.locate = Locate::Pair;
            if (s.length != 8) return damaged(where + ": a pair source must be 8 bytes long");
            const Json* gaps;
            if (!need_array(j, "gaps", where, gaps)) return false;
            if (gaps->items.empty()) return damaged(where + ": gaps is empty");
            for (const Json& g : gaps->items) {
                std::int64_t v;
                if (!g.as_int64(v) || v < 0 || static_cast<std::uint64_t>(v) > kMaxSourceLength)
                    return damaged(where + ": bad gap");
                s.gaps.push_back(static_cast<std::uint64_t>(v));
            }
        } else {
            return damaged(where + ": unknown locate method \"" + locate + "\"");
        }
        if (const Json* keyed = j.find("keyed")) {
            if (!keyed->is_bool()) return damaged(where + ": keyed must be true or false");
            s.keyed = keyed->boolean;
        }
        std::string fnv, sha;
        if (!need_string(j, "fnv1a64", where, fnv)) return false;
        if (!parse_hex_u64(fnv, 16, s.fnv)) return damaged(where + ": bad fnv1a64");
        if (!need_string(j, "sha256", where, sha)) return false;
        if (!parse_digest_hex(sha, s.sha)) return damaged(where + ": bad sha256");
        m.sources.push_back(std::move(s));
        return true;
    }

    // Spec 08 sections 4.3 and 5: the source F. It has "encoding", "optional" and "variants" in place of "length",
    // "fnv1a64" and "sha256"; each variant is one block with its own hashes and field offsets.
    bool load_variant_source(const Json& j, const std::string& where, Source& s) {
        if (s.name != "F") return damaged(where + ": only F is hex32");
        std::string locate, encoding;
        if (!need_string(j, "locate", where, locate)) return false;
        if (locate != "block" && locate != "pair") return damaged(where + ": unknown locate method \"" + locate + "\"");
        if (locate != "block") return damaged(where + ": a hex32 source must be a block source");
        s.locate = Locate::Block;
        const Json* keyed = need(j, "keyed", where);
        if (!keyed) return false;
        if (!keyed->is_bool()) return damaged(where + ": keyed must be true or false");
        if (!keyed->boolean) return damaged(where + ": a hex32 source must be keyed");
        s.keyed = true;
        if (!need_string(j, "encoding", where, encoding)) return false;
        if (encoding != "lz4-block") return damaged(where + ": unknown encoding \"" + encoding + "\"");
        const Json* optional = need(j, "optional", where);
        if (!optional) return false;
        if (!optional->is_bool()) return damaged(where + ": optional must be true or false");
        if (!optional->boolean) return damaged(where + ": a hex32 source must be optional");
        s.optional = true;
        const Json* variants;
        if (!need_array(j, "variants", where, variants)) return false;
        if (variants->items.empty()) return damaged(where + ": variants is empty");
        if (variants->items.size() > kMaxVariants) return damaged(where + ": too many variants");
        for (std::size_t i = 0; i < variants->items.size(); ++i) {
            const Json& vj = variants->items[i];
            const std::string w = where + ", variant " + std::to_string(i + 1);
            if (!vj.is_object()) return damaged(w + ": must be an object");
            Variant v;
            std::int64_t length, decoded;
            if (!need_int(vj, "length", w, length)) return false;
            if (length <= 0 || static_cast<std::uint64_t>(length) > kMaxSourceLength) return damaged(w + ": bad length");
            v.length = static_cast<std::uint64_t>(length);
            std::string fnv, sha, dsha;
            if (!need_string(vj, "fnv1a64", w, fnv)) return false;
            if (!parse_hex_u64(fnv, 16, v.fnv)) return damaged(w + ": bad fnv1a64");
            if (!need_string(vj, "sha256", w, sha)) return false;
            if (!parse_digest_hex(sha, v.sha)) return damaged(w + ": bad sha256");
            if (!need_int(vj, "decoded_length", w, decoded)) return false;
            if (decoded <= 0 || static_cast<std::uint64_t>(decoded) > kMaxDecodedLength)
                return damaged(w + ": bad decoded_length");
            v.decoded_length = static_cast<std::uint64_t>(decoded);
            if (!need_string(vj, "decoded_sha256", w, dsha)) return false;
            if (!parse_digest_hex(dsha, v.decoded_sha)) return damaged(w + ": bad decoded_sha256");
            const Json* fields;
            if (!need_array(vj, "fields", w, fields)) return false;
            if (fields->items.empty()) return damaged(w + ": fields is empty");
            if (fields->items.size() > kMaxFields) return damaged(w + ": too many fields");
            for (const Json& fj : fields->items) {
                std::int64_t f;
                if (!fj.as_int64(f) || f < 0) return damaged(w + ": bad field offset");
                // Eight hexadecimal digits at this offset, inside the decoded text.
                if (v.decoded_length < 8 || static_cast<std::uint64_t>(f) > v.decoded_length - 8)
                    return damaged(w + ": a field lies outside the decoded text");
                v.fields.push_back(static_cast<std::uint64_t>(f));
            }
            // Every variant gives the same elements (one per field).
            if (!s.variants.empty() && v.fields.size() != s.variants[0].fields.size())
                return damaged(w + ": another number of fields than variant 1");
            s.variants.push_back(std::move(v));
        }
        return true;
    }

    // Spec 48 section 5: "segments", a non-empty list of [offset, length] pairs inside the source, each length a
    // positive multiple of 4 (a whole number of Base64 quads).
    bool load_segments(const Json& j, const std::string& where, Source& s) {
        const Json* segs;
        if (!need_array(j, "segments", where, segs)) return false;
        if (segs->items.empty()) return damaged(where + ": segments is empty");
        for (std::size_t i = 0; i < segs->items.size(); ++i) {
            const Json& p = segs->items[i];
            const std::string w = where + ": segment " + std::to_string(i);
            std::int64_t off, len;
            if (!p.is_array() || p.items.size() != 2 || !p.items[0].as_int64(off) || !p.items[1].as_int64(len))
                return damaged(w + " is not an [offset, length] pair");
            if (off < 0 || len <= 0 || len % 4 != 0) return damaged(w + ": bad offset or length");
            if (static_cast<std::uint64_t>(off) > s.length || static_cast<std::uint64_t>(len) > s.length - static_cast<std::uint64_t>(off))
                return damaged(w + " runs past the end of the source");
            s.segments.push_back(Segment{static_cast<std::uint64_t>(off), static_cast<std::uint64_t>(len)});
        }
        return true;
    }

    // Spec 48 section 5: version 2 has exactly the source T; version 3 has T and O.
    bool check_k_sources(const Map& m) {
        const std::size_t want = m.version == 2 ? 1 : 2;
        const int t = m.find_source("T"), o = m.find_source("O");
        if (t < 0) return damaged("sources: missing source T");
        if (m.version >= 3 && o < 0) return damaged("sources: missing source O");
        if (m.sources.size() != want) return damaged("sources: a version " + std::to_string(m.version) + " map has only " +
                                                     (want == 1 ? "T" : "T and O"));
        const Source& st = m.sources[static_cast<std::size_t>(t)];
        if (st.element != Element::Base64 || st.locate != Locate::Block || st.keyed)
            return damaged("sources.T: must be an unkeyed base64 block source");
        if (o >= 0) {
            const Source& so = m.sources[static_cast<std::size_t>(o)];
            if (so.element != Element::I32 || so.locate != Locate::Block || !so.keyed)
                return damaged("sources.O: must be a keyed i32le block source");
        }
        for (const Source& s : m.sources)
            if (s.element == Element::Base64 && s.name != "T") return damaged("sources." + s.name + ": only T is base64");
        return true;
    }

    // What tensors index (see Array in map.h).
    static void build_arrays(Map& m) {
        for (std::size_t i = 0; i < m.sources.size(); ++i) {
            const Source& s = m.sources[i];
            Array a;
            a.source = static_cast<int>(i);
            if (s.element == Element::Base64) {
                std::uint64_t bytes = 0;  // decoded bytes without padding: an upper bound
                for (const Segment& g : s.segments) bytes += g.length / 4 * 3;
                a.name = "W";
                a.element = Element::F16;
                a.max_elements = bytes / 2;
            } else if (s.element == Element::Hex32) {
                a.name = s.name;
                a.element = Element::U32;  // one 32-bit integer per listed field
                a.max_elements = s.variants[0].fields.size();
            } else {
                a.name = s.name;
                a.element = s.element;
                a.max_elements = s.length / a.element_size();
            }
            m.arrays.push_back(std::move(a));
        }
    }

    bool load_perm(const std::string& name, const Json& j, Map& m) {
        const std::string where = "perms." + name;
        if (!j.is_array() || j.items.empty()) return damaged(where + ": must be a non-empty list");
        Perm p;
        p.name = name;
        std::vector<bool> seen(j.items.size(), false);
        for (const Json& v : j.items) {
            std::int64_t x;
            if (!v.as_int64(x) || x < 0 || static_cast<std::uint64_t>(x) >= j.items.size() ||
                seen[static_cast<std::size_t>(x)])
                return damaged(where + ": not a permutation of 0..b-1");
            seen[static_cast<std::size_t>(x)] = true;
            p.table.push_back(x);
        }
        m.perms.push_back(std::move(p));
        return true;
    }

    bool load_op(const Json& j, const std::string& where, Op& op) {
        if (!j.is_object()) return damaged(where + ": op must be an object");
        std::string name;
        if (!need_string(j, "name", where + ".op", name)) return false;
        if (name == "f16") {
            op.kind = OpKind::F16;
        } else if (name == "fp8") {
            op.kind = OpKind::FP8;
        } else if (name == "f16_scaled") {
            op.kind = OpKind::F16Scaled;
            if (!need_f32_bits(j, "mul", where + ".op", op.mul)) return false;
            if (!need_f32_bits(j, "div", where + ".op", op.div)) return false;
        } else if (name == "exp") {
            op.kind = OpKind::Exp;
            if (j.find("max")) {
                op.has_max = true;
                if (!need_f32_bits(j, "max", where + ".op", op.max)) return false;
                if (std::isnan(op.max)) return damaged(where + ".op: max is NaN");
            }
        } else if (name == "u32_as_f32") {
            op.kind = OpKind::U32AsF32;
        } else if (name == "u16_shr") {
            op.kind = OpKind::U16Shr;
            std::int64_t shift;
            if (!need_int(j, "shift", where + ".op", shift)) return false;
            if (shift < 0 || shift > 63) return damaged(where + ".op: bad shift");
            op.shift = static_cast<int>(shift);
        } else if (name == "capped_softmax" && k_) {
            op.kind = OpKind::CappedSoftmax;
            if (!need_f32_bits(j, "cap", where + ".op", op.cap)) return false;
            if (!need_f32_bits(j, "div", where + ".op", op.div)) return false;
            if (!need_f32_bits(j, "sub", where + ".op", op.sub)) return false;
            if (!std::isfinite(op.cap) || !std::isfinite(op.div) || !std::isfinite(op.sub))
                return damaged(where + ".op: capped_softmax constants must be finite");
        } else if (name == "i32" && version_ == 3) {
            op.kind = OpKind::I32;
        } else {
            return damaged(where + ": unknown op \"" + name + "\"");
        }
        return true;
    }

    static bool checked_mul(std::int64_t a, std::int64_t b, std::int64_t& out) {
        const std::int64_t lim = std::int64_t(1) << 62;
        if (a == 0 || b == 0) {
            out = 0;
            return true;
        }
        const std::int64_t aa = a < 0 ? -a : a, bb = b < 0 ? -b : b;
        if (aa >= lim || bb >= lim || aa > lim / bb) return false;
        out = a * b;
        return true;
    }
    static bool checked_add(std::int64_t a, std::int64_t b, std::int64_t& out) {
        const std::int64_t lim = std::int64_t(1) << 62;
        if (a >= lim || a <= -lim || b >= lim || b <= -lim) return false;
        out = a + b;
        return true;
    }

    bool load_tensor(const Json& j, const std::string& stage, std::size_t idx, const Map& m, Tensor& t,
                     std::uint64_t& total) {
        std::string where = "stage " + stage + ", tensor " + std::to_string(idx);
        if (!j.is_object()) return damaged(where + ": must be an object");
        if (!need_string(j, "name", where, t.name)) return false;
        if (!safe_name(t.name)) return damaged(where + ": bad tensor name");
        where = "stage " + stage + ", tensor " + t.name;
        for (const char* v5 : {"also", "sites", "require", "allow_nonfinite"})
            if (j.find(v5)) return damaged(where + ": \"" + v5 + "\" is a version-5 feature");
        if (const Json* opj = j.find("op"))
            if (opj->is_object() && opj->find("max_src")) return damaged(where + ": \"max_src\" is a version-5 feature");

        const Json* shape;
        if (!need_array(j, "shape", where, shape)) return false;
        if (shape->items.empty()) return damaged(where + ": shape is empty");
        if (shape->items.size() > kMaxRank) return damaged(where + ": too many axes");
        std::uint64_t count = 1;
        for (const Json& v : shape->items) {
            std::int64_t n;
            if (!v.as_int64(n) || n <= 0) return damaged(where + ": bad axis length in shape");
            if (static_cast<std::uint64_t>(n) > kMaxTensorElements || count * static_cast<std::uint64_t>(n) > kMaxTensorElements)
                return damaged(where + ": tensor too large");
            count *= static_cast<std::uint64_t>(n);
            t.shape.push_back(n);
        }
        t.count = count;
        total += count;
        if (total > kMaxTotalElements) return damaged(where + ": map too large");

        std::string dtype;
        if (!need_string(j, "dtype", where, dtype)) return false;
        if (dtype == "float32") t.dtype = DType::Float32;
        else if (dtype == "uint8") t.dtype = DType::UInt8;
        else if (dtype == "int32" && m.version == 3) t.dtype = DType::Int32;
        else return damaged(where + ": unknown dtype \"" + dtype + "\"");

        std::string src;
        if (!need_string(j, "src", where, src)) return false;
        t.src = m.find_array(src);
        if (t.src < 0) return damaged(where + ": unknown source \"" + src + "\"");
        const Array& s = m.arrays[static_cast<std::size_t>(t.src)];

        const Json* op = need(j, "op", where);
        if (!op || !load_op(*op, where, t.op)) return false;

        // Each operation reads one element type and produces one dtype (08 section 6, 48 section 6).
        Element want_el;
        DType want_dt = DType::Float32;
        switch (t.op.kind) {
            case OpKind::U32AsF32: want_el = Element::U32; break;
            case OpKind::U16Shr: want_el = Element::U16; want_dt = DType::UInt8; break;
            case OpKind::I32: want_el = Element::I32; want_dt = DType::Int32; break;
            default: want_el = Element::F16; break;
        }
        if (s.element != want_el) return damaged(where + ": the op does not fit the source's element type");
        if (t.dtype != want_dt) return damaged(where + ": the op does not fit the dtype");
        // capped_softmax works on groups of 64: the last two axes must be 8 x 8 (48 section 6).
        if (t.op.kind == OpKind::CappedSoftmax &&
            (t.shape.size() < 2 || t.shape[t.shape.size() - 2] != 8 || t.shape.back() != 8))
            return damaged(where + ": capped_softmax needs last two axes of length 8");

        if (!need_int(j, "base", where, t.base)) return false;

        const Json* axes;
        if (!need_array(j, "axes", where, axes)) return false;
        if (axes->items.size() != t.shape.size()) return damaged(where + ": axes and shape have different lengths");
        for (std::size_t a = 0; a < axes->items.size(); ++a) {
            const Json& aj = axes->items[a];
            if (!aj.is_object()) return damaged(where + ": axis must be an object");
            Axis ax;
            ax.length = t.shape[a];
            if (!need_int(aj, "stride", where + ", axis " + std::to_string(a), ax.stride)) return false;
            if (aj.find("limit")) return damaged(where + ": an axis \"limit\" is a version-5 feature");
            if (const Json* pj = aj.find("perm")) {
                if (!pj->is_string()) return damaged(where + ": perm must be text");
                for (std::size_t p = 0; p < m.perms.size(); ++p)
                    if (m.perms[p].name == pj->text) ax.perm = static_cast<int>(p);
                if (ax.perm < 0) return damaged(where + ": unknown perm \"" + pj->text + "\"");
                const std::int64_t b = static_cast<std::int64_t>(m.perms[static_cast<std::size_t>(ax.perm)].table.size());
                if (ax.length % b != 0) return damaged(where + ": perm \"" + pj->text + "\" does not divide its axis");
            }
            t.axes.push_back(ax);
        }

        // Index range (section 5.1). pi_a maps 0..n-1 onto 0..n-1, so stride * pi_a(i) takes exactly
        // the values stride * j, j = 0..n-1, and the extremes of e are reached.
        std::int64_t lo = t.base, hi = t.base;
        for (const Axis& ax : t.axes) {
            std::int64_t span;
            if (!checked_mul(ax.stride, ax.length - 1, span)) return out_of_range(where + ": index overflow");
            std::int64_t& end = (span < 0) ? lo : hi;
            if (!checked_add(end, span, end)) return out_of_range(where + ": index overflow");
        }
        const std::int64_t n = static_cast<std::int64_t>(s.max_elements);
        if (lo < 0 || hi >= n)
            return out_of_range(where + ": indices " + std::to_string(lo) + ".." + std::to_string(hi) +
                                " outside source " + s.name + " (0.." + std::to_string(n - 1) + ")");
        return true;
    }

    bool load_stage(const Json& j, std::size_t idx, const Map& m, Stage& st, std::uint64_t& total) {
        std::string where = "stage " + std::to_string(idx);
        if (!j.is_object()) return damaged(where + ": must be an object");
        if (!need_string(j, "stage", where, st.name)) return false;
        if (!safe_name(st.name)) return damaged(where + ": bad stage name");
        where = "stage " + st.name;
        // One digest (text), checked here as in every map version; or, in a version-4 map, a list with one digest
        // per variant, checked below once the tensors say which source the group reads.
        for (const char* v5 : {"checks", "requires_variant"})
            if (j.find(v5)) return damaged(where + ": \"" + v5 + "\" is a version-5 feature");
        const Json* digest = need(j, "digest", where);
        if (!digest) return false;
        const bool list = version_ == 4 && digest->is_array();
        if (!list) {
            if (!digest->is_string()) return damaged(where + ": field \"digest\" must be text");
            if (!parse_digest_hex(digest->text, st.digest)) return damaged(where + ": bad digest");
        }
        const Json* tensors;
        if (!need_array(j, "tensors", where, tensors)) return false;
        if (tensors->items.empty()) return damaged(where + ": no tensors");
        std::set<std::string> names;
        for (std::size_t i = 0; i < tensors->items.size(); ++i) {
            Tensor t;
            if (!load_tensor(tensors->items[i], st.name, i, m, t, total)) return false;
            if (!names.insert(fold(t.name)).second) return damaged(where + ": duplicate tensor \"" + t.name + "\"");
            st.tensors.push_back(std::move(t));
        }
        // Spec 08 section 5: a group whose tensors all come from a source with variants has a list of digests, one
        // per variant; every other group has one digest. A group that mixes the two cannot be described.
        bool plain = false;
        for (const Tensor& t : st.tensors) {
            const int si = m.arrays[static_cast<std::size_t>(t.src)].source;
            if (!m.sources[static_cast<std::size_t>(si)].has_variants()) {
                plain = true;
            } else if (st.variant_source >= 0 && st.variant_source != si) {
                return damaged(where + ": tensors from two sources with variants");
            } else {
                st.variant_source = si;
            }
        }
        if (st.variant_source < 0) {
            if (list) return damaged(where + ": field \"digest\" must be text (no tensor reads a source with variants)");
            return true;
        }
        const Source& vs = m.sources[static_cast<std::size_t>(st.variant_source)];
        if (plain) return damaged(where + ": tensors from source " + vs.name + " and from sources without variants");
        if (!list) return damaged(where + ": field \"digest\" must be a list (one entry per variant)");
        if (digest->items.size() != vs.variants.size())
            return damaged(where + ": digest needs one entry per variant of source " + vs.name);
        for (const Json& dj : digest->items) {
            Digest d;
            if (!dj.is_string() || !parse_digest_hex(dj.text, d)) return damaged(where + ": bad digest");
            st.variant_digests.push_back(d);
        }
        return true;
    }

    // ======================================================================================================
    // Map version 5 (spec 57 section 5): preset F. Sources W (unkeyed f16le block), keyed blocks, keyed windows
    // and keyed code images (with or without variants); output groups in "groups". Every check here depends on
    // the map only and runs before any DLL is read (57 section 5.4); any failure is "converter data damaged".
    // ======================================================================================================

    bool load_v5(const Json& root, Map& m) {
        if (root.find("stages")) return damaged("map: \"stages\" is not a version-5 feature (use \"groups\")");
        const Json* sources;
        if (!need_object(root, "sources", "map", sources)) return false;
        std::set<std::string> folded;
        for (const auto& kv : sources->members) {
            if (!load_source_v5(kv.first, kv.second, m)) return false;
            if (!folded.insert(fold(kv.first)).second) return damaged("sources." + kv.first + ": duplicate source name");
        }
        const int w = m.find_source("W");
        if (w < 0) return damaged("sources: missing source W");
        {
            const Source& sw = m.sources[static_cast<std::size_t>(w)];
            if (sw.element != Element::F16 || sw.locate != Locate::Block || sw.keyed || sw.has_variants())
                return damaged("sources.W: must be an unkeyed f16le block without variants");
        }
        for (std::size_t i = 0; i < m.sources.size(); ++i) {  // one array per source, same index and name
            Array a;
            a.name = m.sources[i].name;
            a.source = static_cast<int>(i);
            a.element = m.sources[i].element;
            a.max_elements = m.sources[i].has_variants() ? 0 : m.sources[i].elements();
            m.arrays.push_back(std::move(a));
        }

        const Json* key;
        if (!need_object(root, "key", "map", key)) return false;
        if (!need_string(*key, "label", "key", m.key_label)) return false;
        std::string from;
        if (!need_string(*key, "from", "key", from)) return false;
        if (from != "W") return damaged("key: must be made from source W");
        m.key_source = w;

        const Json* perms;
        if (!need_object(root, "perms", "map", perms)) return false;
        for (const auto& kv : perms->members) {
            if (!load_perm(kv.first, kv.second, m)) return false;
        }

        const Json* groups;
        if (!need_array(root, "groups", "map", groups)) return false;
        if (groups->items.empty()) return damaged("groups: the list is empty");
        std::set<std::string> names;
        std::uint64_t total = 0;
        for (std::size_t i = 0; i < groups->items.size(); ++i) {
            Stage g;
            if (!load_group_v5(groups->items[i], i, m, g, total)) return false;
            if (!names.insert(fold(g.name)).second) return damaged("groups: duplicate group \"" + g.name + "\"");
            if (!check_group_reads_v5(m, g)) return false;
            m.stages.push_back(std::move(g));
        }
        return true;
    }

    // FNV-1a-64: exactly 16 hexadecimal digits, optionally prefixed by 0x (57 section 5.4).
    bool need_fnv16(const Json& obj, const std::string& where, std::uint64_t& out) {
        std::string s;
        if (!need_string(obj, "fnv1a64", where, s)) return false;
        const bool prefixed = s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
        const std::size_t digits = s.size() - (prefixed ? 2 : 0);
        if (digits != 16 || !parse_hex_u64(s, 16, out)) return damaged(where + ": fnv1a64 must be 16 hexadecimal digits");
        return true;
    }
    bool need_nonneg(const Json& obj, const char* key, const std::string& where, std::int64_t& out) {
        if (!need_int(obj, key, where, out)) return false;
        if (out < 0) return damaged(where + ": \"" + key + "\" must not be negative");
        return true;
    }

    bool load_source_v5(const std::string& name, const Json& j, Map& m) {
        const std::string where = "sources." + name;
        if (!safe_name(name)) return damaged(where + ": bad source name");
        if (!j.is_object()) return damaged(where + ": must be an object");
        // Version 1-4-only source fields (57 section 5.1).
        for (const char* old : {"gaps", "segments", "optional", "decoded_length", "decoded_sha256", "fields"})
            if (j.find(old)) return damaged(where + ": \"" + old + "\" is not a version-5 feature");
        Source s;
        s.name = name;
        if (!need_string(j, "what", where, s.what)) return false;
        std::string element, locate;
        if (!need_string(j, "element", where, element)) return false;
        if (element == "f16le") s.element = Element::F16;
        else if (element == "u16le") s.element = Element::U16;
        else if (element == "u32le") s.element = Element::U32;
        else return damaged(where + ": element \"" + element + "\" is not a version-5 element");
        if (!need_string(j, "locate", where, locate)) return false;
        if (locate == "block") s.locate = Locate::Block;
        else if (locate == "window") s.locate = Locate::Window;
        else return damaged(where + ": locate \"" + locate + "\" is not a version-5 locate method");
        if (const Json* keyed = j.find("keyed")) {
            if (!keyed->is_bool()) return damaged(where + ": keyed must be true or false");
            s.keyed = keyed->boolean;
        }
        // 57 section 5.4: every source except W has keyed: true (W makes the key, so it cannot be keyed).
        if (name == "W" ? s.keyed : !s.keyed)
            return damaged(where + (name == "W" ? ": W must not be keyed" : ": must be keyed (keyed: true)"));
        bool encoding_none = false;
        if (const Json* enc = j.find("encoding")) {
            if (!enc->is_string() || enc->text != "none")
                return damaged(where + ": encoding must be \"none\" in a version-5 map");
            encoding_none = true;
        }
        const std::uint64_t esize = s.element_size();
        if (const Json* vars = j.find("variants")) {
            // A code image with variants (57 section 4): no decoding step; the block fields are per variant.
            if (!encoding_none) return damaged(where + ": variants need encoding none");
            if (s.locate != Locate::Block) return damaged(where + ": a source with variants must be a block source");
            for (const char* f : {"length", "fnv1a64", "sha256", "field"})
                if (j.find(f)) return damaged(where + ": \"" + f + "\" belongs into each variant");
            if (!vars->is_array() || vars->items.empty()) return damaged(where + ": variants must be a non-empty list");
            if (vars->items.size() > kMaxVariants) return damaged(where + ": too many variants");
            for (std::size_t i = 0; i < vars->items.size(); ++i) {
                const Json& vj = vars->items[i];
                const std::string w = where + ", variant " + std::to_string(i);
                if (!vj.is_object()) return damaged(w + ": must be an object");
                for (const char* old : {"decoded_length", "decoded_sha256", "fields"})
                    if (vj.find(old)) return damaged(w + ": \"" + old + "\" is not a version-5 feature");
                Variant v;
                std::int64_t length;
                if (!need_int(vj, "length", w, length)) return false;
                if (length <= 0 || static_cast<std::uint64_t>(length) > kMaxSourceLength) return damaged(w + ": bad length");
                v.length = static_cast<std::uint64_t>(length);
                if (v.length % esize != 0) return damaged(w + ": length is not a whole number of elements");
                if (!need_fnv16(vj, w, v.fnv)) return false;
                std::string sha;
                if (!need_string(vj, "sha256", w, sha)) return false;
                if (!parse_digest_hex(sha, v.sha)) return damaged(w + ": sha256 must be 64 hexadecimal digits");
                s.variants.push_back(std::move(v));
            }
            m.sources.push_back(std::move(s));
            return true;
        }
        std::int64_t length;
        if (!need_int(j, "length", where, length)) return false;
        if (length <= 0 || static_cast<std::uint64_t>(length) > kMaxSourceLength) return damaged(where + ": bad length");
        s.length = static_cast<std::uint64_t>(length);
        if (s.length % esize != 0) return damaged(where + ": length is not a whole number of elements");
        if (s.locate == Locate::Window) {
            std::int64_t field;
            if (!need_nonneg(j, "field", where, field)) return false;
            if (static_cast<std::uint64_t>(field) > s.length || s.length - static_cast<std::uint64_t>(field) < esize)
                return damaged(where + ": the field lies outside its window");
            s.field = static_cast<std::uint64_t>(field);
        } else if (j.find("field")) {
            return damaged(where + ": only a window source has a field");
        }
        if (!need_fnv16(j, where, s.fnv)) return false;
        std::string sha;
        if (!need_string(j, "sha256", where, sha)) return false;
        if (!parse_digest_hex(sha, s.sha)) return damaged(where + ": sha256 must be 64 hexadecimal digits");
        m.sources.push_back(std::move(s));
        return true;
    }

    // The source a name refers to; -2 (with the error set) if there is none.
    int source_v5(const Map& m, const Json& j, const char* key, const std::string& where) {
        std::string name;
        if (!need_string(j, key, where, name)) return -2;
        const int s = m.find_source(name);
        if (s < 0) {
            damaged(where + ": unknown source \"" + name + "\"");
            return -2;
        }
        return s;
    }
    static bool is_word_window(const Source& s) { return s.locate == Locate::Window && s.element == Element::U32; }

    // Whether an indexed operation may read source s (57 section 5.4: element size and kind).
    static bool op_reads(OpKind k, const Source& s) {
        switch (k) {
            case OpKind::U16: return s.element == Element::U16;
            case OpKind::U32: case OpKind::U32AsF32: return s.element == Element::U32;
            case OpKind::E4M3: case OpKind::F16Bits: case OpKind::Exp5:
                return s.element == Element::F16 || (s.element == Element::U16 && s.locate == Locate::Window);
            default: return false;
        }
    }

    bool load_op_v5(const Json& j, const std::string& where, const Map& m, Tensor& t, const Stage& g) {
        if (!j.is_object()) return damaged(where + ": op must be an object");
        std::string name;
        if (!need_string(j, "name", where + ".op", name)) return false;
        Op& op = t.op;
        DType want;
        if (name == "e4m3") op.kind = OpKind::E4M3, want = DType::E4M3;
        else if (name == "f16") op.kind = OpKind::F16Bits, want = DType::Float16;
        else if (name == "u16") op.kind = OpKind::U16, want = DType::UInt16;
        else if (name == "u32") op.kind = OpKind::U32, want = DType::UInt32;
        else if (name == "u32_as_f32") op.kind = OpKind::U32AsF32, want = DType::Float32;
        else if (name == "exp") op.kind = OpKind::Exp5, want = DType::Float32;
        else if (name == "score_bias") op.kind = OpKind::ScoreBias, want = DType::Float16;
        else if (name == "const_u32_as_f32") op.kind = OpKind::ConstU32AsF32, want = DType::Float32;
        else if (name == "const_f16x2") op.kind = OpKind::ConstF16x2, want = DType::Float16;
        else if (name == "const_f16lo") op.kind = OpKind::ConstF16lo, want = DType::Float16;
        else if (name == "const_f32_to_f16") op.kind = OpKind::ConstF32ToF16, want = DType::Float16;
        else if (name == "attn_wmax") op.kind = OpKind::AttnWmax, want = DType::Float16;
        else return damaged(where + ": unknown op \"" + name + "\" (version 5)");
        if (t.dtype != want) return damaged(where + ": the dtype does not match the op's output type");
        // Parameters of version 1-4 operations.
        for (const char* old : {"max", "mul", "div", "shift", "cap", "sub"})
            if (j.find(old)) return damaged(where + ".op: \"" + old + "\" is a parameter of a version 1-4 operation");
        if (const Json* ng = j.find("negate")) {
            if (op.kind != OpKind::ConstF32ToF16) return damaged(where + ".op: only const_f32_to_f16 has negate");
            if (!ng->is_bool()) return damaged(where + ".op: negate must be true or false");
            op.negate = ng->boolean;
        }
        if (j.find("max_src")) {
            if (op.kind != OpKind::Exp5) return damaged(where + ".op: only exp has max_src");
            op.max_src = source_v5(m, j, "max_src", where + ".op");
            if (op.max_src < 0) return false;
            if (!is_word_window(m.sources[static_cast<std::size_t>(op.max_src)]))
                return damaged(where + ".op: max_src must be a 32-bit window source");
        }
        if (op.kind == OpKind::ScoreBias) {
            op.a_src = source_v5(m, j, "a_src", where + ".op");
            if (op.a_src < 0) return false;
            op.w_src = source_v5(m, j, "w_src", where + ".op");
            if (op.w_src < 0) return false;
            if (!is_word_window(m.sources[static_cast<std::size_t>(op.a_src)]) ||
                !is_word_window(m.sources[static_cast<std::size_t>(op.w_src)]))
                return damaged(where + ".op: a_src and w_src must be 32-bit window sources");
            const Json* pairs;
            if (!need_array(j, "pairs", where + ".op", pairs)) return false;
            if (pairs->items.size() != 12) return damaged(where + ".op: pairs must have 12 entries");
            for (const Json& p : pairs->items) {
                std::int64_t I, J;
                if (!p.is_array() || p.items.size() != 2 || !p.items[0].as_int64(I) || !p.items[1].as_int64(J) || I < 0 ||
                    I > 3 || J < 0 || J > 3)
                    return damaged(where + ".op: a pair is not [I, J] with I and J in 0..3");
                op.pairs.emplace_back(static_cast<int>(I), static_cast<int>(J));
            }
        } else if (j.find("a_src") || j.find("w_src") || j.find("pairs")) {
            return damaged(where + ".op: only score_bias has a_src, w_src and pairs");
        }
        static const char* const kIn[4] = {"clamp_lo", "clamp_hi", "p1", "p0"};
        if (op.kind == OpKind::AttnWmax) {
            for (int k = 0; k < 4; ++k) {
                std::string in;
                if (!need_string(j, kIn[k], where + ".op", in)) return false;
                const int ti = g.find_tensor(in);  // only the tensors before this one are in g so far
                if (ti < 0) return damaged(where + ".op: " + kIn[k] + " must name an earlier tensor of the group");
                const Tensor& it = g.tensors[static_cast<std::size_t>(ti)];
                if (it.count != 1 || it.dtype != DType::Float16)
                    return damaged(where + ".op: " + kIn[k] + " must be a scalar float16 tensor");
                op.wmax_in[k] = ti;
            }
        } else {
            for (int k = 0; k < 4; ++k)
                if (j.find(kIn[k])) return damaged(where + ".op: only attn_wmax has " + kIn[k]);
        }
        return true;
    }

    bool load_require_v5(const Json& j, const std::string& where, Tensor& t) {
        if (const Json* al = j.find("allow_nonfinite")) {
            if (!al->is_bool()) return damaged(where + ": allow_nonfinite must be true or false");
            t.allow_nonfinite = al->boolean;
        }
        const Json* req = j.find("require");
        if (!req) return true;
        if (!req->is_array()) return damaged(where + ": require must be a list");
        for (const Json& c : req->items) {
            if (!c.is_string()) return damaged(where + ": a require condition must be text");
            if (c.text == "finite") t.require |= kReqFinite;
            else if (c.text == "not_subnormal") t.require |= kReqNotSubnormal;
            else if (c.text == "even") t.require |= kReqEven;
            else if (c.text == "below_8") t.require |= kReqBelow8;
            else return damaged(where + ": unknown require condition \"" + c.text + "\"");
        }
        return true;
    }

    bool load_tensor_v5(const Json& j, const std::string& group, std::size_t idx, const Map& m, const Stage& g,
                        Tensor& t, std::uint64_t& total) {
        std::string where = "group " + group + ", tensor " + std::to_string(idx);
        if (!j.is_object()) return damaged(where + ": must be an object");
        if (!need_string(j, "name", where, t.name)) return false;
        if (!safe_name(t.name)) return damaged(where + ": bad tensor name");
        where = "group " + group + ", tensor " + t.name;

        const Json* shape;
        if (!need_array(j, "shape", where, shape)) return false;
        if (shape->items.empty()) return damaged(where + ": shape is empty");
        if (shape->items.size() > kMaxRank) return damaged(where + ": too many axes");
        std::uint64_t count = 1;
        for (const Json& v : shape->items) {
            std::int64_t n;
            if (!v.as_int64(n) || n <= 0) return damaged(where + ": shape dimensions must be positive integers");
            if (static_cast<std::uint64_t>(n) > kMaxTensorElements || count * static_cast<std::uint64_t>(n) > kMaxTensorElements)
                return damaged(where + ": tensor too large");
            count *= static_cast<std::uint64_t>(n);
            t.shape.push_back(n);
        }
        t.count = count;
        total += count;
        if (total > kMaxTotalElements) return damaged(where + ": map too large");

        std::string dtype;
        if (!need_string(j, "dtype", where, dtype)) return false;
        if (dtype == "e4m3") t.dtype = DType::E4M3;
        else if (dtype == "float16") t.dtype = DType::Float16;
        else if (dtype == "float32") t.dtype = DType::Float32;
        else if (dtype == "uint16") t.dtype = DType::UInt16;
        else if (dtype == "uint32") t.dtype = DType::UInt32;
        else return damaged(where + ": dtype \"" + dtype + "\" is not a version-5 dtype");

        const Json* op = need(j, "op", where);
        if (!op || !load_op_v5(*op, where, m, t, g)) return false;
        if (!load_require_v5(j, where, t)) return false;
        const OpKind k = t.op.kind;
        const bool scalar = t.shape.size() == 1 && t.shape[0] == 1;

        if (is_const_op(k) || k == OpKind::AttnWmax) {
            for (const char* f : {"src", "base", "axes", "also"})
                if (j.find(f))
                    return damaged(where + ": a " + std::string(is_const_op(k) ? "const" : "attn_wmax") +
                                   " operation has no \"" + f + "\"");
            if (!scalar) return damaged(where + ": shape must be [1]");
            if (k == OpKind::AttnWmax) {
                if (j.find("sites")) return damaged(where + ": attn_wmax has no sites");
                return true;
            }
            return load_sites_v5(j, where, m, t);
        }
        if (j.find("sites")) return damaged(where + ": only a const operation has sites");

        t.src = source_v5(m, j, "src", where);
        if (t.src < 0) return false;
        const Source& src = m.sources[static_cast<std::size_t>(t.src)];
        std::int64_t base;
        if (!need_nonneg(j, "base", where, base)) return false;
        t.base = base;
        const Json* axes;
        if (!need_array(j, "axes", where, axes)) return false;

        if (k == OpKind::ScoreBias) {
            // 57 section 5.4: src = W, axes = [], shape [H][12][16][16], base + H*4096 <= elements of W.
            if (src.name != "W") return damaged(where + ": score_bias must read W");
            if (!axes->items.empty()) return damaged(where + ": score_bias has axes = []");
            if (j.find("also")) return damaged(where + ": score_bias has no also");
            if (t.shape.size() != 4 || t.shape[1] != 12 || t.shape[2] != 16 || t.shape[3] != 16)
                return damaged(where + ": score_bias needs shape [H][12][16][16]");
            const std::uint64_t need_el = static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(t.shape[0]) * 4096;
            if (need_el > src.elements()) return damaged(where + ": score_bias reads beyond the end of W");
            return true;
        }

        if (!op_reads(k, src)) return damaged(where + ": the op's element size or kind does not fit source " + src.name);
        if (axes->items.size() != t.shape.size()) return damaged(where + ": axes and shape have different lengths");
        for (std::size_t a = 0; a < axes->items.size(); ++a) {
            const Json& aj = axes->items[a];
            const std::string wa = where + ", axis " + std::to_string(a);
            if (!aj.is_object()) return damaged(wa + ": must be an object");
            Axis ax;
            ax.length = t.shape[a];
            if (!need_int(aj, "stride", wa, ax.stride)) return false;
            if (ax.stride < 0) return damaged(wa + ": negative strides are not a version-5 feature");
            if (aj.find("limit") && !need_nonneg(aj, "limit", wa, ax.limit)) return false;
            if (const Json* pj = aj.find("perm")) {
                if (!pj->is_string()) return damaged(wa + ": perm must be text");
                for (std::size_t p = 0; p < m.perms.size(); ++p)
                    if (m.perms[p].name == pj->text) ax.perm = static_cast<int>(p);
                if (ax.perm < 0) return damaged(wa + ": unknown perm \"" + pj->text + "\"");
                const std::int64_t b = static_cast<std::int64_t>(m.perms[static_cast<std::size_t>(ax.perm)].table.size());
                if (ax.length % b != 0) return damaged(wa + ": perm \"" + pj->text + "\" does not divide its axis");
            }
            t.axes.push_back(ax);
        }
        if (const Json* also = j.find("also")) {
            if (!also->is_array() || also->items.empty()) return damaged(where + ": also must be a non-empty list");
            for (const Json& aj : also->items) {
                if (!aj.is_object()) return damaged(where + ": an also entry must be an object");
                AlsoRead r;
                r.src = source_v5(m, aj, "src", where + ", also");
                if (r.src < 0) return false;
                if (!op_reads(k, m.sources[static_cast<std::size_t>(r.src)]))
                    return damaged(where + ": an also source does not fit the op");
                std::int64_t b;
                if (!need_nonneg(aj, "base", where + ", also", b)) return false;
                r.base = b;
                t.also.push_back(r);
            }
        }
        return true;
    }

    bool load_sites_v5(const Json& j, const std::string& where, const Map& m, Tensor& t) {
        const Json* sites;
        if (!need_array(j, "sites", where, sites)) return false;
        std::size_t total = 0;
        for (std::size_t i = 0; i < sites->items.size(); ++i) {
            const Json& sj = sites->items[i];
            const std::string w = where + ", site " + std::to_string(i);
            if (!sj.is_object()) return damaged(w + ": must be an object");
            Site site;
            site.src = source_v5(m, sj, "src", w);
            if (site.src < 0) return false;
            const Source& s = m.sources[static_cast<std::size_t>(site.src)];
            if (s.element != Element::U32) return damaged(w + ": a site source must have 32-bit elements");
            const Json* fixed = sj.find("elements");
            const Json* byv = sj.find("elements_by_variant");
            if ((fixed != nullptr) == (byv != nullptr))
                return damaged(w + ": exactly one of elements and elements_by_variant");
            auto read_list = [&](const Json& lj, std::vector<std::uint64_t>& out) {
                if (!lj.is_array()) return damaged(w + ": an element list must be a list");
                for (const Json& e : lj.items) {
                    std::int64_t v;
                    if (!e.as_int64(v) || v < 0) return damaged(w + ": site indices must be nonnegative integers");
                    out.push_back(static_cast<std::uint64_t>(v));
                    if (++total > kMaxSiteElements) return damaged(w + ": too many site elements");
                }
                return true;
            };
            if (fixed) {
                if (!read_list(*fixed, site.elements)) return false;
            } else {
                if (!s.has_variants()) return damaged(w + ": elements_by_variant needs a source with variants");
                if (!byv->is_array() || byv->items.size() != s.variants.size())
                    return damaged(w + ": elements_by_variant needs one list per variant of " + s.name);
                site.by_variant = true;
                for (const Json& lj : byv->items) {
                    site.per_variant.emplace_back();
                    if (!read_list(lj, site.per_variant.back())) return false;
                }
            }
            t.sites.push_back(std::move(site));
        }
        return true;
    }

    bool load_group_v5(const Json& j, std::size_t idx, const Map& m, Stage& g, std::uint64_t& total) {
        std::string where = "group " + std::to_string(idx);
        if (!j.is_object()) return damaged(where + ": must be an object");
        if (!need_string(j, "group", where, g.name)) return false;
        if (!safe_name(g.name)) return damaged(where + ": bad group name");
        where = "group " + g.name;
        const Json* digest = need(j, "digest", where);
        if (!digest) return false;
        if (!digest->is_string()) return damaged(where + ": digest must be one 64-hex-digit string");
        if (!parse_digest_hex(digest->text, g.digest)) return damaged(where + ": digest must be 64 hexadecimal digits");
        if (const Json* rv = j.find("requires_variant")) {
            if (!rv->is_object() || rv->members.empty()) return damaged(where + ": requires_variant must be {source: index}");
            for (const auto& kv : rv->members) {
                const int s = m.find_source(kv.first);
                if (s < 0) return damaged(where + ": requires_variant names unknown source \"" + kv.first + "\"");
                if (!m.sources[static_cast<std::size_t>(s)].has_variants())
                    return damaged(where + ": requires_variant names source " + kv.first + ", which has no variants");
                std::int64_t v;
                if (!kv.second.as_int64(v) || v < 0 ||
                    static_cast<std::uint64_t>(v) >= m.sources[static_cast<std::size_t>(s)].variants.size())
                    return damaged(where + ": requires_variant index out of range for " + kv.first);
                g.requires_variant.emplace_back(s, static_cast<int>(v));
            }
        }
        const Json* tensors;
        if (!need_array(j, "tensors", where, tensors)) return false;
        if (tensors->items.empty()) return damaged(where + ": no tensors");
        std::set<std::string> names;
        for (std::size_t i = 0; i < tensors->items.size(); ++i) {
            Tensor t;
            if (!load_tensor_v5(tensors->items[i], g.name, i, m, g, t, total)) return false;
            if (!names.insert(fold(t.name)).second) return damaged(where + ": duplicate tensor \"" + t.name + "\"");
            g.tensors.push_back(std::move(t));
        }
        if (const Json* checks = j.find("checks")) {
            if (!checks->is_array()) return damaged(where + ": checks must be a list");
            for (const Json& cj : checks->items) {
                Check c;
                if (!load_check_v5(cj, where, g, c)) return false;
                g.checks.push_back(std::move(c));
            }
        }
        return true;
    }

    bool load_check_v5(const Json& j, const std::string& where, const Stage& g, Check& c) {
        if (!j.is_object()) return damaged(where + ": a check must be an object");
        std::string rel;
        if (!need_string(j, "rel", where + ", check", rel)) return false;
        if (rel == "ge") c.rel = Rel::Ge;
        else if (rel == "gt") c.rel = Rel::Gt;
        else if (rel == "le") c.rel = Rel::Le;
        else if (rel == "ratio_below_f16_overflow") c.rel = Rel::RatioBelowF16Overflow;
        else return damaged(where + ": unknown check relation \"" + rel + "\"");
        auto scalar_tensor = [&](const std::string& n) {
            const int ti = g.find_tensor(n);
            if (ti < 0 || g.tensors[static_cast<std::size_t>(ti)].count != 1) return -1;
            return ti;
        };
        const Json* aj = need(j, "a", where + ", check");
        if (!aj) return false;
        if (!aj->is_string()) return damaged(where + ": check operand a must name a tensor");  // const: operand b only
        c.a = scalar_tensor(aj->text);
        if (c.a < 0) return damaged(where + ": check operand \"" + aj->text + "\" is not a scalar tensor of the group");
        const Json* bj = need(j, "b", where + ", check");
        if (!bj) return false;
        std::string btext;
        if (bj->is_string()) {
            c.b = scalar_tensor(bj->text);
            if (c.b < 0) return damaged(where + ": check operand \"" + bj->text + "\" is not a scalar tensor of the group");
            btext = bj->text;
        } else if (bj->is_object() && bj->members.size() == 1 && bj->find("const")) {
            const Json& n = *bj->find("const");
            float f;
            if (!n.is_number() || !json_number_exact_f32(n.text, f))
                return damaged(where + ": a check constant must be exactly a binary32 number");
            c.b = -1;
            c.b_const = static_cast<double>(f);
            btext = "const";
        } else {
            return damaged(where + ": check operand b must name a tensor or be {\"const\": n}");
        }
        c.text = rel + " " + aj->text + " " + btext;
        return true;
    }

    // ---- 57 section 5.4, last rule: every source-variant assignment consistent with the group's gate ----

    // The variants of source s that a produced group can see: only the gated one, or all of them ({-1} for a
    // source without variants).
    static std::vector<int> allowed_variants(const Source& s, int si, const Stage& g) {
        if (!s.has_variants()) return {-1};
        for (const auto& rv : g.requires_variant)
            if (rv.first == si) return {rv.second};
        std::vector<int> v;
        for (std::size_t i = 0; i < s.variants.size(); ++i) v.push_back(static_cast<int>(i));
        return v;
    }

    bool check_group_reads_v5(const Map& m, const Stage& g) {
        for (const Tensor& t : g.tensors) {
            const std::string where = "group " + g.name + ", tensor " + t.name;
            if (is_const_op(t.op.kind)) {
                // Bounds of every selected site element, per allowed variant of its source.
                for (std::size_t i = 0; i < t.sites.size(); ++i) {
                    const Site& site = t.sites[i];
                    const Source& s = m.sources[static_cast<std::size_t>(site.src)];
                    for (int v : allowed_variants(s, site.src, g)) {
                        const std::uint64_t n = s.elements(v);
                        for (std::uint64_t e : (site.by_variant ? site.per_variant[static_cast<std::size_t>(v)] : site.elements))
                            if (e >= n)
                                return damaged(where + ": site " + std::to_string(i) + " index " + std::to_string(e) +
                                               " outside source " + s.name +
                                               (v >= 0 ? " (variant " + std::to_string(v) + ")" : ""));
                    }
                }
                // At least one selected element under every assignment: an empty assignment exists iff every
                // source the sites name has an allowed variant for which all of its sites select nothing.
                bool can_be_empty = true;
                std::vector<int> seen;
                for (const Site& site : t.sites) {
                    if (std::find(seen.begin(), seen.end(), site.src) != seen.end()) continue;
                    seen.push_back(site.src);
                    const Source& s = m.sources[static_cast<std::size_t>(site.src)];
                    bool some_variant_empty = false;
                    for (int v : allowed_variants(s, site.src, g)) {
                        bool all_empty = true;
                        for (const Site& o : t.sites)
                            if (o.src == site.src &&
                                !(o.by_variant ? o.per_variant[static_cast<std::size_t>(v)] : o.elements).empty())
                                all_empty = false;
                        some_variant_empty = some_variant_empty || all_empty;
                    }
                    can_be_empty = can_be_empty && some_variant_empty;
                }
                if (can_be_empty) return damaged(where + ": a const tensor with no selected site in a produced group");
                continue;
            }
            if (!is_indexed_op(t.op.kind)) continue;
            // Index range of the reads (57 section 5.3): elements with an axis at or beyond its limit read nothing.
            std::int64_t hi = 0;
            bool any = true;
            for (const Axis& ax : t.axes) {
                const std::int64_t n = ax.limit >= 0 ? std::min(ax.limit, ax.length) : ax.length;
                if (n == 0) {
                    // The tensor reads nothing, but the axes after this one are still checked: the map is validated
                    // completely before any input is read (an empty axis adds no span).
                    any = false;
                    continue;
                }
                std::int64_t span;
                if (!checked_mul(ax.stride, n - 1, span) || !checked_add(hi, span, hi))
                    return damaged(where + ": index overflow");
            }
            if (!any) continue;
            std::vector<std::pair<int, std::int64_t>> reads = {{t.src, t.base}};
            for (const AlsoRead& r : t.also) reads.emplace_back(r.src, r.base);
            for (const auto& rd : reads) {
                const Source& s = m.sources[static_cast<std::size_t>(rd.first)];
                std::int64_t top;
                if (!checked_add(rd.second, hi, top)) return damaged(where + ": index overflow");
                for (int v : allowed_variants(s, rd.first, g))
                    if (static_cast<std::uint64_t>(top) >= s.elements(v))
                        return damaged(where + ": indices up to " + std::to_string(top) + " outside source " + s.name +
                                       (v >= 0 ? " (variant " + std::to_string(v) + ")" : ""));
            }
        }
        return true;
    }

private:
    MapError& err_;
    int version_ = 0;
    bool k_ = false;  // versions 2 and 3 (preset K, spec 48)
};

}  // namespace

const std::vector<PresetInfo>& known_presets() {
    // the map versions each preset's maps use (bit v: version v), as they were before maps named their preset
    static const std::vector<PresetInfo> presets = {
        {"M", (1u << 1) | (1u << 4), kMsgNoPresetM},  // spec 08 (version 4 adds the source F)
        {"K", (1u << 2) | (1u << 3), kMsgNoPresetK},  // spec 48
        {"F", 1u << 5, kMsgNoPresetF},                // spec 57
        // Model L (#89): the same record kinds as preset M's map (no new element format, op or source kind), so the
        // same map versions; its parameters are a block of their own in the same library file, found by its own hashes
        {"L", (1u << 1) | (1u << 4), kMsgNoPresetL},
    };
    return presets;
}

const PresetInfo* find_preset(const std::string& name) {
    for (const PresetInfo& p : known_presets())
        if (name == p.name) return &p;
    return nullptr;
}

std::string preset_for_version(int version) {
    for (const PresetInfo& p : known_presets())
        if (p.allows(version)) return p.name;
    return "";
}

bool load_map(const std::string& file_bytes, Map& out, MapError& err) {
    out = Map();
    err = MapError();
    out.file_sha256 = Sha256::hash(file_bytes.data(), file_bytes.size());
    Json root;
    std::string jerr;
    if (!parse_json(file_bytes, root, jerr)) {
        err.kind = MapErrorKind::Damaged;
        err.detail = "not valid JSON: " + jerr;
        return false;
    }
    Loader loader(err);
    return loader.load(root, out);
}

}  // namespace wgc
