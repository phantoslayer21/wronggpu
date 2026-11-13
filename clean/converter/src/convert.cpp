// SPDX-License-Identifier: MIT
// Decoding the tensors (spec 08 sections 4.3, 5.1, 6, 7; spec 48 sections 4, 6, 7) and checking the digests (08
// section 7.3). Written for this project.
#include "convert.h"

#include <cmath>

#include "lz4block.h"
#include "numeric.h"

namespace wgc {

Digest make_key(const std::string& label, const std::vector<std::uint8_t>& key_source_bytes) {
    Sha256 s;
    s.update(label.data(), label.size());
    s.update(key_source_bytes.data(), key_source_bytes.size());
    return s.finish();
}

namespace {

// RFC 4648 section 4 alphabet; -1 for every other byte. '=' is handled separately.
int b64_value(std::uint8_t c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

}  // namespace

bool decode_base64(const std::uint8_t* text, std::size_t n, std::vector<std::uint8_t>& out) {
    if (n % 4 != 0) return false;
    for (std::size_t i = 0; i < n; i += 4) {
        const bool last = i + 4 == n;
        // '=' only in the last quad: "xx==" or "xxx=".
        int pad = 0;
        if (last && text[i + 3] == '=') pad = (text[i + 2] == '=') ? 2 : 1;
        int v[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4 - pad; ++k) {
            v[k] = b64_value(text[i + k]);
            if (v[k] < 0) return false;
        }
        const std::uint32_t q = (std::uint32_t(v[0]) << 18) | (std::uint32_t(v[1]) << 12) | (std::uint32_t(v[2]) << 6) |
                                std::uint32_t(v[3]);
        out.push_back(std::uint8_t(q >> 16));
        if (pad < 2) out.push_back(std::uint8_t(q >> 8));
        if (pad < 1) out.push_back(std::uint8_t(q));
    }
    return true;
}

bool decode_variant_block(const Variant& v, const std::vector<std::uint8_t>& block, const Digest& key,
                          std::vector<std::uint8_t>& elements, std::string& why) {
    elements.clear();
    std::vector<std::uint8_t> text;  // local: nothing of it leaves this function except the listed fields
    if (block.size() != v.length ||
        !lz4_block_decode(block.data(), block.size(), static_cast<std::size_t>(v.decoded_length), text)) {
        why = "the block does not decode to " + std::to_string(v.decoded_length) + " bytes";
        return false;
    }
    if (keyed_sha256(key, text.data(), text.size()) != v.decoded_sha) {
        why = "the decoded block does not match decoded_sha256";
        return false;
    }
    for (std::size_t e = 0; e < v.fields.size(); ++e) {
        const std::uint64_t at = v.fields[e];
        if (at > text.size() || text.size() - at < 8) {  // load_map checked this against decoded_length
            elements.clear();
            why ="field " + std::to_string(e) + " lies outside the decoded block";
            return false;
        }
        std::uint32_t x = 0;
        for (std::size_t i = 0; i < 8; ++i) {  // most significant digit first
            const std::uint8_t c = text[static_cast<std::size_t>(at) + i];
            std::uint32_t d;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
            else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
            else {
                elements.clear();
                why = "field " + std::to_string(e) + " is not eight hexadecimal digits";
                return false;
            }
            x = (x << 4) | d;
        }
        for (int i = 0; i < 4; ++i) elements.push_back(std::uint8_t(x >> (8 * i)));
    }
    return true;
}

namespace {

std::uint64_t read_element(const std::vector<std::uint8_t>& src, Element el, std::uint64_t e) {
    if (el == Element::U32 || el == Element::I32) {
        const std::uint8_t* p = src.data() + 4 * e;
        return std::uint64_t(p[0]) | (std::uint64_t(p[1]) << 8) | (std::uint64_t(p[2]) << 16) |
               (std::uint64_t(p[3]) << 24);
    }
    const std::uint8_t* p = src.data() + 2 * e;
    return std::uint64_t(p[0]) | (std::uint64_t(p[1]) << 8);
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t b) {
    out.push_back(std::uint8_t(b));
    out.push_back(std::uint8_t(b >> 8));
    out.push_back(std::uint8_t(b >> 16));
    out.push_back(std::uint8_t(b >> 24));
}

bool fail(DecodeError& err, DecodeErrorKind kind, const std::string& detail) {
    err.kind = kind;
    err.detail = detail;
    return false;
}

// The element arrays tensors index: a copy of each source, except W, the decoded segments of T (spec 48 section 4).
bool build_arrays(const Map& map, const std::vector<std::vector<std::uint8_t>>& sources,
                  std::vector<std::vector<std::uint8_t>>& arrays, DecodeError& err) {
    arrays.assign(map.arrays.size(), {});
    for (std::size_t i = 0; i < map.arrays.size(); ++i) {
        const Array& a = map.arrays[i];
        const Source& s = map.sources[static_cast<std::size_t>(a.source)];
        const std::vector<std::uint8_t>& bytes = sources[static_cast<std::size_t>(a.source)];
        if (s.element != Element::Base64) {
            arrays[i] = bytes;
            continue;
        }
        std::vector<std::uint8_t>& w = arrays[i];
        w.reserve(static_cast<std::size_t>(a.max_elements * 2 + 2));
        for (std::size_t k = 0; k < s.segments.size(); ++k) {
            const Segment& g = s.segments[k];
            if (g.offset + g.length > bytes.size() ||
                !decode_base64(bytes.data() + g.offset, static_cast<std::size_t>(g.length), w))
                return fail(err, DecodeErrorKind::OutOfRange,
                            "source " + s.name + ": segment " + std::to_string(k) + " is not valid Base64");
        }
        if (w.size() % 2 != 0)
            return fail(err, DecodeErrorKind::OutOfRange, "source " + s.name + ": the decoded length is odd");
    }
    return true;
}

bool decode_tensor(const Map& map, const Tensor& t, const std::vector<std::uint8_t>& src, TensorOut& out,
                   DecodeError& err, const std::string& stage) {
    const Array& s = map.arrays[static_cast<std::size_t>(t.src)];
    const std::size_t rank = t.axes.size();
    const std::string where = stage + "/" + t.name + ": element ";

    // Per-axis contribution stride_a * pi_a(i) for every i (section 5.1).
    std::vector<std::vector<std::int64_t>> contrib(rank);
    for (std::size_t a = 0; a < rank; ++a) {
        const Axis& ax = t.axes[a];
        contrib[a].resize(static_cast<std::size_t>(ax.length));
        for (std::int64_t i = 0; i < ax.length; ++i) {
            std::int64_t pi = i;
            if (ax.perm >= 0) {
                const std::vector<std::int64_t>& q = map.perms[static_cast<std::size_t>(ax.perm)].table;
                const std::int64_t b = static_cast<std::int64_t>(q.size());
                pi = b * (i / b) + q[static_cast<std::size_t>(i % b)];
            }
            contrib[a][static_cast<std::size_t>(i)] = ax.stride * pi;  // range-checked by load_map
        }
    }

    out.tensor = &t;
    out.bytes.clear();
    out.bytes.reserve(static_cast<std::size_t>(t.byte_size()));
    std::vector<std::int64_t> idx(rank, 0);
    // The exact element count (for W from T, after decoding; load_map checked against an upper bound only).
    const std::int64_t n_src = static_cast<std::int64_t>(src.size() / s.element_size());
    double group[64];  // capped_softmax: the 64 inputs of the current group (row-major s = 8 i + j)
    for (std::uint64_t k = 0; k < t.count; ++k) {
        std::int64_t e = t.base;
        for (std::size_t a = 0; a < rank; ++a) e += contrib[a][static_cast<std::size_t>(idx[a])];
        if (e < 0 || e >= n_src)
            return fail(err, DecodeErrorKind::OutOfRange, where + std::to_string(k) + " reads outside source " + s.name);
        const std::uint64_t x = read_element(src, s.element, static_cast<std::uint64_t>(e));

        bool finite_input = true;
        double xv = 0.0;
        if (s.element == Element::F16) {
            finite_input = half_is_finite(static_cast<std::uint16_t>(x));
            xv = half_value(static_cast<std::uint16_t>(x));
        }

        if (t.op.kind == OpKind::U16Shr) {
            const std::uint64_t y = (t.op.shift >= 64) ? 0 : (x >> t.op.shift);
            if (y > 255)
                return fail(err, DecodeErrorKind::OutOfRange, where + std::to_string(k) + " does not fit uint8");
            out.bytes.push_back(static_cast<std::uint8_t>(y));
        } else if (t.op.kind == OpKind::I32) {
            put_u32(out.bytes, static_cast<std::uint32_t>(x));  // the int32 bit pattern, unchanged
        } else if (t.op.kind == OpKind::CappedSoftmax) {
            if (!finite_input) return fail(err, DecodeErrorKind::NonFinite, where + std::to_string(k) + " is not finite");
            group[k % 64] = xv;
            if (k % 64 == 63) {
                float y[64];
                capped_softmax64(group, t.op.cap, t.op.div, t.op.sub, y);
                for (int i = 0; i < 64; ++i) {
                    if (!std::isfinite(y[i]))
                        return fail(err, DecodeErrorKind::NonFinite, where + std::to_string(k - 63 + i) + " is not finite");
                    put_u32(out.bytes, float_to_bits(y[i]));
                }
            }
        } else {
            float y = 0.0f;
            switch (t.op.kind) {
                case OpKind::F16: y = static_cast<float>(xv); break;  // exact
                case OpKind::FP8:
                    if (!finite_input)
                        return fail(err, DecodeErrorKind::NonFinite, where + std::to_string(k) + " is not finite");
                    y = q8(xv);
                    break;
                case OpKind::F16Scaled: y = f16_scaled(xv, t.op.mul, t.op.div); break;
                case OpKind::Exp:
                    y = exp_rn32(xv);
                    if (t.op.has_max && y > t.op.max) y = t.op.max;  // min(y, max); a NaN stays NaN
                    break;
                case OpKind::U32AsF32: y = bits_to_float(static_cast<std::uint32_t>(x)); break;
                default: break;
            }
            if (!std::isfinite(y)) return fail(err, DecodeErrorKind::NonFinite, where + std::to_string(k) + " is not finite");
            put_u32(out.bytes, float_to_bits(y));
        }

        // Next index, row-major (last axis fastest).
        for (std::size_t a = rank; a-- > 0;) {
            if (++idx[a] < t.axes[a].length) break;
            idx[a] = 0;
        }
    }
    return true;
}

}  // namespace

bool convert_all(const Map& map, const std::vector<std::vector<std::uint8_t>>& sources,
                 const std::vector<int>& variants, const Digest& key, std::vector<StageOut>& out, DecodeError& err) {
    out.clear();
    err = DecodeError();
    std::vector<std::vector<std::uint8_t>> arrays;
    if (!build_arrays(map, sources, arrays, err)) return false;
    for (const Stage& st : map.stages) {
        StageOut so;
        so.stage = &st;
        if (st.optional()) {
            // Spec 08 sections 4.3 and 7.2: the group exists only when a variant of its source was found.
            const std::size_t vs = static_cast<std::size_t>(st.variant_source);
            so.variant = vs < variants.size() ? variants[vs] : -1;
            if (so.variant < 0) continue;
            if (static_cast<std::size_t>(so.variant) >= st.variant_digests.size())
                return fail(err, DecodeErrorKind::SourceDamaged, "stage " + st.name + ": no such variant");
        }
        Sha256 h;
        h.update(key.data(), key.size());
        for (const Tensor& t : st.tensors) {
            TensorOut to;
            if (!decode_tensor(map, t, arrays[static_cast<std::size_t>(t.src)], to, err, st.name)) return false;
            h.update(to.bytes.data(), to.bytes.size());
            so.tensors.push_back(std::move(to));
        }
        so.digest = h.finish();
        out.push_back(std::move(so));
    }
    // Section 7.3: compare every digest before anything is kept.
    for (const StageOut& so : out) {
        // For a group read from a source with variants: the entry of the variant that was found.
        const Digest& want = so.variant >= 0 ? so.stage->variant_digests[static_cast<std::size_t>(so.variant)] : so.stage->digest;
        if (so.digest != want) {
            err.kind = DecodeErrorKind::DigestMismatch;
            err.detail = "stage " + so.stage->name + ": digest does not match the map";
            return false;
        }
    }
    return true;
}

}  // namespace wgc
