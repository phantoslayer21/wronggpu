// SPDX-License-Identifier: MIT
// Decoding the output groups of a version-5 map (spec 57 sections 5.3, 6, 7) and checking their digests (7.3).
// Written for this project.
//
// Error classes (spec 57 section 8): a value that the data of a supported build never has (sites that disagree,
// a forbidden non-finite value, an RN16 overflow, `also` reads that differ, a failed `require` or check, rejected
// attention constants) is ValueRejected ("not supported yet"); a read outside its source, which the map checks
// before any file is read, is checked again here and is MapDamaged ("converter data damaged"); a group digest that
// does not match is DigestMismatch ("Conversion self-check failed"). The log text never holds a decoded value.
#include <cmath>

#include "convert.h"
#include "numeric.h"
#include "ops5.h"

namespace wgc {

bool group_enabled(const Stage& g, const std::vector<int>& variants) {
    for (const auto& rv : g.requires_variant) {
        const std::size_t s = static_cast<std::size_t>(rv.first);
        if (s >= variants.size() || variants[s] != rv.second) return false;
    }
    return true;
}

namespace {

bool fail(DecodeError& err, DecodeErrorKind kind, const std::string& detail) {
    err.kind = kind;
    err.detail = detail;
    return false;
}

void put_le(std::vector<std::uint8_t>& out, std::uint64_t v, std::uint64_t bytes) {
    for (std::uint64_t i = 0; i < bytes; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

// The element view of one found source (57 section 4): a window has exactly one element, the value at its
// field; every other source has element e at byte e * size.
struct SourceView {
    const Source* s = nullptr;
    const std::vector<std::uint8_t>* bytes = nullptr;

    std::uint64_t count() const {
        if (s->locate == Locate::Window) return 1;
        return bytes->size() / s->element_size();
    }
    // Callers check e < count().
    std::uint64_t at(std::uint64_t e) const {
        const std::uint64_t size = s->element_size();
        const std::uint8_t* p = bytes->data() + (s->locate == Locate::Window ? s->field : e * size);
        std::uint64_t v = 0;
        for (std::uint64_t i = 0; i < size; ++i) v |= std::uint64_t(p[i]) << (8 * i);
        return v;
    }
};

class Decoder {
public:
    Decoder(const Map& m, const std::vector<std::vector<std::uint8_t>>& src, const std::vector<int>& variants,
            DecodeError& err, Convert5Info* info)
        : m_(m), src_(src), variants_(variants), err_(err), info_(info) {}

    SourceView view(int si) const {
        SourceView v;
        v.s = &m_.sources[static_cast<std::size_t>(si)];
        v.bytes = &src_[static_cast<std::size_t>(si)];
        return v;
    }

    bool indexed(const Stage& g, const Tensor& t, TensorOut& out) {
        const std::string where = g.name + "/" + t.name;
        const SourceView sv = view(t.src);
        const std::size_t rank = t.axes.size();
        // Per axis: stride * pi(i), or -1 where pi(i) >= limit (57 section 5.3: the element is zero, nothing is read).
        std::vector<std::vector<std::int64_t>> contrib(rank);
        for (std::size_t a = 0; a < rank; ++a) {
            const Axis& ax = t.axes[a];
            contrib[a].resize(static_cast<std::size_t>(ax.length));
            for (std::int64_t i = 0; i < ax.length; ++i) {
                std::int64_t pi = i;
                if (ax.perm >= 0) {
                    const std::vector<std::int64_t>& q = m_.perms[static_cast<std::size_t>(ax.perm)].table;
                    const std::int64_t b = static_cast<std::int64_t>(q.size());
                    pi = b * (i / b) + q[static_cast<std::size_t>(i % b)];
                }
                contrib[a][static_cast<std::size_t>(i)] = (ax.limit >= 0 && pi >= ax.limit) ? -1 : ax.stride * pi;
            }
        }
        std::vector<SourceView> also;
        for (const AlsoRead& r : t.also) also.push_back(view(r.src));
        std::uint64_t cap_bits = 0;
        if (t.op.kind == OpKind::Exp5 && t.op.max_src >= 0) {
            const SourceView cv = view(t.op.max_src);
            cap_bits = cv.at(0);
            if (!std::isfinite(bits_to_float(static_cast<std::uint32_t>(cap_bits))))
                return fail(err_, DecodeErrorKind::ValueRejected, where + ": the cap " +
                            m_.sources[static_cast<std::size_t>(t.op.max_src)].name + " is not finite");
        }
        const std::uint64_t osize = dtype_size(t.dtype);
        out.tensor = &t;
        out.bytes.clear();
        out.bytes.reserve(static_cast<std::size_t>(t.byte_size()));
        const std::uint64_t n_src = sv.count();
        std::vector<std::int64_t> idx(rank, 0);
        for (std::uint64_t k = 0; k < t.count; ++k) {
            std::int64_t e = 0;
            bool zero = false;
            for (std::size_t a = 0; a < rank; ++a) {
                const std::int64_t c = contrib[a][static_cast<std::size_t>(idx[a])];
                if (c < 0) zero = true;
                else e += c;
            }
            if (zero) {
                put_le(out.bytes, 0, osize);  // the zero of the type (e4m3 code 0)
            } else {
                const std::int64_t ep = t.base + e;
                if (ep < 0 || static_cast<std::uint64_t>(ep) >= n_src)
                    return fail(err_, DecodeErrorKind::MapDamaged, where + ": element " + std::to_string(k) +
                                " reads outside source " + sv.s->name);
                const std::uint64_t x = sv.at(static_cast<std::uint64_t>(ep));
                for (std::size_t i = 0; i < also.size(); ++i) {  // `also`: the same elements again, bit-identical
                    const std::int64_t ea = t.also[i].base + e;
                    if (ea < 0 || static_cast<std::uint64_t>(ea) >= also[i].count())
                        return fail(err_, DecodeErrorKind::MapDamaged, where + ": element " + std::to_string(k) +
                                    " of also read " + std::to_string(i) + " reads outside source " + also[i].s->name);
                    if (also[i].at(static_cast<std::uint64_t>(ea)) != x)
                        return fail(err_, DecodeErrorKind::ValueRejected, where + ": the also read from " +
                                    also[i].s->name + " differs (element " + std::to_string(k) + ")");
                }
                if (!element(t, static_cast<std::uint16_t>(x), x, cap_bits, out.bytes, where, k)) return false;
            }
            for (std::size_t a = rank; a-- > 0;) {  // next index, row-major
                if (++idx[a] < t.axes[a].length) break;
                idx[a] = 0;
            }
        }
        return true;
    }

    bool element(const Tensor& t, std::uint16_t h, std::uint64_t x, std::uint64_t cap_bits, std::vector<std::uint8_t>& out,
                 const std::string& where, std::uint64_t k) {
        switch (t.op.kind) {
            case OpKind::E4M3: {
                const int c = e4m3_code(h);
                if (c < 0) return non_finite(where, k);
                out.push_back(static_cast<std::uint8_t>(c));
                return true;
            }
            case OpKind::F16Bits:
                if (!half_is_finite(h)) return non_finite(where, k);
                put_le(out, h, 2);
                return true;
            case OpKind::U16: put_le(out, x, 2); return true;
            case OpKind::U32: case OpKind::U32AsF32: put_le(out, x, 4); return true;
            case OpKind::Exp5: {
                if (!half_is_finite(h)) return non_finite(where, k);
                float y = exp_rn32(half_value(h));
                if (t.op.max_src >= 0) {
                    const float cap = bits_to_float(static_cast<std::uint32_t>(cap_bits));
                    if (y > cap) y = cap;  // also after an overflow to infinity
                }
                if (!std::isfinite(y)) return non_finite(where, k);
                put_le(out, float_to_bits(y), 4);
                return true;
            }
            default: return fail(err_, DecodeErrorKind::MapDamaged, where + ": not an indexed operation");
        }
    }

    bool non_finite(const std::string& where, std::uint64_t k) {
        return fail(err_, DecodeErrorKind::ValueRejected, where + ": element " + std::to_string(k) + " is not finite");
    }

    bool score_bias(const Stage& g, const Tensor& t, TensorOut& out) {
        const std::string where = g.name + "/" + t.name;
        const SourceView w = view(t.src);
        const int H = static_cast<int>(t.shape[0]);
        const std::uint64_t need = static_cast<std::uint64_t>(t.base) + static_cast<std::uint64_t>(H) * 4096;
        if (t.base < 0 || need > w.count())
            return fail(err_, DecodeErrorKind::MapDamaged, where + ": reads outside source " + w.s->name);
        std::vector<std::uint16_t> S(static_cast<std::size_t>(H) * 4096);
        for (std::size_t i = 0; i < S.size(); ++i) S[i] = static_cast<std::uint16_t>(w.at(static_cast<std::uint64_t>(t.base) + i));
        const std::uint32_t a = static_cast<std::uint32_t>(view(t.op.a_src).at(0));
        const std::uint32_t wd = static_cast<std::uint32_t>(view(t.op.w_src).at(0));
        std::vector<std::uint16_t> tb;
        std::string why;
        if (!score_bias_op(S.data(), H, a, wd, t.op.pairs, tb, why))
            return fail(err_, DecodeErrorKind::ValueRejected, where + ": " + why);
        out.tensor = &t;
        out.bytes.clear();
        for (std::uint16_t v : tb) put_le(out.bytes, v, 2);
        return true;
    }

    bool constant(const Stage& g, const Tensor& t, TensorOut& out) {
        const std::string where = g.name + "/" + t.name;
        std::vector<std::uint32_t> words;
        for (std::size_t i = 0; i < t.sites.size(); ++i) {
            const Site& site = t.sites[i];
            const SourceView sv = view(site.src);
            const int v = variants_[static_cast<std::size_t>(site.src)];
            if (site.by_variant && (v < 0 || static_cast<std::size_t>(v) >= site.per_variant.size()))
                return fail(err_, DecodeErrorKind::MapDamaged, where + ": no variant of source " + sv.s->name);
            for (std::uint64_t e : site.selected(v)) {
                if (e >= sv.count())
                    return fail(err_, DecodeErrorKind::MapDamaged, where + ": site " + std::to_string(i) +
                                " reads outside source " + sv.s->name);
                words.push_back(static_cast<std::uint32_t>(sv.at(e)));
            }
        }
        if (words.empty()) return fail(err_, DecodeErrorKind::MapDamaged, where + ": no site element selected");
        std::uint32_t bits = 0;
        std::string why;
        if (!const_op(t.op.kind, t.op.negate, words, bits, why))
            return fail(err_, DecodeErrorKind::ValueRejected, where + ": " + why);
        out.tensor = &t;
        out.bytes.clear();
        put_le(out.bytes, bits, dtype_size(t.dtype));
        return true;
    }

    bool wmax(const Stage& g, const Tensor& t, const std::vector<TensorOut>& done, TensorOut& out) {
        const std::string where = g.name + "/" + t.name;
        std::uint16_t in[4];
        for (int k = 0; k < 4; ++k) {
            const std::size_t ti = static_cast<std::size_t>(t.op.wmax_in[k]);
            if (ti >= done.size() || done[ti].bytes.size() < 2)
                return fail(err_, DecodeErrorKind::MapDamaged, where + ": an input is not an earlier tensor");
            in[k] = static_cast<std::uint16_t>(done[ti].bytes[0] | (done[ti].bytes[1] << 8));
        }
        std::uint16_t w = 0;
        bool overflow = false;
        std::string why;
        if (!attn_wmax_op(in[0], in[1], in[2], in[3], w, overflow, why))
            return fail(err_, DecodeErrorKind::ValueRejected, where + ": the attention constants are rejected: " + why);
        if (info_) {
            info_->have_wmax = true;
            info_->attention_sum_may_overflow = info_->attention_sum_may_overflow || overflow;
        }
        out.tensor = &t;
        out.bytes.clear();
        put_le(out.bytes, w, 2);
        return true;
    }

    bool group(const Stage& g, const Digest& key, StageOut& so) {
        so.stage = &g;
        Sha256 h;
        h.update(key.data(), key.size());
        for (const Tensor& t : g.tensors) {
            TensorOut to;
            bool ok;
            if (t.op.kind == OpKind::ScoreBias) ok = score_bias(g, t, to);
            else if (is_const_op(t.op.kind)) ok = constant(g, t, to);
            else if (t.op.kind == OpKind::AttnWmax) ok = wmax(g, t, so.tensors, to);
            else ok = indexed(g, t, to);
            if (!ok) return false;
            const std::string where = g.name + "/" + t.name;
            if (to.bytes.size() != t.byte_size())
                return fail(err_, DecodeErrorKind::MapDamaged, where + ": wrong output size");
            std::uint64_t bad = 0;
            if (!t.allow_nonfinite && !output_is_finite(t.dtype, to.bytes.data(), t.count, bad))
                return fail(err_, DecodeErrorKind::ValueRejected, where + ": element " + std::to_string(bad) + " is not finite");
            std::string why;
            if (!check_require(t.dtype, to.bytes.data(), t.count, t.require, why))
                return fail(err_, DecodeErrorKind::ValueRejected, where + ": " + why);
            h.update(to.bytes.data(), to.bytes.size());
            so.tensors.push_back(std::move(to));
        }
        for (const Check& c : g.checks) {  // after all tensors of the group (57 section 6.3)
            const TensorOut& ta = so.tensors[static_cast<std::size_t>(c.a)];
            const double a = scalar_value(ta.tensor->dtype, ta.bytes.data());
            double b = c.b_const;
            if (c.b >= 0) {
                const TensorOut& tb = so.tensors[static_cast<std::size_t>(c.b)];
                b = scalar_value(tb.tensor->dtype, tb.bytes.data());
            }
            if (!check_relation(c.rel, a, b))
                return fail(err_, DecodeErrorKind::ValueRejected, g.name + ": check \"" + c.text + "\" fails");
        }
        so.digest = h.finish();
        return true;
    }

private:
    const Map& m_;
    const std::vector<std::vector<std::uint8_t>>& src_;
    const std::vector<int>& variants_;
    DecodeError& err_;
    Convert5Info* info_;
};

}  // namespace

bool convert_all_v5(const Map& map, const std::vector<std::vector<std::uint8_t>>& sources,
                    const std::vector<int>& variants, const Digest& key, std::vector<StageOut>& out, DecodeError& err,
                    Convert5Info* info) {
    out.clear();
    err = DecodeError();
    if (info) *info = Convert5Info();
    if (sources.size() != map.sources.size() || variants.size() != map.sources.size())
        return fail(err, DecodeErrorKind::MapDamaged, "source list does not fit the map");
    for (std::size_t i = 0; i < map.sources.size(); ++i) {  // defensive: what the search delivered
        const Source& s = map.sources[i];
        const std::uint64_t want = s.has_variants()
                                       ? (variants[i] >= 0 ? s.variants[static_cast<std::size_t>(variants[i])].length : 0)
                                       : s.length;
        if (sources[i].size() != want || (s.has_variants() && variants[i] < 0))
            return fail(err, DecodeErrorKind::MapDamaged, "source " + s.name + " has the wrong size");
    }
    Decoder d(map, sources, variants, err, info);
    for (const Stage& g : map.stages) {
        if (!group_enabled(g, variants)) continue;
        StageOut so;
        if (!d.group(g, key, so)) {
            out.clear();
            return false;
        }
        out.push_back(std::move(so));
    }
    for (const StageOut& so : out) {  // 57 section 7.3: every digest before anything is kept
        if (so.digest != so.stage->digest) {
            out.clear();
            return fail(err, DecodeErrorKind::DigestMismatch, "group " + so.stage->name + ": digest does not match the map");
        }
    }
    return true;
}

}  // namespace wgc
