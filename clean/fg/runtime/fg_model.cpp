// SPDX-License-Identifier: GPL-3.0-only
#include "fg_model.h"

#include <cstring>
#include <fstream>

#include "../../converter/src/sha256.h"

namespace wgfg {

namespace {

constexpr size_t kHeader = 88;  // magic, version, stage count, key, map hash, index size
constexpr size_t kTrailer = 32;

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t rd64(const uint8_t* p) { return static_cast<uint64_t>(rd32(p)) | (static_cast<uint64_t>(rd32(p + 4)) << 32); }

// A bounded reader over the index.
struct Cursor {
    const uint8_t* p;
    size_t left;
    bool take(size_t n, const uint8_t** out) {
        if (n > left) return false;
        *out = p;
        p += n;
        left -= n;
        return true;
    }
    bool name(std::string* out) {
        const uint8_t* q = nullptr;
        if (!take(2, &q)) return false;
        const uint16_t n = rd16(q);
        if (!take(n, &q)) return false;
        out->assign(reinterpret_cast<const char*>(q), n);
        return true;
    }
};

FgRefusal fail(FgModelReport* r, FgRefusal why, int stage, const std::string& detail) {
    r->refusal = why;
    r->stage = stage;
    r->detail = detail;
    r->stages.clear();
    return why;
}

bool parse_tensor(Cursor& c, const char* want, FgModelTensor* t, std::string* got_name, uint8_t* dtype) {
    if (!c.name(got_name) || *got_name != want) return false;
    const uint8_t* q = nullptr;
    if (!c.take(2, &q)) return false;
    *dtype = q[0];
    t->rank = q[1];
    if (t->rank == 0 || t->rank > 4) return false;
    if (!c.take(4u * t->rank, &q)) return false;
    for (uint32_t i = 0; i < t->rank; ++i) t->shape[i] = rd32(q + 4 * i);
    if (!c.take(16, &q)) return false;
    t->offset = rd64(q);
    t->bytes = rd64(q + 8);
    return true;
}

bool hex_equals(const wgc::Digest& d, const char* hex) {
    wgc::Digest want{};
    return hex && wgc::parse_digest_hex(hex, want) && want == d;
}

}  // namespace

FgModelExpectation FgShippedModelExpectation() { return {kFgMapTable, kFgMapStages, kFgMapSha256}; }

float FgHalfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0x1F) {
        bits = sign | 0x7F800000u | (m << 13);
    } else if (e) {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    } else if (!m) {
        bits = sign;
    } else {  // subnormal: m * 2^-24
        uint32_t mm = m, ee = 113;
        while (!(mm & 0x400u)) {
            mm <<= 1;
            --ee;
        }
        bits = sign | (ee << 23) | ((mm & 0x3FFu) << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool FgFloatToHalfExact(float f, uint16_t* out) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t e8 = (bits >> 23) & 0xFFu, m = bits & 0x7FFFFFu;
    if (e8 == 0xFF) {
        if (m) return false;  // NaN
        *out = static_cast<uint16_t>(sign | 0x7C00u);
        return true;
    }
    if (e8 == 0) {
        if (m) return false;  // a float32 subnormal is far below float16's range
        *out = sign;
        return true;
    }
    const int e = static_cast<int>(e8) - 127;
    if (e > 15 || e < -24) return false;
    const uint32_t sig = m | 0x800000u;  // 24-bit significand
    if (e >= -14) {
        if (m & 0x1FFFu) return false;
        *out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e + 15) << 10) | (m >> 13));
        return true;
    }
    const int shift = -1 - e;  // 14..23: the significand must be a multiple of 2^shift
    if (sig & ((1u << shift) - 1u)) return false;
    *out = static_cast<uint16_t>(sign | (sig >> shift));
    return true;
}

FgRefusal FgValidateModelBytes(const uint8_t* data, size_t size, const FgModelExpectation& expect,
                               FgModelReport* report) {
    FgModelReport scratch;
    FgModelReport* r = report ? report : &scratch;
    if (!data || size == 0) return fail(r, FgRefusal::ModelMissing, -1, "no bytes");
    if (!expect.stages || expect.count == 0) return fail(r, FgRefusal::ModelMissing, -1, "no expectation");
    if (size < kHeader + kTrailer || std::memcmp(data, "WGCMODEL", 8) != 0)
        return fail(r, FgRefusal::ModelContainer, -1, "not a WrongGPU model file");
    wgc::Digest trailer{};
    std::memcpy(trailer.data(), data + size - kTrailer, kTrailer);
    if (wgc::Sha256::hash(data, size - kTrailer) != trailer)
        return fail(r, FgRefusal::ModelChecksum, -1, "file checksum does not match");
    if (rd32(data + 8) != 1) return fail(r, FgRefusal::ModelContainer, -1, "container version (preset FG uses 1)");
    wgc::Digest key{}, map_hash{};
    std::memcpy(key.data(), data + 16, 32);
    std::memcpy(map_hash.data(), data + 48, 32);
    if (!hex_equals(map_hash, expect.map_sha256))
        return fail(r, FgRefusal::ModelMapHash, -1, "written with another map");
    if (rd32(data + 12) != expect.count)
        return fail(r, FgRefusal::ModelStageCount, -1,
                    "stages " + std::to_string(rd32(data + 12)) + ", the map has " + std::to_string(expect.count));
    const uint64_t index_size = rd64(data + 80);
    const size_t data_begin = kHeader, data_end = size - kTrailer;
    if (index_size > data_end - data_begin) return fail(r, FgRefusal::ModelContainer, -1, "index size");
    Cursor c{data + kHeader, static_cast<size_t>(index_size)};
    const uint64_t tensors_begin = kHeader + index_size;
    std::vector<FgModelStage> stages;
    stages.reserve(expect.count);
    for (uint32_t si = 0; si < expect.count; ++si) {
        const FgMapStage& want = expect.stages[si];
        const int s = static_cast<int>(si);
        FgModelStage st;
        if (!c.name(&st.name)) return fail(r, FgRefusal::ModelContainer, s, "index ends early");
        if (st.name != want.name)
            return fail(r, FgRefusal::ModelStageName, s, "stage " + st.name + " where the map has " + want.name);
        const uint8_t* q = nullptr;
        if (!c.take(32 + 4, &q)) return fail(r, FgRefusal::ModelContainer, s, "index ends early");
        wgc::Digest stored{};
        std::memcpy(stored.data(), q, 32);
        if (rd32(q + 32) != 2) return fail(r, FgRefusal::ModelTensorShape, s, "tensor count (W and b)");
        std::string tn;
        uint8_t wt = 0, bt = 0;
        if (!parse_tensor(c, "W", &st.w, &tn, &wt) || !parse_tensor(c, "b", &st.b, &tn, &bt))
            return fail(r, FgRefusal::ModelTensorShape, s, "tensor entry " + tn);
        if (wt != 0 || bt != 0) return fail(r, FgRefusal::ModelTensorType, s, "preset FG writes float32");
        bool shape_ok = st.w.rank == want.w_rank && st.b.rank == 1 && st.b.shape[0] == want.b_len;
        uint64_t welems = 1;
        for (uint32_t i = 0; shape_ok && i < st.w.rank; ++i) {
            shape_ok = st.w.shape[i] == want.w_shape[i];
            welems *= st.w.shape[i];
        }
        if (!shape_ok) return fail(r, FgRefusal::ModelTensorShape, s, "W / b shape differs from the map");
        for (const FgModelTensor* t : {&st.w, &st.b}) {
            const uint64_t elems = t == &st.w ? welems : st.b.shape[0];
            if (t->bytes != 4 * elems || t->offset % 64 || t->offset < tensors_begin || t->offset > data_end ||
                t->bytes > data_end - t->offset)
                return fail(r, FgRefusal::ModelContainer, s, "tensor data range");
        }
        wgc::Sha256 h;
        h.update(key.data(), key.size());
        h.update(data + st.w.offset, static_cast<size_t>(st.w.bytes));
        h.update(data + st.b.offset, static_cast<size_t>(st.b.bytes));
        if (h.finish() != stored) return fail(r, FgRefusal::ModelStageDigest, s, "stage data does not match its digest");
        if (!hex_equals(stored, want.digest))
            return fail(r, FgRefusal::ModelStageDigest, s, "stage digest differs from the map's");
        for (const FgModelTensor* t : {&st.w, &st.b}) {
            for (uint64_t i = 0; i < t->bytes; i += 4) {
                const uint32_t bits = rd32(data + t->offset + i);
                float f;
                std::memcpy(&f, &bits, 4);
                uint16_t hbits = 0;
                if (!FgFloatToHalfExact(f, &hbits))
                    return fail(r, FgRefusal::ModelValue, s, "a value is not exact in float16");
            }
        }
        stages.push_back(st);
    }
    if (c.left != 0) return fail(r, FgRefusal::ModelContainer, -1, "index has bytes after the last stage");
    r->refusal = FgRefusal::None;
    r->stage = -1;
    r->detail.clear();
    r->stages = std::move(stages);
    return FgRefusal::None;
}

FgRefusal FgValidateModelFile(const std::string& path, const FgModelExpectation& expect, FgModelReport* report,
                              std::vector<uint8_t>* bytes_out) {
    FgModelReport scratch;
    FgModelReport* r = report ? report : &scratch;
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(r, FgRefusal::ModelMissing, -1, "cannot open the model file");
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const FgRefusal v = FgValidateModelBytes(bytes.data(), bytes.size(), expect, r);
    if (v == FgRefusal::None && bytes_out) *bytes_out = std::move(bytes);
    return v;
}

}  // namespace wgfg
