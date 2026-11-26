// SPDX-License-Identifier: MIT
// CPU tests of the FG runtime groundwork (clean/fg/runtime): grid table, contract, model check, net1 plan / recorder /
// packing against the generated parity tables, frame ingest, history, and the create-time worker's failure paths.
// Synthetic values only: constructed float16 patterns, constructed model files, fake resources and a fake recorder.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "../../../converter/src/sha256.h"
#include "fg_contract.h"
#include "fg_frame_dump.h"
#include "fg_grid.h"
#include "fg_history.h"
#include "fg_model.h"
#include "fg_net1_plan.h"
#include "fg_worker.h"
#include "net1_plan_golden.h"

using namespace wgfg;

namespace {

int g_checks = 0, g_failed = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(c)) {                                                           \
            ++g_failed;                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);          \
        }                                                                     \
    } while (0)

struct TestCase {
    const char* name;
    void (*fn)();
};
std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};
#define TEST(name)                                     \
    void name();                                       \
    const Registrar reg_##name(#name, name);           \
    void name()

uint16_t synth_bits(uint64_t i) {
    return static_cast<uint16_t>(((i * 40503u + 12345u) & 0x3FFFu) | ((i & 1u) << 15));
}

std::string sha_hex(const void* p, size_t n) { return wgc::to_hex(wgc::Sha256::hash(p, n)); }

uint32_t float_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// ---- grid table ------------------------------------------------------------------------------------------------------

TEST(grid_table_matches_size_table) {
    struct Row {
        uint32_t wo, ho, d, wd, hd, w0, h0, pl, pt, k, w2, h2;
    };
    const Row rows[] = {
        {1280, 720, 2, 640, 360, 320, 192, 0, 6, 2, 640, 368},
        {1920, 1080, 2, 960, 540, 480, 288, 0, 9, 4, 480, 272},
        {2560, 1440, 2, 1280, 720, 640, 384, 0, 12, 4, 640, 368},
        {3440, 1440, 2, 1720, 720, 864, 384, 2, 12, 4, 864, 368},
        {3840, 2160, 4, 960, 540, 480, 288, 0, 9, 8, 480, 272},
        {7680, 4320, 4, 1920, 1080, 960, 544, 0, 2, 16, 480, 272},
    };
    for (const Row& r : rows) {
        FgGrids g{};
        CHECK(FgGridsFor(r.wo, r.ho, &g));
        CHECK(g.net1.d == r.d && g.net1.wd == r.wd && g.net1.hd == r.hd);
        CHECK(g.net1.w0 == r.w0 && g.net1.h0 == r.h0);
        CHECK(g.net1.pad_left == r.pl && g.net1.pad_top == r.pt);
        CHECK(g.net1.pad_left + g.net1.pad_right + g.net1.wd / 2 == g.net1.w0);
        CHECK(g.net1.pad_top + g.net1.pad_bottom + g.net1.hd / 2 == g.net1.h0);
        CHECK(g.net1.w0 % 32 == 0 && g.net1.h0 % 32 == 0);
        CHECK(g.net2.k == r.k && g.net2.w2 == r.w2 && g.net2.h2 == r.h2);
        CHECK(g.net2.pad_right == 16 * r.k * (r.w2 / 16) - r.wo);
        CHECK(g.net2.pad_bottom == 16 * r.k * (r.h2 / 16) - r.ho);
        CHECK(r.ho / g.net2.k <= 480 && (g.net2.k == 1 || r.ho / (g.net2.k / 2) > 480));
        // net1 at this grid plans, and its three regions fit 32-bit arena offsets together
        FgNet1Plan p;
        CHECK(FgBuildNet1Plan(g.net1.h0, g.net1.w0, &p) == FgRefusal::None);
        CHECK(p.region_bytes[0] + p.region_bytes[1] + p.region_bytes[2] < (1ull << 32));
    }
    static_assert(FgNet1GridFor(1280, 720).h0 == 192, "constexpr grid");
    FgGrids g{};
    CHECK(FgGridsFor(4, 4, &g) && g.net1.w0 == 32 && g.net2.w2 == 16);
    CHECK(FgGridsFor(8192, 7680, &g));
    CHECK(!FgGridsFor(3, 720, &g) && !FgGridsFor(1280, 3, &g));
    CHECK(!FgGridsFor(8193, 720, &g) && !FgGridsFor(1280, 7681, &g));
    // the d switch sits exactly at 1440
    CHECK(FgNet1GridFor(2560, 1440).d == 2 && FgNet1GridFor(2560, 1441).d == 4);
}

// ---- contract --------------------------------------------------------------------------------------------------------

int g_device_tag = 0, g_queue_tag = 0;

FgCreateDesc good_create() {
    FgCreateDesc d;
    d.device = &g_device_tag;
    d.queue = &g_queue_tag;
    d.adapter_luid = {0x1234u, 7};
    d.swapchain_id = 0xA1;
    d.window_id = 0xB2;
    d.queue_epoch = 3;
    d.swapchain_epoch = 9;
    d.viewport_id = 1;
    d.output_width = 1920;
    d.output_height = 1080;
    d.output_format = 28;  // R8G8B8A8_UNORM
    d.render_min_width = 1280;
    d.render_min_height = 720;
    d.render_max_width = 1920;
    d.render_max_height = 1080;
    d.dynamic_resolution = 1;
    d.preset = FgPreset::A;
    d.max_generated = 5;
    d.optional[kFgOptHudless] = FgOptionalDecl::May;
    d.optional[kFgOptUiAlpha] = FgOptionalDecl::Always;
    d.optional[kFgOptDistortion] = FgOptionalDecl::Never;
    d.optional[kFgOptNoWarp] = FgOptionalDecl::May;
    return d;
}

TEST(create_validation_refuses_each_field) {
    CHECK(FgValidateCreate(good_create()) == FgRefusal::None);
    struct M {
        void (*mut)(FgCreateDesc&);
        FgRefusal want;
    };
    const M ms[] = {
        {[](FgCreateDesc& d) { d.struct_size -= 4; }, FgRefusal::StructSize},
        {[](FgCreateDesc& d) { d.abi_version = 2; }, FgRefusal::AbiVersion},
        {[](FgCreateDesc& d) { d.device = nullptr; }, FgRefusal::NullDevice},
        {[](FgCreateDesc& d) { d.queue = nullptr; }, FgRefusal::NullQueue},
        {[](FgCreateDesc& d) { d.adapter_luid = {}; }, FgRefusal::BadLuid},
        {[](FgCreateDesc& d) { d.output_width = 9000; }, FgRefusal::BadOutputSize},
        {[](FgCreateDesc& d) { d.output_height = 2; }, FgRefusal::BadOutputSize},
        {[](FgCreateDesc& d) { d.output_format = 0; }, FgRefusal::BadFormat},
        {[](FgCreateDesc& d) { d.render_min_width = 0; }, FgRefusal::BadRenderBounds},
        {[](FgCreateDesc& d) { d.render_min_width = 1921; }, FgRefusal::BadRenderBounds},
        {[](FgCreateDesc& d) { d.dynamic_resolution = 0; }, FgRefusal::BadRenderBounds},
        {[](FgCreateDesc& d) { d.preset = static_cast<FgPreset>(3); }, FgRefusal::BadPreset},
        {[](FgCreateDesc& d) { d.max_generated = 0; }, FgRefusal::BadGeneratedCount},
        {[](FgCreateDesc& d) { d.max_generated = 6; }, FgRefusal::BadGeneratedCount},
        {[](FgCreateDesc& d) { d.optional[2] = static_cast<FgOptionalDecl>(3); }, FgRefusal::BadOptionalDecl},
    };
    for (const M& m : ms) {
        FgCreateDesc d = good_create();
        m.mut(d);
        CHECK(FgValidateCreate(d) == m.want);
    }
    for (uint32_t n = 1; n <= 5; ++n) {
        FgCreateDesc d = good_create();
        d.max_generated = n;
        CHECK(FgValidateCreate(d) == FgRefusal::None);
    }
}

TEST(packet_shape_presets_and_index_time) {
    const FgCreateDesc s = good_create();
    FgFramePacket p;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::None);
    p.struct_size = 8;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::StructSize);
    p = FgFramePacket{};
    p.abi_version = 0;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::AbiVersion);
    p = FgFramePacket{};
    p.generated_count = 6;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::BadGeneratedCount);
    p.generated_count = 0;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::BadGeneratedCount);
    p.generated_count = 3;
    p.generated_index = 4;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::BadGeneratedIndex);
    p.generated_index = 3;
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::None);
    p.preset = static_cast<FgPreset>(7);
    CHECK(FgValidatePacketShape(s, p) == FgRefusal::BadPreset);

    FgPresetGraph g{};
    CHECK(FgPresetGraphOf(FgPreset::A, &g) && g.full_blend && !g.ui_recompose && !g.lite_blend);
    CHECK(FgPresetGraphOf(FgPreset::B, &g) && g.full_blend && g.ui_recompose && !g.lite_blend);
    CHECK(FgPresetGraphOf(FgPreset::C, &g) && !g.full_blend && !g.ui_recompose && g.lite_blend);
    CHECK(!FgPresetGraphOf(static_cast<FgPreset>(3), &g));

    // t = RN32(i / (N + 1)): exact bit patterns
    CHECK(float_bits(FgIndexTime(1, 1)) == 0x3F000000u);  // 0.5
    CHECK(float_bits(FgIndexTime(1, 2)) == 0x3EAAAAABu);  // 1/3
    CHECK(float_bits(FgIndexTime(2, 2)) == 0x3F2AAAABu);  // 2/3
    CHECK(float_bits(FgIndexTime(1, 4)) == 0x3E4CCCCDu);  // 0.2
    CHECK(float_bits(FgIndexTime(3, 5)) == 0x3F000000u);  // 0.5
    CHECK(float_bits(FgIndexTime(5, 5)) == 0x3F555555u);  // 5/6
    CHECK(FgMultiFrame(2) && !FgMultiFrame(1) && FgLastIndex(3, 3) && !FgLastIndex(2, 3));
}

TEST(route_refuses_in_order_and_advertises_nothing_unqualified) {
    FgRouteFacts f;
    CHECK(FgCheckRoute(f) == FgRefusal::InstallOff);
    f.install_on = true;
    CHECK(FgCheckRoute(f) == FgRefusal::UnsupportedAdapter);
    f.adapter_supported = true;
    CHECK(FgCheckRoute(f) == FgRefusal::ModelNotValidated);
    f.model_validated = true;
    CHECK(FgCheckRoute(f) == FgRefusal::PresenterNotQualified);
    f.presenter_qualified = true;
    CHECK(FgCheckRoute(f) == (kFgModesQualified ? FgRefusal::None : FgRefusal::RouteNotQualified));
    CHECK(FgAdvertisedMaxGenerated(f) == (kFgModesQualified ? kFgMaxGenerated : 0u));
    static_assert(kFgModesQualified == (WG_FG_MODES_QUALIFIED != 0), "the contract's gate is fg_limits.h's");
    f.modes_qualified = false;  // the closed gate: no FG mode is advertised or taken
    CHECK(FgCheckRoute(f) == FgRefusal::RouteNotQualified && FgAdvertisedMaxGenerated(f) == 0u);
    f.modes_qualified = true;  // a qualified build: all facts true
    CHECK(FgCheckRoute(f) == FgRefusal::None && FgAdvertisedMaxGenerated(f) == 5);
    // every refusal has a name
    for (uint32_t v = 0; v < 100; ++v) {
        const char* n = FgRefusalName(static_cast<FgRefusal>(v));
        (void)n;
    }
    const FgRefusal all[] = {FgRefusal::StructSize, FgRefusal::UnknownTag, FgRefusal::HistoryBankBusy,
                             FgRefusal::ModelStageDigest, FgRefusal::Net1RegionAlign, FgRefusal::WorkerAbandoned};
    for (FgRefusal r : all) CHECK(std::string(FgRefusalName(r)) != "unknown");
}

// ---- float16 helpers -------------------------------------------------------------------------------------------------

TEST(half_conversions) {
    uint16_t h = 0;
    for (uint32_t b = 0; b < 65536; ++b) {
        const uint16_t x = static_cast<uint16_t>(b);
        const bool nan = (x & 0x7C00) == 0x7C00 && (x & 0x3FF);
        const float f = FgHalfToFloat(x);
        if (nan) {
            CHECK(f != f);
            CHECK(!FgFloatToHalfExact(f, &h));
        } else if (!(FgFloatToHalfExact(f, &h) && h == x)) {
            CHECK(false);
            break;
        }
    }
    CHECK(FgFloatToHalfExact(65504.0f, &h) && h == 0x7BFF);
    CHECK(!FgFloatToHalfExact(65520.0f, &h));
    CHECK(FgFloatToHalfExact(5.9604644775390625e-8f, &h) && h == 0x0001);  // 2^-24
    CHECK(!FgFloatToHalfExact(8.940696716308594e-8f, &h));                 // 1.5 * 2^-24
    CHECK(!FgFloatToHalfExact(2.98023223876953125e-8f, &h));               // 2^-25
    CHECK(FgFloatToHalfExact(6.103515625e-5f, &h) && h == 0x0400);         // 2^-14
    CHECK(!FgFloatToHalfExact(0.1f, &h));
    CHECK(FgFloatToHalfExact(-0.0f, &h) && h == 0x8000);
}

// ---- model check -----------------------------------------------------------------------------------------------------

struct ModelOpts {
    uint32_t version = 1;
    bool shipped_map_hash = false;
    bool drop_last_stage = false;
    int swap_names = -1;       // swap the names of this stage and the next in the index
    int bad_shape = -1;
    int bad_dtype = -1;
    int bad_stored_digest = -1;
    int bad_value = -1;        // a value not exact in float16 (digest of the stored data stays right)
    bool flip_payload = false; // a byte after the checksum was computed
};

struct SynthModel {
    std::vector<uint8_t> bytes;
    std::vector<FgMapStage> stages;
    std::vector<std::string> digests;
    std::string map_hex;
    FgModelExpectation expect() const {
        return {stages.data(), static_cast<uint32_t>(stages.size()), map_hex.c_str()};
    }
};

void put16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x));
    v.push_back(static_cast<uint8_t>(x >> 8));
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
void put64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}

// natural W of stage s: net1 stages hold the canonical synth_bits pattern (so packing can be checked against the
// generated table); net2 stages a shifted pattern
std::vector<float> stage_weights(uint32_t s, const FgMapStage& st) {
    const uint32_t out = st.w_shape[0], cin = st.w_shape[1], k = st.w_rank == 4 ? 3 : 1;
    std::vector<float> w(static_cast<size_t>(out) * cin * k * k);
    for (uint32_t o = 0; o < out; ++o)
        for (uint32_t i = 0; i < cin; ++i)
            for (uint32_t ky = 0; ky < k; ++ky)
                for (uint32_t kx = 0; kx < k; ++kx) {
                    const uint64_t natural = ((static_cast<uint64_t>(o) * cin + i) * k + ky) * k + kx;
                    const uint64_t canon = ((static_cast<uint64_t>(o) * k + ky) * k + kx) * cin + i;
                    w[static_cast<size_t>(natural)] = FgHalfToFloat(synth_bits(s < 17 ? canon : natural + 7919u * s));
                }
    return w;
}

SynthModel build_model(const ModelOpts& o = ModelOpts{}) {
    SynthModel m;
    const FgModelExpectation shipped = FgShippedModelExpectation();
    const uint32_t nst = shipped.count - (o.drop_last_stage ? 1 : 0);
    std::vector<std::vector<float>> W(shipped.count), Bv(shipped.count);
    for (uint32_t s = 0; s < shipped.count; ++s) {
        W[s] = stage_weights(s, shipped.stages[s]);
        Bv[s].resize(shipped.stages[s].b_len);
        for (uint32_t j = 0; j < shipped.stages[s].b_len; ++j) Bv[s][j] = FgHalfToFloat(synth_bits(1000003u + j + 131u * s));
        if (static_cast<int>(s) == o.bad_value) W[s][0] = 0.1f;
    }
    const char* key_label = "synthetic key";
    const char* map_label = "synthetic map";
    const wgc::Digest key = wgc::Sha256::hash(key_label, std::strlen(key_label));
    wgc::Digest map_hash = wgc::Sha256::hash(map_label, std::strlen(map_label));
    if (o.shipped_map_hash) wgc::parse_digest_hex(kFgMapSha256, map_hash);
    m.map_hex = wgc::to_hex(map_hash);
    // index size (fixed-width fields only)
    uint64_t index = 0;
    for (uint32_t s = 0; s < nst; ++s) {
        const uint32_t rank = shipped.stages[s].w_rank;
        index += 2 + std::strlen(shipped.stages[s].name) + 32 + 4;
        index += (2 + 1) + 2 + 4 * rank + 16;  // "W"
        index += (2 + 1) + 2 + 4 + 16;         // "b"
    }
    uint64_t pos = (88 + index + 63) / 64 * 64;
    std::vector<uint64_t> woff(nst), boff(nst);
    for (uint32_t s = 0; s < nst; ++s) {
        woff[s] = pos;
        pos = (pos + 4 * W[s].size() + 63) / 64 * 64;
        boff[s] = pos;
        pos = (pos + 4 * Bv[s].size() + 63) / 64 * 64;
    }
    std::vector<uint8_t> v;
    v.insert(v.end(), {'W', 'G', 'C', 'M', 'O', 'D', 'E', 'L'});
    put32(v, o.version);
    put32(v, nst);
    v.insert(v.end(), key.begin(), key.end());
    v.insert(v.end(), map_hash.begin(), map_hash.end());
    put64(v, index);
    m.digests.resize(shipped.count);
    m.stages.assign(shipped.stages, shipped.stages + shipped.count);
    for (uint32_t s = 0; s < nst; ++s) {
        const FgMapStage& st = shipped.stages[s];
        wgc::Sha256 h;
        h.update(key.data(), key.size());
        h.update(W[s].data(), 4 * W[s].size());
        h.update(Bv[s].data(), 4 * Bv[s].size());
        wgc::Digest d = h.finish();
        m.digests[s] = wgc::to_hex(d);
        if (static_cast<int>(s) == o.bad_stored_digest) d[5] ^= 0x40;
        int name_from = static_cast<int>(s);
        if (o.swap_names >= 0 && static_cast<int>(s) == o.swap_names) name_from = o.swap_names + 1;
        if (o.swap_names >= 0 && static_cast<int>(s) == o.swap_names + 1) name_from = o.swap_names;
        const char* nm = shipped.stages[name_from].name;
        put16(v, static_cast<uint16_t>(std::strlen(nm)));
        v.insert(v.end(), nm, nm + std::strlen(nm));
        v.insert(v.end(), d.begin(), d.end());
        put32(v, 2);
        put16(v, 1);
        v.push_back('W');
        v.push_back(static_cast<uint8_t>(static_cast<int>(s) == o.bad_dtype ? 1 : 0));
        v.push_back(static_cast<uint8_t>(st.w_rank));
        for (uint32_t i = 0; i < st.w_rank; ++i)
            put32(v, st.w_shape[i] + (static_cast<int>(s) == o.bad_shape && i == 0 ? 16u : 0u));
        put64(v, woff[s]);
        put64(v, 4 * W[s].size());
        put16(v, 1);
        v.push_back('b');
        v.push_back(0);
        v.push_back(1);
        put32(v, st.b_len);
        put64(v, boff[s]);
        put64(v, 4 * Bv[s].size());
    }
    v.resize(static_cast<size_t>(pos), 0);
    for (uint32_t s = 0; s < nst; ++s) {
        std::memcpy(v.data() + woff[s], W[s].data(), 4 * W[s].size());
        std::memcpy(v.data() + boff[s], Bv[s].data(), 4 * Bv[s].size());
    }
    const wgc::Digest all = wgc::Sha256::hash(v.data(), v.size());
    v.insert(v.end(), all.begin(), all.end());
    if (o.flip_payload) v[static_cast<size_t>(woff[0]) + 3] ^= 0x01;
    for (uint32_t s = 0; s < shipped.count; ++s) m.stages[s].digest = m.digests[s].c_str();
    m.bytes = std::move(v);
    return m;
}

TEST(shipped_map_table_structure) {
    const FgModelExpectation e = FgShippedModelExpectation();
    CHECK(e.count == 42 && kFgMapStages == 42);
    CHECK(std::strlen(e.map_sha256) == 64);
    const FgNet1Layer* l1 = FgNet1Layers();
    for (uint32_t s = 0; s < 42; ++s) {
        const FgMapStage& st = e.stages[s];
        char want[16];
        std::snprintf(want, sizeof want, "dl%d.l%02u", s < 17 ? 1 : 2, s < 17 ? s + 1 : s - 16);
        CHECK(std::string(st.name) == want);
        CHECK(std::strlen(st.digest) == 64);
        CHECK(st.b_len == st.w_shape[0]);
        if (s < 17) {
            CHECK(st.w_shape[0] == l1[s].cout && st.w_shape[1] == l1[s].cin);
            CHECK(st.w_rank == (l1[s].k == 3 ? 4u : 2u));
            CHECK(st.w_source == st.b_source);  // one blob per net1 layer
            // blob: weights, a zero area of the same size, bias x 16
            CHECK(st.w_source_bytes == 2u * (2u * l1[s].cout * l1[s].k * l1[s].k * l1[s].cin + 16u * l1[s].cout));
        } else {
            CHECK(st.w_rank == 4 && st.w_shape[2] == 3 && st.w_shape[3] == 3);
            CHECK(st.w_source != st.b_source && st.b_source_bytes == 2u * st.b_len);
        }
    }
}

TEST(model_check_accepts_a_constructed_model) {
    const SynthModel m = build_model();
    FgModelReport r;
    CHECK(FgValidateModelBytes(m.bytes.data(), m.bytes.size(), m.expect(), &r) == FgRefusal::None);
    CHECK(r.stages.size() == 42 && r.stages[0].name == "dl1.l01" && r.stages[41].name == "dl2.l25");
    // the shipped expectation refuses it: a constructed model was not written with the shipped map
    CHECK(FgValidateModelBytes(m.bytes.data(), m.bytes.size(), FgShippedModelExpectation(), &r) ==
          FgRefusal::ModelMapHash);
    ModelOpts o;
    o.shipped_map_hash = true;  // the right map hash, but not the map's content: the first stage digest differs
    const SynthModel m2 = build_model(o);
    CHECK(FgValidateModelBytes(m2.bytes.data(), m2.bytes.size(), FgShippedModelExpectation(), &r) ==
          FgRefusal::ModelStageDigest);
    CHECK(r.stage == 0 && r.stages.empty());
}

TEST(model_check_refuses_each_defect) {
    struct C {
        ModelOpts o;
        FgRefusal want;
        int stage;
    };
    std::vector<C> cs;
    auto add = [&cs](void (*f)(ModelOpts&), FgRefusal w, int st) {
        ModelOpts o;
        f(o);
        cs.push_back({o, w, st});
    };
    add([](ModelOpts& o) { o.version = 2; }, FgRefusal::ModelContainer, -1);
    add([](ModelOpts& o) { o.drop_last_stage = true; }, FgRefusal::ModelStageCount, -1);
    add([](ModelOpts& o) { o.swap_names = 0; }, FgRefusal::ModelStageName, 0);
    add([](ModelOpts& o) { o.swap_names = 20; }, FgRefusal::ModelStageName, 20);
    add([](ModelOpts& o) { o.bad_shape = 5; }, FgRefusal::ModelTensorShape, 5);
    add([](ModelOpts& o) { o.bad_dtype = 3; }, FgRefusal::ModelTensorType, 3);
    add([](ModelOpts& o) { o.bad_stored_digest = 7; }, FgRefusal::ModelStageDigest, 7);
    add([](ModelOpts& o) { o.bad_value = 11; }, FgRefusal::ModelValue, 11);
    add([](ModelOpts& o) { o.bad_value = 30; }, FgRefusal::ModelValue, 30);
    add([](ModelOpts& o) { o.flip_payload = true; }, FgRefusal::ModelChecksum, -1);
    for (const C& c : cs) {
        const SynthModel m = build_model(c.o);
        FgModelReport r;
        const FgRefusal got = FgValidateModelBytes(m.bytes.data(), m.bytes.size(), m.expect(), &r);
        CHECK(got == c.want);
        CHECK(r.stage == c.stage);
        CHECK(r.stages.empty() && !r.detail.empty());
        if (got != c.want) std::printf("   model case: want %s got %s (%s)\n", FgRefusalName(c.want), FgRefusalName(got), r.detail.c_str());
    }
    // the expectation's own digest differs from the (consistent) file
    SynthModel m = build_model();
    m.digests[9][0] = m.digests[9][0] == '0' ? '1' : '0';
    m.stages[9].digest = m.digests[9].c_str();
    FgModelReport r;
    CHECK(FgValidateModelBytes(m.bytes.data(), m.bytes.size(), m.expect(), &r) == FgRefusal::ModelStageDigest);
    CHECK(r.stage == 9);
    // container damage
    const SynthModel good = build_model();
    std::vector<uint8_t> b = good.bytes;
    b[0] = 'X';
    CHECK(FgValidateModelBytes(b.data(), b.size(), good.expect(), &r) == FgRefusal::ModelContainer);
    CHECK(FgValidateModelBytes(good.bytes.data(), 100, good.expect(), &r) == FgRefusal::ModelContainer);
    CHECK(FgValidateModelBytes(good.bytes.data(), good.bytes.size() - 1, good.expect(), &r) == FgRefusal::ModelChecksum);
    CHECK(FgValidateModelBytes(nullptr, 0, good.expect(), &r) == FgRefusal::ModelMissing);
    CHECK(FgValidateModelFile("no-such-dir/none.wgcm", good.expect(), &r, nullptr) == FgRefusal::ModelMissing);
}

TEST(net1_packing_from_model_matches_generated_fragments) {
    const SynthModel m = build_model();
    FgModelReport r;
    CHECK(FgValidateModelBytes(m.bytes.data(), m.bytes.size(), m.expect(), &r) == FgRefusal::None);
    const FgNet1Layer* layers = FgNet1Layers();
    for (uint32_t s = 0; s < 17 && r.stages.size() == 42; ++s) {
        const wgfg_golden::Pack& g = wgfg_golden::kPacks[s];
        CHECK(std::string(g.layer) == layers[s].name);
        // directly from the canonical pattern
        const uint64_t n = static_cast<uint64_t>(layers[s].cout) * layers[s].k * layers[s].k * layers[s].cin;
        std::vector<uint16_t> canon(static_cast<size_t>(n));
        for (uint64_t i = 0; i < n; ++i) canon[static_cast<size_t>(i)] = synth_bits(i);
        std::vector<uint8_t> frag;
        CHECK(FgSwizzleNet1(canon.data(), layers[s].cout, layers[s].k, layers[s].cin, layers[s].order, &frag));
        CHECK(frag.size() == g.bytes && sha_hex(frag.data(), frag.size()) == g.sha256);
        // through the model file's natural layout
        std::vector<uint8_t> frag2, bias;
        CHECK(FgPackNet1Layer(m.bytes.data(), m.bytes.size(), r.stages[s], layers[s], &frag2, &bias));
        CHECK(frag2 == frag && bias.size() == 2u * layers[s].cout);
    }
    // a mutant fragment order is caught by the table
    std::vector<uint16_t> canon(32 * 9 * 48);
    for (size_t i = 0; i < canon.size(); ++i) canon[i] = synth_bits(i);
    std::vector<uint8_t> frag;
    CHECK(FgSwizzleNet1(canon.data(), 32, 3, 48, FgChunkOrder::KxBlockKy, &frag));
    CHECK(sha_hex(frag.data(), frag.size()) != wgfg_golden::kPacks[0].sha256);
    // shapes the fragments cannot hold
    CHECK(!FgSwizzleNet1(canon.data(), 32, 3, 40, FgChunkOrder::BlockKxKy, &frag));
    CHECK(!FgSwizzleNet1(canon.data(), 24, 3, 48, FgChunkOrder::BlockKxKy, &frag));
    CHECK(!FgSwizzleNet1(canon.data(), 32, 1, 48, FgChunkOrder::BlockKxKy, &frag));
    CHECK(!FgSwizzleNet1(canon.data(), 32, 3, 48, FgChunkOrder::Ascending, &frag));
    // a stage of another layer does not pack
    std::vector<uint8_t> f, b;
    CHECK(!FgPackNet1Layer(m.bytes.data(), m.bytes.size(), r.stages[1], layers[0], &f, &b));
}

// ---- net1 plan -------------------------------------------------------------------------------------------------------

FgRegion golden_region(const std::string& n) {
    if (n == "cells") return FgRegion::Input;
    if (n.rfind("wf_", 0) == 0 || n.rfind("b_", 0) == 0) return FgRegion::Weights;
    return FgRegion::Scratch;
}

bool kernel_by_name(const char* n, FgNet1Kernel* out) {
    for (uint32_t k = 0; k < kFgNet1KernelCount; ++k)
        if (std::string(FgNet1KernelName(static_cast<FgNet1Kernel>(k))) == n) {
            *out = static_cast<FgNet1Kernel>(k);
            return true;
        }
    return false;
}

TEST(net1_plan_matches_python_plan) {
    for (const wgfg_golden::Case& gc : wgfg_golden::kCases) {
        FgNet1Plan p;
        CHECK(FgBuildNet1Plan(gc.H0, gc.W0, &p) == FgRefusal::None);
        CHECK(p.buffers.size() == gc.nbufs && p.steps.size() == gc.nsteps && gc.nsteps == kFgNet1Steps);
        if (p.buffers.size() != gc.nbufs || p.steps.size() != gc.nsteps) continue;
        uint64_t end[3] = {};
        for (uint32_t i = 0; i < gc.nbufs; ++i) {
            const FgBufferSpec& b = p.buffers[i];
            CHECK(b.name == gc.bufs[i].name && b.bytes == gc.bufs[i].bytes);
            // the layout re-derived from the table alone: per region, in order, 256-byte aligned
            const FgRegion r = golden_region(gc.bufs[i].name);
            CHECK(b.region == r && b.offset == end[static_cast<uint32_t>(r)]);
            end[static_cast<uint32_t>(r)] = (end[static_cast<uint32_t>(r)] + gc.bufs[i].bytes + 255) / 256 * 256;
        }
        for (uint32_t r = 0; r < 3; ++r) CHECK(p.region_bytes[r] == end[r]);
        for (uint32_t si = 0; si < gc.nsteps; ++si) {
            const FgNet1Step& s = p.steps[si];
            const wgfg_golden::Step& g = gc.steps[si];
            CHECK(std::string(FgNet1KernelName(s.kernel)) == g.kernel && s.grid == g.grid && g.nargs == kFgNet1Args);
            for (uint32_t a = 0; a < kFgNet1Args; ++a) {
                if (g.args[a].buf) {
                    CHECK(s.args[a].buffer && p.buffers[s.args[a].value].name == g.args[a].buf);
                } else {
                    CHECK(!s.args[a].buffer && s.args[a].value == g.args[a].value);
                }
            }
            CHECK(s.nouts == g.nouts);
            for (uint32_t o = 0; o < g.nouts; ++o) CHECK(p.buffers[s.outs[o]].name == g.outs[o]);
        }
    }
    FgNet1Plan p;
    CHECK(FgBuildNet1Plan(31, 64, &p) == FgRefusal::Net1GridUnsupported);
    CHECK(FgBuildNet1Plan(64, 31, &p) == FgRefusal::Net1GridUnsupported);
    CHECK(FgBuildNet1Plan(64, 5000, &p) == FgRefusal::Net1GridUnsupported);
    CHECK(FgRatioBits(10, 20) == float_bits(9.0f / 19.0f) && FgRatioBits(1, 1) == 0 && FgRatioBits(5, 1) == 0);
}

struct Cmd {
    int op;  // 0 bind, 1 barrier, 2 pipeline, 3 constants, 4 dispatch
    uint32_t a;
    std::vector<uint32_t> c;
    bool operator==(const Cmd& o) const { return op == o.op && a == o.a && c == o.c; }
};

class FakeRecorder : public IFgComputeRecorder {
public:
    std::vector<Cmd> cmds;
    void BindArena() override { cmds.push_back({0, 0, {}}); }
    void UavBarrier() override { cmds.push_back({1, 0, {}}); }
    void SetPipeline(FgNet1Kernel k) override { cmds.push_back({2, static_cast<uint32_t>(k), {}}); }
    void SetConstants(const uint32_t* d, uint32_t n) override { cmds.push_back({3, n, std::vector<uint32_t>(d, d + n)}); }
    void Dispatch(uint32_t x) override { cmds.push_back({4, x, {}}); }
};

FgPipelineSet all_pipelines() {
    FgPipelineSet s;
    for (bool& b : s.ready) b = true;
    return s;
}

FgArenaBinding arena_for(const FgNet1Plan& p) {
    FgArenaBinding a;
    a.bound = true;
    a.region_base[1] = 256;                                              // weights after the null header
    a.region_base[0] = (256 + p.region_bytes[1] + 255) / 256 * 256;      // then the input cells
    a.region_base[2] = (a.region_base[0] + p.region_bytes[0] + 255) / 256 * 256;
    a.arena_bytes = a.region_base[2] + p.region_bytes[2];
    return a;
}

// The stream a recording must produce, from the generated table and the arena alone.
std::vector<Cmd> expected_stream(const wgfg_golden::Case& gc, const FgArenaBinding& a, uint32_t max_groups) {
    std::vector<std::pair<std::string, uint64_t>> where;
    uint64_t end[3] = {};
    for (uint32_t i = 0; i < gc.nbufs; ++i) {
        const uint32_t r = static_cast<uint32_t>(golden_region(gc.bufs[i].name));
        where.push_back({gc.bufs[i].name, a.region_base[r] + end[r]});
        end[r] = (end[r] + gc.bufs[i].bytes + 255) / 256 * 256;
    }
    auto addr = [&where](const char* n) {
        for (const auto& w : where)
            if (w.first == n) return static_cast<uint32_t>(w.second);
        return 0xFFFFFFFFu;
    };
    std::vector<Cmd> e;
    e.push_back({0, 0, {}});
    for (uint32_t si = 0; si < gc.nsteps; ++si) {
        const wgfg_golden::Step& g = gc.steps[si];
        FgNet1Kernel k{};
        kernel_by_name(g.kernel, &k);
        e.push_back({1, 0, {}});
        e.push_back({2, static_cast<uint32_t>(k), {}});
        for (uint32_t base = 0; base < g.grid; base += max_groups) {
            std::vector<uint32_t> c = {base, g.grid};
            for (uint32_t i = 0; i < g.nargs; ++i) c.push_back(g.args[i].buf ? addr(g.args[i].buf) : g.args[i].value);
            e.push_back({3, 2 + g.nargs, c});
            e.push_back({4, g.grid - base < max_groups ? g.grid - base : max_groups, {}});
        }
    }
    e.push_back({1, 0, {}});
    return e;
}

TEST(net1_recording_matches_table_through_fake_recorder) {
    for (const wgfg_golden::Case& gc : wgfg_golden::kCases) {
        FgNet1Plan p;
        CHECK(FgBuildNet1Plan(gc.H0, gc.W0, &p) == FgRefusal::None);
        const FgArenaBinding a = arena_for(p);
        for (uint32_t limit : {kFgMaxGroupsPerDispatch, 7u}) {
            FakeRecorder rec;
            std::string why;
            CHECK(FgRecordNet1(p, all_pipelines(), a, rec, &why, limit) == FgRefusal::None);
            const std::vector<Cmd> want = expected_stream(gc, a, limit);
            CHECK(rec.cmds == want);
            uint64_t groups = 0, launches = 0;
            for (const Cmd& c : rec.cmds) {
                if (c.op == 4) groups += c.a;
                if (c.op == 2) ++launches;
                if (c.op == 4) CHECK(c.a >= 1 && c.a <= limit);
            }
            uint64_t want_groups = 0;
            for (uint32_t si = 0; si < gc.nsteps; ++si) want_groups += gc.steps[si].grid;
            CHECK(groups == want_groups && launches == kFgNet1Steps);
            // a corrupted stream is not equal to the expectation (the comparison is not vacuous)
            if (!rec.cmds.empty()) {
                std::vector<Cmd> bad = rec.cmds;
                for (Cmd& c : bad)
                    if (c.op == 3) {
                        c.c[2 + 3] += 256;  // the first launch's output pointer moved by one allocation
                        break;
                    }
                CHECK(!(bad == want));
            }
        }
    }
}

FgNet1Plan base_plan() {
    FgNet1Plan p;
    FgBuildNet1Plan(192, 320, &p);
    return p;
}

TEST(net1_plan_mutants_are_refused_and_record_nothing) {
    struct Mut {
        const char* what;
        void (*f)(FgNet1Plan&);
    };
    const Mut muts[] = {
        {"grid + 1", [](FgNet1Plan& p) { p.steps[0].grid += 1; }},
        {"encoder fed the wrong level", [](FgNet1Plan& p) { p.steps[3].args[0].value = static_cast<uint32_t>(p.Find("a2")); }},
        {"upscale ratio bit", [](FgNet1Plan& p) { p.steps[7].args[12].value ^= 1u; }},
        {"overlapping scratch", [](FgNet1Plan& p) { p.buffers[p.Find("a1")].offset = p.buffers[p.Find("a0")].offset; }},
        {"misaligned offset", [](FgNet1Plan& p) { p.buffers[p.Find("p2")].offset += 16; }},
        {"wrong kernel", [](FgNet1Plan& p) { p.steps[5].kernel = FgNet1Kernel::MPairD4; }},
        {"activation on the output", [](FgNet1Plan& p) { p.steps[10].args[14].value = 1; }},
        {"buffer written twice", [](FgNet1Plan& p) {
             p.steps[9].args[6].value = static_cast<uint32_t>(p.Find("d2"));
             p.steps[9].outs[0] = static_cast<uint32_t>(p.Find("d2"));
         }},
        {"skip of another level", [](FgNet1Plan& p) { p.steps[8].args[1].value = static_cast<uint32_t>(p.Find("a1")); }},
        {"tiles_x", [](FgNet1Plan& p) { p.steps[1].args[13].value += 1; }},
        {"weight size", [](FgNet1Plan& p) {
             FgBufferSpec& b = p.buffers[p.Find("wf_net1_03")];
             b.c -= 1;
             b.bytes -= 2;
         }},
        {"region too small", [](FgNet1Plan& p) { p.region_bytes[2] -= 256; }},
        {"launch missing", [](FgNet1Plan& p) { p.steps.pop_back(); }},
        {"encoder activation off", [](FgNet1Plan& p) { p.steps[2].args[11].value = 0; }},
        {"argument kind", [](FgNet1Plan& p) { p.steps[4].args[0].buffer = false; }},
        {"decoder not fed by the previous launch", [](FgNet1Plan& p) { p.steps[6].args[0].value = static_cast<uint32_t>(p.Find("p5")); }},
        {"pool flag off", [](FgNet1Plan& p) { p.steps[0].args[12].value = 0; }},
        {"bottleneck as upscale", [](FgNet1Plan& p) { p.steps[5].args[10].value = 2; }},
        {"cells shape", [](FgNet1Plan& p) {
             FgBufferSpec& b = p.buffers[p.Find("cells")];
             b.c = 40;
             b.bytes = 2ull * b.h * b.w * 40;
         }},
        {"duplicate name", [](FgNet1Plan& p) { p.buffers[p.Find("d1")].name = "d2"; }},
        {"buffer index out of range", [](FgNet1Plan& p) { p.steps[2].args[1].value = 999; }},
    };
    const FgNet1Plan good = base_plan();
    std::string why;
    CHECK(FgValidateNet1Plan(good, &why) == FgRefusal::None);
    for (const Mut& m : muts) {
        FgNet1Plan p = good;
        m.f(p);
        const FgRefusal v = FgValidateNet1Plan(p, &why);
        CHECK(v != FgRefusal::None);
        if (v == FgRefusal::None) std::printf("   mutant not caught: %s\n", m.what);
        FakeRecorder rec;
        CHECK(FgRecordNet1(p, all_pipelines(), arena_for(good), rec, &why) != FgRefusal::None);
        CHECK(rec.cmds.empty());
    }
}

TEST(net1_recording_refusals_record_nothing) {
    const FgNet1Plan p = base_plan();
    const FgArenaBinding good = arena_for(p);
    std::string why;
    for (uint32_t k = 0; k < kFgNet1KernelCount; ++k) {
        FgPipelineSet s = all_pipelines();
        s.ready[k] = false;
        FakeRecorder rec;
        const FgRefusal r = FgRecordNet1(p, s, good, rec, &why);
        // fg_mconv1 is in no launch of net1 (every encoder's output count is a multiple of 32)
        CHECK(r == (k == 0 ? FgRefusal::None : FgRefusal::Net1PipelineMissing));
        CHECK(k == 0 ? !rec.cmds.empty() : rec.cmds.empty());
    }
    struct A {
        void (*f)(FgArenaBinding&);
        FgRefusal want;
    };
    const A as[] = {
        {[](FgArenaBinding& a) { a.bound = false; }, FgRefusal::Net1ArenaMissing},
        {[](FgArenaBinding& a) { a.arena_bytes -= 2; }, FgRefusal::Net1ArenaTooSmall},
        {[](FgArenaBinding& a) { a.region_base[2] = a.region_base[0]; }, FgRefusal::Net1RegionOverlap},
        {[](FgArenaBinding& a) { a.region_base[0] += 64; }, FgRefusal::Net1RegionAlign},
        {[](FgArenaBinding& a) {
             a.arena_bytes = 1ull << 34;
             a.region_base[2] = 0xFFFF0000ull;
         }, FgRefusal::Net1ArenaTooSmall},
    };
    for (const A& t : as) {
        FgArenaBinding a = good;
        t.f(a);
        FakeRecorder rec;
        CHECK(FgRecordNet1(p, all_pipelines(), a, rec, &why) == t.want);
        CHECK(rec.cmds.empty());
    }
    FakeRecorder rec;
    CHECK(FgRecordNet1(p, all_pipelines(), good, rec, &why, 0) == FgRefusal::Net1PlanInvalid && rec.cmds.empty());
    CHECK(FgRecordNet1(p, all_pipelines(), good, rec, &why, 65536) == FgRefusal::Net1PlanInvalid && rec.cmds.empty());
}

// ---- frame ingest ----------------------------------------------------------------------------------------------------

int g_res[16];

FgTextureRef tex(int slot, uint64_t token, uint32_t w, uint32_t h, uint32_t fmt = 28) {
    FgTextureRef t;
    t.resource = &g_res[slot];
    t.format = fmt;
    t.width = w;
    t.height = h;
    t.frame_token = token;
    t.viewport_id = 1;
    t.rect = {0, 0, w, h};
    return t;
}

FgFramePacket constants(uint64_t token) {
    const FgCreateDesc s = good_create();
    FgFramePacket p;
    p.frame_token = token;
    p.viewport_id = s.viewport_id;
    p.adapter_luid = s.adapter_luid;
    p.queue_epoch = s.queue_epoch;
    p.swapchain_epoch = s.swapchain_epoch;
    p.output_rect = {0, 0, 1920, 1080};
    p.render_rect = {0, 0, 1280, 720};
    p.generated_count = 3;
    return p;
}

// A complete game frame for good_create(): color, render-size motion and depth, the promised UI alpha.
FgFramePacket frame(uint64_t token) {
    FgFramePacket p = constants(token);
    p.flags = kFgFlagMotionRenderRes;
    p.color = tex(0, token, 1920, 1080);
    p.motion = tex(1, token, 1280, 720, 16);
    p.depth = tex(2, token, 1280, 720, 40);
    p.optional[kFgOptUiAlpha] = tex(4, token, 1920, 1080, 61);
    p.optional_present = FgOptBit(kFgOptUiAlpha);
    return p;
}

TEST(assembler_builds_and_refuses) {
    const FgCreateDesc s = good_create();
    FgFrameAssembler as(s);
    FgFramePacket out;
    CHECK(as.Tag(FgTagSlot::Color, tex(0, 5, 1920, 1080)) == FgRefusal::NoFrameOpen);
    CHECK(as.Finish(constants(5), &out) == FgRefusal::NoFrameOpen);
    CHECK(as.Begin(5) == FgRefusal::None);
    CHECK(as.Tag(FgTagSlot::Color, tex(0, 5, 1920, 1080)) == FgRefusal::None);
    CHECK(as.Tag(FgTagSlot::Color, tex(0, 5, 1920, 1080)) == FgRefusal::None);  // the same tag again
    FgTextureRef other = tex(3, 5, 1920, 1080);
    CHECK(as.Tag(FgTagSlot::Color, other) == FgRefusal::ConflictingTag);
    CHECK(as.Tag(FgTagSlot::Motion, tex(1, 6, 1280, 720)) == FgRefusal::TagTokenMismatch);
    CHECK(as.Tag(FgTagSlot::Motion, tex(1, 4, 1280, 720)) == FgRefusal::StaleTag);
    FgTextureRef vp = tex(1, 5, 1280, 720);
    vp.viewport_id = 2;
    CHECK(as.Tag(FgTagSlot::Motion, vp) == FgRefusal::TagTokenMismatch);
    CHECK(as.Tag(static_cast<FgTagSlot>(9), tex(1, 5, 1, 1)) == FgRefusal::UnknownTag);
    CHECK(as.Tag(FgTagSlot::Motion, tex(1, 5, 1280, 720)) == FgRefusal::None);
    CHECK(as.Tag(FgTagSlot::Depth, tex(2, 5, 1280, 720)) == FgRefusal::None);
    CHECK(as.Tag(FgTagSlot::UiAlpha, tex(4, 5, 1920, 1080)) == FgRefusal::None);
    CHECK(as.Tag(FgTagSlot::NoWarp, tex(6, 5, 1920, 1080)) == FgRefusal::None);  // declared May
    CHECK(as.Finish(constants(5), &out) == FgRefusal::None);
    CHECK(!as.open() && out.frame_token == 5 && out.color.resource == &g_res[0] && out.depth.resource == &g_res[2]);
    CHECK(out.optional_present == (FgOptBit(kFgOptUiAlpha) | FgOptBit(kFgOptNoWarp)));
    CHECK(as.Begin(5) == FgRefusal::StaleTag);  // a finished token never reopens
    // missing required and promised inputs; an input declared never
    struct C {
        bool color, motion, depth, ui, distortion;
        FgRefusal want;
    };
    const C cs[] = {
        {false, true, true, true, false, FgRefusal::MissingColor},
        {true, false, true, true, false, FgRefusal::MissingMotion},
        {true, true, false, true, false, FgRefusal::MissingDepth},
        {true, true, true, false, false, FgRefusal::MissingPromisedOptional},
        {true, true, true, true, true, FgRefusal::UndeclaredOptional},
    };
    uint64_t t = 10;
    for (const C& c : cs) {
        ++t;
        CHECK(as.Begin(t) == FgRefusal::None);
        if (c.color) as.Tag(FgTagSlot::Color, tex(0, t, 1920, 1080));
        if (c.motion) as.Tag(FgTagSlot::Motion, tex(1, t, 1280, 720));
        if (c.depth) as.Tag(FgTagSlot::Depth, tex(2, t, 1280, 720));
        if (c.ui) as.Tag(FgTagSlot::UiAlpha, tex(4, t, 1920, 1080));
        if (c.distortion) as.Tag(FgTagSlot::Distortion, tex(5, t, 1920, 1080));
        FgFramePacket o2;
        o2.frame_token = 777;
        CHECK(as.Finish(constants(t), &o2) == c.want);
        CHECK(o2.frame_token == 777 && !as.open());  // nothing written on refusal; the frame is closed
    }
    // a new frame drops the open one and its tags
    CHECK(as.Begin(30) == FgRefusal::None);
    as.Tag(FgTagSlot::Color, tex(0, 30, 1920, 1080));
    CHECK(as.Begin(31) == FgRefusal::None);
    as.Tag(FgTagSlot::Motion, tex(1, 31, 1280, 720));
    as.Tag(FgTagSlot::Depth, tex(2, 31, 1280, 720));
    as.Tag(FgTagSlot::UiAlpha, tex(4, 31, 1920, 1080));
    CHECK(as.Finish(constants(31), &out) == FgRefusal::MissingColor);
}

// ---- history ---------------------------------------------------------------------------------------------------------

bool invariants(const FgHistory& h) {
    std::string why;
    const bool ok = FgCheckHistoryInvariants(h.Snapshot(), &why);
    if (!ok) std::printf("   invariant: %s\n", why.c_str());
    return ok;
}

TEST(history_primes_then_generates_and_never_rotates_on_an_index) {
    FgHistory h(good_create(), 2);
    CHECK(!h.primed() && invariants(h));
    FgIngestResult r = h.Accept(frame(100), 0);
    CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Prime && r.current_bank >= 0);
    CHECK(h.primed() && !h.interval_open() && h.primes() == 1 && invariants(h));
    // an evaluation before an interval exists
    FgFramePacket ev = frame(100);
    ev.generated_index = 1;
    CHECK(h.Evaluate(ev).refusal == FgRefusal::HistoryNotPrimed);
    r = h.Accept(frame(101), 0);
    CHECK(r.status == FgFrameStatus::Generated && r.generated_count == 3 && r.previous_bank != r.current_bank);
    CHECK(h.bank(static_cast<uint32_t>(r.previous_bank)).frame_token == 100);
    CHECK(h.bank(static_cast<uint32_t>(r.current_bank)).frame_token == 101);
    CHECK(h.rotations() == 1 && invariants(h));
    const FgHistorySnapshot before = h.Snapshot();
    for (uint32_t i = 1; i <= 3; ++i) {
        FgFramePacket e = frame(101);
        e.generated_index = i;
        const FgIngestResult x = h.Evaluate(e);
        CHECK(x.status == FgFrameStatus::Generated && x.generated_index == i && x.previous_bank == r.previous_bank &&
              x.current_bank == r.current_bank);
        CHECK(float_bits(x.t) == float_bits(FgIndexTime(i, 3)));
    }
    // an index handed to Accept is refused without any change
    FgFramePacket e = frame(102);
    e.generated_index = 2;
    CHECK(h.Accept(e, 0).refusal == FgRefusal::BadGeneratedIndex);
    const FgHistorySnapshot after = h.Snapshot();
    CHECK(h.rotations() == 1 && after.previous == before.previous && after.current == before.current);
    for (uint32_t b = 0; b < 2; ++b) CHECK(after.banks[b].writes == before.banks[b].writes);
    // evaluation refusals
    e = frame(101);
    e.generated_index = 0;
    CHECK(h.Evaluate(e).refusal == FgRefusal::BadGeneratedIndex);
    e = frame(100);
    e.generated_index = 1;
    CHECK(h.Evaluate(e).refusal == FgRefusal::IntervalMismatch);
    e = frame(101);
    e.generated_count = 2;
    e.generated_index = 1;
    CHECK(h.Evaluate(e).refusal == FgRefusal::IntervalMismatch);
    e = frame(101);
    e.generated_count = 5;
    e.generated_index = 4;
    CHECK(h.Evaluate(e).refusal == FgRefusal::IntervalMismatch);
    e = frame(101);
    e.generated_index = 1;
    e.viewport_id = 2;
    CHECK(h.Evaluate(e).refusal == FgRefusal::ViewportMismatch);
    // the next game frame rotates once more
    r = h.Accept(frame(102), 0);
    CHECK(r.status == FgFrameStatus::Generated && h.rotations() == 2 && h.bank(static_cast<uint32_t>(r.previous_bank)).frame_token == 101);
    CHECK(invariants(h));
}

TEST(history_identity_and_input_refusals_change_nothing) {
    FgHistory h(good_create(), 2);
    h.Accept(frame(10), 0);
    h.Accept(frame(11), 0);
    const FgHistorySnapshot before = h.Snapshot();
    struct M {
        void (*f)(FgFramePacket&);
        FgRefusal want;
    };
    const M ms[] = {
        {[](FgFramePacket& p) { p.viewport_id = 2; }, FgRefusal::ViewportMismatch},
        {[](FgFramePacket& p) { p.adapter_luid.low ^= 1; }, FgRefusal::LuidMismatch},
        {[](FgFramePacket& p) { p.queue_epoch = 4; }, FgRefusal::QueueEpochMismatch},
        {[](FgFramePacket& p) { p.swapchain_epoch = 8; }, FgRefusal::SwapchainEpochMismatch},
        {[](FgFramePacket& p) { p.frame_token = 11; }, FgRefusal::TokenNotMonotonic},
        {[](FgFramePacket& p) { p.frame_token = 5; }, FgRefusal::TokenNotMonotonic},
        {[](FgFramePacket& p) { p.color.resource = nullptr; }, FgRefusal::MissingColor},
        {[](FgFramePacket& p) { p.motion.resource = nullptr; }, FgRefusal::MissingMotion},
        {[](FgFramePacket& p) { p.depth.resource = nullptr; }, FgRefusal::MissingDepth},
        {[](FgFramePacket& p) { p.depth.frame_token = 11; }, FgRefusal::TagTokenMismatch},
        {[](FgFramePacket& p) { p.optional_present = 0; }, FgRefusal::MissingPromisedOptional},
        {[](FgFramePacket& p) { p.optional[kFgOptUiAlpha].resource = nullptr; }, FgRefusal::MissingPromisedOptional},
        {[](FgFramePacket& p) {
             p.optional[kFgOptHudless] = tex(3, 12, 1920, 1080);
             p.optional_present |= FgOptBit(kFgOptHudless);
             p.optional[kFgOptHudless].frame_token = 9;
         }, FgRefusal::TagTokenMismatch},
        {[](FgFramePacket& p) {
             p.optional[kFgOptDistortion] = tex(5, 12, 1920, 1080);
             p.optional_present |= FgOptBit(kFgOptDistortion);
         }, FgRefusal::UndeclaredOptional},
        {[](FgFramePacket& p) { p.optional_present |= 1u << 6; }, FgRefusal::UndeclaredOptional},
        {[](FgFramePacket& p) { p.color.format = 10; }, FgRefusal::BadFormat},
        {[](FgFramePacket& p) { p.output_rect.width = 1280; }, FgRefusal::OutputRectMismatch},
        {[](FgFramePacket& p) { p.color.width = 1280; }, FgRefusal::OutputRectMismatch},
        {[](FgFramePacket& p) { p.render_rect.width = 1000; }, FgRefusal::RenderRectOutsideBounds},
        {[](FgFramePacket& p) { p.render_rect.height = 1200; }, FgRefusal::RenderRectOutsideBounds},
        {[](FgFramePacket& p) { p.depth.width = 1000; }, FgRefusal::TextureExtent},
        {[](FgFramePacket& p) { p.flags &= ~static_cast<uint32_t>(kFgFlagMotionRenderRes); }, FgRefusal::TextureExtent},
        {[](FgFramePacket& p) { p.generated_count = 0; }, FgRefusal::BadGeneratedCount},
        {[](FgFramePacket& p) { p.struct_size = 16; }, FgRefusal::StructSize},
    };
    for (const M& m : ms) {
        FgFramePacket p = frame(12);
        m.f(p);
        const FgIngestResult r = h.Accept(p, 0);
        CHECK(r.status == FgFrameStatus::Refused && r.refusal == m.want);
        if (r.refusal != m.want) std::printf("   want %s got %s\n", FgRefusalName(m.want), FgRefusalName(r.refusal));
        const FgHistorySnapshot now = h.Snapshot();
        CHECK(now.previous == before.previous && now.current == before.current && now.epoch == before.epoch);
        for (uint32_t b = 0; b < 2; ++b) CHECK(now.banks[b].writes == before.banks[b].writes);
    }
    CHECK(h.rotations() == 1 && h.base_presents() == 2);
    // frame 12 never got through, so 13 is a gap: re-prime
    const FgIngestResult r = h.Accept(frame(13), 0);
    CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Gap && r.epoch == before.epoch + 1);
    CHECK(invariants(h));
}

TEST(history_epochs_reset_menu_resize_rebind) {
    FgHistory h(good_create(), 2);
    h.Accept(frame(1), 0);
    CHECK(h.Accept(frame(2), 0).status == FgFrameStatus::Generated);
    const uint64_t e0 = h.epoch();
    // reset / camera cut: this frame primes a new epoch
    FgFramePacket p = frame(3);
    p.flags |= kFgFlagReset;
    FgIngestResult r = h.Accept(p, 0);
    CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Reset && r.epoch == e0 + 1);
    CHECK(h.primed() && !h.interval_open() && invariants(h));
    CHECK(h.Accept(frame(4), 0).status == FgFrameStatus::Generated);
    // menu: real-only and not stored; the next frame primes
    p = frame(5);
    p.flags |= kFgFlagMenu;
    r = h.Accept(p, 0);
    CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Menu && !h.primed() && invariants(h));
    p = frame(6);
    p.flags |= kFgFlagMenu;
    CHECK(h.Accept(p, 0).real_only == FgFrameRealOnly::Menu && !h.primed());
    r = h.Accept(frame(7), 0);
    CHECK(r.real_only == FgFrameRealOnly::Prime && h.primed());
    CHECK(h.Accept(frame(8), 0).status == FgFrameStatus::Generated);
    // a render-size change (dynamic resolution) re-primes until its compatibility rule is qualified
    auto resized = [](uint64_t token) {
        FgFramePacket q = frame(token);
        q.render_rect = {0, 0, 1600, 900};
        q.motion.width = q.depth.width = 1600;
        q.motion.height = q.depth.height = 900;
        return q;
    };
    r = h.Accept(resized(9), 0);
    CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Resize && invariants(h));
    CHECK(h.Accept(resized(10), 0).status == FgFrameStatus::Generated);
    // the chain is replaced: frames of the old epochs are refused, the first one of the new chain primes
    const uint64_t e1 = h.epoch();
    h.Rebind(4, 10);
    CHECK(!h.primed() && h.epoch() == e1 + 1 && invariants(h));
    CHECK(h.Accept(frame(11), 0).refusal == FgRefusal::QueueEpochMismatch);
    p = frame(1);  // a new chain may restart its tokens
    p.queue_epoch = 4;
    p.swapchain_epoch = 10;
    r = h.Accept(p, 0);
    CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Prime);
    p = frame(2);
    p.queue_epoch = 4;
    p.swapchain_epoch = 10;
    CHECK(h.Accept(p, 0).status == FgFrameStatus::Generated && invariants(h));
    h.Invalidate();
    CHECK(!h.primed() && invariants(h));
}

TEST(history_never_overwrites_a_bank_compute_still_reads) {
    {
        FgHistory h(good_create(), 2);
        h.Accept(frame(1), 0);
        FgIngestResult r = h.Accept(frame(2), 0);
        const int prev = r.previous_bank, cur = r.current_bank;
        CHECK(h.MarkRead(10));  // interval (1, 2) is read by compute until fence 10
        const uint32_t w0 = h.bank(0).writes, w1 = h.bank(1).writes;
        r = h.Accept(frame(3), 5);  // the bank frame 3 needs is still read: no rotation, no write
        CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::HistoryBusy);
        CHECK(h.bank(0).writes == w0 && h.bank(1).writes == w1 && !h.primed() && invariants(h));
        CHECK(h.bank(static_cast<uint32_t>(prev)).role == FgBankRole::Retired &&
              h.bank(static_cast<uint32_t>(cur)).role == FgBankRole::Retired);
        r = h.Accept(frame(4), 9);  // still not complete
        CHECK(r.real_only == FgFrameRealOnly::HistoryBusy && h.bank(0).writes == w0 && h.bank(1).writes == w1);
        r = h.Accept(frame(5), 10);  // complete: the banks come back, frame 5 primes
        CHECK(r.status == FgFrameStatus::RealOnly && r.real_only == FgFrameRealOnly::Prime && invariants(h));
        CHECK(h.Accept(frame(6), 10).status == FgFrameStatus::Generated);
        CHECK(!h.MarkRead(0));
    }
    {
        // a third bank absorbs one interval in flight
        FgHistory h(good_create(), 3);
        h.Accept(frame(1), 0);
        FgIngestResult r = h.Accept(frame(2), 0);
        const int prev = r.previous_bank;
        h.MarkRead(10);
        r = h.Accept(frame(3), 5);
        CHECK(r.status == FgFrameStatus::Generated && r.previous_bank != prev && r.current_bank != prev);
        CHECK(h.bank(static_cast<uint32_t>(prev)).role == FgBankRole::Retired && invariants(h));
        // the previous (frame 2) is still read too: frame 4 finds no bank
        r = h.Accept(frame(4), 5);
        CHECK(r.real_only == FgFrameRealOnly::HistoryBusy && invariants(h));
        h.Retire(10);
        for (uint32_t b = 0; b < 3; ++b) CHECK(h.bank(b).role == FgBankRole::Free && h.bank(b).read_fence == 0);
        CHECK(h.Accept(frame(5), 10).real_only == FgFrameRealOnly::Prime);
    }
    {
        // bank counts are clamped to 2..4
        FgHistory a(good_create(), 0), b(good_create(), 9);
        CHECK(a.banks() == 2 && b.banks() == 4);
    }
}

TEST(history_invariant_checker_catches_corruption) {
    FgHistory h(good_create(), 3);
    h.Accept(frame(1), 0);
    h.Accept(frame(2), 0);
    h.MarkRead(4);
    h.Accept(frame(3), 0);
    const FgHistorySnapshot good = h.Snapshot();
    std::string why;
    CHECK(FgCheckHistoryInvariants(good, &why));
    struct M {
        void (*f)(FgHistorySnapshot&);
    };
    const M ms[] = {
        {[](FgHistorySnapshot& s) { s.banks[s.previous].role = FgBankRole::Current; }},
        {[](FgHistorySnapshot& s) { std::swap(s.banks[s.previous].frame_token, s.banks[s.current].frame_token); }},
        {[](FgHistorySnapshot& s) { s.banks[s.previous].epoch = s.epoch + 1; }},
        {[](FgHistorySnapshot& s) { s.banks[s.current].epoch = s.epoch - 1; }},
        {[](FgHistorySnapshot& s) {
             s.banks[s.current].role = FgBankRole::Free;
             s.current = -1;
         }},
        {[](FgHistorySnapshot& s) { s.previous = s.current; }},
        {[](FgHistorySnapshot& s) {
             for (uint32_t i = 0; i < s.nbanks; ++i)
                 if (s.banks[i].role == FgBankRole::Retired) s.banks[i].read_fence = 0;
         }},
        {[](FgHistorySnapshot& s) { s.nbanks = 5; }},
        {[](FgHistorySnapshot& s) { s.current = 3; }},
        {[](FgHistorySnapshot& s) { s.banks[s.previous].role = FgBankRole::Free; }},
    };
    for (const M& m : ms) {
        FgHistorySnapshot s = good;
        m.f(s);
        CHECK(!FgCheckHistoryInvariants(s, &why));
    }
}

// ---- create-time worker ----------------------------------------------------------------------------------------------

std::atomic<int> g_released{0};
void release_count(void*) { ++g_released; }

struct OkJob {
    FgRefusal operator()() const { return FgRefusal::None; }
};
struct ThrowJob {
    FgRefusal operator()() const { throw std::runtime_error("synthetic create failure"); }
};
struct BlockJob {
    std::atomic<bool>* go;
    FgRefusal operator()() const {
        while (!go->load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return FgRefusal::ModelStageDigest;
    }
};
// A callable whose copy throws, like a large std::function failing to allocate.
struct ThrowingCopy {
    ThrowingCopy() = default;
    ThrowingCopy(const ThrowingCopy&) { throw std::bad_alloc(); }
    ThrowingCopy& operator=(const ThrowingCopy&) = delete;
    FgRefusal operator()() const { return FgRefusal::None; }
};

void throwing_launcher(std::function<void()>) { throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again)); }

struct CleanupLog {
    std::atomic<int> calls{0};
    std::atomic<int> last{-1};
};
void cleanup_fn(void* user, FgRefusal r) {
    CleanupLog* l = static_cast<CleanupLog*>(user);
    l->last = static_cast<int>(r);
    ++l->calls;
}

TEST(worker_success_and_hold_released_once) {
    g_released = 0;
    auto p = FgPendingInit::Start(FgPendingInit::Work(FgCopiedJob<OkJob>(OkJob{})), FgHold(release_count, nullptr));
    CHECK(p != nullptr);
    CHECK(p->Wait(UINT32_MAX) && p->Done() && p->Result() == FgRefusal::None);
    for (int i = 0; i < 200 && g_released.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(g_released.load() == 1);
}

TEST(worker_failures_become_refusals) {
    g_released = 0;
    auto p = FgPendingInit::Start(FgPendingInit::Work(FgCopiedJob<ThrowJob>(ThrowJob{})), FgHold(release_count, nullptr));
    CHECK(p && p->Wait(5000) && p->Result() == FgRefusal::WorkerInitFailed);
    for (int i = 0; i < 200 && g_released.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(g_released.load() == 1);
    // the worker cannot start: finished at once, hold given back before Start returns, the work never ran
    g_released = 0;
    FgPendingInit::SetLauncherForTest(throwing_launcher);
    std::atomic<bool> go{true};
    p = FgPendingInit::Start(FgPendingInit::Work(FgCopiedJob<BlockJob>(BlockJob{&go})), FgHold(release_count, nullptr));
    FgPendingInit::SetLauncherForTest(nullptr);
    CHECK(p && p->Done() && p->Result() == FgRefusal::WorkerStartFailed && g_released.load() == 1);
    // empty work
    p = FgPendingInit::Start(FgPendingInit::Work(), FgHold());
    CHECK(p && p->Wait(5000) && p->Result() == FgRefusal::WorkerInitFailed);
}

TEST(worker_named_callable_copy_failure_unwinds) {
    // The copy that throws happens in a named class's constructor (FgCopiedJob) and in std::function's construction
    // from it. Built with /permissive- (/Zc:lambda), both unwind normally; the hold is released by its destructor.
    g_released = 0;
    bool caught = false;
    try {
        FgHold hold(release_count, nullptr);
        const ThrowingCopy t{};
        FgCopiedJob<ThrowingCopy> job(t);  // throws here
        (void)FgPendingInit::Start(FgPendingInit::Work(job), std::move(hold));
    } catch (const std::bad_alloc&) {
        caught = true;
    }
    CHECK(caught && g_released.load() == 1);
    // the model job copies its path in its constructor and runs the model check on the worker
    const SynthModel m = build_model();
    const std::string path = "fg_runtime_test_model.wgcm";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(m.bytes.data()), static_cast<std::streamsize>(m.bytes.size()));
    }
    auto p = FgPendingInit::Start(FgPendingInit::Work(FgModelJob(path, m.expect())), FgHold());
    CHECK(p && p->Wait(30000) && p->Result() == FgRefusal::None);
    p = FgPendingInit::Start(FgPendingInit::Work(FgModelJob(path, FgShippedModelExpectation())), FgHold());
    CHECK(p && p->Wait(30000) && p->Result() == FgRefusal::ModelMapHash);
    std::remove(path.c_str());
    p = FgPendingInit::Start(FgPendingInit::Work(FgModelJob(path, m.expect())), FgHold());
    CHECK(p && p->Wait(30000) && p->Result() == FgRefusal::ModelMissing);
}

TEST(worker_bounded_wait_and_abandon) {
    std::atomic<bool> go{false};
    CleanupLog log;
    g_released = 0;
    auto p = FgPendingInit::Start(FgPendingInit::Work(FgCopiedJob<BlockJob>(BlockJob{&go})), FgHold(release_count, nullptr));
    CHECK(p != nullptr);
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(!p->Wait(20));
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    CHECK(!p->Done() && p->Result() == FgRefusal::WorkerAbandoned);
    FgPendingInit::Abandon(p, cleanup_fn, &log);  // the context goes away while the work runs
    CHECK(log.calls.load() == 0);
    go = true;
    CHECK(p->Wait(5000));
    for (int i = 0; i < 500 && (log.calls.load() == 0 || g_released.load() == 0); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(log.calls.load() == 1 && log.last.load() == static_cast<int>(FgRefusal::ModelStageDigest));
    CHECK(g_released.load() == 1);
    // abandoned after it finished: the cleanup runs at once
    CleanupLog log2;
    FgPendingInit::Abandon(p, cleanup_fn, &log2);
    CHECK(log2.calls.load() == 1);
    CleanupLog log3;
    FgPendingInit::Abandon(nullptr, cleanup_fn, &log3);
    CHECK(log3.calls.load() == 1 && log3.last.load() == static_cast<int>(FgRefusal::WorkerStartFailed));
}

// The developer frame dumps' CPU side (fg_frame_dump.h): slot bytes of each slot format -> RGB8, and the PPM file.
// Synthetic pixels.
TEST(frame_dump_rgb8_and_ppm) {
    std::vector<uint8_t> rgb;
    uint8_t rgba[2 * 16] = {10, 20, 30, 255, 40, 50, 60, 0};  // 2 x 2, row pitch 16 (8 bytes used per row)
    const uint8_t row1[8] = {70, 80, 90, 1, 100, 110, 120, 2};
    std::memcpy(rgba + 16, row1, sizeof row1);
    CHECK(FgDumpToRgb8(rgba, 16, 2, 2, kFgDumpRgba8, &rgb) && rgb.size() == 12);
    CHECK(rgb[0] == 10 && rgb[1] == 20 && rgb[2] == 30 && rgb[3] == 40 && rgb[6] == 70 && rgb[11] == 120);
    CHECK(FgDumpToRgb8(rgba, 16, 2, 2, kFgDumpBgra8Srgb, &rgb) && rgb[0] == 30 && rgb[1] == 20 && rgb[2] == 10);
    const uint32_t px = (1023u << 20) | (512u << 10) | 4u;  // R 4, G 512, B 1023 (10 bits each)
    uint8_t pxb[4];
    std::memcpy(pxb, &px, 4);
    CHECK(FgDumpToRgb8(pxb, 4, 1, 1, kFgDumpRgb10A2, &rgb) && rgb[0] == 1 && rgb[1] == 128 && rgb[2] == 255);
    const uint16_t half[4] = {0x3C00, 0x3800, 0xBC00, 0x3C00};  // 1.0, 0.5, -1.0, alpha
    uint8_t hb[8];
    std::memcpy(hb, half, 8);
    CHECK(FgDumpToRgb8(hb, 8, 1, 1, kFgDumpRgba16Float, &rgb) && rgb[0] == 255 && rgb[1] == 128 && rgb[2] == 0);
    CHECK(FgDumpHalfToFloat(0x0001) == std::ldexp(1.0f, -24) && FgDumpHalfToFloat(0x7BFF) == 65504.0f);
    CHECK(FgDumpHalfToFloat(0x8000) == 0.0f && FgDumpUnit8(FgDumpHalfToFloat(0x7E00)) == 0);  // NaN -> 0
    CHECK(FgDumpUnit8(FgDumpHalfToFloat(0x4000)) == 255);                                     // 2.0 clamped
    // refusals: a pitch too small for a row, an unknown format, a zero size, no source
    CHECK(!FgDumpToRgb8(rgba, 7, 2, 2, kFgDumpRgba8, &rgb));
    CHECK(!FgDumpToRgb8(rgba, 16, 2, 2, 2u, &rgb) && FgDumpBytesPerPixel(2u) == 0);
    CHECK(!FgDumpToRgb8(rgba, 16, 0, 2, kFgDumpRgba8, &rgb) && !FgDumpToRgb8(nullptr, 16, 2, 2, kFgDumpRgba8, &rgb));
    // the file: the P6 header, then the pixels
    CHECK(FgDumpPpmHeader(2, 3) == "P6\n2 3\n255\n");
    CHECK(FgDumpToRgb8(rgba, 16, 2, 2, kFgDumpRgba8, &rgb));
    const std::string path = "fg_frame_dump_test.ppm";
    CHECK(FgDumpWritePpm(path, 2, 2, rgb));
    std::ifstream f(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK(bytes.size() == 11 + 12 && bytes.compare(0, 11, "P6\n2 2\n255\n") == 0 &&
          static_cast<uint8_t>(bytes[11]) == 10 && static_cast<uint8_t>(bytes[22]) == 120);
    CHECK(!FgDumpWritePpm(path, 3, 2, rgb));  // size does not match the pixels
}

}  // namespace

int main() {
    int tests = 0;
    for (const TestCase& t : registry()) {
        const int before = g_failed;
        t.fn();
        ++tests;
        std::printf("%s %s\n", g_failed == before ? "ok  " : "FAIL", t.name);
    }
    std::printf("fg_runtime: %d tests, %d checks, %d failed\n", tests, g_checks, g_failed);
    return g_failed ? 1 : 0;
}
