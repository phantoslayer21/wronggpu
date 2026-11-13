// SPDX-License-Identifier: MIT
// Unit tests for the converter's building blocks. Written for this project.
// Hash test vectors: FIPS 180-4 examples (NIST CSRC "Example Values") and the FNV reference values
// for FNV-1a 64. Numeric tests compare against the hardware's binary32 conversion (round to nearest
// even) and against hand-derived values; the exhaustive comparison against an independent exact
// reference is done end to end by tests/run_tests.py.
//
//   wgc-unit-tests [--work <dir>]         all unit tests; with --work also the tests that write files (the library
//                                         interface the installer uses), inside <dir>
//   wgc-unit-tests --lz4-vectors <file>   only 08-T7: the decompression vectors of data/lz4-block-test.json
//   wgc-unit-tests --t5-vectors <file>    only 57-T5: the vectors of data/57-T5-vectors.json (map version 5 ops)
//   wgc-unit-tests --verify-model <map> <model.wgcm>
//                                         prints what verify_model_file says (exit 0 valid, 1 refused)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <system_error>

#include "api.h"
#include "convert.h"
#include "fnv.h"
#include "json.h"
#include "locate.h"
#include "lz4block.h"
#include "map.h"
#include "numeric.h"
#include "ops5.h"
#include "output.h"
#include "sha256.h"

using namespace wgc;

static int g_failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            if (g_failures < 20)                                             \
                std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static std::string sha_hex(const std::string& s) { return to_hex(Sha256::hash(s.data(), s.size())); }

static void test_sha256() {
    CHECK(sha_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(sha_hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu") ==
          "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    // One million 'a', fed in uneven pieces to exercise the buffering.
    Sha256 s;
    const std::string chunk(997, 'a');
    std::size_t left = 1000000;
    while (left) {
        const std::size_t n = left < chunk.size() ? left : chunk.size();
        s.update(chunk.data(), n);
        left -= n;
    }
    CHECK(to_hex(s.finish()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // Every length around the padding boundary: incremental equals one-shot.
    std::string msg;
    for (int n = 0; n < 200; ++n) {
        Sha256 a;
        for (char c : msg) a.update(&c, 1);
        CHECK(a.finish() == Sha256::hash(msg.data(), msg.size()));
        msg.push_back(static_cast<char>('a' + n % 26));
    }
    Digest d;
    CHECK(parse_digest_hex(sha_hex("abc"), d) && to_hex(d) == sha_hex("abc"));
    CHECK(!parse_digest_hex("xyz", d));
}

static void test_fnv() {
    auto f = [](const char* s) { return fnv1a64(reinterpret_cast<const std::uint8_t*>(s), std::strlen(s)); };
    CHECK(f("") == 0xcbf29ce484222325ULL);
    CHECK(f("a") == 0xaf63dc4c8601ec8cULL);
    CHECK(f("foobar") == 0x85944171f73967e8ULL);
}

static void test_json() {
    Json j;
    std::string err;
    CHECK(parse_json("{\"a\": [1, -2, 3.5, true, null, \"x\\u00e9\\n\"], \"b\": {}}", j, err));
    CHECK(j.is_object() && j.find("a") && j.find("a")->items.size() == 6);
    std::int64_t v;
    CHECK(j.find("a")->items[1].as_int64(v) && v == -2);
    CHECK(!j.find("a")->items[2].as_int64(v));
    CHECK(j.find("a")->items[5].text == "x\xc3\xa9\n");
    CHECK(!parse_json("{\"a\":1,\"a\":2}", j, err));  // duplicate key
    CHECK(!parse_json("{\"a\":1,}", j, err));
    CHECK(!parse_json("[1 2]", j, err));
    CHECK(!parse_json("01", j, err));
    CHECK(!parse_json("{} x", j, err));
    CHECK(parse_json("\"\\ud83d\\ude00\"", j, err) && j.text == "\xf0\x9f\x98\x80");
    CHECK(parse_json("9223372036854775807", j, err) && j.as_int64(v) && v == INT64_MAX);
    CHECK(parse_json("9223372036854775808", j, err) && !j.as_int64(v));
    std::uint64_t h;
    CHECK(parse_hex_u64("0x521bbca1665bd046", 16, h) && h == 0x521bbca1665bd046ULL);
    CHECK(!parse_hex_u64("0x", 16, h));
    CHECK(!parse_hex_u64("0x123456789", 8, h));
}

static void test_numeric() {
    // value(h)
    CHECK(half_value(0x3C00) == 1.0);
    CHECK(half_value(0xC000) == -2.0);
    CHECK(half_value(0x7BFF) == 65504.0);
    CHECK(half_value(0x0001) == std::ldexp(1.0, -24));
    CHECK(half_value(0x0400) == std::ldexp(1.0, -14));
    CHECK(std::isinf(half_value(0x7C00)) && std::isnan(half_value(0x7E00)));
    CHECK(std::signbit(half_value(0x8000)));
    // rn16 round trip of every finite half, and ties.
    for (std::uint32_t h = 0; h < 65536; ++h) {
        if (!half_is_finite(static_cast<std::uint16_t>(h))) continue;
        CHECK(rn16(half_value(static_cast<std::uint16_t>(h))) == h);
    }
    CHECK(rn16(1.0 + std::ldexp(1.0, -11)) == 0x3C00);                        // tie -> even
    CHECK(rn16(1.0 + 3 * std::ldexp(1.0, -11)) == 0x3C02);                    // tie -> even (up)
    CHECK(rn16(65519.99) == 0x7BFF && rn16(65520.0) == 0x7C00);               // overflow boundary
    CHECK(rn16(std::ldexp(1.0, -25)) == 0x0000);                              // tie at 0 -> even
    CHECK(rn16(std::ldexp(3.0, -26)) == 0x0001);
    // rn32 against the hardware conversion (round to nearest even) on random doubles.
    std::mt19937_64 rng(12345);
    for (int i = 0; i < 2000000; ++i) {
        // Random 53-bit significand in [0.5, 1), exponents covering the binary32 range including
        // subnormals and overflow.
        const int e = static_cast<int>(rng() % 320) - 170;
        double x = std::ldexp(static_cast<double>(rng() >> 11) * 0x1p-53 + 0.5, e);
        if (rng() & 1) x = -x;
        const float hw = static_cast<float>(x);
        const float mine = rn32(x);
        CHECK(float_to_bits(hw) == float_to_bits(mine));
        if (float_to_bits(hw) != float_to_bits(mine)) break;
    }
    // Exact midpoints for rn32.
    CHECK(rn32(1.0 + std::ldexp(1.0, -24)) == 1.0f);
    CHECK(float_to_bits(rn32(1.0 + 3 * std::ldexp(1.0, -24))) == 0x3F800002u);
    CHECK(rn32(std::ldexp(1.0, -150)) == 0.0f);
    CHECK(float_to_bits(rn32(std::ldexp(3.0, -151))) == 1u);
    // q8
    CHECK(q8(1.0) == 1.0f && q8(-1.0) == -1.0f);
    CHECK(q8(1.0625) == 1.0f);    // tie between 1.0 (m=0) and 1.125 (m=1) -> 1.0
    CHECK(q8(1.1875) == 1.25f);   // tie between 1.125 (m=1) and 1.25 (m=2) -> 1.25
    CHECK(q8(1000.0) == 448.0f && q8(-1000.0) == -448.0f && q8(450.0) == 448.0f);
    CHECK(q8(440.0) == 448.0f && q8(431.0) == 416.0f);
    CHECK(q8(std::ldexp(1.0, -10)) == 0.0f && !std::signbit(q8(-std::ldexp(1.0, -10))));  // tie to 0, +0
    CHECK(q8(std::ldexp(3.0, -11)) == std::ldexp(1.0f, -9));  // 0.75 subnormal units -> 1 unit
    CHECK(q8(std::ldexp(3.0, -10)) == std::ldexp(2.0f, -9));  // 1.5 units -> 2 (even)
    CHECK(q8(std::ldexp(15.5, -9)) == std::ldexp(1.0f, -5));  // subnormal grid ends at 2^-6
    CHECK(!std::signbit(q8(-0.0)));
    CHECK(q8(1.9375) == 2.0f);  // tie between 1.875 (m=7) and 2.0 (m=0) -> 2.0
    // f16_scaled with mul = div = 8: for |x| < 8192 every step is exact -> identity.
    for (std::uint32_t h = 0; h < 0x7000; ++h)
        CHECK(f16_scaled(half_value(static_cast<std::uint16_t>(h)), 8.0f, 8.0f) == static_cast<float>(half_value(static_cast<std::uint16_t>(h))));
    // f16_scaled against binary32 hardware arithmetic for mul = sqrt(8), div = 1 and sqrt(96).
    const float m1 = bits_to_float(0x403504f3u), m2 = bits_to_float(0x411cc471u);
    for (std::uint32_t h = 0; h < 0x7C00; ++h) {
        const double x = half_value(static_cast<std::uint16_t>(h));
        volatile float xf = static_cast<float>(x);
        volatile float p = xf * m2;
        const float hv = static_cast<float>(half_value(rn16(p)));
        volatile float q = hv / m2;
        CHECK(float_to_bits(f16_scaled(x, m2, m2)) == float_to_bits(q));
        volatile float p1 = xf * m1;
        CHECK(float_to_bits(f16_scaled(x, m1, 1.0f)) == float_to_bits(static_cast<float>(half_value(rn16(p1)))));
    }
    // exp: exact points and agreement with binary64 exp where that is far from a rounding boundary.
    CHECK(exp_rn32(0.0) == 1.0f);
    CHECK(std::isinf(exp_rn32(89.0)) && exp_rn32(-110.0) == 0.0f);
    CHECK(float_to_bits(exp_rn32(88.6875)) == float_to_bits(static_cast<float>(std::exp(88.6875))));
    for (std::uint32_t h = 0; h < 65536; ++h) {
        if (!half_is_finite(static_cast<std::uint16_t>(h))) continue;
        const double x = half_value(static_cast<std::uint16_t>(h));
        if (x > 88.7 || x < -103.0) continue;
        const double d = std::exp(x);
        const float f = static_cast<float>(d);
        const float mine = exp_rn32(x);
        if (float_to_bits(mine) != float_to_bits(f)) {
            // Only allowed when binary64 exp is within a few ulps of a binary32 midpoint.
            const float g = std::nextafter(f, mine);
            const double mid = (static_cast<double>(f) + static_cast<double>(g)) / 2;
            CHECK(std::fabs(d - mid) <= 4 * std::fabs(d) * 0x1p-52);
        }
    }
}

static void test_key_and_digest() {
    // make_key = SHA-256(label || bytes)
    const std::vector<std::uint8_t> w = {1, 2, 3};
    const Digest k = make_key("L", w);
    const std::string cat = std::string("L") + "\x01\x02\x03";
    CHECK(k == Sha256::hash(cat.data(), cat.size()));
    const Digest kd = keyed_sha256(k, "xy", 2);
    std::string kc(reinterpret_cast<const char*>(k.data()), 32);
    kc += "xy";
    CHECK(kd == Sha256::hash(kc.data(), kc.size()));
}

// RFC 4648 section 10 test vectors and the forms spec 48 section 2 excludes.
static void test_base64() {
    auto dec = [](const std::string& t, std::string& out) {
        std::vector<std::uint8_t> v;
        const bool ok = decode_base64(reinterpret_cast<const std::uint8_t*>(t.data()), t.size(), v);
        out.assign(v.begin(), v.end());
        return ok;
    };
    std::string o;
    CHECK(dec("", o) && o.empty());
    CHECK(dec("Zg==", o) && o == "f");
    CHECK(dec("Zm8=", o) && o == "fo");
    CHECK(dec("Zm9v", o) && o == "foo");
    CHECK(dec("Zm9vYg==", o) && o == "foob");
    CHECK(dec("Zm9vYmE=", o) && o == "fooba");
    CHECK(dec("Zm9vYmFy", o) && o == "foobar");
    CHECK(dec("+/+/", o) && o == "\xfb\xff\xbf");
    CHECK(!dec("Zm9", o));         // not a multiple of 4
    CHECK(!dec("Zg==Zm9v", o));    // padding before the end
    CHECK(!dec("Z===", o));        // three '='
    CHECK(!dec("Zm=v", o));        // '=' not at the end of the quad
    CHECK(!dec("Zm9-", o));        // URL-safe alphabet is not RFC 4648 section 4
    CHECK(!dec("Zm9\n", o));
    CHECK(!dec("Zm 9", o));
}

// capped_softmax with cap 15, div 675, sub 3.5 (the preset-K constants) on 64 equal inputs: all y are equal and
// within the FP16 rounding of 1/64 (exactly 1/64 for these inputs).
static void test_capped_softmax() {
    double x[64];
    float y[64];
    for (double v : {0.0, 1.0, -3.0, 15.0, 65504.0, -65504.0}) {
        for (double& xi : x) xi = v;
        capped_softmax64(x, 15.0f, 675.0f, 3.5f, y);
        bool all = std::fabs(y[0] - 0.015625f) <= 0.015625f / 1024;
        for (float yi : y) all = all && yi == y[0];
        CHECK(all);
    }
    // One large input among zeros: it gets almost everything, and the result is FP16.
    for (double& xi : x) xi = 0.0;
    x[5] = 15.0;
    capped_softmax64(x, 15.0f, 675.0f, 3.5f, y);
    CHECK(y[5] > 0.9f && y[5] < 1.0f && y[0] == y[63] && y[0] > 0.0f);
    CHECK(static_cast<float>(half_value(rn16(y[5]))) == y[5]);
    // A zero sum (sub so large that every exp underflows) gives a non-finite result for the caller to reject.
    capped_softmax64(x, 15.0f, 675.0f, 400.0f, y);
    CHECK(!std::isfinite(y[0]));
}

// ---------------------------------------------------------------- spec 08 section 4.4: block decompression

typedef std::vector<std::uint8_t> Bytes;

static bool lz4(const Bytes& block, std::size_t decoded_length, Bytes& out) {
    // An exactly sized copy on the heap, so that a tool that watches memory sees any read past the block.
    const std::unique_ptr<std::uint8_t[]> copy(new std::uint8_t[block.size() ? block.size() : 1]);
    if (!block.empty()) std::memcpy(copy.get(), block.data(), block.size());
    const bool ok = lz4_block_decode(copy.get(), block.size(), decoded_length, out);
    CHECK(out.size() <= decoded_length);          // never more than decoded_length bytes
    CHECK(!ok || out.size() == decoded_length);   // success means exactly decoded_length bytes
    return ok;
}

static std::string text_of(const Bytes& v) { return std::string(v.begin(), v.end()); }

// Test-side encoder: the whole text as one literal-only sequence (steps 1 to 4 of section 4.4).
static Bytes lz4_literals(const std::string& text) {
    Bytes b;
    const std::size_t n = text.size();
    b.push_back(static_cast<std::uint8_t>((n < 15 ? n : 15) << 4));
    if (n >= 15) {
        std::size_t v = n - 15;
        for (; v >= 255; v -= 255) b.push_back(255);
        b.push_back(static_cast<std::uint8_t>(v));
    }
    b.insert(b.end(), text.begin(), text.end());
    return b;
}

// Blocks written by hand for this test from the seven steps of section 4.4.
static void test_lz4_block() {
    Bytes out;
    CHECK(lz4({}, 0, out) && out.empty());
    CHECK(!lz4({}, 1, out));
    CHECK(lz4({0x30, 'a', 'b', 'c'}, 3, out) && text_of(out) == "abc");
    CHECK(lz4({0x00}, 0, out) && out.empty());  // an empty last sequence
    // One literal, then a match of length 2 + 4 at distance 1: the byte repeats.
    CHECK(lz4({0x12, 'x', 1, 0}, 7, out) && text_of(out) == "xxxxxxx");
    // "ab", a match of length 4 at distance 2, then a last sequence with one literal.
    CHECK(lz4({0x20, 'a', 'b', 2, 0, 0x10, '!'}, 7, out) && text_of(out) == "ababab!");
    // A sequence without literals in the middle: "abcd", match 4 at distance 4, match 4 at distance 8, "."
    CHECK(lz4({0x40, 'a', 'b', 'c', 'd', 4, 0, 0x00, 8, 0, 0x10, '.'}, 13, out) && text_of(out) == "abcdabcdabcd.");
    // Length extensions: 15 literals (extension byte 0), 270 literals (255, 0), 525 literals (255, 255, 0).
    for (std::size_t n : {std::size_t(14), std::size_t(15), std::size_t(16), std::size_t(269), std::size_t(270),
                          std::size_t(271), std::size_t(525), std::size_t(5000)}) {
        std::string t;
        for (std::size_t i = 0; i < n; ++i) t.push_back(static_cast<char>('a' + i % 23));
        const Bytes b = lz4_literals(t);
        CHECK(lz4(b, n, out) && text_of(out) == t);
        CHECK(!lz4(b, n - 1, out) && !lz4(b, n + 1, out));
        Bytes cut(b.begin(), b.end() - 1);
        CHECK(!lz4(cut, n, out) && !lz4(cut, n - 1, out));  // the input ends inside the literals
    }
    // A match of length 15 + 4 + 255 + 3 = 277 at distance 3 (extension bytes 255, 3), then "end".
    {
        const Bytes b = {0x3F, 'x', 'y', 'z', 3, 0, 255, 3, 0x30, 'e', 'n', 'd'};
        std::string want = "xyz";
        for (int i = 0; i < 277; ++i) want.push_back(want[want.size() - 3]);
        want += "end";
        CHECK(lz4(b, want.size(), out) && text_of(out) == want);
        CHECK(!lz4(b, want.size() - 1, out) && !lz4(b, want.size() + 1, out) && !lz4(b, 5, out));
    }
    // Malformed blocks.
    CHECK(!lz4({0x20, 'a', 'b', 0, 0}, 6, out));           // distance 0
    CHECK(!lz4({0x20, 'a', 'b', 3, 0}, 6, out));           // distance beyond the output
    CHECK(!lz4({0x00, 1, 0}, 4, out));                     // a match with no output yet
    CHECK(!lz4({0x20, 'a'}, 2, out));                      // ends inside the literals
    CHECK(!lz4({0x20, 'a', 'b', 2}, 6, out));              // ends inside the distance
    CHECK(!lz4({0x2F, 'a', 'b', 2, 0}, 300, out));         // ends inside the match-length extension
    CHECK(!lz4({0x2F, 'a', 'b', 2, 0, 255}, 300, out));    // ... after a 255
    CHECK(!lz4({0xF0}, 15, out));                          // ends inside the literal-length extension
    CHECK(!lz4({0xF0, 255, 255}, 600, out));
    CHECK(!lz4({0x12, 'x', 1, 0}, 6, out));                // output too long (inside a match)
    CHECK(!lz4({0x12, 'x', 1, 0}, 8, out));                // output too short
    CHECK(!lz4({0x30, 'a', 'b', 'c'}, 2, out));            // output too long (inside the literals)
    // Lengths far beyond anything real: a long run of 255 must be refused without overflow or a large buffer.
    {
        Bytes big(200000, 255);
        big[0] = 0xF0;
        CHECK(!lz4(big, 1000, out));
        Bytes m = {0x1F, 'x', 1, 0};
        m.insert(m.end(), 200000, 255);
        m.push_back(0);
        CHECK(!lz4(m, 1000, out));
        CHECK(lz4(m, 1 + 19 + 200000 * 255, out) && out.size() == 1 + 19 + 200000u * 255u && out.front() == 'x' && out.back() == 'x');
    }
    // Damaged and random blocks never write more than decoded_length bytes and never crash; a block that still
    // decodes gives exactly decoded_length bytes (checked inside lz4()).
    const Bytes good = {0x8F, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 8, 0, 81, 0x30, 'e', 'n', 'd'};  // 8 + 100 + 3
    CHECK(lz4(good, 111, out) && out.size() == 111 && out[107] == 'd' && out[110] == 'd');
    for (std::size_t cut = 0; cut < good.size(); ++cut) {
        const Bytes prefix(good.begin(), good.begin() + static_cast<std::ptrdiff_t>(cut));
        CHECK(!lz4(prefix, 111, out));
    }
    std::mt19937 rng(2468);
    int still_ok = 0;
    for (int i = 0; i < 300000; ++i) {
        Bytes b;
        if (i % 3 == 0) {
            b.resize(rng() % 40);
            for (auto& x : b) x = static_cast<std::uint8_t>(rng());
        } else {
            b = good;
            for (unsigned k = 0, n = 1 + rng() % 3; k < n; ++k) b[rng() % b.size()] = static_cast<std::uint8_t>(rng());
            if (i % 3 == 2) b.resize(rng() % (b.size() + 1));
        }
        const std::size_t want = (rng() & 1) ? 111 : rng() % 400;
        still_ok += lz4(b, want, out) ? 1 : 0;
    }
    CHECK(still_ok > 0 && still_ok < 300000);
}

// Spec 08 section 4.3: decoding a located block and reading its hexadecimal fields.
static void test_variant_block() {
    const Digest key = Sha256::hash("unit key", 8);
    const std::string text = "first = 3F800000; second = bf000000;\nthird = 0000000A; fourth = fFfFfFfF.\n";
    Variant v;
    const Bytes block = lz4_literals(text);
    v.length = block.size();
    v.decoded_length = text.size();
    v.decoded_sha = keyed_sha256(key, text.data(), text.size());
    v.fields = {text.find("0000000A"), text.find("3F800000"), text.find("bf000000"), text.find("fFfFfFfF")};
    Bytes el;
    std::string why;
    const Bytes want = {0x0A, 0, 0, 0, 0, 0, 0x80, 0x3F, 0, 0, 0, 0xBF, 0xFF, 0xFF, 0xFF, 0xFF};
    CHECK(decode_variant_block(v, block, key, el, why) && el == want);
    Variant bad = v;
    bad.decoded_sha[5] ^= 1;
    CHECK(!decode_variant_block(bad, block, key, el, why) && el.empty() && why.find("decoded_sha256") != std::string::npos);
    CHECK(!decode_variant_block(v, block, Sha256::hash("other key", 9), el, why) && el.empty());  // wrong key
    for (std::uint64_t off : {v.fields[1] - 1, v.fields[1] + 1, v.fields[3] + 1, std::uint64_t(0)}) {
        bad = v;
        bad.fields[2] = off;  // eight bytes that are not all hexadecimal digits
        CHECK(!decode_variant_block(bad, block, key, el, why) && el.empty() && why.find("field 2") != std::string::npos);
    }
    bad = v;
    bad.fields[0] = text.size() - 7;  // would read past the end (load_map refuses such a map)
    CHECK(!decode_variant_block(bad, block, key, el, why) && el.empty());
    bad = v;
    bad.decoded_length += 1;
    CHECK(!decode_variant_block(bad, block, key, el, why) && el.empty());
    bad.decoded_length -= 2;
    CHECK(!decode_variant_block(bad, block, key, el, why) && el.empty());
    Bytes cut(block.begin(), block.end() - 1);
    CHECK(!decode_variant_block(v, cut, key, el, why) && el.empty());
    // The failure text names the step, never content.
    CHECK(why.find("3F8") == std::string::npos && why.find("first") == std::string::npos);
}

// 08-T7: data/lz4-block-test.json. Returns the number of cases run, or -1 if the file cannot be used.
static int run_lz4_vectors(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("cannot read %s\n", path);
        return -1;
    }
    const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Json root;
    std::string err;
    const Json* cases = nullptr;
    if (!parse_json(bytes, root, err) || !(cases = root.find("cases")) || !cases->is_array()) {
        std::printf("%s is not the vector file: %s\n", path, err.c_str());
        return -1;
    }
    auto unhex = [](const std::string& s, Bytes& out) {
        out.clear();
        if (s.size() % 2) return false;
        for (std::size_t i = 0; i < s.size(); i += 2) {
            std::uint64_t v;
            if (!parse_hex_u64(s.substr(i, 2), 2, v)) return false;
            out.push_back(static_cast<std::uint8_t>(v));
        }
        return true;
    };
    int ok_cases = 0, error_cases = 0;
    for (const Json& c : cases->items) {
        const Json *name = c.find("name"), *block = c.find("block"), *len = c.find("decoded_length"), *result = c.find("result");
        std::int64_t n = -1;
        Bytes in, out;
        if (!name || !block || !len || !result || !block->is_string() || !result->is_string() || !len->as_int64(n) || n < 0 ||
            !unhex(block->text, in)) {
            std::printf("FAIL: a case of %s cannot be read\n", path);
            ++g_failures;
            continue;
        }
        const bool got = lz4(in, static_cast<std::size_t>(n), out);
        bool pass;
        if (result->text == "ok") {
            ++ok_cases;
            const Json *sha = c.find("decoded_sha256"), *decoded = c.find("decoded");
            pass = got && sha && sha->is_string() && to_hex(Sha256::hash(out.data(), out.size())) == sha->text;
            if (pass && decoded && decoded->is_string()) {
                Bytes want;
                pass = unhex(decoded->text, want) && want == out;
            }
        } else if (result->text == "error") {
            ++error_cases;
            pass = !got;
        } else {
            pass = false;
        }
        if (!pass) {
            std::printf("FAIL 08-T7: %s (expected %s)\n", name->text.c_str(), result->text.c_str());
            ++g_failures;
        }
    }
    std::printf("08-T7: %d decompression cases (%d ok, %d malformed)\n", ok_cases + error_cases, ok_cases, error_cases);
    if (ok_cases == 0 || error_cases == 0) {
        std::printf("FAIL 08-T7: the vector file has no cases of one kind\n");
        ++g_failures;
    }
    return ok_cases + error_cases;
}

// ---------------------------------------------------------------- the library interface (api.h) with source F
//
// A small synthetic file and version-4 map made here: W, D, K and one of two F blocks (or none). Covers
// convert_file, write_converted and verify_model_file, which the installer uses.

namespace fs = std::filesystem;

struct UnitModel {
    Bytes w, d, k, file;
    Bytes block[2];
    std::string text[2];
    std::vector<std::uint64_t> fields[2];
    Bytes taps[2];  // the expected output_taps bytes per variant
    Digest key{};
    std::string map_v4, map_v1;
};

static std::string hex64(std::uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(v));
    return buf;
}

static std::string block_source(const char* what, const char* element, const Bytes& data, const Digest* key,
                                const char* extra = "") {
    const std::size_t n = data.size() < 64 ? data.size() : 64;
    const std::uint64_t fnv = fnv1a64(data.data(), n, fnv_initial_state(key != nullptr, key));
    const Digest sha = key ? keyed_sha256(*key, data.data(), data.size()) : Sha256::hash(data.data(), data.size());
    return std::string("{\"what\":\"") + what + "\",\"element\":\"" + element + "\",\"length\":" + std::to_string(data.size()) +
           "," + extra + (key ? "\"keyed\":true," : "") + "\"fnv1a64\":\"" + hex64(fnv) + "\",\"sha256\":\"" + to_hex(sha) + "\"}";
}

static UnitModel make_unit_model() {
    UnitModel u;
    for (int i = 0; i < 128; ++i) {  // finite binary16 values around 1
        const std::uint16_t h = static_cast<std::uint16_t>(0x3C00 + 7 * i);
        u.w.push_back(static_cast<std::uint8_t>(h));
        u.w.push_back(static_cast<std::uint8_t>(h >> 8));
    }
    for (int i = 0; i < 64; ++i) u.d.push_back(static_cast<std::uint8_t>(37 * i + 11));
    u.k = {0x00, 0x00, 0x40, 0x3F, 0x00, 0x00, 0xC0, 0xBF};  // 0.75, -1.5
    u.key = make_key("unit label", u.w);
    // Our own text with five fields each; the map lists them in another order than they appear.
    u.text[0] = "# synthetic filter text, variant one\nrow a: 3E000000 3F000000 3E800000\nrow b: bd800000, BD000000;\n";
    u.text[1] = "# synthetic filter text, variant two\nvalues = 3DCCCCCD 3E4CCCCD 3F19999A 3e4ccccd 3dcccccd (end)\n";
    const char* order[2][5] = {{"bd800000", "3E000000", "3F000000", "3E800000", "BD000000"},
                               {"3DCCCCCD", "3E4CCCCD", "3F19999A", "3e4ccccd", "3dcccccd"}};
    for (int v = 0; v < 2; ++v) {
        u.block[v] = lz4_literals(u.text[v]);
        for (const char* f : order[v]) {
            u.fields[v].push_back(u.text[v].find(f));
            std::uint64_t x = 0;
            parse_hex_u64(f, 8, x);
            for (int i = 0; i < 4; ++i) u.taps[v].push_back(static_cast<std::uint8_t>(x >> (8 * i)));
        }
    }
    // The file: fixed pseudo-random bytes with the sources placed in it (the F block is added by the caller).
    std::mt19937 rng(97531);
    u.file.resize(8192);
    for (auto& b : u.file) b = static_cast<std::uint8_t>(rng());
    std::copy(u.w.begin(), u.w.end(), u.file.begin() + 64);
    std::copy(u.d.begin(), u.d.end(), u.file.begin() + 512);
    std::copy(u.k.begin(), u.k.begin() + 4, u.file.begin() + 1024);
    std::copy(u.k.begin() + 4, u.k.end(), u.file.begin() + 1024 + 8);

    Bytes enc0;  // tensor b: f16 of the first four elements of W
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t bits = float_to_bits(static_cast<float>(half_value(static_cast<std::uint16_t>(0x3C00 + 7 * i))));
        for (int j = 0; j < 4; ++j) enc0.push_back(static_cast<std::uint8_t>(bits >> (8 * j)));
    }
    const std::string kfnv = hex64(fnv1a64(u.k.data(), 8, fnv_initial_state(true, &u.key)));
    const std::string head = ",\"model\":\"unit test\",\"channel_order\":\"natural\",\"sources\":{\"W\":" +
                             block_source("w", "f16le", u.w, nullptr, "\"locate\":\"block\",") +
                             ",\"D\":" + block_source("d", "u16le", u.d, nullptr, "\"locate\":\"block\",") +
                             ",\"K\":{\"what\":\"k\",\"element\":\"u32le\",\"length\":8,\"locate\":\"pair\",\"gaps\":[8],\"keyed\":true,"
                             "\"fnv1a64\":\"" + kfnv + "\",\"sha256\":\"" + to_hex(keyed_sha256(u.key, u.k.data(), 8)) + "\"}";
    std::string f = ",\"F\":{\"what\":\"f\",\"element\":\"hex32\",\"locate\":\"block\",\"keyed\":true,\"encoding\":\"lz4-block\","
                    "\"optional\":true,\"variants\":[";
    for (int v = 0; v < 2; ++v) {
        const Bytes& b = u.block[v];
        f += std::string(v ? "," : "") + "{\"length\":" + std::to_string(b.size()) + ",\"fnv1a64\":\"" +
             hex64(fnv1a64(b.data(), b.size() < 64 ? b.size() : 64, fnv_initial_state(true, &u.key))) + "\",\"sha256\":\"" +
             to_hex(keyed_sha256(u.key, b.data(), b.size())) + "\",\"decoded_length\":" + std::to_string(u.text[v].size()) +
             ",\"decoded_sha256\":\"" + to_hex(keyed_sha256(u.key, u.text[v].data(), u.text[v].size())) + "\",\"fields\":[";
        for (std::size_t i = 0; i < 5; ++i) f += (i ? "," : "") + std::to_string(u.fields[v][i]);
        f += "]}";
    }
    f += "]}";
    const std::string tail = "},\"key\":{\"label\":\"unit label\",\"from\":\"W\"},\"perms\":{},\"stages\":[{\"stage\":\"enc0\",\"digest\":\"" +
                             to_hex(keyed_sha256(u.key, enc0.data(), enc0.size())) +
                             "\",\"tensors\":[{\"name\":\"b\",\"shape\":[4],\"dtype\":\"float32\",\"src\":\"W\",\"op\":{\"name\":\"f16\"},"
                             "\"base\":0,\"axes\":[{\"stride\":1}]}]}";
    const std::string filter = ",{\"stage\":\"filter\",\"digest\":[\"" + to_hex(keyed_sha256(u.key, u.taps[0].data(), 20)) + "\",\"" +
                               to_hex(keyed_sha256(u.key, u.taps[1].data(), 20)) +
                               "\"],\"tensors\":[{\"name\":\"output_taps\",\"shape\":[5],\"dtype\":\"float32\",\"src\":\"F\","
                               "\"op\":{\"name\":\"u32_as_f32\"},\"base\":0,\"axes\":[{\"stride\":1}]}]}";
    u.map_v4 = "{\"format\":\"wronggpu-converter-map\",\"version\":4" + head + f + tail + filter + "]}";
    u.map_v1 = "{\"format\":\"wronggpu-converter-map\",\"version\":1" + head + tail + "]}";
    return u;
}

static bool write_bytes(const fs::path& p, const Bytes& b) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    return static_cast<bool>(f);
}

static void test_api_files(const fs::path& work) {
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work, ec);
    CHECK(fs::is_directory(work, ec));
    const UnitModel u = make_unit_model();
    std::vector<std::string> log;
    const LogSink sink = [&](const std::string& s) { log.push_back(s); };
    auto logged = [&](const std::string& what) {
        for (const std::string& s : log)
            if (s.find(what) != std::string::npos) return true;
        return false;
    };
    LoadedMap m4, m1;
    CHECK(load_map_bytes(u.map_v4, m4, sink).code == kOk);
    CHECK(load_map_bytes(u.map_v1, m1, sink).code == kOk);
    if (!m4.map || !m1.map) return;
    CHECK(m4.map->version == 4 && m4.map->preset() == 'M' && m1.map->preset() == 'M');
    // The preset by name: a map without the field gets its version's preset as before; with it, the name must be known
    // and allow the map's version (map.h known_presets)
    CHECK(m4.map->preset_name == "M" && m1.map->preset_name == "M" && m1.map->preset_info() == find_preset("M"));
    CHECK(preset_for_version(1) == "M" && preset_for_version(4) == "M" && preset_for_version(2) == "K" && preset_for_version(3) == "K" &&
          preset_for_version(5) == "F" && preset_for_version(0).empty() && preset_for_version(6).empty());
    CHECK(find_preset("M") && find_preset("K") && find_preset("F") && !find_preset("m") && !find_preset("") && !find_preset("X") &&
          known_presets().size() == 3);
    CHECK(std::string(find_preset("M")->no_model_message) == kMsgNoPresetM && std::string(find_preset("K")->no_model_message) == kMsgNoPresetK &&
          std::string(find_preset("F")->no_model_message) == kMsgNoPresetF);
    {
        auto with_preset = [&](const std::string& map, const std::string& version, const std::string& field) {
            std::string t = map;
            const std::string v = "\"version\":" + version;
            const std::size_t at = t.find(v);
            if (at != std::string::npos) t.insert(at + v.size(), ",\"preset\":" + field);
            return t;
        };
        LoadedMap pm1, pm4;
        CHECK(load_map_bytes(with_preset(u.map_v1, "1", "\"M\""), pm1, sink).code == kOk && pm1.map && pm1.map->preset_name == "M" &&
              pm1.map->preset() == 'M');
        CHECK(load_map_bytes(with_preset(u.map_v4, "4", "\"M\""), pm4, sink).code == kOk && pm4.map && pm4.map->preset_info() == find_preset("M"));
        // the same map bytes with and without "preset":"M" describe the same tensors (only the file hash differs)
        CHECK(pm4.map && pm4.map->stages.size() == m4.map->stages.size() && pm4.map->sources.size() == m4.map->sources.size());
        for (const char* bad : {"\"K\"", "\"F\"", "\"X\"", "\"m\"", "\"\"", "7", "null"}) {
            LoadedMap bm;
            const Result r = load_map_bytes(with_preset(u.map_v1, "1", bad), bm, sink);
            if (r.code != kMapDamaged) std::printf("preset field %s\n", bad);
            CHECK(r.code == kMapDamaged && !bm.map);
        }
        CHECK(logged("unknown preset \"X\"") && logged("preset K does not use map version 1") && logged("field \"preset\" must be text"));
    }
    const int fi = m4.map->find_source("F");
    CHECK(fi >= 0 && m4.map->sources[static_cast<std::size_t>(fi)].variants.size() == 2 && m4.map->stages.size() == 2);
    CHECK(m4.map->stages[1].optional() && m4.map->stages[1].variant_source == fi && !m4.map->stages[0].optional());
    if (fi < 0) return;
    SearchOptions opt;
    opt.threads = 2;
    std::string why;
    ModelFileInfo info;

    // Variant 1, variant 2, and both blocks in one file (variant 1 wins although its block comes later).
    for (int c = 0; c < 3; ++c) {
        Bytes file = u.file;
        const int want = c == 1 ? 1 : 0;
        if (c == 2) std::copy(u.block[1].begin(), u.block[1].end(), file.begin() + 2048);
        const std::size_t at = c == 2 ? 4099 : 2048 + 5 * static_cast<std::size_t>(c);
        std::copy(u.block[want].begin(), u.block[want].end(), file.begin() + static_cast<std::ptrdiff_t>(at));
        const fs::path dll = work / ("with-f-" + std::to_string(c) + ".bin"), out = work / ("out-" + std::to_string(c));
        CHECK(write_bytes(dll, file));
        Converted conv;
        log.clear();
        const Result r = convert_file(*m4.map, dll, opt, conv, sink);
        CHECK(r.code == kOk && r.message == "The model was converted successfully." && conv.notes.empty());
        CHECK(has_every_group(*m4.map, conv));  // several DLLs: such a file is preferred (spec 08 section 8)
        CHECK(conv.variants.size() == m4.map->sources.size() && conv.variants[static_cast<std::size_t>(fi)] == want);
        CHECK(conv.stages.size() == 2);
        if (conv.stages.size() == 2) {
            CHECK(conv.stages[1].stage->name == "filter" && conv.stages[1].variant == want);
            CHECK(conv.stages[1].tensors.size() == 1 && conv.stages[1].tensors[0].bytes == u.taps[want]);
            CHECK(conv.stages[1].digest == m4.map->stages[1].variant_digests[static_cast<std::size_t>(want)]);
        }
        CHECK(logged("source F at offset " + std::to_string(at) + " (variant " + std::to_string(want + 1) + " of 2)"));
        CHECK(logged("filter: variant " + std::to_string(want + 1)));
        for (const std::string& s : log) CHECK(s.find("3F000000") == std::string::npos && s.find("synthetic filter") == std::string::npos);
        CHECK(write_converted(*m4.map, conv, out, true, sink).code == kOk);
        CHECK(verify_model_file(*m4.map, out / kModelFileName, why, &info) && info.absent_groups.empty() && info.same_map);
        CHECK(verify_model_file(*m4.map, out / kModelFileName, why));
        CHECK(fs::file_size(out / "raw" / "filter" / "output_taps.bin", ec) == 20);
        // A map without the group does not accept a file that has it.
        CHECK(!verify_model_file(*m1.map, out / kModelFileName, why));
    }

    // No block: the conversion succeeds without the group and says so.
    {
        const fs::path dll = work / "without-f.bin", out = work / "out-none";
        CHECK(write_bytes(dll, u.file));
        Converted conv;
        log.clear();
        const Result r = convert_file(*m4.map, dll, opt, conv, sink);
        CHECK(r.code == kOk && conv.stages.size() == 1 && conv.variants[static_cast<std::size_t>(fi)] == -1);
        CHECK(conv.notes.size() == 1 && conv.notes[0] == kMsgFilterUnknown);
        CHECK(!has_every_group(*m4.map, conv));
        CHECK(r.message.find("converted successfully") != std::string::npos && r.message.find(kMsgFilterUnknown) != std::string::npos);
        CHECK(logged("source F not found") && logged("no group filter") && logged(kMsgFilterUnknown));
        CHECK(write_converted(*m4.map, conv, out, true, sink).code == kOk);
        CHECK(!fs::exists(out / "raw" / "filter", ec) && fs::exists(out / "raw" / "enc0" / "b.bin", ec));
        CHECK(verify_model_file(*m4.map, out / kModelFileName, why, &info) && info.same_map &&
              info.absent_groups.size() == 1 && info.absent_groups[0] == "filter");
        CHECK(verify_model_file(*m4.map, out / kModelFileName, why));
    }
    // A model file written with the version-1 map (before the group existed) still counts as valid for the
    // version-4 map, with the group absent; and the version-1 map converts as before.
    {
        const fs::path dll = work / "with-f-0.bin", out = work / "out-v1";
        Converted conv;
        const Result r = convert_file(*m1.map, dll, opt, conv, sink);
        CHECK(r.code == kOk && conv.stages.size() == 1 && conv.notes.empty());
        CHECK(has_every_group(*m1.map, conv));  // a map without an optional group: every file that converts has them all
        CHECK(write_converted(*m1.map, conv, out, false, sink).code == kOk);
        CHECK(verify_model_file(*m1.map, out / kModelFileName, why, &info) && info.absent_groups.empty() && info.same_map);
        CHECK(verify_model_file(*m4.map, out / kModelFileName, why, &info) && info.absent_groups.size() == 1 &&
              !info.same_map && info.map_sha256 == m1.map->file_sha256);
        // A damaged copy is refused (whole-file hash), and so is one whose filter digest is no variant's.
        Bytes bytes;
        {
            std::ifstream f(work / "out-0" / kModelFileName, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        CHECK(bytes.size() > 200);
        if (bytes.size() > 200) {
            Bytes damaged = bytes;
            damaged[bytes.size() - 40] ^= 1;
            CHECK(write_bytes(work / "damaged.wgcm", damaged) && !verify_model_file(*m4.map, work / "damaged.wgcm", why));
            // The stored digest of the last stage (filter) changed and the trailer recomputed: no variant's digest.
            const Digest& fd = m4.map->stages[1].variant_digests[0];
            auto at = std::search(bytes.begin(), bytes.end() - 32, fd.begin(), fd.end());
            CHECK(at != bytes.end() - 32);
            if (at != bytes.end() - 32) {
                damaged = bytes;
                damaged[static_cast<std::size_t>(at - bytes.begin())] ^= 1;
                const Digest whole = Sha256::hash(damaged.data(), damaged.size() - 32);
                std::copy(whole.begin(), whole.end(), damaged.end() - 32);
                CHECK(write_bytes(work / "other-digest.wgcm", damaged) &&
                      !verify_model_file(*m4.map, work / "other-digest.wgcm", why) && why.find("filter") != std::string::npos);
                // The digest of the other variant with this variant's data: the data does not match its digest.
                damaged = bytes;
                const Digest& od = m4.map->stages[1].variant_digests[1];
                std::copy(od.begin(), od.end(), damaged.begin() + (at - bytes.begin()));
                const Digest whole2 = Sha256::hash(damaged.data(), damaged.size() - 32);
                std::copy(whole2.begin(), whole2.end(), damaged.end() - 32);
                CHECK(write_bytes(work / "swapped-digest.wgcm", damaged) &&
                      !verify_model_file(*m4.map, work / "swapped-digest.wgcm", why) && why.find("filter") != std::string::npos);
            }
        }
    }
    // Damaged copies of the block: no variant matches, so the group is left out (not an error).
    for (int c = 0; c < 3; ++c) {
        Bytes file = u.file;
        Bytes b = u.block[0];
        if (c == 0) b[b.size() / 2] ^= 0x10;
        if (c == 1) b.back() ^= 0x01;
        std::copy(b.begin(), b.end(), file.end() - static_cast<std::ptrdiff_t>(b.size()));
        if (c == 2) file.pop_back();  // the file ends one byte before the block does
        const fs::path dll = work / ("damaged-f-" + std::to_string(c) + ".bin");
        CHECK(write_bytes(dll, file));
        Converted conv;
        const Result r = convert_file(*m4.map, dll, opt, conv, sink);
        CHECK(r.code == kOk && conv.stages.size() == 1 && conv.notes.size() == 1);
    }
    // The undamaged block at the very end of the file is found.
    {
        Bytes file = u.file;
        std::copy(u.block[1].begin(), u.block[1].end(), file.end() - static_cast<std::ptrdiff_t>(u.block[1].size()));
        const fs::path dll = work / "f-at-end.bin";
        CHECK(write_bytes(dll, file));
        Converted conv;
        CHECK(convert_file(*m4.map, dll, opt, conv, sink).code == kOk && conv.stages.size() == 2);
    }
    // Maps that do not fit the block they describe: installer errors, nothing converted.
    auto replaced = [&](const std::string& from, const std::string& to) {
        std::string s = u.map_v4;
        const std::size_t p = s.find(from);
        CHECK(p != std::string::npos);
        if (p != std::string::npos) s.replace(p, from.size(), to);
        return s;
    };
    const std::string dsha = to_hex(keyed_sha256(u.key, u.text[0].data(), u.text[0].size()));
    const std::string fdig = to_hex(keyed_sha256(u.key, u.taps[0].data(), 20));
    const std::string flip = std::string(1, dsha[0] == '0' ? '1' : '0');
    const std::string flip2 = std::string(1, fdig[0] == '0' ? '1' : '0');
    struct BadMap { std::string text; int code; };
    const BadMap bad_maps[] = {
        {replaced(dsha, flip + dsha.substr(1)), kMapDamaged},                    // wrong decoded_sha256
        {replaced(fdig, flip2 + fdig.substr(1)), kSelfCheckFailed},              // wrong filter digest
        {replaced("\"fields\":[" + std::to_string(u.fields[0][0]), "\"fields\":[" + std::to_string(u.fields[0][0] - 1)), kMapDamaged},
    };
    for (const BadMap& bm : bad_maps) {
        LoadedMap lm;
        CHECK(load_map_bytes(bm.text, lm, sink).code == kOk);
        if (!lm.map) continue;
        Converted conv;
        const Result r = convert_file(*lm.map, work / "with-f-0.bin", opt, conv, sink);
        CHECK(r.code == bm.code && conv.stages.empty() && is_installer_error(r.code));
        // The same map with the file of the other variant converts: only the found variant is checked.
        CHECK(convert_file(*lm.map, work / "with-f-1.bin", opt, conv, sink).code == kOk && conv.stages.size() == 2);
    }
    LoadedMap lm;
    CHECK(load_map_bytes(replaced("\"version\":4", "\"version\":5"), lm, sink).code == kMapDamaged);
    CHECK(load_map_bytes(replaced("\"version\":4", "\"version\":1"), lm, sink).code == kMapDamaged);  // hex32 needs version 4
    CHECK(load_map_bytes(replaced("lz4-block", "lz5-block"), lm, sink).code == kMapDamaged);
}


// ---- 57-T5: the spec's synthetic vectors for the version-5 operations ------------------------------------------

static std::vector<std::uint8_t> unhex(const std::string& h) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        out.push_back(static_cast<std::uint8_t>(std::stoul(h.substr(i, 2), nullptr, 16)));
    return out;
}
static std::uint32_t hexu(const Json* j) { return static_cast<std::uint32_t>(std::stoul(j->text, nullptr, 16)); }
static std::string hexw(std::uint32_t v, int digits) {
    char b[16];
    std::snprintf(b, sizeof b, "%0*x", digits, v);
    return b;
}

static int run_t5_vectors(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Json root;
    std::string err;
    if (!parse_json(text, root, err)) {
        std::printf("cannot read %s: %s\n", path, err.c_str());
        return 0;
    }
    int n = 0;
    // e4m3 over all 65,536 binary16 inputs; ff marks a non-finite input (an error).
    const std::vector<std::uint8_t> codes = unhex(root.find("e4m3")->find("codes_hex")->text);
    CHECK(codes.size() == 65536);
    int bad = 0;
    for (std::uint32_t h = 0; h < 65536 && codes.size() == 65536; ++h, ++n) {
        const int c = e4m3_code(static_cast<std::uint16_t>(h));
        const int want = (codes[h] == 0xFF && !half_is_finite(static_cast<std::uint16_t>(h))) ? -1 : codes[h];
        if (c != want && bad++ < 5) std::printf("e4m3 %04x: got %d want %d\n", h, c, want);
    }
    CHECK(bad == 0);
    for (const Json& c : root.find("score_bias")->items) {
        ++n;
        const std::vector<std::uint8_t> sb = unhex(c.find("S_f16le")->text);
        std::vector<std::uint16_t> S(sb.size() / 2);
        for (std::size_t i = 0; i < S.size(); ++i) S[i] = static_cast<std::uint16_t>(sb[2 * i] | (sb[2 * i + 1] << 8));
        std::int64_t H = 0;
        c.find("H")->as_int64(H);
        std::vector<std::pair<int, int>> pairs;
        for (const Json& p : c.find("pairs")->items) {
            std::int64_t I = 0, J = 0;
            p.items[0].as_int64(I);
            p.items[1].as_int64(J);
            pairs.emplace_back(static_cast<int>(I), static_cast<int>(J));
        }
        std::vector<std::uint16_t> out;
        std::string why;
        std::string got = "error";
        if (score_bias_op(S.data(), static_cast<int>(H), hexu(c.find("A_bits")), hexu(c.find("Wd_bits")), pairs, out, why)) {
            got.clear();
            for (std::uint16_t v : out) got += hexw(v & 0xFFu, 2) + hexw(static_cast<std::uint32_t>(v >> 8), 2);
        }
        const bool ok = got == c.find("expect")->text;
        if (!ok) std::printf("score_bias %s: mismatch\n", c.find("case")->text.c_str());
        CHECK(ok);
    }
    for (const Json& c : root.find("const")->items) {
        ++n;
        const std::string name = c.find("op")->text;
        const OpKind k = name == "const_f16x2"        ? OpKind::ConstF16x2
                         : name == "const_f16lo"      ? OpKind::ConstF16lo
                         : name == "const_f32_to_f16" ? OpKind::ConstF32ToF16
                                                      : OpKind::ConstU32AsF32;
        std::vector<std::uint32_t> words;
        for (const Json& w : c.find("site_values")->items)
            words.push_back(static_cast<std::uint32_t>(std::stoul(w.text, nullptr, 16)));
        const Json* ng = c.find("negate");
        std::uint32_t bits = 0;
        std::string why, got = "error";
        if (const_op(k, ng && ng->boolean, words, bits, why)) {
            const DType d = k == OpKind::ConstU32AsF32 ? DType::Float32 : DType::Float16;
            const std::uint8_t b[4] = {static_cast<std::uint8_t>(bits), static_cast<std::uint8_t>(bits >> 8),
                                       static_cast<std::uint8_t>(bits >> 16), static_cast<std::uint8_t>(bits >> 24)};
            std::uint64_t first = 0;
            if (output_is_finite(d, b, 1, first)) got = hexw(bits, d == DType::Float32 ? 8 : 4);
        }
        const bool ok = got == c.find("expect")->text;
        if (!ok) std::printf("const %s: got %s want %s\n", name.c_str(), got.c_str(), c.find("expect")->text.c_str());
        CHECK(ok);
    }
    for (const Json& c : root.find("attn_wmax")->items) {
        ++n;
        std::uint16_t w = 0;
        bool ovf = false;
        std::string why, got = "error";
        if (attn_wmax_op(static_cast<std::uint16_t>(hexu(c.find("clamp_lo"))),
                         static_cast<std::uint16_t>(hexu(c.find("clamp_hi"))), static_cast<std::uint16_t>(hexu(c.find("p1"))),
                         static_cast<std::uint16_t>(hexu(c.find("p0"))), w, ovf, why))
            got = hexw(w, 4);
        const bool ok = got == c.find("expect")->text;
        if (!ok) std::printf("attn_wmax %s: got %s\n", c.find("case")->text.c_str(), got.c_str());
        CHECK(ok);
    }
    for (const Json& c : root.find("require")->items) {
        ++n;
        const std::string dt = c.find("dtype")->text;
        const DType d = dt == "float32"   ? DType::Float32
                        : dt == "float16" ? DType::Float16
                        : dt == "uint16"  ? DType::UInt16
                        : dt == "e4m3"    ? DType::E4M3
                                          : DType::UInt32;
        const std::vector<std::uint8_t> v = unhex(c.find("values_hex")->text);
        unsigned req = 0;
        for (const Json& r : c.find("require")->items)
            req |= r.text == "finite" ? kReqFinite : r.text == "not_subnormal" ? kReqNotSubnormal : r.text == "even" ? kReqEven : kReqBelow8;
        std::uint64_t first = 0;
        std::string why;
        const std::uint64_t count = v.size() / dtype_size(d);
        const bool pass = output_is_finite(d, v.data(), count, first) && check_require(d, v.data(), count, req, why);
        const bool ok = std::string(pass ? "ok" : "error") == c.find("expect")->text;
        if (!ok) std::printf("require %s: wrong\n", c.find("case")->text.c_str());
        CHECK(ok);
    }
    for (const Json& c : root.find("checks")->items) {
        ++n;
        const std::string rel = c.find("rel")->text;
        const Rel r = rel == "ge" ? Rel::Ge : rel == "gt" ? Rel::Gt : rel == "le" ? Rel::Le : Rel::RatioBelowF16Overflow;
        const double a = bits_to_float(hexu(c.find("a_bits")));
        const Json* bj = c.find("b");
        double b;
        if (const Json* k = bj->find("const")) {
            std::int64_t iv = 0;
            b = k->as_int64(iv) ? static_cast<double>(iv) : std::strtod(k->text.c_str(), nullptr);
        } else {
            b = bits_to_float(hexu(bj->find("tensor_bits")));
        }
        const bool ok = std::string(check_relation(r, a, b) ? "ok" : "error") == c.find("expect")->text;
        if (!ok) std::printf("checks %s %s: wrong\n", rel.c_str(), c.find("a_bits")->text.c_str());
        CHECK(ok);
    }
    return n;
}

// ---- e4m3, const operations and checks against direct definitions (independent of the vector file) ------------

static void test_ops5() {
    // Every e4m3 code's value converts back to the same code; -0 keeps its sign. NaN codes never appear.
    for (int code = 0; code < 256; ++code) {
        if ((code & 0x7F) == 0x7F) continue;
        const int s = code >> 7, e = (code >> 3) & 15, m = code & 7;
        const double v = (e == 0 ? std::ldexp(m, -9) : std::ldexp(8 + m, e - 10)) * (s ? -1.0 : 1.0);
        const std::uint16_t h = rn16(s && v == 0.0 ? -0.0 : v);
        CHECK(e4m3_code(h) == code);
    }
    CHECK(e4m3_code(0x7BFF) == 0x7E);  // 65504 saturates to 448
    CHECK(e4m3_code(0xFBFF) == 0xFE);
    CHECK(e4m3_code(0x7C00) == -1 && e4m3_code(0xFE00) == -1);
    CHECK(e4m3_code(0x8001) == 128);  // a negative value that rounds to zero
    CHECK(e4m3_code(0x0001) == 0);
    // Ties to even: 1 + 1/16 lies halfway between 1 and 1 + 1/8 -> 1; 1 + 3/16 -> 1 + 1/4.
    CHECK(e4m3_code(rn16(1.0625)) == 0x38 && e4m3_code(rn16(1.1875)) == 0x3A);
    std::uint32_t bits = 0;
    std::string why;
    CHECK(const_op(OpKind::ConstF32ToF16, true, {0x80000000u}, bits, why) && bits == 0x0000);   // -(-0) = +0
    CHECK(const_op(OpKind::ConstF32ToF16, false, {0x80000001u}, bits, why) && bits == 0x8000);  // -tiny -> -0
    CHECK(!const_op(OpKind::ConstF32ToF16, true, {0xFF800000u}, bits, why));
    CHECK(!const_op(OpKind::ConstU32AsF32, false, {1u, 2u}, bits, why));
    CHECK(!const_op(OpKind::ConstU32AsF32, false, {}, bits, why));
    CHECK(check_relation(Rel::Le, -0.0, 0.0) && check_relation(Rel::Ge, 0.0, -0.0) && !check_relation(Rel::Gt, 0.0, -0.0));
    CHECK(!check_relation(Rel::RatioBelowF16Overflow, 1.0, -1.0));
    CHECK(!check_relation(Rel::RatioBelowF16Overflow, 1.0, 0.0));
}

int main(int argc, char** argv) {
    const char* work = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--lz4-vectors" && i + 1 < argc) {
            const int n = run_lz4_vectors(argv[++i]);
            if (n <= 0 || g_failures) {
                std::printf("%d failure(s)\n", g_failures ? g_failures : 1);
                return 1;
            }
            std::printf("all %d decompression cases passed\n", n);
            return 0;
        } else if (a == "--t5-vectors" && i + 1 < argc) {
            const int n = run_t5_vectors(argv[++i]);
            if (n <= 0 || g_failures) {
                std::printf("%d failure(s)\n", g_failures ? g_failures : 1);
                return 1;
            }
            std::printf("all %d 57-T5 vector checks passed\n", n);
            return 0;
        } else if (a == "--verify-model" && i + 2 < argc) {
            std::ifstream mf(argv[i + 1], std::ios::binary);
            const std::string mb((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
            LoadedMap lm;
            if (load_map_bytes(mb, lm, nullptr).code != kOk) {
                std::printf("map refused\n");
                return 2;
            }
            std::string why;
            ModelFileInfo info;
            const bool ok = verify_model_file(*lm.map, fs::u8path(argv[i + 2]), why, &info);
            std::string absent;
            for (const std::string& g : info.absent_groups) absent += " " + g;
            std::printf("%s%s; absent:%s; container %d\n", ok ? "valid" : "refused: ", ok ? "" : why.c_str(),
                        absent.c_str(), info.container_version);
            return ok ? 0 : 1;
        } else if (a == "--work" && i + 1 < argc) {
            work = argv[++i];
        } else {
            std::printf("usage: wgc-unit-tests [--work <dir>] | --lz4-vectors <lz4-block-test.json> |"
                        " --t5-vectors <57-T5-vectors.json> | --verify-model <map> <model.wgcm>\n");
            return 2;
        }
    }
    test_sha256();
    test_fnv();
    test_json();
    test_numeric();
    test_key_and_digest();
    test_base64();
    test_capped_softmax();
    test_lz4_block();
    test_variant_block();
    test_ops5();
    if (work) test_api_files(fs::u8path(work));
    else std::printf("note: no --work folder given; the library-interface file tests were skipped\n");
    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all unit tests passed\n");
    return 0;
}
