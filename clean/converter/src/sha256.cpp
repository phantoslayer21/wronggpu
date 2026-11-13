// SPDX-License-Identifier: MIT
// SHA-256 as specified in FIPS 180-4, sections 4.1.2, 4.2.2, 5.1.1, 5.3.3 and 6.2.
#include "sha256.h"

#include <cstring>

namespace wgc {
namespace {

const std::uint32_t kRound[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

Sha256::Sha256()
    : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::block(const std::uint8_t* p) {
    std::uint32_t w[64];
    for (int t = 0; t < 16; ++t) {
        w[t] = (std::uint32_t(p[4 * t]) << 24) | (std::uint32_t(p[4 * t + 1]) << 16) |
               (std::uint32_t(p[4 * t + 2]) << 8) | std::uint32_t(p[4 * t + 3]);
    }
    for (int t = 16; t < 64; ++t) {
        const std::uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
        const std::uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
        w[t] = s1 + w[t - 7] + s0 + w[t - 16];
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int t = 0; t < 64; ++t) {
        const std::uint32_t big1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = h + big1 + ch + kRound[t] + w[t];
        const std::uint32_t big0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = big0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
    const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
    total_ += size;
    if (used_ > 0) {
        const std::size_t take = (64 - used_ < size) ? 64 - used_ : size;
        std::memcpy(buf_ + used_, p, take);
        used_ += take;
        p += take;
        size -= take;
        if (used_ < 64) return;
        block(buf_);
        used_ = 0;
    }
    while (size >= 64) {
        block(p);
        p += 64;
        size -= 64;
    }
    if (size > 0) {
        std::memcpy(buf_, p, size);
        used_ = size;
    }
}

Digest Sha256::finish() {
    const std::uint64_t bits = total_ * 8;
    std::uint8_t pad[72] = {0x80};
    const std::size_t pad_len = (used_ < 56) ? 56 - used_ : 120 - used_;
    std::uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[i] = std::uint8_t(bits >> (56 - 8 * i));
    update(pad, pad_len);
    update(len_be, 8);
    Digest out;
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = std::uint8_t(h_[i] >> 24);
        out[4 * i + 1] = std::uint8_t(h_[i] >> 16);
        out[4 * i + 2] = std::uint8_t(h_[i] >> 8);
        out[4 * i + 3] = std::uint8_t(h_[i]);
    }
    return out;
}

Digest Sha256::hash(const void* data, std::size_t size) {
    Sha256 s;
    s.update(data, size);
    return s.finish();
}

Digest keyed_sha256(const Digest& key, const void* data, std::size_t size) {
    Sha256 s;
    s.update(key.data(), key.size());
    s.update(data, size);
    return s.finish();
}

std::string to_hex(const Digest& d) {
    static const char digits[] = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (std::uint8_t b : d) {
        s.push_back(digits[b >> 4]);
        s.push_back(digits[b & 15]);
    }
    return s;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool parse_digest_hex(const std::string& s, Digest& out) {
    if (s.size() != 64) return false;
    for (std::size_t i = 0; i < 32; ++i) {
        const int hi = hex_value(s[2 * i]), lo = hex_value(s[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = std::uint8_t(hi * 16 + lo);
    }
    return true;
}

}  // namespace wgc
