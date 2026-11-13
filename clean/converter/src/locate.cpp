// SPDX-License-Identifier: MIT
// Locating the sources in the DLL (spec 08 section 4.2). Written for this project.
//
// Block sources: every offset is a candidate; the search tries offsets that are multiples of 16
// first, then the remaining even offsets, then the odd ones (the order the spec allows). Within a
// pass the lowest matching offset wins. The 64-byte FNV-1a check runs on several threads (eight
// independent multiply chains at a time for the 16-byte pass, a vectorisable low-16-bit prefilter
// for the rest); the SHA-256 check runs only on FNV hits.
//
// Pair sources: for each gap in map order, every offset in ascending order; first match wins.
#include "locate.h"

#include <algorithm>
#include <climits>
#include <atomic>
#include <cstring>
#include <mutex>
#include <cstdint>
#include <thread>

#include "fnv.h"

namespace wgc {

std::uint64_t fnv_initial_state(bool keyed, const Digest* key) {
    std::uint64_t s0 = kFnvOffsetBasis;
    if (keyed && key) {
        std::uint64_t k = 0;
        for (int i = 7; i >= 0; --i) k = (k << 8) | (*key)[static_cast<std::size_t>(i)];  // little-endian
        s0 ^= k;
    }
    return s0;
}

namespace {

unsigned thread_count(const SearchOptions& opt) {
    unsigned n = opt.threads ? opt.threads : std::thread::hardware_concurrency();
    if (n == 0) n = 1;
    return std::min(n, 64u);
}

// Runs fn(chunk_begin, chunk_end, hits) over [0, n) in chunks on several threads and returns all
// hits, sorted.
template <typename Fn>
std::vector<std::uint64_t> parallel_scan(std::uint64_t n, unsigned threads, Fn fn) {
    const std::uint64_t chunk = std::uint64_t(1) << 20;
    const std::uint64_t chunks = (n + chunk - 1) / chunk;
    std::atomic<std::uint64_t> next{0};
    std::mutex mu;
    std::vector<std::uint64_t> all;
    auto worker = [&]() {
        std::vector<std::uint64_t> local;
        for (;;) {
            const std::uint64_t c = next.fetch_add(1);
            if (c >= chunks) break;
            fn(c * chunk, std::min(n, (c + 1) * chunk), local);
        }
        std::lock_guard<std::mutex> lock(mu);
        all.insert(all.end(), local.begin(), local.end());
    };
    const unsigned t = static_cast<unsigned>(std::min<std::uint64_t>(threads, std::max<std::uint64_t>(chunks, 1)));
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < t; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    std::sort(all.begin(), all.end());
    return all;
}

// FNV of `len` bytes at eight offsets at once (the eight multiply chains are independent).
inline void fnv8(const std::uint8_t* file, const std::uint64_t* offs, std::size_t len, std::uint64_t s0,
                 std::uint64_t* out) {
    std::uint64_t h0 = s0, h1 = s0, h2 = s0, h3 = s0, h4 = s0, h5 = s0, h6 = s0, h7 = s0;
    const std::uint8_t *p0 = file + offs[0], *p1 = file + offs[1], *p2 = file + offs[2], *p3 = file + offs[3];
    const std::uint8_t *p4 = file + offs[4], *p5 = file + offs[5], *p6 = file + offs[6], *p7 = file + offs[7];
    for (std::size_t i = 0; i < len; ++i) {
        h0 = (h0 ^ p0[i]) * kFnvPrime;
        h1 = (h1 ^ p1[i]) * kFnvPrime;
        h2 = (h2 ^ p2[i]) * kFnvPrime;
        h3 = (h3 ^ p3[i]) * kFnvPrime;
        h4 = (h4 ^ p4[i]) * kFnvPrime;
        h5 = (h5 ^ p5[i]) * kFnvPrime;
        h6 = (h6 ^ p6[i]) * kFnvPrime;
        h7 = (h7 ^ p7[i]) * kFnvPrime;
    }
    out[0] = h0, out[1] = h1, out[2] = h2, out[3] = h3, out[4] = h4, out[5] = h5, out[6] = h6, out[7] = h7;
}

// Low 16 bits of the FNV-1a state for the n consecutive offsets begin..begin+n-1. Multiplication and
// XOR modulo 2^16 depend only on the low 16 bits of their operands, so this equals the low 16 bits of
// the full 64-bit hash. The inner loop runs over consecutive offsets with 16-bit lanes, which
// compilers vectorise; it is used as a prefilter before the full hash.
void fnv_low16(const std::uint8_t* file, std::uint64_t begin, std::size_t n, std::size_t len, std::uint16_t s0,
               std::uint16_t* out) {
    const std::uint16_t prime16 = static_cast<std::uint16_t>(kFnvPrime & 0xFFFF);
    for (std::size_t j = 0; j < n; ++j) out[j] = s0;
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint8_t* p = file + begin + i;
        for (std::size_t j = 0; j < n; ++j)
            out[j] = static_cast<std::uint16_t>(static_cast<unsigned>(out[j] ^ p[j]) * prime16);
    }
}

Digest sha_of(const std::uint8_t* data, std::size_t n, bool keyed, const Digest* key) {
    return keyed ? keyed_sha256(*key, data, n) : Sha256::hash(data, n);
}

Found locate_block(const std::uint8_t* file, std::uint64_t size, const Source& s, const Digest* key,
                   const SearchOptions& opt) {
    Found f;
    if (s.length > size) return f;
    const std::uint64_t last = size - s.length;  // highest candidate offset
    const std::size_t hlen = static_cast<std::size_t>(std::min<std::uint64_t>(64, s.length));
    const std::uint64_t s0 = fnv_initial_state(s.keyed, key);
    const std::uint64_t target = s.fnv;
    auto verify = [&](const std::vector<std::uint64_t>& hits) {
        for (std::uint64_t o : hits) {
            if (sha_of(file + o, static_cast<std::size_t>(s.length), s.keyed, key) == s.sha) {
                f.found = true;
                f.offset = f.offset2 = o;
                f.bytes.assign(file + o, file + o + s.length);
                return true;
            }
        }
        return false;
    };

    // Pass 1: every 16th offset, full hashes, eight independent chains at a time.
    auto scan16 = [&](std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& hits) {
        std::uint64_t offs[8], h[8];
        int n = 0;
        for (std::uint64_t o = (begin + 15) & ~std::uint64_t(15); o < end; o += 16) {
            offs[n++] = o;
            if (n == 8) {
                fnv8(file, offs, hlen, s0, h);
                for (int i = 0; i < 8; ++i)
                    if (h[i] == target) hits.push_back(offs[i]);
                n = 0;
            }
        }
        for (int i = 0; i < n; ++i)
            if (fnv1a64(file + offs[i], hlen, s0) == target) hits.push_back(offs[i]);
    };
    if (verify(parallel_scan(last + 1, thread_count(opt), scan16))) return f;

    // Passes 2 (other even offsets) and 3 (odd offsets) share one scan: a low-16-bit prefilter over
    // all offsets, then the full hash on its hits. The even hits are verified before the odd ones.
    auto scan_rest = [&](std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& hits) {
        const std::size_t block = 4096;
        std::uint16_t low[block];
        const std::uint16_t s0_16 = static_cast<std::uint16_t>(s0), target16 = static_cast<std::uint16_t>(target);
        for (std::uint64_t b = begin; b < end; b += block) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(block, end - b));
            fnv_low16(file, b, n, hlen, s0_16, low);
            for (std::size_t j = 0; j < n; ++j) {
                const std::uint64_t o = b + j;
                if (low[j] == target16 && (o & 15) != 0 && fnv1a64(file + o, hlen, s0) == target) hits.push_back(o);
            }
        }
    };
    const std::vector<std::uint64_t> hits = parallel_scan(last + 1, thread_count(opt), scan_rest);
    std::vector<std::uint64_t> even, odd;
    for (std::uint64_t o : hits) ((o & 1) ? odd : even).push_back(o);
    if (verify(even)) return f;
    verify(odd);
    return f;
}

Found locate_pair(const std::uint8_t* file, std::uint64_t size, const Source& s, const Digest* key,
                  const SearchOptions& opt) {
    Found f;
    const std::uint64_t s0 = fnv_initial_state(s.keyed, key);
    for (std::uint64_t g : s.gaps) {
        if (g + 4 > size) continue;
        const std::uint64_t count = size - g - 4 + 1;  // offsets o with o + g + 4 <= size
        auto scan = [&](std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& hits) {
            for (std::uint64_t o = begin; o < end; ++o) {
                std::uint8_t cand[8];
                std::memcpy(cand, file + o, 4);
                std::memcpy(cand + 4, file + o + g, 4);
                if (fnv1a64(cand, 8, s0) == s.fnv) hits.push_back(o);
            }
        };
        const std::vector<std::uint64_t> hits = parallel_scan(count, thread_count(opt), scan);
        for (std::uint64_t o : hits) {
            std::uint8_t cand[8];
            std::memcpy(cand, file + o, 4);
            std::memcpy(cand + 4, file + o + g, 4);
            if (sha_of(cand, 8, s.keyed, key) == s.sha) {
                f.found = true;
                f.offset = o;
                f.offset2 = o + g;
                f.bytes.assign(cand, cand + 8);
                return f;
            }
        }
    }
    return f;
}

// Shared scan for several targets with one prefix length and one initial state (all in `group`).
void locate_group(const std::uint8_t* file, std::uint64_t size, const std::vector<BlockTarget>& targets,
                  const std::vector<std::size_t>& group, std::size_t hlen, std::uint64_t s0, const Digest* key,
                  const SearchOptions& opt, std::vector<Found>& found) {
    std::uint64_t min_len = UINT64_MAX;
    for (std::size_t t : group) min_len = std::min(min_len, targets[t].length);
    if (min_len > size) return;
    const std::uint64_t last = size - min_len;  // highest candidate offset of any target
    std::vector<std::uint64_t> wanted;          // sorted prefix hashes of the targets still missing
    auto refresh = [&]() {
        wanted.clear();
        for (std::size_t t : group)
            if (!found[t].found) wanted.push_back(targets[t].fnv);
        std::sort(wanted.begin(), wanted.end());
    };
    auto is_wanted = [&](std::uint64_t h) { return std::binary_search(wanted.begin(), wanted.end(), h); };
    // Hits in ascending order: each target takes the first offset whose full block matches.
    auto verify = [&](const std::vector<std::uint64_t>& hits) {
        for (std::uint64_t o : hits) {
            const std::uint64_t h = fnv1a64(file + o, hlen, s0);
            for (std::size_t t : group) {
                const BlockTarget& bt = targets[t];
                if (found[t].found || bt.fnv != h || o + bt.length > size) continue;
                if (sha_of(file + o, static_cast<std::size_t>(bt.length), bt.keyed, key) == bt.sha) {
                    found[t].found = true;
                    found[t].offset = found[t].offset2 = o;
                    found[t].bytes.assign(file + o, file + o + bt.length);
                }
            }
        }
    };
    refresh();
    // Pass 1: every 16th offset.
    auto scan16 = [&](std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& hits) {
        std::uint64_t offs[8], h[8];
        int n = 0;
        for (std::uint64_t o = (begin + 15) & ~std::uint64_t(15); o < end; o += 16) {
            offs[n++] = o;
            if (n == 8) {
                fnv8(file, offs, hlen, s0, h);
                for (int i = 0; i < 8; ++i)
                    if (is_wanted(h[i])) hits.push_back(offs[i]);
                n = 0;
            }
        }
        for (int i = 0; i < n; ++i)
            if (is_wanted(fnv1a64(file + offs[i], hlen, s0))) hits.push_back(offs[i]);
    };
    verify(parallel_scan(last + 1, thread_count(opt), scan16));
    refresh();
    if (wanted.empty()) return;
    // Passes 2 and 3: the other even offsets, then the odd ones, through a low-16-bit prefilter.
    std::vector<std::uint8_t> low_map(65536 / 8, 0);
    for (std::uint64_t h : wanted) low_map[(h & 0xFFFF) >> 3] |= static_cast<std::uint8_t>(1u << (h & 7));
    const std::uint16_t s0_16 = static_cast<std::uint16_t>(s0);
    auto scan_rest = [&](std::uint64_t begin, std::uint64_t end, std::vector<std::uint64_t>& hits) {
        const std::size_t block = 4096;
        std::uint16_t low[block];
        for (std::uint64_t b = begin; b < end; b += block) {
            const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(block, end - b));
            fnv_low16(file, b, n, hlen, s0_16, low);
            for (std::size_t j = 0; j < n; ++j) {
                const std::uint64_t o = b + j;
                if ((low_map[low[j] >> 3] >> (low[j] & 7)) & 1)
                    if ((o & 15) != 0 && is_wanted(fnv1a64(file + o, hlen, s0))) hits.push_back(o);
            }
        }
    };
    const std::vector<std::uint64_t> hits = parallel_scan(last + 1, thread_count(opt), scan_rest);
    std::vector<std::uint64_t> even, odd;
    for (std::uint64_t o : hits) ((o & 1) ? odd : even).push_back(o);
    verify(even);
    verify(odd);
}

}  // namespace

std::vector<Found> locate_blocks(const std::uint8_t* file, std::uint64_t size, const std::vector<BlockTarget>& targets,
                                 const Digest* key, const SearchOptions& opt) {
    std::vector<Found> found(targets.size());
    // Groups of equal prefix length and key use (each a shared scan).
    std::vector<bool> done(targets.size(), false);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        if (done[i]) continue;
        const std::size_t hlen = static_cast<std::size_t>(std::min<std::uint64_t>(64, targets[i].length));
        std::vector<std::size_t> group;
        for (std::size_t j = i; j < targets.size(); ++j) {
            const std::size_t hj = static_cast<std::size_t>(std::min<std::uint64_t>(64, targets[j].length));
            if (!done[j] && hj == hlen && targets[j].keyed == targets[i].keyed) {
                group.push_back(j);
                done[j] = true;
            }
        }
        if (hlen == 0) continue;
        locate_group(file, size, targets, group, hlen, fnv_initial_state(targets[i].keyed, key), key, opt, found);
    }
    return found;
}

Found locate_source(const std::uint8_t* file, std::uint64_t size, const Source& s, const Digest* key,
                    const SearchOptions& opt) {
    return s.locate == Locate::Block ? locate_block(file, size, s, key, opt) : locate_pair(file, size, s, key, opt);
}

FoundVariant locate_variant(const std::uint8_t* file, std::uint64_t size, const Source& s, const Digest* key,
                            const SearchOptions& opt) {
    FoundVariant fv;
    for (std::size_t i = 0; i < s.variants.size(); ++i) {
        const Variant& v = s.variants[i];
        Source block;  // the variant as a block source of section 4.2
        block.name = s.name;
        block.length = v.length;
        block.locate = Locate::Block;
        block.keyed = s.keyed;
        block.fnv = v.fnv;
        block.sha = v.sha;
        Found f = locate_block(file, size, block, key, opt);
        if (!f.found) continue;
        fv.variant = static_cast<int>(i);
        fv.offset = f.offset;
        fv.bytes = std::move(f.bytes);
        break;
    }
    return fv;
}

}  // namespace wgc
