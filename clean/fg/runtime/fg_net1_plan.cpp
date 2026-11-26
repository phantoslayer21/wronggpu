// SPDX-License-Identifier: MIT
#include "fg_net1_plan.h"

#include <algorithm>
#include <cstring>
#include <set>

#include "fg_model.h"

namespace wgfg {

namespace {

constexpr FgNet1Layer kLayers[17] = {
    {"net1_01", 32, 3, 48, FgChunkOrder::BlockKxKy, true},  {"net1_02", 64, 1, 32, FgChunkOrder::Ascending, true},
    {"net1_03", 128, 3, 64, FgChunkOrder::BlockKxKy, true}, {"net1_04", 128, 3, 128, FgChunkOrder::BlockKxKy, true},
    {"net1_05", 256, 3, 128, FgChunkOrder::KxBlockKy, true}, {"net1_06", 512, 1, 256, FgChunkOrder::Ascending, true},
    {"net1_07", 256, 1, 512, FgChunkOrder::Ascending, true}, {"net1_08", 256, 1, 256, FgChunkOrder::Ascending, true},
    {"net1_09", 128, 1, 256, FgChunkOrder::Ascending, true}, {"net1_10", 128, 1, 128, FgChunkOrder::Ascending, true},
    {"net1_11", 128, 1, 128, FgChunkOrder::Ascending, true}, {"net1_12", 128, 1, 128, FgChunkOrder::Ascending, true},
    {"net1_13", 64, 1, 128, FgChunkOrder::Ascending, true},  {"net1_14", 64, 1, 64, FgChunkOrder::Ascending, true},
    {"net1_15", 32, 1, 64, FgChunkOrder::Ascending, true},   {"net1_16", 32, 1, 32, FgChunkOrder::Ascending, true},
    {"net1_17", 16, 1, 32, FgChunkOrder::Ascending, false},
};

// fused 1 x 1 pairs: (mode, first conv's output tiles, second conv's output tiles, 16-pixel groups per wave)
struct PairShape {
    uint32_t mode, c1t, c2t, nt;
};
constexpr PairShape kPairs[6] = {{0, 32, 16, 1}, {1, 16, 8, 1}, {1, 8, 8, 1}, {1, 8, 4, 1}, {1, 4, 2, 2}, {1, 2, 1, 2}};
// MPairB, MPairD4 .. MPairD0
const PairShape& pair_shape(FgNet1Kernel k) { return kPairs[static_cast<uint32_t>(k) - static_cast<uint32_t>(FgNet1Kernel::MPairB)]; }
bool is_pair(FgNet1Kernel k) { return static_cast<uint32_t>(k) >= static_cast<uint32_t>(FgNet1Kernel::MPairB); }

constexpr FgNet1Kernel kSequence[kFgNet1Steps] = {
    FgNet1Kernel::MConv2,  FgNet1Kernel::MConv4,  FgNet1Kernel::MConv4,  FgNet1Kernel::MConv4,
    FgNet1Kernel::MConv4,  FgNet1Kernel::MPairB,  FgNet1Kernel::MPairD4, FgNet1Kernel::MPairD3,
    FgNet1Kernel::MPairD2, FgNet1Kernel::MPairD1, FgNet1Kernel::MPairD0,
};

constexpr int kArgs = static_cast<int>(kFgNet1Args);

uint32_t cdiv(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }
uint64_t align_up(uint64_t v) { return (v + kFgArenaAlign - 1) & ~(kFgArenaAlign - 1); }
uint32_t mconv_mt(uint32_t cout) { return cout % 64 == 0 ? 4u : (cout % 32 == 0 ? 2u : 1u); }
FgNet1Kernel mconv_kernel(uint32_t mt) {
    return mt == 4 ? FgNet1Kernel::MConv4 : (mt == 2 ? FgNet1Kernel::MConv2 : FgNet1Kernel::MConv1);
}

FgArg buf(int index) { return {true, static_cast<uint32_t>(index)}; }
FgArg val(uint32_t v) { return {false, v}; }

struct Builder {
    FgNet1Plan* p;
    int add(const std::string& name, FgRegion r, uint32_t h, uint32_t w, uint32_t c) {
        FgBufferSpec b;
        b.name = name;
        b.region = r;
        b.h = h;
        b.w = w;
        b.c = c;
        b.bytes = 2ull * h * w * c;
        uint64_t& end = p->region_bytes[static_cast<uint32_t>(r)];
        b.offset = end;
        end = align_up(end + b.bytes);
        p->buffers.push_back(b);
        return static_cast<int>(p->buffers.size() - 1);
    }
};

}  // namespace

const char* FgNet1KernelName(FgNet1Kernel k) {
    switch (k) {
        case FgNet1Kernel::MConv1: return "fg_mconv1";
        case FgNet1Kernel::MConv2: return "fg_mconv2";
        case FgNet1Kernel::MConv4: return "fg_mconv4";
        case FgNet1Kernel::MPairB: return "fg_mpair_b";
        case FgNet1Kernel::MPairD4: return "fg_mpair_d4";
        case FgNet1Kernel::MPairD3: return "fg_mpair_d3";
        case FgNet1Kernel::MPairD2: return "fg_mpair_d2";
        case FgNet1Kernel::MPairD1: return "fg_mpair_d1";
        case FgNet1Kernel::MPairD0: return "fg_mpair_d0";
    }
    return "?";
}

const FgNet1Layer* FgNet1Layers() { return kLayers; }

int FgNet1Plan::Find(const std::string& name) const {
    for (size_t i = 0; i < buffers.size(); ++i)
        if (buffers[i].name == name) return static_cast<int>(i);
    return -1;
}

uint32_t FgRatioBits(uint32_t n_in, uint32_t n_out) {
    if (n_out <= 1) return 0;
    // One correctly rounded single-precision division of exactly representable integers (both < 2^24).
    const volatile float a = static_cast<float>(n_in - 1);
    const volatile float b = static_cast<float>(n_out - 1);
    const float r = a / b;
    uint32_t bits;
    std::memcpy(&bits, &r, 4);
    return bits;
}

FgRefusal FgBuildNet1Plan(uint32_t h0, uint32_t w0, FgNet1Plan* out) {
    if (h0 < 32 || w0 < 32 || h0 > 4096 || w0 > 4096) return FgRefusal::Net1GridUnsupported;
    FgNet1Plan p;
    p.h0 = h0;
    p.w0 = w0;
    Builder b{&p};
    uint32_t H[6], W[6];
    H[0] = h0;
    W[0] = w0;
    for (int l = 1; l < 6; ++l) {
        H[l] = H[l - 1] / 2;
        W[l] = W[l - 1] / 2;
    }
    const int cells = b.add("cells", FgRegion::Input, h0, w0, kFgCellChannels);
    int wf[17], bi[17];
    for (int i = 0; i < 17; ++i) {
        const FgNet1Layer& l = kLayers[i];
        wf[i] = b.add(std::string("wf_") + l.name, FgRegion::Weights, 1, 1, l.cout * l.k * l.k * l.cin);
        bi[i] = b.add(std::string("b_") + l.name, FgRegion::Weights, 1, 1, l.cout);
    }
    // encoders: a0..a4 with the pooled p1..p5
    int a[5], pool[6];
    pool[0] = cells;
    for (int lvl = 0; lvl < 5; ++lvl) {
        const FgNet1Layer& l = kLayers[lvl];
        a[lvl] = b.add("a" + std::to_string(lvl), FgRegion::Scratch, H[lvl], W[lvl], l.cout);
        pool[lvl + 1] = b.add("p" + std::to_string(lvl + 1), FgRegion::Scratch, H[lvl + 1], W[lvl + 1], l.cout);
        const uint32_t mt = mconv_mt(l.cout), tx = cdiv(W[lvl], 16), ty = cdiv(H[lvl], 16);
        FgNet1Step s;
        s.kernel = mconv_kernel(mt);
        s.grid = tx * ty * (l.cout / (16 * mt));
        const FgArg args[kFgNet1Args] = {buf(pool[lvl]), buf(wf[lvl]), buf(bi[lvl]), buf(a[lvl]), buf(pool[lvl + 1]),
                                         val(l.cin), val(l.cout), val(H[lvl]), val(W[lvl]), val(l.k),
                                         val(static_cast<uint32_t>(l.order)), val(l.leaky ? 1u : 0u), val(1),
                                         val(tx), val(ty)};
        std::copy(args, args + kFgNet1Args, s.args);
        s.nouts = 2;
        s.outs[0] = static_cast<uint32_t>(a[lvl]);
        s.outs[1] = static_cast<uint32_t>(pool[lvl + 1]);
        p.steps.push_back(s);
    }
    // the bottleneck pair and the five decoder pairs
    auto pair = [&](FgNet1Kernel kernel, int x, int skip, int l1, int l2, int y, uint32_t lvl) {
        const PairShape& ps = pair_shape(kernel);
        const FgBufferSpec& sk = p.buffers[static_cast<size_t>(skip)];
        const FgBufferSpec& xs = p.buffers[static_cast<size_t>(x)];
        const uint32_t hi = ps.mode == 1 ? xs.h : 1, wi = ps.mode == 1 ? xs.w : 1;
        FgNet1Step s;
        s.kernel = kernel;
        s.grid = cdiv(static_cast<uint64_t>(sk.h) * sk.w, 128ull * ps.nt);
        const FgArg args[kFgNet1Args] = {buf(x), buf(skip), buf(wf[l1]), buf(bi[l1]), buf(wf[l2]), buf(bi[l2]), buf(y),
                                         val(sk.c), val(sk.h), val(sk.w), val(hi), val(wi),
                                         val(ps.mode == 1 ? FgRatioBits(wi, sk.w) : 0u),
                                         val(ps.mode == 1 ? FgRatioBits(hi, sk.h) : 0u),
                                         val(kLayers[l2].leaky ? 1u : 0u)};
        std::copy(args, args + kFgNet1Args, s.args);
        s.nouts = 1;
        s.outs[0] = static_cast<uint32_t>(y);
        (void)lvl;
        p.steps.push_back(s);
    };
    const int d5 = b.add("d5", FgRegion::Scratch, H[5], W[5], kLayers[6].cout);
    pair(FgNet1Kernel::MPairB, pool[5], pool[5], 5, 6, d5, 5);
    int prev = d5;
    const FgNet1Kernel dk[5] = {FgNet1Kernel::MPairD4, FgNet1Kernel::MPairD3, FgNet1Kernel::MPairD2,
                                FgNet1Kernel::MPairD1, FgNet1Kernel::MPairD0};
    for (int j = 0; j < 5; ++j) {
        const int lvl = 4 - j, l1 = 7 + 2 * j, l2 = 8 + 2 * j;
        const int y = b.add(lvl ? "d" + std::to_string(lvl) : std::string("out"), FgRegion::Scratch,
                            H[lvl], W[lvl], kLayers[l2].cout);
        pair(dk[j], prev, a[lvl], l1, l2, y, static_cast<uint32_t>(lvl));
        prev = y;
    }
    std::string why;
    const FgRefusal v = FgValidateNet1Plan(p, &why);
    if (v != FgRefusal::None) return v;
    *out = std::move(p);
    return FgRefusal::None;
}

// ---- independent validation ------------------------------------------------------------------------------------------

namespace {

struct Check {
    std::string* why;
    FgRefusal code = FgRefusal::None;
    FgRefusal fail(FgRefusal c, const std::string& w) {
        if (code == FgRefusal::None) {
            code = c;
            if (why) *why = w;
        }
        return code;
    }
};

}  // namespace

FgRefusal FgValidateNet1Plan(const FgNet1Plan& p, std::string* why) {
    Check ck{why};
    const size_t nb = p.buffers.size();
    // buffers: names, sizes, offsets
    std::set<std::string> names;
    for (size_t i = 0; i < nb; ++i) {
        const FgBufferSpec& b = p.buffers[i];
        if (!names.insert(b.name).second) return ck.fail(FgRefusal::Net1PlanInvalid, "duplicate buffer " + b.name);
        if (static_cast<uint32_t>(b.region) >= kFgRegionCount)
            return ck.fail(FgRefusal::Net1PlanInvalid, "region of " + b.name);
        if (b.bytes == 0 || b.bytes != 2ull * b.h * b.w * b.c)
            return ck.fail(FgRefusal::Net1PlanInvalid, "byte size of " + b.name);
        if (b.offset % kFgArenaAlign) return ck.fail(FgRefusal::Net1RegionAlign, "offset of " + b.name);
        if (b.offset + b.bytes > p.region_bytes[static_cast<uint32_t>(b.region)])
            return ck.fail(FgRefusal::Net1PlanInvalid, b.name + " ends past its region");
        for (size_t j = 0; j < i; ++j) {
            const FgBufferSpec& o = p.buffers[j];
            if (o.region == b.region && b.offset < o.offset + o.bytes && o.offset < b.offset + b.bytes)
                return ck.fail(FgRefusal::Net1RegionOverlap, b.name + " overlaps " + o.name);
        }
    }
    const int cells = p.Find("cells"), out = p.Find("out");
    if (cells < 0 || out < 0) return ck.fail(FgRefusal::Net1PlanInvalid, "cells / out missing");
    const FgBufferSpec& cb = p.buffers[static_cast<size_t>(cells)];
    if (cb.region != FgRegion::Input || cb.h != p.h0 || cb.w != p.w0 || cb.c != kFgCellChannels)
        return ck.fail(FgRefusal::Net1PlanInvalid, "cells buffer");
    if (p.steps.size() != kFgNet1Steps) return ck.fail(FgRefusal::Net1PlanInvalid, "launch count");

    std::vector<int> writes(nb, 0);
    auto B = [&](const FgNet1Step& s, int a) -> const FgBufferSpec* {
        if (!s.args[a].buffer || s.args[a].value >= nb) return nullptr;
        return &p.buffers[s.args[a].value];
    };
    auto readable = [&](const FgNet1Step& s, int a) {
        const FgBufferSpec* b = B(s, a);
        return b && (b->region != FgRegion::Scratch || writes[s.args[a].value] == 1);
    };
    auto V = [](const FgNet1Step& s, int a) { return s.args[a].buffer ? 0xFFFFFFFFu : s.args[a].value; };
    int prev_out = -1;
    for (size_t si = 0; si < p.steps.size(); ++si) {
        const FgNet1Step& s = p.steps[si];
        const std::string at = "launch " + std::to_string(si) + " (" + FgNet1KernelName(s.kernel) + "): ";
        if (s.kernel != kSequence[si]) return ck.fail(FgRefusal::Net1PlanInvalid, at + "kernel order");
        const int nbuf = is_pair(s.kernel) ? 7 : 5;
        for (int a = 0; a < kArgs; ++a)
            if (s.args[a].buffer != (a < nbuf)) return ck.fail(FgRefusal::Net1PlanInvalid, at + "argument kinds");
        for (int a = 0; a < nbuf; ++a)
            if (!B(s, a)) return ck.fail(FgRefusal::Net1PlanInvalid, at + "buffer index");
        if (!is_pair(s.kernel)) {
            const FgBufferSpec &x = *B(s, 0), &wf = *B(s, 1), &bi = *B(s, 2), &y = *B(s, 3), &pl = *B(s, 4);
            const uint32_t cin = V(s, 5), cout = V(s, 6), H = V(s, 7), W = V(s, 8), k = V(s, 9);
            const uint32_t order = V(s, 10), act = V(s, 11), pool = V(s, 12), tx = V(s, 13), ty = V(s, 14);
            if (!readable(s, 0) || x.region == FgRegion::Weights || (si == 0) != (s.args[0].value == static_cast<uint32_t>(cells)))
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "input buffer");
            if (wf.region != FgRegion::Weights || bi.region != FgRegion::Weights || y.region != FgRegion::Scratch ||
                pl.region != FgRegion::Scratch)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "buffer regions");
            if (x.h != H || x.w != W || x.c != cin || y.h != H || y.w != W || y.c != cout || pl.h != H / 2 ||
                pl.w != W / 2 || pl.c != cout)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "tensor shapes");
            if ((k != 1 && k != 3) || order > 2 || (k == 1) != (order == 0) || act != 1 || pool != 1 || cin % 16 ||
                cout % 16)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "constants");
            if (wf.bytes != 2ull * cout * k * k * cin || bi.bytes != 2ull * cout)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "weight sizes");
            const uint32_t mt = mconv_mt(cout);
            if (s.kernel != mconv_kernel(mt) || tx != cdiv(W, 16) || ty != cdiv(H, 16) ||
                s.grid != tx * ty * (cout / (16 * mt)))
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "grid");
            if (s.nouts != 2 || s.outs[0] != s.args[3].value || s.outs[1] != s.args[4].value ||
                s.args[3].value == s.args[4].value)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "outputs");
            if (si > 0 && s.args[0].value != static_cast<uint32_t>(prev_out))
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "not fed by the previous pool");
            prev_out = static_cast<int>(s.args[4].value);
        } else {
            const PairShape& ps = pair_shape(s.kernel);
            const FgBufferSpec &x = *B(s, 0), &a = *B(s, 1), &w1 = *B(s, 2), &b1 = *B(s, 3), &w2 = *B(s, 4),
                               &b2 = *B(s, 5), &y = *B(s, 6);
            const uint32_t C = V(s, 7), H = V(s, 8), W = V(s, 9), hi = V(s, 10), wi = V(s, 11), rx = V(s, 12),
                           ry = V(s, 13), act2 = V(s, 14);
            if (!readable(s, 0) || !readable(s, 1) || x.region != FgRegion::Scratch || a.region != FgRegion::Scratch)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "input buffers");
            if (w1.region != FgRegion::Weights || b1.region != FgRegion::Weights || w2.region != FgRegion::Weights ||
                b2.region != FgRegion::Weights || y.region != FgRegion::Scratch)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "buffer regions");
            const uint32_t c1 = 16 * ps.c1t, c2 = 16 * ps.c2t;
            if (w1.bytes != 2ull * c1 * C || b1.bytes != 2ull * c1 || w2.bytes != 2ull * c2 * c1 || b2.bytes != 2ull * c2)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "weight sizes");
            if (a.h != H || a.w != W || a.c != C || y.h != H || y.w != W || y.c != c2)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "tensor shapes");
            if (s.args[0].value != static_cast<uint32_t>(prev_out))
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "not fed by the previous launch");
            if (ps.mode == 0) {
                if (s.args[0].value != s.args[1].value || hi != 1 || wi != 1 || rx != 0 || ry != 0)
                    return ck.fail(FgRefusal::Net1PlanInvalid, at + "bottleneck constants");
            } else {
                // the skip is the encoder output at this level (written by the encoder launch of the same size)
                if (x.h != hi || x.w != wi || x.c != C || hi != H / 2 || wi != W / 2 || rx != FgRatioBits(wi, W) ||
                    ry != FgRatioBits(hi, H) || a.name[0] != 'a')
                    return ck.fail(FgRefusal::Net1PlanInvalid, at + "upscale constants");
            }
            const bool last = si + 1 == p.steps.size();
            if (act2 != (last ? 0u : 1u)) return ck.fail(FgRefusal::Net1PlanInvalid, at + "activation");
            if (s.grid != cdiv(static_cast<uint64_t>(H) * W, 128ull * ps.nt))
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "grid");
            if (s.nouts != 1 || s.outs[0] != s.args[6].value)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "outputs");
            if (last && s.args[6].value != static_cast<uint32_t>(out))
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "the last launch must write out");
            prev_out = static_cast<int>(s.args[6].value);
        }
        for (uint32_t o = 0; o < s.nouts; ++o) {
            if (s.outs[o] >= nb || p.buffers[s.outs[o]].region != FgRegion::Scratch || ++writes[s.outs[o]] != 1)
                return ck.fail(FgRefusal::Net1PlanInvalid, at + "a buffer written twice");
        }
    }
    for (size_t i = 0; i < nb; ++i)
        if (p.buffers[i].region == FgRegion::Scratch && writes[i] != 1)
            return ck.fail(FgRefusal::Net1PlanInvalid, "scratch buffer " + p.buffers[i].name + " never written");
    return FgRefusal::None;
}

// ---- recording -------------------------------------------------------------------------------------------------------

FgRefusal FgRecordNet1(const FgNet1Plan& plan, const FgPipelineSet& pipelines, const FgArenaBinding& arena,
                       IFgComputeRecorder& rec, std::string* why, uint32_t max_groups) {
    std::string local;
    std::string* w = why ? why : &local;
    if (max_groups == 0 || max_groups > kFgMaxGroupsPerDispatch) {
        *w = "dispatch group limit";
        return FgRefusal::Net1PlanInvalid;
    }
    // Everything is checked before the first command: a refusal leaves the list untouched.
    const FgRefusal v = FgValidateNet1Plan(plan, w);
    if (v != FgRefusal::None) return v;
    for (const FgNet1Step& s : plan.steps) {
        if (!pipelines.ready[static_cast<uint32_t>(s.kernel)]) {
            *w = std::string("no pipeline for ") + FgNet1KernelName(s.kernel);
            return FgRefusal::Net1PipelineMissing;
        }
    }
    if (!arena.bound || arena.arena_bytes == 0) {
        *w = "no arena bound";
        return FgRefusal::Net1ArenaMissing;
    }
    for (uint32_t r = 0; r < kFgRegionCount; ++r) {
        if (arena.region_base[r] % kFgArenaAlign) {
            *w = "region base alignment";
            return FgRefusal::Net1RegionAlign;
        }
        const uint64_t end = arena.region_base[r] + plan.region_bytes[r];
        if (end > arena.arena_bytes || end > 0xFFFFFFFFull) {
            *w = "region " + std::to_string(r) + " ends past the arena (or past 32-bit offsets)";
            return FgRefusal::Net1ArenaTooSmall;
        }
        for (uint32_t o = 0; o < r; ++o) {
            if (arena.region_base[r] < arena.region_base[o] + plan.region_bytes[o] &&
                arena.region_base[o] < arena.region_base[r] + plan.region_bytes[r]) {
                *w = "regions " + std::to_string(o) + " and " + std::to_string(r) + " overlap";
                return FgRefusal::Net1RegionOverlap;
            }
        }
    }
    rec.BindArena();
    for (const FgNet1Step& s : plan.steps) {
        uint32_t c[2 + kFgNet1Args];
        c[1] = s.grid;
        for (int a = 0; a < kArgs; ++a) {
            if (s.args[a].buffer) {
                const FgBufferSpec& b = plan.buffers[s.args[a].value];
                c[2 + a] = static_cast<uint32_t>(arena.region_base[static_cast<uint32_t>(b.region)] + b.offset);
            } else {
                c[2 + a] = s.args[a].value;
            }
        }
        rec.UavBarrier();
        rec.SetPipeline(s.kernel);
        for (uint32_t base = 0; base < s.grid; base += max_groups) {
            c[0] = base;
            rec.SetConstants(c, 2 + kFgNet1Args);
            rec.Dispatch(std::min(s.grid - base, max_groups));
            if (s.grid - base <= max_groups) break;
        }
    }
    rec.UavBarrier();
    return FgRefusal::None;
}

// ---- weight packing --------------------------------------------------------------------------------------------------

bool FgSwizzleNet1(const uint16_t* w, uint32_t cout, uint32_t k, uint32_t cin, FgChunkOrder order,
                   std::vector<uint8_t>* out) {
    if (!w || !out || cout % 16 || cin % 16 || (k != 1 && k != 3) || (order == FgChunkOrder::Ascending) != (k == 1))
        return false;
    const uint32_t nb = cin / 16, nq = nb * k * k;
    out->assign(2ull * cout * k * k * cin, 0);
    uint8_t* dst = out->data();
    for (uint32_t mt = 0; mt < cout / 16; ++mt) {
        for (uint32_t q = 0; q < nq; ++q) {
            uint32_t cb = 0, ky = 0, kx = 0;
            if (order == FgChunkOrder::Ascending) {
                cb = q;
            } else if (order == FgChunkOrder::BlockKxKy) {
                cb = q / (k * k);
                const uint32_t t = q % (k * k);
                kx = t / k;
                ky = t % k;
            } else {
                kx = q / (nb * k);
                const uint32_t r = q % (nb * k);
                cb = r / k;
                ky = r % k;
            }
            for (uint32_t h = 0; h < 2; ++h)
                for (uint32_t m = 0; m < 16; ++m)
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t o = 16 * mt + m, i = 16 * cb + 8 * h + j;
                        const uint16_t v = w[((static_cast<uint64_t>(o) * k + ky) * k + kx) * cin + i];
                        const uint64_t at = ((((static_cast<uint64_t>(mt) * nq + q) * 2 + h) * 16 + m) * 8 + j) * 2;
                        dst[at] = static_cast<uint8_t>(v & 0xFF);
                        dst[at + 1] = static_cast<uint8_t>(v >> 8);
                    }
        }
    }
    return true;
}

bool FgPackNet1Layer(const uint8_t* bytes, size_t size, const FgModelStage& st, const FgNet1Layer& l,
                     std::vector<uint8_t>* fragments, std::vector<uint8_t>* bias) {
    if (!bytes || !fragments || !bias) return false;
    const bool ok_shape = (l.k == 3 ? st.w.rank == 4 && st.w.shape[2] == 3 && st.w.shape[3] == 3 : st.w.rank == 2) &&
                          st.w.shape[0] == l.cout && st.w.shape[1] == l.cin && st.b.rank == 1 &&
                          st.b.shape[0] == l.cout;
    const uint64_t elems = static_cast<uint64_t>(l.cout) * l.k * l.k * l.cin;
    if (!ok_shape || st.w.bytes != 4 * elems || st.b.bytes != 4ull * l.cout || st.w.offset > size ||
        st.w.bytes > size - st.w.offset || st.b.offset > size || st.b.bytes > size - st.b.offset)
        return false;
    auto at = [bytes](uint64_t off) {
        float f;
        std::memcpy(&f, bytes + off, 4);
        return f;
    };
    std::vector<uint16_t> canon(static_cast<size_t>(elems));
    const uint32_t k = l.k, cin = l.cin;
    for (uint32_t o = 0; o < l.cout; ++o)
        for (uint32_t i = 0; i < cin; ++i)
            for (uint32_t ky = 0; ky < k; ++ky)
                for (uint32_t kx = 0; kx < k; ++kx) {
                    const uint64_t natural = ((static_cast<uint64_t>(o) * cin + i) * k + ky) * k + kx;
                    uint16_t h = 0;
                    if (!FgFloatToHalfExact(at(st.w.offset + 4 * natural), &h)) return false;
                    canon[static_cast<size_t>(((static_cast<uint64_t>(o) * k + ky) * k + kx) * cin + i)] = h;
                }
    bias->assign(2ull * l.cout, 0);
    for (uint32_t o = 0; o < l.cout; ++o) {
        uint16_t h = 0;
        if (!FgFloatToHalfExact(at(st.b.offset + 4ull * o), &h)) return false;
        (*bias)[2 * o] = static_cast<uint8_t>(h & 0xFF);
        (*bias)[2 * o + 1] = static_cast<uint8_t>(h >> 8);
    }
    return FgSwizzleNet1(canon.data(), l.cout, l.k, l.cin, l.order, fragments);
}

}  // namespace wgfg
