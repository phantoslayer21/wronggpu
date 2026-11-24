// SPDX-License-Identifier: MIT
// Frame generation (issue #96, F7b): the D3D12 executor's CPU side. See fg_exec_plan.h.
#include "fg_exec_plan.h"

#include <cmath>
#include <cstring>

namespace wgfg {

namespace {

uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// The float next to the exact product p of two floats, rounded up (the smallest float >= p) or down.
float round_dir(double p, bool up) {
    float f = static_cast<float>(p);
    if (up && static_cast<double>(f) < p) f = std::nextafter(f, INFINITY);
    if (!up && static_cast<double>(f) > p) f = std::nextafter(f, -INFINITY);
    return f;
}

constexpr FgNet2Layer kNet2[kFgNet2Layers] = {
    {16, 10, 32}, {32, 32, 32}, {32, 32, 64},
    {64, 64, 64}, {64, 64, 64}, {64, 64, 64}, {64, 64, 64}, {64, 64, 64}, {64, 64, 64}, {64, 64, 64}, {64, 64, 64},
    {64, 64, 96}, {32, 32, 8},
    {18, 18, 16}, {16, 16, 32},
    {32, 32, 32}, {32, 32, 32}, {32, 32, 32}, {32, 32, 32}, {32, 32, 32}, {32, 32, 32}, {32, 32, 32}, {32, 32, 32},
    {32, 32, 48}, {16, 16, 8},
};

void clear_op(uint32_t role, std::initializer_list<float> values, std::vector<FgExecOp>* ops) {
    FgExecOp o;
    o.op = FgOp::Clear;
    o.role = role;
    uint32_t n = 0;
    for (float v : values) o.value[n++] = FgExecHalfBits(v);
    o.period = n;
    ops->push_back(o);
}

void word_clear_op(uint32_t role, uint32_t word, std::vector<FgExecOp>* ops) {
    FgExecOp o;
    o.op = FgOp::Clear;
    o.role = role;
    o.value[0] = word & 0xFFFFu;
    o.value[1] = word >> 16;
    o.period = 2;
    ops->push_back(o);
}

void simple(FgOp op, std::vector<FgExecOp>* ops, uint32_t role = 0) {
    FgExecOp o;
    o.op = op;
    o.role = role;
    if (op == FgOp::Fill64) {
        o.value[0] = 0xFFFFu;
        o.value[1] = 0x7FFFu;
        o.period = 2;
    }
    ops->push_back(o);
}

}  // namespace

const FgNet2Layer* FgNet2Layers() { return kNet2; }

uint32_t FgExecFloatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

uint16_t FgExecHalfBits(float f) {
    const uint32_t x = FgExecFloatBits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u | ((ax >> 13) & 0x3FFu) : 0u));
    if (ax >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // rounds to infinity
    if (ax < 0x33000001u) return static_cast<uint16_t>(sign);              // below half the smallest subnormal
    const int32_t e = static_cast<int32_t>(ax >> 23);
    uint32_t mant = (ax & 0x7FFFFFu) | 0x800000u;
    int32_t shift;
    uint32_t base;
    if (e < 113) {  // subnormal half
        shift = 126 - e;  // 13 + (113 - e)
        base = 0;
    } else {
        shift = 13;
        base = static_cast<uint32_t>(e - 112) << 10;
        mant &= 0x7FFFFFu;
    }
    const uint32_t q = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    uint32_t r = base + q;
    if (rem > halfway || (rem == halfway && (q & 1u))) ++r;
    return static_cast<uint16_t>(sign | r);
}

float FgExecHalfToFloat(uint16_t h) {
    const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0x1F) {
        bits = s | 0x7F800000u | (m << 13);
    } else if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            int32_t ee = -1;
            do {
                ++ee;
                m <<= 1;
            } while (!(m & 0x400u));
            bits = s | (static_cast<uint32_t>(112 - ee) << 23) | ((m & 0x3FFu) << 13);
        }
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void FgExecThresholds(uint32_t ho, float* t1, float* t2) {
    const volatile float q = static_cast<float>(static_cast<uint64_t>(ho) * ho);
    const volatile float a = 900.0f, b = 10000.0f;
    *t1 = q / a;
    *t2 = q / b;
}

bool FgExecMakeGeometry(uint32_t wo, uint32_t ho, FgExecGeometry* out, std::string* why) {
    if (!(wo >= 4 && wo <= 8192 && ho >= 4 && ho <= 7680)) {
        if (why) *why = "frame size outside 4 x 4 .. 8192 x 7680";
        return false;
    }
    FgExecGeometry g;
    g.wo = wo;
    g.ho = ho;
    g.d = ho > 1440 ? 4 : 2;
    g.wd = wo / g.d;
    g.hd = ho / g.d;
    const uint32_t cw = g.wd / 2, ch = g.hd / 2;
    g.W0 = 32 * cdiv(cw, 32);
    g.H0 = 32 * cdiv(ch, 32);
    if (g.W0 < 32 || g.H0 < 32) {
        if (why) *why = "net1 cell grid below 32 x 32";
        return false;
    }
    g.pad_l = (g.W0 - cw) / 2;
    g.pad_t = (g.H0 - ch) / 2;
    g.dp = ho > 2880 ? 4 : 2;
    g.w1 = wo / g.dp;
    g.h1 = ho / g.dp;
    g.lw.push_back(g.w1);
    g.lh.push_back(g.h1);
    while (g.lw.size() < 12 && !(g.lw.size() >= 9 && g.lw.back() <= 1 && g.lh.back() <= 1)) {
        const uint32_t a = g.lw.back(), b = g.lh.back();
        g.lw.push_back(cdiv(a, 2));
        g.lh.push_back(cdiv(b, 2));
    }
    const uint32_t n = static_cast<uint32_t>(g.lw.size());
    g.top = (n > 9 ? 8u : 4u) < n - 1 ? (n > 9 ? 8u : 4u) : n - 1;
    g.k = 1;
    while (ho / g.k > 480) g.k *= 2;
    const uint32_t q = 16 * g.k;
    g.Wp = q * cdiv(wo, q);
    g.Hp = q * cdiv(ho, q);
    g.W2 = g.Wp / g.k;
    g.H2 = g.Hp / g.k;
    uint32_t w = wo, h = ho, levels = 1;
    while (levels < 12 && (levels < 9 || w > 1 || h > 1)) {
        w = cdiv(w, 2);
        h = cdiv(h, 2);
        ++levels;
    }
    g.out_levels = levels;
    g.out_coarsest = levels > 9 ? 8 : 4;
    *out = g;
    return true;
}

bool FgExecMakeRegion(uint32_t tex_w, uint32_t tex_h, uint32_t bx, uint32_t by, uint32_t ew, uint32_t eh,
                      FgExecRegion* out) {
    if (ew < 1 || eh < 1 || bx + ew > tex_w || by + eh > tex_h || tex_w > (1u << 24) || tex_h > (1u << 24))
        return false;
    FgExecRegion r;
    r.tex_w = tex_w;
    r.tex_h = tex_h;
    r.bx = bx;
    r.by = by;
    r.ew = ew;
    r.eh = eh;
    const volatile float one = 1.0f;
    r.rx = one / static_cast<float>(tex_w);
    r.ry = one / static_cast<float>(tex_h);
    const float bx5 = static_cast<float>(bx) + 0.5f, by5 = static_cast<float>(by) + 0.5f;
    const float ex5 = static_cast<float>(bx + ew) - 0.5f, ey5 = static_cast<float>(by + eh) - 0.5f;
    r.lox = round_dir(static_cast<double>(bx5) * static_cast<double>(r.rx), true);
    r.hix = round_dir(static_cast<double>(ex5) * static_cast<double>(r.rx), false);
    r.loy = round_dir(static_cast<double>(by5) * static_cast<double>(r.ry), true);
    r.hiy = round_dir(static_cast<double>(ey5) * static_cast<double>(r.ry), false);
    *out = r;
    return true;
}

void FgExecRegion::Args(uint32_t o[10]) const {
    const float v[10] = {static_cast<float>(ew), static_cast<float>(eh), static_cast<float>(bx),
                         static_cast<float>(by), rx, ry, lox, hix, loy, hiy};
    for (int i = 0; i < 10; ++i) o[i] = FgExecFloatBits(v[i]);
}

const char* FgRoleName(uint32_t role) {
    static const char* const kNames[kRoleP1] = {"Cc", "Cp", "Lc", "Lp", "Uc", "Up", "Dc", "Dp", "Ld", "m", "Um",
                                                "Ump", "Hc", "Hp", "Om", "Qc", "Qp", "NA", "NB", "Mv", "Vf", "Zc",
                                                "Zp", "Fl", "G", "Kc", "Kp", "E", "F", "W", "S", "MA", "MB"};
    if (role < kRoleP1) return kNames[role];
    static const char* const kLevels[11] = {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"};
    return role < kRoleCount ? kLevels[role - kRoleP1] : "?";
}

uint64_t FgRoleBytes(const FgExecGeometry& g, uint32_t role) {
    const uint64_t fr = static_cast<uint64_t>(g.wo) * g.ho, nd = static_cast<uint64_t>(g.wd) * g.hd,
                   sc = static_cast<uint64_t>(g.w1) * g.h1;
    switch (role) {
        case kRoleCc: case kRoleCp: case kRoleLc: case kRoleLp: case kRoleUc: case kRoleUp: return fr * 8;
        case kRoleDc: case kRoleDp: return nd * 8;
        case kRoleLd: case kRoleM: case kRoleUm: case kRoleUmp: case kRoleHc: case kRoleHp: case kRoleOm: return nd * 2;
        case kRoleQc: case kRoleQp: case kRoleNA: case kRoleNB: return nd * 4;
        case kRoleMv: case kRoleVf: return sc * 8;
        case kRoleZc: case kRoleZp: case kRoleFl: case kRoleG: return sc * 4;
        case kRoleKc: case kRoleKp: case kRoleE: return sc * 2;
        case kRoleF: case kRoleW: return sc * 4;
        case kRoleS: case kRoleMA: case kRoleMB: return sc * 8;
        default: break;
    }
    if (role >= kRoleP1 && role < kRoleCount) {
        const uint32_t l = role - kRoleP1 + 1;
        if (l > g.top) return 0;
        return static_cast<uint64_t>(g.lw[l]) * g.lh[l] * 6;
    }
    return 0;
}

FgExecFlags FgExecDerivedFlags(const FgExecSettings& st, bool lite, bool ui_recomposition, bool hudless, bool ui4,
                               bool ui1, bool distortion) {
    FgExecFlags f;
    f.distortion = distortion && !st.ignore_distortion;
    f.hudless = hudless && !st.ignore_hudless;
    f.ui_alpha = (ui4 || ui1) && !st.ignore_ui;
    f.ui_recomp = !lite && f.hudless && f.ui_alpha && ui_recomposition;
    f.lite1 = lite && st.lite_l1 && f.hudless && f.ui_alpha;
    f.lite2 = lite && st.lite_l2 && f.hudless && !f.lite1;
    return f;
}

uint32_t FgExecR3Bits(const FgExecFlags& f) {
    const bool f1 = f.ui_recomp || f.lite1;
    return (f1 ? 1u : 0u) | (f.ui_alpha ? 2u : 0u) | (f.hudless ? 4u : 0u) | (f.lite1 ? 8u : 0u) |
           (f.lite2 ? 16u : 0u);
}

char FgExecR3Case(const FgExecFlags& f) {
    if (f.ui_recomp || f.lite1) return 'U';
    if (f.ui_alpha) return 'A';
    return f.hudless ? 'H' : '0';
}

void FgInterleaveLaunches(const std::vector<std::vector<FgLaunchKey>>& frames, std::vector<FgLaunchSlot>* out) {
    out->clear();
    if (frames.empty()) return;
    bool same = true;
    for (size_t f = 1; f < frames.size() && same; ++f) {
        same = frames[f].size() == frames[0].size();
        for (size_t j = 0; same && j < frames[0].size(); ++j)
            same = frames[f][j].kernel == frames[0][j].kernel && frames[f][j].barrier == frames[0][j].barrier;
    }
    if (!same) {
        for (uint32_t f = 0; f < frames.size(); ++f)
            for (uint32_t j = 0; j < frames[f].size(); ++j) out->push_back({f, j, frames[f][j].barrier, true});
        return;
    }
    for (uint32_t j = 0; j < frames[0].size(); ++j)
        for (uint32_t f = 0; f < frames.size(); ++f) out->push_back({f, j, f == 0 && frames[0][j].barrier, f == 0});
}

const char* FgOpName(FgOp op) {
    switch (op) {
        case FgOp::Clear: return "clear";
        case FgOp::Detect: return "detect";
        case FgOp::SetDisable: return "set_disable";
        case FgOp::Downsample: return "downsample";
        case FgOp::Mvec: return "mvec";
        case FgOp::Fill64: return "fill64";
        case FgOp::P2cScatter: return "p2c_scatter";
        case FgOp::P2cUnpack: return "p2c_unpack";
        case FgOp::P2cInpaint: return "p2c_inpaint";
        case FgOp::P2cPull: return "p2c_pull";
        case FgOp::P2cPush: return "p2c_push";
        case FgOp::HoleMask: return "hole_mask";
        case FgOp::QualityInit: return "quality_init";
        case FgOp::Net1Input: return "net1_input";
        case FgOp::Net1: return "net1";
        case FgOp::Net1Output: return "net1_output";
        case FgOp::Interm: return "interm";
        case FgOp::Weights: return "weights";
        case FgOp::Prefill: return "prefill";
        case FgOp::LiteBlend: return "lite_blend";
        case FgOp::FullBlend: return "full_blend";
        case FgOp::Net2: return "net2";
        case FgOp::OutPyramid: return "out_pyramid";
        case FgOp::MenuOutput: return "menu_output";
        case FgOp::FinalCopy: return "final_copy";
    }
    return "?";
}

void FgExecHost::BeginFrame(bool lite, bool ui_recomposition, bool hudless, bool ui4, bool ui1, bool reset,
                            std::vector<FgExecOp>* ops) {
    lite_ = lite;
    const FgExecFlags f = FgExecDerivedFlags(st_, lite, ui_recomposition, hudless, ui4, ui1, false);
    if (reset || (have_prev_ && f != prev_flags_)) {
        history_ = 0;
        have_prev_ = false;
    }
    frame_ = FgExecFrameInfo();
    frame_.flags = f;
    frame_.history = history_ > 0;
    frame_.r3_bits = FgExecR3Bits(f);
    frame_.r3_case = FgExecR3Case(f);
    const bool hist = frame_.history;
    if (!hist) {
        clear_op(kRoleCp, {0.0f, 0.0f, 0.0f, 1.0f}, ops);
        clear_op(kRoleKp, {1.0f}, ops);
        word_clear_op(kRoleZp, 0u, ops);
        clear_op(kRoleM, {0.5f}, ops);
        if (frame_.r3_bits & 1u) {
            clear_op(kRoleUc, {0.0f, 0.0f, 0.0f, 1.0f}, ops);
            clear_op(kRoleUp, {0.0f, 0.0f, 0.0f, 1.0f}, ops);
        }
    }
    const uint32_t base = hist ? 0u : 1u;
    int kind = -1;
    if (st_.menu_detection) kind = ui4 ? 0 : (hudless ? 1 : -1);
    FgExecOp d;
    if (kind >= 0) {
        d.op = FgOp::Detect;
        d.detect_kind = static_cast<uint32_t>(kind);
        d.hysteresis_off = !hist && !detect_ran_prev_;
        d.base = base;
    } else {
        d.op = FgOp::SetDisable;
        d.base = base;
    }
    ops->push_back(d);
    detect_ran_prev_ = kind >= 0;
    frame_.detect_ran = kind >= 0;
    simple(FgOp::Downsample, ops);
    simple(FgOp::Mvec, ops);
    simple(FgOp::Fill64, ops, kRoleS);
    simple(FgOp::P2cScatter, ops);
    simple(FgOp::P2cUnpack, ops);
    simple(FgOp::P2cInpaint, ops);
    simple(FgOp::P2cPull, ops);
    simple(FgOp::P2cPush, ops);
    simple(FgOp::HoleMask, ops);
    if (hist) {
        simple(FgOp::Net1Input, ops);
        simple(FgOp::Net1, ops);
        FgExecOp o;
        o.op = FgOp::Net1Output;
        o.ui_forcing = (f.hudless || f.ui_alpha) && !(frame_.r3_bits & 1u);
        ops->push_back(o);
    } else {
        simple(FgOp::QualityInit, ops);
    }
    in_frame_ = true;
}

void FgExecHost::IndexOps(float t, bool hudless_now, bool ui4_now, bool out_subrect, std::vector<FgExecOp>* ops) const {
    const FgExecFlags& f = frame_.flags;
    bool wrote = false;
    if (frame_.history) {
        simple(FgOp::Fill64, ops, kRoleMA);
        simple(FgOp::Fill64, ops, kRoleMB);
        FgExecOp o;
        o.op = FgOp::Interm;
        o.t = t;
        ops->push_back(o);
        simple(FgOp::Weights, ops);
        if (out_subrect) simple(FgOp::Prefill, ops);
        if (lite_) {
            FgExecOp b;
            b.op = FgOp::LiteBlend;
            b.t = t;
            b.flag1 = f.lite1;
            b.flag2 = f.lite2;
            ops->push_back(b);
        } else {
            FgExecOp b;
            b.op = FgOp::FullBlend;
            b.ui = f.ui_recomp;
            b.t = t;
            ops->push_back(b);
            FgExecOp n;
            n.op = FgOp::Net2;
            n.t = t;
            ops->push_back(n);
            simple(FgOp::OutPyramid, ops);
        }
        wrote = true;
        if (st_.menu_detection && (ui4_now || hudless_now)) simple(FgOp::MenuOutput, ops);
    } else if (out_subrect) {
        simple(FgOp::Prefill, ops);
        wrote = true;
    }
    if (!wrote) simple(FgOp::FinalCopy, ops);
}

void FgExecHost::EndFrame(std::vector<uint32_t>* pairs) {
    if (!in_frame_) return;
    const FgExecFlags& f = frame_.flags;
    ++history_;
    have_prev_ = true;
    prev_flags_ = f;
    const uint32_t base_pairs[6][2] = {{kRoleCc, kRoleCp}, {kRoleDc, kRoleDp}, {kRoleZc, kRoleZp},
                                       {kRoleKc, kRoleKp}, {kRoleQc, kRoleQp}, {kRoleHc, kRoleHp}};
    for (const auto& p : base_pairs) {
        pairs->push_back(p[0]);
        pairs->push_back(p[1]);
    }
    if (f.lite2) {
        pairs->push_back(kRoleLc);
        pairs->push_back(kRoleLp);
    }
    if (f.ui_recomp || f.lite1) {
        pairs->push_back(kRoleUc);
        pairs->push_back(kRoleUp);
    }
    if (frame_.r3_case == 'A' || frame_.r3_case == 'H') {
        pairs->push_back(kRoleUm);
        pairs->push_back(kRoleUmp);
    }
    in_frame_ = false;
}

}  // namespace wgfg
