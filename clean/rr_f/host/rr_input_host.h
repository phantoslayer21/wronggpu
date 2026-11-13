// SPDX-License-Identifier: MIT
// Host side of the spec 52 input stage: the per-frame constants (52.1.1, 52.8), the camera maps of 52.7.3 and the
// packing of the constant block that clean/rr_f/kernels/rr_input.hlsl reads (layout K_* there and in
// clean/rr_f/gpu/input_stage.py, which is the reference this header is tested against: tests/test_input_host.py).
//
// Arithmetic: binary32, round to nearest even, subnormal operands and results flushed to zero (sign kept) after the
// IEEE rounding. Compile with SSE arithmetic (x64), /fp:strict (no contraction, no reassociation) and the default
// MXCSR (no hardware flush): the explicit ftz below then gives "round, then flush".
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace rr_input {

inline float bits_f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline uint32_t f_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float ftz(float x) { const uint32_t u = f_bits(x); return (u & 0x7f800000u) == 0u ? bits_f(u & 0x80000000u) : x; }
inline float add(float a, float b) { return ftz(ftz(a) + ftz(b)); }
inline float sub(float a, float b) { return ftz(ftz(a) - ftz(b)); }
inline float mul(float a, float b) { return ftz(ftz(a) * ftz(b)); }
inline float divf(float a, float b) { return ftz(ftz(a) / ftz(b)); }
inline float rcp(float x) { return divf(1.0f, x); }
// fl32(a / b) of integers: the binary64 quotient (correctly rounded) narrowed once more; innocuous for 24-bit operands
inline float q32(int64_t a, int64_t b) { return ftz((float)((double)a / (double)b)); }

// round a binary32 value to binary16 (nearest even, subnormals kept, overflow to infinity) and widen exactly
inline float f16rt(float x)
{
    if (std::isnan(x) || std::isinf(x)) return x;
    const double a = std::fabs((double)x);
    if (a >= 65520.0) return std::copysign(INFINITY, x);
    int e = 0;
    std::frexp(a, &e);
    const int q = a >= 0x1p-14 ? e - 11 : -24;
    const double r = std::nearbyint(std::ldexp(a, -q));      // exact scaling, then ties to even
    return (float)std::copysign(std::ldexp(r, q), (double)x);
}

struct Mat { float m[4][4]; };

inline Mat matmul(const Mat& A, const Mat& B)
{
    Mat o;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = mul(A.m[i][0], B.m[0][j]);
            for (int k = 1; k < 4; ++k) s = add(s, mul(A.m[i][k], B.m[k][j]));
            o.m[i][j] = s;
        }
    return o;
}

// I32 of 52.7.3: rounded cofactor inverse; false when some row expansion det[i] is zero
inline bool inv4(const Mat& A, Mat& out)
{
    float cof[4][4];
    for (int i = 0; i < 4; ++i) {
        int rows[3], n = 0;
        for (int r = 0; r < 4; ++r) if (r != i) rows[n++] = r;
        for (int j = 0; j < 4; ++j) {
            int cols[3], m = 0;
            for (int c = 0; c < 4; ++c) if (c != j) cols[m++] = c;
            const float* a = A.m[rows[0]];
            const float* b = A.m[rows[1]];
            const float* c = A.m[rows[2]];
            auto E = [&](int p, int q) { return sub(mul(b[cols[p]], c[cols[q]]), mul(b[cols[q]], c[cols[p]])); };
            const float det = add(sub(mul(a[cols[0]], E(1, 2)), mul(a[cols[1]], E(0, 2))), mul(a[cols[2]], E(0, 1)));
            cof[i][j] = ((i + j) & 1) ? -det : det;
        }
    }
    float dets[4];
    for (int i = 0; i < 4; ++i) {
        float s = 0.0f;
        for (int j = 0; j < 4; ++j) s = add(s, mul(A.m[i][j], cof[i][j]));
        dets[i] = s;
    }
    for (int i = 0; i < 4; ++i) if (dets[i] == 0.0f) return false;
    const float r = rcp(dets[3]);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out.m[j][i] = mul(cof[i][j], r);
    return true;
}

// Mpos, Mrep of 52.7.3; false when the frame has no usable camera matrices
inline bool camera_maps(const Mat& V2C, const Mat& W2V, const Mat& V2Cp, const Mat& W2Vp, int Wr, int Hr,
                        Mat& Mpos, Mat& Mrep)
{
    Mat Pc, Pp;
    if (!inv4(W2V, Pc) || !inv4(W2Vp, Pp)) return false;
    Mat Rc = Pc, Rp = Pp;
    for (int c = 0; c < 3; ++c) { Rc.m[3][c] = 0.0f; Rp.m[3][c] = sub(Pp.m[3][c], Pc.m[3][c]); }
    Mat iRc, iRp;
    if (!inv4(Rc, iRc) || !inv4(Rp, iRp)) return false;
    const Mat Am = matmul(iRc, V2C), Bm = matmul(iRp, V2Cp);
    Mat iAm;
    if (!inv4(Am, iAm)) return false;
    const Mat Tcc = matmul(iAm, Bm);
    Mat K = {};
    for (int i = 0; i < 4; ++i) K.m[i][i] = 1.0f;
    K.m[0][0] = K.m[3][0] = (float)Wr * 0.5f;
    K.m[1][1] = -(float)Hr * 0.5f;
    K.m[3][1] = (float)Hr * 0.5f;
    if (!inv4(matmul(Am, K), Mpos)) return false;
    Mrep = matmul(matmul(Am, Tcc), K);
    return true;
}

// roles of the texture table (kernels/rr_input.hlsl R_*)
enum Role { R_C, R_Z, R_V, R_SV, R_DD, R_DS, R_N, R_R, R_Q, R_B, R_G1, R_G2, R_HD, ROLE_COUNT };
// symbols in block order (gpu/input_stage.py SYMS)
enum Sym { S_K4, S_K5, S_K7, S_K7N, S_K8, S_K10, S_K12, S_HDEC, S_FCLIP, S_AMAX, S_GCL, S_GCLN, S_LWRB, S_LWG,
           S_MDL, S_MDLN, S_GM, S_GD, S_GH, S_GS, S_SV1, S_SV2, S_SV3, S_SV4, S_SV4N, SYM_COUNT };

struct Frame {
    int Wr, Hr, Wo, Ho;                 // render size of this call, creation-time output size
    float jx, jy, pjx, pjy;             // jitter, previous jitter pj (53.5)
    float mv_raw_x, mv_raw_y;           // normalised motion-vector scale
    bool mv_jittered;                   // creation decision
    float E, Ep, X3;                    // page 58 (or read on the GPU from the exposure result)
    bool reset;
    int par, fc;
    uint32_t SH[2][11][2];
    bool hw_depth, depth_inverted, nrm_dec, rgh_packed, specular_from_v;
    bool have_camera;
    Mat V2C, W2V, V2Cp, W2Vp;           // V2Cp / W2Vp: the stored CP pair (replaced by the current pair on reset)
    bool attached[ROLE_COUNT];
    bool b2;                            // constants_b2 present
    float sym[SYM_COUNT];               // C_SV4 / C_SV4n only with b2
    bool ms_given;                      // the runtime (adapter): ms = (scale.x Wr, scale.y Hr) already formed by the
    float ms_x, ms_y;                   // proxy (52.8); otherwise ms = mul(mv_raw, Wr / Hr) below
    bool hd_filter = false;           // WG_RR_SET_HIT_DISTANCE_FILTER (opt-in): hit distance = median of its 1x4 run
};

struct Texture { uint32_t offset, width, height, channels, ox, oy; };

// words 0 .. 99 of the constant block; returns whether the camera maps are usable
inline bool scalar_words(const Frame& f, uint32_t w[100])
{
    std::memset(w, 0, 100 * sizeof(uint32_t));
    const int Wr = f.Wr, Hr = f.Hr;
    const int Wh = 3 * (f.Wo + f.Wo % 2) / 2, Hh = 3 * (f.Ho + f.Ho % 2) / 2;
    const int Wt = 32 * ((Wr + 31) / 32), Ht = 32 * ((Hr + 31) / 32);
    const int par = f.par & 1;
    Mat Mpos = {}, Mrep = {};
    bool usable = false;
    if (f.have_camera) {
        const Mat& V2Cp = (f.reset ? f.V2C : f.V2Cp);
        const Mat& W2Vp = (f.reset ? f.W2V : f.W2Vp);
        usable = camera_maps(f.V2C, f.W2V, V2Cp, W2Vp, Wr, Hr, Mpos, Mrep);
    }
    uint32_t mode = 0;
    if (f.attached[R_SV]) mode = 1;
    else if (f.attached[R_HD] && !f.specular_from_v && usable) mode = 2;
    uint32_t flags = (f.reset ? 1u : 0u) | ((f.hw_depth && f.depth_inverted) ? 2u : 0u) | (f.hw_depth ? 4u : 0u) |
                     (f.nrm_dec ? 8u : 0u) | (f.rgh_packed ? 16u : 0u) | (mode << 5) | (f.b2 ? 128u : 0u) |
                     (f.attached[R_Q] ? 256u : 0u) | (f.attached[R_B] ? 512u : 0u) | (f.attached[R_G1] ? 1024u : 0u) |
                     (f.attached[R_G2] ? 2048u : 0u) | ((f.have_camera && f.V2C.m[3][3] < 1.0f) ? 4096u : 0u) |
                     (f.hd_filter ? 8192u : 0u);
    const uint32_t ints[10] = {(uint32_t)Wr, (uint32_t)Hr, (uint32_t)Wt, (uint32_t)Ht, (uint32_t)Wh, (uint32_t)Hh,
                               f.SH[par][0][0], f.SH[par][0][1], (uint32_t)(f.fc & 7), flags};
    for (int i = 0; i < 10; ++i) w[i] = ints[i];
    // 52.8 motion and jitter
    const float ms[2] = {f.ms_given ? f.ms_x : mul(f.mv_raw_x, (float)Wr), f.ms_given ? f.ms_y : mul(f.mv_raw_y, (float)Hr)};
    const float j[2] = {f.jx, f.jy}, pj[2] = {f.pjx, f.pjy};
    float mo[2];
    for (int a = 0; a < 2; ++a) {
        float off = 0.0f;
        if (f.mv_jittered) off = add(off, f16rt(sub(j[a], pj[a])));
        mo[a] = f16rt(divf(off, ms[a]));
    }
    const float rWrWh = q32(Wr, Wh), rHrHh = q32(Hr, Hh), invWr = q32(1, Wr), invHr = q32(1, Hr);
    const float vals[28] = {
        f.E, 0.0f, f.X3, f.jx, f.jy, f.pjx, f.pjy, mo[0], mo[1], divf(ms[0], rWrWh), divf(ms[1], rHrHh),
        q32(Wh, Wr), q32(Hh, Hr), rWrWh, rHrHh, invWr, invHr, q32(1, Wh), q32(1, Hh),
        q32(1, 2 * ((Wh + 1) / 2)), q32(1, 2 * ((Hh + 1) / 2)), mul(f.pjx, invWr), mul(f.pjy, invHr),
        f.depth_inverted ? 0.0f : 1.0f,
        f.have_camera ? f.V2C.m[2][2] : 0.0f, f.have_camera ? f.V2C.m[3][2] : 0.0f,
        f.have_camera ? f.V2C.m[2][3] : 0.0f, rcp(f.Ep)};
    for (int i = 0; i < 28; ++i) w[10 + i] = f_bits(vals[i]);
    w[11] = 0u;                                               // word 11 unused
    if (usable)
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) { w[40 + 4 * r + c] = f_bits(Mpos.m[r][c]); w[56 + 4 * r + c] = f_bits(Mrep.m[r][c]); }
    for (int s = 0; s < SYM_COUNT; ++s)
        if (s < S_SV4 || f.b2) w[72 + s] = f_bits(f.sym[s]);
    return usable;
}

// texture table entries (words 100 + 8 r): the reader origin of 52.1.2 (SV: V's origin; B, G1, G2: C's origin)
inline void texture_words(const Frame& f, const Texture tex[ROLE_COUNT], uint32_t w[8 * ROLE_COUNT])
{
    std::memset(w, 0, 8 * ROLE_COUNT * sizeof(uint32_t));
    for (int r = 0; r < ROLE_COUNT; ++r) {
        if (!f.attached[r]) continue;
        const int src = r == R_SV ? R_V : (r == R_B || r == R_G1 || r == R_G2) ? R_C : r;
        const uint32_t e[7] = {tex[r].offset, tex[r].width, tex[r].height, tex[r].channels, tex[src].ox, tex[src].oy, 1u};
        for (int i = 0; i < 7; ++i) w[8 * r + i] = e[i];
    }
}

// Tables of the kernel's float-float log2 / exp2 paths (gpu/ffmath.py is the reference): log2 [256][4] = (c, Th, Tl, 0)
// with c = 1 / bucket centre rounded to 8 significant bits (c = 1 for the two buckets that contain 1) and Th + Tl =
// -log2(c); exp2 [65][2] = 2^(j/64), j = -32 .. 32. Binary64 library values split into two f32 are accurate to about
// 2^-50 relative, well inside the kernel's flag threshold; the kernel's results do not depend on the last bits.
inline float round8(double q)
{
    int e = 0;
    std::frexp(q, &e);                                   // q = f 2^e, f in [0.5, 1)
    return (float)std::ldexp(std::nearbyint(std::ldexp(q, 8 - e)), e - 8);
}
inline void ff_tables(float logt[256][4], float expt[65][2])
{
    for (int h = 0; h < 2; ++h)
        for (int i = 0; i < 128; ++i) {
            double centre = 1.0 + (2.0 * i + 1.0) / 256.0;
            if (h) centre *= 0.5;
            const bool one = (h == 0 && i == 0) || (h == 1 && i == 127);
            const float c = one ? 1.0f : round8(1.0 / centre);
            const double T = -std::log2((double)c);
            const float Th = (float)T;
            float* e = logt[h * 128 + i];
            e[0] = c; e[1] = Th; e[2] = (float)(T - (double)Th); e[3] = 0.0f;
        }
    for (int j = -32; j <= 32; ++j) {
        const double v = std::exp2(j / 64.0);
        expt[j + 32][0] = (float)v;
        expt[j + 32][1] = (float)(v - (double)(float)v);
    }
}

}  // namespace rr_input
