// SPDX-License-Identifier: MIT
// Page 54b on D3D12: output resolve (plain / extended: transparency layer with the colour-before-particles rule,
// opacity layer, alpha upscaling), quarter pooled history, and probes of the exact elementary functions.
// Own implementation of the final page 54b (with errata 50-53); reference: clean/rr_f/reference/resolve.py
// (bit-identical with RR_ELEM_EXACT=1). No model constant is embedded: every symbol comes from the parameter block.
//
// Parameter block (prm, uint words; layout = clean/rr_f/gpu/resolve.py PARAMS):
//   0 We  1 He  2 Wh  3 Hh  4 Wr  5 Hr  6 Wo  7 Ho  8 flags  9 E  10 Ep  11 jx  12 jy  13 lbx  14 lby
//   17..21 w_even[-2..2] (binary16 bits: RW_B RW_D RW_0 RW_C RW_A)  22 K1 23 K2n 24 K2x2n 25 K4 26 K5 27 K7 28 K8
//   29 K10 30 K11 31 K12 32 K13 33 C_HDEC 34 T_EFL 35 TBW 36 alpha-source channel
//   resources, 8 words each (offset, width, height, pitch in texels, format, channels, origin x, origin y):
//   40 Q 48 OUT 56 AOUT 64 LAYER 72 OPAC 80 ASRC 88 MV 96 P 104 G 112 THr 120 THw 128 OHr 136 OHw 144 AHr 152 AHw
//   160 PH1 168 PQ   (offset 0 = not attached; the motion origin is MV's origin)
// Formats: 0 f32, 1 f16, 2 unorm8, 3 rgb10a2 (packed 32-bit word), 4 rgb10a2 as four uint16 codes.
// flags: 1 extended, 2 layer, 4 opacity used, 8 alpha upscaling, 16 separate output alpha, 32 transparency reset
//        (reset or history created this frame), 64 alpha reset (reset or alpha history created), 128 bounded stores,
//        256 E / Ep read from the exposure pass's result buffer at the arena offset in word 15 (E, Ep, invE, X3).
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
 uint prm, aux, n;
WG_ARGS_END
#include "rr_resolve_math.hlsli"

// Negative controls (gpu/resolve.py --mutation): each must make the bit-exact qualification fail.
#ifndef RR_MUT
#define RR_MUT 0      // 1: read the history copy written this frame; 2: half-way test with <=; 3: f32-rounded fraction
#endif

uint PU(uint i) { return WaveReadLaneFirst(wg_ld32(prm, i * 4u)); }
float PF(uint i) { return asfloat(PU(i)); }

struct Res { uint off, w, h, pitch, fmt, ch; int ox, oy; };
Res res(uint b)
{
    Res r;
    r.off = PU(b); r.w = PU(b + 1u); r.h = PU(b + 2u); r.pitch = PU(b + 3u);
    r.fmt = PU(b + 4u); r.ch = PU(b + 5u); r.ox = (int)PU(b + 6u); r.oy = (int)PU(b + 7u);
    return r;
}

float cv(uint code, uint maxc) { return wg_div((float)code, (float)maxc); }   // unorm read: IEEE division, one rounding

// Integer fetch with the 54b.N read conversion (f32 values are flushed: every use is an f32 operation);
// outside the texture every channel reads 0; missing channels read 0 (alpha: 1).
float4 texel(Res r, int x, int y)
{
    float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (x < 0 || y < 0 || x >= (int)r.w || y >= (int)r.h) return v;
    uint idx = (uint)y * r.pitch + (uint)x;
    if (r.fmt == 3u) {
        uint wv = wg_ld32(r.off, idx * 4u);
        return float4(cv(wv & 1023u, 1023u), cv((wv >> 10) & 1023u, 1023u), cv((wv >> 20) & 1023u, 1023u), cv(wv >> 30, 3u));
    }
    if (r.fmt == 4u) {
        uint b = r.off + idx * 8u;
        return float4(cv(wg_ld16(b, 0u), 1023u), cv(wg_ld16(b, 2u), 1023u), cv(wg_ld16(b, 4u), 1023u), cv(wg_ld16(b, 6u), 3u));
    }
    [unroll] for (uint c = 0; c < 4u; ++c) {
        float t = c == 3u ? 1.0f : 0.0f;
        if (c < r.ch) {
            uint e = idx * r.ch + c;
            if (r.fmt == 0u) t = ftz(asfloat(wg_ld32(r.off, e * 4u)));
            else if (r.fmt == 1u) t = f16tof32(wg_ld16(r.off, e * 2u));
            else t = cv(wg_ld8(r.off, e), 255u);
        }
        v[c] = t;
    }
    return v;
}

// 8-bit unorm read as binary16 (54b.N, 55.3): code / 255 rounded once to binary16.
float g_half(Res r, int x, int y, uint c)
{
    uint e = ((uint)y * r.pitch + (uint)x) * r.ch + c;
    return H(wg_div((float)wg_ld8(r.off, e), 255.0f));      // f32 then binary16: 24 >= 2*11 + 2, no double rounding
}

// ---- stores (54b.N conversions) -------------------------------------------------------------------------------------
uint unorm_code(float v, uint maxc)
{
    if (nanf(v)) return 0u;
    double s = (double)min(max(v, 0.0f), 1.0f) * (double)maxc;   // exact
    uint a = (uint)s; double f = s - (double)a;
    return min(maxc, a + ((f > 0.5 || (f == 0.5 && (a & 1u) != 0u)) ? 1u : 0u));
}
void st_byte(uint addr, uint v)
{
    uint word = addr & ~3u, sh = (addr & 3u) * 8u, old;
    wg_mem.InterlockedAnd(word, ~(255u << sh), old);
    wg_mem.InterlockedOr(word, (v & 255u) << sh, old);
}
void store_px(Res r, int x, int y, float4 v)
{
    if (r.off == 0u || x < 0 || y < 0 || x >= (int)r.w || y >= (int)r.h) return;
    uint idx = (uint)y * r.pitch + (uint)x;
    if (r.fmt == 3u) {
        wg_st32(r.off, idx * 4u, unorm_code(v.x, 1023u) | (unorm_code(v.y, 1023u) << 10) | (unorm_code(v.z, 1023u) << 20) | (unorm_code(v.w, 3u) << 30));
        return;
    }
    if (r.fmt == 4u) {
        uint b = r.off + idx * 8u;
        wg_st16(b, 0u, unorm_code(v.x, 1023u)); wg_st16(b, 2u, unorm_code(v.y, 1023u));
        wg_st16(b, 4u, unorm_code(v.z, 1023u)); wg_st16(b, 6u, unorm_code(v.w, 3u));
        return;
    }
    if (r.fmt == 2u && r.ch == 4u) {
        wg_st32(r.off, idx * 4u, unorm_code(v.x, 255u) | (unorm_code(v.y, 255u) << 8) | (unorm_code(v.z, 255u) << 16) | (unorm_code(v.w, 255u) << 24));
        return;
    }
    [unroll] for (uint c = 0; c < 4u; ++c) {
        if (c < r.ch) {
            uint e = idx * r.ch + c;
            if (r.fmt == 0u) wg_st32(r.off, e * 4u, asuint(v[c]));
            else if (r.fmt == 1u) wg_st16(r.off, e * 2u, wg_f2h(v[c]));
            else st_byte(r.off + e, unorm_code(v[c], 255u));
        }
    }
}

// ---- the bilinear sample of page 52.1.4 (exact fractions; for s in [-1/2, 0) the fraction 1 + s is applied
// through one fma: (1 + s) T rounded once = fma(s, T, T)) ----------------------------------------------------------
float wterm(float g, float s, bool neg, float t)
{
    if (!neg) return M(g, t);
    return (t == 0.0f || !isfinite(t)) ? t : F(s, t, t);   // (1 + s) t with 1 + s in (1/2, 1]: zeros, infinities, NaN kept
}
float4 bilin(Res r, float u, float v)
{
    u = nanf(u) ? 0.0f : (u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u));
    v = nanf(v) ? 0.0f : (v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
    float s = A(M(u, (float)r.w), -0.5f), t = A(M(v, (float)r.h), -0.5f);
    float i0 = floor(s), j0 = floor(t);
#if RR_MUT == 3
    bool nx = false, ny = false;
#else
    bool nx = s < 0.0f, ny = t < 0.0f;
#endif
    float fxv = nx ? 0.0f : A(s, -i0), fyv = ny ? 0.0f : A(t, -j0);
    float gx = nx ? -s : A(1.0f, -fxv), gy = ny ? -t : A(1.0f, -fyv);
    int xa = clamp((int)i0, 0, (int)r.w - 1), xb = clamp((int)i0 + 1, 0, (int)r.w - 1);
    int ya = clamp((int)j0, 0, (int)r.h - 1), yb = clamp((int)j0 + 1, 0, (int)r.h - 1);
    float4 t00 = texel(r, xa, ya), t10 = texel(r, xb, ya), t01 = texel(r, xa, yb), t11 = texel(r, xb, yb);
    float4 o;
    [unroll] for (uint c = 0; c < 4u; ++c) {
        float r0 = A(M(gx, t00[c]), wterm(fxv, s, nx, t10[c]));
        float r1 = A(M(gx, t01[c]), wterm(fxv, s, nx, t11[c]));
        o[c] = A(M(gy, r0), wterm(fyv, t, ny, r1));
    }
    return o;
}

float DIV(float a, float b) { return rounded((double)ftz(a) / (double)ftz(b)); }   // f32 division (finite, b != 0)
float RND(float v)                                        // 54b.3 rnd: nearest integer, halves away from zero
{
    float t = trunc(v), f = A(v, -t);
    return abs(f) >= 0.5f ? A(t, v < 0.0f ? -1.0f : 1.0f) : t;
}
float clampHist(float h, float mn, float mx) { return (h > mn || nanf(h) || nanf(mn)) ? (h >= mx ? mx : h) : mn; }

// ---- 54b.1 -----------------------------------------------------------------------------------------------------------
float DECODE(float c)                                      // (exp2((c'*K8)*K9) - 1) * K12
{
    float cp = (c > PF(26) || nanf(c)) ? min_nan(c, PF(29)) : PF(26);
    return M(A(X(M(M(cp, PF(28)), RR_K9)), -1.0f), PF(31));
}
float3 plain_col(int x, int y)
{
    Res q = res(40u);
    int Wh = (int)PU(2), Hh = (int)PU(3);
    int cx = (int)floor(M(A((float)x, 0.5f), 1.5f)), cy = (int)floor(M(A((float)y, 0.5f), 1.5f));
    float we[5];
    [unroll] for (uint k = 0; k < 5u; ++k) we[k] = f16tof32(PU(17u + k));
    float3 hrow[5];
    [unroll] for (int j = 0; j < 5; ++j) {
        int r = clamp(cy + j - 2, 0, Hh - 1);
        float3 acc = 0.0f, mn = RR_HMAX, mx = -RR_HMAX;
        [unroll] for (int n = 0; n < 5; ++n) {
            float w = (x & 1) == 0 ? we[n] : we[4 - n];
            float3 tap = texel(q, clamp(cx + n - 2, 0, Wh - 1), r).xyz;
            [unroll] for (uint c = 0; c < 3u; ++c) { acc[c] = HF(w, tap[c], acc[c]); mn[c] = pmin(mn[c], tap[c]); mx[c] = pmax(mx[c], tap[c]); }
        }
        [unroll] for (uint c = 0; c < 3u; ++c) hrow[j][c] = pmin(pmax(acc[c], mn[c]), mx[c]);
    }
    float3 acc = 0.0f, mn = RR_HMAX, mx = -RR_HMAX, col;
    [unroll] for (int j2 = 0; j2 < 5; ++j2) {
        float w = (y & 1) == 0 ? we[j2] : we[4 - j2];
        [unroll] for (uint c = 0; c < 3u; ++c) { acc[c] = HF(w, hrow[j2][c], acc[c]); mn[c] = pmin(mn[c], hrow[j2][c]); mx[c] = pmax(mx[c], hrow[j2][c]); }
    }
    [unroll] for (uint c = 0; c < 3u; ++c) col[c] = DECODE(pmin(pmax(acc[c], mn[c]), mx[c]));
    return col;
}

// ---- 54b.2-54b.4 -----------------------------------------------------------------------------------------------------
// One output pixel (x, y) of the We x He grid, every read and store of it (also the fast kernel's per-pixel fallback,
// rr_resolve_fast.hlsl).
void rr_resolve_px(int x, int y)
{
    uint We = PU(0), He = PU(1);
    uint flags = PU(8);
    float E = (flags & 256u) != 0u ? asfloat(wg_ld32(PU(15), 0u)) : PF(9);
    float Eprev = (flags & 256u) != 0u ? asfloat(wg_ld32(PU(15), 4u)) : PF(10);
    float3 col = plain_col(x, y);
    float alphaC = 1.0f, oaSep = 0.0f;
    if ((flags & 1u) != 0u) {
        int Wr = (int)PU(4), Hr = (int)PU(5);
        float fWe = (float)We, fHe = (float)He, fWr = (float)Wr, fHr = (float)Hr, jx = PF(11), jy = PF(12);
        float sxr = DIV(fWr, fWe), syr = DIV(fHr, fHe), sxh = DIV(fWe, (float)PU(2)), syh = DIV(fHe, (float)PU(3));
        float jxp = M(DIV(fWe, fWr), jx), jyp = M(DIV(fHe, fHr), jy);
        float rWe = RCP(fWe), rHe = RCP(fHe), xf = (float)x, yf = (float)y;
        float xh = A(xf, 0.5f), yh = A(yf, 0.5f);
        int kx = (int)RND(A(F(xh, sxr, -0.5f), jx)), ky = (int)RND(A(F(yh, syr, -0.5f), jy));
        Res mvr = res(88u);
        float mxo = 0.0f, myo = 0.0f;
        if (mvr.off != 0u) {
            float4 mv = texel(mvr, clamp(kx, 0, Wr - 1) + mvr.ox, clamp(ky, 0, Hr - 1) + mvr.oy);
            mxo = M(mv.x, sxh); myo = M(mv.y, syh);
        }
        float hu = M(A(A(xf, mxo), 0.5f), rWe), hv = M(A(A(yf, myo), 0.5f), rHe);
        if ((flags & 2u) != 0u) {
            Res L = res(64u), O = res(72u);
            int lbx = (int)PU(13), lby = (int)PU(14);
            bool useO = (flags & 4u) != 0u;
            float a = A(xf, jxp), b = A(yf, jyp);
#if RR_MUT == 2
            bool sflag = abs(A(RND(a), -a)) <= 0.5f && abs(A(RND(b), -b)) <= 0.5f;
#else
            bool sflag = abs(A(RND(a), -a)) < 0.5f && abs(A(RND(b), -b)) < 0.5f;
#endif
            float TBW = PF(35), w = sflag ? TBW : M(TBW, TBW), w1 = A(1.0f, -w);
            float u = M(A(xh, jxp), rWe), v = M(A(yh, jyp), rHe);
            float4 cur = bilin(L, u, v);
            cur.x = M(E, cur.x); cur.y = M(E, cur.y); cur.z = M(E, cur.z);
            float3 curO = useO ? bilin(O, u, v).xyz : float3(0.0f, 0.0f, 0.0f);
            float4 hist = cur;
            float3 histO = curO;
            if ((flags & 32u) == 0u) {
                float4 mn = RR_FMAX, mx = RR_FMINN;
                float3 mnO = RR_FMAX, mxO = RR_FMINN;
                [unroll] for (int j = -1; j <= 1; ++j) [unroll] for (int i = -1; i <= 1; ++i) {
                    int tx = clamp(kx + i, 0, Wr - 1), ty = clamp(ky + j, 0, Hr - 1);
                    float4 t = texel(L, tx + lbx, ty + lby);
                    t.x = M(E, t.x); t.y = M(E, t.y); t.z = M(E, t.z);
                    [unroll] for (uint c = 0; c < 4u; ++c) { mn[c] = pmin(mn[c], t[c]); mx[c] = pmax(mx[c], t[c]); }
                    if (useO) {
                        float3 o = texel(O, tx + O.ox, ty + O.oy).xyz;
                        [unroll] for (uint c2 = 0; c2 < 3u; ++c2) { mnO[c2] = pmin(mnO[c2], o[c2]); mxO[c2] = pmax(mxO[c2], o[c2]); }
                    }
                }
                float4 Hp = bilin(res(RR_MUT == 1 ? 120u : 112u), hu, hv);
                float fac = M(E, RCP(pmax(Eprev, PF(34))));
                Hp.x = M(Hp.x, fac); Hp.y = M(Hp.y, fac); Hp.z = M(Hp.z, fac);
                [unroll] for (uint c = 0; c < 4u; ++c) hist[c] = clampHist(Hp[c], mn[c], mx[c]);
                if (useO) {
                    float3 HO = bilin(res(128u), hu, hv).xyz;
                    [unroll] for (uint c2 = 0; c2 < 3u; ++c2) histO[c2] = clampHist(HO[c2], mnO[c2], mxO[c2]);
                }
            }
            float4 T;
            [unroll] for (uint c = 0; c < 4u; ++c) T[c] = F(w, cur[c], M(w1, hist[c]));
            store_px(res(120u), x, y, T);
            float4 Ts = float4(H(T.x), H(T.y), H(T.z), H(T.w));
            if (useO) {
                float3 Tp;
                [unroll] for (uint c = 0; c < 3u; ++c) Tp[c] = F(w, curO[c], M(w1, histO[c]));
                store_px(res(136u), x, y, float4(Tp, 1.0f));
                [unroll] for (uint c2 = 0; c2 < 3u; ++c2) col[c2] = F(col[c2], A(1.0f, -H(Tp[c2])), Ts[c2]);
            } else {
                [unroll] for (uint c = 0; c < 3u; ++c) col[c] = F(col[c], A(1.0f, -Ts.w), Ts[c]);
            }
        }
        if ((flags & 8u) != 0u) {
            Res Ar = res(80u), Pr = res(96u), Gr = res(104u);
            uint ach = PU(36);
            float px = F(xh, sxr, jx), py = F(yh, syr, jy);
            float fxf = floor(A(px, -0.5f)), fyf = floor(A(py, -0.5f));
            float dx0 = A(A(fxf, 0.5f), -px), dy0 = A(A(fyf, 0.5f), -py);
            int fxi = (int)fxf, fyi = (int)fyf;
            float K2n = PF(23), K2x2n = PF(24);
            float W = 0.0f, Aw = 0.0f;
            float3 Gs = 0.0f;
            [unroll] for (int i = -1; i <= 2; ++i) {
                float dx = A(dx0, (float)i);
                int tx = clamp(fxi + i, 0, Wr - 1);
                [unroll] for (int j = -1; j <= 2; ++j) {
                    float dy = A(dy0, (float)j);
                    int ty = clamp(fyi + j, 0, Hr - 1);
                    float3 p = texel(Pr, tx, ty).xyz;
                    float qx = M(p.x, K2n), qy = M(p.y, K2n), qxy = M(p.z, K2x2n);
                    float e = F(qxy, M(dx, dy), F(qx, M(dx, dx), M(qy, M(dy, dy))));
                    float wt = X(e);
                    W = A(W, wt);
                    Aw = F(wt, texel(Ar, tx + Ar.ox, ty + Ar.oy)[ach], Aw);
                    [unroll] for (uint q = 0; q < 3u; ++q) Gs[q] = F(wt, g_half(Gr, tx, ty, q), Gs[q]);
                }
            }
            float rW = RCP(pmax(W, PF(32)));
            float3 nq = float3(M(Gs.x, rW), M(Gs.y, rW), M(Gs.z, rW));
            float g = S(M(nq.x, M(W, PF(22))));
            float beta = F(g, A(1.0f, -nq.y), nq.y), gamma = F(g, A(1.0f, -nq.z), nq.z);
            float a_up = M(Aw, rW), hc = 0.0f;
            if ((flags & 64u) == 0u) {
                float h = bilin(res(144u), hu, hv).x, mn = RR_FMAX, mx = -RR_FMAX;
                [unroll] for (int j = -1; j <= 1; ++j) [unroll] for (int i = -1; i <= 1; ++i) {
                    float t = texel(Ar, clamp(kx + i, 0, Wr - 1) + Ar.ox, clamp(ky + j, 0, Hr - 1) + Ar.oy)[ach];
                    mn = pmin(mn, t); mx = pmax(mx, t);
                }
                hc = clampHist(h, mn, mx);
            }
            float ah = F(a_up, beta, M(A(1.0f, -beta), hc));
            ah = nanf(ah) ? a_up : ah;
            float oa = F(a_up, gamma, M(A(1.0f, -gamma), hc));
            oa = nanf(oa) ? a_up : oa;
            store_px(res(152u), x, y, float4(ah, 0.0f, 0.0f, 0.0f));
            if ((flags & 16u) != 0u) oaSep = oa; else alphaC = oa;
        }
    }
    float rE = RCP(E);
    float4 lin = float4(M(rE, col.x), M(rE, col.y), M(rE, col.z), alphaC);
    bool inside = (flags & 128u) == 0u || (x < (int)PU(6) && y < (int)PU(7));
    if (inside) {
        Res o = res(48u);
        store_px(o, x + o.ox, y + o.oy, lin);
        if ((flags & 25u) == 25u) { Res ao = res(56u); store_px(ao, x + ao.ox, y + ao.oy, float4(oaSep, 0.0f, 0.0f, 0.0f)); }
    }
}

WG_KERNEL(rr_resolve_main)
{
    WG_SETUP();
    uint idx = wg_block_id() * WG_BLOCK + wg_lane_id();
    uint We = PU(0), He = PU(1);
    if (idx >= We * He) return;
    rr_resolve_px((int)(idx % We), (int)(idx / We));
}

// ---- 54b.5 -----------------------------------------------------------------------------------------------------------
// One PH2 texel (index idx of the ceil(Wh/4) x ceil(Hh/4) area; also rr_resolve_fast.hlsl's pool kernel).
void rr_pool_px(uint idx)
{
    Res ph = res(160u), pq = res(168u);
    uint w2 = (ph.w + 1u) / 2u, h2 = (ph.h + 1u) / 2u;
    if (idx >= w2 * h2) return;
    int x = (int)(idx % w2), y = (int)(idx / w2);
    float K5 = PF(26), K10 = PF(29);
    float3 acc = 0.0f;
    uint cnt = 0u;
    [unroll] for (uint t = 0; t < 4u; ++t) {
        int tx = 2 * x + (int)(t & 1u), ty = 2 * y + (int)(t >> 1);
        if (tx < (int)ph.w && ty < (int)ph.h) {
            float3 c = texel(ph, tx, ty).xyz;
            [unroll] for (uint k = 0; k < 3u; ++k) {
                float cc = M(c[k], PF(33));
                float tt = (cc > K5 || nanf(cc)) ? min_nan(cc, K10) : K5;
                acc[k] = F(A(X(M(M(tt, PF(28)), RR_K9)), -1.0f), PF(31), acc[k]);
            }
            cnt += 1u;
        }
    }
    float rn = RCP((float)cnt);
    float4 o = float4(0.0f, 0.0f, 0.0f, 1.0f);
    [unroll] for (uint k = 0; k < 3u; ++k) {
        float mean = M(rn, acc[k]);
        o[k] = M(M(M(L(pmax(F(mean, PF(25), 1.0f), K5)), RR_K6), PF(27)), PF(30));
    }
    store_px(pq, x, y, o);
}

WG_KERNEL(rr_resolve_pool)
{
    WG_SETUP();
    rr_pool_px(wg_block_id() * WG_BLOCK + wg_lane_id());
}

// ---- probes ---------------------------------------------------------------------------------------------------------
// aux = 0: exp2, 1: log2, 2: rcp of n f32 inputs at prm (bits), results after them.
WG_KERNEL(rr_resolve_elem)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    float x = asfloat(wg_ld32(prm, i * 4u));
    bool h;
    float r = aux == 0u ? EXP2_CR(x, 0u, h) : (aux == 1u ? LOG2_CR(x, 0u, h) : RCP(x));
    wg_st32(prm, (n + i) * 4u, asuint(r));
}

// Arithmetic probe: records of three f32 (a, b, c) at prm; eight result words per record after them:
// M(a,b) A(a,b) F(a,b,c) DIV(a,b) RND(a) pmin(a,b) pmax(a,b) HF(h(a),h(b),h(c)) (binary16 bits).
WG_KERNEL(rr_resolve_arith)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    float a = asfloat(wg_ld32(prm, i * 12u)), b = asfloat(wg_ld32(prm, i * 12u + 4u)), c = asfloat(wg_ld32(prm, i * 12u + 8u));
    uint o = n * 12u + i * 32u;
    wg_st128(prm, o, uint4(asuint(M(a, b)), asuint(A(a, b)), asuint(F(a, b, c)), asuint(DIV(a, b))));
    float za = ftz(a), zb = ftz(b);                  // in the kernels these take f32 operation results / reads
    wg_st128(prm, o + 16u, uint4(asuint(RND(za)), asuint(pmin(za, zb)), asuint(pmax(za, zb)), wg_fma_f16(H(a), H(b), H(c))));
}

// Bilinear probe (52.1.4 as used by 54b.3): the texture at parameter slot 64 sampled at n coordinate pairs (f32
// bits at aux); four f32 results per sample after the coordinates.
WG_KERNEL(rr_resolve_bilin)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    float u = asfloat(wg_ld32(aux, i * 8u)), v = asfloat(wg_ld32(aux, i * 8u + 4u));
    wg_st128(aux, n * 8u + i * 16u, asuint(bilin(res(64u), u, v)));
}

// Exhaustive agreement of the fast path (margin test) with the double-double decision over a range of f32 bit
// patterns: prm = counters [mismatch, window, hard, listed] + list of (input, fast result) pairs of the hard
// inputs (capacity in counter word 4); aux = first bit pattern, n = count. RR_EXH_LOG2 selects log2.
#ifndef RR_EXH_LOG2
#define RR_EXH_LOG2 0
#endif
WG_KERNEL(rr_resolve_exhaustive)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    uint bits = aux + i;
    float x = asfloat(bits);
    bool hard, h2;
    float fast, slow;
    bool window;
#if RR_EXH_LOG2
    if (!((bits & 0x80000000u) == 0u && bits >= 0x00800000u && bits < 0x7f800000u)) return;
    fast = LOG2_CR(x, 0u, hard);
    double y = log2_fast64(x);
    uint lo, hi; asuint(y, lo, hi);
    window = y != 0.0 && abs((int)(lo & 0x1fffffffu) - 0x10000000) <= (1 << 20);
    slow = window ? LOG2_CR(x, 1u, h2) : fast;
#else
    float xf = ftz(x);
    if (nanf(xf) || !(xf >= -126.0f && xf < 128.0f)) return;
    fast = EXP2_CR(x, 0u, hard);
    double sN;
    double y = exp2_fast64(xf, sN);
    uint lo, hi; asuint(y, lo, hi);
    window = abs((int)(lo & 0x1fffffffu) - 0x10000000) <= (1 << 20);
    slow = window ? EXP2_CR(x, 1u, h2) : fast;
#endif
    uint old;
    if (window) wg_mem.InterlockedAdd(prm + 4u, 1u, old);
    if (asuint(fast) != asuint(slow)) wg_mem.InterlockedAdd(prm, 1u, old);
    if (hard) {
        wg_mem.InterlockedAdd(prm + 8u, 1u, old);
        uint cap = wg_ld32(prm, 16u), slot;
        wg_mem.InterlockedAdd(prm + 12u, 1u, slot);
        if (slot < cap) { wg_st32(prm, 32u + slot * 8u, bits); wg_st32(prm, 36u + slot * 8u, asuint(fast)); }
    }
}
