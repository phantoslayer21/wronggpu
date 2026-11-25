// SPDX-License-Identifier: MIT
// Phase-split form of model K's wide stage m (deep-split lever, probe: stage m only). WrongGPU's own code. D3D12 only:
// the same arithmetic as k_stagew4_m (kernels/k_block_wide.hlsl, hlsl/k_wide.hlsli; HIP: k_block_wide.co), whose one
// workgroup per window runs every phase of the block (normalization, q / k / v, attention, Wo, MLP) in sequence on 16
// waves. At 6 .. 24 windows that leaves most CUs idle and the window's dependent chain of weight loads and WMMAs sets the
// time. Here the window's phases run as four launches, each spread over many more workgroups, joined by the dispatch
// boundaries (the device's UAV barrier between launches):
//   k_split_m_attn  one workgroup (4 waves) per (window, head): normalization, q / k / v of the head, scores, softmax,
//                   z of the head -> Z. Every head's K / V is computed once (no redundant K / V).
//   k_split_m_wo    one workgroup per (window, output tile): y = Wo z + bo over the z tiles in order, x1 = f16(x + y),
//                   a = f16(x1 g2) -> X1, A.
//   k_split_m_w1    one workgroup per (window, KS_HB hidden tiles): h = act(f16(W1 a + b1)) -> H.
//   k_split_m_w2    one workgroup per (window, output tile): x2 = f16(x1 + f16(W2 h + b2)) over the hidden tiles in
//                   order, stored for the real positions of the block output map.
// Wave wv of every workgroup holds the window's tokens 16 wv .. 16 wv + 15 (lane: token 16 wv + lane % 16, channel half
// lane / 16), as in k_stagew4_m. Exactness: every accumulator is the same chain of v_wmma_f32_16x16x16_f16 with the same
// A / B operands in the same order from the same initial value as in k_stagew4_m (bias first; q / k / v over kt; scores
// bias then ct; z over st; Wo over the z tiles 0 .. DT - 1; W1 over kt; W2 over the hidden tiles 0 .. HT - 1), and every
// FP16 value handed between the phases (z, x1, a, h) is the FP16 value k_stagew4_m keeps in LDS or registers at that
// point (the driver's RNE matrix conversion, then the same FP16 operations): the output bytes are k_stagew4_m's. The
// helpers (normalization operations, softmax polynomial, weights k_smx.hlsli, activation, mirror) are k_wide.hlsli's
// and k_smx.hlsli's, unchanged. The create-time self-check runs this form for the m case when the runtime takes it
// (KRuntime::launch redirects k_stagew4_m), against the same HIP reference hashes (k_smx_selfcheck.h).
// Scratch (runtime_k.cpp split_*_, FP16, per window w and token t): Z [w][64][DT 16], X1 / A [w][64][C], H [w][64][HT 16].
// Arguments: KBlkParams by value (s184:p19.b32, the stage's own), then the four scratch pointers.
// @wg-kernel k_split_m_attn co=k_block_wide.co block=128 wave=32 args=s184:p19.b32,p,p,p,p entry=ks_attn defines=KS_PHASE=1,KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4
// @wg-kernel k_split_m_attn__rodd co=k_block_wide.co block=128 wave=32 args=s184:p19.b32,p,p,p,p entry=ks_attn defines=K_SMX=1,KS_PHASE=1,KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4
// @wg-kernel k_split_m_wo co=k_block_wide.co block=128 wave=32 args=s184:p19.b32,p,p,p,p entry=ks_wo defines=KS_PHASE=2,KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4
// @wg-kernel k_split_m_w1 co=k_block_wide.co block=128 wave=32 args=s184:p19.b32,p,p,p,p entry=ks_w1 defines=KS_PHASE=3,KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4
// @wg-kernel k_split_m_w2 co=k_block_wide.co block=128 wave=32 args=s184:p19.b32,p,p,p,p entry=ks_w2 defines=KS_PHASE=4,KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4
#ifndef KW_LD_A16
#define KW_LD_A16 1   // 16-byte aligned loads: the whole address masked (hlsl/k_wide.hlsli; 2 = scalar base only: loses RADV's b128 loads)
#endif
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint P_x, P_skip, P_y, P_merged, P_head, P_g1, P_g2, P_wqkv, P_mix, P_wo, P_bo, P_w1, P_b1, P_w2, P_b2, P_wa, P_ba,
         P_wz, P_bz;                                   // pointers (arena offsets; 0 = null)
    int P_w, P_h, P_ox, P_oy, P_nwx, P_nwy, P_cw, P_ch;
    uint P_sz, P_sx1, P_sa, P_sh;                      // the split scratch: Z, X1, A, H
WG_ARGS_END

#include "k_wide.hlsli"
#include "k_smx.hlsli"

#if KW_IN != 0 || KW_CNEXT != 0
#error "k_block_split.hlsl: map input, no merge (stage m) only"
#endif

#ifndef KS_RD
#define KS_RD 20                                       // weight ring depth of the q / k / v chains (k_split_m_attn)
#endif
#ifndef KS_B2
#define KS_B2 20                                       // W2 products per fenced operand batch (k_split_m_w2)
#endif
#ifndef KS_HB
#define KS_HB 2                                        // hidden tiles per workgroup (k_split_m_w1)
#endif
// The hidden tiles HT = 4 KW_C / 16 are split into the W2 batches (HT / KS_B2) and the W1 groups (HT / KS_HB). The
// runtime's W1 launch (runtime_k.cpp, split_m) runs HT / 2 groups per window: that is this grid only for KS_HB = 2, so a
// change of KS_HB must change that launch too.
#if (4 * KW_C / 16) % KS_B2 != 0 || (4 * KW_C / 16) % KS_HB != 0
#error "k_block_split.hlsl: KS_B2 and KS_HB must divide HT = 4 KW_C / 16 (the W2 batches and the W1 groups)"
#endif

// Operand fence: the D3D12 driver's scheduler moves each global load of a short, low-pressure chain down to its
// product (one memory latency per product, measured in the RGA ISA of the first build). A phase therefore loads a
// product group's operands, XORs one word of each into f and XORs KS_FENCE(f) into its first operand (the bias of the
// accumulator, or the first fragment): kw_zero() is 0 at
// run time (grids are below 2^31) but unknown to the compiler, so every load is issued before the group's first product
// and waited once. No bit of any operand changes.
#define KS_FENCE(f) ((f) & kw_zero())

static const int SA = 2 * KT * 3;                      // q / k / v products of one head
static const int NHG = HT / KS_HB;                     // hidden-tile groups of a window

// The lane's window geometry (as kw_stage): token t, the mirrored member it reads, whether it is a real position.
struct KsLane {
    uint lane, lane8;
    int wv, n16, hh, t, gx, gy;
    bool real;
    uint mtok;
};
KsLane ks_lane(int win)
{
    KsLane L;
    const uint tid = wg_lane_id();
    L.lane = tid & 31u;
    L.lane8 = L.lane * 8u;
    L.wv = wg_uniform((int)(tid >> 5));
    L.n16 = (int)(L.lane & 15u);
    L.hh = (int)(L.lane >> 4);
    L.t = 16 * L.wv + L.n16;
    const int wa = win % P_nwx, wb = win / P_nwx;
    L.gx = 8 * wa - P_ox + (L.t & 7);
    L.gy = 8 * wb - P_oy + (L.t >> 3);
    L.real = L.gx >= 0 && L.gx < P_w && L.gy >= 0 && L.gy < P_h;
    L.mtok = (uint)(kw_mirror(L.gy, P_h) * P_w + kw_mirror(L.gx, P_w));   // mirrored member (spec 41)
    return L;
}

// scratch element offsets (halfs) of token t of window win, channel half hh
uint ks_zrow(int win, int t, int hh) { return (uint)((win * 64 + t) * DT * 16 + 8 * hh); }
uint ks_crow(int win, int t, int hh) { return (uint)((win * 64 + t) * KW_C + 8 * hh); }
uint ks_hrow(int win, int t, int hh) { return (uint)((win * 64 + t) * HT * 16 + 8 * hh); }

#if KS_PHASE == 1
// ---- k_split_m_attn: one (window, head) ------------------------------------------------------------------------------
// LDS: K [64 tokens][4 chunks] and VT [32 channels][8 chunks] of the head, 16-byte chunks xor-swizzled (k_wide.hlsli's
// NG = 4 layout of one group); 8 KB.
groupshared uint4 ks_lds[512];
uint ks_kidx(int row, int c) { return (uint)(row * 4 + (c ^ (row & 3))) & 511u; }
uint ks_vidx(int row, int c) { return (uint)(256 + row * 8 + (c ^ (row & 7))) & 511u; }

static uint ksA_qkv;
// product s = (ct KT + kt) 3 + {q, k, v} of the head's q / k / v chains (k_block_wide.hlsl's stream order of a round)
uint4 ks_wq(int s) { return kw_ld128(ksA_qkv, (uint)(((s % 3) * 2 * KT + s / 3) * 512)); }
#define KS_TAKE(s) rg[(s) % KS_RD]
#define KS_NEXT(s) { if ((s) + KS_RD < SA) rg[(s) % KS_RD] = ks_wq((s) + KS_RD); }

WG_KERNEL(ks_attn)
{
    WG_SETUP();
    const int bid = (int)wg_block_id();
    const int win = bid / KW_HEADS, hd = bid % KW_HEADS;
    const KsLane L = ks_lane(win);
    const int t = L.t, wv = L.wv, n16 = L.n16, hh = L.hh;
    ksA_qkv = P_wqkv + 2u * ((uint)(hd * 3 * 2 * KT * 256) + L.lane8);
    uint4 rg[KS_RD];

    // the block input (all channels of the token) and g1, then the weight ring
    uint4 n[KT];
    uint4 g1v[KT];
    {
        const uint xt = L.mtok * KW_C + 8u * hh;
        [unroll] for (int kt = 0; kt < KT; ++kt) n[kt] = kw_ldg(P_x, xt + 16u * kt);
        [unroll] for (int kt4 = 0; kt4 < KT; ++kt4) g1v[kt4] = kw_ldg(P_g1, (uint)(16 * kt4 + 8 * hh));
        [unroll] for (int s0 = 0; s0 < KS_RD; ++s0) rg[s0] = ks_wq(s0);
        uint f = 0u;                                                   // (the chain starts with the last loads)
        [unroll] for (int s1 = KS_RD - 1; s1 >= 0; --s1) f ^= rg[s1].x;
        [unroll] for (int kt5 = KT - 1; kt5 >= 0; --kt5) f ^= g1v[kt5].x ^ n[kt5].x;
        n[0].x ^= KS_FENCE(f);                                         // the input, g1 and the ring in one wait
    }

    // normalization (k_block_wide.hlsl step 1): s = f16(sum f16(x^2)), r = f16(rsq(s)); s = 0 gives r = 0
    {
        float ss = 0.0f;
        [unroll] for (int kt = 0; kt < KT; ++kt) {
            const uint4 sq = kw_opq4(kw_mul8(n[kt], n[kt]));
            [unroll] for (uint i = 0; i < 4; ++i) {
                ss += f16tof32(sq[i] & 0xffffu);
                ss += f16tof32(sq[i] >> 16);
            }
        }
        ss += wg_xor16(ss);
        const uint s16 = wg_f2h(ss);
        const uint r16 = (s16 & 0x7fffu) == 0u ? 0u : wg_f2h(rsqrt(f16tof32(s16)));
        const uint4 rv = kw_splat8(r16);
        [unroll] for (int kt = 0; kt < KT; ++kt)
            n[kt] = kw_mul8(n[kt], kw_mulo8(rv, g1v[kt]));
    }

    // q, k, v^T of head hd (k_block_wide.hlsl step 2, one round's group)
    uint4 q[2], kk[2], vv[2];
    [unroll] for (int ct = 0; ct < 2; ++ct) {
        v8f aq = wg_splat(0.0f), ak = wg_splat(0.0f), av = wg_splat(0.0f);
        [unroll] for (int kt = 0; kt < KT; ++kt) {
            const int s = (ct * KT + kt) * 3;
            aq = kw_mma(KS_TAKE(s), n[kt], aq);
            KS_NEXT(s);
            ak = kw_mma(KS_TAKE(s + 1), n[kt], ak);
            KS_NEXT(s + 1);
            av = kw_mma(n[kt], KS_TAKE(s + 2), av);
            KS_NEXT(s + 2);
        }
        q[ct] = kw_out(aq);
        kk[ct] = kw_out(ak);
        vv[ct] = kw_out(av);
    }
    const uint mrow = (uint)((hd * 64 + t) * 64 + 8 * hh);
    uint4 bm[4];
    [unroll] for (int st0 = 0; st0 < 4; ++st0) bm[st0] = kw_ldg(P_mix, mrow + 16u * st0);
    [unroll] for (int ct1 = 0; ct1 < 2; ++ct1) {
        ks_lds[ks_kidx(t, 2 * ct1 + hh)] = kk[ct1];                    // K[token t][channels]
        ks_lds[ks_vidx(16 * ct1 + n16, 2 * wv + hh)] = vv[ct1];        // VT[channel][tokens]
    }
    wg_barrier_lds();   // K / VT of the head complete

    // scores^T = K q + B, the clamped polynomial, softmax weights A (k_block_wide.hlsl, the same order)
    uint4 pa[4];
    {
        uint4 kf[4][2];
        [unroll] for (int st = 0; st < 4; ++st)
            [unroll] for (int ct = 0; ct < 2; ++ct) kf[st][ct] = ks_lds[ks_kidx(16 * st + n16, 2 * ct + hh)];
        v8f sc[4];
        [unroll] for (int st1 = 0; st1 < 4; ++st1) sc[st1] = kw_widen8(bm[st1]);
        [unroll] for (int ct = 0; ct < 2; ++ct)
            [unroll] for (int st2 = 0; st2 < 4; ++st2) sc[st2] = kw_mma(kf[st2][ct], q[ct], sc[st2]);
        uint4 gg[4];
        [unroll] for (int st5 = 0; st5 < 4; ++st5) gg[st5] = kw_softpoly8(kw_out(sc[st5]));
        float w[4][8];
        float sum = 0.0f;
        [unroll] for (int st3 = 0; st3 < 4; ++st3)
            [unroll] for (int j = 0; j < 8; ++j) {
                w[st3][j] = exp2(f16tof32((gg[st3][j >> 1] >> (16 * (j & 1))) & 0xffffu));   // v_exp_f32
                sum += w[st3][j];
            }
        sum += wg_xor16(sum);
        const float inv = rcp(sum);                                                           // v_rcp_f32
        [unroll] for (int st4 = 0; st4 < 4; ++st4) pa[st4] = k_smx_weights8(w[st4], inv);
    }
    // z_h^T = VT A^T (the two channel tiles' chains interleaved, each over st) -> Z tiles 2 hd, 2 hd + 1
    {
        uint4 vf[2][4];
        [unroll] for (int ct = 0; ct < 2; ++ct)
            [unroll] for (int st = 0; st < 4; ++st) vf[ct][st] = ks_lds[ks_vidx(16 * ct + n16, 2 * st + hh)];
        v8f az[2];
        az[0] = wg_splat(0.0f);
        az[1] = wg_splat(0.0f);
        [unroll] for (int st = 0; st < 4; ++st)
            [unroll] for (int ct2 = 0; ct2 < 2; ++ct2) az[ct2] = kw_mma(vf[ct2][st], pa[st], az[ct2]);
        const uint zr = ks_zrow(win, t, hh) + (uint)(32 * hd);
        kw_stg(P_sz, zr, kw_out(az[0]));
        kw_stg(P_sz, zr + 16u, kw_out(az[1]));
    }
}
#endif

#if KS_PHASE == 2
// ---- k_split_m_wo: one (window, output tile ot) ----------------------------------------------------------------------
WG_KERNEL(ks_wo)
{
    WG_SETUP();
    const int bid = (int)wg_block_id();
    const int win = bid / KT, ot = bid % KT;
    const KsLane L = ks_lane(win);
    const int hh = L.hh;
    const uint wo = P_wo + 2u * ((uint)(ot * DT * 256) + L.lane8);   // fragment dt at + 512 dt
    const uint zr = ks_zrow(win, L.t, hh);
    uint4 wf[DT], zf[DT];
    [unroll] for (int d0 = 0; d0 < DT; ++d0) {
        wf[d0] = kw_ld128(wo, (uint)(d0 * 512));
        zf[d0] = kw_ldg(P_sz, zr + 16u * d0);
    }
    const uint4 xr = kw_ldg(P_x, L.mtok * KW_C + 8u * hh + 16u * ot);
    const uint4 g2v = kw_ldg(P_g2, (uint)(16 * ot + 8 * hh));
    uint4 bo = kw_ldg(P_bo, (uint)(16 * ot + 8 * hh));
    uint f = bo.x ^ xr.x ^ g2v.x;                                    // (the chain starts with the last loads)
    [unroll] for (int d1 = 0; d1 < DT; ++d1) f ^= wf[d1].x ^ zf[d1].x;
    bo.x ^= KS_FENCE(f);
    v8f yacc = kw_widen8(bo);
    [unroll] for (int dt = 0; dt < DT; ++dt) yacc = kw_mma(wf[dt], zf[dt], yacc);
    const uint4 x1 = kw_add8(xr, kw_out(yacc));                     // x1 = f16(x + y)
    const uint cr = ks_crow(win, L.t, hh) + 16u * ot;
    kw_stg(P_sx1, cr, x1);
    kw_stg(P_sa, cr, kw_mul8(x1, g2v));                               // a = f16(x1 g2)
}
#endif

#if KS_PHASE == 3
// ---- k_split_m_w1: one (window, KS_HB hidden tiles) ------------------------------------------------------------------
WG_KERNEL(ks_w1)
{
    WG_SETUP();
    const int bid = (int)wg_block_id();
    const int win = bid / NHG, h0 = (bid % NHG) * KS_HB;
    const KsLane L = ks_lane(win);
    const int hh = L.hh;
    const uint w1 = P_w1 + 2u * ((uint)(h0 * KT * 256) + L.lane8);   // hidden tile h0 + m, kt: + 512 (m KT + kt)
    const uint cr = ks_crow(win, L.t, hh);
    uint4 a[KT];
    uint4 wf[KS_HB][KT];
    [unroll] for (int kt0 = 0; kt0 < KT; ++kt0) {
        [unroll] for (int m0 = 0; m0 < KS_HB; ++m0) wf[m0][kt0] = kw_ld128(w1, (uint)((m0 * KT + kt0) * 512));
        a[kt0] = kw_ldg(P_sa, cr + 16u * kt0);
    }
    uint4 b1v[KS_HB];
    [unroll] for (int m4 = 0; m4 < KS_HB; ++m4) b1v[m4] = kw_ldg(P_b1, (uint)(16 * (h0 + m4) + 8 * hh));
    uint f = 0u;                                                       // (the chain starts with the last loads)
    [unroll] for (int m5 = 0; m5 < KS_HB; ++m5) f ^= b1v[m5].x;
    [unroll] for (int kt1 = KT - 1; kt1 >= 0; --kt1) {
        f ^= a[kt1].x;
        [unroll] for (int m3 = KS_HB - 1; m3 >= 0; --m3) f ^= wf[m3][kt1].x;
    }
    v8f p[KS_HB];
    [unroll] for (int m1 = 0; m1 < KS_HB; ++m1) {
        b1v[m1].x ^= KS_FENCE(f);
        p[m1] = kw_widen8(b1v[m1]);
    }
    [unroll] for (int kt = 0; kt < KT; ++kt)                           // the hidden tiles' chains interleaved, each over kt
        [unroll] for (int m = 0; m < KS_HB; ++m) p[m] = kw_mma(wf[m][kt], a[kt], p[m]);
    const uint hr = ks_hrow(win, L.t, hh);
    [unroll] for (int m2 = 0; m2 < KS_HB; ++m2) kw_stg(P_sh, hr + (uint)(16 * (h0 + m2)), kw_act8(kw_out(p[m2])));
}
#endif

#if KS_PHASE == 4
// ---- k_split_m_w2: one (window, output tile ot) ----------------------------------------------------------------------
// The chain over the HT hidden tiles in batches of KS_B2 products, each batch's operands (W2 fragment (j, ot), h fragment
// of hidden tile j) loaded and fenced before its first product.
static const int NB2 = HT / KS_B2;
WG_KERNEL(ks_w2)
{
    WG_SETUP();
    const int bid = (int)wg_block_id();
    const int win = bid / KT, ot = bid % KT;
    const KsLane L = ks_lane(win);
    const int hh = L.hh;
    const uint w2 = P_w2 + 2u * ((uint)(ot * 256) + L.lane8);        // hidden tile j: + 512 j KT
    const uint hb = P_sh + 2u * ks_hrow(win, L.t, hh);                // hidden tile j: + 32 j
    const uint cr = ks_crow(win, L.t, hh) + 16u * ot;
    const uint4 x1 = kw_ldg(P_sx1, cr);
    uint4 b2v = kw_ldg(P_b2, (uint)(16 * ot + 8 * hh));
    v8f acc;
    [unroll] for (int b = 0; b < NB2; ++b) {                          // the hidden tiles 0 .. HT - 1, in order
        uint4 rw[KS_B2], rh[KS_B2];
        [unroll] for (int j0 = 0; j0 < KS_B2; ++j0) {
            rw[j0] = kw_ld128(w2, (uint)((b * KS_B2 + j0) * KT * 512));
            rh[j0] = kw_ld128(hb, (uint)((b * KS_B2 + j0) * 32));
        }
        uint f = b == 0 ? b2v.x ^ x1.x : 0u;
        [unroll] for (int j1 = 0; j1 < KS_B2; ++j1) f ^= rw[j1].x ^ rh[j1].x;
        if (b == 0) {                                                 // the bias after the fence: no early wait on it
            b2v.x ^= KS_FENCE(f);
            acc = kw_widen8(b2v);
        } else {
            rh[0].x ^= KS_FENCE(f);
        }
        [unroll] for (int j = 0; j < KS_B2; ++j) acc = kw_mma(rw[j], rh[j], acc);
    }
    const uint4 x2 = kw_pin(kw_add8(x1, kw_out(acc)));   // x2 = f16(x1 + f16(W2 h + b2)), before the divergent store
    if (L.real && P_y != 0u) kw_stg(P_y, (uint)((L.gy * P_w + L.gx) * KW_C + 8 * hh) + 16u * ot, x2);
}
#endif
