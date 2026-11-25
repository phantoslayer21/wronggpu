// SPDX-License-Identifier: MIT
// D3D12 ports of model K's wide stage kernels k_stagew<NG>_<name> (clean/k/kernels/k_block_wide.hip, FP32 accumulation;
// HIP build: k_block_wide.co). WrongGPU's own code. Basis, operation orders and every deviation: hlsl/k_wide.hlsli.
// The runtime (clean/k/runtime/runtime_k.cpp wide_ng) launches e3 / d3 with NG = 2, m with NG = 4, e4 / d4 with NG = 4
// up to 24 windows (1440p output) and NG = 2 above (4K output). One workgroup of 128 NG lanes per window, KBlkParams by
// value (s184:p19.b32).
// @wg-kernel k_stagew2_e3 co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=96,KW_HEADS=4,KW_IN=0,KW_CIN=96,KW_CNEXT=128,KW_NG=2,KW_RD=4
// @wg-kernel k_stagew4_e4 co=k_block_wide.co block=512 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=128,KW_HEADS=4,KW_IN=0,KW_CIN=128,KW_CNEXT=160,KW_NG=4,KW_RD=12
// @wg-kernel k_stagew2_e4 co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=128,KW_HEADS=4,KW_IN=0,KW_CIN=128,KW_CNEXT=160,KW_NG=2
// @wg-kernel k_stagew4_m co=k_block_wide.co block=512 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4,KW_RD=4
// @wg-kernel k_stagew4_d4 co=k_block_wide.co block=512 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=128,KW_HEADS=4,KW_IN=1,KW_CIN=160,KW_CNEXT=0,KW_NG=4,KW_RD=12
// @wg-kernel k_stagew2_d4 co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=128,KW_HEADS=4,KW_IN=1,KW_CIN=160,KW_CNEXT=0,KW_NG=2
// @wg-kernel k_stagew2_d3 co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=KW_C=96,KW_HEADS=4,KW_IN=1,KW_CIN=128,KW_CNEXT=0,KW_NG=2,KW_RD=4
// The __rodd forms: the same stages with the round-to-odd softmax weights (hlsl/k_smx.hlsli K_SMX 1), launched by the
// native D3D12 runtime when the mix-form probe (kernels/k_mix_probe.hlsl) fails on the adapter's driver.
// @wg-kernel k_stagew2_e3__rodd co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=96,KW_HEADS=4,KW_IN=0,KW_CIN=96,KW_CNEXT=128,KW_NG=2,KW_RD=4
// @wg-kernel k_stagew4_e4__rodd co=k_block_wide.co block=512 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=128,KW_HEADS=4,KW_IN=0,KW_CIN=128,KW_CNEXT=160,KW_NG=4,KW_RD=12
// @wg-kernel k_stagew2_e4__rodd co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=128,KW_HEADS=4,KW_IN=0,KW_CIN=128,KW_CNEXT=160,KW_NG=2
// @wg-kernel k_stagew4_m__rodd co=k_block_wide.co block=512 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=160,KW_HEADS=8,KW_IN=0,KW_CIN=160,KW_CNEXT=0,KW_NG=4,KW_RD=4
// @wg-kernel k_stagew4_d4__rodd co=k_block_wide.co block=512 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=128,KW_HEADS=4,KW_IN=1,KW_CIN=160,KW_CNEXT=0,KW_NG=4,KW_RD=12
// @wg-kernel k_stagew2_d4__rodd co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=128,KW_HEADS=4,KW_IN=1,KW_CIN=160,KW_CNEXT=0,KW_NG=2
// @wg-kernel k_stagew2_d3__rodd co=k_block_wide.co block=256 wave=32 args=s184:p19.b32 entry=kw_stage defines=K_SMX=1,KW_C=96,KW_HEADS=4,KW_IN=1,KW_CIN=128,KW_CNEXT=0,KW_NG=2,KW_RD=4
#ifndef KW_LD_A16
#define KW_LD_A16 2   // 16-byte loads, alignment proven on the scalar base only (no per-lane v_and -16; 1 = whole address, 0 = none) (hlsl/k_wide.hlsli)
#endif
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint P_x, P_skip, P_y, P_merged, P_head, P_g1, P_g2, P_wqkv, P_mix, P_wo, P_bo, P_w1, P_b1, P_w2, P_b2, P_wa, P_ba,
         P_wz, P_bz;                                   // pointers (arena offsets; 0 = null)
    int P_w, P_h, P_ox, P_oy, P_nwx, P_nwy, P_cw, P_ch;
WG_ARGS_END

#include "k_wide.hlsli"
#include "k_smx.hlsli"

// ---- the weight stream ----------------------------------------------------------------------------------------------
// Every global weight fragment of the kernel's matrix products (A operands; v^T: the B operand), in product order:
//   [expand: NK tiles x KTI, kt outer, tile inner]  (KW_IN = 1)
//   [round r = 0 .. NR - 1: q / k / v (SQ, per ct: kt outer, {q, k, v} inner), then Wo (SO, z tile outer, owned inner)]
//   [chunk c = 0 .. NC - 1: W1 (S1, kt outer, the group's MM hidden tiles inner), then W2 (S2, hidden outer, owned inner)]
//   [merge (MS): KTM tiles of the wave's output tile]  (KW_CNEXT > 0 with one tile per wave at most)
// One ring of KW_RD registers runs through the whole stream: the fragment of product s + KW_RD is loaded right after
// product s, also across phase borders (the next phase's first fragments are in flight while the current one ends).
// Every loop around it is unrolled, so each ring slot is a fixed register and every address folds to a constant offset.
// Each accumulator still sums in exactly the order of k_wide.hlsli; only independent accumulators are interleaved.
#ifndef KW_RD
#define KW_RD 8
#endif
#ifndef KW_ILV
#define KW_ILV 1                                             // interleave independent chains (scores, z, W1, expand)
#endif

static const int NK = KW_IN == 1 ? TOT / NW : 0;           // expand tiles per wave
static const int SE = NK * KTI;
static const int SQ = 2 * KT * 3;
static const int SO = 2 * NG * NOWN;
static const int RLEN = SQ + SO;
static const int NR = KW_HEADS / NG;
static const int S1 = MM * KT;
static const int S2 = CH * NOWN;
static const int CLEN = S1 + S2;
static const int NC = HT / CH;
static const bool MSTREAM = KW_CNEXT > 0 && NTM <= NW;       // the merge in the stream (one tile per wave at most)
static const int MS = MSTREAM ? KTM : 0;
static const int B_R = SE;
static const int B_C = B_R + NR * RLEN;
static const int B_M = B_C + NC * CLEN;
static const int SLEN = B_M + MS;

int kw_ot(int grp, int i) { return ((KT % NG) == 0 || grp + NG * i < KT) ? grp + NG * i : KT - 1; }

// Byte addresses of each weight tensor's first fragment for this lane (pointer + 2 (wave-uniform tile part + lane8)),
// set once by kw_stage; every fragment of the stream is then one of them plus a constant (the load's immediate offset).
static uint kwA_wa, kwA_qkv, kwA_w1, kwA_wz;
static uint kwA_wo[NOWN], kwA_w2[NOWN];

uint4 kw_wld(int s)
{
    if (s < B_R) {                                           // expand: product s = kt NK + k (tile wave + k NW)
        const int kt = s / max(NK, 1), k = s % max(NK, 1);
        return kw_ld128(kwA_wa, (uint)((k * NW * KTI + kt) * 512));
    }
    if (s < B_C) {
        const int r = (s - B_R) / RLEN, f = (s - B_R) % RLEN;
        if (f < SQ)                                          // f = (ct KT + kt) 3 + {q, k, v}; head r NG + grp
            return kw_ld128(kwA_qkv, (uint)((r * NG * 3 * 2 * KT + (f % 3) * 2 * KT + f / 3) * 512));
        const int f2 = f - SQ;                               // f2 = j NOWN + i: z tile 2 r NG + j, owned tile i
        if ((KT % NG) == 0)                                  // every owned tile present: grp + NG i, one address
            return kw_ld128(kwA_wo[0], (uint)(((f2 % NOWN) * NG * DT + 2 * r * NG + f2 / NOWN) * 512));
        return kw_ld128(kwA_wo[f2 % NOWN], (uint)((2 * r * NG + f2 / NOWN) * 512));
    }
    if (s < B_M) {
        const int c = (s - B_C) / CLEN, f = (s - B_C) % CLEN;
        if (f < S1) {                                        // hidden tile c CH + grp MM + m
            const int m = KW_ILV ? f % MM : f / KT, kt = KW_ILV ? f / MM : f % KT;
            return kw_ld128(kwA_w1, (uint)(((c * CH + m) * KT + kt) * 512));
        }
        const int f2 = f - S1;                               // f2 = j NOWN + i: hidden tile c CH + j, owned tile i
        if ((KT % NG) == 0)
            return kw_ld128(kwA_w2[0], (uint)(((c * CH + f2 / NOWN) * KT + (f2 % NOWN) * NG) * 512));
        return kw_ld128(kwA_w2[f2 % NOWN], (uint)((c * CH + f2 / NOWN) * KT * 512));
    }
    return kw_ld128(kwA_wz, (uint)((s - B_M) * 512));       // merge: tile min(wave, NTM - 1), kt = s - B_M
}

// product s of the stream: its fragment, and the refill of its ring slot with product s + KW_RD
#define KW_TAKE(s) rg[(s) % KW_RD]
#define KW_NEXT(s) { if ((s) + KW_RD < SLEN) rg[(s) % KW_RD] = kw_wld((s) + KW_RD); }

WG_KERNEL(kw_stage)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint lane = tid & 31u;
    const int wave = wg_uniform((int)(tid >> 5));
    const int grp = wave >> 2, wv = wave & 3;
    const int n16 = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int t = 16 * wv + n16;                       // window-local token of this lane
    const int win = (int)wg_block_id();
    const int wa = win % P_nwx, wb = win / P_nwx;
    const int gx = 8 * wa - P_ox + (t & 7), gy = 8 * wb - P_oy + (t >> 3);
    const bool real = gx >= 0 && gx < P_w && gy >= 0 && gy < P_h;
    const uint mtok = (uint)(kw_mirror(gy, P_h) * P_w + kw_mirror(gx, P_w));   // mirrored member (spec 41)
    const uint lane8 = lane * 8u;

    // owned output tiles ot = grp + NG i (present when < KT; else tile KT - 1 again, not stored)
    bool own[NOWN];
    int otv[NOWN];
    [unroll] for (int i0 = 0; i0 < NOWN; ++i0) {
        own[i0] = (KT % NG) == 0 || grp + NG * i0 < KT;
        otv[i0] = kw_ot(grp, i0);
    }

    kwA_wa = P_wa + 2u * ((uint)(wave * KTI * 256) + lane8);
    kwA_qkv = P_wqkv + 2u * ((uint)(grp * 3 * 2 * KT * 256) + lane8);
    kwA_w1 = P_w1 + 2u * ((uint)(grp * MM * KT * 256) + lane8);
    kwA_wz = P_wz + 2u * ((uint)(min(wave, max(NTM, 1) - 1) * KTM * 256) + lane8);
    [unroll] for (int i8 = 0; i8 < NOWN; ++i8) {
        kwA_wo[i8] = P_wo + 2u * ((uint)(otv[i8] * DT * 256) + lane8);
        kwA_w2[i8] = P_w2 + 2u * ((uint)(otv[i8] * 256) + lane8);
    }
    uint4 rg[KW_RD];                                   // the weight stream's ring

    // 1. the block input (n: all channels; xr: the owned tiles, for the residual; for KW_IN = 0 reloaded at step 3)
    uint4 n[KT];
    uint4 g1v[KT];
#if KW_IN == 1
    uint4 xr[NOWN];
    {
        const int jr = n16 >> 2, jc = n16 & 3;
        const int cr = (8 * wb - P_oy) / 2 + jr, cc = (8 * wa - P_ox) / 2 + jc;   // even offsets: exact halves
        const uint yc = (uint)((kw_mirror(cr, P_ch) * P_cw + kw_mirror(cc, P_cw)) * KW_CIN + 8 * hh);
        uint4 yb[KTI];
        [unroll] for (int kt = 0; kt < KTI; ++kt) yb[kt] = kw_ldg(P_x, yc + 16u * kt);
        [unroll] for (int s0 = 0; s0 < KW_RD; ++s0) rg[s0] = kw_wld(s0);
        v8f e[NK];
        [unroll] for (int k0 = 0; k0 < NK; ++k0) e[k0] = kw_widen8(kw_ldg(P_ba, (uint)(16 * (wave + k0 * NW) + 8 * hh)));
        [unroll] for (int kt2 = 0; kt2 < KTI; ++kt2)
            [unroll] for (int k = 0; k < NK; ++k) {           // the NK output tiles' chains interleaved, each over kt
                const int s = kt2 * NK + k;
                e[k] = kw_mma(KW_TAKE(s), yb[kt2], e[k]);
                KW_NEXT(s);
            }
        [unroll] for (int k1 = 0; k1 < NK; ++k1) {
            const int tt = wave + k1 * NW;              // output tile tt of 4C: sub-position s, channels c0 ..
            const int sp = (16 * tt) / KW_C, c0 = 16 * tt - sp * KW_C;
            const int ft = (2 * jr + (sp >> 1)) * 8 + 2 * jc + (sp & 1);   // the window member this sub-position feeds
            kw_lw((uint)(ft * (KW_C / 8) + c0 / 8 + hh), kw_out(e[k1]));
        }
        const uint sk = mtok * KW_C + 8u * hh;
        uint4 sv[KT];
        [unroll] for (int kt3 = 0; kt3 < KT; ++kt3) sv[kt3] = kw_ldg(P_skip, sk + 16u * kt3);
        [unroll] for (int kt4 = 0; kt4 < KT; ++kt4) g1v[kt4] = kw_ldg(P_g1, (uint)(16 * kt4 + 8 * hh));
        wg_barrier_lds();
        [unroll] for (int kt = 0; kt < KT; ++kt)
            n[kt] = kw_add8(kw_lr((uint)(t * (KW_C / 8) + 2 * kt + hh)), sv[kt]);   // f16(e + S)
        [unroll] for (int i = 0; i < NOWN; ++i)
            xr[i] = kw_add8(kw_lr((uint)(t * (KW_C / 8) + 2 * otv[i] + hh)), kw_ldg(P_skip, sk + 16u * otv[i]));
        wg_barrier_lds();   // the block input is read; LDS goes to K / VT
    }
#else
    const uint xt = mtok * KW_C + 8u * hh;
    {
        [unroll] for (int kt = 0; kt < KT; ++kt) n[kt] = kw_ldg(P_x, xt + 16u * kt);
        [unroll] for (int kt4 = 0; kt4 < KT; ++kt4) g1v[kt4] = kw_ldg(P_g1, (uint)(16 * kt4 + 8 * hh));
        [unroll] for (int s0 = 0; s0 < KW_RD; ++s0) rg[s0] = kw_wld(s0);
    }
#endif

    // normalization: s = f16(sum f16(x^2)) (FP32 sum in channel order), r = f16(rsq(s)); s = 0 gives r = 0
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

    // 2. token mixing in rounds of NG heads; yacc (owned tiles) accumulates the output projection
    v8f yacc[NOWN];
    [unroll] for (int i1 = 0; i1 < NOWN; ++i1) yacc[i1] = kw_widen8(kw_ldg(P_bo, (uint)(16 * otv[i1] + 8 * hh)));

    [unroll] for (int r = 0; r < NR; ++r) {
        const int hd = r * NG + grp;
        const int sb = B_R + r * RLEN;
        // q, k, v^T: product order per accumulator over kt; stream product (ct KT + kt) 3 + {q, k, v}
        uint4 q[2], kk[2], vv[2];
        [unroll] for (int ct = 0; ct < 2; ++ct) {
            v8f aq = wg_splat(0.0f), ak = wg_splat(0.0f), av = wg_splat(0.0f);
            [unroll] for (int kt = 0; kt < KT; ++kt) {
                const int s = sb + (ct * KT + kt) * 3;
                aq = kw_mma(KW_TAKE(s), n[kt], aq);
                KW_NEXT(s);
                ak = kw_mma(KW_TAKE(s + 1), n[kt], ak);
                KW_NEXT(s + 1);
                // v^T: rows = channels 16 ct + lane % 16 (B columns), elements = tokens 16 wv + 8 hh + 0..7
                av = kw_mma(n[kt], KW_TAKE(s + 2), av);
                KW_NEXT(s + 2);
            }
            q[ct] = kw_out(aq);
            kk[ct] = kw_out(ak);
            vv[ct] = kw_out(av);
        }
        // scores bias B[h][t][s] (lane = query t, elements = keys 16 st + 8 hh + 0..7), loaded before the barrier
        const uint mrow = (uint)((hd * 64 + t) * 64 + 8 * hh);
        uint4 bm[4];
        [unroll] for (int st0 = 0; st0 < 4; ++st0) bm[st0] = kw_ldg(P_mix, mrow + 16u * st0);
        if (TIGHT && r > 0) wg_barrier_lds();   // the previous round's Z reads are done (Z overlaps K / VT)
        [unroll] for (int ct1 = 0; ct1 < 2; ++ct1) {
            kw_lw(kw_kidx(grp, t, 2 * ct1 + hh), kk[ct1]);                 // K[token t][channels]
            kw_lw(kw_vidx(grp, 16 * ct1 + n16, 2 * wv + hh), vv[ct1]);     // VT[channel][tokens]
        }
        wg_barrier_lds();   // K / VT of the group's head complete

        // scores^T = K q + B (the four key tiles' chains interleaved, each bias then ct 0, 1), the clamped polynomial,
        // softmax weights A
        uint4 pa[4];
        {
            uint4 kf[4][2];
            [unroll] for (int st = 0; st < 4; ++st)
                [unroll] for (int ct = 0; ct < 2; ++ct) kf[st][ct] = kw_lr(kw_kidx(grp, 16 * st + n16, 2 * ct + hh));
            v8f sc[4];
            [unroll] for (int st1 = 0; st1 < 4; ++st1) sc[st1] = kw_widen8(bm[st1]);
#if KW_ILV
            [unroll] for (int ct = 0; ct < 2; ++ct)
                [unroll] for (int st2 = 0; st2 < 4; ++st2) sc[st2] = kw_mma(kf[st2][ct], q[ct], sc[st2]);
#else
            [unroll] for (int st2 = 0; st2 < 4; ++st2)
                [unroll] for (int ct = 0; ct < 2; ++ct) sc[st2] = kw_mma(kf[st2][ct], q[ct], sc[st2]);
#endif
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
            // A = f16(w inv), one rounding of the exact product (k_smx.hlsli: the mix form, or round to odd in the
            // __rodd ports)
            [unroll] for (int st4 = 0; st4 < 4; ++st4) pa[st4] = k_smx_weights8(w[st4], inv);
        }
        // z_h^T = VT A^T (the two channel tiles' chains interleaved, each over st)
        uint4 zz[2];
        {
            uint4 vf[2][4];
            [unroll] for (int ct = 0; ct < 2; ++ct)
                [unroll] for (int st = 0; st < 4; ++st) vf[ct][st] = kw_lr(kw_vidx(grp, 16 * ct + n16, 2 * st + hh));
            v8f az[2];
            az[0] = wg_splat(0.0f);
            az[1] = wg_splat(0.0f);
#if KW_ILV
            [unroll] for (int st = 0; st < 4; ++st)
                [unroll] for (int ct2 = 0; ct2 < 2; ++ct2) az[ct2] = kw_mma(vf[ct2][st], pa[st], az[ct2]);
#else
            [unroll] for (int ct2 = 0; ct2 < 2; ++ct2)
                [unroll] for (int st = 0; st < 4; ++st) az[ct2] = kw_mma(vf[ct2][st], pa[st], az[ct2]);
#endif
            zz[0] = kw_out(az[0]);
            zz[1] = kw_out(az[1]);
        }
        if (TIGHT) wg_barrier_lds();   // every group's K / VT reads are done (Z overlaps them)
        [unroll] for (int ct3 = 0; ct3 < 2; ++ct3) kw_lw(kw_zidx(t, 4 * grp + 2 * ct3 + hh), zz[ct3]);
        wg_barrier_lds();   // the round's z complete
        // the output projection for the owned tiles over the round's z tiles dt = 2 r NG + j, in order; stream product
        // j NOWN + i
        uint4 zf[2 * NG];
        [unroll] for (int j0 = 0; j0 < 2 * NG; ++j0) zf[j0] = kw_lr(kw_zidx(t, 2 * j0 + hh));
        [unroll] for (int j = 0; j < 2 * NG; ++j)
            [unroll] for (int i = 0; i < NOWN; ++i) {
                const int s = sb + SQ + j * NOWN + i;
                yacc[i] = kw_mma(KW_TAKE(s), zf[j], yacc[i]);
                KW_NEXT(s);
            }
    }

    // 3. x1 = f16(x + y) and a = f16(x1 g2) for the owned tiles; a goes to A
    uint4 x1[NOWN];
    {
        uint4 g2v[NOWN];
#if KW_IN == 0
        uint4 xr[NOWN];
        [unroll] for (int i6 = 0; i6 < NOWN; ++i6) xr[i6] = kw_ldg(P_x, xt + 16u * otv[i6]);
#endif
        [unroll] for (int i7 = 0; i7 < NOWN; ++i7) g2v[i7] = kw_ldg(P_g2, (uint)(16 * otv[i7] + 8 * hh));
        if (TIGHT) wg_barrier_lds();   // Z reads are done (A overlaps Z)
        [unroll] for (int i2 = 0; i2 < NOWN; ++i2) {
            x1[i2] = kw_add8(xr[i2], kw_out(yacc[i2]));
            if (own[i2]) kw_lw(kw_aidx(t, 2 * otv[i2] + hh), kw_mul8(x1[i2], g2v[i2]));
        }
    }
    wg_barrier_lds();
    uint4 a[KT];
    [unroll] for (int kt0 = 0; kt0 < KT; ++kt0) a[kt0] = kw_lr(kw_aidx(t, 2 * kt0 + hh));
    if (TIGHT) wg_barrier_lds();   // A reads are done (H overlaps A)

    // 4. MLP in chunks of CH hidden tiles: W1 + act by the groups (MM tiles each, their chains interleaved), W2 for the
    // owned output tiles
    v8f acc[NOWN];
    [unroll] for (int i3 = 0; i3 < NOWN; ++i3) acc[i3] = kw_widen8(kw_ldg(P_b2, (uint)(16 * otv[i3] + 8 * hh)));
    [unroll] for (int c = 0; c < NC; ++c) {
        const uint hb = (uint)(OFF_H / 8 + (c & 1) * CH * 128 + t * 2 + hh);   // + 128 j: tile j of the chunk
        const int sc1 = B_C + c * CLEN;
        v8f p[MM];
        [unroll] for (int m0 = 0; m0 < MM; ++m0) p[m0] = kw_widen8(kw_ldg(P_b1, (uint)(16 * (c * CH + grp * MM + m0) + 8 * hh)));
        [unroll] for (int f6 = 0; f6 < S1; ++f6) {
            const int m = KW_ILV ? f6 % MM : f6 / KT, kt = KW_ILV ? f6 / MM : f6 % KT;
            const int s = sc1 + f6;
            p[m] = kw_mma(KW_TAKE(s), a[kt], p[m]);
            KW_NEXT(s);
        }
        [unroll] for (int m1 = 0; m1 < MM; ++m1) kw_lw(hb + (uint)((grp * MM + m1) * 128), kw_act8(kw_out(p[m1])));
        wg_barrier_lds();   // h of the chunk complete (the other buffer's reads finished before the previous barrier)
        uint4 hv[CH];
        [unroll] for (int j1 = 0; j1 < CH; ++j1) hv[j1] = kw_lr(hb + (uint)(j1 * 128));
        [unroll] for (int j = 0; j < CH; ++j)
            [unroll] for (int i = 0; i < NOWN; ++i) {
                const int s = sc1 + S1 + j * NOWN + i;
                acc[i] = kw_mma(KW_TAKE(s), hv[j], acc[i]);
                KW_NEXT(s);
            }
    }

    // 5. x2 = f16(x1 + f16(W2 h + b2)), stored for the real positions
    [unroll] for (int i4 = 0; i4 < NOWN; ++i4) x1[i4] = kw_pin(kw_add8(x1[i4], kw_out(acc[i4])));   // x1 now holds x2
    if (real && P_y != 0u) {
        const uint yt = (uint)((gy * P_w + gx) * KW_C + 8 * hh);
        [unroll] for (int i = 0; i < NOWN; ++i)
            if (own[i]) kw_stg(P_y, yt + 16u * otv[i], x1[i]);
    }

#if KW_CNEXT > 0
    // 2x2 merge of the window's 16 groups: f16(Wm concat(TL, TR, BL, BR) + bm); x2 goes to A (members outside the map
    // are zero, as in k_block.hip)
    if (TIGHT) wg_barrier_lds();   // H reads are done (A overlaps H)
    [unroll] for (int i5 = 0; i5 < NOWN; ++i5)
        if (own[i5]) kw_lw(kw_aidx(t, 2 * otv[i5] + hh), real ? x1[i5] : uint4(0u, 0u, 0u, 0u));
    wg_barrier_lds();
    {
        const int gr = n16 >> 2, gc = n16 & 3;   // group (column n16): window rows 2 gr, 2 gr + 1, columns 2 gc, 2 gc + 1
        const int mw = (P_w + 1) / 2, mh = (P_h + 1) / 2;
        const int my = (8 * wb - P_oy) / 2 + gr, mx = (8 * wa - P_ox) / 2 + gc;
        const bool inside = my >= 0 && my < mh && mx >= 0 && mx < mw;
        const uint mo = inside ? (uint)((my * mw + mx) * KW_CNEXT + 8 * hh) : 0u;
        if (MSTREAM) {
            // one output tile per wave (tt = wave; waves >= NTM have none): weights from the stream, the LDS operands
            // in a ring of KW_RD read KW_RD products ahead
            if (wave < NTM) {
                const int tt = wave;
                uint4 rb[KW_RD];
                [unroll] for (int f5 = 0; f5 < KW_RD; ++f5) {
                    const int sp = (16 * f5) / KW_C, c0 = 16 * f5 - sp * KW_C;
                    const int ft = (2 * gr + (sp >> 1)) * 8 + 2 * gc + (sp & 1);
                    rb[f5] = kw_lr(kw_aidx(ft, c0 / 8 + hh));
                }
                v8f mm = kw_widen8(kw_ldg(P_bz, (uint)(16 * tt + 8 * hh)));
                [unroll] for (int kt = 0; kt < KTM; ++kt) {
                    const int s = B_M + kt;
                    mm = kw_mma(KW_TAKE(s), rb[kt % KW_RD], mm);
                    KW_NEXT(s);
                    if (kt + KW_RD < KTM) {
                        const int k2 = kt + KW_RD;
                        const int sp = (16 * k2) / KW_C, c0 = 16 * k2 - sp * KW_C;
                        const int ft = (2 * gr + (sp >> 1)) * 8 + 2 * gc + (sp & 1);
                        rb[kt % KW_RD] = kw_lr(kw_aidx(ft, c0 / 8 + hh));
                    }
                }
                const uint4 mo16 = kw_pin(kw_out(mm));   // computed before the divergent store (kw_pin)
                if (inside) kw_stg(P_merged, mo + 16u * tt, mo16);
            }
        } else {
            [loop] for (int tt = wave; tt < NTM; tt += NW) {
                const uint wzb = (uint)(tt * KTM * 256) + lane8;
                uint4 rw[KW_PD], rb[KW_PD];
                [unroll] for (int f5 = 0; f5 < KW_PD; ++f5) {
                    const int sp = (16 * f5) / KW_C, c0 = 16 * f5 - sp * KW_C;
                    const int ft = (2 * gr + (sp >> 1)) * 8 + 2 * gc + (sp & 1);
                    rw[f5] = kw_ldg(P_wz, wzb + (uint)(f5 * 256));
                    rb[f5] = kw_lr(kw_aidx(ft, c0 / 8 + hh));
                }
                v8f mm = kw_widen8(kw_ldg(P_bz, (uint)(16 * tt + 8 * hh)));
                [unroll] for (int kt = 0; kt < KTM; ++kt) {
                    mm = kw_mma(rw[kt % KW_PD], rb[kt % KW_PD], mm);
                    if (kt + KW_PD < KTM) {
                        const int k2 = kt + KW_PD;
                        const int sp = (16 * k2) / KW_C, c0 = 16 * k2 - sp * KW_C;
                        const int ft = (2 * gr + (sp >> 1)) * 8 + 2 * gc + (sp & 1);
                        rw[kt % KW_PD] = kw_ldg(P_wz, wzb + (uint)(k2 * 256));
                        rb[kt % KW_PD] = kw_lr(kw_aidx(ft, c0 / 8 + hh));
                    }
                }
                const uint4 mo16 = kw_pin(kw_out(mm));   // computed before the divergent store (kw_pin)
                if (inside) kw_stg(P_merged, mo + 16u * tt, mo16);
            }
        }
    }
#endif
}
