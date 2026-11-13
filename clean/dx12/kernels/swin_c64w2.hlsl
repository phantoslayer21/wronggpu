// SPDX-License-Identifier: MIT
// swin_block_c64_nc (enc1's Swin block) with TWO windows per wave in 128-lane workgroups (tracks lev3-swin, r3-c64w2).
//
// @wg-kernel swin_block_c64_nc co=swin.co block=256 group=128 gscale=1 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=64,SW_HEADS=2,SW_MLP=256,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_W2=1
//
// Launch-shape switch: the line above runs this file's kernel (SW_W2 1: 128-lane workgroups, gscale 1). The one-window-
// per-wave kernel of swin_block.hlsl (256-lane workgroups, HIP's shape) is the same line without "group=128 gscale=1"
// and with SW_W2=0 (its DXIL is then byte-identical to swin_block.hlsl's own build of the kernel).
//
// The launch keeps HIP's geometry (8 windows per 256-lane workgroup, grid = ceil(windows / 8)); the D3D12 device runs
// each launched workgroup as one workgroup of 128 lanes (manifest "group" 128, "gscale" 1), and here each D3D12 workgroup
// does the same 8 windows with 4 waves: wave v does windows 8 g + 2 v and 8 g + 2 v + 1 of its launched workgroup g, the
// two windows' instruction streams interleaved, every weight fragment read from LDS once for both.
// Why: the D3D12 driver runs these workgroups in CU mode (all waves of a workgroup on the 2 SIMDs of one CU). With 8 waves
// per workgroup every unit barrier waits for 8 waves (removing the barriers was worth -15 us in an ablation); here a
// barrier waits for 4 waves, and the two windows of a wave are independent chains for the matrix unit. Occupancy: the
// workgroup keeps the 16 KB of staging (64 KB of LDS per CU: up to 4 workgroups); the one-window kernel at <= 120 VGPRs
// runs 3 workgroups of 8 waves per CU (24 windows), this one 3 workgroups of 4 waves at <= 256 VGPRs (24 windows) or
// 4 workgroups at <= 192 VGPRs (32 windows; the defaults: 192). VGPR granule 24 (wave32, 1536 per SIMD lane).
// Every window's arithmetic is that of swin_block.hlsl's C = 64 path (= HIP's swin_block<64, 2, 256>): the same
// operations on the same operands in the same order, only interleaved with the other window's (and the per-workgroup
// weight / bias loads shared by both).
// lev3-swin (on diet-c64 round 1): -2.7 us enc1 stage, -4.0 us in game. Rolled head loop: +4..+5 us; MLP in 2-unit
// passes: +13; groupshared-table GELU patterns: see there. Tried and slower: 128-lane workgroups of one window per wave
// (+21..+36 us), the unit barriers replaced by LDS counters (+15..+31 us).
// r3-c64w2 (on diet-c64 round 3: no LDS GELU table, 16 KB per workgroup), enc1 stage A/B (cap-a2 capture, paired, against
// the one-window kernel, 0.278 ms): defaults (no group pipelining, GELU 10333 late, 192 VGPRs, 4 workgroups per CU)
// -12.4..-12.8 us. Measured on the way: lev3-swin's form (MLP unrolled, pipelined, 13033, 225 VGPRs) -4.2; without the
// group pipelining (216 VGPRs) -8.1..-8.8; + SW_GELU_LATE0 3 (192) -10.7..-11.4; GELU 10303 -9.3, 10330 -7.6, 10313 -8.0,
// 13030 -5.0 / -5.7, 13333 -5.9, 10003 +3.7, 11333 / 10033 215-216 VGPRs; MLP loop of 2 units +4.6..+8 (REV), of 1 unit
// +17.7; SW_W2_N2B 0 -0.2 (noise; 235 VGPRs with pipelining); ADDST 0 +2; SW_GROUP 1 -8.9 (13333: -12.2, 10303: -3.4);
// rolled head loop 215 VGPRs; SW_W2_COMMIT 0 237 VGPRs.
#ifndef SW_W2
#define SW_W2 1
#endif
#if SW_W2
#define SW_ENTRY_OVERRIDE sw_c64_w1_unused   // swin_block.hlsl's entry under another name (not compiled: not the entry)
// GELU (swin_block.hlsl's knobs, own defaults here): pattern 10333 (tile 0 of the even groups by HIP's chain, the rest by
// the arena table) with every group's table loads after W1 (SW_GELU_LATE0 3): 192 VGPRs = 4 workgroups per CU
#ifndef SW_GELU_PAT
#define SW_GELU_PAT 10333
#endif
#ifndef SW_GELU_LATE0
#define SW_GELU_LATE0 3
#endif
#endif
#include "swin_block.hlsl"

#if SW_W2
#if SW_C != 64 || THREADS != 128 || SW_ADD || SW_ECL > 0 || (GROUP != 2 && GROUP != 1) || SW_GELU != 3 || SW_NORM2_EARLY || SW_HEAD_PAIRS
#error "swin_c64w2.hlsl: the c64 block, 128 lanes"
#endif
#define NWW 2   // windows per wave
#ifndef SW_W2_MLP_STEP
#define SW_W2_MLP_STEP 4   // MLP units per loop pass (NMLP = 4: the whole MLP unrolled)
#endif
#ifndef SW_W2_HUNROLL
#define SW_W2_HUNROLL 1    // the head loop unrolled
#endif
#ifndef SW_W2_REV
#define SW_W2_REV 0        // the residual accumulators carried permuted through the MLP loop (SW_REV_PHI)
#endif
#ifndef SW_W2_N2B
#define SW_W2_N2B 1        // norm2's gamma2 / beta2 loads k by k behind the previous k's pack (SW_N2_BATCH 1)
#endif
#ifndef SW_W2_ADDST
#define SW_W2_ADDST SW_SMAX_ADDST   // the soft cap's bias additions as their own stage
#endif
#ifndef SW_W2_PIPE
#define SW_W2_PIPE 0       // 1: W1 of the next group issued before the GELU of this one (0: each group's W1 at its
                           // start; the other window's chain is the independent work)
#endif
#ifndef SW_W2_COMMIT
#define SW_W2_COMMIT SW_COMMIT_EARLY
#endif

WG_KERNEL(swin_block_c64_nc)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
    const int nwx = (W + ox + 3) >> 2, nwy = (H + oy + 3) >> 2, nwin = nwx * nwy;
    const uint wave_id = (uint)wg_uniform((int)(tid >> 5));

    Stager stage;
    st_fetch(stage, w + L_head(0), HEAD_BYTES / 16, tid);   // unit 0
#if SW_GL_LDS
    [unroll] for (uint gi = 0; gi < 256u / THREADS; ++gi) sw_gl_tab[tid + gi * THREADS] = sw_2q(wg_gl_entry(tid + gi * THREADS));
#endif

    // Gather: token t of each window, mirrored at the borders.
    int dst[NWW];
    uint2 xa[NWW][KC];
    [unroll] for (uint i = 0; i < NWW; ++i) {
        const int win0 = (int)(wg_block_id() * (THREADS / 32 * NWW) + wave_id * NWW + i);
        const int win = win0 < nwin ? win0 : nwin - 1;
        const int wy = win / nwx, wx = win - wy * nwx;
        const int pr = 4 * wy - oy + (int)(t >> 2), pc = 4 * wx - ox + (int)(t & 3u);
        const bool real = win0 < nwin && pr >= 0 && pr < H && pc >= 0 && pc < W;
        dst[i] = real ? pr * W + pc : -1;
        const int mr = sw_mirror(pr, H), mc = sw_mirror(pc, W);
        load_frags_g(x, (uint)(mr * W + mc) * SW_C + hh * HALF, xa[i]);
    }
    v8f ng[KC], nb[KC];
    {
        const uint g1 = w + L_G1 + hh * (HALF * 4), bp = w + L_BP + hh * 32u;
        [unroll] for (uint k = 0; k < KC; ++k) { ng[k] = wg_load8f(g1, 32u * k); nb[k] = wg_load8f(bp, 64u * k); }
    }
    st_commit(stage, 0u, HEAD_BYTES / 16, tid);
    wg_barrier_lds();
    uint buf = 0u;

    // Steps 1-2: a = q8(RMSNorm(x; gamma1)); the residual accumulator starts as x + bp.
    uint2 af[NWW][KC];
    v8f acc[NWW][KC];
    [unroll] for (uint i1 = 0; i1 < NWW; ++i1) {
        [unroll] for (uint k = 0; k < KC; ++k) acc[i1][k] = sw_unpack_fp8(xa[i1][k]);
        const float r = norm_r(acc[i1]);
        [unroll] for (uint k2 = 0; k2 < KC; ++k2) {
            v8f d;
            [unroll] for (uint v = 0; v < 8; ++v) d.v[v] = acc[i1][k2].v[v] * r * ng[k2].v[v];
            af[i1][k2] = pack_in(d);
            acc[i1][k2] = sw_add8(acc[i1][k2], nb[k2]);
        }
    }

    // Steps 3-7, one head at a time; one staged unit per head (QKV fragments, then the head's slice of Wp).
#if SW_W2_HUNROLL
    const uint nheads = SW_HEADS;
    [unroll] for (uint hd = 0; hd < nheads; ++hd) {
#else
    const uint nheads = SW_HEADS + sw_zero();
    [loop] for (uint hd = 0; hd < nheads; ++hd) {
#endif
        const bool last = hd + 1u == nheads;
        if (last) st_fetch(stage, w + L_mlp(0), MLP_BYTES / 16, tid);
        else st_fetch(stage, w + L_head(hd + 1u), HEAD_BYTES / 16, tid);
        const uint fb = sw_lane_base(buf * MAXU, lane);
        const v8f bias = wg_load8f(w + L_BIAS, ((hd * 16u + t) * 2u + hh) * 32u);
        uint2 uf[NWW][2];
        uint4 vt[NWW][2];
        [unroll] for (uint mt = 0; mt < 2; ++mt) {
            uint2 fq[KC], fv[KC];
            [unroll] for (uint kc = 0; kc < KC; ++kc) {
                fq[kc] = sw_lfrag_b(fb, (mt * KC + kc) * 2u);
                fv[kc] = sw_lfrag_b(fb, (mt * KC + kc) * 2u + 1u);
            }
            v8f accq[NWW], accv[NWW];
            [unroll] for (uint i = 0; i < NWW; ++i) { accq[i] = sw_zero8(); accv[i] = sw_zero8(); }
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (uint i2 = 0; i2 < NWW; ++i2) {
                    accq[i2] = wg_wmma_fp8(fq[kc2], af[i2][kc2], accq[i2]);
                    accv[i2] = wg_wmma_fp8(af[i2][kc2], fv[kc2], accv[i2]);
                }
            [unroll] for (uint i3 = 0; i3 < NWW; ++i3) {
                uf[i3][mt] = pack_in(accq[i3]);
                vt[i3][mt] = wg_to_f16_ags(accv[i3]);
            }
        }
        uint2 of[NWW][2];
        [unroll] for (uint i4 = 0; i4 < NWW; ++i4) {
            v8f s = wg_wmma_fp8(uf[i4][0], uf[i4][0], sw_zero8());   // S = U^T U
            s = wg_wmma_fp8(uf[i4][1], uf[i4][1], s);
            v8f c, tc;
#if SW_W2_ADDST
            [unroll] for (uint va0 = 0; va0 < 8; ++va0) c.v[va0] = s.v[va0] + bias.v[va0];
            [unroll] for (uint v = 0; v < 8; ++v) c.v[v] = min(max(c.v[v], asfloat(0xc20179dbu)), asfloat(0x420179dbu));
#else
            [unroll] for (uint v = 0; v < 8; ++v) c.v[v] = min(max(s.v[v] + bias.v[v], asfloat(0xc20179dbu)), asfloat(0x420179dbu));
#endif
            [unroll] for (uint v1 = 0; v1 < 8; ++v1) tc.v[v1] = c.v[v1] * c.v[v1];
            [unroll] for (uint v2 = 0; v2 < 8; ++v2) tc.v[v2] = mad(c.v[v2] * tc.v[v2], asfloat(0xb9a6cc38u), c.v[v2]);
            float mx = sw_hmax8(tc);
            mx = max(mx, wg_xor16(mx));
            mx = mx * asfloat(0x3f028297u);
            v8f e;
            [unroll] for (uint v4 = 0; v4 < 8; ++v4) e.v[v4] = mad(tc.v[v4], asfloat(0x3f028297u), -mx);
            [unroll] for (uint v5 = 0; v5 < 8; ++v5) e.v[v5] = wg_exp2(e.v[v5]);
            float sum = sw_hsum8(e);
            sum = sum + wg_xor16(sum);
            const float rs = wg_rcp(sum);
            v8f pm;
            [unroll] for (uint v6 = 0; v6 < 8; ++v6) pm.v[v6] = mad(e.v[v6], rs, 0.0f);
            const uint4 p = wg_to_f16_ags(pm);
            [unroll] for (uint dt = 0; dt < 2; ++dt) of[i4][dt] = pack_in(wg_wmma_f16(vt[i4][dt], p, sw_zero8()));
        }
        // x1 += Wp[:, head hd] q8(o_hd)
        uint2 fp[KC][2];
        [unroll] for (uint m = 0; m < KC; ++m) {
            fp[m][0] = sw_lfrag_b(fb, QKV_FRAGS + m * 2u);
            fp[m][1] = sw_lfrag_b(fb, QKV_FRAGS + m * 2u + 1u);
        }
        [unroll] for (uint m2 = 0; m2 < KC; ++m2)
            [unroll] for (uint i5 = 0; i5 < NWW; ++i5) {
                acc[i5][m2] = wg_wmma_fp8(fp[m2][0], of[i5][0], acc[i5][m2]);
                acc[i5][m2] = wg_wmma_fp8(fp[m2][1], of[i5][1], acc[i5][m2]);
            }
        if (last) st_commit(stage, (buf ^ 1u) * MAXU, MLP_BYTES / 16, tid);
        else st_commit(stage, (buf ^ 1u) * MAXU, HEAD_BYTES / 16, tid);
        wg_barrier_lds();
        buf ^= 1u;
    }

    // Step 8: m = q8(RMSNorm(x1; gamma2)), then b2 into the residual accumulator.
    uint2 mf[NWW][KC];
    {
        uint g2 = w + L_G2 + hh * (HALF * 4), b2 = w + L_B2 + hh * 32u;
        float r[NWW];
        [unroll] for (uint i = 0; i < NWW; ++i) r[i] = norm_r(acc[i]);
#if SW_W2_N2B
        [unroll] for (uint k = 0; k < KC; ++k) {
            if (k > 0u) {
                const uint dz = mf[NWW - 1u][k - 1u].x & sw_zero_bits();   // 0 at run time
                g2 += dz; b2 += dz;
            }
            const v8f g = wg_load8f(g2, 32u * k), b = wg_load8f(b2, 64u * k);
            [unroll] for (uint i1 = 0; i1 < NWW; ++i1) {
                v8f d;
                [unroll] for (uint v = 0; v < 8; ++v) d.v[v] = acc[i1][k].v[v] * r[i1] * g.v[v];
                mf[i1][k] = pack_in(d);
                acc[i1][k] = sw_add8(acc[i1][k], b);
            }
        }
#else
        v8f gg[KC], bb[KC];
        [unroll] for (uint k = 0; k < KC; ++k) { gg[k] = wg_load8f(g2, 32u * k); bb[k] = wg_load8f(b2, 64u * k); }
        [unroll] for (uint i1 = 0; i1 < NWW; ++i1) {
            [unroll] for (uint k1 = 0; k1 < KC; ++k1) {
                v8f d;
                [unroll] for (uint v = 0; v < 8; ++v) d.v[v] = acc[i1][k1].v[v] * r[i1] * gg[k1].v[v];
                mf[i1][k1] = pack_in(d);
                acc[i1][k1] = sw_add8(acc[i1][k1], bb[k1]);
            }
        }
#endif
    }

    // Steps 9-10: x2 = x1 + W2 q8(GELU(W1 m + b1)) + b2, GROUP hidden tiles of 16 at a time, software-pipelined by one
    // group within a unit (W1 of group j0 + GROUP before the GELU of group j0), as swin_block.hlsl.
    const uint b1 = w + L_B1 + hh * 32u;
#if SW_W2_REV
    v8f racc[NWW][KC];
    [unroll] for (uint ri = 0; ri < NWW; ++ri) [unroll] for (uint rk = 0; rk < KC; ++rk) racc[ri][rk] = sw_perm8(acc[ri][rk], 7u, 7u);
#endif
#if SW_W2_MLP_STEP == NMLP
    const uint nmlp = NMLP;
    [unroll] for (uint us = 0; us < nmlp; us += SW_W2_MLP_STEP) [unroll] for (uint ui = 0; ui < SW_W2_MLP_STEP; ++ui) {
#else
    const uint nmlp = NMLP + sw_zero();
    [loop] for (uint us = 0; us < nmlp; us += SW_W2_MLP_STEP) [unroll] for (uint ui = 0; ui < SW_W2_MLP_STEP; ++ui) {
#endif
        const uint u = us + ui;
        const bool more = ui + 1u < SW_W2_MLP_STEP || u + 1u < nmlp;
#if SW_W2_REV
        [unroll] for (uint ri0 = 0; ri0 < NWW; ++ri0) [unroll] for (uint rk0 = 0; rk0 < KC; ++rk0) acc[ri0][rk0] = sw_unperm8(racc[ri0][rk0], 7u, 7u);
#endif
        st_fetch(stage, w + L_mlp(more ? u + 1u : u), MLP_BYTES / 16, tid);
        v8f hcur[NWW][GROUP];
#if SW_W2_PIPE
        {
            const uint fb0 = sw_lane_base(buf * MAXU, lane);
            uint2 f1[GROUP][KC];
            [unroll] for (uint g = 0; g < GROUP; ++g)
                [unroll] for (uint kc = 0; kc < KC; ++kc) f1[g][kc] = sw_lfrag_b(fb0, g * 2u * KC + kc);
            v8f bh[GROUP];
            [unroll] for (uint g2 = 0; g2 < GROUP; ++g2) bh[g2] = wg_load8f(b1, (u * HPU + g2) * 64u);
            [unroll] for (uint i = 0; i < NWW; ++i) [unroll] for (uint g1 = 0; g1 < GROUP; ++g1) hcur[i][g1] = bh[g1];
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (uint i2 = 0; i2 < NWW; ++i2)
                    [unroll] for (uint g3 = 0; g3 < GROUP; ++g3) hcur[i2][g3] = wg_wmma_fp8(f1[g3][kc2], mf[i2][kc2], hcur[i2][g3]);
        }
#endif
        [unroll] for (uint j0 = 0; j0 < HPU; j0 += GROUP) {
            sw_gj = j0 / GROUP;
            const uint fb = sw_lane_base(buf * MAXU, lane) + j0 * (2u * KC * 32u);
#if !SW_W2_PIPE
            // W1 of this group for both windows (the two windows' chains are the independent work under the GELU)
            {
                uint2 f1[GROUP][KC];
                [unroll] for (uint g = 0; g < GROUP; ++g)
                    [unroll] for (uint kc = 0; kc < KC; ++kc) f1[g][kc] = sw_lfrag_b(fb, g * 2u * KC + kc);
                v8f bh[GROUP];
                [unroll] for (uint g2 = 0; g2 < GROUP; ++g2) bh[g2] = wg_load8f(b1, (u * HPU + j0 + g2) * 64u);
                [unroll] for (uint i = 0; i < NWW; ++i) [unroll] for (uint g1 = 0; g1 < GROUP; ++g1) hcur[i][g1] = bh[g1];
                [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                    [unroll] for (uint i2 = 0; i2 < NWW; ++i2)
                        [unroll] for (uint g3 = 0; g3 < GROUP; ++g3) hcur[i2][g3] = wg_wmma_fp8(f1[g3][kc2], mf[i2][kc2], hcur[i2][g3]);
            }
#endif
            uint2 f2[GROUP][KC];
            [unroll] for (uint g4 = 0; g4 < GROUP; ++g4)
                [unroll] for (uint m = 0; m < KC; ++m) f2[g4][m] = sw_lfrag_b(fb, g4 * 2u * KC + KC + m);
            const bool late = (SW_GELU_LATE0 >> ((j0 / GROUP) % 2u)) & 1u;
            SwGeluPre gp[NWW];
            if (!late) [unroll] for (uint ia = 0; ia < NWW; ++ia) gp[ia] = sw_gelu_pre(hcur[ia]);
#if SW_W2_PIPE
            v8f hnext[NWW][GROUP];
            if (j0 + GROUP < HPU) {
                const uint fbn = fb + GROUP * (2u * KC * 32u);
                uint2 f1[GROUP][KC];
                [unroll] for (uint g = 0; g < GROUP; ++g)
                    [unroll] for (uint kc = 0; kc < KC; ++kc) f1[g][kc] = sw_lfrag_b(fbn, g * 2u * KC + kc);
                v8f bh[GROUP];
                [unroll] for (uint g2 = 0; g2 < GROUP; ++g2) bh[g2] = wg_load8f(b1, (u * HPU + j0 + GROUP + g2) * 64u);
                [unroll] for (uint i = 0; i < NWW; ++i) [unroll] for (uint g1 = 0; g1 < GROUP; ++g1) hnext[i][g1] = bh[g1];
                [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                    [unroll] for (uint i2 = 0; i2 < NWW; ++i2)
                        [unroll] for (uint g3 = 0; g3 < GROUP; ++g3) hnext[i2][g3] = wg_wmma_fp8(f1[g3][kc2], mf[i2][kc2], hnext[i2][g3]);
            } else {
                [unroll] for (uint i7 = 0; i7 < NWW; ++i7) [unroll] for (uint g7 = 0; g7 < GROUP; ++g7) hnext[i7][g7] = hcur[i7][g7];
            }
#endif
            uint2 hf[NWW][GROUP];
            [unroll] for (uint ib = 0; ib < NWW; ++ib) {
                if (late) gp[ib] = sw_gelu_pre(hcur[ib]);
                sw_gelu_fin(hcur[ib], gp[ib], hf[ib]);
            }
            [unroll] for (uint m2 = 0; m2 < KC; ++m2)
                [unroll] for (uint i6 = 0; i6 < NWW; ++i6)
                    [unroll] for (uint g6 = 0; g6 < GROUP; ++g6) acc[i6][m2] = wg_wmma_fp8(f2[g6][m2], hf[i6][g6], acc[i6][m2]);
#if SW_W2_PIPE
            [unroll] for (uint i8 = 0; i8 < NWW; ++i8) [unroll] for (uint g8 = 0; g8 < GROUP; ++g8) hcur[i8][g8] = hnext[i8][g8];
#endif
#if SW_W2_COMMIT
            if (more && j0 == GROUP * (SW_W2_COMMIT - 1)) st_commit(stage, (buf ^ 1u) * MAXU, MLP_BYTES / 16, tid);
#endif
        }
        if (more) {
#if !SW_W2_COMMIT
            st_commit(stage, (buf ^ 1u) * MAXU, MLP_BYTES / 16, tid);
#endif
            wg_barrier_lds();
            buf ^= 1u;
        }
#if SW_W2_REV
        [unroll] for (uint ri1 = 0; ri1 < NWW; ++ri1) [unroll] for (uint rk1 = 0; rk1 < KC; ++rk1) racc[ri1][rk1] = sw_perm8(acc[ri1][rk1], 7u, 7u);
#endif
    }
#if SW_W2_REV
    [unroll] for (uint ri2 = 0; ri2 < NWW; ++ri2) [unroll] for (uint rk2 = 0; rk2 < KC; ++rk2) acc[ri2][rk2] = sw_unperm8(racc[ri2][rk2], 7u, 7u);
#endif

    // Output q8(x2), real window members only.
    [unroll] for (uint io = 0; io < NWW; ++io) {
        uint2 o[KC];
        float am = 0.0f;
        [unroll] for (uint k0 = 0; k0 < KC; ++k0) am = sw_absmax8(am, acc[io][k0]);
        if (sw_in448(am)) {
            [unroll] for (uint k = 0; k < KC; ++k) o[k] = wg_pack_q8_nc_ags(acc[io][k]);
        } else {
            [unroll] for (uint k = 0; k < KC; ++k) o[k] = sw_pack_q8_v2(acc[io][k]);
        }
        if (dst[io] >= 0) {
            const uint ob = (uint)dst[io] * SW_C + hh * HALF;
            [unroll] for (uint i = 0; i < KC / 2; ++i) wg_st128(out_p, ob + 16u * i, uint4(o[2 * i], o[2 * i + 1]));
        }
    }
}
#endif
