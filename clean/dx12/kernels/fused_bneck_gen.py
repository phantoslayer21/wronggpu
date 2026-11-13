# SPDX-License-Identifier: MIT
"""Generates fused_bneck.hlsl (this directory) from bneck.hlsl: enc2's patch merge (patch_merge_c96) as a first phase of
the persistent bottleneck kernel (merge items ahead of b0's in the queue, with their own row counters).

    python clean/dx12/kernels/fused_bneck_gen.py

build_dx12.py runs it before every build (every replacement asserts that its anchor is unique); after bneck.hlsl changes
compare and
re-measure WG_D3D12_FUSION=pm,xe,mb against pm,xe: the item loop of this kernel is sensitive to how the driver compiles
it (equivalent formulations of the scheduler changes measured from -10 to +80 us per frame).
"""
import os
K = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(K, "bneck.hlsl"), encoding="utf-8").read()
i = s.index('#include "wg_platform.hlsli"')
hdr = """\
// FUSION track: the persistent bottleneck kernel with enc2's patch merge (patch_merge_c96) as its first phase
// (GENERATED from bneck.hlsl by fused_bneck_gen.py, do not edit; the block arithmetic of b0..b5 and the item loop are
// those of bneck.hlsl):
//   bneck/swin_bneck6_mb_nc  extra arguments: mx = enc2's 96-channel map (skip2, 2H x 2W tokens, row pitch W4), mw = the
//                            merge weights (enc2's tail: bias, then the fragments), W4 = mx's width. The queue starts
//                            with nm = (items of the odd window grid) merge items: 8 windows of that grid each, one per
//                            wave, whose real tokens are merged from mx and stored to x (the merged map, still
//                            written), counted per window row in their own row counters (sync rows 6 NR .. 7 NR - 1,
//                            after the blocks'). b0's items wait on them exactly as b1's wait on b0's (b0 uses the even
//                            grid, so its dependency rows are computed by the same code), then read x as in bneck.hlsl.
//                            The sync words need 16 + 7 NR words (the runtime allocates them).
// The merge code sits before the item loop on purpose: placed inside it (a prologue of b0's items), the kernel measured
// 40 % slower even with the merge never executed. The item loop's speed depends strongly on how the driver compiles it
// (equivalent forms of the scheduler changes measured -10 .. +12 us, 2 windows per wave in the merge items +75 us, the
// heads rolled into a loop of 2 x 2 (a smaller loop) +80 us per frame); this form keeps the loop's 216 VGPRs.
// The 48 KB of merge fragments are staged per merge item in two 24 KB passes (tiles 0..3, 4..7) through the LDS unit
// buffers. The merge arithmetic is that of patch_merge_c96 (swin_gemm.hlsl, HIP patch_merge<96, 128>): the bias as the
// first accumulator, the 24 K groups in order, the clamping FP8 pack (pack_q8_out = sw_pack_q8): bit-identical.
// Forward progress: merge items precede every block item in the queue and wait on nothing, so the argument of
// bneck.hlsl (an item only waits on smaller items, all held by running workgroups) is unchanged.
//
// @wg-kernel bneck/swin_bneck6_mb_nc co=swin__bmegahg.co entry=swin_bneck6_mb_nc block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,p,p,u
//   bneck/swin_bneck6_mbo_nc (MB_XO 1, fusion xo) the same, but the last workgroup leaves block 5's row counters: dec2's
//                            block (swin_block_c96_add_xe_xo_nc, launched without a barrier) waits on them and resets them.
// @wg-kernel bneck/swin_bneck6_mbo_nc co=swin__bmegahg.co entry=swin_bneck6_mb_nc block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u,p,p,u defines=MB_XO=1
"""
s = hdr + s[i:]


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''    int a_H, a_W, a_oxe, a_oye, a_oxo, a_oyo, a_groups;
WG_ARGS_END''', '''    int a_H, a_W, a_oxe, a_oye, a_oxo, a_oyo, a_groups;
    uint a_mx, a_mw;         // merge source (enc2's map), merge weights (enc2's tail: bias, then the fragments)
    int a_W4;                // the merge source's width
WG_ARGS_END''')

rep('''groupshared uint2 lds[4096];''', '''groupshared uint2 lds[4096];
#if BN_GELU == 2
#error "fused_bneck.hlsl: the merge staging uses the whole LDS (BN_GELU 1 or 0 only)"
#endif''')

# Grid must be declared before merge_item: move the scheduler's struct up
GRID = '''struct Grid { int oy, ox, nwx, nwy, nwin, items; };
Grid grid_of(int H, int W, int ox, int oy)
{
    Grid g;
    g.ox = ox;
    g.oy = oy;
    g.nwx = (W + ox + 3) >> 2;
    g.nwy = (H + oy + 3) >> 2;
    g.nwin = g.nwx * g.nwy;
    g.items = (g.nwin + 7) / 8;
    return g;
}
'''
rep(GRID, '')

# the merge phase's helpers (after frag, st_lds4, mirror, pack_q8_out)
rep('''// ---- block steps (swin_v2.hip norm1 / attention / mlp / norm2 / store, WPW 1, SAFE) ------------------------------------''',
GRID + '''
// ---- enc2's patch merge (patch_merge_c96: merged(i, j) = q8(Wm concat(TL, TR, BL, BR) + bm)) ---------------------------
// Lane (t, hh) of a wave: merged token (mr, mc); lane half hh holds the 2 x 96 bytes of input row 2 mr + hh (TL | TR or
// BL | BR), 24 K groups, as patch_merge_c96. Output tile mt (16 channels) gives this lane's bytes hh 64 + 8 mt of the
// merged token.
#ifndef MB_CH
#define MB_CH 8u    // K groups per fragment batch (a divisor of 24): 2 tiles x MB_CH fragments loaded before their multiplies
#endif
void merge_tiles(uint p, uint2 zf[24], inout uint2 res[8], uint bm, uint lb)
{
    [unroll] for (uint mp = 0; mp < 4u; mp += 2u) {
        const uint mt = p * 4u + mp;
        v8f m0 = wg_load8f(bm, mt * 64u), m1 = wg_load8f(bm, (mt + 1u) * 64u);
        [unroll] for (uint g0 = 0; g0 < 24u; g0 += MB_CH) {
            uint2 fa[MB_CH], fb[MB_CH];
            [unroll] for (uint g = 0; g < MB_CH; ++g) {
                fa[g] = frag(0u, mp * 24u + g0 + g, lb);
                fb[g] = frag(0u, (mp + 1u) * 24u + g0 + g, lb);
            }
            [unroll] for (uint g2 = 0; g2 < MB_CH; ++g2) {
                m0 = wg_wmma_fp8(fa[g2], zf[g0 + g2], m0);
                m1 = wg_wmma_fp8(fb[g2], zf[g0 + g2], m1);
            }
        }
        res[mt] = pack_q8_out(m0);
        res[mt + 1u] = pack_q8_out(m1);
    }
}
// One merge item: 8 MB_WPW windows of the grid g (window wf + wave + 8 j for j < MB_WPW in each wave); the merged tokens
// of the real window members are stored to the merged map (coherent stores: b0's items read them in this launch). The
// 48 KB of fragments go through LDS in two 24 KB passes (tiles 0..3, 4..7). Starts with an LDS barrier (the claim slot
// lives in buffer 0); ends with every wave's stores complete and the LDS free.
#ifndef MB_WPW
#define MB_WPW 1u   // windows per wave of a merge item (2: the inputs are loaded again for the second pass;
                    // measured 75 us per frame slower: the item loop compiled differently)
#endif
void merge_load(uint zo, out uint2 zf[24])
{
    [unroll] for (uint q = 0; q < 12; ++q) {
        uint4 v = wg_ld128(a_mx, zo + 16u * q);
        zf[2 * q] = v.xy; zf[2 * q + 1] = v.zw;
    }
}
void merge_item(uint xo, int wf, int wl, Grid g, int H, int W, uint tid, uint wave, uint t, uint hh, uint lb)
{
    uint zo[MB_WPW];
    int dst[MB_WPW];
    [unroll] for (uint j = 0; j < MB_WPW; ++j) {
        const int win = wf + (int)wave + 8 * (int)j;
        const int wn = win <= wl ? win : wl;
        const int wy = wn / g.nwx, wx = wn - wy * g.nwx;
        const int pr = 4 * wy - g.oy + (int)(t >> 2), pc = 4 * wx - g.ox + (int)(t & 3u);
        const bool real = win <= wl && pr >= 0 && pr < H && pc >= 0 && pc < W;
        dst[j] = real ? pr * W + pc : -1;
        zo[j] = (uint)((2 * mirror(pr, H) + (int)hh) * a_W4 + 2 * mirror(pc, W)) * 96u;
    }
    const uint fw = a_mw + 4u * 128u;   // the fragments (after the bias)
    uint4 sv[6];
    [unroll] for (uint i = 0; i < 6; ++i) sv[i] = wg_ld128(fw, (tid + i * 256u) * 16u);
    uint2 zf[24];
    merge_load(zo[0], zf);
    wg_barrier_lds();
    [unroll] for (uint i1 = 0; i1 < 6; ++i1) st_lds4(tid * 2u + i1 * 512u, sv[i1]);
    [unroll] for (uint i2 = 0; i2 < 6; ++i2) sv[i2] = wg_ld128(fw, 24576u + (tid + i2 * 256u) * 16u);
    wg_barrier_lds();
    const uint bm = a_mw + hh * 32u;
    uint2 res[MB_WPW][8];
    [unroll] for (uint j1 = 0; j1 < MB_WPW; ++j1) {
        if (j1 > 0u) merge_load(zo[j1], zf);
        merge_tiles(0u, zf, res[j1], bm, lb);
    }
    wg_barrier_lds();
    [unroll] for (uint i3 = 0; i3 < 6; ++i3) st_lds4(tid * 2u + i3 * 512u, sv[i3]);
    wg_barrier_lds();
    [unroll] for (uint j2 = 0; j2 < MB_WPW; ++j2) {
        if (MB_WPW > 1u) merge_load(zo[j2], zf);
        merge_tiles(1u, zf, res[j2], bm, lb);
    }
    [unroll] for (uint j3 = 0; j3 < MB_WPW; ++j3) {
        if (dst[j3] >= 0) {
            const uint mo = xo + (uint)dst[j3] * 128u + hh * 64u;
            [unroll] for (uint i5 = 0; i5 < 4; ++i5)
                wg_mem_gc.Store4(mo + 16u * i5, uint4(res[j3][2 * i5], res[j3][2 * i5 + 1]));
        }
    }
    wg_barrier();   // every wave's stores are complete (device scope) and the LDS is free
}

// ---- block steps (swin_v2.hip norm1 / attention / mlp / norm2 / store, WPW 1, SAFE) ------------------------------------''')

rep('''WG_KERNEL(swin_bneck6_nc)''', '''WG_KERNEL(swin_bneck6_mb_nc)''')

# queue: nm merge items (8 windows of the odd grid go each: b0's items wait on their rows as b1's on b0's), then the
# blocks; row counters: the merge's at block index 0, block k's at k + 1
rep('''    const int total = 3 * (ge.items + go.items);''', '''    // merge items first in the queue (mx = 0: none, b0 reads x written before the launch, as bneck.hlsl; diagnostics)
    const int nm = a_mx != 0u ? (go.nwin + 8 * (int)MB_WPW - 1) / (8 * (int)MB_WPW) : 0;
    const int total = nm + 3 * (ge.items + go.items);''')

rep('''    int cur = WaveReadLaneFirst((int)lds[SLOT_EL].x);
    const uint lb = WaveGetLaneIndex();''', '''    int cur = WaveReadLaneFirst((int)lds[SLOT_EL].x);
    const uint lb = WaveGetLaneIndex();
    // ---- phase 1: merge items (enc2's patch merge into the merged map x); their row counters after the blocks' (6 NR ..)
    [loop] while (cur < nm) {
        uint mnext = (uint)total;
        if (tid == 0u) mnext = wg_atomic_add(a_sync, 0u, 1u);
        const int wf = cur * 8 * (int)MB_WPW, wl = (wf + 8 * (int)MB_WPW < go.nwin ? wf + 8 * (int)MB_WPW : go.nwin) - 1;
        merge_item(a_x, wf, wl, go, H, W, tid, wave, t, hh, lb);
        if (tid == 0u) {
            const int r0 = wf / go.nwx, r1 = wl / go.nwx;
            [loop] for (int rr = r0; rr <= r1; ++rr) {
                const int s0 = rr * go.nwx > wf ? rr * go.nwx : wf;
                const int s1 = (rr + 1) * go.nwx - 1 < wl ? (rr + 1) * go.nwx - 1 : wl;
                wg_atomic_add(rows, (uint)(6 * NR + rr) * 4u, (uint)(s1 - s0 + 1));
            }
            lds[SLOT_EL + 1u].x = mnext;
        }
        wg_barrier_lds();
        cur = WaveReadLaneFirst((int)lds[SLOT_EL + 1u].x);
    }''')

rep('''        int k = 0, local = cur;''', '''        int k = 0, local = cur - nm;''')
rep('''        if (k > 0 && tid == 0u) {''', '''        if ((k > 0 || nm > 0) && tid == 0u) {   // (b0: the rows of the merge items)''')
rep('''            const uint rc = rows + (uint)((k - 1) * NR) * 4u;''', '''            const uint rc = rows + (uint)((k > 0 ? k - 1 : 6) * NR) * 4u;''')
rep('''            for (int j = 0; j < 6 * NR; ++j) wg_st32_gc(rows, (uint)j * 4u, 0u);''',
    '''            for (int j = 0; j < 7 * NR; ++j)
                if (!MB_XO || j < 5 * NR || j >= 6 * NR) wg_st32_gc(rows, (uint)j * 4u, 0u);   // xo: not block 5's''')
rep('''groupshared uint2 lds[4096];''', '''#ifndef MB_XO
#define MB_XO 0
#endif
groupshared uint2 lds[4096];''')



D = os.path.join(K, "fused_bneck.hlsl")
if not os.path.exists(D) or open(D, encoding="utf-8").read() != s:
    open(D, "w", encoding="utf-8").write(s)
    print("fused_bneck.hlsl regenerated")
else:
    print("fused_bneck.hlsl up to date")
