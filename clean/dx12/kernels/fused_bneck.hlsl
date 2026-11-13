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
#include "wg_platform.hlsli"
#include "bneck_gelu.hlsli"   // wg_gelu_lut.hlsli's exact table with one VALU op fewer per value pair

WG_ARGS_BEGIN
    uint a_x, a_out, a_tmp, a_sync, a_w0, a_w1, a_w2, a_w3, a_w4, a_w5;   // pointers
    int a_H, a_W, a_oxe, a_oye, a_oxo, a_oyo, a_groups;
    uint a_mx, a_mw;         // merge source (enc2's map), merge weights (enc2's tail: bias, then the fragments)
    int a_W4;                // the merge source's width
WG_ARGS_END

// Packed block section of C = 128, 4 heads, MLP 512 (FORMATS.md; swin_v2.hip Lay<128, 4, 512>): byte offsets.
#define KC 8
#define SEC_G1 16u
#define SEC_G2 528u
#define SEC_BP 1040u
#define SEC_B2 1552u
#define SEC_B1 2064u
#define SEC_BIAS 4112u
#define SEC_STREAM 8208u
#define HEAD_BYTES 12288u
#define TILE_BYTES 4096u
#define SEC_HEAD(hd) (SEC_STREAM + (hd) * HEAD_BYTES)
#define SEC_TILE(ht) (SEC_STREAM + 4u * HEAD_BYTES + (ht) * TILE_BYTES)
#ifndef BN_GELU
#define BN_GELU 1             // 0: transcendental, 1: 16 KB arena table (bneck_gelu.hlsli, default: fastest here),
                              // 2: 2 KB groupshared table (wg_gelu_lds.hlsli)
#endif
#ifndef BN_TPU
#if BN_GELU == 2
#define BN_TPU 2u             // hidden tiles per MLP unit: 2 (8 KB) leaves the buffer tails free for the GELU table
#else
#define BN_TPU 4u
#endif
#endif
#define NMLP (32u / BN_TPU)   // MLP units of BN_TPU hidden tiles (4096 bytes each)
#define MLP_ROWS BN_TPU       // 16-byte chunks per lane of an MLP unit
#define HEAD_ROWS 3u          // 12288 bytes per head unit (with BN_GELU 2 the tails of both buffers are never staged into)
#define BUF_EL 2048u          // uint2 elements of one LDS unit buffer (16384 bytes)
#define SLOT_EL 1536u         // byte 12288 of buffer 0: claim slots
#define SYNC_HEAD 16          // sync words: [0] queue, [1] finished workgroups, [15] spin timeouts (never reset),
                              // [16 + k NR + r] row counters
#ifndef BNECK_SPIN_LIMIT
#define BNECK_SPIN_LIMIT (1u << 18)
#endif

// soft cap / softmax / GELU constants of swin.hip (bits as in HIP's ISA)
#define K_CAP asfloat(0x420179dbu)     // 32.369
#define K_NC3 asfloat(0xb9a6cc38u)     // -1 / (3 cap^2)
#define K_TL asfloat(0x3f028297u)      // tau log2(e)
#define K_EPS asfloat(0x39000000u)     // 2^-13
#define K_INVC asfloat(0x3c000000u)    // 1 / 128
#define K_GA asfloat(0xbdcd6502u)      // -0.10029031
#define K_GB asfloat(0xc013b8edu)      // -2.30816186

#ifndef MB_XO
#define MB_XO 0
#endif
groupshared uint2 lds[4096];
#if BN_GELU == 2
#error "fused_bneck.hlsl: the merge staging uses the whole LDS (BN_GELU 1 or 0 only)"
#endif
#if BN_GELU == 2
#define GL_EL (BUF_EL + 1536u)   // the GELU table: bytes 12288 .. 14335 of buffer 1 (filled once per workgroup)
#define WG_GL_GET(off) lds[GL_EL + ((off) >> 3)]
#include "wg_gelu_lds.hlsli"
#endif
// LDS addresses are formed as (per-lane index) + constant, with the per-lane index from WaveGetLaneIndex() / the thread
// id, whose ranges DXC does not know: it keeps the additions (with lane & 31 it turns them into ORs, and the driver then
// materializes one address register per fragment instead of using the instructions' offset field).

// ---- small helpers ----------------------------------------------------------------------------------------------------
float hsum8(v8f a) { return ((a.v[0] + a.v[1]) + (a.v[2] + a.v[3])) + ((a.v[4] + a.v[5]) + (a.v[6] + a.v[7])); }
// hsum8(a * a) as HIP's ISA forms it: each pair a_2i^2 + a_2i+1^2 is fma(a_2i, a_2i, a_2i+1 * a_2i+1)
float hsum_sq(v8f a)
{
    return (mad(a.v[0], a.v[0], a.v[1] * a.v[1]) + mad(a.v[2], a.v[2], a.v[3] * a.v[3])) +
           (mad(a.v[4], a.v[4], a.v[5] * a.v[5]) + mad(a.v[6], a.v[6], a.v[7] * a.v[7]));
}
float hmax8(v8f a)
{
    return max(max(max(a.v[0], a.v[1]), max(a.v[2], a.v[3])), max(max(a.v[4], a.v[5]), max(a.v[6], a.v[7])));
}
// FP8 -> FP32 of 8 values (HIP v_cvt_pk_f32_fp8): the driver's matrix element conversion (FP8 B -> FP32 accumulator,
// lane by lane, element i = byte i; the bits of wg_unpack_fp8 for all 256 codes: work/bneck/probe). Wave-uniform flow.
v8f unpack8(uint2 p) { return wg_v8f_frag(wg_ags_convert(wg_frag2(p), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC)); }
// HIP wg_pack_q8 (v_maximumminimum_f32 x, -448, 448, then v_cvt_pk_fp8_f32): the clamp keeps a NaN, which the conversion
// turns into 0xff whatever its sign or payload (as the hardware conversion of HIP's quieted NaN)
uint2 pack_q8_out(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = d.v[i] == d.v[i] ? min(max(d.v[i], -448.0f), 448.0f) : d.v[i];
    return wg_pack_q8_nc_ags(d);
}
int mirror(int p, int n)
{
    p = p < 0 ? -p : p;
    return p >= n ? 2 * (n - 1) - p : p;
}
// Fragment `index` of the unit at LDS element `el`: the lane's 8 bytes (lb = the lane).
uint2 frag(uint el, uint index, uint lb) { return lds[(lb + el) + index * 32u]; }

// Weight staging: a 16384-byte unit copied global -> LDS in four rows of 16-byte chunks (row i = chunk tid + 256 i), spread
// over `points` steps: step(p) writes the rows fetched at step p - 1 and issues the global loads of rows lo(p)..lo(p+1)-1,
// lo(p) = 4 p / points (points 3: rows 0 | 1 | 2, 3; points 4: one row per step). p and points are constants.
void st_lds4(uint e, uint4 v) { lds[e] = v.xy; lds[e + 1u] = v.zw; }
void stage_step_r(uint p, uint points, uint rows, uint src, uint dst_el, uint tid, inout uint4 r[4])
{
    if (p > 0u) {
        [unroll] for (uint i = (p - 1u) * rows / points; i < p * rows / points; ++i) st_lds4(tid * 2u + (dst_el + i * 512u), r[i]);
    }
    if (p < points) {
        [unroll] for (uint i = p * rows / points; i < (p + 1u) * rows / points; ++i)
            r[i] = wg_ld128(src, (tid + i * 256u) * 16u);
    }
}
void stage_step(uint p, uint points, uint src, uint dst_el, uint tid, inout uint4 r[4])
{
    stage_step_r(p, points, 4u, src, dst_el, tid, r);
}

struct Grid { int oy, ox, nwx, nwy, nwin, items; };
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

// ---- block steps (swin_v2.hip norm1 / attention / mlp / norm2 / store, WPW 1, SAFE) ------------------------------------
// The RMSNorm output and the bias add of both norms: n = (a rs) g, q = q8(n), a = a + b, value by value (the operations of
// HIP's ISA), two k at a time: four 32-byte loads per batch from per-lane bases made opaque once (the driver then puts
// the constant parts in the instructions' offset field) and the products stage by stage over 16 values.
void norm_apply(inout v8f acc[KC], float rs, uint sec, uint hh, uint goff, uint boff, out uint2 q[KC])
{
    const uint zb = wg_grid_ >> 31;
    const uint gb = (sec + hh * 256u) ^ zb, bb = (sec + hh * 32u) ^ zb;
    [unroll] for (uint k = 0; k < KC; k += 2u) {
        v8f g0 = wg_load8f(gb, goff + k * 32u), g1 = wg_load8f(gb, goff + (k + 1u) * 32u);
        v8f b0 = wg_load8f(bb, boff + k * 64u), b1 = wg_load8f(bb, boff + (k + 1u) * 64u);
        v8f n0, n1;
        [unroll] for (uint v = 0; v < 8; ++v) { n0.v[v] = acc[k].v[v] * rs; n1.v[v] = acc[k + 1u].v[v] * rs; }
        [unroll] for (uint v1 = 0; v1 < 8; ++v1) { n0.v[v1] = n0.v[v1] * g0.v[v1]; n1.v[v1] = n1.v[v1] * g1.v[v1]; }
        q[k] = wg_pack_q8_nc_ags(n0);
        q[k + 1u] = wg_pack_q8_nc_ags(n1);
        [unroll] for (uint v2 = 0; v2 < 8; ++v2) {
            acc[k].v[v2] = acc[k].v[v2] + b0.v[v2];
            acc[k + 1u].v[v2] = acc[k + 1u].v[v2] + b1.v[v2];
        }
    }
}
// Steps 1-2: acc = x + bp, af = q8(RMSNorm(x; gamma1)) (no clamp: SAFE).
void norm1(uint4 x[4], uint sec, uint hh, out v8f acc[KC], out uint2 af[KC])
{
    float ss = 0.0f;
    [unroll] for (uint k = 0; k < KC; ++k) {
        acc[k] = unpack8((k & 1u) ? x[k >> 1].zw : x[k >> 1].xy);
        ss = ss + hsum_sq(acc[k]);
    }
    ss = ss + wg_xor16(ss);
    float rs = rsqrt((ss + K_EPS) * K_INVC);
    norm_apply(acc, rs, sec, hh, SEC_G1, SEC_BP, af);
}

// Step 8 and b2: mf = q8(RMSNorm(x1; gamma2)), acc = x1 + b2.
void norm2(inout v8f acc[KC], uint sec, uint hh, out uint2 mf[KC])
{
    float ss = 0.0f;
    [unroll] for (uint k = 0; k < KC; ++k) ss = ss + hsum_sq(acc[k]);
    ss = ss + wg_xor16(ss);
    float rs = rsqrt((ss + K_EPS) * K_INVC);
    norm_apply(acc, rs, sec, hh, SEC_G2, SEC_B2, mf);
}

// Steps 3-7 for head hd: U = LDS element of the head's unit; the next unit (16384 bytes from global `ssrc`) is staged into
// LDS element `sdst` in three steps.
void attention(uint U, uint sec, uint hd, uint2 af[KC], inout v8f acc[KC], uint lb, uint t, uint hh, uint tid,
               uint ssrc, uint sdst, inout uint4 sr[4])
{
    const uint nrows = hd < 3u ? HEAD_ROWS : MLP_ROWS;   // hd is a constant at every call (heads unrolled)
    stage_step_r(0u, 3u, nrows, ssrc, sdst, tid, sr);
    uint2 uf[2];
    uint4 vt[2];
    [unroll] for (uint mt = 0; mt < 2; ++mt) {
        if (mt == 1u) stage_step_r(1u, 3u, nrows, ssrc, sdst, tid, sr);
        uint2 fq[KC], fv[KC];
        [unroll] for (uint kc = 0; kc < KC; ++kc) {
            fq[kc] = frag(U, (mt * KC + kc) * 2u, lb);
            fv[kc] = frag(U, (mt * KC + kc) * 2u + 1u, lb);
        }
        v8f aq = wg_splat(0.0f), av = wg_splat(0.0f);
        [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2) {
            aq = wg_wmma_fp8(fq[kc2], af[kc2], aq);
            av = wg_wmma_fp8(af[kc2], fv[kc2], av);
        }
        uf[mt] = wg_pack_q8_nc_ags(aq);
        vt[mt] = wg_to_f16_ags(av);
    }
    // S = U^T U, c = clamp(S + B), z = tau log2e (c - c^3 / (3 cap^2)), softmax with the max subtracted
    v8f bias = wg_load8f(sec, SEC_BIAS + ((hd * 16u + t) * 2u + hh) * 32u);
    v8f s = wg_wmma_fp8(uf[0], uf[0], wg_splat(0.0f));
    s = wg_wmma_fp8(uf[1], uf[1], s);
    // (stage by stage over the 8 values, as HIP's ISA: the driver pairs them into VOPD)
    v8f zp, zm, cc, c2;
    [unroll] for (uint v = 0; v < 8; ++v) cc.v[v] = s.v[v] + bias.v[v];
    [unroll] for (uint v0 = 0; v0 < 8; ++v0) cc.v[v0] = min(max(cc.v[v0], -K_CAP), K_CAP);
    [unroll] for (uint v1 = 0; v1 < 8; ++v1) c2.v[v1] = cc.v[v1] * cc.v[v1];
    [unroll] for (uint v4 = 0; v4 < 8; ++v4) c2.v[v4] = cc.v[v4] * c2.v[v4];
    [unroll] for (uint v5 = 0; v5 < 8; ++v5) zp.v[v5] = mad(c2.v[v5], K_NC3, cc.v[v5]);
    [unroll] for (uint v6 = 0; v6 < 8; ++v6) zm.v[v6] = zp.v[v6] * K_TL;
    float mx = hmax8(zm);
    mx = max(mx, wg_xor16(mx));
    v8f e;
    [unroll] for (uint v2 = 0; v2 < 8; ++v2) e.v[v2] = exp2(mad(K_TL, zp.v[v2], -mx));
    float sum = hsum8(e);
    sum = sum + wg_xor16(sum);
    float r = rcp(sum);
    // p = (f16)(r e) with one rounding: HIP's v_fma_mixlo/hi_f16 r, e, 0. The driver folds mad(r, e, 0) into the matrix
    // FP16 conversion as exactly that instruction (one v_fma_mix per value; the build must show 32 per head). Same bits as
    // the earlier form fma(r, e, C) - C on the FP16 grid of r e's binade for every FP32 e in [2^-126, 1) and 24 sums s in
    // [1, 16] (r = rcp(s); work tool pr/prt.hlsl, 2^30 values per r, 0 differences).
    v8f po;
    [unroll] for (uint v3 = 0; v3 < 8; ++v3) po.v[v3] = mad(r, e.v[v3], 0.0f);
    uint4 pv = wg_to_f16_ags(po);
    uint2 of[2];
    [unroll] for (uint dt = 0; dt < 2; ++dt) of[dt] = wg_pack_q8_nc_ags(wg_wmma_f16(vt[dt], pv, wg_splat(0.0f)));
    // x1 += Wp[:, head hd] q8(o)
    stage_step_r(2u, 3u, nrows, ssrc, sdst, tid, sr);
    uint2 fp[2 * KC];
    [unroll] for (uint i = 0; i < 2 * KC; ++i) fp[i] = frag(U, 4u * KC + i, lb);
    // (dt outer, as HIP's ISA orders them: no two dependent multiplies back to back; each acc[m] still adds dt 0, then 1)
    [unroll] for (uint dt2 = 0; dt2 < 2; ++dt2)
        [unroll] for (uint m = 0; m < KC; ++m) acc[m] = wg_wmma_fp8(fp[2u * m + dt2], of[dt2], acc[m]);
    stage_step_r(3u, 3u, nrows, ssrc, sdst, tid, sr);
}

// W1 of hidden tile j of the unit at LDS element T: h = W1 m + b1 (the bias is the first accumulator).
v8f w1(uint T, uint j, uint b1, uint2 mf[KC], uint lb)
{
    v8f h = wg_load8f(b1, j * 64u);
    uint2 f[KC];
    [unroll] for (uint kc = 0; kc < KC; ++kc) f[kc] = frag(T, j * 16u + kc, lb);
    [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2) h = wg_wmma_fp8(f[kc2], mf[kc2], h);
    return h;
}

// GELU (wg_platform.h mode 2, sigmoid form) x / (1 + 2^(x (-0.1003 x^2 - 2.308))) and q8 (swin.hip pack_in<SAFE>: no
// clamp) of one hidden tile, step by step over the 8 values (per value the operations of HIP's ISA in its order:
// x^2, fma(-0.1003, x^2, -2.308), x t, exp2, 1 + e, rcp, x r; one value at a time would be a chain of dependent
// instructions).
uint2 gelu_q8_tr(v8f h)
{
    v8f a, g;
    [unroll] for (uint v = 0; v < 8; ++v) a.v[v] = h.v[v] * h.v[v];
    [unroll] for (uint v1 = 0; v1 < 8; ++v1) a.v[v1] = mad(K_GA, a.v[v1], K_GB);
    [unroll] for (uint v2 = 0; v2 < 8; ++v2) a.v[v2] = h.v[v2] * a.v[v2];
    [unroll] for (uint v3 = 0; v3 < 8; ++v3) a.v[v3] = exp2(a.v[v3]);
    [unroll] for (uint v4 = 0; v4 < 8; ++v4) a.v[v4] = 1.0f + a.v[v4];
    [unroll] for (uint v5 = 0; v5 < 8; ++v5) a.v[v5] = rcp(a.v[v5]);
    [unroll] for (uint v6 = 0; v6 < 8; ++v6) g.v[v6] = h.v[v6] * a.v[v6];
    return wg_pack_q8_nc_ags(g);
}

uint2 gelu_q8(v8f h)
{
#if BN_GELU == 2
    return wg_gelu_q8_lds(h);
#elif BN_GELU == 1
    return bn_gelu_q8(h);
#else
    return gelu_q8_tr(h);
#endif
}

// One MLP unit of BN_TPU hidden tiles at LDS element T: for each tile j, h_j = W1_j m + b1_j and acc += W2_j q8(GELU(h_j)),
// tile by tile: W1 chain, GELU, then W2's fragments and multiplies. HIP's ISA interleaves tile j + 1's W1 chain with tile
// j's W2 multiplies (WG_V2_PIPE); here that schedule measured no faster (8 waves per SIMD hide the dependent-multiply
// latency) and it needs more registers (two hidden tiles, both fragment sets and the table entries live at once), which
// puts the driver's allocation at the 192-VGPR occupancy line. With `stage`, the next unit is staged in BN_TPU steps.
void mlp_unit(uint T, uint b1, uint2 mf[KC], inout v8f acc[KC], uint lb, uint tid, bool stage, uint ssrc, uint sdst,
              inout uint4 sr[4])
{
    [unroll] for (uint j = 0; j < BN_TPU; ++j) {
        if (stage) stage_step_r(j, BN_TPU, MLP_ROWS, ssrc, sdst, tid, sr);
        v8f h = w1(T, j, b1, mf, lb);
        uint2 hf = gelu_q8(h);
        uint2 f2[KC];
        [unroll] for (uint kf = 0; kf < KC; ++kf) f2[kf] = frag(T, j * 16u + KC + kf, lb);
        [unroll] for (uint m = 0; m < KC; ++m) acc[m] = wg_wmma_fp8(f2[m], hf, acc[m]);
    }
    if (stage) stage_step_r(BN_TPU, BN_TPU, MLP_ROWS, ssrc, sdst, tid, sr);
}

// ---- persistent scheduler (swin_v2.hip bneck6) --------------------------------------------------------------------------

uint weights(int k)
{
    return k == 0 ? a_w0 : k == 1 ? a_w1 : k == 2 ? a_w2 : k == 3 ? a_w3 : k == 4 ? a_w4 : a_w5;
}

// Current value of a sync word (an atomic, so that the spin loop cannot be hoisted).
uint sync_read(uint off)
{
    uint v;
    wg_mem_gc.InterlockedOr(off, 0u, v);
    return v;
}

WG_KERNEL(swin_bneck6_mb_nc)
{
    WG_SETUP();
    const uint tid = wg_lane_id(), lane = tid & 31u, wave = tid >> 5, t = lane & 15u, hh = lane >> 4;
    const int H = a_H, W = a_W;
    const Grid ge = grid_of(H, W, a_oxe, a_oye), go = grid_of(H, W, a_oxo, a_oyo);
    // merge items first in the queue (mx = 0: none, b0 reads x written before the launch, as bneck.hlsl; diagnostics)
    const int nm = a_mx != 0u ? (go.nwin + 8 * (int)MB_WPW - 1) / (8 * (int)MB_WPW) : 0;
    const int total = nm + 3 * (ge.items + go.items);
    const int NR = ge.nwy > go.nwy ? ge.nwy : go.nwy;
    const uint rows = a_sync + SYNC_HEAD * 4u;

#if BN_GELU == 2
    lds[GL_EL + tid] = wg_gl_entry(tid);   // 256 lanes = 256 entries; the barrier below publishes them
#endif
    if (tid == 0u) {
        lds[SLOT_EL].x = wg_atomic_add(a_sync, 0u, 1u);
#ifdef BNECK_DEBUG_STRESS
        wg_atomic_max(a_sync, 24u, wg_atomic_add(a_sync, 20u, 1u) + 1u);   // sync[6] = most workgroups ever live at once
#endif
    }
    wg_barrier_lds();
    int cur = WaveReadLaneFirst((int)lds[SLOT_EL].x);
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
    }
    // a trip count the driver's compiler cannot see (a_groups < 2^31): it would unroll the MLP loop completely (with the
    // heads: 57 KB of code and 240 VGPRs, 6 waves per SIMD instead of 8)
    const uint opaque0 = (uint)a_groups >> 31;
#ifdef BNECK_DEBUG_COUNT
    uint done = 0;   // debug builds (work tools only): items run by this workgroup
#endif
    [loop] while (cur < total) {
        int k = 0, local = cur - nm;
        [loop] while (k < 5 && local >= ((k & 1) ? go.items : ge.items)) {
            local -= (k & 1) ? go.items : ge.items;
            ++k;
        }
        Grid g = ge;
        if (k & 1) g = go;
        const uint src = k == 0 ? a_x : ((k & 1) ? a_tmp : a_out);
        const uint dstbuf = (k & 1) ? a_out : a_tmp;
        const int wf = local * 8, wl = (wf + 8 < g.nwin ? wf + 8 : g.nwin) - 1;
        const int r0 = (int)((uint)wf / (uint)g.nwx), r1 = (int)((uint)wl / (uint)g.nwx);   // wf, wl >= 0, nwx > 0
        const uint sec = weights(k);
        {   // this item's head-0 unit -> LDS buffer 0 (published by the barrier below)
            uint4 pre[3];
            [unroll] for (uint i = 0; i < 3; ++i) pre[i] = wg_ld128(sec, SEC_HEAD(0u) + (tid + i * 256u) * 16u);
            [unroll] for (uint i2 = 0; i2 < 3; ++i2) st_lds4(tid * 2u + i2 * 512u, pre[i2]);
        }
        // the next item, claimed now (one item ahead; HIP claims two ahead). Claiming two ahead puts twice as many
        // claimed-but-unstarted items in the queue window, and the items depending on them waited at the row counters:
        // 15-20 % of the launch at the start of blocks 2-5 (shader-clock profile); one ahead measured 4-5 % faster.
        // Claiming later (after the heads, or at the end of the item) measured 10-15 % slower.
        uint after = (uint)total;
        if (tid == 0u) after = wg_atomic_add(a_sync, 0u, 1u);
        if ((k > 0 || nm > 0) && tid == 0u) {   // (b0: the rows of the merge items)
            // rows of block k - 1 this item depends on (writers of its input rows, readers of the rows it overwrites)
            Grid p = go;
            if (k & 1) p = ge;
            const int a = 4 * r0 - g.oy, b = 4 * r1 - g.oy + 3;
            const int lo = a < 0 ? 0 : (b >= H && 2 * (H - 1) - b < a ? 2 * (H - 1) - b : a);
            const int mm = b > -a ? b : -a;
            const int hi = mm < H - 1 ? mm : H - 1;
            int q0 = (lo + p.oy) >> 2, q1 = (hi + p.oy) >> 2;
            if (lo <= 3) q0 = 0;
            if (hi >= H - 4 || q1 > p.nwy - 1) q1 = p.nwy - 1;
            const uint rc = rows + (uint)((k > 0 ? k - 1 : 6) * NR) * 4u;
            // Bounded wait (BNECK_SPIN_LIMIT polls per row, roughly 0.1 s). The schedule needs no co-residency, but it does
            // need every workgroup that holds a claimed item to keep running, and under a concurrent graphics load that
            // fails: a few workgroups of this launch stop being scheduled for a long time (stress harness, 100k launches
            // back to back next to another process's draws and dispatches: about 1 in 15-40k launches; a trace showed the
            // 18 highest-numbered workgroups frozen in the middle of their items while every other one spun on rows only
            // they could complete; presumably evicted by a preemption and not restored while the spinners hold the CUs).
            // An unbounded spin then never ends (the rare 20 s fence timeout). Here an item that has waited that long runs
            // on what is there: the spinners finish, the stalled workgroups come back and finish theirs, and the launch
            // completes (that frame's output is wrong; sync[15], never reset, counts the events). The longest wait seen in
            // a correct launch under that load was about 12k polls.
            uint polls = 0;
            [loop] for (int q = q0; q <= q1; ++q) {
                uint lim = 0;
                [loop] while (sync_read(rc + (uint)q * 4u) < (uint)p.nwx) {
                    ++polls;
                    if (++lim == BNECK_SPIN_LIMIT) {
                        wg_atomic_add(a_sync, 60u, 1u);
                        break;
                    }
                }
            }
#ifdef BNECK_DEBUG_STRESS
            wg_atomic_max(a_sync, 28u, polls);   // sync[7] = longest wait (polls), sync[8] = all polls
            wg_atomic_add(a_sync, 32u, polls);
#endif
        }
        wg_barrier();
        // the item's window (one per wave): gather the input (coherent loads: written by other workgroups)
        const int win = wf + WaveReadLaneFirst((int)wave);   // wave-uniform: the window math on SALU
        const bool valid = win <= wl;
        const int wn = valid ? win : wl;
        const int wy = (int)((uint)wn / (uint)g.nwx), wx = wn - wy * g.nwx;
        const int pr = 4 * wy - g.oy + (int)(t >> 2), pc = 4 * wx - g.ox + (int)(t & 3u);
        const bool real = valid && pr >= 0 && pr < H && pc >= 0 && pc < W;
        const int dst = real ? pr * W + pc : -1;
        const uint soff = (uint)(mirror(pr, H) * W + mirror(pc, W)) * 128u + hh * 64u;
        uint4 xr[4];
        [unroll] for (uint i = 0; i < 4; ++i) xr[i] = wg_mem_gc.Load4(src + soff + i * 16u);

        // ---- the block on this window (streamed_body): heads 0..3, MLP units 0..7; unit u of the 12 in buffer u & 1
        v8f acc[KC];
        uint2 f[KC];
        uint4 sr[4];
        norm1(xr, sec, hh, acc, f);
        // heads 0..3 (head hd in buffer hd & 1; constant LDS addresses), head 3 stages MLP unit 0
        [unroll] for (uint hd = 0; hd < 4u; ++hd) {
            attention((hd & 1u) * BUF_EL, sec, hd, f, acc, lb, t, hh, tid, sec + (hd < 3u ? SEC_HEAD(hd + 1u) : SEC_TILE(0u)),
                      ((hd + 1u) & 1u) * BUF_EL, sr);
            wg_barrier_lds();
        }
        norm2(acc, sec, hh, f);
        const uint b1 = sec + SEC_B1 + hh * 32u;
        // MLP units 0..7 (unit u in buffer u & 1), two per pass so that the LDS addresses are constants
        [loop] for (uint u = 0; u < NMLP - 2u + opaque0; u += 2u) {
            mlp_unit(0u, b1 + u * (64u * BN_TPU), f, acc, lb, tid, true, sec + SEC_TILE((u + 1u) * BN_TPU), BUF_EL, sr);
            wg_barrier_lds();
            mlp_unit(BUF_EL, b1 + (u + 1u) * (64u * BN_TPU), f, acc, lb, tid, true, sec + SEC_TILE((u + 2u) * BN_TPU), 0u, sr);
            wg_barrier_lds();
        }
        mlp_unit(0u, b1 + (NMLP - 2u) * (64u * BN_TPU), f, acc, lb, tid, true, sec + SEC_TILE((NMLP - 1u) * BN_TPU), BUF_EL, sr);
        wg_barrier_lds();
        mlp_unit(BUF_EL, b1 + (NMLP - 1u) * (64u * BN_TPU), f, acc, lb, tid, false, 0u, 0u, sr);
        // output q8(x2) (swin.hip wg_pack_q8: clamp with NaN kept) for the real window members, with a wave-uniform skip
        // of the clamp (as the Swin blocks' sw_in448): am = max |x| of the wave's values. Both paths hand a NaN unchanged
        // to the same conversion (0xff for every NaN), so they can only differ on a non-NaN |x| > 448. Such a value (an
        // infinity included) makes am > 448 whether the driver's max drops a NaN (am = max of the non-NaN |x|) or keeps it
        // (am = NaN, the test fails); the values are arithmetic results, so a NaN is quiet (no signalling NaN that an IEEE
        // max would turn into a quiet one mid-chain). The plain conversion runs only when the clamp is the identity.
        uint4 o[KC / 2];
        float am = 0.0f;
        [unroll] for (uint k0 = 0; k0 < KC; ++k0)
            am = max(max(max(am, abs(acc[k0].v[0])), max(abs(acc[k0].v[1]), abs(acc[k0].v[2]))),
                     max(max(abs(acc[k0].v[3]), abs(acc[k0].v[4])), max(max(abs(acc[k0].v[5]), abs(acc[k0].v[6])), abs(acc[k0].v[7]))));
        if (WaveActiveAllTrue(am <= 448.0f)) {
            [unroll] for (uint k2 = 0; k2 < KC; k2 += 2u) o[k2 >> 1] = uint4(wg_pack_q8_nc_ags(acc[k2]), wg_pack_q8_nc_ags(acc[k2 + 1u]));
        } else {
            [unroll] for (uint k2 = 0; k2 < KC; k2 += 2u) o[k2 >> 1] = uint4(pack_q8_out(acc[k2]), pack_q8_out(acc[k2 + 1u]));
        }
#ifdef BNECK_DEBUG_STRESS
        if (k == 5) {   // sync[9] += sum of the block-5 output dwords of this item
            uint cs = 0;
            if (dst >= 0) [unroll] for (uint i4 = 0; i4 < KC / 2; ++i4) cs += o[i4].x + o[i4].y + o[i4].z + o[i4].w;
            cs = WaveActiveSum(cs);
            if (lane == 0u) wg_atomic_add(a_sync, 36u, cs);
        }
#endif
        if (dst >= 0) {
            const uint doff = dstbuf + (uint)dst * 128u + hh * 64u;
            [unroll] for (uint i3 = 0; i3 < KC / 2; ++i3) wg_mem_gc.Store4(doff + i3 * 16u, o[i3]);
        }
        if (tid == 0u) lds[SLOT_EL + 1u].x = after;
        wg_barrier();   // every wave's stores are complete (device scope) before the counters move
        if (tid == 0u) {
            const uint rc = rows + (uint)(k * NR) * 4u;
            [loop] for (int rr = r0; rr <= r1; ++rr) {
                const int s0 = rr * g.nwx > wf ? rr * g.nwx : wf;
                const int s1 = (rr + 1) * g.nwx - 1 < wl ? (rr + 1) * g.nwx - 1 : wl;
                wg_atomic_add(rc, (uint)rr * 4u, (uint)(s1 - s0 + 1));
            }
        }
        cur = WaveReadLaneFirst((int)lds[SLOT_EL + 1u].x);
#ifdef BNECK_DEBUG_COUNT
        ++done;
#endif
    }
    if (tid == 0u) {
        wg_fence_device();
#ifdef BNECK_DEBUG_COUNT
        wg_atomic_add(a_sync, 8u, done);
#endif
#ifdef BNECK_DEBUG_STRESS
        wg_atomic_add(a_sync, 20u, 0xffffffffu);
#endif
        if (wg_atomic_add(a_sync, 4u, 1u) == (uint)a_groups - 1u) {  // last workgroup: every other one has finished
#ifdef BNECK_DEBUG_COUNT
            // sync[2] = items run in this launch (reset), sync[3] += 1 when that is not every item (never reset)
            if (wg_atomic_exch(a_sync, 8u, 0u) != (uint)total) wg_atomic_add(a_sync, 12u, 1u);
#endif
#ifdef BNECK_DEBUG_STRESS
            // checksum of this launch's output against the first launch's (sync[10]); mismatches -> sync[11]; launches -> sync[12]
            const uint cs = wg_atomic_exch(a_sync, 36u, 0u), r = wg_atomic_cas(a_sync, 40u, 0u, cs);
            if (r != 0u && r != cs) wg_atomic_add(a_sync, 44u, 1u);
            wg_atomic_add(a_sync, 48u, 1u);
#endif
            wg_st32_gc(a_sync, 0u, 0u);
            for (int j = 0; j < 7 * NR; ++j)
                if (!MB_XO || j < 5 * NR || j >= 6 * NR) wg_st32_gc(rows, (uint)j * 4u, 0u);   // xo: not block 5's
            wg_st32_gc(a_sync, 4u, 0u);
        }
    }
}
