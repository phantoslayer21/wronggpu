// SPDX-License-Identifier: MIT
// The HYBRID form's bottleneck in one launch (WG_L_FORM = hybrid, native D3D12): b0 .. b5 as a persistent queue of
// items, bneck.hlsl's schedule (items of 8 windows in block order; one claimed ahead; per-row counters of every block;
// an item waits for the rows of the previous block it reads and overwrites; the bounded wait; the last workgroup resets
// the sync words), with swin_block.hlsl's hybrid C128 block body as the item (SW_MEGA: sw_item(), the same instructions
// as swin_block_c128_hyb on the same values). b5's output equals six swin_block_c128_hyb launches byte for byte; b0 ..
// b4's outputs are not kept (ping-pong in out / tmp, as swin_bneck6_nc).
// Arguments (bneck.hlsl's): x (b0's input, enc2's merged map), out (b5's output), tmp (scratch map), sync (the zeroed
// sync words: [0] queue, [1] finished workgroups, [12] the frame's abort word, [13] zero (the 8-byte frame guard is
// [12 .. 13]), [15] wait timeouts (never reset), [16 + k NR + r] row counters), the six blocks' hybrid weight sections,
// H, W, the even and odd blocks' window offsets, the workgroup count.
// A wait that times out (BNECK_SPIN_LIMIT polls on one row) REFUSES the frame: it adds 1 to [12] and [15], and every
// wait of the launch then stops at once (no item waits on rows that may never come); the launch completes, its maps
// are not valid. The runtime reads [12 .. 13] right after this launch into the frame guard (Device::guard_begin):
// every later launch and copy of the frame, and of every frame after it until the runtime has seen the count [15]
// move, is skipped on the GPU (no output, no history); the runtime then refuses one frame, starts the next one over
// (reset) and clears [12]. The kernel never clears [12].
//   swin_bneck6_hyb           the twin (Model L hybrid route)
//   swin_bneck6_hyb_scanmut   the fused-square scan's mutant (probe=1, outside the manifest: the scan must FAIL on it)
//   swin_bneck6_hyb_mut       check mutant (probe=1): order mutant 4 (W2 chain end one float32 ulp off); the byte check
//                             against the six launches must FAIL on it
//   swin_bneck6_hyb_tmo       timeout control (probe=1): BNECK_SPIN_LIMIT 1 (a wait gives up at its first poll that
//                             finds a producer not done): the check requires [12] and [15] nonzero after it (and zero
//                             after the twin)
// @wg-kernel swin_bneck6_hyb co=- block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u defines=SW_C=128,SW_HEADS=4,SW_MLP=512,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_MEGA=1,SW_ENTRY_OVERRIDE=swin_bneck6_hyb
// @wg-kernel swin_bneck6_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u defines=SW_C=128,SW_HEADS=4,SW_MLP=512,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_MEGA=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=swin_bneck6_hyb_scanmut
// @wg-kernel swin_bneck6_hyb_mut co=- probe=1 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u defines=SW_C=128,SW_HEADS=4,SW_MLP=512,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_MEGA=1,SW_HYB_MUT_ORDER=4,SW_ENTRY_OVERRIDE=swin_bneck6_hyb_mut
// @wg-kernel swin_bneck6_hyb_tmo co=- probe=1 block=256 wave=32 args=p,p,p,p,p,p,p,p,p,p,u,u,u,u,u,u,u defines=SW_C=128,SW_HEADS=4,SW_MLP=512,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HYB=1,SW_MEGA=1,BNECK_SPIN_LIMIT=1u,SW_ENTRY_OVERRIDE=swin_bneck6_hyb_tmo
#include "swin_block.hlsl"

#define BQ_SYNC_HEAD 16
#define BQ_ABORT 48u      // byte offset of sync word 12: the frame's abort word
#define BQ_TIMEOUTS 60u   // byte offset of sync word 15: wait timeouts (never reset)
#ifndef BNECK_SPIN_LIMIT
#define BNECK_SPIN_LIMIT (1u << 18)   // polls per row, roughly 0.1 s (bneck.hlsl's bounded wait)
#endif

// The claim slot: LDS element 0 of the block body's staging buffers (sw_lds_q, every byte in use by the C128 body),
// written and read only while the body is not running (between barriers), and read before the next item's body starts
// (the barrier ahead of sw_item).

struct BqGrid { int oy, ox, nwx, nwy, nwin, items; };
BqGrid bq_grid(int h, int wd, int gx, int gy)
{
    BqGrid g;
    g.ox = gx;
    g.oy = gy;
    g.nwx = (wd + gx + 3) >> 2;
    g.nwy = (h + gy + 3) >> 2;
    g.nwin = g.nwx * g.nwy;
    g.items = (g.nwin + 7) / 8;
    return g;
}

uint bq_weights(int k)
{
    return k == 0 ? a_w0 : k == 1 ? a_w1 : k == 2 ? a_w2 : k == 3 ? a_w3 : k == 4 ? a_w4 : a_w5;
}

// Current value of a sync word (an atomic, so that the spin loop cannot be hoisted).
uint bq_sync_read(uint off)
{
    uint v;
    wg_mem_gc.InterlockedOr(off, 0u, v);
    return v;
}

WG_KERNEL(SW_ENTRY)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const BqGrid ge = bq_grid(H, W, a_oxe, a_oye), go = bq_grid(H, W, a_oxo, a_oyo);
    const int total = 3 * (ge.items + go.items);
    const int NR = ge.nwy > go.nwy ? ge.nwy : go.nwy;
    const uint rows = a_sync + BQ_SYNC_HEAD * 4u;

    if (tid == 0u) sw_lds_q[0] = (uint64_t)wg_atomic_add(a_sync, 0u, 1u);
    wg_barrier_lds();
    int cur = WaveReadLaneFirst((int)(uint)sw_lds_q[0]);
    [loop] while (cur < total) {
        int k = 0, local = cur;
        [loop] while (k < 5 && local >= ((k & 1) ? go.items : ge.items)) {
            local -= (k & 1) ? go.items : ge.items;
            ++k;
        }
        BqGrid g = ge;
        if (k & 1) g = go;
        // the item's block (sw_item's statics): even blocks write tmp, odd blocks out (b5 -> out)
        // every per-item value through WaveReadLaneFirst: the driver formed the selects (v_cndmask) and the row
        // division in VGPRs and kept them across the body (200 VGPRs, against the block kernel's 192)
        x = WaveReadLaneFirst(k == 0 ? a_x : ((k & 1) ? a_tmp : a_out));
        out_p = WaveReadLaneFirst((k & 1) ? a_out : a_tmp);
        w = WaveReadLaneFirst(bq_weights(k));
        ox = WaveReadLaneFirst(g.ox);
        oy = WaveReadLaneFirst(g.oy);
        wg_bid_ = WaveReadLaneFirst((uint)local);
        const int wf = local * 8, wl = WaveReadLaneFirst((wf + 8 < g.nwin ? wf + 8 : g.nwin) - 1);
        const int r0 = WaveReadLaneFirst((int)((uint)wf / (uint)g.nwx)),
                  r1 = WaveReadLaneFirst((int)((uint)wl / (uint)g.nwx));   // wf, wl >= 0, nwx > 0
        // the next item, claimed now (one ahead, as bneck.hlsl)
        uint after = (uint)total;
        if (tid == 0u) after = wg_atomic_add(a_sync, 0u, 1u);
        after = WaveReadLaneFirst(after);   // wave 0's lane 0 (only it stores it): a scalar across the body, not a VGPR
        if (k > 0 && tid == 0u) {
            // rows of block k - 1 this item depends on (writers of its input rows, readers of the rows it overwrites):
            // bneck.hlsl's range, mirror included
            BqGrid p = go;
            if (k & 1) p = ge;
            const int a = 4 * r0 - g.oy, b = 4 * r1 - g.oy + 3;
            const int lo = a < 0 ? 0 : (b >= H && 2 * (H - 1) - b < a ? 2 * (H - 1) - b : a);
            const int mm = b > -a ? b : -a;
            const int hi = mm < H - 1 ? mm : H - 1;
            int q0 = (lo + p.oy) >> 2, q1 = (hi + p.oy) >> 2;
            if (lo <= 3) q0 = 0;
            if (hi >= H - 4 || q1 > p.nwy - 1) q1 = p.nwy - 1;
            const uint rc = rows + (uint)((k - 1) * NR) * 4u;
            // bounded wait: a launch whose workgroups stop being scheduled still completes. A wait of
            // BNECK_SPIN_LIMIT polls on one row refuses the frame (the abort word; the frame guard skips the rest of
            // it); once the abort word is set (by this or another workgroup), no item waits any more (checked every
            // 64th poll of a wait): the remaining items run on what is there, into maps nothing reads.
            bool stop = false;
            [loop] for (int q = q0; q <= q1 && !stop; ++q) {
                uint lim = 0;
                [loop] while (bq_sync_read(rc + (uint)q * 4u) < (uint)p.nwx) {
                    ++lim;
                    if (lim == BNECK_SPIN_LIMIT) {
                        wg_atomic_add(a_sync, BQ_ABORT, 1u);
                        wg_atomic_add(a_sync, BQ_TIMEOUTS, 1u);
                        stop = true;
                        break;
                    }
                    if ((lim & 63u) == 0u && bq_sync_read(a_sync + BQ_ABORT) != 0u) {
                        stop = true;
                        break;
                    }
                }
            }
        }
        wg_barrier();
        sw_item();
        wg_barrier();   // every wave's stores are complete (device scope) before the counters move; the LDS is free
        if (tid == 0u) {
            sw_lds_q[0] = (uint64_t)after;
            const uint rc = rows + (uint)(k * NR) * 4u;
            [loop] for (int rr = r0; rr <= r1; ++rr) {
                const int s0 = rr * g.nwx > wf ? rr * g.nwx : wf;
                const int s1 = (rr + 1) * g.nwx - 1 < wl ? (rr + 1) * g.nwx - 1 : wl;
                wg_atomic_add(rc, (uint)rr * 4u, (uint)(s1 - s0 + 1));
            }
        }
        wg_barrier_lds();
        cur = WaveReadLaneFirst((int)(uint)sw_lds_q[0]);
    }
    if (tid == 0u) {
        wg_fence_device();
        if (wg_atomic_add(a_sync, 4u, 1u) == (uint)a_groups - 1u) {   // last workgroup: every other one has finished
            wg_st32_gc(a_sync, 0u, 0u);
            for (int j = 0; j < 6 * NR; ++j) wg_st32_gc(rows, (uint)j * 4u, 0u);
            wg_st32_gc(a_sync, 4u, 0u);
        }
    }
}
