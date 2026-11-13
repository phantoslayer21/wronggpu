// SPDX-License-Identifier: MIT
// clean/rr-f-s12-dflow: the body as a dataflow (RR_DFLOW 1, kernels staged.*; opt-in WG_RR_DFLOW=1). The runtime drops
// the UAV barriers after the body stages in its drop set; instead every window of such a stage counts itself done in
// cnt[stage][window row] after its last store, and every window of a later stage that reads such a stage's tensors
// waits until all producer window rows its reads touch are complete (count = the producer's nwx). Same dispatch order,
// same arithmetic, same bytes read: only the time at which a window starts changes (bit-identical by construction; the
// footprints below are exactly the rows the kernel's loads touch, so a window never reads a row before its last store).
//
// Coherence: without the barriers nothing invalidates the per-CU caches, so the producer tensors are read and written
// through wg_mem_gc (globallycoherent: device scope); weights keep the normal view. Producer: every lane's stores, then
// the group barrier (all memory, device scope), then one atomic add (release order as in dx12/kernels/bneck.hlsl).
// Consumer: wave 0 polls (atomic reads: never hoisted), then the group barrier, then the coherent loads.
//
// No deadlock: every producer belongs to an earlier dispatch on the same queue, whose groups have all launched before
// any group of a later dispatch. The spin is bounded anyway (RR_DF_SPIN polls per window): a window that waited that
// long runs on what is there and counts the event in the error word (cnt[RR_DF_ERR]); that frame's output is wrong.
//
// Kernel arguments after the stage's own (rr_block_wide.hlsl): dfCnt = arena offset of the counter array, dfSelf = this
// stage's index, dfQ0..2 = the producer slots: stage (bits 0-7, 0xff none) | read kind (8-10) | negative-control bit (11) | producer sy (12-15) |
// producer nwx (16-31). Read kinds (the producer window row of a token row R at the producer's level: (R + sy) >> 3):
//   0 same level      R = refl(Y)                      (y2 / hnew of a same-level stage: skip reads, Dk-R's fusion y2)
//   1 merged          R = 2 refl(Y) at the finer level (Ek's merged + hmerged of E(k-1))
//   2 coarse 4 x 4    R = refl(cy0 + 0..3) one level coarser (Dk-F's up-projection input)
//   3 bilinear        R = refl(((Y - 1) >> 1) + 0..1) one level coarser (Dk-R's coarse hidden state)
#ifndef RR_DF_K1E
#define RR_DF_K1E 0
#endif
#ifndef RR_DF_ROWS
#define RR_DF_ROWS 512u          // counter rows per stage (runtime kDfRows)
#endif
#define RR_DF_ERR (15u * RR_DF_ROWS)
#ifndef RR_DF_SPIN
#define RR_DF_SPIN 262144u
#endif

#define RR_PLD64(b, o) wg_mem_gc.Load2((b) + (o))
#define RR_PST64(b, o, v) wg_mem_gc.Store2((b) + (o), (v))

#ifndef RR_DF_POLL
#define RR_DF_POLL 0
#endif
#if RR_DF_POLL
// clean/rr-f-a14-frame E5 b (build-time WG_RR_DFPOLL=1, A/B): plain globallycoherent loads instead of an atomic RMW per poll
uint rr_df_poll(uint a) { return wg_mem_gc.Load(a); }
#else
uint rr_df_poll(uint a) { uint v; wg_mem_gc.InterlockedOr(a, 0u, v); return v; }
#endif

uint rr_df_row(uint q, int wj, int sy, int Ny, int My, bool r1, uint i, uint e)
{
    const uint ty = (q >> 8) & 0x7u;
    const int Y = 8 * wj - sy + (int)i;
    int R;
    if (ty == 0u) R = rr_refl(Y, Ny, r1);
#if RR_DF_K1E
    else if (ty == 1u) R = 2 * rr_refl(Y, Ny, r1) + (int)e;   // clean/rr-f-a14-frame: both finer rows (children 2R, 2R + 1)
#else
    else if (ty == 1u) R = 2 * rr_refl(Y, Ny, r1);
#endif
    else if (ty == 2u) R = rr_refl(((8 * wj - sy) >> 1) + (int)(i & 3u), My, r1);
    else R = rr_refl(((Y - 1) >> 1) + (int)e, My, r1);
    return (uint)((R + (int)((q >> 12) & 0xfu)) >> 3);
}

// every lane of the group calls it (contains the group barrier)
void rr_df_wait(uint w, int wj, int sy, int Ny, int My, bool r1)
{
    if (w == 0u) {
        const uint L = WaveGetLaneIndex(), qi = L >> 3, i = L & 7u;
        const uint q = qi == 0u ? dfQ0 : (qi == 1u ? dfQ1 : (qi == 2u ? dfQ2 : 0xffu));
        const uint ps = q & 0xffu, need = q >> 16;
        bool ok = ps == 0xffu;
        uint a0 = 0u, a1 = 0u;
        if (!ok) {
            uint w0 = rr_df_row(q, wj, sy, Ny, My, r1, i, 0u), w1 = rr_df_row(q, wj, sy, Ny, My, r1, i, 1u);
            // bit 11 (negative control only, WG_RR_DFLOW_NEG; output wrong): only the producer row of the window's first
            // token row is checked (a 2-3 row footprint cut to 1 row)
            if ((q >> 11) & 1u) w0 = w1 = rr_df_row(q, wj, sy, Ny, My, r1, 0u, 0u);
            a0 = dfCnt + (ps * RR_DF_ROWS + w0) * 4u;
            a1 = dfCnt + (ps * RR_DF_ROWS + w1) * 4u;
        }
        uint lim = 0u;
        [loop] while (!WaveActiveAllTrue(ok)) {
            if (!ok) ok = rr_df_poll(a0) >= need && rr_df_poll(a1) >= need;
            if (++lim == RR_DF_SPIN) {
                if (L == 0u) { uint o; wg_mem_gc.InterlockedAdd(dfCnt + RR_DF_ERR * 4u, 1u, o); }
                break;
            }
        }
    }
    AllMemoryBarrierWithGroupSync();
}

// after the window's last store (every lane)
void rr_df_signal(int wj)
{
    AllMemoryBarrierWithGroupSync();
    if (wg_lane_id() == 0u) {
        wg_fence_device();
        uint o;
        wg_mem_gc.InterlockedAdd(dfCnt + (dfSelf * RR_DF_ROWS + (uint)wj) * 4u, 1u, o);
    }
}
