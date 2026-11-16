// SPDX-License-Identifier: MIT
// Probe kernels of the native RDNA 3 K = 16 operands (hlsl/wg_gfx11.hlsli: wg_wmma_f16_k16n, wg_ldA16, wg_ldB16, wg_b16,
// wg_rows_to_gfx12). WrongGPU's own code. Test-only (this folder is never built into a kernel set; the declarations
// below are read by the emulation, clean/dx12/emu/hlsl_emu.py, and by test_gfx11_native.py, which compiles them with
// DXC and translates them with dxil-spirv). One 32-lane workgroup; `n` cases, case e:
//   pa + 512 e: a 16 x 16 F16 tile in the gfx12 fragment order (slot l = 16 h + r: row r, K 8 h .. 8 h + 7)
//   pb + 512 e: the same for B (slot 16 h + c: column c, K 8 h .. 8 h + 7)
//   pc / po + 1024 e: an FP32 accumulator in the gfx12 layout, lane-major (lane l, element i: row 8 h + i, column l % 16)
// The channel entries (k16n, k16_chain, k16_xchg) take the gfx11 instruction's own operands (16 halves per lane,
// replicated: pa / pb + 1024 e, lane l: K 0 .. 7 at 32 l, K 8 .. 15 at 32 l + 16) and give its own accumulator layout.
//
// @wg-kernel nat_ldA co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel nat_b16 co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel nat_vt co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel nat_vt_nofix co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel k16n co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel k16_chain co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel k16_xchg co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
// @wg-kernel k64n co=- block=32 wave=32 args=p,p,p,p,u gfx=11 probe=1
#include "wg_gfx11.hlsli"
WG_ARGS_BEGIN
    uint pa, pb, pc, po, n;
WG_ARGS_END

v8f load_acc(uint base, uint e, uint l)
{
    v8f c;
    [unroll] for (uint i = 0; i < 8u; ++i) c.v[i] = wg_ldf(base, ((e * 32u + l) * 8u + i) * 4u);
    return c;
}
void store_acc(uint base, uint e, uint l, v8f d)
{
    [unroll] for (uint i = 0; i < 8u; ++i) wg_stf(base, ((e * 32u + l) * 8u + i) * 4u, d.v[i]);
}

// D = A B + C with A row-mapped from its tile and B's column from its tile: D in the gfx12 layout.
WG_KERNEL(nat_ldA)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_mma16(wg_ldA16(pa, 256u * e), wg_ldB16(pb, 256u * e), load_acc(pc, e, l)));
}
// The same with B made in registers from the lane's own gfx12 fragment (one exchange).
WG_KERNEL(nat_b16)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e) {
        const uint4 own = wg_ld128(pb, (256u * e + 8u * l) * 2u);
        store_acc(po, e, l, wg_mma16(wg_ldA16(pa, 256u * e), wg_b16(own), load_acc(pc, e, l)));
    }
}
// v^T's form: A = the B form of the lane's own fragment of A's tile (rows = A's rows in the instruction's order, no row
// map), B from its tile, C = 0; the FP16 result put into the gfx12 fragment order (slot 16 h + c: rows 8 h .. 8 h + 7
// of column c) by wg_rows_to_gfx12. nat_vt_nofix stores it without that step (the negative control).
WG_KERNEL(nat_vt)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e) {
        const uint4 own = wg_ld128(pa, (256u * e + 8u * l) * 2u);
        const v8f d = wg_mma16(wg_b16(own), wg_ldB16(pb, 256u * e), wg_splat(0.0f));
        wg_st128(po, (256u * e + 8u * l) * 2u, wg_rows_to_gfx12(wg_to_f16_ags(d)));
    }
}
WG_KERNEL(nat_vt_nofix)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e) {
        const uint4 own = wg_ld128(pa, (256u * e + 8u * l) * 2u);
        const v8f d = wg_mma16(wg_b16(own), wg_ldB16(pb, 256u * e), wg_splat(0.0f));
        wg_st128(po, (256u * e + 8u * l) * 2u, wg_to_f16_ags(d));
    }
}

// ---- the K = 16 channel itself (instruction operands as given) --------------------------------------------------------
uint4 load_half(uint base, uint e, uint l, uint h) { return wg_ld128(base, (e * 32u + l) * 32u + 16u * h); }
// One K = 16 step in one instruction.
WG_KERNEL(k16n)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_wmma_f16_k16n(load_half(pa, e, l, 0u), load_half(pb, e, l, 0u), load_half(pa, e, l, 1u),
                                             load_half(pb, e, l, 1u), load_acc(pc, e, l)));
}
// The same step as two chained K = 8 instructions (wg_wmma_f16_k16).
WG_KERNEL(k16_chain)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_wmma_f16_k16(load_half(pa, e, l, 0u), load_half(pb, e, l, 0u), load_half(pa, e, l, 1u),
                                            load_half(pb, e, l, 1u), load_acc(pc, e, l)));
}
// HIP's gfx12-shaped wg_wmma_f16 (the per-call exchange); operands: the K 0 .. 7 halves of each lane.
WG_KERNEL(k16_xchg)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    [loop] for (uint e = 0; e < n; ++e)
        store_acc(po, e, l, wg_wmma_f16(load_half(pa, e, l, 0u), load_half(pb, e, l, 0u), load_acc(pc, e, l)));
}
// A K = 64 product: four native steps on one accumulator (cases 0 .. 3 of pa / pb).
WG_KERNEL(k64n)
{
    WG_SETUP();
    const uint l = wg_lane_id();
    v8f c = load_acc(pc, 0u, l);
    [unroll] for (uint s = 0; s < 4u; ++s)
        c = wg_wmma_f16_k16n(load_half(pa, s, l, 0u), load_half(pb, s, l, 0u), load_half(pa, s, l, 1u),
                             load_half(pb, s, l, 1u), c);
    store_acc(po, 0u, l, c);
}
