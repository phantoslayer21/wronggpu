// SPDX-License-Identifier: MIT
// Motion vectors at output resolution (spec 03 draft 4 "Motion at output resolution", spec 06 section 3): the raw motion
// texel both the input encoder (encoder.hlsl, WG_ENC_MO) and the reconstruction (post.hlsl, WG_POST_MO) read. HLSL
// counterpart of clean/kernels/wg_motion_hi.h. Written for WrongGPU.
//
// Buffer form (the arena buffer of wg_motion_hi.h: rows of `pitch` bytes, a texel being two FP16 (wide = 0) or two f32
// (wide = 1), at least 4 spare bytes after the last texel): the two 32-bit words at the texel's byte offset, selected by
// the (uniform) format as wg_mo_value.
// Direct form (WG_DIRECT_TEXTURES: the game's motion texture through native_direct.hlsli's t2 view): the interop's
// motionhi pass (interop/convert.cpp) fills the HIP path's buffer with Load(min(tx, w - 1), min(ty, h - 1)).xy of that
// view (the texture's texel (0, 0) is the origin; a texture smaller than M repeats its last row and column) through a
// typed RG16F store. Here the same texel of the same view; an FP16 view's values are FP16 values (the store keeps them),
// a wider view's values go through f32tof16 (round toward zero, as the typed FP16 stores of this driver) and back.
// b1 word 15 (NativeDirectParams.padding[4], interop/executor_d3d12.cpp): bit 0 = the motion view's format is FP16
// (RG16F, RGBA16F); bits 1..15 = w - 1, bits 16..30 = h - 1, the clamp size (w, h) of the motionhi pass (the WgTexture's
// size as the game gives it, not the resource's: the HIP path's Load(min(tx, cw - 1), min(ty, ch - 1))).
#pragma once

#if defined(WG_DIRECT_TEXTURES)
#define nd_motion_f16 (nd_pad5 & 1u)

// The motion texture's clamp size (w, h) (uniform: b1 word 15)
uint2 wg_mo_dims()
{
    return uint2(((nd_pad5 >> 1) & 0x7fffu) + 1u, (nd_pad5 >> 16) + 1u);
}

// The raw texel (tx, ty) of the region M (tx < Mw, ty < Mh), as the HIP path's buffer holds it; dims = wg_mo_dims().
float2 wg_mo_texel_direct(uint tx, uint ty, uint2 dims)
{
    float2 v = native_direct_motion_tex.Load(int3((int)min(tx, dims.x - 1u), (int)min(ty, dims.y - 1u), 0)).xy;
    [branch] if (nd_motion_f16 == 0u) v = f16tof32(f32tof16(v));
    return v;
}
#endif

// The two words of the raw texel at byte offset off of the arena buffer at base (wg_motion_hi.h wg_mo_load).
uint2 wg_mo_load(uint base, uint off) { return uint2(wg_mem.Load(base + off), wg_mem.Load(base + off + 4u)); }

// Their values (wg_motion_hi.h wg_mo_value): FP16 halves of word 0, or the two f32 words.
float2 wg_mo_value(uint2 w, int wide)
{
    const float hx = f16tof32(w.x & 0xffffu), hy = f16tof32(w.x >> 16);
    return float2(wide != 0 ? asfloat(w.x) : hx, wide != 0 ? asfloat(w.y) : hy);
}
