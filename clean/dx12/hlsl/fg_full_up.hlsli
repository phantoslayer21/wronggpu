// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): the output pyramid's 9/3/3/1 upsample on four already-loaded taps, for the kernels whose taps
// do not come from the arena (fg_out_push.hlsl FGU_TAIL: LDS; FGU_L1: texels pushed on the fly). It is the second half of
// fgf_up (fg_full.hlsli), line for line: clean/fg/tests/test_fg_turbo_v2.py compares the two texts. Include after
// fg_full.hlsli.
#pragma once

// fgf_up's taps: the element indices (iP, iX, iY, iD) of the coarse level (wc x hc) around fine element (qx, qy)
void fgf_up_idx(int wc, int hc, int qx, int qy, out int iP, out int iX, out int iY, out int iD)
{
    const int px = qx >> 1, py = qy >> 1;
    const int nx = clamp(px + ((qx & 1) != 0 ? 1 : -1), 0, wc - 1);
    const int ny = clamp(py + ((qy & 1) != 0 ? 1 : -1), 0, hc - 1);
    iP = py * wc + px;
    iX = py * wc + nx;
    iY = ny * wc + px;
    iD = ny * wc + nx;
}

// fgf_up's arithmetic from the four taps' float values (eP: own column and row, eX: neighbour column, eY: neighbour row,
// eD: both)
float4 fgf_up_taps(float4 eP, float4 eX, float4 eY, float4 eD)
{
    const float z = fgf_zero();
    const float ws = mad(eD.w, 0.0625f, mad(eY.w, 0.1875f, mad(eX.w, 0.1875f, mad(eP.w, 0.5625f, z))));
    const float3 vs = mad(fgf_opq3(eD.xyz * eD.w), 0.0625f, mad(fgf_opq3(eY.xyz * eY.w), 0.1875f,
                      mad(fgf_opq3(eX.xyz * eX.w), 0.1875f, mad(fgf_opq3(eP.xyz * eP.w), 0.5625f, z))));
    const float den = min(max(ws, 1e-12f), 1.0f);
    return float4(wg_div(vs.x, den), wg_div(vs.y, den), wg_div(vs.z, den), ws);
}
