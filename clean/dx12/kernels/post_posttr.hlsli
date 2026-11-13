// SPDX-License-Identifier: MIT
// Direct-form (POST_OUT == 3) helpers of post.hlsl around its transcendentals (lev3 track "posttr"): the same values with
// fewer vector instructions. Include after post.hlsl's enc_e / clampf.
#pragma once

// Dec's "some value below kDecPolyMax" flag of six clamped values: one compare of their minimum instead of six compares
// and their mask chain (the same flag: a value below the bound makes the minimum below it, and the clamps leave no NaN)
bool pt_any_small6(float q[6])
{
    return min(min(min(q[0], q[1]), min(q[2], q[3])), min(q[4], q[5])) < kDecPolyMax;
}

// "some |Hl| is infinite" of the three channels: one compare of the largest magnitude instead of three compares and
// their mask chain (the same flag: the maximum of the magnitudes is infinite exactly when one of them is, and a NaN
// magnitude is neither infinite nor, in the maximum, kept over an infinite one)
bool pt_any_inf3(float a, float b, float c)
{
    return max(max(abs(a), abs(b)), abs(c)) == asfloat(0x7f800000u);
}
