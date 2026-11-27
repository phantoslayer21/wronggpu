// SPDX-License-Identifier: GPL-3.0-only
// Model L's HYBRID form on the fast kernels (runtime WG_L_FORM=hybrid, native D3D12): the block stages' parameters in
// the values the hybrid kernels read. The runtime packs them through the unchanged M-form packers
// (pack_stage_blob, l::pack_model), under the same blob names. This is a C++ port of
// clean/l/runner/l_hyb_stage_pack.py hyb_stage_values (reference: clean/l/runner/l_hyb_pack.py hyb_block_values),
// byte for byte; test: clean/l/runtime/tests/test_l_runtime_hyb_pack.py.
//   gamma1, gamma2           -> RN16(float32(gamma) * sqrtf(C)), C = the gamma's length (oracle.gamma_q7)
//   bp, b1, b2, B, bm, be    -> RN16(value) (FP16 values, stored as float32: exact)
//   Wqk, Wv, Wp, W1, W2, Wm, We -> q8(value) (FP8 E4M3FN values, saturating at +-448)
// Every other tensor keeps its value. One rounding per operation (a product, then RN16; no fused forms).
#pragma once

#include <cmath>
#include <string>

#include "../../runtime/fpconv.h"
#include "../../runtime/model.h"

namespace wgrt {
namespace l {

// The block stages whose tensors take the hybrid values (enc0 / dec0 keep theirs).
inline bool hyb_block_stage(const std::string& s) {
  static const char* const kS[] = {"enc0_block", "dec0_block", "enc1", "enc2", "b0",   "b1",
                                   "b2",         "b3",         "b4",   "b5",   "dec2", "dec1"};
  for (const char* x : kS)
    if (s == x) return true;
  return false;
}

inline StageTensors hyb_stage_values(const StageTensors& t) {
  StageTensors o = t;
  for (const char* n : {"gamma1", "gamma2"}) {
    const auto it = o.find(n);
    if (it == o.end()) continue;
    const float r = std::sqrt((float)it->second.f.size());   // sqrtf(C): correctly rounded
    for (float& v : it->second.f) {
      const float g = v * r;                                 // float32 product, one rounding
      v = rn16(g);
    }
  }
  for (const char* n : {"bp", "b1", "b2", "B", "bm", "be"}) {
    const auto it = o.find(n);
    if (it == o.end()) continue;
    for (float& v : it->second.f) v = rn16(v);
  }
  for (const char* n : {"Wqk", "Wv", "Wp", "W1", "W2", "Wm", "We"}) {
    const auto it = o.find(n);
    if (it == o.end()) continue;
    for (float& v : it->second.f) v = (float)q8((double)v);
  }
  return o;
}

}  // namespace l
}  // namespace wgrt
