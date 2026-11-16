// SPDX-License-Identifier: MIT
// Model M on the FP16-form (q7) kernels of Model L's shared stages (WG_M_FORM=q7, runtime.cpp): the blobs of Model M's
// ten shared stages for l_q7.co / l_q7w.co, packed from Model M's own tensors with the q7 packers of Model L
// (clean/l/runtime/l_q7_pack.h, byte for byte clean/l/runner/l_pack.py pack_q7_block / pack_q7_linear / q7_tables).
// Model M's stage tensors carry the same names and shapes as Model L's shared stages (spec 08 section 7.2): the block
// tensors gamma1, gamma2, bp, b1, b2, B, Wqk, Wv, Wp, W1, W2, the merges Wm / bm (enc1, enc2) and the expansions
// We / be (dec2, dec1). Blob names as l::pack_q7_model's: "<stage>.q7", "<stage>.q7lin", "q7.tab".
#pragma once

#include <map>
#include <string>
#include <vector>

#include "../l/runtime/l_q7_pack.h"
#include "model.h"

namespace wgrt {
namespace m_q7 {

inline const Tensor* tensor(const StageTensors& t, const char* stage, const char* name, size_t n, std::string* err) {
  const auto it = t.find(name);
  if (it == t.end()) return *err = std::string("Model M on q7: stage ") + stage + " has no tensor " + name, nullptr;
  if (it->second.f.size() != n)
    return *err = std::string("Model M on q7: ") + stage + "." + name + " has " + std::to_string(it->second.f.size()) +
                  " values, the q7 block needs " + std::to_string(n),
           nullptr;
  return &it->second;
}

// The ten shared stages' q7 blobs and the tables. False with *err when a stage or tensor is missing or mis-sized.
inline bool pack(const std::map<std::string, StageTensors>& model, std::map<std::string, std::vector<uint8_t>>* out,
                 std::string* err) {
  struct S {
    const char* stage;
    int C, heads, lin_n, lin_k;  // lin: merge (n = C_out, K = 4 C) or expansion (n = 4 C, K = C_low); 0 = none
    const char *lw, *lb;
  };
  static const S kS[] = {{"enc1", 64, 2, 96, 256, "Wm", "bm"},   {"enc2", 96, 4, 128, 384, "Wm", "bm"},
                         {"b0", 128, 4, 0, 0, nullptr, nullptr}, {"b1", 128, 4, 0, 0, nullptr, nullptr},
                         {"b2", 128, 4, 0, 0, nullptr, nullptr}, {"b3", 128, 4, 0, 0, nullptr, nullptr},
                         {"b4", 128, 4, 0, 0, nullptr, nullptr}, {"b5", 128, 4, 0, 0, nullptr, nullptr},
                         {"dec2", 96, 4, 384, 128, "We", "be"},  {"dec1", 64, 2, 256, 96, "We", "be"}};
  for (const S& s : kS) {
    const auto st = model.find(s.stage);
    if (st == model.end()) return *err = std::string("Model M on q7: the model has no stage ") + s.stage, false;
    const StageTensors& t = st->second;
    const size_t C = (size_t)s.C, h = (size_t)s.heads, m = 4 * C;
    const Tensor* g1 = tensor(t, s.stage, "gamma1", C, err);
    const Tensor* g2 = g1 ? tensor(t, s.stage, "gamma2", C, err) : nullptr;
    const Tensor* bp = g2 ? tensor(t, s.stage, "bp", C, err) : nullptr;
    const Tensor* b1 = bp ? tensor(t, s.stage, "b1", m, err) : nullptr;
    const Tensor* b2 = b1 ? tensor(t, s.stage, "b2", C, err) : nullptr;
    const Tensor* B = b2 ? tensor(t, s.stage, "B", h * 256, err) : nullptr;
    const Tensor* Wqk = B ? tensor(t, s.stage, "Wqk", h * 32 * C, err) : nullptr;
    const Tensor* Wv = Wqk ? tensor(t, s.stage, "Wv", h * 32 * C, err) : nullptr;
    const Tensor* Wp = Wv ? tensor(t, s.stage, "Wp", C * 32 * h, err) : nullptr;
    const Tensor* W1 = Wp ? tensor(t, s.stage, "W1", m * C, err) : nullptr;
    const Tensor* W2 = W1 ? tensor(t, s.stage, "W2", C * m, err) : nullptr;
    if (!W2) return false;
    std::vector<uint8_t> blob;
    if (!l::q7pack::pack_block(g1->f, g2->f, bp->f, b1->f, b2->f, B->f, Wqk->f, Wv->f, Wp->f, W1->f, W2->f, s.C,
                               s.heads, (int)m, &blob, err))
      return *err = std::string("Model M on q7: ") + s.stage + ": " + *err, false;
    (*out)[std::string(s.stage) + ".q7"] = std::move(blob);
    if (s.lin_n) {
      const Tensor* lw = tensor(t, s.stage, s.lw, (size_t)s.lin_n * s.lin_k, err);
      const Tensor* lb = lw ? tensor(t, s.stage, s.lb, (size_t)s.lin_n, err) : nullptr;
      if (!lb) return false;
      if (!l::q7pack::pack_linear(lw->f, lb->f, s.lin_n, s.lin_k, &blob, err))
        return *err = std::string("Model M on q7: ") + s.stage + ": " + *err, false;
      (*out)[std::string(s.stage) + ".q7lin"] = std::move(blob);
    }
  }
  (*out)["q7.tab"] = l::q7pack::q7_tables();
  return true;
}

}  // namespace m_q7
}  // namespace wgrt
