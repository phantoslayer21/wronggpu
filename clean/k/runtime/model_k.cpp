// SPDX-License-Identifier: MIT
// Model K layer table of the per-frame runtime; see model_k.h.
#include "model_k.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "../../runtime/model_select.h"
#include "../../runtime/wg_runtime.h"

namespace wgrt {

namespace {

int cdiv(int a, int b) { return (a + b - 1) / b; }
size_t align256(size_t n) { return (n + 255) / 256 * 256; }
size_t pad16(size_t n) { return (n + 15) / 16 * 16; }


size_t linear_bytes(size_t n, size_t k) { return align256(pad16(n) * k * 2) + align256(pad16(n) * 2); }

size_t block_bytes(const KStageInfo& s) {
  const size_t C = s.C, D = s.D(), H = s.heads, HID = s.hidden();
  const size_t qkv = (s.fixed ? 1 : 3) * H * 32 * C * 2;
  return align256(C * 2) * 2 + align256(qkv) + align256(H * 64 * 64 * 2) + align256(C * D * 2) + align256(C * 2) +
         align256(HID * C * 2) + align256(HID * 2) + align256(C * HID * 2) + align256(C * 2);
}

}  // namespace

const std::vector<KStageInfo>& k_stage_table() {
  //                                    name kind lvl  C  heads fixed  next prev embed  head
  static const std::vector<KStageInfo> t = {
      {"e0", 'e', 0, 32, 2, true, 64, 0, true, false},     {"e1", 'e', 1, 64, 2, true, 64, 0, false, false},
      {"e2", 'e', 2, 64, 2, false, 96, 0, false, false},   {"e3", 'e', 3, 96, 4, false, 128, 0, false, false},
      {"e4", 'e', 4, 128, 4, false, 160, 0, false, false}, {"m", 'm', 5, 160, 8, false, 0, 0, false, false},
      {"d4", 'd', 4, 128, 4, false, 0, 160, false, false}, {"d3", 'd', 3, 96, 4, false, 0, 128, false, false},
      {"d2", 'd', 2, 64, 2, false, 0, 96, false, false},   {"d1", 'd', 1, 64, 2, false, 0, 64, false, false},
      {"d0", 'd', 0, 32, 2, true, 0, 64, false, true},
  };
  return t;
}

const KStageInfo* k_stage(const std::string& name) {
  for (const KStageInfo& s : k_stage_table())
    if (name == s.name) return &s;
  return nullptr;
}

void k_base_size(int ow, int oh, double s, int* wb, int* hb) {
  auto one = [s](int n) {
    const int scaled = (int)std::ceil(s * n - 1e-9);
    return std::max(64 * cdiv(scaled, 64), 512) / 2;
  };
  *wb = one(ow);
  *hb = one(oh);
}

void k_level_size(int wb, int hb, int level, int* w, int* h) {
  *w = cdiv(wb, 1 << level);
  *h = cdiv(hb, 1 << level);
}

void k_stage_offsets(int f, const int table[8][2], int off[11][2]) {
  const int m = ((f % 8) + 8) % 8;
  const int oxm = table[m][0], oym = table[m][1];
  const int o45x = ((2 * oxm - 4) % 8 + 8) % 8, o45y = ((2 * oym - 4) % 8 + 8) % 8;
  const auto& t = k_stage_table();
  for (size_t i = 0; i < t.size(); ++i) {
    const std::string n = t[i].name;
    off[i][0] = n == "m" ? oxm : (n == "e4" || n == "d4") ? o45x : 4;
    off[i][1] = n == "m" ? oym : (n == "e4" || n == "d4") ? o45y : 4;
  }
}

int k_window_count(int w, int h, int ox, int oy) { return cdiv(w + ox, 8) * cdiv(h + oy, 8); }

std::vector<KBuffer> k_buffer_plan(int wb, int hb) {
  auto tokens = [&](int level) {
    int w, h;
    k_level_size(wb, hb, level, &w, &h);
    return (size_t)w * h;
  };
  std::vector<KBuffer> plan = {{"X0", tokens(0) * 16 * 2}};
  auto need = [&](const std::string& name, size_t bytes) {
    for (KBuffer& b : plan)
      if (b.name == name) {
        b.bytes = std::max(b.bytes, bytes);
        return;
      }
    plan.push_back({name, bytes});
  };
  // the fused stage kernels, as clean/k/reference/layout.py FUSED_IO: merged maps and decoder outputs alternate
  // between A and B; d0 writes only the head output
  static const char* const kIn[11] = {"X0", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"};
  const auto& t = k_stage_table();
  for (size_t i = 0; i < t.size(); ++i) {
    const KStageInfo& s = t[i];
    const std::string other = std::string(kIn[i]) == "A" ? "B" : "A";
    if (s.kind == 'e') {
      need("S" + std::to_string(s.level), tokens(s.level) * s.C * 2);
      need(other, tokens(s.level + 1) * s.C_next * 2);
    } else if (s.head) {
      need("OUT", tokens(0) * 40 * 2);
    } else {
      need(other, tokens(s.level) * s.C * 2);
    }
  }
  for (KBuffer& b : plan) b.bytes = align256(b.bytes);
  return plan;
}

size_t k_stage_blob_bytes(const KStageInfo& s) {
  size_t n = block_bytes(s);
  if (s.embed) n += linear_bytes(32, 16);
  if (s.kind == 'e') n += linear_bytes(s.C_next, 4 * s.C);
  if (s.kind == 'd') n += linear_bytes(4 * s.C, s.C_prev);
  if (s.head) n += linear_bytes(40, s.C);
  return n;
}

bool model_from_options(const WgRtConfig& c, ModelId* model, std::string* err) {
  const char* env = std::getenv("WG_RUNTIME_OPTIONS");
  const std::string text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
  *model = ModelId::M;
  std::string bad;
  if (!last_model_option(text, model, &bad)) {
    *err = "unknown model \"" + bad + "\" in the runtime options (known: M, K" + (model_l_built() ? ", L)" : ")");
    return false;
  }
  if (*model == ModelId::L && !model_l_built()) {  // #89: recognised; runs when its sources are compiled in
    *err = model_not_built_yet(ModelId::L);
    return false;
  }
  return true;
}

bool model_k_create_check(const WgRtConfig& c, std::string* err) {
  ModelId m;
  return model_from_options(c, &m, err);
}

}  // namespace wgrt
