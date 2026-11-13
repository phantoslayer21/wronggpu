// SPDX-License-Identifier: MIT
// Model K weight packing for the runtime; see k_pack.h. Port of clean/k/runner/k_pack.py (d_row "gfx12") and
// clean/k/reference/params.py normalize; layouts in clean/k/FORMATS.md.
#include "k_pack.h"

#include <cmath>
#include <cstring>

#include "../../runtime/fpconv.h"
#include "model_k.h"

namespace wgrt {

namespace {

constexpr size_t kAlign = 256;

size_t pad16(size_t n) { return (n + 15) / 16 * 16; }

std::string shape_text(const std::vector<int>& s) {
  std::string t = "[";
  for (size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]);
  return t + "]";
}

// A stage's tensors in the canonical form of params.py: name -> float32 data (row-major) with its canonical shape.
struct Canon {
  std::map<std::string, std::pair<std::vector<int>, const std::vector<float>*>> t;
  const float* data(const std::string& name) const { return t.at(name).second->data(); }
};

// params.py expected_shapes: canonical shape and the converter's (spec 48 section 7.2) shape when it differs.
struct Expected {
  std::string name;
  std::vector<int> canon, conv;
};

std::vector<Expected> expected_shapes(const KStageInfo& s) {
  const int C = s.C, H = s.heads, D = s.D(), HID = s.hidden();
  std::vector<Expected> v = {{"g1", {C}, {}},       {"g2", {C}, {}},   {"Wv", {H, 32, C}, {}}, {"Wo", {C, D}, {}},
                             {"bo", {C}, {}},       {"W1", {HID, C}, {}}, {"b1", {HID}, {}}, {"W2", {C, HID}, {}},
                             {"b2", {C}, {}}};
  if (s.fixed) {
    v.push_back({"M", {H, 64, 64}, {H, 8, 8, 8, 8}});
  } else {
    v.insert(v.end(), {{"Wq", {H, 32, C}, {}}, {"Wk", {H, 32, C}, {}}, {"B", {H, 64, 64}, {H, 8, 8, 8, 8}}});
  }
  if (s.embed) v.insert(v.end(), {{"We", {32, 16}, {}}, {"be", {32}, {}}});
  if (s.kind == 'e') v.insert(v.end(), {{"Wm", {s.C_next, 4 * C}, {s.C_next, 4, C}}, {"bm", {s.C_next}, {}}});
  if (s.kind == 'd') v.insert(v.end(), {{"Wx", {4 * C, s.C_prev}, {4, C, s.C_prev}}, {"bx", {4 * C}, {4, C}}});
  if (s.head) v.insert(v.end(), {{"Wh", {40, C}, {}}, {"bh", {40}, {}}});
  return v;
}

// params.py normalize: both shape forms flatten to the same row-major order, so normalising only checks the shape.
bool normalize(const std::string& stage, const KStageInfo& s, const StageTensors& in, Canon* out, std::string* err) {
  for (const Expected& e : expected_shapes(s)) {
    auto it = in.find(e.name);
    if (it == in.end() && e.name == "g1") it = in.find("gamma1");
    if (it == in.end()) return *err = stage + ": tensor " + e.name + " missing", false;
    const Tensor& x = it->second;
    size_t n = 1;
    for (int d : x.shape) n *= (size_t)(d < 0 ? 0 : d);
    if (x.shape != e.canon && (e.conv.empty() || x.shape != e.conv))
      return *err = stage + ": " + e.name + " has shape " + shape_text(x.shape) + ", expected " +
                    shape_text(e.canon),
             false;
    if (x.f.size() != n)
      return *err = stage + ": " + e.name + " has " + std::to_string(x.f.size()) + " float32 values for shape " +
                    shape_text(x.shape),
             false;
    out->t[e.name] = {e.canon, &x.f};
  }
  return true;
}

// k_pack.py f16_exact: FP16 bits of every value (round to nearest even), or false if one is not exactly representable.
// NaN is refused (the loader never returns one; k_pack.py would keep it as some FP16 NaN).
bool f16_exact(const float* x, size_t n, uint16_t* out, const std::string& what, std::string* err) {
  for (size_t i = 0; i < n; ++i) {
    const uint16_t h = f32_to_f16(x[i]);
    if (std::isnan(x[i]) || f16_to_f32(h) != x[i])
      return *err = what + " must be exactly representable in FP16 (value " + std::to_string(i) + ")", false;
    out[i] = h;
  }
  return true;
}

// k_pack.py pack_tiles, gfx12 d_row (physical A row r = logical row r): W [n, k] -> tiles [NT][KT] ("nk") or [KT][NT]
// ("kn") of 32 lanes x 8 halfs, lane 16 h + r = row 16 nt + r, inputs 16 kt + 8 h + 0..7; rows padded to 16 with 0.
bool pack_tiles(const float* W, size_t n, size_t k, bool kn, const std::string& what, std::vector<uint16_t>* out,
                std::string* err) {
  if (k % 16) return *err = what + ": the input side must be a multiple of 16", false;
  const size_t np = pad16(n), NT = np / 16, KT = k / 16;
  std::vector<uint16_t> w(np * k, 0);
  if (!f16_exact(W, n * k, w.data(), what, err)) return false;
  const size_t base = out->size();
  out->resize(base + np * k);
  uint16_t* o = out->data() + base;
  for (size_t nt = 0; nt < NT; ++nt)
    for (size_t kt = 0; kt < KT; ++kt) {
      uint16_t* tile = o + (kn ? kt * NT + nt : nt * KT + kt) * 256;
      for (size_t h = 0; h < 2; ++h)
        for (size_t r = 0; r < 16; ++r)
          std::memcpy(tile + (16 * h + r) * 8, &w[(16 * nt + r) * k + 16 * kt + 8 * h], 16);
    }
  return true;
}

// k_pack.py Blob: sections padded to multiples of 256 bytes.
void add_section(std::vector<uint8_t>* blob, const std::vector<uint16_t>& h) {
  const size_t base = blob->size(), bytes = h.size() * 2;
  blob->resize(base + (bytes + kAlign - 1) / kAlign * kAlign, 0);
  if (bytes) std::memcpy(blob->data() + base, h.data(), bytes);
}

bool add_vector(std::vector<uint8_t>* blob, const Canon& c, const std::string& name, size_t pad_to,
                const std::string& stage, std::string* err) {
  const auto& e = c.t.at(name);
  const size_t n = e.second->size();
  std::vector<uint16_t> h(n > pad_to ? n : pad_to, 0);
  if (!f16_exact(e.second->data(), n, h.data(), stage + ": " + name, err)) return false;
  add_section(blob, h);
  return true;
}

// k_pack.py pack_linear: tiles (nk) | bias [N padded to 16].
bool pack_linear(std::vector<uint8_t>* blob, const Canon& c, const char* wn, const char* bn, const std::string& stage,
                 std::string* err) {
  const std::vector<int>& sh = c.t.at(wn).first;
  std::vector<uint16_t> tiles;
  if (!pack_tiles(c.data(wn), (size_t)sh[0], (size_t)sh[1], false, stage + ": " + wn, &tiles, err)) return false;
  add_section(blob, tiles);
  return add_vector(blob, c, bn, pad16((size_t)sh[0]), stage, err);
}

// k_pack.py pack_block: g1 | g2 | qkv | mix | wo | bo | w1 | b1 | w2 (kn) | b2.
bool pack_block(std::vector<uint8_t>* blob, const Canon& c, const KStageInfo& s, const std::string& stage,
                std::string* err) {
  const size_t C = s.C, H = s.heads, D = s.D(), HID = s.hidden();
  if (!add_vector(blob, c, "g1", 0, stage, err) || !add_vector(blob, c, "g2", 0, stage, err)) return false;
  std::vector<const char*> names = {"Wv"};
  if (!s.fixed) names = {"Wq", "Wk", "Wv"};
  std::vector<uint16_t> qkv;
  for (size_t h = 0; h < H; ++h)
    for (const char* nm : names)
      if (!pack_tiles(c.data(nm) + h * 32 * C, 32, C, false, stage + ": " + nm, &qkv, err)) return false;
  add_section(blob, qkv);
  if (!add_vector(blob, c, s.fixed ? "M" : "B", 0, stage, err)) return false;
  std::vector<uint16_t> t;
  if (!pack_tiles(c.data("Wo"), C, D, false, stage + ": Wo", &t, err)) return false;
  add_section(blob, t);
  if (!add_vector(blob, c, "bo", 0, stage, err)) return false;
  t.clear();
  if (!pack_tiles(c.data("W1"), HID, C, false, stage + ": W1", &t, err)) return false;
  add_section(blob, t);
  if (!add_vector(blob, c, "b1", 0, stage, err)) return false;
  t.clear();
  if (!pack_tiles(c.data("W2"), C, HID, true, stage + ": W2", &t, err)) return false;
  add_section(blob, t);
  return add_vector(blob, c, "b2", 0, stage, err);
}

}  // namespace

std::vector<uint8_t> k_pack_stage(const std::string& stage, const StageTensors& t, std::string* err) {
  const KStageInfo* s = k_stage(stage);
  if (!s) return *err = "unknown model K stage \"" + stage + "\"", std::vector<uint8_t>();
  Canon c;
  if (!normalize(stage, *s, t, &c, err)) return {};
  std::vector<uint8_t> blob;
  blob.reserve(k_stage_blob_bytes(*s));
  bool ok = true;
  // k_pack.py stage_parts: embed | expand | block | merge | head
  if (ok && s->embed) ok = pack_linear(&blob, c, "We", "be", stage, err);
  if (ok && s->kind == 'd') ok = pack_linear(&blob, c, "Wx", "bx", stage, err);
  if (ok) ok = pack_block(&blob, c, *s, stage, err);
  if (ok && s->kind == 'e') ok = pack_linear(&blob, c, "Wm", "bm", stage, err);
  if (ok && s->head) ok = pack_linear(&blob, c, "Wh", "bh", stage, err);
  if (!ok) return {};
  if (blob.size() != k_stage_blob_bytes(*s)) {
    *err = stage + ": packed " + std::to_string(blob.size()) + " bytes, expected " +
           std::to_string(k_stage_blob_bytes(*s)) + " (internal error)";
    return {};
  }
  return blob;
}

}  // namespace wgrt
