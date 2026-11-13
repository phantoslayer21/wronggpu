// SPDX-License-Identifier: MIT
// Loading the converted model K; see model_k_load.h. Container: clean/converter/OUTPUT-FORMAT.md.
#include "model_k_load.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>

#include "../../converter/src/json.h"
#include "../../converter/src/sha256.h"
#include "model_k.h"

namespace wgrt {

namespace {

enum DType : uint64_t { kF32 = 0, kU8 = 1, kI32 = 2 };  // OUTPUT-FORMAT.md index dtype codes

bool read_all(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

// Little-endian reader of [pos, end) with bounds checks; any failure sticks in ok.
struct Reader {
  const std::vector<uint8_t>& d;
  size_t pos, end;
  bool ok = true;
  bool need(size_t n) {
    if (!ok || n > end - pos) ok = false;
    return ok;
  }
  uint64_t le(int n) {
    if (!need((size_t)n)) return 0;
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v |= (uint64_t)d[pos + i] << (8 * i);
    pos += (size_t)n;
    return v;
  }
  std::string str() {
    const size_t n = (size_t)le(2);
    if (!need(n)) return std::string();
    std::string s((const char*)&d[pos], n);
    pos += n;
    return s;
  }
  void bytes(uint8_t* out, size_t n) {
    if (!need(n)) return;
    std::memcpy(out, &d[pos], n);
    pos += n;
  }
};

struct Entry {  // one tensor of the index
  std::string name;
  uint64_t dtype;
  std::vector<int> shape;
  uint64_t off, size;
};

struct Group {  // one output group of the index
  std::string name;
  wgc::Digest digest;
  std::vector<Entry> tensors;
};

}  // namespace

const std::vector<std::string>& k_model_groups() {
  static const std::vector<std::string> g = {"e0", "e1", "e2", "e3", "e4", "m", "d4", "d3", "d2", "d1", "d0", "net"};
  return g;
}

std::vector<KTensorSpec> k_stage_tensor_specs(const std::string& stage) {
  const KStageInfo* s = k_stage(stage);
  if (!s) return {};
  const int C = s->C, H = s->heads, D = s->D(), HID = s->hidden();
  std::vector<KTensorSpec> v;
  if (s->embed) v.insert(v.end(), {{"We", {32, 16}}, {"be", {32}}});  // e0 embedding
  v.push_back({"g1", {C}});
  if (s->fixed) {
    v.insert(v.end(), {{"Wv", {H, 32, C}}, {"M", {H, 8, 8, 8, 8}}});  // fixed mixing (e0, e1, d0)
  } else {
    v.insert(v.end(), {{"Wq", {H, 32, C}}, {"Wk", {H, 32, C}}, {"Wv", {H, 32, C}}, {"B", {H, 8, 8, 8, 8}}});
  }
  v.insert(v.end(), {{"Wo", {C, D}}, {"bo", {C}}, {"g2", {C}}, {"W1", {HID, C}}, {"b1", {HID}}, {"W2", {C, HID}},
                     {"b2", {C}}});
  if (s->head) v.insert(v.end(), {{"Wh", {40, 32}}, {"bh", {40}}});  // d0 output head (before the expand in map order)
  if (s->kind == 'e') v.insert(v.end(), {{"Wm", {s->C_next, 4, C}}, {"bm", {s->C_next}}});
  if (s->kind == 'd') v.insert(v.end(), {{"Wx", {4, C, s->C_prev}}, {"bx", {4, C}}});
  return v;
}

bool load_k_model(const std::string& path, const std::string& map_path, KModel* out, std::string* err) {
  std::vector<uint8_t> mbytes;
  if (!read_all(map_path, &mbytes)) return *err = "converter map missing: " + map_path, false;
  return load_k_model_map(path, mbytes, map_path, out, err);
}

bool load_k_model_map(const std::string& path, const std::vector<uint8_t>& mbytes, const std::string& map_path,
                      KModel* out, std::string* err) {
  std::vector<uint8_t> d;
  if (!read_all(path, &d)) return *err = "model data missing: " + path, false;
  const std::string bad = "model data damaged: " + path;
  if (d.size() < 88 + 32 || std::memcmp(d.data(), "WGCMODEL", 8) != 0)
    return *err = bad + " is not a WrongGPU model file", false;
  const size_t body = d.size() - 32;  // the trailer: SHA-256 of every byte before it
  const wgc::Digest whole = wgc::Sha256::hash(d.data(), body);
  if (std::memcmp(whole.data(), d.data() + body, 32) != 0) return *err = bad + " fails its integrity check", false;
  Reader r{d, 8, body};
  const uint64_t version = r.le(4), ngroups = r.le(4);
  if (version != 1) return *err = bad + ": unknown container version " + std::to_string(version), false;
  wgc::Digest key, map_hash;
  r.bytes(key.data(), 32);
  r.bytes(map_hash.data(), 32);
  const uint64_t isize = r.le(8);
  if (!r.ok || isize > body - 88) return *err = bad + ": bad header", false;

  // The map shipped with the runtime: the file must have been converted with it (spec 48 section 5 / 08 section 5),
  // and its digests are the reference.
  if (wgc::Sha256::hash(mbytes.data(), mbytes.size()) != map_hash)
    return *err = "model data does not match this installation (converted with a different map): " + path, false;
  wgc::Json map;
  std::string jerr;
  if (!wgc::parse_json(std::string(mbytes.begin(), mbytes.end()), map, jerr))
    return *err = "converter map damaged: " + map_path + " (" + jerr + ")", false;
  std::map<std::string, wgc::Digest> map_digest;
  const wgc::Json* mstages = map.find("stages");
  if (!mstages || !mstages->is_array()) return *err = "converter map damaged: " + map_path + " has no stages", false;
  for (const wgc::Json& st : mstages->items) {
    const wgc::Json* nm = st.find("stage");
    const wgc::Json* dg = st.find("digest");
    wgc::Digest v;
    if (!nm || !nm->is_string() || !dg || !dg->is_string() || !wgc::parse_digest_hex(dg->text, v) ||
        map_digest.count(nm->text))
      return *err = "converter map damaged: " + map_path + " (stage entry)", false;
    map_digest[nm->text] = v;
  }

  // Index: output groups in map order, each with its digest and tensors.
  Reader ix{d, 88, 88 + (size_t)isize};
  const size_t data_start = 88 + (size_t)isize;
  std::vector<Group> groups;
  for (uint64_t g = 0; g < ngroups && ix.ok; ++g) {
    Group grp;
    grp.name = ix.str();
    ix.bytes(grp.digest.data(), 32);
    const uint64_t nt = ix.le(4);
    if (!ix.ok || nt > 64) return *err = bad + ": bad index", false;
    for (uint64_t t = 0; t < nt && ix.ok; ++t) {
      Entry e;
      e.name = ix.str();
      e.dtype = ix.le(1);
      const uint64_t rank = ix.le(1);
      if (rank > 8) return *err = bad + ": bad tensor entry " + grp.name + "/" + e.name, false;
      uint64_t n = 1;
      for (uint64_t a = 0; a < rank && ix.ok; ++a) {
        const uint64_t len = ix.le(4);
        n *= len;  // n <= 2^32 before, len < 2^32: no overflow
        if (len > (1u << 24) || n > ((uint64_t)1 << 32))
          return *err = bad + ": bad tensor entry " + grp.name + "/" + e.name, false;
        e.shape.push_back((int)len);
      }
      e.off = ix.le(8);
      e.size = ix.le(8);
      const uint64_t esize = e.dtype == kU8 ? 1 : 4;
      if (!ix.ok || e.dtype > kI32 || e.size != n * esize || e.off % 64 ||
          e.off < data_start || e.off > body || e.size > body - e.off)
        return *err = bad + ": bad tensor entry " + grp.name + "/" + e.name, false;
      grp.tensors.push_back(std::move(e));
    }
    groups.push_back(std::move(grp));
  }
  if (!ix.ok || ix.pos != ix.end) return *err = bad + ": truncated or malformed index", false;

  // Digests: keyed SHA-256 (spec 08 section 4.1: SHA-256(key || data)) of each group's tensor data in index order
  // (spec 48 section 7.4), against the stored digest and the map's. Every group of the map must be in the file.
  std::set<std::string> seen;
  for (const Group& grp : groups) {
    if (!seen.insert(grp.name).second) return *err = bad + ": group " + grp.name + " appears twice", false;
    wgc::Sha256 h;
    h.update(key.data(), key.size());
    for (const Entry& e : grp.tensors) h.update(&d[(size_t)e.off], (size_t)e.size);
    const wgc::Digest computed = h.finish();
    auto md = map_digest.find(grp.name);
    if (computed != grp.digest || md == map_digest.end() || md->second != computed)
      return *err = bad + ": stage " + grp.name + " does not match the map's digest", false;
  }
  for (const auto& md : map_digest)
    if (!seen.count(md.first)) return *err = bad + ": stage " + md.first + " missing", false;

  // Every group and tensor the runtime needs (spec 48 sections 7.2, 7.3), with dtypes and shapes.
  auto find_group = [&](const std::string& name) -> const Group* {
    for (const Group& g : groups)
      if (g.name == name) return &g;
    return nullptr;
  };
  KModel model;
  for (const std::string& name : k_model_groups()) {
    const Group* g = find_group(name);
    if (!g) return *err = bad + ": stage " + name + " missing", false;
    if (name == "net") {
      if (g->tensors.size() != 1 || g->tensors[0].name != "window_offsets" || g->tensors[0].dtype != kI32 ||
          g->tensors[0].shape != std::vector<int>{8, 2})
        return *err = bad + ": net must hold exactly window_offsets, int32 [8][2]", false;
      for (int i = 0; i < 16; ++i) {
        const size_t p = (size_t)g->tensors[0].off + 4 * (size_t)i;
        const int32_t v = (int32_t)((uint32_t)d[p] | (uint32_t)d[p + 1] << 8 | (uint32_t)d[p + 2] << 16 |
                                    (uint32_t)d[p + 3] << 24);
        if (v < 0 || v > 7) return *err = bad + ": net/window_offsets has a value outside 0..7", false;
        model.window_offsets[i / 2][i % 2] = v;
      }
      continue;
    }
    const std::vector<KTensorSpec> specs = k_stage_tensor_specs(name);
    if (g->tensors.size() != specs.size())
      return *err = bad + ": stage " + name + " has " + std::to_string(g->tensors.size()) + " tensors, expected " +
                    std::to_string(specs.size()),
             false;
    StageTensors st;
    for (size_t i = 0; i < specs.size(); ++i) {
      const Entry& e = g->tensors[i];
      const std::string where = name + "/" + specs[i].name;
      if (e.name != specs[i].name) return *err = bad + ": tensor " + where + " missing (found " + e.name + ")", false;
      if (e.dtype != kF32 || e.shape != specs[i].shape)
        return *err = bad + ": tensor " + where + " has the wrong shape or type", false;
      Tensor t;
      t.shape = e.shape;
      t.f.resize((size_t)e.size / 4);
      std::memcpy(t.f.data(), &d[(size_t)e.off], (size_t)e.size);  // little-endian hosts only (x86-64)
      for (float v : t.f)
        if (!std::isfinite(v)) return *err = "model data damaged: " + where + " has a non-finite value", false;
      st[specs[i].name] = std::move(t);
    }
    model.stages[name] = std::move(st);
  }
  *out = std::move(model);
  return true;
}

}  // namespace wgrt
