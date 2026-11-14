// SPDX-License-Identifier: MIT
// Model L weight packing for the runtime; see l_pack.h.
#include "l_pack.h"

namespace wgrt {
namespace l {

namespace {

const Tensor* find(const StageTensors& t, const std::string& name) {
  const auto it = t.find(name);
  return it == t.end() ? nullptr : &it->second;
}

bool need(const StageTensors& t, const std::string& stage, const std::string& name, size_t n, const Tensor** out,
          std::string* err) {
  const Tensor* x = find(t, name);
  if (!x) return *err = stage + ": missing tensor " + name, false;
  if (x->f.size() != n)
    return *err = stage + ": tensor " + name + " has " + std::to_string(x->f.size()) + " values, expected " +
                  std::to_string(n),
           false;
  *out = x;
  return true;
}

const StageTensors* stage(const std::map<std::string, StageTensors>& p, const std::string& name, std::string* err) {
  const auto it = p.find(name);
  if (it == p.end()) return *err = "missing stage " + name, nullptr;
  return &it->second;
}

int sum(const std::vector<int>& v) {
  int n = 0;
  for (int x : v) n += x;
  return n;
}

Launch swin(const std::string& src, const std::string& offsets, const std::string& params) {
  Launch l;
  l.kind = "swin", l.blob = params, l.src = src, l.dst = src, l.C = 32, l.heads = 2, l.mlp = 128;
  l.offsets = offsets, l.params = params;
  return l;
}

}  // namespace

const std::vector<Launch>& encoder_launches() {
  static const std::vector<Launch> v = [] {
    std::vector<Launch> l(5);
    l[0].kind = "values", l[0].blob = "enc0.values", l[0].dst = "q", l[0].C = 32;
    l[1].kind = "linear", l[1].blob = "enc0.Win1", l[1].src = "q", l[1].dst = "x";
    l[1].C_in = 32, l[1].C_out = 32, l[1].w = "Win1", l[1].b = "bin1", l[1].acc = "f16", l[1].out_round = "relu_q8";
    l[2].kind = "linear", l[2].blob = "enc0.Win2", l[2].src = "x", l[2].dst = "x";
    l[2].C_in = 32, l[2].C_out = 32, l[2].w = "Win2", l[2].b = "bin2", l[2].acc = "f16", l[2].out_round = "q8";
    l[3] = swin("x", "first", "enc0_block");
    l[4].kind = "project", l[4].blob = "enc0.project", l[4].src = "x", l[4].groups = {32, 32, 32, 32};
    l[4].w = "W", l[4].b = "b", l[4].clamp = 6.28125f;
    return l;
  }();
  return v;
}

const std::vector<Launch>& decoder_launches() {
  static const std::vector<Launch> v = [] {
    std::vector<Launch> l(3);
    l[0].kind = "dec_project", l[0].blob = "dec0.project", l[0].dst = "p", l[0].w = "W", l[0].b = "b";
    l[0].groups = {32, 32};
    l[1] = swin("p", "second", "dec0_block");
    l[2].kind = "coef", l[2].blob = "dec0.coef", l[2].src = "p";
    l[2].values = {0, 1, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9};  // table.DEC0_VALUES
    return l;
  }();
  return v;
}

std::vector<int> kord_groups32(int n) {
  std::vector<int> K(n);
  for (int kb = 0; kb < n / 16; ++kb)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < 8; ++j) K[16 * kb + 8 * h + j] = 32 * (kb / 2) + 16 * h + 8 * (kb % 2) + j;
  return K;
}

bool pack_linear(const std::vector<float>& W, const std::vector<float>& b, int C_in, int C_out, int lo, const DRow& d,
                 std::vector<uint8_t>* out, std::string* err) {
  out->clear();
  if (C_in % 16 || C_out % 16 || C_out <= 0 || lo < 0 || b.size() != (size_t)C_out || W.size() % C_out)
    return *err = "linear: bad shape (C_in " + std::to_string(C_in) + ", C_out " + std::to_string(C_out) + ")", false;
  const int n = (int)(W.size() / C_out);
  if (lo + n > C_in) return *err = "linear: input channels " + std::to_string(lo) + " + " + std::to_string(n) +
                                   " outside " + std::to_string(C_in),
                     false;
  std::vector<float> Wx((size_t)C_out * C_in, 0.0f);  // zero weights outside lo .. lo + n
  for (int r = 0; r < C_out; ++r)
    for (int c = 0; c < n; ++c) Wx[(size_t)r * C_in + lo + c] = W[(size_t)r * n + c];
  const pack_parts::Order RL = pack_parts::rows_layout_l(C_out, d);
  pack_parts::put_tile_vector(out, b.data(), RL, d);
  pack_parts::put_frags(out, pack_parts::frags(Wx.data(), C_out, C_in, RL, pack_parts::kord_layout_l(C_in)));
  return true;
}

bool pack_project(const std::vector<float>& W, const std::vector<float>& b, const std::vector<int>& groups, int n_out,
                  const DRow& d, std::vector<uint8_t>* out, std::string* err) {
  out->clear();
  for (int g : groups)
    if (g != 32) return *err = "project: groups of 32 channels only", false;
  const int n = sum(groups);
  if (n_out <= 0 || n_out % 16 || b.size() != (size_t)n_out || W.size() != (size_t)n_out * n)
    return *err = "project: bad shape", false;
  const pack_parts::Order RL = pack_parts::rows_layout_l(n_out, d);
  pack_parts::put_tile_vector(out, b.data(), RL, d);
  pack_parts::put_frags(out, pack_parts::frags(W.data(), n_out, n, RL, kord_groups32(n)));
  return true;
}

bool pack_dec_project(const std::vector<float>& W, const std::vector<float>& b, const std::vector<int>& groups,
                      const DRow& d, std::vector<uint8_t>* out, std::string* err) {
  out->clear();
  for (int g : groups)
    if (g != 32) return *err = "dec_project: groups of 32 channels only", false;
  const int n = sum(groups);
  if (b.size() != 128 || W.size() != (size_t)128 * n) return *err = "dec_project: bad shape", false;
  const pack_parts::Order RE = pack_parts::rows_expand(32, d);  // tile 2 s + g: sub-pixel s, channels 16 g ..
  pack_parts::put_tile_vector(out, b.data(), RE, d);
  pack_parts::put_frags(out, pack_parts::frags(W.data(), 128, n, RE, kord_groups32(n)));
  return true;
}

bool pack_model(const std::map<std::string, StageTensors>& params, const DRow& d,
                std::map<std::string, std::vector<uint8_t>>* out, std::string* err) {
  out->clear();
  const StageTensors* enc = stage(params, "enc0", err);
  const StageTensors* dec = enc ? stage(params, "dec0", err) : nullptr;
  if (!dec) return false;
  auto block = [&](const Launch& ln, std::vector<uint8_t>* blob) {
    const StageTensors* t = stage(params, ln.params, err);
    if (!t) return false;
    if (!pack_parts::pack_swin_block(*t, "", ln.C, ln.heads, ln.mlp, d, blob, err))
      return *err = ln.params + ": " + *err, false;
    return true;
  };
  for (const Launch& ln : encoder_launches()) {
    std::vector<uint8_t> blob;
    if (ln.kind == "values") {
      // Model M's enc0 blob with a zero projection: only the dither table and the two scalars are used
      StageTensors t;
      t["W"].f.assign(64 * 80, 0.0f);
      t["b"].f.assign(64, 0.0f);
      const Tensor* T = find(*enc, "T");
      if (T && T->u.size() != 8 * 16 * 16) return *err = "enc0: tensor T must hold 2048 uint8 values", false;
      t["T"].u = T ? T->u : std::vector<uint8_t>(8 * 16 * 16, 0);
      for (const char* s : {"sigma1", "sigma2"}) {
        const Tensor* x = find(*enc, s);
        if (x && x->f.size() != 1) return *err = std::string("enc0: tensor ") + s + " must hold one value", false;
        t[s].f = {x ? x->f[0] : 0.0f};
      }
      if (!pack_stage_blob("enc0", t, d, &blob, err)) return false;
    } else if (ln.kind == "linear") {
      const Tensor *b, *W = find(*enc, ln.w);
      if (!W) return *err = "enc0: missing tensor " + ln.w, false;
      if (!need(*enc, "enc0", ln.b, (size_t)ln.C_out, &b, err)) return false;
      if (!pack_linear(W->f, b->f, ln.C_in, ln.C_out, 0, d, &blob, err))
        return *err = "enc0." + ln.w + ": " + *err, false;
    } else if (ln.kind == "swin") {
      if (!block(ln, &blob)) return false;
    } else if (ln.kind == "project") {
      const int n_out = 64;
      const Tensor *W, *b;
      if (!need(*enc, "enc0", ln.w, (size_t)n_out * sum(ln.groups), &W, err) ||
          !need(*enc, "enc0", ln.b, (size_t)n_out, &b, err))
        return false;
      if (!pack_project(W->f, b->f, ln.groups, n_out, d, &blob, err)) return *err = "enc0: " + *err, false;
    }
    (*out)[ln.blob] = std::move(blob);
  }
  for (const Launch& ln : decoder_launches()) {
    std::vector<uint8_t> blob;
    if (ln.kind == "dec_project") {
      const Tensor *W, *b;
      if (!need(*dec, "dec0", ln.w, (size_t)128 * sum(ln.groups), &W, err) ||
          !need(*dec, "dec0", ln.b, 128, &b, err))
        return false;
      if (!pack_dec_project(W->f, b->f, ln.groups, d, &blob, err)) return *err = "dec0: " + *err, false;
    } else if (ln.kind == "swin") {
      if (!block(ln, &blob)) return false;
    } else if (ln.kind == "coef") {
      // Model M's dec0 blob with a zero projection: the scalars S, T and the code tables
      StageTensors t;
      t["W"].f.assign(68 * 64, 0.0f);
      t["b"].f.assign(68, 0.0f);
      for (const char* s : {"S", "T"}) {
        const Tensor* x;
        if (!need(*dec, "dec0", s, 1, &x, err)) return false;
        t[s] = *x;
      }
      if (!pack_stage_blob("dec0", t, d, &blob, err)) return false;
    }
    (*out)[ln.blob] = std::move(blob);
  }
  return true;
}

}  // namespace l
}  // namespace wgrt
