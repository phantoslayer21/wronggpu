// SPDX-License-Identifier: MIT
// Loader of the runtime's model (rr_model.h): the converter's model.wgcm (checked and packed by rr_pack.cpp) or the
// runtime's own container WGRRMOD1 (layout: clean/rr_f/runtime/tools/pack_model.py). Checks of WGRRMOD1: magic,
// version, table bounds, the whole-file SHA-256.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>

#include "rr_model.h"

#include <stdio.h>
#include <string.h>

#pragma comment(lib, "bcrypt.lib")

namespace wgrr_rt {

static bool sha256(const uint8_t* data, size_t n, uint8_t out[32]) {
  BCRYPT_ALG_HANDLE alg = nullptr;
  BCRYPT_HASH_HANDLE h = nullptr;
  bool ok = BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0 &&
            BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) == 0;
  while (ok && n) {
    const ULONG part = (ULONG)(n > 0x40000000u ? 0x40000000u : n);
    ok = BCryptHashData(h, (PUCHAR)data, part, 0) == 0;
    data += part;
    n -= part;
  }
  ok = ok && BCryptFinishHash(h, out, 32, 0) == 0;
  if (h) BCryptDestroyHash(h);
  if (alg) BCryptCloseAlgorithmProvider(alg, 0);
  return ok;
}

static std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

std::string sha256_hex(const uint8_t* data, size_t n) {
  uint8_t h[32];
  return sha256(data, n, h) ? hex(h, 32) : std::string();
}

static bool load_own(const std::vector<uint8_t>& d, Model* out, std::string* err) {
  if (d.size() < 64 + 32) return *err = "not a WrongGPU RR model file", false;
  uint8_t h[32];
  if (!sha256(d.data(), d.size() - 32, h) || memcmp(h, d.data() + d.size() - 32, 32) != 0)
    return *err = "model file damaged (SHA-256 mismatch)", false;
  auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, d.data() + o, 4); return v; };
  auto u64 = [&](size_t o) { uint64_t v; memcpy(&v, d.data() + o, 8); return v; };
  out->version = u32(8);
  if (out->version != 1) return *err = "unsupported model container version", false;
  out->flags = u32(12);
  const uint32_t nb = u32(16), ns = u32(20);
  const uint64_t isize = u64(24);
  out->logical_sha256 = hex(d.data() + 32, 32);
  size_t o = 64;
  const size_t tables = (size_t)nb * 64 + (size_t)ns * 52;
  if (o + tables > d.size()) return *err = "model file damaged (tables)", false;
  for (uint32_t i = 0; i < nb; ++i, o += 64) {
    std::string name((const char*)d.data() + o, strnlen((const char*)d.data() + o, 48));
    const uint64_t off = u64(o + 48), size = u64(o + 56);
    if (off > isize || size > isize - off) return *err = "model file damaged (blob " + name + ")", false;
    out->blobs[name] = {off, size};
  }
  for (uint32_t i = 0; i < ns; ++i, o += 52) {
    std::string name((const char*)d.data() + o, strnlen((const char*)d.data() + o, 48));
    out->scalars[name] = u32(o + 48);
  }
  o = (o + 255) / 256 * 256;
  // (d.size() >= 96 here; written so that neither sum can wrap, and so that `o` is inside the file before it is used)
  if (o > d.size() - 32 || isize != d.size() - 32 - o || isize % 256) return *err = "model file damaged (image size)", false;
  out->image.assign(d.begin() + o, d.begin() + o + isize);
  out->kind = "wgrrmod1";
  return true;
}

bool load_model(const std::wstring& path, Model* out, std::string* err) {
  *out = Model();
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) return *err = "model file not found", false;
  std::vector<uint8_t> d;
  {
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) d.insert(d.end(), buf, buf + n);
    fclose(f);
  }
  {
    const int n = WideCharToMultiByte(CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0, nullptr, nullptr);
    out->path.assign((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), (int)path.size(), &out->path[0], n, nullptr, nullptr);
  }
  out->file_sha256 = sha256_hex(d.data(), d.size());
  bool ok;
  if (d.size() >= 8 && memcmp(d.data(), "WGRRMOD1", 8) == 0) {
    ok = load_own(d, out, err);
  } else if (d.size() >= 8 && memcmp(d.data(), "WGCMODEL", 8) == 0) {
    std::vector<WgcmGroup> groups;
    ok = read_wgcm(d, &groups, err) && pack_wgcm(groups, out, err);
    if (ok) {
      out->kind = "wgcm";
      out->logical_sha256 = out->file_sha256;
    }
  } else {
    return *err = "not a model file (neither a converter model.wgcm nor a WrongGPU RR container)", false;
  }
  if (!ok) {
    const std::string p = out->path;
    *out = Model();
    out->path = p;
    return false;
  }
  out->ok = true;
  return true;
}

// ---- model.wgcm (clean/converter/OUTPUT-FORMAT.md, container version 2) ------------------------------------------
namespace {
struct Hasher {
  BCRYPT_ALG_HANDLE alg = nullptr;
  BCRYPT_HASH_HANDLE h = nullptr;
  bool ok = false;
  Hasher() {
    ok = BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0 &&
         BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) == 0;
  }
  ~Hasher() {
    if (h) BCryptDestroyHash(h);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
  }
  void add(const uint8_t* p, uint64_t n) {
    while (ok && n) {
      const ULONG part = (ULONG)(n > 0x40000000u ? 0x40000000u : n);
      ok = BCryptHashData(h, (PUCHAR)p, part, 0) == 0;
      p += part;
      n -= part;
    }
  }
  bool finish(uint8_t out[32]) { return ok && BCryptFinishHash(h, out, 32, 0) == 0; }
};
const uint32_t kDtypeSize[7] = {4, 1, 4, 2, 2, 4, 1};
}  // namespace

bool read_wgcm(const std::vector<uint8_t>& d, std::vector<WgcmGroup>* groups, std::string* err) {
  groups->clear();
  if (d.size() < 88 + 32 || memcmp(d.data(), "WGCMODEL", 8) != 0) return *err = "not a converter model file", false;
  const size_t body = d.size() - 32;
  uint8_t h[32];
  if (!sha256(d.data(), body, h) || memcmp(h, d.data() + body, 32) != 0)
    return *err = "model.wgcm damaged: the whole-file hash does not match", false;
  auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, d.data() + o, 2); return v; };
  auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, d.data() + o, 4); return v; };
  auto u64 = [&](size_t o) { uint64_t v; memcpy(&v, d.data() + o, 8); return v; };
  if (u32(8) != 2) return *err = "model.wgcm: container version is not 2 (not a preset-F file)", false;
  const uint32_t ngroups = u32(12);
  const uint8_t* key = d.data() + 16;
  const uint64_t index_size = u64(80);
  const uint64_t end = 88 + index_size;
  if (end > body) return *err = "model.wgcm damaged: index runs past the data", false;
  uint64_t pos = 88;
  auto need = [&](uint64_t n) { return pos + n <= end; };
  auto name_at = [&](std::string* s) {
    if (!need(2)) return false;
    const uint16_t n = u16((size_t)pos);
    if (!need(2ull + n)) return false;
    s->assign((const char*)d.data() + pos + 2, n);
    pos += 2ull + n;
    return true;
  };
  for (uint32_t g = 0; g < ngroups; ++g) {
    WgcmGroup grp;
    if (!name_at(&grp.name) || !need(36)) return *err = "model.wgcm damaged: index entry runs past the index", false;
    uint8_t digest[32];
    memcpy(digest, d.data() + pos, 32);
    const uint32_t nt = u32((size_t)pos + 32);
    pos += 36;
    Hasher hs;
    hs.add(key, 32);
    for (uint32_t t = 0; t < nt; ++t) {
      WgcmTensor tn;
      if (!name_at(&tn.name) || !need(2)) return *err = "model.wgcm damaged: index entry of " + grp.name, false;
      tn.dtype = d[(size_t)pos];
      const uint32_t rank = d[(size_t)pos + 1];
      pos += 2;
      if (tn.dtype > 6) return *err = "model.wgcm damaged: unknown dtype in " + grp.name + "/" + tn.name, false;
      if (!need(4ull * rank + 16)) return *err = "model.wgcm damaged: index entry of " + grp.name + "/" + tn.name, false;
      uint64_t count = 1;
      bool count_ok = true;  // the element count never above the data's size (no wrap)
      for (uint32_t r = 0; r < rank; ++r) {
        const uint32_t dim = u32((size_t)pos + 4 * r);
        tn.shape.push_back(dim);
        if (dim && count > body / dim) count_ok = false;
        else count *= dim;
      }
      pos += 4ull * rank;
      const uint64_t off = u64((size_t)pos), size = u64((size_t)pos + 8);
      pos += 16;
      if (!count_ok || off % 64 || off < end || off > body || size > body - off || size != count * kDtypeSize[tn.dtype])
        return *err = "model.wgcm damaged: bad data range of " + grp.name + "/" + tn.name, false;
      if (grp.find(tn.name)) return *err = "model.wgcm damaged: duplicate tensor " + grp.name + "/" + tn.name, false;
      tn.data = d.data() + off;
      tn.size = size;
      hs.add(tn.data, size);
      grp.tensors.push_back(tn);
    }
    uint8_t got[32];
    if (!hs.finish(got)) return *err = "SHA-256 unavailable", false;
    if (memcmp(got, digest, 32) != 0)
      return *err = "model.wgcm: the data of group " + grp.name + " does not give its stored digest (57.7.3)", false;
    for (const WgcmGroup& o : *groups)
      if (o.name == grp.name) return *err = "model.wgcm damaged: duplicate group " + grp.name, false;
    groups->push_back(grp);
  }
  if (pos != end) return *err = "model.wgcm damaged: index size does not match its entries", false;
  return true;
}

}  // namespace wgrr_rt
