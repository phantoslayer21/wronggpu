// SPDX-License-Identifier: MIT
// Bounds checks of the RR runtime's model loaders (src/rr_model.cpp), CPU only, no GPU and no model data: crafted files with
// a correct whole-file SHA-256 (the hash is unkeyed, so it only catches accidents) whose offsets, sizes and shapes wrap
// around 2^64. Driven by test_model_bounds.py, one case per run:
//
//   test_model_bounds <case> <scratch folder>
//
// Prints "OK" or "FAIL <what>". A loader that reads through a wrapped range may also crash: that is a failure too.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/rr_model.h"

using namespace wgrr_rt;

// load_model refers to the packer of rr_pack.cpp, which needs the generated tables; the loaders under test do not use it.
namespace wgrr_rt {
bool pack_wgcm(const std::vector<WgcmGroup>&, Model*, std::string* err) {
  *err = "packing is not part of this test";
  return false;
}
}  // namespace wgrr_rt

typedef std::vector<uint8_t> Bytes;

static void put(Bytes& b, uint64_t v, int n) {
  for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i)));
}
static void put_at(Bytes& b, size_t at, uint64_t v, int n) {
  for (int i = 0; i < n; ++i) b[at + i] = (uint8_t)(v >> (8 * i));
}
static void pad_to(Bytes& b, size_t n) {
  while (b.size() < n) b.push_back(0);
}
static void put_str(Bytes& b, const std::string& s) {
  put(b, s.size(), 2);
  b.insert(b.end(), s.begin(), s.end());
}
static Bytes digest(const Bytes& data) {
  const std::string h = sha256_hex(data.data(), data.size());
  Bytes out;
  for (size_t i = 0; i + 1 < h.size(); i += 2) out.push_back((uint8_t)strtoul(h.substr(i, 2).c_str(), nullptr, 16));
  return out;
}
static void seal(Bytes& b) {  // the trailing whole-file hash
  const Bytes h = digest(b);
  b.insert(b.end(), h.begin(), h.end());
}

// ---- model.wgcm: header (88), index, data at 64-byte offsets, trailing hash ----------------------------------------
struct WgcmTensorSpec {
  uint8_t dtype;
  std::vector<uint32_t> dims;
  uint64_t size;
  bool at_data_start;   // offset = the first 64-aligned offset past the index
  uint64_t off;         // else this offset
  Bytes data;           // placed at the data start, when it is used
};

// One group "g" with one tensor "t". `own_digest`: the group digest is the right one for `data` (else zeros).
static Bytes make_wgcm(const WgcmTensorSpec& t) {
  Bytes key(32);
  for (int i = 0; i < 32; ++i) key[i] = (uint8_t)(0x40 + i);
  Bytes index;
  put_str(index, "g");
  Bytes keyed = key;
  keyed.insert(keyed.end(), t.data.begin(), t.data.end());
  const Bytes dg = digest(keyed);
  index.insert(index.end(), dg.begin(), dg.end());
  put(index, 1, 4);  // one tensor
  put_str(index, "t");
  put(index, t.dtype, 1);
  put(index, t.dims.size(), 1);
  for (uint32_t d : t.dims) put(index, d, 4);
  const uint64_t end = 88 + index.size() + 16;  // the index ends after this tensor's offset and size
  const uint64_t start = (end + 63) / 64 * 64;
  put(index, t.at_data_start ? start : t.off, 8);
  put(index, t.size, 8);
  Bytes f;
  const char magic[] = "WGCMODEL";
  f.insert(f.end(), magic, magic + 8);
  put(f, 2, 4);   // container version 2
  put(f, 1, 4);   // groups
  f.insert(f.end(), key.begin(), key.end());
  pad_to(f, 80);
  put(f, index.size(), 8);
  f.insert(f.end(), index.begin(), index.end());
  pad_to(f, (size_t)start);
  f.insert(f.end(), t.data.begin(), t.data.end());
  pad_to(f, (size_t)start + ((t.data.size() + 63) / 64 * 64));
  seal(f);
  return f;
}

// ---- WGRRMOD1: header (64), blob table (64 each), scalar table (52 each), image at a 256-byte offset, hash ----------
static Bytes own_header(uint32_t nb, uint32_t ns, uint64_t isize) {
  Bytes f;
  const char magic[] = "WGRRMOD1";
  f.insert(f.end(), magic, magic + 8);
  put(f, 1, 4);
  put(f, 0, 4);
  put(f, nb, 4);
  put(f, ns, 4);
  put(f, isize, 8);
  pad_to(f, 64);
  return f;
}
static void own_blob(Bytes& f, const char* name, uint64_t off, uint64_t size) {
  const size_t at = f.size();
  pad_to(f, at + 48);
  memcpy(&f[at], name, strlen(name));
  put(f, off, 8);
  put(f, size, 8);
}
static void own_scalar(Bytes& f, const char* name, uint32_t v) {
  const size_t at = f.size();
  pad_to(f, at + 48);
  memcpy(&f[at], name, strlen(name));
  put(f, v, 4);
}

static bool load_file(const std::wstring& dir, const Bytes& file, std::string* err, Model* m) {
  const std::wstring path = dir + L"\\model_bounds_case.bin";
  FILE* fp = _wfopen(path.c_str(), L"wb");
  if (!fp) {
    *err = "cannot write the scratch file";
    return false;
  }
  fwrite(file.data(), 1, file.size(), fp);
  fclose(fp);
  const bool ok = load_model(path, m, err);
  DeleteFileW(path.c_str());
  return ok;
}

static int fail(const std::string& what) {
  printf("FAIL %s\n", what.c_str());
  return 0;
}
static bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

int wmain(int argc, wchar_t** argv) {
  if (argc < 3) {
    printf("FAIL usage\n");
    return 2;
  }
  const std::wstring which = argv[1], dir = argv[2];
  std::string err;
  std::vector<WgcmGroup> groups;

  if (which == L"wgcm_valid") {
    // control: a well-formed file is still accepted (f32, shape 2x3, 24 bytes of data)
    WgcmTensorSpec t{0, {2, 3}, 24, true, 0, Bytes(24, 0x11)};
    if (!read_wgcm(make_wgcm(t), &groups, &err)) return fail("a valid file was refused: " + err);
    if (groups.size() != 1 || groups[0].tensors.size() != 1 || groups[0].tensors[0].size != 24) return fail("wrong content");
  } else if (which == L"wgcm_offset_wraps") {
    // off + size wraps around 2^64 to a value inside the file: the range must still be refused, before anything reads it
    WgcmTensorSpec t{1, {64}, 64, false, 0xFFFFFFFFFFFFFFC0ull, Bytes()};
    if (read_wgcm(make_wgcm(t), &groups, &err)) return fail("accepted a wrapped data range");
    if (!has(err, "bad data range")) return fail("wrong message: " + err);
  } else if (which == L"wgcm_size_past_end_after_wrap") {
    // the same with a size that is itself the product of the shape (a 2^32-1 square) and an offset above 2^33
    WgcmTensorSpec t{1, {0xFFFFFFFFu, 0xFFFFFFFFu}, 0xFFFFFFFE00000001ull, false, 1ull << 33, Bytes()};
    if (read_wgcm(make_wgcm(t), &groups, &err)) return fail("accepted a wrapped data range");
    if (!has(err, "bad data range")) return fail("wrong message: " + err);
  } else if (which == L"wgcm_shape_wraps") {
    // the shape's element count wraps around 2^64 (2^30 * 2^30 * 4 elements of 4 bytes = 0 bytes): refused
    WgcmTensorSpec t{0, {1u << 30, 1u << 30, 4}, 0, true, 0, Bytes()};
    if (read_wgcm(make_wgcm(t), &groups, &err)) return fail("accepted a shape whose element count wraps");
    if (!has(err, "bad data range")) return fail("wrong message: " + err);
  } else if (which == L"own_valid") {
    // control: a well-formed WGRRMOD1 with one blob and one scalar is still accepted
    Bytes f = own_header(1, 1, 256);
    own_blob(f, "a", 0, 256);
    own_scalar(f, "s", 7);
    pad_to(f, 256 + 256);
    seal(f);
    Model m;
    if (!load_file(dir, f, &err, &m)) return fail("a valid file was refused: " + err);
    if (m.image.size() != 256 || m.blob("a") != 0 || m.scalar("s") != 7) return fail("wrong content");
  } else if (which == L"own_blob_wraps") {
    // off + size wraps around 2^64 to a small value: the blob lies outside the image and must be refused
    Bytes f = own_header(1, 0, 256);
    own_blob(f, "a", 0xFFFFFFFFFFFFFFF0ull, 32);
    pad_to(f, 256 + 256);
    seal(f);
    Model m;
    if (load_file(dir, f, &err, &m)) return fail("accepted a blob whose range wraps");
    if (!has(err, "blob a")) return fail("wrong message: " + err);
  } else if (which == L"own_image_start_past_end") {
    // the tables end (rounded up to 256) behind the end of the file, and isize wraps so that offset + isize + 32 equals the
    // file size: the image's range would be reversed
    Bytes f = own_header(1, 3, 0xFFFFFFFFFFFFFF00ull);
    own_blob(f, "a", 0, 0);
    own_scalar(f, "s0", 0);
    own_scalar(f, "s1", 0);
    own_scalar(f, "s2", 0);
    f.resize(256);  // the file is 256 bytes and its hash: 288 in all (the scalar table runs into the hash, as it may)
    seal(f);
    Model m;
    if (load_file(dir, f, &err, &m)) return fail("accepted an image that starts behind the end of the file");
    if (!has(err, "image size")) return fail("wrong message: " + err);
  } else {
    return fail("unknown case");
  }
  printf("OK\n");
  return 0;
}
