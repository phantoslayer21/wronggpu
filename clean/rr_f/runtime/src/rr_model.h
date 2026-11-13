// SPDX-License-Identifier: MIT
// The runtime's model: a device image in the kernels' buffer layouts plus a name -> (offset, size) table and a
// name -> 32-bit scalar table. Two sources:
//   * the converter's output `model.wgcm` (clean/converter/OUTPUT-FORMAT.md, container version 2, map version 5:
//     the logical tensors of page 56.4 and the symbol table of 57.7.2), checked (whole-file hash, index bounds,
//     the keyed digest of every group 57.7.3, the 56.4 / 57.7 inventory) and packed here into the kernels' layouts
//     (src/rr_pack.cpp: the same bytes as tools/pack_model.py pack() of the same logical model);
//   * the runtime's own pre-packed container `WGRRMOD1` (tools/pack_model.py; synthetic test models).
#pragma once
#include <stdint.h>

#include <map>
#include <string>
#include <vector>

namespace wgrr_rt {

struct Model {
  bool ok = false;
  uint32_t version = 0, flags = 0;   // flags: bit 0 constants_b2 (build B2), bit 1 synthetic, bit 2 admission subset
  std::string kind;                  // "wgcm" (converter output) or "wgrrmod1" (own container)
  std::vector<uint8_t> image;                                     // device image (multiple of 256 bytes)
  std::map<std::string, std::pair<uint64_t, uint64_t>> blobs;     // offset in the image, size
  std::map<std::string, uint32_t> scalars;
  std::string file_sha256, logical_sha256, path;
  uint32_t rho_guarded = 0;          // wgcm: rho entries the binary64 tanh did not decide (exact path used)

  bool b2() const { return (flags & 1u) != 0; }
  bool synthetic() const { return (flags & 2u) != 0; }
  // offset of a blob in the image, or UINT64_MAX
  uint64_t blob(const std::string& n) const {
    auto it = blobs.find(n);
    return it == blobs.end() ? UINT64_MAX : it->second.first;
  }
  bool has_scalar(const std::string& n) const { return scalars.count(n) != 0; }
  uint32_t scalar(const std::string& n) const {
    auto it = scalars.find(n);
    return it == scalars.end() ? 0u : it->second;
  }
};

// Loads either container (by its magic).
bool load_model(const std::wstring& path, Model* out, std::string* err);
std::string sha256_hex(const uint8_t* data, size_t n);

// ---- converter output (rr_pack.cpp)
struct WgcmTensor {
  std::string name;
  uint32_t dtype = 0;               // 0 f32, 1 u8, 2 i32, 3 f16, 4 u16, 5 u32, 6 e4m3
  std::vector<uint32_t> shape;
  const uint8_t* data = nullptr;
  uint64_t size = 0;
};
struct WgcmGroup {
  std::string name;
  std::vector<WgcmTensor> tensors;   // file order
  const WgcmTensor* find(const std::string& n) const {
    for (const WgcmTensor& t : tensors)
      if (t.name == n) return &t;
    return nullptr;
  }
};
// Parses and checks a model.wgcm held in `file` (57.7.3 read-back rule: every group's keyed digest; inventory of
// 56.4 / 57.7); fills the groups in file order.
bool read_wgcm(const std::vector<uint8_t>& file, std::vector<WgcmGroup>* groups, std::string* err);
// Packs the checked groups into the device image / tables (the layouts of tools/pack_model.py pack()).
bool pack_wgcm(const std::vector<WgcmGroup>& groups, Model* out, std::string* err);

// rho of page 55.3: correctly rounded binary32 tanh(fl32(v * s2)) for every binary16 bit pattern v; returns the
// number of entries the binary64 evaluation did not decide (computed with the double-double path). force_exact:
// every entry through the double-double path (test hook).
uint32_t head_rho(float s2, float* out65536, bool force_exact = false);

}  // namespace wgrr_rt
