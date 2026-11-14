// SPDX-License-Identifier: MIT
// The runtime's C++ packers (clean/runtime/model.cpp) on the CPU, for clean/tests/test_model_pack_cpp.py:
//
//   model_pack_dump stages <model dir> <out dir>
//       loads the converter's raw tensors (load_raw_model) and writes <out dir>/<stage>.bin = pack_stage_blob for
//       every stage with a blob (enc0, enc1 ... dec1, dec0); prints "OK <n>" or "ERR <message>"
//   model_pack_dump block <tensor dir> <prefix> <C> <heads> <mlp> <out file>
//       reads <tensor dir>/<name>.bin (float32, the file name is the tensor name, e.g. "blk.W1.bin"), packs one Swin
//       block section with pack_parts::pack_swin_block and writes it; prints "OK <bytes>" or "ERR <message>"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../runtime/model.h"

namespace {

bool write_file(const std::string& path, const std::vector<uint8_t>& b) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
  return (bool)f;
}

bool read_f32(const std::string& path, std::vector<float>* out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize n = f.tellg();
  if (n % 4) return false;
  out->resize((size_t)n / 4);
  f.seekg(0);
  return (bool)f.read(reinterpret_cast<char*>(out->data()), n);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  const wgrt::DRow d = wgrt::drow_gfx12();
  std::string err;
  if (mode == "stages" && argc == 4) {
    std::map<std::string, wgrt::StageTensors> model;
    if (!wgrt::load_raw_model(argv[2], &model, &err)) return std::printf("ERR %s\n", err.c_str()), 0;
    int n = 0;
    for (const std::string& s : wgrt::model_stages()) {
      if (s == "post") continue;  // no blob of its own
      std::vector<uint8_t> blob;
      if (!wgrt::pack_stage_blob(s, model.at(s), d, &blob, &err)) return std::printf("ERR %s\n", err.c_str()), 0;
      if (!write_file(std::string(argv[3]) + "/" + s + ".bin", blob)) return std::printf("ERR write %s\n", s.c_str()), 0;
      ++n;
    }
    std::printf("OK %d\n", n);
    return 0;
  }
  if (mode == "block" && argc == 8) {
    const std::string dir = argv[2], prefix = argv[3];
    const char* names[] = {"gamma1", "gamma2", "Wqk", "Wv", "B", "Wp", "bp", "W1", "b1", "W2", "b2"};
    wgrt::StageTensors t;
    for (const char* nm : names) {
      wgrt::Tensor x;
      if (read_f32(dir + "/" + prefix + nm + ".bin", &x.f)) t[prefix + nm] = x;  // a missing file: the packer says so
    }
    std::vector<uint8_t> blob;
    if (!wgrt::pack_parts::pack_swin_block(t, prefix, std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]), d,
                                           &blob, &err))
      return std::printf("ERR %s\n", err.c_str()), 0;
    if (!write_file(argv[7], blob)) return std::printf("ERR write\n"), 0;
    std::printf("OK %zu\n", blob.size());
    return 0;
  }
  std::printf("ERR usage\n");
  return 2;
}
