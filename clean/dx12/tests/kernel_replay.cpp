// SPDX-License-Identifier: MIT
// Kernel replay on the native D3D12 Device (issue #89): runs single kernels by their launch names on synthetic inputs
// given in files, so that a D3D12 port can be compared byte for byte with its HIP kernel on the same inputs (driver:
// clean/l/runner/l_d3d12_check.py). Uses only the project's Device API (clean/runtime/device.h).
//
//   kernel_replay <build dir with dx12/manifest.json> <cases.txt>
//
// cases.txt, one case per block of lines (blank line between cases; paths relative to the cases file):
//   case <name>
//   module <code object name> [<tag>]          (zero or more)
//   kernel <launch name> <grid> <block>
//   arg buf <file>                             device buffer with the file's bytes
//   arg out <bytes> <file>                     device buffer of that size (zero-filled), written to <file> afterwards
//   arg null                                   null pointer
//   arg u32 <value>                            32-bit scalar (decimal, may be negative)
//   arg bytes <file>                           a struct passed by value (the file's bytes)
#include "../../runtime/device.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace wgrt;

namespace {

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}

bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());
  return (bool)f;
}

struct Arg {
  std::string kind, file;
  size_t bytes = 0;
  int64_t value = 0;
  DevPtr ptr = 0;
  uint32_t u = 0;
  std::vector<uint8_t> blob;
};

struct Case {
  std::string name, kernel;
  unsigned grid = 0, block = 0;
  std::vector<std::pair<std::string, std::string>> modules;
  std::vector<Arg> args;
};

std::string dir_of(const std::string& path) {
  const size_t s = path.find_last_of("/\\");
  return s == std::string::npos ? std::string(".") : path.substr(0, s);
}

int run_case(Device* d, Case& c, const std::string& base) {
  for (auto& m : c.modules) {
    const bool ok = m.second.empty() ? d->load_module(m.first) : d->load_module_tagged(m.first, m.second);
    if (!ok) { std::printf("%s: module %s: %s\n", c.name.c_str(), m.first.c_str(), d->err().c_str()); return 1; }
  }
  std::vector<void*> argv;
  std::vector<DevPtr> owned;
  for (auto& a : c.args) {
    if (a.kind == "buf") {
      std::vector<uint8_t> data;
      if (!read_file(base + "/" + a.file, &data)) { std::printf("%s: cannot read %s\n", c.name.c_str(), a.file.c_str()); return 1; }
      if (!d->alloc(&a.ptr, data.size() ? data.size() : 4) || !d->upload(a.ptr, data.data(), data.size())) {
        std::printf("%s: buffer %s: %s\n", c.name.c_str(), a.file.c_str(), d->err().c_str());
        return 1;
      }
      owned.push_back(a.ptr);
      argv.push_back(&a.ptr);
    } else if (a.kind == "out") {
      if (!d->alloc(&a.ptr, a.bytes) || !d->memset(a.ptr, 0, a.bytes)) {
        std::printf("%s: output: %s\n", c.name.c_str(), d->err().c_str());
        return 1;
      }
      owned.push_back(a.ptr);
      argv.push_back(&a.ptr);
    } else if (a.kind == "null") {
      a.ptr = 0;
      argv.push_back(&a.ptr);
    } else if (a.kind == "u32") {
      a.u = (uint32_t)a.value;
      argv.push_back(&a.u);
    } else if (a.kind == "bytes") {
      if (!read_file(base + "/" + a.file, &a.blob)) { std::printf("%s: cannot read %s\n", c.name.c_str(), a.file.c_str()); return 1; }
      argv.push_back(a.blob.data());
    }
  }
  if (!d->launch(c.kernel, c.grid, c.block, argv.data())) {
    std::printf("%s: launch %s: %s\n", c.name.c_str(), c.kernel.c_str(), d->err().c_str());
    return 1;
  }
  if (!d->sync()) { std::printf("%s: sync: %s\n", c.name.c_str(), d->err().c_str()); return 1; }
  for (auto& a : c.args) {
    if (a.kind != "out") continue;
    std::vector<uint8_t> data(a.bytes);
    if (!d->download(data.data(), a.ptr, a.bytes) || !write_file(base + "/" + a.file, data)) {
      std::printf("%s: download %s: %s\n", c.name.c_str(), a.file.c_str(), d->err().c_str());
      return 1;
    }
  }
  for (DevPtr p : owned) d->free(p);
  std::printf("%s: ok (%s)\n", c.name.c_str(), c.kernel.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) { std::printf("usage: kernel_replay <build dir> <cases.txt>\n"); return 2; }
  std::string err;
  auto d = make_d3d12_device(argv[1], &err);
  if (!d) { std::printf("device: %s\n", err.c_str()); return 1; }
  std::printf("device: %s\n", d->name().c_str());
  std::ifstream f(argv[2]);
  if (!f) { std::printf("cannot read %s\n", argv[2]); return 2; }
  const std::string base = dir_of(argv[2]);
  std::vector<Case> cases;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream s(line);
    std::string w;
    if (!(s >> w)) continue;
    if (w == "case") { cases.emplace_back(); s >> cases.back().name; continue; }
    if (cases.empty()) { std::printf("line before the first case: %s\n", line.c_str()); return 2; }
    Case& c = cases.back();
    if (w == "module") {
      std::string m, tag;
      s >> m >> tag;
      c.modules.emplace_back(m, tag);
    } else if (w == "kernel") {
      s >> c.kernel >> c.grid >> c.block;
    } else if (w == "arg") {
      Arg a;
      s >> a.kind;
      if (a.kind == "buf" || a.kind == "bytes") s >> a.file;
      else if (a.kind == "out") s >> a.bytes >> a.file;
      else if (a.kind == "u32") s >> a.value;
      else if (a.kind != "null") { std::printf("unknown arg kind %s\n", a.kind.c_str()); return 2; }
      c.args.push_back(a);
    } else {
      std::printf("unknown line: %s\n", line.c_str());
      return 2;
    }
  }
  int fails = 0;
  for (auto& c : cases) fails += run_case(d.get(), c, base);
  if (!d->sync()) { std::printf("final sync: %s\n", d->err().c_str()); return 1; }
  std::printf("kernel_replay: %d of %zu cases failed\n", fails, cases.size());
  return fails ? 1 : 0;
}
