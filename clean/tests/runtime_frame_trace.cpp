// SPDX-License-Identifier: MIT
// The runtime's frame sequence on the CPU, without a GPU: the runtime (clean/runtime/runtime.cpp) runs whole frames on
// a recording device (host memory; launches are recorded, not executed). For clean/tests/test_runtime_frame_trace.py.
//
//   runtime_frame_trace <build dir> <model dir> <rw> <rh> <ow> <oh> <frames> [runtime options]
//
// The model is read in the converter's raw layout (model_format 1). Prints the create result, then one line per
// device event of every frame: "load <code object file>", "launch <kernel> <grid> <block>"; "ERR <message>" on a
// failure. Inputs are zeros (the recording device does not compute), so only the sequence is meaningful.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../runtime/device.h"
#include "../runtime/runtime_core.h"
#include "../runtime/wg_runtime.h"

using namespace wgrt;

namespace {

class RecordingDevice : public Device {
 public:
  std::vector<std::string>* log;
  std::map<DevPtr, std::vector<uint8_t>> mem;
  DevPtr next = 0x10000;
  explicit RecordingDevice(std::vector<std::string>* l) : log(l) {}
  std::string name() const override { return "recording device (CPU test)"; }
  bool alloc(DevPtr* p, size_t bytes) override {
    *p = next;
    mem[next].assign(bytes ? bytes : 1, 0);
    next += (bytes + 0xFFFF) / 0x10000 * 0x10000 + 0x10000;
    return true;
  }
  void free(DevPtr p) override { mem.erase(p); }
  uint8_t* at(DevPtr p, size_t n) {
    auto it = mem.upper_bound(p);
    if (it == mem.begin()) return nullptr;
    --it;
    if (p - it->first + n > it->second.size()) return nullptr;
    return it->second.data() + (p - it->first);
  }
  bool upload(DevPtr dst, const void* src, size_t n) override {
    uint8_t* d = at(dst, n);
    if (d && n) std::memcpy(d, src, n);
    return d != nullptr;
  }
  bool download(void* dst, DevPtr src, size_t n) override {
    uint8_t* s = at(src, n);
    if (s && n) std::memcpy(dst, s, n);
    return s != nullptr;
  }
  bool copy(DevPtr dst, DevPtr src, size_t n) override {
    uint8_t *d = at(dst, n), *s = at(src, n);
    if (d && s && n) std::memmove(d, s, n);
    return d && s;
  }
  bool memset(DevPtr dst, int v, size_t n) override {
    uint8_t* d = at(dst, n);
    if (d && n) std::memset(d, v, n);
    return d != nullptr;
  }
  bool load_module(const std::string& path) override {
    const size_t s = path.find_last_of("\\/");
    log->push_back("load " + (s == std::string::npos ? path : path.substr(s + 1)));
    return true;
  }
  bool launch(const std::string& kernel, unsigned grid, unsigned block, void**) override {
    log->push_back("launch " + kernel + " " + std::to_string(grid) + " " + std::to_string(block));
    return true;
  }
  bool sync() override { return true; }
  bool record(int) override { return true; }
  bool elapsed_ms(int, int, float* ms) override { return *ms = 0.0f, true; }
  bool reserve_events(int) override { return true; }
  int event_done(int) override { return 1; }
  bool elapsed_done_ms(int, int, float* ms) override { return *ms = 0.0f, true; }
};

}  // namespace

// The device backends are not linked (runtime_create gets the recording device); the runtime still names them.
namespace wgrt {
#if defined(_WIN32)
std::unique_ptr<Device> make_dx12_device(const std::string&, const std::string&, std::string* err) {
  return *err = "not linked in this test", nullptr;
}
#endif
std::unique_ptr<Device> make_d3d12_device(const std::string&, std::string* err) {
  return *err = "not linked in this test", nullptr;
}
}  // namespace wgrt

int main(int argc, char** argv) {
  if (argc < 8) return std::printf("ERR usage\n"), 2;
  std::vector<std::string> log;
  WgRtConfig cfg = WgRtConfig();
  cfg.build_dir = argv[1];
  cfg.model_dir = argv[2];
  cfg.backend = 0;
  cfg.model_format = 1;  // raw
  cfg.max_rw = std::atoi(argv[3]);
  cfg.max_rh = std::atoi(argv[4]);
  cfg.ow = std::atoi(argv[5]);
  cfg.oh = std::atoi(argv[6]);
  cfg.depth_inverted = 1;
  cfg.auto_exposure = 0;
  cfg.runtime_options = argc > 8 ? argv[8] : "";
  const int frames = std::atoi(argv[7]);
  std::string err;
  Runtime* rt = runtime_create(cfg, std::make_unique<RecordingDevice>(&log), &err);
  if (!rt) return std::printf("ERR create: %s\n", err.c_str()), 0;
  std::printf("create ok model %s\n", runtime_model(rt));
  for (const std::string& l : log) std::printf("%s\n", l.c_str());
  log.clear();
  const size_t px = (size_t)cfg.max_rw * cfg.max_rh;
  std::vector<float> color(px * 3, 0.0f), depth(px, 0.0f), motion(px * 2, 0.0f);
  for (int f = 0; f < frames; ++f) {
    if (wg_rt_upload_inputs(rt, cfg.max_rw, cfg.max_rh, color.data(), depth.data(), motion.data()) != 0)
      return std::printf("ERR upload: %s\n", runtime_error(rt).c_str()), 0;
    WgRtFrame fr = WgRtFrame();
    fr.rw = cfg.max_rw;
    fr.rh = cfg.max_rh;
    fr.mv_scale[0] = fr.mv_scale[1] = 1.0f;
    fr.exposure = 1.0f;
    fr.pre_exposure = 1.0f;
    if (!runtime_frame(rt, fr, nullptr)) return std::printf("ERR frame %d: %s\n", f, runtime_error(rt).c_str()), 0;
    std::printf("frame %d\n", f);
    for (const std::string& l : log) std::printf("%s\n", l.c_str());
    log.clear();
  }
  std::printf("kernels %s\n", runtime_kernels(rt));
  runtime_destroy(rt);
  return 0;
}
