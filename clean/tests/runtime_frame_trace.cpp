// SPDX-License-Identifier: MIT
// The runtime's frame sequence on the CPU, without a GPU: the runtime (clean/runtime/runtime.cpp) runs whole frames on
// a recording device (host memory; launches are recorded, not executed). For clean/tests/test_runtime_frame_trace.py
// and clean/l/runtime/tests/test_l_runtime_frame.py.
//
//   runtime_frame_trace <build dir> <model dir> <rw> <rh> <ow> <oh> <frames> [runtime options] [flags]
//
// rw x rh is the maximum render size. Flags:
//   --format=<n>      model_format (default 1: the converter's raw layout; 2: model.wgcm, with --map)
//   --map=<path>      the converter map (map_path)
//   --sizes=WxH,...   the render size of frame 0, 1, ... (repeated; default: the maximum every frame)
//   --reset=f,...     frames the game asks for a reset (WgRtFrame.reset = 1)
//   --fail=<kernel>@<frame>  that kernel's launch fails in that frame (the device returns false)
//
// Prints the create result and the allocations ("alloc b<k> <bytes>"), the code objects loaded ("load <file>[ as
// <tag>]"), then per frame "frame <f> index <i> parity <p> reset <r> rw <w> rh <h>" (wg_rt_info) and one line per
// launch: "launch <kernel> <grid> <block>", followed for the kernels of the table below by " | " and the decoded
// arguments: a device address as b<k>+<offset> (allocation k) or 0, an int as its value, the encoder and decoder
// parameters as E{rw,rh,fw,fh,frame,reset} / D{rw,rh,fw,fh,reset}. A failed frame prints what was launched up to the
// failure and "ERR frame <f>: <message>"; the next frame runs. "ERR <message>" at create. The code object files are not
// read: names in the build folder (empty files) select the runtime's optional paths as real code objects would. Inputs
// are zeros (the recording device does not compute): the sequence and the arguments are what is checked.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../runtime/device.h"
#include "../runtime/runtime_core.h"
#include "../runtime/wg_runtime.h"

using namespace wgrt;

namespace {

// The parameter structures the runtime passes (clean/runtime/runtime.cpp; kernels/encoder.hip WgEncParams,
// kernels/decoder.hip WgDecParams): only the fields printed are named.
struct EncParams {
  int rw, rh, fw, fh, hw, hh, frame, reset, depth_inverted, pad0;
  float f[19];
};
static_assert(sizeof(EncParams) == 116, "EncParams");
struct DecParams {
  int rw, rh, fw, fh, reset, pad0;
  float f[8];
};
static_assert(sizeof(DecParams) == 56, "DecParams");

// Argument kinds per kernel (name after any "tag/"): p = device address, i = int, E / D = the parameters above. A name
// ending in '*' matches every kernel that starts with it (only its first arguments, which all forms share).
struct Signature {
  const char* name;
  const char* kinds;
};
const Signature kSignatures[] = {
    {"input_encoder", "ppppppppppppE"},     // Model M's encoder and Model L's values (q in place of the features)
    {"coef_decoder", "ppppppppD"},          // Model M's decoder (also its pack12 form)
    {"swin_block_*", "ppppiiii"},           // Model M's blocks: x, skip, out, w, H, W, ox, oy
    {"l_swin_block_c32*", "ppppiiii"},      // Model L's blocks (clean/l/FORMATS.md)
    {"l_linear_32_32_relu", "pppi"},
    {"l_linear_32_32_f16", "pppi"},
    {"l_project_gather", "pppii"},
    {"l_dec_project", "pppii"},
    {"l_coef", "ppppppD"},
};

const char* kinds_of(const std::string& kernel) {
  const size_t slash = kernel.find('/');
  const std::string k = slash == std::string::npos ? kernel : kernel.substr(slash + 1);
  // dec1's block fused with the decoder (swin_block_c64_add_dec0*): x, skip, w, H, W, ox, oy, then the decoder's
  // arguments: not the blocks' common form
  if (k.find("_add_dec0") != std::string::npos) return "pppiiii";
  for (const Signature& s : kSignatures) {
    const size_t n = std::strlen(s.name);
    if (s.name[n - 1] == '*' ? k.compare(0, n - 1, s.name, n - 1) == 0 : k == s.name) return s.kinds;
  }
  return nullptr;
}

class RecordingDevice : public Device {
 public:
  std::vector<std::string>* log;
  std::map<DevPtr, std::vector<uint8_t>> mem;
  std::map<DevPtr, int> index;  // allocation number of each block
  int allocs = 0, frame = -1, fail_frame = -1;
  std::string fail_kernel;
  DevPtr next = 0x10000;
  explicit RecordingDevice(std::vector<std::string>* l) : log(l) {}
  std::string name() const override { return "recording device (CPU test)"; }
  bool alloc(DevPtr* p, size_t bytes) override {
    *p = next;
    mem[next].assign(bytes ? bytes : 1, 0);
    index[next] = allocs;
    log->push_back("alloc b" + std::to_string(allocs++) + " " + std::to_string(bytes));
    next += (bytes + 0xFFFF) / 0x10000 * 0x10000 + 0x10000;
    return true;
  }
  void free(DevPtr p) override {
    mem.erase(p);
    index.erase(p);
  }
  uint8_t* at(DevPtr p, size_t n) {
    auto it = mem.upper_bound(p);
    if (it == mem.begin()) return nullptr;
    --it;
    if (p - it->first + n > it->second.size()) return nullptr;
    return it->second.data() + (p - it->first);
  }
  std::string addr(DevPtr p) {
    if (!p) return "0";
    auto it = mem.upper_bound(p);
    if (it == mem.begin()) return "?" + std::to_string(p);
    --it;
    if (p - it->first >= it->second.size()) return "?" + std::to_string(p);
    return "b" + std::to_string(index[it->first]) + "+" + std::to_string(p - it->first);
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
  bool load_module_tagged(const std::string& path, const std::string& tag) override {
    if (!load_module(path)) return false;
    log->back() += " as " + tag;
    return true;
  }
  bool launch(const std::string& kernel, unsigned grid, unsigned block, void** args) override {
    if (frame == fail_frame && kernel == fail_kernel) {
      err_ = "injected launch failure: " + kernel;
      log->push_back("failed " + kernel);
      return false;
    }
    std::string line = "launch " + kernel + " " + std::to_string(grid) + " " + std::to_string(block);
    if (const char* kinds = kinds_of(kernel)) {
      line += " |";
      for (size_t i = 0; kinds[i]; ++i) {
        line += " ";
        switch (kinds[i]) {
          case 'p': line += addr(*static_cast<const DevPtr*>(args[i])); break;
          case 'i': line += std::to_string(*static_cast<const int*>(args[i])); break;
          case 'E': {
            const EncParams& e = *static_cast<const EncParams*>(args[i]);
            line += "E{" + std::to_string(e.rw) + "," + std::to_string(e.rh) + "," + std::to_string(e.fw) + "," +
                    std::to_string(e.fh) + "," + std::to_string(e.frame) + "," + std::to_string(e.reset) + "}";
            break;
          }
          case 'D': {
            const DecParams& d = *static_cast<const DecParams*>(args[i]);
            line += "D{" + std::to_string(d.rw) + "," + std::to_string(d.rh) + "," + std::to_string(d.fw) + "," +
                    std::to_string(d.fh) + "," + std::to_string(d.reset) + "}";
            break;
          }
        }
      }
    }
    log->push_back(line);
    return true;
  }
  bool sync() override { return true; }
  bool record(int) override { return true; }
  bool elapsed_ms(int, int, float* ms) override { return *ms = 0.0f, true; }
  bool reserve_events(int) override { return true; }
  int event_done(int) override { return 1; }
  bool elapsed_done_ms(int, int, float* ms) override { return *ms = 0.0f, true; }
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  for (size_t a = 0; a <= s.size();) {
    size_t b = s.find(sep, a);
    if (b == std::string::npos) b = s.size();
    if (b > a) out.push_back(s.substr(a, b - a));
    a = b + 1;
  }
  return out;
}

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
  const int frames = std::atoi(argv[7]);
  std::string options, map_path, fail;
  std::vector<std::pair<int, int>> sizes;
  std::set<int> resets;
  for (int a = 8; a < argc; ++a) {
    const std::string s = argv[a];
    if (s.compare(0, 9, "--format=") == 0) {
      cfg.model_format = std::atoi(s.c_str() + 9);
    } else if (s.compare(0, 6, "--map=") == 0) {
      map_path = s.substr(6);
    } else if (s.compare(0, 8, "--sizes=") == 0) {
      for (const std::string& wh : split(s.substr(8), ',')) {
        const size_t x = wh.find('x');
        if (x == std::string::npos) return std::printf("ERR bad size %s\n", wh.c_str()), 2;
        sizes.emplace_back(std::atoi(wh.c_str()), std::atoi(wh.c_str() + x + 1));
      }
    } else if (s.compare(0, 8, "--reset=") == 0) {
      for (const std::string& f : split(s.substr(8), ',')) resets.insert(std::atoi(f.c_str()));
    } else if (s.compare(0, 7, "--fail=") == 0) {
      fail = s.substr(7);
    } else if (s.compare(0, 2, "--") == 0) {
      return std::printf("ERR unknown flag %s\n", s.c_str()), 2;
    } else {
      options = s;
    }
  }
  cfg.runtime_options = options.c_str();
  cfg.map_path = map_path.empty() ? nullptr : map_path.c_str();
  auto dev = std::make_unique<RecordingDevice>(&log);
  RecordingDevice* rec = dev.get();
  if (!fail.empty()) {
    const size_t at = fail.find('@');
    rec->fail_kernel = fail.substr(0, at);
    rec->fail_frame = at == std::string::npos ? 0 : std::atoi(fail.c_str() + at + 1);
  }
  std::string err;
  Runtime* rt = runtime_create(cfg, std::move(dev), &err);
  if (!rt) return std::printf("ERR create: %s\n", err.c_str()), 0;
  std::printf("create ok model %s\n", runtime_model(rt));
  for (const std::string& l : log) std::printf("%s\n", l.c_str());
  log.clear();
  const size_t px = (size_t)cfg.max_rw * cfg.max_rh;
  std::vector<float> color(px * 3, 0.0f), depth(px, 0.0f), motion(px * 2, 0.0f);
  for (int f = 0; f < frames; ++f) {
    const int rw = sizes.empty() ? cfg.max_rw : sizes[f % sizes.size()].first;
    const int rh = sizes.empty() ? cfg.max_rh : sizes[f % sizes.size()].second;
    rec->frame = f;
    if (wg_rt_upload_inputs(rt, rw, rh, color.data(), depth.data(), motion.data()) != 0)
      return std::printf("ERR upload: %s\n", runtime_error(rt).c_str()), 0;
    WgRtFrame fr = WgRtFrame();
    fr.rw = rw;
    fr.rh = rh;
    fr.mv_scale[0] = fr.mv_scale[1] = 1.0f;
    fr.exposure = 1.0f;
    fr.pre_exposure = 1.0f;
    fr.reset = resets.count(f) ? 1 : 0;
    const bool ok = runtime_frame(rt, fr, nullptr);
    WgRtFrameInfo info = WgRtFrameInfo();
    wg_rt_info(rt, &info);
    std::printf("frame %d index %d parity %d reset %d rw %d rh %d\n", f, info.index, info.parity, info.reset, info.rw,
                info.rh);
    for (const std::string& l : log) std::printf("%s\n", l.c_str());
    log.clear();
    if (!ok) std::printf("ERR frame %d: %s\n", f, runtime_error(rt).c_str());
  }
  std::printf("kernels %s\n", runtime_kernels(rt));
  runtime_destroy(rt);
  return 0;
}
