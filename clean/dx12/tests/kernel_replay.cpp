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
//   arg out <bytes> <file> [fill]              device buffer of that size, every byte set to fill (0..255, default 0)
//                                              before the launch, written to <file> afterwards
//   arg null                                   null pointer
//   arg u32 <value>                            32-bit scalar (decimal, may be negative)
//   arg bytes <file>                           a struct passed by value (the file's bytes)
//   direct <color file> <rw> <rh> <ow> <oh> <output file> <fill u32>
//                                              a direct frame for an rs=direct port (bind_direct_frame): color = an
//                                              RGBA32F texture rw x rh from the file (16 bytes per texel, rows packed),
//                                              depth (R32F) and motion (R16G16F) textures of zeros, no exposure texture,
//                                              color_pi 1, output = an RGBA32F UAV texture ow x oh whose every 32-bit
//                                              word starts as <fill>; after the launch the output texture's texels are
//                                              written to <output file> (16 bytes per texel, rows packed)
#include "../../runtime/device.h"
#include "../../runtime/device_d3d12.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

struct Direct {
  bool on = false;
  std::string color_file, out_file;
  unsigned rw = 0, rh = 0, ow = 0, oh = 0;
  uint32_t fill = 0;
};

struct Case {
  std::string name, kernel;
  unsigned grid = 0, block = 0;
  std::vector<std::pair<std::string, std::string>> modules;
  std::vector<Arg> args;
  Direct direct;
};

// ---- direct frames (rs=direct ports): the textures, their upload and the output's readback -------------------------
template <class T>
struct Com {
  T* p = nullptr;
  ~Com() { if (p) p->Release(); }
  T** pp() { return &p; }
};

struct DirectFrame {
  Com<ID3D12Resource> color, depth, motion, output, upload, readback;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT out_fp{};
};

bool make_texture(ID3D12Device* dev, DXGI_FORMAT format, unsigned w, unsigned h, bool uav, ID3D12Resource** out) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = w;
  desc.Height = h;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = format;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
  return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, __uuidof(ID3D12Resource), (void**)out));
}

bool make_buffer(ID3D12Device* dev, D3D12_HEAP_TYPE type, uint64_t bytes, ID3D12Resource** out) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = type;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = bytes;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  const D3D12_RESOURCE_STATES state =
      type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST;
  return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                __uuidof(ID3D12Resource), (void**)out));
}

void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  D3D12_RESOURCE_BARRIER x{};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition.pResource = r;
  x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  x.Transition.StateBefore = a;
  x.Transition.StateAfter = b;
  l->ResourceBarrier(1, &x);
}

// Creates the textures, uploads color / zeros / the output fill, binds the frame. The inputs end in
// NON_PIXEL_SHADER_RESOURCE, the output in UNORDERED_ACCESS.
bool direct_begin(NativeD3D12Ext* ext, const Direct& dr, const std::vector<uint8_t>& color, DirectFrame& f,
                  std::string& err) {
  ID3D12Device* dev = nullptr;
  if (FAILED(ext->arena_resource()->GetDevice(__uuidof(ID3D12Device), (void**)&dev))) return err = "GetDevice", false;
  Com<ID3D12Device> dev_ref;
  dev_ref.p = dev;
  if (color.size() != (size_t)dr.rw * dr.rh * 16) return err = "color file size", false;
  if (!make_texture(dev, DXGI_FORMAT_R32G32B32A32_FLOAT, dr.rw, dr.rh, false, f.color.pp()) ||
      !make_texture(dev, DXGI_FORMAT_R32_FLOAT, dr.rw, dr.rh, false, f.depth.pp()) ||
      !make_texture(dev, DXGI_FORMAT_R16G16_FLOAT, dr.rw, dr.rh, false, f.motion.pp()) ||
      !make_texture(dev, DXGI_FORMAT_R32G32B32A32_FLOAT, dr.ow, dr.oh, true, f.output.pp()))
    return err = "texture creation", false;
  // one upload buffer: the four textures' footprints one after another
  ID3D12Resource* tex[4] = {f.color.p, f.depth.p, f.motion.p, f.output.p};
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[4];
  UINT rows[4];
  UINT64 row_bytes[4], total = 0;
  for (int i = 0; i < 4; ++i) {
    const D3D12_RESOURCE_DESC d = tex[i]->GetDesc();
    UINT64 bytes = 0;
    dev->GetCopyableFootprints(&d, 0, 1, total, &fp[i], &rows[i], &row_bytes[i], &bytes);
    total = (fp[i].Offset + bytes + 511) & ~(UINT64)511;
  }
  f.out_fp = fp[3];
  f.out_fp.Offset = 0;
  if (!make_buffer(dev, D3D12_HEAP_TYPE_UPLOAD, total, f.upload.pp())) return err = "upload buffer", false;
  uint8_t* map = nullptr;
  D3D12_RANGE none{0, 0};
  if (FAILED(f.upload.p->Map(0, &none, (void**)&map))) return err = "Map(upload)", false;
  std::memset(map, 0, (size_t)total);
  for (UINT y = 0; y < rows[0]; ++y)
    std::memcpy(map + fp[0].Offset + (size_t)y * fp[0].Footprint.RowPitch, color.data() + (size_t)y * dr.rw * 16,
                (size_t)dr.rw * 16);
  for (UINT y = 0; y < rows[3]; ++y) {
    uint32_t* row = (uint32_t*)(map + fp[3].Offset + (size_t)y * fp[3].Footprint.RowPitch);
    for (unsigned x = 0; x < dr.ow * 4; ++x) row[x] = dr.fill;
  }
  f.upload.p->Unmap(0, nullptr);
  const UINT64 readback_bytes = (UINT64)f.out_fp.Footprint.RowPitch * dr.oh;
  if (!make_buffer(dev, D3D12_HEAP_TYPE_READBACK, readback_bytes, f.readback.pp())) return err = "readback buffer", false;
  ID3D12Resource* up = f.upload.p;
  if (!ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, [&](ID3D12GraphicsCommandList* l, std::string&) {
        for (int i = 0; i < 4; ++i) {
          D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
          dst.pResource = tex[i];
          dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          src.pResource = up;
          src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
          src.PlacedFootprint = fp[i];
          l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
          barrier(l, tex[i], D3D12_RESOURCE_STATE_COPY_DEST,
                  i == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        return true;
      }, &err, "replay_direct_upload"))
    return false;
  NativeDirectFrame frame;
  frame.color = f.color.p;
  frame.depth = f.depth.p;
  frame.motion = f.motion.p;
  frame.output = f.output.p;
  frame.color_format = DXGI_FORMAT_R32G32B32A32_FLOAT;
  frame.depth_format = DXGI_FORMAT_R32_FLOAT;
  frame.motion_format = DXGI_FORMAT_R16G16_FLOAT;
  frame.output_format = DXGI_FORMAT_R32G32B32A32_FLOAT;
  frame.params.rw = dr.rw;
  frame.params.rh = dr.rh;
  return ext->bind_direct_frame(frame, &err);
}

// Ends the frame and reads the output texture back (rows packed, 16 bytes per texel).
bool direct_end(NativeD3D12Ext* ext, const Direct& dr, DirectFrame& f, std::vector<uint8_t>& out, std::string& err) {
  if (!ext->end_direct_frame(&err)) return false;
  ID3D12Resource* o = f.output.p;
  ID3D12Resource* rb = f.readback.p;
  const D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = f.out_fp;
  if (!ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, [&](ID3D12GraphicsCommandList* l, std::string&) {
        barrier(l, o, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = rb;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp;
        src.pResource = o;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        return true;
      }, &err, "replay_direct_readback"))
    return false;
  if (!ext->sync(&err)) return false;
  uint8_t* map = nullptr;
  D3D12_RANGE all{0, (SIZE_T)fp.Footprint.RowPitch * dr.oh};
  if (FAILED(rb->Map(0, &all, (void**)&map))) return err = "Map(readback)", false;
  out.resize((size_t)dr.ow * dr.oh * 16);
  for (unsigned y = 0; y < dr.oh; ++y)
    std::memcpy(out.data() + (size_t)y * dr.ow * 16, map + (size_t)y * fp.Footprint.RowPitch, (size_t)dr.ow * 16);
  D3D12_RANGE none{0, 0};
  rb->Unmap(0, &none);
  return true;
}

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
      if (!d->alloc(&a.ptr, a.bytes) || !d->memset(a.ptr, (int)(a.value & 0xff), a.bytes)) {
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
  DirectFrame frame;
  NativeD3D12Ext* ext = c.direct.on ? d3d12_ext(d) : nullptr;
  if (c.direct.on) {
    std::vector<uint8_t> color;
    std::string e;
    if (!ext) { std::printf("%s: direct: not a native D3D12 device\n", c.name.c_str()); return 1; }
    if (!read_file(base + "/" + c.direct.color_file, &color) || !direct_begin(ext, c.direct, color, frame, e)) {
      std::printf("%s: direct frame: %s\n", c.name.c_str(), e.c_str());
      return 1;
    }
  }
  if (!d->launch(c.kernel, c.grid, c.block, argv.data())) {
    std::printf("%s: launch %s: %s\n", c.name.c_str(), c.kernel.c_str(), d->err().c_str());
    return 1;
  }
  if (c.direct.on) {
    std::vector<uint8_t> out;
    std::string e;
    if (!direct_end(ext, c.direct, frame, out, e) || !write_file(base + "/" + c.direct.out_file, out)) {
      std::printf("%s: direct output: %s\n", c.name.c_str(), e.c_str());
      return 1;
    }
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
    } else if (w == "direct") {
      Direct& dr = c.direct;
      int64_t fill = 0;
      if (!(s >> dr.color_file >> dr.rw >> dr.rh >> dr.ow >> dr.oh >> dr.out_file >> fill) || !dr.rw || !dr.rh ||
          !dr.ow || !dr.oh) {
        std::printf("bad direct line: %s\n", line.c_str());
        return 2;
      }
      dr.fill = (uint32_t)fill;
      dr.on = true;
    } else if (w == "arg") {
      Arg a;
      s >> a.kind;
      if (a.kind == "buf" || a.kind == "bytes") s >> a.file;
      else if (a.kind == "out") {
        s >> a.bytes >> a.file;
        if (!(s >> a.value)) a.value = 0;  // the fill byte (a sentinel), default 0
      }
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
