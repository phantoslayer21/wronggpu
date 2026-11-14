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
//   direct <slot> <DXGI format> <w> <h> <file> a texture of a direct-resource kernel (rs=direct; native_direct.hlsli):
//                                              slot color, depth, motion or output; <file> holds the w x h texels row
//                                              by row (tightly packed), "-" for none (the output, written by no
//                                              kernel replayed here). The case binds them as one direct frame
//                                              (NativeD3D12Ext::bind_direct_frame) around its launch.
//   direct_params <rx> <ry> <rw> <rh>          the frame's active render rectangle; motion scale 1, offset 0,
//                                              color_pi 1, no exposure texture
//   direct_out <file>                          after the launch, the output texture's texels are written to <file>,
//                                              row by row (tightly packed); its initial texels come from the output
//                                              slot's file (a sentinel fill, so that an unwritten texel shows)
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

struct DirectTex {
  std::string slot, file;
  unsigned format = 0, width = 0, height = 0;
  ID3D12Resource* tex = nullptr;
  ID3D12Resource* upload = nullptr;
};

struct Case {
  std::string name, kernel;
  unsigned grid = 0, block = 0;
  std::vector<std::pair<std::string, std::string>> modules;
  std::vector<Arg> args;
  std::vector<DirectTex> direct;
  NativeDirectParams dparams;
  std::string direct_out;   // the output texture's readback file ("" = none)
};

// The output texture (UNORDERED_ACCESS) copied to a readback buffer after the case's work, rows packed into `out`.
bool read_direct_output(NativeD3D12Ext* ext, Case& c, std::vector<uint8_t>& out, std::string& err) {
  DirectTex* t = nullptr;
  for (auto& x : c.direct)
    if (x.slot == "output") t = &x;
  if (!t || !t->tex) return err = "no output texture", false;
  ID3D12Device* dev = nullptr;
  if (FAILED(ext->arena_resource()->GetDevice(__uuidof(ID3D12Device), (void**)&dev))) return err = "GetDevice", false;
  const D3D12_RESOURCE_DESC rd = t->tex->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
  UINT rows = 0;
  UINT64 row_bytes = 0, total = 0;
  dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, &rows, &row_bytes, &total);
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = D3D12_HEAP_TYPE_READBACK;
  D3D12_RESOURCE_DESC bd = {};
  bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bd.Width = total;
  bd.Height = 1;
  bd.DepthOrArraySize = 1;
  bd.MipLevels = 1;
  bd.SampleDesc.Count = 1;
  bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ID3D12Resource* rb = nullptr;
  const bool made = SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                           nullptr, __uuidof(ID3D12Resource), (void**)&rb));
  dev->Release();
  if (!made) return err = "readback buffer", false;
  ID3D12Resource* tex = t->tex;
  bool ok = ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, [&](ID3D12GraphicsCommandList* l, std::string&) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = tex;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    l->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = rb;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    src.pResource = tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    return true;
  }, &err) && ext->sync(&err);
  uint8_t* map = nullptr;
  if (ok) ok = SUCCEEDED(rb->Map(0, nullptr, (void**)&map));
  if (ok) {
    out.resize((size_t)row_bytes * rows);
    for (UINT r = 0; r < rows; ++r)
      std::memcpy(out.data() + (size_t)r * row_bytes, map + fp.Offset + (size_t)r * fp.Footprint.RowPitch,
                  (size_t)row_bytes);
    D3D12_RANGE none = {0, 0};
    rb->Unmap(0, &none);
  } else if (err.empty()) {
    err = "readback";
  }
  rb->Release();
  return ok;
}

void release_direct(Case& c) {
  for (auto& t : c.direct) {
    if (t.tex) t.tex->Release();
    if (t.upload) t.upload->Release();
    t.tex = t.upload = nullptr;
  }
}

// The case's direct textures: created (default heap), filled from their files through an upload buffer in one recorded
// pass, left in the states a direct dispatch needs (inputs NON_PIXEL_SHADER_RESOURCE, output UNORDERED_ACCESS), then
// bound as the device's direct frame.
bool bind_direct(NativeD3D12Ext* ext, Case& c, const std::string& base) {
  ID3D12Device* dev = nullptr;
  if (FAILED(ext->arena_resource()->GetDevice(__uuidof(ID3D12Device), (void**)&dev))) {
    std::printf("%s: direct: no D3D12 device\n", c.name.c_str());
    return false;
  }
  bool ok = true;
  std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(c.direct.size());
  for (size_t i = 0; ok && i < c.direct.size(); ++i) {
    DirectTex& t = c.direct[i];
    const bool out = t.slot == "output";
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = t.width;
    rd.Height = t.height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = (DXGI_FORMAT)t.format;
    rd.SampleDesc.Count = 1;
    rd.Flags = out ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ok = SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                nullptr, __uuidof(ID3D12Resource), (void**)&t.tex));
    if (!ok || t.file == "-") continue;
    std::vector<uint8_t> data;
    UINT64 row_bytes = 0, total = 0;
    UINT rows = 0;
    dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp[i], &rows, &row_bytes, &total);
    if (!read_file(base + "/" + t.file, &data) || data.size() != row_bytes * rows) {
      std::printf("%s: direct %s: %s is not %llu bytes\n", c.name.c_str(), t.slot.c_str(), t.file.c_str(),
                  (unsigned long long)(row_bytes * rows));
      ok = false;
      break;
    }
    D3D12_HEAP_PROPERTIES up = {};
    up.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = total;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ok = SUCCEEDED(dev->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                nullptr, __uuidof(ID3D12Resource), (void**)&t.upload));
    uint8_t* map = nullptr;
    if (ok) ok = SUCCEEDED(t.upload->Map(0, nullptr, (void**)&map));
    if (!ok) break;
    for (UINT r = 0; r < rows; ++r)
      std::memcpy(map + fp[i].Offset + (size_t)r * fp[i].Footprint.RowPitch, data.data() + (size_t)r * row_bytes,
                  (size_t)row_bytes);
    t.upload->Unmap(0, nullptr);
  }
  dev->Release();
  if (!ok) {
    std::printf("%s: direct: texture setup failed\n", c.name.c_str());
    return false;
  }
  std::string err;
  ok = ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, [&](ID3D12GraphicsCommandList* l, std::string&) {
    for (size_t i = 0; i < c.direct.size(); ++i) {
      DirectTex& t = c.direct[i];
      if (t.upload) {
        D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
        dst.pResource = t.tex;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = t.upload;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp[i];
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
      }
      D3D12_RESOURCE_BARRIER b = {};
      b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      b.Transition.pResource = t.tex;
      b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
      b.Transition.StateAfter = t.slot == "output" ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                   : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      l->ResourceBarrier(1, &b);
    }
    return true;
  }, &err);
  if (!ok) {
    std::printf("%s: direct: upload pass: %s\n", c.name.c_str(), err.c_str());
    return false;
  }
  NativeDirectFrame f;
  for (auto& t : c.direct) {
    if (t.slot == "color") f.color = t.tex, f.color_format = (DXGI_FORMAT)t.format;
    else if (t.slot == "depth") f.depth = t.tex, f.depth_format = (DXGI_FORMAT)t.format;
    else if (t.slot == "motion") f.motion = t.tex, f.motion_format = (DXGI_FORMAT)t.format;
    else if (t.slot == "output") f.output = t.tex, f.output_format = (DXGI_FORMAT)t.format;
  }
  f.params = c.dparams;
  if (!ext->bind_direct_frame(f, &err)) {
    std::printf("%s: direct: bind: %s\n", c.name.c_str(), err.c_str());
    return false;
  }
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
  NativeD3D12Ext* ext = c.direct.empty() ? nullptr : d3d12_ext(d);
  if (!c.direct.empty() && (!ext || !bind_direct(ext, c, base))) {
    release_direct(c);
    return 1;
  }
  if (!d->launch(c.kernel, c.grid, c.block, argv.data())) {
    std::printf("%s: launch %s: %s\n", c.name.c_str(), c.kernel.c_str(), d->err().c_str());
    return 1;
  }
  if (ext) {
    std::string err;
    if (!ext->end_direct_frame(&err) || !ext->sync(&err)) {
      std::printf("%s: direct: %s\n", c.name.c_str(), err.c_str());
      release_direct(c);
      return 1;
    }
    if (!c.direct_out.empty()) {
      std::vector<uint8_t> texels;
      if (!read_direct_output(ext, c, texels, err) || !write_file(base + "/" + c.direct_out, texels)) {
        std::printf("%s: direct output: %s\n", c.name.c_str(), err.c_str());
        release_direct(c);
        return 1;
      }
    }
  }
  if (!d->sync()) { std::printf("%s: sync: %s\n", c.name.c_str(), d->err().c_str()); return 1; }
  release_direct(c);
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
      DirectTex t;
      s >> t.slot >> t.format >> t.width >> t.height >> t.file;
      if (t.slot != "color" && t.slot != "depth" && t.slot != "motion" && t.slot != "output") {
        std::printf("unknown direct slot %s\n", t.slot.c_str());
        return 2;
      }
      c.direct.push_back(t);
    } else if (w == "direct_params") {
      s >> c.dparams.rx >> c.dparams.ry >> c.dparams.rw >> c.dparams.rh;
    } else if (w == "direct_out") {
      s >> c.direct_out;
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
