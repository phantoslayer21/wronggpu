// SPDX-License-Identifier: MIT
// Original regression against the production output converter. Synthetic finite
// HDR values only; no model or third-party renderer is loaded.
#define main existing_interop_test_main
#include "test_interop_d3d12.cpp"
#undef main
#include "../convert.h"

int main() {
  Gpu g;
  if (!g.init()) return 77;
  wgi::Converter converter;
  std::string why;
  if (!converter.init(g.dev, &why)) return std::puts(why.c_str()), 2;
  const UINT w = 8, h = 8;
  const float values[] = {0.0f, 1.0f, 60000.0f, 65000.0f, 65504.0f, 65564.0f, 100000.0f, -1.0f};
  std::vector<float> data(w * h * 3);
  for (UINT y = 0; y < h; ++y) for (UINT x = 0; x < w; ++x) for (UINT c = 0; c < 3; ++c)
    data[3 * (y * w + x) + c] = values[x];
  ID3D12Resource* upload = g.buffer(data.size() * 4, D3D12_HEAP_TYPE_UPLOAD);
  void* p = nullptr; upload->Map(0, nullptr, &p); std::memcpy(p, data.data(), data.size() * 4); upload->Unmap(0, nullptr);
  D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = data.size() * 4;
  d.Height = d.DepthOrArraySize = d.MipLevels = 1; d.SampleDesc.Count = 1;
  d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  ID3D12Resource* source = nullptr;
  CHECK(SUCCEEDED(g.dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
      nullptr, IID_PPV_ARGS(&source))));
  ID3D12Resource* out = g.texture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  WgFrame frame{}; frame.output = {out, DXGI_FORMAT_R16G16B16A16_FLOAT, w, h, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
  frame.output_width = w; frame.output_height = h;
  g.begin(); g.list->CopyBufferRegion(source, 0, upload, 0, data.size() * 4);
  g.barrier(source, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  converter.record_output(g.list, frame, {source, 0});
  UINT pitch = 0; ID3D12Resource* readback = record_readback(g, out, w, h, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &pitch);
  g.submit(); const auto got = read_rgb(readback, w, h, pitch);
  size_t nonfinite = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i])) ++nonfinite;
    const float expected = q16(std::max(-65504.0f, std::min(65504.0f, data[i])));
    CHECK(got[i] == expected);
  }
  std::printf("HDR output: %zu nonfinite channels, %d failures\n", nonfinite, g_fail);
  readback->Release(); out->Release(); source->Release(); upload->Release();
  return g_fail ? 1 : 0;
}
