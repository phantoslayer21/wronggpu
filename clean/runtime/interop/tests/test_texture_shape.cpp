// SPDX-License-Identifier: MIT
// The game's textures as the D3D12 paths take them (interop/texture_shape.h, 0.6.0 follow-up: a tester's CONTROL
// Resonant gives a colour input with 11 mips): on any host, no GPU, no device. WrongGPU's own code.
//   cl /nologo /std:c++17 /EHsc test_texture_shape.cpp   (or clang++ -std=c++17), then run it
// Exit status 0 when every case passed.
#include <cstdio>
#include <cstring>

#include "../texture_shape.h"

namespace {

int failures = 0;

void expect(const char* name, bool ok) {
  std::printf("%s %s\n", ok ? "pass" : "FAIL", name);
  if (!ok) ++failures;
}

D3D12_RESOURCE_DESC tex(UINT64 w, UINT h, UINT16 mips, UINT16 array = 1, UINT samples = 1,
                        DXGI_FORMAT f = DXGI_FORMAT_R11G11B10_FLOAT) {
  D3D12_RESOURCE_DESC d{};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = w;
  d.Height = h;
  d.DepthOrArraySize = array;
  d.MipLevels = mips;
  d.Format = f;
  d.SampleDesc.Count = samples;
  return d;
}

}  // namespace

int main() {
  // the tester's colour input: R11G11B10_FLOAT 1506x847 with 11 mips
  const D3D12_RESOURCE_DESC color = tex(1506, 847, 11);
  expect("a colour texture with 11 mips is taken", wgi::texture_shape_ok(color));
  expect("...its barriers move subresource 0 only (the other mips keep the game's states)", wgi::barrier_subresource(color) == 0u);
  expect("a single-mip texture: barriers on all subresources, as before",
         wgi::barrier_subresource(tex(1506, 847, 1)) == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
  expect("a single-mip depth texture (two planes when it has stencil): all subresources, as before",
         wgi::barrier_subresource(tex(1506, 847, 1, 1, 1, DXGI_FORMAT_R24G8_TYPELESS)) == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
  expect("depth, motion, exposure and output with mips are taken the same way",
         wgi::texture_shape_ok(tex(1506, 847, 5, 1, 1, DXGI_FORMAT_R32_TYPELESS)) &&
         wgi::texture_shape_ok(tex(1506, 847, 3, 1, 1, DXGI_FORMAT_R16G16_FLOAT)) &&
         wgi::texture_shape_ok(tex(1, 1, 1, 1, 1, DXGI_FORMAT_R32_FLOAT)) &&
         wgi::texture_shape_ok(tex(2560, 1440, 12, 1, 1, DXGI_FORMAT_R16G16B16A16_FLOAT)));
  const D3D12_SHADER_RESOURCE_VIEW_DESC srv = wgi::mip0_srv(DXGI_FORMAT_R11G11B10_FLOAT);
  expect("the shader resource view reads mip 0 only (MostDetailedMip 0, MipLevels 1, Texture2D)",
         srv.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D && srv.Texture2D.MostDetailedMip == 0 && srv.Texture2D.MipLevels == 1 &&
         srv.Format == DXGI_FORMAT_R11G11B10_FLOAT && srv.Shader4ComponentMapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
  const D3D12_UNORDERED_ACCESS_VIEW_DESC uav = wgi::mip0_uav(DXGI_FORMAT_R16G16B16A16_FLOAT);
  expect("the output's unordered access view writes mip 0 (MipSlice 0, Texture2D)",
         uav.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2D && uav.Texture2D.MipSlice == 0 && uav.Format == DXGI_FORMAT_R16G16B16A16_FLOAT);
  expect("an array texture is still refused (with or without mips)", !wgi::texture_shape_ok(tex(1506, 847, 1, 2)) &&
                                                                      !wgi::texture_shape_ok(tex(1506, 847, 11, 6)));
  expect("a multisampled texture is still refused", !wgi::texture_shape_ok(tex(1506, 847, 1, 1, 4)));
  D3D12_RESOURCE_DESC q = tex(1506, 847, 1);
  q.SampleDesc.Quality = 1;
  expect("a nonzero sample quality is refused", !wgi::texture_shape_ok(q));
  D3D12_RESOURCE_DESC t3 = tex(64, 64, 1);
  t3.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  D3D12_RESOURCE_DESC buf = tex(64, 1, 1);
  buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  expect("a 3D texture or a buffer is refused", !wgi::texture_shape_ok(t3) && !wgi::texture_shape_ok(buf));
  expect("an empty texture or zero mips is refused", !wgi::texture_shape_ok(tex(0, 847, 1)) && !wgi::texture_shape_ok(tex(1506, 0, 1)) &&
                                                     !wgi::texture_shape_ok(tex(1506, 847, 0)));
  expect("the refusal's words say what is taken", std::strstr(wgi::texture_shape_rule(), "single-sample non-array Texture2D") != nullptr &&
                                                   std::strstr(wgi::texture_shape_rule(), "mip 0") != nullptr);
  std::printf(failures ? "%d FAILED\n" : "ALL TESTS PASSED\n", failures);
  return failures ? 1 : 0;
}
