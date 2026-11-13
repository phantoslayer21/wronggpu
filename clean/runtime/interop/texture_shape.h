// SPDX-License-Identifier: MIT
// The game's textures as the D3D12 paths read and write them (0.6.0 follow-up; a tester's CONTROL Resonant gives a
// colour input with 11 mips): a single-sample, non-array Texture2D with any number of mips. Only mip 0 is read
// (shader resource view: MostDetailedMip 0, MipLevels 1) or written (unordered access view: MipSlice 0); a copy uses
// subresource 0. Barriers on such a texture: all subresources for a single-mip texture, as before (its planes, e.g. a
// depth-stencil's, together); subresource 0 alone for a texture with more mips, so the other mips are never moved
// out of a state the game did not report for them. Arrays, multisampled and other dimensions stay refused.
// Header only, no device: CPU-tested by tests/test_texture_shape.cpp. Written for this project.
#pragma once
#include "d3d12_include.h"

namespace wgi {

inline bool texture_shape_ok(const D3D12_RESOURCE_DESC& d) {
  return d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.DepthOrArraySize == 1 && d.MipLevels >= 1 &&
         d.SampleDesc.Count == 1 && d.SampleDesc.Quality == 0 && d.Width != 0 && d.Height != 0;
}

// The words of a refusal (logged with the texture's name before them).
inline const char* texture_shape_rule() {
  return "requires a single-sample non-array Texture2D (any number of mips: mip 0 is used)";
}

inline UINT barrier_subresource(const D3D12_RESOURCE_DESC& d) {
  return d.MipLevels > 1 ? 0u : D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
}
inline UINT barrier_subresource(ID3D12Resource* r) {
  return r ? barrier_subresource(r->GetDesc()) : D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
}

inline D3D12_SHADER_RESOURCE_VIEW_DESC mip0_srv(DXGI_FORMAT format) {
  D3D12_SHADER_RESOURCE_VIEW_DESC v{};
  v.Format = format;
  v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  v.Texture2D.MostDetailedMip = 0;
  v.Texture2D.MipLevels = 1;
  return v;
}
inline D3D12_UNORDERED_ACCESS_VIEW_DESC mip0_uav(DXGI_FORMAT format) {
  D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
  v.Format = format;
  v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  v.Texture2D.MipSlice = 0;
  return v;
}

}  // namespace wgi
