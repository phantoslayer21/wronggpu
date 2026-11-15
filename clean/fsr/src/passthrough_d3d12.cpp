// SPDX-License-Identifier: MIT
// The last resort of the front end's refused-frame contract (B8, frontend.h Platform::passThrough): when nothing could
// upscale a frame, the game's own colour is copied into the output texture, unscaled, so that the game gets an image and an
// answer it can use instead of an error and an output nobody wrote. WrongGPU's own code.
#include "passthrough_d3d12.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace wgfsr {

namespace {

// The typeless group of a format: CopyTextureRegion needs both textures in the same one.
DXGI_FORMAT TypelessGroup(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_UINT:
        case DXGI_FORMAT_R32G32B32A32_SINT: return DXGI_FORMAT_R32G32B32A32_TYPELESS;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R16G16B16A16_SINT:
            return DXGI_FORMAT_R16G16B16A16_TYPELESS;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_UINT:
            return DXGI_FORMAT_R10G10B10A2_TYPELESS;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SNORM: case DXGI_FORMAT_R8G8B8A8_SINT:
            return DXGI_FORMAT_R8G8B8A8_TYPELESS;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_TYPELESS;
        default: return f;
    }
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    return b;
}

}  // namespace

bool CopyColorToOutput(const WgFrame& f, std::string* why) {
    auto fail = [&](const char* m) {
        if (why) *why = m;
        return false;
    };
    auto* list = static_cast<ID3D12GraphicsCommandList*>(f.command_list);
    auto* src = static_cast<ID3D12Resource*>(f.color.resource);
    auto* dst = static_cast<ID3D12Resource*>(f.output.resource);
    if (!list || !src || !dst) return fail("no command list, colour or output texture");
    const D3D12_RESOURCE_DESC sd = src->GetDesc(), dd = dst->GetDesc();
    if (sd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || dd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return fail("the colour or the output is not a 2D texture");
    if (sd.SampleDesc.Count != 1 || dd.SampleDesc.Count != 1) return fail("a multisampled texture cannot be copied");
    if (TypelessGroup(sd.Format) != TypelessGroup(dd.Format))
        return fail("the colour and the output texture are in different format families");
    const uint32_t x = f.render_x, y = f.render_y;
    if (x >= sd.Width || y >= sd.Height) return fail("the active region lies outside the colour texture");
    const uint32_t w = (uint32_t)std::min<uint64_t>({f.render_width ? f.render_width : sd.Width, sd.Width - x, dd.Width});
    const uint32_t h = (uint32_t)std::min<uint64_t>({f.render_height ? f.render_height : sd.Height, sd.Height - y, dd.Height});
    if (!w || !h) return fail("an empty region");
    const D3D12_RESOURCE_STATES from_src = static_cast<D3D12_RESOURCE_STATES>(f.color.state);
    const D3D12_RESOURCE_STATES from_dst = static_cast<D3D12_RESOURCE_STATES>(f.output.state);

    D3D12_RESOURCE_BARRIER before[2], after[2];
    UINT nb = 0, na = 0;
    if (from_src != D3D12_RESOURCE_STATE_COPY_SOURCE) {
        before[nb++] = Transition(src, from_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        after[na++] = Transition(src, D3D12_RESOURCE_STATE_COPY_SOURCE, from_src);
    }
    if (from_dst != D3D12_RESOURCE_STATE_COPY_DEST) {
        before[nb++] = Transition(dst, from_dst, D3D12_RESOURCE_STATE_COPY_DEST);
        after[na++] = Transition(dst, D3D12_RESOURCE_STATE_COPY_DEST, from_dst);
    }
    if (nb) list->ResourceBarrier(nb, before);
    D3D12_TEXTURE_COPY_LOCATION s = {}, d = {};
    s.pResource = src;
    s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    s.SubresourceIndex = 0;
    d.pResource = dst;
    d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    d.SubresourceIndex = 0;
    D3D12_BOX box = {x, y, 0, x + w, y + h, 1};
    list->CopyTextureRegion(&d, 0, 0, 0, &s, &box);
    if (na) list->ResourceBarrier(na, after);
    return true;
}

}  // namespace wgfsr
