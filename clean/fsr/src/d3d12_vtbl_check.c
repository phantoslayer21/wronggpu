// SPDX-License-Identifier: MIT
/* Compile-time check of d3d12_vtbl.h against the C declarations of the D3D12 vtables in the Windows SDK or MinGW-w64
 * d3d12.h. Produces no code; compiled with every Windows build in the object library wg_sdk_layout_check, linked into
 * no DLL. */
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <stddef.h>

#include "d3d12_vtbl.h"

#define WG_SLOT(vtbl, method) (offsetof(vtbl, method) / sizeof(void*))
#define WG_CHECK(vtbl, method, value) typedef char wg_check_##vtbl##_##method[WG_SLOT(vtbl, method) == (value) ? 1 : -1]

WG_CHECK(ID3D12DeviceVtbl, CreateCommandList, WG_VT_DEVICE_CREATE_COMMAND_LIST);
#ifdef __ID3D12Device4_INTERFACE_DEFINED__
WG_CHECK(ID3D12Device4Vtbl, CreateCommandList1, WG_VT_DEVICE4_CREATE_COMMAND_LIST1);
#endif
WG_CHECK(ID3D12CommandQueueVtbl, ExecuteCommandLists, WG_VT_QUEUE_EXECUTE_COMMAND_LISTS);

WG_CHECK(ID3D12GraphicsCommandListVtbl, Close, WG_VT_LIST_CLOSE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, Reset, WG_VT_LIST_RESET);
WG_CHECK(ID3D12GraphicsCommandListVtbl, ClearState, WG_VT_LIST_CLEAR_STATE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, IASetPrimitiveTopology, WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY);
WG_CHECK(ID3D12GraphicsCommandListVtbl, RSSetViewports, WG_VT_LIST_RS_SET_VIEWPORTS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, RSSetScissorRects, WG_VT_LIST_RS_SET_SCISSOR_RECTS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, OMSetBlendFactor, WG_VT_LIST_OM_SET_BLEND_FACTOR);
WG_CHECK(ID3D12GraphicsCommandListVtbl, OMSetStencilRef, WG_VT_LIST_OM_SET_STENCIL_REF);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetPipelineState, WG_VT_LIST_SET_PIPELINE_STATE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, ExecuteBundle, WG_VT_LIST_EXECUTE_BUNDLE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetDescriptorHeaps, WG_VT_LIST_SET_DESCRIPTOR_HEAPS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRootSignature, WG_VT_LIST_SET_COMPUTE_ROOT_SIGNATURE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRootSignature, WG_VT_LIST_SET_GRAPHICS_ROOT_SIGNATURE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRootDescriptorTable, WG_VT_LIST_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRootDescriptorTable, WG_VT_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRoot32BitConstant, WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANT);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRoot32BitConstant, WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANT);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRoot32BitConstants, WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANTS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRoot32BitConstants, WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRootConstantBufferView, WG_VT_LIST_SET_COMPUTE_ROOT_CBV);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRootConstantBufferView, WG_VT_LIST_SET_GRAPHICS_ROOT_CBV);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRootShaderResourceView, WG_VT_LIST_SET_COMPUTE_ROOT_SRV);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRootShaderResourceView, WG_VT_LIST_SET_GRAPHICS_ROOT_SRV);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetComputeRootUnorderedAccessView, WG_VT_LIST_SET_COMPUTE_ROOT_UAV);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetGraphicsRootUnorderedAccessView, WG_VT_LIST_SET_GRAPHICS_ROOT_UAV);
WG_CHECK(ID3D12GraphicsCommandListVtbl, IASetIndexBuffer, WG_VT_LIST_IA_SET_INDEX_BUFFER);
WG_CHECK(ID3D12GraphicsCommandListVtbl, IASetVertexBuffers, WG_VT_LIST_IA_SET_VERTEX_BUFFERS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SOSetTargets, WG_VT_LIST_SO_SET_TARGETS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, OMSetRenderTargets, WG_VT_LIST_OM_SET_RENDER_TARGETS);
WG_CHECK(ID3D12GraphicsCommandListVtbl, BeginQuery, WG_VT_LIST_BEGIN_QUERY);
WG_CHECK(ID3D12GraphicsCommandListVtbl, EndQuery, WG_VT_LIST_END_QUERY);
WG_CHECK(ID3D12GraphicsCommandListVtbl, ResolveQueryData, WG_VT_LIST_RESOLVE_QUERY_DATA);
WG_CHECK(ID3D12GraphicsCommandListVtbl, SetPredication, WG_VT_LIST_SET_PREDICATION);
WG_CHECK(ID3D12GraphicsCommandList1Vtbl, OMSetDepthBounds, WG_VT_LIST_OM_SET_DEPTH_BOUNDS);
WG_CHECK(ID3D12GraphicsCommandList1Vtbl, SetSamplePositions, WG_VT_LIST_SET_SAMPLE_POSITIONS);
WG_CHECK(ID3D12GraphicsCommandList1Vtbl, SetViewInstanceMask, WG_VT_LIST_SET_VIEW_INSTANCE_MASK);
#ifdef __ID3D12GraphicsCommandList5_INTERFACE_DEFINED__
WG_CHECK(ID3D12GraphicsCommandList5Vtbl, BeginRenderPass, WG_VT_LIST_BEGIN_RENDER_PASS);
WG_CHECK(ID3D12GraphicsCommandList5Vtbl, EndRenderPass, WG_VT_LIST_END_RENDER_PASS);
WG_CHECK(ID3D12GraphicsCommandList5Vtbl, SetPipelineState1, WG_VT_LIST_SET_PIPELINE_STATE1);
WG_CHECK(ID3D12GraphicsCommandList5Vtbl, RSSetShadingRate, WG_VT_LIST_RS_SET_SHADING_RATE);
WG_CHECK(ID3D12GraphicsCommandList5Vtbl, RSSetShadingRateImage, WG_VT_LIST_RS_SET_SHADING_RATE_IMAGE);
#endif
