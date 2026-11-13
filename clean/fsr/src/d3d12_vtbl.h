// SPDX-License-Identifier: MIT
/*
 * Positions of the D3D12 methods WrongGPU's own vtable hooks patch (d3d12_hooks.cpp), counted from the start of each
 * interface's vtable: IUnknown (3 methods), ID3D12Object (4), ID3D12DeviceChild (1), then the interface chain in
 * declaration order (Microsoft's d3d12.h). Checked against the Windows SDK / MinGW-w64 C declarations of the vtables
 * by d3d12_vtbl_check.c at build time (the ID3D12GraphicsCommandList4/5 entries only where the header declares those
 * interfaces, i.e. with the Windows SDK).
 */
#ifndef WRONGGPU_D3D12_VTBL_H
#define WRONGGPU_D3D12_VTBL_H

enum WgVtblDevice {
    WG_VT_DEVICE_CREATE_COMMAND_LIST = 12,
    WG_VT_DEVICE4_CREATE_COMMAND_LIST1 = 51  /* ID3D12Device4 */
};

enum WgVtblQueue {
    WG_VT_QUEUE_EXECUTE_COMMAND_LISTS = 10
};

enum WgVtblList {
    WG_VT_LIST_CLOSE = 9,
    WG_VT_LIST_RESET = 10,
    WG_VT_LIST_CLEAR_STATE = 11,
    WG_VT_LIST_IA_SET_PRIMITIVE_TOPOLOGY = 20,
    WG_VT_LIST_RS_SET_VIEWPORTS = 21,
    WG_VT_LIST_RS_SET_SCISSOR_RECTS = 22,
    WG_VT_LIST_OM_SET_BLEND_FACTOR = 23,
    WG_VT_LIST_OM_SET_STENCIL_REF = 24,
    WG_VT_LIST_SET_PIPELINE_STATE = 25,
    WG_VT_LIST_EXECUTE_BUNDLE = 27,
    WG_VT_LIST_SET_DESCRIPTOR_HEAPS = 28,
    WG_VT_LIST_SET_COMPUTE_ROOT_SIGNATURE = 29,
    WG_VT_LIST_SET_GRAPHICS_ROOT_SIGNATURE = 30,
    WG_VT_LIST_SET_COMPUTE_ROOT_DESCRIPTOR_TABLE = 31,
    WG_VT_LIST_SET_GRAPHICS_ROOT_DESCRIPTOR_TABLE = 32,
    WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANT = 33,
    WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANT = 34,
    WG_VT_LIST_SET_COMPUTE_ROOT_32BIT_CONSTANTS = 35,
    WG_VT_LIST_SET_GRAPHICS_ROOT_32BIT_CONSTANTS = 36,
    WG_VT_LIST_SET_COMPUTE_ROOT_CBV = 37,
    WG_VT_LIST_SET_GRAPHICS_ROOT_CBV = 38,
    WG_VT_LIST_SET_COMPUTE_ROOT_SRV = 39,
    WG_VT_LIST_SET_GRAPHICS_ROOT_SRV = 40,
    WG_VT_LIST_SET_COMPUTE_ROOT_UAV = 41,
    WG_VT_LIST_SET_GRAPHICS_ROOT_UAV = 42,
    WG_VT_LIST_IA_SET_INDEX_BUFFER = 43,
    WG_VT_LIST_IA_SET_VERTEX_BUFFERS = 44,
    WG_VT_LIST_SO_SET_TARGETS = 45,
    WG_VT_LIST_OM_SET_RENDER_TARGETS = 46,
    WG_VT_LIST_BEGIN_QUERY = 52,
    WG_VT_LIST_END_QUERY = 53,
    WG_VT_LIST_RESOLVE_QUERY_DATA = 54,
    WG_VT_LIST_SET_PREDICATION = 55,
    WG_VT_LIST_OM_SET_DEPTH_BOUNDS = 62,      /* ID3D12GraphicsCommandList1 */
    WG_VT_LIST_SET_SAMPLE_POSITIONS = 63,     /* ID3D12GraphicsCommandList1 */
    WG_VT_LIST_SET_VIEW_INSTANCE_MASK = 65,   /* ID3D12GraphicsCommandList1 */
    WG_VT_LIST_BEGIN_RENDER_PASS = 68,        /* ID3D12GraphicsCommandList4 */
    WG_VT_LIST_END_RENDER_PASS = 69,          /* ID3D12GraphicsCommandList4 */
    WG_VT_LIST_SET_PIPELINE_STATE1 = 75,      /* ID3D12GraphicsCommandList4 */
    WG_VT_LIST_RS_SET_SHADING_RATE = 77,      /* ID3D12GraphicsCommandList5 */
    WG_VT_LIST_RS_SET_SHADING_RATE_IMAGE = 78,/* ID3D12GraphicsCommandList5 */
    WG_VT_LIST_SLOTS = 79                     /* entries WrongGPU keeps originals for */
};

#endif /* WRONGGPU_D3D12_VTBL_H */
