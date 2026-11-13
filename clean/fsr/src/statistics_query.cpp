// SPDX-License-Identifier: MIT
#include "statistics_query.h"
#include <d3dcompiler.h>
#include <mutex>
#include <map>
#include <tuple>
#include <stdexcept>
#include <cstring>

namespace wgwin {
namespace {
thread_local unsigned internalDepth = 0;
// The application may retain a query heap and reuse an index on later frames.
// Small per-index workspaces are retained for process lifetime, just like hook
// metadata; no recorded GPU command can outlive them. No readback or CPU wait.
struct Workspace {
    ID3D12QueryHeap* heap = nullptr;
    ID3D12Device* device = nullptr;
    ID3D12Resource *prefix = nullptr, *current = nullptr, *sum = nullptr;
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso = nullptr;
    unsigned parts = 0;
    UINT words = 11;
};
std::recursive_mutex mutex;
using Key = std::tuple<ID3D12QueryHeap*, unsigned, UINT>;
std::map<Key, Workspace*> workspaces;
Key key(const StatisticsQuery& q) { return {q.heap, unsigned(q.type), q.index}; }
UINT words(D3D12_QUERY_TYPE type) { return unsigned(type) == 10 ? 14u : 11u; }
void check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("pipeline-statistics continuation resource/API failure"); }
void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}
ID3D12Resource* buffer(ID3D12Device* device, bool uav) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = 256; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* result = nullptr;
    check(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
           uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&result)));
    return result;
}
void initialize(Workspace& w, const StatisticsQuery& q) {
    check(q.heap->GetDevice(IID_PPV_ARGS(&w.device)));
    w.heap = q.heap; w.heap->AddRef(); w.words = words(q.type);
    w.prefix = buffer(w.device, false); w.current = buffer(w.device, false); w.sum = buffer(w.device, true);
    D3D12_ROOT_PARAMETER p[4]{};
    p[0].ParameterType = p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[1].Descriptor.ShaderRegister = 1;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[3].Constants.Num32BitValues = 1;
    D3D12_ROOT_SIGNATURE_DESC rd{}; rd.NumParameters = 4; rd.pParameters = p;
    ID3DBlob *blob = nullptr, *error = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
    if (error) error->Release(); check(hr);
    hr = w.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&w.root));
    blob->Release(); check(hr);
    // Exact unsigned 64-bit addition implemented as two 32-bit words for SM5.
    const char shader[] =
        "ByteAddressBuffer a:register(t0); ByteAddressBuffer b:register(t1);"
        "RWByteAddressBuffer c:register(u0); cbuffer N:register(b0){uint n;}"
        "[numthreads(1,1,1)] void main(){for(uint i=0;i<n;++i){"
        "uint2 x=a.Load2(i*8),y=b.Load2(i*8); uint lo=x.x+y.x;"
        "c.Store2(i*8,uint2(lo,x.y+y.y+(lo<x.x?1:0)));}}";
    blob = error = nullptr;
    hr = D3DCompile(shader, sizeof(shader)-1, "WrongGPU statistics addition", nullptr, nullptr, "main", "cs_5_0",
                    D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &error);
    if (error) error->Release(); check(hr);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = w.root; pd.CS = {blob->GetBufferPointer(), blob->GetBufferSize()};
    hr = w.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&w.pso)); blob->Release(); check(hr);
}
Workspace& get(const StatisticsQuery& q) {
    auto& slot = workspaces[key(q)];
    if (!slot) {
        auto* w = new Workspace;
        try { initialize(*w, q); } catch (...) {
            if(w->pso) w->pso->Release(); if(w->root) w->root->Release();
            if(w->prefix) w->prefix->Release(); if(w->current) w->current->Release(); if(w->sum) w->sum->Release();
            if(w->heap) w->heap->Release(); if(w->device) w->device->Release(); delete w; throw;
        }
        slot = w;
    }
    return *slot;
}
void add(ID3D12GraphicsCommandList* list, Workspace& w) {
    barrier(list,w.prefix,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(list,w.current,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->SetPredication(nullptr,0,D3D12_PREDICATION_OP_EQUAL_ZERO);
    list->SetComputeRootSignature(w.root); list->SetPipelineState(w.pso);
    list->SetComputeRootShaderResourceView(0,w.prefix->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1,w.current->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(2,w.sum->GetGPUVirtualAddress());
    list->SetComputeRoot32BitConstant(3,w.words,0); list->Dispatch(1,1,1);
    barrier(list,w.sum,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
}
void finishAdd(ID3D12GraphicsCommandList* list, Workspace& w, bool prefixRestored) {
    if (!prefixRestored) barrier(list,w.prefix,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
    barrier(list,w.current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
    barrier(list,w.sum,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
}
bool IsStatisticsQuery(D3D12_QUERY_TYPE t) {
    // PIPELINE_STATISTICS1=10 in the public Windows SDK; numeric form also builds
    // against older interface headers. Its result adds three mesh-stage counters.
    return t == D3D12_QUERY_TYPE_PIPELINE_STATISTICS || unsigned(t) == 10;
}
bool InternalQueryCommands() { return internalDepth != 0; }
InternalQueryScope::InternalQueryScope() { ++internalDepth; }
InternalQueryScope::~InternalQueryScope() { --internalDepth; }
void StatisticsQueryBegin(const StatisticsQuery& q) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto it = workspaces.find(key(q));
    if (it != workspaces.end() && it->second) it->second->parts = 0;
}
bool PrepareStatisticsQueries(const std::vector<StatisticsQuery>& queries) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    try { for (const auto& q : queries) { if(!IsStatisticsQuery(q.type)) return false; (void)get(q); } }
    catch (...) { return false; }
    return true;
}
void SuspendStatisticsQueries(ID3D12GraphicsCommandList* list,const std::vector<StatisticsQuery>& queries,QueryMethods m) {
    std::lock_guard<std::recursive_mutex> lock(mutex); InternalQueryScope scope;
    // End every active statistics query before running helper compute; otherwise
    // the helper would itself pollute another query's CS invocation count.
    for (const auto& q : queries) m.end(list,q.heap,q.type,q.index);
    list->SetPredication(nullptr,0,D3D12_PREDICATION_OP_EQUAL_ZERO);
    for (const auto& q : queries) {
        auto& w = get(q); const UINT64 bytes = w.words * 8;
        m.resolve(list,q.heap,q.type,q.index,1,w.current,0);
        if (!w.parts) {
            barrier(list,w.current,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(w.prefix,0,w.current,0,bytes);
            barrier(list,w.current,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        } else {
            add(list,w);
            barrier(list,w.prefix,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(w.prefix,0,w.sum,0,bytes); finishAdd(list,w,true);
        }
        ++w.parts;
    }
}
void ResumeStatisticsQueries(ID3D12GraphicsCommandList* list,const std::vector<StatisticsQuery>& queries,QueryMethods m) {
    InternalQueryScope scope;
    for (const auto& q : queries) m.begin(list,q.heap,q.type,q.index);
}
bool HasStatisticsPrefix(ID3D12QueryHeap* heap,D3D12_QUERY_TYPE type,UINT first,UINT count) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for(UINT i=0;i<count;++i) {auto it=workspaces.find(key({heap,type,first+i})); if(it!=workspaces.end() && it->second && it->second->parts) return true;}
    return false;
}
void ResolveStatisticsQuery(ID3D12GraphicsCommandList* list,const StatisticsQuery& q,ID3D12Resource* target,UINT64 offset,QueryMethods m) {
    std::lock_guard<std::recursive_mutex> lock(mutex); InternalQueryScope scope;
    auto it=workspaces.find(key(q));
    if(it==workspaces.end() || !it->second || !it->second->parts) {m.resolve(list,q.heap,q.type,q.index,1,target,offset);return;}
    auto& w=*it->second;
    m.resolve(list,q.heap,q.type,q.index,1,w.current,0); add(list,w);
    list->CopyBufferRegion(target,offset,w.sum,0,w.words*8); finishAdd(list,w,false);
}
}
