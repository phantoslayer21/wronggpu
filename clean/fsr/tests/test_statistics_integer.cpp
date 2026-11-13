// SPDX-License-Identifier: MIT
// GPU regression for the production statistics adder, including carries and
// unsigned 64-bit overflow. All inputs are original synthetic integers.
#define main existing_integration_main
#include "test_dll_d3d12.cpp"
#undef main
#include "../src/statistics_query.cpp"
#include <random>

int main() {
    Gpu gpu;
    if(!gpu.Init()) return 77;
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Type=D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;qd.Count=1;
    ID3D12QueryHeap* heap=nullptr;
    CHECK(SUCCEEDED(gpu.device->CreateQueryHeap(&qd,IID_PPV_ARGS(&heap))));
    if(!heap) return 1;
    gpu.keep.push_back(heap);
    auto& w=wgwin::get({heap,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,0});
    constexpr size_t cases=32, count=11;
    auto* upload=gpu.Buffer(cases*512,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    auto* readback=gpu.Buffer(cases*256,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    CHECK(upload && readback);
    if(!upload || !readback) return 1;
    unsigned char* mapped=nullptr;
    D3D12_RANGE none{0,0};
    CHECK(SUCCEEDED(upload->Map(0,&none,reinterpret_cast<void**>(&mapped))));
    if(!mapped) return 1;
    uint64_t expected[cases][count]{};
    std::mt19937_64 random(20261010);
    const uint64_t edge[]={0,1,0xffffffffull,0x100000000ull,0xffffffffffffffffull,0x8000000000000000ull};
    for(size_t c=0;c<cases;++c) for(size_t i=0;i<count;++i) {
        uint64_t a=c<6?edge[c]:random(), b=c<6?edge[(c+i+1)%6]:random();
        std::memcpy(mapped+c*512+i*8,&a,8);
        std::memcpy(mapped+c*512+256+i*8,&b,8);
        expected[c][i]=a+b;
    }
    D3D12_RANGE written{0,cases*512};upload->Unmap(0,&written);
    gpu.Begin();
    for(size_t c=0;c<cases;++c) {
        gpu.list->CopyBufferRegion(w.prefix,0,upload,c*512,count*8);
        gpu.list->CopyBufferRegion(w.current,0,upload,c*512+256,count*8);
        wgwin::add(gpu.list,w);
        gpu.list->CopyBufferRegion(readback,c*256,w.sum,0,count*8);
        wgwin::finishAdd(gpu.list,w,false);
    }
    CHECK(gpu.SubmitAndWait());
    D3D12_RANGE read{0,cases*256};mapped=nullptr;
    CHECK(SUCCEEDED(readback->Map(0,&read,reinterpret_cast<void**>(&mapped))));
    if(mapped) {
        for(size_t c=0;c<cases;++c) for(size_t i=0;i<count;++i) {
            uint64_t value;std::memcpy(&value,mapped+c*256+i*8,8);
            CHECK(value==expected[c][i]);
        }
        readback->Unmap(0,&none);
    }
    gpu.CheckDebugMessages();
    std::printf("statistics integer GPU: %zu exact values, %d failures\n",cases*count,g_failures);
    return g_failures?1:0;
}
