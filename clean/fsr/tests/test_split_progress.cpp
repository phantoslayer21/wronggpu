// SPDX-License-Identifier: MIT
// Public D3D12 WARP fence metadata only: no recorded GPU commands or neural data.
#include "../src/split_progress.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <string>
#include <vector>
using namespace wgwin;
static std::vector<std::string> lines;
static int failures=0,checks=0;
#define CHECK(x) do{++checks;if(!(x)){++failures;std::printf("FAIL line %d: %s\n",__LINE__,#x);}}while(0)
static void Log(void*,const char* text){lines.emplace_back(text);}
int main(){
    IDXGIFactory4* factory=nullptr; IDXGIAdapter* adapter=nullptr; ID3D12Device* device=nullptr;
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
       FAILED(D3D12CreateDevice(adapter,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)))) return 1;
    ID3D12Fence *order=nullptr,*done=nullptr,*resume=nullptr;
    CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&order))));
    CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&done))));
    CHECK(SUCCEEDED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&resume))));
    if(!order||!done||!resume)return 1;
    {
        SplitProgress trace;
        CHECK(trace.Track(2,nullptr,device,order,2,done,3,resume,3,0));
        trace.Poll(999999,&Log,nullptr); CHECK(lines.empty()&&trace.Count()==1);
        trace.Poll(1000000,&Log,nullptr); CHECK(lines.size()==1&&lines.back().find("stage=order")!=std::string::npos);
        CHECK(order->GetCompletedValue()==0 && done->GetCompletedValue()==0 && resume->GetCompletedValue()==0);
        trace.Submitted(2,reinterpret_cast<void*>(uintptr_t(0x1234)));
        order->Signal(2); trace.Poll(2000000,&Log,nullptr);
        CHECK(lines.back().find("stage=first-half")!=std::string::npos);
        char queueText[32]; std::snprintf(queueText,sizeof queueText,"%p",reinterpret_cast<void*>(uintptr_t(0x1234)));
        CHECK(lines.back().find(std::string("queue=")+queueText)!=std::string::npos);
        done->Signal(3); trace.Poll(3000000,&Log,nullptr);
        CHECK(lines.back().find("stage=resume")!=std::string::npos&&trace.Count()==1);
        trace.Poll(3100000,&Log,nullptr); CHECK(lines.size()==3);
        resume->Signal(3); trace.Poll(4000000,&Log,nullptr);
        CHECK(lines.back().find("stage=recovered")!=std::string::npos&&trace.Count()==0);
        lines.clear(); CHECK(trace.Track(3,nullptr,device,order,2,done,3,resume,3,5000000));
        trace.Poll(5000001,&Log,nullptr); CHECK(lines.empty()&&trace.Count()==0);
        for(uint64_t id=10;id<74;++id) CHECK(trace.Track(id,nullptr,device,order,4,done,5,resume,5,0));
        CHECK(trace.Count()==64&&!trace.Track(74,nullptr,device,order,4,done,5,resume,5,0));
        trace.CpuReleased(10); trace.Poll(1000000,&Log,nullptr);
        CHECK(lines.front().find("cpu_release=1")!=std::string::npos);
        // A removal sentinel must not be silently treated as successful inference or completion.
        order->Signal(UINT64_MAX); trace.Poll(2000000,&Log,nullptr);
        CHECK(trace.Count()==0&&lines.back().find("stage=device-status")!=std::string::npos&&lines.back().find("sentinel=1")!=std::string::npos);
        CHECK(done->GetCompletedValue()==3 && resume->GetCompletedValue()==3);
    }
    resume->Release(); done->Release(); order->Release(); device->Release(); adapter->Release(); factory->Release();
    std::printf("split progress: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
