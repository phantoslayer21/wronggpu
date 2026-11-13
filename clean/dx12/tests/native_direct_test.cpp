// SPDX-License-Identifier: MIT
// Original direct-texture versus staged supplied-device synthetic validation/timing.
// Converter arithmetic is shared verbatim by both paths; reference network inputs
// are downloaded AFTER GPU conversion, never regenerated with host float math.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <tlhelp32.h>
#include "../../runtime/interop/native_candidate.h"
#include "../../runtime/runtime_core.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void hr(HRESULT h,const char* what){need(SUCCEEDED(h),std::string(what)+" HRESULT="+std::to_string((uint32_t)h));}
template<class T>void release(T*& p){if(p)p->Release();p=nullptr;}
D3D12_RESOURCE_BARRIER trans(ID3D12Resource* r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){
  D3D12_RESOURCE_BARRIER z={};z.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  z.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,a,b};return z;
}
bool modules(const char* label,bool forbid){
  HANDLE h=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE|TH32CS_SNAPMODULE32,GetCurrentProcessId());
  need(h!=INVALID_HANDLE_VALUE,"module snapshot failed");MODULEENTRY32W m={};m.dwSize=sizeof(m);
  need(Module32FirstW(h,&m)!=FALSE,"module enumeration failed");bool found=false;
  std::printf("MODULES %s\n",label);
  do{std::wstring n=m.szModule;std::transform(n.begin(),n.end(),n.begin(),::towlower);
    std::printf("MODULE %ls\n",m.szExePath);
    if(n.find(L"amdhip")!=std::wstring::npos || n.find(L"hiprtc")!=std::wstring::npos)found=true;
  }while(Module32NextW(h,&m));CloseHandle(h);
  need(!forbid||!found,"HIP DLL loaded in native-only process");return found;
}
struct Gpu {
  ID3D12Device* d=nullptr;ID3D12CommandQueue* q=nullptr;ID3D12CommandAllocator* a=nullptr;
  ID3D12GraphicsCommandList* l=nullptr;ID3D12Fence* f=nullptr;ID3D12InfoQueue* iq=nullptr;
  HANDLE ev=nullptr;uint64_t serial=0;D3D12_COMMAND_LIST_TYPE type;
  uint64_t timestamp_frequency=0;
  Gpu(D3D12_COMMAND_LIST_TYPE t,bool warp=false):type(t){
    auto dll=LoadLibraryW(L"d3d12.dll");need(dll!=nullptr,"d3d12 load");
    using Create=HRESULT(WINAPI*)(IUnknown*,D3D_FEATURE_LEVEL,REFIID,void**);
    auto create=(Create)GetProcAddress(dll,"D3D12CreateDevice");need(create!=nullptr,"D3D12CreateDevice symbol");
    IUnknown* adapter=nullptr;
    if(warp){
      using Factory=HRESULT(WINAPI*)(UINT,REFIID,void**);
      auto dx=LoadLibraryW(L"dxgi.dll");auto factory=(Factory)GetProcAddress(dx,"CreateDXGIFactory2");
      IDXGIFactory4* fac=nullptr;hr(factory(0,__uuidof(IDXGIFactory4),(void**)&fac),"DXGI factory");
      hr(fac->EnumWarpAdapter(__uuidof(IUnknown),(void**)&adapter),"WARP adapter");fac->Release();
    }
    hr(create(adapter,D3D_FEATURE_LEVEL_12_0,__uuidof(ID3D12Device),(void**)&d),"create test device");release(adapter);
    D3D12_COMMAND_QUEUE_DESC desc={};desc.Type=t;
    hr(d->CreateCommandQueue(&desc,__uuidof(ID3D12CommandQueue),(void**)&q),"create queue");
    hr(q->GetTimestampFrequency(&timestamp_frequency),"actual queue timestamp frequency");
    hr(d->CreateCommandAllocator(t,__uuidof(ID3D12CommandAllocator),(void**)&a),"create allocator");
    hr(d->CreateCommandList(0,t,a,nullptr,__uuidof(ID3D12GraphicsCommandList),(void**)&l),"create list");hr(l->Close(),"close");
    hr(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,__uuidof(ID3D12Fence),(void**)&f),"create fence");ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    d->QueryInterface(__uuidof(ID3D12InfoQueue),(void**)&iq);
    // The synthetic application owns this newly created device and configures
    // complete diagnostics before any borrowed candidate observes its queue.
    // Never change filters or clear stored messages while a candidate is alive.
    if(iq){
      iq->ClearStorageFilter();iq->ClearRetrievalFilter();
      hr(iq->SetMessageCountLimit(65536),"test InfoQueue message limit");
    }
  }
  ~Gpu(){release(iq);release(l);release(a);release(f);release(q);release(d);if(ev)CloseHandle(ev);}
  void wait(ID3D12Fence* fence,uint64_t v){
    if(fence->GetCompletedValue()<v){hr(fence->SetEventOnCompletion(v,ev),"set event");need(WaitForSingleObject(ev,20000)==WAIT_OBJECT_0,"20 second test fence timeout");}
    need(fence->GetCompletedValue()!=UINT64_MAX && fence->GetCompletedValue()>=v,"unknown completion");
  }
  void begin(){hr(a->Reset(),"allocator reset");hr(l->Reset(a,nullptr),"list reset");}
  uint64_t submit(){hr(l->Close(),"close list");ID3D12CommandList* one=l;q->ExecuteCommandLists(1,&one);
    hr(q->Signal(f,++serial),"signal");wait(f,serial);return serial;}
  void barrier(ID3D12Resource* r,D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to){if(from!=to){auto b=trans(r,from,to);l->ResourceBarrier(1,&b);}}
  ID3D12Resource* buffer(size_t n,D3D12_HEAP_TYPE h){
    D3D12_HEAP_PROPERTIES hp={};hp.Type=h;D3D12_RESOURCE_DESC rd={};rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width=n;rd.Height=1;rd.DepthOrArraySize=1;rd.MipLevels=1;rd.SampleDesc.Count=1;rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if(h==D3D12_HEAP_TYPE_DEFAULT)rd.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto state=h==D3D12_HEAP_TYPE_UPLOAD?D3D12_RESOURCE_STATE_GENERIC_READ:h==D3D12_HEAP_TYPE_READBACK?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ID3D12Resource* r=nullptr;hr(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,state,nullptr,__uuidof(ID3D12Resource),(void**)&r),"buffer");return r;
  }
  WgTexture texture(int w,int h,DXGI_FORMAT format,bool output=false){
    D3D12_HEAP_PROPERTIES hp={};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC rd={};
    rd.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;rd.Width=w;rd.Height=h;rd.DepthOrArraySize=1;rd.MipLevels=1;
    rd.Format=format;rd.SampleDesc.Count=1;if(output)rd.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r=nullptr;hr(d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,__uuidof(ID3D12Resource),(void**)&r),"texture");
    return {r,(uint32_t)format,(uint32_t)w,(uint32_t)h,D3D12_RESOURCE_STATE_COMMON};
  }
  void upload(WgTexture& t,const void* data,size_t pixel){
    auto* r=(ID3D12Resource*)t.resource;auto rd=r->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp={};UINT64 n=0;
    d->GetCopyableFootprints(&rd,0,1,0,&fp,nullptr,nullptr,&n);auto* u=buffer(n,D3D12_HEAP_TYPE_UPLOAD);
    void* m=nullptr;hr(u->Map(0,nullptr,&m),"map upload");
    for(UINT y=0;y<t.height;++y)std::memcpy((char*)m+y*fp.Footprint.RowPitch,(const char*)data+y*t.width*pixel,t.width*pixel);u->Unmap(0,nullptr);
    begin();barrier(r,(D3D12_RESOURCE_STATES)t.state,D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst={};dst.pResource=r;dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src={};src.pResource=u;src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=fp;
    l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);barrier(r,D3D12_RESOURCE_STATE_COPY_DEST,(D3D12_RESOURCE_STATES)t.state);submit();u->Release();
  }
  std::vector<unsigned char> read(const WgTexture& t,size_t pixel){
    auto* r=(ID3D12Resource*)t.resource;auto rd=r->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp={};UINT64 n=0;
    d->GetCopyableFootprints(&rd,0,1,0,&fp,nullptr,nullptr,&n);auto* b=buffer(n,D3D12_HEAP_TYPE_READBACK);
    begin();barrier(r,(D3D12_RESOURCE_STATES)t.state,D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src={};src.pResource=r;src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst={};dst.pResource=b;dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;
    l->CopyTextureRegion(&dst,0,0,0,&src,nullptr);barrier(r,D3D12_RESOURCE_STATE_COPY_SOURCE,(D3D12_RESOURCE_STATES)t.state);submit();
    void* m=nullptr;hr(b->Map(0,nullptr,&m),"map read");std::vector<unsigned char> out(size_t(t.width)*t.height*pixel);
    for(UINT y=0;y<t.height;++y)std::memcpy(out.data()+y*t.width*pixel,(char*)m+y*fp.Footprint.RowPitch,t.width*pixel);
    b->Unmap(0,nullptr);b->Release();return out;
  }
  void output(wgi::Converter& cv,const WgFrame& fr,const std::vector<unsigned char>& raw){
    auto* b=buffer(raw.size(),D3D12_HEAP_TYPE_DEFAULT);auto* u=buffer(raw.size(),D3D12_HEAP_TYPE_UPLOAD);
    void* m=nullptr;hr(u->Map(0,nullptr,&m),"map result");std::memcpy(m,raw.data(),raw.size());u->Unmap(0,nullptr);
    begin();barrier(b,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);l->CopyBufferRegion(b,0,u,0,raw.size());
    barrier(b,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);cv.record_output(l,fr,{b,0});submit();release(u);release(b);
  }
  void debug(){
    if(!iq){std::printf("DEBUG unavailable\n");return;}
    need(iq->GetNumMessagesDiscardedByMessageCountLimit()==0,"debug messages discarded");
    auto n=iq->GetNumStoredMessagesAllowedByRetrievalFilter();size_t errors=0;
    for(UINT64 i=0;i<n;++i){SIZE_T len=0;hr(iq->GetMessage(i,nullptr,&len),"debug size");std::vector<char> buf(len);
      auto* msg=(D3D12_MESSAGE*)buf.data();hr(iq->GetMessage(i,msg,&len),"debug read");
      if(msg->Severity<=D3D12_MESSAGE_SEVERITY_ERROR){++errors;std::printf("DEBUG ERROR %s\n",msg->pDescription);}}
    need(errors==0,"D3D12 debug errors");std::printf("DEBUG clean messages=%llu\n",(unsigned long long)n);
  }
};
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
std::string utc(){SYSTEMTIME s;GetSystemTime(&s);char b[64];std::snprintf(b,sizeof b,"%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",s.wYear,s.wMonth,s.wDay,s.wHour,s.wMinute,s.wSecond,s.wMilliseconds);return b;}
uint64_t qpc_ns(){
  LARGE_INTEGER value={},frequency={};need(QueryPerformanceCounter(&value)&&QueryPerformanceFrequency(&frequency)&&frequency.QuadPart>0,"QPC clock");
  const uint64_t v=(uint64_t)value.QuadPart,f=(uint64_t)frequency.QuadPart;
  return (v/f)*1000000000ull+(v%f)*1000000000ull/f;
}
std::string quote(const std::string& text){
  std::string out="\"";for(char c:text){if(c=='\"'||c=='\\')out+='\\';
    if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else if(c=='\t')out+="\\t";else out+=c;}
  return out+'\"';
}
void marker(const char* kind,const char* config,int round,double actual_warm,bool fixture=false){
  const char* path=std::getenv("WG_GPU_EVIDENCE_MARKER_DIR");
  need(path&&*path,"WG_GPU_EVIDENCE_MARKER_DIR is required for benchmark/marker-contract execution");
  std::filesystem::create_directories(path);
  std::string phase=std::string(fixture?"cpu-contract-":"native-")+config+"-round"+std::to_string(round);
  std::ostringstream json;json<<std::setprecision(17)<<"{\"kind\":"<<quote(kind)<<",\"phase\":"<<quote(phase)
    <<",\"config\":"<<quote(config)<<",\"round\":"<<round<<",\"pid\":"<<GetCurrentProcessId()
    <<",\"monotonic_ns\":"<<qpc_ns()<<",\"utc\":"<<quote(utc())<<",\"warm_seconds\":"<<actual_warm
    <<",\"cpu_contract_fixture\":"<<(fixture?"true":"false")<<"}";
  std::ofstream f(std::filesystem::path(path)/(std::to_string(GetCurrentProcessId())+".jsonl"),std::ios::app);
  need(!!f,"marker file open");f<<json.str()<<'\n';f.flush();need(!!f,"marker file write");
  std::printf("EVIDENCE_MARKER %s\n",json.str().c_str());std::fflush(stdout);
}
void adapter_metadata(Gpu& g){
  using Factory=HRESULT(WINAPI*)(UINT,REFIID,void**);auto dxgi=LoadLibraryW(L"dxgi.dll");
  auto create=(Factory)GetProcAddress(dxgi,"CreateDXGIFactory2");need(create!=nullptr,"DXGI metadata function");
  IDXGIFactory4* factory=nullptr;IDXGIAdapter1* adapter=nullptr;
  hr(create(0,__uuidof(IDXGIFactory4),(void**)&factory),"DXGI metadata factory");LUID luid=g.d->GetAdapterLuid();
  hr(factory->EnumAdapterByLuid(luid,__uuidof(IDXGIAdapter1),(void**)&adapter),"actual adapter by LUID");
  DXGI_ADAPTER_DESC1 desc={};hr(adapter->GetDesc1(&desc),"actual adapter description");
  char name[512]={};need(WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name,sizeof name,nullptr,nullptr)>0,"adapter UTF8");
  LARGE_INTEGER frequency={};need(QueryPerformanceFrequency(&frequency)!=0,"QPC frequency");
  std::printf("ADAPTER_JSON {\"pid\":%u,\"description\":%s,\"luid_low\":%u,\"luid_high\":%u,\"vendor_id\":%u,\"device_id\":%u,\"revision\":%u,\"queue_type\":%u,\"timestamp_frequency_hz\":%llu,\"qpc_frequency_hz\":%llu}\n",
    (unsigned)GetCurrentProcessId(),quote(name).c_str(),(unsigned)luid.LowPart,(unsigned)luid.HighPart,desc.VendorId,desc.DeviceId,desc.Revision,(unsigned)g.type,
    (unsigned long long)g.timestamp_frequency,(unsigned long long)frequency.QuadPart);
  adapter->Release();factory->Release();
}
void enable_debug(){
  auto dll=LoadLibraryW(L"d3d12.dll");using Debug=HRESULT(WINAPI*)(REFIID,void**);
  auto get=(Debug)GetProcAddress(dll,"D3D12GetDebugInterface");ID3D12Debug* d=nullptr;
  if(get&&SUCCEEDED(get(__uuidof(ID3D12Debug),(void**)&d))){d->EnableDebugLayer();d->Release();std::printf("DEBUG enabled\n");}
  else std::printf("DEBUG unavailable at startup\n");
}
void bytes_equal(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b,const std::string& where,size_t active=0){
  need(a.size()==b.size(),where+" size mismatch");size_t n=active?active:a.size();need(n<=a.size(),"comparison active range too large");
  for(size_t i=0;i<n;++i)if(a[i]!=b[i]){
    if(where!="deliberate output mutation"){
      std::ofstream("first-difference-direct.bin",std::ios::binary).write((const char*)a.data(),a.size());
      std::ofstream("first-difference-staged.bin",std::ios::binary).write((const char*)b.data(),b.size());
    }
    throw std::runtime_error(where+" first byte differs at "+std::to_string(i)+" direct="+std::to_string(a[i])+" staged="+std::to_string(b[i]));
  }
}
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
// Original synthetic format packer. Generated finite values are zero or normal
// and exactly representable; reject an unexpected domain rather than infer data.
uint32_t finite_small_float(float x,unsigned mantissa,bool sign){
  uint32_t u=bits(x),s=u>>31,a=u&0x7fffffffu;if(!a)return sign?s<<15:0;
  need(sign||!s,"negative unsigned synthetic float");
  int e=int(a>>23)-127+15;need(e>0&&e<31,"synthetic normal-only format packing domain");
  unsigned shift=23-mantissa;uint32_t m=(a&0x7fffffu)>>shift;
  uint32_t discarded=a&((1u<<shift)-1),half=1u<<(shift-1);
  if(discarded>half||(discarded==half&&(m&1)))++m;
  if(m==(1u<<mantissa)){m=0;++e;}need(e<31,"synthetic format overflow");
  return (sign?s<<15:0)|(uint32_t(e)<<mantissa)|m;
}
size_t pixel_bytes(uint32_t format){
  switch(format){case DXGI_FORMAT_R32G32B32A32_FLOAT:return 16;case DXGI_FORMAT_R16G16B16A16_FLOAT:return 8;
    case DXGI_FORMAT_R32G32_FLOAT:return 8;case DXGI_FORMAT_R16G16_FLOAT:return 4;case DXGI_FORMAT_R32_FLOAT:return 4;
    case DXGI_FORMAT_R16_FLOAT:return 2;case DXGI_FORMAT_R11G11B10_FLOAT:return 4;default:throw std::runtime_error("unknown synthetic texture format");}
}
std::vector<unsigned char> pack(const std::vector<float>& f,unsigned components,uint32_t format){
  size_t pixels=f.size()/components;std::vector<unsigned char> out(pixels*pixel_bytes(format));
  if(format==DXGI_FORMAT_R32G32B32A32_FLOAT||format==DXGI_FORMAT_R32G32_FLOAT||format==DXGI_FORMAT_R32_FLOAT){
    need(out.size()==f.size()*4,"float texture component count");std::memcpy(out.data(),f.data(),out.size());return out;}
  if(format==DXGI_FORMAT_R11G11B10_FLOAT){
    need(components==4,"R11 components");for(size_t i=0;i<pixels;++i){
      uint32_t v=finite_small_float(f[4*i],6,false)|(finite_small_float(f[4*i+1],6,false)<<11)|(finite_small_float(f[4*i+2],5,false)<<22);
      std::memcpy(out.data()+4*i,&v,4);}return out;
  }
  need(out.size()==f.size()*2,"half texture component count");
  for(size_t i=0;i<f.size();++i){uint16_t v=(uint16_t)finite_small_float(f[i],10,true);std::memcpy(out.data()+2*i,&v,2);}return out;
}
struct Scene {
  Gpu& g;WgFrame frame={};WgTexture staged_output={};int rw,rh,ow,oh;
  Scene(Gpu& gpu,int rwidth,int rheight,int owidth,int oheight,DXGI_FORMAT color,DXGI_FORMAT depth,DXGI_FORMAT motion,DXGI_FORMAT output)
    :g(gpu),rw(rwidth),rh(rheight),ow(owidth),oh(oheight){
    frame.struct_size=sizeof(frame);frame.command_queue=g.q;frame.command_list_type=(uint32_t)g.type;
    frame.render_x=2;frame.render_y=1;frame.render_width=rw;frame.render_height=rh;frame.output_width=ow;frame.output_height=oh;
    frame.color=g.texture(rw+4,rh+2,color);frame.depth=g.texture(rw+4,rh+2,depth);frame.motion=g.texture(rw+4,rh+2,motion);
    frame.exposure=g.texture(1,1,DXGI_FORMAT_R32_FLOAT);frame.output=g.texture(ow,oh,output,true);staged_output=g.texture(ow,oh,output,true);
    frame.pre_exposure=1;frame.exposure_value=1;
  }
  ~Scene(){for(auto* t:{&frame.color,&frame.depth,&frame.motion,&frame.exposure,&frame.output,&staged_output})if(t->resource)((ID3D12Resource*)t->resource)->Release();}
  void update(int frame_index,bool reset_each=false){
    const int tw=rw+4,th=rh+2;std::vector<float> col(size_t(tw)*th*4),depth(size_t(tw)*th),motion(size_t(tw)*th*2);
    for(int y=0;y<th;++y)for(int x=0;x<tw;++x){size_t i=size_t(y)*tw+x;
      bool foreground=((x+frame_index*7)%tw)<tw/3&&y>th/4&&y<th*3/4;
      col[i*4]=float((x+3*frame_index)%61)*0.015625f+(foreground?0.25f:0);
      col[i*4+1]=float((y+5*frame_index)%47)*0.015625f;col[i*4+2]=foreground?0.75f:0.125f;col[i*4+3]=1;
      depth[i]=foreground?0.75f:0.25f;motion[i*2]=foreground?-7.0f:float((x%9)-4)*0.03125f;
      motion[i*2+1]=float((y%7)-3)*0.0625f;}
    auto c=pack(col,4,frame.color.format),d=pack(depth,1,frame.depth.format),m=pack(motion,2,frame.motion.format);
    g.upload(frame.color,c.data(),pixel_bytes(frame.color.format));g.upload(frame.depth,d.data(),pixel_bytes(frame.depth.format));
    g.upload(frame.motion,m.data(),pixel_bytes(frame.motion.format));
    float ev=0.5f+(frame_index%8)*0.125f;g.upload(frame.exposure,&ev,4);
    frame.pre_exposure=frame_index%2?1.375f:1.0f;frame.exposure_mode=(uint32_t)(frame_index%3);frame.exposure_value=ev;
    frame.motion_scale[0]=0.625f;frame.motion_scale[1]=1.375f;frame.motion_offset[0]=0.03125f;frame.motion_offset[1]=-0.0625f;
    frame.jitter[0]=float((frame_index%4)-2)*0.125f;frame.jitter[1]=float((frame_index%3)-1)*0.1875f;
    frame.reset=(frame_index==0||reset_each)?2:(frame_index==3||frame_index==5?1:0);
    auto state=frame_index%2?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    g.begin();for(auto* t:{&frame.output,&staged_output}){g.barrier((ID3D12Resource*)t->resource,(D3D12_RESOURCE_STATES)t->state,state);t->state=state;}g.submit();
  }
  WgFrame staged()const{auto f=frame;f.output=staged_output;return f;}
};
void dispatch(Gpu& g,wgi::NativeCandidate& c,const WgFrame& f,bool delayed=false){
  std::string e;ID3D12Fence *done=nullptr,*gate=nullptr;uint64_t value=0;
  if(delayed)hr(g.d->CreateFence(0,D3D12_FENCE_FLAG_NONE,__uuidof(ID3D12Fence),(void**)&gate),"delayed ready fence");
  bool ok=c.dispatch(f,gate?gate:g.f,gate?1:g.serial,&done,&value,&e);
  bool premature=false;double held_ms=0;
  if(gate&&done&&ok){auto begin=Clock::now();do{
      if(done->GetCompletedValue()>=value)premature=true;
      Sleep(5);held_ms=ms(begin,Clock::now());
    }while(held_ms<150);}
  if(gate){hr(gate->Signal(1),"release delayed ready");gate->Release();}
  need(ok,e);need(done&&value,"final completion missing");need(!premature,"completion before producer ready");
  g.wait(done,value);done->Release();need(c.sync(&e),e);
  if(delayed)std::printf("PASS delayed input-ready fence held final completion held_ms=%.6f\n",held_ms);
}
void alpha_one(const std::vector<unsigned char>& data,uint32_t format){
  const size_t stride=pixel_bytes(format);for(size_t i=0;i<data.size();i+=stride){
    if(stride==8){uint16_t v;std::memcpy(&v,data.data()+i+6,2);need(v==0x3c00,"output alpha is not FP16 one");}
    else{uint32_t v;std::memcpy(&v,data.data()+i+12,4);need(v==0x3f800000,"output alpha is not FP32 one");}}
}
void compare_frame(Gpu& g,wgi::NativeCandidate& direct,wgi::NativeCandidate& staged,const Scene& s,bool& mutation){
  auto di=direct.last_frame_inventory(),si=staged.last_frame_inventory();
  need(di.direct_resources&&!si.direct_resources,"inventory route mismatch");
  need(di.input_conversion_passes==0&&di.output_conversion_passes==0,"direct recorded a conversion pass");
  need(si.input_conversion_passes==1&&si.output_conversion_passes==1,"staged conversion oracle missing");
  need(di.removed_io_bytes==uint64_t(72)*s.rw*s.rh+uint64_t(24)*s.ow*s.oh+4&&si.removed_io_bytes==0,"logical removed IO inventory mismatch");
  std::printf("INVENTORY direct_removed_logical_bytes=%llu direct_runtime_bytes=%llu staged_runtime_bytes=%llu direct_input_passes=%u direct_output_passes=%u staged_input_passes=%u staged_output_passes=%u\n",
    (unsigned long long)di.removed_io_bytes,(unsigned long long)di.runtime_bytes,(unsigned long long)si.runtime_bytes,
    di.input_conversion_passes,di.output_conversion_passes,si.input_conversion_passes,si.output_conversion_passes);
  std::string e;const char* names[]={"exposure.pair","enc0.features","enc0.m_near","enc0.m_far","enc0.m_center","enc1.skip","enc1.merged",
    "enc2.skip","enc2.merged","b5.out","dec2.out","dec0.A","dec0.C","dec0.latent","post.history","post.reduced"};
  for(auto* name:names){std::vector<unsigned char>a,b;need(direct.read_buffer(name,&a,&e),e);need(staged.read_buffer(name,&b,&e),e);
    bytes_equal(a,b,name,std::string(name)=="dec0.C"?size_t(s.rw)*s.rh*4:0);}
  if(!s.frame.reset)for(auto* name:{"state.latent_prev","state.motion_history","state.history_prev"}){
    std::vector<unsigned char>a,b;need(direct.read_buffer(name,&a,&e),e);need(staged.read_buffer(name,&b,&e),e);bytes_equal(a,b,name);}
  for(auto* name:{"input.color","input.depth","input.motion","filter.out"}){
    std::vector<unsigned char> absent;need(!direct.read_buffer(name,&absent,&e),std::string("direct retained removed buffer: ")+name);}
  auto a=g.read(s.frame.output,pixel_bytes(s.frame.output.format)),b=g.read(s.staged_output,pixel_bytes(s.staged_output.format));
  bytes_equal(a,b,"output texture");alpha_one(a,s.frame.output.format);
  if(!mutation){auto broken=b;broken[broken.size()/2]^=1;try{bytes_equal(a,broken,"deliberate output mutation");}
    catch(const std::exception&){mutation=true;}need(mutation,"comparator mutation missed");}
}
void errors(Gpu& g,wgi::NativeCandidate& c,const Scene& s){
  auto reject=[&](WgFrame f,ID3D12Fence* ready,const char* label){ID3D12Fence* end=nullptr;uint64_t value=0;std::string e;
    need(!c.dispatch(f,ready,1,&end,&value,&e),std::string("accepted ")+label);need(!end&&!value,"error completion not cleared");
    std::printf("REJECT %s: %s\n",label,e.c_str());};
  auto f=s.frame;f.output_width++;reject(f,g.f,"output extent");f=s.frame;f.color.width--;reject(f,g.f,"metadata mismatch");
  f=s.frame;f.motion.state=D3D12_RESOURCE_STATE_COPY_DEST|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;reject(f,g.f,"combined write state");
  f=s.frame;f.command_list=g.l;reject(f,g.f,"caller command list");reject(s.frame,nullptr,"missing input-ready fence");
  {Gpu wrong(g.type,true);auto foreign=wrong.texture(s.rw+4,s.rh+2,DXGI_FORMAT_R16G16B16A16_FLOAT);f=s.frame;f.color=foreign;
    reject(f,g.f,"wrong-device texture");reject(s.frame,wrong.f,"wrong-device fence");((ID3D12Resource*)foreign.resource)->Release();}
}
struct QueueTimer {
  Gpu& g;ID3D12QueryHeap* query=nullptr;ID3D12Resource* readback=nullptr;
  ID3D12CommandAllocator* alloc[2]={};ID3D12GraphicsCommandList* list[2]={};uint64_t frequency=0;
  QueueTimer(Gpu& gpu):g(gpu){
    D3D12_QUERY_HEAP_DESC desc={};desc.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;desc.Count=2;
    hr(g.d->CreateQueryHeap(&desc,__uuidof(ID3D12QueryHeap),(void**)&query),"timestamp heap");readback=g.buffer(16,D3D12_HEAP_TYPE_READBACK);
    hr(g.q->GetTimestampFrequency(&frequency),"timestamp frequency");need(frequency!=0,"zero timestamp frequency");
    for(int i=0;i<2;++i){hr(g.d->CreateCommandAllocator(g.type,__uuidof(ID3D12CommandAllocator),(void**)&alloc[i]),"timer allocator");
      hr(g.d->CreateCommandList(0,g.type,alloc[i],nullptr,__uuidof(ID3D12GraphicsCommandList),(void**)&list[i]),"timer list");hr(list[i]->Close(),"timer close");}
  }
  ~QueueTimer(){for(auto*& p:list)release(p);for(auto*& p:alloc)release(p);release(query);release(readback);}
  void stamp(unsigned i){hr(alloc[i]->Reset(),"timer reset allocator");hr(list[i]->Reset(alloc[i],nullptr),"timer reset list");
    // The GPU endpoint precedes ResolveQueryData. Resolve cost belongs to the
    // complete host wall/final wait, never to the reported GPU timestamp span.
    list[i]->EndQuery(query,D3D12_QUERY_TYPE_TIMESTAMP,i);if(i)list[i]->ResolveQueryData(query,D3D12_QUERY_TYPE_TIMESTAMP,0,2,readback,0);
    hr(list[i]->Close(),"timer close stamp");ID3D12CommandList* one=list[i];g.q->ExecuteCommandLists(1,&one);}
  double result(){void* p=nullptr;hr(readback->Map(0,nullptr,&p),"timestamp map");uint64_t q[2];std::memcpy(q,p,16);readback->Unmap(0,nullptr);
    need(q[1]>=q[0],"backward timestamp");return double(q[1]-q[0])*1000.0/double(frequency);}
};
bool benchmark(Gpu& g,wgi::NativeCandidate& direct,wgi::NativeCandidate& staged,Scene& s,double warm,int rounds,int samples,double minimum_seconds){
  std::string e;need(warm>=3,"benchmark warm-up must be at least three seconds");
  // Upload only once before all measurements. Inputs, formats, pre-exposure,
  // motion transforms, jitter and exposure mode stay identical for both routes.
  s.update(1);s.frame.exposure_mode=WG_EXPOSURE_COMPUTE;s.frame.reset=2;
  QueueTimer timer(g);
  bool valid_windows=true;
  std::printf("BENCH_CONFIG warm=%.3f rounds=%d samples=%d color=%u depth=%u motion=%u output=%u pi=%g exposure=%u\n",
    warm,rounds,samples,s.frame.color.format,s.frame.depth.format,s.frame.motion.format,s.frame.output.format,s.frame.pre_exposure,s.frame.exposure_mode);
  for(int round=0;round<rounds;++round)for(int leg=0;leg<2;++leg){
    bool d=((round+leg)%2)==0;auto& c=d?direct:staged;auto f=d?s.frame:s.staged();const char* name=d?"direct":"staged";
    f.reset=2;dispatch(g,c,f);f.reset=0;
    auto begin=Clock::now();unsigned warm_frames=0;
    std::printf("PHASE warm_begin route=%s round=%d utc=%s\n",name,round,utc().c_str());std::fflush(stdout);
    do{dispatch(g,c,f);++warm_frames;}while(ms(begin,Clock::now())<warm*1000);
    double warmed=ms(begin,Clock::now());
    std::printf("PHASE warm_end route=%s round=%d utc=%s elapsed_ms=%.9f frames=%u\n",name,round,utc().c_str(),warmed,warm_frames);
    // Warm duration can require a different number of frames for each route.
    // Reset after warming, outside measurements, to match every measured frame's
    // temporal state/counter/jitter history rather than extrapolate convergence.
    f.reset=2;dispatch(g,c,f);f.reset=0;
    std::printf("PHASE measure_begin route=%s round=%d utc=%s\n",name,round,utc().c_str());
    marker("steady_begin",name,round,warmed/1000.0);
    auto measured_begin=Clock::now();
    for(int sample=0;sample<samples;++sample){
      ID3D12Fence* done=nullptr;uint64_t dv=0;
      auto wall0=Clock::now();timer.stamp(0);auto api0=Clock::now();
      need(c.dispatch(f,g.f,g.serial,&done,&dv,&e),e);auto api1=Clock::now();need(done&&dv,"benchmark completion missing");
      timer.stamp(1);hr(g.q->Signal(g.f,++g.serial),"timer completion signal");auto submitted=Clock::now();
      g.wait(g.f,g.serial);auto completed=Clock::now();
      done->Release();double gpu=timer.result(); // query readback is outside the measured wall span
      std::printf("SAMPLE route=%s round=%d sample=%d native_completion_value=%llu gpu_span_ms=%.9f wall_ms=%.9f dispatch_api_ms=%.9f submission_ms=%.9f final_wait_ms=%.9f utc=%s\n",
        name,round,sample,(unsigned long long)dv,gpu,ms(wall0,completed),ms(api0,api1),ms(wall0,submitted),ms(submitted,completed),utc().c_str());
    }
    double measured_seconds=ms(measured_begin,Clock::now())/1000.0;
    marker("steady_end",name,round,warmed/1000.0);
    std::printf("WINDOW route=%s round=%d actual_samples=%d actual_seconds=%.9f minimum_seconds=%.9f valid_duration=%d\n",
      name,round,samples,measured_seconds,minimum_seconds,int(measured_seconds>=minimum_seconds));
    valid_windows=valid_windows&&measured_seconds>=minimum_seconds;
    need(c.sync(&e),e);std::printf("PHASE measure_end route=%s round=%d utc=%s\n",name,round,utc().c_str());
  }
  std::printf("BENCH_DEFINITION gpu_span=queue_timestamps_before_dispatch_through_final_output_restore_excludes_query_resolve_includes_CPU_submission_gaps wall=start_timestamp_submission_through_query_resolve_and_final_queue_wait submission=host_time_before_final_wait dispatch_api=includes_internal_sync_and_resource_retirement final_wait=query_resolve_fence_wait uploads_and_texture_readbacks=outside_measurements\n");
  return valid_windows;
}
}
int main(int argc,char** argv){
 try{
  for(auto* name:{"wg_native_candidate_create","wg_native_candidate_dispatch","wg_native_candidate_sync","wg_native_candidate_error","wg_native_candidate_destroy"})
    need(GetProcAddress(GetModuleHandleW(nullptr),name)!=nullptr,std::string("missing C ABI export ")+name);
  if(argc==2&&std::string(argv[1])=="--api-only"){std::printf("PASS direct test C ABI exports\n");return 0;}
  if(argc==2&&std::string(argv[1])=="--marker-contract"){
    for(auto* route:{"direct","staged"}){marker("steady_begin",route,0,0,true);Sleep(1100);marker("steady_end",route,0,0,true);}
    std::printf("PASS CPU marker contract fixture; no GPU created and no GPU warm-up claimed\n");return 0;
  }
  need(argc>=9,"usage: native_direct_test BUILD MODEL direct|compute check|matrix|bench RW RH OW OH [frames=8] [cycles=2] [warm=3] [rounds=3] [samples=600] [min-seconds=1] [pre-exposure=scale|original] [hdr=0|1]");
  std::string build=argv[1],model=argv[2],queue=argv[3],mode=argv[4],e;
  need(queue=="direct"||queue=="compute","queue type");need(mode=="check"||mode=="matrix"||mode=="bench","mode");
  int rw=std::stoi(argv[5]),rh=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]);
  int frames=argc>9?std::stoi(argv[9]):8,cycles=argc>10?std::stoi(argv[10]):2;
  double warm=argc>11?std::stod(argv[11]):3;int rounds=argc>12?std::stoi(argv[12]):3,samples=argc>13?std::stoi(argv[13]):600;
  double minimum_seconds=argc>14?std::stod(argv[14]):1;std::string pre_exposure=argc>15?argv[15]:"scale";int hdr=argc>16?std::stoi(argv[16]):0;
  need(std::isfinite(minimum_seconds)&&minimum_seconds>=1,"minimum measurement window must be >=1 second");
  need((pre_exposure=="scale"||pre_exposure=="original")&&(hdr==0||hdr==1),"pre-exposure/HDR options");
  need(rw>0&&rh>0&&ow>0&&oh>0&&frames>0&&cycles>0,"positive test sizes/counts");
  _putenv_s("WG_D3D12_BATCH","0");
  if(mode!="bench")enable_debug();else std::printf("DEBUG disabled for matched performance; separate correctness run required\n");
  Gpu g(queue=="compute"?D3D12_COMMAND_LIST_TYPE_COMPUTE:D3D12_COMMAND_LIST_TYPE_DIRECT);
  adapter_metadata(g);
  _putenv_s("WG_DX12_DEBUG",mode!="bench"&&g.iq?"1":"");
  modules("fresh",true);
  WgRtConfig cfg={};cfg.build_dir=build.c_str();cfg.model_dir=model.c_str();cfg.backend=3;cfg.max_rw=rw;cfg.max_rh=rh;
  cfg.ow=ow;cfg.oh=oh;cfg.depth_inverted=1;cfg.auto_exposure=1;cfg.model_format=1;
  std::string options="model=M;output_filter=original;pre_exposure="+pre_exposure;
  cfg.runtime_options=options.c_str();cfg.hdr=hdr;
  const uint32_t flags=WG_CREATE_DEPTH_INVERTED|WG_CREATE_AUTO_EXPOSURE|(hdr?WG_CREATE_HDR:0);
  std::printf("OPTIONS pre_exposure=%s hdr=%d\n",pre_exposure.c_str(),hdr);
  bool mutation=false;unsigned compared=0,combinations=0;
  for(int cycle=0;cycle<(mode=="bench"?1:cycles);++cycle){
    auto direct=wgi::NativeCandidate::create(g.d,g.q,cfg,flags,&e,true);need(!!direct,e);
    auto staged=wgi::NativeCandidate::create(g.d,g.q,cfg,flags,&e,false);need(!!staged,e);
    modules("candidates-created",true);std::printf("PROVIDERS direct=%s staged=%s\n",direct->provider(),staged->provider());
    if(mode=="bench"){
      Scene s(g,rw,rh,ow,oh,DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R32_FLOAT,DXGI_FORMAT_R16G16_FLOAT,DXGI_FORMAT_R16G16B16A16_FLOAT);
      bool valid=benchmark(g,*direct,*staged,s,warm,rounds,samples,minimum_seconds);need(direct->sync(&e)&&staged->sync(&e),e);
      // Keep the actual measured contexts/resources alive through the final PDH
      // sampling edge. This fixed tail is outside every steady window; telemetry
      // must still qualify its real intervals without treating missing data as idle.
      if(const char* marker_dir=std::getenv("WG_GPU_EVIDENCE_MARKER_DIR");marker_dir&&*marker_dir){
        auto begin=Clock::now();
        std::printf("TAIL context_alive_begin utc=%s monotonic_ns=%llu requested_seconds=1.0\n",utc().c_str(),(unsigned long long)qpc_ns());std::fflush(stdout);
        Sleep(1000);
        std::printf("TAIL context_alive_end utc=%s monotonic_ns=%llu actual_seconds=%.9f\n",utc().c_str(),(unsigned long long)qpc_ns(),ms(begin,Clock::now())/1000.0);std::fflush(stdout);
      }
      need(valid,"short measurement window; all samples retained; rerun BOTH routes with a larger common sample count");
    }else{
      const DXGI_FORMAT colors[]={DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT,DXGI_FORMAT_R11G11B10_FLOAT};
      const DXGI_FORMAT depths[]={DXGI_FORMAT_R32_FLOAT,DXGI_FORMAT_R16_FLOAT};
      const DXGI_FORMAT motions[]={DXGI_FORMAT_R16G16_FLOAT,DXGI_FORMAT_R32G32_FLOAT};
      const DXGI_FORMAT outputs[]={DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R32G32B32A32_FLOAT};
      for(int ci=0;ci<(mode=="matrix"?3:1);++ci)for(int di=0;di<(mode=="matrix"?2:1);++di)
      for(int mi=0;mi<(mode=="matrix"?2:1);++mi)for(int oi=0;oi<(mode=="matrix"?2:1);++oi){
        Scene s(g,rw,rh,ow,oh,colors[ci],depths[di],motions[mi],outputs[oi]);++combinations;
        std::printf("FORMAT color=%u depth=%u motion=%u output=%u cycle=%d combination=%u\n",colors[ci],depths[di],motions[mi],outputs[oi],cycle,combinations);
        if(combinations==1)errors(g,*direct,s);
        for(int frame=0;frame<frames;++frame){
          s.update(frame);std::printf("FRAME_BEGIN cycle=%d combination=%u frame=%d mode=%u pi=%g reset=%u utc=%s\n",cycle,combinations,frame,s.frame.exposure_mode,s.frame.pre_exposure,s.frame.reset,utc().c_str());
          dispatch(g,*direct,s.frame,frame==0&&combinations==1);dispatch(g,*staged,s.staged());compare_frame(g,*direct,*staged,s,mutation);++compared;
          std::printf("PASS cycle=%d combination=%u frame=%d output_bytes=%zu\n",cycle,combinations,frame,size_t(ow)*oh*pixel_bytes(outputs[oi]));g.debug();
        }
        need(direct->sync(&e)&&staged->sync(&e),e);
        if(mode=="check"){
          // Final token covers direct output and state restoration on teardown.
          ID3D12Fence* done=nullptr;uint64_t value=0;need(direct->dispatch(s.frame,g.f,g.serial,&done,&value,&e),e);
          direct.reset();need(done&&done->GetCompletedValue()>=value&&done->GetCompletedValue()!=UINT64_MAX,"outstanding direct teardown");done->Release();
        }
      }
    }
    direct.reset();staged.reset();if(mode!="bench")g.debug();modules("after-destroy",true);
  }
  if(mode=="bench")std::printf("PASS native direct benchmark queue=%s rounds=%d samples=%d\n",queue.c_str(),rounds,samples);
  else std::printf("PASS native direct check queue=%s frames_compared=%u format_combinations=%u cycles=%d mutation=%d\n",queue.c_str(),compared,combinations,cycles,int(mutation));
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
