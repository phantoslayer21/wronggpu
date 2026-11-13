// SPDX-License-Identifier: MIT
// Original packaged-DLL proof, derived from native_game_test.cpp synthetic helpers.
// The actual candidate is loaded from the requested DLL and used only through
// its opaque C ABI. The linked C++ candidate is used only as a staged oracle.
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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include "../../converter/src/sha256.h"

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
std::vector<unsigned char> read(wgrt::Runtime* r,const char* name){
  uintptr_t p=0;size_t n=0;need(wg_rt_buffer(r,name,&p,&n)==0,name);if(std::string(name)=="exposure.pair")n=20;
  std::vector<unsigned char> v(n);auto* d=wgrt::runtime_device(r);need(d->download(v.data(),p,n),d->err());return v;
}
void bytes_equal(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b,const std::string& where,size_t active=0){
  need(a.size()==b.size(),where+" size mismatch");size_t n=active?active:a.size();
  for(size_t i=0;i<n;++i)if(a[i]!=b[i])throw std::runtime_error(where+" first byte differs at "+std::to_string(i)+" native="+std::to_string(a[i])+" oracle="+std::to_string(b[i]));
}
void enable_debug(){
  auto dll=LoadLibraryW(L"d3d12.dll");using Debug=HRESULT(WINAPI*)(REFIID,void**);
  auto get=(Debug)GetProcAddress(dll,"D3D12GetDebugInterface");ID3D12Debug* d=nullptr;
  if(get&&SUCCEEDED(get(__uuidof(ID3D12Debug),(void**)&d))){d->EnableDebugLayer();d->Release();std::printf("DEBUG enabled\n");}
  else std::printf("DEBUG unavailable at startup\n");
}
}
namespace {
std::string file_sha(const std::filesystem::path& p) {
  std::ifstream in(p,std::ios::binary);need(in.good(),"hash open failed");
  wgc::Sha256 s;char block[65536];
  while(in){in.read(block,sizeof(block));s.update(block,(size_t)in.gcount());}
  need(in.eof(),"hash read failed");return wgc::to_hex(s.finish());
}
struct Dll {
  HMODULE module=nullptr;
  decltype(&wg_native_candidate_create) create=nullptr;
  decltype(&wg_native_candidate_dispatch) dispatch=nullptr;
  decltype(&wg_native_candidate_sync) sync=nullptr;
  decltype(&wg_native_candidate_error) error=nullptr;
  decltype(&wg_native_candidate_destroy) destroy=nullptr;
  std::filesystem::path path;std::string digest;
  explicit Dll(const char* requested) {
    path=std::filesystem::canonical(requested);digest=file_sha(path);
    module=LoadLibraryW(path.c_str());need(module!=nullptr,"LoadLibrary packaged DLL failed");
    wchar_t loaded[32768];DWORD n=GetModuleFileNameW(module,loaded,32768);
    need(n&&n<32768,"GetModuleFileName packaged DLL failed");
    need(std::filesystem::equivalent(path,std::filesystem::path(loaded)),"loaded DLL is not requested file");
    std::printf("DLL_PATH %ls\n",loaded);std::printf("DLL_SHA256 %s\n",digest.c_str());
    load(create,"wg_native_candidate_create");load(dispatch,"wg_native_candidate_dispatch");
    load(sync,"wg_native_candidate_sync");load(error,"wg_native_candidate_error");load(destroy,"wg_native_candidate_destroy");
  }
  template<class T>void load(T& f,const char* name) {
    auto address=GetProcAddress(module,name);need(address!=nullptr,std::string("DLL export missing: ")+name);
    HMODULE owner=nullptr;
    need(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(address),&owner)!=FALSE&&owner==module,std::string("export is not owned by requested DLL: ")+name);
    f=reinterpret_cast<T>(address);std::printf("DLL_EXPORT %s verified_owner=1\n",name);
  }
  void unchanged(){need(file_sha(path)==digest,"packaged DLL changed during process");std::printf("DLL_UNCHANGED %s\n",digest.c_str());}
  ~Dll(){if(module)FreeLibrary(module);}
};
struct OpaqueCandidate {
  Dll& api;void* handle=nullptr;
  explicit OpaqueCandidate(Dll& a):api(a){}
  ~OpaqueCandidate(){if(handle){int code=api.destroy(handle);if(code)std::fprintf(stderr,"DLL cleanup failed code=%d\n",code);}}
  std::string error(){const char* p=api.error(handle);return p?p:"DLL returned null error string";}
  void close(){void* old=handle;handle=nullptr;need(api.destroy(old)==0,"DLL destroy failed");}
};
}
int main(int argc,char** argv){
 try {
  need(argc==3||argc>=10,"usage: native_direct_dll_test DLL --api-only | DLL BUILD MODEL direct|compute RW RH OW OH FRAMES [half|float]");
  modules("fresh",true);Dll api(argv[1]);modules("loaded-dll",true);
  void* invalid=(void*)1;char api_error[256]={};
  need(api.create(nullptr,nullptr,nullptr,0,&invalid,api_error,sizeof(api_error))!=0&&!invalid&&api_error[0],"DLL invalid create");
  need(api.sync(nullptr)!=0&&api.destroy(nullptr)==0,"DLL null lifecycle");
  need(api.dispatch(nullptr,nullptr,nullptr,0,nullptr,nullptr)!=0,"DLL invalid dispatch");
  need(api.error(nullptr)!=nullptr,"DLL null error accessor");
  std::printf("PASS packaged DLL C ABI null/error paths\n");
  if(argc==3){need(std::string(argv[2])=="--api-only","unknown short invocation");api.unchanged();return 0;}
  std::string build=argv[2],model=argv[3],qname=argv[4],err,format=argc>10?argv[10]:"half";
  int rw=std::stoi(argv[5]),rh=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=std::stoi(argv[8]),frames=std::stoi(argv[9]);
  need(qname=="direct"||qname=="compute","queue choice");need(format=="half"||format=="float","output format choice");
  need(rw>0&&rh>0&&ow>0&&oh>0&&frames>=6,"positive shapes and at least six frames required");
  _putenv_s("WG_D3D12_BATCH","0");enable_debug();
  Gpu g(qname=="compute"?D3D12_COMMAND_LIST_TYPE_COMPUTE:D3D12_COMMAND_LIST_TYPE_DIRECT);
  need(g.iq!=nullptr,"D3D12 debug layer required for restored resource-state proof");
  _putenv_s("WG_DX12_DEBUG","1");
  WgRtConfig cfg={};cfg.build_dir=build.c_str();cfg.model_dir=model.c_str();cfg.backend=3;
  cfg.max_rw=rw;cfg.max_rh=rh;cfg.ow=ow;cfg.oh=oh;cfg.depth_inverted=1;cfg.auto_exposure=1;cfg.model_format=1;
  cfg.runtime_options="model=M;output_filter=original;pre_exposure=scale";
  const uint32_t flags=WG_CREATE_DEPTH_INVERTED|WG_CREATE_AUTO_EXPOSURE;
  OpaqueCandidate candidate(api);
  int created=api.create(g.d,g.q,&cfg,flags,&candidate.handle,api_error,sizeof(api_error));
  need(created==0&&candidate.handle!=nullptr,std::string("DLL create: ")+api_error);
  auto reference=wgi::NativeCandidate::create(g.d,g.q,cfg,flags,&err,false);need(!!reference,err);
  std::printf("REFERENCE_PROVIDER %s\n",reference->provider());modules("created",true);
  WgFrame f={};f.struct_size=sizeof(f);f.command_queue=g.q;f.command_list_type=(uint32_t)g.type;
  f.render_width=rw;f.render_height=rh;f.output_width=ow;f.output_height=oh;f.render_x=2;f.render_y=1;
  f.pre_exposure=f.exposure_value=1;
  f.color=g.texture(rw+4,rh+2,DXGI_FORMAT_R32G32B32A32_FLOAT);f.depth=g.texture(rw+4,rh+2,DXGI_FORMAT_R32_FLOAT);
  f.motion=g.texture(rw+4,rh+2,DXGI_FORMAT_R32G32_FLOAT);f.exposure=g.texture(1,1,DXGI_FORMAT_R32_FLOAT);
  auto out_format=format=="half"?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R32G32B32A32_FLOAT;
  size_t pixel=format=="half"?8:16;
  f.output=g.texture(ow,oh,out_format,true);auto reference_output=g.texture(ow,oh,out_format,true);
  bool mutation_caught=false;
  for(int frame=0;frame<frames;++frame){
    const int tw=rw+4,th=rh+2;std::vector<float> color(size_t(tw)*th*4),depth(size_t(tw)*th),motion(size_t(tw)*th*2);
    for(int y=0;y<th;++y)for(int x=0;x<tw;++x){size_t i=size_t(y)*tw+x;
      bool foreground=((x+frame*7)%tw)<tw/3&&y>th/4&&y<th*3/4;
      color[i*4]=float((x+3*frame)%61)*0.015625f+(foreground?0.25f:0);
      color[i*4+1]=float((y+5*frame)%47)*0.015625f;color[i*4+2]=foreground?0.75f:0.125f;color[i*4+3]=1;
      depth[i]=foreground?0.75f:0.25f;motion[i*2]=foreground?-7.0f:float((x%9)-4)*0.03125f;
      motion[i*2+1]=float((y%7)-3)*0.0625f;
    }
    g.upload(f.color,color.data(),16);g.upload(f.depth,depth.data(),4);g.upload(f.motion,motion.data(),8);
    float ev=0.5f+frame*0.125f;g.upload(f.exposure,&ev,4);
    f.pre_exposure=frame%2?1.375f:1.0f;f.exposure_mode=frame%3;f.exposure_value=ev;
    f.motion_scale[0]=0.625f;f.motion_scale[1]=1.375f;f.motion_offset[0]=0.03125f;f.motion_offset[1]=-0.0625f;
    f.jitter[0]=float((frame%4)-2)*0.125f;f.jitter[1]=float((frame%3)-1)*0.1875f;
    f.reset=frame==0?2:(frame==3||frame==5?1:0);
    auto input_state=frame%2?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_COMMON;
    auto output_state=frame%2?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    g.begin();
    for(auto* t:{&f.color,&f.depth,&f.motion,&f.exposure}){g.barrier((ID3D12Resource*)t->resource,(D3D12_RESOURCE_STATES)t->state,input_state);t->state=input_state;}
    for(auto* t:{&f.output,&reference_output}){g.barrier((ID3D12Resource*)t->resource,(D3D12_RESOURCE_STATES)t->state,output_state);t->state=output_state;}
    g.submit();
    ID3D12Fence* delayed=nullptr;
    if(frame==0)hr(g.d->CreateFence(0,D3D12_FENCE_FLAG_NONE,__uuidof(ID3D12Fence),(void**)&delayed),"delayed-ready fence");
    void* token=nullptr;uint64_t value=0;
    int code=api.dispatch(candidate.handle,&f,delayed?delayed:g.f,delayed?1:g.serial,&token,&value);
    auto* done=(ID3D12Fence*)token;bool premature=delayed&&done&&done->GetCompletedValue()>=value;
    if(delayed){hr(delayed->Signal(1),"release delayed producer");delayed->Release();}
    need(code==0,candidate.error());need(done&&value,"DLL completion missing");need(!premature,"DLL completed before input-ready");
    g.wait(done,value);done->Release();need(api.sync(candidate.handle)==0,candidate.error());
    if(frame==0)std::printf("PASS DLL delayed input-ready fence held final completion\n");
    WgFrame rf=f;rf.output=reference_output;done=nullptr;value=0;
    need(reference->dispatch(rf,g.f,g.serial,&done,&value,&err),err);g.wait(done,value);done->Release();need(reference->sync(&err),err);
    auto actual=g.read(f.output,pixel),wanted=g.read(reference_output,pixel);bytes_equal(actual,wanted,"DLL output texture");
    if(!mutation_caught){auto bad=wanted;bad[bad.size()/2]^=1;try{bytes_equal(actual,bad,"deliberate DLL comparator mutation");}
      catch(const std::exception&){mutation_caught=true;}need(mutation_caught,"DLL comparator missed mutation");}
    g.debug();std::printf("PASS DLL frame=%d mode=%u pi=%g reset=%u output_bytes=%zu input_state=%u output_state=%u\n",
      frame,f.exposure_mode,f.pre_exposure,f.reset,actual.size(),(unsigned)input_state,(unsigned)output_state);
  }
  candidate.close();reference.reset();g.debug();modules("after-destroy",true);api.unchanged();
  for(auto* t:{&f.color,&f.depth,&f.motion,&f.exposure,&f.output,&reference_output})((ID3D12Resource*)t->resource)->Release();
  std::printf("PASS packaged direct DLL queue=%s shape=%dx%d-%dx%d frames=%d format=%s mutation=%d\n",
    qname.c_str(),rw,rh,ow,oh,frames,format.c_str(),int(mutation_caught));
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
