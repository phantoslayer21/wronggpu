// SPDX-License-Identifier: MIT
// Original private synthetic game-device test. No game files or learned tensors.
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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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
int main(int argc,char** argv){
 try {
  for(const char* name:{"wg_native_candidate_create","wg_native_candidate_dispatch","wg_native_candidate_sync",
                         "wg_native_candidate_error","wg_native_candidate_destroy"})
    need(GetProcAddress(GetModuleHandleW(nullptr),name)!=nullptr,std::string("missing C ABI export: ")+name);
  void* invalid=(void*)1;char api_error[256]={};
  need(wg_native_candidate_create(nullptr,nullptr,nullptr,0,&invalid,api_error,sizeof(api_error))!=0 && !invalid && api_error[0],"C ABI invalid create");
  need(wg_native_candidate_sync(nullptr)!=0 && wg_native_candidate_destroy(nullptr)==0,"C ABI null lifecycle");
  need(wg_native_candidate_dispatch(nullptr,nullptr,nullptr,0,nullptr,nullptr)!=0,"C ABI invalid dispatch");
  std::printf("PASS C ABI exports and null/error paths\n");
  if(argc==2 && std::string(argv[1])=="--api-only")return 0;
  need(argc>=9,"usage: native_game_test BUILD MODEL direct|compute native|hip RW RH OW OH [frames=8] [cycles=2]");
  std::string build=argv[1],model=argv[2],qname=argv[3],oracle=argv[4],err;
  need(qname=="direct"||qname=="compute","queue choice");need(oracle=="native"||oracle=="hip","oracle choice");
  int rw=std::stoi(argv[5]),rh=std::stoi(argv[6]),ow=std::stoi(argv[7]),oh=argc>8?std::stoi(argv[8]):ow;
  int frames=argc>9?std::stoi(argv[9]):8,cycles=argc>10?std::stoi(argv[10]):2;
  // A pending-ready-fence test intentionally queues a whole frame before the
  // producer signals; avoid unrelated per-dispatch batching limits in this test.
  _putenv_s("WG_D3D12_BATCH","0");
  enable_debug();Gpu g(qname=="compute"?D3D12_COMMAND_LIST_TYPE_COMPUTE:D3D12_COMMAND_LIST_TYPE_DIRECT);
  WgRtConfig cfg={};cfg.build_dir=build.c_str();cfg.model_dir=model.c_str();cfg.backend=3;cfg.max_rw=rw;cfg.max_rh=rh;
  cfg.ow=ow;cfg.oh=oh;cfg.depth_inverted=1;cfg.auto_exposure=1;cfg.model_format=1;
  cfg.runtime_options="model=M;output_filter=original;pre_exposure=scale";
  const uint32_t flags=WG_CREATE_DEPTH_INVERTED|WG_CREATE_AUTO_EXPOSURE;
  WgFrame f={};f.struct_size=sizeof(f);f.command_queue=g.q;f.command_list_type=(uint32_t)g.type;
  f.render_width=rw;f.render_height=rh;f.output_width=ow;f.output_height=oh;
  f.pre_exposure=1.0f;f.exposure_value=1.0f;
  // A nonzero active origin verifies region handling independently from network shapes.
  f.render_x=2;f.render_y=1;
  f.color=g.texture(rw+4,rh+2,DXGI_FORMAT_R32G32B32A32_FLOAT);f.depth=g.texture(rw+4,rh+2,DXGI_FORMAT_R32_FLOAT);
  f.motion=g.texture(rw+4,rh+2,DXGI_FORMAT_R32G32_FLOAT);f.exposure=g.texture(1,1,DXGI_FORMAT_R32_FLOAT);
  f.output=g.texture(ow,oh,DXGI_FORMAT_R16G16B16A16_FLOAT,true);auto reference_output=g.texture(ow,oh,DXGI_FORMAT_R16G16B16A16_FLOAT,true);
  wgi::Converter reference_converter;need(reference_converter.init(g.d,&err),err);
  modules("fresh",true);
  bool mutation_caught=false;
  for(int cycle=0;cycle<cycles;++cycle){
    // The harness enabled the layer before the first device. Borrowed creation
    // observes its queue; standalone reference creation must not re-enable a
    // debug layer after devices exist or clear the shared diagnostic stream.
    _putenv_s("WG_DX12_DEBUG",g.iq?"1":"");
    cfg.backend=3;auto candidate=wgi::NativeCandidate::create(g.d,g.q,cfg,flags,&err);need(!!candidate,err);
    std::printf("PROVIDER %s cycle=%d\n",candidate->provider(),cycle);
    if(oracle=="native"||cycle==0)modules("borrowed-created",true);
    _putenv_s("WG_DX12_DEBUG","");
    cfg.backend=oracle=="hip"?0:3;
    auto* reference=wgrt::runtime_create(cfg,nullptr,&err,false);need(reference!=nullptr,err);
    auto* rdev=wgrt::runtime_device(reference);wgrt::DevPtr ex=0,rc=0,rd=0,rm=0,ro=0;
    need(rdev->alloc(&ex,4)&&rdev->alloc(&rc,size_t(rw)*rh*16)&&rdev->alloc(&rd,size_t(rw)*rh*4)&&
      rdev->alloc(&rm,size_t(rw)*rh*8)&&rdev->alloc(&ro,size_t(ow)*oh*12),rdev->err());
    auto reference_read=[&](const char* name){
      if(std::string(name)!="filter.out")return read(reference,name);
      std::vector<unsigned char> v(size_t(ow)*oh*12);need(rdev->download(v.data(),ro,v.size()),rdev->err());return v;
    };
    // Every error is rejected before conversion and must leave the candidate usable.
    auto reject=[&](WgFrame bad,ID3D12Fence* ready,uint64_t v,const char* what){ID3D12Fence* done=nullptr;uint64_t dv=0;
      need(!candidate->dispatch(bad,ready,v,&done,&dv,&err),std::string("accepted ")+what);need(!done&&!dv,"error returned completion");
      std::printf("REJECT %s: %s\n",what,err.c_str());};
    WgFrame bad=f;bad.output_width++;reject(bad,g.f,1,"output extent");bad=f;bad.color.width--;reject(bad,g.f,1,"texture metadata");
    bad=f;bad.motion.state=D3D12_RESOURCE_STATE_COPY_DEST|D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;reject(bad,g.f,1,"combined write state");
    bad=f;bad.output=f.color;reject(bad,g.f,1,"output flags/alias");bad=f;bad.command_list=g.l;reject(bad,g.f,1,"caller command list");
    reject(f,nullptr,1,"missing input-ready fence");
    {Gpu other(g.type,true);bad=f;auto foreign=other.texture(rw+4,rh+2,DXGI_FORMAT_R32G32B32A32_FLOAT);bad.color=foreign;
      reject(bad,g.f,1,"wrong-device texture");reject(f,other.f,1,"wrong-device fence");((ID3D12Resource*)foreign.resource)->Release();}
    for(int frame=0;frame<frames;++frame){
      const int tw=rw+4,th=rh+2;std::vector<float> col(size_t(tw)*th*4),depth(size_t(tw)*th),motion(size_t(tw)*th*2);
      for(int y=0;y<th;++y)for(int x=0;x<tw;++x){size_t i=size_t(y)*tw+x;
        bool foreground=((x+frame*7)%tw)<tw/3 && y>th/4&&y<th*3/4;
        col[i*4]=float((x+3*frame)%61)*0.015625f+(foreground?0.25f:0);
        col[i*4+1]=float((y+5*frame)%47)*0.015625f;col[i*4+2]=foreground?0.75f:0.125f;col[i*4+3]=1;
        depth[i]=foreground?0.75f:0.25f;motion[i*2]=foreground?-7.0f:float((x%9)-4)*0.03125f;
        motion[i*2+1]=float((y%7)-3)*0.0625f;}
      g.upload(f.color,col.data(),16);g.upload(f.depth,depth.data(),4);g.upload(f.motion,motion.data(),8);
      float ev=0.5f+frame*0.125f;g.upload(f.exposure,&ev,4);
      f.pre_exposure=(frame%2)?1.375f:1.0f;f.exposure_mode=(uint32_t)(frame%3);f.exposure_value=ev;
      f.motion_scale[0]=0.625f;f.motion_scale[1]=1.375f;f.motion_offset[0]=0.03125f;f.motion_offset[1]=-0.0625f;
      f.jitter[0]=float((frame%4)-2)*0.125f;f.jitter[1]=float((frame%3)-1)*0.1875f;
      f.reset=frame==0?2:(frame==3||frame==5?1:0);
      auto desired=frame%2?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      g.begin();g.barrier((ID3D12Resource*)f.output.resource,(D3D12_RESOURCE_STATES)f.output.state,desired);g.submit();f.output.state=desired;
      ID3D12Fence* done=nullptr;uint64_t dv=0;
      ID3D12Fence* delayed=nullptr;
      if(frame==0)hr(g.d->CreateFence(0,D3D12_FENCE_FLAG_NONE,__uuidof(ID3D12Fence),(void**)&delayed),"delayed-ready fence");
      bool dispatched=candidate->dispatch(f,delayed?delayed:g.f,delayed?1:g.serial,&done,&dv,&err);
      bool premature=delayed&&done&&done->GetCompletedValue()>=dv;
      // Signal even after a failed dispatch so teardown cannot be stranded by
      // this deliberate producer delay. No dependency on candidate GPU output.
      if(delayed){hr(delayed->Signal(1),"release delayed producer");delayed->Release();}
      need(dispatched,err);need(done&&dv,"completion missing");need(!premature,"completion signalled before input-ready");
      if(frame==0)std::printf("PASS delayed input-ready fence held final completion\n");
      // The final token must cover output writes and restored state, not only neural dispatch.
      g.wait(done,dv);done->Release();need(candidate->sync(&err),err);
      for(const char* name:{"input.color","input.depth","input.motion"}){
        std::vector<unsigned char> bytes;need(candidate->read_buffer(name,&bytes,&err),err);
        auto ptr=std::string(name)=="input.color"?rc:std::string(name)=="input.depth"?rd:rm;
        need(rdev->upload(ptr,bytes.data(),bytes.size()),rdev->err());
      }
      need(rdev->upload(ex,&ev,4),rdev->err());WgRtFrame rf={};rf.rw=rw;rf.rh=rh;rf.mv_scale[0]=rf.mv_scale[1]=1;
      rf.jitter[0]=f.jitter[0];rf.jitter[1]=f.jitter[1];rf.exposure=ev;rf.pre_exposure=f.pre_exposure;rf.reset=f.reset;
      wgrt::FrameIo io;io.color=rc;io.depth=rd;io.motion=rm;io.out=ro;io.exposure_mode=(int)f.exposure_mode;io.expo_src=ex;
      need(wgrt::runtime_frame(reference,rf,&io),wgrt::runtime_error(reference));need(rdev->sync(),rdev->err());
      const char* names[]={"exposure.pair","enc0.features","enc0.m_near","enc0.m_far","enc0.m_center","enc1.skip","enc1.merged",
        "enc2.skip","enc2.merged","b5.out","dec2.out","dec0.A","dec0.C","dec0.latent","post.history","post.reduced","filter.out"};
      for(auto* name:names){std::vector<unsigned char> bytes;need(candidate->read_buffer(name,&bytes,&err),err);auto ref=reference_read(name);
        bytes_equal(bytes,ref,name,std::string(name)=="dec0.C"?size_t(rw)*rh*4:0);}
      if(!f.reset)for(auto* name:{"state.latent_prev","state.motion_history","state.history_prev"}){
        std::vector<unsigned char> bytes;need(candidate->read_buffer(name,&bytes,&err),err);bytes_equal(bytes,read(reference,name),name);}
      auto expected=reference_read("filter.out");WgFrame of=f;of.output=reference_output;g.output(reference_converter,of,expected);
      auto actual=g.read(f.output,8),wanted=g.read(reference_output,8);bytes_equal(actual,wanted,"output texture");
      if(!mutation_caught){auto broken=wanted;broken[broken.size()/2]^=1;try{bytes_equal(actual,broken,"deliberate output mutation");}
        catch(const std::exception&){mutation_caught=true;}need(mutation_caught,"comparator mutation missed");}
      std::printf("PASS cycle=%d frame=%d mode=%u pi=%g reset=%u bytes=%zu\n",cycle,frame,f.exposure_mode,f.pre_exposure,f.reset,actual.size());
      g.debug();
    }
    // Destroy with a valid submitted frame still outstanding; the object must drain all final consumers.
    ID3D12Fence* outstanding=nullptr;uint64_t last=0;
    need(candidate->dispatch(f,g.f,g.serial,&outstanding,&last,&err),err);candidate.reset();
    need(outstanding->GetCompletedValue()>=last&&outstanding->GetCompletedValue()!=UINT64_MAX,"destroy failed to drain output");outstanding->Release();
    for(auto ptr:{ex,rc,rd,rm,ro})rdev->free(ptr);wgrt::runtime_destroy(reference);g.debug();
    if(oracle=="native")modules("after-destroy",true);
  }
  for(auto* t:{&f.color,&f.depth,&f.motion,&f.exposure,&f.output,&reference_output})((ID3D12Resource*)t->resource)->Release();
  std::printf("PASS native game candidate %s %dx%d -> %dx%d oracle=%s frames=%d cycles=%d mutation=%d\n",qname.c_str(),rw,rh,ow,oh,oracle.c_str(),frames,cycles,(int)mutation_caught);
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
