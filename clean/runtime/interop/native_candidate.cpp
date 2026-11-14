// SPDX-License-Identifier: MIT
#include "native_candidate.h"
#include "../device_d3d12.h"
#include "../runtime_core.h"
#include "texture_shape.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>

namespace wgi {
namespace {
bool fail(std::string* e, const std::string& s) { if (e) *e=s; return false; }
template<class T> void release(T*& p) { if(p) p->Release(); p=nullptr; }
bool same(IUnknown* a, IUnknown* b) {
  IUnknown *x=nullptr,*y=nullptr;
  if (a) a->QueryInterface(__uuidof(IUnknown),(void**)&x);
  if (b) b->QueryInterface(__uuidof(IUnknown),(void**)&y);
  bool ok=x && y && x==y; release(x); release(y); return ok;
}

bool on_device(ID3D12DeviceChild* c, ID3D12Device* d) {
  ID3D12Device* own=nullptr;
  if(!c || FAILED(c->GetDevice(__uuidof(ID3D12Device),(void**)&own))) return false;
  bool ok=same(own,d); release(own); return ok;
}
}
struct NativeCandidate::Impl {
  ID3D12Device* device=nullptr;
  ID3D12CommandQueue* queue=nullptr;
  wgrt::Runtime* rt=nullptr;
  wgrt::Device* dev=nullptr;
  wgrt::NativeD3D12Ext* ext=nullptr;
  Converter converter;
  InputBuffers in;
  BufferRange out;
  wgrt::DevPtr color=0,depth=0,motion=0,result=0,expo=0;
  uint32_t flags=0;
  int rw=0,rh=0,ow=0,oh=0;
  int last_rw=0,last_rh=0;
  bool poisoned=false;
  bool direct_resources=true;
  bool direct_bound=false, needs_restore=false;
  WgFrame restore_frame={};
  NativeFrameInventory inventory;
  std::vector<ID3D12Resource*> pending;
  ~Impl() {
    for(auto* r:pending) r->Release();
    if(rt) wgrt::runtime_destroy(rt);
    release(queue); release(device);
  }
  bool sync(std::string* e) {
    // A late shader/recording error (or host exception) can leave a partially
    // recorded direct frame. Restore its caller states before draining it.
    // Failure to record/complete this cleanup keeps the whole object quarantined.
    if(needs_restore) {
      if(!ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        [&](ID3D12GraphicsCommandList* list,std::string&){
          texture_states(list,restore_frame,true);needs_restore=false;return true;
        },e,"direct_error_restore"))return false;
    }
    if(direct_bound) {
      if(!ext->end_direct_frame(e))return false;
      direct_bound=false;
    }
    if(ext && !ext->sync(e)) return false;
    for(auto* r:pending) r->Release();
    pending.clear(); return true;
  }
  bool allocate(size_t need, wgrt::DevPtr* ptr, BufferRange* dst, std::string* e) {
    if(!dev->alloc(ptr,need))return fail(e,dev->err());
    dst->resource=ext->arena_resource();
    return ext->arena_offset(*ptr,need,dst->offset,e);
  }
  bool texture(const WgTexture& t, bool output, const char* what, std::string* e) {
    auto* r=static_cast<ID3D12Resource*>(t.resource);
    if(!r || !on_device(r,device)) return fail(e,std::string(what)+": resource missing or wrong device");
    auto d=r->GetDesc();
    // any number of mips: mip 0 is read or written (texture_shape.h, 0.6.0 follow-up)
    if(!wgi::texture_shape_ok(d) || d.Width!=t.width || d.Height!=t.height ||
       d.Format!=(DXGI_FORMAT)t.format || !t.width || !t.height)
      return fail(e,std::string(what)+": requires a matching single-sample non-array typed 2D texture (any number of mips: mip 0 is used)");
    if(output ? !(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) : (d.Flags&D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
      return fail(e,std::string(what)+": incompatible resource flags");
    const uint32_t shared=D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
       D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_COPY_DEST;
    const uint32_t allowed=shared | (queue->GetDesc().Type==D3D12_COMMAND_LIST_TYPE_DIRECT ?
       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_RENDER_TARGET : 0);
    const uint32_t write=D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_COPY_DEST |
       D3D12_RESOURCE_STATE_RENDER_TARGET;
    if((t.state&~allowed) || ((t.state&write) && ((t.state&(t.state-1))!=0)))
      return fail(e,std::string(what)+": unsupported or combined write state for supplied queue");
    if(((t.state&D3D12_RESOURCE_STATE_UNORDERED_ACCESS) && !(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) ||
       ((t.state&D3D12_RESOURCE_STATE_RENDER_TARGET) && !(d.Flags&D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)))
      return fail(e,std::string(what)+": stated write state incompatible with resource flags");
    D3D12_FEATURE_DATA_FORMAT_SUPPORT s={};s.Format=(DXGI_FORMAT)t.format;
    if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT,&s,sizeof(s))) ||
       (output ? !(s.Support2&D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) : !(s.Support1&D3D12_FORMAT_SUPPORT1_SHADER_LOAD)))
      return fail(e,std::string(what)+": required typed format support unavailable");
    return true;
  }
  bool validate(const WgFrame& f, ID3D12Fence* ready,uint64_t v,std::string* e) {
    if(poisoned) return fail(e,"candidate is poisoned by a prior recording/submission failure");
    if(f.struct_size!=sizeof(WgFrame) || f.command_list || (f.command_queue && f.command_queue!=queue))
      return fail(e,"private candidate requires current WgFrame, null caller list and matching supplied queue");
    if(!ready || !v || !on_device(ready,device)) return fail(e,"same-device input-ready fence and nonzero value required");
    if(!f.render_width || !f.render_height || f.render_width>(uint32_t)rw || f.render_height>(uint32_t)rh ||
       f.output.width!=(uint32_t)ow || f.output.height!=(uint32_t)oh ||
       f.output_width!=(uint32_t)ow || f.output_height!=(uint32_t)oh)
      return fail(e,"frame extents outside configured capacity");
    if(f.exposure_mode>WG_EXPOSURE_COMPUTE || !std::isfinite(f.pre_exposure) || f.pre_exposure<=0 ||
       !std::isfinite(f.exposure_value) || f.reset>2) return fail(e,"invalid exposure or reset");
    for(int i=0;i<2;++i) if(!std::isfinite(f.jitter[i]) || !std::isfinite(f.motion_scale[i]) || !std::isfinite(f.motion_offset[i]))
      return fail(e,"nonfinite jitter or motion transform");
    if(!texture(f.color,false,"color",e)||!texture(f.depth,false,"depth",e)||!texture(f.motion,false,"motion",e)||
       !texture(f.output,true,"output",e)) return false;
    if(f.color.format!=DXGI_FORMAT_R16G16B16A16_FLOAT && f.color.format!=DXGI_FORMAT_R32G32B32A32_FLOAT &&
       f.color.format!=DXGI_FORMAT_R11G11B10_FLOAT) return fail(e,"unsupported private color format");
    if(f.depth.format!=DXGI_FORMAT_R32_FLOAT && f.depth.format!=DXGI_FORMAT_R16_FLOAT) return fail(e,"unsupported private depth format");
    if(f.motion.format!=DXGI_FORMAT_R16G16_FLOAT && f.motion.format!=DXGI_FORMAT_R32G32_FLOAT)
      return fail(e,"unsupported private motion format");
    if(f.output.format!=DXGI_FORMAT_R16G16B16A16_FLOAT && f.output.format!=DXGI_FORMAT_R32G32B32A32_FLOAT)
      return fail(e,"unsupported private output format");
    const WgTexture* tex[]={&f.color,&f.depth,&f.motion};
    for(auto* t:tex) if((uint64_t)f.render_x+f.render_width>t->width || (uint64_t)f.render_y+f.render_height>t->height)
      return fail(e,"active region exceeds an input texture");
    std::vector<ID3D12Resource*> resources;
    auto unique=[&](const WgTexture& t){auto* r=static_cast<ID3D12Resource*>(t.resource);
      for(auto* previous:resources)if(same(previous,r))return false;
      resources.push_back(r);return true;
    };
    for(auto* t:tex) if(!unique(*t)) return fail(e,"aliased input/output resources unsupported");
    if(!unique(f.output)) return fail(e,"aliased input/output resources unsupported");
    if(f.exposure_mode==WG_EXPOSURE_TEXTURE) {
      if(!texture(f.exposure,false,"exposure",e) || f.exposure.format!=DXGI_FORMAT_R32_FLOAT)
        return fail(e,"exposure requires same-device R32_FLOAT texture");
      if(!unique(f.exposure)) return fail(e,"aliased exposure resource unsupported");
    }
    return true;
  }
  void texture_states(ID3D12GraphicsCommandList* list,const WgFrame& f,bool restore) {
    D3D12_RESOURCE_BARRIER barriers[6]={};UINT count=0;
    if(restore) {
      barriers[count].Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barriers[count++].UAV.pResource=static_cast<ID3D12Resource*>(f.output.resource);
    }
    const WgTexture* textures[]={&f.color,&f.depth,&f.motion,&f.output,&f.exposure};
    for(int i=0;i<(f.exposure_mode==WG_EXPOSURE_TEXTURE?5:4);++i) {
      const auto want=i==3?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      const auto before=static_cast<D3D12_RESOURCE_STATES>(textures[i]->state);
      if((before&want)==want)continue;
      auto& b=barriers[count++];b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      b.Transition.pResource=static_cast<ID3D12Resource*>(textures[i]->resource);
      b.Transition.Subresource=wgi::barrier_subresource(b.Transition.pResource);  // mip 0 only of a texture with mips
      b.Transition.StateBefore=restore?want:before;b.Transition.StateAfter=restore?before:want;
    }
    if(count)list->ResourceBarrier(count,barriers);
  }
};
NativeCandidate::NativeCandidate(std::unique_ptr<Impl> p):p_(std::move(p)){}
NativeCandidate::~NativeCandidate(){
  std::string e;
  if(p_ && !p_->sync(&e)) {
    std::fprintf(stderr,"native candidate quarantine: completion unknown: %s\n",e.c_str());
    // Preserve runtime, Converter descriptors and texture references as one unit.
    // Never destroy objects still referenced by a possibly live queue.
    p_.release();
  }
}
std::unique_ptr<NativeCandidate> NativeCandidate::create(ID3D12Device* device,ID3D12CommandQueue* queue,
    const WgRtConfig& cfg,uint32_t flags,std::string* e,bool direct_resources) {
  const uint32_t supported=WG_CREATE_HDR|WG_CREATE_DEPTH_INVERTED|WG_CREATE_AUTO_EXPOSURE|
    WG_CREATE_DEPTH_INFINITE|WG_CREATE_DYNAMIC_RESOLUTION;
  if(!device || !queue || !cfg.build_dir || cfg.backend!=3 || (flags&~supported) ||
     !!cfg.depth_inverted!=!!(flags&WG_CREATE_DEPTH_INVERTED) || !!cfg.hdr!=!!(flags&WG_CREATE_HDR)) {
    fail(e,"private candidate requires backend 3, device, explicit queue, build and supported flags"); return {};
  }
  auto p=std::make_unique<Impl>(); p->flags=flags;p->direct_resources=direct_resources;
  p->rw=cfg.max_rw;p->rh=cfg.max_rh;p->ow=cfg.ow;p->oh=cfg.oh;
  auto dev=wgrt::make_d3d12_device_on(device,queue,cfg.build_dir,e); if(!dev) return {};
  p->device=device;device->AddRef();p->queue=queue;queue->AddRef();
  p->rt=wgrt::runtime_create(cfg,std::move(dev),e,direct_resources,direct_resources); if(!p->rt) return {};
  p->dev=wgrt::runtime_device(p->rt);p->ext=wgrt::d3d12_ext(p->dev);
  auto result=std::unique_ptr<NativeCandidate>(new NativeCandidate(std::move(p)));
  auto& x=*result->p_;
  // Named runtime buffers are a last-frame diagnostic API and cannot be queried
  // before a frame. Explicit private slots avoid a warm-up frame/state mutation.
  if(!x.ext)return {};
  if(!direct_resources && (!x.converter.init(device,e) || !x.allocate(size_t(x.rw)*x.rh*16,&x.color,&x.in.color,e) ||
     !x.allocate(size_t(x.rw)*x.rh*4,&x.depth,&x.in.depth,e) || !x.allocate(size_t(x.rw)*x.rh*8,&x.motion,&x.in.motion,e) ||
     !x.allocate(size_t(x.ow)*x.oh*12,&x.result,&x.out,e) || !x.allocate(4,&x.expo,&x.in.expo,e))) return {};
  if(!x.sync(e))return {};
  return result;
}
bool NativeCandidate::dispatch(const WgFrame& f,ID3D12Fence* ready,uint64_t value,
    ID3D12Fence** completion,uint64_t* done,std::string* e) {
  if(completion)*completion=nullptr;if(done)*done=0;
  if(!completion || !done) return fail(e,"completion output pointers are required");
  auto& x=*p_;
  if(!x.validate(f,ready,value,e) || !x.sync(e)) return false;
  WgRtFrame fr={};fr.rw=(int)f.render_width;fr.rh=(int)f.render_height;
  // Both routes apply the Converter's motion mad before the encoder's existing motion transform.
  fr.mv_scale[0]=fr.mv_scale[1]=1;fr.jitter[0]=f.jitter[0];fr.jitter[1]=f.jitter[1];
  fr.exposure=f.exposure_value;fr.pre_exposure=f.pre_exposure;fr.reset=(int)f.reset;
  wgrt::FrameIo io;io.color=x.color;io.depth=x.depth;io.motion=x.motion;io.out=x.result;
  io.exposure_mode=(int)f.exposure_mode;io.expo_src=x.expo;io.native_direct=x.direct_resources;
  if(x.direct_resources && !wgrt::runtime_direct_preflight(x.rt,fr,&io))return fail(e,wgrt::runtime_error(x.rt));
  struct FailedFrame {
    Impl& impl;bool success=false;
    ~FailedFrame(){if(!success)impl.poisoned=true;}
  } frame_guard{x};
  const WgTexture* ts[]={&f.color,&f.depth,&f.motion,&f.output,&f.exposure};
  for(int i=0;i<(f.exposure_mode==WG_EXPOSURE_TEXTURE?5:4);++i) {
    auto* r=static_cast<ID3D12Resource*>(ts[i]->resource);x.pending.push_back(r);r->AddRef();
  }
  auto stop=[&](){x.poisoned=true;return false;};
  x.inventory={};x.inventory.direct_resources=x.direct_resources;
  x.inventory.runtime_bytes=wgrt::runtime_memory_bytes(x.rt);
  if(x.direct_resources) {
    wgrt::NativeDirectFrame df;
    df.color=static_cast<ID3D12Resource*>(f.color.resource);df.depth=static_cast<ID3D12Resource*>(f.depth.resource);
    df.motion=static_cast<ID3D12Resource*>(f.motion.resource);df.output=static_cast<ID3D12Resource*>(f.output.resource);
    df.exposure=f.exposure_mode==WG_EXPOSURE_TEXTURE?static_cast<ID3D12Resource*>(f.exposure.resource):nullptr;
    df.color_format=(DXGI_FORMAT)f.color.format;df.depth_format=(DXGI_FORMAT)f.depth.format;
    df.motion_format=(DXGI_FORMAT)f.motion.format;df.output_format=(DXGI_FORMAT)f.output.format;
    df.exposure_format=DXGI_FORMAT_R32_FLOAT;
    df.params.rx=f.render_x;df.params.ry=f.render_y;df.params.rw=f.render_width;df.params.rh=f.render_height;
    df.params.msx=f.motion_scale[0];df.params.msy=f.motion_scale[1];
    df.params.mox=f.motion_offset[0];df.params.moy=f.motion_offset[1];
    df.params.color_pi=wgrt::runtime_effective_pre_exposure(x.rt,f.pre_exposure);df.params.exposure_mode=f.exposure_mode;
    // Binding may wait for prior descriptor use: it MUST precede a potentially pending producer Queue::Wait.
    if(!x.ext->bind_direct_frame(df,e))return stop();
    x.direct_bound=true;x.restore_frame=f;
    x.inventory.removed_io_bytes=72ull*x.rw*x.rh+24ull*x.ow*x.oh+4;
  }
  wgrt::trace_launch(x.direct_resources?"native_direct_begin":"native_staged_begin",0,0,"d3d12");
  if(!x.ext->queue_wait(ready,value,e))return stop();
  if(x.direct_resources) {
    if(!x.ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      [&](ID3D12GraphicsCommandList* list,std::string&){x.texture_states(list,f,false);x.needs_restore=true;return true;},e,"direct_input_states"))return stop();
  } else {
    if(!x.ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      [&](ID3D12GraphicsCommandList* list,std::string&){x.converter.record_inputs(list,f,x.flags,x.in);return true;},e,"staged_input_conversion"))return stop();
    ++x.inventory.input_conversion_passes;wgrt::trace_launch("native_staged_input_conversion",0,0,"d3d12");
  }
  if(!wgrt::runtime_frame(x.rt,fr,&io)) {fail(e,wgrt::runtime_error(x.rt));return stop();}
  x.last_rw=fr.rw;x.last_rh=fr.rh;
  if(x.direct_resources) {
    if(!x.ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      [&](ID3D12GraphicsCommandList* list,std::string&){x.texture_states(list,f,true);x.needs_restore=false;return true;},e,"direct_output_states") ||
      !x.ext->end_direct_frame(e))return stop();
    x.direct_bound=false;
  } else {
    if(!x.ext->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      [&](ID3D12GraphicsCommandList* list,std::string&){x.converter.record_output(list,f,x.out);return true;},e,"staged_output_conversion"))return stop();
    ++x.inventory.output_conversion_passes;wgrt::trace_launch("native_staged_output_conversion",0,0,"d3d12");
  }
  if(!x.ext->submit_completion(completion,done,e))return stop();
  wgrt::trace_launch(x.direct_resources?"native_direct_end":"native_staged_end",0,0,"d3d12");
  frame_guard.success=true;
  return true;
}
bool NativeCandidate::sync(std::string* e){return p_->sync(e);}
const char* NativeCandidate::provider()const{
  // the model the runtime runs (runtime_model: "M", or "L" with the runtime option model=L in builds with Model L)
  if(p_->rt&&std::strcmp(wgrt::runtime_model(p_->rt),"L")==0)return p_->direct_resources?
    "Model L native D3D12, supplied device/queue, direct textures":"Model L native D3D12, supplied device/queue, staged reference";
  return p_->direct_resources?
  "Model M native D3D12, supplied device/queue, direct textures":"Model M native D3D12, supplied device/queue, staged reference";}
const NativeFrameInventory& NativeCandidate::last_frame_inventory()const{return p_->inventory;}
bool NativeCandidate::read_buffer(const char* name,std::vector<unsigned char>* bytes,std::string* e) {
  if(!bytes || !p_->sync(e)) return false;
  uintptr_t ptr=0;size_t n=0;
  if(!p_->last_rw)return fail(e,"no private candidate frame has run");
  std::string key=name;
  size_t pixels=size_t(p_->last_rw)*p_->last_rh;
  if(key=="input.color"){ptr=p_->color;n=pixels*16;}
  else if(key=="input.depth"){ptr=p_->depth;n=pixels*4;}
  else if(key=="input.motion"){ptr=p_->motion;n=pixels*8;}
  else if(key=="filter.out"){ptr=p_->result;n=size_t(p_->ow)*p_->oh*12;}
  else if(wg_rt_buffer(p_->rt,name,&ptr,&n))return fail(e,std::string("unknown buffer: ")+name);
  if(!ptr)return fail(e,std::string("buffer unavailable on direct-resource route: ")+name);
  if(std::string(name)=="exposure.pair") n=20; // selected diet path writes five floats
  bytes->resize(n);
  if(n && !p_->dev->download(bytes->data(),ptr,n))return fail(e,p_->dev->err());
  return true;
}
}

namespace {
struct NativeCandidateHandle {
  std::unique_ptr<wgi::NativeCandidate> candidate;
  std::string error;
  bool exception_failed=false;
};
void candidate_error_copy(char* dst,int capacity,const std::string& message) {
  if(!dst || capacity<=0)return;
  size_t n=std::min(message.size(),size_t(capacity-1));
  std::memcpy(dst,message.data(),n);dst[n]=0;
}
}
extern "C" int wg_native_candidate_create(void* device,void* queue,const WgRtConfig* config,
    uint32_t flags,void** candidate,char* error,int capacity) {
  if(candidate)*candidate=nullptr;
  if(!config || !candidate){candidate_error_copy(error,capacity,"configuration and candidate output are required");return 1;}
  try {
    auto h=std::make_unique<NativeCandidateHandle>();
    h->candidate=wgi::NativeCandidate::create(static_cast<ID3D12Device*>(device),static_cast<ID3D12CommandQueue*>(queue),
                                             *config,flags,&h->error);
    if(!h->candidate){candidate_error_copy(error,capacity,h->error);return 1;}
    candidate_error_copy(error,capacity,"");*candidate=h.release();return 0;
  }catch(const std::exception& e){candidate_error_copy(error,capacity,e.what());return 1;}
  catch(...){candidate_error_copy(error,capacity,"unexpected native candidate create exception");return 1;}
}
extern "C" int wg_native_candidate_dispatch(void* candidate,const WgFrame* frame,void* ready,uint64_t value,
    void** completion,uint64_t* completed_value) {
  if(completion)*completion=nullptr;if(completed_value)*completed_value=0;
  auto* h=static_cast<NativeCandidateHandle*>(candidate);if(!h)return 1;
  if(h->exception_failed){h->error="candidate cannot resume after an exception; synchronize and destroy it";return 1;}
  if(!frame || !completion || !completed_value){h->error="frame and completion outputs are required";return 1;}
  try {
    ID3D12Fence* fence=nullptr;
    bool ok=h->candidate->dispatch(*frame,static_cast<ID3D12Fence*>(ready),value,&fence,completed_value,&h->error);
    *completion=fence;if(ok)h->error.clear();return ok?0:1;
  }catch(const std::exception& e){h->exception_failed=true;h->error=e.what();return 1;}
  catch(...){h->exception_failed=true;h->error="unexpected native candidate dispatch exception";return 1;}
}
extern "C" int wg_native_candidate_sync(void* candidate) {
  auto* h=static_cast<NativeCandidateHandle*>(candidate);if(!h)return 1;
  try {bool ok=h->candidate->sync(&h->error);if(ok)h->error.clear();return ok?0:1;}
  catch(const std::exception& e){h->error=e.what();return 1;}
  catch(...){h->error="unexpected native candidate sync exception";return 1;}
}
extern "C" const char* wg_native_candidate_error(void* candidate) {
  auto* h=static_cast<NativeCandidateHandle*>(candidate);return h?h->error.c_str():"null native candidate";
}
extern "C" int wg_native_candidate_destroy(void* candidate) {
  auto* h=static_cast<NativeCandidateHandle*>(candidate);if(!h)return 0;
  int status=wg_native_candidate_sync(h);delete h;return status;
}
