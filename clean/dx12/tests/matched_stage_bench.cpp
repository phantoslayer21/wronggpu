// SPDX-License-Identifier: MIT
// Original Model M diagnostic: capture actual synthetic frame launch arguments,
// preserve allocation aliases, and replay one stage on HIP and native D3D12.
// Basis: project's Device/runtime APIs and manifest ABI, 2026-10-15.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "../../runtime/runtime_core.h"
#include "../../converter/src/json.h"
#include "../../converter/src/sha256.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace wgrt;
using Bytes = std::vector<unsigned char>;
using Clock = std::chrono::steady_clock;
namespace {
void need(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
double seconds() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
std::string quote(const std::string& s) {
  std::string r = "\"";
  for (unsigned char c : s) { if (c == '\\' || c == '"') r += '\\';
    if (c == '\n') r += "\\n"; else if (c == '\r') r += "\\r"; else r += c; }
  return r + '"';
}
Bytes read(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate); need(bool(f), "open " + p.string());
  const auto n = f.tellg(); need(n >= 0, "size " + p.string()); Bytes b(static_cast<size_t>(n));
  f.seekg(0); if (!b.empty()) f.read(reinterpret_cast<char*>(b.data()), b.size());
  need(bool(f), "read " + p.string()); return b;
}
void write(const fs::path& p, const Bytes& b) {
  std::ofstream f(p, std::ios::binary); need(bool(f), "create " + p.string());
  f.write(reinterpret_cast<const char*>(b.data()), b.size()); need(bool(f), "write " + p.string());
}
std::string hash(const Bytes& b) { return wgc::to_hex(wgc::Sha256::hash(b.data(), b.size())); }
wgc::Json json(const fs::path& p) {
  auto b = read(p); wgc::Json j; std::string e;
  need(wgc::parse_json(std::string(b.begin(), b.end()), j, e), p.string() + ": " + e); return j;
}
const wgc::Json& member(const wgc::Json& j, const char* key) {
  auto* x = j.find(key); need(x != nullptr, std::string("missing ") + key); return *x;
}
std::string str(const wgc::Json& j, const char* key) { const auto& x = member(j,key); need(x.is_string(),key); return x.text; }
int64_t number(const wgc::Json& j) { int64_t x=0; need(j.as_int64(x),"expected integer"); return x; }
int64_t num(const wgc::Json& j, const char* key) { return number(member(j,key)); }
std::vector<std::string> tokens(const std::string& s) {
  std::vector<std::string> r; std::istringstream f(s); std::string t;
  while(std::getline(f,t,',')) r.push_back(t); return r;
}
struct Abi { std::string name, co, args; unsigned block=0; };
std::map<std::string,Abi> manifest(const fs::path& build) {
  auto j=json(build/"dx12/manifest.json"); std::map<std::string,Abi> r;
  for(const auto& k:member(j,"kernels").items) {
    Abi a{str(k,"name"),str(k,"co"),str(k,"args"),static_cast<unsigned>(num(k,"block"))};
    r.emplace(a.name,a);
  }
  return r;
}
struct Arg { std::string type; Bytes raw; int allocation=-1; size_t offset=0; };
struct Allocation { DevPtr original=0; Bytes pre, post; };
struct Case { Abi abi; unsigned grid=0; std::vector<Arg> args; std::vector<Allocation> allocations; };
size_t first_difference(const Bytes& a,const Bytes& b) {
  if(a.size()!=b.size()) return std::min(a.size(),b.size());
  for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]) return i;
  return SIZE_MAX;
}
void save_case(const fs::path& dir,const Case& c) {
  fs::create_directories(dir); std::ofstream f(dir/"case.json");
  f << "{\n\"format\":1,\"kernel\":" << quote(c.abi.name) << ",\"co\":" << quote(c.abi.co)
    << ",\"abi\":" << quote(c.abi.args) << ",\"grid\":" << c.grid << ",\"block\":" << c.abi.block << ",\n\"args\":[";
  for(size_t i=0;i<c.args.size();++i) {const auto& a=c.args[i]; if(i)f<<',';
    f<<"{\"type\":"<<quote(a.type)<<",\"allocation\":"<<a.allocation<<",\"offset\":"<<a.offset<<",\"raw\":[";
    for(size_t k=0;k<a.raw.size();++k){if(k)f<<',';f<<unsigned(a.raw[k]);} f<<"]}";
  }
  f<<"],\n\"allocations\":[";
  for(size_t i=0;i<c.allocations.size();++i) { const auto& a=c.allocations[i];if(i)f<<',';
    std::string stem="allocation-"+std::to_string(i);write(dir/(stem+"-pre.bin"),a.pre);write(dir/(stem+"-post.bin"),a.post);
    f<<"{\"bytes\":"<<a.pre.size()<<",\"pre_sha256\":"<<quote(hash(a.pre))<<",\"post_sha256\":"<<quote(hash(a.post))
      <<",\"changed\":"<<(a.pre!=a.post?"true":"false")<<'}';
  }
  f<<"],\"domain\":\"original synthetic OC frame 1 after frame 0; complete containing allocations including scratch and padding\"}\n";
  need(bool(f),"write case JSON");
}
Case load_case(const fs::path& dir) {
  auto j=json(dir/"case.json"); need(num(j,"format")==1,"case version"); Case c;
  c.abi={str(j,"kernel"),str(j,"co"),str(j,"abi"),static_cast<unsigned>(num(j,"block"))};c.grid=static_cast<unsigned>(num(j,"grid"));
  for(const auto& x:member(j,"args").items) {Arg a;a.type=str(x,"type");a.allocation=static_cast<int>(num(x,"allocation"));a.offset=num(x,"offset");
    for(const auto& b:member(x,"raw").items){auto v=number(b);need(v>=0&&v<256,"argument byte range");a.raw.push_back(static_cast<unsigned char>(v));}c.args.push_back(std::move(a));}
  for(const auto& x:member(j,"allocations").items) {Allocation a;auto stem="allocation-"+std::to_string(c.allocations.size());
    a.pre=read(dir/(stem+"-pre.bin"));a.post=read(dir/(stem+"-post.bin"));
    need(a.pre.size()==static_cast<size_t>(num(x,"bytes"))&&a.pre.size()==a.post.size(),"allocation lengths");
    need(hash(a.pre)==str(x,"pre_sha256")&&hash(a.post)==str(x,"post_sha256"),"capture hash mismatch");c.allocations.push_back(std::move(a));}
  return c;
}
class Capture : public Device {
 public:
  std::unique_ptr<Device> d; std::map<DevPtr,size_t> live; std::map<std::string,Abi> abis;
  fs::path out; bool enabled=false; size_t count=0; std::vector<std::string> stages;
  Capture(std::unique_ptr<Device> x,std::map<std::string,Abi> a,fs::path p):d(std::move(x)),abis(std::move(a)),out(std::move(p)){}
  bool ok(bool x){if(!x)err_=d->err();return x;}
  std::string name()const override{return d->name();}
  bool alloc(DevPtr* p,size_t n)override {
    if(!ok(d->alloc(p,n)))return false;live[*p]=n;
    // Deterministic poison of otherwise unwritten scratch: applied equally to every later replay.
    return ok(d->memset(*p,0xCD,n));
  }
  void free(DevPtr p)override{d->free(p);live.erase(p);}
  bool upload(DevPtr p,const void* s,size_t n)override{return ok(d->upload(p,s,n));}
  bool download(void* p,DevPtr s,size_t n)override{return ok(d->download(p,s,n));}
  bool copy(DevPtr p,DevPtr s,size_t n)override{return ok(d->copy(p,s,n));}
  bool memset(DevPtr p,int v,size_t n)override{return ok(d->memset(p,v,n));}
  bool load_module(const std::string& p)override{return ok(d->load_module(p));}
  bool load_module_tagged(const std::string& p,const std::string& t)override{return ok(d->load_module_tagged(p,t));}
  bool sync()override{return ok(d->sync());}
  bool record(int i)override{return ok(d->record(i));}
  bool elapsed_ms(int a,int b,float* m)override{return ok(d->elapsed_ms(a,b,m));}
  bool reserve_events(int n)override{return ok(d->reserve_events(n));}
  int event_done(int n)override{return d->event_done(n);}
  bool elapsed_done_ms(int a,int b,float* m)override{return ok(d->elapsed_done_ms(a,b,m));}
  bool launch(const std::string& name,unsigned grid,unsigned block,void** values)override {
    if(!enabled)return ok(d->launch(name,grid,block,values));
    auto found=abis.find(name);need(found!=abis.end(),"no manifest ABI for " +name);Case c;c.abi=found->second;c.grid=grid;
    need(c.abi.block==block,"capture block mismatch");std::map<DevPtr,int> ids;
    auto ts=tokens(c.abi.args);
    for(size_t i=0;i<ts.size();++i){Arg a;a.type=ts[i];size_t n=a.type=="p"?sizeof(DevPtr):a.type=="u"?4:std::stoul(a.type.substr(1));
      a.raw.resize(n);std::memcpy(a.raw.data(),values[i],n);
      if(a.type=="p") {DevPtr p=0;std::memcpy(&p,a.raw.data(),sizeof p);
        if(p){auto it=live.upper_bound(p);need(it!=live.begin(),"untracked pointer "+name);--it;
          need(p-it->first<it->second,"pointer beyond live allocation "+name);a.offset=p-it->first;
          auto id=ids.find(it->first);if(id==ids.end()){int index=static_cast<int>(c.allocations.size());ids[it->first]=index;
            Allocation b;b.original=it->first;b.pre.resize(it->second);b.post.resize(it->second);c.allocations.push_back(std::move(b));a.allocation=index;}else a.allocation=id->second;
        }
      }c.args.push_back(std::move(a));
    }
    for(auto& a:c.allocations)need(ok(d->download(a.pre.data(),a.original,a.pre.size())),err());
    need(ok(d->launch(name,grid,block,values)),err());
    for(auto& a:c.allocations)need(ok(d->download(a.post.data(),a.original,a.post.size())),err());
    char sub[16];std::snprintf(sub,sizeof sub,"stage-%02zu",count++);save_case(out/sub,c);stages.push_back(sub);
    std::cout<<"CAPTURE "<<sub<<' '<<name<<" allocations="<<c.allocations.size()<<std::endl;return true;
  }
};
void capture(const fs::path& build,const fs::path& model,const fs::path& inputs,const fs::path& out) {
  auto j=json(inputs/"inputs.json");std::string e;
  auto d=make_d3d12_device(build.string(),&e);need(d!=nullptr,e);
  auto wrapper=std::make_unique<Capture>(std::move(d),manifest(build),out);auto* cap=wrapper.get();
  std::string bs=build.string(),ms=model.string();WgRtConfig cfg={};cfg.build_dir=bs.c_str();cfg.model_dir=ms.c_str();
  cfg.backend=3;cfg.model_format=1;cfg.max_rw=static_cast<int>(num(j,"rw"));cfg.max_rh=static_cast<int>(num(j,"rh"));
  cfg.ow=static_cast<int>(num(j,"ow"));cfg.oh=static_cast<int>(num(j,"oh"));cfg.depth_inverted=static_cast<int>(num(j,"depth_inverted"));cfg.auto_exposure=1;
  Runtime* raw=runtime_create(cfg,std::move(wrapper),&e);need(raw!=nullptr,e);
  std::unique_ptr<Runtime,decltype(&runtime_destroy)> rt(raw,runtime_destroy);
  for(int i=0;i<2;++i) {
    auto prefix=std::to_string(i);auto color=read(inputs/(prefix+"-color.bin"));auto depth=read(inputs/(prefix+"-depth.bin"));auto motion=read(inputs/(prefix+"-motion.bin"));
    size_t pixels=static_cast<size_t>(cfg.max_rw)*cfg.max_rh;need(color.size()==pixels*12&&depth.size()==pixels*4&&motion.size()==pixels*8,"input lengths");
    need(wg_rt_upload_inputs(raw,cfg.max_rw,cfg.max_rh,reinterpret_cast<float*>(color.data()),reinterpret_cast<float*>(depth.data()),reinterpret_cast<float*>(motion.data()))==0,runtime_error(raw));
    WgRtFrame fr={};auto fb=read(inputs/(prefix+"-frame.bin"));need(fb.size()==sizeof fr,"frame ABI size");std::memcpy(&fr,fb.data(),sizeof fr);
    cap->enabled=i==1;need(runtime_frame(raw,fr,nullptr),runtime_error(raw));need(cap->sync(),cap->err());cap->enabled=false;
  }
  need(cap->count==11,"expected 11 default frame stages, got "+std::to_string(cap->count));
  std::ofstream f(out/"capture.json");f<<"{\"backend\":\"native D3D12\",\"stages\":[";
  for(size_t i=0;i<cap->stages.size();++i){if(i)f<<',';f<<quote(cap->stages[i]);}
  f<<"],\"kernels\":"<<quote(runtime_kernels(raw))<<",\"initial_allocation_poison\":205}\n";need(bool(f),"write capture summary");
}
double thread_seconds() {
  FILETIME created{},exited{},kernel{},user{};
  if(!GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user))return NAN;
  auto ticks=[](const FILETIME& t){return (uint64_t(t.dwHighDateTime)<<32)|t.dwLowDateTime;};
  return double(ticks(kernel)+ticks(user))*1e-7;
}
struct Stamp {double wall,cpu;explicit Stamp(bool sample=true):wall(sample?seconds():0),cpu(sample?thread_seconds():0){}};
struct Diagnostics {
  bool enabled=true;int round=-1;std::vector<std::string> records;
  void add(const std::string& backend,const std::string& phase,const Stamp& a,const Stamp& b,int index=-1,const std::string& extra="") {
    if(!enabled)return;const bool valid=std::isfinite(a.cpu)&&std::isfinite(b.cpu)&&b.cpu>=a.cpu;
    std::ostringstream s;s<<std::setprecision(12)<<"{\"round\":"<<round<<",\"backend\":"<<quote(backend)<<",\"phase\":"<<quote(phase)
      <<",\"index\":"<<index<<",\"start_monotonic_seconds\":"<<a.wall<<",\"wall_ms\":"<<(b.wall-a.wall)*1e3
      <<",\"thread_cpu_valid\":"<<(valid?"true":"false")<<",\"thread_cpu_ms\":";
    if(valid)s<<(b.cpu-a.cpu)*1e3;else s<<"null";s<<extra<<'}';records.push_back(s.str());
  }
};
struct DiagnosticScope {
  Diagnostics* d;std::string backend,phase;int index;Stamp start;
  DiagnosticScope(Diagnostics* x,const std::string& b,const std::string& p,int i=-1):d(x&&x->enabled?x:nullptr),backend(b),phase(p),index(i),start(d!=nullptr){}
  ~DiagnosticScope(){if(d)d->add(backend,phase,start,Stamp{},index);}
};
struct Replay {
  std::unique_ptr<Device> d;const Case& c;std::vector<DevPtr> pointers;std::vector<Bytes> values;std::vector<void*> args;
  std::string kind;bool idempotent=false;size_t compared=0;std::vector<size_t> changed;Diagnostics& diagnostic;
  Replay(const Case& cc,const fs::path& build,const std::string& rocm,const std::string& k,Diagnostics& dg):c(cc),kind(k),diagnostic(dg) {
    DiagnosticScope scope(&diagnostic,kind,"device_and_replay_setup");
    std::string e;d=k=="hip"?make_hip_device(rocm,&e):make_d3d12_device(build.string(),&e);need(d!=nullptr,e);
    auto abis=manifest(build);auto it=abis.find(c.abi.name);need(it!=abis.end()&&it->second.args==c.abi.args&&it->second.co==c.abi.co&&it->second.block==c.abi.block,"replay manifest binding mismatch");
    auto slash=c.abi.name.find('/');auto co=(build/c.abi.co).string();
    need(slash==std::string::npos?d->load_module(co):d->load_module_tagged(co,c.abi.name.substr(0,slash)),d->err());
    pointers.resize(c.allocations.size());
    for(size_t i=0;i<pointers.size();++i){need(d->alloc(&pointers[i],c.allocations[i].pre.size()),d->err());if(c.allocations[i].pre!=c.allocations[i].post)changed.push_back(i);}
    values.reserve(c.args.size());
    for(const auto& a:c.args){values.push_back(a.raw);if(a.type=="p"){DevPtr p=0;if(a.allocation>=0){need(static_cast<size_t>(a.allocation)<pointers.size()&&a.offset<c.allocations[a.allocation].pre.size(),"bad relocation");p=pointers[a.allocation]+a.offset;}std::memcpy(values.back().data(),&p,sizeof p);}}
    for(auto& a:values)args.push_back(a.data());need(d->reserve_events(2),d->err());reset(true);
  }
  ~Replay(){if(d){d->sync();for(auto p:pointers)d->free(p);}}
  void launch(){need(d->launch(c.abi.name,c.grid,c.abi.block,args.data()),d->err());}
  void reset(bool all=false){DiagnosticScope scope(&diagnostic,kind,"reset_uploads_and_sync");for(size_t i=0;i<pointers.size();++i)if(all||c.allocations[i].pre!=c.allocations[i].post)need(d->upload(pointers[i],c.allocations[i].pre.data(),c.allocations[i].pre.size()),d->err());need(d->sync(),d->err());}
  bool compare(bool fail=true){DiagnosticScope scope(&diagnostic,kind,"compare_all_allocations");for(size_t i=0;i<pointers.size();++i){Bytes b;
      {DiagnosticScope transfer(&diagnostic,kind,"allocation_download_with_host_allocation",static_cast<int>(i));b.resize(c.allocations[i].post.size());need(d->download(b.data(),pointers[i],b.size()),d->err());}
      size_t diff=SIZE_MAX;{DiagnosticScope comparison(&diagnostic,kind,"allocation_byte_compare",static_cast<int>(i));diff=first_difference(b,c.allocations[i].post);}
      if(diff!=SIZE_MAX){if(fail)throw std::runtime_error(kind+" "+c.abi.name+" allocation "+std::to_string(i)+" first mismatch byte "+std::to_string(diff)+" actual="+std::to_string(b[diff])+" expected="+std::to_string(c.allocations[i].post[diff]));return false;}compared+=b.size();}return true;}
  void sync_words(){int arg=c.abi.name=="bneck/swin_bneck6_nc"?3:c.abi.name=="exposure_auto_diet"?2:-1;if(arg<0)return;
    const auto& a=c.args[arg];need(a.allocation>=0,"sync pointer null");size_t n=c.abi.name=="exposure_auto_diet"?4:c.allocations[a.allocation].post.size()-a.offset;
    DiagnosticScope scope(&diagnostic,kind,"sync_words_download",a.allocation);Bytes b(n);need(d->download(b.data(),pointers[a.allocation]+a.offset,n),d->err());
    need(std::all_of(b.begin(),b.end(),[](unsigned char v){return v==0;}),kind+" sync words nonzero");}
  void qualify(){DiagnosticScope scope(&diagnostic,kind,"qualify_two_launches");launch();compare();sync_words();launch();idempotent=compare(false);sync_words();reset(true);}
  void warm(double duration){Stamp begin;int batches=0,slow=0;uint64_t launches=0;double submit_ms=0,sync_ms=0,reset_ms=0,max_batch_ms=0;
    while(seconds()-begin.wall<duration){Stamp a;
      // Suppress per-reset records in warm-up: retain aggregate components and only slow batch detail.
      const bool previous=diagnostic.enabled;diagnostic.enabled=false;if(!idempotent)reset();diagnostic.enabled=previous;Stamp ready;
      int n=idempotent?64:1;for(int k=0;k<n;++k)launch();Stamp submitted;need(d->sync(),d->err());Stamp done;
      launches+=n;reset_ms+=(ready.wall-a.wall)*1e3;submit_ms+=(submitted.wall-ready.wall)*1e3;sync_ms+=(done.wall-submitted.wall)*1e3;
      max_batch_ms=std::max(max_batch_ms,(done.wall-a.wall)*1e3);
      if(done.wall-a.wall>.1){++slow;diagnostic.add(kind,"slow_warm_batch",a,done,batches,",\"launches\":"+std::to_string(n));
        diagnostic.add(kind,"slow_warm_reset",a,ready,batches);diagnostic.add(kind,"slow_warm_submit",ready,submitted,batches);diagnostic.add(kind,"slow_warm_sync",submitted,done,batches);}
      ++batches;
    }
    Stamp end;std::ostringstream extra;extra<<",\"batches\":"<<batches<<",\"launches\":"<<launches<<",\"slow_batches_over_100ms\":"<<slow
      <<",\"reset_total_ms\":"<<reset_ms<<",\"submit_total_ms\":"<<submit_ms<<",\"sync_total_ms\":"<<sync_ms<<",\"max_batch_ms\":"<<max_batch_ms;
    diagnostic.add(kind,"warm_aggregate",begin,end,-1,extra.str());sync_words();}
};
std::string marker_record(const char* kind,const std::string& phase,const std::string& backend,double warm,int round) {
  LARGE_INTEGER qpc{},freq{};QueryPerformanceCounter(&qpc);QueryPerformanceFrequency(&freq);
  auto ns=static_cast<uint64_t>(static_cast<long double>(qpc.QuadPart)*1000000000.0L/freq.QuadPart);
  SYSTEMTIME utc{};GetSystemTime(&utc);char iso[40];std::snprintf(iso,sizeof iso,"%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",utc.wYear,utc.wMonth,utc.wDay,utc.wHour,utc.wMinute,utc.wSecond,utc.wMilliseconds);
  std::ostringstream s;s<<std::setprecision(12)<<"{\"kind\":"<<quote(kind)<<",\"phase\":"<<quote(phase)<<",\"config\":"<<quote(backend)
    <<",\"pid\":"<<GetCurrentProcessId()<<",\"monotonic_ns\":"<<ns<<",\"warm_seconds\":"<<warm<<",\"round\":"<<round<<",\"utc\":"<<quote(iso)<<'}';return s.str();
}
void marker(const char* kind,const std::string& phase,const std::string& backend,double warm,int round) {
  const char* root=std::getenv("WG_GPU_EVIDENCE_MARKER_DIR");if(!root||!*root)return;fs::create_directories(root);
  std::ofstream f(fs::path(root)/(std::to_string(GetCurrentProcessId())+".jsonl"),std::ios::app);
  f<<marker_record(kind,phase,backend,warm,round)<<'\n';
}
struct ClockProbe {
  Device& d;DevPtr p=0;bool enabled=false;std::vector<std::string> snapshots;Diagnostics& diagnostic;
  explicit ClockProbe(Device& dev,Diagnostics& dg):d(dev),diagnostic(dg){const char* co=std::getenv("WG_STAGE_CLOCK_CO");if(!co||!*co)return;
    need(d.load_module_tagged(co,"stage_clock"),d.err());need(d.alloc(&p,16),d.err());enabled=true;}
  ~ClockProbe(){if(p)d.free(p);}
  void snapshot(int round,const char* where){if(!enabled)return;int iterations=20000;void* args[]={&p,&iterations};
    for(int i=0;i<3;++i){DiagnosticScope scope(&diagnostic,"hip",std::string("clock_probe_")+where,i);need(d.launch("stage_clock/clk_probe",1,32,args),d.err());uint32_t v[4]={};need(d.download(v,p,sizeof v),d.err());
      uint32_t cycles=v[0]&0xfffffu;double ghz=v[1]?cycles/(10.0*v[1]):0;bool valid=cycles>0&&v[1]>0&&std::isfinite(ghz)&&ghz>0&&ghz<10;
      std::ostringstream s;s<<"{\"round\":"<<round<<",\"where\":"<<quote(where)<<",\"sample\":"<<i<<",\"cycles\":"<<cycles
        <<",\"realtime_ticks_100mhz\":"<<v[1]<<",\"ghz\":"<<ghz<<",\"valid\":"<<(valid?"true":"false")<<'}';snapshots.push_back(s.str());}
  }
};
struct Sample {int round=0,pair=0,batch=0;std::string backend,mode;double gpu=0,wall=0,submit=0,wait=0;};
bool continue_pairs(int count,int minimum,double elapsed,double duration) {
  return count<minimum||elapsed<duration;
}
Sample sample(Replay& r,int round,int pair,int batch,bool events) {
  if(!r.idempotent){r.reset();batch=1;}Sample s;s.round=round;s.pair=pair;s.batch=batch;s.backend=r.kind;s.mode=events?"event":"stream";
  const double a=seconds();if(events)need(r.d->record(0),r.d->err());for(int i=0;i<batch;++i)r.launch();
  if(events)need(r.d->record(1),r.d->err());double b=seconds();need(r.d->sync(),r.d->err());double e=seconds();
  if(events){float v=0;need(r.d->elapsed_ms(0,1,&v),r.d->err());s.gpu=v/batch;}
  s.wall=(e-a)*1e3/batch;s.submit=(b-a)*1e3/batch;s.wait=(e-b)*1e3/batch;return s;
}
void replay(const fs::path& build,const fs::path& captured,const fs::path& output,const std::string& rocm,double warm,int rounds,int repeats,int batch,double minimum_window) {
  need(std::isfinite(warm)&&warm>=3&&rounds>0&&repeats>0&&(batch==4||batch==16)&&std::isfinite(minimum_window)&&minimum_window>=1,
       "warm >=3, positive rounds/repeats, batch 4 or 16, minimum window >=1 second required");
  Case c=load_case(captured);Diagnostics diagnostic;Replay hip(c,build,rocm,"hip",diagnostic),native(c,build,rocm,"d3d12",diagnostic);
  hip.qualify();native.qualify();
  // Use the exact comparison function on a copied output, then deliberately flip one bit.
  need(!c.allocations.empty()&&!c.allocations[0].post.empty(),"mutation source missing");
  Bytes mutation=c.allocations[0].post;mutation[mutation.size()/2]^=1;
  need(first_difference(mutation,c.allocations[0].post)==mutation.size()/2,"one-bit comparator sensitivity failed");
  std::vector<Sample> samples;ClockProbe clock(*hip.d,diagnostic);std::vector<std::string> windows;
  for(int round=0;round<rounds;++round){
    diagnostic.round=round;
    // Warm each backend independently; rotate warm order with paired measurement order.
    if(round%2){native.warm(warm);hip.warm(warm);}else{hip.warm(warm);native.warm(warm);}
    clock.snapshot(round,"after_warm_before_pairs");diagnostic.enabled=false;marker("steady_begin",c.abi.name,"paired-stage",warm,round);
    const double window_start=seconds();int pair=0;
    for(;continue_pairs(pair,repeats,seconds()-window_start,minimum_window);++pair){
      for(bool events:{false,true}){
        Replay* first=(round+pair)%2?&native:&hip;Replay* second=first==&hip?&native:&hip;
        samples.push_back(sample(*first,round,pair,batch,events));samples.push_back(sample(*second,round,pair,batch,events));
      }
    }
    const double duration=seconds()-window_start;std::ostringstream window;
    window<<"{\"round\":"<<round<<",\"actual_pairs\":"<<pair<<",\"duration_seconds\":"<<std::setprecision(12)<<duration<<'}';windows.push_back(window.str());
    marker("steady_end",c.abi.name,"paired-stage",warm,round);diagnostic.enabled=true;clock.snapshot(round,"after_pairs_before_readback");
    // The comparisons deliberately occur outside timing windows. A batch is allowed only after a two-launch
    // idempotence gate, and batch-end sync/scratch/output equality is checked again each round.
    hip.compare();native.compare();hip.sync_words();native.sync_words();
  }
  std::ofstream f(output);f<<std::setprecision(10)<<"{\"kernel\":"<<quote(c.abi.name)<<",\"co\":"<<quote(c.abi.co)<<",\"grid\":"<<c.grid<<",\"block\":"<<c.abi.block
    <<",\"warm_seconds_per_backend_per_round\":"<<warm<<",\"minimum_window_seconds\":"<<minimum_window<<",\"minimum_pairs_per_round\":"<<repeats
    <<",\"hip_idempotent\":"<<(hip.idempotent?"true":"false")<<",\"native_idempotent\":"<<(native.idempotent?"true":"false")
    <<",\"one_bit_mutation_detected\":true,\"all_allocation_exact_gate\":true,\"compared_bytes\":"<<hip.compared+native.compared
    <<",\"hip_adapter\":"<<quote(hip.d->name())<<",\"native_adapter\":"<<quote(native.d->name())<<",\"samples\":[";
  for(size_t i=0;i<samples.size();++i){if(i)f<<',';const auto& s=samples[i];
    f<<"{\"round\":"<<s.round<<",\"pair\":"<<s.pair<<",\"batch\":"<<s.batch<<",\"backend\":"<<quote(s.backend)<<",\"mode\":"<<quote(s.mode)
      <<",\"gpu_ms_per_launch\":"<<(s.mode=="event"&&std::isfinite(s.gpu)?std::to_string(s.gpu):"null")
      <<",\"gpu_raw\":"<<quote(std::to_string(s.gpu))<<",\"gpu_valid\":"<<((s.mode!="event"||(std::isfinite(s.gpu)&&s.gpu>0))?"true":"false")
      <<",\"wall_ms_per_launch\":"<<s.wall<<",\"wall_valid\":"<<((std::isfinite(s.wall)&&s.wall>0)?"true":"false")
      <<",\"submit_ms_per_launch\":"<<s.submit<<",\"wait_ms_per_launch\":"<<s.wait<<'}';}
  f<<"],\"clock_snapshots\":[";for(size_t i=0;i<clock.snapshots.size();++i){if(i)f<<',';f<<clock.snapshots[i];}
  f<<"],\"windows\":[";for(size_t i=0;i<windows.size();++i){if(i)f<<',';f<<windows[i];}
  f<<"],\"outside_window_diagnostics\":[";for(size_t i=0;i<diagnostic.records.size();++i){if(i)f<<',';f<<diagnostic.records[i];}
  f<<"],\"definitions\":{\"gpu\":\"Diagnostic Device record bracket per batch: HIP stream events / native D3D12 timestamps including UAV ordering and query resolve. Not shader-body time.\",\"stream\":\"Uninstrumented batch host launch calls plus final device sync; no uploads/readbacks.\",\"stateful\":\"Non-idempotent cases restore every changed allocation and synchronize before timing; batch forced to 1.\",\"clock\":\"Original shader-cycle/realtime probe on HIP device immediately outside paired window. Snapshots are not simultaneous clock evidence.\",\"window\":\"Paired sampling continues until both minimum pair count and minimum elapsed duration are met; every sample retained.\",\"outside_window_diagnostics\":\"Wall and calling-thread user+kernel CPU time. Nested scopes overlap and must not be summed. Warm batches retain aggregate counts/times; per-batch components retained only above100ms. Other driver threads are not included. Disabled inside paired windows.\",\"scope\":\"Standalone stages from identical captured frame inputs. Does not replace complete-frame promotion measurement.\"}}\n";
  need(bool(f),"write replay JSON");std::cout<<"PASS "<<c.abi.name<<" idempotent="<<hip.idempotent<<'/'<<native.idempotent<<" samples="<<samples.size()<<std::endl;
  if(std::getenv("WG_GPU_EVIDENCE_MARKER_DIR"))Sleep(1000);
}
}
int main(int argc,char** argv){try{
  need(argc>=2,"usage: capture BUILD MODELS INPUTS OUT | replay BUILD CASE OUT ROCM WARM ROUNDS REPEATS BATCH MIN_WINDOW_SECONDS");
  if(std::string(argv[1])=="self-test") {
    need(argc==2,"self-test arguments");
    need(continue_pairs(0,24,0,1)&&continue_pairs(24,24,.999,1)&&continue_pairs(23,24,1.001,1)&&
         !continue_pairs(24,24,1.001,1),"minimum count and duration conjunction");
    Bytes a{1,2,3},b=a;need(first_difference(a,b)==SIZE_MAX,"identity comparator");b[1]^=1;
    need(first_difference(a,b)==1,"one-bit comparator");
    wgc::Json j;std::string error;const auto record=marker_record("steady_begin","cpu-contract-test","paired-stage",3.25,2);
    need(wgc::parse_json(record,j,error),error);need(member(j,"warm_seconds").text=="3.25"&&num(j,"round")==2&&str(j,"utc").back()=='Z',"marker warm/round/UTC contract");
    Diagnostics dg;{DiagnosticScope scope(&dg,"cpu","self_test");}need(dg.records.size()==1,"diagnostic scope record");
    need(wgc::parse_json(dg.records[0],j,error),error);need(member(j,"thread_cpu_valid").boolean,"thread CPU clock");
    dg.enabled=false;{DiagnosticScope scope(&dg,"cpu","disabled");}need(dg.records.size()==1,"paired window diagnostic suppression");
    std::cout<<"PASS CPU-only: minimum count AND duration, comparator mutation, marker warm/round/UTC, thread CPU diagnostic and paired-window suppression; no Device created"<<std::endl;
    std::cout<<"MARKER_CONTRACT "<<record<<std::endl;
  }
  else if(std::string(argv[1])=="capture"){need(argc==6,"capture arguments");capture(argv[2],argv[3],argv[4],argv[5]);}
  else if(std::string(argv[1])=="replay"){need(argc==11,"replay arguments");replay(argv[2],argv[3],argv[4],argv[5],std::stod(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]),std::stod(argv[10]));}
  else throw std::runtime_error("unknown mode");return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
