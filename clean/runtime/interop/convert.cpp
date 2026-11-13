// SPDX-License-Identifier: MIT
// D3D12 copy/convert passes (see convert.h). WrongGPU's own code.
#include "convert.h"
#include "descriptor_publish.h"
#include "texture_shape.h"

#include <d3dcommon.h>

#include <cstring>

namespace wgi {

namespace {

constexpr UINT kSetSize = 8;    // t0..t3, u0..u3
constexpr UINT kRingSets = 512; // descriptor sets before reuse (far more than frames in flight x 3)
constexpr uint32_t kMotionOutputRes = 1, kExposureTexture = 2;
constexpr UINT kOptimizationLevel3 = 1u << 15;  // D3DCOMPILE_OPTIMIZATION_LEVEL3 (d3dcompiler.h)

// Written for WrongGPU from the definitions in spec 03 / spec 20 and clean/FORMATS.md.
const char kShader[] = R"HLSL(
cbuffer C : register(b0) {
  uint rx, ry, rw, rh;
  float msx, msy, mox, moy;
  uint mw, mh, flags, ow;
  uint oh, srgb, cw, ch;
};
#if defined(PASS_INPUTS) || defined(PASS_INPUTS16) || defined(PASS_FALLBACK)
Texture2D<float4> colorTex  : register(t0);
#endif
#if defined(PASS_INPUTS) || defined(PASS_INPUTS16)
Texture2D<float>  depthTex  : register(t1);
Texture2D<float4> motionTex : register(t2);
Texture2D<float>  expoTex   : register(t3);
RWBuffer<float4>  colorBuf  : register(u0);
RWBuffer<float>   depthBuf  : register(u1);
RWBuffer<float2>  motionBuf : register(u2);
RWBuffer<float>   expoBuf   : register(u3);
#endif
#if defined(PASS_OUTPUT)
RWBuffer<float>   resultBuf : register(u1);  // read through a UAV: R32_FLOAT typed UAV loads work everywhere
#endif
#if defined(PASS_OUTPUT) || defined(PASS_FALLBACK) || defined(PASS_OUTTEX)
RWTexture2D<float4> outTex  : register(u0);
#endif
#if defined(PASS_OUTTEX)
Texture2D<float4> proxyTex : register(t0);  // executor=dx12: the network's RGBA32F result texture (output pass input)
#endif

#if defined(PASS_FLAG)
RWBuffer<uint> flagBuf : register(u0);
[numthreads(1, 1, 1)]
void flag(uint3 id : SV_DispatchThreadID) { flagBuf[0] = rx; }
#endif

#if defined(PASS_SPIN)
// done_mode=spin: one thread polls the done flag (written by the HIP stream after the network) until it reaches rx,
// for at most ry polls; a timeout is counted in the second word. The poll is an atomic: a plain globallycoherent load
// in the loop is hoisted by the shader compiler.
globallycoherent RWBuffer<uint> spinBuf : register(u0);
[numthreads(1, 1, 1)]
void spin(uint3 id : SV_DispatchThreadID) {
  uint o = 0;
  for (uint i = 0; i < ry; ++i) {
    InterlockedAdd(spinBuf[0], 0, o);
    if (o >= rx) return;
  }
  InterlockedAdd(spinBuf[1], 1, o);
}
#endif

#if defined(PASS_INPUTS)
[numthreads(8, 8, 1)]
void inputs(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rw || id.y >= rh) return;
  int3 p = int3(rx + id.x, ry + id.y, 0);
  uint i = id.y * rw + id.x;
  colorBuf[i] = float4(colorTex.Load(p).rgb, 0.0);
  depthBuf[i] = depthTex.Load(p);
  int3 mp = p;
  if (flags & 1u) {  // motion at output size: the texel under the render pixel's centre
    mp = int3(min(uint((id.x + 0.5) * mw / rw), mw - 1), min(uint((id.y + 0.5) * mh / rh), mh - 1), 0);
  }
  float2 m = motionTex.Load(mp).xy;
  motionBuf[i] = float2(msx * m.x + mox, msy * m.y + moy);
  if ((flags & 2u) && id.x == 0 && id.y == 0) expoBuf[0] = expoTex.Load(int3(0, 0, 0));
}
#endif

#if defined(PASS_INPUTS16)
// Model K's compact input pass (record_inputs16): colour into RGBA16F and the raw motion texel into RG16F elements
// (both UAVs typed FP16; the game's colour and motion are FP16-representable, so the stores are exact), depth and
// exposure as in inputs. The network side applies m = ms raw + mo when it reads the motion.
[numthreads(8, 8, 1)]
void inputs16(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rw || id.y >= rh) return;
  int3 p = int3(rx + id.x, ry + id.y, 0);
  uint i = id.y * rw + id.x;
  colorBuf[i] = float4(colorTex.Load(p).rgb, 0.0);
  depthBuf[i] = depthTex.Load(p);
  int3 mp = p;
  if (flags & 1u) {  // motion at output size: the texel under the render pixel's centre
    mp = int3(min(uint((id.x + 0.5) * mw / rw), mw - 1), min(uint((id.y + 0.5) * mh / rh), mh - 1), 0);
  }
  motionBuf[i] = motionTex.Load(mp).xy;
  if ((flags & 2u) && id.x == 0 && id.y == 0) expoBuf[0] = expoTex.Load(int3(0, 0, 0));
}
#endif

#if defined(PASS_MOTIONPT)
// executor=dx12 with motion_outres=point: the raw motion texel under each render pixel's centre (the texel `inputs`
// reads when flags & 1, by the same expression), stored unchanged (RG32F: exact) at the render pixel's own texture
// coordinate (rx + x, ry + y) of a texture of the executor's. The network's direct encoder then reads that texture as a
// render-size motion texture and applies motion_scale / offset itself, as it does to the game's own render-size motion.
Texture2D<float4> motionTex : register(t2);
RWTexture2D<float2> motionOut : register(u2);
[numthreads(8, 8, 1)]
void motionpt(uint3 id : SV_DispatchThreadID) {
  if (id.x >= rw || id.y >= rh) return;
  int3 mp = int3(min(uint((id.x + 0.5) * mw / rw), mw - 1), min(uint((id.y + 0.5) * mh / rh), mh - 1), 0);
  motionOut[uint2(rx + id.x, ry + id.y)] = motionTex.Load(mp).xy;
}
#endif

#if defined(PASS_MOTIONHI)
// Motion at output resolution: the raw motion texel of every texel of the region mw x mh (texture size cw x ch).
Texture2D<float4> motionTex : register(t2);
RWBuffer<float2>  motionBuf : register(u2);
[numthreads(8, 8, 1)]
void motionhi(uint3 id : SV_DispatchThreadID) {
  if (id.x >= mw || id.y >= mh) return;
  motionBuf[id.y * mw + id.x] = motionTex.Load(int3(min(id.x, cw - 1), min(id.y, ch - 1), 0)).xy;
}
#endif

float3 to_srgb(float3 c) {
  c = saturate(c);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

#if defined(PASS_OUTPUT)
[numthreads(8, 8, 1)]
void output(uint3 id : SV_DispatchThreadID) {
  if (id.x >= ow || id.y >= oh) return;
  uint i = 3 * (id.y * ow + id.x);
  float3 c = float3(resultBuf[i], resultBuf[i + 1], resultBuf[i + 2]);
  if (srgb) c = to_srgb(c);
  outTex[id.xy] = float4(c, 1.0);
}
#endif

#if defined(PASS_OUTTEX)
// The output pass of the network's result held in an RGBA32F texture (executor=dx12, formats the network's own output
// store does not take, e.g. sRGB): the same operations as `output` on the same FP32 values.
[numthreads(8, 8, 1)]
void outtex(uint3 id : SV_DispatchThreadID) {
  if (id.x >= ow || id.y >= oh) return;
  float3 c = proxyTex.Load(int3(id.xy, 0)).rgb;
  if (srgb) c = to_srgb(c);
  outTex[id.xy] = float4(c, 1.0);
}
#endif

#if defined(PASS_FALLBACK)
[numthreads(8, 8, 1)]
void fallback(uint3 id : SV_DispatchThreadID) {
  if (id.x >= ow || id.y >= oh) return;
  float2 s = (float2(id.xy) + 0.5) * float2(rw, rh) / float2(ow, oh) - 0.5;
  int2 a = int2(floor(s));
  float2 f = s - floor(s);
  int2 lo = int2(rx, ry), hi = int2(rx + rw - 1, ry + rh - 1);
  int2 p0 = clamp(int2(rx, ry) + a, lo, hi), p1 = clamp(int2(rx, ry) + a + 1, lo, hi);
  float3 c00 = colorTex.Load(int3(p0.x, p0.y, 0)).rgb, c10 = colorTex.Load(int3(p1.x, p0.y, 0)).rgb;
  float3 c01 = colorTex.Load(int3(p0.x, p1.y, 0)).rgb, c11 = colorTex.Load(int3(p1.x, p1.y, 0)).rgb;
  float3 c = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
  if (srgb) c = to_srgb(c);
  outTex[id.xy] = float4(c, 1.0);
}
#endif
)HLSL";

typedef HRESULT(WINAPI* PfnD3DCompile)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR,
                                       UINT, UINT, ID3DBlob**, ID3DBlob**);
typedef HRESULT(WINAPI* PfnSerialize)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                      ID3DBlob**);

template <class T>
void release(T*& p) {
  if (p) p->Release();
  p = nullptr;
}

bool compile(const char* entry, const char* pass, ID3DBlob** out, std::string* err) {
  HMODULE dll = LoadLibraryW(L"d3dcompiler_47.dll");
  auto fn = dll ? reinterpret_cast<PfnD3DCompile>(reinterpret_cast<void*>(GetProcAddress(dll, "D3DCompile"))) : nullptr;
  if (!fn) return *err = "d3dcompiler_47.dll (part of Windows) could not be loaded", false;
  ID3DBlob* messages = nullptr;
  const D3D_SHADER_MACRO macros[2] = {{pass, "1"}, {nullptr, nullptr}};
  const HRESULT hr = fn(kShader, sizeof(kShader) - 1, "wronggpu_interop.hlsl", macros, nullptr, entry, "cs_5_0",
                        kOptimizationLevel3, 0, out, &messages);
  if (FAILED(hr)) {
    *err = std::string("D3DCompile(") + entry + ") failed";
    if (messages) err->append(": ").append(static_cast<const char*>(messages->GetBufferPointer()),
                                           messages->GetBufferSize());
  }
  release(messages);
  return SUCCEEDED(hr);
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition.pResource = r;
  x.Transition.Subresource = barrier_subresource(r);  // mip 0 only of a game texture with mips (texture_shape.h)
  x.Transition.StateBefore = a;
  x.Transition.StateAfter = b;
  return x;
}

// Transitions of the game's textures into the needed state and back (duplicates and already-fine states skipped).
struct Barriers {
  D3D12_RESOURCE_BARRIER before[8], after[8];
  UINT n = 0;
  void need(ID3D12Resource* r, uint32_t state, D3D12_RESOURCE_STATES want) {
    if (!r || (state & want) == (uint32_t)want) return;
    for (UINT i = 0; i < n; ++i)
      if (before[i].Transition.pResource == r) return;
    before[n] = transition(r, (D3D12_RESOURCE_STATES)state, want);
    after[n] = transition(r, want, (D3D12_RESOURCE_STATES)state);
    ++n;
  }
};

}  // namespace

DXGI_FORMAT srv_format(DXGI_FORMAT f) {
  switch (f) {
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_UNORM;
    default: return f;
  }
}
DXGI_FORMAT uav_format(DXGI_FORMAT f) {
  if (f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) return DXGI_FORMAT_R8G8B8A8_UNORM;
  if (f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) return DXGI_FORMAT_B8G8R8A8_UNORM;
  return f;
}
bool is_srgb(DXGI_FORMAT f) {
  return f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
}

Converter::~Converter() {
  release(heap_);
  release(fallback_);
  release(flag_);
  release(spin_);
  release(output_);
  release(outtex_);
  release(inputs_);
  release(inputs16_);
  release(motionhi_);
  release(motionpt_);
  release(root_);
  release(device_);
}

bool Converter::init(ID3D12Device* device, std::string* err) {
  device_ = device;
  device_->AddRef();
  D3D12_DESCRIPTOR_RANGE ranges[2] = {};
  ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  ranges[0].NumDescriptors = 4;
  ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  ranges[1].NumDescriptors = 4;
  ranges[1].OffsetInDescriptorsFromTableStart = 4;
  D3D12_ROOT_PARAMETER params[2] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[0].Constants.Num32BitValues = sizeof(Constants) / 4;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable.NumDescriptorRanges = 2;
  params[1].DescriptorTable.pDescriptorRanges = ranges;
  D3D12_ROOT_SIGNATURE_DESC rd = {};
  rd.NumParameters = 2;
  rd.pParameters = params;
  HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
  if (!d3d12) d3d12 = LoadLibraryW(L"d3d12.dll");
  auto serialize =
      d3d12 ? reinterpret_cast<PfnSerialize>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeRootSignature")))
            : nullptr;
  if (!serialize) return *err = "D3D12SerializeRootSignature not found", false;
  ID3DBlob *blob = nullptr, *blobErr = nullptr;
  HRESULT hr = serialize(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &blobErr);
  if (SUCCEEDED(hr))
    hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), __uuidof(ID3D12RootSignature),
                                      (void**)&root_);
  release(blob);
  release(blobErr);
  if (FAILED(hr)) return *err = "interop root signature failed", false;

  const char* entries[7] = {"inputs", "output", "fallback", "flag", "spin", "inputs16", "motionhi"};
  const char* passes[7] = {"PASS_INPUTS", "PASS_OUTPUT", "PASS_FALLBACK", "PASS_FLAG", "PASS_SPIN", "PASS_INPUTS16",
                           "PASS_MOTIONHI"};
  ID3D12PipelineState** psos[7] = {&inputs_, &output_, &fallback_, &flag_, &spin_, &inputs16_, &motionhi_};
  for (int i = 0; i < 7; ++i) {
    ID3DBlob* code = nullptr;
    if (!compile(entries[i], passes[i], &code, err)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = root_;
    pd.CS.pShaderBytecode = code->GetBufferPointer();
    pd.CS.BytecodeLength = code->GetBufferSize();
    hr = device_->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)psos[i]);
    release(code);
    if (FAILED(hr)) return *err = std::string("CreateComputePipelineState(") + entries[i] + ") failed", false;
  }
  D3D12_DESCRIPTOR_HEAP_DESC hd = {};
  hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  hd.NumDescriptors = kSetSize * kRingSets;
  hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  if (FAILED(device_->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&heap_)))
    return *err = "CreateDescriptorHeap failed", false;
  increment_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  return true;
}

UINT Converter::next_set() {
  const UINT s = ring_;
  ring_ = (ring_ + 1) % kRingSets;
  return s;
}

void Converter::bind(ID3D12GraphicsCommandList* list, ID3D12PipelineState* pso, const Constants& c, UINT set) {
  D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap_->GetGPUDescriptorHandleForHeapStart();
  gpu.ptr += (UINT64)set * kSetSize * increment_;
  {  // every pass writes its set right before bind: published here (descriptor_publish.h)
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T)set * kSetSize * increment_;
    wg::publish_descriptors(reinterpret_cast<const void*>(cpu.ptr), (size_t)kSetSize * increment_);
  }
  list->SetDescriptorHeaps(1, &heap_);
  list->SetComputeRootSignature(root_);
  list->SetPipelineState(pso);
  list->SetComputeRoot32BitConstants(0, sizeof(Constants) / 4, &c, 0);
  list->SetComputeRootDescriptorTable(1, gpu);
}

bool Converter::inputs16_ok(const WgFrame& f) {
  const DXGI_FORMAT cf = srv_format((DXGI_FORMAT)f.color.format), mf = srv_format((DXGI_FORMAT)f.motion.format);
  return f.color.resource && f.motion.resource &&
         (cf == DXGI_FORMAT_R16G16B16A16_FLOAT || cf == DXGI_FORMAT_R11G11B10_FLOAT) &&
         (mf == DXGI_FORMAT_R16G16_FLOAT || mf == DXGI_FORMAT_R16G16B16A16_FLOAT);
}

bool Converter::motion_fp16(const WgFrame& f) {
  const DXGI_FORMAT mf = srv_format((DXGI_FORMAT)f.motion.format);
  return mf == DXGI_FORMAT_R16G16_FLOAT || mf == DXGI_FORMAT_R16G16B16A16_FLOAT;
}

void Converter::record_motion_hi(ID3D12GraphicsCommandList* list, const WgFrame& f, const BufferRange& dst,
                                 uint32_t mw, uint32_t mh) {
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)set * kSetSize * increment_;
  D3D12_CPU_DESCRIPTOR_HANDLE t2 = cpu, u2 = cpu;
  t2.ptr += (SIZE_T)2 * increment_;
  u2.ptr += (SIZE_T)6 * increment_;
  auto* mt = static_cast<ID3D12Resource*>(f.motion.resource);
  D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
  sd.Format = srv_format((DXGI_FORMAT)f.motion.format);
  sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sd.Texture2D.MipLevels = 1;
  device_->CreateShaderResourceView(mt, &sd, t2);
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = DXGI_FORMAT_R16G16_FLOAT;
  ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  ud.Buffer.FirstElement = dst.offset / 4;
  ud.Buffer.NumElements = mw * mh;
  device_->CreateUnorderedAccessView(dst.resource, nullptr, &ud, u2);

  Constants c = {};
  c.mw = mw;
  c.mh = mh;
  c.cw = f.motion.width ? f.motion.width : mw;
  c.ch = f.motion.height ? f.motion.height : mh;
  Barriers bar;
  bar.need(mt, f.motion.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  if (bar.n) list->ResourceBarrier(bar.n, bar.before);
  bind(list, motionhi_, c, set);
  list->Dispatch((mw + 7) / 8, (mh + 7) / 8, 1);
  if (bar.n) list->ResourceBarrier(bar.n, bar.after);
}

bool Converter::record_motion_point(ID3D12GraphicsCommandList* list, const WgFrame& f, ID3D12Resource* dst,
                                    std::string* err) {
  if (!motionpt_) {  // compiled on first use (executor=dx12 contexts with motion_outres=point only)
    ID3DBlob* code = nullptr;
    if (!compile("motionpt", "PASS_MOTIONPT", &code, err)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = root_;
    pd.CS.pShaderBytecode = code->GetBufferPointer();
    pd.CS.BytecodeLength = code->GetBufferSize();
    const HRESULT hr = device_->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&motionpt_);
    release(code);
    if (FAILED(hr)) return *err = "CreateComputePipelineState(motionpt) failed", false;
  }
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)set * kSetSize * increment_;
  D3D12_CPU_DESCRIPTOR_HANDLE t2 = cpu, u2 = cpu;
  t2.ptr += (SIZE_T)2 * increment_;
  u2.ptr += (SIZE_T)6 * increment_;
  D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};  // the view the input pass reads (srv(2, f.motion, ...))
  sd.Format = srv_format((DXGI_FORMAT)f.motion.format);
  sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sd.Texture2D.MipLevels = 1;
  device_->CreateShaderResourceView(static_cast<ID3D12Resource*>(f.motion.resource), &sd, t2);
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = DXGI_FORMAT_R32G32_FLOAT;
  ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device_->CreateUnorderedAccessView(dst, nullptr, &ud, u2);
  Constants c = {};  // the input pass's values (record_inputs_pass)
  c.rx = f.render_x;
  c.ry = f.render_y;
  c.rw = f.render_width;
  c.rh = f.render_height;
  c.mw = f.motion.width;
  c.mh = f.motion.height;
  bind(list, motionpt_, c, set);
  list->Dispatch((c.rw + 7) / 8, (c.rh + 7) / 8, 1);
  return true;
}

void Converter::record_inputs16(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                                const InputBuffers& b) {
  record_inputs_pass(list, f, create_flags, b, true);
}

void Converter::record_inputs(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                              const InputBuffers& b) {
  record_inputs_pass(list, f, create_flags, b, false);
}

void Converter::record_inputs_pass(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                                   const InputBuffers& b, bool compact) {
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)set * kSetSize * increment_;
  auto at = [&](UINT i) {
    D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
    h.ptr += (SIZE_T)i * increment_;
    return h;
  };
  auto srv = [&](UINT i, const WgTexture& t, DXGI_FORMAT fallbackFormat) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
    d.Format = t.resource ? srv_format((DXGI_FORMAT)t.format) : fallbackFormat;
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(static_cast<ID3D12Resource*>(t.resource), &d, at(i));
  };
  auto uav = [&](UINT i, const BufferRange& r, DXGI_FORMAT fmt, UINT elementBytes, UINT elements) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
    d.Format = fmt;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = r.offset / elementBytes;
    d.Buffer.NumElements = elements;
    device_->CreateUnorderedAccessView(r.resource, nullptr, &d, at(i));
  };
  const bool expoTex = f.exposure_mode == WG_EXPOSURE_TEXTURE && f.exposure.resource;
  const UINT n = f.render_width * f.render_height;
  srv(0, f.color, DXGI_FORMAT_R16G16B16A16_FLOAT);
  srv(1, f.depth, DXGI_FORMAT_R32_FLOAT);
  srv(2, f.motion, DXGI_FORMAT_R16G16_FLOAT);
  WgTexture none = {};
  srv(3, expoTex ? f.exposure : none, DXGI_FORMAT_R32_FLOAT);
  if (compact) {
    uav(4, b.color, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, n);
    uav(6, b.motion, DXGI_FORMAT_R16G16_FLOAT, 4, n);
  } else {
    uav(4, b.color, DXGI_FORMAT_R32G32B32A32_FLOAT, 16, n);
    uav(6, b.motion, DXGI_FORMAT_R32G32_FLOAT, 8, n);
  }
  uav(5, b.depth, DXGI_FORMAT_R32_FLOAT, 4, n);
  uav(7, b.expo, DXGI_FORMAT_R32_FLOAT, 4, 1);

  Constants c = {};
  c.rx = f.render_x;
  c.ry = f.render_y;
  c.rw = f.render_width;
  c.rh = f.render_height;
  c.msx = f.motion_scale[0];
  c.msy = f.motion_scale[1];
  c.mox = f.motion_offset[0];
  c.moy = f.motion_offset[1];
  c.mw = f.motion.width;
  c.mh = f.motion.height;
  c.flags = ((create_flags & WG_CREATE_MOTION_OUTPUT_RES) ? kMotionOutputRes : 0) | (expoTex ? kExposureTexture : 0);

  Barriers bar;
  const auto srvState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  bar.need(static_cast<ID3D12Resource*>(f.color.resource), f.color.state, srvState);
  bar.need(static_cast<ID3D12Resource*>(f.depth.resource), f.depth.state, srvState);
  bar.need(static_cast<ID3D12Resource*>(f.motion.resource), f.motion.state, srvState);
  if (expoTex) bar.need(static_cast<ID3D12Resource*>(f.exposure.resource), f.exposure.state, srvState);
  if (bar.n) list->ResourceBarrier(bar.n, bar.before);
  bind(list, compact ? inputs16_ : inputs_, c, set);
  list->Dispatch((c.rw + 7) / 8, (c.rh + 7) / 8, 1);
  if (bar.n) list->ResourceBarrier(bar.n, bar.after);
}

void Converter::record_output(ID3D12GraphicsCommandList* list, const WgFrame& f, const BufferRange& range) {
  ID3D12Resource* const result = range.resource;
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)set * kSetSize * increment_;
  D3D12_CPU_DESCRIPTOR_HANDLE t0 = cpu, u0 = cpu, u1 = cpu;
  u0.ptr += (SIZE_T)4 * increment_;
  u1.ptr += (SIZE_T)5 * increment_;
  ID3D12Resource* out = static_cast<ID3D12Resource*>(f.output.resource);
  const DXGI_FORMAT ofmt = (DXGI_FORMAT)f.output.format;
  // bug hunt D: never an unordered access view of an output without unordered access (the backend refuses such an output
  // unless the RGBA16F copy route writes it; a fallback frame of that route leaves the output as it is)
  if (!out || !(out->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return;
  if (result) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
    d.Format = DXGI_FORMAT_R32_FLOAT;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = range.offset / 4;
    d.Buffer.NumElements = 3 * f.output_width * f.output_height;
    device_->CreateUnorderedAccessView(result, nullptr, &d, u1);
  } else {
    D3D12_SHADER_RESOURCE_VIEW_DESC d = {};
    d.Format = srv_format((DXGI_FORMAT)f.color.format);
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(static_cast<ID3D12Resource*>(f.color.resource), &d, t0);
  }
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = uav_format(ofmt);
  ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device_->CreateUnorderedAccessView(out, nullptr, &ud, u0);

  Constants c = {};
  c.rx = f.render_x;
  c.ry = f.render_y;
  c.rw = f.render_width;
  c.rh = f.render_height;
  c.ow = f.output_width;
  c.oh = f.output_height;
  c.srgb = is_srgb(ofmt) ? 1 : 0;
  Barriers bar;
  bar.need(out, f.output.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  if (!result)
    bar.need(static_cast<ID3D12Resource*>(f.color.resource), f.color.state,
             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  if (bar.n) list->ResourceBarrier(bar.n, bar.before);
  bind(list, result ? output_ : fallback_, c, set);
  list->Dispatch((c.ow + 7) / 8, (c.oh + 7) / 8, 1);
  if (bar.n) {
    list->ResourceBarrier(bar.n, bar.after);
  } else {
    D3D12_RESOURCE_BARRIER u = {};
    u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    u.UAV.pResource = out;
    list->ResourceBarrier(1, &u);  // the game reads the output after us, possibly as a UAV
  }
}

bool Converter::record_output_tex(ID3D12GraphicsCommandList* list, const WgFrame& f, ID3D12Resource* proxy,
                                  std::string* err) {
  if (!outtex_) {  // compiled on first use (only frames whose output format the network cannot store itself)
    ID3DBlob* code = nullptr;
    if (!compile("outtex", "PASS_OUTTEX", &code, err)) return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = root_;
    pd.CS.pShaderBytecode = code->GetBufferPointer();
    pd.CS.BytecodeLength = code->GetBufferSize();
    const HRESULT hr = device_->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&outtex_);
    release(code);
    if (FAILED(hr)) return *err = "CreateComputePipelineState(outtex) failed", false;
  }
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
  cpu.ptr += (SIZE_T)set * kSetSize * increment_;
  D3D12_CPU_DESCRIPTOR_HANDLE t0 = cpu, u0 = cpu;
  u0.ptr += (SIZE_T)4 * increment_;
  ID3D12Resource* out = static_cast<ID3D12Resource*>(f.output.resource);
  const DXGI_FORMAT ofmt = (DXGI_FORMAT)f.output.format;
  if (!out || !(out->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))  // bug hunt D (see record_output)
    return *err = "frame not taken: output texture has no unordered access", false;
  D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
  sd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
  sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sd.Texture2D.MipLevels = 1;
  device_->CreateShaderResourceView(proxy, &sd, t0);
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = uav_format(ofmt);
  ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  device_->CreateUnorderedAccessView(out, nullptr, &ud, u0);
  Constants c = {};
  c.ow = f.output_width;
  c.oh = f.output_height;
  c.srgb = is_srgb(ofmt) ? 1 : 0;
  Barriers bar;
  bar.need(out, f.output.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  bar.need(proxy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  list->ResourceBarrier(bar.n, bar.before);
  bind(list, outtex_, c, set);
  list->Dispatch((c.ow + 7) / 8, (c.oh + 7) / 8, 1);
  list->ResourceBarrier(bar.n, bar.after);
  D3D12_RESOURCE_BARRIER u = {};
  u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  u.UAV.pResource = out;
  if (out && ((f.output.state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != 0)) list->ResourceBarrier(1, &u);
  return true;
}

void Converter::record_flag(ID3D12GraphicsCommandList* list, const BufferRange& flag, uint32_t value) {
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE u0 = heap_->GetCPUDescriptorHandleForHeapStart();
  u0.ptr += (SIZE_T)set * kSetSize * increment_ + (SIZE_T)4 * increment_;
  D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
  d.Format = DXGI_FORMAT_R32_UINT;
  d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  d.Buffer.FirstElement = flag.offset / 4;
  d.Buffer.NumElements = 1;
  device_->CreateUnorderedAccessView(flag.resource, nullptr, &d, u0);
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;  // every earlier UAV write (the input pass) before the flag
  list->ResourceBarrier(1, &b);
  Constants c = {};
  c.rx = value;
  bind(list, flag_, c, set);
  list->Dispatch(1, 1, 1);
}

void Converter::record_spin(ID3D12GraphicsCommandList* list, const BufferRange& flag, uint32_t value, uint32_t polls) {
  const UINT set = next_set();
  D3D12_CPU_DESCRIPTOR_HANDLE u0 = heap_->GetCPUDescriptorHandleForHeapStart();
  u0.ptr += (SIZE_T)set * kSetSize * increment_ + (SIZE_T)4 * increment_;
  D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
  d.Format = DXGI_FORMAT_R32_UINT;
  d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  d.Buffer.FirstElement = flag.offset / 4;
  d.Buffer.NumElements = 2;  // the flag and the timeout count
  device_->CreateUnorderedAccessView(flag.resource, nullptr, &d, u0);
  Constants c = {};
  c.rx = value;
  c.ry = polls;
  bind(list, spin_, c, set);
  list->Dispatch(1, 1, 1);
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;  // nothing after this starts before the spin is over
  list->ResourceBarrier(1, &b);
}

void Converter::record_flag_immediate(ID3D12GraphicsCommandList* list, const BufferRange& flag, uint32_t value) {
  ID3D12GraphicsCommandList2* l2 = nullptr;
  if (FAILED(list->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), (void**)&l2)) || !l2) return;
  D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p = {flag.resource->GetGPUVirtualAddress() + flag.offset, value};
  D3D12_WRITEBUFFERIMMEDIATE_MODE m = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
  l2->WriteBufferImmediate(1, &p, &m);
  l2->Release();
}

namespace {

// Footprint of `bytes`-per-texel rows of a texture region, at `offset` in a buffer (format from GetCopyableFootprints,
// so that typeless and depth textures get the format D3D12 expects for buffer copies).
bool footprint(ID3D12Device* device, ID3D12Resource* tex, UINT w, UINT h, UINT bpp, uint64_t offset,
               D3D12_PLACED_SUBRESOURCE_FOOTPRINT* fp, UINT* pitch) {
  D3D12_RESOURCE_DESC d = tex->GetDesc();
  if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1) return false;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT l = {};
  UINT rows = 0;
  UINT64 rowBytes = 0, total = 0;
  device->GetCopyableFootprints(&d, 0, 1, 0, &l, &rows, &rowBytes, &total);
  *pitch = (w * bpp + 255) / 256 * 256;
  fp->Offset = offset;
  fp->Footprint.Format = l.Footprint.Format;
  fp->Footprint.Width = w;
  fp->Footprint.Height = h;
  fp->Footprint.Depth = 1;
  fp->Footprint.RowPitch = *pitch;
  return true;
}

DXGI_FORMAT resource_format(void* r) { return static_cast<ID3D12Resource*>(r)->GetDesc().Format; }

}  // namespace

bool Converter::record_input_copies(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                                    const InputBuffers& b, CopyLayout* layout, std::string* why) {
  const DXGI_FORMAT cf = (DXGI_FORMAT)f.color.format, df = (DXGI_FORMAT)f.depth.format,
                    mf = (DXGI_FORMAT)f.motion.format;
  const DXGI_FORMAT crf = resource_format(f.color.resource), drf = resource_format(f.depth.resource),
                    mrf = resource_format(f.motion.resource);
  if (cf != DXGI_FORMAT_R16G16B16A16_FLOAT ||
      (crf != DXGI_FORMAT_R16G16B16A16_FLOAT && crf != DXGI_FORMAT_R16G16B16A16_TYPELESS))
    return *why = "color is not RGBA16F", false;
  // depth: R32F / D32F, or the depth plane of D32F_S8X24 (view R32_FLOAT_X8X24_TYPELESS = 21): subresource 0
  const bool d32 = df == DXGI_FORMAT_R32_FLOAT || df == DXGI_FORMAT_D32_FLOAT || df == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
  const bool d32r = drf == DXGI_FORMAT_R32_FLOAT || drf == DXGI_FORMAT_R32_TYPELESS || drf == DXGI_FORMAT_D32_FLOAT ||
                    drf == DXGI_FORMAT_R32G8X24_TYPELESS || drf == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
  if (!d32 || !d32r)
    return *why = "depth is not R32F / D32F / D32F_S8X24 (view " + std::to_string((int)df) + ", resource " +
                  std::to_string((int)drf) + ")", false;
  int mch = 0;
  if (mf == DXGI_FORMAT_R16G16B16A16_FLOAT &&
      (mrf == DXGI_FORMAT_R16G16B16A16_FLOAT || mrf == DXGI_FORMAT_R16G16B16A16_TYPELESS))
    mch = 4;
  else if (mf == DXGI_FORMAT_R16G16_FLOAT && (mrf == DXGI_FORMAT_R16G16_FLOAT || mrf == DXGI_FORMAT_R16G16_TYPELESS))
    mch = 2;
  if (!mch) return *why = "motion is not RGBA16F / RG16F", false;
  if (create_flags & WG_CREATE_MOTION_OUTPUT_RES) return *why = "motion is at output resolution", false;
  const bool expoTex = f.exposure_mode == WG_EXPOSURE_TEXTURE && f.exposure.resource;
  if (expoTex && ((DXGI_FORMAT)f.exposure.format != DXGI_FORMAT_R32_FLOAT ||
                  (resource_format(f.exposure.resource) != DXGI_FORMAT_R32_FLOAT &&
                   resource_format(f.exposure.resource) != DXGI_FORMAT_R32_TYPELESS)))
    return *why = "exposure texture is not R32F", false;

  auto* ct = static_cast<ID3D12Resource*>(f.color.resource);
  auto* dt = static_cast<ID3D12Resource*>(f.depth.resource);
  auto* mt = static_cast<ID3D12Resource*>(f.motion.resource);
  auto* et = static_cast<ID3D12Resource*>(f.exposure.resource);
  const UINT rw = f.render_width, rh = f.render_height;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT cfp, dfp, mfp, efp;
  UINT cp = 0, dp = 0, mp = 0, ep = 0;
  if (!footprint(device_, ct, rw, rh, 8, b.color.offset, &cfp, &cp) ||
      !footprint(device_, dt, rw, rh, 4, b.depth.offset, &dfp, &dp) ||
      !footprint(device_, mt, rw, rh, 2 * mch, b.motion.offset, &mfp, &mp) ||
      (expoTex && !footprint(device_, et, 1, 1, 4, b.expo.offset, &efp, &ep)))
    return *why = "a texture is not a single-sample 2D texture", false;

  {
    const D3D12_RESOURCE_DESC d0 = dt->GetDesc();
    const bool ds = (d0.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) || d0.Format == DXGI_FORMAT_D32_FLOAT ||
                    d0.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT || d0.Format == DXGI_FORMAT_R32G8X24_TYPELESS;
    if (ds && (f.render_x || f.render_y || d0.Width != rw || d0.Height != rh))
      return *why = "depth is a depth-stencil texture larger than the active region (only whole copies allowed)", false;
  }
  Barriers bar;
  const auto src = D3D12_RESOURCE_STATE_COPY_SOURCE;
  bar.need(ct, f.color.state, src);
  bar.need(dt, f.depth.state, src);
  bar.need(mt, f.motion.state, src);
  if (expoTex) bar.need(et, f.exposure.state, src);
  if (bar.n) list->ResourceBarrier(bar.n, bar.before);
  const D3D12_BOX box = {f.render_x, f.render_y, 0, f.render_x + rw, f.render_y + rh, 1};
  // A depth-stencil resource can only be copied as a whole subresource (no box).
  const D3D12_RESOURCE_DESC dd = dt->GetDesc();
  const bool dsv = (dd.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) || dd.Format == DXGI_FORMAT_D32_FLOAT ||
                   dd.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT || dd.Format == DXGI_FORMAT_R32G8X24_TYPELESS;
  auto copy = [&](ID3D12Resource* tex, const BufferRange& dst, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,
                  const D3D12_BOX* srcBox) {
    D3D12_TEXTURE_COPY_LOCATION d = {}, s = {};
    d.pResource = dst.resource;
    d.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    d.PlacedFootprint = fp;
    s.pResource = tex;
    s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    s.SubresourceIndex = 0;
    list->CopyTextureRegion(&d, 0, 0, 0, &s, srcBox);
  };
  copy(ct, b.color, cfp, &box);
  copy(dt, b.depth, dfp, dsv ? nullptr : &box);
  copy(mt, b.motion, mfp, &box);
  if (expoTex) {
    const D3D12_BOX one = {0, 0, 0, 1, 1, 1};
    copy(et, b.expo, efp, &one);
  }
  if (bar.n) list->ResourceBarrier(bar.n, bar.after);
  layout->color_pitch = (int)cp;
  layout->depth_pitch = (int)dp;
  layout->motion_pitch = (int)mp;
  layout->motion_channels = mch;
  return true;
}

bool Converter::output_copy_ok(const WgFrame& f, std::string* why) {
  auto* out = static_cast<ID3D12Resource*>(f.output.resource);
  if (!out) return *why = "no output texture", false;
  const D3D12_RESOURCE_DESC d = out->GetDesc();
  if ((DXGI_FORMAT)f.output.format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
      (d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT && d.Format != DXGI_FORMAT_R16G16B16A16_TYPELESS))
    return *why = "output is not RGBA16F", false;
  if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1)
    return *why = "output is not a single-sample 2D texture", false;
  return true;
}

bool Converter::record_output_copy(ID3D12GraphicsCommandList* list, const WgFrame& f, const BufferRange& result,
                                   int pitch, std::string* why) {
  if (!output_copy_ok(f, why)) return false;
  auto* out = static_cast<ID3D12Resource*>(f.output.resource);
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
  UINT p = 0;
  if (!footprint(device_, out, f.output_width, f.output_height, 8, result.offset, &fp, &p) || (int)p != pitch)
    return *why = "output is not a single-sample 2D texture", false;
  Barriers bar;
  bar.need(out, f.output.state, D3D12_RESOURCE_STATE_COPY_DEST);
  if (bar.n) list->ResourceBarrier(bar.n, bar.before);
  D3D12_TEXTURE_COPY_LOCATION d = {}, s = {};
  d.pResource = out;
  d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  d.SubresourceIndex = 0;
  s.pResource = result.resource;
  s.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  s.PlacedFootprint = fp;
  list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
  if (bar.n) list->ResourceBarrier(bar.n, bar.after);
  return true;
}

}  // namespace wgi
