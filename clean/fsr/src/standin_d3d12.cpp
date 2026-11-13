// SPDX-License-Identifier: MIT
// Stand-in upscaler: one compute dispatch per frame recorded into the game's command list (spec 20).
#include "standin_d3d12.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3dcommon.h>

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "standin_shader.h"

namespace wgfsr {

namespace {

template <class T>
T ProcAddress(HMODULE module, const char* name) {
    return reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
}

// Root constants, register for register the cbuffer of standin_shader.h.
struct StandInConstants {
    uint32_t outW, outH, renderX, renderY;
    uint32_t renderW, renderH, view, filterMode;
    float jitterX, jitterY, mvScaleX, mvScaleY;
    float mvOffX, mvOffY, warpWeight, gain;
    uint32_t flags, motionW, motionH, pad0;
};
static_assert(sizeof(StandInConstants) == 80, "root constants must match the shader's cbuffer");

enum : uint32_t {
    F_JITTER_COMP = 1,
    F_MOTION_OUTRES = 2,
    F_HISTORY_VALID = 4,
    F_OUTPUT_SRGB = 8,
    F_DEPTH_INVERTED = 16,
    F_RESET = 32,
};
enum View : uint32_t { VIEW_UPSCALE = 0, VIEW_WARP = 1, VIEW_WARPDIFF = 2, VIEW_MOTION = 3, VIEW_DEPTH = 4, VIEW_JITTER = 5 };

constexpr UINT kDescriptorsPerFrame = 6;  // t0..t3, u0..u1
constexpr UINT kRingFrames = 256;         // descriptor sets in flight before reuse (far more than frames in flight)
constexpr DXGI_FORMAT kHistoryFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

struct Options {
    uint32_t view = VIEW_UPSCALE;
    uint32_t filter = 0;  // 0 bilinear, 1 lanczos
    bool jitterCompensation = true;
    float warpWeight = 0.9f;
    float gain = 4.0f;
    bool splitTest = false;  // test 20-T5: upscale through the front end's split (interface version 2)
};

Options ParseOptions(const char* text) {
    Options o;
    std::string s = text ? text : "";
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(';', start);
        if (end == std::string::npos) end = s.size();
        const std::string item = s.substr(start, end - start);
        start = end + 1;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = item.substr(0, eq), v = item.substr(eq + 1);
        if (k == "view") {
            o.view = v == "warp" ? VIEW_WARP : v == "warpdiff" ? VIEW_WARPDIFF : v == "motion" ? VIEW_MOTION
                   : v == "depth" ? VIEW_DEPTH : v == "jitter" ? VIEW_JITTER : VIEW_UPSCALE;
        } else if (k == "filter") {
            o.filter = v == "lanczos" ? 1u : 0u;
        } else if (k == "jitter_compensation") {
            o.jitterCompensation = v != "0";
        } else if (k == "warp_weight") {
            o.warpWeight = float(std::atof(v.c_str()));
        } else if (k == "debug_gain") {
            o.gain = float(std::atof(v.c_str()));
        } else if (k == "split_test") {
            o.splitTest = v == "1";
        }
    }
    return o;
}

// D3D12 objects of destroyed upscalers stay alive a few seconds: command lists the game has not executed yet may
// still reference them, and the FidelityFX API gives no fence to wait on.
struct Graveyard {
    std::mutex mutex;
    std::vector<std::pair<IUnknown*, ULONGLONG>> objects;
    void Add(IUnknown* object) {
        if (!object) return;
        std::lock_guard<std::mutex> lock(mutex);
        objects.emplace_back(object, GetTickCount64());
    }
    void ReleaseOld() {
        std::vector<IUnknown*> old;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const ULONGLONG now = GetTickCount64();
            for (size_t i = 0; i < objects.size();) {
                if (now - objects[i].second > 5000) {
                    old.push_back(objects[i].first);
                    objects[i] = objects.back();
                    objects.pop_back();
                } else {
                    ++i;
                }
            }
        }
        for (IUnknown* o : old) o->Release();
    }
};
Graveyard g_graveyard;

// Shader bytecode, compiled once per process.
struct ShaderCache {
    std::mutex mutex;
    bool tried = false;
    std::vector<unsigned char> bytecode;
    std::string error;
};
ShaderCache g_shader;

typedef HRESULT(WINAPI* PfnSerializeRootSignature)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                                   ID3DBlob**);
typedef HRESULT(WINAPI* PfnD3DCompile)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR, UINT,
                                       UINT, ID3DBlob**, ID3DBlob**);

bool CompileShader(std::string& error) {
    std::lock_guard<std::mutex> lock(g_shader.mutex);
    if (!g_shader.tried) {
        g_shader.tried = true;
        HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
        auto compile = compiler ? ProcAddress<PfnD3DCompile>(compiler, "D3DCompile") : nullptr;
        if (!compile) {
            g_shader.error = "d3dcompiler_47.dll (part of Windows) could not be loaded";
        } else {
            ID3DBlob* code = nullptr;
            ID3DBlob* messages = nullptr;
            const UINT optimization = 1u << 15;  // D3DCOMPILE_OPTIMIZATION_LEVEL3
            const HRESULT hr = compile(kStandInShader, sizeof kStandInShader - 1, "wronggpu_standin.hlsl", nullptr, nullptr,
                                       "main", "cs_5_0", optimization, 0, &code, &messages);
            if (SUCCEEDED(hr) && code) {
                const auto* p = static_cast<const unsigned char*>(code->GetBufferPointer());
                g_shader.bytecode.assign(p, p + code->GetBufferSize());
            } else {
                char text[64];
                std::snprintf(text, sizeof text, "D3DCompile failed (0x%08lx): ", static_cast<unsigned long>(hr));
                g_shader.error = text;
                if (messages) g_shader.error.append(static_cast<const char*>(messages->GetBufferPointer()), messages->GetBufferSize());
            }
            if (code) code->Release();
            if (messages) messages->Release();
        }
    }
    error = g_shader.error;
    return !g_shader.bytecode.empty();
}

bool IsReadOnlyState(D3D12_RESOURCE_STATES s) {
    const UINT writes = D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_UNORDERED_ACCESS |
                        D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_STREAM_OUT | D3D12_RESOURCE_STATE_COPY_DEST |
                        D3D12_RESOURCE_STATE_RESOLVE_DEST;
    return (UINT(s) & writes) == 0;
}

DXGI_FORMAT UavFormat(DXGI_FORMAT f) {
    if (f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) return DXGI_FORMAT_R8G8B8A8_UNORM;
    if (f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) return DXGI_FORMAT_B8G8R8A8_UNORM;
    return f;
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

// The game's textures (bug hunt F; the rule of clean/runtime/interop/texture_shape.h): all subresources of a single-mip
// texture, as before; subresource 0 alone of one with mips (the views read and write mip 0), so no other mip leaves the
// state the game keeps it in.
D3D12_RESOURCE_BARRIER GameTransition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b = Transition(r, before, after);
    if (r && r->GetDesc().MipLevels > 1) b.Transition.Subresource = 0;
    return b;
}

D3D12_RESOURCE_BARRIER UavBarrier(ID3D12Resource* r) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = r;
    return b;
}

class StandIn {
public:
    WgLogFn log = nullptr;
    void* logUser = nullptr;
    ID3D12Device* device = nullptr;
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pipeline = nullptr;
    ID3D12DescriptorHeap* heap = nullptr;
    UINT increment = 0;
    UINT ring = 0;
    ID3D12Resource* history[2] = {};
    D3D12_RESOURCE_STATES historyState[2] = {};
    uint32_t parity = 0;
    bool historyValid = false;
    uint32_t outW = 0, outH = 0, createFlags = 0;
    Options options;
    std::map<DXGI_FORMAT, bool> uavSupport;
    uint64_t frames = 0;
    // split_test (20-T5): the front end's split service, the two fences a network backend would share with HIP, and
    // a queue that signals resume as soon as done is reached (the "network" that does nothing).
    const WgFrontEndServices* services = nullptr;
    ID3D12CommandQueue* splitQueue = nullptr;
    ID3D12Fence* doneFence = nullptr;
    ID3D12Fence* resumeFence = nullptr;
    uint64_t splitValue = 0;
    uint64_t splits = 0, splitRefusals = 0;

    void Log(int level, const char* fmt, ...) {
        char text[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(text, sizeof text, fmt, args);
        va_end(args);
        log(logUser, level, text);
    }

    WgResult Create(const WgCreateInfo& info) {
        log = info.log;
        logUser = info.log_user;
        options = ParseOptions(info.options);
        outW = info.output_width;
        outH = info.output_height;
        createFlags = info.flags;
        device = static_cast<ID3D12Device*>(info.d3d12_device);
        if (!device) return WG_ERROR_PARAMETER;
        device->AddRef();

        std::string error;
        if (!CompileShader(error)) {
            Log(WG_LOG_ERROR, "shader: %s", error.c_str());
            return WG_ERROR_DEVICE;
        }

        // Root signature: 20 root constants (b0) and one table of 4 SRVs (t0-t3) followed by 2 UAVs (u0-u1).
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 4;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 2;
        ranges[1].BaseShaderRegister = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = 4;
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.ShaderRegister = 0;
        params[0].Constants.Num32BitValues = sizeof(StandInConstants) / 4;
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1].DescriptorTable.NumDescriptorRanges = 2;
        params[1].DescriptorTable.pDescriptorRanges = ranges;
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rootDesc{};
        rootDesc.NumParameters = 2;
        rootDesc.pParameters = params;
        HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
        if (!d3d12) d3d12 = LoadLibraryW(L"d3d12.dll");
        auto serialize = d3d12 ? ProcAddress<PfnSerializeRootSignature>(d3d12, "D3D12SerializeRootSignature") : nullptr;
        if (!serialize) {
            Log(WG_LOG_ERROR, "D3D12SerializeRootSignature not found");
            return WG_ERROR_DEVICE;
        }
        ID3DBlob* blob = nullptr;
        ID3DBlob* blobError = nullptr;
        HRESULT hr = serialize(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &blobError);
        if (FAILED(hr)) {
            Log(WG_LOG_ERROR, "root signature serialization failed (0x%08lx) %s", static_cast<unsigned long>(hr),
                blobError ? static_cast<const char*>(blobError->GetBufferPointer()) : "");
            if (blobError) blobError->Release();
            return WG_ERROR_DEVICE;
        }
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root));
        blob->Release();
        if (blobError) blobError->Release();
        if (FAILED(hr)) {
            Log(WG_LOG_ERROR, "CreateRootSignature failed (0x%08lx)", static_cast<unsigned long>(hr));
            return WG_ERROR_DEVICE;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root;
        pso.CS.pShaderBytecode = g_shader.bytecode.data();
        pso.CS.BytecodeLength = g_shader.bytecode.size();
        hr = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline));
        if (FAILED(hr)) {
            Log(WG_LOG_ERROR, "CreateComputePipelineState failed (0x%08lx)", static_cast<unsigned long>(hr));
            return WG_ERROR_DEVICE;
        }

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = kDescriptorsPerFrame * kRingFrames;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap));
        if (FAILED(hr)) {
            Log(WG_LOG_ERROR, "CreateDescriptorHeap failed (0x%08lx)", static_cast<unsigned long>(hr));
            return WG_ERROR_DEVICE;
        }
        increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (options.splitTest) {
            const bool v2 = info.interface_version >= 2 && info.struct_size >= offsetof(WgCreateInfo, services) + sizeof(void*);
            services = v2 ? info.services : nullptr;
            if (!services) {
                Log(WG_LOG_WARN, "split_test = 1 but the front end offers no split service (d3d12_hooks = off, or the hooks could not be "
                                 "installed): upscaling without the split");
            } else {
                D3D12_COMMAND_QUEUE_DESC qd{};
                qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
                if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&splitQueue))) ||
                    FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&doneFence))) ||
                    FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&resumeFence)))) {
                    Log(WG_LOG_ERROR, "split_test: creating the queue or fences failed");
                    return WG_ERROR_DEVICE;
                }
                splitQueue->SetName(L"WrongGPU split test (stands in for HIP)");
                Log(WG_LOG_INFO, "split_test: upscaling through split_command_list; done fence %p, resume fence %p, resume signalled "
                                 "by queue %p right after done", static_cast<void*>(doneFence), static_cast<void*>(resumeFence),
                    static_cast<void*>(splitQueue));
            }
        }
        Log(WG_LOG_INFO, "stand-in ready: view=%u filter=%s jitter_compensation=%d warp_weight=%g gain=%g output %ux%u",
            options.view, options.filter ? "lanczos" : "bilinear", options.jitterCompensation ? 1 : 0, options.warpWeight,
            options.gain, outW, outH);
        return WG_OK;
    }

    bool CreateHistory() {
        for (int i = 0; i < 2; ++i) {
            D3D12_HEAP_PROPERTIES heapProps{};
            heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = outW;
            desc.Height = outH;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.Format = kHistoryFormat;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const HRESULT hr = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &desc,
                                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                               IID_PPV_ARGS(&history[i]));
            if (FAILED(hr)) {
                Log(WG_LOG_ERROR, "history texture creation failed (0x%08lx)", static_cast<unsigned long>(hr));
                return false;
            }
            historyState[i] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        historyValid = false;
        return true;
    }

    bool UavStoreSupported(DXGI_FORMAT format) {
        auto it = uavSupport.find(format);
        if (it != uavSupport.end()) return it->second;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{};
        support.Format = format;
        bool ok = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof support)) &&
                  (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
        uavSupport[format] = ok;
        return ok;
    }

    void Srv(ID3D12Resource* resource, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE at) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = format;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MostDetailedMip = 0;
        d.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(resource, &d, at);
    }

    void Uav(ID3D12Resource* resource, DXGI_FORMAT format, D3D12_CPU_DESCRIPTOR_HANDLE at) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = format;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        d.Texture2D.MipSlice = 0;
        device->CreateUnorderedAccessView(resource, nullptr, &d, at);
    }

    WgResult Dispatch(const WgFrame& f) {
        auto* list = static_cast<ID3D12GraphicsCommandList*>(f.command_list);
        auto* color = static_cast<ID3D12Resource*>(f.color.resource);
        auto* depth = static_cast<ID3D12Resource*>(f.depth.resource);
        auto* motion = static_cast<ID3D12Resource*>(f.motion.resource);
        auto* output = static_cast<ID3D12Resource*>(f.output.resource);
        if (!list || !color || !output) return WG_ERROR_PARAMETER;
        if (f.output_width != outW || f.output_height != outH) {
            Log(WG_LOG_WARN, "output size %ux%u differs from create (%ux%u); using it", f.output_width, f.output_height, outW, outH);
        }

        const bool needsMotion = options.view == VIEW_WARP || options.view == VIEW_WARPDIFF || options.view == VIEW_MOTION;
        const bool needsDepth = options.view == VIEW_DEPTH;
        const bool needsHistory = options.view == VIEW_WARP || options.view == VIEW_WARPDIFF;
        if (needsMotion && !motion) {
            Log(WG_LOG_ERROR, "view needs motion vectors but the game gave none");
            return WG_ERROR_PARAMETER;
        }
        if (needsDepth && !depth) {
            Log(WG_LOG_ERROR, "depth view needs depth but the game gave none");
            return WG_ERROR_PARAMETER;
        }
        if (needsHistory && f.output_width == outW && f.output_height == outH && !history[0] && !CreateHistory())
            return WG_ERROR_MEMORY;
        const bool useHistory = needsHistory && history[0] && f.output_width == outW && f.output_height == outH;

        // split_test: everything the game recorded so far runs first (in the game's order), then the "network" (here:
        // nothing; resume follows done at once), then the stand-in's upscale and the rest of the game's list.
        if (services) {
            const uint64_t value = ++splitValue;
            WgSplitRequest request{sizeof(WgSplitRequest), doneFence, value, resumeFence, value};
            const WgResult r = services->split_command_list(services->context, &f, &request);
            if (r == WG_OK) {
                splitQueue->Wait(doneFence, value);
                splitQueue->Signal(resumeFence, value);
                if (splits++ == 0) Log(WG_LOG_INFO, "split_test: first split done (frame %llu)", static_cast<unsigned long long>(f.frame_index));
            } else {
                --splitValue;
                if (splitRefusals++ < 5)
                    Log(WG_LOG_INFO, "split_test: no split this frame (code %d, see the d3d12 hooks lines): upscaling without it", int(r));
            }
        }

        const DXGI_FORMAT outputFormat = static_cast<DXGI_FORMAT>(f.output.format);
        const DXGI_FORMAT outputUav = UavFormat(outputFormat);
        if (!UavStoreSupported(outputUav)) {
            Log(WG_LOG_ERROR, "the GPU cannot store to output format %u through a UAV", unsigned(outputUav));
            return WG_ERROR_UNSUPPORTED;
        }

        // Descriptors for this frame.
        const UINT slot = (ring++ % kRingFrames) * kDescriptorsPerFrame;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
        cpu.ptr += SIZE_T(slot) * increment;
        gpu.ptr += UINT64(slot) * increment;
        auto at = [&](UINT i) {
            D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
            h.ptr += SIZE_T(i) * increment;
            return h;
        };
        Srv(color, static_cast<DXGI_FORMAT>(f.color.format), at(0));
        Srv(needsDepth ? depth : nullptr, needsDepth ? static_cast<DXGI_FORMAT>(f.depth.format) : DXGI_FORMAT_R32_FLOAT, at(1));
        Srv(needsMotion ? motion : nullptr, needsMotion ? static_cast<DXGI_FORMAT>(f.motion.format) : DXGI_FORMAT_R16G16_FLOAT, at(2));
        Srv(useHistory ? history[parity] : nullptr, kHistoryFormat, at(3));
        Uav(output, outputUav, at(4));
        Uav(useHistory ? history[1 - parity] : nullptr, kHistoryFormat, at(5));

        // Barriers into the states the shader needs, and back afterwards (the game's states are restored).
        D3D12_RESOURCE_BARRIER before[8], after[8];
        UINT nb = 0, na = 0;
        ID3D12Resource* seen[4] = {};
        UINT nseen = 0;
        auto readable = [&](ID3D12Resource* r, uint32_t state) {
            if (!r) return;
            for (UINT i = 0; i < nseen; ++i)
                if (seen[i] == r) return;
            seen[nseen++] = r;
            const auto s = static_cast<D3D12_RESOURCE_STATES>(state);
            if (s & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) return;
            const auto target = IsReadOnlyState(s) ? static_cast<D3D12_RESOURCE_STATES>(s | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                                                   : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            before[nb++] = GameTransition(r, s, target);
            after[na++] = GameTransition(r, target, s);
        };
        readable(color, f.color.state);
        if (needsDepth) readable(depth, f.depth.state);
        if (needsMotion) readable(motion, f.motion.state);
        const auto outState = static_cast<D3D12_RESOURCE_STATES>(f.output.state);
        if (outState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            before[nb++] = GameTransition(output, outState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            after[na++] = GameTransition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outState);
        } else {
            before[nb++] = UavBarrier(output);
            after[na++] = UavBarrier(output);
        }
        if (useHistory) {
            const uint32_t read = parity, write = 1 - parity;
            if (historyState[read] != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
                before[nb++] = Transition(history[read], historyState[read], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                historyState[read] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            if (historyState[write] != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                before[nb++] = Transition(history[write], historyState[write], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                historyState[write] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
        }

        StandInConstants c{};
        c.outW = f.output_width;
        c.outH = f.output_height;
        c.renderX = f.render_x;
        c.renderY = f.render_y;
        c.renderW = f.render_width;
        c.renderH = f.render_height;
        c.view = options.view;
        c.filterMode = options.filter;
        c.jitterX = f.jitter[0];
        c.jitterY = f.jitter[1];
        c.mvScaleX = f.motion_scale[0];
        c.mvScaleY = f.motion_scale[1];
        c.mvOffX = f.motion_offset[0];
        c.mvOffY = f.motion_offset[1];
        c.warpWeight = options.warpWeight;
        c.gain = options.gain;
        c.flags = (options.jitterCompensation ? F_JITTER_COMP : 0u) | ((createFlags & WG_CREATE_MOTION_OUTPUT_RES) ? F_MOTION_OUTRES : 0u) |
                  (useHistory && historyValid ? F_HISTORY_VALID : 0u) | (outputFormat != outputUav ? F_OUTPUT_SRGB : 0u) |
                  ((createFlags & WG_CREATE_DEPTH_INVERTED) ? F_DEPTH_INVERTED : 0u) | (f.reset ? F_RESET : 0u);
        c.motionW = f.motion.width ? f.motion.width : 1;
        c.motionH = f.motion.height ? f.motion.height : 1;

        if (nb) list->ResourceBarrier(nb, before);
        list->SetComputeRootSignature(root);
        list->SetDescriptorHeaps(1, &heap);
        list->SetPipelineState(pipeline);
        list->SetComputeRoot32BitConstants(0, sizeof c / 4, &c, 0);
        list->SetComputeRootDescriptorTable(1, gpu);
        list->Dispatch((c.outW + 7) / 8, (c.outH + 7) / 8, 1);
        if (na) list->ResourceBarrier(na, after);

        if (useHistory) {
            parity = 1 - parity;
            historyValid = true;
        }
        if (frames++ == 0)
            Log(WG_LOG_INFO, "first frame recorded: R=%ux%u O=%ux%u color fmt %u output fmt %u (uav %u) %u+%u barriers",
                f.render_width, f.render_height, f.output_width, f.output_height, f.color.format, f.output.format,
                unsigned(outputUav), nb, na);
        return WG_OK;
    }

    void Destroy() {
        g_graveyard.Add(pipeline);
        g_graveyard.Add(root);
        g_graveyard.Add(heap);
        g_graveyard.Add(history[0]);
        g_graveyard.Add(history[1]);
        g_graveyard.Add(splitQueue);
        g_graveyard.Add(doneFence);
        g_graveyard.Add(resumeFence);
        g_graveyard.Add(device);
    }

    uint64_t GpuBytes() const { return history[0] ? uint64_t(outW) * outH * 8u * 2u : 0u; }
};

WgResult StandInCreate(const WgCreateInfo* info, WgUpscaler** out) {
    g_graveyard.ReleaseOld();
    if (!info || !out || !info->log || info->struct_size < offsetof(WgCreateInfo, services)) return WG_ERROR_PARAMETER;
    auto* s = new (std::nothrow) StandIn;
    if (!s) return WG_ERROR_MEMORY;
    const WgResult r = s->Create(*info);
    if (r != WG_OK) {
        s->Destroy();
        delete s;
        return r;
    }
    *out = reinterpret_cast<WgUpscaler*>(s);
    return WG_OK;
}

WgResult StandInDispatch(WgUpscaler* upscaler, const WgFrame* frame) {
    if (!upscaler || !frame || frame->struct_size < offsetof(WgFrame, command_queue)) return WG_ERROR_PARAMETER;
    return reinterpret_cast<StandIn*>(upscaler)->Dispatch(*frame);
}

void StandInDestroy(WgUpscaler* upscaler) {
    auto* s = reinterpret_cast<StandIn*>(upscaler);
    if (s) {
        s->Destroy();
        delete s;
    }
    g_graveyard.ReleaseOld();
}

uint64_t StandInGpuBytes(WgUpscaler* upscaler) { return reinterpret_cast<StandIn*>(upscaler)->GpuBytes(); }

const WgUpscalerBackend kStandIn = {sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "WrongGPU stand-in upscaler",
                                    StandInCreate, StandInDispatch, StandInDestroy, StandInGpuBytes, 0};
const WgUpscalerBackend kStandInSplitTest = {sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "WrongGPU stand-in upscaler",
                                             StandInCreate, StandInDispatch, StandInDestroy, StandInGpuBytes, WG_BACKEND_NEEDS_SPLIT};

}  // namespace

const WgUpscalerBackend* GetStandInBackend(bool splitTest) { return splitTest ? &kStandInSplitTest : &kStandIn; }

}  // namespace wgfsr
