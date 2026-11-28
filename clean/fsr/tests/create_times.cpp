// SPDX-License-Identifier: GPL-3.0-only
// create_times: real network-runtime creates on this PC's D3D12 adapter, timed (rc3: the in-game model switch of
// 2026-11-27, where a Model M create took 11.2 s and Model L could not start). A GPU tool: never run by ctest; run it by
// hand through tools/gpu_slot.py (60 s cap), never while a game runs.
//
//     wg_create_times <wronggpu_runtime.dll> <model folder> <case> [<case> ...]
//
// <model folder> holds model.wgcm (Model M), K\model.wgcm and L\model.wgcm (the launcher's layout); it is passed to the
// runtime as model_dir. A case is
//     <M|K|L>@<render W>x<render H>-><output W>x<output H>   one create, then destroy
//     <M|K|L>@native-><output W>x<output H>                   max render = output (as Cyberpunk 2077 declares it)
//     switch:<models, e.g. M,L,K,M>@<render>-><output>        each create while the previous upscaler still exists, the
//                                                             previous one destroyed after it (as OptiScaler switches)
// The options are the front end's for Cyberpunk 2077 in RC2 (executor=dx12, m_mode=fast, k_mode=exact, l_mode=fast).
// One line per create: the model, sizes, result, milliseconds, and the runtime's error / "create steps" lines.
// Exit code: 0 when every create worked, 1 when one failed, 2 for a usage or setup error.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../include/wronggpu_upscaler.h"

namespace {

struct Captured {
    std::vector<std::string> lines;
};

void CaptureLog(void* user, int level, const char* message) {
    auto* c = static_cast<Captured*>(user);
    const char* tag = level == WG_LOG_ERROR ? "ERROR" : level == WG_LOG_WARN ? "warning" : "info";
    c->lines.push_back(std::string(tag) + ": " + (message ? message : ""));
}

bool ParseSize(const std::string& s, uint32_t* w, uint32_t* h) {
    const size_t x = s.find('x');
    if (x == std::string::npos) return false;
    *w = static_cast<uint32_t>(std::strtoul(s.substr(0, x).c_str(), nullptr, 10));
    *h = static_cast<uint32_t>(std::strtoul(s.substr(x + 1).c_str(), nullptr, 10));
    return *w && *h;
}

std::string OptionsFor(char model, const std::string& modelDir) {
    std::string o = "executor=dx12;output_filter=default;pre_exposure=scale;sparkle_guard=off;ready_wait=fence;";
    o += model == 'K' ? "k_mode=exact" : model == 'L' ? "l_mode=fast" : "m_mode=fast";
    o += std::string(";model=") + model + ";model_dir=" + modelDir;
    return o;
}

struct Live {
    WgUpscaler* upscaler = nullptr;
    Captured log;
    std::string options;
};

// One create; prints its line. *ok: the create worked.
WgUpscaler* Create(const WgUpscalerBackend* b, ID3D12Device* device, char model, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh,
                   const std::string& modelDir, const char* label, Live* live, bool* ok) {
    live->options = OptionsFor(model, modelDir);
    WgCreateInfo info{};
    info.struct_size = sizeof(info);
    info.interface_version = WG_UPSCALER_INTERFACE_VERSION;
    info.d3d12_device = device;
    info.max_render_width = rw;
    info.max_render_height = rh;
    info.output_width = ow;
    info.output_height = oh;
    info.flags = WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED | WG_CREATE_AUTO_EXPOSURE;  // Cyberpunk 2077's (flags 0x7)
    info.log = &CaptureLog;
    info.log_user = &live->log;
    info.options = live->options.c_str();
    info.services = nullptr;  // executor dx12 needs no command-list split
    WgUpscaler* u = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const WgResult r = b->create(&info, &u);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    *ok = r == WG_OK && u;
    std::printf("%s Model %c max render %ux%u output %ux%u: %s in %.0f ms%s\n", label, model, rw, rh, ow, oh,
                *ok ? "created" : "FAILED", ms,
                *ok && b->gpu_memory_bytes ? (" (GPU memory " + std::to_string(b->gpu_memory_bytes(u) >> 20) + " MiB)").c_str() : "");
    for (const std::string& l : live->log.lines)
        if (l.rfind("ERROR", 0) == 0 || l.find("create steps") != std::string::npos || l.find("declared max render") != std::string::npos)
            std::printf("    %s\n", l.c_str());
    std::fflush(stdout);
    live->log.lines.clear();
    return *ok ? u : nullptr;
}

void Destroy(const WgUpscalerBackend* b, WgUpscaler* u, const char* label) {
    if (!u) return;
    const auto t0 = std::chrono::steady_clock::now();
    b->destroy(u);
    std::printf("%s destroyed in %.0f ms\n", label,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: wg_create_times <wronggpu_runtime.dll> <model folder> <case> [<case> ...]\n");
        return 2;
    }
    IDXGIFactory6* factory = nullptr;
    if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory6), reinterpret_cast<void**>(&factory)))) {
        std::printf("setup: no DXGI factory\n");
        return 2;
    }
    IDXGIAdapter1* adapter = nullptr;
    if (FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, __uuidof(IDXGIAdapter1),
                                                   reinterpret_cast<void**>(&adapter)))) {
        std::printf("setup: no adapter\n");
        return 2;
    }
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    ID3D12Device* device = nullptr;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device)))) {
        std::printf("setup: D3D12CreateDevice failed\n");
        return 2;
    }
    char name[128] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof name - 1, nullptr, nullptr);
    std::printf("adapter: %s\n", name);
    HMODULE runtime = LoadLibraryA(argv[1]);
    auto get = runtime ? reinterpret_cast<PfnWgGetUpscalerBackend>(reinterpret_cast<void*>(GetProcAddress(runtime, "wgGetUpscalerBackend")))
                       : nullptr;
    const WgUpscalerBackend* b = get ? get(WG_UPSCALER_INTERFACE_VERSION) : nullptr;
    if (!b) {
        std::printf("setup: %s is no WrongGPU runtime (wgGetUpscalerBackend)\n", argv[1]);
        return 2;
    }
    const std::string modelDir = argv[2];
    bool allOk = true;
    for (int i = 3; i < argc; ++i) {
        std::string c = argv[i];
        const bool sw = c.rfind("switch:", 0) == 0;
        if (sw) c = c.substr(7);
        const size_t at = c.find('@'), arrow = c.find("->");
        uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
        if (at == std::string::npos || arrow == std::string::npos || arrow < at || !ParseSize(c.substr(arrow + 2), &ow, &oh)) {
            std::printf("usage: bad case %s\n", argv[i]);
            return 2;
        }
        const std::string render = c.substr(at + 1, arrow - at - 1);
        if (render == "native") {
            rw = ow;
            rh = oh;
        } else if (!ParseSize(render, &rw, &rh)) {
            std::printf("usage: bad render size in %s\n", argv[i]);
            return 2;
        }
        std::vector<char> models;
        for (char m : c.substr(0, at))
            if (m == 'M' || m == 'K' || m == 'L') models.push_back(m);
        if (models.empty() || (!sw && models.size() != 1)) {
            std::printf("usage: bad models in %s\n", argv[i]);
            return 2;
        }
        if (!sw) {
            Live live;
            bool ok = false;
            WgUpscaler* u = Create(b, device, models[0], rw, rh, ow, oh, modelDir, "create", &live, &ok);
            allOk = allOk && ok;
            Destroy(b, u, "  ");
            continue;
        }
        // a switch sequence: the next create while the previous upscaler exists, then the previous one is destroyed
        std::printf("switch %s at %ux%u -> %ux%u\n", c.substr(0, at).c_str(), rw, rh, ow, oh);
        Live lives[2];
        WgUpscaler* previous = nullptr;
        int slot = 0;
        for (size_t k = 0; k < models.size(); ++k) {
            bool ok = false;
            const std::string label = "  step " + std::to_string(k + 1);
            WgUpscaler* u = Create(b, device, models[k], rw, rh, ow, oh, modelDir, label.c_str(), &lives[slot], &ok);
            allOk = allOk && ok;
            if (u) {  // (a failed step keeps the previous upscaler, as the front end's hand-over does)
                Destroy(b, previous, "    previous");
                previous = u;
                slot ^= 1;
            }
        }
        Destroy(b, previous, "  last");
    }
    device->Release();
    adapter->Release();
    factory->Release();
    std::printf("%s\n", allOk ? "all creates worked" : "a create FAILED");
    return allOk ? 0 : 1;
}
