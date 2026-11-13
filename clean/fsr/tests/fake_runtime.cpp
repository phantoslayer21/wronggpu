// SPDX-License-Identifier: MIT
// Test double of the WrongGPU network runtime (wronggpu_runtime.dll, spec 10) for the end-to-end tests of
// `upscaler = network`: it exports wgGetUpscalerBackend like the real runtime and records nothing on the GPU except,
// in mode "split", the same-frame split (a queue signals resume right after done, as HIP would after the network).
// Behaviour from the environment variable WG_FAKE_RUNTIME:
//   ok            interface version 2, no split; every dispatch succeeds
//   fail_create   create fails with WG_ERROR_DEVICE (logs why, as the runtime would)
//   fail_dispatch dispatch fails with WG_ERROR_DEVICE from frame 1 on
//   split         interface version 2 with WG_BACKEND_NEEDS_SPLIT; every dispatch splits the game's list
//   split_fail_create  WG_BACKEND_NEEDS_SPLIT, and create fails as with fail_create (no model: the D3D12 hooks must not
//                 be installed for it)
//   v1            answers only interface version 1
//   bad           answers with tables that lack dispatch
// Built into the test folder only (dlltest/<config>); never part of a payload.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>

#include "../include/wronggpu_upscaler.h"

namespace {

char g_mode[32] = "ok";

struct Fake {
    WgLogFn log = nullptr;
    void* logUser = nullptr;
    const WgFrontEndServices* services = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* done = nullptr;
    ID3D12Fence* resume = nullptr;
    uint64_t value = 0;
    uint64_t frames = 0;
};

bool Mode(const char* m) { return std::strcmp(g_mode, m) == 0; }

WgResult Create(const WgCreateInfo* info, WgUpscaler** out) {
    if (!info || !out || !info->log) return WG_ERROR_PARAMETER;
    if (Mode("fail_create") || Mode("split_fail_create")) {
        info->log(info->log_user, WG_LOG_ERROR, "fake runtime: no HIP device found (hipGetDeviceCount returned 0)");
        return WG_ERROR_DEVICE;
    }
    auto* f = new (std::nothrow) Fake;
    if (!f) return WG_ERROR_MEMORY;
    f->log = info->log;
    f->logUser = info->log_user;
    if (Mode("split")) {
        f->services = info->interface_version >= 2 && info->struct_size >= offsetof(WgCreateInfo, services) + sizeof(void*) ? info->services : nullptr;
        auto* device = static_cast<ID3D12Device*>(info->d3d12_device);
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (!f->services || FAILED(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&f->queue))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&f->done))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&f->resume)))) {
            info->log(info->log_user, WG_LOG_ERROR, f->services ? "fake runtime: queue or fences not created" : "fake runtime: no split service");
            delete f;
            return WG_ERROR_UNSUPPORTED;
        }
    }
    char text[1024];
    std::snprintf(text, sizeof text, "fake runtime created (options: %s)", info->options ? info->options : "(null)");
    info->log(info->log_user, WG_LOG_INFO, text);
    *out = reinterpret_cast<WgUpscaler*>(f);
    return WG_OK;
}

WgResult Dispatch(WgUpscaler* u, const WgFrame* frame) {
    auto* f = reinterpret_cast<Fake*>(u);
    if (!f || !frame) return WG_ERROR_PARAMETER;
    const uint64_t index = f->frames++;
    if (Mode("fail_dispatch") && index >= 1) {
        f->log(f->logUser, WG_LOG_ERROR, "fake runtime: hipModuleLaunchKernel failed (hipErrorLaunchFailure)");
        return WG_ERROR_DEVICE;
    }
    if (f->services) {
        const uint64_t v = ++f->value;
        WgSplitRequest r{sizeof(WgSplitRequest), f->done, v, f->resume, v};
        const WgResult s = f->services->split_command_list(f->services->context, frame, &r);
        if (s != WG_OK) {
            --f->value;
            return s;  // e.g. WG_ERROR_NOT_READY: the front end upscales this frame with the stand-in
        }
        f->queue->Wait(f->done, v);
        f->queue->Signal(f->resume, v);
    }
    return WG_OK;
}

void Destroy(WgUpscaler* u) {
    auto* f = reinterpret_cast<Fake*>(u);
    if (!f) return;
    if (f->queue) {  // wait until the queue has passed its last wait/signal
        ID3D12Fence* idle = nullptr;
        ID3D12Device* device = nullptr;
        if (SUCCEEDED(f->queue->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void**>(&device))) &&
            SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&idle)))) {
            f->queue->Signal(idle, 1);
            for (int i = 0; i < 5000 && idle->GetCompletedValue() < 1; ++i) Sleep(1);
            idle->Release();
        }
        if (device) device->Release();
        f->queue->Release();
    }
    if (f->done) f->done->Release();
    if (f->resume) f->resume->Release();
    delete f;
}

WgUpscalerBackend g_v2 = {sizeof(WgUpscalerBackend), 2, "WrongGPU network (fake runtime)", Create, Dispatch, Destroy, nullptr, 0};
WgUpscalerBackend g_v1 = {offsetof(WgUpscalerBackend, backend_flags), 1, "WrongGPU network (fake runtime, v1)", Create, Dispatch, Destroy,
                          nullptr, 0};

}  // namespace

extern "C" __declspec(dllexport) const WgUpscalerBackend* wgGetUpscalerBackend(uint32_t version) {
    GetEnvironmentVariableA("WG_FAKE_RUNTIME", g_mode, sizeof g_mode);
    if (Mode("bad")) {
        static WgUpscalerBackend bad2 = g_v2, bad1 = g_v1;
        bad2.dispatch = nullptr;
        bad1.dispatch = nullptr;
        return version == 2 ? &bad2 : version == 1 ? &bad1 : nullptr;
    }
    if (Mode("v1")) return version == 1 ? &g_v1 : nullptr;
    g_v2.backend_flags = Mode("split") || Mode("split_fail_create") ? WG_BACKEND_NEEDS_SPLIT : 0;
    return version == 2 ? &g_v2 : nullptr;
}
