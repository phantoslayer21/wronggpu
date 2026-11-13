// SPDX-License-Identifier: MIT
#pragma once
// D3D12 hooks of the FSR front end for same-frame execution (interface version 2, ANSWERS.md spec 20 item 16):
// WrongGPU's own COM vtable patches, each chained to whatever the vtable entry held before (no hook library).
//
//   - ID3D12CommandQueue::ExecuteCommandLists: the submission trace (test 20-T4) and the release of split lists in the
//     game's own submission order;
//   - ID3D12GraphicsCommandList setters, Reset, ClearState, queries, render passes, ExecuteBundle: the bindings of every
//     command list, so a split list can be reset and its state restored;
//   - ID3D12Device::CreateCommandList: the allocator of lists recorded right after creation.
//
// The hooks are process-wide: of WrongGPU's DLLs in a process (amd_fidelityfx_dx12, ffx_fsr3, ffx_fsr3upscaler) the
// first one that needs them installs and owns them (found through a named per-process file mapping); the others use
// the owner's. They are installed when a backend sets WG_BACKEND_NEEDS_SPLIT (d3d12_hooks = auto), at the first upscale
// dispatch with d3d12_hooks = on, and never with d3d12_hooks = off.
#include <cstdint>

#include "../include/wronggpu_upscaler.h"
#include "config.h"

namespace wgwin {

// This DLL's settings (d3d12_hooks, split_timeout_ms, log_dispatches, log_max_mb). The owner uses its own.
void ConfigureHooks(const wgfsr::Config& config);

// platform_.services: the split service for a backend that sets WG_BACKEND_NEEDS_SPLIT (installs the hooks on
// `device`), or null (version 1 backend, no flag, d3d12_hooks = off, hooks not installable).
const WgFrontEndServices* SplitServices(const WgUpscalerBackend* backend, void* device);

// The FidelityFX front end's pair (frontend.h Platform::services / activateServices): the split service a backend that
// sets WG_BACKEND_NEEDS_SPLIT gets in its create info, without installing anything (null when unavailable, or when
// an earlier installation failed), and the installation of the hooks, called only after that backend's create
// succeeded (true also when nothing needs installing). A failed create, or AMD's FSR, never patches a vtable.
const WgFrontEndServices* SplitServicesDeferred(const WgUpscalerBackend* backend, void* device);
bool ActivateSplitServices(const WgUpscalerBackend* backend, void* device);

// platform_.noteDispatch: an upscale dispatch recorded into `commandList` (any route, also AMD's). Installs the hooks
// with d3d12_hooks = on, records the list for the submission trace, and fills frame->command_queue and
// frame->command_list_type when `frame` is given.
void NoteDispatch(void* commandList, WgFrame* frame);

// platform_.splitBlocked: the reason a dispatch into `commandList` can never be split (the list has its own vtable
// carrying another hook layer's entries, which WrongGPU leaves untouched), or null. The text stays valid.
const char* ListSplitBlocked(void* commandList);

}  // namespace wgwin
