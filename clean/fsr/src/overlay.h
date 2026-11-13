// SPDX-License-Identifier: MIT
#pragma once
// In-game overlay: shows the player whether WrongGPU's upscaler is running ("WrongGPU - model M: ACTIVE"), or why AMD's
// FSR upscales instead, with the render and output sizes, the frame rate and the upscaler's GPU time per frame. WrongGPU's own code: the swap chain's
// Present / Present1 / ResizeBuffers / ResizeBuffers1 / SetColorSpace1 are hooked by COM vtable patching (chained to
// what the vtable held before), the text is drawn with our own small D3D12 pipeline on the swap chain's queue just
// before the game's Present, from a glyph atlas rasterised with GDI from a Windows font at run time (nothing shipped),
// shaders compiled at run time with Windows' d3dcompiler_47.dll.
//
// The upscale status comes from every WrongGPU DLL of the process (Platform::status) and is kept in a named per-process
// shared block; the first DLL that sees a D3D12 device installs the hooks and draws. Back buffers R8G8B8A8 / B8G8R8A8 /
// R10G10B10A2 UNORM (SDR, or HDR10 when the swap chain's colour space is ST.2084) and R16G16B16A16_FLOAT (scRGB) are
// drawn; other swap chains are left alone (logged once). Any failure disables the overlay; it never blocks a frame:
// a back buffer whose previous overlay work is still on the GPU is skipped.
#include "config.h"
#include "status.h"

namespace wgwin {

// This DLL's settings (overlay, overlay_key).
void ConfigureOverlay(const wgfsr::Config& config);

// Platform::status: records the dispatch's state; installs the hooks at the first status with a device.
void OverlayStatus(const wgfsr::UpscaleStatus& status);
// Read the latest successful provider dispatch, without installing graphics hooks.
bool ReadOverlayProviderStatus(wgfsr::UpscaleStatus& status);

// Platform::gpuTimer: GPU timestamps recorded into the game's command list before and after an upscale dispatch; the
// time in between is shown as the upscaler time. Nested calls (FSR 3.0 calling its upscaler) count once.
void OverlayGpuTimer(void* commandList, bool end);

// Platform::compareAmd: true while the player has switched to AMD's FSR with compare_key.
bool OverlayCompareAmd();

// The upscaler time the overlay shows (milliseconds, smoothed; 0 when none measured in the last second). For tests.
double OverlayUpscalerMs();

// What the overlay key does (hide; show with the full panel for 10 s). For tests.
void OverlayToggle();

}  // namespace wgwin
