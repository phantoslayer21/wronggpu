// SPDX-License-Identifier: MIT
// Keys of the network runtime's options (runtime_options in wronggpu.ini, WG_RUNTIME_OPTIONS), for the .ini reader: a
// line "network = fused" in [fsr] is a runtime option written on its own line, not a front-end setting (config.cpp
// ParseConfig appends it to runtime_options with a warning). The list follows the runtime 0.5.1: the D3D12 interop's
// own keys (clean/runtime/interop/backend.cpp parse_options), the core runtime's (clean/runtime/runtime.cpp
// runtime_options, model selection) and model K's (clean/k/runtime/runtime_k.cpp, backend.cpp k_input). Keys the
// front end reads itself (view, filter, ...) are matched before this list and never reach it.
#pragma once

#include <cstring>
#include <string>

namespace wgfsr {

inline bool IsRuntimeOptionKey(const std::string& key) {
    static const char* const kKeys[] = {
        // D3D12 interop
        "sync", "executor", "model_dir", "kernels_dir", "map", "rocm_dir", "swin", "model_format", "test_handles",
        "test_keep", "share_path", "timing_log", "timing_stages", "watchdog", "ready_wait", "capture_dir",
        "capture_frames", "sparkle_guard", "sparkle_guard_thr", "recon", "motion_outres", "done_mode", "spin_polls",
        "handover", "input_copy", "output_copy", "test_view", "hip_switch",
        // core runtime (model M) and model selection
        "model", "graph", "stream_priority", "post_bands", "post_tf", "output_filter", "pre_exposure", "d1d0_p12",
        "dec_rdiv", "dec_bdiv", "swin_resident", "post_tf_cpairs", "post_diet", "tf_vdiet", "post_idx", "post_trim",
        "post_sel",
        // model K
        "model_k_dir", "model_k_map", "network", "fp8_stages", "k_fast", "k_input", "k_diag", "k_head", "input_tiled"};
    for (const char* k : kKeys)
        if (key == k) return true;
    return false;
}

}  // namespace wgfsr
