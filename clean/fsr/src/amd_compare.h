// SPDX-License-Identifier: MIT
#pragma once
// AMD's upscaler for the player's comparison (compare_key): a context of an AMD FidelityFX API DLL, made from the
// create parameters WrongGPU saved for its own context, with the newest provider that DLL offers on this GPU. The DLL
// is AMD's FidelityFX SDK 2 upscaler DLL when the player has one (setting compare_dll; FSR 4 on RDNA 4), else AMD's
// original amd_fidelityfx_dx12.dll of the game (FSR 3.1).
//
// No Windows or D3D12 code here.
#include <string>

#include "ffx_headers.h"
#include "param_map.h"

namespace wgfsr {

// The five FidelityFX API entry points of an AMD DLL (the game's original amd_fidelityfx_dx12.dll, renamed to
// amd_fidelityfx_dx12.wronggpu-original.dll by the installer, spec 30; or the comparison DLL).
struct OriginalApi {
    PfnFfxCreateContext createContext = nullptr;
    PfnFfxDestroyContext destroyContext = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    std::string source;  // the DLL's path and what it is, for the log (empty: not said)
};

struct AmdComparison {
    const OriginalApi* api = nullptr;
    ffxContext context = nullptr;
    std::string name;          // "AMD FSR 4.1.1"
    bool unavailable = false;  // creation failed once: not tried again
};

// Creates the comparison context on `device` with `api` (the provider with the highest version AMD's DLL lists for
// upscaling). False, with the reason in `log`, when it cannot; `log` also gets the provider list.
bool CreateComparison(AmdComparison& c, const OriginalApi* api, const UpscaleCreateParams& params, void* device,
                      ffxApiMessage fpMessage, std::string& log);

// One frame on the comparison context; reset: AMD discards its history (first frame after a switch).
ffxReturnCode_t DispatchComparison(AmdComparison& c, const ffxDispatchDescUpscale& dispatch, bool reset);

void DestroyComparison(AmdComparison& c);

}  // namespace wgfsr
