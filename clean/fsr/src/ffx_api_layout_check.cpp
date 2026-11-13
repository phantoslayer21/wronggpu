// SPDX-License-Identifier: MIT
// Compile-time check (Windows builds) of the front end's view of the DX12 backend description against AMD's
// FidelityFX SDK 1.1.4 header ffx_api/dx12/ffx_api_dx12.h. That header also defines two inline helper functions, so it
// is included only here: this file belongs to the wg_sdk_layout_check object library, which is compiled with every
// build but linked into no DLL (owner's rule: no code from AMD's headers in a release binary).
#include <d3d12.h>

#include <cstddef>

#include "ffx_headers.h"
#include "ffx_api/dx12/ffx_api_dx12.h"

using wgfsr::BackendDx12DescView;

static_assert(offsetof(BackendDx12DescView, device) == offsetof(ffxCreateBackendDX12Desc, device), "backend desc layout");
static_assert(sizeof(BackendDx12DescView) == sizeof(ffxCreateBackendDX12Desc), "backend desc size");
static_assert(wgfsr::kCreateBackendDx12 == FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12, "backend desc type");
