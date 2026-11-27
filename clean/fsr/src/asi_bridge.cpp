// SPDX-License-Identifier: GPL-3.0-only
#include "asi_bridge.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "../asi/asi_abi.h"

namespace wgwin {

bool QueryAsiPlugin(char* model) {
    if (model) *model = 0;
    HMODULE m = GetModuleHandleW(WRONGGPU_ASI_MODULE);  // present only if OptiScaler loaded our plugin; no reference taken
    if (!m) return false;
    auto query = reinterpret_cast<PfnWrongGpuAsiQuery>(reinterpret_cast<void*>(GetProcAddress(m, WRONGGPU_ASI_QUERY)));
    if (!query) return false;
    int hooked = 0;
    char mdl = 0;
    if (query(&hooked, &mdl) < int(WRONGGPU_ASI_ABI)) return false;  // older/unknown ABI: ignore
    if (model) *model = mdl;
    return hooked != 0;
}

void ShowAsiModel(char model) {
    HMODULE m = GetModuleHandleW(WRONGGPU_ASI_MODULE);
    if (!m) return;
    if (auto show = reinterpret_cast<PfnWrongGpuAsiShowModelV1>(reinterpret_cast<void*>(GetProcAddress(m, WRONGGPU_ASI_SHOW_MODEL))))
        show(model);
}

void SetAsiChoiceRecord(const std::wstring& path) {
    HMODULE m = GetModuleHandleW(WRONGGPU_ASI_MODULE);
    if (!m) return;
    if (auto set = reinterpret_cast<PfnWrongGpuAsiSetChoiceRecordV1>(reinterpret_cast<void*>(GetProcAddress(m, WRONGGPU_ASI_SET_CHOICE_RECORD))))
        set(path.c_str());
}

}  // namespace wgwin
