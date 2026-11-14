// SPDX-License-Identifier: MIT
#include "asi_bridge.h"

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "../asi/asi_abi.h"

namespace wgwin {

bool QueryAsiPlugin(wgfsr::PluginModelChoice* choice) {
    if (choice) *choice = wgfsr::PluginModelChoice{};
    HMODULE m = GetModuleHandleW(WRONGGPU_ASI_MODULE);  // present only if OptiScaler loaded our plugin; no reference taken
    if (!m) return false;
    if (auto query = reinterpret_cast<PfnWrongGpuAsiQueryChoiceV1>(reinterpret_cast<void*>(GetProcAddress(m, WRONGGPU_ASI_CHOICE_QUERY)))) {
        WrongGpuAsiChoiceV1 c{};
        c.size = sizeof(c);
        c.version = WRONGGPU_ASI_CHOICE_VERSION;
        if (query(&c, sizeof(c)) != WG_ASI_STATUS_OK) return false;
        if (choice) {
            choice->model = char(c.model);
            choice->explicitChoice = c.explicit_choice != 0;
            choice->chosenAt = c.chosen_at;
        }
        return c.hooked != 0;
    }
    auto query = reinterpret_cast<PfnWrongGpuAsiQuery>(reinterpret_cast<void*>(GetProcAddress(m, WRONGGPU_ASI_QUERY)));
    if (!query) return false;
    int hooked = 0;
    char mdl = 0;
    if (query(&hooked, &mdl) < int(WRONGGPU_ASI_ABI)) return false;  // older/unknown ABI: ignore
    if (choice) {  // an earlier plugin: its default entry is Model M; any other entry was chosen, at an unknown time
        choice->model = mdl;
        choice->explicitChoice = mdl != 'M';
        choice->chosenAt = 0;
    }
    return hooked != 0;
}

void ShowAsiModel(char model) {
    HMODULE m = GetModuleHandleW(WRONGGPU_ASI_MODULE);
    if (!m) return;
    if (auto show = reinterpret_cast<PfnWrongGpuAsiShowModelV1>(reinterpret_cast<void*>(GetProcAddress(m, WRONGGPU_ASI_SHOW_MODEL))))
        show(model);
}

}  // namespace wgwin
