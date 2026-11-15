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

static std::wstring FfxChoicePath(const std::wstring& ini) {
    const size_t slash = ini.find_last_of(L"\\/");
    return (slash == std::wstring::npos ? std::wstring(L".") : ini.substr(0, slash)) + L"\\wronggpu-ffx-choice.ini";
}

bool LoadFfxChoice(const std::wstring& ini, wgfsr::FfxModelChoice* choice) {
    HANDLE h = CreateFileW(FfxChoicePath(ini).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    char buf[1024];
    DWORD got = 0;
    const bool ok = ReadFile(h, buf, sizeof buf, &got, nullptr) != 0;
    CloseHandle(h);
    if (!ok) return false;
    if (choice) *choice = wgfsr::ParseFfxChoiceFile(std::string(buf, got));
    return true;
}

void SaveFfxChoice(const std::wstring& ini, const wgfsr::FfxModelChoice& choice) {
    const std::string text = wgfsr::FormatFfxChoiceFile(choice);
    HANDLE h = CreateFileW(FfxChoicePath(ini).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
                           nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(h, text.data(), DWORD(text.size()), &written, nullptr);
    CloseHandle(h);
}

}  // namespace wgwin
