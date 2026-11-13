// SPDX-License-Identifier: MIT
#include "amd_provider.h"

#include <cwctype>

namespace wgfsr {

namespace {

std::wstring Key(const std::wstring& path) {
    std::wstring k = path;
    for (wchar_t& c : k) c = c == L'/' ? L'\\' : wchar_t(std::towlower(c));
    while (k.size() > 3 && k.back() == L'\\') k.pop_back();
    return k;
}

std::wstring Join(const std::wstring& dir, const wchar_t* name) {
    if (dir.empty()) return std::wstring();
    const wchar_t last = dir.back();
    return dir + (last == L'\\' || last == L'/' ? L"" : L"\\") + name;
}

bool NameIs(const std::wstring& path, const wchar_t* name) {
    const std::wstring k = Key(path);
    const size_t slash = k.find_last_of(L'\\');
    return k.compare(slash == std::wstring::npos ? 0 : slash + 1, std::wstring::npos, name) == 0;
}

}  // namespace

const char* ProbeResultName(ProbeResult r) {
    switch (r) {
    case ProbeResult::Ok: return "usable";
    case ProbeResult::Missing: return "not there";
    case ProbeResult::Ours: return "a WrongGPU DLL";
    case ProbeResult::NotLoaded: return "not loaded";
    case ProbeResult::Redirected: return "redirected by another module";
    case ProbeResult::Hooked: return "hooked by another module";
    case ProbeResult::NoExports: return "no FidelityFX API exports";
    }
    return "?";
}

std::vector<ProviderCandidate> ProviderCandidates(const ProviderInputs& in) {
    std::vector<ProviderCandidate> out;
    std::vector<std::wstring> keys;
    auto add = [&](const std::wstring& path, const char* source) {
        if (path.empty()) return;
        const std::wstring k = Key(path);
        for (const std::wstring& seen : keys)
            if (seen == k) return;
        keys.push_back(k);
        out.push_back({path, source});
    };
    add(in.compareDll, "setting compare_dll");
    add(Join(in.moduleDir, L"wronggpu-amd-compare.dll"), "the installer's private copy of AMD's DLL");
    add(Join(in.moduleDir, L"amd_fidelityfx_upscaler_dx12.dll"), "AMD's FSR 4 upscaler DLL next to WrongGPU");
    add(in.installerFound, "the FSR 4 DLL the installer found on this PC");
    add(Join(in.moduleDir, L"amd_fidelityfx_dx12.wronggpu-original.dll"), "the game's original AMD DLL (renamed)");
    for (const std::wstring& p : in.sdkDlls)
        if (NameIs(p, L"amd_fidelityfx_upscaler_dx12.dll")) add(p, "the game's AMD FidelityFX SDK upscaler DLL");
    add(Join(in.exeDir, L"amd_fidelityfx_dx12.dll"), "the game's amd_fidelityfx_dx12.dll");
    add(Join(in.moduleDir, L"amd_fidelityfx_dx12.dll"), "the game's amd_fidelityfx_dx12.dll");
    for (const std::wstring& p : in.sdkDlls)
        if (NameIs(p, L"amd_fidelityfx_loader_dx12.dll")) add(p, "the game's AMD FidelityFX SDK loader DLL");
    return out;
}

ProviderChoice ChooseProvider(const std::vector<ProviderCandidate>& candidates,
                              const std::function<ProbeResult(const ProviderCandidate&, std::string& detail)>& probe,
                              const std::function<std::string(const std::wstring&)>& narrow) {
    ProviderChoice choice;
    std::string tried;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const ProviderCandidate& c = candidates[i];
        std::string detail;
        const ProbeResult r = probe(c, detail);
        const std::string what = narrow(c.path) + " (" + c.source + ")";
        if (r == ProbeResult::Ok) {
            choice.index = int(i);
            choice.log.push_back("AMD provider: chose " + what + (detail.empty() ? "" : ": " + detail));
            return choice;
        }
        if (r == ProbeResult::Missing) {
            // Expected for most of them; said only for the ones the player or the installer named.
            if (c.path == candidates.front().path && std::string(c.source) == "setting compare_dll")
                choice.log.push_back("AMD provider: " + what + " not found");
            else if (std::string(c.source) == "the FSR 4 DLL the installer found on this PC")
                choice.log.push_back("AMD provider: " + what + " is gone; reinstall WrongGPU to search again");
            continue;
        }
        const std::string line = what + ": " + ProbeResultName(r) + (detail.empty() ? "" : " (" + detail + ")");
        choice.log.push_back("AMD provider: " + line + "; not used");
        tried += (tried.empty() ? "" : "; ") + line;
    }
    choice.warning = "FSR 4 / AMD versions not available: " +
                     (tried.empty() ? std::string("no AMD FidelityFX DLL was found (no installer copy wronggpu-amd-compare.dll, no FSR 4 "
                                                  "DLL on this PC, no amd_fidelityfx_dx12.dll or FidelityFX SDK DLL of the game)")
                                    : "no usable AMD FidelityFX DLL: " + tried) +
                     ". Only WrongGPU is listed; run WrongGPU's setup again to set up AMD's DLL";
    return choice;
}

}  // namespace wgfsr
