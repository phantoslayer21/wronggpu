// SPDX-License-Identifier: MIT
#pragma once
// Which AMD FidelityFX DLL gives a hosted WrongGPU (OptiScaler's FidelityFX upscaler library) AMD's upscaler
// versions: the list the host and the game see after WrongGPU's own entries, and the DLL that runs a context when the
// player selects one of them. The order of the candidates and the choice; loading and checking a candidate is the
// platform's (win_util.cpp, CompareApi).
//
// No Windows code here.
#include <functional>
#include <string>
#include <vector>

namespace wgfsr {

struct ProviderCandidate {
    std::wstring path;
    const char* source = "";  // what it is, for the log ("installer's private copy", "the game's amd_fidelityfx_dx12.dll", ...)
};

struct ProviderInputs {
    std::wstring compareDll;      // setting compare_dll (empty: none)
    std::wstring moduleDir;       // folder of WrongGPU's DLL
    std::wstring exeDir;          // folder of the game's exe
    std::wstring installerFound;  // the FSR 4 DLL the installer found on this PC (compare.txt; empty: none)
    // AMD's FidelityFX SDK 2 DLLs found in the exe folder and its subfolders (amd_fidelityfx_upscaler_dx12.dll and
    // amd_fidelityfx_loader_dx12.dll), in the order the platform found them (shallower first).
    std::vector<std::wstring> sdkDlls;
};

// The candidates, best first, each path once (compared without case, '/' as '\'):
//  1. compare_dll;
//  2. wronggpu-amd-compare.dll next to WrongGPU's DLL: the installer's private copy of the player's AMD DLL (OptiScaler
//     package's FSR 4, the FSR 4 found on this PC, or the game's own amd_fidelityfx_dx12.dll). A host that redirects
//     AMD's file names does not know this one;
//  3. amd_fidelityfx_upscaler_dx12.dll next to WrongGPU's DLL;
//  4. the FSR 4 DLL the installer found on this PC;
//  5. amd_fidelityfx_dx12.wronggpu-original.dll next to WrongGPU's DLL (the direct route's renamed original);
//  6. the game's AMD FidelityFX SDK 2 upscaler DLLs (amd_fidelityfx_upscaler_dx12.dll in the exe folder tree);
//  7. the game's amd_fidelityfx_dx12.dll: the exe folder, then next to WrongGPU's DLL. AMD's signed FidelityFX DLL that
//     AMD's driver upgrades to FSR 4 when the player turns FSR 4 on in AMD Software;
//  8. the game's AMD FidelityFX SDK 2 loader DLLs (amd_fidelityfx_loader_dx12.dll in the exe folder tree).
std::vector<ProviderCandidate> ProviderCandidates(const ProviderInputs& in);

enum class ProbeResult {
    Ok,          // loaded, the FidelityFX API exports, not intercepted
    Missing,     // no such file
    Ours,        // a WrongGPU DLL (our own module, or a copy of it)
    NotLoaded,   // the file exists, LoadLibrary failed
    Redirected,  // loading the file gave another module (a host's LoadLibrary hook hands out its own DLL or WrongGPU)
    Hooked,      // the module's FidelityFX entry points jump into another module (a host's hook)
    NoExports,   // not a FidelityFX API DLL
};
const char* ProbeResultName(ProbeResult r);

struct ProviderChoice {
    int index = -1;              // into the candidates; -1: none usable
    std::vector<std::string> log;  // one line per candidate tried (missing ones only when worth saying)
    std::string warning;         // index == -1: "FSR 4 / AMD versions not available: ..." for the log
};

// Tries the candidates in order with `probe` (which loads one and says what it is; detail: for the log) and takes the
// first that is Ok. narrow: UTF-8 of a path for the log.
ProviderChoice ChooseProvider(const std::vector<ProviderCandidate>& candidates,
                              const std::function<ProbeResult(const ProviderCandidate&, std::string& detail)>& probe,
                              const std::function<std::string(const std::wstring&)>& narrow);

}  // namespace wgfsr
