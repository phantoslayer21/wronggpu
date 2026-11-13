// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the per-game rules, REFramework's detection and its mirror (gamerules.h). Written for this
// project; REFramework is recognised by the words its own files carry, never read further.
#include "gamerules.h"

#include "fingerprint.h"
#include "pe.h"

namespace wg {

// RE Requiem (and its demo): its integrity check stops any OptiScaler without REFramework; its RR needs the host's
// command-list restore (the RR plugin reports RR unsupported there without it); its RR comes with path tracing only; its
// driver shader cache is keyed by its own AGS application name (release-060 FINAL-PROGRESS.txt P2-P4: the game reads
// another cache than the launcher's helper fills), so its RR kernels are prepared at its first start.
static const GameRule kGameRules[] = {
    {L"re9.exe", L"RE Requiem", true, true, true, true},
    {L"re9demo.exe", L"RE Requiem", true, true, true, true},
    // rc5: Marvel's Spider-Man 2: its Ray Reconstruction isn't stable (2026-11-04), so only the experimental opt-in
    // sm2_rr_experimental (optiscaler.h kOptiGameKeys) turns it on
    {L"Spider-Man2.exe", L"Spider-Man 2", false, false, false, false, "sm2_rr_experimental",
     L"Ray Reconstruction isn't stable in Spider-Man 2 yet, so it's off. Model M works. (Experimental opt-in: see the release notes.)", nullptr},
    // rc5: LEGO Batman: Legacy of the Dark Knight: its Upscaling list has no DLSS (no RR created in the attended run,
    // 2026-11-04); its Unreal plugin layout made DetectRr count it as a direct-RR game
    {L"LEGOBatmanLotDK-Win64-Shipping.exe", L"LEGO Batman", false, false, false, false, nullptr, nullptr,
     L"LEGO Batman's menu has no Ray Reconstruction option (it would need a separate game mod), so WrongGPU's Ray Reconstruction isn't "
     L"offered here. Model M works."},
};

const GameRule* GameRuleFor(const wstring& exe) {
    const wstring n = BaseName(exe);
    for (auto& r : kGameRules) if (IEq(n, r.exe)) return &r;
    return nullptr;
}
bool IsLauncherStub(const wstring& exe) {
    const wstring n = BaseName(exe);
    for (auto s : kLauncherStubs) if (IEq(n, s)) return true;
    return false;
}

bool IsReFrameworkDll(const wstring& path) {
    if (!IsFile(path)) return false;
    MappedFile f(path, 256ull << 20);
    if (!f.data) return false;
    const PeVersion v = ReadPeVersion(f.data, f.size);
    for (auto k : {L"ProductName", L"FileDescription", L"InternalName", L"OriginalFilename"})
        if (Lower(v.Get(k)).find(L"reframework") != wstring::npos) return true;
    static const char* const needle[] = {"reframework"};
    return ScanMarkers(f.data, f.size, needle, 1) != 0;
}

ReFrameworkState DetectReFramework(const wstring& exeDir) {
    ReFrameworkState s;
    const wstring dll = Join(exeDir, kReFrameworkDll);
    if (!IsReFrameworkDll(dll)) return s;
    s.found = true;
    s.evidence = kReFrameworkDll;
    if (IsDir(Join(exeDir, L"reframework"))) s.evidence += L", folder reframework";
    return s;
}

wstring NeedsReFrameworkText(const GameRule& rule) {
    return wstring(rule.name) + L" needs REFramework installed first. The game blocks OptiScaler without it. WrongGPU doesn't install "
           L"REFramework: install it yourself (" + kReFrameworkPage + L"), then install WrongGPU.";
}

}  // namespace wg
