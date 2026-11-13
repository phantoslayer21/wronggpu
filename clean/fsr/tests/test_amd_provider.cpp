// SPDX-License-Identifier: MIT
// Unit test of the AMD provider chain (amd_provider.h): which AMD FidelityFX DLL gives a hosted WrongGPU AMD's
// upscaler versions. No Windows code, no DLLs: the probe is simulated.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "../src/amd_provider.h"

using namespace wgfsr;

namespace {

int failures = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++failures;                                                \
        }                                                              \
    } while (0)

std::string Narrow(const std::wstring& w) { return std::string(w.begin(), w.end()); }

// A simulated PC: path -> what probing it gives (absent: Missing). Records the order of the probes.
struct Pc {
    std::map<std::wstring, ProbeResult> files;
    std::vector<std::wstring> probed;
    ProviderChoice Choose(const std::vector<ProviderCandidate>& c) {
        return ChooseProvider(c, [this](const ProviderCandidate& k, std::string& detail) {
            probed.push_back(k.path);
            auto it = files.find(k.path);
            if (it == files.end()) return ProbeResult::Missing;
            if (it->second == ProbeResult::Redirected) detail = "loading it gave C:\\Game\\dxgi.dll";
            return it->second;
        }, &Narrow);
    }
};

const std::wstring kGame = L"C:\\Game\\bin\\x64";

ProviderInputs Hosted() {  // WrongGPU next to the exe (the installer's OptiScaler route)
    ProviderInputs in;
    in.moduleDir = kGame;
    in.exeDir = kGame;
    return in;
}

bool Contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

void TestOrder() {
    ProviderInputs in;
    in.compareDll = L"K:\\fsr4\\amd_fidelityfx_upscaler_dx12.dll";
    in.moduleDir = L"C:\\Game\\mods";
    in.exeDir = kGame;
    in.installerFound = L"H:\\Other\\amd_fidelityfx_upscaler_dx12.dll";
    in.sdkDlls = {kGame + L"\\fsr\\amd_fidelityfx_loader_dx12.dll", kGame + L"\\fsr\\amd_fidelityfx_upscaler_dx12.dll"};
    const auto c = ProviderCandidates(in);
    const std::vector<std::wstring> want{
        in.compareDll,
        L"C:\\Game\\mods\\wronggpu-amd-compare.dll",
        L"C:\\Game\\mods\\amd_fidelityfx_upscaler_dx12.dll",
        in.installerFound,
        L"C:\\Game\\mods\\amd_fidelityfx_dx12.wronggpu-original.dll",
        kGame + L"\\fsr\\amd_fidelityfx_upscaler_dx12.dll",
        kGame + L"\\amd_fidelityfx_dx12.dll",
        L"C:\\Game\\mods\\amd_fidelityfx_dx12.dll",
        kGame + L"\\fsr\\amd_fidelityfx_loader_dx12.dll",
    };
    CHECK(c.size() == want.size());
    for (size_t i = 0; i < c.size() && i < want.size(); ++i) CHECK(c[i].path == want[i]);
}

void TestEachPathOnce() {
    ProviderInputs in = Hosted();
    in.installerFound = L"c:/game/BIN/x64/amd_fidelityfx_upscaler_dx12.dll";  // the same file as candidate 3
    in.sdkDlls = {kGame + L"\\amd_fidelityfx_upscaler_dx12.dll"};             // and again from the folder search
    const auto c = ProviderCandidates(in);
    // compare copy, upscaler next to us, renamed original, game's amd_fidelityfx_dx12.dll (exe folder == module folder)
    CHECK(c.size() == 4);
    CHECK(c.size() == 4 && c[3].path == kGame + L"\\amd_fidelityfx_dx12.dll");
    CHECK(Contains(c[1].source, "next to WrongGPU"));
}

// The player's case: no installer copy, no FSR 4 DLL anywhere; only the game's own amd_fidelityfx_dx12.dll.
void TestOnlyGameDll() {
    Pc pc;
    pc.files[kGame + L"\\amd_fidelityfx_dx12.dll"] = ProbeResult::Ok;
    const auto c = ProviderCandidates(Hosted());
    const ProviderChoice choice = pc.Choose(c);
    CHECK(choice.index >= 0 && c[size_t(choice.index)].path == kGame + L"\\amd_fidelityfx_dx12.dll");
    CHECK(choice.warning.empty());
    CHECK(!choice.log.empty() && Contains(choice.log.back(), "AMD provider: chose") &&
          Contains(choice.log.back(), "the game's amd_fidelityfx_dx12.dll"));
    CHECK(pc.probed.size() == 4);  // everything before it was tried first
}

void TestInstallerCopyFirst() {
    Pc pc;
    pc.files[kGame + L"\\wronggpu-amd-compare.dll"] = ProbeResult::Ok;
    pc.files[kGame + L"\\amd_fidelityfx_upscaler_dx12.dll"] = ProbeResult::Ok;
    pc.files[kGame + L"\\amd_fidelityfx_dx12.dll"] = ProbeResult::Ok;
    const auto c = ProviderCandidates(Hosted());
    const ProviderChoice choice = pc.Choose(c);
    CHECK(choice.index == 0 && c[0].path == kGame + L"\\wronggpu-amd-compare.dll");
    CHECK(pc.probed.size() == 1);
}

// OptiScaler redirects amd_fidelityfx_upscaler_dx12.dll to WrongGPU or itself, and hooks the game's FidelityFX DLL:
// both are passed over, the SDK upscaler found in a subfolder is used.
void TestHostInterceptionSkipped() {
    Pc pc;
    ProviderInputs in = Hosted();
    in.installerFound = L"H:\\Other\\amd_fidelityfx_upscaler_dx12.dll";
    in.sdkDlls = {kGame + L"\\sdk\\amd_fidelityfx_upscaler_dx12.dll"};
    pc.files[kGame + L"\\amd_fidelityfx_upscaler_dx12.dll"] = ProbeResult::Ours;
    pc.files[in.installerFound] = ProbeResult::Redirected;
    pc.files[kGame + L"\\sdk\\amd_fidelityfx_upscaler_dx12.dll"] = ProbeResult::Ok;
    pc.files[kGame + L"\\amd_fidelityfx_dx12.dll"] = ProbeResult::Hooked;
    const auto c = ProviderCandidates(in);
    const ProviderChoice choice = pc.Choose(c);
    CHECK(choice.index >= 0 && c[size_t(choice.index)].path == kGame + L"\\sdk\\amd_fidelityfx_upscaler_dx12.dll");
    bool redirectLogged = false;
    for (const std::string& l : choice.log) redirectLogged = redirectLogged || (Contains(l, "redirected") && Contains(l, "dxgi.dll") && Contains(l, "not used"));
    CHECK(redirectLogged);
    // Hooked too: nothing usable is left, and the warning names every reason.
    pc.files[kGame + L"\\sdk\\amd_fidelityfx_upscaler_dx12.dll"] = ProbeResult::NoExports;
    const ProviderChoice none = pc.Choose(c);
    CHECK(none.index == -1);
    CHECK(Contains(none.warning, "FSR 4 / AMD versions not available: ") && Contains(none.warning, "hooked") &&
          Contains(none.warning, "redirected") && Contains(none.warning, "a WrongGPU DLL") && Contains(none.warning, "no FidelityFX API exports"));
}

void TestNothingThere() {
    Pc pc;
    ProviderInputs in = Hosted();
    in.compareDll = L"K:\\gone.dll";
    in.installerFound = L"H:\\Other\\amd_fidelityfx_upscaler_dx12.dll";
    const ProviderChoice choice = pc.Choose(ProviderCandidates(in));
    CHECK(choice.index == -1);
    CHECK(choice.warning.rfind("FSR 4 / AMD versions not available: no AMD FidelityFX DLL was found", 0) == 0);
    bool settingSaid = false, recordSaid = false;
    for (const std::string& l : choice.log) {
        settingSaid = settingSaid || (Contains(l, "gone.dll") && Contains(l, "not found"));
        recordSaid = recordSaid || (Contains(l, "H:\\Other") && Contains(l, "reinstall"));
    }
    CHECK(settingSaid && recordSaid);
}

void TestLoaderLast() {
    Pc pc;
    ProviderInputs in = Hosted();
    in.sdkDlls = {kGame + L"\\amd_fidelityfx_loader_dx12.dll"};
    pc.files[kGame + L"\\amd_fidelityfx_loader_dx12.dll"] = ProbeResult::Ok;
    pc.files[kGame + L"\\amd_fidelityfx_dx12.dll"] = ProbeResult::Ok;
    auto c = ProviderCandidates(in);
    ProviderChoice choice = pc.Choose(c);
    CHECK(choice.index >= 0 && c[size_t(choice.index)].path == kGame + L"\\amd_fidelityfx_dx12.dll");
    pc.files.erase(kGame + L"\\amd_fidelityfx_dx12.dll");
    choice = pc.Choose(c);
    CHECK(choice.index >= 0 && c[size_t(choice.index)].path == kGame + L"\\amd_fidelityfx_loader_dx12.dll");
}

}  // namespace

int main() {
    TestOrder();
    TestEachPathOnce();
    TestOnlyGameDll();
    TestInstallerCopyFirst();
    TestHostInterceptionSkipped();
    TestNothingThere();
    TestLoaderLast();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::puts("PASS: AMD provider chain (order, each path once, the game's own DLL alone, host interception passed over, warning)");
    return 0;
}
