// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the player's OptiScaler: its package, the OptiScaler.ini keys for WrongGPU, the proxy name,
// WrongGPU's OptiScaler plugin (WrongGPU.asi). Written for this project.
#include "optiscaler.h"

#include "coexist.h"
#include "optiedition.h"
#include "pe.h"

namespace wg {

wstring OptiTempDir(Context& ctx) {
    if (!ctx.opti.tempDir.empty()) return ctx.opti.tempDir;
    wchar_t tmp[MAX_PATH] = {}; GetTempPathW(MAX_PATH, tmp);
    ctx.opti.tempDir = Join(tmp, L"wronggpu-optiscaler-" + std::to_wstring(GetCurrentProcessId()));
    EnsureDir(ctx.opti.tempDir);
    return ctx.opti.tempDir;
}
// Runs a program without a window and waits (exit code, or -1).
int RunHiddenWait(const wstring& exe, const wstring& args, DWORD timeoutMs) {
    wstring cmd = L"\"" + exe + L"\" " + args;
    std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si); si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return -1;
    DWORD code = (DWORD)-1;
    if (WaitForSingleObject(pi.hProcess, timeoutMs) == WAIT_OBJECT_0) GetExitCodeProcess(pi.hProcess, &code);
    else TerminateProcess(pi.hProcess, 1);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)code;
}
// The player's OptiScaler: a release archive (.7z / .zip, unpacked with Windows' tar.exe), its unpacked folder, or the
// OptiScaler.dll in that folder.
OptiPackage PrepareOptiScaler(Context& ctx, const wstring& chosen) {
    if (!ctx.opti.tempDir.empty()) { RemoveTree(ctx.opti.tempDir); }
    OptiPackage p;
    p.source = FullPath(chosen);
    wstring root;
    if (IsDir(p.source)) root = p.source;
    else if (IsFile(p.source) && IEq(BaseName(p.source), L"OptiScaler.dll")) root = Parent(p.source);
    else if (IsFile(p.source) && (EndsWithI(p.source, L".7z") || EndsWithI(p.source, L".zip"))) {
        wchar_t tmp[MAX_PATH] = {}; GetTempPathW(MAX_PATH, tmp);
        p.tempDir = Join(tmp, L"wronggpu-optiscaler-" + std::to_wstring(GetCurrentProcessId()));
        RemoveTree(p.tempDir); EnsureDir(Join(p.tempDir, L"package"));
        wchar_t sys[MAX_PATH] = {}; GetSystemDirectoryW(sys, MAX_PATH);
        const wstring tar = Join(sys, L"tar.exe");
        if (!IsFile(tar)) { p.error = L"Windows' tar.exe is missing - unpack the OptiScaler archive yourself and choose its folder"; return p; }
        const int code = RunHiddenWait(tar, L"-xf \"" + p.source + L"\" -C \"" + Join(p.tempDir, L"package") + L"\"", 300000);
        if (code != 0) { p.error = L"Could not unpack " + BaseName(p.source) + L" (tar exit code " + std::to_wstring(code) + L") - unpack it yourself and choose its folder"; return p; }
        root = Join(p.tempDir, L"package");
    } else {
        p.error = L"Choose OptiScaler's release archive (.7z or .zip), the folder you unpacked it to, or the OptiScaler.dll in it";
        return p;
    }
    if (!IsFile(Join(root, L"OptiScaler.dll")))
        for (auto& d : ListDirs(root)) if (IsFile(Join(d, L"OptiScaler.dll"))) { root = d; break; }
    if (!IsFile(Join(root, L"OptiScaler.dll")) || !IsFile(Join(root, L"OptiScaler.ini"))) {
        p.error = L"No OptiScaler.dll and OptiScaler.ini in " + BaseName(p.source) + L" - is this OptiScaler's release?";
        return p;
    }
    string ini;
    ReadAll(Join(root, L"OptiScaler.ini"), ini);
    if (ini.find("FfxDx12SRPath") == string::npos) {
        p.error = L"This OptiScaler is too old for WrongGPU (no FfxDx12SRPath setting) - use " + wstring(kOptiTested) + L" or newer";
        return p;
    }
    p.dir = root;
    p.version = FileVersion(Join(root, L"OptiScaler.dll"));
    p.edition = IsEditionDll(Join(root, L"OptiScaler.dll"));
    std::vector<wstring> stack{L""};
    while (!stack.empty()) {
        const wstring rel = stack.back(); stack.pop_back();
        const wstring abs = rel.empty() ? root : Join(root, rel);
        for (auto& f : ListFiles(abs, L"*")) {
            const wstring n = BaseName(f);
            if (rel.empty() && (IEq(n, L"setup_windows.bat") || IEq(n, L"setup_linux.sh") || n.rfind(L"!!", 0) == 0)) continue;
            p.files.push_back(rel.empty() ? n : Join(rel, n));
        }
        for (auto& d : ListDirs(abs)) stack.push_back(rel.empty() ? BaseName(d) : Join(rel, BaseName(d)));
    }
    std::sort(p.files.begin(), p.files.end());
    LogLine(ctx, L"OptiScaler " + p.version + (p.edition ? L" (WrongGPU edition, plugin interface 5)" : L"") + L" from " + p.source + L": " + std::to_wstring(p.files.size()) + L" files");
    return p;
}
// OptiScaler's proxy name next to the game exe: the proxy picker's choice (coexist.h): where the player's OptiScaler is
// (a switch), else the first name of kProxyOrder that no recognised tool, unknown file or Windows DLL of the game holds.
wstring OptiProxyFor(const wstring& dir) {
    wstring players;
    for (auto n : kProxyOrder)
        if (IsFile(Join(dir, n)) && IdentifyFile(Join(dir, n)).tool == Tool::OptiScaler) { players = n; break; }
    return PickProxy(dir, nullptr, players, IsFile(Join(dir, L"neuralhub.ini")), false).name;
}

// The install routes (setup_route.h of the installer): without the direct FSR route (CMake WRONGGPU_DIRECT_ROUTE=OFF)
// nothing is installed without the player's OptiScaler.
#if WRONGGPU_DIRECT_ROUTE
void RequireOptiScaler(Context&) {}
wstring OptiProblem(Context& ctx) { return L"OptiScaler: " + ctx.opti.error; }
#else
// Why there is no usable OptiScaler package: none was given, or the one given was rejected.
wstring OptiProblem(Context& ctx) { return ctx.opti.chosen() ? L"OptiScaler: " + ctx.opti.error : wstring(wg_wizard::OptiRequired()); }
// A silent install, and the install engine itself: nothing is installed without the player's OptiScaler.
void RequireOptiScaler(Context& ctx) { if (!ctx.opti.ok()) Fail(OptiProblem(ctx), EXIT_REFUSED); }
#endif

}  // namespace wg
