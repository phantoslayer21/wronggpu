// SPDX-License-Identifier: MIT
// WrongGPU-Restore.exe: the small per-game restorer (build/launcher-design.md section 2). The launcher writes it into
// every game it installs as <install folder>\wronggpu-setup\uninstall.exe. It is wg_core without the payload, the
// converter data or the UI: it reads the install record (v1 of the 0.5.x setup and v2 of the launcher), puts the
// game's original files back byte for byte and collects logs. Options (as the installer's uninstaller):
//   (none)              ask, then uninstall the folder it sits in (Apps & features "Uninstall")
//   /S                  silent; /DIR=<install folder> or /EXE=<game .exe> another folder; /OUT=<zip> collect logs first
//   /COLLECT [/OUT=]    collect logs only ("Modify" in Apps & features)
//   /LOG=<file>         log file (default %LOCALAPPDATA%\WrongGPU\setup.log)
// /NOSCAN and /FORCE are accepted and change nothing. Exit codes: wg_core/constants.h. Written for this project.
// What it shows the player is in plain words, as the launcher says them (SHIP-BAR 4): the game by its name, never its
// folder, its program's file name or the core's own sentences (those go to the log).
#include "wg_core/base.h"

#include <shellapi.h>

#include "wg_core/checks.h"
#include "wg_core/launcher.h"
#include "wg_core/logs.h"
#include "wg_core/record.h"

#ifndef WRONGGPU_VERSION_STR
#define WRONGGPU_VERSION_STR L"0.0.0"
#endif

using namespace wg;

namespace {

bool Has(const wstring& text, const wchar_t* part) { return Lower(text).find(Lower(part)) != wstring::npos; }
// The game as the player knows it ("Cyberpunk 2077"), from its record
wstring GameName(const Record& m, const wstring& dir) {
    if (!m.exe.empty()) return GameDisplayName(m.exe);
    const wstring root = GameRoot(dir);
    return BaseName(root).empty() ? wstring(L"this game") : BaseName(root);
}
// The game has its own entry in Installed apps (whose Modify saves the logs); else the launcher manages it
bool OwnEntry(const Record& m) {
    HKEY k = nullptr;
    const wstring key = UninstallRoot() + (m.regkey.empty() ? RegKeyFor(m.game) : m.regkey);
    if (RegOpenKeyExW(UninstallHive(), key.c_str(), 0, KEY_READ, &k) != ERROR_SUCCESS) return false;
    RegCloseKey(k);
    return true;
}
// A warning of the uninstall (DoRestore's) in plain words, as the launcher words it (app/ui_words.cpp PlainWarning); the
// core's sentence (file names, the backup folder) stays in the log
wstring PlainWarning(const wstring& w) {
    if (Has(w, L"The game replaced ")) return L"The game had replaced one of its files since WrongGPU was installed (probably a game update). Its newer file was kept.";
    if (Has(w, L"exists again with different content") || Has(w, L"The backup folder was kept"))
        return L"A game file the game changed was kept as it is. Its original stays in a backup folder in the game's folder.";
    if (Has(w, L"Left in place (changed since install)")) return L"A WrongGPU file was changed after it was installed, so it was left in place.";
    return L"One file was left as it was.";
}
// What went wrong, in plain words with what to do next (the core's text is in the log)
wstring PlainError(const SetupError& e) {
    if (e.code == EXIT_NOT_INSTALLED) return L"WrongGPU isn't installed in this game any more. Nothing was changed.";
    if (Has(e.msg, L"missing or damaged"))
        return L"A backup WrongGPU made of a game file is missing or damaged. Nothing was changed. Verify the game's files in its store, then uninstall again.";
    if (Has(e.msg, L"being used by another process") || Has(e.msg, L"sharing violation") || Has(e.msg, L"locked a portion"))
        return L"A game file is in use. Close the game and any program that has its files open, then uninstall again.";
    if (Has(e.msg, L"access is denied")) return L"Windows didn't allow a change in the game's folder. Close the game, then uninstall again.";
    if (Has(e.msg, L"not enough space") || Has(e.msg, L"disk is full")) return L"The drive is full. Free some space, then uninstall again.";
    return L"Something went wrong. Try again; if it keeps happening, save the logs with Modify in Installed apps and report it.";
}

}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32);
    static const EmbeddedData data{nullptr, 0, 0, L"", 0, L"", 0, WRONGGPU_VERSION_STR};
    Context ctx;
    ctx.data = &data;
    bool silent = false, collect = false, uninstall = false, bad = false;
    wstring dirArg, exeArg, outZip;
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; i < argc; ++i) {
        const wstring a = argv[i], la = Lower(a);
        if (la == L"/s" || la == L"/silent") silent = true;
        else if (la == L"/uninstall") uninstall = true;
        else if (la == L"/collect") collect = true;
        else if (la == L"/noscan" || la == L"/force") {}
        else if (la.rfind(L"/dir=", 0) == 0) dirArg = a.substr(5);
        else if (la.rfind(L"/d=", 0) == 0) dirArg = a.substr(3);
        else if (la.rfind(L"/exe=", 0) == 0) exeArg = a.substr(5);
        else if (la.rfind(L"/out=", 0) == 0) outZip = a.substr(5);
        else if (la.rfind(L"/log=", 0) == 0) ctx.logFile = a.substr(5);
        else bad = true;
    }
    if (argv) LocalFree(argv);
    if (ctx.logFile.empty()) ctx.logFile = DefaultLogPath();
    if (bad && silent) { LogLine(ctx, L"unknown command-line option"); return EXIT_USAGE; }
    const wstring self = SelfPath();
    const bool fromState = IEq(BaseName(Parent(self)), kStateDirName);
    if (!collect) uninstall = true;
    wstring dir;
    if (!exeArg.empty()) {
        dir = Parent(FullPath(exeArg));
        if (!Exists(RecordPath(dir))) { const wstring d = FindInstallDir(FullPath(exeArg)); if (Exists(RecordPath(d))) dir = d; }
    } else if (!dirArg.empty()) dir = FullPath(dirArg);
    else if (fromState) dir = Parent(Parent(self));
    SystemInfo sys;
    Ui ui;
    ui.step = [](int, int, const wstring&) {};
    ui.log = [&ctx](const wstring& s) { LogLine(ctx, s); };
    LogLine(ctx, wstring(L"WrongGPU restorer ") + data.version + (uninstall ? L" - uninstall " : L" - collect logs ") + dir);
    wstring name;  // the game's name once its record is read: the failure's words name it
    try {
        // started by hand from somewhere else, with no folder: said in words, with where to uninstall from (no switches)
        if (dir.empty())
            Fail(L"WrongGPU can't tell which game this uninstaller belongs to. Uninstall WrongGPU from the game's entry in Installed apps or from the launcher.", EXIT_USAGE);
        if (collect && !uninstall) {
            sys.win = WindowsInfo(); sys.gpu = DetectGpu();
            Bundle b = CollectLogs(ctx, dir, outZip.empty() ? DefaultBundlePath(dir) : outZip, sys);
            for (auto& f : b.files) LogLine(ctx, L"  bundled " + f);
            if (!silent) {
                MessageBoxW(nullptr, BundleMessage(b).c_str(), L"WrongGPU", MB_ICONINFORMATION);
                const wstring args = L"/select,\"" + b.zip + L"\"";
                ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
            }
            return EXIT_OK;
        }
        Record m;
        if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + dir + L".", EXIT_NOT_INSTALLED);
        name = GameName(m, dir);
        // the game moved or was uninstalled (its program isn't there any more): what is removed is WrongGPU's files in its old
        // folder, as the launcher's Remove files says it; there is no game to get its files back
        const bool gone = !m.exe.empty() && !IsFile(m.exe);
        if (GameRunning(m.exe)) {
            LogLine(ctx, BaseName(m.exe) + L" is running: not uninstalled");
            if (!silent) MessageBoxW(nullptr, (name + L" is running. Close it, then uninstall again.").c_str(), L"WrongGPU", MB_ICONWARNING);
            return EXIT_REFUSED;
        }
        if (!silent) {
            // where the logs are saved from first: this game's own entry (Modify), or the launcher that manages it
            const wstring logs = OwnEntry(m) ? L" Reporting a problem? Choose No, then use Modify in Installed apps to save the logs first."
                                 : LauncherInstalled() ? L" Reporting a problem? Choose No, then collect the logs from the game's page in WrongGPU first."
                                                       : L"";
            const wstring q = gone ? L"Remove WrongGPU's files from " + name + L"'s old folder?\n\n" + name + L" isn't in its folder any more. WrongGPU removes the files it "
                                     L"left there."
                                   : L"Uninstall WrongGPU from " + name + L"?\n\nThe game's original files are put back exactly as they were." + logs;
            if (MessageBoxW(nullptr, q.c_str(), L"WrongGPU", MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) != IDYES) return EXIT_REFUSED;
        }
        if (!outZip.empty()) {
            sys.win = WindowsInfo(); sys.gpu = DetectGpu();
            CollectLogs(ctx, dir, outZip, sys);
            LogLine(ctx, L"logs collected into " + outZip + L" before uninstalling");
        }
        const std::vector<wstring> warnings = DoRestore(ctx, dir, ui, false, false);
        for (auto& w : warnings) LogLine(ctx, L"warning: " + w);
        LogLine(ctx, L"uninstall complete");
        if (!silent) {
            wstring t = gone ? L"WrongGPU's files are removed from " + name + L"'s old folder."
                             : L"WrongGPU is uninstalled from " + name + L", and the game has its original files back.";
            std::vector<wstring> said;  // each in words once
            for (auto& w : warnings) {
                const wstring p = PlainWarning(w);
                if (std::find(said.begin(), said.end(), p) == said.end()) { said.push_back(p); t += L"\n\n" + p; }
            }
            MessageBoxW(nullptr, t.c_str(), L"WrongGPU", MB_ICONINFORMATION);
        }
        return EXIT_OK;
    } catch (const SetupError& e) {
        LogLine(ctx, L"ERROR: " + e.msg);
        if (!silent) {
            const wstring t = e.code == EXIT_USAGE ? e.msg : name.empty() ? PlainError(e) : L"WrongGPU couldn't be uninstalled from " + name + L".\n\n" + PlainError(e);
            MessageBoxW(nullptr, t.c_str(), L"WrongGPU", MB_ICONWARNING);
        }
        return e.code;
    }
}
