// SPDX-License-Identifier: MIT
// Unit tests of wg_core (the installer's / launcher's engine) that need no game, no payload and no GPU: strings and
// paths, the install record, wronggpu.ini keys, the runtime options of a Model-M-only build, the Context's log file.
// Written for this project. Run by ctest ("wg-core") in a WRONGGPU_TESTING build.
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "wg_core/checks.h"
#include "wg_core/context.h"
#include "wg_core/gamerules.h"
#include "wg_core/model.h"
#include "wg_core/record.h"
#include "wg_core/settings.h"

using namespace wg;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}
wstring TempFolder() {
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    wstring dir = Join(tmp, L"wg-core-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return dir;
}
}  // namespace

int main() {
    try {
        // strings and paths
        require(Join(L"C:\\a", L"b") == L"C:\\a\\b" && Join(L"C:\\a\\", L"b") == L"C:\\a\\b" && Join(L"", L"b") == L"b", "Join");
        require(Parent(L"C:\\a\\b.exe") == L"C:\\a" && BaseName(L"C:\\a\\b.exe") == L"b.exe" && Stem(L"C:\\a\\b.exe") == L"b", "Parent/BaseName/Stem");
        require(IEq(L"ABC", L"abc") && EndsWithI(L"x.EXE", L".exe") && StartsWithI(L"WrongGPU-1", L"wronggpu-"), "case-insensitive compare");
        require(FromUtf8(ToUtf8(L"Cr\u00e9ation")) == L"Cr\u00e9ation", "UTF-8 round trip");
        const char ascii[] = "xx OptiScaler yy";
        const char utf16[] = "o\0p\0t\0i\0s\0c\0a\0l\0e\0r\0";
        require(ContainsText((const uint8_t*)ascii, sizeof ascii - 1, "optiscaler"), "ContainsText ASCII");
        require(ContainsText((const uint8_t*)utf16, sizeof utf16 - 1, "OPTISCALER"), "ContainsText UTF-16LE");
        require(!ContainsText((const uint8_t*)ascii, sizeof ascii - 1, "reshade"), "ContainsText miss");
        require(Sha256Bytes("abc", 3) == L"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 (FIPS 180-4 test vector)");
        require(VersionAtMost(L"0.3.0", L"0.3.0") && VersionAtMost(L"0.2.9", L"0.3.0") && !VersionAtMost(L"0.10.0", L"0.3.0") && VersionAtMost(L"", L"0.3.0"),
                "VersionAtMost");
        // rc6: the running-game watch after Play (WatchPlayed): a game a store's launcher starts later (Cyberpunk 2077 through
        // Steam and REDlauncher: 28 s after Play) is starting until it is seen running, never closed before it ran
        require(WatchPlayed(5000, false, false) == PlayWatch::Starting && WatchPlayed(28000, false, false) == PlayWatch::Starting &&
                    WatchPlayed(179000, false, false) == PlayWatch::Starting, "WatchPlayed: not seen running yet: starting (up to 3 min)");
        require(WatchPlayed(28000, false, true) == PlayWatch::Running && WatchPlayed(400000, true, true) == PlayWatch::Running,
                "WatchPlayed: running");
        require(WatchPlayed(60000, true, false) == PlayWatch::Closed && WatchPlayed(181000, false, false) == PlayWatch::Closed,
                "WatchPlayed: closed once it ran, or never started within 3 min");
        require(WatchPlayed(5000, true, false) == PlayWatch::Starting, "WatchPlayed: the first 15 s a store may start it again: starting");
        // VersionLess (the launcher's start, Update due, the update check, ROCm folders): a pre-release before its release
        require(VersionLess(L"0.6.1-rc1", L"0.6.1") && !VersionLess(L"0.6.1", L"0.6.1-rc1"), "VersionLess: 0.6.1-rc1 < 0.6.1");
        require(VersionLess(L"0.6.0", L"0.6.1-rc1") && !VersionLess(L"0.6.1-rc1", L"0.6.0"), "VersionLess: 0.6.0 < 0.6.1-rc1");
        require(VersionLess(L"0.6.1-rc2", L"0.6.1-rc10") && VersionLess(L"0.6.1-beta2", L"0.6.1-rc1") && !VersionLess(L"0.6.1-rc1", L"0.6.1-rc1"),
                "VersionLess: two pre-releases by their words, then their numbers");
        require(!VersionLess(L"v0.6.1", L"0.6.1") && !VersionLess(L"0.6.1", L"v0.6.1") && VersionLess(L"v0.6.0", L"0.6.1"), "VersionLess: a leading v doesn't count");
        require(!VersionLess(L"0.6.1+abc", L"0.6.1") && !VersionLess(L"0.6.1", L"0.6.1+abc"), "VersionLess: a build suffix is no pre-release");
        require(VersionLess(L"6.4.1", L"7.2") && VersionLess(L"0.9.0", L"0.10.0") && !VersionLess(L"7.2", L"7.2") && VersionLess(L"", L"0.6.0"),
                "VersionLess: plain versions and folder names as before");

        // wronggpu.ini keys: replaced in place, added to the section, a new section at the end; read as the front end does
        string ini = "; comment\r\n[fsr]\r\nupscaler = amd\r\n\r\n[other]\r\nx=1\r\n";
        ini = IniSet(ini, "fsr", "upscaler", "network");
        ini = IniSet(ini, "fsr", "runtime_dll", "wronggpu_runtime.dll");
        ini = IniSet(ini, "new", "k", "v");
        string value, section;
        require(IniGetFrontEnd(ini, "upscaler", &value, &section) && value == "network" && section == "fsr", "IniSet replaces, IniGetFrontEnd reads");
        require(IniGetFrontEnd(ini, "runtime_dll", &value, &section) && value == "wronggpu_runtime.dll", "IniSet adds to the section");
        require(ini.find("; comment\r\n") == 0 && ini.find("[other]\r\nx=1\r\n") != string::npos && ini.find("[new]\r\nk=v") != string::npos, "comments and other sections kept");
        require(!IniGetFrontEnd("[other]\r\nupscaler=amd\r\n", "upscaler", &value, &section), "keys of other sections are not the front end's");
        require(IniGetFrontEnd("upscaler = amd ; note\r\nupscaler = network # last wins\r\n", "upscaler", &value, &section) && value == "network",
                "a comment after a space ends the value; the last one wins");
#if !WRONGGPU_PRESET_K
        require(OnlyModelM("timing_log=1;model=K") == "timing_log=1;model=M" && OnlyModelM("") == "model=M", "OnlyModelM");
#endif

        // the install record: written, read back, paths below the folder only
        const wstring dir = TempFolder();
        // The root itself can be a junction. Never enumerate its target; ordinary folders and child links still work.
        {
            const wstring target = Join(dir, L"junction-target"), root = Join(dir, L"junction-root");
            EnsureDir(target);
            const wstring canary = Join(target, L"canary.txt");
            WriteFileAtomic(canary, "safe", 4);
            const wstring mkroot = L"cmd.exe /d /c mklink /J \"" + root + L"\" \"" + target + L"\" >nul";
            require(_wsystem(mkroot.c_str()) == 0, "create root junction");
            RemoveTree(root);
            require(!Exists(root) && IsFile(canary), "RemoveTree root junction: link gone, target canary kept");
            const wstring normal = Join(dir, L"normal-tree"), child = Join(normal, L"child-link");
            EnsureDir(normal);
            WriteFileAtomic(Join(normal, L"own.txt"), "x", 1);
            const wstring mkchild = L"cmd.exe /d /c mklink /J \"" + child + L"\" \"" + target + L"\" >nul";
            require(_wsystem(mkchild.c_str()) == 0, "create child junction");
            RemoveTree(normal);
            require(!Exists(normal) && IsFile(canary), "RemoveTree normal tree and child junction: target canary kept");
        }
        Record m;
        m.version = L"0.5.5"; m.game = dir; m.exe = Join(dir, L"Game.exe"); m.status = L"experimental"; m.regkey = RegKeyFor(dir); m.model = L"C:\\m\\model.wgcm";
        Entry e; e.action = L"copy"; e.rel = L"plugins\\WrongGPU.asi"; e.inst = L"ab"; m.e.push_back(e);
        e = Entry(); e.action = L"disable"; e.rel = L"dxgi.dll"; e.existed = true; e.orig = L"cd"; m.e.push_back(e);
        SaveRecord(RecordPath(dir), m);
        Record r;
        require(LoadRecord(RecordPath(dir), r) && !r.legacy && r.version == L"0.5.5" && r.game == dir && r.e.size() == 2, "record round trip");
        require(r.e[0].action == L"copy" && !r.e[0].existed && r.e[0].inst == L"ab" && r.e[1].existed && r.e[1].orig == L"cd", "record entries");
        require(RegKeyFor(dir) == m.regkey && StartsWithI(m.regkey, L"WrongGPU-") && m.regkey.size() == 17, "RegKeyFor is stable");
        require(!HostedRecord(r), "no OptiScaler.ini entry: not the hosted route");
        require(Target(dir, L"a\\b.dll") == Join(dir, L"a\\b.dll"), "Target below the folder");
        for (const wchar_t* bad : {L"..\\x.dll", L"a\\..\\..\\x.dll", L"C:\\x.dll", L"\\x.dll", L""}) {
            bool refused = false;
            try { Target(dir, bad); } catch (const SetupError&) { refused = true; }
            require(refused, "Target refuses a path outside the folder");
        }

        // anti-cheat: the game's top folder and the markers it owns (a sibling game's or a library's markers do not count)
        {
            const wstring lib = Join(dir, L"OpenNR Library"), steam = Join(dir, L"SteamLib");
            auto file = [](const wstring& p) { EnsureDir(Parent(p)); WriteFileAtomic(p, "x", 1); };
            for (const wchar_t* d : {L"W3\\bin\\x64_dx12", L"UE\\Proj\\Binaries\\Win64", L"UE\\Engine", L"UE2\\Binaries\\Win64", L"RE", L"GTA V\\BattlEye", L"BattlEye"})
                EnsureDir(Join(lib, d));
            EnsureDir(Join(steam, L"steamapps\\common\\G\\bin\\x64"));
            EnsureDir(Join(steam, L"steamapps\\common\\EasyAntiCheat"));
            file(Join(lib, L"GTA V\\BEService_x64.exe"));
            require(GameTopFolder(Join(lib, L"W3\\bin\\x64_dx12")) == Join(lib, L"W3"), "top folder: bin\\x64_dx12 -> the game folder");
            require(GameTopFolder(Join(lib, L"UE\\Proj\\Binaries\\Win64")) == Join(lib, L"UE"), "top folder: Unreal root with Engine");
            require(GameTopFolder(Join(lib, L"UE2\\Binaries\\Win64")) == Join(lib, L"UE2"), "top folder: Binaries\\Win64 without a project folder");
            require(GameTopFolder(Join(lib, L"RE")) == Join(lib, L"RE"), "top folder: exe in the game folder");
            require(GameTopFolder(Join(steam, L"steamapps\\common\\G\\bin\\x64")) == Join(steam, L"steamapps\\common\\G"), "top folder: Steam library");
            require(LibraryContainer(steam) && LibraryContainer(Join(steam, L"steamapps\\common")) && LibraryContainer(L"K:") && LibraryContainer(L"K:\\") &&
                        LibraryContainer(L"K:\\Games") && LibraryContainer(L"C:\\Program Files (x86)") && !LibraryContainer(Join(lib, L"W3")),
                    "library containers");
            for (const wchar_t* exe : {L"W3\\bin\\x64_dx12\\witcher3.exe", L"UE2\\Binaries\\Win64\\UE2-Win64-Shipping.exe", L"RE\\re9.exe", L"UE\\Proj\\Binaries\\Win64\\P.exe"})
                require(FindAntiCheat(Join(lib, exe)).empty(), "a sibling game's BattlEye and the library's do not count");
            require(FindAntiCheat(Join(steam, L"steamapps\\common\\G\\bin\\x64\\G.exe")).empty(), "EasyAntiCheat in steamapps\\common does not count");
            EnsureDir(Join(lib, L"W3\\EasyAntiCheat"));
            require(FindAntiCheat(Join(lib, L"W3\\bin\\x64_dx12\\witcher3.exe")) == Join(lib, L"W3\\EasyAntiCheat"), "the game's own marker: its full path");
            EnsureDir(Join(lib, L"RE\\tools\\BattlEye"));
            require(FindAntiCheat(Join(lib, L"RE\\re9.exe")) == Join(lib, L"RE\\tools\\BattlEye"), "two levels below the game folder");
            file(Join(lib, L"UE\\start_protected_game.exe"));
            require(FindAntiCheat(Join(lib, L"UE\\Proj\\Binaries\\Win64\\P.exe")) == Join(lib, L"UE\\start_protected_game.exe"), "Unreal root marker");
            require(AntiCheatMessage(L"K:\\G\\BattlEye") == L"Anti-cheat found: K:\\G\\BattlEye - games with anti-cheat are never supported", "AntiCheatMessage");
            const wstring longPath = L"K:\\Games\\Some Very Long Game Folder Name\\bin\\x64\\BattlEye";
            const wstring s = ShortenMiddle(longPath, 30);
            require(s.size() <= 30 && s.find(L'\u2026') != wstring::npos && s.rfind(L"K:\\Games\\", 0) == 0 && EndsWithI(s, L"\\BattlEye") &&
                        ShortenMiddle(L"K:\\G\\BattlEye", 30) == L"K:\\G\\BattlEye",
                    "ShortenMiddle keeps the drive and the marker");
        }

        // a store's launch exe is a hint only: a launcher stub (REDprelauncher.exe) gives way to the exe discovery
        {
            auto file = [](const wstring& p) { EnsureDir(Parent(p)); WriteFileAtomic(p, "MZ", 2); };
            auto json = [](wstring s) { wstring o; for (wchar_t c : s) { if (c == L'\\') o += L'\\'; o += c; } return o; };
            auto item = [&](const wstring& name, const wstring& loc, const wstring& launch) {
                return L"{\n\t\"FormatVersion\": 0,\n\t\"DisplayName\": \"" + name + L"\",\n\t\"InstallLocation\": \"" + json(loc) +
                       L"\",\n\t\"LaunchExecutable\": \"" + json(launch) + L"\",\n\t\"AppName\": \"x\"\n}\n";
            };
            const wstring epic = Join(dir, L"Epic Games");
            const wstring cp = Join(epic, L"Cyberpunk2077");
            file(Join(cp, L"REDprelauncher.exe"));
            file(Join(cp, L"bin\\x64\\Cyberpunk2077.exe"));
            file(Join(cp, L"bin\\x64\\amd_fidelityfx_dx12.dll"));
            const StoreGame cpItem = EpicItemGame(item(L"Cyberpunk 2077", cp, L"REDprelauncher.exe"));
            require(cpItem.name == L"Cyberpunk 2077" && cpItem.root == cp && cpItem.hint == Join(cp, L"REDprelauncher.exe"), "Epic manifest: name, install folder, launch exe");
            require(IsLauncherStub(cpItem.hint) && StoreExe(cpItem.root, cpItem.hint).empty(), "REDprelauncher.exe is a launcher stub: no hint");
            const auto cpFound = DetectGames({cpItem});
            require(cpFound.size() == 1 && cpFound[0].exe == Join(cp, L"bin\\x64\\Cyberpunk2077.exe") && cpFound[0].validated && cpFound[0].source == L"Epic",
                    "Epic Cyberpunk 2077 with LaunchExecutable REDprelauncher.exe: bin\\x64\\Cyberpunk2077.exe, the validated game");
            require(ResolveGameExe(Join(cp, L"REDprelauncher.exe")) == Join(cp, L"bin\\x64\\Cyberpunk2077.exe"), "the stub named by hand (Add game, the CLI): the game in its folder");
            // the same from GOG's registry exe (StoreExe is the same rule for every store)
            require(DetectGames({{L"Cyberpunk 2077", cp, Join(cp, L"REDprelauncher.exe"), L"GOG"}})[0].exe == Join(cp, L"bin\\x64\\Cyberpunk2077.exe"), "GOG: the stub gives way too");
            // a launch exe that isn't a stub but doesn't lead to the known game the folder holds gives way to it
            const wstring cp2 = Join(epic, L"CP Other Start");
            file(Join(cp2, L"start.exe"));
            file(Join(cp2, L"bin\\x64\\Cyberpunk2077.exe"));
            file(Join(cp2, L"bin\\x64\\amd_fidelityfx_dx12.dll"));
            require(StoreExe(cp2, Join(cp2, L"start.exe")).empty() && DetectGames({EpicItemGame(item(L"CP", cp2, L"start.exe"))})[0].exe == Join(cp2, L"bin\\x64\\Cyberpunk2077.exe"),
                    "a launch exe beside a known game's exe gives way to it");
            // a normal Epic game keeps its LaunchExecutable (even where the discovery alone would pick another exe)
            const wstring q = Join(epic, L"QuarryNine");
            file(Join(q, L"Game\\QuarryNine.exe"));
            file(Join(q, L"Game\\amd_fidelityfx_dx12.dll"));
            { const string big(4096, 'x'); EnsureDir(Join(q, L"Extras")); WriteFileAtomic(Join(q, L"Extras\\Gallery.exe"), big.data(), big.size()); }
            const StoreGame qItem = EpicItemGame(item(L"Quarry Nine", q, L"Game\\QuarryNine.exe"));
            require(StoreExe(qItem.root, qItem.hint) == Join(q, L"Game\\QuarryNine.exe"), "a normal Epic game: its launch exe is the hint");
            const auto qFound = DetectGames({qItem});
            require(qFound.size() == 1 && qFound[0].exe == Join(q, L"Game\\QuarryNine.exe"), "a normal Epic game keeps its LaunchExecutable");
            require(StoreExe(q, L"").empty() && StoreExe(q, Join(q, L"Missing.exe")) == Join(q, L"Missing.exe"), "no launch exe: none; a missing one is passed on (FindGameExe ignores it)");
            // rc6 L3: an Unreal game whose real exe isn't *-Shipping (Hogwarts Legacy): the root starter of the same name leads to
            // <Project>\Binaries\Win64\<name>.exe, from a store's launch exe and when added by hand
            const wstring hl = Join(epic, L"Hogwarts Legacy");
            file(Join(hl, L"HogwartsLegacy.exe"));
            file(Join(hl, L"Phoenix\\Binaries\\Win64\\HogwartsLegacy.exe"));
            file(Join(hl, L"Phoenix\\Binaries\\Win64\\nvngx_dlss.dll"));
            const wstring hlGame = Join(hl, L"Phoenix\\Binaries\\Win64\\HogwartsLegacy.exe");
            require(SameNameGameExe(Join(hl, L"HogwartsLegacy.exe")) == hlGame && SameNameGameExe(hlGame).empty(), "SameNameGameExe: the root starter's game; none from Binaries\\Win64");
            require(FindGameExe(hl, Join(hl, L"HogwartsLegacy.exe")) == hlGame, "FindGameExe: a root starter as the hint gives the game below it");
            require(ResolveGameExe(Join(hl, L"HogwartsLegacy.exe")) == hlGame && ResolveGameExe(hlGame) == hlGame, "ResolveGameExe: the root starter added by hand is the game");
            const auto hlFound = DetectGames({EpicItemGame(item(L"Hogwarts Legacy", hl, L"HogwartsLegacy.exe"))});
            require(hlFound.size() == 1 && hlFound[0].exe == hlGame, "Epic Hogwarts Legacy with the root exe as LaunchExecutable: the game in Phoenix\\Binaries\\Win64");
            // ...and <root>\bin\x64\<name>.exe of the same name
            const wstring bx = Join(epic, L"Bin Layout");
            file(Join(bx, L"BinGame.exe"));
            file(Join(bx, L"bin\\x64\\BinGame.exe"));
            require(ResolveGameExe(Join(bx, L"BinGame.exe")) == Join(bx, L"bin\\x64\\BinGame.exe"), "a root starter with bin\\x64\\<name>.exe gives that exe");
            require(ResolveGameExe(Join(q, L"Game\\QuarryNine.exe")) == Join(q, L"Game\\QuarryNine.exe"), "an exe without a game of its name below it stays");
        }

        // the Context's log file
        Context ctx;
        ctx.logFile = Join(dir, L"setup.log");
        LogLine(ctx, L"first");
        LogLine(ctx, L"second \u00e9");
        string log;
        require(ReadAll(ctx.logFile, log) && log.find("] first\r\n") != string::npos && log.find("] second \xc3\xa9\r\n") != string::npos, "LogLine appends UTF-8 lines");
        RemoveTree(dir);
        require(!IsDir(dir), "RemoveTree");
    } catch (const std::exception& ex) {
        std::cerr << "FAILED: " << ex.what() << " (after " << checks << " checks)\n";
        return 1;
    } catch (const SetupError& ex) {
        std::wcerr << L"FAILED: " << ex.msg << L"\n";
        return 1;
    }
    std::cout << checks << " checks passed\n";
    return 0;
}
