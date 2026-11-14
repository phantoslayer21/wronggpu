// SPDX-License-Identifier: MIT
// Unit tests of wg_core (the installer's / launcher's engine) that need no game, no payload and no GPU: strings and
// paths, the install record, wronggpu.ini keys, the runtime options of a Model-M-only build, the Context's log file.
// Written for this project. Run by ctest ("wg-core") in a WRONGGPU_TESTING build.
#include <atomic>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "wg_core/checks.h"
#include "wg_core/context.h"
#include "wg_core/gamerules.h"
#include "wg_core/model.h"
#include "wg_core/record.h"
#include "wg_core/rrprep.h"
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
// StartCleanup's folder move for the retry cases (record.h SetCleanupMoveForTest): the first g_moveFailFirst calls fail
// with g_moveError, later ones really move.
std::atomic<int> g_moveCalls{0}, g_moveFailFirst{0};
std::atomic<DWORD> g_moveError{0};
BOOL WINAPI FakeMove(LPCWSTR from, LPCWSTR to, DWORD flags) {
    if (++g_moveCalls <= g_moveFailFirst) { SetLastError(g_moveError); return FALSE; }
    return MoveFileExW(from, to, flags);
}
size_t CountIn(const string& s, const string& part) {
    size_t n = 0;
    for (size_t at = 0; (at = s.find(part, at)) != string::npos; ++at) ++n;
    return n;
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
        // 0.6.1: WRONGGPU_HIP_RUNTIME (the development override of the HIP runtime search) is read only by a WRONGGPU_TESTING
        // build. The candidate list only names files that exist (nothing is loaded): a few bytes of text stand in for the
        // library in the test's own temp folder; the test's variables are put back afterwards.
        {
            const wstring savedHip = EnvVar(L"WRONGGPU_HIP_RUNTIME"), savedMocks = EnvVar(L"WRONGGPU_TEST_HIP_CANDIDATES");
            const wstring hipDir = TempFolder(), fake = Join(hipDir, L"amdhip64_7.dll");
            if (FILE* f = _wfopen(fake.c_str(), L"wb")) { std::fputs("not a HIP runtime (wg-core-tests)", f); std::fclose(f); }
            require(IsFile(fake), "HIP search: the stand-in file is there");
            SetEnvironmentVariableW(L"WRONGGPU_TEST_HIP_CANDIDATES", nullptr);
            SetEnvironmentVariableW(L"WRONGGPU_HIP_RUNTIME", fake.c_str());
            const std::vector<wstring> cands = HipCandidates();
            bool named = false;
            for (auto& c : cands) named = named || IEq(c, fake);
            SetEnvironmentVariableW(L"WRONGGPU_HIP_RUNTIME", savedHip.empty() ? nullptr : savedHip.c_str());
            SetEnvironmentVariableW(L"WRONGGPU_TEST_HIP_CANDIDATES", savedMocks.empty() ? nullptr : savedMocks.c_str());
            RemoveTree(hipDir);
#ifdef WRONGGPU_TESTING
            require(!cands.empty() && IEq(cands[0], fake), "HIP search: a testing build tries WRONGGPU_HIP_RUNTIME first");
#else
            require(!named, "HIP search: a release build never reads WRONGGPU_HIP_RUNTIME");
#endif
            (void)named;
        }
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

        // RR's kernels under Wine: the game prepares them at its first start (its shader caches are under its own Proton);
        // no helper runs. On Windows a game without a rule is prepared by the launcher.
        {
            Context rctx;
            rctx.logFile = Join(dir, L"rr-prep.log");
            const wstring game = Join(dir, L"No Rule Game\\NoRuleGame.exe");
            wstring why;
            if (UnderWine()) {
                require(RrKernelsInGame(game) && RrPrepStateOf(rctx, dir, game, 1, &why) == RrPrepState::InGame && Contains(why, L"Proton"),
                        "RR kernels under Wine: prepared by the game at its first start");
                require(PrepareRr(rctx, dir, game, Join(dir, L"no-runtime.dll"), 1, nullptr, &why) == 4, "PrepareRr under Wine: no helper run (4)");
            } else {
                require(!RrKernelsInGame(game), "RR kernels on Windows: a game without a rule is prepared by the launcher");
            }
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

        // Linux's device list from a made-up folder (any host): every card in number order, connectors skipped, junk and
        // missing files ignored, the reported name says the card was not confirmed as the game's adapter
        {
            const wstring drm = Join(dir, L"drm-fixture");
            auto card = [&](const wstring& name, const char* vendor, const char* device) {
                EnsureDir(Join(drm, name + L"\\device"));
                if (vendor) WriteFileAtomic(Join(drm, name + L"\\device\\vendor"), vendor, strlen(vendor));
                if (device) WriteFileAtomic(Join(drm, name + L"\\device\\device"), device, strlen(device));
            };
            card(L"card0", "0x1002\n", "0x15bf\n");      // an integrated Radeon, not RDNA 4
            card(L"card1", "0x10de\n", "0x7550\n");      // another vendor with an RDNA 4 device id
            card(L"card2", "zz\n", "0x7550\n");          // a vendor that is not a number
            card(L"card3", "0x1002\n", nullptr);          // the device file missing
            card(L"card4-DP-1", "0x1002\n", "0x7550\n"); // a connector, not a card
            require(LinuxRdna4Name(drm).empty(), "LinuxRdna4Name: none among an iGPU, another vendor, junk, a missing file, a connector");
            require(LinuxRdna4Name(Join(dir, L"no-such-drm")).empty(), "LinuxRdna4Name: a missing device list");
            card(L"card20", "0x1002\n", "0x7590\n");     // past the first 16 cards
            const wstring past16 = LinuxRdna4Name(drm);
            require(Contains(past16, L"7590") && Contains(past16, L"not confirmed"), "LinuxRdna4Name: a card numbered 20, reported as not confirmed");
            card(L"card17", "0x1002\n", "0x7550 \n");    // two RDNA 4 cards: the lower number first
            require(Contains(LinuxRdna4Name(drm), L"7550"), "LinuxRdna4Name: cards in number order");
        }
        // under Wine an RDNA 4 card that Linux reports is found even when Wine's own DXGI names an old Radeon
        if (UnderWine()) {
            auto sysfs = [](const wstring& p) {  // the C runtime reads to the end of file (sysfs reports 4096 bytes)
                string s;
                if (FILE* f = _wfopen(p.c_str(), L"rb")) { char b[64]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n); fclose(f); }
                return s;
            };
            bool linuxRdna4 = false;
            for (int i = 0; i < 16 && !linuxRdna4; ++i) {
                const wstring dev = L"Z:\\sys\\class\\drm\\card" + std::to_wstring(i) + L"\\device\\";
                const unsigned long v = strtoul(sysfs(dev + L"vendor").c_str(), nullptr, 16), id = strtoul(sysfs(dev + L"device").c_str(), nullptr, 16);
                linuxRdna4 = v == 0x1002 && (id == 0x7550 || id == 0x7551 || id == 0x7590 || id == 0x7591);
            }
            const wstring fromLinux = LinuxRdna4Name();
            const GpuInfo gpu = DetectGpu();
            std::wcout << L"under Wine: Linux reports RDNA 4: " << linuxRdna4 << L"; LinuxRdna4Name: \"" << fromLinux << L"\"; DetectGpu: " << gpu.name << L" (rdna4 " << gpu.rdna4 << L")\n";
            require(fromLinux.empty() != linuxRdna4, "LinuxRdna4Name: the RDNA 4 card Linux reports, and only then");
            if (linuxRdna4) require(gpu.amd && gpu.rdna4, "DetectGpu under Wine: the RDNA 4 card Linux reports");
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

        // paths for a game in another Wine prefix (Proton): the Z: name of the same file; unchanged outside Wine
        {
            const wstring f = Join(dir, L"prefix-path.txt");
            WriteFileAtomic(f, "same file", 9);
            const wstring g = PathForGamePrefix(f);
            string back;
            if (UnderWine())
                require(StartsWithI(g, L"Z:\\") && ReadAll(g, back) && back == "same file", "PathForGamePrefix under Wine: a Z: path to the same file");
            else
                require(g == f, "PathForGamePrefix outside Wine: the path unchanged");
            // two folders that do not exist yet (the RR data's F\ before its first conversion): still the Z: path, the tail kept
            const wstring deepTail = L"\\no-such-dir\\F\\model.wgcm", deep = PathForGamePrefix(dir + deepTail);
            if (UnderWine())
                require(StartsWithI(deep, L"Z:\\") && deep.size() > deepTail.size() && deep.compare(deep.size() - deepTail.size(), deepTail.size(), deepTail) == 0,
                        "PathForGamePrefix under Wine: a Z: path for missing folders too");
            require(PathForGamePrefix(L"").empty(), "PathForGamePrefix of an empty path");
            // a drive the prefix doesn't map: no Z: path can be made, the path comes back as it is (the install then
            // reports it, install.cpp ModelPathForGame)
            if (UnderWine()) {
                const wstring unmapped = L"Q:\\wg-no-such-drive\\model";
                require(PathForGamePrefix(unmapped) == unmapped, "PathForGamePrefix under Wine: an unmapped drive comes back unchanged");
            }
        }
        // the Context's log file
        Context ctx;
        ctx.logFile = Join(dir, L"setup.log");
        LogLine(ctx, L"first");
        LogLine(ctx, L"second \u00e9");
        string log;
        require(ReadAll(ctx.logFile, log) && log.find("] first\r\n") != string::npos && log.find("] second \xc3\xa9\r\n") != string::npos, "LogLine appends UTF-8 lines");

        // StartCleanup: the shared folder moved aside to a name that is free, with bounded retries. Synthetic folder in the
        // temp folder; the hidden cmd.exe it starts deletes each moved-aside folder a few seconds later.
        {
            const wstring cleanRoot = Join(dir, L"cleanup"), appData = Join(cleanRoot, L"WrongGPU");
            const wstring modelFile = Join(appData, L"model\\model.wgcm");
            auto makeShared = [&] { EnsureDir(Parent(modelFile)); const char d[] = "synthetic model data"; WriteFileAtomic(modelFile, d, sizeof d - 1); };
            auto contains = [](const string& s, const string& part) { return s.find(part) != string::npos; };
            // two cleanups of one folder back to back, 20 runs (CI has one runner: not more). Within one GetTickCount64 step
            // (~16 ms) the second move used to aim at the first one's folder, which its cmd.exe deletes seconds later: error
            // 183, the folder left in place.
            {
                Context sctx;
                sctx.logFile = Join(cleanRoot, L"cleanup-stress.log");  // outside the folder that is moved
                const int runs = 20;
                int gone = 0;
                for (int i = 0; i < runs; ++i) {
                    makeShared();
                    StartCleanup(sctx, L"", L"", appData);
                    const bool first = !IsDir(appData);
                    makeShared();
                    StartCleanup(sctx, L"", L"", appData);
                    if (first && !IsDir(appData)) ++gone;
                }
                string slog;
                ReadAll(sctx.logFile, slog);
                std::printf("wg-core: cleanup stress %d/%d (deferred %zu)\n", gone, runs, CountIn(slog, "cleanup deferred"));
                require(gone == runs && !contains(slog, "cleanup deferred"), "two cleanups of one folder back to back: it goes both times, 20 of 20 runs");
            }
            // the bounded retry (error 32 sharing violation, error 5 access denied: a scanner's open file), through the test
            // seam: fails for the first N tries, then moves; or fails every time and ends deferred; any other error is final
            {
                Context rctx;
                rctx.logFile = Join(cleanRoot, L"cleanup-retry.log");
                struct Case { DWORD error; int failFirst; bool gone; int calls; size_t retries; const char* what; };
                const Case cases[] = {
                    {ERROR_ACCESS_DENIED, 3, true, 4, 3, "error 5 for 3 tries, then the move: the folder goes after 3 logged retries"},
                    {ERROR_ACCESS_DENIED, 1000, false, kCleanupMoveTries, kCleanupMoveTries - 1,
                     "error 5 every time: 5 tries (4 logged retries), then deferred with the error named, the folder left in place"},
                    {ERROR_SHARING_VIOLATION, 2, true, 3, 2, "error 32 for 2 tries, then the move: the folder goes"},
                    {ERROR_ALREADY_EXISTS, 1, false, 1, 0, "error 183 once: no retry, deferred at once"}};
                for (const Case& rc : cases) {
                    DeleteFileW(rctx.logFile.c_str());
                    makeShared();
                    g_moveCalls = 0; g_moveFailFirst = rc.failFirst; g_moveError = rc.error;
                    SetCleanupMoveForTest(&FakeMove);
                    const ULONGLONG t0 = GetTickCount64();
                    StartCleanup(rctx, L"", L"", appData);
                    const ULONGLONG took = GetTickCount64() - t0;
                    SetCleanupMoveForTest(nullptr);
                    string rlog;
                    ReadAll(rctx.logFile, rlog);
                    const string code = "(error " + std::to_string(rc.error);
                    const size_t retries = CountIn(rlog, code + (rc.error == ERROR_SHARING_VIOLATION ? ", sharing violation); trying again" : ", access denied); trying again"));
                    const size_t deferred = CountIn(rlog, "cleanup deferred: cannot detach " + ToUtf8(appData) + " " + code + ":");
                    std::printf("wg-core: cleanup retry, error %lu x %d: %d tries, %zu retries logged, deferred %zu, %llu ms\n", rc.error, rc.failFirst,
                                g_moveCalls.load(), retries, deferred, (unsigned long long)took);
                    require(IsDir(appData) == !rc.gone && g_moveCalls == rc.calls && retries == rc.retries && deferred == (rc.gone ? 0u : 1u) && took < 5000, rc.what);
                }
                if (IsDir(appData)) {  // the deferred cases left it: the real move now
                    StartCleanup(rctx, L"", L"", appData);
                    require(!IsDir(appData), "after the retry cases: the folder goes");
                }
            }
            for (int i = 0; i < 600; ++i) {  // the hidden cleanups delete the moved-aside folders (ping -n 3 first)
                bool left = false;
                for (auto& d : ListDirs(cleanRoot)) left = left || d.find(L".cleanup-") != wstring::npos;
                if (!left) break;
                Sleep(100);
            }
        }

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
