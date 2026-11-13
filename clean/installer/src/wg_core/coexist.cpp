// SPDX-License-Identifier: MIT
// WrongGPU launcher core: other mods, tools and older installs next to a game (coexist.h): the installer's conflict
// rules, the launcher's rules table (build/launcher-design.md section 5) and the proxy picker. Written for this project.
#include "coexist.h"

#include <tlhelp32.h>

#include <algorithm>
#include <mutex>

#include "checks.h"  // FindAntiCheat, VersionLess
#include "gamerules.h"
#include "optiscaler.h"
#include "pe.h"
#include "rr.h"

namespace wg {

bool IsLegacyLoader(const wstring& path) { return IsFile(path) && FileContainsText(path, "dlss45", 64ull << 20); }

// The names OptiScaler loads as (its proxy names, its ASI name, its older layout's nvngx.dll).
static const wchar_t* const kOptiLoadNames[] = {L"dxgi.dll", L"winmm.dll", L"version.dll", L"dbghelp.dll", L"d3d12.dll", L"wininet.dll",
                                                L"winhttp.dll", L"OptiScaler.asi", L"nvngx.dll"};
static const wchar_t* const kOptiMarkers[] = {L"OptiScaler.ini", L"OptiScaler.log", L"OptiScaler.dll"};

Conflicts FindConflicts(const wstring& dir, const wstring& exeDir) {
    Conflicts c;
    std::set<wstring> files;
    std::vector<wstring> dirs{dir};
    if (!IEq(FullPath(exeDir), FullPath(dir))) dirs.push_back(exeDir);
    for (size_t di = 0; di < dirs.size(); ++di) {
        const wstring& d = dirs[di];
        std::vector<wstring> hits;
        for (auto f : kOptiMarkers) if (IsFile(Join(d, f))) hits.push_back(f);
        // the player's OptiScaler by its fingerprint, under whichever name it loads (never by the name alone)
        for (auto f : kOptiLoadNames) if (IsFile(Join(d, f)) && IdentifyFile(Join(d, f)).tool == Tool::OptiScaler) hits.push_back(f);
        if (hits.empty()) continue;
        if (std::find(c.mods.begin(), c.mods.end(), L"OptiScaler") == c.mods.end()) c.mods.push_back(L"OptiScaler");
        for (auto& h : hits) {
            if (di == 0) files.insert(h);
            else c.outside.push_back(Join(d, h));
        }
    }
    c.modFiles.assign(files.begin(), files.end());
    // Older WrongGPU / DLSS 4.5 AMD: a record next to the exe or in the install folder, or loose files without one.
    for (auto& d : dirs) if (c.legacyRecordDir.empty() && IsFile(LegacyRecordPath(d))) c.legacyRecordDir = d;
    if (c.legacyRecordDir.empty()) {
        for (size_t di = 0; di < dirs.size(); ++di) {
            std::vector<wstring> loose;
            for (auto f : kLegacyFiles) if (IsFile(Join(dirs[di], f))) loose.push_back(f);
            for (auto f : kLegacyLoaderNames) if (IsLegacyLoader(Join(dirs[di], f))) loose.push_back(f);
            std::vector<wstring> stack{Join(dirs[di], L"dlss45-model")};
            while (!stack.empty() && loose.size() < 512) {
                wstring m = stack.back(); stack.pop_back();
                for (auto& sub : ListDirs(m)) stack.push_back(sub);
                for (auto& f : ListFiles(m, L"*")) loose.push_back(f.substr(dirs[di].size() + 1));
            }
            for (auto& l : loose) {
                if (di == 0) c.legacyFiles.push_back(l);
                else c.outside.push_back(Join(dirs[di], l));
            }
        }
    }
    return c;
}

// ---- the rules table -------------------------------------------------------------------------------------------------
const wchar_t* DecisionName(Decision d) {
    switch (d) {
        case Decision::Managed: return L"managed";
        case Decision::AskSwitch: return L"ask-switch";
        case Decision::Chain: return L"chain";
        case Decision::Warn: return L"warn";
        case Decision::Inform: return L"inform";
        case Decision::Refuse: return L"refuse";
        case Decision::Migrate: return L"migrate";
    }
    return L"?";
}
bool ParseDecision(const wstring& s, Decision& d) {
    for (Decision x : {Decision::Managed, Decision::AskSwitch, Decision::Chain, Decision::Warn, Decision::Inform, Decision::Refuse, Decision::Migrate})
        if (IEq(s, DecisionName(x))) { d = x; return true; }
    return false;
}
const Finding* CoexistReport::refusal() const { for (auto& f : findings) if (f.decision == Decision::Refuse) return &f; return nullptr; }
bool CoexistReport::has(Tool t) const { for (auto& f : findings) if (f.tool == t) return true; return false; }
bool CoexistReport::needsSwitch() const { for (auto& f : findings) if (f.decision == Decision::AskSwitch) return true; return false; }
std::vector<wstring> CoexistReport::Warnings() const {
    std::vector<wstring> w;
    for (auto& f : findings)  // (other plugins: the install says what it did about them in its own words)
        if ((f.decision == Decision::Chain || f.decision == Decision::Warn) && f.tool != Tool::OtherPlugin) w.push_back(f.what + L": " + f.message);
    if (proxy.untested && !proxy.name.empty()) w.push_back(L"OptiScaler as " + proxy.name + L" next to ReShade / Special K: an untested combination");
    return w;
}

// The DLLs a fingerprint scan looks at next to the exe: the names Windows loads from a game folder, and the tools' own
// DLL names (a tool is still recognised by what the file contains, whatever its name).
static const wchar_t* const kScanNames[] = {
    L"dxgi.dll", L"winmm.dll", L"version.dll", L"wininet.dll", L"winhttp.dll", L"dbghelp.dll", L"d3d12.dll", L"d3d12core.dll", L"d3d11.dll",
    L"d3d10core.dll", L"d3d9.dll",
    L"dinput8.dll", L"dsound.dll", L"dwmapi.dll", L"xinput1_3.dll", L"xinput1_4.dll", L"xinput9_1_0.dll", L"opengl32.dll",
    L"nvngx.dll", L"nvapi64.dll", L"OptiScaler.dll", L"OptiScaler.asi", L"SpecialK64.dll", L"ReShade64.dll", L"UE4SS.dll",
    L"ue4ss\\UE4SS.dll", L"red4ext\\RED4ext.dll"};

static wstring JoinList(const std::vector<wstring>& v) { wstring s; for (auto& x : v) s += (s.empty() ? L"" : L", ") + x; return s; }

// ---- overlays and injectors (running processes) ------------------------------------------------------------------------
struct OverlayDef { const wchar_t* exe; const wchar_t* name; Decision d; const wchar_t* message; };
static const OverlayDef kOverlays[] = {
    {L"RTSS.exe", L"RivaTuner Statistics Server", Decision::Inform, L"Its overlay works next to WrongGPU."},
    {L"MSIAfterburner.exe", L"MSI Afterburner", Decision::Inform, L"Its overlay (through RTSS) works next to WrongGPU."},
    {L"LosslessScaling.exe", L"Lossless Scaling", Decision::Inform, L"Upscaling twice is pointless: use its frame generation, not its upscaling, with WrongGPU."},
    {L"obs64.exe", L"OBS Studio", Decision::Inform, L"Game capture works next to WrongGPU."},
    {L"Discord.exe", L"Discord", Decision::Inform, L"Its in-game overlay works next to WrongGPU."},
    {L"Overwolf.exe", L"Overwolf", Decision::Inform, L"Its overlay is shown for information only."},
    {L"GameOverlayUI.exe", L"Steam overlay", Decision::Inform, L"Works next to WrongGPU."},
    {L"EOSOverlayRenderer-Win64-Shipping.exe", L"Epic Online Services overlay", Decision::Inform, L"Works next to WrongGPU."},
    {L"SKIF.exe", L"Special K (global injection)", Decision::Warn, L"Special K's global injector is running and may inject into this game: an untested combination with WrongGPU."},
};
const wchar_t* OverlayFor(const wstring& process, Decision& d, const wchar_t** message) {
    for (auto& o : kOverlays)
        if (IEq(process, o.exe)) { d = o.d; if (message) *message = o.message; return o.name; }
    return nullptr;
}
std::vector<wstring> RunningProcesses() {
#ifdef WRONGGPU_TESTING
    {
        wchar_t buf[2048];
        const DWORD n = GetEnvironmentVariableW(L"WRONGGPU_TEST_PROCESSES", buf, 2048);
        std::vector<wstring> r;  // tests see only the processes they name (none when the variable is not set)
        if (n > 0 && n < 2048) {
            wstring s(buf, n), cur;
            for (wchar_t c : s + L";") { if (c == L';') { if (!cur.empty()) r.push_back(cur); cur.clear(); } else cur += c; }
        }
        return r;
    }
#endif
    static std::mutex mu;
    static std::vector<wstring> cache;
    static ULONGLONG at = 0;
    std::lock_guard<std::mutex> lock(mu);
    if (at && GetTickCount64() - at < 3000) return cache;
    cache.clear();
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (s != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W pe{}; pe.dwSize = sizeof(pe);
        for (BOOL ok = Process32FirstW(s, &pe); ok; ok = Process32NextW(s, &pe)) cache.push_back(pe.szExeFile);
        CloseHandle(s);
    }
    at = GetTickCount64();
    return cache;
}

// ---- the proxy picker ------------------------------------------------------------------------------------------------
ProxyChoice PickProxy(const wstring& exeDir, const Record* ours, const wstring& playersOptiProxy, bool hipTool, bool untested) {
    ProxyChoice c;
    c.untested = untested;
    const wstring suffix = untested ? L" (next to ReShade / Special K: an untested combination)" : L"";
    if (ours && !ours->proxy.empty()) {
        const wstring p = Join(exeDir, ours->proxy);
        bool ok = !IsFile(p);
        if (!ok) {
            const wstring h = Sha256File(p);
            for (auto& e : ours->e) ok = ok || (e.action == L"copy" && IEq(e.rel, ours->proxy) && !e.inst.empty() && h == e.inst);
        }
        if (ok) { c.name = ours->proxy; c.reason = L"the name of WrongGPU's earlier install" + suffix; return c; }
    }
    wstring order, skipped;
    for (auto n : kProxyOrder) order += (order.empty() ? L"" : L", ") + wstring(n);
    for (auto n : kProxyOrder) {
        if (hipTool && IEq(n, L"version.dll")) { skipped += (skipped.empty() ? L"" : L", ") + wstring(n) + L" (never next to a HIP-based tool)"; continue; }
        const wstring p = Join(exeDir, n);
        if (!IsFile(p)) {
            c.name = n;
            c.reason = (skipped.empty() ? L"the first free name of " + order : wstring(n) + L" is free; taken: " + skipped) + suffix;
            return c;
        }
        const Fingerprint fp = IdentifyFile(p);
        const wstring who = fp.tool == Tool::Unknown ? L"an unknown file" : fp.tool == Tool::Microsoft ? L"the game's own Windows DLL" : ToolName(fp.tool);
        skipped += (skipped.empty() ? L"" : L", ") + wstring(n) + L" (" + who + L")";
    }
    // A switch: the player's OptiScaler is moved aside (backed up as "moved"); its name is taken only when no other is free.
    if (!playersOptiProxy.empty()) {
        c.name = playersOptiProxy;
        c.reason = L"where your OptiScaler was (" + playersOptiProxy + L"; your file is kept in the backup); taken: " + skipped + suffix;
        return c;
    }
    c.reason = L"no free name: " + skipped;
    return c;
}

// ---- other setups' OptiScaler plugins (0.6.1) ------------------------------------------------------------------------
bool ExportsRrContract(const wstring& path) {
    MappedFile f(path, 256ull << 20);  // the file's bytes only: never loaded as a module
    return f.data && ReadPeExports(f.data, f.size).count(kRrContractExport) > 0;
}
// exeDir relative to dir ("" when the same folder); false when exeDir is not dir or a folder below it
static bool RelativeTo(const wstring& dir, const wstring& exeDir, wstring& rel) {
    const wstring d = FullPath(dir), x = FullPath(exeDir);
    rel.clear();
    if (IEq(d, x)) return true;
    const wstring base = !d.empty() && d.back() == L'\\' ? d : d + L"\\";
    if (x.size() <= base.size() || !StartsWithI(x, base)) return false;
    rel = x.substr(base.size());
    return true;
}
// The .asi files of OptiScaler's plugin folders that the install record does not name with that hash; withMarker: those that
// export WrongGPU RR's marker (true) or the others (false). The record's paths are relative to recordDir (empty: exeDir)
static std::vector<wstring> UnrecordedPlugins(const wstring& exeDir, const Record* ours, bool withMarker, const wstring& recordDir = L"") {
    std::vector<wstring> out;
    wstring prefix;
    if (ours && !recordDir.empty() && !RelativeTo(recordDir, exeDir, prefix)) ours = nullptr;  // (it cannot name them)
    for (auto sub : kPluginDirs)
        for (auto& f : ListFiles(Join(exeDir, sub), L"*.asi")) {
            const wstring rel = f.substr(exeDir.size() + 1), recRel = prefix.empty() ? rel : Join(prefix, rel);
            bool mine = false;
            if (ours)
                for (auto& e : ours->e)
                    if (e.action == L"copy" && IEq(e.rel, recRel) && !e.inst.empty()) { mine = IEq(Sha256File(f), e.inst); break; }
            if (!mine && ExportsRrContract(f) == withMarker) out.push_back(rel);
        }
    return out;
}
std::vector<wstring> ForeignPlugins(const wstring& exeDir, const Record* ours, const wstring& recordDir) {
    return UnrecordedPlugins(exeDir, ours, false, recordDir);
}
PluginPlan PlanPluginsAside(const wstring& exeDir, const wstring& dir, const std::vector<wstring>& plugins, const std::vector<wstring>& agreed,
                            const std::map<wstring, wstring>& movedBefore) {
    PluginPlan p;
    wstring prefix;
    const bool inside = RelativeTo(dir, exeDir, prefix);
    for (auto& f : plugins) {
        const wstring rel = prefix.empty() ? f : Join(prefix, f);
        bool ok = std::any_of(agreed.begin(), agreed.end(), [&](const wstring& a) { return IEq(a, f); });
        if (!ok && inside) {
            const auto it = movedBefore.find(Lower(rel));
            ok = it != movedBefore.end() && IEq(Sha256File(Join(exeDir, f)), it->second);
        }
        if (ok && inside) p.moveAside.push_back(rel);
        else { p.blocking.push_back(f); p.outside = p.outside || !inside; }
    }
    return p;
}
std::vector<wstring> OurPluginsFromOtherPackages(const wstring& exeDir, const Record* ours) { return UnrecordedPlugins(exeDir, ours, true); }
wstring ForeignPluginsText(const std::vector<wstring>& plugins) {
    return L"Other OptiScaler plugins are in this game (" + JoinList(plugins) + L"). They may hook Ray Reconstruction as WrongGPU's RR plugin does, so "
           L"WrongGPU's Ray Reconstruction was left out. To use it, install again and choose to move them aside (they are backed up and put back "
           L"when WrongGPU is uninstalled).";
}

// ---- the assessment --------------------------------------------------------------------------------------------------
CoexistReport AssessCoexistence(Context& ctx, const wstring& exe, const wstring& dirIn, const Record* ours) {
    CoexistReport r;
    const wstring exeDir = Parent(exe), dir = dirIn.empty() ? exeDir : dirIn;
    std::map<wstring, wstring> ourHash;  // lower-case path relative to dir -> installed hash
    if (ours) for (auto& e : ours->e) if (e.action == L"copy") ourHash[Lower(e.rel)] = e.inst;
    const bool sameDir = IEq(FullPath(exeDir), FullPath(dir));
    // WrongGPU's OptiScaler setup (the record has OptiScaler.ini): its log and settings are ours, and so is the OptiScaler
    // DLL the record copied under that name with that hash (isOurs; 0.6.1: never by a hash alone under another name, never
    // by the package's hash: another setup may use the same OptiScaler build)
    const bool hostedOurs = ours && HostedRecord(*ours) && sameDir;
    auto isOurs = [&](const wstring& rel) {
        if (!sameDir) return false;
        if (hostedOurs && IEq(rel, L"OptiScaler.log")) return true;
        if (hostedOurs && IEq(rel, L"OptiScaler.ini")) return OptiIniIsOurs(dir, *ours);  // 0.6.1: beside WrongGPU's OptiScaler, or pointing at WrongGPU's files
        auto it = ourHash.find(Lower(rel));
        if (it == ourHash.end()) return false;
        if (EndsWithI(rel, L".ini") || EndsWithI(rel, L".log")) return true;  // settings and logs of our install
        return !it->second.empty() && Sha256File(Join(dir, rel)) == it->second;
    };
    auto add = [&](Tool t, Decision d, const wstring& what, const std::vector<wstring>& files, const wstring& evidence, const wstring& message, int code = 0) {
        Finding f; f.tool = t; f.decision = d; f.what = what; f.files = files; f.evidence = evidence; f.message = message; f.code = code;
        r.findings.push_back(f);
    };

    // anti-cheat: refused with no override (only removing WrongGPU's own recorded install is allowed)
    const wstring ac = FindAntiCheat(exe);
    if (!ac.empty())
        add(Tool::AntiCheat, Decision::Refuse, L"Anti-cheat (" + ShortenMiddle(ac, 60) + L")", {}, ac,
            (ac.find(L'\\') != wstring::npos ? L"Found: " + ac + L". " : wstring()) +
                L"WrongGPU is never installed into games with anti-cheat; only removing WrongGPU's own recorded install is allowed.", EXIT_ANTICHEAT);

    // the graphics API: a game that imports another API and has no sign of D3D12 is refused
    r.api = ExeGraphics(exe);
    if (r.api.pe && !r.api.d3d12 && !r.api.others.empty()) {
        const bool d3d12Files = IsFile(Join(Join(exeDir, L"D3D12"), L"D3D12Core.dll")) || IsFile(Join(exeDir, L"amd_fidelityfx_dx12.dll")) ||
                                IsFile(Join(dir, L"amd_fidelityfx_dx12.dll"));
        if (!d3d12Files)
            add(Tool::GameApi, Decision::Refuse, L"Not a D3D12 game (" + JoinList(r.api.others) + L")", {BaseName(exe)}, r.api.evidence,
                L"WrongGPU works in Direct3D 12 games only; this game uses " + JoinList(r.api.others) + L".", EXIT_REFUSED);
    }

    // fingerprints of the DLLs next to the exe (ours left out)
    struct Seen { wstring rel; Fingerprint fp; };
    std::vector<Seen> seen;
    bool refw = false;  // REFramework's loader (gamerules.h): known by its own words, never taken for another tool's file
    for (auto n : kScanNames) {
        const wstring p = Join(exeDir, n);
        if (!IsFile(p) || isOurs(n)) continue;
        if (IEq(n, kReFrameworkDll) && IsReFrameworkDll(p)) { refw = true; continue; }
        seen.push_back({n, IdentifyFile(p)});
    }
    auto filesOf = [&](Tool t) { std::vector<wstring> v; for (auto& s : seen) if (s.fp.tool == t) v.push_back(s.rel); return v; };
    auto fpOf = [&](Tool t) -> const Fingerprint* { for (auto& s : seen) if (s.fp.tool == t) return &s.fp; return nullptr; };
    auto verOf = [&](Tool t) { const Fingerprint* f = fpOf(t); return f && !f->version.empty() ? L" " + f->version : wstring(); };

    // REFramework: it stays; a game that needs it for OptiScaler (RE Requiem) says so
    if (refw) {
        const GameRule* rule = GameRuleFor(exe);
        std::vector<wstring> files{kReFrameworkDll};
        if (IsDir(Join(exeDir, L"reframework"))) files.push_back(L"reframework");
        add(Tool::ReFramework, Decision::Inform, wstring(kReFrameworkName) + L" as " + kReFrameworkDll, files, L"text \"REFramework\"",
            rule && rule->needsReFramework ? L"Stays where it is: " + wstring(rule->name) + L" needs it before OptiScaler can run."
                                           : wstring(L"Stays where it is; WrongGPU works next to it."));
    }

    // our OptiScaler setup. 0.6.1: one finding per file: a proxy name another setup's file holds now is that file's finding
    // (listed below), not this one's
    if (ours && HostedRecord(*ours)) {
        const bool proxyOurs = !ours->proxy.empty() && (!IsFile(Join(exeDir, ours->proxy)) || isOurs(ours->proxy));
        add(Tool::OptiScaler, Decision::Managed,
            L"WrongGPU's OptiScaler setup" + (proxyOurs ? L" as " + ours->proxy : ours->proxy.empty() ? wstring() : L" (its " + ours->proxy + L" was replaced)"),
            proxyOurs ? std::vector<wstring>{ours->proxy} : std::vector<wstring>{}, L"install record", L"Installed by WrongGPU: updated and removed with it.");
    }

    // the player's OptiScaler: ask to switch (owner decision); two copies: refused until the player keeps one
    {
        std::vector<wstring> loaded, all;
        const Fingerprint* f = nullptr;
        for (auto& s : seen)
            if (s.fp.tool == Tool::OptiScaler) {
                all.push_back(s.rel);
                if (!IEq(s.rel, L"OptiScaler.dll")) { loaded.push_back(s.rel); if (!f) f = &s.fp; }
            }
        if (!f) f = fpOf(Tool::OptiScaler);
        for (auto m : {L"OptiScaler.ini", L"OptiScaler.log"}) if (IsFile(Join(exeDir, m)) && !isOurs(m)) all.push_back(m);
        if (!all.empty()) {
            const wstring ver = f && !f->version.empty() ? f->version : wstring();
            if (loaded.size() >= 2) {
                add(Tool::OptiScaler, Decision::Refuse, L"Two OptiScaler copies (" + JoinList(loaded) + L")", all, f ? f->evidence : wstring(),
                    L"This game loads OptiScaler twice. Keep one of them (remove the other yourself), then install WrongGPU.", EXIT_OTHER_MODS);
            } else {
                wstring msg = L"Install asks you to switch to WrongGPU's OptiScaler setup: your OptiScaler files are backed up and put back byte for byte when WrongGPU is uninstalled.";
                if (!loaded.empty() && IEq(loaded[0], L"nvngx.dll")) msg += L" (An older OptiScaler layout: it loads as nvngx.dll.)";
                else if (ver.empty() || VersionLess(ver, kOptiTested)) msg += L" (Version " + (ver.empty() ? wstring(L"unknown") : ver) + L": untested with WrongGPU.)";
                if (!loaded.empty())
                    for (auto n : kProxyOrder) if (IEq(loaded[0], n)) r.playersOptiProxy = n;
                add(Tool::OptiScaler, Decision::AskSwitch, L"Your OptiScaler" + (ver.empty() ? wstring() : L" " + ver) + (loaded.empty() ? wstring() : L" as " + loaded[0]),
                    all, f ? f->evidence : L"file " + all[0], msg, EXIT_OTHER_MODS);
            }
        }
    }

    // ReShade and Special K next to the exe: they stay, OptiScaler takes another name
    for (Tool t : {Tool::ReShade, Tool::SpecialK}) {
        std::vector<wstring> f = filesOf(t), extra;
        if (t == Tool::ReShade) {
            for (auto pat : {L"ReShade*.ini", L"*.addon64", L"*.addon"}) for (auto& x : ListFiles(exeDir, pat)) extra.push_back(BaseName(x));
        } else {
            for (auto pat : {L"SpecialK*.ini", L"dxgi.ini"}) for (auto& x : ListFiles(exeDir, pat)) if (filesOf(t).size() || IEq(BaseName(x), L"SpecialK.ini")) extra.push_back(BaseName(x));
        }
        if (!f.empty()) {
            std::vector<wstring> all = f; all.insert(all.end(), extra.begin(), extra.end());
            add(t, Decision::Chain, wstring(ToolName(t)) + verOf(t) + L" as " + f[0], all, fpOf(t)->evidence,
                L"Stays where it is; OptiScaler takes another free name (an untested combination). Its hooks give command lists their own tables, "
                L"which can make WrongGPU fall back to AMD FSR: the game page shows it after a run.");
        } else if (!extra.empty()) {
            add(t, Decision::Inform, wstring(ToolName(t)) + L" settings (no DLL next to the game)", extra, L"file " + extra[0], L"Nothing to do.");
        }
    }

    // DXVK / vkd3d-proton: native D3D12 is required
    for (Tool t : {Tool::Dxvk, Tool::Vkd3d}) {
        const auto f = filesOf(t);
        if (!f.empty())
            add(t, Decision::Refuse, wstring(ToolName(t)) + L" as " + f[0], f, fpOf(t)->evidence,
                L"WrongGPU needs Windows' own Direct3D 12; " + wstring(ToolName(t)) + L" translates the game's graphics to Vulkan. Not installed.", EXIT_REFUSED);
    }
    for (auto& s : seen)
        if (IEq(s.rel, L"d3d12.dll") && s.fp.tool == Tool::Unknown)
            add(Tool::Unknown, Decision::Warn, L"An unknown d3d12.dll", {s.rel}, s.fp.evidence,
                L"Not Windows' own and not recognised (another tool's?). Left alone; if the game fails with WrongGPU, check this file.");

    // Ultimate ASI Loader: WrongGPU's plugin gets a folder of its own
    if (const auto f = filesOf(Tool::AsiLoader); !f.empty()) {
        r.asiLoader = true;
        add(Tool::AsiLoader, Decision::Warn, L"Ultimate ASI Loader" + verOf(Tool::AsiLoader) + L" as " + f[0], f, fpOf(Tool::AsiLoader)->evidence,
            L"It loads every .asi file from the game folder, scripts\\ and plugins\\: WrongGPU's OptiScaler plugin goes to " + wstring(kAsiPluginOwnDir) +
                L"\\ instead (OptiScaler's [Plugins] Path), so it is not started twice.");
    }

    // 0.6.1: other setups' plugins in OptiScaler's plugin folders: in a Ray Reconstruction game WrongGPU's RR stays out unless
    // the player moves them aside (they may hook RR too); elsewhere they stay beside WrongGPU's plugin
    // (the plugin folders are the exe folder's; when the install folder is another one, the record that names their files is
    // the exe folder's own, if there is one: an earlier install there is not "another setup")
    Record exeDirRecord;
    const Record* pluginRecord = sameDir ? ours : LoadRecord(RecordPath(exeDir), exeDirRecord) && !exeDirRecord.legacy ? &exeDirRecord : nullptr;
    if (const auto kit = OurPluginsFromOtherPackages(exeDir, pluginRecord); !kit.empty())
        add(Tool::WrongGPU, Decision::Inform, L"WrongGPU RR (from another package)", kit, L"export " + FromUtf8(kRrContractExport) + L" in " + kit[0],
            L"WrongGPU's own Ray Reconstruction plugin, placed by another package. It stays where it is and is never moved aside.");
    r.foreignPlugins = ForeignPlugins(exeDir, pluginRecord);
    if (!r.foreignPlugins.empty()) {
        r.foreignPluginsBlockRr = DetectRr(exe).uses;
        add(Tool::OtherPlugin, r.foreignPluginsBlockRr ? Decision::Warn : Decision::Inform, L"Other OptiScaler plugins (" + JoinList(r.foreignPlugins) + L")",
            r.foreignPlugins, L"file " + r.foreignPlugins[0],
            r.foreignPluginsBlockRr ? L"They may hook Ray Reconstruction as WrongGPU's RR plugin does: WrongGPU's Ray Reconstruction stays out of this game unless "
                                      L"you choose to move them aside when you install (backed up, put back when WrongGPU is uninstalled)."
                                    : L"They stay; WrongGPU's plugin goes beside them.");
    }

    // CET, RED4ext, UE4SS: they keep their names (the picker never takes them)
    auto notMicrosoft = [&](const wchar_t* n) { for (auto& s : seen) if (IEq(s.rel, n)) return s.fp.tool != Tool::Microsoft; return false; };
    {
        std::vector<wstring> f = filesOf(Tool::Cet);
        if (IsDir(Join(Join(exeDir, L"plugins"), L"cyber_engine_tweaks"))) {
            if (f.empty() && notMicrosoft(L"version.dll")) f.push_back(L"version.dll");
            if (!f.empty()) f.push_back(L"plugins\\cyber_engine_tweaks");
        }
        if (!f.empty())
            add(Tool::Cet, Decision::Chain, L"Cyber Engine Tweaks", f, fpOf(Tool::Cet) ? fpOf(Tool::Cet)->evidence : L"folder plugins\\cyber_engine_tweaks",
                L"Stays where it is: its " + f[0] + L" is never taken as OptiScaler's name.");
    }
    {
        std::vector<wstring> f = filesOf(Tool::Red4ext);
        if (IsDir(Join(exeDir, L"red4ext")) && f.empty() && notMicrosoft(L"winmm.dll")) f.push_back(L"winmm.dll");
        if (!f.empty())
            add(Tool::Red4ext, Decision::Chain, L"RED4ext", f, fpOf(Tool::Red4ext) ? fpOf(Tool::Red4ext)->evidence : L"folder red4ext",
                L"Stays where it is: its " + f[0] + L" is never taken as OptiScaler's name.");
    }
    if (const auto f = filesOf(Tool::Ue4ss); !f.empty())
        add(Tool::Ue4ss, Decision::Chain, L"UE4SS", f, fpOf(Tool::Ue4ss)->evidence, L"Stays where it is: its " + f[0] + L" is never taken as OptiScaler's name.");

    // another HIP-based tool: a second HIP runtime in the game
    {
        std::vector<wstring> hip;
        wstring ev;
        if (IsFile(Join(exeDir, L"neuralhub.ini"))) { hip.push_back(L"neuralhub.ini"); ev = L"file neuralhub.ini"; }
        for (auto pat : {L"amdhip64*.dll", L"amd_comgr*.dll", L"rocm_kpack*.dll", L"hiprtc*.dll"})
            for (auto& x : ListFiles(exeDir, pat)) {
                const Fingerprint fp = IdentifyFile(x);
                if (fp.tool == Tool::HipRuntime || Lower(fp.company).find(L"advanced micro devices") != wstring::npos) {
                    hip.push_back(BaseName(x));
                    if (ev.empty()) ev = BaseName(x) + L": " + fp.evidence;
                }
            }
        if (!hip.empty()) {
            r.hipTool = true;
            add(Tool::HipRuntime, Decision::Warn, L"A HIP-based tool (" + hip[0] + L")", hip, ev,
                L"Its own HIP runtime may be loaded into the game next to WrongGPU's: keep WrongGPU on the D3D12 executor (the default). version.dll is never taken as OptiScaler's name here.");
        }
    }

    // Another upscaler mod (Tool::NrOnAmd; constants.h kOtherMods): refused, never touched, never offered for cleanup
    {
        std::vector<wstring> f = filesOf(Tool::NrOnAmd);
        for (const wstring& d : {exeDir, dir})
            for (auto& om : kOtherMods)  // its own files: the detection keys of constants.h kOtherMods (not OptiScaler's)
                for (auto m : om.files)
                    if (m && strcmp(om.signature, "optiscaler") && IsFile(Join(d, m)) && std::find(f.begin(), f.end(), wstring(m)) == f.end()) f.push_back(m);
        if (!f.empty())
            // what players see names neither the mod nor its files (owner 2026-11-04): the files are only the detection keys
            add(Tool::NrOnAmd, Decision::Refuse, L"Another upscaler mod", {}, L"its own settings or log file",
                L"WrongGPU does not install next to it and does not touch it.", EXIT_OTHER_MODS);
    }

    // nvngx.dll that is the vendor's own DLL: the DLSS DLL copied there by an older OptiScaler layout (a warning, never a refusal)
    for (auto& s : seen)
        if (IEq(s.rel, L"nvngx.dll") && s.fp.tool == Tool::VendorDll)
            add(Tool::VendorDll, Decision::Warn, L"Old OptiScaler layout (nvngx.dll = DLSS DLL copy)", {s.rel}, s.fp.evidence,
                L"An older OptiScaler setup copied the game's DLSS DLL as nvngx.dll. WrongGPU leaves it alone and never writes this name; "
                L"if you remove that OptiScaler, this nvngx.dll copy can go too.");
    // loose nvapi64.dll / nvngx.dll of another tool: a warning; WrongGPU never writes these names
    for (auto& s : seen)
        if ((IEq(s.rel, L"nvapi64.dll") || IEq(s.rel, L"nvngx.dll")) && s.fp.tool != Tool::OptiScaler && !(IEq(s.rel, L"nvngx.dll") && s.fp.tool == Tool::VendorDll))
            add(s.fp.tool == Tool::FakeNvapi ? Tool::FakeNvapi : s.fp.tool == Tool::VendorDll ? Tool::VendorDll : Tool::Unknown, Decision::Warn,
                L"A loose " + s.rel + (s.fp.tool == Tool::FakeNvapi ? L" (fakenvapi)" : L""), {s.rel}, s.fp.evidence,
                L"Another tool's file: WrongGPU leaves it alone and never writes this name.");

    // an older WrongGPU / DLSS 4.5 AMD install: migrated with the player's OK
    {
        const Conflicts c = FindConflicts(dir, exeDir);
        if (c.oldFound())
            add(Tool::WrongGPULegacy, Decision::Migrate, L"An older WrongGPU (DLSS 4.5 AMD) install",
                c.legacyRecordDir.empty() ? c.legacyFiles : std::vector<wstring>{wstring(kLegacyStateDir) + L"\\install.manifest"},
                c.legacyRecordDir.empty() ? L"loose files" : L"its install record", L"Install removes it first with your OK (its original files are put back byte for byte).",
                EXIT_OLD_INSTALL);
    }

    // overlays and injectors that run now
    std::set<wstring> overlays;
    for (auto& p : RunningProcesses()) {
        Decision d = Decision::Inform;
        const wchar_t* msg = nullptr;
        const wchar_t* name = OverlayFor(p, d, &msg);
        if (!name || !overlays.insert(name).second) continue;
        add(IEq(p, L"SKIF.exe") ? Tool::SpecialK : Tool::Overlay, d, name, {p}, L"running process " + p, msg);
    }

    // unknown DLLs under proxy names: shown; the picker leaves them alone
    for (auto& s : seen)
        if (s.fp.tool == Tool::Unknown)
            for (auto n : kProxyOrder)
                if (IEq(s.rel, n)) add(Tool::Unknown, Decision::Inform, L"An unknown " + s.rel, {s.rel}, s.fp.evidence, L"Not recognised: left alone and never taken as OptiScaler's name.");

    const bool untested = filesOf(Tool::ReShade).size() + filesOf(Tool::SpecialK).size() > 0;
    r.proxy = PickProxy(exeDir, ours, r.playersOptiProxy, r.hipTool, untested);
    LogLine(ctx, L"coexistence " + exe + L": " + std::to_wstring(r.findings.size()) + L" finding(s); OptiScaler's name: " +
                     (r.proxy.name.empty() ? wstring(L"none free") : r.proxy.name) + L" (" + r.proxy.reason + L")");
    return r;
}

}  // namespace wg
