// SPDX-License-Identifier: MIT
// WrongGPU launcher core: Ray Reconstruction in the games that use it (rr.h). Written for this project.
#include "rr.h"

#include <filesystem>
#include <mutex>

#include "gamerules.h"
#include "launcher.h"
#include "optiedition.h"
#include "optiscaler.h"
#include "payload.h"
#include "pe.h"
#include "settings.h"

namespace wg {

// ---- which games ---------------------------------------------------------------------------------------------------
// The files a game that uses DLSS Ray Reconstruction ships: the RR library (the RR model file) and the game interface's RR
// plugin. The plugin shows that the game takes RR through the game interface; the library alone, that it creates RR itself
// (RrUse::direct).
static const wchar_t* const kRrLibrary = L"nvngx_dlssd.dll";      // DLSS Ray Reconstruction
static const wchar_t* const kRrIfacePlugin = L"sl.dlss_d.dll";    // the game interface's DLSS Ray Reconstruction plugin
// rc6: Hogwarts Legacy since its November 2026 update ships a game-interface integration (its DLSS Frame Generation plugin,
// below, is part of it; the builds before had none): its Ray Reconstruction option is greyed out there and the game never asks
// for RR (attended rc6 runs), so WrongGPU's RR is not offered in it (as gamerules.h rrNotOffered). Older builds keep RR.
static const wchar_t* const kIfaceFrameGen = L"sl.dlss_g.dll";    // the game interface's DLSS Frame Generation plugin
static const wchar_t* const kRrNotOfferedHogwartsUpdate =
    L"Since the game's November 2026 update, Hogwarts Legacy doesn't offer Ray Reconstruction with WrongGPU yet (its option is greyed "
    L"out). Model M works. We're working on it.";
// Games that use Ray Reconstruction and may keep its files where the walk does not look (their exe names), and whether they
// create it themselves (direct) or through the game interface. The table decides `direct` for them over the files found
// (Cyberpunk 2077 and RE Requiem take RR through the game interface whatever their folder holds; Cyberpunk's own files show
// that it uses RR, so its row only fixes `direct`).
static const struct { const wchar_t* exe; bool direct; bool uses; } kRrGames[] = {
    {L"HogwartsLegacy.exe", true, true}, {L"re9.exe", false, true}, {L"Cyberpunk2077.exe", false, false}};

RrUse DetectRrUncached(const wstring& exe) {
    RrUse r;
    if (exe.empty() || !IsFile(exe)) return r;
    if (const GameRule* rule = GameRuleFor(exe))  // rc5: a game whose menu has no RR option (gamerules.h rrNotOffered)
        if (rule->rrNotOffered) { r.notOffered = true; r.why = rule->rrNotOffered; return r; }
    const wstring exeDir = Parent(exe), root = GameRoot(exeDir);
    auto known = [&exe](RrUse& u) {
        for (auto& g : kRrGames)
            if (IEq(BaseName(exe), g.exe)) {
                if (!u.uses && g.uses) {
                    u.why = L"known game";
                    u.uses = true;
                }
                u.direct = u.uses && g.direct;
            }
        return u;
    };
    if (IsFile(Join(exeDir, kRrIfacePlugin))) { r.uses = true; r.why = kRrIfacePlugin; return known(r); }
    // the RR library found: the walk goes on for the game interface's RR plugin (which decides "not direct"); `library` is
    // the first library file found (the reason shown)
    wstring library;
    if (IsFile(Join(exeDir, kRrLibrary))) library = kRrLibrary;
    // the game's folder: a bounded walk (the store's own folders hold no game files; a big game is not read in full). Not
    // from a drive's root (an exe added from G:\ would walk the whole drive and find other games' RR files)
    std::vector<std::pair<wstring, int>> stack;
    wstring rootFull = FullPath(root);
    while (!rootFull.empty() && rootFull.back() == L'\\') rootFull.pop_back();
    if (rootFull.size() > 2) stack.push_back({root, 0});  // "G:" is a drive's root
    size_t visited = 0;
    bool frameGen = false;  // rc6: the game interface's frame-generation plugin seen (kIfaceFrameGen)
    while (!stack.empty() && visited < 40000) {
        auto [dir, depth] = stack.back(); stack.pop_back();
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileExW(Join(dir, L"*").c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            ++visited;
            const wstring n = fd.cFileName;
            if (n == L"." || n == L"..") continue;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                if (depth < 9 && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && !IEq(n, kStateDirName) && !IEq(n, kLegacyStateDir))
                    stack.push_back({Join(dir, n), depth + 1});
            } else if (IEq(n, kIfaceFrameGen)) {
                frameGen = true;
            } else if (IEq(n, kRrIfacePlugin) || IEq(n, kRrLibrary)) {
                const wstring full = Join(dir, n);
                const wstring rel = full.size() > root.size() + 1 ? full.substr(root.size() + 1) : n;
                if (IEq(n, kRrIfacePlugin)) {
                    r.uses = true;
                    r.why = rel;
                    FindClose(h);
                    return known(r);
                }
                if (library.empty()) library = rel;
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    if (frameGen && IEq(BaseName(exe), L"HogwartsLegacy.exe")) {  // rc6: the updated Hogwarts Legacy (kIfaceFrameGen)
        RrUse n;
        n.notOffered = true;
        n.why = kRrNotOfferedHogwartsUpdate;
        return n;
    }
    if (!library.empty()) {  // the RR library without the game interface's RR plugin: the game creates RR itself
        r.uses = true;
        r.direct = true;
        r.why = library;
    }
    return known(r);
}

bool RecordHasEdition(const Record& m) {
    const EditionInfo e = PinnedEdition();
    if (e.dllSha256.empty()) return false;
    for (auto& x : m.e)
        if (x.action == L"copy" && !m.proxy.empty() && IEq(x.rel, m.proxy)) return IEq(x.inst, e.dllSha256);
    for (auto& x : m.e)  // a record without its proxy line: OptiScaler's DLL under one of the proxy names
        for (auto px : kOptiProxies)
            if (x.action == L"copy" && IEq(x.rel, px) && IEq(x.inst, e.dllSha256)) return true;
    return false;
}

wstring RrDirectNoEditionText(const wstring& game) {
    return L"This OptiScaler is no OptiScaler edition with plugin interface 4: WrongGPU's Ray Reconstruction isn't offered in " + game +
           L", and the game keeps its own ray tracing.";
}

static std::mutex g_rrMutex;
static std::map<wstring, RrUse> g_rrSeen;
RrUse DetectRr(const wstring& exe) {
    const wstring key = Lower(FullPath(exe));
    {
        std::lock_guard<std::mutex> lock(g_rrMutex);
        auto it = g_rrSeen.find(key);
        if (it != g_rrSeen.end()) return it->second;
    }
    const RrUse r = DetectRrUncached(exe);
    std::lock_guard<std::mutex> lock(g_rrMutex);
    g_rrSeen[key] = r;
    return r;
}
void ForgetRrDetection() {
    std::lock_guard<std::mutex> lock(g_rrMutex);
    g_rrSeen.clear();
}

// ---- this build's RR files -------------------------------------------------------------------------------------------
bool RrPayloadAvailable(Context& ctx) {
    if (!ctx.data || !ctx.data->rrCount) return false;
    bool plugin = false;
    for (size_t i = 0; i < ctx.data->rrCount; ++i) plugin = plugin || IEq(ctx.data->rrFiles[i].rel, kRrPayloadPlugin);
    return plugin;  // the RR runtime: part of the payload's wronggpu_runtime.dll
}

// plugins\WrongGPU-RR.asi, or plugins\<major>.<minor>\WrongGPU-RR.asi
wstring RrPluginVersion(const wstring& rel) {
    if (!IEq(BaseName(rel), kRrPluginName)) return L"";
    const wstring dir = Parent(rel), base = BaseName(dir);
    if (!IEq(Parent(dir), L"plugins")) return L"";
    const size_t dot = base.find(L'.');
    if (dot == 0 || dot == wstring::npos || dot + 1 >= base.size() || base.find_first_not_of(L"0123456789.") != wstring::npos ||
        base.find(L'.', dot + 1) != wstring::npos)
        return L"";
    return base;
}
bool IsRrPayloadPlugin(const wstring& rel) { return IEq(rel, kRrPayloadPlugin) || !RrPluginVersion(rel).empty(); }

wstring GameInterfaceVersion(Context& ctx, const wstring& exe, wstring* where) {
    if (where) where->clear();
    if (!ctx.data || !ctx.data->rrInterface || !*ctx.data->rrInterface || exe.empty()) return L"";
    const wstring name = ctx.data->rrInterface, exeDir = Parent(exe);
    wstring file = Join(exeDir, name);
    if (!IsFile(file)) {
        // not from a drive's root (an exe added from G:\ would walk the whole drive), as DetectRr
        const wstring root = GameRoot(exeDir);
        wstring full = FullPath(root);
        while (!full.empty() && full.back() == L'\\') full.pop_back();
        const std::vector<wstring> found = full.size() > 2 ? FindFilesNamed(root, name.c_str(), 40000, 9) : std::vector<wstring>{};
        if (found.empty()) return L"";
        file = found[0];
    }
    const PeVersion v = FileVersionInfo(file);
    if (v.fileVersion.empty()) return L"";
    if (where) *where = file;
    return v.fileVersion;
}

wstring PickRrPlugin(const std::vector<wstring>& rels, const wstring& version) {
    // "2.8.0.0" -> "2.8" (the plugin checks major.minor)
    wstring mm;
    const size_t a = version.find(L'.');
    if (a != wstring::npos) {
        const size_t b = version.find(L'.', a + 1);
        mm = version.substr(0, b);
    }
    wstring general;
    for (auto& r : rels) {
        if (IEq(r, kRrPayloadPlugin)) general = r;
        else if (!mm.empty() && RrPluginVersion(r) == mm) return r;
    }
    return general;
}

const PayloadFile* RrPluginFor(Context& ctx, const wstring& version, wstring* forVersion) {
    if (forVersion) forVersion->clear();
    if (!ctx.data || !ctx.data->rrCount) return nullptr;
    std::vector<wstring> rels;
    for (size_t i = 0; i < ctx.data->rrCount; ++i) rels.push_back(ctx.data->rrFiles[i].rel);
    const wstring pick = PickRrPlugin(rels, version);
    for (size_t i = 0; i < ctx.data->rrCount; ++i)
        if (!pick.empty() && IEq(ctx.data->rrFiles[i].rel, pick)) {
            if (forVersion) *forVersion = RrPluginVersion(pick);
            return &ctx.data->rrFiles[i];
        }
    return nullptr;
}

// ---- the RR data (preset F) ----------------------------------------------------------------------------------------
bool LoadFMap(Context& ctx) {
    static std::mutex loading;  // the window reads the data's state on a worker while an operation may convert it
    std::lock_guard<std::mutex> lock(loading);
    if (ctx.fMap.map) return true;
    if (!ctx.data || !ctx.data->fMapId) { ctx.fMapError = L"this build has no Ray Reconstruction converter data"; return false; }
    DWORD sz = 0;
    const void* d = ResourceData(ctx.data->fMapId, sz);
    if (!d || Sha256Bytes(d, sz) != ctx.data->fMapSha256) { ctx.fMapError = L"the Ray Reconstruction converter data is missing or damaged"; return false; }
    const wgc::Result r = wgc::load_map_bytes(string(static_cast<const char*>(d), sz), ctx.fMap,
                                               [&ctx](const string& s) { LogLine(ctx, L"converter (RR): " + FromUtf8(s)); });
    if (r.code != wgc::kOk) { ctx.fMapError = FromUtf8(r.message); return false; }
    return true;
}
wstring RrDataDir() { return Join(ModelBaseDir(), L"F"); }
wstring RrDataFile() { return Join(RrDataDir(), FromUtf8(wgc::kModelFileName)); }

void RecoverRrData(Context& ctx) {
    const wstring old = RrDataDir() + L".old";
    if (IsDir(RrDataDir()) || !IsDir(old)) return;
    if (MoveFileExW(old.c_str(), RrDataDir().c_str(), MOVEFILE_WRITE_THROUGH)) LogLine(ctx, L"RR data: put back from " + old + L" (an earlier swap did not finish)");
}

bool RrDataValid(Context& ctx, wstring* why) {
    if (!LoadFMap(ctx)) { if (why) *why = ctx.fMapError; return false; }
    if (!IsFile(RrDataFile())) { if (why) *why = L"no converted Ray Reconstruction data on this PC"; return false; }
    string w;
    const bool ok = wgc::verify_model_file(*ctx.fMap.map, std::filesystem::path(RrDataFile()), w);
    if (why) *why = FromUtf8(w);
    return ok;
}

wstring RrDataSource() {
    string src;
    if (!ReadAll(Join(RrDataDir(), L"model-source.txt"), src)) return L"";
    const wstring s = FromUtf8(src);
    const size_t p = s.find(L"source\t");
    if (p == wstring::npos) return L"";
    const size_t e = s.find_first_of(L"\r\n", p);
    return s.substr(p + 7, e == wstring::npos ? wstring::npos : e - p - 7);
}

// RR data converted from a listed file and nothing allowed found: what the player does (name the listed source,
// ask for an allowed file from the player's own game)
static const wchar_t* const kRrChooseOwnFile =
    L". To use Ray Reconstruction, choose the Ray Reconstruction model file (nvngx_dlssd.dll) that came with one of your own games "
    L"(Convert on the game's page, or Settings > Ray Reconstruction data), or install or update such a game";
RrDataState FindRrData(Context& ctx, const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries, bool fresh) {
    RrDataState m;
    if (!LoadFMap(ctx)) {
        m.available = !(ctx.data && !ctx.data->fMapId);
        m.error = ctx.fMapError;
        LogLine(ctx, L"RR data: " + m.error);
        return m;
    }
    RecoverRrData(ctx);  // a swap that did not finish: its data back first (it may be valid)
    // The ignore lists (convert_ignore.h), read first: what may be converted, and whether the data on this PC may be reused.
    // A list that is there but can't be read: nothing is converted or reused (fail closed).
    const IgnoreLists ign = LoadConvertIgnore(ctx, L"RR data", &m.ignored);
    if (!ign.unreadable.empty()) { m.error = ign.unreadable; LogLine(ctx, L"RR data: " + m.error); return m; }
    wstring why;
    const bool existing = !fresh && userDll.empty() && RrDataValid(ctx, &why);
    // As the model data (model.cpp FindModel): while a list has entries, data converted from a listed file (path or SHA-256 in
    // its model-source.txt) is not reused, and data without a recorded SHA-256 (an earlier version) is converted again when an
    // allowed file is found, else kept with a note.
    const wstring source = existing ? RrDataSource() : wstring();
    wstring distrustWhy;
    if (existing && !ign.empty()) {
        const wstring sha = SourceField(Join(RrDataDir(), L"model-source.txt"), L"sha256");
        const wstring listed = RecordedSourceReason(ign, source, sha);
        if (!listed.empty()) {
            m.trust = RrDataState::LISTED;
            distrustWhy = L"it was converted from " + (source.empty() ? wstring(L"a file") : source) + L", which is " + listed;
        } else if (sha.empty()) {
            m.trust = RrDataState::UNKNOWN;
            distrustWhy = L"the SHA-256 of the file it was converted from is not on record (an earlier version converted it), so the ignore lists can't check it";
        }
    }
    auto keep = [&] {
        m.ok = m.reused = true;
        m.origin = L"converted earlier on this PC";
        m.dll = source;
    };
    if (existing && m.trust == RrDataState::TRUSTED) {
        keep();
        LogLine(ctx, L"RR data: reusing " + RrDataFile());
        return m;
    }
    if (existing) why = RrDataFile() + L" is valid, but " + distrustWhy;
    if (why.empty()) why = fresh ? L"converted again" : L"a file was chosen";
    LogLine(ctx, L"RR data: " + why + L"; looking for a DLSS Ray Reconstruction model file (nvngx_dlssd.dll)");
    auto keepUnknown = [&] {
        keep();
        m.notes.push_back(L"The Ray Reconstruction data on this PC was converted by an earlier version of WrongGPU, which did not record the SHA-256 of "
                          L"the file it came from, so the ignore lists can't check it. No other Ray Reconstruction model file was found to convert it "
                          L"again, so it stays in use.");
        LogLine(ctx, L"RR data: keeping " + RrDataFile() + L" (its source can't be checked against the ignore lists; nothing else to convert)");
    };
    std::vector<std::pair<wstring, wstring>> cands;
    std::set<wstring> seen;
    auto add = [&](const wstring& p, const wstring& origin) { if (seen.insert(Lower(FullPath(p))).second) cands.push_back({FullPath(p), origin}); };
    if (!userDll.empty()) add(userDll, L"the file you chose");
    if (t && !t->root.empty()) for (auto& f : FindFilesNamed(t->root, L"nvngx_dlssd.dll")) add(f, L"this game");  // DLSS RR
    for (auto& g : libraries) for (auto& f : FindFilesNamed(g.root, L"nvngx_dlssd.dll", 30000)) add(f, L"another game (" + g.name + L")");  // DLSS RR
    // A file on a list is left out before anything is tried; the order of the others stays. The rule is absolute: a
    // file the player chose is left out too, and the player is told which list and line (or --ignore=) matched.
    wstring chosenSkipped;
    {
        std::vector<wstring> paths;
        for (auto& c : cands) paths.push_back(c.first);
        const std::vector<bool> skip = IgnoredMask(ctx, ign, paths, L"RR data", m.ignored);
        std::vector<std::pair<wstring, wstring>> kept;
        for (size_t i = 0; i < cands.size(); ++i) {
            if (!skip[i]) kept.push_back(cands[i]);
            else if (i == 0 && !userDll.empty()) {
                const wstring& s = m.ignored.skipped.front();
                chosenSkipped = L"the file you chose (" + cands[0].first + L") is not converted: it is " + s.substr(std::min(s.size(), cands[0].first.size() + 2)) +
                                L". If it was listed by mistake, remove that line (or that --ignore argument)";
            }
        }
        cands.swap(kept);
    }
    const size_t skipped = m.ignored.skipped.size();
    // the player's line naming the first file skipped, its list and line (the error is a fragment: no full stop at its end)
    wstring example = IgnoredExample(ctx, m.ignored);
    if (!example.empty() && example.back() == L'.') example.pop_back();
    if (cands.empty() && skipped) {  // nothing left to try
        if (existing && m.trust == RrDataState::UNKNOWN) { keepUnknown(); return m; }
        m.error = !chosenSkipped.empty() && skipped == 1
                      ? chosenSkipped
                      : L"every Ray Reconstruction model file found (" + std::to_wstring(skipped) + L") is on an ignore list, so none was converted "
                        L"(WrongGPU converts only the files that came with your own games). " + example;
        if (existing) m.error = L"the Ray Reconstruction data on this PC is not used: " + distrustWhy + L"; and " + m.error + kRrChooseOwnFile;
        LogLine(ctx, L"RR data: " + m.error);
        return m;
    }
    wgc::SearchOptions opt;
    for (auto& c : cands) {
        auto conv = std::make_shared<wgc::Converted>();
        const wgc::Result r = wgc::convert_file(*ctx.fMap.map, std::filesystem::path(c.first), opt, *conv,
                                                [&ctx](const string& s) { LogLine(ctx, L"converter (RR): " + FromUtf8(s)); });
        if (r.code == wgc::kOk) {
            m.ok = true; m.dll = c.first; m.origin = c.second; m.version = FromUtf8(conv->version); m.converted = conv;
            m.dllSha256 = Lower(Sha256File(c.first));
            LogLine(ctx, L"RR data: using " + c.first + L" (" + c.second + L")" + (existing ? L"; it replaces the data on this PC (" + distrustWhy + L")" : wstring()));
            return m;
        }
        m.rejections.push_back(c.first + L": " + FromUtf8(r.message));
        LogLine(ctx, L"RR data: rejected " + c.first + L": " + FromUtf8(r.message));
        if (wgc::is_installer_error(r.code)) {
            if (existing && m.trust == RrDataState::UNKNOWN) { keepUnknown(); return m; }
            m.error = FromUtf8(r.message);
            return m;
        }
    }
    if (existing && m.trust == RrDataState::UNKNOWN) { keepUnknown(); return m; }
    if (!userDll.empty() && cands.size() == 1 && !m.rejections.empty())  // only the chosen file was tried: its own reason
        m.error = L"the file you chose can't be converted by this version: " + m.rejections[0];
    else
        m.error = cands.empty() ? L"no Ray Reconstruction model file was found in this game or your other games"
                                : L"none of the " + std::to_wstring(cands.size()) + L" Ray Reconstruction model file(s) found can be converted by this version";
    if (skipped) m.error += L" (" + std::to_wstring(skipped) + L" file(s) on an ignore list were not tried). " + example;
    if (!chosenSkipped.empty()) m.error += L"; " + chosenSkipped;
    if (existing) m.error = L"the Ray Reconstruction data on this PC is not used: " + distrustWhy + L"; and " + m.error + kRrChooseOwnFile;
    return m;
}

bool WriteRrData(Context& ctx, const RrDataState& m, wstring* why) {
    if (!m.ok) { if (why) *why = m.error; return false; }
    if (m.reused) return RrDataValid(ctx, why);
    RecoverRrData(ctx);
    const wstring side = RrDataDir() + L".new";
    if (IsDir(side)) RemoveTree(side);
    EnsureDir(ModelBaseDir());
    const wgc::Result r = wgc::write_converted(*ctx.fMap.map, *m.converted, std::filesystem::path(side), false,
                                               [&ctx](const string& s) { LogLine(ctx, L"converter (RR): " + FromUtf8(s)); });
    const wstring sideFile = Join(side, FromUtf8(wgc::kModelFileName));
    string w;
    if (r.code != wgc::kOk || !wgc::verify_model_file(*ctx.fMap.map, std::filesystem::path(sideFile), w)) {
        if (why) *why = r.code != wgc::kOk ? FromUtf8(r.message) : L"the converted Ray Reconstruction data failed its check (" + FromUtf8(w) + L")";
        // the converter names no cause: a full disk is the one the player can fix, so it is said when it is the case
        ULARGE_INTEGER freeBytes{};
        if (why && GetDiskFreeSpaceExW(ModelBaseDir().c_str(), &freeBytes, nullptr, nullptr) && freeBytes.QuadPart < 256ull * 1024 * 1024)
            *why += L" (the disk is full: " + std::to_wstring(freeBytes.QuadPart / (1024 * 1024)) + L" MB free)";
        RemoveTree(side);
        return false;
    }
    const wstring text = L"source\t" + m.dll + L"\r\nversion\t" + m.version + L"\r\norigin\t" + m.origin + L"\r\nconverted\t" + NowStamp(false) +
                         L"\r\nmap\t" + FromUtf8(wgc::to_hex(ctx.fMap.map->file_sha256)) + L"\r\n" +
                         // the source file's SHA-256: a later reuse is checked against the ignore lists with it (FindRrData)
                         (m.dllSha256.empty() ? wstring() : L"sha256\t" + m.dllSha256 + L"\r\n");
    const string info = ToUtf8(text);
    WriteFileAtomic(Join(side, L"model-source.txt"), info.data(), info.size());
    // the folder in use moves aside first (a running game may hold its file: then nothing changes). A reader that holds it
    // for a moment (the window's health read hashes it after every operation) is waited for, a little.
    const wstring old = RrDataDir() + L".old";
    if (IsDir(old)) RemoveTree(old);
    DWORD moveErr = 0;
    for (int i = 0; i < 20 && IsDir(RrDataDir()) && !MoveFileExW(RrDataDir().c_str(), old.c_str(), MOVEFILE_WRITE_THROUGH); ++i) {
        moveErr = GetLastError();
        Sleep(150);
    }
    if (IsDir(RrDataDir())) {
        if (why) *why = L"the Ray Reconstruction data in use could not be moved (" + ErrText(moveErr) + L"; a game that uses Ray Reconstruction may be running); the data in use stays";
        RemoveTree(side);
        return false;
    }
    if (!MoveFileExW(side.c_str(), RrDataDir().c_str(), MOVEFILE_WRITE_THROUGH)) {
        const DWORD e = GetLastError();
        if (IsDir(old)) MoveFileExW(old.c_str(), RrDataDir().c_str(), MOVEFILE_WRITE_THROUGH);
        if (why) *why = L"the converted Ray Reconstruction data could not be put in place (" + ErrText(e) + L")";
        return false;
    }
    if (IsDir(old)) RemoveTree(old);
    LogLine(ctx, L"RR data: converted from " + m.dll + L" into " + RrDataFile());
    return RrDataValid(ctx, why);
}

bool ConvertRrData(Context& ctx, const wstring& dll, const std::vector<StoreGame>& libraries, wstring* why, std::vector<RrSettingsFailure>* notWritten,
                   IgnoreReport* ignored) {
    // a chosen file is the only one tried: when it does not convert, that is the answer (never another game's file instead)
    std::vector<StoreGame> where;
    if (dll.empty()) {
        for (auto& d : ManagedGames()) {  // the launcher's games first (their own RR model files), then the libraries
            Record m;
            if (LoadRecord(RecordPath(d), m) && !m.legacy) where.push_back({GameDisplayName(m.exe), GameRoot(m.exe.empty() ? d : Parent(m.exe)), L"", L"Managed"});
        }
        for (auto& g : libraries) where.push_back(g);
    }
    const RrDataState st = FindRrData(ctx, nullptr, dll, where, true);
    if (ignored) *ignored = st.ignored;
    if (!st.ok) { if (why) *why = st.error; return false; }
    if (!WriteRrData(ctx, st, why)) return false;
    const std::vector<RrSettingsFailure> failed = RefreshAllRrSettings(ctx);
    if (notWritten) *notWritten = failed;
    return true;
}

wstring RunningRrGame() {
    for (auto& d : ManagedGames()) {
        Record m;
        if (LoadRecord(RecordPath(d), m) && !m.legacy && RecordHasRr(m) && !m.exe.empty() && GameRunning(m.exe)) return GameDisplayName(m.exe);
    }
    return L"";
}

// ---- settings ------------------------------------------------------------------------------------------------------
static string TrimA(string t) {
    while (!t.empty() && (t.back() == '\r' || t.back() == '\n' || t.back() == ' ' || t.back() == '\t')) t.pop_back();
    size_t i = 0;
    while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
    return t.substr(i);
}
static string LowerA(string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); return s; }

RrFileState ParseRrSettings(const string& text) {
    RrFileState s;
    s.present = true;
    string sec;
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find('\n', a);
        if (e == string::npos) e = text.size();
        const string t = TrimA(text.substr(a, e - a));
        a = e + 1;
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;
        if (t[0] == '[') { sec = LowerA(t); continue; }
        if (sec != "[wronggpu]") continue;
        const size_t eq = t.find('=');
        if (eq == string::npos) continue;
        const string k = LowerA(TrimA(t.substr(0, eq))), v = LowerA(TrimA(t.substr(eq + 1)));
        if (k == "rr_enabled") s.enabled = !(v == "0" || v == "off" || v == "false");
        else if (k == "rr_mode" && (v == "exact" || v == "accurate" || v == "fast")) s.mode = v;
    }
    return s;
}
RrFileState ReadRrSettings(const wstring& dir) {
    string text;
    if (!ReadAll(Join(dir, kRrSettings), text)) return RrFileState();
    return ParseRrSettings(text);
}

string RenderRrSettings(const Profile& p, bool dataReady, const string& in) {
    const string mode = p.get("rr_mode");
    const bool on = p.get("rr_enabled") != "off" && dataReady;
    string text = in;
    if (text.empty())
        text = "# WrongGPU Ray Reconstruction settings (written by the WrongGPU launcher; OptiScaler's menu changes rr_mode too)\r\n[wronggpu]\r\n";
    text = IniSet(text, "wronggpu", "rr_enabled", on ? "1" : "0");
    text = IniSet(text, "wronggpu", "rr_mode", mode.empty() ? "fast" : mode);  // always written (rc5: Fast by default)
    text = IniSet(text, "wronggpu", "rr_exact_allowed", mode == "exact" ? "1" : "0");
    return text;
}

string RrIniText(const wstring& dataFile) {
    return "# WrongGPU Ray Reconstruction (written by the WrongGPU launcher; the RR plugin adds its own keys)\r\nrr_model=" + ToUtf8(dataFile) + "\r\n";
}

bool FoldRrFromGame(const wstring& dir, Profile& p) {
    const RrFileState s = ReadRrSettings(dir);
    if (!s.present || s.mode.empty() || s.mode == p.get("rr_mode")) return false;
    p.v["rr_mode"] = s.mode;
    return true;
}

int RefreshRrSettings(Context& ctx, const wstring& dir, bool dataReady, wstring* why) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy || !RecordHasRr(m)) return 0;
    Profile p = GameProfile(dir);
    string in;
    ReadAll(Join(dir, kRrSettings), in);
    const string out = RenderRrSettings(p, dataReady, in);
    if (out == in) return 0;
    try {  // the file and the record that follows it (a protected folder refuses both: the caller elevates)
        WriteFileAtomic(Join(dir, kRrSettings), out.data(), out.size());
        const wstring h = Sha256Bytes(out.data(), out.size());
        for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, kRrSettings)) e.inst = h;
        SaveRecord(RecordPath(dir), m);
        WriteOwnedFiles(dir, m);
    } catch (const SetupError& e) {
        LogLine(ctx, L"RR settings of " + dir + L" not written: " + e.msg);
        if (why) *why = e.msg;
        return -1;
    }
    LogLine(ctx, L"RR settings of " + dir + L" written for the Ray Reconstruction data's state");
    return 1;
}

std::vector<RrSettingsFailure> RefreshAllRrSettings(Context& ctx, const wstring& except) {
    std::vector<RrSettingsFailure> failed;
    std::vector<wstring> dirs;
    for (auto& d : ManagedGames()) {
        Record m;
        if (!except.empty() && IEq(FullPath(d), FullPath(except))) continue;
        if (LoadRecord(RecordPath(d), m) && !m.legacy && RecordHasRr(m)) dirs.push_back(d);
    }
    if (dirs.empty()) return failed;
    const bool ready = RrDataValid(ctx);  // once (it reads and hashes the whole file), not per game
    for (auto& d : dirs) {
        wstring why;
        if (RefreshRrSettings(ctx, d, ready, &why) < 0) failed.push_back({d, why});
    }
    return failed;
}

}  // namespace wg
