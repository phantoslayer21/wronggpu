// SPDX-License-Identifier: MIT
// WrongGPU launcher core: leftover WrongGPU files without an install record (leftovers.h). Written for this project.
#include "leftovers.h"

#include "coexist.h"
#include "detect.h"
#include "fingerprint.h"
#include "known_releases.h"
#include "optiscaler.h"

namespace wg {

namespace {
struct Known { wstring version, name; };
// hash -> release; sizes first (most files of a game folder are not checked further)
struct KnownSet {
    std::map<wstring, Known> byHash;
    std::set<unsigned long long> sizes;
};
KnownSet KnownFiles(Context& ctx) {
    KnownSet k;
    for (auto& f : kKnownFiles) { k.byHash.emplace(Lower(f.sha256), Known{f.version, f.name}); k.sizes.insert(f.size); }
    if (ctx.data)
        for (auto& f : *ctx.data) {
            const wstring rel = f.rel;
            if (!EndsWithI(rel, L".dll") && !EndsWithI(rel, L".asi") && !StartsWithI(BaseName(rel), L"wronggpu-")) continue;
            k.byHash.emplace(Lower(f.sha256), Known{ctx.data->version, BaseName(rel)});
            k.sizes.insert(f.size);
        }
    return k;
}
}  // namespace

// The files an older direct install wrote beside its DLLs in an engine plugin folder (removed only beside a DLL of ours).
static const wchar_t* const kDirectCompanions[] = {L"wronggpu.ini", L"wronggpu-fsr.log", L"wronggpu-fsr.log.old",
                                                    L"wronggpu-license.txt", L"wronggpu-notices.txt"};

bool ProvablyOurs(Context& ctx, const wstring& path) {
    const Fingerprint f = IdentifyFile(path);
    if (f.tool == Tool::WrongGPU && f.evidence.rfind(L"version resource", 0) == 0) return true;
    const KnownSet k = KnownFiles(ctx);
    return k.sizes.count(FileSize(path)) && k.byHash.count(Lower(Sha256File(path)));
}

// The "signedbin" folders of an Unreal Engine game's plugin folders: under <ancestor>\Engine\Plugins (the ancestors of
// `dir`, up to four levels up) and under <ancestor>\Plugins of the ancestors above `dir` (a project's plugins; `dir`'s own
// plugins\ folder is OptiScaler's and scanned as part of the folder). Bounded: at most 10 levels deep and 4000 folders
// read per game, junctions not followed (ListDirs). Each with the folder its shown path starts from.
static std::vector<std::pair<wstring, wstring>> EngineSlotDirs(const wstring& dir) {
    std::vector<std::pair<wstring, wstring>> out;  // (signedbin folder, the folder its shown path is relative to)
    std::set<wstring> seen;
    size_t budget = 4000;
    wstring cur = dir;
    for (int up = 0; up <= 4 && cur.find(L'\\') != wstring::npos; ++up) {
        std::vector<std::pair<wstring, wstring>> roots;  // (search root, shown-from folder)
        if (IsDir(Join(cur, L"Engine\\Plugins"))) roots.push_back({Join(cur, L"Engine\\Plugins"), cur});
        if (up > 0 && IsDir(Join(cur, L"Plugins"))) roots.push_back({Join(cur, L"Plugins"), Parent(cur)});
        for (auto& r : roots) {
            std::vector<std::pair<wstring, int>> stack{{r.first, 0}};
            while (!stack.empty() && budget) {
                auto [d, depth] = stack.back();
                stack.pop_back();
                --budget;
                if (IEq(BaseName(d), L"signedbin")) {
                    if (seen.insert(Lower(d)).second) out.push_back({d, r.second});
                    continue;
                }
                if (depth < 10)
                    for (auto& s : ListDirs(d)) stack.push_back({s, depth + 1});
            }
        }
        const wstring par = Parent(cur);
        if (par.empty() || IEq(par, cur)) break;
        cur = par;
    }
    return out;
}

std::vector<Leftover> FindLeftovers(Context& ctx, const wstring& dir) {
    std::vector<Leftover> out;
    if (dir.empty() || !IsDir(dir) || IsFile(RecordPath(dir))) return out;
    const KnownSet k = KnownFiles(ctx);
    for (const wchar_t* sub : {L"", L"plugins", kAsiPluginOwnDir, kOptiSlotDir}) {
        const wstring d = *sub ? Join(dir, sub) : dir;
        if (!IsDir(d)) continue;
        for (auto pat : {L"*.dll", L"*.asi", L"wronggpu-*.txt"})
            for (auto& f : ListFiles(d, pat)) {
                if (!k.sizes.count(FileSize(f))) continue;
                const wstring h = Lower(Sha256File(f));
                auto it = k.byHash.find(h);
                if (it == k.byHash.end()) continue;
                out.push_back({f.substr(dir.size() + 1), h, it->second.version});
            }
    }
    // an older direct install of ours in the engine's plugin folders (the hosted install never writes there)
    for (auto& [sb, from] : EngineSlotDirs(dir)) {
        auto shownOf = [&from = from](const wstring& f) {
            return (from.size() < f.size() && StartsWithI(f, from + L"\\")) ? f.substr(from.size() + 1) : f;
        };
        wstring proven;  // the version of the first DLL of ours in this folder (empty: none)
        for (auto& f : ListFiles(sb, L"*.dll")) {
            const wstring h = Lower(Sha256File(f));
            auto it = k.byHash.find(h);
            wstring version;
            if (it != k.byHash.end()) version = it->second.version;
            else if (ProvablyOurs(ctx, f)) {
                const Fingerprint fp = IdentifyFile(f);
                version = fp.version.empty() ? wstring(L"(an older direct install)") : fp.version + L" (an older direct install)";
            } else continue;
            if (proven.empty()) proven = version;
            out.push_back({shownOf(f), h, version, f});
        }
        // the old install's settings, logs and text files: only their names say they're ours, so they count only beside a
        // DLL of ours in the same folder
        if (!proven.empty())
            for (const wchar_t* name : kDirectCompanions) {
                const wstring f = Join(sb, name);
                if (IsFile(f)) out.push_back({shownOf(f), Lower(Sha256File(f)), proven, f});
            }
    }
    std::sort(out.begin(), out.end(), [](const Leftover& a, const Leftover& b) { return Lower(a.rel) < Lower(b.rel); });
    return out;
}

wstring LeftoverSummary(const std::vector<Leftover>& files) {
    if (files.empty()) return L"";
    std::map<wstring, int> versions;
    for (auto& f : files) ++versions[f.version];
    wstring v, names;
    for (auto& x : versions) v += (v.empty() ? L"" : L", ") + x.first;
    for (size_t i = 0; i < files.size() && i < 4; ++i) names += (names.empty() ? L"" : L", ") + files[i].rel;
    if (files.size() > 4) names += L", ...";
    return std::to_wstring(files.size()) + (files.size() == 1 ? L" file" : L" files") + L" of WrongGPU " + v + L" (" + names + L")";
}

// The game's FidelityFX DLL a direct install of ours kept under <name>.wronggpu-original.dll, next to our DLL.
static const FfxSlot* SlotOf(const wstring& rel) {
    for (auto& sl : kSlots) if (IEq(BaseName(rel), sl.name)) return &sl;
    return nullptr;
}

std::vector<wstring> CleanLeftovers(Context& ctx, const wstring& dir, const std::vector<Leftover>& files) {
    std::vector<wstring> notes;
    for (auto& f : files) {
        const wstring p = f.abs.empty() ? Target(dir, f.rel) : f.abs;
        if (!IsFile(p)) continue;
        if (Lower(Sha256File(p)) != f.sha) { notes.push_back(L"kept " + f.rel + L" (it changed since it was checked)"); continue; }
        if (!f.abs.empty())
            if (const FfxSlot* sl = SlotOf(f.rel)) {  // the engine's FidelityFX plugin needs a DLL of that name: never leave it without
                const wstring orig = Join(Parent(p), sl->original);
                wstring info;
                if (!IsFile(orig) || ClassifySlot(*sl, orig, nullptr, info) != FFX_GAME) {
                    notes.push_back(L"kept " + f.rel + L": the game's own " + wstring(sl->name) + L" is not there to put back "
                                    L"(verify the game's files in your store, then Clean up again)");
                    continue;
                }
            }
        ForceDelete(p);
        notes.push_back(L"removed " + f.rel + L" (WrongGPU " + f.version + L")");
        if (const FfxSlot* sl = SlotOf(f.rel)) {
            const wstring orig = Join(Parent(p), sl->original);
            wstring info;
            if (IsFile(orig) && ClassifySlot(*sl, orig, nullptr, info) == FFX_GAME) {
                if (!MoveFileExW(orig.c_str(), p.c_str(), MOVEFILE_WRITE_THROUGH)) Fail(L"Cannot put the game's " + wstring(sl->name) + L" back: " + ErrText(GetLastError()));
                notes.push_back(L"the game's " + wstring(sl->name) + L" is back under its own name");
            }
        }
    }
    for (const wchar_t* sub : {L"plugins", kAsiPluginOwnDir}) {
        const wstring d = Join(dir, sub);
        if (IsDir(d) && ListFiles(d, L"*").empty() && ListDirs(d).empty() && RemoveDirectoryW(d.c_str())) notes.push_back(L"removed the empty folder " + wstring(sub));
    }
    LogLine(ctx, L"leftovers cleaned up in " + dir + L": " + std::to_wstring(files.size()) + L" file(s)");
    return notes;
}

std::vector<wstring> AdoptLeftovers(Context& ctx, const wstring& exe, const wstring& dir, const std::vector<Leftover>& files) {
    std::vector<wstring> notes;
    if (files.empty()) Fail(L"No leftover WrongGPU files in " + dir + L".", EXIT_NOT_INSTALLED);
    for (auto& f : files)
        if (!f.abs.empty())
            Fail(L"An older direct install of WrongGPU is in the game's engine folder (" + f.rel + L"). It can't be adopted: "
                 L"Clean up removes it and puts the game's own file back.", EXIT_REFUSED);
    if (IsFile(RecordPath(dir))) Fail(L"WrongGPU already has an install record in " + dir + L".", EXIT_REFUSED);
    std::map<wstring, int> versions;
    for (auto& f : files) ++versions[f.version];
    wstring version;
    int best = 0;
    for (auto& v : versions) if (v.second > best) { best = v.second; version = v.first; }
    Record m;
    m.version = version; m.game = dir; m.exe = exe; m.status = Validated(exe) ? L"validated" : L"experimental";
    m.regkey = ctx.perGameEntry ? RegKeyFor(dir) : L"";
    m.model = ModelFile(ctx);
    m.payloadVersion = version;
    for (auto& f : files) {
        Entry e; e.action = L"copy"; e.rel = f.rel; e.inst = f.sha; e.owner = L"wg";
        if (IEq(BaseName(f.rel), L"wronggpu_runtime.dll")) m.runtimeSha = f.sha;
        else if (IEq(BaseName(f.rel), kOptiFrontEnd) || IsOptiFrontEnd(f.rel) || IEq(BaseName(f.rel), L"amd_fidelityfx_dx12.dll")) m.frontEndSha = f.sha;
        else if (EndsWithI(f.rel, L".asi")) m.pluginSha = f.sha;
        if (const FfxSlot* sl = SlotOf(f.rel)) {
            // A direct install of ours: the game's DLL is kept as <name>.wronggpu-original.dll. As the install engine
            // records it: the game's file is in the backup (restored under its own name on uninstall) and the kept copy
            // is a "rename" entry (removed then).
            const wstring orig = Join(Parent(Target(dir, f.rel)), sl->original);
            wstring info;
            if (IsFile(orig) && ClassifySlot(*sl, orig, nullptr, info) == FFX_GAME) {
                Entry r; r.action = L"rename"; r.rel = f.rel.find(L'\\') == wstring::npos ? wstring(sl->original) : Join(Parent(f.rel), sl->original);
                r.inst = Sha256File(orig); r.owner = L"moved";
                m.e.insert(m.e.begin(), r);
                e.existed = true; e.orig = r.inst;
                CopyChecked(orig, BackupPath(dir, f.rel));
                if (Sha256File(BackupPath(dir, f.rel)) != e.orig) Fail(L"Backup verification failed for " + f.rel);
            }
        }
        m.e.push_back(e);
    }
    // folders that already exist are not provably ours: no "dir" entries
    const wstring state = StateDir(dir);
    EnsureDir(state);
    SaveRecord(RecordPath(dir), m);
    if (ctx.restorer && ctx.restorerSize) WriteFileAtomic(Join(state, L"uninstall.exe"), ctx.restorer, ctx.restorerSize);
    else CopyChecked(SelfPath(), Join(state, L"uninstall.exe"));
    if (ctx.perGameEntry) RegisterUninstall(ctx, m, state, GameDisplayName(exe));
    WriteOwnedFiles(dir, m);
    AddManagedGame(dir);
    notes.push_back(L"an install record was made for " + LeftoverSummary(files) + L": Update or Repair brings this game to WrongGPU " +
                    wstring(ctx.data ? ctx.data->version : L"") + L", Uninstall removes the files");
    LogLine(ctx, L"leftovers adopted in " + dir + L" (WrongGPU " + version + L")");
    return notes;
}

}  // namespace wg
