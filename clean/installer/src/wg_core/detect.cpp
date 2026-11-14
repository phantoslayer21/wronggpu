// SPDX-License-Identifier: MIT
// WrongGPU launcher core: game discovery (Steam, Epic, GOG), the game executable and its folders, the FidelityFX DLLs
// WrongGPU replaces (the game's, ours or another mod's) and the inspection of one game. Written for this project.
#include "detect.h"
#include "checks.h"
#include "gamerules.h"
#include "payload.h"
#include "pe.h"
#include "settings.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// Game discovery (store scan from the 0.2.0 installer; the filter is now "ships amd_fidelityfx_dx12.dll")
// ------------------------------------------------------------------------------------------------
wstring UnescapeJsonLike(wstring s) {
    wstring r;
    for (size_t i = 0; i < s.size(); ++i) { if (s[i] == L'\\' && i + 1 < s.size()) { r += s[i + 1]; ++i; } else r += s[i]; }
    return r;
}
// Returns the text of the n-th quoted string on a line (0-based).
wstring QuotedAt(const wstring& line, int n) {
    size_t p = 0;
    for (int k = 0; k <= n; ++k) {
        size_t a = line.find(L'"', p);
        if (a == wstring::npos) return L"";
        size_t b = a + 1;
        while (b < line.size() && !(line[b] == L'"' && line[b - 1] != L'\\')) ++b;
        if (b >= line.size()) return L"";
        if (k == n) return line.substr(a + 1, b - a - 1);
        p = b + 1;
    }
    return L"";
}
// Value of "key" in a VDF/ACF or JSON-like text (first occurrence).
wstring KeyValue(const wstring& text, const wchar_t* key) {
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find(L'\n', pos);
        wstring line = text.substr(pos, nl == wstring::npos ? wstring::npos : nl - pos);
        pos = nl == wstring::npos ? text.size() : nl + 1;
        if (IEq(QuotedAt(line, 0), key)) return UnescapeJsonLike(QuotedAt(line, 1));
    }
    return L"";
}
wstring RegString(HKEY root, const wchar_t* key, const wchar_t* value, REGSAM extra) {
    HKEY k;
    if (RegOpenKeyExW(root, key, 0, KEY_READ | extra, &k) != ERROR_SUCCESS) return L"";
    wchar_t buf[MAX_PATH * 2]; DWORD sz = sizeof(buf) - sizeof(wchar_t), type = 0;
    LSTATUS st = RegQueryValueExW(k, value, nullptr, &type, (LPBYTE)buf, &sz);
    RegCloseKey(k);
    if (st != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ)) return L"";
    buf[sz / sizeof(wchar_t)] = 0;
    return buf;
}

// Store layouts: <root>\bin\x64\Game.exe (REDengine and others), <root>\<Project>\Binaries\Win64\X-Win64-Shipping.exe (Unreal).
wstring GameRoot(const wstring& exeDir) {
    wstring d = exeDir, leaf = Lower(BaseName(d)), up = Lower(BaseName(Parent(d)));
    if ((leaf == L"win64" || leaf == L"wingdk") && up == L"binaries") return Parent(Parent(Parent(d)));
    if (leaf == L"x64" && up == L"bin") return Parent(Parent(d));
    return d;
}

// Programs that are never the game: launchers, crash reporters, redistributables, anti-cheat, and the tools of the
// DirectX Agility SDK that a game ships next to D3D12Core.dll (D3D12StateObjectCompiler.exe, d3dconfig.exe).
bool LooksLikeHelperExe(const wstring& exe) {
    static const wchar_t* bad[] = {L"crash", L"launcher", L"setup", L"unins", L"redist", L"vcredist", L"vc_redist", L"dxsetup", L"prereq",
                                   L"easyanticheat", L"start_protected", L"helper", L"report", L"server", L"uploader", L"updater", L"installer",
                                   L"dotnet", L"directx", L"touchup", L"cleanup", L"cefprocess", L"webhelper", L"qtwebengine", L"_be.exe",
                                   L"benchmark", L"eaanticheat", L"battleye", L"overlay", L"7za", L"oalinst", L"physx", L"wronggpu", L"dlss45",
                                   L"d3d12stateobjectcompiler", L"d3dconfig"};
    wstring l = Lower(BaseName(exe));
    for (auto s : bad) if (l.find(s) != wstring::npos) return true;
    return false;
}
// Unreal: <root>\Game.exe is a small bootstrapper; the real game is <root>\<Project>\Binaries\Win64\*-Shipping.exe.
wstring UnrealShipping(const wstring& root) {
    for (auto& project : ListDirs(root))
        for (const wchar_t* plat : {L"Binaries\\Win64", L"Binaries\\WinGDK"})
            for (auto& exe : ListFiles(Join(project, plat), L"*-Shipping.exe")) if (!LooksLikeHelperExe(exe)) return exe;
    return L"";
}
// rc6: a starter in the game's root whose game has the same name below it: <root>\<Project>\Binaries\Win64 (or WinGDK)\
// <name>.exe, an Unreal game whose real exe isn't named *-Shipping (Hogwarts Legacy: a small HogwartsLegacy.exe in the root,
// the game in Phoenix\Binaries\Win64\HogwartsLegacy.exe), or <root>\bin\x64\<name>.exe. Empty: none (or exe is not in a
// root: already in Binaries\Win64 or bin\x64).
wstring SameNameGameExe(const wstring& exe) {
    const wstring root = Parent(exe), name = BaseName(exe);
    if (root.empty() || !IEq(GameRoot(root), root)) return L"";
    if (const wstring x = Join(root, L"bin\\x64\\" + name); IsFile(x)) return x;
    for (auto& project : ListDirs(root))
        for (const wchar_t* plat : {L"Binaries\\Win64", L"Binaries\\WinGDK"})
            if (const wstring x = Join(Join(project, plat), name); IsFile(x)) return x;
    return L"";
}
// Other upscalers' files that a game keeps next to its exe (the game's DLSS model file, XeSS, AMD's FSR 4 upscaler).
const wchar_t* const kUpscalerFiles[] = {L"nvngx_dlss.dll", L"libxess.dll", L"amd_fidelityfx_upscaler_dx12.dll"};
// Picks the game executable below root (depth <= 4): the Unreal shipping exe first. Else, of the exes that are not
// helpers:
//   1. Where the exe is decides first: next to one of the FidelityFX DLLs WrongGPU replaces (2), next to another
//      upscaler's file (1, kUpscalerFiles), anywhere else (0), in a folder that holds the DirectX Agility SDK
//      (D3D12Core.dll: the SDK's own programs, -1; such a folder supplies the game only when nothing else does).
//   2. Among those, only the large ones count: at least a quarter of the largest. A small tool never wins on its name.
//   3. Among the large ones the name breaks the tie: a DX12 build before a plain name before a DX11 / Vulkan build
//      (Control_DX12.exe next to Control_DX11.exe), then the larger file.
wstring FindGameExe(const wstring& root, const wstring& hint) {
    if (!hint.empty() && IsFile(hint)) {
        wstring ue = UnrealShipping(Parent(hint));
        if (ue.empty()) ue = SameNameGameExe(hint);  // a root starter of the same name (rc6)
        return ue.empty() ? hint : ue;
    }
    if (wstring ue = UnrealShipping(root); !ue.empty()) return ue;
    struct Candidate { wstring path; long long size; int where, name; };
    std::vector<Candidate> all;
    std::vector<std::pair<wstring, int>> stack{{root, 0}};
    size_t visited = 0;
    while (!stack.empty() && visited < 20000) {
        auto [dir, depth] = stack.back(); stack.pop_back();
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        const size_t first = all.size();
        do {
            ++visited;
            wstring n = fd.cFileName;
            if (n == L"." || n == L"..") continue;
            wstring p = Join(dir, n);
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                if (depth < 4 && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && !IEq(n, kStateDirName) && !IEq(n, kLegacyStateDir)) stack.push_back({p, depth + 1});
            } else if (EndsWithI(n, L".exe") && !LooksLikeHelperExe(p)) {
                const int name = Contains(n, L"dx12") || Contains(n, L"d3d12") ? 1
                               : Contains(n, L"dx11") || Contains(n, L"d3d11") || Contains(n, L"vulkan") || Contains(n, L"_vk") ? -1 : 0;
                all.push_back({p, ((long long)fd.nFileSizeHigh << 32) | fd.nFileSizeLow, 0, name});
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
        if (first == all.size()) continue;
        int where = 0;
        for (auto& sl : kSlots) if (IsFile(Join(dir, sl.name))) where = 2;
        for (auto f : kUpscalerFiles) if (!where && IsFile(Join(dir, f))) where = 1;
        if (!where && IsFile(Join(dir, L"D3D12Core.dll"))) where = -1;
        for (size_t i = first; i < all.size(); ++i) all[i].where = where;
    }
    int where = -2;
    long long largest = 0;
    for (auto& c : all) where = std::max(where, c.where);
    for (auto& c : all) if (c.where == where) largest = std::max(largest, c.size);
    const Candidate* best = nullptr;
    for (auto& c : all) {
        if (c.where != where || c.size < largest / 4) continue;
        if (!best || c.name > best->name || (c.name == best->name && c.size > best->size)) best = &c;
    }
    return best ? best->path : wstring();
}
const KnownGame* Validated(const wstring& exe) {
    for (auto& k : kValidated) if (IEq(BaseName(exe), k.exe)) return &k;
    return nullptr;
}

// One Epic Games Store manifest (<ProgramData>\Epic\EpicGamesLauncher\Data\Manifests\*.item, JSON): its DisplayName,
// InstallLocation and LaunchExecutable (the hint: InstallLocation\LaunchExecutable; StoreExe decides whether it is the game).
static wstring EpicField(const wstring& t, const wchar_t* key) {
    size_t p = t.find(wstring(L"\"") + key + L"\"");
    return p == wstring::npos ? wstring() : UnescapeJsonLike(QuotedAt(t.substr(p, t.find(L'\n', p) - p), 1));
}
StoreGame EpicItemGame(const wstring& t) {
    const wstring loc = EpicField(t, L"InstallLocation"), launch = EpicField(t, L"LaunchExecutable");
    return {EpicField(t, L"DisplayName"), loc, launch.empty() ? L"" : Join(loc, launch), L"Epic"};
}
// rc6 L7: the Epic manifest whose InstallLocation holds exe (read as the library scan reads them): its AppName, for Play's
// com.epicgames.launcher://apps/<AppName> address. Only letters, digits, '.', '_' and '-' (an address part); else none.
wstring EpicAppName(const wstring& exe) {
    const wstring pd = EnvVar(L"ProgramData");
    if (pd.empty() || exe.empty()) return L"";
    const wstring full = FullPath(exe);
    for (auto& item : ListFiles(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests"), L"*.item")) {
        string raw;
        if (!ReadAll(item, raw)) continue;
        const wstring t = FromUtf8(raw);
        wstring loc = EpicField(t, L"InstallLocation");
        for (auto& c : loc) if (c == L'/') c = L'\\';
        while (loc.size() > 3 && loc.back() == L'\\') loc.pop_back();
        if (loc.empty()) continue;
        loc = FullPath(loc);
        if (full.size() <= loc.size() + 1 || !StartsWithI(full, loc) || full[loc.size()] != L'\\') continue;
        const wstring app = EpicField(t, L"AppName");
        if (app.empty()) continue;
        for (wchar_t c : app)
            if (!((c >= L'0' && c <= L'9') || (c >= L'a' && c <= L'z') || (c >= L'A' && c <= L'Z') || c == L'.' || c == L'_' || c == L'-')) return L"";
        return app;
    }
    return L"";
}
// A store's launch exe (Epic's LaunchExecutable, GOG's exe) is a hint only: a launcher stub (gamerules.h kLauncherStubs)
// gives none, and so does one beside which the exe discovery finds a known game's exe (a validated game, a game with rules)
// while the hint leads to none; FindGameExe then finds the game as for a Steam game (no hint). Any other launch exe is kept.
static bool KnownGameExe(const wstring& exe) { return !exe.empty() && (Validated(exe) || GameRuleFor(exe)); }
wstring StoreExe(const wstring& root, const wstring& hint) {
    if (hint.empty() || IsLauncherStub(hint)) return L"";
    if (!IsFile(hint) || KnownGameExe(FindGameExe(root, hint))) return hint;
    return KnownGameExe(FindGameExe(root)) ? wstring() : hint;
}

// Every game folder of the Steam, Epic and GOG libraries on this PC. Read-only.
std::vector<StoreGame> ScanLibraries() {
    std::vector<StoreGame> roots;
    auto add = [&](const wstring& name, const wstring& root, const wstring& hint, const wchar_t* source) {
        if (name.empty() || root.empty() || !IsDir(root)) return;
        for (auto& r : roots) if (IEq(FullPath(r.root), FullPath(root))) return;
        roots.push_back({name, FullPath(root), hint, source});
    };
    // Steam: every library in libraryfolders.vdf, every app manifest in it.
    wstring steam = RegString(HKEY_CURRENT_USER, L"Software\\Valve\\Steam", L"SteamPath");
    if (!steam.empty()) {
        for (auto& c : steam) if (c == L'/') c = L'\\';
        std::vector<wstring> libs{steam};
        string raw;
        if (ReadAll(Join(steam, L"steamapps\\libraryfolders.vdf"), raw)) {
            wstring t = FromUtf8(raw); size_t pos = 0;
            while (pos < t.size()) {
                size_t nl = t.find(L'\n', pos); wstring line = t.substr(pos, nl == wstring::npos ? wstring::npos : nl - pos); pos = nl == wstring::npos ? t.size() : nl + 1;
                if (Lower(QuotedAt(line, 0)) == L"path") {
                    wstring lib = UnescapeJsonLike(QuotedAt(line, 1));
                    bool dup = false; for (auto& l : libs) dup |= IEq(FullPath(l), FullPath(lib));
                    if (!dup) libs.push_back(lib);
                }
            }
        }
        for (auto& lib : libs) {
            for (auto& acf : ListFiles(Join(lib, L"steamapps"), L"appmanifest_*.acf")) {
                string a;
                if (!ReadAll(acf, a)) continue;
                wstring t = FromUtf8(a), dir = KeyValue(t, L"installdir");
                // benchmarks that ship upscaler test files but are not games (3DMark and its demo)
                static const wchar_t* const kNotGames[] = {L"223850", L"231350"};
                const wstring appid = KeyValue(t, L"appid");
                bool tool = false;
                for (auto* n : kNotGames) tool = tool || appid == n;
                if (!dir.empty() && !tool) add(KeyValue(t, L"name"), Join(Join(lib, L"steamapps\\common"), dir), L"", L"Steam");
            }
        }
    }
    // Epic Games Store.
    wstring pd = EnvVar(L"ProgramData");
    if (!pd.empty()) {
        for (auto& item : ListFiles(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests"), L"*.item")) {
            string raw;
            if (!ReadAll(item, raw)) continue;
            const StoreGame e = EpicItemGame(FromUtf8(raw));
            add(e.name, e.root, e.hint, L"Epic");
        }
    }
    // GOG Galaxy and GOG offline installers.
    HKEY gog;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\GOG.com\\Games", 0, KEY_READ | KEY_WOW64_32KEY, &gog) == ERROR_SUCCESS) {
        wchar_t sub[256];
        for (DWORD i = 0;; ++i) {
            DWORD n = (DWORD)std::size(sub);
            if (RegEnumKeyExW(gog, i, sub, &n, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
            wstring key = wstring(L"SOFTWARE\\GOG.com\\Games\\") + sub;
            add(RegString(HKEY_LOCAL_MACHINE, key.c_str(), L"gameName", KEY_WOW64_32KEY), RegString(HKEY_LOCAL_MACHINE, key.c_str(), L"path", KEY_WOW64_32KEY),
                RegString(HKEY_LOCAL_MACHINE, key.c_str(), L"exe", KEY_WOW64_32KEY), L"GOG");
        }
        RegCloseKey(gog);
    }
    return roots;
}

// The FSR games among the library folders: their install contains the FSR 3.1 API DLL or the FSR 3.0 component DLLs.
std::vector<Found> DetectGames(const std::vector<StoreGame>& roots) {
    std::vector<Found> found;
    for (auto& r : roots) {
        bool fsr = false;
        for (auto& sl : kSlots) fsr = fsr || !FindFilesNamed(r.root, sl.name, 30000, 8).empty();
        // games with DLSS or XeSS only: installable with the player's OptiScaler
        const bool other = !fsr && (!FindFilesNamed(r.root, L"nvngx_dlss.dll", 30000, 8).empty() || !FindFilesNamed(r.root, L"libxess.dll", 30000, 8).empty());
        if (!fsr && !other) continue;
        wstring exe = FindGameExe(r.root, StoreExe(r.root, r.hint));
        if (exe.empty()) continue;
        Found f; f.name = r.name; f.exe = exe; f.source = r.source; f.validated = Validated(exe) != nullptr; f.opti = !fsr;
        found.push_back(f);
    }
    std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) {
        if (a.validated != b.validated) return a.validated;
        return Lower(a.name) < Lower(b.name);
    });
    return found;
}

// Accepts a game .exe, a store folder or the folder with the exe; returns the game executable (or empty).
wstring ResolveGameExe(wstring in) {
    while (!in.empty() && (in.front() == L'"' || in.front() == L' ')) in.erase(in.begin());
    while (!in.empty() && (in.back() == L'"' || in.back() == L' ' || in.back() == L'\\' || in.back() == L'/')) in.pop_back();
    if (in.empty()) return L"";
    in = FullPath(in);
    if (IsFile(in) && EndsWithI(in, L".exe") && IsLauncherStub(in)) return ResolveGameExe(Parent(in));  // a stub: its game folder
    if (IsFile(in) && EndsWithI(in, L".exe")) return FindGameExe(Parent(in), in);
    if (IsDir(in)) {
        for (auto& k : kValidated) if (IsFile(Join(in, k.exe))) return Join(in, k.exe);
        for (auto& k : kValidated) if (IsFile(Join(in, wstring(L"bin\\x64\\") + k.exe))) return Join(in, wstring(L"bin\\x64\\") + k.exe);
        return FindGameExe(in);
    }
    return L"";
}

// ------------------------------------------------------------------------------------------------
// Target game: executable, the FidelityFX DLL WrongGPU replaces, conflicts
// ------------------------------------------------------------------------------------------------
bool IsSlotName(const wstring& name) { for (auto& sl : kSlots) if (IEq(name, sl.name) || IEq(name, sl.original)) return true; return false; }

// Authenticode check of a file, for the log only (spec 30 answer 5: logged, never required). Revocation checks and
// any network retrieval are switched off: the installer never uses the network. wintrust/crypt32 load from System32.
wstring SignatureInfo(const wstring& path) {
    HMODULE wt = LoadSystemDll(L"wintrust.dll"), c32 = LoadSystemDll(L"crypt32.dll");
    if (!wt || !c32) return L"signature not checked";
    using VerifyFn = LONG(WINAPI*)(HWND, GUID*, LPVOID);
    using ProvFn = void*(WINAPI*)(HANDLE);
    using SignerFn = void*(WINAPI*)(void*, DWORD, BOOL, DWORD);
    using NameFn = DWORD(WINAPI*)(const void*, DWORD, DWORD, void*, LPWSTR, DWORD);
    auto verify = reinterpret_cast<VerifyFn>(reinterpret_cast<void*>(GetProcAddress(wt, "WinVerifyTrust")));
    auto provData = reinterpret_cast<ProvFn>(reinterpret_cast<void*>(GetProcAddress(wt, "WTHelperProvDataFromStateData")));
    auto signer = reinterpret_cast<SignerFn>(reinterpret_cast<void*>(GetProcAddress(wt, "WTHelperGetProvSignerFromChain")));
    auto certName = reinterpret_cast<NameFn>(reinterpret_cast<void*>(GetProcAddress(c32, "CertGetNameStringW")));
    if (!verify) return L"signature not checked";
    // WINTRUST_ACTION_GENERIC_VERIFY_V2 {00AAC56B-CD44-11d0-8CC2-00C04FC295EE}; structures as in wintrust.h.
    GUID action = {0x00AAC56B, 0xCD44, 0x11d0, {0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE}};
    struct FileInfo { DWORD cbStruct; LPCWSTR path; HANDLE file; GUID* subject; } fi{sizeof(FileInfo), path.c_str(), nullptr, nullptr};
    struct TrustData {
        DWORD cbStruct; LPVOID policy; LPVOID sip; DWORD ui; DWORD revocation; DWORD choice; FileInfo* file;
        DWORD stateAction; HANDLE state; LPWSTR url; DWORD provFlags; DWORD context; void* signature;
    } td{};
    td.cbStruct = sizeof(td); td.ui = 2 /* WTD_UI_NONE */; td.revocation = 0 /* WTD_REVOKE_NONE */; td.choice = 1 /* WTD_CHOICE_FILE */;
    td.file = &fi; td.stateAction = 1 /* WTD_STATEACTION_VERIFY */;
    td.provFlags = 0x00000010 /* WTD_REVOCATION_CHECK_NONE */ | 0x00001000 /* WTD_CACHE_ONLY_URL_RETRIEVAL */;
    const LONG r = verify(reinterpret_cast<HWND>(INVALID_HANDLE_VALUE), &action, &td);
    wstring out;
    if (r == 0) {
        out = L"Authenticode signature valid";
        if (provData && signer && certName) {
            void* prov = provData(td.state);
            // CRYPT_PROVIDER_SGNR: cbStruct, sftVerifyAsOf (FILETIME), csCertChain, pasCertChain, ...
            if (void* sg = prov ? signer(prov, 0, FALSE, 0) : nullptr) {
                struct Sgnr { DWORD cbStruct; FILETIME asOf; DWORD chain; struct Cert { DWORD cbStruct; const void* cert; }* certs; };
                const Sgnr* sgnr = static_cast<const Sgnr*>(sg);
                if (sgnr->chain && sgnr->certs && sgnr->certs[0].cert) {
                    wchar_t name[256] = {};
                    if (certName(sgnr->certs[0].cert, 4 /* CERT_NAME_SIMPLE_DISPLAY_TYPE */, 0, nullptr, name, 256) > 1) out += wstring(L", signer ") + name;
                }
            }
        }
    } else if ((unsigned long)r == 0x800B0100u) out = L"no Authenticode signature";
    else out = L"Authenticode check failed (" + Hex32((DWORD)r) + L")";
    td.stateAction = 2;  // WTD_STATEACTION_CLOSE
    verify(reinterpret_cast<HWND>(INVALID_HANDLE_VALUE), &action, &td);
    return out;
}

// Ours: the version resource names the product WrongGPU (contract with the front end), or the file is the one the install
// record says was installed. The FFX API DLL is the game's if its version resource names Advanced Micro Devices; an
// FSR 3.0 component DLL is the game's if it exports that component's context functions.
FfxKind ClassifySlot(const FfxSlot& slot, const wstring& path, const Record* rec, wstring& info) {
    if (!IsFile(path)) { info = L"missing"; return FFX_NONE; }
    MappedFile f(path, 2ull << 30);
    PeVersion v = ReadPeVersion(f.data, f.size);
    const wstring company = v.Get(L"CompanyName"), product = v.Get(L"ProductName");
    info = (company.empty() ? wstring(L"no company name") : company) + L", version " + (v.fileVersion.empty() ? wstring(L"unknown") : v.fileVersion);
    if (rec) {
        const wstring sha = Sha256File(path);
        for (auto& e : rec->e) if (e.action == L"copy" && IEq(e.rel, slot.name) && e.inst == sha) return FFX_OURS;
    }
    if (IEq(product, kAppName)) return FFX_OURS;
    if (slot.amdBuilt) return Contains(company, L"Advanced Micro Devices") ? FFX_GAME : FFX_FOREIGN;
    const std::set<string> ex = ReadPeExports(f.data, f.size);
    const bool fsr = ex.count(slot.exports[0]) && ex.count(slot.exports[1]);
    info = wstring(fsr ? L"exports the " : L"does not export the ") + slot.what + L" functions, " + info;
    return fsr ? FFX_GAME : FFX_FOREIGN;
}
std::vector<SlotState> InspectSlots(Context& ctx, const wstring& dir, const Record* rec) {
    std::vector<SlotState> out;
    for (auto& sl : kSlots) {
        SlotState st; st.slot = &sl; st.inPayload = PayloadHas(ctx, sl.name);
        st.kind = ClassifySlot(sl, Join(dir, sl.name), rec, st.info);
        out.push_back(st);
    }
    return out;
}

// Display name: the Steam app name when the game lives in a Steam library, else the exe name without Unreal suffixes.
wstring GameDisplayName(const wstring& exe) {
    const wstring root = GameRoot(Parent(exe)), common = Parent(root), steamapps = Parent(common);
    if (IEq(BaseName(common), L"common") && IEq(BaseName(steamapps), L"steamapps")) {
        for (auto& acf : ListFiles(steamapps, L"appmanifest_*.acf")) {
            string a;
            if (!ReadAll(acf, a)) continue;
            wstring t = FromUtf8(a);
            if (IEq(KeyValue(t, L"installdir"), BaseName(root))) { wstring n = KeyValue(t, L"name"); if (!n.empty()) return n; }
        }
    }
    // outside Steam: the game's folder when the exe sits in a known layout (bin\x64, Binaries\Win64): "Quarry Nine",
    // not "Game" or "QuarryNine-Win64-Shipping"; else the exe's name
    if (!IEq(root, Parent(exe))) { const wstring n = BaseName(root); if (!n.empty()) return n; }
    wstring stem = Stem(exe);
    for (const wchar_t* suffix : {L"-Win64-Shipping", L"-WinGDK-Shipping"}) if (EndsWithI(stem, suffix)) stem.resize(stem.size() - wcslen(suffix));
    return stem;
}
// The Steam app id of a game in a Steam library: the appmanifest of the library's steamapps folder whose installdir is the
// game's folder there (the exe's folder or the one above it that sits directly in steamapps\common, whatever the layout
// below it). Play starts a game with an id through Steam (PlayGame).
wstring SteamAppId(const wstring& exe) {
    if (exe.empty()) return L"";
    wstring root = Parent(exe);
    while (!root.empty() && !(IEq(BaseName(Parent(root)), L"common") && IEq(BaseName(Parent(Parent(root))), L"steamapps"))) root = Parent(root);
    if (root.empty() || BaseName(root).empty()) return L"";
    const wstring steamapps = Parent(Parent(root));
    for (auto& acf : ListFiles(steamapps, L"appmanifest_*.acf")) {
        string a;
        if (!ReadAll(acf, a)) continue;
        const wstring t = FromUtf8(a);
        if (!IEq(KeyValue(t, L"installdir"), BaseName(root))) continue;
        wstring id = KeyValue(t, L"appid");
        if (id.empty()) {  // appmanifest_<id>.acf
            const wstring n = BaseName(acf);
            const size_t u = n.find(L'_'), d = n.rfind(L'.');
            if (u != wstring::npos && d != wstring::npos && d > u) id = n.substr(u + 1, d - u - 1);
        }
        for (wchar_t c : id) if (c < L'0' || c > L'9') return L"";
        return id;
    }
    return L"";
}
// The folder with the FidelityFX DLLs the game loads: the exe folder if it has one of them, else the first folder below
// the game root that has the FFX API DLL, else the first that has an FSR 3.0 component DLL.
wstring FindInstallDir(const wstring& exe) {
    const wstring exeDir = Parent(exe);
    if (IsFile(RecordPath(exeDir))) return exeDir;  // an install next to the exe (OptiScaler mode)
    for (auto& sl : kSlots) if (IsFile(Join(exeDir, sl.name))) return exeDir;
    for (auto& sl : kSlots) {
        auto all = FindFilesNamed(GameRoot(exeDir), sl.name, 60000, 8);
        if (!all.empty()) return Parent(all.front());
    }
    return exeDir;
}
bool StartsOnPresetK(const wstring& dir, const Record* rec) {
    auto lower = [](string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; };
    string text;
    if (ReadAll(Join(dir, L"wronggpu-asi.ini"), text)) {  // the plugin's choice decides over wronggpu.ini's model=
        for (size_t a = 0; a < text.size();) {
            size_t e = text.find(char(10), a);
            if (e == string::npos) e = text.size();
            string line = lower(text.substr(a, e - a));
            while (!line.empty() && (line.back() == char(13) || line.back() == ' ' || line.back() == char(9))) line.pop_back();
            if (line.rfind("selection=", 0) == 0) return line == "selection=wronggpu-k";
            a = e + 1;
        }
    }
    string options, section;
    if (ReadAll(Join(dir, L"wronggpu.ini"), text) && IniGetFrontEnd(text, "runtime_options", &options, &section)) {
        string model;  // the last model item wins
        for (size_t a = 0; a <= options.size();) {
            size_t e = options.find(';', a);
            if (e == string::npos) e = options.size();
            const string item = lower(options.substr(a, e - a));
            if (item.rfind("model=", 0) == 0) model = item;
            a = e + 1;
        }
        if (!model.empty()) return model == "model=k";
    }
    return rec && !rec->model.empty() && IEq(Parent(rec->model), Join(ModelBaseDir(), L"K"));
}
GameTarget InspectGame(Context& ctx, const wstring& exe, const wstring& displayName) {
    GameTarget t;
    t.exe = exe; t.exeDir = Parent(exe); t.root = GameRoot(t.exeDir);
    const KnownGame* k = Validated(exe);
    t.validated = k != nullptr;
    t.name = k ? k->name : !displayName.empty() ? displayName : GameDisplayName(exe);
    t.dir = ctx.opti.ok() || ctx.hostedLayout ? t.exeDir : FindInstallDir(exe);  // OptiScaler loads as a proxy DLL next to the exe
    t.installed = LoadRecord(RecordPath(t.dir), t.record) && !t.record.legacy;
    if (t.installed && !t.record.movedFrom.empty()) SaveRebasedRecord(ctx, t.dir, t.record);  // a moved game folder (rc6 L1)
    t.presetK = t.installed && StartsOnPresetK(t.dir, &t.record);
    if (!t.installed && (ctx.opti.ok() || ctx.hostedLayout) && !IEq(FindInstallDir(exe), t.dir)) {  // an earlier install in the FSR folder
        Record r;
        if (LoadRecord(RecordPath(FindInstallDir(exe)), r) && !r.legacy) LogLine(ctx, L"note: an earlier WrongGPU install is in " + FindInstallDir(exe));
    }
    t.slots = InspectSlots(ctx, t.dir, t.installed ? &t.record : nullptr);
    t.antiCheat = FindAntiCheat(exe);
    t.conflicts = FindConflicts(t.dir, t.exeDir);
    if (t.installed) {  // the OptiScaler files of our own earlier install are not "another mod", nor is the log it wrote
        // 0.6.1: a DLL under a name the record copied counts as ours only with the record's hash; another setup's OptiScaler
        // that replaced WrongGPU's proxy is another mod (Repair and Update ask the switch question for it). Settings and
        // logs of our setup are not asked about (OptiScaler's menu saves its ini, its OptiScaler writes the log); this decides
        // the question only: what a restore or install deletes or overwrites goes by bytes (record.h, the ownership rule).
        std::map<wstring, wstring> ours;  // lower-case name -> installed hash
        for (auto& e : t.record.e) if (e.action == L"copy") ours[Lower(e.rel)] = e.inst;
        if (HostedRecord(t.record)) ours[L"optiscaler.log"] = L"";
        auto isOurs = [&](const wstring& f) {
            const auto it = ours.find(Lower(f));
            if (it == ours.end()) return false;
            if (IEq(f, L"OptiScaler.ini")) return OptiSetupIsOurs(t.dir, t.record);  // (what the scan asks about; the restore goes by bytes)
            if (EndsWithI(f, L".ini") || EndsWithI(f, L".log")) return true;
            return !it->second.empty() && IEq(Sha256File(Join(t.dir, f)), it->second);
        };
        auto& mf = t.conflicts.modFiles;
        mf.erase(std::remove_if(mf.begin(), mf.end(), isOurs), mf.end());
        if (mf.empty() && t.conflicts.outside.empty()) t.conflicts.mods.clear();
    }
    t.coexist = AssessCoexistence(ctx, exe, t.dir, t.installed ? &t.record : nullptr);
    return t;
}


}  // namespace wg
