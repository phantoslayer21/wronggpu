// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU launcher core: the Context, the log file, the model data folder and the embedded converter map. Written for
// this project.
#include "context.h"
#include <cwctype>
#include <shlobj.h>
#include "payload.h"

namespace wg {

void LogLine(Context& ctx, const wstring& s) {
    SYSTEMTIME t; GetLocalTime(&t);
    wchar_t ts[40]; swprintf(ts, 40, L"[%02d:%02d:%02d] ", t.wHour, t.wMinute, t.wSecond);
    string line = ToUtf8(wstring(ts) + s + L"\r\n");
    std::lock_guard<std::mutex> lock(ctx.logMutex);  // the file's name read under it too (the window can move the log meanwhile)
    if (ctx.logFile.empty()) return;
    // Logging must not write through a junction used as the shared data-folder root, even before uninstall starts.
    const wstring shared = AppDataPath(), fullShared = FullPath(shared), fullLog = FullPath(ctx.logFile);
    const DWORD attrs = GetFileAttributesW(shared.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_REPARSE_POINT) &&
        (IEq(fullLog, fullShared) || StartsWithI(fullLog, fullShared + L"\\"))) {
        wchar_t tmp[MAX_PATH + 1] = {};
        const DWORD n = GetTempPathW(MAX_PATH + 1, tmp);
        const wstring temp = n && n <= MAX_PATH ? FullPath(tmp) : wstring();
        if (temp.empty() || IEq(temp, fullShared) || StartsWithI(temp, fullShared + L"\\")) { ctx.logFile.clear(); return; }
        ctx.logFile = Join(temp, L"WrongGPU-junction-" + std::to_wstring(GetCurrentProcessId()) + L".log");
    }
    const bool sharedLog = PathWithin(ctx.logFile, shared);
    if (!GamePathAllowed(ctx.logFile) || (sharedLog && !PathWithin(ctx.logFile, shared, true))) return;
    HANDLE h = CreateFileW(ctx.logFile.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(h, &info) || info.nNumberOfLinks != 1 ||
        (info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))) { CloseHandle(h); return; }
    if (!DataOverride().empty()) {
        const wstring actual = ResolvedHandlePath(h);
        if (actual.empty() || !GamePathAllowed(actual) || (sharedLog && !PathWithin(actual, shared))) {
            CloseHandle(h);
            return;
        }
    }
    DWORD w; WriteFile(h, line.data(), (DWORD)line.size(), &w, nullptr); CloseHandle(h);
}
static wstring g_dataOverride;
#ifdef WRONGGPU_TESTING
static wstring g_fixtureRoot;
#endif
const wstring& DataOverride() { return g_dataOverride; }

bool SetDataOverride(const wstring& path, wstring& why, bool specified) {
    if (specified && path.empty()) { why = L"--data needs a folder."; return false; }
    const wstring full = path.empty() ? wstring() : ResolvedPath(path);
    if (!path.empty() && (full.empty() || !PathWithin(path, path, true) || (Exists(path) && !IsDir(path)))) {
        why = L"The launcher data folder cannot be resolved or goes through a link.";
        return false;
    }
#ifdef WRONGGPU_TESTING
    PWSTR local = nullptr;
    const HRESULT got = SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_DONT_VERIFY, nullptr, &local);
    const wstring real = SUCCEEDED(got) && local ? Join(local, L"WrongGPU") : wstring();
    CoTaskMemFree(local);
    const wstring live = real.empty() ? wstring() : ResolvedPath(real);
    if (full.empty() || live.empty() || PathWithin(full, live) || PathWithin(live, full)) {
        why = L"This WrongGPU test build requires --data pointing to an isolated fixture folder.";
        return false;
    }
    // The process suite can name its fixture workspace explicitly. Only its data child can activate that scope.
    const wstring fixture = EnvVar(L"WRONGGPU_TEST_FIXTURE_ROOT");
    const wstring fixtureRoot = fixture.empty() ? wstring() : ResolvedPath(fixture);
    wstring allowedFixture;
    if (!fixtureRoot.empty() && IEq(full, Join(fixtureRoot, L"data"))) {
        if (!PathWithin(fixture, fixture, true) || PathWithin(fixtureRoot, live) || PathWithin(live, fixtureRoot) ||
            !PathWithin(EnvVar(L"LOCALAPPDATA"), fixtureRoot, true) || !PathWithin(EnvVar(L"APPDATA"), fixtureRoot, true)) {
            why = L"The launcher fixture workspace requires isolated LOCALAPPDATA and APPDATA.";
            return false;
        }
        allowedFixture = fixtureRoot;
    }
#endif
    if (!full.empty()) {
        for (const auto* name : {L"model", L"logs", L"jobs", L"rr-prepare", L"temp", L"launcher.ini", L"launcher.log",
                                 L"managed-games.txt", L"ignore-files.txt", L"launcher-registry.hiv", L"setup.log",
                                 L"optiscaler", L"Programs", L"StartMenu", L"localappdata", L"Documents", L"portable-launchers.txt",
                                 L"library-list.txt", L"hip-runtime.txt", L"compare.txt"}) {
            if (!PathWithin(Join(full, name), full, true)) {
                why = L"A launcher path in --data resolves through a link.";
                return false;
            }
        }
    }
    g_dataOverride = full;
#ifdef WRONGGPU_TESTING
    g_fixtureRoot = allowedFixture;
#endif
    return true;
}

bool GamePathAllowed(const wstring& path) {
#ifdef WRONGGPU_TESTING
    return g_dataOverride.empty() || PathWithin(path, g_dataOverride, true) ||
           (!g_fixtureRoot.empty() && PathWithin(path, g_fixtureRoot, true));
#else
    (void)path;
    return true;
#endif
}

#ifdef WRONGGPU_TESTING
wstring TestRegistryDir() {
    // The process suite's private hive lives beside its data directory so that Windows can
    // release its registry files independently of launcher data removal.
    if (!g_fixtureRoot.empty()) {
        const wstring profile = EnvVar(L"LOCALAPPDATA");
        if (PathWithin(profile, g_fixtureRoot, true)) return profile;
    }
    return DataOverride();
}
#endif

// The active shared data path, without creating it.
wstring AppDataPath() {
    if (!g_dataOverride.empty()) return g_dataOverride;
    wstring base = EnvVar(L"LOCALAPPDATA");
    if (base.empty()) { wchar_t b[MAX_PATH]; GetTempPathW(MAX_PATH, b); base = b; }
    return Join(base, L"WrongGPU");
}
// %LOCALAPPDATA%\WrongGPU (created on first use).
wstring AppDataDir() {
    const wstring dir = AppDataPath();
    EnsureDir(dir);
    return dir;
}
wstring DefaultLogPath() { return Join(AppDataDir(), L"setup.log"); }
// Preset K: CMake option WRONGGPU_PRESET_K (CMakeLists.txt). OFF: Model M only, one model cache. Only the name: the
// converter's writer makes the folders it writes into, and reading or showing the path never brings the folder back (the
// launcher's uninstall moves it away while its window still paints).
wstring ModelBaseDir() { return Join(AppDataPath(), L"model"); }
#if WRONGGPU_PRESET_K
int MapResourceId(Context& ctx, bool k) { return k ? ctx.data->kMapId : ctx.data->mapId; }
const wchar_t* MapSha256(Context& ctx, bool k) { return k ? ctx.data->kMapSha256 : ctx.data->mapSha256; }
wstring ModelDir(Context& ctx) {
    return ctx.presetFG ? Join(ModelBaseDir(), L"FG") : ctx.presetL ? Join(ModelBaseDir(), L"L") : ctx.presetK ? Join(ModelBaseDir(), L"K") : ModelBaseDir();
}
#else
int MapResourceId(Context& ctx, bool) { return ctx.data->mapId; }
const wchar_t* MapSha256(Context& ctx, bool) { return ctx.data->mapSha256; }
wstring ModelDir(Context& ctx) { return ctx.presetFG ? Join(ModelBaseDir(), L"FG") : ctx.presetL ? Join(ModelBaseDir(), L"L") : ModelBaseDir(); }
#endif
wchar_t SelectedPreset(const Context& ctx) { return ctx.presetFG ? L'G' : ctx.presetL ? L'L' : ctx.presetK ? L'K' : L'M'; }
wstring PresetName(wchar_t preset) { return preset == L'G' ? wstring(L"FG") : wstring(1, preset); }
bool SelectPreset(Context& ctx, const wstring& name, wstring& error) {
    // frame generation (#96): "FG" by name, in a build with its converter map
    if (name.size() == 2 && towupper(name[0]) == L'F' && towupper(name[1]) == L'G') {
        if (!ctx.data || !ctx.data->fgMapId) {
            error = L"This build of WrongGPU has no converter data for frame generation.";
            return false;
        }
        if (ctx.presetFG) return true;
        ctx.presetFG = true;
        ctx.presetL = false;
#if WRONGGPU_PRESET_K
        ctx.presetK = false;
#endif
        ctx.map = wgc::LoadedMap();  // the other model's map is loaded on first use (LoadEmbeddedMap)
        ctx.mapError.clear();
        return true;
    }
    const wchar_t c = name.size() == 1 ? static_cast<wchar_t>(towupper(name[0])) : L'\0';
#if WRONGGPU_PRESET_K
    const wstring known = L"M, L or K";
    const bool isK = c == L'K';
#else
    const wstring known = L"M or L";
    const bool isK = false;
#endif
    if (c != L'M' && c != L'L' && !isK) {
        error = L"Unknown model \"" + name + L"\" (the models of this build: " + known +
                (ctx.data && ctx.data->fgMapId ? L"; FG: frame generation" : L"") + L").";
        return false;
    }
    if (c == L'L' && (!ctx.data || !ctx.data->lMapId)) {
        error = L"This build of WrongGPU has no converter data for Model L.";
        return false;
    }
    if (c == SelectedPreset(ctx)) return true;
    ctx.presetFG = false;
    ctx.presetL = c == L'L';
#if WRONGGPU_PRESET_K
    ctx.presetK = c == L'K';
#endif
    ctx.map = wgc::LoadedMap();  // the other model's map is loaded on first use (LoadEmbeddedMap)
    ctx.mapError.clear();
    return true;
}
wstring ModelFile(Context& ctx) { return Join(ModelDir(ctx), FromUtf8(wgc::kModelFileName)); }
wstring ModelSourceFile(Context& ctx) { return Join(ModelDir(ctx), L"model-source.txt"); }

// The converter map, loaded and validated once (spec 08 section 5). The launcher's worker threads (the health check, the
// first run) and its window can ask at the same moment: one loads, the others wait for it.
bool LoadEmbeddedMap(Context& ctx) {
    static std::mutex loading;
    std::lock_guard<std::mutex> lock(loading);
    if (ctx.map.map) return true;
    // Model L: its own map, loaded as preset L (the converter refuses a map that names another preset: "converter data
    // damaged"). Models M and K: as before (the map's own preset).
    // Frame generation (#96): its own map, loaded as preset FG.
    const bool l = ctx.presetL, fg = ctx.presetFG;
    if (fg && !ctx.data->fgMapId) { ctx.mapError = L"this build has no converter data for frame generation"; return false; }
    if (l && !ctx.data->lMapId) { ctx.mapError = L"this build has no converter data for Model L"; return false; }
    DWORD sz = 0;
    const void* d = ResourceData(fg ? ctx.data->fgMapId : l ? ctx.data->lMapId : MapResourceId(ctx, ctx.presetK), sz);
    const wchar_t* const want = fg ? ctx.data->fgMapSha256 : l ? ctx.data->lMapSha256 : MapSha256(ctx, ctx.presetK);
    if (!d || Sha256Bytes(d, sz) != want) { ctx.mapError = L"the converter data is missing or damaged"; return false; }
    const wgc::Result r = wgc::load_map_bytes(string(static_cast<const char*>(d), sz), ctx.map,
                                               [&ctx](const string& s) { LogLine(ctx, L"converter: " + FromUtf8(s)); }, fg ? "FG" : l ? "L" : "");
    if (r.code != wgc::kOk) { ctx.mapError = FromUtf8(r.message); return false; }
    return true;
}

Context::~Context() {
    if (!opti.tempDir.empty()) RemoveTree(opti.tempDir);
}

}  // namespace wg
