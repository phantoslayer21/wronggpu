// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the Context that the modules get instead of globals (what were setup.cpp's g_logFile,
// g_presetK, g_map, g_opti, g_noLibraryScan and the embedded payload table), the Ui callbacks of a long operation, the
// log file and the model data folder. Written for this project.
#pragma once
#include <functional>
#include <memory>
#include <mutex>

#include "api.h"     // the converter (clean/converter): wgc::LoadedMap
#include "output.h"  // wgc::kModelFileName
#include "util.h"

namespace wg {

// One file of the embedded payload (the table is written by tools/make_payload.py).
struct PayloadFile { int id; const wchar_t* rel; const wchar_t* sha256; unsigned long long size; };
// What the executable embeds (app/payload_embed.cpp fills it in from the generated payload_table.h): the payload
// files, the converter maps (Preset K's only in a build with WRONGGPU_PRESET_K), the wordmark and the version.
struct EmbeddedData {
    const PayloadFile* files = nullptr;
    size_t count = 0;
    int mapId = 0;
    const wchar_t* mapSha256 = L"";
    int kMapId = 0;
    const wchar_t* kMapSha256 = L"";
    int logoId = 0;
    const wchar_t* version = L"";
    const wchar_t* buildId = L"";      // the build's own name (0.5.5-launcher-dev); the version when not a pre-release
    const wchar_t* payloadName = L"";  // the release build the payload comes from (0.5.5-ea)
    // Ray Reconstruction (rr.h): the RR plugin and runtime (relative to the payload's RR folder: plugins\WrongGPU-RR.asi,
    // wronggpu_rr_runtime.dll) and the RR converter map (preset F). rrCount 0 / fMapId 0: not in this build.
    const PayloadFile* rrFiles = nullptr;
    size_t rrCount = 0;
    int fMapId = 0;
    const wchar_t* fMapSha256 = L"";
    // the file name of the game interface the RR plugin hooks (the RR folder's interposer.json; empty: none given): which
    // build of the plugin a game gets follows its version (rr.h, RrPluginFor)
    const wchar_t* rrInterface = L"";
    const PayloadFile* begin() const { return files; }
    const PayloadFile* end() const { return files + count; }
};

// ------------------------------------------------------------------------------------------------
// Logging (file + optional UI sink). The log lives in %LOCALAPPDATA%\WrongGPU\setup.log unless /LOG= is given; a
// copy is kept in <install dir>\wronggpu-setup\setup.log after a successful install.
// ------------------------------------------------------------------------------------------------
struct Ui {
    std::function<void(int, int, const wstring&)> step;
    std::function<void(const wstring&)> log;
};

// The player's OptiScaler package of this run (optiscaler.h, PrepareOptiScaler).
struct OptiPackage {
    wstring source, dir, version, tempDir, error;
    std::vector<wstring> files;  // relative to dir: what goes next to the game exe (OptiScaler.dll renamed to the proxy)
    bool edition = false;        // its OptiScaler.dll is WrongGPU's edition of OptiScaler (optiedition.h: the pinned DLL, plugin interface 5)
    bool chosen() const { return !source.empty(); }
    bool ok() const { return chosen() && error.empty() && !dir.empty(); }
};

// Everything the modules share for one run of the program. One Context per process (the wizard's or the silent
// run's, later the launcher's); passed by reference, never copied.
struct Context {
    const EmbeddedData* data = nullptr;  // set before anything else (app/payload_embed.cpp)
    wstring logFile;                     // /LOG= or %LOCALAPPDATA%\WrongGPU\setup.log; empty: no log
    std::mutex logMutex;
#if WRONGGPU_PRESET_K
    bool presetK = false;                // Preset K starts first (/PRESET=K, the wizard's choice)
#else
    static constexpr bool presetK = false;  // CMake WRONGGPU_PRESET_K=OFF: Model M only
#endif
    // The converter map of the selected preset, loaded and validated once (spec 08 section 5; LoadEmbeddedMap).
    wgc::LoadedMap map;
    wstring mapError;
    // The RR converter map (preset F), loaded once when the RR data is looked for (rr.h, LoadFMap).
    wgc::LoadedMap fMap;
    wstring fMapError;
    OptiPackage opti;                    // the player's OptiScaler package (/OPTISCALER=, the wizard's choice)
    bool noLibraryScan = false;          // /NOSCAN: the store libraries are not searched
    // The launcher (build/launcher-design.md): one Apps & features entry for all games instead of one per game; the
    // HIP runtime is optional (D3D12 is the default executor on RX 9000; a missing HIP runtime is a warning); the
    // uninstaller written into each game is the small restorer the launcher embeds (WrongGPU-Restore.exe).
    bool perGameEntry = true;
    bool hipOptional = false;
    bool hostedLayout = false;           // games are inspected for the OptiScaler layout (install folder = exe folder) without a package chosen yet
    const void* restorer = nullptr;
    size_t restorerSize = 0;
    // The player's Documents folder (rrconfig.h): the elevated worker gets it from its job (it may run as another account);
    // empty: the Documents known folder of this process's user.
    wstring playerDocuments;
    // The ignore lists of the converter's file search (convert_ignore.h): the launcher's own data folder (its well-known list
    // when that is not %LOCALAPPDATA%\WrongGPU: a portable launcher's WrongGPU-data, a --data= folder), and this run's
    // command line (--ignore-list=<file>, --ignore=<entry>), added to the well-known lists.
    wstring ownDataDir;
    std::vector<wstring> ignoreListFiles, ignoreEntries;
    // ignoreListBytes[i]: the bytes of ignoreListFiles[i] when the command line read them already (once, with the bounded
    // reader: convert_ignore.h ReadIgnoreListBytes); a file without them is read when the search loads the lists.
    std::vector<string> ignoreListBytes;
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context();                          // removes the OptiScaler package's temporary folder
};

void LogLine(Context& ctx, const wstring& s);
wstring AppDataPath();  // %LOCALAPPDATA%\WrongGPU, not created
wstring AppDataDir();   // the same, created if missing
wstring DefaultLogPath();
wstring ModelBaseDir();
int MapResourceId(Context& ctx, bool k);
const wchar_t* MapSha256(Context& ctx, bool k);
wstring ModelDir(Context& ctx);
wstring ModelFile(Context& ctx);
wstring ModelSourceFile(Context& ctx);
bool LoadEmbeddedMap(Context& ctx);

}  // namespace wg
