// SPDX-License-Identifier: MIT
// WrongGPU launcher core: WrongGPU's edition of OptiScaler, downloaded by the launcher from its own project's release page.
// Written for this project.
//
// OptiScaler is a separate program (GPL) with its own project. The launcher never ships it (owner 2026-11-02: nothing of
// the OptiScaler fork goes into WrongGPU's download): it downloads the WrongGPU edition of OptiScaler (OptiScaler with a
// plugin interface WrongGPU's Ray Reconstruction uses, interface 5) from the fork's published release, over HTTPS with
// Windows' own WinHTTP, checks it against the size and SHA-256 pinned here, keeps it in the launcher's data folder
// (optiscaler\) and uses that copy for every install. A download that does not match is deleted and never used. Offline
// or after a failed download the player can still choose an OptiScaler download themselves (the launcher's "Choose your
// OptiScaler download"); a game whose RR needs plugin interface 5 (gamerules.h) then gets no RR.
#pragma once
#include "context.h"

#include <cstdint>
#include <functional>

namespace wg {

struct EditionInfo {
    wstring url;        // the release asset (https://)
    wstring page;       // the release page, shown to the player (Settings, notices)
    wstring file;       // the cached file's name
    unsigned long long size = 0;
    wstring sha256;     // of the archive
    wstring dllSha256;  // of OptiScaler.dll inside it (the edition DLL: plugin interface 5)
    wstring version;    // in words: "10.0.0-pre1 (WrongGPU edition v5)"
};
// The pinned release. WRONGGPU_TESTING builds read WRONGGPU_TEST_EDITION ("url|size|sha256|dllsha256"; a test server on
// 127.0.0.1 may use http://); without it they never download (as if offline), so tests never reach the network unasked.
EditionInfo PinnedEdition();
const wchar_t* const kEditionProject = L"https://github.com/phantoslayer21/OptiScaler";
const wchar_t* const kEditionPage = L"https://github.com/phantoslayer21/OptiScaler/releases/tag/wronggpu-edition-20261102.1";

struct EditionResult {
    wstring path;          // the verified archive in the cache (empty: not available)
    wstring error;         // in words, for the player
    wstring detail;        // the technical cause (log)
    bool cached = false;   // the cache already had it (nothing downloaded)
    bool mismatch = false; // a download arrived but was not the pinned file (deleted)
    bool cancelled = false;
};
// <dataDir>\optiscaler (created when needed).
wstring EditionCacheDir(const wstring& dataDir);
// The verified edition archive: the cached copy when its size and SHA-256 match, else a download (progress(done, total)
// on the worker; returning false cancels). Never throws.
EditionResult GetEdition(Context& ctx, const wstring& dataDir, const std::function<bool(unsigned long long, unsigned long long)>& progress);
// The cached copy is there and matches (size and SHA-256), nothing downloaded.
bool EditionCached(const wstring& dataDir, wstring* path = nullptr);
// OptiScaler.dll in this package folder is the pinned edition DLL.
bool IsEditionDll(const wstring& optiDll);

}  // namespace wg
