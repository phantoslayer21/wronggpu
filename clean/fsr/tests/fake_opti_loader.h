// SPDX-License-Identifier: MIT
#pragma once
// A loader of our own that behaves like OptiScaler 10's custom upscaler folder as spec 65 describes it (65.1, 65.2,
// 65.6), for the FFX slot: the DLL custom\<folder>\ffx\<FFX DX12 upscaler file name> is loaded once per folder, from
// its full path (never the search path), kept for the life of the process; its upscaler versions are asked through
// the FFX API (QUERY_GET_VERSIONS); the version is chosen by UpscalerIndex in the OptiScaler.ini section named after
// the folder; the upscaler context is created with that version id (an override-version description in the create
// chain) and released and created again when the choice changes. Written from the spec's description only: no
// OptiScaler code or binary is used.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>
#include <vector>

#include "../src/ffx_headers.h"

struct FakeOptiLoader {
    static constexpr const wchar_t* kSlotFile = L"amd_fidelityfx_upscaler_dx12.dll";  // the FFX DX12 upscaler file name

    std::wstring root;    // the game folder: OptiScaler.ini and the OptiScaler folder (OptiDllPath) are here
    std::wstring folder;  // the custom folder's name = its list entry = its ini section
    HMODULE module = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    std::vector<uint64_t> ids;
    std::vector<std::string> names;
    ffxContext context = nullptr;
    uint64_t contextVersion = 0;

    std::wstring SlotDir() const { return root + L"\\OptiScaler\\custom\\" + folder + L"\\ffx"; }
    std::wstring SlotDll() const { return SlotDir() + L"\\" + kSlotFile; }
    std::wstring HostIni() const { return root + L"\\OptiScaler.ini"; }

    // Loads the slot DLL by full path (its own folder first for its dependencies) and resolves the FFX entry points.
    bool Load();
    // QUERY_GET_VERSIONS (count, then ids and names) with the game's device, as the host builds its version box.
    bool QueryVersions(void* device);
    // [<folder>] UpscalerIndex of OptiScaler.ini; "auto", missing or not a number: -1.
    int IndexFromIni() const;
    // Writes [<folder>] UpscalerIndex (what the host does when the player saves its ini).
    bool WriteIndex(int index) const;
    // The player picked version `index`: the current upscaler context is released, a new one is created with that
    // version (override-version in the chain), the game's D3D12 device and the given sizes.
    ffxReturnCode_t Select(int index, void* device, uint32_t renderW, uint32_t renderH, uint32_t outW, uint32_t outH);
    // GET_PROVIDER_VERSION of the current context ("" without one).
    std::string ProviderName(uint64_t* id = nullptr);
    void Release();
};
