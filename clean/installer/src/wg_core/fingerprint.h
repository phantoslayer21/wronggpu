// SPDX-License-Identifier: MIT
// WrongGPU launcher core: fingerprints of the DLLs other tools put next to a game (build/launcher-design.md section
// 5): a file is recognised by what it contains (its version resource, its exports, text in it), never by its name.
// File names only decide which files are looked at (the names Windows loads from a game folder). Also the graphics
// API an executable uses (its imports), for the "not a D3D12 game" rule. Read-only. Written for this project.
#pragma once
#include "util.h"

namespace wg {

enum class Tool {
    Unknown,         // a file none of the rules recognise (never overwritten, never taken as a proxy name)
    Microsoft,       // a Windows component the game ships (its version resource says Microsoft)
    WrongGPU,        // ours (version resource "WrongGPU", or a hash of one of our releases)
    WrongGPULegacy,  // WrongGPU 0.1.x / 0.2.x, published as "DLSS 4.5 AMD"
    OptiScaler,
    ReShade,
    SpecialK,
    Dxvk,            // DXVK (Direct3D 8-11 on Vulkan)
    Vkd3d,           // vkd3d-proton (Direct3D 12 on Vulkan)
    AsiLoader,       // Ultimate ASI Loader
    Cet,             // Cyber Engine Tweaks
    Red4ext,
    Ue4ss,
    HipRuntime,      // a HIP runtime or compiler DLL (another HIP-based tool's)
    AmdFidelityFx,   // AMD's FidelityFX DLL (the game's own)
    FakeNvapi,       // a stand-in nvapi64.dll (OptiScaler's companion)
    VendorDll,          // a DLL whose version resource or signer names NVIDIA (e.g. the game's DLSS DLL copied as nvngx.dll)
    NrOnAmd,         // the "DLSS-NR on AMD" mod
    ReFramework,     // REFramework (a mod framework of RE Engine games): recognised by gamerules.h, it stays
    AntiCheat,       // (rules table only: anti-cheat markers, not a fingerprint)
    Overlay,         // (rules table only: a running overlay or injector)
    GameApi,         // (rules table only: the game's graphics API)
};
const wchar_t* ToolName(Tool t);  // for people: "ReShade", "Ultimate ASI Loader", ...
const char* ToolKey(Tool t);      // for files and JSON: "reshade", "asi-loader", ...
bool ParseToolKey(const string& key, Tool& t);

struct Fingerprint {
    Tool tool = Tool::Unknown;
    bool pe = false;                   // a PE file (else: text or other data)
    wstring version, company, product; // from the version resource (empty without one)
    wstring evidence;                  // why: "version resource: ReShade", "export ReShadeRegisterAddon", "text \"dxvk\""
    wstring Describe() const;          // "ReShade 6.3.3 (version resource: ReShade)"
};
// name: the file's name (IdentifyFile passes it). It never decides a tool by itself; it only limits DXVK to the names
// DXVK loads as (d3d9.dll, d3d10core.dll, d3d11.dll, dxgi.dll) and vkd3d-proton to d3d12.dll / d3d12core.dll, and both
// are recognised only by their own markers (their version resource, or their own settings names such as
// DXVK_CONFIG_FILE / VKD3D_CONFIG), never by a bare "dxvk" in a large file or in another tool's text.
Fingerprint IdentifyBytes(const uint8_t* b, size_t n, const wstring& name = L"");
Fingerprint IdentifyFile(const wstring& path);  // files over 256 MB: Unknown

// Which of `count` (at most 32) lower-case ASCII needles occur in the data, as ASCII or UTF-16LE text, ignoring case:
// bit i set for needle i. One pass over the data.
uint32_t ScanMarkers(const uint8_t* d, size_t n, const char* const* needles, size_t count);

// The graphics API of a game executable: its imported and delay-loaded DLLs, else the D3D12 names in it.
struct GraphicsApi {
    bool pe = false;                 // the exe could be read as a PE file (fixtures and stubs: no verdict)
    bool d3d12 = false;              // links or loads d3d12.dll
    std::vector<wstring> others;     // other APIs it imports: "Direct3D 11", "Vulkan", ...
    wstring evidence;
};
GraphicsApi ExeGraphics(const wstring& exe);

}  // namespace wg
