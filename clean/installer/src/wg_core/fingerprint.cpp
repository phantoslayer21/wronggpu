// SPDX-License-Identifier: MIT
// WrongGPU launcher core: DLL fingerprints and the graphics API of an executable (fingerprint.h). Written for this
// project from Microsoft's public PE/COFF and VS_VERSIONINFO documentation and the tools' public descriptions (product
// names, the text their files carry); no third-party file is read here except the player's own, on the player's PC.
#include "fingerprint.h"

#include "pe.h"

namespace wg {

namespace {
struct ToolInfo { Tool tool; const wchar_t* name; const char* key; };
const ToolInfo kTools[] = {
    {Tool::Unknown, L"an unknown file", "unknown"},
    {Tool::Microsoft, L"Windows component", "microsoft"},
    {Tool::WrongGPU, L"WrongGPU", "wronggpu"},
    {Tool::WrongGPULegacy, L"an older WrongGPU (DLSS 4.5 AMD)", "wronggpu-legacy"},
    {Tool::OptiScaler, L"OptiScaler", "optiscaler"},
    {Tool::ReShade, L"ReShade", "reshade"},
    {Tool::SpecialK, L"Special K", "specialk"},
    {Tool::Dxvk, L"DXVK", "dxvk"},
    {Tool::Vkd3d, L"vkd3d-proton", "vkd3d"},
    {Tool::AsiLoader, L"Ultimate ASI Loader", "asi-loader"},
    {Tool::Cet, L"Cyber Engine Tweaks", "cet"},
    {Tool::Red4ext, L"RED4ext", "red4ext"},
    {Tool::Ue4ss, L"UE4SS", "ue4ss"},
    {Tool::HipRuntime, L"a HIP runtime of another tool", "hip-tool"},
    {Tool::AmdFidelityFx, L"AMD FidelityFX", "amd-fidelityfx"},
    {Tool::FakeNvapi, L"fakenvapi", "fakenvapi"},
    {Tool::VendorDll, L"an NVIDIA DLL", "nvidia"},  // e.g. the DLSS DLL an older OptiScaler layout copies as nvngx.dll
    // another upscaler mod, recognised by its own file names (constants.h kOtherMods, coexist.cpp) and never named in
    // what players read (owner 2026-11-04): its name and key are generic
    {Tool::NrOnAmd, L"Another upscaler mod", "other-upscaler-mod"},
    {Tool::ReFramework, L"REFramework", "reframework"},
    {Tool::AntiCheat, L"anti-cheat", "anti-cheat"},
    {Tool::Overlay, L"an overlay", "overlay"},
    {Tool::GameApi, L"the game's graphics API", "graphics-api"},
};
wstring LowerW(wstring s) { for (auto& c : s) c = (wchar_t)towlower(c); return s; }

// Content markers, most specific first: OptiScaler's own files mention ReShade, Special K and DXVK (it detects them),
// Special K's mention ReShade; so the first marker found in this order decides. A bare "dxvk" or "vkd3d" is not a
// marker: four or five letters that a large binary holds by chance and that other tools name (see the API markers).
const char* const kMarkers[] = {
    "optiscaler",          // 0
    "ue4ss",               // 1
    "red4ext",             // 2
    "cyber_engine_tweaks", // 3
    "cyber engine tweaks", // 4
    "ultimate asi loader", // 5
    "thirteenag",          // 6  (the ASI loader's author)
    "specialk",            // 7
    "reshade",             // 8
    "fakenvapi",           // 9
    "dlssnr_on_amd",       // 10 (the DLSS-NR on AMD mod)
    "dlss45",              // 11 (our own older mod, DLSS 4.5 AMD)
    // DXVK's and vkd3d-proton's own settings names (environment variables and config files their DLLs read): counted
    // only under the DLL names they load as (ApiTool) and after every marker above (OptiScaler names DXVK's settings too)
    "dxvk_config_file",    // 12
    "dxvk_log_level",      // 13
    "dxvk_hud",            // 14
    "dxvk_state_cache",    // 15
    "dxvk.conf",           // 16
    "vkd3d_config",        // 17
    "vkd3d_debug",         // 18
    "vkd3d-proton",        // 19
    "vkd3d_shader_debug",  // 20
    "wronggpu",            // 21 (ours: PE files only, last)
};
const size_t kMarkerCount = sizeof(kMarkers) / sizeof(kMarkers[0]);
const size_t kGenericMarkers = 12, kApiMarkersEnd = 21;  // [0, 12): any file; [12, 21): DXVK / vkd3d-proton; 21: ours
const Tool kMarkerTool[] = {Tool::OptiScaler, Tool::Ue4ss, Tool::Red4ext, Tool::Cet, Tool::Cet, Tool::AsiLoader, Tool::AsiLoader, Tool::SpecialK,
                            Tool::ReShade, Tool::FakeNvapi, Tool::NrOnAmd, Tool::WrongGPULegacy,
                            Tool::Dxvk, Tool::Dxvk, Tool::Dxvk, Tool::Dxvk, Tool::Dxvk, Tool::Vkd3d, Tool::Vkd3d, Tool::Vkd3d, Tool::Vkd3d, Tool::WrongGPU};
static_assert(sizeof(kMarkers) / sizeof(kMarkers[0]) == sizeof(kMarkerTool) / sizeof(kMarkerTool[0]), "one tool per marker");
static_assert(sizeof(kMarkers) / sizeof(kMarkers[0]) <= 32, "ScanMarkers takes at most 32 needles");

// The DLL names DXVK (Direct3D 9-11 on Vulkan) and vkd3d-proton (Direct3D 12 on Vulkan) load as: only these can be them.
const wchar_t* const kDxvkNames[] = {L"d3d9.dll", L"d3d10core.dll", L"d3d11.dll", L"dxgi.dll"};
const wchar_t* const kVkd3dNames[] = {L"d3d12.dll", L"d3d12core.dll"};
// Which of the two a file of this name could be (Unknown: neither).
Tool ApiTool(const wstring& name) {
    const wstring n = LowerW(name);
    for (auto x : kDxvkNames) if (n == x) return Tool::Dxvk;
    for (auto x : kVkd3dNames) if (n == x) return Tool::Vkd3d;
    return Tool::Unknown;
}
// The signer's name in the file's Authenticode certificate table (PE data directory 4: a file offset, not an RVA), as the
// plain text the certificates carry.
bool SignerMentions(const PeImage& img, const char* needle) {
    if (!img.ok || img.ndd <= 4 || img.ddOff + 5 * 8 > img.n) return false;
    const size_t off = img.u32(img.ddOff + 4 * 8), size = img.u32(img.ddOff + 4 * 8 + 4);
    if (!off || !size || off >= img.n) return false;
    return ScanMarkers(img.b + off, std::min(size, img.n - off), &needle, 1) != 0;
}
const char* const kVendorSigner = "nvidia corporation";  // the signer of the DLSS DLL (what older OptiScaler layouts copy as nvngx.dll)

// Product words of the version resource (CompanyName, ProductName, FileDescription, InternalName, OriginalFilename).
struct ResourceWord { const wchar_t* word; Tool tool; };
const ResourceWord kResourceWords[] = {
    {L"wronggpu", Tool::WrongGPU}, {L"dlss 4.5 amd", Tool::WrongGPULegacy}, {L"dlss45", Tool::WrongGPULegacy},
    {L"optiscaler", Tool::OptiScaler}, {L"reshade", Tool::ReShade}, {L"special k", Tool::SpecialK}, {L"specialk", Tool::SpecialK},
    {L"ultimate asi loader", Tool::AsiLoader}, {L"cyber engine tweaks", Tool::Cet}, {L"red4ext", Tool::Red4ext}, {L"ue4ss", Tool::Ue4ss},
    {L"fakenvapi", Tool::FakeNvapi},
};
}  // namespace

const wchar_t* ToolName(Tool t) { for (auto& i : kTools) if (i.tool == t) return i.name; return L"?"; }
const char* ToolKey(Tool t) { for (auto& i : kTools) if (i.tool == t) return i.key; return "?"; }
bool ParseToolKey(const string& key, Tool& t) {
    for (auto& i : kTools) if (key == i.key) { t = i.tool; return true; }
    if (key == "nr-on-amd") { t = Tool::NrOnAmd; return true; }  // the key install records before 0.6.0 rc3 hold
    return false;
}

wstring Fingerprint::Describe() const {
    wstring s = ToolName(tool);
    if (!version.empty() && tool != Tool::Unknown && tool != Tool::Microsoft) s += L" " + version;
    if (tool == Tool::Unknown && !company.empty()) s += L" (" + company + L")";
    if (!evidence.empty()) s += L" [" + evidence + L"]";
    return s;
}

uint32_t ScanMarkers(const uint8_t* d, size_t n, const char* const* needles, size_t count) {
    uint32_t found = 0;
    if (!d || !n || !count) return 0;
    count = std::min<size_t>(count, 32);
    const uint32_t all = count == 32 ? 0xFFFFFFFFu : ((1u << count) - 1);
    std::vector<uint8_t> len(count);
    std::vector<uint32_t> byFirst(256, 0);  // needles starting with each (lower-case) byte
    for (size_t k = 0; k < count; ++k) {
        len[k] = (uint8_t)std::min<size_t>(strlen(needles[k]), 255);
        if (len[k]) byFirst[(uint8_t)needles[k][0]] |= 1u << k;
    }
    auto low = [](uint8_t c) { return (uint8_t)(c >= 'A' && c <= 'Z' ? c + 32 : c); };
    for (size_t i = 0; i < n && found != all; ++i) {
        uint32_t cand = byFirst[low(d[i])] & ~found;
        while (cand) {
            const unsigned k = [&] { unsigned b = 0; while (!(cand & (1u << b))) ++b; return b; }();
            cand &= ~(1u << k);
            const size_t m = len[k];
            const char* nd = needles[k];
            if (i + m <= n) {
                size_t j = 1;
                while (j < m && low(d[i + j]) == (uint8_t)nd[j]) ++j;
                if (j == m) { found |= 1u << k; continue; }
            }
            if (i + 2 * m <= n && d[i + 1] == 0) {  // UTF-16LE
                size_t j = 1;
                while (j < m && low(d[i + 2 * j]) == (uint8_t)nd[j] && d[i + 2 * j + 1] == 0) ++j;
                if (j == m) found |= 1u << k;
            }
        }
    }
    return found;
}

Fingerprint IdentifyBytes(const uint8_t* b, size_t n, const wstring& name) {
    Fingerprint f;
    if (!b || !n) { f.evidence = L"empty file"; return f; }
    const PeImage img(b, n);
    const Tool api = ApiTool(name);  // DXVK or vkd3d-proton: possible only under their own DLL names
    const wstring as = name.empty() ? wstring() : L" (as " + name + L")";
    f.pe = img.ok;
    if (f.pe) {
        const PeVersion v = ReadPeVersion(b, n);
        f.version = v.fileVersion;
        f.company = v.Get(L"CompanyName");
        f.product = v.Get(L"ProductName");
        const wstring words = LowerW(f.company + L"|" + f.product + L"|" + v.Get(L"FileDescription") + L"|" + v.Get(L"InternalName") + L"|" + v.Get(L"OriginalFilename"));
        for (auto& w : kResourceWords)
            if (words.find(w.word) != wstring::npos) { f.tool = w.tool; f.evidence = wstring(L"version resource: ") + (f.product.empty() ? f.company : f.product); return f; }
        if ((api == Tool::Dxvk && words.find(L"dxvk") != wstring::npos) || (api == Tool::Vkd3d && words.find(L"vkd3d") != wstring::npos)) {
            f.tool = api; f.evidence = L"version resource: " + (f.product.empty() ? f.company : f.product) + as; return f;
        }
        // the vendor's own DLL (the DLSS DLL that an older OptiScaler layout copies as nvngx.dll): its version resource or signer
        if (words.find(L"nvidia") != wstring::npos) { f.tool = Tool::VendorDll; f.evidence = L"version resource: " + (f.product.empty() ? f.company : f.product); return f; }  // nvngx / DLSS
        if (SignerMentions(img, kVendorSigner)) { f.tool = Tool::VendorDll; f.evidence = L"signed by the DLSS DLL's vendor (nvngx signer)"; return f; }
        const std::set<string> ex = ReadPeExports(b, n);
        if (ex.count("ReShadeRegisterAddon")) { f.tool = Tool::ReShade; f.evidence = L"export ReShadeRegisterAddon"; return f; }
        if (ex.count("hipInit") || ex.count("hipMalloc") || ex.count("amd_comgr_get_version")) {
            f.tool = Tool::HipRuntime; f.evidence = ex.count("amd_comgr_get_version") ? L"export amd_comgr_get_version" : L"export hipInit"; return f;
        }
        const wstring company = LowerW(f.company);
        if (company.find(L"advanced micro devices") != wstring::npos && ex.count("ffxCreateContext")) { f.tool = Tool::AmdFidelityFx; f.evidence = L"AMD version resource, FidelityFX exports"; return f; }
        if (company.find(L"microsoft") != wstring::npos) { f.tool = Tool::Microsoft; f.evidence = L"version resource: " + f.company; return f; }
    }
    const uint32_t m = ScanMarkers(b, n, kMarkers, kMarkerCount);
    for (size_t k = 0; k < kGenericMarkers; ++k)
        if (m & (1u << k)) { f.tool = kMarkerTool[k]; f.evidence = L"text \"" + FromUtf8(kMarkers[k]) + L"\""; return f; }
    for (size_t k = kGenericMarkers; k < kApiMarkersEnd; ++k)
        if ((m & (1u << k)) && kMarkerTool[k] == api) { f.tool = api; f.evidence = L"text \"" + FromUtf8(kMarkers[k]) + L"\"" + as; return f; }
    if (f.pe && (m & (1u << kApiMarkersEnd))) { f.tool = Tool::WrongGPU; f.evidence = L"text \"wronggpu\""; return f; }
    f.evidence = f.pe ? (f.company.empty() ? wstring(L"no version resource, no known text") : L"version resource: " + f.company) : L"not a PE file";
    return f;
}

Fingerprint IdentifyFile(const wstring& path) {
    MappedFile f(path, 256ull << 20);
    if (!f.data) { Fingerprint r; r.evidence = IsFile(path) ? L"cannot be read (or larger than 256 MB)" : L"missing"; return r; }
    return IdentifyBytes(f.data, f.size, BaseName(path));
}

GraphicsApi ExeGraphics(const wstring& exe) {
    GraphicsApi g;
    MappedFile f(exe, 1ull << 31);
    if (!f.data) { g.evidence = L"the exe cannot be read"; return g; }
    const PeImage img(f.data, f.size);
    if (!img.ok) { g.evidence = L"not a PE file"; return g; }
    g.pe = true;
    const std::set<string> im = ReadPeImports(f.data, f.size);
    struct Api { const char* dll; const wchar_t* name; };
    static const Api kApis[] = {{"d3d11.dll", L"Direct3D 11"}, {"d3d10.dll", L"Direct3D 10"}, {"d3d10_1.dll", L"Direct3D 10"}, {"d3d9.dll", L"Direct3D 9"},
                                {"d3d8.dll", L"Direct3D 8"}, {"ddraw.dll", L"DirectDraw"}, {"vulkan-1.dll", L"Vulkan"}, {"opengl32.dll", L"OpenGL"}};
    if (im.count("d3d12.dll")) { g.d3d12 = true; g.evidence = L"imports d3d12.dll"; return g; }
    for (auto& a : kApis)
        if (im.count(a.dll) && std::find(g.others.begin(), g.others.end(), a.name) == g.others.end()) g.others.push_back(a.name);
    // Loaded at run time (LoadLibrary): the DLL name or its entry point is in the file.
    const char* const d3d12Names[] = {"d3d12.dll", "d3d12createdevice"};
    if (ScanMarkers(f.data, f.size, d3d12Names, 2)) { g.d3d12 = true; g.evidence = L"loads d3d12.dll at run time"; return g; }
    wstring list;
    for (auto& o : g.others) list += (list.empty() ? L"" : L", ") + o;
    g.evidence = g.others.empty() ? L"no graphics API in its imports" : L"imports " + list + L", no d3d12.dll";
    return g;
}

}  // namespace wg
