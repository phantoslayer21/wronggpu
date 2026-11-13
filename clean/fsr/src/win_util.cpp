// SPDX-License-Identifier: MIT
#include "win_util.h"

#include <d3d12.h>
#include <winver.h>

#include <cstdio>
#include <cstring>
#include <cwctype>
#include <mutex>

#include "amd_provider.h"
#include "settings_bridge.h"
#include "wg_settings_io.h"
#include "standin_d3d12.h"
#include "provider_label.h"

namespace {
wgfsr::ProviderLabel activeProviderLabel;
}

// Optional host UI contract; buffer belongs to the caller, no pointers into a context escape.
extern "C" __declspec(dllexport) uint32_t wgGetActiveProviderLabel(char* output, uint32_t capacity) {
    try { return activeProviderLabel.Copy(output, capacity); } catch (...) { if (output && capacity) output[0] = 0; return 0; }
}

namespace wgwin {

namespace {
wronggpu::Log& L() { return wronggpu::Log::Get(); }
}  // namespace

void DiagnosticProvider(const wgfsr::UpscaleStatus& status) {
    try {
    activeProviderLabel.Update(status);
    // Sparse CPU-side provider observations. A successful API return alone is
    // deliberately not presented as proof of GPU completion or neural quality.
    static std::atomic<int> last{-1};
    static std::atomic<uint64_t> frames{0};
    const int state = int(status.state);
    const bool changed = last.exchange(state) != state;
    const uint64_t frame = ++frames;
    if (!changed && frame % 300 && status.state != wgfsr::UpscaleState::Failed) return;
    std::string data = "{\"state\":" + std::to_string(state) + ",\"provider\":" + wgdiag::Quote(status.upscaler) +
        ",\"reason\":" + wgdiag::Quote(status.reason) + ",\"render\":[" + std::to_string(status.renderWidth) + "," +
        std::to_string(status.renderHeight) + "],\"output\":[" + std::to_string(status.outputWidth) + "," +
        std::to_string(status.outputHeight) + "],\"gpu_completion_proven\":false";
    if (status.device && changed) {
        const LUID luid = static_cast<ID3D12Device*>(status.device)->GetAdapterLuid();
        data += ",\"adapter_luid_low\":" + std::to_string(luid.LowPart) + ",\"adapter_luid_high\":" + std::to_string(luid.HighPart);
    }
    if (status.device && state == int(wgfsr::UpscaleState::Failed)) {
        const HRESULT hr = static_cast<ID3D12Device*>(status.device)->GetDeviceRemovedReason();
        data += ",\"device_removed_hresult\":" + std::to_string(uint32_t(hr));
    }
    L().Event("provider", data + "}");
    } catch (...) {} // Support recording is observational and never changes the host result.
}

void DiagnosticFailure(uint64_t context, uint64_t frame, void* device) {
    try {
    if (!device) return;
    const HRESULT hr = static_cast<ID3D12Device*>(device)->GetDeviceRemovedReason();
    L().Event("device_failure_observation", "{\"context\":" + std::to_string(context) + ",\"frame\":" +
        std::to_string(frame) + ",\"device_removed_hresult\":" + std::to_string(uint32_t(hr)) + ",\"root_cause_proven\":false}");
    } catch (...) {}
}

std::string Narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

std::wstring Widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), &w[0], n);
    return w;
}

std::vector<std::string> ExportNames(HMODULE module) {
    std::vector<std::string> names;
    auto* base = reinterpret_cast<const unsigned char*>(module);
    auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return names;
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return names;
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!dir.VirtualAddress || !dir.Size) return names;
    auto* exports = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + dir.VirtualAddress);
    auto* nameRvas = reinterpret_cast<const DWORD*>(base + exports->AddressOfNames);
    for (DWORD i = 0; i < exports->NumberOfNames && i < 4096; ++i) names.emplace_back(reinterpret_cast<const char*>(base + nameRvas[i]));
    return names;
}

namespace {

// Walks the VERSIONINFO resource of a file loaded as data. `fixed` receives the VS_FIXEDFILEINFO when present;
// `strings` the UTF-16 text of the StringFileInfo block (searched for keys by the callers).
bool ReadVersionResource(const std::wstring& path, VS_FIXEDFILEINFO* fixed, std::wstring* strings) {
    HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (!m) return false;
    bool found = false;
    if (HRSRC r = FindResourceW(m, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16))) {  // RT_VERSION
        if (HGLOBAL g = LoadResource(m, r)) {
            auto* p = static_cast<const unsigned char*>(LockResource(g));
            const DWORD size = SizeofResource(m, r);
            if (p && strings) strings->assign(reinterpret_cast<const wchar_t*>(p), size / 2);
            for (DWORD i = 0; p && fixed && i + sizeof(VS_FIXEDFILEINFO) <= size; i += 4) {
                auto* info = reinterpret_cast<const VS_FIXEDFILEINFO*>(p + i);
                if (info->dwSignature == 0xFEEF04BD) {
                    *fixed = *info;
                    found = true;
                    break;
                }
            }
            if (strings) found = true;
        }
    }
    FreeLibrary(m);
    return found;
}

std::string StringValue(const std::wstring& block, const wchar_t* key) {
    const size_t at = block.find(key);
    if (at == std::wstring::npos) return {};
    size_t v = at + wcslen(key);
    while (v < block.size() && block[v] == L'\0') ++v;  // key terminator and padding
    const size_t end = block.find(L'\0', v);
    return Narrow(block.substr(v, end == std::wstring::npos ? std::wstring::npos : end - v));
}

}  // namespace

std::string FileVersion(const std::wstring& path) {
    VS_FIXEDFILEINFO info{};
    if (!ReadVersionResource(path, &info, nullptr)) return "unknown";
    char text[64];
    std::snprintf(text, sizeof text, "%u.%u.%u.%u", HIWORD(info.dwFileVersionMS), LOWORD(info.dwFileVersionMS),
                  HIWORD(info.dwFileVersionLS), LOWORD(info.dwFileVersionLS));
    return text;
}

std::string FileDescription(const std::wstring& path) {
    std::wstring block;
    if (!ReadVersionResource(path, nullptr, &block)) return {};
    std::string product = StringValue(block, L"ProductName"), description = StringValue(block, L"FileDescription");
    if (product.empty()) return description;
    return description.empty() ? product : product + " / " + description;
}

HMODULE LoadOriginalNextToUs(const wchar_t* fileName, HMODULE self, const std::vector<std::string>& known, const std::string& tag) {
    const std::wstring path = wronggpu::ModuleDirectory() + L"\\" + fileName;
    HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m) {
        L().WriteLine(tag + "AMD's original DLL not loaded: " + Narrow(path) + " (error " + std::to_string(GetLastError()) +
                      "). Calls that need it fail.");
        return nullptr;
    }
    if (m == self) {
        L().WriteLine(tag + "ERROR: " + Narrow(path) + " is a copy of WrongGPU's own DLL, not AMD's; it is not used");
        return nullptr;
    }
    const std::string description = FileDescription(path);
    L().WriteLine(tag + "AMD's original DLL loaded: " + Narrow(path) + ", file version " + FileVersion(path) +
                  (description.empty() ? "" : ", \"" + description + "\""));
    std::string extra;
    for (const std::string& name : ExportNames(m)) {
        bool ours = false;
        for (const std::string& k : known) ours = ours || k == name;
        if (!ours) extra += " " + name;
    }
    if (!extra.empty()) L().WriteLine(tag + "note: AMD's DLL has exports WrongGPU does not provide:" + extra);
    return m;
}

// The FSR 4 DLL the installer found on this PC (AMD's FidelityFX SDK 2 upscaler DLL of another game, version 4 or newer,
// signed by AMD; WrongGPU never ships or copies it): line "fsr4<TAB><path>" (UTF-8) of
// %LOCALAPPDATA%\WrongGPU\compare.txt. Empty when there is none.
std::wstring InstallerCompareDll() {
    wchar_t base[MAX_PATH] = {};
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", base, MAX_PATH);
    if (!n || n >= MAX_PATH) return std::wstring();
    const std::wstring file = std::wstring(base) + L"\\WrongGPU\\compare.txt";
    HANDLE h = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return std::wstring();
    char buf[4096] = {};
    DWORD got = 0;
    if (!ReadFile(h, buf, sizeof buf - 1, &got, nullptr)) got = 0;
    CloseHandle(h);
    const std::string text(buf, got);
    const size_t p = text.find("fsr4\t");
    if (p == std::string::npos) return std::wstring();
    size_t e = text.find_first_of("\r\n", p);
    if (e == std::string::npos) e = text.size();
    return Widen(text.substr(p + 5, e - p - 5));
}

namespace {

// The file behind a path: volume and file index (the same file under another path or name compares equal).
bool FileIdentity(const std::wstring& path, BY_HANDLE_FILE_INFORMATION& info) {
    HANDLE h = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    const bool ok = GetFileInformationByHandle(h, &info) != 0;
    CloseHandle(h);
    return ok;
}

bool SameFile(const std::wstring& a, const std::wstring& b) {
    BY_HANDLE_FILE_INFORMATION x{}, y{};
    return FileIdentity(a, x) && FileIdentity(b, y) && x.dwVolumeSerialNumber == y.dwVolumeSerialNumber &&
           x.nFileIndexHigh == y.nFileIndexHigh && x.nFileIndexLow == y.nFileIndexLow;
}

std::wstring ModulePath(HMODULE m) {
    std::wstring path(32768, L'\0');
    const DWORD n = GetModuleFileNameW(m, &path[0], DWORD(path.size()));
    path.resize(n < path.size() ? n : 0);
    return path;
}

std::wstring BaseNameLower(const std::wstring& path) {
    const size_t slash = path.find_last_of(L"\\/");
    std::wstring name = path.substr(slash == std::wstring::npos ? 0 : slash + 1);
    for (wchar_t& c : name) c = wchar_t(towlower(c));
    return name;
}

HMODULE SelfModule() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&SelfModule), &self);
    return self;
}

// Where the first instruction at `code` jumps to when it is a hook's jump (jmp rel32 / rel8, jmp [rip+disp32],
// mov rax or r11 with an address + jmp to it), else null.
const unsigned char* JumpTarget(const unsigned char* code) {
    if (code[0] == 0xE9) return code + 5 + *reinterpret_cast<const int32_t*>(code + 1);
    if (code[0] == 0xEB) return code + 2 + static_cast<int8_t>(code[1]);
    if (code[0] == 0xFF && code[1] == 0x25) {
        const unsigned char* slot = code + 6 + *reinterpret_cast<const int32_t*>(code + 2);
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(slot, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
            slot + sizeof(void*) > static_cast<const unsigned char*>(mbi.BaseAddress) + mbi.RegionSize)
            return nullptr;
        return *reinterpret_cast<const unsigned char* const*>(slot);
    }
    if (code[0] == 0x48 && code[1] == 0xB8 && code[10] == 0xFF && code[11] == 0xE0)
        return *reinterpret_cast<const unsigned char* const*>(code + 2);
    if (code[0] == 0x49 && code[1] == 0xBB && code[10] == 0x41 && code[11] == 0xFF && code[12] == 0xE3)
        return *reinterpret_cast<const unsigned char* const*>(code + 2);
    return nullptr;
}

// "name.dll" of the module at `address`; empty when it is in no module (a hook's relay memory).
std::wstring ModuleNameAt(const void* address) {
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(address), &m) || !m)
        return std::wstring();
    return BaseNameLower(ModulePath(m));
}

// Modules a host's LoadLibrary hook hands out instead of AMD's DLL (OptiScaler under a proxy name, WrongGPU's hosted
// name).
bool IsHostModuleName(const std::wstring& lowerName) {
    for (const wchar_t* n : {L"optiscaler.dll", L"optiscaler.asi", L"dxgi.dll", L"winmm.dll", L"version.dll", L"dbghelp.dll",
                             L"wininet.dll", L"winhttp.dll", L"d3d12.dll", L"wronggpu_ffx_dx12.dll"})
        if (lowerName == n) return true;
    return false;
}

// The game's FidelityFX SDK 2 DLLs: amd_fidelityfx_upscaler_dx12.dll, then amd_fidelityfx_loader_dx12.dll, in the
// exe folder and its subfolders (3 levels, shallower first; reparse points are not followed). Folders in `skip` (full
// paths, any case) are not entered: OptiScaler's custom and plugins folders when this DLL runs in a custom slot.
std::vector<std::wstring> GameSdkDlls(const std::wstring& exeDir, const std::vector<std::wstring>& skip) {
    std::vector<std::wstring> upscalers, loaders;
    std::vector<std::wstring> level{exeDir};
    size_t visited = 0;
    for (int depth = 0; depth <= 3 && !level.empty() && visited < 4000; ++depth) {
        std::vector<std::wstring> next;
        for (const std::wstring& dir : level) {
            if (++visited > 4000) break;
            WIN32_FIND_DATAW fd;
            HANDLE h = FindFirstFileExW((dir + L"\\*").c_str(), FindExInfoBasic, &fd, FindExSearchNameMatch, nullptr,
                                        FIND_FIRST_EX_LARGE_FETCH);
            if (h == INVALID_HANDLE_VALUE) continue;
            do {
                const std::wstring name = fd.cFileName;
                if (name == L"." || name == L"..") continue;
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    const std::wstring sub = dir + L"\\" + name;
                    bool skipped = false;
                    for (const std::wstring& s : skip) skipped = skipped || _wcsicmp(s.c_str(), sub.c_str()) == 0;
                    if (!skipped && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) next.push_back(sub);
                    continue;
                }
                const std::wstring lower = BaseNameLower(name);
                if (lower == L"amd_fidelityfx_upscaler_dx12.dll") upscalers.push_back(dir + L"\\" + name);
                else if (lower == L"amd_fidelityfx_loader_dx12.dll") loaders.push_back(dir + L"\\" + name);
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        level.swap(next);
    }
    upscalers.insert(upscalers.end(), loaders.begin(), loaders.end());
    return upscalers;
}

}  // namespace

wgfsr::ProbeResult CheckProviderModule(HMODULE m, const std::wstring& requested, std::string& detail) {
    if (!m) return wgfsr::ProbeResult::NotLoaded;
    if (m == SelfModule()) {  // e.g. OptiScaler hands its FidelityFX upscaler DLL name to WrongGPU's DLL
        detail = "loading it gave WrongGPU's own DLL";
        return wgfsr::ProbeResult::Ours;
    }
    const std::wstring got = ModulePath(m);
    if (IsHostModuleName(BaseNameLower(got)) || !SameFile(got, requested)) {
        detail = "loading it gave " + Narrow(got.empty() ? std::wstring(L"an unnamed module") : got);
        return wgfsr::ProbeResult::Redirected;
    }
    if (FileDescription(got).find("WrongGPU") != std::string::npos) {
        detail = "WrongGPU's DLL under another name";
        return wgfsr::ProbeResult::Ours;
    }
    auto* base = reinterpret_cast<const unsigned char*>(m);
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const unsigned char* end = base + nt->OptionalHeader.SizeOfImage;
    for (const char* name : {"ffxCreateContext", "ffxDestroyContext", "ffxQuery", "ffxDispatch", "ffxConfigure"}) {
        auto* code = reinterpret_cast<const unsigned char*>(reinterpret_cast<void*>(GetProcAddress(m, name)));
        if (!code) {
            if (std::strcmp(name, "ffxConfigure") == 0) continue;  // optional: only forwarded configure calls need it
            detail = std::string("no ") + name + " export";
            return wgfsr::ProbeResult::NoExports;
        }
        // An export outside the module (export table redirected), or code that jumps out of it (an inline hook): to
        // memory of no module (a hook's relay), or to a module that is not AMD's.
        const unsigned char* target = code >= base && code < end ? JumpTarget(code) : code;
        if (target && (target < base || target >= end)) {
            const std::wstring where = ModuleNameAt(target);
            if (where.empty() || where.compare(0, 3, L"amd") != 0) {
                detail = std::string(name) + " goes to " + (where.empty() ? std::string("memory outside any module") : Narrow(where));
                return wgfsr::ProbeResult::Hooked;
            }
        }
    }
    return wgfsr::ProbeResult::Ok;
}

const wgfsr::OriginalApi* CompareApi(const std::string& path) {
    static std::once_flag once;
    static wgfsr::OriginalApi api;
    static bool ok = false;
    std::call_once(once, [&path] {
        wgfsr::ProviderInputs in;
        in.compareDll = Widen(path);
        in.moduleDir = wronggpu::ModuleDirectory();
        const std::wstring exe = ModulePath(nullptr);
        in.exeDir = exe.substr(0, exe.find_last_of(L"\\/"));
        in.installerFound = InstallerCompareDll();
        std::vector<std::wstring> skip;
        if (Place().customSlot) {  // ...\<OptiScaler>\custom\<folder>\ffx\<file>: never look into other custom folders or plugins
            auto up = [](const std::wstring& p) { return p.substr(0, p.find_last_of(L"\\/")); };
            const std::wstring custom = up(up(up(Place().modulePath)));  // file -> ffx -> <folder> -> custom
            skip.push_back(custom);
            skip.push_back(up(custom) + L"\\plugins");
        }
        in.sdkDlls = GameSdkDlls(in.exeDir, skip);
        const std::vector<wgfsr::ProviderCandidate> candidates = wgfsr::ProviderCandidates(in);
        const std::wstring self = ModulePath(SelfModule());
        HMODULE chosen = nullptr;
        auto probe = [&](const wgfsr::ProviderCandidate& c, std::string& detail) {
            if (GetFileAttributesW(c.path.c_str()) == INVALID_FILE_ATTRIBUTES) return wgfsr::ProbeResult::Missing;
            if (SameFile(c.path, self) || FileDescription(c.path).find("WrongGPU") != std::string::npos) {
                detail = "WrongGPU's own file";
                return wgfsr::ProbeResult::Ours;
            }
            // By full path; the DLL's own dependencies from its folder, then the system's.
            HMODULE m = LoadLibraryExW(c.path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!m) {
                detail = "error " + std::to_string(GetLastError());
                return wgfsr::ProbeResult::NotLoaded;
            }
            const wgfsr::ProbeResult r = CheckProviderModule(m, c.path, detail);
            if (r != wgfsr::ProbeResult::Ok) {
                if (m != SelfModule()) FreeLibrary(m);
                return r;
            }
            const std::string description = FileDescription(c.path);
            detail = (description.empty() ? std::string("AMD FidelityFX") : description) + ", file version " + FileVersion(c.path);
            chosen = m;
            return r;
        };
        const wgfsr::ProviderChoice choice = wgfsr::ChooseProvider(candidates, probe, &Narrow);
        for (const std::string& line : choice.log) L().WriteLine(line);
        if (choice.index < 0 || !chosen) {
            L().WriteLine(choice.warning);
            return;
        }
        api.createContext = ProcAddress<PfnFfxCreateContext>(chosen, "ffxCreateContext");
        api.destroyContext = ProcAddress<PfnFfxDestroyContext>(chosen, "ffxDestroyContext");
        api.configure = ProcAddress<PfnFfxConfigure>(chosen, "ffxConfigure");
        api.query = ProcAddress<PfnFfxQuery>(chosen, "ffxQuery");
        api.dispatch = ProcAddress<PfnFfxDispatch>(chosen, "ffxDispatch");
        api.source = Narrow(candidates[size_t(choice.index)].path) + " (" + candidates[size_t(choice.index)].source + ")";
        ok = true;
    });
    return ok ? &api : nullptr;
}

void PrepareHipEnvironment() {
    std::string cleared;
    for (const wchar_t* name : {L"LLVM_PATH", L"ROCM_PATH", L"HIP_PATH", L"HIP_DEVICE_LIB_PATH"}) {
        wchar_t value[1024] = {};
        const DWORD n = GetEnvironmentVariableW(name, value, 1024);
        if (!n) continue;
        SetEnvironmentVariableW(name, nullptr);
        cleared += " " + Narrow(name) + "=" + (n < 1024 ? Narrow(value) : std::string("(long value)"));
    }
    SetEnvironmentVariableW(L"AMD_COMGR_CACHE", L"0");
    // 0.5.1: HIP's resource cache keeps the device memory of destroyed upscalers for the rest of the process (see
    // win_util.h); off unless the player set it.
    wchar_t cache[64] = {};
    const DWORD cacheSet = GetEnvironmentVariableW(L"GPU_RESOURCE_CACHE_SIZE", cache, 64);
    if (!cacheSet) SetEnvironmentVariableW(L"GPU_RESOURCE_CACHE_SIZE", L"0");
    const std::string cacheText = cacheSet ? "HIP's resource cache as set (GPU_RESOURCE_CACHE_SIZE=" + Narrow(cache) + ")"
                                           : std::string("HIP's resource cache off (GPU_RESOURCE_CACHE_SIZE=0)");
    L().Write("HIP environment for this process: AMD's code-compiler cache off; %s%s%s", cacheText.c_str(),
              cleared.empty() ? "" : "; cleared ROCm SDK variables:", cleared.c_str());
}

wgfsr::Config ReadSettings(const std::string& tag) {
    // The data folder rule and the old place: settings_bridge.h (clean/wgshared wg_settings_io.h).
    const std::wstring path = Place().ini;
    if (!Place().note.empty()) L().WriteLine(tag + Place().note);
    std::string text;
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f != INVALID_HANDLE_VALUE) {
        char buffer[4096];
        DWORD n = 0;
        while (ReadFile(f, buffer, sizeof buffer, &n, nullptr) && n) text.append(buffer, n);
        CloseHandle(f);
        L().WriteLine(tag + "settings read from " + Narrow(path));
    } else {
        // First start: write the defaults so the spec team can edit them.
        text = wgfsr::DefaultConfigText();
        f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f != INVALID_HANDLE_VALUE) {
            DWORD n = 0;
            WriteFile(f, text.data(), DWORD(text.size()), &n, nullptr);
            CloseHandle(f);
            L().WriteLine(tag + "no settings file; defaults written to " + Narrow(path));
        }
    }
    std::string warnings;
    wgfsr::Config config = wgfsr::ParseConfig(text, &warnings);
    // Registry options the front end reads (clean/wgshared; [wronggpu] keys, block and environment first): set there,
    // they decide over the [fsr] keys of the same meaning.
    {
        bool isDefault = true;
        std::string described;
        const int32_t fallback = Setting(WG_OPT_SR_FALLBACK_TO_AMD, &described, &isDefault);
        if (!isDefault) {
            config.fallbackToAmd = fallback != 0;
            L().WriteLine(tag + "settings: " + described + " (decides over [fsr] fallback_to_amd)");
        }
        const wg::IniValues ini = wg::parse_ini(text);
        std::string dll;
        if (ini.get("wronggpu", "sr_runtime_dll", &dll) && !dll.empty()) {
            config.runtimeDll = dll;
            L().WriteLine(tag + "settings: " + wg::describe_text("sr_runtime_dll", dll, wg::Source::Ini) + " (decides over [fsr] runtime_dll)");
        }
    }
    size_t start = 0;
    while (start < warnings.size()) {
        const size_t end = warnings.find('\n', start);
        L().WriteLine(tag + "warning: " + warnings.substr(start, end - start));
        start = end == std::string::npos ? warnings.size() : end + 1;
    }
    wchar_t executable[32768] = {};
    const DWORD length = GetModuleFileNameW(nullptr, executable, DWORD(sizeof executable / sizeof executable[0]));
    if (length && length < sizeof executable / sizeof executable[0] &&
        wgfsr::ApplyProcessCompatibility(config, Narrow(executable)))
        L().WriteLine(tag + "Model M game compatibility: shared-fence input synchronization (ready_wait=fence); explicit runtime options retained");
    L().WriteLine(tag + "settings: " + config.Describe());
    return config;
}

namespace {

std::once_flag g_runtimeOnce;
const WgUpscalerBackend* g_runtime = nullptr;
std::string g_runtimeError;

// "126: The specified module could not be found." for the log.
std::string WindowsErrorText(DWORD error) {
    wchar_t* text = nullptr;
    const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                                   error, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    std::string s = std::to_string(error);
    if (n && text) {
        std::wstring w(text, n);
        while (!w.empty() && (w.back() == L'\r' || w.back() == L'\n' || w.back() == L' ')) w.pop_back();
        s += ": " + Narrow(w);
    }
    if (text) LocalFree(text);
    return s;
}

// Why a table returned by wgGetUpscalerBackend(v) cannot be used, or empty when it can.
std::string BackendProblem(const WgUpscalerBackend* b, uint32_t v) {
    if (!b) return "returned null";
    if (b->interface_version != v) return "returned a table of interface version " + std::to_string(b->interface_version);
    if (wgfsr::BackendVersion(b) != v) return "returned a table of " + std::to_string(b->struct_size) + " bytes, too small for version " + std::to_string(v);
    if (!b->name || !b->create || !b->dispatch || !b->destroy) return "returned a table without name, create, dispatch or destroy";
    return {};
}

const WgUpscalerBackend* LoadRuntime(const wgfsr::Config& config, std::string& why) {
    std::call_once(g_runtimeOnce, [&] {
        // Always by full path from WrongGPU's data folder (or, for older installs, the folder of WrongGPU's FSR DLLs;
        // settings_bridge.h), never from the search path.
        const std::wstring path = Place().runtimeDir + L"\\" + Widen(config.runtimeDll);
        const std::string where = "WrongGPU runtime " + Narrow(path) + " (runtime_dll = " + config.runtimeDll + ")";
        if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
            g_runtimeError = where + " not found: it must be in WrongGPU's data folder (or, for older installs, next to WrongGPU's FSR DLLs)";
        } else if (HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH); !m) {
            const DWORD e = GetLastError();
            g_runtimeError = where + " could not be loaded: Windows error " + WindowsErrorText(e);
            if (e == ERROR_MOD_NOT_FOUND)
                g_runtimeError += " (the file exists, so a DLL it depends on is missing, e.g. the AMD HIP runtime amdhip64_7.dll that the "
                                  "AMD Software: Adrenalin Edition driver installs)";
            else if (e == ERROR_BAD_EXE_FORMAT)
                g_runtimeError += " (not a 64-bit Windows DLL)";
        } else {
            const std::string desc = FileDescription(path);
            L().Write("%s loaded: file version %s%s%s", where.c_str(), FileVersion(path).c_str(), desc.empty() ? "" : ", ", desc.c_str());
            auto get = ProcAddress<PfnWgGetUpscalerBackend>(m, "wgGetUpscalerBackend");
            std::string answers;
            // Newest interface first (version 2: same-frame split), then older ones.
            for (uint32_t v = WG_UPSCALER_INTERFACE_VERSION; get && !g_runtime && v >= WG_UPSCALER_INTERFACE_VERSION_MIN; --v) {
                const WgUpscalerBackend* b = get(v);
                const std::string problem = BackendProblem(b, v);
                if (problem.empty()) g_runtime = b;
                else answers += (answers.empty() ? "" : "; ") + std::string("asked for version ") + std::to_string(v) + ": " + problem;
            }
            if (!get) g_runtimeError = where + " does not export wgGetUpscalerBackend";
            else if (!g_runtime) g_runtimeError = where + " offers no usable upscaler interface (" + answers + ")";
            else
                L().Write("WrongGPU runtime upscaler \"%s\": interface version %u, flags 0x%x%s%s%s", g_runtime->name,
                          wgfsr::BackendVersion(g_runtime), wgfsr::BackendFlags(g_runtime),
                          (wgfsr::BackendFlags(g_runtime) & WG_BACKEND_NEEDS_SPLIT) ? " (same-frame split)" : "",
                          answers.empty() ? "" : "; ", answers.c_str());
        }
        if (!g_runtime)
            L().Write("ERROR: %s. Upscaler contexts go to AMD's FSR (with fallback_to_amd = 0 their creation fails)", g_runtimeError.c_str());
    });
    why = g_runtimeError;
    return g_runtime;
}

}  // namespace

const WgUpscalerBackend* ResolveBackend(const wgfsr::Config& config, std::string& why) {
    switch (config.upscaler) {
    case wgfsr::UpscalerChoice::StandIn: return wgfsr::GetStandInBackend(config.splitTest);
    case wgfsr::UpscalerChoice::Network: return LoadRuntime(config, why);
    case wgfsr::UpscalerChoice::Amd: why = "upscaler = amd"; return nullptr;
    }
    return nullptr;
}

const WgUpscalerBackend* StandInForFallback(const wgfsr::Config&) { return wgfsr::GetStandInBackend(false); }

bool D3D12ResourceInfo(void* resource, wgfsr::ResourceInfo* out) {
    if (!resource) return false;
    const D3D12_RESOURCE_DESC d = static_cast<ID3D12Resource*>(resource)->GetDesc();
    out->dxgiFormat = uint32_t(d.Format);
    out->width = uint32_t(d.Width);
    out->height = d.Height;
    return true;
}

void* DeviceFromCommandList(void* commandList) {
    if (!commandList) return nullptr;
    ID3D12Device* device = nullptr;
    if (FAILED(static_cast<ID3D12GraphicsCommandList*>(commandList)->GetDevice(IID_PPV_ARGS(&device))) || !device) return nullptr;
    device->Release();  // the game's device outlives its command lists; the upscaler AddRefs what it keeps
    return device;
}

const char* ModuleNameOf(const void* address) {
    thread_local const void* lastBase = nullptr;
    thread_local char lastName[MAX_PATH] = "?";
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(address), &m) || !m)
        return "?";
    if (m == lastBase) return lastName;
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(m, path, MAX_PATH);
    const wchar_t* name = wcsrchr(path, L'\\');
    const std::string narrow = Narrow(name ? name + 1 : path);
    std::snprintf(lastName, sizeof lastName, "%s", narrow.c_str());
    lastBase = m;
    return lastName;
}

}  // namespace wgwin
