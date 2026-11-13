// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the log collector: one .zip (stored, no compression) with WrongGPU's logs, the Windows events
// that matter and basic system information (support-manifest.json). Written for this project.
#include "logs.h"
#include <cstdio>
#include <filesystem>
#include <dxgi.h>
#include <shlobj.h>
#include <winevt.h>
#include "json.h"
#include "payload.h"
#include "pe.h"
#include "optiscaler.h"
#include "record.h"
#include "rr.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// Log collector: one .zip (stored, no compression) with WrongGPU's logs and basic system info (from the 0.2.0 installer)
// ------------------------------------------------------------------------------------------------
uint32_t Crc32(const void* data, size_t n) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) { uint32_t c = i; for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1; table[i] = c; }
        ready = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
struct ZipWriter {
    string out, central;
    uint16_t count = 0;
    static void U16(string& s, uint16_t v) { s += (char)(v & 0xFF); s += (char)(v >> 8); }
    static void U32(string& s, uint32_t v) { for (int i = 0; i < 4; ++i) s += (char)((v >> (8 * i)) & 0xFF); }
    void Add(const string& name, const string& data) {
        SYSTEMTIME t; GetLocalTime(&t);
        uint16_t dtime = (uint16_t)((t.wHour << 11) | (t.wMinute << 5) | (t.wSecond / 2));
        uint16_t ddate = (uint16_t)(((t.wYear - 1980) << 9) | (t.wMonth << 5) | t.wDay);
        uint32_t crc = Crc32(data.data(), data.size()), offset = (uint32_t)out.size(), size = (uint32_t)data.size();
        U32(out, 0x04034b50); U16(out, 20); U16(out, 0x0800); U16(out, 0); U16(out, dtime); U16(out, ddate);
        U32(out, crc); U32(out, size); U32(out, size); U16(out, (uint16_t)name.size()); U16(out, 0);
        out += name; out += data;
        U32(central, 0x02014b50); U16(central, 20); U16(central, 20); U16(central, 0x0800); U16(central, 0); U16(central, dtime); U16(central, ddate);
        U32(central, crc); U32(central, size); U32(central, size); U16(central, (uint16_t)name.size()); U16(central, 0); U16(central, 0);
        U16(central, 0); U16(central, 0); U32(central, 0); U32(central, offset);
        central += name;
        ++count;
    }
    string Finish() {
        string r = out;
        uint32_t cdOffset = (uint32_t)r.size();
        r += central;
        U32(r, 0x06054b50); U16(r, 0); U16(r, 0); U16(r, count); U16(r, count); U32(r, (uint32_t)central.size()); U32(r, cdOffset); U16(r, 0);
        return r;
    }
};
// Windows event log entries that matter for a report, newest first, from the last 14 days: crashes and hangs of the
// game (Application Error 1000, Application Hang 1002, Windows Error Reporting naming it), GPU driver resets (System
// log, Display 4101, and WER LiveKernelEvent reports). Only the event fields are kept. wevtapi.dll is loaded from
// System32 when needed.
string RecentWindowsEvents(const wstring& exeName) {
    HMODULE w = LoadSystemDll(L"wevtapi.dll");
    if (!w) return "(the Windows event log cannot be read on this system)\r\n";
    using QueryFn = EVT_HANDLE(WINAPI*)(EVT_HANDLE, LPCWSTR, LPCWSTR, DWORD);
    using NextFn = BOOL(WINAPI*)(EVT_HANDLE, DWORD, EVT_HANDLE*, DWORD, DWORD, PDWORD);
    using RenderFn = BOOL(WINAPI*)(EVT_HANDLE, EVT_HANDLE, DWORD, DWORD, PVOID, PDWORD, PDWORD);
    using CloseFn = BOOL(WINAPI*)(EVT_HANDLE);
    auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(w, n)); };
    auto evtQuery = reinterpret_cast<QueryFn>(sym("EvtQuery"));
    auto evtNext = reinterpret_cast<NextFn>(sym("EvtNext"));
    auto evtRender = reinterpret_cast<RenderFn>(sym("EvtRender"));
    auto evtClose = reinterpret_cast<CloseFn>(sym("EvtClose"));
    if (!evtQuery || !evtNext || !evtRender || !evtClose) return "(the Windows event log cannot be read on this system)\r\n";
    struct Query { const wchar_t* channel; const wchar_t* xpath; bool gameOnly; };
    const Query queries[] = {
        {L"Application", L"*[System[(Provider[@Name='Application Error'] or Provider[@Name='Application Hang'] or Provider[@Name='Windows Error Reporting']) and TimeCreated[timediff(@SystemTime) <= 1209600000]]]", true},
        {L"System", L"*[System[Provider[@Name='Display'] and TimeCreated[timediff(@SystemTime) <= 1209600000]]]", false},
    };
    auto attr = [](const wstring& x, size_t from, size_t to, const wstring& name) -> wstring {
        size_t a = x.find(L" " + name + L"=", from);
        if (a == wstring::npos || a > to) return L"";
        a += name.size() + 2;
        const wchar_t q = x[a];
        const size_t e = x.find(q, a + 1);
        return e == wstring::npos ? L"" : x.substr(a + 1, e - a - 1);
    };
    auto unescape = [](wstring s) {
        const std::pair<const wchar_t*, const wchar_t*> ents[] = {{L"&lt;", L"<"}, {L"&gt;", L">"}, {L"&quot;", L"\""}, {L"&apos;", L"'"}, {L"&amp;", L"&"}};
        for (auto& [from, to] : ents) for (size_t p; (p = s.find(from)) != wstring::npos;) s.replace(p, wcslen(from), to);
        for (auto& c : s) if (c == L'\r' || c == L'\n' || c == L'\t') c = L' ';
        return s;
    };
    const wstring exe = Lower(exeName);
    wstring out;
    for (const auto& q : queries) {
        out += wstring(L"[") + q.channel + L" log]\r\n";
        EVT_HANDLE results = evtQuery(nullptr, q.channel, q.xpath, EvtQueryChannelPath | EvtQueryReverseDirection);
        if (!results) { out += L"    (could not read: error " + std::to_wstring(GetLastError()) + L")\r\n"; continue; }
        int kept = 0;
        EVT_HANDLE events[32];
        DWORD got = 0;
        while (kept < 25 && evtNext(results, 32, events, 3000, 0, &got)) {
            for (DWORD i = 0; i < got; ++i) {
                DWORD used = 0, props = 0;
                evtRender(nullptr, events[i], EvtRenderEventXml, 0, nullptr, &used, &props);
                std::vector<wchar_t> buf(used / sizeof(wchar_t) + 1);
                const bool ok = evtRender(nullptr, events[i], EvtRenderEventXml, DWORD(buf.size() * sizeof(wchar_t)), buf.data(), &used, &props);
                evtClose(events[i]);
                if (!ok || kept >= 25) continue;
                const wstring x = buf.data(), lx = Lower(x);
                const bool gpuReset = lx.find(L"livekernelevent") != wstring::npos;
                if (q.gameOnly && !gpuReset && (exe.empty() || lx.find(exe) == wstring::npos)) continue;
                const size_t sys = x.find(L"<System>");
                const size_t prov = x.find(L"<Provider", sys), time = x.find(L"<TimeCreated", sys), id = x.find(L"<EventID", sys);
                wstring line = L"  " + attr(x, time, x.find(L'>', time), L"SystemTime") + L"  " + attr(x, prov, x.find(L'>', prov), L"Name");
                if (id != wstring::npos) { const size_t a = x.find(L'>', id) + 1; line += L" " + x.substr(a, x.find(L'<', a) - a); }
                line += L"\r\n   ";
                for (size_t p = x.find(L"<Data"); p != wstring::npos; p = x.find(L"<Data", p + 5)) {
                    const size_t close = x.find(L'>', p);
                    if (close == wstring::npos) break;
                    const wstring name = attr(x, p, close, L"Name");
                    wstring value;
                    if (x[close - 1] != L'/') { const size_t end = x.find(L"</Data>", close); value = x.substr(close + 1, end == wstring::npos ? 0 : end - close - 1); }
                    if (value.empty()) continue;
                    line += L" " + (name.empty() ? L"" : name + L"=") + unescape(value).substr(0, 300) + L" |";
                }
                out += line + L"\r\n";
                ++kept;
            }
        }
        evtClose(results);
        if (!kept) out += L"    (none)\r\n";
    }
    return ToUtf8(out);
}

string SupportQuote(const string& text) {
    string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) { char b[7]; snprintf(b, sizeof b, "\\u%04x", unsigned(c)); out += b; }
        else out += char(c);
    }
    return out + '"';
}
// Bounded reads avoid loading an entire oversized log just to keep its tail.
bool SupportReadTail(const wstring& path, string& out, uint64_t limit, uint64_t& originalSize, DWORD& error) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = GetLastError(); return false; }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h, &size) || size.QuadPart < 0) { error = GetLastError(); CloseHandle(h); return false; }
    originalSize = uint64_t(size.QuadPart);
    const uint64_t count = std::min(originalSize, limit);
    LARGE_INTEGER offset{}; offset.QuadPart = LONGLONG(originalSize - count);
    if (!SetFilePointerEx(h, offset, nullptr, FILE_BEGIN)) { error = GetLastError(); CloseHandle(h); return false; }
    out.resize(size_t(count));
    size_t read = 0;
    while (read < out.size()) {
        DWORD got = 0;
        if (!ReadFile(h, &out[read], DWORD(std::min<size_t>(out.size() - read, 1u << 20)), &got, nullptr)) {
            error = GetLastError(); CloseHandle(h); out.clear(); return false;
        }
        if (!got) break; // Another process may truncate a file while collection runs.
        read += got;
    }
    out.resize(read); CloseHandle(h); return true;
}
string SupportDriverRegistry() {
    HKEY video = nullptr;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Video", 0, KEY_READ, &video) != ERROR_SUCCESS) return "[]";
    string records = "["; bool first = true;
    for (DWORD i = 0; i < 128; ++i) {
        wchar_t key[256]{}; DWORD length = 255;
        if (RegEnumKeyExW(video, i, key, &length, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
        HKEY adapter = nullptr;
        if (RegOpenKeyExW(video, (wstring(key) + L"\\0000").c_str(), 0, KEY_READ, &adapter) != ERROR_SUCCESS) continue;
        string record = "{"; bool any = false;
        for (const wchar_t* field : {L"DriverDesc", L"HardwareInformation.AdapterString", L"DriverVersion", L"RadeonSoftwareVersion", L"ReleaseVersion"}) {
            wchar_t value[4096]{}; DWORD bytes = sizeof value, type = 0;
            if (RegQueryValueExW(adapter, field, nullptr, &type, reinterpret_cast<BYTE*>(value), &bytes) != ERROR_SUCCESS ||
                (type != REG_SZ && type != REG_EXPAND_SZ) || bytes >= sizeof value) continue;
            if (any) record += ','; any = true;
            record += SupportQuote(ToUtf8(field)) + ":" + SupportQuote(ToUtf8(value));
        }
        RegCloseKey(adapter);
        if (!any) continue;
        if (!first) records += ','; first = false;
        records += record + "}";
    }
    RegCloseKey(video); return records + "]";
}
string SupportAdapters() {
    HMODULE dxgi = LoadSystemDll(L"dxgi.dll");
    if (!dxgi) return "[]";
    using CreateFn = HRESULT(WINAPI*)(REFIID, void**);
    auto create = reinterpret_cast<CreateFn>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    IDXGIFactory1* factory = nullptr;
    if (!create || FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return "[]";
    string out = "["; bool first = true;
    for (UINT i = 0; i < 32; ++i) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) != S_OK || !adapter) break;
        DXGI_ADAPTER_DESC1 d{}; const HRESULT desc = adapter->GetDesc1(&d);
        LARGE_INTEGER version{};
        const HRESULT driverResult = adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version);
        adapter->Release();
        if (FAILED(desc)) continue;
        wchar_t driver[64]{};
        if (SUCCEEDED(driverResult)) swprintf(driver, 64, L"%u.%u.%u.%u", HIWORD(version.HighPart), LOWORD(version.HighPart), HIWORD(version.LowPart), LOWORD(version.LowPart));
        if (!first) out += ','; first = false;
        out += "{\"name\":" + SupportQuote(ToUtf8(d.Description)) + ",\"luid_low\":" + std::to_string(d.AdapterLuid.LowPart) +
            ",\"luid_high\":" + std::to_string(d.AdapterLuid.HighPart) + ",\"vendor_id\":" + std::to_string(d.VendorId) +
            ",\"device_id\":" + std::to_string(d.DeviceId) + ",\"dedicated_vram_bytes\":" + std::to_string(d.DedicatedVideoMemory) +
            ",\"software\":" + (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE ? "true" : "false") +
            ",\"umd_driver\":" + SupportQuote(ToUtf8(driver)) + "}";
    }
    factory->Release(); return out + "]";
}
// 0.6.1: what a process without the converter data or the payload table (uninstall.exe, the restorer: wg_core without them)
// can say truthfully; WrongGPU.exe, which embeds both, keeps its own lines byte for byte.
bool HasConverterData(Context& ctx, bool k) { return ctx.data && MapResourceId(ctx, k) != 0; }
wstring ModelDataUnverified(const wstring& file) {
    if (!IsFile(file)) return IsDir(file) ? L"present but unreadable (a folder, not a file)" : L"missing";
    HANDLE h = CreateFileW(file.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return L"present but unreadable (" + ErrText(GetLastError()) + L")";
    LARGE_INTEGER size{};
    char first = 0;
    DWORD got = 0;
    const bool sized = GetFileSizeEx(h, &size) != 0;
    const DWORD sizeError = sized ? 0 : GetLastError();
    const bool read = sized && ReadFile(h, &first, 1, &got, nullptr) != 0;
    const DWORD readError = read ? 0 : GetLastError();
    CloseHandle(h);
    if (!sized) return L"present but unreadable (" + ErrText(sizeError) + L")";
    if (!read) return L"present but unreadable (" + ErrText(readError) + L")";
    return L"present, " + std::to_wstring(size.QuadPart) + L" bytes, not verified here (uninstall.exe has no converter data; use Collect logs in the WrongGPU "
           L"launcher to verify)";
}

// Collects from the install folder (and the exe folder and game root, where some games write logs).
Bundle CollectLogs(Context& ctx, const wstring& dirIn, const wstring& outZip, const SystemInfo& sys) {
    if (dirIn.empty() || !IsDir(dirIn)) Fail(L"Choose the game first: logs are collected from its folder.", EXIT_REFUSED);
    if (outZip.size() < 4 || Lower(outZip.substr(outZip.size() - 4)) != L".zip")
        Fail(L"Choose a .zip file for the support report so an existing game file is not replaced.", EXIT_REFUSED);
    Record m;
    bool installed = LoadRecord(RecordPath(dirIn), m) && !m.legacy;
    wstring exe = installed && IsFile(m.exe) ? m.exe : ResolveGameExe(dirIn);
    GameTarget t = exe.empty() ? GameTarget{} : InspectGame(ctx, exe);
    const wstring dir = installed ? dirIn : (t.dir.empty() ? dirIn : t.dir);
    if (!installed) installed = LoadRecord(RecordPath(dir), m) && !m.legacy;
    Bundle b;
    ZipWriter z;
    std::vector<std::pair<wstring, wstring>> files;  // (path, name in zip)
    std::set<wstring> seen;
    std::set<wstring> memberNames;
    auto take = [&](const wstring& path, const wstring& name) {
        if (!IsFile(path) || seen.count(Lower(FullPath(path)))) return;
        if (files.size() >= 64) { b.notes.push_back(L"Support file limit reached; " + name + L" was omitted."); return; }
        seen.insert(Lower(FullPath(path)));
        wstring unique = name;
        for (unsigned i = 2; !memberNames.insert(Lower(unique)).second; ++i) unique = L"duplicate-" + std::to_wstring(i) + L"\\" + name;
        files.push_back({path, unique});
    };
    std::vector<std::pair<wstring, wstring>> dirs{{dir, L""}};
    if (!exe.empty() && !IEq(Parent(exe), dir)) dirs.push_back({Parent(exe), L"exe-folder\\"});
    const wstring root = GameRoot(exe.empty() ? dir : Parent(exe));
    if (!IEq(root, dir) && (exe.empty() || !IEq(root, Parent(exe)))) dirs.push_back({root, L"game-root\\"});
    if (IsDir(Join(dir, kOptiSlotDir))) dirs.push_back({Join(dir, kOptiSlotDir), L"optiscaler-slot\\"});  // the front end's log in OptiScaler 10's slot
    auto newest = [](std::vector<wstring> v, size_t n) { if (v.size() > n) v.erase(v.begin(), v.end() - ptrdiff_t(n)); return v; };
    for (auto& [d, prefix] : dirs) {
        for (auto pat : {L"wronggpu-*.log", L"wronggpu-*.log.old", L"wronggpu-report*.txt", L"wronggpu-report*.json", L"wronggpu*.ini",
                         L"wronggpu-diagnostics*.jsonl", L"wronggpu-diagnostics*.jsonl.old", L"OptiScaler.log", L"OptiScaler.previous*.log", L"OptiScaler.ini", L"fakenvapi.log"})
            for (auto& f : ListFiles(d, pat)) take(f, prefix + BaseName(f));
        for (auto& f : newest(ListFiles(d, L"wronggpu-crash-*.txt"), 5)) take(f, L"crash\\" + BaseName(f));
        for (auto& f : newest(ListFiles(d, L"wronggpu-crash-*.dmp"), 2)) {
            b.notes.push_back(L"Memory dump " + BaseName(f) + L" was not included automatically; memory dumps can contain personal data and model data. It stays in the game folder.");
        }
    }
    take(RecordPath(dir), L"install\\install.record");
    take(Join(StateDir(dir), L"setup.log"), L"install\\setup.log");
    take(Join(dir, kRrSettings), L"ray-reconstruction\\wronggpu.ini");  // the RR plugin's settings file (rr.h)
    take(Join(RrDataDir(), L"model-source.txt"), L"ray-reconstruction\\model-source.txt");
    const bool reportK = WRONGGPU_PRESET_K && installed && IEq(FullPath(m.model), FullPath(Join(Join(ModelBaseDir(), L"K"), L"model.wgcm")));
    const wstring reportModelDir = reportK ? Join(ModelBaseDir(), L"K") : ModelBaseDir();
    const wstring reportModelFile = Join(reportModelDir, L"model.wgcm");
    const wstring reportSourceFile = Join(reportModelDir, L"model-source.txt");
    take(reportSourceFile, L"install\\model-source.txt");
    if (!ctx.logFile.empty()) take(ctx.logFile, L"install\\setup-latest.log");
    wstring gameSettingsStatus = L"unsupported game-settings location; use frame/settings observations";
    if (IEq(BaseName(exe), L"Cyberpunk2077.exe")) {
        const wstring settings = Join(Parent(AppDataDir()), L"CD Projekt Red\\Cyberpunk 2077\\UserSettings.json");
        const size_t before = files.size();
        take(settings, L"game-settings\\Cyberpunk-UserSettings.json");
        gameSettingsStatus = files.size() > before ? L"included" : L"not found or collection limit reached";
    }

    // System information.
    wstring s = L"WrongGPU - log bundle (attach this file to your game report)\r\n";
    if (*ctx.data->buildId)
        s += L"Created: " + NowStamp(false) + L"\r\nInstaller version: " + wstring(ctx.data->version) + L" (build " + ctx.data->buildId + L", payload " + ctx.data->payloadName + L")\r\n";
    else  // uninstall.exe: its own version; it carries no build name or payload (0.6.1: no empty fields)
        s += L"Created: " + NowStamp(false) + L"\r\nInstaller version: " + wstring(ctx.data->version) + L" (uninstall.exe of this version; build and payload not known here)\r\n";
    s += L"Install folder: " + dir + L"\r\n";
    if (!exe.empty()) s += L"Game executable: " + exe + L" (file version " + FileVersion(exe) + L")\r\n";
    s += L"Game status: " + wstring(t.validated ? L"Validated" : L"Experimental (not validated yet)") + L"\r\n";
    s += L"Installed: " + (installed ? L"yes, version " + m.version + L", " + m.status : wstring(L"no installation record")) + L"\r\n";
    for (auto& st : InspectSlots(ctx, dir, installed ? &m : nullptr)) {
        if (st.kind == FFX_NONE && !IsFile(Join(dir, st.slot->original))) continue;
        s += wstring(st.slot->name) + L": " + (st.kind == FFX_OURS ? L"WrongGPU" : st.kind == FFX_GAME ? L"the game's" : st.kind == FFX_FOREIGN ? L"other (not the game's, not WrongGPU's)" : L"missing") +
             L" (" + st.info + L")\r\n";
        const wstring orig = Join(dir, st.slot->original);
        if (IsFile(orig)) { wstring oi; const FfxKind ok = ClassifySlot(*st.slot, orig, nullptr, oi); s += wstring(st.slot->original) + L": " + (ok == FFX_GAME ? L"the game's" : L"unexpected") + L" (" + oi + L"; " + SignatureInfo(orig) + L")\r\n"; }
        else if (st.kind == FFX_GAME) s += L"    " + SignatureInfo(Join(dir, st.slot->name)) + L"\r\n";
    }
    s += L"Anti-cheat: " + (t.antiCheat.empty() ? wstring(L"none found") : t.antiCheat) + L"\r\n";
    wstring why;
    // Collection must use the installed preset, independent of the wizard's default M choice.
    // Keep a private map so a concurrent conversion worker's map is never changed.
    // 0.6.1: a process without the converter data (uninstall.exe) never claims the map is invalid: it says what it can see
    if (!HasConverterData(ctx, reportK)) {
        s += L"Model data: " + ModelDataUnverified(reportModelFile) + L"\r\n";
    } else {
    DWORD reportMapSize = 0;
    const void* reportMapBytes = ResourceData(MapResourceId(ctx, reportK), reportMapSize);
    wgc::LoadedMap reportMap;
    string reportWhy;
    bool modelMatches = reportMapBytes && Sha256Bytes(reportMapBytes, reportMapSize) == MapSha256(ctx, reportK);
    if (modelMatches) modelMatches = wgc::load_map_bytes(string(static_cast<const char*>(reportMapBytes), reportMapSize), reportMap, {}).code == wgc::kOk;
    if (modelMatches) modelMatches = wgc::verify_model_file(*reportMap.map, std::filesystem::path(reportModelFile), reportWhy);
    else reportWhy = "the embedded converter map is invalid";
    s += L"Model data: " + (modelMatches ? L"present and matching (" + reportModelFile + L")" : L"not usable (" + FromUtf8(reportWhy) + L")") + L"\r\n";
    }  // (WrongGPU.exe's lines above: unchanged, kept at their indentation so the diff shows it)
#if WRONGGPU_PRESET_K
    s += L"Installed startup preset: " + wstring(reportK ? L"K" : L"M") + L"\r\n";
#endif
    {   // The optional output-filter weights of the model M file (spec 08 draft 2), whichever preset starts first.
        DWORD mSize = 0;
        const void* mBytes = ResourceData(ctx.data->mapId, mSize);
        wgc::LoadedMap mMap;
        const wstring mFile = Join(ModelBaseDir(), L"model.wgcm");
        if (mBytes && Sha256Bytes(mBytes, mSize) == ctx.data->mapSha256 &&
            wgc::load_map_bytes(string(static_cast<const char*>(mBytes), mSize), mMap, {}).code == wgc::kOk && MapHasOptionalGroup(*mMap.map)) {
            string mWhy;
            wgc::ModelFileInfo mInfo;
            wstring state;
            if (!IsFile(mFile)) state = L"no Model M file on this PC";
            else if (!wgc::verify_model_file(*mMap.map, std::filesystem::path(mFile), mWhy, &mInfo)) state = L"the Model M file is not usable (" + FromUtf8(mWhy) + L")";
            else state = wstring(mInfo.absent_groups.empty() ? L"from the DLSS file (weights in the Model M file)" : L"WrongGPU's own (no weights in the Model M file)") +
                         (mInfo.same_map ? L"; file written with this installer's converter data" : L"; file written with other converter data");
            s += L"Model M output filter: " + state + L"\r\n";
        }
        if (installed && !m.modelM.empty()) s += L"Model M file at the last install: " + m.modelM + L"\r\n";
    }
    s += L"The in-game provider can differ from the startup choice; the runtime session/context logs identify the actual model and fallback.\r\n";
    s += L"Windows: " + sys.win.text + L"\r\n";
    s += L"GPU: " + (sys.gpu.name.empty() ? wstring(L"unknown") : sys.gpu.name) + L" (driver " + (sys.gpu.driver.empty() ? wstring(L"unknown") : sys.gpu.driver) + L")\r\n";
    auto hips = HipCandidates();
    s += L"AMD HIP runtime: " + (hips.empty() ? wstring(L"not found") : hips[0]) + L"\r\n";
    s += L"\r\nWrongGPU-related files in the install folder (name, size, SHA-256):\r\n";
    std::vector<wstring> names{L"dxgi.dll", L"d3d12.dll", L"version.dll", L"winmm.dll", L"OptiScaler.ini", L"OptiScaler.dll", L"wronggpu_ffx_dx12.dll", kOptiSlotFrontEnd, L"wronggpu-amd-compare.dll", L"amd_fidelityfx_upscaler_dx12.dll"};
    for (auto& sl : kSlots) { names.push_back(sl.name); names.push_back(sl.original); }
    for (auto& f : *ctx.data) names.push_back(f.rel);
    // 0.6.1: uninstall.exe has no payload table: the install record's files instead, so the list is not silently missing
    // WrongGPU's runtime, plugins and text files (WrongGPU.exe's list: unchanged)
    if (!ctx.data->count && installed)
        for (auto& e : m.e) if (e.action == L"copy") names.push_back(e.rel);
    std::set<wstring> listed;
    string inventory = "["; bool firstInventory = true;
    for (auto& n : names) {
        wstring p = Join(dir, n);
        if (!listed.insert(Lower(n)).second || !IsFile(p)) continue;
        s += L"    " + n + L"  " + std::to_wstring(FileSize(p)) + L"  " + Sha256File(p) + L"\r\n";
        if (!firstInventory) inventory += ','; firstInventory = false;
        inventory += "{\"name\":" + SupportQuote(ToUtf8(n)) + ",\"size\":" + std::to_string(FileSize(p)) +
            ",\"file_version\":" + SupportQuote(ToUtf8(FileVersion(p))) + ",\"sha256\":" + SupportQuote(ToUtf8(Sha256File(p))) + "}";
    }
    string hipInventory = "[";
    for (size_t i = 0; i < hips.size() && i < 16; ++i) {
        if (i) hipInventory += ',';
        hipInventory += "{\"path\":" + SupportQuote(ToUtf8(hips[i])) + ",\"file_version\":" + SupportQuote(ToUtf8(FileVersion(hips[i]))) +
            ",\"sha256\":" + SupportQuote(ToUtf8(Sha256File(hips[i]))) + "}";
    }
    hipInventory += "]";
    s += L"\r\nIncluded files:\r\n    system-info.txt\r\n    windows-events.txt  (game crashes and GPU driver resets from the Windows event log, last 14 days)\r\n";
    for (auto& f : files) s += L"    " + f.second + L"  (from " + f.first + L")\r\n";
    string artifacts = "["; bool firstArtifact = true;
    auto addArtifact = [&](const string& name, const string& data, uint64_t originalSize) {
        z.Add(name, data);
        if (!firstArtifact) artifacts += ','; firstArtifact = false;
        artifacts += "{\"name\":" + SupportQuote(name) + ",\"status\":\"included\",\"bytes\":" + std::to_string(data.size()) +
            ",\"original_bytes\":" + std::to_string(originalSize) + ",\"truncated\":" + (originalSize > data.size() ? "true" : "false") +
            ",\"sha256\":" + SupportQuote(ToUtf8(Sha256Bytes(data.data(), data.size()))) + "}";
    };
    const string systemText = ToUtf8(s);
    addArtifact("system-info.txt", systemText, systemText.size());
    b.files.push_back(L"system-info.txt (Windows version, GPU and driver, HIP runtime, game version, install details)");
    const string eventsText = "WrongGPU - Windows event log entries (game crashes and hangs, GPU driver resets), newest first\r\n\r\n" + RecentWindowsEvents(BaseName(exe));
    addArtifact("windows-events.txt", eventsText, eventsText.size());
    b.files.push_back(L"windows-events.txt (game crashes and GPU driver resets recorded by Windows)");
    string failures = "[", recentFrames = "[", runtime = "["; size_t failureCount = 0, frameCount = 0, malformed = 0, runtimeCount = 0;
    uint64_t remainingBytes = 64u << 20;
    for (auto& f : files) {
        string data;
        uint64_t originalSize = 0; DWORD error = 0;
        if (!remainingBytes) {
            if (!firstArtifact) artifacts += ','; firstArtifact = false;
            artifacts += "{\"name\":" + SupportQuote(ToUtf8(f.second)) + ",\"status\":\"omitted\",\"reason\":\"bundle_byte_limit\"}";
            continue;
        }
        if (!SupportReadTail(f.first, data, std::min<uint64_t>(8u << 20, remainingBytes), originalSize, error)) {
            if (!firstArtifact) artifacts += ','; firstArtifact = false;
            artifacts += "{\"name\":" + SupportQuote(ToUtf8(f.second)) + ",\"status\":\"unreadable\",\"win32_error\":" + std::to_string(error) + "}";
            continue;
        }
        remainingBytes -= data.size();
        string name = ToUtf8(f.second);
        std::replace(name.begin(), name.end(), '\\', '/');
        addArtifact(name, data, originalSize);
        b.files.push_back(f.second);
        if (name.find("wronggpu-diagnostics") != string::npos && name.find(".jsonl") != string::npos) {
            // Keep the newest records first: the failure usually sits at the tail.
            for (size_t end = data.size(), lineCount = 0; end && lineCount < 20000; ++lineCount) {
                while (end && (data[end - 1] == '\n' || data[end - 1] == '\r')) --end;
                if (!end) break;
                const size_t newline = data.rfind('\n', end - 1);
                const size_t start = newline == string::npos ? 0 : newline + 1;
                const string line = data.substr(start, end - start);
                end = start;
                wgc::Json event; string errorText;
                if (line.size() > 32768 || !wgc::parse_json(line, event, errorText)) { ++malformed; continue; }
                const auto* type = event.find("event");
                if (!type || !type->is_string()) { ++malformed; continue; }
                if (type->text == "failure" && failureCount < 256) { if (failureCount++) failures += ','; failures += line; }
                if ((type->text == "failure_frame" || type->text == "frame_begin" || type->text == "backend_return" || type->text == "provider" || type->text == "device_failure_observation") && frameCount < 256) {
                    if (frameCount++) recentFrames += ','; recentFrames += line;
                }
                if ((type->text == "effective_settings" || type->text == "loaded_modules" || type->text == "hip_runtime_version" ||
                     type->text == "cpu_detection_stack" || type->text == "session" || type->text == "records_dropped" ||
                     type->text == "module_snapshot_unavailable") && runtimeCount < 128) {
                    if (runtimeCount++) runtime += ','; runtime += line;
                }
            }
        }
    }
    string support = "{\"schema\":1,\"report_created\":" + SupportQuote(ToUtf8(NowStamp(false))) +
        ",\"installer_version\":" + SupportQuote(ToUtf8(ctx.data->version)) + ",\"game_executable\":" + SupportQuote(ToUtf8(exe)) +
        ",\"windows\":" + SupportQuote(ToUtf8(sys.win.text)) + ",\"gpu\":" + SupportQuote(ToUtf8(sys.gpu.name)) +
        ",\"umd_driver\":" + SupportQuote(ToUtf8(sys.gpu.driver)) + ",\"adapters\":" + SupportAdapters() + ",\"driver_registry\":" + SupportDriverRegistry() +
        ",\"game_settings_capture\":" + SupportQuote(ToUtf8(gameSettingsStatus)) +
        ",\"hip_candidates\":" + hipInventory + ",\"installed_files\":" + inventory + "]" +
        ",\"artifacts\":" + artifacts + "],\"failures\":" + failures + "],\"frame_observations\":" + recentFrames +
        "],\"runtime_observations\":" + runtime + "],\"malformed_diagnostic_lines\":" + std::to_string(malformed) +
        ",\"root_cause_proven\":false,\"gpu_replay_available\":false,\"file_hashes_observed_at\":\"collection_time\",\"collection_limits\":{\"file_tail_bytes\":8388608,\"bundle_bytes\":67108864,\"files\":64,\"failure_records\":256,\"frame_records\":256}}\n";
    z.Add("support-manifest.json", support);
    b.files.push_back(L"support-manifest.json (versions, hashes, linked failure/frame evidence and collection limits)");
    string zip = z.Finish();
    b.zip = outZip;
    EnsureDir(Parent(outZip));
    WriteFileAtomic(outZip, zip.data(), zip.size());
    LogLine(ctx, L"collected logs into " + outZip);
    return b;
}
wstring DefaultBundlePath(const wstring& dir) {
    PWSTR desk = nullptr;
    wstring out;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Desktop, 0, nullptr, &desk)) && desk) out = desk;
    if (desk) CoTaskMemFree(desk);
    if (out.empty()) out = AppDataDir();
    Record m;
    wstring exe = LoadRecord(RecordPath(dir), m) && !m.exe.empty() ? m.exe : ResolveGameExe(dir);
    wstring stem = exe.empty() ? wstring(L"game") : Stem(exe);
    for (auto& c : stem) if (!iswalnum(c) && c != L'-' && c != L'_') c = L'_';
    return Join(out, L"WrongGPU-logs-" + stem + L"-" + NowStamp(true) + L".zip");
}
// The saved logs in words (the restorer's Modify, the setup's Collect logs): where the file is and what is in it, as the
// launcher says it (LogsSavedText); File Explorer opens with the file selected after it
wstring BundleMessage(const Bundle& b) {
    PWSTR desk = nullptr;
    bool desktop = false;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Desktop, 0, nullptr, &desk)) && desk) desktop = IEq(Parent(b.zip), desk);
    if (desk) CoTaskMemFree(desk);
    wstring t = wstring(desktop ? L"Logs saved to your desktop." : L"Logs saved.") + L"\r\n\r\nThe file is " + BaseName(b.zip) +
                (desktop ? L"" : L", in " + Parent(b.zip)) + L". It includes file paths (with your Windows user name), your graphics card, its driver and your "
                L"Windows version. Nothing is uploaded: attach the file to your report.";
    for (auto& n : b.notes) t += L"\r\n\r\n" + n;
    return t;
}

}  // namespace wg
