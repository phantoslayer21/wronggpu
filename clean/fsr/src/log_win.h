// SPDX-License-Identifier: MIT
#pragma once
// Plain-text log file next to the DLL (wronggpu-fsr.log). Adapted from imported/0.2.0-infra/src/common/log.h
// (audited original WrongGPU code): shared append-mode file, one atomic append per line, time and process id on
// every line, rotation to <name>.old when a process opens a file larger than 8 MB. Changes: namespace and file
// name, larger rotation size, module tag on every line, Bytes() for the front end's log-size limit, explicit flush-free design kept.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <wincrypt.h>
#include <map>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include "diagnostics.h"

namespace wronggpu {

// Tag of the module on every line ("amd_fidelityfx_dx12", "ffx_fsr3", "ffx_fsr3upscaler"), defined once per DLL:
// all of WrongGPU's DLLs append to the same file.
extern const char* const kLogTag;

// Directory of the module containing this code (the game's executable folder when installed).
inline std::wstring ModuleDirectory() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&ModuleDirectory), &self);
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(self, path, MAX_PATH);
    std::wstring dir(path);
    return dir.substr(0, dir.find_last_of(L"\\/"));
}

// Lines are queued in memory and written by a background thread (one per DLL), so a game thread never waits for the
// disk (a synchronous WriteFile per line on the render thread caused frame-time spikes). Each WriteFile still appends
// whole lines atomically. The object is never destroyed (the writer thread may run until the process ends); lines
// queued in the last moment before a crash can be lost. Beyond kMaxPending queued bytes new lines are dropped (counted
// and reported) instead of growing the queue.
class Log {
public:
    static Log& Get() {
        static Log* log = new Log(L"wronggpu-fsr.log");
        return *log;
    }
    void Write(const char* fmt, ...) {
        char line[4096];
        SYSTEMTIME t;
        GetLocalTime(&t);
        int n = std::snprintf(line, sizeof line, "[%02u:%02u:%02u.%03u %lu] [%s] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds,
                              GetCurrentProcessId(), kLogTag);
        const int prefix = n;
        va_list args;
        va_start(args, fmt);
        const int m = std::vsnprintf(line + n, sizeof line - n - 2, fmt, args);
        va_end(args);
        n += m < 0 ? 0 : (m < int(sizeof line) - n - 2 ? m : int(sizeof line) - n - 3);
        Diagnose(std::string(line + prefix, size_t(n - prefix)));
        line[n++] = '\r';
        line[n++] = '\n';
        Append(line, n);
    }
    // Long lines (FidelityFX descriptions) are written whole, without the 4 KB limit of Write.
    void WriteLine(const std::string& text) {
        Diagnose(text);
        char prefix[64];
        SYSTEMTIME t;
        GetLocalTime(&t);
        const int n = std::snprintf(prefix, sizeof prefix, "[%02u:%02u:%02u.%03u %lu] [%s] ", t.wHour, t.wMinute, t.wSecond,
                                    t.wMilliseconds, GetCurrentProcessId(), kLogTag);
        std::string line;
        line.reserve(text.size() + 64);
        line.append(prefix, n > 0 ? size_t(n) : 0);
        line += text;
        line += "\r\n";
        Append(line.data(), int(line.size()));
    }
    uint64_t Bytes() const { return bytes_.load(std::memory_order_relaxed); }
    // Writes everything queued so far before returning (tests, and before reading the file from the same process).
    void Flush() {
        if (async_) WritePending();
    }
    const std::wstring& Path() const { return path_; }
    const std::string& Session() const { return session_; }
    const std::wstring& DiagnosticPath() const { return diagnosticPath_; }
    // No resource dereferences, GPU calls, disk I/O or per-frame allocation here.
    void RecordFrame(uint64_t context, const WgFrame& frame) noexcept {
        try {
            if (!async_ || !diagnosticAvailable_) return;
            const auto captured = wgdiag::Capture(context, frame);
            { std::lock_guard<std::mutex> lock(frames_mutex_); frames_.Push(captured); }
            if (frame.frame_index < 8) metadataPending_ = true;
            if (frame.frame_index < 8 || frame.frame_index % 300 == 0)
                Event("frame_begin", wgdiag::Json(captured));
        } catch (...) {} // Diagnostics must never turn a successful host call into failure.
    }
    void Event(const char* kind, const std::string& data) noexcept {
        try {
            if (!async_ || !diagnosticAvailable_) return; // no new synchronous disk work if the writer is unavailable
            SYSTEMTIME t{}; GetSystemTime(&t);
            char utc[40]; std::snprintf(utc, sizeof utc, "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",
                t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
            LARGE_INTEGER q{}; QueryPerformanceCounter(&q);
            std::string line = "{\"schema\":1,\"session\":" + wgdiag::Quote(session_) +
                ",\"module\":" + wgdiag::Quote(kLogTag) + ",\"pid\":" + std::to_string(GetCurrentProcessId()) +
                ",\"tid\":" + std::to_string(GetCurrentThreadId()) + ",\"seq\":" + std::to_string(++sequence_) +
                ",\"utc\":" + wgdiag::Quote(utc) + ",\"qpc\":" + std::to_string(q.QuadPart) +
                ",\"event\":" + wgdiag::Quote(kind) + ",\"data\":" + data + "}\n";
            if (!diagnosticAvailable_) return;
            { std::lock_guard<std::mutex> lock(mutex_);
              if (pendingDiagnostics_.size() + line.size() > kMaxPending) { ++diagnosticDropped_; return; }
              pendingDiagnostics_ += line; }
            SetEvent(wake_);
        } catch (...) {}
    }

private:
    explicit Log(const wchar_t* fileName) : path_(ModuleDirectory() + L"\\" + fileName) {
        FILETIME creation{}, exit{}, kernel{}, user{};
        GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user);
        const uint64_t started = (uint64_t(creation.dwHighDateTime) << 32) | creation.dwLowDateTime;
        session_ = std::to_string(GetCurrentProcessId()) + "-" + std::to_string(started);
        const std::string moduleTag = kLogTag;
        diagnosticPath_ = ModuleDirectory() + L"\\wronggpu-diagnostics-" + std::wstring(moduleTag.begin(), moduleTag.end()) + L".jsonl";
        WIN32_FILE_ATTRIBUTE_DATA da{};
        if (GetFileAttributesExW(diagnosticPath_.c_str(), GetFileExInfoStandard, &da) &&
            (da.nFileSizeHigh || da.nFileSizeLow > (16u << 20)))
            MoveFileExW(diagnosticPath_.c_str(), (diagnosticPath_ + L".old").c_str(), MOVEFILE_REPLACE_EXISTING);
        diagnosticFile_ = CreateFileW(diagnosticPath_.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        diagnosticAvailable_ = diagnosticFile_ != INVALID_HANDLE_VALUE;
        WIN32_FILE_ATTRIBUTE_DATA a{};
        if (GetFileAttributesExW(path_.c_str(), GetFileExInfoStandard, &a) && (a.nFileSizeHigh || a.nFileSizeLow > (8u << 20)))
            MoveFileExW(path_.c_str(), (path_ + L".old").c_str(), MOVEFILE_REPLACE_EXISTING);
        file_ = CreateFileW(path_.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        LARGE_INTEGER size{};
        if (file_ != INVALID_HANDLE_VALUE && GetFileSizeEx(file_, &size)) bytes_ = uint64_t(size.QuadPart);
        wake_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        // CreateThread, not std::thread: Get() can first run inside DllMain, where nothing may wait for a new thread
        if ((file_ != INVALID_HANDLE_VALUE || diagnosticFile_ != INVALID_HANDLE_VALUE) && wake_) {
            // The asynchronous writer executes this module's code until process exit.
            HMODULE pinned = nullptr;
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                               reinterpret_cast<LPCWSTR>(&Log::WriterMain), &pinned);
            HANDLE t = pinned ? CreateThread(nullptr, 0, &Log::WriterMain, this, 0, nullptr) : nullptr;
            if (t) {
                CloseHandle(t);
                async_ = true;
            }
        }
        wchar_t exe[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exe, MAX_PATH);
        const wchar_t* name = wcsrchr(exe, L'\\');
        LARGE_INTEGER frequency{}; QueryPerformanceFrequency(&frequency);
        Event("session", "{\"qpc_frequency\":" + std::to_string(frequency.QuadPart) +
              ",\"capture\":\"metadata_only\",\"gpu_completion_proven\":false}");
        Write("==== session start: %ls (process %lu) ====", name ? name + 1 : exe, GetCurrentProcessId());
        if (!async_ || !diagnosticAvailable_) Write("warning: structured support diagnostics unavailable; legacy text logging only");
    }
    static constexpr size_t kMaxPending = 8u << 20;

    void Diagnose(const std::string& text) noexcept {
        try {
            const bool error = text.find("ERROR") != std::string::npos || text.find("NO_PROVIDER") != std::string::npos;
            if (!error) { Event("log", "{\"message\":" + wgdiag::Quote(text) + "}"); return; }
            const auto f = wgdiag::Classify(text);
            uint64_t repeats = 0;
            {  // bug hunt F1: one full incident per code per window; the repeats in between are counted (IncidentGate)
                std::lock_guard<std::mutex> lock(gate_mutex_);
                if (!gate_.Allow(f.code, GetTickCount64(), &repeats)) return;
            }
            failureSeen_ = true;
            metadataPending_ = true;
            const std::string incident = session_ + "-" + kLogTag + "-" + std::to_string(++incidents_);
            wgdiag::Ring snapshot;
            { std::lock_guard<std::mutex> lock(frames_mutex_); snapshot = frames_; }
            Event("failure", "{\"incident\":" + wgdiag::Quote(incident) + ",\"code\":" + wgdiag::Quote(f.code) +
                ",\"observed_at\":" + wgdiag::Quote(f.observedAt) + ",\"domain\":" + wgdiag::Quote(f.domain) +
                ",\"numeric_code\":" + (f.numeric < 0 ? "null" : std::to_string(f.numeric)) +
                ",\"root_cause_proven\":false,\"message\":" + wgdiag::Quote(text) +
                ",\"recent_frames\":" + std::to_string(snapshot.count) +
                ",\"repeats_since_last\":" + std::to_string(repeats) + "}");
            void* addresses[24]{};
            const USHORT count = CaptureStackBackTrace(1, 24, addresses, nullptr);
            std::string stack = "[";
            for (USHORT i = 0; i < count; ++i) {
                if (i) stack += ',';
                // No loader queries on the failing thread. Resolve these PCs using
                // a contemporaneous background module map; missing maps stay unknown.
                stack += "{\"pc\":" + wgdiag::Quote(std::to_string(uintptr_t(addresses[i]))) + "}";
            }
            Event("cpu_detection_stack", "{\"incident\":" + wgdiag::Quote(incident) + ",\"faulting_stack_proven\":false,\"frames\":" + stack + "]}");
            snapshot.Each([&](const wgdiag::Frame& frame) {
                Event("failure_frame", "{\"incident\":" + wgdiag::Quote(incident) + ",\"frame_metadata\":" + wgdiag::Json(frame) + "}");
            });
            const std::string reference = "[support] " + std::string(f.code) + " incident=" + incident +
                "; collect logs with WrongGPU Setup (metadata, not a GPU replay)\r\n";
            Append(reference.data(), int(reference.size()));
        } catch (...) {}
    }

    void Append(const char* data, int size) {
        if (file_ == INVALID_HANDLE_VALUE) return;
        if (!async_) {  // no writer thread: write directly
            std::lock_guard<std::mutex> lock(mutex_);
            DWORD written = 0;
            WriteFile(file_, data, DWORD(size), &written, nullptr);  // FILE_APPEND_DATA: each call is one atomic append
            bytes_ += written;
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_.size() + size_t(size) > kMaxPending) {
                ++dropped_;
                return;
            }
            pending_.append(data, size_t(size));
        }
        bytes_ += uint64_t(size);
        SetEvent(wake_);
    }

    // Moves the queue to the file (the writer thread, or Flush). write_mutex_ keeps the batches in order.
    void WritePending() {
        std::lock_guard<std::mutex> wl(write_mutex_);
        std::string batch;
        std::string diagnostics;
        uint64_t dropped = 0, diagnosticDropped = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            batch.swap(pending_);
            diagnostics.swap(pendingDiagnostics_);
            dropped = dropped_;
            dropped_ = 0;
            diagnosticDropped = diagnosticDropped_;
            diagnosticDropped_ = 0;
        }
        if (dropped) {
            char note[96];
            const int n = std::snprintf(note, sizeof note, "[log] %llu lines dropped (the disk did not keep up)\r\n",
                                        (unsigned long long)dropped);
            batch.append(note, size_t(n));
        }
        if (!batch.empty()) {
            DWORD written = 0;
            WriteFile(file_, batch.data(), DWORD(batch.size()), &written, nullptr);  // whole lines, one append
        }
        if (diagnosticDropped) diagnostics += "{\"schema\":1,\"session\":" + wgdiag::Quote(session_) + ",\"module\":" +
            wgdiag::Quote(kLogTag) + ",\"event\":\"records_dropped\",\"data\":{\"count\":" + std::to_string(diagnosticDropped) + "}}\n";
        if (!diagnostics.empty() && diagnosticFile_ != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER size{};
            if (GetFileSizeEx(diagnosticFile_, &size) && uint64_t(size.QuadPart) + diagnostics.size() > (16u << 20)) {
                CloseHandle(diagnosticFile_);
                diagnosticFile_ = INVALID_HANDLE_VALUE;
                MoveFileExW(diagnosticPath_.c_str(), (diagnosticPath_ + L".old").c_str(), MOVEFILE_REPLACE_EXISTING);
                diagnosticFile_ = CreateFileW(diagnosticPath_.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            }
            DWORD written = 0;
            WriteFile(diagnosticFile_, diagnostics.data(), DWORD(diagnostics.size()), &written, nullptr);
        }
        // Persist error/flight records before optional module/driver metadata work.
        if (async_ && metadataPending_.exchange(false)) LoadedModules();
    }

    static DWORD WINAPI WriterMain(void* self) {
        auto* log = static_cast<Log*>(self);
        for (;;) {
            WaitForSingleObject(log->wake_, 250);
            log->WritePending();
        }
    }

    static std::string Utf8(const wchar_t* text) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
        std::string s(size_t(n > 0 ? n : 0), '\0');
        if (n > 0) WideCharToMultiByte(CP_UTF8, 0, text, -1, &s[0], n, nullptr, nullptr);
        if (!s.empty()) s.pop_back();
        return s;
    }
    void LoadedModules() noexcept {
        // Own-process module metadata only, captured off the render thread.
        // Query the HIP version only if HIP is already loaded; never hipInit.
        try {
            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
            if (snapshot == INVALID_HANDLE_VALUE) { Event("module_snapshot_unavailable", "{\"win32_error\":" + std::to_string(GetLastError()) + "}"); return; }
            MODULEENTRY32W e{}; e.dwSize = sizeof e;
            std::string modules = "[";
            bool first = true;
            size_t kept = 0;
            for (BOOL ok = Module32FirstW(snapshot, &e); ok && kept < 128; ok = Module32NextW(snapshot, &e)) {
                std::wstring name = e.szModule;
                for (auto& c : name) c = wchar_t(towlower(c));
                if (name.find(L"wronggpu") == std::wstring::npos && name.find(L"amdhip") == std::wstring::npos &&
                    name.find(L"fidelityfx") == std::wstring::npos && name.find(L"ffx_fsr") == std::wstring::npos &&
                    name != L"optiscaler.dll" && name != L"dxgi.dll" && name != L"d3d12.dll" &&
                    name != L"amdxc64.dll" && name != L"amdxx64.dll" && name != L"version.dll" && name != L"winmm.dll") continue;
                if (!first) modules += ',';
                ++kept;
                first = false;
                modules += "{\"name\":" + wgdiag::Quote(Utf8(e.szModule)) + ",\"path\":" + wgdiag::Quote(Utf8(e.szExePath)) +
                    ",\"base_address\":" + wgdiag::Quote(std::to_string(uintptr_t(e.modBaseAddr))) +
                    ",\"image_size\":" + std::to_string(e.modBaseSize);
                if (name.find(L"wronggpu") != std::wstring::npos || name.find(L"fidelityfx") != std::wstring::npos ||
                    name.find(L"ffx_fsr") != std::wstring::npos || name == L"amdhip64_7.dll" || name == L"optiscaler.dll") {
                    const std::wstring path = e.szExePath;
                    auto found = moduleHashes_.find(path);
                    if (found == moduleHashes_.end() && moduleHashes_.size() < 64) found = moduleHashes_.emplace(path, FileHash(path)).first;
                    modules += ",\"sha256\":" + (found == moduleHashes_.end() || found->second.empty() ? "null" : wgdiag::Quote(found->second)) +
                        ",\"hash_basis\":\"module_file_on_disk_at_first_observation\"";
                }
                modules += "}";
                if (name == L"amdhip64_7.dll" && !failureSeen_) {
                    HMODULE hip = nullptr;
                    // Hold a reference while using an already loaded public export.
                    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(e.modBaseAddr), &hip)) {
                        using VersionFn = int (*)(int*);
                        const auto version = reinterpret_cast<VersionFn>(reinterpret_cast<void*>(GetProcAddress(hip, "hipRuntimeGetVersion")));
                        if (version) { int value = 0; const int rc = version(&value);
                            Event("hip_runtime_version", "{\"path\":" + wgdiag::Quote(Utf8(e.szExePath)) +
                                ",\"api_return\":" + std::to_string(rc) + ",\"value\":" + (rc == 0 ? std::to_string(value) : "null") + "}"); }
                        FreeLibrary(hip);
                    }
                }
            }
            CloseHandle(snapshot);
            Event("loaded_modules", "{\"modules\":" + modules + "]}");
        } catch (...) {}
    }
    static std::string FileHash(const std::wstring& path) {
        // Hash only allowlisted module files, on the background logger. Windows'
        // public crypto API; no byte content, code or disassembly enters reports.
        HCRYPTPROV provider = 0; HCRYPTHASH hash = 0;
        HANDLE file = INVALID_HANDLE_VALUE;
        std::string result;
        if (CryptAcquireContextW(&provider, nullptr, nullptr, PROV_RSA_AES, CRYPT_VERIFYCONTEXT) &&
            CryptCreateHash(provider, CALG_SHA_256, 0, 0, &hash)) {
            file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            LARGE_INTEGER size{};
            bool ok = file != INVALID_HANDLE_VALUE && GetFileSizeEx(file, &size) && size.QuadPart <= (512ll << 20);
            unsigned char chunk[65536]; DWORD got = 0;
            while (ok) {
                ok = ReadFile(file, chunk, sizeof chunk, &got, nullptr) != FALSE;
                if (!ok || !got) break;
                ok = CryptHashData(hash, chunk, got, 0) != FALSE;
            }
            unsigned char digest[32]; DWORD bytes = sizeof digest;
            if (ok && CryptGetHashParam(hash, HP_HASHVAL, digest, &bytes, 0) && bytes == 32) {
                constexpr char hex[] = "0123456789abcdef";
                for (unsigned char c : digest) { result += hex[c >> 4]; result += hex[c & 15]; }
            }
        }
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        if (hash) CryptDestroyHash(hash);
        if (provider) CryptReleaseContext(provider, 0);
        return result;
    }

    std::mutex mutex_, write_mutex_, frames_mutex_;
    std::wstring path_;
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE diagnosticFile_ = INVALID_HANDLE_VALUE;
    bool diagnosticAvailable_ = false; // fixed before the writer starts; handle rotation stays under write_mutex_
    std::wstring diagnosticPath_;
    std::string session_, pendingDiagnostics_;
    wgdiag::Ring frames_;
    std::map<std::wstring, std::string> moduleHashes_; // writer/Flush serialized by write_mutex_; bounded by allowed modules
    std::atomic<uint64_t> sequence_{0}, incidents_{0};
    std::mutex gate_mutex_;        // guards gate_
    wgdiag::IncidentGate gate_;    // Diagnose: full incidents per failure code (bug hunt F1)
    std::atomic<bool> metadataPending_{true};
    std::atomic<bool> failureSeen_{false};
    uint64_t diagnosticDropped_ = 0;
    HANDLE wake_ = nullptr;
    std::atomic<bool> async_{false};
    std::string pending_;
    uint64_t dropped_ = 0;
    std::atomic<uint64_t> bytes_{0};
};

}  // namespace wronggpu
