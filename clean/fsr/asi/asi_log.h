// SPDX-License-Identifier: MIT
#pragma once
// Small plain-text log for WrongGPU.asi, written as wronggpu-asi.log into PluginDataDir(), in the same line format as
// the front end's wronggpu-fsr.log ([hh:mm:ss.mmm pid] [tag] text). The plugin logs only at load time, so this is a
// simple synchronous append; it does not share the front end's background-writer log object.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

namespace wgasi {

// Where the plugin keeps its files (wronggpu-asi.log, wronggpu-asi.ini): the OptiScaler folder when the plugin sits in
// its "plugins" folder (where WrongGPU's installer puts it), else the plugin's own folder. Named wronggpu-* there, they
// are found by WrongGPU's log collection and removed by its uninstaller like the front end's own files.
inline std::wstring PluginDataDir() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&PluginDataDir), &self);
    wchar_t file[MAX_PATH] = {};
    GetModuleFileNameW(self, file, MAX_PATH);
    std::wstring dir(file);
    size_t slash = dir.find_last_of(L"\\/");
    dir = slash == std::wstring::npos ? std::wstring() : dir.substr(0, slash);
    slash = dir.find_last_of(L"\\/");
    if (slash != std::wstring::npos && _wcsicmp(dir.c_str() + slash + 1, L"plugins") == 0) dir.resize(slash);
    return dir.empty() ? dir : dir + L"\\";
}

class AsiLog {
public:
    static AsiLog& Get() {
        static AsiLog* log = new AsiLog();  // never destroyed
        return *log;
    }
    void Write(const std::string& text) {
        SYSTEMTIME t;
        GetLocalTime(&t);
        char prefix[64];
        const int n = std::snprintf(prefix, sizeof prefix, "[%02u:%02u:%02u.%03u %lu] [WrongGPU.asi] ", t.wHour, t.wMinute,
                                    t.wSecond, t.wMilliseconds, GetCurrentProcessId());
        std::string line;
        line.append(prefix, n > 0 ? size_t(n) : 0);
        line += text;
        line += "\r\n";
        std::lock_guard<std::mutex> guard(mutex_);
        HANDLE h = CreateFileW(path_.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return;
        DWORD written = 0;
        WriteFile(h, line.data(), DWORD(line.size()), &written, nullptr);
        CloseHandle(h);
    }
    void Writef(const char* fmt, ...) {
        char buf[2048];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);
        Write(buf);
    }

private:
    AsiLog() : path_(PluginDataDir() + L"wronggpu-asi.log") {}
    std::wstring path_;
    std::mutex mutex_;
};

}  // namespace wgasi
