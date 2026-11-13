// SPDX-License-Identifier: MIT
// One log line, one append (RR guard, 0.6.1). Original code, written for this project.
//
// Two copies of WrongGPU RR in one process (an integrator's kit next to WrongGPU's own install: the case the
// coexistence guard is for) write the same wronggpu-rr.log. Lines used to be written through the C runtime's append
// mode (_wfopen "ab" + fprintf), which on Windows moves to the end and then writes on an ordinary write handle: when
// both copies moved to the same end, one write landed on top of the other and lines were torn (and the launcher's
// Last run no longer matched them). Each copy also had only its own in-module mutex.
//
// Now every write is ONE WriteFile of the whole text (timestamp, line, newline; a rotation notice in the same buffer)
// on a handle opened with FILE_APPEND_DATA alone, which the system appends at the end of the file as one piece. A
// process-wide named mutex, named after the log file's full path, puts the two copies' writes (and the size check and
// rotation) in one order; its wait is bounded, and when it fails the line is still written with the one append call.
#pragma once

#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <wctype.h>

#include <string>

namespace wgrr {

constexpr DWORD kLogLockWaitMs = 2000;  // bounded: a copy stuck holding the lock never stops this one

// "Local\WrongGPU-RR-log-<FNV-1a 64 of the lower-case full path>": the same name in every copy for the same file
inline std::wstring log_mutex_name(const std::wstring& path) {
  wchar_t full[MAX_PATH * 2];
  const DWORD n = GetFullPathNameW(path.c_str(), (DWORD)(sizeof full / sizeof full[0]), full, nullptr);
  std::wstring p = (n && n < sizeof full / sizeof full[0]) ? std::wstring(full, n) : path;
  for (wchar_t& c : p) c = (wchar_t)towlower(c);  // (no user32 in the plugin)
  uint64_t h = 1469598103934665603ull;
  for (wchar_t c : p) {
    h ^= (uint16_t)c;
    h *= 1099511628211ull;
  }
  wchar_t name[64];
  swprintf(name, sizeof name / sizeof name[0], L"Local\\WrongGPU-RR-log-%016llx", (unsigned long long)h);
  return name;
}

// Holds the log's process-wide mutex for one write; owned() is false when the wait failed (the caller writes anyway)
class LogLock {
 public:
  LogLock(const std::wstring& path, DWORD wait_ms) {
    m_ = CreateMutexW(nullptr, FALSE, log_mutex_name(path).c_str());
    if (m_) {
      const DWORD r = WaitForSingleObject(m_, wait_ms);
      owned_ = r == WAIT_OBJECT_0 || r == WAIT_ABANDONED;  // abandoned: the other copy's thread ended; the file is whole
    }
  }
  ~LogLock() {
    if (owned_) ReleaseMutex(m_);
    if (m_) CloseHandle(m_);
  }
  bool owned() const { return owned_; }
  LogLock(const LogLock&) = delete;
  LogLock& operator=(const LogLock&) = delete;

 private:
  HANDLE m_ = nullptr;
  bool owned_ = false;
};

// The file's size now (0 when it is not there)
inline uint64_t log_file_size(const std::wstring& path) {
  WIN32_FILE_ATTRIBUTE_DATA a;
  return GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &a) ? ((uint64_t)a.nFileSizeHigh << 32) | a.nFileSizeLow : 0;
}

// Appends `text` (whole lines with their newlines) in one WriteFile on a FILE_APPEND_DATA handle. False: not written.
inline bool append_log_text(const std::wstring& path, const std::string& text, DWORD disposition = OPEN_ALWAYS) {
  HANDLE h = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         disposition, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  DWORD n = 0;
  const BOOL ok = WriteFile(h, text.data(), (DWORD)text.size(), &n, nullptr);
  CloseHandle(h);
  return ok && n == (DWORD)text.size();
}

// "HH:MM:SS.mmm " of now
inline std::string log_stamp() {
  SYSTEMTIME t;
  GetLocalTime(&t);
  char ts[32];
  snprintf(ts, sizeof ts, "%02u:%02u:%02u.%03u ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
  return ts;
}

}  // namespace wgrr
