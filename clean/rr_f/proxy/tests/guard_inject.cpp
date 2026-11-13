// SPDX-License-Identifier: MIT
// RR guard: runs a test program with the guard scene (tests/guard_scene.cpp) loaded before its main
// (tools/guard_modelm.py). The program is started suspended, the scene DLL is loaded into it by a remote thread
// (LoadLibraryW), the scene's own thread sets the scene up and signals; then the program's main thread is resumed and
// runs unchanged. Exit code: the program's, or 90 when the scene failed (its result is printed), 91 on a start failure.
//   guard_inject.exe <scene dll> <stage dir> <mode> <program> [args...]     (the working directory is inherited)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>

#include <string>

static std::wstring quote(const std::wstring& a) {
  if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
  std::wstring q = L"\"";
  for (wchar_t c : a) {
    if (c == L'"') q += L'\\';
    q += c;
  }
  return q + L"\"";
}

int wmain(int argc, wchar_t** argv) {
  if (argc < 5) return 91;
  const std::wstring scene = argv[1], stage = argv[2], mode = argv[3];
  std::wstring cmd;
  for (int i = 4; i < argc; ++i) cmd += (cmd.empty() ? L"" : L" ") + quote(argv[i]);
  const DWORD me = GetCurrentProcessId();
  const std::wstring event_name = L"Local\\wgtest-guard-scene-" + std::to_wstring(me);
  const std::wstring result = stage + L"\\scene-result-" + std::to_wstring(me) + L".txt";
  DeleteFileW(result.c_str());
  HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, event_name.c_str());
  SetEnvironmentVariableW(L"WGTEST_GUARD_STAGE", stage.c_str());
  SetEnvironmentVariableW(L"WGTEST_GUARD_MODE", mode.c_str());
  SetEnvironmentVariableW(L"WGTEST_GUARD_EVENT", event_name.c_str());
  SetEnvironmentVariableW(L"WGTEST_GUARD_RESULT", result.c_str());
  STARTUPINFOW si = {sizeof si};
  PROCESS_INFORMATION pi = {};
  if (!ev || !CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
    fprintf(stderr, "guard_inject: cannot start the program (error %lu)\n", GetLastError());
    return 91;
  }
  const size_t bytes = (scene.size() + 1) * sizeof(wchar_t);
  void* remote = VirtualAllocEx(pi.hProcess, nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  SIZE_T written = 0;
  auto load = reinterpret_cast<LPTHREAD_START_ROUTINE>(
      reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW")));
  HANDLE t = nullptr;
  if (remote && WriteProcessMemory(pi.hProcess, remote, scene.c_str(), bytes, &written) && written == bytes)
    t = CreateRemoteThread(pi.hProcess, nullptr, 0, load, remote, 0, nullptr);
  DWORD loaded = 0;
  const bool ok_thread = t && WaitForSingleObject(t, 30000) == WAIT_OBJECT_0 && GetExitCodeThread(t, &loaded) && loaded;
  const bool ok_scene = ok_thread && WaitForSingleObject(ev, 30000) == WAIT_OBJECT_0;
  if (t) CloseHandle(t);
  std::string r;
  if (FILE* f = _wfopen(result.c_str(), L"rb")) {
    char b[1024] = {};
    fread(b, 1, sizeof b - 1, f);
    fclose(f);
    r = b;
  }
  DeleteFileW(result.c_str());
  if (!ok_scene || r.compare(0, 2, "ok") != 0) {
    fprintf(stderr, "guard_inject: scene failed (%s): %s\n", ok_thread ? "scene" : "load", r.c_str());
    TerminateProcess(pi.hProcess, 90);
    WaitForSingleObject(pi.hProcess, 10000);
    return 90;
  }
  printf("guard_inject: %s\n", r.c_str());
  fflush(stdout);
  ResumeThread(pi.hThread);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 91;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(ev);
  return (int)code;
}
