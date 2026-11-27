// SPDX-License-Identifier: GPL-3.0-only
// The launcher suite's load marker (tests/launcher_build_facts.py self_checks): a DLL that writes <its own path>.loaded when
// it is loaded, so the suite can prove a planted stand-in was never loaded. Not a HIP runtime. Written for this project.
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE self, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        wchar_t path[MAX_PATH + 16] = {0};
        const DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            lstrcatW(path, L".loaded");
            HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        }
    }
    return TRUE;
}
