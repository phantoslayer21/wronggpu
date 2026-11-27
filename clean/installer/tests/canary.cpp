// SPDX-License-Identifier: GPL-3.0-only
// The launcher suite's negative control (tests/launcher_build_facts.py self_checks): a program that is not a WRONGGPU_TESTING
// build. When it starts it writes <its own path>.started, so the suite can prove its preflight never let it run.
// WG_CANARY_HOOK_STRINGS: it also carries the test hooks' strings, as a program that only looks like a testing build would.
// Written for this project.
#include <windows.h>

#ifdef WG_CANARY_HOOK_STRINGS
const wchar_t* volatile kStrings[] = {L"WRONGGPU_TEST_LIBRARIES", L"WRONGGPU_TEST_ACCESS_LOG", L"WRONGGPU_TEST_EDITION",
                                      L"WrongGPU-CandidateTests"};
#endif

int wmain() {
    wchar_t path[MAX_PATH + 16] = {0};
    DWORD n = GetModuleFileNameW(NULL, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return 2;
    lstrcatW(path, L".started");
    HANDLE h = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 3;
    CloseHandle(h);
#ifdef WG_CANARY_HOOK_STRINGS
    return kStrings[0][0] == L'W' ? 0 : 1;
#else
    return 0;
#endif
}
