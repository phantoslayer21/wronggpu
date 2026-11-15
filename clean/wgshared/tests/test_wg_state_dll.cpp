// SPDX-License-Identifier: MIT
// A tiny DLL for test_wg_state: a real image that a process can load, standing in for a WrongGPU binary the game has loaded.
// Built twice: WG_TEST_DLL_VALUE 219 (the "old" binary) and 220 (the "new" one), so a test can tell which complete file a
// process loaded. Written for this project.
#include <windows.h>

#ifndef WG_TEST_DLL_VALUE
#define WG_TEST_DLL_VALUE 219
#endif

extern "C" __declspec(dllexport) int WgStateTestDll() { return WG_TEST_DLL_VALUE; }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
