// SPDX-License-Identifier: MIT
// A tiny DLL for test_wg_state: a real image that a process can load, standing in for a WrongGPU binary the game has loaded.
// Written for this project.
#include <windows.h>

extern "C" __declspec(dllexport) int WgStateTestDll() { return 219; }

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
