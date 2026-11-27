// SPDX-License-Identifier: GPL-3.0-only
// An ASI loader with an adjacent OptiScaler.ini and no plugin API.
#include <windows.h>

extern "C" __declspec(dllexport) DWORD LoadLegacyAsi(void (*initialize)()) {
    initialize();
    return GetCurrentThreadId();
}
