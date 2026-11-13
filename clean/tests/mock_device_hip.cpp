// SPDX-License-Identifier: MIT
// Own ABI mock, no GPU operations. Used only by test_device_targets.cpp.
#include <cstdlib>
#include <cstring>
#include <windows.h>
#define API extern "C" __declspec(dllexport)
static const char* arch() { static char value[32]; return GetEnvironmentVariableA("WG_MOCK_ARCH", value, sizeof value) ? value : "gfx1201"; }
API int hipInit(unsigned) { return 0; }
API int hipGetDeviceCount(int* count) { *count = 1; return 0; }
API int hipDeviceGetName(char* name, int size, int) { std::strncpy(name, "WrongGPU synthetic device", size); return 0; }
API int hipGetDevicePropertiesR0600(void* props, int) { std::strcpy(static_cast<char*>(props) + 1160, arch()); return 0; }
API int hipSetDevice(int) { return 0; }
API int hipStreamCreate(void** out) { *out = reinterpret_cast<void*>(1); return 0; }
API int hipEventCreate(void** out) { *out = reinterpret_cast<void*>(2); return 0; }
API int hipModuleLoadData(void** out, const void* data) {
    if (std::strcmp(static_cast<const char*>(data), arch()) != 0) return 209;
    *out = reinterpret_cast<void*>(3); return 0;
}
API const char* hipGetErrorString(int) { return "synthetic target mismatch"; }
API int hipMalloc(...) { return 0; }
API int hipFree(...) { return 0; }
API int hipMemcpyAsync(...) { return 0; }
API int hipMemsetAsync(...) { return 0; }
API int hipStreamDestroy(...) { return 0; }
API int hipStreamSynchronize(...) { return 0; }
API int hipModuleUnload(...) { return 0; }
API int hipModuleGetFunction(...) { return 0; }
API int hipModuleLaunchKernel(...) { return 0; }
API int hipEventDestroy(...) { return 0; }
API int hipEventRecord(...) { return 0; }
API int hipEventSynchronize(...) { return 0; }
API int hipEventElapsedTime(...) { return 0; }
