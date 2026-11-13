// SPDX-License-Identifier: MIT
/* A stand-in for the AMD HIP runtime with the few public HIP entry points the installer's probe calls (test only).
   MOCK_DEVICE_NAME and MOCK_ARCH select what it reports. Like a real runtime confused by a stray LLVM_PATH, it fails
   hipInit when LLVM_PATH is set, so the tests can check that the installer clears it. Written for this project. */
#include <stdlib.h>
#include <string.h>
#ifndef MOCK_DEVICE_NAME
#define MOCK_DEVICE_NAME "AMD Radeon RX 9070 XT"
#endif
#ifndef MOCK_ARCH
#define MOCK_ARCH "gfx1201"
#endif
static char block[256];
__declspec(dllexport) int hipInit(unsigned flags) { const char* llvm = getenv("LLVM_PATH"); (void)flags; return llvm && *llvm ? 3 : 0; }
__declspec(dllexport) int hipGetDeviceCount(int* n) { *n = 1; return 0; }
__declspec(dllexport) int hipDeviceGetName(char* name, int len, int dev) { (void)dev; strncpy(name, MOCK_DEVICE_NAME, (size_t)len); return 0; }
/* The real structure starts with the device name; the architecture name sits much further in. */
__declspec(dllexport) int hipGetDevicePropertiesR0600(void* props, int dev) {
    (void)dev;
    strcpy((char*)props, MOCK_DEVICE_NAME);
    strcpy((char*)props + 1160, MOCK_ARCH ":sramecc-:xnack-");
    return 0;
}
__declspec(dllexport) int hipSetDevice(int dev) { (void)dev; return 0; }
__declspec(dllexport) int hipMalloc(void** p, size_t n) { *p = n <= sizeof block ? block : 0; return *p ? 0 : 2; }
__declspec(dllexport) int hipFree(void* p) { (void)p; return 0; }
