// SPDX-License-Identifier: MIT
// Lists the DXGI adapters and exits. run_wine_tests.py runs it first with a time limit: a Wine that hangs when a program
// starts Vulkan (the D3D12 tests' first step) hangs here too, in seconds instead of the tests' long limits.
#include <dxgi.h>
#include <cstdio>

int main() {
    IDXGIFactory1* f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&f)))) {
        std::printf("probe: CreateDXGIFactory1 failed\n");
        return 1;
    }
    IDXGIAdapter1* a = nullptr;
    unsigned n = 0;
    while (f->EnumAdapters1(n, &a) == S_OK) {
        a->Release();
        ++n;
    }
    f->Release();
    std::printf("probe: %u adapter(s)\n", n);
    return 0;
}
