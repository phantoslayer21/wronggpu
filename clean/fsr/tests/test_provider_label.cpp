// SPDX-License-Identifier: MIT
#include "../src/provider_label.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
int main() {
    wgfsr::ProviderLabel label;
    char out[256]{};
    label.Copy(out, sizeof out);
    assert(std::strcmp(out, "WrongGPU (no active frame)") == 0);
    auto expect = [&](wgfsr::UpscaleState state, const char* name, const char* wanted) {
        wgfsr::UpscaleStatus s; s.state = state; s.upscaler = name; label.Update(s);
        const auto n = label.Copy(out, sizeof out);
        assert(n == std::strlen(wanted) + 1 && std::strcmp(out, wanted) == 0);
        char shortOut[2] = {'x','x'}; label.Copy(shortOut, sizeof shortOut); assert(shortOut[0] == 0);
    };
    expect(wgfsr::UpscaleState::Network, "WrongGPU network WrongGPU Model M", "WrongGPU Model M");
#if WRONGGPU_MODEL_K
    expect(wgfsr::UpscaleState::Network, "WrongGPU network WrongGPU Model K", "WrongGPU Model K");
#else  // a build without Model K (CMake WG_MODEL_K=OFF) has no label for it
    expect(wgfsr::UpscaleState::Network, "WrongGPU network WrongGPU Model K", "WrongGPU");
#endif
    expect(wgfsr::UpscaleState::Amd, "4.1.1", "AMD FSR 4.1.1");
    expect(wgfsr::UpscaleState::Compare, "AMD FSR 3.1.5", "AMD FSR 3.1.5");
    expect(wgfsr::UpscaleState::StandIn, "WrongGPU stand-in", "WrongGPU (stand-in)");
    expect(wgfsr::UpscaleState::Failed, "", "Upscaling unavailable");
    std::thread writer([&] {for(int i=0;i<10000;++i) {wgfsr::UpscaleStatus s;s.state=wgfsr::UpscaleState::Network;s.upscaler=i%2?"Model M":"Model K";label.Update(s);}});
    for(int i=0;i<10000;++i) {char b[256]{};const auto n=label.Copy(b,sizeof b);assert(n>0 && n<=sizeof b && std::strlen(b)+1==n);}
    writer.join();
    std::cout << "Provider labels, fallback/failure, capacity and concurrent reads pass\n";
}
