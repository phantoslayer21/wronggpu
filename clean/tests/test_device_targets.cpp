// SPDX-License-Identifier: MIT
// Own HIP mock tests: exact architecture, device matching and fail-closed module selection.
#include "../runtime/device.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <windows.h>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string root(argv[1]);
    auto write = [&](const std::string& name, const char* text) {
        FILE* fp = std::fopen((root + "/" + name).c_str(), "wb");
        if (!fp) std::abort();
        std::fwrite(text, 1, std::strlen(text) + 1, fp); std::fclose(fp);
    };
    write("gfx1200.encoder.co", "gfx1200");
    write("gfx1201.encoder.co", "gfx1201");
    write("gfx1100.encoder.co", "gfx1100");
    write("gfx1101.encoder.co", "gfx1101");
    write("encoder.co", "wrong architecture: must never load this unqualified file");
    int checks = 0;
    // The supported targets are the exact names of clean/runner/family.py: gfx12 and gfx11 parts, each with its own code objects.
    for (const char* arch : {"gfx1200", "gfx1201", "gfx1100", "gfx1101"}) {
        SetEnvironmentVariableA("WG_MOCK_ARCH", arch);
        std::string error;
        auto dev = wgrt::make_hip_device(root, &error, "WrongGPU synthetic device");
        if (!dev || !dev->load_module(root + "/encoder.co")) return 3;
        ++checks;
        if (dev->arch() != arch) return 8;
        ++checks;
        if (dev->load_module(root + "/missing.co")) return 4;
        ++checks;
        write(std::string(arch) + ".bad.co", "wrong-target");
        if (dev->load_module(root + "/bad.co")) return 5;
        ++checks;
        if (wgrt::make_hip_device(root, &error, "a different adapter")) return 6;
        ++checks;
    }
    // No prefix match: gfx1102 / gfx1103 / gfx115x are untested gfx11 parts, gfx1170-1172 are gfx11 names with the gfx12 matrix design
    for (const char* arch : {"gfx1102", "gfx1103", "gfx1151", "gfx1170", "gfx1299", "gfx11", "unknown"}) {
        SetEnvironmentVariableA("WG_MOCK_ARCH", arch);
        std::string error;
        if (wgrt::make_hip_device(root, &error)) return 7;
        ++checks;
    }
    std::printf("%d synthetic device/target checks passed (no physical GPU used)\n", checks);
    return 0;
}
