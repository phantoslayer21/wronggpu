// SPDX-License-Identifier: MIT
// Offline anchor-scan test for the WrongGPU ASI plugin. Loads an OptiScaler DLL as a data file (never executes it),
// maps its sections at their RVAs into a buffer the way the Windows loader would, and runs the plugin's resolver
// (clean/fsr/asi/opti_resolver.h) over it. It prints only addresses and yes/no facts and checks that every anchor
// resolves uniquely. It does not patch or run anything.
//
// The binary path comes from the environment variable WRONGGPU_OPTISCALER_BIN (so no OptiScaler binary is stored in
// this repository). When it is unset or unreadable the test exits 77 (ctest SKIP). Point it at the owner's
// read-only copy, e.g. <folder>\OptiScaler-0.9.5.4-pre4.dll.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../asi/opti_resolver.h"

using namespace wgasi;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) { ++g_failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

namespace {

// Map a PE file's sections to an image-layout buffer (sections at their RVAs), as the loader does.
bool MapImage(const std::string& path, std::vector<uint8_t>& image) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> raw(len > 0 ? size_t(len) : 0);
    size_t got = raw.size() ? std::fread(raw.data(), 1, raw.size(), f) : 0;
    std::fclose(f);
    if (got != raw.size() || raw.size() < 0x40) return false;
    uint32_t peoff;
    std::memcpy(&peoff, raw.data() + 0x3C, 4);
    if (size_t(peoff) + 24 > raw.size() || std::memcmp(raw.data() + peoff, "PE\0\0", 4) != 0) return false;
    uint16_t nsec, optsz;
    std::memcpy(&nsec, raw.data() + peoff + 4 + 2, 2);
    std::memcpy(&optsz, raw.data() + peoff + 4 + 16, 2);
    const uint32_t opt = peoff + 4 + 20;
    uint32_t sizeOfImage, sizeOfHeaders;
    std::memcpy(&sizeOfImage, raw.data() + opt + 56, 4);
    std::memcpy(&sizeOfHeaders, raw.data() + opt + 60, 4);
    image.assign(sizeOfImage, 0);
    std::memcpy(image.data(), raw.data(), std::min<size_t>(sizeOfHeaders, raw.size()));
    const uint32_t sec = opt + optsz;
    for (uint16_t i = 0; i < nsec; ++i) {
        const uint32_t s = sec + 40 * i;
        uint32_t vs, va, rs, ro;
        std::memcpy(&vs, raw.data() + s + 8, 4);
        std::memcpy(&va, raw.data() + s + 12, 4);
        std::memcpy(&rs, raw.data() + s + 16, 4);
        std::memcpy(&ro, raw.data() + s + 20, 4);
        const uint32_t n = vs < rs ? vs : rs;
        if (size_t(va) + n <= image.size() && size_t(ro) + n <= raw.size())
            std::memcpy(image.data() + va, raw.data() + ro, n);
    }
    return true;
}

}  // namespace

int main() {
    const char* path = std::getenv("WRONGGPU_OPTISCALER_BIN");
    if (!path || !*path) {
        std::printf("SKIP: set WRONGGPU_OPTISCALER_BIN to an OptiScaler DLL to run the offline anchor scan\n");
        return 77;
    }
    std::vector<uint8_t> image;
    if (!MapImage(path, image)) {
        std::printf("SKIP: could not read/map %s\n", path);
        return 77;
    }
    std::printf("scanning %s (%zu bytes mapped)\n", path, image.size());

    PeImage img(image.data(), image.size());
    CHECK(img.valid());

    Resolved r;
    const bool ok = Resolve(img, r, [](const std::string& s) { std::printf("  %s\n", s.c_str()); });
    CHECK(ok && r.ok);
    if (!r.ok) {
        std::printf("%d checks, %d failures\n", g_checks, g_failures);
        return g_failures ? 1 : 0;
    }

    // Every anchor present and plausible.
    CHECK(r.selectable != 0);
    CHECK(r.addDx12Lo != 0 && r.addDx12Hi > r.addDx12Lo);
    CHECK(r.selectableSites.size() >= 4);  // the combo has several entries (XeSS, FSR 2.1.2, FSR 2.2.1, FSR 3.X, ...)
    for (uint32_t s : r.selectableSites) CHECK(s >= r.addDx12Lo && s < r.addDx12Hi);

    // Independent confirmation: two distinct Selectable call sites inside AddDx12Backends both target r.selectable.
    int agree = 0;
    for (uint32_t s : r.selectableSites) {
        int32_t rel;
        std::memcpy(&rel, img.at(s + 1), 4);
        if (uint32_t(int64_t(s) + 5 + rel) == r.selectable) ++agree;
    }
    CHECK(agree >= 2);
    std::printf("  %d call sites agree on ImGui::Selectable\n", agree);

    std::printf("RESULT selectable=0x%x begincombo=0x%x addDx12=0x%x-0x%x sites=%zu begincombo_site=0x%x\n",
                r.selectable, r.beginCombo, r.addDx12Lo, r.addDx12Hi, r.selectableSites.size(), r.beginComboSite);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
