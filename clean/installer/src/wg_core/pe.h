// SPDX-License-Identifier: MIT
// WrongGPU launcher core: PE version resource and export reader (Microsoft PE/COFF and VS_VERSIONINFO documentation),
// moved unchanged from setup.cpp. Written for this project.
#pragma once
#include "util.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// PE version resource (new): reads VS_FIXEDFILEINFO and the StringFileInfo strings directly from the file, following
// Microsoft's public PE/COFF and VS_VERSIONINFO documentation. Used to tell AMD's amd_fidelityfx_dx12.dll from
// WrongGPU's front end and from other mods' replacements, and to show versions in logs.
// ------------------------------------------------------------------------------------------------
// Bounds-checked view of a PE file: headers, sections, data directories.
struct PeImage {
    const uint8_t* b = nullptr; size_t n = 0;
    size_t secOff = 0, nsec = 0, ddOff = 0, ndd = 0;
    bool ok = false;
    uint32_t u16(size_t o) const { return o + 2 <= n ? (uint32_t)(b[o] | (b[o + 1] << 8)) : 0; }
    uint32_t u32(size_t o) const { return o + 4 <= n ? (uint32_t)b[o] | ((uint32_t)b[o + 1] << 8) | ((uint32_t)b[o + 2] << 16) | ((uint32_t)b[o + 3] << 24) : 0; }
    PeImage(const uint8_t* data, size_t size) : b(data), n(size) {
        if (!b || n < 0x40 || b[0] != 'M' || b[1] != 'Z') return;
        const size_t pe = u32(0x3C);
        if (pe + 24 > n || memcmp(b + pe, "PE\0\0", 4) != 0) return;
        nsec = u16(pe + 6);
        const size_t optSize = u16(pe + 20), opt = pe + 24;
        const uint32_t magic = u16(opt);
        if (magic == 0x20B) { ddOff = opt + 112; ndd = u32(opt + 108); }
        else if (magic == 0x10B) { ddOff = opt + 96; ndd = u32(opt + 92); }
        else return;
        secOff = opt + optSize;
        ok = true;
    }
    bool toOff(uint32_t rva, size_t& off) const {
        for (size_t i = 0; i < nsec && secOff + (i + 1) * 40 <= n; ++i) {
            const size_t s = secOff + i * 40;
            const uint32_t vsize = u32(s + 8), va = u32(s + 12), rawSize = u32(s + 16), raw = u32(s + 20);
            const uint32_t span = std::max(vsize, rawSize);
            if (rva >= va && rva < va + span) { off = (size_t)rva - va + raw; return off < n; }
        }
        return false;
    }
    uint32_t dirRva(size_t i) const { return ok && i < ndd && ddOff + (i + 1) * 8 <= n ? u32(ddOff + i * 8) : 0; }
    string cstr(size_t off, size_t max = 256) const { string r; while (off < n && b[off] && r.size() < max) r += (char)b[off++]; return r; }
};

struct PeVersion {
    bool ok = false;
    wstring fileVersion;
    std::map<wstring, wstring> strings;  // CompanyName, ProductName, FileDescription, ...
    wstring Get(const wchar_t* key) const { auto it = strings.find(key); return it == strings.end() ? L"" : it->second; }
};

PeVersion ReadPeVersion(const uint8_t* b, size_t n);
std::set<string> ReadPeExports(const uint8_t* b, size_t n);
std::set<string> ReadPeImports(const uint8_t* b, size_t n);  // imported and delay-loaded DLL names, lower case
PeVersion FileVersionInfo(const wstring& path);
wstring FileVersion(const wstring& path);
// The link time the linker wrote into the COFF header (seconds since 1970, UTC; the launcher links without /Brepro, so it
// is the build's real time): orders two builds of the same version. 0 when the file is no PE image.
uint32_t ReadPeLinkTime(const uint8_t* b, size_t n);
uint32_t FileLinkTime(const wstring& path);
}  // namespace wg
