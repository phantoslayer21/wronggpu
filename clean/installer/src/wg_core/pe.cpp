// SPDX-License-Identifier: MIT
// WrongGPU launcher core: PE version resource and export reader (Microsoft PE/COFF and VS_VERSIONINFO documentation),
// moved unchanged from setup.cpp. Written for this project.
#include "pe.h"

namespace wg {

PeVersion ReadPeVersion(const uint8_t* b, size_t n) {
    PeVersion v;
    const PeImage img(b, n);
    if (!img.ok) return v;
    auto u16 = [&](size_t o) { return img.u16(o); };
    auto u32 = [&](size_t o) { return img.u32(o); };
    const uint32_t rsrcRva = img.dirRva(2);
    size_t rsrc;
    if (!rsrcRva || !img.toOff(rsrcRva, rsrc)) return v;
    // Resource directory: type 16 (RT_VERSION) -> first name -> first language -> data entry.
    auto pick = [&](size_t dir, int wantId, uint32_t& target) {
        if (dir + 16 > n) return false;
        const size_t count = (size_t)u16(dir + 12) + u16(dir + 14);
        for (size_t i = 0; i < count && i < 4096; ++i) {
            const size_t e = dir + 16 + i * 8;
            if (e + 8 > n) return false;
            const uint32_t id = u32(e), off = u32(e + 4);
            if (wantId >= 0 && ((id & 0x80000000u) || (int)id != wantId)) continue;
            target = off;
            return true;
        }
        return false;
    };
    uint32_t e1, e2, e3;
    if (!pick(rsrc, 16, e1) || !(e1 & 0x80000000u)) return v;
    if (!pick(rsrc + (e1 & 0x7FFFFFFFu), -1, e2) || !(e2 & 0x80000000u)) return v;
    if (!pick(rsrc + (e2 & 0x7FFFFFFFu), -1, e3) || (e3 & 0x80000000u)) return v;
    const size_t dataEntry = rsrc + e3;
    size_t vi;
    if (!img.toOff(u32(dataEntry), vi)) return v;
    const size_t viSize = std::min<size_t>(u32(dataEntry + 4), n - vi);
    const size_t viEnd = vi + viSize;
    // VS_VERSIONINFO nodes: wLength, wValueLength, wType, szKey (UTF-16, NUL), padding to 4, value, padding, children.
    struct Node { size_t start, end, value, valueBytes, children; wstring key; uint32_t type; };
    auto align4 = [&](size_t o, size_t base) { return base + ((o - base + 3) & ~(size_t)3); };
    auto node = [&](size_t at, size_t limit, Node& nd) {
        if (at + 6 > limit) return false;
        const size_t len = u16(at);
        if (len < 6 || at + len > limit) return false;
        nd.start = at; nd.end = at + len; nd.type = u16(at + 4);
        size_t p = at + 6;
        nd.key.clear();
        while (p + 2 <= nd.end && u16(p)) { nd.key += (wchar_t)u16(p); p += 2; }
        p += 2;
        nd.value = align4(p, vi);
        const size_t vlen = u16(at + 2);
        nd.valueBytes = nd.type == 1 ? vlen * 2 : vlen;
        if (nd.value + nd.valueBytes > nd.end) nd.valueBytes = nd.end > nd.value ? nd.end - nd.value : 0;
        nd.children = align4(nd.value + nd.valueBytes, vi);
        return true;
    };
    Node root;
    if (!node(vi, viEnd, root) || root.key != L"VS_VERSION_INFO") return v;
    if (root.valueBytes >= 16 && u32(root.value) == 0xFEEF04BDu) {
        const uint32_t ms = u32(root.value + 8), ls = u32(root.value + 12);
        wchar_t buf[64];
        swprintf(buf, 64, L"%u.%u.%u.%u", ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF);
        v.fileVersion = buf;
    }
    v.ok = true;
    for (size_t c = root.children; c < root.end;) {
        Node sfi;
        if (!node(c, root.end, sfi)) break;
        if (sfi.key == L"StringFileInfo") {
            for (size_t t = sfi.children; t < sfi.end;) {
                Node table;
                if (!node(t, sfi.end, table)) break;
                for (size_t s2 = table.children; s2 < table.end;) {
                    Node str;
                    if (!node(s2, table.end, str)) break;
                    wstring val;
                    for (size_t p = str.value; p + 2 <= str.value + str.valueBytes && u16(p); p += 2) val += (wchar_t)u16(p);
                    if (!v.strings.count(str.key)) v.strings[str.key] = val;
                    s2 = align4(str.end, vi);
                }
                t = align4(table.end, vi);
            }
        }
        c = align4(sfi.end, vi);
    }
    return v;
}
// Names in the export directory (Microsoft PE/COFF documentation, ".edata"): used to recognise the FSR 3.0
// component DLLs, which game developers build themselves (no AMD version resource or signature).
std::set<string> ReadPeExports(const uint8_t* b, size_t n) {
    std::set<string> out;
    const PeImage img(b, n);
    size_t ed;
    if (!img.ok || !img.toOff(img.dirRva(0), ed) || !img.dirRva(0)) return out;
    const uint32_t count = img.u32(ed + 24), namesRva = img.u32(ed + 32);
    size_t names;
    if (!img.toOff(namesRva, names)) return out;
    for (uint32_t i = 0; i < count && i < 65536; ++i) {
        size_t off;
        if (img.toOff(img.u32(names + 4 * (size_t)i), off)) out.insert(img.cstr(off));
    }
    return out;
}
// DLL names of the import directory (1) and the delay-load import directory (13), lower case (Microsoft PE/COFF
// documentation, ".idata" and "Delay-Load Import Tables"): which graphics API an executable links against.
std::set<string> ReadPeImports(const uint8_t* b, size_t n) {
    std::set<string> out;
    const PeImage img(b, n);
    if (!img.ok) return out;
    auto lower = [](string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); return s; };
    size_t id;
    if (img.dirRva(1) && img.toOff(img.dirRva(1), id))
        for (size_t i = 0; i < 4096 && id + (i + 1) * 20 <= n; ++i) {
            const size_t d = id + i * 20;
            const uint32_t nameRva = img.u32(d + 12);
            if (!nameRva && !img.u32(d) && !img.u32(d + 16)) break;
            size_t off;
            if (nameRva && img.toOff(nameRva, off)) out.insert(lower(img.cstr(off)));
        }
    if (img.dirRva(13) && img.toOff(img.dirRva(13), id))
        for (size_t i = 0; i < 4096 && id + (i + 1) * 32 <= n; ++i) {
            const size_t d = id + i * 32;
            const uint32_t nameRva = img.u32(d + 4);
            if (!nameRva) break;
            size_t off;
            if (img.toOff(nameRva, off)) out.insert(lower(img.cstr(off)));
        }
    return out;
}
PeVersion FileVersionInfo(const wstring& path) {
    MappedFile f(path, 2ull << 30);
    return ReadPeVersion(f.data, f.size);
}
wstring FileVersion(const wstring& path) {
    PeVersion v = FileVersionInfo(path);
    return v.fileVersion.empty() ? L"unknown" : v.fileVersion;
}
uint32_t ReadPeLinkTime(const uint8_t* b, size_t n) {
    const PeImage img(b, n);
    return img.ok ? img.u32(img.u32(0x3C) + 8) : 0;  // the COFF header's TimeDateStamp, after the PE signature, Machine and NumberOfSections
}
uint32_t FileLinkTime(const wstring& path) {
    MappedFile f(path, 2ull << 30);
    return ReadPeLinkTime(f.data, f.size);
}

}  // namespace wg
