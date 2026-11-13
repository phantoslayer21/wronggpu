// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU OptiScaler ASI plugin: resolve the anchors of OptiScaler's DX12 upscaler menu in a mapped PE image,
// by content (UI label strings, their RIP-relative references, function bounds from the exception directory),
// never by file name or fixed offsets. Pure logic over a mapped image buffer; no Windows API, no OptiScaler code.
// The same code runs on the live module (a loaded module is already mapped at its RVAs) and, in the offline test,
// on a buffer into which the test has mapped the file's sections at their RVAs (tests/test_asi_scan.cpp).
//
// What it finds, for OptiScaler 0.9.5-pre4 and any build with the same menu shape:
//   - the ImGui::Selectable(const char*, bool, int, const ImVec2&) function, as the call target shared by several
//     menu entries whose label is a string literal (robust: >= 3 distinct labels must agree on one target);
//   - MenuCommon::AddDx12Backends, as the one menu function that references "FSR 3.X" but no "... w/Dx12" label;
//   - every Selectable call site inside that function (the combo entries, including the FSR 3.X one we inject at);
//   - ImGui::BeginCombo and its call site in that function (for the closed-combo preview text), when unambiguous.
// If anything is missing or ambiguous (a different OptiScaler build), Resolve returns false and the plugin hooks
// nothing. Addresses are image RVAs; the caller adds the module base.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace wgasi {

using LogFn = std::function<void(const std::string&)>;

inline std::string Hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "%x", v);
    return b;
}

// A PE image laid out in memory at its RVAs (a loaded module, or the offline test's section-mapped buffer).
class PeImage {
public:
    PeImage(const uint8_t* base, size_t size) : base_(base), size_(size) { Parse(); }
    bool valid() const { return valid_; }

    const uint8_t* at(uint32_t rva) const { return rva < size_ ? base_ + rva : nullptr; }
    bool contains(uint32_t rva, uint32_t len = 1) const { return uint64_t(rva) + len <= size_; }

    struct Range { uint32_t rva = 0, size = 0; };
    Range text() const { return text_; }
    Range rdata() const { return rdata_; }

    // RVAs of a C string (content + NUL, and the preceding byte is NUL so it is a whole string) in the read-only data.
    std::vector<uint32_t> findString(const char* text) const {
        std::vector<uint32_t> out;
        const size_t n = std::strlen(text);
        const Range r = rdata_.size ? rdata_ : Range{0, uint32_t(size_)};
        if (!r.size || n + 1 > r.size) return out;
        const uint8_t* b = base_ + r.rva;
        for (uint32_t i = 0; i + n + 1 <= r.size; ++i)
            if (b[i + n] == 0 && std::memcmp(b + i, text, n) == 0 && (i == 0 || b[i - 1] == 0))
                out.push_back(r.rva + i);
        return out;
    }

    // RVAs (of the disp32 field) of every RIP-relative reference in .text to `target` (… disp32 where
    // disp32 + pos + 4 == target), i.e. the operand of a lea/mov/call/etc. that addresses `target`.
    std::vector<uint32_t> ripRefs(uint32_t target) const {
        std::vector<uint32_t> out;
        if (!text_.size) return out;
        const uint8_t* b = base_ + text_.rva;
        for (uint32_t i = 0; i + 4 <= text_.size; ++i) {
            int32_t disp;
            std::memcpy(&disp, b + i, 4);
            if (uint32_t(int64_t(text_.rva) + i + 4 + disp) == target) out.push_back(text_.rva + i);
        }
        return out;
    }

    // Target RVA of the first `E8 rel32` (near call) at or after `from` within `window` bytes, else 0.
    uint32_t nextCallTarget(uint32_t from, uint32_t window = 64) const {
        const uint32_t end = from + window;
        for (uint32_t p = from; p + 5 <= end && contains(p, 5); ++p)
            if (base_[p] == 0xE8) {
                int32_t rel;
                std::memcpy(&rel, base_ + p + 1, 4);
                return uint32_t(int64_t(p) + 5 + rel);
            }
        return 0;
    }

    // Every `E8 rel32` call site (RVA of the E8 byte) in [lo, hi) whose target is `target`.
    std::vector<uint32_t> callSites(uint32_t lo, uint32_t hi, uint32_t target) const {
        std::vector<uint32_t> out;
        for (uint32_t p = lo; p + 5 <= hi && contains(p, 5); ++p)
            if (base_[p] == 0xE8) {
                int32_t rel;
                std::memcpy(&rel, base_ + p + 1, 4);
                if (uint32_t(int64_t(p) + 5 + rel) == target) out.push_back(p);
            }
        return out;
    }

    // Function bounds [begin, end) from the exception directory (.pdata RUNTIME_FUNCTION table) containing `rva`.
    bool functionBounds(uint32_t rva, uint32_t& begin, uint32_t& end) const {
        for (uint32_t p = pdata_.rva; p + 12 <= pdata_.rva + pdata_.size; p += 12) {
            uint32_t b, e;
            std::memcpy(&b, base_ + p, 4);
            std::memcpy(&e, base_ + p + 4, 4);
            if (rva >= b && rva < e) { begin = b; end = e; return true; }
        }
        return false;
    }

private:
    void Parse() {
        if (size_ < 0x40) return;
        uint32_t peoff;
        std::memcpy(&peoff, base_ + 0x3C, 4);
        if (uint64_t(peoff) + 24 > size_ || std::memcmp(base_ + peoff, "PE\0\0", 4) != 0) return;
        const uint32_t coff = peoff + 4;
        uint16_t nsec, optsz;
        std::memcpy(&nsec, base_ + coff + 2, 2);
        std::memcpy(&optsz, base_ + coff + 16, 2);
        const uint32_t opt = coff + 20;
        uint16_t magic;
        std::memcpy(&magic, base_ + opt, 2);
        if (magic != 0x20B) return;  // PE32+ (x64) only
        uint32_t ndir;
        std::memcpy(&ndir, base_ + opt + 108, 4);
        if (ndir > 3) {
            std::memcpy(&pdata_.rva, base_ + opt + 112 + 8 * 3, 4);
            std::memcpy(&pdata_.size, base_ + opt + 112 + 8 * 3 + 4, 4);
        }
        const uint32_t sec = opt + optsz;
        for (uint16_t i = 0; i < nsec; ++i) {
            const uint32_t s = sec + 40 * i;
            if (s + 40 > size_) return;
            char name[9] = {};
            std::memcpy(name, base_ + s, 8);
            uint32_t vs, va;
            std::memcpy(&vs, base_ + s + 8, 4);
            std::memcpy(&va, base_ + s + 12, 4);
            if (std::strcmp(name, ".text") == 0) text_ = {va, vs};
            else if (std::strcmp(name, ".rdata") == 0) rdata_ = {va, vs};
        }
        valid_ = text_.size != 0;
    }

    const uint8_t* base_;
    size_t size_;
    Range text_{}, rdata_{}, pdata_{};
    bool valid_ = false;
};

struct Resolved {
    uint32_t selectable = 0;       // ImGui::Selectable RVA
    uint32_t beginCombo = 0;       // ImGui::BeginCombo RVA (0 when not resolved: preview hook skipped)
    uint32_t addDx12Lo = 0, addDx12Hi = 0;  // MenuCommon::AddDx12Backends bounds
    std::vector<uint32_t> selectableSites;  // Selectable call sites (RVA of the E8 byte) inside AddDx12Backends
    uint32_t beginComboSite = 0;   // BeginCombo call site inside AddDx12Backends (0 when not resolved)
    bool ok = false;
};

// The menu label literals the resolver anchors on (plain UI text of OptiScaler's pre4 DX12 upscaler combo). The
// resolver needs no vendor-specific entry names: it triangulates on the FSR/XeSS entries and injects at "FSR 3.X".
inline const char* const* MenuLabels(size_t& n) {
    static const char* const labels[] = {"XeSS", "FSR 2.1.2", "FSR 2.2.1", "FSR 3.X"};
    n = sizeof(labels) / sizeof(labels[0]);
    return labels;
}
inline const char* const* WithDx12Labels(size_t& n) {
    static const char* const labels[] = {"XeSS w/Dx12", "FSR 2.1.2 w/Dx12", "FSR 2.2.1 w/Dx12", "FSR 3.X w/Dx12"};
    n = sizeof(labels) / sizeof(labels[0]);
    return labels;
}

// Resolve the anchors. Returns false (and leaves r.ok false) if anything is missing or ambiguous; logs each step.
inline bool Resolve(const PeImage& img, Resolved& r, const LogFn& log) {
    auto say = [&](const std::string& s) { if (log) log(s); };
    if (!img.valid()) { say("image is not a PE32+ with a .text section"); return false; }

    size_t nLit = 0;
    const char* const* lit = MenuLabels(nLit);
    // Labels whose combo Selectable is called with a string literal (FSR 3.X is called via a std::string, so it is
    // not usable to triangulate the Selectable address).
    const char* const triang[] = {"XeSS", "FSR 2.1.2", "FSR 2.2.1"};

    // 1. Every menu label literal must be present.
    for (size_t i = 0; i < nLit; ++i)
        if (img.findString(lit[i]).empty()) { say(std::string("menu label not found: ") + lit[i]); return false; }

    // 2. ImGui::Selectable = the call target on which several literal-label entries agree.
    std::vector<std::pair<uint32_t, int>> tally;  // target -> number of distinct labels that reach it
    for (const char* label : triang) {
        std::vector<uint32_t> targetsForLabel;
        for (uint32_t s : img.findString(label))
            for (uint32_t ref : img.ripRefs(s))
                if (uint32_t t = img.nextCallTarget(ref + 4)) {
                    bool seen = false;
                    for (uint32_t x : targetsForLabel) seen |= (x == t);
                    if (!seen) targetsForLabel.push_back(t);
                }
        for (uint32_t t : targetsForLabel) {
            bool found = false;
            for (auto& e : tally) if (e.first == t) { ++e.second; found = true; }
            if (!found) tally.push_back({t, 1});
        }
    }
    uint32_t best = 0;
    int bestCount = 0, bestTies = 0;
    for (auto& e : tally) {
        if (e.second > bestCount) { best = e.first; bestCount = e.second; bestTies = 1; }
        else if (e.second == bestCount) ++bestTies;
    }
    if (bestCount < 3 || bestTies != 1) {
        say("could not identify ImGui::Selectable (need >= 3 agreeing labels and a unique winner; got count=" +
            std::to_string(bestCount) + " ties=" + std::to_string(bestTies) + ")");
        return false;
    }
    r.selectable = best;
    say("ImGui::Selectable at RVA 0x" + Hex(best) + " (" + std::to_string(bestCount) + " labels agree)");

    // 3. AddDx12Backends = the menu function that references "FSR 3.X" but no "... w/Dx12" label.
    size_t nWd = 0;
    const char* const* wd = WithDx12Labels(nWd);
    std::vector<std::pair<uint32_t, uint32_t>> withDx12Funcs;  // functions that reference any "... w/Dx12" label
    for (size_t i = 0; i < nWd; ++i)
        for (uint32_t s : img.findString(wd[i]))
            for (uint32_t ref : img.ripRefs(s)) {
                uint32_t b, e;
                if (img.functionBounds(ref, b, e)) withDx12Funcs.push_back({b, e});
            }
    auto hasWithDx12 = [&](uint32_t b, uint32_t e) {
        for (auto& f : withDx12Funcs) if (f.first == b && f.second == e) return true;
        return false;
    };
    std::vector<std::pair<uint32_t, uint32_t>> candidates;
    for (uint32_t s : img.findString("FSR 3.X"))
        for (uint32_t ref : img.ripRefs(s)) {
            uint32_t b, e;
            if (!img.functionBounds(ref, b, e) || hasWithDx12(b, e)) continue;
            if (img.callSites(b, e, r.selectable).size() < 3) continue;  // the upscaler combo has several entries
            bool dup = false;
            for (auto& c : candidates) dup |= (c.first == b && c.second == e);
            if (!dup) candidates.push_back({b, e});
        }
    if (candidates.size() != 1) {
        say("could not uniquely identify AddDx12Backends (candidate functions: " + std::to_string(candidates.size()) + ")");
        return false;
    }
    r.addDx12Lo = candidates[0].first;
    r.addDx12Hi = candidates[0].second;
    say("AddDx12Backends at RVA 0x" + Hex(r.addDx12Lo) + "-0x" + Hex(r.addDx12Hi));

    // 4. Selectable call sites inside AddDx12Backends (the combo items).
    r.selectableSites = img.callSites(r.addDx12Lo, r.addDx12Hi, r.selectable);
    if (r.selectableSites.size() < 3) {
        say("too few Selectable call sites in AddDx12Backends: " + std::to_string(r.selectableSites.size()));
        return false;
    }
    say(std::to_string(r.selectableSites.size()) + " Selectable call sites in AddDx12Backends");

    // 5. BeginCombo (optional, for the closed-combo preview): the call nearest before the first Selectable site whose
    //    target is a real .text address called exactly once in the function (BeginCombo opens the combo the
    //    Selectables fill). A byte that merely happens to be 0xE8 is filtered out by the .text-target test.
    uint32_t firstSel = r.selectableSites.front();
    for (uint32_t s : r.selectableSites) if (s < firstSel) firstSel = s;
    const PeImage::Range text = img.text();
    uint32_t bcSite = 0, bcTarget = 0;
    for (uint32_t p = r.addDx12Lo; p + 5 <= firstSel; ++p) {
        if (!img.at(p) || *img.at(p) != 0xE8) continue;
        int32_t rel;
        std::memcpy(&rel, img.at(p + 1), 4);
        const uint32_t t = uint32_t(int64_t(p) + 5 + rel);
        if (t < text.rva || t >= text.rva + text.size || t == r.selectable) continue;  // not a real call / is Selectable
        if (img.callSites(r.addDx12Lo, r.addDx12Hi, t).size() != 1) continue;          // BeginCombo is called once here
        bcSite = p;  // keep the last such call before the first Selectable
        bcTarget = t;
    }
    if (bcSite) {
        r.beginCombo = bcTarget;
        r.beginComboSite = bcSite;
        say("ImGui::BeginCombo at RVA 0x" + Hex(bcTarget) + ", call site 0x" + Hex(bcSite));
    } else {
        say("BeginCombo not uniquely resolved; the closed-combo preview text will not be relabelled");
    }

    r.ok = true;
    return true;
}

}  // namespace wgasi
