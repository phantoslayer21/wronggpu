// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8a): the per-game FG settings keys (fg_settings.h). Written for this project.
#include "fg_settings.h"

namespace wgfgp {

namespace {

struct Spelling {
    const char* text;
    uint32_t value;
};

constexpr Spelling kPresets[] = {{"A", 0}, {"B", 1}, {"C", 2}};
constexpr Spelling kPresenters[] = {{"auto", 0}, {"hwfq", 1}, {"dxgi_tearing", 2}, {"dxgi_vsync", 3}};
constexpr Spelling kMultipliers[] = {{"auto", kFgAutoMultiplier}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"6", 6}};

template <size_t K>
bool Lookup(const Spelling (&table)[K], const std::string& v, uint32_t* out) {
    for (const Spelling& s : table)
        if (v == s.text) {
            *out = s.value;
            return true;
        }
    return false;
}

template <size_t K>
const char* SpellingOf(const Spelling (&table)[K], uint32_t value) {
    for (const Spelling& s : table)
        if (s.value == value) return s.text;
    return nullptr;
}

void Say(std::string* why, const char* text) {
    if (why) *why = text;
}

}  // namespace

bool FgIsSettingsKey(const std::string& key) {
    return key == kFgKeyEnable || key == kFgKeyPreset || key == kFgKeyMultiplier || key == kFgKeyPresenter ||
           key == kFgKeyLowLatency;
}

bool FgParseSetting(const std::string& key, const std::string& value, FgSettings* s, std::string* why) {
    if (!s) {
        Say(why, "no settings to fill");
        return false;
    }
    uint32_t v = 0;
    if (key == kFgKeyEnable) {
#if defined(WG_FG_PROXY_MUTANT) && WG_FG_PROXY_MUTANT == 6  // test-only mutant: a case-folded spelling taken
        if (value == "on" || value == "On" || value == "ON") s->enabled = true;
#else
        if (value == "on") s->enabled = true;
#endif
        else if (value == "off") s->enabled = false;
        else {
            Say(why, "fg takes off or on");
            return false;
        }
        return true;
    }
    if (key == kFgKeyLowLatency) {
        if (value != "off" && value != "on") {
            Say(why, "fg_low_latency takes off or on");
            return false;
        }
        s->low_latency = value == "on";
        return true;
    }
    if (key == kFgKeyPreset) {
        if (!Lookup(kPresets, value, &v)) {
            Say(why, "fg_preset takes A, B or C");
            return false;
        }
        s->preset = v;
        return true;
    }
    if (key == kFgKeyMultiplier) {
        if (!Lookup(kMultipliers, value, &v)) {
            Say(why, "fg_multiplier takes auto, 2, 3, 4, 5 or 6");
            return false;
        }
        s->multiplier = v;
        return true;
    }
    if (key == kFgKeyPresenter) {
        if (!Lookup(kPresenters, value, &v)) {
            Say(why, "fg_presenter takes auto, hwfq, dxgi_tearing or dxgi_vsync");
            return false;
        }
        s->presenter = static_cast<FgPresenterChoice>(v);
        return true;
    }
    Say(why, "not a frame generation key");
    return false;
}

FgSettings FgSettingsFromKeyValues(const FgKeyValues& kv, std::vector<FgSettingsIssue>* issues) {
    FgSettings s;
    constexpr const char* keys[] = {kFgKeyEnable, kFgKeyPreset, kFgKeyMultiplier, kFgKeyPresenter, kFgKeyLowLatency};
    bool seen[sizeof(keys) / sizeof(*keys)] = {};
    for (const auto& p : kv) {
        for (size_t k = 0; k < sizeof(keys) / sizeof(*keys); ++k) {
            if (p.first != keys[k] || seen[k]) continue;
            seen[k] = true;  // a key's first line counts
            std::string why;
            if (!FgParseSetting(p.first, p.second, &s, &why) && issues) issues->push_back({p.first, p.second, why});
        }
    }
    return s;
}

bool FgSettingsValid(const FgSettings& s) {
    return FgPresetSpelling(s.preset) != nullptr && SpellingOf(kMultipliers, s.multiplier) != nullptr &&
           FgPresenterSpelling(s.presenter) != nullptr;
}

FgKeyValues FgSettingsToKeyValues(const FgSettings& s) {
    FgKeyValues kv;
    if (!FgSettingsValid(s)) return kv;
    kv.emplace_back(kFgKeyEnable, s.enabled ? "on" : "off");
    kv.emplace_back(kFgKeyPreset, FgPresetSpelling(s.preset));
    kv.emplace_back(kFgKeyMultiplier, SpellingOf(kMultipliers, s.multiplier));
    kv.emplace_back(kFgKeyPresenter, FgPresenterSpelling(s.presenter));
    if (s.low_latency) kv.emplace_back(kFgKeyLowLatency, "on");
    return kv;
}

bool FgApplySettings(const FgSettings& s, FgKeyValues* kv) {
    if (!kv || !FgSettingsValid(s)) return false;
    auto settings = FgSettingsToKeyValues(s);
    if (!s.low_latency) {
        for (const auto& have : *kv) {
            if (have.first == kFgKeyLowLatency) {
                settings.emplace_back(kFgKeyLowLatency, "off");
                break;
            }
        }
    }
    for (const auto& want : settings) {
        bool replaced = false;
        for (auto& have : *kv)
            if (have.first == want.first) {
                have.second = want.second;
                replaced = true;
                break;  // only the first line of a key is the key (wg_state.h)
            }
        if (!replaced) kv->push_back(want);
    }
    return true;
}

uint32_t FgGeneratedCountOf(uint32_t multiplier) {
    return multiplier >= kFgMinMultiplier && multiplier <= kFgMaxMultiplier ? multiplier - 1 : 0;
}

const char* FgPresetSpelling(uint32_t preset) { return SpellingOf(kPresets, preset); }

const char* FgPresenterSpelling(FgPresenterChoice p) { return SpellingOf(kPresenters, static_cast<uint32_t>(p)); }

}  // namespace wgfgp
