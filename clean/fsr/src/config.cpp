// SPDX-License-Identifier: MIT
#include "config.h"
#include "runtime_keys.h"
#include "wg_registry_ini_gen.h"  // kWgDefaultIniText: the generated [wronggpu] section
#include "wg_settings_io.h"       // pure part only (is_registry_key); the Win32 part is in settings_bridge.cpp

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <mutex>
#include <map>

namespace wgfsr {

// ---- the registry's entries and models, as this build offers them ----------------------------------------------

bool ModelBuiltIn(int32_t srModel) {
    if (srModel < 0 || srModel >= WG_SR_MODEL_COUNT) return false;
    // Models with a build flag (registry models[].build_flag): the one place of the front end that knows them.
    if (std::strcmp(kWgModelRows[srModel].name, "K") == 0) return WRONGGPU_MODEL_K != 0;
    if (std::strcmp(kWgModelRows[srModel].name, "L") == 0) return WRONGGPU_MODEL_L != 0;
    return true;
}

static bool EntryBuiltIn(const WgEntryDef& e) { return e.slot != WG_OPT_SR_MODEL || ModelBuiltIn(e.value); }

int OwnEntryCount() {
    int n = 0;
    for (const WgEntryDef& e : kWgEntries) n += EntryBuiltIn(e) ? 1 : 0;
    return n;
}

const WgEntryDef* OwnEntry(int index) {
    for (const WgEntryDef& e : kWgEntries)
        if (EntryBuiltIn(e) && index-- == 0) return &e;
    return nullptr;
}

const WgEntryDef* OwnEntryOfId(uint64_t id) {
    for (const WgEntryDef& e : kWgEntries)
        if (e.id == id && EntryBuiltIn(e)) return &e;
    return nullptr;
}

// The entry's text without the hidden "##<API level>" (labels are "<text>##<level>", gen_registry.py). One copy per
// entry, kept for the process: callers hold the pointer (provider names, the host's answers).
const char* EntryProviderName(const WgEntryDef& entry) {
    const char* cut = std::strstr(entry.label, "##");
    if (!cut) return entry.label;
    static std::mutex lock;
    static std::map<const WgEntryDef*, std::string> texts;
    std::lock_guard<std::mutex> hold(lock);
    auto it = texts.find(&entry);
    if (it == texts.end()) it = texts.emplace(&entry, std::string(entry.label, cut)).first;
    return it->second.c_str();
}

const WgEntryDef* SrEntryOfModel(int32_t srModel) {
    for (const WgEntryDef& e : kWgEntries)
        if (e.slot == WG_OPT_SR_MODEL && e.value == srModel && EntryBuiltIn(e)) return &e;
    return nullptr;
}

int32_t ModelOfName(const std::string& name) {
    for (int32_t m = 0; m < WG_SR_MODEL_COUNT; ++m) {
        const char* n = kWgModelRows[m].name;
        if (std::strlen(n) != name.size()) continue;
        bool same = true;
        for (size_t i = 0; i < name.size(); ++i)
            same = same && std::tolower(static_cast<unsigned char>(n[i])) == std::tolower(static_cast<unsigned char>(name[i]));
        if (same) return ModelBuiltIn(m) ? m : -1;
    }
    return -1;
}

namespace {

std::string Trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string Lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool ParseBool(const std::string& v, bool& out) {
    const std::string l = Lower(v);
    if (l == "1" || l == "true" || l == "yes" || l == "on") { out = true; return true; }
    if (l == "0" || l == "false" || l == "no" || l == "off") { out = false; return true; }
    return false;
}

bool ParseFloat(const std::string& v, float& out) {
    char* end = nullptr;
    const float f = std::strtof(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0') return false;
    out = f;
    return true;
}

bool ParseSign(const std::string& v, float& out) {
    float f = 0;
    if (!ParseFloat(v, f) || (f != 1.0f && f != -1.0f)) return false;
    out = f;
    return true;
}

}  // namespace

int OverlayKeyCode(const std::string& name) {
    static const struct {
        const char* name;
        int code;
    } kKeys[] = {{"insert", 0x2D}, {"home", 0x24}, {"end", 0x23}, {"delete", 0x2E}, {"pageup", 0x21},
                 {"pagedown", 0x22}, {"pause", 0x13}, {"scrolllock", 0x91}};
    for (const auto& k : kKeys)
        if (name == k.name) return k.code;
    if (name.size() >= 2 && name.size() <= 3 && name[0] == 'f') {  // f1 ... f12 = 0x70 ... 0x7B
        const int n = std::atoi(name.c_str() + 1);
        if (n >= 1 && n <= 12 && std::to_string(n) == name.substr(1)) return 0x6F + n;
    }
    if (name.size() > 2 && name[0] == '0' && (name[1] == 'x' || name[1] == 'X')) {
        char* end = nullptr;
        const long v = std::strtol(name.c_str() + 2, &end, 16);
        if (end && *end == 0 && v > 0 && v < 0xFF) return static_cast<int>(v);
    }
    return 0;
}

std::string NormalizeOptions(const std::string& text) {
    std::string out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find(';', start);
        if (end == std::string::npos) end = text.size();
        const std::string item = Trim(text.substr(start, end - start));
        if (!item.empty()) out += (out.empty() ? "" : ";") + item;
        start = end + 1;
    }
    return out;
}

const char* RuntimeProviderName(const std::string& runtimeOptions, const char* environmentOptions) {
    // The frontend's options reach the interop after key/value whitespace is trimmed; the runtime then scans
    // semicolon/comma-separated model items, with WG_RUNTIME_OPTIONS appended last. Invalid models fail create,
    // so give them a neutral name rather than promising either model in the menu.
    const std::string environment = environmentOptions ? environmentOptions : "";
    // The interop parses the environment as an override before creating the core runtime; the core also reads
    // the original environment string. Retain that second read, including invalid untrimmed model values.
    const std::string options = runtimeOptions + ";" + (environment.size() < 4095 ? environment : "");
    std::string normalized;
    size_t start = 0;
    while (start <= options.size()) {
        size_t end = options.find(';', start);
        if (end == std::string::npos) end = options.size();
        const std::string item = options.substr(start, end - start);
        const size_t equal = item.find('=');
        const std::string key = Trim(item.substr(0, equal));
        normalized += ";" + key + (equal == std::string::npos ? "" : "=" + Trim(item.substr(equal + 1)));
        start = end + 1;
    }
    const std::string text = normalized + ";" + environment;
    // The registry's models (default: the sr_model default), named as their version-list entries name them.
    const WgEntryDef* fallback = SrEntryOfModel(kWgOptions[WG_OPT_SR_MODEL].default_value);
    const char* name = fallback ? EntryProviderName(*fallback) : "WrongGPU";
    for (size_t a = 0; a <= text.size();) {
        size_t b = text.find_first_of(";,", a);
        if (b == std::string::npos) b = text.size();
        const std::string item = text.substr(a, b - a);
        a = b + 1;
        if (item.compare(0, 6, "model=") != 0) continue;
        const WgEntryDef* entry = SrEntryOfModel(ModelOfName(item.substr(6)));  // (a model this build lacks: none)
        if (!entry) return "WrongGPU";
        name = EntryProviderName(*entry);
    }
    return name;
}

std::string WithoutModelK(const std::string& options, bool* removed) { return WithoutModel(options, 'K', removed); }

std::string WithoutModel(const std::string& options, char model, bool* removed) {
    const char upper = char(std::toupper((unsigned char)model)), lower = char(std::tolower((unsigned char)model));
    std::string kept;
    bool any = false;
    for (size_t a = 0; a < options.size();) {
        size_t b = options.find_first_of(";,", a);
        if (b == std::string::npos) b = options.size();
        const std::string item = options.substr(a, b - a);
        const size_t equal = item.find('=');
        const std::string value = equal == std::string::npos ? "" : Trim(item.substr(equal + 1));
        if (equal != std::string::npos && Trim(item.substr(0, equal)) == "model" && value.size() == 1 &&
            (value[0] == upper || value[0] == lower)) {
            any = true;
        } else {
            if (!kept.empty()) kept += options[a - 1];  // the separator in front of the item
            kept += item;
        }
        a = b + 1;
    }
    if (removed) *removed = any;
    return any ? kept : options;
}

#if !WRONGGPU_MODEL_K || !WRONGGPU_MODEL_L
// model=<letter> dropped from the ini's runtime options and from WG_RUNTIME_OPTIONS (also the process environment);
// *ini / *env: where it was.
static void DropModelOption(Config& config, char model, bool* ini, bool* env) {
    config.runtimeOptions = WithoutModel(config.runtimeOptions, model, ini);
    if (const char* value = std::getenv("WG_RUNTIME_OPTIONS")) {
        const std::string kept = WithoutModel(value, model, env);
        if (*env) {
#ifdef _WIN32
            _putenv_s("WG_RUNTIME_OPTIONS", kept.c_str());  // also the process environment; empty: removed
#else
            if (kept.empty()) unsetenv("WG_RUNTIME_OPTIONS");
            else setenv("WG_RUNTIME_OPTIONS", kept.c_str(), 1);
#endif
        }
    }
}
#endif

#if !WRONGGPU_MODEL_L
std::string IgnoreModelL(Config& config) {
    bool ini = false, env = false;
    DropModelOption(config, 'L', &ini, &env);
    if (!ini && !env) return std::string();
    return std::string("note: model=L (") + (ini ? "runtime_options in wronggpu.ini" : "") + (ini && env ? " and " : "") +
           (env ? "WG_RUNTIME_OPTIONS" : "") + ") is refused: Model L is not part of this build of the front end "
           "(CMake WG_MODEL_L); Model M is used";
}
#endif

#if !WRONGGPU_MODEL_K
std::string IgnoreModelK(Config& config) {
    bool ini = false, env = false;
    DropModelOption(config, 'K', &ini, &env);
    if (!ini && !env) return std::string();
    return std::string("note: model=K (") + (ini ? "runtime_options in wronggpu.ini" : "") + (ini && env ? " and " : "") +
           (env ? "WG_RUNTIME_OPTIONS" : "") + ") is ignored: Model K is not part of this build; Model M is used";
}
#endif

Config ParseConfig(const std::string& text, std::string* warnings) {
    Config c;
    std::istringstream in(text);
    std::string line;
    std::string loose;  // runtime options written as lines of their own (runtime_keys.h), appended to runtime_options
    int number = 0;
    bool ours = true;  // inside a section this front end reads
    bool registrySection = false;  // inside [wronggpu]
    auto warn = [&](const std::string& what) {
        if (warnings) *warnings += "wronggpu.ini line " + std::to_string(number) + ": " + what + "\n";
    };
    while (std::getline(in, line)) {
        ++number;
        // A comment starts at '#' or ';' at the start of the line or after whitespace; elsewhere ';' separates
        // runtime_options items.
        for (size_t i = 0; i < line.size(); ++i) {
            if ((line[i] == '#' || line[i] == ';') && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t')) {
                line.resize(i);
                break;
            }
        }
        line = Trim(line);
        if (line.empty()) continue;
        if (line[0] == '[') {
            const std::string section = Lower(Trim(line.substr(1, line.find(']') == std::string::npos ? std::string::npos : line.find(']') - 1)));
            ours = section == "wronggpu" || section == "fsr";
            registrySection = section == "wronggpu";
            continue;
        }
        if (!ours) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) { warn("no '=' in \"" + line + "\""); continue; }
        const std::string key = Lower(Trim(line.substr(0, eq)));
        // The registry's keys of [wronggpu] (sr_model, rr_mode, ...; clean/wgshared) are read through the shared
        // settings (settings_bridge.h), not here.
        if (registrySection && wg::is_registry_key(key)) continue;
        const std::string value = Trim(line.substr(eq + 1));
        const std::string lv = Lower(value);
        bool ok = true;
        if (key == "upscaler") {
            if (lv == "standin" || lv == "stand-in") c.upscaler = UpscalerChoice::StandIn;
            else if (lv == "network") c.upscaler = UpscalerChoice::Network;
            else if (lv == "amd") c.upscaler = UpscalerChoice::Amd;
            else ok = false;
        } else if (key == "runtime_dll") {
            ok = !value.empty();
            if (ok) c.runtimeDll = value;
        } else if (key == "fallback_to_amd") {
            ok = ParseBool(value, c.fallbackToAmd);
        } else if (key == "view") {
            ok = lv == "upscale" || lv == "warp" || lv == "warpdiff" || lv == "motion" || lv == "depth" || lv == "jitter";
            if (ok) c.view = lv;
        } else if (key == "filter") {
            ok = lv == "bilinear" || lv == "lanczos";
            if (ok) c.filter = lv;
        } else if (key == "jitter_compensation") {
            ok = ParseBool(value, c.jitterCompensation);
        } else if (key == "warp_weight") {
            float f = 0;
            ok = ParseFloat(value, f) && f >= 0.0f && f <= 1.0f;
            if (ok) c.warpWeight = f;
        } else if (key == "debug_gain") {
            float f = 0;
            ok = ParseFloat(value, f) && f > 0.0f;
            if (ok) c.debugGain = f;
        } else if (key == "jitter_sign_x") {
            ok = ParseSign(value, c.jitterSign[0]);
        } else if (key == "jitter_sign_y") {
            ok = ParseSign(value, c.jitterSign[1]);
        } else if (key == "motion_sign_x") {
            ok = ParseSign(value, c.motionSign[0]);
        } else if (key == "motion_sign_y") {
            ok = ParseSign(value, c.motionSign[1]);
        } else if (key == "fsr3_route") {
            ok = lv == "auto" || lv == "upscaler" || lv == "fsr3" || lv == "off";
            if (ok) c.fsr3Route = lv;
        } else if (key == "fsr3_amd_first") {
            ok = lv == "auto" || lv == "always" || lv == "never";
            if (ok) c.fsr3AmdFirst = lv;
        } else if (key == "d3d12_hooks") {
            ok = lv == "auto" || lv == "on" || lv == "off";
            if (ok) c.d3d12Hooks = lv;
        } else if (key == "split_test") {
            ok = ParseBool(value, c.splitTest);
        } else if (key == "create_wait_ms") {
            float f = 0;
            ok = ParseFloat(value, f) && f >= 0.0f && f <= 600000.0f;
            if (ok) c.createWaitMs = static_cast<uint32_t>(f);
        } else if (key == "context_park_ms") {
            float f = 0;
            ok = ParseFloat(value, f) && f >= 0.0f && f <= 60000.0f;
            if (ok) c.contextParkMs = static_cast<uint32_t>(f);
        } else if (key == "split_timeout_ms") {
            float f = 0;
            ok = ParseFloat(value, f) && f >= 10.0f && f <= 600000.0f;
            if (ok) c.splitTimeoutMs = static_cast<uint32_t>(f);
        } else if (key == "runtime_options") {
            c.runtimeOptions = NormalizeOptions(value);
        } else if (key == "overlay") {
            ok = lv == "on" || lv == "badge" || lv == "off";
            if (ok) c.overlay = lv;
        } else if (key == "overlay_position") {
            ok = lv == "top-left" || lv == "top-right" || lv == "bottom-left" || lv == "bottom-right";
            if (ok) c.overlayPosition = lv;
        } else if (key == "overlay_key") {
            ok = OverlayKeyCode(lv) != 0;
            if (ok) c.overlayKey = lv;
        } else if (key == "compare_dll") {
            c.compareDll = value;
        } else if (key == "model_dir") {
            // The front end's own key (Config::modelDir), no longer a loose runtime option: the front end looks for the
            // model there first and passes the folder on (CreateUpscale). ';' would end the item in the runtime options.
            ok = value.find(';') == std::string::npos;
            if (ok) c.modelDir = value;
        } else if (key == "compare_key") {
            ok = lv == "off" || OverlayKeyCode(lv) != 0;
            if (ok) c.compareKey = lv;
        } else if (key == "log_dispatches") {
            ok = ParseBool(value, c.logDispatches);
        } else if (key == "log_max_mb") {
            float f = 0;
            ok = ParseFloat(value, f) && f >= 1.0f && f <= 65536.0f;
            if (ok) c.logMaxMegabytes = static_cast<uint32_t>(f);
        } else if (IsRuntimeOptionKey(key)) {
            // A runtime option on a line of its own (e.g. "network = fused" under runtime_options = model=K): the
            // runtime reads it only from runtime_options. Appended there, after every line has been read, so the item
            // is passed on also when the runtime_options line comes later; a line's item comes after the
            // runtime_options items, and the runtime takes a key's last value.
            const std::string item = key + "=" + value;
            loose += (loose.empty() ? "" : ";") + item;
            warn("\"" + key + " = " + value + "\" is a runtime option on a line of its own: appended to runtime_options as " +
                 item + " (write it as runtime_options = ...;" + item + ")");
            continue;
        } else {
            warn("unknown key \"" + key + "\" (ignored)");
            continue;
        }
        if (!ok) warn("bad value \"" + value + "\" for " + key + " (default kept)");
    }
    if (!loose.empty()) c.runtimeOptions = NormalizeOptions(c.runtimeOptions + ";" + loose);
    return c;
}

std::string Config::BackendOptions() const {
    char buffer[256];
    std::snprintf(buffer, sizeof buffer, "view=%s;filter=%s;jitter_compensation=%d;warp_weight=%g;debug_gain=%g;split_test=%d",
                  view.c_str(), filter.c_str(), jitterCompensation ? 1 : 0, warpWeight, debugGain, splitTest ? 1 : 0);
    const std::string extra = NormalizeOptions(runtimeOptions);
    return extra.empty() ? std::string(buffer) : std::string(buffer) + ";" + extra;
}

std::string OptionValue(const std::string& options, const char* key, bool* found) {
    std::string value;
    bool any = false;
    size_t start = 0;
    while (start <= options.size()) {
        size_t end = options.find(';', start);
        if (end == std::string::npos) end = options.size();
        const std::string item = options.substr(start, end - start);
        const size_t equal = item.find('=');
        if (Lower(Trim(item.substr(0, equal))) == key) {
            any = true;
            value = equal == std::string::npos ? std::string() : Trim(item.substr(equal + 1));
        }
        start = end + 1;
    }
    if (found) *found = any;
    return value;
}

ModelChoice ChooseModelDir(const std::vector<ModelPlace>& places, bool modelK, const std::function<bool(const std::string&)>& exists) {
    return ChooseModelDir(places, modelK ? 'K' : 'M', exists);
}

ModelChoice ChooseModelDir(const std::vector<ModelPlace>& places, char model, const std::function<bool(const std::string&)>& exists) {
    ModelChoice c;
    const char* leaf = model == 'K' ? "\\K\\model.wgcm" : model == 'L' ? "\\L\\model.wgcm" : "\\model.wgcm";
    for (size_t i = 0; i < places.size(); ++i) {
        const ModelPlace& p = places[i];
        if (p.dir.empty()) continue;
        std::string dir = p.dir;
        while (dir.size() > 3 && (dir.back() == '\\' || dir.back() == '/')) dir.pop_back();
        const std::string file = dir + leaf;
        if (!c.searched.empty()) c.searched += "; ";
        if (p.runtimeDefault) {
            // Not probed here: the runtime looks there by itself (and names it in its error when the model is missing).
            c.searched += file + " (" + p.source + ": the runtime's own default)";
            c.index = int(i);
            c.file = file;
            return c;
        }
        if (dir.find(';') != std::string::npos) {
            c.searched += file + " (" + p.source + ": skipped, a ';' cannot be passed in the runtime options)";
            continue;
        }
        const bool there = exists && exists(file);
        c.searched += file + " (" + p.source + (there ? ": found)" : ": not there)");
        if (there) {
            c.index = int(i);
            c.file = file;
            c.option = ";model_dir=" + dir;
            return c;
        }
    }
    return c;
}

bool ApplyProcessCompatibility(Config& config, const std::string& executable) {
    const auto basename = [](const std::string& path) {
        const size_t slash = path.find_last_of("/\\");
        return Lower(path.substr(slash == std::string::npos ? 0 : slash + 1));
    };
    const std::string process = basename(executable);
    const bool fenceProfile = process == "crimsondesert.exe" || process == "cyberpunk2077.exe" || process == "controlresonant.exe";
    if (!fenceProfile || config.upscaler != UpscalerChoice::Network ||
        basename(config.runtimeDll) != "wronggpu_runtime.dll") return false;
    const std::string options = NormalizeOptions(config.runtimeOptions);
    size_t start = 0;
    while (start < options.size()) {
        const size_t end = options.find(';', start);
        const std::string item = options.substr(start, end == std::string::npos ? end : end - start);
        const size_t equal = item.find('=');
        if (equal != std::string::npos && Lower(Trim(item.substr(0, equal))) == "ready_wait") return false;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    config.runtimeOptions = options.empty() ? "ready_wait=fence" : options + ";ready_wait=fence";
    return true;
}

std::string Config::Describe() const {
    const char* up = upscaler == UpscalerChoice::StandIn ? "standin" : upscaler == UpscalerChoice::Network ? "network" : "amd";
    char buffer[512];
    std::snprintf(buffer, sizeof buffer,
                  " jitter_sign=(%g,%g) motion_sign=(%g,%g) fsr3_route=%s fsr3_amd_first=%s d3d12_hooks=%s split_timeout_ms=%u "
                  "log_dispatches=%d log_max_mb=%u",
                  jitterSign[0], jitterSign[1], motionSign[0], motionSign[1], fsr3Route.c_str(), fsr3AmdFirst.c_str(),
                  d3d12Hooks.c_str(), splitTimeoutMs, logDispatches ? 1 : 0, logMaxMegabytes);
    // Paths (runtime_dll, runtime_options) can be long: no fixed-size buffer for them.
    return std::string("upscaler=") + up + " runtime_dll=" + runtimeDll + (modelDir.empty() ? "" : " model_dir=" + modelDir) +
           " fallback_to_amd=" + (fallbackToAmd ? "1" : "0") +
           " create_wait_ms=" + std::to_string(createWaitMs) + " context_park_ms=" + std::to_string(contextParkMs) + " " +
           BackendOptions() + buffer;
}

const char* DefaultConfigText() {
    // [wronggpu]: generated from the registry (clean/wgshared, kWgDefaultIniText); [fsr]: the front end's own keys.
    static const std::string text = std::string(kWgDefaultIniText) + "\r\n" + DefaultFsrSectionText();
    return text.c_str();
}

const char* DefaultFsrSectionText() {
    return "# The SR front end's own keys (FSR front end, spec 20). Other sections belong to other WrongGPU components.\r\n"
           "[fsr]\r\n"
           "# Who upscales FSR upscaler contexts: network (Model M, the WrongGPU runtime DLL), amd (forward everything\r\n"
           "# to AMD's original DLL; WrongGPU only logs), standin (a simple D3D12 test upscaler, for diagnostics).\r\n"
           "upscaler = network\r\n"
           "runtime_dll = wronggpu_runtime.dll\r\n"
           "# Folder of the converted model (model.wgcm; Model K in its K subfolder). Empty: the game's data folder\r\n"
           "# (<game exe folder>\\WrongGPU\\model), then the installer's place (%LOCALAPPDATA%\\WrongGPU\\model). A relative\r\n"
           "# path is taken from the data folder (<game exe folder>\\WrongGPU).\r\n"
           "model_dir =\r\n"
           "# On a create or frame failure, hand the context to AMD's FSR (spec 10 section 8).\r\n"
           "fallback_to_amd = 1\r\n"
           "# How long (ms) the game waits for WrongGPU's upscaler to start; after that AMD's FSR upscales until it has\r\n"
           "# started (with fallback_to_amd = 1).\r\n"
           "create_wait_ms = 3000\r\n"
           "# A destroyed upscaler context keeps its upscaler warm for this long (ms); a context created with the same\r\n"
           "# parameters within that time takes it over (dynamic resolution without a hitch at each step). 0: never.\r\n"
           "context_park_ms = 3000\r\n"
           "# Stand-in view: upscale | warp | warpdiff | motion | depth | jitter (see TESTPLAN-20.md)\r\n"
           "view = upscale\r\n"
           "filter = bilinear\r\n"
           "jitter_compensation = 1\r\n"
           "warp_weight = 0.9\r\n"
           "debug_gain = 4\r\n"
           "# Convention switches for test 20-T2 (clean/FORMATS.md). Values: 1 or -1.\r\n"
           "jitter_sign_x = 1\r\n"
           "jitter_sign_y = 1\r\n"
           "motion_sign_x = 1\r\n"
           "motion_sign_y = 1\r\n"
           "# FSR 3.0 component DLLs (ffx_fsr3_x64.dll, ffx_fsr3upscaler_x64.dll): where the stand-in takes over\r\n"
           "# upscaling (auto | upscaler | fsr3 | off) and whether AMD's upscaler runs first on the same frame, which FSR 3.0\r\n"
           "# frame generation needs (auto | always | never).\r\n"
           "fsr3_route = auto\r\n"
           "fsr3_amd_first = auto\r\n"
           "# Same-frame execution (TESTPLAN-20.md, 20-T4/20-T5). d3d12_hooks: WrongGPU's D3D12 command-list hooks:\r\n"
           "# auto (when the upscaler needs them), on (also with upscaler = amd: logs where the game submits the FSR\r\n"
           "# command list), off. split_test = 1: the stand-in upscales through the command-list split.\r\n"
           "d3d12_hooks = auto\r\n"
           "split_test = 0\r\n"
           "split_timeout_ms = 2000\r\n"
           "# Extra keys for the network runtime, passed on unchanged: key=value items separated by ';' without spaces\r\n"
           "# before it (\" ;\" starts a comment), e.g. runtime_options = timing_log=1;sync=pipelined\r\n"
           "runtime_options =\r\n"
           "# In-game overlay: shows whether WrongGPU's upscaler is running. on (full panel for 10 s, then a small\r\n"
           "# badge) | badge | off. overlay_key toggles it: insert, home, end, delete, pageup, pagedown, pause,\r\n"
           "# scrolllock, f1 ... f12, or a virtual-key code such as 0x2d.\r\n"
           "overlay = on\r\n"
           "overlay_key = insert\r\n"
           "# compare_key switches between WrongGPU's upscaler and AMD's FSR while playing (picture and upscaler time\r\n"
           "# side by side); off disables it. Same keys as overlay_key.\r\n"
           "compare_key = home\r\n"
           "# compare_dll: AMD's FidelityFX upscaler DLL to compare with (AMD's FidelityFX SDK 2 DLL gives FSR 4 on\r\n"
           "# Radeon RX 9000). Empty: amd_fidelityfx_upscaler_dx12.dll next to this file, else the FSR 4 DLL the installer\r\n"
           "# found in another game on this PC, else the game's own AMD FSR.\r\n"
           "compare_dll =\r\n"
           "# overlay_position: bottom-left | top-left | top-right | bottom-right\r\n"
           "overlay_position = bottom-left\r\n"
           "# Logging: every FidelityFX call except the per-frame ffxDispatch calls, of which only the first few per\r\n"
           "# context are logged; log_dispatches = 1 logs every frame (diagnostics; larger log).\r\n"
           "log_dispatches = 0\r\n"
           "log_max_mb = 512\r\n";
}

}  // namespace wgfsr
