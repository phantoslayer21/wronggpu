// SPDX-License-Identifier: MIT
#pragma once
// Front-end configuration, read from wronggpu.ini next to the DLL (the player's one WrongGPU settings file; spec 30). Plain C++, no Windows headers, so the unit
// tests can parse configurations on any host.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "model_k.h"
#include "model_l.h"
#include "wg_registry_gen.h"  // clean/wgshared: options, version-list entries, models (plain C, no Windows headers)

namespace wgfsr {

// WrongGPU's own entries in the host's version list, in list order: the registry's kWgEntries (SR models first, then
// the RR modes), without the SR models this build leaves out (ModelBuiltIn). Adding a mode or a model to the registry
// adds its entry here with no code change.
bool ModelBuiltIn(int32_t srModel);     // the sr_model value is part of this build (Model K: CMake WG_MODEL_K)
int OwnEntryCount();
const WgEntryDef* OwnEntry(int index);  // 0 .. OwnEntryCount() - 1, else null
const WgEntryDef* OwnEntryOfId(uint64_t id);  // the listed entry with this id, else null
// "WrongGPU Model K" / "WrongGPU RR fast": the entry's label without the hidden "##<API level>" (the provider name).
const char* EntryProviderName(const WgEntryDef& entry);
// The SR entry of an sr_model value (its id is the provider version id of a context running that model), or null.
const WgEntryDef* SrEntryOfModel(int32_t srModel);
// The sr_model value named `name` ("M", "k"; registry models, any case) that this build has, else -1.
int32_t ModelOfName(const std::string& name);

enum class UpscalerChoice { StandIn, Network, Amd };

struct Config {
    UpscalerChoice upscaler = UpscalerChoice::Network;  // Model M (the release default; standin is for diagnostics)
    std::string runtimeDll = "wronggpu_runtime.dll";  // network implementation (spec 10), loaded from the DLL folder
    bool fallbackToAmd = true;                        // spec 10 section 8: hand the context to AMD's FSR on failure
    // How long a game thread waits for the upscaler's create (async_create.h); after that AMD's FSR upscales until the
    // create finishes (with fallbackToAmd; without it the game thread waits until the create finishes).
    uint32_t createWaitMs = 3000;
    // A destroyed upscaler context keeps its upscaler warm ("parked") for this long; a context created with the same
    // device, sizes, flags, options and model within that time takes it over instead of creating one (games that
    // re-create their upscaler at every step of a dynamic resolution). One parked upscaler at most; 0: never.
    uint32_t contextParkMs = 3000;

    // Stand-in options, passed to the upscaler as "key=value;..." (see StandInOptions in standin_d3d12.cpp).
    std::string view = "upscale";     // upscale | warp | warpdiff | motion | depth | jitter
    std::string filter = "bilinear";  // bilinear | lanczos
    bool jitterCompensation = true;
    float warpWeight = 0.9f;
    float debugGain = 4.0f;

    // Convention switches for test 20-T2 (FORMATS.md, "FSR 3.1 to network conventions"). Defaults: spec 20 draft 2
    // (network jitter = +jitterOffset, motion = +motionVectorScale * raw); the spec team may flip them without a rebuild.
    float jitterSign[2] = {1.0f, 1.0f};
    float motionSign[2] = {1.0f, 1.0f};

    // FSR 3.0 component DLLs (ffx_fsr3_x64.dll, ffx_fsr3upscaler_x64.dll; spec 20 integration point 2).
    // fsr3Route: where the stand-in takes over upscale dispatches: "auto" (in ffx_fsr3upscaler_x64.dll when it sees the
    // dispatch, else after ffxFsr3ContextDispatchUpscale), "upscaler", "fsr3", or "off" (log and forward only).
    std::string fsr3Route = "auto";
    // fsr3AmdFirst: run AMD's upscaler before the stand-in on the same frame: "auto" (when the dispatch carries the
    // resources FSR 3.0 frame generation reads from the upscaler), "always", "never".
    std::string fsr3AmdFirst = "auto";

    // Same-frame split (interface version 2, ANSWERS.md spec 20 item 16; d3d12_hooks.h).
    // d3d12Hooks: WrongGPU's D3D12 vtable hooks: "auto" (when a backend sets WG_BACKEND_NEEDS_SPLIT), "on" (also at the
    // first upscale dispatch, e.g. with upscaler = amd for the submission trace of test 20-T4), "off".
    std::string d3d12Hooks = "auto";
    // splitTest: the stand-in upscales through the split (test 20-T5): it splits the game's command list at the dispatch,
    // a WrongGPU queue signals resume right after done (in place of the network's HIP work), and the stand-in records
    // its upscale into the second half.
    bool splitTest = false;
    uint32_t splitTimeoutMs = 2000;  // a split list the game does not submit within this time is released from the CPU

    // Extra "key=value;key=value" text for the upscaler (the network runtime's own keys, e.g. timing_log=1;sync=pipelined),
    // appended to BackendOptions() after the front end's keys. Normalized: items trimmed, empty items dropped.
    std::string runtimeOptions;

    // In-game overlay (overlay.h): "on" (the full panel for 10 s after WrongGPU starts upscaling, then a one-line badge),
    // "badge" (the badge only), "off". overlayKey toggles it: insert, home, end, delete, pageup, pagedown, pause,
    // scrolllock, f1 ... f12, or a virtual-key code (0x2d).
    std::string overlay = "on";
    std::string overlayKey = "insert";
    // Switches between WrongGPU's upscaler and AMD's FSR while playing, to compare picture and cost (same keys as
    // overlayKey, or "off").
    std::string compareKey = "home";
    // AMD's upscaler for the comparison: an AMD FidelityFX API DLL (AMD's FidelityFX SDK 2 upscaler DLL brings FSR 4 on
    // RDNA 4). Empty: amd_fidelityfx_upscaler_dx12.dll next to WrongGPU's DLLs if there is one, else the game's own
    // AMD DLL (FSR 3.1 through amd_fidelityfx_dx12, FSR 3.0 in games that only have the FSR 3.0 DLLs).
    std::string compareDll;
    // Corner: bottom-left (default; free of most games' health bars and minimaps), top-left, top-right, bottom-right.
    std::string overlayPosition = "bottom-left";

    // Set by the DLL, not read from the .ini: another upscaler mod (OptiScaler) loaded this DLL under a name of its own
    // as its FidelityFX upscaler. WrongGPU is then one of the upscaler versions the host lists in its own menu (listed
    // first, before AMD's). The host's choice decides who upscales. Actual dispatch status is tracked, but the hosted
    // overlay experiment is held after a real-game restart watchdog; overlay/compare key remain off in this mode.
    bool hosted = false;
    // Set by the DLL, not read from the .ini: hosted from OptiScaler's custom upscaler folder (custom\<name>\ffx\,
    // spec 65). The FidelityFX DLLs the host's process offers (the game's amd_fidelityfx_* files, redirected to and
    // hooked by OptiScaler) lead back into the host, which routes their upscale calls to its selected upscaler: this
    // DLL (Cyberpunk, 2026-10-30: an AMD fallback dispatch re-entered the custom slot 36 levels deep and froze the
    // game). So, in the custom slot: an AMD fallback, AMD's list entries and comparisons only through an AMD DLL placed
    // next to this one (amd_fidelityfx_dx12.wronggpu-original.dll), never through the host's FidelityFX DLLs; without
    // one, a WrongGPU create that fails returns a failure code and the host's own upscaler takes over; and a failed
    // create is remembered for the session (the same upscaler and options are not created again).
    bool customSlot = false;
    // [fsr] model_dir: a folder holding the converted model (model.wgcm; Model K in its K\ subfolder). Relative paths
    // are taken from the data folder (<game exe folder>\WrongGPU). Looked at first; then <game exe folder>\WrongGPU\model;
    // then the runtime's own default (%LOCALAPPDATA%\WrongGPU\model). A model_dir in runtime_options decides over all.
    std::string modelDir;

    // Logging.
    bool logDispatches = false;        // one line per ffxDispatch when true; off: the first calls of each context only
    uint32_t logMaxMegabytes = 512;    // above this size, dispatch lines are thinned to one in 600 (noted in the log)

    std::string BackendOptions() const;  // "view=...;filter=...;...[;<runtime_options>]" (WgCreateInfo::options)
    std::string Describe() const;        // one line for the log
};

// Parses an .ini text ("key = value" lines; '#' or ';' starts a comment at the start of a line or after a space or tab,
// so "runtime_options = a=1;b=2" keeps its ';'). Keys before any section and in the sections
// [wronggpu] and [fsr] are read; other sections belong to other WrongGPU components and are skipped. Unknown keys and
// bad values are reported through `warnings` (one per line) and otherwise ignored. A line whose key is a runtime option
// (runtime_keys.h, e.g. "network = fused") is appended to runtime_options as "network=fused", with a warning.
Config ParseConfig(const std::string& text, std::string* warnings);

// Windows virtual-key code of an overlay_key value ("insert", "f9", "0x2d", ...); 0 when not recognised.
int OverlayKeyCode(const std::string& name);

// "a=1; b=2;;" -> "a=1;b=2": items trimmed, empty items dropped.
std::string NormalizeOptions(const std::string& text);
// The options (items separated by ';' or ',') without their model=K items; *removed: whether there was one. For
// builds without Model K (model_k.h), which drop the item before the runtime reads it.
std::string WithoutModelK(const std::string& options, bool* removed);
// The same for any model letter (model=<letter>, either case): WithoutModelK is WithoutModel(options, 'K', removed).
std::string WithoutModel(const std::string& options, char model, bool* removed);
#if !WRONGGPU_MODEL_L
// A build without Model L (model_l.h, CMake WG_MODEL_L=OFF): drops model=L from config.runtimeOptions and from
// WG_RUNTIME_OPTIONS as IgnoreModelK does, so that Model M runs. Returns the one line for the log, or "".
std::string IgnoreModelL(Config& config);
#endif
#if !WRONGGPU_MODEL_K
// A build without Model K: drops model=K from config.runtimeOptions (wronggpu.ini) and from WG_RUNTIME_OPTIONS in the
// process environment, so that Model M runs (the runtime library refuses the option, and it reads the environment as
// it is when it is loaded: this runs before, when the front end starts). Returns the one line for the log, or "".
std::string IgnoreModelK(Config& config);
#endif
const char* RuntimeProviderName(const std::string&, const char*);

// The last value of `key` in "key=value;..." options (items trimmed, last one wins as in the runtime); *found: whether
// the key is there at all.
std::string OptionValue(const std::string& options, const char* key, bool* found = nullptr);

// Where the converted model is looked for (Config::modelDir, ModelChoice). A place is a folder: Model M is
// <dir>\model.wgcm, Model K <dir>\K\model.wgcm (the runtime's layout). runtimeDefault: the runtime's own default
// place (%LOCALAPPDATA%\WrongGPU\model): it is not probed and never passed on (the runtime looks there by itself).
struct ModelPlace {
    std::string dir;
    std::string source;  // for the log: "wronggpu.ini model_dir", "the game's data folder", ...
    bool runtimeDefault = false;
};
struct ModelChoice {
    int index = -1;        // the chosen place; -1: none of the probed places has the model
    std::string file;      // the model file of the chosen place
    std::string option;    // ";model_dir=<dir>" to append to the runtime options, or "" (runtime default / none)
    std::string searched;  // one log line: every place, in order, and what was found there
};
// The first place (in order) whose model file exists; a runtime-default place ends the search without a probe.
ModelChoice ChooseModelDir(const std::vector<ModelPlace>& places, bool modelK, const std::function<bool(const std::string&)>& exists);
// The same for a model by its letter: M looks for <place>\model.wgcm, K and L for <place>\K\model.wgcm and
// <place>\L\model.wgcm (the folders the launcher converts them into and the runtime reads under model_dir).
ModelChoice ChooseModelDir(const std::vector<ModelPlace>& places, char model, const std::function<bool(const std::string&)>& exists);

// Original process-specific synchronization default. Returns true when an option was added; explicit user
// choices and other games/backends are preserved. No image/model settings are changed.
bool ApplyProcessCompatibility(Config& config, const std::string& executable);

// The text written when no .ini exists (documents every key with its default): the registry's generated [wronggpu]
// section (clean/wgshared/ini/wronggpu.default.ini) followed by the front end's own [fsr] section.
const char* DefaultConfigText();
const char* DefaultFsrSectionText();  // the [fsr] part alone (appended to a wronggpu.ini that has no [fsr] section)

}  // namespace wgfsr
