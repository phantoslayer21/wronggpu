// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8a): the FG game-adapter core (fg_proxy_core.h). Written for this project.
#include "fg_proxy_core.h"

#include <cstring>

// Test builds only (tests/CMakeLists: test_fg_proxy_mutant_<k>): WG_FG_PROXY_MUTANT=k breaks one rule on purpose, and the
// test suite must then fail. Never defined in a product build (0 = the core as written).
#ifndef WG_FG_PROXY_MUTANT
#define WG_FG_PROXY_MUTANT 0
#endif

namespace wgfgp {

namespace {

constexpr uint32_t kAmdVendor = 0x1002;
constexpr size_t kMaxLogLines = 2000;

const char* ReasonName(uint32_t r) {
    switch (r) {
        case WG_FG_REASON_NONE: return "none";
        case WG_FG_REASON_STRUCT_SIZE: return "struct size";
        case WG_FG_REASON_ABI_VERSION: return "contract version";
        case WG_FG_REASON_BAD_PRESET: return "preset not offered";
        case WG_FG_REASON_BAD_GENERATED_COUNT: return "no generated count range";
        case WG_FG_REASON_BAD_OPTIONAL_DECL: return "optional input declaration";
        case WG_FG_REASON_TOKEN_NOT_MONOTONIC: return "interval already presented";
        case WG_FG_REASON_VIEWPORT_MISMATCH: return "viewport mismatch";
        case WG_FG_REASON_LUID_MISMATCH: return "adapter LUID mismatch";
        case WG_FG_REASON_QUEUE_EPOCH_MISMATCH: return "queue epoch changed";
        case WG_FG_REASON_SWAPCHAIN_EPOCH_MISMATCH: return "swap chain epoch changed";
        case WG_FG_REASON_MISSING_COLOR: return "no backbuffer";
        case WG_FG_REASON_MISSING_MOTION: return "no motion vectors";
        case WG_FG_REASON_MISSING_DEPTH: return "no depth";
        case WG_FG_REASON_MISSING_PROMISED_OPTIONAL: return "promised optional input missing";
        case WG_FG_REASON_UNDECLARED_OPTIONAL: return "undeclared optional input";
        case WG_FG_REASON_CONFLICTING_TAG: return "conflicting tags";
        case WG_FG_REASON_STALE_TAG: return "stale frame token";
        case WG_FG_REASON_INSTALL_OFF: return "install switch off (fg=off)";
        case WG_FG_REASON_UNSUPPORTED_ADAPTER: return "adapter not supported";
        case WG_FG_REASON_MODEL_NOT_VALIDATED: return "model not validated";
        case WG_FG_REASON_PRESENTER_NOT_QUALIFIED: return "presenter not qualified";
        case WG_FG_REASON_ROUTE_NOT_QUALIFIED: return "modes not qualified";
        case WG_FG_REASON_BINDING_NOT_QUALIFIED: return "title binding not qualified";
        case WG_FG_REASON_NOT_ATTACHED: return "no entry attached";
        case WG_FG_REASON_NULL_HOST: return "no host or next link";
        case WG_FG_REASON_DOUBLE_ATTACH: return "an entry is already attached";
        case WG_FG_REASON_VERSION_MISMATCH: return "interface version differs from the binding";
        case WG_FG_REASON_HOST_REFUSED: return "the host refused the registration";
        case WG_FG_REASON_PRESENT_OWNED: return "another presenter owns the chain";
        case WG_FG_REASON_NOT_PRESENT_OWNER: return "caller does not own the chain";
        case WG_FG_REASON_MISSING_CONSTANTS: return "no constants for the frame";
        case WG_FG_REASON_SECOND_VIEWPORT: return "second viewport";
        case WG_FG_REASON_UNKNOWN_FUNCTION: return "unknown function";
        case WG_FG_REASON_NO_EXECUTOR: return "no executor";
        case WG_FG_REASON_EXECUTOR_REFUSED: return "executor refused the frame";
        case WG_FG_REASON_NULL_ARGUMENT: return "missing argument";
        default: return "unknown reason";
    }
}

const char* RealOnlyName(uint32_t r) {
    switch (r) {
        case WG_FG_REAL_ONLY_NONE: return "none";
        case WG_FG_REAL_ONLY_PRIME: return "prime";
        case WG_FG_REAL_ONLY_RESET: return "reset";
        case WG_FG_REAL_ONLY_MENU: return "menu";
        case WG_FG_REAL_ONLY_RESIZE: return "resize";
        case WG_FG_REAL_ONLY_GAP: return "gap";
        case WG_FG_REAL_ONLY_EPOCH_CHANGE: return "epoch change";
        case WG_FG_REAL_ONLY_HISTORY_BUSY: return "history busy";
        case WG_FG_REAL_ONLY_DISABLED: return "generation off";
        default: return "unknown";
    }
}

const char* OwnerName(uint32_t o) {
    switch (o) {
        case WG_FG_OWNER_NONE: return "nobody";
        case WG_FG_OWNER_HOST_WRAPPER: return "host wrapper";
        case WG_FG_OWNER_NATIVE_ADAPTER: return "native adapter";
        case WG_FG_OWNER_SWAPCHAIN_WRAPPER: return "swap-chain wrapper";
        default: return "unknown owner";
    }
}

uint32_t Min3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t m = a < b ? a : b;
    return m < c ? m : c;
}

bool LuidZero(const WgFgLuid& l) { return l.low == 0 && l.high == 0; }
bool LuidEq(const WgFgLuid& a, const WgFgLuid& b) { return a.low == b.low && a.high == b.high; }

// A struct the other side handed over: big enough and of this contract version.
template <typename T>
uint32_t Shape(const T* p) {
    if (!p) return WG_FG_REASON_NULL_ARGUMENT;
    if (p->struct_size < sizeof(T)) return WG_FG_REASON_STRUCT_SIZE;
    if (p->abi_version != WG_FG_ADAPTER_ABI_VERSION) return WG_FG_REASON_ABI_VERSION;
    return WG_FG_REASON_NONE;
}

}  // namespace

const char* FgReasonName(uint32_t reason) { return ReasonName(reason); }
const char* FgRealOnlyName(uint32_t real_only) { return RealOnlyName(real_only); }

bool FgBindingVersionMatches(uint32_t major, uint32_t minor, uint32_t want_major, uint32_t want_minor,
                             uint32_t want_minor_max) {
    const uint32_t hi = want_minor_max > want_minor ? want_minor_max : want_minor;
    return major == want_major && minor >= want_minor && minor <= hi;
}

// ---- present owners -------------------------------------------------------------------------------------------------

uint32_t FgPresentOwners::claim(uint64_t swapchain_id, uint32_t owner) {
    if (swapchain_id == 0 || owner == WG_FG_OWNER_NONE || owner > WG_FG_OWNER_SWAPCHAIN_WRAPPER)
        return WG_FG_REASON_NULL_ARGUMENT;
    std::lock_guard<std::mutex> lock(mu_);
    auto it = owners_.find(swapchain_id);
    if (it == owners_.end()) {
        owners_.emplace(swapchain_id, owner);
        return WG_FG_REASON_NONE;
    }
    return it->second == owner ? WG_FG_REASON_NONE : WG_FG_REASON_PRESENT_OWNED;
}

bool FgPresentOwners::release(uint64_t swapchain_id, uint32_t owner) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = owners_.find(swapchain_id);
    if (it == owners_.end() || it->second != owner) return false;
    owners_.erase(it);
    return true;
}

uint32_t FgPresentOwners::owner_of(uint64_t swapchain_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = owners_.find(swapchain_id);
    return it == owners_.end() ? static_cast<uint32_t>(WG_FG_OWNER_NONE) : it->second;
}

FgPresentOwners& FgProcessPresentOwners() {
    static FgPresentOwners* owners = new FgPresentOwners();  // intentional immortal singleton for late swapchain release
    return *owners;
}

// ---- construction, entries ------------------------------------------------------------------------------------------

FgProxy::FgProxy(const FgProxyConfig& cfg, FgPresentOwners* owners, FgLogFn log, void* log_ctx)
    : cfg_(cfg), owners_(owners ? owners : &FgProcessPresentOwners()), log_(log), log_ctx_(log_ctx) {
    provider_.struct_size = sizeof(WgFgProviderV1);
    provider_.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    provider_.self = this;
    provider_.get_capability = &FgProxy::P_get_capability;
    provider_.set_options = &FgProxy::P_set_options;
    provider_.tag = &FgProxy::P_tag;
    provider_.constants = &FgProxy::P_constants;
    provider_.present = &FgProxy::P_present;
    provider_.get_state = &FgProxy::P_get_state;
    if (!FgSettingsValid(cfg_.settings)) {
        // never reached through FgSettingsFromKeyValues (it only produces valid settings); a hand-built invalid one
        // keeps the switch off
        cfg_.settings = FgSettings{};
        log_locked("fg: settings not valid; frame generation stays off");
    }
}

FgProxy::~FgProxy() { detach(); }

uint32_t FgProxy::entry_kind() const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    return entry_;
}

uint32_t FgProxy::attach_interposer(const WgFgChainV1* chain, uint32_t file_major, uint32_t file_minor) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    uint32_t r = WG_FG_REASON_NONE;
    if (entry_ != WG_FG_ENTRY_NONE) r = WG_FG_REASON_DOUBLE_ATTACH;
    else if (!chain) r = WG_FG_REASON_NULL_HOST;
    else if ((r = Shape(chain)) != WG_FG_REASON_NONE) {
    } else if (!chain->is_supported || !chain->get_function || !chain->set_options || !chain->get_state ||
               !chain->tag || !chain->constants || !chain->present)
        r = WG_FG_REASON_NULL_HOST;
    else if (cfg_.binding.qualified &&
             !FgBindingVersionMatches(file_major, file_minor, cfg_.binding.version_major, cfg_.binding.version_minor,
                                      cfg_.binding.version_minor_max))
        r = WG_FG_REASON_VERSION_MISMATCH;
    if (r != WG_FG_REASON_NONE) {
        log_locked(std::string("fg: interposer not attached: ") + ReasonName(r) + " (interface " +
                   std::to_string(file_major) + "." + std::to_string(file_minor) + ")");
        return r;
    }
    chain_ = *chain;
    entry_ = WG_FG_ENTRY_INTERPOSER;
    entry_max_ = chain->max_generated > WG_FG_MAX_GENERATED ? WG_FG_MAX_GENERATED : chain->max_generated;
    log_locked("fg: interposer attached (interface " + std::to_string(file_major) + "." + std::to_string(file_minor) +
               ", ABI max " + std::to_string(entry_max_) + ")");
    note_route_locked(true);
    return WG_FG_REASON_NONE;
}

uint32_t FgProxy::register_with_host(const WgFgHostV1* host) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    uint32_t r = WG_FG_REASON_NONE;
    if (entry_ != WG_FG_ENTRY_NONE) r = WG_FG_REASON_DOUBLE_ATTACH;
    else if (!host) r = WG_FG_REASON_NULL_HOST;
    else if ((r = Shape(host)) != WG_FG_REASON_NONE) {
    } else if (!host->register_provider || !host->unregister_provider)
        r = WG_FG_REASON_NULL_HOST;
    else if (!cfg_.settings.enabled)
        r = WG_FG_REASON_INSTALL_OFF;  // zero takeover: the host never sees this provider
    if (r == WG_FG_REASON_NONE && host->register_provider(host->self, &provider_) != 0) r = WG_FG_REASON_HOST_REFUSED;
    if (r != WG_FG_REASON_NONE) {
        log_locked(std::string("fg: host provider not registered: ") + ReasonName(r));
        return r;
    }
    host_ = *host;
    entry_ = WG_FG_ENTRY_HOST_PROVIDER;
    entry_max_ = host->max_generated > WG_FG_MAX_GENERATED ? WG_FG_MAX_GENERATED : host->max_generated;
    log_locked("fg: host provider registered (ABI max " + std::to_string(entry_max_) + ")");
    note_route_locked(true);
    return WG_FG_REASON_NONE;
}

void FgProxy::detach() {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (claimed_owner_ != WG_FG_OWNER_NONE) {
        owners_->release(claimed_chain_, claimed_owner_);
        claimed_owner_ = WG_FG_OWNER_NONE;
        claimed_chain_ = 0;
    }
    if (entry_ == WG_FG_ENTRY_HOST_PROVIDER && host_.unregister_provider) host_.unregister_provider(host_.self, &provider_);
    if (entry_ != WG_FG_ENTRY_NONE) log_locked("fg: entry detached");
    entry_ = WG_FG_ENTRY_NONE;
    chain_ = {};
    host_ = {};
    entry_max_ = 0;
}

void FgProxy::set_adapter(const WgFgAdapterIdentity& id) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    adapter_ = id;
    adapter_known_ = Shape(&id) == WG_FG_REASON_NONE;
    note_route_locked();
}

void FgProxy::set_device(void* device, void* queue) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    device_ = device;
    queue_ = queue;
}

void FgProxy::set_runtime_facts(bool model_validated, bool presenter_qualified) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    model_validated_ = model_validated;
    presenter_qualified_ = presenter_qualified;
    note_route_locked();
}

void FgProxy::set_executor(const WgFgExecutorV1* executor) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    has_executor_ = executor && Shape(executor) == WG_FG_REASON_NONE && executor->submit;
    executor_ = has_executor_ ? *executor : WgFgExecutorV1{};
}

bool FgProxy::update_live_settings(uint32_t preset, uint32_t multiplier) {
    // cfg_.binding is fixed after construction: read without the lock (this call never takes it)
    if (preset >= WG_FG_PRESET_COUNT || !(cfg_.binding.presets_mask & (1u << preset)) ||
        (multiplier != kFgAutoMultiplier && FgGeneratedCountOf(multiplier) == 0))
        return false;
    uint64_t cur = live_.load(std::memory_order_relaxed);
    for (;;) {
        const uint64_t seq = ((cur >> 32) + 1) & 0xffffffffu;
        const uint64_t next = ((seq ? seq : 1) << 32) | (static_cast<uint64_t>(preset) << 8) | multiplier;
        if (live_.compare_exchange_weak(cur, next, std::memory_order_release, std::memory_order_relaxed)) return true;
    }
}

bool FgProxy::use_game_count(uint32_t game_count) const {
    return cfg_.panel_multipliers &&
           (cfg_.game_has_multiplier || (game_count >= 1 && game_count <= WG_FG_MAX_GENERATED));
}

uint32_t FgProxy::count_for_options_locked(uint32_t game_count) const {
    if (game_count != WG_FG_COUNT_NOT_SET) return game_count;
    return use_game_count(game_count) ? 1u : manual_count_locked();
}

// A menu change takes effect on the game thread. Recompute the same count used by set_options.
void FgProxy::apply_live_locked() {
    const uint64_t v = live_.load(std::memory_order_acquire);
    const uint32_t seq = static_cast<uint32_t>(v >> 32);
    if (seq == 0 || seq == live_applied_) return;
    live_applied_ = seq;
    const uint32_t preset = static_cast<uint32_t>((v >> 8) & 0xffu), multiplier = static_cast<uint32_t>(v & 0xffu);
    if (preset == cfg_.settings.preset && multiplier == cfg_.settings.multiplier) return;
    cfg_.settings.preset = preset;
    cfg_.settings.multiplier = multiplier;
    std::string game;
    if (has_options_) {
        if (options_.preset != WG_FG_PRESET_NOT_SET) {
            game = std::string("; the game asks for preset ") + (FgPresetSpelling(options_.preset) ? FgPresetSpelling(options_.preset) : "?") +
                   ", which stays";
        } else {
            effective_preset_ = preset;
        }
        if (options_.enable) {
            const uint32_t game_count = options_.generated_count;
            const uint32_t cap = use_game_count(game_count) ? advertised_max_generated() : effective_max_locked();
            const uint32_t requested = count_for_options_locked(game_count);
            effective_count_ = requested > cap ? cap : requested;
            game += "; generating " + std::to_string(effective_count_) + " per game frame (the game asks for " +
                    std::to_string(requested) + ")";
        }
    }
    const std::string multiplier_text = multiplier == kFgAutoMultiplier ? "auto" : std::to_string(multiplier) + "x";
    log_locked(std::string("fg: live settings: preset ") + FgPresetSpelling(preset) + ", multiplier " +
               multiplier_text + " (player cap " + std::to_string(manual_count_locked()) + "), from this frame" + game);
}

void FgProxy::set_feature_function(const std::string& name, void* fn) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    fns_[name] = fn;
}

// ---- route ----------------------------------------------------------------------------------------------------------

uint32_t FgProxy::route_reason() const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    return route_reason_locked();
}

uint32_t FgProxy::route_reason_locked() const {
#if WG_FG_PROXY_MUTANT != 1
    if (!cfg_.settings.enabled) return WG_FG_REASON_INSTALL_OFF;
#endif
    if (!cfg_.binding.qualified) return WG_FG_REASON_BINDING_NOT_QUALIFIED;
    if (entry_ == WG_FG_ENTRY_NONE) return WG_FG_REASON_NOT_ATTACHED;
    if (!adapter_known_ || adapter_.vendor_id != kAmdVendor || adapter_.rdna4 != 1 || adapter_.probes_passed != 1 ||
        LuidZero(adapter_.luid))
        return WG_FG_REASON_UNSUPPORTED_ADAPTER;
    if (!model_validated_) return WG_FG_REASON_MODEL_NOT_VALIDATED;
    if (!presenter_qualified_) return WG_FG_REASON_PRESENTER_NOT_QUALIFIED;
#if WG_FG_PROXY_MUTANT != 7
    if (!cfg_.modes_qualified) return WG_FG_REASON_ROUTE_NOT_QUALIFIED;
#endif
    if (entry_max_ == 0 || cfg_.binding.ui_max_generated == 0 || (cfg_.binding.presets_mask & WG_FG_PRESETS_ALL) == 0)
        return WG_FG_REASON_BAD_GENERATED_COUNT;
    return WG_FG_REASON_NONE;
}

uint32_t FgProxy::advertised_max_generated() const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (route_reason_locked() != WG_FG_REASON_NONE) return 0;
    return Min3(WG_FG_MAX_GENERATED, entry_max_, cfg_.binding.ui_max_generated);
}

uint32_t FgProxy::options_max_generated() const {
    return cfg_.panel_multipliers ? WG_FG_MAX_GENERATED : advertised_max_generated();
}

uint32_t FgProxy::manual_count_locked() const {
    return std::max(1u, FgGeneratedCountOf(cfg_.settings.multiplier));
}

uint32_t FgProxy::effective_max_locked() const {
    if (route_reason_locked() != WG_FG_REASON_NONE) return 0;
    return Min3(Min3(WG_FG_MAX_GENERATED, entry_max_, cfg_.binding.ui_max_generated),
                manual_count_locked(), WG_FG_MAX_GENERATED);
}

WgFgCapability FgProxy::capability() const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    WgFgCapability c = {};
    c.struct_size = sizeof(c);
    c.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    c.reason = route_reason_locked();
    c.supported = c.reason == WG_FG_REASON_NONE ? 1u : 0u;
    c.max_generated = c.supported ? Min3(WG_FG_MAX_GENERATED, entry_max_, cfg_.binding.ui_max_generated) : 0u;
    c.presets_mask = c.supported ? (cfg_.binding.presets_mask & WG_FG_PRESETS_ALL) : 0u;
    if (adapter_known_) c.adapter = adapter_;  // as read from the game's device, never altered
    c.adapter.struct_size = sizeof(WgFgAdapterIdentity);
    c.adapter.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    return c;
}

void FgProxy::note_route_locked(bool defer_unprobed_adapter) {
    const uint32_t r = route_reason_locked();
    if (defer_unprobed_adapter && r == WG_FG_REASON_UNSUPPORTED_ADAPTER && !adapter_known_) return;
    if (route_noted_ && r == last_route_reason_) return;
    route_noted_ = true;
    last_route_reason_ = r;
    if (r == WG_FG_REASON_NONE)
        log_locked("fg: frame generation taken (max generated " +
                   std::to_string(Min3(WG_FG_MAX_GENERATED, entry_max_, cfg_.binding.ui_max_generated)) +
                   ", player cap " + std::to_string(manual_count_locked()) + ")");
    else
        log_locked(std::string("fg: frame generation not taken: ") + ReasonName(r) + "; every call forwards");
}

void FgProxy::log_locked(const std::string& line) const {
    if (log_lines_.size() < kMaxLogLines) log_lines_.push_back(line);
    if (log_) log_(log_ctx_, line.c_str());
}

uint32_t FgProxy::log_count(const std::string& needle) const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    uint32_t n = 0;
    for (const auto& l : log_lines_)
        if (l.find(needle) != std::string::npos) ++n;
    return n;
}

// ---- feature-scoped calls -------------------------------------------------------------------------------------------

int32_t FgProxy::is_supported(uint32_t feature, const WgFgLuid* luid, const void* native_adapter) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (is_fg_feature(feature)) note_route_locked();
    if (!owns_feature(feature)) {
        if (entry_ == WG_FG_ENTRY_INTERPOSER) return chain_.is_supported(chain_.self, feature, native_adapter);
        return entry_ == WG_FG_ENTRY_NONE ? WG_E_NOT_INITIALISED : WG_E_FEATURE_NOT_SUPPORTED;
    }
    if (luid && !LuidEq(*luid, adapter_.luid)) return WG_E_ADAPTER_NOT_SUPPORTED;
    return WG_OK;
}

int32_t FgProxy::get_function(uint32_t feature, const char* name, void** out) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (is_fg_feature(feature)) note_route_locked();
    if (!owns_feature(feature)) {
        if (entry_ == WG_FG_ENTRY_INTERPOSER) return chain_.get_function(chain_.self, feature, name, out);
        return entry_ == WG_FG_ENTRY_NONE ? WG_E_NOT_INITIALISED : WG_E_FEATURE_NOT_SUPPORTED;
    }
    if (!name || !out) return WG_E_INVALID_PARAMETER;
    *out = nullptr;
    for (const auto& bound : cfg_.binding.fn_names) {
        if (bound != name) continue;
        auto it = fns_.find(bound);
        if (it != fns_.end() && it->second) {
            *out = it->second;
            return WG_OK;
        }
    }
    // FG is ours: a function the binding does not know must not reach the genuine feature (two owners of one feature)
    log_locked(std::string("fg: function '") + name + "' refused: " + ReasonName(WG_FG_REASON_UNKNOWN_FUNCTION));
    return WG_E_FEATURE_MISSING;
}

int32_t FgProxy::set_options(uint32_t feature, uint32_t viewport_id, const WgFgOptions* decoded,
                             const void* native_options) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (is_fg_feature(feature)) note_route_locked();
    apply_live_locked();
    if (!owns_feature(feature)) {
        if (entry_ == WG_FG_ENTRY_INTERPOSER) return chain_.set_options(chain_.self, feature, viewport_id, native_options);
        return entry_ == WG_FG_ENTRY_NONE ? WG_E_NOT_INITIALISED : WG_E_FEATURE_NOT_SUPPORTED;
    }
    uint32_t r = Shape(decoded);
    uint32_t preset = cfg_.settings.preset, count = 0, requested = 0;
    if (r == WG_FG_REASON_NONE && has_vp_ && viewport_id != session_vp_) r = WG_FG_REASON_SECOND_VIEWPORT;
    if (r == WG_FG_REASON_NONE && decoded->preset != WG_FG_PRESET_NOT_SET) {
        preset = decoded->preset;
        if (preset >= WG_FG_PRESET_COUNT || !(cfg_.binding.presets_mask & (1u << preset))) r = WG_FG_REASON_BAD_PRESET;
    }
    for (uint32_t i = 0; r == WG_FG_REASON_NONE && i < WG_FG_OPTIONAL_COUNT; ++i)
        if (decoded->optional_decl[i] > WG_FG_DECL_ALWAYS) r = WG_FG_REASON_BAD_OPTIONAL_DECL;
    const uint32_t game_count = r == WG_FG_REASON_NONE ? decoded->generated_count : WG_FG_COUNT_NOT_SET;
    if (r == WG_FG_REASON_NONE && decoded->enable) {
        const uint32_t cap = use_game_count(game_count) ? advertised_max_generated() : effective_max_locked();
        requested = count_for_options_locked(game_count);
        count = requested > cap ? cap : requested;
        if (count == 0) r = WG_FG_REASON_BAD_GENERATED_COUNT;
    }
    if (r != WG_FG_REASON_NONE) {
        if (r != last_options_reason_) log_locked(std::string("fg: options refused: ") + ReasonName(r));
        last_options_reason_ = r;
        if (r == WG_FG_REASON_NULL_ARGUMENT) return WG_E_MISSING_INPUT_PARAMETER;
        return r == WG_FG_REASON_SECOND_VIEWPORT ? WG_E_INVALID_STATE : WG_E_INVALID_PARAMETER;
    }
    if (last_options_reason_ != WG_FG_REASON_NONE) log_locked("fg: options accepted again");
    last_options_reason_ = WG_FG_REASON_NONE;
    if (count != requested && requested != last_clamped_)
        log_locked("fg: requested generated count " + std::to_string(requested) + " clamped to " + std::to_string(count));
    last_clamped_ = count != requested ? requested : 0;
    if (cfg_.panel_multipliers && (decoded->enable || cfg_.game_has_multiplier))
        game_requested_count_.store(game_count >= 1 && game_count <= WG_FG_MAX_GENERATED ? game_count : 0);
    options_ = *decoded;
    has_options_ = true;
    session_vp_ = viewport_id;
    has_vp_ = true;
    effective_preset_ = preset;
    effective_count_ = count;
    return WG_OK;
}

int32_t FgProxy::get_state(uint32_t feature, uint32_t viewport_id, WgFgState* decoded, void* native_state) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!owns_feature(feature)) {
        if (entry_ == WG_FG_ENTRY_INTERPOSER) return chain_.get_state(chain_.self, feature, viewport_id, native_state);
        return entry_ == WG_FG_ENTRY_NONE ? WG_E_NOT_INITIALISED : WG_E_FEATURE_NOT_SUPPORTED;
    }
    if (!decoded) return WG_E_MISSING_INPUT_PARAMETER;
    if (decoded->struct_size < sizeof(WgFgState)) return WG_E_INVALID_PARAMETER;
    *decoded = state();
    return WG_OK;
}

WgFgState FgProxy::state() const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    WgFgState s = {};
    s.struct_size = sizeof(s);
    s.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    s.taken = route_reason_locked() == WG_FG_REASON_NONE ? 1u : 0u;
    s.enabled = has_options_ && options_.enable && menu_enabled_.load() ? 1u : 0u;
    s.effective_preset = effective_preset_;
    s.effective_count = s.enabled ? effective_count_ : 0u;
    s.max_generated = s.taken ? Min3(WG_FG_MAX_GENERATED, entry_max_, cfg_.binding.ui_max_generated) : 0u;
    s.last_reason = s.taken ? last_frame_reason_ : route_reason_locked();
    s.base_presents = base_presents_;
    s.generate_requests = generate_requests_;
    s.real_only_presents = real_only_presents_;
    return s;
}

// ---- shared calls: tags and constants -------------------------------------------------------------------------------

int32_t FgProxy::role_of(uint32_t native_type) const {
    if (native_type == WG_FG_NATIVE_TYPE_UNBOUND) return -1;
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r)
        if (cfg_.binding.native_type[r] == native_type) return static_cast<int32_t>(r);
    return -1;
}

FgProxy::Frame& FgProxy::frame_locked(uint64_t token) {
    auto it = frames_.find(token);
    if (it != frames_.end()) return it->second;
    while (frames_.size() >= kFgMaxFramesKept) frames_.erase(frames_.begin());  // the oldest token goes
    return frames_[token];
}

void FgProxy::prune_locked(uint64_t presented_token) {
    while (!frames_.empty() && frames_.begin()->first <= presented_token) frames_.erase(frames_.begin());
}

size_t FgProxy::frames_kept() const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    return frames_.size();
}

int32_t FgProxy::tag(const WgFgTag* decoded, uint32_t count, const void* native_tags) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (fg_taken() && decoded) {
        for (uint32_t i = 0; i < count; ++i) {
            const WgFgTag& t = decoded[i];
            if (t.struct_size < sizeof(WgFgTag)) continue;
            const int32_t role = role_of(t.native_type);
            if (role < 0 || !t.texture.resource) continue;  // not an FG input: the other features' business
            if (has_vp_ && t.texture.viewport_id != session_vp_) continue;
            uint64_t& seen = latest_seen_[static_cast<uint32_t>(role)];
            if (t.texture.frame_token > seen) seen = t.texture.frame_token;
            if (any_present_ && t.texture.frame_token <= last_token_) continue;  // an interval already presented
            Frame& f = frame_locked(t.texture.frame_token);
            const uint32_t bit = 1u << static_cast<uint32_t>(role);
            if ((f.have & bit) && f.tex[role].resource != t.texture.resource) f.conflict |= bit;
            f.tex[role] = t.texture;
            f.have |= bit;
        }
    }
    if (entry_ == WG_FG_ENTRY_INTERPOSER) return chain_.tag(chain_.self, native_tags, count);
    return WG_OK;
}

int32_t FgProxy::constants(const WgFgConstants* decoded, const void* native_constants) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (fg_taken() && Shape(decoded) == WG_FG_REASON_NONE && !(has_vp_ && decoded->viewport_id != session_vp_) &&
        !(any_present_ && decoded->frame_token <= last_token_)) {
        Frame& f = frame_locked(decoded->frame_token);
        f.constants = *decoded;
        f.has_constants = true;
    }
    if (entry_ == WG_FG_ENTRY_INTERPOSER) return chain_.constants(chain_.self, native_constants);
    return WG_OK;
}

// ---- present --------------------------------------------------------------------------------------------------------

uint32_t FgProxy::claim_present(uint64_t swapchain_id, uint32_t owner) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    const uint32_t route = route_reason_locked();
    if (route != WG_FG_REASON_NONE) return route;  // zero takeover: the chain stays with whoever presents it now
    const uint32_t r = owners_->claim(swapchain_id, owner);
    if (r != WG_FG_REASON_NONE) {
        log_locked(std::string("fg: present owner refused for ") + OwnerName(owner) + ": " + ReasonName(r) + " (" +
                   OwnerName(owners_->owner_of(swapchain_id)) + ")");
        return r;
    }
    if (claimed_owner_ != WG_FG_OWNER_NONE && claimed_chain_ != swapchain_id)
        owners_->release(claimed_chain_, claimed_owner_);
    claimed_owner_ = owner;
    claimed_chain_ = swapchain_id;
    return WG_FG_REASON_NONE;
}

bool FgProxy::release_present(uint64_t swapchain_id, uint32_t owner) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    const bool released = owners_->release(swapchain_id, owner);
    if (released && claimed_chain_ == swapchain_id && claimed_owner_ == owner) {
        claimed_owner_ = WG_FG_OWNER_NONE;
        claimed_chain_ = 0;
    }
    return released;
}

void FgProxy::finish_result_locked(WgFgPresentResult* out, uint32_t status, uint32_t reason, uint32_t real_only) const {
    if (!out) return;
    WgFgPresentResult r = {};
    r.struct_size = sizeof(r);
    r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    r.status = status;
    r.reason = reason;
    r.real_only = real_only;
    r.requested_preset = has_options_ ? options_.preset : WG_FG_PRESET_NOT_SET;
    r.effective_preset = effective_preset_;
    r.requested_count = has_options_ ? options_.generated_count : 0u;
    r.effective_count = status == WG_FG_PRESENT_GENERATE ? effective_count_ : 0u;
    r.base_presents = base_presents_;
    r.generate_requests = generate_requests_;
    r.real_only_presents = real_only_presents_;
    r.forwarded_presents = forwarded_presents_;
    *out = r;
}

void FgProxy::note_frame_locked(uint32_t reason, uint32_t real_only) {
    if (frame_noted_ && reason == last_frame_reason_ && real_only == last_frame_real_only_) return;
    frame_noted_ = true;
    last_frame_reason_ = reason;
    last_frame_real_only_ = real_only;
    if (reason == WG_FG_REASON_NONE && real_only == WG_FG_REAL_ONLY_NONE)
        log_locked("fg: frames: generating");
    else
        log_locked(std::string("fg: frames: real-only (") + (reason ? ReasonName(reason) : RealOnlyName(real_only)) +
                   ")");
}

uint32_t FgProxy::assemble_locked(const WgFgPresentDesc& d, WgFgPacket* p, uint32_t* real_only) {
    *real_only = WG_FG_REAL_ONLY_NONE;
    auto it = frames_.find(d.frame_token);
    Frame empty;
    const Frame& f = it == frames_.end() ? empty : it->second;
    // a required input of this frame: present, unique, of this viewport; otherwise why not
    auto required = [&f, &d, this](uint32_t role, uint32_t missing, WgFgTexture* out) -> uint32_t {
        const uint32_t bit = 1u << role;
        if (f.conflict & bit) return WG_FG_REASON_CONFLICTING_TAG;
        if (!(f.have & bit)) {
            auto seen = latest_seen_.find(role);
#if WG_FG_PROXY_MUTANT != 8
            return seen != latest_seen_.end() && seen->second < d.frame_token ? WG_FG_REASON_STALE_TAG : missing;
#else
            (void)seen;
            return missing;
#endif
        }
        if (f.tex[role].viewport_id != d.viewport_id) return WG_FG_REASON_VIEWPORT_MISMATCH;
        *out = f.tex[role];
        return WG_FG_REASON_NONE;
    };
    uint32_t r = WG_FG_REASON_NONE;
    if (d.backbuffer.resource) {
        p->color = d.backbuffer;
        p->color.frame_token = d.frame_token;
        p->color.viewport_id = d.viewport_id;
    } else if ((r = required(WG_FG_ROLE_BACKBUFFER, WG_FG_REASON_MISSING_COLOR, &p->color)) != WG_FG_REASON_NONE) {
        return r;
    }
    if ((r = required(WG_FG_ROLE_MOTION, WG_FG_REASON_MISSING_MOTION, &p->motion)) != WG_FG_REASON_NONE) return r;
    if ((r = required(WG_FG_ROLE_DEPTH, WG_FG_REASON_MISSING_DEPTH, &p->depth)) != WG_FG_REASON_NONE) return r;
    p->optional_present = 0;
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) {
        const uint32_t role = WG_FG_OPTIONAL_FIRST_ROLE + i, bit = 1u << role;
        const uint32_t decl = options_.optional_decl[i];
        if (f.conflict & bit) return WG_FG_REASON_CONFLICTING_TAG;
        const bool have = (f.have & bit) != 0;
        if (decl == WG_FG_DECL_ALWAYS && !have) return WG_FG_REASON_MISSING_PROMISED_OPTIONAL;
        if (decl == WG_FG_DECL_NEVER && have) return WG_FG_REASON_UNDECLARED_OPTIONAL;
        if (have) {
            if (f.tex[role].viewport_id != d.viewport_id) return WG_FG_REASON_VIEWPORT_MISMATCH;
            p->optional[i] = f.tex[role];
            p->optional_present |= 1u << i;
        }
    }
    if (!f.has_constants) return WG_FG_REASON_MISSING_CONSTANTS;
    const WgFgConstants& c = f.constants;
    p->frame_token = d.frame_token;
    p->viewport_id = d.viewport_id;
    p->adapter_luid = adapter_.luid;
    p->queue_epoch = d.queue_epoch;
    p->swapchain_epoch = d.swapchain_epoch;
    p->render_rect = c.render_rect;
    p->output_rect = c.output_rect;
    // a title that leaves the constants' rects at zero: the output is the colour input's rect, the render area the
    // depth input's
    if (p->output_rect.width == 0 || p->output_rect.height == 0) p->output_rect = p->color.rect;
    if (p->render_rect.width == 0 || p->render_rect.height == 0) p->render_rect = p->depth.rect;
    p->jitter_x = c.jitter_x;
    p->jitter_y = c.jitter_y;
    if (c.flags & WG_FG_FLAG_MOTION_SCALE_UV) {
        const float mw = static_cast<float>(p->motion.rect.width ? p->motion.rect.width : p->motion.width);
        const float mh = static_cast<float>(p->motion.rect.height ? p->motion.rect.height : p->motion.height);
        p->motion_scale_x = c.motion_scale_x * (mw > 0.0f ? mw : 1.0f);
        p->motion_scale_y = c.motion_scale_y * (mh > 0.0f ? mh : 1.0f);
    } else {
        p->motion_scale_x = c.motion_scale_x;
        p->motion_scale_y = c.motion_scale_y;
    }
    std::memcpy(p->camera_current, c.camera_current, sizeof(p->camera_current));
    std::memcpy(p->camera_previous, c.camera_previous, sizeof(p->camera_previous));
    p->flags = c.flags & WG_FG_FLAGS_KNOWN;
    // motion smaller than the output (a known, non-empty rect): render-resolution motion, whether or not the title said so
    if (p->motion.rect.width != 0 && p->motion.rect.height != 0 &&
        (p->motion.rect.width < p->output_rect.width || p->motion.rect.height < p->output_rect.height))
        p->flags |= WG_FG_FLAG_MOTION_RENDER_RES;
    if (c.color_space != WG_FG_CS_SDR || options_.color_space != WG_FG_CS_SDR || swapchain_cs_ != WG_FG_CS_SDR)
        p->flags |= WG_FG_FLAG_HDR;
    p->preset = effective_preset_;
    p->generated_count = effective_count_;
    p->generated_index = 0;
    if (p->flags & WG_FG_FLAG_MENU) *real_only = WG_FG_REAL_ONLY_MENU;
    return WG_FG_REASON_NONE;
}

int32_t FgProxy::present(const WgFgPresentDesc* desc, WgFgPresentResult* out) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    apply_live_locked();
    const uint32_t shape = Shape(desc);
    if (!fg_taken()) {
        // zero takeover: the present goes on as if WrongGPU were not here
        ++forwarded_presents_;
        finish_result_locked(out, WG_FG_PRESENT_FORWARDED, WG_FG_REASON_NONE, WG_FG_REAL_ONLY_NONE);
        if (entry_ == WG_FG_ENTRY_INTERPOSER && desc) return chain_.present(chain_.self, desc);
        return WG_OK;
    }
    auto refuse = [this, out](uint32_t reason) -> int32_t {
        note_frame_locked(reason, WG_FG_REAL_ONLY_NONE);
        finish_result_locked(out, WG_FG_PRESENT_REFUSED, reason, WG_FG_REAL_ONLY_NONE);
        return reason == WG_FG_REASON_NULL_ARGUMENT ? WG_E_MISSING_INPUT_PARAMETER
               : (reason == WG_FG_REASON_STRUCT_SIZE || reason == WG_FG_REASON_ABI_VERSION) ? WG_E_INVALID_PARAMETER
                                                                                             : WG_E_INVALID_STATE;
    };
    if (shape != WG_FG_REASON_NONE) return refuse(shape);
    const uint32_t holder = owners_->owner_of(desc->swapchain_id);
#if WG_FG_PROXY_MUTANT != 3
    if (desc->owner == WG_FG_OWNER_NONE || holder == WG_FG_OWNER_NONE) return refuse(WG_FG_REASON_NOT_PRESENT_OWNER);
    if (holder != desc->owner) return refuse(WG_FG_REASON_PRESENT_OWNED);
#else
    (void)holder;
#endif
    const bool new_chain = latched_ && desc->swapchain_id != swapchain_id_;
#if WG_FG_PROXY_MUTANT != 4
    if (any_present_ && !new_chain && desc->frame_token <= last_token_) return refuse(WG_FG_REASON_TOKEN_NOT_MONOTONIC);
#endif
    if (new_chain) frames_.clear();

    // the interval is consumed here, once
    any_present_ = true;
    last_token_ = desc->frame_token;
    ++base_presents_;
    uint32_t reason = WG_FG_REASON_NONE, real_only = WG_FG_REAL_ONLY_NONE;
    WgFgPacket packet = {};
    if (!LuidEq(desc->luid, adapter_.luid) && WG_FG_PROXY_MUTANT != 5) {
        reason = WG_FG_REASON_LUID_MISMATCH;
    } else if (has_vp_ && desc->viewport_id != session_vp_) {
        reason = WG_FG_REASON_VIEWPORT_MISMATCH;
    } else if (latched_ && (desc->swapchain_id != swapchain_id_ || desc->swapchain_epoch != swapchain_epoch_)) {
        reason = WG_FG_REASON_SWAPCHAIN_EPOCH_MISMATCH;
        real_only = WG_FG_REAL_ONLY_EPOCH_CHANGE;
    } else if (latched_ && desc->queue_epoch != queue_epoch_) {
        reason = WG_FG_REASON_QUEUE_EPOCH_MISMATCH;
        real_only = WG_FG_REAL_ONLY_EPOCH_CHANGE;
    }
    if (reason == WG_FG_REASON_NONE || real_only == WG_FG_REAL_ONLY_EPOCH_CHANGE) {
        // (re)latch: the next interval of the new epoch can generate again
        latched_ = true;
        swapchain_id_ = desc->swapchain_id;
        window_id_ = desc->window_id;
        queue_epoch_ = desc->queue_epoch;
        swapchain_epoch_ = desc->swapchain_epoch;
    }
    if (reason == WG_FG_REASON_NONE &&
        (!menu_enabled_.load() || !has_options_ || !options_.enable || effective_count_ == 0))
        real_only = WG_FG_REAL_ONLY_DISABLED;
    if (reason == WG_FG_REASON_NONE && real_only == WG_FG_REAL_ONLY_NONE)
        reason = assemble_locked(*desc, &packet, &real_only);
    if (reason == WG_FG_REASON_NONE && real_only == WG_FG_REAL_ONLY_NONE) {
        // A/B/C change the blend graph or UI inputs. Prime those inputs at the next interval.
        if (cfg_.panel_multipliers && submitted_preset_ != WG_FG_PRESET_NOT_SET && submitted_preset_ != packet.preset)
            packet.flags |= WG_FG_FLAG_RESET;
        WgFgSessionDesc session = {};
        if (!has_executor_) reason = WG_FG_REASON_NO_EXECUTOR;
        else if (!session_desc(&session) || executor_.submit(executor_.self, &packet, &session) != 0)
            reason = WG_FG_REASON_EXECUTOR_REFUSED;
        else
            submitted_preset_ = packet.preset;
    }
    prune_locked(desc->frame_token);
    note_frame_locked(reason, real_only);
    if (reason == WG_FG_REASON_NONE && real_only == WG_FG_REAL_ONLY_NONE) {
        ++generate_requests_;
        finish_result_locked(out, WG_FG_PRESENT_GENERATE, reason, real_only);
    } else {
        ++real_only_presents_;
        finish_result_locked(out, WG_FG_PRESENT_REAL_ONLY, reason, real_only);
    }
    return WG_OK;
}

bool FgProxy::session_viewport(uint32_t* out) const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!out || !has_vp_) return false;
    *out = session_vp_;
    return true;
}

bool FgProxy::session_desc(WgFgSessionDesc* out) const {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!out || !has_options_ || !latched_) return false;
    WgFgSessionDesc s = {};
    s.device = device_;
    s.queue = queue_;
    s.adapter_luid = adapter_.luid;
    s.swapchain_id = swapchain_id_;
    s.window_id = window_id_;
    s.queue_epoch = queue_epoch_;
    s.swapchain_epoch = swapchain_epoch_;
    s.viewport_id = session_vp_;
    s.output_width = options_.output_width;
    s.output_height = options_.output_height;
    s.output_format = options_.output_format;
    s.render_min_width = options_.render_min_width;
    s.render_min_height = options_.render_min_height;
    s.render_max_width = options_.render_max_width;
    s.render_max_height = options_.render_max_height;
    s.dynamic_resolution = options_.dynamic_resolution;
    s.preset = effective_preset_;
    s.max_generated = cfg_.panel_multipliers ? advertised_max_generated() : effective_max_locked();
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) s.optional_decl[i] = options_.optional_decl[i];
    s.hdr = (options_.color_space != WG_FG_CS_SDR || swapchain_cs_ != WG_FG_CS_SDR) ? 1u : 0u;
    s.presenter_mode = static_cast<uint32_t>(cfg_.settings.presenter);
    *out = s;
    return true;
}

void FgProxy::set_color_space(uint32_t color_space) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    swapchain_cs_ = color_space;
}

void FgProxy::free_viewport(uint32_t feature, uint32_t viewport_id) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!owns_feature(feature)) return;
    if (has_vp_ && viewport_id == session_vp_) {
        has_vp_ = false;
        has_options_ = false;
        frames_.clear();
    }
}

// ---- host provider trampolines: no exception leaves them ------------------------------------------------------------

int32_t FgProxy::P_get_capability(void* self, WgFgCapability* out) {
    if (!self || !out) return WG_E_MISSING_INPUT_PARAMETER;
    if (out->struct_size < sizeof(WgFgCapability)) return WG_E_INVALID_PARAMETER;
    try {
        *out = static_cast<FgProxy*>(self)->capability();
        return WG_OK;
    } catch (...) {
        return WG_E_EXCEPTION_HANDLER;
    }
}

int32_t FgProxy::P_set_options(void* self, uint32_t viewport_id, const WgFgOptions* options) {
    if (!self) return WG_E_MISSING_INPUT_PARAMETER;
    try {
        FgProxy* p = static_cast<FgProxy*>(self);
        return p->set_options(p->cfg_.ids.fg, viewport_id, options, nullptr);
    } catch (...) {
        return WG_E_EXCEPTION_HANDLER;
    }
}

int32_t FgProxy::P_tag(void* self, const WgFgTag* tags, uint32_t count) {
    if (!self) return WG_E_MISSING_INPUT_PARAMETER;
    try {
        return static_cast<FgProxy*>(self)->tag(tags, count, nullptr);
    } catch (...) {
        return WG_E_EXCEPTION_HANDLER;
    }
}

int32_t FgProxy::P_constants(void* self, const WgFgConstants* constants) {
    if (!self) return WG_E_MISSING_INPUT_PARAMETER;
    try {
        return static_cast<FgProxy*>(self)->constants(constants, nullptr);
    } catch (...) {
        return WG_E_EXCEPTION_HANDLER;
    }
}

int32_t FgProxy::P_present(void* self, const WgFgPresentDesc* present, WgFgPresentResult* out) {
    if (!self) return WG_E_MISSING_INPUT_PARAMETER;
    try {
        return static_cast<FgProxy*>(self)->present(present, out);
    } catch (...) {
        return WG_E_EXCEPTION_HANDLER;
    }
}

int32_t FgProxy::P_get_state(void* self, WgFgState* out) {
    if (!self) return WG_E_MISSING_INPUT_PARAMETER;
    try {
        FgProxy* p = static_cast<FgProxy*>(self);
        if (!p->fg_taken()) return WG_E_FEATURE_NOT_SUPPORTED;
        return p->get_state(p->cfg_.ids.fg, 0, out, nullptr);
    } catch (...) {
        return WG_E_EXCEPTION_HANDLER;
    }
}

}  // namespace wgfgp
