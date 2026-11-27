// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8a): the FG game-adapter core.
//
// The same split as the RR plugin (clean/rr_f/proxy: proxy_core = logic, hook_dll / proxy_dll = transport). This core
// is platform-free: a transport (a per-title interposer built on the RR plugin's hook transport, or a host plugin
// entry) decodes the game's calls into fg_adapter_abi.h structs, hands over the native arguments unchanged, and
// supplies the next link (WgFgChainV1) or the host (WgFgHostV1). Tests supply fakes.
//
// Rules the core enforces:
//  - INSTALL-OFF GATE. FG is off by default (fg=off, fg_settings.h). Taking frame generation needs every route fact,
//    checked in this order, first failure reported: install switch on, a qualified title binding, an attached entry,
//    a supported adapter (AMD, RDNA 4, probes passed, a LUID), a validated model, a qualified presenter, and qualified
//    modes (kFgModesQualified, open from 0.7.0; ../fg_limits.h). While any fact fails, every call - frame generation's too -
//    goes to the next link with the game's own arguments: zero takeover.
//  - FEATURE-SCOPED ROUTING. Only the FG feature id is ever answered here. Super resolution, ray reconstruction, the
//    latency features and every unknown id go to the next link unchanged. Resource tags and constants are shared by all
//    features: they are always forwarded, and observed only while FG is taken.
//  - TRUTHFUL CAPABILITY. The advertised adapter is the one the game's device runs on. max_generated is 0 unless every
//    route fact holds; then min(5, the entry ABI's maximum, the game's UI range).
//  - TAG-TO-PACKET ASSEMBLY with exact frame token, viewport, LUID, queue and swap-chain epoch matching. A present whose
//    inputs are missing, stale, conflicting or mismatched shows the game image real-only for that interval, with the
//    reason; nothing falls back to another route.
//  - ONE PRESENT OWNER per swap chain (FgPresentOwners): a host FG wrapper, this adapter and a swap-chain wrapper can
//    never each consume the same interval, and one interval (frame token) is consumed once.
//  - DIAGNOSTICS: one log line per transition of the route decision and of the per-frame outcome (not per frame).
// No exceptions cross the C entry points; no lambda captures a copy that can throw (MSVC /Zc:lambda).
// Written for this project.
#pragma once
#include <stdint.h>

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../../rr_f/proxy/include/rr_proxy_abi.h"
#include "../fg_limits.h"
#include "fg_adapter_abi.h"
#include "fg_settings.h"

namespace wgfgp {

// The release gate (the runtime contract's wgfg::kFgModesQualified): open from 0.7.0 (../fg_limits.h). Tests override
// FgProxyConfig::modes_qualified to check the closed gate.
inline constexpr bool kFgModesQualified = WG_FG_MODES_QUALIFIED != 0;
inline constexpr uint32_t kFgMaxFramesKept = 8;  // per-frame tag / constants sets kept (frames in flight)

// The game interface's feature ids (clean/rr_f/proxy/include/rr_proxy_abi.h); data, so a binding can name others.
struct FgFeatureIds {
    uint32_t fg = WG_FEAT_FRAME_GENERATION;
    uint32_t sr = WG_FEAT_SUPER_RESOLUTION;
    uint32_t rr = WG_FEAT_RAY_RECONSTRUCTION;
    uint32_t latency = WG_FEAT_LATENCY;
    uint32_t pc_latency = WG_FEAT_PC_LATENCY;
};

// One title binding: what call-shape evidence established for a game build. Data (win32/fg_title_bindings.h fills it
// from bindings/<title>.json).
struct FgBinding {
    bool qualified = false;            // false: the core forwards everything
    uint32_t version_major = 0;        // the interface file version the binding was qualified on (interposer entry)
    uint32_t version_minor = 0;
    // a generic binding's range of minor versions [version_minor, version_minor_max] of the same major; 0 (or not above
    // version_minor): exactly version_minor
    uint32_t version_minor_max = 0;
    uint32_t native_type[WG_FG_ROLE_COUNT] = {WG_FG_NATIVE_TYPE_UNBOUND, WG_FG_NATIVE_TYPE_UNBOUND,
                                              WG_FG_NATIVE_TYPE_UNBOUND, WG_FG_NATIVE_TYPE_UNBOUND,
                                              WG_FG_NATIVE_TYPE_UNBOUND, WG_FG_NATIVE_TYPE_UNBOUND,
                                              WG_FG_NATIVE_TYPE_UNBOUND};
    std::vector<std::string> fn_names; // the FG feature-function names <get-function> hands out
    uint32_t ui_max_generated = 1;     // the highest N the game's own UI offers
    uint32_t presets_mask = WG_FG_PRESETS_ALL;
};

struct FgProxyConfig {
    FgFeatureIds ids;
    FgBinding binding;
    FgSettings settings;               // the player's per-game settings (install switch, preset, multiplier, presenter)
    bool modes_qualified = kFgModesQualified;
};

using FgLogFn = void (*)(void* ctx, const char* line);

// A route / refusal reason (WgFgReason) in words, as the core's log lines name it.
const char* FgReasonName(uint32_t reason);
// A real-only reason (WgFgRealOnly) in words.
const char* FgRealOnlyName(uint32_t real_only);
// True when the interface version major.minor is inside the binding's version gate (exact, or a generic binding's range).
bool FgBindingVersionMatches(uint32_t major, uint32_t minor, uint32_t want_major, uint32_t want_minor,
                             uint32_t want_minor_max);

// The single present owner per swap chain. One registry per process (FgProcessPresentOwners) is shared by every
// adapter; tests make their own.
class FgPresentOwners {
public:
    // WG_FG_REASON_NONE: `owner` holds the chain (again, if it already did). PRESENT_OWNED: another owner holds it.
    // NULL_ARGUMENT: chain 0 or owner NONE.
    uint32_t claim(uint64_t swapchain_id, uint32_t owner);
    // true: `owner` held it and no longer does.
    bool release(uint64_t swapchain_id, uint32_t owner);
    uint32_t owner_of(uint64_t swapchain_id) const;  // WG_FG_OWNER_NONE when free

private:
    mutable std::mutex mu_;
    std::map<uint64_t, uint32_t> owners_;
};
FgPresentOwners& FgProcessPresentOwners();

class FgProxy {
public:
    FgProxy(const FgProxyConfig& cfg, FgPresentOwners* owners, FgLogFn log, void* log_ctx);
    ~FgProxy();
    FgProxy(const FgProxy&) = delete;
    FgProxy& operator=(const FgProxy&) = delete;

    // ---- entries (one per instance) ----
    // Interposer: the next link and the interface file version the transport read from the loaded module. Refused
    // (and nothing attached): null chain / missing calls (NULL_HOST), a smaller struct (STRUCT_SIZE), another contract
    // version (ABI_VERSION), an entry already attached (DOUBLE_ATTACH), a qualified binding of another version
    // (VERSION_MISMATCH). With the install switch off the chain is still attached: everything forwards.
    uint32_t attach_interposer(const WgFgChainV1* chain, uint32_t file_major, uint32_t file_minor);
    // Host provider: registers this adapter's provider table with the host. Refused as above, plus INSTALL_OFF (the
    // switch is off: nothing is registered, the host keeps its own path) and HOST_REFUSED (the host's register call
    // failed; its own path stays).
    uint32_t register_with_host(const WgFgHostV1* host);
    void detach();  // unregisters from the host; forgets the chain; releases a claimed chain
    uint32_t entry_kind() const;  // WgFgEntry

    // ---- route facts the transport learns at run time ----
    void set_adapter(const WgFgAdapterIdentity& id);
    void set_device(void* device, void* queue);
    void set_runtime_facts(bool model_validated, bool presenter_qualified);
    void set_executor(const WgFgExecutorV1* executor);  // null: none (a taken present then refuses: NO_EXECUTOR)
    // The player's preset and multiplier changed while the game runs (the host menu). Lock-free and callable from any
    // thread (a menu drawn inside a present): the values are taken on the game's thread at its next options call or
    // present, under the core's lock, and the session keeps running (N and the preset may change per game frame). The
    // game's own preset request still wins over the player's preset, as at start. false (nothing changes): a preset
    // outside the binding's presets or a multiplier outside 2..6.
    bool update_live_settings(uint32_t preset, uint32_t multiplier);

    // The route decision: WG_FG_REASON_NONE = FG is taken; otherwise the first failing fact.
    uint32_t route_reason() const;
    bool fg_taken() const { return route_reason() == WG_FG_REASON_NONE; }
#if defined(WG_FG_PROXY_MUTANT) && WG_FG_PROXY_MUTANT == 2  // test-only mutant: every feature taken for FG
    bool is_fg_feature(uint32_t) const { return true; }
#else
    bool is_fg_feature(uint32_t feature) const { return feature == cfg_.ids.fg; }
#endif
    bool owns_feature(uint32_t feature) const { return is_fg_feature(feature) && fg_taken(); }
    WgFgCapability capability() const;
    uint32_t advertised_max_generated() const;

    // ---- interposer calls (game -> transport -> here). Feature-scoped calls go to the next link unless the feature
    // is FG and FG is taken; the result is then the core's (0, or a game-interface result code). ----
    int32_t is_supported(uint32_t feature, const WgFgLuid* luid, const void* native_adapter);
    int32_t get_function(uint32_t feature, const char* name, void** out);
    int32_t set_options(uint32_t feature, uint32_t viewport_id, const WgFgOptions* decoded, const void* native_options);
    int32_t get_state(uint32_t feature, uint32_t viewport_id, WgFgState* decoded, void* native_state);
    // shared calls: observed while taken, always forwarded; the result is the next link's
    int32_t tag(const WgFgTag* decoded, uint32_t count, const void* native_tags);
    int32_t constants(const WgFgConstants* decoded, const void* native_constants);
    // The FG feature-function pointers <get-function> hands out, by binding name (they must stay valid until unload).
    void set_feature_function(const std::string& name, void* fn);

    // ---- present (from the chain's present owner) ----
    uint32_t claim_present(uint64_t swapchain_id, uint32_t owner);
    bool release_present(uint64_t swapchain_id, uint32_t owner);
    // Not taken: forwarded to the next link (FORWARDED). Taken: owner and interval checked, inputs assembled and handed
    // to the executor (GENERATE), or the game image shown without generation for this interval (WG_FG_PRESENT_REAL_ONLY
    // + reason), or the call
    // refused (REFUSED + reason: not the owner, an interval already consumed, bad arguments). Returns 0 unless REFUSED.
    int32_t present(const WgFgPresentDesc* desc, WgFgPresentResult* out);

    // ---- host provider table (C trampolines over this instance) ----
    const WgFgProviderV1* provider() const { return &provider_; }

    // ---- views for the status word, tests and diagnostics ----
    WgFgState state() const;
    bool session_desc(WgFgSessionDesc* out) const;  // false until options and a present latched the session
    bool session_viewport(uint32_t* out) const;     // the viewport accepted options named; false before any
    uint32_t log_count(const std::string& needle) const;
    size_t frames_kept() const;

private:
    struct Frame {
        WgFgTexture tex[WG_FG_ROLE_COUNT] = {};
        uint32_t have = 0;       // bit per role
        uint32_t conflict = 0;   // bit per role tagged twice with different resources
        bool has_constants = false;
        WgFgConstants constants = {};
    };

    uint32_t route_reason_locked() const;
    uint32_t effective_max_locked() const;
    void apply_live_locked();
    void note_route_locked();
    void note_frame_locked(uint32_t reason, uint32_t real_only);
    void log_locked(const std::string& line) const;
    int32_t role_of(uint32_t native_type) const;
    void prune_locked(uint64_t presented_token);
    Frame& frame_locked(uint64_t token);
    void finish_result_locked(WgFgPresentResult* out, uint32_t status, uint32_t reason, uint32_t real_only) const;
    uint32_t assemble_locked(const WgFgPresentDesc& d, WgFgPacket* p, uint32_t* real_only);

    static int32_t P_get_capability(void* self, WgFgCapability* out);
    static int32_t P_set_options(void* self, uint32_t viewport_id, const WgFgOptions* options);
    static int32_t P_tag(void* self, const WgFgTag* tags, uint32_t count);
    static int32_t P_constants(void* self, const WgFgConstants* constants);
    static int32_t P_present(void* self, const WgFgPresentDesc* present, WgFgPresentResult* out);
    static int32_t P_get_state(void* self, WgFgState* out);

    FgProxyConfig cfg_;
    FgPresentOwners* owners_;
    FgLogFn log_;
    void* log_ctx_;

    mutable std::recursive_mutex mu_;
    uint32_t entry_ = WG_FG_ENTRY_NONE;
    WgFgChainV1 chain_ = {};
    WgFgHostV1 host_ = {};
    WgFgProviderV1 provider_ = {};
    uint32_t entry_max_ = 0;
    WgFgAdapterIdentity adapter_ = {};
    bool adapter_known_ = false;
    void* device_ = nullptr;
    void* queue_ = nullptr;
    bool model_validated_ = false, presenter_qualified_ = false;
    WgFgExecutorV1 executor_ = {};
    bool has_executor_ = false;
    std::map<std::string, void*> fns_;

    // session
    bool has_options_ = false;
    WgFgOptions options_ = {};
    uint32_t session_vp_ = 0;
    bool has_vp_ = false;
    uint32_t effective_preset_ = 0, effective_count_ = 0;
    bool latched_ = false;            // a present latched chain, window and epochs
    uint64_t swapchain_id_ = 0, window_id_ = 0, queue_epoch_ = 0, swapchain_epoch_ = 0;
    bool any_present_ = false;
    uint64_t last_token_ = 0;
    std::map<uint64_t, Frame> frames_;  // by frame token, session viewport only
    std::map<uint32_t, uint64_t> latest_seen_;  // role -> newest token a tag of that role was seen for
    uint32_t claimed_owner_ = WG_FG_OWNER_NONE;
    uint64_t claimed_chain_ = 0;

    // counters
    uint64_t base_presents_ = 0, generate_requests_ = 0, real_only_presents_ = 0, forwarded_presents_ = 0;

    // transitions (one line each)
    bool route_noted_ = false;
    uint32_t last_route_reason_ = 0;
    bool frame_noted_ = false;
    uint32_t last_frame_reason_ = 0, last_frame_real_only_ = 0;
    uint32_t last_options_reason_ = 0;
    uint32_t last_clamped_ = 0;
    mutable std::vector<std::string> log_lines_;

    // live settings (update_live_settings): sequence number << 32 | preset << 8 | multiplier, applied once per change
    std::atomic<uint64_t> live_{0};
    uint32_t live_applied_ = 0;
};

}  // namespace wgfgp
