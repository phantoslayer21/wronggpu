// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the native codec of the 2.14 game interface's FG structures.
//
// What a qualified title binding on interface build 2.14.x (bindings/cp2077-2.14.1.json) hands the transport as its
// IFgNativeCodec: the FG options (version 5, 120 B) and state (version 4, 88 B) layouts, the chain walk over the
// interface's 32-byte structure header, and the decoding of the shared resource tags and per-frame constants
// (rr_proxy_abi.h, the same objects the RR plugin decodes) into fg_adapter_abi.h structs.
//
//  - CHAIN WALK. Every structure starts with the header (next pointer, 16-byte type GUID, version). A structure is found
//    by GUID at the given pointer or behind it, at most kFgMaxChainLinks links; a cycle or an over-long chain ends the
//    walk. A structure whose version is lower than the binding's is not read.
//  - OPTIONS (set-options). The checks the original makes, in its order: the options or the viewport missing from the
//    chain -> 20; generated count 0 (even with the mode off) -> 25; generated count above the reported maximum (state
//    +52) -> 38. Then the mode: 0 off, 1 on, 2 auto (WrongGPU's own pacing), 3 dynamic (refused with 25 while the
//    binding reports no dynamic support, which is what the state's +80 says). Decoded: the switch, the count, preset B
//    when the UI-recomposition byte (+112) is exactly 1 (otherwise the player's preset), the dynamic-resolution flag and
//    size, the colour and motion/depth sizes and formats the game may fill (both supported titles leave them 0), the
//    optional-input declarations from the binding.
//  - STATE (get-state). Written only up to the passed version: v1 the VRAM estimate (always 0: WrongGPU makes none),
//    the status bits it can know (1 resolution below the minimum in the passed options, 8 a depth-inverted /
//    camera-motion / reset tri-state left unset in the last constants), the minimum size (100) and the presented count;
//    v2 the highest generated count (+52) and the v-sync byte; v3 the fence pointer and value (none: 0); v4 the dynamic
//    support byte (+80).
//  - TAGS. One decoded tag per native tag. A tag is an FG input only when its GUID and its resource's GUID are the
//    binding's, the resource is a 2D texture, the buffer type is one of the binding's FG inputs and its lifetime is one
//    the binding admits (2.14.1 / Cyberpunk: "valid until present" only; a tag valid only during the call would need a
//    copy, which this stage does not make). Every other tag is marked unbound and still forwarded by the transport.
//  - FRAME CLOCK. The game tags with the plain tag call (no frame token). All tags and constants between two game
//    presents belong to one interval: they carry frame token (presents so far + 1); the present bridge presents that
//    token and then advances the clock (frame_for_present / note_present).
//  - CONSTANTS. view-to-clip of this frame and of the previous decoded frame, jitter, motion-vector scale, flags from
//    the reset and depth-inverted tri-states. Unset tri-states are remembered for the state's status bit 8.
// Only structures; no Windows calls. Thread safety: atomics for the clock and the remembered values, one mutex for the
// previous matrix.
// Written for this project.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <mutex>

#include "fg_asi_transport.h"

namespace wgfgp {
namespace win32 {

inline constexpr uint32_t kFgMaxChainLinks = 32;

// ---- native FG structures of interface build 2.14.x (offsets from the structure start; names our own) ---------------

struct WgNatFgOptions {          // version 5, 120 B
    WgHeader h;                  // +0
    uint32_t mode;               // +32 0 off, 1 on, 2 auto, 3 dynamic
    uint32_t frames;             // +36 generated count, 1..max even when off
    uint32_t flags;              // +40 bit 1 dynamic resolution, bit 2 VRAM estimate requested
    uint32_t dynres_width;       // +44 with flag bit 1
    uint32_t dynres_height;      // +48
    uint32_t back_buffers;       // +52
    uint32_t mvdepth_width;      // +56
    uint32_t mvdepth_height;     // +60
    uint32_t colour_width;       // +64
    uint32_t colour_height;      // +68
    uint32_t colour_format;      // +72 DXGI format number
    uint32_t mv_format;          // +76
    uint32_t depth_format;       // +80
    uint32_t hudless_format;     // +84
    uint32_t ui_format;          // +88
    uint32_t pad0_;              // +92
    void* error_callback;        // +96
    uint8_t latency_matrices;    // +104 tri-state, default 2
    uint8_t pad1_[3];
    uint32_t queue_mode;         // +108 default 0
    uint8_t ui_recomposition;    // +112 exactly 1: on
    uint8_t pad2_[3];
    float dynamic_target_fps;    // +116 with mode 3
};

struct WgNatFgState {            // version 4, 88 B
    WgHeader h;                  // +0
    uint64_t estimated_vram;     // +32 v1
    uint32_t status;             // +40 v1 bit field (kFgNatStatus*)
    uint32_t min_size;           // +44 v1 always 100
    uint32_t frames_presented;   // +48 v1
    uint32_t frames_max;         // +52 v2 highest generated count
    uint8_t reserved56;          // +56 not written
    uint8_t vsync_available;     // +57 v2
    uint8_t pad0_[6];
    void* fence;                 // +64 v3
    uint64_t fence_value;        // +72 v3
    uint8_t dynamic_supported;   // +80 v4
    uint8_t pad1_[7];
};

static_assert(sizeof(WgNatFgOptions) == 120 && offsetof(WgNatFgOptions, mode) == 32 &&
                  offsetof(WgNatFgOptions, frames) == 36 && offsetof(WgNatFgOptions, flags) == 40 &&
                  offsetof(WgNatFgOptions, dynres_width) == 44 && offsetof(WgNatFgOptions, back_buffers) == 52 &&
                  offsetof(WgNatFgOptions, mvdepth_width) == 56 && offsetof(WgNatFgOptions, colour_width) == 64 &&
                  offsetof(WgNatFgOptions, colour_format) == 72 && offsetof(WgNatFgOptions, ui_format) == 88 &&
                  offsetof(WgNatFgOptions, error_callback) == 96 && offsetof(WgNatFgOptions, latency_matrices) == 104 &&
                  offsetof(WgNatFgOptions, queue_mode) == 108 && offsetof(WgNatFgOptions, ui_recomposition) == 112 &&
                  offsetof(WgNatFgOptions, dynamic_target_fps) == 116,
              "fg options v5 layout");
static_assert(sizeof(WgNatFgState) == 88 && offsetof(WgNatFgState, estimated_vram) == 32 &&
                  offsetof(WgNatFgState, status) == 40 && offsetof(WgNatFgState, min_size) == 44 &&
                  offsetof(WgNatFgState, frames_presented) == 48 && offsetof(WgNatFgState, frames_max) == 52 &&
                  offsetof(WgNatFgState, reserved56) == 56 && offsetof(WgNatFgState, vsync_available) == 57 &&
                  offsetof(WgNatFgState, fence) == 64 && offsetof(WgNatFgState, fence_value) == 72 &&
                  offsetof(WgNatFgState, dynamic_supported) == 80,
              "fg state v4 layout");

enum FgNatMode : uint32_t { kFgNatModeOff = 0, kFgNatModeOn = 1, kFgNatModeAuto = 2, kFgNatModeDynamic = 3 };
inline constexpr uint32_t kFgNatFlagDynamicResolution = 1u << 1;
inline constexpr uint32_t kFgNatFlagVramEstimate = 1u << 2;
inline constexpr uint32_t kFgNatStatusResolutionLow = 0x1u;
inline constexpr uint32_t kFgNatStatusLatencyInactive = 0x2u;
inline constexpr uint32_t kFgNatStatusHdr64 = 0x4u;
inline constexpr uint32_t kFgNatStatusConstantUnset = 0x8u;
inline constexpr uint32_t kFgNatStatusNoBackBufferIndex = 0x10u;
// the minimum state version that carries each member
inline constexpr uint64_t kFgNatStateV2 = 2, kFgNatStateV3 = 3, kFgNatStateV4 = 4;

// ---- the binding's data the codec needs (filled from the title binding; tests fill synthetic GUIDs) -----------------

struct FgSl214Layout {
    WgGuid options_guid{}, state_guid{}, viewport_guid{}, tag_guid{}, resource_guid{}, constants_guid{};
    WgGuid requirements_guid{};          // all zero: the requirements header is not checked
    uint32_t options_version = 5, options_size = 120;
    uint32_t state_version = 4, state_size = 88;
    uint32_t constants_min_version = 1;
    uint32_t native_type[WG_FG_ROLE_COUNT] = {WG_FG_NATIVE_TYPE_UNBOUND, WG_FG_NATIVE_TYPE_UNBOUND,
                                              WG_FG_NATIVE_TYPE_UNBOUND, WG_FG_NATIVE_TYPE_UNBOUND,
                                              WG_FG_NATIVE_TYPE_UNBOUND, WG_FG_NATIVE_TYPE_UNBOUND,
                                              WG_FG_NATIVE_TYPE_UNBOUND};
    uint32_t optional_decl[WG_FG_OPTIONAL_COUNT] = {WG_FG_DECL_NEVER, WG_FG_DECL_NEVER, WG_FG_DECL_NEVER,
                                                    WG_FG_DECL_NEVER};
    uint32_t tag_lifetime_mask = 1u << WG_LIFE_UNTIL_PRESENT;  // bit per admitted WgTagLifetime
    uint32_t min_size = 100;
    bool dynamic_supported = false;
};
// true: the six structure GUIDs are set and pairwise different, the sizes match this codec's layouts
bool FgSl214LayoutValid(const FgSl214Layout& l);

bool FgGuidEq(const WgGuid& a, const WgGuid& b);
bool FgGuidZero(const WgGuid& g);
// "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" -> the in-memory GUID bytes (Data1/2/3 little-endian, Data4 as written)
bool FgParseGuid(const char* text, WgGuid* out);
// The first structure of type `type` at `start` or behind it (null: none within kFgMaxChainLinks links).
const WgHeader* FgFindInChain(const void* start, const WgGuid& type, uint32_t max_links = kFgMaxChainLinks);

struct FgSl214Counters {
    uint64_t tags_seen = 0, tags_bound = 0;
    uint64_t tags_wrong_guid = 0, tags_lifetime_refused = 0, tags_not_texture = 0;
    uint64_t constants_decoded = 0, constants_refused = 0;
    uint64_t options_checked = 0, options_refused = 0, states_written = 0;
};

class FgSl214Codec final : public IFgNativeCodec {
public:
    explicit FgSl214Codec(const FgSl214Layout& layout);

    // the original's set-options checks (see the header comment); WG_OK: decode and hand to the core
    WgResultCode check_options(const WgViewport* vp, const void* native, uint32_t max_generated) override;
    bool options(const WgViewport* vp, const void* native, WgFgOptions* out) override;
    bool write_state(const WgFgState& state, void* native) override;
    bool write_state_for(const WgFgState& state, const void* native_options, void* native) override;
    bool tags(const WgFrameToken* token, const WgViewport* vp, const WgTag* native, uint32_t count,
              WgFgTag* out) override;
    bool constants(const WgConstants* native, const WgFrameToken* token, const WgViewport* vp,
                   WgFgConstants* out) override;
    bool write_requirements(const WgFgCapability& cap, WgRequirements* native) override;
    uint64_t frame_for_present() override;
    void note_present() override;

    const FgSl214Layout& layout() const { return l_; }
    uint64_t presents() const { return interval_.load(); }
    uint32_t unset_constants() const { return unset_.load(); }  // bit 0 depth inverted, 1 camera motion, 2 reset
    FgSl214Counters counters() const;

    // The options structure in the chain at `native` (null: missing or older than the binding's version).
    const WgNatFgOptions* find_options(const void* native) const;
    // The viewport: in the chain at `native`, else the explicit argument when its GUID is the viewport's.
    const WgViewport* find_viewport(const void* native, const WgViewport* explicit_vp) const;

private:
    FgSl214Layout l_;
    std::atomic<uint64_t> interval_{0};
    std::atomic<uint32_t> unset_{0};
    std::atomic<bool> have_constants_{false};
    mutable std::mutex mu_;
    float cur_view_to_clip_[16] = {};
    float prev_view_to_clip_[16] = {};
    uint64_t cur_frame_ = 0;
    bool have_cur_ = false;
    std::atomic<uint64_t> c_tags_seen_{0}, c_tags_bound_{0}, c_tags_wrong_guid_{0}, c_tags_lifetime_{0},
        c_tags_not_texture_{0}, c_constants_{0}, c_constants_refused_{0}, c_options_checked_{0},
        c_options_refused_{0}, c_states_{0};
    uint32_t required_tags_[2] = {};
};

}  // namespace win32
}  // namespace wgfgp
