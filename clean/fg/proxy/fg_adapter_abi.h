// SPDX-License-Identifier: MIT
/* WrongGPU frame generation (issue #96, F8a): the host-neutral, versioned contract between a game adapter's transport
 * and the FG adapter core (fg_proxy_core.h).
 *
 * Two entries share it:
 *  - interposition of the game interface's frame generation feature (feature 1000), for a title whose interface is
 *    reached the way the RR plugin reaches it (clean/rr_f/proxy/HOOK-MODE.md): the transport decodes the game's calls
 *    into these structs, passes the native arguments along, and the core forwards every call it does not own to the
 *    next link (WgFgChainV1) unchanged;
 *  - a host-owned FG provider: the core registers a WgFgProviderV1 table with a host that offers WgFgHostV1.
 *
 * Every struct starts with its size (and, where it crosses the boundary on its own, the contract version), so a
 * smaller struct from an older side is recognised and refused instead of being read past its end. Numbers that also
 * exist in the FG runtime contract (clean/fg/runtime/fg_contract.h: presets, optional inputs, frame flags, refusal and
 * real-only reasons, N = 1..5) have the same values here; fg_contract_bridge.h converts between the two.
 * Plain C, no Windows or D3D12 headers: resources are opaque pointers, formats are DXGI_FORMAT values as integers.
 * Written for this project.
 */
#ifndef WG_FG_ADAPTER_ABI_H
#define WG_FG_ADAPTER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WG_FG_ADAPTER_ABI_VERSION 1u
#define WG_FG_MAX_GENERATED 5u          /* N = 1..5 generated images per base interval (2x .. 6x) */
#define WG_FG_PRESET_COUNT 3u
#define WG_FG_PRESET_NOT_SET 0xffffffffu /* options: the game leaves the preset to WrongGPU's setting */
#define WG_FG_COUNT_NOT_SET 0u           /* options: the game leaves N to WrongGPU's setting */
#define WG_FG_NATIVE_TYPE_UNBOUND 0xffffffffu

enum WgFgPreset { WG_FG_PRESET_A = 0, WG_FG_PRESET_B = 1, WG_FG_PRESET_C = 2 };
#define WG_FG_PRESETS_ALL 0x7u

/* Which entry an adapter instance uses; one entry per instance. */
enum WgFgEntry { WG_FG_ENTRY_NONE = 0, WG_FG_ENTRY_INTERPOSER = 1, WG_FG_ENTRY_HOST_PROVIDER = 2 };

/* Who presents a swap chain. Exactly one owner per chain (FgPresentOwners in fg_proxy_core.h). */
enum WgFgPresentOwner {
  WG_FG_OWNER_NONE = 0,
  WG_FG_OWNER_HOST_WRAPPER = 1,      /* a host's FG swap-chain wrapper */
  WG_FG_OWNER_NATIVE_ADAPTER = 2,    /* this adapter, present-driven through the game interface */
  WG_FG_OWNER_SWAPCHAIN_WRAPPER = 3  /* WrongGPU's own presenter wrapper */
};

/* The inputs the core assembles. Roles 3..6 are the optional inputs, in the runtime contract's order (role - 3). */
enum WgFgRole {
  WG_FG_ROLE_BACKBUFFER = 0, WG_FG_ROLE_MOTION = 1, WG_FG_ROLE_DEPTH = 2, WG_FG_ROLE_HUDLESS = 3,
  WG_FG_ROLE_UI_ALPHA = 4, WG_FG_ROLE_DISTORTION = 5, WG_FG_ROLE_NO_WARP = 6
};
#define WG_FG_ROLE_COUNT 7u
#define WG_FG_OPTIONAL_FIRST_ROLE 3u
#define WG_FG_OPTIONAL_COUNT 4u

enum WgFgOptionalDecl { WG_FG_DECL_NEVER = 0, WG_FG_DECL_MAY = 1, WG_FG_DECL_ALWAYS = 2 };

/* Per-frame flags (same bits as the runtime contract's FgFrameFlag). */
enum WgFgFlag {
  WG_FG_FLAG_RESET = 1u << 0, WG_FG_FLAG_MENU = 1u << 1, WG_FG_FLAG_DEPTH_REVERSED = 1u << 2,
  WG_FG_FLAG_MOTION_RENDER_RES = 1u << 3, WG_FG_FLAG_HDR = 1u << 4, WG_FG_FLAG_REAL_OUTPUT = 1u << 5
};
#define WG_FG_FLAGS_KNOWN 0x3fu

enum WgFgColorSpace { WG_FG_CS_SDR = 0, WG_FG_CS_HDR10_PQ = 1, WG_FG_CS_SCRGB_LINEAR = 2 };
#define WG_FG_CS_COUNT 3u

/* Presenter choice (launcher setting fg_presenter; the runtime contract's presenter_mode). */
enum WgFgPresenterMode {
  WG_FG_PRESENTER_AUTO = 0, WG_FG_PRESENTER_HWFQ = 1, WG_FG_PRESENTER_DXGI_TEARING = 2, WG_FG_PRESENTER_DXGI_VSYNC = 3
};

/* Reasons. 1..82 are the runtime contract's FgRefusal numbers (only the ones an adapter reports are named here);
 * 90.. are the adapter's own. Stable numbers: they reach logs and the front end. */
enum WgFgReason {
  WG_FG_REASON_NONE = 0,
  WG_FG_REASON_STRUCT_SIZE = 1,
  WG_FG_REASON_ABI_VERSION = 2,
  WG_FG_REASON_BAD_PRESET = 8,
  WG_FG_REASON_BAD_GENERATED_COUNT = 9,
  WG_FG_REASON_BAD_OPTIONAL_DECL = 11,
  WG_FG_REASON_TOKEN_NOT_MONOTONIC = 20,
  WG_FG_REASON_VIEWPORT_MISMATCH = 21,
  WG_FG_REASON_LUID_MISMATCH = 22,
  WG_FG_REASON_QUEUE_EPOCH_MISMATCH = 23,
  WG_FG_REASON_SWAPCHAIN_EPOCH_MISMATCH = 24,
  WG_FG_REASON_MISSING_COLOR = 25,
  WG_FG_REASON_MISSING_MOTION = 26,
  WG_FG_REASON_MISSING_DEPTH = 27,
  WG_FG_REASON_MISSING_PROMISED_OPTIONAL = 28,
  WG_FG_REASON_UNDECLARED_OPTIONAL = 29,
  WG_FG_REASON_CONFLICTING_TAG = 30,
  WG_FG_REASON_STALE_TAG = 31,
  WG_FG_REASON_INSTALL_OFF = 60,
  WG_FG_REASON_UNSUPPORTED_ADAPTER = 61,
  WG_FG_REASON_MODEL_NOT_VALIDATED = 62,
  WG_FG_REASON_PRESENTER_NOT_QUALIFIED = 63,
  WG_FG_REASON_ROUTE_NOT_QUALIFIED = 64,  /* the FG modes are not qualified (kFgModesQualified) */
  WG_FG_REASON_BINDING_NOT_QUALIFIED = 90,
  WG_FG_REASON_NOT_ATTACHED = 91,
  WG_FG_REASON_NULL_HOST = 92,
  WG_FG_REASON_DOUBLE_ATTACH = 93,
  WG_FG_REASON_VERSION_MISMATCH = 94,
  WG_FG_REASON_HOST_REFUSED = 95,
  WG_FG_REASON_PRESENT_OWNED = 96,        /* another presenter owns the chain */
  WG_FG_REASON_NOT_PRESENT_OWNER = 97,    /* the caller has not claimed the chain */
  WG_FG_REASON_MISSING_CONSTANTS = 98,
  WG_FG_REASON_SECOND_VIEWPORT = 99,
  WG_FG_REASON_UNKNOWN_FUNCTION = 100,
  WG_FG_REASON_NO_EXECUTOR = 101,
  WG_FG_REASON_EXECUTOR_REFUSED = 102,
  WG_FG_REASON_NULL_ARGUMENT = 103
};

/* Why a frame is shown real-only (same numbers as the runtime contract's FgRealOnlyReason). */
enum WgFgRealOnly {
  WG_FG_REAL_ONLY_NONE = 0, WG_FG_REAL_ONLY_PRIME = 1, WG_FG_REAL_ONLY_RESET = 2, WG_FG_REAL_ONLY_MENU = 3,
  WG_FG_REAL_ONLY_RESIZE = 4, WG_FG_REAL_ONLY_GAP = 5, WG_FG_REAL_ONLY_EPOCH_CHANGE = 6,
  WG_FG_REAL_ONLY_HISTORY_BUSY = 7, WG_FG_REAL_ONLY_DISABLED = 8
};

enum WgFgPresentStatus {
  WG_FG_PRESENT_FORWARDED = 0,  /* FG not taken: the call went to the next link unchanged */
  WG_FG_PRESENT_GENERATE = 1,   /* a packet was assembled and accepted by the executor */
  WG_FG_PRESENT_REAL_ONLY = 2,  /* the game image is shown without generation for this interval (reason says why) */
  WG_FG_PRESENT_REFUSED = 3     /* the call itself was refused (not the owner, a repeated interval, bad arguments) */
};

typedef struct WgFgLuid { uint32_t low; int32_t high; } WgFgLuid;
typedef struct WgFgRect { uint32_t x, y, width, height; } WgFgRect;

/* The D3D12 adapter the game's device runs on, as the transport read it (never spoofed by the adapter). */
typedef struct WgFgAdapterIdentity {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t vendor_id;
  uint32_t device_id;
  WgFgLuid luid;
  uint32_t rdna4;          /* 1: an RX 9000 / RDNA 4 part */
  uint32_t probes_passed;  /* 1: the required D3D12 feature probes passed on it */
} WgFgAdapterIdentity;

/* What the adapter advertises for FG. max_generated is 0 unless every route fact holds and the modes are qualified;
 * otherwise min(5, the entry's ABI maximum, the game's UI range). */
typedef struct WgFgCapability {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t supported;      /* 1: WrongGPU takes FG on this binding and adapter */
  uint32_t max_generated;  /* 0..5 */
  uint32_t presets_mask;   /* bit per WgFgPreset; 0 when not supported */
  uint32_t reason;         /* WgFgReason: the first route fact that failed, 0 when supported */
  WgFgAdapterIdentity adapter;
} WgFgCapability;

/* FG options as decoded from the game's (or the host's) own options call. */
typedef struct WgFgOptions {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t enable;           /* the game's FG switch: 0 off, 1 on */
  uint32_t preset;           /* WgFgPreset, or WG_FG_PRESET_NOT_SET */
  uint32_t generated_count;  /* N requested (1..5), or WG_FG_COUNT_NOT_SET */
  uint32_t optional_decl[WG_FG_OPTIONAL_COUNT];  /* WgFgOptionalDecl per optional input */
  uint32_t color_space;      /* WgFgColorSpace */
  uint32_t output_width, output_height, output_format;
  uint32_t render_min_width, render_min_height, render_max_width, render_max_height;
  uint32_t dynamic_resolution;
} WgFgOptions;

/* One texture as the runtime contract's FgTextureRef (same fields, same order). */
typedef struct WgFgTexture {
  void* resource;
  uint32_t format;
  uint32_t width, height;
  uint32_t subresource;
  uint32_t state;
  uint64_t producer_epoch;
  uint64_t frame_token;
  uint32_t viewport_id;
  WgFgRect rect;
} WgFgTexture;

/* One decoded resource tag: the binding's buffer type number plus the texture. A null resource removes nothing (tags
 * are observed, never owned). */
typedef struct WgFgTag {
  uint32_t struct_size;
  uint32_t native_type;
  WgFgTexture texture;
} WgFgTag;

/* Per-frame constants, decoded from the game's constants call. */
typedef struct WgFgConstants {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t frame_token;
  uint32_t viewport_id;
  uint32_t flags;            /* WgFgFlag */
  float camera_current[16];
  float camera_previous[16];
  float jitter_x, jitter_y;
  float motion_scale_x, motion_scale_y;
  WgFgRect render_rect;
  WgFgRect output_rect;
  uint32_t color_space;      /* WgFgColorSpace */
  uint32_t pad_;
} WgFgConstants;

/* One game present on a chain, from its present owner. */
typedef struct WgFgPresentDesc {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t owner;            /* WgFgPresentOwner of the caller */
  uint32_t viewport_id;
  uint64_t frame_token;
  uint64_t swapchain_id;     /* opaque identity of the chain */
  uint64_t window_id;
  uint64_t queue_epoch;
  uint64_t swapchain_epoch;
  WgFgLuid luid;
  WgFgTexture backbuffer;    /* the final backbuffer; a null resource: take the backbuffer tag if the binding has one */
} WgFgPresentDesc;

/* The assembled frame, as the runtime contract's FgFramePacket (same fields, same order, no size/version pair: the
 * bridge adds them). */
typedef struct WgFgPacket {
  uint64_t frame_token;
  uint32_t viewport_id;
  WgFgLuid adapter_luid;
  uint64_t queue_epoch;
  uint64_t swapchain_epoch;
  WgFgTexture color;
  WgFgTexture motion;
  WgFgTexture depth;
  WgFgTexture optional[WG_FG_OPTIONAL_COUNT];
  uint32_t optional_present;  /* bit (role - 3) per optional input carried */
  WgFgRect render_rect;
  WgFgRect output_rect;
  float jitter_x, jitter_y;
  float motion_scale_x, motion_scale_y;
  float camera_current[16];
  float camera_previous[16];
  uint32_t flags;
  uint32_t preset;
  uint32_t generated_count;   /* N, 1..5 */
  uint32_t generated_index;   /* 0: present-driven (the executor expands i = 1..N) */
} WgFgPacket;

/* The session an executor is created for, as the runtime contract's FgCreateDesc (same fields, same order). */
typedef struct WgFgSessionDesc {
  void* device;
  void* queue;
  WgFgLuid adapter_luid;
  uint64_t swapchain_id;
  uint64_t window_id;
  uint64_t queue_epoch;
  uint64_t swapchain_epoch;
  uint32_t viewport_id;
  uint32_t output_width, output_height;
  uint32_t output_format;
  uint32_t render_min_width, render_min_height;
  uint32_t render_max_width, render_max_height;
  uint32_t dynamic_resolution;
  uint32_t preset;
  uint32_t max_generated;
  uint32_t optional_decl[WG_FG_OPTIONAL_COUNT];
  uint32_t hdr;
  uint32_t presenter_mode;   /* WgFgPresenterMode */
} WgFgSessionDesc;

typedef struct WgFgPresentResult {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t status;           /* WgFgPresentStatus */
  uint32_t reason;           /* WgFgReason (0 when generating or forwarded) */
  uint32_t real_only;        /* WgFgRealOnly */
  uint32_t requested_preset, effective_preset;
  uint32_t requested_count, effective_count;
  uint32_t pad_;
  uint64_t base_presents;    /* game presents this adapter accounted (taken route only) */
  uint64_t generate_requests;
  uint64_t real_only_presents;
  uint64_t forwarded_presents;
} WgFgPresentResult;

typedef struct WgFgState {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t taken;            /* 1: WrongGPU owns FG on this adapter */
  uint32_t enabled;          /* the game's FG switch */
  uint32_t effective_preset;
  uint32_t effective_count;
  uint32_t max_generated;
  uint32_t last_reason;
  uint64_t base_presents, generate_requests, real_only_presents;
} WgFgState;

/* The executor the core hands assembled packets to (F3a / F7 side; a fake in tests). 0 = accepted. */
typedef struct WgFgExecutorV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void* self;
  int32_t (*submit)(void* self, const WgFgPacket* packet, const WgFgSessionDesc* session);
} WgFgExecutorV1;

/* ---- interposer entry: the next link of the game interface (the genuine interface, or whatever the host chained
 * in front of it). The transport fills it; every call the core does not own goes here with the game's own native
 * arguments. Result codes are the game interface's (0 = success; clean/rr_f/proxy/include/rr_proxy_abi.h). ---- */
typedef struct WgFgChainV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void* self;
  uint32_t max_generated;    /* the game interface ABI's highest generated count (0..5) */
  uint32_t pad_;
  int32_t (*is_supported)(void* self, uint32_t feature, const void* native_adapter);
  int32_t (*get_function)(void* self, uint32_t feature, const char* name, void** out);
  int32_t (*set_options)(void* self, uint32_t feature, uint32_t viewport_id, const void* native_options);
  int32_t (*get_state)(void* self, uint32_t feature, uint32_t viewport_id, void* native_state);
  int32_t (*tag)(void* self, const void* native_tags, uint32_t count);
  int32_t (*constants)(void* self, const void* native_constants);
  int32_t (*present)(void* self, const WgFgPresentDesc* present);
} WgFgChainV1;

/* ---- host provider entry ---- */
typedef struct WgFgProviderV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void* self;
  int32_t (*get_capability)(void* self, WgFgCapability* out);
  int32_t (*set_options)(void* self, uint32_t viewport_id, const WgFgOptions* options);
  int32_t (*tag)(void* self, const WgFgTag* tags, uint32_t count);
  int32_t (*constants)(void* self, const WgFgConstants* constants);
  int32_t (*present)(void* self, const WgFgPresentDesc* present, WgFgPresentResult* out);
  int32_t (*get_state)(void* self, WgFgState* out);
} WgFgProviderV1;

typedef struct WgFgHostV1 {
  uint32_t struct_size;
  uint32_t abi_version;
  void* self;
  uint32_t max_generated;    /* the host ABI's highest generated count (0..5) */
  uint32_t pad_;
  int32_t (*register_provider)(void* self, const WgFgProviderV1* provider);    /* 0 = accepted */
  int32_t (*unregister_provider)(void* self, const WgFgProviderV1* provider);
} WgFgHostV1;

#ifdef __cplusplus
} /* extern "C" */

static_assert(sizeof(WgFgLuid) == 8 && sizeof(WgFgRect) == 16, "fg adapter abi: luid / rect");
static_assert(sizeof(WgFgTexture) == 72 && offsetof(WgFgTexture, producer_epoch) == 32 &&
              offsetof(WgFgTexture, rect) == 52, "fg adapter abi: texture");
static_assert(offsetof(WgFgTag, texture) == 8, "fg adapter abi: tag");
static_assert(offsetof(WgFgPresentDesc, backbuffer) == 64, "fg adapter abi: present");
static_assert(WG_FG_OPTIONAL_FIRST_ROLE + WG_FG_OPTIONAL_COUNT == WG_FG_ROLE_COUNT, "fg adapter abi: roles");
#endif

#endif
