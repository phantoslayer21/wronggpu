// SPDX-License-Identifier: MIT
/* WrongGPU RR proxy: our own declarations of the game-facing structures (spec 60.2, 60.4, 60.5, 60.6).
 *
 * Written from the sizes, offsets and values of spec page 60 only (functional description; no vendor header
 * was used). Every structure is checked against the page with static_assert below. Names are our own.
 * Structure GUIDs and the version magic are NOT here: they arrive as data from the binding tables (annex A9.6,
 * A9.1; binding/, BindingIds in proxy_core.h).
 */
#ifndef WG_RR_PROXY_ABI_H
#define WG_RR_PROXY_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 60.5 result codes (identical in both versions) ---- */
enum WgResult {
  WG_OK = 0, WG_E_IO = 1, WG_E_DRIVER_OUT_OF_DATE = 2, WG_E_OS_OUT_OF_DATE = 3, WG_E_OS_HW_SCHEDULING = 4,
  WG_E_DEVICE_NOT_CREATED = 5, WG_E_NO_SUPPORTED_ADAPTER = 6, WG_E_ADAPTER_NOT_SUPPORTED = 7, WG_E_NO_PLUGINS = 8,
  WG_E_VULKAN_API = 9, WG_E_DXGI_API = 10, WG_E_D3D_API = 11, WG_E_REMOVED_12 = 12, WG_E_VENDOR_DRIVER_API = 13,
  WG_E_LATENCY_API = 14, WG_E_VENDOR_RUNTIME_FAILED = 15, WG_E_JSON_PARSING = 16, WG_E_MISSING_PROXY = 17,
  WG_E_MISSING_RESOURCE_STATE = 18, WG_E_INVALID_INTEGRATION = 19, WG_E_MISSING_INPUT_PARAMETER = 20,
  WG_E_NOT_INITIALISED = 21, WG_E_COMPUTE_FAILED = 22, WG_E_INIT_NOT_CALLED = 23, WG_E_EXCEPTION_HANDLER = 24,
  WG_E_INVALID_PARAMETER = 25, WG_E_MISSING_CONSTANTS = 26, WG_E_DUPLICATED_CONSTANTS = 27,
  WG_E_MISSING_OR_INVALID_API = 28, WG_E_COMMON_CONSTANTS_MISSING = 29, WG_E_UNSUPPORTED_INTERFACE = 30,
  WG_E_FEATURE_MISSING = 31, WG_E_FEATURE_NOT_SUPPORTED = 32, WG_E_FEATURE_MISSING_HOOKS = 33,
  WG_E_FEATURE_FAILED_TO_LOAD = 34, WG_E_FEATURE_WRONG_PRIORITY = 35, WG_E_FEATURE_MISSING_DEPENDENCY = 36,
  WG_E_FEATURE_MANAGER_INVALID_STATE = 37, WG_E_INVALID_STATE = 38, WG_W_OUT_OF_VIDEO_MEMORY = 39
};
#define WG_RESULT_COUNT 40u
/* A10.1: every call returns Result, a 32-bit signed enumeration in the integer return register; 0 = success. */
typedef int32_t WgResultCode;

/* ---- 60.4 feature ids ---- */
enum WgFeatureId {
  WG_FEAT_SUPER_RESOLUTION = 0, WG_FEAT_RESERVED_1 = 1, WG_FEAT_IMAGE_SCALING = 2, WG_FEAT_LATENCY = 3,
  WG_FEAT_PC_LATENCY = 4, WG_FEAT_COLOUR_ENHANCEMENT = 5, WG_FEAT_LATE_WARP = 6, WG_FEAT_FRAME_GENERATION = 1000,
  WG_FEAT_RAY_RECONSTRUCTION = 1001, WG_FEAT_PROFILING = 1002, WG_FEAT_DIRECTSR = 1003,
  WG_FEAT_NEURAL_1004 = 1004 /* 2.14.1 only */, WG_FEAT_DEBUG_UI = 9999
};
#define WG_FEAT_COMMON 0xffffffffu

/* ---- 60.6.3 resource kinds, tag lifetimes; 62.2 buffer types used by the RR path ---- */
enum WgResourceKind { WG_RES_TEX2D = 0, WG_RES_BUFFER = 1, WG_RES_CMD_QUEUE = 2, WG_RES_CMD_LIST = 3,
                      WG_RES_CMD_POOL = 4, WG_RES_FENCE = 5, WG_RES_SWAPCHAIN = 6, WG_RES_HOST_FENCE = 7,
                      WG_RES_UNKNOWN = 8 };
enum WgTagLifetime { WG_LIFE_ONLY_DURING_TAG = 0, WG_LIFE_UNTIL_PRESENT = 1, WG_LIFE_UNTIL_EVALUATE = 2 };
enum WgBufferType {
  WG_BT_DEPTH = 0, WG_BT_MOTION = 1, WG_BT_COLOUR_IN = 3, WG_BT_COLOUR_OUT = 4, WG_BT_NORMALS = 5,
  WG_BT_ROUGHNESS = 6, WG_BT_DIFFUSE_ALBEDO = 7, WG_BT_SPECULAR_ALBEDO = 8, WG_BT_SPECULAR_MOTION = 10,
  WG_BT_DISOCCLUSION = 11, WG_BT_EXPOSURE = 13, WG_BT_NORMAL_ROUGHNESS = 14, WG_BT_INPUT_ALPHA = 34,
  WG_BT_COLOUR_BEFORE_PARTICLES = 39, WG_BT_COLOUR_BEFORE_TRANSPARENCY = 40, WG_BT_COLOUR_BEFORE_FOG = 41,
  WG_BT_SPECULAR_HIT_DISTANCE = 42, WG_BT_LINEAR_DEPTH = 49, WG_BT_TRANSPARENCY_LAYER = 51,
  WG_BT_TRANSPARENCY_OPACITY = 52, WG_BT_SSS_GUIDE = 58, WG_BT_DOF_GUIDE = 64, WG_BT_OUTPUT_ALPHA = 67,
  WG_BT_RESPONSIVITY = 68
};
#define WG_BT_COUNT_2_7 67u   /* 0-66 */
#define WG_BT_COUNT_2_14 73u  /* + 67-72 */

/* ---- 60.6.5 RR modes ---- */
enum WgRrMode { WG_RR_MODE_OFF = 0, WG_RR_MODE_MAX_PERFORMANCE = 1, WG_RR_MODE_BALANCED = 2,
                WG_RR_MODE_MAX_QUALITY = 3, WG_RR_MODE_ULTRA_PERFORMANCE = 4, WG_RR_MODE_ULTRA_QUALITY = 5,
                WG_RR_MODE_NATIVE_AA = 6 };

/* ---- 60.2 tri-state boolean (1 byte) ---- */
#define WG_TRI_FALSE 0u
#define WG_TRI_TRUE 1u
#define WG_TRI_UNSET 2u

/* ---- 60.2 structure header (32 B) ---- */
typedef struct WgGuid { uint8_t b[16]; } WgGuid;
typedef struct WgHeader {
  struct WgHeader* next; /* +0 null-terminated chain */
  WgGuid type;           /* +8 */
  uint64_t version;      /* +24 */
} WgHeader;

/* Frame token (60.2, A10.1): an object created and owned by the original (<new-frame-token>), passed back by
 * reference. +0 pointer to a table of functions, +8 the 32-byte header (type GUID at +16), size 40. The frame index
 * is obtained by calling the first function of that table with the token as its only argument (a member call:
 * under the single x64 convention the object pointer is the first argument); it returns u32. The proxy never
 * creates or copies a token and reads the index only through that function. */
typedef struct WgFrameToken WgFrameToken;
typedef uint32_t (*WgFrameTokenIndexFn)(const WgFrameToken* self);
struct WgFrameToken {
  const WgFrameTokenIndexFn* vtable; /* +0 */
  WgHeader header;                   /* +8 */
};

/* 60.6.1 viewport handle (40 B, v1) */
typedef struct WgViewport { WgHeader h; uint32_t id; uint32_t pad_; } WgViewport;
#define WG_VIEWPORT_INVALID 0xffffffffu

/* 60.6.2 extent (16 B) */
typedef struct WgExtent { uint32_t top, left, width, height; } WgExtent;

/* 60.6.3 resource description (112 B, v1) */
typedef struct WgResource {
  WgHeader h;
  uint8_t kind;          /* +32 WgResourceKind */
  uint8_t pad0_[7];
  void* native;          /* +40 */
  void* memory;          /* +48 Vulkan */
  void* view;            /* +56 Vulkan */
  uint32_t state;        /* +64 D3D12 state bits, all ones = not given */
  uint32_t width;        /* +68 */
  uint32_t height;       /* +72 */
  uint32_t native_format;/* +76 DXGI format number */
  uint32_t mip_levels;   /* +80 */
  uint32_t array_layers; /* +84 */
  uint64_t gpu_va;       /* +88 */
  uint32_t flags;        /* +96 Vulkan */
  uint32_t usage;        /* +100 Vulkan */
  uint32_t tail;         /* +104: 2.7.x reserved u32; 2.14.1 u16 internal flags (do not modify) + u16 reserved */
  uint32_t pad1_;        /* +108 */
} WgResource;
#define WG_STATE_NOT_GIVEN 0xffffffffu

/* 60.6.3 tag (64 B, v1) and its optional chained conversion extension (48 B, v1) */
typedef struct WgTag {
  WgHeader h;
  WgResource* resource;  /* +32 null = remove this tag */
  uint32_t buffer_type;  /* +40 */
  uint32_t lifetime;     /* +44 */
  WgExtent extent;       /* +48 */
} WgTag;
typedef struct WgTagConversion {
  WgHeader h;
  uint32_t kind;         /* +32 0 none, 1 linear: high = low * scale + bias */
  float bias;            /* +36 */
  float scale;           /* +40 */
  uint32_t pad_;         /* +44 */
} WgTagConversion;

/* 60.6.4 per-frame constants (456 B, v2; v1 ends after +450) */
typedef struct WgConstants {
  WgHeader h;
  float view_to_clip[16];        /* +32 */
  float clip_to_view[16];        /* +96 */
  float clip_to_lens_clip[16];   /* +160 */
  float clip_to_prev_clip[16];   /* +224 */
  float prev_clip_to_clip[16];   /* +288 */
  float jitter[2];               /* +352 pixels */
  float mv_scale[2];             /* +360 */
  float pinhole_offset[2];       /* +368 */
  float camera_pos[3];           /* +376 */
  float camera_up[3];            /* +388 */
  float camera_right[3];         /* +400 */
  float camera_fwd[3];           /* +412 */
  float near_plane;              /* +424 */
  float far_plane;               /* +428 */
  float fov_y;                   /* +432 */
  float aspect;                  /* +436 */
  float mv_invalid_value;        /* +440 */
  uint8_t depth_inverted;        /* +444 tri-state */
  uint8_t camera_motion_included;/* +445 */
  uint8_t mv_3d;                 /* +446 */
  uint8_t reset;                 /* +447 */
  uint8_t orthographic;          /* +448 */
  uint8_t mv_dilated;            /* +449 */
  uint8_t mv_jittered;           /* +450 */
  uint8_t pad_;                  /* +451 */
  float min_depth_separation;    /* +452 v2 */
} WgConstants;

/* 60.6.5 RR options (224 B, v3; v1 ends after +191, v2 adds +192, v3 adds +196..+219) */
typedef struct WgRrOptions {
  WgHeader h;
  uint32_t mode;                 /* +32 WgRrMode */
  uint32_t output_width;         /* +36 all ones = not set */
  uint32_t output_height;        /* +40 */
  float sharpness;               /* +44 */
  float pre_exposure;            /* +48 */
  float exposure_scale;          /* +52 */
  uint8_t colour_is_hdr;         /* +56 tri-state */
  uint8_t indicator_invert_x;    /* +57 */
  uint8_t indicator_invert_y;    /* +58 */
  uint8_t pad0_;                 /* +59 */
  uint32_t normal_roughness_packing; /* +60 0 separate, 1 roughness in normals.w */
  float world_to_view[16];       /* +64 */
  float view_to_world[16];       /* +128 */
  uint8_t alpha_upscaling;       /* +192 tri-state, v2 */
  uint8_t pad1_[3];
  uint32_t presets[6];           /* +196 native AA, quality, balanced, performance, ultra performance, ultra quality (v3) */
  uint32_t pad2_;                /* +220 */
} WgRrOptions;
#define WG_RR_OPTIONS_V1_END 192u
#define WG_RR_OPTIONS_V2_END 196u
#define WG_RR_OPTIONS_V3_END 220u
#define WG_SIZE_NOT_SET 0xffffffffu

/* 60.6.6 optimal settings (64 B, v1) and state (40 B, v1) */
typedef struct WgRrOptimal {
  WgHeader h;
  uint32_t optimal_width, optimal_height; /* +32, +36 */
  float optimal_sharpness;                /* +40 */
  uint32_t min_width, min_height;         /* +44, +48 */
  uint32_t max_width, max_height;         /* +52, +56 */
  uint32_t pad_;                          /* +60 */
} WgRrOptimal;
typedef struct WgRrState { WgHeader h; uint64_t estimated_vram; } WgRrState;

/* 60.6.7 feature requirements (184 B, v2) and version (56 B, v1) */
typedef struct WgRequirements {
  WgHeader h;
  uint32_t flags;               /* +32 bit0 D3D11, bit1 D3D12, bit2 Vulkan, bit3 vsync off, bit4 HW scheduling */
  uint32_t cpu_threads;         /* +36 */
  uint32_t max_viewports;       /* +40 */
  uint32_t required_tag_count;  /* +44 */
  const uint32_t* required_tags;/* +48 */
  uint32_t os_detected[3];      /* +56 */
  uint32_t os_required[3];      /* +68 */
  uint32_t driver_detected[3];  /* +80 */
  uint32_t driver_required[3];  /* +92 */
  uint8_t vulkan_only[80];      /* +104..+183 (queue counts, extension/feature name lists, optical-flow queues +176) */
} WgRequirements;
typedef struct WgFeatureVersion {
  WgHeader h;
  uint32_t api_version[3];      /* +32 Streamline version major, minor, build */
  uint32_t runtime_version[3];  /* +44 feature runtime version */
} WgFeatureVersion;

/* 60.6.8 adapter info (56 B, v1) */
typedef struct WgAdapterInfo {
  WgHeader h;
  const void* luid;             /* +32 */
  uint32_t luid_size;           /* +40 */
  uint32_t pad_;
  void* vk_physical_device;     /* +48 if set the LUID is ignored */
} WgAdapterInfo;

/* 60.6.9 preferences (144 B, v1) */
typedef struct WgPreferences {
  WgHeader h;
  uint8_t show_console;         /* +32 */
  uint8_t pad0_[3];
  uint32_t log_level;           /* +36 */
  const wchar_t* const* plugin_paths; /* +40 */
  uint32_t plugin_path_count;   /* +48 */
  uint32_t pad1_;
  const wchar_t* log_path;      /* +56 */
  void* alloc_cb;               /* +64 */
  void* release_cb;             /* +72 */
  void* log_cb;                 /* +80 */
  uint64_t flags;               /* +88 */
  const uint32_t* features;     /* +96 */
  uint32_t feature_count;       /* +104 */
  uint32_t application_id;      /* +108 */
  uint32_t engine;              /* +112 */
  uint32_t pad2_;
  const char* engine_version;   /* +120 */
  const char* project_id;       /* +128 */
  uint32_t render_api;          /* +136 0 D3D11, 1 D3D12, 2 Vulkan */
  uint32_t pad3_;
} WgPreferences;
#define WG_PREF_FLAG_NO_STATE_TRACKING (1ull << 0)
#define WG_PREF_FLAG_FRAME_TAGGING (1ull << 7) /* 2.14.1 only */

/* 60.2 version encoding passed to <init>: major<<48 | minor<<32 | patch<<16 | magic (A9.1: 0xfedc, data). */
#define WG_API_VERSION_MAJOR(v) ((uint32_t)(((uint64_t)(v) >> 48) & 0xffffu))
#define WG_API_VERSION_MINOR(v) ((uint32_t)(((uint64_t)(v) >> 32) & 0xffffu))
#define WG_API_VERSION_PATCH(v) ((uint32_t)(((uint64_t)(v) >> 16) & 0xffffu))
#define WG_API_VERSION_MAGIC(v) ((uint32_t)((uint64_t)(v) & 0xffffu))

#ifdef __cplusplus
} /* extern "C" */

static_assert(sizeof(WgHeader) == 32 && offsetof(WgHeader, type) == 8 && offsetof(WgHeader, version) == 24, "60.2");
static_assert(sizeof(WgFrameToken) == 40 && offsetof(WgFrameToken, header) == 8, "60.2 frame token");
static_assert(sizeof(WgViewport) == 40 && offsetof(WgViewport, id) == 32, "60.6.1");
static_assert(sizeof(WgExtent) == 16 && offsetof(WgExtent, width) == 8, "60.6.2");
static_assert(sizeof(WgResource) == 112 && offsetof(WgResource, kind) == 32 && offsetof(WgResource, native) == 40 &&
              offsetof(WgResource, state) == 64 && offsetof(WgResource, native_format) == 76 &&
              offsetof(WgResource, gpu_va) == 88 && offsetof(WgResource, tail) == 104, "60.6.3 resource");
static_assert(sizeof(WgTag) == 64 && offsetof(WgTag, buffer_type) == 40 && offsetof(WgTag, extent) == 48, "60.6.3 tag");
static_assert(sizeof(WgTagConversion) == 48 && offsetof(WgTagConversion, scale) == 40, "60.6.3 tag extension");
static_assert(sizeof(WgConstants) == 456 && offsetof(WgConstants, jitter) == 352 &&
              offsetof(WgConstants, camera_pos) == 376 && offsetof(WgConstants, near_plane) == 424 &&
              offsetof(WgConstants, mv_invalid_value) == 440 && offsetof(WgConstants, depth_inverted) == 444 &&
              offsetof(WgConstants, mv_jittered) == 450 && offsetof(WgConstants, min_depth_separation) == 452,
              "60.6.4");
static_assert(sizeof(WgRrOptions) == 224 && offsetof(WgRrOptions, sharpness) == 44 &&
              offsetof(WgRrOptions, colour_is_hdr) == 56 && offsetof(WgRrOptions, normal_roughness_packing) == 60 &&
              offsetof(WgRrOptions, world_to_view) == 64 && offsetof(WgRrOptions, view_to_world) == 128 &&
              offsetof(WgRrOptions, alpha_upscaling) == 192 && offsetof(WgRrOptions, presets) == 196, "60.6.5");
static_assert(sizeof(WgRrOptimal) == 64 && offsetof(WgRrOptimal, min_width) == 44 &&
              offsetof(WgRrOptimal, max_height) == 56, "60.6.6");
static_assert(sizeof(WgRrState) == 40 && offsetof(WgRrState, estimated_vram) == 32, "60.6.6 state");
static_assert(sizeof(WgRequirements) == 184 && offsetof(WgRequirements, required_tags) == 48 &&
              offsetof(WgRequirements, os_detected) == 56 && offsetof(WgRequirements, driver_required) == 92 &&
              offsetof(WgRequirements, vulkan_only) == 104, "60.6.7");
static_assert(sizeof(WgFeatureVersion) == 56 && offsetof(WgFeatureVersion, runtime_version) == 44, "60.6.7 version");
static_assert(sizeof(WgAdapterInfo) == 56 && offsetof(WgAdapterInfo, luid_size) == 40 &&
              offsetof(WgAdapterInfo, vk_physical_device) == 48, "60.6.8");
static_assert(sizeof(WgPreferences) == 144 && offsetof(WgPreferences, log_level) == 36 &&
              offsetof(WgPreferences, plugin_path_count) == 48 && offsetof(WgPreferences, flags) == 88 &&
              offsetof(WgPreferences, features) == 96 && offsetof(WgPreferences, feature_count) == 104 &&
              offsetof(WgPreferences, engine_version) == 120 && offsetof(WgPreferences, render_api) == 136,
              "60.6.9");
#endif

#endif
