// SPDX-License-Identifier: MIT
/* WrongGPU RR runtime adapter, version 1: the owned boundary between the proxy (pages 60-63) and WrongGPU's
 * preset-F runtime (pages 50-58). Entirely WrongGPU's own design; nothing here is a game interface.
 *
 * Split of responsibilities:
 *  - The proxy owns the game interface: viewport, options, creation decisions (61.3), input lookup and
 *    substitutions (61.4, 62.2), every rejection before a pass (61.4 steps 1-4 and 9, 62.3, 62.5), the result
 *    codes, and the host bookkeeping of page 53 (par, fc, pj, CP, pending reset, render size last built for),
 *    ported from clean/rr_f/reference/lifecycle.py. It hands the runtime a fully decided frame.
 *  - The runtime owns GPU work: the persistent images of 53.1, their zero-filled (re)creation, the passes of
 *    53.2, resource-state transitions on the game's command list (61.4 step 7; exact states: open item 16),
 *    and reports how far a frame got (jitter consumption point of 53.10, camera fallback of 62.4).
 *
 * The runtime library is found by the proxy in its own directory under WG_RR_RUNTIME_FILE and exports
 * WG_RR_RUNTIME_EXPORT returning a pointer to a WgRrRuntimeV1 that stays valid until the library unloads.
 * All functions are called from the game's threads; evaluate/create/destroy are never called concurrently
 * for one instance (the proxy serialises RR calls).
 */
#ifndef WG_RR_ADAPTER_H
#define WG_RR_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WG_RR_ADAPTER_VERSION 1u
/* release 0.6.0 (owner 2026-11-02: one runtime DLL): the RR runtime is part of WrongGPU's runtime library, beside the
 * Model M interface (before: wronggpu_rr_runtime.dll of its own) */
#define WG_RR_RUNTIME_FILE L"wronggpu_runtime.dll"
#define WG_RR_RUNTIME_EXPORT "WrongGPU_RrRuntimeV1"

/* Runtime status bits (match the settings-contract status word bits 0 and 1, 60.9). */
#define WG_RR_RT_READY 1u        /* runtime initialised on a supported GPU */
#define WG_RR_RT_MODEL_LOADED 2u /* preset-F model admitted and loaded */
#define WG_RR_RT_PREPARING 4u    /* release 0.6.0: the frame kernels are still being created (frames pass through) */

/* Format classes (62.5). */
enum WgRrFormatClass { WG_FMT_UNSUPPORTED = 0, WG_FMT_FLOAT = 1, WG_FMT_UNORM = 2, WG_FMT_SNORM = 3, WG_FMT_SRGB = 4,
                       WG_FMT_DEPTH_FLOAT = 5, WG_FMT_DEPTH_UNORM = 6 };

/* Input / output roles of one frame, after the proxy's routing of 62.2. */
enum WgRrRole {
  WG_ROLE_COLOUR = 0,          /* type 3 */
  WG_ROLE_OUTPUT,              /* type 4 (written) */
  WG_ROLE_DEPTH,               /* type 49 if tagged, else type 0; see WgRrFrame.depth_is_linear */
  WG_ROLE_MOTION,              /* type 1 */
  WG_ROLE_DIFFUSE_ALBEDO,      /* type 7 */
  WG_ROLE_SPECULAR_ALBEDO,     /* type 8 */
  WG_ROLE_NORMALS,             /* packing 0: type 5; packing 1: type 14 */
  WG_ROLE_ROUGHNESS,           /* packing 0: type 6; absent with packing 1 */
  WG_ROLE_SPECULAR_MOTION,     /* type 10 */
  WG_ROLE_SPECULAR_HIT_DIST,   /* type 42 (channel 0, own origin) */
  WG_ROLE_DISOCCLUSION,        /* type 11 */
  WG_ROLE_G1,                  /* type 41 when attached, else type 58 (own origin handed on; model uses C's) */
  WG_ROLE_DOF_GUIDE,           /* type 64 */
  WG_ROLE_BEFORE_TRANSPARENCY, /* type 40 */
  WG_ROLE_LAYER,               /* type 39 when attached, else type 51; origin: WgRrFrame.layer_origin */
  WG_ROLE_OPACITY,             /* type 52, only handed on together with a layer */
  WG_ROLE_INPUT_ALPHA,         /* type 34, 2.14.1 with alpha upscaling only */
  WG_ROLE_OUTPUT_ALPHA,        /* type 67 (written), 2.14.1 with alpha upscaling only */
  WG_ROLE_COUNT
};

typedef struct WgRrImage {
  void* resource;          /* native resource (ID3D12Resource*); a proxy-owned copy for lifetime-0 tags */
  uint32_t buffer_type;    /* the public buffer type that supplied it */
  uint32_t format;         /* DXGI format number (description, or the native resource's when that is 0) */
  uint32_t format_class;   /* WgRrFormatClass as accepted by 62.5 */
  uint32_t state;          /* declared D3D12 state bits the resource is in; 0xffffffff = not given */
  uint32_t width, height;  /* resource size */
  uint32_t origin_x;       /* extent left (61.4 step 3) */
  uint32_t origin_y;       /* extent top */
  uint32_t extent_width;   /* 0 = whole resource */
  uint32_t extent_height;
  uint32_t lifetime;       /* WgTagLifetime as tagged */
  uint32_t conversion;     /* 0 none, 1 linear: high = low * scale + bias (60.6.3 extension, WrongGPU's choice) */
  float conversion_scale;
  float conversion_bias;
} WgRrImage;

typedef struct WgRrCreate {
  uint32_t size;               /* sizeof(WgRrCreate) */
  uint32_t api_minor;          /* 7 or 14: the Streamline variant the proxy runs as */
  void* device;                /* native device recorded at <set-device> */
  uint32_t output_width, output_height;
  uint32_t render_width, render_height; /* creation-time render size; 0 x 0 when unknown (open item 13) */
  uint32_t mode;
  uint32_t packing;            /* 0 separate normals / roughness, 1 packed */
  uint32_t presets[6];         /* accepted and ignored (60.6.5) */
  /* creation decisions (61.3), each 0/1 */
  uint32_t colour_hdr;
  uint32_t depth_inverted;
  uint32_t mv_jittered;
  uint32_t mv_output_resolution; /* always 0 for a created instance: such configurations are rejected (62.3) */
  uint32_t hw_depth;           /* 1 when linear depth was not tagged at creation */
  uint32_t alpha_upscaling;
  uint32_t sharpen_request;    /* recorded, not used: preset F has no sharpening pass */
  uint32_t latched_settings;   /* settings with flag bit 0, latched at creation (60.9) */
} WgRrCreate;

/* Settings snapshot bits handed to the runtime per frame (60.9). */
#define WG_RR_SET_SPECULAR_MOTION_FROM_V (1u << 0)
#define WG_RR_SET_BOUNDED_STORES (1u << 1)
/* WrongGPU variation (owner 2026-10-31, opt-in): the second-vector reconstruction (52.7.2) reads the specular hit
   distance as the median of its aligned 1x4 pixel run, so lone outliers (missed reflection rays reporting the game's
   maximum) do not reproject single pixels' reflections far away (speckles in motion). Off = native behaviour. */
#define WG_RR_SET_HIT_DISTANCE_FILTER (1u << 2)

typedef struct WgRrFrame {
  uint32_t size;               /* sizeof(WgRrFrame) */
  uint32_t api_minor;
  void* command_list;          /* native command list (unwrapped) */
  uint32_t render_width, render_height;   /* one render size per call (61.4 step 3) */
  uint32_t output_width, output_height;
  float jitter[2];             /* current jitter, pixels, unchanged sign (62.3) */
  float prev_jitter[2];        /* pj (53.5, 53.10) */
  float mv_scale[2];           /* ms = (scale.x * Wr, scale.y * Hr), f32 (61.4 step 4, 52.8) */
  uint32_t reset;              /* the combined reset (63.3) */
  uint32_t parity;             /* par */
  uint32_t counter;            /* fc */
  uint32_t rebuild_render_items; /* render-size change detected: re-create HID, PM, Pe, Pn, head images, CH pair
                                    zero-filled at this size before the frame (53.9) */
  uint32_t depth_is_linear;    /* this frame's depth image is type 49 */
  uint32_t settings;           /* WG_RR_SET_* snapshot of this evaluate */
  float view_to_clip[16];      /* current V2C (constants +32) */
  float world_to_view[16];     /* current W2V (options +64) */
  float prev_view_to_clip[16]; /* CP as used by this call: the current pair on a reset frame (62.4) */
  float prev_world_to_view[16];
  uint32_t attached;           /* bit per WgRrRole */
  uint32_t layer_origin_x, layer_origin_y; /* type 51's origin, (0, 0) when 51 is absent (62.2, 54b.3) */
  WgRrImage images[WG_ROLE_COUNT];
} WgRrFrame;

typedef struct WgRrFrameResult {
  uint32_t size;                  /* sizeof(WgRrFrameResult) */
  uint32_t passes_started;        /* 0: failed before any pass */
  uint32_t input_jitter_consumed; /* the input stage used pj (53.10): pj is replaced even if the call fails */
  uint32_t camera_fallback_used;  /* unusable camera matrices: SV = V for this frame (62.4) */
} WgRrFrameResult;

typedef struct WgRrRuntimeV1 {
  uint32_t size;      /* sizeof(WgRrRuntimeV1) */
  uint32_t version;   /* WG_RR_ADAPTER_VERSION */
  void* ctx;
  uint32_t (*status)(void* ctx);                                      /* WG_RR_RT_* bits */
  uint32_t (*adapter_supported)(void* ctx, const void* luid, uint32_t luid_size); /* 1 = supported GPU */
  void (*release_version)(void* ctx, uint32_t out[3]);                /* WrongGPU's published release version */
  uint32_t (*set_device)(void* ctx, void* native_device);             /* result code */
  uint32_t (*create_instance)(void* ctx, const WgRrCreate* desc, void** out_instance); /* result code */
  void (*destroy_instance)(void* ctx, void* instance);  /* must defer GPU destruction past frames in flight */
  uint64_t (*instance_vram)(void* ctx, void* instance); /* own allocations of the instance (60.8 <rr-state>) */
  uint32_t (*evaluate)(void* ctx, void* instance, const WgRrFrame* frame, WgRrFrameResult* result);
  /* Lifetime-0 inputs (62.1): copy during the tagging call on its command list; returns a runtime-owned
     resource that stays valid until release_transient (which must defer past frames in flight). */
  uint32_t (*retain_transient)(void* ctx, void* native_command_list, void* native_resource, uint32_t format,
                               void** out_copy);
  void (*release_transient)(void* ctx, void* copy);
  /* Format / sample count of a native resource (62.5: description format 0 and the multisample check). */
  uint32_t (*describe_resource)(void* ctx, void* native_resource, uint32_t* format, uint32_t* sample_count);
  /* ---- appended in clean/rr-f-live1 (still version 1; present when size >= WG_RR_RUNTIME_V1_LIVE_SIZE) ---- */
  /* Mip-0 width / height of a native resource: games that fill no size in the resource description (0 x 0). */
  uint32_t (*describe_resource_size)(void* ctx, void* native_resource, uint32_t* width, uint32_t* height);
  /* Fail-safe (WrongGPU's choice, not preset F): record a bilinear copy of the colour's render rectangle into the
     output rectangle on frame->command_list (declared states honoured and restored), so a frame whose evaluate failed
     stays presentable. Needs no instance; returns 0 when recorded. */
  uint32_t (*passthrough)(void* ctx, const WgRrFrame* frame);
  /* ---- appended in clean/i20-rr-panel (still version 1; present when size >= WG_RR_RUNTIME_V1_TIME_SIZE) ---- */
  /* GPU time in ms of the newest completed frame of the newest live instance, smoothed over about 30 frames (the
     frame's first to last timestamp); 0 when no instance has completed a frame. Any thread, never waits. */
  float (*frame_gpu_ms)(void* ctx);
  /* ---- appended in the 0.6.0 follow-up (bug hunt C; still version 1; present when size >= WG_RR_RUNTIME_V1_STATE_SIZE) ---- */
  /* retain_transient with the declared D3D12 state of the native resource (0xffffffff = not given: as retain_transient,
     which relies on implicit promotion from COMMON). A given state is moved to COPY_SOURCE for the copy and back. */
  uint32_t (*retain_transient_state)(void* ctx, void* native_command_list, void* native_resource, uint32_t format,
                                     uint32_t state, void** out_copy);
} WgRrRuntimeV1;

/* size of a v1 table without the live additions / with them / with the frame time / with retain_transient_state */
#define WG_RR_RUNTIME_V1_BASE_SIZE ((uint32_t)offsetof(WgRrRuntimeV1, describe_resource_size))
#define WG_RR_RUNTIME_V1_LIVE_SIZE ((uint32_t)offsetof(WgRrRuntimeV1, frame_gpu_ms))
#define WG_RR_RUNTIME_V1_TIME_SIZE ((uint32_t)offsetof(WgRrRuntimeV1, retain_transient_state))
#define WG_RR_RUNTIME_V1_STATE_SIZE ((uint32_t)sizeof(WgRrRuntimeV1))

typedef const WgRrRuntimeV1* (*WgRrRuntimeEntryFn)(void);

#ifdef __cplusplus
}
#endif

#endif
