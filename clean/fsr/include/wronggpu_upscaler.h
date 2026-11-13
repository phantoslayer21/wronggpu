// SPDX-License-Identifier: MIT
/*
 * WrongGPU upscaler interface (spec 20 "Boundary to the upscaler", spec 10 section 1).
 *
 * The narrow C boundary between the FidelityFX front end (amd_fidelityfx_dx12.dll replacement) and an upscaler
 * implementation. Two implementations exist or are planned:
 *   - the stand-in D3D12 compute upscaler built into the front end (clean/fsr/src/standin_d3d12.cpp);
 *   - the WrongGPU network runtime (spec 10, branch clean/runtime-core), shipped as a separate DLL that exports
 *     wgGetUpscalerBackend (see the end of this file).
 *
 * Every value in WgFrame is already converted to the network's conventions (spec 20 table, spec 01, spec 03);
 * the conversions from FidelityFX are listed in clean/FORMATS.md, section "FSR 3.1 to network conventions".
 * The header is plain C and needs no D3D12 header: D3D12 objects are passed as void*, formats as DXGI_FORMAT
 * values and resource states as D3D12_RESOURCE_STATES bit masks (both as uint32_t).
 *
 * Version 2 (QUESTIONS.md / ANSWERS.md spec 20 item 16) adds same-frame execution for a backend that runs its work
 * outside D3D12 (HIP): the front end can split the game's command list at the dispatch point ("ordered split", see
 * WgFrontEndServices). New fields are appended to the structures; struct_size tells which fields exist. The front end
 * asks a backend DLL for version 2 first and falls back to version 1; a version 1 backend gets no services.
 */
#ifndef WRONGGPU_UPSCALER_H
#define WRONGGPU_UPSCALER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WG_UPSCALER_INTERFACE_VERSION 2u
#define WG_UPSCALER_INTERFACE_VERSION_MIN 1u /* oldest version the front end still accepts */

/* Return codes. Any non-zero result from create or dispatch makes the front end hand the context to AMD's FSR
 * (spec 10 section 8). */
typedef enum WgResult {
    WG_OK = 0,
    WG_ERROR = 1,             /* unspecified failure, details already logged */
    WG_ERROR_UNSUPPORTED = 2, /* this input combination is not supported (format, size, flag) */
    WG_ERROR_DEVICE = 3,      /* the D3D12 (or HIP) runtime returned an error */
    WG_ERROR_MEMORY = 4,
    WG_ERROR_PARAMETER = 5,   /* a required input is missing or invalid */
    WG_ERROR_NOT_READY = 6,   /* v2: not possible this frame, try again next frame (e.g. split of a command list the
                                 front end has not tracked since its last Reset); from dispatch it makes the front end
                                 use AMD's FSR (or the stand-in) for this frame only, the context stays the backend's */
    WG_ERROR_INPUT = 7        /* v2 (0.6.0 follow-up): the game's inputs can never be taken (a texture's shape, format
                                 or flags the upscaler refuses), the same on every frame: from dispatch the context is
                                 failed as for WG_ERROR, and in OptiScaler's custom slot the front end remembers it for
                                 the session (no new context is made). A front end that does not know the code treats
                                 it as WG_ERROR. */
} WgResult;

/* Creation flags (spec 10 section 1 "create"). */
#define WG_CREATE_HDR                  (1u << 0) /* color is linear HDR; otherwise linear SDR in [0, 1] */
#define WG_CREATE_DEPTH_INVERTED       (1u << 1) /* near = 1, far = 0 */
#define WG_CREATE_AUTO_EXPOSURE        (1u << 2) /* "compute it" (spec 07) when no exposure texture is given */
#define WG_CREATE_DEPTH_INFINITE       (1u << 3) /* informational: infinite far plane */
#define WG_CREATE_MOTION_OUTPUT_RES    (1u << 4) /* motion texture is at output size O, not render size (FORMATS.md) */
#define WG_CREATE_NONLINEAR_COLOR      (1u << 5) /* informational: game says color is perceptual, not linear */
#define WG_CREATE_DYNAMIC_RESOLUTION   (1u << 6) /* informational: render size R may change every frame */

/* Log levels for WgLogFn. */
#define WG_LOG_INFO  0
#define WG_LOG_WARN  1
#define WG_LOG_ERROR 2

/* Log callback provided by the front end; the message is one line without a newline. */
typedef void (*WgLogFn)(void* user, int level, const char* message);

typedef struct WgCreateInfo {
    uint32_t struct_size;            /* sizeof(WgCreateInfo) */
    uint32_t interface_version;      /* WG_UPSCALER_INTERFACE_VERSION */
    void*    d3d12_device;           /* ID3D12Device* of the game; the upscaler AddRefs it if it keeps it */
    uint32_t max_render_width;       /* maximum render size; every frame has R <= this */
    uint32_t max_render_height;
    uint32_t output_width;           /* output size O; a change of O means a new context */
    uint32_t output_height;
    uint32_t flags;                  /* WG_CREATE_* */
    WgLogFn  log;                    /* never null */
    void*    log_user;
    const char* options;             /* front-end configuration as "key=value;key=value", may be empty, never null */
    /* version 2: */
    const struct WgFrontEndServices* services; /* null for a v1 backend, or when the front end cannot split (d3d12_hooks
                                                  = off, hooks not installable); valid until destroy */
} WgCreateInfo;

/* One D3D12 texture handed in by the game. */
typedef struct WgTexture {
    void*    resource;  /* ID3D12Resource*, or null when the input is absent */
    uint32_t format;    /* DXGI_FORMAT to view it with (typed; the front end resolves typeless formats) */
    uint32_t width;     /* size of the whole texture (the active region may be smaller) */
    uint32_t height;
    uint32_t state;     /* D3D12_RESOURCE_STATES the game left it in; must be the same after the call */
} WgTexture;

/* Exposure source for this frame (spec 10 section 5 step 1). */
#define WG_EXPOSURE_VALUE   0u /* use exposure_value (1.0 when the game gives nothing and auto-exposure is off) */
#define WG_EXPOSURE_TEXTURE 1u /* read texel (0, 0), channel R, of the exposure texture */
#define WG_EXPOSURE_COMPUTE 2u /* compute it (spec 07) from this frame's color and pre_exposure */

/* Per-frame inputs (spec 20 table "What WrongGPU needs from each upscale dispatch"). */
typedef struct WgFrame {
    uint32_t struct_size;            /* sizeof(WgFrame) */
    void*    command_list;           /* ID3D12GraphicsCommandList* of the game; record all work into it */

    WgTexture color;                 /* render-resolution color, linear (see WG_CREATE_HDR) */
    WgTexture depth;                 /* render-resolution depth, one channel (see WG_CREATE_DEPTH_INVERTED) */
    WgTexture motion;                /* two channels, raw game values; render size, or O with WG_CREATE_MOTION_OUTPUT_RES */
    WgTexture exposure;              /* 1x1, used when exposure_mode == WG_EXPOSURE_TEXTURE */
    WgTexture output;                /* output-resolution texture the result is written to (UAV-capable) */

    /* Motion in render pixels: m = motion_scale * raw + motion_offset, per axis, sampled at the render pixel
     * (x, y) (or at the output texel under its center when the motion texture is at output size). Convention:
     * previous_position = current_position + m, x to the right, y downwards, in render pixels. */
    float    motion_scale[2];
    float    motion_offset[2];

    /* Image shift caused by this frame's jitter (spec 20 draft 2, spec 03), render pixels, x right, y down: render
     * pixel (x, y) shows the scene at the unjittered position (x + 1/2 - jx, y + 1/2 - jy). */
    float    jitter[2];

    /* Exposure in AMD's sense: it applies to color / pre_exposure (spec 20 draft 2). The network runtime divides the
     * color by pre_exposure and multiplies its output by it (spec 10 draft 2, section 5). */
    uint32_t exposure_mode;          /* WG_EXPOSURE_* */
    float    exposure_value;         /* for WG_EXPOSURE_VALUE */
    float    pre_exposure;           /* pi of spec 07; > 0 (the front end replaces 0, negative or non-finite by 1) */

    uint32_t reset;                  /* 1: discard history (camera cut); the upscaler also resets on its own rules.
                                        2: discard history and start over as a newly created upscaler would (frame
                                        counters, exposure state): the front end gave this upscaler, kept from a
                                        destroyed context, to a new context with the same create parameters, and this
                                        is the first frame it takes for it. An upscaler that does not know 2 treats
                                        every non-zero value as 1. */

    uint32_t render_x;               /* active region inside the render textures: origin ... */
    uint32_t render_y;
    uint32_t render_width;           /* ... and this frame's render size R (<= maximum) */
    uint32_t render_height;
    uint32_t output_width;           /* region of the output texture to fill, from (0, 0); equals O of create */
    uint32_t output_height;

    float    frame_time_ms;          /* informational */
    uint64_t frame_index;            /* front end's count of dispatches on this context, from 0 */

    /* version 2: */
    void*    command_queue;          /* ID3D12CommandQueue* this command list was last executed on (learned by the front
                                        end's ExecuteCommandLists hook), or null; informational: the split needs no
                                        queue from the backend */
    uint32_t command_list_type;      /* D3D12_COMMAND_LIST_TYPE of command_list (0 = DIRECT, 2 = COMPUTE) */
} WgFrame;

/* Version 2: same-frame execution through an "ordered split" of the game's command list.
 *
 * The backend, inside dispatch, on the game's recording thread:
 *   1. records into frame->command_list whatever must happen before its outside work (copies of the inputs into its
 *      interop textures, barriers);
 *   2. calls services->split_command_list(services->context, frame, &request) with two fences it created on the
 *      game's device and owns (D3D12_FENCE_FLAG_SHARED, imported into HIP by the backend);
 *   3. enqueues its outside work: wait until done_fence >= done_value, run, then make resume_fence = resume_value;
 *   4. records into frame->command_list (the same pointer, open again) whatever follows (copy of the result into
 *      the output, barriers back to the game's states), then returns WG_OK.
 *
 * What the front end guarantees on WG_OK:
 *   - done_fence reaches done_value on the GPU once every command recorded into the list before the call has
 *     completed, and these commands run only after everything the game submits before this list (in its own order
 *     of ExecuteCommandLists calls on its queues) has been submitted ahead of them;
 *   - the commands recorded into the list after the call (the backend's and the game's) run only after resume_fence
 *     reaches resume_value;
 *   - the list is open again with all of the game's bindings restored (descriptor heaps, pipeline state, root
 *     signatures and arguments, render targets, viewports, scissors, input assembler, stencil reference, blend
 *     factor, depth bounds, sample positions, shading rate, view-instance mask, predication);
 *   - if the game does not submit the list within the front end's timeout (setting split_timeout_ms, default 2000),
 *     the front end releases the first half from the CPU, so done_fence is still reached and nothing hangs.
 * The backend must make resume_fence reach resume_value in every case, also when its outside work fails (signal it
 * from the CPU then): the game's queue waits for it. done_value / resume_value must be greater than any value the
 * fence had before. Several splits per dispatch are allowed (each with new values).
 *
 * On an error the list is unchanged and still open: WG_ERROR_NOT_READY (the list's Reset was not seen since the hooks
 * were installed, e.g. the first frame), WG_ERROR_UNSUPPORTED (hooks unavailable, bundle or render pass state the
 * front end cannot restore, command list type other than DIRECT or COMPUTE), WG_ERROR_DEVICE (a D3D12 call failed),
 * WG_ERROR_PARAMETER. The backend then does not do its outside work this frame and returns the error from dispatch.
 */
typedef struct WgSplitRequest {
    uint32_t struct_size;            /* sizeof(WgSplitRequest) */
    void*    done_fence;             /* ID3D12Fence*, signalled to done_value by the GPU after the first half */
    uint64_t done_value;
    void*    resume_fence;           /* ID3D12Fence*, waited for (>= resume_value) before the second half */
    uint64_t resume_value;
} WgSplitRequest;

typedef struct WgFrontEndServices {
    uint32_t struct_size;            /* sizeof(WgFrontEndServices) */
    void*    context;                /* first argument of every service */
    /* Only from within dispatch, with the frame passed to dispatch. */
    WgResult (*split_command_list)(void* context, const WgFrame* frame, const WgSplitRequest* request);
} WgFrontEndServices;

typedef struct WgUpscaler WgUpscaler; /* opaque, owned by the implementation */

/* Backend flags (version 2). */
#define WG_BACKEND_NEEDS_SPLIT (1u << 0) /* the backend calls split_command_list: the front end installs its D3D12
                                            command-list hooks when the backend is loaded (d3d12_hooks = auto) */

/* The implementation's entry points. */
typedef struct WgUpscalerBackend {
    uint32_t struct_size;            /* sizeof(WgUpscalerBackend) */
    uint32_t interface_version;      /* WG_UPSCALER_INTERFACE_VERSION */
    const char* name;                /* shown in logs and returned as the provider name */
    WgResult (*create)(const WgCreateInfo* info, WgUpscaler** out_upscaler);
    WgResult (*dispatch)(WgUpscaler* upscaler, const WgFrame* frame);
    void     (*destroy)(WgUpscaler* upscaler); /* waits for in-flight work, frees everything */
    uint64_t (*gpu_memory_bytes)(WgUpscaler* upscaler); /* may be null */
    /* version 2: */
    uint32_t backend_flags;          /* WG_BACKEND_* */
} WgUpscalerBackend;

/* Exported by an upscaler DLL (the network runtime) as "wgGetUpscalerBackend". Returns null if the requested
 * interface version is not supported. The front end asks for WG_UPSCALER_INTERFACE_VERSION first, then for older
 * versions down to WG_UPSCALER_INTERFACE_VERSION_MIN; the returned interface_version and struct_size must match the
 * version asked for. */
typedef const WgUpscalerBackend* (*PfnWgGetUpscalerBackend)(uint32_t interface_version);

#ifdef __cplusplus
}
#endif

#endif /* WRONGGPU_UPSCALER_H */
