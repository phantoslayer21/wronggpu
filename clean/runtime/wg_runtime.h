// SPDX-License-Identifier: MIT
/* WrongGPU per-frame runtime, core part (spec/10-runtime.md section 0: HIP only, no D3D12).
 *
 * Owns the network's model data and state, and runs one whole frame (exposure, enc0, the ten Swin stages, dec0,
 * post, filter) on one HIP stream with the state rules of spec 10 sections 3-5. Inputs are in runtime-owned GPU
 * memory (wg_rt_upload_inputs is the offline test path; the interop part of spec 10 will fill the same buffers).
 * Buffers and formats: clean/FORMATS.md, "Per-frame runtime". Python front end: clean/runner/pipeline.py.
 */
#ifndef WG_RUNTIME_H
#define WG_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define WG_RT_API __declspec(dllexport)
#else
#define WG_RT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WgRtConfig {
  const char* build_dir;         /* code objects from clean/runner/build.py */
  const char* rocm_dir;          /* HIP SDK directory (for amdhip64); NULL or "" = system search path */
  const char* model_dir;         /* converter output folder: model.wgcm, and raw/<stage>/<name>.bin with --raw */
  int backend;                   /* 0 = HIP, 1 = host emulator (tests), 2 = hybrid D3D12 (Windows: the kernels with a
                                    D3D12 port in <build_dir>/dx12 run on D3D12, the rest on HIP),
                                    3 = native D3D12 (Model M only, unsupported routes fail; no HIP fallback). */
  int max_rw, max_rh;            /* maximum render size */
  int ow, oh;                    /* output size O; D = floor((3 O + 1) / 2) */
  int depth_inverted;            /* near = 1 */
  int auto_exposure;             /* compute the exposure with spec 07 */
  float alpha;                   /* spec 07 key (0 = default 0.18) */
  int swin_threads, gemm_threads, nt, wpw64, wpw96, wpw128;  /* Swin launch geometry; 0 = defaults */
  const char* swin_code_object;  /* NULL = "swin.co" (or a build variant, see clean/runner/variants.py) */
  int d_row[16];                 /* WMMA accumulator layout [h * 8 + v]; all 0 = gfx12 (row = v + 8 h) */
  const char* emu_library;       /* backend 1: NULL = <build_dir>/wg_emu.so */
  const char* map_path;          /* the converter map shipped with the runtime (needed for model.wgcm) */
  int model_format;              /* 0 = model.wgcm if present, else raw/; 1 = raw/ (tests); 2 = model.wgcm only */
  int block_threads[5];          /* lanes per workgroup of swin_block_c64, _c64_add, _c96, _c96_add, _c128 (0 =
                                    swin_threads, else the default 256; clean/runner/variants.py geometry) */
  /* Kernel choice per stage, for A/B timing and quality runs without a rebuild. Each: NULL or "" = the environment
   * variable in brackets, else the default (the fastest exact variant). Swin: swin_code_object above, else
   * [WG_SWIN_VARIANT] (a name in <build_dir>/variants.json, geometry from there), else swin.co.
   * enc0 / dec0: a variant of clean/runner/perf_variants.py ("base" = encoder.co / decoder.co, other names =
   * encoder__<name>.co / decoder__<name>.co from build.py --perf-variants; dec0 "v1" is not supported here; the
   * base builds are the fastest exact ones: enc0 A fragments from global memory, dec0 latent loads late).
   * exposure: "base" (exposure_partial + exposure_final; default), "v1". Format-changing builds (enc0 colorh,
   * motionh, in16; dec0 pack12) are refused.
   * post: a kernel of kernels/post.hip with post_recon's arguments (post_recon, post_recon2*, post_recon3*,
   * post_recon4_*; default: the banded path post_recon2_r_band[_p12] + post_filter_c1_s4_band (post_pack), or
   * post_recon2_r with an unbanded filter or recon_output) or post_recon5*, post_recon6* (run after the pre-pass
   * post_prep, with its record buffer); kernels other than post_recon need D >= about 1.43 R (3* and later also
   * D >= 4 columns, 6* also R < D per axis) and the runtime uses post_recon on frames that do not satisfy it;
   * post_fused* / post_fused3* replace post and filter by one launch (FORMATS.md). filter:
   * post_filter[_c1][_s4|_s16] (default post_filter_c1_s4, banded post_filter_c1_s4_band). wg_rt_kernels() reports
   * the choice. */
  const char* enc0_variant;      /* [WG_ENC0_VARIANT] */
  const char* dec0_variant;      /* [WG_DEC0_VARIANT] */
  const char* exposure_variant;  /* [WG_EXPOSURE_VARIANT] */
  const char* post_kernel;       /* [WG_POST_KERNEL] */
  const char* filter_kernel;     /* [WG_FILTER_KERNEL] */
  int block_wpg[5];              /* windows per workgroup of the five block kernels (grid = ceil(windows / wpg)); 0 =
                                    lanes / 32 x windows per wave. Swin v2 splits a window over waves (variants.py) */
  int swin_force_clamp;          /* 1 = always the clamping Swin block kernels, never the clamp-free <name>_nc ones that
                                    the weights' FLAGS word allows (variant geometry "force_clamp"; Swin v2 builds) */
  const char* swin_stage_variants; /* [WG_SWIN_STAGE_VARIANTS] per-stage Swin build variants, "enc2=smax16,b0=gelu16"
                                    (names in variants.json; other stages keep the global choice) */
  int swin_fuse;                 /* 1 = encoder / decoder blocks with the patch merge / expand fused in (variant
                                    geometry "fuse"; set from variants.json for WG_SWIN_VARIANT) */
  const char* swin_fuse_stages;  /* [WG_SWIN_FUSE_STAGES] encoder / decoder stages run fused from the base build,
                                    "dec1,dec2" or "none"; default "dec1" (with swin.co) */
  int post_wave;                 /* [WG_POST_WAVE] 32 = post / filter kernels from post.co; 0 or 64 = from post_w64.co
                                    (wave64 build, default) when present */
  const char* runtime_options;   /* "graph=1;stream_priority=high" (then [WG_RUNTIME_OPTIONS], whose keys win): graph=1
                                    launches each frame as one HIP graph (stream capture, updated in place),
                                    stream_priority=high creates the stream with the highest priority; both off by
                                    default */
  int recon_output;              /* experimental: reconstruct at D = O (output filter = identity); also turned on by the
                                    environment variable WG_RECON_SIZE=output. 0 = spec 01: D = floor((3 O + 1) / 2) */
  const char* swin_bneck;        /* [WG_SWIN_BNECK] b0 ... b5 in one launch from this bmega* build (swin_bneck6,
                                    bit-identical to base); default "bmegahg" with swin.co when built; "none" = six
                                    block launches */
  const char* dec1_dec0;         /* [WG_DEC1_DEC0] "fused": dec1's block and dec0 in one kernel (swin_block_c64_add_dec0
                                    of dec1_dec0.co, bit-identical; dec1's map not written); default "separate" */
  const char* swin_mid;          /* [WG_SWIN_MID] "1" (= bmegahg) or a bmega* build: enc2 + merge, b0 ... b5,
                                    expand + dec2 in one persistent launch (swin_mid, bit-identical); enc1 and dec1
                                    stay separate; default off */
  const char* post_pack;         /* [WG_POST_PACK] "p12": dec0 writes the 12-byte pack12 coefficients and post reads
                                    them (the _p12 form of the post kernel, e.g. post_recon2_r_band_p12; bit-identical);
                                    frames the _p12 kernels do not take run decoder.co and post_recon. Default "p12"
                                    when decoder__pack12.co is there and post / dec0 / dec1_dec0 are not chosen
                                    otherwise (not with recon_output); "none" = the base formats */
  int hdr;                       /* the colour is linear HDR (front end flag WG_CREATE_HDR): model K's [HDR] variant
                                    (specs 42, 43), else [LDR]; model M does not use it */
} WgRtConfig;

typedef struct WgRtFrame {
  int rw, rh;                    /* this frame's render size R (<= maximum) */
  float mv_scale[2];             /* factor converting raw motion to render pixels */
  float mv_bias[2];              /* per-axis offset added before scaling (spec 03; usually 0) */
  float jitter[2];               /* render pixels */
  float exposure;                /* used as is unless auto-exposure is on (spec 10 section 5.1) */
  float pre_exposure;            /* spec 07 pi */
  int reset;                     /* the game asks for a reset; 2: and the runtime starts over with this frame as newly
                                    created (frame counters, first-frame exposure), models M and K */
} WgRtFrame;

typedef struct WgRtFrameInfo {   /* what the last frame used (spec 10 section 3) */
  int index;                     /* frames since create, counting from 0 */
  int n;                         /* frame counter (dither row) */
  int parity;                    /* f */
  int reset;                     /* effective reset (game, first frame or size change) */
  int rw, rh;
  float jitter_prev[2];
  int offsets[10][2];            /* (ox, oy) of enc1, enc2, b0..b5, dec2, dec1 */
} WgRtFrameInfo;

/* Returns 0 on success; on failure non-zero, *rt = NULL and a plain-words message in err. */
WG_RT_API int wg_rt_create(const WgRtConfig* cfg, void** rt, char* err, int errlen);
WG_RT_API void wg_rt_destroy(void* rt);
WG_RT_API const char* wg_rt_error(void* rt);

/* Test path: copy this frame's inputs (dense [rh][rw] arrays: RGB float32, depth float32, motion float32 x 2). */
WG_RT_API int wg_rt_upload_inputs(void* rt, int rw, int rh, const float* color_rgb, const float* depth,
                                  const float* motion);
/* Test path for motion vectors at output resolution (spec 03 draft 4 "Motion at output resolution", spec 06 section
 * 3; the runtime must have been created with the runtime option motion_outres=original): the motion of the region
 * M = mw x mh (the output size) as dense rows, a texel being two FP16 (wide = 0) or two f32 (wide = 1). Every later
 * frame reads this buffer in the input encoder and the reconstruction, with mv_scale (raw motion -> pixels of M) and
 * b_mv (offset added to the raw motion first) given here; WgRtFrame.mv_scale / mv_bias and the render-size motion of
 * wg_rt_upload_inputs are then not used. mw = 0 switches back. */
WG_RT_API int wg_rt_upload_motion_hi(void* rt, int mw, int mh, int wide, const void* texels, const float* mv_scale,
                                     const float* b_mv);
/* Enqueues one whole frame on the runtime's stream and updates the frame state; does not wait for the GPU. */
WG_RT_API int wg_rt_frame(void* rt, const WgRtFrame* frame);
/* Same, and returns the frame's GPU time in ms (waits). */
WG_RT_API int wg_rt_frame_timed(void* rt, const WgRtFrame* frame, float* ms);
WG_RT_API int wg_rt_sync(void* rt);
WG_RT_API int wg_rt_info(void* rt, WgRtFrameInfo* info);
/* A named buffer of the last frame (names in clean/FORMATS.md): device address and size in bytes. */
WG_RT_API int wg_rt_buffer(void* rt, const char* name, uintptr_t* ptr, size_t* bytes);
WG_RT_API int wg_rt_download(void* rt, uintptr_t ptr, void* dst, size_t bytes);
/* Diagnostics (the frame checks' sentinels): the allocation of a named buffer (as wg_rt_buffer; also before the first
   frame) set to the byte value, from the buffer's start to the allocation's end. */
WG_RT_API int wg_rt_fill(void* rt, const char* name, int value);
/* The reconstruction size D in use (spec 01, or O with recon_output / WG_RECON_SIZE=output). */
WG_RT_API int wg_rt_recon_size(void* rt, int* dw, int* dh);
/* Where the model data came from: "model.wgcm" or "raw". */
WG_RT_API const char* wg_rt_model_source(void* rt);
/* The kernels in use, as "swin=<code object> enc0=<code object> dec0=<code object> exposure=<variant>
 * post=<kernel> filter=<kernel> post_last=<kernel of the last frame>". */
WG_RT_API const char* wg_rt_kernels(void* rt);
/* Notes of create for the caller's log, one per line, or an empty string. At present one: the output filter is
 * "analytic" because the model file has no output filter weights (spec 06 draft 4), or because this build cannot
 * run them. The filter in use is also in wg_rt_kernels() ("output_filter=original|analytic|quality5|quality9
 * output_taps=model|none"); runtime option output_filter=original|analytic|quality5|quality9 selects it (default:
 * original when the model file has the weights, else analytic; original without them fails create). */
WG_RT_API const char* wg_rt_notes(void* rt);
/* The packed weight blob of a stage (host copy), for checks. */
WG_RT_API int wg_rt_stage_blob(void* rt, const char* stage, const void** data, size_t* bytes);

/* Interop test path (the D3D12 interop uses the same C++ entry points with HIP external memory): a runtime whose
 * frames take their inputs and output from external device buffers in the formats of clean/FORMATS.md "Per-frame
 * runtime" (color [rh][rw] 4 x f32, depth f32, motion 2 x f32, out [oh][ow] 3 x f32); it has no input or output
 * buffers of its own (wg_rt_upload_inputs fails). exposure_mode: 0 frame->exposure, 1 the f32 at expo_src,
 * 2 compute (spec 07). */
WG_RT_API int wg_rt_create_external(const WgRtConfig* cfg, void** rt, char* err, int errlen);
WG_RT_API int wg_rt_alloc(void* rt, size_t bytes, uintptr_t* ptr);  /* device memory, freed with the runtime */
WG_RT_API int wg_rt_upload(void* rt, uintptr_t dst, const void* src, size_t bytes);
WG_RT_API int wg_rt_frame_external(void* rt, const WgRtFrame* frame, uintptr_t color, uintptr_t depth,
                                   uintptr_t motion, uintptr_t out, int exposure_mode, uintptr_t expo_src);

/* Interop test path with the copy layouts (options input_copy / output_copy of the D3D12 interop): as
 * wg_rt_frame_external, plus raw_inputs (color RGBA16F, depth R32F, motion motion_channels x FP16 rows of the given
 * pitches in bytes, m = ms * raw + mo) and out_half_pitch > 0 (output as RGBA16F rows of that pitch). */
typedef struct WgRtExtIo {
  uintptr_t color, depth, motion, out, expo_src;
  int exposure_mode;
  int raw_inputs, color_pitch, depth_pitch, motion_pitch, motion_channels;
  float ms[2], mo[2];
  int out_half_pitch;
} WgRtExtIo;
WG_RT_API int wg_rt_frame_io(void* rt, const WgRtFrame* frame, const WgRtExtIo* io);
/* The same with the interop's timing marks (timing_log=1): waits, then ms[0..3] = the GPU times of the four parts the
 * marks bracket, named by wg_rt_timing_labels ("enc0,Swin,dec0,post" for model M, "exposure,input,network,output" for
 * model K). */
WG_RT_API int wg_rt_frame_io_timed(void* rt, const WgRtFrame* frame, const WgRtExtIo* io, float ms[4]);
WG_RT_API const char* wg_rt_timing_labels(void* rt);

#ifdef __cplusplus
}
#endif

#endif /* WG_RUNTIME_H */
