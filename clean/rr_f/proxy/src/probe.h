// SPDX-License-Identifier: MIT
// Probe build of the RR proxy (build flag WG_RR_PROBE, output wg_rr_proxy_probe.dll; never in the release build).
//
// Purpose: one run in a game answers the probe questions of spec 63.6 (open items 3, 10, 13, 15) with names and
// numbers. The probe claims RR support as 60.8 says, answers <rr-optimal> with the real 61.5 rule, and does not
// fail an RR evaluate: it writes a plain bilinear upscale of the colour input into the output and returns 0. It is
// read-only on every other game resource (resource-state transitions only, restored). Every other call is forwarded
// unchanged. What it logs is listed in RrProbe below; the log rules are in probe_log.h.
//
// Platform-free: the D3D12 work goes through ProbeGpu (probe_gpu_d3d12.cpp in the DLL, a fake in the tests).
#pragma once
#include <stdint.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../include/rr_proxy_abi.h"
#include "probe_log.h"
#include "proxy_core.h"

namespace wgrr {

// A rectangle of a native texture as the probe reads or writes it.
struct ProbeView {
  void* resource = nullptr;  // native resource (ID3D12Resource*)
  uint32_t format = 0;       // the tag's DXGI format (0: the resource's)
  uint32_t state = 0;        // declared D3D12 state bits the resource is in (60.6.3), restored afterwards
  uint32_t x = 0, y = 0, w = 0, h = 0;
};

// Description of a native texture (D3D12 resource description), numbers only.
struct ProbeTexDesc {
  uint32_t dimension = 0, format = 0, width = 0, height = 0, depth_or_array = 0, mips = 0, samples = 0, flags = 0;
};

// ---- capture option (probe build only; off unless the probe settings file says capture=1) -------------------------
// For a window of RR evaluates the probe copies every tagged input (not the outputs) into readback memory on the
// evaluate's command list (copies and resource-state transitions only, every input back in its declared state;
// nothing is bound on the list), reads the copies back a few evaluates later once a completion marker written after
// the copies is seen (no fence on the game's queue), and writes one folder per evaluate (CAPTURE.md).
struct CaptureSettings {
  bool enabled = false;
  uint32_t start = 300;                // RR evaluate number (1-based) of the first captured evaluate
  uint32_t count = 8;                  // evaluates captured
  uint32_t every = 1;                  // every n-th evaluate from `start`
  uint32_t read_delay = 2;             // evaluates between the copy and its first readback attempt
  uint32_t give_up = 240;              // evaluates after which a copy whose marker never arrived is abandoned
  uint64_t max_bytes = 2048ull << 20;  // total readback bytes of all captured evaluates (capture_max_mb)
  bool on_file = false;                // capture_on_file=1: the window starts at the first evaluate after the trigger file
                                       // appears (checked every 16 evaluates; the file is removed when seen)
  uint32_t shots = 1;                  // capture_shots=N (with capture_on_file): up to N windows, each started by a new trigger file
  std::string trigger = "capture-now.txt";  // capture_trigger=<path>: the trigger file
  std::string folder;                  // capture_dir=<path>: capture folder instead of <game exe dir>\wronggpu-rr-capture
};
// Parses the probe settings file (wronggpu-rr-probe.ini): "key=value" lines, ';' / '#' comments, [sections] ignored.
// Keys: capture, capture_start, capture_count, capture_every, capture_max_mb, capture_read_delay. Unknown keys and
// bad values are reported in `notes` (one line each) and leave the default.
CaptureSettings parse_capture_settings(const std::string& text, std::string* notes);

// One input as copied: subresource 0 of the resource, rows of `row_pitch` bytes (tight: no padding).
struct CaptureImage {
  uint32_t type = 0;                   // buffer type of the tag
  uint32_t format = 0;                 // DXGI format of the resource description
  uint32_t width = 0, height = 0;      // resource description size (mip 0)
  uint32_t row_pitch = 0;              // bytes per row in `data`
  std::vector<uint8_t> data;           // height x row_pitch bytes
};

// An input to copy: buffer type and the tag's view (resource, tag format, declared state, rectangle).
struct CaptureSource {
  uint32_t type = 0;
  ProbeView view;
};

// Where a capture goes. The DLL: next to the game executable (probe settings file and wronggpu-rr-capture\);
// the tests: a fake. Writes may complete asynchronously; flush() waits for them.
class ProbeCaptureHost {
 public:
  virtual ~ProbeCaptureHost() {}
  // Text of the probe settings file; false when there is none.
  virtual bool read_settings(std::string* text) = 0;
  // Prepares the capture folder (once per process; an existing folder of an earlier run is renamed, never deleted).
  // `where` names it for the log. False: capturing is not possible.
  virtual bool open(std::string* where) = 0;
  // Optional (capture_dir): the capture folder to use instead of the default one, called before open().
  virtual void set_folder(const std::string& path) { (void)path; }
  // Writes `data` to <capture folder>\<rel_path> ('/' separated; folders created). False: refused.
  virtual bool write(const std::string& rel_path, std::vector<uint8_t>&& data) = 0;
  // Waits until every queued write has finished; returns the number of failed writes so far.
  virtual uint64_t flush() = 0;
};

// GPU side of the probe. All work is recorded on the game's command list (the evaluate's list, A10.2; the tagging
// call's list for lifetime-0 copies, 62.1); resource states are transitioned from the declared state and back.
class ProbeGpu {
 public:
  virtual ~ProbeGpu() {}
  virtual void set_device(void* native_device) = 0;
  virtual bool has_device() const = 0;
  virtual bool describe(void* resource, ProbeTexDesc* out) = 0;
  // Bilinear upscale of src's rectangle into dst's rectangle. 0, or a result code with `why` for the log.
  virtual WgResultCode upscale(void* native_cmd, const ProbeView& src, const ProbeView& dst, std::string* why) = 0;
  // Mean of the 16 x 16 texels at (px, py) of `src` (clamped into src's rectangle) into readback slot `slot`,
  // tagged with `marker`. read_patch returns true once the GPU has written that slot with that marker.
  virtual WgResultCode patch_mean(void* native_cmd, const ProbeView& src, uint32_t px, uint32_t py, uint32_t slot,
                                  uint32_t marker, std::string* why) = 0;
  virtual bool read_patch(uint32_t slot, uint32_t marker, float mean[4]) = 0;
  // Copy of a lifetime-0 input during its tagging call (62.1). Returns the probe's copy (valid until the next copy
  // for `key`) and the state the copy is in, or null with `why`.
  virtual void* copy_input(void* native_cmd, const ProbeView& src, uint32_t key, uint32_t* copy_state,
                           std::string* why) = 0;

  // ---- capture option (defaults: not supported) ----
  virtual ProbeCaptureHost* capture_host() { return nullptr; }
  // Records copies of subresource 0 of every source into readback memory of capture `id` on the game's list (each
  // input transitioned from its declared state to a copy source and back; copies and barriers only), followed by
  // `marker` once the copies have completed. Refuses (no work recorded) when the readback memory would exceed
  // `budget` bytes. Returns 0 and the bytes reserved, with one line per source that was not copied in `skipped`.
  virtual WgResultCode capture_copy(void* native_cmd, uint32_t id, uint32_t marker,
                                    const std::vector<CaptureSource>& sources, uint64_t budget, uint64_t* bytes,
                                    std::vector<std::string>* skipped, std::string* why) {
    (void)native_cmd; (void)id; (void)marker; (void)sources; (void)budget; (void)bytes; (void)skipped;
    *why = "capture not supported by this GPU side";
    return WG_E_COMPUTE_FAILED;
  }
  // 1: the marker of capture `id` was seen: `out` holds the copied inputs (in the order of the sources copied) and
  // the readback memory is released; 0: not yet; -1: unknown capture.
  virtual int capture_read(uint32_t id, uint32_t marker, std::vector<CaptureImage>* out) {
    (void)id; (void)marker; (void)out;
    return -1;
  }
  // The marker never arrived: the readback memory is kept (the GPU may still write it) until the device changes.
  virtual void capture_abandon(uint32_t id) { (void)id; }
};

struct ProbeConfig {
  uint32_t api_minor = 14;
  BindingIds ids;
  WgGuid frame_token{};             // A9.6 frame-token GUID, checked at +16 of the first token (A10.1)
  bool has_frame_token = false;
  uint32_t release_version[3] = {0, 0, 0};
  std::string fn_optimal, fn_state, fn_options;
  std::vector<std::pair<std::string, WgGuid>> guid_names;  // A9.6 names for the log
  // Normal patch samples: 30 evaluates, the first at RR evaluate number `patch_first`, then every `patch_every`;
  // a sample is read back `patch_read_delay` evaluates later (no fence on the game's queue; a marker checks it).
  uint32_t patch_first = 60, patch_every = 20, patch_count = 30, patch_read_delay = 8;
  size_t max_frames = 8;            // per-frame copies kept (>= the 6 frames in flight of 60.7)
  bool frame_by_token = false;      // clean/rr-re9-pt: as ProxyConfig::frame_by_token
  // Hook-mode build (HOOK-MODE.md): the passes that bind the probe's own descriptor heap, root signature and
  // pipeline on the game's list (upscale, normal patch mean) run only when the game set preference bit 0 at an
  // <init> the probe saw (60.6.9: the game restores its own command-list state). Otherwise they are skipped and
  // logged; the evaluate still returns 0 and lifetime-0 copies (barriers and copies only) still run.
  bool passes_need_restore_bit = false;
};

class RrProbe {
 public:
  using TidFn = std::function<uint32_t()>;
  RrProbe(const ProbeConfig& cfg, const OriginalApi& orig, ProbeGpu* gpu, ProbeLog* log, TidFn tid);
  ~RrProbe();

  // ---- every Streamline API role (60.3.1), typed per A10.2; each call is logged with its result ----
  WgResultCode init(const WgPreferences* prefs, uint64_t api_version);
  WgResultCode shutdown();
  WgResultCode is_supported(uint32_t feature, const WgAdapterInfo* adapter);
  WgResultCode is_loaded(uint32_t feature, bool* loaded);
  WgResultCode set_loaded(uint32_t feature, bool loaded);
  WgResultCode requirements(uint32_t feature, WgRequirements* out);
  WgResultCode version(uint32_t feature, WgFeatureVersion* out);
  WgResultCode get_function(uint32_t feature, const char* name, void** out);
  WgResultCode allocate(void* cmd_list, uint32_t feature, const WgViewport* vp);
  WgResultCode free_resources(uint32_t feature, const WgViewport* vp);
  WgResultCode set_device(void* device);
  WgResultCode set_vulkan_info(const void* info);
  WgResultCode tag(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd_list);
  WgResultCode tag_for_frame(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags, uint32_t count,
                             void* cmd_list);
  WgResultCode constants(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp);
  WgResultCode new_frame_token(WgFrameToken** out, const uint32_t* frame_index);
  WgResultCode evaluate(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs, uint32_t count,
                        void* cmd_list);
  WgResultCode upgrade_interface(void** interface_ptr);
  WgResultCode native_interface(void* wrapped, void** native);
  // A graphics / Vulkan entry point forwarded by a stub (logged by name; the result is not seen).
  void note_forward(const char* export_name);
  // Hook mode only: the game's <init> ran before the hook was installed (HOOK-MODE.md). Marks the probe
  // initialised with the safe defaults: preference flags unknown (bit 0 taken as 0: the game does not restore its
  // command-list state), RR taken as requested (lifetime-0 inputs are copied), SDK version unknown (0).
  void adopt_unseen_init();

  // ---- RR feature functions (60.3.2) ----
  WgResultCode rr_optimal(const WgRrOptions* options, WgRrOptimal* out);
  WgResultCode rr_state(const WgViewport* vp, WgRrState* out);
  WgResultCode rr_options(const WgViewport* vp, const WgRrOptions* options);
  void set_feature_function_pointers(void* optimal, void* state, void* options);

  // 60.9 status word: runtime ready and model loaded (claimed), active within 1 s of a completed evaluate.
  uint32_t status_word() const;
  void set_clock(std::function<uint64_t()> clock_ms) { clock_ = std::move(clock_ms); }

  // clean/rr-re9-pt: the host's command-list restore (OptiScaler plugin interface 5) around the probe's own passes on
  // the game's list in an RR evaluate (RE Engine does not rebind its compute root state after the call)
  void set_cmd_list_bracket(CmdListBracket bracket);

  // test views
  uint64_t calls(const std::string& role) const;
  uint32_t rr_evaluates() const;

 private:
  CmdListBracket bracket_;
  struct Viewport;
  struct FrameData;

  uint64_t begin_call();
  void log_call(uint64_t seq, const char* role, const uint32_t* feature, WgResultCode rc, const WgFrameToken* token,
                const std::string& detail = std::string());
  void first_call(const char* role, uint64_t seq);
  std::string guid_text(const WgGuid& g) const;
  std::string chain_text(const WgHeader* h) const;
  void log_chain(const std::string& where, const WgHeader* h);
  void unwrap(void* wrapped, void** native) const;
  bool precheck(WgResultCode* rc, bool needs_device) const;
  void store_tags(std::map<std::pair<uint32_t, uint32_t>, StoredTag>& dst, const char* via, const WgViewport* vp,
                  const WgTag* tags, uint32_t count, void* cmd_list, const WgFrameToken* token);
  std::string tag_text(const WgTag& t, const ProbeTexDesc* d) const;
  FrameData* frame_for(const WgFrameToken* token, bool create);
  const StoredTag* lookup(uint32_t type, uint32_t vp, const std::vector<StoredTag>& locals,
                          const FrameData* frame) const;
  Viewport& viewport(uint32_t id);
  void arm_watch(const char* event, uint64_t seq);
  void read_patches(uint32_t now_eval);
  bool passes_allowed() const { return !cfg_.passes_need_restore_bit || (pref_flags_ & 1u) != 0; }
  // capture option
  void capture_settings_once();
  void capture_take(uint32_t n, const Viewport& v, const FrameData* frame, const std::vector<StoredTag>& locals,
                    const WgConstants* consts, const WgFrameToken* token, void* native_cmd, const StoredTag* colour,
                    const StoredTag* output);
  void capture_poll(uint32_t now_eval, bool final_pass);

  ProbeConfig cfg_;
  OriginalApi orig_;
  ProbeGpu* gpu_;
  ProbeLog* log_;
  TidFn tid_;

  mutable std::recursive_mutex mu_;
  uint64_t seq_ = 0;
  std::map<std::string, uint64_t> counts_;
  bool initialised_ = false, device_set_ = false;
  uint64_t api_version_ = 0, pref_flags_ = 0;
  uint32_t render_api_ = 0;
  bool rr_requested_ = false;
  bool set_loaded_given_ = false, set_loaded_value_ = true;
  void* fn_ptrs_[3] = {nullptr, nullptr, nullptr};
  bool token_checked_ = false;

  std::map<uint32_t, std::unique_ptr<Viewport>> vps_;
  std::map<std::pair<uint32_t, uint32_t>, StoredTag> viewport_tags_;
  std::map<uint64_t, std::unique_ptr<FrameData>> frames_;  // keyed by frame_key (60.7)
  uint64_t frame_seq_ = 0;

  // colour-extent watch: the first two evaluates after every <rr-options> / <allocate> / resize
  std::string watch_event_;
  uint64_t watch_seq_ = 0;
  uint32_t watch_left_ = 0;

  uint32_t rr_evals_ = 0;
  struct PendingPatch { uint32_t slot, marker, eval, type, format, px, py; float w2v[16]; };
  std::vector<PendingPatch> pending_patches_;
  uint32_t patches_taken_ = 0;
  uint64_t last_complete_ms_ = 0;
  bool any_complete_ = false;
  std::function<uint64_t()> clock_;

  // capture option
  struct CaptureInputMeta {
    uint32_t type, tag_format, state, lifetime, kind;
    WgExtent extent;
    uint32_t tag_width, tag_height;  // as the game filled the resource description (0 x 0 for game A)
  };
  struct PendingCapture {
    uint32_t id, marker, eval;
    uint64_t bytes;
    std::string json_head;  // everything of frame.json but the inputs list
    std::vector<CaptureInputMeta> metas;
    std::vector<std::string> skipped;
  };
  bool cap_loaded_ = false, cap_active_ = false, cap_done_logged_ = false;
  CaptureSettings cap_;
  uint32_t cap_taken_ = 0, cap_written_ = 0;
  bool cap_fired_ = false;
  uint32_t cap_shots_done_ = 0;
  uint64_t cap_bytes_ = 0;
  std::vector<PendingCapture> cap_pending_;
};

}  // namespace wgrr
