// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation (issue #96, F7b): the CPU side of the D3D12 executor (fg_d3d12_executor.h), plain C++17 with no
// platform headers so that it is tested on its own:
//
//  - FgExecSettings: the create-time / layered settings of clean/fg/reference/pipeline.py Settings with its defaults.
//  - FgExecGeometry: the grid sizes of one frame size (pipeline.py Geometry: the net1 image grid with d = 4 above 1440
//    lines, its cell grid with the centring pad, the scatter grid with d' = 4 above 2880 lines and its pyramid), plus the
//    Full path's sizes (net2's k and grid, the output pyramid).
//  - FgExecRegion: an application texture's region and the ten single-precision values a kernel takes for it
//    (stages_common.py Region: rx = 1 / texture width as one IEEE division, the clamp bounds rounded up / down).
//  - FgExecHost: the light path's schedule (pipeline.py Host) split at the executor seam of fg_session.h. BeginFrame is
//    the first call of a game frame (reset / flag-change check, history clears, menu detection or the disable word, R3
//    .. R8 or net1's input / net1 / output); IndexOps is the per-index part (the two intermediate-time scatters, the
//    blend weights, the blend and the menu output with history, else the copy of the game frame); EndFrame is the
//    last-index part (history advances, the current / previous image pairs swap). The session calls the indices of a
//    game frame in order and may replace them with exact copies (menu latch, priming); the executor runs EndFrame once
//    per game frame before the next BeginFrame, which is the order pipeline.py gives when every index is called.
//    Presets: C is the light path (lite = true); A and B run the same shared stages with the light flags off (B: UI
//    recomposition on) and the Full blend, net2 and the output pyramid per index.
#include <cstdint>
#include <string>
#include <vector>

#include "fg_contract.h"

namespace wgfg {

struct FgExecSettings {
    bool menu_detection = false;
    bool lite_l1 = false;
    bool lite_l2 = true;
    bool ignore_distortion = false, ignore_hudless = false, ignore_ui = false;
    float alpha_threshold = 0.7f;
    float fraction = 0.7f;
    float kmax = 600.0f;
    float tau = 40.0f;
    float s_k = 1.0f;
};

struct FgExecGeometry {
    uint32_t wo = 0, ho = 0;          // frame (output) size
    uint32_t d = 0, wd = 0, hd = 0;   // net1 image grid
    uint32_t W0 = 0, H0 = 0, pad_l = 0, pad_t = 0;  // net1 cell grid
    uint32_t dp = 0, w1 = 0, h1 = 0;  // scatter grid
    std::vector<uint32_t> lw, lh;     // scatter-grid pyramid, level 0 first
    uint32_t top = 0;                 // coarsest pull level
    // Full path
    uint32_t k = 0, W2 = 0, H2 = 0;   // net2 factor and grid (Wp / k x Hp / k)
    uint32_t Wp = 0, Hp = 0;          // the padded frame
    uint32_t out_levels = 0;          // output pyramid level count (level 0 = the frame)
    uint32_t out_coarsest = 0;        // 8 or 4
};
// False outside 4 x 4 .. 8192 x 7680.
bool FgExecMakeGeometry(uint32_t wo, uint32_t ho, FgExecGeometry* out, std::string* why);
inline void FgExecOutLevelSize(const FgExecGeometry& g, uint32_t l, uint32_t* w, uint32_t* h) {
    *w = (g.wo + (1u << l) - 1) >> l;
    *h = (g.ho + (1u << l) - 1) >> l;
}

struct FgExecRegion {
    uint32_t tex_w = 0, tex_h = 0, bx = 0, by = 0, ew = 0, eh = 0;
    float rx = 0, ry = 0, lox = 0, hix = 0, loy = 0, hiy = 0;
    // ew, eh, bx, by, rx, ry, lox, hix, loy, hiy as float bits
    void Args(uint32_t out[10]) const;
};
// False when the region is outside its texture or empty.
bool FgExecMakeRegion(uint32_t tex_w, uint32_t tex_h, uint32_t bx, uint32_t by, uint32_t ew, uint32_t eh,
                      FgExecRegion* out);

uint32_t FgExecFloatBits(float f);
uint16_t FgExecHalfBits(float f);   // round to nearest even, subnormals kept, NaN quietened
float FgExecHalfToFloat(uint16_t h);
// T1 = Ho^2 / 900, T2 = Ho^2 / 10000 (IEEE single divisions)
void FgExecThresholds(uint32_t ho, float* t1, float* t2);

// The internal images of the light path (pipeline.py image_shapes) whose current / previous pairs swap.
enum FgRole : uint32_t {
    kRoleCc, kRoleCp, kRoleLc, kRoleLp, kRoleUc, kRoleUp, kRoleDc, kRoleDp, kRoleLd, kRoleM, kRoleUm, kRoleUmp,
    kRoleHc, kRoleHp, kRoleOm, kRoleQc, kRoleQp, kRoleNA, kRoleNB, kRoleMv, kRoleVf, kRoleZc, kRoleZp, kRoleFl,
    kRoleG, kRoleKc, kRoleKp, kRoleE, kRoleF, kRoleW, kRoleS, kRoleMA, kRoleMB,
    kRoleP1,  // P1 .. P11: pyramid levels (kRoleP1 + l - 1)
    kRoleCount = kRoleP1 + 11,
};
const char* FgRoleName(uint32_t role);
// Bytes of one role's image for a geometry (0 for an unused pyramid level).
uint64_t FgRoleBytes(const FgExecGeometry& g, uint32_t role);

struct FgExecFlags {
    bool distortion = false, hudless = false, ui_alpha = false, ui_recomp = false, lite1 = false, lite2 = false;
    bool operator==(const FgExecFlags& o) const {
        return distortion == o.distortion && hudless == o.hudless && ui_alpha == o.ui_alpha &&
               ui_recomp == o.ui_recomp && lite1 == o.lite1 && lite2 == o.lite2;
    }
    bool operator!=(const FgExecFlags& o) const { return !(*this == o); }
};
FgExecFlags FgExecDerivedFlags(const FgExecSettings& st, bool lite, bool ui_recomposition, bool hudless, bool ui4,
                               bool ui1, bool distortion);
// R3: F1 (UI recomposition or Lite flag 1), A (UI alpha), Hd (hudless), L1, L2; and the case "U" / "A" / "H" / "0".
uint32_t FgExecR3Bits(const FgExecFlags& f);  // F1 1, A 2, Hd 4, L1 8, L2 16
char FgExecR3Case(const FgExecFlags& f);

enum class FgOp : uint32_t {
    Clear,        // role, value (float16 units, or a 32-bit word for uint images)
    Detect,       // detect_kind (0 alpha, 1 hudless), hysteresis_off, base
    SetDisable,   // value
    Downsample, Mvec, Fill64, P2cScatter, P2cUnpack, P2cInpaint, P2cPull, P2cPush, HoleMask, QualityInit,
    Net1Input, Net1, Net1Output,  // ui_forcing
    Interm,       // t
    Weights,
    Prefill,
    LiteBlend,    // t, flag1, flag2
    FullBlend,    // ui (preset B)
    Net2,         // f = t
    OutPyramid,   // pull, middle pull, push, push-fine
    MenuOutput,
    FinalCopy,
};
const char* FgOpName(FgOp op);

struct FgExecOp {
    FgOp op = FgOp::Clear;
    uint32_t role = 0;
    uint32_t value[4] = {};   // Clear: the pattern (float16 bits, or the low / high halves of a 32-bit word)
    uint32_t period = 0;      // Clear / Fill64: units in the pattern
    uint32_t detect_kind = 0;
    bool hysteresis_off = false;
    uint32_t base = 0;
    bool ui_forcing = false;
    float t = 0.0f;
    bool flag1 = false, flag2 = false;
    bool ui = false;
    bool lite_blend = false;  // Weights: the blend that follows is the light blend (else the Full blend)
};

struct FgExecFrameInfo {
    bool history = false;      // the frame has history (generation with net1), else the game frame is copied
    FgExecFlags flags;
    uint32_t r3_bits = 0;
    char r3_case = '0';
    bool detect_ran = false;
};

class FgExecHost {
public:
    void Reset() { *this = FgExecHost(); }
    void Configure(const FgExecSettings& st, const FgExecGeometry& g) {
        st_ = st;
        g_ = g;
        Reset2();
    }
    // The first call of a game frame. lite: preset C; ui_recomposition: preset B. reset: no history (priming, reset,
    // camera cut). Appends the once-per-game-frame operations.
    void BeginFrame(bool lite, bool ui_recomposition, bool hudless, bool ui4, bool ui1, bool reset,
                    std::vector<FgExecOp>* ops);
    // One generated index (t = i / (N + 1)), with the frame's frozen flags.
    void IndexOps(float t, bool hudless_now, bool ui4_now, bool out_subrect, std::vector<FgExecOp>* ops) const;
    // The last-index part: history advances; the swaps to apply (pairs of roles).
    void EndFrame(std::vector<uint32_t>* swap_pairs);
    bool InFrame() const { return in_frame_; }
    const FgExecFrameInfo& Frame() const { return frame_; }
    uint64_t HistoryFrames() const { return history_; }

private:
    void Reset2() {
        history_ = 0;
        have_prev_ = false;
        detect_ran_prev_ = false;
        in_frame_ = false;
        frame_ = FgExecFrameInfo();
    }
    FgExecSettings st_;
    FgExecGeometry g_;
    uint64_t history_ = 0;
    bool have_prev_ = false;
    FgExecFlags prev_flags_;
    bool detect_ran_prev_ = false;
    bool in_frame_ = false;
    bool lite_ = true;
    FgExecFrameInfo frame_;
};

// The order in which the launches of several generated indices that share one command list are recorded (the executor's
// RecordGeneratedBatch; plain C++ so that the rule is tested on its own). Every frame is the launch list of one index:
// its kernels and whether each launch needs a barrier before it (it reads what an earlier launch of the same index wrote).
struct FgLaunchKey {
    uint32_t kernel = 0;
    bool barrier = true;
};
struct FgLaunchSlot {
    uint32_t frame = 0, index = 0;  // frame f's launch `index`
    bool barrier = true;            // a barrier is recorded before this launch
    bool mark_stage = true;         // the stage timestamp is written before this launch
};
// Frames that run the same plan (equal launch counts, kernels and barrier needs at every position) are interleaved:
// launch j of every frame back to back, a barrier only before frame 0's launch j and only when it needs one (the frames
// write disjoint images and read only state that is final before the batch, so the barrier before frame 0's launch j,
// which waits for every launch recorded before it, also covers the other frames' launch j - 1). Any other set of frames
// is recorded frame after frame with each launch's own barrier flag.
void FgInterleaveLaunches(const std::vector<std::vector<FgLaunchKey>>& frames, std::vector<FgLaunchSlot>* out);

// A18: whether the same launch of `frames` frames (args[f] = frame f's n argument DWORDs) can be one (groups, 1, frames)
// launch of a z-sliced kernel, and the mask of the pointer arguments the slices must offset. Every argument is either the
// same in every frame (weights, null pointers, all scalars), or one of the first `pointers` arguments that frame f has at
// frame 0's value + f * stride (the scratch banks are `stride` bytes apart). Any other difference refuses; so does a launch
// with no banked pointer. mask bit i = argument i is banked.
bool FgPlanZBatch(const uint32_t* const* args, uint32_t frames, uint32_t n, uint32_t pointers, uint64_t stride,
                  uint32_t* mask);

// net2's 25 convolutions (clean/fg/layout.py NET2): stored input channels, used input channels, outputs.
struct FgNet2Layer {
    uint32_t cin, cin_used, cout;
};
inline constexpr uint32_t kFgNet2Layers = 25;
const FgNet2Layer* FgNet2Layers();

}  // namespace wgfg
