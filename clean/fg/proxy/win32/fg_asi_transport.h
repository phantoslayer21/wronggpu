// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the Win32 hook transport of the FG game adapter.
//
// The same delivery as the RR plugin's hook mode (clean/rr_f/proxy/HOOK-MODE.md): an ASI plugin that redirects, in the
// running process, the import-address-table slots of the modules that import the game interface DLL. The genuine,
// signed interface file stays on disk unchanged and is loaded and initialised by the game as usual. Only the transport
// is new; every decision is the FG adapter core's (fg_proxy_core.h), reached through fg_adapter_abi.h.
//
//  - INSTALL-OFF GATE. install() reads nothing from the process and patches nothing unless the player's per-game
//    switch is on (fg=on, fg_settings.h). With the switch off it returns at once: zero hooks, zero module reads.
//  - QUALIFIED BINDING ONLY. A title binding (FgImportBinding: the interface file name and version, the export name
//    of every hooked role, the FG feature-function names) and a native codec (IFgNativeCodec: decodes the binding's
//    native structs) must both be present and qualified; otherwise nothing is patched.
//  - FIND, GATE, REDIRECT as in HOOK-MODE.md: the interface module already loaded (or, when the exe delay-loads it,
//    the game's own file next to the exe), its version resource checked against the binding, then the by-name import
//    and delay-load slots of every other loaded module redirected for the hooked roles only. The interface module and
//    this module are never patched. Imports by ordinal are left alone.
//  - BOUNDED, FAIL-OPEN SCAN. Every header and table is read through the RR guard's bounded PE view and safe reads
//    (clean/rr_f/proxy/src/rr_guard.h: extent = min(SizeOfImage, the mapping), every RVA checked before an address is
//    formed, names without their NUL inside the image skipped). The scan counts modules, descriptors and name-table
//    entries against a budget (200000 entries / 100 ms by default). A scan over its budget patches nothing: the game
//    keeps its genuine interface for every call.
//  - FEATURE-SCOPED ROUTING. Each handler passes the game's call to the core, which answers only the FG feature and
//    only while FG is taken; every other call (super resolution, ray reconstruction, the latency features, unknown
//    ids, and FG itself while not taken) reaches the genuine export with the game's own arguments and its result is
//    returned unchanged. Forwarding calls the genuine export resolved by name, the same address the slot held, so a
//    host's inline patch on the export stays in the chain.
//  - SWAP CHAIN. When the binding names the DXGI factory creation the game imports from the interface module
//    (FgImportBinding::factory2_name; Cyberpunk 2077 creates its factory that way), that import slot is redirected too:
//    the genuine creation runs first and its result is returned unchanged; the swap-chain creation hook
//    (fg_swapchain_hook.h) then goes on the returned factory, so the chain the game itself holds is wrapped. A chain
//    created while FG is taken is claimed for the swap-chain wrapper owner and its game presents run through the
//    present bridge: the core gets the interval (the backbuffer as colour input, the binding's frame token) before the
//    presenter (FgAsiConfig::presenter) presents. Not taken: the wrapper stays a pass-through.
//  - No exception crosses a handler: a codec or core failure forwards the call unchanged. No lambda captures a copy
//    that can throw (MSVC /Zc:lambda under /permissive-).
// Written for this project.
#pragma once
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <stdint.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "../../../rr_f/proxy/include/rr_proxy_abi.h"
#include "../../../rr_f/proxy/src/rr_guard.h"
#include "../fg_adapter_abi.h"
#include "../fg_proxy_core.h"
#include "../fg_settings.h"
#include "fg_swapchain_hook.h"

namespace wgfgp {
namespace win32 {

// The game interface calls the transport redirects. The export name of each comes from the title binding.
enum FgRole : uint32_t {
    kFgRoleIsSupported = 0,
    kFgRoleRequirements = 1,
    kFgRoleGetFunction = 2,
    kFgRoleTag = 3,
    kFgRoleTagForFrame = 4,
    kFgRoleConstants = 5,
    kFgRoleEvaluate = 6,
    kFgRoleFree = 7,
    kFgRoleCount = 8
};
const char* FgRoleWord(uint32_t role);  // "is-supported", "requirements", ... (log text)

// Native signatures of the redirected calls (the game interface's common structs, rr_proxy_abi.h).
using FgIsSupportedFn = WgResultCode (*)(uint32_t feature, const WgAdapterInfo* adapter);
using FgRequirementsFn = WgResultCode (*)(uint32_t feature, WgRequirements* req);
using FgGetFunctionFn = WgResultCode (*)(uint32_t feature, const char* name, void** out);
using FgTagFn = WgResultCode (*)(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd);
using FgTagForFrameFn = WgResultCode (*)(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags,
                                         uint32_t count, void* cmd);
using FgConstantsFn = WgResultCode (*)(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp);
using FgEvaluateFn = WgResultCode (*)(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs,
                                      uint32_t count, void* cmd);
using FgFreeFn = WgResultCode (*)(uint32_t feature, const WgViewport* vp);
// The FG feature functions the game fetches through get-function: options in, state out (layouts: the codec's). The
// state call also takes the game's options (may be null).
using FgOptionsFn = WgResultCode (*)(const WgViewport* vp, const void* native_options);
using FgStateFn = WgResultCode (*)(const WgViewport* vp, void* native_state, const void* native_options);
// The DXGI factory creation the game imports from the interface module (CreateDXGIFactory2's shape).
using FgCreateFactory2Fn = HRESULT(WINAPI*)(UINT flags, REFIID riid, void** factory);
// The import slot index of that factory creation in FgPatch::role / the scan (not a game-interface role).
inline constexpr uint32_t kFgSlotCreateFactory2 = kFgRoleCount;

// One title binding of the interface file: what call-shape evidence established. Data (fg_title_bindings.h).
struct FgImportBinding {
    bool qualified = false;
    std::string interposer_leaf;        // the interface file name (as the importers name it)
    uint32_t version_major = 0;         // the interface file version the binding was qualified on
    uint32_t version_minor = 0;
    std::string role_names[kFgRoleCount];  // export name per role; empty: the role is not redirected
    std::string fn_options;             // the FG options feature-function name
    std::string fn_state;               // the FG state feature-function name
    uint32_t abi_max_generated = 0;     // the interface ABI's highest generated count (0..5)
    // The DXGI factory creation export of the interface module the game imports (empty: not redirected). The swap-chain
    // creation hook (fg_swapchain_hook.h) goes on the factory it returns, so the chain the game holds is wrapped.
    std::string factory2_name;
};

// Decodes the binding's native FG structs into fg_adapter_abi.h structs (and writes the core's answers back). Supplied
// with a qualified binding; tests supply a synthetic one. Any method may fail (false / 0); a throw is caught by the
// transport and the call forwards unchanged.
class IFgNativeCodec {
public:
    virtual ~IFgNativeCodec() = default;
    virtual bool options(const WgViewport* vp, const void* native, WgFgOptions* out) = 0;
    virtual bool write_state(const WgFgState& state, void* native) = 0;
    // exactly `count` decoded tags, one per native tag (a tag that is no FG input: native_type
    // WG_FG_NATIVE_TYPE_UNBOUND); false: not decodable (the tags still forward unchanged)
    virtual bool tags(const WgFrameToken* token, const WgViewport* vp, const WgTag* native, uint32_t count,
                      WgFgTag* out) = 0;
    virtual bool constants(const WgConstants* native, const WgFrameToken* token, const WgViewport* vp,
                           WgFgConstants* out) = 0;
    virtual bool write_requirements(const WgFgCapability& cap, WgRequirements* native) = 0;
    // The interface's own set-options checks, made while FG is taken before anything is decoded (result codes of
    // rr_proxy_abi.h); WG_OK: decode and hand to the core. max_generated: what the state reports. Default: none.
    virtual WgResultCode check_options(const WgViewport*, const void*, uint32_t) { return WG_OK; }
    // get-state with the game's options argument (may be null). Default: write_state.
    virtual bool write_state_for(const WgFgState& state, const void*, void* native) { return write_state(state, native); }
    // Present-driven titles whose tag call carries no frame token: the frame token of the interval a game present
    // closes, and the step after that present (fg_sl214_codec.h, frame clock). Default: 0 / nothing.
    virtual uint64_t frame_for_present() { return 0; }
    virtual void note_present() {}
};

// Why install() did what it did. Stable numbers (log text and the status export).
enum FgInstallResult : uint32_t {
    kFgInstalled = 0,
    kFgInstallOff = 1,            // fg=off: nothing read, nothing patched
    kFgNoBinding = 2,             // no qualified title binding or codec: nothing patched
    kFgInterfaceNotFound = 3,     // the interface module is not loaded and no importer delay-loads it
    kFgVersionUnreadable = 4,
    kFgVersionMismatch = 5,
    kFgAttachRefused = 6,         // the core refused the chain (fg_proxy_core.h attach_interposer)
    kFgScanOverBudget = 7,        // the bounded scan ran out of its budget: nothing patched
    kFgNothingToPatch = 8,        // no importer slot names a hooked role
    kFgAlreadyActive = 9,         // another transport is active in this process (double init)
    kFgNullArgument = 10,
    kFgRuntimeMissing = 11        // F9 (WrongGPU-FG.asi): the FG model or kernel folder failed the precheck
};
const char* FgInstallResultWord(uint32_t r);

using FgVersionReader = bool (*)(HMODULE module, uint32_t* major, uint32_t* minor);
// F9: the present bridge's call into the core (FgAsiConfig::present_route; null: FgProxy::present). Returns what
// FgProxy::present returns; *out is the core's result.
using FgPresentRouteFn = int32_t (*)(void* ctx, FgProxy* core, const WgFgPresentDesc* desc, WgFgPresentResult* out);
// F9: told of every factory the game creates through the redirected slot (after the creation hook), so the route's
// adapter facts can be read from it before the game asks whether frame generation is supported.
using FgFactoryNoteFn = void (*)(void* ctx, FgProxy* core, IUnknown* factory);

struct FgAsiConfig {
    FgSettings settings;                // the player's per-game FG settings (fg_settings.h)
    FgImportBinding imports;            // the interface binding (export names, file version)
    FgProxyConfig core;                 // feature ids, the core's title binding, modes gate (settings replaced above)
    IFgNativeCodec* codec = nullptr;    // not owned; must outlive the transport
    FgPresentOwners* owners = nullptr;  // null: the process registry
    FgLogFn log = nullptr;
    void* log_ctx = nullptr;
    FgVersionReader read_version = nullptr;  // null: the module's version resource
    unsigned scan_max_entries = 200000;      // the load scan's budget (rr_guard.h kScanMaxEntries / kScanMaxMs)
    unsigned scan_max_ms = 100;
    bool load_delay_linked = true;      // the exe delay-loads the interface: load the game's own file next to the exe
    // The presenter behind the present bridge (not owned; null: the bridge presents the game image itself). Chains
    // created through the hooked factory are claimed for WG_FG_OWNER_SWAPCHAIN_WRAPPER while FG is taken; every game
    // present then reaches the core (FgProxy::present, the backbuffer as colour input) before this presenter.
    IFgPresentCallback* presenter = nullptr;
    // F9 (the FG session host, fg_asi_session.h): the bridge's present call into the core and the factory note; when
    // the presenter serves the facade (GetBuffer / CurrentBackBufferIndex), the colour input is the facade's buffer.
    FgPresentRouteFn present_route = nullptr;
    void* present_route_ctx = nullptr;
    FgFactoryNoteFn factory_note = nullptr;
    void* factory_note_ctx = nullptr;
};

// What the present bridge saw (the transport's swap-chain sink and present callback; the process's one transport).
struct FgPresentBridgeStats {
    uint64_t factories = 0;          // factory creations through the redirected slot
    uint64_t factory_hooked = 0;     // 1: the swap-chain creation hook went on a returned factory
    uint64_t chains_created = 0;     // chains created through the hooked factory
    uint64_t chains_attached = 0;    // ... claimed for the bridge (FG taken at creation)
    uint64_t submits = 0;            // game intervals handed to the core
    uint64_t generate = 0, real_only = 0, forwarded = 0, refused = 0;  // the core's present status, per interval
    uint64_t failures = 0;           // a submit that threw (the interval is still presented)
    WgFgPresentResult last = {};     // the core's last present result
    WgFgPresentDesc last_desc = {};  // the last present descriptor (backbuffer included)
};

struct FgPatch {
    void** slot;
    void* original;   // what the slot held before
    void* handler;    // what we wrote
    uint32_t role;
    bool delay_load;
    HMODULE module;
};

struct FgTransportStats {
    uint64_t calls[kFgRoleCount] = {};
    uint64_t forwarded[kFgRoleCount] = {};  // reached the genuine export (directly or through the core's chain)
    uint64_t fg_answered = 0;               // answered for the FG feature without the genuine export
    uint64_t option_calls = 0, state_calls = 0;
    uint64_t codec_failures = 0;            // a decode that failed or threw (the call forwarded unchanged)
    uint64_t scan_entries = 0;
};

// The transport. At most one is active per process (the handlers are free functions with the game's signatures).
class FgAsiTransport {
public:
    FgAsiTransport();
    ~FgAsiTransport();  // restore()
    FgAsiTransport(const FgAsiTransport&) = delete;
    FgAsiTransport& operator=(const FgAsiTransport&) = delete;

    // Install-off gate, binding gate, find, version gate, core attach, bounded scan, patch. Never throws.
    uint32_t install(const FgAsiConfig& cfg);
    // Puts back every slot that still holds our handler (a slot rewritten since by another module is left alone),
    // stops routing (handlers then forward to the genuine export), waits for calls in flight, detaches the core.
    void restore();

    bool active() const { return active_; }
    uint32_t result() const { return result_; }
    const std::vector<FgPatch>& patches() const { return patches_; }
    size_t patch_count() const { return patches_.size(); }
    FgProxy* core() { return core_.get(); }
    HMODULE interface_module() const { return iface_; }
    bool interface_loaded_by_us() const { return loaded_by_us_; }
    FgTransportStats stats() const;
    FgPresentBridgeStats present_stats() const;
    // the genuine export of a role or of kFgSlotCreateFactory2 (null when the binding has no name for it or the
    // interface does not export it)
    void* genuine(uint32_t role) const;
    // the handler written into the slots of a role or of kFgSlotCreateFactory2 (fixed addresses in this module)
    static void* handler_for(uint32_t role);
    // the swap-chain sink and present callback the factory hook uses (process lifetime; inert while no route is active)
    static IFgSwapChainSink* swapchain_sink();
    static IFgPresentCallback* present_bridge();
    static void* options_handler();
    static void* state_handler();

    // What the handlers route through while this transport is active (internal).
    struct Impl;

private:
    void log(const std::string& line) const;
    uint32_t finish(uint32_t result, const std::string& line);

    std::unique_ptr<Impl> impl_;
    bool active_ = false;
    uint32_t result_ = kFgNullArgument;
    std::unique_ptr<FgProxy> core_;
    std::vector<FgPatch> patches_;
    HMODULE iface_ = nullptr;
    bool loaded_by_us_ = false;
    FgLogFn log_fn_ = nullptr;
    void* log_ctx_ = nullptr;
};

// The by-name slots of `module` that import a name in `names` from `leaf`: the import table, then the delay-load table
// (both through the RR guard's bounded PE view; `budget` counts every entry). Exposed for the tests' synthetic images.
struct FgSlot {
    void** slot;
    uint32_t name_index;  // index into names
    bool delay_load;
};
std::vector<FgSlot> FgCollectSlots(const void* module, const char* leaf, const std::vector<std::string>& names,
                                   wgrr::guard::Budget& budget);

// The interface module's VERSIONINFO major.minor (version.dll, as the RR hook reads it).
bool FgModuleFileVersion(HMODULE module, uint32_t* major, uint32_t* minor);

}  // namespace win32
}  // namespace wgfgp
