// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): CPU tests of the Win32 hook transport and the swap-chain hook seam.
//
// The game side is a stand-in: fg_mock_interposer.dll (synthetic export names, a version resource, a call record) and
// fg_mock_game.dll / fg_mock_game_delay.dll, which import it by name through a static and a delay-load import table.
// The swap chain and the factory are fake COM objects. Every value is synthetic. Built with CI's flags
// (/W4 /WX /permissive-, static runtime).
//
//   test_fg_win32_transport                in-process suite: bounded PE walk on synthetic images, install-off gate,
//                                          routing and coexistence, restore, failures, swap-chain wrapper and hook
//   test_fg_win32_transport --asi <leg>    the shipped plugin (WrongGPU-FG.asi) in its own process: <leg> = off |
//                                          default | on | invalid; the settings come from WG_STATE_GAME_DIR, set
//                                          before the process starts (ctest)
//   test_fg_win32_delay_exe                (FG_TEST_DELAY_EXE) an exe that delay-loads the interface and has not
//                                          loaded it: the transport loads the game's own file next to the exe and
//                                          redirects the exe's delay-load slots
// Written for this project.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../win32/fg_asi_transport.h"
#include "../win32/fg_swapchain_hook.h"
#include "fg_mock_interposer.h"

using namespace wgfgp;
using namespace wgfgp::win32;

namespace {

int g_checks = 0, g_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failed;                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

constexpr uint32_t kFeatFg = WG_FEAT_FRAME_GENERATION;
constexpr uint32_t kFeatSr = WG_FEAT_SUPER_RESOLUTION;
constexpr uint32_t kFeatRr = WG_FEAT_RAY_RECONSTRUCTION;
constexpr uint32_t kFeatLatency = WG_FEAT_LATENCY;
constexpr uint32_t kFeatPcLatency = WG_FEAT_PC_LATENCY;
constexpr uint32_t kAmdPciVendor = 4098;  // the PCI vendor id the core requires
const char* const kLeaf = "fg_mock_interposer.dll";

std::vector<std::string> g_log;
void CollectLog(void*, const char* line) { g_log.push_back(line ? line : ""); }

// ---- synthetic native structs and their codec ----------------------------------------------------------------------

struct MockOptions {
    uint32_t enable, preset, count;
};
struct MockState {
    uint32_t taken, enabled, preset, count, max_generated;
};

class MockCodec final : public IFgNativeCodec {
public:
    enum Where { kNone = -1, kOptions = 0, kState = 1, kTags = 2, kConstants = 3, kRequirements = 4 };
    int throw_in = kNone;
    int fail_in = kNone;
    uint64_t token_value = 1;
    uint32_t decoded_tags = 0;

    bool options(const WgViewport*, const void* native, WgFgOptions* out) override {
        Hit(kOptions);
        if (fail_in == kOptions) return false;
        const MockOptions* o = static_cast<const MockOptions*>(native);
        out->enable = o->enable;
        out->preset = o->preset;
        out->generated_count = o->count;
        out->output_width = 64;
        out->output_height = 32;
        return true;
    }
    bool write_state(const WgFgState& s, void* native) override {
        Hit(kState);
        if (fail_in == kState) return false;
        MockState* m = static_cast<MockState*>(native);
        m->taken = s.taken;
        m->enabled = s.enabled;
        m->preset = s.effective_preset;
        m->count = s.effective_count;
        m->max_generated = s.max_generated;
        return true;
    }
    bool tags(const WgFrameToken*, const WgViewport* vp, const WgTag* native, uint32_t count, WgFgTag* out) override {
        Hit(kTags);
        if (fail_in == kTags) return false;
        for (uint32_t i = 0; i < count; ++i) {
            out[i].native_type = native[i].buffer_type;
            out[i].texture.resource = native[i].resource;
            out[i].texture.frame_token = token_value;
            out[i].texture.viewport_id = vp ? vp->id : 0;
            out[i].texture.width = native[i].extent.width;
            out[i].texture.height = native[i].extent.height;
        }
        decoded_tags += count;
        return true;
    }
    bool constants(const WgConstants* native, const WgFrameToken*, const WgViewport* vp, WgFgConstants* out) override {
        Hit(kConstants);
        if (fail_in == kConstants) return false;
        out->frame_token = token_value;
        out->viewport_id = vp ? vp->id : 0;
        out->jitter_x = native->jitter[0];
        out->jitter_y = native->jitter[1];
        return true;
    }
    bool write_requirements(const WgFgCapability& cap, WgRequirements* native) override {
        Hit(kRequirements);
        if (fail_in == kRequirements) return false;
        native->flags = 2u;  // D3D12
        native->max_viewports = 1;
        native->required_tag_count = cap.supported ? 3u : 0u;
        return true;
    }

private:
    void Hit(int where) {
        if (throw_in == where) throw std::runtime_error("synthetic codec failure");
    }
};

const char* const kRoleNames[kFgRoleCount] = {"fgmock_is_supported", "fgmock_requirements", "fgmock_get_function",
                                              "fgmock_tag",          "fgmock_tag_for_frame", "fgmock_constants",
                                              "fgmock_evaluate",     "fgmock_free"};

FgAsiConfig MakeConfig(MockCodec* codec, FgPresentOwners* owners, bool enabled) {
    FgAsiConfig c;
    c.settings.enabled = enabled;
    c.settings.multiplier = 6;
    c.imports.qualified = true;
    c.imports.interposer_leaf = kLeaf;
    c.imports.version_major = 3;
    c.imports.version_minor = 1;
    for (uint32_t i = 0; i < kFgRoleCount; ++i) c.imports.role_names[i] = kRoleNames[i];
    c.imports.fn_options = FG_MOCK_FN_FG_OPTIONS;
    c.imports.fn_state = FG_MOCK_FN_FG_STATE;
    c.imports.abi_max_generated = 5;
    c.core.binding.qualified = true;
    c.core.binding.version_major = 3;
    c.core.binding.version_minor = 1;
    c.core.binding.fn_names = {FG_MOCK_FN_FG_OPTIONS, FG_MOCK_FN_FG_STATE};
    c.core.binding.ui_max_generated = 5;
    c.core.binding.native_type[WG_FG_ROLE_BACKBUFFER] = 900;
    c.core.binding.native_type[WG_FG_ROLE_MOTION] = 901;
    c.core.binding.native_type[WG_FG_ROLE_DEPTH] = 902;
    c.core.modes_qualified = true;
    c.codec = codec;
    c.owners = owners;
    c.log = &CollectLog;
    return c;
}

void MakeTaken(FgProxy* core) {
    WgFgAdapterIdentity id = {};
    id.struct_size = sizeof(id);
    id.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    id.vendor_id = kAmdPciVendor;
    id.device_id = 1;
    id.luid = {77, 0};
    id.rdna4 = 1;
    id.probes_passed = 1;
    core->set_adapter(id);
    core->set_runtime_facts(true, true);
}

void SetMockResults() {
    FgMockRecord* r = fgmock_record();
    for (uint32_t i = 0; i < kMockFnCount; ++i) r->result[i] = static_cast<int32_t>(40 + i);  // distinct, synthetic
}

#if !defined(FG_TEST_DELAY_EXE)

size_t LogCount(const char* needle) {
    size_t n = 0;
    for (const auto& l : g_log)
        if (l.find(needle) != std::string::npos) ++n;
    return n;
}

std::wstring ExeDir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
    std::wstring w(buf, n);
    return w.substr(0, w.find_last_of(L"\\/"));
}

// ---- synthetic PE images for the bounded walk ----------------------------------------------------------------------

constexpr uint32_t kPage = 4096;

struct SyntheticImage {
    uint8_t* base = nullptr;
    uint32_t mapped = 0;
    explicit SyntheticImage(uint32_t pages) {
        mapped = pages * kPage;
        base = static_cast<uint8_t*>(VirtualAlloc(nullptr, mapped, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (base) memset(base, 0, mapped);
    }
    ~SyntheticImage() {
        if (base) VirtualFree(base, 0, MEM_RELEASE);
    }
    SyntheticImage(const SyntheticImage&) = delete;
    SyntheticImage& operator=(const SyntheticImage&) = delete;

    IMAGE_NT_HEADERS64* nt() { return reinterpret_cast<IMAGE_NT_HEADERS64*>(base + 64); }
    void headers(uint32_t size_of_image) {
        IMAGE_DOS_HEADER* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 64;
        IMAGE_NT_HEADERS64* h = nt();
        h->Signature = IMAGE_NT_SIGNATURE;
        h->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
        h->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
        h->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        h->OptionalHeader.SizeOfImage = size_of_image;
        h->OptionalHeader.NumberOfRvaAndSizes = 16;
    }
    void dir(unsigned index, uint32_t rva, uint32_t size) {
        nt()->OptionalHeader.DataDirectory[index].VirtualAddress = rva;
        nt()->OptionalHeader.DataDirectory[index].Size = size;
    }
    void str(uint32_t rva, const char* s, bool nul = true) {
        memcpy(base + rva, s, strlen(s) + (nul ? 1 : 0));
    }
    void import_by_name(uint32_t thunk_rva, uint32_t name_rva, const char* name) {
        reinterpret_cast<IMAGE_THUNK_DATA64*>(base + thunk_rva)->u1.AddressOfData = name_rva;
        str(name_rva + 2, name);  // IMAGE_IMPORT_BY_NAME: hint, then the name
    }
    void import_desc(uint32_t at, uint32_t names, uint32_t iat, uint32_t dll) {
        IMAGE_IMPORT_DESCRIPTOR* d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + at);
        d->OriginalFirstThunk = names;
        d->FirstThunk = iat;
        d->Name = dll;
    }
    void delay_desc(uint32_t at, uint32_t dll, uint32_t iat, uint32_t names) {
        IMAGE_DELAYLOAD_DESCRIPTOR* d = reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(base + at);
        d->Attributes.RvaBased = 1;
        d->DllNameRVA = dll;
        d->ImportAddressTableRVA = iat;
        d->ImportNameTableRVA = names;
    }
    void** slot(uint32_t rva) { return reinterpret_cast<void**>(base + rva); }
};

// A well-formed image: one import descriptor and one delay-load descriptor naming "synthetic.dll", each importing
// "fgt_role_a" by name.
void BuildGood(SyntheticImage& im, uint32_t size_of_image) {
    im.headers(size_of_image);
    im.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, 1024, 2 * sizeof(IMAGE_IMPORT_DESCRIPTOR));
    im.import_desc(1024, 1400, 1500, 1600);
    im.import_by_name(1400, 1700, "fgt_role_a");
    im.str(1600, "synthetic.dll");
    im.dir(IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT, 1200, 2 * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR));
    im.delay_desc(1200, 1600, 1800, 1900);
    im.import_by_name(1900, 1700, "fgt_role_a");
}

void TestBoundedPeWalk() {
    const std::vector<std::string> names = {"fgt_other", "fgt_role_a"};
    {  // well formed: one import slot and one delay-load slot, each the right address
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        wgrr::guard::Budget b(1000, 1000);
        auto s = FgCollectSlots(im.base, "SYNTHETIC.DLL", names, b);
        CHECK(s.size() == 2);
        CHECK(s.size() == 2 && s[0].slot == im.slot(1500) && s[0].name_index == 1 && !s[0].delay_load);
        CHECK(s.size() == 2 && s[1].slot == im.slot(1800) && s[1].name_index == 1 && s[1].delay_load);
        auto none = FgCollectSlots(im.base, "other.dll", names, b);
        CHECK(none.empty());
    }
    {  // an import name at the image's last bytes with its NUL outside the image (readable memory after it): skipped
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        const uint32_t name_rva = 2 * kPage - 2 - 10;  // hint + "fgt_role_a" (10 chars) end exactly at the image end
        im.import_by_name(1400, name_rva, "fgt_role_a");
        im.import_by_name(1900, name_rva, "fgt_role_a");
        CHECK(im.base[2 * kPage] == 0);  // the NUL lies in the third (committed, readable) page, outside the image
        wgrr::guard::Budget b(1000, 1000);
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
        // the same name one byte earlier (its NUL inside the image) is found: the refusal is the bound, not the bytes
        im.import_by_name(1400, name_rva - 1, "fgt_role_a");
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).size() == 1);
    }
    {  // a forged huge SizeOfImage does not widen the image: a directory past the mapping is refused, nothing read
        SyntheticImage im(3);
        BuildGood(im, 2147418112u);  // far past the three mapped pages
        im.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, 5 * kPage, 2 * sizeof(IMAGE_IMPORT_DESCRIPTOR));
        im.dir(IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT, 6 * kPage, 2 * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR));
        wgrr::guard::Budget b(1000, 1000);
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
    }
    {  // the import directory declares one descriptor: the matching second one is never read
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        im.import_desc(1024, 1400, 1500, 1640);
        im.str(1640, "unrelated.dll");
        im.import_desc(1024 + sizeof(IMAGE_IMPORT_DESCRIPTOR), 1400, 1500, 1600);
        im.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, 1024, sizeof(IMAGE_IMPORT_DESCRIPTOR));
        im.dir(IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT, 0, 0);
        wgrr::guard::Budget b(1000, 1000);
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
        im.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, 1024, 3 * sizeof(IMAGE_IMPORT_DESCRIPTOR));
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).size() == 1);
    }
    {  // ordinal imports are left alone; a VA-based delay descriptor is skipped
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        reinterpret_cast<IMAGE_THUNK_DATA64*>(im.base + 1400)->u1.Ordinal = IMAGE_ORDINAL_FLAG64 | 7u;
        reinterpret_cast<IMAGE_DELAYLOAD_DESCRIPTOR*>(im.base + 1200)->Attributes.RvaBased = 0;
        wgrr::guard::Budget b(1000, 1000);
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
    }
    {  // broken headers: bad e_lfanew (negative, past the mapping), bad signature, PE32 magic, null module
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        wgrr::guard::Budget b(1000, 1000);
        reinterpret_cast<IMAGE_DOS_HEADER*>(im.base)->e_lfanew = -64;
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
        reinterpret_cast<IMAGE_DOS_HEADER*>(im.base)->e_lfanew = static_cast<LONG>(3 * kPage);
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
        reinterpret_cast<IMAGE_DOS_HEADER*>(im.base)->e_lfanew = 64;
        im.nt()->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR32_MAGIC;
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
        im.nt()->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
        im.nt()->Signature = 0;
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
        CHECK(FgCollectSlots(nullptr, "synthetic.dll", names, b).empty());
        CHECK(FgCollectSlots(im.base, nullptr, names, b).empty());
    }
    {  // an unterminated DLL name running to the end of the image: refused
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        const uint32_t at = 2 * kPage - 13;
        im.str(at, "synthetic.dll", false);
        im.import_desc(1024, 1400, 1500, at);
        im.delay_desc(1200, at, 1800, 1900);
        wgrr::guard::Budget b(1000, 1000);
        CHECK(FgCollectSlots(im.base, "synthetic.dll", names, b).empty());
    }
    {  // the budget: a scan out of entries stops and says so
        SyntheticImage im(3);
        BuildGood(im, 2 * kPage);
        wgrr::guard::Budget b(2, 1000);
        auto s = FgCollectSlots(im.base, "synthetic.dll", names, b);
        CHECK(b.over());
        CHECK(s.size() < 2);
    }
}

// ---- the stand-in game ---------------------------------------------------------------------------------------------

struct GameModule {
    HMODULE h = nullptr;
    WgResultCode (*is_supported)(uint32_t, const WgAdapterInfo*) = nullptr;
    WgResultCode (*requirements)(uint32_t, WgRequirements*) = nullptr;
    WgResultCode (*get_function)(uint32_t, const char*, void**) = nullptr;
    WgResultCode (*tag)(const WgViewport*, const WgTag*, uint32_t, void*) = nullptr;
    WgResultCode (*tag_for_frame)(const WgFrameToken*, const WgViewport*, const WgTag*, uint32_t, void*) = nullptr;
    WgResultCode (*constants)(const WgConstants*, const WgFrameToken*, const WgViewport*) = nullptr;
    WgResultCode (*evaluate)(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*) = nullptr;
    WgResultCode (*free_)(uint32_t, const WgViewport*) = nullptr;
    WgResultCode (*init)(uint64_t) = nullptr;
    void (*slot_values)(void**) = nullptr;

    template <typename F>
    void bind(F& f, const char* name) {
        f = reinterpret_cast<F>(GetProcAddress(h, name));
    }
    bool load(const wchar_t* leaf) {
        h = LoadLibraryW((ExeDir() + L"\\" + leaf).c_str());
        if (!h) return false;
        bind(is_supported, "game_is_supported");
        bind(requirements, "game_requirements");
        bind(get_function, "game_get_function");
        bind(tag, "game_tag");
        bind(tag_for_frame, "game_tag_for_frame");
        bind(constants, "game_constants");
        bind(evaluate, "game_evaluate");
        bind(free_, "game_free");
        bind(init, "game_init");
        bind(slot_values, "game_slot_values");
        return is_supported && requirements && get_function && tag && tag_for_frame && constants && evaluate && free_ &&
               init && slot_values;
    }
    std::vector<void*> slots() const {
        std::vector<void*> v(FG_MOCK_GAME_SLOTS, nullptr);
        slot_values(v.data());
        return v;
    }
};

void* GenuineExport(const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA(kLeaf), name));
}

bool SlotsAreGenuine(const GameModule& g) {
    const auto v = g.slots();
    for (uint32_t i = 0; i < kFgRoleCount; ++i)
        if (v[i] != GenuineExport(kRoleNames[i])) return false;
    return true;
}

bool SlotsAreOurs(const GameModule& g) {
    const auto v = g.slots();
    for (uint32_t i = 0; i < kFgRoleCount; ++i)
        if (v[i] != FgAsiTransport::handler_for(i)) return false;
    return true;
}

// Every role called once through the game's slots with synthetic arguments; each call must reach the stand-in with
// exactly these arguments and return exactly the stand-in's result.
void CheckEveryCallPassesThrough(const GameModule& g, uint32_t feature) {
    FgMockRecord* r = fgmock_record();
    WgAdapterInfo adapter_info = {};
    WgRequirements rq = {};
    WgViewport vp = {};
    vp.id = 3;
    WgTag tags[2] = {};
    WgConstants cs = {};
    const WgFrameToken* token = reinterpret_cast<const WgFrameToken*>(&cs);  // opaque to the transport: never read
    void* cmd = &rq;
    const WgHeader* inputs[1] = {&vp.h};
    void* out = nullptr;
    const uint32_t before[kMockFnCount] = {r->calls[0], r->calls[1], r->calls[2], r->calls[3], r->calls[4], r->calls[5],
                                           r->calls[6], r->calls[7], r->calls[8], r->calls[9], r->calls[10], r->calls[11]};

    CHECK(g.is_supported(feature, &adapter_info) == r->result[kMockIsSupported]);
    CHECK(r->last[kMockIsSupported].feature == feature && r->last[kMockIsSupported].a == &adapter_info);
    CHECK(g.requirements(feature, &rq) == r->result[kMockRequirements]);
    CHECK(r->last[kMockRequirements].feature == feature && r->last[kMockRequirements].a == &rq);
    CHECK(g.get_function(feature, "fgmock_fn_none", &out) == WG_E_FEATURE_MISSING);
    CHECK(r->last[kMockGetFunction].feature == feature && r->last[kMockGetFunction].b == &out);
    CHECK(g.tag(&vp, tags, 2, cmd) == r->result[kMockTag]);
    CHECK(r->last[kMockTag].a == &vp && r->last[kMockTag].b == tags && r->last[kMockTag].c == cmd &&
          r->last[kMockTag].count == 2);
    CHECK(g.tag_for_frame(token, &vp, tags, 2, cmd) == r->result[kMockTagForFrame]);
    CHECK(r->last[kMockTagForFrame].a == token && r->last[kMockTagForFrame].b == &vp &&
          r->last[kMockTagForFrame].c == tags && r->last[kMockTagForFrame].d == cmd && r->last[kMockTagForFrame].count == 2);
    CHECK(g.constants(&cs, token, &vp) == r->result[kMockConstants]);
    CHECK(r->last[kMockConstants].a == &cs && r->last[kMockConstants].b == token && r->last[kMockConstants].c == &vp);
    CHECK(g.evaluate(feature, token, inputs, 1, cmd) == r->result[kMockEvaluate]);
    CHECK(r->last[kMockEvaluate].feature == feature && r->last[kMockEvaluate].a == token &&
          r->last[kMockEvaluate].b == inputs && r->last[kMockEvaluate].c == cmd && r->last[kMockEvaluate].count == 1);
    CHECK(g.free_(feature, &vp) == r->result[kMockFree]);
    CHECK(r->last[kMockFree].feature == feature && r->last[kMockFree].a == &vp);
    for (uint32_t i = 0; i < kFgRoleCount; ++i) CHECK(r->calls[i] == before[i] + 1);
}

void TestInstallOff(const GameModule& game, const GameModule& delay) {
    fgmock_reset();
    SetMockResults();
    MockCodec codec;
    FgPresentOwners owners;
    FgAsiTransport t;
    g_log.clear();
    FgAsiConfig cfg = MakeConfig(&codec, &owners, false);  // fg=off (the default)
    CHECK(FgSettings{}.enabled == false);
    CHECK(t.install(cfg) == kFgInstallOff);
    CHECK(!t.active() && t.patch_count() == 0 && t.core() == nullptr);
    CHECK(t.interface_module() == nullptr);  // nothing was even looked up
    CHECK(SlotsAreGenuine(game));
    const auto dv = delay.slots();
    for (uint32_t i = 0; i < kFgRoleCount; ++i) CHECK(dv[i] != FgAsiTransport::handler_for(i));
    CheckEveryCallPassesThrough(game, kFeatFg);
    CheckEveryCallPassesThrough(game, kFeatSr);
    const FgTransportStats s = t.stats();
    for (uint32_t i = 0; i < kFgRoleCount; ++i) CHECK(s.calls[i] == 0);  // no handler ran
    CHECK(LogCount("fg=off") == 1);
}

void TestGatesPatchNothing(const GameModule& game) {
    MockCodec codec;
    FgPresentOwners owners;
    {  // no qualified binding (imports, core binding, or codec): nothing patched
        FgAsiTransport t;
        FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
        cfg.imports.qualified = false;
        CHECK(t.install(cfg) == kFgNoBinding);
        cfg = MakeConfig(&codec, &owners, true);
        cfg.core.binding.qualified = false;
        CHECK(t.install(cfg) == kFgNoBinding);
        cfg = MakeConfig(nullptr, &owners, true);
        CHECK(t.install(cfg) == kFgNoBinding);
        CHECK(t.patch_count() == 0 && SlotsAreGenuine(game));
    }
    {  // the version gate: the default reader reads the stand-in's resource (3.1); another version patches nothing
        uint32_t ma = 0, mi = 0;
        CHECK(FgModuleFileVersion(GetModuleHandleA(kLeaf), &ma, &mi) && ma == 3 && mi == 1);
        CHECK(!FgModuleFileVersion(nullptr, &ma, &mi));
        FgAsiTransport t;
        FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
        cfg.imports.version_minor = 2;
        CHECK(t.install(cfg) == kFgVersionMismatch);
        CHECK(t.patch_count() == 0 && SlotsAreGenuine(game));
        cfg = MakeConfig(&codec, &owners, true);
        cfg.read_version = [](HMODULE, uint32_t*, uint32_t*) { return false; };
        CHECK(t.install(cfg) == kFgVersionUnreadable);
        CHECK(t.patch_count() == 0 && SlotsAreGenuine(game));
    }
    {  // the interface not loaded and not delay-loaded by the exe
        FgAsiTransport t;
        FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
        cfg.imports.interposer_leaf = "fg_not_loaded_anywhere.dll";
        CHECK(t.install(cfg) == kFgInterfaceNotFound);
        CHECK(t.patch_count() == 0);
    }
    {  // the core refuses the chain (a core binding of another version): nothing patched
        FgAsiTransport t;
        FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
        cfg.core.binding.version_minor = 9;
        CHECK(t.install(cfg) == kFgAttachRefused);
        CHECK(t.patch_count() == 0 && SlotsAreGenuine(game));
    }
    {  // the scan out of its budget: fails open, nothing patched, every call still reaches the genuine interface
        FgAsiTransport t;
        FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
        cfg.scan_max_entries = 3;
        CHECK(t.install(cfg) == kFgScanOverBudget);
        CHECK(t.patch_count() == 0 && !t.active() && SlotsAreGenuine(game));
        CHECK(t.stats().scan_entries > 3);
        fgmock_reset();
        SetMockResults();
        CheckEveryCallPassesThrough(game, kFeatFg);
    }
    {  // no role named: nothing to patch
        FgAsiTransport t;
        FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
        for (auto& n : cfg.imports.role_names) n = "fgmock_not_exported";
        CHECK(t.install(cfg) == kFgNothingToPatch);
        CHECK(t.patch_count() == 0);
    }
}

void TestRoutingAndCoexistence(const GameModule& game, const GameModule& delay) {
    fgmock_reset();
    SetMockResults();
    MockCodec codec;
    FgPresentOwners owners;
    FgAsiTransport t;
    g_log.clear();
    const auto delay_before = delay.slots();
    FgAsiConfig cfg = MakeConfig(&codec, &owners, true);
    CHECK(t.install(cfg) == kFgInstalled);
    CHECK(t.active() && t.core() != nullptr && !t.interface_loaded_by_us());
    // exactly the eight role slots of each importer: the static game's import table and the delay game's delay table
    size_t in_game = 0, in_delay = 0, other = 0;
    for (const FgPatch& p : t.patches()) {
        if (p.module == game.h && !p.delay_load) ++in_game;
        else if (p.module == delay.h && p.delay_load) ++in_delay;
        else ++other;
        CHECK(p.handler == FgAsiTransport::handler_for(p.role));
    }
    CHECK(in_game == kFgRoleCount && in_delay == kFgRoleCount && other == 0);
    CHECK(SlotsAreOurs(game) && SlotsAreOurs(delay));
    const auto gv = game.slots();
    CHECK(gv[8] == GenuineExport("fgmock_init"));  // not a hooked role: never redirected
    for (uint32_t i = 0; i < kFgRoleCount; ++i) CHECK(t.genuine(i) == GenuineExport(kRoleNames[i]));
    // the delay game's slots held its own delay thunks before (not yet bound)
    for (uint32_t i = 0; i < kFgRoleCount; ++i) CHECK(delay_before[i] != GenuineExport(kRoleNames[i]));

    // not taken yet (no supported adapter): every call, FG included, reaches the genuine interface unchanged
    CHECK(t.core()->route_reason() == WG_FG_REASON_UNSUPPORTED_ADAPTER);
    CheckEveryCallPassesThrough(game, kFeatFg);
    CheckEveryCallPassesThrough(delay, kFeatFg);
    for (uint32_t f : {kFeatSr, kFeatRr, kFeatLatency, kFeatPcLatency}) CheckEveryCallPassesThrough(game, f);
    void* fn = nullptr;
    CHECK(game.get_function(kFeatFg, FG_MOCK_FN_FG_OPTIONS, &fn) == fgmock_record()->result[kMockGetFunction]);
    CHECK(fn == fgmock_feature_fn(kMockFgOptions));  // the genuine FG feature function while not taken

    // taken: FG answered by the core, everything else unchanged
    MakeTaken(t.core());
    CHECK(t.core()->fg_taken());
    FgMockRecord* r = fgmock_record();
    const uint32_t calls_is = r->calls[kMockIsSupported];
    LUID luid = {77, 0};
    WgAdapterInfo adapter_info = {};
    adapter_info.luid = &luid;
    adapter_info.luid_size = sizeof(luid);
    CHECK(game.is_supported(kFeatFg, &adapter_info) == WG_OK);
    CHECK(r->calls[kMockIsSupported] == calls_is);  // the genuine FG feature never saw it
    LUID other_luid = {78, 0};
    adapter_info.luid = &other_luid;
    CHECK(game.is_supported(kFeatFg, &adapter_info) == WG_E_ADAPTER_NOT_SUPPORTED);  // the truthful adapter only
    CHECK(game.is_supported(kFeatFg, nullptr) == WG_OK);
    CHECK(r->calls[kMockIsSupported] == calls_is);

    // get-function: our FG feature functions for the binding's names, a refusal for unknown FG names, SR unchanged
    fn = nullptr;
    CHECK(game.get_function(kFeatFg, FG_MOCK_FN_FG_OPTIONS, &fn) == WG_OK && fn == FgAsiTransport::options_handler());
    void* fn_state = nullptr;
    CHECK(game.get_function(kFeatFg, FG_MOCK_FN_FG_STATE, &fn_state) == WG_OK && fn_state == FgAsiTransport::state_handler());
    void* fn_unknown = nullptr;
    CHECK(game.get_function(kFeatFg, "fgmock_fn_unknown", &fn_unknown) == WG_E_FEATURE_MISSING && !fn_unknown);
    void* fn_sr = nullptr;
    CHECK(game.get_function(kFeatSr, FG_MOCK_FN_SR_OPTIONS, &fn_sr) == r->result[kMockGetFunction] &&
          fn_sr == fgmock_feature_fn(kMockSrOptions));

    // FG options and state through the handed-out functions, decoded by the codec, decided by the core
    const auto options_fn = reinterpret_cast<FgOptionsFn>(fn);
    const auto state_fn = reinterpret_cast<FgStateFn>(fn_state);
    WgViewport vp = {};
    vp.id = 4;
    const uint32_t fg_opt_calls = r->calls[kMockFgOptions];
    MockOptions mo = {1, WG_FG_PRESET_B, 3};
    CHECK(options_fn(&vp, &mo) == WG_OK);
    CHECK(r->calls[kMockFgOptions] == fg_opt_calls);  // the genuine FG options never saw it
    WgFgState st = t.core()->state();
    CHECK(st.enabled == 1 && st.effective_preset == WG_FG_PRESET_B && st.effective_count == 3);
    MockState ms = {};
    CHECK(state_fn(&vp, &ms, &mo) == WG_OK);
    CHECK(ms.taken == 1 && ms.enabled == 1 && ms.preset == WG_FG_PRESET_B && ms.count == 3 && ms.max_generated == 5);
    CHECK(r->calls[kMockFgState] == 0);
    mo.count = 9;  // outside 1..5: clamped by the core to the cap
    CHECK(options_fn(&vp, &mo) == WG_OK && t.core()->state().effective_count == 5);
    mo.preset = 7;  // not a preset: refused by the core
    CHECK(options_fn(&vp, &mo) == WG_E_INVALID_PARAMETER);
    CHECK(options_fn(&vp, nullptr) == WG_E_MISSING_INPUT_PARAMETER);
    CHECK(state_fn(&vp, nullptr, nullptr) == WG_E_MISSING_INPUT_PARAMETER);

    // requirements: FG answered by the codec from the core's capability, the latency feature unchanged
    WgRequirements rq = {};
    const uint32_t calls_rq = r->calls[kMockRequirements];
    CHECK(game.requirements(kFeatFg, &rq) == WG_OK && rq.flags == 2u && rq.required_tag_count == 3u);
    CHECK(r->calls[kMockRequirements] == calls_rq);
    CHECK(game.requirements(kFeatFg, nullptr) == WG_E_MISSING_INPUT_PARAMETER);
    CHECK(game.requirements(kFeatLatency, &rq) == r->result[kMockRequirements] && r->calls[kMockRequirements] == calls_rq + 1);

    // tags and constants: observed for FG while taken, and always forwarded unchanged (shared by every feature)
    WgTag tags[3] = {};
    tags[0].buffer_type = 900;  // backbuffer role in the synthetic binding
    tags[0].resource = reinterpret_cast<WgResource*>(&tags[0]);
    tags[1].buffer_type = 901;
    tags[1].resource = reinterpret_cast<WgResource*>(&tags[1]);
    tags[2].buffer_type = WG_BT_NORMALS;  // an RR input: not FG's
    tags[2].resource = reinterpret_cast<WgResource*>(&tags[2]);
    void* cmd = &rq;
    codec.token_value = 5;
    const uint32_t calls_tag = r->calls[kMockTag];
    CHECK(game.tag(&vp, tags, 3, cmd) == r->result[kMockTag]);
    CHECK(r->calls[kMockTag] == calls_tag + 1 && r->last[kMockTag].a == &vp && r->last[kMockTag].b == tags &&
          r->last[kMockTag].c == cmd && r->last[kMockTag].count == 3);
    CHECK(codec.decoded_tags == 3 && t.core()->frames_kept() == 1);
    const WgFrameToken* token = reinterpret_cast<const WgFrameToken*>(&mo);
    CHECK(delay.tag_for_frame(token, &vp, tags, 3, cmd) == r->result[kMockTagForFrame]);
    CHECK(r->last[kMockTagForFrame].a == token && r->last[kMockTagForFrame].b == &vp && r->last[kMockTagForFrame].c == tags &&
          r->last[kMockTagForFrame].d == cmd && r->last[kMockTagForFrame].count == 3);
    WgConstants cs = {};
    cs.jitter[0] = 0.25f;
    CHECK(game.constants(&cs, token, &vp) == r->result[kMockConstants]);
    CHECK(r->last[kMockConstants].a == &cs && r->last[kMockConstants].b == token && r->last[kMockConstants].c == &vp);

    // evaluate / free: FG acknowledged (present-driven), the others forwarded with their arguments
    const uint32_t calls_ev = r->calls[kMockEvaluate], calls_fr = r->calls[kMockFree];
    CHECK(game.evaluate(kFeatFg, token, nullptr, 0, cmd) == WG_OK && r->calls[kMockEvaluate] == calls_ev);
    CHECK(game.free_(kFeatFg, &vp) == WG_OK && r->calls[kMockFree] == calls_fr);
    CHECK(game.evaluate(kFeatSr, token, nullptr, 0, cmd) == r->result[kMockEvaluate] && r->calls[kMockEvaluate] == calls_ev + 1);
    CHECK(r->last[kMockEvaluate].feature == kFeatSr && r->last[kMockEvaluate].a == token && r->last[kMockEvaluate].c == cmd);
    CHECK(game.free_(kFeatRr, &vp) == r->result[kMockFree] && r->calls[kMockFree] == calls_fr + 1);
    for (uint32_t f : {kFeatSr, kFeatRr, kFeatLatency, kFeatPcLatency}) CheckEveryCallPassesThrough(game, f);

    // a non-role import goes to the genuine export (the delay helper binds that one slot only)
    const uint32_t calls_init = r->calls[kMockInit];
    CHECK(delay.init(11) == r->result[kMockInit] && r->calls[kMockInit] == calls_init + 1);
    CHECK(SlotsAreOurs(delay));
    const FgTransportStats s = t.stats();
    CHECK(s.fg_answered > 0 && s.option_calls >= 4 && s.state_calls >= 2 && s.codec_failures == 0);

    // the single present owner seen from the core: the native adapter can claim, a second presenter cannot
    CHECK(t.core()->claim_present(91, WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
    CHECK(owners.claim(91, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_PRESENT_OWNED);

    // double init: the same transport and a second one are both refused; nothing changes
    const size_t n = t.patch_count();
    CHECK(t.install(cfg) == kFgAlreadyActive && t.patch_count() == n && t.active());
    FgAsiTransport second;
    CHECK(second.install(cfg) == kFgAlreadyActive && second.patch_count() == 0 && !second.active());
    CHECK(SlotsAreOurs(game));

    // restore: every slot back, routing off; a late call through a kept handler pointer still forwards
    t.restore();
    CHECK(!t.active() && t.patch_count() == 0 && t.core() == nullptr);
    CHECK(SlotsAreGenuine(game));
    CHECK(owners.owner_of(91) == WG_FG_OWNER_NONE);  // the core's claim went with it
    const auto late = reinterpret_cast<FgIsSupportedFn>(FgAsiTransport::handler_for(kFgRoleIsSupported));
    const uint32_t calls_late = r->calls[kMockIsSupported];
    CHECK(late(kFeatFg, &adapter_info) == r->result[kMockIsSupported] && r->calls[kMockIsSupported] == calls_late + 1);
    r->result[kMockGetFunction] = WG_OK;  // the genuine lookup of the FG function succeeds as a real one would
    CHECK(options_fn(&vp, &mo) == r->result[kMockFgOptions]);  // a kept FG function pointer: the genuine one now
    CHECK(r->last[kMockFgOptions].a == &vp && r->last[kMockFgOptions].b == &mo);
    r->result[kMockGetFunction] = 42;
    CheckEveryCallPassesThrough(game, kFeatFg);
    // a second install after restore works again
    CHECK(second.install(cfg) == kFgInstalled && SlotsAreOurs(game));
    second.restore();
    CHECK(SlotsAreGenuine(game));
}

void TestThrowingCodecAndNulls(const GameModule& game) {
    fgmock_reset();
    SetMockResults();
    MockCodec codec;
    FgPresentOwners owners;
    FgAsiTransport t;
    CHECK(t.install(MakeConfig(&codec, &owners, true)) == kFgInstalled);
    MakeTaken(t.core());
    FgMockRecord* r = fgmock_record();
    void* fn = nullptr;
    CHECK(game.get_function(kFeatFg, FG_MOCK_FN_FG_OPTIONS, &fn) == WG_OK);
    const auto options_fn = reinterpret_cast<FgOptionsFn>(fn);
    void* sfn = nullptr;
    CHECK(game.get_function(kFeatFg, FG_MOCK_FN_FG_STATE, &sfn) == WG_OK);
    const auto state_fn = reinterpret_cast<FgStateFn>(sfn);
    WgViewport vp = {};
    MockOptions mo = {1, 0, 1};
    MockState ms = {};

    // a throwing codec: the call never escapes; FG options/state answer an error, never the genuine FG feature
    codec.throw_in = MockCodec::kOptions;
    CHECK(options_fn(&vp, &mo) == WG_E_INVALID_PARAMETER && r->calls[kMockFgOptions] == 0);
    codec.throw_in = MockCodec::kState;
    CHECK(state_fn(&vp, &ms, nullptr) == WG_E_INVALID_PARAMETER && r->calls[kMockFgState] == 0);
    codec.throw_in = MockCodec::kRequirements;
    WgRequirements rq = {};
    CHECK(game.requirements(kFeatFg, &rq) == WG_E_INVALID_PARAMETER && r->calls[kMockRequirements] == 0);
    // tags / constants: a throw before the forward forwards exactly once
    codec.throw_in = MockCodec::kTags;
    WgTag tags[1] = {};
    tags[0].buffer_type = 900;
    tags[0].resource = reinterpret_cast<WgResource*>(&tags[0]);
    CHECK(game.tag(&vp, tags, 1, nullptr) == r->result[kMockTag] && r->calls[kMockTag] == 1);
    codec.throw_in = MockCodec::kConstants;
    WgConstants cs = {};
    CHECK(game.constants(&cs, nullptr, &vp) == r->result[kMockConstants] && r->calls[kMockConstants] == 1);
    codec.throw_in = MockCodec::kNone;
    CHECK(t.stats().codec_failures == 5);
    // a failing (not throwing) codec: tags still forward, FG options refused by the core (nothing decoded)
    codec.fail_in = MockCodec::kTags;
    CHECK(game.tag(&vp, tags, 1, nullptr) == r->result[kMockTag] && r->calls[kMockTag] == 2);
    codec.fail_in = MockCodec::kOptions;
    CHECK(options_fn(&vp, &mo) == WG_E_MISSING_INPUT_PARAMETER && r->calls[kMockFgOptions] == 0);
    codec.fail_in = MockCodec::kNone;

    // null arguments pass through as the game gave them
    CHECK(game.tag(nullptr, nullptr, 0, nullptr) == r->result[kMockTag]);
    CHECK(r->last[kMockTag].a == nullptr && r->last[kMockTag].b == nullptr && r->last[kMockTag].count == 0);
    CHECK(game.constants(nullptr, nullptr, nullptr) == r->result[kMockConstants]);
    CHECK(r->last[kMockConstants].a == nullptr);
    CHECK(game.get_function(kFeatFg, nullptr, nullptr) == WG_E_INVALID_PARAMETER);
    CHECK(game.is_supported(kFeatSr, nullptr) == r->result[kMockIsSupported] && r->last[kMockIsSupported].a == nullptr);
    WgAdapterInfo short_luid = {};
    LUID l = {77, 0};
    short_luid.luid = &l;
    short_luid.luid_size = 4;  // too small to be a LUID: not decoded (no LUID check), answered by the core
    CHECK(game.is_supported(kFeatFg, &short_luid) == WG_OK);
    t.restore();
    CHECK(SlotsAreGenuine(game));
    // install(null codec) and restore() on an idle transport are harmless
    FgAsiTransport idle;
    idle.restore();
    CHECK(!idle.active());
}

// ---- fake DXGI objects ---------------------------------------------------------------------------------------------

class FakeSwapChain final : public IDXGISwapChain4 {
public:
    long refs = 1;
    bool destroyed = false;
    std::vector<HRESULT> results;  // consumed by Present / Present1 in order; empty: S_OK
    uint32_t presents = 0, present1s = 0, tests = 0, resizes = 0, resize1s = 0, fullscreens = 0, colorspaces = 0,
             getbuffers = 0, indexes = 0, waitables = 0;
    UINT last_sync = 99, last_flags = 99;
    const DXGI_PRESENT_PARAMETERS* last_params = nullptr;
    HRESULT resize_result = S_OK, fullscreen_result = S_OK, colorspace_result = S_OK;
    UINT backbuffer_index = 2;
    HANDLE waitable = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(4242));

    HRESULT Next() {
        if (results.empty()) return S_OK;
        const HRESULT hr = results.front();
        results.erase(results.begin());
        return hr;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
            riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) || riid == __uuidof(IDXGISwapChain2) ||
            riid == __uuidof(IDXGISwapChain3) || riid == __uuidof(IDXGISwapChain4)) {
            *ppv = static_cast<IDXGISwapChain4*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(++refs); }
    ULONG STDMETHODCALLTYPE Release() override {
        const long n = --refs;
        if (n == 0) destroyed = true;
        return static_cast<ULONG>(n);
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE Present(UINT sync, UINT flags) override {
        last_sync = sync;
        last_flags = flags;
        last_params = nullptr;
        if (flags & DXGI_PRESENT_TEST) {
            ++tests;
            return DXGI_STATUS_OCCLUDED;
        }
        ++presents;
        return Next();
    }
    HRESULT STDMETHODCALLTYPE GetBuffer(UINT, REFIID, void** s) override {
        ++getbuffers;
        if (s) *s = nullptr;
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL, IDXGIOutput*) override {
        ++fullscreens;
        return fullscreen_result;
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* f, IDXGIOutput** t) override {
        if (f) *f = FALSE;
        if (t) *t = nullptr;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT, UINT, UINT, DXGI_FORMAT, UINT) override {
        ++resizes;
        return resize_result;
    }
    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput**) override { return DXGI_ERROR_UNSUPPORTED; }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS*) override { return DXGI_ERROR_FRAME_STATISTICS_DISJOINT; }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* c) override {
        if (c) *c = presents;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* h) override {
        if (h) *h = nullptr;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID, void**) override { return DXGI_ERROR_INVALID_CALL; }
    HRESULT STDMETHODCALLTYPE Present1(UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) override {
        last_sync = sync;
        last_flags = flags;
        last_params = p;
        if (flags & DXGI_PRESENT_TEST) {
            ++tests;
            return DXGI_STATUS_OCCLUDED;
        }
        ++present1s;
        return Next();
    }
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** o) override {
        if (o) *o = nullptr;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT*, UINT*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT*) override { return S_OK; }
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override {
        ++waitables;
        return waitable;
    }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F*) override { return S_OK; }
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override {
        ++indexes;
        return backbuffer_index;
    }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE, UINT* s) override {
        if (s) *s = 1;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE) override {
        ++colorspaces;
        return colorspace_result;
    }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT*, IUnknown* const*) override {
        ++resize1s;
        return resize_result;
    }
    HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE, UINT, void*) override { return S_OK; }
};

// The presenter behind the seam: N generated presents then the game image, all through the inner presenter.
class TestCallback final : public IFgPresentCallback {
public:
    uint32_t generated = 2;
    uint32_t begins = 0, presents = 0, retries = 0, before_resize = 0, after_resize = 0, before_fs = 0, after_fs = 0,
             colorspace = 0, detached = 0;
    HRESULT last_after_resize = E_FAIL;
    std::vector<std::string> order;
    bool throw_in_begin = false, throw_before_present = false, throw_after_present = false;
    IDXGISwapChain4* reenter = nullptr;  // present through the game-facing pointer from inside the callback
    bool serve_facade = false;
    FgPresentCall last_call;

    void BeginInterval(const FgPresentCall& call) override {
        if (throw_in_begin) throw std::runtime_error("synthetic presenter failure");
        ++begins;
        last_call = call;
        order.push_back("begin");
    }
    HRESULT PresentInterval(const FgPresentCall& call, IFgInnerPresenter& inner) override {
        ++presents;
        if (call.retry) ++retries;
        last_call = call;
        if (throw_before_present) throw std::runtime_error("synthetic presenter failure");
        if (!call.retry)
            for (uint32_t i = 0; i < generated; ++i) inner.Present(0, 0);  // the generated images (synthetic)
        if (reenter) reenter->Present(0, 0);
        const HRESULT hr = call.present1 ? inner.Present1(call.sync_interval, call.flags, call.params)
                                         : inner.Present(call.sync_interval, call.flags);
        if (throw_after_present) throw std::runtime_error("synthetic presenter failure");
        return hr;
    }
    void BeforeResize() override {
        ++before_resize;
        order.push_back("before-resize");
    }
    void AfterResize(HRESULT hr) override {
        ++after_resize;
        last_after_resize = hr;
        order.push_back("after-resize");
    }
    void BeforeFullscreen(BOOL) override { ++before_fs; }
    void AfterFullscreen(BOOL, HRESULT) override { ++after_fs; }
    void ColorSpaceChanged(DXGI_COLOR_SPACE_TYPE) override { ++colorspace; }
    bool GetBuffer(UINT buffer, REFIID, void** out, HRESULT* hr) override {
        if (!serve_facade) return false;
        if (out) *out = reinterpret_cast<void*>(static_cast<uintptr_t>(100 + buffer));
        *hr = S_FALSE;
        return true;
    }
    bool CurrentBackBufferIndex(UINT* i) override {
        if (!serve_facade) return false;
        *i = 7;
        return true;
    }
    void Detached() override { ++detached; }
};

FgSwapChain* MakeWrapper(FakeSwapChain* fake) {
    FgSwapChain* w = nullptr;
    CHECK(FgSwapChain::Create(fake, nullptr, &CollectLog, nullptr, &w) == S_OK && w);
    return w;
}

void TestSwapChainIdentityAndRefcount() {
    const long live0 = FgSwapChain::LiveCount();
    FakeSwapChain fake;
    FgSwapChain* w = MakeWrapper(&fake);
    CHECK(fake.refs == 1);  // the creation reference moved into the wrapper
    CHECK(w->RefCount() == 1 && FgSwapChain::LiveCount() == live0 + 1);
    CHECK(w->Id() == FgSwapChainId(&fake) && w->Id() != 0);
    IDXGISwapChain4* self = static_cast<IDXGISwapChain4*>(w);
    const IID iids[] = {__uuidof(IUnknown),       __uuidof(IDXGIObject),     __uuidof(IDXGIDeviceSubObject),
                        __uuidof(IDXGISwapChain), __uuidof(IDXGISwapChain1), __uuidof(IDXGISwapChain2),
                        __uuidof(IDXGISwapChain3), __uuidof(IDXGISwapChain4)};
    for (const IID& iid : iids) {
        void* p = nullptr;
        CHECK(w->QueryInterface(iid, &p) == S_OK && p == self);
        if (p) static_cast<IUnknown*>(p)->Release();
    }
    IUnknown* id1 = nullptr;
    IDXGISwapChain1* sc1 = nullptr;
    CHECK(w->QueryInterface(__uuidof(IDXGISwapChain1), reinterpret_cast<void**>(&sc1)) == S_OK);
    CHECK(sc1 && sc1->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void**>(&id1)) == S_OK && id1 == self);
    if (id1) id1->Release();
    if (sc1) sc1->Release();
    void* other = reinterpret_cast<void*>(1);
    CHECK(w->QueryInterface(__uuidof(IDXGIFactory), &other) == E_NOINTERFACE && other == nullptr);
    CHECK(w->QueryInterface(__uuidof(IUnknown), nullptr) == E_POINTER);
    CHECK(FgSwapChain::FromInterface(self) == w);
    CHECK(FgSwapChain::FromInterface(&fake) == nullptr);
    CHECK(FgSwapChain::FromInterface(nullptr) == nullptr);
    CHECK(w->RefCount() == 1);
    // forwarded calls
    UINT count = 0;
    CHECK(w->GetLastPresentCount(&count) == S_OK);
    CHECK(w->GetFrameStatistics(nullptr) == DXGI_ERROR_FRAME_STATISTICS_DISJOINT);
    CHECK(w->GetFrameLatencyWaitableObject() == fake.waitable && fake.waitables == 1);
    CHECK(w->GetCurrentBackBufferIndex() == 2 && fake.indexes == 1);
    // the last release: refcount 0 on both sides, the wrapper gone
    CHECK(w->Release() == 0);
    CHECK(fake.refs == 0 && fake.destroyed && FgSwapChain::LiveCount() == live0);
    // a chain that is not an IDXGISwapChain4 is not wrapped and keeps its reference
    FgSwapChain* none = nullptr;
    CHECK(FgSwapChain::Create(nullptr, nullptr, nullptr, nullptr, &none) == E_POINTER && none == nullptr);
    CHECK(FgSwapChain::Create(nullptr, nullptr, nullptr, nullptr, nullptr) == E_POINTER);
}

void TestPresentPaths() {
    FakeSwapChain fake;
    FgSwapChain* w = MakeWrapper(&fake);
    IDXGISwapChain4* game = static_cast<IDXGISwapChain4*>(w);
    FgPresentOwners owners;

    // no callback: straight through, flags and result as they are
    fake.results = {DXGI_STATUS_OCCLUDED};
    CHECK(game->Present(1, DXGI_PRESENT_DO_NOT_WAIT) == DXGI_STATUS_OCCLUDED);
    CHECK(fake.presents == 1 && fake.last_sync == 1 && fake.last_flags == DXGI_PRESENT_DO_NOT_WAIT);
    CHECK(w->Stats().forwarded_presents == 1);

    // single owner: another presenter holds the chain, the attach is refused and the chain stays pass-through
    TestCallback cb;
    CHECK(owners.claim(w->Id(), WG_FG_OWNER_NATIVE_ADAPTER) == WG_FG_REASON_NONE);
    CHECK(w->Attach(&cb, nullptr, &owners) == WG_FG_REASON_PRESENT_OWNED && !w->Attached());
    CHECK(game->Present(0, 0) == S_OK && cb.begins == 0 && fake.presents == 2);
    CHECK(owners.release(w->Id(), WG_FG_OWNER_NATIVE_ADAPTER));
    CHECK(w->Attach(&cb, nullptr, &owners) == WG_FG_REASON_NONE && w->Attached());
    CHECK(owners.owner_of(w->Id()) == WG_FG_OWNER_SWAPCHAIN_WRAPPER);
    CHECK(owners.claim(w->Id(), WG_FG_OWNER_HOST_WRAPPER) == WG_FG_REASON_PRESENT_OWNED);
    TestCallback cb2;
    CHECK(w->Attach(&cb2, nullptr, &owners) == WG_FG_REASON_DOUBLE_ATTACH);
    CHECK(w->Attach(nullptr, nullptr, &owners) == WG_FG_REASON_NULL_ARGUMENT);

    // a test present: straight through, never an interval
    const uint32_t p0 = fake.presents;
    CHECK(game->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED);
    CHECK(fake.tests == 1 && fake.presents == p0 && cb.begins == 0 && cb.presents == 0);
    CHECK(fake.last_flags == DXGI_PRESENT_TEST);

    // one interval: one BeginInterval, N generated + the game image through the inner presenter
    CHECK(game->Present(1, 0) == S_OK);
    CHECK(cb.begins == 1 && cb.presents == 1 && fake.presents == p0 + 3);
    CHECK(cb.last_call.interval == w->Stats().game_presents && !cb.last_call.retry && cb.last_call.sync_interval == 1);

    // a retry: DXGI_ERROR_WAS_STILL_DRAWING keeps the interval open; the retry presents again without a second
    // BeginInterval (no history rotation, no second generation submit)
    fake.results = {S_OK, S_OK, DXGI_ERROR_WAS_STILL_DRAWING};
    const uint32_t p1 = fake.presents;
    CHECK(game->Present(0, DXGI_PRESENT_DO_NOT_WAIT) == DXGI_ERROR_WAS_STILL_DRAWING);
    CHECK(cb.begins == 2 && fake.presents == p1 + 3);
    const uint64_t interval = cb.last_call.interval;
    CHECK(game->Present(0, DXGI_PRESENT_DO_NOT_WAIT) == S_OK);
    CHECK(cb.begins == 2 && cb.retries == 1 && cb.last_call.retry && cb.last_call.interval == interval);
    CHECK(fake.presents == p1 + 4);  // only the game image again: the generated images were not submitted twice
    CHECK(fake.last_flags == DXGI_PRESENT_DO_NOT_WAIT);
    CHECK(w->Stats().retries == 1);

    // occlusion and device errors reach the game unchanged and close the interval
    fake.results = {S_OK, S_OK, DXGI_STATUS_OCCLUDED};
    CHECK(game->Present(0, 0) == DXGI_STATUS_OCCLUDED && cb.begins == 3);
    fake.results = {S_OK, S_OK, DXGI_ERROR_DEVICE_REMOVED};
    CHECK(game->Present(0, 0) == DXGI_ERROR_DEVICE_REMOVED && cb.begins == 4);
    CHECK(game->Present(0, 0) == S_OK && cb.begins == 5 && !cb.last_call.retry);

    // Present1: the parameters pointer and flags reach the real chain as given
    DXGI_PRESENT_PARAMETERS pp = {};
    const uint32_t q0 = fake.present1s;
    CHECK(game->Present1(2, DXGI_PRESENT_ALLOW_TEARING, &pp) == S_OK);
    CHECK(cb.last_call.present1 && cb.last_call.params == &pp && fake.present1s == q0 + 1);
    CHECK(fake.last_params == &pp && fake.last_flags == DXGI_PRESENT_ALLOW_TEARING && fake.last_sync == 2);
    CHECK(game->Present1(0, DXGI_PRESENT_TEST, &pp) == DXGI_STATUS_OCCLUDED && fake.present1s == q0 + 1);

    // re-entrancy: a present through the game-facing pointer inside the callback goes straight through, once
    cb.reenter = game;
    const uint32_t p2 = fake.presents;
    const uint32_t b2 = cb.begins;
    CHECK(game->Present(0, 0) == S_OK);
    CHECK(cb.begins == b2 + 1 && fake.presents == p2 + 4 && w->Stats().bypass_presents == 1);
    cb.reenter = nullptr;

    // resize: before / after around the real call, its result unchanged, the epoch bumped on success only, and an
    // open interval closed
    fake.results = {S_OK, S_OK, DXGI_ERROR_WAS_STILL_DRAWING};
    CHECK(game->Present(0, DXGI_PRESENT_DO_NOT_WAIT) == DXGI_ERROR_WAS_STILL_DRAWING);
    const uint64_t e0 = w->Epoch();
    cb.order.clear();
    CHECK(game->ResizeBuffers(0, 64, 32, DXGI_FORMAT_UNKNOWN, 0) == S_OK);
    CHECK(fake.resizes == 1 && w->Epoch() == e0 + 1 && cb.last_after_resize == S_OK);
    CHECK(cb.order.size() == 2 && cb.order[0] == "before-resize" && cb.order[1] == "after-resize");
    const uint32_t b3 = cb.begins;
    CHECK(game->Present(0, 0) == S_OK && cb.begins == b3 + 1 && !cb.last_call.retry);  // a new interval
    CHECK(cb.last_call.swapchain_epoch == e0 + 1);
    fake.resize_result = DXGI_ERROR_INVALID_CALL;
    CHECK(game->ResizeBuffers(0, 64, 32, DXGI_FORMAT_UNKNOWN, 0) == DXGI_ERROR_INVALID_CALL);
    CHECK(w->Epoch() == e0 + 1 && cb.last_after_resize == DXGI_ERROR_INVALID_CALL);
    fake.resize_result = S_OK;
    const UINT masks[2] = {1, 1};
    IUnknown* queues[2] = {nullptr, nullptr};
    CHECK(game->ResizeBuffers1(2, 64, 32, DXGI_FORMAT_UNKNOWN, 0, masks, queues) == S_OK);
    CHECK(fake.resize1s == 1 && w->Epoch() == e0 + 2 && cb.before_resize == 3 && cb.after_resize == 3);
    fake.fullscreen_result = DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
    CHECK(game->SetFullscreenState(TRUE, nullptr) == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE);
    CHECK(cb.before_fs == 1 && cb.after_fs == 1 && w->Epoch() == e0 + 2);
    CHECK(game->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) == S_OK && cb.colorspace == 1);
    fake.colorspace_result = DXGI_ERROR_INVALID_CALL;
    CHECK(game->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) == DXGI_ERROR_INVALID_CALL && cb.colorspace == 1);

    // the facade seam: served by the callback when it wants, forwarded otherwise
    void* buf = nullptr;
    CHECK(game->GetBuffer(1, __uuidof(IUnknown), &buf) == E_NOTIMPL && fake.getbuffers == 1);
    cb.serve_facade = true;
    CHECK(game->GetBuffer(1, __uuidof(IUnknown), &buf) == S_FALSE && buf == reinterpret_cast<void*>(101) &&
          fake.getbuffers == 1);
    CHECK(game->GetCurrentBackBufferIndex() == 7);
    CHECK(game->GetFrameLatencyWaitableObject() == fake.waitable);
    cb.serve_facade = false;
    CHECK(game->GetCurrentBackBufferIndex() == 2);

    // detach: the owner released, the callback told, the chain pass-through again
    w->Detach();
    CHECK(!w->Attached() && cb.detached == 1 && owners.owner_of(w->Id()) == WG_FG_OWNER_NONE);
    const uint32_t b4 = cb.begins;
    CHECK(game->Present(0, 0) == S_OK && cb.begins == b4);
    CHECK(game->Release() == 0 && fake.destroyed);
}

void TestThrowingCallbacks() {
    FakeSwapChain fake;
    FgSwapChain* w = MakeWrapper(&fake);
    IDXGISwapChain4* game = static_cast<IDXGISwapChain4*>(w);
    FgPresentOwners owners;
    {  // BeginInterval throws: detached, owner released, the interval presented once as the game gave it
        TestCallback cb;
        cb.throw_in_begin = true;
        CHECK(w->Attach(&cb, nullptr, &owners) == WG_FG_REASON_NONE);
        fake.results = {DXGI_STATUS_OCCLUDED};
        const uint32_t p = fake.presents;
        CHECK(game->Present(1, 0) == DXGI_STATUS_OCCLUDED);
        CHECK(fake.presents == p + 1 && !w->Attached() && owners.owner_of(w->Id()) == WG_FG_OWNER_NONE);
        CHECK(w->Stats().callback_faults == 1 && cb.detached == 0);
        CHECK(game->Present(0, 0) == S_OK && fake.presents == p + 2);
    }
    {  // PresentInterval throws before presenting: one raw present
        TestCallback cb;
        cb.throw_before_present = true;
        CHECK(w->Attach(&cb, nullptr, &owners) == WG_FG_REASON_NONE);
        const uint32_t p = fake.presents;
        CHECK(game->Present(0, 0) == S_OK && fake.presents == p + 1 && !w->Attached());
    }
    {  // PresentInterval throws after presenting: no second present of that interval
        TestCallback cb;
        cb.throw_after_present = true;
        cb.generated = 1;
        CHECK(w->Attach(&cb, nullptr, &owners) == WG_FG_REASON_NONE);
        const uint32_t p = fake.presents;
        CHECK(game->Present(0, 0) == S_OK && fake.presents == p + 2 && !w->Attached());
        CHECK(w->Stats().callback_faults == 3);
    }
    {  // the core as the owner gate: FG not taken, so the claim is refused and the chain stays pass-through
        FgProxyConfig pc;
        FgProxy core(pc, &owners, nullptr, nullptr);
        TestCallback cb;
        CHECK(w->Attach(&cb, &core, nullptr) == WG_FG_REASON_INSTALL_OFF && !w->Attached());
    }
    {  // the last release with a callback attached: the callback told, the owner released, refcount 0
        TestCallback cb;
        CHECK(w->Attach(&cb, nullptr, &owners) == WG_FG_REASON_NONE);
        const uint64_t id = w->Id();
        CHECK(game->Release() == 0);
        CHECK(cb.detached == 1 && owners.owner_of(id) == WG_FG_OWNER_NONE && fake.refs == 0 && fake.destroyed);
    }
}

class FakeFactory final : public IDXGIFactory2 {
public:
    std::vector<FakeSwapChain*> made;
    HRESULT create_result = S_OK;
    uint32_t creates = 0;
    ~FakeFactory() {
        for (FakeSwapChain* s : made) delete s;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** ppv) override {
        if (ppv) *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE EnumAdapters(UINT, IDXGIAdapter**) override { return DXGI_ERROR_NOT_FOUND; }
    HRESULT STDMETHODCALLTYPE MakeWindowAssociation(HWND, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetWindowAssociation(HWND*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE CreateSwapChain(IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE CreateSoftwareAdapter(HMODULE, IDXGIAdapter**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE EnumAdapters1(UINT, IDXGIAdapter1**) override { return DXGI_ERROR_NOT_FOUND; }
    BOOL STDMETHODCALLTYPE IsCurrent() override { return TRUE; }
    BOOL STDMETHODCALLTYPE IsWindowedStereoEnabled() override { return FALSE; }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForHwnd(IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*,
                                                     IDXGISwapChain1** out) override {
        ++creates;
        if (FAILED(create_result)) return create_result;
        FakeSwapChain* s = new FakeSwapChain();
        made.push_back(s);
        *out = s;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE CreateSwapChainForCoreWindow(IUnknown*, IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                           IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE GetSharedResourceAdapterLuid(HANDLE, LUID*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusWindow(HWND, UINT, DWORD*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterStereoStatusEvent(HANDLE, DWORD*) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE UnregisterStereoStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusWindow(HWND, UINT, DWORD*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE RegisterOcclusionStatusEvent(HANDLE, DWORD*) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE UnregisterOcclusionStatus(DWORD) override {}
    HRESULT STDMETHODCALLTYPE CreateSwapChainForComposition(IUnknown*, const DXGI_SWAP_CHAIN_DESC1*, IDXGIOutput*,
                                                            IDXGISwapChain1**) override {
        return E_NOTIMPL;
    }
};

class TestSink final : public IFgSwapChainSink {
public:
    uint32_t created = 0;
    HWND last_hwnd = nullptr;
    bool throw_it = false;
    TestCallback* attach = nullptr;
    FgPresentOwners* owners = nullptr;
    void Created(FgSwapChain* chain, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1*) override {
        ++created;
        last_hwnd = hwnd;
        if (throw_it) throw std::runtime_error("synthetic sink failure");
        if (attach) chain->Attach(attach, nullptr, owners);
    }
};

// Calls through the method table, as a game does (a volatile pointer: no devirtualised call).
HRESULT CreateThroughTable(IDXGIFactory2* f, IDXGISwapChain1** out) {
    IDXGIFactory2* volatile vf = f;
    DXGI_SWAP_CHAIN_DESC1 desc = {};
    return vf->CreateSwapChainForHwnd(nullptr, reinterpret_cast<HWND>(static_cast<uintptr_t>(64)), &desc, nullptr,
                                      nullptr, out);
}

void TestCreationHook() {
    FakeFactory factory;
    TestSink sink;
    FgPresentOwners owners;
    TestCallback cb;
    sink.attach = &cb;
    sink.owners = &owners;
    CHECK(!FgSwapChainHook::Installed());
    CHECK(!FgSwapChainHook::Install(nullptr, &sink, &CollectLog, nullptr));
    CHECK(FgSwapChainHook::Install(&factory, &sink, &CollectLog, nullptr));
    CHECK(FgSwapChainHook::Installed());
    CHECK(!FgSwapChainHook::Install(&factory, &sink, &CollectLog, nullptr));  // one install per process
    IDXGISwapChain1* sc = nullptr;
    CHECK(CreateThroughTable(&factory, &sc) == S_OK && sc);
    CHECK(factory.creates == 1 && sink.created == 1 && FgSwapChainHook::Wrapped() == 1);
    CHECK(sink.last_hwnd == reinterpret_cast<HWND>(static_cast<uintptr_t>(64)));
    FgSwapChain* w = FgSwapChain::FromInterface(sc);
    CHECK(w != nullptr && w->Attached() && w->Inner() == factory.made[0]);
    CHECK(sc->Present(0, 0) == S_OK && cb.begins == 1 && factory.made[0]->presents == 3);
    CHECK(sc->Release() == 0 && factory.made[0]->refs == 0 && cb.detached == 1);
    // a sink that throws: the chain stays a pass-through wrapper
    sink.throw_it = true;
    IDXGISwapChain1* sc2 = nullptr;
    CHECK(CreateThroughTable(&factory, &sc2) == S_OK && sc2 && FgSwapChain::FromInterface(sc2) != nullptr);
    CHECK(!FgSwapChain::FromInterface(sc2)->Attached());
    if (sc2) sc2->Release();
    // a failed creation: the HRESULT unchanged, nothing wrapped
    factory.create_result = DXGI_ERROR_INVALID_CALL;
    IDXGISwapChain1* sc3 = nullptr;
    CHECK(CreateThroughTable(&factory, &sc3) == DXGI_ERROR_INVALID_CALL && sc3 == nullptr);
    CHECK(FgSwapChainHook::Wrapped() == 2);
    factory.create_result = S_OK;
    // uninstalled: the original slot back, chains created raw
    FgSwapChainHook::Uninstall();
    CHECK(!FgSwapChainHook::Installed());
    IDXGISwapChain1* sc4 = nullptr;
    CHECK(CreateThroughTable(&factory, &sc4) == S_OK && sc4 == factory.made.back());
    CHECK(FgSwapChain::FromInterface(sc4) == nullptr);
    if (sc4) sc4->Release();
}

// ---- the shipped plugin in its own process --------------------------------------------------------------------------

struct WgFgAsiStatus {
    uint32_t struct_size, result, settings_enabled, transport_created, patched_slots, init_calls;
};

int RunAsiLeg(const char* leg) {
    GameModule game;
    CHECK(game.load(L"fg_mock_game.dll"));
    fgmock_reset();
    SetMockResults();
    HMODULE asi = LoadLibraryW((ExeDir() + L"\\WrongGPU-FG.asi").c_str());
    CHECK(asi != nullptr);
    if (!asi || !game.h) return 1;
    auto init = reinterpret_cast<void (*)()>(GetProcAddress(asi, "InitializeASI"));
    auto query = reinterpret_cast<int (*)(WgFgAsiStatus*)>(GetProcAddress(asi, "WrongGPU_FgQuery"));
    auto shutdown = reinterpret_cast<void (*)()>(GetProcAddress(asi, "WrongGPU_FgShutdown"));
    CHECK(init && query && shutdown);
    if (!init || !query || !shutdown) return 1;
    init();
    init();  // a second call changes nothing
    WgFgAsiStatus st = {};
    st.struct_size = sizeof(st);
    CHECK(query(&st) == 1);
    WgFgAsiStatus short_struct = {};
    short_struct.struct_size = 4;
    CHECK(query(&short_struct) == 0 && query(nullptr) == 0);
    CHECK(st.init_calls == 2 && st.patched_slots == 0);
    const std::string l = leg;
    if (l == "off" || l == "default" || l == "invalid") {
        CHECK(st.result == kFgInstallOff && st.settings_enabled == 0 && st.transport_created == 0);
    } else if (l == "on") {
        // fg=on, but this build carries no qualified title binding: a transport, nothing patched
        CHECK(st.result == kFgNoBinding && st.settings_enabled == 1 && st.transport_created == 1);
    } else {
        CHECK(false);
    }
    CHECK(SlotsAreGenuine(game));
    CheckEveryCallPassesThrough(game, kFeatFg);
    CheckEveryCallPassesThrough(game, kFeatSr);
    shutdown();
    CHECK(SlotsAreGenuine(game));
    std::printf("asi leg %s: result %u (%s)\n", leg, st.result, FgInstallResultWord(st.result));
    return 0;
}

#endif  // !FG_TEST_DELAY_EXE

}  // namespace

#if defined(FG_TEST_DELAY_EXE)
// The exe delay-loads the stand-in interface (linker /DELAYLOAD) and calls it directly; nothing has loaded it yet.
void Step(const char* what) {
    std::printf("delay exe: %s\n", what);
    std::fflush(stdout);
}

int RunDelayExe() {
    Step("start");
    CHECK(GetModuleHandleA(kLeaf) == nullptr);
    MockCodec codec;
    FgPresentOwners owners;
    FgAsiTransport t;
    FgAsiConfig off = MakeConfig(&codec, &owners, false);
    CHECK(t.install(off) == kFgInstallOff && GetModuleHandleA(kLeaf) == nullptr);  // fg=off loads nothing
    FgAsiConfig no_load = MakeConfig(&codec, &owners, true);
    no_load.load_delay_linked = false;
    CHECK(t.install(no_load) == kFgInterfaceNotFound && GetModuleHandleA(kLeaf) == nullptr);
    CHECK(t.install(MakeConfig(&codec, &owners, true)) == kFgInstalled);
    CHECK(t.interface_loaded_by_us() && GetModuleHandleA(kLeaf) != nullptr);
    size_t exe_delay = 0;
    for (const FgPatch& p : t.patches())
        if (p.module == GetModuleHandleW(nullptr) && p.delay_load) ++exe_delay;
    CHECK(exe_delay == kFgRoleCount && t.patch_count() == kFgRoleCount);
    for (const auto& l : g_log) Step(l.c_str());
    if (!t.active()) {
        std::printf("%d checks, %d failed\n", g_checks, g_failed);
        return 1;
    }
    Step("installed");
    fgmock_reset();
    Step("reset");
    SetMockResults();
    FgMockRecord* r = fgmock_record();
    WgViewport vp = {};
    // not taken: forwarded unchanged through the exe's own (redirected) delay-load slots
    CHECK(fgmock_is_supported(kFeatFg, nullptr) == r->result[kMockIsSupported] && r->calls[kMockIsSupported] == 1);
    Step("first call");
    CHECK(fgmock_free(kFeatSr, &vp) == r->result[kMockFree] && r->last[kMockFree].a == &vp);
    MakeTaken(t.core());
    CHECK(fgmock_is_supported(kFeatFg, nullptr) == WG_OK && r->calls[kMockIsSupported] == 1);
    CHECK(fgmock_evaluate(kFeatFg, nullptr, nullptr, 0, nullptr) == WG_OK && r->calls[kMockEvaluate] == 0);
    CHECK(fgmock_evaluate(kFeatRr, nullptr, nullptr, 0, nullptr) == r->result[kMockEvaluate] && r->calls[kMockEvaluate] == 1);
    void* out = nullptr;
    CHECK(fgmock_get_function(kFeatFg, FG_MOCK_FN_FG_OPTIONS, &out) == WG_OK && out == FgAsiTransport::options_handler());
    WgRequirements rq = {};
    CHECK(fgmock_requirements(kFeatFg, &rq) == WG_OK && rq.flags == 2u);
    CHECK(fgmock_tag(&vp, nullptr, 0, nullptr) == r->result[kMockTag]);
    CHECK(fgmock_tag_for_frame(nullptr, &vp, nullptr, 0, nullptr) == r->result[kMockTagForFrame]);
    CHECK(fgmock_constants(nullptr, nullptr, &vp) == r->result[kMockConstants]);
    t.restore();
    CHECK(fgmock_is_supported(kFeatFg, nullptr) == r->result[kMockIsSupported]);  // the delay helper binds it now
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

int main() { return RunDelayExe(); }

#else

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--asi") == 0) {
        const int rc = RunAsiLeg(argv[2]);
        std::printf("%d checks, %d failed\n", g_checks, g_failed);
        return rc == 0 && g_failed == 0 ? 0 : 1;
    }
    TestBoundedPeWalk();
    GameModule game, delay;
    CHECK(game.load(L"fg_mock_game.dll"));
    CHECK(delay.load(L"fg_mock_game_delay.dll"));
    if (game.h && delay.h) {
        TestInstallOff(game, delay);
        TestGatesPatchNothing(game);
        TestRoutingAndCoexistence(game, delay);
        TestThrowingCodecAndNulls(game);
    }
    TestSwapChainIdentityAndRefcount();
    TestPresentPaths();
    TestThrowingCallbacks();
    TestCreationHook();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

#endif
