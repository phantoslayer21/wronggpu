// SPDX-License-Identifier: MIT
// Tests of the shared settings block and the settings I/O (clean/wgshared/include/wg_settings_io.h): create / open /
// seed, magic published last, seqlock torn reads, writer election, version / size / slot_count refusals, text area,
// unset slots, precedence, WG_DATA_DIR, comment-preserving ini persistence, the host-ini poll, the pure helpers.
// Windows, no GPU. Every block uses its own mapping name, so the test never touches a real game's block.
#define WG_SETTINGS_IO_IMPLEMENTATION
#include "wg_settings_io.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static int g_checks = 0, g_failures = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(c)) {                                                                  \
            ++g_failures;                                                            \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
        }                                                                            \
    } while (0)

static std::wstring g_dir;  // per-run scratch folder

static std::wstring Scratch(const wchar_t* leaf) { return g_dir + L"\\" + leaf; }

static std::wstring UniqueName(const char* tag) {
    static int n = 0;
    wchar_t buf[128];
    swprintf(buf, 128, L"Local\\WrongGPU.settings.test.%lu.%hs.%d", static_cast<unsigned long>(GetCurrentProcessId()), tag, ++n);
    return buf;
}

static void WriteText(const std::wstring& path, const std::string& text) { CHECK(wg::write_file_atomic(path, text)); }

static std::string ReadText(const std::wstring& path) {
    std::string t;
    wg::read_file(path, &t);
    return t;
}

static void TestPure() {
    CHECK(wg::describe(WG_OPT_RR_MODE, WG_RR_MODE_FAST, wg::Source::Menu) == "rr_mode=fast (menu)");
    CHECK(wg::describe(WG_OPT_SR_MODEL, WG_SR_MODEL_K, wg::Source::Ini) == "sr_model=K (ini)");
    CHECK(wg::describe(WG_OPT_RR_ENABLED, 1, wg::Source::Default) == "rr_enabled=1 (default)");
    CHECK(wg_describe(WG_OPT_RR_MODE, WG_RR_MODE_EXACT, wg::Source::HostIni) == "rr_mode=exact (optiscaler.ini)");
    int32_t v = -1;
    CHECK(wg::parse_value(WG_OPT_RR_MODE, " Fast ", &v) && v == WG_RR_MODE_FAST);
    CHECK(wg::parse_value(WG_OPT_RR_MODE, "0", &v) && v == WG_RR_MODE_EXACT);
    CHECK(!wg::parse_value(WG_OPT_RR_MODE, std::to_string(WG_RR_MODE_COUNT), &v));  // first number past the table (table-driven: an extended registry has more modes)
    CHECK(!wg::parse_value(WG_OPT_RR_MODE, "quick", &v));
    CHECK(wg::parse_value(WG_OPT_RR_ENABLED, "off", &v) && v == 0);
    CHECK(wg::parse_value(WG_OPT_RR_ENABLED, "true", &v) && v == 1);
    CHECK(!wg::parse_value(WG_OPT_RR_ENABLED, "2", &v));
    CHECK(wg::slot_of_key("RR_MODE") == WG_OPT_RR_MODE && wg::slot_of_key("nope") == -1);
    CHECK(wg::is_registry_key("diag_env_3") && wg::is_registry_key("sr_model") && !wg::is_registry_key("upscaler"));
    CHECK(wg::reader_mask(WG_READER_SR_FRONTEND) == ((1ull << WG_OPT_SR_MODEL) | (1ull << WG_OPT_SR_FALLBACK_TO_AMD) | (1ull << WG_OPT_ENTRY_SOURCE)));
    CHECK(wg::entry_of_id(0x5747000000000002ull) && wg::entry_of_id(0x5747000000000002ull)->slot == WG_OPT_SR_MODEL);
    CHECK(!wg::entry_of_id(0x5747000000000103ull));  // i20: the RR modes are no version-list entries (RR plugin's box)
    CHECK(wg::entry_index_of_id(0x5747000000000001ull) == 0 && wg::entry_index_of_id(0x1234) == -1);

    // ini reader / writer on text
    const std::string text = "# head comment\r\n[other]\r\nrr_mode = exact\r\n[WrongGPU]\r\n; note\r\nrr_mode = accurate   ; keep me\r\nunknown_key = 7\r\n\r\n[fsr]\r\nupscaler = network\r\n";
    const wg::IniValues ini = wg::parse_ini(text);
    std::string s;
    CHECK(ini.get("wronggpu", "RR_MODE", &s) && s == "accurate");
    CHECK(ini.get("other", "rr_mode", &s) && s == "exact");
    CHECK(ini.get("fsr", "upscaler", &s) && s == "network");
    const std::string set1 = wg::ini_set(text, "wronggpu", "rr_mode", "fast");
    CHECK(set1 == "# head comment\r\n[other]\r\nrr_mode = exact\r\n[WrongGPU]\r\n; note\r\nrr_mode = fast   ; keep me\r\nunknown_key = 7\r\n\r\n[fsr]\r\nupscaler = network\r\n");
    const std::string set2 = wg::ini_set(text, "wronggpu", "sr_model", "K");
    CHECK(set2 == "# head comment\r\n[other]\r\nrr_mode = exact\r\n[WrongGPU]\r\n; note\r\nrr_mode = accurate   ; keep me\r\nunknown_key = 7\r\nsr_model = K\r\n\r\n[fsr]\r\nupscaler = network\r\n");
    const std::string set3 = wg::ini_set("[fsr]\nupscaler = amd\n", "wronggpu", "rr_mode", "exact");
    CHECK(set3 == "[fsr]\nupscaler = amd\n\n[wronggpu]\nrr_mode = exact\n");
    CHECK(wg::ini_set("", "wronggpu", "rr_mode", "exact") == "[wronggpu]\r\nrr_mode = exact\r\n");

    // host ini, custom slot, sidecar
    int idx = -1;
    CHECK(wg::parse_host_index("[FSR]\nUpscalerIndex=auto\n[WrongGPU]\nUpscalerIndex=3\n", "WrongGPU", &idx) && idx == 3);
    CHECK(!wg::parse_host_index("[WrongGPU]\nUpscalerIndex=auto\n", "WrongGPU", &idx));
    CHECK(!wg::parse_host_index("[FSR]\nUpscalerIndex=2\n", "WrongGPU", &idx));
    std::wstring hostIni;
    CHECK(wg::custom_slot_folder(L"G:\\Game\\bin\\OptiScaler\\custom\\WrongGPU\\ffx\\amd_fidelityfx_upscaler_dx12.dll", &hostIni) == L"WrongGPU");
    CHECK(hostIni == L"G:\\Game\\bin\\OptiScaler.ini");
    CHECK(wg::custom_slot_folder(L"G:/Game/OptiScaler/Custom/WrongGPU RR fast/FFX/x.dll", &hostIni) == L"WrongGPU RR fast");
    CHECK(wg::custom_slot_folder(L"G:\\Game\\bin\\wronggpu_ffx_dx12.dll").empty());
    CHECK(wg::custom_slot_folder(L"G:\\Game\\custom\\WrongGPU\\ffx_vk\\x.dll").empty());
    uint64_t id = 0;
    CHECK(wg::parse_sidecar_entry("entry=0x5747000000000002\n", &id) && id == 0x5747000000000002ull);
    CHECK(!wg::parse_sidecar_entry("entry=0x5747000000000102\n", &id));  // a retired RR entry id
    CHECK(!wg::parse_sidecar_entry("entry=0x5747000000000999\n", &id));
}

static void TestCreateOpenSeed() {
    const std::wstring ini = Scratch(L"seed.ini");
    WriteText(ini, "# comment\r\n[wronggpu]\r\nrr_mode = fast\r\nrr_enabled = 0\r\nrr_bounded_stores = bogus\r\nrr_model = C:\\models\\rr.wgcm\r\n");
    const std::wstring name = UniqueName("seed");
    wg::SharedSettings a, b;
    std::string log;
    CHECK(a.open(WG_SS_WRITER_SR_FRONTEND, ini, &log, name.c_str()) == wg::SharedSettings::Status::Created);
    CHECK(log.find("created, seeded from") != std::string::npos && log.find("rr_mode=fast (ini)") != std::string::npos);
    const WgSharedSettingsBlock* v = a.view();
    CHECK(v->header.magic == WG_SS_MAGIC && v->header.version == WG_SS_VERSION && v->header.size == WG_SS_SIZE);
    CHECK(v->header.slot_count == WG_SS_SLOT_COUNT && v->header.creator == WG_SS_WRITER_SR_FRONTEND && v->header.seed_source == 1);
    CHECK(v->header.registry_hash == WG_REGISTRY_HASH && v->header.writer == WG_SS_WRITER_NONE);
    CHECK(b.open(WG_SS_WRITER_RR_PLUGIN, ini, nullptr, name.c_str()) == wg::SharedSettings::Status::Opened);
    wg::Snapshot s;
    CHECK(b.snapshot(&s) && s.valid);
    CHECK(s.is_set(WG_OPT_RR_MODE) && s.values[WG_OPT_RR_MODE] == WG_RR_MODE_FAST);
    CHECK(s.is_set(WG_OPT_RR_ENABLED) && s.values[WG_OPT_RR_ENABLED] == 0);
    CHECK(!s.is_set(WG_OPT_RR_BOUNDED_STORES) && s.values[WG_OPT_RR_BOUNDED_STORES] == 0);  // bad value: default, unset
    CHECK(!s.is_set(WG_OPT_SR_MODEL) && s.values[WG_OPT_SR_MODEL] == WG_SR_MODEL_M);
    // unset slot -> the reader's default (and its env)
    wg::Value r = wg::resolve(WG_OPT_SR_MODEL, &s, nullptr);
    CHECK(r.value == WG_SR_MODEL_M && r.source == wg::Source::Default);
    _putenv_s("WG_RUNTIME_MODEL", "K");
    r = wg::resolve(WG_OPT_SR_MODEL, &s, nullptr);
    CHECK(r.value == WG_SR_MODEL_K && r.source == wg::Source::Env);
    _putenv_s("WG_RUNTIME_MODEL", "");
    _putenv_s("WG_RR_MODE", "exact");  // the block slot (set) wins over the environment
    r = wg::resolve(WG_OPT_RR_MODE, &s, nullptr);
    CHECK(r.value == WG_RR_MODE_FAST && r.source == wg::Source::Ini);
    r = wg::resolve(WG_OPT_RR_MODE, nullptr, nullptr);  // without a block the environment wins over the ini
    CHECK(r.value == WG_RR_MODE_EXACT && r.source == wg::Source::Env);
    _putenv_s("WG_RR_MODE", "");
    const wg::IniValues own = wg::parse_ini("[wronggpu]\nrr_mode = exact\n");
    r = wg::resolve(WG_OPT_RR_MODE, nullptr, &own);
    CHECK(r.value == WG_RR_MODE_EXACT && r.source == wg::Source::Ini);
    // text area
    std::string t;
    CHECK(b.text("rr_model", &t) && t == "C:\\models\\rr.wgcm");
    CHECK(b.text("rr_runtime_dll", &t) && t == "wronggpu_runtime.dll");
    CHECK(b.text("diag_env_4", &t) && t.empty());
    CHECK(!b.text("no_such_key", &t));
    CHECK(v->text_used > 0 && v->text_used < WG_SS_TEXT_SIZE);
    // no ini: registry defaults, nothing set
    wg::SharedSettings c;
    CHECK(c.open(WG_SS_WRITER_RR_PLUGIN, Scratch(L"absent.ini"), nullptr, UniqueName("noini").c_str()) == wg::SharedSettings::Status::Created);
    CHECK(c.view()->header.seed_source == 2 && c.view()->set_mask[0] == 0 && c.view()->values[WG_OPT_RR_MODE] == WG_RR_MODE_ACCURATE);
}

// The creator writes magic last: an opener on another thread sees either nothing published or a complete block.
static void TestMagicLast() {
    const std::wstring ini = Scratch(L"magic.ini");
    WriteText(ini, "[wronggpu]\r\nrr_mode = exact\r\nsr_model = K\r\nrr_model = X\r\n");
    int complete = 0, early = 0, bad = 0;
    for (int i = 0; i < 300; ++i) {
        const std::wstring name = UniqueName("magic");
        std::atomic<bool> go{false};
        std::thread reader([&] {
            while (!go.load()) {}
            HANDLE m = nullptr;
            for (int spin = 0; spin < 2000000 && !m; ++spin) m = OpenFileMappingW(FILE_MAP_READ, FALSE, name.c_str());
            if (!m) { ++bad; return; }
            auto* b = static_cast<const WgSharedSettingsBlock*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, WG_SS_SIZE));
            const ULONGLONG until = GetTickCount64() + 2000;
            while (GetTickCount64() < until) {
                const uint32_t magic = *reinterpret_cast<const volatile uint32_t*>(&b->header.magic);
                if (magic == 0) { ++early; continue; }
                MemoryBarrier();
                const bool ok = magic == WG_SS_MAGIC && b->header.version == WG_SS_VERSION && b->header.size == WG_SS_SIZE &&
                                b->header.slot_count == WG_SS_SLOT_COUNT && b->header.registry_hash == WG_REGISTRY_HASH &&
                                b->values[WG_OPT_RR_MODE] == WG_RR_MODE_EXACT && b->values[WG_OPT_SR_MODEL] == WG_SR_MODEL_K &&
                                b->text_used > 0;
                if (ok) ++complete; else ++bad;
                break;
            }
            UnmapViewOfFile(b);
            CloseHandle(m);
        });
        wg::SharedSettings creator;
        go.store(true);
        CHECK(creator.open(WG_SS_WRITER_SR_FRONTEND, ini, nullptr, name.c_str()) == wg::SharedSettings::Status::Created);
        reader.join();
    }
    std::printf("magic-last: %d complete views, %d reads before publication, %d torn\n", complete, early, bad);
    CHECK(bad == 0 && complete > 0);
}

// Writer flips two slots together 10^6 times; the reader never sees a mixed pair.
static void TestTornRead() {
    const std::wstring name = UniqueName("torn");
    wg::SharedSettings w, r;
    CHECK(w.open(WG_SS_WRITER_TEST, L"", nullptr, name.c_str()) == wg::SharedSettings::Status::Created);
    CHECK(r.open(WG_SS_WRITER_RR_PLUGIN, L"", nullptr, name.c_str()) == wg::SharedSettings::Status::Opened);
    CHECK(w.claim_writer(WG_SS_WRITER_TEST, 0));
    const uint32_t slots[2] = {WG_OPT_RR_SPECULAR_MOTION_FROM_V, WG_OPT_RR_RESET_AFTER_FAILED_EVALUATE};
    std::atomic<bool> done{false};
    long long reads = 0, failed = 0, mixed = 0;
    std::thread reader([&] {
        wg::Snapshot s;
        while (!done.load(std::memory_order_relaxed)) {
            if (!r.snapshot(&s, (1ull << slots[0]) | (1ull << slots[1]))) { ++failed; continue; }
            ++reads;
            if (s.values[slots[0]] != s.values[slots[1]]) ++mixed;
        }
    });
    const int kFlips = 1000000;
    bool allWrites = true;
    for (int i = 0; i < kFlips; ++i) {
        const int32_t v[2] = {i & 1, i & 1};
        allWrites &= w.write_slots(slots, v, 2, false);
    }
    done.store(true);
    reader.join();
    wg::Snapshot last;
    CHECK(r.snapshot(&last));
    std::printf("torn-read: %d flips, %lld consistent snapshots, %lld retries exhausted (kept last good), %lld mixed\n", kFlips, reads,
                failed, mixed);
    CHECK(allWrites && mixed == 0 && reads > 0);
    CHECK(last.generation == 2u * kFlips && last.values[slots[0]] == 1 && last.values[slots[1]] == 1);
    CHECK(last.is_live(slots[0]) && last.is_set(slots[1]));
}

static void TestWriterElection() {
    for (int round = 0; round < 50; ++round) {
        const std::wstring name = UniqueName("elect");
        wg::SharedSettings a, b, c, d;
        CHECK(a.open(WG_SS_WRITER_SR_FRONTEND, L"", nullptr, name.c_str()) == wg::SharedSettings::Status::Created);
        CHECK(b.open(WG_SS_WRITER_RR_PLUGIN, L"", nullptr, name.c_str()) == wg::SharedSettings::Status::Opened);
        std::atomic<int> go{0};
        bool wa = false, wb = false;
        std::thread ta([&] { ++go; while (go.load() < 2) {} wa = a.claim_writer(WG_SS_WRITER_SR_FRONTEND, WG_SS_FLAG_HOSTED_BY_OPTISCALER); });
        std::thread tb([&] { ++go; while (go.load() < 2) {} wb = b.claim_writer(WG_SS_WRITER_RR_PLUGIN, 0); });
        ta.join();
        tb.join();
        CHECK(wa != wb);
        const uint32_t winner = wa ? WG_SS_WRITER_SR_FRONTEND : WG_SS_WRITER_RR_PLUGIN;
        CHECK(a.writer() == winner && b.writer() == winner);
        wg::SharedSettings& loser = wa ? b : a;
        CHECK(!loser.write_slot(WG_OPT_RR_MODE, WG_RR_MODE_FAST, false));
        CHECK((wa ? a : b).write_slot(WG_OPT_RR_MODE, WG_RR_MODE_FAST, false));
        CHECK(((a.flags() & WG_SS_FLAG_HOSTED_BY_OPTISCALER) != 0) == wa);
        // the folder fallback: another instance with the winner's id may share it only when it asks to
        CHECK(c.open(winner, L"", nullptr, name.c_str()) == wg::SharedSettings::Status::Opened);
        CHECK(!c.claim_writer(winner, 0, false));
        CHECK(d.open(winner, L"", nullptr, name.c_str()) == wg::SharedSettings::Status::Opened);
        CHECK(d.claim_writer(winner, 0, true));
        CHECK(!c.claim_writer(winner == WG_SS_WRITER_SR_FRONTEND ? WG_SS_WRITER_RR_PLUGIN : WG_SS_WRITER_SR_FRONTEND, 0, true));
    }
}

// A block made by hand with a wrong header: refused (absent) or accepted with fewer slots.
static wg::SharedSettings::Status OpenForged(uint32_t version, uint32_t size, uint32_t slotCount, uint32_t hash, bool publish,
                                             std::string* log, wg::SharedSettings* keep = nullptr) {
    const std::wstring name = UniqueName("forged");
    HANDLE m = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, WG_SS_SIZE, name.c_str());
    auto* b = static_cast<WgSharedSettingsBlock*>(MapViewOfFile(m, FILE_MAP_WRITE, 0, 0, WG_SS_SIZE));
    b->header.version = version;
    b->header.size = size;
    b->header.slot_count = slotCount;
    b->header.registry_hash = hash;
    b->values[WG_OPT_ENTRY_SOURCE] = WG_ENTRY_SOURCE_FOLDERS;
    b->values[WG_OPT_RR_MODE] = WG_RR_MODE_FAST;
    b->set_mask[0] = (1u << WG_OPT_ENTRY_SOURCE) | (1u << WG_OPT_RR_MODE);
    if (publish) b->header.magic = WG_SS_MAGIC;
    wg::SharedSettings local;
    wg::SharedSettings& s = keep ? *keep : local;
    const wg::SharedSettings::Status st = s.open(WG_SS_WRITER_RR_PLUGIN, L"", log, name.c_str());
    UnmapViewOfFile(b);
    CloseHandle(m);
    return st;
}

static void TestRefusals() {
    using St = wg::SharedSettings::Status;
    std::string log;
    CHECK(OpenForged(2, WG_SS_SIZE, WG_SS_SLOT_COUNT, WG_REGISTRY_HASH, true, &log) == St::Refused);
    CHECK(log.find("ignored: version 2") != std::string::npos);
    log.clear();
    CHECK(OpenForged(WG_SS_VERSION, 128, WG_SS_SLOT_COUNT, WG_REGISTRY_HASH, true, &log) == St::Refused);
    CHECK(log.find("ignored: size 128") != std::string::npos);
    log.clear();
    CHECK(OpenForged(WG_SS_VERSION, WG_SS_SIZE, 0, WG_REGISTRY_HASH, true, &log) == St::Refused);
    CHECK(log.find("slot_count 0") != std::string::npos);
    log.clear();
    CHECK(OpenForged(WG_SS_VERSION, WG_SS_SIZE, 65, WG_REGISTRY_HASH, true, &log) == St::Refused);
    log.clear();
    const auto t0 = GetTickCount64();
    CHECK(OpenForged(WG_SS_VERSION, WG_SS_SIZE, WG_SS_SLOT_COUNT, WG_REGISTRY_HASH, false, &log) == St::Refused);
    const auto waited = GetTickCount64() - t0;
    CHECK(log.find("not published") != std::string::npos && waited < 1000);
    log.clear();
    // an older creator with 8 slots: accepted; slots 8.. use the reader's defaults even if a bit is set
    wg::SharedSettings older;
    CHECK(OpenForged(WG_SS_VERSION, WG_SS_SIZE, 8, WG_REGISTRY_HASH ^ 1u, true, &log, &older) == St::Opened);
    CHECK(log.find("registry hash") != std::string::npos && log.find("has 8 slots") != std::string::npos);
    wg::Snapshot s;
    CHECK(older.snapshot(&s) && s.is_set(WG_OPT_RR_MODE) && !s.is_set(WG_OPT_ENTRY_SOURCE));
    CHECK(wg::resolve(WG_OPT_ENTRY_SOURCE, &s, nullptr).value == WG_ENTRY_SOURCE_VERSIONS);
    CHECK(wg::resolve(WG_OPT_RR_MODE, &s, nullptr).value == WG_RR_MODE_FAST);
}

static void TestDataDir() {
    _putenv_s("WG_DATA_DIR", "");
    CHECK(wg::data_dir() == wg::exe_dir() + L"\\WrongGPU");
    CHECK(wg::ini_path() == wg::exe_dir() + L"\\WrongGPU\\wronggpu.ini");
    _putenv_s("WG_DATA_DIR", "K:\\elsewhere\\data\\");
    CHECK(wg::data_dir() == L"K:\\elsewhere\\data");
    CHECK(wg::ini_path() == L"K:\\elsewhere\\data\\wronggpu.ini");
    _putenv_s("WG_DATA_DIR", "");
}

static void TestPersistence() {
    const std::wstring ini = Scratch(L"persist.ini");
    const std::string original =
        "# WrongGPU settings, edited by hand\r\n[wronggpu]\r\n# my note about the mode\r\nrr_mode = accurate  ; the game I play\r\n"
        "custom_thing = 42\r\n\r\n[fsr]\r\nupscaler = network\r\nruntime_options = a=1;b=2\r\n[unknown]\r\nx = y\r\n";
    WriteText(ini, original);
    const std::wstring name = UniqueName("persist");
    wg::SharedSettings w;
    CHECK(w.open(WG_SS_WRITER_SR_FRONTEND, ini, nullptr, name.c_str()) == wg::SharedSettings::Status::Created);
    CHECK(!w.write_slot(WG_OPT_RR_MODE, WG_RR_MODE_FAST));  // not the writer yet
    CHECK(ReadText(ini) == original);
    CHECK(w.claim_writer(WG_SS_WRITER_SR_FRONTEND, WG_SS_FLAG_HOSTED_BY_OPTISCALER));
    CHECK(!w.write_slot(WG_OPT_RR_MODE, 7));  // out of range
    CHECK(w.write_slot(WG_OPT_RR_MODE, WG_RR_MODE_FAST));
    CHECK(w.write_slot(WG_OPT_SR_MODEL, WG_SR_MODEL_K));
    const std::string after = ReadText(ini);
    CHECK(after ==
          "# WrongGPU settings, edited by hand\r\n[wronggpu]\r\n# my note about the mode\r\nrr_mode = fast  ; the game I play\r\n"
          "custom_thing = 42\r\nsr_model = K\r\n\r\n[fsr]\r\nupscaler = network\r\nruntime_options = a=1;b=2\r\n[unknown]\r\nx = y\r\n");
    wg::Snapshot s;
    CHECK(w.snapshot(&s) && s.is_live(WG_OPT_RR_MODE) && wg::resolve(WG_OPT_RR_MODE, &s, nullptr).source == wg::Source::Menu);
    // a new block seeded from the persisted file starts with the menu's choice
    wg::SharedSettings next;
    CHECK(next.open(WG_SS_WRITER_RR_PLUGIN, ini, nullptr, UniqueName("persist2").c_str()) == wg::SharedSettings::Status::Created);
    CHECK(next.snapshot(&s) && s.values[WG_OPT_RR_MODE] == WG_RR_MODE_FAST && s.values[WG_OPT_SR_MODEL] == WG_SR_MODEL_K);
    // no ini at all: created with the generated defaults plus the value
    const std::wstring fresh = Scratch(L"fresh.ini");
    DeleteFileW(fresh.c_str());
    CHECK(wg::persist_ini_value(fresh, "wronggpu", "rr_mode", "exact"));
    const std::string made = ReadText(fresh);
    CHECK(made.find("rr_mode = exact\r\n") != std::string::npos && made.find("rr_mode = accurate") == std::string::npos);
    CHECK(made.find("# rr_mode: ray-reconstruction mode.") != std::string::npos && made.find("sr_model = M\r\n") != std::string::npos);
}

static void TestHostPoll() {
    const std::wstring path = Scratch(L"OptiScaler.ini");
    WriteText(path, "[Upscalers]\r\nDx12Upscaler=WrongGPU\r\n[WrongGPU]\r\nUpscalerIndex=auto\r\n");
    wg::HostIniPoll p;
    p.configure(path, "WrongGPU");
    int idx = -1;
    uint64_t t = 10000;
    CHECK(!p.poll(&idx, t));  // auto: nothing
    WriteText(path, "[Upscalers]\r\nDx12Upscaler=WrongGPU\r\n[WrongGPU]\r\nUpscalerIndex=4\r\n");
    CHECK(!p.poll(&idx, t + 500));  // within the interval: not even looked at
    CHECK(p.poll(&idx, t + 1000) && idx == 4);
    CHECK(!p.poll(&idx, t + 2000));  // unchanged
    WriteText(path, "; saved again\r\n[Upscalers]\r\nDx12Upscaler=WrongGPU\r\n[WrongGPU]\r\nUpscalerIndex=4\r\n");
    CHECK(!p.poll(&idx, t + 3000));  // rewritten, same index: nothing new
    WriteText(path, "[WrongGPU]\r\nUpscalerIndex=1\r\n");
    CHECK(p.poll(&idx, t + 4000) && idx == 1);
}

int main() {
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    g_dir = std::wstring(tmp) + L"wg_shared_settings_test_" + std::to_wstring(GetCurrentProcessId());
    CreateDirectoryW(g_dir.c_str(), nullptr);
    TestPure();
    TestCreateOpenSeed();
    TestMagicLast();
    TestTornRead();
    TestWriterElection();
    TestRefusals();
    TestDataDir();
    TestPersistence();
    TestHostPoll();
    for (const wchar_t* f : {L"seed.ini", L"magic.ini", L"persist.ini", L"fresh.ini", L"OptiScaler.ini"}) DeleteFileW(Scratch(f).c_str());
    RemoveDirectoryW(g_dir.c_str());
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
