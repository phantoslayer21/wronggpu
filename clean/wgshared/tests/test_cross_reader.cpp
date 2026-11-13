// SPDX-License-Identifier: MIT
// Cross-implementation test of the shared settings block (round i15 merge): the SR side's implementation
// (clean/wgshared/include/wg_settings_io.h, namespace wg) and the RR side's independent reader
// (clean/rr_f/proxy/include/wg_settings_reader.h, namespace wgss; used by the RR plugin and the RR runtime) must
// agree on the block's layout and rules without sharing any code except the two normative headers.
//   A. wg creates and seeds the block from <data dir>\wronggpu.ini; wgss opens it and sees the seeded values, the
//      unset slots as unset and the text option.
//   B. wg (the SR front end's role) claims the writer and writes the way a menu choice does; wgss sees value, live
//      flag, writer id, source word "menu" and a new even generation; wgss cannot take the writer or write.
//   C. Reverse: wgss creates and seeds, wg opens, then wgss (writer) writes and wg reads.
// Windows, CPU only. The block uses the real per-process name of this test process; nothing else touches it.
#define WG_SETTINGS_IO_IMPLEMENTATION
#include "wg_settings_io.h"

#include "wg_settings_reader.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_checks = 0, g_failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(c)) {                                                      \
            ++g_failures;                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
        }                                                                \
    } while (0)

static const std::string kIni =
    "[wronggpu]\r\nrr_mode = exact\r\nrr_model = C:\\models\\f.wgcm\r\nsr_fallback_to_amd = 0\r\n";

int main() {
    // A scratch data folder with the one settings file.
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    const std::wstring dir = std::wstring(tmp) + L"wg_cross_reader_" + std::to_wstring(GetCurrentProcessId());
    CreateDirectoryW(dir.c_str(), nullptr);
    SetEnvironmentVariableW(L"WG_DATA_DIR", dir.c_str());
    const std::wstring ini = dir + L"\\wronggpu.ini";
    CHECK(wg::write_file_atomic(ini, kIni));
    // The environment must not interfere with the seeded view.
    SetEnvironmentVariableW(L"WG_RR_MODE", nullptr);

    {  // ---- A + B: wg (SR side) creates; wgss (RR side) reads
        wg::SharedSettings w;
        std::string wlog;
        CHECK(w.open(WG_SS_WRITER_SR_FRONTEND, ini, &wlog) == wg::SharedSettings::Status::Created);
        wgss::Block r;
        std::vector<std::string> rlog;
        CHECK(r.open(WG_SS_WRITER_RR_PLUGIN, false, L"", &rlog));
        wgss::Snapshot s;
        CHECK(r.snapshot(0, &s) && s.present);
        CHECK(s.is_set(WG_OPT_RR_MODE) && s.v[WG_OPT_RR_MODE] == WG_RR_MODE_EXACT);
        CHECK(s.is_set(WG_OPT_SR_FALLBACK_TO_AMD) && s.v[WG_OPT_SR_FALLBACK_TO_AMD] == 0);
        CHECK(!s.is_set(WG_OPT_SR_MODEL));  // key absent from the ini: slot unset, the readers' own rules apply
        CHECK(!s.is_live(WG_OPT_RR_MODE));
        CHECK(r.text("rr_model") == "C:\\models\\f.wgcm");
        CHECK(r.writer() == WG_SS_WRITER_NONE);
        const wgss::Resolved seeded = wgss::resolve(*wgss::option_by_key("rr_mode"), &s, nullptr);
        CHECK(seeded.value == WG_RR_MODE_EXACT);

        CHECK(w.claim_writer(WG_SS_WRITER_SR_FRONTEND, 0));
        CHECK(!r.claim_writer(WG_SS_WRITER_RR_PLUGIN));  // one writer
        CHECK(!r.write_slot(WG_SS_WRITER_RR_PLUGIN, WG_OPT_RR_MODE, WG_RR_MODE_ACCURATE));
        const uint32_t g0 = s.generation;
        CHECK(w.write_slot(WG_OPT_RR_MODE, WG_RR_MODE_FAST, false));
        CHECK(r.snapshot(0, &s));
        CHECK(s.v[WG_OPT_RR_MODE] == WG_RR_MODE_FAST && s.is_set(WG_OPT_RR_MODE) && s.is_live(WG_OPT_RR_MODE));
        CHECK(s.writer == WG_SS_WRITER_SR_FRONTEND);
        CHECK(std::string(s.source(WG_OPT_RR_MODE)) == "menu");
        CHECK(s.generation > g0 && (s.generation & 1u) == 0);
        const wgss::Resolved menu = wgss::resolve(*wgss::option_by_key("rr_mode"), &s, nullptr);
        CHECK(menu.value == WG_RR_MODE_FAST && std::string(menu.source) == "menu");
        // The reader's description text is the same words as the writer's.
        CHECK(wgss::describe(WG_OPT_RR_MODE, WG_RR_MODE_FAST, "menu") == wg::describe(WG_OPT_RR_MODE, WG_RR_MODE_FAST, wg::Source::Menu));
        // An SR choice is seen in its own slot and leaves the RR slot alone.
        CHECK(w.write_slot(WG_OPT_SR_MODEL, WG_SR_MODEL_K, false));
        CHECK(r.snapshot(0, &s) && s.v[WG_OPT_SR_MODEL] == WG_SR_MODEL_K && s.v[WG_OPT_RR_MODE] == WG_RR_MODE_FAST);
        // A snapshot limited to the RR runtime's slots still sees the mode.
        wgss::Snapshot only;
        CHECK(r.snapshot(WG_READER_RR_RUNTIME, &only) && only.v[WG_OPT_RR_MODE] == WG_RR_MODE_FAST);
        r.close();
        w.close();
    }

    {  // ---- C: wgss creates and seeds; wg opens and reads; wgss is the writer
        wgss::Block r;
        std::vector<std::string> rlog;
        CHECK(r.open(WG_SS_WRITER_RR_PLUGIN, true, L"", &rlog));
        CHECK(r.created());
        wg::SharedSettings w;
        std::string wlog;
        CHECK(w.open(WG_SS_WRITER_SR_FRONTEND, ini, &wlog) == wg::SharedSettings::Status::Opened);
        wg::Snapshot s;
        CHECK(w.snapshot(&s) && s.valid);
        CHECK(s.is_set(WG_OPT_RR_MODE) && s.values[WG_OPT_RR_MODE] == WG_RR_MODE_EXACT);
        CHECK(!s.is_set(WG_OPT_SR_MODEL));
        std::string t;
        CHECK(w.text("rr_model", &t) && t == "C:\\models\\f.wgcm");
        CHECK(r.claim_writer(WG_SS_WRITER_RR_PLUGIN));
        CHECK(!w.claim_writer(WG_SS_WRITER_SR_FRONTEND, 0));  // taken by the other side
        CHECK(r.write_slot(WG_SS_WRITER_RR_PLUGIN, WG_OPT_RR_MODE, WG_RR_MODE_ACCURATE));
        CHECK(w.snapshot(&s) && s.values[WG_OPT_RR_MODE] == WG_RR_MODE_ACCURATE && s.is_live(WG_OPT_RR_MODE));
        CHECK(s.writer == WG_SS_WRITER_RR_PLUGIN && (s.generation & 1u) == 0);
        w.close();
        r.close();
    }

    DeleteFileW(ini.c_str());
    RemoveDirectoryW(dir.c_str());
    std::printf("test_cross_reader: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
