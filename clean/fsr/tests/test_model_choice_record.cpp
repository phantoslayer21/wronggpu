// SPDX-License-Identifier: MIT
// Unit tests of the game's model choice record (clean/fsr/src/model_choice_record.h) on REAL files in a fresh folder under
// the system temporary folder: the strict format, seq order, the OptiScaler list rule (ffx), atomic replacement, a stale
// temporary file, corrupt and overflowing records, and a reader running while a writer replaces the record.
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "../src/model_choice_record.h"

using namespace wgchoice;
namespace fs = std::filesystem;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                                                     \
    do {                                                                                                \
        ++g_checks;                                                                                     \
        if (!(cond)) { ++g_failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); }      \
    } while (0)

namespace {

std::string Valid(const char* model = "L", const char* seq = "7", const char* source = "launcher", const char* ffx = "L") {
    return std::string("; c\r\n\r\nmodel=") + model + "\r\nseq=" + seq + "\r\nsource=" + source + "\r\nffx=" + ffx + "\r\n";
}

bool Parses(const std::string& text, Record* r = nullptr) {
    Record tmp;
    std::string why;
    return ParseRecord(text, r ? r : &tmp, &why);
}

void TestStrictFormat() {
    Record r;
    CHECK(Parses(Valid(), &r) && r.model == 'L' && r.seq == 7 && r.source == Source::Launcher && r.ffx == 'L');
    CHECK(Parses("model=M\nseq=1\nsource=ffx\nffx=-\n", &r) && r.model == 'M' && r.source == Source::Ffx && r.ffx == kUnknown);
    CHECK(Parses("model=K\nseq=18446744073709551615\nsource=plugin\nffx=K", &r) && r.seq == UINT64_MAX && r.source == Source::Plugin);
    CHECK(!Parses(""));                                                           // no record
    CHECK(!Parses("model=L\nseq=7\nsource=launcher\n"));                          // a key missing
    CHECK(!Parses(Valid() + "model=K\r\n"));                                      // a key twice
    CHECK(!Parses(Valid() + "extra=1\r\n"));                                      // an unknown key
    CHECK(!Parses(Valid("l")) && !Parses(Valid("Q")) && !Parses(Valid("LL")));  // model: M, K or L exactly
    CHECK(!Parses(Valid("L", "0")) && !Parses(Valid("L", "-1")) && !Parses(Valid("L", "7x")) && !Parses(Valid("L", "")));
    CHECK(!Parses(Valid("L", "18446744073709551616")));                          // 2^64: overflow refused
    CHECK(!Parses(Valid("L", "99999999999999999999")));                          // 20 digits, overflow refused
    CHECK(!Parses(Valid("L", "000000000000000000001")));                         // 21 digits refused
    CHECK(!Parses(Valid("L", "7", "game")) && !Parses(Valid("L", "7", "Launcher")));
    CHECK(!Parses(Valid("L", "7", "launcher", "x")) && !Parses(Valid("L", "7", "launcher", "")));
    CHECK(!Parses("model = L\nseq=7\nsource=launcher\nffx=L\n"));                // no spaces around '='
    CHECK(!Parses("model=L\nseq=7\nsource=launcher\nffx=L\ngarbage\n"));         // a line that is not key=value
    CHECK(Parses(FormatRecord(Record{'K', 42, Source::Plugin, 'M'}), &r) && r.model == 'K' && r.seq == 42 && r.ffx == 'M');
}

void TestOrderAndFfxRule() {
    Record a, b, c;
    CHECK(NextRecord(nullptr, 'L', Source::Launcher, 'L', &a) && a.seq == 1 && a.ffx == 'L');
    CHECK(NextRecord(&a, 'K', Source::Plugin, kUnknown, &b) && b.seq == 2 && b.model == 'K' && b.ffx == 'L');  // keeps ffx
    Record max{'M', UINT64_MAX, Source::Launcher, 'M'};
    CHECK(!NextRecord(&max, 'L', Source::Launcher, 'L', &c));  // seq cannot grow: refused, never wraps to 0
    // the list names the entry the record knows: nothing (also after the plugin's choice: the list did not move)
    CHECK(FfxObservation(&b, 'L', &c) == FfxStep::None);
    CHECK(FfxObservation(&b, kUnknown, &c) == FfxStep::None);  // no WrongGPU entry (an AMD version)
    // another entry: the player's choice in OptiScaler's list
    CHECK(FfxObservation(&b, 'M', &c) == FfxStep::Choice && c.model == 'M' && c.seq == 3 && c.source == Source::Ffx && c.ffx == 'M');
    // the first create without a record: the entry seen counts as a choice
    CHECK(FfxObservation(nullptr, 'K', &c) == FfxStep::Choice && c.model == 'K' && c.seq == 1 && c.ffx == 'K');
    // a record that does not know the list's entry: adopted, not a choice (same model, same seq)
    Record unk{'L', 5, Source::Plugin, kUnknown};
    CHECK(FfxObservation(&unk, 'M', &c) == FfxStep::Adopt && c.model == 'L' && c.seq == 5 && c.ffx == 'M');
    CHECK(FfxObservation(&max, 'L', &c) == FfxStep::None);  // seq at its end: no write
}

struct Folder {
    fs::path dir;
    Folder() {
        dir = fs::temp_directory_path() / ("wg-choice-test-" + std::to_string(
#ifdef _WIN32
                                               GetCurrentProcessId()
#else
                                               0
#endif
                                                   ));
        std::error_code ec;
        fs::create_directories(dir / kStateFolder, ec);
    }
    fs::path record() const { return RecordPathIn(dir); }
    ~Folder() {  // our own files only, one by one
        std::error_code ec;
        for (auto& e : fs::directory_iterator(dir / kStateFolder, ec)) fs::remove(e.path(), ec);
        fs::remove(dir / kStateFolder, ec);
        fs::remove(dir, ec);
    }
};

void WriteRaw(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << text;
}

void TestFiles() {
    Folder f;
    Record r;
    std::string why;
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Missing);
    // each source winning in turn: launcher L, then the player in OptiScaler's list (K), then the plugin's menu (M), then the
    // launcher again (L): the highest seq is always the record
    Record cur, next;
    CHECK(NextRecord(nullptr, 'L', Source::Launcher, 'L', &next) && WriteRecordFile(f.record(), next, &why));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.model == 'L' && cur.seq == 1 && cur.source == Source::Launcher);
    CHECK(FfxObservation(&cur, 'K', &next) == FfxStep::Choice && WriteRecordFile(f.record(), next, &why));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.model == 'K' && cur.seq == 2 && cur.source == Source::Ffx);
    CHECK(NextRecord(&cur, 'M', Source::Plugin, kUnknown, &next) && WriteRecordFile(f.record(), next, &why));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.model == 'M' && cur.seq == 3 && cur.ffx == 'K');
    CHECK(FfxObservation(&cur, 'K', &next) == FfxStep::None);  // the list still names K: no false choice after the plugin's
    // the launcher applies L and sets the list's entry with it: its own index rewrite is never a choice in the game
    CHECK(NextRecord(&cur, 'L', Source::Launcher, 'L', &next) && WriteRecordFile(f.record(), next, &why));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.model == 'L' && cur.seq == 4 && cur.ffx == 'L');
    CHECK(FfxObservation(&cur, 'L', &next) == FfxStep::None);
    // a stale temporary file a crash left next to the record (and garbage in it): never read, writes still work
    const fs::path stale = f.record().string() + ".999.0.0.tmp";
    WriteRaw(stale, "model=K\nseq=999999");
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.seq == 4);
    CHECK(NextRecord(&cur, 'K', Source::Plugin, kUnknown, &next) && WriteRecordFile(f.record(), next, &why));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.seq == 5 && cur.model == 'K');
    // only a temporary file, no record (a crash before the move): no record
    std::error_code ec;
    fs::remove(f.record(), ec);
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Missing);
    // corrupt and overflowing records: Corrupt, never a partial choice
    WriteRaw(f.record(), Valid("L", "18446744073709551616"));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Corrupt && why.find("seq") != std::string::npos);
    WriteRaw(f.record(), "model=L\nseq=3\n");
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Corrupt);
    WriteRaw(f.record(), std::string(4096, 'x'));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Corrupt);
    // ... and a write over a corrupt record replaces it whole
    CHECK(NextRecord(nullptr, 'M', Source::Ffx, 'M', &next) && WriteRecordFile(f.record(), next, &why));
    CHECK(ReadRecordFile(f.record(), &cur, &why) == ReadResult::Ok && cur.seq == 1 && cur.model == 'M');
    fs::remove(stale, ec);
    // no temporary file is left behind by successful writes
    int tmps = 0;
    for (auto& e : fs::directory_iterator(f.dir / kStateFolder, ec))
        if (e.path().extension() == ".tmp") ++tmps;
    CHECK(tmps == 0);
}

// A reader that runs while a writer replaces the record many times sees a whole record every time (or, before the first
// write, none), with seq never going down: never an empty or partial file.
void TestConcurrentReader() {
    Folder f;
    std::atomic<bool> done{false};
    std::atomic<int> partial{0}, reads{0}, backwards{0};
    std::thread reader([&] {
        uint64_t last = 0;
        while (!done.load()) {
            Record r;
            std::string why;
            const ReadResult rr = ReadRecordFile(f.record(), &r, &why);
            if (rr == ReadResult::Corrupt) ++partial;
            if (rr == ReadResult::Ok) {
                ++reads;
                if (r.seq < last) ++backwards;
                last = r.seq;
            }
        }
    });
    Record cur, next;
    int failedWrites = 0;
    for (int i = 0; i < 400; ++i) {
        std::string why;
        NextRecord(i ? &cur : nullptr, "MKL"[i % 3], Source::Launcher, "MKL"[i % 3], &next);
        if (WriteRecordFile(f.record(), next, &why)) cur = next;
        else ++failedWrites;
    }
    done.store(true);
    reader.join();
    CHECK(partial.load() == 0);
    CHECK(backwards.load() == 0);
    CHECK(failedWrites == 0);
    CHECK(reads.load() > 0);
    Record r;
    std::string why;
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Ok && r.seq == 400);
}

}  // namespace

int main() {
    TestStrictFormat();
    TestOrderAndFfxRule();
    TestFiles();
    TestConcurrentReader();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
