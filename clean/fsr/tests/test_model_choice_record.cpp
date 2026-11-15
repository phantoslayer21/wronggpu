// SPDX-License-Identifier: MIT
// Unit tests of the game's model choice record (clean/fsr/src/model_choice_record.h) on REAL files in a fresh folder under
// the system temporary folder, and with REAL processes (this program started again as a writer, a reader or a lock
// holder that dies): the strict three-key format, seq order with each source winning in turn, the cross-process lock
// (concurrent writers lose no update, seq never goes back; an abandoned lock is taken over), the upgrade migration,
// corrupt and overflowing records, the end of the sequence numbers, a failed write and its retry, a stale temporary file.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

std::string Valid(const char* model = "L", const char* seq = "7", const char* source = "launcher") {
    return std::string("; c\r\n\r\nmodel=") + model + "\r\nseq=" + seq + "\r\nsource=" + source + "\r\n";
}

bool Parses(const std::string& text, Record* r = nullptr) {
    Record tmp;
    std::string why;
    return ParseRecord(text, r ? r : &tmp, &why);
}

void TestStrictFormat() {
    Record r;
    CHECK(Parses(Valid(), &r) && r.model == 'L' && r.seq == 7 && r.source == Source::Launcher);
    CHECK(Parses("model=M\nseq=1\nsource=ffx\n", &r) && r.model == 'M' && r.source == Source::Ffx);
    CHECK(Parses("model=K\nseq=18446744073709551615\nsource=plugin", &r) && r.seq == UINT64_MAX && r.source == Source::Plugin);
    CHECK(!Parses(""));                                                           // no record
    CHECK(!Parses("model=L\nseq=7\n"));                                           // a key missing
    CHECK(!Parses(Valid() + "model=K\r\n"));                                      // a key twice
    CHECK(!Parses(Valid() + "extra=1\r\n"));                                      // an unknown key
    CHECK(!Parses(Valid() + "ffx=L\r\n"));                                        // the earlier design's fourth key: corrupt
    CHECK(!Parses(Valid("l")) && !Parses(Valid("Q")) && !Parses(Valid("LL")));  // model: M, K or L exactly
    CHECK(!Parses(Valid("L", "0")) && !Parses(Valid("L", "-1")) && !Parses(Valid("L", "7x")) && !Parses(Valid("L", "")));
    CHECK(!Parses(Valid("L", "18446744073709551616")));                          // 2^64: overflow refused
    CHECK(!Parses(Valid("L", "99999999999999999999")));                          // 20 digits, overflow refused
    CHECK(!Parses(Valid("L", "000000000000000000001")));                         // 21 digits refused
    CHECK(!Parses(Valid("L", "7", "game")) && !Parses(Valid("L", "7", "Launcher")));
    CHECK(!Parses("model = L\nseq=7\nsource=launcher\n"));                       // no spaces around '='
    CHECK(!Parses("model=L\nseq=7\nsource=launcher\ngarbage\n"));                // a line that is not key=value
    CHECK(Parses(FormatRecord(Record{'K', 42, Source::Plugin}), &r) && r.model == 'K' && r.seq == 42 && r.source == Source::Plugin);
}

void TestLockName() {
    const std::wstring a = LockName(L"C:\\Games\\X\\wronggpu-setup\\model-choice.ini");
    CHECK(a.rfind(L"Local\\WrongGPU-choice-", 0) == 0 && a.size() == std::wstring(L"Local\\WrongGPU-choice-").size() + 16);
    CHECK(LockName(L"c:/games/x/WRONGGPU-SETUP/model-choice.ini") == a);  // the same file however it is spelt
    CHECK(LockName(L"C:\\Games\\Y\\wronggpu-setup\\model-choice.ini") != a);  // another game: another lock
}

struct Folder {
    fs::path dir;
    explicit Folder(const char* name = "") {
        dir = fs::temp_directory_path() / ("wg-choice-test-" + std::to_string(
#ifdef _WIN32
                                               GetCurrentProcessId()
#else
                                               0
#endif
                                                   ) + name);
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

std::string ReadRaw(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

Record Rec(const fs::path& p, ReadResult* rr = nullptr) {
    Record r;
    std::string why;
    const ReadResult res = ReadRecordFile(p, &r, &why);
    if (rr) *rr = res;
    return res == ReadResult::Ok ? r : Record{0, 0, Source::Launcher};
}

int TmpFiles(const Folder& f) {
    int n = 0;
    std::error_code ec;
    for (auto& e : fs::directory_iterator(f.dir / kStateFolder, ec))
        if (e.path().extension() == ".tmp") ++n;
    return n;
}

void TestChoicesInTurn() {
    Folder f;
    Record r, after;
    std::string why;
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Missing);
    // each source winning in turn: launcher L, the player in OptiScaler's list (K, an event), the plugin's menu (M), the
    // launcher again (L), the list re-selecting the SAME entry after a plugin choice (it wins): the highest seq is the record
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written && after.seq == 1);
    r = Rec(f.record());
    CHECK(r.model == 'L' && r.seq == 1 && r.source == Source::Launcher);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why) == Outcome::Written);
    r = Rec(f.record());
    CHECK(r.model == 'K' && r.seq == 2 && r.source == Source::Ffx);
    CHECK(RecordChoice(f.record(), 'M', Source::Plugin, &after, &why) == Outcome::Written);
    r = Rec(f.record());
    CHECK(r.model == 'M' && r.seq == 3 && r.source == Source::Plugin);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why) == Outcome::Written);  // K again in the list: wins
    r = Rec(f.record());
    CHECK(r.model == 'K' && r.seq == 4 && r.source == Source::Ffx);
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);
    r = Rec(f.record());
    CHECK(r.model == 'L' && r.seq == 5 && r.source == Source::Launcher);
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);  // the same choice again: seq grows
    CHECK(Rec(f.record()).seq == 6);
    // a stale temporary file a crash left next to the record (and garbage in it): never read, writes still work
    const fs::path stale = f.record().string() + ".999.0.0.tmp";
    WriteRaw(stale, "model=K\nseq=999999");
    CHECK(Rec(f.record()).seq == 6);
    CHECK(RecordChoice(f.record(), 'K', Source::Plugin, &after, &why) == Outcome::Written && Rec(f.record()).seq == 7);
    std::error_code ec;
    fs::remove(stale, ec);
    // only a temporary file, no record (a crash before the move): no record
    fs::remove(f.record(), ec);
    WriteRaw(stale, Valid());
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Missing);
    fs::remove(stale, ec);
    CHECK(TmpFiles(f) == 0);  // no temporary file is left behind by the writes
}

void TestCorruptAndEnd() {
    Folder f;
    Record r, after;
    std::string why;
    // corrupt and overflowing records: Corrupt, never a partial choice; the next choice replaces them whole (seq 1)
    for (const std::string& bad : {Valid("L", "18446744073709551616"), std::string("model=L\nseq=3\n"), std::string(4096, 'x'),
                                   Valid() + "ffx=L\r\n"}) {
        WriteRaw(f.record(), bad);
        CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Corrupt);
        why.clear();
        CHECK(RecordChoice(f.record(), 'M', Source::Ffx, &after, &why) == Outcome::Written && after.seq == 1);
        CHECK(why.find("corrupt") != std::string::npos);
        r = Rec(f.record());
        CHECK(r.model == 'M' && r.seq == 1 && r.source == Source::Ffx);
    }
    // the end of the sequence numbers: refused loudly (SeqExhausted), the record unchanged, never wrapped to 0
    WriteRaw(f.record(), Valid("K", "18446744073709551615", "plugin"));
    const std::string before = ReadRaw(f.record());
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::SeqExhausted);
    CHECK(ReadRaw(f.record()) == before);
    CHECK(std::string(OutcomeWords(Outcome::SeqExhausted)).find("refused") != std::string::npos);
    CHECK(TmpFiles(f) == 0);
}

void TestMigration() {
    Folder f;
    Record r;
    bool migrated = false;
    std::string why;
    // an install from before 0.6.5 (no record): wronggpu.ini's model item, seq 1, source=launcher; the stamp files of the
    // earlier designs next to it are ignored
    WriteRaw(f.dir / kStateFolder / "wronggpu-ffx-choice.ini", "model=K\nchosen=99999999999\n");
    CHECK(LoadOrMigrateRecord(f.record(), 'L', &r, &migrated, &why) == ReadResult::Ok && migrated);
    CHECK(r.model == 'L' && r.seq == 1 && r.source == Source::Launcher);
    r = Rec(f.record());
    CHECK(r.model == 'L' && r.seq == 1 && r.source == Source::Launcher);
    // a record now: returned as it is, nothing written
    const std::string text = ReadRaw(f.record());
    CHECK(LoadOrMigrateRecord(f.record(), 'K', &r, &migrated, &why) == ReadResult::Ok && !migrated && r.model == 'L');
    CHECK(ReadRaw(f.record()) == text);
    // no model item: Model M
    std::error_code ec;
    fs::remove(f.record(), ec);
    CHECK(LoadOrMigrateRecord(f.record(), 0, &r, &migrated, &why) == ReadResult::Ok && migrated && r.model == 'M' && r.seq == 1);
    // a corrupt record is not migrated over (it stays corrupt until a real choice replaces it)
    WriteRaw(f.record(), "model=L\n");
    CHECK(LoadOrMigrateRecord(f.record(), 'K', &r, &migrated, &why) == ReadResult::Corrupt && !migrated);
    CHECK(ReadRaw(f.record()) == "model=L\n");
    fs::remove(f.dir / kStateFolder / "wronggpu-ffx-choice.ini", ec);
}

#ifdef _WIN32
// A write that fails (the record held open without delete sharing, so it cannot be replaced) and its retry; a choice
// superseded while it waited; a lock another thread holds (NoLock); an unreadable record never written over.
void TestFailureAndRetry() {
    Folder f;
    Record after;
    std::string why;
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);  // seq 1
    HANDLE h = CreateFileW(f.record().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(h != INVALID_HANDLE_VALUE);
    Pending p;
    CHECK(RecordChoice(f.record(), 'K', Source::Plugin, &after, &why, &p) == Outcome::WriteFailed);
    CHECK(p.active() && p.model == 'K' && p.baseKnown && p.baseSeq == 1 && !why.empty());
    CHECK(RetryPending(f.record(), &p, &after, &why) == Outcome::WriteFailed && p.active());  // still held: still pending
    CloseHandle(h);
    CHECK(Rec(f.record()).seq == 1 && Rec(f.record()).model == 'L');  // nothing changed while it failed
    CHECK(RetryPending(f.record(), &p, &after, &why) == Outcome::Written && !p.active());
    Record r = Rec(f.record());
    CHECK(r.model == 'K' && r.seq == 2 && r.source == Source::Plugin);
    // a pending choice superseded meanwhile (the launcher wrote after it): dropped, the newer choice stays
    h = CreateFileW(f.record().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(RecordChoice(f.record(), 'M', Source::Ffx, &after, &why, &p) == Outcome::WriteFailed && p.baseSeq == 2);
    CloseHandle(h);
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written && after.seq == 3);
    CHECK(RetryPending(f.record(), &p, &after, &why) == Outcome::Superseded && !p.active());
    r = Rec(f.record());
    CHECK(r.model == 'L' && r.seq == 3 && r.source == Source::Launcher);
    // the lock held by another thread: NoLock (base unknown), nothing written; once free the retry writes
    std::atomic<bool> held{false}, release{false};
    std::thread holder([&] {
        Lock l(f.record());
        held.store(l.ok());
        while (!release.load()) Sleep(1);
    });
    while (!held.load()) Sleep(1);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why, &p, 100) == Outcome::NoLock && p.active() && !p.baseKnown);
    CHECK(Rec(f.record()).seq == 3);
    release.store(true);
    holder.join();
    CHECK(RetryPending(f.record(), &p, &after, &why) == Outcome::Written && Rec(f.record()).seq == 4 && Rec(f.record()).model == 'K');
    // a record held open without any sharing: Unreadable, never written over
    h = CreateFileW(f.record().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(RecordChoice(f.record(), 'M', Source::Plugin, &after, &why, &p) == Outcome::Unreadable && p.active() && !p.baseKnown);
    CloseHandle(h);
    CHECK(Rec(f.record()).seq == 4 && Rec(f.record()).model == 'K');
    CHECK(TmpFiles(f) == 0);
}

std::wstring Self() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return buf;
}

HANDLE Spawn(const std::wstring& args) {
    std::wstring cmd = L"\"" + Self() + L"\" " + args;
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return nullptr;
    CloseHandle(pi.hThread);
    return pi.hProcess;
}

DWORD Wait(HANDLE p) {
    WaitForSingleObject(p, 120000);
    DWORD code = 999;
    GetExitCodeProcess(p, &code);
    CloseHandle(p);
    return code;
}

// Several REAL processes writing the record at once (the launcher, the front end and the plugin are separate programs or
// modules that all use RecordChoice), and one reading it under the lock: no update is lost (the final seq is the number
// of writes), each writer sees its own seq grow, the reader never sees a partial record or seq going back.
void TestConcurrentProcesses() {
    Folder f("-procs");
    const int kWriters = 4, kWrites = 150;
    const wchar_t* models = L"MKLM";
    const wchar_t* sources[] = {L"launcher", L"ffx", L"plugin", L"plugin"};
    HANDLE reader = Spawn(L"reader \"" + f.record().wstring() + L"\" " + std::to_wstring(kWriters * kWrites));
    std::vector<HANDLE> writers;
    for (int i = 0; i < kWriters; ++i)
        writers.push_back(Spawn(L"writer \"" + f.record().wstring() + L"\" " + std::wstring(1, models[i]) + L" " + sources[i] + L" " +
                                std::to_wstring(kWrites)));
    int failed = 0;
    for (HANDLE w : writers) {
        CHECK(w != nullptr);
        if (w && Wait(w) != 0) ++failed;
    }
    CHECK(failed == 0);
    CHECK(reader != nullptr && Wait(reader) == 0);
    Record r = Rec(f.record());
    CHECK(r.seq == uint64_t(kWriters * kWrites));  // no lost update
    std::printf("concurrent processes: %d writers x %d writes, final seq %llu\n", kWriters, kWrites, static_cast<unsigned long long>(r.seq));
    CHECK(TmpFiles(f) == 0);
}

// A process that dies while it holds the lock: the next one takes it over (WAIT_ABANDONED) and the record is whole.
void TestAbandonedLock() {
    Folder f("-abandon");
    Record after;
    std::string why;
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);
    HANDLE p = Spawn(L"lock-and-die \"" + f.record().wstring() + L"\"");
    CHECK(p != nullptr && Wait(p) == 97);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why, nullptr, 2000) == Outcome::Written && after.seq == 2);
    CHECK(Rec(f.record()).model == 'K');
}

int ChildWriter(const fs::path& record, char model, const std::string& source, int count) {
    const Source s = source == "launcher" ? Source::Launcher : source == "ffx" ? Source::Ffx : Source::Plugin;
    uint64_t last = 0;
    int bad = 0;
    for (int i = 0; i < count; ++i) {
        Record after;
        std::string why;
        const Outcome o = RecordChoice(record, model, s, &after, &why, nullptr, 20000);
        if (o != Outcome::Written) {
            std::printf("writer %c: %s (%s)\n", model, OutcomeWords(o), why.c_str());
            ++bad;
            continue;
        }
        if (after.seq <= last) ++bad;  // seq went back or stood still for this writer
        last = after.seq;
    }
    return bad;
}

int ChildReader(const fs::path& record, uint64_t until) {
    uint64_t last = 0;
    int bad = 0;
    const ULONGLONG start = GetTickCount64();
    while (last < until && GetTickCount64() - start < 100000) {
        Lock lock(record, 20000);
        if (!lock.ok()) return 100;
        Record r;
        std::string why;
        const ReadResult rr = ReadRecordFile(record, &r, &why);
        if (rr == ReadResult::Corrupt || rr == ReadResult::Unreadable) ++bad;
        if (rr == ReadResult::Ok) {
            if (r.seq < last) ++bad;
            last = r.seq;
        }
    }
    return bad + (last < until ? 1000 : 0);
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc >= 6 && std::string(argv[1]) == "writer") return ChildWriter(argv[2], argv[3][0], argv[4], std::atoi(argv[5]));
    if (argc >= 4 && std::string(argv[1]) == "reader") return ChildReader(argv[2], std::strtoull(argv[3], nullptr, 10));
    if (argc >= 3 && std::string(argv[1]) == "lock-and-die") {
        Lock lock(argv[2]);
        if (!lock.ok()) return 1;
        TerminateProcess(GetCurrentProcess(), 97);  // dies holding the lock
    }
#else
    (void)argc;
    (void)argv;
#endif
    TestStrictFormat();
    TestLockName();
    TestChoicesInTurn();
    TestCorruptAndEnd();
    TestMigration();
#ifdef _WIN32
    TestFailureAndRetry();
    TestConcurrentProcesses();
    TestAbandonedLock();
#endif
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
