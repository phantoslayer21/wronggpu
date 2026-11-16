// SPDX-License-Identifier: MIT
// Unit tests of the game's model choice record (clean/fsr/src/model_choice_record.h) on REAL files in a fresh folder under
// the system temporary folder, and with REAL processes (this program started again as a writer, a reader, a launcher
// apply, a lock holder, or a lock holder that dies): the strict three-key format, seq order with each source winning in
// turn, the game's state lock (wg_state.h: concurrent writers lose no update, seq never goes back; an abandoned lock is
// taken over; two path aliases of one folder are one lock), the upgrade migration (under the lock, against a launcher
// apply in another process), corrupt and overflowing records, the end of the sequence numbers, failures that change
// nothing (no queue), a stale temporary file.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cwctype>
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
        if (wgstate::IsTempName(e.path().filename().wstring())) ++n;
    return n;
}

// An install with an id (the install record's install-id line, #224): every write of the record carries WrongGPU's stamp
// for model-choice.ini and that id (wg_state.h WriteStamped), so the launcher's uninstall knows the record as WrongGPU's;
// the stamp is a comment line the parser skips. Another install's id, or the file under another name: not stamped for it.
void TestStampedRecord() {
    Folder f("stamp");
    const std::string id = "0123456789abcdef0123456789abcdef";
    {
        std::ofstream o(f.dir / kStateFolder / "install.record", std::ios::binary);
        o << "WRONGGPU-RECORD 2\r\ninstall-id\t" << id << "\r\n";
    }
    Record after;
    std::string why;
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);
    std::string bytes;
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes));
    CHECK(wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    CHECK(!wgstate::StampedByWrongGPU(bytes, "model-choice.ini", "fedcba9876543210fedcba9876543210"));
    CHECK(!wgstate::StampedByWrongGPU(bytes, "settings.ini", id));
    Record r;
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Ok && r.model == 'L' && r.seq == 1);
    // the next choice: still stamped, the seq grows (the stamp line is not read as part of the record)
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why) == Outcome::Written && after.seq == 2);
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes) && wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Ok && r.model == 'K' && r.seq == 2 && r.source == Source::Ffx);
    // without an install id (no install record): nothing is stamped (the earlier stamp line is no longer valid and stays as a
    // comment, as #224's updater keeps any line that is not a valid stamp); a new record without an id: no stamp line at all
    fs::remove(f.dir / kStateFolder / "install.record");
    CHECK(RecordChoice(f.record(), 'M', Source::Plugin, &after, &why) == Outcome::Written);
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes) && !wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Ok && r.model == 'M' && r.seq == 3);
    fs::remove(f.record());
    CHECK(RecordChoice(f.record(), 'K', Source::Launcher, &after, &why) == Outcome::Written && after.seq == 1);
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes) && bytes.find(wgstate::kStampPrefix) == std::string::npos);
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
    const fs::path stale = f.record().wstring() + wgstate::kTempMark + L"999-1";
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

// #224's rule for the game's state files: a choice changes only the record's three key lines, in place. A record WrongGPU
// wrote and nobody changed keeps WrongGPU's stamp; one the player edited (a comment added to a stamped record) or made
// (no stamp) keeps every other byte (comments, blank lines, line ends) and stays UNSTAMPED, the player's, which the
// launcher's uninstall keeps.
void TestPlayerFiles() {
    Folder f("player");
    const std::string id = "00112233445566778899aabbccddeeff";
    {
        std::ofstream o(f.dir / kStateFolder / "install.record", std::ios::binary);
        o << "WRONGGPU-RECORD 2\r\ninstall-id\t" << id << "\r\n";
    }
    Record after, r;
    std::string why, bytes;
    // WrongGPU's own record: re-stamped
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes) && wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    // edited: the player adds a comment line to the stamped record; the next choice keeps it, writes K, no stamp
    WriteRaw(f.record(), "; my note: keep L for this game\r\n" + bytes);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why) == Outcome::Written && after.seq == 2);
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes));
    CHECK(bytes.rfind("; my note: keep L for this game\r\n", 0) == 0);
    CHECK(bytes.find("model=K\r\nseq=2\r\nsource=ffx\r\n") != std::string::npos);
    CHECK(!wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Ok && r.model == 'K' && r.seq == 2);
    // and it stays the player's on the choice after that (still no stamp)
    CHECK(RecordChoice(f.record(), 'M', Source::Plugin, &after, &why) == Outcome::Written && after.seq == 3);
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes) && !wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    CHECK(bytes.rfind("; my note: keep L for this game\r\n", 0) == 0);
    // player-made: LF line ends, a blank line, comments around the keys, keys in another order, no stamp
    const std::string mine = "# mine\n\nsource=launcher\n; the model:\nmodel=L\nseq=40\n# end\n";
    WriteRaw(f.record(), mine);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why) == Outcome::Written && after.seq == 41);
    CHECK(ReadRaw(f.record()) == "# mine\n\nsource=ffx\n; the model:\nmodel=K\nseq=41\n# end\n");
    CHECK(wgstate::ReadBytes(f.record().wstring(), &bytes) && !wgstate::StampedByWrongGPU(bytes, "model-choice.ini", id));
    CHECK(TmpFiles(f) == 0);
}

void TestCorruptAndEnd() {
    Folder f;
    Record r, after;
    std::string why;
    // corrupt and overflowing records (and text WrongGPU cannot keep: a NUL): Corrupt, never a partial choice, and NEVER
    // replaced: a choice is refused with what is wrong, the file keeps its bytes (the player's to fix or delete)
    for (const std::string& bad : {Valid("L", "18446744073709551616"), std::string("model=L\nseq=3\n"), std::string(4096, 'x'),
                                   Valid() + "ffx=L\r\n", std::string("model=L\0\nseq=3\nsource=ffx\n", 26)}) {
        WriteRaw(f.record(), bad);
        CHECK(ReadRecordFile(f.record(), &r, &why) == ReadResult::Corrupt);
        why.clear();
        CHECK(RecordChoice(f.record(), 'M', Source::Ffx, &after, &why) == Outcome::Corrupt);
        CHECK(!why.empty());
        CHECK(ReadRaw(f.record()) == bad);
    }
    CHECK(std::string(OutcomeWords(Outcome::Corrupt)).find("left as it is") != std::string::npos);
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
    auto model = [](char m) { return [m] { return m; }; };
    // an install from before 0.6.5 (no record): wronggpu.ini's model item, seq 1, source=launcher; the stamp files of the
    // earlier designs next to it are ignored
    WriteRaw(f.dir / kStateFolder / "wronggpu-ffx-choice.ini", "model=K\nchosen=99999999999\n");
    CHECK(LoadOrMigrateRecord(f.record(), model('L'), &r, &migrated, &why) == LoadResult::Ok && migrated);
    CHECK(r.model == 'L' && r.seq == 1 && r.source == Source::Launcher);
    r = Rec(f.record());
    CHECK(r.model == 'L' && r.seq == 1 && r.source == Source::Launcher);
    // a record now: returned as it is, nothing written, the model item not even read
    const std::string text = ReadRaw(f.record());
    bool asked = false;
    CHECK(LoadOrMigrateRecord(f.record(), [&] { asked = true; return 'K'; }, &r, &migrated, &why) == LoadResult::Ok && !migrated &&
          r.model == 'L' && !asked);
    CHECK(ReadRaw(f.record()) == text);
    // no model item: Model M
    std::error_code ec;
    fs::remove(f.record(), ec);
    CHECK(LoadOrMigrateRecord(f.record(), model(0), &r, &migrated, &why) == LoadResult::Ok && migrated && r.model == 'M' && r.seq == 1);
    // a corrupt record is not migrated over (it stays corrupt until a real choice replaces it)
    WriteRaw(f.record(), "model=L\n");
    CHECK(LoadOrMigrateRecord(f.record(), model('K'), &r, &migrated, &why) == LoadResult::Corrupt && !migrated);
    CHECK(ReadRaw(f.record()) == "model=L\n");
    fs::remove(f.dir / kStateFolder / "wronggpu-ffx-choice.ini", ec);
}

#ifdef _WIN32
// Failures change nothing and nothing is kept for later: a write that fails (the record held open without delete sharing,
// so it cannot be replaced), the game's state lock held by another thread (through another spelling of the game folder),
// an unreadable record (never written over). After each, the next choice is written normally from the record as it is.
void TestFailures() {
    Folder f;
    Record after, before;
    std::string why;
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written);  // seq 1
    CHECK(RecordChoice(f.record(), 'K', Source::Plugin, &after, &why, &before) == Outcome::Written && before.model == 'L' &&
          before.seq == 1 && after.seq == 2);  // the record replaced is reported
    HANDLE h = CreateFileW(f.record().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(h != INVALID_HANDLE_VALUE);
    const std::string kept = ReadRaw(f.record());
    CHECK(RecordChoice(f.record(), 'M', Source::Ffx, &after, &why) == Outcome::WriteFailed && !why.empty());
    CloseHandle(h);
    CHECK(ReadRaw(f.record()) == kept);  // nothing changed, nothing to be written later
    CHECK(RecordChoice(f.record(), 'L', Source::Launcher, &after, &why) == Outcome::Written && after.seq == 3);
    // the game's state lock (wg_state.h, the lock settings.ini's writers take too) held by another thread, through the game
    // folder spelt differently: NoLock within the wait, nothing written
    std::atomic<bool> held{false}, release{false};
    std::thread holder([&] {
        std::wstring other = f.dir.wstring();
        for (auto& c : other) c = c == L'\\' ? L'/' : wchar_t(towupper(c));
        wgstate::StateLock l(other);
        held.store(l.held());
        while (!release.load()) Sleep(1);
    });
    while (!held.load()) Sleep(1);
    CHECK(RecordChoice(f.record(), 'K', Source::Ffx, &after, &why, nullptr, 100) == Outcome::NoLock);
    bool migrated = false;
    Record r;
    CHECK(LoadOrMigrateRecord(f.record(), [] { return 'M'; }, &r, &migrated, &why, 100) == LoadResult::NoLock);
    release.store(true);
    holder.join();
    r = Rec(f.record());
    CHECK(r.model == 'L' && r.seq == 3);
    // a record held open without any sharing: Unreadable, never written over
    h = CreateFileW(f.record().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(RecordChoice(f.record(), 'M', Source::Plugin, &after, &why) == Outcome::Unreadable);
    CHECK(LoadOrMigrateRecord(f.record(), [] { return 'M'; }, &r, &migrated, &why) == LoadResult::Unreadable);
    CloseHandle(h);
    CHECK(Rec(f.record()).seq == 3 && Rec(f.record()).model == 'L');
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

HANDLE Spawn2(std::wstring cmd) {  // another program's command line
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return nullptr;
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
    const DWORD readerCode = reader ? Wait(reader) : 999;
    if (readerCode != 0) std::printf("reader exit %lu\n", static_cast<unsigned long>(readerCode));
    CHECK(readerCode == 0);
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

// The upgrade migration reads wronggpu.ini's model item WHILE it holds the game's state lock, so a launcher apply (which
// holds the same lock across wronggpu.ini and the record) is wholly before or after it, in two REAL processes:
// (1) a launcher process takes the lock, writes wronggpu.ini (model=L) and dies before it writes the record: the waiting
//     migration reads L, never the M it would have read before the lock;
// (2) a launcher process takes the lock, writes wronggpu.ini and then the record (L, seq 1): the waiting migration finds
//     that record and writes nothing.
char IniModel(const fs::path& ini) {
    const std::string text = ReadRaw(ini);
    const size_t at = text.rfind("model=");
    return at == std::string::npos || at + 6 >= text.size() ? 0 : text[at + 6];
}

void TestMigrationUnderLock() {
    for (int crash = 1; crash >= 0; --crash) {
        Folder f(crash ? "-mig1" : "-mig2");
        const fs::path ini = f.dir / "wronggpu.ini", signal = f.dir / "locked.txt";
        WriteRaw(ini, "[fsr]\r\nruntime_options = model=M\r\n");
        HANDLE p = Spawn(std::wstring(crash ? L"apply-and-die" : L"apply") + L" \"" + f.dir.wstring() + L"\"");
        CHECK(p != nullptr);
        for (int i = 0; i < 5000 && !fs::exists(signal); ++i) Sleep(1);  // the launcher holds the lock now
        CHECK(fs::exists(signal));
        CHECK(IniModel(ini) == 'M');  // read now, before the lock, it would still be M
        Record r;
        bool migrated = false;
        std::string why;
        CHECK(LoadOrMigrateRecord(f.record(), [&] { return IniModel(ini); }, &r, &migrated, &why, 10000) == LoadResult::Ok);
        CHECK(r.model == 'L' && r.seq == 1 && r.source == Source::Launcher && migrated == (crash == 1));
        CHECK(Wait(p) == DWORD(crash ? 97 : 0));
        r = Rec(f.record());
        CHECK(r.model == 'L' && r.seq == 1);
        std::error_code ec;
        fs::remove(ini, ec);
        fs::remove(signal, ec);
    }
}

// Two path aliases of one game folder (a directory junction, and the \\?\ form) name ONE lock: a process holding the
// game's state lock through the junction keeps a writer that uses the real path out.
void TestAliasesOneLock() {
    Folder f("-alias");
    const std::wstring alias = f.dir.wstring() + L"-junction";
    RemoveDirectoryW(alias.c_str());
    HANDLE mk = Spawn2(L"cmd.exe /d /c mklink /J \"" + alias + L"\" \"" + f.dir.wstring() + L"\" >nul");
    CHECK(mk != nullptr && Wait(mk) == 0);
    std::wstring a, b, c;
    CHECK(wgstate::LockNameFor(f.dir.wstring(), &a) && wgstate::LockNameFor(alias, &b) &&
          wgstate::LockNameFor(L"\\\\?\\" + f.dir.wstring(), &c));
    CHECK(!a.empty() && a == b && a == c);
    const fs::path signal = f.dir / "held.txt";
    HANDLE p = Spawn(L"hold \"" + alias + L"\" 1500");
    for (int i = 0; i < 5000 && !fs::exists(signal); ++i) Sleep(1);
    CHECK(fs::exists(signal));
    Record after;
    std::string why;
    CHECK(RecordChoice(f.record(), 'K', Source::Plugin, &after, &why, nullptr, 200) == Outcome::NoLock);
    CHECK(p != nullptr && Wait(p) == 0);
    CHECK(RecordChoice(f.record(), 'K', Source::Plugin, &after, &why, nullptr, 2000) == Outcome::Written && after.seq == 1);
    std::error_code ec;
    fs::remove(signal, ec);
    CHECK(RemoveDirectoryW(alias.c_str()) != 0);  // the junction only; the folder stays
    CHECK(fs::exists(f.record()));
}

int ChildApply(const fs::path& gameDir, bool crash) {
    const fs::path record = RecordPathIn(gameDir);
    Lock lock(record);
    if (!lock.ok()) return 1;
    WriteRaw(gameDir / "locked.txt", "1");
    Sleep(300);
    WriteRaw(gameDir / "wronggpu.ini", "[fsr]\r\nruntime_options = model=L\r\n");
    if (crash) TerminateProcess(GetCurrentProcess(), 97);  // dies holding the lock, before the record
    Record after;
    std::string why;
    return RecordChoiceUnlocked(record, 'L', Source::Launcher, &after, &why) == Outcome::Written ? 0 : 2;
}

// The front end's upgrade migration for the game in gameDir, run by the launcher's end-to-end test against a REAL launcher
// apply in another process (run_model_l_tests.py): reads wronggpu.ini's model item under the state lock; prints the result.
int ChildMigrate(const fs::path& gameDir) {
    Record r;
    bool migrated = false;
    std::string why;
    const LoadResult lr = LoadOrMigrateRecord(RecordPathIn(gameDir), [&] { return IniModel(gameDir / "wronggpu.ini"); }, &r, &migrated, &why, 20000);
    std::printf("migrate: %s model=%c seq=%llu migrated=%d\n", lr == LoadResult::Ok ? "ok" : "failed", lr == LoadResult::Ok ? r.model : '-',
                static_cast<unsigned long long>(r.seq), migrated ? 1 : 0);
    return lr == LoadResult::Ok ? 0 : 1;
}

int ChildHold(const std::wstring& gameDir, int ms) {
    wgstate::StateLock lock(gameDir);
    if (!lock.held()) return 1;
    WriteRaw(fs::path(gameDir) / "held.txt", "1");
    Sleep(ms);
    return 0;
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
        Sleep(0);  // let the other processes take the lock between two writes
    }
    return bad;
}

int ChildReader(const fs::path& record, uint64_t until) {
    uint64_t last = 0;
    int bad = 0;
    const ULONGLONG start = GetTickCount64();
    while (last < until && GetTickCount64() - start < 100000) {
        {
            Lock lock(record, 20000);
            if (!lock.ok()) { std::printf("reader: lock not taken\n"); return 100; }
            Record r;
            std::string why;
            const ReadResult rr = ReadRecordFile(record, &r, &why);
            // a partial or corrupt record would be a failure; a read another program (a virus scanner looking at the new
            // file) briefly refused is not, and is only reported
            if (rr == ReadResult::Corrupt) {
                ++bad;
                std::printf("reader: corrupt (%s)\n", why.c_str());
            }
            if (rr == ReadResult::Unreadable) std::printf("reader: one read refused (%s)\n", why.c_str());
            if (rr == ReadResult::Ok) {
                if (r.seq < last) { ++bad; std::printf("reader: seq went back %llu -> %llu\n", (unsigned long long)last, (unsigned long long)r.seq); }
                last = r.seq;
            }
        }
        Sleep(1);  // a Windows mutex is not fair: a reader that takes it again at once could keep the writers out
    }
    return bad + (last < until ? 1000 : 0);
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc >= 6 && std::string(argv[1]) == "writer") return ChildWriter(argv[2], argv[3][0], argv[4], std::atoi(argv[5]));
    if (argc >= 4 && std::string(argv[1]) == "reader") return ChildReader(argv[2], std::strtoull(argv[3], nullptr, 10));
    if (argc >= 3 && std::string(argv[1]) == "migrate") return ChildMigrate(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "apply") return ChildApply(argv[2], false);
    if (argc >= 3 && std::string(argv[1]) == "apply-and-die") return ChildApply(argv[2], true);
    if (argc >= 4 && std::string(argv[1]) == "hold") return ChildHold(fs::path(argv[2]).wstring(), std::atoi(argv[3]));
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
    TestChoicesInTurn();
    TestStampedRecord();
    TestPlayerFiles();
    TestCorruptAndEnd();
    TestMigration();
#ifdef _WIN32
    TestFailures();
    TestConcurrentProcesses();
    TestAbandonedLock();
    TestMigrationUnderLock();
    TestAliasesOneLock();
#endif
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
