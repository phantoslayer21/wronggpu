// SPDX-License-Identifier: GPL-3.0-only
#pragma once

namespace {
void VerifyIniEncoding() {
    const string from = "[FrameGen]\r\n\tFGOutput \t= auto \t; player's note\r\nKeep=1\n";
    const string to = "[FrameGen]\r\n\tFGOutput \t= dlssg \t; player's note\r\nKeep=1\n";
    auto wide = [](const string& text, bool bigEndian) {
        string bytes = bigEndian ? "\xfe\xff" : "\xff\xfe";
        for (char c : text) {
            bytes.push_back(bigEndian ? '\0' : c);
            bytes.push_back(bigEndian ? c : '\0');
        }
        return bytes;
    };
    for (const auto& pair : {std::make_pair(from, to), std::make_pair("\xef\xbb\xbf" + from, "\xef\xbb\xbf" + to),
                             std::make_pair(wide(from, false), wide(to, false)), std::make_pair(wide(from, true), wide(to, true))}) {
        string value;
        require(IniHasKey(pair.first, "framegen", "fgoutput", &value) && value == "auto" &&
                IniSet(pair.first, "FrameGen", "FGOutput", "dlssg") == pair.second,
                "OptiScaler INI editing preserves BOM, line endings, indentation and inline comments");
    }
    bool refused = false;
    try { IniSet(from + "[FrameGen]\r\nFGOutput=other\r\n", "FrameGen", "FGOutput", "dlssg"); }
    catch (const SetupError& error) { refused = error.msg.find(L"duplicate") != wstring::npos; }
    require(refused, "duplicate OptiScaler keys are refused with a clear error");
    const string empty = "[FrameGen]\r\nFGOutput= ; note\r\n";
    const string filled = IniSet(empty, "FrameGen", "FGOutput", "dlssg");
    string value;
    require(filled == "[FrameGen]\r\nFGOutput=dlssg ; note\r\n" &&
                IniHasKey(filled, "FrameGen", "FGOutput", &value) && value == "dlssg",
            "setting an empty value leaves whitespace before the comment");
    const string unknown = "[Other]\r\nChoice=hello ; world #2\r\n";
    require(IniHasKey(unknown, "Other", "Choice", &value) && value == "hello ; world #2",
            "an unknown key's entire value is preserved");
    const string duplicates = from + "[FrameGen]\r\nFGOutput=other\r\n";
    require(IniHasKey(duplicates, "FrameGen", "FGOutput", &value) && value == "auto" &&
                !OptiReadWarning(duplicates).empty() && !OptiFgConflictText(duplicates).empty(),
            "duplicate OptiScaler keys remain readable and produce a warning");
    const string broken = wide(from, false) + "x";
    require(!IniHasKey(broken, "FrameGen", "FGOutput") && OptiTabRows(broken).empty() &&
                !OptiReadWarning(broken).empty(), "odd UTF-16 yields a read warning");
    string output;
    wstring why;
    const string dupMenu = "[Menu]\r\nShortcutKey=auto\r\nShortcutKey=0\r\n";
    require(!ApplyOptiEdits(dupMenu, {{"Menu.ShortcutKey", "36"}}, output, &why) && !why.empty(),
            "OptiScaler tab edits refuse a duplicate key");
}

void VerifyLauncherScope() {
    require(OwnsLauncherRegistration(false, L"C:\\Portable\\WrongGPU.exe", L"C:\\Installed\\WrongGPU.exe") &&
            OwnsLauncherRegistration(true, L"C:\\Launcher\\WrongGPU.exe", L"c:\\launcher\\wronggpu.exe") &&
            !OwnsLauncherRegistration(true, L"C:\\Portable\\WrongGPU.exe", L"C:\\Installed\\WrongGPU.exe") &&
            !OwnsLauncherRegistration(true, L"C:\\Portable\\WrongGPU.exe", L""),
            "an explicit root only owns the running program's registration");
    const wstring longFixture = TempFolder() + L" long path fixture";
    const wstring longData = Join(longFixture, L"long data folder");
    EnsureDir(longData);
    wchar_t shortFixture[MAX_PATH * 2] = {}, shortData[MAX_PATH * 2] = {};
    const DWORD fixtureLength = GetShortPathNameW(longFixture.c_str(), shortFixture, static_cast<DWORD>(std::size(shortFixture)));
    const DWORD dataLength = GetShortPathNameW(longData.c_str(), shortData, static_cast<DWORD>(std::size(shortData)));
    if (fixtureLength && fixtureLength < std::size(shortFixture) && dataLength && dataLength < std::size(shortData) &&
        !IEq(longData, shortData)) {
        for (const auto& pair : {std::make_pair(longFixture, wstring(shortFixture)), std::make_pair(longData, wstring(shortData))}) {
            require(PathWithin(Join(pair.first, L"missing\\child"), pair.second, true) &&
                        PathWithin(Join(pair.second, L"missing\\child"), pair.first, true),
                    "scope accepts long and short ancestors in both directions, including missing descendants");
        }
    } else std::cout << "Skipped short-name check (volume does not expose short folder names).\n";
    RemoveTree(longFixture);
}
void VerifyPublishedBinaryPin() {
    const wstring folder = TempFolder();
    const wstring path = Join(folder, L"published.dll");
    string image;
    require(ReadAll(SelfPath(), image), "binary publication fixture can read its own PE image");
    WriteFileAtomic(path, "old", 3);
    HANDLE section = nullptr;
    bool refused = false;
    {
        const auto operation = GuardOperation(folder, nullptr);
        PinnedFile old;
        require(old.Open(path, true), "binary publication fixture pins the old target");
        SetFaultHook([&](const char* point) {
            if (strcmp(point, "install-binary-reopen")) return false;
            HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file != INVALID_HANDLE_VALUE) {
                section = CreateFileMappingW(file, nullptr, PAGE_READONLY | SEC_IMAGE, 0, 0, nullptr);
                CloseHandle(file);
            }
            return false;
        });
        try { PutBinary(path, image.data(), image.size(), &old); }
        catch (const SetupError& error) { refused = error.code == EXIT_REFUSED; }
        SetFaultHook({});
    }
    const bool mapped = section != nullptr;
    if (section) CloseHandle(section);
    RemoveTree(folder);
    require(mapped && refused, "re-opening a published binary refuses a mapped image");
}

void VerifyAdoptionRecordLock() {
    const wstring folder = TempFolder();
    const wstring record = RecordPath(folder);
    Record initial;
    initial.game = folder;
    initial.exe = Join(folder, L"Game.exe");
    initial.version = L"before update";
    EnsureDir(StateDir(folder));
    SaveRecord(record, initial);
    SaveProfile(ProfilePath(folder), DefaultProfile());

    const HANDLE locked = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    const HANDLE adopted = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    require(locked && adopted, "adoption interleaving events created");
    bool updateHeld = false, updateSaved = false;
    std::thread update([&] {
        wgstate::StateLock lock(folder);
        updateHeld = lock.held();
        SetEvent(locked);
        if (!updateHeld) return;
        WaitForSingleObject(adopted, 1200);
        Record later = initial;
        later.version = L"concurrent update";
        later.extra.push_back(L"fixture-field\tpublished by update");
        try {
            SaveRecord(record, later);
            updateSaved = true;
        } catch (const SetupError&) {}
    });
    WaitForSingleObject(locked, 5000);
    const string restorer = "fixture restorer";
    Context ctx;
    ctx.restorer = restorer.data();
    ctx.restorerSize = restorer.size();
    bool adoptedOK = false;
    try { AdoptInstall(ctx, folder); adoptedOK = true; } catch (const SetupError&) {}
    SetEvent(adopted);
    update.join();
    CloseHandle(adopted);
    CloseHandle(locked);

    Record actual;
    const bool correct = LoadRecord(record, actual) && actual.version == L"concurrent update" &&
        actual.restorerSha == Sha256Bytes(restorer.data(), restorer.size()) &&
        actual.extra == std::vector<wstring>{L"fixture-field\tpublished by update"};
    RemoveTree(folder);
    require(updateHeld && updateSaved && adoptedOK && correct, "adoption waits for an update and saves its restorer hash on the updated record");
}

#ifdef WRONGGPU_TESTING
void VerifyScopedUninstallSnapshot(const wstring& fixture) {
    const wstring root = Join(fixture, L"scoped-uninstall");
    EnsureDir(root);
    wstring why;
    require(SetDataOverride(root, why, true), "managed list race uses its own data root");
    const wstring program = Join(root, L"portable.exe");
    const wstring list = Join(root, L"managed-games.txt");
    const wstring other = Join(root, L"player-list.txt");
    WriteFileAtomic(program, "MZ", 2);
    WriteFileAtomic(list, "; empty\n", 8);
    WriteFileAtomic(other, "; preserve\n", 11);
    bool swapped = false;
    LauncherPaths paths;
    paths.custom = true; paths.dataDir = root; paths.exe = program;
    Context ctx;
    Ui quiet{[](int, int, const wstring&) {}, [](const wstring&) {}};
    const bool ok = SelfUninstall(ctx, paths, false, L"-", quiet, &why,
        [&](const std::vector<OpRequest>&) {
            swapped = DeleteFileW(list.c_str()) && CreateHardLinkW(list.c_str(), other.c_str(), nullptr);
            return std::vector<OpResult>{};
        });
    string bytes;
    require(swapped && ok && !IsFile(program) && ReadAll(other, bytes) && bytes == "; preserve\n" && IsFile(list),
            "scoped uninstall uses its preflight snapshot if the list is swapped during the game operations");
}
#endif

}  // namespace
