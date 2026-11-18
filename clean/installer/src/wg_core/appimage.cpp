// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the native Linux launcher's own removal (appimage.h). Written for this project.
#include "appimage.h"

#include <filesystem>

#include "launcher.h"

namespace wg {

// A Linux path from the environment ("/home/x/WrongGPU.AppImage") as the core spells it (Z:\home\x\WrongGPU.AppImage).
static wstring CoreSpelling(wstring p) {
    if (!p.empty() && p[0] == L'/') p = L"Z:" + p;
    for (auto& c : p) if (c == L'/') c = L'\\';
    return p;
}

bool OwnAppImage(wstring& file, wstring* why) {
    auto no = [&](const wstring& w) { if (why) *why = w; return false; };
    const wstring env = EnvVar(L"APPIMAGE");
    if (env.empty()) return no(L"WrongGPU is not running from an AppImage here, so there is no file for it to delete. Delete the program file yourself.");
    const wstring p = CoreSpelling(env);
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(FsPath(p), ec);  // (a link is not followed: it is not the file)
    if (ec || !std::filesystem::is_regular_file(status))
        return no(p + L" is not a plain file (it is missing, a folder or a link), so WrongGPU will not delete it.");
    const wstring name = BaseName(p);
    if (!StartsWithI(name, L"WrongGPU") || !EndsWithI(name, L".AppImage"))
        return no(L"The file is called " + name + L", not WrongGPU...AppImage, so WrongGPU cannot tell it is its own and will not delete it.");
    // an AppImage is an ELF file with "AI" and the type number 2 in bytes 8 to 10 of its header
    unsigned char head[16] = {};
    DWORD got = 0;
    HANDLE h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    const bool read = h != INVALID_HANDLE_VALUE && ReadFile(h, head, sizeof head, &got, nullptr) && got == sizeof head;
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    if (!read || head[0] != 0x7f || head[1] != 'E' || head[2] != 'L' || head[3] != 'F' || head[8] != 'A' || head[9] != 'I' || head[10] != 2)
        return no(name + L" is not an AppImage file, so WrongGPU will not delete it.");
    const wstring appdir = CoreSpelling(EnvVar(L"APPDIR"));
    const wstring self = SelfPath();
    if (appdir.empty() || !StartsWithI(self, appdir + L"\\"))
        return no(L"This running program is not inside " + name + L", so WrongGPU cannot tell the file is the one it was started from and will not delete it.");
    file = p;
    return true;
}

bool UninstallAppImage(Context& ctx, wstring* why, wstring* removedFile, wstring* kept) {
    if (removedFile) removedFile->clear();
    wstring file;
    if (!OwnAppImage(file, why)) return false;
    // the data first, all or nothing: if it can't go (a game still has WrongGPU, a file in use) the file stays too
    if (!RemoveSharedData(ctx, why, kept, true)) return false;
    if (!DeleteFileW(file.c_str())) {
        if (why) *why = L"WrongGPU's data is removed, but " + file + L" could not be deleted (" + ErrText(GetLastError()) + L"). Delete it yourself.";
        return false;
    }
    if (removedFile) *removedFile = file;
    return true;
}

}  // namespace wg
