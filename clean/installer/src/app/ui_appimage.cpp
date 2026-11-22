// SPDX-License-Identifier: MIT
// WrongGPU launcher window: Settings > Uninstall WrongGPU, in the native Linux launcher (wg_core/appimage.h). An AppImage has no
// installer entry: the one question names the AppImage file and WrongGPU's data folder, says nothing else is touched, and the
// answer deletes those two. Written for this project.
#include "app/launcher_window.h"

#include <atomic>
#include <thread>

#include "app/launcher_ui_state.h"
#include "wg_core/appimage.h"
#include "wg_core/convert_ignore.h"

namespace wg::app::win {

extern bool dataRemoved;  // launcher_ui.cpp

// The question, with what goes by its path (the player decides on what they can read). Nothing in the games is touched. The
// player's ignore list is put back when the data folder goes (RemoveSharedData), so a folder that has one says so.
wstring AppImageUninstallText(const wstring& fileIn, const wstring& dataIn, bool keepsList) {
    const wstring file = ShownPath(fileIn), data = ShownPath(dataIn);
    return L"Uninstall WrongGPU?\n\nThis deletes two things and cannot be undone:\n\n"
           L"• The program file:\n  " + file + L"\n• WrongGPU's data folder (the converted Model M and Ray Reconstruction data, the games' list, "
           L"settings and logs):\n  " + data + L"\n\n" + (keepsList ? wstring(L"Your ignore-files.txt list is kept. ") : wstring()) +
           L"Nothing else is deleted. Your games and their files are not touched.";
}

// The dialog's text for this running copy (false: it is not a proven AppImage). The folder named and looked into for the ignore list is
// the one the uninstall deletes.
bool AppImageUninstallQuestion(wstring& text) {
    wstring file;
    if (!OwnAppImage(file, nullptr)) return false;
    text = AppImageUninstallText(file, AppDataPath(), lui::IgnoreListThere(AppDataPath()));
    return true;
}

void AppImageUninstallAsked() {
    using namespace lui;
    if (!v.live || v.busy || !NativeLinux()) return;
    wstring file, why;
    if (!OwnAppImage(file, &why)) { Inform(L"WrongGPU can't uninstall itself from here\n\n" + why, 2); return; }
    // what a game still has is read on a worker first, as the launcher's other uninstall reads it (a game on a share that is offline
    // holds each look, never the window)
    auto look = std::make_shared<SharedDataLook>();
    if (!WaitForRead(L"", kLookVerb, T_NONE, [look] { *look = LookAtSharedData(g.ctx); })) return;
    if (look->anyGame) { Inform(L"Your games still use WrongGPU\n\nUninstall WrongGPU from each game first (its page, More, Uninstall), then uninstall WrongGPU here. Nothing was deleted.", 1); return; }
    wstring question;
    if (!AppImageUninstallQuestion(question)) return;
    if (!Confirm(question, 0, L"Uninstall WrongGPU", L"Cancel", true)) return;
    auto kept = std::make_shared<wstring>();
    auto problem = std::make_shared<wstring>();
    auto ok = std::make_shared<bool>(false);
    if (!WaitForRead(L"", L"Uninstalling…", T_NONE, [kept, problem, ok] { *ok = UninstallAppImage(g.ctx, problem.get(), nullptr, kept.get()); })) return;
    if (!*ok) {
        InformDetails(L"WrongGPU couldn't be uninstalled.\n\nA file in its data folder may be in use, or a game may still have WrongGPU. Close what uses it, then try again.", kIconError, *problem);
        return;
    }
    dataRemoved = true;  // the window's close does not write its place and list into the folder that is gone
    Inform(L"WrongGPU is uninstalled\n\nThe program file and WrongGPU's data folder are deleted." +
               (kept->empty() ? wstring() : L" " + *kept) + L"\n\nThe window closes now.", 0);
    if (v.hwnd) DestroyWindow(v.hwnd);
}

}  // namespace wg::app::win

namespace wg::app::lui {
// What Settings says about the removal: where the program and its data are, or why this copy can't remove itself.
wstring AppImageSettingsWords(bool own, const wstring& file, const wstring& data, const wstring& why, bool keepsList, bool gamesHaveIt) {
    if (!own) return L"Uninstall WrongGPU is not available here. " + why;
    return L"The file: " + ShownPath(file) + L". Uninstall deletes that file and WrongGPU's data folder (" + ShownPath(data) + L") after you confirm. " +
           (keepsList ? wstring(L"Your ignore-files.txt list is kept. ") : wstring()) + L"Nothing else is deleted." +
           (gamesHaveIt ? wstring(L" Uninstall WrongGPU from your games first.") : wstring());
}
// Whether this copy runs from its own proven AppImage (the file and the environment do not change while it runs)
bool RunsFromOwnAppImage(wstring* file, wstring* why, wstring* host) {
    static wstring imageFile, imageWhy, imageHost;
    static const bool own = OwnAppImage(imageFile, &imageWhy, &imageHost);
    if (file) *file = imageFile;
    if (why) *why = imageWhy;
    if (host) *host = imageHost;
    return own;
}
// Only a plain file is promised: the uninstall keeps a link to the list, but the file the link points to goes with the folder
bool IgnoreListThere(const wstring& dataDir) {
    const DWORD a = GetFileAttributesW(Join(dataDir, wg::kIgnoreListName).c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY) && !(a & FILE_ATTRIBUTE_REPARSE_POINT);
}
// Whether a game still has WrongGPU, as the uninstall asks it (LookAtSharedData: the registered and the managed games), not by the
// loaded Library. In the live window the look runs on a worker (a game on an offline share holds each look, never the painting) and
// the page is drawn again when it is done; Settings shows the last finished answer, refreshed every few seconds. The button looks
// again itself before it asks anything. The dumps and offscreen frames look at once.
bool GamesBlockAppImageUninstall() {
    static std::atomic<int> answer{-1};  // -1 not looked yet, 0 no, 1 yes
    static std::atomic<bool> looking{false};
    static std::atomic<ULONGLONG> at{0};
    if (!v.live || !v.hwnd) return LookAtSharedData(Ctx()).anyGame;
    const ULONGLONG now = GetTickCount64();
    if (!looking.load() && (answer.load() < 0 || now - at.load() > 3000)) {
        looking = true;
        const HWND w = v.hwnd;
        std::thread([w] {
            answer = LookAtSharedData(Ctx()).anyGame ? 1 : 0;
            at = GetTickCount64();
            looking = false;
            PostMessageW(w, win::WM_APP_ART, 0, 0);  // drawn again (WM_APP_ART only repaints)
        }).detach();
    }
    return answer.load() == 1;
}
}  // namespace wg::app::lui
