// SPDX-License-Identifier: MIT
// WrongGPU launcher window: Settings > Uninstall WrongGPU, in the native Linux launcher (wg_core/appimage.h). An AppImage has no
// installer entry: the one question names the AppImage file and WrongGPU's data folder, says nothing else is touched, and the
// answer deletes those two. Written for this project.
#include "app/launcher_window.h"

#include "wg_core/appimage.h"

namespace wg::app::win {

extern bool dataRemoved;  // launcher_ui.cpp

// The question, with what goes by its path (the player decides on what they can read). Nothing in the games is touched.
wstring AppImageUninstallText(const wstring& fileIn, const wstring& dataIn) {
    const wstring file = ShownPath(fileIn), data = ShownPath(dataIn);
    return L"Uninstall WrongGPU?\n\nThis deletes two things and cannot be undone:\n\n"
           L"• The program file:\n  " + file + L"\n• WrongGPU's data folder (the converted Model M and Ray Reconstruction data, the games' list, "
           L"settings and logs):\n  " + data + L"\n\nNothing else is deleted. Your games and their files are not touched.";
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
    if (!Confirm(AppImageUninstallText(file, AppDataPath()), 0, L"Uninstall WrongGPU", L"Cancel", true)) return;
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
wstring AppImageSettingsWords(bool own, const wstring& file, const wstring& data, const wstring& why) {
    if (!own) return L"Uninstall WrongGPU is not available here. " + why;
    return L"Runs from " + ShownPath(file) + L". Uninstall deletes that file and WrongGPU's data folder (" + ShownPath(data) + L"), after you confirm. Nothing else is deleted.";
}
}  // namespace wg::app::lui
