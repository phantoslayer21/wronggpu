// SPDX-License-Identifier: MIT
// WrongGPU launcher window: an operation's result in plain words (SHIP-BAR 4): the game by its name, the action by the
// button's word, a full stop, the cause and what to do; the core's own text (folders, file names, Windows' error text) goes
// behind "Show details" and into the log. Split from launcher_ui.cpp (the window, its input and operations), which uses them
// with the offscreen frames (launcher_window.h). Written for this project.
#include "app/launcher_window.h"

#include <shellapi.h>
#include <shlobj.h>

#include "wg_core/rr.h"
#include "wg_core/rrconfig.h"

namespace wg::app::win {

wstring Sentence(wstring s) {  // a capital first letter and a full stop
    while (!s.empty() && (s.back() == L' ' || s.back() == L'\n' || s.back() == L'\r')) s.pop_back();
    if (s.empty()) return s;
    s[0] = towupper(s[0]);
    if (s.back() != L'.' && s.back() != L'?' && s.back() != L'!') s += L'.';
    return s;
}
bool Has(const wstring& text, const wchar_t* part) { return Lower(text).find(Lower(part)) != wstring::npos; }
// The failure in the page's callout, with the verb of its note's title ("... couldn't be repaired." -> "Couldn't repair.")
wstring OpFailLead(Op op, int as) {
    if (op == Op::Uninstall && as == RES_GONE) return L"Couldn't remove the files.";  // a game that is gone: its button's words
    switch (op) {
        case Op::Install: return L"Couldn't install."; case Op::Update: return L"Couldn't update."; case Op::Repair: return L"Couldn't repair.";
        case Op::Uninstall: return L"Couldn't uninstall."; case Op::Rollback: return L"Couldn't roll back."; case Op::Apply: return L"Couldn't apply the settings.";
        case Op::CleanLeftovers: return L"Couldn't clean up."; default: return L"Couldn't adopt.";
    }
}
wstring OpFailTitle(Op op, const wstring& name, int as) {
    if (op == Op::Uninstall && as == RES_GONE) return L"WrongGPU's files couldn't be removed from " + name + L"'s old folder.";
    switch (op) {
        case Op::Install: return L"WrongGPU couldn't be installed in " + name + L".";
        case Op::Update: return name + L" couldn't be updated.";
        case Op::Repair: return name + L" couldn't be repaired.";
        case Op::Uninstall: return L"WrongGPU couldn't be uninstalled from " + name + L".";
        case Op::Rollback: return name + L" couldn't be rolled back.";
        case Op::Apply: return L"The settings for " + name + L" couldn't be applied.";
        case Op::CleanLeftovers: return L"The leftover files in " + name + L" couldn't be removed.";
        default: return name + L" couldn't be adopted.";
    }
}
// The core's message without its code ("... [WG-SETUP-4]": the log's and Show details', never the surface's)
static wstring WithoutCode(wstring s) {
    const size_t at = s.rfind(L" [WG-SETUP-");
    if (at != wstring::npos && !s.empty() && s.back() == L']') s.resize(at);
    return s;
}
// What went wrong in plain words, with what to do next, from the core's message (its exit code where it says more). Every
// kind the launcher can meet is named here (an allow-list): a core sentence not named below, new or changed, never reaches
// the surface as it is (it names files, folders, Windows' error text, or the command line's words); it gets the general
// line, its own words behind Show details and in the log.
wstring PlainCause(const wstring& rawIn, int code) {
    const wstring raw = WithoutCode(rawIn);
    // the launcher's own uninstall: the game it stopped at, then that game's cause in words (and what it did before it)
    static const wchar_t* const kLeads[] = {L"WrongGPU could not be uninstalled from ", L"WrongGPU could not get its own uninstaller back in "};
    for (int k = 0; k < 2; ++k) {
        if (raw.rfind(kLeads[k], 0) != 0) continue;
        const size_t a = wcslen(kLeads[k]), dot = raw.find(L". ", a);
        if (dot == wstring::npos) break;
        const wstring game = raw.substr(a, dot - a);
        wstring inner = raw.substr(dot + 2), others;
        if (const size_t cut = inner.find(L" The launcher stays installed"); cut != wstring::npos) inner.resize(cut);
        if (const size_t o = inner.find(L" other game"); o != wstring::npos) {  // "... 2 other games have their original files back."
            const size_t s = inner.rfind(L". ", o);
            if (s != wstring::npos) { others = inner.substr(s + 2); inner.resize(s + 1); }
        }
        return (k == 0 ? L"WrongGPU couldn't be removed from " : L"WrongGPU couldn't give its own uninstaller back to ") + game + L". " + PlainCause(inner, 0) +
               (others.empty() ? L"" : L" " + others) + L" The launcher stays installed.";
    }
    if (code == EXIT_NO_MODEL || Has(raw, L"no usable DLSS") || Has(raw, L"No DLSS model file") || Has(raw, L"No model data to install"))
        return L"No DLSS file WrongGPU can use was found. Install or update a recent game with DLSS, or choose its DLSS file in Settings.";
    if (Has(raw, L" is running")) return L"The game is running. Close it and try again.";
    // what the game is or has (the game page says these before an install; a scan since can find them first)
    if (code == EXIT_ANTICHEAT || Has(raw, L"anti-cheat")) return L"This game uses anti-cheat, and WrongGPU stays out of games with anti-cheat.";
    if (Has(raw, L"Not a D3D12 game") || Has(raw, L"Direct3D 12 games only")) return L"This game doesn't use Direct3D 12, and WrongGPU works in Direct3D 12 games only.";
    if (Has(raw, L"protected WindowsApps folder"))
        return L"This game is in the Xbox app's protected folder, which doesn't allow mods. WrongGPU can't be installed there.";
    if (Has(raw, L"Two OptiScaler copies") || Has(raw, L"loads OptiScaler twice"))
        return L"This game loads OptiScaler twice. Remove one of the two copies yourself, then try again.";
    if (Has(raw, L"older WrongGPU (DLSS 4.5 AMD) install") || Has(raw, L"older install's record") || Has(raw, L"backup of the older install"))
        return L"An older WrongGPU (DLSS 4.5 AMD) install in this game can't be removed automatically. Uninstall it with its own uninstaller, then try again.";
    // the PC (the first run and Settings say these too)
    if (Has(raw, L"RX 9000-series")) return L"WrongGPU needs an AMD Radeon RX 9000-series graphics card.";
    if (Has(raw, L"64-bit Windows 10 or 11")) return L"WrongGPU needs 64-bit Windows 10 or 11.";
    if (Has(raw, L"HIP runtime") || Has(raw, L"HIP SDK"))
        return L"The AMD HIP runtime WrongGPU needs isn't working. Update the AMD Software: Adrenalin Edition driver, then try again.";
    // the launcher's own Start menu shortcut and Installed apps entry (installing it for this user)
    if (Has(raw, L"Apps & features entry"))
        return L"Windows didn't let WrongGPU add its entry to Installed apps. Try again; if it keeps happening, collect the logs and report it.";
    if (Has(raw, L"cannot write the shortcut"))
        return L"Windows didn't let WrongGPU add its Start menu shortcut. Try again; if it keeps happening, collect the logs and report it.";
    if (Has(raw, L"installed launcher aside") || Has(raw, L"new launcher in place") || Has(raw, L"after copying the launcher"))
        return L"The installed copy of WrongGPU couldn't be replaced. Close every WrongGPU window, then try again.";
    // this copy of WrongGPU itself
    if (Has(raw, L"installer is damaged") || Has(raw, L"Cannot read OptiScaler") || raw.rfind(L"OptiScaler: ", 0) == 0)
        return L"This copy of WrongGPU is damaged. Download it again.";
    if (Has(raw, L"is the game running") || Has(raw, L"being used by another process") || Has(raw, L"sharing violation") || Has(raw, L"locked a portion"))
        return L"A game file is in use. Close the game and any program that has its files open, then try again.";
    if (Has(raw, L"access is denied")) return L"Windows didn't allow a change in the game's folder. Close the game and try again.";
    if (Has(raw, L"not enough space") || Has(raw, L"disk is full")) return L"The drive is full. Free some space and try again.";
    if (Has(raw, L"needs a free name"))
        return L"OptiScaler needs a free file name next to the game, and every name it can use is taken by other mods' files. Remove one of them (see Game folder), "
               L"then try again.";
    if (Has(raw, L"missing or damaged") || (raw.rfind(L"Backup of ", 0) == 0 && Has(raw, L" is missing")))
        return L"A backup WrongGPU made of a game file is missing or damaged. Nothing was changed. Verify the game's files in its store, then use Repair.";
    // WrongGPU's own record of an install, and what it keeps for Roll back
    if (Has(raw, L"in the install record") || Has(raw, L"record belongs to a different folder"))
        return L"WrongGPU's record of its files in this game is damaged, so it won't change them. Collect the logs and report it.";
    if (Has(raw, L"Refusing to follow a link"))
        return L"A folder in this game is a link to another place, and WrongGPU doesn't change files through links. Collect the logs and report it.";
    if (Has(raw, L"no earlier version to roll back")) return L"There's no earlier version of WrongGPU in this game to go back to.";
    if (Has(raw, L"kept earlier version is damaged"))
        return L"The earlier version WrongGPU kept in this game is damaged, so it can't go back to it. Nothing was changed. Use Repair instead.";
    if (Has(raw, L"Cannot keep the previous version"))
        return L"WrongGPU couldn't keep the current version for Roll back, so nothing was updated. Close the game and try again.";
    if (Has(raw, L"OptiScaler.ini is missing")) return L"OptiScaler's settings file is missing in this game. Use Repair to put it back.";
    if (Has(raw, L"no leftover WrongGPU files")) return L"There are no leftover WrongGPU files in this game any more. Rescan the Library.";
    if (code == EXIT_NOT_INSTALLED || Has(raw, L"is not installed in") || Has(raw, L"no wrongGPU installation record"))
        return L"WrongGPU isn't installed in this game any more. Rescan the Library.";
    if (Has(raw, L"verification failed")) return L"A file didn't match after it was copied. Try again; if it keeps happening, collect the logs and report it.";
    if (Has(raw, L"cannot write to the game folder")) return L"WrongGPU can't write to the game's folder. Close the game and try again.";
    if (Has(raw, L"game's files could not be read")) return L"WrongGPU couldn't read the game's files. Check that its drive is connected, then try again.";
    if (Has(raw, L"No game executable found") || Has(raw, L"exe was not found") || Has(raw, L"logs are collected from its folder"))
        return L"The game isn't in its folder any more. Rescan the Library.";
    // the game's upscaler files: none WrongGPU can replace, or another mod's in their place
    if (Has(raw, L"No AMD FSR 3 DLL") || Has(raw, L"can be replaced by this installer") || Has(raw, L"has no WrongGPU replacement"))
        return L"This game has no upscaler file WrongGPU can replace.";
    if (Has(raw, L"is not the game's FSR DLL") || Has(raw, L"Files of another mod") || code == EXIT_OTHER_MODS)
        return L"Another upscaler mod's files are in this game. WrongGPU stays out so the two don't fight over the upscaler. Remove that mod, then try again.";
    if (Has(raw, L"already has an install record")) return L"WrongGPU is already installed in this game. Rescan the Library.";
    // Model M's data as an install found it (model.cpp): changed under it, or failing its check once saved
    if (Has(raw, L"changed while setup was running")) return L"Model M's data changed while WrongGPU was installing. Try again.";
    if (Has(raw, L"model data failed its check")) return L"Model M's data is damaged. Convert it again in Settings.";
    if (Has(raw, L"without an install record")) return L"Leftover WrongGPU files are in this game. Use Clean up or Adopt on its page, then try again.";
    if (Has(raw, L"settings not applied")) return L"One of these settings has a value WrongGPU can't use. Choose it again on the game's page, then Apply.";
    // a game that blocks OptiScaler without the player's REFramework (gamerules.cpp NeedsReFrameworkText): what to do, without
    // the web address the core names (Learn more on the game's page opens it)
    if (const size_t n = raw.find(L" needs REFramework installed first"); n != wstring::npos && n > 0)
        return raw.substr(0, n) + L" needs REFramework first. Install it in the game's folder, then try again.";
    // nothing else is said as the core put it (no pass-through: a sentence the list above doesn't name may hold a path, a
    // file name, Windows' error text or the command line's words); its own text stays behind Show details
    return L"Something went wrong. Try again; if it keeps happening, collect the logs and report it.";
}
// The game's name in a core sentence, between `before` and `after` ("... isn't offered in <game>, and ..."); `fallback`
// when the sentence is not as expected
static wstring NameBetween(const wstring& w, const wchar_t* before, const wchar_t* after, const wchar_t* fallback) {
    const size_t a = w.find(before);
    const size_t s = a == wstring::npos ? a : a + wcslen(before), b = s == wstring::npos ? s : w.find(after, s);
    return b == wstring::npos || b <= s ? wstring(fallback) : w.substr(s, b - s);
}
// A warning of a successful operation in plain words (SHIP-BAR 4); empty: not for the player. Every warning the core can
// give an Install, Update, Repair or Uninstall is named here by its kind; the core's own sentence (DLL names, OptiScaler's
// settings, folders) never reaches the note's body, only Show details and the log.
wstring PlainWarning(const wstring& w) {
    // the HIP runtime check: first run already says HIP is optional
    if (Has(w, L"HIP runtime not usable")) return L"";
    // what else is in the game folder (coexist.cpp CoexistReport::Warnings): a tool that simply stays (Cyber Engine Tweaks,
    // RED4ext, UE4SS; Ultimate ASI Loader with WrongGPU's plugin in a folder of its own; another tool's loose file; an older
    // OptiScaler's DLSS copy) is the game page's to show (Game folder > Also here), not a note after every update
    if (Has(w, L"Stays where it is: its ") || Has(w, L"It loads every .asi file") || Has(w, L"Old OptiScaler layout") ||
        Has(w, L"Another tool's file: WrongGPU leaves it alone"))
        return L"";
    // ReShade or Special K next to it: the pair is untested, and what happens then shows after a run (the proxy line says
    // the same as the tool's own)
    if (w.rfind(L"OptiScaler as ", 0) == 0 && Has(w, L"untested combination")) return L"";
    if (Has(w, L"OptiScaler takes another free name"))
        return wstring(w.rfind(L"Special K", 0) == 0 ? L"Special K" : L"ReShade") + L" is in this game too. If Model M switches to AMD FSR next to it, Last run shows it.";
    if (Has(w, L"global injector is running")) return L"Special K's global injector is running and may load into this game too. That pair isn't tested with WrongGPU.";
    if (Has(w, L"Its own HIP runtime may be loaded"))
        return L"Another program in this game brings its own HIP runtime. Keep Runs on at Auto or Direct3D 12 so the two don't clash.";
    if (Has(w, L"An unknown d3d12.dll"))
        return L"Another program's Direct3D 12 file is in this game's folder. WrongGPU leaves it alone; if the game doesn't start with WrongGPU, that file may be why.";
    if (Has(w, L"no [Plugins] Path setting")) return L"Another mod loader in this game might start WrongGPU's plugin twice.";
    // the checks before an install (checks.cpp L_FFX)
    if (Has(w, L"no DLSS, XeSS or FSR files found"))
        return L"No DLSS, XeSS or FSR files were found in this game, so it may not offer an upscaler for WrongGPU to replace.";
    // Model M's conversion (the converter's note on this DLSS build's output filter)
    if (Has(w, L"output filter of this DLSS build is not known")) return L"WrongGPU doesn't know the output filter of this game's DLSS file yet, so Model M uses its own.";
    if (Has(w, L"Only the selected model is prepared")) return L"Only the model you chose is ready. The other one needs a DLSS file it can be converted from.";
    // a game that needs the player's REFramework (gamerules.h): gone since the install, or this build without the OptiScaler
    // edition its Ray Reconstruction needs (the build detail stays in the log)
    if (Has(w, L"REFramework isn't in this game any more")) {
        const wstring game = NameBetween(w, L"any more: ", L" blocks OptiScaler", L"This game");
        return L"REFramework isn't in " + game + L"'s folder any more, so the game won't start WrongGPU. Install REFramework again to use WrongGPU there.";
    }
    if (Has(w, L"no OptiScaler edition")) {
        const wstring game = NameBetween(w, L"isn't offered in ", L", and the game", L"this game");
        return L"WrongGPU's Ray Reconstruction isn't available in " + game + L" with the OptiScaler you chose. It needs the WrongGPU edition of OptiScaler, which the launcher downloads when it can. The game keeps its own ray tracing.";
    }
    if (Has(w, L"The game replaced ")) return L"The game had replaced one of its files since WrongGPU was installed (probably a game update). Its newer file was kept.";
    if (Has(w, L"exists again with different content") || Has(w, L"The backup folder was kept"))
        return L"A game file the game changed was kept as it is. Its original stays in a backup folder in the game's folder.";
    if (Has(w, L"Older install: left in place")) return L"A file of the older WrongGPU install was changed since, so it was left in place.";
    if (Has(w, L"Left in place (changed since install)")) return L"A WrongGPU file was changed after it was installed, so it was left in place.";
    if (Has(w, L"newer version could not be kept")) return L"The newer version couldn't be kept for a later update.";
    // Ray Reconstruction (rr.h): the data missing after this game's install, or another game's settings not written
    if (Has(w, L"Ray Reconstruction data isn't converted yet"))
        return L"Ray Reconstruction stays off in this game until its data is converted. Convert it from the game's page or in Settings.";
    if (Has(w, L"other RR games' settings couldn't be written")) return L"Ray Reconstruction is ready, but it stays off in your other games that use it. Use Repair on their pages.";
    if (w.rfind(kRrNotUpdated, 0) == 0) {  // "... the settings of <game> (<folder>) couldn't be written: ..."
        const size_t a = wcslen(kRrNotUpdated), c = w.find(L":\\", a), b = c == wstring::npos ? w.find(L" (", a) : w.rfind(L" (", c);
        const wstring game = b == wstring::npos || b < a ? wstring(L"another game") : w.substr(a, b - a);
        return L"Ray Reconstruction is ready, but it stays off in " + game + L": its settings couldn't be updated. Use Repair on its page.";
    }
    // the game's Engine.ini (wg_core/rrconfig.h): the setting not added (Fix it for me, a repair), or not taken out again
    if (w.rfind(L"Ray Reconstruction setting not added for ", 0) == 0)
        return L"WrongGPU couldn't add the Ray Reconstruction setting to the game's Engine.ini. Select Show details to see why. "
               L"You can add it yourself: the release notes have the steps.";
    if (w.rfind(L"Engine.ini could not be put back", 0) == 0)
        return L"WrongGPU couldn't take its Ray Reconstruction line out of the game's Engine.ini. Select Show details for the file and the line.";
    return L"WrongGPU noted one more thing. Select Show details to read it.";  // a warning not named above: the core's words behind Show details only
}

// The games a launcher uninstall left as they are because their drive isn't connected (SelfUninstall): they keep WrongGPU
// with an entry of their own in Installed apps (written from the launcher's list), and the data they read stays until
// WrongGPU is uninstalled from them there once the drive is back
wstring OfflineGamesWords(const std::vector<wstring>& names, bool removeGames) {
    if (names.empty()) return L"";
    const bool one = names.size() == 1;
    const wstring who = one ? names[0] + L" is on a drive that isn't connected, so it keeps WrongGPU as it is, with its own entry in Installed apps. The data it uses "
                              L"stays in your user folder until WrongGPU is uninstalled from it."
                            : std::to_wstring(names.size()) + L" games are on drives that aren't connected, so they keep WrongGPU as they are, each with its own entry in "
                              L"Installed apps. The data they use stays in your user folder until WrongGPU is uninstalled from them.";
    if (removeGames) return who + (one ? L" To remove WrongGPU from it, connect the drive, then uninstall its WrongGPU entry in Installed apps."
                                       : L" To remove WrongGPU from them, connect the drives, then uninstall their WrongGPU entries in Installed apps.");
    return who;
}
// The launcher's uninstall done (Settings, Installed apps): anyGame and games as asked (UninstallLauncherText, its answer),
// here: the installed launcher's own window; offline: the games left as they are (OfflineGamesWords); others: it did other
// games besides those. One name for the action through its flow (SHIP-BAR 4): the button "Uninstall launcher...", the question
// "Uninstall the WrongGPU launcher?", this "The WrongGPU launcher is uninstalled."
wstring LauncherUninstalledText(bool anyGame, bool games, bool here, const std::vector<wstring>& offline, bool others) {
    const wstring stays = here ? L"" : L" This portable copy stays open.";
    const wstring off = OfflineGamesWords(offline, games), offAfter = off.empty() ? L"" : (others ? L" " : L"") + off;
    const wstring done = here ? L"The WrongGPU launcher is uninstalled." : L"The installed WrongGPU launcher is uninstalled.";
    if (!anyGame) return done + (off.empty() ? L"" : L"\n\n" + off) + (here ? L"" : off.empty() ? L"\n\nThis portable copy stays open." : stays.c_str());
    if (games)
        return done + (off.empty() ? L"\n\nEvery game has its original files back." : others ? L"\n\nThe other games have their original files back." : L"\n\n") + offAfter + stays;
    return done + (off.empty() ? L"\n\nYour games keep WrongGPU. Each one has its own entry in Installed apps."
                               : others ? L"\n\nYour other games keep WrongGPU, each with its own entry in Installed apps." : L"\n\n") + offAfter + stays;
}
// Repair on an older WrongGPU, an update too: its question (rrSetup: the RR row's Set up, a repair, in its own words)
wstring RepairUpdateText(const wstring& name, const wstring& from, const wstring& to, bool rrSetup) {
    if (rrSetup)
        return L"Set up Ray Reconstruction in " + name + L"?\n\nThis also updates " + name + L" from WrongGPU " + from + L" to " + to + L": Ray Reconstruction comes with "
               L"WrongGPU " + to + L"'s files. Its settings stay, and Roll back returns it to " + from + L".";
    return L"Repair and update " + name + L" to WrongGPU " + to + L"?\n\nRepair puts back WrongGPU " + to + L"'s files, so " + name + L" moves from WrongGPU " + from +
           L" to " + to + L". Its settings stay, and Roll back returns it to " + from + L".";
}
// An update or repair of a game an earlier build set up to start on Preset K (a build without Preset K): the plain choice
wstring PresetKText(const wstring& name, const wstring& to) {
    return L"Switch " + name + L" to Model M?\n\n" + name + L" starts on Preset K, which WrongGPU " + to + L" doesn't have. Updating moves it to Model M, "
           L"and its other settings stay. Not now leaves " + name + L" as it is, on Preset K.";
}
// The games Update all promised but left out (name; true: gone since the scan, false: not readable, its note said why), in
// words for its end; empty: none
wstring SkippedWords(const std::vector<std::pair<wstring, bool>>& skipped) {
    if (skipped.empty()) return L"";
    wstring names;
    bool allGone = true;
    for (size_t k = 0; k < skipped.size(); ++k) {
        names += (k == 0 ? L"" : k + 1 == skipped.size() ? L" and " : L", ") + skipped[k].first;
        allGone = allGone && skipped[k].second;
    }
    const bool one = skipped.size() == 1;
    if (allGone) return names + (one ? L" wasn't updated: it isn't in its folder any more. The Library looks for it again."
                                     : L" weren't updated: they aren't in their folders any more. The Library looks for them again.");
    return names + (one ? L" wasn't updated." : L" weren't updated.");
}

// The cause of a failed launcher task in plain words (SHIP-BAR 4): the launcher's own folder, its Start shortcut, its entry
// in Installed apps, the logs file; never the game's folder and the game's advice (PlainCause), except for a game that
// stopped the launcher's uninstall, which is said as that game's
wstring PlainTaskCause(const wstring& raw, int what) {
    static const wchar_t* const kAgain = L" Try again; if it keeps happening, collect the logs and report it.";
    const bool inUse = Has(raw, L"being used by another process") || Has(raw, L"sharing violation") || Has(raw, L"locked a portion");
    const bool denied = Has(raw, L"access is denied");
    const bool full = Has(raw, L"not enough space") || Has(raw, L"disk is full");
    if (full) return L"The drive is full. Free some space and try again.";
    if (what == kCauseLogs) {
        if (Has(raw, L"logs are collected from its folder")) return L"The game's folder isn't there any more. Rescan the Library, then try again.";
        if (inUse) return L"A log file is in use. Close the game and any program that has its logs open, then try again.";
        if (denied) return L"Windows didn't let WrongGPU read a log or save the logs file. Close any program that has them open, then try again.";
        return L"Something went wrong. Try again in a moment.";
    }
    if (what == kCauseLauncherUninstall) {
        // a game the uninstall could not do (SelfUninstall): the launcher stays installed; that game's cause in its words
        if (raw.find(L" is running. ") != wstring::npos && raw.find(L":\\") == wstring::npos) return Sentence(raw);
        const wchar_t* const kGame = L"WrongGPU could not be uninstalled from ";
        const wchar_t* const kDetach = L"WrongGPU could not get its own uninstaller back in ";
        const bool game = raw.rfind(kGame, 0) == 0, detach = raw.rfind(kDetach, 0) == 0;
        if (game || detach) {
            const wstring name = NameBetween(raw, game ? kGame : kDetach, L". ", L"one of your games");
            const wstring head = game ? L"WrongGPU couldn't be uninstalled from " + name + L", so the launcher stays installed."
                                      : name + L" couldn't get its own entry in Installed apps, so the launcher stays installed.";
            // that game's cause alone in words (PlainCause on the whole message would say the game and the launcher again)
            const size_t lead = wcslen(game ? kGame : kDetach), dot = raw.find(L". ", lead);
            wstring inner = dot == wstring::npos ? wstring() : raw.substr(dot + 2);
            if (const size_t cut = inner.find(L" The launcher stays installed"); cut != wstring::npos) inner.resize(cut);
            if (const size_t o = inner.find(L" other game"); o != wstring::npos) {
                const size_t s = inner.rfind(L". ", o);
                if (s != wstring::npos) inner.resize(s + 1);
            }
            const wstring cause = PlainCause(inner, 0);
            // the games it did before that one ("1 other game has its original files back.")
            wstring before;
            const size_t stays = raw.find(L" The launcher stays installed"), other = raw.find(L" other game");
            if (other != wstring::npos && stays != wstring::npos && other < stays) {
                const size_t start = raw.rfind(L". ", other);
                if (start != wstring::npos) before = L" " + raw.substr(start + 2, stays - start - 2);
            }
            return head + L" " + cause + before;
        }
    }
    // the launcher's own files (its program set aside or put in place, its folder), its shortcut, its entry
    if (inUse || denied || Has(raw, L"cannot write to")) return L"Windows didn't let WrongGPU write to its own folder. Close other WrongGPU windows and try again; if it keeps happening, collect the logs and report it.";
    if (Has(raw, L"shortcut")) return wstring(L"WrongGPU couldn't add its shortcut to Start.") + kAgain;
    if (Has(raw, L"Apps & features entry")) return wstring(L"Windows didn't let WrongGPU add its entry to Installed apps.") + kAgain;
    if (Has(raw, L"verification failed")) return wstring(L"A file didn't match after it was copied.") + kAgain;
    return wstring(L"Something went wrong.") + kAgain;
}

// ---- the install's own questions and results (moved from launcher_ui.cpp, which uses them) ----
// Clean up / Adopt's question: what the files are in words; their names wait behind Show details (as on the game page,
// where they are the tooltip)
wstring LeftoversText(const wstring& name, const std::vector<Leftover>& files, bool clean) {
    std::vector<wstring> vers;
    for (auto& f : files)
        if (std::find(vers.begin(), vers.end(), f.version) == vers.end()) vers.push_back(f.version);
    wstring ver;
    for (size_t k = 0; k < vers.size(); ++k) ver += (k == 0 ? L"" : k + 1 == vers.size() ? L" and " : L", ") + vers[k];
    const wstring what = (files.size() == 1 ? wstring(L"1 file") : std::to_wstring(files.size()) + L" files") + L" from WrongGPU " + ver + L" that the launcher didn't install.";
    return clean ? L"Remove the leftover WrongGPU files from " + name + L"?\n\n" + what + L" Only these files are removed."
                 : L"Adopt the leftover WrongGPU files as an install of " + name + L"?\n\n" + what + L" Then Update, Repair and Uninstall work as for any install.";
}
std::vector<lui::DialogButton> LeftoversButtons(bool clean) {
    return {{L"Show details", 2}, {L"Cancel", 0}, {clean ? L"Clean up" : L"Adopt", 1, true, clean}};
}
// "Installed for this user" with the games that kept their own uninstall entry (an adoption that failed), if any
wstring InstalledWords(const std::vector<wstring>& kept, const wstring& lead) {
    if (kept.empty()) return lead;
    wstring names;
    for (size_t k = 0; k < kept.size(); ++k) names += (k == 0 ? L"" : k + 1 == kept.size() ? L" and " : L", ") + kept[k];
    return lead + L" " + names + (kept.size() == 1 ? L" keeps its own entry in Installed apps." : L" keep their own entries in Installed apps.");
}

// The converted data an uninstall deletes, in words ("" : none on this PC)
static wstring DataWords(bool model, bool rr) {
    return model && rr ? L"the converted Model M and Ray Reconstruction data" : model ? L"the converted Model M data" : rr ? L"the converted Ray Reconstruction data" : L"";
}
// The launcher's uninstall question (Settings, Apps & features), from what it found (LookBeforeLauncherUninstall). A game has
// WrongGPU: whether the games keep it; none: one plain question. here: asked by the installed launcher's own window, not a
// portable copy's; logs: the Library lists a game with WrongGPU, so the question offers Collect logs (answer 3). It says all
// that goes: with no game left, the shared model data goes too (SelfUninstall), unless a portable copy or a game on a drive
// that isn't connected still uses it. One name for the action: "Uninstall launcher..." -> "Uninstall the WrongGPU launcher?"
wstring UninstallLauncherText(const LauncherUninstallLook& look, bool here, bool logs) {
    const wstring stays = here ? L"" : L" This portable copy stays open.";
    const wstring title = here ? L"Uninstall the WrongGPU launcher?" : L"Uninstall the installed WrongGPU launcher?";
    const wstring data = DataWords(look.modelData, look.rrData);
    const wstring kept = look.portableUses && !data.empty() ? L" The converted model data stays: a portable copy of WrongGPU uses it." : L"";
    if (!look.anyGame)
        return title + L"\n\nIt's removed from Start and Installed apps. Its settings and logs are deleted" +
               (data.empty() || look.portableUses ? wstring(L".") : L", and so is " + data + L".") + kept + stays;
    const wstring withGames = data.empty() || look.portableUses || look.offlineGame ? wstring() : L" Uninstall from games too also deletes " + data + L".";
    return title + L"\n\nRemove WrongGPU from your games too, so each gets its original files back, or keep it in them, each with its own entry in Installed apps. "
           L"The launcher's settings and logs are deleted." + withGames + kept + (logs ? L" Reporting a problem? Collect the logs first." : L"") + stays;
}
std::vector<lui::DialogButton> UninstallLauncherButtons(bool anyGame, bool logs) {
    if (!anyGame) return {{L"Cancel", 0}, {L"Uninstall", 1, true, true}};
    if (logs) return {{L"Cancel", 0}, {L"Collect logs", 3}, {L"Keep WrongGPU in games", 2}, {L"Uninstall from games too", 1, true, true}};
    return {{L"Cancel", 0}, {L"Keep WrongGPU in games", 2}, {L"Uninstall from games too", 1, true, true}};
}
// A game's Uninstall question (PrepareOp). gone: the game isn't in its folder any more (Orphaned), so it removes WrongGPU's
// files from the old folder (its button "Remove files", never "Uninstall", which reads as uninstalling the game); data: it is
// the last game with WrongGPU, so the converted data goes too (0 none, 1 Model M, 2 Model M and Ray Reconstruction)
wstring UninstallGameText(const wstring& name, bool gone, int data) {
    const wstring last = data == 2 ? L" It's the last game with WrongGPU, so the converted Model M and Ray Reconstruction data are deleted too."
                         : data == 1 ? L" It's the last game with WrongGPU, so the converted Model M data is deleted too." : L"";
    if (gone)
        return L"Remove WrongGPU's files from " + name + L"'s old folder?\n\n" + name + L" isn't in its folder any more. WrongGPU removes the files it left there." + last;
    return L"Uninstall WrongGPU from " + name + L"?\n\nThe game's original files are put back exactly as they were." + last +
           L" Reporting a problem? Collect the logs first: uninstalling deletes WrongGPU's logs.";
}
// A portable copy's Remove data... question (Settings > Launcher install; RemoveSharedData): what goes, and why now (it has no
// uninstall, so nothing else removes the data once its folder is deleted). model / rr: the converted data that is there.
wstring RemoveDataText(bool model, bool rr) {
    const wstring what = model && rr ? L"The converted Model M and Ray Reconstruction data in your user folder are deleted, with WrongGPU's notes there."
                         : model ? L"The converted Model M data in your user folder is deleted, with WrongGPU's notes there."
                         : rr ? L"The converted Ray Reconstruction data in your user folder is deleted, with WrongGPU's notes there."
                              : L"WrongGPU's notes in your user folder are deleted.";
    return L"Remove WrongGPU's data from this PC?\n\n" + what + L" This portable copy has no uninstall, so remove the data before you delete the copy. "
           L"Its settings and logs stay next to it.";
}

// The 0.5 installs an adoption left as they were (the first run without the launcher installed)
wstring NotAdoptedWords(const std::vector<wstring>& kept) {
    wstring names;
    for (size_t k = 0; k < kept.size(); ++k) names += (k == 0 ? L"" : k + 1 == kept.size() ? L" and " : L", ") + kept[k];
    return names + (kept.size() == 1 ? L" wasn't moved to the launcher. Adopt it from its page to try again." : L" weren't moved to the launcher. Adopt them from their pages to try again.");
}
// Collect logs' result: where the file is and what is in it, in words; Show in File Explorer opens its folder with it selected.
// The place in words, never a path in the text (LIVE-FINDINGS 15): the desktop, WrongGPU's data folder, else the folder's
// name and where it is (PlaceWords).
// One title wherever they went and however many files there are ("Logs collected."); the place is said in the message.
static wstring LogsPlace(const wstring& dir, bool desktop) {
    return desktop ? L"on your desktop" : IEq(FullPath(dir), FullPath(AppDataPath())) ? wstring(L"in WrongGPU's data folder") : L"in the folder " + lui::PlaceWords(dir);
}
wstring LogsSavedText(const wstring& zip, bool desktop) {
    return L"Logs collected.\n\nThe file " + BaseName(zip) + L" is " + LogsPlace(Parent(zip), desktop) +
           L". It includes file paths (with your Windows user name), your graphics card, its driver and your Windows version. Nothing is uploaded: attach the file "
           L"to your report.";
}
// Collect logs from the launcher's uninstall question, the result: one file for each game with WrongGPU (names: those games, in
// the files' order); one file is said as Collect logs says it
wstring LogsSavedText(const std::vector<wstring>& zips, const std::vector<wstring>& names, bool desktop) {
    if (zips.size() == 1) return LogsSavedText(zips[0], desktop);
    const wstring dir = zips.empty() ? wstring() : Parent(zips[0]);
    wstring list;
    for (size_t k = 0; k < names.size(); ++k) list += (k == 0 ? L"" : k + 1 == names.size() ? L" and " : L", ") + names[k];
    return L"Logs collected.\n\nThere's one file for each game with WrongGPU " + LogsPlace(dir, desktop) + L": " + list +
           L". They include file paths (with your Windows user name), your graphics card, its driver and your Windows version. Nothing is uploaded: attach the "
           L"files to your report.";
}

// Collect logs from the launcher's uninstall question (here, not in launcher_ui.cpp: its source size): one file for each game
// with WrongGPU that the Library lists (each has the launcher's log too), on a worker ("Collecting logs..."), then where
// they are, as Collect logs says it. Uninstalling deletes these logs; opened from Installed apps, Cancel closes the window,
// so the question offers it itself.
void CollectLogsBeforeUninstall() {
    using namespace lui;  // the window's view, its toast and dialogs
    struct Game { wstring dir, name; };
    std::vector<Game> games;
    for (auto& e : v.games) if (e.hasRecord) games.push_back({e.dir, e.name});
    struct Out { std::vector<wstring> zips, names, notes; wstring why; };
    auto out = std::make_shared<Out>();
    ShowToast(L"Collecting logs\u2026", 0);
    const bool done = WaitForRead(L"", L"", T_COLLECT, [games, out] {
        SystemInfo sys; sys.win = WindowsInfo(); sys.gpu = DetectGpu();
        for (auto& gm : games) {
            if (!IsDir(gm.dir)) continue;  // a game on a drive that isn't connected: nothing to read
            try {
                // named by the game's exe and the second; two games with the same exe name never share a file
                const wstring base = DefaultBundlePath(gm.dir);
                wstring zip = base;
                for (int k = 2; IsFile(zip) || std::find(out->zips.begin(), out->zips.end(), zip) != out->zips.end(); ++k)
                    zip = base.substr(0, base.size() - 4) + L"-" + std::to_wstring(k) + L".zip";
                const Bundle b = CollectLogs(g.ctx, gm.dir, zip, sys);
                out->zips.push_back(b.zip); out->names.push_back(gm.name);
                for (auto& n : b.notes) out->notes.push_back(n);
            } catch (const SetupError& x) { if (out->why.empty()) out->why = x.msg; }
        }
    });
    v.toast = Toast();
    if (!done || !v.live) return;
    if (out->zips.empty()) {
        const wstring why = out->why.empty() ? wstring(L"no game with WrongGPU could be read") : out->why;
        InformDetails(L"The logs couldn't be collected.\n\n" + PlainTaskCause(why, kCauseLogs), kIconError, why);
        return;
    }
    PWSTR desk = nullptr;
    bool desktop = false;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Desktop, 0, nullptr, &desk)) && desk) desktop = IEq(Parent(out->zips[0]), desk);
    if (desk) CoTaskMemFree(desk);
    wstring text = LogsSavedText(out->zips, out->names, desktop);
    for (auto& n : out->notes) text += L"\n\n" + Sentence(n);
    if (Confirm(text, 2, L"Show in File Explorer", L"Done")) {  // as LogsSaved: Done, or Show in File Explorer (the default)
        const wstring args = L"/select,\"" + out->zips[0] + L"\"";
        ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
    }
}

// ---- moved from launcher_ui.cpp (the window, its input and operations), which uses them ----
// A success in words ("Quarry Nine is updated to WrongGPU 0.5.5."); `as`: the control's own words (SayResultAs, RES_*);
// fresh: the game's entry read again after it (null: not read)
wstring OpSuccess(Op op, const OpResult& res, const OpRequest& req, const GameEntry* fresh, const wstring& name, int as) {
    const wstring ver = g.ctx.data ? wstring(g.ctx.data->version) : wstring();
    switch (op) {
        case Op::Install: return L"WrongGPU is installed in " + name + L".";
        case Op::Update: return name + L" is updated to WrongGPU " + ver + L".";
        case Op::Repair: {
            // on an older WrongGPU a repair is an update too (its question said so): the core's message says it was
            const bool updated = Has(res.message, L" and updated from ");
            if (as == RES_RRREPAIR && (!fresh || fresh->rrInstalled))
                return L"Ray Reconstruction is set up in " + name + (updated ? L", which is now on WrongGPU " + ver : wstring()) + L".";
            return updated ? name + L" is repaired and updated to WrongGPU " + ver + L"." : name + L" is repaired.";
        }
        case Op::Uninstall:  // a game that is gone (RES_GONE): its button was Remove files, and there is no game to have its files back
            if (as == RES_GONE) return L"WrongGPU's files are removed from " + name + L"'s old folder.";
            return L"WrongGPU is uninstalled from " + name + L", and the game has its original files back.";
        case Op::Rollback: {
            wstring to = fresh ? fresh->installedVersion : wstring();
            if (to.empty()) {  // "Rolled back from 0.5.5 to 0.5.0 (7 files)"
                const size_t a = res.message.find(L" to "), b = res.message.find(L" (", a == wstring::npos ? 0 : a);
                if (a != wstring::npos) to = res.message.substr(a + 4, b == wstring::npos ? wstring::npos : b - a - 4);
            }
            return to.empty() ? name + L" is rolled back." : name + L" is back on WrongGPU " + to + L".";
        }
        case Op::CleanLeftovers: return L"Leftover WrongGPU files are removed from " + name + L".";
        case Op::AdoptLeftovers: case Op::Adopt:
            return name + L" is adopted. The launcher manages it now." + (Has(res.message, L"repair suggested") ? L" It needs a repair." : L"");
        case Op::Apply:  // named too: the result arrives wherever the player is (another game's page, where it would read as about that one)
            if (req.optiReset) return L"OptiScaler settings in " + name + L" are back to what WrongGPU's install wrote.";
            for (auto& n : res.notes)  // Fix it for me (ui_rrconfig.cpp): the game's Engine.ini has the line now
                if (Has(n, L": Engine.ini has the Ray Reconstruction setting")) return name + L" has the Ray Reconstruction setting now. It takes effect the next time the game starts.";
            for (auto& n : res.notes)  // ...or before the game's first start: it waits for the game's settings folder (r2b #1)
                if (Has(n, kRrConfigWaitNote))  // (in a protected folder: on the player's click then, never by itself: r2c)
                    return fresh && fresh->rrConfigProtected ? name + L"'s Ray Reconstruction setting can be added after you start the game once: then click Fix it now on its page."
                                                     : name + L"'s Ray Reconstruction setting will be added after you start the game once.";
            if (req.rrConfigOnly && req.rrConfigSet == "off")  // the question's Show me how, saved (r4 #4)
                return L"Saved: you add " + name + L"'s Engine.ini line yourself. The steps are on its page.";
            if (req.rrConfigOnly) {  // Engine.ini alone (r3 #1): what it did there, never "Settings applied" (nothing else was written)
                for (auto& n : res.notes)
                    if (Has(n, L"Engine.ini left as it is")) return name + L"'s Engine.ini is left as it is: WrongGPU doesn't manage its line now.";
                for (auto& n : res.notes)
                    if (Has(n, L": Engine.ini is read-only now")) return name + L"'s Engine.ini is read-only again, so the game keeps the Ray Reconstruction setting.";
                return name + L"'s Engine.ini has the Ray Reconstruction setting.";
            }
            if (req.optiKeep) return L"OptiScaler's own values in " + name + L" are kept.";
            if (!req.optiEdits.empty() && !req.hasProfile) return L"WrongGPU's OptiScaler values are back in " + name + L".";
            if (as == RES_KEEP) return L"The values in " + name + L"'s files are kept. WrongGPU uses them from now on.";  // nothing in the game's files changed
            if (as == RES_REVERT) return L"WrongGPU's values are back in " + name + L"'s files. They take effect the next time the game starts.";
            return L"Settings applied to " + name + L". They take effect the next time the game starts.";  // the row says sparkle guard uses HIP
        default: return Sentence(res.message);
    }
}
// Play while the page has unsaved changes: the question, and its answers (Cancel 0, Play without them 2, Apply and play 1)
// (busy: Apply waits for the work in progress, so the question offers Cancel and Play without them only)
wstring PlayUnsavedText(const wstring& name, bool busy) {
    if (busy)
        return L"Play without your changes?\n\nThe settings you changed on this page aren't saved yet, and Apply waits until " + lui::BusyDoneWords() + L". Without them " +
               name + L" starts with the settings it has now.";
    return L"Apply your changes before playing?\n\nThe settings you changed on this page aren't saved yet. Without them " + name +
           L" starts with the settings it has now.";
}
std::vector<lui::DialogButton> PlayUnsavedButtons(bool busy) {
    if (busy) return {{L"Cancel", 0}, {L"Play without them", 2, true}};
    return {{L"Cancel", 0}, {L"Play without them", 2}, {L"Apply and play", 1, true, false, true}};  // red, as Apply on the capsule (IDENTITY.md)
}
// A step in words for the page (the core's steps name files: "Installing wronggpu_runtime.dll"): what is being done, with
// the count from the step's numbers ("Copying WrongGPU's files (3 of 7)"). The log keeps the core's own words. The last
// step ("Done") and any step this list does not know give no words: the page then says the operation's verb alone
// ("Updating..."), never a core message.
wstring StepWords(const wstring& s, int done, int total) {
    struct K { const wchar_t* from; const wchar_t* to; bool counted; };
    static const K k[] = {
        {L"Looking for AMD FSR 4", L"Getting ready", false}, {L"Verifying the package", L"Checking WrongGPU's files", false},
        {L"Backing up ", L"Backing up the game's files", false}, {L"Backed up ", L"Backing up the game's files", true},
        {L"Creating the uninstaller", L"Finishing up", false}, {L"Creating ", L"Copying WrongGPU's files", true},
        {L"Moving aside ", L"Moving the game's files aside", true}, {L"Keeping the game's ", L"Keeping the game's files", true},
        {L"Installing OptiScaler", L"Installing OptiScaler", true}, {L"Installing ", L"Copying WrongGPU's files", true},
        {L"Checking the backup", L"Checking the backup", false}, {L"Restoring ", L"Restoring the game's files", true},
        {L"Cleaning up", L"Cleaning up", false}, {L"Saving the model data", L"Saving Model M", false},
        {L"Preparing the Ray Reconstruction data", L"Preparing Ray Reconstruction", false},
        {L"Preparing Ray Reconstruction kernels", L"Preparing Ray Reconstruction", true},  // rrprep.h: n of m kernels
        {L"Downloading OptiScaler", L"Downloading OptiScaler (WrongGPU edition)", false}};
    for (auto& x : k)
        if (s.rfind(x.from, 0) == 0) {
            wstring t = x.to;
            if (x.counted && total > 1 && done > 0) t += L" (" + std::to_wstring(done) + L" of " + std::to_wstring(total) + L")";
            return t;
        }
    return L"";
}

// An operation that starts, said to Narrator (StartOp; step: the page's verb, "Updating…"): "Updating WrongGPU in Quarry
// Nine…". WrongGPU is the one at work, not the game (BusyWhy's words; the result toasts' "WrongGPU is uninstalled from
// Quarry Nine.")
wstring StartSaid(const OpRequest& r, const wstring& step, bool rrSetup, const wstring& who) {
    wstring verb = step;
    while (!verb.empty() && verb.back() == L'\u2026') verb.pop_back();
    wstring words = verb == L"Working" ? L"WrongGPU is working in " : verb + L" WrongGPU in ";  // "Repairing and updating WrongGPU in"
    if (r.op == Op::Uninstall) words = verb + L" WrongGPU from ";
    else if (r.op == Op::CleanLeftovers) words = L"Cleaning up WrongGPU's files in ";
    else if (r.op == Op::Apply) words = r.optiReset ? L"Resetting OptiScaler settings in " : L"Applying settings to ";
    if (rrSetup) words = L"Setting up Ray Reconstruction in ";
    return words + who + L"\u2026";
}

// Ctrl+S on a game's page while the unsaved capsule's Apply waits (the game runs, an operation works): the reason its
// tooltip gives, as Apply's ("Apply waits until WrongGPU is installed in Quarry Nine.")
wstring ApplyWaitsWords(const GameEntry& e) {
    if (e.status == GameStatus::Running) return L"Apply waits until " + e.name + L" closes.";
    const wstring why = lui::BusyWhy(e);
    if (why.rfind(L"Waits ", 0) == 0) return L"Apply waits " + why.substr(6);
    return why.empty() ? L"Apply waits until WrongGPU finishes the current task." : why;
}

// (split from launcher_ui.cpp: source size)
bool FirstRunPortable() {
    const wstring marker = Join(Parent(SelfPath()), kPortableMarker);
    LauncherPaths portable;
    bool installedFolder = false;  // refused on purpose: the installed launcher's own folder (not Windows refusing a write)
    try {
        const string t = "WrongGPU runs portable from this folder (its data: WrongGPU-data).\r\n";
        WriteFileAtomic(marker, t.data(), t.size());
        // the paths read once the marker is there (before it, this copy's data folder is the user folder's): this session's
        // settings and logs go next to the app from now on, not only after the next start
        portable = ResolvePaths(SelfPath(), L"");
        if (!portable.portable) {
            installedFolder = true;
            Fail(L"this is the installed launcher's folder (" + Parent(SelfPath()) + L"): a portable copy needs a folder of its own");
        }
        EnsureDir(portable.dataDir);
    } catch (const SetupError& x) {
        DeleteFileW(marker.c_str());
        // in words, the core's text (the path, Windows' error) behind Show details
        if (installedFolder)
            InformDetails(L"WrongGPU can't run portable from this folder.\n\nThis is where the installed WrongGPU lives. To run it portable, copy WrongGPU.exe to a folder "
                          L"of yours (such as Documents) and start it there.", 1, x.msg);
        else
            InformDetails(L"WrongGPU can't run portable from this folder.\n\nWindows doesn't allow WrongGPU to save files here. Move WrongGPU to a folder of yours (such as "
                          L"Documents), or choose Install for this user.", lui::kIconError, x.msg);
        return false;
    }
    gl.paths = portable;
    // the first run's lines so far move next to the app with the rest of its settings and logs: from the temp folder (a
    // downloaded copy's first start), else from where the log was; a user folder this start left empty goes
    struct UserFolderGone { ~UserFolderGone() { RemoveDirectoryW(AppDataPath().c_str()); } } userFolderGone;  // only while empty
    if (!gl.firstRunLog.empty()) { SettleFirstRunLog(); return true; }
    wstring before;  // the first run's lines so far (in %LOCALAPPDATA%\WrongGPU, before the choice)
    {
        std::lock_guard<std::mutex> lock(g.ctx.logMutex);  // workers may be logging (the first run's checks)
        before = g.ctx.logFile;
        g.ctx.logFile = Join(gl.paths.dataDir, L"launcher.log");
    }
    // they move next to the app with the rest of its settings and logs (the data the games share stays in the user folder)
    string lines;
    if (!IEq(FullPath(before), FullPath(g.ctx.logFile)) && ReadAll(before, lines) && !lines.empty()) {
        HANDLE h = CreateFileW(g.ctx.logFile.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            DWORD w = 0;
            const bool ok = WriteFile(h, lines.data(), (DWORD)lines.size(), &w, nullptr) && w == lines.size();
            CloseHandle(h);
            if (ok) DeleteFileW(before.c_str());
        }
    }
    return true;
}


// (split from launcher_ui.cpp: source size)
// A newer download next to the installed launcher, once its question is answered: its window closes and the installed copy
// starts (its one-window rule brings an open window forward instead). Nothing of this copy stays open beside it.
void HandOverToInstalled(HANDLE& one, const wstring& installedExe, const wchar_t* why) {
    using namespace lui;
    if (IsWindow(v.hwnd)) DestroyWindow(v.hwnd);  // first: the installed copy must not find this window as "the open one"
    if (one) { CloseHandle(one); one = nullptr; }  // the one-window lock goes with it
    const bool started = (INT_PTR)ShellExecuteW(nullptr, L"open", installedExe.c_str(), nullptr, Parent(installedExe).c_str(), SW_SHOWNORMAL) > 32;
    LogLine(g.ctx, wstring(L"replace (") + why + L"): " + (started ? L"opened the installed copy " : L"could not open the installed copy ") + installedExe);
}
// Before and after a replace: the other launcher windows were asked to close; true while one is still open after `ms` (busy, or asking
// about its unsaved changes). This window keeps painting meanwhile.
static bool WaitOtherWindowsClosed(DWORD ms) {
    using namespace lui;
    auto other = [] {
        for (HWND w = nullptr; (w = FindWindowExW(nullptr, w, L"WrongGPULauncher", nullptr)) != nullptr;)
            if (w != v.hwnd) return true;
        return false;
    };
    const ULONGLONG until = GetTickCount64() + ms;
    while (other() && GetTickCount64() < until) {
        MsgWaitForMultipleObjects(0, nullptr, FALSE, 50, QS_ALLINPUT);
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
            if (m.message == WM_QUIT) { PostQuitMessage((int)m.wParam); return other(); }
            TranslateMessage(&m);
            DispatchMessageW(&m);
        }
    }
    return other();
}
// The newer download's one question, "Update WrongGPU to X?" (or, at the same version, "... to this build of X?"),
// answered (RunLauncherWindow; one: the one-window lock; have: the installed version; start: DecideLauncherStart). Update:
// the installed launcher is replaced and this window hands over to it. Not now (rc5): true, this copy runs for this
// session; only while another WrongGPU window is open does that one come forward instead, and the player is told.
bool AnswerReplaceOffer(HANDLE& one, const wstring& installedExe, const wstring& have, const LauncherStartDecision& start) {
    using namespace lui;
    const wstring ver = g.ctx.data->version;
    const wstring question = !start.sameVersion
        ? L"Update WrongGPU to " + ver + L"?\n\nWrongGPU " + have + L" is installed for this user. Your games are updated from their pages."
        : L"Update WrongGPU to this build of " + ver + L"?\n\n" +
              (start.installedBuilt ? L"An older build of WrongGPU " + have + L" (built " + BuildTimeText(start.installedBuilt) + L")"
                                    : L"Another build of WrongGPU " + have) +
              L" is installed for this user. Your games are updated from their pages.";
    bool update = Confirm(question, 0, L"Update", L"Not now");
    // The older window, if one is open, closes before anything is replaced (asking first about unsaved changes, as it
    // always does). One left running on its program renamed aside would start its elevated workers and HIP probes from the
    // new file (its own path still names WrongGPU.exe), putting newer files into games under its older version. Busy, or
    // with its question open, it stays: nothing is replaced, and this asks again once the player has closed it.
    for (bool asked = false; update && WaitOtherWindowsClosed(asked ? 1000 : 0);) {
        if (!asked) {
            asked = true;
            for (HWND w = nullptr; (w = FindWindowExW(nullptr, w, L"WrongGPULauncher", nullptr)) != nullptr;)
                if (w != v.hwnd) PostMessageW(w, WM_CLOSE, 0, 0);
            if (!WaitOtherWindowsClosed(3000)) break;
        }
        LogLine(g.ctx, L"replace: WrongGPU " + have + L"'s window is still open (busy, or asking about unsaved changes): nothing replaced yet");
        update = Confirm(L"WrongGPU " + have + L" is still open\n\nIt's busy, or asking about unsaved changes. Finish there and close it, then choose Update. "
                         L"Nothing is replaced while it's open.", 1, L"Update", L"Not now");
    }
    if (!update) {
        bool otherOpen = false;  // a window of the installed copy: two windows would work on the same data
        for (HWND w = nullptr; (w = FindWindowExW(nullptr, w, L"WrongGPULauncher", nullptr)) != nullptr;)
            if (w != v.hwnd) otherOpen = true;
        if (otherOpen) {
            Inform(L"WrongGPU " + have + L" is open.\n\nTo use this download instead, close that window and open this download again.", 0);
            HandOverToInstalled(one, installedExe, L"not now, its window is open");  // brings that window forward
            return false;
        }
        LogLine(g.ctx, L"replace (not now): this copy (" + ver + L", built " + BuildTimeText(start.selfBuilt) + L") runs for this session; the installed copy "
                           L"is kept: " + installedExe + L" (" + have + L", built " + BuildTimeText(start.installedBuilt) + L")");
        return true;
    } else {
        wstring why;
        const bool ok = SelfInstall(g.ctx, SelfPath(), Parent(installedExe), L"", &why);
        // the program is replaced first; only the shortcut or the entry can fail after it
        const bool replaced = ok || Has(why, L"shortcut") || Has(why, L"Apps & features entry");
        if (!ok) LogLine(g.ctx, L"replace: " + why);
        // a window of the older copy opened since (from Start, while this asked) closes too, as above
        bool otherOpen = false;
        if (replaced) {
            for (HWND w = nullptr; (w = FindWindowExW(nullptr, w, L"WrongGPULauncher", nullptr)) != nullptr;)
                if (w != v.hwnd) PostMessageW(w, WM_CLOSE, 0, 0);
            otherOpen = WaitOtherWindowsClosed(3000);
        }
        const wstring then = otherOpen ? L"The WrongGPU window that was open is still busy or has unsaved changes. Close it, then open WrongGPU from Start."
                                       : L"WrongGPU opens now. Next time, open it from Start as before.";
        if (ok) Inform(L"WrongGPU is updated to " + ver + L".\n\n" + then + L" This downloaded file can be deleted.", 2);
        else if (replaced)
            InformDetails(L"WrongGPU is updated to " + ver + L", but its shortcut in Start or its entry in Installed apps couldn't be written.\n\n" + then +
                              L" Open this download again to try once more.", 1, why);
        else
            InformDetails(L"WrongGPU couldn't be updated.\n\nAnother WrongGPU window may still be open. Close it, then open this download again. Nothing was changed.", kIconError, why);
        // this download never stays open as a launcher of its own: the updated copy opens in its place (not while the older
        // window is still open: the start would only bring that one forward)
        if (replaced && !otherOpen) HandOverToInstalled(one, installedExe, L"updated");
        else if (IsWindow(v.hwnd)) DestroyWindow(v.hwnd);
    }
    return false;
}

// Settings > Launcher install > Remove data... (launcher_ui.cpp, split from it: source size)
void RemoveDataAsked() {
    using namespace lui;
    // a portable copy has no uninstall: the shared data it used (the converted data, the notes) goes by hand before its
    // folder is deleted, once no game and no installed launcher use it (RemoveSharedData). What is there is read on a
    // worker first, as the launcher's uninstall reads it (a game on an offline share holds each look, never the window)
    if (!v.live || v.busy || !gl.paths.portable || LauncherInstalled()) return;
    auto look = std::make_shared<SharedDataLook>();
    if (!WaitForRead(L"", kLookVerb, T_NONE, [look] { *look = LookAtSharedData(g.ctx); })) return;
    if (!look->there) { Inform(L"There's no WrongGPU data to remove\n\nNothing of WrongGPU's is in your user folder. You can delete this portable copy once you close it.", 2); return; }
    if (look->installed) { Inform(L"The installed WrongGPU uses this data\n\nIts own uninstall removes it: Settings in the installed WrongGPU, or Installed apps.", 2); return; }
    if (look->anyGame) { Inform(L"Your games still use WrongGPU's data\n\nUninstall WrongGPU from each game first, then remove the data.", 1); return; }
    if (!Confirm(RemoveDataText(look->modelData, look->rrData), 0, L"Remove data", L"Cancel", true)) return;
    auto why = std::make_shared<wstring>();
    auto kept = std::make_shared<wstring>();
    auto ok = std::make_shared<bool>(false);
    if (!WaitForRead(L"", L"Removing…", T_NONE, [why, kept, ok] { *ok = RemoveSharedData(g.ctx, why.get(), kept.get()); })) return;
    if (!*ok) {
        InformDetails(L"WrongGPU's data couldn't be removed.\n\nA file in it may be in use. Close what uses it, then try again. Nothing was removed.", kIconError, *why);
        return;
    }
    // the ignore list stays (convert_ignore.h): said once in the toast; kept elsewhere than in place (another program's new
    // list was there, or it could not be put back): where it is, in full
    if (!kept->empty() && !StartsWithI(*kept, kKeptIgnoreListText))
        Inform(L"WrongGPU's data is removed, but your ignore list was not put back as it was\n\n" + *kept, 1);
    else
        ShowToast(kept->empty() ? L"WrongGPU's data is removed. You can delete this portable copy once you close it."
                                : L"WrongGPU's data is removed; kept ignore-files.txt (your/integrators' ignore list). You can delete this portable copy once you close it.", 1);
    StartHealth(false);  // the model and RR data are gone: the status line and Settings follow
    Invalidate();
}

}  // namespace wg::app::win
