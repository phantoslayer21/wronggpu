// SPDX-License-Identifier: MIT
// The front end's side of WrongGPU's shared settings (settings_bridge.h). This translation unit holds the Win32 part of
// clean/wgshared/include/wg_settings_io.h for every DLL that links wg_fsr_win.
#include "log_win.h"

#define WG_SETTINGS_IO_IMPLEMENTATION
#include "wg_settings_io.h"

#include <mutex>

#include "settings_bridge.h"

namespace wgwin {

namespace {

wronggpu::Log& L() { return wronggpu::Log::Get(); }

std::string N(const std::wstring& w) { return wg::narrow(w); }

std::wstring DirOf(const std::wstring& path) {
    const size_t s = path.find_last_of(L"\\/");
    return s == std::wstring::npos ? std::wstring(L".") : path.substr(0, s);
}

struct State {
    std::once_flag once;
    wg::SharedSettings block;
    wg::IniValues ini;          // wronggpu.ini as read at start (the environment / ini levels of the precedence)
    wg::HostIniPoll poll;
    std::mutex pollMutex;
    uint64_t folderEntry = 0;
    bool hosted = false;
    bool initialised = false;
};

State& S() {
    static State* s = new State;  // never destroyed (the poll timer may run until the process ends)
    return *s;
}

}  // namespace

const SettingsPlace& Place() {
    static const SettingsPlace place = [] {
        SettingsPlace p;
        p.modulePath = wg::module_path(reinterpret_cast<const void*>(&Place));
        const std::wstring moduleDir = DirOf(p.modulePath);
        p.folder = wg::custom_slot_folder(p.modulePath, &p.hostIni);
        p.customSlot = !p.folder.empty();
        p.sidecar = moduleDir + L"\\wronggpu-entry.ini";
        p.dataDir = wg::data_dir();
        const std::wstring dataIni = p.dataDir + L"\\wronggpu.ini";
        // The old place: next to this DLL (the game folder of the direct and older hosted installs); in the custom slot
        // the DLL sits in OptiScaler's folder, so the old place is the game folder (next to the exe).
        const std::wstring oldDir = p.customSlot ? wg::exe_dir() : moduleDir;
        const std::wstring oldIni = oldDir + L"\\wronggpu.ini";
        // The launcher's layout: WrongGPU's runtime (Model M and RR in one DLL, the RR plugin loads it from there too) and
        // the front end's wronggpu.ini in the old place (next to the exe; the front end in OptiScaler's custom slot, or
        // next to them as wronggpu_ffx_dx12.dll with OptiScaler before 10). That place then, so both DLLs share one
        // runtime and the launcher's settings reach the front end; <exe>\WrongGPU\wronggpu.ini is then the launcher's
        // Ray Reconstruction settings file, not the front end's.
        if (wg::file_exists(oldIni) && wg::file_exists(oldDir + L"\\wronggpu_runtime.dll")) {
            p.ini = oldIni;
            p.oldPlace = true;
            p.note = "settings: wronggpu.ini and wronggpu_runtime.dll in " + N(oldDir) + " (the launcher's layout)";
        } else if (wg::file_exists(dataIni)) {
            p.ini = dataIni;
        } else if (wg::file_exists(oldIni)) {
            p.ini = oldIni;
            p.oldPlace = true;
            p.note = "settings: wronggpu.ini from the old place " + N(oldIni) + " (WrongGPU's data folder " + N(p.dataDir) +
                     " has none; the old place keeps working)";
        } else if (wg::directory_exists(p.dataDir) || (p.customSlot && CreateDirectoryW(p.dataDir.c_str(), nullptr))) {
            p.ini = dataIni;  // written with the defaults by ReadSettings
        } else {
            p.ini = oldIni;   // no data folder (direct install as before): next to the DLL
            p.oldPlace = true;
        }
        const std::wstring iniDir = DirOf(p.ini);
        p.runtimeDir = iniDir;
        if (iniDir != moduleDir && !wg::file_exists(iniDir + L"\\wronggpu_runtime.dll") && wg::file_exists(moduleDir + L"\\wronggpu_runtime.dll"))
            p.runtimeDir = moduleDir;  // (only the default name is probed; runtime_dll is resolved against runtimeDir)
        return p;
    }();
    return place;
}

int32_t Setting(uint32_t slot, std::string* described, bool* isDefault) {
    State& s = S();
    wg::Value v;
    if (s.block.present()) {
        wg::Snapshot snap;
        s.block.snapshot(&snap, wg::reader_mask(WG_READER_SR_FRONTEND | WG_READER_SR_RUNTIME));
        v = wg::resolve(slot, &snap, &s.ini);
    } else {
        v = wg::resolve_unblocked(slot, &s.ini);
    }
    if (described) *described = wg::describe(slot, v.value, v.source);
    if (isDefault) *isDefault = v.source == wg::Source::Default;
    return v.value;
}

static bool WriteChoice(uint32_t slot, int32_t value, const char* source) {
    State& s = S();
    if (!s.block.present() || !s.block.is_writer()) return false;
    wg::Snapshot snap;
    s.block.snapshot(&snap);
    if (snap.is_set(slot) && snap.values[slot] == value) return true;  // unchanged: nothing rewritten
    const bool ok = s.block.write_slot(slot, value, /*persist=*/true);
    L().WriteLine(std::string("settings: ") + wg::describe(slot, value, std::strcmp(source, "optiscaler.ini") == 0 ? wg::Source::HostIni
                                                                       : std::strcmp(source, "folder") == 0 ? wg::Source::Sidecar
                                                                                                            : wg::Source::Menu) +
                  (ok ? " written to the shared settings and " + N(s.block.ini()) : std::string(" NOT written (value out of range)")));
    return ok;
}

void InitSettings(bool hosted) {
    State& s = S();
    std::call_once(s.once, [&s, hosted] {
        const SettingsPlace& p = Place();
        s.hosted = hosted;
        std::string text;
        if (wg::read_file(p.ini, &text)) s.ini = wg::parse_ini(text);
        std::string log;
        s.block.open(WG_SS_WRITER_SR_FRONTEND, p.ini, &log);
        // One folder per choice: the sidecar next to this DLL names the entry this folder stands for.
        std::string sidecar;
        if (wg::read_file(p.sidecar, &sidecar)) {
            uint64_t id = 0;
            if (wg::parse_sidecar_entry(sidecar, &id) && wgfsr::OwnEntryOfId(id)) {
                s.folderEntry = id;
                log += "settings: one folder per choice: " + N(p.sidecar) + " names " +
                       wgfsr::EntryProviderName(*wgfsr::OwnEntryOfId(id)) + "\n";
            } else {
                log += "settings: " + N(p.sidecar) + " names no WrongGPU entry of this build; ignored\n";
            }
        }
        if (hosted && s.block.present()) {
            const uint32_t flags = WG_SS_FLAG_HOSTED_BY_OPTISCALER | (s.folderEntry ? WG_SS_FLAG_FOLDER_ENTRY : 0u);
            // Several folders of the fallback are several instances of this DLL: they take turns as the same writer.
            if (s.block.claim_writer(WG_SS_WRITER_SR_FRONTEND, flags, /*shareSameId=*/s.folderEntry != 0))
                log += "settings: this SR front end writes the shared settings (the host's menu choices)\n";
            else
                log += "settings: another module (writer " + std::to_string(s.block.writer()) + ") writes the shared settings; this one only reads\n";
        }
        if (p.customSlot) {
            log += "custom slot: loaded from OptiScaler's custom folder \"" + N(p.folder) + "\"; data folder " + N(p.dataDir) +
                   "; the host's ini " + N(p.hostIni) + " [" + N(p.folder) + "] UpscalerIndex is polled\n";
            s.poll.configure(p.hostIni, N(p.folder), 1000);
        }
        if (!p.note.empty()) log += p.note + "\n";
        // Every option the front end reads, in the registry's words.
        std::string read;
        for (uint32_t i = 0; i < WG_OPT_COUNT; ++i) {
            if (!(kWgOptions[i].readers & WG_READER_SR_FRONTEND)) continue;
            std::string d;
            Setting(i, &d);
            read += (read.empty() ? "" : ", ") + d;
        }
        log += "settings: " + read + "\n";
        if (s.folderEntry && Setting(WG_OPT_ENTRY_SOURCE) != WG_ENTRY_SOURCE_FOLDERS)
            log += "settings: note: a folder sidecar is present, so this folder offers its one entry (entry_source in the ini says versions)\n";
        size_t a = 0;
        while (a < log.size()) {
            size_t b = log.find('\n', a);
            if (b == std::string::npos) b = log.size();
            if (b > a) L().WriteLine(log.substr(a, b - a));
            a = b + 1;
        }
        s.initialised = true;
    });
}

wgfsr::MenuSettings Menu() {
    wgfsr::MenuSettings m;
    m.get = [](uint32_t slot, std::string* described, bool* isDefault) { return Setting(slot, described, isDefault); };
    m.set = [](uint32_t slot, int32_t value, const char* source) { return WriteChoice(slot, value, source); };
    m.folderEntry = S().folderEntry;
    return m;
}

bool HostIniPollActive() {
    State& s = S();
    return s.initialised && s.poll.configured() && s.block.is_writer();
}

bool PollHostIni() {
    State& s = S();
    if (!HostIniPollActive()) return false;
    std::lock_guard lock(s.pollMutex);
    int index = -1;
    if (!s.poll.poll(&index)) return true;
    const wgfsr::MenuSettings m = Menu();
    const WgEntryDef* e = s.folderEntry ? (index == 0 ? wgfsr::OwnEntryOfId(s.folderEntry) : nullptr) : wgfsr::OwnEntry(index);
    if (!e) {
        L().WriteLine("settings: " + N(s.poll.path()) + " [" + s.poll.section() + "] UpscalerIndex=" + std::to_string(index) +
                      " is not one of WrongGPU's entries (one of AMD's versions); nothing to apply");
        return true;
    }
    L().WriteLine("settings: " + N(s.poll.path()) + " [" + s.poll.section() + "] UpscalerIndex=" + std::to_string(index) + " = " +
                  wgfsr::EntryProviderName(*e));
    WriteChoice(e->slot, e->value, "optiscaler.ini");
    return true;
}

bool IsSettingsWriter() { return S().block.is_writer(); }

}  // namespace wgwin
