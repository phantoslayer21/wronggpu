// SPDX-License-Identifier: MIT
// wronggpu-convert: the install-time model converter of spec 08 (command line of section 10; map versions 1 and 4),
// spec 48 (preset K; map versions 2 and 3) and spec 57 (preset F, Ray Reconstruction; map version 5).
// Written for this project.
//
//   wronggpu-convert --map <map.json> --dll <file> [--dll <file> ...] --out <dir> [--raw] [--threads N]
//
// Several files (spec 08 section 8, spec 57 section 1): they are tried in the order given (the player's choice
// first). The first that converts completely is used; for a map with an optional group (`filter`, map version 4),
// the first that converts completely with it, else the first that converts completely. Every rejection is reported.
// Exit code 0 on success. On failure a non-zero code (see README.md) and the player message on standard output; the
// log goes to standard error. The log never contains decoded values or source bytes. A successful conversion
// without the optional `filter` group adds one line for the player after the success message.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "api.h"

namespace fs = std::filesystem;
using namespace wgc;

namespace {

void log_line(const std::string& line) { std::cerr << "[wronggpu-convert] " << line << std::endl; }

void usage() {
    std::cout << "usage: wronggpu-convert --map <map.json> --dll <model dll> [--dll <other> ...]\n"
                 "                        --out <dir> [--raw] [--threads N] [--preset M|K|F|L]\n"
                 "  --dll may be given several times; the files are tried in order: the first that converts\n"
                 "  completely is used (for a map with an optional group, such as preset M's output filter: the\n"
                 "  first that has the group, if any has it). The map names its preset (field \"preset\", or by its\n"
                 "  version: 1 and 4 M, 2 and 3 K, 5 F). --preset names the preset the map is for: a map without the\n"
                 "  field is read as that preset; a map with it must agree.\n"
                 "  --raw also writes <dir>/raw/<group>/<name>.bin for every tensor.\n";
}

int run(const std::vector<fs::path>& args) {
    fs::path map_path, out_dir;
    std::vector<fs::path> dlls;
    bool raw = false;
    std::string preset;  // --preset: the preset the map is loaded for ("" : the map's field, or its version's preset)
    SearchOptions opt;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string a = args[i].u8string();
        auto value = [&](fs::path& dst) {
            if (i + 1 >= args.size()) return false;
            dst = args[++i];
            return true;
        };
        if (a == "--map") {
            if (!value(map_path)) return usage(), kUsage;
        } else if (a == "--out") {
            if (!value(out_dir)) return usage(), kUsage;
        } else if (a == "--dll") {
            fs::path p;
            if (!value(p)) return usage(), kUsage;
            dlls.push_back(p);
        } else if (a == "--preset") {
            fs::path p;
            if (!value(p)) return usage(), kUsage;
            preset = p.u8string();
            if (!find_preset(preset)) {
                std::cout << "unknown preset: " << preset << "\n";
                usage();
                return kUsage;
            }
        } else if (a == "--raw") {
            raw = true;
        } else if (a == "--threads") {
            fs::path p;
            if (!value(p)) return usage(), kUsage;
            try {
                const int n = std::stoi(p.u8string());
                if (n < 1) return usage(), kUsage;
                opt.threads = static_cast<unsigned>(n);
            } catch (...) {
                return usage(), kUsage;
            }
        } else if (a == "--help" || a == "-h") {
            usage();
            return kOk;
        } else {
            std::cout << "unknown argument: " << a << "\n";
            usage();
            return kUsage;
        }
    }
    if (map_path.empty() || out_dir.empty() || dlls.empty()) {
        usage();
        return kUsage;
    }

    // The map is checked completely before any DLL is read or any output is written.
    std::string map_bytes;
    {
        std::error_code ec;
        std::ifstream f(map_path, std::ios::binary);
        if (!f || !fs::is_regular_file(map_path, ec)) {
            log_line("cannot read the map " + map_path.u8string());
            std::cout << kMsgMapDamaged << std::endl;
            return kMapDamaged;
        }
        map_bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    LoadedMap lm;
    log_line("map " + map_path.u8string());
    const Result mr = load_map_bytes(map_bytes, lm, log_line, preset);
    if (mr.code != kOk) {
        std::cout << mr.message << std::endl;
        return mr.code;
    }
    const Map& map = *lm.map;

    std::vector<std::pair<fs::path, Result>> rejected;
    Converted chosen;
    Result chosen_result;
    bool have = false, stopped = false;
    std::size_t without = 0;  // files that convert, but without an optional group
    for (const fs::path& dll : dlls) {
        Converted c;
        Result r = convert_file(map, dll, opt, c, log_line);
        if (r.code == kOk) {
            const bool every_group = has_every_group(map, c);
            if (!every_group) ++without;
            if (every_group || !have) {
                chosen = std::move(c);
                chosen_result = r;
                have = true;
            }
            if (every_group) break;
            if (dlls.size() > 1)
                log_line(dll.u8string() + " converts without an optional group; " +
                         (without == 1 ? "trying the remaining files for one that has it" : "an earlier file stays the choice"));
            continue;
        }
        log_line("rejected " + dll.u8string() + " (code " + std::to_string(r.code) + ")");
        rejected.emplace_back(dll, r);
        if (is_installer_error(r.code)) {  // the next file would fail the same way; nothing is kept
            stopped = true;
            break;
        }
    }
    if (have && !stopped) {
        if (dlls.size() > 1)
            log_line("using " + chosen.dll.u8string() + (chosen.notes.empty() ? "" : " (no file has every optional group)"));
        const Result w = write_converted(map, chosen, out_dir, raw, log_line);
        if (w.code == kOk) {
            log_line("converted from " + chosen.dll.u8string());
            for (const auto& rj : rejected) std::cout << rj.first.u8string() << ": " << rj.second.message << "\n";
            std::cout << "The model was converted successfully.\n";
            for (const std::string& note : chosen.notes) std::cout << note << "\n";
            std::cout.flush();
            return kOk;
        }
        log_line("rejected " + chosen.dll.u8string() + " (code " + std::to_string(w.code) + ")");
        rejected.emplace_back(chosen.dll, w);
    }
    if (rejected.size() == 1) {
        std::cout << rejected[0].second.message << std::endl;
    } else {
        for (const auto& rj : rejected) std::cout << rj.first.u8string() << ": " << rj.second.message << "\n";
        std::cout.flush();
    }
    // An installer error decides the exit code; otherwise the first file's (the player's choice).
    return is_installer_error(rejected.back().second.code) ? rejected.back().second.code : rejected[0].second.code;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<fs::path> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return run(args);
}
#else
int main(int argc, char** argv) {
    std::vector<fs::path> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return run(args);
}
#endif
