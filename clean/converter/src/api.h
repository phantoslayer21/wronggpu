// SPDX-License-Identifier: MIT
// Library interface of the converter (spec 08), used by the installer (spec 30) and by the
// command-line tool. Written for this project.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "convert.h"
#include "locate.h"
#include "map.h"
#include "sha256.h"

namespace wgc {

// Result codes; also the exit codes of wronggpu-convert (README.md).
enum Code {
    kOk = 0,
    kUsage = 1,
    kOutputFailed = 2,
    kDllUnreadable = 10,
    kNotModelFile = 11,
    kNoPresetM = 12,
    kUnsupportedBuild = 13,
    // Map version 5 (spec 57 section 8): a value check failed in a file whose sources were all found (sites that
    // disagree, a forbidden non-finite value, an RN16 overflow, `also` reads that differ, a failed `require` or
    // check, rejected attention constants). Same player message as kUnsupportedBuild; not an installer error.
    kValueRejected = 14,
    kMapDamaged = 20,
    kMapDoesNotFit = 21,
    kBadValue = 22,
    kSelfCheckFailed = 23,
};

// Player messages of spec 08 section 8.
extern const char* const kMsgUnreadable;
extern const char* const kMsgNotModel;
extern const char* const kMsgNoPresetM;
extern const char* const kMsgNoPresetK;
extern const char* const kMsgNoPresetF;
extern const char* const kMsgNoPresetL;
extern const char* const kMsgNoPresetFG;
extern const char* const kMsgUnsupported;
extern const char* const kMsgMapDamaged;
extern const char* const kMsgMapDoesNotFit;
extern const char* const kMsgBadValue;
extern const char* const kMsgSelfCheck;
// Not an error: no variant of the optional source F was found; the output has no `filter` group (section 8).
extern const char* const kMsgFilterUnknown;

// Installer-class failures: the next file would fail the same way (spec 08 section 8, "Several DLLs").
inline bool is_installer_error(int code) { return code >= kMapDamaged || code == kOutputFailed; }

using LogSink = std::function<void(const std::string&)>;

struct Result {
    int code = kOk;
    std::string message;  // for the player
};

// A validated map. Converted stages point into it, so it is kept behind a pointer.
struct LoadedMap {
    std::unique_ptr<Map> map;
};

// Parses and validates a map (no DLL is read). On failure the result carries code 20 or 21. `expected_preset`: the
// preset the caller loads the map for ("" : the map's own field, or its version's preset; map.h load_map).
Result load_map_bytes(const std::string& bytes, LoadedMap& out, const LogSink& log, const std::string& expected_preset = "");

// One converted file, held in memory (nothing is written).
struct Converted {
    std::filesystem::path dll;
    std::string version;  // file version for display ("" if none)
    Digest key{};
    std::vector<StageOut> stages;  // without the groups of a source with variants of which none was found
    // Per map source: for a source with variants (F), the variant that was found (0-based) or -1; -1 otherwise.
    std::vector<int> variants;
    // One line for the player per thing that is not an error but worth saying (kMsgFilterUnknown).
    std::vector<std::string> notes;
    // Map version 5 (spec 57 section 6.4): the attention-sum record (false when the map has no attn_wmax).
    bool attention_sum_may_overflow = false;
};

// Spec 08 sections 4-7 for one file: locate, decode, check every digest. On success fills `out`; the result's
// message then also carries the notes.
Result convert_file(const Map& map, const std::filesystem::path& dll, const SearchOptions& opt, Converted& out,
                    const LogSink& log);

// True if no optional group of the map was left out of `c` (no optional source with variants went unfound; a
// version-5 group left out because its `requires_variant` is not met is not missing). Spec 08 section 8,
// "Several DLLs": the first file that converts completely with every group is used; if none has them all, the
// first that converts completely. The caller walks its list of files with this (the installer and the tool do).
bool has_every_group(const Map& map, const Converted& c);

// Section 8: writes <out_dir>/model.wgcm (and raw tensors) through a staging folder.
Result write_converted(const Map& map, const Converted& c, const std::filesystem::path& out_dir, bool raw,
                       const LogSink& log);

// What verify_model_file found out about a valid file beyond "it matches the map".
struct ModelFileInfo {
    int container_version = 0;  // 1 (map versions 1-4) or 2 (map version 5)
    // Groups of the map that the file does not have. Only a group read from a source with variants (`filter`) can
    // be absent from a valid file: written before the group existed, or made from a DLL without a known variant.
    std::vector<std::string> absent_groups;
    Digest map_sha256{};     // the header's hash of the map file the converter used
    bool same_map = false;   // that hash equals this map's (false for a file written with an earlier map file)
};

// True if `file` is a complete converter output whose stages, tensors and digests match the map
// (the installer reuses such a file instead of converting again). `why` explains a mismatch.
// A group read from a source with variants (`filter`) may be absent from the file, and when present it must
// carry the digest of one of its variants. The header's map hash is not compared (as before): a file written with
// the version-1 preset-M map is valid for the version-4 map, without the group. `info`, if given, tells the caller
// both things, so that it can decide to convert again where that could add the group.
//
// Map version 5 (spec 57 section 7.3, "Reading back"): every group of the map must be present except one whose
// `requires_variant` names a variant (`constants_b2`), which may be absent; every group present must carry the
// map's digest and its data must give that digest with the stored key. `info->absent_groups` then says whether
// `constants_b2` is there (absent: build variant B1; present: B2).
bool verify_model_file(const Map& map, const std::filesystem::path& file, std::string& why,
                       ModelFileInfo* info = nullptr);

}  // namespace wgc
