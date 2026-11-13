// SPDX-License-Identifier: MIT
// Library interface of the converter (spec 08). Written for this project.
#include "api.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <new>
#include <system_error>

#include "output.h"
#include "pe_version.h"

namespace wgc {

namespace fs = std::filesystem;

const char* const kMsgUnreadable = "This file cannot be read.";
const char* const kMsgNotModel = "This is not a DLSS model file.";
const char* const kMsgNoPresetM = "This DLSS file has no preset-M model. Version 310.5 or newer is needed.";
// Spec 48 section 8.
const char* const kMsgNoPresetK = "This DLSS file has no preset-K model. Version 310.1 or newer is needed.";
// Spec 57 section 8.
const char* const kMsgNoPresetF =
    "This DLSS file has no preset-F (Ray Reconstruction) model. Version 310.7 or newer is needed.";
// Model L (#89). The DLSS version named is provisional: the first version known to carry a preset-L model; one line to
// change when the first shipping version is confirmed.
const char* const kMsgNoPresetL = "This DLSS file has no preset-L model. Version 310.9 or newer is needed.";
const char* const kMsgUnsupported = "This DLSS build is not supported yet (model data found, but other parts differ).";
const char* const kMsgMapDamaged = "Installer error: the converter data is damaged. Please reinstall WrongGPU.";
const char* const kMsgMapDoesNotFit =
    "Installer error: the converter data does not fit this model (converter data damaged). Please reinstall WrongGPU.";
const char* const kMsgBadValue =
    "Installer error: a converted value is not a valid number (converter data damaged). Please reinstall WrongGPU.";
const char* const kMsgSelfCheck =
    "Conversion self-check failed. The installer may be damaged; please reinstall WrongGPU.";
const char* const kMsgFilterUnknown =
    "The output filter of this DLSS build is not known to this version; WrongGPU's own filter will be used.";

namespace {

// Above this size a file is not considered a model file (spec: "up to a few hundred MB").
const std::uint64_t kMaxDllSize = std::uint64_t(4) << 30;

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string hex64(std::uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "0x%llx", static_cast<unsigned long long>(v));
    return buf;
}

void emit(const LogSink& log, const std::string& line) {
    if (log) log(line);
}

Result fail(int code, const std::string& message) { return Result{code, message}; }

bool read_file(const fs::path& p, std::vector<std::uint8_t>& data, Result& r, const LogSink& log) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) {
        emit(log, "not a readable regular file: " + p.u8string());
        r = fail(kDllUnreadable, kMsgUnreadable);
        return false;
    }
    const std::uintmax_t size = fs::file_size(p, ec);
    if (ec) {
        emit(log, "cannot get the size of " + p.u8string() + ": " + ec.message());
        r = fail(kDllUnreadable, kMsgUnreadable);
        return false;
    }
    if (size > kMaxDllSize) {
        emit(log, "file too large: " + std::to_string(size) + " bytes");
        r = fail(kNotModelFile, kMsgNotModel);
        return false;
    }
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        emit(log, "cannot open " + p.u8string());
        r = fail(kDllUnreadable, kMsgUnreadable);
        return false;
    }
    try {
        data.resize(static_cast<std::size_t>(size));
    } catch (const std::bad_alloc&) {
        emit(log, "not enough memory to read " + std::to_string(size) + " bytes");
        r = fail(kDllUnreadable, kMsgUnreadable);
        return false;
    }
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (static_cast<std::uintmax_t>(f.gcount()) != size) {
        emit(log, "read error in " + p.u8string());
        r = fail(kDllUnreadable, kMsgUnreadable);
        return false;
    }
    return true;
}

std::uint32_t rd32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
std::uint64_t rd64(const std::uint8_t* p) { return std::uint64_t(rd32(p)) | (std::uint64_t(rd32(p + 4)) << 32); }

// For the log: the output groups that are left out when no variant of this source is found.
std::string absent_groups(const Map& map, std::size_t source) {
    std::string names;
    for (const Stage& st : map.stages)
        if (st.variant_source == static_cast<int>(source)) names += (names.empty() ? "" : ", ") + st.name;
    return names.empty() ? std::string("no output group reads it") : "the output will have no group " + names;
}

std::string found_groups(const std::vector<StageOut>& stages) {
    std::string s;
    for (const StageOut& so : stages)
        if (so.variant >= 0) s += ", " + so.stage->name + ": variant " + std::to_string(so.variant + 1);
    return s;
}

// Spec 57 sections 4, 6-8 after W and the key: every other source by the shared block search (each must be found;
// a source with variants is found as its first variant that matches), then the groups and their digests.
Result convert_rest_v5(const Map& map, const std::vector<std::uint8_t>& data, const SearchOptions& opt, Converted& out,
                       std::vector<std::vector<std::uint8_t>>& src_bytes, std::chrono::steady_clock::time_point t0,
                       const LogSink& log) {
    std::vector<BlockTarget> targets;
    std::vector<std::pair<std::size_t, int>> owner;  // per target: source index, variant (-1 without variants)
    for (std::size_t i = 0; i < map.sources.size(); ++i) {
        if (static_cast<int>(i) == map.key_source) continue;
        const Source& s = map.sources[i];
        if (s.has_variants()) {
            for (std::size_t v = 0; v < s.variants.size(); ++v) {
                targets.push_back(BlockTarget{s.variants[v].length, s.variants[v].fnv, s.variants[v].sha, s.keyed});
                owner.emplace_back(i, static_cast<int>(v));
            }
        } else {
            targets.push_back(BlockTarget{s.length, s.fnv, s.sha, s.keyed});
            owner.emplace_back(i, -1);
        }
    }
    std::vector<Found> found = locate_blocks(data.data(), data.size(), targets, &out.key, opt);
    out.variants.assign(map.sources.size(), -1);
    std::vector<std::uint64_t> offset(map.sources.size(), 0);
    std::vector<bool> have(map.sources.size(), false);
    have[static_cast<std::size_t>(map.key_source)] = true;
    for (std::size_t k = 0; k < targets.size(); ++k) {  // targets of one source are in variant order
        const std::size_t i = owner[k].first;
        if (!found[k].found || have[i]) continue;
        have[i] = true;
        out.variants[i] = owner[k].second;
        offset[i] = found[k].offset;
        src_bytes[i] = std::move(found[k].bytes);
    }
    std::string missing;
    for (std::size_t i = 0; i < map.sources.size(); ++i) {
        if (static_cast<int>(i) == map.key_source) continue;
        const Source& s = map.sources[i];
        if (!have[i]) {
            emit(log, "source " + s.name + " not found" +
                          (s.has_variants() ? " (none of its " + std::to_string(s.variants.size()) + " variants)" : ""));
            missing += (missing.empty() ? "" : ", ") + s.name;
            continue;
        }
        emit(log, "source " + s.name + " at offset " + std::to_string(offset[i]) +
                      (s.has_variants() ? " (variant " + std::to_string(out.variants[i]) + " of " +
                                              std::to_string(s.variants.size()) + ")"
                                        : ""));
    }
    if (!missing.empty()) {
        emit(log, "missing sources: " + missing);
        return fail(kUnsupportedBuild, kMsgUnsupported);
    }
    for (std::size_t i = 0; i < map.sources.size(); ++i)
        if (map.sources[i].has_variants())
            emit(log, "variant used: " + map.sources[i].name + " variant " + std::to_string(out.variants[i]));
    emit(log, "sources located in " + std::to_string(seconds_since(t0)) + " s");

    DecodeError de;
    Convert5Info info;
    if (!convert_all_v5(map, src_bytes, out.variants, out.key, out.stages, de, &info)) {
        emit(log, de.detail);
        out.stages.clear();
        switch (de.kind) {
            case DecodeErrorKind::ValueRejected: return fail(kValueRejected, kMsgUnsupported);
            case DecodeErrorKind::DigestMismatch: return fail(kSelfCheckFailed, kMsgSelfCheck);
            default: return fail(kMapDamaged, kMsgMapDamaged);
        }
    }
    out.attention_sum_may_overflow = info.attention_sum_may_overflow;
    if (info.have_wmax)
        emit(log, std::string("attention_sum_may_overflow: ") + (info.attention_sum_may_overflow ? "true" : "false"));
    std::string absent;
    for (const Stage& st : map.stages)
        if (!group_enabled(st, out.variants)) absent += (absent.empty() ? "" : ", ") + st.name;
    emit(log, "all " + std::to_string(out.stages.size()) + " group digests match" +
                  (absent.empty() ? std::string() : " (not produced for this variant: " + absent + ")") + " (" +
                  std::to_string(seconds_since(t0)) + " s)");
    return Result{kOk, "The model was converted successfully."};
}

}  // namespace

Result load_map_bytes(const std::string& bytes, LoadedMap& out, const LogSink& log) {
    out.map = std::make_unique<Map>();
    MapError me;
    if (!load_map(bytes, *out.map, me)) {
        emit(log, "map: " + me.detail);
        out.map.reset();
        const bool range = me.kind == MapErrorKind::IndexRange;
        return fail(range ? kMapDoesNotFit : kMapDamaged, range ? kMsgMapDoesNotFit : kMsgMapDamaged);
    }
    emit(log, "map sha256 " + to_hex(out.map->file_sha256) + ", " + std::to_string(out.map->stages.size()) +
                  " stages, model: " + out.map->model);
    return Result{};
}

Result convert_file(const Map& map, const fs::path& dll, const SearchOptions& opt, Converted& out, const LogSink& log) {
    Result r;
    out = Converted();
    out.dll = dll;
    const auto t0 = std::chrono::steady_clock::now();
    emit(log, "file: " + dll.u8string());
    std::vector<std::uint8_t> data;
    if (!read_file(dll, data, r, log)) return r;
    emit(log, "size: " + std::to_string(data.size()) + " bytes");
    const Source& ks = map.sources[static_cast<std::size_t>(map.key_source)];
    if (data.size() < ks.length) {
        emit(log, "smaller than source " + ks.name + " (" + std::to_string(ks.length) + " bytes)");
        return fail(kNotModelFile, kMsgNotModel);
    }
    out.version = read_file_version(data.data(), data.size());
    emit(log, "file version: " + (out.version.empty() ? std::string("(none found)") : out.version));

    // Section 4.2: the key source (W) first, then the key, then the other sources.
    std::vector<std::vector<std::uint8_t>> src_bytes(map.sources.size());
    Found fw = locate_source(data.data(), data.size(), ks, nullptr, opt);
    if (!fw.found) {
        emit(log, "source " + ks.name + " not found (" + hex64(ks.fnv) + ")");
        // the message of the map's preset, by name (map.h known_presets)
        const PresetInfo* preset = map.preset_info();
        std::string msg = preset ? preset->no_model_message : kMsgNoPresetM;
        if (!out.version.empty()) msg += " This file is version " + out.version + ".";
        return fail(kNoPresetM, msg);
    }
    emit(log, "source " + ks.name + " at offset " + std::to_string(fw.offset) + " (" + std::to_string(seconds_since(t0)) + " s)");
    out.key = make_key(map.key_label, fw.bytes);
    src_bytes[static_cast<std::size_t>(map.key_source)] = std::move(fw.bytes);
    if (map.version == 5) return convert_rest_v5(map, data, opt, out, src_bytes, t0, log);

    std::string missing;
    out.variants.assign(map.sources.size(), -1);
    for (std::size_t i = 0; i < map.sources.size(); ++i) {
        if (static_cast<int>(i) == map.key_source) continue;
        const Source& s = map.sources[i];
        if (s.has_variants()) continue;  // after the required sources (section 4.2, "Order and results")
        Found f = locate_source(data.data(), data.size(), s, &out.key, opt);
        if (!f.found) {
            emit(log, "source " + s.name + " not found");
            missing += (missing.empty() ? "" : ", ") + s.name;
            continue;
        }
        if (s.locate == Locate::Pair)
            emit(log, "source " + s.name + " at offsets " + std::to_string(f.offset) + " and " + std::to_string(f.offset2));
        else
            emit(log, "source " + s.name + " at offset " + std::to_string(f.offset));
        src_bytes[i] = std::move(f.bytes);
    }
    if (!missing.empty()) {
        emit(log, "missing sources: " + missing);
        return fail(kUnsupportedBuild, kMsgUnsupported);
    }
    // Section 4.3: the optional source with variants (F), tried in map order. No match is not an error.
    for (std::size_t i = 0; i < map.sources.size(); ++i) {
        const Source& s = map.sources[i];
        if (!s.has_variants()) continue;
        const std::string total = std::to_string(s.variants.size());
        FoundVariant fv = locate_variant(data.data(), data.size(), s, &out.key, opt);
        if (fv.variant < 0) {
            emit(log, "source " + s.name + " not found (none of its " + total + " variants): " + absent_groups(map, i));
            emit(log, kMsgFilterUnknown);
            out.notes.push_back(kMsgFilterUnknown);
            continue;
        }
        const std::string which = "variant " + std::to_string(fv.variant + 1);
        emit(log, "source " + s.name + " at offset " + std::to_string(fv.offset) + " (" + which + " of " + total + ")");
        std::string why;
        if (!decode_variant_block(s.variants[static_cast<std::size_t>(fv.variant)], fv.bytes, out.key, src_bytes[i], why)) {
            emit(log, "source " + s.name + ", " + which + ": " + why);
            return fail(kMapDamaged, kMsgMapDamaged);
        }
        out.variants[i] = fv.variant;
    }
    emit(log, "sources located in " + std::to_string(seconds_since(t0)) + " s");

    DecodeError de;
    if (!convert_all(map, src_bytes, out.variants, out.key, out.stages, de)) {
        emit(log, de.detail);
        out.stages.clear();
        switch (de.kind) {
            case DecodeErrorKind::NonFinite: return fail(kBadValue, kMsgBadValue);
            case DecodeErrorKind::OutOfRange: return fail(kMapDoesNotFit, kMsgMapDoesNotFit);
            case DecodeErrorKind::SourceDamaged: return fail(kMapDamaged, kMsgMapDamaged);
            default: return fail(kSelfCheckFailed, kMsgSelfCheck);
        }
    }
    emit(log, "all " + std::to_string(out.stages.size()) + " stage digests match" + found_groups(out.stages) + " (" +
                  std::to_string(seconds_since(t0)) + " s)");
    r.message = "The model was converted successfully.";
    for (const std::string& note : out.notes) r.message += " " + note;
    return r;
}

bool has_every_group(const Map& map, const Converted& c) {
    for (const Stage& st : map.stages) {
        if (!st.optional()) continue;
        bool present = false;
        for (const StageOut& so : c.stages) present = present || so.stage == &st;
        if (!present) return false;
    }
    return true;
}

Result write_converted(const Map& map, const Converted& c, const fs::path& out_dir, bool raw, const LogSink& log) {
    std::string werr;
    if (!write_output(out_dir, map, c.stages, c.key, raw, werr)) {
        emit(log, "output: " + werr);
        return fail(kOutputFailed, "The converted model could not be saved in " + out_dir.u8string() + ".");
    }
    emit(log, "written to " + (out_dir / kModelFileName).u8string() + (raw ? " (and raw tensors)" : ""));
    return Result{kOk, "The model was converted successfully."};
}

bool verify_model_file(const Map& map, const fs::path& file, std::string& why, ModelFileInfo* info) {
    if (info) *info = ModelFileInfo();
    std::vector<std::uint8_t> d;
    Result r;
    if (!read_file(file, d, r, nullptr)) {
        why = "cannot read " + file.u8string();
        return false;
    }
    // Container version 1 for map versions 1-4, 2 for map version 5 (OUTPUT-FORMAT.md).
    const std::uint32_t want_container = map.version == 5 ? 2 : 1;
    if (d.size() < 88 + 32 || std::string(d.begin(), d.begin() + 8) != "WGCMODEL" || rd32(&d[8]) != want_container) {
        why = "not a model file of this converter";
        return false;
    }
    if (info) info->container_version = static_cast<int>(want_container);
    const std::size_t body = d.size() - 32;
    Digest trailer;
    std::copy(d.begin() + static_cast<std::ptrdiff_t>(body), d.end(), trailer.begin());
    if (Sha256::hash(d.data(), body) != trailer) {
        why = "file damaged (whole-file hash)";
        return false;
    }
    // Every group of the map in map order; a group read from a source with variants may be absent.
    const std::uint32_t file_stages = rd32(&d[12]);
    std::uint32_t seen = 0;
    if (info) {
        std::copy(d.begin() + 48, d.begin() + 80, info->map_sha256.begin());
        info->same_map = info->map_sha256 == map.file_sha256;
    }
    Digest key;
    std::copy(d.begin() + 16, d.begin() + 48, key.begin());
    const std::uint64_t index_size = rd64(&d[80]);
    std::size_t pos = 88;
    const std::size_t index_end = 88 + static_cast<std::size_t>(index_size);
    if (index_size > body || index_end > body) {
        why = "bad index";
        return false;
    }
    auto need = [&](std::size_t n) { return pos + n <= index_end; };
    auto name_at = [&](std::string& s) {
        if (!need(2)) return false;
        const std::size_t n = d[pos] | (std::size_t(d[pos + 1]) << 8);
        if (!need(2 + n)) return false;
        s.assign(reinterpret_cast<const char*>(&d[pos + 2]), n);
        pos += 2 + n;
        return true;
    };
    for (const Stage& st : map.stages) {
        std::string name;
        if (st.optional() || st.gated()) {  // may be absent: no variant of F found / another variant than its gate
            const std::size_t at = pos;
            const bool present = seen < file_stages && name_at(name) && name == st.name;
            pos = at;
            if (!present) {
                if (info) info->absent_groups.push_back(st.name);
                continue;
            }
        }
        if (seen >= file_stages || !name_at(name) || name != st.name || !need(36)) {
            why = "stage " + st.name + " missing or out of order";
            return false;
        }
        ++seen;
        Digest dg;
        std::copy(d.begin() + static_cast<std::ptrdiff_t>(pos), d.begin() + static_cast<std::ptrdiff_t>(pos + 32), dg.begin());
        pos += 32;
        bool known = dg == st.digest;
        if (st.optional()) {  // the digest of any one variant
            known = false;
            for (const Digest& vd : st.variant_digests) known = known || dg == vd;
        }
        if (!known || rd32(&d[pos]) != st.tensors.size()) {
            why = "stage " + st.name + " does not match the map";
            return false;
        }
        pos += 4;
        Sha256 h;
        h.update(key.data(), key.size());
        for (const Tensor& t : st.tensors) {
            std::string tname;
            if (!name_at(tname) || tname != t.name || !need(2)) {
                why = "tensor " + st.name + "/" + t.name + " missing";
                return false;
            }
            const std::uint8_t dtype = d[pos], rank = d[pos + 1];
            pos += 2;
            if (dtype != dtype_code(t.dtype) || rank != t.shape.size() || !need(4u * rank + 16)) {
                why = "tensor " + st.name + "/" + t.name + " has another type or shape";
                return false;
            }
            for (std::size_t a = 0; a < rank; ++a, pos += 4) {
                if (rd32(&d[pos]) != static_cast<std::uint64_t>(t.shape[a])) {
                    why = "tensor " + st.name + "/" + t.name + " has another shape";
                    return false;
                }
            }
            const std::uint64_t off = rd64(&d[pos]), size = rd64(&d[pos + 8]);
            pos += 16;
            if (size != t.byte_size() || off < index_end || off > body || size > body - off) {
                why = "tensor " + st.name + "/" + t.name + " has a bad data range";
                return false;
            }
            h.update(&d[static_cast<std::size_t>(off)], static_cast<std::size_t>(size));
        }
        if (h.finish() != dg) {
            why = "stage " + st.name + ": data does not match its digest";
            return false;
        }
    }
    if (seen != file_stages) {
        why = "different number of stages";
        return false;
    }
    return true;
}

}  // namespace wgc
