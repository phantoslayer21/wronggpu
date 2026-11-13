// SPDX-License-Identifier: MIT
// Writing the converter output (spec 08 sections 8 and 10). Written for this project.
#include "output.h"

#include <chrono>
#include <fstream>
#include <random>
#include <system_error>

namespace wgc {

namespace fs = std::filesystem;

namespace {

void put_u8(std::vector<std::uint8_t>& b, std::uint8_t v) { b.push_back(v); }
void put_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(std::uint8_t(v));
    b.push_back(std::uint8_t(v >> 8));
}
void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(v >> (8 * i)));
}
void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(std::uint8_t(v >> (8 * i)));
}
void put_bytes(std::vector<std::uint8_t>& b, const void* p, std::size_t n) {
    const std::uint8_t* q = static_cast<const std::uint8_t*>(p);
    b.insert(b.end(), q, q + n);
}
void put_name(std::vector<std::uint8_t>& b, const std::string& s) {
    put_u16(b, static_cast<std::uint16_t>(s.size()));
    put_bytes(b, s.data(), s.size());
}

const std::uint64_t kAlign = 64;
std::uint64_t align_up(std::uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Index with the given data offsets (one per tensor, in order).
std::vector<std::uint8_t> build_index(const std::vector<StageOut>& stages, const std::vector<std::uint64_t>& offsets) {
    std::vector<std::uint8_t> ix;
    std::size_t k = 0;
    for (const StageOut& so : stages) {
        put_name(ix, so.stage->name);
        put_bytes(ix, so.digest.data(), so.digest.size());
        put_u32(ix, static_cast<std::uint32_t>(so.tensors.size()));
        for (const TensorOut& to : so.tensors) {
            const Tensor& t = *to.tensor;
            put_name(ix, t.name);
            put_u8(ix, dtype_code(t.dtype));
            put_u8(ix, static_cast<std::uint8_t>(t.shape.size()));
            for (std::int64_t d : t.shape) put_u32(ix, static_cast<std::uint32_t>(d));
            put_u64(ix, offsets.empty() ? 0 : offsets[k]);
            put_u64(ix, to.bytes.size());
            ++k;
        }
    }
    return ix;
}

bool write_file(const fs::path& p, const std::uint8_t* data, std::size_t n, std::string& error) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) {
        error = "cannot create " + p.u8string();
        return false;
    }
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    f.flush();
    if (!f) {
        error = "cannot write " + p.u8string();
        return false;
    }
    return true;
}

std::string random_suffix() {
    std::random_device rd;
    std::mt19937_64 g((static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^
                      static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    static const char digits[] = "0123456789abcdef";
    std::string s;
    std::uint64_t v = g();
    for (int i = 0; i < 16; ++i, v >>= 4) s.push_back(digits[v & 15]);
    return s;
}

// Moves src onto dst, replacing dst if it exists. std::filesystem::rename replaces an existing file
// on POSIX; where it refuses, the old file is moved aside first and restored if the move fails.
bool replace_path(const fs::path& src, const fs::path& dst, std::string& error) {
    std::error_code ec;
    fs::rename(src, dst, ec);
    if (!ec) return true;
    if (!fs::exists(dst)) {
        error = "cannot move " + src.u8string() + " to " + dst.u8string() + ": " + ec.message();
        return false;
    }
    const fs::path aside = dst.parent_path() / (dst.filename().u8string() + ".old-" + random_suffix());
    fs::rename(dst, aside, ec);
    if (ec) {
        error = "cannot replace " + dst.u8string() + ": " + ec.message();
        return false;
    }
    fs::rename(src, dst, ec);
    if (ec) {
        std::error_code ec2;
        fs::rename(aside, dst, ec2);
        error = "cannot move " + src.u8string() + " to " + dst.u8string() + ": " + ec.message();
        return false;
    }
    fs::remove_all(aside, ec);
    return true;
}

}  // namespace

std::vector<std::uint8_t> build_container(const Map& map, const std::vector<StageOut>& stages, const Digest& key) {
    const std::uint64_t header_size = 8 + 4 + 4 + 32 + 32 + 8;
    const std::uint64_t index_size = build_index(stages, {}).size();
    std::vector<std::uint64_t> offsets;
    std::uint64_t pos = align_up(header_size + index_size);
    for (const StageOut& so : stages) {
        for (const TensorOut& to : so.tensors) {
            offsets.push_back(pos);
            pos = align_up(pos + to.bytes.size());
        }
    }
    const std::vector<std::uint8_t> index = build_index(stages, offsets);

    std::vector<std::uint8_t> out;
    out.reserve(static_cast<std::size_t>(pos + 32));
    put_bytes(out, "WGCMODEL", 8);
    put_u32(out, map.version == 5 ? 2 : 1);  // container version 2: the dtypes of map version 5 (OUTPUT-FORMAT.md)
    put_u32(out, static_cast<std::uint32_t>(stages.size()));
    put_bytes(out, key.data(), key.size());
    put_bytes(out, map.file_sha256.data(), map.file_sha256.size());
    put_u64(out, index.size());
    put_bytes(out, index.data(), index.size());
    std::size_t k = 0;
    for (const StageOut& so : stages) {
        for (const TensorOut& to : so.tensors) {
            out.resize(static_cast<std::size_t>(offsets[k++]), 0);
            put_bytes(out, to.bytes.data(), to.bytes.size());
        }
    }
    out.resize(static_cast<std::size_t>(pos), 0);
    const Digest whole = Sha256::hash(out.data(), out.size());
    put_bytes(out, whole.data(), whole.size());
    return out;
}

bool write_output(const fs::path& out_dir, const Map& map, const std::vector<StageOut>& stages, const Digest& key,
                  bool raw, std::string& error) {
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (!fs::is_directory(out_dir, ec)) {
        error = "cannot create the output folder " + out_dir.u8string();
        return false;
    }
    const fs::path staging = out_dir / (".wronggpu-convert-staging-" + random_suffix());
    fs::create_directory(staging, ec);
    if (ec) {
        error = "cannot create " + staging.u8string() + ": " + ec.message();
        return false;
    }
    auto fail = [&](const std::string& what) {
        error = what;
        std::error_code ec2;
        fs::remove_all(staging, ec2);
        return false;
    };

    const std::vector<std::uint8_t> container = build_container(map, stages, key);
    if (!write_file(staging / kModelFileName, container.data(), container.size(), error)) return fail(error);

    if (raw) {
        for (const StageOut& so : stages) {
            const fs::path dir = staging / "raw" / so.stage->name;
            fs::create_directories(dir, ec);
            if (ec) return fail("cannot create " + dir.u8string() + ": " + ec.message());
            for (const TensorOut& to : so.tensors) {
                if (!write_file(dir / (to.tensor->name + ".bin"), to.bytes.data(), to.bytes.size(), error))
                    return fail(error);
            }
        }
        // Raw output first (test aid), the model file last: its appearance is the commit point.
        const fs::path old_raw = out_dir / ("raw.old-" + random_suffix());
        if (fs::exists(out_dir / "raw", ec)) {
            fs::rename(out_dir / "raw", old_raw, ec);
            if (ec) return fail("cannot replace " + (out_dir / "raw").u8string() + ": " + ec.message());
        }
        fs::rename(staging / "raw", out_dir / "raw", ec);
        if (ec) {
            std::error_code ec2;
            if (fs::exists(old_raw, ec2)) fs::rename(old_raw, out_dir / "raw", ec2);
            return fail("cannot move the raw output into place: " + ec.message());
        }
        fs::remove_all(old_raw, ec);
    }

    if (!replace_path(staging / kModelFileName, out_dir / kModelFileName, error)) return fail(error);
    fs::remove_all(staging, ec);
    return true;
}

}  // namespace wgc
