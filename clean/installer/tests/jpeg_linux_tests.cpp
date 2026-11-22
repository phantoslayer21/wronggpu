// SPDX-License-Identifier: MIT
// Linux: the JPEG decoder (src/app/jpeg_decode_linux.cpp). The test makes its own JPEGs with libjpeg (no art is stored in the
// repo): four colour blocks, so size, orientation and colours are all visible. JPEG is lossy, so colours are checked within a
// tolerance at the middle of each block, away from the edges where chroma bleeds.
// Usage: jpeg-linux-tests                        (writes its files under $TMPDIR, or the current folder)
//        jpeg-linux-tests --make <file> <w> <h>   only writes that picture (run_linux_ui_tests.py's synthetic Steam art)
#include "app/jpeg_decode_linux.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <jpeglib.h>

using namespace wg::app::lui;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

struct Rgb { int r, g, b; };
// top left, top right, bottom left, bottom right
static const Rgb kBlocks[4] = {{220, 30, 30}, {30, 200, 40}, {40, 50, 230}, {240, 240, 240}};

static bool MakeJpeg(const std::string& path, int w, int h, bool progressive, bool grey) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    jpeg_compress_struct c;
    jpeg_error_mgr e;
    c.err = jpeg_std_error(&e);
    jpeg_create_compress(&c);
    jpeg_stdio_dest(&c, f);
    c.image_width = (JDIMENSION)w;
    c.image_height = (JDIMENSION)h;
    c.input_components = grey ? 1 : 3;
    c.in_color_space = grey ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&c);
    jpeg_set_quality(&c, 92, TRUE);
    if (progressive) jpeg_simple_progression(&c);
    jpeg_start_compress(&c, TRUE);
    std::vector<unsigned char> row((size_t)w * c.input_components);
    while (c.next_scanline < c.image_height) {
        const int y = (int)c.next_scanline;
        for (int x = 0; x < w; ++x) {
            const Rgb& k = kBlocks[(y >= h / 2 ? 2 : 0) + (x >= w / 2 ? 1 : 0)];
            if (grey) row[x] = (unsigned char)((k.r * 30 + k.g * 59 + k.b * 11) / 100);
            else { row[(size_t)x * 3] = (unsigned char)k.r; row[(size_t)x * 3 + 1] = (unsigned char)k.g; row[(size_t)x * 3 + 2] = (unsigned char)k.b; }
        }
        JSAMPROW r = row.data();
        jpeg_write_scanlines(&c, &r, 1);
    }
    jpeg_finish_compress(&c);
    jpeg_destroy_compress(&c);
    std::fclose(f);
    return true;
}

static bool Near(const std::vector<uint8_t>& px, int w, int x, int y, const Rgb& want, int tol) {
    const uint8_t* p = &px[((size_t)y * w + x) * 4];  // BGRA
    auto d = [](int a, int b) { return a > b ? a - b : b - a; };
    return d(p[0], want.b) <= tol && d(p[1], want.g) <= tol && d(p[2], want.r) <= tol && p[3] == 255;
}

static void CheckBlocks(const std::vector<uint8_t>& px, int w, int h, int tol, const char* what) {
    for (int i = 0; i < 4; ++i) {
        const int x = (i % 2 ? 3 * w / 4 : w / 4), y = (i / 2 ? 3 * h / 4 : h / 4);
        if (!Near(px, w, x, y, kBlocks[i], tol)) {
            const uint8_t* p = &px[((size_t)y * w + x) * 4];
            std::fprintf(stderr, "FAIL %s: block %d at %d,%d is B%d G%d R%d A%d, want R%d G%d B%d within %d\n", what, i, x, y, p[0], p[1], p[2], p[3],
                         kBlocks[i].r, kBlocks[i].g, kBlocks[i].b, tol);
            ++fails;
        }
    }
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--make") return MakeJpeg(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), false, false) ? 0 : 1;
    const char* t = std::getenv("TMPDIR");
    const std::string dir = (t && *t) ? t : ".";
    const std::string base = dir + "/jpeg-linux-test-", plain = base + "plain.jpg", prog = base + "prog.jpg", grey = base + "grey.jpg";
    CHECK(MakeJpeg(plain, 192, 128, false, false));
    CHECK(MakeJpeg(prog, 192, 128, true, false));
    CHECK(MakeJpeg(grey, 192, 128, false, true));

    int w = 0, h = 0;
    std::vector<uint8_t> px;
    CHECK(LooksLikeJpeg(plain.c_str()));
    // full size: the asked width is the image's own
    CHECK(DecodeJpegFile(plain.c_str(), 192, w, h, px));
    CHECK(w == 192 && h == 128 && px.size() == (size_t)192 * 128 * 4);
    if (px.size() == (size_t)192 * 128 * 4) CheckBlocks(px, w, h, 14, "full size");
    // a progressive file gives the same picture
    CHECK(DecodeJpegFile(prog.c_str(), 192, w, h, px));
    CHECK(w == 192 && h == 128);
    if (px.size() == (size_t)192 * 128 * 4) CheckBlocks(px, w, h, 14, "progressive");
    // the DCT scaling: never narrower than asked, by 2, 4 or 8
    CHECK(DecodeJpegFile(plain.c_str(), 90, w, h, px));
    CHECK(w == 96 && h == 64);
    if (px.size() == (size_t)96 * 64 * 4) CheckBlocks(px, w, h, 14, "half size");
    CHECK(DecodeJpegFile(plain.c_str(), 40, w, h, px));
    CHECK(w == 48 && h == 32);
    CHECK(DecodeJpegFile(plain.c_str(), 1, w, h, px));
    CHECK(w == 24 && h == 16);
    // grey becomes opaque BGRA with equal channels
    CHECK(DecodeJpegFile(grey.c_str(), 192, w, h, px));
    CHECK(w == 192 && h == 128 && px.size() == (size_t)192 * 128 * 4);
    if (px.size() == (size_t)192 * 128 * 4) { const uint8_t* p = &px[((size_t)96 * 192 + 150) * 4]; CHECK(p[0] == p[1] && p[1] == p[2] && p[3] == 255 && p[0] > 200); }

    // failures leave the outputs alone and do not crash
    w = h = 7; px.assign(3, 9);
    CHECK(!DecodeJpegFile((base + "missing.jpg").c_str(), 100, w, h, px) && w == 7 && px.size() == 3);
    const std::string cut = base + "cut.jpg", junk = base + "junk.jpg", late = base + "late.jpg";
    {
        FILE* in = std::fopen(plain.c_str(), "rb");
        std::vector<unsigned char> all(1 << 20);
        const size_t n = in ? std::fread(all.data(), 1, all.size(), in) : 0;
        if (in) std::fclose(in);
        CHECK(n > 400);
        FILE* o = std::fopen(cut.c_str(), "wb");
        if (o) { std::fwrite(all.data(), 1, 300, o); std::fclose(o); }   // headers only, no picture
        o = std::fopen(junk.c_str(), "wb");
        if (o) { std::fputs("not a picture at all", o); std::fclose(o); }
        // a whole picture, then a Huffman table marker with length 0 before the end: libjpeg fails in
        // jpeg_finish_decompress, after the pixel buffer exists (the error path that has to free it)
        o = std::fopen(late.c_str(), "wb");
        const unsigned char bad[4] = {0xFF, 0xC4, 0x00, 0x00};
        if (o && n > 2) { std::fwrite(all.data(), 1, n - 2, o); std::fwrite(bad, 1, 4, o); std::fwrite(all.data() + n - 2, 1, 2, o); }
        if (o) std::fclose(o);
    }
    CHECK(!DecodeJpegFile(late.c_str(), 100, w, h, px) && w == 7 && px.size() == 3);
    CHECK(LooksLikeJpeg(cut.c_str()));
    CHECK(!DecodeJpegFile(cut.c_str(), 100, w, h, px) || w > 0);   // libjpeg pads a truncated file grey: either answer is a clean one
    CHECK(!LooksLikeJpeg(junk.c_str()));
    CHECK(!DecodeJpegFile(junk.c_str(), 100, w, h, px));
    CHECK(!LooksLikeJpeg((base + "missing.jpg").c_str()));

    for (const std::string* f : {&plain, &prog, &grey, &cut, &junk, &late}) std::remove(f->c_str());
    std::printf(fails ? "jpeg-linux: %d failure(s)\n" : "jpeg-linux: all checks passed\n", fails);
    return fails ? 1 : 0;
}
