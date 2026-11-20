// SPDX-License-Identifier: MIT
// Linux: JPEG decoding with libjpeg-turbo (see jpeg_decode_linux.h). Written for this project.
#include "app/jpeg_decode_linux.h"

#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <jpeglib.h>

namespace wg::app::lui {

namespace {

struct Err {
    jpeg_error_mgr base;  // first: libjpeg hands back this pointer
    jmp_buf jump;
};

void OnError(j_common_ptr c) { longjmp(reinterpret_cast<Err*>(c->err)->jump, 1); }
void Silent(j_common_ptr, int) {}

}  // namespace

bool LooksLikeJpeg(const char* utf8Path) {
    FILE* f = std::fopen(utf8Path, "rb");
    if (!f) return false;
    unsigned char m[3] = {};
    const size_t n = std::fread(m, 1, 3, f);
    std::fclose(f);
    return n == 3 && m[0] == 0xFF && m[1] == 0xD8 && m[2] == 0xFF;
}

bool DecodeJpegFile(const char* utf8Path, unsigned minW, int& w, int& h, std::vector<uint8_t>& bgra) {
    FILE* f = std::fopen(utf8Path, "rb");
    if (!f) return false;
    jpeg_decompress_struct cin;
    Err err;
    cin.err = jpeg_std_error(&err.base);
    err.base.error_exit = OnError;
    err.base.emit_message = Silent;
    // Changed between setjmp and a longjmp, so volatile (C 7.13.2.1), and plain malloc storage: nothing with a destructor
    // lives across the jump, and nothing in the jump region can throw.
    uint8_t* volatile px = nullptr;
    volatile size_t size = 0;
    volatile bool ok = false;
    if (!setjmp(err.jump)) {
        jpeg_create_decompress(&cin);
        jpeg_stdio_src(&cin, f);
        jpeg_read_header(&cin, TRUE);
        const bool colour = cin.jpeg_color_space == JCS_YCbCr || cin.jpeg_color_space == JCS_RGB;
        if ((cin.jpeg_color_space == JCS_GRAYSCALE || colour) && cin.image_width > 0 && cin.image_height > 0 &&
            cin.image_width <= 8192 && cin.image_height <= 8192) {
            cin.out_color_space = JCS_EXT_BGRA;
            unsigned denom = 1;
            while (denom < 8 && (cin.image_width + denom * 2 - 1) / (denom * 2) >= minW) denom *= 2;
            cin.scale_num = 1;
            cin.scale_denom = denom;
            jpeg_start_decompress(&cin);
            const size_t rowBytes = (size_t)cin.output_width * 4;
            size = rowBytes * cin.output_height;
            px = static_cast<uint8_t*>(std::malloc(size));
            if (px) {
                while (cin.output_scanline < cin.output_height) {
                    JSAMPROW row = px + (size_t)cin.output_scanline * rowBytes;
                    jpeg_read_scanlines(&cin, &row, 1);
                }
                jpeg_finish_decompress(&cin);
                w = (int)cin.output_width;
                h = (int)cin.output_height;
                ok = true;
            }
        }
    }
    jpeg_destroy_decompress(&cin);
    std::fclose(f);
    // The copy may throw bad_alloc; the decoder and the file are already released, and the holder frees the pixels.
    const std::unique_ptr<uint8_t, decltype(&std::free)> hold(px, &std::free);
    if (ok) bgra.assign(hold.get(), hold.get() + size);
    return ok;
}

}  // namespace wg::app::lui
