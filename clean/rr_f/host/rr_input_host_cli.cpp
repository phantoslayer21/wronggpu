// SPDX-License-Identifier: MIT
// Test driver of rr_input_host.h (tests/test_input_host.py): reads frames as whitespace-separated fields, prints
// words 0..99 of the constant block per frame. Floats travel as their bit patterns (hex) so nothing is rounded.
// Fields per frame: Wr Hr Wo Ho jx jy pjx pjy rawx rawy mvjit E Ep X3 reset par fc hwdepth inverted nrmdec rghpacked
// sfv b2 have_camera attached_mask, then 64 floats (V2C W2V V2Cp W2Vp, row-major), 44 SH words, 25 symbol floats.
#include <cstdio>
#include <string>
#include "rr_input_host.h"

using namespace rr_input;

static bool rd_u(FILE* f, uint32_t& v) { return fscanf(f, "%x", &v) == 1; }
static float rd_f(FILE* f) { uint32_t v = 0; rd_u(f, v); return bits_f(v); }
static int rd_i(FILE* f) { int v = 0; if (fscanf(f, "%d", &v) != 1) v = -1; return v; }

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--tables") {
        static float logt[256][4], expt[65][2];
        ff_tables(logt, expt);
        for (int i = 0; i < 256; ++i)
            printf("%08x %08x %08x\n", f_bits(logt[i][0]), f_bits(logt[i][1]), f_bits(logt[i][2]));
        for (int j = 0; j < 65; ++j) printf("%08x %08x\n", f_bits(expt[j][0]), f_bits(expt[j][1]));
        return 0;
    }
    FILE* f = argc > 1 ? fopen(argv[1], "r") : stdin;
    if (!f) return 2;
    for (;;) {
        Frame fr = {};
        fr.Wr = rd_i(f);
        if (fr.Wr < 0) break;
        fr.Hr = rd_i(f); fr.Wo = rd_i(f); fr.Ho = rd_i(f);
        fr.jx = rd_f(f); fr.jy = rd_f(f); fr.pjx = rd_f(f); fr.pjy = rd_f(f);
        fr.mv_raw_x = rd_f(f); fr.mv_raw_y = rd_f(f); fr.mv_jittered = rd_i(f) != 0;
        fr.E = rd_f(f); fr.Ep = rd_f(f); fr.X3 = rd_f(f);
        fr.reset = rd_i(f) != 0; fr.par = rd_i(f); fr.fc = rd_i(f);
        fr.hw_depth = rd_i(f) != 0; fr.depth_inverted = rd_i(f) != 0; fr.nrm_dec = rd_i(f) != 0;
        fr.rgh_packed = rd_i(f) != 0; fr.specular_from_v = rd_i(f) != 0; fr.b2 = rd_i(f) != 0;
        fr.have_camera = rd_i(f) != 0;
        const int mask = rd_i(f);
        for (int r = 0; r < ROLE_COUNT; ++r) fr.attached[r] = (mask >> r) & 1;
        Mat* ms[4] = {&fr.V2C, &fr.W2V, &fr.V2Cp, &fr.W2Vp};
        for (Mat* m : ms) for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m->m[i][j] = rd_f(f);
        for (int b = 0; b < 2; ++b) for (int s = 0; s < 11; ++s) for (int c = 0; c < 2; ++c) fr.SH[b][s][c] = (uint32_t)rd_i(f);
        for (int s = 0; s < SYM_COUNT; ++s) fr.sym[s] = rd_f(f);
        uint32_t w[100];
        const bool usable = scalar_words(fr, w);
        printf("%d", usable ? 1 : 0);
        for (int i = 0; i < 100; ++i) printf(" %08x", w[i]);
        printf("\n");
    }
    return 0;
}
