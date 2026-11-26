# SPDX-License-Identifier: MIT
"""The addressing of the hybrid decoder blocks with their expand fused in (swin_block.hlsl, SW_HYB with SW_ECL:
swin_block_c96_add_ex_hyb, swin_block_c64_add_ex_hyb), checked exhaustively on the CPU against the unfused route.

The unfused hybrid decoder block reads its expanded input E at the coarse source of a member (sw_coarse_src: the 2 x 2
group index mirrored with sw_mirror, the sub-pixel kept). The fused block computes the expand of the window's 2 x 2 low-
resolution tokens from the base sw_low0c into LDS records [low token][sub-pixel] and reads the record
(mirrored group - base, sub-pixel). This test proves, for every map size n in 16 .. 4100, every window offset o in
{0, 2} (the offsets of the fused launches: even) and every window of the grid:
  * the window's mirrored groups of its rows (and columns) lie in {base, base + 1} (so the 2 x 2 record block holds
    every token the window reads);
  * the record read for every member (relative group, sub-pixel) is the token and sub-pixel of the coarse source
    (the same E element as the unfused block reads);
  * the record's low token is a real token (the clamp of the fused expand's row / column to the map does not move a
    token that is read).
The E element at a (token, sub-pixel) is the same value in both routes by construction (the same weights, the same
chain order and the same hyb_q8_of_chain point: expand_c96_hyb / expand_c128_hyb), which only a GPU run can confirm
bit for bit (clean/l/runner)."""
import unittest


def sw_mirror(p, n):
    p = -p if p < 0 else p
    return 2 * (n - 1) - p if p >= n else p


def sw_low0c(p0, n):
    g, nc = p0 >> 1, n >> 1
    return min(sw_mirror(g, nc), sw_mirror(g + 1, nc))


class TestFusedExpandAddressing(unittest.TestCase):
    def check_axis(self, n, o):
        nc = n >> 1
        nwin = (n + o + 3) >> 2
        for w in range(nwin):
            p0 = 4 * w - o
            base = sw_low0c(p0, n)
            for p in range(p0, p0 + 4):
                m = sw_mirror(p >> 1, nc)                    # the coarse source's group
                rel = m - base
                self.assertIn(rel, (0, 1), (n, o, w, p, m, base))
                lr = base + rel                              # the record's low token (the fused expand clamps to nc - 1)
                self.assertEqual(min(lr, nc - 1), m, (n, o, w, p))
                self.assertTrue(0 <= m <= nc - 1, (n, o, w, p, m))
                # the sub-pixel of the member is p & 1 in both routes; the source pixel of sw_coarse_src
                src = 2 * m + (p & 1)
                self.assertTrue(0 <= src <= n - 1, (n, o, w, p, src))
            # the record block's tokens base and base + 1 are the clamped ones the fused expand computes
            self.assertTrue(base >= 0 and base <= nc - 1, (n, o, w, base))

    def test_all_sizes(self):
        for n in range(16, 4101):   # (below, sw_mirror itself leaves the map: the unfused route too)
            for o in (0, 2):
                self.check_axis(n, o)


if __name__ == "__main__":
    unittest.main()
