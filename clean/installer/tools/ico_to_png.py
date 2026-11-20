# SPDX-License-Identifier: MIT
"""Writes the largest PNG frame of assets/wronggpu.ico as a .png file (the Linux icon of the AppImage).

    ico_to_png.py <icon.ico> <out.png>

The .ico is the project's own icon (make_payload.py copies it to app.ico for Windows); its biggest frame is stored as PNG, so
this copies bytes and draws nothing.
"""
import struct
import sys


def main():
    src, dst = sys.argv[1:3]
    b = open(src, "rb").read()
    reserved, kind, count = struct.unpack("<HHH", b[:6])
    if reserved != 0 or kind != 1 or count == 0:
        sys.exit("%s is not an .ico file" % src)
    best = None
    for i in range(count):
        w, h, _c, _r, _p, _bpp, size, off = struct.unpack("<BBBBHHII", b[6 + 16 * i:22 + 16 * i])
        w, h = w or 256, h or 256        # 0 means 256 in the directory
        if b[off:off + 8] == b"\x89PNG\r\n\x1a\n" and (best is None or w > best[0]):
            best = (w, b[off:off + size])
    if best is None:
        sys.exit("%s has no PNG frame" % src)
    open(dst, "wb").write(best[1])
    print("%s: %dx%d frame -> %s" % (src, best[0], best[0], dst))


if __name__ == "__main__":
    main()
