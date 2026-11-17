# SPDX-License-Identifier: MIT
"""A WRONGGPU_TESTING Linux build's identity sum: what the PE checksum is to the testing launcher on Windows.

The Linux testing launcher has a section .wg_testing_id (src/platform/posix/testing_identity_linux.cpp): the identity text,
then "wg-sha256:" and 32 bytes. After the link, this tool writes there the SHA-256 of the whole file with those 32 bytes
counted as zero, so a byte changed or data appended afterwards no longer matches. tests/launcher_build_facts.py checks it
(with the identity text and the test hooks' strings) before the launcher suite runs a program.

    elf_testing_identity.py <file>     write the sum (the testing build's POST_BUILD step)

Written for this project; the standard library only.
"""
import hashlib
import struct
import sys

SECTION = b".wg_testing_id"
SUM_MARK = b"wg-sha256:"
SHT_NOBITS = 8


def sections(data):
    """The sections of a 64-bit little-endian ELF file: [(name, file offset, size, type)]. ValueError when it is not one."""
    if data[:4] != b"\x7fELF" or len(data) < 64 or data[4] != 2 or data[5] != 1:
        raise ValueError("not a 64-bit little-endian ELF file")
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    if not shnum or shstrndx >= shnum or shoff + shnum * shentsize > len(data):
        raise ValueError("no readable section table")
    heads = [struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize) for i in range(shnum)]
    names = heads[shstrndx][4]
    out = []
    for name, kind, _flags, _addr, off, size in heads:
        start = names + name
        out.append((data[start:data.index(b"\0", start)], off, size, kind))
    return out


def identity_section(data):
    """(file offset, bytes) of the identity section; ValueError when there is none."""
    for name, off, size, kind in sections(data):
        if name == SECTION and kind != SHT_NOBITS and off + size <= len(data):
            return off, data[off:off + size]
    raise ValueError("no %s section" % SECTION.decode())


def sum_slot(data):
    """The file offset of the 32-byte sum; ValueError when the identity section has no room for one."""
    off, body = identity_section(data)
    at = body.find(SUM_MARK)
    if at < 0 or at + len(SUM_MARK) + 32 > len(body):
        raise ValueError("the identity section holds no sum")
    return off + at + len(SUM_MARK)


def expected_sum(data, slot):
    return hashlib.sha256(data[:slot] + b"\0" * 32 + data[slot + 32:]).digest()


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[1], "rb") as f:
        data = f.read()
    slot = sum_slot(data)
    with open(argv[1], "r+b") as f:
        f.seek(slot)
        f.write(expected_sum(data, slot))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
