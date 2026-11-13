# SPDX-License-Identifier: MIT
"""Minimal PE32+ DLL files with a version resource, for the installer tests (mock FSR DLLs, mock front end).

Written for this project from Microsoft's public PE/COFF and VS_VERSIONINFO documentation. The files have one
.rsrc section and no code: they are never loaded, only inspected (version strings, hashes, text markers).
"""
import struct


def _utf16z(s):
    return s.encode("utf-16-le") + b"\0\0"


def _pad4(b):
    return b + b"\0" * (-len(b) % 4)


def _node(key, value=b"", text=False, children=(), value_len=None):
    head = struct.pack("<HHH", 0, 0, 1 if text else 0) + _utf16z(key)
    body = _pad4(head) + value
    if children:
        body = _pad4(body) + b"".join(_pad4(c) for c in children)
    if value_len is None:
        value_len = len(value) // 2 if text else len(value)
    return struct.pack("<HH", len(body), value_len) + body[4:]


def version_info(strings, version=(1, 0, 0, 0)):
    ms = (version[0] << 16) | version[1]
    ls = (version[2] << 16) | version[3]
    fixed = struct.pack("<13I", 0xFEEF04BD, 0x00010000, ms, ls, ms, ls, 0x3F, 0, 0x40004, 2, 0, 0, 0)
    entries = [_node(k, v.encode("utf-16-le") + b"\0\0", text=True) for k, v in strings.items()]
    table = _node("040904B0", children=entries)
    sfi = _node("StringFileInfo", children=[table])
    var = _node("VarFileInfo", children=[_node("Translation", struct.pack("<HH", 0x409, 1200))])
    return _node("VS_VERSION_INFO", fixed, children=[sfi, var])


def _export_table(rva, dll_name, names):
    """An export directory (.edata) at `rva` naming `names`; every function points at the section start."""
    names = sorted(names)
    n = len(names)
    dir_size = 40
    funcs = rva + dir_size
    name_ptrs = funcs + 4 * n
    ordinals = name_ptrs + 4 * n
    strings = ordinals + 2 * n
    blob = b""
    ptrs = []
    for nm in [dll_name] + names:
        ptrs.append(strings + len(blob))
        blob += nm.encode() + b"\0"
    d = struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, ptrs[0], 1, n, n, funcs, name_ptrs, ordinals)
    d += struct.pack("<%dI" % n, *([rva & ~0xFFF] * n)) if n else b""
    d += struct.pack("<%dI" % n, *ptrs[1:]) if n else b""
    d += struct.pack("<%dH" % n, *range(n)) if n else b""
    return d + blob


def _import_table(rva, dlls):
    """An import directory (.idata) at `rva`: one descriptor per DLL name (no thunks: the files are never loaded, only
    their imported DLL names are read), then the null descriptor, then the names."""
    desc_size = 20 * (len(dlls) + 1)
    names, blob = [], b""
    for d in dlls:
        names.append(rva + desc_size + len(blob))
        blob += d.encode() + b"\0"
    d = b"".join(struct.pack("<IIIII", 0, 0, 0, n, 0) for n in names) + b"\0" * 20
    return d + blob


def make_dll(strings=None, version=(1, 0, 0, 0), extra=b"", exports=None, dll_name="mock.dll", imports=None):
    """A PE32+ DLL with a version resource (if strings is not None), an export table naming `exports` (if given) and an
    import table naming the DLLs in `imports` (if given), followed by `extra` bytes."""
    file_align, sect_align = 0x200, 0x1000
    rsrc = b""
    if strings is not None:
        vi = version_info(strings, version)
        rva = sect_align
        root = struct.pack("<IIHHHH", 0, 0, 0, 0, 0, 1) + struct.pack("<II", 16, 0x80000000 | 0x18)
        lvl2 = struct.pack("<IIHHHH", 0, 0, 0, 0, 0, 1) + struct.pack("<II", 1, 0x80000000 | 0x30)
        lvl3 = struct.pack("<IIHHHH", 0, 0, 0, 0, 0, 1) + struct.pack("<II", 0x409, 0x48)
        data_entry = struct.pack("<IIII", rva + 0x58, len(vi), 0, 0)
        rsrc = root + lvl2 + lvl3 + data_entry
        rsrc += b"\0" * (0x58 - len(rsrc)) + vi
    content = rsrc + b"\0" * (-len(rsrc) % 16) if rsrc else b"\0" * 16  # exported functions point at the section start
    edata_rva = edata_size = 0
    if exports is not None:
        edata_rva = sect_align + len(content)
        ed = _export_table(edata_rva, dll_name, exports)
        edata_size = len(ed)
        content += ed
    idata_rva = idata_size = 0
    if imports is not None:
        content += b"\0" * (-len(content) % 8)
        idata_rva = sect_align + len(content)
        it = _import_table(idata_rva, imports)
        idata_size = len(it)
        content += it
    raw = content + b"\0" * (-len(content) % file_align) if content else b"\0" * file_align
    vsize = max(len(content), 1)
    size_of_image = sect_align + ((vsize + sect_align - 1) // sect_align) * sect_align
    dos = b"MZ" + b"\0" * 0x3A + struct.pack("<I", 0x40)
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = struct.pack("<HBBIIIII", 0x20B, 14, 0, 0, len(raw), 0, 0, sect_align)
    opt += struct.pack("<QIIHHHHHHIIIIHH", 0x180000000, sect_align, file_align, 6, 0, 0, 0, 6, 0, 0, size_of_image, 0x200, 0, 2, 0x160)
    opt += struct.pack("<QQQQII", 0x100000, 0x1000, 0x100000, 0x1000, 0, 16)
    dirs = [(0, 0)] * 16
    if exports is not None:
        dirs[0] = (edata_rva, edata_size)
    if imports is not None:
        dirs[1] = (idata_rva, idata_size)
    if rsrc:
        dirs[2] = (sect_align, len(rsrc))
    opt += b"".join(struct.pack("<II", a, b) for a, b in dirs)
    assert len(opt) == 240
    section = struct.pack("<8sIIIIIIHHI", b".rsrc", vsize, sect_align, len(raw), file_align, 0, 0, 0, 0, 0x40000040)
    headers = dos + b"PE\0\0" + coff + opt + section
    headers += b"\0" * (file_align - len(headers))
    return headers + raw + extra


FFX_API_EXPORTS = ["ffxConfigure", "ffxCreateContext", "ffxDestroyContext", "ffxDispatch", "ffxQuery"]
FSR3_EXPORTS = ["ffxFsr3ContextCreate", "ffxFsr3ContextDispatchUpscale", "ffxFsr3ContextDestroy", "ffxFsr3DispatchFrameGeneration",
                "ffxFsr3ConfigureFrameGeneration", "ffxFsr3GetJitterOffset"]
FSR3UPSCALER_EXPORTS = ["ffxFsr3UpscalerContextCreate", "ffxFsr3UpscalerContextDispatch", "ffxFsr3UpscalerContextDestroy",
                        "ffxFsr3UpscalerGetJitterOffset", "ffxFsr3UpscalerGetSharedResourceDescriptions"]
AMD = {"CompanyName": "Advanced Micro Devices, Inc.", "ProductName": "AMD FidelityFX", "FileDescription": "AMD FidelityFX API (DX12)"}
FRONT_END = {"CompanyName": "WrongGPU", "ProductName": "WrongGPU", "FileDescription": "WrongGPU FSR front end"}
