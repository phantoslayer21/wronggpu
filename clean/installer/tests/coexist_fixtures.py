# SPDX-License-Identifier: MIT
"""Synthetic stand-ins for the files other tools put next to a game, for the launcher's coexistence tests (M2).

Written for this project with mockpe.py: every file is generated here (a version resource, exports, imports or a few
words of text that the launcher's fingerprints look for). No real third-party file is copied, read or shipped; the
product names are only the words those tools put in their own version resources.
"""
import struct

import mockpe

MICROSOFT = {"CompanyName": "Microsoft Corporation", "ProductName": "Microsoft Windows Operating System", "FileDescription": "Windows component (synthetic)"}


def optiscaler(version=(0, 9, 5, 4)):
    return mockpe.make_dll({"ProductName": "OptiScaler", "FileDescription": "OptiScaler (synthetic test file)"}, version=version)


def optiscaler_text():
    """No version resource: recognised by the text in it (as OptiScaler's builds carry their name)."""
    return mockpe.make_dll(None, extra=b"synthetic proxy; reads OptiScaler.ini")


def reshade():
    return mockpe.make_dll({"CompanyName": "crosire", "ProductName": "ReShade", "FileDescription": "ReShade (synthetic)"}, version=(6, 3, 3, 0),
                           exports=["ReShadeRegisterAddon", "ReShadeUnregisterAddon"])


def specialk():
    return mockpe.make_dll({"ProductName": "Special K", "FileDescription": "Special K (synthetic)"}, version=(24, 10, 1, 0))


def dxvk():
    return mockpe.make_dll(None, extra=b"synthetic: DXVK_LOG_LEVEL dxvk.conf")


def vkd3d():
    return mockpe.make_dll(None, extra=b"synthetic: VKD3D_CONFIG vkd3d-proton")


def asi_loader():
    return mockpe.make_dll({"ProductName": "Ultimate ASI Loader", "CompanyName": "ThirteenAG", "FileDescription": "synthetic"}, version=(7, 0, 0, 0))


def cet():
    return mockpe.make_dll(None, extra="plugins\\cyber_engine_tweaks".encode("utf-16-le"))


def red4ext():
    return mockpe.make_dll(None, extra=b"synthetic RED4ext loader")


def ue4ss():
    return mockpe.make_dll(None, extra=b"synthetic UE4SS proxy")


def hip_runtime():
    return mockpe.make_dll({"CompanyName": "Advanced Micro Devices, Inc.", "ProductName": "HIP (synthetic)"}, exports=["hipInit", "hipMalloc"])


def microsoft():
    return mockpe.make_dll(MICROSOFT, version=(10, 0, 26100, 1))


def fakenvapi():
    return mockpe.make_dll(None, extra=b"synthetic fakenvapi")


def unknown(tag="x"):
    return mockpe.make_dll(None, extra=("an unrelated synthetic proxy " + tag).encode())


def legacy_loader():
    """WrongGPU 0.1.x / 0.2.x (DLSS 4.5 AMD): its loader names its log."""
    return b"MZ old loader " + "dlss45-loader.log".encode("utf-16-le")


def exe(imports, extra=b""):
    """A game executable (a PE image whose import table names `imports`)."""
    return mockpe.make_dll({"ProductName": "Synthetic Game"}, imports=imports, extra=extra)


# ---- regression stand-ins of 2026-10-17 (a false "DXVK as nvngx.dll" refusal on the owner's PC). Synthetic only: no byte
# of any real DLSS DLL (nvngx) or OptiScaler build is copied, read or embedded; only the version-resource words are typed.
VENDOR_DLSS = {"CompanyName": "NVIDIA Corporation", "ProductName": "NVIDIA Deep Learning SuperSampling", "FileDescription": "synthetic DLSS stand-in"}  # nvngx copy


def vendor_copy(stray=b"payload bytes dXvK vKd3D by chance"):
    """The DLSS DLL copied as nvngx.dll by an older OptiScaler layout: its vendor's version resource, and a few bytes that
    happen to spell "dxvk" / "vkd3d" (a large binary does by chance); never DXVK."""
    return mockpe.make_dll(VENDOR_DLSS, version=(310, 5, 3, 0), extra=stray)


def with_certificate(data, signer):
    """`data` (a mockpe DLL) with a certificate table (PE data directory 4, a file offset) whose text names `signer`."""
    off = len(data) + (-len(data) % 8)
    blob = signer.encode() + b"\0" * 8
    cert = struct.pack("<IHH", 8 + len(blob), 0x0200, 2) + blob
    out = bytearray(data + b"\0" * (off - len(data)) + cert)
    struct.pack_into("<II", out, 0x40 + 4 + 20 + 112 + 4 * 8, off, len(cert))  # dos stub, PE, COFF header, optional header's 5th directory
    return bytes(out)


def vendor_copy_signed_only():
    """A DLSS stand-in without a version resource, recognised by its signer's name (nvngx copies are signed)."""
    return with_certificate(mockpe.make_dll(None, extra=b"no resource"), "NVIDIA Corporation")  # synthetic DLSS signer text


def optiscaler_mentions_dxvk():
    """An OptiScaler build without a version resource whose text names DXVK (it detects DXVK) and DXVK's settings."""
    return mockpe.make_dll(None, extra=b"synthetic OptiScaler proxy; checks DXVK: DXVK_CONFIG_FILE dxvk.conf DXVK DXVK")


def stray_dxvk_text():
    """A DLL under a DXVK name whose only "dxvk" is four stray letters (no DXVK settings name, no version resource)."""
    return mockpe.make_dll(None, extra=b"data dxvk data")
