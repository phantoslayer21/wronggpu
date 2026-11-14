# SPDX-License-Identifier: MIT
"""A Wine prefix set up the way Proton runs a game, for WrongGPU's Linux tests (clean/linux/run_wine_tests.py).

Proton puts vkd3d-proton (d3d12.dll, d3d12core.dll), DXVK's dxgi.dll and Wine's vkd3d libraries (vkd3d-utils,
vkd3d, vkd3d-shader: what Wine's builtin d3dcompiler_47 compiles with, the compiler the interop shaders meet in a game)
into a game's prefix. A prefix made by Proton's wine
alone has none of them: no D3D12 device ("SKIP: no D3D12 device") and no d3dcompiler_47. This module copies them from
a Proton install into a test prefix and returns the environment to run test programs there.

    env = wine_env.prepare(proton_dir, prefix_dir, vkd3d_dir=None)
    subprocess.run([env["WG_WINE"], "test.exe"], env=env)

vkd3d_dir: a folder with a self-built vkd3d-proton (d3d12.dll, d3d12core.dll) used instead of Proton's own. WrongGPU's
D3D12 kernels need vkd3d-proton with dxil-spirv's support for shader-packed WMMA fragments
(https://github.com/HansKristian-Work/dxil-spirv/pull/362) until a Proton release ships it.
"""
import os
import shutil
import subprocess
from pathlib import Path

# (file, a folder name its path must contain); searched below <proton>/files/lib*/ so that Proton's own layout changes
# (lib, lib64, lib/wine/...) do not need a code change
# Wine's builtin d3dcompiler_47 compiles through vkd3d-utils, which needs vkd3d and vkd3d-shader.
_PROTON_FILES = (("d3d12.dll", "vkd3d-proton"), ("d3d12core.dll", "vkd3d-proton"), ("dxgi.dll", "dxvk"),
                 ("libvkd3d-utils-1.dll", "vkd3d"), ("libvkd3d-1.dll", "vkd3d"), ("libvkd3d-shader-1.dll", "vkd3d"))
# mscoree / mshtml off and winemenubuilder disabled: a new prefix's wineboot shows no Mono / Gecko dialog and
# writes no menu entries under ~/.local/share
DLL_OVERRIDES = "d3d12,d3d12core,dxgi=n,b;mscoree,mshtml=;winemenubuilder.exe=d"


def _find(files_dir, name, folder):
    for p in sorted(files_dir.glob(f"lib*/**/x86_64-windows/{name}")):
        if folder in p.parts and not (folder == "vkd3d" and "vkd3d-proton" in p.parts):
            return p
    return None


def proton_dlls(proton_dir):
    """{file name: path} of the DLLs a game prefix gets from this Proton install; raises if one is missing."""
    files_dir = Path(proton_dir) / "files"
    found = {}
    for name, folder in _PROTON_FILES:
        p = _find(files_dir, name, folder)
        if not p:
            raise SystemExit(f"{proton_dir}: no {folder}/.../x86_64-windows/{name}; is this a Proton install?")
        found[name] = p
    return found


def prepare(proton_dir, prefix_dir, vkd3d_dir=None, log=print):
    """Create (once) the prefix with Proton's wine, copy the DLLs into its system32 (every call, so a changed vkd3d_dir
    takes effect) and return the environment: WINEPREFIX, WINEDLLOVERRIDES, WINEDEBUG, PATH with Proton's bin first,
    and WG_WINE (Proton's wine executable)."""
    proton_dir = Path(proton_dir).resolve()
    prefix_dir = Path(prefix_dir).resolve()
    wine = proton_dir / "files" / "bin" / "wine"
    if not wine.is_file():
        raise SystemExit(f"{proton_dir}: no files/bin/wine; is this a Proton install?")
    dlls = proton_dlls(proton_dir)
    if vkd3d_dir:
        for name in ("d3d12.dll", "d3d12core.dll"):
            p = Path(vkd3d_dir) / name
            if not p.is_file():
                raise SystemExit(f"--vkd3d {vkd3d_dir}: {name} missing")
            dlls[name] = p
    env = dict(os.environ)
    env.update(WINEPREFIX=str(prefix_dir), WINEDLLOVERRIDES=DLL_OVERRIDES, WINEDEBUG="-all",
               PATH=f"{wine.parent}{os.pathsep}{env.get('PATH', '')}", WG_WINE=str(wine))
    if not (prefix_dir / "system.reg").is_file():
        log(f"creating Wine prefix {prefix_dir} with {proton_dir.name}")
        prefix_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(wine), "wineboot", "-i"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        subprocess.run([str(wine.parent / "wineserver"), "-w"], env=env, check=True)
    system32 = prefix_dir / "drive_c" / "windows" / "system32"
    for name, src in dlls.items():
        shutil.copyfile(src, system32 / name)
    log(f"prefix DLLs from {proton_dir.name}" + (f"; d3d12 / d3d12core from {vkd3d_dir}" if vkd3d_dir else ""))
    return env


def windows_path(path):
    """Wine's Z: path of a Linux path (Z: maps the Linux root)."""
    return "Z:" + str(Path(path).resolve()).replace("/", "\\")
