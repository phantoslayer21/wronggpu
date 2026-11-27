# SPDX-License-Identifier: GPL-3.0-only
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


def resolve_wine(wine):
    """Absolute path of a wine executable given as a path or a name on PATH (children run from other folders)."""
    found = shutil.which(str(wine))
    if not found:
        raise SystemExit(f"--wine {wine}: not an executable")
    return Path(os.path.abspath(found))


def find_wineserver(wine):
    """The wineserver that goes with `wine`: beside what it links to (the real install), else beside the link, else the
    one on PATH. A system Wine can be a link in a user bin folder whose neighbour is some other wineserver."""
    for folder in (Path(os.path.realpath(wine)).parent, wine.parent):
        cand = folder / "wineserver"
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    found = shutil.which("wineserver")
    if found:
        return Path(os.path.abspath(found))
    raise SystemExit(f"no wineserver found beside {wine} or on PATH; a wine needs its wineserver")


def prepare(proton_dir, prefix_dir, vkd3d_dir=None, log=print, wine=None):
    """Create (once) the prefix with the selected wine, copy the Proton install's DLLs into its system32 (every call, so
    a changed vkd3d_dir takes effect) and return the environment: WINEPREFIX, WINEDLLOVERRIDES, WINEDEBUG, PATH with the
    selected wine's folder first, WG_WINE (its absolute path) and WG_WINESERVER (its wineserver, also given to wine as
    WINESERVER; the runner's clean-up of a hung run uses it).

    wine: the wine executable (a system Wine, say) to create and run the prefix with. Default: Proton's own
    (<proton_dir>/files/bin/wine). Proton's own wine can hang headless (see run_wine_tests.py's probe)."""
    proton_dir = Path(proton_dir).resolve()
    prefix_dir = Path(prefix_dir).resolve()
    own = wine is None
    wine = proton_dir / "files" / "bin" / "wine" if own else resolve_wine(wine)
    if not wine.is_file():
        raise SystemExit(f"{wine}: no wine here; is {proton_dir} a Proton install?")
    wineserver = find_wineserver(wine)
    dlls = proton_dlls(proton_dir)
    if vkd3d_dir:
        for name in ("d3d12.dll", "d3d12core.dll"):
            p = Path(vkd3d_dir) / name
            if not p.is_file():
                raise SystemExit(f"--vkd3d {vkd3d_dir}: {name} missing")
            dlls[name] = p
    env = dict(os.environ)
    env.update(WINEPREFIX=str(prefix_dir), WINEDLLOVERRIDES=DLL_OVERRIDES, WINEDEBUG="-all",
               PATH=f"{wine.parent}{os.pathsep}{env.get('PATH', '')}", WG_WINE=str(wine),
               WG_WINESERVER=str(wineserver), WINESERVER=str(wineserver))
    if not (prefix_dir / "system.reg").is_file():
        log(f"creating Wine prefix {prefix_dir} with {proton_dir.name if own else wine}")
        prefix_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(wine), "wineboot", "-i"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        subprocess.run([str(wineserver), "-w"], env=env, check=True)
    system32 = prefix_dir / "drive_c" / "windows" / "system32"
    for name, src in dlls.items():
        shutil.copyfile(src, system32 / name)
    log(f"prefix DLLs from {proton_dir.name}" + (f"; d3d12 / d3d12core from {vkd3d_dir}" if vkd3d_dir else ""))
    return env


def windows_path(path):
    """Wine's Z: path of a Linux path (Z: maps the Linux root)."""
    return "Z:" + str(Path(path).resolve()).replace("/", "\\")
