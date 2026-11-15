# SPDX-License-Identifier: MIT
"""The HIP SDK policy of every build and HIP process of this project, enforced at one boundary:

- the SDK is an official AMD HIP SDK install: C:\\Program Files\\AMD\\ROCm\\<version> on Windows, /opt/rocm-<version> or
  /opt/rocm on other systems, with a compatible version (major SUPPORTED_MAJOR); without --rocm the newest one
  (numeric version order). Never an SDK from the environment (ROCM_PATH, HIP_PATH, ...) or from PATH;
- the compiler is the selected SDK's own (a bare name like "clang" means the SDK's bin\\clang.exe; an absolute path must
  lie inside the SDK);
- anything else needs a deliberate override (--force-sdk), which approves exactly that SDK folder or compiler for this
  process (approve);
- the HIP runtime library is loaded from the selected SDK only (runtime_library), never by a bare name the loader would
  search for;
- logs name official folders as they are and custom ones only as "custom SDK (forced, id <hash>)" (label, redact).

device_command / find_clang (clean/runner/build.py) and the HIP runtime loader (clean/runner/hip.py) call check():
every entry point that compiles HIP code or loads the HIP runtime passes it, whichever script started it."""
import hashlib
import os
import re
import sys

SUPPORTED_MAJOR = 7
WINDOWS_ROOT = r"C:\Program Files\AMD\ROCm"
LINUX_ROOT = "/opt"
FALLBACK = os.path.join(WINDOWS_ROOT, "7.2") if sys.platform == "win32" else "/opt/rocm"
_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")

_approved = set()   # normalized folders / compilers approved with --force-sdk in this process
_bannered = set()


def _norm(path):
    return os.path.normcase(os.path.normpath(os.path.realpath(os.path.abspath(path))))


def _exe(name):
    return name + ".exe" if sys.platform == "win32" and not name.lower().endswith(".exe") else name


def _valid_sdk(path):
    return (os.path.isfile(os.path.join(path, "bin", _exe("clang"))) and
            os.path.isdir(os.path.join(path, "amdgcn", "bitcode")))


def official_installs(windows_root=None, linux_root=None, windows=None):
    """[(version tuple, folder)] of the official installs, newest first (numeric order); only folders with the SDK's
    clang and device libraries and a compatible version."""
    windows = sys.platform == "win32" if windows is None else windows
    out = []
    if windows:
        root = windows_root or WINDOWS_ROOT
        names = [(n, os.path.join(root, n)) for n in (os.listdir(root) if os.path.isdir(root) else [])]
    else:
        root = linux_root or LINUX_ROOT
        names = [(n[len("rocm-"):], os.path.join(root, n)) for n in (os.listdir(root) if os.path.isdir(root) else [])
                 if n.startswith("rocm-")]
    for ver, path in names:
        m = _VERSION.match(ver)
        if m and int(m.group(1)) == SUPPORTED_MAJOR and _valid_sdk(path):
            out.append((tuple(int(g or 0) for g in m.groups()), path))
    out.sort(reverse=True)
    if not windows:   # /opt/rocm: official when it is (or links to) one of them
        link = os.path.join(root, "rocm")
        if os.path.isdir(link) and any(_norm(link) == _norm(p) for _, p in out):
            out.insert(0, (out[0][0], link))
    return out


def default_rocm():
    """The newest official compatible install, or None."""
    found = official_installs()
    return found[0][1] if found else None


OFFICIAL_ROCM = default_rocm() or FALLBACK   # for scripts that need one folder name (the default of their --rocm)


def path_id(path):
    return hashlib.sha256(_norm(path).encode("utf-8")).hexdigest()[:10]


def is_official(path):
    return bool(path) and any(_norm(path) == _norm(p) for _, p in official_installs())


def label(path):
    """How a log names an SDK or compiler folder: official ones as they are, others by an id only."""
    if not path:
        return "none"
    if is_official(path) or any(_inside(path, p) for _, p in official_installs()):
        return path
    return f"custom (forced, id {path_id(path)})"


def approve(path):
    """--force-sdk: accept this exact SDK folder or compiler in this process."""
    _approved.add(_norm(path))


def _inside(path, root):
    try:
        a, r = _norm(path), _norm(root)
        return os.path.commonpath([a, r]) == r
    except ValueError:
        return False


def resolve(rocm=None, force=False):
    """The SDK folder for a --rocm value: empty or None = the newest official install; another folder only with
    force (then approved for this process)."""
    if not rocm:
        d = default_rocm()
        if not d:
            raise SystemExit(f"no official HIP SDK {SUPPORTED_MAJOR}.x found ({WINDOWS_ROOT}\\<version> or "
                             f"/opt/rocm-<version>)")
        return d
    if is_official(rocm):
        return rocm
    if force:
        approve(rocm)
        return rocm
    raise SystemExit(f"--rocm names a folder that is not an official HIP SDK {SUPPORTED_MAJOR}.x install (id "
                     f"{path_id(rocm)}); pass --force-sdk to build with it anyway")


def check_rocm(rocm):
    """The boundary for an SDK folder: official, or approved with --force-sdk."""
    if not rocm or not (is_official(rocm) or _norm(rocm) in _approved):
        raise SystemExit(f"HIP SDK folder {label(rocm) if rocm else '(none)'} is neither an official install nor "
                         "approved with --force-sdk")
    return rocm


def compiler(rocm, name_or_path=None, force=False, default="clang"):
    """The canonical compiler for an SDK: None = the SDK's own `default`; a bare name = that file in the SDK's bin
    folder; a path = itself, when inside the SDK. Anything else only with force (then approved)."""
    check_rocm(rocm)
    if not name_or_path:
        name_or_path = default
    bare = os.path.basename(name_or_path) == name_or_path
    if bare:
        for sub in ("bin", os.path.join("lib", "llvm", "bin"), os.path.join("llvm", "bin")):
            p = os.path.join(rocm, sub, _exe(name_or_path))
            if os.path.isfile(p):
                return p
        if force:   # the deliberate override: whatever that name is, approved as such
            import shutil
            p = shutil.which(name_or_path) or name_or_path
            approve(p)
            return p
        raise SystemExit(f"compiler {name_or_path} not found in the HIP SDK {label(rocm)} (another one needs "
                         "--force-sdk and a path)")
    if _inside(name_or_path, rocm) and os.path.isfile(name_or_path):
        return name_or_path
    if force:
        approve(name_or_path)
        return name_or_path
    raise SystemExit(f"compiler {label(name_or_path)} is not inside the HIP SDK {label(rocm)}; pass --force-sdk to "
                     "use it anyway")


def check(rocm, clang):
    """The build boundary (device_command): the SDK folder and the compiler pass the policy; prints the banner once per
    toolchain."""
    check_rocm(rocm)
    if not (_inside(clang, rocm) or _norm(clang) in _approved):
        raise SystemExit(f"compiler {label(clang)} is not inside the HIP SDK {label(rocm)} and not approved with "
                         "--force-sdk")
    key = (_norm(rocm), _norm(clang))
    if key not in _bannered:
        _bannered.add(key)
        print(banner(rocm, clang=clang), flush=True)


def runtime_library(rocm):
    """The HIP runtime library of the checked SDK (never a bare name for the loader to search); raises when missing."""
    check_rocm(rocm)
    if sys.platform == "win32":
        cands = [os.path.join(rocm, "bin", n) for n in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll")]
    else:
        lib = os.path.join(rocm, "lib")
        cands = [os.path.join(lib, "libamdhip64.so")] + sorted(
            os.path.join(lib, n) for n in (os.listdir(lib) if os.path.isdir(lib) else []) if n.startswith("libamdhip64.so."))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit(f"no HIP runtime library in the HIP SDK {label(rocm)}")


def device_lib_dir(rocm):
    return os.path.join(rocm, "amdgcn", "bitcode")


def add_arguments(ap, help_suffix=""):
    """--rocm (default: the newest official install, never an environment variable) and --force-sdk."""
    ap.add_argument("--rocm", default="", help="HIP SDK folder (default: the newest official install; another "
                                               f"folder needs --force-sdk){help_suffix}")
    ap.add_argument("--force-sdk", action="store_true",
                    help="accept a --rocm or compiler outside the official HIP SDK installs")


def banner(rocm, clang=None, cxx=None, device_libs=True):
    """One line naming the SDK, the compilers and the device-library folder of a build (custom ones by id only)."""
    parts = [f"HIP SDK: {label(rocm)}" + (" (official)" if is_official(rocm) else "")]
    if clang:
        parts.append(f"clang: {label(clang)}")
    if cxx:
        parts.append(f"host C++: {label(cxx) if _inside(cxx, rocm) or _norm(cxx) in _approved else cxx}")
    parts.append(f"device libs: {label(device_lib_dir(rocm)) if device_libs else 'none (-nogpulib)'}")
    return "; ".join(parts)


def redact(text):
    """text with every approved custom folder or compiler replaced by its label (both slash forms)."""
    for p in sorted(_approved, key=len, reverse=True):
        lab = f"custom (forced, id {hashlib.sha256(p.encode('utf-8')).hexdigest()[:10]})"
        for form in {p, p.replace("\\", "/"), p.replace("/", "\\")}:
            text = re.sub(re.escape(form), lab, text, flags=re.IGNORECASE)
    return text
