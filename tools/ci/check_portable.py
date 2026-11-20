# SPDX-License-Identifier: MIT
"""Portability check (CI and ctest on Linux): keeps the launcher building on Linux when someone edits it on Windows.

The launcher's shared code (src/wg_core, src/app) is compiled on Windows and on Linux. The Linux build answers Windows calls
with src/platform/posix; two mistakes get past a Windows-only build and break Linux later. This script finds both by reading
the source, without a compiler:

  (a) a std::filesystem::path made straight from a wstring in src/wg_core or src/app (constructed, copy-initialised, or a
      `p = w;` assignment to a path of the same function). On Linux that spells a Windows path
      ("C:\\Games", backslashes) wrongly; wg::FsPath(wstring) (src/wg_core/util.h) converts it for the host.
  (b) a shared UI file (src/app/*.cpp and *.h, except WINDOWS_ONLY below) that uses Direct2D, DirectWrite, WIC, DWM or UI
      Automation outside an `#ifdef _WIN32` block. Those are Windows' own; the shared code reaches the screen through Gfx
      (ui_gfx.cpp on Windows, ui_gfx_cairo.cpp on Linux) and keeps the rest inside `#ifdef _WIN32`.

    check_portable.py [--root <clean/installer>]    check the tree (exit 1 on a finding)
    check_portable.py --self-test                   plant a violation of each rule, expect red; the clean sample, expect green
"""
import argparse
import re
import sys
from pathlib import Path

# src/app files that exist only on Windows (the Direct2D Gfx, the GDI+ wizard, the per-game restorer): not part of the Linux build.
WINDOWS_ONLY = {"ui_gfx.cpp", "wizard.cpp", "restore_main.cpp"}

# Windows identifiers the shared UI must keep inside `#ifdef _WIN32`.
WINDOWS_API = re.compile(
    r"\b(?:ID2D1\w*|D2D1_\w+|D2D1\w*Factory\w*|IDWrite\w*|DWRITE_\w+|DWriteCreateFactory|IWIC\w+|CLSID_WIC\w+|GUID_WIC\w+"
    r"|WIC[A-Z]\w+|Dwm[A-Z]\w+|DWMWA_\w+|DWMWCP_\w+|DWM_\w+|IUIAutomation\w*|UIA_\w+|Uia(?:ReturnRawElementProvider|HostProviderFromHwnd|Raise\w+|ClientsAreListening|Disconnect\w+)|IRawElementProvider\w+)\b"
    r"|#\s*include\s*[<\"](?:d2d1\w*|dwrite\w*|wincodec\w*|dwmapi|uiautomation\w*)\.h[>\"]")

# a path constructed in place: `std::filesystem::path(x)`, `fs::path{x}` or `fs::path p(x)`; a declaration without an
# initializer, a reference or a pointer is not a construction
NS_ALIAS = re.compile(r"\bnamespace\s+(\w+)\s*=\s*(?:std::)?filesystem\s*;")
# the arguments of a function declaration (`path Name(const wstring& p)`), not of a construction (`path p(w)`)
DECLARATION_ARGS = re.compile(r"^\s*(?:const\s+)?[\w:<>]+(?:\s*[&*]+\s*|\s+)\w*\s*(?:,.*)?$", re.S)
NARROW_LITERAL = re.compile(r"""^\s*(?:u8)?"(?:[^"\\]|\\.)*"\s*$""")


def strip_comments_and_strings(text):
    """The text with comments removed and string contents blanked (lines stay where they are). #include lines keep their
    header name, which is what the Windows-API check reads."""
    out, i, n = [], 0, len(text)
    line_start = True
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:j]))
            i = j
            continue
        if c in "\"'":
            if line_start is False or not re.match(r"[ \t]*#[ \t]*include", text[text.rfind("\n", 0, i) + 1:i]):
                j = i + 1
                while j < n and text[j] != c and text[j] != "\n":
                    j += 2 if text[j] == "\\" else 1
                out.append(c + " " * max(0, j - i - 1) + (c if j < n and text[j] == c else ""))
                i = j + 1
                continue
        out.append(c)
        line_start = c == "\n"
        i += 1
    return "".join(out)


def windows_only_lines(code):
    """The set of line numbers (1-based) that only a Windows build compiles: inside `#ifdef _WIN32`, `#if defined(_WIN32)` or
    the #else of `#ifndef _WIN32`. Anything else (#if of other macros, #elif) counts as both."""
    stack, lines = [], set()
    for no, line in enumerate(code.split("\n"), 1):
        m = re.match(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", line)
        if m:
            d, rest = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef", "if"):
                win = re.search(r"\b_WIN32\b", rest) is not None
                neg = d == "ifndef" or re.search(r"!\s*defined\s*\(?\s*_WIN32\b", rest) is not None or re.match(r"\s*!\s*_WIN32\b", rest)
                stack.append("nonwin" if win and neg else "win" if win else "any")
            elif d == "else" and stack:
                stack[-1] = {"win": "nonwin", "nonwin": "win"}.get(stack[-1], "any")
            elif d == "elif" and stack:
                stack[-1] = "any"
            elif d == "endif" and stack:
                stack.pop()
            if "win" in stack:
                lines.add(no)
            continue
        if "win" in stack:
            lines.add(no)
    return lines


def check_fs_path(rel, text):
    """(a): findings for one file of src/wg_core or src/app."""
    code = strip_comments_and_strings(text)
    win = windows_only_lines(code)
    names = {"filesystem"} | set(NS_ALIAS.findall(code))
    pat = re.compile(r"\b(?:%s)::path\b\s*(?:(\w+)\s*)?([({])" % "|".join(re.escape(n) for n in names))
    found = []
    for m in pat.finditer(code):
        close = {"(": ")", "{": "}"}[m.group(2)]
        depth, j = 1, m.end()
        while j < len(code) and depth:
            depth += code[j] == m.group(2)
            depth -= code[j] == close
            j += 1
        arg = code[m.end():j - 1]
        line = code.count("\n", 0, m.start()) + 1
        if not arg.strip() or NARROW_LITERAL.match(arg) or line in win or (m.group(1) and DECLARATION_ARGS.match(arg)):
            continue
        found.append("%s:%d: std::filesystem::path built directly (%s); use wg::FsPath(wstring) from wg_core/util.h"
                     % (rel, line, " ".join(code[m.start():j].split())[:80]))
    # copy-initialisation compiles from a wstring too: `fs::path p = w;` (a narrow literal or wg::FsPath on the right is fine)
    init = re.compile(r"\b(?:%s)::path\s+\w+\s*=\s*([^;]+);" % "|".join(re.escape(n) for n in names))
    for m in init.finditer(code):
        rhs = m.group(1).strip()
        line = code.count("\n", 0, m.start()) + 1
        if NARROW_LITERAL.match(rhs) or rhs.startswith(("FsPath(", "wg::FsPath(")) or line in win:
            continue
        found.append("%s:%d: std::filesystem::path copy-initialised (%s); use wg::FsPath(wstring) from wg_core/util.h"
                     % (rel, line, " ".join(code[m.start():m.end()].split())[:80]))
    # assignment: `fs::path p; ... p = w;` with w a wstring of the same function (a function ends at a `}` in column 0), or a
    # wide literal. Only a plain wstring or wide literal on the right counts, so another path or a call never flags.
    assign = re.compile(r"(?m)^[ \t]*(\w+)\s*=\s*(L\"[^;]*|\w+)\s*;")
    start = 0
    for body in re.split(r"(?m)^\}", code):
        paths = set(re.findall(r"\b(?:%s)::path\s+(\w+)\s*[;=({]" % "|".join(re.escape(n) for n in names), body))
        wides = set(re.findall(r"\b(?:std::)?wstring(?:_view)?\s*[&*]*\s*(\w+)", body)) | set(re.findall(r"\bwchar_t\s*\*?\s*(\w+)", body))
        for m in assign.finditer(body):
            rhs = m.group(2)
            line = code.count("\n", 0, start + m.start()) + 1
            if m.group(1) in paths and (rhs.startswith('L"') or rhs in wides) and line not in win:
                found.append("%s:%d: std::filesystem::path assigned from a wide string (%s); use wg::FsPath(wstring) from wg_core/util.h"
                             % (rel, line, " ".join(m.group(0).split())[:80]))
        start += len(body) + 1
    return found


def check_windows_api(rel, text, standins=frozenset()):
    """(b): findings for one shared UI file. `standins` are the names and headers src/platform/posix provides (the Linux build
    compiles a use of those); a name the file defines itself (a Linux stub struct) is its own, not Windows'."""
    code = strip_comments_and_strings(text)
    win = windows_only_lines(code)
    own = set(re.findall(r"\b(?:struct|class|enum|using|typedef)\s+(\w+)", code))
    found = []
    for no, line in enumerate(code.split("\n"), 1):
        for m in WINDOWS_API.finditer(line):
            name = m.group(0).strip()
            header = re.search(r"[<\"](\w+\.h)[>\"]", name)
            if no in win or name in own or (header.group(1) if header else name) in standins:
                continue
            found.append("%s:%d: %s is Windows-only; put it inside #ifdef _WIN32 or behind Gfx, or add a stand-in in "
                         "src/platform/posix" % (rel, no, name))
    return found


def standins_of(root):
    """What src/platform/posix provides: its header names and every word its headers declare."""
    names = set()
    for h in (root / "src/platform/posix").glob("*.h"):
        names.add(h.name)
        names.update(re.findall(r"\w+", strip_comments_and_strings(h.read_text(encoding="utf-8", errors="replace"))))
    return names


def check_tree(root):
    found = []
    for sub in ("src/wg_core", "src/app"):
        for f in sorted((root / sub).glob("*")):
            if f.suffix in (".cpp", ".h"):
                found += check_fs_path("%s/%s" % (sub, f.name), f.read_text(encoding="utf-8", errors="replace"))
    standins = standins_of(root)
    for f in sorted((root / "src/app").glob("*")):
        if f.suffix in (".cpp", ".h") and f.name not in WINDOWS_ONLY:
            found += check_windows_api("src/app/" + f.name, f.read_text(encoding="utf-8", errors="replace"), standins)
    return found


CLEAN = """\
#include <filesystem>
namespace fs = std::filesystem;
#ifdef _WIN32
#include <d2d1.h>
#include <dwmapi.h>
inline std::filesystem::path Direct(const std::wstring& p) { return std::filesystem::path(p); }  // Windows spells it as is
ID2D1Factory* f = nullptr;
#else
std::filesystem::path FsPath(const std::wstring& p);
#endif
// std::filesystem::path(wide) and ID2D1Factory in a comment, "fs::path(x)" and "DwmSetWindowAttribute" in a string
const char* s = "fs::path(x) DwmSetWindowAttribute";
void F(const fs::path& a, const std::filesystem::path* b) { std::filesystem::path none; auto c = fs::path("narrow"); fs::path d; }
void G(const std::wstring& w) { auto p = wg::FsPath(w); (void)p; std::filesystem::path q = wg::FsPath(w); fs::path r = "narrow"; }
void H(const std::wstring& w, const fs::path& other) {
    fs::path a;
    a = wg::FsPath(w);
    a = other;
    std::wstring s2 = w;
    s2 = w;
}
std::filesystem::path FsPath(const std::wstring& p);
#ifndef _WIN32
struct IWICImagingFactory { void Release() {} };  // the Linux branch's own stub of a name Windows has
#endif
#include <dwmapi.h>
HRESULT r = DwmSetWindowAttribute(h, 13, &v, sizeof v);  // posix/dwmapi.h stands in for it
"""
CLEAN_STANDINS = {"dwmapi.h", "DwmSetWindowAttribute"}

PLANTED = {
    "(a) path from a wstring, fully qualified": ("src/wg_core/x.cpp", "void F(const std::wstring& w) { auto p = std::filesystem::path(w); }\n", check_fs_path),
    "(a) path from a wstring, through an alias": ("src/app/x.cpp", "namespace fs = std::filesystem;\nvoid F(const std::wstring& w) { fs::path p(w); }\n", check_fs_path),
    "(a) path from a wstring, in the Linux branch": ("src/app/x.cpp", "#ifdef _WIN32\n#else\nvoid F(const std::wstring& w) { std::filesystem::path p{w}; }\n#endif\n", check_fs_path),
    "(a) path copy-initialised from a wstring": ("src/wg_core/x.cpp", "void F(const std::wstring& w) { std::filesystem::path p = w; }\n", check_fs_path),
    "(a) path assigned from a wstring": ("src/app/x.cpp", "void F(const std::wstring& w) {\n    std::filesystem::path p;\n    p = w;\n}\n", check_fs_path),
    "(a) path assigned from a wide literal": ("src/wg_core/x.cpp", "namespace fs = std::filesystem;\nvoid F() {\n    fs::path p;\n    p = L\"C:\\\\Games\";\n}\n", check_fs_path),
    "(b) Direct2D outside #ifdef _WIN32": ("src/app/ui_x.cpp", "ID2D1Factory* g = nullptr;\n", check_windows_api),
    "(b) DWM call in the Linux branch": ("src/app/ui_x.cpp", "#ifndef _WIN32\nvoid F(HWND h) { DwmSetWindowAttribute(h, 13, 0, 0); }\n#endif\n", check_windows_api),
    "(b) UI Automation header": ("src/app/ui_x.cpp", "#include <uiautomation.h>\n", check_windows_api),
    "(b) DWM call without a stand-in": ("src/app/ui_x.cpp", "void F(HWND h) { DwmExtendFrameIntoClientArea(h, 0); }\n", check_windows_api),
    "(b) WIC": ("src/app/ui_x.cpp", "void F() { IWICImagingFactory* w = nullptr; }\n", check_windows_api),
}


def self_test():
    ok = True
    for name, (rel, text, fn) in PLANTED.items():
        hits = fn(rel, text)
        print("%s %s" % ("red  (found)  " if hits else "FAIL (not found)", name))
        ok &= bool(hits)
    for fn in (check_fs_path, check_windows_api):
        hits = fn("src/app/x.cpp", CLEAN) if fn is check_fs_path else fn("src/app/x.cpp", CLEAN, CLEAN_STANDINS)
        print("%s clean sample, %s" % ("green" if not hits else "FAIL", fn.__name__))
        ok &= not hits
        for h in hits:
            print("   ", h)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "clean" / "installer",
                    help="the installer folder (default: clean/installer of this checkout)")
    ap.add_argument("--self-test", action="store_true", help="prove each rule fails on a planted violation and passes on a clean sample")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    found = check_tree(a.root)
    for f in found:
        print(f)
    print("portable: %d finding(s)" % len(found))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
