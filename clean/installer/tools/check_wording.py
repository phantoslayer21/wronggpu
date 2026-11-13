#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The launcher's words never claim WrongGPU's output is the same as the original (owner 2026-11-01, rr-notes.md: "It
CANNOT say bit for bit identical"). Every string literal of the launcher's sources (src/app, src/wg_core: the window, the
CLI's words, the notes and logs players read) is checked for the phrases below; modes are described only relative to each
other (Exact is the most precise, Accurate looks the same as Exact, Fast is the fastest). The other upscaler mod the
launcher detects by its file names is never named either (owner 2026-11-04): a literal that names it fails unless it
is exactly one of its detection keys (OTHER_MOD_KEYS).

    check_wording.py [<folder> ...]

Default folders: src/app and src/wg_core of the installer folder this script is in. Comments are not checked (they
describe the code to its readers, not the output to players). Exit 1 when a literal has one of the phrases. Written for
this project; Python standard library only.
"""
import os
import re
import sys

# lower case; a literal matches when its text, lower-cased, contains one
PHRASES = (
    "bit for bit", "bit-for-bit", "bit-exact", "bit exact", "identical", "same image as the original", "same as the original",
    "same as native", "matches native", "match native", "(native)",
)
SOURCES = (".cpp", ".h")
# Another upscaler mod is detected by its signature and its two file names but never named in what players read (owner
# 2026-11-04, Brief 17): a literal with one of these parts fails, unless it is exactly a detection key (or the tool key
# that install records before 0.6.0 rc3 hold, which is only parsed).
def _other_mod():
    """(detection keys, words) of the other upscaler mod from their one source line (src/wg_core/constants.h kOtherMods)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "src", "wg_core", "constants.h"), encoding="utf-8") as f:
        m = re.search(r'\{L"Another upscaler mod", "([^"]+)", \{L"([^"]+)", L"([^"]+)"', f.read())
    sig = m.group(1)
    words = (sig, sig.replace("_", " "), sig.replace("_", "-"), sig[4:].replace("_", " "), sig[4:].replace("_", "-"))
    return (sig, m.group(2).lower(), m.group(3).lower(), "nr-on-amd"), words


OTHER_MOD_KEYS, OTHER_MOD = _other_mod()
# a C++ string literal (narrow, wide, UTF-8), escapes kept; raw strings are not used in these sources
LITERAL = re.compile(r'(?:u8|L|u|U)?"((?:[^"\\\n]|\\.)*)"')


def strip_comments(text):
    """The text with // and /* */ comments blanked (string and character literals kept as they are)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def check_text(text):
    """(line, phrase, literal) for every string literal of `text` with a phrase."""
    hits = []
    code = strip_comments(text)
    for m in LITERAL.finditer(code):
        low = m.group(1).lower()
        for p in PHRASES:
            if p in low:
                hits.append((code.count("\n", 0, m.start()) + 1, p, m.group(1)))
        if low not in OTHER_MOD_KEYS:
            for p in OTHER_MOD:
                if p in low:
                    hits.append((code.count("\n", 0, m.start()) + 1, "names the other upscaler mod (" + p + ")", m.group(1)))
                    break
    return hits


def self_test():
    """The check itself: it finds the phrases in literals, and leaves comments and other words alone."""
    bad = [
        'L"Default (native)"', 'L"Bit-exact with the original."', 'L"Matches native DLSS."', '"identical to the original"',
        'L"Looks bit for bit the same"', 'L"Another upscaler mod (%s)"' % OTHER_MOD[1], 'L"file %s"' % OTHER_MOD_KEYS[1],
        '"%s mod"' % OTHER_MOD[4],
    ]
    good = ['L"Default is WrongGPU\'s standard look. The others are its own variations."', 'L"Looks the same as Exact, and faster."',
            '// identical: a comment', '/* bit-exact */ L"Accurate"', 'L"%s"' % OTHER_MOD_KEYS[1], '"%s"' % OTHER_MOD_KEYS[0],
            '"nr-on-amd"', 'L"Another upscaler mod"', '// the %s mod: a comment' % OTHER_MOD[1]]
    ok = True
    for s in bad:
        if not check_text(s + ";\n"):
            print("wording: self-test: not found in %s" % s)
            ok = False
    for s in good:
        if check_text(s + ";\n"):
            print("wording: self-test: wrongly found in %s" % s)
            ok = False
    return ok


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folders = sys.argv[1:] or [os.path.join(here, "src", "app"), os.path.join(here, "src", "wg_core")]
    if not self_test():
        return 1
    hits, count = [], 0
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = sorted(dirnames)
            for fn in sorted(filenames):
                if not fn.lower().endswith(SOURCES):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                count += 1
                for line, phrase, lit in check_text(text):
                    hits.append("%s:%d: \"%s\" in \"%s\"" % (path, line, phrase, lit))
    for h in hits:
        print("wording: " + h)
    print("wording: %d files checked, %d literals claim the same output as the original or name the other upscaler mod" % (count, len(hits)))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
