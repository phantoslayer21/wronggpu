# SPDX-License-Identifier: GPL-3.0-only
"""The synthetic-values rule for the Model L checkers' real modes (l_hyb_stage_check --real-inputs, l_q7_check /
l_q7w_check --real, l_mm8_f32_check --real): every intermediate made from real weights or recorded frames (case files,
replay inputs and outputs) goes only to a private folder OUTSIDE the repository, resolved (real path) before anything is
read; a folder inside the repository is refused. The records these checkers write stay counts and digests, and a
real run's --json goes outside the repository too (private_json). A prepared cases file made from real data gets a
provenance marker next to it (<file>.real, mark_real); every run path checks the marker and the private folder BEFORE
it opens the file (guard_cases).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def inside_repository(path):
    """Whether `path` (resolved) is the repository or inside it."""
    p = os.path.realpath(os.path.abspath(path))
    root = os.path.realpath(ROOT)
    try:
        return os.path.commonpath([p, root]) == root
    except ValueError:   # another drive
        return False


def private_dir(path, flag="--private-out", what="a real run's intermediates"):
    """The resolved private folder (made if missing); SystemExit when it is not given or is inside the repository."""
    if not path:
        raise SystemExit(f"{flag} is required: {what} go only to a folder outside the repository")
    if inside_repository(path):
        raise SystemExit(f"{flag} {path}: inside the repository ({os.path.realpath(ROOT)}); {what} go only to a "
                         "folder outside it")
    p = os.path.realpath(os.path.abspath(path))
    os.makedirs(p, exist_ok=True)
    return p


MARKER = ".real"


def mark_real(cases_path, what):
    """The provenance marker of a prepared cases file made from real data (written before the file itself)."""
    with open(cases_path + MARKER, "w", encoding="utf-8", newline="\n") as f:
        f.write("prepared from real data (%s): read only from its private folder outside the repository "
                "(--private-out)\n" % what)


def guard_cases(cases_path, private_out):
    """Before a prepared cases file is opened: a marked (real) one only from the given private folder, which is
    outside the repository; SystemExit otherwise. Unmarked files (synthetic) pass."""
    folder = os.path.dirname(os.path.realpath(os.path.abspath(cases_path)))
    if not os.path.exists(cases_path + MARKER):
        return
    if inside_repository(folder):
        raise SystemExit(f"{cases_path}: prepared from real data, inside the repository ({os.path.realpath(ROOT)}); "
                         "refused (it belongs in a private folder outside it)")
    if not private_out:
        raise SystemExit(f"{cases_path}: prepared from real data; read only with --private-out naming its folder")
    if os.path.realpath(os.path.abspath(private_out)) != folder:
        raise SystemExit(f"{cases_path}: prepared from real data; --private-out {private_out} is not its folder")


def private_json(path, flag="--json"):
    """A real run's record: SystemExit when it would be written inside the repository."""
    if path and inside_repository(os.path.dirname(os.path.abspath(path)) or "."):
        raise SystemExit(f"{flag} {path}: a real run's record goes outside the repository (or into the private "
                         "folder), never inside it")
