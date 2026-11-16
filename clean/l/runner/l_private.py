# SPDX-License-Identifier: MIT
"""The synthetic-values rule for the Model L checkers' real modes (l_hyb_stage_check --real-inputs, l_q7_check /
l_q7w_check --real, l_mm8_f32_check --real): every intermediate made from real weights or recorded frames (case files,
replay inputs and outputs) goes only to a private folder OUTSIDE the repository, resolved (real path) before anything is
read; a folder inside the repository is refused. The records these checkers write stay counts and digests.
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
