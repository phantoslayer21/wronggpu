# SPDX-License-Identifier: MIT
"""GPU family of a build target: which matrix design its kernels and weight packing assume.

Exact names only. LLVM files gfx1170-1172 under gfx11 although they carry the gfx12 matrix design, and gfx1102 / gfx1103
and the gfx115x parts are not tested, so each target is added here on purpose, never matched by prefix.
"""
import json
import os

FAMILIES = {"gfx1200": "gfx12", "gfx1201": "gfx12", "gfx1100": "gfx11", "gfx1101": "gfx11"}
# written by build.py into a build folder that is not the default full gfx12 build; the names of the families are the
# names of the accumulator row tables of packing.D_ROW_LAYOUTS
TARGET_FILE = "target.json"


def family(arch):
    """"gfx12" or "gfx11" for a supported target name; ValueError for any other."""
    if arch not in FAMILIES:
        raise ValueError(f"unsupported GPU target {arch!r}; supported: {', '.join(FAMILIES)}")
    return FAMILIES[arch]


def build_family(build_dir):
    """The family of the target a build folder was made for: its target.json, else gfx12 (the default build)."""
    path = os.path.join(build_dir, TARGET_FILE)
    if not os.path.exists(path):
        return "gfx12"
    with open(path) as f:
        return family(json.load(f)["arch"])
