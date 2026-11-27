# SPDX-License-Identifier: GPL-3.0-only
"""Model L comparison records (issue #89): every record a Model L check writes against the oracle names the FP8
matrix-op model the oracle used (clean/l/reference/table.py "mm8", oracle.MM8_MODELS), also when it is the default.
Every such writer goes through write_l_record, which adds the effective model and refuses a record that names none or
names one its frames contradict.

    write_l_record(res, path)                 # the default table's model
    write_l_record(res, path, table=t)        # the model of the table the oracle ran with
"""
import json


def effective_mm8(table=None):
    """The FP8 op model the oracle uses with `table` (default: the shipped table), also when the table leaves it out."""
    from clean.l.reference import table as T
    return (T.TABLE if table is None else table).get("mm8", "exact")


def check_l_record(res, frames_required=False):
    """Raises ValueError unless `res` names a known FP8 op model; every frame that names one (all frames when
    frames_required) must name the same."""
    from clean.l.reference import oracle as L
    model = res.get("fp8_op_model")
    if model not in L.MM8_MODELS:
        raise ValueError("Model L comparison record without its FP8 op model (fp8_op_model)")
    for m in res.get("frames", []):
        if (frames_required or "fp8_op_model" in m) and m.get("fp8_op_model") != model:
            raise ValueError(f"frame {m.get('frame')}: FP8 op model missing or different")
    return res


def write_l_record(res, path, table=None):
    """Adds the effective FP8 op model to the record (top level), checks it and writes the JSON file."""
    res["fp8_op_model"] = effective_mm8(table)
    check_l_record(res)
    with open(path, "w", newline="\n") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")
    return res
