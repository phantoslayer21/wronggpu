# SPDX-License-Identifier: MIT
"""Preset-F model admission.

- `model_file`: loading and checking a converted model file (`model.wgcm` of clean/converter, map version 5; pages
  56, 57.7). Standard library only.
- `admission`: the earlier synthetic known-subset metadata validator (needs NumPy); imported on first use so that
  model files load where NumPy is not installed.
"""
from .model_file import (ADMITTED, REJECTED, AdmittedModel, FileAdmission, ModelFileError, admit_model_file,
                         expected_inventory, read_model_file)

_LAZY = ("AdmissionIssue", "AdmissionResult", "validate_metadata")


def __getattr__(name):
    if name in _LAZY:
        from . import admission
        return getattr(admission, name)
    raise AttributeError(name)


__all__ = ["AdmissionIssue", "AdmissionResult", "validate_metadata", "ADMITTED", "REJECTED", "AdmittedModel",
           "FileAdmission", "ModelFileError", "admit_model_file", "expected_inventory", "read_model_file"]
