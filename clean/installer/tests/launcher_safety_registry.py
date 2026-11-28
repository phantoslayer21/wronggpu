# SPDX-License-Identifier: GPL-3.0-only
"""Inspect only the fixture launcher registry file. The user's registry is never opened."""
import ctypes
import os
from contextlib import contextmanager
from ctypes import wintypes

if os.name == "nt":
    import winreg

@contextmanager
def open_hive(data):
    load = ctypes.WinDLL("advapi32").RegLoadAppKeyW
    load.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.HKEY), wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
    load.restype = wintypes.LONG
    handle = wintypes.HKEY()
    rc = load(os.path.join(data, "launcher-registry.hiv"), ctypes.byref(handle), winreg.KEY_READ | winreg.KEY_WRITE, 0, 0)
    if rc:
        raise ctypes.WinError(rc)
    try:
        yield handle.value
    finally:
        winreg.CloseKey(handle.value)

def relative(key):
    prefix, child = key.split("\\", 1)
    if prefix.upper() != "HKCU":
        raise ValueError(key)
    return child

def entries(data, key):
    if not os.path.isfile(os.path.join(data, "launcher-registry.hiv")):
        return {}
    with open_hive(data) as hive:
        try:
            branch = winreg.OpenKey(hive, relative(key))
        except FileNotFoundError:
            return {}
        with branch:
            out = {}
            for i in range(winreg.QueryInfoKey(branch)[0]):
                name = winreg.EnumKey(branch, i)
                with winreg.OpenKey(branch, name) as entry:
                    values = [winreg.EnumValue(entry, j) for j in range(winreg.QueryInfoKey(entry)[1])]
                    out[name] = {k: str(v) for k, v, _ in values}
            return out

def set_location(data, key, folder):
    with open_hive(data) as hive:
        with winreg.CreateKey(hive, relative(key)) as entry:
            winreg.SetValueEx(entry, "InstallLocation", 0, winreg.REG_SZ, folder)

def delete(data, key):
    with open_hive(data) as hive:
        try:
            winreg.DeleteKey(hive, relative(key))
        except FileNotFoundError:
            pass
