# SPDX-License-Identifier: MIT
"""The launcher's side of the elevated step, for the Python suites that run the worker directly (elevated_job.h).

Written for this project; Python standard library only (ctypes, Windows). run_worker writes the job files and the batch
file, makes the reply pipe, runs "WrongGPU.exe apply --batch=<file> --sha256=<hex> --pipe=<name>" through the suite's
own run function and returns the worker's exit code and what it sent over the pipe.
"""
import ctypes
import ctypes.wintypes as wt
import hashlib
import os
import secrets
import struct
import threading

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateNamedPipeW.restype = wt.HANDLE
_k32.CreateNamedPipeW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, wt.DWORD, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_void_p]
_k32.ConnectNamedPipe.argtypes = [wt.HANDLE, ctypes.c_void_p]
_k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
_k32.CreateFileW.restype = wt.HANDLE
_k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
_k32.CloseHandle.argtypes = [wt.HANDLE]
_INVALID = wt.HANDLE(-1).value
_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA = 535, 232


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package_digest(path):
    """The OptiScaler package's digest as the launcher puts it in a job (package_snapshot.h): the SHA-256 of a manifest of
    each file's relative path and SHA-256; an archive is its one file, a folder (or the OptiScaler.dll in it) every file in
    it and its subfolders, links left out, sorted by the lower-case relative path."""
    if os.path.isfile(path) and path.lower().endswith((".7z", ".zip")):
        kind, root, files = "archive", os.path.dirname(path), [os.path.basename(path)]
    else:
        root = os.path.dirname(path) if os.path.isfile(path) else path
        kind, files = "folder", []
        for d, dirs, names in os.walk(root):
            dirs[:] = [n for n in dirs if not os.path.islink(os.path.join(d, n)) and not _is_junction(os.path.join(d, n))]
            files += [os.path.relpath(os.path.join(d, n), root) for n in names if not os.path.islink(os.path.join(d, n))]
        files.sort(key=lambda r: r.lower())
    manifest = "; WrongGPU OptiScaler package\r\nkind=%s\r\n" % kind
    for rel in files:
        with open(os.path.join(root, rel), "rb") as f:
            manifest += "file=%s\r\nsha256=%s\r\n" % (rel, sha(f.read()))
    return sha(manifest.encode("utf-8"))


def _is_junction(p):
    try:
        return os.path.isjunction(p)  # Python 3.12
    except AttributeError:
        return False


class Reply:
    """What the worker sent: results {job number: result text}, refused (the refusal's text or None), log lines, complete."""

    def __init__(self):
        self.results, self.refused, self.log, self.complete = {}, None, [], False


def _parse(data):
    rep, at = Reply(), 0
    while at + 9 <= len(data):
        kind, n, size = struct.unpack_from("<cII", data, at)
        body = data[at + 9:at + 9 + size].decode("utf-8", "replace")
        at += 9 + size
        if kind == b"L":
            rep.log.append(body)
        elif kind == b"R":
            rep.results[n] = body
        elif kind == b"X":
            rep.refused = body
        elif kind == b"E":
            rep.complete = True
    return rep


def run_worker(run, folder, job_texts, stop=False, batch_sha=None, change=None):
    """run(*args) -> exit code: the suite's own (WrongGPU.exe <args> ...). job_texts: str or bytes, one per job, in order.
    batch_sha: another --sha256 than the batch file's. change(batch, jobs): changes files after they were written.
    Returns (exit code, Reply). The job and batch files are removed afterwards."""
    stamp = secrets.token_hex(4)
    jobs = [os.path.join(folder, "job-%s-%d.txt" % (stamp, i + 1)) for i in range(len(job_texts))]
    for path, text in zip(jobs, job_texts):
        with open(path, "wb") as f:
            f.write(text.encode("utf-8") if isinstance(text, str) else text)
    batch = os.path.join(folder, "batch-%s.txt" % stamp)
    listing = ("stop=1\r\n" if stop else "") + "".join("job=%s\r\nsha256=%s\r\n" % (j, sha(open(j, "rb").read())) for j in jobs)
    with open(batch, "wb") as f:
        f.write(listing.encode("utf-8"))
    digest = batch_sha or sha(listing.encode("utf-8"))
    if change:
        change(batch, jobs)
    name = "\\\\.\\pipe\\WrongGPU-reply-" + secrets.token_hex(16)
    pipe = _k32.CreateNamedPipeW(name, 1, 0, 1, 65536, 65536, 0, None)  # inbound, byte mode, one instance
    if pipe == _INVALID:
        raise OSError("CreateNamedPipeW failed (%d)" % ctypes.get_last_error())
    got = bytearray()

    def reader():
        if not _k32.ConnectNamedPipe(pipe, None) and ctypes.get_last_error() not in (_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA):
            return
        buf, n = ctypes.create_string_buffer(65536), wt.DWORD()
        while _k32.ReadFile(pipe, buf, len(buf), ctypes.byref(n), None) and n.value:
            got.extend(buf.raw[:n.value])

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        rc = run("apply", "--batch=" + batch, "--sha256=" + digest, "--pipe=" + name)
        th.join(5)
        if th.is_alive():  # the worker never connected: connect once ourselves so the reader returns
            h = _k32.CreateFileW(name, 0x40000000, 0, None, 3, 0, None)
            if h != _INVALID:
                _k32.CloseHandle(h)
            th.join(5)
    finally:
        _k32.CloseHandle(pipe)
        for path in jobs + [batch]:
            if os.path.exists(path):
                os.remove(path)
    return rc, _parse(bytes(got))
