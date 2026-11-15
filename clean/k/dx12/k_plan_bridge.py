# SPDX-License-Identifier: MIT
"""Model K port checks: run a launch plan of the stage runner (clean/k/runner/k_stages.py prepare: buffers and launches
as the HIP kernels take them) on HIP in this process, and write the same launches as a kernel_replay chain, so that the
D3D12 ports run on the same input bytes (clean/k/dx12/k_port_check.py compares the raw bytes).

PlanDev wraps the HIP device for the plan's allocations: every buffer gets SLACK bytes after its size and starts filled
with the HIP sentinel (an uploaded input overwrites its first bytes); the uploads are recorded, so the chain knows which
buffers are inputs. chain_lines turns every launch into a chain step: pointer arguments become @<buffer>[+<bytes>] (a
pointer into a buffer, as the weight blob sections are), scalars decimal values, structs by value B:<file>, and structs
with pointers S:<file>:<offset>=@<buffer>[+<bytes>],... where the manifest's argument token (s<N>:<fields>) says which
bytes are pointers (clean/runtime/struct_args.h).
"""
import ctypes
import json
import os
import struct

HIP_SENTINEL, D3D12_SENTINEL, SLACK = 0xA5, 0x5A, 64


class PlanDev:
    """The Hip device as the plan builder (k_stages._Plan) sees it, with slack and sentinels."""

    def __init__(self, hip):
        self.hip = hip
        self.inputs = set()
        self.sizes = {}

    def malloc(self, nbytes):
        p = self.hip.malloc(nbytes + SLACK)
        self.hip.memset(p, HIP_SENTINEL, nbytes + SLACK)
        self.sizes[p] = nbytes
        return p

    def upload(self, ptr, data):
        self.inputs.add(ptr)
        self.hip.upload(ptr, data)

    def free(self, ptr):
        self.hip.free(ptr)

    def __getattr__(self, name):
        return getattr(self.hip, name)


def manifest_args(build):
    """kernel name -> its argument tokens, from <build>/dx12/manifest.json."""
    with open(os.path.join(build, "dx12", "manifest.json")) as f:
        return {k["name"]: k["args"].split(",") for k in json.load(f)["kernels"]}


def struct_pointer_offsets(token):
    """Byte offsets of the pointer fields of an s<N>:<fields> token."""
    offs, at = [], 0
    for f in token.split(":", 1)[1].split("."):
        kind, n = f[0], int(f[1:])
        if kind == "p":
            offs += [at + 8 * i for i in range(n)]
            at += 8 * n
        else:
            at += n
    return offs


def arg_bytes(arg):
    if isinstance(arg, (bytes, bytearray)):
        return bytes(arg)
    return bytes(arg)  # ctypes Structure / array


def scalar_word(arg):
    """A 32-bit scalar argument (ctypes int / uint / float) as an unsigned decimal."""
    v = arg.value
    if isinstance(arg, ctypes.c_float):
        return struct.unpack("<I", struct.pack("<f", v))[0]
    return v & 0xFFFFFFFF


def pointer_value(arg):
    return int(arg.value or 0) if hasattr(arg, "value") else int(arg)


def locate(ptr, bufs):
    """(buffer name, byte offset) of a device pointer inside one of the plan's buffers."""
    for name, (base, size) in bufs.items():
        if base <= ptr < base + size + SLACK:
            return name, ptr - base
    raise ValueError(f"pointer {ptr:#x} is in no buffer of the plan")


def ref(name, off):
    return f"@b_{name}" + (f"+{off}" if off else "")


def chain_step(kernel, grid, block, args, tokens, bufs, cdir, tag):
    """The `step` line of one launch (and the files its struct arguments need)."""
    if len(args) != len(tokens):
        raise ValueError(f"{kernel}: {len(args)} arguments for the manifest's {len(tokens)}")
    out = []
    for i, (a, tok) in enumerate(zip(args, tokens)):
        if tok == "p":
            v = pointer_value(a)
            out.append("null" if v == 0 else ref(*locate(v, bufs)))
        elif tok == "u":
            out.append(str(scalar_word(a)))
        elif tok.startswith("b"):
            path = f"{tag}.a{i}"
            with open(os.path.join(cdir, path), "wb") as f:
                f.write(arg_bytes(a))
            out.append(f"B:{path}")
        elif tok.startswith("s"):
            data = bytearray(arg_bytes(a))
            patch = []
            for off in struct_pointer_offsets(tok):
                v = struct.unpack_from("<Q", data, off)[0]
                if v:
                    patch.append(f"{off}={ref(*locate(v, bufs))}")
            path = f"{tag}.a{i}"
            with open(os.path.join(cdir, path), "wb") as f:
                f.write(bytes(data))
            out.append(f"S:{path}" + (":" + ",".join(patch) if patch else ""))
        else:
            raise ValueError(f"{kernel}: argument token {tok!r}")
    return f"step {kernel} {grid} {block} " + " ".join(out)


def chain_lines(name, plan, pdev, tokens_of, cdir, modules, outputs, prefix="", grid_div=1, d3d12_name=None):
    """The kernel_replay chain of a plan: buffers (inputs from files, the rest filled with the D3D12 sentinel), one step
    per launch, a save per output buffer (<prefix><name>.<buffer>.d3d12). grid_div: launch with grid // grid_div (the
    negative control uses 2). d3d12_name: the manifest's name of a plan kernel (default: the part after the code object's
    colon; the tagged exact input kernels are kx/<name>). modules: module lines, "<code object>[ <tag>]"."""
    bufs = {n: (p, pdev.sizes[p]) for n, p in plan["bufs"].items()}
    lines = [f"chain {prefix}{name}"] + [f"module {m}" for m in modules]
    for n, (p, size) in bufs.items():
        if p in pdev.inputs:
            lines.append(f"buf b_{n} {size + SLACK} {name}.{n}.in {D3D12_SENTINEL}")
        else:
            lines.append(f"buf b_{n} {size + SLACK} - {D3D12_SENTINEL}")
    for li, (kernel, grid, block, args) in enumerate(plan["launches"]):
        dname = d3d12_name(kernel) if d3d12_name else kernel.split(":")[-1]
        lines.append(chain_step(dname, max(1, grid // grid_div), block, args, tokens_of(dname), bufs, cdir,
                                f"{prefix}{name}.l{li}"))
    for n in outputs:
        lines.append(f"save b_{n} {prefix}{name}.{n}.d3d12")
    lines.append("")
    return lines
