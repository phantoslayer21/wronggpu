# SPDX-License-Identifier: GPL-3.0-only
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
from pathlib import Path
import struct

from clean.k.runner.k_pack import block_sections, linear_sections, stage_sections

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


def head_rows():
    """Physical rows for the 40 logical channels, shared with the C++ packer and shader."""
    path = Path(__file__).with_name("hlsl") / "k_head_rows.inc"
    values = " ".join(line.split("//", 1)[0] for line in path.read_text(encoding="utf-8").splitlines())
    return tuple(int(value) for value in values.split(","))


def _bias32(half):
    """Widen FP16 bits as k_bias32_bits does, including NaN sign and payload."""
    payload = half & 0x3FF
    if half & 0x7C00 == 0x7C00 and payload:
        return (half & 0x8000) << 16 | 0x7FC00000 | payload << 13
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", half))[0]))[0]


def narrow_d3d12_inputs(st, blob):
    """Return the FP32 bias plane and optional compact d0 weights for the two-window ports."""
    C, heads, mix = st["C"], st["heads"], st["mix"]
    stage = stage_sections(st)
    block = block_sections(C, heads, mix)
    bias = bytearray()

    def append(data, offset, count):
        for (half,) in struct.iter_unpack("<H", data[offset:offset + count * 2]):
            bias.extend(struct.pack("<I", _bias32(half)))

    if "embed" in stage:
        append(blob, stage["embed"] + linear_sections(C, 16)["b"], C)
    if "expand" in stage:
        append(blob, stage["expand"] + linear_sections(4 * C, st["C_prev"])["b"], 4 * C)
    for name, count in (("bo", C), ("b1", 4 * C), ("b2", C)):
        append(blob, stage["block"] + block[name], count)
    if "merge" in stage:
        append(blob, stage["merge"] + linear_sections(st["C_next"], 4 * C)["b"], st["C_next"])
    compact = None
    if "head" in stage:
        original = blob[stage["head"]:]
        old_bias = linear_sections(40, C)["b"]
        compact = bytearray(linear_sections(32, C)["b"])
        head_bias = bytearray(32 * 2)
        tiles = C // 16
        for logical, physical in enumerate(head_rows()):
            if physical < 0:
                continue
            for tile in range(tiles):
                for half_tile in (0, 16):
                    src = 512 * (logical // 16 * tiles + tile) + 16 * (half_tile + logical % 16)
                    dst = 512 * (physical // 16 * tiles + tile) + 16 * (half_tile + physical % 16)
                    compact[dst:dst + 16] = original[src:src + 16]
            head_bias[physical * 2:physical * 2 + 2] = original[old_bias + logical * 2:old_bias + logical * 2 + 2]
        append(head_bias, 0, 32)
    if mix == "softmax":
        append(blob, stage["block"] + block["mix"], heads * 64 * 64)
    return bytes(bias), None if compact is None else bytes(compact)


def narrow_d3d12_args(plan, pdev, st, blob):
    """Allocate the D3D12-only inputs and return their arguments in manifest order."""
    bias, compact = narrow_d3d12_inputs(st, blob)
    args = []
    for name, data in (("bias32", bias), ("compact_head", compact)):
        if data is None:
            continue
        pointer = pdev.malloc(len(data))
        plan["bufs"][name] = pointer
        pdev.upload(pointer, data)
        args.append(ctypes.c_void_p(pointer))
    return args


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


def chain_lines(name, plan, pdev, tokens_of, cdir, modules, outputs, prefix="", grid_div=1, d3d12_name=None, extra=()):
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
    lines += list(extra)   # e.g. the direct texture lines of kernel_replay
    for li, (kernel, grid, block, args) in enumerate(plan["launches"]):
        dname = d3d12_name(kernel) if d3d12_name else kernel.split(":")[-1]
        lines.append(chain_step(dname, max(1, grid // grid_div), block, args, tokens_of(dname), bufs, cdir,
                                f"{prefix}{name}.l{li}"))
    for n in outputs:
        lines.append(f"save b_{n} {prefix}{name}.{n}.d3d12")
    lines.append("")
    return lines
