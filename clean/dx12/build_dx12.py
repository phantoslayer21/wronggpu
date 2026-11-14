# SPDX-License-Identifier: MIT
"""Builds the D3D12 ports of the HIP kernels (clean/dx12/kernels/*.hlsl) for the hybrid device backend (runtime backend 2,
clean/runtime/device_dx12.cpp): <out>/dx12/<tag>__<kernel>.dxil (or <kernel>.dxil without a tag) + <out>/dx12/manifest.json.

    python clean/dx12/build_dx12.py --out build/dx12-m/hip [--isa] [--only exposure_auto_diet,...] [-j N] [--model-l]

Model L's kernels (#89: those that reproduce one of Model L's code objects, co=l_*.co, direct-route kernels named l_*, and MODEL_L_ONLY) are built only with --model-l
or when --only names them, so that a build without the option is Model M's kernel set unchanged.

Every kernel source declares its kernels in header lines (clean/dx12/README.md):

    // @wg-kernel <name as launched> co=<code object> block=<lanes> wave=<32|64> args=<kinds> [entry=<HLSL entry>]
    //            [defines=K=V,K=V] [group=<lanes> [gscale=<n>]]

name is the name the runtime launches ("exposure_auto_diet", "bneck/swin_bneck6_nc"); the entry defaults to the part after
the tag. co is the file name of the HIP code object whose build of the kernel this port reproduces ("swin.co",
"post_w64__orig.co"): the device runs the port only while the runtime has exactly that code object loaded (untagged, or
under the name's tag), so a different build variant of the same kernel name keeps running on HIP. co=- marks a
D3D12-only kernel with no HIP counterpart (the device self-test's kernels): it always runs on D3D12. One name may have
ports of several builds of its HIP kernel (for example w64/post_tf_f32_p12_diet_vsit of post_w64__orig.co and of
post_w64.co): one manifest entry per (name, co), and the device runs the one whose code object is loaded. Every port but
the first of a name sets file=<stem> (its DXIL is <stem>.dxil; default <tag>__<kernel>). args lists the HIP kernel's arguments in order: p = pointer (a uint byte offset into the arena; 0 = null),
u = any 32-bit scalar, b<N> = a struct of N bytes passed by value (N / 4 uint constants).

Arithmetic mode ("exact", README.md): DXC compiles with IEEE strictness (-Gis: no fast-math flags, so no contraction, no
reassociation and no NaN / infinity assumptions in DXC or the driver); the listing is then rewritten so that the intrinsic
calls keep the driver's default (non-precise) lowering, above all mad() = one fused v_fma_f32 and dot2add = v_dot2_f32_f16,
and rcp() (fdiv 1.0, x) is the hardware v_rcp_f32; it is reassembled (dxil_asm.cpp, DXC's IDxcAssembler) and signed by
dxv. Any other float division ('/', which would be an IEEE division) or float remainder is refused: use wg_div.

VGPR check: every Swin / bottleneck kernel built (a name with a swin_ or l_swin_ part) is run through RGA (gfx1201) and its
USED_VGPRs compared with its bound in clean/dx12/vgpr_limits.json (the count that keeps its present workgroups per CU);
over the bound, or no bound listed, fails the build. WG_DX12_VGPR_CHECK=0 skips it.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.join(HERE, "kernels")
HLSL_DIR = os.path.join(HERE, "hlsl")
# Pinned tools (BUILDING.md): WG_TOOLCHAINS = a folder holding dxc-v1.10.2605.24\ and rga-2.15\ (default <repo>\toolchains);
# WG_DXC_DIR / WG_RGA point at the tools directly.
TOOLCHAINS = os.environ.get("WG_TOOLCHAINS", os.path.join(os.path.dirname(os.path.dirname(HERE)), "toolchains"))
DXC_DIR = os.environ.get("WG_DXC_DIR", os.path.join(TOOLCHAINS, "dxc-v1.10.2605.24", "bin", "x64"))
RGA = os.environ.get("WG_RGA", os.path.join(TOOLCHAINS, "rga-2.15", "rga.exe"))
VSDEVCMD = os.environ.get("VSDEVCMD", r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat")
DXC_ARGS = ["-T", "cs_6_6", "-O3", "-HV", "2021", "-enable-16bit-types", "-Gis"]
MANIFEST = "manifest.json"
FORMAT = 1
MAX_ARG_DWORDS = 57   # root signature: 64 DWORDs - 2 root UAVs (2 each) - 1 descriptor table - 2 runtime constants


def tool_dir(out):
    return os.path.join(out, "dx12", "tools")


def build_asm_tool(out):
    """dxil_asm.exe (clean/dx12/dxil_asm.cpp) into <out>/dx12/tools, rebuilt when the source is newer."""
    exe = os.path.join(tool_dir(out), "dxil_asm.exe")
    src = os.path.join(HERE, "dxil_asm.cpp")
    if os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(src):
        return exe
    os.makedirs(tool_dir(out), exist_ok=True)
    obj = os.path.join(tool_dir(out), "dxil_asm.obj")
    cmd = (f'"{VSDEVCMD}" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /EHsc /std:c++17 "{src}" '
           f'/Fe"{exe}" /Fo"{obj}"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        raise SystemExit("building dxil_asm failed:\n" + r.stdout + r.stderr)
    return exe


_PRECISE = re.compile(r", !dx\.precise ![0-9]+")
_FDIV = re.compile(r"\bfdiv (fast )?(<\d+ x (float|half|double)>|float|half|double) (<[^>]*>|[^,\s]+),")
_ONE = {"float": ("1.000000e+00",), "half": ("0xH3C00",)}


def exact_listing(text, where, allow_fdiv=False):
    """The -Gis listing with the intrinsics' precise marks removed and rcp() as the hardware reciprocal (see the module
    doc). Raises ValueError for a float division other than 1 / x or a float remainder."""
    text = _PRECISE.sub("", text)

    def fdiv(m):
        ty, scalar, num = m.group(2), m.group(3) or m.group(2), m.group(4).strip()
        if scalar == "double":  # FP64 division stays IEEE (HIP's (double) a / b)
            return m.group(0)
        ones = _ONE[scalar]
        vec = re.findall(r"(?:float|half) ([^,>\s]+)", num) if num.startswith("<") else [num]
        if not vec or any(v not in ones for v in vec):
            if allow_fdiv:  # probes only: an IEEE division stays as it is
                return m.group(0)
            raise ValueError(f"{where}: float division '{m.group(0)}' - only rcp() / 1 / x is allowed; use wg_div()")
        return f"fdiv fast {ty} {num},"

    text = _FDIV.sub(fdiv, text)
    if re.search(r"\bfrem\b", text):
        raise ValueError(f"{where}: float remainder (%) is not allowed")
    return text


def compile_hlsl(src, entry, out_dxil, defines=None, includes=(), asm_tool=None, keep_listing=None, allow_fdiv=False):
    """One entry point of an HLSL file in the exact arithmetic mode -> signed DXIL container at out_dxil."""
    dxc, dxv = os.path.join(DXC_DIR, "dxc.exe"), os.path.join(DXC_DIR, "dxv.exe")
    with tempfile.TemporaryDirectory(prefix="wgdx12_") as tmp:
        ll = os.path.join(tmp, "a.ll")
        cmd = [dxc, *DXC_ARGS, "-E", entry, "-I", HLSL_DIR, *sum((["-I", i] for i in includes), []),
               *[f"-D{k}={v}" for k, v in (defines or {}).items()], "-Fc", ll, src]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"dxc failed for {src} ({entry}):\n{r.stdout}{r.stderr}")
        with open(ll, encoding="utf-8", errors="replace") as f:
            text = exact_listing(f.read(), f"{os.path.basename(src)}:{entry}", allow_fdiv)
        ll2 = os.path.join(tmp, "b.ll")
        with open(ll2, "w", encoding="utf-8") as f:
            f.write(text)
        if keep_listing:
            shutil.copyfile(ll2, keep_listing)
        raw = os.path.join(tmp, "a.dxil")
        r = subprocess.run([asm_tool, os.path.join(DXC_DIR, "dxcompiler.dll"), ll2, raw], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"assembling {src} ({entry}) failed:\n{r.stdout}{r.stderr}")
        r = subprocess.run([dxv, raw, "-o", out_dxil], capture_output=True, text=True)
        if r.returncode != 0 or "Validation succeeded" not in r.stdout + r.stderr:
            raise SystemExit(f"dxv rejected {src} ({entry}):\n{r.stdout}{r.stderr}")
    return out_dxil


_DECL = re.compile(r"^\s*//\s*@wg-kernel\s+(\S+)(.*)$")


def kernel_decls(path):
    decls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _DECL.match(line)
            if not m:
                continue
            name, rest = m.group(1), m.group(2)
            kv = dict(t.split("=", 1) for t in rest.split())
            if not (kv.get("co", "").endswith(".co") or kv.get("co") == "-"):
                raise SystemExit(f"{path}: {name}: co=<the HIP code object this port reproduces> is required "
                                 "(co=- only for D3D12-only test kernels)")
            d = {"name": name, "co": kv["co"], "entry": kv.get("entry", name.split("/")[-1]), "block": int(kv["block"]),
                 "wave": int(kv.get("wave", "32")), "args": kv["args"], "source": os.path.basename(path),
                 "defines": dict(x.split("=", 1) for x in kv["defines"].split(",")) if kv.get("defines") else {}}
            if d["wave"] not in (32, 64) or d["block"] % d["wave"] or d["block"] > 1024:
                raise SystemExit(f"{path}: {name}: block {d['block']} / wave {d['wave']}")
            # group=<lanes>: the port runs each launched workgroup of `block` lanes as block / group D3D12 workgroups
            # of `group` lanes (WG_BLOCK = group; the device dispatches grid x block / group workgroups)
            # gscale=<n>: D3D12 workgroups per launched workgroup (default block / group; 1: the port does a launched
            # workgroup's whole work with `group` lanes)
            d["group"] = int(kv.get("group", d["block"]))
            if d["group"] <= 0 or d["block"] % d["group"] or d["group"] % d["wave"]:
                raise SystemExit(f"{path}: {name}: group {d['group']} / block {d['block']} / wave {d['wave']}")
            d["gscale"] = int(kv.get("gscale", d["block"] // d["group"]))
            if d["gscale"] < 1 or d["gscale"] > d["block"] // d["group"]:
                raise SystemExit(f"{path}: {name}: gscale {d['gscale']}")
            d["file"] = kv.get("file", "")   # DXIL file stem of a second port of the same name (module doc)
            if d["file"] and not re.fullmatch(r"[A-Za-z0-9_]+", d["file"]):
                raise SystemExit(f"{path}: {name}: file={d['file']} (letters, digits and _ only)")
            d["rs"] = kv.get("rs", "buffer")
            if d["rs"] not in ("buffer", "tex", "direct"):
                raise SystemExit(f"{path}: {name}: unsupported root-signature kind {d['rs']}")
            d["arg_dwords"] = arg_dwords(d["args"], f"{path}: {name}")
            decls.append(d)
    return decls


# Kernels of Model M's code objects that only Model L's frame launches (#89): dec1 as a standalone block (#139;
# Model M runs dec1 fused with dec0).
MODEL_L_ONLY = {"swin_block_c64_add_ex_nc"}


def is_model_l(decl):
    """A port of one of Model L's code objects (#89: l_swin.co, l_ops.co, l_post.co, l_enc_values.co), a Model L kernel
    of the direct route with no code object (co "-", named l_*), or a kernel only Model L launches (MODEL_L_ONLY)."""
    return decl["co"].startswith("l_") or decl["name"].startswith("l_") or decl["name"] in MODEL_L_ONLY


def kernel_set(model_l=False):
    """(name, code object) of every kernel a full build compiles (no --only), sorted: Model M's set, plus Model L's
    with model_l. From the sources' declarations alone (nothing is compiled)."""
    keys = set()
    for fn in sorted(os.listdir(KERNEL_DIR)) if os.path.isdir(KERNEL_DIR) else []:
        if fn.endswith(".hlsl"):
            for d in kernel_decls(os.path.join(KERNEL_DIR, fn)):
                if model_l or not is_model_l(d):
                    keys.add((d["name"], d["co"]))
    return sorted(keys)


def arg_dwords(args, where):
    n = 0
    for a in args.split(","):
        if a in ("p", "u"):
            n += 1
        elif re.fullmatch(r"b\d+", a) and int(a[1:]) % 4 == 0 and int(a[1:]) > 0:
            n += int(a[1:]) // 4
        else:
            raise SystemExit(f"{where}: argument kind {a!r} (p, u, b<N> with N a multiple of 4)")
    if n > MAX_ARG_DWORDS:
        raise SystemExit(f"{where}: {n} argument DWORDs, at most {MAX_ARG_DWORDS} fit the root signature")
    return n


def output_name(name):
    tag, _, k = name.rpartition("/")
    return (f"{tag}__{k}" if tag else k) + ".dxil"


def rga_isa(dxil, isa_dir, stem):
    """D3D12 ISA of one kernel (RGA, gfx1201) -> <isa_dir>/<stem>.isa.txt and .stats.csv."""
    os.makedirs(isa_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wgrga_") as tmp:
        subprocess.run([RGA, "-s", "dx12", "-c", "gfx1201", "--cs-blob", dxil, "--isa", os.path.join(tmp, "x.txt"),
                        "-a", os.path.join(tmp, "s.csv")], capture_output=True, text=True)
        for f in os.listdir(tmp):
            kind = "isa.txt" if f.endswith("_x_comp.txt") else ("stats.csv" if f.endswith("_s_comp.csv") else None)
            if kind:
                shutil.copyfile(os.path.join(tmp, f), os.path.join(isa_dir, f"{stem}.{kind}"))


VGPR_LIMITS = os.path.join(HERE, "vgpr_limits.json")


def vgpr_checked(name):
    return re.search(r"(^|/|l_)swin_", name) is not None   # l_swin_*: Model L's blocks (#89)


def rga_vgprs(dxil):
    """RGA's USED_VGPRs of one kernel (gfx1201), None when RGA gives no statistics."""
    with tempfile.TemporaryDirectory(prefix="wgrga_") as tmp:
        subprocess.run([RGA, "-s", "dx12", "-c", "gfx1201", "--cs-blob", dxil, "-a", os.path.join(tmp, "s.csv")],
                       capture_output=True, text=True)
        for f in os.listdir(tmp):
            if f.endswith("_s_comp.csv"):
                with open(os.path.join(tmp, f), newline="") as fh:
                    rows = list(csv.DictReader(fh))
                return int(rows[0]["USED_VGPRs"]) if rows else None
    return None


def vgpr_check(built):
    """built: [(kernel name, dxil path)] of this build. Fails when a Swin / bottleneck kernel exceeds its bound."""
    todo = [(n, p) for n, p in built if vgpr_checked(n)]
    if not todo or os.environ.get("WG_DX12_VGPR_CHECK", "1") == "0":
        return
    if not os.path.exists(RGA):
        raise SystemExit(f"VGPR check: RGA not found at {RGA} (WG_RGA, or WG_DX12_VGPR_CHECK=0 to skip)")
    with open(VGPR_LIMITS, encoding="utf-8") as f:
        limits = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    # RGA runs are compiles too: at most 6 at a time, fewer when WG_DX12_JOBS asks for fewer
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(6, int(os.environ.get("WG_DX12_JOBS", 6))))) as ex:
        counts = dict(zip([n for n, _ in todo], ex.map(rga_vgprs, [p for _, p in todo])))
    bad = []
    for n in sorted(counts):
        v, lim = counts[n], limits.get(n)
        print(f"VGPR {n:40s} {v if v is not None else '?':>4} / {lim if lim is not None else '-'}", flush=True)
        if v is None:
            bad.append(f"{n}: RGA gave no VGPR count")
        elif lim is None:
            bad.append(f"{n}: {v} VGPRs, no bound in {os.path.basename(VGPR_LIMITS)} (add the bound of its occupancy)")
        elif v > lim:
            bad.append(f"{n}: {v} VGPRs > {lim} (one workgroup per CU less: restore the bound or re-measure and "
                       "change it in vgpr_limits.json)")
    if bad:
        raise SystemExit("VGPR check failed:\n  " + "\n  ".join(bad))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="build directory of the runtime (kernels go to <out>/dx12); required unless --list")
    ap.add_argument("--only", help="comma list of kernel names to build (default: all declared)")
    ap.add_argument("--isa", action="store_true", help="also write the D3D12 ISA (RGA) to <out>/dx12/isa")
    ap.add_argument("--listing", action="store_true", help="also keep the rewritten DXIL listing (<name>.ll)")
    ap.add_argument("-j", "--jobs", type=int, default=int(os.environ.get("WG_DX12_JOBS", min(12, os.cpu_count() or 1))),
                    help="kernels compiled in parallel (default min(12, CPUs), env WG_DX12_JOBS; 1 = one at a time)")
    ap.add_argument("--model-l", action="store_true",
                    help="also build Model L's kernels (co=l_*.co; off by default: Model M's kernel set unchanged)")
    ap.add_argument("--list", action="store_true",
                    help="print the kernels a build with these options compiles (name and code object, JSON) and exit: "
                         "the cheap check before a build (kernel_sets.json, tests/test_kernel_sets.py)")
    a = ap.parse_args(argv)
    if a.list:
        print(json.dumps([list(k) for k in kernel_set(a.model_l)], indent=1))
        return 0
    if not a.out:
        ap.error("--out is required")
    dst = os.path.join(a.out, "dx12")
    os.makedirs(dst, exist_ok=True)
    asm = build_asm_tool(a.out)
    only = set(a.only.split(",")) if a.only else None
    man_path = os.path.join(dst, MANIFEST)
    old = {}
    if only and os.path.exists(man_path):  # partial build: keep the other kernels' entries
        with open(man_path) as f:
            old = {(k["name"], k.get("co", "")): k for k in json.load(f)["kernels"]}
    kernels = dict(old)   # (name, co) -> manifest entry
    built = []
    # generated kernels (kernels/fused_*_gen.py: the fused forms of swin_block.hlsl / bneck.hlsl) are regenerated from
    # the live sources first, so that no build compiles a stale copy
    for fn in sorted(os.listdir(KERNEL_DIR)) if os.path.isdir(KERNEL_DIR) else []:
        if fn.startswith("fused_") and fn.endswith("_gen.py"):
            subprocess.run([sys.executable, os.path.join(KERNEL_DIR, fn)], check=True)
    jobs = []  # (source, declaration, output path) in declaration order (sorted file name, then line)
    for fn in sorted(os.listdir(KERNEL_DIR)) if os.path.isdir(KERNEL_DIR) else []:
        if not fn.endswith(".hlsl"):
            continue
        src = os.path.join(KERNEL_DIR, fn)
        for d in kernel_decls(src):
            if only and d["name"] not in only:
                continue
            if not only and not a.model_l and is_model_l(d):
                continue
            key = (d["name"], d["co"])
            if key in kernels and key not in old:
                raise SystemExit(f"kernel {d['name']} ({d['co']}) declared twice")
            kernels[key] = None  # placeholder: the declared-twice check sees it before any compile starts
            out = os.path.join(dst, d["file"] + ".dxil" if d["file"] else output_name(d["name"]))
            if any(j[2] == out for j in jobs):
                raise SystemExit(f"kernel {d['name']} ({d['co']}): DXIL file {os.path.basename(out)} already used "
                                 "(a second port of a name needs file=<stem>)")
            jobs.append((src, d, out))

    def build_one(job):
        src, d, out = job
        defs = {"WG_BLOCK": d["group"], "WG_WAVE": d["wave"], **d["defines"]}
        if d["group"] != d["block"]:
            defs["WG_LAUNCH_BLOCK"] = d["block"]
        compile_hlsl(src, d["entry"], out, defs, asm_tool=asm, keep_listing=out[:-5] + ".ll" if a.listing else None)
        with open(out, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if a.isa:
            rga_isa(out, os.path.join(dst, "isa"), os.path.basename(out)[:-5])
        r = {"name": d["name"], "file": os.path.basename(out), "co": d["co"], "entry": d["entry"],
             "block": d["block"], "wave": d["wave"], "args": d["args"], "arg_dwords": d["arg_dwords"],
             "source": d["source"], "sha256": sha, "rs": d["rs"]}
        if d["group"] != d["block"]:
            r["group"] = d["group"]
            r["gscale"] = d["gscale"]
        return r

    # Every kernel is an independent chain of child processes (dxc -> dxil_asm -> dxv [-> rga]) with its own temporary
    # directory and output file, so a thread pool runs the chains in parallel processes; results are taken in
    # declaration order, so the log and the manifest (sorted by name) do not depend on the completion order.
    n_jobs = max(1, min(a.jobs, len(jobs))) if jobs else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(build_one, j) for j in jobs]
        try:
            for (src, d, out), fut in zip(jobs, futures):
                kernels[(d["name"], d["co"])] = fut.result()
                built.append((d["name"], out))
                print(f"{d['name']:40s} -> {os.path.basename(out)}  block {d['block']} wave {d['wave']} "
                      f"args {d['args']}", flush=True)
        except BaseException:
            for fut in futures:
                fut.cancel()  # kernels not started yet; the running ones finish their current child process
            raise
    names = {n for n, _ in kernels}
    if only and not (only <= names):
        raise SystemExit(f"not declared in clean/dx12/kernels: {sorted(only - names)}")
    vgpr_check(built)
    with open(man_path, "w") as f:
        json.dump({"format": FORMAT, "dxc": os.path.basename(os.path.dirname(os.path.dirname(DXC_DIR))),
                   "kernels": [kernels[k] for k in sorted(kernels)]}, f, indent=1)
    print(f"{len(kernels)} kernel(s) in {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
