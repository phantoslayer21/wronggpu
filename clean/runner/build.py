# SPDX-License-Identifier: MIT
"""Build the gfx1201 code objects (spec/90-test-interface.md) and, optionally, the host emulator.

    python clean/runner/build.py --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --out build
    python clean/runner/build.py --rocm <ROCm> --out build --clang <path to clang>   # explicit compiler
    python clean/runner/build.py --out build --no-device-libs --clang clang         # compile check without ROCm
    python clean/runner/build.py --out build --emu-only                              # host emulator (Linux, tests)
    python clean/runner/build.py --rocm <ROCm> --out build --variants v2 --report     # second design (swin_v2.hip)

On Windows the runtime library is wronggpu_runtime.dll: the core runtime plus the D3D12 interop (spec 10 sections 1, 6,
8; the FidelityFX front end loads it through wgGetUpscalerBackend), built with the HIP SDK's clang against the static
Microsoft C runtime (/MT). The converter map is copied next to it (the runtime reads it to check model.wgcm).

Each kernel source is compiled with the command given in spec 90, plus a fixed compilation-unit id (device_command):
    clang -x hip --offload-arch=gfx1201 --cuda-device-only -nogpuinc -O3 --no-gpu-bundle-output -cuid=wg_<out name>
          --rocm-path=<ROCm> --rocm-device-lib-path=<ROCm>/amdgcn/bitcode <src> -o <out>.co
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from clean.runner import perf_variants as PV  # noqa: E402
from clean.runner import variants as V  # noqa: E402
from clean.k.runner import k_build  # noqa: E402  (model K kernels, clean/k)
KERNEL_DIR = os.path.join(os.path.dirname(HERE), "kernels")
SOURCES = ["swin.hip", "probe.hip", "overlap.hip", "rates.hip", "mlpbench.hip", "encoder.hip", "decoder.hip", "exposure.hip",
           "post.hip", "frame.hip", "dec1_dec0.hip"]
RUNTIME_DIR = os.path.join(os.path.dirname(HERE), "runtime")
RUNTIME_SOURCES = ["device.cpp", "embedded.cpp", "model.cpp", "runtime.cpp",
                   os.path.join("..", "converter", "src", "sha256.cpp"),   # shared with the converter (spec 08)
                   os.path.join("..", "converter", "src", "json.cpp"),
                   os.path.join("..", "k", "runtime", "model_k.cpp"),                     # model K layer table
                   os.path.join("..", "k", "runtime", "k_pack.cpp"),                      # model K weight packer
                   os.path.join("..", "k", "runtime", "model_k_load.cpp"),                # model K model.wgcm loader
                   os.path.join("..", "k", "runtime", "k_filter_table.cpp"),              # model K filter table
                   os.path.join("..", "k", "runtime", "runtime_k.cpp")]                   # model K frame (model=K)
INTEROP_SOURCES = [os.path.join("interop", s) for s in ("convert.cpp", "executor_hip.cpp", "executor_test.cpp",
                                                        "executor_d3d12.cpp", "backend.cpp", "native_candidate.cpp")]  # Windows only (D3D12)
MAP_FILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "preset-M.v1.map.json")
K_MAP_FILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "preset-K.v1.map.json")  # model K
EMU_RUNTIME = os.path.join("emu", "wg_emu.cpp")
# Files built into the runtime library (clean/runtime/embedded.h): the runtime's default code objects and data. A file
# of the same name next to the library takes precedence (development builds, in-game A/B payloads).
EMBED_FILES = ["swin.co", "swin__bmegahg.co", "encoder.co", "decoder.co", "decoder__pack12.co", "exposure.co", "post.co",
               "post_w64.co", "post_w64__q5.co", "post_w64__orig.co", "frame.co", "dec1_dec0.co", "dec1_dec0__pack12.co", "variants.json", os.path.basename(MAP_FILE),
               # motion vectors at output resolution (runtime option motion_outres=original; MOTION_OUTRES below)
               "encoder_mo.co", "post_mo_w64.co", "post_mo_w64__q5.co", "post_mo_w64__orig.co",
               # model K (runtime option model=K, clean/k/runtime/runtime_k.cpp): stage kernels, I/O stages, its map
               "k_block.co", "k_block_wide.co", "k_io.co", os.path.basename(K_MAP_FILE)]
# Model K in the runtime library. --no-model-k (set_model_k(False)) builds a library without it: no clean/k/runtime
# source, none of K_EMBED_FILES built in, -DWG_NO_MODEL_K (clean/runtime/no_model_k.cpp: model=K fails at create with
# "Model K is not part of this build"). Model M is the same code and the same code objects either way.
MODEL_K = True
K_RUNTIME_SOURCES = [s for s in RUNTIME_SOURCES if s.startswith(os.path.join("..", "k", ""))]
K_EMBED_FILES = ["k_block.co", "k_block_wide.co", "k_io.co", os.path.basename(K_MAP_FILE)]
NO_MODEL_K_SOURCE = "no_model_k.cpp"

# Optional models of the runtime library (#89). Model M is always built. Each model below is in by default and can be
# left out with --no-model-<name> (set_model(name, False)): its runtime sources and built-in files are dropped, its map
# is not copied, its kernels are not built, and its stand-in source is compiled with its define instead (the stand-in
# refuses model=<name> at create). Model K's row is the former --no-model-k code, unchanged; a model adds its row here.
OPTIONAL_MODELS = {
    "K": {"sources": K_RUNTIME_SOURCES, "embed": K_EMBED_FILES, "map": K_MAP_FILE, "define": "WG_NO_MODEL_K",
          "stand_in": NO_MODEL_K_SOURCE, "build_device": k_build.build_device,
          "help": "a runtime library without model K: its kernels and map are not built or built in, its runtime is "
                  "not compiled, model=K fails at create (release builds that ship model M only)"},
}
MODELS_BUILT = {name: True for name in OPTIONAL_MODELS}

# Model L (#89) in the runtime library: OFF by default (--model-l turns it on), so that a build without the option is
# Model M's build unchanged. On: Model L's runtime sources (clean/l/runtime) compiled with -DWG_MODEL_L
# (model_select.h model_l_built), its code objects (clean/l/runner/l_c32_check.py compile_kernels: l_swin, l_ops,
# l_post from clean/l/kernels, l_enc_values = encoder.hip with WG_ENC_VALUES_OUT=1) and its map built in. The D3D12
# ports of Model L's kernels are built by clean/dx12/build_dx12.py --model-l.
MODEL_L = False
L_RUNTIME_SOURCES = [os.path.join("..", "l", "runtime", "l_model.cpp"), os.path.join("..", "l", "runtime", "l_pack.cpp")]
L_MAP_FILE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "preset-L.v1.map.json")
L_KERNEL_DIR = os.path.join(os.path.dirname(HERE), "l", "kernels")
L_CODE_OBJECTS = ["l_swin.co", "l_ops.co", "l_post.co", "l_enc_values.co"]
L_EMBED_FILES = L_CODE_OBJECTS + [os.path.basename(L_MAP_FILE)]


def set_model_l(on):
    """Model L in or out (default) of the runtime library; changes RUNTIME_SOURCES and EMBED_FILES in place."""
    global MODEL_L
    if bool(on) == MODEL_L:
        return
    MODEL_L = bool(on)
    if on:
        RUNTIME_SOURCES[:] = RUNTIME_SOURCES + L_RUNTIME_SOURCES
        EMBED_FILES[:] = EMBED_FILES + L_EMBED_FILES
    else:
        RUNTIME_SOURCES[:] = [s for s in RUNTIME_SOURCES if s not in L_RUNTIME_SOURCES]
        EMBED_FILES[:] = [n for n in EMBED_FILES if n not in L_EMBED_FILES]


def build_device_l(clang, rocm, out_dir, device_libs=True, report=False, arch=None):
    """Model L's code objects into out_dir (as clean/l/runner/l_c32_check.py compile_kernels builds them), for `arch`
    (default ARCH)."""
    global ARCH
    saved = ARCH
    ARCH = arch or ARCH
    try:
        os.makedirs(out_dir, exist_ok=True)
        for name in ("l_swin", "l_ops", "l_post"):
            run(device_command(clang, rocm, os.path.join(L_KERNEL_DIR, name + ".hip"), os.path.join(out_dir, name + ".co"),
                               device_libs=device_libs, report=report, extra=["-I", KERNEL_DIR]))
        run(device_command(clang, rocm, os.path.join(KERNEL_DIR, "encoder.hip"), os.path.join(out_dir, "l_enc_values.co"),
                           device_libs=device_libs, report=report, defines={"WG_ENC_VALUES_OUT": 1}))
    finally:
        ARCH = saved


def model_built(name):
    """An optional model is part of this build (OPTIONAL_MODELS)."""
    return MODELS_BUILT[name]


def set_model(name, on):
    """Optional model `name` in (default) or out of the runtime library; changes RUNTIME_SOURCES and EMBED_FILES in
    place."""
    global MODEL_K
    m = OPTIONAL_MODELS[name]
    if bool(on) == MODELS_BUILT[name]:
        return
    MODELS_BUILT[name] = bool(on)
    if name == "K":
        MODEL_K = bool(on)  # the name the scripts and tests used before the table
    if on:
        RUNTIME_SOURCES[:] = [s for s in RUNTIME_SOURCES if s != m["stand_in"]] + m["sources"]
        EMBED_FILES[:] = EMBED_FILES + m["embed"]
    else:
        RUNTIME_SOURCES[:] = [s for s in RUNTIME_SOURCES if s not in m["sources"]] + [m["stand_in"]]
        EMBED_FILES[:] = [n for n in EMBED_FILES if n not in m["embed"]]


def set_model_k(on):
    """Model K in (default) or out of the runtime library; changes RUNTIME_SOURCES and EMBED_FILES in place."""
    set_model("K", on)


EMBED_KEY = 0x5747454D42454444   # the same in clean/runtime/embedded.cpp
ARCH = os.environ.get("WG_GPU_ARCH", "gfx1201")  # primary target: gfx1201 (RX 9070 family)
# Further RDNA 4 targets built into the runtime library (option --extra-archs): their code objects are embedded as
# "<arch>.<name>"; the runtime loads them when the GPU refuses the primary ones (clean/runtime/device.cpp).
EXTRA_ARCHS = []


def find_clang(rocm):
    exe = "clang.exe" if sys.platform == "win32" else "clang"
    if rocm:
        for sub in ("bin", os.path.join("lib", "llvm", "bin"), os.path.join("llvm", "bin")):
            p = os.path.join(rocm, sub, exe)
            if os.path.exists(p):
                return p
    p = shutil.which(exe)
    if p:
        return p
    raise SystemExit("clang not found; pass --clang")


def device_command(clang, rocm, src, out, device_libs=True, report=False, extra=(), soft_fp8=False, defines=None):
    # a fixed compilation-unit id from the output file's name, so the __hip_cuid_... symbol of a code object is the same
    # in every build folder (without -cuid, clang derives the id itself)
    cuid = "wg_" + re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(out))[0])
    cmd = [clang, "-x", "hip", f"--offload-arch={ARCH}", "--cuda-device-only", "-nogpuinc", "-O3",
           "--no-gpu-bundle-output", f"-cuid={cuid}"]
    if device_libs:
        cmd += [f"--rocm-path={rocm}", f"--rocm-device-lib-path={os.path.join(rocm, 'amdgcn', 'bitcode')}"]
    else:
        cmd += ["-nogpulib"]
    if report:
        cmd += ["-Rpass-analysis=kernel-resource-usage"]
    if soft_fp8:
        cmd += ["-DWG_SOFT_FP8"]
    for k, v in (defines or {}).items():
        cmd += [f"-D{k}={v}"]
    cmd += list(extra) + [src, "-o", out]
    return cmd


def emu_command(cxx, out, defines=None, kernel_source="swin.hip"):
    """Host emulator library from the Swin kernel source (swin.hip, or swin_v2.hip for the v2 builds) + the other
    kernel sources."""
    srcs = [os.path.join(KERNEL_DIR, s) for s in [kernel_source] + [s for s in SOURCES if s != "swin.hip"]]
    if not defines and kernel_source == "swin.hip":
        srcs += k_build.source_paths()  # model K (base emulator library only)
    return [cxx, "-std=c++17", "-O2", "-g", "-Wno-psabi", "-shared", "-fPIC", "-DWG_EMU", *[f"-D{k}={v}" for k, v in (defines or {}).items()], "-I", KERNEL_DIR,
            os.path.join(KERNEL_DIR, EMU_RUNTIME), "-x", "c++", *srcs, "-o", out]


def runtime_library_name():
    return "wronggpu_runtime.dll" if sys.platform == "win32" else "libwg_runtime.so"


def find_host_cxx(rocm):
    exe = "clang++.exe" if sys.platform == "win32" else "clang++"
    if sys.platform == "win32" and rocm:
        for sub in ("bin", os.path.join("lib", "llvm", "bin"), os.path.join("llvm", "bin")):
            p = os.path.join(rocm, sub, exe)
            if os.path.exists(p):
                return p
    p = shutil.which(exe) or shutil.which("g++")
    if p:
        return p
    raise SystemExit("no host C++ compiler found for the runtime; pass --runtime-cxx")


def scramble(name, data):
    """clean/runtime/embedded.cpp's unscramble, inverted (an XOR keystream: the same operation)."""
    h = 1469598103934665603
    for c in name.encode():
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    x = (EMBED_KEY ^ h) or 1
    out = bytearray(data)
    for i in range(0, len(out), 8):
        x ^= x >> 12
        x = (x ^ (x << 25)) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
        k = (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
        for b in range(min(8, len(out) - i)):
            out[i + b] ^= (k >> (8 * b)) & 0xFF
    return bytes(out)


def write_embedded(out_dir):
    """embedded_files.cpp: the EMBED_FILES present in out_dir, scrambled, for the runtime library. None if none."""
    files = [n for n in EMBED_FILES if os.path.isfile(os.path.join(out_dir, n))]
    if not files:
        return None
    # code objects of the extra targets: built into out_dir/<arch>/ by --extra-archs, embedded as "<arch>.<name>"
    src = {n: os.path.join(out_dir, n) for n in files}
    for n in EMBED_FILES:
        if n.endswith(".co"):
            p = os.path.join(out_dir, n)
            if not os.path.isfile(p):
                raise SystemExit(f"missing required {ARCH} code object: {p}")
            src[f"{ARCH}.{n}"] = p
            shutil.copyfile(p, os.path.join(out_dir, f"{ARCH}.{n}"))
    for arch in EXTRA_ARCHS:
        for n in EMBED_FILES:
            p = os.path.join(out_dir, arch, n)
            if n.endswith(".co"):
                if not os.path.isfile(p):
                    raise SystemExit(f"missing required {arch} code object: {p}")
                src[f"{arch}.{n}"] = p
    # the D3D12 kernels (clean/dx12/build_dx12.py: <out>/dx12/manifest.json and its DXIL files; architecture-independent)
    # for executor=dx12 / backend 3: under their own names (wg_read_file finds a file by its base name), so a read of
    # <kernels>/dx12/manifest.json (the device, the runtime's code-object check) gets the built-in manifest
    man = os.path.join(out_dir, "dx12", "manifest.json")
    if os.path.isfile(man):
        src["manifest.json"] = man
        with open(man) as f:
            for k in json.load(f)["kernels"]:
                p = os.path.join(out_dir, "dx12", k["file"])
                if not os.path.isfile(p):
                    raise SystemExit(f"missing D3D12 kernel file {p} (clean/dx12/build_dx12.py)")
                src[k["file"]] = p
    files = list(src)
    lines = ["// Generated by clean/runner/build.py: files built into the runtime library (clean/runtime/embedded.h).",
             "#include <cstddef>", "namespace wg_embedded {",
             "struct File { const char* name; const unsigned char* data; size_t size; };"]
    for i, n in enumerate(files):
        data = scramble(n, open(src[n], "rb").read())
        lines.append(f"static const unsigned char f{i}[] = {{")
        for j in range(0, len(data), 32):
            lines.append(",".join(str(b) for b in data[j:j + 32]) + ",")
        lines.append("};")
    lines.append("extern const File kFiles[] = {" + ", ".join(
        f'{{"{n}", f{i}, {os.path.getsize(src[n])}}}' for i, n in enumerate(files)) + "};")
    lines.append(f"extern const size_t kCount = {len(files)};")
    lines.append("}  // namespace wg_embedded")
    path = os.path.join(out_dir, "embedded_files.cpp")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"embedded into the runtime library: {', '.join(files)}", flush=True)
    return path


def runtime_command(cxx, out, embedded=None):
    """The per-frame runtime (clean/runtime) as a shared library: plain C++17, no HIP headers (HIP is loaded at run
    time). On Windows the clang of the HIP SDK targets the installed Visual Studio / Windows SDK libraries."""
    srcs = [os.path.join(RUNTIME_DIR, s) for s in RUNTIME_SOURCES]
    cmd = [cxx, "-std=c++17", "-O2", "-shared"]
    if sys.platform == "win32":
        srcs += [os.path.join(RUNTIME_DIR, s) for s in INTEROP_SOURCES]
        srcs.append(os.path.join(RUNTIME_DIR, "device_dx12.cpp"))   # hybrid D3D12 device (backend 2, clean/dx12)
        srcs.append(os.path.join(RUNTIME_DIR, "device_d3d12.cpp"))  # native D3D12 device (backend 3, no HIP calls)
        cmd += ["-D_CRT_SECURE_NO_WARNINGS"]
        if "clang" in os.path.basename(cxx).lower():
            cmd += ["-fms-runtime-lib=static"]   # /MT: no Visual C++ redistributable needed (ANSWERS spec 20 item 15)
    else:
        cmd += ["-fPIC", "-fvisibility=hidden"]
    if embedded:  # the generated table of built-in files
        srcs.append(embedded)
        cmd += ["-DWG_EMBEDDED_TABLE"]
    for name, m in OPTIONAL_MODELS.items():
        if not MODELS_BUILT[name]:
            cmd += ["-D" + m["define"]]
    if MODEL_L:
        cmd += ["-DWG_MODEL_L"]
    cmd += srcs
    if RR_OBJECTS and sys.platform == "win32":
        # release 0.6.0 (owner 2026-11-02: one runtime DLL, "just like before"): the RR runtime (clean/rr_f/runtime, built by
        # its own tools/build.py with its own flags: cl /O2 /fp:strict /MT) is linked into wronggpu_runtime.dll, which then
        # exports both the Model M interface and the RR interface (WrongGPU_Rr*); its DllMain is the library's DllMain
        objs = sorted(os.path.join(RR_OBJECTS, n) for n in os.listdir(RR_OBJECTS) if n.lower().endswith(".obj"))
        if not objs:
            raise SystemExit(f"--with-rr: no RR runtime objects in {RR_OBJECTS}")
        print(f"RR runtime linked in: {', '.join(os.path.basename(o) for o in objs)}", flush=True)
        cmd += objs + ["-ld3d12", "-ldxgi", "-lbcrypt", "-lgdi32"]
    cmd += ["-o", out]
    if sys.platform != "win32":
        cmd += ["-ldl"]
    return cmd


RR_OBJECTS = None  # --with-rr: the RR runtime's object files (clean/rr_f/runtime/tools/build.py <out>/obj)


def build_runtime(cxx, out_dir, embed=True):
    for m in [MAP_FILE] + [o["map"] for name, o in OPTIONAL_MODELS.items() if MODELS_BUILT[name]] + \
            ([L_MAP_FILE] if MODEL_L else []):
        shutil.copyfile(m, os.path.join(out_dir, os.path.basename(m)))
    embedded = write_embedded(out_dir) if embed else None
    run(runtime_command(cxx, os.path.join(out_dir, runtime_library_name()), embedded))


def run(cmd):
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"build failed ({r.returncode})")


# Resource gate of post_w64.co (the runtime's default post code object): the default reconstruction kernels must keep
# 8 waves64 per SIMD (LDS <= 16384 bytes per 256-lane workgroup and <= 96 VGPRs; clean/FORMATS.md, R2Texel). Below
# that they lose a quarter of their resident waves and run about 10 % slower (measured in the post-r1-latency work).
# The true-fusion kernels likewise (post_tf_p12 / _f32: <= 96 VGPRs and <= 32768 bytes of LDS per 512-lane workgroup;
# 6 waves64 cost them about 9 %); the native-AA form post_tf_dl* runs at 6 (41472 bytes of LDS). post_tf_*_diet_vsi: the
# runtime's default true-fusion kernels (post_diet + tf_vdiet, post_sel, post_idx).
POST_W64_GATE = {"post_recon2_r_band_p12": 8, "post_recon2_r_p12": 8, "post_tf_p12": 8, "post_tf_f32_p12": 8,
                 "post_tf_p12_diet_vsi": 8, "post_tf_f32_p12_diet_vsi": 8,
                 "post_tf_p12_diet_vsit": 8, "post_tf_f32_p12_diet_vsit": 8,
                 "post_tf_dl_p12": 6, "post_tf_dl_f32_p12": 6}
POST_W64_REPORT = "post_w64.resources.txt"


# Motion vectors at output resolution (spec 03 draft 4, spec 06 section 3; runtime option motion_outres=original): the
# input encoder and the reconstruction kernels of that case in code objects of their own, so that the code objects of
# every other frame stay the files they were. encoder_mo.co (kernels/encoder_mo.hip) and, from kernels/post_mo.hip in
# wave64, one code object per output-filter build of post_w64*.co (the true-fusion kernel contains the filter).
# The reconstruction kernels keep the occupancy of the kernels they stand in for (POST_W64_GATE).
POST_MO_GATE = {"post_tf_p12_mo": 8, "post_tf_f32_p12_mo": 8, "post_recon2_r_p12_mo": 8}
POST_MO_BUILDS = (("post_mo_w64.co", {}), ("post_mo_w64__q5.co", {"WG_LZ_PROFILE": 1}),
                  ("post_mo_w64__orig.co", {"WG_LZ_PROFILE": 2}))


def build_motion_outres(clang, args):
    extra = dict(d.split('=', 1) for d in args.define)
    run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "encoder_mo.hip"),
                       os.path.join(args.out, "encoder_mo.co"), device_libs=not args.no_device_libs,
                       report=args.report, soft_fp8=args.soft_fp8, defines=extra))
    for name, defines in POST_MO_BUILDS:
        run_post_w64(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "post_mo.hip"),
                                    os.path.join(args.out, name), device_libs=not args.no_device_libs, report=True,
                                    soft_fp8=args.soft_fp8, defines={**extra, **defines},
                                    extra=["-mwavefrontsize64"]),
                     args.out, gate=not (args.no_resource_gate or args.define),
                     report_name=os.path.splitext(name)[0] + ".resources.txt", gates=POST_MO_GATE)


def resource_report(text):
    """{kernel: {field: int}} from clang's -Rpass-analysis=kernel-resource-usage remarks (e.g. "VGPRs",
    "Occupancy [waves/SIMD]", "LDS Size [bytes/block]")."""
    out, cur = {}, None
    for line in text.splitlines():
        if "remark:" not in line:
            continue
        body = line.split("remark:", 1)[1].split("[-Rpass-analysis", 1)[0].strip()
        if body.startswith("Function Name:"):
            cur = out.setdefault(body.split(":", 1)[1].strip(), {})
        elif cur is not None and ":" in body:
            k, v = body.rsplit(":", 1)
            try:
                cur[k.strip()] = int(v.strip())
            except ValueError:
                cur[k.strip()] = v.strip()
    return out


def run_post_w64(cmd, out_dir, gate=True, report_name=POST_W64_REPORT, gates=None):
    """Build post_w64.co with the resource report (written to POST_W64_REPORT) and check POST_W64_GATE (or `gates`)."""
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    with open(os.path.join(out_dir, report_name), "w") as f:
        f.write(r.stderr)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"build failed ({r.returncode})")
    rep = resource_report(r.stderr)
    bad = []
    for k, occ in (POST_W64_GATE if gates is None else gates).items():
        got = rep.get(k, {}).get("Occupancy [waves/SIMD]")
        print(f"  {k}: VGPRs {rep.get(k, {}).get('VGPRs')}, LDS {rep.get(k, {}).get('LDS Size [bytes/block]')}, "
              f"occupancy {got} waves64/SIMD", flush=True)
        if got is None or got < occ:
            bad.append(f"{k}: occupancy {got} < {occ}")
    if bad and gate:
        raise SystemExit("post_w64 resource gate failed (" + "; ".join(bad) + "); see " +
                         os.path.join(out_dir, report_name) + " (--no-resource-gate to build anyway)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""), help="ROCm / HIP SDK directory")
    ap.add_argument("--out", required=True, help="output directory for the code objects")
    ap.add_argument("--clang", help="clang executable (default: from --rocm, then PATH)")
    ap.add_argument("--no-device-libs", action="store_true", help="add -nogpulib instead of the ROCm device libs")
    ap.add_argument("--report", action="store_true", help="print VGPR/spill/occupancy per kernel")
    ap.add_argument("--soft-fp8", action="store_true",
                    help="software FP8 conversions instead of v_cvt_pk_fp8_f32 / v_cvt_f32_fp8 (fallback)")
    ap.add_argument("--define", action="append", default=[], metavar="NAME=VALUE",
                    help="extra -D for the base kernels (experiments), e.g. --define WG_GELU_MODE=1")
    ap.add_argument("--variants", nargs="?", const="all", default=None,
                    help="also build kernel variants (comma-separated names from variants.py, or 'all') as "
                         "swin__<name>.co and write variants.json; time them with sweep.py")
    ap.add_argument("--ablations", action="store_true",
                    help="also build the timing-only ablation variants (output is wrong on purpose; see variants.py)")
    ap.add_argument("--perf-variants", action="store_true",
                    help="also build the enc0 / dec0 / exposure variants and ablations of perf_variants.py "
                         "(<kernel>__<variant>.co, perf_variants.json); time them with perf_bench.py")
    ap.add_argument("--profile", action="store_true",
                    help="also build the instrumented kernels prof / prof2 (per-phase timing; see profile_stages.py)")
    ap.add_argument("--payload", action="store_true",
                    help="in-game A/B payload: same as --variants all --perf-variants (every candidate code object, "
                         "variants.json and perf_variants.json next to the runtime library)")
    ap.add_argument("--emu", action="store_true", help="also build the host emulator library (Linux)")
    ap.add_argument("--emu-only", action="store_true", help="only build the host emulator library")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "clang++"), help="host C++ compiler for --emu")
    ap.add_argument("--no-runtime", action="store_true", help="do not build the per-frame runtime library")
    ap.add_argument("--no-resource-gate", action="store_true",
                    help="build even if post_w64.co misses its occupancy gate (POST_W64_GATE)")
    ap.add_argument("--runtime-only", action="store_true", help="only build the per-frame runtime library")
    ap.add_argument("--runtime-cxx", help="host C++ compiler for the runtime (default: the HIP SDK's clang++)")
    ap.add_argument("--extra-archs", default="",
                    help="comma-separated further GPU targets (e.g. gfx1200 for the RX 9060 family): their default code "
                         "objects are built into <out>/<arch>/ and embedded next to the primary ones")
    ap.add_argument("--no-embed", action="store_true",
                    help="do not build the code objects and data into the runtime library (embedded.h)")
    ap.add_argument("--motion-outres-only", action="store_true",
                    help="only build the code objects of motion vectors at output resolution (encoder_mo.co, "
                         "post_mo_w64*.co; with --extra-archs for those targets too); no runtime library")
    ap.add_argument("--with-rr", metavar="OBJ_DIR",
                    help="link the RR runtime's object files (clean/rr_f/runtime/tools/build.py <out>/obj) into the runtime "
                         "library: one wronggpu_runtime.dll with the Model M and the RR interfaces (release 0.6.0)")
    for name, m in OPTIONAL_MODELS.items():  # --no-model-k, ...
        ap.add_argument("--no-model-" + name.lower(), action="store_true", help=m["help"])
    ap.add_argument("--model-l", action="store_true",
                    help="Model L in the runtime library (#89; off by default): its runtime sources with WG_MODEL_L, its "
                         "code objects and its map built in (with --runtime-only and --rocm its code objects are built "
                         "first, for every target); its D3D12 kernels come from clean/dx12/build_dx12.py --model-l")
    args = ap.parse_args(argv)
    for name in OPTIONAL_MODELS:
        set_model(name, not getattr(args, "no_model_" + name.lower()))
    set_model_l(args.model_l)
    global RR_OBJECTS
    RR_OBJECTS = args.with_rr
    if args.payload:
        args.variants = args.variants or "all"
        args.perf_variants = True

    os.makedirs(args.out, exist_ok=True)
    EXTRA_ARCHS[:] = [a for a in args.extra_archs.split(",") if a and a != ARCH]
    if any(a not in ("gfx1200", "gfx1201") for a in [ARCH] + EXTRA_ARCHS):
        raise SystemExit("supported release targets are gfx1200 and gfx1201 only")
    if args.runtime_only:
        if MODEL_L and args.rocm:  # Model L's code objects next to the given ones (release builds reuse pinned ones)
            clang = args.clang or find_clang(args.rocm)
            build_device_l(clang, args.rocm, args.out, device_libs=not args.no_device_libs)
            for arch in EXTRA_ARCHS:
                build_device_l(clang, args.rocm, os.path.join(args.out, arch), device_libs=not args.no_device_libs,
                               arch=arch)
        build_runtime(args.runtime_cxx or find_host_cxx(args.rocm), args.out, embed=not args.no_embed)
        return
    if args.motion_outres_only:
        if not args.rocm and not args.no_device_libs:
            raise SystemExit("--rocm is required (or --no-device-libs for a compile check)")
        build_motion_outres(args.clang or find_clang(args.rocm), args)
        for arch in EXTRA_ARCHS:
            sub = [sys.executable, os.path.abspath(__file__), "--out", os.path.join(args.out, arch),
                   "--motion-outres-only"]
            sub += (["--rocm", args.rocm] if args.rocm else []) + (["--clang", args.clang] if args.clang else [])
            sub += (["--no-device-libs"] if args.no_device_libs else []) + (["--soft-fp8"] if args.soft_fp8 else [])
            sub += ["--no-resource-gate"] if args.no_resource_gate else []
            # as the ordinary branch's child: the defines of an experiment build (without them the child builds the
            # kernels the experiment changes unchanged, and runs the resource gate that the parent skips for defines)
            sub += ["--report"] if args.report else []
            for d in args.define:
                sub += ["--define", d]
            print(f"-- motion-at-output-resolution code objects for {arch}", flush=True)
            if subprocess.run(sub, env={**os.environ, "WG_GPU_ARCH": arch}).returncode != 0:
                raise SystemExit(f"build for {arch} failed")
        return
    if not args.emu_only:
        if not args.rocm and not args.no_device_libs:
            raise SystemExit("--rocm is required (or --no-device-libs for a compile check)")
        clang = args.clang or find_clang(args.rocm)
        for src in SOURCES:
            out = os.path.join(args.out, os.path.splitext(src)[0] + ".co")
            run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, src), out,
                               device_libs=not args.no_device_libs, report=args.report, soft_fp8=args.soft_fp8,
                               defines=dict(d.split('=', 1) for d in args.define)))
        for name, m in OPTIONAL_MODELS.items():  # the optional models' kernels (model K: clean/k)
            if MODELS_BUILT[name]:
                m["build_device"](clang, args.rocm, args.out, device_libs=not args.no_device_libs, report=args.report)
        if MODEL_L:  # Model L's kernels (clean/l/kernels; off by default)
            build_device_l(clang, args.rocm, args.out, device_libs=not args.no_device_libs, report=args.report)
        # the post kernels in wave64 as well (the runtime's default post code object; post.co for WG_POST_WAVE=32)
        # (always with the resource report: POST_W64_GATE; skipped for experiment builds with --define)
        run_post_w64(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "post.hip"),
                                    os.path.join(args.out, "post_w64.co"), device_libs=not args.no_device_libs,
                                    report=True, soft_fp8=args.soft_fp8,
                                    defines=dict(d.split('=', 1) for d in args.define), extra=["-mwavefrontsize64"]),
                     args.out, gate=not (args.no_resource_gate or args.define))
        # the wave64 post kernels with the quality5 output-filter weights (runtime option output_filter=quality5):
        # the same kernels with other constants, so the same resource gate
        run_post_w64(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "post.hip"),
                                    os.path.join(args.out, "post_w64__q5.co"), device_libs=not args.no_device_libs,
                                    report=True, soft_fp8=args.soft_fp8,
                                    defines={**dict(d.split('=', 1) for d in args.define), "WG_LZ_PROFILE": 1},
                                    extra=["-mwavefrontsize64"]),
                     args.out, gate=not (args.no_resource_gate or args.define), report_name="post_w64__q5.resources.txt")
        # the wave64 post kernels of the output filter "original" (spec 06 draft 4; the runtime's default when the model
        # file has the weights, runtime option output_filter=original): the five weights are kernel arguments and the
        # filter accumulates in FP16 steps (WG_LZ_PROFILE=2). The same kernels, so the same resource gate
        run_post_w64(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "post.hip"),
                                    os.path.join(args.out, "post_w64__orig.co"), device_libs=not args.no_device_libs,
                                    report=True, soft_fp8=args.soft_fp8,
                                    defines={**dict(d.split('=', 1) for d in args.define), "WG_LZ_PROFILE": 2},
                                    extra=["-mwavefrontsize64"]),
                     args.out, gate=not (args.no_resource_gate or args.define),
                     report_name="post_w64__orig.resources.txt")
        build_motion_outres(clang, args)  # motion vectors at output resolution: encoder_mo.co, post_mo_w64*.co
        # dec0 writing the pack12 coefficients: the runtime's default together with the _p12 post kernels
        # (bit-identical; decoder.co stays for the frames the _p12 kernels do not take)
        src, defines, _ = PV.VARIANTS["dec0"]["pack12"]
        run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, src),
                           os.path.join(args.out, PV.code_object("dec0", "pack12")),
                           device_libs=not args.no_device_libs, soft_fp8=args.soft_fp8,
                           defines={**defines, **dict(d.split('=', 1) for d in args.define)}))
        # dec1 + dec0 fused, writing the pack12 coefficients (the runtime's default with the _p12 post kernels)
        run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "dec1_dec0.hip"),
                           os.path.join(args.out, "dec1_dec0__pack12.co"),
                           device_libs=not args.no_device_libs, soft_fp8=args.soft_fp8,
                           defines={**defines, **dict(d.split('=', 1) for d in args.define)}))
        # the same two with dec0's divisions switched off for A/B (bit-identical; not embedded): <name>__div<r><b>.co,
        # r: WG_DEC_RDIV=0 (runtime option dec_rdiv=0, the divisions as written), b: WG_DEC_BDIV=0 (dec_bdiv=0, the B
        # softmax not by the shared rcp(s))
        for sfx, dd in (("r", {"WG_DEC_RDIV": 0}), ("b", {"WG_DEC_BDIV": 0}),
                        ("rb", {"WG_DEC_RDIV": 0, "WG_DEC_BDIV": 0})):
            for s_, co in (("decoder.hip", "decoder__pack12"), ("dec1_dec0.hip", "dec1_dec0__pack12")):
                run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, s_),
                                   os.path.join(args.out, f"{co}__div{sfx}.co"), device_libs=not args.no_device_libs,
                                   soft_fp8=args.soft_fp8, defines={**defines, **dd}))
        # base, and the bottleneck megakernel build the runtime uses for b0..b5 by default (bit-identical to base)
        names = ["base", "bmegahg"]
        if args.variants:
            names += [n for n in (V.VARIANTS if args.variants == "all" else args.variants.split(","))
                      if n not in names]
        if args.ablations:
            names += list(V.ABLATIONS)
        if args.profile:
            names += list(V.PROFILES)
        manifest = {}
        for name in names:
            if name not in V.all_builds():
                raise SystemExit(f"unknown variant {name!r}; known: {', '.join(V.all_builds())}")
            if name != "base":
                run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, V.source(name)),
                                   os.path.join(args.out, V.code_object(name)),
                                   device_libs=not args.no_device_libs, report=args.report and V.is_v2(name),
                                   soft_fp8=args.soft_fp8,
                                   defines={**dict(d.split('=', 1) for d in args.define), **V.all_builds()[name]}))
            manifest[name] = {"file": V.code_object(name), "defines": V.all_builds().get(name, {}),
                              "geometry": V.variant_geometry(name), "ablation": V.is_ablation(name),
                              "source": V.source(name), "profile": V.is_profile(name)}
        with open(os.path.join(args.out, V.MANIFEST), "w") as f:
            json.dump(manifest, f, indent=1)
        if args.perf_variants:
            pmanifest = {}
            for stage, variants in PV.VARIANTS.items():
                for name, (src, defines, ablation) in variants.items():
                    co = PV.code_object(stage, name)
                    if co != PV.STAGES[stage][1]:  # the default code object is built above
                        run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, src), os.path.join(args.out, co),
                                           device_libs=not args.no_device_libs, soft_fp8=args.soft_fp8,
                                           defines=defines))
                    pmanifest.setdefault(stage, {})[name] = {"file": co, "source": src, "defines": defines,
                                                             "ablation": ablation}
            for name, (co, defines) in PV.FUSION.items():  # fused dec1 + dec0 builds other than the default
                if defines:
                    run(device_command(clang, args.rocm, os.path.join(KERNEL_DIR, "dec1_dec0.hip"),
                                       os.path.join(args.out, co), device_libs=not args.no_device_libs,
                                       soft_fp8=args.soft_fp8, defines=defines))
                pmanifest.setdefault("dec1dec0", {})[name] = {"file": co, "source": "dec1_dec0.hip",
                                                              "defines": defines, "ablation": False}
            with open(os.path.join(args.out, PV.MANIFEST), "w") as f:
                json.dump(pmanifest, f, indent=1)
    if EXTRA_ARCHS and not args.emu_only:  # the same default code objects for the further targets
        for arch in EXTRA_ARCHS:
            sub = [sys.executable, os.path.abspath(__file__), "--out", os.path.join(args.out, arch), "--no-runtime"]
            sub += (["--rocm", args.rocm] if args.rocm else []) + (["--clang", args.clang] if args.clang else [])
            sub += (["--no-device-libs"] if args.no_device_libs else []) + (["--soft-fp8"] if args.soft_fp8 else [])
            sub += ["--no-model-" + n.lower() for n in OPTIONAL_MODELS if getattr(args, "no_model_" + n.lower())]
            sub += ["--model-l"] if MODEL_L else []
            for d in args.define:
                sub += ["--define", d]
            print(f"-- code objects for {arch}", flush=True)
            r = subprocess.run(sub, env={**os.environ, "WG_GPU_ARCH": arch})
            if r.returncode != 0:
                raise SystemExit(f"build for {arch} failed ({r.returncode})")
    if args.emu or args.emu_only:
        run(emu_command(args.cxx, os.path.join(args.out, "wg_emu.so")))
        if args.variants:  # emulator libraries of the requested variants (Runner(..., backend="emu", variant=name))
            names = [n for n in (V.VARIANTS if args.variants == "all" else args.variants.split(",")) if n != "base"]
            for name in names:
                run(emu_command(args.cxx, os.path.join(args.out, f"wg_emu__{name}.so"), V.all_builds()[name],
                                V.source(name)))
            if args.emu_only:  # the runner reads the variants' launch geometry from the manifest
                with open(os.path.join(args.out, V.MANIFEST), "w") as f:
                    json.dump({n: {"file": V.code_object(n), "defines": V.all_builds().get(n, {}),
                                   "geometry": V.variant_geometry(n), "ablation": V.is_ablation(n),
                                   "profile": V.is_profile(n), "source": V.source(n)} for n in ["base"] + names},
                              f, indent=1)
    if not args.emu_only and not args.no_runtime:
        build_runtime(args.runtime_cxx or find_host_cxx(args.rocm), args.out, embed=not args.no_embed)


if __name__ == "__main__":
    main()
