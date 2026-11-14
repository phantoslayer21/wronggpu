# SPDX-License-Identifier: MIT
"""The runtime's frame sequence on the CPU (#89): clean/tests/runtime_frame_trace.cpp runs the runtime
(clean/runtime/runtime.cpp) on a recording device, so whole frames run without a GPU and without HIP. Launches are
recorded, not executed: the test checks the sequence (code objects loaded, kernels launched per frame with grid and
block), not pixels. The base for checking other models' frames against Model M's (Model L: Model M's frame with the
encoder and decoder steps replaced).

- a synthetic Model M (converter raw layout): create, the code objects loaded, the launches of four frames;
- a full build's optional code objects (names only) select the default paths: pack12 decoder, dec1 fused with dec0,
  wave64 post;
- the same run twice gives the same trace;
- the recorder's arguments: the encoder's features are enc1's input and dec1's output the decoder's (same buffer),
  every one large enough for its map; the render size, the padded grid, reset and parity in the encoder's and
  decoder's parameters, with the render size changing between frames (aligned and non-aligned) and a game reset;
- a launch that fails stops its frame there (nothing after it), and the next frame runs whole;
- model=L is refused at create for now.

The runtime options of the environment (WG_*) are removed for the runs. Built with the host C++ compiler
(clean/tests/host_cxx.py; skipped without one).

Run: python -m unittest clean.tests.test_runtime_frame_trace
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from clean.runner import synthetic_model as SM
from clean.tests import host_cxx

SOURCES = [os.path.join("clean", "tests", "runtime_frame_trace.cpp")]
SOURCES += [os.path.join("clean", "runtime", s) for s in ("device.cpp", "embedded.cpp", "model.cpp", "runtime.cpp",
                                                          "no_model_k.cpp")]
SOURCES += [os.path.join("clean", "converter", "src", s) for s in ("sha256.cpp", "json.cpp")]
_EXE = None
_MODEL = None


def trace_exe():
    global _EXE
    if _EXE is None:
        _EXE = host_cxx.build("runtime_frame_trace", SOURCES, ["WG_NO_MODEL_K"])
    return _EXE


def model_dir():
    global _MODEL
    if _MODEL is None:
        d = tempfile.mkdtemp(prefix="wg_ft_")
        SM.write_raw(d, SM.synthetic_tensors(1))
        _MODEL = d
    return _MODEL


# the optional code objects of a full build (clean/runner/build.py) that select the runtime's default paths
SHIPPING = ("decoder__pack12.co", "dec1_dec0__pack12.co", "post_w64.co")


def run_trace(rw=64, rh=64, ow=96, oh=96, frames=3, options="", build_files=(), flags=()):
    """The program's output lines. The build folder holds empty files of the names in build_files (the recording
    device reads no code object; the runtime chooses its paths by the names present). flags: the program's --flags."""
    build = tempfile.mkdtemp(prefix="wg_ftb_")
    for n in build_files:
        open(os.path.join(build, n), "wb").close()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("WG_")}
    try:
        r = subprocess.run([trace_exe(), build, model_dir(), str(rw), str(rh), str(ow), str(oh), str(frames), options,
                            *flags], capture_output=True, text=True, env=env, timeout=600)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if r.returncode != 0:
        raise AssertionError("runtime_frame_trace failed (exit %d)\n%s%s" % (r.returncode, r.stdout, r.stderr))
    return r.stdout.splitlines()


class Trace:
    """A parsed run: create (the first line), allocs {k: bytes}, loads, frames [{"info": {...}, "launches": [Launch],
    "failed": kernel or None, "error": message or None}], kernels {key: value}."""

    def __init__(self, lines):
        self.create, self.allocs, self.loads, self.frames, self.kernels = lines[0], {}, [], [], {}
        for ln in lines[1:]:
            if ln.startswith("alloc b"):
                k, n = ln[7:].split(" ")
                self.allocs[int(k)] = int(n)
            elif ln.startswith("load "):
                self.loads.append(ln[5:])
            elif ln.startswith("frame "):
                f = ln.split(" ")
                info = {f[i]: int(f[i + 1]) for i in range(2, len(f) - 1, 2)}
                self.frames.append({"info": info, "launches": [], "failed": None, "error": None})
            elif ln.startswith("launch "):
                self.frames[-1]["launches"].append(Launch(ln[7:]))
            elif ln.startswith("failed "):
                self.frames[-1]["failed"] = ln[7:]
            elif ln.startswith("ERR frame "):
                self.frames[-1]["error"] = ln.split(": ", 1)[1]
            elif ln.startswith("kernels "):
                self.kernels = dict(kv.split("=", 1) for kv in ln[8:].split(" ") if "=" in kv)
            else:
                raise AssertionError("unexpected line: " + ln)

    def extent(self, addr):
        """Bytes from a decoded address b<k>+<offset> to the end of its allocation."""
        k, off = addr[1:].split("+")
        return self.allocs[int(k)] - int(off)


class Launch:
    """One launch line: kernel, grid, block and, for the kernels the recorder decodes, args (strings: b<k>+<offset>,
    0, an int, E{...} / D{...})."""

    def __init__(self, text):
        head, _, tail = text.partition(" | ")
        self.kernel, g, b = head.split(" ")
        self.grid, self.block = int(g), int(b)
        self.args = tail.split(" ") if tail else []

    def kgb(self):
        return (self.kernel, self.grid, self.block)

    def params(self, i):
        """The values of E{...} / D{...} at argument i, as ints."""
        return [int(v) for v in self.args[i][2:-1].split(",")]


def parse(lines):
    """(create line, [loaded code objects], [[(kernel, grid, block)] per frame], kernels line)."""
    t = Trace(lines)
    return t.create, t.loads, [[ln.kgb() for ln in f["launches"]] for f in t.frames], t.kernels


def ceil32(v):
    return -(-v // 32) * 32


class ModelM(unittest.TestCase):
    def test_frames(self):
        create, loads, frames, kernels = parse(run_trace(frames=4))
        self.assertEqual(create, "create ok model M")
        for key in ("swin", "enc0", "dec0"):
            self.assertIn(kernels[key], loads)
        self.assertEqual(len(frames), 4)
        for f in frames:
            names = [k for k, _, _ in f]
            self.assertGreater(len(names), 10)
            self.assertTrue(all(g > 0 and b > 0 for _, g, b in f))
            self.assertEqual(names[0], "exposure_pair_diet")  # exposure first, then the encoder
            self.assertIn(kernels["post"], names)
        # the window offsets alternate from frame to frame (grids of the Swin stages change), so the launches repeat
        # every second frame
        self.assertNotEqual(frames[0], frames[1])
        self.assertEqual(frames[0], frames[2])
        self.assertEqual(frames[1], frames[3])

    def test_shipping_build(self):
        """With the optional code objects of a full build: the pack12 decoder, dec1 fused with dec0, wave64 post."""
        create, loads, frames, kernels = parse(run_trace(frames=2, build_files=SHIPPING))
        self.assertEqual(create, "create ok model M")
        for want in ("decoder__pack12.co as d0p12", "dec1_dec0__pack12.co as d1p12", "post_w64.co as w64"):
            self.assertIn(want, loads)
        for f in frames:
            names = [k for k, _, _ in f]
            self.assertIn("d1p12/swin_block_c64_add_dec0_nc", names)
            self.assertNotIn("coef_decoder", names)
            self.assertTrue(any(n.startswith("w64/") for n in names))

    def test_repeatable(self):
        self.assertEqual(run_trace(rw=80, rh=48, ow=120, oh=72), run_trace(rw=80, rh=48, ow=120, oh=72))

    def test_dataflow_and_extents(self):
        """The encoder's features are what enc1's first block reads, dec1's output what the decoder reads; every buffer
        holds the map it gets (allocation sizes from the recorder)."""
        for rw, rh in ((64, 64), (80, 48)):
            t = Trace(run_trace(rw=rw, rh=rh, ow=96, oh=96, frames=2))
            fw, fh = ceil32(rw), ceil32(rh)
            tokens = (fw // 2) * (fh // 2)
            for f in t.frames:
                ls = f["launches"]
                enc = next(i for i, ln in enumerate(ls) if ln.kernel == "input_encoder")
                dec = next(i for i, ln in enumerate(ls) if ln.kernel == "coef_decoder")
                feat = ls[enc].args[6]
                self.assertEqual(ls[enc + 1].kernel.startswith("swin_block_c64"), True, ls[enc + 1].kernel)
                self.assertEqual(ls[enc + 1].args[0], feat)  # enc1 reads enc0's features
                self.assertGreaterEqual(t.extent(feat), tokens * 64)
                dec1 = [ln for ln in ls[:dec] if ln.kernel.startswith("swin_block_c64")][-1]
                self.assertEqual(ls[dec].args[0], dec1.args[2])  # the decoder reads dec1's output
                self.assertGreaterEqual(t.extent(dec1.args[2]), tokens * 64)
                self.assertEqual(ls[enc].params(12)[:4], [rw, rh, fw, fh])
                self.assertEqual(ls[dec].params(8)[:4], [rw, rh, fw, fh])
                self.assertEqual(dec1.args[4:6], [str(fh // 2), str(fw // 2)])  # H, W of dec1: the F/2 grid
            # the history between frames: the latent the decoder writes (its argument 6) and the near motion the
            # encoder writes (its argument 7) are what the next frame's encoder reads (its arguments 4 and 5)
            for a, b in zip(t.frames, t.frames[1:]):
                prev = {ln.kernel: ln for ln in a["launches"]}
                cur = {ln.kernel: ln for ln in b["launches"]}
                self.assertEqual(cur["input_encoder"].args[4], prev["coef_decoder"].args[6])
                self.assertEqual(cur["input_encoder"].args[5], prev["input_encoder"].args[7])
                self.assertNotEqual(cur["input_encoder"].args[4], prev["input_encoder"].args[4])

    def test_reset_parity_and_sizes(self):
        """Reset (first frame, the game's reset, a size change) and parity reach the encoder and decoder; the render
        size can change from frame to frame (non-aligned sizes included)."""
        sizes = "--sizes=96x96,96x96,96x96,80x48,80x48,64x64"
        t = Trace(run_trace(rw=96, rh=96, ow=144, oh=144, frames=6, flags=(sizes, "--reset=2")))
        resets = [f["info"]["reset"] for f in t.frames]
        self.assertEqual(resets, [1, 0, 1, 1, 0, 1])  # first frame, the game's reset, two size changes
        parities = [f["info"]["parity"] for f in t.frames]
        self.assertTrue(all(p in (0, 1) for p in parities) and parities[0] != parities[1])
        for f, (rw, rh) in zip(t.frames, [(96, 96)] * 3 + [(80, 48)] * 2 + [(64, 64)]):
            self.assertEqual((f["info"]["rw"], f["info"]["rh"]), (rw, rh))
            ls = {ln.kernel: ln for ln in f["launches"]}
            e, d = ls["input_encoder"].params(12), ls["coef_decoder"].params(8)
            self.assertEqual(e[:4], [rw, rh, ceil32(rw), ceil32(rh)])
            self.assertEqual(d[:4], [rw, rh, ceil32(rw), ceil32(rh)])
            self.assertEqual(e[5], f["info"]["reset"])
            self.assertEqual(d[4], f["info"]["reset"])

    def test_failed_launch_stops_the_frame(self):
        t = Trace(run_trace(frames=3, flags=("--fail=coef_decoder@1",)))
        ok0, bad, ok2 = t.frames
        self.assertIsNone(ok0["error"])
        self.assertEqual(bad["failed"], "coef_decoder")
        self.assertIn("injected launch failure: coef_decoder", bad["error"])
        names = [ln.kernel for ln in bad["launches"]]
        self.assertNotIn("coef_decoder", names)
        self.assertEqual(names, [ln.kernel for ln in ok0["launches"]][:len(names)])  # nothing after the failure
        self.assertTrue(names and names[-1].startswith("swin_block_c64"))  # it stopped right after dec1
        self.assertIsNone(ok2["error"])
        self.assertIn("coef_decoder", [ln.kernel for ln in ok2["launches"]])  # the next frame runs whole

    def test_model_l_refused(self):
        lines = run_trace(options="model=L")
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("ERR create: ") and "Model L" in lines[0], lines[0])


if __name__ == "__main__":
    unittest.main()
