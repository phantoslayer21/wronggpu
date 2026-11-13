# WrongGPU on Linux: D3D12 tests under Proton

Players on Linux run WrongGPU's Windows DLLs inside Proton. There, Direct3D 12 is vkd3d-proton on top of RADV,
`d3dcompiler_47.dll` is Wine's own (vkd3d-shader), and the driver channel that the D3D12 kernels use for WMMA
(`clean/dx12/hlsl/wg_ags.hlsli`) is translated by vkd3d-proton's dxil-spirv instead of AMD's driver. These tests run
WrongGPU's D3D12 code on that stack, so a release can be checked for Linux players before it ships.

```sh
python3 clean/linux/run_wine_tests.py --proton <Proton install> [--vkd3d <dir>] [--runtime <wronggpu_runtime.dll>]
```

The tests use WrongGPU's own synthetic data only: no model converted from a DLSS file, no game files.

## What it checks

| Test | What runs | What it would catch |
|---|---|---|
| `interop` | `clean/runtime/interop/tests/run-tests.sh` (31 tests): MinGW build of the runtime with the `d3d12-test` executor | interop shaders that Wine's `d3dcompiler_47` rejects (`globallycoherent` in `PASS_SPIN` failed all 31), copy / convert / split regressions on vkd3d-proton |
| `rr-leak` | the RR runtime test host's `tagloop` job against `--runtime`, with a synthetic RR model (`pack_model --synthetic 5100`) | RR input copies retained and released every frame with no RR instance piling up in video memory (+2,635 MiB in 200 frames before the fix in `release_transient_locked`, +37 MiB after) |
| `ags-layout` | `ags_layout.hlsl`: `wg_ags_mma` (FP16, FP8), `wg_ags_convert` (F32 accumulator to F16 accumulator and FP8 B, FP8 B to F32 for all 256 codes), the F16 `FloatConversion` call, a wave64 conversion | a translation of the driver channel that does not keep the gfx12 register semantics the kernels rely on; checked against a CPU reference (multiplies within 1e-5, everything else bit-exact) |

The `--runtime` DLL is the one to be released: the MSVC release build works as it is. `interop` builds its own
development runtime with MinGW, and the hosts of `rr-leak` and `ags-layout` are MinGW builds too. They are test
programs only, never shipped (the release rule in `clean/fsr/build-mingw.sh`).

## What you need

- Linux with an AMD RDNA 4 GPU (RX 9070 or RX 9060 family) and Mesa's RADV driver.
- A Proton install: `--proton`, or the environment variable `WG_PROTON`. For example, a GE-Proton folder in
  `~/.local/share/Steam/compatibilitytools.d/`, or Steam's `steamapps/common/Proton - Experimental`.
- A MinGW-w64 C++ compiler: `MINGW_CXX`, otherwise `x86_64-w64-mingw32-g++-posix` or `x86_64-w64-mingw32-g++`
  on `PATH`. llvm-mingw works too. The tests add `-fms-extensions` for clang.
- For `ags-layout`: the Linux `dxc` from the DXC release named in BUILDING.md, in `WG_DXC` or on `PATH`.
- Python 3 with NumPy.

## vkd3d-proton

`rr-leak` and `ags-layout` need the WMMA channel. Released vkd3d-proton cannot translate it yet: dxil-spirv accepts only
WMMA operands that come from AGS matrix loads, and WrongGPU's kernels pack their fragments themselves. The support is in
https://github.com/HansKristian-Work/dxil-spirv/pull/362. Until a Proton release ships it, build vkd3d-proton with
that change. Use the vkd3d-proton commit your Proton ships, for example `0bb64c5` in GE-Proton 11-11. Then pass the
folder that holds the resulting `d3d12.dll` and `d3d12core.dll` with `--vkd3d`. Without it these two tests fail at
`CreateComputePipelineState` (`0x80070057`). On the player's side, the same patched DLLs next to the game exe make
Model M and Ray Reconstruction run.

## Reading the result

Each test prints `PASS`, `FAIL` or `SKIP` with a reason, and a summary follows at the end. The exit status is 1 when a
test failed. Everything a test writes goes to `--work`, by default `build/linux-tests/`:

- `interop/result.txt`: the 31 tests' output;
- `rr-leak/host.log`, and `rr-leak/stage/wronggpu-rr.log` for the runtime's own log;
- `ags-layout/host.log`, `inputs.bin` and `outputs.bin`.

Each failure was checked to show up in this suite. Run on the unfixed `convert.cpp`, `interop` fails 29 of 31 with
`Unknown modifier "globallycoherent"`. With the 0.6.0 RR runtime, `rr-leak` reports +2,635 MiB. On stock
vkd3d-proton, `ags-layout` stops at `CreateComputePipelineState`. Fed results with lanes in the wrong order or F16
halves swapped, its comparison fails.
