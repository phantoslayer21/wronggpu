# Building WrongGPU

From a clean Windows install to a built launcher (`WrongGPU.exe`) with its embedded payload. Everything below runs on
the CPU: no build or unit-test step needs a GPU. Steps that do use a GPU (device tests, timing, game tests) are listed
at the end and are optional.

## 1. Tools

| Tool | Version used for 0.6.0 | Used for |
|---|---|---|
| Windows | 10 or 11, x64 | |
| Visual Studio 2026 (Community is enough), workload "Desktop development with C++" | 18.x; MSVC 19.51 (toolset 14.51); its bundled CMake 4.3 and Ninja 1.13 | every C++ build |
| Windows SDK | 10.0.26100 | D3D12 / DXGI headers and libraries, `dxcapi.h` |
| Python | 3.12 (64-bit) | build scripts, reference, tests |
| NumPy | 2.x (2.5 tested) | reference and tests (`python -m pip install numpy`) |
| AMD HIP SDK for Windows | 7.2 (ROCm clang 21) | HIP kernels for gfx1201 / gfx1200 and the runtime library |
| Microsoft DirectX Shader Compiler (DXC) | v1.10.2605.24 (the release zip: `dxc.exe`, `dxv.exe`, `dxcompiler.dll`) | Direct3D 12 kernels |
| AMD Radeon GPU Analyzer (RGA) | 2.15 | the Direct3D 12 kernels' register check |
| Git | any recent | |

Optional: Pillow (two research tools), MinGW-w64 GCC 13 and Wine 9 (development builds and Linux tests; never used for
a release).

The exact DXC and HIP SDK versions matter if you want the kernels to come out bit-identical to a release (see 6).

## 2. Where the build finds the tools

No path of the developer's PC is written in the repository. The tools are found through environment variables, with
these defaults:

| Variable | Meaning | Default |
|---|---|---|
| `WG_TOOLCHAINS` | folder that holds `dxc-v1.10.2605.24\` (the DXC release zip, unpacked) and `rga-2.15\` (the RGA zip, unpacked) | `<repository>\toolchains` (ignored by git) |
| `WG_DXC_DIR` | DXC's `bin\x64` folder (overrides `WG_TOOLCHAINS`) | `%WG_TOOLCHAINS%\dxc-v1.10.2605.24\bin\x64` |
| `WG_RGA` | `rga.exe` (overrides `WG_TOOLCHAINS`) | `%WG_TOOLCHAINS%\rga-2.15\rga.exe` |
| `ROCM_PATH` | HIP SDK folder. Scripts with a `--rocm` argument default to it (the argument wins); the others read it | `C:\Program Files\AMD\ROCm\7.2` |
| `VSDEVCMD` | Visual Studio's `VsDevCmd.bat` (the kernel scripts build two small helper tools); for another edition e.g. `C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\Tools\VsDevCmd.bat` (or `...\18\Enterprise\...`) | `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat` |
| `CXX`, `WG_K_RT_CXX`, `WG_K_RT_BUILD`, `WG_CONVERTER_EXE` | Model K table tests (`clean\k\tests\test_k_runtime_tables`, `test_k_runtime_pack`): the compiler is `%CXX%`, else clang++ / g++ on PATH, else the HIP SDK's clang++, else MSVC (`WG_K_RT_CXX=msvc` picks MSVC first); their build folder; an already built converter. With no compiler found the tests skip themselves with that reason | temporary folder; converter built from source |
| `WG_BUILD_ROOT` | default output root of the scripts that have one | `<repository>\build` (ignored by git) |
| `WG_DX12_JOBS`, `WG_RR_JOBS` | parallel kernel compiles | number of CPUs (max 12) / 8 |
| `WG_GPU_LOCK` | optional: a script run as `python <lock> --timeout <s> -- <command>` around GPU jobs | unset: GPU jobs run directly |
| `WRONGGPU_HIP_RUNTIME` | the HIP runtime the launcher's probe loads (tests: the stand-in of step 5.1) | the system's |

Set `ROCM_PATH` to the HIP SDK folder (the commands below also pass it as `--rocm`). Remove `HIP_PATH`, `LLVM_PATH`,
`HIP_DEVICE_LIB_PATH` and `HIP_CLANG_PATH` from the build's environment unless you set them on purpose; the scripts give
the HIP compiler the folder explicitly.

Open a "x64 Native Tools Command Prompt for VS 2026" (or call `vcvars64.bat`) for every step below, at the repository
root. The commands write into `out\`; any folder outside the repository works the same.

## 3. Kernels and runtime

```bat
set WG_TOOLCHAINS=C:\tools\wronggpu-toolchains
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.2

rem 3.1 Ray Reconstruction kernels (DXIL), its tables, wronggpu_rr_runtime.dll objects and the D3D12 test host
python -m clean.rr_f.runtime.tools.build --out out\rr-rt

rem 3.2 Model M's Direct3D 12 kernels (DXIL + out\runtime\dx12\manifest.json), with the RGA register check
python clean\dx12\build_dx12.py --out out\runtime

rem 3.3 Model M's HIP code objects (gfx1201 and gfx1200) and the one runtime library wronggpu_runtime.dll
rem     (Model M + Ray Reconstruction linked in; no Model K)
python clean\runner\build.py --rocm "%ROCM_PATH%" --out out\runtime --extra-archs gfx1200 --no-model-k --with-rr out\rr-rt\obj
```

3.2 takes the longest (two of its kernels take several minutes each in DXC). Lower `WG_DX12_JOBS` / `WG_RR_JOBS` on a
busy PC.

## 4. FSR front end, Ray Reconstruction plugin, shared settings, converter

```bat
set CMAKE_FLAGS=-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded

rem 4.1 FSR front end: amd_fidelityfx_dx12.dll (and WrongGPU.asi for OptiScaler before 0.10)
cmake -S clean\fsr -B out\frontend %CMAKE_FLAGS% -DWG_MODEL_K=OFF -DWG_BUILD_ID=0.6.0 -DWG_BUILD_ASI=ON
cmake --build out\frontend

rem 4.2 Ray Reconstruction plugin: out\rr-proxy\hook\WrongGPU-RR.asi (--no-tests: see 7 for its tests)
python clean\rr_f\proxy\tools\build.py --out out\rr-proxy --no-tests

rem 4.3 shared settings and converter (both are also built as part of the launcher)
cmake -S clean\wgshared -B out\wgshared %CMAKE_FLAGS%
cmake --build out\wgshared
cmake -S clean\converter -B out\converter %CMAKE_FLAGS%
cmake --build out\converter
```

## 5. Launcher

### 5.1 Payload folders

The launcher embeds two folders:

```
out\payload\amd_fidelityfx_dx12.dll          from out\frontend
out\payload\plugins\WrongGPU.asi             from out\frontend
out\payload\wronggpu_runtime.dll             from out\runtime
out\payload\wronggpu-license.txt             LICENSE (the MIT licence; official builds ship it as their licence)
out\payload\wronggpu-notices.txt             clean\fsr\NOTICES.txt
out\rr\plugins\WrongGPU-RR.asi               from out\rr-proxy\hook
out\rr\interposer.json                       clean\rr_f\proxy\binding\interposer.json
```

```bat
mkdir out\payload\plugins out\rr\plugins
copy /y out\frontend\amd_fidelityfx_dx12.dll out\payload\
copy /y out\frontend\WrongGPU.asi out\payload\plugins\
copy /y out\runtime\wronggpu_runtime.dll out\payload\
copy /y LICENSE out\payload\wronggpu-license.txt
copy /y clean\fsr\NOTICES.txt out\payload\wronggpu-notices.txt
copy /y out\rr-proxy\hook\WrongGPU-RR.asi out\rr\plugins\
copy /y clean\rr_f\proxy\binding\interposer.json out\rr\
```

For tests, build the HIP runtime stand-in and point `WRONGGPU_HIP_RUNTIME` at it, so no test loads the real HIP
runtime:

```bat
mkdir out\mockhip && cl /nologo /LD /O2 clean\installer\tests\mock_hip.c /Fe:out\mockhip\amdhip64_7.dll /Fo:out\mockhip\
set WRONGGPU_HIP_RUNTIME=%CD%\out\mockhip\amdhip64_7.dll
```

### 5.2 Release build

```bat
cmake -S clean\installer -B out\launcher %CMAKE_FLAGS% -DWRONGGPU_PAYLOAD_DIR=%CD%\out\payload -DWRONGGPU_RR_DIR=%CD%\out\rr ^
      -DWRONGGPU_VERSION=0.6.0 -DWRONGGPU_BUILD_ID=0.6.0 -DWRONGGPU_PAYLOAD_NAME=0.6.0 -DWRONGGPU_TESTING=OFF ^
      -DWRONGGPU_PRESET_K=OFF -DWRONGGPU_DIRECT_ROUTE=OFF
cmake --build out\launcher
```

`out\launcher\WrongGPU.exe` is the launcher. The converter maps it embeds come from `data\` (`WRONGGPU_MAP`,
`WRONGGPU_MAP_F`, `WRONGGPU_MAP_K` override them).

## 6. Tests (CPU)

```bat
ctest --test-dir out\converter --output-on-failure
ctest --test-dir out\wgshared --output-on-failure
ctest --test-dir out\launcher --output-on-failure
ctest --test-dir out\launcher\converter --output-on-failure
ctest --test-dir out\frontend --output-on-failure -R "unit|hosted|diagnostic|support" -E "overlay|split|d3d12"

rem Python unit tests. Folders without a GPU module can be discovered whole:
python -m unittest discover -s clean\tests -t .
python -m unittest discover -s clean\rr_f\tests -t .
python -m unittest discover -s clean\rr_f\head\tests -t .
rem Folders with GPU modules (clean\k\tests, clean\rr_f\runtime\tests): one module at a time, leaving out the GPU
rem modules listed below, e.g.
python -m unittest clean.k.tests.test_k_reference clean.k.tests.test_k_runtime_tables clean.k.tests.test_k_runtime_pack
rem Test folders without __init__.py (clean\dx12\tests, clean\fsr\tests, clean\rr_f\proxy\tests) run from their own
rem folder, one file at a time where the folder has a GPU module:
python -m unittest discover -s clean\dx12\tests -t clean\dx12\tests
python -m unittest discover -s clean\fsr\tests -t clean\fsr\tests
set WG_RR_PROXY_BUILD=%CD%\out\rr-proxy
python -m unittest discover -s clean\rr_f\proxy\tests -t clean\rr_f\proxy\tests -p test_optimal.py
python -m unittest discover -s clean\rr_f\proxy\tests -t clean\rr_f\proxy\tests -p test_gen_forwarding.py
```

`clean\runtime\tests\test_wmma_probe.cpp` checks Model M's driver-channel probe (`clean\runtime\wmma_probe.h`) on
any host: build it with any C++17 compiler (`clang++ -std=c++17 test_wmma_probe.cpp`) and run it; exit status 0 when
it passes. `clean\dx12\tests\native_probe_test.cpp` runs the same probe on the GPU (see 7).

`clean\wgshared\tests\*.py` are scripts that the wgshared ctest runs. The Ray Reconstruction plugin's CPU tests are
executables that `clean\rr_f\proxy\tools\build.py --no-tests` builds; run them as that script does:
`out\rr-proxy\core\test_proxy_core.exe`, `test_probe_core.exe`, `test_setupapi_identity.exe`,
`test_ngx_rr.exe <the *-rr-frames.json in clean\rr_f\proxy\tests\data>` and `test_replay_profile.exe <each
*-traffic.json there>`. Its other test executables create a Direct3D 12 device (see 7).

Some Python test modules need a GPU and must be left out of a CPU run: in `clean\k\tests` the modules
`test_k_hidden_guard`, `test_k_input_uniforms`, `test_k_interop_timing`, `test_k_norm_r16`, `test_k_output_arithmetic`,
`test_k_output_selector`, `test_k_runtime_gpu`, `test_k_tiled_input`; `clean\rr_f\proxy\tests\test_capture_replay`,
`clean\rr_f\runtime\tests\test_model_pack`, and `clean\tests\test_no_model_k` when `WG_NO_K_BUILD` is set (without it,
its library part skips itself and the rest runs on the CPU). Tests that need files outside the
repository (the synthetic converter library stand-ins `WGC_SYNTH_BINS`, a model file `WG_RR_MODEL`, a capture folder
`WG_RR_CAPTURE`) skip themselves when those are not given.

### Comparing kernels with a release

The kernels are deterministic for a given DXC / HIP SDK version: `out\rr-rt\kernels.json` (name, arguments, SHA-256 of
each DXIL) and `out\runtime\dx12\manifest.json` (SHA-256 of each DXIL) can be compared with another build of the same
source: for `kernels.json`, compare each kernel's `nargs` and `sha256`.

The code objects `out\runtime\*.co` and `out\runtime\gfx1200\*.co` of builds made with `-cuid` (the compilation-unit
id that `clean\runner\build.py` now fixes) can be compared file by file. In a check of two build folders, all 54
compiled code objects (gfx1201 and gfx1200) were byte-identical.

Release 0.6.0 was built without `-cuid`, so none of its code objects is byte-identical to a new build: the
`__hip_cuid_...` symbol differs. Compare the `.text` and `.note` sections instead (extract them with an ELF tool, for
example `llvm-objcopy --dump-section`); they were identical in all 54, as was the section list. The `.rodata` section
was identical in 52. In `post_w64__orig.co` (both targets) it moves by 64 bytes, and its 93 kernel descriptors differ
only in their code-entry offset, by that amount; so do not use `.rodata` as the comparison for 0.6.0.

## 7. GPU steps (optional; not needed to build)

These load a Direct3D 12 or HIP device: the front end's D3D12 tests (`ctest --test-dir out\frontend -E
"unit|hosted|diagnostic|support"`), the Ray Reconstruction plugin's full test run (`clean\rr_f\proxy\tools\build.py`
without `--no-tests`), `clean\runtime\interop\tests` (the D3D12 interop harness), `clean\dx12\tests`, the Ray
Reconstruction end-to-end test (`clean\rr_f\runtime\tests\e2e.py`), the timing tools, and the GPU test modules listed
in 6. Run them on an RX 9000-series card; `WG_GPU_LOCK` can serialise them when several shells share one GPU.
