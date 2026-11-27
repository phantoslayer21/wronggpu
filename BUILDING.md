# Building WrongGPU

Build `WrongGPU.exe` and its embedded payload from a clean Windows installation. The build and unit tests run on the
CPU and need no GPU. The optional device tests, timing tools, and game tests at the end need a GPU.

## 1. Tools

| Tool | Version used for 0.6.0 | Used for |
|---|---|---|
| Windows | 10 or 11, x64 | |
| Visual Studio 2026, Community or another edition, with the "Desktop development with C++" workload | 18.x; MSVC 19.51 (toolset 14.51); bundled CMake 4.3 and Ninja 1.13 | Every C++ build |
| Windows SDK | 10.0.26100 | D3D12 and DXGI headers and libraries, `dxcapi.h` |
| Python | 3.12, 64-bit | Build scripts, references, and tests |
| NumPy | 2.x; 2.5 tested | References and tests. Install with `python -m pip install numpy`. |
| AMD HIP SDK for Windows | 7.2, ROCm clang 21 | HIP kernels for gfx1201 and gfx1200, and the runtime library |
| Microsoft DirectX Shader Compiler (DXC) | v1.10.2605.24. Use the release zip containing `dxc.exe`, `dxv.exe`, and `dxcompiler.dll`. | Direct3D 12 kernels |
| AMD Radeon GPU Analyzer (RGA) | 2.15 | Register checks for the Direct3D 12 kernels |
| Git | Any recent version | |

Pillow is optional for two research tools. MinGW-w64 GCC 13 and Wine 9 are optional for development builds and Linux
tests. Releases never use them.

Use the exact DXC and HIP SDK versions to reproduce a release's kernels bit for bit. See
[Comparing kernels with a release](#comparing-kernels-with-a-release).

## 2. Where the build finds the tools

The repository contains no paths from a developer's PC. Build scripts find tools through these environment variables
and defaults.

| Variable | Meaning | Default |
|---|---|---|
| `WG_TOOLCHAINS` | Folder containing the unpacked DXC release in `dxc-v1.10.2605.24\` and the unpacked RGA release in `rga-2.15\`. | `<repository>\toolchains`, ignored by git |
| `WG_DXC_DIR` | DXC's `bin\x64` folder. Overrides `WG_TOOLCHAINS`. | `%WG_TOOLCHAINS%\dxc-v1.10.2605.24\bin\x64` |
| `WG_RGA` | Path to `rga.exe`. Overrides `WG_TOOLCHAINS`. | `%WG_TOOLCHAINS%\rga-2.15\rga.exe` |
| `ROCM_PATH` | HIP SDK folder. Scripts with a `--rocm` argument default to this variable, and the argument takes precedence. Other scripts read the variable. | `C:\Program Files\AMD\ROCm\7.2` |
| `VSDEVCMD` | Visual Studio's `VsDevCmd.bat`, used by the kernel scripts to build two small helper tools. For another edition, use a path such as `C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\Tools\VsDevCmd.bat` or `...\18\Enterprise\...`. | `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat` |
| `CXX`, `WG_K_RT_CXX`, `WG_K_RT_BUILD`, `WG_CONVERTER_EXE` | Compiler and build settings for the Model K table tests `clean\k\tests\test_k_runtime_tables` and `test_k_runtime_pack`. The tests try `%CXX%`, clang++ or g++ on `PATH`, the HIP SDK's clang++, then MSVC. Set `WG_K_RT_CXX=msvc` to try MSVC first. The other variables set their build folder and an existing converter. If the tests find no compiler, they skip and report why. | Temporary build folder; converter built from source |
| `WG_BUILD_ROOT` | Default output root for scripts that support one. | `<repository>\build`, ignored by git |
| `WG_DX12_JOBS`, `WG_RR_JOBS` | Parallel kernel compilation limits. | The CPU count, capped at 12, for DX12; 8 for Ray Reconstruction |
| `WG_GPU_LOCK` | Optional lock command run as `python <lock> --timeout <s> -- <command>` around GPU jobs. | Unset; GPU jobs run directly |
| `WRONGGPU_HIP_RUNTIME` | HIP runtime loaded by the launcher's probe in test builds. Step 5.1 builds a stand-in for tests. | The system's HIP runtime |

Set `ROCM_PATH` to the HIP SDK folder. The commands below also pass it through `--rocm`. Unless you set them
intentionally, remove `HIP_PATH`, `LLVM_PATH`, `HIP_DEVICE_LIB_PATH`, and `HIP_CLANG_PATH` from the build environment.
The scripts pass the HIP compiler folder explicitly.

Open an **x64 Native Tools Command Prompt for VS 2026**, or call `vcvars64.bat`, at the repository root. Use that prompt
for every step. These commands write to `out\`. You can use a folder outside the repository instead.

## 3. Kernels and runtime

```bat
set WG_TOOLCHAINS=C:\tools\wronggpu-toolchains
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.2

rem 3.1 Ray Reconstruction kernels (DXIL), its tables, wronggpu_rr_runtime.dll objects and the D3D12 test host
python -m clean.rr_f.runtime.tools.build --out out\rr-rt

rem 3.2 Direct3D 12 kernels for Model M, Model L and Model K (DXIL + out\runtime\dx12\manifest.json), with the RGA
rem     register check
python clean\dx12\build_dx12.py --out out\runtime --model-l --model-k

rem 3.3 HIP code objects (gfx1201 and gfx1200) and one runtime library, wronggpu_runtime.dll
rem     (Model M, Model L, Model K and Ray Reconstruction linked in; Model K is included unless --no-model-k is used)
python clean\runner\build.py --rocm "%ROCM_PATH%" --out out\runtime --extra-archs gfx1200 --model-l --with-rr out\rr-rt\obj
```

Step 3.2 takes the longest. DXC compiles two of its kernels for several minutes each. Lower `WG_DX12_JOBS` or
`WG_RR_JOBS` on a busy PC.

## 4. FSR front end, Ray Reconstruction plugin, shared settings, converter

```bat
set CMAKE_FLAGS=-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded

rem 4.1 FSR front end: amd_fidelityfx_dx12.dll (and WrongGPU.asi for OptiScaler before 0.10), with Model L and
rem     Model K entries (WG_MODEL_L defaults to OFF and WG_MODEL_K to ON; both are set explicitly here)
cmake -S clean\fsr -B out\frontend %CMAKE_FLAGS% -DWG_MODEL_K=ON -DWG_MODEL_L=ON -DWG_BUILD_ID=0.6.5 -DWG_BUILD_ASI=ON
cmake --build out\frontend

rem 4.2 Ray Reconstruction plugin: out\rr-proxy\hook\WrongGPU-RR.asi (--no-tests: see section 7 for its tests)
python clean\rr_f\proxy\tools\build.py --out out\rr-proxy --no-tests

rem 4.3 Shared settings and converter (the launcher also builds both)
cmake -S clean\wgshared -B out\wgshared %CMAKE_FLAGS%
cmake --build out\wgshared
cmake -S clean\converter -B out\converter %CMAKE_FLAGS%
cmake --build out\converter
```

## 5. Launcher

### 5.1 Payload folders

The launcher embeds files from two folders:

```text
out\payload\amd_fidelityfx_dx12.dll          from out\frontend
out\payload\plugins\WrongGPU.asi             from out\frontend
out\payload\wronggpu_runtime.dll             from out\runtime
out\payload\wronggpu_runtime.build.json      from out\runtime (read by the packager, not shipped)
out\payload\wronggpu-license.txt             LICENSE (GPL-3.0-only for official builds of 0.7.0 and later)
out\payload\wronggpu-notices.txt             clean\fsr\NOTICES.txt
out\rr\plugins\WrongGPU-RR.asi               from out\rr-proxy\hook
out\rr\interposer.json                       clean\rr_f\proxy\binding\interposer.json
```

Create the folders and copy the files:

```bat
mkdir out\payload\plugins out\rr\plugins
copy /y out\frontend\amd_fidelityfx_dx12.dll out\payload\
copy /y out\frontend\WrongGPU.asi out\payload\plugins\
copy /y out\runtime\wronggpu_runtime.dll out\payload\
copy /y out\runtime\wronggpu_runtime.build.json out\payload\
copy /y LICENSE out\payload\wronggpu-license.txt
copy /y clean\fsr\NOTICES.txt out\payload\wronggpu-notices.txt
copy /y out\rr-proxy\hook\WrongGPU-RR.asi out\rr\plugins\
copy /y clean\rr_f\proxy\binding\interposer.json out\rr\
```

`clean\runner\build.py` writes `wronggpu_runtime.build.json` next to the runtime library. The payload packager,
`clean\installer\tools\make_payload.py`, refuses a runtime without this manifest or with a SHA-256 that differs from
the manifest. Keep the library and manifest together. Do not change the library after building it, including signing
or stripping it. The packager reads the manifest for this check but does not ship it.

For tests, build the HIP runtime stand-in and point `WRONGGPU_HIP_RUNTIME` at it. This prevents tests from loading the
real HIP runtime:

```bat
mkdir out\mockhip && cl /nologo /LD /O2 clean\installer\tests\mock_hip.c /Fe:out\mockhip\amdhip64_7.dll /Fo:out\mockhip\
set WRONGGPU_HIP_RUNTIME=%CD%\out\mockhip\amdhip64_7.dll
```

### 5.2 Release build

```bat
cmake -S clean\installer -B out\launcher %CMAKE_FLAGS% -DWRONGGPU_PAYLOAD_DIR=%CD%\out\payload -DWRONGGPU_RR_DIR=%CD%\out\rr ^
      -DWRONGGPU_VERSION=0.6.5 -DWRONGGPU_BUILD_ID=0.6.5 -DWRONGGPU_PAYLOAD_NAME=0.6.5 -DWRONGGPU_TESTING=OFF ^
      -DWRONGGPU_PRESET_K=ON -DWRONGGPU_MODEL_K=ON -DWRONGGPU_MODEL_L=ON -DWRONGGPU_DIRECT_ROUTE=OFF
cmake --build out\launcher
```

The build creates `out\launcher\WrongGPU.exe` with the per-game Model L and Model K choices.
`WRONGGPU_MODEL_L` and `WRONGGPU_MODEL_K` default to `OFF`, so the release command enables them. The launcher's
converter maps come from `data\`. Override them with `WRONGGPU_MAP`, `WRONGGPU_MAP_F`, `WRONGGPU_MAP_K`, and
`WRONGGPU_MAP_L`.

## 6. Tests (CPU)

Run the C++ and Python CPU tests:

```bat
ctest --test-dir out\converter --output-on-failure
ctest --test-dir out\wgshared --output-on-failure
ctest --test-dir out\launcher --output-on-failure
ctest --test-dir out\launcher\converter --output-on-failure
ctest --test-dir out\frontend --output-on-failure -R "unit|hosted|diagnostic|support" -E "overlay|split|d3d12"

rem Python unit tests. Discover folders without a GPU module as a whole:
python -m unittest discover -s clean\tests -t .
python -m unittest discover -s clean\rr_f\tests -t .
python -m unittest discover -s clean\rr_f\head\tests -t .
rem For folders with GPU modules (clean\k\tests and clean\rr_f\runtime\tests), run one module at a time and leave out
rem the GPU modules listed below. For example:
python -m unittest clean.k.tests.test_k_reference clean.k.tests.test_k_runtime_tables clean.k.tests.test_k_runtime_pack
python -m unittest clean.rr_f.runtime.tests.test_rr_retire
rem Test folders without __init__.py (clean\dx12\tests, clean\fsr\tests, clean\rr_f\proxy\tests) run from their own
rem folder, one file at a time where the folder has a GPU module:
python -m unittest discover -s clean\dx12\tests -t clean\dx12\tests
python -m unittest discover -s clean\fsr\tests -t clean\fsr\tests
set WG_RR_PROXY_BUILD=%CD%\out\rr-proxy
python -m unittest discover -s clean\rr_f\proxy\tests -t clean\rr_f\proxy\tests -p test_optimal.py
python -m unittest discover -s clean\rr_f\proxy\tests -t clean\rr_f\proxy\tests -p test_gen_forwarding.py
```

`clean\runtime\tests\test_wmma_probe.cpp` checks Model M's driver-channel probe in `clean\runtime\wmma_probe.h` on any
host. Build it with any C++17 compiler, such as `clang++ -std=c++17 test_wmma_probe.cpp`, and run it. A passing test
exits with status 0. `clean\dx12\tests\native_probe_test.cpp` runs the same probe on the GPU, as described in section
7.

The wgshared ctest runs the scripts in `clean\wgshared\tests\*.py`. The Ray Reconstruction plugin build with
`--no-tests` still builds its CPU test executables. Run them as the build script does:

- `out\rr-proxy\core\test_proxy_core.exe`
- `test_probe_core.exe`
- `test_setupapi_identity.exe`
- `test_ngx_rr.exe <the *-rr-frames.json in clean\rr_f\proxy\tests\data>`
- `test_replay_profile.exe <each *-traffic.json in clean\rr_f\proxy\tests\data>`

The plugin's other test executables create a Direct3D 12 device and belong to the GPU steps in section 7.

Leave these GPU-only Python modules out of a CPU run:

- In `clean\k\tests`: `test_k_hidden_guard`, `test_k_input_uniforms`, `test_k_interop_timing`, `test_k_norm_r16`,
  `test_k_output_arithmetic`, `test_k_output_selector`, `test_k_runtime_gpu`, and `test_k_tiled_input`.
- In `clean\rr_f\proxy\tests`: `test_capture_replay`.
- In `clean\rr_f\runtime\tests`: `test_model_pack`.
- In `clean\tests`: `test_no_model_k` when `WG_NO_K_BUILD` is set. Without that variable, its library part skips itself
  and the rest runs on the CPU.

Tests that need files outside the repository skip themselves when those files are not provided. These files include
the synthetic converter library stand-ins in `WGC_SYNTH_BINS`, a model file in `WG_RR_MODEL`, and a capture folder in
`WG_RR_CAPTURE`.

### Comparing kernels with a release

Kernels are deterministic for a given DXC and HIP SDK version. Compare another build of the same source through
`out\rr-rt\kernels.json` and `out\runtime\dx12\manifest.json`. The first file contains each DXIL kernel's name,
arguments, and SHA-256. Compare each kernel's `nargs` and `sha256`. The second file contains each DXIL kernel's
SHA-256.

Builds made with `-cuid`, which `clean\runner\build.py` now fixes as the compilation-unit ID, produce code objects that
can be compared file by file. Compare `out\runtime\*.co` and `out\runtime\gfx1200\*.co`. In one check of two build
folders, all 54 compiled code objects across gfx1201 and gfx1200 were byte-identical.

Release 0.6.0 was built without `-cuid`, so none of its code objects is byte-identical to a new build. The
`__hip_cuid_...` symbol differs. Extract and compare the `.text` and `.note` sections instead, using an ELF tool such
as `llvm-objcopy --dump-section`. In the comparison, those sections and the section list were identical in all 54 code
objects. The `.rodata` section was identical in 52. In `post_w64__orig.co` for both targets, `.rodata` moves by 64
bytes, and its 93 kernel descriptors differ only in their code-entry offset by that amount. Do not use `.rodata` to
compare a new build with 0.6.0.

## 7. GPU steps (optional; not needed to build)

The following steps load a Direct3D 12 or HIP device:

- The front end's D3D12 tests: `ctest --test-dir out\frontend -E "unit|hosted|diagnostic|support"`.
- The Ray Reconstruction plugin's full test run: `clean\rr_f\proxy\tools\build.py` without `--no-tests`.
- The D3D12 interop tests in `clean\runtime\interop\tests`.
- The tests in `clean\dx12\tests`.
- The Ray Reconstruction end-to-end test, `clean\rr_f\runtime\tests\e2e.py`.
- The timing tools and the GPU test modules listed in section 6.

Run them on an RX 9000-series card. `WG_GPU_LOCK` can serialise GPU jobs when several shells share one GPU.

## 8. Model L and Model K in the build

Player builds since 0.6.5 include Model L and Model K alongside Model M. Model M remains the default for every game.
Choose Model L or Model K per game with `set <game> model=L` or `model=K`, or through the **WrongGPU Model L** or
**WrongGPU Model K** entry in OptiScaler's menu.

The release commands above enable the models in all three components:

- Step 3.2 uses `--model-l --model-k`.
- Step 3.3 uses `--model-l`. Model K is included unless `--no-model-k` is used.
- Step 4.1 uses `-DWG_MODEL_K=ON -DWG_MODEL_L=ON`.
- Step 5.2 uses `-DWRONGGPU_PRESET_K=ON -DWRONGGPU_MODEL_K=ON -DWRONGGPU_MODEL_L=ON`.

The launcher, front end, and runtime must support the same models. A launcher choice that the front end or runtime
lacks offers a model that cannot run.

For a Model M-only comparison build, not a release, replace those commands with:

```bat
rem 3.2 and 3.3: no Model L or Model K kernels or code objects
python clean\dx12\build_dx12.py --out out\runtime
python clean\runner\build.py --rocm "%ROCM_PATH%" --out out\runtime --extra-archs gfx1200 --no-model-k --with-rr out\rr-rt\obj

rem 4.1 Front end: no Model L or Model K entry
cmake -S clean\fsr -B out\frontend %CMAKE_FLAGS% -DWG_MODEL_K=OFF -DWG_MODEL_L=OFF -DWG_BUILD_ID=dev-m -DWG_BUILD_ASI=ON

rem 5.2 Launcher: Model M only
cmake -S clean\installer -B out\launcher %CMAKE_FLAGS% -DWRONGGPU_PAYLOAD_DIR=%CD%\out\payload -DWRONGGPU_RR_DIR=%CD%\out\rr ^
      -DWRONGGPU_VERSION=0.6.5 -DWRONGGPU_BUILD_ID=dev-m -DWRONGGPU_PAYLOAD_NAME=0.6.5 -DWRONGGPU_TESTING=OFF ^
      -DWRONGGPU_PRESET_K=OFF -DWRONGGPU_DIRECT_ROUTE=OFF
```

`-DWRONGGPU_MODEL_K=ON` requires `-DWRONGGPU_PRESET_K=ON`. Configuration stops with an error without it. Use a front
end and runtime from the same build.

Model K runs its exact network by default. The launcher and front end do not offer or set another mode. The optional
`k_approx` runtime setting is documented in the
[0.6.5 release notes](clean/installer/RELEASE-NOTES.md#model-k-new).

CI's CPU test jobs configure the launcher and front end with Model K off. The front end also tests both states through
`fsr_frontend_model_k_unit` in a build with `-DWG_MODEL_K=OFF` and `fsr_frontend_no_model_k_unit` in a build with it
on. The launcher's `model-choice-k` and `model-choice-k-only` tests exercise the K choice regardless of the build
option.

`clean\installer\tests\run_model_k_tests.py` runs the choice end to end on generated fixtures without a game or GPU.
Its header explains how to prepare the payload, test maps, and a launcher with the choices enabled.
