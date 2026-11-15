# Release checks

Checks a release build must pass that the CI cannot run, because they need the build's own binaries. The release
lead runs them on the release candidate (blocker C3 in the release list) and records the result with the release.

| Script | What it checks | Needs |
|---|---|---|
| `run_refusal_check.py` | B8: the front end with the real network backend (the build's `wronggpu_runtime.dll`) on the refused-frame paths: AMD FSR present, AMD absent with the stand-in working, and AMD absent with the stand-in unavailable, where the backend's bilinear last resort is checked on every output pixel; for model K and the model M control; no HIP executor, no `amdhip64` (`clean/fsr/tests/test_refusal_integration.cpp`) | the build's runtime DLL, `clang++`, Visual Studio with the Windows SDK, a D3D12 device (WARP is enough) |

```
python tools/release/run_refusal_check.py --runtime-dll <build>\wronggpu_runtime.dll --work <new scratch folder>
```

Each check writes only inside its work folder: APPDATA, LOCALAPPDATA, ProgramData, TMP and TEMP point at
`<work>\scratch` while it runs. Its exit code is the test's (0 = pass). A missing DLL or test file is a failure, never
a skip.
