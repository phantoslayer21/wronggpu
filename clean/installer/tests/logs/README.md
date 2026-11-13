# Log fixtures of the launcher's coexistence tests

Read by `clean/installer/tests/coexist_tests.cpp` (the launcher's "Last run" parser). All files are in the format of
WrongGPU's own logs (`wronggpu-fsr.log` of the FSR front end, `wronggpu-rr.log` of the Ray Reconstruction plugin); none
is another program's log.

| File | Kind |
|---|---|
| `cyberpunk-quality-fsr.log`, `cyberpunk-native-fsr.log`, `cyberpunk-rr-fsr.log` | trimmed excerpts of WrongGPU's own front-end log from release-candidate test runs in Cyberpunk 2077 |
| `bl4-rr-only.log`, `hogwarts-rr-only.log`, `re9-rr-only.log` | trimmed excerpts of WrongGPU's own Ray Reconstruction plugin log from test runs in Borderlands 4, Hogwarts Legacy and RESIDENT EVIL requiem |
| `rr-direct-short.log` | a few lines of the plugin's log from a test run (the direct route's line format) |
| `rr-failed-only.log`, `rr-passthrough-only.log`, `rr-refused-only.log` | synthetic, written for the tests |

In the excerpts, folder paths were replaced by `C:\Games\Example\...`, and process ids (now 1000, 1001, ...), device
pointers (now `0x10000000000`) and the time part of incident ids (now `134000000000000000`) were replaced by made-up
values. They contain no screenshots, frame data or model data.
