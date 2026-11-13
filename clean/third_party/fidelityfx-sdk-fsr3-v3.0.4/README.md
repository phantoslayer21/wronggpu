# AMD FidelityFX SDK fsr3-v3.0.4 — FSR 3.0 host headers (vendored)

Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `fsr3-v3.0.4`, commit
`55ff22bb6981a9b9c087b9465101769fc0acd447`, directory `sdk/include/FidelityFX/host/`. Fetched 2026-10-05.
Licence: MIT (`LICENSE.txt`, copied unchanged from the repository root at that tag; each header carries an MIT notice).
The structures used by WrongGPU are identical in tag `fsr3-v3.0.3`.

Only the host headers that declare the FSR 3.0 component DLL interface (`ffx_fsr3_x64.dll`, `ffx_fsr3upscaler_x64.dll`)
and the types they need, copied byte for byte (no changes). No GPU (shader) file of the SDK is vendored: the one GPU
header `ffx_fsr3upscaler.h` includes (`FidelityFX/gpu/fsr3upscaler/ffx_fsr3upscaler_resources.h`) is replaced by an
empty WrongGPU stub (`clean/fsr/src/sdk30_stub/...`), because the host declarations do not use it.

WrongGPU's code does not include these headers except in `clean/fsr/src/fsr30_layout_check.cpp`, which checks at
compile time that WrongGPU's own plain-type mirror (`clean/fsr/src/fsr30_types.h`) has exactly their layout.

| File | SHA-256 |
|---|---|
| `FidelityFX/host/ffx_assert.h` | 0582c35f37ffc873f6a2773ce4339fd8e99e34eb6c40d63c1820a65ad17a03cc |
| `FidelityFX/host/ffx_error.h` | 8bbcb81e27eb4d83ff088985e9718665d0a3178ce10ada2c63db3265c324a580 |
| `FidelityFX/host/ffx_frameinterpolation.h` | 1d22e3f85b51e53e2f9bc7bc06169105a8511e99871b240fa624e8163fff04b8 |
| `FidelityFX/host/ffx_fsr3.h` | 12f7125a12f5bb2c92109e47e6dbe7951fcf574d4f9b81e37898f3cea2a741a1 |
| `FidelityFX/host/ffx_fsr3upscaler.h` | c4e401e03bcb749f8256abdc088b0597a35138660559d3edb6846efa752009c2 |
| `FidelityFX/host/ffx_interface.h` | a53733e10b23d6132b076e7f1dd83c20c8d951d3c164acb4fac1a2f25ce423b6 |
| `FidelityFX/host/ffx_opticalflow.h` | 90734db2a9b1a9fc2f50c1049cffbd2fde00c02324f8a87d983ee10837c5ec57 |
| `FidelityFX/host/ffx_types.h` | 99b72c89bf4a0a8286f644a46a1baaa91f1a195904d95b694d613052d82f32f7 |
| `FidelityFX/host/ffx_util.h` | 4f174191c58157119b86987f789d161a86ba3aedd2750bbca0b04502b25e6769 |
| `LICENSE.txt` | 5181f1d14a6319dd5661dc032c381a7354513412b690d0692fb612417608a208 |
