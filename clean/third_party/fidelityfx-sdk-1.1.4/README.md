# AMD FidelityFX SDK 1.1.4 — FidelityFX API headers (vendored)

Source: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `v1.1.4`, commit
`c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`, directory `ffx-api/include/ffx_api/`. Fetched 2026-10-05.
Licence: MIT (`LICENSE.txt`, copied unchanged from the repository root; each header carries the same notice).

Only the public API headers WrongGPU's FSR front end needs, copied byte for byte (no changes):

| File | SHA-256 |
|---|---|
| `ffx_api/ffx_api.h` | 408b4203545e8299ed188fbb1d9ddfe0932b73ea6e8b14fe10edcf1f45771561 |
| `ffx_api/ffx_api_types.h` | dee2782813d22047c068e0fb27d1347324f8ce79d280a4672848b3b294cb12fe |
| `ffx_api/ffx_upscale.h` | 1155bcdcea9c10e6a95507bc9422ca062a853ac84250f0b09a637a66836b3ac1 |
| `ffx_api/ffx_framegeneration.h` | 344936dccf49267d4421d1d74dc6db617d0a83c5a9f5991841325e812a38f127 |
| `ffx_api/dx12/ffx_api_dx12.h` | 215804fba316acb81d2c463b6bbce36c53cc63465c15b8c9858554c8111fae24 |
| `LICENSE.txt` | 82cf74fc23885107c7f69249ac3f1dac55898ff68320eb2c753984cae6b2d78a |

Why 1.1.4 and not the newer 2.x SDK: in 2.x the repository's top-level licence (`Kits/FidelityFX/docs/license.md`)
is a binary-only redistribution licence with a no-reverse-engineering clause for "all files except as noted", while
1.1.4 is MIT throughout. The FidelityFX API is ABI-stable (new features come as new chained structure types), so the
1.1.4 headers describe the calls FSR 3.1 games make; structure types they do not know are logged by number and
forwarded to AMD's DLL unchanged.
