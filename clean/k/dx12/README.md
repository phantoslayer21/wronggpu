<!-- SPDX-License-Identifier: MIT -->
# Model K on Direct3D 12

The D3D12 ports of Model K's HIP kernels (issue #15). Each port reproduces one HIP kernel of `clean/k/kernels` bit for
bit; the HIP kernel stays the reference. The ports are built only with `--model-k`
(`python clean/dx12/build_dx12.py --out <dir> --model-k`), so a build without it is Model M's kernel set, and
`--model-l` Model L's, unchanged (`clean/dx12/kernel_sets.json`, `clean/dx12/tests/test_kernel_sets.py`).

| Path | What |
|---|---|
| `kernels/*.hlsl` | the ports, declared with `// @wg-kernel` header lines like the other D3D12 kernels (`co=` names the HIP code object whose kernel the port reproduces) |
| `hlsl/*.hlsli` | headers shared by the K ports (included with `clean/dx12/hlsl`) |
| `k_port_check.py` | the byte-for-byte check of a family of ports against HIP, below |

A HIP kernel that takes a struct with pointers inside by value declares that argument as `s<N>:<fields>`
(`clean/runtime/struct_args.h`): the hybrid and the native device turn each pointer into an arena offset and pass the
other bytes as they are.

## The port check

```bat
python clean\k\dx12\k_port_check.py --build <dir> --rocm "%ROCM_PATH%" --prepare --family expo,warm
python clean\k\dx12\k_port_check.py --build <dir> --rocm "%ROCM_PATH%" --run --family expo,warm --json out.json
```

`--prepare` (CPU) builds the HIP code objects the families need, the D3D12 ports and `kernel_replay.exe`; `--run` (GPU,
under the machine's GPU lock) runs every case on HIP in the process and on D3D12 through `kernel_replay` on the same
input bytes, and compares the raw bytes. Rules, as for the Model L checks: every output buffer has 64 bytes of slack
and starts filled with its run's own sentinel (HIP `0xA5`, D3D12 `0x5A`); the defined bytes must be identical and the
slack must keep its sentinel; special values are among the inputs; a negative control (the port given another input
than HIP) must be caught; Model M's `scale_rgb` runs through the same replay as a control. Output: counts and verdicts.

Families: `expo` (`k_expo_scalars`) against `k_io.co`; `warm` (`k_warm`, a cache warm-up with no HIP kernel: checked
against its definition, through the struct argument path).

Known limit, shared with Model M's ports: D3D12 flushes FP32 denormals (inputs and results) to zero and HIP keeps them.
`k_expo_scalars` only meets one with a pre-exposure above about 8.5e37, so the check's inputs keep clear of them.
