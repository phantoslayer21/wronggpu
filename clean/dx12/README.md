# D3D12 ports of the Model M kernels

Backend 3 (`backend="d3d12"`, native D3D12 with direct resources) is the one the game route uses; its milestone and
qualification notes were internal and are not part of this repository. Backend 2 (`backend="dx12"`) remains the hybrid
oracle documented below.

The runtime's hybrid D3D12 device (backend 2, `clean/runtime/device_dx12.cpp`) runs every kernel that has a D3D12 port
here on a D3D12 compute queue and everything else on HIP, on the same memory. Goal: every port gives **the same bytes**
as its HIP kernel (`clean/dx12/compare.py` proves it per kernel on synthetic frames).

## Files

| path | what |
|---|---|
| `hlsl/wg_platform.hlsli` | HLSL counterpart of `kernels/wg_platform.h` + `wg_mem.h` (same function names): kernel interface, memory, lanes, barriers, math wrappers, FP16 / FP8 packs, WMMA |
| `hlsl/wg_math.hlsli` | exact arithmetic helpers and **the arithmetic rules** (read its header) |
| `hlsl/wg_ags.hlsli` | our encoder of the AMD driver's intrinsic channel (WMMA, matrix conversions) |
| `kernels/<file>.hlsl` | the ports; one file may hold several kernels |
| `build_dx12.py` | compiles every kernel to `<build>/dx12/<tag>__<kernel>.dxil` + `<build>/dx12/manifest.json`, `-j N` kernels in parallel (default min(12, CPUs); outputs identical to `-j 1`) |
| `dxil_asm.cpp` | tool used by the build (assembles the rewritten DXIL listing) |
| `compare.py` | HIP vs hybrid D3D12, every stage output / state buffer byte for byte |
| `kernels/exposure.hlsl` | the reference port (`exposure_auto_diet`): copy its structure |

## Arithmetic mode (why build_dx12.py and nothing else)

`build_dx12.py` compiles with DXC `-T cs_6_6 -O3 -HV 2021 -enable-16bit-types -Gis` (IEEE strict: no fast-math flags),
then rewrites the DXIL listing: the intrinsics lose their `precise` marks (so the driver lowers them as in DXC's default
mode) and `rcp()` stays the hardware reciprocal; then it reassembles and signs the container (dxv). Result, measured by
the probes (0 differences outside FP32 denormals):

* `a * b`, `a + b`, `a - b`: each exactly one IEEE operation as written: in the probes **nothing was fused,
  reassociated or folded**, by DXC or by the driver. So HIP `v_mul_f32` + `v_add_f32` is simply `a * b + c`. The ports
  then found shapes where the driver does regroup or fuse: read "Driver exceptions" below.
* `mad(a, b, c)` is **always** one fused `v_fma_f32`: write it wherever HIP's ISA has `v_fma_f32`, `v_fmac_f32`,
  `v_fmaak_f32`, `v_fmamk_f32`, `v_fma_mix_f32` (FP16 operands widened by `wg_h2f`, exact), `v_pk_fma_f16` (half2).
* Follow the HIP ISA (`<build>\dx12-m\isa\<kernel>.s`), not the HIP source: clang fused across statements
  (`x = a * b; y = x + c` is often one `v_fma`), and the operand order of sums matters. Write sums in the ISA's order.
* No float `/` (the build refuses it): HIP's `a / b` is `wg_div(a, b)`. `rcp()` = `v_rcp_f32`. Double division is
  allowed and exact (`(float)(0.01 / (double)e)`).
* Never use `precise`; never compile a kernel by hand with plain dxc (default mode splits `mad`, fuses `a * b + c`,
  reassociates sums: different bits).
* Conversions: `(half)` / `f32tof16` round toward zero (= HIP `wg_h2`, `v_cvt_pkrtz`); HIP `(_Float16) x`
  (`v_cvt_f16_f32`, RNE) = `wg_f2h` / `wg_pack_h2` / `wg_to_f16` (`wg_to_f16_ags` for whole fragments).
  `v_fma_mixlo_f16 a, b, 0` = `wg_mul_f16(a, b)`, with an addend `wg_fma_f16(a, b, c)` (one rounding each).
* FP8: `wg_pack_q8 / _nc / _inner` (software, any control flow) or `wg_pack_q8*_ags` (driver conversion =
  `v_cvt_pk_fp8_f32`, fast; wave-uniform control flow only), all bit-identical to HIP incl. NaN / overflow.
  `v_maximumminimum_f32` clamp = `wg_sat448`; `v_med3_num_f32` = `wg_med3f` (lo <= hi, x not NaN).
* Accepted differences (never produced by the network's data): FP32 denormal inputs / intermediates / results
  (D3D12 flushes them, HIP keeps them); which payload propagates when two operands are NaN.

## Driver exceptions (found by the ports; every new port must check them)

The rules above hold per operation. Across operations the gfx1201 D3D12 shader compiler was seen to:

* **regroup multiplication chains** by constant / uniform / shared factors (`(v * 0.01) * ratio` -> `v * (0.01 * ratio)`)
  and **turn `mad(a, b, 0)` into `a * b`**, then fuse that product with the next add. Remedy: an opaque zero at the
  link, `asfloat(asuint(x) ^ z)` with `z = wg_grid_ >> 31` (`post.hlsl` `opq()`). The `DisableMathRefactoring` shader
  flag does not stop the regrouping and splits every `mad`: do not use it.
* **fuse FP16 `half` multiply + add** into one `v_fma_mix` even with `-Gis`: do FP16-rounded math in FP32 and round
  explicitly (`wg_f2h`, `wg_mul_f16`, the driver conversion).
* **fold an FP32 operation into the matrix conversion** that consumes it (`wg_to_f16_ags(1 - z)` became one
  `v_fma_mixlo_f16`): fine when the operation is exact, otherwise one rounding instead of HIP's two.
* **drop a store**: stores through two different pointer arguments at the same per-lane offset were merged and only
  the last one kept (encoder: `m_near` / `m_far` / `m_center`, pointers in one 16-byte cbuffer row; dec1_dec0: two
  32-bit stores at `i0 * 4` into different buffers in the odd-width branch). The DXIL had every store, the ISA did not.
  Remedies used: read the pointer through `wg_uniform()` (readfirstlane), or put another store between them. **Count
  the `buffer_store` instructions of the RGA ISA (`--isa`) against what the kernel must write.**
* **unroll loops despite `[loop]`** (43-57 KB of code, up to 240 VGPRs): hide the trip count behind an opaque zero
  (`n + (wg_grid_ >> 31)`; `swin_common.hlsli` `sw_zero()`), and fold LDS offsets with `sw_zero_bits()`.
* HLSL 2021 `?:`, `&&`, `||` short-circuit and become branches (one load in flight per branch): use `select()`,
  `and()`, `or()` in hot code.

The native helper regression additionally constructed tiny exact values that round to the smallest normal FP32
number: `0x00800000 * 0x3f7fffff` and `0x00ffffff / 0x40000000`. The multiply flushes before rounding on this driver;
the shared `wg_div` exponent reconstruction also returned zero at the quotient tie. The native `scale_rgb` and
`color_divide` wrappers repair those normal results with exact integer predicates. No new exclusion is accepted.
Those boundary repairs did not change the shared math helpers or frame ports. The later tuned-port reconciliation is
recorded in the internal status notes (not published); the broader intermediate-domain question stayed open. See
native_helpers_check.py for raw FP32, signed/neighboring boundary tests and mutation sensitivity.

compare.py sees only what reaches a named buffer; FP8 / FP16 outputs hide most one-ulp FP32 slips. The ports were
proven with differential harnesses on the intermediates as well (`tests/tuned_*_diff.cpp`, `tests/native_*`).

## Kernel declaration

```hlsl
// @wg-kernel exposure_auto_diet co=exposure.co block=256 wave=32 args=p,p,p,p,p,b36,u
// @wg-kernel bneck/swin_bneck6_nc co=swin__bmegahg.co entry=swin_bneck6_nc block=256 wave=32 args=p,p,u defines=WG_FOO=1
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint color, partials, sync_p, computed, ex;   // HIP pointers -> uint byte offsets into the arena (0 = null)
    int P_rw, P_rh, P_nx, P_ny, P_nparts, P_pad0; // a struct passed by value, flattened field by field
    float P_inv_pi, P_K, P_inv_n;
    int first;
WG_ARGS_END
groupshared float red[8];
WG_KERNEL(exposure_auto_diet) { WG_SETUP(); ... }   // entry name = HIP kernel name
```

* `@wg-kernel <name as launched>`: exactly the name the runtime passes to `Device::launch`, including the tag of a
  tagged code object (`bneck/`, `d1p12/`, `w64/`). `entry=` defaults to the part after the slash. `block` = HIP block
  size (checked at launch), `wave` = 32 or 64 (build defines `WG_BLOCK`, `WG_WAVE`; `WG_KERNEL` adds
  `[numthreads(WG_BLOCK,1,1)] [WaveSize(WG_WAVE)]`). `defines=` = the HIP build variant's `-D` flags you need.
* `group=<lanes>` (optional, a divisor of `block`) = the port's own workgroup size (`WG_BLOCK` = group); the launch keeps
  the HIP geometry (grid x block, `block` still checked) and the device dispatches grid x `gscale` workgroups of `group`
  lanes (`gscale` default block / group: each launched workgroup split into equal parts; `gscale=1`: one D3D12
  workgroup does a launched workgroup's whole work, e.g. swin_c64w2.hlsl). `WG_BLOCK`, `wg_block_id()` and
  `wg_grid_size()` are then the D3D12 ones. Manifest: `"group"`, `"gscale"` (only when group != block).
* `co` (required) = file name of the HIP code object whose build of the kernel the port reproduces. The device uses
  the port only while the runtime has exactly that code object loaded (untagged, or under the name's tag); any other
  build variant of the same kernel name (`post_w64.co` / `post_w64__q5.co` under `w64`, a `swin.co` variant,
  `dec1_dec0__pack12__div*.co` under `d1p12`, ...) runs on HIP, logged once. One name may have ports of several
  builds (one manifest entry per name and `co`, for example `w64/post_tf_f32_p12_diet_vsit` of `post_w64__orig.co`
  and of `post_w64.co`): the device runs the port whose code object is loaded. Every port of a name but the first
  sets `file=<stem>`, its DXIL file name (`<stem>.dxil`).
* `args` = the HIP kernel's arguments in order: `p` pointer (DWORD: arena byte offset), `u` any 32-bit scalar,
  `b<N>` struct by value (N bytes = N/4 DWORDs, copied verbatim). At most 57 DWORDs. The `WG_ARGS_BEGIN` block must
  declare exactly these DWORDs in order, **32-bit scalars only** (no vectors / arrays / structs: cbuffer packing).
* Output: `<build>/dx12/<tag>__<kernel>.dxil` (`<kernel>.dxil` without a tag) and a `manifest.json` line
  `{"name", "file", "co", "entry", "block", "wave", "args", "arg_dwords", "source", "sha256"}`. Build all kernels at
  once (no `--only`) for a release: `--only` rewrites the shared manifest read-modify-write.

## Platform layer (wg_platform.hlsli)

* Ids: `wg_lane_id()` (= threadIdx.x), `wg_block_id()` (grids above 65535 are split into several dispatches with a
  group base), `wg_grid_size()`, `wg_wave_lane()` (= `wg_lane_id() % WG_WAVE`, checked).
* Memory: `wg_ld8/16/32/64/128(base, off)`, `wg_ld4f`, `wg_ldf`, `wg_load8f`, `wg_st16/32/64/128`, `wg_stf`, all on the
  root UAV `wg_mem` (u0). No byte stores (read-modify-write a dword yourself, or an atomic). Cross-workgroup data
  (atomics, last-workgroup patterns, the bottleneck's spin waits) goes through `wg_mem_gc` (globallycoherent view of the
  same memory): `wg_atomic_add/max/exch/cas`, `wg_ld32_gc`, `wg_st32_gc`, `wg_fence_device()`. A spin wait must load
  through `wg_mem_gc`.
* Workgroup / wave: `wg_barrier()` (LDS + memory), `wg_barrier_lds()`, `wg_xor(v, d)` / `wg_xor16` (ds_bpermute
  lane ^ d), `wg_uniform` (readfirstlane). groupshared limit **32 KB** (patch_merge_c96 and bneck need new layouts).
* Math: `wg_exp2 wg_log2 wg_rcp wg_rsq wg_sqrt wg_rint wg_floor wg_fmin wg_fmax wg_med3f wg_f2i wg_f2u wg_ldexp wg_div`
  `wg_minimum wg_maximum wg_rn16 wg_h2f wg_h2f_lo/hi wg_f2h(_bits) wg_pack_h2 wg_h2 wg_h2_h wg_as_h2 wg_h2_bits`
  `wg_dot2_acc wg_dot_sel0/1 wg_exp2_h2 wg_mul_f16 wg_fma_f16`.
* Types: `v8f` = `struct { float v[8]; }`, v2i (8 FP8 bytes) = `uint2`, v8h (8 FP16) = `uint4`, v4f = `float4`.
* WMMA (wave32): `wg_wmma_fp8(uint2 a, uint2 b, v8f c)`, `wg_wmma_f16(uint4 a, uint4 b, v8f c)`: same fragment layouts
  and bits as HIP's `wg_wmma_*` (probe arith3). Rules of the driver channel: wave-uniform control flow; issue every
  load a group of multiplies needs before its first WMMA (the compiler never moves loads across the calls); keep
  helpers around the calls branchless (branches caused scratch spills); never use the saturate flag.

## Build and compare one kernel

```
set PY=python
cd /d <repository root>
%PY% clean\dx12\build_dx12.py --out <build>\dx12-m\hip --only exposure_auto_diet --isa
%PY% clean\dx12\compare.py --build <build>\dx12-m\hip --kernels exposure_auto_diet
```

`--isa` writes the D3D12 ISA (RGA) to `<build>/dx12/isa/<name>.isa.txt` and `.stats.csv` (VGPRs, LDS, scratch) to
compare with the HIP ISA. `--listing` keeps the rewritten DXIL listing. Every build also runs the Swin / bottleneck kernels it
compiled (names with a `swin_` part) through RGA and fails when one uses more VGPRs than its bound in `vgpr_limits.json`
(the count that keeps its present workgroups per CU, e.g. 192 for two 256-lane wave32 workgroups; a new such kernel
needs an entry; `WG_DX12_VGPR_CHECK=0` skips it). The check caught the MLP GELU T5 table (hlsl/wg_gelu_t5.hlsli) in
the two d1p12 dec1_dec0 kernels: 201 / 215 VGPRs, one workgroup per CU, dec0 +103 us at 1440p Q and +130 us at 4K Perf;
they build with `WG_GELU_T5=0` (the arena-table path, 192 VGPRs, the 67b9609 DXILs byte for byte). compare.py exits 0 only if every buffer of every
frame is identical and the kernel was launched on D3D12 (`--quick`: small case only; `--kernels a,b` several;
`--json` report). The runtime library (with backend 2) is rebuilt with
`%PY% clean\runner\build.py --rocm "%ROCM_PATH%" --out <build>\dx12-m\hip --runtime-only`
(strip HIP_PATH, ROCM_PATH, LLVM_PATH, HIP_DEVICE_LIB_PATH, HIP_CLANG_PATH from the environment of HIP processes).

Device self-test (arena, argument conversion, >65535-group grids, batched dispatches with barriers, HIP <-> D3D12
ordering; kernels of `tests/device_test.hlsl`):
`%PY% clean\dx12\tests\run_device_test.py --out <build>\dx12-m\devtest`.
Code-object binding (a port must stay on HIP when the runtime loads another build of its kernel):
`%PY% clean\dx12\tests\variant_check.py --build <build>\dx12-m\hip`.
The arithmetic probes (HIP vs D3D12 per instruction) that the rules in `hlsl/wg_math.hlsli` came from were scratch
tools and are not part of this repository.

Note on sensitivity: compare.py sees what reaches a named buffer. Exposure is rounded to FP16, so a one-ulp error in
its partial sums is invisible there; port from the ISA, do not rely on compare.py alone to find such slips.

## Runtime side

`Pipeline(..., backend="dx12")` / `WgRtConfig.backend = 2`. Environment: `WG_DX12_KERNELS` (comma list of names as
launched, `all` = default, `none`), `WG_DX12_ARENA_MB` (default 2560; every allocation of the runtime comes from this
shared buffer; model M at 3840x2160 native needs more than 1536; `0` = diagnostics: no arena, HIP memory, every kernel
on HIP), `WG_DX12_BATCH=n` (execute the D3D12 command list every n dispatches instead of only when handing back to HIP;
default 0, measured no gain), `WG_DX12_DEBUG=1` (D3D12 debug layer), `WG_LAUNCH_TRACE=<file>` (one line per launch,
D3D12 launches end in ` dx12`). A launch whose pointer arguments are not all in the arena, or whose code object is
not the port's `co`, falls back to HIP (logged once).

Ordering (device_dx12.cpp): HIP -> D3D12 = a 4-byte HIP memset in the arena's null header (`order_mark`: HIP orders its
external-semaphore signal / wait only with compute work, not with its DMA copies), then a HIP signal of the shared fence
that the D3D12 queue waits for. D3D12 -> HIP = execute the list, queue signal, HIP wait, `order_mark` again.
Consecutive D3D12 dispatches share one list with a UAV barrier before each one after the first.

Whole-frame timing, pure HIP vs hybrid: `clean\dx12\time_frames.py` (under the GPU lock; warms the clocks first, the GPU
needs 2-3 s of sustained load to reach its full shader clock).
