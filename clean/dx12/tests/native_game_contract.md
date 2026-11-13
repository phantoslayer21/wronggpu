# Private supplied-device candidate

`clean/runtime/interop/native_candidate.{h,cpp}` is an additive private API. It does not
replace the FSR frontend, select a new default provider, install a game payload or
adopt a caller's command list. The caller supplies its actual device and DIRECT or
COMPUTE queue, already-submitted producers and an input-ready fence/value. The
candidate uses its own lists and allocators on that queue. Its returned completion
fence covers the final output conversion and restoration of all input/output states.

The private DLL exports `wg_native_candidate_create`, `dispatch`, `sync`, `error`
and `destroy` (each with the `wg_native_candidate_` prefix). Create reports errors
into the supplied bounded string. Dispatch/sync preserve a per-handle error string;
dispatch returns an AddRef completion fence. Destroy consumes the handle and returns
failure if completion could not be established. The frontend/default provider is
unchanged. The test executable checks every symbol and null/error ABI paths;
`native_game_test.exe --api-only` runs those checks without creating a GPU device.

The first implementation permits one serialized frame. Before the next frame can
reuse any Converter descriptor, it synchronizes the previous final-consumer fence.
It retains texture AddRefs through completion. Destroy with outstanding submitted
work drains the queue. Unknown completion quarantines the entire runtime, Converter
and resource-reference set rather than releasing possibly live GPU objects.

The caller must provide truthful resource states and must honor the returned fence
before accessing or releasing output. The caller must not submit simultaneous work
that writes these textures or the arena. Actual resource state is not introspectable;
the debug-layer tests verify that the stated contract is followed by this harness.
No arbitrary caller-list completion marker or implicit signal is used.

Accepted textures are one mip, one array slice and one sample. Color is RGBA32F,
RGBA16F or R11G11B10F; depth R32F or R16F; motion RG32F or RG16F; exposure R32F;
output RGBA32F or RGBA16F with typed UAV stores. Typeless/depth attachment aliases,
output-resolution motion, array/MSAA resources, other formats, aliased textures,
unknown states and incompatible write states are explicitly rejected. This narrow
contract is a private test candidate, not a claim of universal game compatibility.

The original Converter is retained for both input and output. Input pre-exposure
division, exposure processing, motion/jitter conventions, output scaling, typed
output rounding, neural history and required intermediate consumers remain present.
Private explicit input/output slots avoid querying the runtime's last-frame buffer
API before a frame exists; no synthetic warm-up frame alters the temporal state.

`native_game_test.cpp` supplies an actual D3D12 device/queue and exercises the WMMA
driver-channel probe there. Original moving/disoccluding synthetic frames use a
nonzero input subrect, motion scale/offset, jitter, all three exposure modes,
pre-exposure 1/1.375, resets and alternating UAV/SRV output states. Both paths use
exact GPU-Converter input bytes. The oracle is selectable as standalone native or
HIP. It compares 17 active stage buffers, previous history off reset and the final
texture written by the same Converter. It deliberately mutates one output byte to
prove comparator sensitivity. The existing stronger FP32/helper/mutation gates
remain independently required; texture equality alone cannot replace them.

Lifecycle testing includes multiple create/destroy cycles and destruction with a
submitted frame outstanding. Invalid metadata, missing fences, caller-list misuse,
wrong-device textures/fences, output flags and invalid state combinations must fail
before any candidate recording. A WARP device is used only to construct deliberately
wrong-device resources. Native-oracle mode checks fresh-process modules before
creation, during the candidate lifetime and after destruction, rejecting HIP DLLs.
The harness observes debug messages without clearing another active native object's
message stream. Missing debugging or discarded diagnostics do not qualify as clean.
The first frame also deliberately holds a separate input-ready fence unsignalled
until dispatch has returned and verifies the final completion remains pending.
For actual descriptor-ring wrap coverage, use at least 257 frames in one cycle at
a small valid shape such as 65x65 -> 97x97, in addition to the required normal/odd
resolution cases. Short runs verify serialization but do not wrap that ring.

Build with `native_game_build.py --out PRIVATE_BUILD`. Build the complete shader
manifest separately with `build_dx12.py --out FULL_BUILD` before running. The root
integration owner serializes builds and GPU work. Run `native_game_check.py` under
the shared `the GPU lock (WG_GPU_LOCK, BUILDING.md) --timeout 0`, with explicit `--exe`, `--build`,
`--model`, `--out`, `--queue`, `--oracle` and `--shape RW RH OW OH` arguments. The
wrapper refuses to overwrite evidence, stores raw logs/trace and source payload
hashes, and asserts exact neural-stage dispatch counts. These synchronized checks
are correctness tests and provide no complete-path performance measurement.

Basis: original `convert.cpp`, `runtime_core.h`, `device.h`, specs 03/07/10/20 and
the original synthetic interop test's scene/readback patterns. The new HLSL helpers
use the original `io_probe` arithmetic and `exposure_pair_diet` HIP definition,
preserving the FP64 exposure quotients. The header/source/test implementations are
original; no third-party implementation, game data or learned parameters are used.
