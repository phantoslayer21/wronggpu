# Matched real-frame stage diagnostic

`matched_stage_bench.py` prepares and runs `matched_stage_bench.cpp`. These are
original tests based on the project's Device API, runtime ABI, manifest, and
synthetic generators. They do not change runtime or shader code.

Preparation is CPU-only:

```powershell
& $PY clean/dx12/tests/matched_stage_bench.py --prepare --build $BUILD --out $PRIVATE --shape 1440p
```

Use an unused directory under `<build>/`. Preparation compiles a
private executable, writes synthetic parameters using `synthetic_model` seed
61013 with random output taps, and saves OC frames 0 and 1 plus their exact frame
ABI. The build manifest records source, binary, input, model, DXIL and original
HIP code-object hashes. `--prepare --rebuild` recompiles the owned executable,
retains and verifies existing input/model hashes, and updates the build record.
The preceding executable/build record is preserved under `prior-builds/<hash>`.
Actual architecture-qualified HIP payloads are hashed as well as logical `.co`
names. The clock probe's AMDGPU metadata must positively identify gfx1201; its
unchanged bytes are privately packaged as `clock/gfx1201.clk.co`. The Device API
receives `clock/clk.co` and resolves that architecture-qualified file. A bare
`tools/clk/clk.co` is not a supported Device load path, even when it exists.

GPU work is separate. The integration owner must acquire the machine-wide lock
with timeout zero and wrap this command in the current activity collector:

```powershell
& $PY clean/dx12/tests/matched_stage_bench.py --run --build $BUILD --out $PRIVATE --shape 1440p --rounds 3 --repeats 24 --batch 16
```

Capture uses a native D3D12 Device wrapper. Each allocation is initially poisoned
with `0xCD`, then receives the runtime's original initialization and uploads.
Frame 0 initializes history; every launch of frame 1 is captured. The wrapper
reads the manifest ABI, copies argument values, resolves pointers to containing
live allocations, and records allocation-relative offsets. Entire pre/post
allocations are downloaded, including scratch and unwritten bytes. Multiple
pointers to one allocation remain aliases during replay. All eleven default
launches must be present. `capture.json` records the actual selected kernels.

Replay creates fresh HIP/native devices per stage. Both receive precisely the
same allocations and arguments from that capture. The exact declared code object,
argument layout and block size must match. Every post-launch allocation must be
byte-identical to the captured post state, including input buffers, scratch and
counter storage. The comparison is checked with a deliberately flipped bit.
This tests the comparator; the existing intermediate/mutant shader gates remain
necessary for promotion because FP16/FP8 outputs can hide rounding differences.

Each backend executes twice from one pre-state before batching is permitted.
The first launch must match; the second determines observed idempotence. If the
second launch differs, all changed allocations are restored and synchronized
before every measured launch, and its batch is forced to one. Exposure and
bottleneck synchronization words must be zero after each qualification launch
and at warm-up/round boundaries. Within timed batches they are not read back:
doing so would change the schedule being measured.

Each round warms both backends with at least three seconds of sustained stage
work. Paired measurement order alternates HIP/native and native/HIP. Raw samples
retain round, pair, backend, batch length, GPU bracket, synchronized wall,
submission and final wait. There are separate uninstrumented stream batches and
event batches. Every paired marker window continues until **both** `--repeats`
and `--minimum-window-seconds` (default 1 second) are satisfied, so short stages
produce enough activity intervals for the external collector. Every extra sample
is retained; actual pair counts and durations are saved per round. GPU brackets use ordinary Device events: HIP stream events and
native D3D12 timestamps including UAV ordering/query resolve. They are diagnostic
API spans, not isolated shader-body times. Upload/reset/readback work is outside
the timed intervals. There is no subtraction of unrelated timings.

Nonpositive/nonfinite GPU spans are retained with raw values and invalid flags.
Any invalid event sample invalidates that stage's event summary; synchronized
wall results remain independently reported. Samples are never silently filtered.
The original `tools/clk/clk.co` shader cycle/realtime probe records raw counters
and GHz snapshots immediately before/after each paired window. These are nearby
snapshots, not simultaneous clock measurements. Activity markers honor
`WG_GPU_EVIDENCE_MARKER_DIR` and include the actual configured `warm_seconds`,
round and UTC. The tool does not assert GPU isolation itself.

Outside paired windows, wall and calling-thread user+kernel CPU scopes identify
setup, qualification, individual allocation downloads/comparisons, reset/sync,
counter readbacks, clock probes and warm-up work. Warm-up records aggregate batch
and launch counts, submission/synchronization times and the largest batch. Only
batches exceeding 100 ms get per-batch reset/submission/synchronization detail.
These nested scopes overlap; they must not be summed. Calling-thread CPU time
does not include driver worker threads. Diagnostics are disabled during paired
measurement windows. The CPU-only executable self-test validates the marker
metadata contract and diagnostic suppression as well as the count/duration gate.

The stage delta table is for selecting experiments. Whole-frame history/module/
variant gates and minimally instrumented matched complete-path timings still
decide whether a shader change can be promoted.

Additional options:

- `--shape odd` or `--shape 4k-performance`, prepared in another private directory.
- `--kernels 'w64/post_tf_f32_p12_diet_vsit,bneck/swin_bneck6_nc'` captures the whole
  frame but replays only those stages.
- `--replay-only --run-tag UNIQUE` reuses a saved capture without overwriting it.
- `--batch 4` changes the maximum idempotent batch length.

No GPU execution was performed by the harness author during its CPU build gate.
