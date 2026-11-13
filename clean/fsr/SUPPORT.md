# Support diagnostics

WrongGPU keeps `wronggpu-diagnostics-<module>.jsonl` next to its frontend DLLs.
The text log remains available. A session is identified by process ID and Windows
process creation time; each JSON record has a module, thread, sequence and UTC/QPC
time. Report codes include `WG-HIP-719`, `WG-HIP-API`, `WG-NO-PROVIDER`,
`WG-HOOK-TABLE`, `WG-CREATE` and `WG-SPLIT`. Numeric API errors and the original
message are retained. Codes classify the observed failure boundary, not its cause.

The frontend retains the latest 64 mapped frame metadata records per module.
Metadata includes context/frame IDs, render/output/origin dimensions, formats,
resource states, jitter, motion scaling/offset, exposure, reset and command-list/
queue identity. Resource bytes, images and learned data are never captured.
Object IDs identify objects within a session and cannot be used as replay addresses.
The first eight frames and every 300th frame have begin/backend-return records;
failures include the recent metadata window and CPU detection-stack program counters.
Background module maps allow matching those counters to module paths/offsets without
loader queries on the failing thread. Missing/stale maps remain unknown. A detection
stack is not necessarily the original faulting stack.

Provider observations record AMD/network/stand-in/comparison/unavailable states.
CPU success is not proof of GPU completion. Backend failures also record the public
D3D12 device-removal HRESULT. Effective frontend settings include automatic process
profiles and hosted-mode overrides. Loaded module paths/hashes distinguish binary
identity from version labels. An already loaded HIP runtime is queried through
`hipRuntimeGetVersion`; diagnostics never initialize HIP or launch a GPU probe.

File writing and cached module hashing run on the existing background logger.
Frame capture is a bounded CPU metadata copy. No GPU commands, waits, signals,
debug-layer enablement or numerical/history changes are introduced. The queue is
bounded; diagnostic files rotate at 16 MiB with one `.old` file per module. Queue
drops are reported. If the writer/file is unavailable, legacy logging continues.
Abrupt termination/power loss can lose queued records; absence of an error is not
proof of health. Module-file hashes describe on-disk files at first observation,
not a hash of mapped memory. Real-game overhead has not been qualified yet.

Use the installer's **Collect logs** button, or its existing `/COLLECT` route, to
save a support ZIP. It includes `support-manifest.json`, exact available settings,
logs, installation metadata, Windows events, adapter identities/VRAM/driver data,
HIP candidate file hashes and structured failure/frame/runtime observations.
The manifest records hashes of the actual included bytes and flags truncation,
unreadable/omitted artifacts, malformed JSONL and collection limits. Data is bounded
to 64 source files, 8 MiB tails and a 64 MiB source-data budget. The newest diagnostic
records are selected first. Memory dumps and learned/model binaries are not included
automatically. No upload takes place; reports contain local paths/user names.

Developer analysis: `python clean/fsr/tools/analyze_support.py <support.zip> --out <summary.json>`.
It verifies captured artifact hashes, preserves session/incident references, flags
coverage gaps and exports reproduction context. It does not extract/run archive
contents, read memory dumps or claim a GPU replay from metadata. Legacy text reports
are accepted with an explicit evidence limitation. A hash mismatch returns exit 1.

Release builds retain private optimized PDBs outside customer ZIPs. Match a module
hash/source pin before using its symbols to investigate detection-stack offsets.
Runtime/kernel payloads remain separately pinned. The customer ZIP still contains
only installer, checksums, release notes, licence and notices.
