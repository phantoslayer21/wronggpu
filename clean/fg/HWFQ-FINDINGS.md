# Hardware flip queue and frame pacing: findings (read only)

Status: groundwork for frame generation presentation and pacing (Refs #96). Nothing here claims that frame generation
works, how well it paces, or when it will ship. Every statement is either a quotation of public documentation or what
a read-only query or our own test program printed on one test PC (a Radeon RX 9070 XT). No driver, registry, display
or system setting was changed for any of it.

## 1. Is the hardware flip queue active on the test PC?

Yes, as far as the graphics kernel reports it.

`clean/fg/hwfq_probe.cpp` (in this pull request) asks the graphics kernel through the documented `D3DKMTEnumAdapters2`
/ `D3DKMTQueryAdapterInfo` interface. It creates no device and does no GPU work. For the Radeon RX 9070 XT it printed:

    KMTQAITYPE_DRIVERVERSION: status 0x00000000, value 0xc80
    KMTQAITYPE_WDDM_2_9_CAPS: status 0x00000000, Value 0x00000006, HwSchSupportState 2 (STABLE), HwSchEnabled 1
    KMTQAITYPE_WDDM_3_0_CAPS: status 0x00000000, Value 0x0000000e, HwFlipQueueSupportState 2 (STABLE), HwFlipQueueEnabled 1, DisplayableSupported 1
    KMTQAITYPE_FLIPQUEUEINFO: status 0x00000000, MaxHardwareFlipQueueLength 0, MaxSoftwareFlipQueueLength 31, FlipFlags.FlipInterval 0

- Microsoft documents `D3DKMT_WDDM_3_0_CAPS.HwFlipQueueEnabled` as "Specifies whether the hardware flip queue is
  currently enabled for this GPU" ([D3DKMT_WDDM_3_0_CAPS](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmdt/ns-d3dkmdt-d3dkmt_wddm_3_0_caps)).
  The query type is `KMTQAITYPE_WDDM_3_0_CAPS`, "Supported starting with Windows 11, version 21H2"
  ([KMTQUERYADAPTERINFOTYPE](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/ne-d3dkmthk-_kmtqueryadapterinfotype)).
- `KMTQAITYPE_FLIPQUEUEINFO` is the older flip queue structure. The hardware flip queue's own driver capability is
  `MaxHwQueuedFlips` in `DXGK_DRIVERCAPS`, which user mode does not read. So the `MaxHardwareFlipQueueLength 0` above
  does not say anything about the hardware flip queue either way.
- The same PC, from `dxdiag /t` (read only): "Card name: AMD Radeon RX 9070 XT", "Driver Version: 32.0.32015.7029",
  "Driver Date/Size: 10/17/2026", "Driver Model: WDDM 3.2", "Hardware Scheduling: DriverSupportState:Stable
  Enabled:True", "Current Mode: 1920 x 1080 (32 bit) (60Hz)", "MPO Caps: Not Supported".
- Windows: `ver` printed "Microsoft Windows [Version 10.0.26200.9974]"; the registry's DisplayVersion is 25H2.
- The graphics drivers registry key (read only) holds `HwSchMode` = 0x2 (hardware-accelerated GPU scheduling on).

What Microsoft says the feature needs ([Hardware flip queue](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/hardware-flip-queue),
page dated 2025-04-24):
- "supported starting in Windows 11 (WDDM 3.0)".
- "Windows currently requires GPU hardware scheduling to be enabled in order for basic hardware flip queue to be enabled
  on officially released drivers."
- "Advanced hardware flip queue requires both basic hardware flip queue and GPU hardware scheduling stage 2
  functionality to be present." The query above does not say whether the advanced form is in use.

## 2. Which public presentation route gives target present times?

- **Composition swapchain** (`presentation.h`: `IPresentationFactory`, `IPresentationManager`,
  `IPresentationManager::SetTargetTime`, present statistics)
  ([programming guide](https://learn.microsoft.com/en-us/windows/win32/comp_swapchain/comp-swapchain)).
  - "a present can also have a *target time*, which is a system-relative timestamp (an interrupt-time) that describes
    the ideal time that the present should be shown".
  - Requirements: "GPU drivers supporting WDDM (Windows Device Driver Model) 2.0, and Windows 11 (Build
    10.0.22000.194) or greater. To use the composition swapchain API in the most performant way (direct scanout and
    independent flip, or *iflip*), systems will need GPU drivers supporting WDDM 3.0."
  - Surfaces must be allocated displayable to be eligible for independent flip. Microsoft documents that only for
    Direct3D 11 surfaces.
  - "If there are multiple *ready* presents, all but the latest ... will be *skipped*".
- **The hardware flip queue itself** takes a target time per queued flip ([Hardware flip queue](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/hardware-flip-queue)).
  - "When the OS submits a flip request to the hardware flip queue, it also sends the target flip time."
  - "The OS automatically converts existing interval-based Present API requests to timestamp-based flip calls."
  - For fixed refresh, the OS computes the target from the previous VSync and the present interval, minus half a
    refresh period.
- **Plain DXGI flip-model swap chains** (`IDXGISwapChain4`, interval-based `Present`) have no app-chosen target time.
  On a hardware-flip-queue system the OS turns their intervals into target times, per the page above.

Which of the two routes lands evenly on this hardware is what `fg_pacing_test` is for (section 4). It has not been
measured yet.

## 3. What AMD states about its own frame generation pacing (public sources only)

From the GPUOpen manual "AMD FSR Frame Generation (Swapchain)"
(<https://gpuopen.com/manuals/fidelityfx_sdk2/techniques/frame-interpolation-swap-chain/>). The page's sidebar says FSR
SDK 2.4.0; it covers technique version 4.0.0, dated 2026-10-28 in its version history:
- Pacing:
  - "The AMD FSR Frame Generation (Swapchain) handles frame pacing automatically."
  - Starting with 4.0.0, "the final presentation timing can be delegated to the AMD display driver instead of being
    paced by the swapchain itself". AMD calls this "Driver-Assisted Frame Pacing", "selected automatically when the
    driver reports support for it". Otherwise "the swapchain transparently falls back to its own pacing".
- Requirements, from the page's table: "AMD Software: Adrenalin Edition driver version 26.10.1 or later" and "AMD
  Radeon RX 9000 Series or later (AMD RDNA 4 architecture or later)". It is "optimized for Windows 11 with
  Hardware-accelerated GPU scheduling enabled". The page gives no Windows build number.
- Conditions:
  - "Exclusive fullscreen, or borderless fullscreen covering the display area";
  - VSync "Disabled";
  - "Presenting within the variable refresh rate window of the display";
  - "Switching to windowed presentation, enabling VSync, or presenting outside the VRR window" makes the swapchain
    fall back.
  - "VRR and tearing are only available in fullscreen mode."
- Swap chain: "On DirectX 12 it implements the `IDXGISwapChain4` interface"; the page advises
  `GetFrameLatencyWaitableObject`.
- The page does not mention the hardware flip queue. These notes therefore record AMD's term, "driver-assisted frame
  pacing", and make no claim about which display mechanism the driver uses for it.
- Whether this PC's driver 32.0.32015.7029 is Adrenalin 26.10.1 or later is not established here. AMD's 26.10.1
  release notes did not load when we tried them.

Nothing from AMD's SDK is used in this pull request.

## 4. The test programs and what they printed

- `clean/fg/hwfq_probe.cpp`: section 1.
- `clean/fg/fg_pacing_test.cpp`: presents alternating "real" and "stand-in" frames (two low-contrast greys) at a fixed
  target interval through two routes. It reports how evenly they reach the screen, using each route's own statistics:
  - route A, the composition swapchain: a Direct3D 11 device, displayable surfaces, `SetTargetTime` per present, and
    the PresentStatus / CompositionFrame / IndependentFlipFrame statistics;
  - route B, Direct3D 12 with a DXGI flip-model swap chain: `IDXGISwapChain4`, `FLIP_DISCARD`, a frame-latency
    waitable object, `Present(1, 0)` and `GetFrameStatistics`.

  Before presenting, it prints read-only context: the display mode and exact refresh period, the output's EDID identity,
  tearing support, the console display state, and DWM's completed frames over one second.

Build both from an x64 Native Tools prompt:

    cl /nologo /W4 /EHsc /std:c++17 /O2 hwfq_probe.cpp gdi32.lib dxgi.lib
    cl /nologo /W4 /EHsc /std:c++17 /O2 fg_pacing_test.cpp d3d11.lib d3d12.lib dxgi.lib dcomp.lib user32.lib advapi32.lib mincore.lib

### Runs on the test PC (2026-11-13): no on-screen times, because no live monitor was attached

- 13 runs of 5 seconds each, each in its own locked GPU slot: route A and route B, windowed and borderless; for
  route A also QPC versus interrupt-time targets and `ForceVSyncInterrupt(true)`.
- Every run exited 0. No run produced an on-screen interval, so this pull request reports no interval numbers.
- Route A, every run:
  - The program printed "IsPresentationSupported 1, IsPresentationSupportedWithIndependentFlip 1" and
    "D3D11_FEATURE_DISPLAYABLE DisplayableTexture 1".
  - It then printed "statistics received: independent-flip 0, composition-frame 0, present-status 300", and per
    present "skipped 200" (199 in some runs) with the rest queued. In the runs that printed per-present statistics, each composition frame took
    three presents and showed the last ("skipped", "skipped", "queued").
  - Statistics for successive composition frames arrived about 77 to 80 ms apart.
- Route B, every run: `GetFrameStatistics` returned S_OK with PresentCount 0, SyncRefreshCount 0 and SyncQPCTime 0. It
  printed "no statistic 300". A 5-second route B run (300 `Present(1, 0)` calls) took 16.5 to 17.5 s in its slot.
- DWM, before the test window drew: "completed composition frames" rose by 8 to 11 per second, with "framePeriod
  166666", that is 16.67 ms in 100 ns units.
- The output, read only:
  - `QueryDisplayConfig` gave target "", output technology 10 (DisplayPort, external), "EDID ids valid 0, manufacturer
    0x0000 product 0x0000".
  - `dxdiag` showed no monitor name or model, all colour primaries and luminance values 0, and "HDR Support: Unknown".
  - `GUID_CONSOLE_DISPLAY_STATE` reported 1 (on). That is the power setting, not the panel.
  - The session was the console session, not remote and not locked. The GPU engines were idle.
- Our reading, **inferred, not proven**: the DisplayPort monitor was off or in deep sleep (the owner was away).
  - Without scanout VSync, DWM runs on a slow timer, nothing can be independent-flipped, and DXGI has no refresh
    counts. That matches every line above.
  - A process without administrator rights cannot read the panel's power state directly, and nothing was elevated or
    changed to find out.

## 5. Experiment: AMD's frame generation swap chain with our stand-in frame (test only)

`clean/fg/ffx_swapchain_test.cpp` measures whether AMD's public FSR SDK frame generation swap chain can present a frame
that we supply, and how it paces it. Nothing from the SDK is in this repository, and nothing is shipped.
- The test loads AMD's signed loader and frame generation DLLs from a folder outside the repository.
- It creates the frame generation swap chain and the frame generation context.
- It supplies our stand-in frame through the documented `frameGenerationCallback`. AMD's own interpolation is never
  dispatched.
- It reports:
  - return codes and callback counts;
  - the GPU memory the API and DXGI report for the idle context;
  - on-screen intervals from the swap chain's frame statistics;
  - observable signs of the pacing path: the runtime's own messages, process CPU time, and the busiest threads with
    their priorities.

Fetch the SDK files at the pinned commit (tag v2.4.0), outside the repository, then build against them:

    python clean/fg/fetch_ffx_sdk.py <sdk folder>
    cl /nologo /W4 /EHsc /std:c++17 /O2 ffx_swapchain_test.cpp /I<sdk folder>\Kits\FidelityFX\api\include /I<sdk folder>\Kits\FidelityFX\framegeneration\include d3d12.lib dxgi.lib dcomp.lib user32.lib
    ffx_swapchain_test --sdk <sdk folder>\Kits\FidelityFX\signedbin --mode borderless --vsync 0

The licence of the fetched files is in `<sdk folder>\Kits\FidelityFX\docs\license.md`. No results yet: they need a live
monitor, like section 4.

**Not tested:** on-screen frame-to-frame evenness (mean, standard deviation, p99, misses) for either route. It needs a
live monitor (or a display emulator with an EDID). The same short runs will be repeated then, and their numbers added
with their records.
