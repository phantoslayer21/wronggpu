<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Low latency during frame generation

Set `fg_low_latency=on` in the per-game `wronggpu-setup/settings.ini` to enable this work. The setting defaults to
`off`. It applies when the ASI is loaded. The game's FG toggle can still be changed while playing.

## Ownership and timing

- **Pass-through.** With the switch off, resolving and calling the Reflex and PC Latency functions uses the existing
  path. With the switch on, known function lookups retain the downstream result and wrap successful function pointers
  so a game can cache them before FG starts. The wrappers pass the original arguments and results through while FG
  is off, during first-frame priming, and after the session stops generating.
- **Generating.** Once a real interval records generated images, the host handles the sleep, mode, state and marker
  calls. The sleep runs before simulation or input. A sleep already routed for a native frame keeps its owner across a
  mode transition. A missing or stale estimate, an already started simulation, or a duplicate local sleep returns
  immediately. A supported state query reports pacing availability. Latency reports and ping messages are unavailable
  locally. The game's off mode disables prediction and its frame limit can still apply independently. Boost uses the same
  timing as normal low latency.
- **Menu, resize and loss.** A real-only interval or the menu-copy latch stops prediction. Resize, chain detach, device
  loss and teardown clear the timing history and cancel a wait. The game toggle also cancels a wait. Once generation
  resumes, the pacer learns from fresh samples.

The PC Latency marker records associate simulation start, render submit end and present start with the game's native
frame index. The Present-start marker on the calling thread links the native record to the existing FG interval token.
A missing, stale or out-of-order marker supplies no estimate. Resource tags and camera constants retain their existing
assembly path.

The presenter's latest scheduled real target plus one real-frame period gives the next real-frame due time. Simulation
start to render submit end estimates CPU work. Render submit end to the presenter's fence-ready signal estimates the
remaining GPU and handoff time. The estimates rise immediately and decay gradually. The sleep releases at:

```text
next real-frame due time - predicted CPU time - predicted GPU time - margin
```

The margin reserves the time from the first generated image to the real target, the presenter's submit margin and
0.5 ms of slack. The wake is clamped between now and 50 ms ahead. A recent completion and plan are required for
prediction. Timer failure or loss of ownership releases the current call without forwarding it to another sleep owner.

For a frame paced before input, Present skips the legacy presenter-room and facade release waits. The credit check is
nonblocking and a closed admission uses the existing safe fallback. Unpaced frames retain the existing Present
behavior. The facade, in-flight compute depth and queue fences still protect resource reuse. The pacer receives times
and frame IDs only. It supplies no resources or settings to the executor. Given identical inputs, the kernels,
interpolation parameters and pixel math are identical for the same generated count and index. Timing can change count
admission, drops and the next frame the game renders.

The native layouts, marker IDs and call signatures come from the public Streamline interface.
[Reflex header](https://github.com/NVIDIA-RTX/Streamline/blob/v2.7.2/include/sl_reflex.h),
[PC Latency header](https://github.com/NVIDIA-RTX/Streamline/blob/v2.7.2/include/sl_pcl.h),
[Reflex guide](https://github.com/NVIDIA-RTX/Streamline/blob/v2.7.2/docs/ProgrammingGuideReflex.md),
[PC Latency guide](https://github.com/NVIDIA-RTX/Streamline/blob/v2.7.2/docs/ProgrammingGuidePCL.md).

## Live qualification

First verify on Radeon that the game's sleep call reaches WrongGPU once per real frame and precedes input sampling.
Check that native simulation, submit and Present-start markers match the presented token, including when the game
pipelines frames. Record the planned wake and actual release, marker and fence times, the latency owner, and any
resource wait after simulation. Exercise warm-up, menu entry and exit, resize and the FG toggle.

At 2x in the same GPU-bound scene, compare `fg_low_latency=off` and `fg_low_latency=on` across repeated runs.
Require at least 5 ms lower p95 latency, an FG presentation rate within 3%, and no worse p99 display intervals or
dropped-frame rate. The stretch target is one real-frame period less latency.
