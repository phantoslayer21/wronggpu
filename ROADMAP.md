# Roadmap

What the project would like to work on next. These are plans, not promises: there are no dates, and the order can
change with what contributors pick up and what testing on real hardware shows. Discuss an item in its issue or on the
[WrongGPU Discord server](https://discord.gg/745z9MfKwm) before starting a large change. Every contribution follows
[CONTRIBUTING.md](CONTRIBUTING.md) (clean-room statement, sign-off, tests).

## Next release (0.6.1)

- **Preset K** back as a second Model choice (the code is in `clean/k`; it is built and tested, but not offered in
  0.6.0).
- **Hogwarts Legacy:** Ray Reconstruction again after the game's November 2026 update changed its game-interface
  integration.
- **RESIDENT EVIL requiem:** find the cause of the rare single over- or under-exposed frame at a scene change.
- Test tooling: the Ray Reconstruction end-to-end test (`clean/rr_f/runtime/tests/e2e.py`) should set the compile-wait
  setting it needs by itself; tidy the remaining compiler warnings of the launcher build.

## Ray Reconstruction speed

Make Ray Reconstruction cheaper per frame on RX 9000 cards without changing its output: faster kernels for the body
stages and the output stage, fewer passes, better use of the hardware's matrix instructions. Any change that is meant
to keep the output bit-identical shows the comparison (BUILDING.md, section 6) and is checked on a GPU before merge.

## More hardware: RDNA 3

Today the kernels target RDNA 4 (gfx1201 / gfx1200). Running on RDNA 3 cards needs kernels without RDNA 4's
FP8 matrix instructions and a check of what the results and the speed would be. This starts as an investigation.

## Vulkan and Linux

Today WrongGPU works with DirectX 12 games on Windows. A Vulkan path (for Vulkan games, and for DirectX 12 games under
Proton on Linux) would need a Vulkan executor for the runtime and a way to receive the game's upscaler inputs there.
This starts as a design discussion.

## Always welcome

- Game reports and additions to [COMPATIBILITY.md](COMPATIBILITY.md).
- Tests for code without enough of them, especially CPU tests that CI can run.
- Documentation: clearer build steps, fixes to BUILDING.md from your own fresh build.
