# Roadmap

These are plans, not promises. They have no dates, and their order can change with contributor interest and test
results from real hardware. Discuss an item in its issue or on the
[WrongGPU Discord server](https://discord.gg/745z9MfKwm) before starting a large change. Every contribution follows
[CONTRIBUTING.md](CONTRIBUTING.md), including the clean-room statement, sign-off, and testing rules.

## Game fixes and test tooling

- **Hogwarts Legacy:** restore Ray Reconstruction. The game's November 2026 update greyed out its Ray Reconstruction
  option, and the cause is still unknown.
- **RESIDENT EVIL requiem:** find the cause of the rare single overexposed or underexposed frame at a scene change.
- **Ray Reconstruction test setup:** make the end-to-end test (`clean/rr_f/runtime/tests/e2e.py`) set the compile-wait
  setting that it needs.
- **Launcher build:** fix the remaining compiler warnings.

## Ray Reconstruction speed

Reduce Ray Reconstruction's GPU time per frame on RX 9000 cards without changing its output. Possible work includes
faster kernels for the body and output stages, fewer passes, and better use of the hardware's matrix instructions.

A change intended to keep the output bit-identical must include the comparison described in
[BUILDING.md](BUILDING.md#comparing-kernels-with-a-release) and pass a GPU test before merge.

## More hardware: RDNA 3

The kernels currently target RDNA 4 (gfx1201 and gfx1200). Supporting RDNA 3 cards would require kernels without
RDNA 4's FP8 matrix instructions, followed by output and performance testing. The first step is an investigation.
RDNA 3 is not supported now.

## Vulkan and Linux

WrongGPU currently supports DirectX 12 games on Windows. A Vulkan path for Vulkan games and DirectX 12 games under
Proton on Linux would require a Vulkan executor for the runtime and a way to receive the game's upscaler inputs. The
first step is a design discussion.

## Always welcome

- Game reports and additions to [COMPATIBILITY.md](COMPATIBILITY.md).
- More tests for code with insufficient coverage, especially CPU tests that CI can run.
- Clearer build instructions and corrections to [BUILDING.md](BUILDING.md) based on a fresh build.
