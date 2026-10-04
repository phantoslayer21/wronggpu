# Game compatibility

WrongGPU is built to work with any DirectX 12 game that offers DLSS Super Resolution. It has been **validated on Cyberpunk 2077 only**. Every other game is untested until someone reports back. If you try a game, please send the report. It is the only way a game gets validated.

## Status levels

| Status | Meaning |
| --- | --- |
| **Validated** | The maintainer tested it on the listed game version against a fixed checklist: it loads, the DLSS 4.5 model runs every frame, a full benchmark pass finishes without a crash, the image is checked against native, and uninstall restores the game folder. |
| **Community: works** | At least three independent reports with logs show it running correctly. Not yet reproduced by the maintainer. |
| **Community: issues** | Reports show visual problems or crashes. The reports list what goes wrong. |
| **Not working** | Confirmed incompatible, with the reason (for example no DirectX 12 path, no DLSS option, or anti-cheat). |
| **Untested** | No reports yet. This is the default for every game not listed. |

## Games

| Game | Status | Tested on |
| --- | --- | --- |
| Cyberpunk 2077 | **Validated** | Game version 2.31, Windows 11, Radeon RX 9070 XT, Adrenalin 32.0.31035.1003, DLSS Quality, ray tracing Psycho, frame generation off |
| Call of Duty titles, Marvel Rivals | Not working | Anti-cheat built into the game. The installer refuses them. |
| Grand Theft Auto V Enhanced | Not working | BattlEye anti-cheat. The installer refuses it. |
| Helldivers 2 | Not working | GameGuard anti-cheat. The installer refuses it. |
| NBA 2K27 | Not working | Easy Anti-Cheat. The installer refuses it. |
| Everything else | Untested | [Send a report](../../issues/new?template=game-compatibility-report.yml) |

## Is my game a good candidate?

It needs all of these:

- It uses DirectX 12. DirectX 11 and Vulkan games are not supported.
- Its graphics menu offers DLSS Super Resolution on NVIDIA cards.
- It is single-player. **Do not use this in games with anti-cheat** (Easy Anti-Cheat, BattlEye, Vanguard and similar). The mod replaces DLL files and presents an NVIDIA graphics card to the game, which anti-cheat can treat as tampering and punish with a ban.

How to test and report: [docs/TESTING-OTHER-GAMES.md](docs/TESTING-OTHER-GAMES.md).
