# Compatibility report template (for #compatibility and the README)

Post one report per game. The more of this you fill in, the faster a game can be validated.

**Game**
- Name and store (Steam / Epic / other), and game version or build
- Anti-cheat present? (yes / no / unknown)
- What upscaling options does the game offer? (DLSS / FSR / XeSS / none)

**System**
- GPU model and AMD driver (Adrenalin) version
- Windows version
- Mod version (installer version)
- Other mods or overlays running (for example OptiScaler, ReShade, Cyber Engine Tweaks, MSI Afterburner)

**Result**
- Does the game launch with the mod installed? (yes / no)
- Does it crash? When?
- Does the image look right? (black screen, ghosting, flicker, wrong colors, shimmer, none)
- Resolution, quality mode and key settings (for example ray tracing on or off)
- FPS without the mod and with the mod (average and 1% lows if you have them)

**Evidence**
- The zip from **Collect logs** (in the installer, or **Modify** for the game in Installed apps)
- Screenshot or short clip of any problem

**Maintainer's validation checklist** (what a game needs before it moves to "validated"):
- At least 3 independent reports with no crash, from more than one GPU and driver
- No visual corruption in normal play
- FPS gain shown against the game's native or TAA setting
- Uninstall restores the original files cleanly
