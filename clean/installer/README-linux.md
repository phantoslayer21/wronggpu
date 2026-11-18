# The launcher on Linux

`WrongGPU` builds and runs as a native Linux program: the same screens, the same `src/app` and `src/wg_core` code as on
Windows. This file is for whoever maintains the launcher. The short version: **write the launcher the Windows way, and if
the Linux build then fails at a line, add that Windows call to `src/platform/posix`.**

## How it works

Three layers carry the shared code onto Linux. None of them touch the Windows build.

- **Stand-in headers and the POSIX layer** (`src/platform/posix`). On Linux, CMake puts this folder first on the include
  path, so `<windows.h>`, `<shlobj.h>`, `<winhttp.h>` and the other Windows headers the code includes resolve to
  stand-ins that declare the calls the code makes. `win32_posix*.cpp` implements them on POSIX: files and paths
  (UTF-8, `Z:\` is `/`, case-insensitive names), a registry kept in a file (`$XDG_DATA_HOME/WrongGPU/registry.ini`),
  processes, pipes, a small job object, known folders from the XDG variables. The conventions (string widths, path
  spelling, what fails on purpose) are in the header comment of `src/platform/posix/win32_posix.h`.
- **The window layer** (`win32_window_*.cpp`). It supplies the Win32 window calls `launcher_ui.cpp` makes (windows,
  the message loop, timers, paint, DPI, capture, clipboard, text boxes) on SDL3, turning SDL's events into the `WM_*`
  messages the window procedure already handles. The app draws its own title bar, so there is no libdecor.
- **The Cairo `Gfx`** (`src/app/ui_gfx_cairo.cpp`). All screens draw through one class, `Gfx` (`ui_shell.h`). Windows
  implements it on Direct2D and DirectWrite (`ui_gfx.cpp`); Linux implements the same methods on Cairo, Pango and
  fontconfig. Fonts are vendored in `assets/fonts` (Selawik for text, Noto Sans Math after it for the symbols Selawik lacks, such as the arrow, Fluent System Icons for icons) and found next to
  the program, so the look does not depend on the user's fonts.

## The rule

A Windows call that the layer lacks **fails the Linux build** at that line. That is the point: the error tells you what
to add. Add it in `src/platform/posix` (declare it in the stand-in header, implement it in a `win32_posix*.cpp` file,
add a check to `src/platform/posix/tests/posix_layer_tests.cpp`). Do not add `#ifdef` branches around shared code unless
the feature is Windows' own (UI Automation, the DWM title bar, the old GDI+ wizard).

One list in `CMakeLists.txt`, `WRONGGPU_APP_SOURCES`, holds the `src/app` files of both launchers, and `wg_core` has one list
for both, so a new shared file is compiled on Linux without any Linux-specific edit. Two short lists name the exceptions:
`WRONGGPU_APP_SOURCES_WINDOWS` (the GDI+ wizard, the Direct2D drawing) and `WRONGGPU_APP_SOURCES_LINUX` (the live-frame
hook); a new shared file goes in the first list only. The Windows-only targets stay under `if(WIN32)`.

Two mistakes get past a Windows-only build, so `tools/ci/check_portable.py` looks for them (it runs as the ctest tests
`portable` and `portable-selftest`, and as a step of the Linux CI job):

1. A `std::filesystem::path` built straight from a `wstring` in `src/wg_core` or `src/app`. Use `wg::FsPath(wstring)`
   (`wg_core/util.h`): it spells a Windows-style path the way the host needs.
2. Direct2D, DirectWrite, WIC, DWM or UI Automation names in a shared `src/app` file outside `#ifdef _WIN32`. Names the
   POSIX layer stands in for (the DWM calls, today) and names a file defines itself are fine. A file that exists only on
   Windows goes in `WINDOWS_ONLY` at the top of the script.

## Build and test on Linux

You need CMake 3.16+, Ninja, GCC 11+, Python 3, the development files of Cairo, Pango, fontconfig and libjpeg-turbo (`libjpeg-turbo8-dev`), and SDL3 3.4
(static; most distributions do not ship it yet: `build-appimage.sh` builds it, and `-DCMAKE_PREFIX_PATH=<its install>`
points CMake at one).

    cmake -S clean/installer -B out/linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DWRONGGPU_TESTING=ON -DWRONGGPU_PRESET_K=OFF -DCMAKE_PREFIX_PATH=<SDL3 install>
    cmake --build out/linux
    ctest --test-dir out/linux --output-on-failure

A `WRONGGPU_TESTING` build without `WRONGGPU_PAYLOAD_DIR` makes its own stand-in payload and test map. The tests never
open a window: they run SDL on its `offscreen` driver and give the program a home, XDG folders and registry file of its
own. Run it yourself the same way (`SDL_VIDEO_DRIVER=offscreen`) unless you want the real window. `linux-ui` renders the
screens to PNG and drives the live window with scripted input; `launcher-native` runs the launcher's verbs suite
(`tests/run_launcher_tests.py --native-linux`) against the binary; like CI's launcher build it needs
`-DWRONGGPU_PRESET_K=OFF`, and its one check that fails on main's own launcher too (#219) is skipped while it fails
(`tests/native_linux.py` `KNOWN_ON_MAIN`).

## Build the AppImage

One script, in the pinned Ubuntu 22.04 container so the AppImage runs on any desktop Linux with glibc 2.35 or newer, with
the X11 client libraries (libX11, libxcb) that every desktop provides, and the other host libraries the script lists
(`HOST_LIBS`: the glibc family, zlib and libuuid):

    clean/installer/build-appimage-container.sh --payload <release payload folder>     # podman or docker
    clean/installer/build-appimage-container.sh --test-payload                         # smoke build, never ship

The payload folder is the one you give `WRONGGPU_PAYLOAD_DIR` on Windows; `--version` and `--out` are optional (version:
`WRONGGPU_VERSION` in `CMakeLists.txt`; output: `build/appimage/out`). Without a container, run `build-appimage.sh`
inside any Ubuntu 22.04 as root (`WG_ALLOW_OTHER_HOST=1` skips the host check for a local trial; the AppImage then
may not run on older systems). The script lists no sources and no screens, only the host libraries it expects from the
system (`HOST_LIBS`, with the reason for each): `cmake --install` decides what ships (see the `install()` lines at the end
of `CMakeLists.txt`; a new asset folder needs an `install()` line there) and linuxdeploy copies in the libraries the
binary needs. The X11 libraries are left to the system on purpose: a second copy would clash with the desktop's. The last
step fails the build if any file needs a glibc or libstdc++ newer than Ubuntu 22.04's, a library that is neither in the
AppDir nor in `HOST_LIBS`, or the fonts are missing.

The CI job `.github/workflows/linux-appimage.yml` runs only when the maintainer starts it from the Actions tab (Linux AppImage,
Run workflow); no pull request or push starts it. It builds the test stand-in payload (`--test-payload`) and keeps the AppImage for 3
days (`actions/upload-artifact`, pinned to a commit SHA); it never ships. The release AppImage comes from
`build-appimage-container.sh --payload <release payload folder>`.

**Updating a pin.** The tool versions, their SHA-256 values and the container image digest are the first lines of
`build-appimage.sh` (the digest is also in the workflow; a CI step fails if the two differ). Change a version and its
checksum together, taken from the project's own release page. Use exact release tags, never "latest" or "continuous".

## Known limits

- **No upscaler yet:** with a released Proton, WrongGPU's upscaler does not start. vkd3d-proton's dxil-spirv cannot
  translate the matrix kernels until dxil-spirv pull request 362 is in a Proton release. For now the native launcher
  covers install and settings only; it does not make a game upscale.
- **Art:** Cairo reads PNG and libjpeg-turbo reads JPEG (`src/app/jpeg_decode_linux.cpp`; Windows uses WIC), so Steam's
  art (hero, header, capsule) shows. Other formats give the page's plain plate, as for a game without art.
- **No downloads:** WinHTTP is answered "not available" and `DownloadsAvailable()` (`util.h`) is false natively, so the in-launcher
  downloads (OptiScaler's edition, the release check) do not run on Linux. Settings says so (the OptiScaler row asks for the
  player's own OptiScaler download, the release check has no switch), and a game's install without one says the same in words.
- **Uninstall:** an AppImage has no installer entry, so Settings > Uninstall WrongGPU (`uninstall-appimage --confirm` on the command line)
  deletes the AppImage file named by `$APPIMAGE` and WrongGPU's data folder (`$XDG_DATA_HOME/WrongGPU`), after one confirmation that names
  both, and nothing else. It refuses unless `$APPIMAGE` is a plain file (not a link) called `WrongGPU...AppImage` with an AppImage's header,
  and the system itself ties that file to this running program: `$APPDIR` (resolved case-sensitively) is a FUSE mount in `/proc/self/mountinfo`
  that this program runs on, and a live process whose executable is that very file (same device and inode) holds the mount's `/dev/fuse`
  connection (`fuse_connection` in its `fdinfo`; the AppImage runtime serves its own mount from its own file). An unpacked AppImage
  (`APPIMAGE_EXTRACT_AND_RUN`) or an unreadable `/proc` gives no such proof, so it refuses. The file is checked again just before it
  is deleted (same device and inode, still a plain AppImage) and goes first, then the data. It also refuses while a game still has
  WrongGPU (uninstall it from the game first).
  The player's own ignore list (`ignore-files.txt`) stays, as in the Windows launcher's "Remove data". There is no "Install for this user"
  on Linux: the first-run sheet and Settings do not offer it.
- **Engine.ini for Borderlands 4 and Hogwarts Legacy:** it goes into the game's Proton prefix (a Steam game:
  `<library>/steamapps/compatdata/<appid>/pfx/drive_c/users/steamuser`). Until Steam has made that folder (the game was started once) the
  launcher says so and writes nothing. A game that is not from Steam needs the Windows user folder of its Wine prefix once:
  `WrongGPU windows-folder "<game exe>" "<folder>"`, remembered in `windows-folders.txt` in the data folder. There is no text field for it in
  the window yet (no file picker either).
- **Ray Reconstruction preparation:** the game prepares its kernels at its first start (its shader caches are Proton's), so the launcher
  runs no helper and its Play never fails on it.
- **No restorer:** `WrongGPU-Restore.exe` is a Windows program. A Linux install does not embed or write it.
- **UI Automation:** the stubs answer nothing; there is no screen-reader support yet (AT-SPI is the Linux equivalent).
- **GPU check:** there is no DXGI here, so the launcher reads Linux's own device list (`/sys/class/drm`): the card's name
  comes from a built-in list of the RDNA 4 PCI ids (`kRdna4Cards` in `checks.cpp`; any other card is named by its code), and
  only an RDNA 4 card (RX 9000 series) counts as supported. It names the card the system has, not the one a game picks.
- **Parked:** the Ray Reconstruction preparation (it must run in the game's Wine prefix).
- **Not emulated:** NTFS junctions, `.lnk` shortcuts, Apps & features, UAC elevation, share-mode locks between processes.
  The launcher tests skip the checks that depend on them and print why.
- **Game running:** the per-game state lock works between Linux processes (file locks), and uninstall, repair and update
  refuse while one of your Wine processes has a WrongGPU DLL or program loaded, as Windows does (`/proc` mappings). If
  one of your Wine processes cannot be read, the launcher refuses rather than guess. Not covered: other users' processes,
  files a program only has open, and a game started after the check.
- **Hidden attribute:** kept as the user extended attribute `user.wronggpu.hidden`, so it shows only on a file system that
  allows those. Read-only is the missing write bit, as before.
