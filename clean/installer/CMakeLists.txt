# SPDX-License-Identifier: MIT
# WrongGPU installer (spec 30). Windows: MSVC or MinGW-w64. Linux: the core (wg_core) and its tests, on a POSIX
# implementation of the Windows calls the core makes (src/platform/posix, README-linux.md).
#   cmake -S . -B build -DWRONGGPU_PAYLOAD_DIR=<release payload folder> [-DWRONGGPU_MAP=<map.json>] [-DWRONGGPU_VERSION=x.y.z]
#   cmake --build build --config Release
# -DWRONGGPU_PRESET_K=OFF builds an installer for a release that ships Model M only: no Preset K choice or wording,
# /PRESET=K is a bad command line (exit 10), the Preset K converter map is not built in (WRONGGPU_MAP_K is not read),
# only Model M's cache is written, OptiScaler's UpscalerIndex is always 0.
# -DWRONGGPU_DIRECT_ROUTE=OFF builds an installer that requires OptiScaler: no method choice and no wording of the
# direct FSR route, /DIRECTFSR is a bad command line (exit 10), a silent install without /OPTISCALER= is refused
# (exit 2), an earlier direct install is removed before the hosted one is made, the FSR 3.0 component DLLs of the
# payload folder are not built in (src/setup_route.h).
cmake_minimum_required(VERSION 3.16)
project(wronggpu_installer LANGUAGES CXX)
if(WIN32)
  enable_language(RC)
else()
  enable_language(ASM)  # resources.S: the embedded payload (tools/make_payload.py)
endif()
# MinGW's _mingw.h (reached through <functional> in context.h before base.h) would otherwise fix Windows 7 as the target.
if(MINGW)
  add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # no runtime DLL needed on the player's PC

set(WRONGGPU_PAYLOAD_DIR "" CACHE PATH "Folder with WrongGPU's amd_fidelityfx_dx12.dll, the runtime and its kernels (one release build)")
set(WRONGGPU_MAP "${CMAKE_CURRENT_SOURCE_DIR}/../../data/preset-M.v1.map.json" CACHE FILEPATH "Converter map (spec 08)")
set(WRONGGPU_VERSION "0.6.0" CACHE STRING "Installer version")
set(WRONGGPU_BUILD_ID "" CACHE STRING "The build's own name, e.g. 0.5.5-launcher-dev (empty: the version)")
set(WRONGGPU_PAYLOAD_NAME "" CACHE STRING "The release build the payload folder holds, e.g. 0.5.5-ea (empty: the version)")
set(WRONGGPU_MAP_K "${CMAKE_CURRENT_SOURCE_DIR}/../../data/preset-K.v1.map.json" CACHE FILEPATH "Preset K converter map")
# Ray Reconstruction (src/wg_core/rr.h): the RR plugin and runtime of one RR release build (a folder with
# plugins\WrongGPU-RR.asi and wronggpu_rr_runtime.dll; empty: an installer without them, RR games are installed without
# RR) and the RR converter map (preset F). The folder may also hold builds of the plugin for one version of the game
# interface it hooks (plugins\<major>.<minor>\WrongGPU-RR.asi, with interposer.json, the RR proxy's binding file that
# names that interface's file). Never OptiScaler (owner 2026-11-02: tools/make_payload.py refuses any OptiScaler file; the
# launcher downloads WrongGPU's edition of OptiScaler itself, src/wg_core/optiedition.h). Local build inputs only. A
# WRONGGPU_TESTING build without a folder gets generated stand-ins (tests/make_mock_rr.py, all of the above) and the spec
# team's synthetic preset-F map, so its tests never need a real RR file.
set(WRONGGPU_RR_DIR "" CACHE PATH "Folder with the RR plugin (plugins/WrongGPU-RR.asi) and the RR runtime (wronggpu_rr_runtime.dll)")
set(WRONGGPU_MAP_F "" CACHE FILEPATH "RR converter map (preset F); empty: data/preset-F.v1.map.json (test builds: the synthetic rr-f-synth/synth-r0 map)")
# Model L (#89): its converter map, embedded for "model convert --preset=L" (the model's folder: <model>\L). Default: the
# shipped map (data/preset-L.v1.map.json, the spec side's revision v1.1). Tests of the conversion itself give a map of
# its exact layout with their own digests (tests/run_model_l_tests.py --write-test-map); empty: a build without Model L.
set(WRONGGPU_MAP_L "${CMAKE_CURRENT_SOURCE_DIR}/../../data/preset-L.v1.map.json" CACHE FILEPATH "Model L converter map (preset L); empty: none")
if(WRONGGPU_MAP_L AND EXISTS "${WRONGGPU_MAP_L}")
  set(MAP_L_ARGS --map-l ${WRONGGPU_MAP_L})
  set(MAP_L_FILE ${WRONGGPU_MAP_L})
else()
  set(MAP_L_ARGS)
  set(MAP_L_FILE)
endif()
option(WRONGGPU_PRESET_K "Preset K in the installer (second model choice, /PRESET=K, its converter map); OFF = Model M only" ON)
if(WRONGGPU_PRESET_K)
  set(PRESET_K_MAP_ARGS --map-k ${WRONGGPU_MAP_K})
  set(PRESET_K_MAP_FILE ${WRONGGPU_MAP_K})
  set(PRESET_K_DEFINITION WRONGGPU_PRESET_K=1)
else()
  set(PRESET_K_MAP_ARGS)
  set(PRESET_K_MAP_FILE)
  set(PRESET_K_DEFINITION WRONGGPU_PRESET_K=0)
endif()
option(WRONGGPU_DIRECT_ROUTE "The direct FSR route in the installer (the wizard's advanced choice, /DIRECTFSR, a silent install without /OPTISCALER=); OFF = OptiScaler is required" ON)
if(WRONGGPU_DIRECT_ROUTE)
  set(DIRECT_ROUTE_PAYLOAD_ARGS)
  set(DIRECT_ROUTE_DEFINITION WRONGGPU_DIRECT_ROUTE=1)
else()
  set(DIRECT_ROUTE_PAYLOAD_ARGS --hosted-only)
  set(DIRECT_ROUTE_DEFINITION WRONGGPU_DIRECT_ROUTE=0)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../converter converter)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Every source file stays below the spec side's scan limit (256 KB, tools/nv_leak_scan.py): checked when the build is
# configured and again as the test "source-size".
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_source_size.py
                RESULT_VARIABLE SOURCE_SIZE_RESULT OUTPUT_VARIABLE SOURCE_SIZE_OUTPUT)
message(STATUS "${SOURCE_SIZE_OUTPUT}")
if(NOT SOURCE_SIZE_RESULT EQUAL 0)
  message(FATAL_ERROR "A source file is at or over 256 KB (tools/check_source_size.py): split it.")
endif()
option(WRONGGPU_TESTING "Isolate registry and enable mock-only HIP candidate selection (never ship)" OFF)
if(NOT WRONGGPU_PAYLOAD_DIR AND WRONGGPU_TESTING AND NOT WIN32)
  # a Linux test build makes the stand-in payload and test map the Windows test build is given by hand: the harness's own
  # (tests/run_installer_tests.py --prepare-only), the model case tests/run_launcher_tests.py expects
  set(MOCK_PAYLOAD_WORK ${CMAKE_CURRENT_BINARY_DIR}/mock-payload)
  execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_installer_tests.py --work ${MOCK_PAYLOAD_WORK} --prepare-only
                  RESULT_VARIABLE MOCK_PAYLOAD_RESULT OUTPUT_QUIET)
  if(NOT MOCK_PAYLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "tests/run_installer_tests.py --prepare-only failed")
  endif()
  set(WRONGGPU_PAYLOAD_DIR ${MOCK_PAYLOAD_WORK}/prepared/payload)
  set(WRONGGPU_MAP ${MOCK_PAYLOAD_WORK}/prepared/test-map.json)
endif()
if(NOT WRONGGPU_PAYLOAD_DIR)
  message(FATAL_ERROR "Set WRONGGPU_PAYLOAD_DIR to the release payload folder (see README.md).")
endif()
set(RR_PAYLOAD_DIR "${WRONGGPU_RR_DIR}")
if(NOT RR_PAYLOAD_DIR AND WRONGGPU_TESTING)
  set(RR_PAYLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/mock-rr)
  execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_mock_rr.py ${RR_PAYLOAD_DIR} RESULT_VARIABLE MOCK_RR_RESULT)
  if(NOT MOCK_RR_RESULT EQUAL 0)
    message(FATAL_ERROR "tests/make_mock_rr.py failed")
  endif()
endif()
set(MAP_F_FILE "${WRONGGPU_MAP_F}")
if(NOT MAP_F_FILE)
  if(WRONGGPU_TESTING)
    set(MAP_F_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../data/rr-f-synth/synth-r0.map.json)
  else()
    set(MAP_F_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../data/preset-F.v1.map.json)
  endif()
endif()
if(RR_PAYLOAD_DIR)
  set(RR_PAYLOAD_ARGS --rr ${RR_PAYLOAD_DIR})
  file(GLOB_RECURSE WRONGGPU_RR_FILES CONFIGURE_DEPENDS "${RR_PAYLOAD_DIR}/*")
else()
  set(RR_PAYLOAD_ARGS)
  set(WRONGGPU_RR_FILES)
endif()
if(EXISTS "${MAP_F_FILE}")
  set(MAP_F_ARGS --map-f ${MAP_F_FILE})
else()
  set(MAP_F_ARGS)
  set(MAP_F_FILE)
endif()

file(GLOB_RECURSE WRONGGPU_PAYLOAD_FILES CONFIGURE_DEPENDS "${WRONGGPU_PAYLOAD_DIR}/*")
set(GEN ${CMAKE_CURRENT_BINARY_DIR}/gen)
set(MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/src/setup.manifest)
# The application manifest is embedded once: by the MSVC linker (CMake merges .manifest sources into the manifest the
# linker generates), or through the resource script with MinGW, whose linker embeds none.
if(MSVC)
  set(MANIFEST_ARGS)
  set(MANIFEST_SOURCE ${MANIFEST})
else()
  set(MANIFEST_ARGS --manifest ${MANIFEST})
  set(MANIFEST_SOURCE)
endif()
add_custom_command(
  OUTPUT ${GEN}/payload_table.h ${GEN}/resources.rc ${GEN}/resources.S ${GEN}/app.ico
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_payload.py
          --payload ${WRONGGPU_PAYLOAD_DIR} --map ${WRONGGPU_MAP} ${PRESET_K_MAP_ARGS} --version ${WRONGGPU_VERSION}
          --build-id=${WRONGGPU_BUILD_ID} --payload-name=${WRONGGPU_PAYLOAD_NAME}
          ${DIRECT_ROUTE_PAYLOAD_ARGS} ${RR_PAYLOAD_ARGS} ${MAP_F_ARGS} ${MAP_L_ARGS} ${MANIFEST_ARGS} --out ${GEN}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_payload.py ${MANIFEST} ${WRONGGPU_MAP} ${PRESET_K_MAP_FILE} ${MAP_F_FILE} ${MAP_L_FILE} ${WRONGGPU_PAYLOAD_FILES} ${WRONGGPU_RR_FILES}
  COMMENT "Embedding the payload and the converter map")

# The launcher's sources, for both targets: a new shared file goes in WRONGGPU_APP_SOURCES only, never in the two lists below it.
set(WRONGGPU_APP_SOURCES src/app/main.cpp src/app/cli.cpp src/app/launcher_ui.cpp src/app/ui_art.cpp src/app/ui_widgets.cpp
    src/app/ui_library.cpp src/app/ui_game.cpp src/app/ui_settings.cpp src/app/ui_uia.cpp src/app/ui_render.cpp src/app/ui_words.cpp
    src/app/ui_dump.cpp src/app/ui_frame.cpp src/app/ui_play.cpp src/app/ui_rrprep.cpp src/app/ui_rrconfig.cpp src/app/ui_activate.cpp
    src/app/ui_checks.cpp src/app/payload_embed.cpp)
# Windows only: the GDI+ wizard and the Direct2D/DirectWrite drawing (Linux draws through ui_gfx_cairo.cpp, in wg_gfx_cairo).
set(WRONGGPU_APP_SOURCES_WINDOWS src/app/wizard.cpp src/app/ui_gfx.cpp)
# Linux only: the live-frame hook for the SDL window layer.
set(WRONGGPU_APP_SOURCES_LINUX src/app/ui_live_hook_linux.cpp)

# wg_core: the installer's engine, split into modules (src/wg_core; build/launcher-design.md section 2). No globals:
# a wg::Context is passed in. The wizard (and later the launcher) is built on top of it.
add_library(wg_core STATIC
  src/wg_core/util.cpp src/wg_core/pe.cpp src/wg_core/context.cpp src/wg_core/payload.cpp src/wg_core/record.cpp
  src/wg_core/coexist.cpp src/wg_core/detect.cpp src/wg_core/settings.cpp src/wg_core/optiscaler.cpp src/wg_core/optiedition.cpp
  src/wg_core/model.cpp src/wg_core/checks.cpp src/wg_core/install.cpp src/wg_core/logs.cpp
  src/wg_core/profile.cpp src/wg_core/logtail.cpp src/wg_core/launcher.cpp
  src/wg_core/fingerprint.cpp src/wg_core/optikeys.cpp src/wg_core/leftovers.cpp src/wg_core/rr.cpp src/wg_core/rrprep.cpp src/wg_core/gamerules.cpp
  src/wg_core/rrconfig.cpp src/wg_core/convert_ignore.cpp)
target_include_directories(wg_core PUBLIC src)
target_compile_definitions(wg_core PUBLIC ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION})
if(WRONGGPU_TESTING)
  target_compile_definitions(wg_core PUBLIC WRONGGPU_TESTING)
endif()
if(WIN32)
  target_link_libraries(wg_core PUBLIC wgc_core advapi32 user32 shell32 ole32 uuid winhttp)
else()
  # Linux: <windows.h> and the other Windows headers the core includes resolve to src/platform/posix, which implements
  # the calls the core makes on POSIX. A Windows call it lacks fails the Linux build at that line.
  target_include_directories(wg_core BEFORE PUBLIC src/platform/posix)
  target_sources(wg_core PRIVATE src/platform/posix/win32_posix.cpp src/platform/posix/win32_posix_process.cpp
                 src/platform/posix/win32_posix_registry.cpp)
  target_link_libraries(wg_core PUBLIC wgc_core)
endif()
if(MSVC)
  target_compile_options(wg_core PRIVATE /Zi)
  target_compile_definitions(wg_core PRIVATE NDEBUG)
endif()

if(WIN32)
# wg_ui: the widget kit (owner-drawn buttons, check boxes, progress bar, cards, palette, fonts) of the wizard and the
# launcher (src/wg_ui). Windows' own GDI, GDI+, DWM and comctl32 only.
add_library(wg_ui STATIC src/wg_ui/kit.cpp)
target_link_libraries(wg_ui PUBLIC wg_core comctl32 gdi32 gdiplus dwmapi uxtheme user32 ole32)
if(MSVC)
  target_compile_options(wg_ui PRIVATE /Zi)
  target_compile_definitions(wg_ui PRIVATE NDEBUG)
endif()

# WrongGPU-Restore.exe: the per-game restorer (uninstall.exe in each game's wronggpu-setup folder): wg_core without the
# payload, the converter data or the UI. WrongGPU.exe embeds it (resource 960) and writes it into every game it installs.
file(GENERATE OUTPUT ${GEN}/restore_app.rc CONTENT "1 ICON \"${GEN}/app.ico\"\n")
add_executable(WrongGPU-Restore WIN32 src/app/restore_main.cpp ${GEN}/restore_app.rc ${GEN}/app.ico ${MANIFEST_SOURCE})
target_compile_definitions(WrongGPU-Restore PRIVATE "WRONGGPU_VERSION_STR=L\"${WRONGGPU_VERSION}\"")
target_link_libraries(WrongGPU-Restore PRIVATE wg_core shell32 user32 advapi32)
set_source_files_properties(${GEN}/restore_app.rc PROPERTIES OBJECT_DEPENDS ${GEN}/app.ico)
if(MSVC)
  target_compile_options(WrongGPU-Restore PRIVATE /Zi)
  target_link_options(WrongGPU-Restore PRIVATE /INCREMENTAL:NO /MANIFESTUAC:NO /DEBUG:FULL)
else()
  target_link_options(WrongGPU-Restore PRIVATE -municode -static)
endif()
add_custom_command(OUTPUT ${GEN}/restorer.bin
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:WrongGPU-Restore> ${GEN}/restorer.bin
  DEPENDS WrongGPU-Restore COMMENT "Embedding the restorer")
file(GENERATE OUTPUT ${GEN}/restore.rc CONTENT "960 RCDATA \"${GEN}/restorer.bin\"\n")
set_source_files_properties(${GEN}/restore.rc PROPERTIES OBJECT_DEPENDS ${GEN}/restorer.bin)

# WrongGPU.exe: the launcher (window, command line, elevated worker) with the installer's /switches as aliases.
add_executable(WrongGPU WIN32 ${WRONGGPU_APP_SOURCES} ${WRONGGPU_APP_SOURCES_WINDOWS}
               ${GEN}/resources.rc ${GEN}/restore.rc ${GEN}/restorer.bin ${GEN}/payload_table.h ${MANIFEST_SOURCE})
add_dependencies(WrongGPU WrongGPU-Restore)
endif()
if(WRONGGPU_TESTING)
  add_executable(wizard-policy-tests tests/wizard_policy_tests.cpp)
  target_compile_definitions(wizard-policy-tests PRIVATE ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION})
endif()
# wg_core's unit tests (strings and paths, install record, wronggpu.ini keys, the log file; no game, payload or GPU).
add_executable(wg-core-tests tests/wg_core_tests.cpp)
target_link_libraries(wg-core-tests PRIVATE wg_core)
# The launcher's unit tests (settings rendering and validation, ini round trips, record v1 -> v2, the log-tail parser,
# job files, the update reply, the game environment; no game, payload or GPU).
add_executable(launcher-core-tests tests/launcher_core_tests.cpp)
target_link_libraries(launcher-core-tests PRIVATE wg_core)
# The coexistence tests (M2): fingerprints on synthetic bytes, the proxy picker, byte-exact OptiScaler.ini edits and the
# inikey journal, the OptiScaler tab's key rules, the record's inikey and coexist lines, the log warnings.
add_executable(coexist-core-tests tests/coexist_tests.cpp)
target_link_libraries(coexist-core-tests PRIVATE wg_core)
# real front-end log excerpts (trimmed, paths replaced) for the Last run parser (UI review r3 #2)
target_compile_definitions(coexist-core-tests PRIVATE WG_TEST_LOGS="${CMAKE_CURRENT_SOURCE_DIR}/tests/logs")
# The converter search's ignore lists (src/wg_core/convert_ignore.h): the list format, bad lines, path and hash matching
# (case, slashes, '.' and '..', hard link, junction, symbolic link where the account may make one), the well-known lists,
# and an empty list keeping every candidate set as it was (synthetic files in the temp folder).
add_executable(convert-ignore-tests tests/convert_ignore_tests.cpp)
target_link_libraries(convert-ignore-tests PRIVATE wg_core)
enable_testing()
add_test(NAME source-size COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_source_size.py)
# The launcher's words never claim the same output as the original ("bit for bit", "identical", "(native)": rr-notes.md)
add_test(NAME wording COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_wording.py)
add_test(NAME wg-core COMMAND wg-core-tests)
add_test(NAME launcher-core COMMAND launcher-core-tests)
add_test(NAME coexist-core COMMAND coexist-core-tests)
add_test(NAME convert-ignore COMMAND convert-ignore-tests)
if(NOT WIN32)
  # the Linux layer's own checks: sharing, processes and the registry file go further than the core's tests do
  add_executable(posix-layer-tests src/platform/posix/tests/posix_layer_tests.cpp)
  target_link_libraries(posix-layer-tests PRIVATE wg_core)
  add_test(NAME posix-layer COMMAND posix-layer-tests)
  # keeps Linux from breaking silently: no wstring-built std::filesystem::path, no Direct2D/WIC/DWM/UIA in the shared UI
  # (tools/ci/check_portable.py); the second test proves each rule fails on a planted violation
  set(PORTABLE_CHECK ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/ci/check_portable.py)
  if(EXISTS ${PORTABLE_CHECK})
    add_test(NAME portable COMMAND ${Python3_EXECUTABLE} -I ${PORTABLE_CHECK} --root ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME portable-selftest COMMAND ${Python3_EXECUTABLE} -I ${PORTABLE_CHECK} --self-test)
  endif()
  # The launcher's drawing on Linux (Gfx on Cairo and Pango, fonts through fontconfig) and its smoke test (offscreen PNGs)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(WG_CAIRO REQUIRED IMPORTED_TARGET cairo pangocairo fontconfig)
  # The window layer: the Win32 window calls the launcher's window makes, on SDL3 (static; -DCMAKE_PREFIX_PATH=<its install> or
  # -DSDL3_DIR=<its lib/cmake/SDL3>). Only the drawing's live frame and the text fields reach into it; ui-dump and ui-render
  # (a message-only window) never start SDL.
  find_package(SDL3 CONFIG REQUIRED)
  if(TARGET SDL3::SDL3-static)
    set(WG_SDL3_TARGET SDL3::SDL3-static)
  else()
    set(WG_SDL3_TARGET SDL3::SDL3)
  endif()
  add_library(wg_window STATIC src/platform/posix/win32_window_sdl.cpp src/platform/posix/win32_window_edit.cpp
              src/platform/posix/win32_window_misc.cpp)
  target_link_libraries(wg_window PUBLIC wg_core ${WG_SDL3_TARGET})
  add_library(wg_gfx_cairo STATIC src/app/ui_gfx_cairo.cpp)
  target_link_libraries(wg_gfx_cairo PUBLIC wg_core wg_window PkgConfig::WG_CAIRO)
  target_compile_definitions(wg_gfx_cairo PRIVATE WRONGGPU_FONT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/fonts")
  add_executable(gfx-linux-smoke tests/gfx_linux_smoke.cpp)
  target_link_libraries(gfx-linux-smoke PRIVATE wg_gfx_cairo)
  add_test(NAME gfx-linux-smoke COMMAND gfx-linux-smoke ${CMAKE_CURRENT_BINARY_DIR}/gfx-smoke)
  # The launcher itself on Linux: the same window code as Windows (src/app, the Win32 calls answered by src/platform/posix), without
  # the old GDI+ wizard (wizard.cpp, wg_ui), the DWM title bar, tooltips and UI Automation, which Windows alone has.
  add_executable(WrongGPU ${WRONGGPU_APP_SOURCES} ${WRONGGPU_APP_SOURCES_LINUX} ${GEN}/resources.S ${GEN}/payload_table.h)
  target_include_directories(WrongGPU PRIVATE ${GEN})
  # the assembler reads the payload through .incbin, which CMake's dependency scan does not see
  set_source_files_properties(${GEN}/resources.S PROPERTIES OBJECT_DEPENDS "${GEN}/payload_table.h;${WRONGGPU_PAYLOAD_FILES};${WRONGGPU_RR_FILES}")
  target_compile_definitions(WrongGPU PRIVATE ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION} UNICODE _UNICODE
                             WRONGGPU_VERSION_TEXT="${WRONGGPU_VERSION}" WRONGGPU_BUILD_ID_TEXT="${WRONGGPU_BUILD_ID}"
                             WRONGGPU_FONT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/fonts")
  target_link_libraries(WrongGPU PRIVATE wg_gfx_cairo wg_window wg_core)
  # an installed launcher finds its fonts at ../share/wronggpu/assets/fonts from its binary (ui_gfx_cairo.cpp FontDir)
  if(WRONGGPU_TESTING)
    # the window's pages (ui-dump, ui-render) and the live window under SDL's offscreen driver with scripted events (a click, a
    # scroll, typing, WM_CLOSE): each on a home, XDG folders and registry file of its own
    add_test(NAME linux-ui COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_linux_ui_tests.py
             --exe $<TARGET_FILE:WrongGPU> --work ${CMAKE_CURRENT_BINARY_DIR}/linux-ui-tests)
    set_tests_properties(linux-ui PROPERTIES TIMEOUT 900)
    if(MOCK_PAYLOAD_WORK)
      # the launcher's verbs suite (tests/run_launcher_tests.py) against this binary: it needs the test map and stand-in payload
      # the build embeds only when it made them itself; its checks that only mean something on Windows say so and are skipped
      add_test(NAME launcher-native COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_launcher_tests.py
               --native-linux $<TARGET_FILE:WrongGPU> --work ${CMAKE_CURRENT_BINARY_DIR}/launcher-native-tests)
      set_tests_properties(launcher-native PROPERTIES TIMEOUT 1800)
    endif()
    # the window layer's message handling with SDL events pushed by the test (display scale change, a batch of keys, text in an edit,
    # WM_QUIT order, posts from a thread, capture, PeekMessageW's filters); SDL's offscreen driver, no window opens
    add_executable(window-linux-tests tests/window_linux_tests.cpp)
    target_link_libraries(window-linux-tests PRIVATE wg_window)
    add_test(NAME window-linux COMMAND window-linux-tests)
    set_tests_properties(window-linux PROPERTIES TIMEOUT 120 ENVIRONMENT "SDL_VIDEO_DRIVER=offscreen;SDL_AUDIODRIVER=dummy")
  endif()
  install(TARGETS WrongGPU RUNTIME DESTINATION bin)
  install(DIRECTORY assets/fonts DESTINATION share/wronggpu/assets)
  # the AppImage (build-appimage.sh) takes everything it ships from these install rules: the menu entry and the app's own icon
  # (the biggest frame of assets/wronggpu.ico)
  file(MAKE_DIRECTORY ${GEN})
  execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/ico_to_png.py ${CMAKE_CURRENT_SOURCE_DIR}/assets/wronggpu.ico
                  ${GEN}/wronggpu.png RESULT_VARIABLE ICON_PNG_RESULT OUTPUT_QUIET)
  if(NOT ICON_PNG_RESULT EQUAL 0)
    message(FATAL_ERROR "tools/ico_to_png.py failed")
  endif()
  install(FILES assets/wronggpu.desktop DESTINATION share/applications)
  install(FILES ${GEN}/wronggpu.png DESTINATION share/icons/hicolor/256x256/apps)
endif()
if(WRONGGPU_TESTING)
  add_test(NAME wizard-policy COMMAND wizard-policy-tests)
endif()
if(WIN32)
target_include_directories(WrongGPU PRIVATE ${GEN})
target_compile_definitions(WrongGPU PRIVATE ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION})
target_link_libraries(WrongGPU PRIVATE wg_ui wg_core wgc_core comctl32 shell32 ole32 oleaut32 uuid advapi32 user32 gdi32 gdiplus dwmapi uxtheme d2d1 d3d11 dxgi dwrite windowscodecs uiautomationcore)
if(MSVC)
  target_compile_options(WrongGPU PRIVATE /Zi)
  target_compile_definitions(WrongGPU PRIVATE NDEBUG)
  # The manifest source already requests asInvoker: no second trustInfo block from the linker.
  target_link_options(WrongGPU PRIVATE /INCREMENTAL:NO /MANIFESTUAC:NO /DEBUG:FULL)
else()
  target_link_options(WrongGPU PRIVATE -municode -static)
endif()
endif()
