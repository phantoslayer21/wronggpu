# SPDX-License-Identifier: GPL-3.0-only
# WrongGPU installer (spec 30). Windows only: MSVC or MinGW-w64.
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
project(wronggpu_installer LANGUAGES CXX RC)
# MinGW's headers set _WIN32_WINNT to Windows 7 as soon as a standard header comes first (context.h includes
# <functional> before base.h), which hides the Windows 8+ declarations the core uses (FILE_ID_INFO in convert_ignore.cpp):
# the whole build gets base.h's Windows 10 instead. MSVC's headers leave it unset.
if(MINGW)
  add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # no runtime DLL needed on the player's PC

set(WRONGGPU_PAYLOAD_DIR "" CACHE PATH "Folder with WrongGPU's amd_fidelityfx_dx12.dll, the runtime and its kernels (one release build)")
set(WRONGGPU_MAP "${CMAKE_CURRENT_SOURCE_DIR}/../../data/preset-M.v1.map.json" CACHE FILEPATH "Converter map (spec 08)")
set(WRONGGPU_VERSION "0.6.5" CACHE STRING "Installer version")
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
# Frame generation (#96, src/wg_core/fg.h): strictly off by default. ON (a build whose payload runs WrongGPU's frame generation):
# the per-game settings fg, fg_preset, fg_multiplier and fg_presenter (`set <game> fg=on`, off by default) and preset FG's
# converter map embedded for "model convert --preset=FG" (the model's folder: <model>\FG). OFF: none of it; the keys are refused
# by name and nothing about frame generation is written into any game.
option(WRONGGPU_FG "Frame generation's per-game settings and converter map (preset FG) in the launcher; OFF = none" OFF)
set(WRONGGPU_MAP_FG "${CMAKE_CURRENT_SOURCE_DIR}/../../data/fg.v1.map.json" CACHE FILEPATH "Frame generation converter map (preset FG; read only with WRONGGPU_FG)")
if(WRONGGPU_FG AND WRONGGPU_MAP_FG AND EXISTS "${WRONGGPU_MAP_FG}")
  set(MAP_FG_ARGS --map-fg ${WRONGGPU_MAP_FG})
  set(MAP_FG_FILE ${WRONGGPU_MAP_FG})
else()
  set(MAP_FG_ARGS)
  set(MAP_FG_FILE)
endif()
if(WRONGGPU_FG)
  set(FG_DEFINITION WRONGGPU_FG=1)
else()
  set(FG_DEFINITION WRONGGPU_FG=0)
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
# The Model L choice in the launcher (src/wizard_policy.h): ON only for a release whose front end and runtime run Model L.
option(WRONGGPU_MODEL_L "The Model L choice in the launcher (model=L, the OptiScaler entry); OFF = no model setting" OFF)
if(WRONGGPU_MODEL_L)
  set(MODEL_L_DEFINITION WRONGGPU_MODEL_L=1)
else()
  set(MODEL_L_DEFINITION WRONGGPU_MODEL_L=0)
endif()
# The Model K choice in the launcher (src/wizard_policy.h): a development build for a front end with WG_MODEL_K and a runtime
# with Model K, never for a release that players get (K is off by default). `model` becomes a setting with the value k
# (m|k, m|k|l with WRONGGPU_MODEL_L) and `set <game> model=K` writes wronggpu.ini's model=K and OptiScaler's UpscalerIndex.
# It needs Preset K's converter map and model folder (WRONGGPU_PRESET_K).
option(WRONGGPU_MODEL_K "The Model K choice in the launcher (model=K, the OptiScaler entry; needs WRONGGPU_PRESET_K); OFF = no Model K choice" OFF)
if(WRONGGPU_MODEL_K)
  if(NOT WRONGGPU_PRESET_K)
    message(FATAL_ERROR "WRONGGPU_MODEL_K needs WRONGGPU_PRESET_K=ON (Model K's converter map and model folder).")
  endif()
  set(MODEL_K_DEFINITION WRONGGPU_MODEL_K=1)
else()
  set(MODEL_K_DEFINITION WRONGGPU_MODEL_K=0)
endif()
option(WRONGGPU_DIRECT_ROUTE "The direct FSR route in the installer (the wizard's advanced choice, /DIRECTFSR, a silent install without /OPTISCALER=); OFF = OptiScaler is required" ON)
if(WRONGGPU_DIRECT_ROUTE)
  set(DIRECT_ROUTE_PAYLOAD_ARGS)
  set(DIRECT_ROUTE_DEFINITION WRONGGPU_DIRECT_ROUTE=1)
else()
  set(DIRECT_ROUTE_PAYLOAD_ARGS --hosted-only)
  set(DIRECT_ROUTE_DEFINITION WRONGGPU_DIRECT_ROUTE=0)
endif()
if(NOT WRONGGPU_PAYLOAD_DIR)
  message(FATAL_ERROR "Set WRONGGPU_PAYLOAD_DIR to the release payload folder (see README.md).")
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
# Frame generation's files (WRONGGPU_FG only, src/wg_core/fg.h): the frame generation plugin and its kernels of one release
# build (a folder with plugins\WrongGPU-FG.asi and fg-dxil\<name>.dxil, the release's stage folder layout; the kernels may also
# be under wronggpu-setup\ or plugins\wronggpu-setup\). An install puts the plugin beside WrongGPU's plugin and the kernels in
# wronggpu-setup\fg-dxil beside it, where the plugin reads them. Empty: an installer without them (FG games are installed
# without frame generation's files). Never OptiScaler (tools/make_payload.py refuses it). A WRONGGPU_TESTING build without a
# folder gets generated stand-ins (tests/make_mock_fg.py), so its tests never need a real FG file.
set(WRONGGPU_FG_DIR "" CACHE PATH "Folder with the frame generation plugin (plugins/WrongGPU-FG.asi) and its kernels (fg-dxil/*.dxil); read only with WRONGGPU_FG")
set(FG_PAYLOAD_DIR)
if(WRONGGPU_FG)
  set(FG_PAYLOAD_DIR "${WRONGGPU_FG_DIR}")
  if(NOT FG_PAYLOAD_DIR AND WRONGGPU_TESTING)
    set(FG_PAYLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/mock-fg)
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_mock_fg.py ${FG_PAYLOAD_DIR} RESULT_VARIABLE MOCK_FG_RESULT)
    if(NOT MOCK_FG_RESULT EQUAL 0)
      message(FATAL_ERROR "tests/make_mock_fg.py failed")
    endif()
  endif()
endif()
if(FG_PAYLOAD_DIR)
  set(FG_PAYLOAD_ARGS --fg ${FG_PAYLOAD_DIR})
  file(GLOB_RECURSE WRONGGPU_FG_FILES CONFIGURE_DEPENDS "${FG_PAYLOAD_DIR}/*")
else()
  set(FG_PAYLOAD_ARGS)
  set(WRONGGPU_FG_FILES)
endif()
if(EXISTS "${MAP_F_FILE}")
  set(MAP_F_ARGS --map-f ${MAP_F_FILE})
else()
  set(MAP_F_ARGS)
  set(MAP_F_FILE)
endif()

# wg_core: the installer's engine, split into modules (src/wg_core; build/launcher-design.md section 2). No globals:
# a wg::Context is passed in. The wizard (and later the launcher) is built on top of it.
add_library(wg_core STATIC
  src/wg_core/util.cpp src/wg_core/pe.cpp src/wg_core/context.cpp src/wg_core/payload.cpp src/wg_core/record.cpp
  src/wg_core/coexist.cpp src/wg_core/detect.cpp src/wg_core/settings.cpp src/wg_core/optiscaler.cpp src/wg_core/optiedition.cpp
  src/wg_core/model.cpp src/wg_core/checks.cpp src/wg_core/install.cpp src/wg_core/logs.cpp
  src/wg_core/profile.cpp src/wg_core/logtail.cpp src/wg_core/launcher.cpp
  src/wg_core/fingerprint.cpp src/wg_core/optikeys.cpp src/wg_core/leftovers.cpp src/wg_core/rr.cpp src/wg_core/rrprep.cpp src/wg_core/gamerules.cpp
  src/wg_core/rrconfig.cpp src/wg_core/convert_ignore.cpp src/wg_core/fg.cpp)
target_include_directories(wg_core PUBLIC src)
target_compile_definitions(wg_core PUBLIC ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION})
# (private: model-choice-tests compiles profile.cpp with the Model L choice on, against this library)
target_compile_definitions(wg_core PRIVATE ${MODEL_L_DEFINITION} ${MODEL_K_DEFINITION} ${FG_DEFINITION})
if(WRONGGPU_TESTING)
  target_compile_definitions(wg_core PUBLIC WRONGGPU_TESTING)
  if(NOT MSVC)
    # record.cpp's #pragma comment(lib, "rstrtmgr.lib") is MSVC-only; clang and gcc for MinGW need the library named
    target_link_libraries(wg_core PUBLIC rstrtmgr)
  endif()
endif()
target_link_libraries(wg_core PUBLIC wgc_core advapi32 user32 shell32 ole32 uuid winhttp)
if(MSVC)
  target_compile_options(wg_core PRIVATE /Zi)
  target_compile_definitions(wg_core PRIVATE NDEBUG)
endif()

# wg_ui: the widget kit (owner-drawn buttons, check boxes, progress bar, cards, palette, fonts) of the wizard and the
# launcher (src/wg_ui). Windows' own GDI, GDI+, DWM and comctl32 only.
add_library(wg_ui STATIC src/wg_ui/kit.cpp)
target_link_libraries(wg_ui PUBLIC wg_core comctl32 gdi32 gdiplus dwmapi uxtheme user32 ole32)
if(MSVC)
  target_compile_options(wg_ui PRIVATE /Zi)
  target_compile_definitions(wg_ui PRIVATE NDEBUG)
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
# A WRONGGPU_TESTING build may embed the test suite's stand-in runtime (tests/run_installer_tests.py --prepare-only: the
# marker and a build manifest that says stand_in); any other build refuses it at this step (make_payload.py --testing).
if(WRONGGPU_TESTING)
  set(TESTING_PAYLOAD_ARGS --testing)
else()
  set(TESTING_PAYLOAD_ARGS)
endif()
add_custom_command(
  OUTPUT ${GEN}/payload_table.h ${GEN}/resources.rc ${GEN}/app.ico
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_payload.py
          --payload ${WRONGGPU_PAYLOAD_DIR} --map ${WRONGGPU_MAP} ${PRESET_K_MAP_ARGS} --version ${WRONGGPU_VERSION}
          --build-id=${WRONGGPU_BUILD_ID} --payload-name=${WRONGGPU_PAYLOAD_NAME}
          ${DIRECT_ROUTE_PAYLOAD_ARGS} ${RR_PAYLOAD_ARGS} ${FG_PAYLOAD_ARGS} ${MAP_F_ARGS} ${MAP_L_ARGS} ${MAP_FG_ARGS} ${MANIFEST_ARGS} ${TESTING_PAYLOAD_ARGS} --out ${GEN}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_payload.py ${MANIFEST} ${WRONGGPU_MAP} ${PRESET_K_MAP_FILE} ${MAP_F_FILE} ${MAP_L_FILE} ${MAP_FG_FILE} ${WRONGGPU_PAYLOAD_FILES} ${WRONGGPU_RR_FILES} ${WRONGGPU_FG_FILES}
  COMMENT "Embedding the payload and the converter map")

# WrongGPU-Restore.exe: the per-game restorer (uninstall.exe in each game's wronggpu-setup folder): wg_core without the
# payload, the converter data or the UI. WrongGPU.exe embeds it (resource 960) and writes it into every game it installs.
file(GENERATE OUTPUT ${GEN}/restore_app.rc CONTENT "1 ICON \"${GEN}/app.ico\"\n
1 VERSIONINFO
FILEVERSION 0,0,0,0
PRODUCTVERSION 0,0,0,0
BEGIN
  BLOCK \"StringFileInfo\"
  BEGIN
    BLOCK \"040904B0\"
    BEGIN
      VALUE \"OriginalFilename\", \"WrongGPU-Restore.exe\\0\"
      VALUE \"ProductName\", \"WrongGPU\\0\"
    END
  END
  BLOCK \"VarFileInfo\"
  BEGIN
    VALUE \"Translation\", 0x0409, 1200
  END
END
")
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
# A WRONGGPU_TESTING build's identity, which the launcher suite checks before it runs a program (tests/launcher_build_facts.py):
# a named resource only such a build has, and a PE checksum (MSVC /RELEASE) that a program changed after its link fails.
if(WRONGGPU_TESTING)
  file(GENERATE OUTPUT ${GEN}/testing_identity.rc
       CONTENT "WRONGGPU_TESTING_BUILD RCDATA\nBEGIN\n  \"wg-testing-build 1: store libraries, access log, edition download, uninstall root\\0\"\nEND\n")
  target_sources(WrongGPU-Restore PRIVATE ${GEN}/testing_identity.rc)
  if(MSVC)
    target_link_options(WrongGPU-Restore PRIVATE /RELEASE)
  endif()
endif()
add_custom_command(OUTPUT ${GEN}/restorer.bin
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:WrongGPU-Restore> ${GEN}/restorer.bin
  DEPENDS WrongGPU-Restore COMMENT "Embedding the restorer")
file(GENERATE OUTPUT ${GEN}/restore.rc CONTENT "960 RCDATA \"${GEN}/restorer.bin\"\n")
set_source_files_properties(${GEN}/restore.rc PROPERTIES OBJECT_DEPENDS ${GEN}/restorer.bin)

# WrongGPU.exe: the launcher (window, command line, elevated worker) with the installer's /switches as aliases.
add_executable(WrongGPU WIN32 src/app/main.cpp src/app/wizard.cpp src/app/cli.cpp src/app/launcher_ui.cpp src/app/ui_gfx.cpp src/app/ui_art.cpp
               src/app/ui_widgets.cpp src/app/ui_library.cpp src/app/ui_game.cpp src/app/ui_settings.cpp src/app/ui_uia.cpp src/app/ui_render.cpp src/app/ui_words.cpp src/app/ui_dump.cpp src/app/ui_frame.cpp src/app/ui_play.cpp src/app/ui_rrprep.cpp src/app/ui_rrconfig.cpp src/app/ui_activate.cpp src/app/ui_checks.cpp src/app/payload_embed.cpp
               ${GEN}/resources.rc ${GEN}/restore.rc ${GEN}/restorer.bin ${GEN}/payload_table.h ${MANIFEST_SOURCE})
add_dependencies(WrongGPU WrongGPU-Restore)
if(WRONGGPU_TESTING)
  target_sources(WrongGPU PRIVATE ${GEN}/testing_identity.rc)
  if(MSVC)
    target_link_options(WrongGPU PRIVATE /RELEASE)
  endif()
  # the launcher suite's negative controls: programs that are not testing builds (no identity), one with the hook strings;
  # each leaves <program>.started next to itself when it starts, so the suite can prove it never started them
  add_executable(wg-test-canary tests/canary.cpp)
  add_executable(wg-test-canary-strings tests/canary.cpp)
  target_compile_definitions(wg-test-canary-strings PRIVATE WG_CANARY_HOOK_STRINGS)
  if(NOT MSVC)
    target_link_options(wg-test-canary PRIVATE -municode)  # (wmain, as the launcher and restorer)
    target_link_options(wg-test-canary-strings PRIVATE -municode)
  endif()
  add_library(wg-test-canary-dll SHARED tests/canary_dll.cpp)  # a load marker: writes <dll>.loaded when loaded
  set_target_properties(wg-test-canary-dll PROPERTIES PREFIX "")
  # built with the testing launcher (a single-target build has them), and the build list the suite checks before it starts
  # one: the SHA-256 of this WrongGPU.exe and of the canaries, written after the launcher's link
  add_dependencies(WrongGPU wg-test-canary wg-test-canary-strings wg-test-canary-dll)
  add_custom_command(TARGET WrongGPU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DF1=$<TARGET_FILE:WrongGPU> -DF2=$<TARGET_FILE:wg-test-canary> -DF3=$<TARGET_FILE:wg-test-canary-strings>
            -DF4=$<TARGET_FILE:wg-test-canary-dll>
            -DOUT=$<TARGET_FILE_DIR:WrongGPU>/wg-test-build.sha256 -P ${CMAKE_CURRENT_SOURCE_DIR}/tools/sha256_list.cmake
    VERBATIM)
  add_executable(wizard-policy-tests tests/wizard_policy_tests.cpp)
  target_compile_definitions(wizard-policy-tests PRIVATE ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION} ${MODEL_L_DEFINITION} ${MODEL_K_DEFINITION})
endif()
# wg_core's unit tests (strings and paths, install record, wronggpu.ini keys, the log file; no game, payload or GPU).
add_executable(wg-core-tests tests/wg_core_tests.cpp)
target_link_libraries(wg-core-tests PRIVATE wg_core)
# The launcher's unit tests (settings rendering and validation, ini round trips, record v1 -> v2, the log-tail parser,
# job files, the update reply, the game environment; no game, payload or GPU).
add_executable(launcher-core-tests tests/launcher_core_tests.cpp)
target_link_libraries(launcher-core-tests PRIVATE wg_core)
target_compile_definitions(launcher-core-tests PRIVATE ${MODEL_L_DEFINITION} ${MODEL_K_DEFINITION} ${FG_DEFINITION})
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
# The Model L choice (profile.cpp with WRONGGPU_MODEL_L on, whatever this build's switch): the model setting, wronggpu.ini's
# model item and OptiScaler's UpscalerIndex. Its own copy of profile.cpp comes first; the rest is wg_core.
add_executable(model-choice-tests tests/model_choice_tests.cpp src/wg_core/profile.cpp)
target_compile_definitions(model-choice-tests PRIVATE WRONGGPU_MODEL_L=1)
target_link_libraries(model-choice-tests PRIVATE wg_core)
# The Model K choice (profile.cpp with WRONGGPU_MODEL_K on, whatever this build's switch, so that CI, which keeps Model K off,
# runs it): with the Model L choice too (the setting m|k|l: K, and an L and an M control) and without (m|k). Each has its own
# copy of profile.cpp.
add_executable(model-choice-k-tests tests/model_choice_k_tests.cpp src/wg_core/profile.cpp)
target_compile_definitions(model-choice-k-tests PRIVATE WRONGGPU_MODEL_K=1 WRONGGPU_MODEL_L=1)
target_link_libraries(model-choice-k-tests PRIVATE wg_core)
add_executable(model-choice-k-only-tests tests/model_choice_k_tests.cpp src/wg_core/profile.cpp)
target_compile_definitions(model-choice-k-only-tests PRIVATE WRONGGPU_MODEL_K=1 WRONGGPU_MODEL_L=0)
target_link_libraries(model-choice-k-only-tests PRIVATE wg_core)
add_test(NAME launcher-core COMMAND launcher-core-tests)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/launcher-core-localappdata")
set_tests_properties(launcher-core PROPERTIES ENVIRONMENT "LOCALAPPDATA=${CMAKE_CURRENT_BINARY_DIR}/launcher-core-localappdata")
add_test(NAME model-choice COMMAND model-choice-tests)
add_test(NAME model-choice-k COMMAND model-choice-k-tests)
add_test(NAME model-choice-k-only COMMAND model-choice-k-only-tests)
# The frame generation choice (profile.cpp with WRONGGPU_FG on, whatever this build's switch, so that CI, which keeps it off,
# runs it): the four settings in their exact spellings, profile.ini and settings.ini round trips beside the other keys, the
# opt-in rule (nothing written for a game never opted in), fg kept off in the game while its data is not ready.
add_executable(fg-choice-tests tests/fg_choice_tests.cpp src/wg_core/profile.cpp)
target_compile_definitions(fg-choice-tests PRIVATE WRONGGPU_FG=1)
target_link_libraries(fg-choice-tests PRIVATE wg_core)
add_test(NAME fg-choice COMMAND fg-choice-tests)
# Frame generation's files (wg_core/fg.h, this build's switch): WrongGPU-FG.asi never another setup's plugin (ForeignPlugins),
# where an install puts the plugin and its kernels, the record's entry, the library check, OptiScaler's own frame generation, and
# (WRONGGPU_FG) the state folder's keys folded back into the profile.
add_executable(fg-files-tests tests/fg_files_tests.cpp)
target_link_libraries(fg-files-tests PRIVATE wg_core)
target_compile_definitions(fg-files-tests PRIVATE ${FG_DEFINITION})
add_test(NAME fg-files COMMAND fg-files-tests)
# Frame generation's files in the payload step (tools/make_payload.py --fg, any build): the stage folder's layout embedded as
# plugins\WrongGPU-FG.asi and fg-dxil\<name>.dxil; OptiScaler, NVIDIA names, a developer runtime and any other file refused.
add_test(NAME fg-payload COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fg_payload_tests.py --work ${CMAKE_CURRENT_BINARY_DIR}/fg-payload-work)
# A testing build with frame generation: install, repair and uninstall of its files in generated fixture games with the test
# launcher (tests/run_fg_install_tests.py; never a real game, never a GPU program).
if(WRONGGPU_TESTING AND WRONGGPU_FG AND FG_PAYLOAD_DIR)
  add_test(NAME fg-install COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fg_install_tests.py --launcher $<TARGET_FILE:WrongGPU>
           --work ${CMAKE_CURRENT_BINARY_DIR}/fg-install-work --fg-dir ${FG_PAYLOAD_DIR})
endif()
add_test(NAME coexist-core COMMAND coexist-core-tests)
add_test(NAME convert-ignore COMMAND convert-ignore-tests)
if(WRONGGPU_TESTING)
  add_test(NAME wizard-policy COMMAND wizard-policy-tests)
endif()
target_include_directories(WrongGPU PRIVATE ${GEN})
target_compile_definitions(WrongGPU PRIVATE ${PRESET_K_DEFINITION} ${DIRECT_ROUTE_DEFINITION} ${MODEL_L_DEFINITION} ${MODEL_K_DEFINITION} ${FG_DEFINITION})
target_link_libraries(WrongGPU PRIVATE wg_ui wg_core wgc_core comctl32 shell32 ole32 oleaut32 uuid advapi32 user32 gdi32 gdiplus dwmapi uxtheme d2d1 d3d11 dxgi dwrite windowscodecs uiautomationcore)
if(MSVC)
  target_compile_options(WrongGPU PRIVATE /Zi)
  target_compile_definitions(WrongGPU PRIVATE NDEBUG)
  # The manifest source already requests asInvoker: no second trustInfo block from the linker.
  target_link_options(WrongGPU PRIVATE /INCREMENTAL:NO /MANIFESTUAC:NO /DEBUG:FULL)
else()
  target_link_options(WrongGPU PRIVATE -municode -static)
endif()
