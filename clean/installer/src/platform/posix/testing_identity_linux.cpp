// SPDX-License-Identifier: MIT
// Linux, WRONGGPU_TESTING builds of the launcher only: the identity the launcher suite checks before it runs a program
// (tests/launcher_build_facts.py), as the WRONGGPU_TESTING_BUILD resource on Windows (CMakeLists.txt): the same text, then
// "wg-sha256:" and 32 bytes that tools/elf_testing_identity.py fills after the link with the file's SHA-256 (those bytes
// counted as zero), as the PE checksum on Windows.
extern "C" __attribute__((section(".wg_testing_id"), used)) const char wg_testing_identity[] =
    "wg-testing-build 1: store libraries, access log, edition download, uninstall root\0"
    "wg-sha256:"
    "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
