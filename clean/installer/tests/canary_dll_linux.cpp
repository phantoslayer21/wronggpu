// SPDX-License-Identifier: MIT
// Linux: the launcher suite's load marker (tests/canary_dll.cpp on Windows): a shared library that writes <its own path>.loaded
// when it is loaded, so the suite can prove a planted stand-in was never loaded. Written for this project.
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <string>

static void Marker() {}

__attribute__((constructor)) static void Loaded() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&Marker), &info) || !info.dli_fname) return;
    const int fd = open((std::string(info.dli_fname) + ".loaded").c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fd >= 0) close(fd);
}
