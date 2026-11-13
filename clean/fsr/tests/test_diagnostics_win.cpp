// SPDX-License-Identifier: MIT
// Exercises the production asynchronous logger, not a replacement JSON writer.
#include "../src/log_win.h"
#include <fstream>
#include <limits>
#include <thread>
#include <vector>
namespace wronggpu { extern const char* const kLogTag = "test_diagnostics"; }
int main() {
    const std::wstring hipPath = wronggpu::ModuleDirectory() + L"\\amdhip64_7.dll";
    HMODULE hip = LoadLibraryExW(hipPath.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!hip) return 10;
    auto& log = wronggpu::Log::Get();
    log.Flush(); // Capture the already loaded metadata fixture before injecting a failure.
    log.Flush(); // First Flush queued background metadata, second persists it.
    for (uint64_t i = 0; i < 80; ++i) {
        WgFrame f{}; f.frame_index = i; f.render_width = 1707; f.render_height = 960;
        f.output_width = 2560; f.output_height = 1440; f.jitter[0] = 0.25f;
        f.color = {reinterpret_cast<void*>(1), 26, 1707, 960, 64};
        f.pre_exposure = 1; f.motion_scale[0] = -1707; f.command_list_type = 2;
        log.RecordFrame(42, f);
    }
    log.WriteLine("ERROR: hipStreamWaitValue32 failed: HIP error 719 (unspecified launch failure)");
    log.Flush();
    log.Flush();
    std::ifstream input(log.DiagnosticPath());
    const std::string content((std::istreambuf_iterator<char>(input)), {});
    if (content.find("WG-HIP-719") == std::string::npos || content.find("input_wait") == std::string::npos ||
        content.find("root_cause_proven\":false") == std::string::npos || content.find("session\"") == std::string::npos ||
        content.find("recent_frames\":64") == std::string::npos || content.find("render\":[1707,960]") == std::string::npos ||
        content.find("frame_metadata\":{\"context\":42,\"frame\":16,") == std::string::npos ||
        content.find("frame_metadata\":{\"context\":42,\"frame\":79,") == std::string::npos ||
        content.find("hip_runtime_version") == std::string::npos || content.find("70260201") == std::string::npos ||
        content.find("sha256\"") == std::string::npos || content.find("cpu_detection_stack") == std::string::npos) {
        std::puts("FAIL: missing correlated structured HIP failure with explicit attribution limit"); return 1;
    }
    // Bug hunt F1: the same failure on every later frame writes no further full incident within the window (only counted)
    auto count = [](const std::string& text, const std::string& what) {
        size_t n = 0;
        for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size())) ++n;
        return n;
    };
    const size_t failuresBefore = count(content, "\"event\":\"failure\""), framesBefore = count(content, "\"event\":\"failure_frame\"");
    for (int i = 0; i < 40; ++i) log.WriteLine("ERROR: hipStreamWaitValue32 failed: HIP error 719 (unspecified launch failure)");
    log.Flush();
    log.Flush();
    {
        std::ifstream again(log.DiagnosticPath());
        const std::string later((std::istreambuf_iterator<char>(again)), {});
        if (failuresBefore < 1 || count(later, "\"event\":\"failure\"") != failuresBefore ||  // (the file is appended to across runs) count(later, "\"event\":\"failure_frame\"") != framesBefore ||
            later.find("repeats_since_last\":0") == std::string::npos) {
            std::puts("FAIL: a repeated failure wrote another full incident within the window"); return 3;
        }
    }
    {
        wgdiag::IncidentGate gate;
        uint64_t repeats = 99;
        bool ok = gate.Allow("WG-A", 1000, &repeats) && repeats == 0;
        for (int i = 0; i < 5; ++i) ok &= !gate.Allow("WG-A", 1000 + 100 * i, &repeats);
        ok &= gate.Allow("WG-B", 1200, &repeats) && repeats == 0;          // another code: its own incident
        ok &= !gate.Allow("WG-A", 1000 + wgdiag::IncidentGate::kWindowMs - 1, &repeats);
        ok &= gate.Allow("WG-A", 1000 + wgdiag::IncidentGate::kWindowMs + 1, &repeats) && repeats == 6;  // the window passed
        if (!ok) { std::puts("FAIL: IncidentGate"); return 4; }
    }
    const auto queries = reinterpret_cast<int (*)()>(GetProcAddress(hip, "diagnosticHipQueries"));
    const auto initializations = reinterpret_cast<int (*)()>(GetProcAddress(hip, "diagnosticHipInitializations"));
    if (!queries || !initializations || queries() < 1 || initializations() != 0) return 11;
    if (wgdiag::Quote("a\n\"\\\t") != "\"a\\u000a\\\"\\\\\\u0009\"" ||
        wgdiag::Number(std::numeric_limits<float>::quiet_NaN()) != "null") return 2;
    // Concurrent readers/writers use the same bounded production recorder.
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) writers.emplace_back([&] { for (uint64_t f = 8; f < 108; ++f) {
        WgFrame frame{}; frame.frame_index = f; log.RecordFrame(7, frame); } });
    for (auto& thread : writers) thread.join();
    log.Flush();
    std::puts("PASS: structured production diagnostic failure");
    return 0;
}
