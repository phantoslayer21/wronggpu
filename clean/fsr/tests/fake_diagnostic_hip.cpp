// SPDX-License-Identifier: MIT
// Original CPU-only HIP metadata fixture. Never copied into a game/release payload.
#include <atomic>
static std::atomic<int> queries{0}, initializations{0};
extern "C" __declspec(dllexport) int hipRuntimeGetVersion(int* value) { ++queries; if (!value) return 1; *value = 70260201; return 0; }
extern "C" __declspec(dllexport) int hipInit(unsigned) { ++initializations; return 1; }
extern "C" __declspec(dllexport) int diagnosticHipQueries() { return queries.load(); }
extern "C" __declspec(dllexport) int diagnosticHipInitializations() { return initializations.load(); }
