// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the log collector: one .zip (stored, no compression) with WrongGPU's logs, the Windows events
// that matter and basic system information (support-manifest.json). Written for this project.
#pragma once
#include "checks.h"

namespace wg {

struct Bundle { wstring zip; std::vector<wstring> files; std::vector<wstring> notes; };

uint32_t Crc32(const void* data, size_t n);
string RecentWindowsEvents(const wstring& exeName);
string SupportQuote(const string& text);
bool SupportReadTail(const wstring& path, string& out, uint64_t limit, uint64_t& originalSize, DWORD& error);
string SupportDriverRegistry();
string SupportAdapters();
Bundle CollectLogs(Context& ctx, const wstring& dirIn, const wstring& outZip, const SystemInfo& sys);
wstring DefaultBundlePath(const wstring& dir);
wstring BundleMessage(const Bundle& b);
}  // namespace wg
