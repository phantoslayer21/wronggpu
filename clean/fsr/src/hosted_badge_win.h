// SPDX-License-Identifier: MIT
#pragma once
// Original Windows-composed provider badge. No game swap-chain, command-list,
// queue, fence, query heap, GPU timestamp or neural-kernel operations.
#include "config.h"
#include "status.h"
#include <string>
namespace wgwin {
std::wstring HostedBadgeCaption(const wgfsr::UpscaleStatus& status);
void ConfigureHostedBadge(const wgfsr::Config& config);
void ReportHostedBadge();
}
