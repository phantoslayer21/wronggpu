// SPDX-License-Identifier: MIT
// Original provider truth regression, no GPU/device/third-party host implementation.
#include "overlay.h"
#include "hosted_badge_win.h"
#include <cstdio>
namespace wronggpu { extern const char* const kLogTag = "hosted badge unit"; }
int main() {
    wgfsr::Config config; config.hosted=true; config.overlay="off";
    wgwin::ConfigureOverlay(config);
    wgfsr::UpscaleStatus sent, read;
    sent.state=wgfsr::UpscaleState::Network; sent.upscaler="WrongGPU network";
    wgwin::OverlayStatus(sent);
    if (!wgwin::ReadOverlayProviderStatus(read) || wgwin::HostedBadgeCaption(read)!=L"WrongGPU Model M") return 1;
    sent.state=wgfsr::UpscaleState::Amd; sent.upscaler="4.1.1";
    wgwin::OverlayStatus(sent);
    if (!wgwin::ReadOverlayProviderStatus(read) || read.state!=wgfsr::UpscaleState::Amd ||
        wgwin::HostedBadgeCaption(read)!=L"AMD FSR 4.1.1") return 2;
    sent.state=wgfsr::UpscaleState::StandIn; wgwin::OverlayStatus(sent);
    if (!wgwin::ReadOverlayProviderStatus(read) || wgwin::HostedBadgeCaption(read)!=L"WrongGPU (stand-in)") return 3;
    sent.state=wgfsr::UpscaleState::Network; sent.upscaler="WrongGPU stand-in upscaler";
    wgwin::OverlayStatus(sent);
    if (!wgwin::ReadOverlayProviderStatus(read) || wgwin::HostedBadgeCaption(read)==L"WrongGPU Model M") return 4;
    sent.state=wgfsr::UpscaleState::Failed; sent.reason="No fallback provider";
    wgwin::OverlayStatus(sent);
    if (!wgwin::ReadOverlayProviderStatus(read) || read.state!=wgfsr::UpscaleState::Failed ||
        read.reason!=sent.reason || wgwin::HostedBadgeCaption(read)!=L"Upscaling unavailable") return 5;
    std::puts("PASS: latest provider wins; numeric AMD versions named; stand-in never labeled Model M; no GPU objects created");
    return 0;
}
