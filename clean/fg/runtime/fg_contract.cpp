// SPDX-License-Identifier: GPL-3.0-only
#include "fg_contract.h"

#include "fg_grid.h"

namespace wgfg {

bool FgPresetGraphOf(FgPreset preset, FgPresetGraph* out) {
    switch (preset) {
        case FgPreset::A: *out = {true, false, false}; return true;
        case FgPreset::B: *out = {true, true, false}; return true;
        case FgPreset::C: *out = {false, false, true}; return true;
    }
    return false;
}

const char* FgRefusalName(FgRefusal r) {
    switch (r) {
        case FgRefusal::None: return "none";
        case FgRefusal::StructSize: return "struct size";
        case FgRefusal::AbiVersion: return "ABI version";
        case FgRefusal::NullDevice: return "no device";
        case FgRefusal::NullQueue: return "no queue";
        case FgRefusal::BadLuid: return "no adapter LUID";
        case FgRefusal::BadOutputSize: return "output size outside 4..8192 x 4..7680";
        case FgRefusal::BadRenderBounds: return "render size bounds";
        case FgRefusal::BadPreset: return "preset";
        case FgRefusal::BadGeneratedCount: return "generated count outside 1..max";
        case FgRefusal::BadGeneratedIndex: return "generated index outside 0..N";
        case FgRefusal::BadOptionalDecl: return "optional input declaration";
        case FgRefusal::BadFormat: return "format";
        case FgRefusal::TokenNotMonotonic: return "frame token not increasing";
        case FgRefusal::ViewportMismatch: return "viewport mismatch";
        case FgRefusal::LuidMismatch: return "adapter LUID mismatch";
        case FgRefusal::QueueEpochMismatch: return "queue epoch mismatch";
        case FgRefusal::SwapchainEpochMismatch: return "swap-chain epoch mismatch";
        case FgRefusal::MissingColor: return "color missing";
        case FgRefusal::MissingMotion: return "motion vectors missing";
        case FgRefusal::MissingDepth: return "depth missing";
        case FgRefusal::MissingPromisedOptional: return "promised optional input missing";
        case FgRefusal::UndeclaredOptional: return "optional input declared never";
        case FgRefusal::ConflictingTag: return "conflicting tag";
        case FgRefusal::StaleTag: return "stale tag";
        case FgRefusal::TagTokenMismatch: return "tag of another frame or viewport";
        case FgRefusal::TextureExtent: return "texture extent";
        case FgRefusal::RenderRectOutsideBounds: return "render rectangle outside the session bounds";
        case FgRefusal::OutputRectMismatch: return "output rectangle";
        case FgRefusal::NoFrameOpen: return "no frame open";
        case FgRefusal::UnknownTag: return "unknown tag";
        case FgRefusal::HistoryBankBusy: return "history bank busy";
        case FgRefusal::HistoryNotPrimed: return "history not primed";
        case FgRefusal::IndexOutsideInterval: return "index outside the interval";
        case FgRefusal::IntervalMismatch: return "evaluation of another interval";
        case FgRefusal::ModelMissing: return "model missing";
        case FgRefusal::ModelContainer: return "model container";
        case FgRefusal::ModelChecksum: return "model checksum";
        case FgRefusal::ModelMapHash: return "model written with another map";
        case FgRefusal::ModelStageCount: return "model stage count";
        case FgRefusal::ModelStageName: return "model stage name or order";
        case FgRefusal::ModelTensorShape: return "model tensor shape";
        case FgRefusal::ModelTensorType: return "model tensor type";
        case FgRefusal::ModelStageDigest: return "model stage digest";
        case FgRefusal::ModelValue: return "model value not exact in float16";
        case FgRefusal::InstallOff: return "install switch off";
        case FgRefusal::UnsupportedAdapter: return "unsupported adapter";
        case FgRefusal::ModelNotValidated: return "model not validated";
        case FgRefusal::PresenterNotQualified: return "presenter not qualified";
        case FgRefusal::RouteNotQualified: return "FG modes not qualified";
        case FgRefusal::Net1GridUnsupported: return "net1 grid unsupported";
        case FgRefusal::Net1PipelineMissing: return "net1 pipeline missing";
        case FgRefusal::Net1ArenaMissing: return "net1 arena missing";
        case FgRefusal::Net1ArenaTooSmall: return "net1 arena too small";
        case FgRefusal::Net1PlanInvalid: return "net1 plan invalid";
        case FgRefusal::Net1RegionOverlap: return "net1 regions overlap";
        case FgRefusal::Net1RegionAlign: return "net1 region alignment";
        case FgRefusal::WorkerStartFailed: return "worker could not start";
        case FgRefusal::WorkerInitFailed: return "worker init failed";
        case FgRefusal::WorkerAbandoned: return "worker abandoned";
    }
    return "unknown";
}

FgRefusal FgValidateCreate(const FgCreateDesc& d) {
    if (d.struct_size != sizeof(FgCreateDesc)) return FgRefusal::StructSize;
    if (d.abi_version != kFgAbiVersion) return FgRefusal::AbiVersion;
    if (!d.device) return FgRefusal::NullDevice;
    if (!d.queue) return FgRefusal::NullQueue;
    if (d.adapter_luid.low == 0 && d.adapter_luid.high == 0) return FgRefusal::BadLuid;
    if (!FgOutputSizeSupported(d.output_width, d.output_height)) return FgRefusal::BadOutputSize;
    if (d.output_format == 0) return FgRefusal::BadFormat;
    if (d.render_min_width == 0 || d.render_min_height == 0 || d.render_min_width > d.render_max_width ||
        d.render_min_height > d.render_max_height || d.render_max_width > 16384 || d.render_max_height > 16384 ||
        (!d.dynamic_resolution &&
         (d.render_min_width != d.render_max_width || d.render_min_height != d.render_max_height)))
        return FgRefusal::BadRenderBounds;
    FgPresetGraph g{};
    if (!FgPresetGraphOf(d.preset, &g)) return FgRefusal::BadPreset;
    if (d.max_generated < 1 || d.max_generated > kFgMaxGenerated) return FgRefusal::BadGeneratedCount;
    for (uint32_t i = 0; i < kFgOptCount; ++i)
        if (static_cast<uint32_t>(d.optional[i]) > static_cast<uint32_t>(FgOptionalDecl::Always))
            return FgRefusal::BadOptionalDecl;
    return FgRefusal::None;
}

FgRefusal FgValidatePacketShape(const FgCreateDesc& s, const FgFramePacket& p) {
    if (p.struct_size != sizeof(FgFramePacket)) return FgRefusal::StructSize;
    if (p.abi_version != kFgAbiVersion) return FgRefusal::AbiVersion;
    FgPresetGraph g{};
    if (!FgPresetGraphOf(p.preset, &g)) return FgRefusal::BadPreset;
    if (p.generated_count < 1 || p.generated_count > s.max_generated) return FgRefusal::BadGeneratedCount;
    if (p.generated_index > p.generated_count) return FgRefusal::BadGeneratedIndex;
    return FgRefusal::None;
}

float FgIndexTime(uint32_t i, uint32_t n) {
    // One IEEE single-precision division of exactly representable operands (i, n <= 5): rounded once, to nearest even.
    const volatile float num = static_cast<float>(i);
    const volatile float den = static_cast<float>(n) + 1.0f;
    return num / den;
}

FgRefusal FgCheckRoute(const FgRouteFacts& f) {
    if (!f.install_on) return FgRefusal::InstallOff;
    if (!f.adapter_supported) return FgRefusal::UnsupportedAdapter;
    if (!f.model_validated) return FgRefusal::ModelNotValidated;
    if (!f.presenter_qualified) return FgRefusal::PresenterNotQualified;
    if (!f.modes_qualified) return FgRefusal::RouteNotQualified;
    return FgRefusal::None;
}

uint32_t FgAdvertisedMaxGenerated(const FgRouteFacts& f) {
    return FgCheckRoute(f) == FgRefusal::None ? kFgMaxGenerated : 0;
}

}  // namespace wgfg
