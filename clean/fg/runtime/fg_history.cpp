// SPDX-License-Identifier: GPL-3.0-only
#include "fg_history.h"

namespace wgfg {

namespace {

bool same_texture(const FgTextureRef& a, const FgTextureRef& b) {
    return a.resource == b.resource && a.format == b.format && a.width == b.width && a.height == b.height &&
           a.subresource == b.subresource && a.state == b.state && a.producer_epoch == b.producer_epoch &&
           a.frame_token == b.frame_token && a.viewport_id == b.viewport_id && a.rect.x == b.rect.x &&
           a.rect.y == b.rect.y && a.rect.width == b.rect.width && a.rect.height == b.rect.height;
}

bool covers(const FgTextureRef& t, const FgRect& r) {
    return r.width > 0 && r.height > 0 && static_cast<uint64_t>(r.x) + r.width <= t.width &&
           static_cast<uint64_t>(r.y) + r.height <= t.height;
}

}  // namespace

// ---- FgFrameAssembler ------------------------------------------------------------------------------------------------

FgRefusal FgFrameAssembler::Begin(uint64_t token) {
    if ((have_last_ && token <= last_token_) || (open_ && token <= token_)) return FgRefusal::StaleTag;
    Abort();
    open_ = true;
    token_ = token;
    return FgRefusal::None;
}

FgRefusal FgFrameAssembler::Tag(FgTagSlot slot, const FgTextureRef& t) {
    const uint32_t s = static_cast<uint32_t>(slot);
    if (s >= kFgTagSlots) return FgRefusal::UnknownTag;
    if (!open_) return FgRefusal::NoFrameOpen;
    if (t.frame_token < token_ || (have_last_ && t.frame_token <= last_token_)) return FgRefusal::StaleTag;
    if (t.frame_token != token_ || t.viewport_id != session_.viewport_id) return FgRefusal::TagTokenMismatch;
    if (set_[s]) return same_texture(tags_[s], t) ? FgRefusal::None : FgRefusal::ConflictingTag;
    set_[s] = true;
    tags_[s] = t;
    return FgRefusal::None;
}

FgRefusal FgFrameAssembler::Finish(const FgFramePacket& c, FgFramePacket* out) {
    if (!open_) return FgRefusal::NoFrameOpen;
    FgFramePacket p = c;
    p.frame_token = token_;
    p.viewport_id = session_.viewport_id;
    const bool* set = set_;
    FgRefusal r = FgRefusal::None;
    if (!set[0] || !tags_[0].resource) r = FgRefusal::MissingColor;
    else if (!set[1] || !tags_[1].resource) r = FgRefusal::MissingMotion;
    else if (!set[2] || !tags_[2].resource) r = FgRefusal::MissingDepth;
    if (r == FgRefusal::None) {
        p.color = tags_[0];
        p.motion = tags_[1];
        p.depth = tags_[2];
        p.optional_present = 0;
        for (uint32_t i = 0; i < kFgOptCount; ++i) {
            const bool have = set[3 + i] && tags_[3 + i].resource;
            const FgOptionalDecl d = session_.optional[i];
            if (have && d == FgOptionalDecl::Never) { r = FgRefusal::UndeclaredOptional; break; }
            if (!have && d == FgOptionalDecl::Always) { r = FgRefusal::MissingPromisedOptional; break; }
            if (set[3 + i] && !tags_[3 + i].resource) { r = FgRefusal::MissingPromisedOptional; break; }
            p.optional[i] = have ? tags_[3 + i] : FgTextureRef{};
            if (have) p.optional_present |= FgOptBit(static_cast<FgOptionalInput>(i));
        }
    }
    last_token_ = token_;
    have_last_ = true;
    Abort();
    if (r == FgRefusal::None && out) *out = p;
    return r;
}

void FgFrameAssembler::Abort() {
    open_ = false;
    for (uint32_t i = 0; i < kFgTagSlots; ++i) {
        set_[i] = false;
        tags_[i] = FgTextureRef{};
    }
}

// ---- FgHistory -------------------------------------------------------------------------------------------------------

FgHistory::FgHistory(const FgCreateDesc& session, uint32_t banks)
    : session_(session),
      nbanks_(banks < kMinBanks ? kMinBanks : (banks > kMaxBanks ? kMaxBanks : banks)),
      queue_epoch_(session.queue_epoch),
      swapchain_epoch_(session.swapchain_epoch) {}

FgRefusal FgHistory::CheckIdentity(const FgFramePacket& p) const {
    if (p.viewport_id != session_.viewport_id) return FgRefusal::ViewportMismatch;
    if (p.adapter_luid != session_.adapter_luid) return FgRefusal::LuidMismatch;
    if (p.queue_epoch != queue_epoch_) return FgRefusal::QueueEpochMismatch;
    if (p.swapchain_epoch != swapchain_epoch_) return FgRefusal::SwapchainEpochMismatch;
    return FgRefusal::None;
}

FgRefusal FgHistory::CheckInputs(const FgFramePacket& p) const {
    if (!p.color.resource) return FgRefusal::MissingColor;
    if (!p.motion.resource) return FgRefusal::MissingMotion;
    if (!p.depth.resource) return FgRefusal::MissingDepth;
    const FgTextureRef* req[3] = {&p.color, &p.motion, &p.depth};
    for (const FgTextureRef* t : req)
        if (t->frame_token != p.frame_token || t->viewport_id != p.viewport_id) return FgRefusal::TagTokenMismatch;
    for (uint32_t i = 0; i < kFgOptCount; ++i) {
        const bool bit = (p.optional_present & FgOptBit(static_cast<FgOptionalInput>(i))) != 0;
        const FgOptionalDecl d = session_.optional[i];
        if (bit && d == FgOptionalDecl::Never) return FgRefusal::UndeclaredOptional;
        if (!bit && d == FgOptionalDecl::Always) return FgRefusal::MissingPromisedOptional;
        if (bit && !p.optional[i].resource) return FgRefusal::MissingPromisedOptional;
        if (bit && (p.optional[i].frame_token != p.frame_token || p.optional[i].viewport_id != p.viewport_id))
            return FgRefusal::TagTokenMismatch;
    }
    if (p.optional_present >> kFgOptCount) return FgRefusal::UndeclaredOptional;
    auto canonical_color_format = [](uint32_t f) -> uint32_t {
        switch (f) {
            case 27: case 28: case 29: return 28;  // R8G8B8A8 TYPELESS / UNORM / UNORM_SRGB
            case 87: case 90: case 91: return 87;  // B8G8R8A8 UNORM / TYPELESS / UNORM_SRGB
            case 23: case 24: return 24;           // R10G10B10A2 TYPELESS / UNORM
            case 9: case 10: return 10;            // R16G16B16A16 TYPELESS / FLOAT
            default: return f;
        }
    };
    if (p.color.format == 0 || canonical_color_format(p.color.format) != canonical_color_format(session_.output_format))
        return FgRefusal::BadFormat;
    const FgRect& o = p.output_rect;
    if (o.x != 0 || o.y != 0 || o.width != session_.output_width || o.height != session_.output_height ||
        !covers(p.color, o))
        return FgRefusal::OutputRectMismatch;
    const FgRect& r = p.render_rect;
    if (r.width < session_.render_min_width || r.width > session_.render_max_width ||
        r.height < session_.render_min_height || r.height > session_.render_max_height)
        return FgRefusal::RenderRectOutsideBounds;
    const FgRect& mrect = (p.flags & kFgFlagMotionRenderRes) ? r : o;
    if (!covers(p.motion, mrect) || !covers(p.depth, r)) return FgRefusal::TextureExtent;
    return FgRefusal::None;
}

void FgHistory::Drop(int b) {
    if (b < 0) return;
    FgHistoryBank& k = banks_[b];
    k.role = k.read_fence ? FgBankRole::Retired : FgBankRole::Free;
}

void FgHistory::Invalidate() {
    Drop(previous_);
    Drop(current_);
    previous_ = current_ = -1;
    interval_n_ = 0;
    ++epoch_;
}

void FgHistory::Rebind(uint64_t queue_epoch, uint64_t swapchain_epoch) {
    queue_epoch_ = queue_epoch;
    swapchain_epoch_ = swapchain_epoch;
    have_token_ = false;  // a new chain may restart its frame tokens
    Invalidate();
}

int FgHistory::FreeBank(uint64_t completed, int keep) const {
    for (uint32_t i = 0; i < nbanks_; ++i) {
        const FgHistoryBank& k = banks_[i];
        if (static_cast<int>(i) == keep) continue;
        const bool reading = k.read_fence && k.read_fence > completed;
        if (reading) continue;
        // a rotation releases the previous frame, so its bank is a candidate once compute no longer reads it
        const bool leaving = keep >= 0 && static_cast<int>(i) == previous_;
        if (k.role == FgBankRole::Free || k.role == FgBankRole::Retired || leaving) return static_cast<int>(i);
    }
    return -1;
}

FgIngestResult FgHistory::Accept(const FgFramePacket& p, uint64_t completed) {
    FgIngestResult res;
    res.epoch = epoch_;
    res.preset = p.preset;
    FgRefusal r = FgValidatePacketShape(session_, p);
    if (r == FgRefusal::None && p.generated_index != 0) r = FgRefusal::BadGeneratedIndex;
    if (r == FgRefusal::None) r = CheckIdentity(p);
    if (r == FgRefusal::None && have_token_ && p.frame_token <= last_token_) r = FgRefusal::TokenNotMonotonic;
    if (r == FgRefusal::None) r = CheckInputs(p);
    if (r != FgRefusal::None) {  // nothing changes; the skipped token re-primes the next frame
        res.status = FgFrameStatus::Refused;
        res.refusal = r;
        return res;
    }
    Retire(completed);
    // what breaks continuity, in order
    FgFrameRealOnly why = FgFrameRealOnly::None;
    const FgHistoryBank* cur = current_ >= 0 ? &banks_[current_] : nullptr;
    if (p.flags & kFgFlagMenu) why = FgFrameRealOnly::Menu;
    else if (!cur) why = FgFrameRealOnly::Prime;
    else if (p.flags & kFgFlagReset) why = FgFrameRealOnly::Reset;
    else if (p.frame_token != last_token_ + 1) why = FgFrameRealOnly::Gap;
    else if (cur->color_format != p.color.format || cur->output_width != p.output_rect.width ||
             cur->output_height != p.output_rect.height || cur->render_rect.width != p.render_rect.width ||
             cur->render_rect.height != p.render_rect.height)
        why = FgFrameRealOnly::Resize;
    last_token_ = p.frame_token;
    have_token_ = true;
    ++base_presents_;
    if (why != FgFrameRealOnly::None && why != FgFrameRealOnly::Prime) Invalidate();
    res.epoch = epoch_;
    if (why == FgFrameRealOnly::Menu) {  // shown real-only, not stored: the next valid frame primes
        res.status = FgFrameStatus::RealOnly;
        res.real_only = why;
        return res;
    }
    const bool rotate = why == FgFrameRealOnly::None;
    const int nb = FreeBank(completed, rotate ? current_ : -1);
    if (nb < 0) {  // every candidate bank is still read by compute: never overwrite it
        if (primed()) Invalidate();
        res.epoch = epoch_;
        res.status = FgFrameStatus::RealOnly;
        res.real_only = FgFrameRealOnly::HistoryBusy;
        return res;
    }
    if (rotate) {  // the previous frame leaves history (its bank may be the one reused)
        Drop(previous_);
        previous_ = -1;
    }
    FgHistoryBank& k = banks_[nb];
    k.role = FgBankRole::Current;
    k.frame_token = p.frame_token;
    k.epoch = epoch_;
    k.color_format = p.color.format;
    k.output_width = p.output_rect.width;
    k.output_height = p.output_rect.height;
    k.render_rect = p.render_rect;
    ++k.writes;
    if (rotate) {
        banks_[current_].role = FgBankRole::Previous;
        previous_ = current_;
        current_ = nb;
        ++rotations_;
        interval_n_ = p.generated_count;
        interval_preset_ = p.preset;
        res.status = FgFrameStatus::Generated;
        res.previous_bank = previous_;
        res.current_bank = current_;
        res.generated_count = interval_n_;
        return res;
    }
    previous_ = -1;
    current_ = nb;
    ++primes_;
    interval_n_ = 0;
    res.status = FgFrameStatus::RealOnly;
    res.real_only = why;
    res.current_bank = current_;
    return res;
}

FgIngestResult FgHistory::Evaluate(const FgFramePacket& p) const {
    FgIngestResult res;
    res.epoch = epoch_;
    res.preset = p.preset;
    res.status = FgFrameStatus::Refused;
    FgRefusal r = FgValidatePacketShape(session_, p);
    if (r == FgRefusal::None && p.generated_index == 0) r = FgRefusal::BadGeneratedIndex;
    if (r == FgRefusal::None) r = CheckIdentity(p);
    if (r == FgRefusal::None && !interval_open()) r = FgRefusal::HistoryNotPrimed;
    if (r == FgRefusal::None && (p.frame_token != banks_[current_].frame_token || p.generated_count != interval_n_ ||
                                 p.preset != interval_preset_))
        r = FgRefusal::IntervalMismatch;
    if (r == FgRefusal::None && (p.generated_index < 1 || p.generated_index > interval_n_))
        r = FgRefusal::IndexOutsideInterval;
    if (r != FgRefusal::None) {
        res.refusal = r;
        return res;
    }
    res.status = FgFrameStatus::Generated;
    res.previous_bank = previous_;
    res.current_bank = current_;
    res.generated_count = interval_n_;
    res.generated_index = p.generated_index;
    res.t = FgIndexTime(p.generated_index, interval_n_);
    return res;
}

bool FgHistory::MarkRead(uint64_t fence) {
    if (!interval_open() || fence == 0) return false;
    for (int b : {previous_, current_})
        if (banks_[b].read_fence < fence) banks_[b].read_fence = fence;
    return true;
}

FgHistorySnapshot FgHistory::Snapshot() const {
    FgHistorySnapshot s;
    s.nbanks = nbanks_;
    for (uint32_t i = 0; i < kMaxBanks; ++i) s.banks[i] = banks_[i];
    s.previous = previous_;
    s.current = current_;
    s.epoch = epoch_;
    return s;
}

bool FgCheckHistoryInvariants(const FgHistorySnapshot& s, std::string* why) {
    auto no = [why](const char* w) {
        if (why) *why = w;
        return false;
    };
    if (s.nbanks < FgHistory::kMinBanks || s.nbanks > FgHistory::kMaxBanks) return no("bank count");
    const int n = static_cast<int>(s.nbanks);
    if (s.current < -1 || s.current >= n || s.previous < -1 || s.previous >= n) return no("bank index");
    if (s.previous >= 0 && s.current < 0) return no("previous without current");
    if (s.previous >= 0 && s.previous == s.current) return no("previous and current share a bank");
    int currents = 0, previouses = 0;
    for (int i = 0; i < n; ++i) {
        const FgHistoryBank& b = s.banks[i];
        if (b.role == FgBankRole::Current) {
            ++currents;
            if (i != s.current) return no("a Current bank that is not current");
        }
        if (b.role == FgBankRole::Previous) {
            ++previouses;
            if (i != s.previous) return no("a Previous bank that is not previous");
        }
        if (b.role == FgBankRole::Retired && !b.read_fence) return no("a retired bank nobody reads");
    }
    if (currents != (s.current >= 0 ? 1 : 0) || previouses != (s.previous >= 0 ? 1 : 0)) return no("role counts");
    if (s.current >= 0 && s.banks[s.current].epoch != s.epoch) return no("current bank of another epoch");
    if (s.previous >= 0) {
        const FgHistoryBank &p = s.banks[s.previous], &c = s.banks[s.current];
        if (p.epoch != s.epoch) return no("previous bank of another epoch");
        if (p.frame_token >= c.frame_token) return no("previous not older than current");
    }
    return true;
}

void FgHistory::Retire(uint64_t completed) {
    for (uint32_t i = 0; i < nbanks_; ++i) {
        FgHistoryBank& k = banks_[i];
        if (k.read_fence && k.read_fence <= completed) k.read_fence = 0;
        if (k.role == FgBankRole::Retired && !k.read_fence) k.role = FgBankRole::Free;
    }
}

}  // namespace wgfg
