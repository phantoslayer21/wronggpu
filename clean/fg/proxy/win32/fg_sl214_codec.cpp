// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): the native codec of the 2.14 game interface's FG structures. See
// fg_sl214_codec.h. Written for this project.
#include "fg_sl214_codec.h"

#include <string.h>

#include <cstdio>

// Test-only mutants (CMakeLists.txt builds test_fg_cp2077_binding against each; every one must fail; fg_asi_transport.cpp
// has 6 and 7): 1 a generated count of 0 accepted  2 a count above the maximum accepted  3 state members written past
// the passed version  4 the chain walk ignores the GUID  5 the tag lifetime gate ignored  8 the frame clock never advances
#ifndef WG_FG_CP2077_MUTANT
#define WG_FG_CP2077_MUTANT 0
#endif

namespace wgfgp {
namespace win32 {

namespace {

int HexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// `n` hex digits at text[*pos] -> value (most significant first)
bool ReadHex(const char* text, size_t* pos, int n, uint64_t* out) {
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) {
        const int d = HexDigit(text[*pos]);
        if (d < 0) return false;
        v = (v << 4) | static_cast<uint64_t>(d);
        ++*pos;
    }
    *out = v;
    return true;
}

bool Dash(const char* text, size_t* pos) {
    if (text[*pos] != '-') return false;
    ++*pos;
    return true;
}

bool LifetimeAdmitted(uint32_t mask, uint32_t lifetime) { return lifetime < 32 && (mask & (1u << lifetime)) != 0; }

bool BelowMin(uint32_t v, uint32_t min) { return v != 0 && v < min; }

}  // namespace

bool FgGuidEq(const WgGuid& a, const WgGuid& b) { return memcmp(a.b, b.b, sizeof a.b) == 0; }

bool FgGuidZero(const WgGuid& g) {
    for (uint8_t x : g.b)
        if (x) return false;
    return true;
}

bool FgParseGuid(const char* text, WgGuid* out) {
    if (!text || !out) return false;
    size_t pos = 0;
    uint64_t d1 = 0, d2 = 0, d3 = 0, d4a = 0, d4b = 0;
    if (!ReadHex(text, &pos, 8, &d1) || !Dash(text, &pos) || !ReadHex(text, &pos, 4, &d2) || !Dash(text, &pos) ||
        !ReadHex(text, &pos, 4, &d3) || !Dash(text, &pos) || !ReadHex(text, &pos, 4, &d4a) || !Dash(text, &pos) ||
        !ReadHex(text, &pos, 12, &d4b) || text[pos] != '\0')
        return false;
    WgGuid g = {};
    for (int i = 0; i < 4; ++i) g.b[i] = static_cast<uint8_t>(d1 >> (8 * i));  // little-endian
    for (int i = 0; i < 2; ++i) g.b[4 + i] = static_cast<uint8_t>(d2 >> (8 * i));
    for (int i = 0; i < 2; ++i) g.b[6 + i] = static_cast<uint8_t>(d3 >> (8 * i));
    g.b[8] = static_cast<uint8_t>(d4a >> 8);  // as written
    g.b[9] = static_cast<uint8_t>(d4a);
    for (int i = 0; i < 6; ++i) g.b[10 + i] = static_cast<uint8_t>(d4b >> (8 * (5 - i)));
    *out = g;
    return true;
}

const WgHeader* FgFindInChain(const void* start, const WgGuid& type, uint32_t max_links) {
    const WgHeader* h = static_cast<const WgHeader*>(start);
    for (uint32_t i = 0; h && i <= max_links; ++i, h = h->next)
#if WG_FG_CP2077_MUTANT == 4
        (void)type;
        if (h) return h;
#else
        if (FgGuidEq(h->type, type)) return h;
#endif
    return nullptr;
}

bool FgSl214LayoutValid(const FgSl214Layout& l) {
    const WgGuid* g[6] = {&l.options_guid, &l.state_guid, &l.viewport_guid, &l.tag_guid, &l.resource_guid,
                          &l.constants_guid};
    for (int i = 0; i < 6; ++i) {
        if (FgGuidZero(*g[i])) return false;
        for (int k = 0; k < i; ++k)
            if (FgGuidEq(*g[i], *g[k])) return false;
    }
    return l.options_size == sizeof(WgNatFgOptions) && l.state_size == sizeof(WgNatFgState) && l.options_version >= 1 &&
           l.state_version >= 1 && l.min_size > 0;
}

FgSl214Codec::FgSl214Codec(const FgSl214Layout& layout) : l_(layout) {
    required_tags_[0] = l_.native_type[WG_FG_ROLE_DEPTH];
    required_tags_[1] = l_.native_type[WG_FG_ROLE_MOTION];
}

FgSl214Counters FgSl214Codec::counters() const {
    FgSl214Counters c;
    c.tags_seen = c_tags_seen_.load();
    c.tags_bound = c_tags_bound_.load();
    c.tags_wrong_guid = c_tags_wrong_guid_.load();
    c.tags_lifetime_refused = c_tags_lifetime_.load();
    c.tags_not_texture = c_tags_not_texture_.load();
    c.constants_decoded = c_constants_.load();
    c.constants_refused = c_constants_refused_.load();
    c.options_checked = c_options_checked_.load();
    c.options_refused = c_options_refused_.load();
    c.states_written = c_states_.load();
    return c;
}

const WgNatFgOptions* FgSl214Codec::find_options(const void* native) const {
    const WgHeader* h = FgFindInChain(native, l_.options_guid);
    if (!h || h->version < l_.options_version) return nullptr;
    return reinterpret_cast<const WgNatFgOptions*>(h);
}

const WgViewport* FgSl214Codec::find_viewport(const void* native, const WgViewport* explicit_vp) const {
    const WgHeader* h = FgFindInChain(native, l_.viewport_guid);
    if (h) return reinterpret_cast<const WgViewport*>(h);
    if (explicit_vp && FgGuidEq(explicit_vp->h.type, l_.viewport_guid)) return explicit_vp;
    return nullptr;
}

uint32_t FgSl214Codec::resolve_viewport_id(const WgViewport* vp, const void* native) const {
    const WgViewport* found = find_viewport(native, vp);
    return found ? found->id : (vp ? vp->id : 0u);
}

WgResultCode FgSl214Codec::check_options(const WgViewport* vp, const void* native, uint32_t max_generated) {
    c_options_checked_.fetch_add(1);
    const WgNatFgOptions* o = find_options(native);
    WgResultCode r = WG_OK;
    if (!o || !find_viewport(native, vp)) r = WG_E_MISSING_INPUT_PARAMETER;               // 20
    else if (o->frames == 0 && WG_FG_CP2077_MUTANT != 1) r = WG_E_INVALID_PARAMETER;    // 25, before the mode
    else if (o->frames > max_generated && WG_FG_CP2077_MUTANT != 2) r = WG_E_INVALID_STATE;  // 38
    else if (o->mode > kFgNatModeDynamic) r = WG_E_INVALID_PARAMETER;
    else if (o->mode == kFgNatModeDynamic && !l_.dynamic_supported) r = WG_E_INVALID_PARAMETER;
    if (r != WG_OK) c_options_refused_.fetch_add(1);
    return r;
}

bool FgSl214Codec::options(const WgViewport*, const void* native, WgFgOptions* out) {
    const WgNatFgOptions* o = find_options(native);
    if (!o || !out) return false;
    out->enable = o->mode != kFgNatModeOff ? 1u : 0u;
    // the UI-recomposition byte only from the version that carries it (an older structure ends before +112)
    const bool ui = o->h.version >= kFgNatOptionsUiRecompositionV && o->ui_recomposition == 1;
    out->preset = ui ? static_cast<uint32_t>(WG_FG_PRESET_B) : WG_FG_PRESET_NOT_SET;
    out->generated_count = o->frames;
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) out->optional_decl[i] = l_.optional_decl[i];
    out->color_space = WG_FG_CS_SDR;
    out->output_width = o->colour_width;
    out->output_height = o->colour_height;
    out->output_format = o->colour_format;
    out->dynamic_resolution = (o->flags & kFgNatFlagDynamicResolution) ? 1u : 0u;
    out->render_min_width = out->dynamic_resolution ? o->dynres_width : o->mvdepth_width;
    out->render_min_height = out->dynamic_resolution ? o->dynres_height : o->mvdepth_height;
    out->render_max_width = o->mvdepth_width;
    out->render_max_height = o->mvdepth_height;
    return true;
}

std::string FgSl214Codec::describe_options(const void* native) {
    const WgNatFgOptions* o = find_options(native);
    if (!o) return "options not in the chain";
    static const char* const kModes[] = {"off", "on", "auto", "dynamic"};
    char buf[160];
    if (o->h.version >= kFgNatOptionsUiRecompositionV)
        std::snprintf(buf, sizeof buf, "mode %u (%s), frames %u, flags 0x%x, ui recomposition %u", o->mode,
                      o->mode < 4 ? kModes[o->mode] : "unknown", o->frames, o->flags,
                      static_cast<unsigned>(o->ui_recomposition));
    else  // an older structure: nothing past what its version carries is read
        std::snprintf(buf, sizeof buf, "mode %u (%s), frames %u, flags 0x%x, options v%u", o->mode,
                      o->mode < 4 ? kModes[o->mode] : "unknown", o->frames, o->flags,
                      static_cast<unsigned>(o->h.version));
    return buf;
}

std::string FgSl214Codec::describe_state(const void* native) {
    const WgHeader* found = FgFindInChain(native, l_.state_guid);
    if (!found || found->version < 1) return "state not in the chain";
    const WgNatFgState* s = reinterpret_cast<const WgNatFgState*>(found);
    char buf[160];
    if (s->h.version >= kFgNatStateV2)
        std::snprintf(buf, sizeof buf, "frames max %u, status 0x%x, state v%u", s->frames_max, s->status,
                      static_cast<unsigned>(s->h.version));
    else
        std::snprintf(buf, sizeof buf, "status 0x%x, state v%u (no frames max)", s->status,
                      static_cast<unsigned>(s->h.version));
    return buf;
}

bool FgSl214Codec::write_state(const WgFgState& state, void* native) { return write_state_for(state, nullptr, native); }

bool FgSl214Codec::write_state_for(const WgFgState& st, const void* native_options, void* native) {
    const WgHeader* found = FgFindInChain(native, l_.state_guid);
    if (!found || found->version < 1) return false;
    WgNatFgState* s = reinterpret_cast<WgNatFgState*>(const_cast<WgHeader*>(found));
    const uint64_t ver = s->h.version;
    const WgNatFgOptions* o = native_options ? find_options(native_options) : nullptr;
    uint32_t status = 0;
    if (o) {
        if (BelowMin(o->colour_width, l_.min_size) || BelowMin(o->colour_height, l_.min_size))
            status |= kFgNatStatusResolutionLow;
        if ((o->flags & kFgNatFlagDynamicResolution) &&
            (BelowMin(o->dynres_width, l_.min_size) || BelowMin(o->dynres_height, l_.min_size)))
            status |= kFgNatStatusResolutionLow;
    }
    if (have_constants_.load() && unset_.load() != 0) status |= kFgNatStatusConstantUnset;
    // v1 members
    s->estimated_vram = 0;  // WrongGPU makes no VRAM estimate
    s->status = status;
    s->min_size = l_.min_size;
    s->frames_presented = st.enabled && st.generate_requests > 0 ? st.effective_count + 1u
                                                                 : (st.base_presents > 0 ? 1u : 0u);
#if WG_FG_CP2077_MUTANT == 3
    const bool all = true;
#else
    const bool all = false;
#endif
    if (ver >= kFgNatStateV2 || all) {
        s->frames_max = st.max_generated;
        s->vsync_available = o && o->mode == kFgNatModeOff ? 1u : 0u;
    }
    if (ver >= kFgNatStateV3 || all) {
        s->fence = nullptr;
        s->fence_value = 0;
    }
    if (ver >= kFgNatStateV4 || all) s->dynamic_supported = l_.dynamic_supported ? 1u : 0u;
    c_states_.fetch_add(1);
    return true;
}

bool FgSl214Codec::tags(const WgFrameToken*, const WgViewport* vp, const WgTag* native, uint32_t count, WgFgTag* out) {
    if (!out) return false;
    if (count && !native) return false;
    if (vp && !FgGuidEq(vp->h.type, l_.viewport_guid)) return false;
    const uint64_t frame = interval_.load() + 1;
    const uint32_t vp_id = vp ? vp->id : 0u;
    for (uint32_t i = 0; i < count; ++i) {
        WgFgTag& d = out[i];
        d.struct_size = sizeof(WgFgTag);
        d.native_type = WG_FG_NATIVE_TYPE_UNBOUND;
        c_tags_seen_.fetch_add(1);
        const WgHeader* th = FgFindInChain(&native[i], l_.tag_guid);
        if (!th) {
            c_tags_wrong_guid_.fetch_add(1);
            continue;
        }
        const WgTag& t = *reinterpret_cast<const WgTag*>(th);
        bool fg_input = false;
        for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r)
            if (l_.native_type[r] != WG_FG_NATIVE_TYPE_UNBOUND && l_.native_type[r] == t.buffer_type) fg_input = true;
        if (!fg_input) continue;  // another feature's input: forwarded, never observed
        const WgResource* res = t.resource;
        if (!res) continue;  // a removal: tags are observed, never owned
        if (!LifetimeAdmitted(l_.tag_lifetime_mask, t.lifetime) && WG_FG_CP2077_MUTANT != 5) {
            c_tags_lifetime_.fetch_add(1);
            continue;
        }
        const WgHeader* rh = FgFindInChain(res, l_.resource_guid);
        if (!rh) {
            c_tags_wrong_guid_.fetch_add(1);
            continue;
        }
        res = reinterpret_cast<const WgResource*>(rh);
        if (res->kind != WG_RES_TEX2D) {
            c_tags_not_texture_.fetch_add(1);
            continue;
        }
        d.native_type = t.buffer_type;
        WgFgTexture& x = d.texture;
        x.resource = res->native;
        x.format = res->native_format;
        x.width = res->width;
        x.height = res->height;
        x.subresource = 0;
        x.state = res->state;
        x.producer_epoch = 0;
        x.frame_token = frame;
        x.viewport_id = vp_id;
        if (t.extent.width && t.extent.height)
            x.rect = WgFgRect{t.extent.left, t.extent.top, t.extent.width, t.extent.height};
        else
            x.rect = WgFgRect{0, 0, res->width, res->height};
        c_tags_bound_.fetch_add(1);
    }
    return true;
}

bool FgSl214Codec::constants(const WgConstants* c, const WgFrameToken*, const WgViewport* vp, WgFgConstants* out) {
    if (!c || !out || !FgGuidEq(c->h.type, l_.constants_guid) || c->h.version < l_.constants_min_version ||
        (vp && !FgGuidEq(vp->h.type, l_.viewport_guid))) {
        c_constants_refused_.fetch_add(1);
        return false;
    }
    const uint64_t frame = interval_.load() + 1;
    out->frame_token = frame;
    out->viewport_id = vp ? vp->id : 0u;
    uint32_t flags = WG_FG_FLAG_MOTION_SCALE_UV;
    if (c->reset == WG_TRI_TRUE) flags |= WG_FG_FLAG_RESET;
    if (c->depth_inverted == WG_TRI_TRUE) flags |= WG_FG_FLAG_DEPTH_REVERSED;
    out->flags = flags;
    memcpy(out->camera_current, c->clip_to_prev_clip, sizeof(out->camera_current));
    memcpy(out->camera_previous, c->prev_clip_to_clip, sizeof(out->camera_previous));
    out->jitter_x = c->jitter[0];
    out->jitter_y = c->jitter[1];
    out->motion_scale_x = c->mv_scale[0];
    out->motion_scale_y = c->mv_scale[1];
    out->color_space = WG_FG_CS_SDR;
    uint32_t unset = 0;
    if (c->depth_inverted == WG_TRI_UNSET) unset |= 1u;
    if (c->camera_motion_included == WG_TRI_UNSET) unset |= 2u;
    if (c->reset == WG_TRI_UNSET) unset |= 4u;
    unset_.store(unset);
    have_constants_.store(true);
    c_constants_.fetch_add(1);
    return true;
}

bool FgSl214Codec::write_requirements(const WgFgCapability& cap, WgRequirements* native) {
    if (!native) return false;
    if (!FgGuidZero(l_.requirements_guid) && !FgGuidEq(native->h.type, l_.requirements_guid)) return false;
    native->flags = 1u << 1;  // D3D12
    native->max_viewports = 1;
    const bool both = required_tags_[0] != WG_FG_NATIVE_TYPE_UNBOUND && required_tags_[1] != WG_FG_NATIVE_TYPE_UNBOUND;
    native->required_tag_count = cap.supported && both ? 2u : 0u;
    native->required_tags = native->required_tag_count ? required_tags_ : nullptr;
    return true;
}

uint64_t FgSl214Codec::frame_for_present() { return interval_.load() + 1; }

void FgSl214Codec::note_present() {
#if WG_FG_CP2077_MUTANT != 8
    interval_.fetch_add(1);
#endif
}

}  // namespace win32
}  // namespace wgfgp
