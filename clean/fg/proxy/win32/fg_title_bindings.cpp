// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): the qualified title bindings this build carries. See fg_title_bindings.h.
// Written for this project.
#include "fg_title_bindings.h"

#include <wchar.h>

// Generated at configure time from bindings/cp2077-2.14.1.json and bindings/sl2-generic.json (build tree;
// bindings/fg_binding_gen.cmake).
#include "fg_binding_cp2077.h"
#include "fg_binding_sl2_generic.h"

namespace wgfgp {
namespace win32 {

namespace {

bool Guid(const char* text, WgGuid* out) { return FgParseGuid(text, out); }

// One generated binding header's values, in one place (WG_FG_TB_DATA below fills it from a header's prefix).
struct BindingData {
    const char* id;
    const char* title;
    const char* exe_leaf;
    uint32_t fg_feature;
    const char* interface_leaf;
    uint32_t major, minor, minor_max;
    const char* roles[kFgRoleCount];
    const char* fn_options;
    const char* fn_state;
    uint32_t abi_max_generated;
    const char* factory_import;
    uint32_t dxgi_factory_fallback;
    uint32_t bt[WG_FG_ROLE_COUNT];  // backbuffer, motion, depth, hudless, ui alpha, distortion, no warp
    uint32_t ui_max_generated;
    uint32_t presets_mask;
    const char* guid_options;
    const char* guid_state;
    const char* guid_viewport;
    const char* guid_tag;
    const char* guid_resource;
    const char* guid_constants;
    const char* guid_requirements;
    uint32_t options_version, options_size, state_version, state_size, constants_min_version;
    uint32_t decl[WG_FG_OPTIONAL_COUNT];  // hudless, ui alpha, distortion, no warp
    uint32_t tag_lifetime_mask;
    uint32_t min_size;
    uint32_t dynamic_supported;
    uint32_t qualified, game_has_multiplier;
};

#define WG_FG_TB_DATA(P)                                                                                               \
    BindingData {                                                                                                      \
        WG_FG_TB_##P##_ID, WG_FG_TB_##P##_TITLE, WG_FG_TB_##P##_EXE_LEAF, WG_FG_TB_##P##_FG_FEATURE,                   \
            WG_FG_TB_##P##_INTERFACE_LEAF, WG_FG_TB_##P##_INTERFACE_MAJOR, WG_FG_TB_##P##_INTERFACE_MINOR,             \
            WG_FG_TB_##P##_INTERFACE_MINOR_MAX,                                                                        \
            {WG_FG_TB_##P##_ROLE_IS_SUPPORTED, WG_FG_TB_##P##_ROLE_REQUIREMENTS, WG_FG_TB_##P##_ROLE_GET_FUNCTION,     \
             WG_FG_TB_##P##_ROLE_TAG, WG_FG_TB_##P##_ROLE_TAG_FOR_FRAME, WG_FG_TB_##P##_ROLE_CONSTANTS,                \
             WG_FG_TB_##P##_ROLE_EVALUATE, WG_FG_TB_##P##_ROLE_FREE, WG_FG_TB_##P##_ROLE_IS_LOADED,                    \
             WG_FG_TB_##P##_ROLE_SET_LOADED},                                                                          \
            WG_FG_TB_##P##_FN_OPTIONS, WG_FG_TB_##P##_FN_STATE, WG_FG_TB_##P##_ABI_MAX_GENERATED,                      \
            WG_FG_TB_##P##_FACTORY_IMPORT, WG_FG_TB_##P##_DXGI_FACTORY_FALLBACK,                                       \
            {WG_FG_TB_##P##_BT_BACKBUFFER, WG_FG_TB_##P##_BT_MOTION, WG_FG_TB_##P##_BT_DEPTH,                          \
             WG_FG_TB_##P##_BT_HUDLESS, WG_FG_TB_##P##_BT_UI_ALPHA, WG_FG_TB_##P##_BT_DISTORTION,                      \
             WG_FG_TB_##P##_BT_NO_WARP},                                                                               \
            WG_FG_TB_##P##_UI_MAX_GENERATED, WG_FG_TB_##P##_PRESETS_MASK, WG_FG_TB_##P##_GUID_OPTIONS,                 \
            WG_FG_TB_##P##_GUID_STATE, WG_FG_TB_##P##_GUID_VIEWPORT, WG_FG_TB_##P##_GUID_TAG,                          \
            WG_FG_TB_##P##_GUID_RESOURCE, WG_FG_TB_##P##_GUID_CONSTANTS, WG_FG_TB_##P##_GUID_REQUIREMENTS,             \
            WG_FG_TB_##P##_OPTIONS_VERSION, WG_FG_TB_##P##_OPTIONS_SIZE, WG_FG_TB_##P##_STATE_VERSION,                 \
            WG_FG_TB_##P##_STATE_SIZE, WG_FG_TB_##P##_CONSTANTS_MIN_VERSION,                                           \
            {WG_FG_TB_##P##_DECL_HUDLESS, WG_FG_TB_##P##_DECL_UI_ALPHA, WG_FG_TB_##P##_DECL_DISTORTION,                \
             WG_FG_TB_##P##_DECL_NO_WARP},                                                                             \
            WG_FG_TB_##P##_TAG_LIFETIME_MASK, WG_FG_TB_##P##_MIN_SIZE, WG_FG_TB_##P##_DYNAMIC_SUPPORTED,               \
            WG_FG_TB_##P##_QUALIFIED, WG_FG_TB_##P##_GAME_HAS_MULTIPLIER                                     \
    }

FgTitleBinding Build(const BindingData& d) {
    FgTitleBinding b;
    b.id = d.id;
    b.title = d.title;
    b.exe_leaf = d.exe_leaf;
    b.game_has_multiplier = d.game_has_multiplier != 0;
    b.fg_feature = d.fg_feature;

    FgImportBinding& im = b.imports;
    im.interposer_leaf = d.interface_leaf;
    im.version_major = d.major;
    im.version_minor = d.minor;
    im.version_minor_max = d.minor_max > d.minor ? d.minor_max : 0u;  // 0: exactly the minor version
    for (uint32_t r = 0; r < kFgRoleCount; ++r) im.role_names[r] = d.roles[r];
    im.fn_options = d.fn_options;
    im.fn_state = d.fn_state;
    im.abi_max_generated = d.abi_max_generated;
    im.factory2_name = d.factory_import;
    im.dxgi_factory_fallback = d.dxgi_factory_fallback != 0;

    FgBinding& co = b.core;
    co.version_major = im.version_major;
    co.version_minor = im.version_minor;
    co.version_minor_max = im.version_minor_max;
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r) co.native_type[r] = d.bt[r];
    co.fn_names = {im.fn_options, im.fn_state};
    co.ui_max_generated = d.ui_max_generated;
    co.presets_mask = d.presets_mask;

    FgSl214Layout& l = b.layout;
    bool ok = Guid(d.guid_options, &l.options_guid) && Guid(d.guid_state, &l.state_guid) &&
              Guid(d.guid_viewport, &l.viewport_guid) && Guid(d.guid_tag, &l.tag_guid) &&
              Guid(d.guid_resource, &l.resource_guid) && Guid(d.guid_constants, &l.constants_guid) &&
              Guid(d.guid_requirements, &l.requirements_guid);
    l.options_version = d.options_version;
    l.options_size = d.options_size;
    l.state_version = d.state_version;
    l.state_size = d.state_size;
    l.constants_min_version = d.constants_min_version;
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r) l.native_type[r] = co.native_type[r];
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) l.optional_decl[i] = d.decl[i];
    l.tag_lifetime_mask = d.tag_lifetime_mask;
    l.min_size = d.min_size;
    l.dynamic_supported = d.dynamic_supported != 0;

    ok = ok && FgSl214LayoutValid(l) && im.abi_max_generated >= 1 && im.abi_max_generated <= WG_FG_MAX_GENERATED &&
         co.ui_max_generated >= 1 && co.ui_max_generated <= WG_FG_MAX_GENERATED &&
         co.native_type[WG_FG_ROLE_MOTION] != WG_FG_NATIVE_TYPE_UNBOUND &&
         co.native_type[WG_FG_ROLE_DEPTH] != WG_FG_NATIVE_TYPE_UNBOUND && !im.interposer_leaf.empty() &&
         !im.fn_options.empty() && !im.fn_state.empty() && im.fn_options != im.fn_state;
    b.valid = ok;
    im.qualified = ok && d.qualified != 0;
    co.qualified = im.qualified;
    return b;
}

}  // namespace

FgTitleBinding FgCp2077Binding() { return Build(WG_FG_TB_DATA(CP2077)); }

FgTitleBinding FgSl2GenericBinding() { return Build(WG_FG_TB_DATA(SL2_GENERIC)); }

bool FgSelectShippedBinding(const std::wstring& exe_leaf, FgTitleBinding* out) {
    if (!out || exe_leaf.empty()) return false;
    const FgTitleBinding cp = FgCp2077Binding();
    std::wstring want;
    for (char c : cp.exe_leaf) want.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    if (!cp.valid || _wcsicmp(exe_leaf.c_str(), want.c_str()) != 0) return false;
    *out = cp;
    return true;
}

void FgApplyBinding(const FgTitleBinding& b, FgAsiConfig* cfg) {
    if (!cfg) return;
    cfg->imports = b.imports;
    cfg->core.binding = b.core;
    cfg->core.ids.fg = b.fg_feature;
}

}  // namespace win32
}  // namespace wgfgp
