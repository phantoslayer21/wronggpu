// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the qualified title bindings this build carries. See fg_title_bindings.h.
// Written for this project.
#include "fg_title_bindings.h"

#include <wchar.h>

// Generated at configure time from bindings/cp2077-2.14.1.json (build tree; bindings/fg_binding_gen.cmake).
#include "fg_binding_cp2077.h"

namespace wgfgp {
namespace win32 {

namespace {

bool Guid(const char* text, WgGuid* out) { return FgParseGuid(text, out); }

}  // namespace

FgTitleBinding FgCp2077Binding() {
    FgTitleBinding b;
    b.id = WG_FG_TB_CP2077_ID;
    b.title = WG_FG_TB_CP2077_TITLE;
    b.exe_leaf = WG_FG_TB_CP2077_EXE_LEAF;
    b.fg_feature = WG_FG_TB_CP2077_FG_FEATURE;

    FgImportBinding& im = b.imports;
    im.interposer_leaf = WG_FG_TB_CP2077_INTERFACE_LEAF;
    im.version_major = WG_FG_TB_CP2077_INTERFACE_MAJOR;
    im.version_minor = WG_FG_TB_CP2077_INTERFACE_MINOR;
    im.role_names[kFgRoleIsSupported] = WG_FG_TB_CP2077_ROLE_IS_SUPPORTED;
    im.role_names[kFgRoleRequirements] = WG_FG_TB_CP2077_ROLE_REQUIREMENTS;
    im.role_names[kFgRoleGetFunction] = WG_FG_TB_CP2077_ROLE_GET_FUNCTION;
    im.role_names[kFgRoleTag] = WG_FG_TB_CP2077_ROLE_TAG;
    im.role_names[kFgRoleTagForFrame] = WG_FG_TB_CP2077_ROLE_TAG_FOR_FRAME;
    im.role_names[kFgRoleConstants] = WG_FG_TB_CP2077_ROLE_CONSTANTS;
    im.role_names[kFgRoleEvaluate] = WG_FG_TB_CP2077_ROLE_EVALUATE;
    im.role_names[kFgRoleFree] = WG_FG_TB_CP2077_ROLE_FREE;
    im.fn_options = WG_FG_TB_CP2077_FN_OPTIONS;
    im.fn_state = WG_FG_TB_CP2077_FN_STATE;
    im.abi_max_generated = WG_FG_TB_CP2077_ABI_MAX_GENERATED;
    im.factory2_name = WG_FG_TB_CP2077_FACTORY_IMPORT;

    FgBinding& co = b.core;
    co.version_major = im.version_major;
    co.version_minor = im.version_minor;
    co.native_type[WG_FG_ROLE_BACKBUFFER] = WG_FG_TB_CP2077_BT_BACKBUFFER;
    co.native_type[WG_FG_ROLE_MOTION] = WG_FG_TB_CP2077_BT_MOTION;
    co.native_type[WG_FG_ROLE_DEPTH] = WG_FG_TB_CP2077_BT_DEPTH;
    co.native_type[WG_FG_ROLE_HUDLESS] = WG_FG_TB_CP2077_BT_HUDLESS;
    co.native_type[WG_FG_ROLE_UI_ALPHA] = WG_FG_TB_CP2077_BT_UI_ALPHA;
    co.native_type[WG_FG_ROLE_DISTORTION] = WG_FG_TB_CP2077_BT_DISTORTION;
    co.native_type[WG_FG_ROLE_NO_WARP] = WG_FG_TB_CP2077_BT_NO_WARP;
    co.fn_names = {im.fn_options, im.fn_state};
    co.ui_max_generated = WG_FG_TB_CP2077_UI_MAX_GENERATED;
    co.presets_mask = WG_FG_TB_CP2077_PRESETS_MASK;

    FgSl214Layout& l = b.layout;
    bool ok = Guid(WG_FG_TB_CP2077_GUID_OPTIONS, &l.options_guid) && Guid(WG_FG_TB_CP2077_GUID_STATE, &l.state_guid) &&
              Guid(WG_FG_TB_CP2077_GUID_VIEWPORT, &l.viewport_guid) && Guid(WG_FG_TB_CP2077_GUID_TAG, &l.tag_guid) &&
              Guid(WG_FG_TB_CP2077_GUID_RESOURCE, &l.resource_guid) &&
              Guid(WG_FG_TB_CP2077_GUID_CONSTANTS, &l.constants_guid) &&
              Guid(WG_FG_TB_CP2077_GUID_REQUIREMENTS, &l.requirements_guid);
    l.options_version = WG_FG_TB_CP2077_OPTIONS_VERSION;
    l.options_size = WG_FG_TB_CP2077_OPTIONS_SIZE;
    l.state_version = WG_FG_TB_CP2077_STATE_VERSION;
    l.state_size = WG_FG_TB_CP2077_STATE_SIZE;
    l.constants_min_version = WG_FG_TB_CP2077_CONSTANTS_MIN_VERSION;
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r) l.native_type[r] = co.native_type[r];
    l.optional_decl[WG_FG_ROLE_HUDLESS - WG_FG_OPTIONAL_FIRST_ROLE] = WG_FG_TB_CP2077_DECL_HUDLESS;
    l.optional_decl[WG_FG_ROLE_UI_ALPHA - WG_FG_OPTIONAL_FIRST_ROLE] = WG_FG_TB_CP2077_DECL_UI_ALPHA;
    l.optional_decl[WG_FG_ROLE_DISTORTION - WG_FG_OPTIONAL_FIRST_ROLE] = WG_FG_TB_CP2077_DECL_DISTORTION;
    l.optional_decl[WG_FG_ROLE_NO_WARP - WG_FG_OPTIONAL_FIRST_ROLE] = WG_FG_TB_CP2077_DECL_NO_WARP;
    l.tag_lifetime_mask = WG_FG_TB_CP2077_TAG_LIFETIME_MASK;
    l.min_size = WG_FG_TB_CP2077_MIN_SIZE;
    l.dynamic_supported = WG_FG_TB_CP2077_DYNAMIC_SUPPORTED != 0;

    ok = ok && FgSl214LayoutValid(l) && im.abi_max_generated >= 1 && im.abi_max_generated <= WG_FG_MAX_GENERATED &&
         co.ui_max_generated >= 1 && co.ui_max_generated <= WG_FG_MAX_GENERATED &&
         co.native_type[WG_FG_ROLE_MOTION] != WG_FG_NATIVE_TYPE_UNBOUND &&
         co.native_type[WG_FG_ROLE_DEPTH] != WG_FG_NATIVE_TYPE_UNBOUND && !im.interposer_leaf.empty() &&
         !im.fn_options.empty() && !im.fn_state.empty() && im.fn_options != im.fn_state;
    b.valid = ok;
    im.qualified = ok && WG_FG_TB_CP2077_QUALIFIED != 0;
    co.qualified = im.qualified;
    return b;
}

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
