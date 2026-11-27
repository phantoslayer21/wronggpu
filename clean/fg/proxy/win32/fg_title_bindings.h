// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): the qualified title bindings this build carries.
//
// A title binding is data (bindings/<id>.json): the game exe it is for, the game interface's file name, version gate and
// export name per role, the FG feature id and feature-function names, the DXGI factory creation the game imports from
// the interface, the native FG structure GUIDs, versions and sizes, the FG input buffer types and optional-input
// declarations, and the UI range. CMakeLists.txt turns each JSON into a header of macros in the build tree
// (bindings/fg_binding_gen.cmake); this file turns those macros into the transport's structs. No vendor name is written
// in the code: the names reach the build only through the binding data.
//
// Carried: Cyberpunk 2077 on interface build 2.14.1 (bindings/cp2077-2.14.1.json), selected by the exe's file name,
// and the generic interface 2.x binding (bindings/sl2-generic.json: major 2, minor 4 or later), used for any other game
// that loads the interface. The transport then still checks the loaded interface's version (the title binding's
// major.minor exactly, the generic binding's range) and patches only the import slots that name a bound role (or the
// host routes them). The core takes frame generation only when every route fact holds (fg_proxy_core.h); until then
// every call keeps forwarding.
// Written for this project.
#pragma once
#include <string>

#include "fg_asi_transport.h"
#include "fg_sl214_codec.h"

namespace wgfgp {
namespace win32 {

struct FgTitleBinding {
    bool valid = false;        // every value parsed (GUIDs, sizes, numbers); a binding that is not valid is never used
    std::string id, title, exe_leaf;
    uint32_t fg_feature = WG_FEAT_FRAME_GENERATION;
    FgImportBinding imports;   // interface file, version gate, export names, feature-function names, factory import
    FgBinding core;            // the core's binding: input buffer types, function names, UI range, presets
    FgSl214Layout layout;      // the codec's data: structure GUIDs, versions, sizes, optional inputs, lifetimes
};

// Cyberpunk 2077, interface build 2.14.1.
FgTitleBinding FgCp2077Binding();
// Any game on interface 2.4 or later without a title binding of its own (options and state from version 1 on).
FgTitleBinding FgSl2GenericBinding();

// The title binding for the running exe (its file name, compared without case); false: this build carries none for it
// (the generic binding is never returned here: FgSl2GenericBinding).
bool FgSelectShippedBinding(const std::wstring& exe_leaf, FgTitleBinding* out);

// The binding's parts into a transport configuration (imports, the core's binding, the FG feature id). The codec
// (an FgSl214Codec over b.layout) is the caller's and must outlive the transport.
void FgApplyBinding(const FgTitleBinding& b, FgAsiConfig* cfg);

}  // namespace win32
}  // namespace wgfgp
