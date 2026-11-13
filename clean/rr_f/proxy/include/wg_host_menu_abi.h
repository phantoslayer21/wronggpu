// SPDX-License-Identifier: MIT
// The host menu interface the RR plugin uses to add its own tab to OptiScaler's menu (OptiScaler builds with the plugin
// interface: an ASI that exports OptiPluginInit is handed a table of plain C functions). The interface is part of the
// WrongGPU edition of OptiScaler (the project owner's GPL-licensed fork). This file declares the binary interface the
// plugin needs (a version-1 table: two uint32 fields, then function pointers in the order below); it includes no
// OptiScaler header and no text was copied into it from any third-party file. Fields are only ever appended by the host, so a table whose `size` covers a
// field has it. Strings are UTF-8. Widget functions only work inside a tab draw callback (render thread, menu open).
#pragma once

#include <stddef.h>
#include <stdint.h>

extern "C" {

struct WgHostMenuPlugin;  // the host's handle of this plugin (opaque)
struct WgHostSlRoute;     // version 3, below
struct WgHostNgxRr;       // version 4, below
typedef void (*WgHostDrawFn)(void* user);

struct WgHostMenuApi {
  uint32_t size;     // bytes of the host's table
  uint32_t version;  // 1
  // registration
  bool (*register_tab)(WgHostMenuPlugin* plugin, const char* name, const char* description, WgHostDrawFn draw, void* user);
  // layout (inside a draw callback)
  void (*section)(const char* title);
  void (*separator)(void);
  void (*same_line)(void);
  void (*spacing)(void);
  void (*disabled)(bool disabled);
  void (*item_tooltip)(const char* text);
  // text
  void (*text)(const char* text);
  void (*text_disabled)(const char* text);
  void (*text_colored)(float r, float g, float b, float a, const char* text);
  // widgets: true when clicked / changed
  bool (*button)(const char* label);
  bool (*checkbox)(const char* label, bool* value);
  bool (*slider_float)(const char* label, float* value, float min, float max, const char* format);
  bool (*slider_int)(const char* label, int32_t* value, int32_t min, int32_t max, const char* format);
  bool (*combo)(const char* label, int32_t* current, const char* const* items, int32_t count);
  bool (*input_text)(const char* label, char* buffer, size_t size);
  bool (*color_edit)(const char* label, float* rgba, bool alpha);
  // menu state
  float (*menu_scale)(void);
  bool (*menu_open)(void);
  // host settings ([Plugin.<name>] of OptiScaler.ini); not used: WrongGPU's settings live in WrongGPU\wronggpu.ini
  bool (*get_bool)(WgHostMenuPlugin* plugin, const char* key, bool def);
  int32_t (*get_int)(WgHostMenuPlugin* plugin, const char* key, int32_t def);
  float (*get_float)(WgHostMenuPlugin* plugin, const char* key, float def);
  bool (*get_string)(WgHostMenuPlugin* plugin, const char* key, char* buffer, size_t size, const char* def);
  void (*set_bool)(WgHostMenuPlugin* plugin, const char* key, bool value);
  void (*set_int)(WgHostMenuPlugin* plugin, const char* key, int32_t value);
  void (*set_float)(WgHostMenuPlugin* plugin, const char* key, float value);
  void (*set_string)(WgHostMenuPlugin* plugin, const char* key, const char* value);
  void (*reset_value)(WgHostMenuPlugin* plugin, const char* key);
  bool (*save_settings)(WgHostMenuPlugin* plugin);
  // the host's log, prefixed with the plugin name; level 0 debug, 1 info, 2 warning, 3 error
  void (*log)(WgHostMenuPlugin* plugin, int32_t level, const char* message);
  // version 2: a card at the top of the host's Custom tab (the tab its menu opens on), shown in every upscaler state
  bool (*register_custom_card)(WgHostMenuPlugin* plugin, const char* title, WgHostDrawFn draw, void* user);
  // version 2: an upscaler the plugin runs itself: its name and the GPU time of its last frame (ms), shown in the
  // host's overlay and menu while none of the host's own upscalers runs; once per frame, any thread
  void (*report_upscaler)(WgHostMenuPlugin* plugin, const char* name, float gpu_ms);
  // version 3: game-interface functions the plugin handles. The host puts `handler` in front of the named export of
  // the game interface it hooks (games that load the interface at run time and resolve it by name included) and
  // stores the function to continue with in *next (null while not installed); the array is copied
  bool (*register_streamline_routes)(WgHostMenuPlugin* plugin, const WgHostSlRoute* routes, uint32_t count);
  // version 3: called each time the host hooked a game interface module (and at once if it already has), with its
  // file version, before routes are installed; the plugin registers its routes from here
  bool (*set_streamline_callback)(WgHostMenuPlugin* plugin, void (*hooked)(void* module, uint32_t major, uint32_t minor,
                                                                           uint32_t patch, void* user),
                                  void* user);
  // version 4: a ray reconstruction provider for games that call the NGX interface directly (spec page 66). While one
  // is registered the host reports RR as available in the NGX capability parameters and sends the D3D12 RR feature's
  // create / evaluate / release to it (one provider per process; the structure is copied; false when one is already
  // registered or the structure is incomplete). The host brackets each evaluate like begin/end_cmd_list_work below
  // (version 5) and maps a refused create / failed evaluate to the NGX failure codes the game sees.
  bool (*register_ngx_rr)(WgHostMenuPlugin* plugin, const WgHostNgxRr* provider);
  // version 5: work the plugin records on the game's own command list, outside of the host's own upscaler calls.
  // begin_cmd_list_work before the first command, end_cmd_list_work after the last one, on the game's thread; the host
  // stops recording the list's bound state in between and, at the end, puts back the game's root signatures (and with
  // its extended restore also the root arguments, descriptor heaps and pipeline state) as the game last bound them.
  void (*begin_cmd_list_work)(WgHostMenuPlugin* plugin, void* command_list);
  void (*end_cmd_list_work)(WgHostMenuPlugin* plugin, void* command_list);
};

struct WgHostSlRoute {
  const char* export_name;  // the export of the game interface module
  void* handler;            // ours, same signature
  void** next;              // the function to continue with, set by the host
};

// Version 4: read access to the game's NGX parameter block for the duration of one call. Keys are the block's own key
// strings (spec annex 66.A). Each getter returns false when the key is not set and then leaves the output unchanged.
struct WgHostNgxParams {
  void* block;                                                   // the game's parameter block (opaque)
  bool (*get_resource)(void* block, const char* key, void** resource);  // ID3D12Resource*
  bool (*get_pointer)(void* block, const char* key, void** value);      // plain pointers (the camera matrices)
  bool (*get_uint)(void* block, const char* key, uint32_t* value);
  bool (*get_int)(void* block, const char* key, int32_t* value);
  bool (*get_float)(void* block, const char* key, float* value);
};

// Version 4: the provider. Calls arrive on the game's thread inside its NGX call; the command list is the game's own
// (record into it; never close or execute it). create returns false to refuse the feature (the game sees a failure
// and keeps its own path), evaluate returns false on failure.
struct WgHostNgxRr {
  uint32_t size;  // sizeof(WgHostNgxRr)
  void* user;     // handed back to every function
  bool (*create)(void* user, void* d3d12_device, void* command_list, const WgHostNgxParams* params, void** context);
  bool (*evaluate)(void* user, void* context, void* command_list, const WgHostNgxParams* params);
  void (*release)(void* user, void* context);  // the game released the feature (or the host shuts down)
};
static_assert(sizeof(WgHostNgxParams) == 6 * sizeof(void*), "WgHostNgxParams layout");
static_assert(sizeof(WgHostNgxRr) == 5 * sizeof(void*) && offsetof(WgHostNgxRr, user) == sizeof(void*),
              "WgHostNgxRr layout");

// True when the host's table has `field` (and it is set).
#define WG_HOST_MENU_HAS(api, field) \
  ((api) != nullptr && (api)->size >= offsetof(WgHostMenuApi, field) + sizeof((api)->field) && (api)->field != nullptr)

// The export the host looks for; return false to add nothing.
typedef bool (*WgHostMenuInitFn)(const WgHostMenuApi* api, WgHostMenuPlugin* plugin);
#define WG_HOST_MENU_INIT_EXPORT "OptiPluginInit"

}  // extern "C"
