// SPDX-License-Identifier: MIT
// RR guard: a stand-in "other Ray Reconstruction plugin" of our own (wgtest_other_rr.dll) for
// tests/test_rr_guard_dll.cpp. It knows nothing of any real plugin; it only takes the game interface's RR calls the
// generic ways a plugin can, so that WrongGPU RR's guard has something to find:
//   wgtest_or_patch_iat     redirects the exe's import slots of the hooked roles to its handlers (forward = 1: each
//                           handler continues with what the slot held, i.e. it wraps whatever was there; 0: it continues
//                           with the interface's export and replaces what was there)
//   wgtest_or_patch_export  overwrites the first bytes of an interface export with a jump to its handler (relay = 1:
//                           a jmp rel32 to an anonymous relay near the export that jumps on, as hooking libraries do)
//   wgtest_or_set_slot      writes any value into one import slot (code outside every module, an unreadable page)
//   wgtest_or_routes        registers its handlers as game-interface routes with the host (plugin interface 3)
//   wgtest_or_register_ngx  registers an NGX RR provider with the host (plugin interface 4)
// Its handlers answer the RR feature themselves (is-supported 0, evaluate / allocate / get-function counted and 0) and
// pass every other call on. The role names and order come from the hook binding (wg_binding.inc).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <string.h>

#include "wg_binding.inc"
#include "../include/rr_proxy_abi.h"
#include "../include/wg_host_menu_abi.h"

namespace {

using Fwd = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

struct OrState {
  long calls[32];     // calls that reached each handler, by role
  long rr_calls;      // RR-feature calls the stand-in answered itself
  long ngx_creates;   // NGX provider creates
};
OrState g_st;
Fwd g_next[32];       // where each handler continues
bool g_inline[32];    // the export of this role was overwritten (no way back to it): non-RR calls are answered here

// the feature argument of a role, if it has one
bool feature_of(int role, uint64_t a, uint64_t b, uint32_t* f) {
  switch (role) {
    case 2: case 3: case 4: case 5: case 6: case 7: case 9: case 16: *f = (uint32_t)a; return true;
    case 8: *f = (uint32_t)b; return true;  // allocate(cmd_list, feature, viewport)
    default: return false;
  }
}

template <int R>
uint64_t handler(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
  InterlockedIncrement(&g_st.calls[R]);
  uint32_t feat = 0;
  if (feature_of(R, a, b, &feat) && feat == WG_FEAT_RAY_RECONSTRUCTION && (R == 2 || R == 7 || R == 8 || R == 16)) {
    InterlockedIncrement(&g_st.rr_calls);
    if (R == 7 && c) *reinterpret_cast<void**>(c) = reinterpret_cast<void*>(&handler<R>);  // "its" feature function
    return 0;
  }
  if (g_inline[R]) return feature_of(R, a, b, &feat) ? 100u + feat % 50u : 0;  // what the stand-in interface returns
  return g_next[R] ? g_next[R](a, b, c, d, e, f) : (uint64_t)WG_E_MISSING_OR_INVALID_API;
}

void* const kHandlers[19] = {
    (void*)&handler<0>,  (void*)&handler<1>,  (void*)&handler<2>,  (void*)&handler<3>,  (void*)&handler<4>,
    (void*)&handler<5>,  (void*)&handler<6>,  (void*)&handler<7>,  (void*)&handler<8>,  (void*)&handler<9>,
    (void*)&handler<10>, (void*)&handler<11>, (void*)&handler<12>, (void*)&handler<13>, (void*)&handler<14>,
    (void*)&handler<15>, (void*)&handler<16>, (void*)&handler<17>, (void*)&handler<18>};

int role_of(const char* name) {
  for (unsigned i = 0; i < WG_BINDING_HOOK_ROLE_COUNT; ++i) {
    const int r = kWgHookRoles[i];
    if (kWgRoleNames[r][0] && strcmp(kWgRoleNames[r], name) == 0) return r;
  }
  return -1;
}

bool write_ptr(void** slot, void* value) {
  DWORD old = 0;
  if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
  InterlockedExchangePointer(slot, value);
  VirtualProtect(slot, sizeof(void*), old, &old);
  return true;
}

// f(role or -1, name, slot) for every by-name import of `leaf` in `module`
template <typename F>
void each_import(HMODULE module, const char* leaf, F f) {
  uint8_t* base = reinterpret_cast<uint8_t*>(module);
  const IMAGE_NT_HEADERS64* nt =
      reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
  const IMAGE_DATA_DIRECTORY& d = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (!d.VirtualAddress) return;
  for (auto* imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + d.VirtualAddress); imp->Name; ++imp) {
    if (_stricmp(reinterpret_cast<const char*>(base + imp->Name), leaf) != 0) continue;
    auto* names = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk
                                                                                         : imp->FirstThunk));
    auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + imp->FirstThunk);
    for (; names->u1.AddressOfData; ++names, ++iat) {
      if (names->u1.Ordinal & IMAGE_ORDINAL_FLAG64) continue;
      const char* nm = reinterpret_cast<const char*>(
          reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData)->Name);
      f(role_of(nm), nm, reinterpret_cast<void**>(&iat->u1.Function));
    }
  }
}

// a page within +-1.5 GB of `close_to` (for a jmp rel32), searched downward then upward
uint8_t* alloc_near(const void* close_to) {
  const uintptr_t p = reinterpret_cast<uintptr_t>(close_to) & ~(uintptr_t)0xFFFF;
  for (uintptr_t d = 0x10000; d < 0x60000000; d += 0x10000) {
    for (int s = 0; s < 2; ++s) {
      const uintptr_t a = s ? p + d : p - d;
      void* m = VirtualAlloc(reinterpret_cast<void*>(a), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
      if (m) return static_cast<uint8_t*>(m);
    }
  }
  return nullptr;
}

void put_abs(uint8_t* at, const void* target) {  // jmp [rip+0]; dq target
  at[0] = 0xFF;
  at[1] = 0x25;
  memset(at + 2, 0, 4);
  memcpy(at + 6, &target, 8);
}

// host routes: the host's next pointers of the routes this stand-in registered
void* g_route_next[32];

bool ngx_create(void*, void*, void*, const WgHostNgxParams*, void** ctx) {
  InterlockedIncrement(&g_st.ngx_creates);
  *ctx = &g_st;
  return true;
}
bool ngx_evaluate(void*, void*, void*, const WgHostNgxParams*) { return true; }
void ngx_release(void*, void*) {}

}  // namespace

extern "C" {

__declspec(dllexport) const OrState* wgtest_or_state(void) { return &g_st; }

__declspec(dllexport) int wgtest_or_patch_iat(HMODULE exe, const char* iface_leaf, int forward) {
  HMODULE iface = GetModuleHandleA(iface_leaf);
  int n = 0;
  each_import(exe, iface_leaf, [&](int role, const char* nm, void** slot) {
    if (role < 0) return;
    g_next[role] = forward ? reinterpret_cast<Fwd>(*slot) : reinterpret_cast<Fwd>(GetProcAddress(iface, nm));
    if (write_ptr(slot, kHandlers[role])) ++n;
  });
  return n;
}

__declspec(dllexport) int wgtest_or_set_slot(HMODULE exe, const char* iface_leaf, const char* import_name, void* value) {
  int n = 0;
  each_import(exe, iface_leaf, [&](int, const char* nm, void** slot) {
    if (strcmp(nm, import_name) == 0 && write_ptr(slot, value)) ++n;
  });
  return n;
}

__declspec(dllexport) int wgtest_or_patch_export(HMODULE iface, const char* name, int relay) {
  const int role = role_of(name);
  uint8_t* e = reinterpret_cast<uint8_t*>(GetProcAddress(iface, name));
  if (role < 0 || !e) return 0;
  DWORD old = 0;
  if (!VirtualProtect(e, 16, PAGE_EXECUTE_READWRITE, &old)) return 0;
  g_inline[role] = true;
  if (relay) {
    uint8_t* r = alloc_near(e);
    if (!r) return 0;
    put_abs(r, kHandlers[role]);
    e[0] = 0xE9;
    const int32_t rel = (int32_t)(r - (e + 5));
    memcpy(e + 1, &rel, 4);
  } else {
    put_abs(e, kHandlers[role]);
  }
  VirtualProtect(e, 16, old, &old);
  FlushInstructionCache(GetCurrentProcess(), e, 16);
  return 1;
}

__declspec(dllexport) int wgtest_or_routes(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
  WgHostSlRoute routes[4];
  const int roles[4] = {2, 7, 8, 16};
  for (int i = 0; i < 4; ++i) routes[i] = {kWgRoleNames[roles[i]], kHandlers[roles[i]], &g_route_next[roles[i]]};
  if (!api->register_streamline_routes(plugin, routes, 4)) return 0;
  for (int i = 0; i < 4; ++i) g_next[roles[i]] = reinterpret_cast<Fwd>(g_route_next[roles[i]]);
  return 4;
}

__declspec(dllexport) bool wgtest_or_register_ngx(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
  static WgHostNgxRr p;
  p.size = sizeof p;
  p.user = &g_st;
  p.create = ngx_create;
  p.evaluate = ngx_evaluate;
  p.release = ngx_release;
  return api->register_ngx_rr(plugin, &p);
}

}  // extern "C"
