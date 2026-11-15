// SPDX-License-Identifier: MIT
// The identity of the adapter a D3D12 device runs on, read where another layer in the game process cannot rewrite it
// (R1, oversight 2026-11-15; R1-AUDIT H8). WrongGPU's own code. Header only, Windows only: every caller (the interop
// backend's executor choice, the native D3D12 device) includes it; nothing here needs an import library (the kernel-mode
// thunks come from gdi32.dll at run time), so the many small test builds of device_d3d12.cpp link as before.
//
// Why: OptiScaler (and other proxies) can sit in the game folder as dxgi.dll and rewrite DXGI's adapter description and
// vendor id, to make a game offer another vendor's options (the launcher's GPU-report switch in its OptiScaler tab; an
// experimental RR game profile). Reading the adapter through the game's dxgi.dll then shows another GPU, and a real RX 9000
// would lose WrongGPU. Two sources are used instead:
//  - the system's own DXGI: <system directory>\dxgi.dll loaded by its full path (a bare "dxgi.dll", with or without
//    LOAD_LIBRARY_SEARCH_SYSTEM32, returns a proxy that is already loaded under that name), enumerated by the device's
//    LUID;
//  - the kernel-mode driver's vendor and device id for that LUID (D3DKMTQueryAdapterInfo, PHYSICALADAPTERDEVICEIDS),
//    which does not go through DXGI at all.
// wgrt::rdna4_identity (rdna4.h) decides from them.
#pragma once

#include <windows.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <string>

#include "rdna4.h"

namespace wgrt {

namespace kmt {
// The few kernel-mode thunk declarations needed (d3dkmthk.h names and layouts), so that no WDK header and no import
// library is required.
struct OpenAdapterFromLuid {
  LUID AdapterLuid;
  UINT hAdapter;
};
struct QueryAdapterInfo {
  UINT hAdapter;
  UINT Type;  // KMTQAITYPE_PHYSICALADAPTERDEVICEIDS = 31
  void* pPrivateDriverData;
  UINT PrivateDriverDataSize;
};
struct DeviceIds {
  UINT VendorID, DeviceID, SubVendorID, SubSystemID, RevisionID, BusType;
};
struct QueryDeviceIds {
  UINT PhysicalAdapterIndex;
  DeviceIds Ids;
};
struct CloseAdapter {
  UINT hAdapter;
};
constexpr UINT kPhysicalAdapterDeviceIds = 31;
}  // namespace kmt

// The kernel-mode driver's PCI vendor and device id of the adapter with this LUID. False when the query is not
// available (Wine without the thunks, a software adapter, a refused call).
inline bool driver_device_ids(const LUID& luid, uint32_t* vendor, uint32_t* device) {
  using OpenFn = LONG(WINAPI*)(kmt::OpenAdapterFromLuid*);
  using QueryFn = LONG(WINAPI*)(kmt::QueryAdapterInfo*);
  using CloseFn = LONG(WINAPI*)(kmt::CloseAdapter*);
  HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
  if (!gdi) gdi = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!gdi) return false;
  auto open = reinterpret_cast<OpenFn>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid")));
  auto query = reinterpret_cast<QueryFn>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTQueryAdapterInfo")));
  auto close = reinterpret_cast<CloseFn>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTCloseAdapter")));
  if (!open || !query || !close) return false;
  kmt::OpenAdapterFromLuid o = {};
  o.AdapterLuid = luid;
  if (open(&o) != 0) return false;
  kmt::QueryDeviceIds ids = {};
  kmt::QueryAdapterInfo q = {};
  q.hAdapter = o.hAdapter;
  q.Type = kmt::kPhysicalAdapterDeviceIds;
  q.pPrivateDriverData = &ids;
  q.PrivateDriverDataSize = sizeof ids;
  const LONG st = query(&q);
  kmt::CloseAdapter c = {};
  c.hAdapter = o.hAdapter;
  close(&c);
  if (st != 0) return false;
  *vendor = ids.Ids.VendorID;
  *device = ids.Ids.DeviceID;
  return true;
}

inline std::string wide_to_utf8(const wchar_t* w) {
  char buf[512] = {};
  WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, (int)sizeof buf - 1, nullptr, nullptr);
  return buf;
}

// A DLL of the system directory by its full path: a bare name returns a module of that name that is already loaded, a proxy
// in the game folder included. nullptr when it cannot be loaded.
inline HMODULE load_system_dll(const wchar_t* name) {
  wchar_t dir[MAX_PATH] = {};
  const UINT n = GetSystemDirectoryW(dir, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) return nullptr;
  const std::wstring path = std::wstring(dir) + L"\\" + name;
  return LoadLibraryW(path.c_str());
}

// The adapter with this LUID as the SYSTEM's DXGI describes it (never the game's dxgi.dll).
inline bool system_dxgi_description(const LUID& luid, uint32_t* vendor, bool* software, std::string* description) {
  HMODULE dxgi = load_system_dll(L"dxgi.dll");
  if (!dxgi) return false;
  using MakeFactory = HRESULT(WINAPI*)(REFIID, void**);
  auto create = reinterpret_cast<MakeFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
  if (!create) return false;
  bool ok = false;
  IDXGIFactory4* f = nullptr;
  if (SUCCEEDED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&f))) && f) {
    IDXGIAdapter1* a = nullptr;
    if (SUCCEEDED(f->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&a))) && a) {
      DXGI_ADAPTER_DESC1 d = {};
      if (SUCCEEDED(a->GetDesc1(&d))) {
        ok = true;
        *vendor = d.VendorId;
        *software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        *description = wide_to_utf8(d.Description);
      }
      a->Release();
    }
    f->Release();
  }
  return ok;
}

// Everything the executor choice and the native device need to know about the adapter with this LUID.
inline AdapterIdentity read_adapter_identity(const LUID& luid) {
  AdapterIdentity id;
  id.dxgi_known = system_dxgi_description(luid, &id.dxgi_vendor, &id.software, &id.description);
  id.kmt_known = driver_device_ids(luid, &id.kmt_vendor, &id.kmt_device);
  return id;
}

}  // namespace wgrt
