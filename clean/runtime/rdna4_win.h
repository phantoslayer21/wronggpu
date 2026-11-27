// SPDX-License-Identifier: GPL-3.0-only
// The system's sources for an adapter's identity (rdna4.h has the rule and the injectable lookup). R1, oversight
// 2026-11-15; R1-AUDIT H8. WrongGPU's own code. Header only, Windows only: the interop backend's executor choice, the
// native D3D12 device and the Ray Reconstruction runtime include it; nothing here needs an import library (the kernel-mode
// thunks come from gdi32.dll at run time), so the many small test builds of device_d3d12.cpp link as before.
//
// Why: OptiScaler (and other proxies) can sit in the game folder as dxgi.dll and rewrite DXGI's adapter description and
// vendor id, to make a game offer another vendor's options (the launcher's GPU-report switch in its OptiScaler tab; an
// experimental RR game profile). Two sources a layer in the game process does not reach are used:
//  - the kernel-mode driver's vendor and device id for the adapter's LUID (D3DKMTQueryAdapterInfo,
//    PHYSICALADAPTERDEVICEIDS), which does not go through DXGI at all: this decides;
//  - the system's own DXGI, <system directory>\dxgi.dll loaded by its full path (a bare "dxgi.dll", with or without
//    LOAD_LIBRARY_SEARCH_SYSTEM32, returns a proxy that is already loaded under that name), enumerated by the same LUID:
//    its description is shown in log lines, it never decides.
// Under Wine only, where the driver's query answers STATUS_NOT_IMPLEMENTED, the system DXGI's ids stand in (rdna4.h, #248).
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

inline AdapterLuid to_luid(const LUID& l) {
  AdapterLuid r;
  r.low = l.LowPart;
  r.high = l.HighPart;
  return r;
}

inline LUID to_win_luid(const AdapterLuid& l) {
  LUID r;
  r.LowPart = l.low;
  r.HighPart = l.high;
  return r;
}

// The kernel-mode driver's PCI vendor and device id of the adapter with this LUID, and the query's own status: 0 when the
// ids were read, the query's NTSTATUS when it refused (Wine: STATUS_NOT_IMPLEMENTED), kStatusUnsuccessful when it could not
// be asked at all (no thunks, the adapter not opened).
inline int32_t driver_query_status(const AdapterLuid& luid, uint32_t* vendor, uint32_t* device) {
  using OpenFn = LONG(WINAPI*)(kmt::OpenAdapterFromLuid*);
  using QueryFn = LONG(WINAPI*)(kmt::QueryAdapterInfo*);
  using CloseFn = LONG(WINAPI*)(kmt::CloseAdapter*);
  HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
  if (!gdi) gdi = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!gdi) return kStatusUnsuccessful;
  auto open = reinterpret_cast<OpenFn>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid")));
  auto query = reinterpret_cast<QueryFn>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTQueryAdapterInfo")));
  auto close = reinterpret_cast<CloseFn>(reinterpret_cast<void*>(GetProcAddress(gdi, "D3DKMTCloseAdapter")));
  if (!open || !query || !close) return kStatusUnsuccessful;
  kmt::OpenAdapterFromLuid o = {};
  o.AdapterLuid = to_win_luid(luid);
  if (open(&o) != 0) return kStatusUnsuccessful;
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
  if (st != 0) return static_cast<int32_t>(st);
  *vendor = ids.Ids.VendorID;
  *device = ids.Ids.DeviceID;
  return 0;
}

// The same as a yes or no: the ids were read.
inline bool driver_device_ids(const AdapterLuid& luid, uint32_t* vendor, uint32_t* device) {
  return driver_query_status(luid, vendor, device) == 0;
}

// The process runs under Wine (Proton included): Wine's ntdll exports wine_get_version, Windows' never does.
inline bool running_under_wine() {
  HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  return ntdll && GetProcAddress(ntdll, "wine_get_version") != nullptr;
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

// The adapter with this LUID as the SYSTEM's DXGI describes it (never the game's dxgi.dll): for log lines.
inline bool system_dxgi_adapter(const AdapterLuid& luid, AdapterRecord* out) {
  HMODULE dxgi = load_system_dll(L"dxgi.dll");
  if (!dxgi) return false;
  using MakeFactory = HRESULT(WINAPI*)(REFIID, void**);
  auto create = reinterpret_cast<MakeFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
  if (!create) {
    FreeLibrary(dxgi);
    return false;
  }
  bool ok = false;
  IDXGIFactory4* f = nullptr;
  if (SUCCEEDED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&f))) && f) {
    IDXGIAdapter1* a = nullptr;
    if (SUCCEEDED(f->EnumAdapterByLuid(to_win_luid(luid), __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&a))) && a) {
      DXGI_ADAPTER_DESC1 d = {};
      if (SUCCEEDED(a->GetDesc1(&d))) {
        ok = true;
        out->luid = to_luid(d.AdapterLuid);
        out->vendor = d.VendorId;
        out->device = d.DeviceId;
        out->software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        out->description = wide_to_utf8(d.Description);
      }
      a->Release();
    }
    f->Release();
  }
  FreeLibrary(dxgi);
  return ok;
}

// The system DXGI's vendor and device id of the adapter with this LUID: the Wine fallback's source only (rdna4.h).
inline bool system_dxgi_ids(const AdapterLuid& luid, uint32_t* vendor, uint32_t* device) {
  AdapterRecord r;
  if (!system_dxgi_adapter(luid, &r)) return false;
  *vendor = r.vendor;
  *device = r.device;
  return true;
}

inline AdapterSources system_sources() {
  AdapterSources s;
  s.find_adapter = system_dxgi_adapter;
  s.driver_query = driver_query_status;
  s.under_wine = running_under_wine;
  s.dxgi_ids = system_dxgi_ids;
  return s;
}

// Everything the executor choice, the native device and the RR runtime need to know about the adapter with this LUID
// (the game device's GetAdapterLuid()).
inline AdapterIdentity read_adapter_identity(const LUID& luid) {
  return lookup_adapter(to_luid(luid), system_sources());
}

}  // namespace wgrt
