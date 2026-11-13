// SPDX-License-Identifier: MIT
// Unit tests of the SetupAPI identity shim (src/setupapi_identity.h; clean/rr-sm2-shim). Synthetic IDs through fake
// SetupAPI functions: match / no match, multi-strings with several entries, the buffer-too-small path untouched, other
// classes / devices / properties untouched, spoof inactive -> no rewrite, the decision (made lazily at the first calls,
// with retries and the host-settings fallback; the host ini parser). Then the import-slot
// redirect on this test exe itself (it imports both functions from setupapi.dll): read-only SetupAPI queries of the
// PC's Display-class devices (no GPU device is created; skipped when the PC lists no PCI display device).
#include "../src/setupapi_identity.h"

#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include <map>
#include <string>
#include <vector>

#include "test_util.h"

#pragma comment(lib, "setupapi.lib")

using wgsi::PciPair;

namespace {

const GUID kDisplay = {0x4d36e968, 0xe325, 0x11ce, {0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18}};
const GUID kNet = {0x4d36e972, 0xe325, 0x11ce, {0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18}};
const PciPair kReal{0x1002, 0x7550}, kSpoof{0x10de, 0x2206};
const DWORD kSuccessError = 0x1234abcd;  // the fakes leave this last error on success; it must come back unchanged

// ---- a fake device list
struct FakeDevice {
  std::wstring id;
  GUID cls;
  std::map<DWORD, std::pair<DWORD, std::wstring>> props;  // property -> (type, data; multi-strings with embedded L'\0')
};
std::vector<FakeDevice> g_devices;
HDEVINFO const kSet = reinterpret_cast<HDEVINFO>(0x5e7);

std::wstring multi(std::initializer_list<const wchar_t*> entries) {
  std::wstring s;
  for (auto e : entries) { s += e; s.push_back(L'\0'); }
  s.push_back(L'\0');
  return s;
}

SP_DEVINFO_DATA dev(unsigned i) {
  SP_DEVINFO_DATA d = {};
  d.cbSize = sizeof d;
  d.ClassGuid = g_devices[i].cls;
  d.DevInst = i;
  return d;
}

BOOL WINAPI fake_instance_id(HDEVINFO set, PSP_DEVINFO_DATA d, PWSTR buf, DWORD size, PDWORD required) {
  if (set != kSet || !d || d->DevInst >= g_devices.size()) { SetLastError(ERROR_INVALID_PARAMETER); return FALSE; }
  const std::wstring& id = g_devices[d->DevInst].id;
  const DWORD need = (DWORD)id.size() + 1;
  if (required) *required = need;
  if (!buf || size < need) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return FALSE; }
  memcpy(buf, id.c_str(), need * sizeof(wchar_t));
  SetLastError(kSuccessError);
  return TRUE;
}

BOOL WINAPI fake_registry_property(HDEVINFO set, PSP_DEVINFO_DATA d, DWORD prop, PDWORD type, PBYTE buf, DWORD size,
                                   PDWORD required) {
  if (set != kSet || !d || d->DevInst >= g_devices.size()) { SetLastError(ERROR_INVALID_PARAMETER); return FALSE; }
  auto it = g_devices[d->DevInst].props.find(prop);
  if (it == g_devices[d->DevInst].props.end()) { SetLastError(ERROR_INVALID_DATA); return FALSE; }
  const DWORD bytes = (DWORD)(it->second.second.size() * sizeof(wchar_t));
  if (type) *type = it->second.first;
  if (required) *required = bytes;
  if (!buf || size < bytes) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return FALSE; }
  memcpy(buf, it->second.second.data(), bytes);
  SetLastError(kSuccessError);
  return TRUE;
}

std::vector<std::string> g_log;
wgsi::Decision active_decision() {
  wgsi::Decision d;
  d.active = true;
  d.real = kReal;
  d.spoof = kSpoof;
  return d;
}
void configure(wgsi::Shim& s, bool active) {
  wgsi::Decision d = active ? active_decision() : wgsi::Decision();
  if (!active) { d.real = kReal; d.spoof = kSpoof; }  // the pairs are known, but the spoof is off
  s.configure(d, &fake_instance_id, &fake_registry_property, [](const std::string& l) { g_log.push_back(l); });
}

const wchar_t* kGpuId = L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0\\4&1A2B3C4D&0&00000008";
const wchar_t* kGpuIdSpoofed = L"PCI\\VEN_10DE&DEV_2206&SUBSYS_12345678&REV_C0\\4&1A2B3C4D&0&00000008";
const wchar_t* kDriverKey = L"{4d36e968-e325-11ce-bfc1-08002be10318}\\0001";

void make_devices() {
  g_devices.clear();
  FakeDevice gpu;  // 0: the real active adapter
  gpu.id = kGpuId;
  gpu.cls = kDisplay;
  gpu.props[SPDRP_HARDWAREID] = {REG_MULTI_SZ, multi({L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0",
                                                      L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678",
                                                      L"PCI\\VEN_1002&DEV_7550&CC_030000",
                                                      L"PCI\\VEN_1002&DEV_7550&CC_0300"})};
  gpu.props[SPDRP_COMPATIBLEIDS] = {REG_MULTI_SZ, multi({L"PCI\\VEN_1002&DEV_7550&REV_C0", L"PCI\\VEN_1002&DEV_7550",
                                                         L"PCI\\VEN_1002&CC_030000", L"PCI\\VEN_1002&CC_0300",
                                                         L"PCI\\CC_030000", L"PCI\\CC_0300"})};
  gpu.props[SPDRP_DRIVER] = {REG_SZ, std::wstring(kDriverKey) + L'\0'};
  gpu.props[SPDRP_DEVICEDESC] = {REG_SZ, std::wstring(L"PCI\\VEN_1002&DEV_7550 description") + L'\0'};
  g_devices.push_back(gpu);
  FakeDevice igpu;  // 1: another Display-class device (an integrated GPU of the same vendor)
  igpu.id = L"PCI\\VEN_1002&DEV_13C0&SUBSYS_87654321&REV_C1\\4&1A2B3C4D&0&0041";
  igpu.cls = kDisplay;
  igpu.props[SPDRP_HARDWAREID] = {REG_MULTI_SZ, multi({L"PCI\\VEN_1002&DEV_13C0&SUBSYS_87654321&REV_C1",
                                                       L"PCI\\VEN_1002&DEV_13C0"})};
  g_devices.push_back(igpu);
  FakeDevice net;  // 2: another class whose IDs happen to name the real pair (synthetic)
  net.id = kGpuId;
  net.cls = kNet;
  net.props[SPDRP_HARDWAREID] = gpu.props[SPDRP_HARDWAREID];
  g_devices.push_back(net);
  FakeDevice audio;  // 3: the board's HD audio function: Display class here (synthetic) but an HDAUDIO instance ID
  audio.id = L"HDAUDIO\\FUNC_01&VEN_1002&DEV_AA01&SUBSYS_00AA0100&REV_1008\\5&1&0&0001";
  audio.cls = kDisplay;
  audio.props[SPDRP_HARDWAREID] = {REG_MULTI_SZ, multi({L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0"})};
  g_devices.push_back(audio);
}

std::wstring entries_of(const wchar_t* buf, size_t nchars) { return std::wstring(buf, nchars); }

// ------------------------------------------------------------------ the rules
void test_rules() {
  CHECK(wgsi::instance_id_matches(kGpuId, wcslen(kGpuId), kReal));
  CHECK(wgsi::instance_id_matches(L"pci\\ven_1002&dev_7550&subsys_24161eae&rev_c0\\x", 46, kReal));  // any case
  CHECK(wgsi::instance_id_matches(L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0", 44, kReal));       // no suffix
  CHECK(!wgsi::instance_id_matches(L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C", 43, kReal));       // cut short
  CHECK(!wgsi::instance_id_matches(L"PCI\\VEN_1002&DEV_7551&SUBSYS_12345678&REV_C0\\x", 46, kReal));   // other device
  CHECK(!wgsi::instance_id_matches(L"PCI\\VEN_10DE&DEV_7550&SUBSYS_12345678&REV_C0\\x", 46, kReal));   // other vendor
  CHECK(!wgsi::instance_id_matches(L"PCI\\VEN_1002&DEV_7550&REV_C0\\x", 30, kReal));                   // no SUBSYS
  CHECK(!wgsi::instance_id_matches(L"PCI\\VEN_1002&DEV_7550&SUBSYS_2416XEAE&REV_C0", 44, kReal));      // not hex
  CHECK(!wgsi::instance_id_matches(L"HDAUDIO\\FUNC_01&VEN_1002&DEV_7550", 33, kReal));
  CHECK(!wgsi::instance_id_matches(nullptr, 0, kReal));
  CHECK(wgsi::entry_matches(L"PCI\\VEN_1002&DEV_7550", 21, kReal));
  CHECK(wgsi::entry_matches(L"PCI\\VEN_1002&DEV_7550&CC_0300", 29, kReal));
  CHECK(!wgsi::entry_matches(L"PCI\\VEN_1002&DEV_75501", 22, kReal));
  CHECK(!wgsi::entry_matches(L"PCI\\VEN_1002&CC_030000", 22, kReal));
  CHECK(!wgsi::entry_matches(L"PCI\\VEN_1002&DEV_755", 20, kReal));
  wchar_t w[] = L"pci\\ven_1002&dev_7550&SUBSYS_12345678&REV_C0";
  wgsi::write_pair(w, kSpoof);
  CHECK(wcscmp(w, L"pci\\ven_10DE&dev_2206&SUBSYS_12345678&REV_C0") == 0);  // digits only; same length

  // a multi-string with several entries: the matching ones rewritten, the others and the terminators kept
  std::wstring m = multi({L"PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0", L"PCI\\VEN_1002&CC_030000",
                          L"PCI\\VEN_1002&DEV_7550", L"PCI\\VEN_1002&DEV_13C0", L"PCI\\VEN_1002&DEV_7550&CC_0300"});
  std::wstring want = multi({L"PCI\\VEN_10DE&DEV_2206&SUBSYS_12345678&REV_C0", L"PCI\\VEN_1002&CC_030000",
                             L"PCI\\VEN_10DE&DEV_2206", L"PCI\\VEN_1002&DEV_13C0", L"PCI\\VEN_10DE&DEV_2206&CC_0300"});
  std::vector<std::pair<std::wstring, std::wstring>> ch;
  CHECK_EQ(wgsi::rewrite_multi_sz(&m[0], m.size(), kReal, kSpoof, &ch), 3u);
  CHECK(m == want);
  CHECK_EQ(ch.size(), (size_t)3);
  CHECK(ch.size() == 3 && ch[1].first == L"PCI\\VEN_1002&DEV_7550" && ch[1].second == L"PCI\\VEN_10DE&DEV_2206");
  // bounded: an entry cut by the buffer's end (no terminator) is read only up to the end, never past it
  wchar_t cut[30];
  wmemcpy(cut, L"PCI\\VEN_1002&DEV_7550&CC_0300XYZ", 30);
  CHECK_EQ(wgsi::rewrite_multi_sz(cut, 29, kReal, kSpoof, nullptr), 1u);
  CHECK(wmemcmp(cut, L"PCI\\VEN_10DE&DEV_2206&CC_0300X", 30) == 0);
  CHECK_EQ(wgsi::rewrite_multi_sz(cut, 20, kReal, kSpoof, nullptr), 0u);  // too short to hold the pair: untouched
  CHECK_EQ(wgsi::rewrite_multi_sz(nullptr, 10, kReal, kSpoof, nullptr), 0u);
}

// ------------------------------------------------------------------ SetupDiGetDeviceInstanceIdW
void test_instance_id() {
  make_devices();
  wgsi::Shim s;
  configure(s, true);
  g_log.clear();
  wchar_t buf[200];
  DWORD req = 0;
  SP_DEVINFO_DATA d = dev(0);
  SetLastError(0);
  BOOL ok = s.instance_id(kSet, &d, buf, 200, &req);
  CHECK(ok == TRUE);
  CHECK_EQ(GetLastError(), kSuccessError);  // the real call's last error, unchanged
  CHECK_EQ(req, (DWORD)wcslen(kGpuId) + 1);  // the real call's size
  CHECK(wcscmp(buf, kGpuIdSpoofed) == 0);    // VEN / DEV replaced; SUBSYS, REV and the instance suffix kept
  CHECK_EQ(g_log.size(), (size_t)1);
  CHECK(!g_log.empty() && g_log[0].find("SetupDiGetDeviceInstanceIdW PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0\\4&1A2B3C4D&0&00000008 -> "
                                        "PCI\\VEN_10DE&DEV_2206&SUBSYS_12345678&REV_C0\\4&1A2B3C4D&0&00000008") != std::string::npos);

  // buffer too small: the real call's FALSE, error and size; the buffer untouched; nothing logged
  wchar_t small[10];
  wmemset(small, L'#', 10);
  req = 0;
  ok = s.instance_id(kSet, &d, small, 10, &req);
  CHECK(ok == FALSE);
  CHECK_EQ(GetLastError(), (DWORD)ERROR_INSUFFICIENT_BUFFER);
  CHECK_EQ(req, (DWORD)wcslen(kGpuId) + 1);
  bool untouched = true;
  for (wchar_t c : small) untouched = untouched && c == L'#';
  CHECK(untouched);
  ok = s.instance_id(kSet, &d, nullptr, 0, &req);  // the size query
  CHECK(ok == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER && req == wcslen(kGpuId) + 1);
  // exactly large enough
  ok = s.instance_id(kSet, &d, buf, (DWORD)wcslen(kGpuId) + 1, nullptr);
  CHECK(ok == TRUE && wcscmp(buf, kGpuIdSpoofed) == 0);

  // another Display-class device (not the real adapter's pair): untouched
  d = dev(1);
  ok = s.instance_id(kSet, &d, buf, 200, &req);
  CHECK(ok == TRUE && wcscmp(buf, g_devices[1].id.c_str()) == 0);
  // another class with the same ID: untouched
  d = dev(2);
  ok = s.instance_id(kSet, &d, buf, 200, &req);
  CHECK(ok == TRUE && wcscmp(buf, kGpuId) == 0);
  // a Display-class device whose instance ID is not PCI: untouched
  d = dev(3);
  ok = s.instance_id(kSet, &d, buf, 200, &req);
  CHECK(ok == TRUE && wcscmp(buf, g_devices[3].id.c_str()) == 0);
  CHECK_EQ(g_log.size(), (size_t)2);  // the two rewrites of device 0 only
  CHECK_EQ(s.rewritten_calls(), 2u);

  // spoof inactive: no rewrite, nothing logged
  wgsi::Shim off;
  configure(off, false);
  g_log.clear();
  d = dev(0);
  ok = off.instance_id(kSet, &d, buf, 200, &req);
  CHECK(ok == TRUE && wcscmp(buf, kGpuId) == 0 && GetLastError() == kSuccessError);
  CHECK(g_log.empty());
}

// ------------------------------------------------------------------ SetupDiGetDeviceRegistryPropertyW
void test_registry_property() {
  make_devices();
  wgsi::Shim s;
  configure(s, true);
  g_log.clear();
  SP_DEVINFO_DATA d = dev(0);
  wchar_t buf[400];
  DWORD type = 0, req = 0;
  const std::wstring& hw = g_devices[0].props[SPDRP_HARDWAREID].second;
  BOOL ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && type == REG_MULTI_SZ && GetLastError() == kSuccessError);
  CHECK_EQ(req, (DWORD)(hw.size() * sizeof(wchar_t)));  // the real call's size
  CHECK(entries_of(buf, hw.size()) == multi({L"PCI\\VEN_10DE&DEV_2206&SUBSYS_12345678&REV_C0",
                                             L"PCI\\VEN_10DE&DEV_2206&SUBSYS_12345678", L"PCI\\VEN_10DE&DEV_2206&CC_030000",
                                             L"PCI\\VEN_10DE&DEV_2206&CC_0300"}));
  CHECK_EQ(g_log.size(), (size_t)1);  // one line for the call, every entry in it
  CHECK(!g_log.empty() && g_log[0].find("hardware IDs: PCI\\VEN_1002&DEV_7550&SUBSYS_12345678&REV_C0 -> PCI\\VEN_10DE&DEV_2206&SUBSYS_12345678&REV_C0; ") != std::string::npos &&
        g_log[0].find("PCI\\VEN_1002&DEV_7550&CC_0300 -> PCI\\VEN_10DE&DEV_2206&CC_0300") != std::string::npos);

  const std::wstring& compat = g_devices[0].props[SPDRP_COMPATIBLEIDS].second;
  ok = s.registry_property(kSet, &d, SPDRP_COMPATIBLEIDS, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && req == compat.size() * sizeof(wchar_t));
  CHECK(entries_of(buf, compat.size()) == multi({L"PCI\\VEN_10DE&DEV_2206&REV_C0", L"PCI\\VEN_10DE&DEV_2206",
                                                 L"PCI\\VEN_1002&CC_030000", L"PCI\\VEN_1002&CC_0300", L"PCI\\CC_030000",
                                                 L"PCI\\CC_0300"}));
  // no type pointer: the hardware IDs are still a multi-string
  ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, nullptr, reinterpret_cast<PBYTE>(buf), sizeof buf, nullptr);
  CHECK(ok == TRUE && wcsncmp(buf, L"PCI\\VEN_10DE&DEV_2206", 21) == 0);

  // the driver key and other properties: untouched (the game opens the real class key and reads the real DriverVersion)
  ok = s.registry_property(kSet, &d, SPDRP_DRIVER, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && type == REG_SZ && wcscmp(buf, kDriverKey) == 0);
  ok = s.registry_property(kSet, &d, SPDRP_DEVICEDESC, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && wcscmp(buf, L"PCI\\VEN_1002&DEV_7550 description") == 0);
  const size_t logged = g_log.size();
  CHECK_EQ(logged, (size_t)3);

  // buffer too small: FALSE, the real error and size, the buffer untouched
  BYTE small[16];
  memset(small, 0xCC, sizeof small);
  req = 0;
  ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, small, sizeof small, &req);
  CHECK(ok == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER && req == hw.size() * sizeof(wchar_t));
  bool untouched = true;
  for (BYTE b : small) untouched = untouched && b == 0xCC;
  CHECK(untouched);
  ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, nullptr, 0, &req);  // the size query
  CHECK(ok == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER && req == hw.size() * sizeof(wchar_t));
  // a property the device does not have: the real failure, unchanged
  ok = s.registry_property(kSet, &d, SPDRP_FRIENDLYNAME, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == FALSE && GetLastError() == ERROR_INVALID_DATA);

  // other Display-class device, other class, a non-PCI instance ID: untouched even where an entry names the real pair
  d = dev(1);
  ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && wcscmp(buf, L"PCI\\VEN_1002&DEV_13C0&SUBSYS_87654321&REV_C1") == 0);
  d = dev(2);
  ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && wcsncmp(buf, L"PCI\\VEN_1002&DEV_7550", 21) == 0);
  d = dev(3);
  ok = s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && wcsncmp(buf, L"PCI\\VEN_1002&DEV_7550", 21) == 0);
  CHECK_EQ(g_log.size(), logged);

  // spoof inactive: nothing rewritten
  wgsi::Shim off;
  configure(off, false);
  g_log.clear();
  d = dev(0);
  ok = off.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, &req);
  CHECK(ok == TRUE && entries_of(buf, hw.size()) == hw && GetLastError() == kSuccessError);
  CHECK(g_log.empty());

  // the log stops after kLogLimit rewritten calls, with one line saying so
  wgsi::Shim many;
  configure(many, true);
  g_log.clear();
  d = dev(0);
  for (unsigned i = 0; i < wgsi::Shim::kLogLimit + 20; ++i) many.instance_id(kSet, &d, buf, 200, nullptr);
  CHECK_EQ(g_log.size(), (size_t)wgsi::Shim::kLogLimit + 1);
  CHECK(!g_log.empty() && g_log.back().find("not logged any more") != std::string::npos);
  CHECK(wcscmp(buf, kGpuIdSpoofed) == 0);  // still rewritten
}

// ------------------------------------------------------------------ the start decision
void test_decide() {
  wgsi::GateInputs in;
  in.exe = "Spider-Man2.exe";
  in.dxgi_ok = in.kmt_ok = true;
  in.dxgi = kSpoof;
  in.kmt = kReal;
  wgsi::Decision d = wgsi::decide(in);
  CHECK(d.active && d.real == kReal && d.spoof == kSpoof);
  CHECK(d.why.find("active: DXGI reports 10de:2206 to the game, the kernel driver 1002:7550") == 0);
  in.exe = "SPIDER-MAN2.EXE";
  CHECK(wgsi::decide(in).active);
  in.setting = "AUTO";
  CHECK(wgsi::decide(in).active);
  in.setting = "maybe";
  d = wgsi::decide(in);
  CHECK(d.active && d.why.find("not recognised, using auto") != std::string::npos);
  in.setting = " off ";
  d = wgsi::decide(in);
  CHECK(!d.active && d.why.find("setupapi_identity=off") != std::string::npos);
  in.setting = "";
  in.exe = "Cyberpunk2077.exe";
  d = wgsi::decide(in);
  CHECK(!d.active && d.why.find("Spider-Man2.exe only") != std::string::npos);
  in.exe = "Spider-Man2.exe";
  in.dxgi = kReal;  // the host's spoof is off
  d = wgsi::decide(in);
  CHECK(!d.active && d.why.find("real pair 1002:7550") != std::string::npos);
  in.dxgi = PciPair{0x1002, 0x7551};  // same vendor, another device: a different pair
  CHECK(wgsi::decide(in).active);
  in.dxgi = kSpoof;
  in.kmt_ok = false;
  CHECK(!wgsi::decide(in).active);
  in.kmt_ok = true;
  in.dxgi_ok = false;
  CHECK(!wgsi::decide(in).active);
  CHECK(wgsi::setting_off("Off", nullptr) && !wgsi::setting_off("", nullptr) && !wgsi::setting_off("auto", nullptr));
  CHECK(wgsi::target_exe("spider-man2.exe") && !wgsi::target_exe("Spider-Man2Launcher.exe"));
}

// ------------------------------------------------------------------ the lazy decision
void test_lazy() {
  make_devices();
  wchar_t buf[200];
  SP_DEVINFO_DATA d = dev(0);
  auto log = [](const std::string& l) { g_log.push_back(l); };

  // DXGI not readable for the first two calls, then readable: those two pass through, the third is rewritten; the
  // decision is logged once and cached (the resolver is not run again)
  {
    wgsi::Shim s;
    int runs = 0;
    s.configure_lazy([&runs] {
                       wgsi::Resolution r;
                       if (++runs < 3) { r.d.why = "inactive: the adapter the game sees through DXGI could not be read"; return r; }
                       r.final = true;
                       r.d = active_decision();
                       r.d.why = "active: DXGI reports 10de:2206 to the game, the kernel driver 1002:7550; ...";
                       return r;
                     },
                     wgsi::Shim::kDecisionTries, &fake_instance_id, &fake_registry_property, log);
    g_log.clear();
    CHECK(!s.decided());
    CHECK(s.instance_id(kSet, &d, buf, 200, nullptr) == TRUE && wcscmp(buf, kGpuId) == 0 && GetLastError() == kSuccessError);
    DWORD type = 0;
    CHECK(s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, nullptr) == TRUE &&
          wcsncmp(buf, L"PCI\\VEN_1002&DEV_7550", 21) == 0);
    CHECK(!s.decided() && g_log.empty());
    CHECK(s.instance_id(kSet, &d, buf, 200, nullptr) == TRUE && wcscmp(buf, kGpuIdSpoofed) == 0);
    CHECK(s.decided() && s.active() && runs == 3 && s.tries() == 3u);
    CHECK_EQ(g_log.size(), (size_t)2);  // the decision line, then the rewrite
    CHECK(g_log.size() == 2 && g_log[0] == "setupapi identity: active: DXGI reports 10de:2206 to the game, the kernel driver 1002:7550; ..." &&
          g_log[1].find("SetupDiGetDeviceInstanceIdW") != std::string::npos);
    for (int i = 0; i < 5; ++i) s.instance_id(kSet, &d, buf, 200, nullptr);
    CHECK_EQ(runs, 3);  // cached
    CHECK(s.registry_property(kSet, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(buf), sizeof buf, nullptr) == TRUE &&
          wcsncmp(buf, L"PCI\\VEN_10DE&DEV_2206", 21) == 0);
  }
  // never readable: after the limit, inactive, logged once with the count; no rewrite, the resolver not run again
  {
    wgsi::Shim s;
    int runs = 0;
    s.configure_lazy([&runs] {
                       ++runs;
                       wgsi::Resolution r;
                       r.d.why = "inactive: the adapter the game sees through DXGI could not be read";
                       return r;
                     },
                     3, &fake_instance_id, &fake_registry_property, log);
    g_log.clear();
    for (int i = 0; i < 6; ++i) CHECK(s.instance_id(kSet, &d, buf, 200, nullptr) == TRUE && wcscmp(buf, kGpuId) == 0);
    CHECK(s.decided() && !s.active() && runs == 3);
    CHECK_EQ(g_log.size(), (size_t)1);
    CHECK(!g_log.empty() && g_log[0] == "setupapi identity: inactive: the adapter the game sees through DXGI could not be read (after 3 tries)");
  }
  // final inactive at the first call (DXGI reports the real pair): decided at once
  {
    wgsi::Shim s;
    int runs = 0;
    s.configure_lazy([&runs] {
                       ++runs;
                       wgsi::Resolution r;
                       r.final = true;
                       r.d.why = "inactive: DXGI reports the real pair 1002:7550 to the game (the host's GPU-name spoof is off)";
                       return r;
                     },
                     wgsi::Shim::kDecisionTries, &fake_instance_id, &fake_registry_property, log);
    g_log.clear();
    for (int i = 0; i < 3; ++i) s.instance_id(kSet, &d, buf, 200, nullptr);
    CHECK(s.decided() && !s.active() && runs == 1 && wcscmp(buf, kGpuId) == 0);
    CHECK_EQ(g_log.size(), (size_t)1);
  }
  // the buffer-too-small path while deciding: the real call's FALSE and error reach the caller
  {
    wgsi::Shim s;
    s.configure_lazy([] {
                       SetLastError(ERROR_FILE_NOT_FOUND);  // whatever the decision's own calls leave behind
                       wgsi::Resolution r;
                       r.final = true;
                       r.d = active_decision();
                       return r;
                     },
                     wgsi::Shim::kDecisionTries, &fake_instance_id, &fake_registry_property, log);
    wchar_t small[4] = {L'#', L'#', L'#', L'#'};
    DWORD req = 0;
    CHECK(s.instance_id(kSet, &d, small, 4, &req) == FALSE && GetLastError() == ERROR_INSUFFICIENT_BUFFER &&
          req == wcslen(kGpuId) + 1 && small[0] == L'#' && small[3] == L'#');
    CHECK(s.active());
  }
}

// ------------------------------------------------------------------ the host's settings (OptiScaler.ini) and resolve()
void test_host_ini() {
  wgsi::HostSpoof h = wgsi::parse_host_ini(
      "[Upscalers]\r\nSpoofedDeviceId=0x1111\r\n[Spoofing]\r\n; the GPU-name spoof\r\nDxgi=true\r\n"
      "SpoofedDeviceId=0x2206 ; a comment\r\n[Other]\r\nDxgi=false\r\nSpoofedDeviceId=0x3333\r\n");
  CHECK(h.dxgi_on && h.pair_ok && h.pair == kSpoof);  // vendor 10de, the device from [Spoofing] only
  h = wgsi::parse_host_ini("[spoofing]\nDXGI = TRUE\nspoofeddeviceid = 8710\n");
  CHECK(h.dxgi_on && h.pair_ok && h.pair == (PciPair{0x10de, 8710}));
  h = wgsi::parse_host_ini("[Spoofing]\nDxgi=auto\nSpoofedDeviceId=0x2206\n");
  CHECK(!h.dxgi_on && h.pair_ok);
  h = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\nSpoofedDeviceId=auto\n");
  CHECK(h.dxgi_on && !h.pair_ok && h.note == "SpoofedDeviceId=auto is not a device ID" && h.pair == PciPair());
  h = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\n");
  CHECK(h.dxgi_on && !h.pair_ok && h.note == "no SpoofedDeviceId");
  h = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\nSpoofedDeviceId=0x12345\n");  // > 16 bits
  CHECK(!h.pair_ok);
  h = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\nSpoofedDeviceId=22x6\n");  // not a number
  CHECK(!h.pair_ok);
  h = wgsi::parse_host_ini("");
  CHECK(!h.dxgi_on && !h.pair_ok);

  wgsi::ResolveInputs in;
  in.exe = "Spider-Man2.exe";
  in.kmt_ok = true;
  in.kmt = kReal;
  in.host = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\nSpoofedDeviceId=0x2206\n");
  // DXGI readable: preferred over the host's settings
  in.dxgi_ok = true;
  in.dxgi = PciPair{0x10de, 0x2684};
  wgsi::Resolution r = wgsi::resolve(in);
  CHECK(r.final && r.d.active && r.d.spoof == (PciPair{0x10de, 0x2684}) && r.d.real == kReal);
  CHECK(r.d.why.find("active: DXGI reports 10de:2684 to the game") == 0);
  in.dxgi = kReal;  // DXGI readable and real: inactive even though the ini says otherwise
  r = wgsi::resolve(in);
  CHECK(r.final && !r.d.active);
  // DXGI not readable: the host's settings (Dxgi=true and the pair)
  in.dxgi_ok = false;
  r = wgsi::resolve(in);
  CHECK(r.final && r.d.active && r.d.spoof == kSpoof && r.d.real == kReal);
  CHECK(r.d.why.find("active: the host's settings report 10de:2206 ([Spoofing] Dxgi=true; DXGI is not readable yet), the kernel "
                     "driver 1002:7550; for Display-class devices") == 0);
  in.kmt_ok = false;  // no kernel pair: try again later
  r = wgsi::resolve(in);
  CHECK(!r.final && !r.d.active);
  in.kmt_ok = true;
  in.host.dxgi_on = false;  // the host's spoof is off in its settings: no fallback
  r = wgsi::resolve(in);
  CHECK(!r.final && r.d.why.find("not on in its settings") != std::string::npos);
  in.host = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\nSpoofedDeviceId=auto\n");
  r = wgsi::resolve(in);
  CHECK(!r.final && r.d.why.find("the host's settings give no spoofed pair (SpoofedDeviceId=auto is not a device ID") != std::string::npos);
  in.host = wgsi::parse_host_ini("[Spoofing]\nDxgi=true\nSpoofedDeviceId=0x2206\n");
  in.kmt = PciPair{0x10de, 0x2206};  // the host's pair is the real one
  r = wgsi::resolve(in);
  CHECK(r.final && !r.d.active);
  // other process / switched off: final inactive whatever DXGI and the ini say
  in.kmt = kReal;
  in.exe = "Other.exe";
  CHECK(wgsi::resolve(in).final && !wgsi::resolve(in).d.active);
  in.exe = "Spider-Man2.exe";
  in.setting = "off";
  CHECK(wgsi::resolve(in).final && !wgsi::resolve(in).d.active);
}

// ------------------------------------------------------------------ the import-slot redirect (this exe)
bool parse_pair(const wchar_t* id, PciPair* p) {
  unsigned v = 0, d = 0;
  if (_wcsnicmp(id, L"PCI\\VEN_", 8) != 0 || swscanf_s(id + 8, L"%4x&DEV_%4x", &v, &d) != 2) return false;
  p->ven = (uint16_t)v;
  p->dev = (uint16_t)d;
  return true;
}

void test_import_slots() {
  const std::vector<wgsi::ImportSlot> slots = wgsi::find_import_slots(
      GetModuleHandleW(nullptr), "SETUPAPI.dll", {"SetupDiGetDeviceInstanceIdW", "SetupDiGetDeviceRegistryPropertyW"});
  CHECK_EQ(slots.size(), (size_t)2);  // this exe imports both (below), by name, from the import table
  for (auto& sl : slots) CHECK(!sl.delay && sl.original != nullptr);

  std::string detail;
  // the PC's Display-class devices, read before the redirect (through the functions themselves, not the slots)
  HMODULE sa = GetModuleHandleW(L"setupapi.dll");
  auto real_id = reinterpret_cast<wgsi::InstanceIdFn>(GetProcAddress(sa, "SetupDiGetDeviceInstanceIdW"));
  auto real_prop = reinterpret_cast<wgsi::RegistryPropertyFn>(GetProcAddress(sa, "SetupDiGetDeviceRegistryPropertyW"));
  HDEVINFO set = SetupDiGetClassDevsW(&kDisplay, nullptr, nullptr, DIGCF_PRESENT);
  CHECK(set != INVALID_HANDLE_VALUE);
  if (set == INVALID_HANDLE_VALUE) return;
  SP_DEVINFO_DATA d = {};
  d.cbSize = sizeof d;
  wchar_t id[MAX_DEVICE_ID_LEN] = {};
  PciPair real;
  bool found = false;
  for (DWORD i = 0; !found && SetupDiEnumDeviceInfo(set, i, &d); ++i)
    found = real_id(set, &d, id, MAX_DEVICE_ID_LEN, nullptr) && parse_pair(id, &real) &&
            wgsi::instance_id_matches(id, wcslen(id), real);
  if (!found) {
    printf("import slots: no PCI Display-class device on this PC; the live part is skipped\n");
    SetupDiDestroyDeviceInfoList(set);
    return;
  }
  wchar_t driver_before[200] = {};
  DWORD type = 0;
  const BOOL had_driver = real_prop(set, &d, SPDRP_DRIVER, &type, reinterpret_cast<PBYTE>(driver_before), sizeof driver_before, nullptr);
  wchar_t hw_before[1024] = {};
  DWORD hw_bytes = 0;
  CHECK(real_prop(set, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(hw_before), sizeof hw_before, &hw_bytes));

  // redirect with a synthetic spoofed pair (no other process, no game; this exe's own slots)
  wgsi::Decision dec;
  dec.active = true;
  dec.real = real;
  dec.spoof = PciPair{real.ven == 0x10de ? (uint16_t)0x1002 : (uint16_t)0x10de, 0x2206};
  g_log.clear();
  dec.why = "active: synthetic";
  CHECK_EQ(wgsi::install(GetModuleHandleW(nullptr), [dec] { wgsi::Resolution r; r.final = true; r.d = dec; return r; },
                         wgsi::Shim::kDecisionTries, [](const std::string& l) { g_log.push_back(l); }, &detail), 2u);
  CHECK(!wgsi::process_shim().decided());  // deferred to the first call
  CHECK(detail == "SetupDiGetDeviceInstanceIdW (import), SetupDiGetDeviceRegistryPropertyW (import)");

  // the calls through this exe's imports now meet the shim
  wchar_t after[MAX_DEVICE_ID_LEN] = {};
  DWORD req = 0;
  CHECK(SetupDiGetDeviceInstanceIdW(set, &d, after, MAX_DEVICE_ID_LEN, &req));
  CHECK_EQ(req, (DWORD)wcslen(id) + 1);
  CHECK(wcslen(after) == wcslen(id) && wcscmp(after + 21, id + 21) == 0);  // SUBSYS, REV, suffix kept
  PciPair seen;
  CHECK(parse_pair(after, &seen) && seen == dec.spoof);
  wchar_t hw_after[1024] = {};
  DWORD hw_bytes_after = 0;
  CHECK(SetupDiGetDeviceRegistryPropertyW(set, &d, SPDRP_HARDWAREID, &type, reinterpret_cast<PBYTE>(hw_after), sizeof hw_after,
                                          &hw_bytes_after));
  CHECK_EQ(hw_bytes_after, hw_bytes);
  CHECK(parse_pair(hw_after, &seen) && seen == dec.spoof);
  CHECK(wcscmp(hw_after + 21, hw_before + 21) == 0);
  if (had_driver) {  // the driver key: as before
    wchar_t driver_after[200] = {};
    CHECK(SetupDiGetDeviceRegistryPropertyW(set, &d, SPDRP_DRIVER, &type, reinterpret_cast<PBYTE>(driver_after), sizeof driver_after,
                                            nullptr));
    CHECK(wcscmp(driver_after, driver_before) == 0);
  }
  CHECK_EQ(g_log.size(), (size_t)3);  // the decision line at the first call, then one line per rewritten call
  CHECK(!g_log.empty() && g_log[0] == "setupapi identity: active: synthetic");
  printf("import slots: live check on %s\n", g_log.size() < 2 ? "?" : g_log[1].c_str());
  SetupDiDestroyDeviceInfoList(set);
  // put the slots back (the plugin never does; this exe runs nothing after)
  for (auto& sl : slots) {
    DWORD old = 0;
    if (VirtualProtect(sl.slot, sizeof(void*), PAGE_READWRITE, &old)) {
      *sl.slot = sl.original;
      VirtualProtect(sl.slot, sizeof(void*), old, &old);
    }
  }
}

}  // namespace

int main() {
  test_rules();
  test_instance_id();
  test_registry_property();
  test_decide();
  test_lazy();
  test_host_ini();
  test_import_slots();
  TEST_SUMMARY("setupapi identity");
}
