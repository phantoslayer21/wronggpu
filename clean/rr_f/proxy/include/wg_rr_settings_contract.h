// SPDX-License-Identifier: MIT
/* RR settings contract between the proxy and WrongGPU.asi (spec 60.9), contract version 1, x64.
 * Owned by WrongGPU. The proxy exports WG_RR_CONTRACT_EXPORT (C linkage, no arguments); the table it returns
 * stays valid until the proxy unloads. Shared by clean/rr_f/proxy (producer) and the ASI side (consumer,
 * clean/rr_f/proxy/asi/wg_rr_controls.h).
 */
#ifndef WG_RR_SETTINGS_CONTRACT_H
#define WG_RR_SETTINGS_CONTRACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WG_RR_CONTRACT_EXPORT "WrongGPU_GetSettingsContract"
#define WG_RR_CONTRACT_VERSION 1u
#define WG_RR_CONTRACT_V1_SIZE 40u

#define WG_RR_SETTING_KIND_ONOFF 0u
#define WG_RR_SETTING_KIND_CHOICE 1u
#define WG_RR_SETTING_FLAG_AT_CREATION 1u  /* takes effect when the game next creates the RR feature */
#define WG_RR_SETTING_FLAG_OPT_IN 2u       /* opt-in variation that changes output versus the original */

#define WG_RR_STATUS_RUNTIME_READY 1u
#define WG_RR_STATUS_MODEL_LOADED 2u
#define WG_RR_STATUS_ACTIVE 4u              /* most recent RR evaluate completed, less than one second ago */

typedef struct WgRrSettingEntry {
  const char* key;          /* +0 UTF-8, stable across contract versions */
  const char* label;        /* +8 UTF-8 */
  uint32_t kind;            /* +16 */
  uint32_t flags;           /* +20 */
  int32_t default_value;    /* +24 */
  uint32_t max_value;       /* +28 1 for kind 0 */
  int32_t (*get)(void);     /* +32 */
  uint32_t (*set)(int32_t value); /* +40 0 ok, 25 outside 0..max (unchanged) */
} WgRrSettingEntry;

typedef struct WgRrSettingsContract {
  uint32_t size;            /* +0 40 */
  uint32_t version;         /* +4 1 */
  uint32_t count;           /* +8 */
  uint32_t reserved;        /* +12 0 */
  const WgRrSettingEntry* entries; /* +16 */
  uint32_t (*status)(void); /* +24 */
  void* reserved_ptr;       /* +32 null */
} WgRrSettingsContract;

typedef const WgRrSettingsContract* (*WgRrGetSettingsContractFn)(void);

#ifdef __cplusplus
}
static_assert(sizeof(WgRrSettingEntry) == 48 && offsetof(WgRrSettingEntry, kind) == 16 &&
              offsetof(WgRrSettingEntry, default_value) == 24 && offsetof(WgRrSettingEntry, get) == 32 &&
              offsetof(WgRrSettingEntry, set) == 40, "60.9 setting entry");
static_assert(sizeof(WgRrSettingsContract) == 40 && offsetof(WgRrSettingsContract, entries) == 16 &&
              offsetof(WgRrSettingsContract, status) == 24 && offsetof(WgRrSettingsContract, reserved_ptr) == 32,
              "60.9 contract table");
#endif

#endif
