// SPDX-License-Identifier: MIT
#pragma once
// The interface WrongGPU's front end (amd_fidelityfx_dx12.dll) uses to ask the WrongGPU ASI plugin which upscaler
// entry the player selected in OptiScaler's menu. The front end finds the plugin by module (GetModuleHandle of the
// plugin file name) and resolves this one export by name, so load order does not matter and neither DLL links the
// other. Defined once, shared by the plugin (which exports it) and the front-end bridge (which calls it).
#include <stdint.h>

#define WRONGGPU_ASI_MODULE  L"WrongGPU.asi"
#define WRONGGPU_ASI_QUERY   "WrongGpuAsiQuery"
#define WRONGGPU_ASI_ABI     1u

// Query result. `hooked` is 1 after the prepared menu call-patch transaction commits; it does not promise
// complete menu adaptation. The FFX list still contains WrongGPU and AMD entries. Zero means inactive: the
// front end then behaves as if no plugin were present. `model` is 0 (OptiScaler's own FSR 3.X / AMD), 'M' or 'K'.
// Returns the ABI version (>= 1), or 0 on error. Signature: int WrongGpuAsiQuery(int* hooked, char* model).
typedef int (*PfnWrongGpuAsiQuery)(int* hooked, char* model);

// Additive WrongGPU-owned diagnostic API; this is NOT a host plugin or RR frame ABI.
#define WRONGGPU_ASI_STATUS_QUERY "WrongGpuAsiQueryStatusV1"
#define WRONGGPU_ASI_STATUS_VERSION 1u
#define WG_ASI_STATUS_OK 1
#define WG_ASI_STATUS_INVALID_ARGUMENT -1
#define WG_ASI_STATUS_INVALID_SIZE -2
#define WG_ASI_STATUS_UNSUPPORTED_VERSION -3

#define WG_ASI_INIT_NOT_ATTEMPTED 0u
#define WG_ASI_INIT_IN_PROGRESS 1u
#define WG_ASI_INIT_COMPLETED 2u
#define WG_ASI_MENU_UNAVAILABLE 0u
#define WG_ASI_MENU_PARTIAL 1u
#define WG_ASI_MENU_PATCHED 2u
#define WG_ASI_REASON_NOT_INITIALIZED 1u
#define WG_ASI_REASON_INITIALIZING 2u
#define WG_ASI_REASON_HOST_NOT_RECOGNIZED 3u
#define WG_ASI_REASON_STUB_ALLOCATION_FAILED 4u
#define WG_ASI_REASON_PATCH_FAILED 5u
#define WG_ASI_REASON_INCOMPLETE_PATCHES 6u
#define WG_ASI_REASON_ROLLBACK_FAILED 7u
#define WG_ASI_REASON_MODULE_PIN_FAILED 8u
#define WG_ASI_REASON_NONE 0u
#define WG_ASI_EFFECTIVE_BACKEND_UNKNOWN 0u
#define WG_ASI_RR_MISSING_INTERFACE 1u
#define WG_ASI_RR_MISSING_F_CONTRACTS 2u
#define WG_ASI_RR_BACKEND_UNAVAILABLE 4u
#define WG_ASI_RR_HOST_UNQUALIFIED 8u

// All fields are fixed-width, with no pointers or platform-specific types. Initialize size and version;
// pass the actual writable allocation size separately. Both sizes must cover this structure, and size
// must not exceed caller_size. A larger caller buffer is accepted; only this structure is written.
// Failure leaves every caller byte unchanged. Null is refused without dereferencing it. Callers must
// provide a valid aligned writable buffer for caller_size bytes; arbitrary pointers cannot be validated.
// Loaded means the export is callable. Completed means the hook attempt returned, even if it failed.
// Menu PATCHED describes successful own patch writes, not visible menu/host qualification.
// ROLLBACK_FAILED reports possibly partial bytes/protection/cache state while hooked remains zero.
// Before any host write, the plugin pins its own module until process termination. Pin failure
// leaves host call sites untouched. Query-only loads do not pin. Neither query promises host
// lifetime, executing-thread atomicity, or safe live unhook/unload.
// Selection is 0 (AMD), 'M', or 'K'; it is NOT evidence of the effective rendering backend.
// Fields are independent observations during initialization/selection changes, not a transactional snapshot.
// rr_ready is always zero in this implementation; rr_blockers is a bitmask of the reasons above.
typedef struct WrongGpuAsiStatusV1 {
    uint32_t size;
    uint32_t version;
    uint32_t loaded;
    uint32_t initialization;
    uint32_t hooked;
    uint32_t menu_state;
    uint32_t menu_reason;
    uint32_t selected_model;
    uint32_t effective_backend;
    uint32_t rr_ready;
    uint32_t rr_blockers;
    uint32_t reserved;
} WrongGpuAsiStatusV1;

typedef int (*PfnWrongGpuAsiQueryStatusV1)(WrongGpuAsiStatusV1* status, uint32_t caller_size);
