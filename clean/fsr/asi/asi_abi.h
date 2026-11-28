// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// The interface WrongGPU's front end (amd_fidelityfx_dx12.dll) uses to ask the WrongGPU ASI plugin which upscaler
// entry the player selected in OptiScaler's menu. The front end finds the plugin by module (GetModuleHandle of the
// plugin file name) and resolves this one export by name, so load order does not matter and neither DLL links the
// other. Defined once, shared by the plugin (which exports it) and the front-end bridge (which calls it).
#include <stddef.h>
#include <stdint.h>

#define WRONGGPU_ASI_MODULE  L"WrongGPU.asi"
#define WRONGGPU_ASI_QUERY   "WrongGpuAsiQuery"
#define WRONGGPU_ASI_ABI     1u

// Query result. `hooked` is 1 after the prepared menu call-patch transaction commits; it does not promise
// complete menu adaptation. The FFX list still contains WrongGPU and AMD entries. Zero means inactive: the
// front end then behaves as if no plugin were present. `model` is 0 (OptiScaler's own FSR 3.X / AMD), 'M' or 'K'
// (a saved Model L reads as 'M' here, for an older front end; which WrongGPU model runs is the model choice
// record's, src/model_choice_record.h). Returns the ABI version (>= 1), or 0 on error. Signature: int WrongGpuAsiQuery(int* hooked, char* model).
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
// Selection is 0 (AMD), 'M', 'K' or 'L' (the saved choice); it is NOT evidence of the effective rendering backend.
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

// The front end tells the plugin where the game's model choice record is (wronggpu-setup\model-choice.ini next to
// its wronggpu.ini; src/model_choice_record.h). A click on one of the plugin's model entries is written there (seq + 1,
// source=plugin). Without this call the plugin uses wronggpu-setup\model-choice.ini in its own data folder.
#define WRONGGPU_ASI_SET_CHOICE_RECORD "WrongGpuAsiSetChoiceRecordV1"
typedef void (*PfnWrongGpuAsiSetChoiceRecordV1)(const wchar_t* path);

// The front end reports which model it runs (0 AMD, 'M', 'K', 'L'), so the plugin's menu marks that entry; 1 returns
// to marking the saved choice. Display only: nothing is saved.
#define WRONGGPU_ASI_SHOW_MODEL "WrongGpuAsiShowModelV1"
typedef void (*PfnWrongGpuAsiShowModelV1)(char model);

// The front end reports what it actually runs (rc3: the card's status lines, sr_card.cpp), so the card never names a model
// from the saved choice as running: the model and mode that upscaled the last frame, the model a switch is starting (the
// running one keeps upscaling meanwhile), and the last model that could not start, with the output size and the reason.
// Display only: nothing is saved. Sent on changes (create, switch, failure, destroy, the first frame of another model).
#define WRONGGPU_ASI_REPORT_SR "WrongGpuAsiReportSrV1"
#define WG_ASI_SR_IDLE 0u       // WrongGPU upscales nothing (no context of its own, or the context runs AMD's FSR)
#define WG_ASI_SR_RUNNING 1u    // running_model upscaled the last frame
#define WG_ASI_SR_SWITCHING 2u  // running_model upscales while pending_model starts
typedef struct WrongGpuAsiSrV1 {
    uint32_t size;          // sizeof(WrongGpuAsiSrV1); a smaller report is ignored
    uint32_t state;         // WG_ASI_SR_*
    char running_model;     // 'M', 'K', 'L'; 0 while idle
    char pending_model;     // while switching: the model that is starting
    char error_model;       // the last model that could not start (0: none since the last model that started)
    char reserved;
    int32_t running_mode;   // the running model's mode: 0 fast, 1 exact (wg_sr_modes.h), -1 not known
    uint32_t error_width;   // the output size it could not start at (0: not known)
    uint32_t error_height;
    char error_reason[160]; // NUL-terminated, short
} WrongGpuAsiSrV1;
typedef void (*PfnWrongGpuAsiReportSrV1)(const WrongGpuAsiSrV1* status);
