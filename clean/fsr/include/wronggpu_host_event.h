/* SPDX-License-Identifier: MIT */
#pragma once
/* The event a host (the WrongGPU edition of OptiScaler) sends WrongGPU's front end (amd_fidelityfx_dx12.dll in the
 * custom WrongGPU slot or next to the game) when the PLAYER selects one of the listed upscaler versions in its menu during
 * the session. The host resolves this one export by name in the front-end DLL it loaded (GetProcAddress) and calls it
 * with the selected entry's version id (from the list the front end returned to FFX_API_QUERY_DESC_TYPE_GET_VERSIONS),
 * BEFORE it recreates the upscaler, so the new context runs the selected model. Only a player's selection is an event: a
 * host never sends one for an index it read from its ini or applied at start.
 *
 * A selection of one of WrongGPU's model entries is written to the game's model choice record
 * (clean/fsr/src/model_choice_record.h, source=ffx) and decides the model from then on, until a newer choice (the
 * launcher's, the WrongGPU menu plugin's, or another selection). An AMD version or a Ray Reconstruction mode entry is no
 * model choice.
 *
 * Signature: int32_t WrongGpuHostSelectedV1(uint64_t versionId). Callable from any thread; it takes the record's
 * cross-process lock for a few milliseconds at most (3 s when another process holds it). */
#include <stdint.h>

#define WRONGGPU_HOST_SELECTED_EXPORT "WrongGpuHostSelectedV1"

#define WG_HOST_SELECTED_RECORDED 1      /* a WrongGPU model: written to the record */
#define WG_HOST_SELECTED_SESSION_ONLY 2  /* a WrongGPU model, no record for this game (no launcher state folder): this session */
#define WG_HOST_SELECTED_NOT_A_MODEL 0   /* an AMD version or an RR mode entry: nothing recorded */
#define WG_HOST_SELECTED_PENDING -1      /* not written yet (the record is busy or a write failed): used this session and
                                            written at the next create; the front end's log says why. Hosts show it. */
#define WG_HOST_SELECTED_REFUSED -2      /* the record's sequence number is at its end: refused (logged). Hosts show it. */
#define WG_HOST_SELECTED_NOT_READY -3    /* the front end is not hosted or not initialised */

typedef int32_t (*PfnWrongGpuHostSelectedV1)(uint64_t versionId);
