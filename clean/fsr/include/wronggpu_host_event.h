/* SPDX-License-Identifier: GPL-3.0-only */
#pragma once
/* The event a host (the WrongGPU edition of OptiScaler) sends WrongGPU's front end (amd_fidelityfx_dx12.dll in the
 * custom WrongGPU slot or next to the game) when the PLAYER selects one of the listed upscaler versions in its menu. The
 * host resolves this one export by name in the front-end DLL it loaded (GetProcAddress) and calls it with the selected
 * entry's version id (from the list the front end returned to FFX_API_QUERY_DESC_TYPE_GET_VERSIONS), BEFORE it recreates
 * the upscaler. Only a player's selection is an event: a host never sends one for an index it read from its ini or applied
 * at start. The export is an in-process call with no caller check: any code in the game's process could call it, as it
 * could write the record itself; the record only ever holds a model choice, and the next create applies it.
 *
 * A selection of one of WrongGPU's model entries is written to the game's model choice record
 * (clean/fsr/src/model_choice_record.h, source=ffx) and decides the model from then on, until a newer choice (the
 * launcher's, the WrongGPU menu plugin's, or another selection). An AMD version or a Ray Reconstruction mode entry is no
 * model choice. Re-selecting the entry that is already selected is a choice too (it wins over a newer choice made
 * elsewhere).
 *
 * Signature: int32_t WrongGpuHostSelectedV1(uint64_t versionId). Callable from any thread; it takes the game's state lock
 * for a few milliseconds (3 s at most when another WrongGPU program holds it). Nothing is ever kept for later: a choice
 * that is not saved changes nothing. */
#include <stdint.h>

#define WRONGGPU_HOST_SELECTED_EXPORT "WrongGpuHostSelectedV1"

#define WG_HOST_SELECTED_RECORDED 1         /* recorded, and this model runs from now on instead of another: recreate the
                                               upscaler (also when the entry was already selected) */
#define WG_HOST_SELECTED_NOTHING_CHANGED 0  /* nothing changes in what runs: an AMD version or an RR mode entry, a model
                                               that already ran (recorded again), or a game without a record (no launcher
                                               state folder: the host's entry decides, as before) */
#define WG_HOST_SELECTED_NOT_SAVED -1       /* not saved (the record busy or unreadable, or a failed write): nothing changes
                                               and nothing is kept; the host says "not saved, try again" and does not
                                               recreate the upscaler */
#define WG_HOST_SELECTED_REFUSED -2         /* the record's sequence number is at its end: refused (logged); as NOT_SAVED */
#define WG_HOST_SELECTED_NOT_READY -3       /* the front end is not hosted or not initialised; as NOT_SAVED */

typedef int32_t (*PfnWrongGpuHostSelectedV1)(uint64_t versionId);
