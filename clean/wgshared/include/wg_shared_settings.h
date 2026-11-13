// SPDX-License-Identifier: MIT
/* WrongGPU shared settings block, version 1 (i15). Owned by WrongGPU; clean/wgshared/include/wg_shared_settings.h.
 *
 * One process-local block that every WrongGPU DLL in a game process (SR front end, RR plugin, RR runtime, SR runtime)
 * can open without depending on another DLL's internals: a named file mapping of WG_SS_SIZE bytes, a fixed header, then
 * one int32 slot per registry option (slot index = the option's registry id, wg_registry_gen.h), then a text area for
 * the few string options. Pure C, x64, no padding surprises (every field is 4 or 8 bytes aligned by position).
 *
 * Lifetime: whichever DLL opens it first creates it (CreateFileMappingW, Local namespace, name WG_SS_NAME_FMT with
 * the process id), seeds every slot from wronggpu.ini (the one settings file, <exe dir>\WrongGPU\wronggpu.ini) and
 * publishes it by writing `magic` last (release barrier). Openers wait for `magic` (bounded: 50 x 1 ms), else treat
 * the block as absent and fall back to their own ini / environment reading. The block lives until the process ends.
 *
 * Writer rules (single writer): writes of user choices come from ONE module, elected by an atomic compare-exchange
 * of `writer` from WG_SS_WRITER_NONE. The SR front end claims it when it is hosted in OptiScaler's custom slot (it
 * is the menu's voice); the RR plugin claims it only when it is alone (no front end claimed within 2 s of the first
 * RR call) and then only writes what its own ini / OptiScaler.ini polling gives. A module that did not win never
 * writes a slot, with one exception (slot owners, i20): the RR plugin's own menu panel owns rr_mode (owner 2026-10-31:
 * RR mode is a box of its own, not an entry of the host's version list) and writes that slot whether or not it won;
 * no other module writes rr_mode. Every writer enters the seqlock by compare-exchange of `generation` from an even
 * value to the next odd one, so two writers never overlap. The creator's seeding happens before publication, so it
 * is not a write in this sense.
 *
 * Seqlock: a writer increments `generation` to an odd value (acquire), writes the slot(s) and the set_mask bit, then
 * increments it to the next even value (release). A reader reads `generation` (even, else retry), copies the slots it
 * needs, re-reads `generation`; if it changed, retry (bounded: 16 tries, then keep the last good snapshot). Readers
 * snapshot once per frame / evaluate (cheap: 64 ints), never per dispatch.
 *
 * Versioning: `version` is this header's WG_SS_VERSION; `size` is the mapping size; `slot_count` the slots the creator
 * wrote. A reader built against a newer header with more slots treats slots >= slot_count as unset (its own default).
 * A reader built against an older header ignores slots it does not know. A different `version` or a `size` smaller
 * than WG_SS_HEADER_SIZE + 4 * slot_count: the reader logs once and treats the block as absent. `registry_hash` is the
 * FNV-1a-32 of the generated registry (wg_registry_gen.h WG_REGISTRY_HASH); a mismatch is logged once, values are
 * still used (slot ids are stable: the registry only ever appends).
 */
#ifndef WG_SHARED_SETTINGS_H
#define WG_SHARED_SETTINGS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WG_SS_NAME_FMT "Local\\WrongGPU.settings.v1.%lu" /* %lu = GetCurrentProcessId() */
#define WG_SS_MAGIC 0x54534757u                           /* 'WGST' little-endian */
#define WG_SS_VERSION 1u
#define WG_SS_SIZE 4096u
#define WG_SS_SLOT_COUNT 64u
#define WG_SS_TEXT_SIZE 2048u
#define WG_SS_HEADER_SIZE 64u

/* writer ids */
#define WG_SS_WRITER_NONE 0u
#define WG_SS_WRITER_SR_FRONTEND 1u
#define WG_SS_WRITER_RR_PLUGIN 2u
#define WG_SS_WRITER_TEST 0xFFu

/* header flags */
#define WG_SS_FLAG_HOSTED_BY_OPTISCALER 1u /* the SR front end runs in OptiScaler's custom slot */
#define WG_SS_FLAG_FOLDER_ENTRY 2u         /* one-folder-per-choice fallback is in use (entry from the folder sidecar) */

typedef struct WgSharedSettingsHeader {
  uint32_t magic;         /* +0  WG_SS_MAGIC, written last by the creator */
  uint32_t version;       /* +4  WG_SS_VERSION */
  uint32_t size;          /* +8  WG_SS_SIZE */
  uint32_t slot_count;    /* +12 slots the creator seeded (WG_SS_SLOT_COUNT) */
  uint32_t generation;    /* +16 seqlock: odd = write in progress */
  uint32_t writer;        /* +20 WG_SS_WRITER_* */
  uint32_t writer_pid;    /* +24 */
  uint32_t registry_hash; /* +28 WG_REGISTRY_HASH of the creator */
  uint32_t flags;         /* +32 WG_SS_FLAG_* */
  uint32_t creator;       /* +36 WG_SS_WRITER_* id of the module that created and seeded the block */
  uint32_t seed_source;   /* +40 1 ini, 2 defaults (no ini), 3 test */
  uint32_t reserved[5];   /* +44..63 zero */
} WgSharedSettingsHeader;

typedef struct WgSharedSettingsBlock {
  WgSharedSettingsHeader header;          /* +0, 64 bytes */
  int32_t values[WG_SS_SLOT_COUNT];       /* +64  one int32 per registry option: choice index / 0-1 / integer */
  uint32_t set_mask[2];                   /* +320 bit i: slot i was seeded or written (else the reader's default applies) */
  uint32_t live_mask[2];                  /* +328 bit i: slot i changed since the writer was elected (diagnostics) */
  uint32_t reserved2[6];                  /* +336..359 zero */
  uint32_t text_used;                     /* +360 bytes used in text[] */
  uint32_t reserved3;                     /* +364 */
  char text[WG_SS_TEXT_SIZE];             /* +368 NUL-separated "key=value" strings for text options (rr_model, sr_runtime_dll...) */
  /* the rest of the 4096 bytes is zero */
} WgSharedSettingsBlock;

typedef const WgSharedSettingsBlock* WgSharedSettingsView;

#ifdef __cplusplus
}
static_assert(sizeof(WgSharedSettingsHeader) == WG_SS_HEADER_SIZE, "shared settings header");
static_assert(offsetof(WgSharedSettingsBlock, values) == 64, "shared settings values");
static_assert(offsetof(WgSharedSettingsBlock, set_mask) == 320, "shared settings set_mask");
static_assert(offsetof(WgSharedSettingsBlock, text) == 368, "shared settings text");
static_assert(sizeof(WgSharedSettingsBlock) <= WG_SS_SIZE, "shared settings block fits the mapping");
#endif

#endif
