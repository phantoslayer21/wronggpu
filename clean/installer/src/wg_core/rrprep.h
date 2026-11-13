// SPDX-License-Identifier: MIT
// WrongGPU launcher core: Ray Reconstruction's kernels prepared by the launcher (release 0.6.0, owner 2026-11-02: "It HAS to
// pre compile inside the launcher"). Written for this project.
//
// The first RR start in a game makes the graphics driver compile RR's kernels (about 40 s for the default mode, minutes
// for all modes, with a cold driver cache). Measured on the RDNA 4 driver (release-060 FINAL-PROGRESS.txt): its shader
// cache is kept per program FILE NAME (the same name in another folder, with other bytes, finds it warm: 47.6 s cold,
// 67 ms warm), and neither a D3D12 pipeline library nor the pipelines' cached blobs hold the compiled code (only
// references into that cache), so a cache of WrongGPU's own cannot carry the kernels into a game. The launcher therefore
// runs itself as a helper under the game's program name, in a folder of its data folder (rr-prepare\run\<game exe name>),
// which loads the game's wronggpu_runtime.dll and creates RR's pipelines (WrongGPU_RrPrepareKernelsV1): the driver then
// has them for the game's first start. Progress per kernel; cancelling stops the helper (what was compiled stays in the
// driver's cache: the next run goes on from there).
//
// "Prepared" is kept per game program name in rr-prepare\prepared.txt with the driver's version and the runtime's
// SHA-256: a driver update or another WrongGPU runtime makes it "not prepared" again (prepared again after the next
// install / update / repair, at Play, or with the game page's Prepare). A driver that drops its cache on its own (or a
// "reset shader cache") is not seen: the in-game "WrongGPU RR (preparing...)" report stays as the safety net.
//
// Release 0.6.0 polish (owner 2026-11-02: an AMD driver update clears the driver's shader cache): at its start the
// launcher looks for games prepared before whose driver version or WrongGPU runtime changed since (RrStaleAfterUpdate)
// and prepares them again in the background, the default mode first, with a plain status line; their game page shows
// "not prepared" until it is done.
//
// Measured 2026-11-02 (FINAL-PROGRESS.txt P2-P4): the AMD driver names a program's DX12 shader cache after the AMD AGS
// application name the program gives when it makes its device through AGS (FNV-1a of that name), else after its program
// file name. A game that gives one (RE Requiem) reads another cache than the helper fills, whatever the helper is named;
// such games are marked in gamerules.h (rrKernelsInGame) and prepare their kernels at their own first start.
#pragma once
#include "context.h"

#include <functional>

namespace wg {

struct RrPrepProgress { unsigned done = 0, total = 0, pass = 0; wstring kernel; };
// The graphics driver's version ("32.0.21037.1004") of the first AMD adapter; empty: none (or not readable).
wstring RrDriverVersion();
// The SHA-256 of this build's runtime (wronggpu_runtime.dll in the payload; the RR runtime is part of it).
wstring RrRuntimeSha(Context& ctx);
// Passes for an RR mode (the runtime's compile passes): accurate (the default) 1; fast / exact 2 (the other modes' kernels).
int RrPassesFor(const string& mode);
// The game's program name was prepared with this driver and runtime, for at least `passes`. *why: why not, in words.
bool RrPrepared(Context& ctx, const wstring& dataDir, const wstring& exe, int passes, wstring* why = nullptr);
// Its state in more detail: prepared, never prepared, prepared before but the driver / the runtime changed since, prepared
// for fewer passes, or a game that prepares them at its own first start (gamerules.h rrKernelsInGame).
enum class RrPrepState { Ready, NotPrepared, DriverChanged, RuntimeChanged, MorePasses, InGame };
RrPrepState RrPrepStateOf(Context& ctx, const wstring& dataDir, const wstring& exe, int passes, wstring* why = nullptr);
const wchar_t* RrPrepStateName(RrPrepState s);  // "ready", "not-prepared", "driver-changed", ... (JSON, logs)
// What the launcher's start prepares again in the background: prepared before, and the driver or the runtime changed.
inline bool RrStaleAfterUpdate(RrPrepState s) { return s == RrPrepState::DriverChanged || s == RrPrepState::RuntimeChanged; }
// gamerules.h: the game prepares RR's kernels at its own first start (the launcher's helper can't reach its cache)
bool RrKernelsInGame(const wstring& exe);
// The helper run: the launcher's own program copied as rr-prepare\run\<game exe name>, started with --rr-prepare
// <runtime dll> <passes>, its progress read line by line (progress answers false: cancel). 0 prepared (recorded), 1
// cancelled, 2 no RDNA 4 graphics card (nothing to prepare), 3 failed (*why), 4 the game prepares them at its own first
// start (RrKernelsInGame: no helper runs).
int PrepareRr(Context& ctx, const wstring& dataDir, const wstring& exe, const wstring& runtimeDll, int passes,
              const std::function<bool(const RrPrepProgress&)>& progress, wstring* why);
// The helper side: `<exe> --rr-prepare <runtime dll> <passes>` (main.cpp, before anything else of the launcher runs).
int RunRrPrepareHelper(int argc, wchar_t** argv);

}  // namespace wg
