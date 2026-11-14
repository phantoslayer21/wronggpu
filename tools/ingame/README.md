# In-game run tools

One in-game run of a WrongGPU candidate build, with evidence and a byte-for-byte restore. Used for Model L in
Cyberpunk 2077 and Resident Evil Requiem (#89). Windows only, Python 3, no GPU work of their own except
`offline_create.py`.

Nothing in these tools is specific to one PC. Every path comes from a game config (`run.py --config`) or from
arguments (`offline_create.py`, `make_record.py`).

## Before a game: the offline create test

```
python offline_create.py build  --work <dir>
python offline_create.py create --work <dir> --runtime-dir <built runtime> --model <L model.wgcm> --model-sha <sha256> [--extra l_post=own] [-- <GPU slot wrapper>]
```

`create` runs the interop harness once:
- `model=L` and `executor=dx12`;
- 600 frames, 1280x720 to 1920x1080, with a launch trace.

It passes only if all of these hold:
- the runtime ran on D3D12;
- the timing windows came back;
- Model L kernels appear in the trace.

Run it before every game run with a new build.

`timing` measures Model L against Model M on six shapes, interleaved, and writes `l-timing-harness.json`.

Everything after a lone `--` wraps each GPU run, for example a GPU-slot tool.

## The game run

1. Copy `examples/<game>.json` and fill in every `<placeholder>`:
   - `game_root`, `exe_dir`, `exe`, `launchers`;
   - `out`: an empty folder for this run;
   - `launcher_package` and `payload`: from `build.py --model-l`;
   - `model`: the file and its SHA-256;
   - `aside`: files moved out of the game for the run, then moved back. Never opened, only hashed and renamed;
   - `hash_only`: files hashed before and after but never touched, for example a script loader the game needs;
   - `settings`: the file, its format (`ini` or `cp2077`) and the keys to set;
   - `gpu_flag`: the GPU flag file;
   - `disk_free_gb`: minimum free space per drive.

   `%VARIABLES%` are expanded. `exe_dir` is relative to `game_root`. Optional `runtime_options_extra` adds runtime
   options to every model choice (e.g. `l_post=own`); `--extra` on `prepare`, `use-m` or `use-l` replaces it for that
   choice (`--extra ''` for none), so two variants can run in one session.

2. Run the phases:

```
python run.py --config my.json all --dry-run      every phase, nothing changed
python run.py --config my.json preflight
python run.py --config my.json prepare
python run.py --config my.json launch             then start the game in its launcher
python run.py --config my.json hold
python run.py --config my.json shot <label>       ...  use-m / use-l between runs (game closed)
python run.py --config my.json collect            after the game has exited
python run.py --config my.json restore
```

What each phase does:
- **`prepare`**:
  - makes a manifest of the game folder;
  - backs up our install's files and the settings file;
  - moves the `aside` files out and hashes the `hash_only` files;
  - installs the candidate with the launcher's `update`;
  - sets `runtime_options model=<model>` and OptiScaler's `[WrongGPU] UpscalerIndex`;
  - adds the timing and launch-trace options;
  - copies the model to `<exe dir>\WrongGPU\model\<model>\model.wgcm`.

  The model goes into the game's own folder, not `%LOCALAPPDATA%`, because a tool running inside a packaged
  (MSIX) app writes new `%LOCALAPPDATA%` files to a private copy that the game never sees.
- **`launch`**:
  - takes the GPU flag for this one game run;
  - starts `game_hold.py`, which releases the flag in `finally` when the game exits or never starts;
  - starts the VRAM log.
- **`collect`**:
  - copies our own logs. OptiScaler's log is never read;
  - writes `proof.json` with the model option, the model file and digest, Model L kernels, the executor, dispatch
    counts, refusals and the kernel counts of each launch trace (one trace per model choice,
    `launch-trace-<time>.txt`).

  The verdict is PASS or NOT PROVEN.
- **`restore`**:
  - puts every backed-up file back;
  - removes what the update added and the placed model, with the folders it created;
  - moves the `aside` files back and checks the `hash_only` hashes;
  - restores the settings file;
  - verifies the game folder against the manifest.

3. Timing record from the runtime's own timing lines, split by game process and scene:

```
python make_record.py --log <out>\logs\wronggpu-fsr.log --scenes scenes.json --out record.json [--meta meta.json] [--checkout <src>] [--gpu-name-like "*name*"]
```

## Rules these tools keep

- Only our files and the files the config names are touched.
- Vendor files (`nvngx*`, `*dlss*`, `sl.*`, `nvapi*`) are recorded by name, size and time only. They are never opened.
- Another mod's files are hashed at most.
- The model stays in place until the game has exited. Only `restore` removes it.
- Do not use these tools on games with anti-cheat.
