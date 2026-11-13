# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
  Installs or removes WrongGPU's FSR DLLs (spec 20) in one game folder, for tests 20-T0 to 20-T3.

.DESCRIPTION
  WrongGPU replaces up to three of the game's FidelityFX DLLs, each forwarding to AMD's original kept next to it:
    amd_fidelityfx_dx12.dll   -> amd_fidelityfx_dx12.wronggpu-original.dll   (FSR 3.1 API, integration point 1)
    ffx_fsr3_x64.dll          -> ffx_fsr3_x64.wronggpu-original.dll          (FSR 3.0 components, integration point 2)
    ffx_fsr3upscaler_x64.dll  -> ffx_fsr3upscaler_x64.wronggpu-original.dll  (FSR 3.0 components, integration point 2)
  A DLL is installed when both the game folder and the payload have it (or only those named with -Components).
  WrongGPU's network runtime (wronggpu_runtime.dll, for upscaler = network) is copied next to them when the payload
  holds it or -Runtime names it, and removed again on uninstall.

  Install
    - checks the folder: no anti-cheat, no other upscaler mod or proxy DLL, WrongGPU not installed yet; the game's
      amd_fidelityfx_dx12.dll must be signed by AMD (the FSR 3.0 component DLLs are built by the game's developer
      and usually not signed by AMD: their signer is shown);
    - records the size and SHA-256 of every file in the folder (wronggpu-fsr-install.json);
    - copies each replaced DLL to wronggpu-backup\, renames it to <name>.wronggpu-original.dll and puts WrongGPU's
      DLL in its place.
  Uninstall
    - moves WrongGPU's log and settings (wronggpu-fsr.log, wronggpu.ini) out of the game folder (to -LogsTo), so the
      folder can be compared byte for byte and the settings are kept for the next install;
    - removes WrongGPU's DLLs and renames AMD's back (or restores them from the backup);
    - compares every file with the record made at install time and reports PASS (byte for byte, nothing left
      behind) or each difference. This is test 20-T3.
    - If a game update replaced one of the DLLs in the meantime, the game's new file is kept.
  Status
    - shows what is installed.

  Nothing is written outside the game folder, except the logs copied to -LogsTo on uninstall.
  Windows PowerShell 5.1 or later. Game folders under Program Files need an administrator PowerShell.

.PARAMETER Action
  Install, Uninstall or Status.
.PARAMETER GameDir
  The folder that holds the game's executable and its FidelityFX DLLs
  (Cyberpunk 2077: <Steam library>\steamapps\common\Cyberpunk 2077\bin\x64).
.PARAMETER Payload
  Folder holding WrongGPU's DLLs (default: this script's folder).
.PARAMETER Components
  Install only these DLL names (default: every one present in both the game folder and the payload).
.PARAMETER Runtime
  Install: path of WrongGPU's network runtime DLL to put into the game folder as wronggpu_runtime.dll (default: the
  payload's wronggpu_runtime.dll when present; none otherwise).
.PARAMETER LogsTo
  Uninstall: folder that receives WrongGPU's log and settings (default: .\wronggpu-logs-<time> in the current folder).
.PARAMETER Yes
  Do not ask for confirmation.

.EXAMPLE
  .\install-wronggpu-fsr.ps1 -Action Install -GameDir "C:\Games\Cyberpunk 2077\bin\x64" -Payload ..\..\..\build\fsr\Release
.EXAMPLE
  .\install-wronggpu-fsr.ps1 -Action Uninstall -GameDir "C:\Games\Cyberpunk 2077\bin\x64"
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet('Install', 'Uninstall', 'Status')][string]$Action,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [string]$Payload = $PSScriptRoot,
    [string[]]$Components = @(),
    [string]$Runtime = '',
    [string]$LogsTo = '',
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

# The DLLs WrongGPU can replace. RequireAmd: the game's file must carry AMD's signature.
$Known = @(
    [pscustomobject]@{ name = 'amd_fidelityfx_dx12.dll'; requireAmd = $true },
    [pscustomobject]@{ name = 'ffx_fsr3_x64.dll'; requireAmd = $false },
    [pscustomobject]@{ name = 'ffx_fsr3upscaler_x64.dll'; requireAmd = $false }
)
function OriginalNameOf([string]$Name) { return [System.IO.Path]::GetFileNameWithoutExtension($Name) + '.wronggpu-original.dll' }

$RecordName = 'wronggpu-fsr-install.json'
$RuntimeName = 'wronggpu_runtime.dll'
$BackupName = 'wronggpu-backup'
# WrongGPU's settings and logs next to the DLLs (wronggpu-fsr.ini: name used by builds before 0.3.2).
$OwnFiles = @('wronggpu.ini', 'wronggpu-fsr.log', 'wronggpu-fsr.log.old', 'wronggpu-fsr.ini')

# Files that show another mod hooks the game (proxy DLLs, other upscaler mods, older WrongGPU / "DLSS 4.5 AMD",
# leftovers of an earlier WrongGPU install). Only their names are checked (-like, so no 8.3-name surprises); their
# contents are never read.
$ConflictPatterns = @('version.dll', 'winmm.dll', 'dxgi.dll', 'd3d12.dll', 'dinput8.dll', 'winhttp.dll',
                      'nvngx.dll', 'nvapi64.dll', 'OptiScaler*', '*.asi', 'dlss45*', '*wronggpu*',
                      'amd_fidelityfx_dx12.?*.dll', 'ffx_fsr3*.?*.dll')
# Anti-cheat: games with it are not supported (spec 20).
$AntiCheatPatterns = @('EasyAntiCheat*', 'BattlEye', 'BEService*', '*AntiCheat*', 'vgk.sys', 'xigncode*', 'nProtect*')

function Fail([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Confirm([string]$Question) {
    if ($Yes) { return }
    $answer = Read-Host "$Question [y/N]"
    if ($answer -notmatch '^(y|yes)$') { Write-Host 'Nothing changed.'; exit 2 }
}

function Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash }

# Size and hash of every file directly in the game folder, except WrongGPU's own files.
function Snapshot([string]$Dir) {
    $skip = @($RecordName) + $OwnFiles + @($Known | ForEach-Object { OriginalNameOf $_.name })
    $items = @()
    foreach ($f in Get-ChildItem -LiteralPath $Dir -File -Force | Sort-Object Name) {
        if ($skip -contains $f.Name) { continue }
        $items += [pscustomobject]@{ name = $f.Name; size = $f.Length; sha256 = (Sha256 $f.FullName) }
    }
    return , $items
}

# Test hook for tests/test-install.ps1 only: WRONGGPU_INSTALL_TEST_IDENTITY names a file of "<sha256> amd|wronggpu"
# lines that replaces the signature and version-resource checks for mock files.
$TestIdentity = @{}
if ($env:WRONGGPU_INSTALL_TEST_IDENTITY) {
    foreach ($line in Get-Content -LiteralPath $env:WRONGGPU_INSTALL_TEST_IDENTITY) {
        $parts = $line.Trim() -split '\s+', 2
        if ($parts.Count -eq 2) { $TestIdentity[$parts[0].ToUpperInvariant()] = $parts[1] }
    }
}

function DescriptionOf([string]$Path) {
    try { return [System.Diagnostics.FileVersionInfo]::GetVersionInfo($Path).FileDescription } catch { return '' }
}

function IsWrongGpuDll([string]$Path) {
    if ($TestIdentity.Count) { return $TestIdentity[(Sha256 $Path)] -eq 'wronggpu' }
    return (DescriptionOf $Path) -like '*WrongGPU*'
}

function SignerOf([string]$Path) {
    if ($TestIdentity.Count) { if ($TestIdentity[(Sha256 $Path)] -eq 'amd') { return 'Advanced Micro Devices (test)' } else { return '' } }
    $sig = Get-AuthenticodeSignature -LiteralPath $Path
    if ($sig.Status -ne 'Valid' -or $null -eq $sig.SignerCertificate) { return '' }
    return $sig.SignerCertificate.Subject
}

function IsAmdSigned([string]$Path) { return (SignerOf $Path) -match 'Advanced Micro Devices' }

function VersionOf([string]$Path) {
    try { return (Get-Item -LiteralPath $Path).VersionInfo.FileVersion } catch { return '' }
}

function CheckFolder([string]$Dir) {
    $found = @()
    $names = @(Get-ChildItem -LiteralPath $Dir -Force -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
    foreach ($p in $ConflictPatterns) { $found += @($names | Where-Object { $_ -like $p }) }
    if ($found.Count) {
        Fail ("files of another mod, a proxy DLL or an earlier WrongGPU install are in the game folder: " +
              (($found | Sort-Object -Unique) -join ', ') +
              ". Ask the owner to remove that mod first (with its own uninstaller), then run this again.")
    }
    # Anti-cheat folders or files in the game folder or up to two folders above it.
    $look = @($Dir)
    $parent = Split-Path -Parent $Dir
    if ($parent) { $look += $parent; $grand = Split-Path -Parent $parent; if ($grand) { $look += $grand } }
    foreach ($d in $look) {
        $here = @(Get-ChildItem -LiteralPath $d -Force -ErrorAction SilentlyContinue)
        foreach ($p in $AntiCheatPatterns) {
            $hit = $here | Where-Object { $_.Name -like $p } | Select-Object -First 1
            if ($hit) { Fail "this game uses anti-cheat ($($hit.FullName)); games with anti-cheat are not supported." }
        }
    }
}

$GameDir = (Resolve-Path -LiteralPath $GameDir).Path
$Record = Join-Path $GameDir $RecordName
$Backup = Join-Path $GameDir $BackupName

if ($Action -eq 'Status') {
    Write-Host "Game folder: $GameDir"
    foreach ($k in $Known) {
        $p = Join-Path $GameDir $k.name
        $o = Join-Path $GameDir (OriginalNameOf $k.name)
        if (Test-Path -LiteralPath $p) {
            $who = if (IsWrongGpuDll $p) { 'WrongGPU' } elseif (IsAmdSigned $p) { 'AMD (signed)' } else { "game's (signer: $(if (SignerOf $p) { SignerOf $p } else { 'none' }))" }
            Write-Host "$($k.name) : $who, $(VersionOf $p), SHA-256 $(Sha256 $p)"
        } else { Write-Host "$($k.name) : not present" }
        if (Test-Path -LiteralPath $o) { Write-Host "$(OriginalNameOf $k.name) : present, SHA-256 $(Sha256 $o)" }
    }
    if (Test-Path -LiteralPath $Record) {
        $r = Get-Content -LiteralPath $Record -Raw | ConvertFrom-Json
        Write-Host "Installed $($r.installedAt) (WrongGPU $($r.wronggpuVersion))"
    } else { Write-Host 'WrongGPU install record: none' }
    foreach ($f in $OwnFiles) { if (Test-Path -LiteralPath (Join-Path $GameDir $f)) { Write-Host "present: $f" } }
    $rt = Join-Path $GameDir $RuntimeName
    if (Test-Path -LiteralPath $rt) { Write-Host "$RuntimeName : present, $(VersionOf $rt), SHA-256 $(Sha256 $rt)" }
    exit 0
}

if ($Action -eq 'Install') {
    if (Test-Path -LiteralPath $Record) { Fail 'WrongGPU is already installed here (wronggpu-fsr-install.json exists). Run -Action Uninstall first.' }
    CheckFolder $GameDir
    $plan = @()
    foreach ($k in $Known) {
        if ($Components.Count -and -not ($Components -contains $k.name)) { continue }
        $game = Join-Path $GameDir $k.name
        $ours = Join-Path $Payload $k.name
        if (-not (Test-Path -LiteralPath $game)) { if ($Components.Count) { Fail "$($k.name) is not in $GameDir." }; continue }
        if (-not (Test-Path -LiteralPath $ours)) { if ($Components.Count) { Fail "WrongGPU's $($k.name) is not in $Payload." }; continue }
        if (-not (IsWrongGpuDll $ours)) { Fail "$ours is not WrongGPU's DLL (its file description does not say WrongGPU)." }
        if (IsWrongGpuDll $game) { Fail "the game's $($k.name) is already a WrongGPU DLL without an install record. Restore the game's files (Steam: Verify integrity of game files) first." }
        $signer = SignerOf $game
        if ($k.requireAmd -and -not ($signer -match 'Advanced Micro Devices')) {
            Fail "the game's $($k.name) is not signed by AMD; it may belong to another mod. Restore the game's files (Steam: Verify integrity of game files) and try again."
        }
        $plan += [pscustomobject]@{ name = $k.name; game = $game; ours = $ours; signer = $signer }
    }
    if (-not $plan.Count) { Fail "none of $(($Known | ForEach-Object { $_.name }) -join ', ') is in both $GameDir and $Payload." }
    $runtimeSource = ''
    if ($Runtime) {
        if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf)) { Fail "-Runtime: $Runtime is not a file." }
        $runtimeSource = (Resolve-Path -LiteralPath $Runtime).Path
    } elseif (Test-Path -LiteralPath (Join-Path $Payload $RuntimeName)) {
        $runtimeSource = (Resolve-Path -LiteralPath (Join-Path $Payload $RuntimeName)).Path
    }
    Write-Host "Game folder: $GameDir"
    foreach ($c in $plan) {
        Write-Host ("  {0}: game's {1} (signer: {2}) -> WrongGPU {3}" -f $c.name, (VersionOf $c.game),
                    $(if ($c.signer) { $c.signer } else { 'none' }), (VersionOf $c.ours))
    }
    if ($runtimeSource) { Write-Host "  $RuntimeName : WrongGPU network runtime $(VersionOf $runtimeSource), from $runtimeSource" }
    else { Write-Host "  $RuntimeName : not installed (none in the payload, no -Runtime); upscaler = network will report it missing" }
    Confirm 'Install WrongGPU into this game folder (the game''s DLLs are kept and backed up)?'

    Write-Host 'Recording the game folder (hashes of every file)...'
    $snapshot = Snapshot $GameDir
    New-Item -ItemType Directory -Force -Path $Backup | Out-Null
    $done = @()
    try {
        foreach ($c in $plan) {
            $hash = Sha256 $c.game
            $size = (Get-Item -LiteralPath $c.game).Length
            $backupFile = Join-Path $Backup $c.name
            Copy-Item -LiteralPath $c.game -Destination $backupFile -Force
            if ((Sha256 $backupFile) -ne $hash) { throw "the backup copy of $($c.name) does not match" }
            Rename-Item -LiteralPath $c.game -NewName (OriginalNameOf $c.name)
            $done += $c
            Copy-Item -LiteralPath $c.ours -Destination $c.game
            $c | Add-Member -NotePropertyName original -NotePropertyValue ([ordered]@{ size = $size; sha256 = $hash })
            $c | Add-Member -NotePropertyName wronggpuSha256 -NotePropertyValue (Sha256 $c.game)
        }
        $runtimeInfo = $null
        if ($runtimeSource) {
            $rtPath = Join-Path $GameDir $RuntimeName
            Copy-Item -LiteralPath $runtimeSource -Destination $rtPath
            $runtimeInfo = [ordered]@{ name = $RuntimeName; sha256 = (Sha256 $rtPath); source = $runtimeSource }
        }
        $info = [ordered]@{
            format          = 2
            installedAt     = (Get-Date).ToString('s')
            gameDir         = $GameDir
            wronggpuVersion = (VersionOf $plan[0].ours)
            components      = @($plan | ForEach-Object { [ordered]@{ name = $_.name; original = $_.original; wronggpuSha256 = $_.wronggpuSha256 } })
            backupDir       = $Backup
            runtime         = $runtimeInfo
            snapshot        = $snapshot
        }
        $info | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Record -Encoding UTF8
    } catch {
        Write-Host "Install failed: $($_.Exception.Message). Undoing..." -ForegroundColor Red
        foreach ($c in $done) {
            if ((Test-Path -LiteralPath $c.game) -and (IsWrongGpuDll $c.game)) { Remove-Item -LiteralPath $c.game -Force }
            $o = Join-Path $GameDir (OriginalNameOf $c.name)
            if (Test-Path -LiteralPath $o) { Rename-Item -LiteralPath $o -NewName $c.name }
            if (-not (Test-Path -LiteralPath $c.game)) { Copy-Item -LiteralPath (Join-Path $Backup $c.name) -Destination $c.game }
        }
        if ($runtimeSource -and (Test-Path -LiteralPath (Join-Path $GameDir $RuntimeName))) { Remove-Item -LiteralPath (Join-Path $GameDir $RuntimeName) -Force }
        if (Test-Path -LiteralPath $Record) { Remove-Item -LiteralPath $Record -Force }
        Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
        Fail 'the game folder was restored; if this was a permission error, run PowerShell as administrator.'
    }
    Write-Host 'Installed.' -ForegroundColor Green
    foreach ($c in $plan) {
        Write-Host "  $($c.name) = WrongGPU, SHA-256 $($c.wronggpuSha256)"
        Write-Host "  $(OriginalNameOf $c.name) = the game's, SHA-256 $($c.original.sha256)"
    }
    if ($runtimeInfo) { Write-Host "  $RuntimeName = WrongGPU network runtime, SHA-256 $($runtimeInfo.sha256)" }
    Write-Host "  backup: $Backup"
    Write-Host 'Start the game; WrongGPU writes wronggpu-fsr.log and wronggpu.ini (settings) next to the DLLs.'
    exit 0
}

# ---- Uninstall ----
if (-not (Test-Path -LiteralPath $Record)) {
    Fail 'no WrongGPU install record here (wronggpu-fsr-install.json). Nothing to uninstall, or it was installed another way.'
}
$r = Get-Content -LiteralPath $Record -Raw | ConvertFrom-Json
if ($r.format -eq 1) {
    # Records of the first test build (amd_fidelityfx_dx12.dll only).
    $entries = @([pscustomobject]@{ name = $r.original.name; original = $r.original; wronggpuSha256 = $r.wronggpuSha256 })
    $backupDir = Split-Path -Parent $r.backup
} else {
    $entries = @($r.components)
    $backupDir = $r.backupDir
}
Confirm "Uninstall WrongGPU from $GameDir and restore the game's DLLs?"

# 1. Logs and settings out of the game folder.
if (-not $LogsTo) { $LogsTo = Join-Path (Get-Location).Path ('wronggpu-logs-' + (Get-Date).ToString('yyyyMMdd-HHmmss')) }
$moved = @()
foreach ($f in $OwnFiles) {
    $p = Join-Path $GameDir $f
    if (Test-Path -LiteralPath $p) {
        New-Item -ItemType Directory -Force -Path $LogsTo | Out-Null
        Copy-Item -LiteralPath $p -Destination (Join-Path $LogsTo $f) -Force
        Remove-Item -LiteralPath $p -Force
        $moved += $f
    }
}
if ($moved.Count) { Write-Host "Logs and settings moved to $LogsTo : $($moved -join ', ')" }

# 2. The DLLs.
$updated = @()
foreach ($e in $entries) {
    $dll = Join-Path $GameDir $e.name
    $original = Join-Path $GameDir (OriginalNameOf $e.name)
    $spec = $Known | Where-Object { $_.name -eq $e.name }
    $gameUpdated = $false
    if (Test-Path -LiteralPath $dll) {
        $current = Sha256 $dll
        if ($current -eq $e.wronggpuSha256 -or (IsWrongGpuDll $dll)) {
            Remove-Item -LiteralPath $dll -Force
        } elseif ($current -eq $e.original.sha256) {
            Write-Host "$($e.name) is already the game's original."
        } elseif ((IsAmdSigned $dll) -or ($spec -and -not $spec.requireAmd)) {
            # A game update put a new DLL in place: keep it, drop the old renamed copy.
            $gameUpdated = $true
            $updated += $e.name
            Write-Host "A game update replaced $($e.name); the new file is kept." -ForegroundColor Yellow
        } else {
            Fail "$($e.name) is neither WrongGPU's nor AMD's (SHA-256 $current); another mod may have replaced it. Nothing more removed."
        }
    }
    if (Test-Path -LiteralPath $original) {
        if ($gameUpdated -or (Test-Path -LiteralPath $dll)) { Remove-Item -LiteralPath $original -Force }
        else { Rename-Item -LiteralPath $original -NewName $e.name }
    }
    if (-not $gameUpdated -and ((-not (Test-Path -LiteralPath $dll)) -or (Sha256 $dll) -ne $e.original.sha256)) {
        $backupFile = Join-Path $backupDir $e.name
        Write-Host "Restoring $($e.name) from the backup..."
        if (-not (Test-Path -LiteralPath $backupFile)) { Fail "the backup $backupFile is missing; use Steam's 'Verify integrity of game files'." }
        Copy-Item -LiteralPath $backupFile -Destination $dll -Force
    }
}

# 2b. WrongGPU's network runtime, when this install put one there.
$rtProp = $r.PSObject.Properties['runtime']
if ($rtProp -and $rtProp.Value) {
    $rt = Join-Path $GameDir $RuntimeName
    if (Test-Path -LiteralPath $rt) {
        if ((Sha256 $rt) -ne $rtProp.Value.sha256) { Write-Host "$RuntimeName was replaced since the install; removed anyway (it is WrongGPU's file)." -ForegroundColor Yellow }
        Remove-Item -LiteralPath $rt -Force
    }
}

# 3. Compare the folder with the record (test 20-T3).
$now = Snapshot $GameDir
$problems = @()
$before = @{}
foreach ($e in $r.snapshot) { $before[$e.name] = $e }
$after = @{}
foreach ($e in $now) { $after[$e.name] = $e }
foreach ($name in $before.Keys) {
    if (-not $after.ContainsKey($name)) { $problems += "missing: $name"; continue }
    if ($after[$name].sha256 -ne $before[$name].sha256) {
        if ($updated -contains $name) { $problems += "changed by the game update (expected): $name" }
        else { $problems += "changed: $name" }
    }
}
foreach ($name in $after.Keys) { if (-not $before.ContainsKey($name)) { $problems += "new file: $name" } }

# 4. Remove the record and the backup when the restore is exact.
$exact = @($problems | Where-Object { $_ -notlike 'changed by the game update*' }).Count -eq 0
if ($exact) {
    Remove-Item -LiteralPath $Record -Force
    Remove-Item -LiteralPath $backupDir -Recurse -Force -ErrorAction SilentlyContinue
}
$left = @(Get-ChildItem -LiteralPath $GameDir -Force | Where-Object { $_.Name -like '*wronggpu*' } | ForEach-Object { $_.Name })
if ($exact -and $left.Count -eq 0) {
    Write-Host "20-T3 PASS: $($before.Count) files identical to before the install (SHA-256), nothing of WrongGPU left." -ForegroundColor Green
    foreach ($p in $problems) { Write-Host "  note: $p" }
    exit 0
}
Write-Host '20-T3 FAIL: the folder differs from the record made at install time:' -ForegroundColor Red
foreach ($p in $problems) { Write-Host "  $p" }
foreach ($n in $left) { Write-Host "  still present: $n" }
Write-Host "The install record and backup were kept in $GameDir for inspection."
exit 1
