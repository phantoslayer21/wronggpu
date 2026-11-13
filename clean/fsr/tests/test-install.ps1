# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
  Tests install\install-wronggpu-fsr.ps1 on mock game folders (no real game, no real DLLs needed).
  Runs on Windows PowerShell 5.1 and on PowerShell 7 (also on Linux). Exit code 0 when every case passes.
  The mock "AMD" and "WrongGPU" DLLs are small files whose identity comes from the script's test hook
  (WRONGGPU_INSTALL_TEST_IDENTITY), because they carry no signature or version resource.
#>
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

$script = Join-Path (Split-Path -Parent $PSScriptRoot) 'install\install-wronggpu-fsr.ps1'
$root = Join-Path ([System.IO.Path]::GetTempPath()) ('wg-install-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$failures = 0
$checks = 0

function Check([bool]$Condition, [string]$What) {
    $script:checks++
    if (-not $Condition) { $script:failures++; Write-Host "FAIL: $What" -ForegroundColor Red }
}

function Sha([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash }

function WriteBytes([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllBytes($Path, [System.Text.Encoding]::ASCII.GetBytes($Text))
}

$payload = Join-Path $root 'payload'
New-Item -ItemType Directory -Force -Path $payload | Out-Null
WriteBytes (Join-Path $payload 'amd_fidelityfx_dx12.dll') 'MZ wronggpu front end (mock)'
WriteBytes (Join-Path $payload 'ffx_fsr3_x64.dll') 'MZ wronggpu fsr3 forwarder (mock)'
WriteBytes (Join-Path $payload 'ffx_fsr3upscaler_x64.dll') 'MZ wronggpu fsr3upscaler forwarder (mock)'
$dlls = @('amd_fidelityfx_dx12.dll', 'ffx_fsr3_x64.dll', 'ffx_fsr3upscaler_x64.dll')
$identity = Join-Path $root 'identity.txt'

function NewGame([string]$Name) {
    $dir = Join-Path $root $Name
    $bin = Join-Path $dir 'bin\x64'
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    WriteBytes (Join-Path $bin 'Game.exe') 'MZ game'
    WriteBytes (Join-Path $bin 'amd_fidelityfx_dx12.dll') 'MZ amd fidelityfx 3.1 (mock)'
    WriteBytes (Join-Path $bin 'ffx_fsr3upscaler_x64.dll') 'MZ fsr 3.0 upscaler component (mock, unsigned)'
    WriteBytes (Join-Path $bin 'ffx_fsr3_x64.dll') 'MZ fsr 3.0 component (mock, unsigned)'
    WriteBytes (Join-Path $bin 'ffx_frameinterpolation_x64.dll') 'MZ fsr 3.0 frame interpolation (mock)'
    WriteBytes (Join-Path $bin 'other.dat') ('x' * 5000)
    return $bin
}

function Run([string[]]$Arguments) {
    $exe = (Get-Process -Id $PID).Path
    $out = & $exe -NoProfile -ExecutionPolicy Bypass -File $script @Arguments 2>&1 | Out-String
    return [pscustomobject]@{ code = $LASTEXITCODE; text = $out }
}

try {
    $gameA = NewGame 'A'
    $amdHash = Sha (Join-Path $gameA 'amd_fidelityfx_dx12.dll')
    $wgHash = Sha (Join-Path $payload 'amd_fidelityfx_dx12.dll')
    $newAmd = Join-Path $root 'new-amd.dll'
    WriteBytes $newAmd 'MZ amd fidelityfx 3.1.5 from a game update (mock)'
    $ids = @("$amdHash amd", "$(Sha $newAmd) amd")
    foreach ($d in $dlls) { $ids += "$(Sha (Join-Path $payload $d)) wronggpu" }
    Set-Content -LiteralPath $identity -Value $ids
    $gameHashes = @{}
    foreach ($d in $dlls) { $gameHashes[$d] = Sha (Join-Path $gameA $d) }
    $env:WRONGGPU_INSTALL_TEST_IDENTITY = $identity
    $before = @{}
    foreach ($f in Get-ChildItem -LiteralPath $gameA -File) { $before[$f.Name] = Sha $f.FullName }

    # 1. Install, simulate a game run, uninstall: byte-exact restore, logs moved out, nothing left.
    $r = Run @('-Action', 'Install', '-GameDir', $gameA, '-Payload', $payload, '-Yes')
    Check ($r.code -eq 0) "install exits 0: $($r.text)"
    foreach ($d in $dlls) {
        $orig = [System.IO.Path]::GetFileNameWithoutExtension($d) + '.wronggpu-original.dll'
        Check ((Sha (Join-Path $gameA $d)) -eq (Sha (Join-Path $payload $d))) "WrongGPU's $d in place"
        Check ((Sha (Join-Path $gameA $orig)) -eq $gameHashes[$d]) "the game's $d renamed to $orig"
        Check (Test-Path (Join-Path $gameA "wronggpu-backup\$d")) "backup of $d"
    }
    Check ($r.text -match 'ffx_fsr3_x64.dll: game''s .* \(signer: none\)') "unsigned component DLL shown: $($r.text)"
    Check (Test-Path (Join-Path $gameA 'wronggpu-fsr-install.json')) 'install record written'
    Check (Test-Path (Join-Path $gameA 'wronggpu-backup\amd_fidelityfx_dx12.dll')) 'backup written'
    $r = Run @('-Action', 'Install', '-GameDir', $gameA, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'already installed') 'second install refused'
    WriteBytes (Join-Path $gameA 'wronggpu-fsr.log') 'log'
    WriteBytes (Join-Path $gameA 'wronggpu.ini') 'ini'
    $logs = Join-Path $root 'logsA'
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameA, '-LogsTo', $logs, '-Yes')
    Check ($r.code -eq 0 -and $r.text -match '20-T3 PASS') "uninstall passes 20-T3: $($r.text)"
    $after = @{}
    foreach ($f in Get-ChildItem -LiteralPath $gameA -File) { $after[$f.Name] = Sha $f.FullName }
    Check ($after.Count -eq $before.Count) 'same number of files as before'
    foreach ($k in $before.Keys) { Check ($after[$k] -eq $before[$k]) "unchanged: $k" }
    Check (-not (Test-Path (Join-Path $gameA 'wronggpu-backup'))) 'backup folder removed'
    Check ((Test-Path (Join-Path $logs 'wronggpu-fsr.log')) -and (Test-Path (Join-Path $logs 'wronggpu.ini'))) 'logs moved out'

    # 2. Refusals: other mod's proxy DLL, unsigned FidelityFX DLL, anti-cheat, missing DLL.
    $gameB = NewGame 'B'
    WriteBytes (Join-Path $gameB 'version.dll') 'MZ proxy'
    $r = Run @('-Action', 'Install', '-GameDir', $gameB, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'another mod') 'refused with a proxy DLL present'
    Check (-not (Test-Path (Join-Path $gameB 'amd_fidelityfx_dx12.wronggpu-original.dll'))) 'nothing changed (B)'
    $gameC = NewGame 'C'
    WriteBytes (Join-Path $gameC 'amd_fidelityfx_dx12.dll') 'MZ unknown replacement'
    $r = Run @('-Action', 'Install', '-GameDir', $gameC, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'not signed by AMD') 'refused with an unsigned FidelityFX DLL'
    $gameD = NewGame 'D'
    New-Item -ItemType Directory -Force -Path (Join-Path $root 'D\EasyAntiCheat') | Out-Null
    $r = Run @('-Action', 'Install', '-GameDir', $gameD, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'anti-cheat') 'refused with anti-cheat'
    $gameE = NewGame 'E'
    Remove-Item (Join-Path $gameE 'amd_fidelityfx_dx12.dll')
    $r = Run @('-Action', 'Install', '-GameDir', $gameE, '-Payload', $payload, '-Yes')
    Check ($r.code -eq 0) "FSR 3.0-only game: the component DLLs are installed: $($r.text)"
    Check ((Test-Path (Join-Path $gameE 'ffx_fsr3_x64.wronggpu-original.dll')) -and -not (Test-Path (Join-Path $gameE 'amd_fidelityfx_dx12.dll'))) 'E: no FidelityFX API DLL added'
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameE, '-LogsTo', (Join-Path $root 'logsE'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match '20-T3 PASS') "E uninstall: $($r.text)"
    $gameE2 = NewGame 'E2'
    foreach ($d in $dlls) { Remove-Item (Join-Path $gameE2 $d) }
    $r = Run @('-Action', 'Install', '-GameDir', $gameE2, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'none of') 'refused without any FidelityFX DLL'

    # 3. A game update replaced the DLL while WrongGPU was installed: the new AMD file is kept.
    $gameF = NewGame 'F'
    $r = Run @('-Action', 'Install', '-GameDir', $gameF, '-Payload', $payload, '-Yes')
    Check ($r.code -eq 0) 'install F'
    Copy-Item -LiteralPath $newAmd -Destination (Join-Path $gameF 'amd_fidelityfx_dx12.dll') -Force
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameF, '-LogsTo', (Join-Path $root 'logsF'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match 'game update') "game update handled: $($r.text)"
    Check ((Sha (Join-Path $gameF 'amd_fidelityfx_dx12.dll')) -eq (Sha $newAmd)) "game's new DLL kept"
    Check (-not (Test-Path (Join-Path $gameF 'amd_fidelityfx_dx12.wronggpu-original.dll'))) 'old renamed DLL removed'

    # 4. The renamed AMD DLL went missing: restored from the backup.
    $gameG = NewGame 'G'
    $r = Run @('-Action', 'Install', '-GameDir', $gameG, '-Payload', $payload, '-Yes')
    Remove-Item (Join-Path $gameG 'amd_fidelityfx_dx12.wronggpu-original.dll')
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameG, '-LogsTo', (Join-Path $root 'logsG'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match '20-T3 PASS') "restored from backup: $($r.text)"
    Check ((Sha (Join-Path $gameG 'amd_fidelityfx_dx12.dll')) -eq $amdHash) "AMD's DLL restored"

    # 5. A file added to the game folder while installed is reported (20-T3 must not hide it).
    $gameH = NewGame 'H'
    $r = Run @('-Action', 'Install', '-GameDir', $gameH, '-Payload', $payload, '-Yes')
    WriteBytes (Join-Path $gameH 'crash.dmp') 'dump'
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameH, '-LogsTo', (Join-Path $root 'logsH'), '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'new file: crash.dmp') "new file reported: $($r.text)"

    # 6. -Components: only the FidelityFX API DLL; then a record of the first test build (format 1) still uninstalls.
    $gameI = NewGame 'I'
    $r = Run @('-Action', 'Install', '-GameDir', $gameI, '-Payload', $payload, '-Components', 'amd_fidelityfx_dx12.dll', '-Yes')
    Check ($r.code -eq 0) "install I: $($r.text)"
    Check (Test-Path (Join-Path $gameI 'amd_fidelityfx_dx12.wronggpu-original.dll')) 'I: FidelityFX API DLL replaced'
    Check (-not (Test-Path (Join-Path $gameI 'ffx_fsr3_x64.wronggpu-original.dll'))) 'I: component DLL left alone'
    $rec = Get-Content -LiteralPath (Join-Path $gameI 'wronggpu-fsr-install.json') -Raw | ConvertFrom-Json
    $c0 = $rec.components[0]
    $old = [ordered]@{ format = 1; installedAt = $rec.installedAt; gameDir = $rec.gameDir; wronggpuVersion = $rec.wronggpuVersion
                       wronggpuSha256 = $c0.wronggpuSha256; original = [ordered]@{ name = $c0.name; size = $c0.original.size; sha256 = $c0.original.sha256 }
                       backup = (Join-Path $rec.backupDir $c0.name); snapshot = $rec.snapshot }
    $old | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $gameI 'wronggpu-fsr-install.json') -Encoding UTF8
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameI, '-LogsTo', (Join-Path $root 'logsI'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match '20-T3 PASS') "format-1 record uninstalls: $($r.text)"

    # 7. A game update replaced an (unsigned) FSR 3.0 component DLL: the new file is kept.
    $gameJ = NewGame 'J'
    $r = Run @('-Action', 'Install', '-GameDir', $gameJ, '-Payload', $payload, '-Yes')
    WriteBytes (Join-Path $gameJ 'ffx_fsr3upscaler_x64.dll') 'MZ fsr 3.0 upscaler from a game update (mock)'
    $newHash = Sha (Join-Path $gameJ 'ffx_fsr3upscaler_x64.dll')
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameJ, '-LogsTo', (Join-Path $root 'logsJ'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match 'game update replaced ffx_fsr3upscaler_x64.dll') "component update kept: $($r.text)"
    Check ((Sha (Join-Path $gameJ 'ffx_fsr3upscaler_x64.dll')) -eq $newHash) 'J: new component kept'
    Check (-not (Test-Path (Join-Path $gameJ 'ffx_fsr3upscaler_x64.wronggpu-original.dll'))) 'J: old renamed component removed'
    Check ((Sha (Join-Path $gameJ 'ffx_fsr3_x64.dll')) -eq (Sha (Join-Path (NewGame 'Jref') 'ffx_fsr3_x64.dll'))) 'J: other component restored'

    # 8. A leftover renamed original without a record (an interrupted earlier install) is refused.
    $gameK = NewGame 'K'
    WriteBytes (Join-Path $gameK 'ffx_fsr3_x64.wronggpu-original.dll') 'MZ leftover'
    $r = Run @('-Action', 'Install', '-GameDir', $gameK, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'ffx_fsr3_x64.wronggpu-original.dll') "leftover refused: $($r.text)"

    # 9. The network runtime: copied from the payload, recorded, removed on uninstall (20-T3 still byte for byte);
    #    -Runtime names another file; a runtime already in the folder is refused like any WrongGPU leftover.
    $payloadRt = Join-Path $root 'payloadRt'
    New-Item -ItemType Directory -Force -Path $payloadRt | Out-Null
    foreach ($d in $dlls) { Copy-Item -LiteralPath (Join-Path $payload $d) -Destination $payloadRt }
    WriteBytes (Join-Path $payloadRt 'wronggpu_runtime.dll') 'MZ wronggpu network runtime (mock)'
    $gameL = NewGame 'L'
    $r = Run @('-Action', 'Install', '-GameDir', $gameL, '-Payload', $payloadRt, '-Yes')
    Check ($r.code -eq 0 -and $r.text -match 'wronggpu_runtime.dll = WrongGPU network runtime') "runtime installed: $($r.text)"
    Check ((Sha (Join-Path $gameL 'wronggpu_runtime.dll')) -eq (Sha (Join-Path $payloadRt 'wronggpu_runtime.dll'))) 'L: runtime in place'
    $r = Run @('-Action', 'Status', '-GameDir', $gameL)
    Check ($r.text -match 'wronggpu_runtime.dll : present') "L: status shows the runtime: $($r.text)"
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameL, '-LogsTo', (Join-Path $root 'logsL'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match '20-T3 PASS') "L: uninstall with runtime is exact: $($r.text)"
    Check (-not (Test-Path (Join-Path $gameL 'wronggpu_runtime.dll'))) 'L: runtime removed'
    $gameM = NewGame 'M'
    $other = Join-Path $root 'my-runtime.dll'
    WriteBytes $other 'MZ wronggpu network runtime, other build (mock)'
    $r = Run @('-Action', 'Install', '-GameDir', $gameM, '-Payload', $payload, '-Runtime', $other, '-Yes')
    Check ($r.code -eq 0 -and (Sha (Join-Path $gameM 'wronggpu_runtime.dll')) -eq (Sha $other)) "M: -Runtime installed under the runtime's name: $($r.text)"
    $r = Run @('-Action', 'Uninstall', '-GameDir', $gameM, '-LogsTo', (Join-Path $root 'logsM'), '-Yes')
    Check ($r.code -eq 0 -and $r.text -match '20-T3 PASS') "M: exact: $($r.text)"
    $r = Run @('-Action', 'Install', '-GameDir', $gameM, '-Payload', $payload, '-Runtime', (Join-Path $root 'nothing.dll'), '-Yes')
    Check ($r.code -ne 0 -and $r.text -match '-Runtime: .* is not a file') "M: missing -Runtime refused: $($r.text)"
    $gameN = NewGame 'N'
    WriteBytes (Join-Path $gameN 'wronggpu_runtime.dll') 'MZ leftover runtime'
    $r = Run @('-Action', 'Install', '-GameDir', $gameN, '-Payload', $payload, '-Yes')
    Check ($r.code -ne 0 -and $r.text -match 'wronggpu_runtime.dll') "N: leftover runtime refused: $($r.text)"

    # 10. Status runs.
    $r = Run @('-Action', 'Status', '-GameDir', $gameA)
    Check ($r.code -eq 0 -and $r.text -match 'install record: none') "status: $($r.text)"
} finally {
    Remove-Item Env:\WRONGGPU_INSTALL_TEST_IDENTITY -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "$checks checks, $failures failures"
if ($failures) { exit 1 }
exit 0
