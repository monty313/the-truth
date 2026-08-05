# MARK HERE! from the-truth — opens the SAME MarkOS Second Brain as ARMY.
# Soul match: canonical soul lives under ARMY 01_SYSTEM\config\agents\
$ErrorActionPreference = "Continue"
$ArmySystem = "C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM"
$ChatUrl = "http://127.0.0.1:8000/chat"
$HealthUrl = "http://127.0.0.1:8000/api/health"
$StartScript = Join-Path $ArmySystem "scripts\start_service.ps1"
$SoulFile = Join-Path $ArmySystem "config\agents\MARK_PERSONALITY.md"
$Doctrine = Join-Path $ArmySystem "data\knowledge\skills\trading\llm_basic_thinking\00_INDEX.md"

function Test-MarkOSApi {
  try {
    $r = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
    return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300)
  } catch { return $false }
}

Write-Host ""
Write-Host "  ========================================"
Write-Host "   MARK HERE!  --  the-truth bridge"
Write-Host "   Same MarkOS Second Brain as ARMY"
Write-Host "  ========================================"
Write-Host ""
Write-Host "  Soul (canonical): $SoulFile"
Write-Host "  Doctrine pack:    $Doctrine"
Write-Host "  Chat:             $ChatUrl"
Write-Host ""

if (-not (Test-Path $ArmySystem)) {
  Write-Host "  ERROR: ARMY 01_SYSTEM not found at $ArmySystem"
  Write-Host "  Soul match requires ARMY on this machine."
  Write-Host "  Press any key..."
  $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
  exit 1
}

if (-not (Test-MarkOSApi)) {
  Write-Host "  Second Brain offline -- starting Army service..."
  if (Test-Path $StartScript) {
    try { & $StartScript } catch { Write-Host "  start failed: $_" }
  } else {
    Write-Host "  Missing start_service.ps1"
  }
  $ready = $false
  for ($i = 1; $i -le 25; $i++) {
    Start-Sleep -Seconds 1
    if (Test-MarkOSApi) { $ready = $true; break }
    Write-Host "  waiting for chat... ($i/25)"
  }
  if (-not $ready) {
    Write-Host "  Chat not ready. Open when up: $ChatUrl"
    Write-Host "  Press any key..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
  }
}

Write-Host "  Opening MarkOS Second Brain..."
Start-Process $ChatUrl
Write-Host "  Talk to Mark (same soul as ARMY)."
Start-Sleep -Seconds 2
