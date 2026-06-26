<#
  Kevin's Cat App - Windows setup (PowerShell).

  From a PowerShell window in this folder, run:

      powershell -ExecutionPolicy Bypass -File .\setup.ps1

  (or just double-click setup.bat, which does that for you).

  Then start the app:

      .\venv\Scripts\python.exe run.py

  Creates a Python virtualenv, installs dependencies, generates the default
  treat chime, and creates config.yaml. No Docker, no services - same as the
  Linux setup.sh.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Find a Python 3.11+ interpreter. Prefer the Windows launcher (py), then python.
function Find-Python {
    $candidates = @(
        @{ Exe = 'py';      Args = @('-3.11') },
        @{ Exe = 'py';      Args = @('-3')    },
        @{ Exe = 'python';  Args = @()        },
        @{ Exe = 'python3'; Args = @()        }
    )
    foreach ($c in $candidates) {
        try {
            $v = & $c.Exe @($c.Args) -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) {
                $p = $v.Trim().Split('.')
                if ([int]$p[0] -gt 3 -or ([int]$p[0] -eq 3 -and [int]$p[1] -ge 11)) {
                    return $c
                }
            }
        } catch { }
    }
    return $null
}

Write-Host "==> Looking for Python 3.11+"
$py = Find-Python
if (-not $py) {
    Write-Host ""
    Write-Host "ERROR: Python 3.11 or newer is required and was not found." -ForegroundColor Red
    Write-Host "       Install it from https://www.python.org/downloads/windows/"
    Write-Host "       (tick 'Add python.exe to PATH'), then re-run this script."
    exit 1
}
Write-Host ("==> Using " + (& $py.Exe @($py.Args) --version))

if (-not (Test-Path "venv")) {
    Write-Host "==> Creating virtualenv in .\venv"
    & $py.Exe @($py.Args) -m venv venv
}

$vpy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "ERROR: venv was not created correctly ($vpy missing)." -ForegroundColor Red
    exit 1
}

Write-Host "==> Installing dependencies (this can take a few minutes the first time)"
& $vpy -m pip install --upgrade pip | Out-Null
& $vpy -m pip install -r requirements.txt

Write-Host "==> Generating the default treat chime"
& $vpy d20app\sounds\generate_chime.py

if (-not (Test-Path "config.yaml")) {
    Write-Host "==> Creating config.yaml from config.example.yaml"
    Copy-Item config.example.yaml config.yaml
} else {
    Write-Host "==> Keeping existing config.yaml"
}

# Detect a LAN IP just for the friendly hint below (reuse the app's own helper).
$ip = & $vpy -c "from d20app.caster import detect_lan_ip; print(detect_lan_ip())"
if (-not $ip) { $ip = "localhost" }

Write-Host ""
Write-Host "============================================================"
Write-Host "  Setup complete!"
Write-Host ""
Write-Host "  Start the app:"
Write-Host "      .\venv\Scripts\python.exe run.py"
Write-Host ""
Write-Host "  Then open the GUI in a browser on the same WiFi:"
Write-Host ("      http://{0}:8080" -f $ip)
Write-Host ""
Write-Host "  First run: Windows Firewall will ask to allow Python on the"
Write-Host "  network - click Allow (tick Private networks) so the web page"
Write-Host "  and your Google Home can reach the app."
Write-Host "============================================================"
