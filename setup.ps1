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

# Refresh this session's PATH from the registry (so a freshly-installed Python
# is visible without reopening the terminal).
function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = @($machine, $user | Where-Object { $_ }) -join ';'
}

# Install Python 3.12 per-user (no admin): prefer winget, fall back to python.org.
function Install-Python {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "==> Installing Python 3.12 via winget (per-user)..."
        & winget install --id Python.Python.3.12 -e --silent --scope user `
            --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host "    winget didn't succeed; trying the python.org installer..."
    } else {
        Write-Host "==> winget not available; downloading the python.org installer..."
    }
    $ver = "3.12.8"
    $url = "https://www.python.org/ftp/python/$ver/python-$ver-amd64.exe"
    $exe = Join-Path $env:TEMP "python-$ver-amd64.exe"
    try {
        Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    } catch {
        Write-Host "    Download failed: $($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
    Write-Host "==> Running the Python installer (per-user, silent)..."
    $proc = Start-Process -FilePath $exe -Wait -PassThru -ArgumentList @(
        '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_pip=1', 'Include_launcher=1'
    )
    Remove-Item $exe -ErrorAction SilentlyContinue
    return ($proc.ExitCode -eq 0)
}

Write-Host "==> Looking for Python 3.11+"
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.11+ was not found on this PC."
    $doInstall = $true
    if ([Environment]::UserInteractive) {
        $ans = Read-Host "Install Python 3.12 now? (per-user, no admin needed) [Y/n]"
        if ($ans -and $ans.Trim().ToLower().StartsWith('n')) { $doInstall = $false }
    }
    if ($doInstall -and (Install-Python)) {
        Update-SessionPath
        $py = Find-Python
    }
}
if (-not $py) {
    Write-Host ""
    Write-Host "ERROR: Python 3.11 or newer is required and isn't available yet." -ForegroundColor Red
    Write-Host "       If it was just installed, CLOSE this window, open a new one, and"
    Write-Host "       re-run setup (double-click setup.bat). Otherwise install it from"
    Write-Host "       https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH')."
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

# Optional: play the chime on this machine's own speakers (vs. only a Google Home).
$wantLocalAudio = $true
if ([Environment]::UserInteractive) {
    $a = Read-Host "Install local PC audio output (play the chime on this machine's speakers)? [Y/n]"
    if ($a -and $a.Trim().ToLower().StartsWith('n')) { $wantLocalAudio = $false }
}
if ($wantLocalAudio) {
    & $vpy -m pip install playsound3
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    (playsound3 install failed - the 'This PC (local audio)' option will be unavailable)"
    }
}

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
Write-Host "  Start the app:  double-click start.bat"
Write-Host "      (or run:  .\venv\Scripts\python.exe run.py)"
Write-Host ""
Write-Host "  Then open the GUI in a browser on the same WiFi:"
Write-Host ("      http://{0}:8080" -f $ip)
Write-Host ""
Write-Host "  First run: Windows Firewall will ask to allow Python on the"
Write-Host "  network - click Allow (tick Private networks) so the web page"
Write-Host "  and your Google Home can reach the app."
Write-Host "============================================================"
