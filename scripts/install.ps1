# Install AI Engineer on Windows into an isolated virtual environment.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Prefix DIR] [-Extras all] [-Source PATH_OR_URL]
param(
    [string]$Prefix = "$env:USERPROFILE\.ai-engineer",
    [string]$Extras = "all",
    [string]$Source = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Stop"

$python = $null
foreach ($candidate in @("py -3.13", "py -3.12", "py -3.11", "python")) {
    $parts = $candidate.Split(" ")
    try {
        & $parts[0] $parts[1..($parts.Length - 1)] -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $python = $parts; break }
    } catch { }
}
if (-not $python) { throw "Python 3.11 or newer is required (https://www.python.org/downloads/)." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Warning "git not found; checkpoints will use file backups and git features are disabled"
}

& $python[0] $python[1..($python.Length - 1)] -m venv "$Prefix\venv"
$venvPython = "$Prefix\venv\Scripts\python.exe"
& $venvPython -m pip install --quiet --upgrade pip
$spec = if ($Extras -eq "none") { $Source } else { "$Source[$Extras]" }
& $venvPython -m pip install --quiet $spec

$aie = "$Prefix\venv\Scripts\aie.exe"
Write-Host "Installed: $(& $aie --version)"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$Prefix\venv\Scripts*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$Prefix\venv\Scripts", "User")
    Write-Host "Added $Prefix\venv\Scripts to your user PATH (open a new terminal)."
}
Write-Host "Next: cd your\project; aie init; aie doctor"
