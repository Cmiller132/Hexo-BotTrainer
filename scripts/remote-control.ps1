<#
  remote-control.ps1  —  Launch a Claude Code "Remote Control" server session
  so you can drive THIS machine's Claude Code from your phone / browser.

  WHAT THIS IS:
    Remote Control is Anthropic's official, supported way to continue a LOCAL
    Claude Code session from claude.ai/code or the Claude mobile app. Claude
    keeps running on THIS machine; the phone is just a window into it.

  SECURITY (important, read once):
    - This makes OUTBOUND HTTPS connections only. It opens NO inbound ports on
      this machine and exposes NOTHING to the public internet. Traffic goes
      through the Anthropic API over TLS with short-lived, scoped credentials.
    - It is NOT an SSH server or a tunnel. You do NOT need port-forwarding,
      ngrok, or to weaken your firewall.

  SAFE BY DESIGN re: the training run:
    - This launches a separate Claude Code CLI process. It does NOT touch the
      WSL dense_cnn_restnet training run and starts nothing related to it.
    - It runs in whatever directory you pass with -Dir (default: this repo).

  REQUIREMENTS:
    - Claude Code v2.1.51+ (you have 2.1.170).  Push notifications need 2.1.110+.
    - A claude.ai subscription (Pro/Max/Team/Enterprise). API keys do NOT work.
    - You must be logged in via claude.ai OAuth (run once: see step 1 below).

  USAGE:
    pwsh -File scripts\remote-control.ps1                # server mode in this repo
    pwsh -File scripts\remote-control.ps1 -Dir C:\some\other\project
    pwsh -File scripts\remote-control.ps1 -Interactive   # local terminal + remote at once
    pwsh -File scripts\remote-control.ps1 -Name "Hexo box"
#>
[CmdletBinding()]
param(
    [string]$Dir = (Split-Path -Parent $PSScriptRoot),   # default = repo root
    [string]$Name = "$env:COMPUTERNAME hexo",
    [switch]$Interactive                                   # use --remote-control instead of server mode
)

$ErrorActionPreference = 'Stop'

# Resolve the bundled Claude Code CLI (managed install path).
$cli = Get-ChildItem "$env:APPDATA\Claude\claude-code" -Directory -ErrorAction SilentlyContinue |
       Sort-Object Name -Descending |
       ForEach-Object { Join-Path $_.FullName 'claude.exe' } |
       Where-Object { Test-Path $_ } |
       Select-Object -First 1
if (-not $cli) {
    if (Get-Command claude -ErrorAction SilentlyContinue) { $cli = (Get-Command claude).Source }
    else { throw "Could not find claude.exe. Open the Claude Desktop app once, or install the CLI." }
}
Write-Host "Using CLI: $cli" -ForegroundColor DarkGray
& $cli --version

# Step 1: ensure a full-scope claude.ai login (Remote Control rejects API keys
# and the inference-only setup-token). This is interactive (opens a browser) the
# FIRST time only; after that the stored login is reused.
$status = (& $cli auth status 2>&1 | Out-String)
if ($status -notmatch '"loggedIn"\s*:\s*true') {
    Write-Host "`nNot logged in to claude.ai for the terminal CLI." -ForegroundColor Yellow
    Write-Host "Launching 'claude auth login' (choose the claude.ai option, NOT an API key)..." -ForegroundColor Yellow
    if ($env:ANTHROPIC_API_KEY) {
        Write-Host "WARNING: ANTHROPIC_API_KEY is set; unset it first or Remote Control will be refused." -ForegroundColor Red
    }
    & $cli auth login
}

if (-not (Test-Path $Dir)) { throw "Directory not found: $Dir" }
Set-Location $Dir
Write-Host "`nStarting Remote Control in: $Dir" -ForegroundColor Cyan
Write-Host "When it's up: press SPACE for a QR code, scan it with the Claude mobile app," -ForegroundColor Cyan
Write-Host "or open https://claude.ai/code and pick this session (computer icon, green dot)." -ForegroundColor Cyan
Write-Host "Press Ctrl+C here to end the remote session.`n" -ForegroundColor Cyan

if ($Interactive) {
    & $cli --remote-control $Name
} else {
    & $cli remote-control --name $Name
}
