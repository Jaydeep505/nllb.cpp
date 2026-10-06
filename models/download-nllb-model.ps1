# Download a pre-converted nllb.cpp GGUF on Windows - PowerShell 5.1 or 7.
#
#   .\models\download-nllb-model.ps1                 # f16, the default
#   .\models\download-nllb-model.ps1 q8emb
#   .\models\download-nllb-model.ps1 f16 C:\models
#   .\models\download-nllb-model.ps1 -List
#
# If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\models\download-nllb-model.ps1
#
# Same contract as download-nllb-model.sh: SHA256SUMS first, the file into
# <name>.part, verified, then renamed. Same environment variables:
# NLLB_HF_REPO, NLLB_HF_REV, NLLB_BASE_URL, HF_TOKEN.

param(
    [Parameter(Position = 0)] [string] $Model = "f16",
    [Parameter(Position = 1)] [string] $Dir = $PSScriptRoot,
    [switch] $List
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x slower with it on

$Repo = if ($env:NLLB_HF_REPO) { $env:NLLB_HF_REPO } else { "JaydeepGupta/nllb-200-distilled-600M-GGUF" }
$Rev  = if ($env:NLLB_HF_REV)  { $env:NLLB_HF_REV }  else { "main" }
$Base = if ($env:NLLB_BASE_URL) { $env:NLLB_BASE_URL } else { "https://huggingface.co/$Repo/resolve/$Rev" }

$Models = [ordered]@{
    "f16"   = @("nllb-600M-f16.gguf",   "1.24 GB", "the default. Greedy ids identical to HF on every golden")
    "q8emb" = @("nllb-600M-q8emb.gguf", "0.99 GB", "token_embd at Q8_0, rest f16. 1.33x faster on M1 CPU, chrF 96.57")
    "f32"   = @("nllb-600M-f32.gguf",   "2.46 GB", "reference precision. For the gates and as nllb-quantize input")
}

function Show-Models {
    "{0,-7} {1,-24} {2,-8} {3}" -f "name", "file", "size", "notes"
    foreach ($k in $Models.Keys) { "{0,-7} {1,-24} {2,-8} {3}" -f $k, $Models[$k][0], $Models[$k][1], $Models[$k][2] }
    ""
    "source: $Base"
}

if ($List) { Show-Models; exit 0 }
if (-not $Models.Contains($Model)) {
    Write-Host "unknown model '$Model'`n"
    Show-Models
    exit 2
}

$File = $Models[$Model][0]
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Out  = Join-Path $Dir $File
$Part = "$Out.part"

$Headers = @{}
if ($env:HF_TOKEN) { $Headers["Authorization"] = "Bearer $($env:HF_TOKEN)" }

# curl.exe ships with Windows 10 1803+ and can resume; Invoke-WebRequest cannot.
$Curl = Get-Command curl.exe -ErrorAction SilentlyContinue

function Fetch([string] $Url, [string] $Dest, [bool] $Resume) {
    if ($Curl) {
        $a = @("-fL", "--retry", "3", "-o", $Dest)
        if ($Resume) { $a += @("-C", "-", "--progress-bar") } else { $a += "-sS" }
        if ($env:HF_TOKEN) { $a += @("-H", "Authorization: Bearer $($env:HF_TOKEN)") }
        & $Curl.Source @a $Url
        if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE) on $Url" }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -Headers $Headers -UseBasicParsing
    }
}

# --- checksum first -----------------------------------------------------------
$Sums = [System.IO.Path]::GetTempFileName()
try {
    try { Fetch "$Base/SHA256SUMS" $Sums $false }
    catch { Write-Host "could not fetch $Base/SHA256SUMS - check NLLB_HF_REPO / NLLB_HF_REV"; exit 1 }

    $Expected = $null
    foreach ($line in Get-Content $Sums) {
        $p = $line -split '\s+', 2
        if ($p.Count -eq 2 -and ($p[1] -eq $File -or $p[1] -eq "*$File")) { $Expected = $p[0].ToLower() }
    }
    if ($Expected -notmatch '^[0-9a-f]{64}$') { Write-Host "SHA256SUMS has no entry for $File"; exit 1 }
} finally { Remove-Item -Force $Sums -ErrorAction SilentlyContinue }

if (Test-Path $Out) {
    Write-Host "$Out exists - verifying..."
    if ((Get-FileHash -Algorithm SHA256 $Out).Hash.ToLower() -eq $Expected) { Write-Host "ok, already up to date."; exit 0 }
    # A complete file with the wrong bytes cannot be repaired by resuming.
    Write-Host "checksum differs from $Rev; re-downloading."
    Remove-Item -Force $Out
}

function Test-Part { (Test-Path $Part) -and ((Get-FileHash -Algorithm SHA256 $Part).Hash.ToLower() -eq $Expected) }

Write-Host "downloading $File from $Base"
$ok = $false
if ((Test-Path $Part) -and (Get-Item $Part).Length -gt 0) {
    Write-Host "resuming from $((Get-Item $Part).Length) bytes in $Part"
    # No range support, or a .part from a different upload: one clean retry.
    try { Fetch "$Base/$File" $Part $true; $ok = Test-Part } catch { $ok = $false }
    if (-not $ok) { Write-Host "resume did not produce a valid file; restarting from zero"; Remove-Item -Force $Part -ErrorAction SilentlyContinue }
}
if (-not $ok) {
    try { Fetch "$Base/$File" $Part $true }
    catch { Write-Host "download failed; re-run to resume from $Part"; exit 1 }
    Write-Host "verifying sha256..."
    if (-not (Test-Part)) {
        $Got = (Get-FileHash -Algorithm SHA256 $Part).Hash.ToLower()
        Write-Host "CHECKSUM MISMATCH for $File`n  expected $Expected`n  got      $Got"
        Remove-Item -Force $Part
        exit 1
    }
}
Move-Item -Force $Part $Out

Write-Host "`nsaved $Out`n"
Write-Host "try it:"
Write-Host "  .\build\bin\Release\nllb-cli.exe -m $Out --src eng_Latn --tgt deu_Latn `"Hello, world!`""
