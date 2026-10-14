# nllb.cpp on a Snapdragon X Elite (Adreno X1-85) - the QDC acceptance run.
#
# Unzip nllb-xelite.zip anywhere on the device and run, from PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 -Model C:\m\nllb-600M-f16.gguf -Ref C:\m\ref
#
# Everything lands in .\results\ (plus results.zip) - copy that back with scp
# or the QDC file browser BEFORE the session ends: QDC wipes the device.
#
# The zip holds TWO builds of the same commit, on purpose:
#   cpu\     no device backend at all. Always starts, even if vulkan-1.dll is
#            missing, and is the reference every GPU number is measured against.
#   vulkan\  -DNLLB_VULKAN=ON   (vulkan-1.dll comes from the Adreno driver)
#
# Sections, in the order that each one's answer is assumed by the next:
#   0  environment      is vulkan-1.dll there, what the CPU and GPU are
#   1  probe            nllb-probe-kv-ops: the 4 KV-cache ops, device vs CPU
#   2  registry         --list-backends from each build
#   3  CPU bit-identity this device's CPU path vs the CI runner's, same binary
#   4  cache gate       nllb-test-cache ON each GPU backend: full = exact
#   5  cross-backend    nllb-test-precision cpu:<gpu> - exact ids + margins,
#                       and nllb-test-batch on the GPU
#   6  states diff      nllb-dump-states on each GPU vs this CPU (a report)
#   7  real model       only with -Model/-Ref: the same gates + text + speed
param(
    [string] $Model   = "",          # real nllb-600M-f16.gguf (optional)
    [string] $Ref     = "",          # its ref\ directory from the Mac (optional)
    [int]    $Threads = 8,
    [string[]] $Backends = @("vulkan"),
    [switch] $SkipFixture
)

$ErrorActionPreference = "Continue"
# Windows PowerShell 5.1 pipes text to native programs as ASCII by default and
# decodes their stdout with the OEM code page: both would mangle every
# non-ASCII byte of a translation (step 15 met this as mojibake). UTF-8 both ways.
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$Root = $PSScriptRoot
$Out  = Join-Path $Root "results"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Start-Transcript -Path (Join-Path $Out "run-log.txt") -Force | Out-Null

$Summary = New-Object System.Collections.Generic.List[string]
function Note([string] $line) { $Summary.Add($line); Write-Host $line }

# Run one exe, tee stdout+stderr to a log file, return the exit code. stderr
# matters: ggml explains a dropped device there and nowhere else.
function Run([string] $name, [string] $exe, [string[]] $argv, [hashtable] $envs = @{}) {
    $log = Join-Path $Out "$name.txt"
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    Write-Host "$exe $($argv -join ' ')"
    if (-not (Test-Path $exe)) { Note "  [$name] MISSING $exe"; return 127 }
    $saved = @{}
    foreach ($k in $envs.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $envs[$k]) }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $exe @argv *>&1 | Tee-Object -FilePath $log | Out-Host
    $rc = $LASTEXITCODE
    $sw.Stop()
    foreach ($k in $envs.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    # An exit code that is not a number means the process never started -
    # almost always a missing DLL (0xC0000135 = -1073741515).
    if ($rc -eq -1073741515) { Note "  [$name] DID NOT START - a DLL is missing (0xC0000135). See section 0." }
    Write-Host ("  -> exit {0}, {1:N1} s" -f $rc, $sw.Elapsed.TotalSeconds)
    return $rc
}

function Verdict([string] $name, [int] $rc, [string] $claim) {
    $s = if ($rc -eq 0) { "PASS" } else { "FAIL (exit $rc)" }
    Note ("  {0,-34} {1,-16} {2}" -f $name, $s, $claim)
}

$Fx16 = Join-Path $Root "models\shape-small-f16.gguf"
$Fx32 = Join-Path $Root "models\shape-small-f32.gguf"
$FxRef = Join-Path $Root "ref-shape"   # no trailing separator: the gates append "/..." themselves
$T = "$Threads"

# ---------------------------------------------------------------- 0
Write-Host "`n=== 0. environment ===" -ForegroundColor Cyan
$sys = "$env:WINDIR\System32"
foreach ($dll in @("vulkan-1.dll")) {
    $p = Join-Path $sys $dll
    if (Test-Path $p) { Note ("  {0,-14} {1}  ({2})" -f $dll, "present", (Get-Item $p).VersionInfo.FileVersion) }
    else              { Note ("  {0,-14} ABSENT from System32" -f $dll) }
}
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List | Out-String | Tee-Object -FilePath (Join-Path $Out "env.txt") | Out-Host
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate | Format-List | Out-String | Tee-Object -FilePath (Join-Path $Out "env.txt") -Append | Out-Host
Get-Content (Join-Path $Root "BUILD-INFO.txt") -ErrorAction SilentlyContinue | Tee-Object -FilePath (Join-Path $Out "env.txt") -Append | Out-Host

# ---------------------------------------------------------------- 1
# The four KV-cache ops alone, device vs CPU, no model. Every row should be
# ok or half (half = the GPU ran an f16 matmul kernel, trap 48). A BAD row in
# "mul_mat(K view, q)" is trap 55 (a gapped batch stride ignored) coming back
# on this driver: then section 4's retry with NLLB_KV_CONT=1 is the workaround.
foreach ($b in $Backends) {
    $rc = Run "1-probe-kv-ops-$b" "$Root\$b\nllb-probe-kv-ops.exe" @($b)
    $script:probeRc = $rc
}

# ---------------------------------------------------------------- 2
foreach ($b in @("cpu") + $Backends) {
    Run "2-list-backends-$b" "$Root\$b\nllb-cli.exe" @("--list-backends") | Out-Null
}

Note "`n----- SUMMARY -----"
if ($Backends.Count -gt 0) { Verdict "kv-op probe vulkan" $script:probeRc "4 KV-cache ops vs CPU, no BAD row" }

if (-not $SkipFixture) {
    # ------------------------------------------------------------ 3
    # Same binary as the CI runner ran (cpu\), different Arm64 CPU. Expected
    # IDENTICAL: GGML_NATIVE=OFF and one -march, so no runtime dispatch can
    # pick a different kernel here. If this is not identical, no GPU number
    # below means anything yet.
    foreach ($p in @("f16", "f32")) {
        $fx = if ($p -eq "f16") { $Fx16 } else { $Fx32 }
        $mine = Join-Path $Out "xelite-cpu-$p.bin"
        Run "3-dump-cpu-$p" "$Root\cpu\nllb-dump-states.exe" @($fx, $FxRef, $mine, "-t", $T) | Out-Null
        $base = Join-Path $Root "baseline\runner-cpu-$p.bin"
        if (Test-Path $base) {
            $rc = Run "3-diff-cpu-$p" "$Root\cpu\nllb-dump-states.exe" @("--diff", $base, $mine)
            Verdict "cpu ${p}: device vs CI runner" $rc "same binary, same bytes"
        }
    }

    foreach ($b in $Backends) {
        $bin = "$Root\$b"
        # -------------------------------------------------------- 4
        foreach ($p in @("f32", "f16")) {
            $fx = if ($p -eq "f16") { $Fx16 } else { $Fx32 }
            $rc = Run "4-cache-$b-$p" "$bin\nllb-test-cache.exe" @($fx, $FxRef, "-t", $T) @{ NLLB_BACKEND = $b; NLLB_ARENA_TRACE = "1" }
            Verdict "cache gate $b $p" $rc "full = exact, [arena] exact (traps 15/16)"
            if ($rc -ne 0) {
                # Save a second session: try the one workaround this project
                # knows about, flipped the other way, and record which it was.
                $alt = "1"
                $rc2 = Run "4-cache-$b-$p-KV_CONT$alt" "$bin\nllb-test-cache.exe" @($fx, $FxRef, "-t", $T) @{ NLLB_BACKEND = $b; NLLB_KV_CONT = $alt }
                Verdict "  ...with NLLB_KV_CONT=$alt" $rc2 "does the K-view copy change the verdict?"
            }
        }
        # -------------------------------------------------------- 5
        foreach ($p in @("f32", "f16")) {
            $fx = if ($p -eq "f16") { $Fx16 } else { $Fx32 }
            $rc = Run "5-precision-cpu-$b-$p" "$bin\nllb-test-precision.exe" @($fx, $fx, $FxRef, "-t", $T, "--backend", "cpu:$b")
            Verdict "precision cpu:$b $p" $rc "exact ids + margin audit"
        }
        $rc = Run "5-batch-$b-f16" "$bin\nllb-test-batch.exe" @($Fx16, "-t", $T, "--backend", $b)
        Verdict "batch gate $b f16" $rc "[equal]/[perm]/[mask] bit-identical"
        # -------------------------------------------------------- 6
        # A REPORT, not a gate: a GPU is not expected to be bit-identical to
        # the CPU (trap 48 - the kernel decides the precision). The size of the
        # difference, and whether it splits by sequence length, is the finding.
        $gd = Join-Path $Out "xelite-$b-f32.bin"
        Run "6-dump-$b-f32" "$bin\nllb-dump-states.exe" @($Fx32, $FxRef, $gd, "-t", $T, "--backend", $b) | Out-Null
        Run "6-diff-cpu-vs-$b-f32" "$Root\cpu\nllb-dump-states.exe" @("--diff", (Join-Path $Out "xelite-cpu-f32.bin"), $gd) | Out-Null
        # -------------------------------------------------------- speed
        Run "6-bench-$b-fixture" "$bin\nllb-bench.exe" @($Fx16, "-t", $T, "--src", "24", "--steps", "32") @{ NLLB_BACKEND = $b } | Out-Null
    }
}

# ---------------------------------------------------------------- 7
if ($Model -and (Test-Path $Model)) {
    $RefDir = if ($Ref) { $Ref.TrimEnd("\", "/") } else { "" }
    $sentences = Join-Path $Root "sentences.txt"
    Run "7-cpu-hello" "$Root\cpu\nllb-cli.exe" @("-m", $Model, "--tgt", "deu_Latn", "--backend", "cpu", "-t", $T, "-v", "Hello, world!") | Out-Null
    foreach ($b in $Backends) {
        $bin = "$Root\$b"
        Run "7-$b-hello" "$bin\nllb-cli.exe" @("-m", $Model, "--tgt", "deu_Latn", "--backend", $b, "-t", $T, "-v", "Hello, world!") | Out-Null
        if ($RefDir -and (Test-Path $RefDir)) {
            $rc = Run "7-cache-$b-real" "$bin\nllb-test-cache.exe" @($Model, $RefDir, "-t", $T) @{ NLLB_BACKEND = $b; NLLB_ARENA_TRACE = "1" }
            Verdict "REAL cache gate $b" $rc "full = exact on the 600M"
            $rc = Run "7-precision-cpu-$b-real" "$bin\nllb-test-precision.exe" @($Model, $Model, $RefDir, "-t", $T, "--backend", "cpu:$b")
            Verdict "REAL precision cpu:$b" $rc "exact ids 9/9 + margins"
        }
        # text -> text, CPU vs GPU, byte for byte (no Python on the device, so
        # this stands in for verify_translation.py's [stdin] pass).
        $cpuOut = Join-Path $Out "7-cpu-translate.stdout.txt"
        $gpuOut = Join-Path $Out "7-$b-translate.stdout.txt"
        # (PowerShell has no `<` redirect; pipe the file in instead.)
        Get-Content -Encoding utf8 $sentences | & "$Root\cpu\nllb-cli.exe" -m $Model --tgt deu_Latn --backend cpu -t $T 2>$null | Set-Content -Encoding utf8 $cpuOut
        Get-Content -Encoding utf8 $sentences | & "$bin\nllb-cli.exe"      -m $Model --tgt deu_Latn --backend $b  -t $T 2>$null | Set-Content -Encoding utf8 $gpuOut
        # Two EMPTY outputs are byte-identical too - which is how the first
        # rehearsal of this script passed with a CLI that had translated
        # nothing. Every line must come back, non-empty, before equality means
        # anything (the project's "did the process do the work" rule).
        $nIn  = @(Get-Content -Encoding utf8 $sentences).Count
        $cpuL = @(Get-Content -Encoding utf8 $cpuOut | Where-Object { $_.Trim() })
        $gpuL = @(Get-Content -Encoding utf8 $gpuOut | Where-Object { $_.Trim() })
        $same = ($cpuL.Count -eq $nIn) -and ($gpuL.Count -eq $nIn) -and
                ((Get-FileHash $cpuOut).Hash -eq (Get-FileHash $gpuOut).Hash)
        Verdict "REAL text cpu vs $b" ([int](-not $same)) ("stdout byte-identical ({0}/{1}/{2} lines in/cpu/{3})" -f $nIn, $cpuL.Count, $gpuL.Count, $b)
        Run "7-bench-$b-real" "$bin\nllb-bench.exe" @($Model, "-t", $T, "--src", "24", "--steps", "64") @{ NLLB_BACKEND = $b; NLLB_ARENA_TRACE = "1" } | Out-Null
    }
    Run "7-bench-cpu-real" "$Root\cpu\nllb-bench.exe" @($Model, "-t", $T, "--src", "24", "--steps", "64") | Out-Null
}

Write-Host ""
$Summary | Set-Content -Encoding utf8 (Join-Path $Out "SUMMARY.txt")
$Summary | Out-Host
Stop-Transcript | Out-Null
Compress-Archive -Force -Path "$Out\*" -DestinationPath (Join-Path $Root "results.zip")
Write-Host "`nCopy results.zip off the device now - QDC wipes it when the session ends." -ForegroundColor Yellow
