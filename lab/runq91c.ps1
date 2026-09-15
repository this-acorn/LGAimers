# Chain runner (self-sufficient): waits for exp/91b arm D to finish (or takes over if the
# runq91 queue died), then runs
#   exp/91c_trackman_arrival.py  ->  exp/91b --arm T  ->  exp/91b --arm T0
# ASCII-ONLY: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI; non-ASCII
# text breaks the parse and the process dies silently.
# Usage: powershell -ExecutionPolicy Bypass -File lab\runq91c.ps1   (registered as scheduled task 'runq91c')
$ErrorActionPreference = 'Continue'
$root = 'c:\Users\gwonn\Desktop\open'
$pyexe = 'C:\Users\gwonn\AppData\Local\Programs\Python\Python312\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$logf = "$root\lab\chain_log.txt"
$hb = "$root\lab\heartbeat.txt"
$boot = "$root\lab\runq91c_boot.txt"

function Mark($tag) {
    try { ("{0} {1} pid {2}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $tag, $PID) | Out-File -FilePath $boot -Append -Encoding ascii -ErrorAction Stop } catch { }
}
Mark 'boot'
try { Start-Transcript -Path "$root\lab\runq91c_transcript.txt" -Append -ErrorAction Stop | Out-Null } catch { Mark ("transcript-fail " + $_.Exception.Message) }

$mylog = "$root\lab\runq91c_log.txt"
function Note($m) {
    $line = ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m)
    # primary: runner-specific log via Out-File (Add-Content to the shared chain_log fails under the task context)
    try { $line | Out-File -FilePath $mylog -Append -Encoding ascii -ErrorAction Stop } catch { Mark ("note-fail " + $m) }
    try { $line | Out-File -FilePath $logf -Append -Encoding ascii -ErrorAction Stop } catch { }
}

function Finished($f) {
    if (-not (Test-Path $f)) { return $false }
    return (Select-String -Path $f -Pattern 'proj8' -Quiet)
}

function PyRunning() {
    return ((Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
}

try {
    Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);' -ErrorAction Stop
    [W.P]::SetThreadExecutionState([uint32]2147483651) | Out-Null
    Mark 'addtype-ok'
} catch {
    Mark ("addtype-fail " + $_.Exception.Message)
}

function KeepAwake() {
    try { [W.P]::SetThreadExecutionState([uint32]2147483651) | Out-Null } catch { }
}

function RunStep($label, $argv, $outf, $errf) {
    Note "start $label"
    $p = Start-Process -FilePath $pyexe -ArgumentList $argv -WorkingDirectory $root -RedirectStandardOutput $outf -RedirectStandardError $errf -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 5
    try { $p.PriorityClass = 'High' } catch { }
    Note ("  $label PID {0} priority {1}" -f $p.Id, $p.PriorityClass)
    $lastCpu = 0.0
    $stall = 0
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 300
        KeepAwake
        if ($p.HasExited) { break }
        $p.Refresh()
        $cpu = $p.CPU
        $delta = $cpu - $lastCpu
        $lastCpu = $cpu
        $line = "{0} $label cpu_total={1:N1}min delta_5min={2:N1}min ram={3:N2}GB" -f (Get-Date -Format 'MM-dd HH:mm:ss'), ($cpu/60), ($delta/60), ($p.WorkingSet64/1GB)
        Add-Content -Path $hb -Value $line -ErrorAction SilentlyContinue
        if ($delta -lt 30) { $stall++ } else { $stall = 0 }
        if ($stall -ge 3) { Note "  WARNING $label appears stalled (3 x 5min with <30s cpu)"; $stall = 0 }
    }
    Note "done $label exit=$($p.ExitCode)"
    return $p.ExitCode
}

$fB = "$root\lab\91b_B_result.txt"
$fD = "$root\lab\91b_D_result.txt"
Note 'runq91c started (waiting for exp/91b arm D; takes over if runq91 dies)'
Mark 'note-ok'
$idle = 0
while ($true) {
    if (Finished $fD) { Note 'arm D finished by runq91 -> chain continues'; break }
    if (PyRunning) { $idle = 0 } else { $idle++ }
    # runq91 polls every 300s; only declare it dead after 8 idle minutes
    if ($idle -ge 8) {
        Note 'no python for 8 min and arm D not finished -> runq91 presumed dead, taking over'
        if (-not (Finished $fB)) { RunStep 'exp/91b arm=B (takeover)' @('-u', 'exp/91b_intent_ablation.py', '--arm', 'B') $fB "$root\lab\91b_B_err.txt" | Out-Null }
        if (-not (Finished $fD)) { RunStep 'exp/91b arm=D (takeover)' @('-u', 'exp/91b_intent_ablation.py', '--arm', 'D') $fD "$root\lab\91b_D_err.txt" | Out-Null }
        break
    }
    Start-Sleep -Seconds 60
    KeepAwake
}
Start-Sleep -Seconds 30

$rc = RunStep 'exp/91c' @('-u', 'exp/91c_trackman_arrival.py') "$root\lab\91c_result.txt" "$root\lab\91c_err.txt"
# ExitCode from Start-Process -PassThru can come back $null after exit; judge by the output artifact instead
if (-not (Test-Path "$root\lab\91c_feats.npz")) { Note 'exp/91c FAILED (no npz) - skipping arms T/T0'; Note 'runq91c finished'; Mark 'end-91c-fail'; exit 1 }
RunStep 'exp/91b arm=T' @('-u', 'exp/91b_intent_ablation.py', '--arm', 'T') "$root\lab\91b_T_result.txt" "$root\lab\91b_T_err.txt" | Out-Null
RunStep 'exp/91b arm=T0' @('-u', 'exp/91b_intent_ablation.py', '--arm', 'T0') "$root\lab\91b_T0_result.txt" "$root\lab\91b_T0_err.txt" | Out-Null
Note 'runq91c finished'
Mark 'end-ok'
