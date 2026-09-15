# Runner for exp/94_train_deploy.py capacity probe (it=1000, 8 seeds, arm=none).
# ASCII-ONLY: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI; non-ASCII breaks the parse.
# Usage: powershell -ExecutionPolicy Bypass -File lab\runq130.ps1
# Owner: Claude. Codex owns exp/103-115; this uses out-dir submit30_src and lab/130_* only.
$ErrorActionPreference = 'Continue'
$root = 'c:\Users\gwonn\Desktop\open'
$pyexe = "$root\venv311\Scripts\python.exe"     # rebuilt 08-30 23:05 (old scratchpad venv311 was corrupted)
$env:PYTHONIOENCODING = 'utf-8'
$logf = "$root\lab\chain_log.txt"
$hb = "$root\lab\heartbeat.txt"

function Note($m) {
    $line = ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m)
    try { Add-Content -Path $logf -Value $line -ErrorAction Stop } catch { $line | Out-File -FilePath $logf -Append -Encoding ascii }
}

Add-Type -Name P130 -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
[W.P130]::SetThreadExecutionState([uint32]2147483651) | Out-Null

$other = Get-Process python -ErrorAction SilentlyContinue
if ($other) { Note ("runq130 ABORT - another python is running (PID " + ($other.Id -join ',') + ")"); exit 1 }

$bat = Get-CimInstance Win32_Battery
Note ("runq130 start: exp/94 --arm none --iterations 1000 --threads 14 --out-dir submit30_src (battery {0}, {1}%)" -f $bat.BatteryStatus, $bat.EstimatedChargeRemaining)
$p = $null
$p = Start-Process -FilePath $pyexe -ArgumentList '-u', 'exp/94_train_deploy.py', '--arm', 'none', '--iterations', '1000', '--threads', '14', '--out-dir', 'submit30_src' -WorkingDirectory $root -RedirectStandardOutput "$root\lab\130_it1000_result.txt" -RedirectStandardError "$root\lab\130_it1000_err.txt" -PassThru -WindowStyle Hidden
if (-not $p) { Note '  runq130 FAILED to start (Start-Process returned null)'; exit 1 }
Start-Sleep -Seconds 5
try { $p.PriorityClass = 'High' } catch { }
Note ("  PID {0} priority {1}" -f $p.Id, $p.PriorityClass)

$lastCpu = 0.0
$stall = 0
while (-not $p.HasExited) {
    Start-Sleep -Seconds 300
    [W.P130]::SetThreadExecutionState([uint32]2147483651) | Out-Null
    if ($p.HasExited) { break }
    $p.Refresh()
    $cpu = $p.CPU
    $delta = $cpu - $lastCpu
    $lastCpu = $cpu
    $bat = Get-CimInstance Win32_Battery
    $line = "{0} exp/94 it1000 cpu_total={1:N1}min delta_5min={2:N1}min ram={3:N2}GB bat={4}/{5}%" -f (Get-Date -Format 'MM-dd HH:mm:ss'), ($cpu/60), ($delta/60), ($p.WorkingSet64/1GB), $bat.BatteryStatus, $bat.EstimatedChargeRemaining
    try { Add-Content -Path $hb -Value $line -ErrorAction Stop } catch { $line | Out-File -FilePath $hb -Append -Encoding ascii }
    if ($delta -lt 30) { $stall++ } else { $stall = 0 }
    if ($stall -ge 3) { Note '  WARNING exp/94 it1000 appears stalled (3 x 5min with <30s cpu)'; $stall = 0 }
}
if (Test-Path "$root\submit30_src\model\model.pkl") { Note 'runq130 done OK (model.pkl present)' } else { Note 'runq130 done FAILED (no model.pkl) - see lab/130_it1000_err.txt' }
