# Runner for exp/89_cat5_val.py (champion-config validation for error analysis).
# ASCII-ONLY: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI; non-ASCII
# text breaks the parse and the process dies silently.
# Usage: powershell -ExecutionPolicy Bypass -File lab\runq89.ps1
$ErrorActionPreference = 'Continue'
$root = 'c:\Users\gwonn\Desktop\open'
$pyexe = 'C:\Users\gwonn\AppData\Local\Programs\Python\Python312\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$logf = "$root\lab\chain_log.txt"
$hb = "$root\lab\heartbeat.txt"

function Note($m) {
    Add-Content -Path $logf -Value ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m)
}

Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
[W.P]::SetThreadExecutionState(0x80000003) | Out-Null

Note 'runq89 started'
$n = '89_cat5_val'
$num = '89'
Note "start exp/$n"
$p = Start-Process -FilePath $pyexe -ArgumentList '-u', "exp/$n.py" -WorkingDirectory $root -RedirectStandardOutput "$root\lab\${num}_result.txt" -RedirectStandardError "$root\lab\${num}_err.txt" -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 5
try { $p.PriorityClass = 'High' } catch { }
Note ("  exp/$n PID {0} priority {1}" -f $p.Id, $p.PriorityClass)

$lastCpu = 0.0
$stall = 0
while (-not $p.HasExited) {
    Start-Sleep -Seconds 300
    [W.P]::SetThreadExecutionState(0x80000003) | Out-Null
    if ($p.HasExited) { break }
    $p.Refresh()
    $cpu = $p.CPU
    $delta = $cpu - $lastCpu
    $lastCpu = $cpu
    $line = "{0} exp/$n cpu_total={1:N1}min delta_5min={2:N1}min ram={3:N2}GB" -f (Get-Date -Format 'MM-dd HH:mm:ss'), ($cpu/60), ($delta/60), ($p.WorkingSet64/1GB)
    Add-Content -Path $hb -Value $line
    if ($delta -lt 30) { $stall++ } else { $stall = 0 }
    if ($stall -ge 3) { Note "  WARNING exp/$n appears stalled (3 x 5min with <30s cpu)"; $stall = 0 }
}
Note "done exp/$n exit=$($p.ExitCode)"
Note 'runq89 finished'
