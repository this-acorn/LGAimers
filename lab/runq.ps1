# Serial experiment queue with stall detection.
# ASCII-ONLY: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI; non-ASCII
# text breaks the parse and the process dies silently.
#
# Fixes over the previous chain (which lost 16 hours on 08-25):
#   1) Launch python.exe DIRECTLY. The old chain launched `py` (the launcher),
#      set AboveNormal on the launcher, and the real python.exe child stayed at
#      Normal priority.
#   2) Set High priority on the actual worker.
#   3) Heartbeat: sample worker CPU every 5 min into lab/heartbeat.txt so a stall
#      is visible within minutes instead of overnight.
#
# Usage: powershell -ExecutionPolicy Bypass -File lab\runq.ps1
$ErrorActionPreference = 'Continue'
$root = 'c:\Users\gwonn\Desktop\open'
$pyexe = 'C:\Users\gwonn\AppData\Local\Programs\Python\Python312\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$logf = "$root\lab\chain_log.txt"
$hb = "$root\lab\heartbeat.txt"

function Note($m) {
    Add-Content -Path $logf -Value ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m)
}

# keep the machine awake for the whole queue
Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
[W.P]::SetThreadExecutionState(0x80000003) | Out-Null   # CONTINUOUS|SYSTEM|DISPLAY

Note 'runq started'
# Reordered 08-25 13:00 after exp/74 (seed-scaling audit).
# exp/67 demoted: the lr0.04 gain on multiclass is variance-reduction, not quality.
# S_inf(MC)=859.1 vs S_inf(MC04)=859.0 -> identical true model. At 8 seeds the gap is
# only +3.8 local (~+5 LB), so 5 hours of the only machine buys almost nothing.
# Measure the untested axes first, then deploy the winning config once.
$queue = @('69_drift_axis', '72_cs_extend', '67_train_mc04')

foreach ($n in $queue) {
    $num = $n.Split('_')[0]
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
}
Note 'queue finished'
