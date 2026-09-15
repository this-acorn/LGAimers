# Runner for exp/100_champ_arms.py arms (default C6), sequential, with keep-awake + heartbeat.
# ASCII-ONLY: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI; non-ASCII text breaks the parse.
# Usage: powershell -ExecutionPolicy Bypass -File lab\runq100.ps1 [-Arms C6,HT,IT]
# Success of an arm is judged by the presence of lab/100_<arm>.npy (not ExitCode).
param([string[]]$Arms = @('C6'))
$ErrorActionPreference = 'Continue'
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
$root = 'c:\Users\gwonn\Desktop\open'
$pyexe = 'C:\Users\gwonn\AppData\Local\Programs\Python\Python312\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$logf = "$root\lab\chain_log.txt"
$hb = "$root\lab\heartbeat.txt"

function Note($m) {
    $line = ("[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m)
    try { Add-Content -Path $logf -Value $line -ErrorAction Stop } catch { $line | Out-File -FilePath $logf -Append -Encoding ascii }
}

Add-Type -Name P -Namespace W -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
[W.P]::SetThreadExecutionState([uint32]2147483651) | Out-Null

Note ("runq100 started arms=" + ($Arms -join ','))
foreach ($arm in $Arms) {
    $outNpy = "$root\lab\100_${arm}.npy"
    if (Test-Path $outNpy) { Note "skip exp/100 arm=$arm (lab/100_${arm}.npy exists)"; continue }
    $other = Get-Process python -ErrorAction SilentlyContinue
    if ($other) { Note ("  WARNING another python is running (PID " + ($other.Id -join ',') + ") - two trainings at once break thread pairing") }
    $bat = Get-CimInstance Win32_Battery
    Note ("start exp/100 arm=$arm  (battery status {0}, charge {1}%)" -f $bat.BatteryStatus, $bat.EstimatedChargeRemaining)
    $p = $null
    $p = Start-Process -FilePath $pyexe -ArgumentList '-u', 'exp/100_champ_arms.py', '--arm', $arm -WorkingDirectory $root -RedirectStandardOutput "$root\lab\100_${arm}_result.txt" -RedirectStandardError "$root\lab\100_${arm}_err.txt" -PassThru -WindowStyle Hidden
    if (-not $p) { Note "  exp/100 arm=$arm FAILED to start (Start-Process returned null)"; continue }
    Start-Sleep -Seconds 5
    try { $p.PriorityClass = 'High' } catch { }
    Note ("  exp/100 arm=$arm PID {0} priority {1}" -f $p.Id, $p.PriorityClass)
    $lastCpu = 0.0
    $stall = 0
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 300
        [W.P]::SetThreadExecutionState([uint32]2147483651) | Out-Null
        if ($p.HasExited) { break }
        $p.Refresh()
        $cpu = $p.CPU
        $delta = $cpu - $lastCpu
        $lastCpu = $cpu
        $bat = Get-CimInstance Win32_Battery
        $line = "{0} exp/100 arm=$arm cpu_total={1:N1}min delta_5min={2:N1}min ram={3:N2}GB bat={4}/{5}%" -f (Get-Date -Format 'MM-dd HH:mm:ss'), ($cpu/60), ($delta/60), ($p.WorkingSet64/1GB), $bat.BatteryStatus, $bat.EstimatedChargeRemaining
        try { Add-Content -Path $hb -Value $line -ErrorAction Stop } catch { $line | Out-File -FilePath $hb -Append -Encoding ascii }
        if ($delta -lt 30) { $stall++ } else { $stall = 0 }
        if ($stall -ge 3) { Note "  WARNING exp/100 arm=$arm appears stalled (3 x 5min with <30s cpu)"; $stall = 0 }
    }
    if (Test-Path $outNpy) { Note "done exp/100 arm=$arm OK (npy present)" } else { Note "done exp/100 arm=$arm FAILED (no npy) - see lab/100_${arm}_err.txt" }
}
Note 'runq100 finished'
