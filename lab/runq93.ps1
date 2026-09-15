# Runner for exp/93_catcombo.py arms (A, B, C, M4 by default), sequential.
# ASCII-ONLY: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI; non-ASCII
# text breaks the parse and the process dies silently.
# Usage: powershell -ExecutionPolicy Bypass -File lab\runq93.ps1 [-Arms A,B,C,M4]
# Success of an arm is judged by the presence of lab/93_<arm>.npy (not ExitCode,
# which can come back null after exit - see HANDOFF 1.14 trap 5).
param([string[]]$Arms = @('A', 'B', 'C', 'M4'))
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

Note ("runq93 started arms=" + ($Arms -join ','))
foreach ($arm in $Arms) {
    $outNpy = "$root\lab\93_${arm}.npy"
    if (Test-Path $outNpy) { Note "skip exp/93 arm=$arm (lab/93_${arm}.npy exists)"; continue }
    Note "start exp/93 arm=$arm"
    $p = $null
    $p = Start-Process -FilePath $pyexe -ArgumentList '-u', 'exp/93_catcombo.py', '--arm', $arm -WorkingDirectory $root -RedirectStandardOutput "$root\lab\93_${arm}_result.txt" -RedirectStandardError "$root\lab\93_${arm}_err.txt" -PassThru -WindowStyle Hidden
    if (-not $p) { Note "  exp/93 arm=$arm FAILED to start (Start-Process returned null)"; continue }
    Start-Sleep -Seconds 5
    try { $p.PriorityClass = 'High' } catch { }
    Note ("  exp/93 arm=$arm PID {0} priority {1}" -f $p.Id, $p.PriorityClass)
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
        $line = "{0} exp/93 arm=$arm cpu_total={1:N1}min delta_5min={2:N1}min ram={3:N2}GB" -f (Get-Date -Format 'MM-dd HH:mm:ss'), ($cpu/60), ($delta/60), ($p.WorkingSet64/1GB)
        try { Add-Content -Path $hb -Value $line -ErrorAction Stop } catch { $line | Out-File -FilePath $hb -Append -Encoding ascii }
        if ($delta -lt 30) { $stall++ } else { $stall = 0 }
        if ($stall -ge 3) { Note "  WARNING exp/93 arm=$arm appears stalled (3 x 5min with <30s cpu)"; $stall = 0 }
    }
    if (Test-Path $outNpy) { Note "done exp/93 arm=$arm OK (npy present)" } else { Note "done exp/93 arm=$arm FAILED (no npy) - see lab/93_${arm}_err.txt" }
}
Note 'runq93 finished'
