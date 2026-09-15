# Serial experiment queue. Runs after the given PID exits.
# ASCII-ONLY on purpose: Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI,
# and non-ASCII text here breaks the parse (the previous chain died this way).
#
# Queue order rationale:
#   1) exp/69 drift axis (season weight + regularization) -- never measured
#   2) exp/72 CS reverse/breaking re-isolation -- exp/51 rejected the BUNDLE at
#      -24.7, but it bundled a provably redundant column (offspeed, R2=1.000000)
#      plus an unrelated feature, at lr=0.08 which was later shown wrong by +38.5.
#      Lower prior, so it runs second.
#
# Usage: powershell -File lab\chain.ps1 <PID-to-wait-for>
$ErrorActionPreference = 'Continue'
$root = 'c:\Users\gwonn\Desktop\open'
$env:PYTHONIOENCODING = 'utf-8'
$logf = "$root\lab\chain_log.txt"

function Note($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m
    Add-Content -Path $logf -Value $line
}

Note 'chain started'

if ($args.Count -ge 1 -and $args[0] -match '^\d+$') {
    $waitPid = [int]$args[0]
    Note "waiting for PID $waitPid to exit"
    while (Get-Process -Id $waitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Note "PID $waitPid exited"
}

$queue = @('69_drift_axis', '72_cs_extend')

foreach ($n in $queue) {
    $num = $n.Split('_')[0]
    Note "start exp/$n"
    $p = Start-Process -FilePath 'py' -ArgumentList '-3.12', '-u', "exp/$n.py" -WorkingDirectory $root -RedirectStandardOutput "$root\lab\${num}_result.txt" -RedirectStandardError "$root\lab\${num}_err.txt" -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 3
    try { $p.PriorityClass = 'AboveNormal' } catch { }
    Note "  exp/$n PID $($p.Id)"
    $p.WaitForExit()
    Note "done exp/$n exit=$($p.ExitCode)"
}
Note 'queue finished'
