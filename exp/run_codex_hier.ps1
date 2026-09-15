param(
    [int]$WaitPid = 38132,
    [double]$MinFreeGB = 6.0
)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\gwonn\Desktop\open'
$python = 'C:\Users\gwonn\AppData\Local\Programs\Python\Python312\python.exe'
$queueLog = Join-Path $repo 'lab\codex_hier_queue.log'

function Write-QueueLog([string]$Message) {
    $line = "$(Get-Date -Format 'MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}

Set-Location -LiteralPath $repo
Write-QueueLog "queued; waiting for Claude-owned PID=$WaitPid and free memory >= ${MinFreeGB}GB"

while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}

while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = $os.FreePhysicalMemory / 1MB
    if ($freeGB -ge $MinFreeGB) { break }
    Write-QueueLog ("Claude PID exited, but free memory is {0:N2}GB; still waiting" -f $freeGB)
    Start-Sleep -Seconds 15
}

Write-QueueLog ("starting smoke with {0:N2}GB free" -f $freeGB)
& $python -u exp/codex_hier_head.py --seed 42 --smoke
if ($LASTEXITCODE -ne 0) {
    Write-QueueLog "smoke failed exit=$LASTEXITCODE; full run cancelled"
    exit $LASTEXITCODE
}

Write-QueueLog "smoke passed; starting full seed42"
& $python -u exp/codex_hier_head.py --seed 42
$code = $LASTEXITCODE
Write-QueueLog "full seed42 exited code=$code"
exit $code
