$ErrorActionPreference = 'SilentlyContinue'
$targetProcessId = 42148

Add-Type @'
using System.Runtime.InteropServices;
public static class CodexExecutionState104 {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@

$esContinuous = [Convert]::ToUInt32('80000000', 16)
$esSystemRequired = [Convert]::ToUInt32('00000001', 16)

while ($null -ne (Get-Process -Id $targetProcessId -ErrorAction SilentlyContinue)) {
    [void][CodexExecutionState104]::SetThreadExecutionState($esContinuous -bor $esSystemRequired)
    Start-Sleep -Seconds 30
}

[void][CodexExecutionState104]::SetThreadExecutionState($esContinuous)
