Add-Type -Name PW -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
# ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x1) — 시스템 유휴 절전 진입 차단
while ($true) { [Win32.PW]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null; Start-Sleep -Seconds 50 }
