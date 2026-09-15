$f = "c:\Users\gwonn\Desktop\open\lab\probe_task.txt"
("probe {0} pid {1} user {2}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $PID, $env:USERNAME) | Out-File -FilePath $f -Append -Encoding ascii
