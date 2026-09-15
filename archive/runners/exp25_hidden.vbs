' exp/25 콘솔 창 숨김 실행기
' 창이 보이면 실수로 닫히거나 포커스 이벤트로 죽는다 (exp/18에서 0xC000013A로 실제 발생)
Set sh = CreateObject("WScript.Shell")
sh.Run """c:\Users\gwonn\Desktop\open\exp25_runner.cmd""", 0, False
