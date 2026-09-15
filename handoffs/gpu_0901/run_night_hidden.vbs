' exp/18 세션 독립 실행 래퍼 — 콘솔 창을 숨겨서(0) 실수로 닫히는 사고 방지
Set sh = CreateObject("WScript.Shell")
sh.Run """c:\Users\gwonn\Desktop\open\run_night.cmd""", 0, False
