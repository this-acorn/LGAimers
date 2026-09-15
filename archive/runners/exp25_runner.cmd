@echo off
rem =====================================================================
rem exp/25 세션 독립 실행 래퍼 — 작업 스케줄러가 실행한다.
rem Claude Code 세션이 닫혀도 계속 돈다. (exp/18에서 검증된 방식)
rem
rem 절차: exp/24 완료 대기(최대 45분) -> exp/25 실행 -> 완료 마커 기록
rem 결과: lab\25_result.txt  /  완료 마커: lab\exp25_done.txt
rem =====================================================================
cd /d c:\Users\gwonn\Desktop\open
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=6

rem ---- exp/24가 끝날 때까지 대기 (동시 실행 시 CPU 경쟁으로 둘 다 느려짐) ----
rem timeout 대신 ping을 쓴다 — 숨긴 창에는 콘솔 입력 핸들이 없어 timeout이 실패한다
set /a tries=0
:wait
findstr /C:"EXIT_CODE" lab\24_result.txt >nul 2>&1
if not errorlevel 1 goto run
set /a tries+=1
if %tries% GEQ 90 goto run
ping -n 31 127.0.0.1 >nul
goto wait

:run
echo %date% %time% exp25 start (waited %tries% x30s) > lab\exp25_started.txt
py -3.12 -u exp\25_catboost_multiyear.py > lab\25_result.txt 2>&1
echo %date% %time% exp25 done rc=%ERRORLEVEL% > lab\exp25_done.txt
