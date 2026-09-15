@echo off
rem 세션 독립 실행 래퍼 — 작업 스케줄러가 실행. 결과는 lab\18_result.txt
cd /d c:\Users\gwonn\Desktop\open
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=8
py -3.12 -u exp\18_year_stability.py > lab\18_result.txt 2>&1
echo %date% %time% exp18 done > lab\night_done.txt
