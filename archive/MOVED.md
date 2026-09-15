# 폴더 정리 기록 (2026-08-30)

루트가 지저분해져서 아래처럼 옮겼습니다. **`exp/` 스크립트 일부가 옛 경로를 하드코딩**하고 있으므로,
과거 스크립트를 다시 돌릴 일이 생기면 경로 앞에 `archive/...` 를 붙이세요.

## 이동 내역

| 이동 대상 | 새 위치 |
|---|---|
| FEATURE_BRIEF.md, MEETING_NOTES.md, data_description.md | `docs/` |
| calib_result.txt, calib_backtest_result.txt, xgb_result.txt, xgb_stage2/3_result.txt | `archive/results/` |
| 01_diagnose.py.py, exp25_runner.cmd, exp25_hidden.vbs, run_night.cmd, run_night_hidden.vbs | `archive/runners/` |
| submit.zip, submit4~16.zip, submit18~22*.zip, submit_gbdt.zip | `archive/submissions/` |
| submit8_extract, submit10/11/13/14/15_src | `archive/src/` |
| submit18_src_it300_m012, submit18_src_it500_* (4개) | `archive/src/` |
| _tmp_sim_112_candidate_it300, _tmp_sim_112_current_it500 | `archive/tmp_sim/` |
| model.pkl (루트 잔여 사본) | `archive/misc/` |
| REJECTED_DO_NOT_SUBMIT_v18g030_it300.zip 및 source | `archive/rejected/v18_it300/` |

## 루트에 그대로 둔 것 (스크립트가 참조 중이거나 현재 작업분)

- `data/`, `lab/`, `exp/` — 경로 하드코딩 다수, 절대 이동 금지
- `submit12_src/`, `submit16_src/`, `submit17_src/`, `submit18_src_it300/`, `submit_champ_src/` — exp/100·101·103·104·105 가 참조
- `candidate_v18g030_src/`, `candidate_v18g030.zip` — 현재 챔피언
- IT300+V18 교차 artifact는 direct-LB 반증 후 `archive/rejected/v18_it300/`로 격리
- `champion_target5_teamcat.zip` — 현 챔피언 기준선
- `submit/`, `baseline_submit/`, `probe_kit/`, `teampack_100/`(+zip), `team_test/`, `reference/`, `학습자료/`, `2026-Lg-Aimers-hamin/`
- `HANDOFF.md`, `claude.md`, `.gitignore`
