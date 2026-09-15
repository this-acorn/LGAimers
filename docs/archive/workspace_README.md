# 문서 인덱스

최종 갱신: 2026-08-30 PDT / 2026-08-31 KST

## 운영 정본

| 문서 | 역할 | 갱신 원칙 |
|---|---|---|
| [CURRENT_STATE.md](CURRENT_STATE.md) | 현재 챔피언·금지 artifact·실행 소유권 | 상태가 바뀔 때 즉시 |
| [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) | 분류별 실험 결과와 증거등급 | 결과를 덮어쓰지 않고 행 추가 |
| [DEAD_AXES.md](DEAD_AXES.md) | 사망 범위와 재개 조건 | exact variant 단위 |
| [SUBMISSION_LOG.md](SUBMISSION_LOG.md) | 실제 제출 ZIP·SHA·LB | 제출 직후 append-only |
| [STRATEGY_9_SUBMISSIONS.md](STRATEGY_9_SUBMISSIONS.md) | 남은 9회의 조건부 의사결정 | 슬롯 사용 전 확인 |
| [REPOSITORY_MAP.md](REPOSITORY_MAP.md) | 폴더 지도·이동 금지 경로 | 구조 변경 때 |
| [templates/EXPERIMENT_TEMPLATE.md](templates/EXPERIMENT_TEMPLATE.md) | 새 실험 기록 양식 | 양식 변경 시 |

## 근거 문서

- `HANDOFF.md`: 시간순 상세 원장. 역사 보존용이며 파일 앞부분의 현재 상태는 낡았다.
- `lab/105_reference_dissection.md`: x2 공개 저장소 감사.
- `lab/110_mkisos_dissection.md`: mk-isos 공개 저장소 감사.
- `docs/data_description.md`: 공식 데이터·행 독립 규칙.
- `claude.md`: 대회 환경과 규정 요약.

## 역사 문서

- `FEATURE_BRIEF.md`: 2026-08-23 스냅샷.
- `MEETING_NOTES.md`: 2026-08-24 스냅샷.

두 문서의 점수·우선순위는 현재 판단에 사용하지 않는다.

## 기록 규칙

1. 실제 LB, strict temporal 검증, paired local, 단일시드 진단을 섞어 쓰지 않는다.
2. 직접 LB 반증은 양수 로컬 결과를 항상 덮는다.
3. “비슷한 축이 실패”와 “그 exact variant가 실패”를 구분한다.
4. 제출물은 ZIP 이름뿐 아니라 SHA256, parent, 단일 변경, 실행시간을 기록한다.
5. Claude와 Codex가 동시에 일하면 [CURRENT_STATE.md](CURRENT_STATE.md)의 소유권 표를 먼저 갱신한다.
