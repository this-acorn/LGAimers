# 저장소 지도

## 루트에 남겨야 하는 활성 항목

| 경로 | 역할 | 이동 여부 |
|---|---|---|
| `data/` | 공식 학습·형식 확인 데이터 | **이동 금지** — 다수 스크립트 하드코딩 |
| `exp/` | 실험 코드 | **이동 금지** |
| `lab/` | 결과·예측 배열·러너 | **이동 금지** |
| `candidate_v18g030.zip` | 현 챔피언 | 루트 유지 |
| `candidate_v18g030_src/` | 현 챔피언 추론 소스 | 루트 유지 |
| `champion_target5_teamcat.zip` | 직전 기준선/재구성 source | 루트 유지 |
| `submit*_src/` 일부 | 과거 실험이 직접 참조 | 현 단계 이동 금지 |
| `submit/`, `baseline_submit/` | 공식 제출 형식·기준선 | 유지 |
| `reference/` | 공개 저장소 읽기 전용 clone | 유지 |
| `docs/` | 현재 정본과 역사 문서 | 유지 |
| `archive/` | 비활성 결과·제출·소스·폐기물 | 유지 |

## archive 구조

```text
archive/
├── MOVED.md          # 실제 이동 기록
├── submissions/      # 과거 제출 ZIP
├── src/              # 과거 제출 source
├── results/          # 구형 단독 결과 txt
├── runners/          # 비활성 실행 스크립트
├── tmp_sim/          # 보존한 가짜서버 디렉터리
├── rejected/         # 제출 금지 artifact
└── misc/             # 분류되지 않은 구형 파일
```

`archive/rejected/`는 “낮은 점수 후보”가 아니라 **제출 금지**만 둔다.

## reference 중복

`reference/mk_isos_lgaimers9/`와
`reference/mk-isos_lg-aimers-9-pitch-control/`은 같은 공개 저장소의 중복 clone으로 보인다.
현재 분석 문서와 스크립트가 두 경로를 모두 인용하므로 마감 전에는 삭제·이름변경하지 않는다.
새 재현 작업의 기준 경로는 짧은 이름인 `reference/mk_isos_lgaimers9/`로 통일한다.

## 역사 스냅샷

- `docs/FEATURE_BRIEF.md`: 08-23 상태.
- `docs/MEETING_NOTES.md`: 08-24 상태.
- `2026-Lg-Aimers-hamin/README.md`: 해당 브랜치의 1059 챔피언 상태.

현재 프로젝트 상태는 [CURRENT_STATE.md](CURRENT_STATE.md)만 기준으로 삼는다.

## 정리 원칙

1. 다른 세션이 실행 중인 파일은 이동하지 않는다.
2. 경로를 옮기면 `archive/MOVED.md`, 코드의 상수 경로, 관련 문서 링크를 같은 변경에서 갱신한다.
3. 과거 결과를 삭제하지 않는다. 중복이 확실해도 마감 전에는 보존한다.
4. 루트에는 현재 챔피언과 재구성에 필요한 기준선만 남긴다.
