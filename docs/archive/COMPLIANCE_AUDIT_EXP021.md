# EXP021 제출 컴플라이언스 감사

감사 시점: 2026-08-31 KST

## 결론

`candidate_champ55.zip`과 `candidate_exp021_w0356557.zip`은 **COMPLIANCE HOLD**다.
운영진의 서면 허용과 원 저작권자의 유효한 라이선스/허락을 확보하기 전에는 최종 진출 코드로
사용하거나 같은 EXP021 코드를 포함한 추가 제출을 만들지 않는다.

이는 외부 데이터·API나 행 독립성 위반 판정이 아니다. 문제는 활성 대회 중 다른 팀이 GitHub에
공개한 코드의 복제·재배포 권한과 대회 코드 공유 규정이다.

## 확인된 사실

- 원 저장소: `https://github.com/mk-isos/lg-aimers-9-pitch-control`
- 잠금 커밋: `2c469c10ab96dcd7fb24af609b57c508fd1eb3dc`
- 저장소는 public이지만 Git tree·전체 이력·GitHub license API 모두 LICENSE가 없다.
- 제출 소스와 원본 추론 파일은 모두 33,047 bytes다.
- 양쪽 MD5: `2913B0ED2EB505492BA2016186F65EC9`
- 양쪽 SHA256: `156B0FC519A3AA0F1373193BE6E07A9B7D7941F1FD29D19535ADB2C4E8854728`
- 따라서 `model/exp021_inference.py`는 원본과 byte-identical하다.
- 모델 파일은 공식 train으로 로컬 재학습했지만, 재현·패키징에 원 저장소의 byte-identical 소스를
  사용했다. 외부 데이터/API는 사용하지 않았다.

## 적용 규정과 위험

공식 규정 페이지: `https://dacon.io/competitions/official/236743/overview/rules`

- 공개 모델·가중치는 최소 비상업적 이용을 허용하는 라이선스 아래 배포된 경우만 허용한다.
- 공개 코드 공유는 제3자의 지식재산권을 침해하면 안 되며 Dacon 플랫폼을 통해야 한다.
- 참가자는 제출물이 자신의 고유 저작물이고 필요한 권리를 보유한다고 보증한다.

GitHub 공식 라이선스 안내:
`https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository`

GitHub 공개 저장소에 LICENSE가 없으면 기본 저작권이 적용된다. 즉 공개 열람 가능성만으로 복제,
배포, 파생물 작성 권한이 주어지지 않는다. 출처 표기는 라이선스 부재를 치료하지 않는다.

## 즉시 조치

1. EXP021을 포함한 추가 제출·재패키징·코드 은폐를 중지한다.
2. 두 제출과 SHA를 보존하고 삭제·개작으로 이력을 숨기지 않는다.
3. Dacon에 원 저장소 URL·commit·byte-identical 사실을 밝히고 허용 여부와, 불허 시 기존 두
   제출의 제외/처리 방법을 서면으로 문의한다.
4. 원 저작권자에게 최소 비상업적 이용·복제·수정·재배포를 명시적으로 허용하는 공개 LICENSE 또는
   서면 허락을 요청한다. 이것만으로 Dacon의 플랫폼 공유 요건이 자동 해결되는 것은 아니다.
5. 답변 전까지 공식 train과 우리 팀 고유 코드만 쓴 `candidate_v18g030.zip`을 안전 기준선으로 둔다.
6. 독립 재구현은 단순 변수명 변경이나 포맷팅이 아니다. 복제 코드를 보지 않은 독립 구현과 자체
   학습·검증 기록이 있어야 하며, 그래도 운영진 허용을 먼저 받는 것이 안전하다.

## 운영진 문의 초안

```text
[DACON 답변 요청] 공개 GitHub 코드 사용 가능 범위 및 기존 제출 처리 문의

다른 참가팀이 활성 대회 중 공개한 GitHub 저장소의 알고리즘을 검토해, 공식 제공 train.csv만으로
모델을 로컬 재학습했습니다. 다만 감사 결과 제출 ZIP 내부 추론 파일 1개가 해당 저장소 파일과
byte-identical이고, 저장소에는 LICENSE가 없으며 Dacon 코드공유 게시물 경유 여부도 확인되지 않습니다.

원 저장소: https://github.com/mk-isos/lg-aimers-9-pitch-control
commit: 2c469c10ab96dcd7fb24af609b57c508fd1eb3dc
동일 파일 SHA256: 156B0FC519A3AA0F1373193BE6E07A9B7D7941F1FD29D19535ADB2C4E8854728
영향 제출: 2026-08-31 16:43:23 / 17:05:33 KST의 두 제출

외부 데이터나 외부 API는 사용하지 않았습니다. 이 사용이 대회 규정상 허용되는지, 불허된다면
위 두 제출을 평가·진출 선정에서 제외하기 위해 어떤 절차를 밟아야 하는지 서면 안내 부탁드립니다.
답변 전에는 해당 코드를 포함한 추가 제출을 중지하겠습니다.
```
