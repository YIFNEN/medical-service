# MEDIC 프로토타입 시험 가이드

작업 단위가 준비되면 저장소 루트에서 `./scripts/checkpoint.ps1 -Message '작업 설명' -Python $Python`으로 전체 시험·커밋·푸시를 수행한다.
푸시된 브랜치는 GitHub Actions의 **Offline tests**에서 검증한다. 실패하면 해당 작업 단위에서 수정·재검증하고 다음 기능으로 넘어가지 않는다.
로컬 커밋까지만 필요하면 `-NoPush`를 지정한다. 스크립트는 임의의 주기로 실행하지 않고 개발자가 검토 가능한 작업 단위마다 호출한다.

이 문서는 구현과 함께 누적한다. 각 단계의 미구현·미검증 상태는 [진행 기록](IMPLEMENTATION_STATUS_KO.md)을 확인한다.
아래 설치/합성/견적 명령은 유료 생성 호출이나 개인 자료 전송을 하지 않는다.

## 1. 설치와 준비

저장소 루트에서 PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
$Python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$Store = 'D:\Medic\.medic\prototype'
& $Python -X utf8 -m medic doctor --store $Store
& $Python -X utf8 -m medic doctor --store $Store --slicer 'D:\Medic\.medic\Slicer-5.12.4\Slicer.exe'
```

예상: JSON `network_calls_made: 0`, 패키지 버전, 키의 존재 여부, Slicer 경로·존재 여부.
`offline_ready`가 false면 누락 패키지/Python/저장 경로를 수정한다.
키 존재는 실제 모델·가격·승인 검증을 의미하지 않는다. 키 값은 출력하지 않는다.

## 2. 키 없는 합성 회귀시험

```powershell
& $Python -X utf8 -m pytest -q
& $Python -X utf8 -m pytest -q tests/test_workflow_resume.py tests/test_tribunal.py tests/test_tribunal_resume.py tests/test_budget.py
```

모든 실행은 합성·녹화 응답이며 provider 호출 경계·재개·예산 실패를 시험한다.
합격: 실패 0, 중단 후 성공한 호출이 반복되지 않음, 부족한 제공자 검토는 완료 표시하지 않음.
실패 시 traceback과 `pytest` 요약을 보존하고 관련 단일 시험으로 재현한다. 테스트 저장소는 pytest 임시 경로이며 환자 저장소와 분리한다.

## 3. 문헌 수량·견적만 확인

```powershell
& $Python -X utf8 -m medic deliberate ANKLE-PILOT-001 --preset pilot --store $Store
& $Python -X utf8 -m medic deliberate ANKLE-PILOT-001 --preset pilot --evidence-limit 8 --fulltext-limit 3 --store $Store
```

예상 종료 코드 2는 승인 요청용 견적이며 실패가 아니다. `paid_calls_made: 0`.
기본 pilot는 검색 5/선별 20/카드 3/전문 1, 두 번째 명령은 카드 8/전문 3.
가격표는 설정 당시 기준이며 실제 청구액 보장이 아니다. 확정 설정을 출력에서 보존한다.

JSON 설정 파일은 예를 들어 `{"evidence_limit": 6, "fulltext_limit": 2}`이며 `--config`로 연결한다.
CLI 값을 명시하면 파일을 덮어쓴다. 기존 run은 `resume`에 수량 옵션을 끼워 넣어 바꾸지 못한다.

## 4. 목록에서 합성 자료 선택·가져오기

아래는 공개해도 되는 합성 문자열만 사용한다. 실제 원본 경로를 이 디렉터리로 바꾸거나 덮어쓰지 않는다.

```powershell
$Case = 'ANKLE-PILOT-001'
$InputDir = Join-Path $Store 'synthetic-input'
New-Item -ItemType Directory -Force $InputDir | Out-Null
Set-Content -LiteralPath (Join-Path $InputDir 'record.txt') -Encoding utf8 -Value 'SYNTHETIC: right ankle pain after walking. Not a patient record.'
$Inventory = (& $Python -X utf8 -m medic inventory $InputDir --store $Store) | ConvertFrom-Json
$View = (& $Python -X utf8 -m medic inventory-review status --inventory $Inventory.path --store $Store) | ConvertFrom-Json
$Leaf = (Get-Content -Raw -LiteralPath $View.path | ConvertFrom-Json).records | Where-Object classification -eq 'UNKNOWN' | Select-Object -First 1
& $Python -X utf8 -m medic inventory-review assign --inventory $Inventory.path --record-id $Leaf.record_id --role PATIENT_RECORD --case $Case --reviewer local-user --reason '합성 예제 자료 분류' --store $Store
& $Python -X utf8 -m medic ingest-inventory $Case --inventory $Inventory.path --record-id $Leaf.record_id --store $Store
```

예상: inventory JSON 경로와 snapshot별 record ID, 가져오기 intake ID·file_count 1·paid_calls_made 0.
`inventory-staging/`, `raw/`, `ledger.jsonl`은 저장소 아래에 생성된다.
ZIP/중첩 ZIP/ISO 내부 항목도 검수한 leaf ID로 선택한다. 원본·모든 컨테이너·leaf 해시가 달라지면 거부한다.
역할 미검수, 다른 사례, 지원하지 않는 문서 형식은 가져오지 않는다. DICOM은 별도 inspect를 사용한다.
합격: 원본 수정 없음, 같은 항목 재실행에서 파싱 재사용. 실패 시 새 목록을 만들고 분류부터 검토한다.

실자료에서는 `inventory 'D:\Documents\Medical'`로 새 목록을 만든 뒤 최대 문서 3개를 명시적으로 고른다.
목록 파일에는 개인 경로·자료 정보가 있으므로 공개하지 않는다. 모든 PDF를 환자 기록이나 논문으로 자동 분류하지 않는다.

## 5. 개인 조건·질문·기록 의미

```powershell
& $Python -X utf8 -m medic case profile --case $Case --file tests\fixtures\prototype\profile.json --reviewer local-user --reason '합성 조건 입력' --store $Store
& $Python -X utf8 -m medic case questions --case $Case --file tests\fixtures\prototype\questions.json --store $Store
& $Python -X utf8 -m medic case status --case $Case --store $Store
& $Python -X utf8 -m medic case questions --case $Case --store $Store
```

예제는 SYNTHETIC-PERSON의 조건이며 실제 환자에 그대로 적용하지 않는다.
profile과 questions의 event ID·version을 보존한다. 질문 예제의 q-current는 관찰 ID가 없어 보류,
q-prognosis는 horizon/outcome/treatment_state가 없어 보류가 정상이다. `ANSWER`는 구조적 사용 자격이며 의학적 정답 표시가 아니다.
원문 검수 후 질문 `allowed_fact_ids`와 `allowed_measurement_ids`에는 실제 저장소 ID만 연결한다.
그 질문에 필요한 조건이 갖춰지면 관련 질문만 분석 가능해진다.

원문 추출·검수 후 `source-review.json`을 개인 저장소에 복사하여 fact ID·환자/episode/epoch·부위·좌우와 의미를 수정한다.

```powershell
$InputReview = Join-Path $Store 'source-review.json'
Copy-Item tests\fixtures\prototype\source-review.json $InputReview
# 로컬 편집기로 REPLACE_WITH_REVIEWED_FACT_ID 및 실제 원문 의미를 수정한 뒤 실행
& $Python -X utf8 -m medic case review-input --case $Case --file $InputReview --reviewer local-user --reason '원문과 의미·연결 확인' --store $Store
```

템플릿을 수정하지 않으면 unknown fact ID로 거부되는 것이 정상이다.
추출 검수의 ID는 원문 검수 프롬프트와 ledger의 HIGH_IMPACT_FACTS_CONFIRMED.context.fact_details에서 확인한다.
분류: observation / clinician_assertion / request / procedure / self_report / actual_exposure / derived_measure.
처방·요청은 실제 노출이 아니다. actual_exposure는 시행 근거, derived_measure는 derived_from_ids가 필요하다.
날짜는 authoring/issuance/acquisition/event 등을 구분하고 year/month/day/unknown 정밀도를 유지한다.
missingness는 present/explicit_negative/not_provided/not_measured/not_assessable이다.
정정에는 기존 event ID를 `--replaces`로 연결하고, 철회는 해당 명령의 `--withdraw EVENT_ID`를 사용한다.
원문 문구 변경·원문 철회로 검수 참조가 오래되면 재검수를 요구하고 이전 버전을 자동 복구하지 않는다.

## 6. Slicer 합성 왕복과 수동 계측

먼저 기술적 합성 정답 시험:

```powershell
$Prepared = (& $Python -X utf8 -m medic dicom prepare-slicer-test --store $Store) | ConvertFrom-Json
$env:MEDIC_SLICER_VALIDATION_CONFIG = $Prepared.config_path
$Slicer = 'D:\Medic\.medic\Slicer-5.12.4\Slicer.exe'
$TestProcess = Start-Process -FilePath $Slicer -ArgumentList @('--ignore-slicerrc','--no-splash','--no-main-window','--testing','--python-script', (Join-Path (Get-Location) 'scripts\slicer_roundtrip_test.py')) -WindowStyle Hidden -PassThru -Wait
Get-Content -Raw -LiteralPath $Prepared.result_path | ConvertFrom-Json
```

합격은 새 result.json의 `status: PASSED`와 오차 ≤1e-6이다. launcher 종료 0만으로 판정하지 않는다.
거리 √145 mm, 각도 90°, 표시 부피 576 mm³, 평가 영역 6912 mm³, 비율 1/12,
좌표·원본 픽셀 순서·표시 위치·측정 ID/이름/방법 메타데이터를 확인한다.
실패하면 stdout/stderr와 새 결과를 보존하고 새 prepare로 재시험한다. 예전 PASSED 파일을 재사용하지 않는다.

실자료 수동 계측 준비:

```powershell
$Inspection = (& $Python -X utf8 -m medic dicom inspect --inventory $Inventory.path --store $Store) | ConvertFrom-Json
$Manifest = Get-Content -Raw -LiteralPath $Inspection.path | ConvertFrom-Json
$Series = $Manifest.series | Where-Object measurement_eligible | Select-Object -First 1
& $Python -X utf8 -m medic dicom launch-review --manifest $Inspection.path --series $Series.series_uid --slicer $Slicer --launch --store $Store
```

합성 text inventory에는 적합 영상이 없으므로 실제 DICOM 목록이나 prepare가 만든 manifest로 실행한다.
자동 첫 시리즈 선택은 예시일 뿐, 실제 대상·좌우·촬영 범위를 확인해 명시적으로 선택한다.
Slicer는 선택한 볼륨을 열며 기준점은 사용자가 표준 Markups Line/Angle로 지정한다.
Python Console에서 명시적으로 해당 노드만 연결·내보낸다:

```python
line = slicer.util.getNode('Line')  # 실제 만든 노드 이름으로 변경
medic_bind(line)
# 방법·좌우·epoch·기준점·조건은 실제 검수한 measurement-metadata.json 내용으로 설정
line.SetAttribute('Medic.Metadata', json.dumps(metadata))
review_path = medic_export([line])
print(review_path)
```

`metadata`는 검수한 JSON을 `json.loads`로 읽은 dict다. UNKNOWN 촬영 조건을 추측으로 채우지 않는다.
다른 시리즈에 연결된 노드나 연결하지 않은 노드는 거부한다. 장면의 모든 기준점을 일괄 내보내지 않는다.
내보낸 review.json 경로로:

```powershell
$ReviewPath = Read-Host 'Slicer가 출력한 review.json 절대 경로'
& $Python -X utf8 -m medic dicom import-review --manifest $Inspection.path --review $ReviewPath --case $Case --reviewer local-user --store $Store
```

원본 해시·시리즈·좌표·단위·기하를 검증하고 review ID를 반환한다. USER_REVIEWED는 전문가 인증이 아니다.
수정은 `--replaces-review-id OLD_ID --amendment-reason '수정 이유'`, 철회는
`dicom withdraw-review --case CASE --review-id ID --reviewer USER --reason TEXT --store STORE`를 사용한다.

## 7. 두 시점 수동 지표 비교

```powershell
& $Python -X utf8 -m medic dicom compare-measurements --case $Case --before-review BEFORE_REVIEW_ID --before-measurement BEFORE_MEASUREMENT_ID --after-review AFTER_REVIEW_ID --after-measurement AFTER_MEASUREMENT_ID --store $Store
```

두 review ID는 import 출력, measurement ID는 해당 review JSON에서 읽는다. 예시 토큰을 실제 ID로 바꾼다.
결과는 `imaging/comparisons/<hash>.json`. COMPARABLE에서만 scalar 차이를 계산하며 검사 좌표를 직접 빼지 않는다.
좌우·방법·단위·epoch·촬영 조건이 다르면 INCOMPATIBLE, UNKNOWN이면 LIMITED_UNKNOWN.
반복측정 오차가 검수되면 `--repeatability-bound 값 --repeatability-source 출처`를 함께 지정한다.
오차 이내는 안정의 증명이 아니며 clinical_change는 NOT_ESTABLISHED다. 실제 두 시점 자료가 없으면 미검증으로 기록한다.

## 8. 비용·전송 승인 후 실제 제공자 시험

이 절은 자동 실행하지 않는다. 설정 모델·현재 가격·키와 승인한 전송 범위를 확인한 후 사용자가 수행한다.
가격/모델은 `MEDIC_MODEL_REGISTRY` JSON으로 고정한다. 설정 가격은 실제 청구액 보장이 아니다.

먼저 합성 입력만 사용하는 smoke 견적:

```powershell
& $Python -X utf8 -m medic providers smoke --run-id smoke-pilot-001 --store $Store
```

승인된 smoke 예산을 정한 뒤에만:

```powershell
$SmokeBudget = [double](Read-Host '별도로 승인한 smoke 예산 USD')
& $Python -X utf8 -m medic providers smoke --run-id smoke-pilot-001 --approve-cost --budget-usd $SmokeBudget --approval-id smoke-pilot-001 --store $Store
```

세 제공자 PASSED, 모델·응답 ID·사용량을 확인한다. 한 제공자 실패는 INCOMPLETE이고 같은 ID 재실행은 완료 호출을 재사용한다.
키 없음·미지원 모델·스키마·잘린 응답이면 설정을 수정하고 예산/모델을 바꿀 경우 새 run ID를 만든다.

실제 문서 추출은 견적부터 확인:

```powershell
& $Python -X utf8 -m medic review $Case --interactive --store $Store
$ExtractionBudget = [double](Read-Host '별도로 승인한 추출 예산 USD')
& $Python -X utf8 -m medic review $Case --interactive --approve-cost --approve-transmission --budget-usd $ExtractionBudget --approval-id extraction-pilot-001 --store $Store
```

스캔 PDF/사진은 영상 토큰 상한도 별도 검토하여 `--image-tokens-bound`로 지정한다. 산정할 수 없으면 중단이 정상이다.
검수는 문서의 기재 내용 확인이며 의학적 사실의 확정이 아니다. 답변은 항목별 저장되어 중단 후 재사용한다.
원문·검수 다음에 개인 조건/의미/질문을 갱신하고 판단 견적을 재확인한다.

```powershell
& $Python -X utf8 -m medic deliberate $Case --preset pilot --run-id pilot-001 --store $Store
$DecisionBudget = [double](Read-Host '별도로 승인한 판단 예산 USD')
& $Python -X utf8 -m medic deliberate $Case --preset pilot --run-id pilot-001 --approve-cost --approve-transmission --budget-usd $DecisionBudget --approval-id decision-pilot-001 --store $Store
& $Python -X utf8 -m medic literature search-plan pilot-001 --store $Store
# 출력 검색식의 식별정보를 검토한 뒤에만 승인하여 재개
& $Python -X utf8 -m medic resume pilot-001 --approve-cost --approve-transmission --approve-search --store $Store
& $Python -X utf8 -m medic report pilot-001 --format html --store $Store
```

검색 검수로 일시 중단되는 것은 정상이다. 외부 검색은 비식별 임상 검색식만 사용하고 DICOM 원본은 보내지 않는다.
완성 보고서는 `reports/pilot-001/`. 의견별 다른 두 제공자 비평과 세 제공자 종합 검토, 주장별 원문 근거·보류를 확인한다.
비용과 개인 자료 전송 승인은 별개다. `--approve-cost`만으로 개인 입력을 전송하지 않는다.

## 9. 의견 가림·봉인·사후 비교

입력 분류를 완료하고 blind-review.json의 질문·개인 조건 항목·영상 review ID를 실제 검수한 값으로 설정한다.
기관 제약·의료진 권고·전언·과거 AI 결론을 본인 경험이나 관찰로 재분류하지 않는다.

```powershell
$BlindReview = Join-Path $Store 'blind-review.json'
Copy-Item tests\fixtures\prototype\blind-review.json $BlindReview
# 로컬에서 검수·수정. 새 질문/자료면 새 run ID 사용
& $Python -X utf8 -m medic deliberate $Case --preset pilot --mode clinician-blind --blind-review $BlindReview --run-id blind-001 --store $Store
```

위 명령은 견적뿐이다. 별도 승인 후 같은 명령에 `--approve-cost --approve-transmission --budget-usd 승인액 --approval-id 새승인ID`를 붙인다.
검색식 검토·재개는 8절과 같다. 해시·cutoff가 고정되고 완료된 세 제공자 결과만 BLIND_RESULT_SEALED로 봉인한다.
가림 mode의 문장 일치 검사는 완전한 의미적 비식별/가림 보장을 뜻하지 않으므로 입력 분류의 사람 검수가 필요하다.

```powershell
& $Python -X utf8 -m medic compare-clinician blind-001 --store $Store
```

봉인 전이면 거부한다. 봉인 후에는 비교 입력 경로와 별도 AI 비교 견적만 제공한다.
비교 AI 호출은 별도 비용·전송 승인 후 `--approve-cost --approve-transmission --budget-usd 승인액 --approval-id 새비교승인ID`로 실행한다.
수동 비교는 `--file`로 `{"rows":[{"clinician_fact_id":"실제 ID","ai_claim_id":"실제 ID","relationship":"INSUFFICIENT_EVIDENCE","explanation":"검수한 설명"}],"unresolved":[]}`를 제공한다.
양측 ID는 출력된 comparison input JSON에서 확인한다. 모르는 ID·다른 run·다른 사례는 거부한다.
AGREEMENT/CONFLICT/ASSUMPTION_DIFFERENCE/TIME_DIFFERENCE/INSUFFICIENT_EVIDENCE를 구분하고 정확도 점수는 만들지 않는다.
결과는 `comparisons/results/`에 새로 저장하며 원래 보고서·봉인 결과를 바꾸지 않는다.

## 10. 장애·재개·후속 자료·증빙

| 상황 | 다음 행동·합격 기준 |
|---|---|
| 키 없음/모델 미지원 | doctor 및 registry 확인. 자동 모델 대체 없음 |
| 제공자 장애/스키마/잘림 | run ID와 오류 종류 보존, 동일 승인으로 resume. 완료 호출 재사용 |
| 예산 부족/응답 유실 | 미결 예약 유지. 임의로 예약 삭제하지 않고 과금 확인 후 새 승인·새 실행 |
| 설정·가격·모델·입력 변경 | 기존 run 변조 거부. 새 run ID와 예산 검토 |
| 검색식 식별정보 | 승인하지 않고 원인 입력을 수정한 새 실행 |
| 원본/압축/영상 해시 불일치 | 원본 선택·목록·검수부터 갱신. mismatch 결과 가져오기 거부 |
| source review가 stale | 바뀐 사실만 재검수. 이전 기록과 보고서는 보존 |
| 지원하지 않는 영상 | 목록에는 남기고 계측하지 않음 |
| 전문 미확보/청크 한도 | 후보 대기·미분석 위치와 영향받는 주장 보류 확인 |
| 가림 검사 실패 | 가린 의견의 재분류/요약/질문/자유 메모를 로컬 검토. 일반 분석으로 자동 전환 없음 |

후속 자료는 새 intake·검수·새 run으로 분석하고 비교:

```powershell
& $Python -X utf8 -m medic compare-runs pilot-001 pilot-002 --store $Store
& $Python -X utf8 -m pytest -q tests/test_workflow_resume.py tests/test_question_runtime.py tests/test_blinding.py tests/test_inventory_ingest.py tests/test_measurement_comparison.py
```

실행 비교는 문서 사실·근거·계측·의견의 차이다. 촬영 조건이 다르면 차이를 악화·호전으로 해석하지 않는다.
시험 기록에는 시각·코드 버전·명령·입출력 ID·증빙 경로·예상/실제·합격/실패/미검증·다음 행동을 기록한다.
합성 수학 정답·기술적 Slicer 왕복·사람 기준점·실제 세 모델 실행·임상 정확도를 별도 행으로 기록한다.
개인 자료와 실행 산출물은 공개 저장소에 추가하지 않는다.
