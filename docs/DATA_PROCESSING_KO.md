# 의료자료를 재사용 가능한 기록으로 가공하는 경로

CD를 컴퓨터에 백업하면서 생긴 질문은 파일을 잃지 않는 보관을 넘어, 과거 지표를 이후 판단에서도 다시 사용할 수 있느냐는 것이었습니다. 현재 구현은 사용자가 로컬에 복사한 파일을 입력으로 받아 출처·원문 위치·검수 사실·영상 좌표를 연결합니다. CD 자동 복제, 병원 포털 자동 다운로드, 자동 비식별화 서비스는 구현 범위에 포함하지 않습니다.

## 구현과 검증 연결

| 처리 | 코드 | 오프라인 검증 |
| --- | --- | --- |
| 파일·중첩 ZIP·지원 ISO 읽기 전용 조사, 경로·해시·처리 상태와 안전 한도 | `medic/inventory.py` | `tests/test_inventory.py` |
| 자료 역할의 명시적 배정, 사례·서지정보·원본 재검증과 검수 이력 | `medic/material_roles.py` | `tests/test_material_roles.py` |
| 인입 원본 보존, PDF 페이지·영역별 텍스트와 이미지 보존, 출처 위치 | `medic/intake.py`, `medic/ledger.py` | `tests/test_intake.py`, `tests/test_compiler_pipeline.py` |
| 사실 추출 계약과 원문 연결, 부분 검수·재개 | `medic/extraction.py`, `medic/extraction_batches.py` | `tests/test_extraction.py`, `tests/test_extraction_resume.py`, `tests/test_partial_review.py` |
| DICOM 인스턴스·검사·시리즈 정리, UID 충돌·기하·슬라이스 순서·계측 적합성 | `medic/imaging.py`, `medic/imaging_geometry.py` | `tests/test_imaging.py` |
| 수동 계측·영역 표시의 원본/좌표/해시 검증, 수정·철회 연결 | `medic/slicer_review.py`, `medic/imaging_history.py` | `tests/test_imaging_amendment.py`, `tests/test_imaging_withdrawal.py`, `tests/test_slicer_validation.py` |
| 실행 간 사실·근거·영상 검수·의견 차이와 원문별 경과 보고 | `medic/comparison.py`, `medic/reporting.py` | `tests/test_run_comparison.py`, `tests/test_longitudinal_reporting.py` |

해시를 기준으로 중복을 식별하고 원본 변경을 확인합니다. 이름이나 파일 위치가 비슷하다는 이유만으로 서로 다른 자료를 같은 검사로 합치지 않습니다. 목록에 있는 PDF를 자동으로 특정 환자 기록이나 논문으로 확정하지 않습니다.

## 픽셀 계산과 실제 계측 정확도

픽셀/복셀 좌표를 실제 공간과 연결하려면 간격·방향·원점과 촬영 조건이 필요합니다. 단순 스크린샷의 픽셀 수만으로 실제 길이를 제시하지 않습니다. 적합성이 확인된 시리즈에서 사람이 지정한 점과 평가면으로 거리와 각도를 계산하며, 수동 표시 영역의 면적·부피를 계산합니다. 평가 대상 영역이 함께 있어야 표시 영역의 비율을 계산합니다.

임의 각도를 임상적인 변형 각도로 명명하지 않습니다. 임상 명칭을 붙이는 경우 출처가 있는 축과 방법을 요구하며, 그 임상 프로토콜의 정확성을 자동 인증하지 않습니다. 미표시 조직은 NOT_EVALUATED로 남깁니다.

실제 Slicer의 합성 정답 비교와 CT 2개 시리즈 / 510장의 원본 보존·빈 검수 왕복은 [검증 요약](VALIDATION_KO.md)에 기록했습니다. 수학적 계산 오차, 원본 픽셀 보존, 임상 기준점의 정확도는 서로 다른 검증입니다.

## 과거 기록 활용의 현재 경계

검수된 문서 사실과 영상 계측에는 출처와 조건이 연결되어 이후 사례 컨텍스트에서 다시 사용할 수 있습니다. 실행 비교는 추가·제거된 사실과 근거, 영상 검수 및 모델 의견의 변경을 구분합니다. 서로 다른 촬영 자세나 체중부하 조건의 수치를 바로 경과 변화로 해석하지 않습니다.

현재 자동 영상 정합, 동일 해부학적 부위 자동 추적, 영상 간 자동 변화 탐지, 임상적 악화·호전 판정과 자동 진단은 구현하지 않았습니다. 장기 관찰을 위한 자료 기반과 검수 결과 재사용 경로를 구현한 상태입니다.
