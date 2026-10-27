# 로컬 DICOM 인벤토리와 수동 검토

이 하위 시스템은 파일 출처와 영상 좌표를 검증하는 기술적 도구입니다. 임상 검증을 완료한 의료기기가 아니며 진단, 정상 판정, 수술 추천을 수행하지 않습니다. `measurement_eligible`은 아래 기술 조건의 통과만 뜻합니다.

## 부모 CLI 연동 API

일반 사용은 다음 CLI를 사용합니다. 출력의 `path`를 다음 명령의 해당 인수에 지정하십시오. DICOM 저장소는 `.medic` 아래여야 합니다.

```powershell
medic inventory "C:\example-data\medical" --store .medic/personal
medic dicom inspect --inventory INVENTORY_JSON --store .medic/personal
medic dicom export --manifest MANIFEST_JSON --series SERIES_UID --store .medic/personal
medic dicom import-review --manifest MANIFEST_JSON --review REVIEW_JSON --case MY-ANKLE-001 --reviewer USER_NAME --store .medic/personal
```

`export`는 검수 템플릿과 Slicer 스크립트 경로를 반환하며 GUI를 자동 실행하지 않습니다. 계측·수동 영역에는 모델 검토에 사용할 고유 `fact_id`가 부여됩니다. 기본 `USER_REVIEWED`와 `EXPERT_REVIEWED`는 구별하지만 전문가 자격은 사용자가 선언한 것이며 시스템이 검증하지 않습니다. 미검수 결과는 근거 ID로 인정하지 않습니다.

```python
from pathlib import Path
import json
from medic.inventory import scan, Limits
from medic.imaging import inspect_inventory
from medic.slicer_review import import_review, review_template

local = Path('.medic')
local.mkdir(exist_ok=True)
inventory = scan(Path(r'C:\example-data\medical'))
manifest = inspect_inventory(inventory, local / 'dicom')
(local / 'inventory.json').write_text(
    json.dumps(inventory, ensure_ascii=False, allow_nan=False), encoding='utf-8')
(local / 'imaging.json').write_text(
    json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding='utf-8')
```

- `scan(root, *, limits=None) -> dict`: 읽기 전용 재귀 검사. `records`, `limits`, `bytes_read`, `truncated` 반환.
- `Limits(max_member_bytes=134217728, max_total_bytes=1073741824, max_entries=100000, max_depth=4, max_ratio=200)`.
- 레코드: `source_path`는 root 기준 상대 경로, `archive_path`는 바깥에서 안쪽 순서의 멤버 경로 배열. `container_hashes`도 같은 순서이며 `sha256`은 해당 멤버의 바이트 해시입니다. 해시 형식은 `sha256:<hex>`입니다.
- `classification`: `ZIP`, `ISO`, `UNKNOWN`. `UNKNOWN`을 환자 자료나 논문으로 자동 배정하지 않습니다. `status`: `ACCEPTED` 또는 `REJECTED`. 안전 제한으로 읽지 않은 파일은 해시가 null입니다.
- `read_record(inventory, record) -> bytes`: 원본과 모든 상위 아카이브 해시를 다시 검증합니다. 변경되면 예외가 발생합니다.
- `inspect_inventory(inventory, store_path) -> dict`: `.medic` 아래 로컬 경로만 허용. 식별된 DICOM/DICOMDIR만 해시 이름으로 복사하고 `instances`, `directories`, `series`, `conflicts`, `rejected`를 반환합니다. 원본은 수정하지 않습니다.
- `review_template(manifest, series_uid, *, study_uid=None) -> dict`: 검토용 출처/좌표 계약 생성.
- `import_review(review_dict, manifest, *, review_dir=None) -> dict`: 원본 해시, series/study, affine, shape, 좌표 범위 및 분할 파일 해시를 검증하고 수동 측정값 반환.

경로, 날짜, UID, 영상 및 출력 JSON에는 개인정보가 있을 수 있습니다. 상세 출력은 `.medic`에만 저장하고 콘솔에는 집계만 표시하십시오. `.medic/`는 기존 `.gitignore`에서 제외됩니다. 외부 전송이나 유료 API 호출은 없습니다. 복사 파일은 비식별화되지 않았습니다.

## 지원 범위와 보수적 제한

ZIP 안 ZIP/ISO, ISO 안 ZIP을 깊이 제한 안에서 검사합니다. ISO는 pycdlib의 ISO9660/Rock Ridge/Joliet 읽기를 사용합니다. 아카이브를 디스크에 풀지 않습니다. 경로 탈출, 절대 경로, 역슬래시, 대체 스트림 경로, 심볼릭 링크/reparse point, 실행 파일 확장자/일부 실행 헤더, 중복 ZIP 멤버, 암호화 ZIP, 과도한 압축률은 거부합니다. 실행 파일을 실행하지 않습니다.

파일과 아카이브 전체에 동일한 128 MiB 기본 크기 제한을 적용합니다. 따라서 대형 ISO는 기본 설정에서 거부될 수 있습니다. 전체 압축 파일과 현재 멤버를 메모리에 읽으므로 제한 상향 전 메모리를 고려해야 합니다. 총 바이트 예산은 컨테이너와 압축 해제 멤버를 모두 계산합니다. UDF 전용 이미지, 손상된/특수 ISO, 암호화 파일, RAR/7z/TAR는 지원하지 않습니다. 실시간으로 공격자가 변경하는 파일시스템에 대한 원자적 보안 경계는 아닙니다. 검사 중 원본을 변경하지 마십시오.

확장자 없는 DICOM 및 preamble 없는 식별 가능한 데이터셋을 pydicom으로 읽습니다. DICOMDIR는 인덱스로 기록하며 참조 경로를 따라가지 않습니다. 멀티프레임은 프레임 수를 기록하되 Enhanced functional-group geometry를 재구성하지 않습니다.

CT/MR 측정은 표준 단일 프레임 SOP class, 비압축 단색 8/16-bit 픽셀 길이, 일치하는 FrameOfReferenceUID, 최소 두 장, 유효한 방향 코사인/양수 간격/동일 행열 크기, 일정한 평행 슬라이스 간격이 필요합니다. 중복 위치, gantry tilt, 비균일 간격, SOP 해시 충돌, 다른 study에서 재사용된 series UID는 차단합니다. 압축 전송 문법, multiframe, NM/SC 및 기타 모달리티는 검사 전용입니다. 픽셀 디코딩/육안 품질은 별도 검토가 필요합니다.

DX/CR은 단일 영상, 지원 SOP class, 명시적인 `PixelSpacingCalibrationType` (`GEOMETRY`/`FIDUCIAL`), 유효한 PixelSpacing 및 환자 기준 방향/위치가 모두 있을 때 영상 평면 안에서만 측정합니다. 보정 태그가 있다는 사실은 보정의 임상적 정확도를 보증하지 않습니다. `ImagerPixelSpacing`만으로 환자 공간 보정을 추정하지 않습니다. 환자 방향/위치가 없는 일반 방사선 영상은 이 버전에서 검사 전용입니다.

체중부하 여부는 항상 `UNKNOWN`입니다. 촬영일, laterality, modality는 원본 태그를 기록하며 영상 내용으로 추론하지 않습니다.

## 좌표와 측정

### 잘못된 검수 철회

수정 계측을 가져올 때 `--replaces-review-id PREVIOUS_REVIEW_ID --amendment-reason "기준점 수정"`을 함께 지정하면 이전 검수와의 연결을 보존합니다. 대상은 같은 사례·검사·시리즈의 최신 가져오기여야 합니다. 최신 검수가 철회된 상태에서도 수정 연결은 가능합니다. 오래된 대상, 다른 사례·시리즈, 자기 자신 또는 사유 누락은 거부합니다. 연결 정보는 이후 분석 컨텍스트와 보고서의 영상 검수 항목에 포함됩니다.

```powershell
medic dicom withdraw-review --case MY-ANKLE-001 --review-id REVIEW_ID --reviewer USER_NAME --reason "기준점 위치 오류" --store .medic/personal
```

철회는 이유·선언된 검수자·UTC 시각을 새 이력으로 저장합니다. 원본 영상과 가져온 검수 JSON, 과거 보고서는 삭제하거나 수정하지 않습니다. 최신 검수를 철회하면 이후 분석 입력에서 해당 검수가 빠지며, 과거 계측을 자동 복구하지 않습니다. 과거 검수만 철회하면 최신 검수는 유지됩니다. 같은 내용의 철회 재시도는 이력을 중복 저장하지 않습니다.

수정 계측은 Slicer에서 다시 내보낸 뒤 `import-review`로 새 검수로 가져오십시오. 동일한 기존 검수 파일을 다시 가져오는 것만으로 철회가 취소되지는 않습니다. 다른 임상 기록 없이 유일한 영상 검수까지 철회했다면 분석 가능한 검수 자료가 없으므로 새 검수가 필요합니다. 완료된 run 재개는 기존 결과를 반환하므로, 철회를 반영한 판단에는 새 run을 사용해야 합니다.

```python
from medic.imaging_geometry import measure_series, voxel_to_lps, lesion_metrics

series = manifest['series'][0]
point = voxel_to_lps(series['geometry'], [1, 2, 0])
measurement = measure_series(series, [[0, 0, 0], point], kind='line')
```

voxel 순서는 `(column, row, slice)`이며 LPS 좌표의 단위는 mm입니다. RAS에서 LPS로 변환할 때 x/y 부호를 반전합니다. 길이는 mm, 각도는 degrees, 슬라이스 단면적은 mm², 부피는 mm³입니다. 두 점 길이, 세 점 각도, 한 점 위치를 지원합니다. CLI는 자격/범위 검사를 수행하는 `measure_series`를 사용해야 합니다. 저수준 `measure`는 임의의 유한 LPS 점을 계산하는 수학 함수입니다.

선택적인 `plane={'origin': [...], 'normal': [...]}`는 단위 법선으로 정의된 LPS 평면에 점들을 투영합니다. 이름이 있는 protocol에는 `axes={'x': [...], 'y': [...], 'source': '명시적 출처'}`가 필수입니다. 이는 출처 기록이며 특정 임상 프로토콜의 자동 구현/검증이 아닙니다. protocol/axes/plane 좌표는 항상 LPS입니다.

수동 lesion mask는 boolean `(columns, rows, slices)` 배열만 허용합니다. `lesion_metrics(mask, geometry, evaluation_roi=None)`는 국소 표시 부피/슬라이스 면적을 반환합니다. 평가 ROI가 없으면 분모와 비율은 null입니다. 제공한 ROI는 비어 있지 않고 lesion을 포함해야 합니다. 표시되지 않은 영역은 `NOT_EVALUATED`이며 정상 조직/전신 질환 유무로 해석하지 않습니다. 여러 병변이 겹칠 때 각 병변의 개별 값이며 합산하지 않습니다.

## 3D Slicer

2026-10-19에 공식 Slicer 5.12.4로 합성 CT의 계측·수동 영역 왕복과 실제 CT 두 시리즈(각 255장)의 픽셀·기하·빈 검수 왕복을 자동 실행으로 확인했습니다. [검증 요약](VALIDATION_KO.md)을 참조하십시오. GUI에서 사람이 기준점·병변을 지정하는 작업과 전문가 비교는 아직 검증 대기입니다. 아래 경로는 예시이며 자신의 설치·저장소 절대 경로로 바꾸십시오. Slicer Python 환경에는 pynrrd가 필요합니다.

합성 검증을 다시 실행하려면 새 묶음을 생성하십시오. 매번 새 결과 경로를 사용하며 이전 결과를 덮어쓰지 않습니다.

```powershell
$medicCheck = medic dicom prepare-slicer-test --store .medic/personal | ConvertFrom-Json
$env:MEDIC_SLICER_VALIDATION_CONFIG = $medicCheck.config_path
& 'C:\tools\Slicer-5.12.4\Slicer.exe' --ignore-slicerrc --no-splash --no-main-window --testing --python-script (Resolve-Path .\scripts\slicer_roundtrip_test.py).Path
Get-Content -LiteralPath $medicCheck.result_path
```

`PREPARED_NOT_RUN`은 준비 완료만 뜻합니다. 실제 성공은 새 `result.json`의 `status: PASSED`로 확인해야 합니다. 실행 프로세스의 종료 코드만으로 판정하지 마십시오. 실자료 기술 시험은 `dicom prepare-slicer-source-test --manifest MANIFEST_JSON --series SERIES_UID`로 준비하고 반환된 별도 스크립트 `slicer_source_roundtrip_test.py`를 사용합니다. 해당 시험은 픽셀·기하·출처와 빈 검수만 확인합니다.

Slicer Python console에서 저장소 경로를 `sys.path`에 추가한 뒤 `scripts/slicer_review.py`를 `runpy.run_path`로 읽습니다. 표준 Slicer의 DICOMUtils/Markups/Segmentation API를 사용합니다.

```python
import runpy
bridge = runpy.run_path(r'C:\projects\medical-service\scripts\slicer_review.py')
volume = bridge['load_series'](manifest, selected_series_uid, r'C:\projects\medical-service\.medic\slicer')
review_path = bridge['export_review'](
    manifest, volume, selected_markup_nodes,
    r'C:\projects\medical-service\.medic\reviews',
    segmentation=manual_lesion_node, evaluation_roi=evaluation_roi_node)
```

선택한 시리즈의 검증된 파일만 임시 DICOM DB로 불러옵니다. 로드된 volume의 실제 IJK→RAS affine 및 크기가 원본 geometry와 다르면 중단합니다. 변환이 붙은 reference volume/segmentation은 지원하지 않습니다. Markups의 world RAS 좌표와 `.mrk.json`, 원본 `.seg.nrrd`, 참조 영상 격자로 내보낸 개별 segment labelmap, 검토 manifest를 새 디렉터리에 저장합니다. evaluation ROI 노드는 단일 segment만 사용하십시오. 다중 ROI는 모호성을 이유로 거부됩니다.

재입력은 manifest에 포함된 inline Markups와 해시가 있는 labelmap을 기준으로 합니다. `.mrk.json`이나 `.seg.nrrd`만 따로 편집했다면 Slicer에서 다시 export해야 합니다. 분할 labelmap은 inline NRRD raw/gzip 및 명시적 LPS/RAS, 원본과 같은 affine/shape만 지원합니다. detached NRRD, 임의 resampling/transform, 매우 큰 분할은 거부합니다.

Slicer의 GUI에서 실제 로드/편집/내보내기까지 수행한 검증은 아직 없습니다. 로컬 Python에서는 합성 DICOM, oblique/anisotropic affine, 좌표 변환, 충돌/잘못된 metadata, NRRD 해시/geometry 검사를 테스트합니다. 실제 환자 자료로 임상 정확도를 검증했다는 의미가 아닙니다.

참고: [Slicer DICOM API 예제](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/dicom.html), [Slicer segmentation API 예제](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/segmentations.html).
