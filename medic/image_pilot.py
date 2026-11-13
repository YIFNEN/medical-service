"""Local DICOM previews and source-bound, blinded research observations."""
import html
import io
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import pydicom
from PIL import Image
from pydantic import Field, model_validator
from pydicom.uid import (ComputedRadiographyImageStorage, CTImageStorage,
                         DigitalXRayImageStorageForPresentation, DigitalXRayImageStorageForProcessing)

from .extended_cli import read_json, run_id, save_json
from .imaging import _inspect
from .inventory import _linked
from .ledger import AppendOnlyJsonlLedger, canonical_json_bytes, sha256_digest
from .research_contracts import StrictFrozenModel


class ImageRegion(StrictFrozenModel):
    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)

    @model_validator(mode='after')
    def check_area(self):
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError('empty image region')
        return self


class Candidate(StrictFrozenModel):
    image_id: str
    location: str = Field(min_length=1)
    # Normalized image coordinates, from the top-left of the attached PNG.
    region: ImageRegion
    observation: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


class ImageFindings(StrictFrozenModel):
    candidates: tuple[Candidate, ...]
    limitations: tuple[str, ...] = Field(min_length=1)


class ComparisonRow(StrictFrozenModel):
    candidate_index: int | None = Field(ge=1)
    reference_index: int | None = Field(ge=1)
    relationship: Literal['AGREEMENT', 'DISAGREEMENT', 'AI_MISSED', 'REPORT_SILENT', 'NOT_COMPARABLE']
    explanation: str = Field(min_length=1)


class CTDisplay(StrictFrozenModel):
    plane: Literal['SAGITTAL', 'CORONAL', 'AXIAL', 'SOURCE_SAGITTAL', 'SOURCE_CORONAL', 'SOURCE_AXIAL']
    window_hu: tuple[float, float]
    sampling: Literal['NEAREST']
    inverted: bool
    row_direction_lps: tuple[float, float, float]
    column_direction_lps: tuple[float, float, float]


class XrayDisplay(StrictFrozenModel):
    projection: Literal['AP', 'PA', 'LAT', 'LL', 'RL', 'OBLIQUE', 'UNKNOWN']
    display_range: tuple[float, float]
    photometric: Literal['MONOCHROME1', 'MONOCHROME2']
    physical_measurement_eligible: bool


class ReportReference(StrictFrozenModel):
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


def _hash(value):
    return sha256_digest(canonical_json_bytes(value))


def _store(root):
    root = Path(root).absolute()
    if '.medic' not in root.parts or str(root).startswith('\\\\') or any(
        p.exists() and _linked(p) for p in (root, *root.parents)
    ):
        raise ValueError('local unlinked .medic store required')
    return root


def _inside(root, path):
    path = Path(path).absolute()
    if ('..' in path.parts or not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() or (p.exists() and _linked(p)) for p in (path, *path.parents))):
        raise ValueError('pilot path must remain within its local store')
    return path


def _pilot_ledger(root, identifier):
    path = _inside(root, root / 'imaging' / 'pilot-runs' / run_id(identifier) / 'ledger.jsonl')
    return AppendOnlyJsonlLedger(path)


def _save_pilot_json(root, category, data):
    _inside(root, root / category / (_hash(data).split(':')[1] + '.json'))
    return save_json(root, category, data)


def _pixels(root, image):
    digest = image['sha256']
    name = digest.removeprefix('sha256:')
    if len(name) != 64 or any(c not in '0123456789abcdef' for c in name):
        raise ValueError('invalid pixel hash')
    path = _inside(root, root / 'imaging' / 'pilot-pixels' / (name + '.png'))
    raw = path.read_bytes()
    if sha256_digest(raw) != digest:
        raise ValueError('approved pixel bytes changed')
    return path, raw


def _save_pixels(root, png):
    digest = sha256_digest(png)
    target = _inside(root, root / 'imaging' / 'pilot-pixels' / (digest.split(':')[1] + '.png'))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open('xb') as stream:
            stream.write(png)
    except FileExistsError:
        _pixels(root, {'sha256': digest})
    return digest


def _select(manifest, uid, modalities):
    matches = [s for s in manifest['series'] if s['series_uid'] == uid]
    if len(matches) != 1:
        raise ValueError('missing or ambiguous series')
    series = matches[0]
    instances = {i['sha256']: i for i in manifest['instances']
                 if (i['study_uid'], i['series_uid']) == (series['study_uid'], uid)}
    if (not instances or set(instances) != set(series['original_hashes'])
            or any(i['modality'] not in modalities for i in instances.values())):
        raise ValueError('unexpected series modality or source set')
    conflicts = {i['sop_uid'] for i in manifest['conflicts']}
    if any(i['sop_uid'] in conflicts for i in instances.values()):
        raise ValueError('conflicting DICOM instance identity')
    return series, instances


def _decode(item, root):
    source = _inside(root, item['stored_path'])
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('invalid local DICOM source')
    raw = source.read_bytes()
    if sha256_digest(raw) != item['sha256']:
        raise ValueError('DICOM source hash mismatch')
    actual = _inspect(raw)
    if (not actual or not actual.get('pixel_payload_valid') or actual['frames'] != 1
            or any(actual[k] != item[k] for k in
                   ('study_uid', 'series_uid', 'sop_uid', 'modality', 'orientation',
                    'position', 'spacing', 'rows', 'columns', 'calibration', 'sop_class_uid',
                    'frame_of_reference_uid', 'laterality'))
            or actual['rows'] * actual['columns'] > 16_000_000):
        raise ValueError('unsupported or inconsistent DICOM pixels')
    allowed = {'CT': (CTImageStorage,), 'CR': (ComputedRadiographyImageStorage,),
               'DX': (DigitalXRayImageStorageForPresentation, DigitalXRayImageStorageForProcessing)}
    if actual['sop_class_uid'] not in allowed[actual['modality']]:
        raise ValueError('unsupported preview SOP class')
    ds = pydicom.dcmread(io.BytesIO(raw), force=True)
    if ds.PhotometricInterpretation not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('only monochrome source pixels are supported')
    return ds


def _render(pixels, low, high, invert=False):
    if not np.isfinite(pixels).all() or not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('invalid display values')
    image = np.rint(np.clip((pixels - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    if invert:
        image = 255 - image
    output = io.BytesIO()
    Image.fromarray(image).save(output, format='PNG')
    return output.getvalue()


def _patient_identity(ds):
    patient_id = str(getattr(ds, 'PatientID', '')).strip()
    if not patient_id:
        raise ValueError('DICOM patient identity unavailable for cross-exam preparation')
    return tuple(str(getattr(ds, key, '')).strip() for key in
                 ('PatientID', 'IssuerOfPatientID', 'PatientName', 'PatientBirthDate'))


def prepare(root, manifest, case, ct_uid, xray_uid, xray_hashes=()):
    from .imaging_geometry import validate_geometry

    root = _store(root)
    ct, ct_items = _select(manifest, ct_uid, {'CT'})
    dx, dx_items = _select(manifest, xray_uid, {'DX', 'CR'})
    selected = list(xray_hashes) or sorted(k for k, i in dx_items.items() if i['pixel_payload_valid'])[:2]
    if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected) or set(selected) - set(dx_items):
        raise ValueError('select one or two distinct X-ray instances in the series')
    ordered = ct.get('ordered_hashes', [])
    if not ct['measurement_eligible'] or set(ordered) != set(ct_items) or len(ordered) != len(ct_items):
        raise ValueError('CT requires inspected three-dimensional geometry')
    ct_frames = [ct_items[k] for k in ordered]
    if len({i['frame_of_reference_uid'] for i in ct_frames}) != 1 or not ct_frames[0]['frame_of_reference_uid']:
        raise ValueError('missing or inconsistent CT frame of reference')
    sides = {i['laterality'].upper() for i in (*ct_frames, *(dx_items[k] for k in selected))} & {'L', 'R'}
    if len(sides) > 1:
        raise ValueError('CT and X-ray DICOM laterality differs')
    geometry = validate_geometry(ct_frames)
    if (geometry['shape'] != ct['geometry']['shape'] or
            not np.allclose(geometry['affine_lps'], ct['geometry']['affine_lps'], atol=1e-5, rtol=0)
            or geometry['slice_order'] != list(range(len(ct_frames)))
            or int(np.prod(geometry['shape'])) > 64 * 1024 * 1024):
        raise ValueError('CT geometry mismatch or preview size limit')
    volume = np.empty(tuple(reversed(geometry['shape'])), dtype=np.float32)
    photometric = set()
    identities = set()
    for n, item in enumerate(ct_frames):
        ds = _decode(item, root)
        identities.add(_patient_identity(ds))
        photometric.add(str(ds.PhotometricInterpretation))
        slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        if slope <= 0 or not np.isfinite([slope, intercept]).all():
            raise ValueError('invalid CT intensity transform')
        volume[n] = ds.pixel_array.astype(np.float32) * slope + intercept
    if len(photometric) != 1:
        raise ValueError('inconsistent CT photometric interpretation')
    invert_ct = 'MONOCHROME1' in photometric

    images = []

    def add(identifier, modality, png, locator, display):
        images.append(dict(image_id=identifier, modality=modality, sha256=_save_pixels(root, png),
                           locator=locator, display=display))

    affine = np.asarray(geometry['affine_lps'])
    # Patient-aligned stacks may be transposed or flipped. Preserve the exact
    # source index transform. Oblique stacks are shown in their source axes.
    unit = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    axes = np.argmax(np.abs(unit), axis=0)
    aligned = len(set(axes)) == 3 and np.allclose(np.max(np.abs(unit), axis=0), 1, atol=1e-5)
    for axis, fallback in enumerate(('SOURCE_SAGITTAL', 'SOURCE_CORONAL', 'SOURCE_AXIAL')):
        plane = ('SAGITTAL', 'CORONAL', 'AXIAL')[axes[axis]] if aligned else fallback
        array_axis = 2 - axis
        size = volume.shape[array_axis]
        varying = [i for i in (2, 1, 0) if i != axis]
        spacing = [float(np.linalg.norm(affine[:3, i])) for i in varying]
        for index in sorted({(size - 1) * fraction // 4 for fraction in (1, 2, 3)}):
            pixels = np.take(volume, index, axis=array_axis)
            # Restore physical aspect ratio for viewing, with nearest-neighbor
            # sampling so no intermediate intensity is invented.
            rows, cols = pixels.shape
            scale = min(1024 / max(rows * spacing[0], cols * spacing[1]), 1 / min(spacing))
            width, height = max(1, round(cols * spacing[1] * scale)), max(1, round(rows * spacing[0] * scale))
            native = _render(pixels, -300, 1700, invert_ct)
            resized = io.BytesIO()
            with Image.open(io.BytesIO(native)) as image:
                image.resize((width, height), Image.Resampling.NEAREST).save(resized, format='PNG')
            add(f'CT-{plane}-{index}', 'CT', resized.getvalue(),
                dict(series_uid=ct_uid, fixed_ijk_axis=axis, index=index, row_ijk_axis=varying[0],
                     column_ijk_axis=varying[1], native_shape=[rows, cols],
                     image_shape=[height, width], affine_lps=geometry['affine_lps'],
                     source_hashes=[ordered[index]] if axis == 2 else ordered),
                dict(plane=plane, window_hu=[-300, 1700], sampling='NEAREST', inverted=invert_ct,
                     row_direction_lps=unit[:, varying[0]].round(6).tolist(),
                     column_direction_lps=unit[:, varying[1]].round(6).tolist()))

    for n, key in enumerate(selected, start=1):
        ds = _decode(dx_items[key], root)
        if identities != {_patient_identity(ds)}:
            raise ValueError('CT and X-ray DICOM patient identity differs')
        pixels = ds.pixel_array.astype(np.float32)
        low, high = map(float, np.percentile(pixels, (1, 99)))
        if high == low:
            low, high = float(pixels.min()), float(pixels.max())
        view = str(getattr(ds, 'ViewPosition', '')).upper()
        add(f'XRAY-{n}', 'XRAY', _render(pixels, low, max(high, low + 1), ds.PhotometricInterpretation == 'MONOCHROME1'),
            dict(series_uid=xray_uid, source_hashes=[key], frame=0, image_shape=list(pixels.shape)),
            dict(projection=view if view in {'AP', 'PA', 'LAT', 'LL', 'RL', 'OBLIQUE'} else 'UNKNOWN',
                 display_range=[low, max(high, low + 1)], photometric=str(ds.PhotometricInterpretation),
                 physical_measurement_eligible=dx['measurement_eligible']))
    data = dict(version=1, case_id=case, images=images, studies={
        'CT': dict(study_uid=ct['study_uid'], series_uid=ct_uid),
        'XRAY': dict(study_uid=dx['study_uid'], series_uid=xray_uid)})
    return {**_save_bundle(root, data), 'ct_images': len(images) - len(selected), 'xray_images': len(selected),
            'xray_measurement_eligible': dx['measurement_eligible']}


def _image_requests(root, data):
    requests = {}
    if any(image['modality'] not in ('CT', 'XRAY') for image in data['images']):
        raise ValueError('unknown image modality')
    for modality, maximum, contract, pattern in (
        ('CT', 9, CTDisplay, r'CT-(?:SOURCE_)?(?:SAGITTAL|CORONAL|AXIAL)-\d{1,4}'),
        ('XRAY', 2, XrayDisplay, r'XRAY-[12]'),
    ):
        items = [i for i in data['images'] if i['modality'] == modality]
        ids = [i['image_id'] for i in items]
        if (not 1 <= len(ids) <= maximum or len(set(ids)) != len(ids)
                or any(not re.fullmatch(pattern, identifier, re.ASCII) for identifier in ids)):
            raise ValueError('invalid image selection')
        attachments, metadata = [], []
        for item in items:
            path, raw = _pixels(root, item)
            display = contract.model_validate_json(json.dumps(item['display'], allow_nan=False))
            boxes = item.get('redactions', [])
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.mode != 'L' or image.info or image.size[::-1] != tuple(item['locator']['image_shape']):
                    raise ValueError('unexpected preview format, metadata, or dimensions')
                _validate_boxes(boxes, image)
            attachments.append((path, raw))
            metadata.append({'image_id': item['image_id'], **display.model_dump(mode='json'),
                             'redacted_regions_px': boxes})
        requests[modality] = ids, attachments, metadata
    return requests


def _save_bundle(root, data):
    requests = _image_requests(root, data)
    path = _save_pilot_json(root, 'imaging/pilot-inputs', data)
    digest = _hash(data)
    figures = []
    for ids, attachments, metadata in requests.values():
        for identifier, (image, _), display in zip(ids, attachments, metadata, strict=True):
            label = html.escape(identifier)
            image_url = '../pilot-pixels/' + image.name
            details = html.escape(json.dumps(display, indent=2))
            figures.append(f'<figure><h2>{label}</h2><a href="{image_url}"><img alt="{label}" '
                           f'src="{image_url}"></a><pre>{details}</pre></figure>')
    preview = _inside(root, path.with_suffix('.html'))
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>MEDIC 영상 전송 검수</title>'
                '<style>body{font:16px system-ui;margin:2rem}figure{margin:2rem 0}img{max-width:100%;'
                'height:auto}pre{white-space:pre-wrap}</style><h1>MEDIC 영상 전송 검수</h1>'
                '<p>그림을 열면 원래 픽셀 크기로 볼 수 있습니다. 영상에 남은 이름, 번호, 날짜 등의 '
                '식별정보를 확인하고 필요한 영역을 가린 후 새 묶음을 검수하세요.</p>'
                f'<p><code>{digest}</code></p>' + ''.join(figures) + '</html>')
    try:
        with preview.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(document)
    except FileExistsError:
        if _linked(preview) or preview.read_text(encoding='utf-8') != document:
            raise ValueError('stored preview differs from the generated bundle')
    return {'path': str(path), 'preview_path': str(preview), 'input_sha256': digest, 'status': 'REVIEW_REQUIRED'}


def _read_artifact(path, expected=None):
    path = Path(path)
    data = read_json(path)
    digest = _hash(data)
    if path.stem != digest.split(':')[1] or (expected is not None and expected != digest):
        raise ValueError('pilot artifact hash mismatch')
    return data, digest


def _validate_boxes(boxes, image):
    if not isinstance(boxes, list):
        raise ValueError('redaction rectangles must be a list')
    for box in boxes:
        if not (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)):
            raise ValueError('redaction rectangle must be four pixel integers')
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError('redaction rectangle outside preview')


def redact(root, path, masks_path):
    """Apply operator-selected pixel rectangles to exact local previews."""
    root = _store(root)
    data, original_hash = _read_artifact(path)
    masks = read_json(masks_path)
    by_id = {image['image_id']: image for image in data['images']}
    if not isinstance(masks, dict) or not masks or set(masks) - set(by_id):
        raise ValueError('redaction rectangles must identify preview image IDs')
    for identifier, boxes in masks.items():
        item = by_id[identifier]
        if not isinstance(boxes, list) or not boxes:
            raise ValueError('at least one rectangle required for each selected image')
        _, raw = _pixels(root, item)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.copy()
        _validate_boxes(boxes, image)
        for box in boxes:
            image.paste(0, tuple(box))
        output = io.BytesIO()
        image.save(output, format='PNG')
        item['sha256'] = _save_pixels(root, output.getvalue())
        item.setdefault('redactions', []).extend(boxes)
    data['parent_input_sha256'] = original_hash
    return _save_bundle(root, data)


def analyze(root, identifier, path, approved_hash, reviewer, estimate, registry, gateway):
    from .call_cache import cached_generate

    root = _store(root)
    if (not estimate.approved or not estimate.transmission_approved
            or estimate.backend != 'codex' or estimate.data_scope != 'PERSONAL'):
        raise PermissionError('Codex attempt and image transmission approval required')
    data, digest = _read_artifact(Path(path), approved_hash)
    if not reviewer.strip() or not approved_hash:
        raise ValueError('reviewer and exact inspected image bundle hash required')
    if estimate.codex_calls is None or estimate.codex_calls < 2:
        raise ValueError('two modality calls require at least two approved Codex attempts')
    ledger = _pilot_ledger(root, identifier)
    requests = _image_requests(root, data)
    ledger.append_once('IMAGE_PILOT_CREATED', identifier, {
        'input_sha256': digest, 'reviewer': reviewer, 'estimate': estimate.model_dump(mode='json'), 'model': gateway.model})
    reports = {}
    for modality, (ids, attachments, metadata) in requests.items():
        prompt = ('Research-only visual observations. The attached images follow the supplied order. '
                  'Use image_id, image-relative location and normalized region with left, top, right and bottom '
                  'coordinates from the top-left. '
                  'Report what is visible and uncertainties. '
                  'Do not infer absence in unshown areas, diagnoses, or measurements. The reference report is withheld.\n'
                  + json.dumps(metadata))
        def validate_findings(findings):
            if any(candidate.image_id not in ids for candidate in findings.candidates):
                raise ValueError('candidate references an invalid image region')

        # The shared source files may change while the first request is active.
        # Attach only the bytes already verified against the approved bundle.
        with TemporaryDirectory(prefix='medic-images-', dir=ledger.path.parent) as directory:
            paths = []
            for image_id, (_, raw) in zip(ids, attachments, strict=True):
                attachment = Path(directory) / (image_id + '.png')
                attachment.write_bytes(raw)
                paths.append(attachment)
            response = cached_generate(ledger, gateway, scope={'version': 'image-pilot/1', 'bundle': digest, 'modality': modality},
                                       estimate=estimate, stage='IMAGE_PILOT_' + modality, prompt=prompt,
                                       output_model=ImageFindings, max_output_tokens=4000, image_paths=tuple(paths),
                                       validate_result=validate_findings)
        reports[modality] = {'findings': response.value.model_dump(mode='json'),
                             'call': registry.call_record('IMAGE_PILOT_' + modality, response).model_dump(mode='json')}
    sealed = {'version': 1, 'run_id': identifier, 'case_id': data['case_id'], 'input_sha256': digest,
              'studies': data['studies'], 'results': reports, 'clinical_validation': 'NOT_VALIDATED'}
    ledger.append_once('IMAGE_PILOT_SEALED', identifier, {'sha256': _hash(sealed), 'result': sealed})
    return {'path': str(_save_pilot_json(root, 'imaging/pilot-results', sealed)), 'status': 'SEALED',
            'run_id': identifier, 'result_sha256': _hash(sealed)}


def _load_seal(root, sealed_path):
    sealed, seal_hash = _read_artifact(Path(sealed_path))
    ledger = _pilot_ledger(root, sealed['run_id'])
    if not any(r['kind'] == 'IMAGE_PILOT_SEALED' and r['payload'] == {'sha256': seal_hash, 'result': sealed}
               for r in ledger.records()):
        raise ValueError('persisted sealed image result required')
    return sealed, seal_hash


def pilot_status(root, identifier):
    root = _store(root)
    records = _pilot_ledger(root, identifier).records()
    created = next((r['payload'] for r in records if r['kind'] == 'IMAGE_PILOT_CREATED'), None)
    if created is None:
        raise ValueError('unknown image pilot run')
    seal = next((r['payload'] for r in records if r['kind'] == 'IMAGE_PILOT_SEALED'), None)
    stages = {r['payload']['stage'] for r in records if r['kind'] == 'PROVIDER_CALL_COMPLETED'}
    estimate = created['estimate']
    reserved = sum(r['kind'] == 'CODEX_ATTEMPT_RESERVED' and r['payload']['approval_id'] == estimate['approval_id']
                   for r in records)
    result = {'run_id': identifier, 'status': 'SEALED' if seal else 'INCOMPLETE',
              'input_sha256': created['input_sha256'],
              'completed_modalities': [mode for mode in ('CT', 'XRAY') if 'IMAGE_PILOT_' + mode in stages],
              'attempts_reserved': reserved, 'attempts_remaining': estimate['codex_calls'] - reserved}
    if seal:
        result['result_sha256'] = seal['sha256']
    return result


def _reference_quotes(raw, suffix, references, password):
    def normalize(value):
        return ' '.join(value.split())

    if not references:
        raise ValueError('reference observations require original pages and quotes')
    if suffix == '.pdf':
        import pymupdf

        with pymupdf.open(stream=raw, filetype='pdf') as document:
            if document.needs_pass and (not password or not document.authenticate(password)):
                raise ValueError('locked reference PDF: use --pdf-password-prompt locally')
            pages = {ref.page: normalize(document[ref.page - 1].get_text(sort=True))
                     for ref in references if ref.page <= len(document)}
    elif suffix == '.txt':
        pages = {1: normalize(raw.decode('utf-8-sig'))}
    else:
        raise ValueError('reference source must be a PDF or UTF-8 text file')
    for ref in references:
        quote = normalize(ref.quote)
        if not quote or quote not in pages.get(ref.page, ''):
            raise ValueError('reference quote not found on its original page')


def compare(root, sealed_path, review_path, *, password=None):
    root = _store(root)
    sealed, seal_hash = _load_seal(root, sealed_path)
    review = read_json(review_path)
    modality = review['modality']
    if (review.get('result_sha256') != seal_hash or review['case_id'] != sealed['case_id']
            or review['study_uid'] != sealed['studies'][modality]['study_uid']
            or not review['reviewer'].strip() or review['same_exam_confirmed'] is not True):
        raise ValueError('reviewed same-case and same-exam report binding required')
    source = Path(review['report_path']).resolve(strict=True)
    raw = source.read_bytes()
    if sha256_digest(raw) != review['report_sha256']:
        raise ValueError('reference report hash mismatch')
    references = [ReportReference.model_validate(ref) for ref in review['references']]
    _reference_quotes(raw, source.suffix.lower(), references, password)
    findings = sealed['results'][modality]['findings']['candidates']
    rows = [ComparisonRow.model_validate_json(json.dumps(row)) for row in review['rows']]
    covered_candidates, covered_references = set(), set()
    for row in rows:
        a, b = row.candidate_index, row.reference_index
        if (a is None and b is None or a is not None and a > len(findings)
                or b is not None and b > len(references)):
            raise ValueError('comparison references an unknown observation')
        if row.relationship in ('AGREEMENT', 'DISAGREEMENT') and (a is None or b is None):
            raise ValueError('agreement and disagreement require both observations')
        if row.relationship == 'AI_MISSED' and (a is not None or b is None):
            raise ValueError('AI_MISSED requires a reference observation alone')
        if row.relationship == 'REPORT_SILENT' and (a is None or b is not None):
            raise ValueError('REPORT_SILENT requires a candidate alone')
        if a is not None:
            covered_candidates.add(a)
        if b is not None:
            covered_references.add(b)
    if covered_candidates != set(range(1, len(findings) + 1)) or covered_references != set(range(1, len(references) + 1)):
        raise ValueError('comparison must account for each candidate and reference')
    result = {'version': 1, 'result_sha256': seal_hash, 'case_id': sealed['case_id'], 'modality': modality,
              'report_sha256': review['report_sha256'], 'reviewer': review['reviewer'],
              'references': [ref.model_dump(mode='json') for ref in references],
              'rows': [row.model_dump(mode='json') for row in rows],
              'clinical_validation': 'NOT_VALIDATED'}
    return {'path': str(_save_pilot_json(root, 'imaging/pilot-comparisons', result)), 'status': 'COMPARED'}


def review_observations(root, sealed_path, review_path):
    """Record confirmed observations and append explicit corrections or withdrawals."""
    from .ledger import deterministic_id

    root = _store(root)
    sealed, digest = _load_seal(root, sealed_path)
    review = read_json(review_path)
    if (review.get('result_sha256') != digest or review['case_id'] != sealed['case_id'] or not review['reviewer'].strip()
            or review['source_pixels_reviewed'] is not True):
        raise ValueError('same-case review against original pixels required')
    facts, withdrawals, replacements = [], [], {}
    for item in review['observations']:
        modality = item['modality']
        index = item['candidate_index']
        candidates = sealed['results'][modality]['findings']['candidates']
        if type(index) is not int or not 1 <= index <= len(candidates):
            raise ValueError('review must reference an existing candidate')
        fact_id = deterministic_id('visual-review', {'seal': digest, 'modality': modality, 'index': index})
        if fact_id in replacements:
            raise ValueError('review must contain distinct observations')
        replacements[fact_id] = item.get('replaces_event_id')
        if item.get('withdraw') is True:
            if item.get('observation') is not None or not replacements[fact_id]:
                raise ValueError('withdrawal requires only the current review event and candidate')
            withdrawals.append(fact_id)
            continue
        statement = item.get('observation', '').strip()
        if not statement:
            raise ValueError('reviewed observation required')
        if re.search(r'\d\s*(?:mm|cm|degrees?|°|도|밀리미터|센티미터|%)', statement, re.I):
            raise ValueError('quantitative values require the Slicer measurement path')
        candidate = candidates[index - 1]
        facts.append({
            'fact_id': fact_id, 'fact_key': 'reviewed_visual_observation', 'value': statement,
            'source_locators': [f"image-pilot:{digest}:{candidate['image_id']}:{index}"],
            'user_confirmed': True, 'confirmation_scope': 'IMAGE_OBSERVATION_REVIEWED',
            'scope': {'modality': modality, 'assertion': 'UNKNOWN', 'source_kind': 'UNKNOWN'},
        })
    if not replacements:
        raise ValueError('review must contain at least one observation')
    payload = {'case_id': sealed['case_id'], 'reviewer': review['reviewer'], 'seal_sha256': digest, 'facts': facts}
    if any(replacements.values()):
        reason = review.get('reason', '').strip()
        if not reason:
            raise ValueError('an amendment or withdrawal requires a reason')
        payload.update(reason=reason, replaces={key: event for key, event in replacements.items() if event},
                       withdrawn_fact_ids=withdrawals)
    ledger = AppendOnlyJsonlLedger(_inside(root, root / 'ledger.jsonl'))
    kind = 'IMAGE_OBSERVATIONS_REVIEWED'
    key = deterministic_id('visual-review-event', payload)
    with ledger.key_lock('image-observation-review', sealed['case_id']):
        history = [r for r in ledger.records() if r['kind'] == kind and r['payload']['case_id'] == sealed['case_id']]
        existing = next((r for r in history if r['natural_key'] == key), None)
        heads, inactive = {}, set()
        for record in history:
            for fact in record['payload']['facts']:
                heads[fact['fact_id']] = record['record_id']
                inactive.discard(fact['fact_id'])
            for identifier in record['payload'].get('withdrawn_fact_ids', ()):
                heads[identifier] = record['record_id']
                inactive.add(identifier)
        if not existing:
            if any(heads.get(identifier) != event for identifier, event in replacements.items()):
                raise ValueError('explicit replacement of the current review event is required')
            if inactive.intersection(withdrawals):
                raise ValueError('only an active observation can be withdrawn')
        event_id = ledger.append_once(kind, key, payload)
    return {'status': 'WITHDRAWN' if withdrawals and not facts else 'USER_REVIEWED',
            'event_id': event_id, 'fact_ids': [f['fact_id'] for f in facts], 'withdrawn_fact_ids': withdrawals}


def register(parsers):
    for action in ('prepare-pilot', 'redact-pilot', 'analyze-pilot', 'status-pilot', 'compare-pilot', 'review-pilot'):
        parser = parsers.add_parser(action)
        parser.add_argument('--store', type=Path, default=Path('.medic'))
        if action == 'prepare-pilot':
            parser.add_argument('--manifest', type=Path, required=True)
            for name in ('case', 'ct-series', 'xray-series'):
                parser.add_argument('--' + name, required=True)
            parser.add_argument('--xray-hash', action='append', default=[])
        elif action == 'redact-pilot':
            parser.add_argument('--input', type=Path, required=True)
            parser.add_argument('--masks', type=Path, required=True)
        elif action == 'status-pilot':
            parser.add_argument('--run-id', required=True)
        elif action == 'analyze-pilot':
            parser.add_argument('--input', type=Path, required=True)
            parser.add_argument('--run-id', required=True)
            parser.add_argument('--input-sha256', required=True)
            parser.add_argument('--reviewer', required=True)
            parser.add_argument('--codex-calls', type=int, required=True)
            parser.add_argument('--approve-transmission', action='store_true')
        else:
            parser.add_argument('--sealed', type=Path, required=True)
            parser.add_argument('--review', type=Path, required=True)
            if action == 'compare-pilot':
                parser.add_argument('--pdf-password-prompt', action='store_true')


def execute(args):
    if args.command != 'dicom':
        return None
    if args.action == 'prepare-pilot':
        return 0, prepare(args.store, read_json(args.manifest), args.case, args.ct_series,
                          args.xray_series, args.xray_hash)
    if args.action == 'status-pilot':
        return 0, pilot_status(args.store, args.run_id)
    if args.action == 'compare-pilot':
        from getpass import getpass

        password = getpass('Reference PDF password: ') if args.pdf_password_prompt else None
        return 0, compare(args.store, args.sealed, args.review, password=password)
    if args.action == 'review-pilot':
        return 0, review_observations(args.store, args.sealed, args.review)
    if args.action == 'redact-pilot':
        return 0, redact(args.store, args.input, args.masks)
    if args.action == 'analyze-pilot':
        from .config import ModelRegistry, CostEstimate
        from .providers import build_live_gateways

        registry = ModelRegistry.for_cli()
        estimate = CostEstimate(pricing_as_of=registry.pricing_as_of, lines=(), call_count=2,
                                estimated_usd=None, approved=True, backend='codex',
                                codex_calls=args.codex_calls, approval_id=args.run_id,
                                transmission_approved=args.approve_transmission)
        return 0, analyze(args.store, args.run_id, args.input, args.input_sha256, args.reviewer,
                          estimate, registry, build_live_gateways(registry, backend='codex')['openai'])
    return None
