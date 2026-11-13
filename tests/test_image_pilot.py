import json
from pathlib import Path

import pytest
from PIL import Image
from pydicom.uid import DigitalXRayImageStorageForPresentation

from medic.cli import main
from medic.config import CostEstimate, ModelRegistry
from medic.extended_cli import save_json
from medic.image_pilot import analyze, compare, prepare, redact, review_observations
from medic.imaging import inspect_inventory
from medic.inventory import scan
from medic.ledger import sha256_digest
from medic.providers import ApprovedGateway, RawProviderResponse
from tests.test_imaging import dicom


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / '.medic'
    source = tmp_path / 'source'
    source.mkdir()
    ct = dicom(source / 'ct1', PatientID='synthetic', RescaleSlope=1, RescaleIntercept=-1000)
    dicom(source / 'ct2', study=ct.StudyInstanceUID, series=ct.SeriesInstanceUID, z=4,
          PatientID='synthetic', RescaleSlope=1, RescaleIntercept=-1000)
    dx = dicom(source / 'dx', modality='DX', PatientID='synthetic', ViewPosition='AP')
    dx.SOPClassUID = dx.file_meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    dx.save_as(source / 'dx', enforce_file_format=True)
    manifest = inspect_inventory(scan(source), root / 'dicom')
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return root, path, ct, dx


def test_prepare_local_ct_and_uncalibrated_xray(sources, capsys):
    root, path, ct, dx = sources
    args = ['dicom', 'prepare-pilot', '--case', 'synthetic', '--manifest', str(path),
            '--ct-series', ct.SeriesInstanceUID, '--xray-series', dx.SeriesInstanceUID,
            '--store', str(root)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['ct_images'] >= 3
    assert result['xray_images'] == 1
    assert result['xray_measurement_eligible'] is False
    assert result['status'] == 'REVIEW_REQUIRED'
    assert 'XRAY-1' in Path(result['preview_path']).read_text(encoding='utf-8')


def test_rejects_ct_with_noncanonical_slice_order(sources):
    root, path, ct, dx = sources
    manifest = json.loads(path.read_bytes())
    series = next(s for s in manifest['series'] if s['series_uid'] == ct.SeriesInstanceUID)
    series['ordered_hashes'].reverse()
    with pytest.raises(ValueError, match='CT geometry mismatch'):
        prepare(root, manifest, 'synthetic', ct.SeriesInstanceUID, dx.SeriesInstanceUID)


def test_image_calls_are_blind_and_sealed_before_source_comparison(sources, tmp_path):
    root, path, ct, dx = sources
    result = prepare(root, json.loads(path.read_text(encoding='utf-8')), 'synthetic',
                     ct.SeriesInstanceUID, dx.SeriesInstanceUID)
    registry = ModelRegistry.for_cli()
    estimate = CostEstimate(pricing_as_of=registry.pricing_as_of, lines=(), call_count=2, estimated_usd=None,
                            approved=True, backend='codex', codex_calls=2)
    calls = []
    mutable_pixel = None

    class FakeCodex:
        backend = 'codex'
        provider = 'openai'
        model = 'test'

        def generate_raw(self, *, stage, prompt, image_paths, **kwargs):
            calls.append((stage, len(image_paths)))
            inputs = json.loads(prompt.split('\n')[-1])
            assert 'secret report' not in prompt
            assert 'study_uid' not in prompt
            if stage.endswith('XRAY'):
                with Image.open(image_paths[0]) as image:
                    assert image.getpixel((0, 0)) == 0
            elif mutable_pixel is not None:
                mutable_pixel.write_bytes(b'changed after approval check')
            return RawProviderResponse({'candidates': [{
                'image_id': inputs[0]['image_id'], 'location': 'upper portion',
                'region': {'left': 0.1, 'top': 0.1, 'right': 0.8, 'bottom': 0.8},
                'observation': 'A visible contour', 'uncertainty': 'Partial coverage',
            }], 'limitations': ['Selected images only']}, stage)

    gateway = ApprovedGateway(FakeCodex())
    with pytest.raises(PermissionError):
        analyze(root, 'synthetic', result['path'], result['input_sha256'], 'reviewer', estimate, registry, gateway)
    estimate = estimate.model_copy(update={'transmission_approved': True})
    masks = tmp_path / 'masks.json'
    masks.write_text('{"XRAY-1":[[0,0,1,1]]}', encoding='utf-8')
    masked = redact(root, result['path'], masks)
    assert masked['input_sha256'] != result['input_sha256']
    with pytest.raises(ValueError, match='hash mismatch'):
        analyze(root, 'synthetic', masked['path'], result['input_sha256'], 'reviewer', estimate, registry, gateway)
    tampered = json.loads(Path(masked['path']).read_bytes())
    tampered['images'][0]['display']['report'] = 'secret report'
    invalid = save_json(root, 'imaging/pilot-inputs', tampered)
    with pytest.raises(ValueError):
        analyze(root, 'synthetic', invalid, 'sha256:' + invalid.stem, 'reviewer', estimate, registry, gateway)
    args = (root, 'synthetic', masked['path'], masked['input_sha256'], 'reviewer', estimate, registry, gateway)
    dx_preview = next(i for i in tampered['images'] if i['modality'] == 'XRAY')
    mutable_pixel = root / 'imaging' / 'pilot-pixels' / (dx_preview['sha256'].split(':')[1] + '.png')
    original_bytes = mutable_pixel.read_bytes()
    sealed = analyze(*args)
    mutable_pixel.write_bytes(original_bytes)
    assert analyze(*args) == sealed
    assert [stage for stage, _ in calls] == ['IMAGE_PILOT_CT', 'IMAGE_PILOT_XRAY']

    report = tmp_path / 'reference.txt'
    report.write_text('A contour is visible.', encoding='utf-8')
    review = {'result_sha256': sealed['result_sha256'], 'case_id': 'synthetic', 'modality': 'XRAY',
              'study_uid': dx.StudyInstanceUID, 'reviewer': 'reviewer',
              'same_exam_confirmed': True, 'report_path': str(report), 'report_sha256': sha256_digest(report.read_bytes()),
              'references': [{'page': 1, 'quote': 'invented wording'}],
              'rows': [{'candidate_index': 1, 'reference_index': 1,
                        'relationship': 'AGREEMENT', 'explanation': 'same contour'}]}
    review_path = tmp_path / 'review.json'
    review_path.write_text(json.dumps({**review, 'result_sha256': 'other-seal'}), encoding='utf-8')
    with pytest.raises(ValueError, match='same-case'):
        compare(root, sealed['path'], review_path)
    review_path.write_text(json.dumps(review), encoding='utf-8')
    with pytest.raises(ValueError, match='quote'):
        compare(root, sealed['path'], review_path)
    review['references'][0]['quote'] = 'A contour is visible.'
    review_path.write_text(json.dumps(review), encoding='utf-8')
    assert compare(root, sealed['path'], review_path)['status'] == 'COMPARED'

    observation = {'result_sha256': sealed['result_sha256'], 'case_id': 'synthetic', 'reviewer': 'reviewer',
                   'source_pixels_reviewed': True,
                   'observations': [{'modality': 'XRAY', 'candidate_index': 1, 'observation': '10 mm'}]}
    review_path.write_text(json.dumps(observation), encoding='utf-8')
    with pytest.raises(ValueError, match='quantitative'):
        review_observations(root, sealed['path'], review_path)
    observation['observations'][0]['observation'] = 'A contour is visible.'
    review_path.write_text(json.dumps(observation), encoding='utf-8')
    assert review_observations(root, sealed['path'], review_path)['status'] == 'USER_REVIEWED'
