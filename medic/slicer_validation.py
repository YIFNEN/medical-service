"""Prepare a synthetic-only Slicer round trip with independent analytic answers."""
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from .imaging import inspect_inventory
from .inventory import scan, _linked


def check_metrics(observed, expected, tolerance=1e-6):
    for key, target in expected.items():
        value = observed.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or abs(value - target) > tolerance):
            raise ValueError(f'Slicer round-trip metric mismatch: {key}')


def check_pixels(array, expected_hash):
    array = np.asarray(array)
    if (array.shape != (4, 16, 16) or array.dtype.kind not in 'uif'
            or not np.isfinite(array).all() or (array < 0).any() or (array > 65535).any()
            or not np.array_equal(array, np.floor(array))
            or hashlib.sha256(array.astype('<u2').tobytes()).hexdigest() != expected_hash):
        raise ValueError('loaded pixel content/order differs from synthetic source')


def check_mask(exported_ijk, expected_kji):
    expected_ijk = np.transpose(expected_kji, (2, 1, 0)) != 0
    if not np.array_equal(np.asarray(exported_ijk) != 0, expected_ijk):
        raise ValueError('exported segmentation voxel positions differ from placed region')


def _validation_output(store, prefix='synthetic-'):
    store = Path(store).absolute()
    if ('.medic' not in store.parts or str(store).startswith('\\\\')
            or any(p.exists() and _linked(p) for p in (store, *store.parents))):
        raise ValueError('local nonlinked .medic store required')
    parent = store / 'validation' / 'slicer'
    if any(p.exists() and _linked(p) for p in (parent, *parent.parents)):
        raise ValueError('linked validation directory')
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    return output


def prepare_validation(store):
    output = _validation_output(store)
    source = output / 'input'
    source.mkdir()
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    cosine, sine = math.cos(math.pi / 6), math.sin(math.pi / 6)
    pixels = np.indices((4, 16, 16))
    pixels = (pixels[2] + 10 * pixels[1] + 100 * pixels[0]).astype('<u2')
    for index in range(4):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        path = source / f'slice-{index}.dcm'
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.PatientName, ds.PatientID = 'SYNTHETIC^TEST', 'MEDIC_SYNTHETIC'
        ds.StudyDate, ds.StudyTime, ds.Modality = '20000101', '120000', 'CT'
        ds.StudyID, ds.SeriesNumber, ds.InstanceNumber = 'SYNTHETIC', 1, index + 1
        ds.Rows, ds.Columns = 16, 16
        ds.ImageOrientationPatient = [f'{v:.12g}' for v in (cosine, sine, 0, -sine, cosine, 0)]
        ds.ImagePositionPatient = [30, -40, 10 + index * 4]
        ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [2, 3], 4, 4
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = pixels[index].tobytes()
        ds.save_as(path, enforce_file_format=True)
    manifest = inspect_inventory(scan(source), output / 'originals')
    if len(manifest['series']) != 1 or not manifest['series'][0]['measurement_eligible']:
        raise ValueError('synthetic CT failed inspection')
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2), encoding='utf-8')
    config = {'version': 1, 'kind': 'SYNTHETIC_ONLY', 'series_uid': series,
              'manifest_path': str(manifest_path), 'output': str(output),
              'result_path': str(output / 'result.json'),
              'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
              'line_voxels': [[1, 1, 1], [4, 5, 1]],
              'angle_voxels': [[2, 2, 1], [5, 2, 1], [5, 6, 1]],
              'expected': {'distance_mm': math.sqrt(145), 'angle_degrees': 90.0,
                           'volume_mm3': 576.0, 'evaluation_roi_volume_mm3': 6912.0,
                           'fraction_of_evaluation_roi': 1 / 12}}
    return _package(output, config, 'slicer_roundtrip_test.py')


def prepare_source_validation(store, manifest_path, series_uid=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    eligible = [s for s in manifest['series'] if s['measurement_eligible']
                and (series_uid is None or s['series_uid'] == series_uid)]
    if not eligible:
        raise ValueError('no eligible source series for technical validation')
    output = _validation_output(store, prefix='source-')
    config = {'version': 1, 'kind': 'LOCAL_SOURCE_TECHNICAL_ONLY',
              'series_uid': eligible[0]['series_uid'], 'manifest_path': str(manifest_path),
              'output': str(output), 'result_path': str(output / 'result.json')}
    return _package(output, config, 'slicer_source_roundtrip_test.py')


def _package(output, config, script_name):
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, allow_nan=False, indent=2), encoding='utf-8')
    script = Path(__file__).resolve().parent.parent / 'scripts' / script_name
    return {'status': 'PREPARED_NOT_RUN', 'config_path': str(config_path), 'result_path': config['result_path'],
            'environment': {'MEDIC_SLICER_VALIDATION_CONFIG': str(config_path)},
            'command': ['Slicer.exe', '--ignore-slicerrc', '--no-splash', '--no-main-window',
                        '--testing', '--python-script', str(script)],
            'clinical_validation': 'NOT_VALIDATED'}
