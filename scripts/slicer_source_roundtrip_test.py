"""Local source technical validation; exports no diagnostic measurements or lesions."""
import json
from pathlib import Path
import runpy
import sys


def run_source(config):
    import numpy as np
    import pydicom
    from pydicom.pixels import apply_modality_lut
    import slicer
    from medic.slicer_review import import_review
    if config.get('kind') != 'LOCAL_SOURCE_TECHNICAL_ONLY':
        raise ValueError('local source technical configuration required')
    manifest = json.loads(Path(config['manifest_path']).read_text(encoding='utf-8'))
    output = Path(config['output'])
    bridge = runpy.run_path(str(Path(__file__).with_name('slicer_review.py')))
    volume = bridge['load_series'](manifest, config['series_uid'], output / 'runtime')
    series = next(s for s in manifest['series'] if s['series_uid'] == config['series_uid'])
    affine = np.asarray(series['geometry']['affine_lps'])
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    unique = {item['sha256']: item for item in manifest['instances']
              if item['sha256'] in series['original_hashes']}
    instances = sorted(unique.values(), key=lambda item: np.dot(item['position'], normal))
    array = slicer.util.arrayFromVolume(volume)
    if array.shape != tuple(reversed(series['geometry']['shape'])) or len(instances) != array.shape[0]:
        raise ValueError('source frame count or shape differs from loaded volume')
    for index, item in enumerate(instances):
        dataset = pydicom.dcmread(item['stored_path'])
        expected = apply_modality_lut(dataset.pixel_array, dataset)
        if not np.isfinite(array[index]).all() or not np.allclose(array[index], expected, atol=1e-6, rtol=0):
            raise ValueError(f'loaded pixel values/order differ at frame {index}')
    review_path = bridge['export_review'](manifest, volume, [], output / 'exports')
    imported = import_review(json.loads(Path(review_path).read_text(encoding='utf-8')),
                             manifest, review_dir=Path(review_path).parent)
    if imported['measurements'] or imported['segmentations'] or imported['original_hashes'] != series['original_hashes']:
        raise ValueError('unexpected annotations or changed provenance')
    return {'status': 'PASSED', 'kind': 'LOCAL_SOURCE_TECHNICAL_ONLY',
            'slicer_version': slicer.app.applicationVersion, 'shape': list(array.shape),
            'frames_verified': len(instances), 'review_path': review_path,
            'measurement_count': 0, 'segmentation_count': 0, 'geometry_and_hashes_verified': True,
            'clinical_validation': 'NOT_VALIDATED', 'visual_quality': 'NOT_ASSESSED'}


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    runner = runpy.run_path(str(Path(__file__).with_name('slicer_roundtrip_test.py')))
    runner['main'](task=run_source)
