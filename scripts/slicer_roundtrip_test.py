"""Run with Slicer --python-script; config path comes from the documented environment variable."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


def run(config):
    import numpy as np
    import slicer
    import vtk
    import nrrd  # Preflight before importing or exporting any nodes.
    from medic.slicer_validation import check_metrics, check_pixels, check_mask
    from medic.slicer_review import import_review, _mask

    if config.get('kind') != 'SYNTHETIC_ONLY':
        raise ValueError('synthetic validation configuration required')
    manifest = json.loads(Path(config['manifest_path']).read_text(encoding='utf-8'))
    if (len(manifest['series']) != 1 or manifest['series'][0]['geometry']['shape'] != [16, 16, 4]
            or len(manifest['instances']) != 4):
        raise ValueError('unexpected synthetic fixture')
    output = Path(config['output'])
    bridge = runpy.run_path(str(Path(__file__).with_name('slicer_review.py')))
    volume = bridge['load_series'](manifest, config['series_uid'], output / 'runtime')
    array = slicer.util.arrayFromVolume(volume)
    check_pixels(array, config['pixel_sha256'])
    affine = np.array(manifest['series'][0]['geometry']['affine_lps'])
    nodes = []
    expected_ras = []
    for kind, points in (('Line', config['line_voxels']), ('Angle', config['angle_voxels'])):
        node = slicer.mrmlScene.AddNewNodeByClass(f'vtkMRMLMarkups{kind}Node')
        bridge['bind_markup'](volume, node)
        node.SetName('Synthetic ' + kind)
        node.SetAttribute('Medic.MeasurementID', 'synthetic-' + kind.lower())
        node.SetAttribute('Medic.Metadata', json.dumps({'method': 'synthetic-known-landmarks', 'protocol_version': '1'}))
        ras_points = []
        for point in points:
            lps = (affine @ np.array([*point, 1.0]))[:3]
            ras = lps * [-1, -1, 1]
            node.AddControlPoint(vtk.vtkVector3d(*ras))
            ras_points.append(ras)
        nodes.append(node)
        expected_ras.append(ras_points)
    segmentations = []
    expected_masks = []
    for name, mask in (('Synthetic lesion', np.zeros(array.shape, dtype=np.uint8)),
                       ('Synthetic evaluation ROI', np.zeros(array.shape, dtype=np.uint8))):
        if name == 'Synthetic lesion':
            mask[1:3, 4:7, 5:9] = 1
        else:
            mask[:, 2:10, 3:12] = 1
        segmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        segment_id = segmentation.GetSegmentation().AddEmptySegment('', name)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, segmentation, segment_id, volume)
        segmentations.append(segmentation)
        expected_masks.append(mask.copy())
    review_path = bridge['export_review'](manifest, volume, nodes, output / 'exports',
                                         segmentation=segmentations[0], evaluation_roi=segmentations[1])
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    for markup, expected in zip(review['markups'], expected_ras):
        if markup['coordinate_system'] != 'RAS' or not np.allclose(markup['points'], expected, atol=1e-6, rtol=0):
            raise ValueError('exported markup coordinates differ from placed points')
    imported = import_review(review, manifest, review_dir=Path(review_path).parent)
    for index, kind in enumerate(('Line', 'Angle')):
        observation = imported['measurements'][index]
        if (observation['measurement_id'] != 'synthetic-' + kind.lower() or observation['name'] != 'Synthetic ' + kind
                or observation['metadata'] != {'method': 'synthetic-known-landmarks', 'protocol_version': '1'}):
            raise ValueError('measurement metadata did not survive Slicer round trip')
    exported_roles = [item['role'] for item in review['segmentations']]
    if sorted(exported_roles) != ['evaluation_roi', 'manual_lesion']:
        raise ValueError('missing or duplicated segmentation roles')
    for item in review['segmentations']:
        expected = expected_masks[0 if item['role'] == 'manual_lesion' else 1]
        decoded = _mask(item, Path(review_path).parent, manifest['series'][0]['geometry'])
        check_mask(decoded, expected)
    if len(imported['measurements']) != 2 or len(imported['segmentations']) != 1:
        raise ValueError('missing or duplicated exported observations')
    metrics = imported['segmentations'][0]['metrics']
    observed = {'distance_mm': imported['measurements'][0]['value'],
                'angle_degrees': imported['measurements'][1]['value'],
                **{k: metrics[k] for k in ('volume_mm3', 'evaluation_roi_volume_mm3', 'fraction_of_evaluation_roi')}}
    check_metrics(observed, config['expected'])
    check_metrics({'distance_mm': nodes[0].GetLineLengthWorld(), 'angle_degrees': nodes[1].GetAngleDegrees()},
                  {k: config['expected'][k] for k in ('distance_mm', 'angle_degrees')})
    if imported['unmarked_tissue'] != 'NOT_EVALUATED':
        raise ValueError('unexpected interpretation of unmarked tissue')
    return {'status': 'PASSED', 'kind': 'SYNTHETIC_ONLY', 'slicer_version': slicer.app.applicationVersion,
            'observed': observed, 'review_path': review_path, 'pixel_content_verified': True,
            'coordinate_tolerance_mm': 1e-6, 'clinical_validation': 'NOT_VALIDATED'}


def _write_result(config, result):
    with Path(config['result_path']).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False, indent=2)


def main(config_path=None, task=None):
    import slicer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    config = None
    result = {'status': 'FAILED', 'clinical_validation': 'NOT_VALIDATED'}
    try:
        config = json.loads(Path(config_path or os.environ['MEDIC_SLICER_VALIDATION_CONFIG']).read_text(encoding='utf-8'))
        if Path(config['result_path']).exists():
            raise FileExistsError('validation result already exists; prepare a new bundle')
        result = (task or run)(config)
        _write_result(config, result)
    except Exception as exc:
        result = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}',
                  'traceback': traceback.format_exc(), 'clinical_validation': 'NOT_VALIDATED'}
        if config is not None and not Path(config['result_path']).exists():
            try:
                _write_result(config, result)
            except OSError:
                pass
    finally:
        try:
            print(json.dumps(result))
        finally:
            slicer.app.exit(0 if result['status'] == 'PASSED' else 1)
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    main()
