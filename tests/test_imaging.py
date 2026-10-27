import json
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid


def dicom(path, *, study=None, series=None, sop=None, z=0, modality='CT', **attrs):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop or generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study or generate_uid()
    ds.SeriesInstanceUID = series or generate_uid()
    ds.FrameOfReferenceUID = '1.2.3.99'
    ds.Modality = modality
    ds.Rows = 4
    ds.Columns = 5
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, z]
    ds.PixelSpacing = [2, 3]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = b'\0' * 40
    for k, v in attrs.items():
        setattr(ds, k, v)
    ds.save_as(path, enforce_file_format=True)
    return ds


def test_inspect_export_group_and_conflicts(tmp_path):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    root = tmp_path / 'input'
    root.mkdir()
    a = dicom(root / 'extensionless')
    dicom(root / 'second', study=a.StudyInstanceUID, series=a.SeriesInstanceUID, z=4)
    (root / 'unknown').write_bytes(b'not medical')
    manifest = inspect_inventory(scan(root), tmp_path / '.medic' / 'dicom')
    json.dumps(manifest, allow_nan=False)
    assert len(manifest['instances']) == 2
    s = manifest['series'][0]
    assert s['measurement_eligible']
    assert s['weight_bearing'] == 'UNKNOWN'
    assert np.allclose(np.array(s['geometry']['affine_lps'])[:3, :3], np.diag([3, 2, 4]))
    assert len(list((tmp_path / '.medic' / 'dicom').glob('*.dcm'))) == 2
    dicom(root / 'conflict', study=a.StudyInstanceUID, series=a.SeriesInstanceUID,
          sop=a.SOPInstanceUID, z=9)
    manifest = inspect_inventory(scan(root), tmp_path / '.medic' / 'dicom')
    assert manifest['conflicts']
    assert not manifest['series'][0]['measurement_eligible']


@pytest.mark.parametrize('attrs', [
    {'modality': 'NM'}, {'NumberOfFrames': 2}, {'PixelSpacing': [0, 2]},
    {'ImageOrientationPatient': [1, 0, 0, 1, 0, 0]}, {'ImagePositionPatient': [float('nan'), 0, 0]},
])
def test_invalid_and_unsupported_inspect_only(tmp_path, attrs):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    a = dicom(tmp_path / 'a', **attrs)
    dicom(tmp_path / 'b', study=a.StudyInstanceUID, series=a.SeriesInstanceUID, z=4, **attrs)
    m = inspect_inventory(scan(tmp_path), tmp_path / '.medic' / 'dicom')
    assert not m['series'][0]['measurement_eligible']
    json.dumps(m, allow_nan=False)


def test_oblique_geometry_metrics_and_roi():
    from medic.imaging_geometry import validate_geometry, voxel_to_lps, ras_to_lps, measure, lesion_metrics
    q = 2 ** -0.5
    frames = [dict(position=[0, 0, z], orientation=[q,q,0,-q,q,0], spacing=[2,3], rows=4, columns=5) for z in [0,4,8]]
    g = validate_geometry(frames)
    assert np.allclose(voxel_to_lps(g, [1,2,1]), [-q,7*q,4])
    assert ras_to_lps([1,2,3]) == [-1,-2,3]
    assert measure([[0,0,0],[3,4,0]], source='series', kind='line')['value'] == 5
    assert measure([[1,0,0],[0,0,0],[0,1,0]], source='series', kind='angle')['value'] == 90
    with pytest.raises(ValueError):
        measure([[0,0,0],[1,1,1]], source='series', kind='line', protocol='named')
    lesion = np.zeros((5,4,3), dtype=bool)
    lesion[0,0,0] = True
    result = lesion_metrics(lesion, g)
    assert result['volume_mm3'] == pytest.approx(24)
    assert result['fraction_of_evaluation_roi'] is None
    roi = np.ones_like(lesion)
    assert lesion_metrics(lesion, g, roi)['fraction_of_evaluation_roi'] == pytest.approx(1/60)
    with pytest.raises(ValueError):
        lesion_metrics(lesion, g, np.zeros_like(lesion))


def test_geometry_mismatch_rejected():
    from medic.imaging_geometry import validate_geometry
    frame = dict(position=[0,0,0], orientation=[1,0,0,0,1,0], spacing=[1,2], rows=3, columns=4)
    for change in [{'position':[0,0,0]}, {'spacing':[2,2]}, {'position':[1,0,2]}, {'orientation':[0,1,0,1,0,0]}]:
        with pytest.raises(ValueError):
            validate_geometry([frame, dict(frame, position=[0,0,2], **{k:v for k,v in change.items() if k != 'position'}) if 'position' not in change else dict(frame, **change)])


def test_calibrated_dx_plane_and_uncalibrated_gate(tmp_path):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    from medic.imaging_geometry import measure_series
    from pydicom.uid import DigitalXRayImageStorageForPresentation
    p = tmp_path / 'dx'
    ds = dicom(p, modality='DX', PixelSpacingCalibrationType='FIDUCIAL')
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    ds.save_as(p, enforce_file_format=True)
    m = inspect_inventory(scan(tmp_path), tmp_path / '.medic' / 'dicom')
    s = m['series'][0]
    assert s['measurement_eligible']
    assert s['geometry']['shape'] == [5,4,1]
    assert measure_series(s, [[0,0,0],[3,2,0]])['value'] == pytest.approx(13**0.5)
    with pytest.raises(ValueError):
        measure_series(s, [[0,0,0],[3,2,1]])
    del ds.PixelSpacingCalibrationType
    ds.save_as(p, enforce_file_format=True)
    s = inspect_inventory(scan(tmp_path), tmp_path / '.medic' / 'dicom')['series'][0]
    assert not s['measurement_eligible']
    with pytest.raises(ValueError):
        measure_series(s, [[0,0,0],[3,2,0]])


def test_slicer_review_provenance_and_coordinates(tmp_path):
    from copy import deepcopy
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    from medic.slicer_review import import_review, review_template
    a = dicom(tmp_path / 'a')
    dicom(tmp_path / 'b', study=a.StudyInstanceUID, series=a.SeriesInstanceUID, z=4)
    m = inspect_inventory(scan(tmp_path), tmp_path / '.medic' / 'dicom')
    review = review_template(m, a.SeriesInstanceUID)
    review['markups'] = [{'kind':'line','coordinate_system':'RAS','points':[[-3,-2,0],[-6,-2,0]]}]
    result = import_review(review, m)
    assert result['measurements'][0]['value'] == pytest.approx(3)
    for field, value in [('original_hashes', []), ('series_uid','1.2.9'), ('affine_lps',np.eye(4).tolist())]:
        bad = deepcopy(review)
        bad[field] = value
        with pytest.raises(ValueError):
            import_review(bad, m)
    review['markups'][0]['points'][0] = [100000,0,0]
    with pytest.raises(ValueError):
        import_review(review, m)


def test_segmentation_review_validates_hash_and_geometry(tmp_path):
    import nrrd
    from medic.inventory import scan, digest
    from medic.imaging import inspect_inventory
    from medic.slicer_review import import_review, review_template
    a = dicom(tmp_path / 'a')
    dicom(tmp_path / 'b', study=a.StudyInstanceUID, series=a.SeriesInstanceUID, z=4)
    m = inspect_inventory(scan(tmp_path), tmp_path / '.medic' / 'dicom')
    review = review_template(m, a.SeriesInstanceUID)
    out = tmp_path / '.medic' / 'review'
    out.mkdir()
    mask = np.zeros((5,4,2), dtype=np.uint8)
    mask[0,0,0] = 1
    nrrd.write(str(out / 'lesions.nrrd'), mask, header={'space':'left-posterior-superior',
        'space directions':np.diag([3.,2.,4.]), 'space origin':[0.,0.,0.]})
    item = {'path':'lesions.nrrd','sha256':digest((out/'lesions.nrrd').read_bytes()),'role':'manual_lesion'}
    review['segmentations'] = [item]
    r = import_review(review,m,review_dir=out)
    assert r['segmentations'][0]['metrics']['volume_mm3'] == pytest.approx(24)
    assert r['segmentations'][0]['metrics']['fraction_of_evaluation_roi'] is None
    item['sha256'] = 'sha256:' + '0'*64
    with pytest.raises(ValueError):
        import_review(review,m,review_dir=out)


def test_dicomdir_and_preambleless(tmp_path):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    a = dicom(tmp_path / 'raw')
    (tmp_path / 'raw').write_bytes((tmp_path / 'raw').read_bytes()[132:])
    ds = dicom(tmp_path / 'DICOMDIR')
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.1.3.10'
    ds.save_as(tmp_path / 'DICOMDIR', enforce_file_format=True)
    m = inspect_inventory(scan(tmp_path),tmp_path / '.medic' / 'dicom')
    assert len(m['directories']) == 1
    assert m['instances'][0]['sop_uid'] == a.SOPInstanceUID


def test_sop_metadata_mismatch_and_missing_pixels_reject(tmp_path):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    a = dicom(tmp_path / 'a')
    b = dicom(tmp_path / 'b',study=a.StudyInstanceUID,series=a.SeriesInstanceUID,z=4)
    del b.PixelData
    b.save_as(tmp_path / 'b',enforce_file_format=True)
    m = inspect_inventory(scan(tmp_path),tmp_path / '.medic' / 'dicom')
    assert not m['series'][0]['measurement_eligible']


def test_loaded_slicer_geometry_requires_exact_reference():
    from medic.slicer_review import validate_loaded_geometry
    g = {'shape':[5,4,3], 'affine_lps':np.diag([3.,2.,4.,1.]).tolist()}
    ras = np.diag([-3.,-2.,4.,1.])
    validate_loaded_geometry(g, ras, [5,4,3])
    with pytest.raises(ValueError):
        validate_loaded_geometry(g, np.eye(4), [5,4,3])
    with pytest.raises(ValueError):
        validate_loaded_geometry(g, ras, [5,4,2])


def test_series_uid_reused_in_different_study_is_blocked(tmp_path):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    a = dicom(tmp_path / 'a')
    dicom(tmp_path/'b',study=a.StudyInstanceUID,series=a.SeriesInstanceUID,z=4)
    dicom(tmp_path/'c',series=a.SeriesInstanceUID,z=8)
    m = inspect_inventory(scan(tmp_path),tmp_path/'.medic'/'dicom')
    assert all(not s['measurement_eligible'] for s in m['series'])


def test_protocol_and_plane_and_degenerate_angle():
    from medic.imaging_geometry import measure
    plane = {'origin':[0,0,0],'normal':[0,0,1]}
    axes = {'x':[1,0,0],'y':[0,1,0],'source':'explicit test protocol reference'}
    m = measure([[0,0,0],[3,4,9]], source='series',plane=plane,protocol='test',axes=axes)
    assert m['value'] == pytest.approx(5)
    with pytest.raises(ValueError):
        measure([[0,0,0],[0,0,0],[1,0,0]],source='series',kind='angle')


def test_deflated_dataset_rejected_before_unbounded_parser(tmp_path, monkeypatch):
    import pydicom
    from pydicom.uid import DeflatedExplicitVRLittleEndian
    from medic.imaging import _inspect
    path = tmp_path/'deflated'
    ds = dicom(path)
    ds.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
    ds.save_as(path,enforce_file_format=True)
    called = []
    original = pydicom.dcmread
    def spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(pydicom,'dcmread',spy)
    with pytest.raises(ValueError,match='deflated'):
        _inspect(path.read_bytes())
    assert not called
