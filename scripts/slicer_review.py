"""Run inside 3D Slicer's Python console, with the repository on sys.path.

load_series(manifest, series_uid, workspace) -> scalar volume node
export_review(manifest, volume, markup_nodes, output, segmentation=None, evaluation_roi=None)
All workspace/output directories must be local and beneath .medic.
"""
import json
from pathlib import Path
import tempfile
import uuid

from medic.inventory import digest, _linked
from medic.slicer_review import review_template, validate_loaded_geometry, import_review


def _local_directory(path):
    path = Path(path).absolute()
    if '.medic' not in path.parts or str(path).startswith('\\\\'):
        raise ValueError('local .medic directory required')
    if any(p.exists() and _linked(p) for p in (path,*path.parents)):
        raise ValueError('linked output is not permitted')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_volume(volume, geometry):
    import vtk
    if volume.GetParentTransformNode() is not None:
        raise ValueError('transformed reference volumes are unsupported')
    matrix = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(matrix)
    validate_loaded_geometry(geometry, [[matrix.GetElement(r,c) for c in range(4)] for r in range(4)],
                             volume.GetImageData().GetDimensions())


def validate_markup_binding(volume, node):
    if (not volume.GetAttribute('Medic.SeriesUID') or not volume.GetAttribute('Medic.OriginalHashes')
            or node.GetAttribute('Medic.SeriesUID') != volume.GetAttribute('Medic.SeriesUID')
            or node.GetAttribute('Medic.OriginalHashes') != volume.GetAttribute('Medic.OriginalHashes')):
        raise ValueError('markup must be explicitly bound to the selected source volume')


def bind_markup(volume, node):
    for key in ('Medic.SeriesUID', 'Medic.OriginalHashes'):
        if node.GetAttribute(key) and node.GetAttribute(key) != volume.GetAttribute(key):
            raise ValueError('markup is already bound to a different source volume')
        node.SetAttribute(key, volume.GetAttribute(key))
    validate_markup_binding(volume, node)


def load_series(manifest, series_uid, workspace):
    import slicer
    from DICOMLib import DICOMUtils
    template = review_template(manifest, series_uid)
    workspace = _local_directory(workspace)
    with tempfile.TemporaryDirectory(dir=workspace) as temporary:
        directory = Path(temporary) / 'dicom'
        directory.mkdir()
        found = set()
        for item in manifest['instances']:
            if item['sha256'] in template['original_hashes'] and item['sha256'] not in found:
                source = Path(item['stored_path'])
                if any(_linked(p) for p in (source,*source.parents)):
                    raise ValueError('linked DICOM source')
                data = source.read_bytes()
                if digest(data) != item['sha256']:
                    raise ValueError('original hash mismatch')
                (directory / (item['sha256'].split(':')[1]+'.dcm')).write_bytes(data)
                found.add(item['sha256'])
        if found != set(template['original_hashes']):
            raise ValueError('missing original instances')
        with DICOMUtils.TemporaryDICOMDatabase(str(Path(temporary)/'database')) as db:
            DICOMUtils.importDicom(str(directory), db)
            ids = DICOMUtils.loadSeriesByUID([series_uid])
        volumes = [slicer.mrmlScene.GetNodeByID(i) for i in ids
                   if slicer.mrmlScene.GetNodeByID(i).IsA('vtkMRMLScalarVolumeNode')]
        if len(volumes) != 1:
            raise ValueError('expected exactly one scalar volume')
        volume = volumes[0]
        _validate_volume(volume, dict(affine_lps=template['affine_lps'],shape=template['shape']))
        volume.SetAttribute('Medic.SeriesUID',series_uid)
        volume.SetAttribute('Medic.OriginalHashes',json.dumps(template['original_hashes']))
        return volume


def export_review(manifest, volume, markup_nodes, output, segmentation=None, evaluation_roi=None):
    import slicer
    import vtk
    series_uid = volume.GetAttribute('Medic.SeriesUID')
    review = review_template(manifest, series_uid)
    if json.loads(volume.GetAttribute('Medic.OriginalHashes')) != review['original_hashes']:
        raise ValueError('reference provenance mismatch')
    _validate_volume(volume,dict(affine_lps=review['affine_lps'],shape=review['shape']))
    output = _local_directory(output)
    # Each export gets a fresh directory, preserving previous reviews.
    output = Path(tempfile.mkdtemp(prefix='review-',dir=output))
    review['markups'] = []
    for index, node in enumerate(markup_nodes):
        validate_markup_binding(volume, node)
        kind = {'vtkMRMLMarkupsFiducialNode':'point','vtkMRMLMarkupsLineNode':'line',
                'vtkMRMLMarkupsAngleNode':'angle'}.get(node.GetClassName())
        if kind is None:
            raise ValueError('unsupported markup node type')
        measurement_id = node.GetAttribute('Medic.MeasurementID')
        if not measurement_id:
            measurement_id = 'measurement-' + uuid.uuid4().hex
            node.SetAttribute('Medic.MeasurementID', measurement_id)
        metadata = json.loads(node.GetAttribute('Medic.Metadata') or '{}')
        annotation = dict(measurement_id=measurement_id, name=node.GetName(), metadata=metadata)
        points = []
        for i in range(node.GetNumberOfControlPoints()):
            if node.GetNthControlPointPositionStatus(i) != node.PositionDefined:
                raise ValueError('undefined markup point')
            point = [0.,0.,0.]
            node.GetNthControlPointPositionWorld(i, point)
            points.append(point)
        if kind == 'point':
            review['markups'].extend(dict(kind='point',coordinate_system='RAS',points=[p],
                **{**annotation, 'measurement_id': measurement_id + '-' + str(i)}) for i,p in enumerate(points))
        else:
            review['markups'].append(dict(kind=kind,coordinate_system='RAS',points=points,**annotation))
        if not slicer.util.saveNode(node,str(output/f'markup-{index}.mrk.json')):
            raise ValueError('markup save failed')
    for role, node in [('manual_lesion',segmentation),('evaluation_roi',evaluation_roi)]:
        if node is None:
            continue
        if node.GetParentTransformNode() is not None:
            raise ValueError('transformed segmentation unsupported')
        if not slicer.util.saveNode(node,str(output/f'{role}.seg.nrrd')):
            raise ValueError('segmentation save failed')
        # Export each segment separately so overlapping manual lesions are preserved.
        for index in range(node.GetSegmentation().GetNumberOfSegments()):
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(node.GetSegmentation().GetNthSegmentID(index))
            labelmap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
            try:
                success = slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(
                    node,ids,labelmap,volume)
                if not success:
                    raise ValueError('labelmap export failed')
                path = output/f'{role}-{index}.nrrd'
                if not slicer.util.saveNode(labelmap,str(path)):
                    raise ValueError('labelmap save failed')
                segment = node.GetSegmentation().GetSegment(node.GetSegmentation().GetNthSegmentID(index))
                review['segmentations'].append(dict(path=path.name,sha256=digest(path.read_bytes()),role=role,
                                                    structure=segment.GetName()))
            finally:
                slicer.mrmlScene.RemoveNode(labelmap)
    import_review(review,manifest,review_dir=output)
    (output/'review.json').write_text(json.dumps(review,allow_nan=False,indent=2),encoding='utf-8')
    return str(output/'review.json')
