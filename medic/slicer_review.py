"""Validate local Slicer manual annotations against an inspected source manifest."""
from copy import deepcopy
import io
from pathlib import Path

import numpy as np

from .inventory import digest, safe_name, _linked
from .imaging_geometry import measure_series, ras_to_lps, lesion_metrics


def validate_loaded_geometry(geometry, ijk_to_ras, dimensions):
    affine = np.asarray(ijk_to_ras, dtype=float)
    if (affine.shape != (4,4) or not np.isfinite(affine).all()
            or list(dimensions) != geometry['shape']
            or not np.allclose(np.diag([-1,-1,1,1]) @ affine, geometry['affine_lps'], atol=1e-4)):
        raise ValueError('loaded volume geometry differs from inspected source')


def _series(manifest, series_uid, study_uid=None):
    matches = [s for s in manifest['series'] if s['series_uid'] == series_uid
               and (study_uid is None or s['study_uid'] == study_uid)]
    if len(matches) != 1 or not matches[0]['measurement_eligible']:
        raise ValueError('missing, ambiguous, or inspect-only series')
    return matches[0]


def review_template(manifest, series_uid, *, study_uid=None):
    s = _series(manifest, series_uid, study_uid)
    return dict(version=1, study_uid=s['study_uid'], series_uid=s['series_uid'],
        original_hashes=list(s['original_hashes']), affine_lps=deepcopy(s['geometry']['affine_lps']),
        shape=list(s['geometry']['shape']), coordinate_system='LPS', units='mm', markups=[], segmentations=[])


def _mask(item, directory, geometry):
    import nrrd
    if directory is None or not safe_name(item['path']) or len(Path(item['path']).parts) != 1:
        raise ValueError('local review file required')
    path = Path(directory).absolute() / item['path']
    if '.medic' not in path.parts or str(path).startswith('\\\\') or any(_linked(p) for p in (path,*path.parents)):
        raise ValueError('unsafe review file')
    if path.stat().st_size > 128*1024*1024:
        raise ValueError('review file size limit')
    content = path.read_bytes()
    if digest(content) != item['sha256']:
        raise ValueError('review hash mismatch')
    stream = io.BytesIO(content)
    header = nrrd.read_header(stream)
    if 'data file' in header or 'datafile' in header:
        raise ValueError('detached NRRD not permitted')
    sizes = np.asarray(header['sizes'])
    if sizes.tolist() != geometry['shape'] or int(np.prod(sizes)) > 64*1024*1024:
        raise ValueError('segmentation shape mismatch or size limit')
    if header.get('encoding') not in ('raw', 'gzip', 'gz'):
        raise ValueError('unsupported NRRD encoding')
    directions = np.asarray(header['space directions'], dtype=float).T
    origin = np.asarray(header['space origin'], dtype=float)
    if header.get('space') == 'right-anterior-superior':
        directions = np.diag([-1,-1,1]) @ directions
        origin = origin * [-1,-1,1]
    elif header.get('space') != 'left-posterior-superior':
        raise ValueError('explicit LPS or RAS space required')
    affine = np.asarray(geometry['affine_lps'])
    if (directions.shape != (3,3) or origin.shape != (3,)
            or not np.allclose(directions, affine[:3,:3], atol=1e-5)
            or not np.allclose(origin, affine[:3,3], atol=1e-5)):
        raise ValueError('segmentation geometry mismatch')
    # Decode a bounded payload rather than letting a compressed NRRD expand without a cap.
    dtype_map = {'uint8':'u1','unsigned char':'u1','uchar':'u1','uint16':'<u2','unsigned short':'<u2',
                 'short':'<i2','int16':'<i2','unsigned int':'<u4','uint32':'<u4'}
    if header['type'] not in dtype_map:
        raise ValueError('integer labelmap required')
    dtype = np.dtype(dtype_map[header['type']])
    if dtype.itemsize > 1 and header.get('endian') == 'big':
        dtype = dtype.newbyteorder('>')
    expected = int(np.prod(sizes))*dtype.itemsize
    if expected > 128*1024*1024:
        raise ValueError('decoded labelmap limit')
    if header['encoding'] in ('gzip','gz'):
        import gzip
        with gzip.GzipFile(fileobj=stream) as compressed:
            raw = compressed.read(expected+1)
    else:
        raw = stream.read(expected+1)
    if len(raw) != expected:
        raise ValueError('labelmap payload length mismatch')
    data = np.frombuffer(raw, dtype=dtype).reshape(tuple(sizes), order='F')
    if (data < 0).any():
        raise ValueError('negative segment labels')
    return data


def import_review(review, manifest, *, review_dir=None):
    """Validate hashes/geometry/coordinates; return manual metrics, never diagnoses."""
    s = _series(manifest, review['series_uid'], review['study_uid'])
    template = review_template(manifest, s['series_uid'], study_uid=s['study_uid'])
    for key in ('version','original_hashes','affine_lps','shape','coordinate_system','units'):
        if review.get(key) != template[key]:
            raise ValueError('review provenance or geometry mismatch')
    found = set()
    for item in manifest['instances']:
        if item['sha256'] in s['original_hashes']:
            path = Path(item['stored_path'])
            if any(_linked(p) for p in (path, *path.parents)) or digest(path.read_bytes()) != item['sha256']:
                raise ValueError('stored original hash mismatch')
            found.add(item['sha256'])
    if found != set(s['original_hashes']):
        raise ValueError('missing original instances')
    measurements = []
    source_instances = [i for i in manifest['instances'] if i['sha256'] in s['original_hashes']]
    source_conditions = {'weight_bearing': s.get('weight_bearing', 'UNKNOWN'),
                         'modality': sorted({i.get('modality', 'UNKNOWN') for i in source_instances}),
                         'laterality': sorted({i.get('laterality', 'UNKNOWN') for i in source_instances})}
    measurement_ids = set()
    for markup in review['markups']:
        coord = markup.get('coordinate_system')
        if coord not in ('LPS','RAS'):
            raise ValueError('explicit markup coordinate system required')
        points = markup['points']
        if coord == 'RAS':
            points = [ras_to_lps(p) for p in points]
        result = measure_series(s, points, kind=markup['kind'],
            protocol=markup.get('protocol'), axes=markup.get('axes'), plane=markup.get('plane'))
        identifier = markup.get('measurement_id')
        if identifier is not None:
            if not isinstance(identifier, str) or not identifier.strip() or identifier in measurement_ids:
                raise ValueError('measurement ID must be nonempty and unique within review')
            measurement_ids.add(identifier)
        metadata = markup.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('measurement metadata must be an object')
        result.update(measurement_id=identifier, name=markup.get('name'), metadata=deepcopy(metadata),
                      source_conditions=deepcopy(source_conditions),
                      metadata_status='USER_DECLARED_NOT_CLINICALLY_VALIDATED' if metadata else 'UNREVIEWED')
        from .measurement_comparison import validate_source_conditions
        validate_source_conditions(result, source_conditions)
        measurements.append(result)
    masks = []
    roi = None
    for item in review['segmentations']:
        if item['role'] not in ('manual_lesion','evaluation_roi'):
            raise ValueError('explicit manual lesion or evaluation ROI role required')
        data = _mask(item, review_dir, s['geometry'])
        if item['role'] == 'evaluation_roi':
            if roi is not None:
                raise ValueError('ambiguous evaluation ROI')
            roi = data != 0
        else:
            masks.append((item,data))
    segments = []
    for item, data in masks:
        for label in np.unique(data):
            if label:
                segments.append(dict(sha256=item['sha256'], label=int(label),
                    structure=item.get('structure', 'UNASSIGNED_REQUIRES_REVIEW'),
                    metrics=lesion_metrics(data == label, s['geometry'], roi)))
    source_instances = [i for i in manifest['instances'] if i['sha256'] in s['original_hashes']]
    return dict(version=1, study_uid=s['study_uid'], series_uid=s['series_uid'],
                original_hashes=s['original_hashes'], measurements=measurements, segmentations=segments,
                conditions={'weight_bearing': s.get('weight_bearing', 'UNKNOWN'),
                            'modality': sorted({i.get('modality', 'UNKNOWN') for i in source_instances}),
                            'laterality': sorted({i.get('laterality', 'UNKNOWN') for i in source_instances}),
                            'dates': sorted({i['date'] for i in source_instances if i.get('date')}),
                            'geometry': s['geometry']},
                interpretation='MANUAL_LOCAL_ANNOTATIONS_NOT_DIAGNOSIS', unmarked_tissue='NOT_EVALUATED')
