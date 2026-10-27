"""Non-diagnostic geometry. Voxel coordinates are (column, row, slice); world is LPS mm."""
import numpy as np


def _array(value, shape=None):
    a = np.asarray(value, dtype=float)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('invalid finite coordinates')
    return a


def validate_geometry(frames):
    if len(frames) < 2:
        raise ValueError('at least two slices required')
    first = frames[0]
    orientation = _array(first['orientation'], (6,))
    x, y = orientation[:3], orientation[3:]
    if not np.allclose([np.linalg.norm(x), np.linalg.norm(y), x @ y], [1,1,0], atol=1e-5):
        raise ValueError('non-orthonormal orientation')
    spacing = _array(first['spacing'], (2,))
    if (spacing <= 0).any() or min(first['rows'], first['columns']) <= 0:
        raise ValueError('invalid spacing or dimensions')
    normal = np.cross(x, y)
    positions = []
    for f in frames:
        if (not np.allclose(_array(f['orientation'], (6,)), orientation, atol=1e-5)
                or not np.allclose(_array(f['spacing'], (2,)), spacing, atol=1e-5)
                or (f['rows'], f['columns']) != (first['rows'], first['columns'])):
            raise ValueError('inconsistent geometry')
        positions.append(_array(f['position'], (3,)))
    order = np.argsort(np.array(positions) @ normal)
    positions = np.array(positions)[order]
    steps = np.diff(positions, axis=0)
    dz = float(steps[0] @ normal)
    if dz <= 1e-6 or not np.allclose(steps, normal * dz, rtol=1e-4, atol=1e-3):
        raise ValueError('duplicate, tilted, or irregular slices')
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack((x * spacing[1], y * spacing[0], normal * dz))
    affine[:3, 3] = positions[0]
    return dict(affine_lps=affine.tolist(), shape=[first['columns'],first['rows'],len(frames)],
                slice_order=order.tolist(), coordinate_system='LPS', units='mm')


def voxel_to_lps(geometry, point):
    a = _array(geometry['affine_lps'], (4,4))
    return (a @ np.append(_array(point, (3,)), 1))[:3].tolist()


def ras_to_lps(point):
    return (_array(point, (3,)) * [-1,-1,1]).tolist()


def validate_plane(frame):
    normal = np.cross(_array(frame['orientation'], (6,))[:3], _array(frame['orientation'], (6,))[3:])
    second = dict(frame, position=(_array(frame['position'], (3,)) + normal).tolist())
    geometry = validate_geometry([frame, second])
    geometry.update(shape=[frame['columns'],frame['rows'],1], slice_order=[0], image_plane_only=True)
    return geometry


def measure_series(series, points, **options):
    """Gated entry point for physical measurements; points must lie inside the source."""
    if not series.get('measurement_eligible') or not series.get('geometry'):
        raise ValueError('series is inspect only')
    geometry = series['geometry']
    points = _array(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('invalid points')
    inverse = np.linalg.inv(_array(geometry['affine_lps'], (4,4)))
    voxels = (inverse @ np.column_stack((points, np.ones(len(points)))).T).T[:,:3]
    if (voxels < -1e-4).any() or (voxels > np.asarray(geometry['shape'])-1+1e-4).any():
        raise ValueError('coordinates outside source geometry')
    return measure(points, source=series['series_uid'], **options)


def measure(points, *, source, kind='line', plane=None, protocol=None, axes=None):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('explicit source required')
    p = _array(points)
    counts = {'point':1, 'line':2, 'angle':3}
    if kind not in counts or p.shape != (counts[kind],3):
        raise ValueError('invalid points or measurement type')
    if protocol and (not axes or not axes.get('source')):
        raise ValueError('named protocol requires explicit sourced axes')
    if axes:
        x = _array(axes['x'], (3,))
        y = _array(axes['y'], (3,))
        if not np.allclose([np.linalg.norm(x), np.linalg.norm(y), x@y], [1,1,0], atol=1e-6):
            raise ValueError('invalid protocol axes')
    if plane is not None:
        origin = _array(plane['origin'], (3,))
        normal = _array(plane['normal'], (3,))
        if not np.isclose(np.linalg.norm(normal), 1):
            raise ValueError('plane normal must be unit length')
        p = p - np.outer((p-origin) @ normal, normal)
    if kind == 'point':
        value, units = p[0].tolist(), 'mm'
    elif kind == 'line':
        value, units = float(np.linalg.norm(p[1]-p[0])), 'mm'
    else:
        a, b = p[0]-p[1], p[2]-p[1]
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        if denom <= 1e-12:
            raise ValueError('degenerate angle')
        value, units = float(np.degrees(np.arccos(np.clip(a@b/denom,-1,1)))), 'degrees'
    return dict(kind=kind, value=value, units=units, source=source, coordinate_system='LPS',
                points_lps=_array(points).tolist(), plane=plane, protocol=protocol, axes=axes)


def lesion_metrics(mask, geometry, evaluation_roi=None):
    mask = np.asarray(mask)
    if mask.dtype != bool or list(mask.shape) != geometry['shape']:
        raise ValueError('mask must be boolean and match column-row-slice geometry')
    a = _array(geometry['affine_lps'], (4,4))[:3,:3]
    voxel_volume = abs(float(np.linalg.det(a)))
    if voxel_volume <= 0:
        raise ValueError('singular geometry')
    fraction = denominator = None
    if evaluation_roi is not None:
        roi = np.asarray(evaluation_roi)
        if roi.dtype != bool or roi.shape != mask.shape or not roi.any() or (mask & ~roi).any():
            raise ValueError('evaluation ROI must contain lesion and be nonempty')
        denominator = int(roi.sum()) * voxel_volume
        fraction = float(mask.sum()/roi.sum())
    return dict(volume_mm3=int(mask.sum())*voxel_volume,
                slice_area_mm2=(mask.sum(axis=(0,1))*np.linalg.norm(np.cross(a[:,0],a[:,1]))).tolist(),
                evaluation_roi_volume_mm3=denominator, fraction_of_evaluation_roi=fraction,
                interpretation='MANUAL_LOCAL_ANNOTATION_ONLY', unmarked_tissue='NOT_EVALUATED')
