"""Explicit pairing of reviewed scalar measurements, never scanner coordinates."""
import math


IDENTITY_FIELDS = ('person_id', 'episode_id', 'epoch_id', 'anatomy', 'laterality', 'metric',
                   'method', 'protocol_version', 'landmarks', 'plane_definition')
ACQUISITION_FIELDS = ('modality', 'weight_bearing', 'position', 'protocol')


def validate_source_conditions(measurement, conditions):
    metadata = measurement.get('metadata', {})
    normalize_side = lambda v: {'L': 'LEFT', 'R': 'RIGHT', 'B': 'BILATERAL'}.get(str(v).upper(), str(v).upper())
    actual = conditions.get('laterality', ())
    actual = actual if isinstance(actual, (list, tuple)) else [actual]
    sides = {normalize_side(v) for v in actual if not _unknown(v)}
    declared = normalize_side(metadata.get('laterality', 'UNKNOWN'))
    if sides and declared != 'UNKNOWN' and sides != {declared}:
        raise ValueError('measurement laterality conflicts with inspected source')
    acquisition = metadata.get('acquisition', {})
    modalities = conditions.get('modality', ())
    modalities = modalities if isinstance(modalities, (list, tuple)) else [modalities]
    known = {str(v).upper() for v in modalities if not _unknown(v)}
    if known and not _unknown(acquisition.get('modality')) and known != {str(acquisition['modality']).upper()}:
        raise ValueError('measurement modality conflicts with inspected source')
    if not _unknown(conditions.get('weight_bearing')) and not _unknown(acquisition.get('weight_bearing')):
        if conditions['weight_bearing'] != acquisition['weight_bearing']:
            raise ValueError('measurement weight_bearing conflicts with inspected source')


def _unknown(value):
    if isinstance(value, str):
        return value.strip().upper() in ('', 'UNKNOWN', 'UNREVIEWED')
    if isinstance(value, (list, tuple)):
        return not value or any(_unknown(v) for v in value)
    if isinstance(value, dict):
        return not value or any(_unknown(v) for v in value.values())
    return value is None


def compare_measurements(before, after, *, repeatability_bound=None, repeatability_source=None):
    for measurement in (before, after):
        if 'source_conditions' in measurement:
            validate_source_conditions(measurement, measurement['source_conditions'])
    if any(item.get('kind') not in ('line', 'angle') for item in (before, after)):
        raise ValueError('only two-point distance and three-point angle comparisons are supported')
    if any(not isinstance(item.get('value'), (int, float)) or isinstance(item['value'], bool)
           or not math.isfinite(item['value']) for item in (before, after)):
        raise ValueError('finite scalar measurements required')
    mismatch, missing = [], []
    before_plane, after_plane = before.get('plane'), after.get('plane')
    if (before_plane is None) != (after_plane is None):
        mismatch.append('calculation_plane: 3D versus projected')
    elif before_plane is not None:
        # A shared scanner orientation alone does not establish an anatomical plane across studies.
        # Require the same reviewed plane protocol and matching normal direction conservatively.
        a, b = before_plane.get('normal'), after_plane.get('normal')
        if a != b:
            mismatch.append('calculation_plane.normal')
    for field in ('kind', 'units'):
        if _unknown(before.get(field)) or _unknown(after.get(field)):
            missing.append(field)
        elif before[field] != after[field]:
            mismatch.append(field)
    metadata = [item.get('metadata', {}) for item in (before, after)]
    for field in IDENTITY_FIELDS:
        a, b = (m.get(field) for m in metadata)
        if _unknown(a) or _unknown(b):
            missing.append(field)
        elif a != b:
            mismatch.append(field)
    for field in ACQUISITION_FIELDS:
        a, b = (m.get('acquisition', {}).get(field) for m in metadata)
        if _unknown(a) or _unknown(b):
            missing.append('acquisition.' + field)
        elif a != b:
            mismatch.append('acquisition.' + field)
    comparable = not mismatch and not missing
    delta = float(after['value'] - before['value']) if comparable else None
    repeatability = 'UNKNOWN'
    if repeatability_bound is not None:
        if (isinstance(repeatability_bound, bool) or not math.isfinite(repeatability_bound)
                or repeatability_bound < 0 or not repeatability_source):
            raise ValueError('nonnegative repeatability bound and explicit source required')
        if comparable:
            repeatability = ('WITHIN_REVIEWED_BOUND_NOT_PROOF_OF_STABILITY' if abs(delta) <= repeatability_bound
                             else 'EXCEEDS_REVIEWED_BOUND_NOT_CLINICAL_SIGNIFICANCE')
    return {'schema_version': 'medic-measurement-comparison/1',
            'before_id': before.get('measurement_id'), 'after_id': after.get('measurement_id'),
            'before_value': before['value'], 'after_value': after['value'], 'units': before.get('units'),
            'numeric_difference': delta, 'comparability': 'INCOMPATIBLE' if mismatch else 'LIMITED_UNKNOWN' if missing else 'COMPARABLE',
            'reasons': [f'mismatch: {key}' for key in mismatch] + [f'unknown: {key}' for key in missing],
            'repeatability_bound': repeatability_bound, 'repeatability_source': repeatability_source,
            'repeatability_assessment': repeatability, 'clinical_change': 'NOT_ESTABLISHED',
            'clinical_validation': 'NOT_VALIDATED', 'coordinate_subtraction_performed': False}


def active_measurement(ledger, case_id, review_id, measurement_id):
    latest, withdrawn = {}, set()
    for event in ledger.records():
        payload = event['payload']
        if payload.get('case_id') != case_id:
            continue
        if event['kind'] == 'IMAGING_REVIEW_IMPORTED':
            review = payload['review']
            latest[(review['study_uid'], review['series_uid'])] = payload
        elif event['kind'] == 'IMAGING_REVIEW_WITHDRAWN':
            withdrawn.add(payload['review_id'])
    matches = [p for p in latest.values() if p['review_id'] == review_id and review_id not in withdrawn]
    if len(matches) != 1:
        raise ValueError('measurement review is unknown, replaced or withdrawn for this case')
    found = [m for m in matches[0]['review']['measurements'] if m.get('measurement_id') == measurement_id]
    if len(found) != 1:
        raise ValueError('measurement ID missing or ambiguous in active review')
    result = {**found[0], 'source_conditions': matches[0]['review'].get('conditions', {})}
    validate_source_conditions(result, result['source_conditions'])
    return result
