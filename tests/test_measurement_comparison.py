import copy

import pytest


def metric(value=5.0):
    return {'measurement_id': 'm1', 'kind': 'line', 'value': value, 'units': 'mm',
            'metadata': {'person_id': 'person', 'episode_id': 'episode', 'epoch_id': 'epoch',
                         'anatomy': 'ankle', 'laterality': 'RIGHT', 'metric': 'reviewed distance',
                         'method': 'two_landmarks', 'protocol_version': '1',
                         'landmarks': ['A', 'B'], 'plane_definition': '3D',
                         'acquisition': {'modality': 'CT', 'weight_bearing': 'NON_WEIGHT_BEARING',
                                         'position': 'supine', 'protocol': 'ankle'}, 'reviewer': 'operator'}}


def test_manual_pairs_compare_values_and_never_scanner_coordinates():
    from medic.measurement_comparison import compare_measurements
    before, after = metric(), metric(7.0)
    before['points_lps'] = [[100, 0, 0], [105, 0, 0]]
    after['points_lps'] = [[-90, 0, 0], [-83, 0, 0]]
    result = compare_measurements(before, after)
    assert result['numeric_difference'] == 2.0
    assert result['comparability'] == 'COMPARABLE'
    assert result['clinical_change'] == 'NOT_ESTABLISHED'
    result = compare_measurements(before, after, repeatability_bound=3.0, repeatability_source='reviewed repeatability study')
    assert result['clinical_change'] == 'NOT_ESTABLISHED'
    assert result['repeatability_assessment'] == 'WITHIN_REVIEWED_BOUND_NOT_PROOF_OF_STABILITY'


def test_projection_provenance_cannot_be_overridden_by_matching_metadata():
    from medic.measurement_comparison import compare_measurements
    before, after = metric(5.0), metric(3.0)
    before['plane'] = None
    after['plane'] = {'origin': [0,0,0], 'normal': [0,1,0]}
    result = compare_measurements(before, after)
    assert result['comparability'] == 'INCOMPATIBLE'
    assert result['numeric_difference'] is None


@pytest.mark.parametrize('field,value', [('laterality', 'LEFT'), ('epoch_id', 'new-epoch'),
                                         ('method', 'different'), ('protocol_version', '2')])
def test_incompatible_measurements_cannot_claim_change(field, value):
    from medic.measurement_comparison import compare_measurements
    before, after = metric(), metric(6.0)
    after['metadata'][field] = value
    result = compare_measurements(before, after)
    assert result['comparability'] == 'INCOMPATIBLE'
    assert result['numeric_difference'] is None
    assert field in ' '.join(result['reasons'])


def test_unknown_conditions_are_not_equivalent():
    from medic.measurement_comparison import compare_measurements
    before, after = metric(), metric(6.0)
    for item in (before, after):
        item['metadata']['acquisition']['weight_bearing'] = 'UNKNOWN'
    result = compare_measurements(before, after)
    assert result['comparability'] == 'LIMITED_UNKNOWN'
    assert result['numeric_difference'] is None
    before, after = metric(), metric(6.0)
    for item in (before, after):
        item['metadata']['landmarks'] = ['UNKNOWN', 'UNKNOWN']
    assert compare_measurements(before, after)['comparability'] == 'LIMITED_UNKNOWN'


def test_withdrawn_or_replaced_review_cannot_be_compared(tmp_path):
    from medic.measurement_comparison import active_measurement
    from medic.imaging_history import record_import
    from medic.ledger import AppendOnlyJsonlLedger
    ledger = AppendOnlyJsonlLedger(tmp_path / 'ledger.jsonl')
    review = {'study_uid': 'study', 'series_uid': 'series', 'measurements': [metric()]}
    record_import(ledger, 'case', 'old', review, 'sha256:old')
    record_import(ledger, 'case', 'new', review, 'sha256:new', replaces='old', reason='corrected landmarks')
    with pytest.raises(ValueError, match='replaced'):
        active_measurement(ledger, 'case', 'old', 'm1')
    assert active_measurement(ledger, 'case', 'new', 'm1')['value'] == 5.0
    ledger.append_once('IMAGING_REVIEW_WITHDRAWN', 'withdraw', {'case_id': 'case', 'review_id': 'new'})
    with pytest.raises(ValueError, match='withdrawn'):
        active_measurement(ledger, 'case', 'new', 'm1')


def test_slicer_markup_metadata_survives_local_import(tmp_path):
    from medic.inventory import scan
    from medic.imaging import inspect_inventory
    from medic.slicer_review import import_review, review_template
    from tests.test_imaging import dicom
    source = tmp_path / 'input'
    source.mkdir()
    a = dicom(source / 'a')
    dicom(source / 'b', study=a.StudyInstanceUID, series=a.SeriesInstanceUID, z=4)
    manifest = inspect_inventory(scan(source), tmp_path / '.medic' / 'dicom')
    review = review_template(manifest, str(a.SeriesInstanceUID))
    metadata = metric()['metadata']
    review['markups'] = [{'kind': 'line', 'coordinate_system': 'LPS', 'points': [[0, 0, 0], [3, 4, 0]],
                          'measurement_id': 'manual-distance-1', 'name': 'two reviewed landmarks', 'metadata': metadata}]
    result = import_review(review, manifest)['measurements'][0]
    assert result['measurement_id'] == 'manual-distance-1'
    assert result['name'] == 'two reviewed landmarks'
    assert result['metadata'] == metadata
    manifest['instances'][0]['laterality'] = 'L'
    with pytest.raises(ValueError, match='laterality'):
        import_review(review, manifest)
