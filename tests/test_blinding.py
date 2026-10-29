import json

import pytest

from medic.ledger import AppendOnlyJsonlLedger


def context():
    def fact(identifier, value):
        return {'fact_id': identifier, 'fact_key': identifier, 'value': value,
                'user_confirmed': True, 'source_locators': ['synthetic|text|0-10']}
    facts = [fact('direct', 'walking limited'), fact('doctor', 'SEALED CLINICIAN PROGNOSIS')]
    return {'fact_details': facts, 'confirmed_facts': [f"{f['fact_key']}: {f['value']}" for f in facts],
            'uncertainties': ['doctor says SEALED CLINICIAN PROGNOSIS'],
            'raw_images': ['unreviewed-page.png'],
            'case_context': {'case_id': 'case-1', 'profile': None, 'review_inputs': [
                {'review': {'fact_id': 'direct', 'classification': 'self_report', 'context': {'case_id': 'case-1'}}},
                {'review': {'fact_id': 'doctor', 'classification': 'clinician_assertion', 'context': {'case_id': 'case-1'}}}]}}


def approval():
    return {'schema_version': 'medic-blind-approval/1', 'case_id': 'case-1', 'reviewer': 'operator',
            'reason': 'synthetic separation review', 'questions_reviewed': True,
            'profile_items': [], 'imaging_review_ids': []}


def test_blind_input_rebuilds_views_and_never_carries_hidden_summary_or_image(tmp_path):
    from medic.blinding import create_blind_input, assert_no_leak
    visible = create_blind_input(tmp_path, 'blind', 'case-1', context(), {'question': 'walking'}, approval())
    text = json.dumps(visible)
    assert 'walking limited' in text
    assert 'SEALED CLINICIAN PROGNOSIS' not in text
    assert 'unreviewed-page' not in text
    assert 'raw_images' not in visible
    assert_no_leak(tmp_path, 'blind', 'query walking limitation')
    with pytest.raises(ValueError, match='withheld'):
        assert_no_leak(tmp_path, 'blind', 'query SEALED CLINICIAN PROGNOSIS')
    with pytest.raises((ValueError, RuntimeError)):
        create_blind_input(tmp_path, 'blind', 'case-1', context(), {'question': 'changed'}, approval())


def test_korean_and_escaped_hidden_input_is_rejected_before_snapshot_or_calls(tmp_path):
    from medic.blinding import create_blind_input
    ctx = context()
    ctx['fact_details'][1]['value'] = '의료진의 가려진 장기 예후 판단입니다'
    with pytest.raises(ValueError, match='withheld'):
        create_blind_input(tmp_path, 'korean', 'case-1', ctx,
            {'question': '의료진의 가려진 장기 예후 판단입니다'}, approval())
    ledger = AppendOnlyJsonlLedger(tmp_path / 'runs' / 'korean' / 'ledger.jsonl')
    assert not ledger.records()


def test_recursive_derived_lineage_and_imaging_projection(tmp_path):
    from medic.blinding import create_blind_input
    ctx = context()
    for identifier, parent in [('d1', 'doctor'), ('d2', 'd1')]:
        ctx['fact_details'].append({'fact_id': identifier, 'fact_key': identifier, 'value': identifier,
                                   'user_confirmed': True, 'source_locators': ['synthetic']})
        ctx['case_context']['review_inputs'].append({'review': {'fact_id': identifier,
            'classification': 'derived_measure', 'derived_from_ids': [parent], 'context': {'case_id': 'case-1'}}})
    ctx['imaging_reviews'] = [{'review_id': 'review', 'review_status': 'USER_REVIEWED', 'measurements': [
        {'measurement_id': 'm', 'kind': 'line', 'value': 5, 'units': 'mm',
         'metadata': {'laterality': 'RIGHT', 'clinician_interpretation': 'hidden arbitrary clinical recommendation'},
         'axes': {'x': [1,0,0], 'y': [0,1,0], 'source': 'reviewed protocol', 'clinician_interpretation': 'AXES COMMENTARY LEAK'},
         'plane': {'origin': [0,0,0], 'normal': [0,0,1], 'comment': 'PLANE COMMENTARY LEAK'}}], 'segmentations': []}]
    approved = approval()
    approved['imaging_review_ids'] = ['review']
    visible = create_blind_input(tmp_path, 'recursive', 'case-1', ctx, {}, approved)
    assert [f['fact_id'] for f in visible['fact_details']] == ['direct']
    assert 'clinician_interpretation' not in json.dumps(visible)
    assert 'hidden arbitrary clinical recommendation' not in json.dumps(visible)
    assert 'AXES COMMENTARY LEAK' not in json.dumps(visible)
    assert 'PLANE COMMENTARY LEAK' not in json.dumps(visible)


def test_comparison_requires_seal_and_same_case_input_and_detects_tampering(tmp_path):
    from medic.blinding import create_blind_input, seal_result, comparison_input
    from medic.config import ModelRegistry
    from medic.research_contracts import QuestionSet
    from medic.tribunal import TribunalService
    import tests.test_tribunal as fixtures
    ctx = context()
    # Existing recorded responses cite literature and require no private content.
    visible = create_blind_input(tmp_path, 'blind', 'case-1', ctx,
        QuestionSet.for_comprehensive_review('case-1').model_dump(mode='json'), approval())
    with pytest.raises(ValueError, match='sealed'):
        comparison_input(tmp_path, 'blind')
    _, gateways = fixtures.gateways()
    result = TribunalService(tmp_path, gateways).run(run_id='blind', case_id='case-1', patient_context=visible,
        question_set=QuestionSet.for_comprehensive_review('case-1'),
        evidence_cards=(fixtures.TribunalTests().evidence(),),
        estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
    seal = seal_result(tmp_path, result)
    assert seal['case_id'] == 'case-1'
    inputs = comparison_input(tmp_path, 'blind')
    assert inputs['clinician_facts'][0]['fact_id'] == 'doctor'
    ledger = AppendOnlyJsonlLedger(tmp_path / 'runs' / 'blind' / 'ledger.jsonl')
    assert any(r['kind'] == 'BLIND_RESULT_SEALED' for r in ledger.records())
    assert seal_result(tmp_path, result) == seal
    import shutil
    shutil.copytree(tmp_path / 'runs' / 'blind', tmp_path / 'runs' / 'copied')
    with pytest.raises(ValueError, match='run'):
        comparison_input(tmp_path, 'copied')
