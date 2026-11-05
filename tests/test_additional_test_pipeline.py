"""Synthetic information gathering must not imply an established diagnosis."""
import json

import pytest

from medic.claim_grounding import synthesis_claims
from medic.research_contracts import ProposedSynthesis
from tests.test_question_grounding_pipeline import claim, grounded, question, synthesis


def proposal(**updates):
    return {'proposal_id': 'test-1', 'question_id': 'current', 'test': 'Synthetic examination',
        'question': 'Which observation is missing?',
        'results_change': [{'result': 'Observation present', 'change': 'Reconsider hypothesis'},
                           {'result': 'Observation absent', 'change': 'Keep uncertainty'}],
        'burdens': ['Synthetic burden'], 'urgency': 'Discuss at planned review',
        'alternatives': ['Review existing observations'],
        'citations': {'evidence_ids': ['ev-1']}, **updates}


def proposed(items=None, **action_updates):
    parsed = synthesis([claim(status='MISSING_DATA')], **{'action_kind': 'ADDITIONAL_TEST',
        'proposal_ids': ['test-1'], 'claim_ids': [], **action_updates})
    data = parsed.model_dump(mode='json')
    data['additional_tests'] = items if items is not None else [proposal()]
    data['conditional_actions'][0]['action'] = 'Synthetic examination'
    data['conditional_action_ranking'] = ['Synthetic examination']
    return ProposedSynthesis.model_validate_json(json.dumps(data))


def test_missing_answer_can_have_separately_grounded_information_gathering():
    parsed = proposed()
    result, withheld = grounded(parsed, [question(allowed_evidence_ids=['ev-1'])])
    assert result.conditional_actions[0].action == 'Synthetic examination'
    assert result.additional_tests[0] == parsed.additional_tests[0]
    assert not any(c.location.startswith('additional_tests.') for c in withheld)
    paths = {c['location'] for c in synthesis_claims(parsed)}
    assert {'additional_tests.0.test', 'additional_tests.0.question',
            'additional_tests.0.results_change.0.result', 'additional_tests.0.results_change.1.change',
            'additional_tests.0.burdens.0', 'additional_tests.0.urgency',
            'additional_tests.0.alternatives.0'} <= paths


@pytest.mark.parametrize('updates', [
    {'question_id': 'foreign'}, {'citations': {'evidence_ids': ['invented']}},
    {'citations': {'fact_ids': ['foreign']}}, {'citations': {}},
])
def test_invalid_proposal_withholds_only_proposal_and_dependent_action(updates):
    parsed = proposed([proposal(**updates)])
    result, withheld = grounded(parsed, [question(allowed_evidence_ids=['ev-1'])])
    assert result.additional_tests[0].test.startswith('판단 보류')
    assert result.conditional_actions[0].action.startswith('판단 보류')
    assert result.agreements == parsed.agreements
    assert any(c.location.startswith('additional_tests.') for c in withheld)


def test_one_rejected_result_branch_withholds_entire_proposal_and_action():
    parsed = proposed()
    branch = next(c for c in synthesis_claims(parsed) if c['location'] == 'additional_tests.0.results_change.1.change')
    result, _ = grounded(parsed, [question(allowed_evidence_ids=['ev-1'])],
                         {branch['claim_id']: {'verdict': 'UNSUPPORTED'}})
    assert result.additional_tests[0].test.startswith('판단 보류')
    assert result.conditional_actions[0].action.startswith('판단 보류')
    assert result.agreements == parsed.agreements


@pytest.mark.parametrize('action_updates', [
    {'proposal_ids': ['foreign']}, {'question_ids': ['other']},
    {'claim_ids': ['answer']}, {'decision_mode': 'SINGLE_BEST'},
])
def test_information_gathering_requires_exact_proposal_scope(action_updates):
    parsed = proposed(**action_updates)
    result, _ = grounded(parsed, [question(allowed_evidence_ids=['ev-1']), question('other')])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_treatment_cannot_borrow_a_test_proposal_instead_of_answer():
    parsed = proposed(action_kind='TREATMENT_OR_MANAGEMENT')
    result, _ = grounded(parsed, [question(allowed_evidence_ids=['ev-1'])])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_duplicate_proposal_ids_cannot_authorize_action():
    parsed = proposed([proposal(), proposal()])
    result, _ = grounded(parsed, [question(allowed_evidence_ids=['ev-1'])])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_test_action_cannot_change_the_verified_test_into_treatment():
    parsed = proposed()
    payload = parsed.model_dump(mode='json')
    payload['conditional_actions'][0]['action'] = 'Start synthetic treatment'
    payload['conditional_action_ranking'] = ['Start synthetic treatment']
    result, _ = grounded(ProposedSynthesis.model_validate_json(json.dumps(payload)),
                         [question(allowed_evidence_ids=['ev-1'])])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_proposal_audit_cannot_borrow_a_source_outside_question_scope():
    parsed = proposed()
    target = next(c for c in synthesis_claims(parsed) if c['location'] == 'additional_tests.0.test')
    result, _ = grounded(parsed, [question(allowed_evidence_ids=['ev-1'], allowed_fact_ids=[])],
                         {target['claim_id']: {'patient_fact_ids': ['observed']}})
    assert result.additional_tests[0].test.startswith('판단 보류')


def test_test_action_benefit_cannot_borrow_out_of_scope_patient_fact():
    parsed = proposed()
    target = next(c for c in synthesis_claims(parsed) if c['location'] == 'conditional_actions.0.benefits.0')
    result, _ = grounded(parsed, [question(allowed_evidence_ids=['ev-1'], allowed_fact_ids=[])],
                         {target['claim_id']: {'patient_fact_ids': ['observed']}})
    assert result.conditional_actions[0].action.startswith('판단 보류')


@pytest.mark.parametrize('remaining_uncertainty', [False, True])
def test_answered_question_needs_explicit_remaining_uncertainty_for_test(remaining_uncertainty):
    parsed = proposed()
    payload = parsed.model_dump(mode='json')
    payload['question_claims'] = [claim(**({'uncertainty': {'measurement': 'Repeatability unknown'}}
                                        if remaining_uncertainty else {}))]
    result, _ = grounded(ProposedSynthesis.model_validate_json(json.dumps(payload)),
                         [question(allowed_evidence_ids=['ev-1'])])
    assert result.additional_tests[0].test.startswith('판단 보류') is not remaining_uncertainty
    assert result.conditional_actions[0].action.startswith('판단 보류') is not remaining_uncertainty


@pytest.mark.parametrize('bad_reference', [False, True])
def test_recorded_tribunal_report_and_resume_include_test_proposal(tmp_path, bad_reference):
    from medic.config import ModelRegistry
    from medic.reporting import ReportRenderer
    from medic.research_contracts import QuestionSet
    from medic.tribunal import TribunalService
    from tests.test_question_grounding_pipeline import patient, review
    import tests.test_tribunal as fixtures

    raw, gateways = fixtures.gateways()
    parsed = proposed([proposal(citations={'evidence_ids': ['invented']})] if bad_reference else None)
    raw['openai']._responses[3] = parsed.model_dump(mode='json')
    for provider in fixtures.PROVIDERS:
        raw[provider]._responses[-1] = review(parsed, provider).model_dump(mode='json')
    context = patient([question(allowed_evidence_ids=['ev-1'])])
    context['fact_details'][0].update(fact_key='observation', value='Synthetic observation',
                                    source_locators=['synthetic:1'])
    context['confirmed_facts'] = ['observation: Synthetic observation']
    service = TribunalService(tmp_path, gateways)
    args = dict(run_id='test-proposals', case_id='case-1', patient_context=context,
                question_set=QuestionSet.for_comprehensive_review('case-1'),
                evidence_cards=(fixtures.TribunalTests().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
    result = service.run(**args)
    assert service.run(**args) == result
    assert sum(g.call_count for g in raw.values()) == 13
    report = ReportRenderer.build(result)
    markdown = ReportRenderer.markdown(report)
    assert '추가검사 제안' in markdown
    assert 'test\\-1' in markdown
    assert 'current' in markdown
    if bad_reference:
        assert result.synthesis.additional_tests[0].test.startswith('판단 보류')
    else:
        assert 'Observation present' in markdown
        assert 'Synthetic burden' in markdown
        assert 'Discuss at planned review' in markdown
        assert 'Review existing observations' in markdown

