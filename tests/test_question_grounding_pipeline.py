"""Recorded M3 integration: claim-local failures must not erase other answers."""
import json

import pytest

from medic.claim_grounding import ground_synthesis, known_patient_fact_ids, synthesis_claims
from medic.config import ModelRegistry
from medic.question_runtime import question_claim_trace, scope_questions
from medic.research_contracts import ProposedSynthesis, QuestionSet, SynthesisReview
from medic.tribunal import TribunalService
import tests.test_tribunal as fixtures


def question(identifier='current', **overrides):
    return {'question_id': identifier, 'type': 'CURRENT_STATE', 'text': 'Synthetic question',
            'target': 'ankle', 'assessment_time': 'reviewed time', 'allowed_fact_ids': ['observed'], **overrides}


def claim(identifier='answer', question_id='current', **overrides):
    return {'claim_id': identifier, 'question_id': question_id, 'text': 'Synthetic reviewed observation',
            'status': 'ANSWER', 'citations': {'fact_ids': ['observed']}, **overrides}


def patient(questions):
    return {'confirmed_facts': [], 'fact_details': [{'fact_id': 'observed', 'user_confirmed': True}],
            'scoped_questions': questions, 'uncertainties': []}


def synthesis(claims, **action_overrides):
    payload = fixtures.synthesis()
    payload['question_claims'] = claims
    payload['conditional_actions'][0].update({'question_ids': ['current'], 'claim_ids': ['answer'], **action_overrides})
    return ProposedSynthesis.model_validate_json(json.dumps(payload))


def review(parsed, provider='openai', overrides=None):
    payload = fixtures.synthesis_review(provider)
    payload['claim_reviews'] = [{'claim_id': item['claim_id'], 'verdict': 'SUPPORTED',
        'rationale': 'Synthetic support', 'evidence_ids': ['ev-1'],
        'source_quotes': [{'evidence_id': 'ev-1', 'quote': 'finding'}]}
        for item in synthesis_claims(parsed)]
    for audit in payload['claim_reviews']:
        audit.update((overrides or {}).get(audit['claim_id'], {}))
    return SynthesisReview.model_validate_json(json.dumps(payload))


def grounded(parsed, questions, overrides=None):
    return ground_synthesis(parsed, (review(parsed, overrides=overrides),),
                            (fixtures.TribunalTests().evidence(),), patient(questions))


@pytest.mark.parametrize('status', ['MISSING_DATA', 'OUT_OF_SCOPE', 'CONFLICT', 'SYSTEM_FAILURE'])
def test_non_answer_cannot_support_action_while_other_claims_survive(status):
    parsed = synthesis([claim(status=status)])
    result, withheld = grounded(parsed, [question()])
    assert result.conditional_actions[0].action.startswith('판단 보류')
    assert result.agreements == parsed.agreements
    assert any(c.location == 'conditional_actions.0.action' for c in withheld)


def test_every_action_question_requires_its_own_answering_support():
    parsed = synthesis([claim()], question_ids=['current', 'decision'])
    decision = question('decision', type='DECISION', options=['A', 'B'])
    result, _ = grounded(parsed, [question(), decision])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_single_best_action_cannot_borrow_options_claim_without_preferences():
    parsed = synthesis([claim(question_id='decision', decision_mode='OPTIONS')],
                       question_ids=['decision'], decision_mode='SINGLE_BEST')
    decision = question('decision', type='DECISION', options=['A', 'B'])
    result, _ = grounded(parsed, [decision])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_single_best_action_requires_an_explicit_decision_question():
    parsed = synthesis([claim()], decision_mode='SINGLE_BEST')
    result, _ = grounded(parsed, [question()])
    assert result.conditional_actions[0].action.startswith('판단 보류')


def test_model_attestation_alone_cannot_validate_an_individual_numeric_model():
    sources = ['ev-1']
    scenario = {'target': 'ankle', 'assessment_time': 'reviewed time', 'horizon': 'one year',
                'outcome': 'synthetic outcome', 'treatment_state': 'unchanged'}
    prognosis = question('future', type='PROGNOSIS', allowed_evidence_ids=sources, **scenario)
    numeric = claim(question_id='future', text='Individual risk 10%', citations={'evidence_ids': sources},
        numeric_estimate={'kind': 'INDIVIDUAL_RISK', 'value': 0.1},
        numeric_support={'validated_model_id': 'model-claimed-id', 'model_validated': True,
            'validation_source_id': 'ev-1', 'calibration_source_id': 'ev-1', 'method_source_id': 'ev-1',
            'scenario': scenario})
    parsed = synthesis([numeric], question_ids=['future'])
    result, withheld = grounded(parsed, [prognosis])
    assert result.question_claims[0].text.startswith('판단 보류')
    assert any('numeric model' in reason for c in withheld for reason in c.reasons)


def test_repeated_claim_ids_keep_each_question_and_distinct_occurrence():
    questions = [question(), question('future', type='PROGNOSIS')]
    parsed = synthesis([claim('duplicate'), claim('duplicate', 'future')])
    _, withheld = grounded(parsed, questions)
    trace = question_claim_trace(patient(questions), parsed, (fixtures.TribunalTests().evidence(),), withheld)
    assert [(c['claim_id'], c['question_id']) for c in trace['claims']] == [('duplicate', 'current'), ('duplicate', 'future')]
    assert all(not c['accepted'] for c in trace['claims'])
    assert len({c['occurrence_id'] for c in trace['claims']}) == 2
    assert any('horizon' in reason for reason in trace['claims'][1]['reasons'])


@pytest.mark.parametrize('failure', [
    {'verdict': 'CONTRADICTED', 'rationale': 'Synthetic contradiction'},
    {'source_quotes': [{'evidence_id': 'ev-1', 'quote': 'absent quotation'}]},
    {'requires_fulltext_verification': True},
])
def test_later_evidence_rejection_propagates_to_dependent_action(failure):
    parsed = synthesis([claim()])
    result, withheld = grounded(parsed, [question()], {'answer': failure})
    assert result.question_claims[0].text.startswith('판단 보류')
    assert result.conditional_actions[0].action.startswith('판단 보류')
    assert result.agreements == parsed.agreements
    trace = question_claim_trace(patient([question()]), parsed, (fixtures.TribunalTests().evidence(),), withheld)
    assert trace['claims'][0]['accepted'] is False
    expected_status = 'CONFLICT' if failure.get('verdict') == 'CONTRADICTED' else 'MISSING_DATA'
    assert trace['claims'][0]['status'] == expected_status


def test_missing_scenario_does_not_erase_a_reviewed_contradiction():
    prognosis = question('future', type='PROGNOSIS')
    parsed = synthesis([claim(question_id='future')], question_ids=['future'])
    result, withheld = grounded(parsed, [prognosis], {'answer': {
        'verdict': 'CONTRADICTED', 'rationale': 'Synthetic source contradicts the assertion'}})
    assert result.question_claims[0].status == 'CONFLICT'
    assertion = next(c for c in withheld if c.location == 'question_claims.0.text')
    assert any('horizon' in reason for reason in assertion.reasons)


def test_foreign_patient_and_unreviewed_fact_cannot_support_general_claims():
    context = {'fact_details': [{'fact_id': 'valid', 'user_confirmed': True},
                               {'fact_id': 'foreign', 'user_confirmed': True},
                               {'fact_id': 'draft', 'user_confirmed': False}],
               'case_context': {'ineligible_fact_ids': ['foreign']}}
    assert known_patient_fact_ids(context) == {'valid'}


def test_readiness_and_grounding_use_the_same_reviewed_fact_boundary(monkeypatch):
    from medic.question_runtime import available_references
    from medic.questions import _reviewed_references
    context = {'fact_details': [{'fact_id': 'valid', 'user_confirmed': True},
        {'fact_id': 'foreign', 'user_confirmed': True}, {'fact_id': 'not-boolean', 'user_confirmed': 'false'},
        {'fact_id': 'extracted', 'user_confirmed': True, 'confirmation_scope': 'UNREVIEWED_EXTRACTION'}],
        'case_context': {'ineligible_fact_ids': ['foreign']}}
    assert available_references(context).fact_ids == frozenset({'valid'})
    monkeypatch.setattr('medic.cli._latest_context', lambda *a: context)
    assert _reviewed_references(None, 'case-1').fact_ids == frozenset({'valid'})


@pytest.mark.parametrize('numeric', [False, True])
def test_generated_claim_without_any_question_scope_is_withheld_and_traced(numeric):
    extra = {'text': 'Individual risk 10%', 'numeric_estimate': {'kind': 'INDIVIDUAL_RISK', 'value': 0.1}} if numeric else {}
    parsed = synthesis([claim(**extra)])
    result, withheld = grounded(parsed, [])
    assert result.question_claims[0].status == 'OUT_OF_SCOPE'
    assert result.question_claims[0].text.startswith('판단 보류')
    assert result.conditional_actions[0].action.startswith('판단 보류')
    assert result.agreements == parsed.agreements
    trace = question_claim_trace(patient([]), parsed, (fixtures.TribunalTests().evidence(),), withheld)
    assert trace['claims'][0]['status'] == 'OUT_OF_SCOPE'
    assert trace['claims'][0]['accepted'] is False


def test_unlocalized_summary_citation_still_fails_closed():
    from medic.evidence import CitationGate
    parsed = synthesis([claim()])
    payload = review(parsed).model_dump(mode='json')
    payload['evidence_ids'].append('unlocalized')
    with pytest.raises(ValueError, match='unlocalized'):
        TribunalService._validate_citations(SynthesisReview.model_validate_json(json.dumps(payload)),
                                           CitationGate((fixtures.TribunalTests().evidence(),)))


def test_explicit_evidence_does_not_expand_relevance_scope():
    from medic.literature import DecisionIssue
    issue = DecisionIssue(issue_id='issue', question='Synthetic', question_ids=('current',),
        required_evidence=(), missing_information=(), decision_changers=())
    related = fixtures.TribunalTests().evidence().model_copy(update={'decision_issue_ids': ('issue',)})
    other = related.model_copy(update={'evidence_id': 'other', 'decision_issue_ids': ('other-issue',)})
    result = scope_questions({'typed_questions': [question(allowed_evidence_ids=['ev-1', 'other'])]},
                             (related, other), (issue,))
    assert result[0]['allowed_evidence_ids'] == ['ev-1']


@pytest.mark.parametrize('bad_id_in_summary', [False, True])
def test_unknown_claim_citation_completes_tribunal_and_resume_without_new_calls(tmp_path, bad_id_in_summary):
    raw, gateways = fixtures.gateways()
    parsed = synthesis([claim(citations={'evidence_ids': ['invented']})])
    raw['openai']._responses[3] = parsed.model_dump(mode='json')
    for provider in fixtures.PROVIDERS:
        raw[provider]._responses[-1] = review(parsed, provider, {
            'answer': {'evidence_ids': ['invented'], 'source_quotes': []}
        }).model_dump(mode='json')
        if bad_id_in_summary:
            raw[provider]._responses[-1]['evidence_ids'].append('invented')
    args = dict(run_id='claim-local', case_id='case-1', patient_context=patient([question()]),
                question_set=QuestionSet.for_comprehensive_review('case-1'),
                evidence_cards=(fixtures.TribunalTests().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
    service = TribunalService(tmp_path, gateways)
    result = service.run(**args)
    assert result.synthesis.question_claims[0].text.startswith('판단 보류')
    assert result.synthesis.agreements == parsed.agreements
    assert len(result.synthesis_reviews) == 3
    assert service.run(**args) == result
    assert sum(g.call_count for g in raw.values()) == 13
