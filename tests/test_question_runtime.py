import json

from medic.claim_grounding import ground_synthesis, synthesis_claims
from medic.research_contracts import ProposedSynthesis, SynthesisReview
import tests.test_tribunal as fixtures


def test_current_state_without_any_allowed_observation_is_missing_data():
    from medic.questions import AvailableReferences, evaluate_question
    result = evaluate_question({'question_id': 'q', 'type': 'CURRENT_STATE', 'text': 'Current state?',
                                'target': 'ankle', 'assessment_time': 'now'}, available=AvailableReferences())
    assert result.status == 'MISSING_DATA'
    assert 'observations_or_measurements' in result.missing_fields


def test_question_contracts_withhold_only_invalid_claim_and_dependent_action():
    current = {'question_id': 'current', 'type': 'CURRENT_STATE', 'text': 'Current state?',
               'target': 'ankle', 'assessment_time': 'now', 'allowed_fact_ids': ['observed']}
    prognosis = {'question_id': 'future', 'type': 'PROGNOSIS', 'text': 'Future risk?',
                 'target': 'ankle', 'assessment_time': 'now', 'allowed_fact_ids': ['observed']}
    context = {'scoped_questions': [current, prognosis], 'fact_details': [
        {'fact_id': 'observed', 'user_confirmed': True}]}
    synthesis = fixtures.synthesis()
    synthesis['question_claims'] = [
        {'claim_id': 'good', 'question_id': 'current', 'text': 'Reviewed current observation',
         'status': 'ANSWER', 'citations': {'fact_ids': ['observed']}},
        {'claim_id': 'bad', 'question_id': 'future', 'text': 'Future outcome asserted without horizon',
         'status': 'ANSWER', 'citations': {'fact_ids': ['observed']}}]
    synthesis['conditional_actions'][0].update(question_ids=['future'], claim_ids=['bad'])
    parsed = ProposedSynthesis.model_validate_json(json.dumps(synthesis))
    review = fixtures.synthesis_review('openai')
    existing = {r['claim_id']: r for r in review['claim_reviews']}
    for claim in synthesis_claims(parsed):
        if claim['claim_id'] not in existing:
            review['claim_reviews'].append({'claim_id': claim['claim_id'], 'verdict': 'SUPPORTED',
                'rationale': 'synthetic structural test', 'evidence_ids': [], 'source_quotes': [],
                'patient_fact_ids': ['observed']})
    grounded, withheld = ground_synthesis(parsed, (SynthesisReview.model_validate_json(json.dumps(review)),),
        (fixtures.TribunalTests().evidence(),), context)
    assert grounded.question_claims[0].text == 'Reviewed current observation'
    assert grounded.question_claims[1].text.startswith('판단 보류')
    assert grounded.conditional_actions[0].action.startswith('판단 보류')
    assert grounded.agreements == parsed.agreements
    assert any('horizon' in ' '.join(c.reasons) for c in withheld)
