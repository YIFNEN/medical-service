"""Necessary passages belong to each cited paper, not to a pooled quote set."""
import json

import pytest

from medic.claim_grounding import ground_synthesis, synthesis_claims
from medic.research_contracts import EvidenceCard, ProposedSynthesis, SynthesisReview
import tests.test_tribunal as fixtures


def card(identifier, **updates):
    text = 'Adults; cohort; finding; comparator.'
    fields = [{'field': field, 'value': quote, 'quote': quote, 'source_start': text.index(quote),
               'source_end': text.index(quote) + len(quote), 'source_sha256': 'sha256:synthetic-raw'}
              for field, quote in [('population', 'Adults'), ('study_design', 'cohort'),
                                   ('findings', 'finding'), ('comparator', 'comparator')]]
    data = fixtures.TribunalTests().evidence().model_dump(mode='json')
    data.update(evidence_id=identifier, source_text=text, access_level='FULL_TEXT',
        analysis_scope='PARTIAL_FULL_TEXT', source_sha256='sha256:synthetic-raw',
        field_support=fields, analyzed_locators=[f'text:characters:0-{len(text)}'])
    data.update(updates)
    return EvidenceCard.model_validate_json(json.dumps(data))


def run(cards, **audit_updates):
    parsed = ProposedSynthesis.model_validate_json(json.dumps(fixtures.synthesis()))
    data = fixtures.synthesis_review('openai')
    for audit in data['claim_reviews']:
        audit['evidence_ids'] = ['ev-1']
        audit['source_quotes'] = [{'evidence_id': 'ev-1', 'quote': 'finding'}]
    target = data['claim_reviews'][0]
    target.update(evidence_ids=[c.evidence_id for c in cards], requires_fulltext_verification=True,
                  source_quotes=[{'evidence_id': c.evidence_id, 'quote': field.quote}
                                 for c in cards for field in c.field_support])
    target.update(audit_updates)
    review = SynthesisReview.model_validate_json(json.dumps(data))
    return parsed, ground_synthesis(parsed, (review,), tuple(cards), {})


def test_one_complete_paper_cannot_validate_an_unread_comparison_paper():
    complete = card('ev-1', analysis_scope='FULL_TEXT')
    unread = card('other', access_level='ABSTRACT', analysis_scope='ABSTRACT')
    parsed, (result, withheld) = run([complete, unread])
    assert len(withheld) == 1
    assert result.agreements[0].startswith('판단 보류')
    assert result.disagreements == parsed.disagreements


def test_explicit_fields_are_verified_for_each_paper_and_partial_text_is_enough():
    _, (_, withheld) = run([card('ev-1'), card('other')], fulltext_requirements=[
        {'evidence_id': 'ev-1', 'fields': ['population', 'findings']},
        {'evidence_id': 'other', 'fields': ['comparator', 'findings']}])
    assert not withheld


@pytest.mark.parametrize('failure', ['missing_field', 'unread_range', 'hash', 'no_ranges', 'bad_span'])
def test_one_papers_invalid_required_passage_withholds_only_affected_claim(failure):
    other = card('other')
    data = other.model_dump(mode='json')
    if failure == 'missing_field':
        data['field_support'] = [f for f in data['field_support'] if f['field'] != 'comparator']
    elif failure == 'unread_range':
        data['analyzed_locators'] = ['text:characters:0-6']
    elif failure == 'no_ranges':
        data['analyzed_locators'] = []
    elif failure == 'hash':
        data['field_support'][-1]['source_sha256'] = 'sha256:foreign-raw'
    else:
        data['field_support'][-1]['source_start'] = -12
    other = EvidenceCard.model_validate_json(json.dumps(data))
    parsed, (result, withheld) = run([card('ev-1'), other], fulltext_requirements=[
        {'evidence_id': 'ev-1', 'fields': ['population']},
        {'evidence_id': 'other', 'fields': ['comparator']}])
    assert len(withheld) == 1
    assert result.agreements[0].startswith('판단 보류')
    assert result.disagreements == parsed.disagreements


@pytest.mark.parametrize('requirements', [
    [{'evidence_id': 'foreign', 'fields': ['findings']}],
    [{'evidence_id': 'ev-1', 'fields': ['findings']}, {'evidence_id': 'ev-1', 'fields': ['population']}],
])
def test_unknown_or_duplicated_source_requirements_fail_claim_locally(requirements):
    _, (_, withheld) = run([card('ev-1')], fulltext_requirements=requirements)
    assert len(withheld) == 1


def test_explicit_requirement_does_not_let_other_required_sources_escape():
    unread = card('other', access_level='ABSTRACT', analysis_scope='ABSTRACT')
    _, (_, withheld) = run([card('ev-1'), unread], fulltext_requirements=[
        {'evidence_id': 'ev-1', 'fields': ['findings']}])
    assert len(withheld) == 1


def test_per_source_fields_cannot_remove_global_required_fields():
    incomplete = card('ev-1')
    incomplete = incomplete.model_copy(update={'field_support': tuple(
        f for f in incomplete.field_support if f.field != 'population')})
    _, (_, withheld) = run([incomplete], required_evidence_fields=['population'],
                           fulltext_requirements=[{'evidence_id': 'ev-1', 'fields': ['findings']}])
    assert len(withheld) == 1


@pytest.mark.parametrize('missing_hash', [None, ''])
def test_missing_source_hash_cannot_establish_fulltext_verification(missing_hash):
    source = card('ev-1')
    source = source.model_copy(update={'source_sha256': missing_hash, 'field_support': tuple(
        f.model_copy(update={'source_sha256': missing_hash}) for f in source.field_support)})
    _, (_, withheld) = run([source])
    assert len(withheld) == 1


def test_context_only_abstract_does_not_require_whole_paper_verification():
    _, (_, withheld) = run([card('ev-1'), card('other', access_level='ABSTRACT', analysis_scope='ABSTRACT')],
        requires_fulltext_verification=False,
        fulltext_requirements=[{'evidence_id': 'ev-1', 'fields': ['findings']}])
    assert not withheld


@pytest.mark.parametrize('ranges,passes', [
    (['text:characters:0-10', 'text:characters:10-35'], True),
    (['text:characters:0-25', 'text:characters:27-35'], False),
    (['text:characters:0-9999'], False),
])
def test_analyzed_ranges_must_cover_required_quote_without_gaps(ranges, passes):
    _, (_, withheld) = run([card('ev-1', analyzed_locators=ranges)],
                          fulltext_requirements=[{'evidence_id': 'ev-1', 'fields': ['comparator']}])
    assert bool(withheld) is not passes


def test_report_shows_requested_per_source_verification_and_original_quote():
    from medic.reporting import ReportRenderer
    from tests.test_reporting import ReportingTests
    report = ReportRenderer.build(ReportingTests().result())
    data = report.synthesis_reviews[0].model_dump(mode='json')
    data['claim_reviews'][0]['fulltext_requirements'] = [
        {'evidence_id': 'ev-1', 'fields': ['population', 'comparator']}]
    review = SynthesisReview.model_validate_json(json.dumps(data))
    report = report.model_copy(update={'synthesis_reviews': (review,)})
    markdown = ReportRenderer.markdown(report)
    assert '주장별 전문 검증 요청' in markdown
    assert 'population' in markdown and 'comparator' in markdown
    assert 'ev\\-1' in markdown and 'finding' in markdown


@pytest.mark.parametrize('unread', [False, True])
def test_recorded_three_provider_verification_and_resume_are_claim_local(tmp_path, unread):
    from medic.config import ModelRegistry
    from medic.research_contracts import QuestionSet
    from medic.tribunal import TribunalService
    raw, gateways = fixtures.gateways()
    cards = (card('ev-1'), card('other', **({'access_level': 'ABSTRACT', 'analysis_scope': 'ABSTRACT'}
                                         if unread else {})))
    for provider in fixtures.PROVIDERS:
        data = fixtures.synthesis_review(provider)
        data['claim_reviews'][0].update(requires_fulltext_verification=True,
            evidence_ids=[c.evidence_id for c in cards],
            source_quotes=[{'evidence_id': c.evidence_id, 'quote': f.quote}
                           for c in cards for f in c.field_support],
            fulltext_requirements=[{'evidence_id': 'ev-1', 'fields': ['population']},
                                   {'evidence_id': 'other', 'fields': ['findings']}])
        raw[provider]._responses[-1] = data
    service = TribunalService(tmp_path, gateways)
    args = dict(run_id='per-source-fulltext', case_id='case-1',
        patient_context={'confirmed_facts': [], 'fact_details': [], 'uncertainties': []},
        question_set=QuestionSet.for_comprehensive_review('case-1'), evidence_cards=cards,
        estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
    result = service.run(**args)
    assert len(result.withheld_claims) == (1 if unread else 0)
    assert result.synthesis.disagreements == tuple(fixtures.synthesis()['disagreements'])
    assert len(result.synthesis_reviews) == 3
    assert service.run(**args) == result
    assert sum(g.call_count for g in raw.values()) == 13
