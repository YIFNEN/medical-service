import json
from datetime import UTC, datetime

from medic.research_contracts import ProposedSynthesis, SynthesisReview
from medic.claim_grounding import ground_synthesis
import tests.test_tribunal as fixtures


def test_pubmed_update_status_is_explicit_and_retraction_blocks_support():
    from medic.evidence import normalize_pubmed, CitationGate
    import pytest
    card = normalize_pubmed({'uid': '123', 'title': 'synthetic', '_abstract': 'study',
                              '_publication_updates': [{'relation': 'RetractionIn', 'pmid': '456'}]}, datetime.now(UTC))
    assert card.publication_status == 'RETRACTED'
    assert card.publication_updates[0]['pmid'] == '456'
    card = card.model_copy(update={'appraisal_status': 'APPRAISED'})
    with pytest.raises(ValueError):
        CitationGate((card,)).validate((card.evidence_id,))
    unverified = normalize_pubmed({'uid': '123', 'title': 'synthetic'}, datetime.now(UTC))
    assert unverified.publication_status == 'NOT_CHECKED'


def test_partial_paper_supports_only_claims_with_verified_required_fields():
    from medic.research_contracts import EvidenceFieldSupport
    card = fixtures.TribunalTests().evidence()
    text = 'Adults; observational cohort; union.'
    fields = tuple(EvidenceFieldSupport(field=field, value=quote, quote=quote,
                                        source_start=text.index(quote), source_end=text.index(quote)+len(quote))
                   for field, quote in [('population', 'Adults'), ('study_design', 'observational cohort'), ('findings', 'union')])
    card = card.model_copy(update={'access_level': 'FULL_TEXT', 'analysis_scope': 'PARTIAL_FULL_TEXT',
        'fulltext_required': True, 'source_text': text, 'field_support': fields,
        'analyzed_locators': (f'text:characters:0-{len(text)}',)})
    synthesis = ProposedSynthesis.model_validate_json(json.dumps(fixtures.synthesis()))
    payload = fixtures.synthesis_review('openai')
    for audit in payload['claim_reviews']:
        audit['source_quotes'] = [{'evidence_id': card.evidence_id, 'quote': field.quote} for field in fields]
        audit['requires_fulltext_verification'] = True
    review = SynthesisReview.model_validate_json(json.dumps(payload))
    _, withheld = ground_synthesis(synthesis, (review,), (card,), {})
    assert not withheld
    payload['claim_reviews'][0]['required_evidence_fields'] = ['comparator']
    review = SynthesisReview.model_validate_json(json.dumps(payload))
    _, withheld = ground_synthesis(synthesis, (review,), (card,), {})
    assert len(withheld) == 1


def test_study_counts_do_not_add_review_and_primary_as_independent_studies():
    from medic.literature_integrity import study_counts
    base = fixtures.TribunalTests().evidence()
    primary = base.model_copy(update={'evidence_id': 'primary', 'study_id': 'cohort-1', 'study_link_status': 'USER_REVIEWED'})
    followup = base.model_copy(update={'evidence_id': 'followup', 'study_id': 'cohort-1', 'study_link_status': 'USER_REVIEWED'})
    review = base.model_copy(update={'evidence_id': 'review', 'study_design': 'systematic review',
                                     'included_study_ids': ('cohort-1',), 'study_link_status': 'USER_REVIEWED'})
    counts = study_counts((primary, followup, review))
    assert counts['selected_cards'] == 3
    assert counts['verified_unique_primary_studies'] == 1
    assert counts['review_cards'] == 1
    assert counts['independence_status'] != 'ALL_INDEPENDENT'


def test_dedup_does_not_lose_retraction_from_shorter_source():
    from medic.evidence import EvidenceBundle
    base = fixtures.TribunalTests().evidence()
    retracted = base.model_copy(update={'pmid': '123', 'source_text': 'short', 'publication_status': 'RETRACTED',
                                        'publication_updates': ({'relation': 'RetractionIn', 'pmid': '456'},)})
    other = base.model_copy(update={'source': 'EUROPE_PMC', 'pmid': '123', 'source_text': 'longer source text'})
    merged = EvidenceBundle.deduplicated((retracted, other)).cards[0]
    assert merged.publication_status == 'RETRACTED'
    assert merged.publication_updates
