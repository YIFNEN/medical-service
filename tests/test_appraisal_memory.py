import tracemalloc

from medic.evidence_appraisal import EvidenceAppraiser
from medic.settings import ProcessingSettings
import tests.test_tribunal as fixtures


def test_chunk_limit_bounds_materialization_not_only_provider_calls(monkeypatch):
    card = fixtures.TribunalTests().evidence().model_copy(update={
        'source_text': 'x' * 300000, 'access_level': 'FULL_TEXT', 'fulltext_status': 'ACQUIRED'})
    appraiser = EvidenceAppraiser(None, None, settings=ProcessingSettings(fulltext_chunk_chars=1, fulltext_max_chunks=2))
    monkeypatch.setattr(appraiser, '_appraise_single', lambda card, *a: card)
    tracemalloc.start()
    result = appraiser.appraise(card, {}, None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.analysis_scope == 'PARTIAL_FULL_TEXT'
    assert len(result.analyzed_locators) == 2
    assert peak < 5 * 1024 * 1024
