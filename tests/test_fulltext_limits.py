from types import SimpleNamespace
from unittest.mock import patch

from medic.config import ModelRegistry
from medic.evidence import EvidenceBundle
from medic.research_contracts import QuestionSet
from medic.settings import ProcessingSettings
from medic.workflow import ResearchWorkflow
import tests.test_tribunal as fixtures


def test_already_acquired_fulltext_still_consumes_analysis_target_limit(tmp_path):
    _, approved = fixtures.gateways()
    first = fixtures.TribunalTests().evidence().model_copy(update={'access_level': 'FULL_TEXT', 'source_text': 'finding',
        'appraisal_status': 'UNAPPRAISED', 'screening_status': 'INCLUDE', 'fulltext_status': 'ACQUIRED'})
    second = first.model_copy(update={'evidence_id': 'second'})
    cards = (first, second)
    retriever = SimpleNamespace(sources=(), search=lambda *a, **k: EvidenceBundle(cards=cards))
    settings = ProcessingSettings(evidence_limit=2, fulltext_limit=1)
    with patch('medic.workflow.LiteratureSearchPlanner.plan', return_value=SimpleNamespace(queries=())), \
         patch('medic.workflow.LiteratureScreener') as screener, \
         patch('medic.workflow.EvidenceAppraiser') as appraiser:
        screener.return_value.issues.return_value = ()
        screener.return_value.screen.return_value = cards
        appraiser.return_value.appraise.side_effect = lambda c,*a: c.model_copy(update={'appraisal_status': 'APPRAISED', 'analysis_scope': 'FULL_TEXT'})
        result = ResearchWorkflow(tmp_path, approved, retriever).run(run_id='fulltext-cap', case_id='case-1',
            patient_context={'confirmed_facts': [], 'fact_details': [], 'uncertainties': []},
            question_set=QuestionSet.for_comprehensive_review('case-1'), settings=settings, search_approved=True,
            estimate=ModelRegistry.default().estimate_research_cost(100, settings=settings).approve())
    assert appraiser.return_value.appraise.call_count == 1
    assert result.patient_context['literature_review']['counts']['fulltext_analyzed'] == 1
    assert 'second' in result.patient_context['literature_review']['pending_fulltext_ids']
