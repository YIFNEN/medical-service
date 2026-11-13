from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import QuestionSet
import tests.test_tribunal as fixtures


def plan(provider):
    return {"provider": provider, "model": f"model-{provider}", "provider_response_id": f"plan-{provider}",
            "picots": {"population": "ankle fracture", "intervention": "ORIF", "comparator": "none",
                       "outcomes": ["union"], "timing": "followup", "study_designs": ["cohort"]},
            "queries": ["ankle fracture union"]}


class WorkflowResumeTests(unittest.TestCase):
    def test_search_failure_keeps_run_and_successful_plan_and_source_calls(self):
        from medic.workflow import ResearchWorkflow, ResearchRunFailed
        from medic.evidence import OfficialLiteratureRetriever
        class Source:
            def __init__(self, card, fail=False):
                self.card, self.fail, self.calls = card, fail, 0
            def search(self, query):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("search temporarily unavailable")
                return (self.card,)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, approved = fixtures.gateways()
            for provider in raw:
                raw[provider]._responses.insert(0, plan(provider))
            raw['anthropic']._responses[1:1] = [
                {'issues': [{'issue_id': 'i', 'question': 'General explanation', 'required_evidence': ['review'],
                             'missing_information': [], 'decision_changers': [], 'triggers': []}]},
                {'decisions': [{'evidence_id': fixtures.TribunalTests().evidence().evidence_id,
                                'verdict': 'INCLUDE', 'reason': 'Relevant review', 'issue_ids': ['i'],
                                'impact': 1, 'applicability': 1, 'methods_needed': False}]},
            ]
            first = Source(fixtures.TribunalTests().evidence())
            second = Source(first.card, fail=True)
            retriever = OfficialLiteratureRetriever((first, second))
            workflow = ResearchWorkflow(root, approved, retriever)
            args = dict(run_id="workflow-resume", case_id="case-1",
                        search_approved=True,
                        patient_context={"confirmed_facts": [], "fact_details": [], "uncertainties": []},
                        question_set=QuestionSet.for_comprehensive_review("case-1"),
                        estimate=ModelRegistry.default().estimate_deliberation_cost(100).approve())
            with self.assertRaisesRegex(ResearchRunFailed, "workflow-resume"):
                workflow.run(**args)
            second.fail = False
            result = workflow.run(**args)
            self.assertEqual(first.calls, 1)
            self.assertEqual(second.calls, 2)
            self.assertEqual(sum(g.call_count for g in raw.values()), 18)
            self.assertEqual(len(result.calls), 18)
            self.assertEqual(sum(call.stage == "LITERATURE_SEARCH_PLANNING" for call in result.calls), 3)
            self.assertEqual(workflow.run(**args), result)
            self.assertEqual(sum(g.call_count for g in raw.values()), 18)

    def test_fulltext_comparison_reuses_pinned_search_results(self):
        from medic.evidence import OfficialLiteratureRetriever
        from medic.ledger import AppendOnlyJsonlLedger
        from medic.workflow import ResearchWorkflow, ResearchRunFailed

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, gateways = fixtures.gateways()
            workflow = ResearchWorkflow(root, gateways, OfficialLiteratureRetriever(()))
            questions = QuestionSet.for_comprehensive_review('case-1')
            args = dict(case_id='case-1', patient_context={'confirmed_facts': []}, question_set=questions,
                        estimate=ModelRegistry.default().estimate_deliberation_cost(100).approve(),
                        search_approved=True)
            card = fixtures.TribunalTests().evidence()
            from medic.evidence import EvidenceBundle
            bundle = EvidenceBundle(cards=(card,), queries=('ankle fracture union',))
            from types import SimpleNamespace
            with patch('medic.workflow.LiteratureSearchPlanner.plan', return_value=SimpleNamespace(queries=bundle.queries)) as plan_calls, \
                    patch.object(workflow.retriever, 'search', return_value=bundle) as searches, \
                    patch('medic.workflow.LiteratureScreener.issues', side_effect=RuntimeError):
                with self.assertRaises(ResearchRunFailed):
                    workflow.run(**args, run_id='baseline', fulltext_limit=0)
                with self.assertRaises(ResearchRunFailed):
                    workflow.run(**args, run_id='followup', fulltext_limit=1, reuse_search_from='baseline')
                self.assertEqual(searches.call_count, 1)
                self.assertEqual(plan_calls.call_count, 1)
                records = AppendOnlyJsonlLedger(root / 'runs' / 'followup' / 'ledger.jsonl').records()
                self.assertTrue(any(r['kind'] == 'LITERATURE_SEARCH_SNAPSHOT' for r in records))
                with self.assertRaisesRegex(ValueError, 'inputs differ'):
                    workflow.run(**{**args, 'patient_context': {'confirmed_facts': ['changed']}},
                                 run_id='changed', fulltext_limit=1, reuse_search_from='baseline')
