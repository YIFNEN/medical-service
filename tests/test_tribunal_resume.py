from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import QuestionSet
from medic.tribunal import TribunalService
import tests.test_tribunal as fixtures


class TribunalResumeTests(unittest.TestCase):
    def test_peer_prompt_hides_provider_metadata_and_restores_target_ids(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            raw, approved = fixtures.gateways()
            result = TribunalService(Path(temp), approved).run(case_id="case-1",
                patient_context={"confirmed_facts": [], "uncertainties": []},
                question_set=QuestionSet.for_comprehensive_review("case-1"),
                evidence_cards=(fixtures.TribunalTests().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
            for gateway in raw.values():
                payload = json.loads(gateway.prompts[1].split("\n", 2)[-1])
                for target in payload["targets"]:
                    self.assertNotIn("provider", target)
                    self.assertNotIn("model", target)
                    self.assertTrue(target["opinion_id"].startswith("blind-"))
            for opinion in result.opinions:
                reviews = [r for r in result.peer_reviews if r.target_opinion_id == opinion.opinion_id]
                self.assertEqual({r.reviewer_provider for r in reviews}, set(fixtures.PROVIDERS) - {opinion.provider})

    def test_completed_resume_is_identical_and_model_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, approved = fixtures.gateways()
            service = TribunalService(root, approved)
            args = dict(run_id="completed-run", case_id="case-1",
                        patient_context={"confirmed_facts": [], "uncertainties": []},
                        question_set=QuestionSet.for_comprehensive_review("case-1"),
                        evidence_cards=(fixtures.TribunalTests().evidence(),),
                        estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
            first = service.run(**args)
            second = service.run(**args)
            self.assertEqual(first, second)
            self.assertEqual(sum(g.call_count for g in raw.values()), 13)
            approved["google"] = ApprovedGateway(RecordedGateway("google", "changed-model", []))
            with self.assertRaisesRegex(ValueError, "model|configuration"):
                TribunalService(root, approved).run(**args)
