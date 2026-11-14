from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import QuestionSet
from medic.tribunal import TribunalService
import tests.test_tribunal as fixtures


class TribunalResumeTests(unittest.TestCase):
    def test_missing_claim_audit_is_retried_and_exact_aliases_are_restored(self):
        from medic.ledger import AppendOnlyJsonlLedger

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, approved = fixtures.gateways()
            gateway = raw["openai"]
            correct = fixtures.synthesis_review("openai")
            invalid = {**correct, "claim_reviews": correct["claim_reviews"][:-1]}
            aliased = {**correct, "claim_reviews": [
                {**review, "claim_id": f"C{index:03}"}
                for index, review in enumerate(correct["claim_reviews"], 1)
            ]}
            gateway._responses[-1:] = [invalid, aliased]
            args = dict(run_id="audit-resume", case_id="case-1",
                        patient_context={"confirmed_facts": [], "uncertainties": []},
                        question_set=QuestionSet.for_comprehensive_review("case-1"),
                        evidence_cards=(fixtures.TribunalTests().evidence(),),
                        estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
            service = TribunalService(root, approved)
            with self.assertRaisesRegex(ValueError, "cover every synthesis claim"):
                service.run(**args)
            ledger = AppendOnlyJsonlLedger(root / "runs" / "audit-resume" / "ledger.jsonl")
            self.assertFalse(any(r["kind"] == "SYNTHESIS_REVIEW_COMPLETED" for r in ledger.records()))
            result = service.run(**args)
            self.assertEqual(gateway.call_count, 6)
            self.assertEqual([r.claim_id for r in result.synthesis_reviews[0].claim_reviews],
                             [r["claim_id"] for r in correct["claim_reviews"]])

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
