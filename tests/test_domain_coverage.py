from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.providers import ApprovedGateway, RecordedGateway
from medic.research_contracts import QuestionSet
from medic.tribunal import TribunalService
import tests.test_tribunal as fixtures


class DomainCoverageTests(unittest.TestCase):
    def test_missing_domain_blocks_completion_before_peer_review(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = fixtures.opinion("openai")
            payload["domain_assessments"] = []
            raw, approved = fixtures.gateways()
            approved["openai"] = ApprovedGateway(RecordedGateway("openai", "model-openai", [payload]))
            with self.assertRaisesRegex(ValueError, "seven domains"):
                TribunalService(Path(temp), approved).run(case_id="case-1",
                    patient_context={"confirmed_facts": [], "fact_details": [], "uncertainties": []},
                    question_set=QuestionSet.for_comprehensive_review("case-1"),
                    evidence_cards=(fixtures.TribunalTests().evidence(),),
                    estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
