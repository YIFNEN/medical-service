from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.reporting import ReportRenderer
from medic.research_contracts import QuestionSet
from medic.tribunal import TribunalService
from tests.test_tribunal import gateways, TribunalTests as Fixtures


class LongitudinalReportingTests(unittest.TestCase):
    def test_same_key_findings_reach_all_report_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, approved = gateways()
            details = [{"fact_id": f"fact-{index}", "fact_key": "union", "value": value,
                        "source_locators": [f"source-{index}"], "user_confirmed": True,
                        "scope": {"modality": modality}, "conflict": True,
                        "confirmation_scope": "SOURCE_TRANSCRIPTION_ONLY"}
                       for index, (value, modality) in enumerate((("union", "CT"), ("R/O nonunion", "SPECT")))]
            result = TribunalService(root, approved).run(case_id="case-1",
                patient_context={"confirmed_facts": [f"union: {d['value']}" for d in details],
                                 "fact_details": details, "uncertainties": []},
                question_set=QuestionSet.for_comprehensive_review("case-1"),
                evidence_cards=(Fixtures().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve())
            report = ReportRenderer.build(result)
            paths = ReportRenderer.write(report, root / "reports")
            for path in paths.values():
                text = path.read_text(encoding="utf-8")
                self.assertIn("SPECT", text)
                self.assertIn("R/O nonunion", text)
