from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from medic.config import ModelRegistry
from medic.reporting import ReportRenderer
from medic.research_contracts import MedicalDecisionReport, QuestionSet
from medic.tribunal import TribunalResult, TribunalService
from tests.test_tribunal import TribunalTests, gateways


class ReportingTests(unittest.TestCase):
    def test_unread_source_locations_are_visible(self):
        report = ReportRenderer.build(self.result())
        card = report.evidence_cards[0].model_copy(update={
            'unresolved_source_locators': ('xml:article/body[1]/graphic[1]',),
            'fulltext_status': 'ACQUIRED_PARTIAL', 'analysis_scope': 'PARTIAL_FULL_TEXT'})
        report = report.model_copy(update={'evidence_cards': (card,)})
        self.assertIn('xml:article/body[1]/graphic[1]', ReportRenderer.markdown(report))

    def result(self) -> TribunalResult:
        with tempfile.TemporaryDirectory() as temp:
            _, approved = gateways()
            return TribunalService(Path(temp), approved).run(
                run_id="run-report",
                case_id="case-1",
                question_set=QuestionSet.for_comprehensive_review("case-1", "관절염 가능성?"),
                patient_context={
                    "confirmed_facts": [
                        "surgery_name: ORIF",
                        "unsafe_note: <script>alert('patient')</script>\n# 위조 섹션",
                    ],
                    "fact_details": [
                        {
                            "fact_key": "surgery_name", "value": "ORIF",
                            "source_locators": ["artifact-1|pdf_bbox|page:1;bbox:1,2,3,4"],
                            "user_confirmed": True,
                        },
                        {
                            "fact_key": "unsafe_note",
                            "value": "<script>alert('patient')</script>\n# 위조 섹션",
                            "source_locators": ["artifact-2|text|characters:0-end"],
                            "user_confirmed": True,
                        },
                    ],
                    "uncertainties": ["체중부하 X-ray 정렬 미확인"],
                },
                evidence_cards=(TribunalTests().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(1000).approve(),
            )

    def test_build_and_write_korean_markdown_html_and_json(self):
        report = ReportRenderer.build(self.result())
        self.assertIsInstance(report, MedicalDecisionReport)
        self.assertEqual(report.confirmed_facts[0], "surgery_name: ORIF")
        self.assertEqual(report.confirmed_fact_details[0].fact_key, "surgery_name")
        with tempfile.TemporaryDirectory() as temp:
            paths = ReportRenderer.write(report, Path(temp))
            self.assertEqual(set(paths), {"markdown", "html", "json"})
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("# 개인 발목 의학적 판단 지원 보고서", markdown)
            self.assertIn("## 4. 일곱 영역별 독립 의견", markdown)
            self.assertIn("## 8. 조건부 검사·치료 행동순위", markdown)
            self.assertIn("artifact-1|pdf_bbox|page:1", markdown)
            self.assertIn("전제조건", markdown)
            self.assertIn("alignment assessment", markdown)
            self.assertIn("radiation exposure", markdown)
            self.assertIn("PERSONAL_RESEARCH", markdown)
            self.assertIn("AI_MEDICAL_DECISION_SUPPORT", markdown)
            self.assertIn("추가 필요 정보", markdown)
            self.assertIn("판단 보류", markdown)
            self.assertIn("state imaging limitation", markdown)
            self.assertIn("retain dissent", markdown)
            self.assertIn("근거 ID", markdown)
            self.assertNotIn("<script>", markdown)
            self.assertNotIn("\n# 위조 섹션", markdown)
            self.assertIn("https://pubmed.ncbi.nlm.nih.gov/1/", markdown)
            self.assertIn("의료행위를 자동 실행하지 않습니다", markdown)
            html = paths["html"].read_text(encoding="utf-8")
            self.assertIn("<html", html)
            self.assertNotIn("<script>", html)
            self.assertNotIn("<h1>위조 섹션</h1>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertRegex(html, r"<p>추가 필요 정보:</p>\s*<ul>")
            self.assertIn("<li>weight-bearing radiograph</li>", html)
            self.assertIn("<li>cannot infer cartilage loss</li>", html)
            self.assertIn("<li>state imaging limitation</li>", html)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run-report")

    def test_build_rejects_unlinked_or_non_string_context(self):
        result = self.result()
        contexts = (
            {**result.patient_context, "confirmed_facts": ["unlinked fact"]},
            {**result.patient_context, "confirmed_facts": [1]},
            {**result.patient_context, "uncertainties": [1]},
            {
                **result.patient_context,
                "fact_details": [{
                    "fact_key": "surgery_name", "value": "ORIF",
                    "source_locators": [], "user_confirmed": True,
                }],
                "confirmed_facts": ["surgery_name: ORIF"],
            },
        )
        for context in contexts:
            with self.subTest(context=context), self.assertRaises(
                (ValueError, TypeError)
            ):
                ReportRenderer.build(result.model_copy(update={"patient_context": context}))

    def test_report_contract_rejects_noncanonical_facts_time_and_cost(self):
        report = ReportRenderer.build(self.result())
        payload = report.model_dump()
        duplicate_details = report.confirmed_fact_details + (report.confirmed_fact_details[0],)
        invalid_updates = (
            {"confirmed_facts": tuple(reversed(report.confirmed_facts))},
            {
                "confirmed_facts": report.confirmed_facts + (report.confirmed_facts[0],),
                "confirmed_fact_details": duplicate_details,
            },
            {"approved_cost_upper_bound_usd": float("inf")},
            {"approved_cost_upper_bound_usd": -1.0},
            {"data_cutoff": datetime.now()},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                MedicalDecisionReport.model_validate({**payload, **updates})

if __name__ == "__main__":
    unittest.main()
