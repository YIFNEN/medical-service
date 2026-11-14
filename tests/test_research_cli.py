from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image

from medic.cli import main
from medic.config import ModelRegistry
from medic.ledger import AppendOnlyJsonlLedger
from medic.tribunal import TribunalService
from tests.test_tribunal import TribunalTests, gateways
from medic.research_contracts import QuestionSet


class ResearchCliTests(unittest.TestCase):
    def test_context_preserves_conflicting_same_key_facts_and_all_intakes(self):
        from medic.cli import _latest_context, _latest_intake
        from medic.intake import FileIntakeService
        from medic.ledger import ContentAddressedStore
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            for index, value in enumerate(("CT union", "SPECT R/O nonunion")):
                source = root / f"record-{index}.txt"
                source.write_text(value)
                FileIntakeService(ContentAddressedStore(root), ledger).ingest_files("case-1", [source])
                ledger.append_once("HIGH_IMPACT_FACTS_CONFIRMED", f"context-{index}", {
                    "case_id": "case-1", "context": {"confirmed_facts": [f"union: {value}"],
                    "fact_details": [{"fact_key": "union", "value": value,
                                      "source_locators": [str(index)]}], "uncertainties": []}})
            self.assertEqual(len(_latest_context(ledger, "case-1")["fact_details"]), 2)
            self.assertEqual(len(_latest_intake(ledger, "case-1").files), 2)

    def invoke(self, argv):
        output = StringIO()
        with redirect_stdout(output):
            code = main(argv)
        return code, json.loads(output.getvalue())

    def test_ingest_files_positional_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "record.txt"
            source.write_text("right ankle ORIF", encoding="utf-8")
            code, payload = self.invoke([
                "ingest-files", "case-1", str(source), "--store", str(root / "store")
            ])
            self.assertEqual(code, 0)
            self.assertEqual(payload["case_id"], "case-1")
            self.assertEqual(payload["file_count"], 1)

    def test_review_without_approval_estimates_only_three_extraction_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            code, payload = self.invoke([
                "review", "case-1", "--interactive", "--store", temp
            ])
            self.assertEqual(code, 2)
            self.assertEqual(payload["call_count"], 3)
            self.assertEqual(payload["paid_calls_made"], 0)

    def test_review_selects_only_named_pdf_pages_and_preserves_original_locators(self):
        from medic.intake import FileIntakeService
        from medic.ledger import ContentAddressedStore

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "record.pdf"
            with fitz.open() as document:
                for number in range(1, 6):
                    page = document.new_page()
                    if number != 5:
                        page.insert_text((72, 72), f"Page marker {number}")
                document.save(pdf)
            unrelated = root / "other.txt"
            unrelated.write_text("Other artifact marker", encoding="utf-8")
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            store = ContentAddressedStore(root)
            intake = FileIntakeService(store, ledger).ingest_files("case-1", [pdf, unrelated])
            artifact = intake.files[0].artifact.artifact_id
            args = ["review", "case-1", "--interactive", "--store", temp, "--backend", "codex",
                    "--pdf-page", artifact, "5", "--pdf-page", artifact, "2"]
            code, payload = self.invoke(args)
            self.assertEqual(code, 2)
            self.assertEqual(payload["call_count"], 3)
            self.assertEqual(payload["selected_pdf_pages"], [
                {"artifact_id": artifact, "page": 2}, {"artifact_id": artifact, "page": 5}
            ])

            from medic.providers import ApprovedGateway, RecordedGateway

            independent = {"provider": "openai", "model": "fixture",
                           "provider_response_id": "fixture-extraction", "facts": []}
            reconciliation = {"provider_response_id": "fixture-summary", "facts": [], "uncertainties": []}
            raw = RecordedGateway("openai", "fixture", [independent, independent, reconciliation])
            raw.backend = "codex"
            with patch("medic.cli.build_live_gateways", return_value={"openai": ApprovedGateway(raw)}), \
                    patch.object(raw, "generate_raw", wraps=raw.generate_raw) as generate:
                approved = args + ["--codex-calls", "3", "--approve-transmission"]
                code, result = self.invoke(approved)
                self.assertEqual(code, 0)
                self.assertEqual(result["selected_pdf_pages"], payload["selected_pdf_pages"])
                self.assertEqual(self.invoke(approved)[1]["extraction_id"], result["extraction_id"])
                self.assertEqual(generate.call_count, 3)
                selected_blocks = [block for block in intake.blocks if block.page_number in {2, 5}]
                for call in generate.call_args_list[:2]:
                    prompt = call.kwargs["prompt"]
                    self.assertIn(f"{artifact}|pdf_bbox|page:2;", prompt)
                    self.assertIn("Page marker 2", prompt)
                    self.assertNotIn("Page marker 1", prompt)
                    self.assertNotIn("Page marker 3", prompt)
                    self.assertNotIn("Other artifact marker", prompt)
                    self.assertEqual(call.kwargs["image_paths"], tuple(
                        Path(block.rendered_image_path) for block in selected_blocks
                        if block.rendered_image_path
                    ))

            for invalid in (("unknown-artifact", "2"), (artifact, "6"), (artifact, "0")):
                with self.subTest(invalid=invalid), patch("medic.cli.build_live_gateways") as gateways:
                    with self.assertRaisesRegex(ValueError, "PDF page selection"):
                        main(["review", "case-1", "--interactive", "--store", temp, "--backend", "codex",
                              "--pdf-page", *invalid, "--codex-calls", "3", "--approve-transmission"])
                    gateways.assert_not_called()

    def test_scoped_review_rejects_non_pdf_from_legacy_intake(self):
        from medic.intake import FileIntakeService
        from medic.ledger import ContentAddressedStore

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            disguised = root / "non-pdf.pdf"
            Image.new("RGB", (10, 10)).save(disguised, format="PNG")
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            intake = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files(
                "case-1", [disguised]
            )
            with self.assertRaisesRegex(ValueError, "original PDF format"):
                self.invoke(["review", "case-1", "--interactive", "--store", temp,
                             "--pdf-page", intake.files[0].artifact.artifact_id, "1"])

    def test_deliberate_without_approval_only_prints_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            AppendOnlyJsonlLedger(root / "ledger.jsonl").append_once(
                "HIGH_IMPACT_FACTS_CONFIRMED", "context-1",
                {"case_id": "case-1", "context": {"confirmed_facts": ["laterality: RIGHT"], "uncertainties": []}},
            )
            code, payload = self.invoke([
                "deliberate", "case-1", "--question", "관절염 위험?", "--store", str(root)
            ])
            self.assertEqual(code, 2)
            self.assertFalse(payload["approved"])
            self.assertGreater(payload["estimated_usd"], 0)
            self.assertEqual(payload["paid_calls_made"], 0)
            self.assertEqual(payload["call_count"], 117)

    def test_patient_context_accumulates_new_events_without_mutating_old_context(self):
        from medic.cli import _latest_context

        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / "ledger.jsonl")
            ledger.append_once("HIGH_IMPACT_FACTS_CONFIRMED", "context-1", {
                "case_id": "case-1", "context": {"confirmed_facts": ["surgery: ORIF"],
                "fact_details": [{"fact_key": "surgery", "value": "ORIF", "source_locators": ["artifact-1|text|characters:0-4"],
                                  "user_confirmed": True}], "uncertainties": []},
            })
            ledger.append_once("HIGH_IMPACT_FACTS_CONFIRMED", "context-2", {
                "case_id": "case-1", "context": {"confirmed_facts": ["union: PROGRESSING"],
                "fact_details": [{"fact_key": "union", "value": "PROGRESSING", "source_locators": ["artifact-2|text|characters:0-11"],
                                  "user_confirmed": True}],
                "uncertainties": ["alignment pending"]},
            })
            context = _latest_context(ledger, "case-1")
            self.assertEqual(context["confirmed_facts"], ["surgery: ORIF", "union: PROGRESSING"])
            self.assertEqual(context["uncertainties"], ["alignment pending"])
    def test_report_cli_exports_selected_format(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, approved = gateways()
            TribunalService(root, approved).run(
                run_id="run-cli", case_id="case-1",
                patient_context={
                    "confirmed_facts": ["surgery_name: ORIF"],
                    "fact_details": [{
                        "fact_key": "surgery_name", "value": "ORIF",
                        "source_locators": ["artifact-1|text|characters:0-4"], "user_confirmed": True,
                    }],
                    "uncertainties": [],
                },
                question_set=QuestionSet.for_comprehensive_review("case-1"),
                evidence_cards=(TribunalTests().evidence(),),
                estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve(),
            )
            code, payload = self.invoke([
                "report", "run-cli", "--format", "markdown", "--store", str(root)
            ])
            self.assertEqual(code, 0)
            self.assertTrue(Path(payload["path"]).exists())
            self.assertEqual(payload["format"], "markdown")


if __name__ == "__main__":
    unittest.main()
