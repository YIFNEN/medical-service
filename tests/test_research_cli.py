from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
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
            source_page = Path(next(block.rendered_image_path for block in intake.blocks
                                    if block.page_number == 5))
            source_bytes = source_page.read_bytes()
            actual_generate = raw.generate_raw

            def generate_with_tampered_cache(**request):
                source_page.write_bytes(b"changed after page verification")
                try:
                    self.assertEqual([path.read_bytes() for path in request["image_paths"]], [source_bytes])
                    self.assertNotIn(source_page, request["image_paths"])
                    return actual_generate(**request)
                finally:
                    source_page.write_bytes(source_bytes)

            with patch("medic.cli.build_live_gateways", return_value={"openai": ApprovedGateway(raw)}), \
                    patch.object(raw, "generate_raw", side_effect=generate_with_tampered_cache) as generate:
                approved = args + ["--codex-calls", "3", "--approve-transmission"]
                code, result = self.invoke(approved)
                self.assertEqual(code, 0)
                self.assertEqual(result["selected_pdf_pages"], payload["selected_pdf_pages"])
                self.assertEqual(self.invoke(approved)[1]["extraction_id"], result["extraction_id"])
                self.assertEqual(generate.call_count, 3)
                for call in generate.call_args_list[:2]:
                    prompt = call.kwargs["prompt"]
                    self.assertIn(f"{artifact}|pdf_bbox|page:2;", prompt)
                    self.assertIn("Page marker 2", prompt)
                    self.assertNotIn("Page marker 1", prompt)
                    self.assertNotIn("Page marker 3", prompt)
                    self.assertNotIn("Other artifact marker", prompt)

            for invalid in (("unknown-artifact", "2"), (artifact, "6"), (artifact, "0")):
                with self.subTest(invalid=invalid), patch("medic.cli.build_live_gateways") as gateways:
                    with self.assertRaisesRegex(ValueError, "PDF page selection"):
                        main(["review", "case-1", "--interactive", "--store", temp, "--backend", "codex",
                              "--pdf-page", *invalid, "--codex-calls", "3", "--approve-transmission"])
                    gateways.assert_not_called()

            Image.new("RGB", (10, 10)).save(source_page)
            with patch("medic.cli.build_live_gateways") as gateways:
                with self.assertRaisesRegex(ValueError, "PDF page image does not match"):
                    main(approved)
                gateways.assert_not_called()
            source_page.write_bytes(source_bytes)
            blocks = tuple(block.model_copy(update={"text": "Unrelated altered text"})
                           if block.page_number == 2 else block for block in intake.blocks)
            corrupted = intake.model_copy(update={"blocks": blocks})
            ledger.append_once("REAL_FILES_INGESTED", "altered-intake",
                               {"intake": corrupted.model_dump(mode="json")})
            with patch("medic.cli.build_live_gateways") as gateways:
                with self.assertRaisesRegex(ValueError, "PDF page text does not match"):
                    main(approved)
                gateways.assert_not_called()

    def test_scoped_review_does_not_reuse_batch_for_replaced_unscoped_image(self):
        from medic.cli import _latest_intake, _select_pdf_pages
        from medic.extraction import extraction_batches
        from medic.intake import FileIntakeService
        from medic.ledger import ContentAddressedStore
        from medic.providers import ApprovedGateway, RecordedGateway

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            pdf = root / "scan.pdf"
            with fitz.open() as document:
                document.new_page()
                document.save(pdf)
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            intake = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files("case-1", [pdf])
            page_path = Path(intake.blocks[0].rendered_image_path)
            genuine = page_path.read_bytes()
            Image.new("RGB", (10, 10)).save(page_path)
            independent = {"provider": "openai", "model": "fixture", "provider_response_id": "reply", "facts": []}
            reconciled = {"provider_response_id": "summary", "facts": [], "uncertainties": []}
            raw = RecordedGateway("openai", "fixture", [independent, independent, reconciled] * 2)
            raw.backend = "codex"
            args = ["review", "case-1", "--interactive", "--store", str(root),
                    "--backend", "codex", "--codex-calls", "3", "--approve-transmission"]
            with patch("medic.cli.build_live_gateways", return_value={"openai": ApprovedGateway(raw)}):
                self.assertEqual(self.invoke(args + ["--approval-id", "unscoped"])[0], 0)
                page_path.write_bytes(genuine)
                original = _latest_intake(ledger, "case-1")
                selected, _ = _select_pdf_pages(original, [[intake.files[0].artifact.artifact_id, "1"]],
                                                ContentAddressedStore(root))
                self.assertEqual(extraction_batches(selected)[0].intake_id,
                                 extraction_batches(original)[0].intake_id)
                scoped = args + ["--pdf-page", intake.files[0].artifact.artifact_id, "1",
                                 "--approval-id", "scoped"]
                self.assertEqual(self.invoke(scoped)[0], 0)
                self.assertEqual(self.invoke(scoped)[0], 0)
            self.assertEqual(raw.call_count, 6)

    def test_scoped_review_authenticates_password_protected_pdf_page(self):
        from medic.cli import _select_pdf_pages
        from medic.intake import FileIntakeService
        from medic.ledger import ContentAddressedStore

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "encrypted.pdf"
            with fitz.open() as document:
                document.new_page()
                document.save(pdf, encryption=fitz.PDF_ENCRYPT_AES_256,
                              owner_pw="fixture-owner", user_pw="fixture-secret")
            with chdir(root):
                store = ContentAddressedStore(Path("relative-store"))
                ledger = AppendOnlyJsonlLedger(store.root / "ledger.jsonl")
                intake = FileIntakeService(store, ledger).ingest_files(
                    "case-1", [pdf], password_provider=lambda _: "fixture-secret"
                )
                artifact = intake.files[0].artifact.artifact_id
                with patch("medic.cli.getpass.getpass", return_value="fixture-secret"):
                    selected, images = _select_pdf_pages(intake, [[artifact, "1"]], store,
                                                         prompt_password=True)
                self.assertEqual(Path(selected.blocks[0].rendered_image_path),
                                 Path(intake.blocks[0].rendered_image_path).resolve())
                self.assertEqual(len(images), 1)
            args = ["review", "case-1", "--interactive", "--store", str(root / "relative-store"),
                    "--pdf-page", artifact, "1"]
            with self.assertRaisesRegex(ValueError, "PDF password required"):
                self.invoke(args)
            with patch("medic.cli.getpass.getpass", return_value="fixture-secret"):
                self.assertEqual(self.invoke(args + ["--pdf-password-prompt"])[0], 2)

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

    def test_scoped_review_rejects_mismatched_content_hash_and_duplicate_image_block_id(self):
        from medic.cli import _select_pdf_pages
        from medic.intake import FileIntakeService
        from medic.ledger import ContentAddressedStore

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "pages.pdf"
            with fitz.open() as document:
                document.new_page()
                document.new_page()
                document.save(pdf)
            store = ContentAddressedStore(root)
            intake = FileIntakeService(store, AppendOnlyJsonlLedger(root / "ledger.jsonl")).ingest_files(
                "case-1", [pdf]
            )
            file = intake.files[0]
            artifact_id = file.artifact.artifact_id
            wrong_artifact = file.artifact.model_copy(update={"content_hash": "sha256:" + "0" * 64})
            wrong_intake = intake.model_copy(update={"files": (file.model_copy(update={"artifact": wrong_artifact}),)})
            with self.assertRaisesRegex(ValueError, "PDF artifact does not match"):
                _select_pdf_pages(wrong_intake, [[artifact_id, "1"]], store)
            first, second = intake.blocks
            wrong_intake = intake.model_copy(update={
                "blocks": (first, second.model_copy(update={"block_id": first.block_id})),
            })
            with self.assertRaisesRegex(ValueError, "PDF page image does not match"):
                _select_pdf_pages(wrong_intake, [[artifact_id, "1"], [artifact_id, "2"]], store)

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

    def test_unreviewed_extraction_does_not_enter_model_context(self):
        from medic.cli import _latest_context

        with tempfile.TemporaryDirectory() as temp:
            ledger = AppendOnlyJsonlLedger(Path(temp) / "ledger.jsonl")
            confirmed = {"fact_id": "confirmed", "fact_key": "alignment", "value": "Verified wording",
                         "source_locators": ["artifact|pdf_bbox|page:1"], "user_confirmed": True,
                         "confirmation_scope": "SOURCE_TRANSCRIPTION_ONLY"}
            tentative = {**confirmed, "fact_id": "candidate", "value": "Unreviewed conclusion",
                         "user_confirmed": False, "confirmation_scope": "UNREVIEWED_EXTRACTION"}
            ledger.append_once("HIGH_IMPACT_FACTS_CONFIRMED", "fixture", {
                "case_id": "case-1", "context": {
                    "confirmed_facts": ["alignment: Verified wording", "alignment: Unreviewed conclusion"],
                    "fact_details": [confirmed, tentative],
                    "uncertainties": ["Unreviewed conclusion"], "extraction_id": "source-extraction",
                },
            })
            context = _latest_context(ledger, "case-1")
            self.assertEqual(context["fact_details"], [confirmed])
            self.assertEqual(context["confirmed_facts"], ["alignment: Verified wording"])
            self.assertNotIn("Unreviewed conclusion", context["uncertainties"])
            self.assertEqual(len(ledger.records()[0]["payload"]["context"]["fact_details"]), 2)
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
