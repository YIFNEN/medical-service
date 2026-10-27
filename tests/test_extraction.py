from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path

from medic.config import ModelRegistry
from medic.intake import FileIntakeService
from medic.ledger import AppendOnlyJsonlLedger, ContentAddressedStore
from medic.providers import ApprovedGateway, RecordedGateway


class ExtractionTests(unittest.TestCase):
    def test_longitudinal_facts_have_distinct_ids_and_reuse_source_confirmation(self):
        from medic.extraction import (
            ClinicalExtractionService, ExtractionResult, FactReconciliation, ReconciledFact,
        )
        with tempfile.TemporaryDirectory() as temp:
            service = ClinicalExtractionService(Path(temp), {
                name: ApprovedGateway(RecordedGateway(name, name, []))
                for name in ("openai", "google", "anthropic")
            })
            facts = tuple(ReconciledFact.model_validate_json(json.dumps({
                "fact_key": "union", "proposed_value": value,
                "source_locators": [source], "supporting_providers": ["openai", "google"],
                "conflict": True, "requires_user_confirmation": True,
                "verification_note": "source wording only",
                "scope": {"event_date": day, "anatomical_site": site,
                          "modality": modality, "assertion": assertion,
                          "source_kind": "RADIOLOGY_REPORT"},
            })) for value, source, day, site, modality, assertion in (
                ("union", "ct|pdf_page|page:1", "2026-08-01", "medial malleolus", "CT", "AFFIRMED"),
                ("R/O nonunion", "spect|pdf_page|page:1", "2026-09-01", "posterior malleolus", "SPECT", "SUSPECTED"),
            ))
            result = ExtractionResult(extraction_id="e1", intake_id="i1", case_id="case-1",
                independent_batches=(), reconciliation=FactReconciliation(
                    provider_response_id="r1", facts=facts, uncertainties=()))
            self.assertNotEqual(facts[0].fact_id, facts[1].fact_id)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                service.confirm(result, {"union": "union"})
            context = service.confirm(result, {fact.fact_id: fact.proposed_value for fact in facts})
            self.assertEqual(len(context["fact_details"]), 2)
            self.assertEqual(context["fact_details"][1]["scope"]["assertion"], "SUSPECTED")
            self.assertEqual(context["fact_details"][0]["confirmation_scope"], "SOURCE_TRANSCRIPTION_ONLY")
            self.assertEqual(service.pending_facts(result), ())
            service.confirm(result, {})

    def test_two_independent_extractors_and_anthropic_reconciliation(self):
        from medic.extraction import ClinicalExtractionService

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "record.txt"
            source.write_text("2024-01-02 오른쪽 발목 삼과골절 ORIF", encoding="utf-8")
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            intake = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files(
                "case-1", [source]
            )
            from medic.extraction import source_locator_key
            locator = source_locator_key(intake.blocks[0])
            openai = RecordedGateway("openai", "o", [{
                "provider": "hallucinated", "model": "hallucinated", "provider_response_id": "hallucinated", "_response_id": "actual-openai-1",
                "facts": [{"fact_key": "laterality", "value": "RIGHT", "source_locator": locator,
                           "high_impact": True, "confidence": 0.98}],
            }])
            google = RecordedGateway("google", "g", [{
                "provider": "google", "model": "g", "provider_response_id": "g-1",
                "facts": [{"fact_key": "laterality", "value": "RIGHT", "source_locator": locator,
                           "high_impact": True, "confidence": 0.95}],
            }])
            anthropic = RecordedGateway("anthropic", "a", [{
                "provider_response_id": "a-1",
                "facts": [{"fact_key": "laterality", "proposed_value": "RIGHT",
                           "source_locators": [locator], "supporting_providers": ["openai", "google"],
                           "conflict": False, "requires_user_confirmation": False,
                           "verification_note": "원문과 일치"}],
                "uncertainties": [],
            }])
            service = ClinicalExtractionService(
                root,
                {"openai": ApprovedGateway(openai), "google": ApprovedGateway(google),
                 "anthropic": ApprovedGateway(anthropic)},
            )
            approved = ModelRegistry.default().estimate_tribunal_cost(100).approve()
            result = service.extract(intake, approved)
            repeated = service.extract(intake, approved)

            self.assertEqual(repeated, result)
            self.assertEqual((openai.call_count, google.call_count, anthropic.call_count), (1, 1, 1))
            self.assertEqual(len(result.independent_batches), 2)
            self.assertEqual(result.independent_batches[0].provider, "openai")
            self.assertEqual(result.independent_batches[0].model, "o")
            self.assertEqual(result.independent_batches[0].provider_response_id, "actual-openai-1")
            self.assertEqual(result.reconciliation.facts[0].proposed_value, "RIGHT")
            self.assertTrue(result.reconciliation.facts[0].requires_user_confirmation)
            self.assertNotIn("google", openai.prompts[0])
            self.assertNotIn("openai", google.prompts[0])

    def test_reference_screenshot_role_and_no_dicom_measurement_reach_models(self):
        from PIL import Image
        from medic.extraction import ClinicalExtractionService
        from medic.intake import ImageRole

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "xray.jpg"
            Image.new("RGB", (12, 12), "white").save(image)
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            intake = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files(
                "case-1", [image], image_roles={image: ImageRole.REFERENCE_IMAGE_SCREENSHOT}
            )
            openai = RecordedGateway("openai", "o", [{"provider": "openai", "model": "o", "provider_response_id": "o", "facts": []}])
            google = RecordedGateway("google", "g", [{"provider": "google", "model": "g", "provider_response_id": "g", "facts": []}])
            anthropic = RecordedGateway("anthropic", "a", [{"provider_response_id": "a", "facts": [], "uncertainties": []}])
            ClinicalExtractionService(root, {
                "openai": ApprovedGateway(openai), "google": ApprovedGateway(google),
                "anthropic": ApprovedGateway(anthropic),
            }).extract(intake, ModelRegistry.default().estimate_extraction_cost(100).approve())

            self.assertIn("REFERENCE_IMAGE_SCREENSHOT", openai.prompts[0])
            self.assertIn("DICOM measurement is forbidden", openai.prompts[0])
            self.assertIn("REFERENCE_IMAGE_SCREENSHOT", anthropic.prompts[0])
    def test_fact_key_is_server_owned_enum(self):
        import json
        from pydantic import ValidationError
        from medic.extraction import ClinicalFactBatch

        payload = {
            "provider": "openai", "model": "m", "provider_response_id": "r",
            "facts": [{"fact_key": "surgery-date", "value": "2024-01-01",
                       "source_locator": "x", "high_impact": False, "confidence": 0.9}],
        }
        with self.assertRaises(ValidationError):
            ClinicalFactBatch.model_validate_json(json.dumps(payload))
    def test_reconciliation_rejects_unknown_source_locator(self):
        from medic.extraction import ClinicalExtractionService

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "record.txt"
            source.write_text("right ankle", encoding="utf-8")
            ledger = AppendOnlyJsonlLedger(root / "ledger.jsonl")
            intake = FileIntakeService(ContentAddressedStore(root), ledger).ingest_files("case-1", [source])
            batch = {"provider": "openai", "model": "o", "provider_response_id": "o-1", "facts": []}
            gateways = {
                "openai": ApprovedGateway(RecordedGateway("openai", "o", [batch])),
                "google": ApprovedGateway(RecordedGateway("google", "g", [{**batch, "provider": "google", "model": "g"}])),
                "anthropic": ApprovedGateway(RecordedGateway("anthropic", "a", [{
                    "provider_response_id": "a-1", "uncertainties": [], "facts": [{
                        "fact_key": "laterality", "proposed_value": "RIGHT",
                        "source_locators": ["invented"], "supporting_providers": ["openai"],
                        "conflict": False, "requires_user_confirmation": True,
                        "verification_note": "bad",
                    }],
                }])),
            }
            with self.assertRaisesRegex(ValueError, "source locator"):
                ClinicalExtractionService(root, gateways).extract(
                    intake, ModelRegistry.default().estimate_tribunal_cost(100).approve()
                )


if __name__ == "__main__":
    unittest.main()
