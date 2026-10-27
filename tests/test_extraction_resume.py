from pathlib import Path
import tempfile
import unittest

from medic.config import ModelRegistry
from medic.extraction import ClinicalExtractionService
from medic.intake import FileIntakeService
from medic.ledger import AppendOnlyJsonlLedger, ContentAddressedStore
from medic.providers import ApprovedGateway, RecordedGateway


def responses(provider, count):
    return [{"provider_response_id": f"{provider}-{index}", "facts": [],
             **({"uncertainties": []} if provider == "anthropic" else
                {"provider": provider, "model": provider})} for index in range(count)]


class ExtractionResumeTests(unittest.TestCase):
    def test_successful_call_survives_failure_and_new_file_reuses_previous_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = []
            intake_service = FileIntakeService(ContentAddressedStore(root), AppendOnlyJsonlLedger(root / "ledger.jsonl"))
            for index in range(2):
                path = root / f"record-{index}.txt"
                path.write_text(f"Clinical report {index}")
                paths.append(path)
            raw = {name: RecordedGateway(name, name, responses(name, 1 if name == "openai" else 0))
                   for name in ("openai", "google", "anthropic")}
            service = ClinicalExtractionService(root, {name: ApprovedGateway(g) for name, g in raw.items()})
            estimate = ModelRegistry.default().estimate_extraction_cost(100).approve()
            intake = intake_service.ingest_files("case-1", paths[:1])
            with self.assertRaisesRegex(RuntimeError, "no recorded response"):
                service.extract(intake, estimate)
            raw["google"] = RecordedGateway("google", "google", responses("google", 1))
            raw["anthropic"] = RecordedGateway("anthropic", "anthropic", responses("anthropic", 1))
            service = ClinicalExtractionService(root, {name: ApprovedGateway(g) for name, g in raw.items()})
            service.extract(intake, estimate)
            self.assertEqual(raw["openai"].call_count, 1)
            added = {name: RecordedGateway(name, name, responses(name, 1)) for name in raw}
            service = ClinicalExtractionService(root, {name: ApprovedGateway(g) for name, g in added.items()})
            result = service.extract(intake_service.ingest_files("case-1", paths), estimate)
            self.assertEqual([g.call_count for g in added.values()], [1, 1, 1])
            self.assertEqual(len(result.independent_batches), 4)
            self.assertEqual(len(result.calls), 6)
            context = service.confirm(result, {})
            self.assertEqual(len(context["extraction_calls"]), 6)
            self.assertEqual(service.extract(intake_service.ingest_files("case-1", paths), estimate), result)
            self.assertEqual([g.call_count for g in added.values()], [1, 1, 1])

    def test_partition_bounds_images_and_text_without_losing_source_content(self):
        from medic.extraction import extraction_batches
        import pymupdf
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "long.pdf"
            with pymupdf.open() as doc:
                for _ in range(9):
                    doc.new_page()
                doc.save(pdf)
            source = root / "long.txt"
            source.write_text("x" * 50001)
            intake = FileIntakeService(ContentAddressedStore(root), AppendOnlyJsonlLedger(root / "ledger.jsonl")).ingest_files("case-1", [pdf, source])
            batches = extraction_batches(intake)
            self.assertTrue(all(sum(bool(b.rendered_image_path) for b in batch.blocks) <= 4 for batch in batches))
            self.assertTrue(all(sum(len(b.text) for b in batch.blocks) <= 24000 for batch in batches))
            self.assertEqual(sum(len(b.text) for batch in batches for b in batch.blocks), 50001)
            self.assertEqual(sum(bool(b.rendered_image_path) for batch in batches for b in batch.blocks), 9)
