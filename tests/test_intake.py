from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from medic.ledger import AppendOnlyJsonlLedger, ContentAddressedStore


class FileIntakeTests(unittest.TestCase):
    def test_encrypted_pdf_authentication_preserves_original_and_hides_password(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "encrypted.pdf"
            with fitz.open() as document:
                document.new_page().insert_text((72, 72), "Encrypted clinical record")
                document.save(pdf, encryption=fitz.PDF_ENCRYPT_AES_256,
                              owner_pw="fixture-owner", user_pw="fixture-secret")
            original = pdf.read_bytes()
            service = self.build_service(root / "store")
            with self.assertRaisesRegex(ValueError, "password required"):
                service.ingest_files("case-1", [pdf])
            with self.assertRaisesRegex(ValueError, "authentication failed"):
                service.ingest_files("case-1", [pdf], password_provider=lambda path: "wrong")
            result = service.ingest_files(
                "case-1", [pdf], password_provider=lambda path: "fixture-secret"
            )
            self.assertIn("Encrypted clinical record", result.blocks[0].text)
            self.assertEqual(pdf.read_bytes(), original)
            self.assertEqual(service.store.read_raw(result.files[0].artifact).read_bytes(), original)
            self.assertNotIn("fixture-secret", (root / "store" / "ledger.jsonl").read_text())
            self.assertEqual(service.ingest_files(
                "case-1", [pdf], password_provider=lambda path: "fixture-secret"
            ), result)

    def test_mixed_scan_keeps_body_image_even_with_text_footer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "mixed.pdf"
            with fitz.open() as scan:
                scan.new_page().insert_text((72, 72), "SCANNED BODY")
                image = scan[0].get_pixmap().tobytes("png")
            with fitz.open() as document:
                page = document.new_page()
                page.insert_image(fitz.Rect(20, 20, 550, 700), stream=image)
                page.insert_text((72, 780), "Verification footer only")
                document.save(pdf)
            result = self.build_service(root / "store").ingest_files("case-1", [pdf])
            self.assertTrue(any("Verification footer" in block.text for block in result.blocks))
            rendered = [block for block in result.blocks if block.rendered_image_path]
            self.assertEqual(len(rendered), 1)
            self.assertEqual(rendered[0].page_number, 1)
            self.assertTrue(Path(rendered[0].rendered_image_path).is_file())

    def build_service(self, root: Path):
        from medic.intake import FileIntakeService

        return FileIntakeService(
            ContentAddressedStore(root), AppendOnlyJsonlLedger(root / "ledger.jsonl")
        )

    def test_ingests_text_with_source_locator_and_immutable_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "operation.txt"
            source.write_text("Right ankle ORIF on 2024-01-02", encoding="utf-8")

            result = self.build_service(root / "store").ingest_files("case-1", [source])

            self.assertEqual(result.files[0].kind, "TEXT_DOCUMENT")
            self.assertIn("Right ankle ORIF", result.blocks[0].text)
            self.assertEqual(
                result.blocks[0].source_locator.artifact_id,
                result.files[0].artifact.artifact_id,
            )
            self.assertTrue(result.files[0].artifact.content_hash.startswith("sha256:"))

    def test_extracts_text_pdf_and_renders_scanned_page(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "records.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Left ankle follow-up")
            document.new_page()
            document.save(pdf)
            document.close()

            result = self.build_service(root / "store").ingest_files("case-1", [pdf])

            self.assertEqual(len(result.blocks), 2)
            self.assertIn("Left ankle follow-up", result.blocks[0].text)
            self.assertIn("bbox:", result.blocks[0].source_locator.locator)
            self.assertIsNotNone(result.blocks[1].rendered_image_path)
            self.assertTrue(Path(result.blocks[1].rendered_image_path).exists())

    def test_image_requires_explicit_reference_screenshot_role(self):
        from PIL import Image

        from medic.intake import ImageRole

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "xray.jpg"
            Image.new("RGB", (20, 20), "white").save(image)

            result = self.build_service(root / "store").ingest_files(
                "case-1", [image], image_roles={image: ImageRole.REFERENCE_IMAGE_SCREENSHOT}
            )

            self.assertEqual(result.files[0].kind, "IMAGE")
            self.assertEqual(result.files[0].image_role, "REFERENCE_IMAGE_SCREENSHOT")
            self.assertFalse(result.files[0].dicom_measurement_eligible)

    def test_rejects_unsupported_file_type(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "archive.zip"
            source.write_bytes(b"not supported")
            with self.assertRaisesRegex(ValueError, "unsupported input"):
                self.build_service(root / "store").ingest_files("case-1", [source])


if __name__ == "__main__":
    unittest.main()
