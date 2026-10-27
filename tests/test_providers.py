from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PIL import Image

from medic.config import ModelRegistry
from medic.research_contracts import IndependentOpinion


class ProviderGatewayTests(unittest.TestCase):
    def test_image_mime_is_detected_from_bytes_when_store_path_has_no_extension(self):
        from medic.providers import _image_data

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "content-addressed-blob"
            Image.new("RGB", (8, 8), "white").save(source, format="JPEG")
            mime, encoded = _image_data(source)

        self.assertEqual(mime, "image/jpeg")
        self.assertTrue(encoded)
    def test_gateway_is_never_called_without_cost_approval(self):
        from medic.providers import ApprovedGateway, RecordedGateway

        inner = RecordedGateway("openai", "model", responses=[])
        gateway = ApprovedGateway(inner)

        with self.assertRaisesRegex(PermissionError, "cost approval"):
            gateway.generate(
                stage="analyst",
                prompt="analyze",
                output_model=IndependentOpinion,
                estimate=ModelRegistry.default().estimate_tribunal_cost(100),
            )
        self.assertEqual(inner.call_count, 0)

    def test_recorded_gateway_returns_strict_model_and_metadata(self):
        from medic.providers import ApprovedGateway, RecordedGateway

        inner = RecordedGateway(
            "openai",
            "model",
            responses=[
                {
                    "opinion_id": "op-1",
                    "provider": "openai",
                    "model": "model",
                    "provider_response_id": "response-1",
                    "hypotheses": ["PTOA"],
                    "evidence_ids": ["ev-1"],
                    "missing_information": [],
                    "action_ranking": ["weight-bearing radiograph"],
                    "reasoning_summary": "summary",
                }
            ],
        )
        result = ApprovedGateway(inner).generate(
            stage="analyst",
            prompt="analyze",
            output_model=IndependentOpinion,
            estimate=ModelRegistry.default().estimate_tribunal_cost(100).approve(),
        )

        self.assertEqual(result.value.opinion_id, "op-1")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(inner.call_count, 1)

    def test_live_gateway_requires_its_provider_key(self):
        from medic.providers import OpenAIGateway

        gateway = OpenAIGateway(model="model", api_key_env="MISSING_MEDIC_TEST_KEY")
        with self.assertRaisesRegex(RuntimeError, "MISSING_MEDIC_TEST_KEY"):
            gateway.ensure_available()


if __name__ == "__main__":
    unittest.main()
