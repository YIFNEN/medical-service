import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from medic.config import ModelRegistry
from medic.extraction import ClinicalFactBatch
from medic.providers import OpenAIGateway, AnthropicGateway, GeminiGateway


class ProviderEnvelopeTests(unittest.TestCase):
    def test_openai_strict_schema_reasoning_and_incomplete_response(self):
        with patch.dict("os.environ", {"TEST_KEY": "fixture"}), patch("openai.OpenAI") as client:
            create = client.return_value.responses.create
            create.return_value = NS(id="api-response", model="snapshot", status="completed",
                                     output=[], output_text='{}', usage=NS(input_tokens=7, output_tokens=13))
            gateway = OpenAIGateway("gpt-5.6-sol", "TEST_KEY", reasoning="max")
            result = gateway.generate_raw(stage="test", prompt="test",
                json_schema=ClinicalFactBatch.model_json_schema(), max_output_tokens=100)
            self.assertEqual(create.call_args.kwargs["reasoning"], {"effort": "max"})
            schema = create.call_args.kwargs["text"]["format"]["schema"]
            scope = schema["$defs"]["FactScope"]
            self.assertEqual(set(scope["required"]), set(scope["properties"]))
            self.assertEqual(result.model, "snapshot")
            create.return_value.status = "incomplete"
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                gateway.generate_raw(stage="test", prompt="test", json_schema={}, max_output_tokens=100)

    def test_anthropic_reasoning_and_truncation(self):
        with patch.dict("os.environ", {"TEST_KEY": "fixture"}), patch("anthropic.Anthropic") as client:
            create = client.return_value.messages.create
            create.return_value = NS(id="api-response", model="snapshot", stop_reason="end_turn",
                content=[NS(type="text", text='{}')], usage=NS(input_tokens=7, output_tokens=13))
            gateway = AnthropicGateway("claude-opus-4-7", "TEST_KEY", reasoning="adaptive")
            gateway.generate_raw(stage="test", prompt="test", json_schema=ClinicalFactBatch.model_json_schema(), max_output_tokens=100)
            self.assertEqual(create.call_args.kwargs["thinking"], {"type": "adaptive"})
            create.return_value.stop_reason = "max_tokens"
            with self.assertRaisesRegex(RuntimeError, "max_tokens"):
                gateway.generate_raw(stage="test", prompt="test", json_schema=ClinicalFactBatch.model_json_schema(), max_output_tokens=100)

    def test_gemini_bills_thinking_and_rejects_blocked_output(self):
        with patch.dict("os.environ", {"TEST_KEY": "fixture"}), patch("google.genai.Client") as client:
            create = client.return_value.models.generate_content
            create.return_value = NS(response_id="api-response", model_version="snapshot", text='{}',
                candidates=[NS(finish_reason="STOP")], usage_metadata=NS(prompt_token_count=7,
                candidates_token_count=13, thoughts_token_count=19))
            gateway = GeminiGateway("gemini-3.7-flash", "TEST_KEY", reasoning="high")
            result = gateway.generate_raw(stage="test", prompt="test", json_schema={}, max_output_tokens=100)
            self.assertEqual(result.output_tokens, 32)
            create.return_value.candidates = [NS(finish_reason="SAFETY")]
            with self.assertRaisesRegex(RuntimeError, "SAFETY"):
                gateway.generate_raw(stage="test", prompt="test", json_schema={}, max_output_tokens=100)

    def test_model_override_cannot_reuse_default_model_price(self):
        with patch.dict("os.environ", {"MEDIC_OPENAI_MODEL": "different-model"}):
            with self.assertRaisesRegex(ValueError, "pricing"):
                ModelRegistry.default()
